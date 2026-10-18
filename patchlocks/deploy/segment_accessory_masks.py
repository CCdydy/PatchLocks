#!/usr/bin/env python3
"""2D masks of an accessory (a collar) in every view, so colour baking can ignore those pixels.

The collar sits on the fur, centimetres outside the furless mesh, so it cannot be located on the
mesh by colour; it is segmented in the photos instead. Per view:
  1. candidate pixels: inside the body mask, in the accessory hue band and saturated enough, and
     whose nearest visible mesh vertex lies in the given slab along an axis (the neck), which
     keeps a pink tongue out;
  2. SAM2 prompted with points on the candidates and their box segments the accessory; the
     result is kept near the candidates;
  3. dark pixels (a black clasp) next to the accessory and in the slab are added, then the mask is
     dilated a little.

Writes <data_dir>/masks_2/<name>/<view>.png (255 = accessory) and a QA contact sheet.

  python patchlocks/deploy/segment_accessory_masks.py --data_dir <in>/nf_data --mesh <in>/furless.obj \\
      --axis <i> --lo <a> --hi <b> [--name accessory]
"""
import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bake_mesh_albedo_robust import front_visible_vertices, load_obj_mesh  # noqa: E402


def hsv_gate(rgb, hue, sat, val):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV_FULL).astype(np.float32)
    h, s, v = hsv[..., 0] * 360.0 / 256.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    h0, h1 = hue
    band = (h >= h0) | (h <= h1) if h0 > h1 else (h >= h0) & (h <= h1)
    return band & (s > sat) & (v > val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--axis", type=int, default=2)
    ap.add_argument("--lo", type=float, required=True, help="slab along --axis where the accessory is worn")
    ap.add_argument("--hi", type=float, required=True)
    ap.add_argument("--hue", default="290,15", help="hue band in degrees (wraps)")
    ap.add_argument("--sat", type=float, default=0.45)
    ap.add_argument("--val", type=float, default=0.30)
    ap.add_argument("--dark_val", type=float, default=0.22, help="value below which pixels next to the accessory join it (0 = off)")
    ap.add_argument("--dark_px", type=int, default=151, help="how far from the accessory dark pixels may be")
    ap.add_argument("--min_px", type=int, default=150, help="fewer candidate pixels: no accessory in the view")
    ap.add_argument("--keep_px", type=int, default=61, help="SAM2 mask is kept within this dilation of the candidates")
    ap.add_argument("--grow_px", type=int, default=4)
    ap.add_argument("--name", default="accessory")
    ap.add_argument("--sam2_model", default="facebook/sam2-hiera-large")
    ap.add_argument("--no_sam", action="store_true", help="candidates only (dilated)")
    a = ap.parse_args()

    import torch
    seg = None
    if not a.no_sam:
        from transformers import Sam2Model, Sam2Processor
        proc = Sam2Processor.from_pretrained(a.sam2_model)
        net = Sam2Model.from_pretrained(a.sam2_model).cuda().eval()
        seg = (proc, net)

    v, f, vn = load_obj_mesh(a.mesh)
    diag = float(np.linalg.norm(v.max(0) - v.min(0)))
    in_slab = (v[:, a.axis] >= a.lo) & (v[:, a.axis] <= a.hi)
    mats = pickle.load(open(os.path.join(a.data_dir, "3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl"), "rb"))
    out_dir = os.path.join(a.data_dir, "masks_2", a.name)
    os.makedirs(out_dir, exist_ok=True)
    hue = [float(x) for x in a.hue.split(",")]
    rng = np.random.default_rng(0)
    stats, sheet = {}, []
    for vid in sorted(mats):
        bgr = cv2.imread(os.path.join(a.data_dir, "images_2", f"{vid}.png"))
        rgb = bgr[..., ::-1].copy()
        H, W = rgb.shape[:2]
        body = cv2.imread(os.path.join(a.data_dir, "masks_2/body", f"{vid}.png"), 0) > 0
        vis, px, py = front_visible_vertices(v, vn, np.asarray(mats[vid], np.float32), W, H, diag, 700)
        ids = np.flatnonzero(vis)
        cand = hsv_gate(rgb, hue, a.sat, a.val) & body
        dark = (cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 2] < 255 * a.dark_val) & body
        tree = cKDTree(np.c_[px[ids], py[ids]]) if len(ids) else None

        def slab_only(m):
            ys, xs = np.nonzero(m)
            out = np.zeros_like(m)
            if len(xs) and tree is not None:
                _, nn = tree.query(np.c_[xs, ys])
                ok = in_slab[ids[nn]]
                out[ys[ok], xs[ok]] = True
            return out
        cand = slab_only(cand)
        cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(cand)
        cand = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 50])
        out = np.zeros((H, W), np.uint8)
        how = "none"
        if cand.sum() >= a.min_px:
            near = cv2.dilate(cand.astype(np.uint8), np.ones((a.keep_px, a.keep_px), np.uint8)) > 0
            final = cand.copy()
            how = "colour"
            if seg is not None:
                proc, net = seg
                yy, xx = np.nonzero(cv2.erode(cand.astype(np.uint8), np.ones((5, 5), np.uint8)) if cand.sum() > 400 else cand)
                pick = rng.choice(len(xx), min(6, len(xx)), replace=False)
                ys, xs = np.nonzero(cand)
                m = 0.25 * max(np.ptp(xs), np.ptp(ys)) + 20
                box = [[[float(xs.min() - m), float(ys.min() - m), float(xs.max() + m), float(ys.max() + m)]]]
                pts = [[[[float(xx[i]), float(yy[i])] for i in pick]]]
                inp = proc(images=rgb, input_points=pts, input_labels=[[[1] * len(pick)]], input_boxes=box,
                           return_tensors="pt").to("cuda")
                with torch.no_grad():
                    o = net(**inp, multimask_output=True)
                masks = proc.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"])[0][0].numpy() > 0
                best, best_s = None, -1e9
                for k in range(len(masks)):
                    area = max(int(masks[k].sum()), 1)
                    recall = (masks[k] & cand).sum() / cand.sum()
                    spill = (masks[k] & ~near).sum() / area
                    s = recall - 1.5 * spill
                    if s > best_s:
                        best, best_s = masks[k], s
                if best is not None and best_s > 0.3:
                    final = (best & near & body) | cand
                    how = f"sam2 {best_s:.2f}"
            if a.dark_val > 0:
                ring = cv2.dilate(final.astype(np.uint8), np.ones((a.dark_px, a.dark_px), np.uint8)) > 0
                d = slab_only(dark & ring)
                n_d, lab_d, st_d, _ = cv2.connectedComponentsWithStats(d.astype(np.uint8))
                # keep dark blobs that touch the accessory (a clasp), not scattered dark fur
                touch = cv2.dilate(final.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                keep = [i for i in range(1, n_d) if st_d[i, cv2.CC_STAT_AREA] >= 30 and (touch & (lab_d == i)).any()]
                if keep:
                    final = final | np.isin(lab_d, keep)
                    how += f" +dark{int(np.isin(lab_d, keep).sum())}"
            out = cv2.dilate(final.astype(np.uint8) * 255, np.ones((2 * a.grow_px + 1,) * 2, np.uint8))
        cv2.imwrite(os.path.join(out_dir, f"{vid}.png"), out)
        stats[vid] = {"pixels": int((out > 0).sum()), "method": how}
        if out.any():
            ys, xs = np.nonzero(out)
            cx, cy, s = int(xs.mean()), int(ys.mean()), int(max(np.ptp(xs), np.ptp(ys)) * 0.75) + 60
            ov = bgr.copy()
            ov[out > 0] = (0.4 * ov[out > 0] + 0.6 * np.array([255, 255, 0])).astype(np.uint8)
            crop = ov[max(cy - s, 0):cy + s, max(cx - s, 0):cx + s]
            crop = cv2.resize(crop, (240, 240))
            cv2.putText(crop, vid, (5, 20), 0, 0.6, (255, 255, 255), 2)
            sheet.append(crop)
        print(vid, stats[vid], flush=True)
    if sheet:
        while len(sheet) % 12:
            sheet.append(np.zeros_like(sheet[0]))
        rows = [np.concatenate(sheet[i:i + 12], 1) for i in range(0, len(sheet), 12)]
        os.makedirs(os.path.join(a.data_dir, "qa"), exist_ok=True)
        cv2.imwrite(os.path.join(a.data_dir, "qa", f"{a.name}_masks.jpg"), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])
    json.dump(stats, open(os.path.join(out_dir, "stats.json"), "w"), indent=1)
    print("views with accessory:", sum(1 for s in stats.values() if s["pixels"] > 0), "of", len(stats))


if __name__ == "__main__":
    main()
