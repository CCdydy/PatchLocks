"""Import raw Artemis DFA frames into the NeuralFur-style 2D folder we use.

The PatchLocks geodesic optimizer expects:

  images_2/<view>.png
  masks_2/body/<view>.png
  masks_2/hair/<view>.png
  masks_2/bald/<view>.png
  orientations_2/angles/<view>.png
  orientations_2/vars/<view>.npy
  3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl

Raw Artemis DFA data is organized as:

  <animal>/img/<motion>/<frame>/img_%04d.jpg
  <animal>/img/<motion>/<frame>/img_%04d_alpha.png
  <animal>/CamPose.inf
  <animal>/Intrinsic.inf

This script does not require the Artemis Python environment. It implements the
camera parsing convention used by Artemis:
  campose columns are stored as z, x, y, t and represent camera-to-world.
"""
import argparse
import json
import os
import pickle
import shutil

import cv2
import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_intrinsics(path):
    lines = open(path).read().splitlines()
    mats = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        parts = line.split()
        # Intrinsic.inf stores an index line before each 3x3 matrix.
        if len(parts) == 1:
            i += 1
            if i + 2 >= len(lines):
                break
            rows = []
            for _ in range(3):
                rows.append([float(x) for x in lines[i].split()])
                i += 1
            mats.append(np.asarray(rows, np.float32))
            continue
        if len(parts) == 3 and i + 2 < len(lines):
            rows = []
            for _ in range(3):
                rows.append([float(x) for x in lines[i].split()])
                i += 1
            mats.append(np.asarray(rows, np.float32))
            continue
        i += 1
    if not mats:
        raise ValueError(f"No intrinsics parsed from {path}")
    return np.stack(mats).astype(np.float32)


def campose_to_c2w(camposes):
    camposes = np.asarray(camposes, np.float32)
    if camposes.ndim == 1:
        camposes = camposes[None]
    if camposes.shape[1] != 12:
        raise ValueError(f"Expected CamPose rows with 12 values, got {camposes.shape}")
    out = np.zeros((camposes.shape[0], 4, 4), np.float32)
    out[:, 0:3, 2] = camposes[:, 0:3]
    out[:, 0:3, 0] = camposes[:, 3:6]
    out[:, 0:3, 1] = camposes[:, 6:9]
    out[:, 0:3, 3] = camposes[:, 9:12]
    out[:, 3, 3] = 1.0
    return out


def c2w_k_to_row_clip(c2w, k, width, height):
    """Build the row-vector clip matrix used by patch_graph_opt_geodesic.py."""
    w2c = np.linalg.inv(c2w).astype(np.float32)
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    p = np.zeros((4, 4), np.float32)
    p[0, 0] = 2.0 * fx / float(width)
    p[2, 0] = 2.0 * cx / float(width) - 1.0
    p[1, 1] = -2.0 * fy / float(height)
    p[2, 1] = 1.0 - 2.0 * cy / float(height)
    p[2, 2] = 1.0
    p[2, 3] = 1.0
    return (w2c.T @ p).astype(np.float32)


def image_paths(frame_dir, view):
    stems = [
        f"img_{view:04d}",
        f"{view:04d}",
        f"image_{view:04d}",
    ]
    img_exts = (".jpg", ".jpeg", ".png")
    alpha_exts = ("_alpha.png", "_mask.png", ".alpha.png")
    img = None
    alpha = None
    for stem in stems:
        for ext in img_exts:
            p = os.path.join(frame_dir, stem + ext)
            if os.path.exists(p):
                img = p
                break
        for ext in alpha_exts:
            p = os.path.join(frame_dir, stem + ext)
            if os.path.exists(p):
                alpha = p
                break
        if img is not None:
            break
    return img, alpha


def orientation_from_image(rgb, mask, sigma):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(sigma * 0.45, 0.1))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigmaX=sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigmaX=sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigmaX=sigma)
    grad_ang = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy + 1e-12)
    hair_ang = np.mod(grad_ang + 0.5 * np.pi, np.pi)
    tr = jxx + jyy
    det_term = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy ** 2, 0.0))
    conf = np.clip(det_term / np.maximum(tr, 1e-6), 0.0, 1.0)
    conf = conf * (mask.astype(np.float32) / 255.0)
    ang_u8 = np.clip(hair_ang / np.pi * 179.0, 0, 179).astype(np.uint8)
    return ang_u8, conf.astype(np.float32)


def parse_views(spec, available):
    if spec:
        out = []
        for x in spec.split(","):
            x = x.strip()
            if not x:
                continue
            out.append(int(x))
        return out
    return list(range(available))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw Artemis animal directory, e.g. .../beagle_dog")
    ap.add_argument("--motion", required=True, help="motion folder under img/, e.g. s1, walk, walk_final")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", default="", help="comma-separated camera ids; empty = all cameras in Intrinsic.inf")
    ap.add_argument("--orientation_sigma", type=float, default=4.0)
    ap.add_argument("--alpha_threshold", type=int, default=8)
    ap.add_argument("--copy", action="store_true", help="copy source RGB files instead of re-encoding PNG")
    args = ap.parse_args()

    frame_dir = os.path.join(args.raw, "img", args.motion, str(args.frame))
    cam_path = os.path.join(args.raw, "CamPose.inf")
    k_path = os.path.join(args.raw, "Intrinsic.inf")
    if not os.path.isdir(frame_dir):
        raise FileNotFoundError(f"Missing Artemis frame directory: {frame_dir}")
    if not os.path.exists(cam_path) or not os.path.exists(k_path):
        raise FileNotFoundError(f"Missing CamPose.inf or Intrinsic.inf under {args.raw}")

    c2w = campose_to_c2w(np.loadtxt(cam_path))
    ks = read_intrinsics(k_path)
    ncam = min(len(c2w), len(ks))
    views = parse_views(args.views, ncam)

    for sub in [
        "images_2",
        "masks_2/body",
        "masks_2/hair",
        "masks_2/bald",
        "orientations_2/angles",
        "orientations_2/vars",
        "3d_gaussian_splatting/GS_/cameras",
    ]:
        ensure_dir(os.path.join(args.out, sub))

    mats = {}
    imported = []
    for view in views:
        if view < 0 or view >= ncam:
            raise ValueError(f"View {view} outside available camera range 0..{ncam - 1}")
        img_path, alpha_path = image_paths(frame_dir, view)
        if img_path is None:
            raise FileNotFoundError(f"No RGB image found for view {view:04d} in {frame_dir}")
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Could not read image: {img_path}")
        h, w = img_bgr.shape[:2]
        if alpha_path and os.path.exists(alpha_path):
            alpha = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
            if alpha is None:
                raise ValueError(f"Could not read alpha: {alpha_path}")
            if alpha.shape[:2] != (h, w):
                alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            alpha = np.full((h, w), 255, np.uint8)
        alpha = np.where(alpha >= args.alpha_threshold, 255, 0).astype(np.uint8)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        vid = f"{view:04d}"
        out_img = os.path.join(args.out, "images_2", f"{vid}.png")
        if args.copy and img_path.lower().endswith(".png"):
            shutil.copy2(img_path, out_img)
        else:
            cv2.imwrite(out_img, img_bgr)
        cv2.imwrite(os.path.join(args.out, "masks_2/body", f"{vid}.png"), alpha)
        cv2.imwrite(os.path.join(args.out, "masks_2/hair", f"{vid}.png"), alpha)
        cv2.imwrite(os.path.join(args.out, "masks_2/bald", f"{vid}.png"), np.zeros_like(alpha))
        ang, var = orientation_from_image(rgb, alpha, args.orientation_sigma)
        cv2.imwrite(os.path.join(args.out, "orientations_2/angles", f"{vid}.png"), ang)
        np.save(os.path.join(args.out, "orientations_2/vars", f"{vid}.npy"), var)
        mats[vid] = c2w_k_to_row_clip(c2w[view], ks[view], w, h)
        imported.append({"view": vid, "rgb": img_path, "alpha": alpha_path or "", "width": w, "height": h})

    with open(os.path.join(args.out, "3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl"), "wb") as f:
        pickle.dump(mats, f)
    manifest = {
        "source_raw": os.path.abspath(args.raw),
        "motion": args.motion,
        "frame": args.frame,
        "views": imported,
        "camera_convention": "Artemis CamPose camera-to-world z,x,y,t -> PatchLocks row clip",
    }
    with open(os.path.join(args.out, "import_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"imported {len(imported)} Artemis views -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
