#!/usr/bin/env python3
"""Rule-based per-patch region labels for a standing quadruped (for INDIVIDUAL_ANNOTATION_JSON).

The automatic labels are quantiles of the bounding box. When they put the neck ruff and chest
into `face`/`ears` or the whole lower leg into `paws`, the recipe's region scales act on the wrong
coat. This labels every patch centre from a few thresholds read off the furless mesh in the
capture frame (slice the mesh along the body axis: the neck is the narrowest section between the
chest and the head; the belly line is the lowest point of the trunk between the legs):

  height  = coordinate along --up         forward = coordinate along --forward (towards the head)
  paws      height < --paws_below
  legs      height < --legs_below
  face      forward > --head_from                                 (wins over legs)
  ears      face and forward < --ears_before and height > --ears_above
  tail      forward < --tail_behind, or forward < --root_forward and height > --root_height
  tail_base --root_forward <= forward < --tail_base_to and height > --root_height
  body      everything else (trunk, neck, chest, thighs)

  python patchlocks/deploy/label_patch_regions.py --surfpack <in>/surfpack/surface_patch_dataset.npz \\
      --up=-y --forward=+z --paws_below <m> --legs_below <m> --head_from <m> --ears_above <m> \\
      --ears_before <m> --tail_behind <m> --root_forward <m> --root_height <m> --tail_base_to <m> \\
      --out patchlocks/deploy/refine_annotations/<animal>_patch_region.npz

then point refine_annotations/<animal>.json at the npz ({"patch_region_npz": ...}) and set
INDIVIDUAL_ANNOTATION_JSON in <animal>.env. With --mesh <furless.obj> instead of --surfpack the
same rules label the mesh vertices (`vertex_region`, for fit_furless_mesh.py).
"""
import argparse

import numpy as np


def axis(spec, pts):
    s = spec.strip().lower()
    sign = -1.0 if s.startswith("-") else 1.0
    return sign * pts[:, "xyz".index(s.lstrip("+-"))]


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--surfpack", help="label the patch centres of this surface patch pack")
    src.add_argument("--mesh", help="label the vertices of this OBJ mesh instead")
    ap.add_argument("--out", required=True)
    ap.add_argument("--up", default="+y", help="axis pointing up, e.g. -y for a COLMAP frame")
    ap.add_argument("--forward", default="+z", help="axis pointing from the tail to the head")
    ap.add_argument("--paws_below", type=float, required=True)
    ap.add_argument("--legs_below", type=float, required=True)
    ap.add_argument("--head_from", type=float, required=True)
    ap.add_argument("--ears_above", type=float, required=True)
    ap.add_argument("--ears_before", type=float, required=True)
    ap.add_argument("--tail_behind", type=float, required=True)
    ap.add_argument("--root_forward", type=float, required=True, help="tail root along --forward")
    ap.add_argument("--root_height", type=float, required=True, help="tail root height (rump top)")
    ap.add_argument("--tail_base_to", type=float, required=True)
    a = ap.parse_args()
    if a.mesh:
        c = np.array([[float(x) for x in ln.split()[1:4]] for ln in open(a.mesh) if ln.startswith("v ")])
    else:
        c = np.asarray(np.load(a.surfpack, mmap_mode="r", allow_pickle=True)["centers"], np.float64)
    h, f = axis(a.up, c), axis(a.forward, c)
    reg = np.full(len(c), "body", dtype="U24")
    reg[h < a.legs_below] = "legs"
    reg[h < a.paws_below] = "paws"
    head = f > a.head_from
    reg[head] = "face"
    reg[head & (f < a.ears_before) & (h > a.ears_above)] = "ears"
    reg[(f < a.tail_behind) | ((f < a.root_forward) & (h > a.root_height))] = "tail"
    reg[(f >= a.root_forward) & (f < a.tail_base_to) & (h > a.root_height)] = "tail_base"
    np.savez_compressed(a.out, **({"vertex_region": reg} if a.mesh else {"patch_region": reg}))
    names, counts = np.unique(reg, return_counts=True)
    print(dict(zip(names.tolist(), counts.tolist())))


if __name__ == "__main__":
    main()
