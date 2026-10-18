#!/usr/bin/env python3
"""Pick the Step-2 fit views and write the tangent frame for a new animal.

1. views.txt: N views spread around the animal (farthest-point sampling of camera directions
   from the mesh centre, starting from the view with the largest silhouette). Used for the
   surface-pack export and the Step-2 fit.
2. tangent_field_xaxis.npz: a constant world +x tangent per face. Passed to
   neuralfur_export_surface_patch_dataset.py --field_pack, it makes each patch's tangent frame
   the projected world x axis, which is exactly the frame Step 2 (patch_graph_opt_geo.py) fits
   orientation offsets in. Grow then applies the fitted offsets in the same frame, so the final
   strand orientation comes from the image fit alone (no external flow field).

  python patchlocks/deploy/prepare_fit_views_and_frame.py --data_dir <nf_data> --mesh furless.obj \\
      --out_dir patchlocks/deploy/inputs/<animal> [--n 12]
"""
import argparse
import os
import pickle

import cv2
import numpy as np
import trimesh


def camera_centre(M):
    P = np.asarray(M, np.float64).T
    A = np.array([P[0], P[1], P[3]])
    return -np.linalg.solve(A[:, :3], A[:, 3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()

    mats = pickle.load(open(os.path.join(a.data_dir, "3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl"), "rb"))
    vids = sorted(mats)
    mesh = trimesh.load(a.mesh, process=False)
    c = np.asarray(mesh.vertices).mean(0)
    dirs = np.array([(camera_centre(mats[v]) - c) / np.linalg.norm(camera_centre(mats[v]) - c) for v in vids])
    area = np.array([(cv2.imread(os.path.join(a.data_dir, "masks_2/body", f"{v}.png"), 0) > 0).sum() for v in vids])
    sel = [int(np.argmax(area))]
    while len(sel) < min(a.n, len(vids)):
        d = np.min(np.arccos(np.clip(dirs @ dirs[sel].T, -1, 1)), 1)
        d[sel] = -1
        sel.append(int(np.argmax(d)))
    views = ",".join(vids[i] for i in sorted(sel))
    os.makedirs(a.out_dir, exist_ok=True)
    open(os.path.join(a.out_dir, "views.txt"), "w").write(views + "\n")
    np.savez(os.path.join(a.out_dir, "tangent_field_xaxis.npz"),
             face_flow=np.tile(np.array([1, 0, 0], np.float32), (len(mesh.faces), 1)),
             note=np.asarray("constant world +x per face: surface-pack tangent frame = Step-2 fit frame"))
    print("views:", views)
    print("tangent field:", len(mesh.faces), "faces ->", os.path.join(a.out_dir, "tangent_field_xaxis.npz"))


if __name__ == "__main__":
    main()
