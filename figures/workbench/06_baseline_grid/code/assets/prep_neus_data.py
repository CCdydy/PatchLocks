#!/usr/bin/env python
"""Convert our multi-view captures -> vanilla-NeuS data format (baseline repro).

Our cameras: 30000_matrices.pkl holds M (4x4) used ROW-vector style: clip = [X,1] @ M,
px = (ndc_x/2+.5)*W, py = (1-(ndc_y/2+.5))*H  (patch_graph_opt_geo.project_row).
NeuS wants world_mat = COLUMN-vector world->PIXEL projection P (3x4, cv2-decomposable)
and scale_mat mapping a unit sphere to the scene (we use the furless mesh bounds).

Derivation (A = M^T, column world->clip):  px*w = (W/2)(A0+A3)X ; py*w = (H/2)(A3-A1)X ; w = A3 X.
Self-check: reproject mesh vertices both ways, assert max |delta| < 0.1 px.

Usage: python prep_neus_data.py <animal>   (beagle|whiteTiger|panda|cat)
"""
import sys, os, pickle, glob, shutil
import numpy as np
import cv2
import trimesh

ROOT = "/home/rog/Desktop/diffusionF"
CFG = {
    "beagle":     (f"{ROOT}/patchlocks/deploy/neuralfur_beagle_official/realviews/nf_data_f210",
                   f"{ROOT}/patchlocks/datasets/neuralfur/official_results/extracted/neuralfur/beagle_dog/furless_reshaped.obj"),
    "whiteTiger": (f"{ROOT}/patchlocks/deploy/neuralfur_whiteTiger_official/realviews/nf_data_roaringwalk_f5",
                   f"{ROOT}/patchlocks/datasets/neuralfur/official_results/extracted/neuralfur/whiteTiger/furless_reshaped.obj"),
    "panda":      (f"{ROOT}/patchlocks/datasets/neuralfur/extracted/data/Artemis/panda_processed_GH2/walk",
                   f"{ROOT}/patchlocks/datasets/neuralfur/official_results/extracted/neuralfur/panda/furless.obj"),
    "cat":        (f"{ROOT}/patchlocks/deploy/neuralfur_cat_official/realviews/nf_data_walk_final_f0",
                   f"{ROOT}/patchlocks/datasets/neuralfur/official_results/extracted/neuralfur/cat/furless_reshaped.obj"),
}

def orig_project(pts, M, W, H):
    homo = np.concatenate([pts, np.ones((len(pts), 1))], 1).astype(np.float64)
    clip = homo @ M
    ndc = clip[:, :3] / clip[:, 3:4]
    px = (ndc[:, 0] * .5 + .5) * W
    py = (1. - (ndc[:, 1] * .5 + .5)) * H
    return px, py

def to_pixel_P(M, W, H):
    A = np.asarray(M, np.float64).T           # column-vector world->clip
    P = np.zeros((3, 4))
    P[0] = (W / 2.) * (A[0] + A[3])
    P[1] = (H / 2.) * (A[3] - A[1])
    P[2] = A[3]
    return P

def main(animal):
    data_dir, mesh_path = CFG[animal]
    out = f"{ROOT}/_baselines/neus_data/{animal}"
    os.makedirs(f"{out}/image", exist_ok=True)
    os.makedirs(f"{out}/mask", exist_ok=True)
    mats = pickle.load(open(f"{data_dir}/3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl", "rb"))
    views = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{data_dir}/images_2/*.png"))
    im0 = cv2.imread(f"{data_dir}/images_2/{views[0]}.png")
    H, W = im0.shape[:2]
    mesh = trimesh.load(mesh_path, process=False)
    V = np.asarray(mesh.vertices)
    c = (V.min(0) + V.max(0)) / 2.
    r = float(np.linalg.norm(V - c, axis=1).max()) * 1.1
    scale_mat = np.eye(4); scale_mat[:3, :3] *= r; scale_mat[:3, 3] = c

    cam = {}
    sub = V[np.random.RandomState(0).choice(len(V), 500, replace=False)]
    worst = 0.
    for i, v in enumerate(views):
        M = np.asarray(mats[v], np.float64)
        if hasattr(mats[v], "detach"):
            M = mats[v].detach().cpu().numpy().astype(np.float64)
        P = to_pixel_P(M, W, H)
        # self-check on mesh points
        px0, py0 = orig_project(sub, M, W, H)
        homo = np.concatenate([sub, np.ones((len(sub), 1))], 1)
        pr = (P @ homo.T).T
        px1, py1 = pr[:, 0] / pr[:, 2], pr[:, 1] / pr[:, 2]
        worst = max(worst, float(np.abs(px0 - px1).max()), float(np.abs(py0 - py1).max()))
        w44 = np.eye(4); w44[:3, :4] = P
        cam[f"world_mat_{i}"] = w44
        cam[f"scale_mat_{i}"] = scale_mat
        shutil.copy(f"{data_dir}/images_2/{v}.png", f"{out}/image/{i:03d}.png")
        m = cv2.imread(f"{data_dir}/masks_2/body/{v}.png", 0)
        cv2.imwrite(f"{out}/mask/{i:03d}.png", cv2.merge([m, m, m]))
    assert worst < 0.1, f"projection mismatch {worst}px"
    np.savez(f"{out}/cameras_sphere.npz", **cam)
    print(f"{animal}: {len(views)} views {W}x{H}, sphere r={r:.3f}, reproj err {worst:.2e}px -> {out}")

if __name__ == "__main__":
    main(sys.argv[1])
