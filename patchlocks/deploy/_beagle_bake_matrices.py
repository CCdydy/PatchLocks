"""Bake the furless->world similarity into the imported world->clip matrices so
that mesh-frame patchpack points project correctly. Verify with patchpack root_pos
and furless silhouette vs the GT alpha."""
import pickle
import cv2
import numpy as np

BASE = "patchlocks/deploy/neuralfur_beagle_official"
DATA = f"{BASE}/realviews/nf_data_f210"
PKL = f"{DATA}/3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl"
IMG = f"{BASE}/realviews/frame210_raw"
PACK = "patchlocks/deploy/neuralfur_beagle_geopatch_fine/geodesic_patchpack.npz"
W, H = 1920, 1080

tw = np.load(f"{BASE}/realviews/_furless_to_world.npz", allow_pickle=True)
s = float(tw["scale"]); R = np.asarray(tw["R"], np.float64); mc = np.asarray(tw["mesh_center"], np.float64); Xc = np.asarray(tw["world_center"], np.float64)
T = np.eye(4)
T[0:3, 0:3] = s * R.T
T[3, 0:3] = Xc - s * (mc @ R.T)
print(f"T (mesh->world, row-form):\n{T.round(4)}")

mats = pickle.load(open(PKL, "rb"))
with open(PKL.replace("30000_matrices.pkl", "30000_matrices_raw.pkl"), "wb") as f:
    pickle.dump(mats, f)
baked = {k: (T @ np.asarray(v, np.float64)).astype(np.float32) for k, v in mats.items()}
with open(PKL, "wb") as f:
    pickle.dump(baked, f)
print(f"baked {len(baked)} matrices (raw backed up to 30000_matrices_raw.pkl)")


def project(P, m):
    homo = np.concatenate([P, np.ones((len(P), 1))], 1).astype(np.float32)
    clip = homo @ m
    w = clip[:, 3]; ok = w > 1e-6
    ndc = clip[:, :3] / np.where(np.abs(w[:, None]) > 1e-8, w[:, None], 1e-8)
    px = (ndc[:, 0] * 0.5 + 0.5) * W; py = (1 - (ndc[:, 1] * 0.5 + 0.5)) * H
    return px, py, ok


d = np.load(PACK, allow_pickle=True)
root_pos = d["root_pos"].astype(np.float64)
sub = root_pos[np.random.default_rng(0).choice(len(root_pos), 40000, replace=False)]
panels = []
for v in [0, 4, 9, 13, 18, 27]:
    m = baked[f"{v:04d}"]
    px, py, ok = project(sub, m)
    a = (cv2.imread(f"{IMG}/img_{v:04d}_alpha.png", 0) > 30).astype(np.uint8)
    xi = px.astype(np.int32); yi = py.astype(np.int32)
    g = ok & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    inside = a[yi[g].clip(0, H - 1), xi[g].clip(0, W - 1)].mean()  # frac of roots landing on dog
    over = np.zeros((H, W, 3), np.uint8); over[..., 2] = a * 255
    over[yi[g].clip(0, H - 1), xi[g].clip(0, W - 1), 1] = 255
    over = cv2.resize(over, (480, 270))
    cv2.putText(over, f"v{v} on-dog {inside:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    panels.append(over)
    print(f"view {v:2d}: {g.sum()}/{len(sub)} roots in-frame, {inside:.3f} land on dog alpha")
cv2.imwrite(f"{BASE}/realviews/_baked_roots_check.png", np.concatenate([np.concatenate(panels[:3], 1), np.concatenate(panels[3:], 1)], 0))
print(f"saved {BASE}/realviews/_baked_roots_check.png")
