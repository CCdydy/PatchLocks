"""Register furless_reshaped.obj to the frame-210 multi-view silhouettes.

Finds a similarity transform (scale s, rotation R, translation t) mapping the mesh
frame into the raw Artemis camera world. If a similarity yields high multi-view IoU
the mismatch is just a frame transform; if no rotation/scale gets there, the mesh
pose differs from frame 210 (a harder problem). Reports best transform + overlay.
"""
import itertools
import cv2
import numpy as np
import trimesh

RAW = "patchlocks/deploy/neuralfur_beagle_official/raw_cams"
IMG = "patchlocks/deploy/neuralfur_beagle_official/realviews/frame210_raw"
OBJ = "patchlocks/datasets/neuralfur/official_results/extracted/neuralfur/beagle_dog/furless_reshaped.obj"
OUT = "patchlocks/deploy/neuralfur_beagle_official/realviews"
W, H = 1920, 1080


def read_intrinsic(path):
    L = [l.strip() for l in open(path) if l.strip() != ""]
    Ks = []
    i = 0
    while i + 3 < len(L) + 1 and i + 3 < len(L):
        Ks.append(np.array([list(map(float, L[i + 1].split())), list(map(float, L[i + 2].split())), list(map(float, L[i + 3].split()))]))
        i += 4
    return Ks


def read_campose(path):
    a = np.loadtxt(path)
    C = np.zeros((len(a), 4, 4)); C[:, 3, 3] = 1
    C[:, 0:3, 2] = a[:, 0:3]; C[:, 0:3, 0] = a[:, 3:6]; C[:, 0:3, 1] = a[:, 6:9]; C[:, 0:3, 3] = a[:, 9:12]
    return C


def proj_m(K, c2w, n=0.05):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    w2c = np.linalg.inv(c2w)
    Pcol = np.array([[2 * fx / W, 0, 2 * cx / W - 1, 0], [0, -2 * fy / H, 1 - 2 * cy / H, 0], [0, 0, 1, -n], [0, 0, 1, 0]])
    return (Pcol @ w2c).T.astype(np.float32)


def silh(V, m):
    homo = np.concatenate([V, np.ones((len(V), 1))], 1).astype(np.float32)
    clip = homo @ m
    w = clip[:, 3]; ok = w > 1e-6
    ndc = clip[:, :3] / np.where(np.abs(w[:, None]) > 1e-8, w[:, None], 1e-8)
    px = (ndc[:, 0] * 0.5 + 0.5) * W; py = (1 - (ndc[:, 1] * 0.5 + 0.5)) * H
    xi = px.astype(np.int32); yi = py.astype(np.int32)
    g = ok & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    mk = np.zeros((H, W), np.uint8); mk[yi[g], xi[g]] = 255
    return cv2.morphologyEx(mk, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))


Ks = read_intrinsic(f"{RAW}/Intrinsic.inf")
C2W = read_campose(f"{RAW}/CamPose.inf")
NV = len(C2W)
alphas = {v: (cv2.imread(f"{IMG}/img_{v:04d}_alpha.png", 0) > 30).astype(np.uint8) for v in range(NV)}

# --- world animal center via ray intersection of alpha centroids ---
A = np.zeros((3, 3)); b = np.zeros(3)
for v in range(NV):
    ys, xs = np.where(alphas[v] > 0)
    if len(xs) < 50:
        continue
    u, vv = xs.mean(), ys.mean()
    K = Ks[v]; d = C2W[v, :3, :3] @ np.array([(u - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], 1.0])
    d = d / np.linalg.norm(d); C = C2W[v, :3, 3]
    P = np.eye(3) - np.outer(d, d); A += P; b += P @ C
Xc = np.linalg.solve(A, b)
print(f"world animal center X* = {Xc.round(4)}")

# --- world scale from mean alpha radius ---
rad_px = np.mean([np.sqrt((alphas[v] > 0).sum() / np.pi) for v in range(NV)])
depth = np.mean([np.linalg.norm(Xc - C2W[v, :3, 3]) for v in range(NV)])
rho_world = rad_px * depth / Ks[0][0, 0]
print(f"alpha mean radius {rad_px:.1f}px depth {depth:.3f} -> world radius {rho_world:.4f}")

mesh = trimesh.load(OBJ, process=False)
V0 = np.asarray(mesh.vertices, np.float64)
Vsub = V0[np.random.default_rng(0).choice(len(V0), 15000, replace=False)]
mc = V0.mean(0)
rho_mesh = np.sqrt(((V0 - mc) ** 2).sum(1).mean())
s0 = rho_world / rho_mesh
print(f"mesh center {mc.round(3)} mesh radius {rho_mesh:.4f} -> base scale {s0:.4f}")

ms = [proj_m(Ks[v], C2W[v]) for v in range(NV)]
test_views = [0, 4, 9, 13, 18, 22, 27, 31]


def total_iou(Vw):
    tot = 0.0
    for v in test_views:
        s = silh(Vw, ms[v]); a = alphas[v] * 255
        inter = ((s > 0) & (a > 0)).sum(); uni = ((s > 0) | (a > 0)).sum()
        tot += inter / max(uni, 1)
    return tot / len(test_views)


def rot(axis, deg):
    t = np.deg2rad(deg); c, s = np.cos(t), np.sin(t)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

# coarse search: which mesh axis is world-up (align mesh axis->world Y), x scale factor, yaw
best = (-1, None)
up_aligns = {"+y": np.eye(3), "-y": rot("x", 180), "+z": rot("x", -90), "-z": rot("x", 90), "+x": rot("z", 90), "-x": rot("z", -90)}
for upname, Rup in up_aligns.items():
    for sf in [0.7, 0.85, 1.0, 1.15, 1.3]:
        s = s0 * sf
        for yaw in range(0, 360, 30):
            R = rot("y", yaw) @ Rup
            Vw = (s * (Vsub - mc) @ R.T) + Xc
            iou = total_iou(Vw)
            if iou > best[0]:
                best = (iou, (upname, sf, yaw, s, R))
print(f"COARSE best IoU {best[0]:.3f} up={best[1][0]} sf={best[1][1]} yaw={best[1][2]} scale={best[1][3]:.4f}")

# fine yaw + scale around best
up, sf, yaw0, _, _ = best[1]; Rup = up_aligns[up]
bf = best
for sf2 in [sf * k for k in [0.9, 0.95, 1.0, 1.05, 1.1]]:
    s = s0 * sf2
    for yaw in range(yaw0 - 25, yaw0 + 26, 5):
        R = rot("y", yaw) @ Rup
        Vw = (s * (Vsub - mc) @ R.T) + Xc
        iou = total_iou(Vw)
        if iou > bf[0]:
            bf = (iou, (up, sf2, yaw, s, R))
print(f"FINE best IoU {bf[0]:.3f} up={bf[1][0]} sf={bf[1][1]:.3f} yaw={bf[1][2]} scale={bf[1][3]:.5f}")

up, sf, yaw, s, R = bf[1]
np.savez(f"{OUT}/_furless_to_world.npz", scale=s, R=R, mesh_center=mc, world_center=Xc, up=up, yaw=yaw, iou=bf[0])
# overlay with best transform (full verts)
Vw = (s * (V0 - mc) @ R.T) + Xc
panels = []
for v in [0, 4, 9, 13, 18, 27]:
    sm = silh(Vw, ms[v]); a = alphas[v] * 255
    over = np.zeros((H, W, 3), np.uint8); over[..., 2] = a; over[..., 1] = sm
    over = cv2.resize(over, (480, 270))
    inter = ((sm > 0) & (a > 0)).sum(); uni = ((sm > 0) | (a > 0)).sum()
    cv2.putText(over, f"v{v} IoU{inter/max(uni,1):.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    panels.append(over)
cv2.imwrite(f"{OUT}/_register_check.png", np.concatenate([np.concatenate(panels[:3], 1), np.concatenate(panels[3:], 1)], 0))
print(f"saved {OUT}/_register_check.png and _furless_to_world.npz")
