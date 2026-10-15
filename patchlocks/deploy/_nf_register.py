"""Generic: register a NeuralFur furless mesh to one Artemis frame's multi-view
silhouettes -> similarity transform mesh->camera-world. Parameterized version of
_beagle_register.py so it works for any animal (cat/whiteTiger/...).

  python _nf_register.py --furless <obj> --raw_cams <dir> --img_dir <frame views> --out <dir>
"""
import argparse
import cv2
import numpy as np
import trimesh

ap = argparse.ArgumentParser()
ap.add_argument("--furless", required=True)
ap.add_argument("--raw_cams", required=True, help="dir with CamPose.inf + Intrinsic.inf")
ap.add_argument("--img_dir", required=True, help="dir with img_%04d.png + img_%04d_alpha.png")
ap.add_argument("--out", required=True)
ap.add_argument("--W", type=int, default=1920)
ap.add_argument("--H", type=int, default=1080)
a = ap.parse_args()
W, H = a.W, a.H


def read_intrinsic(path):
    L = [l.strip() for l in open(path) if l.strip() != ""]
    Ks, i = [], 0
    while i + 3 < len(L):
        Ks.append(np.array([list(map(float, L[i + 1].split())), list(map(float, L[i + 2].split())), list(map(float, L[i + 3].split()))]))
        i += 4
    return Ks


def read_campose(path):
    arr = np.loadtxt(path)
    C = np.zeros((len(arr), 4, 4)); C[:, 3, 3] = 1
    C[:, 0:3, 2] = arr[:, 0:3]; C[:, 0:3, 0] = arr[:, 3:6]; C[:, 0:3, 1] = arr[:, 6:9]; C[:, 0:3, 3] = arr[:, 9:12]
    return C


def proj_m(K, c2w, n=0.05):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    w2c = np.linalg.inv(c2w)
    P = np.array([[2 * fx / W, 0, 2 * cx / W - 1, 0], [0, -2 * fy / H, 1 - 2 * cy / H, 0], [0, 0, 1, -n], [0, 0, 1, 0]])
    return (P @ w2c).T.astype(np.float32)


def silh(V, m):
    homo = np.concatenate([V, np.ones((len(V), 1))], 1).astype(np.float32)
    clip = homo @ m; w = clip[:, 3]; ok = w > 1e-6
    ndc = clip[:, :3] / np.where(np.abs(w[:, None]) > 1e-8, w[:, None], 1e-8)
    px = (ndc[:, 0] * 0.5 + 0.5) * W; py = (1 - (ndc[:, 1] * 0.5 + 0.5)) * H
    xi = px.astype(np.int32); yi = py.astype(np.int32)
    g = ok & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    mk = np.zeros((H, W), np.uint8); mk[yi[g], xi[g]] = 255
    return cv2.morphologyEx(mk, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))


def rot(axis, deg):
    t = np.deg2rad(deg); c, s = np.cos(t), np.sin(t)
    return {"y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]), "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


Ks = read_intrinsic(f"{a.raw_cams}/Intrinsic.inf")
C2W = read_campose(f"{a.raw_cams}/CamPose.inf")
NV = len(C2W)
alphas = {v: (cv2.imread(f"{a.img_dir}/img_{v:04d}_alpha.png", 0) > 30).astype(np.uint8) for v in range(NV)}

A = np.zeros((3, 3)); b = np.zeros(3)
for v in range(NV):
    ys, xs = np.where(alphas[v] > 0)
    if len(xs) < 50:
        continue
    u, vv = xs.mean(), ys.mean()
    K = Ks[v]; d = C2W[v, :3, :3] @ np.array([(u - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], 1.0])
    d /= np.linalg.norm(d); C = C2W[v, :3, 3]
    P = np.eye(3) - np.outer(d, d); A += P; b += P @ C
Xc = np.linalg.solve(A, b)
rad_px = np.mean([np.sqrt((alphas[v] > 0).sum() / np.pi) for v in range(NV)])
depth = np.mean([np.linalg.norm(Xc - C2W[v, :3, 3]) for v in range(NV)])
rho_world = rad_px * depth / Ks[0][0, 0]

V0 = np.asarray(trimesh.load(a.furless, process=False).vertices, np.float64)
Vsub = V0[np.random.default_rng(0).choice(len(V0), 15000, replace=False)]
mc = V0.mean(0); rho_mesh = np.sqrt(((V0 - mc) ** 2).sum(1).mean()); s0 = rho_world / rho_mesh
print(f"world center {Xc.round(4)} | world R {rho_world:.4f} mesh R {rho_mesh:.4f} base scale {s0:.4f}")
ms = [proj_m(Ks[v], C2W[v]) for v in range(NV)]
test_views = list(np.linspace(0, NV - 1, 8).astype(int))


def total_iou(Vw):
    tot = 0.0
    for v in test_views:
        s = silh(Vw, ms[v]); al = alphas[v] * 255
        inter = ((s > 0) & (al > 0)).sum(); uni = ((s > 0) | (al > 0)).sum()
        tot += inter / max(uni, 1)
    return tot / len(test_views)


up_aligns = {"+y": np.eye(3), "-y": rot("x", 180), "+z": rot("x", -90), "-z": rot("x", 90), "+x": rot("z", 90), "-x": rot("z", -90)}
best = (-1, None)
for upn, Rup in up_aligns.items():
    for sf in [0.7, 0.85, 1.0, 1.15, 1.3]:
        s = s0 * sf
        for yaw in range(0, 360, 30):
            R = rot("y", yaw) @ Rup
            Vw = (s * (Vsub - mc) @ R.T) + Xc
            iou = total_iou(Vw)
            if iou > best[0]:
                best = (iou, (upn, sf, yaw, s, R))
upn, sf, yaw0, _, _ = best[1]; Rup = up_aligns[upn]; bf = best
for sf2 in [sf * k for k in [0.9, 0.95, 1.0, 1.05, 1.1]]:
    s = s0 * sf2
    for yaw in range(yaw0 - 25, yaw0 + 26, 5):
        R = rot("y", yaw) @ Rup
        Vw = (s * (Vsub - mc) @ R.T) + Xc
        iou = total_iou(Vw)
        if iou > bf[0]:
            bf = (iou, (upn, sf2, yaw, s, R))
upn, sf, yaw, s, R = bf[1]
np.savez(f"{a.out}/_furless_to_world.npz", scale=s, R=R, mesh_center=mc, world_center=Xc, up=upn, yaw=yaw, iou=bf[0])
print(f"BEST IoU {bf[0]:.3f} up={upn} scale={s:.5f} yaw={yaw}")
Vw = (s * (V0 - mc) @ R.T) + Xc
panels = []
for v in test_views[:6]:
    sm = silh(Vw, ms[v]); al = alphas[v] * 255
    over = np.zeros((H, W, 3), np.uint8); over[..., 2] = al; over[..., 1] = sm
    over = cv2.resize(over, (440, 248))
    inter = ((sm > 0) & (al > 0)).sum(); uni = ((sm > 0) | (al > 0)).sum()
    cv2.putText(over, f"v{v} IoU{inter/max(uni,1):.2f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    panels.append(over)
cv2.imwrite(f"{a.out}/_register_check.png", np.concatenate([np.concatenate(panels[:3], 1), np.concatenate(panels[3:], 1)], 0))
print(f"saved {a.out}/_furless_to_world.npz + _register_check.png")
