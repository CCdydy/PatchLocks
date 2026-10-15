"""Generic: regenerate non-singular mesh->clip matrices (proj_m + near-plane) with
the furless->world transform T baked in, overwrite a NeuralFur package's matrices.pkl.
Parameterized version of _beagle_rebake_matrices.py.

  python _nf_rebake_matrices.py --furless_to_world <npz> --raw_cams <dir> --out_pkl <pkl>
"""
import argparse
import pickle
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--furless_to_world", required=True)
ap.add_argument("--raw_cams", required=True)
ap.add_argument("--out_pkl", required=True)
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
    return (P @ w2c).T


Ks = read_intrinsic(f"{a.raw_cams}/Intrinsic.inf")
C2W = read_campose(f"{a.raw_cams}/CamPose.inf")
tw = np.load(a.furless_to_world, allow_pickle=True)
s = float(tw["scale"]); R = np.asarray(tw["R"], np.float64); mc = np.asarray(tw["mesh_center"], np.float64); Xc = np.asarray(tw["world_center"], np.float64)
T = np.eye(4); T[0:3, 0:3] = s * R.T; T[3, 0:3] = Xc - s * (mc @ R.T)

mats = {f"{v:04d}": (T @ proj_m(Ks[v], C2W[v])).astype(np.float32) for v in range(len(C2W))}
dets = [abs(float(np.linalg.det(m.astype(np.float64)))) for m in mats.values()]
with open(a.out_pkl, "wb") as f:
    pickle.dump(mats, f)
print(f"baked {len(mats)} mesh->clip matrices -> {a.out_pkl} | |det| min {min(dets):.3e} (nonzero => front_visible OK)")
