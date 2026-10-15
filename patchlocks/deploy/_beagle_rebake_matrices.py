"""Regenerate beagle matrices with a NON-singular depth row (proj_m, near-plane -n)
composed with the furless->world transform T. The import_artemis_dfa matrices put
clip_z==clip_w (rank-deficient) which breaks front_visible's camera-center inverse."""
import pickle
import numpy as np

BASE = "patchlocks/deploy/neuralfur_beagle_official"
RAW = f"{BASE}/raw_cams"
PKL = f"{BASE}/realviews/nf_data_f210/3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl"
W, H = 1920, 1080


def read_intrinsic(path):
    L = [l.strip() for l in open(path) if l.strip() != ""]
    Ks = []
    i = 0
    while i + 3 < len(L):
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
    return (Pcol @ w2c).T


Ks = read_intrinsic(f"{RAW}/Intrinsic.inf")
C2W = read_campose(f"{RAW}/CamPose.inf")
tw = np.load(f"{BASE}/realviews/_furless_to_world.npz", allow_pickle=True)
s = float(tw["scale"]); R = np.asarray(tw["R"], np.float64); mc = np.asarray(tw["mesh_center"], np.float64); Xc = np.asarray(tw["world_center"], np.float64)
T = np.eye(4); T[0:3, 0:3] = s * R.T; T[3, 0:3] = Xc - s * (mc @ R.T)

mats = {}
for v in range(len(C2W)):
    m_world = proj_m(Ks[v], C2W[v])      # non-singular world->clip (row form)
    mats[f"{v:04d}"] = (T @ m_world).astype(np.float32)   # mesh->clip
# sanity: invertible?
dets = [float(np.linalg.det(m.astype(np.float64))) for m in mats.values()]
print(f"regenerated {len(mats)} matrices; |det| min={min(abs(np.array(dets))):.4e} (nonzero => front_visible OK)")
with open(PKL, "wb") as f:
    pickle.dump(mats, f)
print(f"overwrote {PKL}")
