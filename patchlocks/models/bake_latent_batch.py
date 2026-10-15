"""Batch-bake a StrandCodec 64-D latent texture (+ density + mask) for every sample under <renders_dir>.
Defaults match the old pipeline, but --codec/--out-name allow re-baking with a new codec without overwriting
target_latent64.npz.
Run in difflocks env from repo root:
  python patchlocks/models/bake_latent_batch.py patchlocks/datasets/v6/renders 64 \
    --codec patchlocks/models/fur_strand_codec_v6upright.pt \
    --out-name target_latent64_v6codec.npz
"""
import os, sys, glob, argparse
sys.path.insert(0, os.getcwd())
import numpy as np, torch
from models.strand_codec import StrandCodec, normalize_gt_data
from utils.strand_util import compute_dirs

torch.set_grad_enabled(False)
ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("H", nargs="?", type=int, default=32)
ap.add_argument("legacy_codec", nargs="?", help="back-compat positional codec path")
ap.add_argument("legacy_outname", nargs="?", help="back-compat positional latent filename")
ap.add_argument("--codec", default=None, help="StrandCodec checkpoint used to encode z64")
ap.add_argument("--out-name", "--latent-name", dest="out_name", default=None, help="latent filename written inside each render dir")
ap.add_argument("--straight", action="store_true", help="ablation: replace each strand by the straight root->tip line before encoding (removes CURL, keeps direction+length+elevation). z64 then has no curl.")
a = ap.parse_args()
root = a.root; H = a.H
CODEC = a.codec or a.legacy_codec or "patchlocks/models/fur_strand_codec.pt"
OUTNAME = a.out_name or a.legacy_outname or "target_latent64.npz"
ck = torch.load(CODEC, weights_only=False)
print(f"codec={CODEC} | out={OUTNAME}", flush=True)
norm = {k: torch.tensor(v, device="cuda", dtype=torch.float32) for k, v in ck["fur_norm"].items()}
codec = StrandCodec(do_vae=False, decode_type="dir", scale_init=30.0, nr_verts_per_strand=256,
                    nr_values_to_decode=255, dim_per_value_decoded=3).cuda().eval()
codec.load_state_dict(ck["state_dict"])

def resample(s, M=256):
    seg = np.linalg.norm(np.diff(s, axis=0), axis=1); cl = np.concatenate([[0], np.cumsum(seg)])
    tt = np.linspace(0, cl[-1] + 1e-9, M); return np.stack([np.interp(tt, cl, s[:, i]) for i in range(3)], 1)

dirs = sorted(d for d in glob.glob(root + "/*") if os.path.exists(d + "/patch.npz"))
print(f"baking latent64 for {len(dirs)} samples")
for i, D in enumerate(dirs):
    p = np.load(f"{D}/patch.npz", allow_pickle=True)
    S = p["strands"].astype(np.float32); uv = p["root_uv"].astype(np.float32); N = len(S)
    S256 = np.stack([resample(s) for s in S]).astype(np.float32); S256 = S256 - S256[:, :1]
    if a.straight:                                                                   # remove curl: straight root->tip line (keeps dir+len+elevation)
        S256 = (np.linspace(0, 1, S256.shape[1], dtype=np.float32)[None, :, None] * S256[:, -1:, :]).astype(np.float32)
    pos = torch.tensor(S256, device="cuda")
    gt = {"strand_positions": pos, "strand_directions": compute_dirs(pos, append_last_dir=False)}
    z = codec.encoder(normalize_gt_data(gt, norm))["z"].cpu().numpy()
    ti = np.clip((uv[:, 0] * (H - 1)).round().astype(int), 0, H - 1)
    tj = np.clip((uv[:, 1] * (H - 1)).round().astype(int), 0, H - 1)
    T = np.zeros((H, H, 64)); cnt = np.zeros((H, H))
    np.add.at(T, (tj, ti), z); np.add.at(cnt, (tj, ti), 1.0); occ = cnt > 0; T[occ] /= cnt[occ, None]
    np.savez(f"{D}/{OUTNAME}", target=T.astype(np.float32), density=cnt[..., None].astype(np.float32), mask=occ)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(dirs)}", flush=True)
print("BAKE64-BATCH-DONE")
