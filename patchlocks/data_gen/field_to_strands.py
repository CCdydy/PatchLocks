"""Stage-2 field-conditioned sampler. Turns a DETERMINISTIC fur field (lambda/direction/length/curl/
radius/color) back into a STOCHASTIC strand realization: sample roots from lambda, look up per-root
field values (bilinear) + add the stochastic residual the field deliberately dropped, then grow strands
with the v4 growth model. This is the generative half of field-first (Stage 3 will replace growth with a
learned field-conditioned decoder). Returns the standard patch dict so render_loop/qa work unchanged.
"""
import numpy as np


def _unit(v, eps=1e-9):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def _bilinear(field, uv):
    """Sample (H,W,C) field at uv in [0,1] (uv[:,0]=u=col, uv[:,1]=v=row). Returns (N,C)."""
    H, W = field.shape[:2]
    x = np.clip(uv[:, 0] * W - 0.5, 0, W - 1); y = np.clip(uv[:, 1] * H - 0.5, 0, H - 1)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, W - 1); y1 = np.minimum(y0 + 1, H - 1)
    fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    f = field.reshape(H, W, -1)
    return ((f[y0, x0] * (1 - fx) + f[y0, x1] * fx) * (1 - fy) + (f[y0, x1 * 0 + x0] * (1 - fx) + f[y1, x1] * fx) * fy)


def _grow(roots, d0, length, curl_amp, rng, L=32, gravity=0.2, noise_amp=0.006, curl_freq=1.2):
    N = len(roots); seg = (length / (L - 1)).astype(np.float32)
    pts = np.zeros((N, L, 3), np.float32); pts[:, 0] = roots
    dirs = _unit(d0 + rng.normal(0, 0.02, (N, 3)).astype(np.float32)); gz = np.array([0, 0, -1.0], np.float32)
    for k in range(1, L):
        t = k / (L - 1)
        dirs = _unit(dirs + gravity * t * gz + noise_amp * rng.normal(0, 1, (N, 3)).astype(np.float32))
        pts[:, k] = pts[:, k - 1] + seg[:, None] * dirs; pts[:, k, 2] = np.maximum(pts[:, k, 2], 0.0)
    axis = _unit(d0); ref = np.tile(np.array([0, 0, 1.0], np.float32), (N, 1))
    ref[np.abs((axis * ref).sum(1)) > 0.95] = np.array([1, 0, 0.0], np.float32)
    e1 = _unit(np.cross(axis, ref)); e2 = _unit(np.cross(axis, e1)); tt = np.linspace(0, 1, L)[None]
    phase = rng.uniform(0, 2 * np.pi, N).astype(np.float32)
    ang = 2 * np.pi * curl_freq * tt + phase[:, None]; amp = curl_amp[:, None] * tt
    pts = pts + (amp[..., None] * (np.cos(ang)[..., None] * e1[:, None] + np.sin(ang)[..., None] * e2[:, None])).astype(np.float32)
    pts[..., 2] = np.maximum(pts[..., 2], 0.0); pts[:, 0] = roots
    # resample each strand to target ARC length (curl/gravity inflate arc above the straight extent)
    seg = np.linalg.norm(pts[:, 1:] - pts[:, :-1], axis=2); cum = np.concatenate([np.zeros((N, 1), np.float32), np.cumsum(seg, 1)], 1)
    td = np.linspace(0, 1, L)[None] * length[:, None]                       # (N,L) target arc distances
    idx = np.clip((td[..., None] >= cum[:, None, :]).sum(-1) - 1, 0, L - 2)
    c0 = np.take_along_axis(cum, idx, 1); c1 = np.take_along_axis(cum, idx + 1, 1)
    w = ((td - c0) / np.maximum(c1 - c0, 1e-9))[..., None]
    p0 = np.take_along_axis(pts, idx[..., None], 1); p1 = np.take_along_axis(pts, (idx + 1)[..., None], 1)
    out = (p0 * (1 - w) + p1 * w).astype(np.float32); out[..., 2] = np.maximum(out[..., 2], 0.0); out[:, 0] = roots
    return out


def sample_from_fields(fields, size=1.0, seed=0, n_scale=1.0, curl_k=0.5,
                       jit_dir=0.03, jit_len=0.06, flow_jit=0.45, L=32):
    """fields: dict with lambda_map(H,W), direction_map(H,W,3), length_mu(H,W[,1]), curl_mu(H,W[,1]),
    radius_mu(H,W[,1]), color_map(H,W,3), valid_mask(H,W). Returns patch dict."""
    rng = np.random.default_rng(seed)
    lam = np.asarray(fields["lambda_map"], np.float32); H, W = lam.shape
    valid = np.asarray(fields.get("valid_mask", np.ones_like(lam)), np.float32) > 0.5
    # STRATIFIED root sampling (per-cell count = floor(lam)+Bernoulli(frac)) -> even, blue-noise-ish, low clumping
    lam2 = np.clip(lam, 0, None) * valid * n_scale
    base = np.floor(lam2).astype(int); extra = (rng.uniform(0, 1, lam2.shape) < (lam2 - base)).astype(int)
    ncell = (base + extra).ravel()
    nz = np.nonzero(ncell)[0]; cells = np.repeat(nz, ncell[nz]); N = max(len(cells), 1)
    if len(cells) == 0: cells = np.array([int(np.argmax(lam2))]);
    cj, ci = np.divmod(cells, W)                                            # row=v, col=u
    u = (ci + rng.uniform(0, 1, len(cells))) / W; v = (cj + rng.uniform(0, 1, len(cells))) / H
    uv = np.stack([u, v], 1).astype(np.float32); N = len(uv)
    d0 = _unit(_bilinear(fields["direction_map"], uv))
    d0 = _unit(d0 + rng.normal(0, jit_dir, (N, 3)).astype(np.float32))
    length = _bilinear(fields["length_mu"], uv)[:, 0] * (1 + rng.normal(0, jit_len, N)).astype(np.float32)
    length = np.clip(length, 0.03, None).astype(np.float32)
    curl_ratio = np.clip(_bilinear(fields["curl_mu"], uv)[:, 0], 0, 1)
    curl_amp = (curl_ratio * length * curl_k).astype(np.float32)            # ratio -> helix amplitude (approx)
    radius = _bilinear(fields["radius_mu"], uv)[:, 0].astype(np.float32) if "radius_mu" in fields else np.full(N, 0.002, np.float32)
    color = np.clip(_bilinear(fields["color_map"], uv), 0, 1).astype(np.float32) if "color_map" in fields else np.full((N, 3), 0.5, np.float32)
    roots = np.stack([(u - 0.5) * size, (v - 0.5) * size, np.zeros(N)], 1).astype(np.float32)
    strands = _grow(roots, d0, length, curl_amp, rng, L=L)
    if "flow2d" in fields:                                                  # inject clump/attractor convergence
        flow = _bilinear(fields["flow2d"], uv).astype(np.float32)           # (N,2) GT net lateral tip-root (cell mean)
        fmag = np.linalg.norm(flow, axis=1, keepdims=True)
        flow = flow + rng.normal(0, 1, (N, 2)).astype(np.float32) * (flow_jit * fmag + 0.012)  # per-strand scatter -> keep natural tip spread
        cur = strands[:, -1, :2] - strands[:, 0, :2]                        # current net lateral from growth
        corr = (flow - cur)[:, None, :] * (np.linspace(0, 1, L) ** 2)[None, :, None]   # quadratic lean: roots stay aligned, tips converge (clump pull ~t^2)
        strands[:, :, :2] = strands[:, :, :2] + corr.astype(np.float32); strands[:, 0] = roots
    tip = np.clip(color * 1.08 + 0.02, 0, 1).astype(np.float32)
    params = dict(source="field_to_strands", n_strands=int(N), seed=int(seed), version="stage2")
    return dict(strands=strands, root_uv=uv, color=color, color_tip=tip,
                radius=radius, size=np.float32(size), params=params)


if __name__ == "__main__":
    import argparse, os, json
    ap = argparse.ArgumentParser(); ap.add_argument("field_npz"); ap.add_argument("--out", default="patchlocks/datasets/debug/stage2_patch.npz")
    ap.add_argument("--size", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); z = np.load(a.field_npz)
    d = sample_from_fields({k: z[k] for k in z.files}, size=a.size, seed=a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, strands=d["strands"], root_uv=d["root_uv"], color=d["color"], color_tip=d["color_tip"],
             radius=d["radius"], size=d["size"], params=json.dumps(d["params"]))
    print(f"sampled {len(d['strands'])} strands -> {a.out}")
