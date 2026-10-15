"""Fur-DIRECTION extractor: a dedicated module that pulls the in-plane fur flow out of the INPUT IMAGE (where it
is strong, structure-tensor coherence ~0.7) and expresses it in the PATCH-UV / canonical frame (where RGBPROJ on
the scalp shell loses it, coherence ~0.05). Output is a per-texel axial orientation field [cos2θ, sin2θ, coh].

V2 use: do not ask diffusion to hide absolute groom direction inside z64. Use this measured field to rotate
direction-canonical decoded strands back into image/surface direction. Older dir_cond experiments fed this into
the diffusion and collapsed to a mean z64 direction; that path is kept only for ablations.

Why full-res + Jacobian: the flow lives in the strand body and as thin lines; sampling raw RGB onto a 64-px
shell blurs it away. So we (1) measure the structure tensor on the full image, (2) sample it at each UV texel's
image projection, (3) rotate the image-space orientation into UV space via the grid's local Jacobian — which
absorbs the camera azimuth + oblique foreshortening + the canonical rotation automatically.
"""
import numpy as np, cv2


def fur_texture_highlight(img_rgb):
    """Return a contrast-normalized fur texture map for direction extraction.

    This is a lightweight, dependency-free texture highlighter: local contrast
    equalization + band-pass detail + bright/dark line morphology.  It preserves
    the line/ridge structure that structure tensor needs, while reducing broad
    albedo/shading ramps that contaminate direction on markings.
    """
    g0 = cv2.cvtColor((np.clip(img_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g0).astype(np.float32) / 255.0
    fine = cv2.GaussianBlur(clahe, (0, 0), 0.7)
    coarse = cv2.GaussianBlur(clahe, (0, 0), 4.0)
    dog = np.abs(fine - coarse)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    top = cv2.morphologyEx(clahe, cv2.MORPH_TOPHAT, k)
    black = cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, k)
    tex = 0.58 * dog + 0.22 * top + 0.20 * black
    lo, hi = np.percentile(tex, [1.0, 99.0])
    tex = (tex - lo) / max(float(hi - lo), 1e-6)
    return np.clip(tex, 0.0, 1.0).astype(np.float32)


def fur_texture_line_prior(img_rgb):
    """Return a stricter black/white fur-line map for direction extraction.

    `fur_texture_highlight` is intentionally permissive: it preserves many local
    contrast cues and works well on clean synthetic patches.  On real animals,
    broad albedo/marking boundaries can dominate the structure tensor and pull
    the groom direction toward the color edge.  This variant keeps the same
    "black-white line" spirit used by the V2 texture prior, but suppresses broad
    color edges before measuring orientation.
    """
    g0 = cv2.cvtColor((np.clip(img_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g0).astype(np.float32) / 255.0

    # Thin fur strokes: small-band DoG plus small top/black-hat features.
    fine = cv2.GaussianBlur(g, (0, 0), 0.55)
    mid = cv2.GaussianBlur(g, (0, 0), 1.8)
    dog_small = np.abs(fine - mid)
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    top = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k_small)
    black = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k_small)
    tex = 0.55 * dog_small + 0.25 * top + 0.20 * black

    # Suppress broad marking/shading edges.  Those are strong in a coarse
    # gradient image, while real hair lines are mostly high-frequency.
    coarse = cv2.GaussianBlur(g, (0, 0), 5.0)
    bgx = cv2.Sobel(coarse, cv2.CV_32F, 1, 0, ksize=3)
    bgy = cv2.Sobel(coarse, cv2.CV_32F, 0, 1, ksize=3)
    broad_grad = cv2.GaussianBlur(np.sqrt(bgx * bgx + bgy * bgy), (0, 0), 2.0)
    scale = np.percentile(broad_grad, 90.0) + 1e-9
    edge_gate = np.exp(-((broad_grad / scale) ** 2)).astype(np.float32)
    tex *= edge_gate

    # Local contrast normalization keeps dark fur usable without letting one
    # bright marking dominate the whole patch.
    loc = cv2.GaussianBlur(tex, (0, 0), 6.0)
    loc2 = cv2.GaussianBlur(tex * tex, (0, 0), 6.0)
    tex = (tex - loc) / np.sqrt(np.maximum(loc2 - loc * loc, 1e-6))
    tex = np.maximum(tex, 0.0)
    lo, hi = np.percentile(tex, [1.0, 99.5])
    tex = (tex - lo) / max(float(hi - lo), 1e-6)
    return np.clip(tex, 0.0, 1.0).astype(np.float32)


def _remap_scalar(x, px, py):
    return cv2.remap(x.astype(np.float32), px, py, cv2.INTER_LINEAR)


def extract_texture_prior(img_rgb, grid_ndc, inf, res, view_cos=1.0, sigma=2.5):
    """Build the V2 strong geometry-diffusion texture prior in patch UV.

    Channels are:
      0 gray          luminance, no color
      1 highlight_tex local contrast / bright-dark fur line enhancement
      2 HF_detail     signed-ish high-frequency fur detail, normalized to [0,1]
      3 ori_cos2      structure-tensor axial direction in UV
      4 ori_sin2
      5 ori_conf      structure-tensor coherence
      6 hf_strength   local high-frequency magnitude
      7 view_cos      view-normal cosine / projected reliability cue

    Raw RGB is intentionally not returned.  RGB is only the upstream source for
    this prior, DINO, and the separate color branch.
    """
    H = grid_ndc.shape[0]
    px = ((grid_ndc[..., 0] + 1) / 2 * res).astype(np.float32)
    py = ((grid_ndc[..., 1] + 1) / 2 * res).astype(np.float32)
    valid = np.asarray(inf, bool)

    gray = cv2.cvtColor((np.clip(img_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    high = fur_texture_highlight(img_rgb)
    blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
    hf = gray - blur
    lo, hi = np.percentile(hf, [1.0, 99.0])
    hf_detail = np.clip((hf - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0).astype(np.float32)

    gx = cv2.Sobel(high, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(high, cv2.CV_32F, 0, 1, ksize=3)
    hf_strength_img = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 1.0)
    scale = np.percentile(hf_strength_img, 95.0) + 1e-9
    hf_strength_img = np.clip(hf_strength_img / scale, 0.0, 1.0).astype(np.float32)

    Jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    Jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    Jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    sxx = _remap_scalar(Jxx, px, py)
    syy = _remap_scalar(Jyy, px, py)
    sxy = _remap_scalar(Jxy, px, py)
    ang_img = 0.5 * np.arctan2(2 * sxy, sxx - syy) + np.pi / 2
    coh = np.hypot(sxx - syy, 2 * sxy) / (sxx + syy + 1e-9)

    ex, ey = np.cos(ang_img), np.sin(ang_img)
    dpx_du = np.gradient(px, axis=1) * H
    dpx_dv = np.gradient(px, axis=0) * H
    dpy_du = np.gradient(py, axis=1) * H
    dpy_dv = np.gradient(py, axis=0) * H
    det = dpx_du * dpy_dv - dpx_dv * dpy_du
    det = np.where(np.abs(det) < 1e-9, 1e-9, det)
    duv_x = (dpy_dv * ex - dpx_dv * ey) / det
    duv_y = (-dpy_du * ex + dpx_du * ey) / det
    ang_uv = np.arctan2(duv_y, duv_x)

    out = np.stack(
        [
            _remap_scalar(gray, px, py),
            _remap_scalar(high, px, py),
            _remap_scalar(hf_detail, px, py),
            np.cos(2 * ang_uv).astype(np.float32),
            np.sin(2 * ang_uv).astype(np.float32),
            np.clip(coh, 0, 1).astype(np.float32),
            _remap_scalar(hf_strength_img, px, py),
            np.full((H, H), float(np.clip(view_cos, 0.0, 1.0)), np.float32),
        ],
        0,
    ).astype(np.float32)
    out[:, ~valid] = 0.0
    return out


def resolve_axial_sign(axial, conf=None, valid=None):
    """Turn an axial field [cos2θ,sin2θ] into a locally continuous oriented vector.

    Structure tensors only give an undirected axis.  Calling 0.5*atan2 per
    texel independently creates arbitrary 180-degree branch flips, which are
    especially visible around cowlicks/whorls.  This keeps the same axes but
    propagates signs through each visible component so neighboring vectors point
    in the locally consistent direction.
    """
    a = np.asarray(axial, np.float32)
    if a.ndim != 3 or a.shape[-1] != 2:
        raise ValueError(f"resolve_axial_sign expects (H,W,2), got {a.shape}")
    h, w = a.shape[:2]
    mag = np.linalg.norm(a, axis=-1)
    theta = 0.5 * np.arctan2(a[..., 1], a[..., 0])
    base = np.stack([np.cos(theta), np.sin(theta)], -1).astype(np.float32)

    if conf is None:
        c = mag.astype(np.float32)
    else:
        c = np.asarray(conf, np.float32) * np.clip(mag, 0.0, 1.0)
    mask = np.isfinite(base).all(-1) & (mag > 1e-6)
    if valid is not None:
        mask &= np.asarray(valid, bool)
    if not mask.any():
        return base

    order = np.argsort(-c.reshape(-1))
    seen = np.zeros((h, w), bool)
    out = np.zeros_like(base)
    neigh = ((1, 0), (-1, 0), (0, 1), (0, -1))
    from collections import deque

    for flat in order:
        y, x = divmod(int(flat), w)
        if seen[y, x] or not mask[y, x]:
            continue
        seen[y, x] = True
        out[y, x] = base[y, x]
        q = deque([(y, x)])
        while q:
            cy, cx = q.popleft()
            cur = out[cy, cx]
            for dy, dx in neigh:
                ny, nx = cy + dy, cx + dx
                if ny < 0 or nx < 0 or ny >= h or nx >= w or seen[ny, nx] or not mask[ny, nx]:
                    continue
                cand = base[ny, nx]
                out[ny, nx] = cand if float((cur * cand).sum()) >= 0.0 else -cand
                seen[ny, nx] = True
                q.append((ny, nx))

    out[~seen] = base[~seen]
    return out.astype(np.float32)


def extract_dir_field(img_rgb, grid_ndc, inf, res, sigma=2.5, texture_mode="gray"):
    """img_rgb (Hi,Wi,3) in [0,1]; grid_ndc (H,H,2) UV->image NDC; inf (H,H) bool; res = image pixel size.
    Returns (3,H,H) float = [cos2θ, sin2θ, coherence] in the UV frame, zeroed where not visible."""
    H = grid_ndc.shape[0]
    px = ((grid_ndc[..., 0] + 1) / 2 * res).astype(np.float32); py = ((grid_ndc[..., 1] + 1) / 2 * res).astype(np.float32)
    if texture_mode in ("gray", "highlight", "line"):
        if texture_mode == "highlight":
            g = fur_texture_highlight(img_rgb)
        elif texture_mode == "line":
            g = fur_texture_line_prior(img_rgb)
        else:
            g = cv2.cvtColor((np.clip(img_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        Jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma); Jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma); Jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
        sxx = cv2.remap(Jxx, px, py, cv2.INTER_LINEAR); syy = cv2.remap(Jyy, px, py, cv2.INTER_LINEAR); sxy = cv2.remap(Jxy, px, py, cv2.INTER_LINEAR)
        ang_img = 0.5 * np.arctan2(2 * sxy, sxx - syy) + np.pi / 2                            # fur flow ⟂ dominant gradient
        coh = np.hypot(sxx - syy, 2 * sxy) / (sxx + syy + 1e-9)
    elif texture_mode == "gabor":
        g = fur_texture_highlight(img_rgb)
        nori = 16
        phis = np.linspace(0, np.pi, nori, endpoint=False).astype(np.float32)                  # fur-line direction
        resp = []
        for phi in phis:
            # OpenCV Gabor theta is the stripe normal; fur line direction = theta - pi/2.
            kern = cv2.getGaborKernel((25, 25), 4.0, float(phi + np.pi / 2), 9.0, 0.35, 0, ktype=cv2.CV_32F)
            kern = kern - kern.mean()
            r = np.abs(cv2.filter2D(g, cv2.CV_32F, kern))
            resp.append(cv2.GaussianBlur(r, (0, 0), 1.0))
        R = np.stack(resp, 0)
        R = R / (R.reshape(nori, -1).mean(1)[:, None, None] + 1e-6)                            # orientation gain equalization
        W = np.maximum(R - R.mean(0, keepdims=True), 0.0) ** 2
        wsum = W.sum(0) + 1e-9
        c2_img = (W * np.cos(2 * phis)[:, None, None]).sum(0) / wsum
        s2_img = (W * np.sin(2 * phis)[:, None, None]).sum(0) / wsum
        mag = np.sqrt(c2_img * c2_img + s2_img * s2_img)
        strength = W.sum(0)
        scale = np.percentile(strength, 90.0) + 1e-9
        coh_img = np.clip(mag * strength / scale, 0.0, 1.0).astype(np.float32)
        c2 = cv2.remap(c2_img.astype(np.float32), px, py, cv2.INTER_LINEAR)
        s2 = cv2.remap(s2_img.astype(np.float32), px, py, cv2.INTER_LINEAR)
        coh = cv2.remap(coh_img, px, py, cv2.INTER_LINEAR)
        ang_img = 0.5 * np.arctan2(s2, c2)
    else:
        raise ValueError(f"unknown texture_mode {texture_mode!r}")
    ex, ey = np.cos(ang_img), np.sin(ang_img)                                                 # image-space flow unit vector
    # grid Jacobian d(px,py)/d(u,v): u along axis 1, v along axis 0 (shell_grids convention); ×H -> per unit uv
    dpx_du = np.gradient(px, axis=1) * H; dpx_dv = np.gradient(px, axis=0) * H
    dpy_du = np.gradient(py, axis=1) * H; dpy_dv = np.gradient(py, axis=0) * H
    det = dpx_du * dpy_dv - dpx_dv * dpy_du                                                    # solve J·d_uv = e_img
    det = np.where(np.abs(det) < 1e-9, 1e-9, det)
    duv_x = (dpy_dv * ex - dpx_dv * ey) / det
    duv_y = (-dpy_du * ex + dpx_du * ey) / det
    ang_uv = np.arctan2(duv_y, duv_x)
    out = np.stack([np.cos(2 * ang_uv), np.sin(2 * ang_uv), np.clip(coh, 0, 1)], 0).astype(np.float32)
    out[:, ~inf] = 0.0
    return out


def shell0_grid(meta, H):
    """UV->image NDC grid + visibility for the scalp shell, from a cameras.json view-0 (same projection RGBPROJ uses)."""
    size = float(meta["size"]); cam = meta["views"][0]; res = cam["res"]
    Minv = np.linalg.inv(np.array(cam["matrix_world"], np.float64)); fpx = cam["lens"] / cam["sensor_width"] * res
    cx = float(cam.get("cx", res / 2)); cy = float(cam.get("cy", res / 2))
    cm = meta.get("context_mosaic")
    if cm:
        grid_n = float(cm["grid"])
        xlo = float(cm["target_col"]) * res / grid_n; xhi = (float(cm["target_col"]) + 1.0) * res / grid_n
        ylo = float(cm["target_row"]) * res / grid_n; yhi = (float(cm["target_row"]) + 1.0) * res / grid_n
    else:
        xlo, xhi, ylo, yhi = 0.0, float(res), 0.0, float(res)
    ii, jj = np.meshgrid(np.arange(H), np.arange(H)); ug = (ii + 0.5) / H; vg = (jj + 0.5) / H
    Xw = np.stack([(ug - .5) * size, (vg - .5) * size, np.zeros_like(ug), np.ones_like(ug)], -1)
    Xc = (Xw.reshape(-1, 4) @ Minv.T)[:, :3].reshape(H, H, 3); z = Xc[..., 2]; fr = z < 0; zz = np.where(fr, z, -1e-6)
    px = cx + fpx * (Xc[..., 0] / -zz); py = cy - fpx * (Xc[..., 1] / -zz)
    gx = (px / res) * 2 - 1; gy = (py / res) * 2 - 1
    return np.stack([gx, gy], -1).astype(np.float32), (fr & (px >= xlo) & (px < xhi) & (py >= ylo) & (py < yhi))


if __name__ == "__main__":                                                                   # self-test: extracted orientation vs GT direction
    import sys, json
    H = 64
    for d in sys.argv[1:]:
        meta = json.load(open(f"{d}/cameras.json")); grid, inf = shell0_grid(meta, H)
        im = cv2.cvtColor(cv2.imread(f"{d}/rgb_0.png", -1)[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        f = extract_dir_field(im, grid, inf, meta["views"][0]["res"])
        c2, s2, coh = f[0][inf], f[1][inf], f[2][inf]
        w = coh; dom = 0.5 * np.degrees(np.arctan2((s2 * w).sum(), (c2 * w).sum()))           # coherence-weighted dominant axial angle
        p = np.load(f"{d}/patch.npz", allow_pickle=True); S = p["strands"].astype(np.float32)
        v = S[:, -1, :2] - S[:, 0, :2]; ln = np.linalg.norm(v, axis=1); a = np.arctan2(v[ln > 1e-6][:, 1], v[ln > 1e-6][:, 0])
        gt = np.degrees(0.5 * np.arctan2(np.sin(2 * a).mean(), np.cos(2 * a).mean()))
        dif = abs(((dom - gt + 90) % 180) - 90)
        print(f"{d.split('/')[-1]:22s} | extracted dir {dom:+5.0f} (mean coh {coh.mean():.2f}) | GT {gt:+5.0f} | |Δ|={dif:3.0f}deg", flush=True)
