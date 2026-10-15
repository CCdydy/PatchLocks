"""v10 inference: PIXEL+DINO conditioned diffusion -> z64 geometry + canonical root/tip COLOR (70ch), single image.
Loads stage3_v10_pixel/stage3_model.pt. Conditions on raw RGB back-projection (RGBPROJ) + B3 DINO + rho;
samples the 70ch target, splits z64(64)/color(6), decodes strands from z64 and paints each strand with the
GENERATED canonical color sampled at its root_uv. This is the consumer that puts v10's learned color+geometry
into the pipeline (vs infer_v9lite which used field-direction + fixed morphology + photo-projected color).
Run: python patchlocks/inference/infer_v10_pixel.py --geo_dir patchlocks/datasets/stage3_v10_pixel --patch_dir <dir> --out <dir>
"""
import os, sys, glob, json, argparse
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "patchlocks"); sys.path.insert(0, "patchlocks/data_gen")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
from models.strand_codec import StrandCodec
from field_to_strands import _bilinear
torch.set_grad_enabled(False); dev = "cuda"
ap = argparse.ArgumentParser()
ap.add_argument("--geo_dir", required=True); ap.add_argument("--patch_dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--model", default="", help="stage3_model*.pt path; default is <geo_dir>/stage3_model.pt")
ap.add_argument("--density", type=float, default=4.0, help="uniform strands per texel when --rho_source=uniform")
ap.add_argument("--rho_source", choices=["uniform", "field_gt"], default="uniform", help="field_gt is for synthetic visual QA; real deployment should use a predicted/solved rho map")
ap.add_argument("--rho_npz", default="", help="optional field_gt_<H>.npz path used when --rho_source=field_gt")
ap.add_argument("--res", type=int, default=64); ap.add_argument("--dino_res", type=int, default=448); ap.add_argument("--shells", type=int, default=5)
ap.add_argument("--steps", type=int, default=40); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--codec_path", default="")
ap.add_argument("--rotate_from_dir", choices=["none", "measured", "dir_head", "field_gt"], default="none",
                help="opt-in direction decoupling: strip each decoded strand's own XY direction, then rotate to an external UV direction field")
ap.add_argument("--dir_texture_mode", choices=["gray", "highlight", "line", "gabor"], default="highlight",
                help="texture source for measured/image direction extraction")
ap.add_argument("--dir_head", default="", help="TinyDirUNet checkpoint when --rotate_from_dir=dir_head")
ap.add_argument("--dir_shell", type=int, default=0, help="shell index used by measured/dir_head UV direction extraction; -1 = coherence-weighted all-shell fusion")
ap.add_argument("--dir_conf_min", type=float, default=0.15, help="below this external-direction confidence, keep the original V10 strand direction")
ap.add_argument("--dir_sign", choices=["align_v10", "as_is"], default="align_v10",
                help="axial directions have no arrow; align_v10 flips the external axis to the nearest original V10 root-tip sign")
ap.add_argument("--color_source", choices=["generated", "photo", "blend"], default="generated",
                help="generated: V10 color map; photo: RGBPROJ root color; blend: mix generated color with photo color")
ap.add_argument("--photo_color_blend", type=float, default=0.65, help="photo weight when --color_source=blend (luma weight if --color_chroma_blend is set)")
ap.add_argument("--color_chroma_blend", type=float, default=-1.0,
                help="color improvement #1: when >=0 and --color_source=blend, blend luma and chroma SEPARATELY -- photo_color_blend is the luma weight, this is the (usually higher) chroma weight. Takes the input photo's saturation/marking while keeping the generated lit brightness; fixes the desaturation/marking-washout. <0 = legacy uniform blend. Recommended: --photo_color_blend 0.4 --color_chroma_blend 0.85.")
a = ap.parse_args(); H = a.res; S = a.shells; C = 1024; GLOBC = 2 * C; Gdim = 256; os.makedirs(a.out, exist_ok=True)


def blend_luma_chroma(gen, photo, w_luma, w_chroma):
    """Blend gen->photo with separate luma/chroma weights. w_chroma>w_luma keeps the
    generated (lit) brightness while taking the input photo's saturation/marking (color improvement #1)."""
    w = np.array([0.299, 0.587, 0.114], np.float32)
    yg = (gen * w).sum(-1, keepdims=True); yp = (photo * w).sum(-1, keepdims=True)
    yo = (1.0 - w_luma) * yg + w_luma * yp                                    # luma: low weight -> keep generated brightness
    co = (1.0 - w_chroma) * (gen - yg) + w_chroma * (photo - yp)              # chroma: high weight -> take photo saturation/marking
    return np.clip(yo + co, 0.0, 1.0)
model_path = a.model or f"{a.geo_dir}/stage3_model.pt"
GEO = torch.load(model_path); B3 = torch.load(f"{a.geo_dir}/b3_fuser.pt"); GA = GEO["args"]; dim = GA["dim"]
XD = GEO.get("xd", 70); SD = 1.0; smin, smax = GA["sigma_min"], GA["sigma_max"]
CODEC_PATH = a.codec_path or GA.get("codec_path", "patchlocks/models/fur_strand_codec_v6upright.pt")
xm = GEO["xm"].to(dev); xs = GEO["xs"].to(dev); gm = GEO["cond_mean"].to(dev); gs = GEO["cond_std"].to(dev); glm = GEO["glob_mean"].to(dev); gls = GEO["glob_std"].to(dev)
print(f"v10 | model={model_path} | XD={XD} (z64+color) | rho_source={a.rho_source} density={a.density} | codec={CODEC_PATH}", flush=True)
d = a.patch_dir; meta = json.load(open(f"{d}/cameras.json")); size = float(meta["size"]); K = len(meta["views"])
ii, jj = np.meshgrid(np.arange(H), np.arange(H)); ug = (ii + 0.5) / H; vg = (jj + 0.5) / H

rho_map = np.full((H, H), float(a.density), np.float32)
radius_map = np.full((H, H), 0.0013, np.float32)
if a.rho_source == "field_gt":
    rho_path = a.rho_npz or f"{d}/field_gt_{H}.npz"
    fg = np.load(rho_path)
    rho_map = fg["lambda_map"].astype(np.float32)
    if "radius_mu" in fg.files:
        rm = fg["radius_mu"].astype(np.float32)
        radius_map = rm[..., 0] if rm.ndim == 3 else rm

def shell_grids(meta, hmax):
    hs = np.linspace(0, 1, S) * hmax; G = np.zeros((K, S, H, H, 2), np.float32); I = np.zeros((K, S, H, H), np.float32)
    for k in range(K):
        cam = meta["views"][k]; res = cam["res"]; Minv = np.linalg.inv(np.array(cam["matrix_world"], np.float64)); fpx = cam["lens"] / cam["sensor_width"] * res
        for si, h in enumerate(hs):
            Xw = np.stack([(ug - .5) * size, (vg - .5) * size, np.full_like(ug, h), np.ones_like(ug)], -1)
            Xc = (Xw.reshape(-1, 4) @ Minv.T)[:, :3].reshape(H, H, 3); z = Xc[..., 2]; fr = z < 0; zz = np.where(fr, z, -1e-6)
            px = res / 2 + fpx * (Xc[..., 0] / -zz); py = res / 2 - fpx * (Xc[..., 1] / -zz)
            G[k, si, ..., 0] = (px / res) * 2 - 1; G[k, si, ..., 1] = (py / res) * 2 - 1
            I[k, si] = (fr & (px >= 0) & (px < res) & (py >= 0) & (py < res)).astype(np.float32)
    return G, I

dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg').to(dev).eval()
mn = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]; st = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]; Hf = a.dino_res // 14
FMAP = torch.zeros(1, K, C, Hf, Hf, dtype=torch.float16, device=dev); GRID = torch.zeros(1, K, S, H, H, 2, dtype=torch.float16, device=dev)
INF = torch.zeros(1, K, S, H, H, dtype=torch.float16, device=dev); CAM = torch.zeros(1, K, 6, device=dev); GLOB = torch.zeros(1, GLOBC, device=dev)
ims = [cv2.resize(cv2.cvtColor(cv2.imread(f"{d}/rgb_{k}.png", -1), cv2.COLOR_BGR2RGB), (a.dino_res, a.dino_res)).astype(np.float32) / 255 for k in range(K)]
im = torch.tensor(np.stack(ims)).permute(0, 3, 1, 2).to(dev); feat = dino.forward_features((im - mn) / st)
FMAP[0] = feat["x_norm_patchtokens"].transpose(1, 2).reshape(K, C, Hf, Hf).half()
GLOB[0] = torch.cat([feat["x_norm_patchtokens"].mean((0, 1)), feat["x_norm_clstoken"].mean(0)], 0)
hmax = max(0.25 * size, 1e-3); Gg, I = shell_grids(meta, hmax); GRID[0] = torch.tensor(Gg, device=dev).half(); INF[0] = torch.tensor(I, device=dev).half()
g0 = GRID[0][:, 0].float(); samp = F.grid_sample(im.float(), g0, align_corners=False); inf0 = INF[0][:, 0].float()[:, None]
RGBRAW = ((samp * inf0).sum(0) / (inf0.sum(0) + 1e-6))[None]                                  # (1,3,H,H) raw RGB at scalp shell
RGBPROJ = (RGBRAW - 0.5) / 0.25                                                               # same centering as training
loc = [np.array(v["location"], np.float32) for v in meta["views"]]; vd = [-np.array(v["matrix_world"], np.float32)[:3, 2] for v in meta["views"]]
CAM[0] = torch.tensor(np.concatenate([np.stack(loc), np.stack(vd)], 1), device=dev); del dino; torch.cuda.empty_cache()

class B3Fuser(nn.Module):
    def __init__(s, dim, nsh, nb=6):
        super().__init__(); s.nsh = nsh; s.proj = nn.Conv2d(C, dim, 1)
        s.vemb = nn.Parameter(torch.randn(K, dim) * 0.02); s.semb = nn.Parameter(torch.randn(nsh, dim) * 0.02)
        s.cam = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim)); s.attn = nn.Conv2d(dim, 1, 1)
        s.body = nn.ModuleList([nn.Sequential(nn.GroupNorm(8, dim), nn.SiLU(), nn.Conv2d(dim, dim, 3, 1, 1)) for _ in range(nb)])
    def forward(s, bi):
        B = len(bi); fmap = FMAP[bi].reshape(B * K, C, Hf, Hf).float(); grid = GRID[bi][:, :, :s.nsh].reshape(B * K, s.nsh * H, H, 2).float()
        sm = F.grid_sample(fmap, grid, align_corners=False).reshape(B, K, C, s.nsh, H, H); inf = INF[bi][:, :, :s.nsh].float()
        x = s.proj(sm.permute(0, 1, 3, 2, 4, 5).reshape(B * K * s.nsh, C, H, H)).reshape(B, K, s.nsh, -1, H, H)
        ce = s.cam(CAM[bi].reshape(B * K, 6)).reshape(B, K, 1, -1, 1, 1)
        x = x + s.vemb[None, :, None, :, None, None] + s.semb[None, None, :, :, None, None] + ce
        x = x.reshape(B, K * s.nsh, -1, H, H); mm = inf.reshape(B, K * s.nsh, 1, H, H)
        w = torch.softmax(s.attn(x.reshape(B * K * s.nsh, -1, H, H)).reshape(B, K * s.nsh, 1, H, H).masked_fill(mm < 0.5, -1e4), 1) * (mm.sum(1, keepdim=True) > 0)
        fused = (x * w).sum(1)
        for b in s.body: fused = fused + b(fused)
        return fused
class Res(nn.Module):
    def __init__(s, ci, co, ed):
        super().__init__(); s.n1 = nn.GroupNorm(8, ci); s.c1 = nn.Conv2d(ci, co, 3, 1, 1); s.n2 = nn.GroupNorm(8, co); s.c2 = nn.Conv2d(co, co, 3, 1, 1)
        s.emb = nn.Linear(ed, 2 * co); s.sk = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()
    def forward(s, x, e):
        h = s.c1(F.silu(s.n1(x))); sh = s.emb(e)[:, :, None, None]; sc, bi = sh.chunk(2, 1); h = s.c2(F.silu(s.n2(h) * (1 + sc) + bi)); return s.sk(x) + h
class UNet(nn.Module):
    def __init__(s, dim, ed=256, cond_ch=None):
        super().__init__(); cond_ch = cond_ch or C
        s.gp = nn.Conv2d(cond_ch, Gdim, 1); s.gp2 = nn.Conv2d(cond_ch, dim, 1)
        uv = torch.tensor(np.stack([ug * 2 - 1, vg * 2 - 1], 0), dtype=torch.float32); s.register_buffer("uv", uv[None])
        s.inp = nn.Conv2d(XD + 1 + 3 + Gdim + 2, dim, 3, 1, 1)
        s.semb = nn.Sequential(nn.Linear(1, ed), nn.SiLU(), nn.Linear(ed, ed)); s.gemb = nn.Linear(GLOBC, ed)
        s.d1 = Res(dim, dim, ed); s.d2 = Res(dim, 2 * dim, ed); s.mid = Res(2 * dim + dim, 2 * dim, ed)
        s.u2 = Res(4 * dim, dim, ed); s.u1 = Res(2 * dim, dim, ed); s.out = nn.Conv2d(dim, XD, 3, 1, 1)
        s.dn = nn.AvgPool2d(2); s.up = nn.Upsample(scale_factor=2, mode="nearest")
    def forward(s, x, cnoise, gc, gl, rhoc, rgbp):
        gcf = gc.float(); e = s.semb(cnoise[:, None]) + s.gemb(gl.float())
        h0 = s.inp(torch.cat([x, rhoc, rgbp.float(), s.gp(gcf), s.uv.expand(len(x), -1, -1, -1)], 1)); a1 = s.d1(h0, e); a2 = s.d2(s.dn(a1), e)
        gmid = s.gp2(F.avg_pool2d(gcf, 4)); mm = s.mid(torch.cat([s.dn(a2), gmid], 1), e)
        b2 = s.u2(torch.cat([s.up(mm), a2], 1), e); b1 = s.u1(torch.cat([s.up(b2), a1], 1), e); return s.out(b1)

b3 = B3Fuser(dim, S).to(dev).eval(); b3.load_state_dict(B3["b3"]); gnet = UNet(160, cond_ch=dim).to(dev).eval(); gnet.load_state_dict(GEO["net"])
ck = torch.load(CODEC_PATH, weights_only=False); cnorm = {k: torch.tensor(v, device=dev, dtype=torch.float32) for k, v in ck["fur_norm"].items()}
codec = StrandCodec(do_vae=False, decode_type="dir", scale_init=30.0, nr_verts_per_strand=256, nr_values_to_decode=255, dim_per_value_decoded=3).cuda().eval(); codec.load_state_dict(ck["state_dict"])
G_COND = ((b3(torch.arange(1, device=dev)) - gm) / gs).half(); GLOBn = (GLOB - glm) / gls
RHOC = torch.log1p(torch.tensor(rho_map[None, None], device=dev, dtype=torch.float32))        # clean rho condition; GT rho only for synthetic QA

def precond(x, sig):
    sig = sig.reshape(-1, 1, 1, 1); cs = SD**2 / (sig**2 + SD**2); co = sig * SD / (sig**2 + SD**2).sqrt(); ci = 1 / (sig**2 + SD**2).sqrt(); cn = sig.flatten().log() / 4
    return cs * x + co * gnet(ci * x, cn, G_COND, GLOBn, RHOC, RGBPROJ)
sigs = torch.linspace(0, 1, a.steps + 1, device=dev); sigs = (smax**(1 / 7) + sigs * (smin**(1 / 7) - smax**(1 / 7)))**7; sigs = torch.cat([sigs, torch.zeros(1, device=dev)])
gen = torch.Generator(device=dev).manual_seed(a.seed); x = torch.randn(1, XD, H, H, device=dev, generator=gen) * sigs[0]
for j in range(a.steps):
    s0, s1 = sigs[j], sigs[j + 1]; d0 = (x - precond(x, s0.expand(1))) / s0; xn = x + (s1 - s0) * d0
    if s1 > 0: d2 = (xn - precond(xn, s1.expand(1))) / s1; xn = x + (s1 - s0) * 0.5 * (d0 + d2)
    x = xn
out70 = (x * xs + xm)[0].permute(1, 2, 0).cpu().numpy()                                      # (H,H,XD)
Tz = out70[:, :, :64]; CMAP = np.clip(out70[:, :, 64:], 0, 1) if XD > 64 else None           # z64 geometry [+ color map if present]

def direction_from_input():
    """Return an oriented UV direction map (H,H,2) + confidence (H,H).

    This is a consumer-side module: V10 still provides shape/color, while the
    absolute groom direction can come from image texture or a small direction
    head.  Low-confidence texels are handled later by falling back to V10's
    original decoded direction per root.
    """
    if a.rotate_from_dir == "field_gt":
        if "fg" not in globals() or "flow2d" not in fg.files:
            raise RuntimeError("--rotate_from_dir=field_gt requires --rho_source field_gt and field_gt flow2d")
        v = fg["flow2d"].astype(np.float32)
        if v.shape[:2] != (H, H):
            raise RuntimeError(f"field_gt flow2d has shape {v.shape}, expected ({H},{H},2)")
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.maximum(n, 1e-6), (n[..., 0] > 1e-6).astype(np.float32)

    acc = np.zeros((H, H, 2), np.float32)
    wsum = np.zeros((H, H), np.float32)
    dir_shells = range(S) if int(a.dir_shell) < 0 else [int(np.clip(a.dir_shell, 0, S - 1))]
    source_count = 0
    if a.rotate_from_dir == "dir_head":
        if not a.dir_head:
            raise RuntimeError("--rotate_from_dir=dir_head needs --dir_head <checkpoint>")
        from models.dir_head_light import load_dir_head, predict_dir_head_image
        net_dir = load_dir_head(a.dir_head, dev)
        for k in range(K):
            for shell in dir_shells:
                axial, conf = predict_dir_head_image(
                    net_dir, ims[k], Gg[k, shell], I[k, shell].astype(bool),
                    int(meta["views"][k]["res"]), texture_mode=a.dir_texture_mode, device=dev
                )
                acc += axial.astype(np.float32) * conf[..., None].astype(np.float32)
                wsum += conf.astype(np.float32)
                source_count += 1
    elif a.rotate_from_dir == "measured":
        from dir_extractor import extract_dir_field
        for k in range(K):
            for shell in dir_shells:
                df = extract_dir_field(
                    ims[k], Gg[k, shell], I[k, shell].astype(bool),
                    int(meta["views"][k]["res"]), texture_mode=a.dir_texture_mode
                )
                conf = df[2].astype(np.float32)
                acc += np.moveaxis(df[:2].astype(np.float32), 0, -1) * conf[..., None]
                wsum += conf
                source_count += 1
    else:
        return None, None

    axial = acc / np.maximum(wsum[..., None], 1e-6)
    from dir_extractor import resolve_axial_sign
    conf_norm = np.clip(wsum / max(source_count, 1), 0.0, 1.0)
    vec = resolve_axial_sign(axial, conf=conf_norm, valid=wsum > 1e-6)
    n = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.maximum(n, 1e-6), conf_norm.astype(np.float32)

DIRMAP, DIRCONF = direction_from_input()

# ---- roots -> decode bilinear z64 -> strands; paint GENERATED color at root_uv ----
rng = np.random.default_rng(a.seed)
lam = np.clip(rho_map, 0, None)
base = np.floor(lam).astype(np.int32)
extra = (rng.uniform(0, 1, lam.shape) < (lam - base)).astype(np.int32)
nc = (base + extra).reshape(-1)
cells = np.repeat(np.nonzero(nc)[0], nc[nc > 0])
if len(cells) == 0:
    raise RuntimeError("rho map sampled zero roots")
cj, ci = np.divmod(cells, H)
uv = np.stack([(ci + rng.uniform(0, 1, len(cells))) / H, (cj + rng.uniform(0, 1, len(cells))) / H], 1).astype(np.float32)
M = len(uv)
roots = np.stack([(uv[:, 0] - 0.5) * size, (uv[:, 1] - 0.5) * size, np.zeros(M)], 1).astype(np.float32)
z = torch.tensor(_bilinear(Tz, uv), device=dev, dtype=torch.float32)
pr = codec.decoder(z, None, cnorm); rec = (pr["strand_positions"] if "strand_positions" in pr else pr[[k for k in pr if "pos" in k.lower()][0]]).cpu().numpy()
L = 32; outp = np.zeros((M, L, 3), np.float32)
for n in range(M):
    sgt = rec[n] - rec[n][:1]; seg = np.linalg.norm(np.diff(sgt, axis=0), axis=1); cl = np.concatenate([[0], np.cumsum(seg)]); tt = np.linspace(0, cl[-1] + 1e-9, L)
    outp[n] = np.stack([np.interp(tt, cl, sgt[:, c]) for c in range(3)], 1)
if DIRMAP is not None:
    rel = outp - outp[:, :1]
    od = rel[:, -1, :2] - rel[:, 0, :2]
    on = np.linalg.norm(od, axis=1)
    oang = np.zeros(M, np.float32)
    good_orig = on > 1e-7
    oang[good_orig] = np.arctan2(od[good_orig, 1], od[good_orig, 0])
    c0, s0 = np.cos(-oang), np.sin(-oang)
    x0, y0 = rel[..., 0].copy(), rel[..., 1].copy()
    rel[..., 0] = c0[:, None] * x0 - s0[:, None] * y0
    rel[..., 1] = s0[:, None] * x0 + c0[:, None] * y0

    dv = _bilinear(DIRMAP, uv).astype(np.float32)
    dc = _bilinear(DIRCONF[..., None], uv)[:, 0].astype(np.float32)
    dn = np.linalg.norm(dv, axis=1)
    valid = dn > 1e-6
    dv[valid] /= dn[valid, None]
    if a.dir_sign == "align_v10":
        flip = (dv * od).sum(1) < 0
        dv[flip] *= -1
    tang = np.stack([np.cos(oang), np.sin(oang)], 1).astype(np.float32)
    use_external = valid & (dc >= float(a.dir_conf_min))
    tv = np.where(use_external[:, None], dv, tang)
    tangang = np.arctan2(tv[:, 1], tv[:, 0])
    c1, s1 = np.cos(tangang), np.sin(tangang)
    x1, y1 = rel[..., 0].copy(), rel[..., 1].copy()
    rel[..., 0] = c1[:, None] * x1 - s1[:, None] * y1
    rel[..., 1] = s1[:, None] * x1 + c1[:, None] * y1
    outp = rel + outp[:, :1]
outp = outp - outp[:, :1] + roots[:, None]; outp[..., 2] = np.maximum(outp[..., 2], 0); outp[:, 0] = roots
photo_map = RGBRAW[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
if CMAP is not None:
    col = _bilinear(CMAP[:, :, :3], uv).astype(np.float32); ctip = _bilinear(CMAP[:, :, 3:], uv).astype(np.float32)   # GENERATED canonical color
else:                                                                                        # no-color model (--no_color): flat gray for geometry/direction QA
    col = np.full((M, 3), 0.45, np.float32); ctip = np.full((M, 3), 0.58, np.float32)
photo = _bilinear(photo_map, uv).astype(np.float32)
if a.color_source == "photo":
    col = photo
    ctip = photo
elif a.color_source == "blend":
    wb = float(np.clip(a.photo_color_blend, 0.0, 1.0))
    if a.color_chroma_blend < 0:                                              # legacy uniform blend
        col = np.clip((1.0 - wb) * col + wb * photo, 0.0, 1.0)
        ctip = np.clip((1.0 - wb) * ctip + wb * photo, 0.0, 1.0)
    else:                                                                    # color improvement #1: chroma-emphasized blend
        wc = float(np.clip(a.color_chroma_blend, 0.0, 1.0))
        col = blend_luma_chroma(col, photo, wb, wc)
        ctip = blend_luma_chroma(ctip, photo, wb, wc)
rad = _bilinear(radius_map[..., None], uv)[:, 0].astype(np.float32)
np.savez(
    f"{a.out}/patch.npz",
    strands=outp.astype(np.float32),
    root_uv=uv,
    color=col,
    color_tip=ctip,
    radius=rad,
    size=np.float32(size),
    z64_map=Tz.astype(np.float32),
    color_map=(CMAP.astype(np.float32) if CMAP is not None else np.zeros((H, H, 6), np.float32)),
    photo_color_map=photo_map.astype(np.float32),
    rho_map=rho_map.astype(np.float32),
    direction_vec_map=(DIRMAP.astype(np.float32) if DIRMAP is not None else np.zeros((H, H, 2), np.float32)),
    direction_conf_map=(DIRCONF.astype(np.float32) if DIRCONF is not None else np.zeros((H, H), np.float32)),
    rotate_from_dir=np.array(a.rotate_from_dir),
    color_source=np.array(a.color_source),
)
arc = np.linalg.norm(np.diff(outp, axis=1), axis=-1).sum(1)
print(f"v10 wrote {M} strands -> {a.out}/patch.npz | arc={np.median(arc)/size:.2f}x | dir={a.rotate_from_dir} color={a.color_source} | color root {col.mean(0).round(3)} tip {ctip.mean(0).round(3)}", flush=True)
