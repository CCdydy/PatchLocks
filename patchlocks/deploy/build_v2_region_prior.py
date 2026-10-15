"""V2 mainline foundation block #1: the DENSITY/coverage REGION-PRIOR aggregation layer.

Per-patch diffusion coverage is real (cross-patch corr 0.81) but must NOT be consumed as a raw per-texel rho_map
(within-patch corr ~0, magnitude compressed). The stable thing to consume is a region-aggregated coverage:

    diffusion per-patch coverage  ->  GLOBAL calibration  ->  REGION robust average  ->  patch residual (clipped)

so the whole animal can't go blotchy (one patch dense, the neighbor sparse). This module builds the v2_region_prior
data contract and ships a 3-tier acceptance gate (field_gt density | pred calibrated | pred calibrated + region avg).
If tier-3 doesn't lose accuracy and is MORE stable than tier-2, "patch predict, region consume" is validated.

Consumption rules baked in here:
  z64        -> patch-local (diffusion direct)        [pass-through; full map preferred]
  color      -> patch-local + region calibration       [region_color_root_tip provided]
  density    -> NEVER raw per-patch: global calib -> region robust avg -> small clipped residual   [THIS module]
  length     -> predicted patch proposal -> region robust avg -> small clipped ratio residual
  radius/mat -> next block, global/large-region robust average (placeholders here)
  direction  -> independent measured or diffusion-predicted, stored as axial [cos 2theta, sin 2theta]   [pass-through]

Run (validation on collected per-patch density, region = coat proxy):
  python patchlocks/deploy/build_v2_region_prior.py --table /tmp/dens_data.txt --out /tmp/v2_region_prior.npz

Run (whole-surface contract/region scaffold, before batch diffusion proposals exist):
  python patchlocks/deploy/build_v2_region_prior.py \
    --pack patchlocks/deploy/neuralfur_beagle_official/realviews/surfpack_beagle_f210_geopatch/surface_patch_dataset.npz \
    --out /tmp/v2_region_prior.npz

Run (whole-surface density gate, table rows can be either
     "patch_id pred_density gt_density" or legacy "region patch_id pred_density gt_density"):
  python patchlocks/deploy/build_v2_region_prior.py \
    --pack <surface_patch_dataset.npz> --table <per_patch_density.txt> --out <v2_region_prior.npz>
"""
import argparse
import os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--table", default="", help="per-patch rows: legacy 'region patch_id pred_density gt_density', or with --pack: 'patch_id pred_density [gt_density]'")
ap.add_argument("--proposal_npz", default="", help="optional v10_surfpack_proposals.npz; fills coverage, patch_z64, and patch_color into the prior")
ap.add_argument(
    "--drop_z64_map",
    action="store_true",
    help="do not duplicate full patch_z64_map into the prior; V2-A swatch mainline reads it directly from proposals",
)
ap.add_argument(
    "--drop_heavy_maps",
    action="store_true",
    help="do not duplicate per-texel maps such as patch_z64_map or patch_dir_vec_map into the prior",
)
ap.add_argument("--pack", default="", help="optional surface_patch_dataset.npz; enables whole-dog patch ids, face/bary, spatial regions, and patch confidences")
ap.add_argument("--patch_params", default="", help="optional patch_params.npz/patch_region npz with patch_region labels")
ap.add_argument("--region_npz", default="", help="optional npz with patch_region labels")
ap.add_argument("--gt_patch_params", default="", help="optional patch_params.npz providing per-patch GT/target density for calibration/gate")
ap.add_argument("--gt_key", default="rho", help="key inside --gt_patch_params used as GT density")
ap.add_argument("--out", default="/tmp/v2_region_prior.npz")
ap.add_argument("--resid_frac", type=float, default=0.1, help="max |patch residual| as a fraction of region density (clip) -> anti-blotchy")
ap.add_argument("--length_resid_frac", type=float, default=0.25, help="max |patch length-ratio residual| as a fraction of region length ratio")
ap.add_argument("--color_resid_frac", type=float, default=0.1, help="COLOR local deviation budget on top of the (bilateral) anchor, as a fraction of the GLOBAL per-channel color spread. Small keeps the anchor's rainbow-smoothing; large undoes it back to raw per-patch. 0.1 ~= -25%% rainbow, markings ~0.93 kept.")
ap.add_argument("--color_mode", choices=["bilateral", "region"], default="bilateral", help="color anchor: bilateral (spatial+color-similarity local average, PRESERVES high-contrast markings like a white chest) vs region (coarse region median, lumps markings). bilateral needs --pack patch_pos.")
ap.add_argument("--color_sim_mult", type=float, default=1.5, help="bilateral color-similarity bandwidth as a multiple of global color std; smaller=sharper marking edges, larger=more rainbow smoothing. 1.5: -31%% neighbor rainbow, markings 0.89 kept")
ap.add_argument("--color_spatial_frac", type=float, default=0.06, help="bilateral spatial bandwidth as a fraction of the animal extent")
ap.add_argument("--color_source", choices=["rgbproj", "diffusion"], default="rgbproj", help="ROOT color albedo: rgbproj (DEFAULT) = REAL multi-view projected input RGB (reliable at dark/saturated markings); diffusion = the 6ch lit_color (drifts purple on dark fur, yellow on tan -> not lockable). rgbproj keeps the diffusion (tip-root) only as a bounded residual.")
ap.add_argument("--color_tip_resid", type=float, default=0.05, help="when --color_source rgbproj: max |diffusion (tip-root) residual| added to the real-albedo tip, per channel. Small -> tip variation without importing the diffusion's purple/yellow.")
ap.add_argument("--auto_region_level", choices=["coarse", "important", "fine"], default="important")
ap.add_argument("--auto_tail_end", choices=["auto", "min", "max"], default="auto")
ap.add_argument("--calib_scale", type=float, default=np.nan, help="deploy-time density scale; if GT is present and this is omitted, fit from table")
ap.add_argument("--calib_offset", type=float, default=np.nan, help="deploy-time density offset; if GT is present and this is omitted, fit from table")
ap.add_argument("--default_density", type=float, default=1.0, help="proposal density used when --table is absent; useful for region-contract smoke")
# GLOBAL material hook (user rule: fixed defaults, NOT per-patch, NOT from diffusion, NOT heuristically estimated from radius/coverage). Consumed by the PBR Hair-BSDF render (render_dog_fur_pro). Mirrors its arg defaults.
ap.add_argument("--mat_roughness", type=float, default=0.32); ap.add_argument("--mat_radial_rough", type=float, default=0.35)
ap.add_argument("--mat_coat", type=float, default=0.08); ap.add_argument("--mat_spec", type=float, default=1.0); ap.add_argument("--mat_hair_ior", type=float, default=1.55)
ap.add_argument("--max_patches", type=int, default=0, help="debug subset when --pack is used; matches infer_v10_surfpack_proposals --max_patches")
ap.add_argument("--viz", default="", help="optional region scatter png")
a = ap.parse_args()

def detect_surface_regions(pos, auto_tail_end="auto", region_level="important"):
    """Geometry fallback matching patch_graph_opt_geo.detect_patch_regions.

    It is intentionally coarse and dog-generic: width = smallest extent,
    length = largest extent, vertical = remaining axis.
    """
    pos = pos.astype(np.float32)
    ext = pos.max(0) - pos.min(0)
    vertical_axis = int(np.argsort(ext)[1])
    length_axis = int(np.argmax(ext))
    width_axis = int([i for i in range(3) if i not in (vertical_axis, length_axis)][0])
    important = region_level in ("important", "fine")
    labels = np.full(len(pos), "other" if not important else "body", dtype="U24")

    y = pos[:, vertical_axis]
    lo = y < np.quantile(y, 0.06)
    hi = y > np.quantile(y, 0.94)

    def spread_score(mask):
        if not np.any(mask):
            return 0.0
        oth = [i for i in range(3) if i != vertical_axis]
        return float(
            len(pos[mask]) / max(len(pos), 1)
            + 0.25 * np.linalg.norm(np.ptp(pos[mask][:, oth], axis=0)) / max(np.linalg.norm(ext), 1e-6)
        )

    foot_end = "max" if spread_score(hi) > spread_score(lo) else "min"
    if foot_end == "max":
        paw = y > np.quantile(y, 0.86)
        leg = y > np.quantile(y, 0.70)
    else:
        paw = y < np.quantile(y, 0.14)
        leg = y < np.quantile(y, 0.30)
    labels[leg] = "legs"
    labels[paw] = "paws"

    z = pos[:, length_axis]
    tail_is_min = False
    if auto_tail_end == "min":
        tail = z < np.quantile(z, 0.14)
        tail_is_min = True
    elif auto_tail_end == "max":
        tail = z > np.quantile(z, 0.86)
    else:
        zlo = z < np.quantile(z, 0.14)
        zhi = z > np.quantile(z, 0.86)

        def end_score(mask):
            if not np.any(mask):
                return 0.0
            oth = [i for i in range(3) if i != length_axis]
            return float(
                len(pos[mask]) / max(len(pos), 1)
                + 0.35 * np.linalg.norm(np.ptp(pos[mask][:, oth], axis=0)) / max(np.linalg.norm(ext), 1e-6)
            )

        tail_is_min = end_score(zlo) >= end_score(zhi)
        tail = zlo if tail_is_min else zhi

    if important:
        q = lambda x: np.quantile(z, x)
        x = pos[:, width_axis]
        x0 = float(np.median(x))
        half_w = max(float(np.max(np.abs(x - x0))), 1e-6)
        lateral = np.abs(x - x0) > 0.50 * half_w
        not_low_limb = ~leg
        if tail_is_min:
            tail_base = (z >= q(0.14)) & (z < q(0.25))
            head = z > q(0.78)
            ears = (z > q(0.70)) & lateral & not_low_limb
        else:
            tail_base = (z <= q(0.86)) & (z > q(0.75))
            head = z < q(0.22)
            ears = (z < q(0.30)) & lateral & not_low_limb
        labels[tail_base & ~paw] = "tail_base"
        labels[head & not_low_limb] = "face"
        labels[ears & ~paw] = "ears"

    labels[tail] = "tail"
    labels[paw] = "paws"
    return labels.astype("U24")


def load_patch_region_npz(path, p_count):
    if not path:
        return None
    pp = np.load(path, allow_pickle=True)
    if "patch_region" not in pp.files:
        return None
    reg = pp["patch_region"].astype(str)
    if len(reg) == p_count:
        return reg.astype("U24")
    if a.max_patches > 0 and len(reg) > p_count:
        return reg[:p_count].astype("U24")
    if "used_patch" in pp.files and len(reg) == len(pp["used_patch"]):
        out = np.full(p_count, "body", dtype="U24")
        out[pp["used_patch"].astype(np.int64)] = reg.astype("U24")
        return out
    raise ValueError(f"{path}: patch_region shape {reg.shape} cannot map to P={p_count}")


def load_pack(path):
    if not path:
        return None
    d = np.load(path, allow_pickle=True)
    centers = d["centers"].astype(np.float32)
    if a.max_patches > 0:
        centers = centers[: a.max_patches]
    p_count = len(centers)
    region = load_patch_region_npz(a.region_npz, p_count)
    if region is None:
        region = load_patch_region_npz(a.patch_params, p_count)
    if region is None and "patch_region" in d.files and len(d["patch_region"]) >= p_count:
        region = d["patch_region"][:p_count].astype("U24")
    if region is None:
        region = detect_surface_regions(centers, a.auto_tail_end, a.auto_region_level)
        region_source = f"geometry:{a.auto_region_level}:tail_{a.auto_tail_end}"
    else:
        region_source = "provided"
    valid = d["valid"].astype(np.float32)[:p_count] if "valid" in d.files else np.ones((p_count, 1), np.float32)
    conf = np.clip(valid.reshape(p_count, -1).mean(1), 0, 1).astype(np.float32)
    patch_face = (d["patch_face"][:p_count] if "patch_face" in d.files else d.get("root_face", np.arange(p_count))[:p_count]).astype(np.int32)
    patch_bary = (d["patch_bary"][:p_count] if "patch_bary" in d.files else d.get("root_bary", np.zeros((len(d["centers"]), 3), np.float32))[:p_count]).astype(np.float32)
    return {
        "patch_id": np.arange(p_count).astype(str),
        "region": region.astype("U24"),
        "conf": conf,
        "patch_face": patch_face,
        "patch_bary": patch_bary,
        "patch_pos": centers.astype(np.float32),
        "region_source": region_source,
    }


def read_table(path):
    if not path:
        return []
    rows = []
    for line in open(path):
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("DONE"):
            continue
        t = s.split()
        if len(t) < 2:
            continue
        rows.append(t)
    return rows


pack = load_pack(a.pack)
rows = read_table(a.table)

if pack is None and not rows:
    raise SystemExit("need --table and/or --pack")

if pack is not None:
    pid = pack["patch_id"].copy()
    region = pack["region"].copy()
    patch_conf_base = pack["conf"].copy()
    pred = np.full(len(pid), float(a.default_density), np.float32)
    gt = np.full(len(pid), np.nan, np.float32)
    patch_z64 = np.zeros((len(pid), 0), np.float32)
    patch_z64_map = None
    patch_z64_map_path = ""
    patch_z64_map_shape = None
    patch_z64_map_dtype = ""
    patch_color_root_tip = np.zeros((len(pid), 0), np.float32)
    patch_dir = np.zeros((len(pid), 2), np.float32)                                            # chosen axial [cos2theta,sin2theta] per patch
    patch_dir_coh = np.zeros(len(pid), np.float32)
    patch_dir_measured = np.zeros((len(pid), 2), np.float32)
    patch_dir_measured_coh = np.zeros(len(pid), np.float32)
    patch_dir_diffusion = np.zeros((len(pid), 2), np.float32)
    patch_dir_diffusion_coh = np.zeros(len(pid), np.float32)
    patch_dir_vec_map = None
    patch_dir_vec_map_conf = None
    patch_dir_source = np.full(len(pid), "measured", dtype="U16")
    patch_radius = np.zeros(len(pid), np.float32)                                              # predicted radius proposal -> GLOBAL robust avg (never raw per-patch)
    patch_length_mean = np.zeros(len(pid), np.float32)                                         # predicted length proposal -> normalized region/patch ratio
    patch_texture_hf = np.zeros(len(pid), np.float32)
    patch_texture_conf = np.zeros(len(pid), np.float32)
    patch_native_px = np.zeros(len(pid), np.float32)
    patch_effective_px = np.zeros(len(pid), np.float32)
    patch_crop_px = np.zeros(len(pid), np.float32)
    patch_model_px = np.zeros(len(pid), np.float32)
    patch_upscale = np.ones(len(pid), np.float32)
    patch_scale_conf = np.zeros(len(pid), np.float32)
    patch_length_remap = np.ones(len(pid), np.float32)
    patch_selected_view = np.full(len(pid), -1, np.int32)
    patch_rgbproj = np.zeros((len(pid), 3), np.float32)                                         # REAL projected albedo (color source; diffusion color demoted to residual)
    if a.proposal_npz:
        prop = np.load(a.proposal_npz, allow_pickle=True)
        pids_prop = prop["patch_id"].astype(str) if "patch_id" in prop.files else np.arange(len(prop["patch_coverage_pred"])).astype(str)
        if "patch_z64_mean" in prop.files:
            patch_z64 = np.zeros((len(pid), prop["patch_z64_mean"].shape[1]), np.float32)
        keep_heavy_maps = not (a.drop_z64_map or a.drop_heavy_maps)
        if keep_heavy_maps and "patch_z64_map" in prop.files:
            zm = prop["patch_z64_map"]
            if zm.ndim != 4 or zm.shape[-1] != 64:
                raise ValueError(f"patch_z64_map must be (P,H,W,64), got {zm.shape}")
            patch_z64_map = np.zeros((len(pid),) + zm.shape[1:], np.float16)
        elif keep_heavy_maps and "patch_z64_map_path" in prop.files:
            patch_z64_map_path = str(np.asarray(prop["patch_z64_map_path"]).reshape(-1)[0])
            if "patch_z64_map_shape" in prop.files:
                patch_z64_map_shape = prop["patch_z64_map_shape"].astype(np.int32)
            if "patch_z64_map_dtype" in prop.files:
                patch_z64_map_dtype = str(np.asarray(prop["patch_z64_map_dtype"]).reshape(-1)[0])
        elif (a.drop_z64_map or a.drop_heavy_maps) and ("patch_z64_map" in prop.files or "patch_z64_map_path" in prop.files):
            print("Z64 SOURCE = dropped full map from region prior (swatch reads proposals directly)", flush=True)
        if "patch_color_root_tip" in prop.files:
            patch_color_root_tip = np.zeros((len(pid), prop["patch_color_root_tip"].shape[1]), np.float32)
        if keep_heavy_maps and "patch_dir_vec_map" in prop.files:
            dm = prop["patch_dir_vec_map"]
            if dm.ndim != 4 or dm.shape[-1] != 2:
                raise ValueError(f"patch_dir_vec_map must be (P,H,W,2), got {dm.shape}")
            patch_dir_vec_map = np.zeros((len(pid),) + dm.shape[1:], np.float16)
            patch_dir_vec_map_conf = np.zeros((len(pid),) + dm.shape[1:3], np.float16)
        elif a.drop_heavy_maps and "patch_dir_vec_map" in prop.files:
            print("DIR MAP SOURCE = dropped patch_dir_vec_map from region prior", flush=True)
        for ii, pp in enumerate(pids_prop):
            try:
                idx = int(pp)
            except ValueError:
                hit = np.nonzero(pid == pp)[0]
                if len(hit) == 0:
                    continue
                idx = int(hit[0])
            if 0 <= idx < len(pid):
                if "patch_coverage_pred" in prop.files:
                    pred[idx] = float(prop["patch_coverage_pred"][ii])
                if "patch_conf" in prop.files:
                    patch_conf_base[idx] = float(prop["patch_conf"][ii])
                if patch_z64.shape[1] and "patch_z64_mean" in prop.files:
                    patch_z64[idx] = prop["patch_z64_mean"][ii]
                if patch_z64_map is not None and "patch_z64_map" in prop.files:
                    patch_z64_map[idx] = zm[ii]                                  # zm preloaded; prop["..."][ii] re-decompresses the full 937MB map each iter
                if "patch_radius_mean" in prop.files:
                    patch_radius[idx] = float(prop["patch_radius_mean"][ii])
                if "patch_length_mean" in prop.files:
                    patch_length_mean[idx] = float(prop["patch_length_mean"][ii])
                if "patch_texture_hf" in prop.files:
                    patch_texture_hf[idx] = float(prop["patch_texture_hf"][ii])
                if "patch_texture_conf" in prop.files:
                    patch_texture_conf[idx] = float(prop["patch_texture_conf"][ii])
                if "patch_native_px" in prop.files:
                    patch_native_px[idx] = float(prop["patch_native_px"][ii])
                if "patch_effective_px" in prop.files:
                    patch_effective_px[idx] = float(prop["patch_effective_px"][ii])
                if "patch_crop_px" in prop.files:
                    patch_crop_px[idx] = float(prop["patch_crop_px"][ii])
                if "patch_model_px" in prop.files:
                    patch_model_px[idx] = float(prop["patch_model_px"][ii])
                if "patch_upscale" in prop.files:
                    patch_upscale[idx] = float(prop["patch_upscale"][ii])
                if "patch_scale_conf" in prop.files:
                    patch_scale_conf[idx] = float(prop["patch_scale_conf"][ii])
                if "patch_length_remap" in prop.files:
                    patch_length_remap[idx] = float(prop["patch_length_remap"][ii])
                if "patch_selected_view" in prop.files:
                    patch_selected_view[idx] = int(prop["patch_selected_view"][ii])
                if patch_color_root_tip.shape[1] and "patch_color_root_tip" in prop.files:
                    patch_color_root_tip[idx] = prop["patch_color_root_tip"][ii]
                if "patch_rgbproj_color" in prop.files:
                    patch_rgbproj[idx] = prop["patch_rgbproj_color"][ii]
                if "patch_dir" in prop.files:
                    patch_dir[idx] = prop["patch_dir"][ii]
                if "patch_dir_coh" in prop.files:
                    patch_dir_coh[idx] = float(prop["patch_dir_coh"][ii])
                if "patch_dir_measured" in prop.files:
                    patch_dir_measured[idx] = prop["patch_dir_measured"][ii]
                if "patch_dir_measured_coh" in prop.files:
                    patch_dir_measured_coh[idx] = float(prop["patch_dir_measured_coh"][ii])
                if "patch_dir_diffusion" in prop.files:
                    patch_dir_diffusion[idx] = prop["patch_dir_diffusion"][ii]
                if "patch_dir_diffusion_coh" in prop.files:
                    patch_dir_diffusion_coh[idx] = float(prop["patch_dir_diffusion_coh"][ii])
                if patch_dir_vec_map is not None and "patch_dir_vec_map" in prop.files:
                    patch_dir_vec_map[idx] = prop["patch_dir_vec_map"][ii]
                if patch_dir_vec_map_conf is not None and "patch_dir_vec_map_conf" in prop.files:
                    patch_dir_vec_map_conf[idx] = prop["patch_dir_vec_map_conf"][ii]
                if "patch_dir_source" in prop.files:
                    src_arr = np.asarray(prop["patch_dir_source"])
                    patch_dir_source[idx] = str(src_arr.item() if src_arr.ndim == 0 else src_arr[ii])
    if a.gt_patch_params:
        gtp = np.load(a.gt_patch_params, allow_pickle=True)
        if a.gt_key not in gtp.files:
            raise KeyError(f"{a.gt_patch_params} has no key {a.gt_key!r}")
        gv = gtp[a.gt_key].astype(np.float32)
        if len(gv) == len(pid):
            gt[:] = gv[: len(pid)]
        elif "used_patch" in gtp.files and len(gv) == len(gtp["used_patch"]):
            used = gtp["used_patch"].astype(np.int64)
            ok = (used >= 0) & (used < len(pid))
            gt[used[ok]] = gv[ok]
        elif a.max_patches > 0 and len(gv) > len(pid):
            gt[:] = gv[: len(pid)]
        else:
            raise ValueError(f"{a.gt_patch_params}:{a.gt_key} shape {gv.shape} cannot map to P={len(pid)}")
    for r in rows:
        if len(r) >= 4:
            rr, pp, pv, gv = r[0], r[1], r[2], r[3]
        elif len(r) == 3:
            pp, pv, gv = r
            rr = None
        else:
            pp, pv = r
            gv = None
            rr = None
        try:
            idx = int(pp)
        except ValueError:
            hit = np.nonzero(pid == pp)[0]
            if len(hit) == 0:
                continue
            idx = int(hit[0])
        if 0 <= idx < len(pid):
            pred[idx] = float(pv)
            if gv is not None:
                gt[idx] = float(gv)
            if rr is not None and not a.region_npz and not a.patch_params:
                region[idx] = rr
else:
    # Legacy table-only mode.
    if any(len(r) < 4 for r in rows):
        raise SystemExit("table-only mode needs rows: region patch_id pred_density gt_density")
    region = np.array([r[0] for r in rows]).astype("U24")
    pid = np.array([r[1] for r in rows]).astype(str)
    pred = np.array([float(r[2]) for r in rows], np.float32)
    gt = np.array([float(r[3]) for r in rows], np.float32)
    patch_conf_base = np.ones(len(pid), np.float32)
    patch_z64 = np.zeros((len(pid), 0), np.float32)
    patch_z64_map = None
    patch_z64_map_path = ""
    patch_z64_map_shape = None
    patch_z64_map_dtype = ""
    patch_color_root_tip = np.zeros((len(pid), 0), np.float32)
    patch_dir = np.zeros((len(pid), 2), np.float32)
    patch_dir_coh = np.zeros(len(pid), np.float32)
    patch_dir_measured = np.zeros((len(pid), 2), np.float32)
    patch_dir_measured_coh = np.zeros(len(pid), np.float32)
    patch_dir_diffusion = np.zeros((len(pid), 2), np.float32)
    patch_dir_diffusion_coh = np.zeros(len(pid), np.float32)
    patch_dir_vec_map = None
    patch_dir_vec_map_conf = None
    patch_dir_source = np.full(len(pid), "measured", dtype="U16")
    patch_radius = np.zeros(len(pid), np.float32)
    patch_length_mean = np.zeros(len(pid), np.float32)
    patch_texture_hf = np.zeros(len(pid), np.float32)
    patch_texture_conf = np.zeros(len(pid), np.float32)
    patch_native_px = np.zeros(len(pid), np.float32)
    patch_effective_px = np.zeros(len(pid), np.float32)
    patch_crop_px = np.zeros(len(pid), np.float32)
    patch_model_px = np.zeros(len(pid), np.float32)
    patch_upscale = np.ones(len(pid), np.float32)
    patch_scale_conf = np.zeros(len(pid), np.float32)
    patch_length_remap = np.ones(len(pid), np.float32)
    patch_selected_view = np.full(len(pid), -1, np.int32)
    patch_rgbproj = np.zeros((len(pid), 3), np.float32)

N = len(pred)
gt_valid = np.isfinite(gt)
rk0, rv0 = np.unique(region, return_counts=True)
print(f"{N} patches, {len(rk0)} regions {dict(zip(rk0.tolist(), rv0.astype(int).tolist()))}", flush=True)
if pack is not None:
    print(f"pack={a.pack} | region_source={pack['region_source']} | table_rows={len(rows)}", flush=True)

# ---- 1) GLOBAL calibration: moment-match the compressed pred distribution to the true density scale ----
# (deploy stores these constants; fit here only when GT is available)
if np.isfinite(a.calib_scale):
    scale = float(a.calib_scale)
    offset = float(0.0 if not np.isfinite(a.calib_offset) else a.calib_offset)
elif gt_valid.any():
    scale = float(gt[gt_valid].std() / max(pred[gt_valid].std(), 1e-6))
    offset = float(gt[gt_valid].mean() - scale * pred[gt_valid].mean())
else:
    scale = 1.0
    offset = float(0.0 if not np.isfinite(a.calib_offset) else a.calib_offset)
calib = np.clip(scale * pred + offset, 0.0, None)                                          # per-patch calibrated coverage
if gt_valid.any():
    corr = float(np.corrcoef(pred[gt_valid], gt[gt_valid])[0, 1]) if gt_valid.sum() > 1 else float("nan")
    print(f"global_density_scale={scale:.3f} offset={offset:.3f} | corr(pred,GT)={corr:+.2f}", flush=True)
else:
    print(f"global_density_scale={scale:.3f} offset={offset:.3f} | no GT gate", flush=True)

# ---- 2) REGION robust average (median) + confidence (tight spread + enough patches) ----
regs = sorted(set(region.tolist())); region_density = {}; region_conf = {}
for rg in regs:
    v = calib[region == rg]
    med = float(np.median(v)); mad = float(np.median(np.abs(v - med)) + 1e-6)
    region_density[rg] = med
    region_conf[rg] = float(min(1.0, len(v) / 4.0) / (1.0 + mad / max(med, 1e-6)))            # more patches + tighter -> higher conf
rdens = np.array([region_density[r] for r in region]); rconf = np.array([region_conf[r] for r in region])

# ---- 3) patch residual, CLIPPED to a fraction of the region level (small local correction only) ----
raw_delta = calib - rdens
clip = a.resid_frac * rdens
rho_delta = np.clip(raw_delta, -clip, clip)
final_density = rdens + rho_delta                                                            # what deploy consumes

# ---- COLOR SOURCE: real RGB-projection albedo is the ROOT color (diffusion lit_color drifts purple/yellow at extremes -> demoted to a bounded tip residual only) ----
if a.color_source == "rgbproj" and np.abs(patch_rgbproj).sum() > 0:
    _alb = np.clip(patch_rgbproj, 0.0, 1.0)                                                  # real observed albedo (reliable at dark/saturated markings)
    if patch_color_root_tip.shape[1] >= 6:                                                   # bounded diffusion (tip-root) variation -> tip gets realistic lightening WITHOUT importing purple/yellow
        _tip_resid = np.clip(patch_color_root_tip[:, 3:6] - patch_color_root_tip[:, :3], -a.color_tip_resid, a.color_tip_resid)
    else:
        _tip_resid = np.zeros_like(_alb)
    patch_color_root_tip = np.concatenate([_alb, np.clip(_alb + _tip_resid, 0.0, 1.0)], 1).astype(np.float32)
    print(f"COLOR SOURCE = rgbproj (real albedo) | root RGB mean {_alb.mean(0).round(3)} | diffusion color -> bounded tip residual (+/-{a.color_tip_resid})", flush=True)
else:
    print(f"COLOR SOURCE = diffusion lit_color (WARN: drifts purple on dark / yellow on tan)", flush=True)

# ---- COLOR calibration (data-driven, NO fixed color/coat priors -> transfers). bilateral anchor preserves markings ----
NC = patch_color_root_tip.shape[1]
if NC > 0:
    cstd = patch_color_root_tip.std(0) + 1e-6                                                # global per-channel color spread (data-driven clip scale)
    region_color = {rg: np.median(patch_color_root_tip[region == rg], 0) for rg in regs}     # region medians (still in the contract)
    rmed = np.stack([region_color[r] for r in region]).astype(np.float32)
    _pos = pack["patch_pos"].astype(np.float32) if (pack is not None and "patch_pos" in pack and len(pack["patch_pos"]) == len(patch_color_root_tip)) else None
    if a.color_mode == "bilateral" and _pos is not None:                                     # BILATERAL: spatially-broad (covers a marking region) AND color-similar -> smooths rainbow WITHIN a marking, keeps the marking edge
        dpos = np.linalg.norm(_pos[:, None] - _pos[None], axis=2)
        extent = float(np.linalg.norm(_pos.max(0) - _pos.min(0)))
        sig_s = a.color_spatial_frac * extent + 1e-6                                         # spatial bandwidth = frac of the animal extent (region-scale, data-driven)
        dcol = np.linalg.norm(patch_color_root_tip[:, None] - patch_color_root_tip[None], axis=2)
        sig_c = float(patch_color_root_tip.std()) * a.color_sim_mult + 1e-6                  # color bandwidth ~ noise level -> does not cross white/brown marking boundary
        W = np.exp(-(dpos / sig_s) ** 2) * np.exp(-(dcol / sig_c) ** 2)
        anchor = ((W[:, :, None] * patch_color_root_tip[None]).sum(1) / (W.sum(1, keepdims=True) + 1e-9)).astype(np.float32)
        mode_used = "bilateral (spatial+color, marking-preserving)"
    else:
        anchor = rmed; mode_used = "region-median (lumps markings)"
    rcol = anchor                                                                            # consumed anchor
    color_delta = np.clip(patch_color_root_tip - anchor, -a.color_resid_frac * cstd, a.color_resid_frac * cstd)
    final_color = np.clip(anchor + color_delta, 0.0, 1.0).astype(np.float32)
    if _pos is not None:                                                                     # spatial NEIGHBOR JUMP = the real rainbow (within-region std wrongly penalizes kept markings)
        _d = np.linalg.norm(_pos[:, None] - _pos[None], axis=2); np.fill_diagonal(_d, 1e9)
        _knn = np.argsort(_d, 1)[:, :6]
        def _jump(x): return float(np.mean([np.linalg.norm(x[i] - x[_knn[i]], axis=1).mean() for i in range(len(x))]))
        rainbow_raw, rainbow_fin = _jump(patch_color_root_tip), _jump(final_color)
    else:
        def _cwstd(x): return float(np.mean([x[region == rg].std(0).mean() for rg in regs]))
        rainbow_raw, rainbow_fin = _cwstd(patch_color_root_tip), _cwstd(final_color)
    devm = np.linalg.norm(patch_color_root_tip - rmed, axis=1); hi = devm > np.quantile(devm, 0.8)   # the high-contrast markings
    keep = float(1.0 - np.abs(final_color[hi] - patch_color_root_tip[hi]).mean() / (np.abs(rmed[hi] - patch_color_root_tip[hi]).mean() + 1e-9))
    print(f"\n==== COLOR CALIBRATION ({mode_used}) ====")
    print(f"  neighbor rainbow: raw={rainbow_raw:.4f} -> final={rainbow_fin:.4f}  ({100*(1-rainbow_fin/(rainbow_raw+1e-9)):+.0f}% smoother)")
    print(f"  high-contrast marking preservation={keep:+.2f}  (1=white chest/face kept, 0=pulled to region brown)")
else:
    region_color = {rg: np.zeros(6, np.float32) for rg in regs}; rcol = np.zeros((len(pid), 6), np.float32)
    color_delta = np.zeros((len(pid), 6), np.float32); final_color = np.zeros((len(pid), 6), np.float32)

# ---- region DIRECTION (chosen measured/model, coherence-weighted axial mean per region; no prior) ----
region_dir = {}; region_dir_conf = {}
for rg in regs:
    m = region == rg; w = patch_dir_coh[m]
    acc = (patch_dir[m] * w[:, None]).sum(0); nrm = float(np.linalg.norm(acc))
    region_dir[rg] = (acc / nrm).astype(np.float32) if nrm > 1e-9 else np.zeros(2, np.float32)
    region_dir_conf[rg] = float(min(nrm / (float(w.sum()) + 1e-9), 1.0)) if w.sum() > 1e-6 else 0.0

# ---- RADIUS: GLOBAL robust average (user rule: radius/material never raw per-patch -> blotchy; global, coarse-region at most) ----
rvalid = patch_radius > 0
global_radius = float(np.median(patch_radius[rvalid])) if rvalid.any() else 0.0
region_radius = {rg: (float(np.median(patch_radius[(region == rg) & rvalid])) if ((region == rg) & rvalid).any() else global_radius) for rg in regs}
if rvalid.any():
    print(f"\nRADIUS (global robust median): {global_radius:.5f} | per-region [{min(region_radius.values()):.5f},{max(region_radius.values()):.5f}]  (consume: global, optional coarse region; NOT per-patch)", flush=True)

# ---- LENGTH: predicted patch proposal becomes a normalized ratio, consumed after world-scale arclen locking ----
lvalid = np.isfinite(patch_length_mean) & (patch_length_mean > 1e-8)
patch_length_ratio = np.ones(len(pid), np.float32)
region_length_ratio = {rg: 1.0 for rg in regs}
if lvalid.any():
    base_len = float(np.median(patch_length_mean[lvalid]))
    raw_len_ratio = np.ones(len(pid), np.float32)
    raw_len_ratio[lvalid] = patch_length_mean[lvalid] / max(base_len, 1e-8)
    for rg in regs:
        mv = (region == rg) & lvalid
        region_length_ratio[rg] = float(np.median(raw_len_ratio[mv])) if mv.any() else 1.0
    rlen = np.array([region_length_ratio[r] for r in region], np.float32)
    clip_len = float(a.length_resid_frac) * np.maximum(rlen, 1e-6)
    patch_length_ratio = rlen + np.clip(raw_len_ratio - rlen, -clip_len, clip_len)
    med_ratio = float(np.median(patch_length_ratio[lvalid]))
    if med_ratio > 1e-8:
        patch_length_ratio /= med_ratio
    region_length_ratio = {rg: (float(np.median(patch_length_ratio[region == rg])) if np.any(region == rg) else 1.0) for rg in regs}
    print(
        f"\nLENGTH (diffusion proposal): median={base_len:.5f} -> ratio "
        f"[{float(patch_length_ratio.min()):.2f},{float(patch_length_ratio.max()):.2f}] "
        f"region [{min(region_length_ratio.values()):.2f},{max(region_length_ratio.values()):.2f}]",
        flush=True,
    )

# ---- 3-tier acceptance gate vs GT ----
def err(x): return float(np.mean(np.abs(x - gt) / np.clip(gt, 1e-3, None)))                  # mean relative density error
def within_region_std(x):                                                                    # stability: lower = less blotchy
    return float(np.mean([np.std(x[region == rg]) for rg in regs]))
t1, t2, t3 = gt, calib, final_density
if gt_valid.any():
    def corr_gt(x):
        return float(np.corrcoef(x[gt_valid], gt[gt_valid])[0, 1]) if gt_valid.sum() > 1 else float("nan")
    print("\n==== 3-TIER DENSITY GATE (vs GT) ====")
    print(f"{'tier':36s} {'rel.err↓':>9s} {'corr↑':>7s} {'within-region std↓ (blotchy)':>30s}")
    print(f"{'1 field_gt (oracle upper bound)':36s} {err(t1):>9.3f} {1.00:>7.2f} {within_region_std(t1):>30.3f}")
    print(f"{'2 pred calibrated (per-patch)':36s} {err(t2):>9.3f} {corr_gt(t2):>7.2f} {within_region_std(t2):>30.3f}")
    print(f"{'3 calibrated + region avg + resid':36s} {err(t3):>9.3f} {corr_gt(t3):>7.2f} {within_region_std(t3):>30.3f}")
    print("PASS if tier-3 rel.err <= tier-2 AND tier-3 within-region std < tier-2 (region consumption is stabler, not worse).")
else:
    print("\n==== REGION PRIOR SMOKE (no GT) ====")
    print(f"pred calibrated mean={float(calib.mean()):.3f} final mean={float(final_density.mean()):.3f} within-region std={within_region_std(final_density):.3f}")

if a.viz and pack is not None:
    import matplotlib.pyplot as plt
    pos = pack["patch_pos"]
    ext = pos.max(0) - pos.min(0)
    length_axis = int(np.argmax(ext))
    vertical_axis = int(np.argsort(ext)[1])
    colors = {
        "body": "#7f8c8d", "other": "#95a5a6", "face": "#3498db", "head": "#3498db",
        "ears": "#e67e22", "ear": "#e67e22", "legs": "#f1c40f", "leg": "#f1c40f",
        "paws": "#e74c3c", "paw": "#e74c3c", "tail_base": "#1abc9c", "tail": "#34495e",
    }
    plt.figure(figsize=(7, 4))
    for rg in regs:
        m = region == rg
        plt.scatter(pos[m, length_axis], pos[m, vertical_axis], s=7, c=colors.get(rg, "#000000"), label=rg, linewidths=0)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(markerscale=2, fontsize=8, ncol=4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(a.viz) or ".", exist_ok=True)
    plt.savefig(a.viz, dpi=180)
    plt.close()
    print(f"region viz -> {a.viz}", flush=True)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
if patch_z64_map is not None:
    zstd = float(np.mean([patch_z64_map[i].astype(np.float32).std(axis=(0, 1)).mean() for i in range(len(patch_z64_map))]))
    print(f"Z64 SOURCE = full map passthrough | shape={patch_z64_map.shape} | mean within-patch std={zstd:.3f}", flush=True)
elif patch_z64_map_path:
    print(f"Z64 SOURCE = external map path passthrough | {patch_z64_map_path}", flush=True)
else:
    print("Z64 SOURCE = per-patch mean only (legacy; patch texture/locks are collapsed)", flush=True)
save_kwargs = dict(                                                                            # ===== v2_region_prior data contract =====
         patch_id=pid, patch_region=region, patch_conf=np.minimum(1.0, rconf * patch_conf_base).astype(np.float32),
         patch_coverage_pred=pred.astype(np.float32), patch_rho_delta=rho_delta.astype(np.float32),
         global_density_scale=np.float32(scale), global_density_offset=np.float32(offset),
         region_names=np.array(regs), region_density=np.array([region_density[r] for r in regs], np.float32),
         region_density_conf=np.array([region_conf[r] for r in regs], np.float32),
         final_density=final_density.astype(np.float32),
         gt_density=gt.astype(np.float32),
         calibrated_density=calib.astype(np.float32),
         # pass-through / placeholders populated by the diffusion + independent direction + next blocks:
         patch_z64=patch_z64.astype(np.float32), patch_color_root_tip=patch_color_root_tip.astype(np.float32),
         region_color_root_tip=(np.stack([region_color[r] for r in regs]).astype(np.float32) if NC else np.zeros((len(regs), 6), np.float32)),
         final_color_root_tip=final_color.astype(np.float32), patch_color_delta=color_delta.astype(np.float32),
         patch_dir=patch_dir.astype(np.float32), patch_dir_coh=patch_dir_coh.astype(np.float32),
         patch_dir_source=patch_dir_source.astype("U16"),
         patch_dir_measured=patch_dir_measured.astype(np.float32),
         patch_dir_measured_coh=patch_dir_measured_coh.astype(np.float32),
         patch_dir_diffusion=patch_dir_diffusion.astype(np.float32),
         patch_dir_diffusion_coh=patch_dir_diffusion_coh.astype(np.float32),
         region_dir=np.stack([region_dir[r] for r in regs]).astype(np.float32),
         region_dir_conf=np.array([region_dir_conf[r] for r in regs], np.float32),
         patch_radius=patch_radius.astype(np.float32), global_radius=np.float32(global_radius),
         region_radius=np.array([region_radius[r] for r in regs], np.float32),
         patch_length_mean=patch_length_mean.astype(np.float32),
         patch_length_ratio=patch_length_ratio.astype(np.float32),
         region_length_ratio=np.array([region_length_ratio[r] for r in regs], np.float32),
         patch_texture_hf=patch_texture_hf.astype(np.float32),
         patch_texture_conf=patch_texture_conf.astype(np.float32),
         patch_native_px=patch_native_px.astype(np.float32),
         patch_effective_px=patch_effective_px.astype(np.float32),
         patch_crop_px=patch_crop_px.astype(np.float32),
         patch_model_px=patch_model_px.astype(np.float32),
         patch_upscale=patch_upscale.astype(np.float32),
         patch_scale_conf=patch_scale_conf.astype(np.float32),
         patch_length_remap=patch_length_remap.astype(np.float32),
         patch_selected_view=patch_selected_view.astype(np.int32),
         global_radius_scale=np.float32(global_radius / 0.0013 if rvalid.any() else 1.0),    # grow consumes args.radius(0.0013)*scale = learned global radius (data-driven, NOT per-patch)
         global_roughness=np.float32(a.mat_roughness), global_radial_roughness=np.float32(a.mat_radial_rough),  # GLOBAL material hook -> PBR Hair BSDF
         global_coat=np.float32(a.mat_coat), global_spec=np.float32(a.mat_spec), global_hair_ior=np.float32(a.mat_hair_ior))  # fixed defaults; NOT per-patch / NOT learned
if patch_z64_map is not None:
    save_kwargs["patch_z64_map"] = patch_z64_map.astype(np.float16)
elif patch_z64_map_path:
    save_kwargs["patch_z64_map_path"] = np.asarray(patch_z64_map_path)
    if patch_z64_map_shape is not None:
        save_kwargs["patch_z64_map_shape"] = np.asarray(patch_z64_map_shape, dtype=np.int32)
    if patch_z64_map_dtype:
        save_kwargs["patch_z64_map_dtype"] = np.asarray(patch_z64_map_dtype)
if patch_dir_vec_map is not None:
    save_kwargs["patch_dir_vec_map"] = patch_dir_vec_map.astype(np.float16)
if patch_dir_vec_map_conf is not None:
    save_kwargs["patch_dir_vec_map_conf"] = patch_dir_vec_map_conf.astype(np.float16)
if pack is not None:
    save_kwargs.update(
        patch_face=pack["patch_face"].astype(np.int32),
        patch_bary=pack["patch_bary"].astype(np.float32),
        patch_pos=pack["patch_pos"].astype(np.float32),
        region_source=np.array(pack["region_source"]),
        source_pack=np.array(a.pack),
    )
np.savez(a.out, **save_kwargs)
print(f"\nwrote v2_region_prior contract -> {a.out}", flush=True)
