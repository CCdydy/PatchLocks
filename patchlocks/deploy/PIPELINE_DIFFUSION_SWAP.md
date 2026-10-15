# Diffusion-swap runbook (locked whole-animal pipeline)

The whole-animal pipeline is locked so the **diffusion is the only swappable part**.
When the new, stronger/more-general diffusion is trained, swap it in and re-run;
every per-animal "characteristic" is preserved automatically.

## 1. The single swap point

The diffusion is used in **Step 1 only** (`infer_v10_surfpack_proposals.py`,
via `run_v2_character_mainline.sh`). Everything else consumes its proposals.

```bash
# edit the top of run_locked_3animals.sh, or pass as env:
DIFFUSION_BACKEND=custom \
GEO_DIR=patchlocks/datasets/<new_model_dir> \
MODEL_PATH=patchlocks/datasets/<new_model_dir>/stage3_model.pt \
  bash patchlocks/deploy/run_locked_3animals.sh
```

`run_v2_character_mainline.sh` already supports `DIFFUSION_BACKEND={v10,v11_2b,custom}`
and verifies that cached Step-1 proposals were made by the current model
(`REUSE_STEP1` guard), so a stale prior can never silently leak into a new run.

## 2. What regenerates vs. what is locked

| Stage | Script | On diffusion swap |
|------|--------|-------------------|
| 1 proposals | `infer_v10_surfpack_proposals.py` | **regenerates** (new model) |
| 1b region prior | `build_v2_region_prior.py` | **regenerates** (density/footprint) |
| 1c.6 annotation | `apply_individual_region_annotation.py` | **re-applied, locked labels** |
| 1c/1d character + swatch | `extract_fur_character.py` / `build_canon_swatch.py` | **regenerates** (new character) |
| 2 Step2 direction/length | `patch_graph_opt_geo.py` | **regenerates** (anchored on new proposals) |
| 3 grow | `grow_v2_region_prior.py` | **regenerates** geometry, **locked policy** |
| R render | `render_dog_fur_pro.py` | re-render |

**Locked, diffusion-independent (the per-animal characteristics):**
`refine_annotations/<animal>.env` (region-detail / tail / seam policy) +
`<animal>.json` (+ `whiteTiger_patch_region_fixed.npz` for the tiger's corrected
labels), the surface patch packs, geopacks, meshes, multi-view data, and
`TARGET_STRANDS=1200000`.

## 3. Per-animal locked policy (from the approved refine_v4_light run)

| | beagle | panda | whiteTiger |
|---|---|---|---|
| route | **light** | **light** | **strong-fix v3** |
| region labels | auto | auto | **fixed** (`whiteTiger_patch_region_fixed.npz`) |
| leg/body seam | untouched (scale 1.0, cap 0.0, feather 0.0) | untouched | compacted (scale 0.88, cap 0.78, **feather 0.10**) |
| global feather | 0.0 | 0.0 | 0.035 |
| paws | 0.55/0.70/0.55, feather 0.09 | 0.60/0.76/0.60, feather 0.09 | 0.60/0.75/0.60, feather 0.09 |
| ears | 0.78/0.95/0.82 | 0.76/0.94/0.78 | 0.72/0.92/0.70 |
| face | 0.70/0.90/0.45 | 0.78/0.92/0.55 | 0.82/0.92/0.62 |
| tail | uniform, tip 0.90/pow 1.5 | uniform, tip 0.90/pow 1.5 | uniform, base_dist, tip 0.45/pow 1.6, tip_keep 0.15 (meteor-hammer fix) |
| density | 1.2M | 1.2M | 1.2M |

(ear/face/paw triples are scale/keep/body_cap; full values live in the `.env` files.)

## 4. Verification targets (V10 reference, refine_v4_light, 2026-06-28)

| animal | roots | arc_med |
|---|---|---|
| beagle | 1,105,838 | 0.02511 |
| whiteTiger | 1,095,712 | 0.04070 |
| panda | 1,128,000 | 0.05007 |

Root counts should land near these (same density budget + meshes). `arc_med`
**will move** with the new model's length character — that is the intended effect
of a stronger diffusion, not a regression.

## 5. panda Step2 — RESOLVED: honest image-flow (no oracle)

The earlier panda result used `flow_target=face_flow` (a borrowed NeuralFur
direction oracle), but that was only a shortcut — panda's real multi-view capture
was in the repo all along. The ARTEMIS panda dir
`patchlocks/datasets/neuralfur/extracted/data/Artemis/panda_processed_GH2/walk`
contains everything Step2 needs for the honest image-flow route:
`images_2/` (36 RGB views), `masks_2/{body,hair}/`, `orientations_2/{angles,vars}/`
(image-measured orientation), and `3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl`
(36 camera matrices, keyed by view) — the exact paths `load_view_np` and the
auto-derived `load_matrices` read. `run_locked_3animals.sh` now points
`DATA_DIR[panda]` there, so all three animals use real image-measured orientation
with `STEP2_FLOW_TARGET=image` (wrapper default). **No per-instance direction
oracle is used anywhere** — consistent with the paper's claim. (The old face_flow
panda run is discarded; do not reuse it.)
