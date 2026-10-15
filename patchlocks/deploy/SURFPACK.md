# Surface-Patch Pipeline (GlassGS ↔ PatchLocks)

_2026-06-17. The clean bridge: GlassGS turns a dog's views into surface-consistent multi-view patch
conditioning; PatchLocks turns each surface patch into drivable 3D fur._

**2026-06-21 reclassification:** real Artemis/NeuralFur runs that use `--flow_target face_flow` or
`--basis_source root_flow` are oracle/teacher-flow tests, not standalone method results. `face_flow`
is baked from finished NeuralFur strands and must not be used as a mainline inference input.

## Why
PatchLocks reconstructs **test** patches well (clean canonical patches) but **real-dog photos** are off
because viewpoint / occlusion / lighting / position / scale all vary, and each patch lands on only ~3 DINO
tokens in a downsampled studio photo (vs ~32 in training). Re-organizing the views into **surface-aligned
multi-view patch conditioning** restores the canonical-patch cleanliness while carrying real (or A14B) fur.

## Closed loop (validated end-to-end on the Shiba A14B turntable, 2026-06-17)
```
A14B clean multi-view (132 frames, white bg, view-consistent, breed-controllable)
  → glassgs_export_surface_patch_dataset.py --src a14b   [surface patches + index→bary→KDtree correspondence]
  → surface_patch_dataset.npz
  → deploy_from_surfpack.py   [PnP-recover camera (0.88px) → multi-shell GRID/INF/CAM → v6 b3+density+z64 diffusion]
  → surfpack_v6fur.npz   [3.89M strands; color/density follow A14B; carries root_face/root_bary = drivable]
```

## `surface_patch_dataset.npz` (the interchange)
| field | shape | meaning |
|---|---|---|
| centers, Tt, Bb, Nn | (P,3) | patch center + consistent TBN frame |
| surf | (P,H,H,3) | per-patch surface grid (3D, mesh frame) |
| normg | (P,H,H,3) | smooth per-cell surface normal |
| valid | (P,H,H) | on-surface mask |
| cell_face | (P,H,H) int32 | mesh face id per cell (→ mesh_tex color, drive) |
| cell_bary | (P,H,H,3) | barycentric per cell |
| root_face, root_bary | (P,), (P,3) | patch root binding |
| uv | (P,Kc,H,H,2) f16 | per-cell normalized pixel in each candidate view |
| vis | (P,Kc,H,H) f16 | per-cell visibility (index/z-buffer occlusion) |
| sel | (P,K) int32 | top-K candidate views per patch (model fixed K=4) |
| cand_names | (Kc,) | candidate view image paths (full paths for a14b) |

## Exporter `glassgs_export_surface_patch_dataset.py`
- `--src photos --dog <id> --mesh <D-SMAL.obj>`: real multi-view photos + COLMAP cameras; occlusion = mesh z-buffer.
- `--src a14b --manifest <json>`: GlassGS turntable; **camera-free** correspondence — each fg pixel → 3D surface
  point via index `(face_id,bary)` → KDtree match to patch cells. Caveats handled: A14B is fatter than the
  control silhouette (flyaways have no index) → `--erode` alpha + `--min_luma` drop dark. Loads the mesh at the
  index's pose (`drive_params.pose_anim/pose_frame`).

## Consumer `deploy_from_surfpack.py`
Reads the pack, **recovers each candidate camera by PnP** from the (surf ↔ uv) index correspondences
(intrinsics from `--fovy/--res`; reproj ~0.9px), builds the v6 model's multi-shell `GRID/INF/CAM`, runs the
SAME v6 model (`stage3_v6_codec` + `fur_strand_codec_v6upright.pt`, `--znn`), places strands via the patch TBN,
and writes per-strand `root_face/root_bary`. **Topology-agnostic: swap pack + mesh, no model change.**

## NeuralFur geodesic pack -> v9-lite, 2026-06-19
For NeuralFur-style assets the clean data layer is:

```
geodesic_patchpack.npz
  + NeuralFur images_2/masks_2/3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl
  -> neuralfur_export_surface_patch_dataset.py
  -> surface_patch_dataset.npz
  -> deploy_v9lite_from_surfpack.py
  -> surfpack_v6fur.npz
```

`neuralfur_export_surface_patch_dataset.py` converts the geodesic pack's fixed anchors into the same
`surface_patch_dataset.npz` interchange used above. It keeps patch coordinates fixed (`patch_face+patch_bary`),
builds each patch's local HxH surface grid by snapping to the dense `root_pos/root_face/root_bary` cloud, and
adds per-view `uv/vis` from NeuralFur row-major projection matrices plus body/hair masks. This is the missing
2D correspondence layer for running patch models on the NeuralFur dog.

Validated on the synthetic brown beagle folder:

```bash
/home/rog/miniconda3/bin/python patchlocks/deploy/neuralfur_export_surface_patch_dataset.py \
  --pack patchlocks/deploy/neuralfur_beagle_geopatch_fine/geodesic_patchpack.npz \
  --data_dir patchlocks/deploy/neuralfur_beagle_synth2d_brown \
  --field_pack patchlocks/deploy/neuralfur_beagle_official/beagle_surface_field.npz \
  --H 64 --K 4 \
  --out patchlocks/deploy/neuralfur_beagle_synth2d_brown_surfpack
```

Observed: `P=1787`, `H=64`, `Kc=4`, `psize=0.08933`, PnP reprojection in v9-lite `0.03px`, all 1787 patches
covered by at least one view.

v9-lite geometry-first deploy:

```bash
/home/rog/miniconda3/bin/python patchlocks/deploy/deploy_v9lite_from_surfpack.py \
  --pack patchlocks/deploy/neuralfur_beagle_synth2d_brown_surfpack/surface_patch_dataset.npz \
  --geo_dir patchlocks/datasets/stage3_v9_attr \
  --vlm_json patchlocks/deploy/neuralfur_beagle_synth2d_brown_surfpack/vlm_geometry.json \
  --length 0.24 \
  --target_strands 120000 \
  --dino_res 224 --fovy 38 --res 768 --chunk 24 \
  --out patchlocks/deploy/neuralfur_beagle_synth2d_brown_surfpack/v9lite_len024_debug
```

This produces `surfpack_v6fur.npz` with `strands/color/color_tip/radius/root_face/root_bary`.
Color is constant/debug-only in this path; geometry and 2D optimization should use mask, projected coverage, and
orientation losses, not RGB. Current calibration: the NeuralFur brown target has median arclength about `0.029`;
`long_golden` default (`length=0.42`) gives `0.0506`, `--length 0.24` gives `0.039`, and `--length 0.16` gives
`0.0348`. The mapping is not linear because `_grow` adds curl/gravity arclength. For a closer brown match, use
`--length 0.08..0.12` or let the 2D optimization solve length from silhouette/coverage.

2D geometry optimization can now feed back into v9-lite:

```bash
/home/rog/miniconda3/envs/difflocks/bin/python patchlocks/deploy/patch_graph_opt_geo.py \
  --pack patchlocks/deploy/neuralfur_beagle_geopatch_fine/geodesic_patchpack.npz \
  --data_dir patchlocks/deploy/neuralfur_beagle_synth2d_brown \
  --field_pack patchlocks/deploy/neuralfur_beagle_official/beagle_surface_field.npz \
  --views 0000,0004,0008,0012 \
  --out patchlocks/deploy/neuralfur_beagle_synth2d_brown_surfpack/opt_geo_smoke \
  --render_w 384 --max_roots 25000 --steps 60 \
  --screen_align none --align_min_iou 0.2
```

The smoke run selected `1768/1787` active patches and drove orientation loss from `1.22` to `0.25`.
It writes `patch_params.npz` with `rho/length/lift/flow_delta`. These are patch-level geometry latents;
coordinates stay fixed.

Feed those patch latents back into v9-lite:

```bash
/home/rog/miniconda3/bin/python patchlocks/deploy/deploy_v9lite_from_surfpack.py \
  --pack patchlocks/deploy/neuralfur_beagle_synth2d_brown_surfpack/surface_patch_dataset.npz \
  --geo_dir patchlocks/datasets/stage3_v9_attr \
  --vlm_json patchlocks/deploy/neuralfur_beagle_synth2d_brown_surfpack/vlm_geometry.json \
  --length 0.14 \
  --patch_params patchlocks/deploy/neuralfur_beagle_synth2d_brown_surfpack/opt_geo_proper/patch_params.npz \
  --patch_length_norm median --patch_rho_norm mean \
  --target_strands 120000 \
  --target_arclen 0.029 \
  --flow_source patch_params --basis_source root_flow --allow_oracle_flow \
  --comb 0 --grow_dir_jit 0.0 --grow_noise 0.0008 --curl_scale 0.15 \
  --paw_taper --paw_taper_strength 0.65 --paw_taper_start 0.86 \
  --dino_res 224 --fovy 38 --res 768 --chunk 32 \
  --out patchlocks/deploy/neuralfur_beagle_synth2d_brown_surfpack/v9lite_optgeo_proper_flow2d_rootflow_pawclean65_arclen029
```

`--target_arclen` is a final deterministic rescale around each fixed root. It makes absolute length
independent of the optimized `rho/length` distribution; `--length` remains a coarse initializer.
`--flow_source patch_params` uses the 2D-optimized groom direction directly instead of adding it on top
of v9-lite's local direction map. `--basis_source root_flow` uses each root cell's `face_flow` basis
when explicitly enabled and falls back to patch basis otherwise; this is an oracle feedback path because
curved thin parts such as tails should not reuse one fixed patch frame. `--paw_taper` auto-detects
the foot end and shortens only paw-tip strands to suppress foot frizz.
With the proper 300-step geometry run, the feedback surfpack has `120133` strands and median arclength
`0.02900`. This is an oracle-flow closed-loop plumbing test: VLM whole-dog prior -> v9-lite patch
field -> 2D geometry optimization -> patch latent feedback -> v9-lite surface fur. It is not evidence
of independent real-dog fur-direction prediction.

Feet/tail cleanup pass (2026-06-19): `patch_part` in the NeuralFur beagle pack is all `other`, so
`patch_graph_opt_geo.py` now falls back to geometry-derived regions (`paws/legs/tail/other`) and can weight
mask/silhouette/orientation losses by region. The 300-step warm-start run used
`--part_weighted_loss --part_priors --auto_tail_end min` from `opt_geo_proper/patch_params.npz` and wrote:

- `neuralfur_beagle_synth2d_brown_surfpack/opt_geo_partweighted_proper/patch_params.npz`
- final optimization metrics: `mask 0.0320`, `sil 0.0951`, `flow 0.2048`
- active region stats: paw `rho` median `0.939`, paw `length` median `0.350`; tail `rho` median `1.121`,
  tail `length` median `0.662`

The historical feedback render from this pass used the older patch-basis grow path and wrote:

- `neuralfur_beagle_synth2d_brown_surfpack/v9lite_optgeo_partweighted_flow2d_patchbasis_pawclean65_arclen029/surfpack_v6fur.npz`
- `neuralfur_beagle_synth2d_brown_surfpack/v9lite_optgeo_partweighted_flow2d_patchbasis_pawclean65_arclen029/render_texture_diag/_montage.png`
- `119901` strands, median arclength `0.02900`

Historical oracle-flow feedback used the root-flow command above.

C-class edge cleanup (2026-06-19): toe-tip spikes and tail-base spikes are treated separately. For tail base,
`patch_graph_opt_geo.py` has `--tail_base_seam_factor`, which keeps tail-to-body boundary flow weakly coupled
without restoring full body/tail smoothing. The warm-start run from the A+B params used
`--tail_base_seam_factor 0.55` and wrote:

- `neuralfur_beagle_synth2d_brown_surfpack/opt_geo_partweighted_tailbase55/patch_params.npz`
- final optimization metrics: `mask 0.0318`, `sil 0.0946`, `flow 0.2072`
- region stats stayed stable: paw `rho` median `0.945`, paw `length` median `0.355`; tail `rho` median `1.121`,
  tail `length` median `0.658`

For toe tips, `deploy_v9lite_from_surfpack.py` can read `patch_region` and apply local grow scales plus
`--paw_tip_trim`, which shortens only foot-tip strands whose lay direction is a local voxel-flow outlier. The
current rendered cleanup uses:

```bash
--paw_noise_scale 0.35 --paw_curl_scale 0.45 \
--tail_noise_scale 0.75 --tail_curl_scale 0.85 \
--paw_taper --paw_taper_strength 0.65 --paw_taper_start 0.86 \
--paw_tip_trim --paw_tip_trim_strength 0.55 --paw_tip_trim_start 0.86 --paw_tip_trim_angle 50
```

It wrote:

- `neuralfur_beagle_synth2d_brown_surfpack/v9lite_optgeo_tailbase55_tiptrim_pawclean65_arclen029/surfpack_v6fur.npz`
- `neuralfur_beagle_synth2d_brown_surfpack/v9lite_optgeo_tailbase55_tiptrim_pawclean65_arclen029/render_texture_diag/_montage.png`
- `120128` strands, median arclength `0.02900`; toe-tip trim affected `4187` strands

Real Artemis beagle f210 pass (2026-06-19): the old real mv12 run used image structure-tensor orientation and
stalled around `flow 0.417`, consistent with marking/texture contamination. `patch_graph_opt_geo.py` now has
`--flow_target face_flow`, which projects the 3D `face_flow` basis into each camera and uses that as the
orientation target. Because this `face_flow` is baked from NeuralFur's finished strands, this pass is an
oracle/teacher-flow test only. It keeps `flow_delta` near zero and lets the optimizer fit
coverage/length/lift instead of learning color-marking directions.

The real frame-210 geodesic surfpack must be regenerated from the 1787-patch geodesic pack; the older
`surfpack_beagle_f210` has only 170 patches and cannot consume the geodesic `patch_params` directly. The
12-view export wrote:

- `neuralfur_beagle_official/realviews/surfpack_beagle_f210_geopatch/surface_patch_dataset.npz`
- `P=1787`, `H=64`, `Kc=12`, `K=4`, `covered_patches=1786/1787`

The mature real optimizer run wrote:

- `neuralfur_beagle_official/realviews/opt_geo_f210_faceflow_partweighted_tailbase55/patch_params.npz`
- final metrics: `mask 0.1146`, `sil 0.1355`, `flow 0.0040`
- active stats: paw `rho` median `0.925`, paw `length` median `0.319`; tail `rho` median `1.141`,
  tail `length` median `0.759`; `flow_delta` std `0.061`

The v9-lite feedback render wrote:

- `neuralfur_beagle_official/realviews/v9lite_f210_faceflow_partweighted_arclen029/surfpack_v6fur.npz`
- `neuralfur_beagle_official/realviews/v9lite_f210_faceflow_partweighted_arclen029/render_texture_diag/_montage.png`
- `119972` strands, median arclength `0.02900`; toe-tip trim affected `1477` strands
- ~~caveat: patch-local PnP on the real surfpack reports `22.46px`~~ **FIXED 2026-06-20**: pass
  `--matrices <nf_data_f210>/3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl` to `deploy_v9lite_from_surfpack.py`
  → crop shells and the camera-conditioning direction use the baked world→clip matrices (0.013px) instead of the pinhole
  solvePnP refit (22.48px; a pinhole can't represent the baked proj_m matrix, error ~constant across all fovy → NOT a
  calib error). With `--matrices`, the deploy path is PnP-free; real close-up patch-local conditioning now lands on the
  correct fur region, so it is safe to judge image-conditioned detail.
- Region labels are intentionally broad for general animals, not fine anatomy: `body`, `face`, `ears`, `legs`, `paws`,
  `tail_base`, `tail`. Use `patch_graph_opt_geo.py --auto_region_level important`; legacy `fine` is accepted as an alias
  but no longer emits `muzzle`/`neck`.

Formal real comparison (2026-06-20): the baked-matrix path was run at `dino_res=448`, `target_strands=120000`,
`target_arclen=0.029`, with identical grow cleanup for V9LITE-only and optimized feedback. Outputs are under:

- `neuralfur_beagle_official/realviews/compare_matrix_important_20260620/v9lite_only_arclen029/surfpack_v6fur.npz`
- `neuralfur_beagle_official/realviews/compare_matrix_important_20260620/opt_geo_important_tailbase55/patch_params.npz`
- `neuralfur_beagle_official/realviews/compare_matrix_important_20260620/optgeo_arclen029/surfpack_v6fur.npz`
- `neuralfur_beagle_official/realviews/compare_matrix_important_20260620/optgeo_arclen029_rootflow_tailclean/surfpack_v6fur.npz`
- `neuralfur_beagle_official/realviews/compare_matrix_important_20260620/optgeo_mainline/surfpack_v6fur.npz`
- `neuralfur_beagle_official/realviews/compare_matrix_important_20260620/projection_compare_v9only_vs_optgeo.png`
- `neuralfur_beagle_official/realviews/compare_matrix_important_20260620/blender_compare_v9only_vs_optgeo.png`
- `neuralfur_beagle_official/realviews/compare_matrix_important_20260620/tail_patchbasis_vs_rootflow_r001.png`

Comparison: V9LITE-only produced `119020` strands, median arclength `0.02900`, rough 2D point-coverage IoU `0.7642`.
The optimized feedback produced `119928` strands, median arclength `0.02900`, rough IoU `0.7784`; optimizer final
metrics were `mask 0.1065`, `sil 0.1214`, `flow 0.0028`. The improvement is modest but consistent, mostly reducing
missing coverage (`FN 0.0783 -> 0.0565`) rather than expanding false positives.
The historical oracle-flow 3D output is the root-flow tailclean feedback: `120274` strands, median arclength
`0.02900`, tail outlier trim affected only `75` strands. This keeps our higher tail plume quality while
avoiding the patch-basis direction mismatch on the curved tail surface.

Historical Shiba A14B defaults:

- `--app_model patchlocks/datasets/stage3_v6_codec/appearance_model.pt`
- `--color_src surf` (sample A14B surface-projected RGB instead of trusting the appearance head for color)
- `--pou 1` (face-based partition-of-unity density normalization)
- `--comb 0.5` (smooth cross-patch fur direction)
- `--color_blend 0.5` (surface color-atlas blend to reduce patch tone blocks)

### ⚠️ Per-patch CLOSE-UP crop is mandatory (the whole point)
The model was trained on canonical patch **close-ups** where each patch FILLS the frame → spans ~`Hf=dino_res/14`
≈ 32 DINO tokens. If the consumer feeds the **whole** turntable frame to DINO and samples the patch at its
projected location, the patch covers only **~3.6 tokens** (0.11× canonical) — high-freq fur texture is gone and
the model falls back to a generic flat-shell prior (looks like a painted shell, "no fur detail"). The consumer
therefore, per patch × top-K view: (1) crops the patch's projected bbox out of the native frame and **zooms it to
fill `dino_res`** (`MARGIN≈1.2` → ~27-token span), runs DINO on that crop; (2) builds `GRID` in **crop-local**
coords; (3) computes **per-patch** `GLOB` from that patch's own crops (`cat[tok.mean over views&pos, cls.mean]`,
matching training `GLOB[i]`, NOT a whole-dog global); (4) sets `CAM` to a **patch-local virtual close-up camera**
(view dir in TBN, radius `R_TRAIN≈2.78` = train median `|cam_loc|`) instead of the far whole-dog camera; (5)
samples `RGBPROJ` from the crop. Diagnostic printed at run: `patch token span after zoom: med ~27 (train ~32)`.
Fixed 2026-06-18; before this the surface-patch pipeline existed but the consumer regressed to whole-frame
sampling → starved conditioning. **Geometry is correct fur (lift, curl, length-variation, dir-coherence ~0.53);
on a whole-dog turntable the fur is only ~2% of body diag → sub-pixel, so inspect with a close-up, not the
orbit frame.** Validated close-up recipe (2026-06-18): `render_dog_fur_pro.py --views 1 --res 1536 --samples 64
--camdist 0.6 --diffuse --simple_light --val 0.9 --sat 1.3 --key_warm 0.4` and **NO `--radius` override** (let it
use the npz per-strand radius ~0.00105, aspect ~89:1 = fine hair; a debug `--radius 0.004-0.006` is 4-6× too fat →
"noodles"). `--diffuse` is mandatory: without it the Hair BSDF grazing-washes lighter strands to white →
salt-and-pepper white speckle; in the current Shiba pack the high-luminance strands are about 9% and
should read as coherent cream coat regions, not random white dots.

### Visual validation, 2026-06-18
Consumer output on the Shiba A14B pack passes only under the correct inspection scale:

- `surfpack_a14b_shiba/furview_*.png` whole-dog orbit renders look like a washed shell because the fur is
  sub-pixel at that framing; this is **not** a valid failure signal for patch geometry.
- `surfpack_a14b_shiba/closeup/furview_0.png` after the PoU/color/comb fix resolves the strands: fine fibers,
  coherent directional fur bundles, lifted clumps, and orange/white Shiba markings from A14B are visible.
  This confirms the consumer is producing real 3D fur, not just a texture shell.
- Rendering must not override to a large debug radius (`--radius 0.004/0.005`) when judging fur detail; use the
  NPZ radius or a close-up-specific value intentionally. Large radii make the surface look like a thick shell.

Quality fixes from that historical A14B path:

- **Density/geometry PoU:** face-based `win / WSUM` de-duplicates overlapping patches without per-texel
  over-division. On the Shiba pack, overlap coverage is ~1.7 patches/covered face and strand count drops from
  ~3.98M duplicate-heavy strands to ~0.94M full-coverage strands.
- **Surface color source:** `--color_src surf` samples the A14B multi-view RGB projection directly; unseen cells
  fall back to patch mean. This avoids the appearance-head white speckle/hallucination failure.
- **Surface color atlas:** voxel PoU color averaging + light 3D blur, blended at `--color_blend 0.5`, removes most
  rectangular patch tone blocks while preserving orange/cream markings.
- **Direction comb:** voxel-smoothed tangential flow, applied via Rodrigues rotation about each root normal at
  `--comb 0.5`, removes most cross-patch fur-flow discontinuities without fully flattening natural variation.

## A14B manifest contract (what the GlassGS side provides)
```json
{ "segments":[ {"frames_dir":..., "index_dir":..., "num_frames":N}, ... ],
  "drive_params":{ "mesh_dir":..., "pose_anim":"Idle", "pose_frame":1, "fovy":40, "res":768 } }
```
- `frames/f###.png` = A14B output; `index/f###.npz` = `{face_id int32(HxW,-1=bg), bary float16(HxW×3 [u,v,w]), alpha}`.
- frame i ↔ index f{i} share the camera (construction-guaranteed); index is on the **same mesh** at the drive pose.
- **Topology must match the deploy target**: Shiba pack = pipeline validation only; a **D-SMAL** turntable
  (real dog's D-SMAL mesh → lit 360 → A14B → index on the same D-SMAL mesh) is required so `--src a14b` aligns
  with `--src real` (same 7774 faces) for real-dog deployment.

## Next
- D-SMAL A14B turntable for a real dog (its own short coat) → exporter `--src a14b` → consumer → real-dog
  3D fur, directly comparable to `--src photos` (dirty studio) to confirm clean multi-view fixes "real not like".
- Evaluate D-SMAL results with patch/close-up renders and source-camera crops first; whole-dog orbit is only an
  overview, not the quality gate for short fur.
- Color: `fur_post.color_from_meshtex` already samples the GlassGS `mesh_tex.pt` DB via `root_face/root_bary`.
