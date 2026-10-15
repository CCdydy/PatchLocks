# PatchLocks Oracle-Flow Geometry Plumbing Reference

_Reclassified 2026-06-21. This is not a standalone reconstruction result._

The real-animal runs in this document use `face_flow` baked from NeuralFur's finished 3D fur strands.
That makes them useful for debugging surface attachment, 2D optimization, seam handling, and grow,
but invalid as evidence that PatchLocks independently predicts real fur direction from photos.

## Essence (one line)
**Fix the 3D surface-patch coordinates on the mesh; optimize only each patch's fur
*geometry latent*; with oracle/teacher flow, feed the latent back to grow a full, drivable 3D coat.**

It is NOT free-floating 3DGS, and NOT 2D texture projection. Coordinates are pinned to
the mesh via `face + bary`; only the fur geometry attributes move. This is the
This validates the surface-anchored primitive machinery, not the independent direction prior.

---

## Pipeline (oracle-flow test: synthetic brown beagle AND real Artemis beagle f210)

### 1. Data layer — surface patch graph
- Core input = **geodesic patchpack** (`geodesic_patchpack.npz`):
  - per patch: fixed surface anchor `patch_face + patch_bary + patch_pos`, `patch_neighbors`.
  - per root/strand: `root_face + root_bary`.
  - `patch_part` on the NeuralFur beagle is **all `other` (1787/1787)** → the optimizer
    derives regions geometrically (`paws / legs / tail / other`) as a fallback.
- Exporter (`neuralfur_export_surface_patch_dataset.py`, or `glassgs_export_*` / `--src photos`)
  → **`surface_patch_dataset.npz`** (the 2D↔3D correspondence layer):
  - per patch a `64×64` surface grid; per cell: 3D surface point, normal, `cell_face/cell_bary`,
    and per candidate view `uv / vis`. `sel` = top-K views per patch; `cand_names` = images.
- Synthetic brown beagle: `P=1787`. Real f210 re-exported to `P=1787, Kc=12`
  (the old `surfpack_beagle_f210` had only 170 patches → could not consume geodesic `patch_params`).

### 2. Init — VLM coat prior + v9lite patch field
- VLM gives only the **whole-dog coat prior** (coat type, rough length/density/curl/lie). **No color.**
- v9lite per-patch init: close-up crop per patch×view → DINO → B3 fuser + field head → local fur field.
- After the loop, **direction comes mainly from the optimized `patch_params` feedback, not v9lite's
  one-shot field** → v9lite is an *optimizable initializer*, not a one-shot generator.

### 3. 2D geometry optimization — `patch_graph_opt_geo.py`
- **Optimized variables (the ONLY free params):** per-patch `rho` (density), `length`, `lift`
  (off-surface / lie-down), `flow_delta` (rotation relative to the 3D flow tangent basis).
  **Coordinates (`patch_face + patch_bary`) are NOT optimized.**
- **Losses are geometric, no color:** mask BCE (coverage vs hair/body mask), soft-IoU silhouette,
  orientation/flow loss, seam-aware graph smoothness, length/lie prior.
- Levers added this session:
  - `--part_weighted_loss` — boost paw/leg/tail loss → kills "large body area dominates,
    small limbs under-fit".
  - `--part_priors` — paws short+dense, tail denser+fuller.
  - `--tail_base_seam_factor` — keep tail↔body boundary **weakly but not zero** coupled
    (avoids a flow discontinuity / spikes at the tail base without flattening the tail).
  - `--flow_target face_flow` — on real dogs this uses a **teacher/oracle direction field** projected
    per view instead of the photo's structure tensor. This is useful for debugging the optimizer, but
    it is not a valid method input.

### 4. Feedback / grow — `deploy_v9lite_from_surfpack.py`
Reads `surface_patch_dataset.npz` + `patch_params.npz` + coat prior + v9lite model, then:
- modulates each patch by `rho/length/lift/flow_delta`;
- `--flow_source patch_params` (use optimized groom direction directly, do NOT add it on top of
  v9lite's field-head direction), `--basis_source root_flow`;
- `root_flow` uses each root cell's `cell_face -> face_flow` tangent basis when explicitly enabled,
  falling back to patch basis otherwise. This is an oracle-flow debugging path because curved thin
  parts such as tails should not reuse one fixed patch frame across the whole patch.
- samples roots to `--target_strands`, grows strands;
- `--target_arclen 0.029` = deterministic global rescale → absolute length is decoupled from the
  optimized `rho/length` distribution (`--length` stays a coarse initializer);
- `--paw_taper` shortens the foot end; `--paw_tip_trim` removes only toe-tip direction outliers;
  `--tail_outlier_trim` lightly removes only tail/tail-base direction outliers;
  per-region grow noise/curl scale for paw/tail.
- Output **`surfpack_v6fur.npz`**: `strands` + `root_face/root_bary` (drivable) + frozen debug color.

### 5. Render validation
- Blender render (flat/debug color, orbit views) inspects 3D groom: coverage, feet/tail, direction,
  fullness. **Validates geometry only, not real 2D texture.** Real markings (brown head / dark back /
  white chest+legs) are deliberately NOT in the geometry path.

---

## Design rationale (the "why" — archive of the thinking)

1. **Fixed coords + optimize latent (not free 3DGS, not 2D texture).** Fur stays glued to the mesh
   (no floaters), anchored by `face/bary` → drivable; only geometry attributes are free. Borrowing
   3DGS's free Gaussian positions would let fur drift off the surface.
2. **Color removed from geometry (decoupled appearance).** Geometry is a smooth patch field driven by
   *appearance-invariant* 2D losses (mask/silhouette/orientation). Per-patch color is band-limited and
   OOD-fragile; it belongs in a separate high-frequency appearance track (per-root projection /
   surface-3DGS), added later.
3. **Information-theoretic division of labor.** Single-view determines texture/coverage/lie →
   v9lite/init. Multi-view *fitting* determines length/flow/curl → step-3 optimizer. Truly
   under-determined (true root density, sub-pixel micro-curl) → prior/condition only. Multi-view
   breaks most single-view degeneracies, so length/flow are *fit*, not guessed.
4. **`face_flow` for real-dog direction invalidates this as a method result.** Single-image field-head direction
   is OOD (scatters, R≈0.57 on real); 2D structure-tensor orientation is contaminated by color
   markings; so these real runs used the clean directed 3D `face_flow`. That `face_flow` is baked
   from NeuralFur's finished reconstruction, so the result is an oracle-flow upper-bound/plumbing test,
   not independent recovery. A valid mainline must replace it with diffusion/region/anatomical flow
   initialization plus 2D refinement.
5. **Part-weighting.** A pixel-summed silhouette loss is area-dominated → the body swamps the gradient
   and thin extremities (tail, feet) under-fit → weight small regions up.
6. **Arclen rescale.** Soft over-splat coverage biases *absolute* length short (10–35%). `median`-norm
   on the feedback decouples absolute length; `--target_arclen` locks it deterministically, leaving the
   optimizer to set the *relative* distribution.
7. **Flow contract.** The optimizer's `flow_delta` is an *absolute* groom angle relative to the
   selected 3D tangent basis. It must be used AS the direction (`flow_source=patch_params`), not ADDED
   on top of v9lite's field-head direction (adding double-counts and scrambles a direction that was
   already good). Mainline feedback uses `basis_source=root_flow` so every root follows its own face's
   `face_flow` basis; this removes most tail spikes caused by applying one patch frame to a curved
   tail surface.
8. **v9lite = optimizable init.** The closed loop (VLM → v9lite → 2D opt → patch latent → feedback)
   turns v9lite from a one-shot generator into a NeuralFur-style initializer that the 2D fit corrects.

---

## Current state

**Done — oracle-flow plumbing only:**
- Synthetic brown beagle: geometry loop closed under controlled/in-distribution conditions.
- Real Artemis beagle f210: 1787-patch surfpack; baked-matrix patch crops (`0.013px` reprojection);
  teacher `face_flow` direction; important region labels; part weighting; root-flow feedback;
  `120274` strands; arclen `0.029`. Current output:
  `neuralfur_beagle_official/realviews/compare_matrix_important_20260620/optgeo_mainline/surfpack_v6fur.npz`.

**Not done — the actual method result:**
- independent real-image fur-direction prediction without NeuralFur-derived `face_flow`;
- diffusion-owned region/patch initialization for direction and fur quality;
- Real color / markings (the geometry path is intentionally color-free).
- Source-camera photometric render (vs flat debug color).

**One line:** this pipeline validated the attachment/optimization/grow mechanics under oracle flow;
it is not a publishable or defensible first result.

---

## Forward options (from here)
- **Appearance loop** — per-root projection / surface-3DGS for sharp markings; first fix the real
  close-up camera (baked matrices, not PnP).
- **Driving** — `root_face/root_bary` is ready; transport fur via per-root TBN onto a deforming body
  (Artemis native bones replay, or SMAL/D-SMAL weight transfer for novel motion).
- **Direction generalization** — diffusion/region or anatomical flow initialization as a replacement
  for oracle `face_flow`.
