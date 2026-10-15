# Pipeline, stage by stage

Every stage has one entry script in `pipeline/`. Run them from the repository root, in order.
Stages 1–3 are optional because their products are released (see [DATA.md](DATA.md)).
Every stage reads linked data read-only and writes to `outputs/`.

| Stage | Entry script | Reads | Writes | Time (RTX PRO 6000) |
|---|---|---|---|---|
| 0 Link data | `pipeline/00_link_data.sh DATA_ROOT` | a data root | symlinks per `data/manifest.txt` | seconds |
| 1 Synthetic data *(optional)* | `pipeline/01_synthetic_data.sh` | strand codec, Blender | `outputs/data/v11_longfur/renders_mv` | hours |
| 2 Train diffusion *(optional)* | `pipeline/02_train_diffusion.sh` | `renders_mv`, codec | `outputs/train/v16_e2a_cw15/v14_best.pt` | ~30 min (early stopping) |
| 3 Preprocess a new animal *(optional)* | [NEW_ANIMAL.md](NEW_ANIMAL.md) | multi-view capture + furless mesh | surface patch pack, geodesic cover, cameras | ~1 h, partly manual |
| 4 Reconstruct animals | `pipeline/04_reconstruct_animals.sh` | model, per-animal inputs and recipes | `outputs/refine_v16/<animal>/` | ~15 min per animal (+20 min close-ups) |
| 5 Paper renders | `pipeline/05_render_paper_views.sh` | grooms | `outputs/renders/<src>/` | ~5 min per animal |
| 6 Paper tables | `pipeline/06_evaluate_paper_tables.sh` | model, baselines, frozen test sets, grooms | `outputs/tables/` + tracked JSONs | ~1–1.5 h |
| 7 Paper figures | `pipeline/07_build_paper_figures.sh` | archived renders and grooms | `outputs/figures/` | ~1 min, CPU |

## Stage 4 in detail — one animal

`patchlocks/deploy/run_locked_3animals.sh` loops over animals; for each it sources
`patchlocks/deploy/refine_annotations/<animal>.env` and calls
`patchlocks/deploy/run_v2_character_mainline.sh`:

| Step | What it does | Script | Output in `outputs/refine_v16/<animal>/` |
|---|---|---|---|
| 1 | Image-conditioned diffusion predicts a 64×64 strand-latent texture for every surface patch, fusing the 4 most visible calibrated views | `infer_v14_surfpack_proposals.py` | `proposals_maps/` |
| 1b | Per-patch density and region prior | `build_v2_region_prior.py` | `v2_region_prior.npz` |
| 1c | Decodes predictions and measures the direction-free character (relative length, straightness, curl, lift, tip spread, clump, radius) | `extract_fur_character.py` | `character_prior.npz` |
| 1c.6 | Applies the animal's region annotation (e.g. the corrected white-tiger labels) | `apply_individual_region_annotation.py` | `v2_region_prior.annotated.npz` |
| 1d | Distils reliable patches into one canonical character swatch | `build_canon_swatch.py` | `canon_swatch_reliable_straight.npz` |
| 2 | Screen-space Gaussian fit of per-patch orientation and relative length against fur masks and image orientation in 12 calibrated views | `patch_graph_opt_geo.py` | `step2/patch_params.npz` |
| 3 | Grows a fixed 1.2M-root budget on the furless mesh from the swatch and the fitted fields | `grow_v2_region_prior.py` | `grow/surfpack_v2fur.npz` |
| R | Hair-BSDF QA renders (Blender) | `render_dog_fur_pro.py` | `render/`, then `details/`, `side_full/` |

The `.env` recipes hold the groom-level target length, lift scale and region-specific growth
policy (ears, paws, legs, tail) for each animal. The diffusion model is the only swappable part:
point `GEO_DIR`/`MODEL_PATH` at another checkpoint and everything downstream regenerates
([`PIPELINE_DIFFUSION_SWAP.md`](../patchlocks/deploy/PIPELINE_DIFFUSION_SWAP.md)).

## Determinism

| Stage | Reproducibility |
|---|---|
| Patch generation (stage 1) | seeded, deterministic |
| Blender Cycles renders | GPU path tracing is not bit-exact across runs; rendering an existing groom with the same command reproduced the paper image pixel for pixel on the original machine |
| Training (stage 2) | seeded, but cuDNN determinism is not enforced |
| Diffusion inference (step 1) | bit-identical across runs |
| Steps 1b–1d | bit-identical |
| Step 2 fit | GPU float non-determinism: mean \|Δ\| 3e-4 rad orientation, 0.4 % length |
| Grow (step 3) | deterministic given step 2; identical roots and strand counts |
| Tables (stage 6) | see [REPRODUCIBILITY.md](REPRODUCIBILITY.md) |
| Figures (stage 7) | byte-identical |
