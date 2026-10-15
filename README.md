# PatchLocks

Code for **PatchLocks**, which reconstructs animal fur as individual strands from multi-view images.
An image-conditioned diffusion model predicts local 3D fur on surface patches. The prediction is
distilled into a direction-free fur *character*, a screen-space fit estimates per-patch orientation
and relative length, and surface growth places 1.1–1.2M strands on a furless body mesh.
The same patch model reconstructs a beagle, a white tiger, a panda and a cat.

This repository contains the exact code that produced the paper's results, plus everything
needed to regenerate them: the whole-animal pipeline, training and evaluation of the diffusion
model, all paper experiments, figure builders, synthetic-data generators and the per-animal
preprocessing chain.

> **Status.** Prepared from the paper package. Model weights and data are not in git
> (see [Data and weights](#data-and-weights)); download locations will be added when the
> release assets are published.

## What reproduces, and how exactly

Verified on the original machine (RTX PRO 6000 Blackwell, CUDA 12.8) by rerunning from this
repository and comparing with the archived paper outputs:

| Result | How it was checked | Outcome |
|---|---|---|
| Diffusion inference on the real animals (Step 1, 1b–1d) | rerun from weights, compared array by array | **bit-identical** |
| Multi-view orientation/length fit (Step 2) | same | identical up to GPU float non-determinism (mean \|Δ\| 0.0003 rad orientation, 0.4 % length) |
| Final grooms | same | identical strand count and roots; tip displacement median 2.7e-6 (0.02 % of strand length); length percentiles equal to 4 decimals |
| Whole-animal renders | rerender rerun grooms with the paper settings | visually indistinguishable (PSNR 34–38 dB close-ups, 48–50 dB full body) |
| Rendering an archived groom | same Blender command | **pixel-identical** |
| 12 script-built paper figures | `figures/build_paper_figures.sh` + `figures/verify_paper_figures.py` | **byte-identical** |
| Paper tables | archived result JSONs are included; every table maps to a script below | see [Paper tables](#paper-tables) |

The remaining figure, the pipeline overview, was drawn by hand in draw.io (`figures/pipeline/`).

## Repository layout

The layout mirrors the original working directory, because every script resolves inputs by path
relative to the repository root. Run all commands from the repository root.

```
models/ modules/ utils/            strand codec (from DiffLocks) and shared utilities
patchlocks/
  v16/                             paper diffusion model: training, inference, Table 1 evaluation
  v14/v14_common.py                shared model/eval harness (multi-view fuser, sampler, split)
  deploy/                          whole-animal pipeline, per-animal preprocessing, render scripts
    run_locked_3animals.sh         entry point: all animals end to end
    run_v2_character_mainline.sh   one animal: Step 1 -> 1b -> 1c -> 1d -> Step 2 -> grow -> render
    refine_annotations/            per-animal recipe (.env) and region annotations (.json)
  experiments/rebuttal_2027/       paper experiments (Table 2, supplement tables) and result JSONs
  data_gen/ v11_longfur/           synthetic fur-patch generators and dataset builders
  models/                          codec fine-tuning and latent baking
  inference/infer_v10_pixel.py     single-view baseline (Table 1 "1-view" column)
  qa/                              patch renderers used by figures and test sets
figures/                           figure builders, verification, draw.io source
data/manifest.txt                  every checkpoint / dataset the code expects, with size and purpose
scripts/link_data.sh               symlinks the manifest entries from a data root
```

`.provenance/verbatim_manifest.tsv` records the sha256 of every file as imported from the paper
package. The first commit is that verbatim import, so `git diff <first commit>` shows exactly
what changed relative to the code that produced the paper (portability fixes and figure-builder
restorations only).

## Installation

```bash
conda create -n patchlocks python=3.11 -y && conda activate patchlocks
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Rendering uses the Blender 4.5.10 LTS executable (not the `bpy` module), expected at
`patchlocks/third_party/blender-4.5.10-linux-x64/blender`. DINOv2 is fetched through
`torch.hub` on first use. Scripts call `python` unless `PY` is set.

## Data and weights

Nothing large is tracked in git. [`data/manifest.txt`](data/manifest.txt) lists every
checkpoint and dataset the code reads, with size, category and the result that uses it.
Put them under a data root with the same relative layout, then link them into the repository:

```bash
bash scripts/link_data.sh /path/to/DATA_ROOT            # everything
bash scripts/link_data.sh /path/to/DATA_ROOT weights    # or only some categories
```

| Category | Contents | Size |
|---|---|---|
| `weights` | paper diffusion model + fuser, strand codec, single-view baseline, E1 ablation, Table 2 baselines | ~0.6 GB |
| `ours` | synthetic training corpus (1,600 patches × 4 views), frozen test sets, archived grooms and renders | ~28 GB |
| `derived` | preprocessed multi-view captures of the four animals (surface patch packs, cameras, masks, orientations) | ~15 GB |
| `thirdparty` | NeuralFur released meshes/grooms and the ARTEMIS panda capture; Blender 4.5.10 | ~10 GB |

`derived` and `thirdparty` data come from the NeuralFur release (CC BY-NC-SA 4.0,
[github.com/Vanessik/NeuralFur](https://github.com/Vanessik/NeuralFur)) and the ARTEMIS captures
([github.com/HaiminLuo/Artemis](https://github.com/HaiminLuo/Artemis)); their redistribution
follows those licenses.

## Reconstruct the four animals

```bash
bash patchlocks/deploy/run_locked_3animals.sh                 # beagle whiteTiger panda cat
ANIMALS=beagle bash patchlocks/deploy/run_locked_3animals.sh  # one animal
```

Defaults reproduce the paper grooms: the v16 checkpoint
(`patchlocks/datasets/v16/e2a_cw15/v14_best.pt`), a 1.2M-root budget, and each animal's recipe in
`patchlocks/deploy/refine_annotations/<animal>.env`. Outputs go to `outputs/refine_v16/<animal>/`
(`grow/surfpack_v2fur.npz` is the groom; `render/`, plus `details/` and `side_full/` close-ups).
One animal takes about 15 min for the pipeline plus about 20 min for the high-resolution close-up
renders on an RTX PRO 6000. Step 1 is deterministic; Step 2 is reproducible up to GPU float
non-determinism (see the table above). The diffusion model is the only swappable part; see
[`patchlocks/deploy/PIPELINE_DIFFUSION_SWAP.md`](patchlocks/deploy/PIPELINE_DIFFUSION_SWAP.md).

To render a groom with the paper's display material and cameras:

```bash
bash patchlocks/deploy/_render_global4_champagne.sh    # 4-view turntables
bash patchlocks/deploy/_render_details_champagne.sh    # close-ups
```

## Paper tables

Result JSONs used in the paper are included next to the scripts that produce them. Commands run
from the repository root. Scripts whose output path is fixed overwrite the tracked JSON, so
`git diff` shows any change.

| Paper | Command | Result file |
|---|---|---|
| `tab:capability` (n=80, 1-view vs ours) | `python patchlocks/v16/eval_v16_vs_v10.py` (~30 min) | `patchlocks/v16/eval_v16_vs_v10.json` |
| `tab:baselines`, ID column (n=160) | `python patchlocks/experiments/rebuttal_2027/eval_3arm.py --mode all --renders patchlocks/experiments/rebuttal_2027/testB_renders --ckpt_cond_stats --out outputs/eval3arm_testB.json` then `python patchlocks/experiments/rebuttal_2027/bootstrap_ci.py --json outputs/eval3arm_testB.json --out outputs/ci_testB.json` | `eval3arm_testB.json`, `ci_testB.json` |
| `tab:baselines`, EXT column (n=80) | same with `--renders patchlocks/experiments/rebuttal_2027/ood_renders` | `eval3arm_ood.json`, `ci_ood.json` |
| One-view ablation (0.631 → 0.720) | `eval_3arm.py --mode val --out outputs/eval3arm_val80.json`, then `eval_k1.py --k4_json outputs/eval3arm_val80.json --out outputs/eval_k1_val80.json` | `eval_k1_val80.json` |
| Character Spearman correlations | `python patchlocks/experiments/rebuttal_2027/eval_init_positioning.py` | `init_positioning.json` |
| `tab:ceiling` | `eval_clumpaware.py --ckpt_cond_stats --out outputs/clumpaware_testB.json` | `clumpaware_testB.json` |
| `tab:recipe` (E1 middle row) | `python patchlocks/v16/eval_e1_vs_v10_sameprotocol.py` | `eval_e1_vs_v10_sameprotocol.json` |
| `tab:identifiability` | `python patchlocks/experiments/rebuttal_2027/eval_identifiability.py` (CPU) | `identifiability.json` |
| `tab:seams` | `python patchlocks/experiments/rebuttal_2027/eval_seam_continuity.py` (CPU, uses `v2_results/refine_v16_4animals`) | `seam_continuity.json` |
| `tab:grow_scaling` | `bash patchlocks/experiments/rebuttal_2027/run_strand_budget_scaling.sh` | `strand_budget_scaling.json` (assembled from the printed timings) |
| `tab:animals` | from the grooms (`v2_results/refine_v16_fluffy/*/grow`) and `refine_annotations/*.env` | — |

`patchlocks/experiments/rebuttal_2027/RESULTS_SUMMARY.md` and `CONTRACT.md` describe the
evaluation protocol (frozen test sets, seeds, metrics). The Table 2 baselines are trained with
`train_det_z64.py` and `train_direct7.py` in the same folder.

## Paper figures

```bash
bash figures/build_paper_figures.sh            # all figures -> outputs/figures/ (CPU, ~1 min)
python figures/verify_paper_figures.py --paper-figures /path/to/paper/Figures
```

[`figures/README.md`](figures/README.md) maps every figure to its label, builder, inputs and
upstream Blender render.

## Train the diffusion model

```bash
CLUMP_CW=1.5 CLUMP_BUCKET=0 OUT=outputs/v16_retrain bash patchlocks/v16/launch_train_v16.sh
```

These settings match the paper checkpoint's saved arguments (multi-view EDM on z64 latents,
σ_max 80, CFG 3, EMA 0.999, clump-channel weighting 1.5, seed 0, early stopping on the composite
score; the paper checkpoint is the iteration-15000 best). Training reads
`patchlocks/datasets/v11_longfur/renders_mv` (1,600 patches; the patch-level 1,360/240 split is
generated with seed 0 by `patchlocks/v14/v14_common.py`). cuDNN determinism is not enforced, so a
retrained model will be close to, not bit-identical with, the paper checkpoint.

## Regenerate the data

* **Synthetic training corpus** (`renders_mv`): `patchlocks/v11_longfur/build_full_dataset.sh`
  (patch generation with `gen_realdog_v11.py`, seeds 210000+, nine coat archetypes; 4-view
  Blender renders with `patchlocks/data_gen/photoreal_render_loop_v8.py`), then
  `patchlocks/data_gen/bake_mv_targets.py` (z64 latents and field targets) and
  `patchlocks/data_gen/prepare_v13_mv_dataset.py --write-pose` (pose features).
* **Frozen test sets**: `patchlocks/experiments/rebuttal_2027/gen_testset_B.sh` (n=160, seeds
  610000+) and `gen_ood_patches.py --n_per 20` (n=80, seeds 810000+), each followed by its
  `*_postrender.sh`.
* **Strand codec** (`fur_strand_codec_v6upright.pt`): fine-tuned from the DiffLocks strand VAE
  with `patchlocks/models/fine_tune_codec.py` on a pool built by
  `patchlocks/data_gen/{gen_v5_batch.py,render_loop.py,build_pool_v6.py}`. Only needed to retrain
  the codec; the DiffLocks checkpoint is available from
  [difflocks.is.tue.mpg.de](https://difflocks.is.tue.mpg.de/) after registration.
* **A new animal**: `patchlocks/deploy/SURFPACK.md` and `GEOMETRY_PIPELINE.md` describe the
  preprocessing chain (capture import `import_artemis_dfa.py`, mesh registration
  `_nf_register.py`, geodesic patch cover `build_geodesic_patchpack.py`, surface patch export
  `neuralfur_export_surface_patch_dataset.py`, pose sidecar `add_v13_pose_features.py`).

Patch generation is seeded and deterministic; Blender Cycles GPU renders are not bit-exact.

## Known issues

* **Orientation reference frame in the deployed grooms.** The surface patch packs of the four
  animals were exported with `--field_pack`, so each patch's tangent frame `Tt` follows the
  strand direction of NeuralFur's released reconstruction
  (`neuralfur_export_surface_patch_dataset.py`). Step 2 fits orientation offsets relative to a
  purely geometric frame, but the grow applies them relative to `Tt`. As a result, the final
  strand directions of the archived grooms follow NeuralFur's reconstructed flow rotated by the
  fitted offset, rather than the image-fitted orientation alone. *(Resolution pending.)*
* Small rounding differences between the paper text and the result JSONs: one-view lift
  correlation −0.0849 (printed −0.09), cat >150° seam pairs 4.949 % (printed 5.0 %), EXT CI bound
  −0.365 (printed −0.37).
* Runtime numbers in the paper (pipeline minutes, grow-scaling shares) were recorded by hand;
  no timing log exists.

## Citation

To be added.

## License

To be decided. Third-party components keep their own licenses: the strand codec architecture
and initial weights come from DiffLocks; the real-animal data come from NeuralFur and ARTEMIS.
