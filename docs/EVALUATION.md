# Evaluation protocol

How the quantitative results in the paper are computed. Scripts are in
`patchlocks/experiments/rebuttal_2027/` (the folder name is historical) and `patchlocks/v16/`;
`pipeline/06_evaluate_paper_tables.sh` runs all of them.

## Data

| Set | Size | Use | Generation |
|---|---|---|---|
| validation split of the training corpus | 240 patches (80 scored) | Table 1, one-view ablation, model selection | patch-level split of `renders_mv`, seed 0 (`v14_common.build_env`) |
| in-distribution test set B | 160 patches | Table 2 ID column, `tab:ceiling`, Spearman correlations | `gen_testset_B.sh`: the nine training coats at one tenth of the training counts (8/11/10/10/21/26/27/21/26), patch seeds 610000+, render seed 710000 |
| extrapolation test set | 80 patches | Table 2 EXT column | `gen_ood_patches.py --n_per 20`: four coats beyond the training ranges, patch seeds 810000+, render seed 910000 |

The test sets use seeds disjoint from training (patches 210000–211599, renders 310000/410000),
were generated once, and are frozen (`*_FROZEN_manifest.txt`): no patch was regenerated or
filtered, and they were never used for model selection. Targets are baked with the same codec
(`bake_mv_targets.py`, H=64) and pose features with `prepare_v13_mv_dataset.py --write-pose`.
When scoring a test set, conditioning statistics come from the training run stored in the
checkpoint (`--ckpt_cond_stats`), not from the test set.

## Predictors compared (Table 2)

| Arm | Model | Inference | Measured from |
|---|---|---|---|
| Diffusion (ours) | `patchlocks/datasets/v16/e2a_cw15` | Heun sampler, 40 steps, CFG 3.0, seed = patch index | sampled z64 latents, decoded by the codec |
| Deterministic latent | `train_det_z64.py`: same U-Net (256 channels, attention, ray-direction conditioning), single forward pass, masked L2, EMA 0.999, no condition dropout | one forward pass | predicted z64, decoded by the codec |
| Scalar regression | `train_direct7.py`: CNN on the frozen fused condition → MLP regressing the six evaluated attributes + radius ratio | one forward pass | (a) the regressed scalars; (b) a 3D patch synthesized from them with the same stroke primitive used in deployment |

Fairness constraints: all arms read identical conditions from one frozen multi-view fuser
(`e2a_cw15/b3_fuser.pt`) built once per test set in a single process; the deterministic arm
matches the diffusion U-Net's capacity and, like it, uses no decoded-parameter losses; new arms
are selected only on held-out correlation on the original validation split.
The stroke primitive consumes only length, curl, lift and radius, so straightness, tip spread and
clump of a synthesized patch come from the primitive itself; both the regressed and the
synthesized scores are reported.

## Metrics

One implementation (`v14_common.decode_params`) measures every arm. Evaluation texels are sampled
with seed 1000000 + patch index, 32 strands per texel group; a patch dropped for one arm is
dropped for all. Ground truth is the baked codec latent decoded through the same codec.

1. **Attribute accuracy:** normalized MAE and Pearson correlation for relative length,
   straightness, curl, lift ratio, tip spread and clump.
2. **Distribution fidelity:** per-patch Wasserstein-1 distance of the per-strand curvature
   distribution, and across-patch standard-deviation ratios (a value well below 1 indicates
   regression to the mean).
3. **Visual:** input | ground truth | predictions, rendered with one camera and one neutral
   material (`patchlocks/qa/render_patch_samecam.py`, figure `fig_baselines_qual`).

## Confidence intervals

`eval_3arm.py` stores per-patch values; `bootstrap_ci.py` computes 95 % intervals from 10,000
resamples (seed 0) for each arm and attribute and for the differences between arms.

## One view vs four views

`eval_k1.py` evaluates the same checkpoint with only view 0: the fuser receives one view and the
ray-direction, global and back-projection features are rebuilt from view 0 alone. The network and
its conditioning statistics were fit on four-view inputs, so this test-time ablation includes
that distribution shift.
