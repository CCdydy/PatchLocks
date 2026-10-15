# Reproducibility report

All checks below were run on the original machine (NVIDIA RTX PRO 6000 Blackwell, CUDA 12.8,
the versions in `requirements.txt`) from a clean checkout of this repository with the data linked
by `pipeline/00_link_data.sh`, and compared with the archived outputs that the paper was made from.

## Whole-animal reconstructions (stage 4)

`pipeline/04_reconstruct_animals.sh` with default settings, compared per step:

| Step | Comparison with the archived paper run |
|---|---|
| 1 diffusion predictions (42 arrays) | bit-identical |
| 1b–1d region prior, character, swatch | bit-identical (only stored file paths differ) |
| 2 orientation/length fit | GPU float non-determinism: mean \|Δ\| 3e-4 rad (max 0.05) orientation, 0.4 % (max 7 %) length |
| 3 groom | same strand count and roots for every animal (beagle 1,153,092; white tiger 1,123,693; panda 1,156,964; cat 1,200,215); beagle tip displacement median 2.7e-6 (0.02 % of strand length), 99th percentile 4e-4; length percentiles equal to four decimals |
| renders | visually indistinguishable; PSNR 34–38 dB on 1536 px close-ups, 48–50 dB full body |

Rendering an *archived* groom with the paper's Blender command reproduced the paper render pixel
for pixel, so the remaining render differences come only from the step-2 float differences.

**Panda recipes.** The paper uses two panda grooms. The main one (`refine_v16_fluffy`: teaser,
close-ups, Fig. 3, `tab:animals`) was grown with a random torso loft; `panda.env` now reproduces
it (strand tips within 0.14 % of strand length at the median, 2.4 % at the 99th percentile;
close-up renders at PSNR 32–35 dB, visually indistinguishable). The supplementary turntables
and the four-animal NeuralFur comparison use a later smooth-loft variant
(`refine_v16_paper_smooth`), reproduced approximately by `panda_paper_smooth.env`
(`RECIPE_SUFFIX=_paper_smooth`): 1,156,964 strands versus the archived 1,157,462 (0.04 %).
The original package shipped only the later smooth recipe as `panda.env`, which does not
reproduce the main paper panda.

## Figures (stage 7)

All 12 script-built figures and the 4 full-resolution intermediates are byte-identical to the
published files (`figures/verify_paper_figures.py`). The pipeline overview was drawn in draw.io.

## Tables (stage 6)

The result JSONs the paper's numbers were taken from are included; the protocol is in
[EVALUATION.md](EVALUATION.md). Recomputing everything with `pipeline/06_evaluate_paper_tables.sh`
reproduced every table:

| Result | Recomputed vs archived |
|---|---|
| `tab:capability` (Table 1, n=80) | byte-identical JSON (1.381 → 0.631) |
| `tab:recipe` (E1 row) | byte-identical JSON (0.674) |
| `tab:baselines` ID (n=160) and EXT (n=80), bootstrap CIs | all 11,784 numbers identical |
| one-view ablation, val80 three-arm evaluation | identical (largest difference 1.8e-7) |
| `tab:ceiling` | identical |
| character Spearman correlations, `tab:identifiability` | byte-identical JSONs |
| `tab:seams` | identical |

The newer `eval_3arm.py` additionally reports a `v16s` aggregate absent from the archived val80
file; all shared values match. `tab:grow_scaling` reports wall-clock timings and was not rerun.

**GPU memory.** The Table 1 and `tab:recipe` scripts score the single-view baseline in one
subprocess per patch and skip a patch whose subprocess fails. On a GPU shared with other jobs,
out-of-memory failures silently reduce n (a first run reached n=77 and shifted the means by
~0.01). Check that the log reports `n=80`.

Differences between the paper text and the archived JSONs (rounding):

* one-view lift correlation: JSON −0.0849, paper −0.09;
* cat seam pairs above 150°: JSON 4.949 %, paper 5.0 %;
* extrapolation CI bound vs the deterministic latent: JSON −0.365, paper −0.37.

The runtime figures in the paper (pipeline minutes and grow-scaling shares) were recorded by hand;
no timing log exists. `tab:seams` is computed on the `refine_v16_4animals` grooms, an earlier
deployment that differs from the figure grooms (`refine_v16_fluffy`) only in grow-time region
policy.

## Known issues

### Orientation reference frame of the deployed grooms

The surface patch packs of the four animals were exported with `--field_pack`, so each patch's
tangent frame `Tt` follows the strand direction of NeuralFur's released reconstruction
(`neuralfur_export_surface_patch_dataset.py`, `face_flow`). Step 2 fits the orientation offset
relative to a purely geometric frame (the projected world x axis, `patch_graph_opt_geo.py`), but
the grow applies that offset relative to `Tt` (`grow_v2_region_prior.py`). Measured on the
archived grooms:

| | final strands vs NeuralFur flow | final strands vs step-2 fitted orientation |
|---|---|---|
| beagle | median 22.5° | median 63.9° |
| white tiger | 21.0° | 65.0° |
| panda | 26.2° | 61.4° |
| cat | 24.2° | 68.5° |

(axial angles; uniformly random orientation gives a 45° median). The final orientation equals
`Tt` rotated by the fitted offset to within 0.2° (median). The archived grooms therefore take
their orientation reference from NeuralFur's reconstruction rather than from the image fit alone.
*Resolution pending.*
