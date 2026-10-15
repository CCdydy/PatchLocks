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
| 3 groom | same strand count and roots for every animal (beagle 1,153,092; white tiger 1,123,693; panda 1,156,964); beagle tip displacement median 2.7e-6 (0.02 % of strand length), 99th percentile 4e-4; length percentiles equal to four decimals |
| renders | visually indistinguishable; PSNR 34–38 dB on 1536 px close-ups, 48–50 dB full body |

Rendering an *archived* groom with the paper's Blender command reproduced the paper render pixel
for pixel, so the remaining render differences come only from the step-2 float differences.

## Figures (stage 7)

All 12 script-built figures and the 4 full-resolution intermediates are byte-identical to the
published files (`figures/verify_paper_figures.py`). The pipeline overview was drawn in draw.io.

## Tables (stage 6)

The result JSONs the paper's numbers were taken from are included; the protocol is in [EVALUATION.md](EVALUATION.md). Every table maps to a
command in `pipeline/06_evaluate_paper_tables.sh`. *A full recomputation from this repository
is in progress; results will be recorded here.*

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
