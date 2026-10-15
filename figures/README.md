# Paper figures

Every figure in `PatchLocks.tex` (main paper) and `PatchLocks_supp.tex` (supplement) can be
rebuilt from this repository, except the hand-drawn pipeline overview. The figure builders
are CPU-only composition scripts (PIL / numpy / matplotlib). They read archived renders and
grooms from the data linked by `scripts/link_data.sh` (see `data/manifest.txt`), and they
never run Blender or a GPU. The Blender renders they consume are listed in the
"upstream" column, but rebuilding the figures does not re-render them.

```bash
bash scripts/link_data.sh /path/to/DATA_ROOT          # once
bash figures/build_paper_figures.sh                   # all figures -> outputs/figures/ (~40 s)
bash figures/build_paper_figures.sh teaser nfcompare  # or a subset of targets
python figures/verify_paper_figures.py --paper-figures /path/to/PatchLocks_paper/Figures \
    [--intermediates /path/to/PatchLocks_paper_unused_20260827/Figures]
```

`outputs/figures/` mirrors the paper's `Figures/` folder: it uses the same file names and the
same `final_v16_4animals/`, `supplement_panels/` and `_supp60/` sub-directories. You can drop
it in as `Figures/`. The supplement uses `\graphicspath{{Figures/_supp60/}{Figures/}}`, so the
60 % copies in `_supp60/` shadow the full-resolution montages, which are also written to
`final_v16_4animals/`. All commands run from the repository root, and `PY` selects the
interpreter (default: `python`).

## Figure map

| # | Paper file (`Figures/...`) | Label | Target | Builder | Direct inputs | Upstream (Blender / GPU, not rerun) |
|---|---|---|---|---|---|---|
| main 1 | `final_v16_4animals/TEASER_4ANIMALS_CLEAN.png` | `fig:teaser` | `teaser` | `figures/workbench/02_teaser_four_animals/code/build_teaser_clean.py` | `v2_results/refine_v16_fluffy/teaser_camera_relative_preview/{beagle,whiteTiger,panda,cat}/camera_relative_soft_champagne_1600.png` | `patchlocks/deploy/render_dog_fur_pro.py --camera_relative_light --camera_relative_soft` (ad-hoc runs, no wrapper script) |
| main 2 | `pipeline.png` | `fig:overview` | - (manual) | draw.io: `figures/pipeline/patchlocks_pipeline_overview_v4.drawio` | embedded in the .drawio | - |
| main 3 | `final_v16_4animals/NF_COMPARE_2ANIMALS.png` | `fig:nfcompare` | `nfcompare` | `patchlocks/deploy/_build_nfcompare_nf.py` with `NF_ANIMALS=beagle,panda NF_OURS=panda=render_panda_champ_ours_len120 NF_OURS_OWN_CROP=panda NF_PNG_OPTIMIZE=1` | `patchlocks/deploy/nf_official_render/render_{beagle,panda}_champ/furview_1.png`, `render_beagle_champ_ours/furview_1.png`, `render_panda_champ_ours_len120/furview_1.png`; input photos `neuralfur_beagle_official/realviews/nf_data_f210/{images_2,masks_2/body}/0021.png`, `datasets/neuralfur/extracted/data/Artemis/panda_processed_GH2/walk/images_2/0009.png` | `patchlocks/deploy/_render_nfcompare_champagne.sh` (panda Ours at length x1.2) |
| main 4 | `fig_baselines_qual.png` | `fig:baselines_qual` | `baselines_qual` | `patchlocks/experiments/rebuttal_2027/paper_draft/compose_qual.py` (defaults = paper: `grid5_B`, rows curly,long,wavy,wiry) | `patchlocks/experiments/rebuttal_2027/grid5_B/{picks.json,<coat>/render_{gt,v16,det,d7s}/rgb_0.png}`, `testB_renders/<pick>/rgb_0.png` | `patchlocks/experiments/rebuttal_2027/grid_dump_3arm.py` on the frozen test set (GPU + Blender) |
| supp | `supplement_panels/global_turntable_A.jpg`, `_B.jpg` | `fig:qual_full`, `fig:qual_full_more` | `supp_panels` | `patchlocks/deploy/_build_qual_nf.py` -> `final_v16_4animals/ALL_GLOBAL_4ANIMALS.png` -> `figures/workbench/11_supplement_panels/code/build_supplement_panels.py` | `v2_results/refine_v16_fluffy/global4_champagne/{beagle,whiteTiger,cat}/furview_{0..3}.png`, `v2_results/refine_v16_paper_smooth/global4_champagne/panda/furview_{0..3}.png` | `patchlocks/deploy/_render_global4_champagne.sh` (panda: `SRC=v2_results/refine_v16_paper_smooth ANIMALS=panda`) |
| supp | `supplement_panels/nf_compare_A.jpg`, `_B.jpg` | `fig:nfcompare_full`, `fig:nfcompare_full_more` | `supp_panels` | `_build_nfcompare_nf.py` (defaults) -> `final_v16_4animals/NF_COMPARE_4ANIMALS.png` -> splitter | `nf_official_render/render_<a>_champ{,_ours}/furview_1.png` (panda Ours = `refine_v16_paper_smooth` groom, length x1.0) + the four input photos | `_render_nfcompare_champagne.sh` |
| supp | `_supp60/final_v16_4animals/BASELINE_GRID_NF.png` | `fig:baselinegrid` | `baselinegrid` | `patchlocks/deploy/_build_baseline_grid_nf.py` -> `figures/make_supp60.py` | `_baselines/surf_render/<a>_neus/furview_1.png`, `<a>_genzoo/furview_0.png`, `v2_results/refine_v16_fluffy/global4_champagne/<a>/furview_1.png`, input photos | `figures/workbench/06_baseline_grid/code/assets/` (NeuS / GenZoo surface renders), `_render_global4_champagne.sh` |
| supp | `_supp60/final_v16_4animals/ALL_PATCHES_AND_CHARACTER_TIGHT.png` | `fig:patch` | `patchfig` | `patchlocks/deploy/_build_patchfig_dark.py` -> `figures/make_supp60.py` | `v2_results/refine_v16_fluffy/<a>/{canon_swatch_reliable_straight.npz,character_prior.npz,grow/surfpack_v2fur.npz}` + the surface patch packs and capture images they reference (`patchlocks/deploy/neuralfur_<a>_official/...`, ~5 GB) | the whole-animal pipeline (`run_v2_character_mainline.sh`) |
| supp | `PATCH_RESULT_SINGLE_COLUMN.png` | `fig:patch_result` | `patchresult` | `figures/workbench/05_detail_closeups/code/build_patch_result_single.py` | `v2_results/paper_cover_patch/beagle_p855/{metadata.json,v16_pred_patch_top_champagne.png}`, beagle surface pack + `nf_data_f210/images_2/0003.png`, `v2_results/refine_v16_fluffy/details_champagne/beagle/beagle_detail_v3.png` | `patchlocks/qa/render_pred_swatch.py` (predicted patch), `_render_details_champagne.sh` |
| supp | `supplement_panels/visual_gate_A.jpg`, `_B.jpg` | `fig:gate_a`, `fig:gate_b` | `supp_panels` | `build_supplement_panels.py` | `patchlocks/datasets/v16/v16_visual_gate_agx/v16_visual_gate.png` | `patchlocks/v16/v16_visual_gate.py` (GPU + Blender) |

`_supp60` copies use `Image.resize((int(0.6*w), int(0.6*h)), LANCZOS)` saved with
`optimize=True` (`figures/make_supp60.py`). The supplement panels are JPEGs written by the
splitter (`quality=94, subsampling=0, optimize, progressive`).

## Verification

`verify_paper_figures.py` compares each output with the published file. The last run of
`build_paper_figures.sh` + `verify_paper_figures.py --intermediates ...` matched every figure
byte for byte:

| Paper file | Result |
|---|---|
| `final_v16_4animals/TEASER_4ANIMALS_CLEAN.png` | byte-identical |
| `pipeline.png` | manual draw.io export (the reference copy in `figures/pipeline/` is byte-identical) |
| `final_v16_4animals/NF_COMPARE_2ANIMALS.png` | byte-identical |
| `fig_baselines_qual.png` | byte-identical |
| `supplement_panels/{global_turntable,nf_compare,visual_gate}_{A,B}.jpg` | byte-identical (6 files) |
| `_supp60/final_v16_4animals/{BASELINE_GRID_NF,ALL_PATCHES_AND_CHARACTER_TIGHT}.png` | byte-identical |
| `PATCH_RESULT_SINGLE_COLUMN.png` | byte-identical |
| intermediates `ALL_GLOBAL_4ANIMALS`, `NF_COMPARE_4ANIMALS`, `BASELINE_GRID_NF`, `ALL_PATCHES_AND_CHARACTER_TIGHT` (full res) | byte-identical |

## Steps that are not (fully) scripted

* **`pipeline.png` (fig:overview)** was drawn by hand in draw.io, so no script builds it.
  The source is `figures/pipeline/patchlocks_pipeline_overview_v4.drawio`, and its labels
  match the published figure. The published PNG is in the same folder as
  `pipeline_published.png`. To regenerate it, open the .drawio in draw.io (desktop 29.x) and
  use File > Export as > PNG with a white background, cropped to the diagram. The published
  file is 3744x1418. The PNG (2026-08-22) is three days older than the last save of the
  .drawio (2026-08-25), so a fresh export can differ slightly from the paper.
* **Teaser crop boxes.** The published teaser was composed ad hoc and no script survived.
  `build_teaser_clean.py` reproduces it byte for byte. Its four crop boxes (the alpha bbox plus
  a margin of about 1.1 %), the tile height of 592 px and the 4 px margins and gaps were
  recovered by fitting against the published pixels, so they are listed explicitly. The
  original margin rule is unknown. The same geometry also reproduces the sibling
  `camera_relative_shared_champagne_4animals_nolabel_ultratight_1600.png` exactly.
* **Renders and model outputs are inputs, not build steps.** All `furview_*.png`, detail and
  teaser renders, `grid5_B` and the visual-gate sheet come from Blender and/or GPU runs that
  are archived in the data bundle. The upstream scripts in the table document how they were
  made. Some upstream steps cannot be rerun from the shipped data:
  * the panda `refine_v16_paper_smooth` groom (continuous surface loft) is not shipped, only
    its renders (`refine_v16_paper_smooth/global4_champagne/panda`,
    `nf_official_render/render_panda_champ_ours`);
  * the NeuralFur strand subsamples `nf500k_<a>.npz` used by `_render_nfcompare_champagne.sh`
    are not shipped, only the NeuralFur renders;
  * the teaser renders were made by direct `render_dog_fur_pro.py` calls (no wrapper script);
  * `grid5_B` was presumably produced by `grid_dump_3arm.py` on `testB_renders` (its
    `picks.json` points there), but the exact command line was not recorded.

## Builder options added for the paper versions

The defaults reproduce the paper. The options only exist to rebuild other variants.

* `_build_nfcompare_nf.py`: `NF_ANIMALS`, `NF_OUT`, `NF_OURS=<animal>=<render dir>`,
  `NF_OURS_OWN_CROP=<animal>` (crop the Ours full-body cell from the Ours render's own alpha),
  and `NF_PNG_OPTIMIZE=1`.
* `_build_qual_nf.py`: `G4`, `PANDA_G4` (default `v2_results/refine_v16_paper_smooth/global4_champagne/panda`),
  `Q_OUT`, `Q_VIEWS`, `Q_TRANSPOSE`.
* `compose_qual.py`: `QUAL_ROOT` (default `grid5_B`), `QUAL_ROWS` (default `curly,long,wavy,wiry`),
  `QUAL_OUT`.
* `build_supplement_panels.py`: `SUPP_GLOBAL`, `SUPP_NF`, `SUPP_GATE`, `SUPP_CAPABILITY`, `SUPP_OUT`.
* `_build_baseline_grid_nf.py`: `BG_OUT`. `_build_patchfig_dark.py`: `PF_OUT`.
  `build_patch_result_single.py`: `PATCH_OUT`. `build_teaser_clean.py`: `TEASER_SRC`,
  `TEASER_RENDER`, `TEASER_OUT`.

Every builder writes only to `outputs/figures/`, which is gitignored. None of them writes into
the linked data directories.

## `figures/workbench/`

`figures/workbench/<NN_name>/` holds the per-figure working folders that were copied from
the paper's figure workbench: their READMEs, the figure-specific code, and the
baseline-render assets. Several of these folders describe earlier figure versions that are
no longer in the paper. The `rebuild.sh` and `publish.sh` scripts their READMEs mention were
not carried over, so use the commands above instead.
