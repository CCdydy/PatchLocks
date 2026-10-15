# 02_teaser_four_animals

> **Current rebuild commands:** see [`figures/README.md`](../../README.md) and `bash figures/build_paper_figures.sh`. This README is the archived workbench note; the `rebuild.sh` / `publish.sh` / `code/` copies it mentions were not carried over unless present here.

Paper label: `fig:teaser`  
Final paper version: `Figures/final_v16_4animals/TEASER_4ANIMALS_CLEAN.png`, built by
`code/build_teaser_clean.py` (camera-relative soft-champagne renders, no labels). The
`TEASER_4ANIMALS_SIDE.png` layout described below is an earlier version.  
Published image: `PatchLocks_paper/Figures/final_v16_4animals/TEASER_4ANIMALS_SIDE.png`

## Contents

- `final.png`: editable working copy of the currently published figure.
- `code/`: figure-specific composition and rendering code copied from the pipeline.
- `publish.sh`: copies `final.png` back to the exact path used by the paper.

## Code

- `code/_build_paper_figs_dark.py` (source: `patchlocks/deploy/_build_paper_figs_dark.py`)
- `code/_render_teaser_color_rgba.sh` (source: `patchlocks/deploy/_render_teaser_color_rgba.sh`)
- `code/paper_render_style.env` (source: `patchlocks/deploy/paper_render_style.env`)
- `code/render_dog_fur_pro.py` (source: `patchlocks/deploy/render_dog_fur_pro.py`)

## Source assets

- `v2_results/refine_v16_fluffy/teaser_color_rgba/`

## Rebuild

Run from the repository root. The copied scripts retain repository-relative paths:

```bash
bash PatchLocks_paper/figure_workbench/02_teaser_four_animals/rebuild.sh
```

Expanded build command:

```bash
python PatchLocks_paper/figure_workbench/02_teaser_four_animals/code/_build_paper_figs_dark.py teaser
```

After manual refinement of `final.png`:

```bash
bash PatchLocks_paper/figure_workbench/02_teaser_four_animals/publish.sh
```

Do not run `sync_workbench.py` after making local script edits unless you intend
to refresh these copies from the pipeline sources.
