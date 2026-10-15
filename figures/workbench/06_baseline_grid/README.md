# 06_baseline_grid

Paper label: `fig:baselinegrid`  
Published image: `PatchLocks_paper/Figures/final_v16_4animals/BASELINE_GRID_NF.png`

## Contents

- `final.png`: editable working copy of the currently published figure.
- `code/`: figure-specific composition and rendering code copied from the pipeline.
- `publish.sh`: copies `final.png` back to the exact path used by the paper.

## Code

- `code/_build_baseline_grid_nf.py` (source: `patchlocks/deploy/_build_baseline_grid_nf.py`)
- `code/_build_paper_figs_dark.py` (source: `patchlocks/deploy/_build_paper_figs_dark.py`)

## Source assets

- `_baselines/surf_render/`
- `v2_results/refine_v16_fluffy/global4_champagne/`

## Rebuild

Run from the repository root. The copied scripts retain repository-relative paths:

```bash
bash PatchLocks_paper/figure_workbench/06_baseline_grid/rebuild.sh
```

Expanded build command:

```bash
python PatchLocks_paper/figure_workbench/06_baseline_grid/code/_build_baseline_grid_nf.py
```

After manual refinement of `final.png`:

```bash
bash PatchLocks_paper/figure_workbench/06_baseline_grid/publish.sh
```

Do not run `sync_workbench.py` after making local script edits unless you intend
to refresh these copies from the pipeline sources.
