# 09_patch_capability_ablation

Paper label: `fig:patch_visual_ablation`  
Published image: `PatchLocks_paper/Figures/patch_capability_v16_clean.png`

## Contents

- `final.png`: editable working copy of the currently published figure.
- `code/`: figure-specific composition and rendering code copied from the pipeline.
- `publish.sh`: copies `final.png` back to the exact path used by the paper.

## Code

- `code/_build_capability_clean.py` (source: `patchlocks/deploy/_build_capability_clean.py`)
- `code/render_patch_samecam.py` (source: `patchlocks/qa/render_patch_samecam.py`)

## Source assets

- `v2_results/paper_cover_patch/dataset_3coats/`

## Rebuild

Run from the repository root. The copied scripts retain repository-relative paths:

```bash
bash PatchLocks_paper/figure_workbench/09_patch_capability_ablation/rebuild.sh
```

Expanded build command:

```bash
python PatchLocks_paper/figure_workbench/09_patch_capability_ablation/code/_build_capability_clean.py
```

After manual refinement of `final.png`:

```bash
bash PatchLocks_paper/figure_workbench/09_patch_capability_ablation/publish.sh
```

Do not run `sync_workbench.py` after making local script edits unless you intend
to refresh these copies from the pipeline sources.
