# 07_patch_character

Paper label: `fig:patch`  
Published image: `PatchLocks_paper/Figures/final_v16_4animals/ALL_PATCHES_AND_CHARACTER_TIGHT.png`

## Contents

- `final.png`: editable working copy of the currently published figure.
- `code/`: figure-specific composition and rendering code copied from the pipeline.
- `publish.sh`: copies `final.png` back to the exact path used by the paper.

## Code

- `code/_build_patchfig_dark.py` (source: `patchlocks/deploy/_build_patchfig_dark.py`)

## Source assets

- `v2_results/refine_v16_fluffy/*/canon_swatch_reliable_straight.npz`
- `v2_results/refine_v16_fluffy/*/proposals_maps/`

## Rebuild

Run from the repository root. The copied scripts retain repository-relative paths:

```bash
bash PatchLocks_paper/figure_workbench/07_patch_character/rebuild.sh
```

Expanded build command:

```bash
/home/rog/miniconda3/envs/difflocks/bin/python PatchLocks_paper/figure_workbench/07_patch_character/code/_build_patchfig_dark.py
```

After manual refinement of `final.png`:

```bash
bash PatchLocks_paper/figure_workbench/07_patch_character/publish.sh
```

Do not run `sync_workbench.py` after making local script edits unless you intend
to refresh these copies from the pipeline sources.
