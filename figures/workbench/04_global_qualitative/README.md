# 04_global_qualitative

Paper label: `fig:qual`  
Published image: `PatchLocks_paper/Figures/final_v16_4animals/ALL_DETAIL_4ANIMALS_ROW.png`

## Contents

- `final.png`: editable working copy of the currently published figure.
- `code/`: figure-specific composition and rendering code copied from the pipeline.
- `publish.sh`: copies `final.png` back to the exact path used by the paper.

## Code

- `code/build_qual_detail_row.py` (main-paper body-detail row)
- `code/_render_global4_champagne.sh` (source: `patchlocks/deploy/_render_global4_champagne.sh`)
- `code/paper_render_style.env` (source: `patchlocks/deploy/paper_render_style.env`)
- `code/render_dog_fur_pro.py` (source: `patchlocks/deploy/render_dog_fur_pro.py`)

## Source assets

- `v2_results/refine_v16_fluffy/details_champagne/`
- `v2_results/refine_v16_paper_smooth/details_champagne/panda/` (same v16 prior and Step2, continuous surface loft replaces random torso lift)

## Rebuild

Run from the repository root. The copied scripts retain repository-relative paths:

```bash
bash PatchLocks_paper/figure_workbench/04_global_qualitative/rebuild.sh
```

Expanded build command:

```bash
python PatchLocks_paper/figure_workbench/04_global_qualitative/code/build_qual_detail_row.py
```

After manual refinement of `final.png`:

```bash
bash PatchLocks_paper/figure_workbench/04_global_qualitative/publish.sh
```

Do not run `sync_workbench.py` after making local script edits unless you intend
to refresh these copies from the pipeline sources.
