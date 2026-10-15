# 08_neuralfur_comparison

Paper label: `fig:nfcompare`  
Published image: `PatchLocks_paper/Figures/final_v16_4animals/NF_COMPARE_4ANIMALS.png`

## Contents

- `final.png`: editable working copy of the currently published figure.
- `code/`: figure-specific composition and rendering code copied from the pipeline.
- `publish.sh`: copies `final.png` back to the exact path used by the paper.

## Code

- `code/_build_nfcompare_nf.py` (source: `patchlocks/deploy/_build_nfcompare_nf.py`)
- `code/_render_nfcompare_champagne.sh` (source: `patchlocks/deploy/_render_nfcompare_champagne.sh`)
- `code/_render_nfcompare_hires.sh` (source: `patchlocks/deploy/_render_nfcompare_hires.sh`)
- `code/paper_render_style.env` (source: `patchlocks/deploy/paper_render_style.env`)
- `code/render_dog_fur_pro.py` (source: `patchlocks/deploy/render_dog_fur_pro.py`)

## Source assets

- `patchlocks/deploy/nf_official_render/`
- `v2_results/refine_v16_fluffy/`

## Rebuild

Run from the repository root. The copied scripts retain repository-relative paths:

```bash
bash PatchLocks_paper/figure_workbench/08_neuralfur_comparison/rebuild.sh
```

Expanded build command:

```bash
/home/rog/miniconda3/envs/difflocks/bin/python PatchLocks_paper/figure_workbench/08_neuralfur_comparison/code/_build_nfcompare_nf.py
```

After manual refinement of `final.png`:

```bash
bash PatchLocks_paper/figure_workbench/08_neuralfur_comparison/publish.sh
```

Do not run `sync_workbench.py` after making local script edits unless you intend
to refresh these copies from the pipeline sources.
