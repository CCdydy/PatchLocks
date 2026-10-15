# 10_visual_gate

Paper label: `fig:gate`  
Published image: `PatchLocks_paper/Figures/v16_visual_gate_agx.png`

## Contents

- `final.png`: editable working copy of the currently published figure.
- `code/`: figure-specific composition and rendering code copied from the pipeline.
- `publish.sh`: copies `final.png` back to the exact path used by the paper.

## Code

- `code/v16_visual_gate.py` (source: `patchlocks/v16/v16_visual_gate.py`)
- `code/infer_v16_patch.py` (source: `patchlocks/v16/infer_v16_patch.py`)
- `code/render_patch_samecam.py` (source: `patchlocks/qa/render_patch_samecam.py`)

## Source assets

- `patchlocks/datasets/v11_longfur/renders_mv/`
- `patchlocks/datasets/v16/e2a_cw15/`

## Rebuild

Run from the repository root. The copied scripts retain repository-relative paths:

```bash
bash PatchLocks_paper/figure_workbench/10_visual_gate/rebuild.sh
```

Expanded build command:

```bash
/home/rog/miniconda3/envs/difflocks/bin/python PatchLocks_paper/figure_workbench/10_visual_gate/code/v16_visual_gate.py --model patchlocks/datasets/v16/e2a_cw15/v14_best.pt --renders patchlocks/datasets/v11_longfur/renders_mv --out patchlocks/datasets/v16/v16_visual_gate_agx
```

After manual refinement of `final.png`:

```bash
bash PatchLocks_paper/figure_workbench/10_visual_gate/publish.sh
```

Do not run `sync_workbench.py` after making local script edits unless you intend
to refresh these copies from the pipeline sources.
