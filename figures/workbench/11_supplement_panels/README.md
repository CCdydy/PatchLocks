# Supplement panels

Publication-scale crops of the existing whole-animal montages and the visual-gate
sheet (`global_turntable_A/B`, `nf_compare_A/B`, `visual_gate_A/B`; plus the unused
`patch_capability_A/B` when its source exists). The script changes layout only;
model outputs and source images are unchanged.

Run from the repository root (normally via `bash figures/build_paper_figures.sh supp_panels`,
which first builds the two full-resolution montages):

```bash
python figures/workbench/11_supplement_panels/code/build_supplement_panels.py
```

Inputs default to `outputs/figures/final_v16_4animals/{ALL_GLOBAL_4ANIMALS,NF_COMPARE_4ANIMALS}.png`
and `patchlocks/datasets/v16/v16_visual_gate_agx/v16_visual_gate.png`; outputs go to
`outputs/figures/supplement_panels/`. See `figures/README.md`.
