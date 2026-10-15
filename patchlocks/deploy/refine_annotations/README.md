# Individual refinement annotations

These files are the after-Step2, per-animal cleanup layer used for paper-quality
visuals. They do not change the learned V10 patch prior. They only correct
semantic region labels and pass local grow-policy parameters for fragile parts
such as ears, paws, leg/body seams, and tails.

Usage pattern:

```bash
set -a
source patchlocks/deploy/refine_annotations/<animal>.env
set +a
bash patchlocks/deploy/run_v2_character_mainline.sh
```

`INDIVIDUAL_ANNOTATION_JSON` is applied after `build_v2_region_prior.py` and
after the optional character-field merge, before Step2/grow consume
`patch_region`. This keeps each animal's annotation traceable in the output
`v2_region_prior.annotated.npz`.

Current status:

- `whiteTiger`: uses a fixed per-patch region annotation. This corrects the
  previous tail/head confusion and separates face/ears/tail before grow.
- `beagle` and `panda`: keep automatic region labels, but have their own local
  grow parameters so paw/leg/ear/tail polish can diverge per animal.
