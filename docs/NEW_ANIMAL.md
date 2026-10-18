# Adding an animal (stage 3)

Inputs: a calibrated multi-view capture of one moment (one frame, all cameras) and a furless body
mesh in the capture's world frame. Everything for an animal goes under
`patchlocks/deploy/inputs/<animal>/`; `run_locked_3animals.sh` (and `pipeline/04`) pick that
folder up automatically for any animal that is not one of the four paper animals.

```
patchlocks/deploy/inputs/<animal>/
  furless.obj                          furless mesh (OBJ, capture world frame)
  nf_data/                             images_2/, masks_2/{body,hair,bald}/, orientations_2/{angles,vars}/,
                                       3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl
                                       (+ masks_2/accessory/ when the animal wears a collar)
  geopatch/geodesic_patchpack.npz      geodesic patch cover of the mesh
  surfpack/surface_patch_dataset.npz   per-patch UV charts + projections into the fit views
  surfpack/v13_pose/                   pose features for the multi-view fuser
  views.txt                            the Step-2 fit views
  furless_sub2.obj, appearance/        only for the photo-colour showcase (below)
patchlocks/deploy/refine_annotations/<animal>.env   the groom recipe, from colmap_template.env
                                                    (+ <animal>.json region labels if needed)
```

## COLMAP capture

This route was run end to end on a multi-camera COLMAP capture (`FULL_OPENCV` cameras, furless
mesh supplied in the COLMAP frame, metres). Times are for one RTX PRO 6000.

| # | Step | Command | Time |
|---|---|---|---|
| 1 | Import: undistort and halve the photos, SAM2 masks prompted by the projected mesh, structure-tensor orientation (as in the paper), invertible camera matrices | `python patchlocks/deploy/import_colmap_capture.py --colmap <capture>/colmap --mesh furless.obj --out <in>/nf_data --scale 0.5 --qa` (`--exclude` for broken cameras) | 1.5 min |
| 1b | Accessory masks, only if the animal wears a collar or harness: colour candidates in the slab along the body axis where it is worn prompt SAM2; dark parts touching it (a clasp) are added | `python patchlocks/deploy/segment_accessory_masks.py --data_dir <in>/nf_data --mesh <in>/furless.obj --axis <i> --lo <a> --hi <b>` (check `nf_data/qa/accessory_masks.jpg`) | 2 min |
| 2 | Geodesic patch cover (radius 3.5 % of the bbox diagonal, as for the paper animals) | `python patchlocks/deploy/build_geodesic_patchpack.py --mesh <in>/furless.obj --annotations "" --out <in>/geopatch --npatch 1200 --nroots 240000 --base_radius 0.035` | 4 s |
| 3 | Fit views (12, spread around the animal) and the tangent frame | `python patchlocks/deploy/prepare_fit_views_and_frame.py --data_dir <in>/nf_data --mesh <in>/furless.obj --out_dir <in>` | 10 s |
| 4 | Surface patch pack | `python patchlocks/deploy/neuralfur_export_surface_patch_dataset.py --pack <in>/geopatch/geodesic_patchpack.npz --data_dir <in>/nf_data --field_pack <in>/tangent_field_xaxis.npz --views $(cat <in>/views.txt) --out <in>/surfpack` | 15 s |
| 5 | Pose features | `python patchlocks/deploy/add_v13_pose_features.py --pack <in>/surfpack/surface_patch_dataset.npz` | 10 s |
| 6 | Recipe | copy `refine_annotations/colmap_template.env` to `refine_annotations/<animal>.env` and calibrate it (below) | — |
| 7 | Reconstruct | `ANIMALS=<animal> bash pipeline/04_reconstruct_animals.sh` | ~12 min with 3M strands and two Step-3b rounds, plus the QA renders |

Checks worth running:

* `nf_data/qa/*.jpg` (written by `--qa`): every mask should contain the whole animal and nothing else.
* The mesh must project onto the animal: with the written matrices,
  `patch_graph_opt_geo.project_row` should put nearly all mesh vertices inside the masks.
* The furless mesh must be an OBJ (the patch tools read OBJ only); convert without reordering
  vertices.

**Tangent frame.** Step 4 passes a constant world-x field so that every patch's tangent frame is
the frame Step 2 fits orientation in; the fitted orientation then reaches the groom unchanged.
Without `--field_pack` the export uses the mesh's longest axis instead, which does not match
Step 2. (The paper animals used NeuralFur's reconstructed flow here; see Known issues in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).)

### Calibrating the recipe from the photos

The groom-level strand length is a reconstruction-time input. Calibrate it against the photos
instead of by eye:

1. Grow with a guess, reusing Step 1 and Step 2 (`REUSE_STEP1=1 PATCH_PARAMS=<run>/step2/patch_params.npz`
   with `run_v2_character_mainline.sh`; only the grow reruns, a few minutes).
2. Measure how far the fur extends beyond the furless silhouette, in the photos and in the groom:
   `python patchlocks/qa/silhouette_fur_thickness.py --data_dir <in>/nf_data --mesh <in>/furless.obj --groom <run>/grow/surfpack_v2fur.npz [--surfpack <in>/surfpack/surface_patch_dataset.npz --region_prior <run>/v2_region_prior.npz]`
   (the region options add per-region numbers).
3. Pick the length (`V16_TARGET_ARCLEN`) whose thickness matches the photos' (median and 90th
   percentile); bracket it with a shorter and a longer guess.
4. Compare details at the capture cameras:
   `blender -b --factory-startup --python patchlocks/qa/render_groom_realcam.py -- --surfpack <groom> --matrices <in>/nf_data/3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl --view <v> --W <w> --H <h> --out <dir> --tag fur --srgb_colors`
   then `python patchlocks/qa/compose_realcam_compare.py --data_dir <in>/nf_data --renders <dir> --tag fur --views <v,...> --out compare.jpg`.
   A uniform length usually leaves the muzzle, ears, legs and paws too shaggy; set the region
   scales in the recipe (`REGION_DETAIL_*_SCALE`) from these comparisons.
5. Check the region labels before trusting the region scales: project them onto a few photos.
   The automatic labels are quantiles of the bounding box and can put a neck ruff and chest into
   `face`/`ears` or the whole lower leg into `paws`, so the short-fur scales land on the thickest
   coat. Write rule-based labels from a few thresholds read off the mesh
   (`patchlocks/deploy/label_patch_regions.py`) and apply them with `INDIVIDUAL_ANNOTATION_JSON`
   (a `<animal>.json` naming the npz, as `whiteTiger.json` does).
6. Refine the per-patch length against the silhouettes of all views (Step 3b:
   `SILHOUETTE_LENGTH_ITERS=2 SILHOUETTE_LENGTH_REGIONS=body,legs`). Step 2 fits length on its 12
   fit views; Step 3b (`patchlocks/deploy/calibrate_patch_length.py`) compares the
   90th-percentile fur thickness of photo and groom per patch over every capture view, multiplies
   Step 2's length by the ratio (shrunk towards 1 for patches with few samples, smoothed, clipped
   to 0.7-1.6) and grows again. With `SILHOUETTE_LENGTH_EXCLUDE=accessory` the pixels around a
   collar, which stands off the fur, are left out. In our tests two rounds brought the body and
   leg fur thickness to within a few millimetres of the photos'.
7. Density: `TARGET_STRANDS` (3M in the template; the paper animals use 1.2M). More strands give
   a denser, smoother coat.

## Showcase appearance: photo colours

The pipeline colours each strand with its patch's photo-projected colour, which flattens the
coat's gradients (a patch is 3-4 cm across). For a showcase in the photos' own colours (display
only; the groom geometry is unchanged):

1. Accessory masks (step 1b above), so a collar is never baked into the coat.
2. A denser mesh for colour, the same surface subdivided twice:
   `python -c "import trimesh; m = trimesh.load('<in>/furless.obj', process=False); v, f = trimesh.remesh.subdivide(m.vertices, m.faces); v, f = trimesh.remesh.subdivide(v, f); trimesh.Trimesh(v, f, process=False).export('<in>/furless_sub2.obj')"`
3. Bake the photo colours onto it, leaving the accessory out:
   `python patchlocks/deploy/bake_mesh_albedo_robust.py --mesh <in>/furless_sub2.obj --data_dir <in>/nf_data --views all --exclude_dir accessory --zbuf_w 700 --out <in>/appearance/albedo_sub2_srgb.npy`
4. Colour every strand from its root:
   `python patchlocks/deploy/color_groom_from_albedo.py --groom <run>/grow/surfpack_v2fur.npz --mesh <in>/furless_sub2.obj --albedo <in>/appearance/albedo_sub2_srgb.npy --out <run>/grow/surfpack_v2fur_photo.npz`
5. Render at the capture cameras with the skin under the fur:
   `blender -b --factory-startup --python patchlocks/qa/render_groom_realcam.py -- --surfpack <run>/grow/surfpack_v2fur_photo.npz --matrices <in>/nf_data/3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl --view <v> --W <w> --H <h> --out <dir> --tag fur --subsample 0 --srgb_colors --mesh <in>/furless_sub2.obj --mesh_colors <in>/appearance/albedo_sub2_srgb.npy --fur_shader fiber --light_rig camera --world_rgb 0.9,0.9,0.9 --world_strength 1.3 --key_scale 0.3 --view_transform Standard --exposure 0.8`

Why these settings:

* Skin under the fur: without it the backdrop shows between the strands and the coat turns grey.
* `--fur_shader fiber`: the Principled Hair BSDF's white specular lobe renders a saturated coat
  lighter and paler than its baked colour; a diffuse fibre keeps the hue. Set `--exposure` so the
  median luminance over a few views around the animal matches the photos'.
* `--light_rig camera`: the default rig offsets the lights along world axes and points them along
  world -Z, which in a COLMAP frame faces away from the animal for cameras behind it.

Known limits: dark facial features such as the eyes may not survive the bake where the furless
mesh lies inside the surface the photos see (the views then disagree and the robust average drops
them), and where the mesh's shape differs from the photographed one (a straight tail for a
curled one) the colour under it is a guess.

## The four paper animals (ARTEMIS / NeuralFur captures)

The paper animals ship preprocessed (`derived` in [DATA.md](DATA.md)). **Most of these commands
were reconstructed from file metadata and notes; they were not rerun for this release.** Each step
is marked *recorded* (the original command line survives) or *reconstructed*.

| # | Step | Command | Status |
|---|---|---|---|
| a | Extract one frame from the ARTEMIS archive | `unzip -j <zip> '<animal>/img/<motion>/<frame>/*' -d patchlocks/deploy/neuralfur_<a>_official/realviews/frame<F>_raw` (+ `CamPose.inf`, `Intrinsic.inf`) | recorded in notes |
| b | Convert to the NeuralFur layout (`images_2`, `masks_2/{body,hair}`, `orientations_2/{angles,vars}`, camera matrices) | `python patchlocks/deploy/import_artemis_dfa.py --raw <artemis_raw> --motion <motion> --frame <F> --out <nf_data_dir>` | reconstructed |
| c | Register the furless mesh to the capture frame and re-bake camera matrices (non-singular, needed by step e) | `python patchlocks/deploy/_nf_register.py --furless <mesh> --raw_cams <dir> --img_dir <frame_raw> --out <realviews>`, then `python patchlocks/deploy/_nf_rebake_matrices.py --furless_to_world <realviews>/_furless_to_world.npz --raw_cams <dir> --out_pkl <nf_data_dir>/3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl` (the beagle used the `_beagle_*` variants) | reconstructed |
| d | Geodesic patch cover of the furless mesh | `python patchlocks/deploy/build_geodesic_patchpack.py --mesh <mesh> --annotations "" --out patchlocks/deploy/neuralfur_<a>_geopatch_fine --nroots 240000 --base_radius 0.035` (beagle: 260000 / 0.033; seed 7) | reconstructed from the stored metadata; `--npatch` unknown |
| e | Surface patch pack | `python patchlocks/deploy/neuralfur_export_surface_patch_dataset.py --pack <geopack> --data_dir <nf_data_dir> --field_pack <animal>_surface_field.npz --out <surfpack_dir> [--views 0000,0003,...,0033]` | panda and white tiger recorded; others reconstructed |
| f | Pose features | `python patchlocks/deploy/add_v13_pose_features.py --pack <surfpack_dir>/surface_patch_dataset.npz` | white tiger recorded |

Step e used `--field_pack <animal>_surface_field.npz`, NeuralFur's reconstructed flow, as the
patch tangent frame; for a new animal use the constant frame from step 3 of the COLMAP route.
