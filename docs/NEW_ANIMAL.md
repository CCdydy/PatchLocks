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
| 1 | Import: undistort and halve the photos, SAM2 masks prompted by the projected mesh, structure-tensor orientation (as in the paper), invertible camera matrices | `python patchlocks/deploy/import_colmap_capture.py --colmap <capture>/colmap --mesh furless.obj --out <in>/nf_data --scale 0.5 --up=<axis> --qa` (`--up=-y` etc., the frame's up axis: negative prompts on the floor keep a stage out of the masks; `--exclude` for broken cameras) | 1.5 min |
| 1b | Accessory masks, only if the animal wears a collar or harness: colour candidates in the slab along the body axis where it is worn prompt SAM2; dark parts touching it (a clasp) are added | `python patchlocks/deploy/segment_accessory_masks.py --data_dir <in>/nf_data --mesh <in>/furless.obj --axis <i> --lo <a> --hi <b>` (check `nf_data/qa/accessory_masks.jpg`) | 2 min |
| 1c | Fit the furless mesh to the photos (below): label its vertices by region, then move it only where the photos agree it is wrong | `python patchlocks/deploy/label_patch_regions.py --mesh <in>/furless.obj --up=<axis> --forward=<axis> <thresholds> --out <in>/furless_regions.npz`, then `python patchlocks/deploy/fit_furless_mesh.py --mesh <in>/furless.obj --data_dir <in>/nf_data --regions <in>/furless_regions.npz --up=<axis> --out_dir <in>/furless_fit` | 5 min |
| 2 | Geodesic patch cover (radius 3.5 % of the bbox diagonal, as for the paper animals) | `python patchlocks/deploy/build_geodesic_patchpack.py --mesh <in>/furless.obj --annotations "" --out <in>/geopatch --npatch 1200 --nroots 240000 --base_radius 0.035` (add `--nofur_small_components 0.05` when the mesh models eyes or a tongue as separate pieces: they then grow no fur) | 4 s |
| 3 | Fit views (12, spread around the animal) and the tangent frame | `python patchlocks/deploy/prepare_fit_views_and_frame.py --data_dir <in>/nf_data --mesh <in>/furless.obj --out_dir <in>` | 10 s |
| 4 | Surface patch pack | `python patchlocks/deploy/neuralfur_export_surface_patch_dataset.py --pack <in>/geopatch/geodesic_patchpack.npz --data_dir <in>/nf_data --field_pack <in>/tangent_field_xaxis.npz --views $(cat <in>/views.txt) --out <in>/surfpack` | 15 s |
| 5 | Pose features | `python patchlocks/deploy/add_v13_pose_features.py --pack <in>/surfpack/surface_patch_dataset.npz` | 10 s |
| 6 | Recipe | copy `refine_annotations/colmap_template.env` to `refine_annotations/<animal>.env` and calibrate it (below) | — |
| 7 | Reconstruct | `ANIMALS=<animal> bash pipeline/04_reconstruct_animals.sh` | ~12 min with 3M strands and two Step-3b rounds, plus the QA renders |

Checks worth running:

* `nf_data/qa/*.jpg` (written by `--qa`): every mask should contain the whole animal and nothing else.
  SAM2 is given the mask candidate that covers the projected mesh, the largest one up to
  `--max_mask_over_mesh` times the mesh's area, so long fur and a tail standing off the mesh are
  kept; a mask that takes in the floor or a stage needs `--up`.
* The mesh must project onto the animal: with the written matrices,
  `patch_graph_opt_geo.project_row` should put nearly all mesh vertices inside the masks.
* The furless mesh must be an OBJ (the patch tools read OBJ only); convert without reordering
  vertices.

**Fitting the furless mesh (step 1c).** The photos see the coat, not the body, so each view can
only say two things about the furless mesh: every vertex must lie inside the animal's silhouette
under at least the thinnest plausible fur of its region (`--band region:d_min:d_max`, metres), and
the outline of the body may not leave more than the thickest plausible fur between it and the
silhouette. `fit_furless_mesh.py` visits the views in small random batches (the worst-fitting
more often) and moves one shared mesh; an as-rigid-as-possible energy keeps the template's anatomy
and separate small pieces (eyeballs) follow rigidly. Silhouettes also contain long fringes, a tail
lying against a hip, shadows and segmentation errors, so the mesh is pushed outwards only where
most views that see a vertex on the outline agree, containment tolerates the mask accuracy
(3 mm), and with `--up` no vertex sinks below the floor and cameras below the floor (which see the
paws past the edge of the stage) do not test points near it. `--landmarks` pulls vertex groups
to triangulated points (eyes, nose). It writes `furless_fit.obj` (the template's vertex order, so
UVs and labels still apply), `furless_fit_fine.obj` (once Loop-subdivided and refined), an overlay
per view in `qa/` (silhouette green, template outline blue, fit red) and `report.json`. Use the fit
as `furless.obj` (keep the template). Its `outline_gap_mm_p10_50_90_by_region` is the fur depth the
photos show around the body, per region: a check for the recipe's strand length. The fit corrects
shape locally; it does not repose a limb or a tail (that needs the template's rig). On our capture
the supplied mesh already agreed with all views to within a few millimetres on the body and legs.

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
   fit views; Step 3b (`patchlocks/deploy/calibrate_patch_length.py`) compares a percentile
   (`SILHOUETTE_LENGTH_PCT`, 90 in the template) of the fur thickness of photo and groom per patch
   over every capture view, multiplies Step 2's length by the ratio (shrunk towards 1 for patches
   with few samples, smoothed, clipped to 0.7-1.6) and grows again. A lower percentile (60)
   follows the bulk of the coat rather than its outer fringe and gives a shorter coat; in our
   side-by-side comparisons with the photos the 90th percentile looked closer. With `SILHOUETTE_LENGTH_EXCLUDE=accessory` the pixels around a collar, which stands off
   the fur, are left out. In our tests two rounds brought the body and leg fur thickness to within
   a few millimetres of the photos'.
7. Strand shape from the diffusion (Step 3c: `NATURALIZE_REGIONS=body,legs`). The grow stamps one
   primitive everywhere, synthesized from the statistics of a few of the straightest reliable
   patches, so every strand comes out nearly straight, while the diffusion predicts for every
   patch strands that bend and wave. `patchlocks/deploy/transfer_diffusion_shape.py` decodes each
   patch's predicted strands from Step 1's latent maps, keeps the plausible ones (no loops or
   hooks) as templates, and gives every grown strand the bend of a template from its patch while
   it keeps its chord, so direction and reach still come from the multi-view fit and the
   silhouette refinement; patches with too few plausible strands borrow from the nearest patch
   that has enough. `STRAND_SHAPE_STRENGTH` (0.7) weights this prior against a straight strand.
   In our renders it turned long, fluffy fur (the hind legs' feathering) from parallel bundles
   into a soft, voluminous coat; at full strength smooth fur on the flanks got rough. The strands
   are also thinned (`NATURALIZE_RADIUS_SCALE`, 0.35: the grow's radius is set to cover the skin,
   far thicker than hair, and thick strands read as straw). `STRAND_SHAPE_SOURCE=rules` uses a
   fixed curve towards the skin, sideways sway and length spread instead (`naturalize_strands.py`).
   Two options of the script that the pipeline does not use yet: `--assign texel` gives each
   strand the prediction at its root (blended over the nearest texels) instead of a random one
   from its patch, which keeps the waves and clumps that neighbouring texels share; and several
   `--proposals` (Step 1 rerun with other `SEED` values) are averaged texel by texel, so bends the
   runs contradict cancel instead of every bend being scaled by one fixed strength.
   Judge the coat with
   `python patchlocks/qa/coat_texture_stats.py --data_dir <in>/nf_data --views <v,...> --renders <dir>/render_fur_{view}_raw.png`:
   while individual strands stand out, the local contrast of the coat (luminance spread in 9x9
   windows) is a few times the photo's; thinner strands and the render settings below halved it.
   It does not see whether long fur reads as bundles or as a soft mass, and a lower value does not
   always look more real, so look at the renders too.
8. Where the coat is too thick or too thin: `python patchlocks/qa/fur_depth_map.py --groom <run>/grow/surfpack_v2fur.npz --data_dir <in>/nf_data --regions <run>/v2_region_prior.annotated.npz --out_dir <run>/fur_depth`
   compares, per vertex, how far the fur reaches beyond the body in the photos and in the groom
   (median over the views that see the vertex on the outline), with a colour map. Step 3b matches
   each patch's outer fringe; in our tests that left the bulk of the coat too thick on the back,
   lower legs and paws. The outline is the envelope of all fur near it, so a per-vertex
   correction of strand length from this map did not converge (a tail lying over the rump or a
   neighbour's feathering holds the outline out); use it to choose region scales.
9. Against spikes only: a soft cap per region (`LENGTH_SOFT_CAPS=body:<m>,tail:<m>,...`, metres;
   length beyond the cap is kept at `LENGTH_SOFT_CAP_SLOPE`, 0.35) and a trim to the visual hull of
   the masks (`python patchlocks/deploy/carve_groom_with_hull.py --groom <run>/grow/surfpack_v2fur.npz --data_dir <in>/nf_data --out <run>/grow/surfpack_v2fur_carved.npz`,
   which shortens strands that leave the hull about their root). Both raise the silhouette
   agreement, but both cost realism, so they are not part of the template recipe: the cap shortens
   and stiffens the whole coat, and the trim ends many strands exactly on the voxel surface of the
   hull, which shows as bands of tips along the outline (the back line in a side view). Compare
   renders with the photos before adding either.
10. Density: `TARGET_STRANDS` (3M in the template; the paper animals use 1.2M). More strands give
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
   `python patchlocks/deploy/color_groom_from_albedo.py --groom <run>/grow/surfpack_v2fur_carved.npz --mesh <in>/furless_sub2.obj --albedo <in>/appearance/albedo_sub2_srgb.npy --out <run>/grow/surfpack_v2fur_photo.npz`
   To keep the eyes, mouth and nose visible, add `--expose_small_components 0.05` (strands next to
   the mesh's separate small pieces, eyeballs and tongue) and `--expose_point x,y,z,r` per feature
   that is part of the main surface (the nose): strands rooted within `--expose_bald` of a feature
   are removed and shortened up to `--expose_soft`.
5. Render at the capture cameras with the skin under the fur:
   `blender -b --factory-startup --python patchlocks/qa/render_groom_realcam.py -- --surfpack <run>/grow/surfpack_v2fur_photo.npz --matrices <in>/nf_data/3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl --view <v> --W <w> --H <h> --out <dir> --tag fur --subsample 0 --srgb_colors --mesh <in>/furless_sub2.obj --mesh_colors <in>/appearance/albedo_sub2_srgb.npy --fur_shader fiber --light_rig camera --world_rgb 0.9,0.9,0.9 --world_strength 1.3 --key_scale 0.3 --view_transform Standard --exposure 0.8 --filter_width 2.5 --bounces 16`
   (`--mesh_roughness <r.npy>` and `--mesh_specular <s.npy>` give the skin a per-vertex roughness
   and specular level: glossy eyes, and a dark nose that does not mirror the bright, uniform world
   (at the default specular level it turns grey). OptiX builds the strands' acceleration structure
   on the GPU, so a dense groom needs a fraction of the host memory it takes with `--device cuda`;
   fall back to CUDA if OptiX faults with an illegal address.)

Why these settings:

* Skin under the fur: without it the backdrop shows between the strands and the coat turns grey.
* `--filter_width 2.5 --bounces 16`: the photos carry lens blur, and light needs many bounces
  between the strands of a dense, light coat; Blender's defaults (1.5 px, 4 diffuse bounces) give
  a harsher, greyer coat than the photos.
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
