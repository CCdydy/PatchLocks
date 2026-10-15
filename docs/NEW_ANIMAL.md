# Preprocessing an animal (stage 3)

The four paper animals ship preprocessed (`derived` in [DATA.md](DATA.md)). This page describes
how those inputs were produced, for adding another animal. **Most of these commands were
reconstructed from file metadata and notes; they were not rerun for this release.** Each step is
marked *recorded* (the original command line survives) or *reconstructed*.

Inputs: a calibrated multi-view capture of one frame (RGB, body and hair masks, orientation maps)
and a furless body mesh. The paper uses NeuralFur's furless meshes and the ARTEMIS captures.

| # | Step | Command | Status |
|---|---|---|---|
| a | Extract one frame from the ARTEMIS archive | `unzip -j <zip> '<animal>/img/<motion>/<frame>/*' -d patchlocks/deploy/neuralfur_<a>_official/realviews/frame<F>_raw` (+ `CamPose.inf`, `Intrinsic.inf`) | recorded in notes |
| b | Convert to the NeuralFur layout (`images_2`, `masks_2/{body,hair}`, `orientations_2/{angles,vars}`, camera matrices) | `python patchlocks/deploy/import_artemis_dfa.py --raw <artemis_raw> --motion <motion> --frame <F> --out <nf_data_dir>` | reconstructed |
| c | Register the furless mesh to the capture frame and re-bake camera matrices | `python patchlocks/deploy/_nf_register.py --furless <mesh> --raw_cams <dir> --img_dir <frame_raw> --out <realviews>`, then `python patchlocks/deploy/_nf_rebake_matrices.py --furless_to_world <realviews>/_furless_to_world.npz --raw_cams <dir> --out_pkl <nf_data_dir>/3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl` (the beagle used the `_beagle_*` variants) | reconstructed |
| d | Geodesic patch cover of the furless mesh | `python patchlocks/deploy/build_geodesic_patchpack.py --mesh <mesh> --annotations "" --out patchlocks/deploy/neuralfur_<a>_geopatch_fine --nroots 240000 --base_radius 0.035` (beagle: 260000 / 0.033; seed 7) | reconstructed from the stored metadata; `--npatch` unknown |
| e | Surface patch pack: per-patch UV charts, visibility and projections into every view | `python patchlocks/deploy/neuralfur_export_surface_patch_dataset.py --pack <geopack> --data_dir <nf_data_dir> --out <surfpack_dir> [--views 0000,0003,...,0033]` | panda and white tiger recorded; others reconstructed |
| f | Pose features for the multi-view fuser | `python patchlocks/deploy/add_v13_pose_features.py --pack <surfpack_dir>/surface_patch_dataset.npz` | white tiger recorded |
| g | Recipe | copy `patchlocks/deploy/refine_annotations/cat.env` (the plainest recipe) to `<a>.env`, add the animal's paths to `run_locked_3animals.sh` | — |

> **Note on step e.** The paper packs were exported with
> `--field_pack <animal>_surface_field.npz`, which sets each patch's tangent frame from the strand
> direction of NeuralFur's released reconstruction. Do not pass `--field_pack` for a new animal:
> it requires an existing strand reconstruction and makes the grow orientation depend on it
> (see Known issues in [REPRODUCIBILITY.md](REPRODUCIBILITY.md)).
