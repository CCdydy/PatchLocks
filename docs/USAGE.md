# Download and use

Code lives on GitHub; weights and data live on Hugging Face.

| Where | What | Size |
|---|---|---|
| [github.com/CCdydy/PatchLocks](https://github.com/CCdydy/PatchLocks) | code, documentation, per-animal recipes, paper result JSONs | 30 MB |
| [huggingface.co/zengzzy/PatchLocks](https://huggingface.co/zengzzy/PatchLocks) | 9 model checkpoints | 549 MB |
| [huggingface.co/datasets/zengzzy/PatchLocks-data](https://huggingface.co/datasets/zengzzy/PatchLocks-data) | synthetic training corpus, frozen test sets, the paper's four animal reconstructions, figure inputs | 28 GB |
| [NeuralFur](https://github.com/Vanessik/NeuralFur) / [ARTEMIS](https://github.com/HaiminLuo/Artemis) | real-animal multi-view captures and furless meshes (not redistributed here) | — |

Both Hugging Face repositories use the same paths as the code repository, so every file lands
where the code expects it.

## 1. Get the code and environment

```bash
git clone https://github.com/CCdydy/PatchLocks.git && cd PatchLocks
conda create -n patchlocks python=3.11 -y && conda activate patchlocks
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt huggingface_hub
```

## 2. Download weights

```bash
bash scripts/download_weights.sh
```

This downloads the 9 checkpoints with `curl`/`wget` and verifies them against
`scripts/weights_SHA256SUMS.txt`. Alternatives:

```bash
hf download zengzzy/PatchLocks --local-dir . --exclude "README.md" ".gitattributes"   # Hugging Face CLI
```
```python
from huggingface_hub import snapshot_download                                          # Python
snapshot_download("zengzzy/PatchLocks", local_dir=".", ignore_patterns=["README.md", ".gitattributes"])
```

| Checkpoint | Used for |
|---|---|
| `patchlocks/datasets/v16/e2a_cw15/v14_best.pt` + `b3_fuser.pt` | **the paper model**; everything in stage 4 |
| `patchlocks/models/fur_strand_codec_v6upright.pt` | strand codec, used by every stage |
| `patchlocks/v2_locked/V10_PATCH_STABLE/*` | single-view baseline (Table 1) |
| `patchlocks/datasets/v14/e1_full_sm80/*` | recipe ablation (supplement) |
| `patchlocks/experiments/rebuttal_2027/{det_z64,direct7}/*` | Table 2 baselines |

## 3. Download data

Pick only what you need. The data root can be anywhere; the script links it into the repository.

```bash
bash scripts/download_data.sh /path/to/DATA_ROOT                     # everything, ~28 GB
bash scripts/download_data.sh /path/to/DATA_ROOT testsets animals    # or subsets
```

| Subset | Contents | Size | Needed for |
|---|---|---|---|
| `training` | 1,600 synthetic fur patches × 4 views, with codec latents and pose features | 13 GB | training (stage 2), Table 1, one-view ablation |
| `testsets` | frozen in-distribution (n=160) and extrapolation (n=80) test sets | 3.4 GB | Table 2, ceiling and correlation results |
| `animals` | the paper's reconstructions of the four animals (every intermediate output and the final grooms) plus their renders | 10 GB | figures, seam table, using the grooms |
| `figures` | remaining figure inputs | 0.2 GB | stage 7 |

Equivalent without the script:
`hf download zengzzy/PatchLocks-data --repo-type dataset --local-dir /path/to/DATA_ROOT --include "patchlocks/experiments/rebuttal_2027/testB_*"`,
then `bash pipeline/00_link_data.sh /path/to/DATA_ROOT`.

## 4. What you can do with which downloads

| Goal | Weights | Data subsets | Third-party data | Command |
|---|---|---|---|---|
| Use the reconstructed grooms | — | `animals` | — | see [Using a groom](#using-a-groom) |
| Rebuild the paper figures | — | `animals`, `figures` | NeuralFur renders for Fig. 3 and the baseline grid | `bash pipeline/07_build_paper_figures.sh` |
| Recompute Table 1 / Table 2 / supplement tables | all | `training`, `testsets`, `animals` | — | `bash pipeline/06_evaluate_paper_tables.sh` |
| Retrain the diffusion model | codec | `training` | — | `bash pipeline/02_train_diffusion.sh` |
| Reconstruct the four animals from photos | paper model + codec | — | captures, meshes, preprocessed patch packs | `bash pipeline/04_reconstruct_animals.sh` |
| Reconstruct a new animal | paper model + codec | — | your capture + furless mesh | [NEW_ANIMAL.md](NEW_ANIMAL.md), then stage 4 |

Rendering (stages 4, 5) needs Blender 4.5.10 at
`patchlocks/third_party/blender-4.5.10-linux-x64/` (download from
[blender.org](https://download.blender.org/release/Blender4.5/)).

## 5. Real-animal inputs (third-party)

The whole-animal pipeline reads, per animal, a calibrated multi-view capture (`images_2/`,
`masks_2/{body,hair}/`, `orientations_2/{angles,vars}/`, camera matrices), a furless mesh, a
geodesic patch cover and a surface patch pack (exact paths: `data/manifest.txt`, categories
`derived` and `thirdparty`). They come from NeuralFur's released data and the ARTEMIS captures
(CC BY-NC-SA 4.0) and are not redistributed. To rebuild them:

1. Download the NeuralFur release (furless meshes, reconstructions and the processed ARTEMIS panda)
   from [github.com/Vanessik/NeuralFur](https://github.com/Vanessik/NeuralFur) and the raw ARTEMIS
   captures from [github.com/HaiminLuo/Artemis](https://github.com/HaiminLuo/Artemis); frames used:
   beagle `s1`/210, white tiger `roaringwalk`/5, cat `walk_final`/0, panda `walk`/0.
2. Preprocess each animal with the steps in [NEW_ANIMAL.md](NEW_ANIMAL.md). Most of those
   commands were reconstructed from the original runs rather than rerun for this release.

## Using a groom

Each reconstruction is one `npz` file, e.g.
`v2_results/refine_v16_fluffy/beagle/grow/surfpack_v2fur.npz` (from the `animals` subset):

| Key | Shape | Meaning |
|---|---|---|
| `strands` | (N, 32, 3) float32 | strand polylines, root first, in the coordinate frame and units of the animal's furless mesh |
| `radius` | (N,) | strand radius |
| `color`, `color_tip` | (N, 3) | RGB in [0, 1] at root and tip, projected from the capture (the paper renders use a fixed material instead) |
| `root_face`, `root_bary` | (N,), (N, 3) | the mesh triangle and barycentric coordinates each strand is anchored to (for skinning / simulation) |
| `root_patch`, `root_uv` | (N,), (N, 2) | the surface patch the root belongs to and its position in that patch's UV chart |

N is 1,153,092 (beagle), 1,123,693 (white tiger), 1,156,964 (panda) and 1,200,215 (cat). The
file also stores every grow setting as scalar entries.

```python
import numpy as np

g = np.load("v2_results/refine_v16_fluffy/beagle/grow/surfpack_v2fur.npz")
strands = g["strands"]                      # (N, 32, 3)

# export a 10 % subset as OBJ polylines (File > Import > Wavefront in Blender)
sel = np.random.default_rng(0).choice(len(strands), len(strands) // 10, replace=False)
with open("beagle_10pct.obj", "w") as f:
    for s in strands[sel]:
        f.write("".join(f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in s))
    for i in range(len(sel)):
        f.write("l " + " ".join(str(i * 32 + 1 + j) for j in range(32)) + "\n")
```

To render a groom exactly like the paper (needs the furless mesh and Blender):

```bash
SRC=v2_results/refine_v16_fluffy ANIMALS=beagle bash pipeline/05_render_paper_views.sh
```
