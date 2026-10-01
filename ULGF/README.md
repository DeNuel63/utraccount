# ULGF underwater image and video research

Layout-guided underwater image generation, training utilities, and a video-extension
research workstream linking frames, bounding boxes, classes and persistent IDs.

Based on **ULGF: A Diffusion Model-Based Image Generation Framework for Underwater
Object Detection**, credited upstream to Yaoming Zhuang, Longyu Ma, Jiaming Liu,
Yonghao Xian, Baoquan Chen, Li Li, Chengdong Wu, Wei Cui, and Zhanlin Liu.
Original model and pipeline sources are retained.

## Status

- Safe still-image inference with validated inputs and separate outputs.
- CPU layout sequences with bounded motion, stable IDs and schema validation.
- Training contract, validation and checkpoint/resume utilities; real-model GPU
  qualification remains required.
- An isolated training-free video experiment with controlled frame generation
  and motion-warping studies.

Assigned IDs and requested boxes do not prove correct generated-object identity
or localization. This repository does not claim a fully validated temporally
consistent video generator.

## Structure

| Directory | Purpose |
| --- | --- |
| `ulgf_baseline/` | Baseline configuration, validation and training utilities |
| `ulgf_video/` | Video schema, layout motion and validation |
| `pipeline_custom/`, `utils/`, `MaskLossModel/` | Model and supporting implementations |
| `scripts/` | Baseline inference and layout entry points |
| `configs/` | Dataset and experiment configuration |
| `tools/` | Launcher, preflight, reusable data preparation and evaluation |
| `tests/` | Core unit tests |
| `colab/` | Baseline notebook |
| `experiments/ulgf_training_free_video/` | Isolated experiment and frozen source snapshot |
| `docs/video_output_schema.md` | Canonical output contract |
| `assets/` | Local checkpoints and smoke inputs, excluded from Git |

## Environment

The legacy baseline targets Python 3.7.16, PyTorch 1.12.1 with CUDA 11.3, and
torchvision 0.13.1. Install the matching PyTorch build before the pinned inference
dependencies in `requirements-inference.txt`. OpenCV and ftfy are also needed.
`requirements.txt` is the upstream training list, not a newly qualified lockfile.

Training additionally requires compatible MMCV/MMDetection compiled operators.
Do not install multiple OpenCV distributions or both COCO package variants in one
environment. Check the selected environment before training:

```sh
python tools/colab_training_preflight.py --repo /path/to/ULGF-main --python /path/to/python
```

This preflight resolves wheels and checks imports/operators. It does not install
the environment or certify dataset leakage clearance or model quality.

## Baseline inference

Provide a complete trusted Diffusers-format checkpoint and paired images/YOLO labels.
Weights and dataset samples are not distributed with the code.

```sh
python scripts/generate_image_baseline.py \
  --checkpoint /path/to/checkpoint/final \
  --image-dir /path/to/images --label-dir /path/to/labels \
  --output-dir work_dirs/baseline --dry-run
```

Remove `--dry-run` to generate. Existing outputs are rejected unless overwrite is
explicitly requested. The safety checker remains enabled by default.

## Layout sequences

```sh
python scripts/generate_layout_sequence.py \
  --clip-id example --source-image /path/to/seed.jpg \
  --source-label /path/to/seed.txt \
  --output work_dirs/example/manifest.json --frames 5 --fps 10 --seed 42
python scripts/validate_video_clip.py work_dirs/example/manifest.json
python scripts/visualize_layout_sequence.py work_dirs/example/manifest.json \
  --output-dir work_dirs/example/layout-preview
```

These commands generate layouts and box canvases, not underwater video pixels.
The canonical schema uses zero-based frame indices and normalized `xyxy` boxes.
The initial layout MVP has fixed object count and full visibility.

## Training

Use an approved reviewed split, never official test as validation. Verify the
initialization provenance and real-model update/resume behavior first. Example
launcher wiring, not a declaration of training readiness:

```sh
RUOD_SPLIT_ROOT=/path/to/reviewed-split \
PRETRAINED_MODEL=/path/to/trusted-checkpoint \
OUTPUT_DIR=work_dirs/training \
DATASET_CONFIG=configs/data/ruod_256x256.py \
NUM_PROCESSES=1 GPU_IDS=0 bash tools/dist_train.sh
```

## Tests and isolated experiment

```sh
python -m unittest discover -s tests -v
```

Tensor tests may be skipped without PyTorch/Accelerate; skips do not qualify the
GPU training stack. See the experiment README for its separate commands.
Its frozen baseline is hash-verified: do not edit or prune it. Smoke inputs and
reviewed masks are excluded from Git and must be restored to their documented
paths for asset-dependent experiments. A fresh clone does not contain these inputs.

## Upload and recovery

Historical reports, review batches, packaged uploads, and task-specific tests are
preserved in `_local_archive/20260930_github_cleanup/`. The move manifest records
their original paths. Restore those paths before executing archived tools.
No historical results or model files were deleted.

`.gitignore` excludes archives, checkpoints, dataset samples, experiment outputs,
private review inputs, credentials and caches. Use a Git-based upload: manually
zipping the whole folder or uploading it through a browser bypasses these rules.
No repository was published by the cleanup.

Before public release, confirm the upstream source license and redistribution
rights for additional assets. No license is invented or granted by this cleanup.

## Acknowledgements

The original project credits [Diffusers](https://github.com/huggingface/diffusers),
[GeoDiffusion](https://github.com/KaiChen1998/GeoDiffusion), and
[SCP-Diff](https://air-discover.github.io/SCP-Diff/). RUOD redistribution conditions
must be checked separately from code ownership.
