# Isolated ULGF training free video experiment

This folder is independent of normal ULGF training preparation. Nothing here
updates weights, invokes the trainer or writes to the original dataset.

## Baseline

`baseline/` is a frozen source and smoke-input copy. `baseline_snapshot.json`
records SHA-256 for each file, including copies of the trainer and configuration
for provenance. Those training files are not executed. The checkpoint is supplied
read-only at runtime and is not duplicated. Preserve this snapshot; make future
experimental modifications outside it. This is a file snapshot, not a Git commit.

## Five frame test

Use a separate Colab notebook or inference environment. Do not install experiment
dependencies into a training environment or run alongside training on one GPU.
The previously validated Python 3.7 / torch 1.12.1 CUDA 11.3 inference stack is
supported. Snapshot requirements-inference.txt plus OpenCV and ftfy are needed.
No MMCV/MMDetection training stack is required by this runner.

From this directory:

```sh
python run_five_frames.py --dry-run
python run_five_frames.py --checkpoint /content/ulgf-official-download/ruod_256_bz16/checkpoint --run-name five_frame_001
python -m unittest discover -s tests -v
```

Upload this whole experiment folder to Drive for Colab. No edits to the normal
repository are required. Local dry runs need Pillow only. Real generation needs
the complete trusted ULGF checkpoint and CUDA. Legacy .bin model deserialization
must only be used with trusted model sources; recorded hashes alone do not prove
trust or licensing.

The runner checks snapshot integrity, loads the model once, freezes parameters,
retains the safety checker and generates five independently denoised frames from
moving layouts using the same seed, camera and prompt template. It uses the
existing simple fish smoke seed; this is a functional test, not a leakage-cleared
training dataset or final evaluation.

Outputs stay under `runs/<new-name>/`: PNG frames, canonical manifest, requested
per-frame annotations, contact sheet, run status and checkpoint provenance.
Existing runs are never overwritten. Failed runs are retained with failure status.
Requested boxes are NOT verified ground truth. IDs encode the planned identity;
appearance consistency and layout fidelity require inspection. An MP4 and the
temporal correction method are subsequent work, not implemented here.

## Checklist

- [x] Separate experimental folder and hashed baseline snapshot.
- [x] Implement model-loaded-once, inference-only five-frame runner.
- [ ] Execute five real GPU generations and inspect contact sheet.
- [ ] Verify generated objects against requested boxes and identities.
- [ ] Compare temporal method against this controlled-independent baseline.

Normal training preparation remains a separate workstream.

## Single keyframe motion warping control

Run `run_warp_control.py` with the original moving-layout run as `--reference-run`
and a fresh `--run-name`. This is CPU-only: Pillow and NumPy, no checkpoint,
PyTorch, or optimization. Original source frames are hash-verified and unchanged.

The control takes frame zero and translates its selected region using the
existing five-frame trajectory, rounded to integer pixels. It preserves foreground
pixels exactly. It rejects changing object counts, changed class/ID, scale changes,
motion exceeding one-quarter canvas size, and any clipping of the mask.
Only one object is supported; inter-object occlusion is explicitly unsupported.

Default selection is the requested bounding box plus four pixels of context.
This is NOT segmentation: it can include reef pixels and exclude protruding fins.
An optional `--mask` supplies a full-size binary PNG for a manually reviewed animal
mask. Vacated pixels are filled by boundary-color propagation; that operation
does not recover hidden reef geometry and may leave visible smear or seams.

Outputs include comparison.png, five frames, masks, per-frame annotation adapters,
qc.json, and diagnostics showing source selection, background fill and exposed
repair regions. The report distinguishes requested boxes from integer-translated
source boxes; neither is promoted to verified ground truth. No MP4 is required.

Animal realism cannot be repaired through pixel translation. The current frame
zero was rejected visually and is only usable here as a mechanics diagnostic.
Default `animal_resemblance_approved` and `dataset_accepted` remain false.
To record a later human-reviewed keyframe/mask, `--approval` accepts JSON with
keyframe_sha256, mask_pixels_sha256 (decoded grayscale pixel bytes), reviewer,
animal_resemblance_approved=true, and mask_covers_complete_animal=true.
An approval never clears background-artifact or final-dataset review automatically.

- [x] CPU warping control implemented; 12 local experiment tests passed.
- [x] Normal baseline/source hashes still match the original snapshot.
- [ ] Run the control on saved Colab frames and inspect masks, repairs and motion.
- [ ] Obtain a visibly credible aquatic-animal keyframe and a complete reviewed mask.
- [ ] Accept or reject the warped sequence after visual and annotation review.
