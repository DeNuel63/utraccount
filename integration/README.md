# Connect the existing models

This integration consumes existing ULGF clip pixels, runs CountGD++ on each frame,
then sends its detections to COVTrack's existing RoI feature heads and tracker.
No model training, video-generation redesign or performance evaluation is added.

See `RUNTIME_SETUP.md` for identified inference functions, checkpoint sources,
separate environment requirements and the runtime inspection commands.

## One command

Copy `config.example.json` to a personal config and replace the checkpoint,
interpreter, clip-manifest and output paths with real paths. All relative paths
are resolved against the config file's directory. Keep this checkout named
`utraccount`, as the existing COVTrack source uses that Python namespace.

```sh
python integration/run_pipeline.py --config integration/config.local.json --dry-run
python integration/run_pipeline.py --config integration/config.local.json
```

The launcher needs Python with Pillow. CountGD++ uses its own Python >=3.10
environment with its official inference dependencies and checkpoint. COVTrack
uses its separately configured MMCV/MMDetection/CLIP runtime and checkpoint.
Stages run sequentially in separate processes, releasing GPU memory between them.
Do not combine incompatible dependency stacks into one environment.

The example Drive paths are placeholders, not discovered files. Mount Drive in
Colab first and use actual mounted paths. The COVTrack repository documents the
checkpoint `ctao_public_res/ctao_public.pth` in its model distribution at
https://huggingface.co/clarkqian/COVTrack. Obtain that artifact in the COVTrack
environment and set the config path accordingly; this integration does not
download weights or install environments. Existing CLIP caches and any additional
assets required by the selected COVTrack configuration must also be available.

## ULGF handoff and optional generation

By default `clip_manifest` points to an existing ULGF schema-1.0 `manifest.json`
with actual image files. Frames must be ordered from zero; `image_path` is relative
to the manifest. Layout-only manifests without generated images cannot be used.
The existing five-frame ULGF experiment is supported without changing its frozen
baseline. To execute it first, add the following object to your config (and use a
new `run_name`):

```json
"ulgf": {
  "python": "/content/ulgf-env/bin/python",
  "checkpoint": "/content/drive/MyDrive/ULGF/checkpoint",
  "run_name": "connected-five-frames-001",
  "seed": 0,
  "steps": 100,
  "guidance": 5.0
}
```

This invokes the existing `run_five_frames.py`, which requires its documented
smoke inputs and intact snapshot. Its output manifest becomes the CountGD++ input.
No temporal quality claim is made for those independently generated frames.

## Exact handoffs

- ULGF supplies RGB image files, clip ID, consecutive frame indices, FPS and
  normalized `xyxy` reference boxes. Timestamps are derived from index/FPS.
- CountGD++ currently uses one positive text class per run. The prompt must match
  a shared class name exactly. It saves raw query output and pixel
  `[centre_x, centre_y, width, height]` detections with confidence.
- COVTrack receives the same frame image plus pixel `[left, top, right, bottom,
  confidence]` detections and class IDs. Preprocessing scales only the boxes used
  for RoI extraction; exported/tracked boxes remain in original-image pixels.
  No RPN or internal detector boxes replace the external detections.
- The external path reuses appearance and CEM heads; configured uncertainty heads
  also reuse semantic projections, fusion and (when present) location/MAN heads.
  Association thresholds and learned weights remain those of the chosen config.
- The example uses the complete RUOD class order and COVTrack's existing custom
  vocabulary mode. With `custom_vocabulary: false`, IDs must match the configured
  `roi_head.CLASSES` exactly. No source class-name file needs editing.
- Empty frames produce empty tracking records and preserve native tracker memory
  behavior. Frame zero resets the tracker; out-of-order frames are rejected.
- ULGF reference IDs/boxes are saved separately as unverified reference annotations.
  They are never fed to association or attached to predicted detections as truth.

The first connection targets the existing `OVTrack` model with an unflipped,
single-scale test pipeline. Other variants such as hybrid SORT require their own
existing motion initialization and are outside this connection.

## Output and validation

Each new output directory contains `resolved_config.json`, `status.json`, raw
queries, per-frame detection JSON, `detection_sequence.json`, separate reference
annotations, per-frame tracking JSON, and `tracking_sequence.json`.
Tracking rows contain predicted ID, original-image box, class ID/name and score.
Existing output directories are rejected; failures retain an explicit status.
`--dry-run` checks paths and handoff inputs, not model loading or inference.

```sh
python -B -m unittest discover -s integration/tests -v
```

These tests exercise handoffs with injected inference backends. A successful
mocked run is not real-model verification. Only running the launch command with
all actual runtime inputs verifies the complete model connection.
