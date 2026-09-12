# U-TracCount

U-TracCount provides the CountGD++ detection component of an underwater object
counting and tracking pipeline. It runs CountGD++ on individual video frames,
preserves raw model output, and exports validated detections through a stable
interface for downstream tracking.

## Features

- CountGD++ positive-text inference wrapper.
- Raw inference retention before confidence filtering.
- Versioned, validated per-frame detection records.
- Original-frame pixel coordinates with bounds enforcement.
- Dataset-register and run-manifest contracts for reproducible experiments.
- Detector-only count and localisation metrics.
- Explicit preparation of detections for COVTrack's box and label conventions.

## Architecture

```text
Frame + text prompt
       |
       v
  CountGD++ inference
       |
       +--> raw model response
       |
       v
validated detection batch
       |
       v
downstream tracking boundary
```

Detection batches use original-frame pixel boxes in
`[centre_x, centre_y, width, height]` format. The public specification is
[`contracts/detection-record.schema.json`](contracts/detection-record.schema.json).

## Repository layout

```text
contracts/              Versioned JSON contracts
examples/               Example contract payloads
src/utraccount_count/   Detection, validation, evaluation, and integration code
tests/                  Unit tests
CountGDPlusPlus-main/  Official CountGD++ source archive
```

## Getting started

Install this package in an environment that also satisfies the official
CountGD++ runtime requirements:

```bash
pip install -e .
python -m unittest discover -s tests -v
```

The CountGD++ checkpoint is intentionally not distributed with this repository.
Record the checkpoint hash, runtime settings, prompt, and data split in a run
manifest before conducting an experiment.

## COVTrack boundary

`to_covtrack_detections()` converts a validated batch to COVTrack's
`[left, top, right, bottom, score]` boxes and integer class IDs.
`validate_covtrack_sequence()` verifies the zero-based, consecutive frame order
required by COVTrack state management.

This is a boundary-preparation layer, not a complete tracker integration.
COVTrack currently requires RoI-derived appearance embeddings in addition to
boxes and labels; an external-detection injection path is still required.

## Evaluation

The package provides frame-level count MAE/RMSE and label-aware IoU
localisation precision, recall, and F1. Report detector performance separately
from tracking performance.

## Project status

See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for the current
implementation status, prerequisites, and integration work.
