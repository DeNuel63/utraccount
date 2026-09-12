# Implementation Status

This file records the current state of the CountGD++ component of U-TracCount.

## Implemented

| Component | Description |
| --- | --- |
| CountGD++ adapter | Runs the supported CountGD++ backend through a model-independent interface. |
| Raw-output retention | Saves the unfiltered model response for every processed frame. |
| Detection contract | Exports versioned `utraccount.detection-batch/v1` JSON records. |
| Boundary validation | Rejects malformed, duplicate, unsupported, and out-of-bounds detections. |
| Coordinate safety | Clips accepted boxes to original-frame bounds and discards empty intersections. |
| Experiment metadata | Provides versioned dataset-register and run-manifest contracts. |
| Detector evaluation | Computes count MAE/RMSE and label-aware IoU localisation metrics. |
| COVTrack preparation | Converts boxes to `xyxy + score`, maps class IDs, and validates frame ordering. |

## Verification

The local unit suite contains 14 tests covering contracts, filtering, geometry,
validation, metadata, evaluation, and COVTrack boundary preparation. The suite
does not require a GPU, model checkpoint, or dataset.

```bash
python -m unittest discover -s tests -v
```

## External prerequisites

- Official CountGD++ checkpoint, retained outside version control.
- Compatible CUDA, PyTorch, Detectron2, and GroundingDINO runtime.
- Approved underwater dataset, class vocabulary, licence, and split register.

## Integration status

The CountGD++ output contract is ready for downstream consumption. COVTrack's
native tracker accepts pixel `xyxy + score` boxes, integer labels, and
RoI-derived appearance embeddings. This repository prepares the boxes and
labels only. A COVTrack-side external-detection path that derives embeddings
for CountGD++ boxes is required before end-to-end tracking can be validated.

## Next steps

1. Configure the model checkpoint and compatible GPU environment.
2. Register a licensed dataset and final train/validation/test splits.
3. Run and inspect labelled validation frames; record the selected prompt,
   threshold, and evaluation protocol in a run manifest.
4. Implement and validate COVTrack external-detection injection.
