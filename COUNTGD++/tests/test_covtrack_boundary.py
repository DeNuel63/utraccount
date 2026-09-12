import unittest

from utraccount_count.contracts import Detection, DetectionBatch, FrameInfo, InferenceSource
from utraccount_count.covtrack_boundary import (
    COVTrackCompatibilityError,
    to_covtrack_detections,
    validate_covtrack_sequence,
)


def make_batch(frame_index: int) -> DetectionBatch:
    return DetectionBatch(
        sequence_id="sequence-a",
        frame_index=frame_index,
        frame=FrameInfo(uri=f"frames/{frame_index}.jpg", width_px=100, height_px=80),
        source=InferenceSource(model_version="checkpoint-a", prompt="fish", confidence_threshold=0.2),
        detections=(Detection("detection-a", (50, 40, 20, 10), "fish", 0.8),),
    )


class COVTrackBoundaryTests(unittest.TestCase):
    def test_converts_countgd_box_to_covtrack_xyxy_score(self) -> None:
        prepared = to_covtrack_detections(make_batch(0), {"fish": 3})
        self.assertEqual(((40.0, 35.0, 60.0, 45.0, 0.8),), prepared.boxes_xyxy_score)
        self.assertEqual((3,), prepared.label_ids)

    def test_rejects_unknown_covtrack_label(self) -> None:
        with self.assertRaises(COVTrackCompatibilityError):
            to_covtrack_detections(make_batch(0), {})

    def test_requires_consecutive_sequence_from_zero(self) -> None:
        with self.assertRaises(COVTrackCompatibilityError):
            validate_covtrack_sequence((make_batch(1),))
        self.assertEqual((make_batch(0), make_batch(1)), validate_covtrack_sequence((make_batch(0), make_batch(1))))
