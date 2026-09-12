import unittest

from utraccount_count.contracts import Detection, DetectionBatch, FrameInfo, InferenceSource


class DetectionContractTests(unittest.TestCase):
    def test_serialises_tracker_ready_batch(self) -> None:
        batch = DetectionBatch(
            sequence_id="sequence-a",
            frame_index=0,
            frame=FrameInfo(uri="frame_000000.jpg", width_px=640, height_px=480),
            source=InferenceSource(model_version="checkpoint-abc", prompt="fish", confidence_threshold=0.3),
            detections=(Detection("sequence-a-000000-000", (320, 240, 50, 40), "fish", 0.8),),
        )
        result = batch.to_dict()
        self.assertEqual("utraccount.detection-batch/v1", result["schema_version"])
        self.assertEqual([320, 240, 50, 40], result["detections"][0]["bbox_xywh_center"])

    def test_rejects_duplicate_detection_ids(self) -> None:
        detection = Detection("same", (1, 1, 1, 1), "fish", 0.8)
        with self.assertRaises(ValueError):
            DetectionBatch(
                sequence_id="sequence-a",
                frame_index=0,
                frame=FrameInfo(uri="frame.jpg", width_px=10, height_px=10),
                source=InferenceSource(model_version="checkpoint", prompt="fish", confidence_threshold=0.3),
                detections=(detection, detection),
            )

    def test_rejects_box_outside_original_frame(self) -> None:
        with self.assertRaises(ValueError):
            DetectionBatch(
                sequence_id="sequence-a",
                frame_index=0,
                frame=FrameInfo(uri="frame.jpg", width_px=10, height_px=10),
                source=InferenceSource(model_version="checkpoint", prompt="fish", confidence_threshold=0.3),
                detections=(Detection("outside", (9, 5, 4, 4), "fish", 0.8),),
            )
