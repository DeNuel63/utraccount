import json
import tempfile
import unittest
from pathlib import Path

from utraccount_count.adapter import CountGDAdapter
from utraccount_count.backend import RawDetection, RawInference
from utraccount_count.contracts import FrameInfo


class StubBackend:
    def infer(self, image_path: Path, *, prompt: str) -> RawInference:
        return RawInference(
            candidates=[
                RawDetection((20, 30, 10, 8), "fish", 0.9),
                RawDetection((40, 50, 12, 6), "fish", 0.1),
                RawDetection((60, 70, 12, 6), "fish", 0.8, negative_confidence=0.9),
                RawDetection((2, 2, 10, 10), "fish", 0.95),
            ],
            model_response={"prompt": prompt, "unfiltered_count": 4},
        )


class CountGDAdapterTests(unittest.TestCase):
    def test_saves_raw_output_before_thresholding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_path = Path(temporary_directory) / "raw.json"
            batch = CountGDAdapter(StubBackend(), model_version="test-checkpoint").infer_frame(
                Path("frame.jpg"),
                sequence_id="seq-1",
                frame_index=3,
                frame=FrameInfo(uri="frame.jpg", width_px=100, height_px=100),
                prompt="fish",
                confidence_threshold=0.5,
                raw_output_path=raw_path,
            )
            self.assertEqual(2, len(batch.detections))
            self.assertEqual((3.5, 3.5, 7.0, 7.0), batch.detections[1].bbox_xywh_center)
            self.assertEqual({"prompt": "fish", "unfiltered_count": 4}, json.loads(raw_path.read_text()))
