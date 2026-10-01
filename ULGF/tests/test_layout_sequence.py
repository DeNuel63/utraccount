import tempfile
import unittest
from pathlib import Path

from ulgf_video.layout_sequence import build_layout_manifest
from ulgf_video.motion import MotionConfig
from ulgf_video.validation import validate_manifest


class LayoutSequenceTests(unittest.TestCase):
    def test_builds_valid_manifest_with_stable_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "seed.jpg"
            label = root / "seed.txt"
            image.write_bytes(b"fixture")
            label.write_text("0 0.5 0.5 0.2 0.2\n1 0.25 0.25 0.1 0.1\n")
            manifest = build_layout_manifest(
                "clip-1", image, label, ("fish", "turtle"), 256, 256, 11, "blue water",
                MotionConfig(frames=5), "test-config",
            )
            validate_manifest(manifest)
            self.assertEqual(5, len(manifest.frames))
            self.assertEqual([1, 2], [obj.instance_id for obj in manifest.frames[0].objects])
            for frame in manifest.frames:
                self.assertEqual([1, 2], [obj.instance_id for obj in frame.objects])


if __name__ == "__main__":
    unittest.main()
