import tempfile
import unittest
from pathlib import Path

from ulgf_video.schema import ClipManifest, FrameLayout, GenerationMetadata, MediumSpec, ObjectInstance
from ulgf_video.validation import ManifestValidationError, validate_manifest


def valid_manifest():
    obj = ObjectInstance(1, 0, "fish", (0.1, 0.2, 0.3, 0.4))
    return ClipManifest(
        clip_id="clip-1",
        seed=5,
        fps=10,
        width=256,
        height=256,
        medium=MediumSpec("blue water"),
        frames=(FrameLayout(0, "frames/000000.png", (obj,)), FrameLayout(1, "frames/000001.png", (obj,))),
        generation=GenerationMetadata("", "test", motion={"model": "test"}),
    )


class VideoSchemaTests(unittest.TestCase):
    def test_round_trip_is_lossless_and_stable(self):
        original = valid_manifest()
        restored = ClipManifest.from_dict(original.to_dict())
        self.assertEqual(original, restored)
        self.assertEqual(original.to_json(), restored.to_json())
        validate_manifest(restored)

    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            valid_manifest().write(path)
            self.assertEqual(valid_manifest(), ClipManifest.read(path))

    def test_rejects_class_change_for_identity(self):
        manifest = valid_manifest()
        changed = ObjectInstance(1, 1, "shark", (0.1, 0.2, 0.3, 0.4))
        manifest = ClipManifest(
            clip_id=manifest.clip_id, seed=manifest.seed, fps=manifest.fps, width=manifest.width,
            height=manifest.height, medium=manifest.medium,
            frames=(manifest.frames[0], FrameLayout(1, "frames/000001.png", (changed,))),
            generation=manifest.generation,
        )
        with self.assertRaisesRegex(ManifestValidationError, "changes class"):
            validate_manifest(manifest)

    def test_rejects_path_traversal_and_duplicate_id(self):
        obj = valid_manifest().frames[0].objects[0]
        manifest = valid_manifest()
        manifest = ClipManifest(
            clip_id=manifest.clip_id, seed=manifest.seed, fps=manifest.fps, width=manifest.width,
            height=manifest.height, medium=manifest.medium,
            frames=(FrameLayout(0, "../escape.png", (obj, obj)),), generation=manifest.generation,
        )
        with self.assertRaises(ManifestValidationError) as context:
            validate_manifest(manifest)
        self.assertIn("relative POSIX path", str(context.exception))
        self.assertIn("repeats instance_id", str(context.exception))


if __name__ == "__main__":
    unittest.main()
