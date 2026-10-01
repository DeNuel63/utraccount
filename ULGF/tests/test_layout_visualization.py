import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.visualize_layout_sequence import render
from ulgf_video.schema import ClipManifest, FrameLayout, GenerationMetadata, MediumSpec, ObjectInstance


class LayoutVisualizationTests(unittest.TestCase):
    def test_render_writes_one_preview_per_frame(self):
        manifest = ClipManifest(
            clip_id="preview",
            seed=1,
            fps=10,
            width=64,
            height=48,
            medium=MediumSpec("blue water"),
            frames=(
                FrameLayout(0, "frames/000000.png", (ObjectInstance(1, 0, "fish", (0.1, 0.1, 0.4, 0.4)),)),
                FrameLayout(1, "frames/000001.png", (ObjectInstance(1, 0, "fish", (0.2, 0.1, 0.5, 0.4)),)),
            ),
            generation=GenerationMetadata("", "test"),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "preview"
            render(manifest, output)
            images = sorted(output.glob("*.png"))
            self.assertEqual(["000000.png", "000001.png"], [path.name for path in images])
            with Image.open(str(images[0])) as preview:
                self.assertEqual((64, 48), preview.size)
            with self.assertRaisesRegex(ValueError, "not empty"):
                render(manifest, output)


if __name__ == "__main__":
    unittest.main()
