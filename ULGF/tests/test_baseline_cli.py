import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "generate_image_baseline.py"


class BaselineCliTests(unittest.TestCase):
    def test_help_does_not_require_ml_dependencies(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, cwd=str(REPOSITORY_ROOT)
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("--dry-run", result.stdout)

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            images = root / "images"
            labels = root / "labels"
            output = root / "output"
            checkpoint.mkdir()
            (checkpoint / "model_index.json").write_text("{}")
            (checkpoint / "generation_config.json").write_text("{}")
            images.mkdir()
            labels.mkdir()
            (images / "sample.jpg").write_bytes(b"fixture")
            (labels / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--checkpoint",
                    str(checkpoint),
                    "--image-dir",
                    str(images),
                    "--label-dir",
                    str(labels),
                    "--output-dir",
                    str(output),
                    "--classes",
                    "fish",
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                cwd=str(REPOSITORY_ROOT),
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn("no output was written", result.stdout)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
