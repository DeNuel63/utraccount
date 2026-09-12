import json
import tempfile
import unittest
from pathlib import Path

from utraccount_count.run_manifest import RunManifest
from utraccount_count.validation import ContractValidationError, load_detection_batch


class ValidationAndManifestTests(unittest.TestCase):
    def test_loads_example_detection_batch(self) -> None:
        batch = load_detection_batch(Path("examples/detection-batch.v1.json"))
        self.assertEqual("mft25_001", batch.sequence_id)
        self.assertEqual(1, len(batch.detections))

    def test_rejects_unexpected_contract_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps({"unexpected": True}), encoding="utf-8")
            with self.assertRaises(ContractValidationError):
                load_detection_batch(path)

    def test_writes_run_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            RunManifest(
                run_id="baseline-001",
                model_version="countgd_plusplus.pth",
                checkpoint_sha256="a" * 64,
                dataset_name="DeepFish",
                dataset_version="v1",
                split="validation",
                prompt="fish",
                confidence_threshold=0.23,
                runtime={"python": "3.10", "gpu": "NVIDIA GPU"},
            ).write(path)
            self.assertEqual("utraccount.run-manifest/v1", json.loads(path.read_text())["manifest_version"])
