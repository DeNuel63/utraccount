import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.generate_image_baseline import _run
from ulgf_baseline.config import BaselineConfig
from ulgf_baseline.planning import build_generation_plan


class _Result:
    def __init__(self, images):
        self.images = images


class _FakePipeline:
    def __init__(self):
        self.device = None
        self.calls = []
        self.safety_checker = object()

    def to(self, device):
        self.device = device
        return self

    def __call__(self, prompts, **kwargs):
        self.calls.append((prompts, kwargs))
        return _Result([Image.new("RGB", (kwargs["width"], kwargs["height"]), "navy") for _ in prompts])


class BaselineGenerationTests(unittest.TestCase):
    def test_mocked_generation_writes_separate_complete_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            images = root / "source-images"
            labels = root / "source-labels"
            output = root / "output"
            checkpoint.mkdir()
            (checkpoint / "model_index.json").write_text("{}")
            (checkpoint / "generation_config.json").write_text("{}")
            images.mkdir()
            labels.mkdir()
            (images / "sample.jpg").write_bytes(b"fixture")
            (labels / "sample.txt").write_text("0 0.5 0.5 0.2 0.4\n")
            config = BaselineConfig(
                checkpoint=checkpoint,
                image_dir=images,
                label_dir=labels,
                output_dir=output,
                classes=("fish",),
                device="cpu",
                seed=7,
                samples_per_layout=2,
            )
            items = build_generation_plan(config)
            pipeline = _FakePipeline()
            observed_seeds = []

            def loader(_checkpoint):
                return pipeline, {
                    "cfg_scale": 7.5,
                    "num_inference_steps": 2,
                    "height": 16,
                    "width": 16,
                    "prompt_template": "underwater scene: {bbox}",
                }

            _run(
                config,
                items,
                pipeline_loader=loader,
                seed_setter=observed_seeds.append,
                bbox_encoder=lambda boxes, _config: "encoded-{}".format(len(boxes)),
            )

            self.assertEqual([7], observed_seeds)
            self.assertEqual("cpu", pipeline.device)
            self.assertEqual(2, len(pipeline.calls[0][0]))
            self.assertTrue((output / "images" / "Geosample_00.jpg").is_file())
            self.assertTrue((output / "images" / "Geosample_01.jpg").is_file())
            self.assertTrue((output / "labels" / "Geosample.txt").is_file())
            metadata = json.loads((output / "run.json").read_text())
            self.assertEqual(7, metadata["items"][0]["seed"])
            self.assertFalse(metadata["safety_checker_disabled"])
            self.assertEqual(b"fixture", (images / "sample.jpg").read_bytes())
            self.assertEqual("0 0.5 0.5 0.2 0.4\n", (labels / "sample.txt").read_text())


if __name__ == "__main__":
    unittest.main()
