import tempfile
import unittest
from pathlib import Path

from ulgf_baseline.config import BaselineConfig
from ulgf_baseline.layouts import read_yolo_layout, to_yolo_lines
from ulgf_baseline.planning import PreflightError, build_generation_plan


class BaselinePlanningTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.checkpoint = self.root / "checkpoint"
        self.images = self.root / "source-images"
        self.labels = self.root / "source-labels"
        self.output = self.root / "output"
        self.checkpoint.mkdir()
        (self.checkpoint / "model_index.json").write_text("{}")
        (self.checkpoint / "generation_config.json").write_text("{}")
        self.images.mkdir()
        self.labels.mkdir()
        (self.images / "sample.jpg").write_bytes(b"fixture")
        (self.labels / "sample.txt").write_text("0 0.5 0.5 0.2 0.4\n")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def config(self, **changes):
        values = dict(
            checkpoint=self.checkpoint,
            image_dir=self.images,
            label_dir=self.labels,
            output_dir=self.output,
            classes=("fish",),
        )
        values.update(changes)
        return BaselineConfig(**values)

    def test_plan_is_deterministic_and_separate_from_sources(self):
        plan = build_generation_plan(self.config(seed=41))
        self.assertEqual(1, len(plan))
        self.assertEqual(41, plan[0].seed)
        self.assertEqual(self.output / "images" / "Geosample.jpg", plan[0].output_images[0])
        self.assertEqual(self.output / "labels" / "Geosample.txt", plan[0].output_label)
        self.assertFalse(str(plan[0].output_label).startswith(str(self.labels)))

    def test_plan_refuses_existing_outputs_without_overwrite(self):
        existing = self.output / "images" / "Geosample.jpg"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"do not replace")
        with self.assertRaisesRegex(PreflightError, "Refusing to overwrite"):
            build_generation_plan(self.config())
        plan = build_generation_plan(self.config(overwrite=True))
        self.assertEqual(existing, plan[0].output_images[0])

    def test_plan_requires_matching_label(self):
        (self.labels / "sample.txt").unlink()
        with self.assertRaisesRegex(PreflightError, "Missing labels"):
            build_generation_plan(self.config())

    def test_plan_rejects_incomplete_checkpoint_directory(self):
        (self.checkpoint / "generation_config.json").unlink()
        with self.assertRaisesRegex(PreflightError, "Checkpoint directory is incomplete"):
            build_generation_plan(self.config())

    def test_seed_file_must_cover_every_image(self):
        (self.images / "second.png").write_bytes(b"fixture")
        (self.labels / "second.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        seed_file = self.root / "seeds.txt"
        seed_file.write_text("100\n")
        with self.assertRaisesRegex(PreflightError, "contains 1 values but 2 images"):
            build_generation_plan(self.config(seed_file=seed_file))

    def test_output_must_be_separate_from_inputs(self):
        with self.assertRaisesRegex(PreflightError, "must not be the same as or nested"):
            build_generation_plan(self.config(output_dir=self.images / "generated"))

    def test_plan_refuses_existing_run_metadata(self):
        self.output.mkdir()
        (self.output / "run.json").write_text("{}")
        with self.assertRaisesRegex(PreflightError, "Refusing to overwrite"):
            build_generation_plan(self.config())

    def test_yolo_layout_round_trip(self):
        objects = read_yolo_layout(self.labels / "sample.txt", ("fish",))
        self.assertEqual("fish", objects[0][0])
        lines = to_yolo_lines(objects, ("fish",))
        self.assertEqual("0 0.50000000 0.50000000 0.20000000 0.40000000", lines[0])


if __name__ == "__main__":
    unittest.main()
