import unittest
from pathlib import Path


class PipelinePortabilityTests(unittest.TestCase):
    def test_custom_pipelines_do_not_load_scheduler_from_private_path(self):
        repository = Path(__file__).resolve().parents[1]
        for relative_path in (
            "pipeline_custom/pipeline_prior.py",
            "pipeline_custom/pipeline_priornoisefusion.py",
        ):
            source = (repository / relative_path).read_text(encoding="utf-8")
            self.assertNotIn("DDPMScheduler.from_config('/mnt/data0", source)
            self.assertNotIn("DDPMScheduler.from_config(", source)
            self.assertIn("noise_scheduler = DDPMScheduler(", source)
            self.assertIn(
                "num_train_timesteps=self.scheduler.config.num_train_timesteps",
                source,
            )


if __name__ == "__main__":
    unittest.main()
