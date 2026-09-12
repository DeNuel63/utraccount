import tempfile
import unittest
from pathlib import Path

from utraccount_count.dataset_registry import DatasetRegister, load_dataset_register


class DatasetRegisterTests(unittest.TestCase):
    def test_rejects_split_leakage(self) -> None:
        with self.assertRaises(ValueError):
            DatasetRegister(
                dataset_name="demo", dataset_version="v1", licence="permitted", intended_use="evaluation",
                train_ids=("shared",), validation_ids=("validation",), test_ids=("shared",),
                label_types=("box",),
            )

    def test_round_trips_valid_register(self) -> None:
        register = DatasetRegister(
            dataset_name="demo", dataset_version="v1", licence="permitted", intended_use="evaluation",
            train_ids=("train",), validation_ids=("validation",), test_ids=("test",),
            label_types=("box",), preprocessing=("none",),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "register.json"
            register.write(path)
            self.assertEqual(register, load_dataset_register(path))
