import random
import unittest
from types import SimpleNamespace

import numpy as np

from ulgf_baseline.training_contract import checked_token_ids, generation_config, make_loss_mask


class TrainingContractTests(unittest.TestCase):
    def test_four_way_overlap_uses_valid_owner_and_is_reproducible(self):
        boxes = [[0, 0, 1, 1], [0, 0, .75, .75], [0, 0, .5, .5], [0, 0, .25, .25]]
        a = make_loss_mask(boxes, 8, 8, 'constant', 2, False, random.Random(3))
        b = make_loss_mask(boxes, 8, 8, 'constant', 2, False, random.Random(3))
        np.testing.assert_array_equal(a, b)
        allowed = [2 * area ** -.2 for area in (64, 36, 16, 4)]
        self.assertTrue(any(np.isclose(a[0, 0], value) for value in allowed))

    def test_weights_do_not_collide_with_object_ids(self):
        # First weight equals the second object's old ID (2).
        mask = make_loss_mask([[0, 0, .5, 1], [.5, 0, 1, .5]], 2, 2,
                              'constant', 2 * 2 ** .2, False)
        self.assertAlmostEqual(float(mask[0, 0]), 2)
        self.assertAlmostEqual(float(mask[0, 1]), 2 * 2 ** .2, places=6)

    def test_tiny_edge_box_gets_one_cell(self):
        mask = make_loss_mask([[.999, .999, 1, 1]], 8, 8, 'constant', 2, False)
        self.assertEqual(mask[-1, -1], 2)
        self.assertLess(mask[0, 0], mask[-1, -1])

    def test_empty_layout_and_unweighted_loss_are_valid(self):
        np.testing.assert_allclose(make_loss_mask([], 4, 4, 'constant', 2, True), 1)
        np.testing.assert_array_equal(make_loss_mask([], 4, 4, None, 2, False), 1)

    def test_area_mode_and_normalization(self):
        mask = make_loss_mask([[0, 0, .5, .5]], 4, 4, 'area', 1, False)
        self.assertEqual(mask[0, 0], .25)
        self.assertEqual(mask[-1, -1], 1 / 16)
        mask = make_loss_mask([[0, 0, .5, .5]], 4, 4, 'area', 1, True)
        self.assertAlmostEqual(float(mask.mean()), 1)

    def test_invalid_geometry_and_weight_rejected(self):
        for box in ([0, 0, 0, 1], [-.1, 0, 1, 1], [0, 0, float('nan'), 1]):
            with self.assertRaises(ValueError):
                make_loss_mask([box], 4, 4, None, 1, False)
        with self.assertRaises(ValueError):
            make_loss_mask([], 4, 4, 'constant', 0, True)

    def test_no_silent_token_truncation(self):
        class Tokenizer:
            model_max_length = 3
            def __call__(self, captions, **kwargs):
                return SimpleNamespace(input_ids=[[1] * len(caption) for caption in captions])
        self.assertEqual(checked_token_ids(Tokenizer(), ['abc']), [[1, 1, 1]])
        with self.assertRaisesRegex(ValueError, 'truncation'):
            checked_token_ids(Tokenizer(), ['abcd'])

    def test_export_matches_training_prompt_and_dimensions(self):
        config = generation_config('ruod', [256, 256], (256, 384))
        self.assertEqual((config['height'], config['width']), (256, 384))
        self.assertEqual(config['prompt_template'].format(camera='front', bbox='fish <l1> <l2>'),
                         'A underwater scene image of front camera with fish <l1> <l2>')


if __name__ == '__main__':
    unittest.main()
