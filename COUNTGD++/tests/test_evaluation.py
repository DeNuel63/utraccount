import unittest

from utraccount_count.contracts import Detection
from utraccount_count.evaluation import evaluate_counts, evaluate_localisation, intersection_over_union


def detection(identifier: str, box: tuple[float, float, float, float], label: str = "fish") -> Detection:
    return Detection(identifier, box, label, 0.9)


class EvaluationTests(unittest.TestCase):
    def test_localisation_metrics_use_label_aware_iou_matching(self) -> None:
        prediction = detection("prediction", (5, 5, 4, 4))
        truth = detection("truth", (5, 5, 4, 4))
        other_label = detection("other", (5, 5, 4, 4), "shark")
        self.assertEqual(1.0, intersection_over_union(prediction, truth))
        result = evaluate_localisation([prediction], [truth, other_label], iou_threshold=0.5)
        self.assertEqual((1, 0, 1), (result.true_positives, result.false_positives, result.false_negatives))
        self.assertEqual(1.0, result.precision)
        self.assertEqual(0.5, result.recall)

    def test_count_metrics(self) -> None:
        result = evaluate_counts([2, 5, 4], [1, 5, 7])
        self.assertAlmostEqual(4 / 3, result.mae)
        self.assertAlmostEqual((10 / 3) ** 0.5, result.rmse)
