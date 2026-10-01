import unittest

from ulgf_video.motion import MotionConfig, generate_object_sequences
from ulgf_video.schema import ObjectInstance


class VideoMotionTests(unittest.TestCase):
    def setUp(self):
        self.objects = (
            ObjectInstance(1, 0, "fish", (0.1, 0.2, 0.3, 0.4)),
            ObjectInstance(2, 0, "fish", (0.7, 0.7, 0.9, 0.9)),
        )

    def test_motion_is_deterministic_and_preserves_identity(self):
        config = MotionConfig(frames=6)
        first = generate_object_sequences(self.objects, 17, config)
        second = generate_object_sequences(self.objects, 17, config)
        self.assertEqual(first, second)
        self.assertEqual(6, len(first))
        for frame in first:
            self.assertEqual([(1, "fish"), (2, "fish")], [(o.instance_id, o.class_name) for o in frame])

    def test_zero_motion_keeps_all_boxes_fixed(self):
        config = MotionConfig(frames=4, max_speed_box_fraction=0, max_acceleration_box_fraction=0)
        frames = generate_object_sequences(self.objects, 1, config)
        self.assertTrue(all(frame == self.objects for frame in frames))

    def test_boxes_stay_in_bounds_and_keep_size(self):
        config = MotionConfig(frames=100, max_speed_box_fraction=0.5, max_acceleration_box_fraction=0.1)
        frames = generate_object_sequences(self.objects, 9, config)
        initial_sizes = [(o.bbox_xyxy[2] - o.bbox_xyxy[0], o.bbox_xyxy[3] - o.bbox_xyxy[1]) for o in self.objects]
        for frame in frames:
            for obj, size in zip(frame, initial_sizes):
                x1, y1, x2, y2 = obj.bbox_xyxy
                self.assertTrue(0 <= x1 < x2 <= 1)
                self.assertTrue(0 <= y1 < y2 <= 1)
                self.assertAlmostEqual(size[0], x2 - x1)
                self.assertAlmostEqual(size[1], y2 - y1)

    def test_displacement_obeys_configured_speed_limit(self):
        objects = (ObjectInstance(1, 0, "fish", (0.4, 0.4, 0.6, 0.6)),)
        config = MotionConfig(frames=10, max_speed_box_fraction=0.05, max_acceleration_box_fraction=0.01)
        frames = generate_object_sequences(objects, 12, config)
        speed_limit = 0.2 * config.max_speed_box_fraction
        for previous, current in zip(frames, frames[1:]):
            px1, py1, _, _ = previous[0].bbox_xyxy
            cx1, cy1, _, _ = current[0].bbox_xyxy
            self.assertLessEqual(abs(cx1 - px1), speed_limit + 1e-12)
            self.assertLessEqual(abs(cy1 - py1), speed_limit + 1e-12)

    def test_clamp_boundary_stops_outward_velocity(self):
        object_at_edge = (ObjectInstance(1, 0, "fish", (0.0, 0.0, 0.2, 0.2)),)
        config = MotionConfig(frames=20, max_speed_box_fraction=1.0, max_acceleration_box_fraction=0, boundary_policy="clamp")
        frames = generate_object_sequences(object_at_edge, 4, config)
        for frame in frames:
            x1, y1, x2, y2 = frame[0].bbox_xyxy
            self.assertTrue(0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1)


if __name__ == "__main__":
    unittest.main()
