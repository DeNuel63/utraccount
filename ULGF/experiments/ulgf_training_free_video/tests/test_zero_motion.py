import sys
from pathlib import Path
import tempfile
import unittest
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_zero_motion as control
import run_five_frames as runner

class ZeroMotionTests(unittest.TestCase):
    def test_first_layout_repeated_without_mutating_reference(self):
        original=runner.layout()
        before=original.to_json()
        fixed=control.fixed_layout(original)
        self.assertEqual(len(fixed.frames),5)
        self.assertTrue(all(f.objects==original.frames[0].objects for f in fixed.frames))
        self.assertEqual(original.to_json(),before)
        self.assertEqual(fixed.generation.temporal_mode,'zero_motion_control')

    def test_exact_and_single_pixel_difference(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=root/'a.png';b=root/'b.png'
            image=Image.new('RGB',(10,10),(0,0,0));image.save(a);image.save(b)
            self.assertTrue(control.compare([a,b])[1]['pixels_identical_to_frame0'])
            image.putpixel((0,0),(1,0,0));image.save(b)
            report=control.compare([a,b])[1]
            self.assertFalse(report['pixels_identical_to_frame0'])
            self.assertEqual(report['maximum_channel_difference'],1)
            self.assertAlmostEqual(report['mean_absolute_channel_difference'],1/300)

if __name__=='__main__':unittest.main()
