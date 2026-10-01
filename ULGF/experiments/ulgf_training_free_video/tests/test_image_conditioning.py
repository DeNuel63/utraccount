import sys
from pathlib import Path
import unittest
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from image_conditioning import prepare, prepare_crop, normalized

class ImageConditioningTests(unittest.TestCase):
    def test_crop_box_roundtrip(self):
        im=Image.new('RGB',(1920,1080))
        box=(.425520835,.152777775,.604687505,.623148145)
        out,b,g=prepare_crop(im,box)
        l,t,r,d=g['crop_xyxy'];side=r-l
        self.assertEqual(side,d-t);self.assertFalse(g['padding'])
        for actual,expected in zip(((b[0]*side+l)/1920,(b[1]*side+t)/1080,
                                    (b[2]*side+l)/1920,(b[3]*side+t)/1080),box):
            self.assertAlmostEqual(actual,expected)
        self.assertEqual(out.size,(256,256))
    def test_edge_crop_contains_box(self):
        _,b,g=prepare_crop(Image.new('RGB',(512,512)),(0,0,.2,.3))
        self.assertEqual(g['crop_xyxy'][:2],[0,0])
        self.assertTrue(all(0<=v<=1 for v in b))
    def test_crop_rejects_unfit_context(self):
        with self.assertRaises(ValueError):prepare_crop(Image.new('RGB',(512,256)),(0,0,1,1))
    def test_letterbox_and_box(self):
        image,b,g=prepare(Image.new('RGB',(512,256),'white'),(0,0,1,1))
        self.assertEqual(image.size,(256,256));self.assertEqual(b,(0,.25,1,.75))
        self.assertEqual(g['offset'],[0,64])
    def test_normalization(self):
        self.assertTrue((normalized(Image.new('RGB',(256,256),'black'))==-1).all())
        self.assertEqual(normalized(Image.new('RGB',(256,256))).shape,(1,3,256,256))
    def test_orientation_rejected(self):
        im=Image.new('RGB',(256,256));im.getexif()[274]=6
        with self.assertRaises(ValueError):prepare(im,(0,0,1,1))
