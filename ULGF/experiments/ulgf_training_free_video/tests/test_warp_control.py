import sys
from pathlib import Path
import unittest
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_warp_control as warp

class WarpTests(unittest.TestCase):
    def example(self):
        im=Image.new('RGB',(32,32),'blue');ImageDraw.Draw(im).rectangle((10,10,15,15),fill='red')
        mask=Image.new('L',im.size,0);ImageDraw.Draw(mask).rectangle((10,10,15,15),fill=255)
        return im,mask
    def test_zero_motion_exact(self):
        im,mask=self.example();bg=warp.fill_hole(im,mask)
        out,_=warp.translated(im,bg,mask,0,0)
        self.assertTrue(np.array_equal(np.asarray(out),np.asarray(im)))
    def test_translation_and_hole(self):
        im,mask=self.example();bg=warp.fill_hole(im,mask)
        out,moved=warp.translated(im,bg,mask,8,0)
        self.assertEqual(out.getpixel((11,11)),(0,0,255))
        self.assertEqual(out.getpixel((19,11)),(255,0,0))
        self.assertEqual(moved.getbbox(),(18,10,24,16))
    def test_clipping_rejected(self):
        im,mask=self.example()
        with self.assertRaises(ValueError):warp.translated(im,im,mask,25,0)
    def test_full_mask_rejected(self):
        im,_=self.example()
        with self.assertRaises(ValueError):warp.fill_hole(im,Image.new('L',im.size,255))

if __name__=='__main__':unittest.main()
