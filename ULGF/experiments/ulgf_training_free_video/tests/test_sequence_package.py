import sys,unittest
from pathlib import Path
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from package_sequence_controls import composite,check_frame,long_offsets

class SequencePackageTests(unittest.TestCase):
    def fixture(self):
        im=Image.new('RGB',(256,256),'yellow');m=Image.new('L',im.size)
        ImageDraw.Draw(m).rectangle((70,50,180,195),fill=255)
        plate=Image.new('RGB',im.size,'blue')
        f,mask,a=composite(im,m,plate,(3,2));a['frame_index']=0
        return f,mask,a,im,m,plate,(3,2)
    def test_valid(self):self.assertTrue(check_frame(0,*self.fixture())['bbox_matches_mask'])
    def test_bad_id(self):
        args=self.fixture();args[2]['instance_id']=2
        with self.assertRaises(ValueError):check_frame(0,*args)
    def test_bad_bbox(self):
        args=self.fixture();args[2]['bbox_xyxy_pixels'][0]+=1
        with self.assertRaises(ValueError):check_frame(0,*args)
    def test_bad_mask(self):
        args=self.fixture();args[1].putpixel((0,0),255)
        with self.assertRaises(ValueError):check_frame(0,*args)
    def test_long_trajectory(self):
        moves=long_offsets();self.assertEqual(len(moves),48)
        self.assertEqual(moves[0],(0,0));self.assertEqual(moves[-1],(40,20))
        self.assertTrue(all(0<=b[0]-a[0]<=2 and 0<=b[1]-a[1]<=1 for a,b in zip(moves,moves[1:])))
    def test_clipping_rejected(self):
        _,_,_,im,m,plate,_=self.fixture()
        with self.assertRaises(ValueError):composite(im,m,plate,(100,0))
