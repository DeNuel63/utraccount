"""Portable CPU verifier: python verify_delivery.py <extracted handoff directory>."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image

def verify(root):
    total=0
    for name in ('sequence_5frame','sequence_48frame'):
        folder=root/name
        sums=json.loads((folder/'SHA256SUMS.json').read_text())
        for relative,expected in sums.items():
            path=(folder/relative).resolve();path.relative_to(folder.resolve())
            assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,relative
        manifest=json.loads((folder/'manifest.json').read_text())
        source=np.asarray(Image.open(folder/'provenance/keyframe.png').convert('RGB'))
        plate=np.asarray(Image.open(folder/'provenance/clean_plate.png').convert('RGB'))
        base=np.asarray(Image.open(folder/'provenance/approved_mask.png').convert('L'))>0
        ys,xs=np.where(base);frames=[]
        for i,entry in enumerate(manifest['frames']):
            ann=json.loads((folder/'annotations'/('%06d.json'%i)).read_text())
            mask=Image.open(folder/'masks'/('%06d.png'%i)).convert('L');m=np.asarray(mask)
            frame=np.asarray(Image.open(folder/entry['image_path']).convert('RGB'));frames.append(frame)
            assert frame.shape==(256,256,3) and m.shape==(256,256)
            assert set(np.unique(m)).issubset({0,255})
            assert entry['frame_index']==ann['frame_index']==i
            obj=entry['objects'];assert len(obj)==1 and obj[0]['instance_id']==ann['instance_id']==1
            assert obj[0]['class_id']==ann['class_id']==4 and obj[0]['class_name']==ann['class_name']=='fish'
            assert list(mask.getbbox())==ann['bbox_xyxy_pixels'] and ann['bbox_max_exclusive'] is True
            assert np.allclose(np.array(ann['bbox_xyxy_pixels'])/256,ann['bbox_xyxy'],rtol=0,atol=1e-10)
            assert obj[0]['bbox_xyxy']==ann['bbox_xyxy']
            dx,dy=ann['translation_pixels']
            assert min(xs+dx)>=0 and max(xs+dx)<256 and min(ys+dy)>=0 and max(ys+dy)<256
            expected=np.zeros((256,256),dtype=bool);expected[ys+dy,xs+dx]=True
            assert np.array_equal(m>0,expected)
            assert np.array_equal(frame[ys+dy,xs+dx],source[ys,xs])
            assert np.array_equal(frame[~expected],plate[~expected])
        duration=round(1000/manifest['fps']);tick=0
        with Image.open(folder/'playback.webp') as animation:
            for i in range(animation.n_frames):
                animation.seek(i);pixels=np.asarray(animation.convert('RGB'))
                ms=animation.info['duration'];assert ms%duration==0
                for _ in range(ms//duration):
                    assert tick<len(frames) and np.array_equal(pixels,frames[tick]);tick+=1
        assert tick==len(frames)
        total+=tick;print(name+': hashes, masks, boxes, IDs, foreground, background and playback PASS')
    print('TOTAL:',total,'frames verified. No model loaded; no training started.')

if __name__=='__main__':verify(Path(sys.argv[1] if len(sys.argv)>1 else '.'))
