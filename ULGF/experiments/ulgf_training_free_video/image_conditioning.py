"""CPU preprocessing; preserve aspect ratio and transform requested boxes."""
from PIL import Image
import numpy as np
import math

def prepare_crop(image, box):
    if image.getexif().get(274,1) not in (None,1):
        raise ValueError('EXIF orientation requires an explicit box convention')
    if len(box)!=4 or not all(math.isfinite(v) for v in box) or not (0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1):
        raise ValueError('Invalid normalized box')
    image=image.convert('RGB');w,h=image.size
    x1,y1,x2,y2=box[0]*w,box[1]*h,box[2]*w,box[3]*h
    side=math.ceil(max(x2-x1,y2-y1)*1.6)
    if side>min(w,h):raise ValueError('Cannot retain requested crop context without padding')
    left=max(0,min(w-side,round((x1+x2-side)/2)))
    top=max(0,min(h-side,round((y1+y2-side)/2)))
    if not (left<=x1<x2<=left+side and top<=y1<y2<=top+side):
        raise ValueError('Crop would truncate requested object')
    out=image.crop((left,top,left+side,top+side)).resize((256,256),Image.Resampling.LANCZOS)
    b=((x1-left)/side,(y1-top)/side,(x2-left)/side,(y2-top)/side)
    return out,b,dict(original_size=[w,h],crop_xyxy=[left,top,left+side,top+side],
                      resized_size=[256,256],padding=False,context_factor=1.6)

def prepare(image, box):
    if image.getexif().get(274,1) not in (None,1):
        raise ValueError('EXIF orientation requires an explicit box convention')
    image=image.convert('RGB');w,h=image.size
    scale=min(256/w,256/h);nw,nh=round(w*scale),round(h*scale)
    x,y=(256-nw)//2,(256-nh)//2
    out=Image.new('RGB',(256,256),(127,127,127))
    out.paste(image.resize((nw,nh),Image.Resampling.LANCZOS),(x,y))
    b=((box[0]*nw+x)/256,(box[1]*nh+y)/256,(box[2]*nw+x)/256,(box[3]*nh+y)/256)
    return out,b,dict(original_size=[w,h],resized_size=[nw,nh],offset=[x,y],padding_rgb=[127]*3)

def normalized(image):
    return np.asarray(image,dtype=np.float32).transpose(2,0,1)[None]/127.5-1.0
