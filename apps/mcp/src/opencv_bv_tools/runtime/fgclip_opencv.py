# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# Fixed-point image resizing adapts PyTorch 2.9.0 ATen UpSampleKernel.cpp.
# Copyright PyTorch contributors; see docs/licenses/PYTORCH-LICENSE.txt.
"""Experimental CPU FG-CLIP2 runtime: OpenCV 5 + NumPy + Python stdlib only.

Model export is separate. Batch=1, 256 image patches, 64 text tokens.
No downloads, arbitrary model code, or fallback inference.
"""
import argparse
import json
import math
from pathlib import Path
import re
import numpy as np
import cv2


class Tokenizer:
    """Pinned Gemma BPE subset, checked against the original fast tokenizer."""
    def __init__(self,path):
        x=json.loads(Path(path).read_text(encoding='utf-8'))
        m=x['model']
        assert m['type']=='BPE' and m['byte_fallback'] and not m['ignore_merges']
        assert x['normalizer']=={'type':'Replace','pattern':{'String':' '},'content':'▁'}
        assert all(not a['normalized'] and not a['single_word'] and not a['lstrip'] and not a['rstrip'] for a in x['added_tokens'])
        self.vocab=m['vocab']
        self.ranks={tuple(pair):i for i,pair in enumerate(m['merges'])}
        self.special={a['content']:a['id'] for a in x['added_tokens']}
        self.pattern=re.compile('('+'|'.join(re.escape(t) for t in sorted(self.special,key=len,reverse=True))+')')

    def _bpe(self,text):
        parts=list(text.replace(' ','▁'))
        while len(parts)>1:
            best=min(range(len(parts)-1),key=lambda i:(self.ranks.get((parts[i],parts[i+1]),math.inf),i))
            if (parts[best],parts[best+1]) not in self.ranks:break
            parts[best:best+2]=[parts[best]+parts[best+1]]
        out=[]
        for p in parts:
            if p in self.vocab:out.append(self.vocab[p])
            else:out.extend(self.vocab[f'<0x{b:02X}>'] for b in p.encode('utf-8'))
        return out

    def encode(self,text):
        if len(text)>4096:raise ValueError('Prototype query limit is 4096 Unicode characters')
        ids=[]
        for part in self.pattern.split(text.lower()):
            if part in self.special:ids.append(self.special[part])
            elif part:ids.extend(self._bpe(part))
        truncated=len(ids)+1>64
        ids=ids[:63]+[1]
        return np.array([ids+[0]*(64-len(ids))],dtype=np.int64),truncated


def resize_weights(source,target,dtype=np.float32):
    scale=source/target
    support=max(scale,1.0)
    centers=(np.arange(target,dtype=dtype)+dtype(.5))*dtype(scale)
    distances=np.abs((np.arange(source,dtype=dtype)+dtype(.5))[None,:]-centers[:,None])
    w=np.maximum(dtype(0),dtype(1)-distances/dtype(support))
    w/=w.sum(axis=1,keepdims=True)
    return w


def resize_uint8(array,height,width):
    """Two-pass fixed-point bilinear AA matching PyTorch's CPU uint8 route.

    Weight quantization/rounding follows ATen UpSampleKernel.cpp v2.9.0.
    """
    def coefficients(source,target):
        w=resize_weights(source,target,np.float64)
        maximum=float(w.max());precision=0
        while precision<22:
            if int(.5+maximum*(1<<(precision+1)))>=1<<15:break
            precision+=1
        return np.floor(w*(1<<precision)+.5).astype(np.int64),precision
    x=array.astype(np.int64)
    if width!=array.shape[1]:
        w,p=coefficients(array.shape[1],width)
        x=np.clip((np.einsum('hwc,kw->hkc',x,w,optimize=True)+(1<<(p-1)))>>p,0,255)
    if height!=array.shape[0]:
        w,p=coefficients(array.shape[0],height)
        x=np.clip((np.einsum('kh,hwc->kwc',w,x,optimize=True)+(1<<(p-1)))>>p,0,255)
    return x.astype(np.uint8)


def resize_aa(array,height,width):
    """Separable half-pixel bilinear antialias; float32 positional/image arrays."""
    sh,sw=array.shape[:2]
    x=np.asarray(array,dtype=np.float32)
    # Torch/Pillow antialias first filters width, then height.
    x=np.einsum('hwc,kw->hkc',x,resize_weights(sw,width),optimize=True)
    return np.einsum('kh,hwc->kwc',resize_weights(sh,height),x,optimize=True)


def target_size(height,width):
    low,high=1e-6,100.
    def scaled(scale,dim):return max(16,math.ceil(dim*scale/16)*16)
    while high-low>=1e-5:
        mid=(low+high)/2
        if scaled(mid,height)*scaled(mid,width)//256<=256:low=mid
        else:high=mid
    return scaled(low,height),scaled(low,width)


def prepare_image(rgb,position_grid):
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('Expected uint8 RGB image')
    h,w=target_size(*rgb.shape[:2]);ph,pw=h//16,w//16
    image=resize_uint8(rgb,h,w).astype(np.float32)
    image=(image-np.float32(127.5))/np.float32(127.5)
    patches=image.reshape(ph,16,pw,16,3).transpose(0,2,1,3,4).reshape(ph*pw,768)
    pixels=np.zeros((1,256,768),dtype=np.float32);pixels[0,:ph*pw]=patches
    mask=np.zeros((1,256),dtype=np.float32);mask[0,:ph*pw]=1
    pos=resize_aa(position_grid,ph,pw).reshape(ph*pw,768)
    positions=np.broadcast_to(pos[0],(1,256,768)).copy();positions[0,:ph*pw]=pos
    return pixels,positions,mask


def normalize(y):
    if not np.isfinite(y).all():raise ValueError('Nonfinite embedding')
    norm=np.linalg.norm(y,axis=-1,keepdims=True)
    if np.any(norm<=0):raise ValueError('Zero embedding')
    return y/norm


class Encoder:
    def __init__(self,assets):
        assets=Path(assets)
        if int(cv2.__version__.split('.')[0])<5:raise RuntimeError('OpenCV 5 is required')
        cv2.setNumThreads(2)
        self.tokenizer=Tokenizer(assets/'model/tokenizer.json')
        self.grid=np.load(assets/'position-grid.npy',allow_pickle=False)
        self.assets=assets
        self.text=None
        self.vision=None

    def _network(self, name):
        net=getattr(self,name)
        if net is None:
            net=cv2.dnn.readNetFromONNX(str(self.assets/('text-raw.onnx' if name=='text' else 'vision-body.onnx')))
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            setattr(self,name,net)
        return net

    def encode_text(self,query):
        tokens,truncated=self.tokenizer.encode(query)
        net=self._network('text')
        net.setInput(tokens,'input_ids')
        return normalize(net.forward()),truncated

    def encode_rgb(self,image):
        net=self._network('vision')
        for name,x in zip(['pixel_values','position_values','pixel_attention_mask'],prepare_image(image,self.grid)):
            net.setInput(x,name)
        return normalize(net.forward())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,default=Path(__file__).resolve().parent/'artifacts')
    p.add_argument('--query',required=True);p.add_argument('images',nargs='+',type=Path)
    args=p.parse_args();encoder=Encoder(args.assets)
    text,truncated=encoder.encode_text(args.query)
    results=[]
    for path in args.images:
        bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if bgr is None:raise ValueError('Cannot read image: '+str(path))
        vector=encoder.encode_rgb(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        results.append({'path':str(path),'similarity':float((text@vector.T)[0,0])})
    print(json.dumps({'query_truncated':truncated,'results':sorted(results,key=lambda r:r['similarity'],reverse=True)},indent=2))
