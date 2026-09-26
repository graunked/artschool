import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(HERE)
def load_ref():
    return np.array(Image.open(os.path.join(ROOT,'img','00_native.png')).convert('RGB'))
def up(a,s): return np.repeat(np.repeat(a,s,0),s,1)
def save(a,name,s=1):
    if a.dtype!=np.uint8: a=np.clip(a,0,255).astype(np.uint8)
    if a.ndim==2: a=np.stack([a]*3,-1)
    Image.fromarray(up(a,s)).save(os.path.join(ROOT,'img',name))
def srgb2lin(c):
    c=np.asarray(c,float)/255; return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
def lum(a):
    """relative luminance -> CIE L* (0..100)"""
    l=srgb2lin(a); Y=l[...,0]*0.2126+l[...,1]*0.7152+l[...,2]*0.0722
    return np.where(Y>0.008856,116*np.cbrt(Y)-16,903.3*Y)
def rgb2lab(a):
    l=srgb2lin(a)
    M=np.array([[0.4124,0.3576,0.1805],[0.2126,0.7152,0.0722],[0.0193,0.1192,0.9505]])
    xyz=l@M.T/np.array([0.95047,1.0,1.08883])
    f=np.where(xyz>0.008856,np.cbrt(xyz),7.787*xyz+16/116)
    L=116*f[...,1]-16; A=500*(f[...,0]-f[...,1]); B=200*(f[...,1]-f[...,2])
    return np.stack([L,A,B],-1)
def label(img,text,pos=(4,4),fill=(255,255,255),bg=(0,0,0)):
    im=Image.fromarray(img) if isinstance(img,np.ndarray) else img
    d=ImageDraw.Draw(im)
    try: f=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',14)
    except: f=ImageFont.load_default()
    bb=d.textbbox(pos,text,font=f); d.rectangle([bb[0]-2,bb[1]-2,bb[2]+2,bb[3]+2],fill=bg); d.text(pos,text,fill=fill,font=f)
    return np.array(im)
