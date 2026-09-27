"""Register an oblique photo to the DEM's sea plane with a homography from hand-picked pairs.
map coords: (col,row) metres in a frame with NW origin (X0,Y1) UTM. photo coords: full-res px."""
import numpy as np, json, sys
from PIL import Image, ImageDraw
from dem import window
def homog(src, dst):
    A=[]
    for (x,y),(u,v) in zip(src,dst):
        A.append([-x,-y,-1,0,0,0,u*x,u*y,u]); A.append([0,0,0,-x,-y,-1,v*x,v*y,v])
    _,_,Vt=np.linalg.svd(np.array(A,float)); H=Vt[-1].reshape(3,3); return H/H[2,2]
def apply(H,x,y):
    p=H@np.stack([x,y,np.ones_like(x)]); return p[0]/p[2],p[1]/p[2]
def overlay(photo, H, X0, Y1, box, out, lvl=0.0, scale=0.4):
    c0,r0,c1,r1=box
    a=window('olycoast','elev',X0+c0,Y1-r1,X0+c1,Y1-r0)
    im=Image.open(photo).convert('RGB'); W,Hh=im.size
    sm=im.resize((int(W*scale),int(Hh*scale)))
    d=ImageDraw.Draw(sm)
    rr,cc=np.nonzero(a>lvl)
    # edge pixels only
    m=a>lvl; e=m&~(np.roll(m,1,0)&np.roll(m,-1,0)&np.roll(m,1,1)&np.roll(m,-1,1))
    rr,cc=np.nonzero(e)
    u,v=apply(H,cc+c0+0.5,rr+r0+0.5)
    for x,y in zip(u*scale,v*scale):
        if 0<=x<sm.width and 0<=y<sm.height: sm.putpixel((int(x),int(y)),(255,0,255))
    for g in range(c0,c1,100):
        rs=np.arange(r0,r1,5.0); u,v=apply(H,np.full_like(rs,g),rs); d.line(list(zip(u*scale,v*scale)),fill=(255,255,0))
    for g in range(r0,r1,100):
        cs=np.arange(c0,c1,5.0); u,v=apply(H,cs,np.full_like(cs,g)); d.line(list(zip(u*scale,v*scale)),fill=(0,255,255))
    sm.save(out)
def warp(photo, H, box, mpp, out=None):
    """photo -> map-space image (top-down), box in map metres."""
    c0,r0,c1,r1=box
    cs=np.arange(c0,c1,mpp)+mpp/2; rs=np.arange(r0,r1,mpp)+mpp/2
    C,R=np.meshgrid(cs,rs); u,v=apply(H,C.ravel(),R.ravel())
    im=np.asarray(Image.open(photo).convert('RGB'))
    ok=(u>=0)&(u<im.shape[1]-1)&(v>=0)&(v<im.shape[0]-1)
    o=np.zeros((C.size,3),np.uint8); o[ok]=im[v[ok].astype(int),u[ok].astype(int)]
    o=o.reshape(C.shape+(3,))
    if out: Image.fromarray(o).save(out)
    return o, ok.reshape(C.shape)
def overlay2(photo, H, X0, Y1, box, crop, out, lvl=0.45, labels=()):
    """draw forward-projected rock mask (above lvl) as translucent magenta on a full-res photo crop."""
    c0,r0,c1,r1=box
    a=window('olycoast','elev',X0+c0,Y1-r1,X0+c1,Y1-r0)
    im=np.asarray(Image.open(photo).convert('RGB')).astype(float)
    x0,y0,x1,y1=crop; sub=im[y0:y1,x0:x1].copy()
    # inverse map each crop pixel to map coords
    Hi=np.linalg.inv(H); yy,xx=np.mgrid[y0:y1,x0:x1]
    c,r=apply(Hi,xx.ravel().astype(float),yy.ravel().astype(float))
    c=c.reshape(xx.shape)-c0; r=r.reshape(xx.shape)-r0
    ok=(c>=0)&(c<a.shape[1]-1)&(r>=0)&(r<a.shape[0]-1)
    z=np.full(xx.shape,-99.0); z[ok]=a[r[ok].astype(int),c[ok].astype(int)]
    m=z>lvl; e=m&~(np.roll(m,2,0)&np.roll(m,-2,0)&np.roll(m,2,1)&np.roll(m,-2,1))
    sub[e]=[255,0,255]
    m3=(z>-3)&(z<=lvl); e3=m3&~(np.roll(m3,1,0)&np.roll(m3,-1,0)&np.roll(m3,1,1)&np.roll(m3,-1,1)); sub[e3]=sub[e3]*0.3+np.array([255,255,0])*0.7
    o=Image.fromarray(sub.astype(np.uint8)); d=ImageDraw.Draw(o)
    for (mc,mr),name in labels:
        u,v=apply(H,np.array([mc],float),np.array([mr],float)); d.text((u[0]-x0,v[0]-y0),name,fill=(255,255,255))
    o.save(out)
