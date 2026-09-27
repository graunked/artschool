import numpy as np, sys
from PIL import Image, ImageDraw
from dem import *
def hillshade(a):
    gy,gx=np.gradient(a); n=np.dstack([-gx,gy,np.ones_like(a)]); n/=np.linalg.norm(n,axis=2,keepdims=True)
    L=np.array([-0.5,0.5,0.7]);L/=np.linalg.norm(L); return np.clip(n@L,0,1)
def view(lon,lat,half,out,step=1,grid=100):
    x,y=lonlat_to_utm(lon,lat); x0,y0=int(x-half),int(y-half)
    a=window('olycoast','elev',x0,y0,x0+2*half,y0+2*half)[::step,::step]
    h=hillshade(np.nan_to_num(a,nan=0)/step)
    rgb=np.zeros(a.shape+(3,))
    sea=a<0
    d=np.clip(-a/25,0,1)
    rgb[sea]=np.stack([0.2+0.5*h[sea]*(1-d[sea]),0.4+0.5*h[sea]*(1-d[sea]),0.6+0.3*h[sea]],-1)
    land=~sea; rgb[land]=np.stack([h[land]*0.9+0.1*np.clip(a[land]/40,0,1)]*3,-1)*np.array([1,0.95,0.8])
    im=Image.fromarray((rgb*255).astype(np.uint8)); d_=ImageDraw.Draw(im)
    for g in range(0,im.width,grid//step): d_.line([(g,0),(g,im.height)],fill=(255,0,0)); d_.text((g+2,2),str(g*step),fill=(255,255,0))
    for g in range(0,im.height,grid//step): d_.line([(0,g),(im.width,g)],fill=(255,0,0)); d_.text((2,g+2),str(g*step),fill=(255,255,0))
    im.save(out); print('origin NW utm',x0,y0+2*half)
if __name__=='__main__':
    view(float(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3]),sys.argv[4],int(sys.argv[5]) if len(sys.argv)>5 else 1)
