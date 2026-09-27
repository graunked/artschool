"""Sheet C: the surf vocabulary around a headland and a stack (Point of the Arches, swell, sun).
Top-left: the physics map (wave height, breaking, crest lines). Others: labelled crops at 0.3 and 1 m/px."""
import sys; sys.path.insert(0,'.')
import numpy as np
from PIL import Image, ImageDraw
from sheets import get, params
from sea.render import Cam
from stage1 import render
P,S,zl=get('poa','swell')
d=np.load('../cache/poa_swell.npz')
H=d['H'].astype(float); Q=d['Q'].astype(float); L=d['land']; tt=d['t'].astype(float)
# physics map, crop around the cove
sl=(slice(300,1100),slice(250,1150))
rgb=np.zeros(H[sl].shape+(3,))
rgb[...,2]=0.25+0.6*np.clip(H[sl]/1.3,0,1); rgb[...,1]=0.15+0.3*np.clip(H[sl]/1.3,0,1)
rgb[...,0]=np.clip(Q[sl]*3,0,1)
ph=np.mod(tt[sl]/8.3,1.0); crest=(ph<0.05)&~L[sl]; rgb[crest]=[1,1,1]
rgb[L[sl]]=[0.45,0.42,0.38]
phys=Image.fromarray((rgb*255).astype(np.uint8)).resize((450,400))
dd=ImageDraw.Draw(phys); dd.text((6,6),'swell physics: blue = wave height, red = breaking,\nwhite = crest lines (every period)',fill=(255,255,255))
crops=[('stack: surge collar on the weather face, jade bubble\npool, filaments curling into the lee',0.3,(720,750)),
       ('south point: saturated surf on the shelf, one bright bore\nband per wave, old foam marbling behind it',0.3,(694,1086)),
       ('outer islet off the north headland (largest Hrms near land):\ncollar on the weather side, calm jade lee',1.0,(204,221)),
       ('cove: waves spread and shrink; a quiet bay,\nsurf only on the reef heads and the steep faces',1.0,(900,760)),
       ('foam shed from the south rocks, drifting north on the\ncurrent (the photo streaks are 2-3x longer: a miss)',1.0,(443,1050)),
       ('the whole cove at region scale',3.0,(720,750))]
W,Hh=225,200
cells=[phys]
for lab,m,(cx,cy) in crops:
    cam=Cam(cx=cx,cy=cy,mpp=m,W=W,H=Hh,yaw=90,elev=30,zc=0.45)
    img,G=render(S,zl,cam,params=params('sun','swell'))
    im=Image.fromarray(img).resize((W*2,Hh*2),Image.NEAREST)
    di=ImageDraw.Draw(im); di.rectangle([0,0,W*2,30],fill=(18,18,22)); di.text((4,2),lab+f'  [{m} m/px]',fill=(235,235,235))
    cells.append(im)
out=Image.new('RGB',(3*460,3*410+30),(18,18,22)); do=ImageDraw.Draw(out)
do.text((6,6),'Sheet C: surf vocabulary around a headland and a stack. Point of the Arches (real lidar), swell Hs 1.6 m 8.3 s from 281 deg, sun',fill=(235,235,235))
for i,c in enumerate(cells):
    out.paste(c.resize((450,400),Image.NEAREST) if i==0 else c,( (i%3)*460, 30+(i//3)*410))
out.save('../sheets/C_surf_vocabulary.png')
