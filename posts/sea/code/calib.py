import sys; sys.path.insert(0,'.')
import numpy as np
from sea.scene import scene
from sea.render import Cam
from stage1 import render, photo_view
from lockstep import photo_at
from lab import srgb_to_lab
P,S,zl=scene('poa','poa_summer.npz'); Hm=np.load('../notes/H_poa.npy')
cam=Cam(cx=720,cy=750,mpp=1.0,W=480,H=300,yaw=90,elev=30,zc=0.45)
ph=photo_at(Hm,720,750,1.0,480,300)
Lp=srgb_to_lab(ph.astype(float))[...,0]; v=ph.sum(2)>0
for g in [float(a) for a in sys.argv[1:]]:
    rgb,G=render(S,zl,cam,params=dict(foam_gain=g))
    Lm=srgb_to_lab(rgb.astype(float))[...,0]; m=v&G['sea']
    print(g,'white photo',round((Lp[m]>75).mean(),3),'mine',round((Lm[m]>75).mean(),3))
