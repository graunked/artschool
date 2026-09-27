import sys; sys.path.insert(0,'.')
import numpy as np
from PIL import Image
from sea.scene import scene
from sea.render import Cam
from stage1 import render
P,S,zl=scene('alava','alava_summer.npz')
mpp=float(sys.argv[1]); cx,cy=float(sys.argv[2]),float(sys.argv[3]); out=sys.argv[4]
cam=Cam(cx=cx,cy=cy,mpp=mpp,W=480,H=300,yaw=90,elev=30,zc=0.45)
rgb,G=render(S,zl,cam)
Image.fromarray(rgb).resize((1440,900),Image.NEAREST).save(out)
