import os, sys, time
import numpy as np
from PIL import Image
from grasslod.world import World
from grasslod.render import Camera
from grasslod.painter import paint, centre_on_bank
from grasslod import palette as P

w = World(seed=1)
cy, cz = centre_on_bank(w)
out=[]
for ppm in [float(a) for a in (sys.argv[1:] or ["30"])]:
    t=time.time()
    cam = Camera(ppm=ppm, theta=30, W=320, H=200, cx=0, cy=cy, cz=cz)
    idx, pal = paint(w, cam, "day")
    print(ppm, round(time.time()-t,1))
    out.append(P.to_rgb(idx, pal))
img=np.concatenate(out,0)
Image.fromarray(img).resize((img.shape[1]*3,img.shape[0]*3),Image.NEAREST).save("../img/wip.png")
