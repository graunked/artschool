import os, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import numpy as np
from PIL import Image
from grasslod.world import World
from grasslod.render import Camera, Raster, value_design
from grasslod import palette as P

w = World(seed=1)
yc,hw = w.channel(np.array(0.0))
cy = float(yc)+hw+1.2
cz = float(w.height(np.array(0.0), np.array(cy)))
print("centre", cy, cz)
pal, sun = P.build("day")
tiles=[]
for ppm in [30, 10, 3, 1, 0.3]:
    t=time.time()
    cam = Camera(ppm=ppm, theta=30, W=320, H=200, cx=0, cy=cy, cz=cz)
    R = Raster(w, cam)
    V, lit, lam, graze = value_design(R, sun)
    rng=np.random.default_rng(0)
    q = np.clip(np.floor(V + rng.random(V.shape)), 0, 6).astype(int)
    idx = P.GREEN0 + q
    idx[R.water] = P.WATER0+2
    idx[R.sky] = P.SKY0+4
    idx[R.mat==3] = P.SOIL0
    tiles.append(P.to_rgb(idx, pal))
    print(ppm, time.time()-t)
img=np.concatenate([np.pad(t,((0,4),(0,0),(0,0)),constant_values=255) for t in tiles],0)
Image.fromarray(img).resize((img.shape[1]*2,img.shape[0]*2),Image.NEAREST).save("../img/dbg_value.png")
