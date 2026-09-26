"""ladder.py out.png ppm1 ppm2 ... : same bank at several zooms, stacked, with mean luminance."""
import sys, time
import numpy as np
from PIL import Image, ImageDraw
from grasslod.world import World
from grasslod.render import Camera
from grasslod.painter import paint, centre_on_bank
from grasslod import palette as P

def lum(rgb):
    return (0.2126*rgb[...,0]+0.7152*rgb[...,1]+0.0722*rgb[...,2])

def ladder(out, ppms, hour="day", theta=30, W=320, H=160, scale=2, world=None, **kw):
    w = world or World(seed=1)
    cy, cz = centre_on_bank(w)
    tiles=[]
    for ppm in ppms:
        t=time.time()
        cam = Camera(ppm=ppm, theta=theta, W=W, H=H, cx=0, cy=cy, cz=cz)
        idx, pal, L = paint(w, cam, hour, return_layers=True, **kw)
        rgb = P.to_rgb(idx, pal)
        g = L['R'].mat==1
        print(f"ppm {ppm:6.2f} ell {getattr(L['R'],'ell',0):5.2f} keep {getattr(L['R'],'keep',0):.4f} "
              f"grass-lum {lum(rgb.astype(float))[g].mean():6.1f}  {time.time()-t:.1f}s", flush=True)
        im = Image.fromarray(rgb).resize((W*scale,H*scale),Image.NEAREST)
        d = ImageDraw.Draw(im); d.rectangle((0,0,70,11),fill=(0,0,0)); d.text((2,0),f"{ppm:g} px/m",fill=(255,255,255))
        tiles.append(np.array(im))
    img=np.concatenate([np.pad(t,((0,3),(0,0),(0,0))) for t in tiles],0)
    Image.fromarray(img).save(out)

if __name__=="__main__":
    ladder(sys.argv[1], [float(a) for a in sys.argv[2:]])
