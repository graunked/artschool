"""Stage 1 hero: one log in greyscale; photo | pixels | Ferrari at 8x, plus the light sweep."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
import geo, render
from paint import Light
from util import board, fit_photo
from s0a_pale_trunk import load

ROOT = os.path.expanduser('~/work/pixelart-studies/elements/driftwood')
PHOTO = os.path.expanduser('~/work/pixelart-studies/transfer/refs/braided_driftwood_bar.jpg')


def hero(az=300, el=40, W=150, H=70, scale=40, pitch=30, seed=3, ends=('sawn', 'snapped'), yaw=10,
         L=3.2, r=0.17, flat=False, **kw):
    cam = geo.Camera(W, H, scale, pitch=pitch)
    lg = geo.make_log(L=L, r=r, yaw=yaw, seed=seed, ends=ends, **kw)
    cv, buf, sh = render.render([lg], cam, Light(cam, az=az, el=el), seed=2, stone_m=0.06,
                                opts=dict(gravel=not flat))
    return render.to_rgb(cv, render.grey_ramps())


if __name__ == '__main__':
    a = hero()
    board([('left-front 40, 8x crop', a[15:60, 0:75]), ('right half', a[15:60, 75:150])], scale=8, cols=1).save(f'{ROOT}/scratch/s1_hero8.png')
