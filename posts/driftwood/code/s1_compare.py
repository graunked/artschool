"""Stage 1 comparison: photo crop | my log | Ferrari trunk (rotated to lie down), at 8x."""
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


def hero(W=72, H=56, scale=19.0, az=280, el=55, pitch=40, seed=3, yaw=40, r=0.3, L=3.6,
         ends=('sawn', 'worn'), flat=False):
    cam = geo.Camera(W, H, scale, pitch=pitch, look=(-0.1, 0.0, 0.2))
    lg = geo.make_log(L=L, r=r, yaw=yaw, seed=seed, ends=ends, bend=0.07, taper=0.7)
    li = Light(cam, az=az, el=el)
    cv, buf, sh = render.render([lg], cam, li, seed=seed, stone_m=0.06, opts=dict(gravel=not flat))
    return render.to_rgb(cv, render.grey_ramps())


if __name__ == '__main__':
    ph = fit_photo(PHOTO, (310, 470, 526, 638), (72, 56))
    phg = np.repeat(np.array(Image.fromarray(ph).convert('L'))[..., None], 3, -1)
    mine = hero(flat=True)
    mine_g = hero()
    d = load('V02'); f = d['pal'][d['idx'][358:414, 250:322]]
    board([('photo (grey, /3)', phg), ('mine, plain ground', mine), ('mine, gravel', mine_g),
           ('Ferrari V02 fallen trunk', f)], scale=6, cols=4,
          title='s1 one log: photo | pixels | Ferrari (6x)').save(f'{ROOT}/stages/s1_compare_6x.png')
