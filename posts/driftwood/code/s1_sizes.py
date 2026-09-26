"""Stage 1c: the same log from 120 px long down to a few pixels."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import geo, render
from paint import Light
from util import board

ROOT = os.path.expanduser('~/work/pixelart-studies/elements/driftwood')


def size_cell(length_px, W=None, H=None, az=300, el=40, pitch=30, seed=5, yaw_log=12, L=3.0, r=0.16,
              ends=('sawn', 'snapped')):
    scale = length_px / L
    W = W or int(length_px * 1.15) + 8; H = H or max(12, int(length_px * 0.45) + 6)
    cam = geo.Camera(W, H, scale, pitch=pitch, look=(0, 0, 0.1))
    a = np.radians(yaw_log); ax = np.array([np.cos(a), np.sin(a), 0.0])
    lg = geo.Log([-ax * L / 2 + [0, 0, r - 0.02], ax * L / 2 + [0, 0, r * 0.8 - 0.02]], [r, r * 0.8],
                 ends=ends, seed=seed)
    li = Light(cam, az=az, el=el)
    cv, buf, sh = render.render([lg], cam, li, seed=seed, stone_m=0.06)
    return render.to_rgb(cv, render.grey_ramps())


if __name__ == '__main__':
    P = [(f'{n} px long', size_cell(n)) for n in (120, 60, 32, 16, 8, 4)]
    board(P, scale=4, cols=2, title='s1 sizes (native x4)').save(f'{ROOT}/scratch/s1_sizes.png')
