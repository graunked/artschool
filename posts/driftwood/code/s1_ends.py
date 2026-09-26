"""Stage 1b: the ends -- sawn, snapped, splintered -- at two sizes."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import geo, render
from paint import Light
from util import board

ROOT = os.path.expanduser('~/work/pixelart-studies/elements/driftwood')


def end_cell(kind, scale=60.0, W=64, H=48, az=300, el=40, seed=4, yaw_log=200, r=0.22, pitch=25):
    cam = geo.Camera(W, H, scale, pitch=pitch, look=(0.25, 0, 0.1))
    a = np.radians(yaw_log)
    ax = np.array([np.cos(a), np.sin(a), 0.0])
    # the END is at the image centre, the log recedes behind it
    p1 = np.array([0.0, 0.0, r - 0.02]); p0 = p1 - ax * 2.5
    lg = geo.Log([p0 + [0, 0, 0.0], p1], [r * 1.05, r], ends=('sawn', kind), seed=seed)
    li = Light(cam, az=az, el=el)
    cv, buf, sh = render.render([lg], cam, li, seed=seed)
    return render.to_rgb(cv, render.grey_ramps()), cv, buf


if __name__ == '__main__':
    P = []
    for k in ('sawn', 'snapped', 'splinter'):
        for sc in (60, 30):
            P.append((f'{k} {sc}px/m', end_cell(k, scale=sc)[0]))
    board(P, scale=6, cols=2, title='s1 ends').save(f'{ROOT}/scratch/s1_ends.png')
