"""Stage 4: colour -- silver-grey wood under warm and cool light; wet versus dry."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import geo, render, palette
from paint import Light
from util import board, fit_photo

ROOT = os.path.expanduser('~/work/pixelart-studies/elements/driftwood')
PHOTO = os.path.expanduser('~/work/pixelart-studies/transfer/refs/braided_driftwood_bar.jpg')
HOUR_LIGHT = {'noon': (300, 55), 'morning': (60, 25), 'golden': (285, 12), 'overcast': (0, 80)}


def cell(hour='noon', wet=0.0, W=140, H=80, scale=36, pitch=32, seed=3, yaw=15, L=3.0, r=0.2,
         rootwad=True, n_stubs=2, flat=False, look=(-0.2, 0, 0.2), az=None, el=None, ends=('snapped', 'worn')):
    cam = geo.Camera(W, H, scale, pitch=pitch, look=look)
    rw = dict(n=6, end=0) if rootwad else None
    lg = geo.make_log(L=L, r=r, yaw=yaw, seed=seed, ends=ends, rootwad=rw, n_stubs=n_stubs, wet=wet)
    a, e = HOUR_LIGHT[hour]
    li = Light(cam, az=az if az is not None else a, el=el if el is not None else e, overcast=(hour == 'overcast'))
    cv, buf, sh = render.render([lg], cam, li, seed=seed, stone_m=0.06, opts=dict(gravel=not flat))
    return render.to_rgb(cv, palette.ramps(hour, wet))


if __name__ == '__main__':
    t = time.time()
    P = []
    for hour in ('noon', 'morning', 'golden', 'overcast'):
        for wet in (0.0, 1.0):
            P.append((f'{hour}, {"wet" if wet else "dry"}', cell(hour, wet)))
    board(P, scale=4, cols=2, title='s4 colour: hour x wetness (native x4)').save(f'{ROOT}/stages/s4_hours_wet_4x.png')
    from s0a_pale_trunk import load
    d = load('V09'); f = d['pal'][d['idx'][150:222, 358:414]]
    ph = fit_photo(PHOTO, (310, 470, 526, 638), (72, 56))
    ph2 = fit_photo(f'{ROOT}/refs/commons_Cliffs_and_driftwood_at_St_ngehuvud_b_jpg_10933a5d.jpg', (150, 540, 1100, 730), (100, 20))
    a = cell('noon', 0.0); b = cell('noon', 1.0)
    board([('photo (braided bar, /3)', ph), ('mine noon dry', a[10:66, 60:132]), ('mine noon wet', b[10:66, 60:132]),
           ('Ferrari V09 pale trunk', f[:56])], scale=6, cols=4, title='s4 colour at 6x').save(f'{ROOT}/stages/s4_compare_6x.png')
    print(time.time() - t)
