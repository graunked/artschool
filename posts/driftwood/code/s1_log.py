"""Stage 1: one log in greyscale -- lights, ends, sizes.  Writes stages/s1_*.png"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import geo, render
from paint import Light
from util import board

ROOT = os.path.expanduser('~/work/pixelart-studies/elements/driftwood')


def cell(az, el, flat=False, W=150, H=70, scale=40, pitch=30, seed=3, ends=('sawn', 'snapped'), yaw=10,
         L=3.2, r=0.17, overcast=False, look=(0, 0, 0), **kw):
    cam = geo.Camera(W, H, scale, pitch=pitch, look=look)
    lg = geo.make_log(L=L, r=r, yaw=yaw, seed=seed, ends=ends, **kw)
    li = Light(cam, az=az, el=el, overcast=overcast)
    cv, buf, sh = render.render([lg], cam, li, seed=2, stone_m=0.06, opts=dict(gravel=not flat))
    return render.to_rgb(cv, render.grey_ramps())


LIGHTS = [('left-front 40', 300, 40, False), ('right-front 40', 60, 40, False),
          ('back-left 30', 220, 30, False), ('back-right 25', 150, 25, False),
          ('high 70', 330, 70, False), ('overcast', 0, 80, True)]

if __name__ == '__main__':
    t = time.time()
    for flat in (True, False):
        P = [(n, cell(a, e, flat=flat, overcast=o)) for n, a, e, o in LIGHTS]
        tag = 'flat' if flat else 'gravel'
        board(P, scale=4, cols=2, title=f's1 one log, six lights ({tag} ground), 30 deg, 40 px/m, native x4'
              ).save(f'{ROOT}/stages/s1_lights_{tag}_4x.png')
    # ends: each end at the centre, two sizes
    P = []
    for k in ('sawn', 'worn', 'snapped', 'splinter'):
        for sc in (60, 30):
            a = np.radians(235); endp = 1.2 * np.array([np.cos(a), np.sin(a), 0.0])
            P.append((f'{k} end, {sc} px/m', cell(300, 40, flat=True, W=70, H=50, scale=sc, yaw=235, L=2.4,
                                                   r=0.2, ends=('sawn', k), look=tuple(endp * 0.6), seed=5)))
    board(P, scale=5, cols=2, title='s1 ends (left-front sun), native x5').save(f'{ROOT}/stages/s1_ends_5x.png')
    # sizes
    P = []
    for n in (120, 60, 32, 16, 8, 4):
        sc = n / 3.2
        P.append((f'{n} px long', cell(300, 40, flat=False, W=int(n * 1.2) + 10, H=max(14, int(n * 0.45) + 8),
                                       scale=sc)))
    board(P, scale=4, cols=3, title='s1 sizes, native x4').save(f'{ROOT}/stages/s1_sizes_4x.png')
    print('done', time.time() - t)
