"""Stage 3: branch stubs and root wads."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import geo, render
from paint import Light
from util import board

ROOT = os.path.expanduser('~/work/pixelart-studies/elements/driftwood')


def cell(az=300, el=40, W=140, H=80, scale=36, pitch=32, seed=3, yaw=140, L=3.0, r=0.2, rootwad=True,
         n_stubs=3, flat=True, look=(0.0, 0, 0.3), marks=True, overcast=False, ends=('snapped', 'worn'), **kw):
    cam = geo.Camera(W, H, scale, pitch=pitch, look=look)
    rw = dict(n=6, end=0) if rootwad else None
    lg = geo.make_log(L=L, r=r, yaw=yaw, seed=seed, ends=ends, rootwad=rw, n_stubs=n_stubs, **kw)
    li = Light(cam, az=az, el=el, overcast=overcast)
    cv, buf, sh = render.render([lg], cam, li, seed=seed, stone_m=0.06, opts=dict(gravel=not flat, marks=marks))
    return render.to_rgb(cv, render.grey_ramps())


if __name__ == "__main__" and len(sys.argv) == 1:
    t = time.time()
    P = [(f'seed {s}', cell(seed=s)) for s in (3, 5)]
    board(P, scale=4, cols=1).save(f'{ROOT}/scratch/s3_try.png')
    print(time.time() - t)


def boards():
    from PIL import Image
    from util import fit_photo
    from s0a_pale_trunk import load
    ph = fit_photo(f'{ROOT}/refs/openverse_Driftwood_log_on_sand_in_wind_7550a86a.jpg', (100, 60, 480, 400), (104, 90))
    d = load('V02'); f = d['pal'][d['idx'][358:412, 250:340]]
    mp = np.array(Image.open(os.path.expanduser('~/work/pixelart-studies/shared/refs/ferrari_mossy_forest_pool.png')).convert('RGB'))[::2, ::2]
    roots = mp[330:420, 150:260]
    board([('photo: root wad + forks', ph), ('mine: root wad log', cell(seed=3, W=110, H=90, scale=34, look=(0.6, 0.3, 0.3))),
           ('Ferrari V02: twigs as strokes', f), ('Ferrari forest pool: root flare', roots)], scale=4, cols=2,
          title='s3 stubs and root wads, 4x').save(f'{ROOT}/stages/s3_compare_4x.png')
    P = []
    for sd, az, el, lab in ((3, 300, 40, 'left-front'), (5, 60, 40, 'right-front'), (7, 180, 25, 'back'), (9, 0, 80, 'overcast')):
        P.append((f'seed {sd}, {lab}', cell(seed=sd, az=az, el=el, overcast=(lab == 'overcast'))))
    board(P, scale=4, cols=2, title='s3 root wads and stubs (flat ground), native x4').save(f'{ROOT}/stages/s3_rootwads_4x.png')
    P = []
    for sd in (2, 4, 6):
        P.append((f'stubs, seed {sd}', cell(seed=sd, rootwad=False, n_stubs=5, ends=('snapped', 'worn'), yaw=15)))
    board(P, scale=4, cols=1, title='s3 branch stubs, native x4').save(f'{ROOT}/stages/s3_stubs_4x.png')
    a = cell(seed=3)
    board([('8x: the root wad', a[5:75, 70:140])], scale=8).save(f'{ROOT}/stages/s3_detail_8x.png')


if __name__ == '__main__' and len(sys.argv) > 1:
    boards()
