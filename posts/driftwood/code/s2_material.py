"""Stage 2: the wood's material -- grain, checks, knots; fresh bark for contrast."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import geo, render
from paint import Light
from util import board

ROOT = os.path.expanduser('~/work/pixelart-studies/elements/driftwood')


def cell(az=300, el=40, W=120, H=64, scale=50, pitch=35, seed=3, ends=('sawn', 'worn'), yaw=12, L=2.2,
         r=0.2, kind='drift', weather=1.0, marks=True, flat=True, look=(0, 0, 0.1), overcast=False, **kw):
    cam = geo.Camera(W, H, scale, pitch=pitch, look=look)
    lg = geo.make_log(L=L, r=r, yaw=yaw, seed=seed, ends=ends, kind=kind, **kw)
    li = Light(cam, az=az, el=el, overcast=overcast)
    cv, buf, sh = render.render([lg], cam, li, seed=seed, stone_m=0.06,
                                opts=dict(gravel=not flat, marks=marks, weather=weather))
    return render.to_rgb(cv, render.grey_ramps())


if __name__ == '__main__':
    t = time.time()
    P = [('no marks', cell(marks=False)), ('grain + checks + knots', cell()),
         ('seed 7', cell(seed=7)), ('fresh bark (inverse)', cell(kind='bark'))]
    board(P, scale=5, cols=2, title='s2 material').save(f'{ROOT}/scratch/s2_try.png')
    print(time.time() - t)


def boards():
    from PIL import Image
    from util import fit_photo
    from s0a_pale_trunk import load
    d = load('V09'); grey = d['pal'][d['idx'][60:200, 452:486]]
    grey = np.rot90(grey, 1)
    sf = np.array(Image.open(os.path.expanduser('~/work/pixelart-studies/shared/refs/ferrari_snow_forest_boulders.webp')).convert('RGB'))
    sf = sf[::2, ::2][150:330, 400:480]
    sf = np.rot90(sf, 1)[:, :140]
    ph = fit_photo(f'{ROOT}/refs/commons_Cliffs_and_driftwood_at_St_ngehuvud_b_jpg_10933a5d.jpg', (150, 540, 1100, 730), (140, 28))
    ph2 = fit_photo(f'{ROOT}/refs/openverse_Driftwood_on_Gravel_bar_540722fe.jpg', (250, 180, 850, 768), (70, 69))
    a = cell(W=140, H=56, look=(0, 0, 0.1))
    b = cell(W=140, H=56, kind='bark', look=(0, 0, 0.1))
    board([('photo (Stangehuvud, /7)', ph), ('Ferrari V09 grey trunk, laid down', grey[:, :140]),
           ('mine: silvered driftwood', a[12:48])], scale=5, cols=1,
          title='s2 silvered wood at 5x').save(f'{ROOT}/stages/s2_compare_5x.png')
    board([('mine: fresh bark (inverse of driftwood)', b[12:48]), ('Ferrari snow-forest bark, laid down', sf[:40])],
          scale=5, cols=1, title='s2 fresh bark at 5x').save(f'{ROOT}/stages/s2_bark_5x.png')
    P = []
    for sd in (3, 7, 11, 4):
        for w, lab in ((0.4, 'light weathering'), (1.0, 'heavy weathering')):
            P.append((f'seed {sd}, {lab}', cell(seed=sd, weather=w)))
    board(P, scale=4, cols=2, title='s2 grain, checks, knots (flat ground), native x4').save(f'{ROOT}/stages/s2_material_4x.png')
    a = cell(W=80, H=48, scale=90, look=(0.3, 0, 0.1))
    board([('8x detail, 90 px/m', a)], scale=8).save(f'{ROOT}/stages/s2_detail_8x.png')


if __name__ == '__main__' and len(sys.argv) > 1:
    boards()
