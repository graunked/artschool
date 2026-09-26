"""Stage 5: a pile -- logs crossing and resting on each other, on gravel, at three distances."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import geo, render, palette, pile
from paint import Light
from util import board, fit_photo

ROOT = os.path.expanduser('~/work/pixelart-studies/elements/driftwood')
PHOTO = os.path.expanduser('~/work/pixelart-studies/transfer/refs/braided_driftwood_bar.jpg')


def pile_cell(seed=1, scale=30, W=160, H=100, pitch=30, hour='noon', az=300, el=45, n=9, flat=False,
              look=(0, 0, 0.3), stone_m=0.06, grey=False, **kw):
    cam = geo.Camera(W, H, scale, pitch=pitch, look=look)
    logs = pile.make_pile(seed=seed, n=n, **kw)
    li = Light(cam, az=az, el=el, overcast=(hour == 'overcast'))
    cv, buf, sh = render.render(logs, cam, li, seed=seed, stone_m=stone_m, opts=dict(gravel=not flat))
    if grey:
        return render.to_rgb(cv, render.grey_ramps())
    return render.to_rgb(cv, palette.ramps(hour, 0.0))


if __name__ == "__main__" and len(sys.argv) == 1:
    t = time.time()
    a = pile_cell(grey=True, flat=True)
    b = pile_cell()
    board([('grey, flat ground', a), ('colour, gravel', b)], scale=4, cols=1).save(f'{ROOT}/scratch/s5_try.png')
    print(time.time() - t)


def boards():
    from PIL import Image
    from s0a_pale_trunk import load
    P = []
    for sc, lab in ((32, 'near, 32 px/m'), (14, 'mid, 14 px/m'), (6, 'far, 6 px/m'), (3, 'distant, 3 px/m')):
        P.append((lab, pile_cell(seed=4, scale=sc, n=14, W=160, H=90, stone_m=max(0.06, 1.6 / sc),
                                 extent=(6.0, 3.0))))
    board(P, scale=3, cols=2, title='s5 one pile at four distances (noon), native x3').save(f'{ROOT}/stages/s5_distance_3x.png')
    ph = fit_photo(PHOTO, (0, 330, 560, 480), (160, 43))
    ph2 = fit_photo(PHOTO, (0, 380, 480, 560), (160, 60))
    mine = pile_cell(seed=6, scale=20, n=18, W=160, H=60, stone_m=0.08, extent=(11.0, 6.0), pitch=22,
                     look=(0, 0, 0.3), field=True, az=300, el=45, crossing=0.15)
    d = load('V19'); f = d['pal'][d['idx'][250:310, 225:385]]
    board([('photo: the braided bar pile', ph2), ('mine: a strewn bar, 20 px/m, pitch 22', mine),
           ('Ferrari V19: a pile of forms separated by value', f)], scale=4, cols=1,
          title='s5 pile: photo | pixels | Ferrari (4x)').save(f'{ROOT}/stages/s5_compare_4x.png')
    P = []
    for sd in (1, 2, 3, 5):
        P.append((f'seed {sd}', pile_cell(seed=sd, n=12, grey=True, flat=True)))
    board(P, scale=3, cols=2, title='s5 piles in greyscale on plain ground (value separation), native x3').save(f'{ROOT}/stages/s5_grey_3x.png')


if __name__ == '__main__' and len(sys.argv) > 1:
    boards()
