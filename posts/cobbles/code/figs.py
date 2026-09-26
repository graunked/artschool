"""figs.py -- makes the teaching figures for the cobbles tutorial from the REAL study code.

It imports the study folder read-only (~/work/pixelart-studies/elements/cobbles) and writes only
into this post's img/.  Run:  OPENBLAS_NUM_THREADS=1 nice -n 10 python3 figs.py
"""
import sys, os
sys.dont_write_bytecode = True                     # don't litter the study folder with __pycache__
STUDY = os.path.expanduser('~/work/pixelart-studies/elements/cobbles')
sys.path.insert(0, STUDY)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'img')
import numpy as np
from PIL import Image
from stone import cobble_geom, paint_stone, paint_ground, paint_shadow_on_ground, contact_under, to_rgb, GREY9
from util import up, label, hcat, vcat, fit, ferrari
from palette import ramp, HOUR
from bar import bar


def save(im, name):
    im.save(os.path.join(OUT, name)); print('wrote', name)


def one_stone(D=40, sun=(140, 40), seed=2, P=None):
    """exactly stage1.one(), with the painter's options exposed."""
    g = cobble_geom(D, sun=sun, seed=seed)
    m, N, L = g['mask'], g['N'], g['L']
    out = np.zeros(m.shape, int)
    paint_ground(out, ~m)
    paint_shadow_on_ground(out, g['shadow'], m)
    paint_stone(out, m, L, dark_nb=g['shadow'] & ~m, N=N, P=P)
    if D >= 9:
        contact_under(out, m)
    return out


def fig_passes():
    """the lit stone, one pass at a time (the real paint_stone, with passes switched off)."""
    stages = [('1 face: N.L onto one\ndither transition', dict(band=None, lump=0, crescent=False, rim=False, crown_rows=0, turn=(-9, -9, -9))),
              ('2 + lumps, banded\n(planes, not gradient)', dict(crescent=False, rim=False, crown_rows=0, turn=(-9, -9, -9))),
              ('3 + turn: walk down\nnear the terminator', dict(crescent=False, rim=False, crown_rows=0)),
              ('4 + crescent, in px\nfrom the silhouette', dict(rim=False, crown_rows=0)),
              ('5 + crown and\nreflected rim: all', dict())]
    rows = []
    for sun in [(140, 40), (215, 35)]:
        outs = [one_stone(sun=sun, P=p) for t, p in stages]
        # crop every panel to the stone (+2 px) so the passes are visible at 8x
        diff = np.any(np.stack([o != outs[0] for o in outs] + [outs[-1] >= 4]), 0)
        ys, xs = np.nonzero(outs[-1] >= 4)
        box = (max(0, ys.min() - 2), ys.max() + 3, max(0, xs.min() - 2), xs.max() + 3)
        cells = [label(up(to_rgb(o[box[0]:box[1], box[2]:box[3]], GREY9), 8), f'{t}\nsun {sun}', sz=13)
                 for (t, p), o in zip(stages, outs)]
        rows.append(hcat(cells))
    save(vcat(rows, gap=10), 'passes.png')


def fig_hours():
    """one index canvas, five palettes: the palette is the light."""
    cv, info = bar(W=96, H=64, sun=(140, 40), seed=11, head_m=0.12, tail_m=0.12,
                   px_per_m_far=110, px_per_m_near=140, density=1.0)
    cells = [label(up(cv.render(h), 3), h, sz=13) for h in HOUR]
    save(hcat(cells), 'same_map_five_hours.png')


def fig_ramps():
    from PIL import ImageDraw
    rows = []
    for rock in ['granite', 'greywacke', 'basalt', 'gravel']:
        cells = []
        for h in HOUR:
            pal = ramp(rock, h)
            im = Image.new('RGB', (9 * 14, 22)); d = ImageDraw.Draw(im)
            for i, c in enumerate(pal):
                d.rectangle([i * 14, 0, i * 14 + 13, 21], fill=c)
            cells.append(label(im, f'{rock} / {h}', sz=11))
        rows.append(hcat(cells, gap=10))
    save(vcat(rows, gap=6), 'ramps.png')


def fig_ferrari_rank():
    save(label(up(ferrari('V19', (228, 252, 300, 300)), 8), 'Ferrari, Mountain Stream V19 (228,252) 72x48, 8x'), 'ferrari_v19_crop.png')


def fig_ladder():
    """which mark class paints each pixel of a bar: stone (painter), tick (lozenge stamp), grain."""
    cv, info = bar(W=160, H=96, sun=(140, 40), seed=4, head_m=0.2, tail_m=0.05,
                   px_per_m_far=14, px_per_m_near=110)
    img = cv.render('noon')
    cls = np.zeros(img.shape, np.uint8)
    stone = info['anyst']
    tick = (cv.mat >= 0) & ~stone
    cls[:] = (70, 70, 70)                                # grain (the matrix stipple)
    cls[tick] = (230, 170, 60)                           # tick stamps
    cls[stone] = (90, 160, 220)                          # full stone painter
    save(hcat([label(up(img, 4), 'bar, noon'),
               label(up(Image.fromarray(cls), 4), 'blue: stone painter (>= 5 px)   amber: tick (2-5 px)   grey: grain')]),
         'ladder_classes.png')


if __name__ == '__main__':
    which = sys.argv[1:] or ['passes', 'hours', 'ramps', 'ferrari', 'ladder']
    if 'passes' in which: fig_passes()
    if 'hours' in which: fig_hours()
    if 'ramps' in which: fig_ramps()
    if 'ferrari' in which: fig_ferrari_rank()
    if 'ladder' in which: fig_ladder()
