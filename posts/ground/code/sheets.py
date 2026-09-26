"""Labelled parameter sheets.  nice -n 10 python3 sheets.py [name ...]"""
import sys, os, numpy as np
from PIL import Image, ImageDraw
from concurrent.futures import ProcessPoolExecutor
import ground
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, '../sheets')

def cell(kw):
    kw = dict(kw); W = kw.pop('W', 200); H = kw.pop('H', 130); kw.pop('label', None)
    return ground.paint_ground(W, H, **kw).rgb

def sheet(name, rows, rlabels, clabels, title, k=3):
    flat = [c for r in rows for c in r]
    with ProcessPoolExecutor(4) as ex:
        ims = list(ex.map(cell, flat))
    H, W = ims[0].shape[:2]
    nr, nc = len(rows), len(rows[0])
    top, left, gap = 40, 110, 8
    S = Image.new('RGB', (left + nc * (W * k + gap), top + nr * (H * k + gap) + 20), (22, 22, 26))
    d = ImageDraw.Draw(S)
    d.text((8, 8), title, fill=(230, 230, 230))
    for j, cl in enumerate(clabels):
        d.text((left + j * (W * k + gap) + 4, top - 14), cl, fill=(200, 200, 200))
    for i, rl in enumerate(rlabels):
        d.text((6, top + i * (H * k + gap) + 6), rl, fill=(200, 200, 200))
        for j in range(nc):
            im = Image.fromarray(ims[i * nc + j]).resize((W * k, H * k), Image.NEAREST)
            S.paste(im, (left + j * (W * k + gap), top + i * (H * k + gap)))
    S.save(os.path.join(OUT, f'{name}.png'))
    print('wrote', name)

BASE = dict(landform='rolling', ppm=24, pitch=12, cam_h=1.7, seed=8, W=200, H=130)

def s_ground_hour():
    grounds = ['turf', 'meadow', 'earth', 'gravel', 'forest']
    hours = ['morning', 'noon', 'golden', 'dusk', 'overcast']
    extra = {'forest': dict(shade=0.7, shade_scale=4, canopy=0.2, canopy_scale=0.7, moisture=0.3)}
    rows = [[{**BASE, 'ground': g, 'hour': h, 'shade': 0.3, 'shade_scale': 10, **extra.get(g, {}),
              **({'sun': (80, 2)} if h == 'dusk' else {})} for h in hours] for g in grounds]
    sheet('A_ground_x_hour', rows, grounds, hours, 'Sheet A: ground type (rows) x hour (cols); rolling land, seed 8, ppm 24, off-frame shade 0.3', 2)

def s_landform_light():
    lfs = ['flat', 'rolling', 'hummocks', 'bank', 'ledges']
    suns = [(-60, 35), (0, 20), (60, 12), (85, 5)]
    rows = [[{**BASE, 'ground': 'turf', 'hour': 'golden', 'landform': lf, 'sun': s, 'relief': 1.2} for s in suns] for lf in lfs]
    sheet('B_landform_x_sun', rows, lfs, ['sun behind camera 35', 'sun left 20', 'sun left-behind 12', 'contre-jour 5'],
          'Sheet B: landform (rows) x sun (cols); turf, golden palette, no off-frame shade -- masses from land and light alone', 2)

def s_distance():
    ps = [(120, 35), (40, 20), (12, 12), (4, 8), (1.2, 5)]
    gs = ['turf', 'earth', 'gravel', 'forest']
    extra = {'forest': dict(shade=0.7, shade_scale=4, canopy=0.2, canopy_scale=0.7)}
    rows = [[dict(landform='rolling', path=(g == 'turf'), ground=g, hour='golden', sun=(15, 14), ppm=p, pitch=pi,
                  cam_h=1.7 * (40 / p) ** 0.6, relief=0.8, W=200, H=130, seed=8,
                  **{'shade': 0.3, 'shade_scale': 12, **extra.get(g, {})}) for p, pi in ps] for g in gs]
    sheet('C_distance', rows, gs, [f'{p} px/m, pitch {pi}' for p, pi in ps],
          'Sheet C: distance ladder (px per metre at screen centre); golden, seed 8', 2)

def s_moisture():
    ms = [0.0, 0.35, 0.7, 1.0]
    gs = ['earth', 'turf', 'forest']
    rows = [[dict(landform='rolling', ground=g, hour='noon', ppm=30, pitch=16, cam_h=1.7, relief=0.8, moisture=m,
                  path=(g == 'turf'), shade=0.25, shade_scale=8, W=200, H=130, seed=5) for m in ms] for g in gs]
    sheet('D_moisture', rows, gs, [f'moisture {m}' for m in ms], 'Sheet D: moisture (cols); wet ground drops a step, puddles in hollows take the sky', 2)

def s_seeds():
    rows = [[{**BASE, 'ground': 'turf', 'hour': 'golden', 'seed': sd, 'shade': 0.35, 'shade_scale': 12, 'path': True, 'sun': (15, 12)}
             for sd in range(1, 6)]]
    rows.append([{**r, 'hour': 'noon', 'sun': None} for r in rows[0]])
    sheet('E_seeds', rows, ['golden', 'noon'], [f'seed {i}' for i in range(1, 6)], 'Sheet E: seeds; rolling turf with a path and off-frame shade', 2)

SHEETS = dict(A=s_ground_hour, B=s_landform_light, C=s_distance, D=s_moisture, E=s_seeds)

def s_scenes():
    sc = [
        ('meadow bank, golden, sun left-behind', dict(ground='turf', landform='bank', hour='golden', sun=(25, 10), ppm=30, pitch=10, cam_h=1.7, shade=0.4, shade_scale=14, seed=5, plan='side-right')),
        ('rolling turf + path, noon, cloud/forest shade', dict(ground='turf', landform='rolling', path=True, hour='noon', ppm=22, pitch=11, cam_h=2.0, shade=0.4, shade_scale=16, seed=3)),
        ('forest floor, morning pools of light', dict(ground='forest', landform='rolling', hour='morning', ppm=40, pitch=18, cam_h=1.7, shade=0.72, shade_scale=4, canopy=0.2, canopy_scale=0.7, moisture=0.35, relief=0.5, seed=21, plan='side-left')),
        ('wet earth track, overcast', dict(ground='turf', landform='rolling', path=True, hour='overcast', ppm=36, pitch=16, cam_h=1.7, moisture=0.8, relief=0.7, seed=5)),
        ('gravel bar, golden, contre-jour', dict(ground='gravel', landform='flat', hour='golden', sun=(80, 8), ppm=30, pitch=14, cam_h=2.0, shade=0.4, shade_scale=10, seed=9, plan='foreground')),
        ('terraces at dusk (after V13)', dict(ground='turf', landform='rolling', path=True, hour='dusk', sun=(80, -2), ppm=10, pitch=7, cam_h=6.0, relief=1.3, seed=7)),
    ]
    rows = [[{**kw, 'W': 320, 'H': 200} for _, kw in sc[i:i + 3]] for i in (0, 3)]
    sheet('F_scenes', rows, ['', ''], [s[0] for s in sc[:3]], 'Sheet F: designed scenes (320x200 native, 2x). Row 2: ' + ' | '.join(s[0] for s in sc[3:]), 2)

SHEETS['F'] = s_scenes

if __name__ == '__main__':
    for n in (sys.argv[1:] or SHEETS):
        SHEETS[n]()
