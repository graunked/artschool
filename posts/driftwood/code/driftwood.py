"""driftwood.py -- the technique as one call, and the parameter sheets.

    paint(length=3.0, diameter=0.35, orientation=15, rootwad=False, weathering=1.0, wetness=0.0,
          light='left-front', hour='noon', camera='30', distance='near', seed=0, pile=0,
          ground='gravel', W=160, H=96) -> RGB uint8 array (native pixels)

  length, diameter   metres
  orientation        compass yaw of the log in degrees (0 = across the view, 90 = pointing away)
  rootwad            a root wad at the butt end
  weathering         0..1: grain gaps, checks and knots (0 = smooth, 1 = checked and knotted)
  wetness            0..1: >= 0.5 switches the log to the wet ramp with specular glints
  light              'left-front' | 'right-front' | 'back' | 'high' (sun azimuth/elevation)
  hour               'noon' | 'morning' | 'golden' | 'overcast' (the palette; overcast ignores light)
  camera             '30' (30 deg oblique) | 'low' (10 deg)
  distance           'near' (40 px/m) | 'mid' (16) | 'far' (6) | a number (px/m)
  pile               0 = one log; n > 0 = a pile of n logs around it (seed-driven)

    python3 driftwood.py sheets        # all sheets into ../sheets (4 worker processes)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import geo, render, palette, pile as pilemod
from paint import Light
from util import board

ROOT = os.path.expanduser('~/work/pixelart-studies/elements/driftwood')
LIGHTS = {'left-front': (300, 40), 'right-front': (60, 40), 'back': (180, 22), 'high': (330, 68)}
HOUR_EL = {'golden': 10, 'morning': 22}
DIST = {'near': 40.0, 'mid': 16.0, 'far': 6.0}
PITCH = {'30': 30.0, 'low': 10.0}


def paint(length=3.0, diameter=0.35, orientation=15.0, rootwad=False, weathering=1.0, wetness=0.0,
          light='left-front', hour='noon', camera='30', distance='near', seed=0, pile=0,
          ground='gravel', W=160, H=96, grey=False):
    scale = DIST.get(distance, distance) if isinstance(distance, str) else float(distance)
    pitch = PITCH[camera]
    r = diameter / 2
    az, el = LIGHTS[light]
    if hour in HOUR_EL:
        el = min(el, HOUR_EL[hour] if light != 'high' else HOUR_EL[hour] * 2.2)
    rng = np.random.default_rng(seed)
    rw = dict(n=6, end=0) if rootwad else None
    lg = geo.make_log(L=length, r=r, yaw=orientation, seed=seed, rootwad=rw,
                      ends=('snapped' if rootwad else str(rng.choice(['sawn', 'worn', 'worn', 'snapped'])),
                            str(rng.choice(['snapped', 'worn', 'splinter']))),
                      n_stubs=int(round(3 * weathering)), wet=1.0 if wetness >= 0.5 else 0.0,
                      n_knots=int(round(3 * weathering)))
    logs = [lg]
    if pile:
        extra = pilemod.make_pile(seed=seed + 1, n=pile, extent=(length * 1.6, length * 0.8),
                                  flow_yaw=orientation, r_range=(r * 0.4, r * 0.9))
        placed = [lg]
        for q in extra:
            placed.append(pilemod.settle(q, placed))
        logs = placed
    cz = float(np.mean([q.pts[:, 2].mean() for q in logs]))
    cx = float(np.mean(lg.pts[:, 0])); cy = float(np.mean(lg.pts[:, 1]))
    cam = geo.Camera(W, H, scale, pitch=pitch, look=(cx, cy, cz))
    li = Light(cam, az=az, el=el, overcast=(hour == 'overcast'))
    cv, buf, sh = render.render(logs, cam, li, seed=seed, stone_m=max(0.06, 1.6 / scale),
                                opts=dict(gravel=(ground == 'gravel'), weather=weathering))
    if grey:
        return render.to_rgb(cv, render.grey_ramps())
    return render.to_rgb(cv, palette.ramps(hour, 1.0 if wetness >= 0.5 else 0.0))


# ------------------------------------------------------------------ sheets
def _cell(args):
    label, kw = args
    os.nice(10) if hasattr(os, 'nice') else None
    t = time.time()
    img = paint(**kw)
    return label, img, time.time() - t


def sheet(name, cells, cols, title, scale=3):
    from multiprocessing import Pool
    with Pool(4) as p:
        res = p.map(_cell, cells)
    panels = [(lab, img) for lab, img, _ in res]
    board(panels, scale=scale, cols=cols, title=title).save(f'{ROOT}/sheets/{name}_{scale}x.png')
    board(panels, scale=1, cols=cols, title=None, fsz=9, label_h=12).save(f'{ROOT}/sheets/{name}_1x.png')
    print(name, 'cells', len(cells), 'mean s/cell', round(np.mean([r[2] for r in res]), 1))


SHEETS = {}


def _def(name):
    def deco(f):
        SHEETS[name] = f
        return f
    return deco


@_def('length_diameter')
def s_len_diam():
    cells = []
    for L in (1.5, 3.0, 5.0):
        for d in (0.15, 0.3, 0.55):
            cells.append((f'L {L} m, d {d} m', dict(length=L, diameter=d, seed=11, distance=28.0)))
    sheet('sheet_length_diameter', cells, 3, 'length (rows) x diameter (cols); 28 px/m, 30 deg, noon, left-front')


@_def('orientation_camera')
def s_orient_cam():
    cells = []
    for cam in ('30', 'low'):
        for o in (0, 30, 60, 90):
            cells.append((f'camera {cam}, yaw {o}', dict(orientation=o, camera=cam, seed=12, distance=30.0,
                                                          H=80 if cam == 'low' else 96)))
    sheet('sheet_orientation_camera', cells, 4, 'orientation (cols) x camera 30 deg / low 10 deg (rows)')


@_def('rootwad_weather_wet')
def s_rw():
    cells = []
    for rw in (False, True):
        for w, wt in ((0.2, 0.0), (1.0, 0.0), (0.2, 1.0), (1.0, 1.0)):
            cells.append((f'{"root wad" if rw else "no wad"}, weather {w}, {"wet" if wt else "dry"}',
                          dict(rootwad=rw, weathering=w, wetness=wt, seed=13, distance=30.0, diameter=0.4)))
    sheet('sheet_rootwad_weathering_wetness', cells, 4, 'root wad (rows) x weathering and wetness (cols)')


@_def('light_hour')
def s_light_hour():
    cells = []
    for hour in ('noon', 'morning', 'golden', 'overcast'):
        for lt in ('left-front', 'right-front', 'back', 'high'):
            cells.append((f'{hour}, {lt}', dict(light=lt, hour=hour, seed=14, distance=30.0, rootwad=True, diameter=0.4)))
    sheet('sheet_light_hour', cells, 4, 'hour (rows) x sun direction (cols)')


@_def('distance_seed')
def s_dist_seed():
    cells = []
    for dist in ('near', 'mid', 'far'):
        for sd in (21, 22, 23, 24):
            cells.append((f'{dist}, seed {sd}', dict(distance=dist, seed=sd, rootwad=(sd % 2 == 0), pile=6,
                                                     length=3.5, diameter=0.4)))
    sheet('sheet_distance_seed', cells, 4, 'distance (rows: 40, 16, 6 px/m) x seed (cols), with a 6-log pile')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'sheets':
        names = sys.argv[2:] or list(SHEETS)
        for n in names:
            SHEETS[n]()
    else:
        from PIL import Image
        img = paint()
        Image.fromarray(img).resize((img.shape[1] * 4, img.shape[0] * 4), Image.NEAREST).save(f'{ROOT}/scratch/paint_default.png')
