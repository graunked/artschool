"""Integration: the driftwood student's logs standing on this ground."""
import sys, os, numpy as np
sys.path.insert(0, os.path.expanduser('~/work/pixelart-studies/elements/driftwood/src'))
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import ground, lw
import driftwood as dw, geo, render as dwr, palette as dwp, pile as pilemod
from paint import Light, GRAVEL

def logs_on_ground(W=240, H=150, hour='golden', ground_kw=None, seed=24, scale=40.0, pitch=30.0, pile=4):
    rng = np.random.default_rng(seed)
    lg = geo.make_log(L=3.0, r=0.17, yaw=15.0, seed=seed, rootwad=dict(n=6, end=0), ends=('snapped', 'worn'),
                      n_stubs=3, wet=0.0, n_knots=3)
    logs = [lg]; placed = [lg]
    for q in pilemod.make_pile(seed=seed + 1, n=pile, extent=(4.8, 2.4), flow_yaw=15.0, r_range=(0.07, 0.15)):
        placed.append(pilemod.settle(q, placed))
    logs = placed
    cz = float(np.mean([q.pts[:, 2].mean() for q in logs])); cx = float(np.mean(lg.pts[:, 0])); cy = float(np.mean(lg.pts[:, 1]))
    cam = geo.Camera(W, H, scale, pitch=pitch, look=(cx, cy, cz))
    az, el = {'golden': (300, 10), 'noon': (300, 40), 'overcast': (300, 40)}[hour]
    li = Light(cam, az=az, el=el, overcast=(hour == 'overcast'))
    cv, buf, sh = dwr.render(logs, cam, li, seed=seed, stone_m=0.06, opts=dict(gravel=True, weather=1.0))
    wood_rgb = dwr.to_rgb(cv, dwp.ramps(hour, 0.0))
    wood = cv.mat != GRAVEL
    # contact: darkest ground pixels the log student drew right under the wood
    near = np.zeros_like(wood)
    near[:-1] |= wood[1:]; near[1:] |= wood[:-1]
    contact = (~wood) & (cv.step == 0) & near
    kw = dict(ground='gravel', landform='flat', ppm=scale, pitch=pitch, cam_h=4.0, hour=hour, seed=seed,
              sun=(20 if az > 180 else 160, el))
    kw.update(ground_kw or {})
    g = ground.paint_ground(W, H, shadow_mask=sh & ~wood, **kw)
    out = g.rgb.copy()
    out[wood | contact] = wood_rgb[wood | contact]
    return out, g.rgb, wood_rgb

if __name__ == '__main__':
    rows = []
    for hr in ('golden', 'noon'):
        a, gr, old = logs_on_ground(hour=hr, ground_kw=dict(shade=0.35, shade_scale=3, moisture=0.3))
        b, _, _ = logs_on_ground(hour=hr, ground_kw=dict(ground='turf', shade=0.3, shade_scale=3))
        p = np.zeros((a.shape[0], 4, 3), np.uint8)
        rows.append(np.concatenate([old, p, a, p, b], 1))
    q = np.zeros((4, rows[0].shape[1], 3), np.uint8)
    lw.save(np.concatenate([rows[0], q, rows[1]], 0), os.path.join(here, '../sheets/integration_driftwood_3x.png'), 3)
