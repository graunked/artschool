"""technique.py -- the braided-channel painter as one function with parameters.

    render(dict(width=3.5, depth=0.4, flow=0.0, hour='overcast', bank='gravel', camera='low',
                distance=7.0, seed=1, milky=True, bend=0.0, W=160, H=90)) -> (rgb, index canvas)

width    channel width across (m)            depth   max depth (m)
flow     0 still .. 1 riffle (sill strength)  hour    midday | overcast | dusk
bank     gravel | cut | grass | tussock_near  camera  low (1.6 m, ~7 deg down) | oblique (30 deg, 12 m up)
distance near edge distance (m)               seed    layout only
milky    glacial water (True) or clear (False)  bend  lateral wander of the strip (m)
"""
import numpy as np
import world, painter, colour

Pg = painter.grey_palette()
BANKS = {
    'gravel': dict(far=('slope', 0.2, 0.6), near=('slope', 0.12, 1.0), grass=None),
    'cut': dict(far=('cut', 0.35), near=('slope', 0.12, 1.0), grass=None),
    'grass': dict(far=('cut', 0.22), near=('slope', 0.12, 1.0),
                  grass=dict(side='far', min_y=0.08, blade_m=0.35, density=0.4, lean=0.3, lit=0.4)),
    'tussock_near': dict(far=('slope', 0.2, 0.6), near=('cut', 0.3),
                         grass=dict(side='near', min_y=0.15, blade_m=0.4, density=0.4, lean=-0.3, lit=0.4)),
}


def camera_for(p):
    W, H = p.get('W', 160), p.get('H', 90)
    zmid = p['distance'] + p['width'] / 2
    if p['camera'] == 'low':
        h = 1.6
        # frame the strip: put its centre about 60% down the frame
        ang = np.degrees(np.arctan(h / zmid))
        f = 300 * (zmid / 9.0) ** 0.5
        pitch = ang + np.degrees(np.arctan((0.05 * H) / f))
        return world.Cam(W, H, f=f, h=h, pitch_deg=pitch)
    # 30 degrees down from a terrace/bluff: the strip's centre at the frame centre, the strip about a
    # third of the frame tall
    h = zmid * np.tan(np.radians(30.0))
    dist = np.hypot(h, zmid)
    f = 0.33 * H * dist / (p['width'] * np.sin(np.radians(30.0))) * 0.9
    f = float(np.clip(f, 60, 900))
    return world.Cam(W, H, f=f, h=h, pitch_deg=30.0)


def render(p):
    p = dict(dict(width=3.5, depth=0.4, flow=0.0, hour='overcast', bank='gravel', camera='low', distance=7.0,
                  seed=1, milky=True, bend=0.0), **p)
    b = BANKS[p['bank']]
    zn = p['distance']; zf = zn + p['width']
    t = world.StraightChannel(zn, zf, depth=p['depth'], far=b['far'], near=b['near'], seed=p['seed'] + 4,
                              bend=p['bend'], bend_len=6.0 + zn * 0.3)
    if p['flow'] > 0:
        t.add_riffle(-0.45 * p['width'] - 0.05 * zn * p['width'], 0.8 + 0.1 * zn, p['flow'], n_stones=int(10 + 30 * p['flow']), seed=p['seed'])
    cam = camera_for(p)
    G = world.raycast(cam, t, z_min=1.0, z_max=900.0)
    bd = dict(ridge=lambda u: 2.0 + 0.9 * np.sin(u * 3.1 + p['seed']) + 0.5 * np.sin(u * 7.3 + 2 * p['seed']) + 0.25 * np.sin(u * 17 + p['seed']),
              scrub=(max(80.0, zf * 8), 3.0))
    q = dict(terrain=t, backdrop=bd, stones=(0.03, 0.14), grass=b['grass'], milky=p['milky'])
    if not p['milky']:
        q.update(visibility=0.6)
    cv = painter.paint(G, cam, Pg, np.random.default_rng(p['seed']), q)
    P = colour.palette(Pg, p['hour'])
    return P.rgb(cv), cv
