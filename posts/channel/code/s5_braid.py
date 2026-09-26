"""Stage 5: the channel as one form -- a braid of ribbons that bend, narrow, split and rejoin,
seen at a low angle and at 30 degrees oblique."""
import sys, time
import numpy as np, pixkit as pk, world, painter, colour

Pg = painter.grey_palette()


def render(view='low', seed=3, hour='overcast', W=256, H=144, **kw):
    kw = dict(dict(n_threads=16, width=(1.0, 6.0), spread=90.0, lam=(10.0, 50.0), angle_sd=22.0), **kw)
    if view == 'low':
        cam = world.Cam(W, H, f=320, h=1.7, pitch_deg=4.0)
    else:
        cam = world.Cam(W, H, f=260, h=30.0, pitch_deg=30.0)
    kw.setdefault('lod', 0.06 * (1.7 * 320) / (cam.h * cam.f))   # meanders under a row's footprint average out
    t = world.BraidPlain(seed=seed, **kw)
    t0 = time.time()
    G = world.raycast(cam, t, z_min=2.0, z_max=900.0)
    bd = dict(ridge=lambda u: 2.4 + 1.4 * np.sin(u * 7 + seed) + 0.6 * np.sin(u * 19), scrub=(700.0, 6.0))
    q = dict(backdrop=bd, stones=(0.03, 0.14))
    cv = painter.paint(G, cam, Pg, np.random.default_rng(seed), q)
    P = colour.palette(Pg, hour)
    return P.rgb(cv), cv, time.time() - t0


if __name__ == '__main__':
    rgb, cv, dt = render(sys.argv[1] if len(sys.argv) > 1 else 'low')
    print('%.1fs' % dt)
    pk.save(rgb, '../study/s5_%s.png' % (sys.argv[1] if len(sys.argv) > 1 else 'low'), 4)


def render_designed(view='low', hour='dusk', W=256, H=144, z0=8.0, length=60.0, seed=2, eye=1.7, lateral=None):
    if view == 'low':
        cam = world.Cam(W, H, f=320, h=eye, pitch_deg=4.0 if eye < 3 else 7.0)
    else:
        cam = world.Cam(W, H, f=260, h=30.0, pitch_deg=30.0)
    t = world.designed_channel(z0=z0, length=length, lateral=lateral if lateral else 6.0 * max(1.0, z0 / 20.0) ** 0.8)
    G = world.raycast(cam, t, z_min=2.0, z_max=900.0)
    bd = dict(ridge=lambda u: 2.4 + 1.4 * np.sin(u * 7 + seed) + 0.6 * np.sin(u * 19), scrub=(700.0, 6.0))
    cv = painter.paint(G, cam, Pg, np.random.default_rng(seed), dict(backdrop=bd, stones=(0.03, 0.14)))
    return colour.palette(Pg, hour).rgb(cv), cv
