"""Stage 3: moving water. A channel crossing the view (flow left to right) with a riffle over a sill."""
import sys
import numpy as np
import pixkit as pk
import world, painter

P = painter.grey_palette()


def render(strength=1.0, pitch=9.0, zn=6.0, zf=9.5, f=380, W=160, H=90, seed=1, extra=None):
    cam = world.Cam(W, H, f=f, h=1.6, pitch_deg=pitch)
    t = world.StraightChannel(zn, zf, far=('slope', 0.2, 0.6), near=('slope', 0.12, 1.0), seed=seed + 4)
    if strength > 0:
        t.add_riffle(0.0, 1.2, strength, n_stones=int(30 * strength) + 4, seed=seed)
    G = world.raycast(cam, t)
    bd = dict(ridge=lambda u: 2.0 + 1.2 * np.sin(u * 9) + 0.5 * np.sin(u * 23), scrub=(120.0, 3.0))
    q = dict(terrain=t, backdrop=bd, stones=(0.03, 0.14))
    q.update(extra or {})
    cv = painter.paint(G, cam, P, np.random.default_rng(seed), q)
    return P.rgb(cv)


if __name__ == '__main__':
    rgbs = [(render(s), 'flow %.1f' % s) for s in (0.0, 0.5, 1.0)]
    pk.stack([pk.panel(rgbs, 4)]).save('../study/s3_v1.png')
