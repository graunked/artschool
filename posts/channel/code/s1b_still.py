"""Stage 1, rebuilt on the stage-2 machinery (world.raycast + painter): the still strip with a
boulder and a willow on the far bank. Same scene as s1_still.py, one technique."""
import sys
import numpy as np
import pixkit as pk
import world, painter

P = painter.grey_palette()
W, H = 192, 108


def render(seed=1, prm=None):
    cam = world.Cam(W, H, f=300, h=1.6, pitch_deg=7.0)
    t = world.StraightChannel(7.0, 11.0, far=('slope', 0.2, 0.6), near=('slope', 0.12, 1.0), seed=seed + 4)
    G = world.raycast(cam, t)
    objs = [dict(kind='boulder', x=-2.3, z=11.9, w=1.5, h=0.62, y0=0.2, seed=11),
            dict(kind='willow', x=2.8, z=12.6, w=2.6, h=1.9, y0=0.2, seed=5)]
    bd = dict(ridge=lambda u: 2.2 + 1.3 * np.sin(u * 192 / 23.0) + 0.6 * np.sin(u * 192 / 7.3), scrub=(140.0, 3.0))
    q = dict(objects=objs, stones=(0.03, 0.14), backdrop=bd)
    q.update(prm or {})
    cv = painter.paint(G, cam, P, np.random.default_rng(seed), q)
    return P.rgb(cv), G, cam


if __name__ == '__main__':
    rgb, G, cam = render()
    pk.save(rgb, '../out/s1_still.png', 4)
    pk.save(rgb[44:96, 0:96], '../out/s1_still_8x_left.png', 8)
    pk.save(rgb[0:96, 96:192], '../out/s1_still_8x_right.png', 8)
    grey = lambda a: np.stack([(a @ np.array([0.3, 0.59, 0.11])).astype(np.uint8)] * 3, -1)
    ph = grey(pk.photo_crop('~/work/pixelart-studies/transfer/refs/braided_bars_denali.jpg', (60, 400, 450, 230), (90, 46)).astype(float))
    d = pk.lw_load('V16PM'); fr = grey(pk.rgb(d['idx'], d['pal']).astype(float))[392:438, 250:340]
    pk.triptych(ph, rgb[50:96, 0:90], fr, '../out/s1_triptych.png', s=6,
                labels=('photo: Denali bar, grey', 'mine: stage 1', 'Ferrari: Mirror Pond water, grey'))
