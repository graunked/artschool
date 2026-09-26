"""Stage 4: colour. The same index canvases re-lit by palette: midday, overcast, dusk."""
import numpy as np, pixkit as pk, world, painter, colour

Pg = painter.grey_palette()


def still(seed=1):
    cam = world.Cam(192, 108, f=300, h=1.6, pitch_deg=7.0)
    t = world.StraightChannel(7.0, 11.0, far=('slope', 0.2, 0.6), near=('slope', 0.12, 1.0), seed=seed + 4)
    G = world.raycast(cam, t)
    objs = [dict(kind='boulder', x=-2.3, z=11.9, w=1.5, h=0.62, y0=0.2, seed=11),
            dict(kind='willow', x=2.8, z=12.6, w=2.6, h=1.9, y0=0.2, seed=5)]
    bd = dict(ridge=lambda u: 2.2 + 1.3 * np.sin(u * 192 / 23.0) + 0.6 * np.sin(u * 192 / 7.3), scrub=(140.0, 3.0))
    return painter.paint(G, cam, Pg, np.random.default_rng(seed), dict(objects=objs, stones=(0.03, 0.14), backdrop=bd))


def riffle(seed=1):
    cam = world.Cam(160, 90, f=380, h=1.6, pitch_deg=9.0)
    t = world.StraightChannel(6.0, 9.5, far=('slope', 0.2, 0.6), near=('slope', 0.12, 1.0), seed=seed + 4)
    t.add_riffle(0.0, 1.2, 1.0, n_stones=34, seed=seed)
    G = world.raycast(cam, t)
    bd = dict(ridge=lambda u: 2.0 + 1.2 * np.sin(u * 9) + 0.5 * np.sin(u * 23), scrub=(120.0, 3.0))
    wil = [dict(kind='willow', x=1.2, z=10.4, w=2.6, h=1.8, y0=0.2, seed=3)]
    return painter.paint(G, cam, Pg, np.random.default_rng(seed), dict(terrain=t, backdrop=bd, stones=(0.03, 0.14), objects=wil))


if __name__ == '__main__':
    a, b = still(), riffle()
    rows = []
    for hour in ('midday', 'overcast', 'dusk'):
        P = colour.palette(Pg, hour)
        rows.append(pk.panel([(P.rgb(a), 'still strip, ' + hour), (P.rgb(b), 'riffle, ' + hour)], 3))
        pk.save(P.rgb(a)[40:100, 0:120], '../out/s4_still_%s_8x.png' % hour, 8)
    pk.stack(rows).save('../out/s4_colour.png')
    # palette cards
    cards = []
    for hour in ('midday', 'overcast', 'dusk'):
        P = colour.palette(Pg, hour)
        n = len(P.arr); card = np.zeros((12, n * 6, 3), np.uint8)
        for i in range(n):
            card[:, i * 6:(i + 1) * 6] = P.arr[i]
        cards.append((card, hour + ': ' + ' | '.join(P.ramps)))
    pk.stack([pk.panel([c], 4) for c in cards]).save('../out/s4_palettes.png')


def triptychs():
    a = still()
    R = '~/work/pixelart-studies/transfer/refs/'
    rows = []
    specs = [('midday', R + '../../elements/channel/photos/commons_Wilkin_River_close_to_its_confluence_with_Makaro_3b477161.jpg', (640, 290, 400, 190), 'V11AM', (330, 355, 96, 46)),
             ('overcast', R + 'braided_lowangle_overcast_nz.jpg', (430, 240, 420, 200), 'V14', (380, 380, 96, 46)),
             ('dusk', R + 'braided_lowangle_golden.jpg', (200, 250, 500, 240), 'V02', (0, 300, 96, 46))]
    for hour, ph, box, lw, fb in specs:
        P = colour.palette(Pg, hour)
        mine = P.rgb(a)[50:96, 0:96]
        photo = pk.photo_crop(ph, box, (96, 46))
        d = pk.lw_load(lw); fr = pk.rgb(d['idx'], d['pal'])
        x, y, w, h = fb
        rows.append(pk.panel([(photo, 'photo'), (mine, 'mine, ' + hour), (fr[y:y + h, x:x + w], 'Ferrari ' + lw)], 5))
    pk.stack(rows).save('../out/s4_triptych.png')


if __name__ == '__main__':
    triptychs()
