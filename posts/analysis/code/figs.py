"""Make every tutorial figure from boulder.py. Run from this folder: python figs.py"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from boulder import paint, notan_shares, RGB, W, H

OUT = '../img/'
try:
    FONT = ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', 20)
except Exception:
    FONT = ImageFont.load_default()


def up(a, s=3): return np.repeat(np.repeat(a, s, 0), s, 1)


def labelled(a, text):
    im = Image.fromarray(a.astype(np.uint8)); c = Image.new('RGB', (im.width, im.height + 30), (250, 248, 244))
    c.paste(im, (0, 30)); ImageDraw.Draw(c).text((6, 4), text, fill=(20, 20, 20), font=FONT); return np.array(c)


def row(tiles, gap=10):
    h = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, h - t.shape[0]), (0, gap), (0, 0)), constant_values=250) for t in tiles]
    return np.concatenate(tiles, 1)[:, :-gap]


def save(a, name): Image.fromarray(a.astype(np.uint8)).save(OUT + name)


def grey(L): return np.stack([np.clip(L * 2.55, 0, 255)] * 3, -1).astype(np.uint8)


def notan_img(rgb):
    _, q = notan_shares(rgb); lv = np.array([25, 130, 240], np.uint8); return np.stack([lv[q]] * 3, -1)


PLANE_COL = np.array([(214, 170, 196), (150, 110, 160), (70, 60, 50), (120, 150, 210), (220, 170, 80),
                      (180, 180, 180), (30, 30, 30), (120, 110, 50), (10, 10, 10)], np.uint8)

if __name__ == '__main__':
    base = dict(size='medium', light='left', distance='10m', seed=1)
    rgb, lay = paint(**base)
    save(up(rgb), 't_final.png')
    # 1 planes
    save(up(PLANE_COL[lay['P']]), 't1_planes.png')
    # 2 value plan + notan of the plan, vs the same without the tall darks
    _, lay_ng = paint(**base, traps=('no_grove', 'toy_forest'))
    rgb_ng, _ = paint(**base, traps=('no_grove', 'toy_forest'))
    s_good = notan_shares(rgb)[0]; s_bad = notan_shares(rgb_ng)[0]
    save(row([labelled(up(notan_img(rgb_ng), 2), f'far trees at toy size, no grove: {s_bad[0]:.0%} dark'),
              labelled(up(notan_img(rgb), 2), f'trees at true size, cropped: {s_good[0]:.0%} dark')]), 't2_notan_trap.png')
    save(up(grey(lay['value_plan'])), 't2_value_plan.png')
    # 3 the boulder's light plan (grey), and the lumpy-terminator trap
    for dist in ('close',):
        _, a = paint(**dict(base, distance=dist), steps=3)
        _, b = paint(**dict(base, distance=dist), steps=3, traps=('lumpy_terminator',))
        crop = (slice(20, 150), slice(20, 230))
        save(row([labelled(up(grey(b['value_full'])[crop], 2), 'terminator from the lumpy surface'),
                  labelled(up(grey(a['value_full'])[crop], 2), 'terminator from the big form')]), 't3_terminator_trap.png')
    # 4-5 colour + clusters (no strokes yet), and colour traps
    c45, lay45 = paint(**dict(base, distance='close'), steps=5)
    save(up(c45), 't5_colour_clusters.png')
    cn, _ = paint(**dict(base, distance='close'), steps=5, traps=('nearest_colour',))
    crop = (slice(20, 150), slice(20, 230))
    save(row([labelled(up(cn[crop], 2), 'nearest of all 32 colours'),
              labelled(up(c45[crop], 2), 'warm ramp lit, cool ramp shade')]), 't4_ramp_trap.png')
    cs, _ = paint(**dict(base, distance='close'), steps=5, traps=('short_strokes',))
    crop = (slice(125, 180), slice(0, 90))
    save(row([labelled(up(cs[crop], 5), 'weak vertical coherence'),
              labelled(up(c45[crop], 5), 'strokes: sigma_y 4.5')]), 't5_stroke_trap.png')
    # 6 strokes
    c6, _ = paint(**dict(base, distance='close'), steps=6)
    crop = (slice(40, 150), slice(0, 180))
    save(row([labelled(up(c45[crop], 3), 'before strokes'), labelled(up(c6[crop], 3), 'blades, sedge, cracks')]), 't6_strokes.png')
    # 7 pixel geometry trap: boxy silhouette
    cb, _ = paint(**dict(base, distance='close'), steps=6, traps=('boxy',))
    crop = (slice(20, 150), slice(20, 230))
    save(row([labelled(up(cb[crop], 2), 'over-snapped: 3 slopes, tol 2.5'),
              labelled(up(c6[crop], 2), 'clean runs: 10 slopes, tol 0.9')]), 't7_boxy_trap.png')
    # 8 reflection: before / after, and the checker trap
    c7, _ = paint(**base, steps=7)
    save(row([labelled(up(c7, 2), 'before the reflection'), labelled(up(rgb, 2), 'the water mirrors it')]), 't8_reflection.png')
    rc, _ = paint(**base, traps=('checker_water',))
    crop = (slice(70, 140), slice(60, 220))
    save(row([labelled(up(rc[crop], 4), 'checker everywhere'), labelled(up(rgb[crop], 4), 'horizontal ripples')]), 't8_checker_trap.png')
    # the camera
    lvl, _ = paint(**base, pitch_deg=3.0)
    s_l = notan_shares(lvl)[0]
    save(row([labelled(up(rgb, 2), f'10 deg down (task): {s_good[0]:.0%} dark'),
              labelled(up(lvl, 2), f'3 deg down: {s_l[0]:.0%} dark')]), 't9_camera.png')
    a_, _ = paint(**base, traps=('toy_forest',)); b_, _ = paint(**base, traps=('toy_forest',), pitch_deg=3.0)
    sa, sb = notan_shares(a_)[0], notan_shares(b_)[0]
    save(row([labelled(up(a_, 2), f'toy trees, 10 deg down: {sa[0]:.0%} dark'),
              labelled(up(b_, 2), f'toy trees, 3 deg down: {sb[0]:.0%} dark')]), 't9_camera_toytrees.png')
    # sheets
    def sheet(name, cells, cols):
        tiles = []
        for p, t in cells:
            a, _ = paint(**p); tiles.append(labelled(up(a, 2), t))
        rows_ = [row(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        w = max(r.shape[1] for r in rows_)
        rows_ = [np.pad(r, ((0, 10), (0, w - r.shape[1]), (0, 0)), constant_values=250) for r in rows_]
        save(np.concatenate(rows_, 0), name)
    sheet('t_sheet_light.png', [(dict(base, light=l), f'light={l}') for l in ('left', 'front', 'right', 'back')], 2)
    sheet('t_sheet_size_dist.png', [(dict(base, size=s_, distance=d_), f'{s_} / {d_}')
                                     for d_ in ('close', '10m', '40m') for s_ in ('small', 'large')], 2)
    sheet('t_sheet_seed.png', [(dict(base, seed=k), f'seed={k}') for k in (1, 2, 3, 4)], 2)
    print('notan final', s_good, 'no grove', s_bad, 'level', s_l)
