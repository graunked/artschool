"""Figures for the people tutorial, made by re-running the study's own code.

The study lives in ~/work/pixelart-studies/elements/people/src/ and is imported unchanged; this
script only calls it with different arguments and writes into ./fig/. Run from this folder:
    PYTHONDONTWRITEBYTECODE=1 nice -n 10 python3 figs.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.expanduser('~/work/pixelart-studies/elements/people/src')
sys.dont_write_bytecode = True
sys.path.insert(0, SRC)
os.chdir(SRC)                      # stage scripts resolve ../refs and ../stages from here (read only)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from render import Scene, render_scene, shade, light_vec, sprite_ss
from person import build
from pal import palette, grey_lut, M
from util import up, save, side, vstack, grid
from scenes import SCENES
FIG = os.path.join(HERE, 'fig')
os.makedirs(FIG, exist_ok=True)
BG = np.array([112, 120, 108], np.uint8)


def out(name, img):
    save(img, os.path.join(FIG, name))
    print('wrote', name, img.shape[:2])


def cell(spec, size, kind='SAND', hour='noon', sun=(40, 55), w=1.7, h=1.35, rules_px=None, figs_same=False,
         edges=True, k=6):
    """One person on a ground at `size` px. rules_px: compute the size-dependent rules (head boost,
    pixel floors, glyph collapse) as if the figure were rules_px tall, but draw it at `size`."""
    W, H = int(size * w), int(size * h)
    sc = Scene(W, H, size, 10, ground_y=H - max(2, size // 7))
    if rules_px is None:
        sc.add_person(spec)
        rgb, _ = render_scene(sc, kind, hour, sun, seed=spec.get('seed', 0), edges=edges)
    else:
        prims, J, meta = build(spec, rules_px)
        for p in prims:
            sc.cv.draw(p, sc.ox, sc.oy, sc.scale, 0)
        sc.figs.append(J); sc.meta.append(dict(meta, spec=spec))
        rgb, _ = render_scene(sc, kind, hour, sun, seed=spec.get('seed', 0), size_px=rules_px, edges=edges)
    return up(rgb, k)


def fig_rules_ablation():
    """The small-size rules on and off: the same figure drawn at 8 and 6 px with the rules computed
    for a 24 px figure (no head boost, no floors, no glyph collapse) and for its real size."""
    specs = [('walking', dict(activity='walk', phase=0.1, facing=15, seed=4)),
             ('carrying', dict(activity='carry', phase=0.3, facing=15, seed=4)),
             ('blanket', dict(activity='stand', facing=100, sex='m', dress='coast_m_blanket', seed=4)),
             ('digging', dict(activity='dig', phase=0.5, facing=20, seed=4)),
             ('spear', dict(activity='spear', phase=0.5, facing=10, sex='m', seed=4))]
    rows = []
    for size in (8, 6):
        rows.append(side([cell(sp, size, rules_px=24, k=12) for _, sp in specs],
                         [f'{n}, {size} px, 24 px rules' for n, _ in specs], fs=11))
        rows.append(side([cell(sp, size, k=12) for _, sp in specs],
                         [f'{n}, {size} px, own rules' for n, _ in specs], fs=11))
    out('rules_ablation.png', vstack(rows))


def fig_supersample():
    """Direct rendering with floors vs rendering 4x and reducing by salience-weighted mode."""
    lut = palette('noon')
    specs = [dict(activity='stand', facing=0, seed=1), dict(activity='walk', phase=0.1, facing=0, seed=1),
             dict(activity='stand', facing=90, seed=1), dict(activity='carry', phase=0.2, facing=0, seed=1),
             dict(activity='dig', facing=0, seed=1)]
    rows = []
    for size in (8, 6):
        for mode in ('direct', 'ss'):
            cells = []
            for sp in specs:
                if mode == 'direct':
                    W, H = int(size * 1.6), int(size * 1.35)
                    sc = Scene(W, H, size, 10, ground_y=H - 2)
                    sc.add_person(dict(sp))
                    shade(sc.cv, light_vec(40, 40), size, seed=1)
                    idx = sc.cv.idx
                else:
                    idx, _ = sprite_ss(dict(sp), size, pad=3, thresh=0.3)
                img = np.where(idx[..., None] > 0, lut[idx], BG)
                cells.append(up(img, 14))
            rows.append(side(cells, [f'{size} px, {"drawn at size" if mode == "direct" else "4x then reduced"}']
                             + [''] * (len(cells) - 1), fs=11))
    out('supersample_vs_direct.png', vstack(rows))


def fig_seam_between_people():
    """The dark seam where one person overlaps another, on and off, in a tight file at 8 and 6 px."""
    rows = []
    for size in (8, 6):
        panels = []
        for same in (True, False):
            sc = Scene(int(size * 4.6), int(size * 1.7), size, 10, ground_y=int(size * 1.45))
            for i, (x, z, sp) in enumerate([(-1.5, 0.3, dict(activity='carry', sex='f')),
                                            (-1.05, -0.3, dict(activity='walk', sex='f')),
                                            (-0.55, 0.35, dict(activity='carry_load', sex='m')),
                                            (-0.1, -0.25, dict(activity='walk', sex='f')),
                                            (0.35, 0.3, dict(activity='dig', sex='f')),
                                            (0.8, -0.3, dict(activity='carry', sex='f'))]):
                sc.add_person(dict(sp, x=x, z=z, facing=0, phase=(0.1 + i * 0.29) % 1, seed=i + 3))
            if same:
                sc.cv.fig[sc.cv.fig >= 0] = 0          # all one "figure": no seam between people
            rgb, _ = render_scene(sc, 'GRAVEL', 'morning', (20, 22), seed=3)
            panels.append(up(rgb, 10))
        rows.append(side(panels, [f'{size} px, no seam between people', f'{size} px, seam on (the study)'], fs=12))
    out('seam_between_people.png', vstack(rows))


def fig_edge_rule():
    """The edge rule on and off on the calm sand, for bare summer figures at 12 and 8 px."""
    specs = [dict(activity='walk', phase=0.1, facing=15, seed=2), dict(activity='stand', facing=40, sex='m', seed=5),
             dict(activity='dig', phase=0.3, facing=20, seed=2), dict(activity='run', age='child', phase=0.2, seed=2)]
    rows = []
    for size in (12, 8):
        for e in (False, True):
            rows.append(side([cell(sp, size, edges=e, k=10) for sp in specs],
                             [f'{size} px, edge rule {"on" if e else "off"}'] + [''] * 3, fs=11))
    out('edge_rule.png', vstack(rows))


def fig_prop_lines():
    """Two-handed tools: the hands are put on the tool's line by IK and the tool drawn through them."""
    acts = [('dig', 'f'), ('spear', 'm'), ('dipnet', 'm'), ('pound', 'f'), ('split', 'm')]
    rows = []
    for a, sx in acts:
        rows.append(side([cell(dict(activity=a, sex=sx, phase=ph, facing=15, seed=3), 32, w=1.9, h=1.3, k=4)
                          for ph in (0.0, 0.25, 0.5, 0.75)], [f'{a} {ph}' for ph in (0.0, 0.25, 0.5, 0.75)], fs=11))
    out('prop_lines.png', vstack(rows))


def fig_ladder_hero():
    """One household member of each kind, 32 -> 6 px, at 6x and at 1x."""
    specs = [dict(activity='carry', phase=0.3, facing=15, seed=4),
             dict(activity='walk', phase=0.6, facing=160, dress='coast_f_rain', seed=4),
             dict(activity='stand', facing=100, sex='m', dress='coast_m_blanket', seed=4),
             dict(activity='staff', facing=10, age='elder', sex='m', dress='coast_m_fur', seed=4, phase=0.3)]
    sizes = (32, 24, 16, 12, 8, 6)
    rows = []
    for sp in specs:
        rows.append(side([cell(sp, s, k=max(3, 96 // s)) for s in sizes], [f'{s} px' for s in sizes], fs=11))
    out('ladder_household.png', vstack(rows))


def fig_scenes():
    for name, size, k, kw in (('canoe', 32, 3, {}), ('carry', 24, 4, dict(hour='golden', sun=(170, 14))),
                              ('fire', 32, 3, {}), ('weir', 24, 3, {}), ('work', 24, 3, {}),
                              ('camas', 24, 3, {}), ('carry', 12, 6, dict(hour='morning'))):
        rgb, _ = SCENES[name](size, **kw)
        out(f'scene_{name}_{size}.png', up(rgb, k))
    rgb, _ = SCENES['carry'](32, hour='morning', sun=(20, 22))
    out('cover.png', up(rgb, 3))


def fig_night_steps():
    """Firelight on one seated woman: sun off, fire on, the ground pool."""
    from setting import hearth, paint_fire
    rows = []
    for hour, fire in (('night', False), ('night', True)):
        sc = Scene(80, 44, 24, 14, ground_y=38)
        sc.add_prims(hearth(0.35, 0.05))
        sc.add_person(dict(activity='sit_log', x=-0.55, z=0.05, facing=5, sex='f', dress='coast_f_wool', seed=1))
        sc.add_person(dict(activity='fire', x=1.1, z=-0.2, facing=170, sex='m', seed=2))
        fires = [((0.35, 0.25, 0.05), 0.95)] if fire else None

        def post(idx, sc):
            p = sc.screen((0.35, 0, 0.05))
            return paint_fire(idx, p[0], p[1] - 0.5, 24, 0.4, zbuf=sc.cv.z, zfire=p[2])
        rgb, _ = render_scene(sc, 'DIRT', hour, (40, 40), fires=fires, post=post)
        rows.append(up(rgb, 6))
    out('firelight_on_off.png', side(rows, ['night, no firelight on people or ground', 'night, firelight (the study)']))


if __name__ == '__main__':
    which = sys.argv[1:]
    for f in [fig_rules_ablation, fig_supersample, fig_seam_between_people, fig_edge_rule, fig_prop_lines,
              fig_ladder_hero, fig_scenes, fig_night_steps]:
        if not which or f.__name__ in which:
            f()
