"""figs.py -- makes every new figure on the channel pages from the study code snapshot beside it.

    cd ~/work/artschool/posts/channel/code && PYTHONDONTWRITEBYTECODE=1 nice -n 10 python3 figs.py [name ...]

Nothing here paints: each figure calls the study's own functions (world.raycast, painter.paint,
colour.palette, channel.to_ramp / to_ramp_checker, the stage scripts' render functions), sometimes
with one of their parameters changed, and lays the results out. Output: ../img/fig_*.png
"""
import sys, os
import numpy as np
from PIL import Image, ImageDraw
import pixkit as pk
import channel as ch
import world, painter, colour

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'img')
PH = '~/work/pixelart-studies/transfer/refs/'
PH2 = '~/work/pixelart-studies/elements/channel/photos/'
Pg = painter.grey_palette()


def out(name):
    return os.path.join(OUT, name)


# ------------------------------------------------------------------ the mirror as a material ---
def fig_quantizers():
    """one gradient, two materials: the sky quantizer (flat bands, short seams) and the water
    quantizer (the checker is the material)."""
    W, H = 96, 14
    V = np.tile(np.linspace(172, 238, W), (H, 1))
    sky_i, sky_v = Pg.r('sky'), painter.vals(Pg, 'sky')
    a = Pg.rgb(ch.to_ramp(V, sky_v, sky_i, 0.5))
    b = Pg.rgb(ch.to_ramp_checker(V, sky_v, sky_i, 0.3))
    pk.stack([pk.panel([(a, 'to_ramp: the sky (bands, seams)')], 6),
              pk.panel([(b, 'to_ramp_checker: the water (the checker is the material)')], 6)]).save(out('fig_quantizers.png'))


def fig_mirror_ablation():
    """the stage-1 strip, with one of the mirror's rules switched off at a time (painter parameters)."""
    import s1b_still as S
    cases = [({}, 'all rules on'), (dict(nearness=0.0), 'nearness = 0: no climb toward the viewer'),
             (dict(sheen=0.0), 'sheen = 0: no bright band under the bank'),
             (dict(clouds=()), 'clouds = (): an empty sky above the frame'),
             (dict(streaks=0.0, wind=0.0), 'no streaks, no wind lines')]
    items = []
    for prm, lab in cases:
        rgb, G, cam = S.render(1, prm)
        items.append((rgb[56:90, 0:100], lab))
    pk.stack([pk.panel(items[:1], 6)] + [pk.panel(items[i:i + 2], 6) for i in (1, 3)]).save(out('fig_mirror_ablation.png'))


# ------------------------------------------------------------------ what the mirror sees ------
def fig_gbuffer():
    """what each pixel is, and what each water pixel's reflected ray hits, for a sloping and a cut
    bank (the stage-2 cells), beside the painting."""
    import s2_edges as S
    rows = []
    for name in ('a_far_slope', 'b_far_cut'):
        rgb, G, cam = S.render(name)
        k = G['kind']; r = G['refl']
        m = np.zeros(k.shape + (3,), np.uint8)
        m[k == 0] = (200, 220, 245)                     # sky
        m[k == 1] = (150, 130, 100)                     # ground
        m[(k == 2) & (r == 0)] = (90, 150, 230)          # water whose ray reaches the sky
        m[(k == 2) & (r == 1)] = (200, 60, 60)           # water whose ray hits the bank
        rows.append(pk.panel([(m, name + ': kind / reflected-ray hit'), (rgb, 'painted')], 4))
    pk.stack(rows).save(out('fig_gbuffer.png'))


def fig_column():
    """one screen column of the cut-bank cell, in section: the view ray to a water pixel, and its
    reflected ray climbing (in the same vertical plane) until it hits the bank."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import s2_edges as S
    pitch, tp, prm = S.CELLS['b_far_cut']
    t = world.StraightChannel(tp['z'][0], tp['z'][1], far=tp['far'], near=tp['near'], seed=3)
    cam = world.Cam(S.W, S.H, f=420, h=1.6, pitch_deg=pitch)
    G = world.raycast(cam, t)
    c = S.W // 2
    zs = np.linspace(0.3, 9.5, 1200)
    y = t.height(cam.x_of(c + 0.5, zs), zs)
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=110)
    ax.fill_between(zs, np.minimum(y, 0), -0.6, color='#8a7a66')
    ax.fill_between(zs, np.maximum(y, 0), -0.6, where=y > 0, color='#8a7a66')
    ax.fill_between(zs, 0, np.minimum(y, 0), where=y < 0, color='#9cc4e8', alpha=0.8)
    ax.plot([0.0], [cam.h], 'ko')
    rows = [r for r in range(S.H) if G['kind'][r, c] == 2]
    for r, col in ((rows[1], '#c03030'), (rows[len(rows) // 2], '#c03030'), (rows[-2], '#2060c0')):
        z0 = G['z'][r, c]
        ax.plot([0, z0], [cam.h, 0], color=col, lw=0.8)
        if G['refl'][r, c] == 1:
            ax.plot([z0, G['refl_z'][r, c]], [0, G['refl_y'][r, c]], color=col, lw=1.6)
        else:
            ax.plot([z0, z0 + 3], [0, 3 * np.tan(G['graze'][r, c])], color=col, lw=1.6, ls='--')
    ax.set_xlim(0, 9.5); ax.set_ylim(-0.6, 1.8); ax.set_xlabel('distance z (m)'); ax.set_ylabel('height (m)')
    ax.set_title('one column: red rays hit the cut bank (dark reflection), blue reaches the sky', fontsize=9)
    fig.tight_layout(); fig.savefig(out('fig_column.png')); plt.close(fig)


# ------------------------------------------------------------------ the edges -----------------
def fig_edge_ablation():
    import s2_edges as S
    pitch, tp, _ = S.CELLS['c_near_gravel']
    cases = [({}, 'all edge rules on'), (dict(wet_drop=0.0), 'wet_drop = 0: no wet foot'),
             (dict(lap=0.0), 'lap = 0: no lap line'), (dict(shelf=(0.0, 0.0)), 'shelf = 0: no see-in shelf')]
    items = []
    for prm, lab in cases:
        t = world.StraightChannel(tp['z'][0], tp['z'][1], far=tp['far'], near=tp['near'], seed=3)
        rgb, G, cam = S.cell('c', pitch, t, prm, 1)
        items.append((rgb[0:64, 0:96], lab))
    pk.stack([pk.panel(items[:2], 5), pk.panel(items[2:], 5)]).save(out('fig_edge_ablation.png'))


def fig_edge_triptych():
    import s2_edges as S
    a, _, _ = S.render('a_far_slope'); b, _, _ = S.render('b_far_cut')
    d = pk.lw_load('V02'); fr = pk.rgb(d['idx'], d['pal'])
    ph = pk.photo_crop(PH + 'braided_bars_denali.jpg', (60, 380, 400, 250), (128, 80))
    pk.panel([(ph, 'photo: Denali cut bank'), (a[:, :], 'sloping bank'), (b, 'cut bank'),
              (fr[284:364, 0:128], 'Ferrari V02: dark far edges, light near lips')], 3).save(out('fig_edge_triptych.png'))


# ------------------------------------------------------------------ colour --------------------
def fig_milky():
    import s4_colour as S
    cam = world.Cam(192, 108, f=300, h=1.6, pitch_deg=7.0)
    t = world.StraightChannel(7.0, 11.0, far=('slope', 0.2, 0.6), near=('slope', 0.12, 1.0), seed=5)
    G = world.raycast(cam, t)
    objs = [dict(kind='boulder', x=-2.3, z=11.9, w=1.5, h=0.62, y0=0.2, seed=11),
            dict(kind='willow', x=2.8, z=12.6, w=2.6, h=1.9, y0=0.2, seed=5)]
    bd = dict(ridge=lambda u: 2.2 + 1.3 * np.sin(u * 192 / 23.0) + 0.6 * np.sin(u * 192 / 7.3), scrub=(140.0, 3.0))
    P = colour.palette(Pg, 'midday')
    items = []
    for milky, lab in ((False, 'midday, the mirror on the sky ramp itself'), (True, 'midday, the milky mirror (wmir)')):
        cv = painter.paint(G, cam, Pg, np.random.default_rng(1), dict(objects=objs, stones=(0.03, 0.14), backdrop=bd, milky=milky))
        items.append((P.rgb(cv)[40:100, 0:192], lab))
    ph = pk.photo_crop(PH2 + 'commons_Wilkin_River_close_to_its_confluence_with_Makaro_3b477161.jpg', (640, 290, 640, 200), (192, 60))
    items.append((ph, 'photo: Wilkin River, glacial water'))
    pk.stack([pk.panel([it], 3) for it in items]).save(out('fig_milky.png'))


def fig_land_key():
    import s5_braid as S
    rgb, cv = S.render_designed('low', 'dusk')
    save = colour.HOURS['dusk'].pop('land_key')
    flat = colour.palette(Pg, 'dusk').rgb(cv)
    colour.HOURS['dusk']['land_key'] = save
    pk.panel([(flat, 'dusk, land keyed like the sky (0.78)'), (rgb, 'dusk, land_key 0.55: the ribbon glows')], 3).save(out('fig_land_key.png'))
    pk.save(rgb, out('fig_low_dusk_4x.png'), 4)


def fig_relight():
    import s5_braid as S
    rgb, cv = S.render_designed('low', 'dusk')
    items = [(pk.zoom(np.stack([Pg.arr[cv][..., 0]] * 3, -1), 1), 'the index canvas, grey')]
    for h in ('midday', 'overcast', 'dusk'):
        items.append((colour.palette(Pg, h).rgb(cv), h))
    pk.stack([pk.panel(items[:2], 2), pk.panel(items[2:], 2)]).save(out('fig_relight.png'))


# ------------------------------------------------------------------ distance ------------------
def fig_threads():
    """the braid plain at a low angle: sub-pixel threads off, then on; meander LOD off, then on."""
    import s5_braid as S
    W, H = 256, 144
    cam = world.Cam(W, H, f=320, h=1.7, pitch_deg=4.0)
    bd = dict(ridge=lambda u: 2.4 + 1.4 * np.sin(u * 7 + 12) + 0.6 * np.sin(u * 19), scrub=(700.0, 6.0))
    kw = dict(n_threads=16, width=(1.0, 6.0), spread=90.0, lam=(10.0, 50.0), angle_sd=22.0)
    P = colour.palette(Pg, 'midday')
    items = []
    for lod, tf, lab in ((0.06, 0.0, 'thread_frac = 0: one ray per pixel only'),
                         (0.0, 0.05, 'lod = 0: far meanders unaveraged'),
                         (0.06, 0.05, 'both on (as delivered)')):
        t = world.BraidPlain(seed=12, lod=lod, **kw)
        G = world.raycast(cam, t, z_min=2.0, z_max=900.0, thread_frac=tf)
        cv = painter.paint(G, cam, Pg, np.random.default_rng(12), dict(backdrop=bd, stones=(0.03, 0.14)))
        items.append((P.rgb(cv)[26:66, 0:192], lab))
    pk.stack([pk.panel([it], 4) for it in items]).save(out('fig_threads.png'))


def fig_ladder():
    import s5_braid as S
    items = [(S.render_designed('low', 'dusk', z0=z0, length=60 * max(1, z0 / 30), eye=4.0)[0][20:90, :],
              'eye 4 m, channel from %d m' % z0) for z0 in (8, 40, 160)]
    pk.stack([pk.panel([it], 3) for it in items]).save(out('fig_ladder.png'))


# ------------------------------------------------------------------ riffle --------------------
def fig_riffle():
    import s3_flow as S
    wil = dict(objects=[dict(kind='willow', x=1.2, z=10.4, w=2.6, h=1.8, y0=0.2, seed=3)])
    a = S.render(0.0, extra=wil); b = S.render(1.0, extra=wil)
    ph = pk.photo_crop(PH + 'braided_lowangle_golden.jpg', (0, 480, 700, 250), (160, 57))
    pk.stack([pk.panel([(a, 'still: the willow mirrored whole'), (b, 'flow 1.0: the same reflection breaks')], 3),
              pk.panel([(ph, 'photo: golden-hour sheets (streaks run with the flow)')], 3)]).save(out('fig_riffle.png'))
    pk.save(b[40:90, 20:140], out('fig_riffle_8x.png'), 6)


FIGS = {k[4:]: v for k, v in globals().items() if k.startswith('fig_')}

if __name__ == '__main__':
    names = sys.argv[1:] or list(FIGS)
    for n in names:
        print(n, flush=True)
        FIGS[n]()
