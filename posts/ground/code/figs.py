"""Figures for the ground tutorials. Every picture is painted by the real study code
(ground.py, a snapshot of elements/ground/src/ground.py); toggles are the painter's own
parameters, or (where a step has no parameter) a monkeypatch that switches one step off.
    cd code && nice -n 10 python3 figs.py [name ...]"""
import os, sys, numpy as np
from PIL import Image, ImageDraw
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); IMG = os.path.join(HERE, '../img')
sys.path.insert(0, HERE)
import ground, lw

def up(a, k): return np.repeat(np.repeat(np.asarray(a, np.uint8), k, 0), k, 1)

def board(cells, labels, cols, k=2, gap=6, title=None, lab_h=16):
    """cells: list of HxWx3 uint8; labels: list of str; returns a PIL image"""
    H, W = cells[0].shape[:2]
    rows = (len(cells) + cols - 1) // cols
    top = 18 if title else 0
    S = Image.new('RGB', (cols * (W * k + gap) - gap, top + rows * (H * k + lab_h + gap)), (24, 24, 28))
    d = ImageDraw.Draw(S)
    if title: d.text((4, 3), title, fill=(235, 235, 235))
    for i, (c, l) in enumerate(zip(cells, labels)):
        x = (i % cols) * (W * k + gap); y = top + (i // cols) * (H * k + lab_h + gap)
        d.text((x + 2, y + 2), l, fill=(220, 220, 200))
        S.paste(Image.fromarray(up(c, k)), (x, y + lab_h))
    return S

def notan(g):
    """two-value notan of the painted ground: light family white, shadow family black"""
    a = np.where(g.step >= 3, 235, 25).astype(np.uint8)
    a[g.mat < 0] = 150
    return np.stack([a] * 3, -1)

# ---------------------------------------------------------------- the meadow bank, step by step
BANK = dict(ground='turf', landform='bank', hour='golden', sun=(25, 10), ppm=30, pitch=10, cam_h=1.7,
            shade=0.4, shade_scale=14, seed=5, plan='side-right')

def _masses_cell(args):
    step, W, H = args
    import ground as G
    kw = dict(BANK)
    if step == 0: kw.update(shade=0.0, plan=None)
    if step == 1: G.design_masses = lambda f, min_frac=0.02: f; kw.update(plan=None)
    if step == 2: kw.update(plan=None)
    g = G.paint_ground(W, H, **kw)
    return g.rgb, notan(g)

def fig_masses():
    W, H = 240, 150
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(_masses_cell, [(s, W, H) for s in range(4)]))
    labs = ['1 land + sun only', '2 + off-frame shade (raw)', '3 + delete small islands', '4 + plan: shade the right']
    cells = [r[0] for r in res] + [r[1] for r in res]
    board(cells, labs + [l.split(' ', 1)[0] + ' notan' for l in labs], 4, k=2).save(f'{IMG}/t_masses_steps.png')

def _toggle_cell(kw):
    kw = dict(kw); W = kw.pop('W'); H = kw.pop('H')
    return ground.paint_ground(W, H, **kw).rgb

def _nomarks_cell(kw):
    import ground as G
    kw = dict(kw); W = kw.pop('W'); H = kw.pop('H')
    def none(buf, mat, v, lit, rng, contrast=1.0, phi=None):
        h, w = v.shape
        return np.zeros((h, w)), np.full((h, w), 0.5), np.zeros((h, w)), np.zeros((h, w), bool), np.zeros((h, w), bool)
    G.marks = none
    return G.paint_ground(W, H, **kw).rgb

def fig_marks_off():
    """the same painting with the marks switched off: the masses carry it alone.
    contrast=0 is not enough: it scales the offsets, but the step-skipping gaps and the
    negative strokes are set directly, so marks() itself is switched off here."""
    W, H = 240, 150
    with ProcessPoolExecutor(3) as ex:
        a = ex.submit(_nomarks_cell, dict(BANK, W=W, H=H, blades=0)).result()
        b = ex.submit(_toggle_cell, dict(BANK, W=W, H=H, contrast=0.0, blades=0)).result()
        c = ex.submit(_toggle_cell, dict(BANK, W=W, H=H)).result()
    board([a, b, c], ['no marks: masses + dither only', 'contrast=0: gaps and strokes only', 'defaults: all marks + hero blades'], 3, k=2).save(f'{IMG}/t_marks_off.png')

def fig_turn():
    """crest band / convexity / mass turn / family keying off vs on, three scenes"""
    scenes = [
        ('hummocks, shadow family', dict(ground='turf', landform='hummocks', hour='noon', sun=(80, -2), ppm=8, pitch=5, cam_h=3.0, relief=1.5, seed=4)),
        ('high meadow, golden', dict(ground='turf', landform='rolling', hour='golden', sun=(-35, 16), ppm=18, pitch=38, cam_h=9, relief=1.8, shade=0.3, shade_scale=5, seed=6, blades=0)),
        ('gravel bar, contre-jour', dict(ground='gravel', landform='flat', hour='golden', sun=(80, 8), ppm=30, pitch=14, cam_h=2.0, shade=0.4, shade_scale=10, seed=9, plan='foreground')),
    ]
    W, H = 240, 150
    jobs = []
    for _, kw in scenes:
        jobs.append(dict(kw, W=W, H=H, crest_band=0, mass_turn=0, lit_key=0))
        jobs.append(dict(kw, W=W, H=H))
    with ProcessPoolExecutor(4) as ex:
        ims = list(ex.map(_toggle_cell, jobs))
    labs = []
    for n, _ in scenes: labs += [n + ': flat masses', n + ': turned + keyed']
    board(ims, labs, 2, k=2).save(f'{IMG}/t_turn_off_on.png')

FIGS = dict(masses=fig_masses, marks_off=fig_marks_off, turn=fig_turn)


# ---------------------------------------------------------------- a bug found while writing this page
def clean_mask(lit, min_px=30):
    """fill specks of either family smaller than min_px (the terminator noise leaves 1-2 px
    'shadow' pixels inside lit ground; the distance transform then sees an edge there)"""
    from scipy import ndimage as ndi
    lit = lit.copy()
    for val in (False, True):
        lab, n = ndi.label(lit == val)
        if n == 0: continue
        s = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        lit[np.isin(lab, np.nonzero(s < min_px)[0] + 1)] = not val
    return lit

def _fixed_cell(kw):
    import ground as G
    orig = G.turn_masses
    G.turn_masses = lambda v, lit, buf, cb=1.0, mt=1.0, cap=None: orig(v, clean_mask(lit), buf, cb, mt, cap)
    kw = dict(kw); W = kw.pop('W'); H = kw.pop('H')
    return G.paint_ground(W, H, **kw).rgb

GRAVEL = dict(ground='gravel', landform='flat', hour='golden', sun=(80, 8), ppm=30, pitch=14, cam_h=2.0,
              shade=0.4, shade_scale=10, seed=9, plan='foreground', W=240, H=150)

def fig_blobs():
    with ProcessPoolExecutor(3) as ex:
        a = ex.submit(_toggle_cell, dict(GRAVEL, mass_turn=0)).result()
        b = ex.submit(_toggle_cell, dict(GRAVEL)).result()
        c = ex.submit(_fixed_cell, dict(GRAVEL)).result()
    board([a, b, c], ['mass_turn=0', 'mass_turn=1 (as shipped): stamped rings', 'mass_turn=1 on a cleaned mask'], 3, k=2).save(f'{IMG}/t_blob_bug.png')

FIGS['blobs'] = fig_blobs


# ---------------------------------------------------------------- reading masses in photos
PHOTOS = os.path.expanduser('~/work/pixelart-studies/elements/ground/photos')

def game_res(path, w=320):
    """box-downsample a photo to game resolution (the steppe student's method)"""
    im = Image.open(path).convert('RGB')
    h = int(round(im.height * w / im.width))
    return np.asarray(im.resize((w, h), Image.BOX))

def mass_map(rgb, sigma=3, n=4):
    """squint (gaussian on L*) then cut at the image's own quartiles: 4 flat value masses"""
    from analyze import squint
    L = squint(rgb, sigma)
    q = np.digitize(L, np.percentile(L, np.linspace(0, 100, n + 1)[1:-1]))
    g = np.array([30, 90, 160, 230][:n])[q].astype(np.uint8)
    return np.stack([g] * 3, -1)

PHOTO_SET = ['alpine_meadow_rolling_hills_late_afterno/commons_Uncompahgre_Wilderness_9503370126_jpg_b1479d97.jpg',
             'grassy_hillside_low_sun_long_shadows/commons_Below_Kinder_Low_geograph_org_uk_1713540_jpg_bae68a64.jpg',
             'forest_floor_sunlight_patches/commons_Sunlit_roots_in_the_woods_Unsplash_jpg_35da6da1.jpg',
             'gravel_river_bar/commons_Nowitna_river_gravel_bar_jpg_de5fb1b4.jpg']

def fig_photo_masses(names=None):
    names = names or PHOTO_SET
    cells, labs = [], []
    for n in names:
        a = game_res(os.path.join(PHOTOS, n))
        a = a[:180] if a.shape[0] >= 180 else np.pad(a, ((0, 180 - a.shape[0]), (0, 0), (0, 0)))
        cells += [a, mass_map(a)]; labs += [os.path.basename(n)[:40], 'squint, 4 masses']
    board(cells, labs, 2, k=2).save(f'{IMG}/t_photo_masses.png')

def _scene(kw):
    kw = dict(kw); W = kw.pop('W', 320); H = kw.pop('H', 200)
    return ground.paint_ground(W, H, **kw).rgb

F_BANK = dict(ground='turf', landform='bank', hour='golden', sun=(25, 10), ppm=30, pitch=10, cam_h=1.7, shade=0.4, shade_scale=14, seed=5, plan='side-right')
F_NOON = dict(ground='turf', landform='rolling', path=True, hour='noon', ppm=22, pitch=11, cam_h=2.0, shade=0.4, shade_scale=16, seed=3)

def fig_mass_compare():
    """the same squint-and-quartile map on Ferrari's ground and on mine"""
    v16 = lw.rgb(lw.load('V16'))[260:440, 0:320]
    v16pm = lw.rgb(lw.load('V16PM'))[260:440, 320:640]
    with ProcessPoolExecutor(2) as ex:
        a, b = list(ex.map(_scene, [F_BANK, F_NOON]))
    cells, labs = [], []
    for im, n in [(v16, 'Ferrari V16, left bank + islet'), (v16pm, 'Ferrari V16PM, right bank'),
                  (a[20:200], 'mine: meadow bank, golden'), (b[20:200], 'mine: rolling turf, noon')]:
        cells += [im, mass_map(im)]; labs += [n, 'squint, 4 masses']
    board(cells, labs, 4, k=2).save(f'{IMG}/t_mass_compare.png')

FIGS['mass_compare'] = fig_mass_compare
FIGS['photo_masses'] = fig_photo_masses


def fig_plans():
    plans = [None, 'side-left', 'side-right', 'foreground', 'far', 'pool', 'V']
    base = dict(F_BANK, W=200, H=125)
    jobs = [dict(base, plan=p) for p in plans] + [dict(base, plan='V', seed=8)]
    with ProcessPoolExecutor(4) as ex:
        ims = list(ex.map(_toggle_cell, jobs))
    board(ims, [f'plan={p!r}' for p in plans] + ["plan='V', seed 8"], 4, k=2).save(f'{IMG}/t_plans.png')

FIGS['plans'] = fig_plans


def _crop_cell(args):
    kw, box = args
    kw = dict(kw); W = kw.pop('W', 240); H = kw.pop('H', 150)
    x0, y0, x1, y1 = box
    return ground.paint_ground(W, H, **kw).rgb[y0:y1, x0:x1]

def fig_texture_close():
    PATH = dict(ground='turf', landform='rolling', path=True, ppm=40, pitch=18, cam_h=1.7, relief=0.6, seed=11, hour='noon', shade=0.3)
    FOREST = dict(ground='forest', landform='rolling', hour='morning', ppm=40, pitch=18, cam_h=1.7, shade=0.72, shade_scale=4,
                  canopy=0.2, canopy_scale=0.7, moisture=0.35, relief=0.5, seed=21, plan='side-left')
    jobs = [(dict(BANK), (20, 90, 120, 150)), (dict(PATH), (120, 60, 220, 120)),
            (dict(GRAVEL), (100, 80, 200, 140)), (dict(FOREST), (130, 80, 230, 140))]
    with ProcessPoolExecutor(4) as ex:
        ims = list(ex.map(_crop_cell, jobs))
    v16pm = lw.rgb(lw.load('V16PM'))[380:440, 0:100]
    v16 = lw.rgb(lw.load('V16'))[240:300, 0:100]
    v09 = lw.rgb(lw.load('V09'))[410:470, 20:120]
    board([v16pm, ims[0], v16, ims[1], v09, ims[3], ims[2], np.zeros_like(ims[2]) + 24],
          ['Ferrari V16PM near lit bank', 'mine: turf, golden (lit + shadow)', 'Ferrari V16 near bank, noon',
           'mine: turf + path, noon', 'Ferrari V09 moss path', 'mine: forest floor', 'mine: gravel, contre-jour', ''],
          2, k=5).save(f'{IMG}/t_texture_close.png')

FIGS['texture_close'] = fig_texture_close


def near_fade_patch(G):
    """proposed fix: fade the whole turn (crest band, convexity, mass turn) out where grass blades are big (>= ~6 px); near grass
    turns by stroke density, far ground by a value ramp. Also cleans the lit mask (t_blob_bug)."""
    orig = G.turn_masses
    def tm(v, lit, buf, cb=1.0, mt=1.0, cap=None):
        lit_c = clean_mask(lit)
        on = orig(v, lit_c, buf, cb, mt, cap)
        off = np.where(lit_c, np.clip(v, 3.0, 5.99), np.clip(v, 0, 2.9))   # no turn at all
        w = np.clip(1 - (0.2 * buf['ppm'] - 2.0) / 4.0, 0, 1)             # 1 below 2 px blades, 0 above 6
        return w * on + (1 - w) * off
    G.turn_masses = tm

def _patched_cell(kw):
    import ground as G
    near_fade_patch(G)
    kw = dict(kw); W = kw.pop('W', 240); H = kw.pop('H', 150)
    g = G.paint_ground(W, H, **kw)
    return g.rgb, g.step, g.lit

def fig_near_fade():
    with ProcessPoolExecutor(2) as ex:
        a = ex.submit(_toggle_cell, dict(BANK, W=240, H=150)).result()
        b, st, lt = ex.submit(_patched_cell, dict(BANK)).result()
    s = st[90:150, 20:120]; l = lt[90:150, 20:120]
    print('patched lit shares', np.round(np.bincount(s[l], minlength=6) / l.sum(), 2))
    v16pm = lw.rgb(lw.load('V16PM'))[380:440, 0:100]
    board([a, b], ['as shipped', 'turn faded out near + cleaned mask'], 2, k=2).save(f'{IMG}/t_near_fade.png')
    board([v16pm, a[90:150, 20:120], b[90:150, 20:120]], ['Ferrari V16PM', 'as shipped', 'near fade'], 3, k=5).save(f'{IMG}/t_near_fade_8x.png')

FIGS['near_fade'] = fig_near_fade

if __name__ == '__main__':
    for n in (sys.argv[1:] or FIGS): FIGS[n](); print('wrote', n)
