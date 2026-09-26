"""figs.py -- the new figures for the driftwood tutorial pages.

Every figure is made by importing the study code in this folder UNCHANGED (a snapshot of
~/work/pixelart-studies/elements/driftwood/src/).  Where a figure shows an intermediate state
(before a rule was applied), it calls the same tech.* functions the renderer calls, in the same
order, and stops early.  Where a figure shows a rule switched off, the switch is named in the
caption and in the function below (usually one module constant or one render option).

    OMP_NUM_THREADS=1 nice -n 10 python3 figs.py            # all figures -> ../img/t_*.png
    OMP_NUM_THREADS=1 nice -n 10 python3 figs.py classes    # just one
"""
import os, sys
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(v, '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy import ndimage as ndi
import geo, tech, render, palette, pile, marks
from paint import Light, WOOD, GRAVEL
from util import board

OUT = os.path.join(HERE, '..', 'img')
FIGS = {}


def fig(f):
    FIGS[f.__name__] = f
    return f


def save(im, name):
    im.save(os.path.join(OUT, name))
    print('wrote', name)


def grey(cv):
    return render.to_rgb(cv, render.grey_ramps())


def one_log(scale=60, W=120, H=56, az=300, el=40, pitch=30, seed=3, yaw=12, L=2.0, r=0.2,
            ends=('sawn', 'worn'), look=(0, 0, 0.1), **kw):
    cam = geo.Camera(W, H, scale, pitch=pitch, look=look)
    lg = geo.make_log(L=L, r=r, yaw=yaw, seed=seed, ends=ends, **kw)
    li = Light(cam, az=az, el=el)
    return cam, lg, li


def steps_to_grey(st, wood, ramp=render.GREY[WOOD], ground=108):
    out = np.full(st.shape + (3,), ground, np.uint8)
    g = np.asarray(ramp, np.uint8)[np.clip(st, 0, 7)]
    out[wood] = np.stack([g] * 3, -1)[wood]
    return out


# ---------------------------------------------------------------- lesson 1: reading the trunk
@fig
def classes():
    """the class coordinate on a lying log: continuous v, then the five classes, then greys."""
    cam, lg, li = one_log(n_knots=0)
    buf = geo.trace(cam, [lg])
    wood = buf['obj'] >= 0
    d, phL = tech.normal_plane_angles(buf['N'], -cam.f, li.L, buf['T'])
    v = tech.cylinder_classes(d, phL)
    vim = np.full(v.shape + (3,), 40, np.uint8)
    vg = (v / 4 * 255).astype(np.uint8)
    vim[wood] = np.stack([vg] * 3, -1)[wood]
    k = np.clip(np.round(v), 0, 4).astype(int)
    hues = np.array([(90, 140, 220), (60, 40, 110), (220, 120, 60), (250, 230, 120), (190, 200, 90)], np.uint8)
    kim = np.full(v.shape + (3,), 40, np.uint8)
    kim[wood] = hues[k][wood]
    st, _ = tech.quantize_classes(tech.sharpen(v, 2 * buf['rad'] * cam.scale))
    gim = steps_to_grey(st, wood, ground=40)
    save(board([('continuous class coordinate v (0..4)', vim),
                ('classes: rim | core | halftone | light | lit edge', kim),
                ('classes -> wood steps 3 2 4 6 5, checker seams', gim)], scale=4, cols=1), 't_classes.png')


@fig
def vcurve():
    """v(d): the piecewise-linear class coordinate through Ferrari's fitted breaks."""
    from PIL import Image, ImageDraw
    from util import font
    W, H, pad = 720, 300, 50
    im = Image.new('RGB', (W, H), (24, 24, 28)); dr = ImageDraw.Draw(im); f = font(12)
    d = np.linspace(-150, 90, 600)
    v = tech.cylinder_classes(d, np.full_like(d, 30.0))
    X = lambda a: pad + (a + 150) / 240 * (W - 2 * pad)
    Y = lambda b: H - pad - b / 4 * (H - 2 * pad)
    cols = [(90, 140, 220), (110, 80, 170), (220, 120, 60), (250, 230, 120), (190, 200, 90)]
    for i in range(len(d) - 1):
        dr.line([(X(d[i]), Y(v[i])), (X(d[i + 1]), Y(v[i + 1]))], fill=cols[int(np.clip(round(v[i]), 0, 4))], width=3)
    for b in tech.CYL_BREAKS:
        dr.line([(X(b), pad - 10), (X(b), H - pad)], fill=(90, 90, 100))
        dr.text((X(b) - 14, pad - 26), f'{b:+.0f}', fill=(220, 220, 220), font=f)
    for k, name in enumerate(['rim', 'core', 'halftone', 'light', 'edge']):
        dr.text((W - pad + 4, Y(k) - 7), name, fill=cols[k], font=f)
    dr.line([(X(-150), H - pad), (X(90), H - pad)], fill=(200, 200, 200))
    for a in (-150, -90, 0, 90):
        dr.text((X(a) - 10, H - pad + 6), str(a), fill=(200, 200, 200), font=f)
    dr.text((pad, H - 20), 'd = angle round the axis from the light peak (deg), lit edge to the right; v = class coordinate',
            fill=(200, 200, 200), font=f)
    save(im, 't_vcurve.png')


# ---------------------------------------------------------------- lesson 2: laying it down
@fig
def ablation():
    """one lying log, the rules added one at a time (8x crop)."""
    cam, lg, li = one_log(W=96, H=46, scale=55, n_knots=0, look=(0.35, 0, 0.1))
    buf = geo.trace(cam, [lg])
    sh = geo.shadow(buf, li.L, [lg])
    obj = buf['obj']; wood = obj >= 0
    ndl = (buf['N'] * li.L).sum(-1)
    # a: a lighting formula -- N.L mapped linearly onto the 8 steps, ordered dither
    x = np.clip(1 + 6 * np.clip(ndl, 0, 1), 0, 7)
    th = np.tile(tech.CHECK3, (60, 80))[:x.shape[0], :x.shape[1]]
    a = steps_to_grey(np.clip(np.floor(x) + ((x - np.floor(x)) > th), 0, 7).astype(int), wood)
    # b: five classes, raw coordinate, checker everywhere (seams as wide as the classes)
    d, phL = tech.normal_plane_angles(buf['N'], -cam.f, li.L, buf['T'])
    v = tech.cylinder_classes(d, phL, sh=sh)
    b = steps_to_grey(tech.quantize_classes(v)[0], wood)
    # c: + sharpen (seams held to 1-2 px at any width)
    diam = 2 * buf['rad'] * cam.scale
    vs = tech.sharpen(v, diam)
    stc = tech.quantize_classes(vs)[0]
    c = steps_to_grey(stc, wood)
    # d: + no dither within 2 px of the outline, outline takes its run's majority class
    nbr = np.zeros_like(wood)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nbr |= np.roll(np.roll(obj, dy, 0), dx, 1) != obj
    sil = wood & nbr
    near = wood & ndi.binary_dilation(sil, iterations=1)
    kk = np.clip(np.round(vs), 0, 4).astype(int)
    std = np.where(near, tech.CLASS_STEPS[kk], stc)
    cnt = np.stack([ndi.uniform_filter(((kk == q) & sil).astype(float), 7) for q in range(5)])
    std = np.where(sil, tech.CLASS_STEPS[np.argmax(cnt, 0)], std)
    dd = steps_to_grey(std, wood)
    # e: the renderer with marks off (adds sun rim, cast-shadow family jump, end face, contact)
    cv, _, _ = render.render([lg], cam, li, opts=dict(gravel=False, marks=False))
    e = grey(cv)
    # f: the full renderer (pure classes, grain turns the form)
    cam2, lg2, li2 = one_log(W=96, H=46, scale=55, n_knots=0, look=(0.35, 0, 0.1))
    cv2, _, _ = render.render([lg2], cam2, li2, opts=dict(gravel=False))
    f = grey(cv2)
    P = [('a  N.L -> 8 greys (a lighting formula)', a), ('b  five classes, raw checker', b),
         ('c  + sharpen: seams 1-2 px', c), ('d  + clean outline (no dither near the edge)', dd),
         ('e  + sun rim, contact, end face (render, marks off)', e), ('f  + grain: pure classes, marks turn the form', f)]
    save(board(P, scale=5, cols=2), 't_ablation.png')


@fig
def ramp():
    """the same index canvas through the first (wide) grey ramp and the final (close) one."""
    cam, lg, li = one_log(W=120, H=50, scale=50, look=(0, 0, 0.1))
    cv, _, _ = render.render([lg], cam, li, opts=dict(gravel=False))
    wide = dict(render.grey_ramps()); wide[WOOD] = [(g, g, g) for g in (22, 48, 84, 108, 140, 172, 202, 232)]
    save(board([('same canvas, first ramp: 22 48 84 108 | 140 172 202 232', render.to_rgb(cv, wide)),
                ('same canvas, final ramp: 24 60 112 128 | 152 178 204 232', grey(cv))], scale=4, cols=1), 't_ramp.png')


@fig
def rim():
    """contre-jour: the sun-side 1-px run is all the light a back-lit log gets."""
    P = []
    for az, el, lab in ((180, 25, 'back 25'), (150, 25, 'back-right 25'), (220, 30, 'back-left 30')):
        cam, lg, li = one_log(W=110, H=50, scale=50, az=az, el=el, look=(0, 0, 0.1))
        cv, _, _ = render.render([lg], cam, li, opts=dict(gravel=False))
        P.append((lab, grey(cv)))
    save(board(P, scale=4, cols=1), 't_rim.png')


# ---------------------------------------------------------------- lesson 3: sitting on the ground
@fig
def contact():
    """contact line + belly pocket off / on; the pocket's clearance limit."""
    cam, lg, li = one_log(W=110, H=46, scale=55, sag=0.05, n_knots=0, look=(0, 0, 0.1), yaw=8, seed=5)
    off, _, _ = render.render([lg], cam, li, opts=dict(gravel=False, marks=False, contact=False))
    cam, lg, li = one_log(W=110, H=46, scale=55, sag=0.05, n_knots=0, look=(0, 0, 0.1), yaw=8, seed=5)
    on, _, _ = render.render([lg], cam, li, opts=dict(gravel=False, marks=False))
    # the first version: pocket = every ground pixel with the log anywhere above it
    old = geo.belly.__defaults__
    geo.belly.__defaults__ = (40.0,)
    cam, lg, li = one_log(W=110, H=46, scale=55, sag=0.05, n_knots=0, look=(0, 0, 0.1), yaw=8, seed=5)
    big, _, _ = render.render([lg], cam, li, opts=dict(gravel=False, marks=False))
    geo.belly.__defaults__ = old
    save(board([('no contact: it floats', grey(off)[8:44]),
                ('first pocket: any gap under the log (max_gap_px=40)', grey(big)[8:44]),
                ('final: contact line + pocket where clearance < 2.5 px', grey(on)[8:44])], scale=5, cols=1), 't_contact.png')


@fig
def castjump():
    """a log lying across another: the cast shadow as a class nudge vs a family jump."""
    a = geo.make_log(L=2.6, r=0.2, yaw=5, seed=2, ends=('worn', 'sawn'), n_knots=0)
    b = geo.make_log(L=2.2, r=0.13, yaw=70, seed=4, ends=('worn', 'snapped'), n_knots=0, centre=(0.1, 0.0))
    b = pile.settle(b, [a])
    cam = geo.Camera(110, 70, 50, pitch=32, look=(0, 0, 0.2))
    li = Light(cam, az=300, el=40)
    cv, buf, sh = render.render([a, b], cam, li, opts=dict(gravel=False, marks=False))
    # the class-nudge version: what the class rule alone does in cast shadow (v clipped to <= 1)
    d, phL = tech.normal_plane_angles(buf['N'], -cam.f, li.L, buf['T'])
    v = tech.cylinder_classes(d, phL, sh=sh)
    nudge = cv.step.copy()
    wood = buf['obj'] >= 0
    k = np.clip(np.round(v), 0, 4).astype(int)
    nudge[wood & sh] = tech.CLASS_STEPS[k][wood & sh]
    cv2 = render.Canvas(*cv.step.shape); cv2.mat = cv.mat.copy(); cv2.step = nudge
    save(board([('cast shadow = the class rule alone', grey(cv2)), ('cast shadow = a family jump (2, or 1 turned away)', grey(cv))],
               scale=4, cols=2), 't_castjump.png')


# ---------------------------------------------------------------- lesson 5: small logs
@fig
def small():
    P = []
    for n in (48, 24, 16, 12, 8, 5):
        sc = n / 3.2
        cam = geo.Camera(int(n * 1.15) + 8, max(12, int(n * 0.4) + 8), sc, pitch=30, look=(0, 0, 0.1))
        lg = geo.make_log(L=3.2, r=0.17, yaw=10, seed=3, ends=('sawn', 'snapped'))
        cv, _, _ = render.render([lg], cam, Light(cam, az=300, el=40), opts=dict(gravel=False))
        P.append((f'{n} px', grey(cv)))
    save(board(P, scale=8, cols=3), 't_small.png')


# ---------------------------------------------------------------- lesson 6: marks
@fig
def marksfig():
    cam, lg, li = one_log(W=120, H=52, scale=55, seed=7, look=(0, 0, 0.1))
    a, _, _ = render.render([lg], cam, li, opts=dict(gravel=False, marks=False))
    cam, lg, li = one_log(W=120, H=52, scale=55, seed=7, look=(0, 0, 0.1))
    b, _, _ = render.render([lg], cam, li, opts=dict(gravel=False))
    # density switched off: every dash drawn whatever class it sits on
    keep = marks.DENSITY.copy(); marks.DENSITY[:] = 1.0
    cam, lg, li = one_log(W=120, H=52, scale=55, seed=7, look=(0, 0, 0.1))
    c, _, _ = render.render([lg], cam, li, opts=dict(gravel=False))
    marks.DENSITY[:] = keep
    save(board([('marks off (pure classes)', grey(a)), ('grain, checks, knots (density by class)', grey(b)),
                ('marks.DENSITY = 1 everywhere: the light band fills up', grey(c))], scale=4, cols=1), 't_marks.png')


@fig
def knot():
    """the knot on the t_marks log, cropped at 8x."""
    cam, lg, li = one_log(W=120, H=52, scale=55, seed=7, look=(0, 0, 0.1))
    cv, _, _ = render.render([lg], cam, li, opts=dict(gravel=False))
    save(board([('the knot from t_marks, 8x', grey(cv)[6:34, 22:78])],
               scale=8), 't_knot.png')


@fig
def bark():
    cam, lg, li = one_log(W=120, H=50, scale=55, kind='bark', look=(0, 0, 0.1))
    a, _, _ = render.render([lg], cam, li, opts=dict(gravel=False))
    cam, lg, li = one_log(W=120, H=50, scale=55, look=(0, 0, 0.1))
    b, _, _ = render.render([lg], cam, li, opts=dict(gravel=False))
    save(board([('silvered: dark lines on a pale body', grey(b)), ('fresh bark: bright braided ridges on a dark body', grey(a))],
               scale=4, cols=1), 't_bark.png')


# ---------------------------------------------------------------- lesson 7: roots
@fig
def roots():
    P = []
    for style in ('mass', 'fan'):
        cam = geo.Camera(120, 80, 36, pitch=32, look=(0.0, 0, 0.3))
        lg = geo.make_log(L=3.0, r=0.2, yaw=140, seed=3, ends=('snapped', 'worn'),
                          rootwad=dict(n=6, end=0, style=style), n_stubs=2)
        cv, _, _ = render.render([lg], cam, Light(cam, az=300, el=40), seed=3, opts=dict(gravel=False))
        P.append(({'mass': 'style="mass": the dark tangled disc (a dead end)',
                   'fan': 'style="fan": the silvered root fan'}[style], grey(cv)))
    save(board(P, scale=4, cols=2), 't_roots.png')


@fig
def fanview():
    P = []
    for yaw, lab in ((15, 'butt seen side-on: the fan collapses to spikes'), (140, 'butt toward the viewer: it reads')):
        cam = geo.Camera(120, 80, 36, pitch=32, look=(0.0, 0, 0.3))
        lg = geo.make_log(L=3.0, r=0.2, yaw=yaw, seed=5, ends=('snapped', 'worn'), rootwad=dict(n=6, end=0), n_stubs=2)
        cv, _, _ = render.render([lg], cam, Light(cam, az=300, el=40), seed=5, opts=dict(gravel=False))
        P.append((lab, grey(cv)))
    save(board(P, scale=4, cols=2), 't_fanview.png')


# ---------------------------------------------------------------- lesson 8: palette
@fig
def onecanvas():
    cam = geo.Camera(120, 80, 36, pitch=32, look=(0.0, 0, 0.3))
    lg = geo.make_log(L=3.0, r=0.2, yaw=140, seed=3, ends=('snapped', 'worn'), rootwad=dict(n=6, end=0), n_stubs=2)
    cv, _, _ = render.render([lg], cam, Light(cam, az=300, el=40), seed=3)
    P = [('grey (the canvas)', grey(cv))]
    for h in ('noon', 'morning', 'golden', 'overcast'):
        P.append((h, render.to_rgb(cv, palette.ramps(h, 0.0))))
    P.append(('noon, wet ramp (same steps)', render.to_rgb(cv, dict(palette.ramps('noon', 1.0)))))
    save(board(P, scale=3, cols=3), 't_onecanvas.png')


@fig
def shadowhue():
    """the first sky tint versus the final one, on the same canvas."""
    cam, lg, li = one_log(W=120, H=50, scale=50, look=(0, 0, 0.1))
    cv, _, _ = render.render([lg], cam, li)
    now = palette.ramps('noon', 0.0)
    keep = dict(palette.HOURS['noon'])
    palette.HOURS['noon'] = dict(sun=(1.00, 0.97, 0.90), sky=(0.86, 0.92, 1.06), lift=0.0)   # the first noon
    first = palette.ramps('noon', 0.0)
    palette.HOURS['noon'] = keep
    save(board([('first noon sky tint (0.86, 0.92, 1.06)', render.to_rgb(cv, first)),
                ('final noon sky tint (0.98, 0.99, 1.01)', render.to_rgb(cv, now))], scale=4, cols=1), 't_shadowhue.png')


@fig
def wet():
    cam, lg, li = one_log(W=120, H=50, scale=50, look=(0, 0, 0.1), wet=1.0, az=60, el=30)
    cv, _, _ = render.render([lg], cam, li)
    cam, lg, li = one_log(W=120, H=50, scale=50, look=(0, 0, 0.1), wet=0.0, az=60, el=30)
    cv2, _, _ = render.render([lg], cam, li)
    save(board([('dry', render.to_rgb(cv2, palette.ramps('morning', 0.0))),
                ('wet: darker, warmer, glints near the half vector', render.to_rgb(cv, palette.ramps('morning', 1.0)))],
               scale=4, cols=1), 't_wet.png')


# ---------------------------------------------------------------- lesson 9: piles
@fig
def settlefig():
    keep = pile.settle
    pile.settle = lambda lg, placed, sink=0.015: lg          # every log simply on the gravel
    a = pile.make_pile(seed=2, n=10)
    pile.settle = keep
    b = pile.make_pile(seed=2, n=10)
    P = []
    for logs, lab in ((a, 'without settle: logs pass through each other'), (b, 'settle: each rests on the upper hull of its supports')):
        cam = geo.Camera(150, 90, 30, pitch=30, look=(0, 0, 0.3))
        cv, _, _ = render.render(logs, cam, Light(cam, az=300, el=45), seed=2, opts=dict(gravel=False))
        P.append((lab, grey(cv)))
    save(board(P, scale=3, cols=2), 't_settle.png')


@fig
def separate():
    # crops are shown at 2x the board scale by upscaling below
    logs = pile.make_pile(seed=5, n=10)
    cam = geo.Camera(150, 90, 30, pitch=30, look=(0, 0, 0.3))
    keep = render.separate_logs
    render.separate_logs = lambda *a, **k: None
    cv0, _, _ = render.render(logs, cam, Light(cam, az=300, el=45), seed=5, opts=dict(gravel=False))
    render.separate_logs = keep
    logs = pile.make_pile(seed=5, n=10)
    cv1, _, _ = render.render(logs, cam, Light(cam, az=300, el=45), seed=5, opts=dict(gravel=False))
    save(board([('separate_logs off', grey(cv0)), ('separate_logs on', grey(cv1)),
                ('off, 6x', grey(cv0)[28:72, 58:118], 6), ('on, 6x: cracks and occlusion steps', grey(cv1)[28:72, 58:118], 6)],
               scale=3, cols=2), 't_separate.png')




@fig
def cover():
    """the thumbnail: driftwood.paint with a root wad and a small pile, golden hour."""
    import driftwood
    from PIL import Image
    im = driftwood.paint(rootwad=True, pile=6, hour='golden', seed=23, length=3.5,
                         diameter=0.4, orientation=120, distance=40.0, W=160, H=100)
    Image.fromarray(im).resize((im.shape[1] * 3, im.shape[0] * 3), Image.NEAREST).save(os.path.join(OUT, 't_cover.png'))
    print('wrote t_cover.png')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    names = sys.argv[1:] or list(FIGS)
    for n in names:
        FIGS[n]()
