"""Figures for the willow tutorial. Every picture here is made by the study's own functions
(copied verbatim beside this file from ~/work/pixelart-studies/elements/willow on 2026-09-26);
this script only calls them with switches turned off, so you can see one layer at a time.

    cd code && nice -n 10 python3 figs.py        # writes ../img/t_*.png
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy import ndimage as ndi
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "..", "img")

from wpaint import *                      # Canvas, sun, upscale, save, label_grid, side_by_side, GREY6, materials
from willow import paint_willow, lod_params, paint_bush, merge, LEAFS
from palette import willow_palette
from foliage import clump_foliage, mid_speckle, willow_spray, fringe, wand_tips, poisson
from context import gravel, cast_shadow, scene_palette
from lw import load
from copyharness import decompose, sketch, render


def bg(w, h, gy):
    cv = Canvas(w, h)
    cv.mat[:gy] = SKY; cv.step[:gy] = 2
    cv.mat[gy:] = GRAVEL; cv.step[gy:] = 3
    return cv


def fig_construction():
    """One bush, built up layer by layer. Same seed in every panel: each switch only removes work
    that happens *after* the layers shown, so the earlier layers are pixel-identical."""
    W, H, GY = 200, 140, 128
    L = sun(25, 40)
    base = lod_params(150)
    steps = [
        ("marks on the clump field", dict(tips=None, fringe=None, airy=None, wands_visible=None, front_sprays=None)),
        ("+ the edge (tips, caps, holes)", dict(airy=None, wands_visible=None, front_sprays=None)),
        ("+ airy holes, wands, front sprays", dict()),
    ]
    pal = willow_palette("summer", "noon")
    cells, labs = [], []
    first = None
    for name, off in steps:
        cv = bg(W, H, GY)
        q = merge(base, off)
        for k, v in off.items():          # merge() keeps dicts; None must replace
            q[k] = v
        r = paint_willow(cv, 100, GY, 150, 110, L, np.random.default_rng(1), q)
        if first is None:
            first = r
        cells.append(cv.render(pal)); labs.append(name)
    F = first["F"]; x = np.nan_to_num(F["x"], nan=-1); m = x >= 0
    skyg = np.full((H, W, 3), 214, np.uint8); skyg[GY:] = 150
    no = skyg.copy(); no[m & F["lit"]] = 158; no[m & ~F["lit"]] = 30
    fv = skyg.copy(); g6 = (np.array(GREY6)[np.clip(np.round(x), 0, 5).astype(int)] * 255).astype(np.uint8)
    fv[m] = np.stack([g6] * 3, -1)[m]
    CL = first["CL"]; M = first["M"]
    cl = skyg.copy(); v = (np.clip(CL, 0, 1) * 255).astype(np.uint8); cl[M] = np.stack([v] * 3, -1)[M]
    pre = [(no, "1 notan: two families"), (fv, "2 form value (6 steps)"), (cl, "3 clump light field")]
    allc = [c for c, _ in pre] + cells
    alll = [t for _, t in pre] + [f"{i + 4} {t}" for i, t in enumerate(labs)]
    save(label_grid([upscale(c, 2) for c in allc], 3, labels=alll,
                    title="one willow, layer by layer: willow.paint_willow with later layers switched off"),
         f"{OUT}/t_construction.png")
    save(cells[-1][40:100, 30:110], f"{OUT}/t_construction_8x.png", 8)
    return first


def fig_spray():
    """The willow spray glyph at 12x: leaves alternating sides (my first version) against leaves
    hanging from the downward side (one_side=0.85), and with silver undersides."""
    pal = willow_palette("summer", "noon")
    cells, labs = [], []
    for name, kw in [("alternating sides (one_side=0.5)", dict(one_side=0.5)),
                     ("hanging (one_side=0.85)", dict(one_side=0.85)),
                     ("hanging + silver undersides", dict(one_side=0.85, silver_p=0.5))]:
        h, w = 40, 56
        S = -np.ones((h, w), int)
        allowed = np.ones((h, w), bool)
        rng = np.random.default_rng(4)
        for (x, y, a) in [(6, 30, -50), (20, 34, -70), (34, 30, -110), (44, 34, -130), (14, 16, -35), (38, 14, -140)]:
            willow_spray(S, allowed, x, y, np.radians(a), 9, 4, rng, lit_side=(-0.6, -0.8), **kw)
        cv = Canvas(w, h); cv.mat[:] = LEAFS; cv.step[:] = 0
        on = S >= 0
        mat = np.where(S >= 10, SILVER, LEAF); st = np.where(S >= 10, S - 10, S)
        cv.mat[on] = mat[on]; cv.step[on] = np.maximum(st[on], 0)
        cells.append(upscale(cv.render(pal), 6)); labs.append(name)
    save(label_grid(cells, 3, labels=labs, title="foliage.willow_spray: a 1-px twig, narrow leaves every 1-2 px, drooping"),
         f"{OUT}/t_spray.png")


def fig_clump():
    """The clump grammar on a plain dome, in Ferrari's own hedge ramp (V09 indices 42,41,50,40,49):
    marks drawn from the raw light field vs from the clump light field."""
    V = load("V09"); pal = V["pal"]; ramp = [42, 41, 50, 40, 49]
    h, w = 70, 110
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    mask = ((xx - 55) / 50) ** 2 + ((yy - 62) / 52) ** 2 < 1
    mask &= yy < 66
    light = np.clip(1.1 - ((xx - 35) ** 2 + (yy - 15) ** 2) ** 0.5 / 70, 0, 1) * mask
    def show(S, M):
        img = np.zeros((h, w, 3), np.uint8); img[:] = pal[252]
        for s, i in enumerate(ramp):
            img[M & (S == s)] = pal[i]
        return img
    rng = np.random.default_rng(3)
    S0 = mid_speckle(None, light, rng, top=4, mask=mask)
    rng = np.random.default_rng(3)
    CL = np.full((h, w), -1.0)
    _, MC = clump_foliage(mask, light, rng, 4, field=CL, r=(3.0, 7.0), vo=0.0, vg=2.6, vl=1.8, vb=2.0)
    MC |= mask
    CL = np.where(CL < 0, light * 0.5, CL)
    S1 = mid_speckle(None, CL, rng, top=4, mask=MC, dens=(0.0, 0.7), hi=(0.45, 0.85), clump=0.2)
    S2, M2 = fringe(S1, MC, np.clip(CL * 1.3, 0, 1), rng, 4, hole_p=0.12, bump_p=0.35, bump_w=(1, 3), peak_p=0.4, rim_p=1.0)
    clg = np.zeros((h, w, 3), np.uint8); clg[:] = 230
    v = (np.clip(CL, 0, 1) * 255).astype(np.uint8); clg[MC] = np.stack([v] * 3, -1)[MC]
    cells = [upscale(show(S0, mask), 4), upscale(clg, 4), upscale(show(S1, MC), 4), upscale(show(S2, M2), 4)]
    labs = ["marks from the raw light: camouflage", "the clump light field", "marks from the clump field",
            "+ fringe: the crest where clumps end"]
    save(label_grid(cells, 2, labels=labs, title="foliage.clump_foliage -> mid_speckle -> fringe, in Ferrari's V09 hedge ramp"),
         f"{OUT}/t_clump.png")


def fig_decompose():
    """Reading Ferrari's hedge as (ramp, step): the copyist's input and the reference."""
    V = load("V09"); idx = V["idx"][222:282, 265:395]; pal = V["pal"]
    ramps = [[42, 41, 50, 40, 49], [86, 83, 84, 85]]
    known = np.isin(idx, sum(ramps, []))
    R, S, table = decompose(idx, pal, ramps)
    ref = pal[idx]
    rmap = np.zeros(idx.shape + (3,), np.uint8); rmap[:] = 40
    rmap[known & (R == 0)] = (60, 200, 60); rmap[known & (R == 1)] = (60, 120, 255)
    smap = np.zeros(idx.shape + (3,), np.uint8); smap[:] = 40
    sv = (S / 4.0 * 255).astype(np.uint8); smap[known] = np.stack([sv] * 3, -1)[known]
    light = ndi.gaussian_filter(np.where(known, S / 4.0, 0), 2) / np.maximum(ndi.gaussian_filter(known.astype(float), 2), 1e-3)
    lmap = np.zeros(idx.shape + (3,), np.uint8); lmap[:] = 40
    lv = (np.clip(light, 0, 1) * 255).astype(np.uint8); lmap[known] = np.stack([lv] * 3, -1)[known]
    save(label_grid([upscale(a, 3) for a in (ref, rmap, smap, lmap)], 2,
                    labels=["Ferrari V09 hedge (native pixels)", "which ramp: lit hedge (green) / far hedge (blue)",
                            "step on its ramp (0 dark .. 4 light)", "the copyist's sketch: blurred light"],
                    title="copyharness.decompose + sketch: the only things the copy is allowed to see"),
         f"{OUT}/t_decompose.png")


def fig_contact():
    """A bush pasted on gravel against the same bush sitting in it (contact line, pocket, stones
    over the foot, cast shadow = the gravel two steps darker)."""
    L = sun(30, 32)
    cells = []
    for sit in (False, True):
        w, h, gy = 150, 90, 78
        cv = Canvas(w, h); cv.mat[:40] = SKY; cv.step[:40] = 2
        rng = np.random.default_rng(8)
        gm = np.zeros((h, w), bool); gm[40:] = True
        gravel(cv, gm, rng, L, y_near=h, y_far=40, stone=(0.8, 3.0), dens=0.45)
        if sit:
            cast_shadow(cv, 70, gy, 90, 60, L, fore=0.2, rng=rng)
            paint_bush(cv, 70, gy, 90, 60, L, np.random.default_rng(2), foot="gravel")
        else:
            paint_bush(cv, 70, gy, 90, 60, L, np.random.default_rng(2), valley=0)
            # undo the contact pass for the "sticker" panel: repaint the gravel under the foot
            fol = (cv.mat != SKY) & (cv.mat != GRAVEL)
            cv2 = Canvas(w, h); cv2.mat[:40] = SKY; cv2.step[:40] = 2
            gravel(cv2, gm, np.random.default_rng(8), L, y_near=h, y_far=40, stone=(0.8, 3.0), dens=0.45)
            cv2.mat[fol] = cv.mat[fol]; cv2.step[fol] = cv.step[fol]
            cv = cv2
        cells.append(cv.render(scene_palette("summer", "golden")))
    save(side_by_side([upscale(c[30:90, 10:140], 5) for c in cells],
                      ["pasted: a flat foot on untouched gravel", "sitting: contact line, pocket, stones over the foot, cast shadow"],
                      height=300), f"{OUT}/t_contact.png")


def fig_micro():
    """The bottom of the LOD ladder at 10x: 14, 8, 5 and 3 px wide."""
    L = sun(25, 40)
    cells = []
    for W_ in (14, 8, 5, 3):
        cv = Canvas(22, 16); cv.mat[:12] = SKY; cv.step[:12] = 2; cv.mat[12:] = GRAVEL; cv.step[12:] = 3
        paint_bush(cv, 11, 12, W_, W_ * 0.74, L, np.random.default_rng(3))
        cells.append(upscale(cv.render(scene_palette("summer", "noon")), 10))
    save(label_grid(cells, 4, labels=["14 px", "8 px", "5 px", "3 px: still a dab"],
                    title="willow.paint_bush below the leaf scale: never vanish"), f"{OUT}/t_micro.png")


if __name__ == "__main__":
    for f in (fig_decompose, fig_clump, fig_spray, fig_construction, fig_contact, fig_micro):
        print(f.__name__); f()
