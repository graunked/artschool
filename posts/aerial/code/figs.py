"""figs.py -- teaching figures for the aerial tutorial, made with the study's real code.

Imports ~/work/pixelart-studies/aerial/lib read-only (no .pyc written there, no scene cache).
    nice -n 10 python3 figs.py [sprites|water|far|all]
Writes ../img/fig_*.png
"""
import os, sys
sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "1")
LIB = os.path.expanduser("~/work/pixelart-studies/aerial/lib")
sys.path.insert(0, LIB)
import numpy as np
from PIL import Image, ImageDraw
from camera import Iso, Persp
from pal import LIGHTS, Palette, MAT_ID
from canvas import Canvas
import tech_canopy as TC, tech_rock as TR, tech_water as TW
import study as S

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "img")


def render(cv, light):
    img, n = Palette(light).render(cv.mat, cv.step, cv.band)
    return Image.fromarray(img)


def sprites():
    """conifer spur sprites by size and stand density, spire marks, broadleaf, stones: 8x."""
    L = LIGHTS["afternoon"]; sun = L.vec
    cam = Iso((0, 0, 0), 1.0, 30, 0)
    cv = Canvas(150, 44)
    cv.fill(np.ones((44, 150), bool), "meadow", 3)
    x = 4
    for w in (5, 7, 10, 14):
        h = int(round(w / 0.4 * np.cos(np.radians(30))))
        for occl in (0.0, 0.9):
            spr = TC.conifer_sprite(w, min(h, 38), sun, cam, seed=3, lit_ok=True, occl=occl)
            cv.blit(spr, x + spr["ax"], 41, 10.0)
            x += w + 4
    for w in (2, 3, 4):
        spr = TC.spire_mark(w, 3 * w, sun, cam, "conifer", True)
        cv.blit(spr, x + spr["ax"], 41, 10.0); x += w + 3
    spr = TC.broadleaf_sprite(12, 10, sun, cam, seed=2); cv.blit(spr, x + 7, 41, 10.0); x += 16
    im = render(cv, L)
    S.up(im, 8).save(os.path.join(OUT, "fig_sprites.png"))
    # stones on a scree matrix, sizes 2..6 px, lit and shaded
    cv = Canvas(64, 20); cv.fill(np.ones((20, 64), bool), "scree", 2)
    x = 3
    for w in (2, 3, 4, 5, 6):
        for lit in (True, False):
            spr = TR.stone_sprite(w, lit, -1)
            cv.blit(spr, x + spr["ax"], 14, 5.0); x += w + 3
    S.up(render(cv, L), 8).save(os.path.join(OUT, "fig_stones.png"))
    print("sprites")


def water():
    """the same river at three Fresnel boosts: physics (1.0), mine (1.8), too much (3.5)."""
    from subjects import river_valley
    from painter import paint
    sc = river_valley(seed=2)
    lv = sc.water["level"]; j = int(470 / sc.T.cell); xs = np.nonzero(np.isfinite(lv[j]))[0]
    tx = float(xs.mean() * sc.T.cell)
    ims = []
    for b in (1.0, 1.8, 3.5):
        TW.WATER_BOOST = b
        cam = Iso((tx, 470, sc.T.height(tx, 470)), 0.8, 30, 0, w=128, h=80)
        img, info = paint(sc, cam, LIGHTS["afternoon"])
        ims.append(Image.fromarray(img))
    TW.WATER_BOOST = 1.8
    S.panel("the same river, Fresnel x1.0 (physics) | x1.8 (what I shipped) | x3.5", ims,
            ["x1.0: darker, flatter, few sky streaks", "x1.8: dark mirror with sky sheen", "x3.5: a pale sky-blue road"], k=4).save(
        os.path.join(OUT, "fig_water_boost.png"))
    print("water")


def far():
    """range evening: no depth bands vs bands; band map; palette budget off vs 64."""
    from subjects import range_valley
    from painter import paint
    import pickle
    SP = os.environ.get("SCRATCH", "/tmp")
    p = os.path.join(SP, "range.pkl")
    if os.path.exists(p):
        sc = pickle.load(open(p, "rb"))
    else:
        sc = range_valley(seed=8); pickle.dump(sc, open(p, "wb"))
    L = LIGHTS["evening"]
    cam = lambda: Persp.at((4500, 300, 3300), yaw=0, pitch=24, vfov=38)
    a, ia = paint(sc, cam(), L, bands=None)
    b, ib = paint(sc, cam(), L, bands=[2500, 4500, 7000], return_canvas=True)
    c, ic = paint(sc, cam(), L, bands=[2500, 4500, 7000], budget=None)
    band = ib["canvas"].band
    cols = np.array([[40, 40, 60], [90, 90, 130], [150, 150, 190], [215, 215, 240]], np.uint8)
    bm = Image.fromarray(cols[np.clip(band, 0, 3)])
    S.panel("far view, evening", [Image.fromarray(a), bm, Image.fromarray(b)],
            [f"no depth bands ({ia['colours']} colours)", "the depth bands (dithered seams)", f"bands + collapse ({ib['colours']} colours)"],
            k=2).save(os.path.join(OUT, "fig_far_bands.png"))
    S.panel("palette budget", [Image.fromarray(c), Image.fromarray(b)],
            [f"no budget: {ic['colours']} colours", f"budget 64: {ib['colours']} colours"], k=3).save(
        os.path.join(OUT, "fig_far_budget.png"))
    print("far", ia["colours"], ib["colours"], ic["colours"])


if __name__ == "__main__":
    what = sys.argv[1:] or ["all"]
    if "sprites" in what or "all" in what: sprites()
    if "water" in what or "all" in what: water()
    if "far" in what or "all" in what: far()
