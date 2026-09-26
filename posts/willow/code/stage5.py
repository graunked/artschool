"""Stage 5: willow scrub in context: on a gravel bar, at the water with its reflection, at the edge
of grass."""
import numpy as np, os, sys
from wpaint import *
from willow import paint_bush, LEAFS, BAND
from context import gravel, contact_shadow, water, grass, scene_palette, WATERB, cast_shadow
from stage4 import sky_ground
OUT = "stage5"; os.makedirs(OUT, exist_ok=True)


def fol_mask(cv):
    return (cv.mat != SKY) & (cv.mat != GRAVEL) & (cv.mat != GRASS) & (cv.mat != WATERB) & (cv.mat != 0)


def row_of_bushes(cv, rng, y, mw, bd, L, x0=0, x1=None, gap=(0.45, 0.8), jitter=2, under=True):
    x1 = cv.w if x1 is None else x1
    if under:
        # a low understorey of small willows first, so the row's base is continuous and the crowns
        # above it rise at irregular heights instead of a scalloped border
        x = x0 - mw * 0.2
        while x < x1 + mw * 0.2:
            W_ = mw * rng.uniform(0.35, 0.7)
            paint_bush(cv, x, int(y + rng.uniform(0, jitter + 1)), W_, W_ * rng.uniform(0.35, 0.6), L, rng, band=bd, valley=0)
            x += W_ * rng.uniform(0.4, 0.7)
    x = x0 - mw * 0.2
    while x < x1 + mw * 0.2:
        W_ = mw * float(np.clip(rng.lognormal(0, 0.45), 0.4, 2.2))   # a treeline is not a scalloped border
        paint_bush(cv, x, int(y + rng.uniform(-jitter, jitter)), W_, W_ * rng.uniform(0.5, 0.95), L, rng, band=bd)
        g = rng.uniform(*gap)
        if rng.random() < 0.15:
            g += rng.uniform(0.5, 1.5)            # an occasional opening
        x += W_ * g


def gravel_bar(seed=11, L=sun(30, 38), w=320, h=180):
    rng = np.random.default_rng(seed)
    cv = sky_ground(w, h, 58)
    # far bank: a low strip of grass with two receding rows of scrub
    bank = np.zeros((h, w), bool); bank[58:66] = True
    grass(cv, bank, rng, L, blade=(1, 2), dens=0.4, crest=False)
    row_of_bushes(cv, rng, 62, 12, 3, L)
    row_of_bushes(cv, rng, 66, 18, 2, L, gap=(0.6, 1.1))
    # river channel reflecting the far bank
    wm = np.zeros((h, w), bool); wm[66:96] = True
    yy, xx = np.mgrid[0:h, 0:w]
    shore = 92 + 3 * np.sin(xx / 23.0) + 2 * np.sin(xx / 7.3)
    wm &= yy < shore
    shallow = wm & (yy > shore - 4)
    water(cv, wm, 66, rng, shallow=shallow)
    # the gravel bar
    gm = (yy >= shore) & ~wm
    gravel(cv, gm, rng, L, y_near=h, y_far=92)
    # willows on the bar: back to front
    specs = [(118, 60, 40), (126, 250, 56), (150, 105, 100), (160, 205, 64), (172, 20, 30)]
    for (gy, cx, W_) in specs:
        H_ = W_ * rng.uniform(0.55, 0.8)
        cast_shadow(cv, cx, gy, W_, H_, L, fore=0.22, rng=rng)
        paint_bush(cv, cx, gy, W_, H_, L, rng, band=0, foot="gravel")
    return cv


def water_edge(seed=12, L=sun(35, 35), w=320, h=180, bank_y=104):
    rng = np.random.default_rng(seed)
    cv = sky_ground(w, h, 40)
    g = np.zeros((h, w), bool); g[40:bank_y] = True
    gravel(cv, g, rng, L, y_near=bank_y, y_far=40, stone=(0.8, 2.0), dens=0.4)
    row_of_bushes(cv, rng, 48, 14, 3, L)
    row_of_bushes(cv, rng, 64, 22, 2, L, gap=(0.55, 0.9))
    row_of_bushes(cv, rng, 82, 36, 1, L, gap=(0.5, 0.85))
    for (cx, W_) in ((40, 70), (120, 96), (220, 80), (300, 60)):
        paint_bush(cv, cx, bank_y, W_, W_ * rng.uniform(0.6, 0.85), L, rng, band=0, foot="gravel")
    # a flat bank edge: the contact row
    wm = np.zeros((h, w), bool); wm[bank_y:] = True
    water(cv, wm, bank_y, rng, dash_p=0.07)
    edge = np.zeros((h, w), bool); edge[bank_y] = True
    cv.step[edge & (cv.mat == WATERB)] = 0
    return cv


def grass_edge(seed=13, L=sun(25, 42), w=320, h=180):
    rng = np.random.default_rng(seed)
    cv = sky_ground(w, h, 70)
    g = np.zeros((h, w), bool); g[70:] = True
    grass(cv, g, rng, L, blade=(1, 3), dens=0.5, crest=False)
    row_of_bushes(cv, rng, 76, 16, 2, L)
    row_of_bushes(cv, rng, 92, 30, 1, L, gap=(0.7, 1.4))
    for (cx, gy, W_) in ((70, 140, 96), (190, 150, 120), (285, 132, 60)):
        H_ = W_ * rng.uniform(0.62, 0.78)
        cast_shadow(cv, cx, gy, W_, H_, L, fore=0.16, rng=rng)
        paint_bush(cv, cx, gy, W_, H_, L, rng, band=0, foot="grass")
    # tall grass in front, its blades overlapping the feet of the bushes
    fg = np.zeros((h, w), bool)
    yy, xx = np.mgrid[0:h, 0:w]
    fg |= yy > 138 + 8 * np.sin(xx / 31.0)
    fol = fol_mask(cv)
    grass(cv, fg & ~(fol & (yy < 125)), rng, L, blade=(3, 8), dens=0.8, over=fol & (yy > 120))
    return cv


if __name__ == "__main__":
    for name, fn in (("gravel_bar", gravel_bar), ("water_edge", water_edge), ("grass_edge", grass_edge)):
        cv = fn()
        for s, hr in (("summer", "noon"), ("autumn", "golden")):
            pal = scene_palette(s, hr)
            img = cv.render(pal)
            save(img, f"{OUT}/s5_{name}_{s}_{hr}_1x.png"); save(img, f"{OUT}/s5_{name}_{s}_{hr}_3x.png", 3)
