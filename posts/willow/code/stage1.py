"""Stage 1: one willow bush as a mass, greyscale.
s1_light.png : notan | form value field | painted, under four suns
s1_compare.png: photo | pixels | Ferrari (grey) at 8x
s1_edge.png  : the crest: photo | pixels | Ferrari at 8x
"""
import numpy as np, os
from wpaint import *
from willow import paint_willow, grey_pal, LEAFS
from lw import load
from PIL import Image

OUT = "stage1"; os.makedirs(OUT, exist_ok=True)
W, H, GY = 128, 100, 88
SUNS = [("sun left, 35 deg", sun(20, 35)), ("sun front-left, 45", sun(55, 45)),
        ("sun right-back, 30", sun(200, 30)), ("sun behind, 25", sun(265, 25))]


def canvas():
    cv = Canvas(W, H); cv.mat[:GY] = SKY; cv.step[:GY] = 2; cv.mat[GY:] = GRAVEL; cv.step[GY:] = 4
    return cv


def field_views(r, pal):
    F = r["F"]; x = np.nan_to_num(F["x"], nan=-1)
    base = canvas().render(pal)
    no = base.copy(); fv = base.copy()
    m = x >= 0
    no[m & (x >= 2.9)] = int(0.62 * 255); no[m & (x < 2.9)] = int(0.12 * 255)
    g6 = np.array(GREY6)[np.clip(np.round(x), 0, 5).astype(int)]
    fv[m] = (np.stack([g6] * 3, -1)[m] * 255).astype(np.uint8)
    return no, fv


def ferrari_grey(crop):
    g = (0.2126 * crop[..., 0] + 0.7152 * crop[..., 1] + 0.0722 * crop[..., 2]).astype(np.uint8)
    return np.stack([g] * 3, -1)


if __name__ == "__main__":
    pal = grey_pal()
    cells, labs = [], []
    for name, L in SUNS:
        cv = canvas(); r = paint_willow(cv, 64, GY, 90, 68, L, np.random.default_rng(1))
        no, fv = field_views(r, pal)
        for im, t in ((no, "notan"), (fv, "form value"), (cv.render(pal), "painted")):
            cells.append(upscale(im, 3)); labs.append(f"{t} | {name}")
    save(label_grid(cells, 3, labels=labs, title="stage 1: one willow as a mass (greyscale), four suns"), f"{OUT}/s1_light.png")
    # comparisons at 8x
    cv = canvas(); r = paint_willow(cv, 64, GY, 90, 68, SUNS[0][1], np.random.default_rng(1))
    img = cv.render(pal)
    ph = np.array(Image.open('refs/crop_nzbush.png').convert('L').convert('RGB'))
    V = load("V09"); fer = V["pal"][V["idx"]]
    save(side_by_side([ph[40:250, 40:300], upscale(img[30:78, 24:84], 8), upscale(ferrari_grey(fer[240:288, 280:340]), 8)],
                      ["photo (grey)", "willow, stage 1, 8x", "Ferrari V09 hedge (grey), 8x"], height=384), f"{OUT}/s1_compare.png")
    save(side_by_side([ph[0:110, 60:300], upscale(img[18:42, 30:84], 8), upscale(ferrari_grey(fer[222:246, 300:354]), 8)],
                      ["photo crest (grey)", "willow crest, 8x", "Ferrari hedge crest (grey), 8x"], height=192), f"{OUT}/s1_edge.png")
    save(img, f"{OUT}/s1_bush_1x.png"); save(img, f"{OUT}/s1_bush_4x.png", 4)
