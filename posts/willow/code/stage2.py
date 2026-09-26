"""Stage 2: stems and leaves. A near willow (150 px) whose marks are narrow willow leaves along the
wands, stems through the canopy and at the base, wand tips at the crest."""
import numpy as np, os, sys
from wpaint import *
from willow import paint_willow, grey_pal, ferrari_pal, LEAFS
from lw import load
from PIL import Image
OUT = "stage2"; os.makedirs(OUT, exist_ok=True)
W, H, GY = 200, 140, 128

NEAR = dict(clump=dict(r=(5.0, 10.0)), sprays=dict(r_sep=3.4, length=(4, 8), dens_lo=0.2, every=(1, 2)), lit=dict(dlen=(3, 5), spacing=2.0, droop=(20, 60)),
            shadow=dict(), tips=dict(p=0.2, lens=(2, 5)), wands_visible=dict(p=0.6, t=(0.3, 0.9), brk=0.1))


def canvas():
    cv = Canvas(W, H); cv.mat[:GY] = SKY; cv.step[:GY] = 2; cv.mat[GY:] = GRAVEL; cv.step[GY:] = 4
    return cv


def near(seed=1, L=sun(25, 40), p=None):
    cv = canvas()
    r = paint_willow(cv, 100, GY, 150, 110, L, np.random.default_rng(seed), p if p is not None else NEAR)
    return cv, r


def photo(path, box, grey=True):
    im = Image.open(path).convert("L" if grey else "RGB").convert("RGB")
    return np.array(im.crop(box))


if __name__ == "__main__":
    gp, fp = grey_pal(), ferrari_pal()
    cv, r = near()
    save(label_grid([upscale(cv.render(gp), 3), upscale(cv.render(fp), 3)], 2,
                    labels=["greyscale", "colour check (Ferrari's V09 hedge palette)"],
                    title="stage 2: a near willow: sprays of narrow leaves on wands, stems at the base"), f"{OUT}/s2_near.png")
    img = cv.render(gp); imc = cv.render(fp)
    # where the light is vs where it is not: same bush, 8x
    lit_box = (30, 45, 90, 85); sh_box = (120, 70, 180, 110); base_box = (70, 88, 130, 128)
    crops = [upscale(imc[b[1]:b[3], b[0]:b[2]], 8) for b in (lit_box, sh_box, base_box)]
    save(side_by_side(crops, ["in the light: sprays, lit leaves", "in shadow: dim base, few dim sprays", "the base: stems, hollow"], height=320),
         f"{OUT}/s2_light_shadow_base.png")
    V = load("V09"); fer = V["pal"][V["idx"]]
    ph = photo("refs/commons_Salix_triandra_Stara_Desna_Zazymya_jpg_d0d308c4.jpg", (380, 120, 780, 420), grey=False)
    save(side_by_side([ph, upscale(imc[40:80, 35:88], 8), upscale(fer[322:362, 40:93], 8)],
                      ["photo: Salix triandra sprays", "willow sprays, 8x", "Ferrari V09 foreground bush, 8x"], height=320),
         f"{OUT}/s2_compare.png")
    ph2 = photo("refs/crop_nzbush.png", (60, 150, 260, 260), grey=False)
    save(side_by_side([ph2, upscale(imc[92:127, 60:130], 8), upscale(fer[262:282, 300:370], 8)],
                      ["photo: stems at the base", "willow base, 8x", "Ferrari hedge base with trunks, 8x"], height=280),
         f"{OUT}/s2_compare_base.png")
