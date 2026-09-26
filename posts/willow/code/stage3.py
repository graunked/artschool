"""Stage 3: colour. The same index canvas under willow palettes: season x hour; silver undersides;
autumn gold. Light as hue shift."""
import numpy as np, os, sys
from wpaint import *
from willow import paint_willow, LEAFS
from palette import willow_palette, SEASONS, HOURS
from stage2 import canvas, NEAR, photo
from lw import load
from PIL import Image
OUT = "stage3"; os.makedirs(OUT, exist_ok=True)

P3 = dict(NEAR)
P3["sprays"] = dict(NEAR["sprays"], silver_p=0.35)


def swatches(pal, mats=((LEAF, 5), (SILVER, 5), (LEAFS, 4), (STEM, 4), (SKY, 4), (GRAVEL, 6)), k=12):
    rows = []
    for m, n in mats:
        row = np.zeros((k, 6 * k, 3), np.uint8)
        for i in range(n):
            row[:, i * k:(i + 1) * k] = pal[(m, i)]
        rows.append(row)
    return np.concatenate(rows, 0)


if __name__ == "__main__":
    cv, r = None, None
    cv = canvas(); paint_willow(cv, 100, 128, 150, 110, sun(25, 40), np.random.default_rng(1), P3)
    combos = [("summer", "noon"), ("silvery", "noon"), ("spring", "morning"), ("autumn", "golden"),
              ("summer", "overcast"), ("autumn", "noon"), ("silvery", "golden"), ("summer", "dusk")]
    cells, labs = [], []
    for s, h in combos:
        pal = willow_palette(s, h)
        img = cv.render(pal)
        sw = swatches(pal)
        img[2:2 + sw.shape[0], 2:2 + sw.shape[1]] = sw
        cells.append(upscale(img, 2)); labs.append(f"{s}, {h}")
    save(label_grid(cells, 4, labels=labs, title="stage 3: one index canvas, eight palettes (swatches: lit, silver, shadow, twig, sky, gravel)"),
         f"{OUT}/s3_palettes.png")
    pal = willow_palette("silvery", "noon")
    imc = cv.render(pal)
    ph = photo("refs/commons_Salix_alba_018_jpg_97338b2b.jpg", (300, 150, 700, 450), grey=False)
    V = load("V09"); fer = V["pal"][V["idx"]]
    save(side_by_side([ph, upscale(imc[40:80, 35:88], 8), upscale(fer[322:362, 40:93], 8)],
                      ["photo: Salix alba, silver undersides", "silvery willow, 8x", "Ferrari V09 foreground bush, 8x"], height=320),
         f"{OUT}/s3_compare_silver.png")
    pal = willow_palette("autumn", "golden"); ima = cv.render(pal)
    pha = photo("refs/more/openverse_Be_Careful_79819297.jpg", (420, 380, 820, 560), grey=False)
    save(side_by_side([pha, upscale(ima[40:80, 35:88], 8), upscale(ima, 2)],
                      ["photo: autumn willow flats, Alaska", "autumn gold, 8x", "whole bush, 2x"], height=320),
         f"{OUT}/s3_compare_autumn.png")
