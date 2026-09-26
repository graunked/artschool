"""Stage 3: one ponderosa, three ages, colour, beside photos."""
import sys, numpy as np, sp, pine, palette
from PIL import Image
tag = sys.argv[1] if len(sys.argv) > 1 else "a"
pal = palette.Palette("summer", "noon")
cells = []
for age, seed in [("young", 1), ("mature", 2), ("mature", 3), ("old", 4)]:
    rng = np.random.default_rng(seed)
    cv, bx, by, info = pine.paint_pine(pine.PineParams(h=150, age=age, az=25, el=40, crown_frac=0.45 if age == "old" else 0.66, width=0.62 if age == "old" else 0.36 if age == "young" else 0.5), rng)
    out = sp.Canvas(cv.w, cv.h)
    out.mat[:] = sp.SKY
    out.step[:] = np.clip((np.arange(cv.h) / cv.h * 6).astype(int), 0, 5)[:, None]
    out.mat[by + 1:] = sp.GROUND
    out.step[by + 1:] = 3
    out.blit(cv, 0, 0)
    cells.append(sp.upscale(out.render(pal), 3))
ph = []
for f in ["refs/ponderosa_pine_savanna/commons_Turnbull_National_Wildlife_Refuge_pine_savanna_j_f9182dfb.jpg",
          "~/work/naturalist/plants/photos/ponderosa_pine_forest_eastern_Washington/commons_1939_Class_4C_Keen_Ponderosa_Pine_Tree_Classific_2f5058e7.jpg"]:
    ph.append(sp.fit(Image.open(f), cells[0].shape[0]))
sp.save(sp.hstack(cells + ph, ["young", "mature", "mature", "old", "photo", "photo"]), f"stage3/s3_pine_{tag}.png")
