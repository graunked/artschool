"""Stage 2: one bunchgrass tussock across sizes (LOD), colour, 6x, beside photo."""
import sys, numpy as np, sp, grass, palette, world
from PIL import Image
tag = sys.argv[1] if len(sys.argv) > 1 else "a"
season = sys.argv[2] if len(sys.argv) > 2 else "summer"
pal = palette.Palette(season, "noon")
cells = []
for hh in [40, 26, 16, 10, 7, 4, 2]:
    W, H = 60, 56
    cv = sp.Canvas(W, H)
    cam = world.Iso(W, H, el=30, ppm=24)
    world.paint_ground(cv, cam, world.GroundParams(seed=3, litter=0.2, stones=0.1), 1)
    rng = np.random.default_rng(hh)
    spr, bx, by = grass.paint_tussock(grass.TussockParams(w=hh * 0.8, h=hh, az=20), rng)
    cv.blit(spr, W // 2 - bx, H - 6 - by)
    cells.append(sp.upscale(cv.render(pal), 5))
sp.save(sp.hstack(cells, [f"h={h}" for h in [40, 26, 16, 10, 7, 4, 2]]), f"stage2/s2_grass_{tag}.png")
