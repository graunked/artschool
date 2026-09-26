"""Stage 2: one sage in colour, side light and iso, 8x, beside photo crops."""
import sys, numpy as np, sp, sage, scene, palette, world
from PIL import Image
tag = sys.argv[1] if len(sys.argv) > 1 else "a"
pal = palette.Palette("summer", "noon")

def cell(w, h, cam_el, seed, az=25, el=40, W=64, H=52, **kw):
    Lw = scene.sun_world(az, el)
    L = scene.view_light(Lw, cam_el)
    cam = world.Iso(W, H, el=30, ppm=24)
    cv = sp.Canvas(W, H)
    world.paint_ground(cv, cam, world.GroundParams(seed=seed, litter=0.3, stones=0.2), 1)
    rng = np.random.default_rng(seed)
    spr, bx, by = scene.sage_sprite(w, h, rng, L, **kw)
    ox, oy = W // 2 - bx, H - 10 - by
    scene._shadow_done = np.zeros((H, W), bool)
    scene._cast_shadow(cv, cam, spr, ox, oy, (W // 2) / 24, 10 / (24 * cam.s), 24, Lw, 0)
    cv.blit(spr, ox, oy)
    return cv.render(pal)

side = [cell(56, 38, 5, s) for s in range(3)]
iso = [cell(24, 21, 30, s, W=40, H=36) for s in range(3)] + [cell(40, 34, 30, 7)]
sp.save(sp.hstack([sp.upscale(c, 6) for c in side], ["side 6x"] * 3), f"stage2/s2_sage_side_{tag}.png")
sp.save(sp.hstack([sp.upscale(c, 8) for c in iso[:3]] + [sp.upscale(iso[3], 5)], ["iso 24px 8x"] * 3 + ["iso 40px 5x"]), f"stage2/s2_sage_iso_{tag}.png")
