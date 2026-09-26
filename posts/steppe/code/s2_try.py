import sys, time, numpy as np, sp, world, scene, palette
tag = sys.argv[1] if len(sys.argv) > 1 else "a"
W, H = 320, 200
t0 = time.time()
cam = world.Iso(W, H, el=30, ppm=24)
cv = scene.paint_scene(W, H, cam, "summer", "noon", seed=1)
sp.save(cv.render(palette.Palette("summer", "noon")), f"stage2/iso_{tag}_3x.png", 3)
print("iso", time.time() - t0)
cam = world.Persp(W, H, eye=1.6, f=260, hy=60)
cv = scene.paint_scene(W, H, cam, "summer", "golden", seed=2, sun_az=15, sun_el=12, plants=dict(backdrop="rim"))
sp.save(cv.render(palette.Palette("summer", "golden")), f"stage2/persp_{tag}_3x.png", 3)
print("persp", time.time() - t0)
