import sys, time, numpy as np, sp, world, scene, palette
tag = sys.argv[1] if len(sys.argv) > 1 else "a"
W, H = 320, 200
cam = world.Persp(W, H, eye=1.6, f=240, hy=70)
pl = dict(pine=0.35, sage=3.0, bitter=2.0, grass=60, backdrop="hills")
cv = scene.paint_scene(W, H, cam, "summer", "morning", plants=pl, seed=5, sun_az=25, sun_el=30,
                       ground=dict(crust=0.1, litter=0.8))
sp.save(cv.render(palette.Palette("summer", "morning")), f"stage4/park_persp_{tag}_3x.png", 3)
cam = world.Iso(W, H, el=30, ppm=10)
cv = scene.paint_scene(W, H, cam, "summer", "noon", plants=pl, seed=6, sun_az=25, sun_el=45,
                       ground=dict(crust=0.1, litter=0.8))
sp.save(cv.render(palette.Palette("summer", "noon")), f"stage4/park_iso_{tag}_3x.png", 3)
