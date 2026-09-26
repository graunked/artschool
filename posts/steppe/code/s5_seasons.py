import sys, numpy as np, sp, world, scene, palette
tag = sys.argv[1] if len(sys.argv) > 1 else "a"
W, H = 240, 150
cells = []
for season in ["spring", "summer", "autumn", "winter"]:
    cam = world.Persp(W, H, eye=1.6, f=200, hy=45)
    cv = scene.paint_scene(W, H, cam, season, "morning", seed=11, sun_az=25, sun_el=28,
                           plants=dict(backdrop="rim", sage=16, grass=50))
    cells.append(sp.upscale(cv.render(palette.Palette(season, "morning")), 2))
sp.save(sp.grid(cells, 2, labels=["spring", "summer", "autumn", "winter"], title="seasons, morning"), f"stage5/seasons_{tag}.png")
