"""Hero scenes at 480x270 (the studio's canvas ladder middle rung), 1x and 3x."""
import sys, numpy as np, sp, world, scene, palette
W, H = 480, 270
jobs = {
    "sage_flat_golden": dict(season="summer", hour="golden", az=15, el=11, hy=78, f=380, seed=21,
                             plants=dict(sage=22, grass=45, backdrop="rim"), ground=dict(crust=0.4)),
    "parkland_morning": dict(season="summer", hour="morning", az=25, el=26, hy=110, f=360, seed=22,
                             plants=dict(sage=2, grass=80, bitter=3, pine=0.3, backdrop="hills"),
                             ground=dict(crust=0.05, litter=0.9, stones=0.3)),
    "spring_balsamroot": dict(season="spring", hour="morning", az=30, el=30, hy=70, f=360, seed=23,
                              plants=dict(sage=14, grass=60, backdrop="rim", forb=1.6), ground=dict(crust=0.3)),
    "winter_sage": dict(season="winter", hour="noon", az=30, el=22, hy=80, f=360, seed=24,
                        plants=dict(sage=20, grass=30, backdrop="rim"), ground=dict(crust=0.2)),
}
for name in (sys.argv[1:] or jobs):
    j = jobs[name]
    cam = world.Persp(W, H, eye=1.6, f=j["f"], hy=j["hy"])
    cv = scene.paint_scene(W, H, cam, j["season"], j["hour"], plants=j["plants"], ground=j["ground"],
                           seed=j["seed"], sun_az=j["az"], sun_el=j["el"])
    img = cv.render(palette.Palette(j["season"], j["hour"]))
    sp.save(img, f"hero/{name}_1x.png")
    sp.save(img, f"hero/{name}_3x.png", 3)
    print(name, flush=True)
