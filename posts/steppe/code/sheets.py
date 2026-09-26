"""Labelled parameter sheets. Usage: python sheets.py [A B C D E F G]

A  season x hour            (standing view of a sage flat under a coulee rim)
B  plant x size (LOD)       (one plant, width/height in px from near to a few px)
C  density x seed           (iso 30 deg, sage flat)
D  landscape x camera       (sage flat, bunchgrass steppe, bitterbrush margin, ponderosa parkland)
E  sun azimuth x elevation  (iso 30 deg, sage + grass + a pine)
F  zoom ladder              (iso, ppm 48 -> 3 in 1.25x steps: check for pops)
G  season x landscape       (iso 30 deg)
"""
import sys, os, time
import numpy as np
import sp, world, scene, palette, sage, grass, pine

OUT = os.path.join(sp.HERE, "sheets")
os.makedirs(OUT, exist_ok=True)

LANDS = {
    "sage flat":      dict(sage=20, grass=45, bitter=0, pine=0, backdrop="rim"),
    "bunchgrass":     dict(sage=3, grass=110, bitter=0, pine=0, backdrop="hills", grass_drift=0.3),
    "bitterbrush":    dict(sage=6, grass=70, bitter=8, pine=0.08, backdrop="hills"),
    "pine parkland":  dict(sage=2, grass=80, bitter=3, pine=0.35, backdrop="hills"),
}
GROUNDS = {
    "sage flat": dict(crust=0.4, stones=0.25),
    "bunchgrass": dict(crust=0.15, stones=0.15, litter=0.8),
    "bitterbrush": dict(crust=0.2, stones=0.2),
    "pine parkland": dict(crust=0.05, stones=0.3, litter=0.9),
}


def persp_cell(W, H, season, hour, plants, ground=None, seed=0, az=25, el=30, hy=None, f=None):
    cam = world.Persp(W, H, eye=1.6, f=f or W * 0.8, hy=hy or int(H * 0.3))
    cv = scene.paint_scene(W, H, cam, season, hour, plants=plants, ground=ground, seed=seed,
                           sun_az=az, sun_el=el)
    return cv.render(palette.Palette(season, hour))


def iso_cell(W, H, season, hour, plants, ground=None, seed=0, az=25, el=40, ppm=12):
    cam = world.Iso(W, H, el=30, ppm=ppm)
    pl = dict(plants)
    pl["backdrop"] = None
    cv = scene.paint_scene(W, H, cam, season, hour, plants=pl, ground=ground, seed=seed,
                           sun_az=az, sun_el=el)
    return cv.render(palette.Palette(season, hour))


HOUR_SUN = {"morning": (30, 22), "noon": (40, 62), "golden": (15, 12), "dusk": (8, 4),
            "overcast": (40, 50)}


def sheet_A():
    cells, rows = [], ["spring", "summer", "autumn", "winter"]
    cols = ["morning", "noon", "golden", "dusk", "overcast"]
    for season in rows:
        for hour in cols:
            az, el = HOUR_SUN[hour]
            cells.append(sp.upscale(persp_cell(200, 120, season, hour, LANDS["sage flat"],
                                               GROUNDS["sage flat"], seed=3, az=az, el=el), 2))
    return sp.grid(cells, 5, rowlabels=rows, collabels=cols,
                   title="A: season x hour. Standing view (eye 1.6 m) of a sage flat under a basalt rim. 2x")


def sheet_B():
    """Each plant alone on calm ground, at decreasing size. Rows: plant; cols: size in px."""
    pal = palette.Palette("summer", "noon")
    L = scene.view_light(scene.sun_world(25, 40), 20)
    sizes = [64, 40, 24, 14, 8, 4, 2]
    rows = ["sage", "sage (autumn)", "bitterbrush", "bunchgrass", "pine young", "pine mature",
            "pine old"]
    cells = []
    for r in rows:
        for s in sizes:
            W, Hc = 72, 80 if r.startswith("pine") else 56
            cv = sp.Canvas(W, Hc)
            cam = world.Iso(W, Hc, el=30, ppm=24)
            world.paint_ground(cv, cam, world.GroundParams(seed=2, litter=0.15, stones=0.08,
                                                           crust=0.2), 1)
            rng = np.random.default_rng(s)
            if r.startswith("sage") or r == "bitterbrush":
                kind = "bitter" if r == "bitterbrush" else "sage"
                season = "autumn" if "autumn" in r else "summer"
                spr, bx, by = scene.sage_sprite(s * 0.8, s * 0.6, rng, L, kind, season)
                if kind == "bitter":
                    spr.mat[spr.mat == sp.SAGE] = sp.BITTER
            elif r == "bunchgrass":
                spr, bx, by = scene.grass_sprite(s * 0.5, s * 0.6, rng, L)
            else:
                age = r.split()[1]
                spr, bx, by = scene.pine_sprite(s * 1.1, rng, L, age)
            cv.blit(spr, W // 2 - bx, Hc - 6 - by)
            cells.append(sp.upscale(cv.render(pal), 3))
    return sp.grid(cells, len(sizes), rowlabels=rows, collabels=[f"{s} px" for s in sizes],
                   title="B: plant x size (level of detail). Summer noon, 3x. Size = sage width / grass height x1.7 / pine height x0.9")


def sheet_C():
    dens = [4, 12, 25, 45]
    cells = []
    for d in dens:
        for seed in range(4):
            cells.append(sp.upscale(iso_cell(160, 110, "summer", "noon",
                                             dict(sage=d, grass=40, cluster=0.5), GROUNDS["sage flat"],
                                             seed=seed, ppm=16), 2))
    return sp.grid(cells, 4, rowlabels=[f"sage {d}/100m2" for d in dens],
                   collabels=[f"seed {s}" for s in range(4)],
                   title="C: sage density x seed. Iso 30 deg, 16 px/m, summer noon. 2x")


def sheet_D():
    names = list(LANDS)
    cams = ["standing", "iso 16 px/m", "iso 6 px/m"]
    cells = []
    for n in names:
        cells.append(sp.upscale(persp_cell(200, 120, "summer", "morning", LANDS[n], GROUNDS[n],
                                           seed=5, az=25, el=28), 2))
        cells.append(sp.upscale(iso_cell(200, 120, "summer", "noon", LANDS[n], GROUNDS[n], seed=5,
                                         ppm=16), 2))
        cells.append(sp.upscale(iso_cell(200, 120, "summer", "noon", LANDS[n], GROUNDS[n], seed=5,
                                         ppm=6), 2))
    return sp.grid(cells, 3, rowlabels=names, collabels=cams,
                   title="D: landscape x camera. Summer; standing = morning, iso = noon. 2x")


def sheet_E():
    azs = [0, 30, 90, 150, 200]
    els = [10, 30, 60]
    cells = []
    pl = dict(sage=14, grass=40, pine=0.25, bitter=2)
    for el in els:
        for az in azs:
            hour = "golden" if el <= 10 else "noon"
            cells.append(sp.upscale(iso_cell(160, 110, "summer", hour, pl, GROUNDS["sage flat"],
                                             seed=7, az=az, el=el, ppm=14), 2))
    return sp.grid(cells, 5, rowlabels=[f"sun el {e}" for e in els],
                   collabels=[f"az {a}" + (" (left)" if a == 0 else " (front)" if a == 90 else " (right)" if a == 180 else "") for a in azs],
                   title="E: sun azimuth x elevation. Iso 30 deg, 14 px/m. az 0 = sun on the left, 90 = behind the viewer. 2x")


def sheet_F():
    ppms = [48 / 1.25 ** k for k in range(0, 13)]
    cells = []
    pl = dict(sage=16, grass=45, pine=0.2, bitter=2)
    for p in ppms:
        cells.append(sp.upscale(iso_cell(120, 90, "summer", "noon", pl, GROUNDS["sage flat"], seed=9,
                                         ppm=p), 2))
    return sp.grid(cells, 5, labels=[f"{p:.1f} px/m" for p in ppms],
                   title="F: zoom ladder, iso 30 deg, 1.25x steps (same world, same seed). 2x")


def sheet_G():
    seasons = ["spring", "summer", "autumn", "winter"]
    names = list(LANDS)
    cells = []
    for n in names:
        for s in seasons:
            cells.append(sp.upscale(iso_cell(150, 100, s, "morning", LANDS[n], GROUNDS[n], seed=4,
                                             az=25, el=35, ppm=12), 2))
    return sp.grid(cells, 4, rowlabels=names, collabels=seasons,
                   title="G: landscape x season. Iso 30 deg, 12 px/m, morning. 2x")


if __name__ == "__main__":
    which = sys.argv[1:] or list("ABCDEFG")
    for w in which:
        t0 = time.time()
        img = globals()["sheet_" + w]()
        names = dict(A="A_season_x_hour", B="B_plant_x_size", C="C_density_x_seed",
                     D="D_landscape_x_camera", E="E_sun_az_x_el", F="F_zoom_ladder",
                     G="G_landscape_x_season")
        sp.save(img, os.path.join(OUT, names[w] + ".png"))
        print(w, round(time.time() - t0, 1), flush=True)
