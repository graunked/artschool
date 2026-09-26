"""Stage 6: the technique as one parameterised call, swept on labelled sheets.

willow_cell(size, density, season, hour, sun_az, sun_el, camera, distance, seed) -> RGB cell.
  size     : bush width in pixels (3..150); the level of detail follows it (willow.lod_params)
  density  : 0 sparse, open, see-through .. 1 dense, closed
  season   : summer | spring | silvery | autumn
  hour     : morning | noon | golden | dusk | overcast (sun colour, ambient, key)
  sun_az/el: sun direction (az 0 = from the left, 90 = from behind the viewer, 180 = right, 270 = behind)
  camera   : 'low' (eye near the ground: the stem hollow shows) | 'oblique' (30 deg from above:
             the crown shows more top, the base hides, the cast shadow lies on the gravel)
  distance : 0 near .. 0.8 far (depth palette: darks lift and cool, lights stay)
"""
import numpy as np, os, sys
from wpaint import *
from willow import paint_bush, BAND, LEAFS
from palette import willow_palette
from context import gravel, scene_palette
from stage4 import sky_ground

OUT = "sheets"; os.makedirs(OUT, exist_ok=True)


def rotate_x(L, deg):
    a = np.radians(deg)
    x, y, z = L
    return np.array([x, y * np.cos(a) - z * np.sin(a), y * np.sin(a) + z * np.cos(a)])


def density_params(d):
    """Sparse willows (young, browsed, flood-thinned) are open: fewer wands, smaller clumps, more
    stems and sky showing; dense ones are closed domes."""
    return dict(form=dict(nw=int(14 + 26 * d), lobe_r=(0.05 + 0.03 * d, 0.09 + 0.04 * d), lobe_gap=1.2 - 0.5 * d,
                          skirt_n=int(1 + 3 * d), crown_w=0.3),
                stems=dict(see=1.0 + 0.6 * (1 - d), zone_t=0.22 - 0.1 * (1 - d)))


def willow_cell(size=90, density=0.7, season="summer", hour="noon", sun_az=25, sun_el=40, camera="low",
                distance=0.0, seed=1, cw=None, ch=None):
    cw = cw or int(max(48, size * 1.35 + 20)); ch = ch or int(max(40, size * 0.95 + 24))
    gy = ch - 8 if camera == "low" else ch - 12
    horizon = gy - 6 if camera == "low" else int(ch * 0.18)
    cv = sky_ground(cw, ch, horizon)
    rng = np.random.default_rng(seed)
    gm = np.zeros((ch, cw), bool); gm[horizon:] = True
    gravel(cv, gm, rng, sun(sun_az, sun_el), y_near=ch, y_far=horizon, stone=(0.8, 2.5 if camera == "low" else 3.0), dens=0.35)
    L = sun(sun_az, sun_el)
    Hf = 0.72
    p = density_params(density)
    from context import cast_shadow
    if camera == "oblique":
        L = rotate_x(L, 30)        # camera 30 deg above: world up tilts toward the viewer by 30 deg
        Hf = 0.78
        # from above, the crown is an irregular spread of wand clumps, not a dome with a skirt:
        # no curtain, wider splay, more droop, a broad crown base
        p = dict(p); p["stems"] = dict(p["stems"], see=0.0)
        p["form"] = dict(p["form"], curtain=0, spread=80, droop=0.5, crown_w=0.5, egg_c=0.5, top_flat=0.8, clump_mix=0.55, nclumps=6, view_tilt=30)
    cast_shadow(cv, cw / 2, gy, size, size * 0.72, sun(sun_az, sun_el), fore=0.35 if camera == "oblique" else 0.15, rng=rng)
    band = int(np.clip(round(distance / 0.27), 0, 3))
    paint_bush(cv, cw / 2, gy, size, size * Hf, L, rng, band=band, p=p if size >= 7 else None,
               base_arc=0.12 if camera == "oblique" else 0.0, foot="gravel")
    depths = (0.0, 0.3, 0.55, 0.8)
    pal = scene_palette(season, hour, depths)
    return cv.render(pal)


def pad_to(img, w, h, bg=(24, 24, 28)):
    out = np.zeros((h, w, 3), np.uint8); out[:] = bg
    y0 = (h - img.shape[0]) // 2; x0 = (w - img.shape[1]) // 2
    out[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    return out


def sheet(rows, cols, fn, rowlab, collab, title, path, k=2):
    cells = [upscale(fn(r, c), k) for r in rows for c in cols]
    H = max(c.shape[0] for c in cells); W = max(c.shape[1] for c in cells)
    cells = [pad_to(c, W, H) for c in cells]
    save(label_grid(cells, len(cols), rowlabels=rowlab, collabels=collab, title=title), path)


if __name__ == "__main__":
    which = sys.argv[1:] or ["season_hour", "size_distance", "density_seed", "camera_light", "silver_season"]
    if "season_hour" in which:
        S = ["summer", "spring", "silvery", "autumn"]; Hh = ["morning", "noon", "golden", "overcast", "dusk"]
        sheet(S, Hh, lambda s, h: willow_cell(90, 0.7, s, h, seed=2), S, Hh,
              "sheet A: season x hour (size 90, density 0.7, sun az 25 el 40, low camera)", f"{OUT}/A_season_x_hour.png")
    if "size_distance" in which:
        sizes = [140, 90, 56, 34, 20, 11, 6, 3]; dists = [0.0, 0.3, 0.55, 0.8]
        sheet(dists, sizes, lambda d, s: willow_cell(s, 0.7, "summer", "noon", distance=d, seed=3, cw=190, ch=140), [f"distance {d}" for d in dists],
              [f"{s} px" for s in sizes], "sheet B: size (level of detail) x distance (depth palette); summer noon", f"{OUT}/B_size_x_distance.png", k=1)
    if "density_seed" in which:
        dens = [0.0, 0.35, 0.7, 1.0]; seeds = [11, 12, 13, 14, 15]
        sheet(dens, seeds, lambda d, s: willow_cell(90, d, "summer", "noon", seed=s), [f"density {d}" for d in dens],
              [f"seed {s}" for s in seeds], "sheet C: density x seed (size 90, summer noon)", f"{OUT}/C_density_x_seed.png")
    if "camera_light" in which:
        cams = ["low", "oblique"]; suns = [(20, 35), (60, 50), (160, 35), (200, 25), (265, 25)]
        sheet(cams, suns, lambda c, sn: willow_cell(90, 0.7, "summer", "golden", sn[0], sn[1], camera=c, seed=4),
              [f"camera {c}" for c in cams], [f"sun az {a} el {e}" for a, e in suns],
              "sheet D: camera x sun direction (size 90, summer golden)", f"{OUT}/D_camera_x_light.png")
    if "silver_season" in which:
        S = ["summer", "spring", "silvery", "autumn"]; sizes = [120, 60, 30]
        sheet(sizes, S, lambda z, s: willow_cell(z, 0.8, s, "noon", camera="oblique", seed=6, cw=170, ch=130), [f"{z} px, oblique" for z in sizes], S,
              "sheet E: season x size, oblique camera, noon", f"{OUT}/E_season_x_size_oblique.png")
