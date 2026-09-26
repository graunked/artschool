"""paint(): world + camera + light -> index canvas.  The whole technique in one call."""
import numpy as np
from .world import World
from .render import Camera, Raster, value_design, MAT_SKY, MAT_WATER, MAT_SOIL
from .marks import tuft_population, glyph_area, paint_tufts, MarkParams, keep_for, quantize_base
from .world import SPECIES
from . import palette as P
from .calibrate import correction
from .noise import stable_threshold


def centre_on_bank(world, x=0.0, back=0.0):
    yc, hw = world.channel(np.array(x))
    if back == 0.0:
        return float(yc), float(world.water_level)
    cy = float(yc) + hw + back
    cz = float(world.height(np.array(x), np.array(cy)))
    return cy, cz


def paint(world: World, cam: Camera, hour="day", vparams=None, mparams=None, seed=0,
          marks=True, return_layers=False):
    pal, sun = P.build(hour)
    R = Raster(world, cam)
    R.sun_screen_x = sun[0]
    V, lit, lam, graze = value_design(R, sun, vparams)
    H, W = V.shape
    mp = MarkParams(**(mparams or {}))
    rng = np.random.default_rng(seed)
    # base layer: design value, stochastic dither between adjacent steps
    base = V.copy()
    canvas = np.full((H, W), np.nan)
    cmat = np.zeros((H, W), np.int8)
    prio = np.full((H, W), -1, np.int64)
    ppm, s, c = cam.ppm, cam.s, cam.c
    fin = np.isfinite(R.depth)
    grass = (R.mat == 1)
    # world-aligned pixel rows for ordered dither (stable under whole-pixel pans)
    R.iy = (np.arange(H)[:, None] - int(np.floor(cam.u0() * ppm + 1e-6)) + np.zeros((1, W), int)).astype(np.int64)
    bgv = V.copy()
    bgv = water_background(R, V, bgv)   # water: light sky tint, or the dark mirrored bank
    bgv[R.sky] = 5.0
    R.bg_value = bgv
    if marks:
        sp = SPECIES[world.species]
        hmean = sp[0] * (1 - 0.75 * world.grazing)
        ell = hmean * ppm * np.sqrt(c * c + (mp['lean_top'] * s ** 3) ** 2)
        keep = keep_for(world, mp, ell, ppm, s)
        R.keep, R.ell = keep, ell
        pop = None
        if fin.any():
            margin = 3 * sp[0] + 3.0 / ppm
            x0, x1 = R.X.min() - margin, R.X.max() + margin
            y0 = R.Y[fin].min() - margin
            y1 = R.Y[fin].max() + margin + 3 * sp[0] * c / max(s, 0.1)
            pop = tuft_population(world, x0, x1, y0, y1, min(1.0, keep * 1.8), seed=world.seed + seed)
            if pop is not None:
                pop['keep'] = keep
        # tone lock: shift the design so the squinted tone at this zoom equals the tone
        # at the reference distance (see calibrate.py)
        Vd = V.copy()
        if mp["tone_lock"]:
            Vd = V - mp["lock"] * correction(world, mp, ell, V, lit, mp["ell_ref"])
            Vd = np.where(lit, np.clip(Vd, 3.0, np.maximum(V, 5.0)), np.clip(Vd, 0.0, 2.9))
        bw = np.where(lit, mp["band_w"], mp["band_w_shadow"])
        R.qbase = np.clip(quantize_base(Vd, R.X, R.Y, ppm, s, bw, iy=R.iy), 0, 6)
        if pop is not None:
            paint_tufts(canvas, cmat, prio, R, Vd, lit, world, pop, mp,
                        fam_top=(2.9, 6.0), fam_bot=(0.0, 0.0))
        base = Vd
    else:
        ell = 99
    band_w = np.where(lit, mp["band_w"], mp["band_w_shadow"])   # lit: flat planes; shadow: checker
    lo = np.floor(base)
    f = base - lo
    g = np.clip((f - band_w / 2) / np.maximum(1 - band_w, 1e-3), 0, 1)   # near: flat bands; far: full stochastic
    q_base = quantize_base(base, R.X, R.Y, ppm, s, band_w, iy=R.iy)
    marked = ~np.isnan(canvas)
    q = np.where(marked, np.floor(np.nan_to_num(canvas) + 0.5).astype(int), q_base)
    q = np.clip(q, 0, 6)
    dead = world.dead(R.X, R.Y)
    base_straw = (dead + 0.25 * (stable_threshold(R.X, R.Y * s, ppm, px=1.0, seed=33) - 0.5)) > 0.62
    straw = np.where(marked, (cmat == 1) & (canvas >= 3.5), base_straw)
    idx = np.where(straw, P.STRAW0, P.GREEN0) + q
    # context materials (only where no mark covers them)
    free = ~marked
    wat = R.water & free
    # water: sky colour, darker toward the near bank, a few horizontal dashes
    wi = np.full((H, W), 2)
    idx[wat] = P.WATER0 + 2
    idx = reflect_water(idx, R, wat, rng)
    soil = (R.mat == MAT_SOIL) & free
    idx[soil] = P.SOIL0 + (lit[soil] & (stable_threshold(R.X, R.Y * s, ppm, px=1.0, seed=35)[soil] < 0.3))
    sky = R.sky & free
    rows = np.broadcast_to(np.arange(H)[:, None], (H, W))
    idx[sky] = P.SKY0 + np.clip((rows[sky] * 8) // H, 0, 7)
    if return_layers:
        return idx, pal, dict(R=R, V=V, lit=lit, marked=marked)
    return idx, pal


def reflect_water(idx, R, wat, rng):
    """Water: the far bank mirrored about its waterline, one step darker, broken by a few
    horizontal dash rows; open water a two-tint sky reflection lightening toward the far
    shore.  A 1-px dark contact line where the far bank meets the water."""
    H, W = idx.shape
    out = idx.copy()
    for c in range(W):
        col = wat[:, c]
        if not col.any():
            continue
        r = 0
        while r < H:
            if not col[r]:
                r += 1
                continue
            r0 = r
            while r < H and col[r]:
                r += 1
            r1 = r            # water run [r0, r1)
            n = r1 - r0
            for k in range(n):
                rr = r0 + k
                # sky tint: lighter near the far shore
                out[rr, c] = P.WATER0 + (3 if k < n * 0.35 else 2)
                src = r0 - 1 - k
                if src >= 0:
                    si = idx[src, c]
                    if P.GREEN0 <= si < P.GREEN0 + 7 or P.STRAW0 <= si < P.STRAW0 + 7:
                        base = P.GREEN0 if si < P.STRAW0 else P.STRAW0
                        q = si - base
                        # reflection: the bank's darks in the grass ramp (darks stay),
                        # its lights fall into the dark water tints (lights drop)
                        out[rr, c] = base + q if q <= 1 else (P.WATER0 if q <= 3 else P.WATER0 + 1)
                    elif P.SOIL0 <= si <= P.SOIL0 + 1:
                        out[rr, c] = P.SOIL0
            # contact line at the far foot
            if r0 > 0 and not wat[r0 - 1, c]:
                si = idx[r0 - 1, c]
                if P.GREEN0 <= si < P.GREEN0 + 7:
                    out[r0 - 1, c] = P.GREEN0
    # wind lines: sparse horizontal dashes of light tint, anchored to the water surface
    # (a world-stable threshold stretched 6:1 along x) so they don't flicker on zoom
    if wat.any():
        t = stable_threshold(R.X / 6.0, R.Y * R.cam.s, R.cam.ppm, px=1.0, seed=91)
        dash = wat & (t > 0.955)
        out[dash] = P.WATER0 + 3
    return out


def water_background(R, V, bgv, sky_value=4.6):
    """What a blade crossing the water is seen against: the mirrored bank above the far
    waterline (its design value, lights dropped), else the light sky tint."""
    H, W = V.shape
    wat = R.water
    out = bgv.copy()
    for c in range(W):
        col = wat[:, c]
        if not col.any():
            continue
        r = 0
        while r < H:
            if not col[r]:
                r += 1
                continue
            r0 = r
            while r < H and col[r]:
                r += 1
            for rr in range(r0, r):
                src = r0 - 1 - (rr - r0)
                if src >= 0 and R.mat[src, c] in (1, 3):
                    out[rr, c] = min(V[src, c], 2.0)
                else:
                    out[rr, c] = sky_value
    return out
