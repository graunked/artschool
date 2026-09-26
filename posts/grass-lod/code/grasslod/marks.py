"""Marks: world-anchored grass tufts painted as strokes whose shape follows their size.

The handoff principle
---------------------
Every tuft is an object in the world with its own stable random numbers.  One continuous
quantity per tuft, its projected blade length in pixels (ell = h * ppm * cos(theta)),
decides everything about how it is painted: how many blades, how long, whether it has a
partner stem and a dark gap, whether it is a tick or a dab or a single dot.  Every
*discrete* decision (blade count, stroke length in whole pixels, pair or no pair, drawn or
dropped) compares a continuous function of ell against one of the tuft's own random
numbers.  So as the camera zooms, each tuft changes at its own moment: the vocabulary
dissolves from blades to ticks to dots instead of switching.  It is dither, applied to
the mark vocabulary itself.

Population: a nested hierarchy of jittered grids.  Level k has one tuft per cell of size
s0*2^k and ranks in [4^-(k+1), 4^-k); keeping tufts with rank < K keeps a fraction K of
all tufts, the kept set only ever grows as K grows, and only the coarse levels need to be
enumerated when K is small.  As the camera pulls back, tufts drop out one by one (lowest
rank survives longest), and the survivors stand for more grass each.
"""
import numpy as np
from .noise import lat
from .world import SPECIES


def tuft_population(world, x0, x1, y0, y1, keep, seed=0):
    """All tufts with rank < keep in the world rectangle.  Returns dict of arrays."""
    s0 = SPECIES[world.species][1]
    keep = float(np.clip(keep, 1e-9, 1.0))
    kmin = max(0, int(np.ceil(np.log(1.0 / keep) / np.log(4.0))) - 1)
    out = {"x": [], "y": [], "u": [], "rank": []}
    span = max(x1 - x0, y1 - y0)
    k = kmin
    while True:
        cs = s0 * 2 ** k
        lo_rank = 4.0 ** -(k + 1)
        if lo_rank >= keep and k > kmin:
            k += 1
            if cs > 4 * span:
                break
            continue
        ix0, ix1 = int(np.floor(x0 / cs)), int(np.floor(x1 / cs))
        iy0, iy1 = int(np.floor(y0 / cs)), int(np.floor(y1 / cs))
        n = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
        if n > 4_000_000:
            k += 1
            continue
        IX, IY = np.meshgrid(np.arange(ix0, ix1 + 1), np.arange(iy0, iy1 + 1))
        IX, IY = IX.ravel(), IY.ravel()
        sd = seed * 31 + k * 7 + 3
        ur = lat(IX, IY, sd)
        rank = lo_rank * (1 + 3 * ur)
        m = rank < keep
        if m.any():
            IX, IY, rank = IX[m], IY[m], rank[m]
            jx = lat(IX, IY, sd + 1)
            jy = lat(IX, IY, sd + 2)
            x = (IX + jx) * cs
            y = (IY + jy) * cs
            inb = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
            U = np.stack([lat(IX, IY, sd + 10 + j) for j in range(16)])
            out["x"].append(x[inb]); out["y"].append(y[inb])
            out["u"].append(U[:, inb]); out["rank"].append(rank[inb])
        if cs > 4 * span:
            break
        k += 1
    if not out["x"]:
        return None
    return {"x": np.concatenate(out["x"]), "y": np.concatenate(out["y"]),
            "u": np.concatenate(out["u"], axis=1), "rank": np.concatenate(out["rank"])}


def glyph_area(ell):
    """Approximate pixels one tuft covers at projected length ell (for coverage)."""
    ell = np.asarray(ell, float)
    nb = np.clip(ell / 3.0 + 0.5, 1, 6)
    L = np.maximum(1.0, 0.75 * ell)
    wid = 1.0 + 0.9 * np.clip((ell - 1.0) / 3.0, 0, 1)   # continuous: no step = no pop
    return nb * L * wid


def target_coverage(mp, ell):
    """Designed fraction of pixels covered by marks, as a smooth function of stroke size:
    near strands ~0.58 (Ferrari's lit bank leaves ~half its base showing), ticks ~0.4,
    thinning to a light peppering far off, where tone takes over."""
    ell = np.asarray(ell, float)
    # Ferrari's tarn bank: sparse holes at the crest, then a *dense band of short ticks*
    # (it reads as a darker mass), then strands that leave half the base showing
    return (mp["cov_far"] + (mp["cov_mid"] - mp["cov_far"]) * np.clip((ell - 0.4) / 1.2, 0, 1)
            + (mp["cov_near"] - mp["cov_mid"]) * np.clip((ell - 3.0) / 3.0, 0, 1))


def keep_for(world, mp, ell, ppm, s, ratio=None):
    """Fraction of tufts to keep so that marks cover target_coverage(ell) of the pixels.
    glyph_area is a guess; `ratio` (measured coverage / nominal, from calibrate) fixes it."""
    sp = SPECIES[world.species]
    rho = 1.0 / (sp[1] ** 2 * ppm ** 2 * max(s, 0.05))
    tgt = float(target_coverage(mp, ell))
    if ratio is not None:          # nominal guess (used only while calibrating)
        return float(min(1.0, tgt / (rho * glyph_area(ell))))
    from .calibrate import coverage_ratio
    a = coverage_ratio(world, mp, ell)      # measured rate at the calibration camera
    # the rate scales with tuft density on screen relative to the calibration (theta 30)
    rho_cal = rho * max(s, 0.05) / 0.5
    a = a * rho / rho_cal
    return float(min(1.0, -np.log(1 - min(tgt, 0.99)) / max(a, 1e-9)))


class MarkParams(dict):
    DEFAULT = dict(
        cov_far=0.08,       # mark coverage far off (light peppering over tone)
        cov_mid=0.62,       # ... when marks are ticks (ell ~1.6-3 px): a dense band
        cov_near=0.58,      # ... when marks are strands (ell >= 6 px)
        blade_div=1.6,      # one blade per this many px of ell
        max_blades=8,
        spread=0.05,
        lean_top=0.45,      # horizontal lean seen from above (top-down cameras)        # outward fan of blades (px per px)
        droop=0.25,         # tip bend
        wind_lean=0.35,     # px/px lean at full wind
        p_pair=0.35,         # chance a blade has a darker partner column
        p_gap=0.25,         # chance a blade has a dark gap column
        tip_boost=1.0,      # extra light at the tip in lit grass
        gap_depth=2.5,
        shadow_light=1.5,
        light_bias=1.05,
        dot_amp=1.0,
        dot_depth=2.2,
        detail=0.8,
        band_w=0.7,         # lit masses: flat bands, dither only in narrow seams
        band_w_shadow=0.0,  # shadow and mid masses: ordered checker everywhere (V16 mid bank)
        clumping=0.5,
        crown_light=1.2,
        silh_gap=1.0,      # metres of depth behind a blade that count as 'background'
        silh_len=1.8,
        cc_step=2.2,
        tone_lock=True,
        ell_ref=0.3,        # the design value V is what grass looks like far off: flat base
        lock=0.5,           # how much of the near strokes' darkening is compensated
    )

    def __init__(self, **kw):
        super().__init__(self.DEFAULT)
        self.update(kw)


def paint_tufts(canvas_v, canvas_m, prio, R, V, lit, world, pop, mp, fam_top, fam_bot):
    """Draw tufts far-to-near into canvas_v (int step) / canvas_m (material 0 green 1 straw).

    canvas_v: int array (H,W) step values, -1 = untouched
    prio: int array (H,W) for painter's order
    """
    cam = R.cam
    H, W = V.shape
    ppm, s, c = cam.ppm, cam.s, cam.c
    x, y, U = pop["x"], pop["y"], pop["u"]
    z = world.height(x, y, R.min_wl)
    ok = z > world.water_level + 0.03
    x, y, z, U = x[ok], y[ok], z[ok], U[:, ok]
    sx, sy = cam.to_screen(x, y, z)
    h = world.sward_height(x, y) * (0.6 + 0.8 * U[0])
    tus = world.tussockness(x, y)
    # projected blade length: the vertical part (cos theta) plus the lean seen from above,
    # so a top-down camera still sees tufts as small radiating marks
    ell = h * ppm * np.sqrt(c * c + (mp["lean_top"] * s ** 3) ** 2)
    dead = world.dead(x, y)
    # straw comes in patches (dry ridges, old growth), not as speckle through green
    straw = ((dead + 0.15 * (U[1] - 0.5)) > 0.55).astype(np.int8)
    # root pixel
    rx = np.floor(sx).astype(int)
    ry = np.floor(sy).astype(int)
    onscreen = (rx >= -8) & (rx < W + 8) & (ry >= 0) & (ry < H + int(ell.max() if len(ell) else 0) + 2)
    sel = onscreen
    x, y, z, U, sx, sy, h, ell, straw, rx, ry, tus = [a[..., sel] for a in (x, y, z, U, sx, sy, h, ell, straw, rx, ry, tus)]
    if len(x) == 0:
        return
    # tuft value: the design value at the root pixel (clamped onto the screen)
    cx = np.clip(rx, 0, W - 1)
    cy = np.clip(ry, 0, H - 1)
    v = V[cy, cx]
    cl = world.clump(x, y)
    # each tuft sits on an integer base step (stochastic rounding by its own number), so
    # its strokes are whole steps from that base: few colours per zone, as Ferrari's 4
    lt = lit[cy, cx]
    # each tuft stands on the *same integer base step the base layer shows at its root*,
    # so its strokes are whole steps away from what surrounds them (a stroke equal to the
    # local base would be invisible) and a zone keeps Ferrari's ~4 colours
    if getattr(R, "qbase", None) is not None:
        v = R.qbase[cy, cx].astype(float)
    else:
        v = np.floor(v + U[12])
    crown = mp["crown_light"] * (cl - 0.5) * lt      # taller clump crowns take more light strokes
    crest = np.clip((R.conv[cy, cx] - 0.35) * 3, 0, 1) if getattr(R, 'conv', None) is not None else np.zeros(len(x))
    edge = np.clip(R.conv[cy, cx] * 2.0 + R.graze[cy, cx] ** 2, 0, 1) if getattr(R, 'conv', None) is not None else np.full(len(x), 0.3)
    # detail at edges: at a distance marks crowd crests and thin in bodies
    # clumps thin the marks between them at a distance (drifts), barely at all up close
    kap = (0.35 + 1.3 * cl) ** (mp["clumping"] * (1 - 0.75 * np.clip((ell - 2.0) / 4.0, 0, 1)))
    if getattr(R, "conv", None) is not None and mp["detail"] > 0:
        cv = R.conv[cy, cx]
        gz = R.graze[cy, cx]
        k0 = np.clip(0.45 + 1.6 * np.maximum(cv, 0) + 0.5 * gz ** 2, 0.25, 1.8)
        strength = mp["detail"] * (1 - np.clip((ell - 3.0) / 4.0, 0, 1) * 0.7)
        kap = kap * (1 + strength * (k0 - 1))
    rank = pop["rank"][ok][sel]
    dsel = rank < pop.get("keep", 1.0) * kap
    x, y, z, U, sx, sy, h, ell, straw, rx, ry, tus, cx, cy, v, lt, crest, cl, edge, crown = [a[..., dsel] for a in
        (x, y, z, U, sx, sy, h, ell, straw, rx, ry, tus, cx, cy, v, lt, crest, cl, edge, crown)]
    if len(x) == 0:
        return
    # step 6 only on crests (sun-struck tips); everywhere else lit strokes top out at 5
    top = np.where(lt, np.minimum(fam_top[1], 5.0 + 1.0 * crest), fam_top[0])
    bot = np.where(lt, fam_bot[1], fam_bot[0])
    bot_f = np.where(lt, 3.0, 0.2)
    # depth order: far first
    order = np.argsort(-y, kind="stable")
    x, y, U, sx, sy, ell, straw, rx, ry, v, lt, top, bot, tus, bot_f, crest, edge, crown, cx, cy = [a[..., order] for a in
        (x, y, U, sx, sy, ell, straw, rx, ry, v, lt, top, bot, tus, bot_f, crest, edge, crown, cx, cy)]
    n = len(x)
    key0 = np.arange(n) * 8 + 8
    eps = 2.0 / (ppm * s)
    wind = world.wind * world.wind_dir
    sp = SPECIES[world.species]
    stiff = sp[4]
    # blade count: stochastic rounding against the tuft's own number
    # silhouette tufts: is there background (water, sky, or ground well behind) just
    # above the root?  Edge-on lip grass shows its full height, is longer (ungrazed
    # lips) and falls into fewer, curving, grouped blades: the fringe.
    probe = np.clip(ry - np.maximum(2, (0.6 * ell).astype(int)), 0, H - 1)
    bgd = R.depth[probe, cx]
    silh = ((bgd > y + mp["silh_gap"]) | (R.mat[probe, cx] != 1)) & (ell >= 2.5)
    ell = np.where(silh, ell * mp["silh_len"], ell)
    nb = np.clip(np.floor(ell / mp["blade_div"] * (1 + 0.8 * tus) * np.where(silh, 1.0, 1.0) + U[2]),
                 1, mp["max_blades"]).astype(int)
    wpx = SPECIES[world.species][2] * ppm * (1 + tus)
    side = -1 if (R.sun_screen_x < 0) else 1      # gap on the side away from the light... mirrored
    xs_all, ys_all, vs_all, ks_all, ms_all, ds_all = [], [], [], [], [], []
    rng_off = 0
    for j in range(mp["max_blades"]):
        has = nb > j
        if not has.any():
            break
        uj = [np.modf(U[3 + (j + q) % 9] * (7.13 + 3.7 * q + j))[0] for q in range(5)]
        L = np.floor(ell * (0.65 + 0.35 * uj[0]) + uj[1]).astype(int)
        L = np.where(has, np.maximum(L, 1), 0)      # a kept tuft always leaves at least a dot
        # middle distance: hold the vertical.  Between ell ~0.6 and 3 a mark is at least a
        # 1x2 tick (each blade decides at its own ell), so grass stays grass down to the
        # far bank instead of turning to isotropic sandpaper; below that, single dots.
        p_tick = np.clip((ell - 0.6) / 0.5, 0, 1)
        L = np.where(has & (uj[1] < p_tick), np.maximum(L, 2), L)
        rel = (j - (nb - 1) / 2.0) / np.maximum(nb, 1)
        x0 = sx + (uj[2] - 0.5) * np.minimum(wpx, 2 + ell * 0.6)
        # blade direction: world lean (radial fan + wind) projected to the screen
        psi = 2 * np.pi * np.modf(U[15] * (4.3 + 1.7 * j))[0]
        lam = mp["spread"] + mp["lean_top"] * s ** 3 * (0.6 + 0.8 * uj[2])
        lx = lam * np.cos(psi) * (1 - stiff * 0.5) + wind * mp["wind_lean"] * (1 - stiff * 0.5)
        ly = lam * np.sin(psi)
        vx, vy = lx, ly * s + c
        vy = np.where(np.abs(vy) < 1e-3, 1e-3, vy)
        big = np.maximum(np.abs(vx), np.abs(vy))
        dx, dy = vx / big, vy / big              # one pixel per step along the major axis
        a = dx
        b = (mp["droop"] * (1 - stiff) * np.sign(a + 1e-6) * (0.5 + uj[4])) * np.where(silh, 2.2, 1.0) * np.clip(c * 1.2, 0, 1)
        # fringe heights vary ~3:1
        L = np.where(silh & has, np.floor(ell * (0.35 + 0.65 * uj[0] ** 0.7) + uj[1]).astype(int), L)
        pair = uj[4] < mp["p_pair"]
        gap = uj[3] < mp["p_gap"]
        # stroke sign and depth from where the tuft's value sits in its family
        pos = np.clip((v - bot_f) / (top - bot_f), 0, 1)
        p_light = np.where(lt, np.clip(mp["light_bias"] - 1.3 * pos + 0.4 * crown, 0.06, 0.85), 0.9)
        ul = np.modf(U[13] * (5.31 + 2.1 * j))[0]
        light = ul < p_light
        ud = np.modf(U[14] * (3.77 + 1.3 * j))[0]
        # contrast inside the stroke grows with its size: at a distance the gaps between
        # blades are sub-pixel and merge, so deep darks are only allowed once strokes are
        # long enough; each blade crosses over at its own ell (staggered by ud)
        a2 = np.clip((ell - 2.0) / 3.0, 0, 1)
        a3 = np.clip((ell - 4.0) / 4.0, 0, 1)
        dark_off = np.where(ud < 1 - 0.6 * a2, -1.0, np.where(ud < 1 - 0.12 * a3, -2.0, -3.2))
        # dot depth: deep holes on crests, shallow ticks of the next step in bodies,
        # and shallower again as the dots shrink toward pure tone
        # (Ferrari's tarn bank, crest -> near: far off only the extremes survive, lit ochre
        # and black holes; ticks of the mid step appear once blades resolve; strands near)
        far_hole = np.clip((1.6 - ell) / 0.8, 0, 1) * np.clip((ell - 0.12) / 0.3, 0.35, 1)
        dotd = np.where(lt, 1.0 + (mp["dot_depth"] - 1.0) * np.maximum(far_hole, edge * 0.7), 1.0)
        # a single-pixel dot at a distance is a hole in the sward: the deepest dark survives
        off = np.where(light, np.where(lt, 1.0, mp["shadow_light"]), np.where(L <= 1, -dotd, dark_off))
        Lmax = int(L.max()) if L.size else 0
        for t in range(Lmax):
            m = L > t
            if not m.any():
                continue
            Lm = np.maximum(L, 1)
            fr = t / Lm
            xo = x0 + dx * t + b * t * t / np.maximum(Lm, 1)
            px = np.floor(xo).astype(int)
            py = np.floor(ry + 0.5 - dy * t).astype(int)
            # minority-stroke rule: a lit mass is painted with dark strokes on a light
            # base, a shadowed mass with light strokes on a dark base; in between, both.
            body = v + off
            main = np.where(fr > 0.8, body + light * mp["tip_boost"] * lt * crest, body)
            main = np.where((t == L - 1) & (L >= 3), v + off * 0.5, main)     # adjacent-step cap
            main = np.clip(main, 0.0, top)
            xs_all.append(px[m]); ys_all.append(py[m]); vs_all.append(main[m])
            ks_all.append(key0[m] + 2); ms_all.append(straw[m]); ds_all.append(y[m])
            # partner stem one step toward the base, beside the blade over its middle
            pm = m & pair & lt & (L >= 3) & (fr > 0.15) & (fr < 0.85)
            xs_all.append(px[pm] - side); ys_all.append(py[pm])
            vs_all.append(np.clip(v[pm] + off[pm] * 0.5, 0, top[pm]))
            ks_all.append(key0[pm] + 1); ms_all.append(straw[pm]); ds_all.append(y[pm])
            # dark gap beside a light blade, deep in the sward
            gm = m & gap & light & (L >= 3) & (fr < 0.7)
            xs_all.append(px[gm] + side); ys_all.append(py[gm]); vs_all.append(np.maximum(v[gm] - mp["gap_depth"], 0.0))
            ks_all.append(key0[gm]); ms_all.append(np.zeros(gm.sum(), np.int8)); ds_all.append(y[gm])
    if not xs_all:
        return
    X = np.concatenate(xs_all); Y = np.concatenate(ys_all)
    VV = np.concatenate(vs_all); K = np.concatenate(ks_all)
    M = np.concatenate(ms_all); D = np.concatenate(ds_all)
    inb = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
    X, Y, VV, K, M, D = X[inb], Y[inb], VV[inb], K[inb], M[inb], D[inb]
    # depth test against the terrain surface
    vis = D <= R.depth[Y, X] + eps
    X, Y, VV, K, M, D = X[vis], Y[vis], VV[vis], K[vis], M[vis], D[vis]
    # counterchange: a blade seen against something behind it (water, sky, far ground)
    # is pushed away from that background's value: dark against light, light against dark
    bgdep = R.depth[Y, X]
    against = (bgdep > D + mp["silh_gap"]) | (R.mat[Y, X] != 1)
    if against.any() and getattr(R, "bg_value", None) is not None:
        bv = R.bg_value[Y, X]
        vv = VV
        # against light (sky tint, lit ground) go dark; against dark (mirrored bank,
        # shadowed ground) go light
        push_dark = np.where(R.mat[Y, X] != 1, bv >= 3.0, bv >= vv - 0.5)
        # compress the blade's own range into the darks (keeps two values of structure)
        dark = np.minimum(np.minimum(vv, (vv - 3.0) * 0.5 + 1.6), bv - mp["cc_step"])
        newv = np.where(push_dark, dark, np.minimum(np.maximum(vv, bv + mp["cc_step"]), 5.0))
        VV = np.where(against, np.clip(newv, 0, 6), VV)
        # against the background only the blade's own 1-px line shows: partner and gap
        # columns (layers 0,1) would thicken the fringe into a hedge
        keepc = ~(against & ((K % 8) < 2))
        X, Y, VV, K, M = X[keepc], Y[keepc], VV[keepc], K[keepc], M[keepc]
    # family clamp using the tuft's own family (encoded by value range)
    np.maximum.at(prio, (Y, X), K)
    win = prio[Y, X] == K
    X, Y, VV, M, K = X[win], Y[win], VV[win], M[win], K[win]
    canvas_v[Y, X] = VV
    canvas_m[Y, X] = M


BAYER4 = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16.0


def quantize_base(Vd, X, Y, ppm, s, band_w, iy=None, ordered=True):
    """Base layer: flat bands of whole steps, dithered only across the seams.

    Seams use Ferrari's patterned (ordered 4x4) dither, anchored to a world-aligned pixel
    grid: column = floor(X * ppm), row = screen row offset by the camera's own pixel
    origin, so whole-pixel pans reproduce the pattern exactly.  (A checker has no scale,
    so zooming never pops it.)  ordered=False uses the world-noise threshold instead."""
    from .noise import stable_threshold
    lo = np.floor(Vd)
    f = Vd - lo
    band_w = np.asarray(band_w, float)
    g = np.clip((f - band_w / 2) / np.maximum(1 - band_w, 1e-3), 0, 1)
    if ordered and iy is not None:
        ix = np.floor(X * ppm + 1e-6).astype(np.int64)
        t = BAYER4[iy % 4, ix % 4]
    else:
        t = stable_threshold(X, Y * s, ppm, px=1.0)
    return (lo + (g > t)).astype(int)
