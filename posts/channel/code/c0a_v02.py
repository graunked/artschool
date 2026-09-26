"""Copy A: the winding low-angle ribbons of V02 (Living Worlds, 'Mountain Fortress Dusk'), native crop
(0,282, 92x82). Painted in Ferrari's own palette indices, by procedure, from a copyist's sketch
(ribbon centre-lines + widths, a few hand values). See LESSONS.md, stage 0.

Discovered procedure (read off his index map):
  WATER FILL  the water is the SKY RAMP (indices 1..15, horizon->zenith), and the index CLIMBS with
              nearness: far threads 2/3, mid 4/5, 6/7, near 7/8 -- nearer water mirrors higher sky.
              Steps hold flat, crossing as sustained 50% checkers (4545 over several rows).
              Hot glints: small horizontal clusters (1-4 px) of 42/43 (bright orange), sparse.
              Near ribbon: a checker of COLOUR-CYCLED peach (128..135) against static 7/8/3/40: glitter.
  SHAPE       at low angle the ribbon is horizontal runs; a far thread is ONE row, stepping a row at a
              time; turns are staircases; thickness in rows grows toward the viewer.
  EDGES       near side (land below / left of water on a down-going edge): LIGHT lip -- 73 runs directly
              under water, and on steep diagonals a 75/water checker, 73, then warm 51/52.
              far side (land above water): DARK -- runs of 80 (bank face + its dark reflection); on
              steep diagonals a water/dark checker fringe (dark pixels are cycled 233..239).
  MARSH       banded ordered ramp 73->75 far (flat rows + 50% seams), 76-78 mid with 2x2 order,
              79/80(/81) near, checkers that skip a step (79/81), + sparse single light pixels next to
              dark ones (contrast pairs) + warm 51/48 specks growing toward the viewer.
"""
import sys
import numpy as np
import pixkit as pk

X0, Y0, W, H = 0, 282, 72, 82


def sketch():
    """copyist's sketch: polylines (x, y) in crop pixels with widths (stretched units, ky)."""
    thread = [(95, 1.5), (72, 2.2), (40, 6.6), (40, 8.0), (55, 9.2), (55, 10.6), (22, 11.2), (4, 12.4),
              (8, 15.4), (0, 17.3)]
    far = [(96, 21.3), (84, 22.3), (64, 24.2), (46, 26.0), (35, 27.3)]
    prong_a = [(35, 27.5), (37, 30.5)]
    prong_b = [(44, 27.6), (46, 30.6)]
    left = [(47, 31.3), (26, 31.6), (-4, 31.2)]
    main = [(37, 32.2), (22, 34.2), (26, 36.6), (26, 38.2), (16, 40.6), (15, 42.6), (20, 44.8), (30, 47.8),
            (40, 50.6), (50, 53.4), (52, 55.0), (48, 57.4), (35, 60.2), (20, 64.0), (6, 68.0), (-8, 73.0)]
    return [(thread, [3.0] * len(thread)),
            (far, [7, 8, 9, 10, 10]),
            (prong_a, [5, 5]), (prong_b, [5, 5]),
            (left, [10, 11, 11]),
            (main, [8, 9, 9, 9, 10, 11, 12, 13, 14, 16, 17, 18, 22, 26, 30, 34])]


KY = 4.0


def fields(h, w):
    """F: signed distance (stretched units, <0 inside) to the ribbons; thread: 1-row centre-line raster."""
    F = np.full((h, w), 1e9)
    thread = np.zeros((h, w), bool)
    for k, (pts, wid) in enumerate(sketch()):
        if k == 0:
            for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
                n = int(max(abs(bx - ax), abs(by - ay)) * 2) + 2
                for t in np.linspace(0, 1, n):
                    x, y = int(round(ax + t * (bx - ax))), int(np.floor(ay + t * (by - ay)))
                    if 0 <= x < w and 0 <= y < h:
                        thread[y, x] = True
            continue
        F = np.minimum(F, pk.polyline_field(h, w, pts, wid, ky=KY))
    return F, thread


def marsh(h, w, rng, F, gy, steep):
    yy, xx = np.mgrid[0:h, 0:w]
    y = yy + Y0
    # base step along the ramp 73..81 (copyist's value study, by row)
    v = np.interp(y, [282, 286, 292, 298, 302, 306, 312, 318, 324, 332, 340, 364],
                  [73.0, 73.0, 74.0, 75.0, 75.5, 76.0, 76.8, 77.6, 78.4, 78.9, 79.3, 79.6])
    v = v + (pk.value_noise(h, w, 10, rng, 2, aniso=(0.35, 2.5)) - 0.5) * np.clip((y - 300) / 30, 0, 1) * 1.2
    # banks: the far side darkens over ~3 rows toward the water, the near side has a lit lip then a trough
    near_mid = np.clip((y - 300) / 12.0, 0, 1)
    db, da = pk.col_dist(F < 0)
    sc = np.clip((y - 276) / 45.0, 0.4, 2.0)   # bank faces project taller toward the viewer
    v = v + near_mid * np.interp(db / sc, [0, 1, 2, 3, 4, 6, 8], [1.6, 1.6, 1.9, 1.6, 1.1, 0.4, 0.0])
    v = v + near_mid * (da == 2) * (db > 3) * -0.8
    far = y < 300
    out = np.zeros((h, w), int)
    out[far] = pk.ordered(pk.banded(v, 0.55))[far]
    amp = np.clip((y - 304) / 40.0, 0, 1)
    o = pk.ordered(v + (rng.random((h, w)) - 0.5) * (0.1 + 0.9 * amp ** 2), pk.BAYER4)
    skip = (y > 336) & (o == 80) & (rng.random((h, w)) < 0.45)
    o = np.where(skip, np.where((xx + yy) % 2 == 0, 79, 81), o)
    out[~far] = o[~far]
    clump = (pk.value_noise(h, w, 5, rng, 2) > 0.68) & (~far) & (F > 8) & ((xx + yy) % 2 == 0)
    out[clump] = np.maximum(out[clump] - 2, 75)
    lp = (~far) & (rng.random((h, w)) < 0.025 * (0.5 + amp)) & (F > 6)
    out[lp] = 75
    ds = (y > 335) & (rng.random((h, w)) < 0.04 * amp)
    out[ds] = 82
    ws = (y > 328) & (rng.random((h, w)) < 0.06 * amp) & (F > 6)
    out[ws] = np.where(rng.random((h, w)) < 0.8, 51, 48)[ws]
    return np.clip(out, 48, 84)


def water_fill(mask, rng, F=None):
    F = np.full(mask.shape, -99.0) if F is None else F
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    y = yy + Y0
    v = np.interp(y, [283, 290, 296, 304, 309, 316, 320, 324, 329, 334, 340], [2.3, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.6])
    v = v + (pk.value_noise(h, w, 8, rng, 1, aniso=(0.4, 2.0)) - 0.5) * 0.5
    base = pk.ordered(pk.banded(v, 0.7))
    out = np.where(mask, base, 0)
    g = mask & (y > 300) & (y < 336)
    dens = np.where(y < 312, 0.06, 0.01)
    seeds = g & (rng.random((h, w)) < dens)
    for sy, sx in zip(*np.nonzero(seeds)):
        L = int(rng.integers(1, 4))
        c = 43 if rng.random() < 0.5 else 42
        for k in range(L):
            if sx + k < w and mask[sy, sx + k]:
                out[sy, sx + k] = c if k % 2 == 0 or rng.random() < 0.5 else 40 + int(rng.integers(0, 2))
    near = mask & (y >= 337) & (F < -KY * 0.9)
    par = (xx + yy) % 2 == 0
    cyc = rng.choice([128, 129, 130, 131, 131, 132, 133, 134, 135], (h, w))
    pink = rng.integers(160, 168, (h, w))
    cyc = np.where((y > 346) & (rng.random((h, w)) < 0.2), pink, cyc)
    stat = rng.choice([7, 8, 8, 7, 3, 40, 40, 41, 4, 36], (h, w))
    out = np.where(near & par, cyc, out)
    out = np.where(near & ~par, stat, out)
    out = np.where(near & (y > 347) & ~par & (rng.random((h, w)) < 0.5), cyc, out)
    return out


def edges(cv, F, gy, steep, rng):
    h, w = F.shape
    yy, xx = np.mgrid[0:h, 0:w]
    y = yy + Y0
    par = (xx + yy) % 2 == 0
    out = cv.copy()
    mid = y >= 300
    nearish = y >= 312
    # near side, land: the lit lip (1 row on flats, a horizontal run on diagonals)
    lip = mid & (gy >= 0) & (F >= 0) & (F < KY)
    lipc = np.where(rng.random((h, w)) < 0.6, 73, np.where(rng.random((h, w)) < 0.5, 74, 75))
    out[lip] = np.where(nearish, lipc, np.where(par, 74, 75))[lip]
    nl = lip & (y > 336) & steep
    out[nl] = np.where(rng.random(nl.sum()) < 0.7, 75, 73)
    # near side, water, steep diagonal: water/75 checker
    wl = (y > 336) & (gy >= 0) & (F < 0) & (F > -KY) & steep & par
    out[wl] = 75
    # warm band beyond the lip on the steep near edge of the near ribbon
    warm = (y > 336) & (gy >= 0) & (F >= KY) & (F < 3 * KY) & steep
    wc = np.where(F < 2.2 * KY, np.where(rng.random((h, w)) < 0.8, 51, 79), np.where(rng.random((h, w)) < 0.6, 52, 51))
    out[warm] = wc[warm]
    # far side, land: the dark bank face
    dk = mid & (gy < 0) & (F >= 0) & (F < KY)
    out[dk] = np.where(nearish, np.where(steep | (rng.random((h, w)) < 0.6), 80, 79), np.where(par, 78, 76))[dk]
    # far side, water, steep: checker fringe of dark (cycled darks 233..239)
    fr = (y > 334) & (gy < 0) & (F < 0) & (F > -KY) & steep & par
    out[fr] = rng.choice([80, 233, 234, 237, 239], fr.sum())
    return out


def breathe(F, rng):
    """the banks bite into the water in stepped notches: blocky noise (held flat over 2-4 px runs)
    added to the field, stronger toward the viewer."""
    h, w = F.shape
    yy, xx = np.mgrid[0:h, 0:w]
    y = yy + Y0
    n = pk.value_noise(h, w, 6, rng, 2, aniso=(0.5, 1.5))
    # hold the noise flat along runs of 2-4 px so every bite is a clean stepped notch
    blk = np.zeros((h, w))
    for r in range(h):
        x = 0
        while x < w:
            L = int(rng.integers(2, 5))
            blk[r, x:x + L] = n[r, x]
            x += L
    amp = np.interp(y, [300, 316, 334, 340, 364], [1.0, 2.0, 2.5, 4.5, 6.0])
    return F + (blk - 0.5) * 2 * amp


def traced_fields():
    """the master-copy allowance: take his water SHAPE exactly (so the test is the procedure)."""
    from scipy.ndimage import distance_transform_edt as edt, gaussian_filter
    d = pk.lw_load('V02')
    idx = d['idx'][Y0:Y0 + H, X0:X0 + W]
    wset = list(range(1, 48)) + list(range(128, 136)) + list(range(160, 168)) + list(range(232, 240))
    m = np.isin(idx, wset)
    thread = m & (np.arange(H)[:, None] < 18)
    body = m & ~thread
    F = edt(~body, sampling=(KY, 1.0)) - edt(body, sampling=(KY, 1.0))
    return F, thread


TRACED = True


def paint(seed=3):
    rng = np.random.default_rng(seed)
    if TRACED:
        F, thread = traced_fields()
        from scipy.ndimage import gaussian_filter
        gy, gx = np.gradient(gaussian_filter(F, 1.2))
    else:
        F, thread = fields(H, W)
        gy, gx = np.gradient(F)            # facing is taken from the clean field
        F = breathe(F, rng)
    steep = np.abs(gy) <= 5 * np.abs(gx)      # edge runs of <= 5 px per row: a diagonal
    m = (F < 0) | thread
    if TRACED:
        m = (F <= 0) | thread
    cv = marsh(H, W, rng, F, gy, steep)
    cv = np.where(m, water_fill(m, rng, F), cv)
    cv = edges(cv, F, gy, steep, rng)
    yy, xx = np.mgrid[0:H, 0:W]; y = yy + Y0; par = (xx + yy) % 2 == 0
    zone = (y > 334) & (gy >= 0)
    spill = zone & ~m & (F < 1.5 * KY) & (rng.random((H, W)) < 0.07)
    cv[spill] = rng.choice([131, 134, 162, 36, 40, 51], spill.sum())
    cv[thread] = water_fill(thread, rng)[thread]
    return cv, m


if __name__ == '__main__':
    d = pk.lw_load('V02')
    ref_idx = d['idx'][Y0:Y0 + H, X0:X0 + W]
    pal = d['pal']
    cv, m = paint(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
    ref = pk.rgb(ref_idx, pal); cp = pk.rgb(cv, pal)
    tag = sys.argv[2] if len(sys.argv) > 2 else 'v1'
    pk.compare_sheet(ref, cp, 'study/c0a_%s.png' % tag, [(0, 34, 60, 40), (12, 0, 60, 30)], label=tag)
    W_set = set(list(range(1, 48)) + list(range(128, 136)) + list(range(160, 168)))
    rm = np.isin(ref_idx, list(W_set))
    print(pk.metrics(cp, ref), 'water IoU %.2f' % ((rm & m).sum() / (rm | m).sum()))
    print('ref  edge profile', pk.edge_profile(pk.luma(ref_idx, pal), rm, 18))
    print('copy edge profile', pk.edge_profile(pk.luma(cv, pal), m, 18))
