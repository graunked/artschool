"""Copy B: the river with a riffle and rocks in V11AM (Living Worlds, monoliths scene, morning),
native crop (370,335, 120x90). Ferrari's own palette indices; procedure + copyist's sketch.

Discovered procedure (read off his index map):
  STILL WATER  the sky ramp 7..19 (lilac -> blue, near-isovalue), climbing ~1 step per 7 rows toward
               the viewer, as clean ORDERED patterns per row (8888 / 8889 / 8989 / 9a9a ...).
  REFLECTION   the dark mesa's reflection has a wide 50% CHECKER edge (5-6 px across), whose water
  EDGES        pixels are colour-cycled darker blues (209..216): the soft edge is the shimmer.
  RIFFLE       a fan of thin strokes of cycled mid-blues (192..202, darker than the mirror) drawn on
               the checker lattice (every other pixel), rising from the rocks and leaning outward,
               dense at the foot, sparse at the tips: flow seen end-on at low angle.
  NEAR DASHES  near the viewer: long dotted rows (cycled index on every other pixel), consecutive
               cycle indices along a dash so a light runs along it when the palette turns.
  ROCKS        flat dark slabs, a lighter top row, warm maroon (70/82) on the sun-side face and in the
               checker below, a dark checker reflection hanging under them.
"""
import sys
import numpy as np
import pixkit as pk
from matplotlib.path import Path

X0, Y0, W, H = 370, 335, 120, 90
DARK = [92, 93, 94, 95, 69, 57, 54]


def poly(h, w, pts):
    yy, xx = np.mgrid[0:h, 0:w]
    return Path(pts).contains_points(np.c_[xx.ravel() + 0.5, yy.ravel() + 0.5]).reshape(h, w)


def sketch():
    mesa = [(-1, -1), (52, -1), (30, 2), (14, 5), (4, 8), (8, 13), (14, 16), (26, 17), (26, 22), (12, 24),
            (4, 27), (-1, 29)]
    bank_r = [(121, 13), (104, 15), (92, 17), (86, 19), (90, 22), (104, 28), (116, 33), (121, 34)]
    rocks = [[(0, 47), (18, 46), (22, 49), (22, 57), (12, 60), (0, 61)],
             [(47, 63), (70, 62), (76, 66), (76, 74), (64, 76), (48, 73)],
             [(35, 69), (42, 68), (45, 70), (40, 72), (34, 71)],
             [(17, 57), (30, 56), (33, 59), (18, 60)]]
    bank_n = [(62, 91), (72, 84), (90, 76), (106, 70), (121, 67), (121, 91)]
    body = [(0, 44), (14, 42), (30, 40), (44, 43), (56, 41), (66, 44), (78, 46), (90, 52), (98, 58), (100, 64),
            (90, 64), (76, 70), (60, 66), (44, 63), (30, 60), (18, 58), (0, 60)]
    fan = dict(cx=58, body=body, n=70, maxlen=16)
    return mesa, bank_r, rocks, bank_n, fan


def fringe(mask, width_px, par):
    """pixels within width_px (horizontal) of the mask, outside it."""
    h, w = mask.shape
    out = np.zeros_like(mask)
    for k in range(1, width_px + 1):
        out[:, :-k] |= mask[:, k:]
        out[:, k:] |= mask[:, :-k]
    return out & ~mask


def traced_masses():
    """master-copy allowance: his solid dark masses, traced (cores only; edges are procedure)."""
    from scipy.ndimage import uniform_filter, label
    d = pk.lw_load('V11AM')
    idx = d['idx'][Y0:Y0 + H, X0:X0 + W]
    dark = np.isin(idx, list(range(53, 74)) + [82] + list(range(92, 97)))
    core = (uniform_filter(dark.astype(float), 3) > 0.62) & dark
    lab, n = label(core)
    parts = {'mesa': np.zeros_like(core), 'bank_r': np.zeros_like(core), 'bank_n': np.zeros_like(core), 'rocks': []}
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() < 6:
            continue
        ys, xs = np.nonzero(m)
        if ys.min() == 0 and xs.min() == 0 or (xs.mean() < 40 and ys.mean() < 32):
            parts['mesa'] |= m
        elif xs.mean() > 80 and ys.mean() < 40:
            parts['bank_r'] |= m
        elif ys.max() >= H - 2 and xs.mean() > 50:
            parts['bank_n'] |= m
        else:
            parts['rocks'].append(m)
    return parts


def paint_rock(cv, R, rng, par):
    h, w = R.shape
    ys, xs = np.nonzero(R)
    x0, x1 = xs.min(), xs.max()
    split = x0 + 0.55 * (x1 - x0)
    yy, xx = np.mgrid[0:h, 0:w]
    lit = R & (xx < split)
    shd = R & (xx >= split)
    cv[lit] = 69
    strips = lit & (xx > split - 4) & ((xx % 2) == 0)
    cv[strips] = 70
    cv[lit & (xx < x0 + 2) & (rng.random((h, w)) < 0.5)] = 70
    cv[shd] = np.where(par, 95, 96)[shd]
    # top plane: first 2-3 rows of each column, value-close lighter
    for x in range(x0, x1 + 1):
        col = np.nonzero(R[:, x])[0]
        if len(col) == 0:
            continue
        t = col.min(); n = 2 if (x1 - x0) < 14 else 3
        for k in range(min(n, len(col) - 1)):
            cv[t + k, x] = rng.choice([56, 58, 57, 53] if k == 0 else [56, 58, 60])
    # reflection: checker of the face colour under the rock, fading over 3 rows
    for x in range(x0, x1 + 1):
        col = np.nonzero(R[:, x])[0]
        if len(col) == 0:
            continue
        b = col.max()
        for k in range(1, 4):
            y = b + k
            if y < h and not R[y, x] and par[y, x] and rng.random() < (0.95, 0.7, 0.35)[k - 1]:
                cv[y, x] = 70 if x < split else rng.choice([92, 95, 71])


def paint(seed=1):
    rng = np.random.default_rng(seed)
    h, w = H, W
    yy, xx = np.mgrid[0:h, 0:w]
    y = yy + Y0
    par = (xx + yy) % 2 == 0
    P = traced_masses()
    v = np.interp(y, [338, 344, 350, 355, 358, 364, 369, 374, 380, 386, 393, 399, 405, 410, 416, 424],
                  [8.0, 9.0, 10.0, 10.5, 11.0, 12.0, 12.5, 13.0, 14.0, 14.5, 15.0, 16.0, 16.5, 17.0, 18.0, 18.5])
    cv = pk.ordered(pk.banded(v, 0.6), pk.BAYER4)
    from scipy.ndimage import distance_transform_edt as edt
    M = P['mesa']
    dm = edt(~M, sampling=(3.0, 1.0))
    cv[M] = rng.choice(DARK[:4], M.sum())
    t4 = pk.tile(pk.BAYER2, h, w)
    for lo, hi, thr in ((0, 5, 0.8), (5, 10, 0.5), (10, 16, 0.22)):
        g = (dm > lo) & (dm <= hi)
        dsel = g & (t4 < thr)
        cv[dsel] = rng.choice(DARK[:3], dsel.sum())
        csel = g & ~dsel & (rng.random((h, w)) < 0.7)
        cv[csel] = rng.integers(209, 217, csel.sum())
    B = P['bank_r']
    warm = pk.value_noise(h, w, 5, rng, 2) > 0.5
    cv[B] = np.where(par & warm, 82, np.where(warm & (rng.random((h, w)) < 0.4), 70, 92))[B]
    be = fringe(B, 3, par) & (yy > 17)
    cv[be & par & (rng.random((h, w)) < 0.7)] = 82
    rocks_all = np.zeros((h, w), bool)
    for R in P['rocks']:
        rocks_all |= R
    blocked = M | B | rocks_all | P['bank_n']
    # riffle: a few distinct JETS rising from a base band, long leaning chains from their tops,
    # plain mirror between them
    base_y = 44
    jets = [(8, 20), (22, 26), (36, 17), (52, 24), (63, 30), (74, 22), (88, 16), (98, 12)]
    for (jx, jh) in jets:
        jx = jx + rng.normal(0, 1.5)
        jw = rng.uniform(2.5, 5.0)
        top = base_y + 12 - jh
        for yr in range(int(top), base_y + 14):
            f = (yr - top) / max(jh, 1)                        # 0 at the tip .. 1 at the base
            half = jw * (0.35 + 0.65 * f)
            for xr in range(int(jx - half - 1), int(jx + half + 2)):
                if not (0 <= xr < w and 0 <= yr < h) or blocked[yr, xr]:
                    continue
                dx = abs(xr - jx) / max(half, 0.5)
                if dx > 1:
                    continue
                dens = f * (1 - dx * 0.6)
                if dens > 0.55 or (par[yr, xr] and rng.random() < dens * 1.6):
                    cv[yr, xr] = 192 + int(np.clip(rng.normal(5 + 3 * (0.5 - dx), 2), 0, 10))
        # chains from the jet's crown, leaning outward, long
        for c_ in range(int(rng.integers(3, 7))):
            side = rng.choice([-1, 1])
            lean = rng.uniform(0.05, 0.45) * side; curl = rng.uniform(0.01, 0.07) * side
            xf = jx + rng.normal(0, 1.5); y0 = int(top + rng.integers(0, 6)); L = int(rng.integers(4, 15))
            k0 = int(rng.integers(0, 11))
            for st in range(L):
                yi = y0 - st; xi = int(round(xf))
                if (xi + yi) % 2:
                    xi += 1 if xf > xi else -1
                if 0 <= yi < h and 0 <= xi < w and not blocked[yi, xi] and rng.random() < 0.9:
                    cv[yi, xi] = 192 + (k0 + st) % 11
                xf += lean + curl * st
    # the base band between the rocks: dense strands, then dark trough runs under it
    band = (yy >= base_y + 6) & (yy <= base_y + 14) & ~blocked
    bn = pk.value_noise(h, w, 3, rng, 2, aniso=(3.0, 0.6))
    selb = band & ((bn > 0.45) | (par & (bn > 0.3)))
    cv[selb] = 192 + np.clip((bn * 11 + rng.normal(0, 1.5, (h, w))).astype(int), 0, 10)[selb]
    lil = selb & (rng.random((h, w)) < 0.05)
    cv[lil] = rng.choice([219, 220, 221], lil.sum())
    for yb in range(base_y + 15, base_y + 19):
        x = 0
        while x < w:
            if rng.random() < 0.18:
                L = int(rng.integers(3, 10))
                for xi in range(x, min(w, x + L)):
                    if not blocked[yb, xi]:
                        cv[yb, xi] = rng.choice([209, 210, 215, 216, 223])
                x += L + int(rng.integers(3, 9))
            else:
                x += 1
    # near dotted dashes
    for yd in range(62, h):
        pr = np.interp(yd, [62, 66, 70, 78, 90], [0.0, 0.5, 0.3, 0.45, 0.35])
        x = int(rng.integers(0, 6))
        while x < w:
            if rng.random() < pr * 0.3:
                L = int(rng.integers(3, 12)); k0 = int(rng.integers(0, 11))
                ramp = (192, 11) if yd < 79 else (209, 8)
                for s_ in range(L):
                    xi = x + 2 * s_; xi += (xi + yd) % 2
                    if xi < w and not blocked[yd, xi]:
                        cv[yd, xi] = ramp[0] + (k0 - s_) % ramp[1]
                x += 2 * L + int(rng.integers(2, 10))
            else:
                x += int(rng.integers(2, 8))
    for R in P['rocks']:
        paint_rock(cv, R, rng, par)
    N = P['bank_n']
    cv[N] = np.where(par & (pk.value_noise(h, w, 4, rng, 2) > 0.62), 82, rng.choice([92, 93, 94, 95], (h, w)))[N]
    ne = fringe(N, 2, par)
    cv[ne & par & (rng.random((h, w)) < 0.7)] = rng.choice([92, 82], (ne & par & (rng.random((h, w)) < 2)).sum()) if False else 92
    return cv


if __name__ == '__main__':
    d = pk.lw_load('V11AM')
    ref_idx = d['idx'][Y0:Y0 + H, X0:X0 + W]
    pal = d['pal']
    cv = paint(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    ref = pk.rgb(ref_idx, pal); cp = pk.rgb(cv, pal)
    tag = sys.argv[2] if len(sys.argv) > 2 else 'v1'
    pk.compare_sheet(ref, cp, 'study/c0b_%s.png' % tag, [(40, 20, 60, 40), (0, 50, 60, 40)], label=tag)
    print(pk.metrics(cp, ref))
