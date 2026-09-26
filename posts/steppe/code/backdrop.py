"""backdrop: the coulee wall (tiers of columnar basalt over talus) and far hills, for the horizon
of a standing view.

Read from the Moses Coulee and Hungate Canyon photos, and from Ferrari's V11AM mesa (the closest
passage he painted: a flat-topped cliff with vertical fluting over a striated apron):
- the wall is horizontal TIERS: each tier a row of vertical columns (colonnade) or a band of
  hackly blocks (entablature), with a lit face and a shadow face per column;
- between the tiers are narrow benches, sage-dotted; below the lowest tier a talus apron fans out
  along its fall lines, grey-brown, with scattered blocks;
- the top is dead flat for long runs (a lava flow top), broken by a few notches;
- with distance the whole wall compresses to two or three values and goes violet (the haze).
"""
import numpy as np
import sp
from sp import BASALT, FAR, SAGE, GROUND


def ridge_profile(W, rng, base, amp, flat=0.7, notch=0.03):
    """A flow-top skyline: long flat runs with small steps and a few notches (row per column)."""
    y = np.zeros(W)
    cur = base
    x = 0
    while x < W:
        run = int(rng.integers(12, 80)) if rng.random() < flat else int(rng.integers(3, 12))
        cur = np.clip(cur + rng.normal(0, amp * 0.25) * (1 if run < 12 else 0.4), base - amp, base + amp)
        y[x:x + run] = round(cur)
        x += run
    # notches: small V gullies
    for _ in range(int(W * notch / 10) + 1):
        c = int(rng.integers(0, W))
        wdt = int(rng.integers(3, 9))
        dep = int(rng.integers(2, 6))
        for k in range(-wdt, wdt + 1):
            if 0 <= c + k < W:
                y[c + k] = max(y[c + k], y[c + k] + dep * (1 - abs(k) / (wdt + 1)))
    return y.astype(int)


def paint_rim(cv, y_top, y_base, band=2, sun_dx=1, seed=0, tiers=3, sage_bench=True):
    """Paint a basalt coulee wall across the canvas between rows y_top (skyline) and y_base (the
    foot of the talus). Big form first: the wall wanders in and out (headlands and bays), so it
    is a sequence of lit and shadowed planes; columns and cracks are drawn only where they read
    (strong in the light, faint in the shadow). Only overwrites SKY pixels."""
    rng = np.random.default_rng(seed)
    W = cv.w
    H = y_base - y_top
    top = ridge_profile(W, rng, y_top, max(2, H * 0.05))
    # facing: -1 faces left .. +1 faces right; lit = facing toward the sun side
    face = (sp.value_noise((1, W), rng.uniform(25, 50), rng, octaves=2)[0] - 0.5) * 2.2
    lit = np.clip(0.62 - face * sun_dx * 0.55, 0.05, 1)
    talus_frac = rng.uniform(0.3, 0.42)
    cliff_h = (1 - talus_frac) * H
    edges = np.sort(rng.uniform(0.25, 0.8, tiers - 1))
    edges = np.concatenate([[0], edges, [1.0]])
    kinds = ["colonnade" if rng.random() < 0.6 else "entab" for _ in range(tiers)]
    colw = [int(rng.integers(2, 4)) for _ in range(tiers)]
    foot = y_base + np.round((sp.value_noise((1, W), 18, rng)[0]) * 3).astype(int)   # never above the base: no sky gaps
    vn = sp.value_noise((max(8, H + 4), W), 6, rng)
    flutes = []
    for ti in range(tiers):
        d = {}
        x = int(rng.integers(0, 3))
        while x < W:
            ln = rng.uniform(0.3, 1.0)
            d[x] = (1, ln)
            d[x - sun_dx] = (-1, ln * rng.uniform(0.6, 1.0))
            x += int(rng.integers(2, 6))
        flutes.append(d)
    # talus cones under the gullies (where the skyline dips)
    dips = np.nonzero(top > np.median(top) + 1)[0]
    cones = list(rng.choice(dips, size=min(len(dips), 4), replace=False)) if len(dips) else []
    for x in range(W):
        y0 = top[x]
        yc = int(y_top + cliff_h)
        for y in range(y0, min(cv.h, foot[x] + 1)):
            # headland edges run slantwise down the wall, and each plane varies a little
            L = float(np.clip(lit[int(np.clip(x + (y - y0) * 0.45 * sun_dx, 0, W - 1))]
                              + (vn[y % vn.shape[0], x] - 0.5) * 0.35, 0, 1))
            if cv.mat[y, x] not in (sp.SKY, 0):
                continue
            if y < yc:
                d = (y - y0) / max(1, yc - y0)
                ti = min(max(int(np.searchsorted(edges, d, side="right") - 1), 0), tiers - 1)
                td = (d - edges[ti]) * (yc - y0)                  # px below this tier's top
                tl = (edges[ti + 1] - edges[ti]) * (yc - y0)     # tier height px
                base = 0.8 + 2.6 * L
                if td < 1.0 and ti > 0:
                    base += 1.0                                   # the bench: a lit flat top
                    if sage_bench and rng.random() < 0.3:
                        cv.px(x, y, SAGE, 2, band)
                        continue
                if td > tl - 2:
                    base -= 1.2                                   # shade under the next ledge
                if kinds[ti] == "colonnade":
                    # V11AM's fluting: irregular vertical strokes of varying length, a light
                    # stroke paired with a dark one on its shadow side, strong only in the light
                    fl = flutes[ti].get(x)
                    if fl is not None and td < fl[1] * tl:
                        base += (0.9 if fl[0] > 0 else -0.9) * (0.35 + 0.65 * L)
                else:
                    if ((x * 3 + y * 7 + ti * 11) % 13) == 0:
                        base -= 1.0
                    if ((x + 2 * y) % 9) == 0:
                        base += 0.5 * L
                s = int(np.clip(np.floor(base + rng.random() * 0.6), 0, 4))
            else:
                t = (y - yc) / max(1, foot[x] - yc)
                # nearest cone brightens its sun side
                cone = 0.0
                for c in cones:
                    dx = (x - c) / max(3.0, (y - yc) * 1.2 + 3)
                    if abs(dx) < 1:
                        cone = max(cone, (1 - abs(dx)) * (0.6 if dx * sun_dx < 0 else -0.3))
                base = 2.1 + 0.9 * L * (1 - t) + cone + 0.2 * t
                streak = ((x * 2 + (y - yc)) % 7 == 0) or ((x * 2 - (y - yc)) % 9 == 0)
                if streak:
                    base -= 0.6
                if t < 0.12:
                    base -= 0.8                                   # the cliff foot's shadow
                if rng.random() < 0.008:
                    base += 1.2
                s = int(np.clip(np.floor(base + 0.25), 0, 4))
            cv.px(x, y, BASALT, s, band)


def paint_far_hills(cv, y_line, amp, band=4, seed=0):
    """Far hills: a soft silhouette in the FAR material, two values, above the horizon line."""
    rng = np.random.default_rng(seed)
    W = cv.w
    prof = (sp.value_noise((1, W), 60, rng, octaves=2)[0] - 0.5) * 2 * amp
    for x in range(W):
        yt = int(round(y_line - amp - prof[x]))
        for y in range(max(0, yt), y_line + 1):
            if cv.mat[y, x] == sp.SKY:
                cv.px(x, y, FAR, 1 if (y - yt) < 2 else 0, band)
