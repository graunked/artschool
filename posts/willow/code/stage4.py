"""Stage 4: clumps and bands. Several bushes nestled and overlapping, separated by value; a band
of scrub along a bank; the same willow from close to far, down to a few pixels."""
import numpy as np, os, sys
from wpaint import *
from willow import paint_bush, band_palette, LEAFS, BAND
from palette import willow_palette
OUT = "stage4"; os.makedirs(OUT, exist_ok=True)
L0 = sun(25, 40)


def sky_ground(w, h, gy, sky_steps=True):
    cv = Canvas(w, h)
    chk = (np.indices((1, w))[1][0] % 2)
    for y in range(gy):
        t = 3.999 * y / max(gy, 1); f = t - int(t)
        base = int(t)
        # dither only at the seams (sky study): a checker row where a band hands over to the next
        row = np.full(w, base)
        # three seam rows: 25%, 50%, 75% of the next band (ordered)
        xs = np.arange(w)
        if f > 0.72:
            k = min(3, int((f - 0.72) / 0.28 * 3) + 1)
            bay = np.array([[0, 2], [3, 1]])[y % 2][xs % 2] if k != 1 else ((xs + 2 * (y % 2)) % 4 == 0) * 4
            thr = {1: 1, 2: 2, 3: 3}[k]
            row = base + ((bay < thr) if k != 1 else (bay > 0))
        cv.mat[y] = SKY; cv.step[y] = np.minimum(row, 3)
    cv.mat[gy:] = GRAVEL; cv.step[gy:] = 4
    return cv


def lod_strip(season="summer", hour="noon", seed=3):
    widths = [150, 100, 64, 40, 24, 14, 8, 5, 3]
    w = sum(int(x * 1.08) + 6 for x in widths) + 10; h = 130; gy = 120
    cv = sky_ground(w, h, gy)
    rng = np.random.default_rng(seed); x = 8
    for W_ in widths:
        paint_bush(cv, x + W_ * 0.54, gy, W_, W_ * 0.74, L0, rng)
        x += int(W_ * 1.08) + 6
    return cv


def clump(seed=5, n=5, w=260, h=150, gy=140):
    """Bushes nestled: big ones behind, small ones in front at their feet; each later bush's crest
    lies against the darkened foliage of the one behind."""
    cv = sky_ground(w, h, gy - 30)
    rng = np.random.default_rng(seed)
    specs = []
    for i in range(n):
        W_ = rng.uniform(60, 120) * (1 - 0.12 * i)
        specs.append((gy - rng.uniform(0, 10) * (n - i) / n, rng.uniform(0.2, 0.8) * w, W_))
    specs.sort(key=lambda s: s[0])      # farther (higher) first
    for (g, cx, W_) in specs:
        paint_bush(cv, cx, int(g), W_, W_ * rng.uniform(0.62, 0.8), L0, rng)
    return cv


def band(seed=7, w=420, h=120, rows=((70, 22, 3), (82, 34, 2), (96, 50, 1), (112, 80, 0))):
    """A band of scrub along a bank, several rows receding: (ground y, mean width, depth band)."""
    cv = sky_ground(w, h, 64)
    cv.mat[64:] = GRAVEL; cv.step[64:] = 3
    rng = np.random.default_rng(seed)
    for (gy, mw, bd) in rows:
        x = -mw * 0.3
        while x < w + mw * 0.3:
            W_ = mw * rng.uniform(0.6, 1.3)
            paint_bush(cv, x, int(gy + rng.uniform(-2, 2)), W_, W_ * rng.uniform(0.6, 0.85), L0, rng, band=bd)
            x += W_ * rng.uniform(0.45, 0.8)
    return cv


if __name__ == "__main__":
    pal = band_palette("summer", "noon")
    cv = lod_strip(); save(cv.render(pal), f"{OUT}/s4_lod_1x.png"); save(cv.render(pal), f"{OUT}/s4_lod_3x.png", 3)
    cv = clump(); save(cv.render(pal), f"{OUT}/s4_clump_3x.png", 3)
    cv = band(); save(cv.render(pal), f"{OUT}/s4_band_2x.png", 2)
