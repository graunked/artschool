import numpy as np
def crag(profile, rng, every=(6, 18), height=(1, 5)):
    """mid-scale structure: sub-peaks and notches every 6-18 px along a clean line, each one
    itself built from clean slopes (1:1, 2:1 or 1:2 px). profile = top row per column."""
    p = profile.astype(float).copy(); ok = profile >= 0
    xs = np.arange(len(p)); x = rng.integers(0, every[0])
    while x < len(p):
        if ok[x]:
            h = rng.integers(height[0], height[1] + 1)
            sl, sr = rng.choice([1.0, 0.5, 2.0], 2)          # px of rise per px of run, each side
            top = p[x] - h
            if rng.random() < 0.75:                           # a crag standing up
                bump = np.where(xs < x, top + (x - xs) * sl, top + (xs - x) * sr)
                p = np.where(ok, np.minimum(p, bump), p)
            else:                                             # a notch cut down
                v = np.where(xs < x, p[x] + h - (x - xs) * sl, p[x] + h - (xs - x) * sr)
                p = np.where(ok & (np.abs(xs - x) < h / min(sl, sr) + 1), np.maximum(p, np.minimum(v, p + h)), p)
        x += rng.integers(*every)
    return np.where(ok, np.round(p).astype(int), -1)
