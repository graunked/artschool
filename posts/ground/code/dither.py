"""Quantizers: continuous step coordinate v in [0, n-1] -> integer step, mixing only adjacent steps.

ordered(v, jitter): DPaint-style gradient-fill dither. 2x2 Bayer thresholds (so 50% = checker),
                    plus a random jitter on the threshold (DPaint's 'dither' slider).
"""
import numpy as np
B2 = np.array([[0.125, 0.625], [0.875, 0.375]])
B4 = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16

def tile(m, h, w, ox=0, oy=0):
    k = m.shape[0]
    return np.tile(m, (h // k + 2, w // k + 2))[oy % k: oy % k + h, ox % k: ox % k + w]

def ordered(v, jitter=0.25, rng=None, bayer=B2, n=None):
    h, w = v.shape
    rng = rng or np.random.default_rng(0)
    t = tile(bayer, h, w) + jitter * (rng.random((h, w)) - 0.5)
    base = np.floor(v)
    s = base + ((v - base) > t)
    if n is not None: s = np.clip(s, 0, n - 1)
    return s.astype(int)

def stochastic(v, rng=None, n=None, clump=None):
    """random threshold (optionally from a clumped noise field in [0,1])"""
    h, w = v.shape
    rng = rng or np.random.default_rng(0)
    t = rng.random((h, w)) if clump is None else clump
    base = np.floor(v)
    s = base + ((v - base) > t)
    if n is not None: s = np.clip(s, 0, n - 1)
    return s.astype(int)
