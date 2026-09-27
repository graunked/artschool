import numpy as np
from scipy import ndimage as nd
def fbm(shape, scale, seed=0, octaves=4, persist=0.5):
    """smooth value-noise fbm on a grid; scale = feature size in cells. Returns ~N(0,1)-ish."""
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, np.float32); amp = 1.0; tot = 0
    s = float(scale)
    for o in range(octaves):
        g = rng.standard_normal((int(shape[0] / s) + 3, int(shape[1] / s) + 3)).astype(np.float32)
        z = nd.zoom(g, s, order=3)[:shape[0], :shape[1]]
        if z.shape != shape: z = np.pad(z, ((0, shape[0] - z.shape[0]), (0, shape[1] - z.shape[1])), mode='reflect')
        out += amp * z; tot += amp * amp; amp *= persist; s = max(s / 2, 1.0)
    return out / np.sqrt(tot)
def hash2(a, b, seed=0):
    x = (np.asarray(a, np.int64) * 374761393 + np.asarray(b, np.int64) * 668265263 + seed * 2246822519) & 0xFFFFFFFF
    x = ((x ^ (x >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((x ^ (x >> 16)) & 0xFFFFFF) / float(0xFFFFFF)
