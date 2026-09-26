"""Band-limited, world-anchored noise.

Everything the painter uses is a function of *world* position, evaluated at whatever
sample spacing the current zoom needs.  Octaves finer than ~2 samples fade out
smoothly (instead of being cut), so a slow zoom never pops a new octave in.
"""
import numpy as np

_P = np.random.RandomState(1234).permutation(4096).astype(np.int32)
_P2 = np.concatenate([_P, _P])
_F = np.random.RandomState(4321).random_sample(4096).astype(np.float32)


def lat(ix, iy, seed=0):
    """Fast lattice value in [0,1) by table lookup (period 4096)."""
    ix = (np.asarray(ix) + seed * 97) & 4095
    iy = (np.asarray(iy) + seed * 57) & 4095
    return _F[(_P2[ix] + iy) & 4095]


def hash2(ix, iy, seed=0):
    """Integer lattice hash -> float in [0,1). Stable across frames and zooms."""
    ix = np.asarray(ix, dtype=np.int64)
    iy = np.asarray(iy, dtype=np.int64)
    h = (ix * 374761393 + iy * 668265263 + seed * 2147483647) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / float(1 << 24)


def hashn(ix, iy, seed, k):
    """k independent hashes per lattice point."""
    return [hash2(ix, iy, seed * 131 + j * 7919 + 17) for j in range(k)]


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def value_noise(x, y, seed=0):
    """Smooth value noise in [-1,1], lattice spacing 1."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    fx0 = np.floor(x)
    fy0 = np.floor(y)
    ix = fx0.astype(np.int64)
    iy = fy0.astype(np.int64)
    fx = _fade(x - fx0)
    fy = _fade(y - fy0)
    a = lat(ix, iy, seed)
    b = lat(ix + 1, iy, seed)
    c = lat(ix, iy + 1, seed)
    d = lat(ix + 1, iy + 1, seed)
    v = a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy
    return v * 2 - 1


def fbm(x, y, base_wl, octaves, gain=0.5, lac=2.0, seed=0, min_wl=0.0):
    """Fractal sum. base_wl = largest wavelength (m).  Octaves whose wavelength is below
    ~2*min_wl (the sample spacing) fade to zero continuously -> no popping on zoom."""
    tot = 0.0
    amp = 1.0
    wl = base_wl
    norm = 0.0
    for o in range(octaves):
        if min_wl > 0:
            w = np.clip((wl / min_wl - 2.0) / 2.0, 0.0, 1.0)  # 0 below 2 samples, 1 above 4
        else:
            w = 1.0
        if np.any(w > 0):
            # rotate each octave a little to kill lattice alignment
            ang = 0.6 * o + seed * 0.37
            ca, sa = np.cos(ang), np.sin(ang)
            xr = (x * ca - y * sa) / wl
            yr = (x * sa + y * ca) / wl
            tot = tot + amp * w * value_noise(xr + 17.3 * o, yr - 9.1 * o, seed + o * 101)
        norm += amp
        amp *= gain
        wl /= lac
    return tot / norm


_CDF = None


def _cdf_table():
    """Empirical CDF of the normalised two-octave blend used by stable_threshold, so the
    mapping to [0,1) is a fixed function (a per-frame rank would depend on what else is
    in the frame, and pixels near the threshold would flicker as the view changes)."""
    global _CDF
    if _CDF is None:
        rs = np.random.RandomState(9)
        x, y = rs.uniform(0, 4000, 200000), rs.uniform(0, 4000, 200000)
        vals = []
        for f in (0.0, 0.25, 0.5):
            a = value_noise(x, y, 3)
            b = value_noise(x / 2, y / 2, 5)
            vals.append((a * (1 - f) + b * f) / np.sqrt((1 - f) ** 2 + f ** 2))
        v = np.sort(np.concatenate(vals))
        _CDF = v[:: max(1, len(v) // 4096)]
    return _CDF


def stable_threshold(X, Y, ppm, px=1.6, seed=5):
    """Dither threshold anchored to the world surface (no shower-door shimmer on zoom).

    Value noise with a lattice ~px screen pixels, living in world coordinates.  As the
    camera zooms, the lattice level (a power of two in world metres) changes; the two
    nearest levels are cross-faded by the fractional zoom octave, so the pattern grows
    and divides continuously instead of swimming or popping (cf. surface-stable
    fractal dithering).  The blend is variance-normalised and mapped through a fixed CDF
    to uniform [0,1).
    """
    cell = px / ppm
    lv = np.log2(cell)
    k = np.floor(lv)
    f = lv - k
    a = value_noise(X / 2 ** k, Y / 2 ** k, seed + int(k) * 3)
    b = value_noise(X / 2 ** (k + 1), Y / 2 ** (k + 1), seed + int(k + 1) * 3)
    v = (a * (1 - f) + b * f) / np.sqrt((1 - f) ** 2 + f ** 2)
    cdf = _cdf_table()
    return (np.searchsorted(cdf, v) + 0.5) / (len(cdf) + 1)
