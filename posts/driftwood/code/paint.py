"""paint.py -- step-space painting for driftwood on gravel.

Index canvas: two arrays, mat (material id) and x (continuous step coordinate on that
material's ramp).  Marks add offsets to x; contact lines and cracks set x to hard values.
quantize() mixes only adjacent steps of one ramp with an ordered 2x2 dither.

Wood ramp (8 steps):   0 contact/crack | 1 core | 2 shadow | 3 reflected   ||   4 halftone | 5 light | 6 top | 7 end-face / glint
Gravel ramp (8 steps): same plan, keyed a step lower than wood so pale logs read against it.
"""
import numpy as np
from geo import norm

WOOD, GRAVEL, RAW, BARK, SAND = 0, 1, 2, 3, 4
BAYER2 = np.array([[0.125, 0.625], [0.875, 0.375]])
BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16 + 1 / 32


def value_noise(H, W, cell, rng, octaves=1):
    out = np.zeros((H, W))
    amp = 1.0; tot = 0
    for o in range(octaves):
        c = max(1, cell / (2 ** o))
        gh, gw = int(H / c) + 3, int(W / c) + 3
        g = rng.random((gh, gw))
        yy = np.arange(H) / c; xx = np.arange(W) / c
        y0 = yy.astype(int); x0 = xx.astype(int)
        fy = (yy - y0)[:, None]; fx = (xx - x0)[None, :]
        fy = fy * fy * (3 - 2 * fy); fx = fx * fx * (3 - 2 * fx)
        a = g[y0][:, x0]; b = g[y0][:, x0 + 1]; cc = g[y0 + 1][:, x0]; d = g[y0 + 1][:, x0 + 1]
        out += amp * (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + cc * (1 - fx) * fy + d * fx * fy)
        tot += amp; amp *= 0.5
    return out / tot


def hash2(a, b, seed=0):
    """deterministic pseudo-random in [0,1) from integer arrays."""
    h = (np.asarray(a, np.int64) * 374761393 + np.asarray(b, np.int64) * 668265263 + seed * 144269504) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / float(0x1000000)


class Light:
    """designed light.  az: compass direction the sun comes FROM relative to the camera
    (0 = from behind the camera / front light, 90 = from the right, 180 = back light, 270 = left);
    el: elevation in degrees."""

    def __init__(self, cam, az=300.0, el=40.0, overcast=False):
        a, e = np.radians(az), np.radians(el)
        back = -cam.f * np.array([1, 1, 0]); back = back / np.linalg.norm(back)
        right = cam.r
        hor = np.cos(a) * back + np.sin(a) * right
        self.L = norm(hor * np.cos(e) + np.array([0, 0, 1.0]) * np.sin(e))
        self.overcast = overcast
        self.az, self.el = az, el


def form_wood(buf, light, sh, terminator=0.12):
    """two value families on the wood.  returns x (step coordinate) for all pixels."""
    N = buf['N']
    ndl = N @ light.L
    up = N[..., 2]
    lit = (ndl > terminator) & ~sh
    # light family 4..7 : halftone at grazing, top at full
    f = np.clip((ndl - terminator) / (0.85 - terminator), 0, 1)
    xl = 4.0 + 2.7 * f ** 0.8
    # shadow family 0.8..3.4: core just past the terminator, sky fill on up-facing, bounce
    # from the pale gravel on down-facing (the reflected light never reaches the halftone)
    sky = np.clip(up, 0, 1)
    bounce = np.clip(-up, 0, 1)
    core = np.exp(-((ndl - terminator * 0.3) / 0.22) ** 2)       # band just past terminator
    xs = 1.4 + 1.1 * sky + 1.4 * bounce ** 0.7 - 0.7 * core
    # a cast-shadowed surface that faces the sun keeps a little more (sky + it is a top)
    xs = np.where(sh & (ndl > terminator), xs + 0.4, xs)
    xs = np.clip(xs, 0.8, 3.45)
    if light.overcast:
        x = 1.6 + 3.8 * np.clip(0.5 + 0.5 * up, 0, 1) ** 1.3
        return x, np.ones_like(lit)
    return np.where(lit, xl, xs), lit


def gravel_field(buf, rng, stone_m=0.07, seed=0, spread=0.6):
    """world-space pebbles on the ground: jittered-grid cells, each a flattened dome.
    returns (inside mask, dome normal, stone value offset, cell id)."""
    P = buf['P']
    X, Y = P[..., 0], P[..., 1]
    c = stone_m
    gx, gy = np.floor(X / c).astype(int), np.floor(Y / c).astype(int)
    best = np.full(X.shape, np.inf); bnx = np.zeros_like(X); bny = np.zeros_like(X)
    brad = np.ones_like(X); bid = np.zeros(X.shape, np.int64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cx, cy = gx + dx, gy + dy
            jx = hash2(cx, cy, seed + 1); jy = hash2(cx, cy, seed + 2)
            rr = (0.45 + spread * hash2(cx, cy, seed + 3) ** 1.5) * c * 0.62
            ang = hash2(cx, cy, seed + 4) * np.pi
            el = 0.6 + 0.4 * hash2(cx, cy, seed + 5)
            px = (cx + 0.15 + 0.7 * jx) * c; py = (cy + 0.15 + 0.7 * jy) * c
            vx, vy = X - px, Y - py
            ca, sa = np.cos(ang), np.sin(ang)
            u = (vx * ca + vy * sa) / rr
            v = (-vx * sa + vy * ca) / (rr * el)
            d = np.sqrt(u * u + v * v)
            m = d < best
            best = np.where(m, d, best)
            bnx = np.where(m, u * ca - v * sa, bnx)
            bny = np.where(m, u * sa + v * ca, bny)
            bid = np.where(m, cx * 7919 + cy, bid)
    inside = best < 1.0
    nz = np.sqrt(np.clip(1 - best ** 2, 0, 1)) * 0.9 + 0.1
    Nst = norm(np.stack([bnx * 0.8, bny * 0.8, nz], -1))
    tone = hash2(bid, 1, seed + 9)
    return inside, Nst, tone, bid, best


def form_gravel(buf, light, sh, rng, stone_m=0.07, seed=0, occ=None):
    """ground: pebbles as small domes (light top, dark bottom), gaps dark."""
    inside, Nst, tone, bid, dist = gravel_field(buf, rng, stone_m, seed)
    ndl = Nst @ light.L
    up = Nst[..., 2]
    ground_ndl = light.L[2]
    lit = ~sh
    # stone value: its own local tone (albedo) + form
    base = 3.4 + 1.3 * (tone - 0.5)
    xl = base + 1.4 * np.clip(ndl - ground_ndl * 0.6, -1, 1)
    xs = 1.3 + 0.9 * np.clip(up, 0, 1) + 0.7 * (tone - 0.5)
    x = np.where(lit, xl, np.minimum(xs, 2.9))
    # gaps between stones: dark sand
    gap = ~inside
    x = np.where(gap, np.where(lit, 2.2, 0.9), x)
    if light.overcast:
        x = np.where(gap, 1.4, 2.6 + 1.3 * (tone - 0.5) + 1.2 * up)
    return x, inside, bid


def quantize(x, bayer=BAYER2, flat_snap=None, nsteps=8):
    """ordered dither between adjacent integer steps."""
    H, W = x.shape
    by, bx = bayer.shape
    th = np.tile(bayer, (H // by + 1, W // bx + 1))[:H, :W]
    lo = np.floor(x)
    fr = x - lo
    q = lo + (fr > th)
    return np.clip(q, 0, nsteps - 1).astype(int)


def to_rgb(mat, q, ramps):
    out = np.zeros(mat.shape + (3,), np.uint8)
    for m, ramp in ramps.items():
        sel = mat == m
        if sel.any():
            r = np.asarray(ramp, np.uint8)
            out[sel] = r[np.clip(q[sel], 0, len(r) - 1)]
    return out


GREY_WOOD = [int(255 * v) for v in (0.05, 0.15, 0.25, 0.35, 0.56, 0.70, 0.82, 0.93)]
GREY_GRAV = [int(255 * v) for v in (0.04, 0.12, 0.21, 0.30, 0.41, 0.50, 0.58, 0.66)]
GREY_RAMPS = {WOOD: [(g, g, g) for g in GREY_WOOD], GRAVEL: [(g, g, g) for g in GREY_GRAV],
              RAW: [(g, g, g) for g in GREY_WOOD], BARK: [(max(0, g - 30),) * 3 for g in GREY_WOOD]}
