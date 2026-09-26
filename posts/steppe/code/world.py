"""world: cameras, the ground (loess, soil crust, basalt pebbles, small grasses), placement of
plants in the world, cast shadows, and depth bands.

World units are metres. X runs right, Z runs away from the viewer, heights are up.
Two cameras:
  Iso(el, ppm):   orthographic, looking down at `el` degrees (30 = the game's iso view).
  Persp(eye, f, horizon_y): a standing viewer; ground at distance Z lands at y = hy + f*eye/Z.
Every mark size is decided in PIXELS from the local pixels-per-metre, so the same world paints
at any zoom, and the ground texture is a function of world position (it does not swim).
"""
import numpy as np
import sp
from sp import GROUND, CRUST, STONE, GRASS, SAGE, SAGESTEM, BITTER, SNOW, FORB


# ---------------------------------------------------------------- hashed value noise (world space)
def _hash(ix, iz, seed):
    h = (ix.astype(np.int64) * 374761393 + iz.astype(np.int64) * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFF) / float(0xFFFFFF)


def vnoise(x, z, scale, seed):
    """Smooth value noise at world points (arrays), feature size `scale` metres, in [0,1]."""
    u, v = x / scale, z / scale
    iu, iv = np.floor(u), np.floor(v)
    fu, fv = u - iu, v - iv
    fu, fv = fu * fu * (3 - 2 * fu), fv * fv * (3 - 2 * fv)
    a = _hash(iu, iv, seed)
    b = _hash(iu + 1, iv, seed)
    c = _hash(iu, iv + 1, seed)
    d = _hash(iu + 1, iv + 1, seed)
    return (a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv


def fbm(x, z, scale, seed, oct=3):
    t, amp, s = 0.0, 1.0, 0.0
    for o in range(oct):
        t = t + amp * vnoise(x, z, scale / 2 ** o, seed + 17 * o)
        s += amp
        amp *= 0.5
    return t / s


# ---------------------------------------------------------------- cameras
class Iso:
    """Orthographic, elevation `el` degrees, `ppm` pixels per metre, world origin at screen
    (x0, y0) (the front-left corner of the ground patch)."""
    kind = "iso"

    def __init__(self, W, H, el=30.0, ppm=12.0):
        self.W, self.H, self.el, self.ppm = W, H, el, ppm
        self.s, self.c = np.sin(np.radians(el)), np.cos(np.radians(el))

    def ground_xz(self, xs, ys):
        X = xs / self.ppm
        Z = (self.H - ys) / (self.ppm * self.s)
        return X, Z

    def project(self, X, Z, hgt=0.0):
        return X * self.ppm, self.H - Z * self.ppm * self.s - hgt * self.ppm * self.c

    def ppm_at(self, Z):
        return self.ppm

    def band(self, Z):
        return 0

    def zrange(self):
        return 0.0, self.H / (self.ppm * self.s) + 4.0


class Persp:
    """A standing viewer: eye height `eye` m, focal length f px, horizon at screen row hy."""
    kind = "persp"

    def __init__(self, W, H, eye=1.7, f=400.0, hy=None, bands=(12, 30, 80, 200), zfar=2000.0):
        self.W, self.H, self.eye, self.f = W, H, eye, f
        self.hy = H * 0.35 if hy is None else hy
        self.el = np.degrees(np.arctan2(eye, f * eye / max(1, H - self.hy)))
        self.bands = bands
        self.zfar = zfar

    def ground_xz(self, xs, ys):
        dy = np.maximum(ys - self.hy, 1e-3)
        Z = self.f * self.eye / dy
        X = (xs - self.W / 2) * Z / self.f
        return X, Z

    def project(self, X, Z, hgt=0.0):
        k = self.f / Z
        return self.W / 2 + X * k, self.hy + (self.eye - hgt) * k

    def ppm_at(self, Z):
        return self.f / Z

    def band(self, Z):
        return int(np.searchsorted(self.bands, Z))

    def zrange(self):
        return self.f * self.eye / (self.H - self.hy), self.zfar


# ---------------------------------------------------------------- the ground
class GroundParams(dict):
    DEFAULT = dict(
        crust=0.35,       # share of bare ground under soil crust
        stones=0.25,      # basalt pebble cover (0..1)
        litter=0.5,       # tiny grasses (Sandberg bluegrass, cheatgrass) between the plants
        undulation=0.8,   # low-frequency value variation, steps
        snow=0.0,         # 0..1 snow cover
        season="summer",
        seed=0,
    )

    def __init__(self, **kw):
        super().__init__(self.DEFAULT)
        self.update(kw)


def paint_ground(cv, cam, gp, sun_dx=1):
    """Fill ground pixels (rows below the horizon) with the loess/crust/pebble texture."""
    H, W = cv.h, cv.w
    ys, xs = np.mgrid[0:H, 0:W].astype(float)
    if cam.kind == "persp":
        below = ys > cam.hy + 0.5
    else:
        below = np.ones((H, W), bool)
    X, Z = cam.ground_xz(xs + 0.5, ys + 0.5)
    seed = gp["seed"]
    ppm = cam.ppm_at(Z)
    # base: lit loess, undulating
    und = (fbm(X, Z, 6.0, seed + 1) - 0.5) * 2 * gp["undulation"]
    # the ground is calm: flat lit loess, a slow undulation dithered only near its seams, and
    # sparse deliberate specks (Ferrari's ground is mostly flat, the marks carry the texture)
    base = 3.0 + und
    fl = np.floor(base)
    # value masses meet on a broken edge (noise at ~2 px), not a dithered halo
    brk = vnoise(X, Z, np.maximum(0.03, 2.0 / ppm), seed + 13) - 0.5
    step = np.clip(np.floor(base + brk * 0.5 + 0.5).astype(np.int16), 2, 4)
    r1 = vnoise(X, Z, np.maximum(0.02, 0.7 / ppm), seed + 2)
    r2 = vnoise(X, Z, np.maximum(0.02, 0.7 / ppm), seed + 12)
    step = np.where((r1 > 0.86) & (ppm > 3), step - 1, step)
    step = np.where((r2 > 0.9) & (ppm > 3), np.minimum(4, step + 1), step)
    mat = np.full((H, W), GROUND, np.uint8)
    # soil crust: dark lumpy patches between plants
    cr = fbm(X, Z, 0.9, seed + 3)
    edge = vnoise(X, Z, np.maximum(0.02, 0.8 / ppm), seed + 5) - 0.5
    dith = np.random.default_rng(seed + 8).random(cr.shape) - 0.5
    crust = (cr + edge * 0.14 + dith * 0.06) < gp["crust"] * 0.62
    lump = vnoise(X, Z, np.maximum(0.025, 1.0 / ppm), seed + 4)
    cstep = np.where(lump > 0.45, 2, np.where(lump > 0.18, 1, 0))
    mat[crust] = CRUST
    step[crust] = cstep[crust]
    # snow
    if gp["snow"] > 0:
        sn = fbm(X, Z, 1.4, seed + 9) + 0.25 * (vnoise(X, Z, 0.2, seed + 10) - 0.5)
        brk = vnoise(X, Z, np.maximum(0.02, 1.0 / ppm), seed + 14) - 0.5      # a broken edge
        mid = vnoise(X, Z, 0.35, seed + 15) - 0.5                              # mid-scale lobes
        snow = (sn + brk * 0.14 + mid * 0.22) < gp["snow"] * 1.05
        mat[snow] = SNOW
        step[snow] = np.where(vnoise(X[snow], Z[snow], 0.5, seed + 11) > 0.25, 2, 1)
    cv.mat[below] = mat[below]
    cv.step[below] = step[below]
    if cam.kind == "persp":
        cv.band[below] = np.searchsorted(cam.bands, Z[below]).astype(np.uint8)
    # pebbles and small grasses: sites on a world grid, drawn only where they are >= 1 px
    _pebbles(cv, cam, gp, sun_dx)
    _litter(cv, cam, gp, sun_dx)
    return X, Z


def _sites(cam, cell, density, seed):
    z0, z1 = cam.zrange()
    if cam.kind == "iso":
        x0, x1 = 0.0, cam.W / cam.ppm
    else:
        x0, x1 = -z1 * cam.W / cam.f / 2, z1 * cam.W / cam.f / 2
    out = []
    # iterate rows of cells by depth, skipping those whose cells are sub-pixel
    zs = np.arange(np.floor(z0 / cell), np.ceil(min(z1, 400.0) / cell) + 1)
    for iz in zs:
        Zc = (iz + 0.5) * cell
        if cam.kind == "persp":
            xa, xb = -Zc * cam.W / cam.f / 2 - cell, Zc * cam.W / cam.f / 2 + cell
            if cam.ppm_at(Zc) * cell < 0.6:
                break
        else:
            xa, xb = x0 - cell, x1 + cell
        ix = np.arange(np.floor(xa / cell), np.ceil(xb / cell) + 1)
        izs = np.full_like(ix, iz)
        r = _hash(ix, izs, seed)
        keep = r < density
        if keep.any():
            jx = _hash(ix, izs, seed + 5)[keep]
            sz = _hash(ix, izs, seed + 7)[keep]
            for a, b, c, d in zip(ix[keep], jx, _hash(ix, izs, seed + 6)[keep], sz):
                out.append(((a + b) * cell, (iz + c) * cell, d))
    return out


def _pebbles(cv, cam, gp, sun_dx):
    if gp["stones"] <= 0:
        return
    for X, Z, s in _sites(cam, 0.35, gp["stones"] * 0.6, gp["seed"] + 21):
        r_m = 0.025 + 0.09 * s ** 2
        ppm = cam.ppm_at(Z)
        r = r_m * ppm
        if r < 0.5:
            continue
        x, y = cam.project(X, Z)
        x, y = int(round(x)), int(round(y))
        if not (0 <= x < cv.w and 0 <= y < cv.h) or cv.mat[y, x] not in (GROUND, CRUST):
            continue
        band = cam.band(Z)
        sq = cam.s if cam.kind == "iso" else max(0.25, cam.eye / Z)
        ry = max(0.5, r * max(0.45, sq + 0.3))
        rr = int(np.ceil(r))
        for dy in range(-int(np.ceil(ry)), int(np.ceil(ry)) + 1):
            for dx in range(-rr, rr + 1):
                if (dx / max(r, 0.5)) ** 2 + (dy / ry) ** 2 <= 1.0:
                    t = (dy + ry) / (2 * ry)           # 0 top .. 1 bottom
                    side = dx * sun_dx
                    st = 3 if t < 0.4 else 2 if t < 0.75 else 1
                    if side < 0 and t > 0.3:
                        st -= 1
                    cv.px(x + dx, y + dy, STONE, max(0, st), band)
        # cast shadow pixel on the ground, away from the sun, and the dark foot
        sx = x + sun_dx * (rr + 1)
        if 0 <= sx < cv.w and 0 <= y < cv.h and cv.mat[y, sx] in (GROUND, CRUST):
            cv.step[y, sx] = max(0, cv.step[y, sx] - 2)


def _litter(cv, cam, gp, sun_dx):
    """Tiny grasses between the plants: at near scale 1-3 px upright ticks (lit top), else nothing
    (their colour is already in the ground ramp)."""
    if gp["litter"] <= 0:
        return
    for X, Z, s in _sites(cam, 0.12, gp["litter"] * 0.25, gp["seed"] + 31):
        ppm = cam.ppm_at(Z)
        hpx = (0.04 + 0.1 * s) * ppm * (cam.c if cam.kind == "iso" else 1.0)
        if hpx < 1.0:
            continue
        x, y = cam.project(X, Z)
        x, y = int(round(x)), int(round(y))
        if not (0 <= x < cv.w and 0 <= y < cv.h) or cv.mat[y, x] not in (GROUND, CRUST):
            continue
        band = cam.band(Z)
        n = int(min(4, hpx))
        for k in range(n):
            cv.px(x, y - k, GRASS, 1 if k == 0 else (3 if k == n - 1 else 2), band)
        if hpx > 2.5:
            cv.px(x + 1, y - 1, GRASS, 2, band)
            cv.px(x - 1, y, GRASS, 1, band)
