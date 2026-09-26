"""Orthographic oblique camera over the world heightfield, and the value design.

Camera: elevation theta (deg above horizontal), scale ppm (pixels per metre).
Screen up-coordinate u = y*sin(theta) + z*cos(theta).  A vertical blade of height h
projects to h*ppm*cos(theta) pixels; a ground patch of depth d to d*ppm*sin(theta).
So one number per tuft, its projected length in pixels, decides how it is painted,
and top-down (theta=90) collapses every blade to a point: tone and texture only.
"""
import numpy as np
from dataclasses import dataclass
from .world import World
from . import palette as P
from .noise import fbm, stable_threshold

MAT_SKY, MAT_GRASS, MAT_WATER, MAT_SOIL = 0, 1, 2, 3


@dataclass
class Camera:
    ppm: float = 30.0
    theta: float = 30.0
    W: int = 320
    H: int = 200
    cx: float = 0.0      # world x at screen centre
    cy: float = 0.0      # world y of the ground point at screen centre
    cz: float = 0.0      # world z of that point

    @property
    def s(self):
        return np.sin(np.radians(self.theta))

    @property
    def c(self):
        return np.cos(np.radians(self.theta))

    def u0(self):
        return self.cy * self.s + self.cz * self.c

    def to_screen(self, x, y, z):
        sx = (x - self.cx) * self.ppm + self.W / 2
        u = y * self.s + z * self.c
        sy = self.H / 2 - (u - self.u0()) * self.ppm
        return sx, sy


class Raster:
    """Per-pixel world position, normal, material and depth of the visible surface."""

    def __init__(self, world: World, cam: Camera, oversample=2.5):
        self.world, self.cam = world, cam
        W, H, ppm = cam.W, cam.H, cam.ppm
        s, c = cam.s, cam.c
        min_wl = 1.0 / ppm                       # a pixel, in metres
        self.min_wl = min_wl
        xs = cam.cx + (np.arange(W) + 0.5 - W / 2) / ppm
        u_top = cam.u0() + (H / 2 + 2) / ppm
        u_bot = cam.u0() - (H / 2 + 2) / ppm
        # z range estimate (coarse probe)
        py = np.linspace(cam.cy - (H / ppm) / s * 1.5 - 5, cam.cy + (H / ppm) / s * 1.5 + 5, 64)
        PX, PY = np.meshgrid(xs[::max(1, W // 32)], py)
        pz = np.maximum(world.height(PX, PY, min_wl), world.water_level)
        zmin, zmax = pz.min() - 1.0, pz.max() + 1.0
        y_near = (u_bot - zmax * c) / s - 0.5
        y_far = (u_top - zmin * c) / s + 0.5
        dy = 1.0 / (ppm * s * oversample)
        dy = max(dy, (y_far - y_near) / 6000.0)
        # world-anchored sample rows (a global grid of spacing dy): panning the camera
        # samples exactly the same ground, so nothing crawls
        ys = dy * np.arange(np.floor(y_near / dy), np.ceil(y_far / dy) + 1)
        X, Y = np.meshgrid(xs, ys)                # [ns, W]
        Z = np.maximum(world.height(X, Y, min_wl), world.water_level)
        U = Y * s + Z * c
        Umax = np.maximum.accumulate(U, axis=0)
        # pixel row r has screen-up coordinate
        rows = np.arange(H)
        ur = cam.u0() + (H / 2 - (rows + 0.5)) / ppm   # [H]
        idx = np.empty((H, W), np.int64)
        for i in range(W):
            idx[:, i] = np.searchsorted(Umax[:, i], ur, side="left")
        sky = idx >= len(ys)
        idx = np.minimum(idx, len(ys) - 1)
        # interpolate world y within the sample interval for a smooth surface position
        i0 = np.maximum(idx - 1, 0)
        cols = np.broadcast_to(np.arange(W), (H, W))
        ua, ub = U[i0, cols], U[idx, cols]
        t = np.clip((ur[:, None] - ua) / np.maximum(ub - ua, 1e-9), 0, 1)
        t = np.where(idx == i0, 1.0, t)
        Yp = ys[i0] + (ys[idx] - ys[i0]) * t
        Xp = np.broadcast_to(xs, (H, W)).copy()
        self.X, self.Y = Xp, Yp
        zt = world.height(Xp, Yp, min_wl)
        self.Zterrain = zt
        self.water = (zt < world.water_level) & ~sky
        self.Z = np.maximum(zt, world.water_level)
        self.sky = sky
        self.N = world.normal(Xp, Yp, min_wl)
        self.depth = np.where(sky, np.inf, Yp)     # world y of visible surface (smaller = nearer)
        mat = np.full((H, W), MAT_GRASS, np.int8)
        mat[self.water] = MAT_WATER
        mat[sky] = MAT_SKY
        # steep cut-bank faces show soil
        steep = (self.N[2] < 0.35) & ~self.water & ~sky
        mat[steep] = MAT_SOIL
        self.mat = mat


def cast_shadow(world, X, Y, Z, sun, min_wl, reach=80.0, steps=16, soft=0.06, simplify_px=8.0):
    """Soft shadow in [0,1]: how far (as an angle) the terrain rises above the ray toward
    the sun, ramped over `soft` radians -> a penumbra that later becomes a dithered edge.
    The march is fixed in *world* units (geometric steps 5 cm .. reach), so what is in
    shadow does not depend on the zoom; each step reads terrain band-limited to its own
    distance, so near steps see fine relief and far steps see hills.
    Shadow shapes are simplified like a painter's: relief smaller than `simplify_px`
    pixels casts no shadow (it is texture, the marks carry it), so shadows stay big masses
    instead of camouflage.  The cut is band-limited, so it moves smoothly with zoom."""
    excess = np.full(X.shape, -1.0)
    horiz = np.hypot(sun[0], sun[1]) + 1e-9
    for d in np.geomspace(0.05, reach, steps):
        xs = X + sun[0] / horiz * d
        ys = Y + sun[1] / horiz * d
        mw = max(min_wl * simplify_px, d / 6.0)
        # the ray starts from the surface *at the same band limit* as the samples it is
        # tested against; otherwise fine hollows fall below the coarse surface and fill
        # with false shadow
        z0 = np.maximum(world.height(X, Y, mw), world.water_level)
        zr = z0 + sun[2] / horiz * d
        excess = np.maximum(excess, (world.height(xs, ys, mw) - zr - 0.02) / d)
    return np.clip(excess / soft + 0.5, 0, 1)


def value_design(R: Raster, sun, params=None):
    """Continuous step coordinate V (0..6) per pixel, plus lit mask.

    Light is *designed*, not computed: slope-to-sun differences are exaggerated relative
    to flat ground (a gentle hummock must read), then split into two families that never
    touch: lit 3.3..6, shadow 0.4..2.8.
    Lit family: facing + crest sheen (convex ground and grazing view show lit tips).
    Shadow family: sky light by upward facing + a faint crest rim.
    Folds (concave ground, where one mass tucks under the next) darken in both.
    Curvature is read at a few pixels' scale, so crests read at every zoom.
    """
    p = dict(exag=2.0, lit_lo=3.5, lit_span=1.6, crest=1.1, sh_lo=0.2, sh_span=1.0,
             sh_crest=0.6, fold=0.7, wet_dark=0.9, lit_thresh=0.12, curv_px=6.0,
             large_px=30.0, exag_large=1.5, term_soft=0.12, shadow_simplify=8.0, mottle=0.45, sheen=1.2, material=0.9)
    if params:
        p.update(params)
    world, cam = R.world, R.cam
    N = R.N
    lam = N[0] * sun[0] + N[1] * sun[1] + N[2] * sun[2]
    flat = sun[2]
    # large-scale facing (hill / bank scale, ~30 px): the big value planes that carry a
    # region view, exaggerated harder than the small facets
    eL = max(p["large_px"] / cam.ppm, 0.3)
    NL = world.normal(R.X, R.Y, max(R.min_wl, eL / 2), eps=eL)
    lamL = NL[0] * sun[0] + NL[1] * sun[1] + NL[2] * sun[2]
    lam_e = flat + p["exag"] * (lam - flat) + p["exag_large"] * (lamL - flat)
    v = np.array([0.0, -cam.c, cam.s])
    nv = N[0] * v[0] + N[1] * v[1] + N[2] * v[2]
    graze = np.clip(1.0 - nv, 0, 1)            # 0 facing camera, 1 edge-on
    # convexity at a few pixels' scale
    # masses made of masses: convexity summed over three scales (3, 12, 40 px)
    X, Y = R.X, R.Y
    mw = R.min_wl
    z0 = world.height(X, Y, mw)
    conv = 0.0
    for k, (px, wk) in enumerate(((p["curv_px"], 0.35), (16.0, 0.35), (48.0, 0.3))):
        e = max(px / cam.ppm, 0.12)
        zn = (world.height(X + e, Y, mw) + world.height(X - e, Y, mw)
              + world.height(X, Y + e, mw) + world.height(X, Y - e, mw)) / 4
        conv = conv + wk * np.clip((z0 - zn) / e * 4.0 * (1 + k), -1, 1)   # >0 crest, <0 fold
    sh = cast_shadow(world, R.X, R.Y, R.Z, sun, R.min_wl, simplify_px=p["shadow_simplify"])
    # soft light weight, then a world-anchored dither decides the family per pixel:
    # terminators and shadow edges become dithered fringes, not cut-outs
    wl = np.clip((lam_e - p["lit_thresh"]) / p["term_soft"] + 0.5, 0, 1) * (1 - sh)
    thr = stable_threshold(R.X, R.Y * cam.s, cam.ppm, px=1.0, seed=77)
    lit = wl > thr
    R.wlit = wl
    wet = world.wetness(R.X, R.Y)
    Vl = p["lit_lo"] + p["lit_span"] * np.clip((lam_e - p["lit_thresh"]) / 0.9, 0, 1) \
        + p["crest"] * (np.maximum(conv, 0) + 0.5 * graze ** 2)
    Vs = p["sh_lo"] + p["sh_span"] * np.clip(N[2], 0, 1) ** 2 + p["sh_crest"] * np.maximum(conv, 0)
    # colonies / swards: tone mottling at 8-60 m, the texture of grassland seen from afar
    mot = fbm(X, Y, 60.0, 3, 0.55, seed=world.seed + 61)
    # material masses that belong to the ground, not the light (species, moisture,
    # trampled patches): dry ground lighter, wet flushes darker; carry day and overcast
    mat = world.material(X, Y)
    fold = np.minimum(conv, 0) * np.where(conv < -0.5, 0.6, 1.0)
    V = np.where(lit, Vl, Vs) + p["fold"] * fold - p["wet_dark"] * (wet - 0.5) + p["mottle"] * mot * np.where(lit, 1.0, 0.5) + p["material"] * mat * np.where(lit, 1.0, 0.5) \
        + p["sheen"] * world.wind * world.sheen(X, Y) * lit
    # step 6 is reserved for sun-struck tips: only crests may dither into it
    top = 5.0 + 0.6 * np.clip((conv - 0.2) * 2.5, 0, 1)
    V = np.where(lit, np.clip(V, 3.0, top), np.clip(V, 0.2, 2.8))
    R.conv = conv
    R.graze = graze
    return V, lit, lam_e, graze
