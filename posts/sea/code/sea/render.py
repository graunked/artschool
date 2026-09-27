"""voxsim's camera: orthographic, 30 deg (row = t*sin(e) - (z - zc)*zf*cos(e)/cos(30)), yawed.
Plus a perspective eye-level camera for the shore view. Produces a G-buffer: world x, y (1 m grid
coords, col/row south-positive), z, sea mask, and land normal."""
import numpy as np, math
from .waves import bilinear

class Cam:
    def __init__(self, cx, cy, mpp, W=640, H=360, yaw=0.0, elev=30.0, zf=1.0, zc=0.0):
        self.cx, self.cy, self.mpp, self.W, self.H = cx, cy, mpp, W, H
        self.yaw, self.elev, self.zf, self.zc = yaw, elev, zf, zc
        a = math.radians(yaw)
        # screen-right and screen-down(ground) unit vectors in grid coords (x east, y south).
        # yaw=0: camera looks north (up the grid): right = +x, forward-down-screen = +y (south = near)
        self.R = (math.cos(a), math.sin(a)); self.V = (-math.sin(a), math.cos(a))
        self.ty = math.sin(math.radians(elev)); self.zy = zf * math.cos(math.radians(elev)) / math.cos(math.radians(30))
    def project(self, x, y, z):
        dx, dy = x - self.cx, y - self.cy
        u = dx * self.R[0] + dy * self.R[1]; t = dx * self.V[0] + dy * self.V[1]
        return u / self.mpp + self.W / 2, (t * self.ty - (z - self.zc) * self.zy) / self.mpp + self.H / 2, t
    def unproject_plane(self, px, py, z):
        u = (px - self.W / 2) * self.mpp
        t = ((py - self.H / 2) * self.mpp + (z - self.zc) * self.zy) / self.ty
        x = self.cx + u * self.R[0] + t * self.V[0]; y = self.cy + u * self.R[1] + t * self.V[1]
        return x, y

def gbuffer(cam, zgrid, sea_level=0.0, ss=2, land_min=None):
    """zgrid: land/sea-floor elevation (NaN = deep). Sea pixels come from the plane z = sea_level;
    land samples are splatted (painter's order by t) wherever z > sea_level."""
    W, H, mpp = cam.W, cam.H, cam.mpp
    py, px = np.mgrid[0:H, 0:W].astype(np.float64) + 0.5
    sx, sy = cam.unproject_plane(px, py, sea_level)
    G = dict(x=sx, y=sy, z=np.full((H, W), sea_level), sea=np.ones((H, W), bool),
             t=np.full((H, W), -1e9), nx=np.zeros((H, W)), ny=np.zeros((H, W)), nz=np.ones((H, W)))
    # land splat: sample the world on a grid finer than a pixel, over the footprint
    h, w = zgrid.shape
    zz = np.nan_to_num(zgrid, nan=-50.0)
    gyz, gxz = np.gradient(zz)
    step = mpp / ss
    # footprint of the screen on the ground (with margin for tall land)
    corners = [cam.unproject_plane(a, b, sea_level) for a in (0, W) for b in (0, H + 80 / mpp * 0 + 60 / mpp)]
    corners += [cam.unproject_plane(a, b, sea_level + 80) for a in (0, W) for b in (0, H)]
    xs_ = [c[0] for c in corners]; ys_ = [c[1] for c in corners]
    x0, x1 = max(0, min(xs_) - 5), min(w - 1, max(xs_) + 5); y0, y1 = max(0, min(ys_) - 5), min(h - 1, max(ys_) + 5)
    gx = np.arange(x0, x1, step); gy = np.arange(y0, y1, step)
    for j0 in range(0, gy.size, 400):                           # chunk rows to bound memory
        GY, GX = np.meshgrid(gy[j0:j0 + 400], gx, indexing='ij')
        z = bilinear(zz, GX, GY)
        m = z > sea_level + 0.02
        if not m.any(): continue
        GXm, GYm, zm = GX[m], GY[m], z[m]
        u, v, t = cam.project(GXm, GYm, zm)
        iu = np.floor(u).astype(int); iv = np.floor(v).astype(int)
        ok = (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
        iu, iv, t, GXm, GYm, zm = iu[ok], iv[ok], t[ok], GXm[ok], GYm[ok], zm[ok]
        # also fill the pixel below (vertical faces stretch downward on screen)
        for dv in (0, 1):
            ivv = np.minimum(iv + dv, H - 1)
            order = np.argsort(t, kind='stable')
            a_, b_ = ivv[order], iu[order]
            tt = t[order] - 1e-3 * dv
            cur = G['t'][a_, b_]
            better = tt >= cur
            a_, b_, o2 = a_[better], b_[better], order[better]
            G['t'][a_, b_] = t[o2]; G['x'][a_, b_] = GXm[o2]; G['y'][a_, b_] = GYm[o2]; G['z'][a_, b_] = zm[o2]
            G['sea'][a_, b_] = False
    # land normals
    L = ~G['sea']
    nx = -bilinear(gxz, G['x'][L], G['y'][L]); ny = -bilinear(gyz, G['x'][L], G['y'][L])
    n = np.sqrt(nx * nx + ny * ny + 1)
    G['nx'][L] = nx / n; G['ny'][L] = ny / n; G['nz'][L] = 1 / n
    return G

class PerspCam:
    """Eye-level camera for the shore view. pos in grid coords (x, y) + eye height z (m, absolute);
    heading = bearing (deg clockwise from north) the camera faces; pitch (deg, negative = down); hfov deg."""
    def __init__(self, x, y, z, heading, pitch=-2.0, hfov=60.0, W=480, H=270):
        self.x, self.y, self.z, self.heading, self.pitch, self.hfov, self.W, self.H = x, y, z, heading, pitch, hfov, W, H
        self.mpp = None

def persp_gbuffer(cam, zgrid, sea_level=0.0, max_dist=4000.0, step0=0.25, eta=None, depth=None):
    W, H = cam.W, cam.H
    f = (W / 2) / np.tan(np.radians(cam.hfov / 2))
    py, px = np.mgrid[0:H, 0:W].astype(np.float64) + 0.5
    # camera basis (grid coords: x east, y south, z up)
    b = np.radians(cam.heading); p = np.radians(cam.pitch)
    fwd = np.array([np.sin(b) * np.cos(p), -np.cos(b) * np.cos(p), np.sin(p)])
    right = np.array([np.cos(b), np.sin(b), 0.0])
    up = -np.cross(right, fwd)          # grid coords are left-handed (y south)
    d = fwd[None, None] * f + right[None, None] * (px - W / 2)[..., None] - up[None, None] * (py - H / 2)[..., None]
    d /= np.linalg.norm(d, axis=2, keepdims=True)
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
    # sea plane
    tsea = np.where(dz < -1e-6, (cam.z - sea_level) / np.maximum(-dz, 1e-9), np.inf)
    tsea = np.minimum(tsea, max_dist)
    # march terrain (geometric steps) until tsea
    zz = np.nan_to_num(zgrid, nan=-50.0); h, w = zz.shape
    if eta is not None:
        # the water surface is a heightfield too: swell faces occlude the water behind them
        wet = np.nan_to_num(depth, nan=30.0) > 0
        zw = np.where(wet, np.maximum(zz, sea_level + eta), zz)
        iswater = wet & (sea_level + eta > zz)
        tsea = np.full((H, W), np.inf)
    thit = np.full((H, W), np.inf)
    t = np.full((H, W), 0.5); prev_below = np.zeros((H, W), bool)
    for i in range(400):
        x = cam.x + dx * t; y = cam.y + dy * t; zr = cam.z + dz * t
        inside = (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1)
        if eta is not None:
            zt = np.where(inside, bilinear(zw, x, y), np.where(zr > sea_level, -50, 1e9))
            hit = (zt > zr) & np.isinf(thit) & np.isinf(tsea)
            xi = np.clip(x, 0, w - 1).astype(int); yi = np.clip(y, 0, h - 1).astype(int)
            hw = hit & (~inside | iswater[yi, xi])
            tsea = np.where(hw, t, tsea); thit = np.where(hit & ~hw, t, thit)
        else:
            zt = np.where(inside, bilinear(zz, x, y), -50)
            hit = (zt > zr) & np.isinf(thit) & (t < tsea)
            thit = np.where(hit, t, thit)
        t = t + np.maximum(step0, t * 0.01)
        if (t > np.minimum(np.minimum(tsea, thit), max_dist)).all(): break
    if eta is not None:
        # refine every hit by bisection (kills the stair-step columns of a fixed-step march)
        th = np.minimum(tsea, thit); fin = np.isfinite(th)
        hi = np.where(fin, th, 0); lo = np.where(fin, np.maximum(hi - np.maximum(step0, hi * 0.01), 0.1), 0)
        for _ in range(7):
            mid = 0.5 * (lo + hi)
            x = cam.x + dx * mid; y = cam.y + dy * mid; zr = cam.z + dz * mid
            zt = bilinear(zw, x, y)
            above = zt > zr
            hi = np.where(fin & above, mid, hi); lo = np.where(fin & ~above, mid, lo)
        tsea = np.where(np.isfinite(tsea), hi, tsea); thit = np.where(np.isfinite(thit), hi, thit)
    sea = np.isfinite(tsea) & (tsea < thit) & (tsea < max_dist)
    land = np.isfinite(thit) & ~sea
    tt = np.where(sea, tsea, np.where(land, thit, np.nan))
    G = dict(x=cam.x + dx * np.nan_to_num(tt), y=cam.y + dy * np.nan_to_num(tt), z=cam.z + dz * np.nan_to_num(tt),
             sea=sea, land=land, sky=~sea & ~land, dist=tt, dz=dz, dirx=dx, diry=dy)
    # pixel footprint on the ground (m): distance / f, stretched vertically by 1/sin(grazing)
    G['graze'] = np.degrees(np.arcsin(np.clip(-dz, 1e-4, 1))); G['graze_geo'] = G['graze'].copy()
    if eta is not None:
        gye, gxe = np.gradient(eta)
        ex = -bilinear(gxe, G['x'], G['y']); ey = -bilinear(gye, G['x'], G['y']); en = np.sqrt(ex * ex + ey * ey + 1)
        cosi = -(dx * ex + dy * ey + dz) / en                   # cos of incidence on the tilted facet
        G['graze'] = np.where(sea, np.degrees(np.arcsin(np.clip(cosi, 1e-4, 1))), G['graze'])
    # LOD footprint: the long axis (along the view, stretched by 1/sin(grazing)) -- features must be
    # >= k px along the view too, or far water turns into Morse rows
    G['mpp_h'] = np.nan_to_num(tt, nan=1e3) / f
    G['mpp'] = G['mpp_h'] / np.maximum(np.sin(np.radians(G['graze_geo'])), 0.02) ** 0.7
    gyz, gxz = np.gradient(zz)
    nx = -bilinear(gxz, G['x'], G['y']); ny = -bilinear(gyz, G['x'], G['y']); n = np.sqrt(nx * nx + ny * ny + 1)
    G['nx'], G['ny'], G['nz'] = nx / n, ny / n, 1 / n
    return G
