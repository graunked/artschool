"""world.py -- the braided plain modelled only as far as the painting needs, and a raycaster that
turns it into a per-pixel G-buffer for the painter (painter.py).

Ground: y = height(x, z) in metres; water surface at y = 0 wherever the ground is below 0.
Camera: eye at height h, pitch small, pinhole f (px). Columns are vertical planes through the eye,
so a reflected ray stays in its column: the reflection of anything is found by marching the same
column upward from the water point.
"""
import numpy as np


class Cam:
    def __init__(self, W, H, f, h, pitch_deg):
        self.W, self.H, self.f, self.h = W, H, f, h
        self.pitch = pitch_deg
        self.hz = H / 2 - f * np.tan(np.radians(pitch_deg))

    def row_of(self, z, y=0.0):
        return self.hz + self.f * (self.h - y) / z

    def mirror_row(self, z, y):
        return self.hz + self.f * (self.h + y) / z

    def z_of_row(self, r):
        return self.f * self.h / np.maximum(r - self.hz, 1e-3)

    def x_of(self, c, z):
        return (c - self.W / 2) * z / self.f

    def col_of(self, x, z):
        return self.W / 2 + self.f * x / z


# ----------------------------------------------------------------------------- terrains --------
class StraightChannel:
    """a channel along x between z_near(x) and z_far(x); banks of a chosen form.
    far/near bank forms: ('slope', height, run) | ('cut', height) ; bed depth profile parabolic.
    Edge positions and bank heights 'breathe' along x (stepped noise) so no edge is a ruled line."""

    def __init__(self, z_near, z_far, depth=0.4, far=('slope', 0.2, 0.6), near=('slope', 0.12, 1.0),
                 breathe=0.08, bar_rise=0.0, seed=0, bend=0.0, bend_len=6.0):
        self.z_near, self.z_far, self.depth = z_near, z_far, depth
        self.bend, self.bend_len = bend, bend_len
        self.far, self.near = far, near
        rng = np.random.default_rng(seed)
        # stepped wander of both edges along x (metres): piecewise constant over 0.3-1.2 m runs,
        # smoothed a little so a run's ends slope for a pixel or two
        xs = np.arange(-60, 60, 0.05)
        def stepped(amp):
            v = np.zeros_like(xs); i = 0
            while i < len(xs):
                L = int(rng.integers(6, 24)); v[i:i + L] = rng.normal(0, amp); i += L
            return v
        self.xs = xs
        self.dfar = stepped(breathe)
        self.dnear = stepped(breathe * 0.7)
        self.hfar = 1 + stepped(0.08)
        self.bar_rise = bar_rise
        self.riffles = []
        self.stones = []

    def add_riffle(self, xc, half, strength=1.0, n_stones=40, seed=0, skew=0.5):
        """a gravel sill crossing the channel: its LIP is a line x_lip(z) = xc + skew*(z - zmid);
        upstream of it the water is a smooth glassy tongue, downstream the surface breaks, hardest
        just below the lip and dying away over `half` metres (a standing-wave train)."""
        rng = np.random.default_rng(seed)
        self.riffles.append((xc, half, strength))
        self.skew = skew
        for i in range(n_stones):
            x = xc + abs(rng.normal(0, half * 0.5))
            z = rng.uniform(self.z_near + 0.1, self.z_far - 0.1)
            r = np.exp(rng.uniform(np.log(0.03), np.log(0.12)))
            self.stones.append((x, z, r))

    def lip_x(self, z, xc):
        """the lip follows the bed: skewed across the channel and bowed downstream mid-channel,
        where the thalweg runs fastest."""
        zmid = 0.5 * (self.z_near + self.z_far)
        sf = np.clip((z - self.z_near) / (self.z_far - self.z_near), 0, 1)
        return xc + getattr(self, 'skew', 0.0) * (z - zmid) + 0.35 * (self.z_far - self.z_near) * 4 * sf * (1 - sf)

    def lip_u(self, x, z, xc):
        return x - self.lip_x(z, xc)

    def roughness(self, x, z):
        R = np.zeros(np.broadcast(x, z).shape)
        for (xc, half, st) in self.riffles:
            u = self.lip_u(x, z, xc)
            r = np.where(u >= 0, st * np.exp(-u / half) * (1 - np.exp(-u / 0.06)), 0.0)
            R = np.maximum(R, r)
        return R

    def _interp(self, arr, x):
        return np.interp(x, self.xs, arr)

    def height(self, x, z):
        b = getattr(self, 'bend', 0.0) * np.sin(x / getattr(self, 'bend_len', 6.0))
        zf = self.z_far + self._interp(self.dfar, x) + b
        zn = self.z_near + self._interp(self.dnear, x) + b
        y = np.zeros(np.broadcast(x, z).shape)
        inside = (z > zn) & (z < zf)
        s = np.clip((z - zn) / np.maximum(zf - zn, 1e-3), 0, 1)
        sill = 1.0
        for (xc, half, st) in getattr(self, 'riffles', []):
            u = self.lip_u(x, z, xc)
            sill = sill * (1 - 0.7 * st * np.exp(-(np.maximum(u, -u * 3) / half) ** 2))
        bed = -self.depth * sill * np.clip(4 * s * (1 - s), 0, 1) ** 1.6 - 0.01
        y = np.where(inside, bed, y)
        # far bank
        kind = self.far[0]
        hb = self.far[1] * self._interp(self.hfar, x)
        d = z - zf
        if kind == 'slope':
            run = self.far[2]
            yf = np.clip(d / run, 0, 1) * hb
        else:  # cut: a near-vertical face of 0.08 m run
            yf = np.clip(d / 0.08, 0, 1) * hb
        yf = yf + np.maximum(d - 1.0, 0) * self.bar_rise
        y = np.where(z >= zf, yf, y)
        # near bank (rises toward the viewer)
        kind = self.near[0]
        hn = self.near[1]
        d = zn - z
        if kind == 'slope':
            yn = np.clip(d / self.near[2], 0, 1) * hn
        else:
            yn = np.clip(d / 0.08, 0, 1) * hn
        y = np.where(z <= zn, yn, y)
        return y

    def material(self, x, z, y):
        """0 gravel, 1 grass (set by subclasses / params)."""
        return np.zeros(np.shape(y), int)


# ----------------------------------------------------------------------------- raycaster -------
def raycast(cam, terrain, z_min=2.0, z_max=400.0, nz=2400, thread_frac=0.05, thread_z=25.0):
    """per pixel: kind (0 sky, 1 ground, 2 water), z, x, y, facing (slope of the hit surface toward
    the camera, dy/dz), depth (water), and for water: refl (0 sky, 1 ground), refl_row, refl_z,
    refl_y, graze (rad)."""
    W, H = cam.W, cam.H
    zs = np.geomspace(z_min, z_max, nz)
    rows = np.arange(H) + 0.5
    s = (rows - cam.hz) / cam.f                                  # ray drop per metre (slope)
    G = {k: np.zeros((H, W)) for k in ('z', 'x', 'y', 'facing', 'depth', 'refl_row', 'refl_z', 'refl_y', 'graze')}
    G['kind'] = np.zeros((H, W), int)
    G['refl'] = np.zeros((H, W), int)
    G['thread'] = np.zeros((H, W), bool)
    for c in range(W):
        xz = cam.x_of(c + 0.5, zs)
        yt = terrain.height(xz, zs)                              # ground height along this column
        ys = np.maximum(yt, 0.0)                                 # the surface seen from above
        ray = cam.h - s[:, None] * zs[None, :]                   # H x nz ray heights
        below = ray <= ys[None, :]
        hitany = below.any(1) & (s > 0)
        k = np.argmax(below, 1)
        # refine: linear interpolation between k-1 and k
        k0 = np.maximum(k - 1, 0)
        d0 = ray[np.arange(H), k0] - ys[k0]; d1 = ray[np.arange(H), k] - ys[k]
        t = np.clip(d0 / np.maximum(d0 - d1, 1e-9), 0, 1)
        zh = zs[k0] + t * (zs[k] - zs[k0])
        yh = np.interp(zh, zs, ys)
        wet = np.interp(zh, zs, yt) < 0
        fac = np.gradient(ys, zs)[k]
        G['z'][:, c] = np.where(hitany, zh, np.inf)
        G['x'][:, c] = cam.x_of(c + 0.5, G['z'][:, c])
        G['y'][:, c] = yh
        G['facing'][:, c] = fac
        G['kind'][:, c] = np.where(hitany, np.where(wet, 2, 1), 0)
        G['depth'][:, c] = np.where(wet, -np.interp(zh, zs, yt), 0)
        # sub-pixel threads (V02): far away a row spans tens of metres of ground, so a channel a few
        # metres wide is missed by a single ray. Ferrari draws it anyway, one row tall. Mark a
        # ground pixel as water when water covers enough of its footprint.
        if thread_frac > 0:
            s_lo = (rows - 0.5 - cam.hz) / cam.f; s_hi = (rows + 0.5 - cam.hz) / cam.f
            zfar = np.where(s_lo > 0, cam.h / np.maximum(s_lo, 1e-6), 1e9); znear = cam.h / np.maximum(s_hi, 1e-6)
            iw = np.searchsorted(zs, znear); iw2 = np.searchsorted(zs, zfar)
            wetc = np.r_[0, np.cumsum(yt < -0.005)]
            frac = (wetc[np.minimum(iw2, len(zs))] - wetc[np.minimum(iw, len(zs))]) / np.maximum(iw2 - iw, 1)
            fl = np.r_[frac[1:], 0]; fu = np.r_[0, frac[:-1]]
            th = hitany & ~wet & (frac > thread_frac) & (zh > thread_z) & (frac >= fl) & (frac > fu)
            G['thread'][:, c] = th
            wet = wet | th
            G['kind'][:, c] = np.where(hitany, np.where(wet, 2, 1), 0)
            G['depth'][:, c] = np.where(th, 0.2, G['depth'][:, c])
            G['y'][:, c] = np.where(th, 0.0, G['y'][:, c])
        # reflections for water pixels
        wrows = np.nonzero(hitany & wet)[0]
        thr_c = G['thread'][:, c]
        for r in wrows:
            if thr_c[r]:                    # Ferrari's far threads are always the sky ramp
                G['graze'][r, c] = np.arctan(s[r]); G['refl'][r, c] = 0
                continue
            z0 = zh[r]
            g = s[r]                                             # tan of the grazing angle
            G['graze'][r, c] = np.arctan(g)
            mask = zs > z0
            rr = (zs[mask] - z0) * g
            hit = rr <= ys[mask]
            hit &= ys[mask] > 0.004
            if hit.any():
                j = np.argmax(hit)
                zm, ym = zs[mask], ys[mask]
                if j > 0:
                    e0 = rr[j - 1] - ym[j - 1]; e1 = rr[j] - ym[j]
                    tt = np.clip(e0 / max(e0 - e1, 1e-9), 0, 1)
                    zr = zm[j - 1] + tt * (zm[j] - zm[j - 1]); yr = (zr - z0) * g
                else:
                    zr = zm[j]; yr = ym[j]
                G['refl'][r, c] = 1
                G['refl_z'][r, c] = zr; G['refl_y'][r, c] = yr
                G['refl_row'][r, c] = cam.row_of(zr, yr)
            else:
                G['refl'][r, c] = 0
    # a thread is a horizontal run (V02): drop marks that don't make a run of >= 3 px in their row
    T = G['thread']
    if T.any():
        keep = np.zeros_like(T)
        for r in range(H):
            x = 0
            while x < W:
                if T[r, x]:
                    e = x
                    while e < W and (T[r, e] or (G['kind'][r, e] == 2 and not T[r, e])):
                        e += 1
                    if e - x >= 3:
                        keep[r, x:e] |= T[r, x:e]
                    x = e
                else:
                    x += 1
        # vary the run pattern: bridge some short gaps so a thread never reads as evenly dotted
        rng_t = np.random.default_rng(7)
        for r in range(H):
            xs_ = np.nonzero(keep[r])[0]
            for a_, b_ in zip(xs_[:-1], xs_[1:]):
                if 1 < b_ - a_ <= 4 and rng_t.random() < 0.6:
                    keep[r, a_:b_] = True
                    G['kind'][r, a_:b_] = 2; G['depth'][r, a_:b_] = 0.2; G['refl'][r, a_:b_] = 0
                    G['graze'][r, a_:b_] = np.arctan((r + 0.5 - cam.hz) / cam.f)
        drop = T & ~keep
        G['kind'][drop] = 1
        G['depth'][drop] = 0
        G['thread'] = keep
    return G


# ----------------------------------------------------------------------------- the braid ------
class BraidPlain:
    """a braided plain: threads x_k(z) that wander, split and rejoin (their union is the braid),
    each with a width w_k(z) and depth; bars between them undulate a little above the water.
    Threads run roughly along z (toward the viewer) by default; `angle` rotates the whole plain.
    height(x, z) is vectorised; distances are computed segment by segment in plan."""

    def __init__(self, seed=0, n_threads=4, z0=4.0, z1=600.0, width=(2.0, 7.0), depth=0.45,
                 spread=18.0, angle=0.0, bar_h=0.25, x_center=0.0, lam=(25.0, 90.0), turb=0.6,
                 angle_sd=40.0, keep_axial=2, lod=0.06):
        rng = np.random.default_rng(seed)
        self.depth, self.bar_h, self.angle = depth, bar_h, np.radians(angle)
        zs = np.geomspace(z0, z1, 700)
        self.threads = []
        for k in range(n_threads):
            x0 = x_center + rng.normal(0, spread * 0.5)
            lam1 = rng.uniform(*lam); lam2 = rng.uniform(lam[0] * 0.4, lam[0])
            a1 = rng.uniform(0.3, 1.0) * spread * 0.5; a2 = a1 * 0.3
            ph1, ph2 = rng.uniform(0, 6.3, 2)
            # amplitude grows with distance a little (the plain widens toward the horizon)
            x = x0 + (a1 * np.sin(zs / lam1 + ph1) + a2 * np.sin(zs / lam2 + ph2)) * (1 + zs / 300.0)
            w = rng.uniform(*width) * (0.3 + 0.7 * np.abs(np.sin(zs / rng.uniform(12, 45) + rng.uniform(0, 6)))) + 0.3
            # some threads dry up for a stretch (they split off and rejoin)
            gate = np.ones_like(zs)
            if rng.random() < 0.6:
                zc = np.exp(rng.uniform(np.log(z0 * 3), np.log(z1 * 0.5)))
                gate = np.clip(np.abs(np.log(zs / zc)) / 0.25, 0, 1) if rng.random() < 0.5 else gate
            # level of detail: far away one screen row spans tens of metres, so meanders shorter than
            # that are averaged into a straight 'belt' as wide as they swing (no aliasing ladders)
            cs = np.r_[0, np.cumsum(x)]; cs2 = np.r_[0, np.cumsum(x * x)]
            kk = np.clip((lod * zs).astype(int), 0, 200)
            i0 = np.clip(np.arange(len(zs)) - kk, 0, len(zs) - 1); i1 = np.clip(np.arange(len(zs)) + kk + 1, 1, len(zs))
            n_ = (i1 - i0).astype(float)
            mu = (cs[i1] - cs[i0]) / n_; var = np.maximum((cs2[i1] - cs2[i0]) / n_ - mu ** 2, 0)
            x = mu; w = w + 0.9 * np.sqrt(var) * (kk > 0)
            th = np.radians(rng.normal(0, angle_sd)) if k >= keep_axial else 0.0
            zc = np.exp(rng.uniform(np.log(z0 * 2), np.log(z1 * 0.4)))
            self.threads.append((zs, x, w * gate, th, zc))
        self.bar_rng = rng.integers(0, 10 ** 6)
        # relic channels on the bars: old dry threads a few cm lower, damp and darker (the streaky
        # texture of every braid bar seen from above)
        self.relics = []
        for k in range(n_threads):
            x0 = x_center + rng.normal(0, spread * 0.5)
            lam1 = rng.uniform(*lam)
            a1 = rng.uniform(0.3, 1.0) * spread * 0.4
            x = x0 + a1 * np.sin(zs / lam1 + rng.uniform(0, 6.3)) * (1 + zs / 300.0)
            w = rng.uniform(1.5, 5.0) * (0.3 + 0.7 * np.abs(np.sin(zs / rng.uniform(10, 30))))
            self.relics.append((zs, x, w))

    def _rot(self, x, z):
        if self.angle == 0:
            return x, z
        c, s = np.cos(self.angle), np.sin(self.angle)
        zm = 40.0
        return c * x - s * (z - zm), s * x + c * (z - zm) + zm

    def channel_depth(self, x, z):
        x, z = self._rot(x, z)
        d = np.zeros(np.broadcast(x, z).shape)
        for (zs, xs, ws, th, zc) in self.threads:
            if th != 0.0:                       # an oblique thread: work in its own rotated frame
                c_, s_ = np.cos(th), np.sin(th)
                xr = c_ * x - s_ * (z - zc)
                zr = s_ * x + c_ * (z - zc) + zc
            else:
                xr, zr = x, z
            xc = np.interp(zr, zs, xs, left=np.nan, right=np.nan); wc = np.interp(zr, zs, ws, left=0, right=0)
            xc = np.where(np.isnan(xc), 1e9, xc)
            # horizontal distance to the thread's centre, corrected for its slope in plan
            dxdz = np.gradient(xs, zs)
            sl = np.interp(zr, zs, dxdz)
            dist = np.abs(xr - xc) / np.sqrt(1 + sl ** 2)
            u = np.clip(1 - (dist / np.maximum(wc / 2, 1e-3)) ** 2, 0, 1)
            d = np.maximum(d, self.depth * u ** 1.6 * np.clip(wc / 3.0, 0.3, 1.0))
        return d

    def height(self, x, z):
        d = self.channel_depth(x, z)
        # bars: gentle undulation above the water line; channels cut below it
        bar = self.bar_h * (0.6 + 0.25 * np.sin(0.21 * x + 0.05 * z) * np.cos(0.13 * z - 0.3 * x))
        rel = np.zeros_like(d)
        for (zs, xs, ws) in getattr(self, 'relics', []):
            xr, zr = self._rot(x, z) if self.angle else (x, z)
            xc = np.interp(zr, zs, xs); wc = np.interp(zr, zs, ws)
            u = np.clip(1 - (np.abs(xr - xc) / np.maximum(wc / 2, 1e-3)) ** 2, 0, 1)
            rel = np.maximum(rel, u)
        bar = bar - 0.6 * bar * rel                     # a relic channel floor: just above the water
        y = bar * (1 - np.clip(d / 0.07, 0, 1)) - d
        # a narrow sloping margin: bars fall to the water over ~0.6 m (the wet foot lives here)
        return y


def designed_channel(z0=8.0, length=60.0, x0=0.0, width=5.0, depth=0.45, bar_h=0.25, split=True, lateral=6.0):
    """one channel as a single form: an S-bend that narrows downstream (toward the viewer), splits
    around a lozenge bar and rejoins. Flows from far (z0+length) to near (z0)."""
    B = BraidPlain.__new__(BraidPlain)
    B.depth, B.bar_h, B.angle = depth, bar_h, 0.0
    zs = np.linspace(z0 - 5, z0 + length + 5, 600)
    t = (zs - z0) / length                                   # 0 near .. 1 far
    xc = x0 + lateral * np.sin(2.4 * np.pi * t + 0.6) * (0.6 + 0.8 * t)
    w = width * (0.55 + 0.6 * t) * (1 + 0.25 * np.sin(9 * t))   # narrows toward the viewer, pinches
    threads = []
    if split:
        # between t=0.35 and 0.6 the flow divides around a bar: two arms bowing apart, then rejoining
        env = np.clip(np.sin(np.pi * np.clip((t - 0.33) / 0.3, 0, 1)), 0, 1)
        sep = 0.95 * w * env
        wa = np.where(env > 0.05, w * (0.5 + 0.12 * (1 - env)), w)
        threads.append((zs, xc - sep, wa, 0.0, 0.0))
        threads.append((zs, xc + sep * 1.15, wa * np.where(env > 0.05, 0.8, 1.0), 0.0, 0.0))
    else:
        threads.append((zs, xc, w, 0.0, 0.0))
    B.threads = threads
    return B
