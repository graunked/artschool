"""geo.py -- the world the painter needs, and no more.

World: Z up, ground is the plane z = 0 (gravel).  Logs are chains of round cones, cut flat
(sawn) or jagged (snapped / splintered) at their ends.  Rendered with an orthographic camera by
sphere tracing, so silhouettes of straight trunks come out as straight lines (clean runs).

Buffers produced per pixel (all float/int arrays H x W):
  obj   object id (-1 = ground)          part  0 side, 1 end face (sawn / break), 2 inner wall,
  P     world position                   N     world normal
  s     arc length along the log (m)     th    angle around the axis (0 = up)
  rho   radial distance / radius on end faces
  t     ray depth
"""
import numpy as np


def norm(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


class Camera:
    """orthographic camera: pitch = angle below horizontal (deg), yaw = compass direction the
    camera looks toward (deg, 0 = +y), scale = px per metre, look = world point at image centre."""

    def __init__(self, W, H, scale, pitch=30.0, yaw=0.0, look=(0, 0, 0)):
        self.W, self.H, self.scale = W, H, scale
        p, y = np.radians(pitch), np.radians(yaw)
        self.f = np.array([np.cos(p) * np.sin(y), np.cos(p) * np.cos(y), -np.sin(p)])
        self.r = np.array([np.cos(y), -np.sin(y), 0.0])
        self.u = np.cross(self.r, self.f)
        self.look = np.array(look, float)
        self.pitch = pitch

    def rays(self):
        jj, ii = np.meshgrid(np.arange(self.W) + 0.5, np.arange(self.H) + 0.5)
        X = (jj - self.W / 2) / self.scale
        Y = (self.H / 2 - ii) / self.scale
        O = self.look + X[..., None] * self.r + Y[..., None] * self.u - self.f * 60.0
        D = np.broadcast_to(self.f, O.shape)
        return O, D

    def project(self, P):
        P = np.atleast_2d(P) - self.look
        x = P @ self.r * self.scale + self.W / 2
        y = self.H / 2 - P @ self.u * self.scale
        return x, y


def sd_round_cone(p, a, b, r1, r2):
    """iq's exact round-cone SDF, vectorised. p: (..., 3)."""
    ba = b - a
    l2 = ba @ ba
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = p - a
    y = pa @ ba
    z = y - l2
    q = pa * l2 - y[..., None] * ba
    x2 = np.einsum('...i,...i', q, q)
    y2 = y * y * l2
    z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    d3 = (np.sqrt(np.maximum(x2 * a2 * il2, 0)) + y * rr) * il2 - r1
    d1 = np.sqrt(x2 + z2) * il2 - r2
    d2 = np.sqrt(x2 + y2) * il2 - r1
    return np.where(np.sign(z) * a2 * z2 > k, d1, np.where(np.sign(y) * a2 * y2 < k, d2, d3))


def sd_ellipsoid(p, c, R, axes):
    """approximate ellipsoid SDF (iq's bound). axes: 3x3 rows = unit axes, R = radii."""
    q = (p - c) @ axes.T
    k0 = np.linalg.norm(q / R, axis=-1)
    k1 = np.linalg.norm(q / (R * R), axis=-1)
    return k0 * (k0 - 1.0) / (k1 + 1e-9)


def cone_culled(p, d, a, b, r1, r2, margin):
    """round-cone distance evaluated only where its bounding sphere could matter (lb < d+margin);
    elsewhere the bounding-sphere distance (a valid lower bound) stands in."""
    c = 0.5 * (a + b)
    R = 0.5 * np.linalg.norm(b - a) + max(r1, r2)
    lb = np.linalg.norm(p - c, axis=-1) - R
    m = lb < d + margin
    if m.any():
        out = lb.copy()
        out[m] = sd_round_cone(p[m], a, b, r1, r2)
        return out
    return lb


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


class Log:
    """a trunk: polyline of axis points pts (n x 3) with radii rad (n).  ends: 'sawn' | 'snapped' |
    'splinter' for the start (0) and finish (1) ends.  stubs: list of (s_frac, theta, length,
    radius_frac, tilt) branch stubs.  rootwad: None or dict(r=..., depth=..., end=0)."""

    def __init__(self, pts, rad, ends=('sawn', 'snapped'), stubs=(), rootwad=None, seed=0,
                 kind='drift', wet=0.0, knots=()):
        self.pts = np.asarray(pts, float)
        self.knots = list(knots)
        self.rad = np.asarray(rad, float)
        self.ends = ends
        self.stubs = list(stubs)
        self.rootwad = rootwad
        self.seed = seed
        self.kind = kind          # 'drift' (silvered) or 'bark' (fresh)
        self.wet = wet
        seg = np.diff(self.pts, axis=0)
        self.seglen = np.linalg.norm(seg, axis=1)
        self.cum = np.concatenate([[0], np.cumsum(self.seglen)])
        self.L = self.cum[-1]
        rng = np.random.default_rng(seed)
        # jagged ends: the shell of the trunk stands out further in some angular sectors (the
        # teeth of the break), and the break surface between them falls toward the axis.
        # profile(theta) = tooth height; scaled by radial fraction so the break face is a rough
        # cone.  snapped = 3-5 broad teeth; splinter = 6-10 narrower, longer teeth plus stroke
        # spikes (drawn in screen space later, since they are < 1 px wide).
        self.jag = []
        self.spikes = []        # (base point, direction, length, end) for stroke splinters
        ang = np.linspace(0, 2 * np.pi, 256, endpoint=False)
        for e in (0, 1):
            kind_e = ends[e]
            prof = np.zeros(256)
            r_end = self.rad[0 if e == 0 else -1]
            if kind_e in ('snapped', 'splinter'):
                nt = rng.integers(3, 6) if kind_e == 'snapped' else rng.integers(6, 10)
                for _ in range(nt):
                    cang = rng.uniform(0, 2 * np.pi)
                    w = rng.uniform(0.5, 1.1) if kind_e == 'snapped' else rng.uniform(0.35, 0.7)
                    h = r_end * (rng.uniform(0.35, 1.0) if kind_e == 'snapped' else rng.uniform(0.6, 2.0))
                    dd = np.abs(np.angle(np.exp(1j * (ang - cang))))
                    prof = np.maximum(prof, h * np.clip(1 - dd / w, 0, 1))
                c, out = self.ends_frames()[e]
                up = np.array([0, 0, 1.0])
                e1 = norm(np.cross(up, out)); e2 = np.cross(out, e1)
                if kind_e == 'splinter':
                    for _ in range(rng.integers(4, 9)):
                        th = rng.uniform(0, 2 * np.pi); rr = r_end * rng.uniform(0.3, 0.95)
                        base = c + rr * (np.cos(th) * e1 + np.sin(th) * e2)
                        k = int(th / (2 * np.pi) * 256) % 256
                        ln = prof[k] * rr / r_end + r_end * rng.uniform(0.5, 1.6)
                        dirv = norm(out + rng.normal(0, 0.12, 3))
                        self.spikes.append((base, dirv, ln, e))
            lip = 1.0 + np.abs(np.diff(np.concatenate([prof, prof[:1]]))).max() / (2 * np.pi / 256) / max(r_end, 1e-6)
            self.jag.append((prof, lip))
        # knot swellings: a small sphere set into the surface at each knot (the outline bulges)
        self.knot_prims = []
        for (ks, kth, kr_s, kr_th) in self.knots:
            p0, ax, r0, (e1, e2) = self.frame_at(ks)
            n = np.cos(kth) * e2 + np.sin(kth) * e1
            self.knot_prims.append((p0 + n * r0 * 0.78, r0 * 0.36))
        # limbs: branch stubs and roots, as round-cone segments (a, b, r1, r2, kind)
        self.limbs = []
        self.rootlets = []      # thin root strands (3-D polylines), drawn as strokes
        self.disc = None
        for (sf, th, ln, rf, tilt) in self.stubs:
            p0, ax, r0, (e1, e2) = self.frame_at(sf * self.L)
            dirv = norm(np.cos(th) * e2 + np.sin(th) * e1 + ax * tilt)
            a = p0 + dirv * r0 * 0.3
            b = p0 + dirv * (r0 + ln)
            self.limbs.append((a, b, r0 * rf, r0 * rf * 0.55, 'stub'))
        if self.rootwad is not None:
            self._make_roots(rng)

    def _make_roots(self, rng):
        """a root wad (photo: Driftwood log on sand; braided bar): a MASS -- a flattened, rough
        disc of tangled roots and trapped grit at the butt, much darker than the silvered wood --
        with a few thick silvered roots breaking out of it radially (strokes/limbs) and a fringe of
        thin rootlets round its rim."""
        rw = self.rootwad
        e = rw.get('end', 0)
        c, out = self.ends_frames()[e]
        r0 = self.rad[0 if e == 0 else -1]
        up = np.array([0, 0, 1.0])
        e1 = norm(np.cross(up, out)); e2 = np.cross(out, e1)
        if rw.get('style', 'fan') == 'fan':
            return self._make_fan(rng, c, out, r0, e1, e2)
        rd = r0 * rw.get('disc', 2.3)
        self.disc = (c + out * r0 * 0.15, rd, rd * rw.get('thick', 0.45), out, e1, e2)
        # thick roots: radial, from inside the disc out past its rim, then sweeping back
        n = rw.get('n', 6)
        base_phi = rng.uniform(0, 2 * np.pi)
        for i in range(n):
            phi = base_phi + (i + rng.uniform(-0.3, 0.3)) / n * 2 * np.pi
            u = np.cos(phi) * e2 + np.sin(phi) * e1
            if u[2] < -0.5:
                continue                     # worn off underneath
            a = c + out * r0 * 0.1 + u * rd * 0.45
            L1 = rd * rng.uniform(0.6, 1.1)
            d1 = norm(u + out * rng.uniform(-0.1, 0.25) + rng.normal(0, 0.12, 3))
            ra = r0 * rng.uniform(0.18, 0.32)
            b = a + d1 * L1
            self.limbs.append((a, b, ra, ra * 0.6, 'root'))
            if rng.random() < 0.6:
                d2 = norm(d1 - out * rng.uniform(0.2, 0.7) + rng.normal(0, 0.25, 3))
                cpt = b + d2 * L1 * rng.uniform(0.3, 0.6)
                self.limbs.append((b, cpt, ra * 0.6, ra * 0.3, 'root'))
                self.rootlets.append(np.array([cpt, cpt + norm(d2 + rng.normal(0, 0.4, 3)) * r0 * 0.5]))
        # the fringe: rootlets round the disc rim, radiating outward, 1-2 segments
        nf = rw.get('fringe', 90)
        for i in range(nf):
            phi = rng.uniform(0, 2 * np.pi)
            u = np.cos(phi) * e2 + np.sin(phi) * e1
            if u[2] < -0.4:
                continue
            lob = 1 + 0.16 * np.sin(3 * phi + self.seed) + 0.1 * np.sin(5 * phi + 2 * self.seed) + 0.06 * np.sin(9 * phi)
            a = self.disc[0] + u * rd * (1 + (lob - 1) * 0.8) * rng.uniform(0.9, 1.0) + out * rng.uniform(-0.3, 0.3) * self.disc[2]
            d = norm(u + rng.normal(0, 0.35, 3))
            p1 = a + d * rd * rng.uniform(0.18, 0.5)
            pts = [a, p1]
            if rng.random() < 0.5:
                pts.append(p1 + norm(d + rng.normal(0, 0.6, 3)) * rd * rng.uniform(0.08, 0.2))
            self.rootlets.append(np.array(pts))

    def _make_fan(self, rng, c, out, r0, e1, e2):
        """the silvered root fan (critic r8; Upturned root wad, Driftwood on Gravel bar): the SAME
        weathered wood as the trunk, which flares continuously into 5-8 thick roots that radiate,
        twist and taper, with big voids between them where the gravel shows through.  Each root
        is 2-3 round-cone segments, so it carries the trunk's own five classes.  Roots that point
        into the gravel are worn off short; thin tips end in 1-px rootlets."""
        rw = self.rootwad
        n = max(rw.get('n', 9), 8)
        base_phi = rng.uniform(0, 2 * np.pi)
        for i in range(n):
            phi = base_phi + (i + rng.uniform(-0.3, 0.3)) / n * 2 * np.pi
            u = np.cos(phi) * e2 + np.sin(phi) * e1
            down = u[2] < -0.35
            a = c - out * r0 * rng.uniform(0.3, 0.8) + u * r0 * 0.75
            ra = r0 * rng.uniform(0.3, 0.45)
            d = norm(u + out * rng.uniform(-0.1, 0.25) + rng.normal(0, 0.1, 3))
            nseg = 1 if down else int(rng.integers(2, 4))
            L = r0 * rng.uniform(1.0, 1.6) * (0.4 if down else 1.0)
            r1 = ra
            for sgi in range(nseg):
                b = a + d * L
                r2 = r1 * rng.uniform(0.5, 0.7)
                self.limbs.append((a, b, r1, r2, 'root'))
                # twist: turn about the axis direction and sweep back toward the trunk
                d = norm(d + np.cross(out, d) * rng.normal(0, 0.45) - out * rng.uniform(0.1, 0.5)
                         + np.array([0, 0, -0.15]))
                a, r1, L = b, r2, L * rng.uniform(0.6, 0.9)
            if not down and r1 > r0 * 0.06:
                pts = [a, a + norm(d + rng.normal(0, 0.4, 3)) * r0 * rng.uniform(0.3, 0.7)]
                self.rootlets.append(np.array(pts))

    def frame_at(self, s):
        i = int(np.clip(np.searchsorted(self.cum, s) - 1, 0, len(self.seglen) - 1))
        f = (s - self.cum[i]) / self.seglen[i]
        p = self.pts[i] * (1 - f) + self.pts[i + 1] * f
        r = self.rad[i] * (1 - f) + self.rad[i + 1] * f
        ax = (self.pts[i + 1] - self.pts[i]) / self.seglen[i]
        up = np.array([0, 0, 1.0])
        e1 = norm(np.cross(up, ax))     # horizontal side
        e2 = np.cross(ax, e1)            # 'up' around the axis
        return p, ax, r, (e1, e2)

    def ends_frames(self):
        a0 = norm(self.pts[1] - self.pts[0])
        a1 = norm(self.pts[-1] - self.pts[-2])
        return (self.pts[0], -a0), (self.pts[-1], a1)

    def local(self, p):
        """(s, theta, rho, surface-dist) of points p relative to the nearest primitive (main
        segments, then limbs).  Also sets self._T (axis tangent) and self._R (local radius);
        s = -1 on limbs."""
        shp = p.shape[:-1]
        best = np.full(shp, np.inf)
        S = np.zeros(shp); TH = np.zeros(shp); RR = np.zeros(shp)
        TT = np.zeros(p.shape); RL = np.ones(shp)
        up = np.array([0, 0, 1.0])
        segs = [(self.pts[i], self.pts[i + 1], self.rad[i], self.rad[i + 1], self.cum[i], True)
                for i in range(len(self.seglen))]
        segs += [(a, b, r1, r2, -1.0, False) for (a, b, r1, r2, kind) in self.limbs]
        for (a, b, r1, r2, s0, main) in segs:
            ln = np.linalg.norm(b - a)
            ax = (b - a) / ln
            tr = (p - a) @ ax
            t = np.clip(tr, 0, ln)
            c = a + t[..., None] * ax
            v = p - c
            dax = np.linalg.norm(v, axis=-1)
            rloc = r1 + (r2 - r1) * t / ln
            dsurf = dax - rloc
            if not main:
                dsurf = dsurf + 1e-4        # ties go to the trunk
            m = dsurf < best
            e1 = norm(np.cross(up, ax)); e2 = np.cross(ax, e1)
            th = np.arctan2(v @ e1, v @ e2)
            best = np.where(m, dsurf, best)
            S = np.where(m, s0 + tr if main else -1.0, S)
            TH = np.where(m, th, TH)
            RR = np.where(m, dax / rloc, RR)
            TT = np.where(m[..., None], ax, TT)
            RL = np.where(m, rloc, RL)
        self._T = TT; self._R = RL
        return S, TH, RR, best

    def sdf_body(self, p):
        d = np.full(p.shape[:-1], np.inf)
        n = len(self.pts)
        for i in range(n - 1):
            a, b = self.pts[i].copy(), self.pts[i + 1].copy()
            ax = (b - a) / self.seglen[i]
            # extend terminal segments so the end cuts make flat / jagged faces
            if i == 0 and self.ends[0] != 'worn':
                a = a - ax * (4.0 * self.rad[0])
            if i == n - 2 and self.ends[1] != 'worn':
                b = b + ax * (4.0 * self.rad[-1])
            d = np.minimum(d, cone_culled(p, d, a, b, self.rad[i], self.rad[i + 1], 0.0))
        return d

    def cut(self, p, d):
        """intersect with the end half-spaces; broken ends raise the cut by the tooth profile,
        scaled by radial fraction.  The cut distance is divided by its Lipschitz bound so sphere
        tracing stays conservative."""
        parts = np.zeros(p.shape[:-1], int)
        for e, ((c, out), (prof, lip)) in enumerate(zip(self.ends_frames(), self.jag)):
            if self.ends[e] == 'worn':
                continue
            r_end = self.rad[0 if e == 0 else -1]
            up = np.array([0, 0, 1.0])
            e1 = norm(np.cross(up, out)); e2 = np.cross(out, e1)
            v = p - c
            th = np.arctan2(v @ e1, v @ e2)
            rho = np.sqrt((v @ e1) ** 2 + (v @ e2) ** 2) / r_end
            idx = ((th % (2 * np.pi)) / (2 * np.pi) * len(prof)).astype(int) % len(prof)
            lim = prof[idx] * np.clip(rho, 0, 1) ** 1.5
            dc = (v @ out - lim) / (lip * 1.6 if prof.max() > 0 else 1.0)
            parts = np.where(dc > d, e + 1, parts)
            d = np.maximum(d, dc)
        return d, parts

    def sdf(self, p, with_parts=False):
        d = self.sdf_body(p)
        d, parts = self.cut(p, d)
        for (c, rk) in self.knot_prims:
            dk = np.linalg.norm(p - c, axis=-1) - rk
            d = smin(d, dk, rk * 0.8)
        if self.disc is not None:
            cc, rd, th_, out, e1, e2 = self.disc
            axes = np.stack([out, e1, e2])
            # lobed rim: radius modulated round the disc (a lumpy mass, not a shield)
            v = p - cc
            phi = np.arctan2(v @ e1, v @ e2)
            lob = 1 + 0.16 * np.sin(3 * phi + self.seed) + 0.1 * np.sin(5 * phi + 2 * self.seed) + 0.06 * np.sin(9 * phi)
            dd = sd_ellipsoid(p, cc, np.array([th_, rd, rd * 0.92]), axes)
            dd = (dd - rd * (lob - 1) * 0.8) / 1.5
            parts = np.where(dd < d, 4, parts)
            d = smin(d, dd, rd * 0.15)
        for (a, b, r1, r2, kind) in self.limbs:
            ds = cone_culled(p, d, a, b, r1, r2, r1 * 0.5)
            parts = np.where(ds < d, 3, parts)
            d = smin(d, ds, r1 * 0.5)
        if with_parts:
            return d, parts
        return d

    def bound(self):
        lo = self.pts.min(0) - self.rad.max() * 4
        hi = self.pts.max(0) + self.rad.max() * 4
        for (a, b, r1, r2, kind) in self.limbs:
            lo = np.minimum(lo, np.minimum(a, b) - r1); hi = np.maximum(hi, np.maximum(a, b) + r1)
        for pl in self.rootlets:
            lo = np.minimum(lo, pl.min(0)); hi = np.maximum(hi, pl.max(0))
        if self.disc is not None:
            lo = np.minimum(lo, self.disc[0] - self.disc[1]); hi = np.maximum(hi, self.disc[0] + self.disc[1])
        c = 0.5 * (lo + hi)
        return c, np.linalg.norm(hi - lo) * 0.5


def trace(cam, logs, steps=200):
    """sphere-trace all logs; ground plane analytic.  Returns dict of buffers."""
    O, D = cam.rays()
    H, W = cam.H, cam.W
    d = cam.f
    # ground hit
    tg = -O[..., 2] / d[2]
    tbest = tg.copy()
    obj = np.full((H, W), -1)
    part = np.zeros((H, W), int)
    for k, lg in enumerate(logs):
        c, R = lg.bound()
        # screen bbox of the bounding sphere
        sx, sy = cam.project(c)
        rp = R * cam.scale + 2
        x0, x1 = int(max(0, sx[0] - rp)), int(min(W, sx[0] + rp + 1))
        y0, y1 = int(max(0, sy[0] - rp)), int(min(H, sy[0] + rp + 1))
        if x0 >= x1 or y0 >= y1:
            continue
        o = O[y0:y1, x0:x1].reshape(-1, 3)
        # start at the bounding sphere entry
        oc = o - c
        bq = oc @ d
        cq = np.einsum('ij,ij->i', oc, oc) - R * R
        disc = bq * bq - cq
        ok = disc > 0
        t = np.where(ok, -bq - np.sqrt(np.maximum(disc, 0)), np.inf)
        tmax = np.minimum(np.where(ok, -bq + np.sqrt(np.maximum(disc, 0)), -np.inf),
                          tbest[y0:y1, x0:x1].reshape(-1))
        act = ok & (t < tmax)
        hit = np.zeros(len(o), bool)
        eps = 0.2 / cam.scale
        for _ in range(steps):
            if not act.any():
                break
            ia = np.nonzero(act)[0]
            p = o[ia] + t[ia, None] * d
            dist = lg.sdf(p)
            t[ia] += dist * 0.8
            h = dist < eps
            hit[ia[h]] = True
            act[ia[h]] = False
            far = t[ia] > tmax[ia]
            act[ia[far]] = False
        tt = np.where(hit, t, np.inf).reshape(y1 - y0, x1 - x0)
        m = tt < tbest[y0:y1, x0:x1]
        tbest[y0:y1, x0:x1] = np.where(m, tt, tbest[y0:y1, x0:x1])
        obj[y0:y1, x0:x1] = np.where(m, k, obj[y0:y1, x0:x1])
    P = O + tbest[..., None] * d
    N = np.zeros_like(P); N[..., 2] = 1.0
    s = np.zeros((H, W)); th = np.zeros((H, W)); rho = np.zeros((H, W))
    T = np.zeros((H, W, 3)); T[..., 0] = 1
    rad = np.zeros((H, W))
    for k, lg in enumerate(logs):
        m = obj == k
        if not m.any():
            continue
        p = P[m]
        e = 0.3 / cam.scale
        g = np.stack([lg.sdf(p + np.array([e, 0, 0])) - lg.sdf(p - np.array([e, 0, 0])),
                      lg.sdf(p + np.array([0, e, 0])) - lg.sdf(p - np.array([0, e, 0])),
                      lg.sdf(p + np.array([0, 0, e])) - lg.sdf(p - np.array([0, 0, e]))], -1)
        N[m] = norm(g)
        _, pr = lg.sdf(p, with_parts=True)
        part[m] = pr
        S, TH, RR, _ = lg.local(p)
        s[m], th[m], rho[m] = S, TH, RR
        T[m] = lg._T; rad[m] = lg._R
    return dict(obj=obj, part=part, P=P, N=N, T=T, rad=rad, s=s, th=th, rho=rho, t=tbest, cam=cam, logs=logs)


def shadow(buf, sun, logs, steps=48):
    """cast shadows: march from each surface point toward the sun against all logs."""
    P = buf['P']; H, W = P.shape[:2]
    cam = buf['cam']
    L = norm(np.asarray(sun, float))
    sh = np.zeros((H, W), bool)
    p0 = (P + buf['N'] * (0.6 / cam.scale)).reshape(-1, 3)
    flat = sh.reshape(-1)
    for k, lg in enumerate(logs):
        c, R = lg.bound()
        # candidate points: those whose sun-ray passes within R of c
        oc = p0 - c
        bq = oc @ L
        cq = np.einsum('ij,ij->i', oc, oc) - R * R
        disc = bq * bq - cq
        cand = (disc > 0) & ((-bq + np.sqrt(np.maximum(disc, 0))) > 0) & ~flat
        cand &= buf['obj'].reshape(-1) != k      # no self-shadow acne: a log's own form shadow
                                                  # is the class rule's job
        if not cand.any():
            continue
        ia = np.nonzero(cand)[0]
        o = p0[ia]
        t = np.maximum(0, -bq[ia] - np.sqrt(disc[ia]))
        tmax = -bq[ia] + np.sqrt(disc[ia])
        act = np.ones(len(ia), bool)
        hit = np.zeros(len(ia), bool)
        for _ in range(steps):
            if not act.any():
                break
            jb = np.nonzero(act)[0]
            dist = lg.sdf(o[jb] + t[jb, None] * L)
            t[jb] += np.maximum(dist * 0.9, 0.05 / cam.scale)
            h = dist < 0.15 / cam.scale
            hit[jb[h]] = True; act[jb[h]] = False
            act[jb[t[jb] > tmax[jb]]] = False
        flat[ia[hit]] = True
    # a surface facing away is not 'cast shadow' but form shadow; keep both separately
    return sh


def occlusion(buf, logs, radius_m):
    """cheap contact occlusion for ground points: distance to nearest log surface."""
    P = buf['P']
    dmin = np.full(P.shape[:2], np.inf)
    g = buf['obj'] == -1
    p = P[g]
    dd = np.full(len(p), np.inf)
    for lg in logs:
        dd = np.minimum(dd, lg.sdf(p))
    dmin[g] = dd
    return dmin


def make_log(L=3.0, r=0.16, yaw=0.0, centre=(0, 0), taper=0.75, bend=0.06, sag=0.03, lumps=0.08,
             nseg=7, ends=('sawn', 'snapped'), seed=0, sink=0.015, stubs=(), rootwad=None,
             kind='drift', wet=0.0, lift=None, n_knots=None, n_stubs=0):
    """a driftwood log lying on the ground: a gently bent, tapering, lumpy axis.
    bend: lateral bow (fraction of L); sag: vertical bow; lumps: radius noise (fraction).
    The log is lowered until its lowest point touches the ground (minus sink), so a bowed log
    rests on two contacts and lifts off between them -- the belly pocket."""
    rng = np.random.default_rng(seed)
    t = np.linspace(-0.5, 0.5, nseg)
    ph = rng.uniform(0, 2 * np.pi)
    lat = bend * L * (np.sin(np.pi * (t + 0.5)) * rng.choice([-1, 1]) + 0.3 * np.sin(2 * np.pi * t + ph))
    vert = sag * L * np.sin(np.pi * (t + 0.5)) * rng.choice([-1, 1])
    rad = r * (1 + (taper - 1) * (t + 0.5)) * (1 + lumps * (rng.random(nseg) - 0.5) * 2)
    a = np.radians(yaw)
    ax = np.array([np.cos(a), np.sin(a), 0.0]); side = np.array([-np.sin(a), np.cos(a), 0.0])
    pts = np.array([centre[0], centre[1], 0.0]) + np.outer(t * L, ax) + np.outer(lat, side)
    pts[:, 2] = rad + vert
    if lift is not None:
        pts[:, 2] += np.interp(t, [-0.5, 0.5], lift)
    if rootwad is not None:
        e = rootwad.get('end', 0)
        j0, j1 = (0, 1) if e == 0 else (-1, -2)
        fl = rootwad.get('flare', 1.4 if rootwad.get('style', 'fan') == 'fan' else 1.35)
        rad[j0] *= fl; rad[j1] *= 1 + (fl - 1) * 0.35
        pts[:, 2] = rad + vert
    pts[:, 2] -= (pts[:, 2] - rad).min() + sink
    # stubs: broken branches, mostly on the upper half (downward ones break off or are buried),
    # tilted toward the top end as branches grow
    st = list(stubs)
    for _ in range(n_stubs):
        st.append((rng.uniform(0.3, 0.92), rng.uniform(-1.6, 1.6), r * rng.uniform(0.4, 2.2),
                   rng.uniform(0.28, 0.5), rng.uniform(0.3, 0.9)))
    stubs = st
    # knots: 1-3 per log, mostly on the upper/front half (theta measured from 'up')
    nk = rng.integers(1, 4) if n_knots is None else n_knots
    knots = []
    for _ in range(nk):
        ks = rng.uniform(0.15, 0.85) * L
        kth = rng.uniform(-1.9, 1.9)
        rr = r * (1 + (taper - 1) * (ks / L))
        knots.append((ks, kth, rr * rng.uniform(0.3, 0.45), rng.uniform(0.2, 0.32)))
    lg = Log(pts, rad, ends=ends, stubs=stubs, rootwad=rootwad, seed=seed, kind=kind, wet=wet,
             knots=knots)
    # anything below the gravel (roots, stubs) lifts the whole log: it rests on its lowest part
    low = min([(a[2] - r1) for (a, b, r1, r2, k) in lg.limbs] + [(b[2] - r2) for (a, b, r1, r2, k) in lg.limbs] + [0.0])
    if lg.disc is not None:
        cc, rd, th_, out, e1, e2 = lg.disc
        low = min(low, cc[2] - rd * 0.92 * abs(e2[2]) - th_ * abs(out[2]) - rd * abs(e1[2]))
    if low < -sink * 2 and rootwad is not None:
        # the butt rides up on its roots; the far end stays on the gravel (a tilt, not a lift)
        e = rootwad.get('end', 0)
        tt = np.linspace(1.0, 0.0, len(pts)) if e == 0 else np.linspace(0.0, 1.0, len(pts))
        pts = pts.copy(); pts[:, 2] += (-low - sink * 2) * tt
        lg = Log(pts, rad, ends=ends, stubs=stubs, rootwad=rootwad, seed=seed, kind=kind, wet=wet,
                 knots=knots)
    return lg


def belly(buf, logs, cam, max_gap_px=2.5):
    """ground pixels UNDER a log with a small clearance (a vertical ray from the ground meets
    the log within max_gap_px): the dark pocket where a log lifts off the gravel.  Larger gaps
    are simply ground in the log's shadow."""
    P = buf['P']; g = buf['obj'] == -1
    out = np.zeros(g.shape, bool)
    if not logs:
        return out
    p = P[g]
    hit = np.zeros(len(p), bool)
    hmax = max_gap_px / cam.scale
    for h in np.linspace(0.2, 1.0, 5) * hmax:
        q = p + np.array([0, 0, h])
        for lg in logs:
            hit |= lg.sdf(q) < 0
    out[g] = hit
    return out
