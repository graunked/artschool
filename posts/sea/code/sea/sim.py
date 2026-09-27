"""The sea state at one instant, from fields a simulation has: depth, bottom, swell (rays + eikonal),
an ambient current steered around land, and three advected foam quantities integrated over the last
few minutes of breaking.

   W  whitewater: rollers and fresh creamy foam (tau ~4 s)       -- Ruskin's 'thick creamy curdling foam'
   F  subsided foam: lace and streaks (tau ~60-300 s)            -- Ruskin's 'thin white coating ... oval gaps'
   A  aeration: bubble cloud under/after a break (tau ~10 s)     -- the jade halo around stacks and reefs
"""
import numpy as np
from scipy import ndimage as nd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from .waves import speeds
from .noise import fbm, hash2

def bottom_type(z, depth):
    """rock where the 1 m lidar is rough at 2-6 m scales, sand where smooth."""
    zz = np.nan_to_num(z, nan=-30.0)
    hp = zz - nd.gaussian_filter(zz, 4.0)
    rough = np.sqrt(nd.gaussian_filter(hp * hp, 3.0))
    rock = np.clip((rough - 0.08) / 0.12, 0, 1)
    return rock.astype(np.float32), rough.astype(np.float32)

def ambient_current(land, U=(0.0, 0.25), cell=4):
    """potential flow of a uniform surface current around the land (no flow through coasts).
    U = (east, south) m/s. Solved on a coarse grid; returns (u, v) on the full grid."""
    h, w = land.shape
    L = land[::cell, ::cell]; H_, W_ = L.shape
    wet = ~L
    idx = -np.ones((H_, W_), int); idx[wet] = np.arange(wet.sum())
    rows, cols, vals = [], [], []; b = np.zeros(wet.sum())
    yy, xx = np.mgrid[0:H_, 0:W_]
    phi_b = (U[0] * xx + U[1] * yy) * cell
    border = np.zeros_like(wet); border[0] = border[-1] = True; border[:, 0] = border[:, -1] = True
    for (dy, dx) in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        ny, nx = yy + dy, xx + dx
        inb = (ny >= 0) & (ny < H_) & (nx >= 0) & (nx < W_)
        nyc, nxc = np.clip(ny, 0, H_ - 1), np.clip(nx, 0, W_ - 1)
        nb_wet = inb & wet[nyc, nxc]
        m = wet & nb_wet
        rows.append(idx[m]); cols.append(idx[nyc[m], nxc[m]]); vals.append(-np.ones(m.sum()))
        rows.append(idx[m]); cols.append(idx[m]); vals.append(np.ones(m.sum()))
        # outside the grid: Dirichlet via ghost = far-field potential
        mo = wet & ~inb
        rows.append(idx[mo]); cols.append(idx[mo]); vals.append(np.ones(mo.sum()) * 2)
        np.add.at(b, idx[mo], 2 * (phi_b[mo] + (U[0] * dx + U[1] * dy) * cell))
    n = wet.sum()
    rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(np.full(n, 1e-6))   # pins isolated ponds
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    phi = np.full((H_, W_), np.nan); phi[wet] = spsolve(A, b)
    phi = np.where(wet, phi, np.nan)
    # gradient on wet cells
    pf = np.where(wet, phi, 0)
    gy, gx = np.gradient(pf, cell)
    ok = wet & np.roll(wet, 1, 0) & np.roll(wet, -1, 0) & np.roll(wet, 1, 1) & np.roll(wet, -1, 1)
    gx = np.where(ok, gx, 0); gy = np.where(ok, gy, 0)
    u = nd.zoom(gx, cell, order=1)[:h, :w]; v = nd.zoom(gy, cell, order=1)[:h, :w]
    u = np.pad(u, ((0, h - u.shape[0]), (0, w - u.shape[1])), mode='edge')
    v = np.pad(v, ((0, h - v.shape[0]), (0, w - v.shape[1])), mode='edge')
    u = nd.gaussian_filter(np.nan_to_num(u), 2); v = nd.gaussian_filter(np.nan_to_num(v), 2)
    sp = np.hypot(u, v); lim = 3 * np.hypot(*U); f_ = np.minimum(1, lim / np.maximum(sp, 1e-9)); u *= f_; v *= f_
    return np.where(land, 0, u).astype(np.float32), np.where(land, 0, v).astype(np.float32)

def advect(q, u, v, dt):
    h, w = q.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return nd.map_coordinates(q, [yy - v * dt, xx - u * dt], order=1, mode='constant', cval=0.0)

class SeaState:
    def __init__(self, depth, z, waves, T, current=(0.0, 0.25), wind=(0.0, 0.0), seed=1,
                 tau_w=6.0, tau_f=450.0, tau_a=14.0, gamma_i=0.62, n_periods=36, steps_per_T=10,
                 kelp_amount=1.0, tide=0.45, eddy=0.12, **extra):
        self.depth, self.z, self.T = depth, z, T
        self.land = waves['land'] | ~(depth > 0.02)
        h, w = depth.shape
        self.seed = seed
        d = np.nan_to_num(depth, nan=30.0)
        self.d = d
        self.db = nd.gaussian_filter(-nd.maximum_filter(-np.maximum(d, 0.02), 5), 1.0)   # reef pinnacles the 1 m lidar blurs
        omega = 2 * np.pi / T
        self.k, self.c, self.cg = speeds(omega, nd.gaussian_filter(np.maximum(d, 0.05), 4))
        self.Hrms = waves['H']; self.tt = waves['t']; self.kx = waves['kx']; self.ky = waves['ky']
        self.rock, self.rough = bottom_type(z, d)
        # currents
        u, v = ambient_current(self.land, current)
        # eddies: a divergence-free meander field (curl of a smooth stream function), ~0.1 m/s at 100-200 m.
        # Without it foam streaks are straight rulers; the photos' streaks swing in S-curves.
        psi = fbm((h, w), 160, seed + 21, octaves=2)
        gyp, gxp = np.gradient(psi)
        ue, ve = gyp, -gxp
        sc = eddy / (np.percentile(np.hypot(ue, ve), 90) + 1e-9)
        dl0 = nd.distance_transform_edt(~self.land)
        fade = np.clip(dl0 / 15.0, 0, 1)
        u = u + ue * sc * fade; v = v + ve * sc * fade
        # surface drift: wind (3%), Stokes drift and surf-zone shoreward push along k
        k = self.k
        a = self.Hrms / np.sqrt(2) / 2 * np.sqrt(2)            # ~ amplitude of an Hs/2 wave
        stokes = omega * k * a * a
        surf = 0.2 * np.clip(waves['Q'] * 4, 0, 1)             # bores carry surface foam shoreward
        self.u = u + wind[0] * 0.03 + (stokes + surf) * self.kx
        self.v = v + wind[1] * 0.03 + (stokes + surf) * self.ky
        self.u = np.where(self.land, 0, self.u); self.v = np.where(self.land, 0, self.v)
        U = float(np.hypot(*wind)); self.U = U
        self.wdir = (wind[0] / (U + 1e-9), wind[1] / (U + 1e-9))
        self.wc_cov = 3.84e-6 * U ** 3.41 if U > 3 else 0.0
        self.params = dict(tau_w=tau_w, tau_f=tau_f, tau_a=tau_a, gamma_i=gamma_i, n_periods=n_periods,
                           steps_per_T=steps_per_T, **extra)
        # along-crest coordinate (for per-crest random heights)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        kx0, ky0 = float(np.median(self.kx[~self.land])), float(np.median(self.ky[~self.land]))
        self.s_along = xx * (-ky0) + yy * kx0
        self.g_field = fbm((h, w), 40, seed + 11, octaves=3)          # slow variation of crest height along crest
        self.kelp = self.make_kelp(kelp_amount, tide)
        # surge collar: waves running against rock faces burst whatever their height relative to depth
        dl, ind = nd.distance_transform_edt(~self.land, return_indices=True)
        gyl, gxl = np.gradient(nd.gaussian_filter(dl, 1.5))
        nrm = np.hypot(gxl, gyl) + 1e-6
        facing = np.clip(-(self.kx * gxl + self.ky * gyl) / nrm, 0, 1)      # wave heading into the face
        # only steep rock surges and bursts; a beach or a low platform gets swash, not a collar
        zz = np.nan_to_num(z, nan=-30.0); gyz, gxz = np.gradient(zz)
        slope = np.hypot(gxz, gyz) * self.land
        steep = np.clip((nd.maximum_filter(slope, 7) - 0.35) / 0.6, 0, 1)
        self.collar = (np.exp(-dl / 1.8) * (0.05 + 0.95 * facing) * steep * (dl < 6)).astype(np.float32)
        self.dl = dl; self.facing = nd.gaussian_filter(facing * (dl < 12), 3.0)
        self.lobes = np.clip(0.35 + 0.9 * fbm((h, w), 14, seed + 31, octaves=2), 0, 1.4)   # pools are lopsided

    def make_kelp(self, amount, tide):
        """bull kelp canopy: on rock, 2-18 m deep, in clumpy beds; streams down-current."""
        h, w = self.d.shape
        d = self.d
        hab = self.rock * np.clip((d - 1.5) / 2.0, 0, 1) * np.clip((18 - d) / 6, 0, 1)
        exp_ = np.clip(1.2 - self.Hrms / 1.2, 0, 1)                 # the most exposed heads carry less canopy
        bed = np.clip((fbm((h, w), 70, self.seed + 3, octaves=3) - 0.1) / 0.6, 0, 1)   # discrete beds, tens of m, with gaps
        clump = fbm((h, w), 7, self.seed + 4, octaves=3)              # clumps of plants (3-8 m)
        # stretch clumps along the current: sample clump noise displaced along (u,v)
        un = np.hypot(self.u, self.v) + 1e-6
        acc = np.zeros_like(clump)
        for s in range(-3, 4):
            acc += nd.map_coordinates(clump, [np.mgrid[0:h, 0:w][0] - self.v / un * s * 1.5,
                                              np.mgrid[0:h, 0:w][1] - self.u / un * s * 1.5], order=1, mode='nearest')
        clump = acc / 7
        # expected canopy cover rho, then a mat mask whose area fraction IS rho (rank-uniformised noise)
        rho = np.clip(amount * hab * (0.55 + 0.45 * exp_) * bed, 0, 1)
        U = np.empty_like(clump); o = np.argsort(clump, axis=None); U.flat[o] = np.linspace(0, 1, clump.size)
        self.kelp_rho = rho.astype(np.float32)
        return (np.clip((rho - U) / 0.08 + 0.5, 0, 1) * (hab > 0.05)).astype(np.float32)

    def whitecaps(self, rng, dt):
        """wind-sea breaking in deep water (Beaufort 4+). Coverage from Monahan & O'Muircheartaigh (1980):
        W = 3.84e-6 U^3.41 (U = 10 m wind, m/s): 5 m/s -> 0.09%, 10 m/s -> 1%, 17 m/s -> 6%, 22 m/s -> 14%.
        Each whitecap is a short dash along the wind, 2-8 m, lasting ~tau_w."""
        h, w = self.d.shape
        P = self.params
        # steady coverage = rate * tau * area: rate per m^2 per s
        area = 6.0
        rate = self.wc_cov / (P['tau_w'] * area)
        n = rng.poisson(rate * dt * h * w)
        out = np.zeros((h, w), np.float32)
        if n == 0: return out
        x0 = rng.random(n) * w; y0 = rng.random(n) * h
        L = rng.uniform(2, 8, n)
        for f in np.linspace(-0.5, 0.5, 7):
            xs = np.clip((x0 + f * L * self.wdir[0]).astype(int), 0, w - 1)
            ys = np.clip((y0 + f * L * self.wdir[1]).astype(int), 0, h - 1)
            out[ys, xs] = 1.0
        out = np.clip(nd.gaussian_filter(out, 0.6) * 3.0, 0, 1)
        return np.where((self.d > 3) & ~self.land, out, 0)

    def crest_break(self, t):
        """q = fraction of a period since the last crest passed; H_inst of that crest; breaking mask."""
        T = self.T
        q = np.mod((t - self.tt) / T, 1.0)
        n = np.floor((t - self.tt) / T)                              # crest index
        # per-crest random multiplier, varying slowly along the crest (Rayleigh-like, mean ~1)
        cell = np.floor(self.s_along / 30.0)
        r1 = hash2(n, cell, self.seed); r2 = hash2(n, cell + 1, self.seed)
        f = (self.s_along / 30.0) - cell; f = f * f * (3 - 2 * f)
        u = r1 * (1 - f) + r2 * f
        g = np.sqrt(-np.log(np.clip(1 - u * 0.985, 1e-4, 1))) * 1.0 + 0.12 * self.g_field
        Hinst = self.Hrms * g * np.sqrt(1.0)
        Hm = self.params['gamma_i'] * self.db
        brk = (Hinst > Hm) & ~self.land
        return q, Hinst, brk

    def run(self, t_end=None, verbose=False):
        T = self.T; P = self.params
        nst = P['n_periods'] * P['steps_per_T']; dt = T / P['steps_per_T']
        t_end = t_end if t_end is not None else nst * dt
        h, w = self.d.shape
        W = np.zeros((h, w), np.float32); F = np.zeros_like(W); A = np.zeros_like(W)
        rng = np.random.default_rng(self.seed)
        # Lagrangian foam: particles shed by decaying whitewater, carried by the current, dying on an
        # exponential clock (tau_f). No numerical diffusion, so streaks stay thin and long.
        px = np.zeros(0, np.float32); py = np.zeros(0, np.float32); pa = np.zeros(0, np.float32); pl = np.zeros(0, np.float32)
        pk = np.zeros(0, np.float32)
        t0 = t_end - nst * dt
        for i in range(nst + 1):
            t = t0 + i * dt
            q, Hinst, brk = self.crest_break(t)
            passed = q * T < dt                                     # crest swept over x in this step
            age = q * T
            src = brk & passed
            if self.wc_cov > 0:
                W = np.maximum(W, self.whitecaps(rng, dt))
            csrc = passed & (self.collar * np.clip(Hinst / 0.5, 0, 1.5) > 0.35)
            # decay
            W *= np.exp(-dt / P['tau_w']); A *= np.exp(-dt / P['tau_a'])
            lost = W * (1 - np.exp(-dt / P['tau_w']))
            F = F * np.exp(-dt / P['tau_f']) + 0.9 * lost
            W = np.where(src, np.maximum(W, np.exp(-age / P['tau_w'])), W)
            A = np.where(src, np.maximum(A, 1.0), A)
            cv = np.clip(self.collar * np.clip(Hinst / 0.5, 0, 1.5), 0, 1) * np.exp(-age / P['tau_w'])
            W = np.where(csrc, np.maximum(W, cv), W)
            ca = np.clip(np.exp(-self.dl / 9.0) * (0.2 + 1.3 * self.facing) * self.lobes * np.clip(Hinst / 0.5, 0, 1.5), 0, 1)
            A = np.where(passed & (ca > 0.2), np.maximum(A, ca), A)
            # shed particles where whitewater is decaying (rate ~ lost foam), jittered inside the cell
            shed = (lost > 0.02) & ~self.land
            ys, xs = np.nonzero(shed)
            if ys.size:
                pr = np.clip(lost[ys, xs] * P.get('shed', 2.5), 0, 1)
                keep = rng.random(ys.size) < pr
                ys, xs = ys[keep], xs[keep]
                px = np.concatenate([px, xs + rng.random(xs.size).astype(np.float32)])
                py = np.concatenate([py, ys + rng.random(ys.size).astype(np.float32)])
                pa = np.concatenate([pa, np.zeros(xs.size, np.float32)])
                lt = np.where(rng.random(xs.size) < P.get('long_frac', 0.12), rng.exponential(P['tau_f'], xs.size), rng.exponential(P.get('tau_short', 25.0), xs.size))
                pl = np.concatenate([pl, lt.astype(np.float32)])
                pk = np.concatenate([pk, rng.random(xs.size).astype(np.float32)])
            if px.size:
                from .waves import bilinear as _bl
                uu = _bl(self.u, px, py); vv = _bl(self.v, px, py)
                sd = np.sqrt(2 * P.get('diff', 0.02) * dt)
                if self.U > 5:
                    # Langmuir windrows: surface convergence into lines along the wind, spacing ~ 25 m
                    nxp, nyp = -self.wdir[1], self.wdir[0]
                    sperp = px * nxp + py * nyp
                    vc = -0.004 * self.U * np.sin(2 * np.pi * sperp / P.get('windrow', 25.0))
                    uu = uu + vc * nxp; vv = vv + vc * nyp
                px = px + uu * dt + rng.normal(0, sd, px.size); py = py + vv * dt + rng.normal(0, sd, px.size)
                pa = pa + dt
                ok = (pa < pl) & (px >= 0) & (px < w - 1) & (py >= 0) & (py < h - 1)
                ok &= ~self.land[np.clip(py.astype(int), 0, h - 1), np.clip(px.astype(int), 0, w - 1)]
                px, py, pa, pl, pk = px[ok], py[ok], pa[ok], pl[ok], pk[ok]
            if i % 2 == 0:
                A = advect(A, self.u, self.v, 2 * dt)
            W = np.where(self.land, 0, W); F = np.where(self.land, 0, F); A = np.where(self.land, 0, A)
            if verbose and i % 50 == 0: print(i, nst, float(W.mean()), float(F.mean()))
        self.W, self.A = W, A
        self.parts = dict(x=px, y=py, age=pa, life=pl, k=pk)
        # foam density on the 1 m grid from the particles (for coarse views and stats)
        Fd = np.zeros((h, w), np.float32)
        np.add.at(Fd, (py.astype(int), px.astype(int)), 1.0)
        self.F = np.clip(nd.gaussian_filter(Fd, 0.7) * P.get('part_gain', 0.5), 0, 3)
        self.q, self.Hinst, self.brk = self.crest_break(t_end)
        self.t = t_end
        return self
