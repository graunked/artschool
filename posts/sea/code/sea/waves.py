"""Swell over real bathymetry: linear dispersion, ray refraction (Snell), shoaling, ray-tube spreading,
random-wave breaking (Battjes & Janssen 1978). Output fields on the 1 m grid.

Grid convention: arrays [row, col], col = x east, row = y SOUTH (image order). cell = 1 m unless given.
"""
import numpy as np
from scipy import ndimage as nd
G = 9.81

def wavenumber(omega, d):
    """solve omega^2 = g k tanh(k d) (vectorised, Newton on a good start)."""
    d = np.maximum(np.asarray(d, float), 0.05)
    k0 = omega ** 2 / G
    x = k0 * d
    kd = np.where(x < 1, np.sqrt(x) * (1 + x / 6), x)       # start
    kd = np.maximum(kd, 1e-4)
    for _ in range(12):
        t = np.tanh(kd); f = kd * t - x; df = t + kd * (1 - t * t)
        kd = kd - f / df
    return kd / d

def speeds(omega, d):
    k = wavenumber(omega, d); dd = np.maximum(d, 0.05); kd = k * dd
    c = omega / k
    n = 0.5 * (1 + np.where(kd < 20, 2 * kd / np.sinh(np.minimum(2 * kd, 40)), 0))
    return k, c, n * c

def bilinear(A, x, y):
    h, w = A.shape
    x = np.clip(x, 0, w - 1.001); y = np.clip(y, 0, h - 1.001)
    x0 = x.astype(int); y0 = y.astype(int); tx = x - x0; ty = y - y0
    return (A[y0, x0] * (1 - tx) * (1 - ty) + A[y0, x0 + 1] * tx * (1 - ty) +
            A[y0 + 1, x0] * (1 - tx) * ty + A[y0 + 1, x0 + 1] * tx * ty)

def swell(depth, Hs=1.6, T=8.3, dir_from=281.0, gamma=0.73, alpha=1.0, ray_dx=0.5, ds=0.5,
          smooth=4.0, deep_fill=30.0, max_len=None, spread_deg=0.0, seed=0, debug=None):
    """depth: metres below still water (negative = land), 1 m grid.
    Returns dict of fields: Hrms, Qb (fraction breaking), D (dissipation, m^3/s... relative),
    kx, ky (unit propagation), phase (travel time, s, complex-averaged), hit (ray weight)."""
    h, w = depth.shape
    omega = 2 * np.pi / T
    d = np.where(np.isnan(depth), deep_fill, depth).astype(float)
    ds_ = nd.gaussian_filter(np.maximum(d, 0.05), smooth)
    land = d <= 0.05
    db_ = nd.gaussian_filter(np.maximum(d, 0.05), 1.5)   # breaking feels the reef heads
    _, C, CG = speeds(omega, ds_)
    C = np.where(land, 0.5, C)
    gy, gx = np.gradient(C)
    # propagation bearing (clockwise from north) -> grid vector (x east, y south)
    b = np.radians((dir_from + 180) % 360)
    kx0, ky0 = np.sin(b), -np.cos(b)
    # launch line: through the grid centre, perpendicular to k, set back so it starts outside the grid
    cx, cy = w / 2, h / 2
    diag = np.hypot(w, h)
    nxv, nyv = -ky0, kx0
    s = np.arange(-diag / 2, diag / 2, ray_dx)
    X = cx - kx0 * diag / 2 + nxv * s; Y = cy - ky0 * diag / 2 + nyv * s
    KX = np.full_like(X, kx0); KY = np.full_like(X, ky0)
    if spread_deg:
        rng = np.random.default_rng(seed); a = np.radians(rng.normal(0, spread_deg, X.size))
        KX, KY = KX * np.cos(a) - KY * np.sin(a), KX * np.sin(a) + KY * np.cos(a)
    n = X.size
    Hrms0 = Hs / np.sqrt(2)
    _, c0, cg0 = speeds(omega, np.array(deep_fill))
    E0 = Hrms0 ** 2 / 8
    F = np.full(n, E0 * cg0 * ray_dx)        # energy flux per tube (rho g dropped)
    alive = np.ones(n, bool); tt = np.zeros(n)
    acc = {k: np.zeros((h, w)) for k in ('wt', 'H', 'Q', 'D', 'kx', 'ky', 're', 'im')}
    steps = int((max_len or diag * 1.05) / ds)
    started = np.zeros(n, bool)
    for it in range(steps):
        inside = (X >= 0) & (X < w - 1) & (Y >= 0) & (Y < h - 1)
        started |= inside
        alive &= ~(started & ~inside)                     # left the grid after entering
        if not alive.any(): break
        dloc = np.where(inside, bilinear(ds_, X, Y), deep_fill)
        hitland = inside & (bilinear(d, X, Y) <= 0.08)
        alive &= ~hitland
        k, c, cg = speeds(omega, dloc)
        # tube width from ALIVE neighbours (a dead ray freezes in place and must not count)
        bw = np.full(n, ray_dx)
        ia = np.nonzero(alive)[0]
        if ia.size > 1:
            xa, ya = X[ia], Y[ia]
            dxa = np.hypot(np.diff(xa), np.diff(ya))
            gap = np.diff(ia)                                  # rays between them that died
            dxa = np.where(gap == 1, dxa, np.nan)              # across a dead ray: unknown
            both = np.zeros(ia.size); cnt = np.zeros(ia.size)
            ok = np.isfinite(dxa)
            both[:-1] += np.where(ok, dxa, 0); cnt[:-1] += ok
            both[1:] += np.where(ok, dxa, 0); cnt[1:] += ok
            bwa = np.where(cnt > 0, both / np.maximum(cnt, 1), ray_dx)
            bw[ia] = bwa
        bw = np.clip(bw, 0.25 * ray_dx, 8 * ray_dx)
        E = F / (np.maximum(cg, 1e-3) * bw)
        Hrms = np.sqrt(8 * np.maximum(E, 0))
        # Battjes-Janssen breaking
        dbk = np.where(inside, bilinear(db_, X, Y), deep_fill)
        Hm = 0.88 / k * np.tanh(gamma * k * dbk / 0.88)
        r = np.clip(Hrms / np.maximum(Hm, 1e-3), 0, 1.0)
        # solve (1-Q)/(-ln Q) = r^2 by fixed point
        Q = np.where(r > 0.3, r ** 4, 0.0)
        for _ in range(8):
            Q = np.clip(np.exp(-(1 - Q) / np.maximum(r * r, 1e-6)), 0, 1)
        Q = np.where(r >= 1, 1.0, Q)
        D = 0.25 * alpha * Q * Hm ** 2 / T       # rho g dropped
        m = alive & inside
        if debug is not None and it % 100 == 0 and m.any():
            debug.append((it, float(np.median(bw[m])), float(np.median(F[m])), float(np.median(Hrms[m])), float(np.median(cg[m])), float(np.median(dloc[m])), int(m.sum()), float(np.median(D[m]))))
        F = np.where(m, np.maximum(F - D * bw * ds, 0), F)
        # splat
        if m.any():
            xi = X[m].astype(int); yi = Y[m].astype(int)
            ph = omega * tt[m]
            for key, val in (('wt', 1.0), ('H', Hrms[m]), ('Q', Q[m]), ('D', D[m]), ('kx', KX[m]), ('ky', KY[m]),
                             ('re', np.cos(ph)), ('im', np.sin(ph))):
                np.add.at(acc[key], (yi, xi), val)
        # advance (only rays in deep-fill outside grid move straight)
        cc = np.where(inside, c, c0)
        gcx = np.where(inside, bilinear(gx, X, Y), 0); gcy = np.where(inside, bilinear(gy, X, Y), 0)
        dot = gcx * KX + gcy * KY
        KX2 = KX - ds / cc * (gcx - dot * KX); KY2 = KY - ds / cc * (gcy - dot * KY)
        nn = np.hypot(KX2, KY2); KX = np.where(alive, KX2 / nn, KX); KY = np.where(alive, KY2 / nn, KY)
        X = np.where(alive, X + KX * ds, X); Y = np.where(alive, Y + KY * ds, Y)
        tt = np.where(alive, tt + ds / cc, tt)
    wt = acc['wt']
    out = {}
    # normalised convolution fill (diffraction-ish smoothing into ray shadows)
    sig = 3.0
    W = nd.gaussian_filter(wt, sig)
    for key in ('H', 'Q', 'D', 'kx', 'ky', 're', 'im'):
        v = nd.gaussian_filter(acc[key], sig)
        out[key] = v / np.maximum(W, 1e-9)
    # shadows: H fades where no rays arrive (weight small relative to typical)
    typ = np.median(W[W > 0]) if (W > 0).any() else 1
    cover = np.clip(W / (0.06 * typ), 0, 1)
    Wb = nd.gaussian_filter(cover, 12)             # broad diffraction into the lee
    out['H'] = out['H'] * cover + nd.gaussian_filter(out['H'] * cover, 12) / np.maximum(Wb, 1e-6) * Wb * (1 - cover) * 0.6
    out['Q'] *= cover; out['D'] *= cover
    nk = np.hypot(out['kx'], out['ky']) + 1e-9
    out['kx'] /= nk; out['ky'] /= nk
    out['phase'] = np.angle(out['re'] + 1j * out['im'])      # radians of omega * travel time
    out['cover'] = cover; out['hit'] = wt
    for key in list(out):
        out[key] = np.where(land, 0, out[key]).astype(np.float32)
    out['land'] = land
    out['T'] = T; out['Hs'] = Hs
    return out

def eikonal(depth, T=8.3, dir_from=281.0, cell=2.0, smooth=6.0, deep_fill=30.0):
    """First-arrival travel time of a plane swell, |grad t| = 1/c, by Dijkstra on a 16-neighbour
    grid with a delayed super-source along the upwind boundary. Wraps around stacks and into lees
    (the lee gets the diffracted first arrival). Returns t (s) on the 1 m grid, and unit k (grad t)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    h, w = depth.shape
    omega = 2 * np.pi / T
    d = np.where(np.isnan(depth), deep_fill, depth).astype(float)
    ds_ = nd.gaussian_filter(np.maximum(d, 0.05), smooth)
    f = int(cell)
    dc = ds_[::f, ::f]; landc = (d[::f, ::f] <= 0.05)
    H_, W_ = dc.shape
    _, C, _ = speeds(omega, dc)
    slow = np.where(landc, 1e3, 1.0 / C)
    idx = np.arange(H_ * W_).reshape(H_, W_)
    offs = [(0, 1), (1, 0), (1, 1), (1, -1), (1, 2), (2, 1), (1, -2), (2, -1)]
    rows, cols, vals = [], [], []
    for dy, dx in offs:
        y0, y1 = max(0, -dy), H_ - max(0, dy); x0, x1 = max(0, -dx), W_ - max(0, dx)
        a = idx[y0:y1, x0:x1]; b = idx[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        L = cell * np.hypot(dx, dy)
        wgt = 0.5 * (slow[y0:y1, x0:x1] + slow[y0 + dy:y1 + dy, x0 + dx:x1 + dx]) * L
        rows += [a.ravel(), b.ravel()]; cols += [b.ravel(), a.ravel()]; vals += [wgt.ravel(), wgt.ravel()]
    # super source
    b_ = np.radians((dir_from + 180) % 360); kx0, ky0 = np.sin(b_), -np.cos(b_)
    border = np.zeros((H_, W_), bool); border[0] = border[-1] = True; border[:, 0] = border[:, -1] = True
    yy, xx = np.mgrid[0:H_, 0:W_]
    proj = (xx * kx0 + yy * ky0) * cell
    upwind = border & (proj < np.percentile(proj[border], 60)) & ~landc
    _, c0, _ = speeds(omega, np.array(deep_fill))
    src = H_ * W_
    s_nodes = idx[upwind]; s_w = (proj[upwind] - proj[upwind].min()) / c0 + 1e-3
    rows.append(np.full(s_nodes.size, src)); cols.append(s_nodes); vals.append(s_w)
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(src + 1, src + 1)).tocsr()
    t = dijkstra(A, directed=True, indices=src)[:src].reshape(H_, W_)
    t = np.where(np.isfinite(t), t, np.nan)
    tf = nd.gaussian_filter(np.nan_to_num(t, nan=np.nanmax(t)), 1.0)
    t1 = nd.zoom(tf, f, order=1)[:h, :w]
    if t1.shape != (h, w):
        t1 = np.pad(t1, ((0, h - t1.shape[0]), (0, w - t1.shape[1])), mode='edge')
    gy, gx = np.gradient(nd.gaussian_filter(t1, 2.0))
    n = np.hypot(gx, gy) + 1e-9
    return t1.astype(np.float32), (gx / n).astype(np.float32), (gy / n).astype(np.float32)
