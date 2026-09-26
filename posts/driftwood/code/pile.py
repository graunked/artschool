"""pile.py -- a drift pile: logs stranded together on a gravel bar, crossing and resting on each
other.

Observed (braided_driftwood_bar.jpg, Hoh gravel bar): stranded logs lie mostly ALONG the old flow
(parallel, within ~25 deg), a minority crossing at steep angles; big trunks form the base, smaller
pieces lie on and against them; root wads point upstream; the pile thins at its edges into lone
sticks.  Logs resting on others bridge between supports and sag toward the ground between them.

Placement (physics only as far as the painting needs): logs are placed largest first; each new
log's axis height is the UPPER CONVEX HULL of its support profile -- at every point along it,
the higher of the gravel and the tops of logs already beneath -- so it rests on its highest
supports and spans the gaps between them.
"""
import numpy as np
import geo


def top_of(lg, x, y):
    """height of log lg's upper surface above ground point (x, y), or -inf if not over it."""
    best = -np.inf
    for i in range(len(lg.pts) - 1):
        a, b = lg.pts[i], lg.pts[i + 1]
        ab = b[:2] - a[:2]; L2 = ab @ ab
        t = np.clip(((np.array([x, y]) - a[:2]) @ ab) / max(L2, 1e-9), 0, 1)
        c = a + (b - a) * t
        r = lg.rad[i] + (lg.rad[i + 1] - lg.rad[i]) * t
        d = np.hypot(x - c[0], y - c[1])
        if d < r:
            best = max(best, c[2] + np.sqrt(r * r - d * d))
    return best


def upper_hull(t, h):
    pts = sorted(zip(t, h))
    hull = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    hx, hy = zip(*hull)
    return np.interp(t, hx, hy)


def settle(lg, placed, sink=0.015):
    """lower/raise lg so it rests on the ground and on the logs already placed."""
    n = len(lg.pts)
    t = lg.cum / lg.L
    # support profile at the axis points (plus midpoints for a finer hull)
    tt = np.linspace(0, 1, 4 * n)
    P = np.array([np.interp(tt, t, lg.pts[:, k]) for k in range(3)]).T
    R = np.interp(tt, t, lg.rad)
    h = R - sink
    for q in placed:
        for j in range(len(tt)):
            tp = top_of(q, P[j, 0], P[j, 1])
            if tp > -np.inf:
                h[j] = max(h[j], tp + R[j] * 0.97)
    z = upper_hull(tt, h)
    pts = lg.pts.copy()
    pts[:, 2] = np.interp(t, tt, z)
    return geo.Log(pts, lg.rad, ends=lg.ends, stubs=lg.stubs, rootwad=lg.rootwad, seed=lg.seed,
                   kind=lg.kind, wet=lg.wet, knots=lg.knots)


def make_pile(seed=0, n=9, extent=(5.0, 2.5), flow_yaw=10.0, centre=(0, 0), r_range=(0.08, 0.26),
              rootwads=0.25, stubs=True, wet_frac=0.0, crossing=0.35, field=False, grey_frac=0.25):
    rng = np.random.default_rng(seed)
    specs = []
    for i in range(n):
        r = np.exp(rng.uniform(np.log(r_range[0]), np.log(r_range[1])))
        # lengths: mostly long trunks, a third short broken pieces (critic r10)
        L = r * (rng.uniform(10, 22) if rng.random() > 0.33 else rng.uniform(3.5, 8))
        u = rng.random()
        if u < crossing:                       # crossing at any angle
            yaw = flow_yaw + rng.uniform(30, 150)
        elif u < crossing + 0.15:              # pointing at the camera: ends face the viewer
            yaw = 90 + rng.normal(0, 18)
        else:                                  # along the old flow
            yaw = flow_yaw + rng.normal(0, 14)
        if field:      # strewn across the bar, loosely clumped
            cx = centre[0] + rng.uniform(-0.5, 0.5) * extent[0]
            cy = centre[1] + rng.uniform(-0.5, 0.5) * extent[1]
        else:
            cx = centre[0] + rng.normal(0, extent[0] * 0.3)
            cy = centre[1] + rng.normal(0, extent[1] * 0.3)
        specs.append((r, L, yaw, cx, cy))
    specs.sort(key=lambda s: -s[0])          # largest first: they form the base
    placed = []
    for i, (r, L, yaw, cx, cy) in enumerate(specs):
        has_rw = rng.random() < rootwads and r > r_range[0] * 1.6
        ends = (('snapped' if has_rw else str(rng.choice(['worn', 'worn', 'sawn', 'snapped']))),
                str(rng.choice(['worn', 'snapped', 'splinter', 'worn'])))
        lg = geo.make_log(L=L, r=r, yaw=yaw, centre=(cx, cy), seed=seed * 101 + i, ends=ends,
                          rootwad=dict(n=5, end=0) if has_rw else None,
                          n_stubs=int(rng.integers(0, 4)) if stubs else 0,
                          wet=1.0 if rng.random() < wet_frac else 0.0,
                          kind='grey' if rng.random() < grey_frac else 'drift',
                          bend=rng.uniform(0.02, 0.08), taper=rng.uniform(0.6, 0.85))
        if not has_rw:
            lg = settle(lg, placed)
        else:
            # root wads keep their tilt; lift them if they sit on something
            lg2 = settle(lg, placed)
            if lg2.pts[:, 2].max() > lg.pts[:, 2].max() + 1e-3:
                lg = lg2
        placed.append(lg)
    return placed
