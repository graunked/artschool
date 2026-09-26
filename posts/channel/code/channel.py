"""channel.py -- the braided-channel painter (grows stage by stage; see LESSONS.md).

Index canvas + named palette. Materials own index ranges; the water REUSES the sky ramp (Ferrari)
plus a few water-only tints. Everything a water pixel shows is chosen by a rule about something
else: the sky at the reflected elevation, the object standing at the reflected ray, the bed seen
through it.
"""
import numpy as np
import pixkit as pk


# ---------------------------------------------------------------- palette ---------------------
class Pal:
    """named ramps -> global indices. `grey` palettes for the value stages; colour later."""

    def __init__(self, ramps):
        self.ramps = {}
        self.cols = []
        for name, cols in ramps:
            self.ramps[name] = list(range(len(self.cols), len(self.cols) + len(cols)))
            self.cols += [tuple(int(v) for v in c) for c in cols]
        self.arr = np.array(self.cols, dtype=np.uint8)

    def r(self, name):
        return np.array(self.ramps[name])

    def rgb(self, idx):
        return self.arr[idx]


def grey(*vals):
    return [(v, v, v) for v in vals]


# ---------------------------------------------------------------- camera ----------------------
class Cam:
    """pinhole, small pitch: row = hz + f*tan(elev). Ground is y=0, eye at height h."""

    def __init__(self, W, H, f, h, pitch_deg):
        self.W, self.H, self.f, self.h = W, H, f, h
        self.hz = H / 2 - f * np.tan(np.radians(pitch_deg))

    def row_of(self, z, y=0.0):
        """screen row of a point at distance z (m) and height y (m)."""
        return self.hz + self.f * (self.h - y) / z

    def mirror_row(self, z, y):
        """row where the reflection of (z, y) appears (mirror camera at -h)."""
        return self.hz + self.f * (self.h + y) / z

    def z_of_row(self, r):
        d = np.maximum(r - self.hz, 1e-3)
        return self.f * self.h / d

    def elev_of_row(self, r):
        """elevation angle (rad) of the view ray through row r (negative = below horizon)."""
        return -np.arctan((r - self.hz) / self.f)

    def x_m(self, col, z):
        return (col - self.W / 2) * z / self.f

    def col_of(self, x, z):
        return self.W / 2 + self.f * x / z


def fresnel(graze):
    """Schlick, graze in radians above the surface."""
    return 0.02 + 0.98 * (1 - np.sin(graze)) ** 5


def fresnel_art(graze):
    """the water study's artistic gain: keeps low-angle water light, as Ferrari does."""
    return np.clip(0.22 + 1.15 * fresnel(graze), 0, 1)


# ---------------------------------------------------------------- quantizers ------------------
def to_ramp(V, ramp_vals, ramp_idx, band=0.55, matrix=pk.BAYER2, ox=0, oy=0):
    """continuous value V -> indices of a ramp (values ascending or descending), mixing only
    adjacent steps, banded (holds flat, crosses as a sustained checker)."""
    vals = np.asarray(ramp_vals, float)
    order = np.argsort(vals)
    vs = vals[order]; ix = np.asarray(ramp_idx)[order]
    pos = np.interp(V, vs, np.arange(len(vs)))            # fractional step
    q = pk.ordered(pk.banded(pos, band), matrix, ox, oy)
    return ix[np.clip(q, 0, len(vs) - 1)]


def pebbles(h, w, rng, rows_z, f, sizes_m=(0.04, 0.16), density=1.0, mask=None):
    """scatter of foreshortened stones, as step OFFSETS on a canvas (0 = none).
    small (under ~2 px): a 1-2 px fleck of the stone's own albedo (+1 or -1);
    larger: flat-bottomed lozenge, lit top row +2, body = albedo, underside -1, contact run -2."""
    out = np.zeros((h, w), int)
    items = []
    for r in range(h):
        z = rows_z[r]
        if not np.isfinite(z) or z <= 0:
            continue
        ppm = f / z
        n = int(density * w * (0.06 + 0.10 * rng.random()))
        for _ in range(n):
            s = np.exp(rng.uniform(np.log(sizes_m[0]), np.log(sizes_m[1])))
            rx = s * ppm / 2
            if rx < 0.5:
                continue
            items.append((r, rng.uniform(0, w), rx, max(0.5, rx * 0.42), int(rng.choice([-1, 0, 0, 1]))))
    items.sort()
    for (r, cx, rx, ry, alb) in items:
        if rx < 1.0:                                   # fleck
            x0 = int(cx); L = 1 + (rng.random() < rx)
            out[r, x0:min(w, x0 + L)] = 1 if alb >= 0 else -1
            continue
        yy, xx = np.mgrid[int(r - ry - 1):int(r + ry + 2), int(cx - rx - 1):int(cx + rx + 2)]
        ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        # flat-bottomed: rows above the centre are an ellipse, the bottom row is straight
        d = np.abs((xx + 0.5 - cx) / rx) ** 2.2 + np.maximum((yy + 0.5 - r) / ry, -9) ** 2
        body = (d <= 1.0) & ok
        if not body.any():
            continue
        ys = yy[body]
        rel = np.where(body, np.clip(alb, -1, 1) + (1 if alb >= 0 else 0), 0)   # body tone
        rel = np.where(body & (yy == ys.min()), 2, rel)                         # lit top row
        if ys.max() > ys.min():
            rel = np.where(body & (yy == ys.max()), -1, rel)                    # shaded underside
        sh = ok & (~body) & (yy == ys.max() + 1) & (np.abs(xx + 0.5 - cx) < rx * 0.8)
        rel = np.where(sh, -2, rel)
        sel = rel != 0
        out[yy[sel], xx[sel]] = rel[sel]
    if mask is not None:
        out[~mask] = 0
    return out


# ---------------------------------------------------------------- bank objects ----------------
def paint_boulder(cv, m, ramp, rng, light=(-0.6, -0.8)):
    """two value families on the ramp (ordered dither inside each), hard terminator, dark contact row,
    lit rim on the sun side. ramp: indices dark->light (>=6)."""
    from scipy.ndimage import distance_transform_edt as edt
    if not m.any():
        return
    ys, xs = np.nonzero(m)
    cy, cx = ys.mean(), xs.mean()
    ry, rx = (ys.max() - ys.min() + 1) / 2, (xs.max() - xs.min() + 1) / 2
    yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    nx = (xx - cx) / rx
    ny = (yy - (ys.max() - 0.2 * ry * 2)) / (ry * 2)        # dome: the normal tips up toward the top
    nz = np.sqrt(np.clip(1 - nx ** 2 - np.clip(ny, -1, 0) ** 2, 0.05, 1))
    lx, ly = light
    u = (nx * lx + ny * ly) * 0.8 + nz * 0.35
    ins = edt(m)
    lit = u > 0.35
    # lit family steps 3.4..5 ; shadow family 1..2.2 ; gradation by ordered dither
    s = np.where(lit, 3.4 + (u - 0.35) * 3.0, 1.0 + np.clip(u + 0.3, 0, 0.65) * 1.9)
    s = s + (pk.value_noise(*m.shape, 2, rng, 1) - 0.5) * 0.5
    q = np.clip(pk.ordered(s), 0, len(ramp) - 1)
    q = np.where(lit, np.clip(q, 3, len(ramp) - 1), np.clip(q, 0, 2))
    q = np.where(ins <= 1.0, np.where(lit, np.minimum(q + 1, len(ramp) - 1), np.maximum(q - 1, 0)), q)
    bottom = m & ~np.roll(m, -1, 0)
    q = np.where(bottom, 0, q)
    cv[m] = np.asarray(ramp)[q][m]


def paint_shrub(cv, m, ramp, rng, clump=(1.6, 3.2), light=(-0.6, -0.8)):
    """foliage mass as clumps: each a small blob lit upper-left, dark lower-right; holes; darker base."""
    if not m.any():
        return
    h, w = m.shape
    ys, xs = np.nonzero(m)
    y0, y1 = ys.min(), ys.max()
    out = np.full((h, w), -1)
    n = int(m.sum() / 5)
    cand = rng.integers(0, len(ys), n)
    order = np.argsort(ys[cand])                    # higher clumps first; lower ones overlap them
    for i in cand[order]:
        cy, cx = ys[i], xs[i]
        r = rng.uniform(*clump)
        yy, xx = np.mgrid[int(cy - r - 1):int(cy + r + 2), int(cx - r - 1):int(cx + r + 2)]
        ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        d = ((yy + 0.5 - cy) ** 2 + (xx + 0.5 - cx) ** 2) / r ** 2
        b = ok & (d <= 1)
        if not b.any():
            continue
        lx, ly = light
        u = (((xx + 0.5 - cx) / r) * lx + ((yy + 0.5 - cy) / r) * ly)
        depth = (cy - y0) / max(y1 - y0, 1)          # 0 top .. 1 base
        st = 2.2 + u * 1.4 - depth * 0.6 + rng.normal(0, 0.35)
        out[yy[b], xx[b]] = np.clip(np.round(st[b]), 0, len(ramp) - 1).astype(int)
    out = np.where(m & (out < 0), 0, out)
    holes = m & (rng.random((h, w)) < 0.05)
    out[holes] = 0
    sel = out >= 0
    cv[sel] = np.asarray(ramp)[out[sel]]
    return sel


def tone_down(idx, ramp, lights=1, dark_up=True):
    """Ferrari's reflection law on one ramp: the top half drops `lights` steps, the darkest comes up."""
    if np.size(idx) == 0:
        return idx
    ramp = list(ramp)
    k = np.array([ramp.index(i) if i in ramp else -1 for i in idx.ravel()]).reshape(idx.shape)
    n = len(ramp)
    k2 = np.where(k >= n // 2, k - lights, k)
    if dark_up:
        k2 = np.where(k == 0, 1, k2)
    return np.where(k >= 0, np.asarray(ramp)[np.clip(k2, 0, n - 1)], idx)


def to_ramp_checker(V, ramp_vals, ramp_idx, flat=0.24, ox=0, oy=0):
    """WATER quantizer (Ferrari's mirrored sky): the checker of two adjacent tints is the material.
    Holds a 50% checker across most of each step, passing through a flat tint only within
    +-flat/2 of an integer step."""
    vals = np.asarray(ramp_vals, float)
    order = np.argsort(vals)
    vs = vals[order]; ix = np.asarray(ramp_idx)[order]
    pos = np.interp(V, vs, np.arange(len(vs)))
    f = np.floor(pos); r = pos - f
    h, w = np.shape(V)
    yy, xx = np.mgrid[0:h, 0:w]
    par = ((xx + ox + yy + oy) % 2) == 0
    lo = f.astype(int); hi = lo + 1
    near_lo = r < flat / 2
    near_hi = r > 1 - flat / 2
    q = np.where(near_lo, lo, np.where(near_hi, hi, np.where(par, lo, hi)))
    return ix[np.clip(q, 0, len(vs) - 1)]


def gravel(mask, base_v, zrow, f, rng, ramp_i, ramp_v, stone_m=(0.03, 0.14), density=1.0, matrix=None,
           aspect=0.6, gap_steps=1.1, top=1.1):
    """gravel at a low angle, as stones on an ordered base.
    base: the value field through a 4x4 ordered dither (the smooth mass).
    stones: each its own albedo (rivers mix rock types); a flat-bottomed lozenge whose rows follow an
    ellipse (short lit top row, full body, shaded underside) with a shorter contact row beneath;
    width = stone size projected at that row; far stones shrink to 1-px flecks; random y; nearer
    stones overlap farther ones."""
    h, w = mask.shape
    matrix = pk.BAYER4 if matrix is None else matrix
    vals = np.asarray(ramp_v, float); n = len(vals)
    step = np.interp(base_v, vals, np.arange(n))
    # the base is the shadow BETWEEN stones; the stones sit on it lighter (light blobs on dark)
    ppm_row = np.where(np.isfinite(zrow), f / np.where(np.isfinite(zrow), zrow, 1), 0)
    gap = np.clip((ppm_row - 20.0) / 50.0, 0, 1)[:, None] * gap_steps            # interstices only read when near
    out_step = pk.ordered(step - gap, matrix).astype(float)
    step = step - gap
    marks = []
    for y in range(h):
        z = zrow[y]
        if not np.isfinite(z) or z <= 0:
            continue
        ppm = f / z
        # fill this row's area (w x 1 px) to `density`: a stone covers wpx x hgt px, so big stones
        # are few -- otherwise every stone shows only its top row and the bank turns to brick courses
        area = 0.0
        dens_row = density * float(np.clip((ppm - 24.0) / 40.0, 0.0, 1.0))    # bars are calm fields: stones only up close
        while area < dens_row * w:
            s = np.exp(rng.uniform(np.log(stone_m[0]), np.log(stone_m[1])))
            wpx = s * ppm
            area += max(wpx, 0.8) * max(1.0, wpx * aspect)
            marks.append((y + rng.random(), rng.uniform(-3, w), wpx, rng.normal(0, 0.8)))
    marks.sort(key=lambda t: t[0])
    for (yf, x0, wpx, alb) in marks:
        y = int(yf)
        if wpx < 0.8:
            if rng.random() < 0.85 + 0.14 * (1 - wpx / 0.8):
                continue
            xi = int(x0)
            if 0 <= xi < w and mask[y, xi]:
                out_step[y, xi] = step[y, xi] + (1.2 if alb >= 0 else -1.2)
            continue
        L = max(1, int(round(wpx)))
        hgt = max(2, int(round(wpx * aspect)))            # rows: a stone seen at a low angle
        cx = x0 + L / 2
        for j in range(hgt + 1):
            yy = y + j
            if yy >= h:
                break
            if j < hgt:
                # ellipse profile with a flat bottom: rows widen quickly from the top
                t = (j + 0.5) / hgt
                half = L / 2 * np.sqrt(max(0.0, 1 - (1 - min(1.0, t * 1.6)) ** 2))
                g0 = gap[yy, 0] if np.ndim(gap) else gap
                if j == 0:
                    off = g0 + top + 0.3 * alb              # lit top
                elif j == hgt - 1:
                    off = g0 - 0.7 + 0.3 * alb              # shaded underside
                else:
                    off = g0 + 0.9 + 0.6 * alb              # body in its own albedo
            else:
                half = L / 2 * 0.7                          # contact shadow, shorter
                off = -0.6
            xs = np.arange(int(np.floor(cx - half + 0.5)), int(np.floor(cx + half + 0.5)))
            xs = xs[(xs >= 0) & (xs < w)]
            if len(xs) == 0:
                continue
            sel = mask[yy, xs]
            out_step[yy, xs[sel]] = step[yy, xs[sel]] + off
    q = np.clip(np.round(out_step), 0, n - 1).astype(int)
    return np.asarray(ramp_i)[q]


def shallows(cv, water, edge_row, rng, tint, mirror_ok, rows=6, stones_per_px=0.10, lap=None):
    """the near-edge shelf where you see the bed (Ferrari's shallow shelf, C4): FLAT tints with
    stepped boundaries (runs of 3-16 px), never a one-row checker. Two shelves: the outer (lighter,
    deeper) and the inner (darker, shallowest). A broken light line on the outer shelf's lip.
    Stones: flat lozenges, dark body, sky-lit top row; stones at the very edge break the surface and
    get the lap line under them."""
    h, w = water.shape
    out = cv.copy()

    def steps(lo, hi):
        t = np.zeros(w, int); x = 0
        while x < w:
            L = int(rng.integers(3, 17)); t[x:x + L] = int(rng.integers(lo, hi + 1)); x += L
        return t
    outer = steps(rows - 2, rows)
    inner = np.minimum(steps(1, max(2, rows // 2)), outer - 1)
    for c in range(w):
        e = edge_row[c]
        for k in range(1, outer[c] + 1):
            r = e - k
            if r < 0 or not water[r, c] or not mirror_ok[r, c]:
                continue
            if k <= inner[c]:
                out[r, c] = tint[0] if k == 1 else tint[1]
            else:
                out[r, c] = tint[2]
        r = e - outer[c] - 1
        if r >= 0 and water[r, c] and mirror_ok[r, c] and rng.random() < 0.55:
            out[r, c] = tint[3]
    n = int(w * stones_per_px)
    for i in range(n):
        c0 = int(rng.integers(0, w - 3)); L = int(rng.integers(3, 9))
        e = edge_row[c0]; k = int(rng.integers(1, max(2, outer[c0])))
        r = e - k
        cs = np.arange(c0, min(w, c0 + L))
        if r - 1 < 0 or not (water[r, cs].all() and mirror_ok[r, cs].all()):
            continue
        out[r, cs] = tint[0]
        out[r - 1, cs[1:-1]] = tint[3] if k > 1 else tint[4]            # sky-lit wet top
        if k <= 1 and lap is not None and r + 1 < h:
            out[r + 1, cs[:-1]] = lap                                  # emergent: lap line under it
    return out


def boulder_mask(h, w, cx, base_y, width, height, rng, nv=7):
    """a faceted boulder silhouette: convex polygon, flat bottom on base_y, top a broken plane.
    Returns (mask, facets) where facets labels 0 top plane, 1 sun face, 2 shadow face."""
    from matplotlib.path import Path
    hw = width / 2
    # vertices: flat bottom corners, shoulders, a tilted top plane with a corner
    top_y = base_y - height
    ridge_x = cx + rng.uniform(-0.15, 0.25) * width          # where the sun face meets the shadow face
    # a faceted superellipse: 8 vertices on the upper half, jittered, joined by straight runs
    ang = np.linspace(0, np.pi, 9)
    ang = ang + np.r_[0, rng.uniform(-0.12, 0.12, 7), 0]
    nexp = 2.6
    pts = []
    for a in ang:
        ca, sa = np.cos(a), np.sin(a)
        px = np.sign(ca) * abs(ca) ** (2 / nexp); py = abs(sa) ** (2 / nexp)
        r = rng.uniform(0.9, 1.05)
        pts.append((cx + hw * px * r, base_y - height * py * r))
    pts = [(cx + hw, base_y)] + pts + [(cx - hw, base_y)]
    yy, xx = np.mgrid[0:h, 0:w]
    m = Path(pts).contains_points(np.c_[xx.ravel() + 0.5, yy.ravel() + 0.5]).reshape(h, w)
    # facets: the top plane is the band within ~22% of the height under the top edge
    lab = np.full((h, w), -1)
    topedge = np.full(w, 10 ** 6)
    for c in range(w):
        col = np.nonzero(m[:, c])[0]
        if len(col):
            topedge[c] = col.min()
    # the top plane is cut by a straight tilted line (a plane, not a band hugging the contour)
    slope = rng.uniform(-0.18, 0.05)
    cut = top_y + height * rng.uniform(0.26, 0.36) + slope * (xx - cx)
    tp = m & (yy < cut)
    lab[m] = np.where(xx[m] < ridge_x + (yy[m] - top_y) * 0.25, 1, 2)
    lab[tp] = 0
    return m, lab


def paint_faceted(cv, m, lab, ramp, rng, top=(4.2, 5.0), sun=(3.0, 4.2), shade=(1.0, 1.8)):
    """planes, not a dome: each facet holds its own value family; within a facet the ordered dither
    DENSITY carries the turn (sparse in the middle of the plane, thicker toward its far edge), a
    lighter echo under the top edge, a dark core where the sun face turns to the shadow face,
    contact row at the bottom, and a couple of short crack ticks in the light."""
    from scipy.ndimage import distance_transform_edt as edt
    h, w = m.shape
    n = len(ramp)
    yy, xx = np.mgrid[0:h, 0:w]
    s = np.zeros((h, w))
    for k, (lo, hi) in enumerate((top, sun, shade)):
        f = lab == k
        if not f.any():
            continue
        d = edt(f)
        dn = np.clip(d / max(d.max(), 1), 0, 1)             # 0 at the facet edge .. 1 in the middle
        s[f] = (lo + (hi - lo) * dn ** 0.7)[f]
    s += (pk.value_noise(h, w, 2.0, rng, 1) - 0.5) * 0.35
    q = np.clip(pk.ordered(s), 0, n - 1)
    # the core: the shadow-face pixels touching the sun face go one darker
    core = (lab == 2) & (np.roll(lab, 1, 1) == 1)
    core |= (lab == 2) & (np.roll(lab, 2, 1) == 1) & (rng.random((h, w)) < 0.5)
    q[core] = np.maximum(q[core] - 1, 0)
    bottom = m & ~np.roll(m, -1, 0)
    q[bottom] = 0
    cv[m] = np.asarray(ramp)[q][m]


# ---- Ferrari lawn strands: ported verbatim from the master-copy student (pedagogy/master-copy/src/fl.py),
# measured on Mirror Pond: stacked adjacent-value runs, paired stem+highlight, black gaps, clean lean.
def strands(h, w, rng, classes_fn, u_fn, runmean_fn, mask=None, spread=0.55, rho_fn=None,
            tuft=None, pair_p=0.6, pair_side=+1, gap_p=0.0, gap_class=None, jog_p=0.3,
            adjacent=True, jump_p=0.2, target_fn=None, lean_dir=0):
    """Ferrari's lawn micro-structure (measured on Mirror Pond, see LESSONS.md):
      * STACKED: down a column, runs step through ADJACENT values (dark -> mid -> lit -> mid);
        a highlight run is capped above and below by the mid class ~75% of the time.
      * PAIRED: a highlight run has a partner run one value darker in the neighbouring
        column on a consistent side, over the same rows (peak at dy=0) -- stem + highlight,
        a blade two pixels wide.  In shadow grass the pair also has a black gap on the
        other side ([black][hi][partner]).
    classes_fn(y, x) -> ordered class ids dark->light;  u_fn(y, x) -> target position (float);
    runmean_fn(y, x) -> mean vertical run length.  Returns index canvas (-1 outside mask)."""
    from scipy.special import ndtr
    cv = np.full((h, w), -1, int)
    runs = []
    for x in range(w):
        y = 0
        prev = None
        while y < h:
            if mask is not None and not mask[y, x]:
                y += 1; prev = None
                continue
            m = runmean_fn(y, x)
            L = 1 if m <= 1.1 else int(rng.geometric(1.0 / m))
            yc = min(h - 1, y + L // 2)
            cls = classes_fn(yc, x)
            n = len(cls)
            rh = rho_fn(yc, x) if rho_fn is not None else 0.0
            z = rh * (tuft[yc, x] if tuft is not None else 0) + np.sqrt(1 - rh ** 2) * rng.normal()
            if target_fn is not None:
                target = int(target_fn(yc, x, z))
            else:
                target = int(np.clip(np.round(u_fn(yc, x) + spread * z), 0, n - 1))
            if prev is None or not adjacent or rng.random() < jump_p:
                k = target
            else:
                pk = min(prev, n - 1)
                if target > pk:
                    k = pk + 1
                elif target < pk:
                    k = pk - 1
                else:
                    k = pk + (rng.choice([-1, 1]) if rng.random() < 0.6 else 0)
                k = int(np.clip(k, 0, n - 1))
            c = cls[k]
            xx = x
            y1 = min(h, y + L)
            # leaning blade: a clean 2:1 or 3:1 staircase (short straight runs, regular steps)
            lean = 0; step = 99
            if L >= 3 and rng.random() < jog_p:
                lean = int(rng.choice([-1, 1])) if lean_dir == 0 else lean_dir
                step = int(rng.choice([2, 3, 3, 4]))
            for yy in range(y, y1):
                if mask is not None and not mask[yy, x]:
                    y1 = yy
                    break
                if lean and yy > y and (y1 - 1 - yy) % step == 0:
                    xx = int(np.clip(xx + lean, 0, w - 1))
                cv[yy, xx] = c
            if y1 > y:
                runs.append((x, y, y1, k, cls))
            prev = k
            y = max(y1, y + 1)
    # holes left where a blade jogged sideways: continue the stroke above
    for y in range(1, h):
        hole = (cv[y] < 0) & (cv[y - 1] >= 0) & (mask[y] if mask is not None else True)
        cv[y, hole] = cv[y - 1, hole]
    hole = cv < 0
    if mask is not None:
        hole &= mask
    if hole.any():
        cv[hole] = classes_fn(0, 0)[0]
    # pairing pass
    for x, y0, y1, k, cls in runs:
        if k >= len(cls) - 2 and k >= 1 and rng.random() < pair_p:
            xp = x + pair_side
            dy = 0 if rng.random() < 0.6 else int(rng.choice([-1, 1]))
            for yy in range(max(0, y0 + dy), min(h, y1 + dy)):
                if 0 <= xp < w and (mask is None or mask[yy, xp]):
                    cv[yy, xp] = cls[k - 1]
            if gap_class is not None and rng.random() < gap_p:
                xg = x - pair_side
                for yy in range(y0, y1):
                    if 0 <= xg < w and (mask is None or mask[yy, xg]):
                        cv[yy, xg] = gap_class
    return cv


def sheared_strands(h, w, rng, shear_step, mask=None, tuft=None, **kw):
    """strands() laid along a clean staircase direction: every `shear_step` rows the
    strand moves one column (+ for '\\' tips up-left, - for '/').  Generated on a sheared
    grid and mapped back, so pairs stay side by side along the lean."""
    s = abs(shear_step); sgn = 1 if shear_step > 0 else -1
    extra = h // s + 2
    Wd = w + extra
    ys = np.arange(h)[:, None]
    xs = np.arange(w)[None, :]
    off = (ys // s) * sgn + (extra if sgn < 0 else 0)   # x' = x - off  ... map sheared col = x - off + extra?
    xd = xs - (ys // s) * sgn + (0 if sgn < 0 else extra)
    md = None
    if mask is not None:
        md = np.zeros((h, Wd), bool)
        np.put_along_axis(md, xd, mask, axis=1)
    td = None
    if tuft is not None:
        td = np.zeros((h, Wd)); np.put_along_axis(td, xd, tuft, axis=1)
    S = strands(h, Wd, rng, mask=md, tuft=td, **kw)
    return np.take_along_axis(S, xd, axis=1)


