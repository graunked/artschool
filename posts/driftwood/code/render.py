"""render.py -- driftwood on gravel: world -> index canvas (mat, step) -> pixels.

Layers (painter core):
  1 form     geo.trace + tech.cylinder_classes (wood), pebble domes (gravel)
  2 value    families: cast shadow forces the shadow family; contact line at 0
  3 marks    (stage 2+) grain, cracks, knots
  4 drawing  contact line, bottom row of the log, silhouette runs
  5 quantize class checker (wood), ordered 2x2 (gravel)
  6 palette  (stage 4) ramps per material
"""
import numpy as np
import geo, tech
from paint import (WOOD, GRAVEL, RAW, BARK, Light, gravel_field, hash2, BAYER2)
ROOTM = 5
WETW = 6

GREY = {
    WOOD:   [24, 60, 112, 128, 152, 178, 204, 232],
    GRAVEL: [18, 40, 64, 86, 108, 128, 146, 164],
    RAW:    [24, 60, 116, 134, 160, 188, 214, 240],
    BARK:   [14, 30, 50, 66, 84, 102, 120, 140],
    ROOTM:  [12, 26, 42, 58, 76, 96, 118, 142],
    7:      [18, 40, 64, 86, 108, 128, 146, 164],
    8:      [18, 40, 64, 86, 108, 128, 146, 164],
    WETW:   [16, 34, 58, 72, 88, 104, 122, 220],
    9:      [20, 50, 90, 104, 124, 146, 168, 200],
}


def grey_ramps():
    return {m: [(g, g, g) for g in v] for m, v in GREY.items()}


class Canvas:
    def __init__(self, H, W):
        self.mat = np.full((H, W), GRAVEL)
        self.step = np.zeros((H, W), int)


def render(logs, cam, light, seed=0, stone_m=0.06, opts=None):
    o = dict(contact=True, bottom_row=True, gravel=True, cast=True)
    o.update(opts or {})
    H, W = cam.H, cam.W
    # what is below a pixel does not exist: limbs thinner than ~0.6 px and rootlets on logs under
    # ~3 px radius are dropped (critic r10: stubs read as antennae at 6-9 px/m)
    for lg in logs:
        lg.limbs = [l for l in lg.limbs if l[2] * cam.scale >= 0.6]
        if lg.rad.mean() * cam.scale < 3:
            lg.rootlets = []
    buf = geo.trace(cam, logs)
    sh = geo.shadow(buf, light.L, logs) if o['cast'] and not light.overcast else np.zeros((H, W), bool)
    cv = Canvas(H, W)
    V = -cam.f
    obj = buf['obj']
    wood = obj >= 0
    # ---------- wood form: five classes around the axis
    d, phL = tech.normal_plane_angles(buf['N'], V, light.L, buf['T'])
    v = tech.cylinder_classes(d, phL, sh=sh)
    if light.overcast:
        # overcast: the sky is the light; peak straight up
        d2, ph2 = tech.normal_plane_angles(buf['N'], V, np.array([0, 0, 1.0]), buf['T'])
        v = tech.cylinder_classes(d2, ph2)
    diam = 2 * buf['rad'] * cam.scale
    # with marks on, big logs turn their form by MARK DENSITY, not by checker: the classes are
    # quantized nearly pure (seams at most 1 px) and the grain supplies the gradation
    if o.get('marks', True):
        v = tech.sharpen(v, diam, max_gain=12.0)
        v = np.where(diam >= 7, tech.sharpen(v, np.full_like(diam, 1.0), max_gain=12.0), v)
    else:
        v = tech.sharpen(v, diam)
    st, k = tech.quantize_classes(v)
    if o.get('marks', True):
        pure = tech.CLASS_STEPS[np.clip(np.round(v), 0, 4).astype(int)]
        st = np.where(diam >= 7, pure, st)
    # near the silhouette there is no dither: Ferrari's outlines are clean runs of ONE tone and
    # his seams sit inside the form.  Within 2 px of the outline the class is rounded, and along
    # the outline itself each pixel takes the majority class of the outline pixels around it
    # (so a diagonal lit edge is one tone, not a sawtooth of two).
    from scipy import ndimage as ndi
    nbr_out = np.zeros_like(wood)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nbr_out |= np.roll(np.roll(obj, dy, 0), dx, 1) != obj
    sil = wood & nbr_out
    near = wood & ndi.binary_dilation(sil, iterations=1)
    kk = np.clip(np.round(v), 0, 4).astype(int)
    st = np.where(near, tech.CLASS_STEPS[kk], st)
    # majority along the outline (window 7 px), per class
    counts = np.stack([ndi.uniform_filter(((kk == c) & sil).astype(float), 7) for c in range(5)])
    maj = np.argmax(counts, 0)
    st = np.where(sil, tech.CLASS_STEPS[maj], st)
    part = buf['part']
    # sun-side silhouette: where the lit plane meets what is behind it, a 1-px light run
    # (contre-jour: this is the only light the log gets -- Ferrari's V02 rim, c5 boulder edge)
    if not light.overcast:
        lx = light.L @ cam.r; ly = -(light.L @ cam.u)
        sdx = int(np.sign(lx)) if abs(lx) > 0.35 else 0
        sdy = int(np.sign(ly)) if abs(ly) > 0.35 else 0
        if sdx or sdy:
            edge = np.zeros_like(wood)
            for ddy, ddx in ((sdy, 0), (0, sdx), (sdy, sdx)):
                if ddy == 0 and ddx == 0:
                    continue
                nb = np.roll(np.roll(obj, -ddy, 0), -ddx, 1)
                tn = np.roll(np.roll(buf['t'], -ddy, 0), -ddx, 1)
                edge |= wood & ((nb != obj) & ((nb < 0) | (tn > buf['t'] + 0.05)))
            ndl0 = (buf['N'] * light.L).sum(-1)
            # a continuous run: it thins/breaks only where the form turns from the light
            rimlit = edge & (ndl0 > -0.15)
            st = np.where(rimlit, np.maximum(st, 6), st)
    # a cast shadow on wood is a FAMILY jump, not a class nudge: shadowed wood drops to the core
    # tone (2), or deep shadow (1) where it also turns from the sun -- the value break that
    # separates one log from the one lying across it (critic r10)
    if not light.overcast:
        ndl_c = (buf['N'] * light.L).sum(-1)
        st = np.where(wood & sh, np.where(ndl_c > 0.2, 2, 1), st)
    # ---------- end faces
    st = end_faces(st, buf, logs, light, sh, cam)
    cv.mat[wood] = WOOD
    for k, lg in enumerate(logs):
        if lg.kind == 'bark':
            cv.mat[obj == k] = BARK
        elif lg.kind == 'grey':
            cv.mat[obj == k] = 9
    cv.step[wood] = st[wood]
    # ---------- gravel form
    g = ~wood
    if o['gravel']:
        import gravel
        gx = gravel.paint_gravel(buf, cam, light, sh, seed=seed, stone_m=stone_m,
                                 density=o.get('density', 0.55))
        cv.step[g] = gx[g]
        cv.mat[g] = gravel.paint_gravel.last_mat[g]
    else:
        cv.step[g] = np.where(sh, 2, 4)[g]
    # ---------- drawing: contact line and the log's bottom row
    if o['contact']:
        contact(cv, buf, logs, cam)
        separate_logs(cv, buf, logs, cam)
    # ---------- root wad disc: a tangled mass (stage 3)
    rootwad_mass(cv, buf, logs, cam, light, sh, seed)
    # ---------- marks: grain, checks, knots (stage 2)
    if o.get('marks', True):
        import marks
        marks.wood_marks(cv, buf, logs, cam, light, sh, weather=o.get('weather', 1.0), seed=seed)
    # ---------- rootlets: thin root strands as clean-run strokes (a mass's fringe)
    rootlets(cv, buf, logs, cam, light)
    # ---------- splinter slivers: long clean 1-px runs past the jagged break
    slivers(cv, buf, logs, cam, light)
    # ---------- wet logs: their own darker, warmer ramp, plus a specular glint run
    wet_logs(cv, buf, logs, cam, light, sh)
    # ---------- thin logs: explicit stroke glyphs (the classes collapse below ~4.5 px)
    thin_logs(cv, buf, logs, cam, light, sh)
    return cv, buf, sh


THIN_ROWS = {1: [6], 2: [6, 1], 3: [6, 4, 1], 4: [5, 6, 4, 2]}
THIN_ROWS_SH = {1: [3], 2: [3, 1], 3: [3, 2, 1], 4: [3, 3, 2, 1]}


def thin_logs(cv, buf, logs, cam, light, sh):
    """trigger: a log whose projected diameter is < 4.5 px.  recipe (Ferrari's narrow limbs:
    the classes shrink to one pixel each, then drop from the shadow side): the axis is drawn as a
    clean-run line, t = round(diameter) pixels thick; rows from the lit side:
        1: light          2: light, core          3: light, half, core
        4: edge, light, half, core
    and a darkest contact pixel under each column (the smallest light-top / dark-bottom pair).
    A lit sawn end adds one end-face pixel (7)."""
    H, W = cv.step.shape
    for k, lg in enumerate(logs):
        dmax = 2 * lg.rad.max() * cam.scale
        if dmax >= 4.5:
            continue
        # clear what the tracer drew (it aliases at this size)
        m = buf['obj'] == k
        cv.mat[m] = GRAVEL
        sx, sy = cam.project(lg.pts)
        for i in range(len(lg.pts) - 1):
            x0, y0, x1, y1 = sx[i], sy[i], sx[i + 1], sy[i + 1]
            d0 = 2 * lg.rad[i] * cam.scale; d1 = 2 * lg.rad[i + 1] * cam.scale
            # top of the silhouette: axis raised by half the projected thickness
            n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
            for j in range(n):
                f = j / max(n - 1, 1)
                x = x0 + (x1 - x0) * f; y = y0 + (y1 - y0) * f
                dd = d0 + (d1 - d0) * f
                t = int(np.clip(round(dd * 0.9), 1, 4))
                xi = int(np.floor(x)); yt = int(np.floor(y - dd * 0.5))
                if not (0 <= xi < W):
                    continue
                shd = 0 <= yt < H and sh[yt, xi]
                rows = (THIN_ROWS_SH if (shd or light.overcast) else THIN_ROWS)[t]
                for rr, v in enumerate(rows):
                    yy = yt + rr
                    if 0 <= yy < H:
                        cv.mat[yy, xi] = WOOD; cv.step[yy, xi] = v
                yy = yt + len(rows)
                if 0 <= yy < H and (t == 1 or t >= 4):
                    cv.mat[yy, xi] = GRAVEL; cv.step[yy, xi] = 0 if t >= 4 else 1
        # sawn end facing the sun: one bright pixel
        for e in (0, 1):
            if lg.ends[e] == 'sawn':
                c, out = lg.ends_frames()[e]
                if out @ light.L > 0.3 and not light.overcast:
                    ex, ey = cam.project(c)
                    xi, yi = int(ex[0]), int(ey[0] - lg.rad[0] * cam.scale * 0.5)
                    if 0 <= xi < W and 0 <= yi < H:
                        cv.mat[yi, xi] = WOOD; cv.step[yi, xi] = 7


def end_faces(st, buf, logs, light, sh, cam):
    """sawn: a flat face -- ONE lit step (6) for the plane, a 1-px darker ring at the rounded
    arris (4), growth rings as 1-px runs one step down (5) when the face is >= 16 px, a radial
    check (3) from the pith when >= 12 px.  In shadow the same marks one family down.
    broken: raw wood by its facet normal -- lit facets 6/7, turned facets 4, shadowed 3 --
    the palest thing on the log (raw wood has not silvered)."""
    obj = buf['obj']; part = buf['part']
    ndl = (buf['N'] * light.L).sum(-1)
    out = st.copy()
    for k, lg in enumerate(logs):
        for e in (0, 1):
            m = (obj == k) & (part == e + 1)
            if not m.any():
                continue
            r_px = lg.rad[0 if e == 0 else -1] * cam.scale
            lit = (ndl > 0.2) & ~sh & ~light.overcast
            if lg.ends[e] == 'sawn':
                base = np.where(lit, np.where(ndl > 0.6, 6, 5), 3)
                if light.overcast:
                    base = np.full(base.shape, 5)
                rho = buf['rho']
                v = base.copy()
                if r_px >= 3:
                    arris = rho > 1 - 1.0 / max(r_px, 1)
                    v = np.where(arris, base - 2, v)
                th = buf['th']
                if r_px >= 20:
                    # growth rings: broken arcs (sectors), not a target of full circles
                    sector = (np.floor((th + np.pi) / (2 * np.pi) * 7) + np.floor(rho * r_px / 3.4) * 3) % 3 != 0
                    ring = (np.abs(((rho * r_px / 3.4) % 1.0) - 0.5) < 0.16) & (rho < 0.8) & sector
                    v = np.where(ring, base - 1, v)
                if r_px >= 5:
                    # radial checks: 1-3 dark splits from near the pith to the arris
                    nchk = 1 + int(hash_angle(lg.seed * 3 + e) / (2 * np.pi) * 3)
                    for q in range(nchk):
                        ca = hash_angle(lg.seed + e + 17 * q)
                        rmin = 0.1 + 0.3 * (q > 0)
                        chk = (np.abs(np.angle(np.exp(1j * (th - ca)))) * rho * r_px < 0.55) & (rho > rmin)
                        v = np.where(chk, np.maximum(base - 3, 1), v)
                out = np.where(m, v, out)
            else:
                # the break: tooth tips stand out and catch light, the torn face between them is
                # recessed (darker), crevices at its root darkest -- value follows how far the
                # point stands out along the axis, then the facet's light.
                c, outv = lg.ends_frames()[e]
                prof = lg.jag[e][0]
                s_rel = (buf['P'] - c) @ outv
                f = np.clip(s_rel / max(prof.max(), 1e-6), 0, 1)
                lit_v = np.where(f > 0.55, 7, np.where(f > 0.2, 6, 4))
                sh_v = np.where(f > 0.55, 4, np.where(f > 0.2, 3, 2))
                v = np.where(lit & (ndl > 0.35), lit_v, np.where(ndl > 0.0, np.minimum(lit_v, 5) - 1, sh_v))
                if light.overcast:
                    v = np.where(f > 0.4, 5, 3)
                out = np.where(m, v, out)
                out = despeckle(out, m)
    return out


def despeckle(st, mask, passes=2):
    """no orphan pixels inside a region: a pixel unlike all four neighbours takes the most common
    neighbour value (only within mask)."""
    st = st.copy()
    H, W = st.shape
    for _ in range(passes):
        nb = np.stack([np.roll(st, 1, 0), np.roll(st, -1, 0), np.roll(st, 1, 1), np.roll(st, -1, 1)])
        orphan = mask & np.all(nb != st[None], axis=0)
        if not orphan.any():
            break
        ys, xs = np.nonzero(orphan)
        for y, x in zip(ys, xs):
            vals = nb[:, y, x]
            u, c = np.unique(vals, return_counts=True)
            st[y, x] = u[np.argmax(c)]
    return st


def clean_line(x0, y0, x1, y1):
    """a clean-run line: DDA along the major axis with a CONSTANT run length (the slope is snapped
    to 1:1, 2:1, 3:1, 4:1 or flat / vertical equivalents) -- Ferrari's narrow-form rule."""
    dx, dy = x1 - x0, y1 - y0
    n = int(max(abs(dx), abs(dy)))
    if n == 0:
        return np.array([int(round(x0))]), np.array([int(round(y0))])
    if abs(dx) >= abs(dy):
        ratio = abs(dx) / max(abs(dy), 1e-6)
        run = min([1, 2, 3, 4, 99], key=lambda q: abs(q - ratio)) if ratio < 6 else 99
        xs = np.arange(n + 1) * np.sign(dx) + round(x0)
        ys = round(y0) + (np.arange(n + 1) // run) * np.sign(dy) * (run < 99)
    else:
        ratio = abs(dy) / max(abs(dx), 1e-6)
        run = min([1, 2, 3, 4, 99], key=lambda q: abs(q - ratio)) if ratio < 6 else 99
        ys = np.arange(n + 1) * np.sign(dy) + round(y0)
        xs = round(x0) + (np.arange(n + 1) // run) * np.sign(dx) * (run < 99)
    return xs.astype(int), ys.astype(int)


def wet_logs(cv, buf, logs, cam, light, sh):
    """trigger: a log with wet >= 0.5 (lying in or just out of the water).  recipe: switch its
    wood pixels to the wet ramp (index canvas unchanged: the palette carries the wetness), and
    add the specular glint -- pixels whose normal is within a few degrees of the half vector
    between sun and eye become step 7, a broken 1-px run along the top; wet things are dark with
    sharp highlights.  Overcast: the glint follows the bright sky overhead, broader and dimmer."""
    V = -cam.f
    Hv = light.L + V
    Hv = Hv / np.linalg.norm(Hv)
    if light.overcast:
        Hv = np.array([0, 0, 1.0]) + V; Hv = Hv / np.linalg.norm(Hv)
    for k, lg in enumerate(logs):
        if lg.wet < 0.5:
            continue
        m = (buf['obj'] == k) & (cv.mat == WOOD)
        cv.mat[m] = WETW
        spec = m & ((buf['N'] @ Hv) > (0.992 if not light.overcast else 0.97)) & ~sh
        cv.step[spec] = 7 if not light.overcast else 6


def rootwad_mass(cv, buf, logs, cam, light, sh, seed=0):
    """trigger: the disc of a root wad.  recipe -- a knobbly MASS (the way Ferrari clusters moss
    and foliage: clumps with a lit side and a dark side, crevices between, the turn carried by
    how much of each clump is lit):
      * its own dark ramp; the mass steps by N.L (lit 4, turned 3, shadow 2), cast shadow <= 2;
      * clumps: screen-space cells ~3.5 px across (root knuckles and tangles).  The sun side of
        each clump is +1, its anti-sun side -1, the crevice between clumps -1, and where three
        clumps meet a void (0);
      * the fringe (rootlets) is drawn separately and only OFF the mass, breaking its outline."""
    H, W = cv.step.shape
    obj = buf['obj']; part = buf['part']
    ndl = (buf['N'] * light.L).sum(-1)
    lx = light.L @ cam.r; ly = -(light.L @ cam.u)
    ln = np.hypot(lx, ly) + 1e-6
    sx, sy = (lx / ln, ly / ln) if not light.overcast else (0.0, -1.0)
    for k, lg in enumerate(logs):
        m = (obj == k) & (part == 4)
        if not m.any():
            continue
        rng = np.random.default_rng(lg.seed * 13 + seed)
        base = np.where(ndl > 0.45, 4, np.where(ndl > 0.1, 3, 2))
        if light.overcast:
            base = np.where(buf['N'][..., 2] > 0.2, 4, 3)
        base = np.where(sh, np.minimum(base, 2), base)
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        cell = max(2.8, min(5.0, 0.14 * np.sqrt(m.sum())))
        n = int((y1 - y0) * (x1 - x0) / cell ** 2) + 4
        cyx = np.stack([rng.uniform(y0 - 2, y1 + 2, n), rng.uniform(x0 - 2, x1 + 2, n)], 1)
        YY, XX = np.mgrid[y0:y1, x0:x1]
        dy = YY[..., None] - cyx[:, 0]; dx = XX[..., None] - cyx[:, 1]
        dist = np.hypot(dy, dx)
        o = np.argsort(dist, axis=-1)
        d1 = np.take_along_axis(dist, o[..., :1], -1)[..., 0]
        d2 = np.take_along_axis(dist, o[..., 1:2], -1)[..., 0]
        d3 = np.take_along_axis(dist, o[..., 2:3], -1)[..., 0]
        j = o[..., 0]
        ddy = np.take_along_axis(dy, j[..., None], -1)[..., 0]
        ddx = np.take_along_axis(dx, j[..., None], -1)[..., 0]
        side = (ddx * sx + ddy * sy) / (d1 + 1e-6)
        sub = base[y0:y1, x0:x1].copy()
        v = sub.copy()
        v = np.where((side > 0.35) & (d1 > 0.6), sub + 1, v)
        v = np.where((side < -0.45) & (d1 > 0.8), sub - 1, v)
        # crevice only on each clump's anti-sun boundary (the lit side of a clump meets the
        # next one without a line): a knuckle reads as light-top / dark-bottom, not a cell
        crev = ((d2 - d1) < 0.9) & (side < 0.1)
        v = np.where(crev, sub - 1, v)
        void = ((d2 - d1) < 0.9) & ((d3 - d1) < 1.2) & (side < -0.2) & (rng.random(side.shape) < 0.5)
        v = np.where(void, 0, v)
        mm = m[y0:y1, x0:x1]
        cvs = cv.step[y0:y1, x0:x1]
        cvs[mm] = np.clip(v, 0, 7)[mm]
        cv.mat[m] = ROOTM
        # the fringe: 1-px root ends breaking the outline where the mass meets the ground/sky,
        # radiating from the mass centre; clean runs 2-6 px, tapering to a half-step tip
        cy, cx = ys.mean(), xs.mean()
        free = obj < 0
        outl = m & (np.roll(free, 1, 0) | np.roll(free, -1, 0) | np.roll(free, 1, 1) | np.roll(free, -1, 1))
        oy, ox = np.nonzero(outl)
        for y_, x_ in zip(oy, ox):
            if rng.random() > 0.3:
                continue
            ang = np.arctan2(y_ - cy, x_ - cx) + rng.normal(0, 0.45)
            L = rng.integers(2, 7)
            xl, yl = clean_line(x_, y_, x_ + np.cos(ang) * L, y_ + np.sin(ang) * L)
            lit = (np.cos(ang) * sx + np.sin(ang) * sy) > -0.2 and not light.overcast
            for q, (x, y) in enumerate(zip(xl[1:], yl[1:])):
                if 0 <= x < W and 0 <= y < H and free[y, x] and cv.mat[y, x] != WOOD:
                    cv.mat[y, x] = ROOTM
                    tip = q >= len(xl) - 2
                    cv.step[y, x] = (6 if lit else 1) - (1 if (tip and lit) else 0)


def rootlets(cv, buf, logs, cam, light):
    """thin roots (< ~1.5 px): each a clean-run 1-px stroke from the root tip outward.  Value by
    which way the strand faces: up-facing strands lit (6), others core (2); a stroke over the
    ground gets a dark partner pixel beneath it (Ferrari's twigs: stroke + rim, never a smear).
    Hidden where the log's own surface is in front."""
    H, W = cv.step.shape
    V = -cam.f
    for k, lg in enumerate(logs):
        for pl in getattr(lg, 'rootlets', []):
            xs_, ys_ = cam.project(pl)
            tpts = (pl - cam.look) @ cam.f + 60.0
            for i in range(len(pl) - 1):
                xs, ys = clean_line(xs_[i], ys_[i], xs_[i + 1], ys_[i + 1])
                n = len(xs)
                seg = pl[i + 1] - pl[i]
                facing_up = (np.cross(seg, np.cross([0, 0, 1.0], seg))[2] > 0)
                lit = not light.overcast
                for j, (x, y) in enumerate(zip(xs, ys)):
                    if not (0 <= x < W and 0 <= y < H):
                        continue
                    tt = tpts[i] + (tpts[i + 1] - tpts[i]) * j / max(n - 1, 1)
                    if buf['obj'][y, x] >= 0 and (buf['t'][y, x] < tt - 0.02 or buf['part'][y, x] == 4):
                        continue
                    cv.mat[y, x] = WOOD
                    cv.step[y, x] = 6 if (lit and j % 7 != 3) else 2
                    if y + 1 < H and buf['obj'][y + 1, x] < 0 and cv.mat[y + 1, x] != WOOD:
                        cv.step[y + 1, x] = min(cv.step[y + 1, x], 2)


def slivers(cv, buf, logs, cam, light):
    """splinter ends: each sliver a clean-run 1-px line following the grain out past the break,
    lit (7) on the sun side of the log or turned (4), with a dark partner pixel beneath it."""
    H, W = cv.step.shape
    for lg in logs:
        for (base, dirv, ln, e) in getattr(lg, 'spikes', []):
            tip = base + dirv * ln
            (bx, tx), (by, ty) = cam.project(np.stack([base, tip]))
            if max(abs(tx - bx), abs(ty - by)) < 2:
                continue
            xs, ys = clean_line(bx, by, tx, ty)
            n = (base - lg.ends_frames()[e][0])
            lit = (n @ light.L > -0.02) and not light.overcast
            for j, (x, y) in enumerate(zip(xs, ys)):
                if 0 <= x < W and 0 <= y < H:
                    if cv.mat[y, x] != WOOD or buf['obj'][y, x] < 0 or j > 0:
                        cv.mat[y, x] = WOOD
                        cv.step[y, x] = 7 if lit else 4
                        if y + 1 < H and buf['obj'][y + 1, x] < 0 and cv.mat[y + 1, x] != WOOD:
                            cv.step[y + 1, x] = min(cv.step[y + 1, x], 2)


def hash_angle(s):
    return (np.sin(s * 12.9898) * 43758.5453) % 1.0 * 2 * np.pi


def gravel_steps(buf, light, sh, stone_m, seed, cam):
    """pebbles: each a flattened dome.  Painted with Ferrari's stone rule: the top of each stone
    light, its bottom row dark, the gaps darkest; cast shadow drops the whole stone into the
    shadow family (steps 1-3) but keeps a lit top pixel one step up."""
    inside, Nst, tone, bid, dist = gravel_field(buf, None, stone_m, seed)
    L = light.L
    ndl = (Nst * L).sum(-1)
    up = Nst[..., 2]
    base = 4.0 + 1.6 * (tone - 0.5)
    x_l = base + 1.6 * (ndl - L[2])
    x_s = 1.8 + 0.8 * up + 0.8 * (tone - 0.5)
    x = np.where(sh, np.minimum(x_s, 3.3), x_l)
    gap = ~inside
    x = np.where(gap, np.where(sh, 0.6, 1.8), x)
    if light.overcast:
        x = np.where(gap, 1.2, 3.0 + 1.4 * (tone - 0.5) + 1.2 * (up - 0.8))
    H, W = x.shape
    th = np.tile(BAYER2, (H // 2 + 1, W // 2 + 1))[:H, :W]
    q = np.floor(x) + ((x - np.floor(x)) > th)
    return np.clip(q, 0, 7).astype(int)


def contact(cv, buf, logs, cam):
    pocket = geo.belly(buf, logs, cam)
    gm = pocket & (buf['obj'] < 0)
    cv.step[gm] = 0
    return _contact_line(cv, buf, logs, cam)


def _contact_line(cv, buf, logs, cam):
    """1 px darkest line where a log meets the ground: ground pixels directly below the log's
    silhouette whose world point lies within ~1.5 px of the log surface; the log's own bottom
    row one class darker (the form turning under)."""
    obj = buf['obj']
    wood = obj >= 0
    below = np.zeros_like(wood); below[1:] = wood[:-1] & ~wood[1:]
    ys, xs = np.nonzero(below)
    P = buf['P']
    for y, x in zip(ys, xs):
        lg = logs[obj[y - 1, x]]
        dd = lg.sdf(P[y, x][None])[0]
        if dd < 2.0 / cam.scale:
            cv.step[y, x] = 0
            cv.mat[y, x] = GRAVEL
    # bottom row of the log where it touches: darkest wood class
    bot = wood.copy(); bot[:-1] &= ~wood[1:]
    by, bx = np.nonzero(bot)
    for y, x in zip(by, bx):
        if y + 1 < wood.shape[0] and cv.step[y + 1, x] == 0 and buf['obj'][y + 1, x] < 0:
            cv.step[y, x] = min(cv.step[y, x], 1)


def separate_logs(cv, buf, logs, cam):
    """neighbours separate by VALUE, not by outline (Ferrari c5):
      * where one log rests on another, the lower log's pixels just under the upper log's
        silhouette and within ~2 px of it in 3-D get the contact step (1) -- the crack between;
      * where a log passes in front of another, the rear log's pixels along the occluding edge
        drop two steps if the two tones would otherwise be within a step of each other."""
    obj = buf['obj']; t = buf['t']; P = buf['P']
    H, W = obj.shape
    wood = obj >= 0
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        o2 = np.roll(np.roll(obj, -dy, 0), -dx, 1)       # neighbour value at (y+dy, x+dx)
        t2 = np.roll(np.roll(t, -dy, 0), -dx, 1)
        s2 = np.roll(np.roll(cv.step, -dy, 0), -dx, 1)
        # pixel p is a REAR log pixel next to a FRONT log pixel
        rear = wood & (o2 >= 0) & (o2 != obj) & (t2 < t - 0.02)
        ys, xs = np.nonzero(rear)
        for y, x in zip(ys, xs):
            front = logs[o2[y, x]]
            close = front.sdf(P[y, x][None])[0] < 2.2 / cam.scale
            if close and dy == 1:
                cv.step[y, x] = min(cv.step[y, x], 1)          # resting contact, below
            elif abs(int(s2[y, x]) - int(cv.step[y, x])) <= 1:
                cv.step[y, x] = max(cv.step[y, x] - 2, 0)


def to_rgb(cv, ramps):
    out = np.zeros(cv.mat.shape + (3,), np.uint8)
    for m, r in ramps.items():
        sel = cv.mat == m
        out[sel] = np.asarray(r, np.uint8)[np.clip(cv.step[sel], 0, len(r) - 1)]
    return out
