"""Foliage techniques, discovered by copying Ferrari (stage 0), then carried to the willow.

All functions work in (ramp, step) space. A ramp is dark->light, step 0 is the shared near-black
gap. `light` is the copyist's sketch or the form field, in [0,1].
"""
import numpy as np
from scipy import ndimage as ndi
from wpaint import clean_line, value_noise


def leaf_glyph_pixels(x, y, L, ang, thick):
    """A leaf mark: an upper edge run of length L at angle ang (radians, screen, y down) and,
    if thick>1, a lower edge run one pixel below. Returns (upper pts, lower pts)."""
    dx, dy = np.cos(ang) * (L - 1), np.sin(ang) * (L - 1)
    up = clean_line(x, y, x + dx, y + dy)
    lo = [(px, py + 1) for (px, py) in up] if thick > 1 else []
    return up, lo


def near_leaves(region, light, rng, top=3, spacing=2.4, lens=(3, 6), thick_p=0.7,
                angles=((0, 3), (26.6, 2), (-26.6, 2), (45, 1), (-45, 1)), dens=(0.15, 0.95),
                base1=(0.35, 0.8), mask=None, jitter=0.6, taper=True):
    """Near-scale leaf glyphs (V09 foreground bush). Each leaf: lit upper edge, one-step-darker
    lower edge, tip one step down, a dark pixel under it. Leaves painted dim-to-bright so lit
    leaves lie in front. Returns S (steps) with region carried as the ramp map."""
    h, w = light.shape
    if mask is None:
        mask = np.ones((h, w), bool)
    S = np.zeros((h, w), int)
    # base: black gap with a clumpy scatter of step 1 where the light field is higher
    n = value_noise((h, w), 2.0, rng)
    t = 0.5 * rng.random((h, w)) + 0.5 * (n - n.min()) / (np.ptp(n) + 1e-9)
    p1 = base1[0] + (base1[1] - base1[0]) * light
    S[(t < p1 * 0.6) & mask] = 1
    # glyph sites: jittered grid
    ys, xs = np.mgrid[0:h:spacing, 0:w:spacing]
    ys = ys.ravel() + rng.uniform(-spacing, spacing, ys.size) * jitter
    xs = xs.ravel() + rng.uniform(-spacing, spacing, xs.size) * jitter
    angs = np.array([a for a, _ in angles]); wts = np.array([k for _, k in angles], float)
    wts /= wts.sum()
    G = []
    for (x, y) in zip(xs, ys):
        xi, yi = int(np.clip(round(x), 0, w - 1)), int(np.clip(round(y), 0, h - 1))
        if not mask[yi, xi]:
            continue
        l = light[yi, xi]
        if rng.random() > dens[0] + (dens[1] - dens[0]) * l:
            continue
        tp = int(np.clip(round(0.6 + l * (top + 0.4) + rng.normal(0, 0.55)), 1, top))
        L = int(rng.integers(lens[0], lens[1] + 1))
        a = np.radians(rng.choice(angs, p=wts)) * (1 if rng.random() < 0.5 else -1)
        if rng.random() < 0.5:
            a = np.pi - a    # leaves point left or right
        th = 2 if rng.random() < thick_p else 1
        G.append((tp + rng.random() * 0.5, xi, yi, L, a, th, tp))
    G.sort()
    for (_, x, y, L, a, th, tp) in G:
        up, lo = leaf_glyph_pixels(x, y, L, a, th)
        # dark tuck under the leaf
        for (px, py) in (lo or up):
            if 0 <= px < w and 0 <= py + 1 < h and mask[py + 1, px]:
                S[py + 1, px] = min(S[py + 1, px], max(0, tp - 2))
        for k, (px, py) in enumerate(lo):
            if 0 <= px < w and 0 <= py < h and mask[py, px]:
                S[py, px] = max(0, tp - 1)
        for k, (px, py) in enumerate(up):
            if 0 <= px < w and 0 <= py < h and mask[py, px]:
                s = tp
                if taper and k == len(up) - 1 and len(up) > 2:
                    s = tp - 1
                S[py, px] = s
    return S


def lozenge(x, y, L, ang, wmax=1.2, tip_dark=True):
    """Leaf lozenge pixels around axis from (x,y) of length L at angle ang (y down).
    Returns list of (px, py, role) with role 'T' (upper/lit half), 'm' (lower half or tip)."""
    ca, sa = np.cos(ang), np.sin(ang)
    # normal pointing 'down' on screen
    nx, ny = -sa, ca
    if ny < 0:
        nx, ny = -nx, -ny
    pts = {}
    r = int(np.ceil(L)) + 2
    for py in range(int(y) - r, int(y) + r + 1):
        for px in range(int(x) - r, int(x) + r + 1):
            dx, dy = px - x, py - y
            t = (dx * ca + dy * sa) / max(L - 1, 1)
            q = dx * nx + dy * ny
            if t < -0.15 or t > 1.15:
                continue
            wid = wmax * np.sin(np.pi * np.clip(t, 0, 1)) ** 0.6 + 0.35
            if abs(q) <= wid * 0.75:
                role = "T" if q < 0.25 else "m"
                if tip_dark and (t > 0.9):
                    role = "m"
                pts[(px, py)] = role
    return pts


def near_leaves2(region, light, rng, top=3, spacing=2.2, lens=(3, 6), wmax=(0.8, 1.6),
                 angles=((0, 3), (26.6, 3), (45, 2), (63, 1)), dens=(0.25, 1.0),
                 base1=(0.2, 0.6), mask=None, jitter=0.7, tuck_p=0.5, bright=0.5, sd=0.55):
    """Leaf lozenges, lit upper half / darker lower half, painted dim-to-bright."""
    h, w = light.shape
    if mask is None:
        mask = np.ones((h, w), bool)
    S = np.zeros((h, w), int)
    n = value_noise((h, w), 2.0, rng)
    t = 0.5 * rng.random((h, w)) + 0.5 * (n - n.min()) / (np.ptp(n) + 1e-9)
    S[(t < base1[0] + (base1[1] - base1[0]) * light) & mask] = 1
    ys, xs = np.mgrid[0:h:spacing, 0:w:spacing]
    ys = ys.ravel() + rng.uniform(-spacing, spacing, ys.size) * jitter
    xs = xs.ravel() + rng.uniform(-spacing, spacing, xs.size) * jitter
    angs = np.array([a for a, _ in angles]); wts = np.array([k for _, k in angles], float); wts /= wts.sum()
    G = []
    for (x, y) in zip(xs, ys):
        xi, yi = int(np.clip(round(x), 0, w - 1)), int(np.clip(round(y), 0, h - 1))
        if not mask[yi, xi]:
            continue
        l = light[yi, xi]
        if rng.random() > dens[0] + (dens[1] - dens[0]) * l:
            continue
        tp = int(np.clip(round(bright + l * top + rng.normal(0, sd)), 1, top))
        L = rng.uniform(*lens)
        a = np.radians(rng.choice(angs, p=wts))
        a = -a if rng.random() < 0.5 else a
        if rng.random() < 0.5:
            a = np.pi - a
        G.append((tp + rng.random(), x, y, L, a, rng.uniform(*wmax), tp))
    G.sort()
    for (_, x, y, L, a, wm, tp) in G:
        pts = lozenge(x, y, L, a, wm)
        low = {}
        for (px, py), role in pts.items():
            if 0 <= px < w and 0 <= py < h and mask[py, px]:
                S[py, px] = tp if role == "T" else max(0, tp - 1)
                low[px] = max(low.get(px, -1), py)
        for px, py in low.items():
            if rng.random() < tuck_p and 0 <= py + 1 < h and (px, py + 1) not in pts and mask[py + 1, px]:
                S[py + 1, px] = min(S[py + 1, px], max(0, tp - 2))
    return S


def poisson(h, w, r, rng, k=20, mask=None):
    """Bridson Poisson-disc sample of points (x, y) with min distance r."""
    cell = r / np.sqrt(2)
    gw, gh = int(w / cell) + 1, int(h / cell) + 1
    grid = -np.ones((gh, gw), int)
    pts = []
    active = []
    def ok(p):
        gx, gy = int(p[0] / cell), int(p[1] / cell)
        for yy in range(max(0, gy - 2), min(gh, gy + 3)):
            for xx in range(max(0, gx - 2), min(gw, gx + 3)):
                j = grid[yy, xx]
                if j >= 0 and (pts[j][0] - p[0]) ** 2 + (pts[j][1] - p[1]) ** 2 < r * r:
                    return False
        return True
    p0 = (rng.uniform(0, w), rng.uniform(0, h))
    pts.append(p0); active.append(0); grid[int(p0[1] / cell), int(p0[0] / cell)] = 0
    while active:
        i = active[int(rng.integers(len(active)))]
        found = False
        for _ in range(k):
            a = rng.uniform(0, 2 * np.pi); d = rng.uniform(r, 2 * r)
            p = (pts[i][0] + d * np.cos(a), pts[i][1] + d * np.sin(a))
            if 0 <= p[0] < w and 0 <= p[1] < h and ok(p):
                pts.append(p); active.append(len(pts) - 1)
                grid[int(p[1] / cell), int(p[0] / cell)] = len(pts) - 1
                found = True
                break
        if not found:
            active.remove(i)
    pts = np.array(pts)
    if mask is not None and len(pts):
        keep = mask[np.clip(pts[:, 1].astype(int), 0, h - 1), np.clip(pts[:, 0].astype(int), 0, w - 1)]
        pts = pts[keep]
    return pts


def paint_leaf(S, mask, x, y, L, a, wm, tp, rng, tuck_p=0.5, extra=None, drop=1):
    h, w = S.shape
    pts = lozenge(x, y, L, a, wm)
    low = {}
    for (px, py), role in pts.items():
        if 0 <= px < w and 0 <= py < h and mask[py, px]:
            S[py, px] = tp if role == "T" else max(0, tp - drop)
            low[px] = max(low.get(px, -1), py)
            if extra is not None:
                extra[py, px] = 1
    for px, py in low.items():
        if rng.random() < tuck_p and 0 <= py + 1 < h and (px, py + 1) not in pts and mask[py + 1, px]:
            S[py + 1, px] = min(S[py + 1, px], max(0, tp - 2))


def near_sprays(region, light, rng, top=3, spray_r=6.0, spray_size=(2.5, 4.5), leaves=(4, 9),
                lens=(3, 5.5), wmax=(0.8, 1.5), up_bias=0.8, dens=(0.35, 1.0), base1=(0.1, 0.4),
                mask=None, bright=1.0, sd=0.45, grad=1.2, tuck_p=0.5, fill_leaves=0.0, stem_p=0.0, stem_len=(2, 5), drop=1, minstep=1):
    """Leaves grouped in sprays: spray centres on a Poisson disc; each spray's leaves radiate from
    its centre, tips biased upward; leaves at the top of a spray are brighter (grad steps across
    the spray). Sprays painted dim-to-bright, leaves inside a spray bottom-to-top."""
    h, w = light.shape
    if mask is None:
        mask = np.ones((h, w), bool)
    S = np.zeros((h, w), int)
    n = value_noise((h, w), 2.0, rng)
    t = 0.5 * rng.random((h, w)) + 0.5 * (n - n.min()) / (np.ptp(n) + 1e-9)
    S[(t < base1[0] + (base1[1] - base1[0]) * light) & mask] = 1
    C = poisson(h, w, spray_r, rng, mask=mask)
    sprays = []
    for (cx, cy) in C:
        l = light[int(cy), int(cx)]
        if rng.random() > dens[0] + (dens[1] - dens[0]) * l:
            continue
        base = bright + l * top + rng.normal(0, sd)
        sprays.append((base, cx, cy))
    sprays.sort()
    for (base, cx, cy) in sprays:
        R = rng.uniform(*spray_size)
        nl = int(rng.integers(leaves[0], leaves[1] + 1))
        Ls = []
        for i in range(nl):
            a = rng.uniform(0, 2 * np.pi)
            d = R * np.sqrt(rng.uniform(0.0, 1))
            x, y = cx + d * np.cos(a), cy + d * np.sin(a) * 0.8
            # leaf direction: outward from spray centre, pulled upward
            ox, oy = np.cos(a), np.sin(a) - up_bias
            ang = np.arctan2(oy, ox)
            rel = (cy - y) / max(R, 1)          # +1 at top of spray
            tp = int(np.clip(round(base + grad * 0.5 * rel), minstep, top))
            Ls.append((-y, x, y, ang, tp))
        Ls.sort()
        if rng.random() < stem_p:     # a petiole/stem under the spray, 1 px, one step below the spray
            sl = int(rng.integers(stem_len[0], stem_len[1] + 1))
            sx = int(round(cx)); sy = int(round(cy))
            lean = rng.choice([0, 0, 1, -1])
            for k, (px, py) in enumerate(clean_line(sx, sy, sx + lean * sl / 2, sy + sl)):
                if 0 <= px < w and 0 <= py < h and mask[py, px]:
                    S[py, px] = int(np.clip(round(base) - 1, 1, top - 1))
        for (_, x, y, ang, tp) in Ls:
            paint_leaf(S, mask, x, y, rng.uniform(*lens), ang, rng.uniform(*wmax), tp, rng, tuck_p, drop=drop)
    return S


MID_GLYPHS = {   # small marks at middle distance: offsets (dx, dy, role) role 0 = top pixel, 1 = body
    "dot": [(0, 0, 0)],
    "h2": [(0, 0, 0), (1, 0, 0)],
    "v2": [(0, 0, 0), (0, 1, 1)],
    "d2": [(0, 0, 0), (1, 1, 1)],
    "a2": [(0, 0, 0), (-1, 1, 1)],
    "h3": [(0, 0, 0), (1, 0, 0), (2, 0, 1)],
    "cap": [(0, 0, 0), (1, 0, 0), (-1, 1, 1), (2, 1, 1)],
    "hook": [(0, 0, 0), (1, 0, 0), (1, 1, 1)],
    "d3": [(0, 0, 0), (1, 1, 0), (2, 2, 1)],
    "s3": [(0, 0, 0), (1, 0, 0), (2, 1, 0), (3, 1, 1)],
    "arc": [(-1, 1, 1), (0, 0, 0), (1, 0, 0), (2, 1, 1)],
}


def mid_speckle(region, light, rng, top=4, mask=None, base=(0.25, 0.75), dens=(0.05, 0.55), base_steps=(0, 1),
                dirs=None, dlen=(2, 3), droop=(20, 60), dir_p=1.0, dens_lo=0.0, dens_gamma=1.0, dark_below=-1.0,
                glyphs=(("dot", 4), ("h2", 3), ("v2", 2), ("d2", 1), ("a2", 1), ("h3", 1), ("cap", 1), ("hook", 1)),
                hi=(0.45, 0.9), body_drop=1, mid_p=0.15, spacing=1.6, jitter=0.7, clump=0.5, tuck_p=0.3):
    """Middle-distance foliage: a dithered dark base (steps 0/1, share of 1 rising with light) and
    small lit marks of 1-4 px at the top two steps (the top step where the light is high), rarely
    the middle step. Marks gather where the light is (density rises with light, clumped by noise)."""
    h, w = light.shape
    if mask is None:
        mask = np.ones((h, w), bool)
    S = np.zeros((h, w), int)
    n = value_noise((h, w), 1.5, rng)
    t = 0.55 * rng.random((h, w)) + 0.45 * (n - n.min()) / (np.ptp(n) + 1e-9)
    S[mask] = base_steps[0]
    S[(t < base[0] + (base[1] - base[0]) * light) & mask & (light > dark_below)] = base_steps[1]
    cn = value_noise((h, w), 3.0, rng); cn = (cn - cn.min()) / (np.ptp(cn) + 1e-9)
    names = [g for g, _ in glyphs]; wts = np.array([k for _, k in glyphs], float); wts /= wts.sum()
    ys, xs = np.mgrid[0:h:spacing, 0:w:spacing]
    ys = ys.ravel() + rng.uniform(-1, 1, ys.size) * spacing * jitter
    xs = xs.ravel() + rng.uniform(-1, 1, xs.size) * spacing * jitter
    order = rng.permutation(len(xs))
    marks = []
    for i in order:
        xi, yi = int(round(xs[i])), int(round(ys[i]))
        if not (0 <= xi < w and 0 <= yi < h) or not mask[yi, xi]:
            continue
        l = light[yi, xi]
        lp = np.clip((l - dens_lo) / max(1e-6, 1 - dens_lo), 0, 1) ** dens_gamma
        p = dens[0] + (dens[1] - dens[0]) * lp
        p *= (1 - clump) + clump * 2 * cn[yi, xi]
        if rng.random() > p:
            continue
        # top step where light is high, else the step below; rarely the middle step
        ph = np.clip((l - hi[0]) / (hi[1] - hi[0]), 0, 1)
        tp = top if rng.random() < ph else top - 1
        if rng.random() < mid_p:
            tp = top - 2
        marks.append((tp, xi, yi, names[rng.choice(len(names), p=wts)]))
    marks.sort(key=lambda m: m[0])
    for tp, xi, yi, g in marks:
        flip = rng.random() < 0.5
        glyph = MID_GLYPHS[g]
        if dirs is not None and rng.random() < dir_p:
            # a leaf hanging off the wand: the wand direction turned down by 20..60 degrees
            wx, wy = dirs[0][yi, xi], dirs[1][yi, xi]
            a = np.arctan2(wy, wx)
            down = 1 if np.sin(a + 0.7) > np.sin(a - 0.7) else -1
            side = down if rng.random() < 0.85 else -down
            a2 = a + side * np.radians(rng.uniform(*droop))
            # droop: bias the leaf toward pointing down (screen +y)
            vx, vy = np.cos(a2), np.sin(a2) + 0.6
            ang = snap_dir(vx, vy)
            Lg = int(rng.integers(dlen[0], dlen[1] + 1))
            pts = stroke_run(0, 0, ang, Lg)
            glyph = [(px, py, 0 if k < Lg - 1 or Lg < 3 else 1) for k, (px, py) in enumerate(pts)]
            flip = False
        for dx, dy, role in glyph:
            px, py = xi + (-dx if flip else dx), yi + dy
            if 0 <= px < w and 0 <= py < h and mask[py, px]:
                S[py, px] = tp if role == 0 else max(0, tp - body_drop)
        if rng.random() < tuck_p and 0 <= yi + 1 < h and mask[yi + 1, xi] and S[yi + 1, xi] < 2:
            S[yi + 1, xi] = 0
    return S


def fringe(S, mask, light, rng, top, hole_p=0.2, bump_p=0.3, bump_w=(1, 4), peak_p=0.3, rim_p=0.6,
           side_scale=0.5, rim_steps=(1, 0), depth=2):
    """A leafy mass has no outline: its boundary is where the texture ends.
    - crest pixels (open above) catch the light: top or top-1 step with prob rim_p*light
    - bumps: 1..bump_h px of lit marks rise above the crest (a pixel, a pair, a little cap)
    - holes: background opens 1-2 px into the mass just under the crest, a lit peak behind
    Side edges get the same at side_scale. Returns (S, mask) both new arrays."""
    h, w = S.shape
    S = S.copy(); M = mask.copy()
    up = np.zeros_like(mask); up[1:] = mask[1:] & ~mask[:-1]; up[0] = False
    left = np.zeros_like(mask); left[:, 1:] = mask[:, 1:] & ~mask[:, :-1]
    right = np.zeros_like(mask); right[:, :-1] = mask[:, :-1] & ~mask[:, 1:]
    side = (left | right) & ~up
    ys, xs = np.nonzero(up | side)
    for y, x in zip(ys, xs):
        l = light[y, x]
        sc = 1.0 if up[y, x] else side_scale
        r = rng.random()
        if r < hole_p * sc:
            # open a hole 1-2 px deep; the pixel below the hole becomes a lit peak edge
            d = 1 if rng.random() < 0.6 else 2
            for k in range(d):
                if y + k < h:
                    M[y + k, x] = False
            if y + d < h and M[y + d, x] and rng.random() < 0.5:
                S[y + d, x] = max(S[y + d, x], top - 1 - rim_steps[1])
        elif r < (hole_p + bump_p) * sc:
            tp = top if rng.random() < l else top - 1
            if up[y, x]:
                # a cap: 1-4 px wide, 1 px tall, sometimes a single pixel on top of it
                cw = int(rng.integers(bump_w[0], bump_w[1] + 1))
                x0 = x - int(rng.integers(0, cw))
                for xx in range(x0, x0 + cw):
                    if 0 <= xx < w and y - 1 >= 0 and not M[y - 1, xx]:
                        M[y - 1, xx] = True; S[y - 1, xx] = tp if rng.random() < 0.7 else max(1, tp - 1)
                if cw >= 2 and rng.random() < peak_p and y - 2 >= 0:
                    xx = x0 + int(rng.integers(0, cw))
                    if 0 <= xx < w:
                        M[y - 2, xx] = True; S[y - 2, xx] = tp
            else:
                xx = x - 1 if left[y, x] else x + 1
                if 0 <= xx < w:
                    M[y, xx] = True; S[y, xx] = tp
        if M[y, x] and rng.random() < rim_p * l * sc:
            S[y, x] = max(S[y, x], top - rim_steps[0] if rng.random() < l else top - 1)
    return S, M


def rim_arcs(S, mask, light, rng, top, r_sep=7.0, rad=(3.5, 8.0), squash=0.6, dens=(0.0, 0.9),
             a0=(150, 175), a1=(10, 60), break_p=0.15, min_light=0.35, step_hi=0.7):
    """Sub-lobe rims: the upper edge of each small lobe inside the mass catches the light as a
    broken 1-px arc (clean runs), drawn only where the light field is high enough."""
    h, w = S.shape
    S = S.copy()
    C = poisson(h, w, r_sep, rng, mask=mask)
    for (cx, cy) in C:
        l = light[int(cy), int(cx)]
        if l < min_light or rng.random() > dens[0] + (dens[1] - dens[0]) * l:
            continue
        R = rng.uniform(*rad)
        s0, s1 = np.radians(rng.uniform(*a0)), np.radians(rng.uniform(*a1))
        n = max(3, int(R * (s0 - s1) / 1.5))
        pts = [(cx + R * np.cos(t), cy - R * squash * np.sin(t)) for t in np.linspace(s0, s1, n)]
        from wpaint import polyline_clean
        pix = polyline_clean([(round(x), round(y)) for x, y in pts])
        on = True
        for (px, py) in pix:
            if rng.random() < break_p:
                on = not on
            if not on or not (0 <= px < w and 0 <= py < h) or not mask[py, px]:
                continue
            ll = light[py, px]
            if ll < min_light * 0.8:
                continue
            S[py, px] = top if (ll > step_hi and rng.random() < ll) else top - 1
    return S


def snap_dir(dx, dy):
    """Snap a direction to the clean-run set: (step pattern) for 1-px strokes."""
    ang = np.degrees(np.arctan2(dy, dx)) % 360
    cands = [0, 26.57, 45, 63.43, 90, 116.57, 135, 153.43, 180, 206.57, 225, 243.43, 270, 296.57, 315, 333.43]
    a = min(cands, key=lambda c: min(abs(c - ang), 360 - abs(c - ang)))
    return np.radians(a)


def stroke_run(x, y, ang, L):
    """Pixels of a clean 1-px stroke starting at (x,y) going along ang for L pixels."""
    ca, sa = np.cos(ang), np.sin(ang)
    pts = []
    if abs(ca) < 1e-6 or abs(sa) < 1e-6 or abs(abs(ca) - abs(sa)) < 1e-6:
        sx, sy = int(np.sign(round(ca, 6))), int(np.sign(round(sa, 6)))
        for k in range(L):
            pts.append((x + sx * k, y + sy * k))
        return pts
    # 2:1 slopes: runs of 2 along the major axis
    if abs(ca) > abs(sa):
        sx, sy = int(np.sign(ca)), int(np.sign(sa))
        for k in range(L):
            pts.append((x + sx * k, y + sy * (k // 2)))
    else:
        sx, sy = int(np.sign(ca)), int(np.sign(sa))
        for k in range(L):
            pts.append((x + sx * (k // 2), y + sy * k))
    return pts


def stroke_marks(mask, light, dirx, diry, rng, top, base_steps=(0, 1), base=(0.2, 0.8),
                 dens=(0.05, 0.6), lens=(2, 4), hi=(0.45, 0.9), spacing=1.8, jitter=0.8,
                 splay=25.0, pair_p=0.6, gap_p=0.5, clump=0.5, mid_p=0.1, S=None,
                 out_mask=None, light_side=(-1.0, -1.0)):
    """Directional strokes (leaves along wands): a 1-px clean run along the local wand direction,
    splayed +-splay degrees, lit at its outer end (the tip), one step darker at its root; a darker
    partner pixel beside it on the side away from the light, and a dark gap pixel on the other side.
    Strokes may run past the mask into out_mask (the fringe of wand tips)."""
    h, w = mask.shape
    if S is None:
        S = -np.ones((h, w), int)
    n = value_noise((h, w), 1.5, rng)
    t = 0.55 * rng.random((h, w)) + 0.45 * (n - n.min()) / (np.ptp(n) + 1e-9)
    S[mask] = base_steps[0]
    S[(t < base[0] + (base[1] - base[0]) * light) & mask] = base_steps[1]
    cn = value_noise((h, w), 3.0, rng); cn = (cn - cn.min()) / (np.ptp(cn) + 1e-9)
    allowed = mask if out_mask is None else (mask | out_mask)
    ys, xs = np.mgrid[0:h:spacing, 0:w:spacing]
    ys = ys.ravel() + rng.uniform(-1, 1, ys.size) * spacing * jitter
    xs = xs.ravel() + rng.uniform(-1, 1, xs.size) * spacing * jitter
    marks = []
    for i in rng.permutation(len(xs)):
        xi, yi = int(round(xs[i])), int(round(ys[i]))
        if not (0 <= xi < w and 0 <= yi < h) or not mask[yi, xi]:
            continue
        l = light[yi, xi]
        p = (dens[0] + (dens[1] - dens[0]) * l) * ((1 - clump) + clump * 2 * cn[yi, xi])
        if rng.random() > p:
            continue
        ph = np.clip((l - hi[0]) / (hi[1] - hi[0] + 1e-9), 0, 1)
        tp = top if rng.random() < ph else top - 1
        if rng.random() < mid_p:
            tp = top - 2
        marks.append((tp, xi, yi))
    marks.sort()
    lsx, lsy = light_side
    for tp, xi, yi in marks:
        dx, dy = dirx[yi, xi], diry[yi, xi]
        a0 = np.arctan2(dy, dx) + np.radians(rng.uniform(-splay, splay))
        ang = snap_dir(np.cos(a0), np.sin(a0))
        L = int(rng.integers(lens[0], lens[1] + 1))
        pts = stroke_run(xi, yi, ang, L)
        # side away from light (screen): perpendicular with positive dot to -light_side
        px_, py_ = -np.sin(ang), np.cos(ang)
        if px_ * lsx + py_ * lsy > 0:
            px_, py_ = -px_, -py_
        ox, oy = int(round(px_)), int(round(py_))
        for k, (x, y) in enumerate(pts):
            if not (0 <= x < w and 0 <= y < h) or not allowed[y, x]:
                break
            s = tp if k >= L - 1 - (L > 3) else max(base_steps[1], tp - 1)
            S[y, x] = s
            if rng.random() < pair_p:
                qx, qy = x + ox, y + oy
                if 0 <= qx < w and 0 <= qy < h and mask[qy, qx] and S[qy, qx] < s:
                    S[qy, qx] = max(base_steps[0], s - 2)
            if rng.random() < gap_p:
                qx, qy = x - ox, y - oy
                if 0 <= qx < w and 0 <= qy < h and mask[qy, qx] and S[qy, qx] < s - 1:
                    S[qy, qx] = base_steps[0]
    return S


def wand_tips(S, mask, dirx, diry, light, rng, top, p=0.35, lens=(2, 6), leaf_p=0.5, lo=1,
              up_only=True, stem_step=None):
    """The willow crest: wand tips run out past the mass, 1-px clean runs along the wand
    direction, with leaflets off their sides and a lit tip. Returns (S, M)."""
    h, w = S.shape
    S = S.copy(); M = mask.copy()
    dist_out = ndi.distance_transform_edt(~mask)
    gy_, gx_ = np.gradient(ndi.gaussian_filter(mask.astype(float), 1.5))
    edge = mask & ndi.binary_dilation(~mask)
    ys, xs = np.nonzero(edge)
    for y, x in zip(ys, xs):
        ox, oy = -gx_[y, x], -gy_[y, x]            # outward normal
        on = np.hypot(ox, oy) + 1e-9; ox /= on; oy /= on
        if up_only and oy > 0.35:
            continue
        dx, dy = dirx[y, x], diry[y, x]
        if dx * ox + dy * oy < 0.2:
            continue
        if rng.random() > p:
            continue
        l = light[y, x]
        ang = snap_dir(dx + 0.3 * ox, dy + 0.3 * oy)
        L = int(rng.integers(lens[0], lens[1] + 1))
        pts = stroke_run(x, y, ang, L + 1)[1:]
        tp = top if rng.random() < l else top - 1
        ss = stem_step if stem_step is not None else max(lo, tp - 2)
        for k, (px, py) in enumerate(pts):
            if not (0 <= px < w and 0 <= py < h) or M[py, px] and k > 0:
                break
            last = k == len(pts) - 1
            M[py, px] = True
            S[py, px] = tp if last or (k >= len(pts) - 2 and rng.random() < 0.5) else ss
            if k % 2 == 1 and rng.random() < leaf_p:
                sd = 1 if rng.random() < 0.5 else -1
                qx, qy = px + int(round(-np.sin(ang) * sd + np.cos(ang))), py + int(round(np.cos(ang) * sd + np.sin(ang)))
                if 0 <= qx < w and 0 <= qy < h and not M[qy, qx]:
                    M[qy, qx] = True; S[qy, qx] = tp if rng.random() < l else max(lo, tp - 1)
    return S, M


def clump_foliage(mask, light, rng, top, r=(3.0, 6.0), sep=0.85, aspect=1.3, L2=(-0.6, -0.8),
                  cap=0.55, body=0.15, dark_band=0.55, rough=0.35, mark_hole=0.25, cap_marks=0.75,
                  steps=None, S=None, M=None, dirs=None, order="y", grow=1.0, lift=0.0, light_gain=1.0,
                  dither=0.35, noise_scale=1.3, keep_inside=None, vo=0.3, vg=2.6, vl=1.4, vb=1.6, hole_drop=2.0, field=None):
    """The clump grammar. Foliage is a heap of clumps (r px), painted top-to-bottom so lower clumps
    lie in front. Each clump: an irregular blob (noise-roughened ellipse, wider than tall or
    stretched along `dirs`), lit cap toward the light L2 (screen dir toward light, y down),
    mid body, dark band along its bottom that separates it from the clump below. The cap is made
    of lit marks with small dark holes (not a flat fill). `light` (0..1) at the clump centre sets
    how bright its cap gets (top..top-2) and whether it has one.
    steps: dict with 'dark','body','cap' lists of steps (low..high) or None for defaults."""
    h, w = mask.shape
    if S is None:
        S = -np.ones((h, w), int)
    if M is None:
        M = np.zeros((h, w), bool)
    st = steps or {}
    dark = st.get("dark", [0, 1]); bodys = st.get("body", [1, 2]); caps = st.get("cap", [top - 1, top])
    C = poisson(h, w, sep * (r[0] + r[1]) / 2, rng, mask=mask)
    if len(C) == 0:
        return S, M
    lx, ly = L2
    ln = np.hypot(lx, ly) + 1e-9; lx /= ln; ly /= ln
    nz = value_noise((h, w), noise_scale, rng)
    nz = (nz - nz.min()) / (np.ptp(nz) + 1e-9)
    key = C[:, 1] + rng.normal(0, 0.8, len(C)) if order == "y" else rng.random(len(C))
    for i in np.argsort(key):
        cx, cy = C[i]
        g = float(np.clip(light[int(cy), int(cx)] * light_gain, 0, 1))
        rr = rng.uniform(*r) * grow
        a, b = rr * aspect, rr
        th = 0.0
        if dirs is not None:
            dx, dy = dirs[0][int(cy), int(cx)], dirs[1][int(cy), int(cx)]
            th = np.arctan2(dy, dx) + np.pi / 2       # long axis along the wand
            a, b = rr, rr * aspect
        cy2 = cy - lift * rr
        R = int(np.ceil(max(a, b))) + 2
        x0, x1 = max(0, int(cx) - R), min(w, int(cx) + R + 1)
        y0, y1 = max(0, int(cy2) - R), min(h, int(cy2) + R + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(float)
        ddx, ddy = xx - cx, yy - cy2
        ct, sn = np.cos(th), np.sin(th)
        u = (ddx * ct + ddy * sn) / a
        v = (-ddx * sn + ddy * ct) / b
        d2 = u * u + v * v
        edge_n = nz[y0:y1, x0:x1]
        inside = d2 < (1.0 - rough * 0.5 + rough * edge_n) ** 2
        if keep_inside is not None:
            inside &= keep_inside[y0:y1, x0:x1] | (d2 < 0.5)
        if not inside.any():
            continue
        # clump-local light: sphere normal in screen coords (x right, y down) dotted with L2 (+ view)
        sx, sy = ddx / max(a, b), ddy / max(a, b)
        nzc = np.sqrt(np.clip(1 - sx * sx - sy * sy, 0, 1))
        lc = sx * lx + sy * ly + 0.35 * nzc          # -1..1.35
        vv = (ddy / b)                               # -1 top .. +1 bottom (screen)
        ns = rng.random(inside.shape)
        out = np.full(inside.shape, -1)
        # continuous value: clump's global light + its own sphere light, the dark band at its bottom
        band = np.clip((vv - (1 - dark_band * 2)) / (dark_band * 2 + 1e-9), 0, 1)
        val = vo + g * vg + lc * vl - band * vb + (ns - 0.5) * dither * 2
        val = np.where(val > top - 0.5, np.where(rng.random(inside.shape) < mark_hole, val - hole_drop, val), val)
        out = np.clip(np.round(val), 0, top).astype(int)
        if field is not None:
            fv = np.clip((vo + g * vg + lc * vl - band * vb) / top, 0, 1)
            fsub = field[y0:y1, x0:x1]; fsub[inside] = fv[inside]
        sub = S[y0:y1, x0:x1]; subm = M[y0:y1, x0:x1]
        sub[inside] = out[inside]; subm[inside] = True
    return S, M


def willow_spray(S, allowed, x, y, ang, L, top, rng, leaf_len=(2, 3), leaf_ang=(30, 55), every=(1, 2),
                 droop=0.5, twig_step=None, tuck_p=0.35, lit_side=(-0.6, -0.8), dim=0, painted=None, silver_p=0.0, one_side=0.85):
    """A willow spray: a 1-px twig along `ang` for L px (clean run), narrow leaves (2-3 px, 1 px wide)
    leaving it every 1-2 px, alternating sides, angled 30-55 deg off the twig and pulled downward
    (they hang). Leaves on the side toward the light take the top step, the others one below; the
    tip leaf is lit; a dark tuck under some leaves. `dim` lowers every step (shadow sprays)."""
    h, w = S.shape
    twig = stroke_run(int(round(x)), int(round(y)), snap_dir(np.cos(ang), np.sin(ang)), L)
    ts = (top - 2 if twig_step is None else twig_step) - dim
    lx, ly = lit_side
    side = 1 if rng.random() < 0.5 else -1
    k = 0
    def put(px, py, s):
        if 0 <= px < w and 0 <= py < h and allowed[py, px]:
            S[py, px] = max(0, s)
            if painted is not None:
                painted[py, px] = True
            return True
        return False
    for i, (px, py) in enumerate(twig):
        put(px, py, ts)
    i = int(rng.integers(0, 2))
    while i < len(twig):
        px, py = twig[i]
        # leaves hang: mostly on the side of the twig that points them downward, so neighbouring
        # leaves run near-parallel instead of crossing in X's
        down = 1 if np.sin(ang + np.radians(40)) > np.sin(ang - np.radians(40)) else -1
        sd = down if rng.random() < one_side else side
        a = ang + sd * np.radians(rng.uniform(*leaf_ang))
        vx, vy = np.cos(a), np.sin(a) + droop
        la = snap_dir(vx, vy)
        ll = int(rng.integers(leaf_len[0], leaf_len[1] + 1))
        pts = stroke_run(px, py, la, ll + 1)[1:]
        # which side faces the light: the leaf's normal (up side) against lit_side
        nx, ny = np.sin(la), -np.cos(la)
        facing = (nx * lx + ny * ly) * (1 if ny < 0 else -1)
        s = top if (facing > -0.2 or i >= len(twig) - 2) else top - 1
        s -= dim
        # a leaf turned so its pale underside shows (hanging on the lower side of the twig)
        silver = facing <= -0.2 and rng.random() < silver_p
        for q, (qx, qy) in enumerate(pts):
            v = s if q < len(pts) - 1 or len(pts) < 3 else s - 1
            if put(qx, qy, v) and silver:
                S[qy, qx] = 10 + max(0, v)      # marker: the painter maps 10+ to the SILVER ramp
        if rng.random() < tuck_p and pts:
            qx, qy = pts[-1]
            if 0 <= qy + 1 < h and 0 <= qx < w and allowed[qy + 1, qx] and S[qy + 1, qx] < s - 1:
                S[qy + 1, qx] = max(0, min(S[qy + 1, qx], 0))
        side = -side
        i += int(rng.integers(every[0], every[1] + 1))
    # the tip: the last twig pixel carries a lit leaf point
    if twig:
        put(*twig[-1], top - dim)


def spray_marks(mask, light, dirs, rng, top, r_sep=4.5, dens=(0.0, 1.0), dens_lo=0.3, length=(4, 8),
                base=(0.2, 0.8), base_steps=(0, 1), lit_side=(-0.6, -0.8), dim_below=0.45, S=None, **kw):
    """Sprays laid on the clump field: dense and lit on clump caps, dim and sparse below, none in
    the dark bands. Over a dark base dithered between two close steps."""
    h, w = mask.shape
    if S is None:
        S = -np.ones((h, w), int)
    n = value_noise((h, w), 1.5, rng)
    t = 0.55 * rng.random((h, w)) + 0.45 * (n - n.min()) / (np.ptp(n) + 1e-9)
    S[mask] = base_steps[0]
    S[(t < base[0] + (base[1] - base[0]) * light) & mask] = base_steps[1]
    C = poisson(h, w, r_sep, rng, mask=mask)
    order = []
    for (cx, cy) in C:
        l = light[int(cy), int(cx)]
        lp = np.clip((l - dens_lo) / (1 - dens_lo), 0, 1)
        if rng.random() > dens[0] + (dens[1] - dens[0]) * lp:
            continue
        order.append((l, cx, cy))
    order.sort()
    for l, cx, cy in order:
        dx, dy = dirs[0][int(cy), int(cx)], dirs[1][int(cy), int(cx)]
        ang = np.arctan2(dy, dx)
        L = int(rng.integers(length[0], length[1] + 1))
        # start the spray behind its centre so it is centred on the clump point
        x0, y0 = cx - np.cos(ang) * L * 0.5, cy - np.sin(ang) * L * 0.5
        dim = 0 if l > dim_below + 0.2 else (1 if l > dim_below else 2)
        willow_spray(S, mask, x0, y0, ang, L, top, rng, lit_side=lit_side, dim=dim, **kw)
    return S
