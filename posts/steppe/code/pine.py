"""pine: ponderosa pine (Pinus ponderosa), open-grown, as in the east-Cascade parkland.

Read from the photos (Turnbull NWR savanna, the 1939 Keen tree-class photos, Dixie Mountain,
the Agropyron/Pinus parkland overlook):
- one straight trunk, thick, bare for the lower third to half of the tree (self-pruned);
  bark in big orange-cinnamon plates split by black furrows (the jigsaw), the plates long and
  vertical; young trees ("blackjacks") are dark grey-brown;
- the crown is OPEN and CLUMPY: a few big branch masses (pads) held out on nearly horizontal
  limbs that droop and then turn up at the ends; sky shows between the masses and the limbs are
  visible as dark lines in the gaps; old trees go flat- or round-topped;
- the needles are long and grouped in tufts at the branch ends: each pad is a heap of bristly
  tufts, lit yellow-green on top and on the sun side, the underside a dark blue-green;
- at every distance the lit tufts catch the light as small starbursts.

Construction (index canvas; NEEDLE steps 0..5, BARK steps 0..4):
  1. trunk: tapered column, bark plates as a vertically stretched Voronoi (furrows = cell edges,
     darkest step), lit across the cylinder;
  2. limbs: clean-run arcs from the trunk, dark;
  3. pads: flattened ellipsoids at the limb ends (and along the outer half), far ones first;
     a calm base in three values (shadow 1 / reflected 2 / lit halftone 3) from the pad normal
     blended with the crown's big form, a dark band along each pad's underside;
  4. tufts: small starburst marks (lit 4, top 5) whose density follows the light, and bristles
     (1-2 px needles) radiating from the pad rims.
"""
import numpy as np
from scipy import ndimage as ndi
import sp
from sp import NEEDLE, BARK
from sage import sun_vec

NS = 6


class PineParams(dict):
    DEFAULT = dict(
        h=120,             # tree height, px
        crown_frac=0.66,   # crown length / height
        width=0.50,        # crown max width / height
        age="mature",      # "young" (conical, dark bark) | "mature" | "old" (flat top, few big limbs)
        limbs=None,        # number of limbs (None: from size)
        pad=0.30,          # pad radius as a fraction of the limb length
        tufts=0.30,        # lit tuft density (x light)
        bristle=0.45,      # rim bristles per rim px
        big_form=0.5,
        term=0.30,
        lean=0.0,
        snow=0.0,
        fill=0.75,         # crown fill with extra pads around the limbs (0: limb-end pads only)
        fill_r=0.11,       # fill pad radius / crown width
        L=None, az=35, el=40,
    )

    def __init__(self, **kw):
        super().__init__(self.DEFAULT)
        self.update(kw)


def _crown_r(t, age):
    """Crown half-width profile (0..1) from crown base t=0 to top t=1."""
    if age == "young":
        return np.clip((1 - t) ** 0.9 * 1.0 + 0.08, 0, 1)
    if age == "old":
        # flat-topped: the upper limbs are the longest (an umbrella of big masses)
        return np.clip((0.45 + 0.6 * t) * (1.0 if t < 0.88 else (1 - t) / 0.12 * 0.7 + 0.3), 0, 1)
    return np.clip(np.sin(np.pi * (0.08 + 0.84 * t)) ** 0.7 * (1.0 - 0.35 * t) + 0.05, 0, 1)


def paint_pine(p, rng, pad=8):
    h = float(p["h"])
    cw = h * p["width"]
    W, H = int(cw * 1.6 + 2 * pad), int(h + 2 * pad)
    cv = sp.Canvas(W, H)
    cx, gy = W / 2.0, H - pad
    L = np.asarray(p["L"]) if p.get("L") is not None else sun_vec(p["az"], p["el"])
    age = p["age"]
    top_y = gy - h
    cb_y = gy - h * (1 - p["crown_frac"])             # crown base
    # trunk centreline
    lean = p["lean"] + rng.normal(0, 0.02)
    def tx(y):
        t = (gy - y) / h
        return cx + lean * (gy - y) + np.sin(t * 5 + rng_phase) * h * 0.004
    rng_phase = rng.uniform(0, 6.28)
    wb = max(1.0, h * (0.05 if age != "young" else 0.03))

    # ---------------------------------------------------------------- trunk + bark
    _trunk(cv, p, rng, tx, gy, top_y, wb, L, h)

    # ---------------------------------------------------------------- limbs and pads
    nl = p["limbs"] or int(np.clip(h * p["crown_frac"] / (7 if age != "old" else 10), 4, 22))
    limbs = []
    for i in range(nl):
        t = (i + rng.uniform(0.2, 0.8)) / nl               # 0 crown base .. 1 top
        y0 = cb_y - t * (cb_y - top_y) * 0.92
        side = rng.choice([-1, 1])
        depth = rng.uniform(-1, 1)                          # -1 far side .. 1 near side
        reach = cw / 2 * _crown_r(t, age) * rng.uniform(0.65, 1.05)
        reach *= np.sqrt(max(0.15, 1 - depth ** 2 * 0.6))  # foreshortened toward/away
        if reach < 2:
            reach = 2
        limbs.append(dict(t=t, y0=y0, side=side, depth=depth, reach=reach))
    # the leader / top tuft
    if age != "old":
        limbs.append(dict(t=1.0, y0=top_y + h * 0.04, side=0, depth=0.2, reach=max(2, cw * 0.12)))

    pads = []
    for lb in limbs:
        x0 = tx(lb["y0"])
        if lb["side"] == 0:
            pts = [(x0, lb["y0"] + 4), (x0, lb["y0"])]
            end = pts[-1]
        else:
            ang = lb["side"] * rng.uniform(1.25, 1.65)       # near horizontal
            bend = -lb["side"] * rng.uniform(0.3, 0.8)       # tips turn up
            pts = sp.arc_points(x0, lb["y0"], ang, lb["reach"], bend, nseg=3)
            # droop in the middle
            pts = [(x, y + np.sin(np.pi * k / 3) * lb["reach"] * 0.08) for k, (x, y) in enumerate(pts)]
            end = pts[-1]
        lb["pts"] = pts
        # pads at the tip and along the outer half
        npad = 1 + (lb["reach"] > 10) + (lb["reach"] > 22 and rng.random() < 0.7)
        for k in range(npad):
            f = 1.0 - k * rng.uniform(0.28, 0.4)
            j = min(len(pts) - 1, int(round(f * (len(pts) - 1))))
            px, py = pts[j]
            r = max(1.8, lb["reach"] * p["pad"] * rng.lognormal(0.0, 0.35) * (1.15 if k == 0 else 0.9))
            if age == "young":
                r *= 0.85
            pads.append(dict(x=px + rng.normal(0, 0.8), y=py - r * 0.18, rx=r, ry=r * rng.uniform(0.5, 0.68),
                             z=lb["depth"] + rng.normal(0, 0.1), t=lb["t"]))
    # fill the crown envelope: most of a ponderosa crown is overlapping tuft masses, the limbs
    # show only in the gaps. Pads cluster around the limbs (clumpy), leaving a few sky holes.
    area = (cb_y - top_y) * cw * 0.6
    small = max(0.0, (90.0 - h) / 90.0)           # small trees: fewer, relatively bigger masses
    rpad = max(2.0, cw * p["fill_r"] * (1 + 0.5 * small))
    nfill = int(area / (np.pi * rpad * rpad * 0.55) * p["fill"] * (1 + 0.9 * small))
    for _ in range(nfill):
        lb = limbs[int(rng.integers(len(limbs)))]
        if not lb.get("pts"):
            continue
        k = rng.uniform(0.35, 1.0)
        j = min(len(lb["pts"]) - 1, int(round(k * (len(lb["pts"]) - 1))))
        px, py = lb["pts"][j]
        r = rpad * rng.lognormal(-0.1, 0.3)
        pads.append(dict(x=px + rng.normal(0, rpad * 0.7), y=py + rng.normal(-rpad * 0.3, rpad * 0.5),
                         rx=r, ry=r * rng.uniform(0.5, 0.7), z=lb["depth"] + rng.normal(0, 0.3),
                         t=lb["t"]))
    # small trees: also fill the crown envelope directly (limbs are too few to carry the mass)
    if small > 0:
        nenv = int(nfill * small * 0.8) + 2
        for _ in range(nenv):
            t = rng.uniform(0.05, 0.95)
            u = rng.uniform(-1, 1) * _crown_r(t, age)
            r = rpad * rng.lognormal(0.0, 0.25)
            pads.append(dict(x=tx(cb_y) + u * cw / 2 * 0.85, y=cb_y - t * (cb_y - top_y), rx=r,
                             ry=r * rng.uniform(0.55, 0.75), z=rng.uniform(-1, 1), t=t))
    # limbs drawn behind pads but in front of the trunk when on the near side
    for lb in sorted(limbs, key=lambda l: l["depth"]):
        if lb["side"] == 0:
            continue
        pix = sp.polyline_clean(lb["pts"])
        thick = 2 if (wb > 3 and lb["reach"] > 14) else 1
        for i, (x, y) in enumerate(pix):
            for k in range(thick if i < len(pix) * 0.5 else 1):
                cv.px(x, y + k, BARK, 0 if k else 1)

    # ---------------------------------------------------------------- pad field
    Y, X = np.mgrid[0:H, 0:W].astype(float)
    f = np.full((H, W), -9.0)
    crown_cy = (cb_y + top_y) / 2
    crown_ry = (cb_y - top_y) / 2 + 1
    bu = (X - cx) / (cw / 2)
    bv = (crown_cy - Y) / crown_ry
    nb = np.stack([bu, bv * 0.9 + 0.25, 0.5 + 0 * bu], -1)
    nb /= np.linalg.norm(nb, axis=-1, keepdims=True)
    pads.sort(key=lambda q: q["z"])
    owner = np.full((H, W), -1, int)
    for i, pd in enumerate(pads):
        x0, x1 = int(max(0, pd["x"] - pd["rx"] - 2)), int(min(W, pd["x"] + pd["rx"] + 3))
        y0, y1 = int(max(0, pd["y"] - pd["ry"] - 2)), int(min(H, pd["y"] + pd["ry"] + 3))
        if x1 <= x0 or y1 <= y0:
            continue
        sub = (slice(y0, y1), slice(x0, x1))
        dx = (X[sub] - pd["x"]) / pd["rx"]
        dy = (Y[sub] - pd["y"]) / pd["ry"]
        ang = np.arctan2(dy, dx)
        lump = 1 + 0.12 * np.sin(5 * ang + rng.uniform(0, 6.28)) + 0.08 * np.sin(8 * ang + rng.uniform(0, 6.28))
        d2 = (dx * dx + dy * dy) / lump ** 2
        inside = d2 <= 1
        nz = np.sqrt(np.clip(1 - d2, 0, 1))
        # pads are flat on top: bias the normal upward
        nl = np.stack([dx / lump, -dy / lump * 1.3 + 0.35, nz + 0.1], -1)
        nl /= np.linalg.norm(nl, axis=-1, keepdims=True)
        nn = nl * (1 - p["big_form"]) + nb[sub] * p["big_form"]
        nn /= np.linalg.norm(nn, axis=-1, keepdims=True)
        lit = np.clip(nn @ L, 0, 1)
        s = np.where(lit < p["term"], 0.8 + 1.6 * lit / p["term"],
                     3.1 + 2.4 * (lit - p["term"]) / (1 - p["term"]))
        s = s - np.clip((dy - 0.05) / 0.95, 0, 1) * 2.2        # the dark underside
        s = s - (1 - (pd["z"] + 1) / 2) * 0.7                   # far-side pads sit in the crown's shade
        fs = f[sub]
        fs[inside] = s[inside]
        f[sub] = fs
        os_ = owner[sub]
        os_[inside] = i
        owner[sub] = os_
    m = f > -9
    q = np.where(f < 1.6, 1, np.where(f < 3.0, 2, 3)).astype(np.int16)
    q = np.where(f < 0.5, 0, q)
    # where a nearer pad overlaps a farther one, its lower rim casts a 1-px dark seam
    edge = m & (ndi.grey_erosion(owner, size=3) != ndi.grey_dilation(owner, size=3))
    q[edge & (f < 3.4)] = np.minimum(q[edge & (f < 3.4)], 1)
    # tufts: starbursts of the lit steps, density following the light
    ys, xs = np.nonzero(m)
    order = rng.permutation(len(ys))
    big = h >= 90
    for i in order:
        y, x = ys[i], xs[i]
        v = f[y, x]
        if v < 3.0:
            if v > 1.8 and rng.random() < 0.05:
                q[y, x] = 3
            elif rng.random() < 0.03:
                q[y, x] = 2
            continue
        t = min(1.0, (v - 3.0) / 2.4)
        if rng.random() < p["tufts"] * (0.3 + 1.5 * t):
            hi = 5 if t > 0.5 else 4
            pts = [(0, 0, hi), (-1, 0, 4), (1, 0, 4)] if not big else \
                  [(0, 0, hi), (-1, 0, 4), (1, 0, 4), (0, -1, 4), (-1, -1, 3), (1, -1, 3)]
            if rng.random() < 0.5:
                pts = pts[:1] + [(rng.choice([-1, 1]), -1, 4)]
            for ddx, ddy, st in pts:
                xx, yy = x + ddx, y + ddy
                if 0 <= xx < W and 0 <= yy < H and m[yy, xx]:
                    q[yy, xx] = max(q[yy, xx], st)
            if y + 1 < H and m[y + 1, x] and q[y + 1, x] <= 3:
                q[y + 1, x] = 1 if t < 0.4 else 2
    cv.put(m, NEEDLE, q)
    # bristles: needles radiating from the pad rims (upward and outward), and gaps
    rim = m & ~ndi.binary_erosion(m)
    ys, xs = np.nonzero(rim)
    for y, x in zip(ys, xs):
        if rng.random() > p["bristle"]:
            continue
        i = owner[y, x]
        if i < 0:
            continue
        pd = pads[i]
        dx, dy = x - pd["x"], (y - pd["y"]) * 1.6
        nrm = np.hypot(dx, dy) + 1e-6
        ux, uy = dx / nrm, dy / nrm
        if uy > 0.55:           # the underside hangs: few bristles below
            if rng.random() > 0.25:
                continue
        ln = 1 + (rng.random() < 0.5) + (big and rng.random() < 0.3)
        st = int(q[y, x])
        for k in range(1, ln + 1):
            xx, yy = int(round(x + ux * k)), int(round(y + uy * k))
            if 0 <= xx < W and 0 <= yy < H and cv.mat[yy, xx] == 0:
                cv.px(xx, yy, NEEDLE, max(1, st - (1 if k == ln else 0)))
    if p["snow"] > 0:
        top = m & ~np.vstack([np.zeros((1, W), bool), m[:-1]])
        ys, xs = np.nonzero(top)
        for y, x in zip(ys, xs):
            if rng.random() < p["snow"]:
                for k in range(1 + (rng.random() < 0.5)):
                    if 0 <= y + k < H and m[y + k, x]:
                        cv.mat[y + k, x] = sp.SNOW
                        cv.step[y + k, x] = 2 if f[y + k, x] > 2.5 else 1
    return cv, int(round(cx)), int(gy), dict(pads=len(pads), limbs=len(limbs))


def _trunk(cv, p, rng, tx, gy, top_y, wb, L, h):
    H, W = cv.h, cv.w
    young = p["age"] == "young"
    # bark plates: Voronoi in a vertically stretched space (plates ~ 3:1)
    plate_w = max(1.6, wb * 0.45)
    npts = int((gy - top_y) * wb * 4 / (plate_w * plate_w * 3) + 20)
    sx = rng.uniform(-wb, wb, npts)
    sy = rng.uniform(top_y, gy + 2, npts)
    for y in range(int(top_y), int(gy) + 2):
        t = (gy - y) / h
        half = max(0.5, wb / 2 * (1 - 0.85 * t) + (wb * 0.25 if (t < 0.03 and wb >= 5) else 0))   # flare at the foot
        xc = tx(y)
        x0, x1 = int(np.floor(xc - half)), int(np.ceil(xc + half))
        for x in range(x0, x1 + 1):
            u = (x + 0.5 - xc) / max(half, 0.5)
            if abs(u) > 1.05:
                continue
            nz = np.sqrt(max(0.0, 1 - min(1, u * u)))
            lit = max(0.0, u * L[0] + nz * L[2] * 0.8 + 0.1)
            base = 1 + 2.6 * lit
            if half >= 1.5 and not young:
                dxs = (x - xc) - sx
                dys = (y - sy) / 3.0
                d = dxs * dxs + dys * dys
                a, b = np.partition(d, 1)[:2]
                furrow = (np.sqrt(b) - np.sqrt(a)) < 0.55
                if furrow:
                    s = 0
                else:
                    s = int(np.clip(round(base + (0.5 if (np.argmin(d) % 3 == 0) else 0)), 1, 4))
            else:
                s = int(np.clip(round(base - (0.6 if young else 0)), 0, 4))
            if young:
                s = min(s, 2)
            cv.px(x, y, BARK, s)
