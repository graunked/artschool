"""sage: big sagebrush (Artemisia tridentata), and bitterbrush as a variant.

A sage bush, read from the photos (Coulee City, Hungate Canyon, Seedskadee), and from those photos
box-downsampled to 64 px (refs/sage_ds_all.png), which is the scale we paint at:
- a few thick, twisted, shreddy grey trunks splay from one root crown and fork upward;
- the leaves are tiny and grey, massed at the twig ends into UPRIGHT SPRAYS: at 64 px a bush is a
  packed row of pale vertical tongues (2-3 px wide, rounded tops) with dark slots between them;
- the sprays gather into 3-7 clumps, so the silhouette is lumpy; each clump's crown is a row
  of rounded finger tips; lower down the sprays thin and the dark stem tangle shows;
- lit sprays are pale and warm, shadowed ones a cool blue-grey.

Construction (index canvas; SAGE steps 0..5, SAGESTEM steps 0..3):
  1. trunks: twisted clean-run lines from the crown to each clump (behind everything);
  2. clump bodies: dark fill (steps 0-1) that shows in the slots between sprays;
  3. sprays: upright capsules, painted far to near; each is lit on its cap and sun side and
     darkens toward its base (the clump grammar at the spray scale); their light comes from the
     clump normal blended with the whole-bush normal (the big form);
  4. twig tips, autumn flower stalks and snow sit on the spray caps.
"""
import numpy as np
from scipy import ndimage as ndi
import sp
from sp import SAGE, SAGESTEM

NS = 6   # SAGE steps: 0 core | 1 shadow | 2 reflected || 3 halftone | 4 light | 5 top light


def sun_vec(az, el):
    """az: 0 = sun on the LEFT, 90 = behind viewer (front light), 180 = right,
    270 = behind the subject (back light). Returns (x right, y up, z toward viewer)."""
    a, e = np.radians(az), np.radians(el)
    v = np.array([-np.cos(a) * np.cos(e), np.sin(e), np.sin(a) * np.cos(e)])
    return v / np.linalg.norm(v)


class SageParams(dict):
    DEFAULT = dict(
        w=48, h=36,            # bush size in px (envelope)
        clumps=None,           # number of clumps (None: from width)
        clump_px=9.0,          # one clump per this many px of width
        density=0.6,           # 0..1: spray fill (low: more slots, more stems)
        spray_w=None,          # spray width px (None: from size)
        spray_h=0.20,          # typical spray height as a fraction of clump height
        base_open=0.16,
        cap_bias=2.6,          # >1: caps crowd toward the clump tops        # fraction of height at the bottom where sprays thin out
        trunks=3,
        big_form=0.55,         # weight of the whole-bush normal against the clump normal
        ambient=0.16,
        fall=2.2,              # steps a spray darkens from cap to base
        noise=0.25,            # step noise per pixel
        tips=0.35,             # 1-px twig tips above caps
        stalks=0.0,            # Aug-Oct flower stalks: 0..1
        az=35, el=40,
        L=None,                # light vector in view space (overrides az/el; set by the camera)
        kind="sage",           # "sage" | "bitter"
        snow=0.0,              # snow on caps: 0..1
        lean=0.35,             # sprays fan outward
        body_max=2,
        term=0.28,             # terminator: light below this is the shadow family
        mode="sprig",          # "sprig" (grouped sprig clumps) | "field" (form field + marks) | "spray"
        sprig=1.25,            # sprig mode: fill (lower = more see-through gaps)
        sprig_px=None,         # sprig width px (None: from size)
        slots=0.05,            # field mode: dark upright slots in the light, per px
        flecks=0.05,           # field mode: pale flecks in the shadow, per px
        fingers=0.5,           # field mode: finger tips on the crest, per top px
        caps=0.16,             # field mode: upright lit marks over a dark tuck, per px (x light)            # highest step the body (between sprays) reaches
    )

    def __init__(self, **kw):
        super().__init__(self.DEFAULT)
        self.update(kw)


def _ellipse_top(cx, cy, rx, ry, x):
    d = 1 - ((x - cx) / rx) ** 2
    return cy - ry * np.sqrt(d) if d > 0 else None


def paint_bush(p, rng, pad=6):
    """Paint one bush into its own canvas. Returns (canvas, base_x, base_y, info) where base is
    the root-crown pixel (for placing on the ground)."""
    w, h = float(p["w"]), float(p["h"])
    W, H = int(w * 1.3 + 2 * pad + 6), int(h * 1.15 + 2 * pad + 8)
    cv = sp.Canvas(W, H)
    cx, gy = W / 2.0 + rng.normal(0, w * 0.02), H - pad
    L = np.asarray(p["L"]) if p.get("L") is not None else sun_vec(p["az"], p["el"])
    bitter = p["kind"] == "bitter"
    sw = p["spray_w"] or (1 if w < 20 else 2 if w < 70 else 3)

    # --- clumps: lumps of the dome
    k = p["clumps"] or int(np.clip(round(w / p["clump_px"] + rng.normal(0, 0.7)), 2, 10))
    clumps = []
    for i in range(k):
        u = -0.78 + 1.56 * (i + rng.uniform(0.25, 0.75)) / k
        env = np.sqrt(max(0.05, 1 - u * u)) ** 0.8
        rx = w / k * rng.uniform(0.75, 1.05) + w * 0.08
        top = gy - h * env * rng.uniform(0.82, 1.0)
        bot = gy - h * (p["base_open"] * (1 - 0.6 * abs(u))) * rng.uniform(0.6, 1.1)
        ry = (bot - top) / 2
        clumps.append(dict(x=cx + u * w / 2 * 0.95, y=(top + bot) / 2, rx=rx * 0.62, ry=ry, u=u))
    # a centre crown clump over them, if tall enough
    if k >= 3 and rng.random() < 0.8:
        clumps.append(dict(x=cx + rng.normal(0, w * 0.06), y=gy - h * 0.62, rx=w * 0.22,
                           ry=h * 0.36, u=0.0))

    # --- trunks (behind)
    for c in clumps:
        pts = [(cx + rng.normal(0, 0.8), gy),
               (cx + (c["x"] - cx) * 0.35 + rng.normal(0, 1.2), gy - (gy - c["y"]) * 0.45),
               (c["x"] + rng.normal(0, 1.0), c["y"] + c["ry"] * 0.2)]
        pix = sp.polyline_clean(pts)
        for i, (x, y) in enumerate(pix):
            thick = 2 if (i < len(pix) * 0.4 and w > 30) else 1
            for t in range(thick):
                cv.px(x + t, y, SAGESTEM, 1 + t)
        # a few side twigs to the clump's lower edge (the tangle)
        for _ in range(2 if not bitter else 4):
            x0, y0 = pix[int(rng.integers(len(pix) // 3, len(pix)))]
            ang = rng.normal(0, 0.8)
            tw = sp.polyline_clean(sp.arc_points(x0, y0, ang, rng.uniform(3, 3 + w * 0.12),
                                                 rng.normal(0, 0.4), 2))
            for x, y in tw:
                cv.px(x, y, SAGESTEM, 0 if rng.random() < 0.5 else 1)

    if p["mode"] == "field":
        _field_marks(cv, clumps, p, rng, cx, gy, w, h, L, sw)
        sprays = []
    elif p["mode"] == "sprig":
        _sprig_marks(cv, clumps, p, rng, cx, gy, w, h, L, sw)
        if w >= 16:
            _open_crest(cv)
        sprays = []
    else:
        sprays = _spray_marks(cv, clumps, p, rng, cx, gy, w, h, L, sw, W, H)

    # --- bitterbrush: more open (holes onto the stems), spiky twig ends
    if bitter:
        m = cv.mat == SAGE
        holes = m & (rng.random((H, W)) < 0.12) & (cv.step <= 2)
        holes = ndi.binary_dilation(holes, [[0, 1, 0], [0, 1, 0], [0, 0, 0]]) & m
        cv.mat[holes] = 0

    # --- twig tips, stalks, snow on the caps
    m = cv.mat == SAGE
    capmask = m & ~np.vstack([np.zeros((1, W), bool), m[:-1]])
    ys, xs = np.nonzero(capmask)
    for y, x in zip(ys, xs):
        if rng.random() < p["tips"] * (1.8 if bitter else 1.0):
            ln = 1 + (rng.random() < (0.5 if bitter else 0.25))
            dx = int(np.sign(x - cx)) if (bitter and rng.random() < 0.5) else 0
            for kk in range(1, ln + 1):
                cv.px(x + (dx if kk == ln else 0), y - kk,
                      SAGESTEM if (bitter and kk == ln) else SAGE,
                      int(max(1, cv.step[y, x] - 1)) if not bitter else 1)
    if p["stalks"] > 0:
        for y, x in zip(ys, xs):
            if y < gy - h * 0.5 and rng.random() < p["stalks"] * 0.18:
                ln = int(rng.integers(2, 4 + int(w / 16)))
                for kk in range(1, ln + 1):
                    cv.px(x, y - kk, sp.FLOWER, 1 if (kk % 2 == 1) else 0)
    if p["snow"] > 0:
        for y, x in zip(ys, xs):
            if rng.random() < p["snow"]:
                for kk in range(0, 1 + (rng.random() < 0.4 * p["snow"])):
                    if 0 <= y + kk < H and cv.mat[y + kk, x] == SAGE:
                        lit_here = cv.step[y + kk, x] >= 3
                        cv.mat[y + kk, x] = sp.SNOW
                        cv.step[y + kk, x] = 2 if lit_here else 1
    return cv, int(round(cx)), int(gy), dict(clumps=clumps, sprays=len(sprays))


def _spray_marks(cv, clumps, p, rng, cx, gy, w, h, L, sw, W, H):
    if True:
    # --- clump bodies: the dark that shows between sprays
        Y, X = np.mgrid[0:H, 0:W].astype(float)
        body = np.zeros((H, W), bool)
        for c in clumps:
            ang = np.arctan2(Y - c["y"], X - c["x"])
            lump = 1 + 0.10 * np.sin(4 * ang + rng.uniform(0, 6.28))
            d = ((X - c["x"]) / c["rx"]) ** 2 + ((Y - c["y"]) / c["ry"]) ** 2
            body |= d <= (0.86 * lump) ** 2
        # the body carries the big form in the low steps (0..body_max); the sprays carry the light
        bu = (X - cx) / (w / 2)
        bv = (gy - Y) / h
        nb = np.stack([bu, bv * 1.3 - 0.15, 0.55 + 0 * bu], -1)
        nb /= np.linalg.norm(nb, axis=-1, keepdims=True)
        val = p["ambient"] + (1 - p["ambient"]) * np.clip(nb @ L, 0, 1)
        bs = val * (p["body_max"] + 0.9) - np.clip((0.3 - bv) / 0.3, 0, 1) * 1.2
        bq = np.clip(sp.quantize(bs, ordered=False, rng=rng), 0, p["body_max"])
        cv.put(body, SAGE, bq)

        # --- sprays
        sprays = []
        for ci, c in enumerate(clumps):
            area = np.pi * c["rx"] * c["ry"]
            n = int(area / (sw * max(3, c["ry"] * p["spray_h"] * 2)) * 1.6 * p["density"]) + 2
            for _ in range(n):
                x = c["x"] + rng.uniform(-1, 1) * c["rx"] * 0.95
                top = _ellipse_top(c["x"], c["y"], c["rx"], c["ry"], x)
                if top is None:
                    continue
                # sprays nearer the viewer have their caps lower down the clump
                bot = c["y"] + c["ry"] * np.sqrt(max(0.0, 1 - ((x - c["x"]) / c["rx"]) ** 2))
                top += (bot - top) * rng.random() ** p["cap_bias"] * 0.8
                ln = max(2.0, c["ry"] * 2 * p["spray_h"] * rng.uniform(0.6, 1.4))
                base = min(top + ln, bot)
                if base - top < 2:
                    base = top + 2
                sprays.append((base, x, top, ci))
        sprays.sort(key=lambda s: s[2])            # higher cap = farther: paint first

        for base, x, top, ci in sprays:
            c = clumps[ci]
            # normal at the spray cap: clump ellipsoid blended with the bush dome
            nx = (x - c["x"]) / c["rx"]
            ny = (c["y"] - top) / c["ry"]
            nloc = np.array([nx, max(ny, 0.1), np.sqrt(max(0.0, 1 - nx * nx - ny * ny)) + 0.2])
            bu = (x - cx) / (w / 2)
            bv = (gy - top) / h
            nbig = np.array([bu, bv * 1.3 - 0.15, 0.55])
            nn = nloc / np.linalg.norm(nloc) * (1 - p["big_form"]) + nbig / np.linalg.norm(nbig) * p["big_form"]
            nn /= np.linalg.norm(nn)
            lit = max(0.0, float(nn @ L))
            # two light families that never touch: shadow sprays live in steps 1-2, lit in 3-5
            if lit < p["term"]:
                cap = 1.0 + 1.6 * (lit / p["term"]) + p["ambient"]
            else:
                cap = 3.2 + 2.3 * (lit - p["term"]) / (1 - p["term"])
            lean = p["lean"] * bu + rng.normal(0, 0.12)
            ys = np.arange(int(round(top)), int(round(base)) + 1)
            sunside = 1 if L[0] > 0 else -1
            for j, y in enumerate(ys):
                t = j / max(1, len(ys) - 1)
                xx = x + lean * (base - y) * 0.35
                wid = sw if j > 0 or sw == 1 else max(1, sw - 1)      # rounded cap
                s0 = cap - p["fall"] * t ** 1.2
                for q in range(wid):
                    px = int(round(xx - (wid - 1) / 2 + q))
                    side = (q - (wid - 1) / 2) * sunside
                    s = s0 + (0.6 if side > 0 else -0.5 if side < 0 else 0) + (0.5 if j == 0 else 0)
                    s += rng.normal(0, p["noise"])
                    cv.px(px, y, SAGE, int(np.clip(np.floor(s + rng.random()), 0, NS - 1)))
                # the slot: a dark pixel on the shadow side of the spray keeps it separate
                if rng.random() < 0.55 and t > 0.15:
                    sx = int(round(xx - sunside * ((wid + 1) / 2)))
                    if 0 <= sx < W and cv.mat[y, sx] == SAGE:
                        cv.step[y, sx] = min(cv.step[y, sx], 1 if t < 0.6 else 0)

        return sprays


def _field_marks(cv, clumps, p, rng, cx, gy, w, h, L, sw):
    """Form field + marks (Ferrari's way): a continuous step field from the clumps (lit caps,
    dark bands where one clump tucks under the next), quantized; then upright marks: dark slots
    in the light, pale flecks in the shadow; then finger tips along the crest."""
    H, W = cv.h, cv.w
    Y, X = np.mgrid[0:H, 0:W].astype(float)
    f = np.full((H, W), -9.0)
    bu = (X - cx) / (w / 2)
    bv = (gy - Y) / h
    nb = np.stack([bu, bv * 1.3 - 0.15, 0.55 + 0 * bu], -1)
    nb /= np.linalg.norm(nb, axis=-1, keepdims=True)
    order = sorted(range(len(clumps)), key=lambda i: clumps[i]["y"] - clumps[i]["ry"])
    for i in order:                              # high clumps first; lower ones overlap them
        c = clumps[i]
        ang = np.arctan2(Y - c["y"], X - c["x"])
        lump = 1 + 0.08 * np.sin(5 * ang + rng.uniform(0, 6.28)) + 0.06 * np.sin(9 * ang + rng.uniform(0, 6.28))
        dx, dy = (X - c["x"]) / c["rx"], (Y - c["y"]) / c["ry"]
        d2 = (dx * dx + dy * dy) / lump ** 2
        inside = d2 <= 1
        nz = np.sqrt(np.clip(1 - d2, 0, 1))
        nl = np.stack([dx / lump, -dy / lump, nz + 0.15], -1)
        nl /= np.linalg.norm(nl, axis=-1, keepdims=True)
        nn = nl * (1 - p["big_form"]) + nb * p["big_form"]
        nn /= np.linalg.norm(nn, axis=-1, keepdims=True)
        lit = np.clip(nn @ L, 0, 1)
        s = np.where(lit < p["term"], 0.9 + 1.6 * lit / p["term"] + p["ambient"],
                     3.1 + 2.4 * (lit - p["term"]) / (1 - p["term"]))
        s = s - np.clip((dy - 0.25) / 0.75, 0, 1) * p["fall"]          # dark band under the clump
        s = s - np.clip((0.3 - bv) / 0.3, 0, 1) * 1.0                   # the dark foot of the bush
        f[inside] = s[inside]
    m = f > -9
    # crevices: where one clump meets another, the light can't reach (1-px dark seam)
    # base values: the mass is calm (three bases: shadow 1, reflected 2, lit halftone 3);
    # the light is carried by marks whose density follows the field (density carries the turn)
    q = np.where(f < 1.7, 1, np.where(f < 3.0, 2, 3)).astype(np.int16)
    q = np.where(f < 0.6, 0, q)
    # soften the reflected band into the shadow with a stochastic seam
    seam = (f > 1.3) & (f < 2.1)
    q[seam & (rng.random(f.shape) < 0.5)] = 1
    ys, xs = np.nonzero(m)
    order = rng.permutation(len(ys))
    dens_lit = p["caps"]
    for i in order:
        y, x = ys[i], xs[i]
        v = f[y, x]
        r = rng.random()
        if v >= 3.0:
            t = min(1.0, (v - 3.0) / 2.4)                 # 0 at the terminator .. 1 full light
            if r < dens_lit * (0.35 + 1.6 * t):
                # an upright lit mark: 1 px wide, 2-3 tall, top pixel brightest, dark tuck below
                ln = 2 + (rng.random() < 0.45)
                for k in range(ln):
                    yy = y - k
                    if 0 <= yy < H and m[yy, x]:
                        q[yy, x] = 5 if (k == ln - 1 and t > 0.45) else 4
                if y + 1 < H and m[y + 1, x] and q[y + 1, x] <= 3:
                    q[y + 1, x] = 2
            elif r < dens_lit * (0.35 + 1.6 * t) + p["slots"] * (1.5 - t):
                ln = 2 + (rng.random() < 0.5)
                for k in range(ln):
                    if 0 <= y + k < H and m[y + k, x] and q[y + k, x] <= 3:
                        q[y + k, x] = 1 if k == 0 else 2
        elif v >= 1.4:
            if r < p["flecks"] * 2.0:
                for k in range(1 + (rng.random() < 0.5)):
                    if 0 <= y - k < H and m[y - k, x]:
                        q[y - k, x] = 3 if (v > 2.4 and k == 0) else 2
        else:
            if r < p["flecks"] * 0.6:
                q[y, x] = 2
    cv.put(m, SAGE, q)
    # finger tips along the crest: short upright columns with gaps
    top = m & ~np.vstack([np.zeros((1, W), bool), m[:-1]])
    ys, xs = np.nonzero(top)
    for y, x in zip(ys, xs):
        if rng.random() < p["fingers"]:
            ln = 1 + (rng.random() < 0.5) + (rng.random() < 0.2 and sw > 1)
            s0 = q[y, x]
            for k in range(1, ln + 1):
                cv.px(x, y - k, SAGE, int(min(NS - 1, s0 + (1 if k == ln and s0 >= 3 else 0))))
            # gap beside the finger: the pixel on its shadow side stays open


def _light_field(clumps, p, rng, cx, gy, w, h, L, H, W):
    """Continuous step field over the bush from the clumps (same as the field mode)."""
    Y, X = np.mgrid[0:H, 0:W].astype(float)
    f = np.full((H, W), -9.0)
    bu = (X - cx) / (w / 2)
    bv = (gy - Y) / h
    nb = np.stack([bu, bv * 1.3 - 0.15, 0.55 + 0 * bu], -1)
    nb /= np.linalg.norm(nb, axis=-1, keepdims=True)
    order = sorted(range(len(clumps)), key=lambda i: clumps[i]["y"] - clumps[i]["ry"])
    for i in order:
        c = clumps[i]
        ang = np.arctan2(Y - c["y"], X - c["x"])
        lump = 1 + 0.08 * np.sin(5 * ang + rng.uniform(0, 6.28)) + 0.06 * np.sin(9 * ang + rng.uniform(0, 6.28))
        dx, dy = (X - c["x"]) / c["rx"], (Y - c["y"]) / c["ry"]
        d2 = (dx * dx + dy * dy) / lump ** 2
        inside = d2 <= 1
        nz = np.sqrt(np.clip(1 - d2, 0, 1))
        nl = np.stack([dx / lump, -dy / lump, nz + 0.15], -1)
        nl /= np.linalg.norm(nl, axis=-1, keepdims=True)
        nn = nl * (1 - p["big_form"]) + nb * p["big_form"]
        nn /= np.linalg.norm(nn, axis=-1, keepdims=True)
        lit = np.clip(nn @ L, 0, 1)
        s = np.where(lit < p["term"], 0.9 + 1.6 * lit / p["term"] + p["ambient"],
                     3.1 + 2.4 * (lit - p["term"]) / (1 - p["term"]))
        s = s - np.clip((dy - 0.25) / 0.75, 0, 1) * p["fall"] * 0.6
        s = s - np.clip((0.3 - bv) / 0.3, 0, 1) * 1.2
        f[inside] = s[inside]
    return f


def _sprig_marks(cv, clumps, p, rng, cx, gy, w, h, L, sw):
    """Grouped sprigs (the critic's round-3 note, and the photos at 64 px): the bush is a heap of
    small upright wedges, 3-6 px wide, each lit on its rounded top and darkening downward,
    streaked inside; between them the dark interior shows (you see into the bush); sprig tops
    break the silhouette as the fringe. Values keep the two families: a sprig in the shadow
    family never rises above step 2."""
    H, W = cv.h, cv.w
    f = _light_field(clumps, p, rng, cx, gy, w, h, L, H, W)
    m = f > -9
    # the interior (what shows in the gaps): two dark values, darker low
    q = np.where(f < 2.6, 0, 1).astype(np.int16)
    Y = np.mgrid[0:H, 0:W][0]
    q = np.where((gy - Y) < h * 0.22, 0, q)
    q = np.where(m & (f > 4.2) & (rng.random((H, W)) < 0.3), 2, q)
    # a ragged foot: the interior's lower edge breaks into streaks and stems, not scallops
    for x in range(W):
        col = np.nonzero(m[:, x])[0]
        if len(col) == 0:
            continue
        yb = col.max()
        cut = int(rng.integers(0, 3))
        for k in range(cut):
            if yb - k >= 0:
                m[yb - k, x] = False
        if rng.random() < 0.12 and yb < gy - 1 and abs(x - cx) < w * 0.35:
            # a stem from the foliage's foot to the root crown (the tangle)
            for (sx, sy) in sp.clean_line(x, yb - cut + 1, cx + (x - cx) * 0.3, gy):
                cv.px(sx, sy, SAGESTEM, 0 if rng.random() < 0.6 else 1)
    cv.put(m, SAGE, q)
    spw = p["sprig_px"] or float(np.clip(w / 14.0, 2.0, 5.5))
    sph = spw * (2.2 + 0.8 * np.clip((w - 50) / 60, 0, 1))   # near sage: taller upright sprigs
    ys, xs = np.nonzero(m)
    area = len(ys)
    n = int(area / (spw * sph * 0.62) * p["sprig"] * (1 + 0.5 * np.clip((w - 50) / 60, 0, 1)))
    # weight toward the upper part of each column (sprigs crowd the crown; the base is tangle)
    wts = np.clip((gy - ys) / h, 0.05, 1.0) ** 0.7
    wts /= wts.sum()
    pick = rng.choice(len(ys), size=n, p=wts)
    sprigs = sorted(((ys[i] - sph * 0.3, xs[i]) for i in pick), key=lambda t: t[0])
    sunside = 1 if L[0] > 0 else -1
    for ty, tx in sprigs:
        ty = int(round(ty))
        tx = int(tx)
        ww = spw * rng.uniform(0.7, 1.25)
        hh = sph * rng.uniform(0.7, 1.3)
        yy = min(max(ty + int(hh * 0.3), 0), H - 1)
        v0 = f[yy, min(max(tx, 0), W - 1)]
        if v0 < -8:
            continue
        shadow = v0 < 3.0
        lean = (tx - cx) / (w / 2) * 0.25
        for j in range(int(hh)):
            t = j / max(1.0, hh - 1)
            half = ww / 2 * (0.55 + 0.45 * np.sin(np.pi * min(1.0, 0.25 + t * 0.9))) * (1 - 0.45 * t)
            if j == 0:
                half *= 0.6                                    # rounded top
            xc = tx + lean * (hh - j) * 0.5
            y = ty + j
            if not (0 <= y < H):
                continue
            for x in range(int(np.floor(xc - half + 0.5)), int(np.floor(xc + half + 0.5)) + 1):
                if not (0 <= x < W):
                    continue
                if not m[y, x] and j > 1:                       # only the top may break out
                    continue
                side = (x - xc) * sunside
                s = v0 + (0.9 if j == 0 else 0.4 if j == 1 else 0) - 1.8 * t
                s += 0.6 if side > 0.4 else (-0.6 if side < -0.4 else 0)
                if (x + (j // 2)) % 2 == 0 and t > 0.3:        # inner streaks
                    s -= 0.7
                s += rng.normal(0, 0.25)
                if shadow:
                    s = min(s - 0.6, 2.6)
                st = int(np.clip(np.floor(s + rng.random() * 0.8), 0, NS - 1))
                cv.mat[y, x] = SAGE
                cv.step[y, x] = st


def _open_crest(cv, rounds=2):
    """Interior pixels left showing at the top of the silhouette read as a dark outline (the
    magazine cut-out). Where the crest is interior-dark and nothing is above it, open it to the
    background, so the silhouette is made of sprig tops and gaps."""
    for _ in range(rounds):
        m = cv.mat == SAGE
        above = np.vstack([np.zeros((1, cv.w), bool), cv.mat[:-1] > 0])
        kill = m & ~above & (cv.step <= 1)
        cv.mat[kill] = 0
