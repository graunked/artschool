"""grass: bunchgrass (bluebunch wheatgrass, Idaho fescue, cheatgrass carpets) from blade to band.

The near-scale mark is Ferrari's paired blade, measured in V11AM's tufts (stage 0): each blade is
a clean-run curve of the darkest step with a 1-px lit line on its sun side (right of steep runs,
above shallow runs); the tip is dark only; low on the tuft the lit line drops to the mid step;
the base is a dense dark core with mid flecks. Only three steps.
"""
import numpy as np
import sp


def world_hash(x, y, seed=7):
    h = (np.asarray(x, np.int64) * 374761393 + np.asarray(y, np.int64) * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFF) / 65535.0


def tuft_blades(cx, base_y, n, rng, len_mu=20, len_sd=7, spread=0.75, bend=0.9, width=4.0,
                upright=0.0, wander=0.8):
    """A fountain: n blades from a base `width` px wide at (cx, base_y).
    Returns list of pixel lists (clean runs), base first."""
    blades = []
    for i in range(n):
        u = rng.uniform(-1, 1)
        u = np.sign(u) * abs(u) ** 0.8
        ang = u * spread * (1 - upright) + rng.normal(0, 0.10)
        L = max(4, rng.normal(len_mu, len_sd) * (1 - 0.35 * abs(u)))
        sgn = np.sign(ang) if rng.random() < wander else -np.sign(ang)
        b = sgn * bend * (0.3 + abs(ang) / max(spread, 1e-3)) * rng.uniform(0.4, 1.2)
        bx = cx + u * width * 0.5 + rng.normal(0, 0.6)
        pts = tip_arc(bx, base_y, ang, L, b, nseg=max(3, int(L / 5)))
        blades.append(sp.polyline_clean(pts))
    # draw long upright ones last? Ferrari's front blades cross the back ones: shuffle order
    rng.shuffle(blades)
    return blades


def tip_arc(x0, y0, ang0, length, bend, nseg=4, power=2.2):
    """Like sp.arc_points, but the bend gathers toward the tip (a grass blade is stiff at the
    base and droops at the end)."""
    pts = [(x0, y0)]
    x, y = x0, y0
    seg = length / nseg
    for i in range(nseg):
        t = (i + 0.5) / nseg
        a = ang0 + bend * t ** power
        x += np.sin(a) * seg
        y -= np.cos(a) * seg
        pts.append((x, y))
    return pts


def blade_roles(pix, lit_side=1, tipdark=0.25, low_mid=0.35, low_dark=0.0):
    """Pixel roles for one blade: [(x,y,role)], role 0 dark, 1 mid, 2 lit."""
    out = []
    n = len(pix)
    for i, (x, y) in enumerate(pix):
        out.append((x, y, 0))
        t = i / max(1, n - 1)       # 0 base -> 1 tip
        if t > 1 - tipdark or t < low_dark:
            continue
        # local direction
        j0, j1 = max(0, i - 2), min(n - 1, i + 2)
        dx = pix[j1][0] - pix[j0][0]
        dy = pix[j1][1] - pix[j0][1]
        role = 2 if t > low_mid else 1
        if abs(dy) >= abs(dx):          # steep: lit on the sun side horizontally
            out.append((x + lit_side, y, role))
        else:                            # shallow: lit on top
            out.append((x, y - 1, role))
    return out


def draw_blade_pairs(arr, blades, dark, mid, lit, lit_side=1, tipdark=0.25, rng=None,
                     mask=None, core=True, low_mid=0.35, low_dark=0.0):
    """Draw paired blades into an index array `arr` (values dark/mid/lit). Later blades cover
    earlier ones, but a lit pixel never covers another blade's dark pixel (the dark line wins),
    which is what keeps the crossing strokes legible."""
    H, W = arr.shape
    isdark = np.zeros_like(arr, bool) if mask is None else mask
    for pix in blades:
        roles = blade_roles(pix, lit_side, tipdark, low_mid, low_dark)
        for x, y, r in roles:
            if 0 <= x < W and 0 <= y < H:
                if r == 0:
                    arr[y, x] = dark
                    isdark[y, x] = True
                else:
                    if isdark[y, x] and rng is not None and rng.random() < 0.7:
                        continue
                    arr[y, x] = lit if r == 2 else mid
                    isdark[y, x] = False
    return arr


# ====================================================================== tussocks on the canvas
class TussockParams(dict):
    DEFAULT = dict(
        w=14, h=16,          # px
        blades=None,         # None: from size
        spread=0.55,         # fan half-angle (rad)
        bend=0.9,            # tip droop
        stalks=0.3,          # seed stalks rising above (bluebunch: many in summer)
        az=35, el=40,
        season="summer",
        kind="bluebunch",    # "bluebunch" | "fescue" (finer, denser, lower) | "cheat" (carpet)
        dark_step=0, mid_step=2, lit_step=4, top_step=4,
        gap_keep=0.2,        # (unused by the near painter now)
        dark_frac=0.22,      # share of blades drawn as dark strands       # share of blade-partner pixels kept as dark gaps
    )

    def __init__(self, **kw):
        super().__init__(self.DEFAULT)
        self.update(kw)


def paint_tussock(p, rng, pad=4):
    """One bunchgrass tussock. Near (h >= 10 px): paired blades from a dense base, the Ferrari
    stroke; mid (4-10 px): a fan of 1-px strands, lit on the sun side, dark core; far (< 4 px):
    a dab (dark foot, lit top pixel). Returns (canvas, base_x, base_y)."""
    w, h = p["w"], p["h"]
    # blades fan and seed stalks rise past the nominal size: give the sprite room (a clipped
    # sprite shows as a ruled box edge in the scene)
    W, H = int(w * 2.2 + 2 * pad + 8), int(h * 1.6 + 2 * pad + 4)
    cv = sp.Canvas(W, H)
    cx, gy = W // 2, H - pad
    sun_right = np.cos(np.radians(p["az"])) < 0
    ls = 1 if sun_right else -1
    fine = p["kind"] == "fescue"
    if h >= 6:
        n = p["blades"] or int(w * h / (9 if not fine else 6)) + 6
        blades = tuft_blades(cx, gy, n, rng, len_mu=h * 0.85, len_sd=h * 0.22,
                             spread=p["spread"], bend=p["bend"], width=w * 0.35, wander=0.7)
        # V16PM's grass, read into a tussock: a lit straw mass whose texture is carried by a
        # FEW dark strands; the side away from the sun a step down; a ragged dark foot.
        rx, ry = w * 0.42, h * 0.6
        for y in range(int(gy - ry), gy + 1):
            for x in range(int(cx - rx), int(cx + rx) + 1):
                u, v = (x - cx) / rx, (gy - y) / ry
                if u * u + v * v <= 1.0 and rng.random() < 0.6:
                    cv.px(x, y, sp.GRASS, p["lit_step"] - 1 if u * ls > -0.3 else p["mid_step"])
        order = list(range(len(blades)))
        dark = set(i for i in order if rng.random() < p["dark_frac"])
        # dark strands first, lit blades cross over them
        for i in sorted(order, key=lambda i: (i in dark)):
            pix = blades[i]
            n_ = len(pix)
            if n_ == 0:
                continue
            u = (pix[-1][0] - cx) / max(1.0, rx) * ls          # >0: sun side
            for j, (x, y) in enumerate(pix):
                t = j / max(1, n_ - 1)
                if i in dark:
                    if t > 0.6 or t < 0.05:
                        continue
                    s = p["dark_step"] + 1 if t > 0.15 else p["dark_step"]
                else:
                    if t > 0.45:
                        s = p["lit_step"] if u > -0.25 else p["lit_step"] - 1
                    elif t > 0.15:
                        s = p["lit_step"] - 1 if u > -0.5 else p["mid_step"]
                    else:
                        s = p["mid_step"]
                cv.px(x, y, sp.GRASS, s)
        # the ragged foot: dark flecks along the ground line only
        for x in range(int(cx - w * 0.3), int(cx + w * 0.3) + 1):
            if rng.random() < 0.6:
                cv.px(x, gy, sp.GRASS, p["dark_step"] + (rng.random() < 0.4))
        # seed stalks: thin straight lines rising above with a 1-2 px lit head
        ns = int(p["stalks"] * n * 0.35)
        for _ in range(ns):
            x0 = cx + rng.normal(0, w * 0.15)
            ang = rng.normal(0, 0.18)
            ln = h * rng.uniform(1.0, 1.35)
            pix = sp.polyline_clean([(x0, gy - 2), (x0 + np.sin(ang) * ln, gy - 2 - np.cos(ang) * ln)])
            for i, (x, y) in enumerate(pix):
                if i > len(pix) * 0.3:
                    cv.px(x, y, sp.GRASS, p["lit_step"] - 1 if i < len(pix) - 3 else p["top_step"])
    elif h >= 4:
        # a fountain dome: a straw body, strands fanning from the foot to the rim in pairs
        # (lit strand, dark gap), tips breaking the rim; darker foot
        rx, ry = w / 2.0, float(h)
        for y in range(int(gy - ry), gy + 1):
            for x in range(int(cx - rx), int(cx + rx) + 1):
                u, v = (x - cx) / rx, (gy - y) / ry
                if u * u + v * v <= 1.0:
                    s = p["lit_step"] - 1 if v > 0.3 else p["mid_step"]
                    cv.px(x, y, sp.GRASS, s)
        n = int(w * 0.9) + 3
        for i in range(n):
            u = -1 + 2 * (i + rng.uniform(0.2, 0.8)) / n
            ang = u * 1.15
            ln = ry * rng.uniform(0.85, 1.2)
            ex, ey = cx + np.sin(ang) * rx * 1.05, gy - np.cos(ang) * ln
            pix = sp.clean_line(cx + u * rx * 0.3, gy, ex, ey)
            lit = (u * ls) > -0.35
            for j, (x, y) in enumerate(pix):
                t = j / max(1, len(pix) - 1)
                if t < 0.25:
                    continue
                s = (p["lit_step"] if lit else p["lit_step"] - 1) if t > 0.45 else p["mid_step"] + 1
                cv.px(x, y, sp.GRASS, s)
                # the dark partner on the shadow side
                if 0.3 < t < 0.9 and rng.random() < 0.6:
                    xx = x - ls
                    if 0 <= xx < W and cv.mat[y, xx] == sp.GRASS and cv.step[y, xx] < p["lit_step"] - 1:
                        cv.step[y, xx] = p["dark_step"] + (1 if t > 0.6 else 0)
    else:
        # dab: a lit straw pixel (or two) over a darker foot; never vanishes
        cv.px(cx, gy, sp.GRASS, p["mid_step"])
        cv.px(cx, gy - 1, sp.GRASS, p["lit_step"])
        if w >= 2.5:
            cv.px(cx + ls, gy, sp.GRASS, p["lit_step"] - 1)
        if h >= 3:
            cv.px(cx, gy - 2, sp.GRASS, p["lit_step"] - 1)
    return cv, cx, gy
