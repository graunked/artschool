"""More riparian and alpine shrubs on the willow pipeline.

Each species is a change to the form (the wand model), the mark, and the palette; the clump
grammar, the value plan, the fringe, level of detail, contact and depth bands are all shared.

- winter willow : leafless. Only the wands and their twigs, 1-px clean runs, coloured red, orange,
                  yellow or grey; far off the twig mass becomes a mauve-grey haze.
- sitka alder   : Alnus viridis ssp. sinuata. A sprawling shrub of avalanche tracks and cold banks;
                  stems curve out low and sweep up (J-shaped), broad serrate leaves: the broadleaf
                  lozenge glyph from the C1 copy instead of the willow spray.
- dwarf willow  : Salix arctica / nivalis / cascadensis. A mat hugging rock and ground, 5-15 cm
                  tall, lobed edge, tiny round leaves as dots, pale catkins upright in summer.
- red alder     : Alnus rubra. A small tree in bands along streams: 1-3 pale grey trunks mottled
                  with white lichen, leaning, a narrow rounded crown of broadleaf clumps.
"""
import numpy as np
from scipy import ndimage as ndi
from wpaint import *
from bush import WillowBush, wand_dirs
from willow import paint_willow, merge, LEAFS, BAND, FOLIAGE, lod_params, ground_contact, paint_micro
from foliage import poisson, stroke_run, snap_dir, lozenge, value_noise

BARK = 13          # alder trunk ramp (6 steps)
BARK2 = 14         # cottonwood: dark grey furrowed bark


# ------------------------------------------------------------------ winter willow
def paint_winter(cv, cx, gy, W_, H_, L, rng, p=None):
    """Leafless willow: every wand drawn full length as a clean run, with short twigs off it that
    angle upward; stems leaning toward the sun take the lit step, the rest the mid step; a darker
    tangle at the base; the upper twig ends thin out. Returns the painted mask."""
    p = p or {}
    h, w = cv.h, cv.w
    b = WillowBush(cx, gy, W_, H_, rng, **merge(dict(nw=int(30 + W_ * 0.45), crown_w=0.5, curtain=0, spread=38, droop=0.04,
                                                    diverge=0.1, top_flat=0.2), p.get("form", {})))
    lx = L[0]
    M = np.zeros((h, w), bool)
    ylim = gy if p.get("clip", True) else h
    def put(x, y, s):
        if 0 <= x < w and 0 <= y < ylim:
            if cv.mat[y, x] == STEM and cv.step[y, x] >= s:
                return
            cv.mat[y, x] = STEM; cv.step[y, x] = s; M[y, x] = True
    step_every = max(2, int(3 * 90 / max(W_, 30)))
    for wd in b.wands:
        pts = wd["pts"]; n = len(pts) - 1
        idx = list(range(0, n + 1, 3)) + ([n] if n % 3 else [])
        poly = [(int(round(pts[k][0])), int(round(pts[k][1]))) for k in idx]
        lean = pts[n][0] - pts[0][0]
        zf = np.mean(pts[:, 2]) / max(W_ * 0.3, 1)     # -1 back .. +1 front
        # value by depth and light: back stems dark (seen through the front ones), front stems
        # facing the sun lit, the rest mid. A leafless willow reads by this layering alone.
        if zf < -0.25:
            base_s = 1
        elif lean * (-lx) < 0 and zf > -0.05:
            base_s = 3
        else:
            base_s = 2
        pix = polyline_clean(poly)
        for i, (x, y) in enumerate(pix):
            t = i / max(len(pix) - 1, 1)
            s = 1 if t < 0.05 else base_s
            if t > 0.9 and rng.random() < 0.4:
                continue                                 # the tips thin out
            put(x, y, s)
        # twigs: short runs off the wand in its upper 60%, angled up and out
        for k in range(int(n * 0.4), n, max(1, step_every // 2)):
            if rng.random() > p.get("twig_p", 0.35):
                continue
            x0, y0 = pts[k][0], pts[k][1]
            tx, ty = pts[min(n, k + 1)][0] - pts[k][0], pts[min(n, k + 1)][1] - pts[k][1]
            a = np.arctan2(ty, tx) + (1 if rng.random() < 0.5 else -1) * np.radians(rng.uniform(12, 28))
            ang = snap_dir(np.cos(a), np.sin(a) - 0.6)
            Lt = int(rng.integers(2, 3 + int(W_ / 30)))
            for j, (x, y) in enumerate(stroke_run(int(round(x0)), int(round(y0)), ang, Lt)):
                put(x, y, base_s if j < Lt - 1 else max(2, base_s - 1))
    return M


def paint_bare_crown(cv, x0, y0, W_, H_, L, rng, depth=4, mat=STEM, haze=1.0):
    """A leafless tree crown: branches fork recursively from the trunk top (clean runs), each
    generation shorter and more spread; the outermost generation is a fine twig haze drawn as
    single pixels at the crown's edge (a two-step dither between twig and sky, Ferrari's way of
    saying 'too fine to draw'). The crown's outline comes out an oval because branch lengths are
    set by an ellipse around the crown centre."""
    h, w = cv.h, cv.w
    lx = -np.sign(L[0]) if abs(L[0]) > 0.05 else 1
    ccx, ccy = x0, y0 - H_ * 0.5
    ra, rb = W_ / 2, H_ / 2
    def inside(x, y):
        return ((x - ccx) / ra) ** 2 + ((y - ccy) / rb) ** 2 < 1
    def put(x, y, s):
        if 0 <= x < w and 0 <= y < h:
            if cv.mat[y, x] == mat and cv.step[y, x] >= s:
                return
            cv.mat[y, x] = mat; cv.step[y, x] = s
    def branch(x, y, ang, length, gen):
        ex, ey = x + np.cos(ang) * length, y + np.sin(ang) * length
        # stop at the oval
        k = 1.0
        while not inside(ex, ey) and k > 0.3:
            k -= 0.1; ex, ey = x + np.cos(ang) * length * k, y + np.sin(ang) * length * k
        s = 1 if gen == 0 else (3 if np.cos(ang) * lx < -0.2 and gen >= 2 else 2)
        for (px, py) in clean_line(x, y, ex, ey):
            put(px, py, s)
        if gen >= depth:
            # twig haze: a spray of single pixels beyond the tip, inside the oval
            for _ in range(int((6 + 2 * length) * haze)):
                a = ang + rng.normal(0, 0.8); d = rng.uniform(1, length * 0.9 + 2)
                tx, ty = int(round(ex + np.cos(a) * d)), int(round(ey + np.sin(a) * d))
                if inside(tx, ty) and rng.random() < 0.7:
                    put(tx, ty, 2 if rng.random() < 0.6 else 3)
            return
        nb = int(rng.integers(2, 4))
        for i in range(nb):
            a = ang + rng.uniform(-0.75, 0.75) - 0.15 * np.sin(ang)   # a slight upward bias
            branch(ex, ey, a, length * rng.uniform(0.55, 0.75), gen + 1)
    for i in range(int(rng.integers(5, 8))):
        a = -np.pi / 2 + rng.uniform(-1.1, 1.1)
        branch(x0, y0 - H_ * 0.12, a, H_ * rng.uniform(0.28, 0.4), 1)


# ------------------------------------------------------------------ broadleaf glyph (alders)
def broadleaf_marks(mask, light, rng, top, dens=(0.05, 1.0), dens_lo=0.25, lens=(2.5, 4.0), wmax=(0.9, 1.5),
                    spacing=2.2, base=(0.1, 0.8), angles=(26.6, 45, 63), S=None):
    """The C1 lozenge leaf: lit upper half, one step darker lower half, a dark tuck under it,
    painted dim-to-bright; density and brightness from the clump field."""
    h, w = mask.shape
    if S is None:
        S = -np.ones((h, w), int)
    n = value_noise((h, w), 1.5, rng)
    t = 0.55 * rng.random((h, w)) + 0.45 * (n - n.min()) / (np.ptp(n) + 1e-9)
    S[mask] = 0
    S[(t < base[0] + (base[1] - base[0]) * light) & mask] = 1
    G = []
    for (x, y) in poisson(h, w, spacing, rng, mask=mask):
        l = light[int(y), int(x)]
        lp = np.clip((l - dens_lo) / (1 - dens_lo), 0, 1)
        if rng.random() > dens[0] + (dens[1] - dens[0]) * lp:
            continue
        tp = int(np.clip(round(1.2 + lp * (top - 0.6) + rng.normal(0, 0.45)), 1, top))
        a = np.radians(rng.choice(angles)) * (1 if rng.random() < 0.5 else -1)
        if rng.random() < 0.5:
            a = np.pi - a
        G.append((tp + rng.random(), x, y, rng.uniform(*lens), a, rng.uniform(*wmax), tp))
    G.sort()
    for (_, x, y, L_, a, wm, tp) in G:
        pts = lozenge(x, y, L_, a, wm)
        low = {}
        for (px, py), role in pts.items():
            if 0 <= px < w and 0 <= py < h and mask[py, px]:
                S[py, px] = tp if role == "T" else max(0, tp - 1)
                low[px] = max(low.get(px, -1), py)
        for px, py in low.items():
            if rng.random() < 0.5 and 0 <= py + 1 < h and mask[py + 1, px] and (px, py + 1) not in pts:
                S[py + 1, px] = min(S[py + 1, px], max(0, tp - 2))
    return S


def paint_sitka_alder(cv, cx, gy, W_, H_, L, rng, p=None):
    """Sitka alder: the willow form made sprawling (stems sweep out low then up), broad leaves."""
    q = merge(lod_params(W_), dict(
        form=dict(spread=85, droop=0.15, diverge=0.55, crown_w=0.45, lobe_r=(0.08, 0.14), lobe_gap=0.8, top_flat=0.4,
                  nw=int(16 + W_ * 0.15)),
        clump=dict(r=(max(2.0, W_ * 0.05), max(3.5, W_ * 0.09)), aspect=1.2),
        sprays=None, broadleaf=True, stems=dict(p=0.5, see=1.2), front_sprays=None))
    return paint_willow(cv, cx, gy, W_, H_, L, rng, merge(q, p or {}))


def paint_dwarf_willow(cv, cx, gy, W_, H_, L, rng, p=None):
    """Dwarf willow mat: very low, spreading, lobed; tiny round leaves as dots and pairs; pale
    upright catkins scattered on top in summer (catkin_p)."""
    q = merge(lod_params(max(W_ * 0.6, 8)), dict(
        form=dict(spread=89, droop=0.05, diverge=0.8, crown_w=0.8, nw=int(12 + W_ * 0.2), lobe_r=(0.05, 0.09),
                  lobe_gap=0.6, top_flat=1.0, skirt_n=4, curtain=0),
        clump=dict(r=(1.5, 3.0), aspect=1.6), sprays=None,
        lit=dict(dir_p=0.0, glyphs=(("dot", 5), ("h2", 3), ("v2", 1), ("cap", 1)), dens=(0.1, 0.9)),
        stems=dict(p=0.0, see=0.0), front_sprays=None, tips=None, wands_visible=None,
        fringe=dict(hole_p=0.1, bump_p=0.3, bump_w=(1, 3), peak_p=0.05, rim_p=0.9)))
    r = paint_willow(cv, cx, gy, W_, H_, L, rng, merge(q, p or {}))
    cp = (p or {}).get("catkin_p", 0.0)
    if cp:
        M = r["M"]
        tops = M & ~np.roll(M, 1, 0)
        for y, x in zip(*np.nonzero(tops)):
            if rng.random() < cp and y - 2 >= 0:
                cv.mat[y - 1, x] = SILVER; cv.step[y - 1, x] = 4
                if rng.random() < 0.5:
                    cv.mat[y - 2, x] = SILVER; cv.step[y - 2, x] = 3
    return r


def paint_red_alder(cv, cx, gy, W_, H_, L, rng, p=None):
    """Red alder, one or two stems from one foot, each leaning apart a little."""
    p = p or {}
    n = p.get("stems", int(rng.choice([1, 1, 1, 2])))
    if n == 1:
        return paint_alder_tree(cv, cx, gy, W_, H_, L, rng, p)
    for i, s_ in enumerate((-1, 1)):
        paint_alder_tree(cv, cx + s_ * W_ * 0.06, gy, W_ * 0.75, H_ * (1.0 if i else 0.88), L, rng,
                         dict(p, lean=s_ * rng.uniform(0.06, 0.18)))
    return None


def paint_red_alder_old(cv, cx, gy, W_, H_, L, rng, p=None):
    """Red alder: 1-3 leaning pale trunks with white lichen patches, a narrow rounded crown of
    broadleaf clumps held on the upper 55% of the height."""
    p = p or {}
    h, w = cv.h, cv.w
    nt = int(rng.choice([1, 1, 2, 2, 3]))
    crown_bot = gy - H_ * 0.32
    trunks = []
    for i in range(nt):
        bx = cx + (i - (nt - 1) / 2) * W_ * 0.1 + rng.uniform(-0.04, 0.04) * W_
        lean = -(bx - cx) / max(gy - crown_bot, 1) * 0.6 + rng.uniform(-0.05, 0.05)   # converge into the crown
        tw = max(1, int(round(W_ * rng.uniform(0.05, 0.08))))
        trunks.append((bx, lean, tw))
    # crown: the willow form, upright, sitting on top of the trunks
    Hc = H_ * 0.66
    q = merge(lod_params(W_), dict(
        form=dict(spread=70, droop=0.5, diverge=0.3, crown_w=0.1, clip_ground=False, base_rise=0.5, lobe_r=(0.1, 0.17), lobe_gap=0.75, leafy_from=0.3,
                  nw=int(14 + W_ * 0.15), curtain=0, skirt_n=0, egg_c=0.5),
        no_contact=True, wands_visible=dict(p=0.7, t=(0.05, 0.7), brk=0.05),
        clump=dict(r=(max(2.0, W_ * 0.06), max(3.5, W_ * 0.11)), aspect=1.15),
        sprays=None, broadleaf=True, stems=dict(p=0.0, see=0.0), front_sprays=None, tips=dict(p=0.08, lens=(1, 2))))
    q = merge(q, p.get("crown", {}))
    tmp = Canvas(w, h)
    if p.get("bare"):
        # winter: the crown is its twigs (mauve-grey haze), the trunks stay white
        paint_bare_crown(tmp, cx, int(crown_bot + Hc * 0.05), W_ * 0.95, Hc, L, rng, depth=5)
    elif False:
        paint_winter(tmp, cx, int(crown_bot - Hc * 0.1), W_, Hc * 0.95, L, rng,
                     dict(form=dict(spread=80, droop=0.5, crown_w=0.1, diverge=0.4, top_flat=0.3, base_rise=0.55,
                                    nw=int(22 + W_ * 0.4)), twig_p=0.8, clip=False))
    else:
        paint_willow(tmp, cx, int(crown_bot), W_, Hc, L, rng, q)
    crown = tmp.mat != 0
    # trunks first (behind the crown's lower clumps), then crown
    lx = -np.sign(L[0]) if abs(L[0]) > 0.05 else 1
    for (bx, lean, tw) in trunks:
        top_y = int(crown_bot - Hc * 0.2)
        for y in range(top_y, gy):
            t = (gy - y) / max(gy - top_y, 1)
            xc = bx + lean * (gy - y)
            half = max(0.5, tw * (1 - 0.35 * t) / 2)
            for x in range(int(np.floor(xc - half)), int(np.ceil(xc + half)) + 1):
                if not (0 <= x < w):
                    continue
                u = (x - xc) / max(half, 0.5)
                if abs(u) > 1.05:
                    continue
                s = 4 if u * lx < -0.2 else (3 if abs(u) <= 0.3 else 1)   # lit side, body, shadow side
                cv.mat[y, x] = BARK; cv.step[y, x] = s
        # lichen: white patches, and dark lenticel dashes across the trunk
        for y in range(top_y, gy):
            if rng.random() < 0.18:
                xc = bx + lean * (gy - y); xx = int(round(xc + rng.uniform(-tw / 2, tw / 2)))
                if 0 <= xx < w and cv.mat[y, xx] == BARK:
                    cv.step[y, xx] = 5
            if rng.random() < 0.08 and tw >= 3:
                xc = bx + lean * (gy - y)
                for xx in range(int(xc - tw / 2) + 1, int(xc + tw / 2)):
                    if 0 <= xx < w and cv.mat[y, xx] == BARK:
                        cv.step[y, xx] = 0
    cv.mat[crown] = tmp.mat[crown]; cv.step[crown] = tmp.step[crown]
    return crown


SPECIES = {
    "willow": None,
    "winter_willow": paint_winter,
    "sitka_alder": paint_sitka_alder,
    "dwarf_willow": paint_dwarf_willow,
    "red_alder": paint_red_alder,
}


def paint_species(cv, species, cx, gy, W_, H_, L, rng, band=0, p=None, foot=None, group=None):
    """Any species at any band: paint in a local canvas, then place with the band's materials."""
    from willow import paint_bush
    g = 0 if group is None else group
    if species == "willow":
        new = paint_bush(cv, cx, gy, W_, H_, L, rng, band=band, p=p, foot=foot)
        if g:
            fol = new & np.isin(cv.mat % BAND, FOLIAGE)
            cv.mat[fol] += 210 * g
        return new
    h, w = cv.h, cv.w
    pw, ph_ = int(W_ * 1.8 + 16), int(H_ * 1.6 + 12)
    ox, oy = int(round(cx - pw / 2)), int(round(gy - ph_ + 2))
    tl = Canvas(pw, ph_)
    lcx, lgy = cx - ox, int(gy) - oy
    if W_ < 6 and species != "winter_willow":
        paint_micro(tl, lcx, lgy, W_, H_, L, rng)
    else:
        SPECIES[species](tl, lcx, lgy, W_, H_, L, rng, p)
    new = np.zeros((h, w), bool)
    ys0, ys1 = max(0, oy), min(h, oy + ph_); xs0, xs1 = max(0, ox), min(w, ox + pw)
    if ys1 > ys0 and xs1 > xs0:
        sub = tl.mat[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox]
        m = sub != 0
        new[ys0:ys1, xs0:xs1] = m
        tgt_m = cv.mat[ys0:ys1, xs0:xs1]; tgt_s = cv.step[ys0:ys1, xs0:xs1]
        off = np.where(np.isin(sub, FOLIAGE), BAND * band + 210 * g, np.where(np.isin(sub, (16, 17)), 210 * g, 0))
        tgt_m[m] = (sub + off)[m]; tgt_s[m] = tl.step[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox][m]
    if species != "winter_willow":
        ground_contact(cv, new, gy, W_, rng, foot, band)
    return new


# ------------------------------------------------------------------ trees: skeleton, bark, crown
def paint_trunk(cv, segs, L, rng, style="cottonwood", mat=BARK):
    """Bark on a limb path. segs: list of (x0, y0, w0, x1, y1, w1). Across the limb: lit side,
    body, shadow side, with the handover between steps as a 2x2 checker band (V30: pure, 50%,
    pure; never skip a step). Cottonwood: deep vertical furrows (dark runs 3-9 px) with a lit ridge
    beside each on the sun side. Alder: white lichen blocks and dark lenticel dashes."""
    h, w = cv.h, cv.w
    lx = -np.sign(L[0]) if abs(L[0]) > 0.05 else 1
    painted = set()
    for sg in segs:
        x0, y0, w0, x1, y1, w1 = sg[:6]
        cur = set()
        n = int(2 * max(abs(x1 - x0), abs(y1 - y0))) + 2      # oversample: never skip a row
        for i in range(n):
            t = i / max(n - 1, 1)
            xc, yc, hw = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, max(0.5, (w0 + (w1 - w0) * t) / 2)
            y = int(round(yc))
            for x in range(int(np.floor(xc - hw)), int(np.ceil(xc + hw)) + 1):
                if not (0 <= x < w and 0 <= y < h):
                    continue
                u = (x - xc) / max(hw, 0.5)
                if abs(u) > 1.02:
                    continue
                v = -u * lx                        # +1 toward the sun
                # steps: 4 lit | 3 body | 2 turning | 1 shadow; 50% checker at each handover
                c = (x + y) % 2
                if v > 0.45:
                    s = 4
                elif v > 0.25:
                    s = 4 if c else 3
                elif v > -0.2:
                    s = 3
                elif v > -0.4:
                    s = 3 if c else 2
                elif v > -0.65:
                    s = 2 if c else 1
                else:
                    s = 1
                if style == "cottonwood" and hw >= 1.5:
                    # furrows: fixed columns across the trunk (in limb coordinates) that break and resume
                    col = int(round((u + 1) * hw))
                    key = (round(xc / 3), col)
                    if col % 3 == 1 and rng.random() > 0.08:
                        s = 0 if v < 0.3 else 1
                    elif col % 3 == 2 and v > 0.2:
                        s = min(5, s + 1)       # the ridge beside a furrow catches the light
                elif style == "alder":
                    if rng.random() < 0.12:
                        s = 5
                    elif hw >= 1.5 and rng.random() < 0.04:
                        s = 0
                if (x, y) in painted:
                    continue                    # the trunk and earlier limbs keep their own form
                cur.add((x, y))
                cv.mat[y, x] = mat; cv.step[y, x] = s
        painted |= cur


class TreeCrown(WillowBush):
    """A tree for the foliage pipeline: a recursive limb skeleton in 3-D; every limb is a 'wand'
    (so wand_dirs and the visible-wand pass work unchanged), and leaf lobes sit on the outer
    generations, clumped per main limb (so each main limb makes one sub-crown)."""

    def __init__(self, cx, gy, W, H, rng, fork=0.4, limbs=(3, 6), depth=3, up=(15, 45), spread=0.8,
                 lobe_r=(0.07, 0.12), gap=0.8, ragged=0.35, lean=0.0, leader=False, hang_p=0.0, low=0.0, **kw):
        """lean: trunk top offset as a fraction of the trunk height. leader: a central stem carries on
        up through the crown. hang_p: share of limbs that leave near-horizontal and droop (the
        cottonwood's ragged lower crown). low: limbs leave the trunk anywhere in its top `low` share."""
        WillowBush.__init__(self, cx, gy, W, H, rng, nw=1, **kw)
        self.wands, self.lobes, self.segs = [], [], []
        th = H * fork
        top = np.array([cx + lean * th, gy - th, 0.0])
        self.trunk = (cx, gy, top)
        nl = int(rng.integers(limbs[0], limbs[1] + 1))
        self.nclumps = nl + (1 if leader else 0)
        for li in range(nl):
            az = rng.uniform(0, 2 * np.pi)
            u = rng.uniform(0, 1) if low > 0 else 1.0
            f = 1 - low * (1 - u)                       # where on the trunk the limb leaves (1 = top)
            base = np.array([cx + lean * th * f, gy - th * f, 0.0])
            if rng.random() < hang_p and f > 0.75:        # only high limbs hang; low ones would float
                tilt = np.radians(rng.uniform(70, 105))
                L0 = H * (1 - fork) * rng.uniform(0.35, 0.55)
            else:
                tilt = np.radians(rng.uniform(*up))
                L0 = H * (1 - fork) * rng.uniform(0.6, 0.85) * (0.8 + 0.2 * f)
            self._grow(base, tilt, az, L0, W * 0.07, 1, depth, li, rng, spread, lobe_r, gap, ragged)
        if leader:
            self._grow(top, np.radians(rng.uniform(0, 8)), rng.uniform(0, 2 * np.pi), H * (1 - fork) * 0.95,
                       W * 0.06, 1, depth, nl, rng, spread * 0.8, lobe_r, gap, ragged)

    def _grow(self, p0, tilt, az, length, width, gen, depth, li, rng, spread, lobe_r, gap, ragged):
        W, H = self.W, self.H
        d = np.array([np.sin(tilt) * np.cos(az) * W / H * 1.2, -np.cos(tilt), np.sin(tilt) * np.sin(az) * W / H * 1.2])
        p1 = p0 + d * length
        # resample as a wand (15 points, slight upward curve)
        pts = np.array([p0 + (p1 - p0) * t + np.array([0, -0.08 * length * np.sin(np.pi * t), 0]) for t in np.linspace(0, 1, 15)])
        wi = len(self.wands)
        self.wands.append(dict(pts=pts, az=az, tilt=tilt, gen=gen))
        self.segs.append((p0[0], p0[1], width, p1[0], p1[1], width * 0.65, p0[2]))
        if gen >= 2:
            t = 0.35 + rng.uniform(0, 0.15)
            while t <= 1.0:
                k = t * 14; j = min(int(k), 13); f = k - j
                p = pts[j] * (1 - f) + pts[j + 1] * f
                r = W * rng.uniform(*lobe_r) * (0.8 + 0.5 * t)
                self.lobes.append(dict(x=p[0], y=p[1], r=r, z=p[2] + W * 0.35, u=0, v=0, wand=wi, t=t, clump=li))
                t += gap * r / max(length, 1) + 0.05
        if gen >= depth or rng.random() < ragged * (gen - 1) * 0.5:
            if gen >= 2:
                return
        nb = int(rng.integers(2, 4))
        for i in range(nb):
            tp = rng.uniform(0.45, 1.0)
            q = p0 + (p1 - p0) * tp
            self._grow(q, np.clip(tilt + rng.uniform(-spread, spread) * 0.7, 0.0, max(1.35, tilt)), az + rng.uniform(-1.0, 1.0),
                       length * rng.uniform(0.45, 0.65), width * 0.6, gen + 1, depth, li, rng, spread, lobe_r, gap, ragged)


def paint_tree(cv, cx, gy, W_, H_, L, rng, p, crown, bark="cottonwood", bark_mat=None, silver=0.0,
               clump_frac=(0.045, 0.08), trunk_frac=0.09):
    """Any broadleaf tree on the foliage pipeline: a TreeCrown skeleton (crown = its kwargs), bark on
    trunk and limbs, the willow painter for the leaves (broadleaf mark), or a bare crown in winter."""
    h, w = cv.h, cv.w
    bark_mat = BARK2 if bark_mat is None else bark_mat
    tc = TreeCrown(cx, gy, W_, H_, rng, clip_ground=False, curtain=0, egg_c=0.45, **crown)
    top = tc.trunk[2]
    tw = max(2.0, W_ * trunk_frac)
    segs = [(cx, gy, tw * 1.3, top[0], top[1], tw)] + [(sg[0], sg[1], max(1.0, sg[2] * (1.6 if i < tc.nclumps else 1)),
                                                        sg[3], sg[4], max(1.0, sg[5])) for i, sg in enumerate(tc.segs)]
    if p.get("bare"):
        paint_trunk(cv, segs, L, rng, bark, mat=bark_mat)
        for wi, wd in enumerate(tc.wands):
            if wd.get("gen", 2) != 2 or wd["tilt"] > 1.2:
                continue
            p0, p1 = wd["pts"][0], wd["pts"][-1]
            ln = np.hypot(*(p1[:2] - p0[:2]))
            # the fine twigs: a sparse fork from each second-generation limb, twig haze kept thin so
            # the crown stays a lace with sky through it, not a mauve cloud
            paint_bare_crown(cv, p1[0], p1[1] + ln * 0.3, ln * 1.1, ln * 0.9, L, rng, depth=3, haze=0.35)
        return None
    q = merge(lod_params(W_), dict(bush_obj=tc, no_contact=True, broadleaf=dict(lens=(2.0, 3.5)),
                                   fringe=dict(hole_p=0.3, bump_p=0.3, rim_p=0.9),
                                   clump=dict(r=(max(2.0, W_ * clump_frac[0]), max(3.5, W_ * clump_frac[1])), aspect=1.1),
                                   sprays=None, stems=dict(p=0.0, see=0.0), front_sprays=None,
                                   wands_visible=dict(p=0.8, t=(0.0, 0.8), brk=0.08),
                                   airy=dict(below=0.0, depth=14, scale=3.5, amount=0.52, cl=0.5),
                                   tips=dict(p=0.1, lens=(1, 2))))
    tmp = Canvas(w, h)
    r = paint_willow(tmp, cx, gy, W_, H_, L, rng, merge(q, p.get("crown", {})))
    if silver:
        sil = (tmp.mat == LEAF) & (tmp.step >= 3) & (rng.random((h, w)) < silver)
        tmp.mat[sil] = SILVER
    paint_trunk(cv, segs, L, rng, bark, mat=bark_mat)
    m = tmp.mat != 0
    cv.mat[m] = tmp.mat[m]; cv.step[m] = tmp.step[m]
    return r


def paint_cottonwood(cv, cx, gy, W_, H_, L, rng, p=None):
    """Black cottonwood: a tall furrowed grey trunk with a central leader through a tall, open,
    ragged crown; limbs leave over the upper half of the trunk and some leave flat and droop; the
    billows stay apart with sky between. Leaves dark glossy above, silver below (they flutter: many
    silver flecks); autumn yellow; winter an upswept bare crown (p['bare'])."""
    p = p or {}
    crown = dict(fork=p.get("fork", rng.uniform(0.25, 0.34)), limbs=(4, 7), depth=3, up=(15, 55),
                 lobe_r=(0.07, 0.12), gap=1.0, ragged=0.5, clump_mix=0.65,
                 leader=True, hang_p=0.35, low=0.3, lean=rng.uniform(-0.08, 0.08))
    return paint_tree(cv, cx, gy, W_, H_, L, rng, p, crown, bark="cottonwood", bark_mat=BARK2, silver=p.get("silver", 0.22))


def paint_alder_tree(cv, cx, gy, W_, H_, L, rng, p=None):
    """A lone red alder: a pale lichen-blotched trunk that leans (alders lean toward light and water),
    a leader, many short upright limbs from the upper 40% of the trunk, so the crown is a narrow,
    irregular oval with its own lumps and a few sky holes; a second stem sometimes (p['stems'])."""
    p = p or {}
    crown = dict(fork=rng.uniform(0.38, 0.5), limbs=(5, 8), depth=3, up=(8, 40), lobe_r=(0.09, 0.15), gap=0.8,
                 ragged=0.45, clump_mix=0.6, leader=True, hang_p=0.08, low=0.12,
                 lean=p.get("lean", rng.uniform(-0.15, 0.15)))
    return paint_tree(cv, cx, gy, W_, H_, L, rng, p, crown, bark="alder", bark_mat=BARK,
                      clump_frac=(0.06, 0.1), trunk_frac=0.08)


def paint_dogwood(cv, cx, gy, W_, H_, L, rng, p=None):
    """Red-osier dogwood. Leafless: a thicket of straight upright red stems (fewer, stiffer and more
    upright than willow), paired twigs at the nodes. In leaf: opposite oval leaves (the broadleaf
    mark), flat white flower heads in June, white berries in autumn (p['flowers'] / p['berries'])."""
    p = p or {}
    if p.get("bare"):
        return paint_winter(cv, cx, gy, W_, H_, L, rng, dict(form=dict(nw=int(16 + W_ * 0.3), spread=30, droop=0.02,
                                                                          diverge=0.08, crown_w=0.55), twig_p=0.25))
    # in leaf: a dense, closed, rounded dome of small pointed-oval leaves; red twigs show through
    q = merge(lod_params(W_), dict(form=dict(spread=50, droop=0.2, diverge=0.15, crown_w=0.5, lobe_r=(0.08, 0.13),
                                             nw=int(20 + W_ * 0.2), lobe_gap=0.6, egg_c=0.5),
                                   clump=dict(r=(max(2.0, W_ * 0.035), max(3.0, W_ * 0.06)), aspect=1.0),
                                   sprays=None, broadleaf=dict(lens=(1.8, 2.8), wmax=(0.8, 1.2), angles=(45, 63, 26.6)),
                                   stems=dict(p=0.8, see=1.0), airy=None,
                                   wands_visible=dict(p=0.8, t=(0.2, 0.9), brk=0.1),
                                   front_sprays=None, tips=dict(p=0.2, lens=(1, 3))))
    r = paint_willow(cv, cx, gy, W_, H_, L, rng, q)
    dots = p.get("flowers", 0) or p.get("berries", 0)
    if dots:
        M = r["M"]
        cand = np.argwhere(M & (np.roll(M, 1, 0) == False) | (M & (rng.random(M.shape) < 0.01)))
        for (y, x) in cand:
            if rng.random() > dots:
                continue
            # a flat-topped cyme: 3 px wide, 1-2 tall, white; berries a looser 2-px cluster
            shape = [(-1, 0), (0, 0), (1, 0), (0, -1)] if p.get("flowers") else [(0, 0), (1, 1), (-1, 1)]
            for dx, dy in shape:
                yy, xx = y + dy, x + dx
                if 0 <= yy < cv.h and 0 <= xx < cv.w:
                    cv.mat[yy, xx] = SILVER; cv.step[yy, xx] = 4 if dy <= 0 else 3
    return r


def paint_salmonberry(cv, cx, gy, W_, H_, L, rng, p=None):
    """Salmonberry thicket: arching canes 1-4 m that bow over at the top (strong droop), broad
    three-part leaves in large flat clumps, magenta flowers in spring, orange-red berries in early
    summer (p['flowers'] / p['berries']), peeling orange-brown canes in winter (p['bare'])."""
    p = p or {}
    if p.get("bare"):
        return paint_winter(cv, cx, gy, W_, H_, L, rng, dict(form=dict(nw=int(12 + W_ * 0.2), spread=55, droop=0.7,
                                                                          diverge=0.1, crown_w=0.7), twig_p=0.2))
    # in leaf: an open thicket of arching canes; big leaves held flat in tiers, so each clump is a
    # wide shelf, lit on top, with a dark band under it and sky between the tiers
    q = merge(lod_params(W_), dict(form=dict(spread=60, droop=0.75, diverge=0.12, crown_w=0.7, lobe_r=(0.08, 0.13),
                                             nw=int(10 + W_ * 0.1), lobe_gap=1.1, top_flat=0.3),
                                   clump=dict(r=(max(2.0, W_ * 0.05), max(4.0, W_ * 0.09)), aspect=2.2, vb=3.4),
                                   sprays=None, broadleaf=dict(lens=(3.0, 5.0), wmax=(1.1, 1.6), angles=(0, 0, 15)),
                                   airy=dict(below=0.1, depth=10, scale=2.5, amount=0.5, cl=0.45),
                                   stems=dict(p=0.4, see=1.2), front_sprays=None, tips=dict(p=0.08, lens=(1, 3))))
    r = paint_willow(cv, cx, gy, W_, H_, L, rng, q)
    dots = p.get("flowers", 0) or p.get("berries", 0)
    if dots:
        M = r["M"]
        for (y, x) in np.argwhere(M):
            if rng.random() > dots * 0.04:
                continue
            m = 16 if p.get("flowers") else 17      # magenta flower / orange berry ramps
            for dx, dy in ([(0, 0), (1, 0)] if p.get("flowers") else [(0, 0)]):
                if 0 <= x + dx < cv.w:
                    cv.mat[y + dy, x + dx] = m; cv.step[y + dy, x + dx] = 1 if dx == 0 else 0
    return r


SPECIES.update(black_cottonwood=paint_cottonwood, dogwood=paint_dogwood, salmonberry=paint_salmonberry)
