"""The willow painter: one bush (or a clump of them) onto an index canvas.

Built from the stage-0 copy of Ferrari's hedge (clump grammar + dark base with lit marks) and
carried onto the willow's own form (a fan of wands from a root crown), its own marks (short
diagonal strokes along the wands, drooping) and its own structure (see-through stems at the base).

Materials: LEAF (lit family ramp, steps 0..4), LEAFS (shadow family ramp, 0..3), STEM (0..3).
"""
import numpy as np
from scipy import ndimage as ndi
from wpaint import *
from bush import WillowBush, wand_dirs
from foliage import clump_foliage, mid_speckle, stroke_marks, wand_tips, fringe, rim_arcs, stroke_run, snap_dir, spray_marks

LEAFS = 10     # shadow-family leaf ramp


DEFAULT = dict(
    form=dict(nw=30, lobe_r=(0.07, 0.12), lobe_gap=0.7, skirt_n=4, crown_w=0.3),
    fam_split=2.9,
    clump=dict(r=(3.5, 7.0), vo=0.0, vg=2.4, vl=2.0, vb=3.0, aspect=1.3, rough=0.4),
    lit=dict(top=4, base_steps=(0, 1), base=(0.1, 0.9), dens=(0.0, 0.95), hi=(0.45, 0.85), mid_p=0.05,
             clump=0.2, dlen=(2, 3), droop=(15, 55), tuck_p=0.4, dens_lo=0.3, dens_gamma=1.0, dark_below=0.2),
    shadow=dict(top=3, base=(0.15, 0.85), base_steps=(0, 1), dens=(0.0, 0.6), hi=(0.55, 0.95), mid_p=0.0, dens_lo=0.35, dark_below=0.15),
    tips=dict(p=0.12, lens=(1, 3)),
    fringe=dict(hole_p=0.15, bump_p=0.2, rim_p=0.9),
    stems=dict(n=None, visible_below=0.5, see=1.0, zone_t=0.22, p=0.55, brk=0.08),
    front_sprays=dict(r_sep=5.0, p=0.5),
    wands_visible=dict(p=0.5, t=(0.3, 0.85), brk=0.12),
    airy=dict(below=0.35, depth=6, scale=2.0, amount=0.4, cl=0.4),
)


def merge(a, b):
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return out


def paint_willow(cv, cx, gy, W_, H_, L, rng, p=None, zbuf=None):
    """Paint one willow bush into canvas cv (in place). Returns dict of intermediate fields."""
    p = merge(DEFAULT, p)
    h, w = cv.h, cv.w
    b = p["bush_obj"] if p.get("bush_obj") is not None else WillowBush(cx, gy, W_, H_, rng, **p["form"])
    F = b.fields(w, h, L)
    x = np.nan_to_num(F["x"]); m = F["mask"]
    dx, dy = wand_dirs(b, F)
    lsd = (L[0], -L[1])                      # screen direction toward the light (x right, y down)
    g = np.clip(x / 5.4, 0, 1)
    litpx = m & (x >= p["fam_split"])
    # ---- see-through base: thin the mass near the bottom centre, where only stems stand
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    sp = p["stems"]
    low = (yy - (gy - sp["visible_below"] * H_)) / (sp["visible_below"] * H_)
    cen = 1 - np.clip(np.abs(xx - cx) / (W_ * 0.38), 0, 1)
    see = np.clip(low, 0, 1) * cen * sp["see"]
    thin = value_noise((h, w), 2.5, rng)
    thin = (thin - thin.min()) / (np.ptp(thin) + 1e-9)
    zone = m & (see > sp.get("zone_t", 0.18))
    mclump = m & ~(zone & (thin < 0.75))
    # ---- clump layout: clumps elongated along the wands, lit caps toward the sun
    CL = np.full((h, w), -1.0)
    _, MC = clump_foliage(mclump, g, rng, 4, field=CL, dirs=(dx, dy), L2=lsd,
                          keep_inside=ndi.binary_dilation(m, iterations=2) & ~(zone & (thin < 0.6)),
                          **p["clump"])
    fam = np.where(MC, ndi.grey_dilation(litpx.astype(float), size=2) > 0, False)
    # clump family: which family owns the clump pixel = the family of the form underneath
    fam = litpx | (MC & ~m & ndi.binary_dilation(litpx, iterations=2))
    CLv = np.where(CL < 0, g * 0.5, CL)
    lit = MC & fam; sh = MC & ~fam
    ll = np.clip((CLv - 0.3) / 0.7, 0, 1)
    ls = np.clip(CLv / 0.6, 0, 1)
    # ---- stems (drawn first; foliage covers them where dense)
    stems_px = draw_stems(b, rng, L, sp, gy) if sp.get("p", 0.55) > 0 else []
    if p.get("broadleaf"):
        from species import broadleaf_marks
        SL = broadleaf_marks(lit, ll, rng, 4, **(p["broadleaf"] if isinstance(p["broadleaf"], dict) else {}))
    elif p.get("sprays"):
        SL = spray_marks(lit, ll, (dx, dy), rng, 4, lit_side=lsd, **p["sprays"])
    else:
        SL = mid_speckle(None, ll, rng, mask=lit, dirs=(dx, dy), **p["lit"])
    SL = np.where(lit, SL, -1)
    SS = mid_speckle(None, ls, rng, mask=sh, **p["shadow"])
    S = np.where(lit, SL, np.where(sh, SS, -1))
    MAT = np.where(lit, LEAF, np.where(sh, LEAFS, 0))
    sil = lit & (S >= 10)
    MAT = np.where(sil, SILVER, MAT); S = np.where(sil, S - 10, S)
    M = MC.copy()
    # ---- crest: wand tips and small caps/holes
    if p["tips"] is not None:
        lf = np.where(lit, 0.4 + 0.6 * ll, 0.0)
        S2, M2 = wand_tips(np.maximum(S, 0), M & lit, dx, dy, lf, rng, 4, lo=2, stem_step=3, **p["tips"])
        M2 = M2 | M
        S3, M3 = wand_tips(np.maximum(S2, 0), M2 & sh & ~lit, dx, dy, np.zeros((h, w)), rng, 2, lo=1, stem_step=1,
                           **dict(p["tips"], p=min(0.6, p["tips"]["p"] * 1.4)))
        newd = M3 & ~M2
        S2 = np.where(newd, np.minimum(S3, 2), S2); M2 = M2 | M3
        MAT = np.where(newd, LEAFS, MAT)
        new = M2 & ~M
        MAT = np.where(new, np.where(ndi.binary_dilation(lit, iterations=7), LEAF, LEAFS), MAT)
        S = np.where(new, np.minimum(S2, np.where(MAT == LEAFS, 3, 4)), S); M = M2
    if p["fringe"] is not None:
        Sf, Mf = fringe(np.maximum(S, 0), M & (MAT == LEAF), np.where(MAT == LEAF, 0.4 + 0.6 * ll, 0), rng, 4, **p["fringe"])
        new = Mf & ~M
        S = np.where(new, Sf, S); MAT = np.where(new, LEAF, MAT); M = M | Mf
        # holes opened by the fringe
        # the shadow side's silhouette breaks up too: caps and holes, no light
        Ss, Ms = fringe(np.maximum(S, 0), M & (MAT == LEAFS), np.zeros((h, w)), rng, 2,
                        hole_p=0.3, bump_p=0.35, bump_w=(1, 3), peak_p=0.35, rim_p=0.0)
        news = Ms & ~M
        S = np.where(news, 1, S); MAT = np.where(news, LEAFS, MAT); M = M | Ms
        holes_s = (M & (MAT == LEAFS)) & ~Ms & (ndi.distance_transform_edt(m) < 3) & ~zone
        M = M & ~holes_s
        holes = (M & (MAT == LEAF)) & ~Mf
        holes &= ndi.distance_transform_edt(m) < 3
        holes &= ~zone
        M = M & ~holes
    if p["form"].get("clip_ground", True):
        M[gy:] = False
    ah = p.get("airy")
    if ah:
        dist_in = ndi.distance_transform_edt(m)
        up_part = yy < gy - H_ * ah["below"]
        hn = value_noise((h, w), ah["scale"], rng); hn = (hn - hn.min()) / (np.ptp(hn) + 1e-9)
        hole = M & up_part & (dist_in < ah["depth"]) & (hn < ah["amount"]) & (np.maximum(S, 0) <= 1) & (MAT != SILVER) & (CLv < ah.get("cl", 0.4))
        M = M & ~hole
        airy_holes = hole
    else:
        airy_holes = np.zeros((h, w), bool)
    # ---- wands seen between clumps: 1-px clean runs through the canopy, only over the dark base
    wandpx = []
    wv = p.get("wands_visible")
    backlit = L[2] < -0.3
    if backlit and wv:
        wv = dict(wv, p=min(1.0, wv["p"] * 1.6), brk=wv["brk"] * 0.6)
    if wv:
        for wd in b.wands:
            if rng.random() > wv["p"]:
                continue
            pts = wd["pts"]; n = len(pts) - 1
            k0 = int(n * wv["t"][0]); k1 = int(n * wv["t"][1])
            poly = [(int(round(pts[k][0])), int(round(pts[k][1]))) for k in list(range(k0, k1 + 1, 4)) + ([k1] if (k1 - k0) % 4 else [])]
            on = True
            for (px, py) in polyline_clean(poly):
                if rng.random() < wv["brk"]:
                    on = not on
                if not on or not (0 <= px < w and 0 <= py < h) or not M[py, px]:
                    continue
                if S[py, px] <= 1 or backlit:
                    litp = MAT[py, px] == LEAF
                    wandpx.append((px, py, 0 if backlit else (2 if litp else 1)))
    # ---- the hollow under the canopy: dark interior (the far side of the bush in shade)
    hol = zone & ~M
    hn = rng.random((h, w))
    cv.mat[hol] = LEAFS; cv.step[hol] = (hn[hol] < 0.3).astype(int)
    # ---- stems over the hollow, under the foliage
    for (px, py, s) in stems_px:
        if 0 <= px < w and 0 <= py < h and py < gy and not M[py, px]:
            cv.mat[py, px] = STEM; cv.step[py, px] = s
    # ---- a few low sprays hang in front of the stems (the base is never a clean picket fence)
    if p.get("front_sprays") and hol.any():
        Sh = -np.ones((h, w), int); painted = np.zeros((h, w), bool)
        from foliage import poisson, willow_spray
        for (fx, fy) in poisson(h, w, p["front_sprays"]["r_sep"], rng, mask=hol):
            if rng.random() > p["front_sprays"]["p"]:
                continue
            ang = np.radians(rng.uniform(200, 340))       # arching, tips down-and-out
            willow_spray(Sh, hol | ndi.binary_dilation(hol), fx, fy, ang, int(rng.integers(3, 6)), 2, rng,
                         lit_side=lsd, painted=painted, tuck_p=0.0)
        cv.mat[painted] = LEAFS; cv.step[painted] = np.clip(Sh[painted], 1, 3)
    cv.mat[M] = MAT[M]; cv.step[M] = np.maximum(S[M], 0)
    for (px, py, s) in wandpx:
        cv.mat[py, px] = STEM; cv.step[py, px] = s
    # sky never shows deep inside the mass: fill interior gaps with the shadow base
    deep = m & (ndi.distance_transform_edt(m) >= 3) & ((cv.mat == SKY) | (cv.mat == 0)) & ~airy_holes
    cv.mat[deep] = LEAFS; cv.step[deep] = 0
    if p.get("no_contact"):
        return dict(bush=b, F=F, M=M, CL=CLv, lit=lit, sh=sh)
    # contact line where foliage or stems meet the gravel
    row = gy - 1
    cm = (cv.mat[row] == LEAF) | (cv.mat[row] == LEAFS) | (cv.mat[row] == STEM)
    cv.mat[row][cm] = LEAFS; cv.step[row][cm] = 0
    return dict(bush=b, F=F, M=M, CL=CLv, lit=lit, sh=sh)


def draw_stems(b, rng, L, sp, gy):
    """Stems: each wand's lower part as a clean 1-px run from the crown, lit on the sun side
    of the fan, dark on the other. Returns [(x, y, step)]."""
    out = []
    lx = L[0]
    for wd in b.wands:
        if rng.random() > sp.get("p", 0.55):
            continue
        pts = wd["pts"]
        n = len(pts) - 1
        k1 = max(2, int(n * rng.uniform(0.35, 0.6)))
        poly = [(int(round(pts[k][0])), int(round(pts[k][1]))) for k in list(range(0, k1 + 1, 4)) + ([k1] if k1 % 4 else [])]
        pix = polyline_clean(poly)
        # stems leaning toward the sun catch light; back/away ones are dark silhouettes
        lean = pts[k1][0] - pts[0][0]
        front = wd["pts"][k1][2] > 0
        s = 2 if (lean * lx > 0 and front and rng.random() < 0.6) else 1
        on = True
        for i, (px, py) in enumerate(pix):
            if py >= gy:
                continue
            if i > 3 and rng.random() < sp.get("brk", 0.08):
                on = not on
            if on:
                out.append((px, py, s if i > 1 else 0))
    return out


def grey_pal():
    """Greys that keep the value structure of the colour ramps (luminance of Ferrari's hedge ramps)."""
    P = grey_palette()
    lit = [59, 79, 103, 131, 175]
    shd = [23, 44, 59, 70]
    for s, v in enumerate(lit):
        P[(LEAF, s)] = v / 255
    for s, v in enumerate(shd):
        P[(LEAFS, s)] = v / 255
    P.update({(STEM, s): v for s, v in enumerate([0.07, 0.16, 0.30, 0.45])})
    P.update({(SKY, s): v for s, v in enumerate([0.80, 0.84, 0.88, 0.92])})
    P.update({(GRAVEL, s): v for s, v in enumerate([0.10, 0.30, 0.44, 0.55, 0.63, 0.72])})
    return P


def ferrari_pal():
    """Colour check palette: Ferrari's own V09 hedge ramps (lit B, shadow = near darks)."""
    from lw import load
    pal = load("V09")["pal"]
    P = {}
    for s, i in enumerate([42, 41, 50, 40, 49]):
        P[(LEAF, s)] = tuple(int(v) for v in pal[i])
    for s, i in enumerate([3, 43, 42, 86]):
        P[(LEAFS, s)] = tuple(int(v) for v in pal[i])
    P.update({(STEM, s): c for s, c in enumerate([(23, 15, 15), (63, 59, 59), (95, 87, 87), (131, 123, 123)])})
    P.update({(SKY, s): tuple(int(v) for v in pal[252]) for s in range(4)})
    P.update({(GRAVEL, s): (150, 146, 136) for s in range(6)})
    return P


BAND = 30          # material offset per depth band
FOLIAGE = (LEAF, LEAFS, SILVER, STEM)


def lod_params(W_, p=None):
    """Level of detail by the bush's width in pixels: marks never shrink below their pixel size;
    a smaller bush gets fewer, relatively bigger marks, then simpler ones."""
    q = {}
    if W_ >= 110:
        q = dict(clump=dict(r=(5.0, 10.0) if W_ >= 120 else (4.0, 8.0)),
                 sprays=dict(r_sep=3.4 if W_ >= 120 else 2.8, length=(4, 8) if W_ >= 120 else (3, 6), dens_lo=0.2 if W_ >= 120 else 0.08, every=(1, 2), silver_p=0.35),
                 tips=dict(p=0.2, lens=(2, 5)), wands_visible=dict(p=0.4, t=(0.3, 0.9), brk=0.1))
    elif W_ >= 30:
        s = min(W_, 90) / 80
        q = dict(clump=dict(r=(max(1.5, 3.5 * s), max(3.0, 7.0 * s))), sprays=None,
                 lit=dict(dlen=(2, 3) if W_ >= 50 else (1, 2), spacing=1.7),
                 tips=dict(p=0.12, lens=(1, 3)), wands_visible=dict(p=0.3, t=(0.35, 0.8), brk=0.15),
                 stems=dict(p=0.3 if W_ >= 50 else 0.12), front_sprays=None if W_ < 50 else dict(r_sep=5.0, p=0.4))
    else:
        s = W_ / 30
        q = dict(form=dict(nw=16, lobe_r=(0.1, 0.16), lobe_gap=0.8, skirt_n=2, crown_w=0.3),
                 clump=dict(r=(max(1.2, 2.0 * s), max(1.8, 3.2 * s)), rough=0.2), sprays=None,
                 lit=dict(dlen=(1, 1), dir_p=0.0, glyphs=(("dot", 4), ("h2", 2), ("v2", 1))),
                 tips=None, wands_visible=None, stems=dict(p=0.0, see=0.0), front_sprays=None,
                 fringe=dict(hole_p=0.1, bump_p=0.15, bump_w=(1, 2), peak_p=0.1, rim_p=0.9))
    return merge(q, p or {}) if p else q


def paint_micro(cv, cx, gy, W_, H_, L, rng, band=0):
    """A willow of a few pixels: a lumpy blob, lit pixels along its sun-side top, a dark base row."""
    h, w = cv.h, cv.w
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    u = (xx - cx) / (W_ / 2); v = (gy - yy) / H_
    r = u ** 2 + (v - 0.45) ** 2 / 0.3
    m = (r < 1.0 + 0.35 * (rng.random((h, w)) - 0.5)) & (yy < gy) & (v >= 0)
    if m.sum() < 3:
        # a willow of 3 px is still a dab: two dark pixels at the foot, one lit on top
        c, g = int(round(cx)), int(gy)
        for (xx, yy_) in ((c, g - 1), (c + 1, g - 1), (c, g - 2)):
            if 0 <= xx < w and 0 <= yy_ < h:
                m[yy_, xx] = True
    open_up = m & ~np.roll(m, 1, 0)
    lsx = -L[0]
    side = (u * -lsx) < 0.2
    for y, x in zip(*np.nonzero(m)):
        mat, st = LEAFS, 1
        if open_up[y, x] and side[y, x]:
            mat, st = LEAF, 4 if rng.random() < 0.6 else 3
        elif v[y, x] > 0.45 and side[y, x]:
            mat, st = LEAF, 1 if rng.random() < 0.6 else 3
        if y == int(gy) - 1:
            mat, st = LEAFS, 0
        if m.sum() <= 4 and y == np.nonzero(m)[0].min():
            mat, st = LEAF, 4
        cv.mat[y, x] = mat + BAND * band; cv.step[y, x] = st
    return m


def paint_bush(cv, cx, gy, W_, H_, L, rng, band=0, p=None, valley=2, base_arc=0.0, jitter=True, foot=None):
    """Paint one willow of any size into cv at a depth band, darkening the foliage already there
    just above this bush's crest (the valley between overlapping bushes)."""
    h, w = cv.h, cv.w
    # paint in a local canvas around the bush, then place it
    pw, ph_ = int(W_ * 1.6 + 16), int(H_ * 1.5 + 12)
    ox, oy = int(round(cx - pw / 2)), int(round(gy - ph_ + 2))
    tmpl = Canvas(pw, ph_)
    lcx, lgy = cx - ox, int(gy) - oy
    if W_ < 7:
        paint_micro(tmpl, lcx, lgy, W_, H_, L, rng, band=0)
    else:
        q = lod_params(W_, p)
        if jitter:
            # every willow its own: splay, droop, lean, crown width, clump count
            fj = dict(spread=rng.uniform(55, 85), droop=rng.uniform(0.3, 0.6), lean=rng.uniform(-0.08, 0.08),
                      crown_w=rng.uniform(0.2, 0.42), nclumps=int(rng.integers(3, 8)))
            q = merge(q, dict(form=merge(fj, q.get("form", {}))))
        paint_willow(tmpl, lcx, lgy, W_, H_, L, rng, q)
    if base_arc > 0:
        # seen from above, the ground contact is an ellipse: its front edge curves down in the middle,
        # so the foliage's bottom follows an arc, with the contact line along it
        yy, xx = np.mgrid[0:ph_, 0:pw].astype(float)
        arc = lgy - 1 - base_arc * H_ * np.clip(((xx - lcx) / (W_ / 2)) ** 2, 0, 1.5)
        cut = (yy > arc) & (tmpl.mat != 0)
        tmpl.mat[cut] = 0
        for xc in range(pw):
            col = np.nonzero(tmpl.mat[:, xc])[0]
            if len(col):
                tmpl.mat[col.max(), xc] = LEAFS; tmpl.step[col.max(), xc] = 0
    tmp = Canvas(w, h)
    ys0, ys1 = max(0, oy), min(h, oy + ph_); xs0, xs1 = max(0, ox), min(w, ox + pw)
    if ys1 > ys0 and xs1 > xs0:
        tmp.mat[ys0:ys1, xs0:xs1] = tmpl.mat[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox]
        tmp.step[ys0:ys1, xs0:xs1] = tmpl.step[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox]
    new = tmp.mat != 0
    if valley and new.any():
        above = np.zeros_like(new)
        for k in range(1, valley + 1):
            above |= np.roll(new, -k, 0)
        above &= ~new
        fol = np.isin(cv.mat % BAND, FOLIAGE) & (cv.mat > 0)
        vz = above & fol
        cv.mat[vz] = LEAFS + BAND * (cv.mat[vz] // BAND); cv.step[vz] = np.minimum(cv.step[vz], 1)
    cv.mat[new] = tmp.mat[new] + BAND * band; cv.step[new] = tmp.step[new]
    ground_contact(cv, new, gy, W_, rng, foot, band)
    return new


def ground_contact(cv, new, gy, W_, rng, foot=None, band=0):
    """How a bush sits: the darkest line where it meets the ground, a dark undercut pocket in the
    ground just under its middle, and stones or tufts overlapping its foot."""
    h, w = cv.h, cv.w
    cols = np.nonzero(new.any(0))[0]
    if not len(cols):
        return
    xa, xb = cols.min(), cols.max()
    for x in cols:
        yb = np.nonzero(new[:, x])[0].max()
        rel = abs(x - (xa + xb) / 2) / max((xb - xa) / 2, 1)
        for k in (1, 2):
            y = yb + k
            if y < h and cv.mat[y, x] in (GRAVEL, GRASS):
                if k == 1 or rel < 0.55:
                    cv.step[y, x] = 0 if (k == 1 and rel < 0.8) else min(cv.step[y, x], 1)
    if foot and W_ >= 14 and xb - xa > 6:
        n = int(W_ / 6)
        for _ in range(n):
            x = int(rng.uniform(xa + 2, xb - 2))
            col = np.nonzero(new[:, x])[0]
            if not len(col):
                continue
            yb = col.max()
            if foot == "gravel":
                a = int(rng.integers(1, 3))
                for dx in range(-a, a + 1):
                    xx = x + dx
                    if 0 <= xx < w and 0 <= yb < h:
                        cv.mat[yb, xx] = GRAVEL; cv.step[yb, xx] = 1
                        if yb - 1 >= 0 and abs(dx) < a:
                            cv.mat[yb - 1, xx] = GRAVEL; cv.step[yb - 1, xx] = 4 if dx <= 0 else 3
            elif foot == "grass":
                for dx in range(int(rng.integers(1, 4))):
                    L_ = int(rng.integers(2, 5)); xx = x + dx * 2
                    for k in range(L_):
                        y = yb + 1 - k
                        if 0 <= xx < w and 0 <= y < h:
                            cv.mat[y, xx] = GRASS; cv.step[y, xx] = 4 if k >= L_ - 1 else 2


def band_palette(season, hour, depths=(0.0, 0.3, 0.55, 0.8)):
    from palette import willow_palette
    P = willow_palette(season, hour, 0.0)
    for b, d in enumerate(depths):
        Pb = willow_palette(season, hour, d)
        for (m, s), c in Pb.items():
            if m in FOLIAGE:
                P[(m + BAND * b, s)] = c
    return P
