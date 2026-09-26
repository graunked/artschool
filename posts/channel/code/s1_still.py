"""Stage 1: a still strip of water, greyscale, seen from 1.6 m looking ~5 deg down.
It mirrors the sky above the frame and what stands on the far bank: a boulder, a willow, a stump.

Pipeline: physical value study (continuous) -> painted index canvas (Ferrari rules).
"""
import sys
import numpy as np
import pixkit as pk
import channel as ch

W, H = 192, 108
cam = ch.Cam(W, H, f=300, h=1.6, pitch_deg=7.0)
Z_FAR, Z_NEAR = 11.0, 7.0            # water edges (m)
BANK_H, BANK_RUN = 0.20, 0.6         # far bank rises 0.2 m over 1.2 m
NEAR_H, NEAR_RUN = 0.12, 1.0


# ------------------------------------------------------------- world (values 0..255) -------------
CLOUDS = [  # (azimuth col centre, half-width cols, base elev deg, top elev deg)
    (40, 34, 7.6, 12.5), (128, 26, 9.0, 13.5), (178, 22, 8.2, 11.0)]


def sky_value(elev, col):
    """clear sky lighter to the horizon; cumulus above the frame: lumpy lit tops, grey flat bases,
    soft edges (a cloud is a gradient, never a cut-out)."""
    e = np.degrees(np.maximum(elev, 0))
    v = 234 - 44 * np.clip(e / 14.0, 0, 1) ** 0.8
    for k, (c0, hw, eb, et) in enumerate(CLOUDS):
        x = (col - c0) / hw
        bulge = np.clip(1 - x ** 2, 0, 1) ** 0.5 * (0.8 + 0.2 * np.cos(x * 7 + k))
        top = eb + (et - eb) * bulge
        # soft membership: inside the silhouette, fading over ~0.8 deg at the top and sides
        a = np.clip((top - e) / 0.8, 0, 1) * np.clip((e - eb) / 0.3 + 1, 0, 1) * np.clip((1 - np.abs(x)) / 0.15, 0, 1)
        t = np.clip((e - eb) / np.maximum(top - eb, 1e-3), 0, 1)
        cval = 204 + 34 * t ** 0.6
        v = v * (1 - a) + cval * a
    return v


OBJECTS = [  # far-bank objects: x (m), z (m), half-width (m), height (m), kind
    dict(x=-2.3, z=12.4, hw=0.6, ht=0.7, kind='boulder'),
    dict(x=2.6, z=13.5, hw=1.2, ht=1.9, kind='willow'),
]


def obj_height(o, xm):
    """silhouette height (m) at lateral position xm, or 0."""
    u = (xm - o['x']) / o['hw']
    if o['kind'] == 'boulder':
        return np.where(np.abs(u) < 1, o['ht'] * np.clip(1 - np.abs(u) ** 2.5, 0, 1) ** 0.5, 0)
    if o['kind'] == 'willow':
        return np.where(np.abs(u) < 1, o['ht'] * (1 - u ** 2) ** 0.6 * (0.85 + 0.15 * np.cos(u * 9)), 0)
    return np.where(np.abs(u) < 1, o['ht'], 0)


def obj_value(o, yrel, from_below=False):
    """value of object at relative height yrel (0 foot .. 1 top). Seen from below -> the underside."""
    if o['kind'] == 'boulder':
        v = 70 + 90 * yrel ** 1.5          # lit top, dark foot
    elif o['kind'] == 'willow':
        v = 55 + 40 * yrel
    else:
        v = 95 + 20 * yrel
    return v - (18 if from_below else 0)


def ground_value(z):
    """dry gravel, lighter and greyer with distance."""
    return 98 + 30 * np.clip((z - 8) / 60, 0, 1)


# ------------------------------------------------------------- the physical value study ----------
def value_study():
    V = np.zeros((H, W))
    what = np.zeros((H, W), 'U8')
    rows = np.arange(H)[:, None] + 0.5
    cols = np.arange(W)[None, :] + 0.5
    elev = cam.elev_of_row(rows) + 0 * cols
    # sky + distant ridge + scrub line
    sky = sky_value(elev, cols)
    ridge = cam.hz - cam.f * np.tan(np.radians(2.2 + 1.3 * np.sin(cols / 23.0) + 0.6 * np.sin(cols / 7.3)))
    scrub_top = cam.row_of(140, 3.0) + 0.8 * np.sin(cols / 3.1)
    V[:] = sky; what[:] = 'sky'
    rg = np.broadcast_to(rows > ridge, (H, W)); V[rg] = 150; what[rg] = 'ridge'
    sc = np.broadcast_to(rows > scrub_top, (H, W)); V[sc] = 80; what[sc] = 'scrub'
    # ground (far bar), rows below the scrub foot
    zg = cam.z_of_row(rows) + 0 * cols
    grd = (rows > cam.row_of(140)) & (zg > Z_FAR)
    V[grd] = ground_value(zg)[grd]; what[grd] = 'ground'
    # far bank slope: from bank top (z=Z_FAR+RUN, y=BANK_H) down to the waterline
    top_row = cam.row_of(Z_FAR + BANK_RUN, BANK_H)
    wl_row = cam.row_of(Z_FAR)
    slope = np.broadcast_to((rows >= top_row) & (rows < wl_row), (H, W))
    t = (rows - top_row) / (wl_row - top_row) + 0 * cols           # 0 top .. 1 waterline
    V[slope] = (112 - 10 * t - 38 * np.clip((t - 0.72) / 0.28, 0, 1))[slope]   # wet band at the foot
    what[slope] = 'bank'
    # water
    nr = cam.row_of(Z_NEAR)
    water = np.broadcast_to((rows >= wl_row) & (rows < nr), (H, W))
    z = cam.z_of_row(rows) + 0 * cols
    graze = np.arctan(cam.h / z)
    F = ch.fresnel_art(graze)
    refl = sky_value(graze, cols)
    hitwhat = np.full((H, W), 'sky', 'U8')
    hity = np.zeros((H, W)); hitz = np.zeros((H, W))
    # reflected ray height at the far bank top
    ry_bank = (Z_FAR + BANK_RUN - z) * np.tan(graze)
    hb = ry_bank < BANK_H
    refl = np.where(hb, 70, refl); hitwhat[hb] = 'rbank'
    refl_noobj = refl.copy()
    # objects (nearest first overrides)
    for o in sorted(OBJECTS, key=lambda o: -o['z']):
        xm = cam.x_m(cols, o['z']) + 0 * rows
        ht = obj_height(o, xm)
        ry = (o['z'] - z) * np.tan(graze)
        hit = (ry < ht) & (ry >= 0) & ~hb          # the bank edge occludes what stands behind it
        yr = np.clip(ry / np.maximum(ht, 1e-6), 0, 1)
        refl = np.where(hit, obj_value(o, yr, True), refl)
        hitwhat[hit] = o['kind']; hity[hit] = ry[hit]; hitz[hit] = o['z']
    # body: depth profile across the channel, shallow at both edges
    s = (z - Z_NEAR) / (Z_FAR - Z_NEAR)                 # 0 near edge .. 1 far edge
    depth = 0.45 * np.clip(4 * s * (1 - s), 0, 1) ** 0.7
    body = 104 - 40 * np.clip(depth / 0.45, 0, 1)       # wet bed seen through -> darker with depth
    Vw = F * refl + (1 - F) * body
    Vsky = F * refl_noobj + (1 - F) * body
    V[water] = Vw[water]; what[water] = np.where(hitwhat == 'sky', 'wsky', 'wobj')[water]
    # near bank lip and near gravel
    ng = np.broadcast_to(rows >= nr, (H, W))
    zn = cam.z_of_row(rows) + 0 * cols
    V[ng] = (ground_value(zn) - 6)[ng]; what[ng] = 'near'
    # objects themselves (direct view), standing on the bar behind the waterline
    for o in OBJECTS:
        xm = cam.x_m(cols, o['z']) + 0 * rows
        ht = obj_height(o, xm)
        foot = cam.row_of(o['z'], BANK_H)
        topr = cam.row_of(o['z'], BANK_H + ht)
        m = (rows >= topr) & (rows < foot) & (ht > 0)
        yrel = np.clip((foot - rows) / np.maximum(foot - topr, 1e-6), 0, 1)
        V[m] = obj_value(o, yrel)[m]; what[m] = o['kind']
    return V, what, dict(wl=wl_row, nr=nr, top=top_row, F=F, depth=depth, hit=hitwhat, hity=hity, hitz=hitz, body=body, Vsky=Vsky)


PAL = ch.Pal([
    ('sky', ch.grey(240, 230, 220, 210, 200, 190, 180, 170)),
    ('ridge', ch.grey(140, 154)),
    ('scrub', ch.grey(54, 70, 88)),
    ('grav', ch.grey(40, 56, 72, 88, 104, 120, 136, 152, 170, 190)),
    ('wat', ch.grey(84, 102, 120, 140, 160)),
    ('bould', ch.grey(46, 66, 90, 116, 144, 174)),
    ('wil', ch.grey(34, 50, 68, 88, 110)),
])


def paint(seed=1, V=None, what=None, info=None):
    rng = np.random.default_rng(seed)
    if V is None:
        V, what, info = value_study()
    P = PAL
    cv = np.zeros((H, W), int)
    yy, xx = np.mgrid[0:H, 0:W]
    par = (xx + yy) % 2 == 0
    sky_i, sky_v = P.r('sky'), [c[0] for c in P.arr[P.r('sky')]]
    grav_i, grav_v = P.r('grav'), [c[0] for c in P.arr[P.r('grav')]]
    wat_i, wat_v = P.r('wat'), [c[0] for c in P.arr[P.r('wat')]]
    # 1 sky: banded ordered on the sky ramp
    m = what == 'sky'
    cv[m] = ch.to_ramp(V, sky_v, sky_i, 0.5)[m]
    # 2 ridge, scrub
    m = what == 'ridge'
    cv[m] = P.r('ridge')[0]
    # lit ridge crest: the top 1-2 rows of the ridge get the lighter tone
    crest = m & ~np.roll(m, 1, 0)
    cv[crest | (m & np.roll(crest, 1, 0) & par)] = P.r('ridge')[1]
    m = what == 'scrub'
    cv[m] = np.where(rng.random((H, W)) < 0.3, P.r('scrub')[1], P.r('scrub')[0])[m]
    tip = m & ~np.roll(m, 1, 0)
    cv[tip & par] = P.r('scrub')[2]
    # 3 far bar + 4 far bank slope: gravel as light-over-dark stone pairs on an ordered base
    zrow = cam.z_of_row(np.arange(H) + 0.5)
    m = np.isin(what, ['ground', 'bank'])
    gv = V + (pk.value_noise(H, W, 6, rng, 2, aniso=(0.4, 2.0)) - 0.5) * 18
    # the bank's wet foot: value drops over the lowest ~45% of the slope, most at the waterline
    tbank = np.clip((np.arange(H)[:, None] + 0.5 - info['top']) / (info['wl'] - info['top']), 0, 1)
    gv = gv - np.where(what == 'bank', np.interp(tbank, [0, 0.55, 0.8, 1.0], [0, 0, 22, 46]), 0)
    # the bank slope faces us: it is the same gravel, a little darker, seen at its own distance
    zb = np.where(what == 'bank', Z_FAR + BANK_RUN * 0.5, zrow[:, None])
    zeff = np.where(what == 'bank', zb, zrow[:, None]).min(axis=1)
    g = ch.gravel(m, gv, np.where(m.any(1), zeff, np.nan), cam.f, rng, grav_i, grav_v, (0.03, 0.16), 0.9)
    cv[m] = g[m]
    # 4b the objects themselves (painted before the water, which copies them)
    bm = what == 'boulder'
    if bm.any():
        ys_, xs_ = np.nonzero(bm)
        m2, lab = ch.boulder_mask(H, W, xs_.mean() + 0.5, ys_.max() + 1, xs_.max() - xs_.min() + 1,
                                  ys_.max() - ys_.min() + 1, np.random.default_rng(7))
        # the faceted silhouette replaces the dome, in the direct view and in the reflection
        what[bm & ~m2] = 'ground'
        what[m2] = 'boulder'
        cv[bm & ~m2] = ch.gravel(bm & ~m2, V, np.full(H, 12.4), cam.f, rng, grav_i, grav_v)[bm & ~m2]
        ch.paint_faceted(cv, m2, lab, P.r('bould'), rng, top=(3.1, 3.9), sun=(1.9, 2.9), shade=(0.5, 1.2))
    ch.paint_shrub(cv, what == 'willow', P.r('wil'), rng)
    # 5 the far bank's wet foot: the rows just above the waterline darken (wet), the last darkest
    wl = int(np.floor(info['wl'])); nr = int(np.floor(info['nr']))
    notobj = ~np.isin(what, ['boulder', 'willow'])
    bankrows = what == 'bank'
    tb = np.zeros((H, W)); 
    for r in range(H):
        if bankrows[r].any():
            tb[r] = (r - info['top']) / (info['wl'] - info['top'])
    # 6 water -- painted row-class by row-class, top to bottom (Ferrari's horizontal logic)
    water = np.isin(what, ['wsky', 'wobj'])
    hit = info['hit'].copy(); hity = info['hity']
    # billboard mirror of the PAINTED objects: a pixel at row rd reflects to 2*R0 - rd, R0 the row of
    # the object's foot at water level; hidden where the bank's own reflection is in front
    rb_end = cam.mirror_row(Z_FAR + BANK_RUN, BANK_H)
    src_row = np.full((H, W), -1)
    hit[np.isin(hit, ['boulder', 'willow'])] = 'sky'
    for o in OBJECTS:
        R0 = cam.row_of(o['z'], 0.0)
        rd, cd = np.nonzero(what == o['kind'])
        rm = np.floor(2 * R0 - rd - 0.5).astype(int)
        ok = (rm >= 0) & (rm < H)
        rd, cd, rm = rd[ok], cd[ok], rm[ok]
        ok = water[rm, cd] & (rm + 0.5 >= rb_end)
        hit[rm[ok], cd[ok]] = o['kind']; src_row[rm[ok], cd[ok]] = rd[ok]
    ws_v = sky_v + wat_v; ws_i = np.r_[sky_i, wat_i]
    # A reflected bank face: mirrored bank texture, smeared sideways, compressed a step
    rb = water & (hit == 'rbank')
    rbank_rows = sorted(set(np.nonzero(rb)[0]))
    for k, r in enumerate(rbank_rows):
        src_r = wl - 1 - k                                     # mirror about the waterline
        rowv = cv[src_r].copy()
        # smear: hold each value over 2-4 px runs (reflections streak horizontally)
        x = 0
        while x < W:
            Lr = int(rng.integers(2, 5)); rowv[x:x + Lr] = rowv[x]; x += Lr
        st = np.searchsorted(grav_i, rowv)
        st = np.clip(st - (1 if k == 0 else 0), 0, len(grav_i) - 1)
        cv[r, rb[r]] = grav_i[st][rb[r]]
    # the reflected band's lower edge steps: runs where the sky already shows through the last row
    if rbank_rows:
        r = rbank_rows[-1]; x = 0
        while x < W:
            Lr = int(rng.integers(2, 9))
            if rng.random() < 0.45:
                sel = slice(x, min(W, x + Lr)); cv[r, sel] = np.where(rb[r, sel], -5, cv[r, sel])
            x += Lr
    # C sheen + D sky reflection: the physical value, lifted just under the reflected bank
    m = water & ((hit == 'sky') | (cv == -5))
    Vs = info['Vsky'].copy()
    below_rb = np.zeros((H, W))
    if rbank_rows:
        last = max(rbank_rows)
        for r in range(last + 1, nr):
            below_rb[r] = r - last
    Vs = Vs + np.where((below_rb >= 1) & (below_rb <= 3), 26 - 6 * below_rb, 0)
    # wind-lines: faint ripples mirror a slightly different sky -> long 1-row streaks, lighter or darker
    streak = pk.value_noise(H, W, 1, rng, 1, aniso=(1.0, 22.0)) - 0.5
    Vs = Vs + streak * 16
    # the index climbs with nearness (Ferrari): nearer water mirrors higher sky -- stated, not left
    # to the few-percent physical drop over 25 rows
    near_t = np.clip((np.arange(H)[:, None] - info['wl']) / (info['nr'] - info['wl']), 0, 1)
    Vs = Vs - near_t * 14
    Vs = np.minimum(Vs, sky_v[1] - 3)          # the mirror never reaches the sky's top step
    cv[m] = ch.to_ramp_checker(Vs, ws_v, ws_i, 0.3)[m]
    # reflected objects: COPY the direct painting at the hit point, then the tone law
    for kind, rname in (('boulder', 'bould'), ('willow', 'wil')):
        m = water & (hit == kind)
        if not m.any():
            continue
        rr, cc = np.nonzero(m)
        src_r = src_row[rr, cc]
        vals = cv[src_r, cc]
        if kind == 'boulder':      # a convex solid: the mirror sees its shaded underside
            cv[rr, cc] = ch.tone_down(ch.tone_down(vals, P.r(rname)), P.r(rname), dark_up=False)
        else:                      # foliage keeps its texture; only the range compresses
            cv[rr, cc] = ch.tone_down(vals, P.r(rname))
        # wind lines cross dark reflections as dotted light dashes
        rows_k = sorted(set(rr))
        for r in rows_k[2::5]:
            xs_ = cc[rr == r]
            if len(xs_) < 4 or rng.random() < 0.3:
                continue
            x0 = int(rng.integers(xs_.min(), max(xs_.min() + 1, xs_.max() - 3)))
            for s_ in range(int(rng.integers(2, 7))):
                xi = x0 + 2 * s_ + ((x0 + r) % 2)
                if xi <= xs_.max() and hit[r, xi] == kind:
                    cv[r, xi] = P.r(rname)[-2]
    # soften reflection silhouettes: a 50% checker one pixel either side of a hit change
    hh = np.where(water, hit, '')
    chg_h = np.zeros((H, W), bool); chg_h[:, 1:] = hh[:, 1:] != hh[:, :-1]
    chg_v = np.zeros((H, W), bool); chg_v[1:] = hh[1:] != hh[:-1]
    objpx = water & np.isin(hit, ['boulder', 'willow'])
    for dx in (-1, 1):
        nb = np.roll(cv, dx, 1); nbh = np.roll(hh, dx, 1)
        sel = water & (nbh != hh) & par & np.isin(nbh, ['boulder', 'willow']) & (hh == 'sky')
        cv[sel] = nb[sel]
    nb = np.roll(cv, 1, 0); nbh = np.roll(hh, 1, 0)
    sel = water & (nbh != hh) & par & np.isin(nbh, ['boulder', 'willow']) & (hh == 'sky')
    cv[sel] = nb[sel]
    # B the far waterline: rare short glints only (the reflected bank carries the edge)
    x = int(rng.integers(0, 10))
    while x < W:
        if rng.random() < 0.25 and hit[wl, x] in ('rbank', 'sky'):
            Lr = int(rng.integers(2, 4)); cv[wl, x:x + Lr] = wat_i[4]
        x += int(rng.integers(8, 30))
    # E near shallows: the shelf where you see the bed (channel.shallows)
    edge_row = np.full(W, nr)
    cv = ch.shallows(cv, water, edge_row, rng, wat_i, hit == 'sky', rows=6, stones_per_px=0.12, lap=wat_i[4])
    # wind lines: solid 1-row runs a step LIGHTER than the lighter tint of the local checker, sparse,
    # longer and fainter far, shorter near (a dotted dash on a checker would vanish into it)
    ws_sorted = list(np.array(ws_i)[np.argsort(ws_v)])
    top_r = wl + len(rbank_rows) + 4
    for r in range(top_r, nr - 5):
        if rng.random() > 0.3:
            continue
        x = int(rng.integers(0, W - 4)); Ld = int(rng.integers(3, 12) * (1.4 - 0.6 * (r - top_r) / max(nr - top_r, 1)))
        seg = [xi for xi in range(x, min(W, x + Ld)) if water[r, xi] and hit[r, xi] == 'sky' and cv[r, xi] in ws_sorted]
        if len(seg) < 3:
            continue
        j = max(ws_sorted.index(cv[r, xi]) for xi in seg)
        for xi in seg:
            cv[r, xi] = ws_sorted[min(j + 1, len(ws_sorted) - 2)]
    # 7 near bank: lapping light line at the waterline, wet dark band, then gravel with pebbles
    ll = np.zeros(W, bool); x = int(rng.integers(0, 3))
    while x < W:
        Lr = int(rng.integers(2, 7)); ll[x:x + Lr] = rng.random() < 0.45; x += Lr + int(rng.integers(2, 8))
    cv[nr, ll] = wat_i[4]
    cv[nr, ~ll] = np.where(rng.random((~ll).sum()) < 0.7, grav_i[1], grav_i[0])
    m = (what == 'near') & (yy > nr)
    wet = np.interp(yy - nr, [1, 2, 3, 4, 6], [48, 34, 20, 8, 0])     # the wet margin, graded, textured
    nv = V + (pk.value_noise(H, W, 5, rng, 2) - 0.5) * 20 - wet
    g = ch.gravel(m, nv, np.where(np.arange(H) > nr, zrow, np.nan), cam.f, rng, grav_i, grav_v, (0.03, 0.16), 1.0)
    cv[m] = g[m]
    return cv, V, what, info


if __name__ == '__main__':
    V, what, info = value_study()
    g = np.clip(V, 0, 255).astype(np.uint8)
    pk.save(np.stack([g] * 3, -1), 'study/s1_value_study.png', 4)
    cv, *_ = paint(1, V, what, info)
    pk.save(PAL.rgb(cv), 'study/s1_paint_%s.png' % (sys.argv[1] if len(sys.argv) > 1 else 'v1'), 4)
