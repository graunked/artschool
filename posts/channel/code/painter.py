"""painter.py -- Ferrari's rules applied to the G-buffer from world.raycast.

Order matters: sky, ground (gravel/grass) are painted first; the water then COPIES the painted
ground for its reflections (tone law), mirrors the sky on the sky ramp as a checker material,
and adds its own marks (sheen, shelves, stones, wind lines, lap line).
"""
import numpy as np
import pixkit as pk
import channel as ch


def grey_palette():
    return ch.Pal([
        ('sky', ch.grey(240, 230, 220, 210, 200, 190, 180, 170)),
        ('grav', ch.grey(40, 56, 72, 88, 104, 120, 136, 152, 170, 190)),
        ('wat', ch.grey(84, 102, 120, 140, 160)),
        ('grass', ch.grey(28, 44, 62, 82, 104, 128, 152)),
        ('earth', ch.grey(26, 38, 52, 68)),
        ('bould', ch.grey(46, 66, 90, 116, 144, 174)),
        ('wil', ch.grey(34, 50, 68, 88, 110)),
        ('ridge', ch.grey(140, 154)),
        ('wmir', ch.grey(240, 230, 220, 210, 200, 190, 180, 170)),
        ('scrub', ch.grey(54, 70, 88)),
    ])


def vals(P, name):
    return [c[0] for c in P.arr[P.r(name)]]


def sky_value(elev_deg, col, W, clouds=()):
    """clear sky, lighter to the horizon; cloud BANKS (stratocumulus): soft horizontal bands whose
    top edge undulates gently and whose base is flat -- mirrored, they lie in the water as soft
    horizontal light shapes, never as hanging cones."""
    e = np.maximum(elev_deg, 0)
    v = 234 - 44 * np.clip(e / 14.0, 0, 1) ** 0.8
    for k, (c0, hw, eb, et) in enumerate(clouds):
        x = (col - c0 * W) / (hw * W)
        along = np.clip(1 - np.abs(x) ** 3, 0, 1)                       # a long flat bank, soft ends
        top = eb + (et - eb) * (0.7 + 0.12 * np.sin(x * 5.0 + k) + 0.08 * np.sin(x * 13.0 + 2 * k)) * along
        a = np.clip((top - e) / 1.2, 0, 1) * np.clip((e - eb) / 0.6 + 1, 0, 1) * along
        t = np.clip((e - eb) / np.maximum(top - eb, 1e-3), 0, 1)
        v = v * (1 - a) + (210 + 24 * t ** 0.7) * a
    return v


DEFAULT = dict(
    clouds=((0.25, 0.35, 7.0, 11.5), (0.8, 0.3, 9.0, 13.0)),
    ground_top=100.0, ground_far=128.0, haze_z=70.0,
    wet_h=0.07, wet_drop=62.0, face_dark=0.35,
    stones=(0.04, 0.2), stone_density=0.9, stone_aspect=0.75,
    bed=86.0, milk=168.0, visibility=0.06,
    shelf=(0.05, 0.13), sheen=26.0, nearness=14.0, streaks=16.0, wind=0.3,
    lap=0.45, grass=None,
)


def paint(G, cam, P, rng, prm=None):
    prm = dict(DEFAULT, **(prm or {}))
    H, W = G['kind'].shape
    cv = np.zeros((H, W), int)
    yy, xx = np.mgrid[0:H, 0:W]
    par = (xx + yy) % 2 == 0
    sky_i, sky_v = P.r('sky'), vals(P, 'sky')
    grav_i, grav_v = P.r('grav'), vals(P, 'grav')
    wat_i, wat_v = P.r('wat'), vals(P, 'wat')
    kind = G['kind']
    elev = np.degrees(np.arctan((cam.hz - (yy + 0.5)) / cam.f))
    # ---- sky
    m = kind == 0
    sv = sky_value(elev, xx + 0.5, W, prm['clouds'])
    cv[m] = ch.to_ramp(sv, sky_v, sky_i, 0.5)[m]
    bd = prm.get('backdrop')
    if bd:
        cols = xx + 0.5
        ridge_e = bd['ridge'](cols / W)                      # degrees above the horizon
        rg = (kind == 0) & (elev < ridge_e)
        ri = P.r('ridge')
        cv[rg] = ri[0]
        crest = rg & ~np.roll(rg, 1, 0)
        cv[crest | (rg & np.roll(crest, 1, 0) & par)] = ri[1]
        zs_, hs_ = bd['scrub']
        # the far scrub: clumps of varied width and height, gaps between, never one bump repeated
        prof = np.zeros(W); x_ = 0
        while x_ < W:
            wdt = int(rng.integers(3, 14)); hgt = rng.uniform(0.3, 1.0) if rng.random() < 0.85 else -0.6
            u_ = (np.arange(wdt) + 0.5) / wdt
            prof[x_:x_ + wdt] = (hgt * np.sqrt(np.clip(1 - (2 * u_ - 1) ** 2, 0, 1)))[:W - x_]
            x_ += wdt
        top = cam.row_of(zs_, 0.0) - (cam.row_of(zs_, 0.0) - cam.row_of(zs_, hs_)) * prof[None, :] + 0 * yy
        sc = ((kind == 0) & (yy + 0.5 > top)) | ((kind == 1) & (G['z'] > zs_))
        si = P.r('scrub')
        cv[sc] = np.where(rng.random((H, W)) < 0.3, si[1], si[0])[sc]
        tip = sc & ~np.roll(sc, 1, 0)
        cv[tip & par] = si[2]
        kind = kind.copy(); kind[sc] = 3                     # the scrub band hides the ground behind it
    # ---- ground: value from distance haze, facing (faces toward us get less sky), wetness by height
    gm = kind == 1
    z = np.where(np.isfinite(G['z']), G['z'], 1e3)
    base = prm['ground_top'] + (prm['ground_far'] - prm['ground_top']) * np.clip((z - 6) / prm['haze_z'], 0, 1)
    fac = np.clip(G['facing'], 0, 4)
    base = base * (1 - prm['face_dark'] * np.clip(fac / 1.5, 0, 1))
    wetness = np.clip(1 - G['y'] / prm['wet_h'], 0, 1) ** 1.5
    base = base - prm['wet_drop'] * wetness * gm
    near_amt = np.clip((cam.f / np.maximum(z, 1)) / 40.0, 0.25, 1.0)
    base = base + (pk.value_noise(H, W, 10, rng, 2, aniso=(0.4, 3.0)) - 0.5) * 12
    zrow = np.array([np.median(z[r][gm[r]]) if gm[r].any() else np.nan for r in range(H)])
    g = ch.gravel(gm, base, zrow, cam.f, rng, grav_i, grav_v, prm['stones'], prm['stone_density'], aspect=prm['stone_aspect'])
    cv[gm] = g[gm]
    # the contact line: ground directly above water is the wet, darkest row (a flat run, never an
    # ordered-dither row, which reads as a dotted line)
    contact = gm & np.roll(kind == 2, -1, 0)
    cv[contact] = np.where(rng.random((H, W)) < 0.85, grav_i[0], grav_i[1])[contact]
    grass_layer = None
    if prm['grass'] is not None:
        grass_layer = paint_grass(cv, G, cam, P, rng, prm['grass'])
    obj_layer = paint_objects(cv, G, cam, P, rng, prm.get('objects', ()))
    # ---- water
    wm = kind == 2
    gr = G['graze']
    F = ch.fresnel_art(gr)
    skyrefl = sky_value(np.degrees(gr), xx + 0.5, W, prm['clouds'])
    d = G['depth']
    # the body seen through the surface: the wet bed fading into the water's own milk with depth
    # (glacial flour makes braided rivers nearly opaque: visibility of a few cm)
    tvis = np.exp(-d / prm['visibility'])
    body = prm['bed'] * tvis + prm['milk'] * (1 - tvis)
    Vs = F * skyrefl + (1 - F) * body
    # sheen: directly under a ground reflection the water mirrors the lowest sky
    under = np.zeros((H, W))
    for c in range(W):
        last = -99
        for r in range(H):
            if wm[r, c] and G['refl'][r, c] == 1:
                last = r
            elif wm[r, c]:
                under[r, c] = r - last if last >= 0 else 99
    Vs = Vs + np.where((under >= 1) & (under <= 3), prm['sheen'] - 6 * under, 0)
    Vs = Vs + (pk.value_noise(H, W, 1, rng, 1, aniso=(1.0, 22.0)) - 0.5) * prm['streaks']
    # the index climbs with nearness
    wr = np.nonzero(wm.any(1))[0]
    if len(wr):
        zw = np.where(wm, z, np.nan)
        zlo, zhi = np.nanpercentile(zw, 2), np.nanpercentile(zw, 98)
        t = np.clip((np.log(zhi) - np.log(np.maximum(z, 1e-3))) / max(np.log(zhi) - np.log(zlo), 1e-3), 0, 1)
        Vs = Vs - t * prm['nearness']
    mir_i = P.r('wmir') if prm.get('milky', True) and 'wmir' in P.ramps else sky_i
    ws_v = sky_v + wat_v; ws_i = np.r_[mir_i, wat_i]
    Vs = np.minimum(Vs, sky_v[1] - 3)
    msky = wm & (G['refl'] == 0)
    cv[msky] = ch.to_ramp_checker(Vs, ws_v, ws_i, 0.3)[msky]
    # reflected ground: copy the painted pixel at the hit's direct row, tone law on its ramp
    mg = wm & (G['refl'] == 1)
    rr, cc = np.nonzero(mg)
    src = np.clip(np.floor(G['refl_row'][rr, cc]).astype(int), 0, H - 1)
    for k in range(4):                         # the source must be painted ground: step up if not
        bad = kind[src, cc] != 1
        src = np.where(bad, np.maximum(src - 1, 0), src)
    vals_src = cv[src, cc]
    out = vals_src.copy()
    for rname in ('grav', 'grass', 'earth'):
        out = ch.tone_down(out, P.r(rname))
    # smear sideways: reflections streak horizontally
    cv[rr, cc] = out
    for r in sorted(set(rr)):
        cs = cc[rr == r]
        x0 = 0
        rowv = cv[r].copy()
        while x0 < len(cs):
            L = int(rng.integers(2, 6))
            rowv[cs[x0:x0 + L]] = rowv[cs[x0]]
            x0 += L
        cv[r, cs] = rowv[cs]
    if grass_layer is not None:
        mirror_grass(cv, grass_layer, G, cam, P, wm)
    if obj_layer is not None:
        mirror_objects(cv, obj_layer, G, cam, P, wm, rng)
    # a reflection is still water: wind lines cross it (dotted, a light water tint), and its lower
    # edge dissolves into the mirror through two rows of checker
    for r in sorted(set(rr)):
        if rng.random() < 0.3:
            cs = cc[rr == r]
            x0 = int(rng.integers(0, max(1, len(cs) - 3)))
            for k in range(x0, min(len(cs), x0 + int(rng.integers(2, 6)))):
                cv[r, cs[k]] = wat_i[2]
    # its lower edge steps: the reflection runs 0-2 rows further down in runs of 3-10 px (never a
    # one-row checker, which reads as a dotted line)
    ext = np.zeros(W, int); x = 0
    while x < W:
        L = int(rng.integers(3, 11)); ext[x:x + L] = rng.choice([0, 0, 1, 2]); x += L
    for c in range(W):
        rows_c = np.nonzero(mg[:, c])[0]
        if len(rows_c) == 0:
            continue
        rb = rows_c.max()
        for k in range(1, ext[c] + 1):
            if rb + k < H and wm[rb + k, c] and G['refl'][rb + k, c] == 0:
                cv[rb + k, c] = cv[rb, c]
    # the reflected band's lower edge steps: where the sky shows through, in runs
    # (already stepped by the terrain's breathing; nothing to add)
    # shelves: see-in where it's shallow -- flat tints from depth bands, a light lip on the outer shelf
    sh1, sh2 = prm['shelf']
    msh = wm & (G['refl'] == 0)
    inner = msh & (d < sh1)
    outer = msh & (d >= sh1) & (d < sh2)
    db_, _ = pk.col_dist(gm)                     # rows to the nearest ground BELOW: the near margin
    near_side = db_ <= prm.get('shelf_rows', 10)
    cv[inner & near_side] = wat_i[1]
    cv[outer & near_side] = np.where(par, wat_i[2], cv)[outer & near_side] if prm.get('shelf_checker') else wat_i[2]
    # stones in the shelf: flat lozenges, sky-lit top; the ones at the edge sit on the lap line
    cand = np.argwhere((inner | outer) & near_side)
    for i in range(int(len(cand) * 0.05)):
        r, c0 = cand[int(rng.integers(0, len(cand)))]
        L = int(rng.integers(2, 7))
        cs = np.arange(c0, min(W, c0 + L))
        if not (wm[r, cs].all() and r > 0):
            continue
        cv[r, cs] = wat_i[0]
        if L >= 3:
            cv[r - 1, cs[1:-1]] = wat_i[3] if d[r, c0] > sh1 * 0.5 else wat_i[4]
    # wind lines on the open mirror
    ws_sorted = list(np.array(ws_i)[np.argsort(ws_v)])
    for r in wr:
        if rng.random() > prm['wind']:
            continue
        x0 = int(rng.integers(0, W - 4)); L = int(rng.integers(3, 12))
        seg = [x for x in range(x0, min(W, x0 + L)) if msky[r, x] and cv[r, x] in ws_sorted and not (inner | outer)[r, x]]
        if len(seg) < 3:
            continue
        j = max(ws_sorted.index(cv[r, x]) for x in seg)
        for x in seg:
            cv[r, x] = ws_sorted[min(j + 1, len(ws_sorted) - 2)]
    # the near waterline: a broken light lap line on the last water row above the near bank
    last_water = wm & np.roll(gm, -1, 0) & near_side
    runs = np.zeros(W, bool); x = int(rng.integers(0, 3))
    while x < W:
        L = int(rng.integers(2, 7)); runs[x:x + L] = rng.random() < prm['lap']; x += L + int(rng.integers(2, 8))
    sel = last_water & runs[None, :]
    cv[sel] = wat_i[4]
    if prm.get('terrain') is not None and getattr(prm['terrain'], 'riffles', None):
        paint_flow(cv, G, cam, P, rng, prm['terrain'], prm)
    if grass_layer is not None:                 # grass stands in front of the water: redraw it
        lay = grass_layer['layer']
        cv[lay >= 0] = lay[lay >= 0]
    return cv


# ----------------------------------------------------------------------------- grass ----------
def paint_grass(cv, G, cam, P, rng, gp):
    """grass on ground above gp['min_y'] (the bank top): masses first, detail at the edges.
    body: vertical strands (runs of adjacent tones), lighter near the crest, darker deeper, with
      paired [dark gap][highlight][stem] marks;
    crest: clumps every 3-8 px, each fanning 3-6 single 1-px blades, dark against what is behind,
      a rim-lit tip now and then (the fringe, never a cut edge);
    lip: where the mass ends over a cut face, a few blades droop down over the dark face, lit.
    The cut face under the grass is earth (dark), not gravel."""
    H, W = cv.shape
    gi = P.r('grass'); ei = P.r('earth')
    layer = np.full((H, W), -1)
    foot = np.full((H, W), -1.0)
    kind = G['kind']
    zw = np.nanmedian(np.where(kind == 2, G['z'], np.nan))
    side = (G['z'] > zw) if gp.get('side', 'far') == 'far' else (G['z'] < zw)
    on = (kind == 1) & (G['y'] > gp.get('min_y', 0.1)) & (G['facing'] < 1.5) & side
    face = (kind == 1) & (G['facing'] >= 1.5) & (G['y'] > 0.04) & side
    # earth face: dark, with a lighter checker near its top (the root mat catches light)
    for c in range(W):
        rows = np.nonzero(face[:, c])[0]
        for k, r in enumerate(rows):
            layer[r, c] = ei[1] if rng.random() < 0.7 else ei[0]
            if k < 2 and (r + c) % 2 == 0:
                layer[r, c] = ei[2]
            foot[r, c] = cam.row_of(G['z'][r, c], 0.0)
    zmed = np.nanmedian(np.where(on, G['z'], np.nan)) if on.any() else 6.0
    bl = max(3, gp.get('blade_m', 0.35) * cam.f / zmed)          # blade length in px
    # body: Ferrari's lawn strands (master-copy C3): down each column, runs step through ADJACENT
    # values, lighter near the crest; highlights paired with a darker stem, black gaps in shade;
    # run length grows with nearness; laid on a clean staircase lean
    topr = np.full(W, H)
    for c in range(W):
        rr_ = np.nonzero(on[:, c])[0]
        if len(rr_):
            topr[c] = rr_.min()
    cls = list(gi[:6])
    dtop = np.clip((np.arange(H)[:, None] - topr[None, :]) / max(bl * 1.6, 1), 0, 1)
    # tussock clumps: domes of ~bl across; crown lit, skirt dark; light falls in drifts across them
    clump = np.zeros((H, W))
    for i in range(int(on.sum() / max(bl * bl * 0.5, 4)) + 1):
        pts_ = np.argwhere(on)
        if not len(pts_):
            break
        r, c = pts_[int(rng.integers(0, len(pts_)))]
        rad = bl * rng.uniform(0.5, 1.0)
        yy_, xx_ = np.mgrid[0:H, 0:W]
        dd = ((xx_ - c) / rad) ** 2 + ((yy_ - r) / (rad * 0.6)) ** 2
        clump = np.where(dd < 1, np.maximum(clump, (1 - dd) * np.where(yy_ < r, 1.0, 0.3)), clump)
    drift = pk.value_noise(H, W, max(4.0, bl), rng, 1, aniso=(0.7, 2.0))
    dtop = np.clip(dtop - 0.8 * clump + 0.5 * (drift - 0.5), 0, 1)
    runm = float(np.clip(1.2 + bl * 0.12, 1.5, 5.5))
    shear = 3 if gp.get('lean', 0.0) >= 0 else -3
    S = ch.sheared_strands(H, W, rng, shear, mask=on,
                           classes_fn=lambda y, x: cls,
                           u_fn=lambda y, x: 4.2 - 3.0 * dtop[y, min(max(x, 0), W - 1)],
                           runmean_fn=lambda y, x: runm, spread=0.6, pair_p=0.65, pair_side=+1,
                           gap_p=0.4, gap_class=gi[0], jog_p=0.25)
    layer[on] = S[on]
    foot[on] = cam.row_of(G['z'][on], 0.0)
    # crest ticks: the mass top is never a ruled line -- dense 1-px ticks of random height (1-5 px),
    # leaning, lit toward their tips; the gaps between them are the holes along the crest
    for x in range(W):
        if topr[x] >= H or rng.random() < 0.3:
            continue
        th = int(rng.integers(1, max(2, int(bl * 0.15)) + 2))
        lean = gp.get('lean', 0.0) + rng.normal(0, 0.3)
        xf = x + 0.5
        for k in range(1, th + 1):
            y = topr[x] - k; xi = int(xf)
            if y < 0 or not (0 <= xi < W):
                break
            layer[y, xi] = gi[4] if k >= th - 1 else gi[3]
            foot[y, xi] = foot[topr[x], x]
            xf += lean * 0.4
    newtop = np.full(W, H)
    for x in range(W):
        rr_ = np.nonzero(layer[:, x] >= 0)[0]
        rr_ = rr_[rr_ <= topr[x]] if topr[x] < H else []
        if len(rr_):
            newtop[x] = rr_.min()
    # crest fringe (V16): a few CLUMPS with long gaps between; in each, 2-6 blades of very different
    # lengths (3:1), mostly leaning one way and curving, a couple crossing
    tops = [(c, int(newtop[c])) for c in range(W) if newtop[c] < H]
    topd = dict(tops)
    c = int(rng.integers(0, 12))
    side = 1 if gp.get('lean', 0.0) >= 0 else -1
    while c < W:
        if c in topd:
            r0 = topd[c]
            nb = int(rng.integers(2, 7))
            for i in range(nb):
                L = int(bl * rng.uniform(0.15, 0.6) * (1.0 if i else 1.3))
                lean = side * rng.uniform(0.2, 1.4) if rng.random() < 0.8 else -side * rng.uniform(0.2, 0.8)
                xf = c + 0.5 + rng.normal(0, 1.2)
                lit = rng.random() < gp.get('lit', 0.35)
                for sidx in range(L):
                    y = r0 - sidx; x = int(xf)
                    if y < 0 or not (0 <= x < W):
                        break
                    layer[y, x] = gi[4] if (lit and sidx >= L - 2) else gi[1]
                    foot[y, x] = cam.row_of(G['z'][r0, min(c, W - 1)], 0.0)
                    xf += lean * (sidx / max(L, 1)) ** 2.0
        c += int(rng.integers(10, 30))
    # lip: blades drooping over the cut face
    for c in range(W):
        rows = np.nonzero(face[:, c])[0]
        if len(rows) == 0 or rng.random() > gp.get('droop', 0.45):
            continue
        r0 = rows.min()
        L = int(rng.integers(2, max(3, int(bl * 0.35))))
        x = c
        for k in range(L):
            if r0 + k < H and face[r0 + k, x]:
                layer[r0 + k, x] = gi[3] if k < L - 1 else gi[2]
            if rng.random() < 0.3:
                x = min(W - 1, max(0, x + rng.choice([-1, 1])))
    sel = layer >= 0
    cv[sel] = layer[sel]
    return dict(layer=layer, foot=foot)


def mirror_grass(cv, gl, G, cam, P, wm):
    """billboard mirror of the grass marks about each mark's own foot row at water level; hidden
    behind nearer ground reflections is automatic since we only write onto water pixels whose
    reflection is sky or ground further than the grass."""
    H, W = cv.shape
    layer, foot = gl['layer'], gl['foot']
    rs, cs = np.nonzero(layer >= 0)
    rm = np.floor(2 * foot[rs, cs] - rs - 0.5).astype(int)
    ok = (rm >= 0) & (rm < H)
    rs, cs, rm = rs[ok], cs[ok], rm[ok]
    ok = wm[rm, cs]
    rs, cs, rm = rs[ok], cs[ok], rm[ok]
    vals = ch.tone_down(layer[rs, cs], P.r('grass'))
    cv[rm, cs] = vals


# ----------------------------------------------------------------------------- objects --------
def paint_objects(cv, G, cam, P, rng, objs):
    """billboard objects standing on the ground: dict(kind='boulder'|'willow', x, z, w, h) in metres.
    Depth-tested against the ground; returns a layer (index, foot row at water level, kind)."""
    if not objs:
        return None
    H, W = cv.shape
    layer = np.full((H, W), -1); foot = np.full((H, W), -1.0); kinds = np.full((H, W), '', 'U8')
    for o in sorted(objs, key=lambda o: -o['z']):
        c0 = cam.col_of(o['x'], o['z'])
        # the ground height under the object: sample the G-buffer's heights near its foot row
        y0 = o.get('y0', 0.0)
        base = cam.row_of(o['z'], y0)
        wpx = o['w'] * cam.f / o['z']; hpx = o['h'] * cam.f / o['z']
        sub = np.random.default_rng(o.get('seed', 7))
        if o['kind'] == 'boulder':
            m, lab = ch.boulder_mask(H, W, c0, base, wpx, hpx, sub)
        else:
            yy, xx = np.mgrid[0:H, 0:W]
            u = (xx + 0.5 - c0) / (wpx / 2)
            top = base - hpx * np.clip(1 - u ** 2, 0, 1) ** 0.6 * (0.85 + 0.15 * np.cos(u * 9 + o.get('seed', 0)))
            m = (np.abs(u) < 1) & (yy + 0.5 >= top) & (yy + 0.5 < base)
        # depth test: hidden where the ground in front is nearer
        m &= ~((G['kind'] != 0) & (G['z'] < o['z'] - 0.3))
        tmp = cv.copy()
        if o['kind'] == 'boulder':
            ch.paint_faceted(tmp, m, lab, P.r('bould'), sub, top=(3.1, 3.9), sun=(1.9, 2.9), shade=(1.0, 1.0))
        else:
            ch.paint_shrub(tmp, m, P.r('wil'), sub)
        layer[m] = tmp[m]; kinds[m] = o['kind']
        foot[m] = cam.row_of(o['z'], 0.0)
    cv[layer >= 0] = layer[layer >= 0]
    return dict(layer=layer, foot=foot, kind=kinds)


def mirror_objects(cv, ol, G, cam, P, wm, rng):
    """a pixel at row r reflects to 2*R0 - r (R0: the object's foot at water level); only onto water
    whose own reflection is sky or ground FARTHER than the object (nearer bank reflections occlude)."""
    H, W = cv.shape
    rs, cs = np.nonzero(ol['layer'] >= 0)
    rm = np.floor(2 * ol['foot'][rs, cs] - rs - 0.5).astype(int)
    ok = (rm >= 0) & (rm < H)
    rs, cs, rm = rs[ok], cs[ok], rm[ok]
    ok = wm[rm, cs] & ~((G['refl'][rm, cs] == 1) & (G['refl_z'][rm, cs] < cam.f * cam.h / np.maximum(ol['foot'][rs, cs] - cam.hz, 1e-3) - 0.2))
    rs, cs, rm = rs[ok], cs[ok], rm[ok]
    vals = ol['layer'][rs, cs]
    kd = ol['kind'][rs, cs]
    out = vals.copy()
    b = kd == 'boulder'
    out[b] = ch.tone_down(ch.tone_down(vals[b], P.r('bould')), P.r('bould'), dark_up=False)
    out[~b] = ch.tone_down(vals[~b], P.r('wil'))
    cv[rm, cs] = out


# ----------------------------------------------------------------------------- moving water ---
def paint_flow(cv, G, cam, P, rng, T, prm):
    """a riffle, said with as few deliberate marks as possible (flow along +x):
    1 the mirror breaks: inside the rough zone, reflected banks are chopped into dashes with glare
      between (a broken reflection is still recognisably the bank);
    2 wave marks: short horizontal light-over-dark PAIRS (the back of a wave mirrors low bright sky,
      its face toward us mirrors high sky / shows the bed), laid along crest lines that cross the
      channel, bowed downstream in the middle, broken where the zone thins;
    3 foam: white 1-3 px flecks on the crests at the sill's lip only;
    4 stones: emergent ones are dark wet lozenges with a white pillow upstream and a dashed V wake
      downstream; submerged ones show as dark pockets with a light hump dash just downstream;
    5 streaks: below the riffle, long faint light lines along the flow."""
    H, W = cv.shape
    sky_i = P.r('sky'); wat_i = P.r('wat')
    wm = G['kind'] == 2
    R = np.where(wm, T.roughness(G['x'], G['z']), 0)
    glare, foam, darkw = sky_i[2], sky_i[0], wat_i[1]
    mir = P.r('wmir') if 'wmir' in P.ramps else sky_i
    glare = mir[2]
    # 1 broken reflections
    refl = wm & (G['refl'] == 1) & (R > 0.2)
    for r in range(H):
        cs = np.nonzero(refl[r])[0]
        x = 0
        while x < len(cs):
            L = int(rng.integers(2, 6))
            if rng.random() < 0.35 * R[r, cs[x]]:
                cv[r, cs[x:x + L]] = mir[4]
            x += L + int(rng.integers(1, 4))

    def put(r, c, v):
        if 0 <= r < H and 0 <= c < W and wm[r, c]:
            cv[r, c] = v

    def proj(x, z):
        return int(np.floor(cam.row_of(z, 0.0))), int(np.floor(cam.col_of(x, z)))
    # 2 the broken zone (V11's grammar): the mirror gives way to darker tones -- facets tilted toward
    # us mirror higher sky and let the bed show -- laid by DENSITY: solid core, checker margin,
    # sparse spray, cut into streaks along the crests (across the flow)
    par = (np.arange(H)[:, None] + np.arange(W)[None, :]) % 2 == 0
    jets = pk.value_noise(H, W, 3, rng, 2, aniso=(0.5, 4.0))       # streaks run WITH the flow
    D = R * (0.45 + 0.9 * jets)
    solid = wm & (D > 0.78)
    chk = wm & (D > 0.45) & ~solid & par
    spr = wm & (D > 0.22) & (D <= 0.45) & par & (rng.random((H, W)) < 0.35)
    cv[solid] = np.where(rng.random((H, W)) < 0.7, wat_i[2], wat_i[1])[solid]
    cv[chk] = wat_i[2]
    cv[spr] = wat_i[3]
    # 3 the lip, then streaks along the flow. Marks follow the STREAMLINES (V11's jets run toward
    # the viewer because its river does); here the flow crosses the view, so they are horizontal:
    # dashed runs of glare with a dark run under, dense just below the lip, thinning downstream.
    r_far = int(np.ceil(cam.row_of(T.z_far, 0))); r_near = int(cam.row_of(T.z_near, 0))
    zmid = 0.5 * (T.z_near + T.z_far)
    for (xc, half, st) in T.riffles:
        # the lip: where the marks are DENSEST (not a line of dots) -- a 2-4 px band of broken foam
        # and glare following the curved lip, jittered row by row
        for rr in range(r_far + 1, r_near):
            z = cam.z_of_row(rr + 0.5)
            ppm = cam.f / z
            c0 = int(cam.col_of(T.lip_x(z, xc), z))
            wband = max(2, int(0.12 * ppm))
            for j in range(wband):
                if rng.random() < 0.7 - 0.4 * j / wband:
                    put(rr, c0 + j + int(rng.integers(-1, 2)), foam if rng.random() < 0.5 * st else glare)
        # streaks along the flow, starting at the lip
        for rr in range(r_far + 1, r_near):
            z = cam.z_of_row(rr + 0.5)
            ppm = cam.f / z
            xl = T.lip_x(z, xc)
            u = rng.exponential(half * 0.5)
            while u < 3 * half:
                a = st * np.exp(-u / half)
                if rng.random() < 0.5 + 0.5 * a:
                    L = int(np.clip(rng.uniform(0.05, 0.2) * ppm * (0.5 + a), 1, 5))
                    c = int(cam.col_of(xl + u, z))
                    white = a > 0.6 and rng.random() < 0.35
                    tone = glare if rng.random() < 0.5 + 0.4 * a else mir[3]
                    for j in range(L):
                        if j == L - 1 and rng.random() < 0.5:
                            break
                        put(rr, c + j, foam if (white and j < 2) else tone)
                    if rr + 1 < r_near and rng.random() < 0.35 + 0.3 * a:
                        off = int(rng.integers(0, 3))
                        for j in range(off, off + max(1, L - 1)):
                            put(rr + 1, c + j, darkw)
                    # now and then the dash is a short crest ARC across the flow (foam in broken
                    # curving lines, as in the Iori photo): it steps a row every 2 px
                    if a > 0.4 and rng.random() < 0.25:
                        for j in range(int(rng.integers(3, 6))):
                            put(rr - (j // 2), c + L + j, foam if (a > 0.7 and rng.random() < 0.5) else glare)
                u += rng.exponential(0.35 + (1 - a) * 1.2) * (1 if rr % 2 else 1.6)
    # 4 stones
    for (x, z, rad) in T.stones:
        dep = -T.height(np.array([x]), np.array([z]))[0]
        if dep <= 0:
            continue
        r0, c0 = proj(x, z)
        ppm = cam.f / z
        wpx = max(1, int(round(2 * rad * ppm)))
        if rad * 0.9 > dep:                                   # emergent
            hpx = max(1, int(round(wpx * 0.4)))
            for j in range(hpx):
                wj = wpx if j < hpx - 1 or hpx == 1 else max(1, wpx - 2)
                for i in range(wj):
                    put(r0 - j, c0 + i - wj // 2, wat_i[0])
            for i in range(max(1, wpx - 2)):
                put(r0 - hpx, c0 + i - wpx // 2 + 1, wat_i[3])          # sky-lit wet top
            put(r0, c0 - wpx // 2 - 1, foam)                             # pillow upstream
            if hpx > 1:
                put(r0 - 1, c0 - wpx // 2 - 1, glare)
            # wake downstream: a dashed light run leaving the stone's downstream side, thinning
            Lw = int(rng.integers(3, 7) + wpx)
            for j in range(Lw):
                if j % 3 != 2 and rng.random() < 1 - j / (Lw + 1):
                    put(r0, c0 + wpx // 2 + 1 + j, glare)
                    if j > 2 and rng.random() < 0.4:
                        put(r0 - 1, c0 + wpx // 2 + 1 + j, glare)
        elif dep < 0.14:                                      # submerged: pocket + hump dash
            for i in range(wpx):
                put(r0, c0 + i - wpx // 2, darkw)
            for i in range(max(1, wpx)):
                put(r0 - 1, c0 + i - wpx // 2 + max(1, wpx // 2), glare)
    # 5 streaks below the riffle
    for (xc, half, st) in T.riffles:
        for i in range(int(12 * st)):
            x = xc + half * rng.uniform(2.5, 4.5)
            z = rng.uniform(T.z_near + 0.2, T.z_far - 0.2)
            r, c = proj(x, z)
            L = int(rng.integers(6, 24))
            for j in range(L):
                if rng.random() < 0.85:
                    put(r, c + j, mir[3])
