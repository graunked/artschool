"""scene: plants placed in the world, painted far to near with cast shadows, under a camera.

    paint_scene(W, H, cam, season, hour, plants, ground, sun_az, sun_el, seed) -> Canvas

`plants` is a dict of densities (plants per 100 m^2) and sizes; see DEFAULT_PLANTS.
Level of detail is chosen per plant from its width in pixels (see sage_sprite / grass_sprite).
"""
import numpy as np
import sp, sage, grass, world, pine
from sp import SAGE, SAGESTEM, GRASS, GROUND, CRUST, STONE, SNOW, SKY, FLOWER, BITTER

DEFAULT_PLANTS = dict(
    sage=18.0,        # big sagebrush per 100 m^2 (Wyoming big sage steppe: 10-40)
    sage_size=(0.7, 1.3),   # width range, m
    bitter=0.0,       # bitterbrush per 100 m^2
    grass=60.0,       # bunchgrass tussocks per 100 m^2
    grass_size=(0.25, 0.45),
    pine=0.0,         # ponderosa per 100 m^2 (parkland: 0.5-1.5)
    pine_size=(14.0, 28.0),
)

SEASON_PARAMS = {
    # grass colour lives in the palette; these are shape/state changes
    "spring": dict(stalks=0.0, sage_stalks=0.0, snow=0.0, forbs=1.0),
    "summer": dict(stalks=0.5, sage_stalks=0.0, snow=0.0, forbs=0.0),
    "autumn": dict(stalks=0.35, sage_stalks=0.8, snow=0.0, forbs=0.0),
    "winter": dict(stalks=0.15, sage_stalks=0.15, snow=0.55, forbs=0.0),
}


def sun_world(az, el):
    """World sun direction (x right, y up, z toward viewer), same convention as sage.sun_vec."""
    return sage.sun_vec(az, el)


def view_light(L, cam_el):
    """Light in the sprite's view space for a camera looking down at cam_el degrees."""
    e = np.radians(cam_el)
    u = np.array([0, np.cos(e), -np.sin(e)])     # screen up in world
    v = np.array([0, np.sin(e), np.cos(e)])      # toward viewer in world
    out = np.array([L[0], L @ u, L @ v])
    return out / np.linalg.norm(out)


# ------------------------------------------------------------------ sprites by LOD
def sage_sprite(w_px, h_px, rng, L, kind="sage", season="summer", density=0.7):
    sp_ = SEASON_PARAMS[season]
    if w_px >= 5:
        p = sage.SageParams(w=w_px, h=h_px, L=L, kind=kind, density=density,
                            stalks=sp_["sage_stalks"] if kind == "sage" else 0,
                            snow=sp_["snow"] * 0.8)
        if w_px < 14:
            p.update(clump_px=max(6.0, w_px / 2.2), fingers=0.35, caps=0.06, slots=0.02,
                     flecks=0.03, tips=0.15, trunks=1)
        cv, bx, by, _ = sage.paint_bush(p, rng, pad=3)
        return cv, bx, by
    # micro: a dab
    W, H = int(w_px) + 4, int(h_px) + 4
    cv = sp.Canvas(W, H)
    bx, by = W // 2, H - 2
    mat = SAGE if kind == "sage" else BITTER
    hw = max(0.5, w_px / 2)
    sun_r = L[0] > 0
    for y in range(by - max(0, int(round(h_px)) - 1), by + 1):
        t = (by - y) / max(1, h_px)       # 0 foot .. 1 top
        half = hw * np.sqrt(max(0.05, 1 - t * t)) if h_px > 1.5 else hw
        for x in range(int(np.floor(bx - half + 0.5)), int(np.ceil(bx + half - 0.5)) + 1):
            side = (x - bx) * (1 if sun_r else -1)
            s = 1 if t < 0.34 else (3 if side >= 0 else 2)
            if t >= 0.66 and side >= 0:
                s = 4
            cv.px(x, y, mat, s)
    if w_px < 2.2:
        # the smallest sage: never vanish (a pop when the camera pulls back); a dark foot pixel
        # under a lit one, or one lit-side pixel beside it
        cv.mat[:] = 0
        cv.px(bx, by, mat, 1)
        cv.px(bx, by - 1, mat, 3)
        if w_px >= 1.5:
            cv.px(bx + (1 if sun_r else -1), by, mat, 2)
    return cv, bx, by


def pine_sprite(h_px, rng, L, age="mature", season="summer"):
    sp_ = SEASON_PARAMS[season]
    if h_px >= 9:
        p = pine.PineParams(h=h_px, age=age, L=L, snow=sp_["snow"] * 0.7,
                            crown_frac=0.45 if age == "old" else 0.66,
                            width=0.62 if age == "old" else 0.36 if age == "young" else 0.5)
        if h_px < 40:
            p.update(bristle=0.2, tufts=0.2, fill=0.9)
        cv, bx, by, _ = pine.paint_pine(p, rng, pad=3)
        return cv, bx, by
    # far: a dark spire-ish dab with a lit top on the sun side, a trunk pixel under it
    H = int(h_px) + 3
    cv = sp.Canvas(7, H + 2)
    bx, by = 3, H
    ch = max(2, int(round(h_px * 0.65)))
    for k in range(ch):
        y = by - int(h_px) + k + 1
        half = 1 if (k > ch * 0.3 and h_px > 5) else 0
        for dx in range(-half, half + 1):
            cv.px(bx + dx, y, sp.NEEDLE, 3 if (k < ch * 0.4 and dx * (1 if L[0] > 0 else -1) >= 0) else 1)
    for y in range(by - int(h_px) + ch + 1, by + 1):
        cv.px(bx, y, sp.BARK, 1)
    return cv, bx, by


def forb_sprite(w_px, rng, L):
    """Arrowleaf balsamroot (April-May): a low clump of big grey-green leaves carrying yellow
    sunflower heads on short stalks. Leaves: GRASS steps (spring green); heads: FORB."""
    W, H = int(w_px) + 6, int(w_px * 0.9) + 6
    cv = sp.Canvas(W, H)
    cx, gy = W // 2, H - 2
    rx, ry = w_px / 2, max(1.0, w_px * 0.35)
    sun_r = L[0] > 0
    for y in range(int(gy - ry), gy + 1):
        for x in range(int(cx - rx), int(cx + rx) + 1):
            u, v = (x - cx) / rx, (gy - y) / ry
            if u * u + v * v <= 1.0 and rng.random() < 0.85:
                side = u * (1 if sun_r else -1)
                cv.px(x, y, sp.GRASS, 3 if (v > 0.45 and side > -0.3) else 2 if v > 0.2 else 1)
    nh = max(1, int(w_px / 3))
    for _ in range(nh):
        hx = int(round(cx + rng.uniform(-rx * 0.8, rx * 0.8)))
        hy = int(round(gy - ry - rng.uniform(0, w_px * 0.3)))
        cv.px(hx, hy, sp.FORB, 1)
        if w_px > 8:
            cv.px(hx + 1, hy, sp.FORB, 1)
            cv.px(hx, hy + 1, sp.FORB, 0)
    return cv, cx, gy


def grass_sprite(w_px, h_px, rng, L, season="summer"):
    sp_ = SEASON_PARAMS[season]
    az = 0 if L[0] < 0 else 180
    p = grass.TussockParams(w=w_px, h=h_px, stalks=sp_["stalks"], az=az)
    cv, bx, by = grass.paint_tussock(p, rng, pad=2)
    return cv, bx, by


# ------------------------------------------------------------------ placement
def _zvis(cam, size_m, min_px=1.0):
    """Farthest distance at which a plant of this size is still >= min_px wide."""
    if cam.kind == "iso":
        return None
    return cam.f * size_m / min_px


def scatter(cam, density_per100, seed, zmax=None, xpad=2.0, cluster=0.0):
    """Poisson-ish (jittered grid) placement over the visible ground. Returns [(X,Z,r)]."""
    if density_per100 <= 0:
        return []
    cell = np.sqrt(100.0 / density_per100)
    z0, z1 = cam.zrange()
    z1 = min(z1, zmax or z1)
    out = []
    iz0, iz1 = int(np.floor(z0 / cell)) - 1, int(np.ceil(z1 / cell)) + 1
    for iz in range(iz0, iz1):
        Zc = (iz + 0.5) * cell
        if cam.kind == "iso":
            xa, xb = -xpad, cam.W / cam.ppm + xpad
        else:
            half = max(Zc, 0.5) * cam.W / cam.f / 2 + xpad
            xa, xb = -half, half
        ixs = np.arange(int(np.floor(xa / cell)), int(np.ceil(xb / cell)) + 1)
        izs = np.full_like(ixs, iz)
        jx = world._hash(ixs, izs, seed)
        jz = world._hash(ixs, izs, seed + 1)
        rr = world._hash(ixs, izs, seed + 2)
        keep = world._hash(ixs, izs, seed + 3) < 0.85
        if cluster > 0:
            # shrubs clump: keep probability follows a low-frequency field
            fx = (ixs + jx) * cell
            fz = np.full_like(fx, (iz + 0.5) * cell, dtype=float)
            fld = world.fbm(fx, fz, 4.0 + 6.0 * cluster, seed + 9)
            keep &= world._hash(ixs, izs, seed + 4) < np.clip(0.5 + (fld - 0.5) * 3.5 * cluster, 0.08, 1.0)
        for ix, a, b, r, k in zip(ixs, jx, jz, rr, keep):
            if k:
                Z = (iz + b) * cell
                if z0 * 0.8 <= Z <= z1:
                    out.append(((ix + a) * cell, Z, r))
    return out


def _cast_shadow(cv, cam, spr, ox, oy, X, Z, ppm, Lw, band):
    """Project the sprite silhouette onto the ground: ground pixels there go two steps down
    (they keep their texture: the shadow is the ground in its shadow family)."""
    if Lw[1] <= 0.05:
        return
    m = spr.mat > 0
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return
    by = ys.max()
    cz = cam.c if cam.kind == "iso" else 1.0
    # horizontal shadow direction in world (away from the sun); shadow length per metre height
    dxw, dzw = -Lw[0], Lw[2]            # toward-viewer light means shadow goes AWAY (+Z)
    k = 1.0 / max(0.12, Lw[1])
    hgt = (by - ys) / (ppm * cz)
    lat = (xs + ox - cam.project(X, Z)[0]) / ppm
    SX = X + lat + hgt * k * dxw
    SZ = Z + hgt * k * dzw
    px, py = cam.project(SX, SZ)
    px = np.round(px).astype(int)
    py = np.round(py).astype(int)
    H, W = cv.h, cv.w
    for dx in (0, 1):
        for dy in (0, 1):
            qx, qy = px + dx, py + dy
            ok = (qx >= 0) & (qx < W) & (qy >= 0) & (qy < H)
            qx, qy = qx[ok], qy[ok]
            g = np.isin(cv.mat[qy, qx], (GROUND, CRUST, STONE, SNOW))
            qx, qy = qx[g], qy[g]
            mark = np.zeros((H, W), bool)
            mark[qy, qx] = True
            sel = mark & ~_shadow_done
            cv.step[sel] = np.maximum(0, cv.step[sel] - 2)
            _shadow_done[sel] = True


_shadow_done = None


def paint_scene(W, H, cam, season="summer", hour="noon", plants=None, ground=None, sun_az=35,
                sun_el=40, seed=0, sky=True):
    global _shadow_done
    pl = dict(DEFAULT_PLANTS)
    pl.update(plants or {})
    gp = world.GroundParams(seed=seed, season=season)
    gp["snow"] = SEASON_PARAMS[season]["snow"]
    gp.update(ground or {})
    cv = sp.Canvas(W, H)
    Lw = sun_world(sun_az, sun_el)
    sun_dx = 1 if Lw[0] < 0 else -1          # shadows fall to +x when the sun is on the left
    world.paint_ground(cv, cam, gp, sun_dx)
    if cam.kind == "persp" and sky:
        _sky(cv, cam)
    if cam.kind == "persp":
        _far_sea(cv, cam, pl, seed)
        import backdrop
        bd = pl.get("backdrop", "hills")
        hy = int(cam.hy)
        if bd == "rim":
            backdrop.paint_rim(cv, hy - int(cam.H * 0.16), hy + 1, band=3, sun_dx=sun_dx, seed=seed)
        elif bd == "hills":
            backdrop.paint_far_hills(cv, hy, max(2, int(cam.H * 0.03)), band=4, seed=seed)
    L = view_light(Lw, cam.el if cam.kind == "iso" else 4.0)
    _shadow_done = np.zeros((H, W), bool)
    rng = np.random.default_rng(seed)
    items = []
    for X, Z, r in scatter(cam, pl["sage"], seed + 101, zmax=_zvis(cam, pl["sage_size"][1]),
                           cluster=pl.get("cluster", 0.5)):
        items.append(("sage", X, Z, r))
    for X, Z, r in scatter(cam, pl["bitter"], seed + 202):
        items.append(("bitter", X, Z, r))
    # grass grows in drifts (swales, lanes between shrubs), rarely against a shrub's foot
    from scipy.spatial import cKDTree
    shrubs = np.array([(X, Z) for k, X, Z, r in items]) if items else np.zeros((0, 2))
    tree = cKDTree(shrubs) if len(shrubs) else None
    g = np.array(scatter(cam, pl["grass"] * 1.6, seed + 303, zmax=_zvis(cam, pl["grass_size"][1])))
    if len(g):
        drift = world.fbm(g[:, 0], g[:, 1], 5.0, seed + 404)
        keep = drift >= 0.5 - pl.get("grass_drift", 0.12) + 0.25 * (1 - g[:, 2])
        keep |= world._hash((g[:, 0] * 100).astype(np.int64), (g[:, 1] * 100).astype(np.int64), seed) < 0.25
        if tree is not None:
            d, _ = tree.query(g[:, :2])
            keep &= d >= 0.55
        for X, Z, r in g[keep]:
            items.append(("grass", X, Z, r))
    forbs = SEASON_PARAMS[season]["forbs"] * pl.get("forb", 1.0)
    if forbs > 0:
        for X, Z, r in scatter(cam, 40.0 * forbs, seed + 606, zmax=_zvis(cam, 0.5), cluster=0.9):
            items.append(("forb", X, Z, r))
    for X, Z, r in scatter(cam, pl["pine"], seed + 505, cluster=0.8):
        items.append(("pine", X, Z, r))
    items.sort(key=lambda t: -t[2])
    shadows = []
    sprites = []
    for kind, X, Z, r in items:
        ppm = cam.ppm_at(Z)
        band = cam.band(Z)
        if kind == "pine":
            lo, hi = pl["pine_size"]
            hm = lo + (hi - lo) * r
            hpx = hm * ppm * (cam.c if cam.kind == "iso" else 1.0)
            if hpx < 3:
                continue
            srng = np.random.default_rng(int(abs(X * 1000 + Z * 7919)) % (2 ** 31) + seed)
            age = "young" if r < 0.3 else ("old" if r > 0.85 else "mature")
            spr, bx, by = pine_sprite(hpx, srng, L, age, season)
            x, y = cam.project(X, Z)
            spr.band[:] = band
            if band >= 2:
                m = spr.mat > 0
                spr.step[m] = np.where(spr.step[m] <= 2, 2, 3)
            sprites.append((Z, spr, int(round(x)) - bx, int(round(y)) - by, X, ppm, band))
            continue
        if kind == "forb":
            wpx = (0.4 + 0.5 * r) * ppm
            if wpx < 1.5:
                continue
            srng = np.random.default_rng(int(abs(X * 1000 + Z * 7919)) % (2 ** 31) + seed)
            spr, bx, by = forb_sprite(wpx, srng, L)
            x, y = cam.project(X, Z)
            spr.band[:] = band
            sprites.append((Z, spr, int(round(x)) - bx, int(round(y)) - by, X, ppm, band))
            continue
        if kind in ("sage", "bitter"):
            lo, hi = pl["sage_size"] if kind == "sage" else (1.0, 1.8)
            wm = lo + (hi - lo) * r
            hm = wm * (0.72 if kind == "sage" else 0.8)
        else:
            lo, hi = pl["grass_size"]
            wm = lo + (hi - lo) * r
            hm = wm * 1.3
        wpx = wm * ppm
        if cam.kind == "iso":
            hpx = hm * ppm * cam.c + wm * 0.5 * ppm * cam.s
        else:
            hpx = hm * ppm
        if wpx < 1.0:
            continue
        srng = np.random.default_rng(int(abs(X * 1000 + Z * 7919)) % (2 ** 31) + seed)
        if kind == "grass" and hpx < 5.5:
            # small tussocks merge into the ground's tone: most vanish into it, the rest are a
            # single soft lit fleck (no dark foot, no shadow). Salt-and-pepper otherwise.
            if srng.random() > np.clip((hpx - 1.5) / 3.5, 0.2, 1.0):
                continue
            spr = sp.Canvas(3, 3)
            spr.px(1, 2, GRASS, 3)
            if hpx >= 3:
                spr.px(1, 1, GRASS, 4)
            x, y = cam.project(X, Z)
            spr.band[:] = band
            sprites.append((Z, spr, int(round(x)) - 1, int(round(y)) - 2, X, ppm, band))
            continue
        if kind == "grass":
            spr, bx, by = grass_sprite(wpx, hpx, srng, L, season)
        else:
            spr, bx, by = sage_sprite(wpx, hpx, srng, L, kind, season)
            if kind == "bitter":
                spr.mat[spr.mat == SAGE] = BITTER
        x, y = cam.project(X, Z)
        ox, oy = int(round(x)) - bx, int(round(y)) - by
        spr.band[:] = band
        if band >= 2:
            # distance: the marks compress to two close values; no darkest index (Ferrari lifts
            # the black point with distance, the palette's haze does the colour)
            m = spr.mat > 0
            spr.step[m] = np.where(spr.step[m] <= 2, 2, 3)
        sprites.append((Z, spr, ox, oy, X, ppm, band))
    # shadows first (on the ground), then sprites far to near
    for Z, spr, ox, oy, X, ppm, band in sprites:
        if spr.w > 3 and hour != "overcast":
            _cast_shadow(cv, cam, spr, ox, oy, X, Z, ppm, Lw, band)
    if gp["snow"] > 0:
        _snow_drifts(cv, sprites, Lw, gp["snow"], seed)
    for Z, spr, ox, oy, X, ppm, band in sorted(sprites, key=lambda t: -t[0]):
        # contact: darken the ground right under the foot
        cv.blit(spr, ox, oy)
        m = spr.mat > 0
        if m.any() and spr.w > 6 and (spr.mat[m] != GRASS).mean() > 0.5:   # shrubs and trees only
            ys, xs = np.nonzero(m)
            by = ys.max()
            for xx in np.unique(xs[ys >= by - 1]):
                gx, gy = xx + ox, by + oy + 1
                if 0 <= gx < W and 0 <= gy < H and cv.mat[gy, gx] in (GROUND, CRUST, STONE):
                    cv.step[gy, gx] = 0
    return cv


def _snow_drifts(cv, sprites, Lw, amount, seed):
    """Snow wraps the shrubs: it drifts against the lee (shaded) side of each plant's foot and
    melts back on its sunny side, so patch edges follow the plants, not a ruled line."""
    rng = np.random.default_rng(seed + 77)
    lee = 1 if Lw[0] < 0 else -1            # the side away from the sun
    for Z, spr, ox, oy, X, ppm, band in sprites:
        m = spr.mat > 0
        if not m.any() or spr.w < 6 or (spr.mat[m] == GRASS).mean() > 0.5:
            continue
        ys, xs = np.nonzero(m)
        by, x0, x1 = ys.max() + oy, xs.min() + ox, xs.max() + ox
        wid = x1 - x0 + 1
        # lee drift: an irregular lens hugging the foot on the shaded side
        cxd = (x0 + x1) / 2 + lee * wid * 0.35
        rx, ry = wid * (0.45 + 0.3 * amount), max(1.0, wid * 0.10)
        for y in range(int(by - ry), int(by + ry * 1.2) + 1):
            for x in range(int(cxd - rx), int(cxd + rx) + 1):
                if 0 <= x < cv.w and 0 <= y < cv.h and cv.mat[y, x] in (GROUND, CRUST, STONE, SNOW):
                    u, v = (x - cxd) / rx, (y - by) / ry
                    if u * u + v * v < 1 + rng.normal(0, 0.15):
                        cv.mat[y, x] = SNOW
                        cv.step[y, x] = 1 if v < 0 else 2
        # sunny side melt-back: bare ground
        cxs = (x0 + x1) / 2 - lee * wid * 0.45
        for y in range(int(by - 1), int(by + ry) + 1):
            for x in range(int(cxs - wid * 0.3), int(cxs + wid * 0.3) + 1):
                if 0 <= x < cv.w and 0 <= y < cv.h and cv.mat[y, x] == SNOW and rng.random() < 0.8:
                    cv.mat[y, x] = GROUND
                    cv.step[y, x] = 3


def _far_sea(cv, cam, pl, seed):
    """Beyond the distance where a sage is ~2 px wide, the shrubs merge: the ground carries them
    as horizontal flecks (foreshortened), their cover modulated by a big swale field so the far
    flat has value masses (a darker draw, a paler flat) instead of one even band."""
    H, W = cv.h, cv.w
    ys, xs = np.mgrid[0:H, 0:W].astype(float)
    X, Z = cam.ground_xz(xs + 0.5, ys + 0.5)
    zlim = cam.f * pl["sage_size"][0] / 2.5
    far = (ys > cam.hy + 0.5) & (Z > zlim)
    cover = min(0.9, pl["sage"] / 40.0)
    swale = world.fbm(X, Z, 60.0, seed + 55)
    local = np.clip(cover + (swale - 0.5) * 1.2, 0.02, 0.95)
    # flecks: 2-3 px horizontal dashes at mid distance, single px farther
    dash = world.vnoise(X * 0.35, Z, np.maximum(0.3, Z / cam.f * 1.0), seed + 56)
    fleck = far & (dash < local)
    band = np.searchsorted(cam.bands, Z).astype(np.uint8)
    cv.mat[fleck] = sp.SAGE
    cv.step[fleck] = np.where(world.vnoise(X, Z, np.maximum(0.3, Z / cam.f * 1.5), seed + 57)[fleck] > 0.55, 3, 2)
    cv.band[fleck] = band[fleck]


def _sky(cv, cam):
    hy = int(cam.hy)
    for y in range(0, hy + 1):
        t = y / max(1, hy)
        s = t * 5.0
        f = np.floor(s)
        row = np.full(cv.w, f)
        frac = s - f
        # dither only at the seams between bands
        if frac > 0.75:
            row[(np.arange(cv.w) + y) % 2 == 0] = f + 1
        cv.mat[y] = SKY
        cv.step[y] = np.clip(row, 0, 5).astype(np.int16)
