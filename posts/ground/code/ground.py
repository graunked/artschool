"""ground.py -- the ground as big designed value masses, with texture living inside them.

    g = paint_ground(W, H, ground='turf', landform='rolling', hour='golden', sun=None,
                     moisture=0.2, ppm=24, pitch=14, cam_h=1.7, canopy=0.0, seed=0)
    g.rgb      (H, W, 3) uint8
    g.mat, g.band, g.step   the index canvas (material, depth band, ramp step) -- palette-swappable
    g.depth, g.wx, g.wz, g.h   per-pixel world buffers, so other students can stand objects on it
    g.relight(hour)         new rgb from the same index canvas

Layers (each a separate function so it can be tuned against a reference):
  1 LANDFORM   analytic height h(x,z) in metres built from a few big features
               (rolling swells, hummocks, bank, ledges, path) -- the masses come from here
  2 RENDER     voxel-space column march: per pixel world x,z,h,normal,depth,px-per-metre
  3 LIGHT      sun visibility (terrain horizon shadow x off-frame canopy shadow) and N.L,
               sky ambient, hollow occlusion; *smoothed at mass scale* so light makes shapes
  4 VALUE PLAN two families that never touch: shadow steps 0..2, light steps 3..5; the
               terminator is a jump. crest separation: land just behind a crest drops a step.
  5 MATERIAL   turf / meadow / earth / gravel / forest floor; path and bare risers; moisture
  6 MARKS      offsets in step space whose size follows px-per-metre and whose contrast follows
               the family and distance: strands -> ticks -> DPaint dither -> flat
  7 QUANTIZE   ordered 2x2 + jitter (the V16 islet recipe), adjacent steps only
  8 PALETTE    per material x depth band x hour ramp, generated from albedo x light colour
"""
import numpy as np
from scipy import ndimage as ndi
import dither

# ------------------------------------------------------------------ 1 LANDFORM
class Landform:
    """h(x, z) as a sum of big features. Everything analytic, so it can be sampled anywhere."""
    def __init__(self, kind='rolling', seed=0, relief=1.0, extent=(-200, 200, 0, 400), path=False,
                 water=None):
        self.kind, self.seed, self.relief = kind, seed, relief
        r = np.random.default_rng(seed)
        self.r = r
        # low-frequency swells: random-direction sinusoids (smooth, big)
        n = 7
        # swells mostly run across the view, so the land recedes in layered crests (V13 terraces)
        ang = np.pi / 2 + r.normal(0, 0.45, n) if kind in ('rolling', 'bank') else r.uniform(0, np.pi, n)
        wl = r.uniform(18, 60, n)
        self.sw = [(np.cos(a) * 2 * np.pi / l, np.sin(a) * 2 * np.pi / l, r.uniform(0, 6.3), l / 60.0)
                   for a, l in zip(ang, wl)]
        # micro relief (only drives marks/light breakup, small amplitude)
        m = 9
        ang = r.uniform(0, np.pi, m); wl = r.uniform(1.5, 6, m)
        self.micro = [(np.cos(a) * 2 * np.pi / l, np.sin(a) * 2 * np.pi / l, r.uniform(0, 6.3), l / 6.0)
                      for a, l in zip(ang, wl)]
        # hummocks: lozenges
        self.hum = []
        if kind in ('hummocks',):
            # V16's far bank: separate lozenges, wide and flat-bottomed, long axis across the
            # view, with gaps of flat ground between them (not an overlapping bumpy mush)
            tries = 0
            while len(self.hum) < 40 and tries < 2000:
                tries += 1
                cz = 8 * (140 / 8) ** r.random(); cx = r.uniform(-0.8, 0.8) * (10 + cz * 0.7)
                ax = r.uniform(3, 10) * (1 + cz / 80); az = ax * r.uniform(0.3, 0.55)
                if any(np.hypot((cx - q[0]) / (ax + q[2]), (cz - q[1]) / (az + q[3])) < 0.8 for q in self.hum):
                    continue
                self.hum.append((cx, cz, ax, az, r.uniform(0.7, 1.6) * (1 + cz / 100), r.uniform(-0.25, 0.25)))
        self.path = path
        self.path_a, self.path_b, self.path_c = r.uniform(2, 5), r.uniform(12, 25), r.uniform(-1.5, 1.5)
        self.water = water

    def path_center(self, z):
        return self.path_c + self.path_a * np.sin(z / self.path_b + 1.3) + 0.02 * z

    def path_dist(self, x, z):
        return np.abs(x - self.path_center(z))

    def h(self, x, z, micro=True):
        k = self.kind
        H = np.zeros(np.broadcast(x, z).shape)
        amp = {'flat': 0.25, 'rolling': 1.6, 'hummocks': 0.12, 'bank': 0.6, 'ledges': 0.4}[k] * self.relief
        for kx, kz, ph, a in self.sw:
            H = H + amp * a * np.sin(kx * x + kz * z + ph)
        if k == 'rolling':
            # rolling land swells more with distance (terraces of V13)
            pass
        if k in ('hummocks', 'bank'):
            for cx, cz, ax, az, ht, rot in self.hum:
                c, s = np.cos(rot), np.sin(rot)
                u = ((x - cx) * c + (z - cz) * s) / ax
                v = (-(x - cx) * s + (z - cz) * c) / az
                q = u * u + v * v
                H = H + ht * self.relief * np.clip(1 - q, 0, None) ** 0.6   # domed top, steep flanks
        if k == 'bank':
            # the land falls toward a shoreline at z = zw (the viewer stands on a bank)
            zw = self.water if self.water is not None else 14.0
            H = H - 1.8 * self.relief * (1 / (1 + np.exp(-(z - zw) / 2.5)))
        if k == 'ledges':
            # a slope rising away from the viewer, cut into terraces with risers facing us
            base = 0.18 * z
            step = 2.4 * self.relief
            q = base / step
            fr = q - np.floor(q)
            riser = 1 / (1 + np.exp(-(fr - 0.82) * 22))
            H = H + step * (np.floor(q) + riser) + 0.4 * np.sin(x / 9 + np.floor(q) * 2.1)
        if micro:
            for kx, kz, ph, a in self.micro:
                H = H + 0.05 * a * np.sin(kx * x + kz * z + ph)
        if self.path:
            d = self.path_dist(x, z)
            H = H - 0.12 * np.exp(-(d / 1.0) ** 2)
        return H

    def normal(self, x, z, e=0.15, micro=True):
        hx = (self.h(x + e, z, micro) - self.h(x - e, z, micro)) / (2 * e)
        hz = (self.h(x, z + e, micro) - self.h(x, z - e, micro)) / (2 * e)
        n = np.stack([-hx, np.ones_like(hx), -hz], -1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)


# ------------------------------------------------------------------ 2 RENDER
def render(land, W, H, ppm=24, pitch=14, cam_h=1.7, steps=900):
    """Column march. ppm = pixels per metre at the ground point under the screen centre."""
    p = np.radians(pitch)
    zc0 = cam_h / np.sin(p)                    # camera-space depth of the centre ground point
    F = ppm * zc0
    cx, cy = W / 2, H / 2
    cols = np.arange(W) + 0.5
    rows = np.arange(H)[:, None] + 0.5
    # nearest ground: ray through the bottom row
    ang_bot = p + np.arctan((H - cy) / F)
    z_near = max(0.05, cam_h / np.tan(min(ang_bot, 1.5)) * 0.7)
    ang_top = p - np.arctan(cy / F)
    z_far = 2000.0 if ang_top <= 0.002 else min(2000.0, cam_h / np.tan(ang_top) * 3 + 50)
    zs = z_near * (z_far / z_near) ** (np.arange(steps) / (steps - 1))
    # the camera stands cam_h above the ground under it (and above the next few metres ahead)
    zz = np.linspace(0, 4, 9)
    base = float(max(land.h(np.zeros(1), np.full(1, q), micro=False)[0] for q in zz))
    Z = np.full((H, W), np.inf); X = np.zeros((H, W))
    ymin = np.full(W, H + 0.0)
    for z in zs:
        zc_est = z * np.cos(p) + cam_h * np.sin(p)
        x = (cols - cx) * zc_est / F
        h = land.h(x, np.full_like(x, z))
        y = h - cam_h - base
        zc = -y * np.sin(p) + z * np.cos(p)
        yc = y * np.cos(p) + z * np.sin(p)
        row = cy - F * yc / np.maximum(zc, 1e-3)
        m = (rows >= row[None, :]) & (rows < ymin[None, :])
        if m.any():
            Z = np.where(m, z, Z)
            X = np.where(m, x[None, :], X)
            ymin = np.minimum(ymin, row)
    sky = ~np.isfinite(Z)
    Zs = np.where(sky, z_far, Z)
    buf = dict(z=Zs, x=X, sky=sky, F=F, pitch=p, cam_h=cam_h, cam_y=base + cam_h)
    zc = Zs * np.cos(p) + cam_h * np.sin(p)
    buf['ppm'] = F / zc                         # local pixels per metre
    buf['h'] = land.h(X, Zs)
    buf['n'] = land.normal(X, Zs)
    buf['n_macro'] = land.normal(X, Zs, e=0.8, micro=False)
    hm = land.h(X, Zs, micro=False)
    ring = np.mean([land.h(X + 3 * np.cos(a), Zs + 3 * np.sin(a), micro=False) for a in np.linspace(0, 6.28, 8, endpoint=False)], 0)
    buf['convex'] = np.where(sky, 0.0, (hm - ring) / 0.6)
    return buf


# ------------------------------------------------------------------ 3 LIGHT
HOURS = {
    #          sun az, el   sun colour         sky/shade colour      haze colour
    'noon':    (35, 58, (1.00, 0.97, 0.88), (0.42, 0.55, 0.80), (0.70, 0.80, 0.92)),
    'morning': (-30, 24, (1.00, 0.90, 0.72), (0.40, 0.50, 0.75), (0.78, 0.80, 0.88)),
    'golden':  (20, 14, (1.00, 0.60, 0.38), (0.36, 0.40, 0.66), (0.90, 0.72, 0.58)),
    'dusk':    (75, 4, (0.95, 0.52, 0.40), (0.16, 0.20, 0.55), (0.45, 0.38, 0.75)),
    'overcast': (40, 70, (0.62, 0.64, 0.66), (0.55, 0.58, 0.63), (0.74, 0.76, 0.78)),
}

def sun_vec(az, el):
    """az: 0 = sun on the left of the frame, 90 = behind the scene (contre-jour), -90 = behind camera."""
    a, e = np.radians(az), np.radians(el)
    return np.array([-np.cos(a) * np.cos(e), np.sin(e), np.sin(a) * np.cos(e)])

def canopy_field(x, z, cover, scale, sunv, seed, bias=None):
    """Shadows of off-frame trees: soft union of crown shadows, stretched along the sun's azimuth
    (longer when the sun is low). Returns sun visibility 0..1 (continuous; the dither makes edges)."""
    if cover <= 0:
        return np.ones_like(x)
    r = np.random.default_rng(seed + 991)
    x0, x1 = np.percentile(x, 1) - 3 * scale, np.percentile(x, 99) + 3 * scale
    z0, z1 = np.percentile(z, 1) - 3 * scale, np.percentile(z, 97) + 3 * scale
    area = (x1 - x0) * (z1 - z0)
    n = int(np.clip(area / (np.pi * scale ** 2) * 1.2, 8, 700))
    cxs = r.uniform(x0, x1, n); czs = r.uniform(z0, z1, n)
    rad = scale * r.uniform(0.5, 1.4, n)
    sx, sz = sunv[0], sunv[2]
    hl = np.hypot(sx, sz) + 1e-6
    ux, uz = sx / hl, sz / hl                        # horizontal sun direction
    stretch = np.clip(1 / np.tan(np.arcsin(np.clip(sunv[1], 0.05, 1))), 1, 4)
    acc = np.zeros_like(x)
    for cx, cz, rr in zip(cxs, czs, rad):
        dx, dz = x - cx, z - cz
        if np.abs(dz).min() > rr * 5 and np.abs(dx).min() > rr * 5:
            continue
        a = (dx * ux + dz * uz) / (rr * stretch)
        b = (-dx * uz + dz * ux) / rr
        acc = acc + np.exp(-(a * a + b * b) * 1.5)
    # world-anchored ragged edge: a few octaves of sinusoids at crown-to-leaf scale, so shadow
    # shapes are lobed and notched (foliage), never a clean ellipse
    rr_ = np.random.default_rng(seed + 5)
    amp = acc.std() + 1e-6
    for k, wl in enumerate([scale * 1.2, scale * 0.55, scale * 0.25, scale * 0.12]):
        for _ in range(3):
            a0 = rr_.uniform(0, np.pi); ph = rr_.uniform(0, 6.3)
            acc = acc + amp * 0.25 * 0.7 ** k * np.sin((np.cos(a0) * x + np.sin(a0) * z) * 2 * np.pi / wl + ph)
    if bias is not None:
        acc = acc + bias * amp * 2.5          # the composition plan: where shadow is wanted
    # threshold so that 'cover' of the ground is in shade; soft band gives dither in the edge
    thr = np.quantile(acc, 1 - cover) if cover < 1 else acc.min() - 1
    return 1 - 1 / (1 + np.exp(-(acc - thr) / (0.03 * amp)))

def design_masses(f, min_frac=0.02):
    """Designed masses: no shadow island and no lit hole smaller than min_frac of the frame.
    Ferrari's shadow shapes are a few big ones; small isolated ovals read as holes in the ground.
    An absorbed island takes its soft penumbra with it (else a ring of edge is left behind)."""
    f = f.copy()
    a = f.size * min_frac
    for val in (True, False):
        m = f < 0.5
        lab, n = ndi.label(m == val)
        if n == 0: continue
        sizes = ndi.sum(np.ones_like(f), lab, range(1, n + 1))
        edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        floating = ~np.isin(np.arange(1, n + 1), edge)
        # a shadow floating free of the frame reads as a hole unless it is big
        lim = np.where(floating & val, a * 3.5, a)
        small = np.isin(lab, np.nonzero(sizes < lim)[0] + 1)
        if not small.any(): continue
        grow = ndi.binary_dilation(small, iterations=4)
        if val:     # shadow island -> full light, including its soft rim
            f[grow & (f < 0.999)] = np.maximum(f[grow & (f < 0.999)], 1.0)
        else:       # small lit hole -> full shadow
            f[grow & (f > 0.001)] = 0.0
    return f

def light(land, buf, az, el, canopy=0.0, canopy_scale=3.0, seed=0, shade=0.0, shade_scale=12.0, bias=None):
    s = sun_vec(az, el)
    n = buf['n']; nm = buf['n_macro']
    X, Z, Hh = buf['x'], buf['z'], buf['h']
    ndl_macro = np.clip((nm * s).sum(-1), 0, 1)
    ndl_micro = np.clip((n * s).sum(-1), 0, 1)
    # terrain horizon shadow: march toward the sun
    vis = np.ones_like(X)
    hor = np.hypot(s[0], s[2]) + 1e-6
    tan_el = s[1] / hor
    for t in [0.5, 1, 2, 3.5, 5.5, 8, 12, 18, 26, 38, 55, 80]:
        xs = X + s[0] / hor * t; zs = Z + s[2] / hor * t
        hs = land.h(xs, zs, micro=False)
        occl = hs - (Hh + tan_el * t)
        vis = np.minimum(vis, np.clip(0.5 - occl / (0.08 + 0.02 * t), 0, 1))
    can = canopy_field(X, Z, canopy, canopy_scale, s, seed)
    sh = canopy_field(X, Z, shade, shade_scale, s, seed + 17, bias)
    # designed masses: terrain and big cast shadows together, small islands absorbed;
    # the dapple of a near canopy is added after (it is meant to be small)
    big = design_masses(vis * sh, min_frac=0.03)
    sunvis = big * can * (1.0 if el > 0 else 0.0)
    # hollow occlusion: height below the local mean (large-scale) -> less sky
    hl = land.h(X, Z, micro=False)
    ring = np.mean([land.h(X + dx, Z + dz, micro=False) for dx, dz in
                    [(3, 0), (-3, 0), (0, 3), (0, -3), (2, 2), (-2, 2), (2, -2), (-2, -2)]], 0)
    hollow = np.clip((ring - hl) / 0.5, -1, 1)
    # skylight comes mostly from the bright side of the sky (the sun's side): surfaces facing
    # the glow are the 'light' of the shadow family -- at dusk this is what shapes the land (V13)
    hs = np.array([s[0], 0, s[2]]); hs = hs / (np.linalg.norm(hs) + 1e-6)
    skyv = np.array([0, 1, 0]) + 0.9 * hs; skyv = skyv / np.linalg.norm(skyv)
    nm2 = buf['n_macro']
    # exaggerated (Ferrari turns the land more than nature): relative to a flat plane
    ns = (nm2 * skyv).sum(-1); n0 = skyv[1]
    sky = np.clip(0.55 + 0.45 * np.tanh(6 * (ns - n0)) - 0.3 * hollow, 0, 1)
    return dict(ndl=ndl_macro, ndl_micro=ndl_micro, sunvis=sunvis, terrain_vis=vis, canopy=can,
                sky=sky, hollow=hollow, s=s)


# ------------------------------------------------------------------ 4 VALUE PLAN
NRESP = {'turf': 0.6, 'meadow': 0.5, 'earth': 1.0, 'gravel': 0.8, 'forest': 0.6, 'moss': 0.5, 'water': 0.0}

def value_plan(buf, L, mat=None, sep=0.9, sheen=0.5, lit_floor=0.10, lit_base=0.6, overcast=False):
    """lit_base: where a flat lit plane sits inside the light family (0..1).
    Grass is a forest of vertical blades: it catches a low sun whatever the ground slope, so
    its N.L response is weak (NRESP) -- turf masses come from cast shadow, not from shading.
    Bare earth, sand and snow take the full N.L (the dunes of V25)."""
    """Continuous step coordinate v in [0, 5.99]: shadow family 0..2.99, light family 3..5.99."""
    direct = L['sunvis'] * L['ndl']
    # a flat plane at this sun takes N.L = s_y; normalize so flat ground sits mid-light family
    sy = max(L['s'][1], 0.06)
    rel = np.clip(direct / max(sy * 1.35, 0.12), 0, 1.25)
    # the terminator of a cast shadow is broken by the material: blade tips cross it on turf,
    # grains on earth -- never a clean cut-paper curve
    Hh, W = L['sunvis'].shape
    rng = np.random.default_rng(12345)
    en = np.zeros((Hh, W))
    if mat is not None:
        nt = ndi.gaussian_filter(rng.normal(size=(Hh, W)), (4, 0.7)); nt /= nt.std() + 1e-9
        ne = ndi.gaussian_filter(rng.normal(size=(Hh, W)), (1.3, 1.3)); ne /= ne.std() + 1e-9
        grass = np.isin(mat, [MI['turf'], MI['meadow'], MI['moss']])
        en = np.where(grass, 0.12 * nt, 0.08 * ne) + 0.04 * rng.normal(size=(Hh, W))
    sv = L['sunvis'] + en * (L['sunvis'] * (1 - L['sunvis']) * 4 + 0.05)
    lit = (sv > 0.5) & (L['ndl'] + en * sy * 1.5 > lit_floor * sy)   # the form terminator is broken too
    lit_ndl_floor = lit_floor * sy
    # light family: 3.0 .. 5.99 ; strongly turned toward the sun -> top step
    r = L['ndl'] * L['sunvis'] / sy                 # 1 = a flat plane in full sun
    k = np.ones_like(r) if mat is None else np.vectorize(lambda m: NRESP[MATS[m]] if m >= 0 else 1.0)(mat)
    r = 1 + k * (np.minimum(r, 3) - 1)
    # exaggerated like the skylight: Ferrari turns land more than nature (tanh around flat)
    v_lit = 3.0 + 3.0 * np.clip(lit_base + 0.45 * np.tanh(2.5 * (r - 1)), 0, 0.999)
    # skylight also turns the lit family a little (faces toward the bright sky read warmer-lighter)
    v_lit = v_lit + 0.5 * (L['sky'] - 0.55)
    # shadow family: 0 .. 2.99 from sky ambient; faces turned to the sun but shadowed (cast)
    # keep a little reflected light (they are the 'lit' plane in shadow) -> v up to 2.7
    v_sh = 0.5 + 1.8 * L['sky'] ** 1.2 + 0.7 * np.clip(L['ndl'] / sy, 0, 1.3) * (1 - L['sunvis'])
    v_sh = np.clip(v_sh, 0, 2.6)
    v = np.where(lit, v_lit, v_sh)
    # the soft canopy edge: blend across the family boundary only inside the penumbra
    pen = np.zeros_like(lit)
    t = np.clip((L['sunvis'] - 0.15) / 0.7, 0, 1)
    v = np.where(pen, (1 - t) * v_sh + t * v_lit, v)
    if overcast:
        # no cast shadows, no terminator: one soft family across the whole ramp, shaped by the sky
        v = 0.9 + 3.0 * L['sky'] ** 1.2 + 1.6 * np.clip(np.log(buf['z'] / 4) / 4, 0, 1)
        lit = v >= 3.0
    # grazing sheen: ground brightens as it recedes (Ferrari's receding planes, V16 banks)
    d = buf['z']
    low = np.clip((12 - np.degrees(np.arcsin(sy))) / 10, 0, 1)      # 1 when the sun is near the horizon
    v = v + sheen * np.clip(np.log(d / 3) / 4, 0, 1) * np.where(v >= 3, 1.0, 0.5 + 0.5 * low)
    # crest separation: land just behind (above on screen) a crest silhouette drops a step
    Hh, W = v.shape
    z = buf['z']
    jump = np.zeros_like(v, bool)
    jump[:-1] = (z[:-1] > z[1:] * 1.25) & (z[:-1] - z[1:] > 2.0) & ~buf['sky'][:-1]
    k = 0
    sepm = np.zeros_like(v)
    cur = jump.copy()
    for k in range(1, 7):
        sepm = np.maximum(sepm, cur * (1 - k / 7))
        cur = np.roll(cur, -1, 0); cur[-1] = False
    v = v - sep * sepm
    return np.clip(v, 0, 5.99), lit, sepm


# ------------------------------------------------------------------ 5 MATERIAL
MATS = ['turf', 'meadow', 'earth', 'gravel', 'forest', 'moss', 'water']
MI = {m: i for i, m in enumerate(MATS)}

def materials(land, buf, ground, moisture, L, seed=0):
    X, Z = buf['x'], buf['z']
    mat = np.full(X.shape, MI[ground], int)
    slope = 1 - buf['n_macro'][..., 1]
    wet = np.clip(moisture + 0.6 * np.clip(L['hollow'], 0, 1) - 0.15, 0, 1)
    if ground in ('turf', 'meadow'):
        if land.kind in ('ledges', 'rolling'):
            mat[slope > 0.22] = MI['earth']      # bare risers and cut banks (turf hummocks stay turf)
    if land.path:
        pd = land.path_dist(X, Z)
        # ragged edge: the path's width breathes and its border is broken by the grass
        r = np.random.default_rng(seed + 5)
        edge = 0.75 + 0.15 * np.sin(Z * 1.7 + 1) + 0.1 * np.sin(Z * 4.3 + X)
        mat[pd < edge] = MI['earth']
    if ground == 'forest':
        mat[(wet > 0.45) & (L['canopy'] < 0.7)] = MI['moss']
        mat[(wet > 0.6)] = MI['moss']
    # puddles: earth in hollows when wet
    # puddles: small, lobed pools where bare earth dips (fine-scale hollows + world noise),
    # growing with moisture; never a band across the valley
    X, Z = buf['x'], buf['z']
    hf = land.h(X, Z)
    ringf = np.mean([land.h(X + dx, Z + dz) for dx, dz in [(0.7, 0), (-0.7, 0), (0, 0.7), (0, -0.7)]], 0)
    rr = np.random.default_rng(seed + 77)
    wn = sum(np.sin((np.cos(a) * X + np.sin(a) * Z) * 2 * np.pi / wl + p)
             for a, wl, p in zip(rr.uniform(0, np.pi, 6), rr.uniform(0.6, 2.5, 6), rr.uniform(0, 6.3, 6))) / 2.5
    pscore = 30 * (ringf - hf) + 0.5 * wn + 1.2 * np.clip(L['hollow'], 0, 1) + 1.6 * moisture
    puddle = (mat == MI['earth']) & (pscore > 2.4) & (moisture > 0.25)
    mat[puddle] = MI['water']
    # the wet rim: mud darkens around every pool
    rim = ndi.binary_dilation(puddle, iterations=2) & ~puddle & (mat == MI['earth'])
    wet = np.where(rim, 1.0, wet)
    if land.water is not None:
        mat[buf['h'] < -1.2] = MI['water']
    return mat, wet


# ------------------------------------------------------------------ 6 MARKS
def smooth_noise(Hh, W, sy, sx, rng):
    """Gaussian-filtered white noise, unit std, correlation lengths sy, sx (px)."""
    n = ndi.gaussian_filter(rng.normal(size=(Hh, W)), (sy, sx), mode='wrap')
    return (n - n.mean()) / (n.std() + 1e-9)

def runs_field(Hh, W, runlen, rng, vertical=True, amp=None):
    """Random offsets constant along runs (strands if vertical, streaks if horizontal).
    runlen: per-pixel mean run length (px). Returns offsets ~ N(0,1) per run."""
    out = np.zeros((Hh, W)); thr = np.zeros((Hh, W))
    if not vertical:
        o, t = runs_field(W, Hh, runlen.T, rng, True)
        return o.T, t.T
    for x in range(W):
        y = Hh - 1
        col = runlen[:, x]
        while y >= 0:
            L = max(1, int(round(col[y] * rng.uniform(0.45, 1.4))))
            out[max(0, y - L + 1): y + 1, x] = rng.normal()
            thr[max(0, y - L + 1): y + 1, x] = rng.random()
            y -= L
    return out, thr

def stones_field(buf, size_m, rng, density=0.7):
    """Gravel / litter: small blobs with a light top row and a dark lower rim (offsets)."""
    Hh, W = buf['z'].shape
    out = np.zeros((Hh, W))
    ppm = buf['ppm']
    n = int(Hh * W * density / 6)
    ys = rng.integers(0, Hh, n); xs = rng.integers(0, W, n)
    order = np.argsort(ys)
    for i in order:
        y, x = ys[i], xs[i]
        s = size_m * ppm[y, x] * rng.uniform(0.6, 1.5)
        if s < 1.6:
            out[y, x] += rng.choice([-1, 1]) * 0.8
            continue
        a = max(1, int(round(s / 2))); b = max(1, int(round(s / 3.2)))
        y0, y1 = max(0, y - b), min(Hh, y + b + 1)
        x0, x1 = max(0, x - a), min(W, x + a + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        q = ((xx - x) / (a + 0.5)) ** 2 + ((yy - y) / (b + 0.5)) ** 2
        m = q < 1
        tone = rng.normal(0.3, 0.5)
        top = (yy - y) < -b * 0.2
        bot = (yy - y) > b * 0.4
        blk = np.where(top, tone + 0.8, np.where(bot, tone - 0.5, tone))
        sub = out[y0:y1, x0:x1]
        sub[m] = blk[m]
        # pocket under the stone
        yb = min(Hh - 1, y + b + 1)
        out[yb, x0:x1][np.abs(np.arange(x0, x1) - x) < a * 0.8] = -1.3
    return out

def leaves_field(buf, size_m, rng, density=1.0):
    Hh, W = buf['z'].shape
    out = np.zeros((Hh, W)); ppm = buf['ppm']
    n = int(Hh * W * density / 8)
    ys = rng.integers(0, Hh, n); xs = rng.integers(0, W, n)
    for i in np.argsort(ys):                      # far to near: nearer leaves overlap
        y, x = ys[i], xs[i]
        s = size_m * ppm[y, x] * rng.uniform(0.6, 1.4)
        if s < 1.5:
            out[y, x] = rng.choice([-1.0, 0.8]); continue
        a = max(1, int(round(s / 2))); b = max(1, int(round(s / 7)))
        tilt = rng.normal(0, 0.15)
        y0, y1 = max(0, y - b - 1), min(Hh, y + b + 2); x0, x1 = max(0, x - a - 1), min(W, x + a + 2)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        yc = y + tilt * (xx - x)
        q = ((xx - x) / (a + 0.5)) ** 2 + ((yy - yc) / (b + 0.5)) ** 2
        m = q < 1
        tone = rng.normal(0.2, 0.4)
        blk = np.where(yy - yc < -0.3 * b, tone + 0.9, np.where(yy - yc > 0.3 * b, tone - 0.6, tone + 0.2))
        out[y0:y1, x0:x1][m] = blk[m]
        yb = int(round(y + b + 1))
        if yb < Hh:
            xs_ = np.arange(max(0, x - a + 1), min(W, x + a))
            out[yb, xs_] = np.minimum(out[yb, xs_], -1.2)   # the dark gap under each leaf
    return out

def lic(noise, phi, length, steps=None):
    """Line-integral convolution: smear noise along the iso-lines of phi (screen space).
    Streaks then follow the form -- a path's length, a slope's contours, a dune's crest."""
    Hh, W = noise.shape
    gy, gx = np.gradient(phi)
    tx, ty = -gy, gx
    nrm = np.hypot(tx, ty) + 1e-9
    tx, ty = tx / nrm, ty / nrm
    tx = np.where(tx < 0, -tx, tx); ty = np.where(tx == 0, np.abs(ty), np.where(tx < 0, -ty, ty))
    L = int(np.ceil(np.max(length))) if steps is None else steps
    acc = noise.copy(); cnt = np.ones_like(noise)
    for sgn in (1, -1):
        py, px = np.mgrid[0:Hh, 0:W].astype(float)
        for k in range(1, L + 1):
            iy = np.clip(np.round(py).astype(int), 0, Hh - 1); ix = np.clip(np.round(px).astype(int), 0, W - 1)
            px = px + sgn * tx[iy, ix]; py = py + sgn * ty[iy, ix]
            iy = np.clip(np.round(py).astype(int), 0, Hh - 1); ix = np.clip(np.round(px).astype(int), 0, W - 1)
            w = (k <= length)
            acc += noise[iy, ix] * w; cnt += w
    o = acc / cnt
    return (o - o.mean()) / (o.std() + 1e-9)

def marks(buf, mat, v, lit, rng, contrast=1.0, phi=None):
    """Offsets in step space. Size follows px-per-metre; contrast follows family & distance."""
    Hh, W = v.shape
    ppm = buf['ppm']
    off = np.zeros((Hh, W)); thr = np.full((Hh, W), 0.5); coh = np.zeros((Hh, W))
    gap = np.zeros((Hh, W), bool); strk = np.zeros((Hh, W), bool)
    # grass blades: vertical strands, height ~0.22 m turf / 0.5 m meadow
    for m, bh_m in (('turf', 0.2), ('meadow', 0.45), ('moss', 0.05)):
        sel = mat == MI[m]
        if not sel.any():
            continue
        bh = bh_m * ppm
        rl = np.clip(bh * 0.6, 1, 7.5)               # measured: 1.6 far, 3.8 mid, 5 near (V16PM)
        o, t = runs_field(Hh, W, rl, rng, True)
        # tufts: tall narrow noise gathers darks into pockets and lights onto tuft tops
        tuft = smooth_noise(Hh, W, 5.0, 0.8, rng)
        from scipy.special import ndtr
        u = ndtr(0.45 * tuft + 0.89 * o)               # run quantile, tuft-correlated
        thr[sel] = 0.5; coh[sel] = np.clip((bh - 3.0) / 2.5, 0, 1)[sel]   # mid distance: checker, not strands (V16)
        near = np.clip((bh - 1.5) / 4.0, 0, 1)
        # MARKS SKIP STEPS: the dark gap between blades in light is a near-black green (step 1);
        # in shadow the blades are 'negative painted' olive strokes on the dark (step 2 on 0)
        ug = runs_field(Hh, W, np.clip(rl * 1.0, 2, 6), rng, True)[1]   # gaps: independent, scattered
        gap[sel] = ((ug < 0.22 * near) & (tuft < -0.35) & lit)[sel]   # gaps sit in the tufts' dark pockets
        strk[sel] = ((u > 1 - 0.42 * near) & ~lit)[sel]
        # the mass's own mixture of base and one-step-down strands (density carries the turn)
        # near: each blade run takes the mass value +- a spread and rounds (always two classes,
        # their shares carry the mass); far: a finer stipple through the ordered dither
        a = np.where(lit, 1.0, 1.1) * np.clip(bh / 3, 0.45, 1)
        off[sel] = (a * (u - 0.5) * 1.7)[sel]
    # earth: horizontal streaks (ruts, the grain of the ground plane) + pebbles
    sel = mat == MI['earth']
    if sel.any():
        # streaks along the form (LIC along phi's iso-lines), few and quiet; a few pebbles near
        rl = np.clip(0.25 * ppm, 1, 9)
        base = rng.normal(size=(Hh, W))
        base = np.where(rng.random((Hh, W)) < 0.25, base * 2.2, base * 0.5)   # sparse strong grains
        o = lic(base, phi if phi is not None else buf['z'], rl)
        a = np.clip((0.3 * ppm - 0.3) / 3, 0.25, 0.8) * np.where(lit, 0.9, 0.6)
        st = stones_field(buf, 0.03, rng, density=0.06)
        off[sel] = (a * o + 0.8 * np.clip(0.03 * ppm, 0.1, 1) * st)[sel]
    sel = mat == MI['gravel']
    if sel.any():
        st = stones_field(buf, 0.06, rng, density=0.9)
        off[sel] = (np.clip(0.05 * ppm / 2, 0.2, 1) * st * np.where(lit, 0.6, 0.45))[sel]
    sel = mat == MI['forest']
    if sel.any():
        # ground-cover leaves (V30): flat horizontal lozenges, lit top row, dark underside,
        # overlapping; bold in light, nearly gone in shade
        st = leaves_field(buf, 0.09, rng, density=1.2)
        a = np.clip(0.09 * ppm / 3, 0.3, 1.2)
        off[sel] = (a * st * np.where(lit, 1.0, 0.7))[sel]
    return off * contrast, thr, coh, gap, strk


# ------------------------------------------------------------------ 8 PALETTE
ALBEDO = {  # linear-ish albedo, hand-picked; hue drift across the ramp added below
    'turf': (0.33, 0.30, 0.10), 'meadow': (0.40, 0.33, 0.10), 'earth': (0.34, 0.22, 0.13),
    'gravel': (0.34, 0.32, 0.29), 'forest': (0.19, 0.25, 0.06), 'moss': (0.20, 0.26, 0.07),
    'water': (0.5, 0.5, 0.5)}

def srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)

# the light family's lower steps keep a greener local colour; only the top light goes yellow/ochre
# shadows of green things go green-teal, not olive-grey (V16 noon: 0,23,7 / 31,67,39 / 0,7,15)
ALBEDO_SH = {'turf': (0.10, 0.30, 0.09), 'meadow': (0.16, 0.28, 0.10), 'forest': (0.08, 0.26, 0.12),
             'moss': (0.08, 0.28, 0.10)}
ALBEDO_LIT = {'forest': (0.16, 0.30, 0.05), 'turf': (0.20, 0.30, 0.07), 'meadow': (0.34, 0.31, 0.10), 'moss': (0.14, 0.30, 0.05)}

LSTEP = {  # target CIE L* per step: few steps, far apart; darks near black (V16 / V16PM ramps)
    'turf':   (3, 8, 29, 36, 50, 66),     # V16 far hummocks: L 2 / 6-9 / 25 in shadow
    'meadow': (6, 14, 25, 40, 56, 72),
    'earth':  (7, 15, 25, 37, 50, 63),
    'gravel': (8, 17, 28, 42, 57, 72),
    'forest': (4, 10, 19, 31, 45, 60),
    'moss':   (5, 12, 22, 36, 52, 70),
    'water':  (38, 46, 54, 62, 70, 78)}

def _lin(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def _Y2L(Y):
    return np.where(Y > 0.008856, 116 * np.cbrt(Y) - 16, 903.3 * Y)

def _L2Y(L):
    return np.where(L > 8, ((L + 16) / 116) ** 3, L / 903.3)

def _set_L(rgb_lin, L):
    """scale a linear colour to hit luminance L* (clipping pushes toward white gracefully)"""
    Y = (rgb_lin * np.array([0.2126, 0.7152, 0.0722])).sum()
    c = rgb_lin * (_L2Y(L) / max(Y, 1e-6))
    for _ in range(6):                                   # redistribute clipped energy
        over = np.clip(c - 1, 0, None).sum()
        c = np.clip(c, 0, 1)
        if over < 1e-4: break
        c = c + (1 - c) * over / 3
    return c

def ramp(mat, hour, band=0, wet=0.0):
    """6 steps: 0 core, 1 shadow, 2 reflected light | 3 halftone, 4 light, 5 top light."""
    az, el, sunc, skyc, haze = HOURS[hour]
    alb = np.array(ALBEDO[mat]); sunc, skyc, haze = map(np.array, (sunc, skyc, haze))
    Ls = np.array(LSTEP[mat], float)
    # the light family's value follows the light's strength (dusk and overcast are low-key)
    key = {'noon': 1.0, 'morning': 0.95, 'golden': 0.92, 'dusk': 0.72, 'overcast': 0.82}[hour]
    Ls[3:] = Ls[3:] * key
    if mat in ('turf', 'meadow'):
        Ls[:2] = Ls[:2] * 0.8           # under skylight grass is darker than bare soil (self-occlusion)
        # (step 2 is kept high: the shadow band is a two-value checker with a big gap, L6/L25)
    if hour == 'dusk':
        Ls[:3] = Ls[:3] * (0.5 if mat in ('turf', 'meadow') else 0.6)          # measured V13: dusk ground lives at L* 0..12
    if hour == 'overcast':
        Ls[:3] = Ls[:3] + np.array([6, 6, 4])            # soft light: shadows open up, no terminator
    if wet > 0:
        Ls = Ls * (1 - 0.3 * wet)
        alb = alb ** (1 + 0.6 * wet)                     # darker and more saturated
    out = []
    for i in range(6):
        if mat == 'water':
            # a puddle is the sky, lighter than the ground around it, cooled; steps are the
            # sky's own ramp from zenith (low) to horizon (high)
            c = (skyc * (1 - i / 6) + haze * (i / 6)) * 1.0
        elif i < 3:
            # in shadow the local colour weakens and the sky's hue takes over (strongest at dusk)
            dom = {'noon': 0.25, 'morning': 0.4, 'golden': 0.5, 'dusk': 0.85, 'overcast': 0.2}[hour]
            dom = dom * (0.55, 0.85, 1.0)[band]     # near shadows keep their local green (V13, V16PM)
            ash = np.array(ALBEDO_SH.get(mat, alb))
            an = ash / ash.max()
            c = (an ** (1 - dom)) * skyc * (1.0 if i else 0.9)
            if i == 0: c = c * np.array([0.75, 0.95, 1.35])   # cool core (V16's teal darks)
            g = (c * np.array([0.2126, 0.7152, 0.0722])).sum()
            c = np.clip(g + (c - g) * 1.25, 0, None)            # shadows stay coloured, never grey
        else:
            a2 = np.array(ALBEDO_LIT.get(mat, alb)) if i == 3 else (0.5 * (np.array(ALBEDO_LIT.get(mat, alb)) + alb) if i == 4 else alb)
            c = a2 * (sunc + skyc * 0.15)
            if i == 5: c = c * np.array([1.10, 1.04, 0.85])    # top light goes yellow/ochre
            if i == 3: c = c * 0.7 + alb * skyc * 0.6          # halftone keeps some sky
            g = (c * np.array([0.2126, 0.7152, 0.0722])).sum()
            c = np.clip(g + (c - g) * 1.1, 0, None)             # lit colours carry the chroma
        out.append(_set_L(c, Ls[i]))
    out = np.array(out)
    f = [0.0, 0.07, 0.2][band]
    # far: the black point lifts toward the haze -- by an amount keyed to the hour (at dusk the
    # far land is still dark, L* 6..18 in V13; at noon it pales)
    hz = _set_L(haze, max(Ls[2] * 1.5, 8) if hour != 'dusk' else Ls[2] * 1.3)
    # the lift is gentle on the darks: V16's far hummocks still hold L* 2 in their bases, and a
    # lifted black point merges steps 0 and 1 into one flat colour (the pale-teal band failure)
    w = f * np.array([0.6, 0.7, 0.9, 1.4, 1.1, 0.6])[:, None]
    out = out * (1 - w) + hz[None] * w
    return (np.clip(srgb(out), 0, 1) * 255).astype(np.uint8)


def palette_sheet(k=12):
    rows = []
    for hr in HOURS:
        row = []
        for m in MATS:
            for b in range(3):
                row.append(np.repeat(ramp(m, hr, b)[None], 1, 0))
            row.append(np.zeros((1, 1, 3), np.uint8))
        rows.append(np.concatenate(row, 1))
    return np.concatenate(rows, 0)


# ------------------------------------------------------------------ assemble
class Ground:
    pass

def crest_fringe(step, mat, band, buf, lit, rng, tip=0.2, holes=0.3):
    """Blade tips break the top of every crest: the near mass's grass pokes up over the land
    behind it (lit peaks), and dark holes open along the crest line (V16PM C2/C3 mound)."""
    z = buf['z']; Hh, W = z.shape
    ppm = buf['ppm']
    crest = np.zeros_like(z, bool)
    crest[1:] = ((z[:-1] > z[1:] * 1.25) & (z[:-1] - z[1:] > 2.0)) | buf['sky'][:-1]
    crest &= ~buf['sky']
    ys, xs = np.nonzero(crest)
    for y, x in zip(ys, xs):
        m = mat[y, x]
        if m not in (MI['turf'], MI['meadow'], MI['moss'], MI['forest']):
            continue
        bh = tip * (2.2 if m == MI['meadow'] else 1) * ppm[y, x]
        if bh < 0.8:
            continue
        u = rng.random()
        if u < holes:
            step[y, x] = max(0, step[y, x] - 2)
            continue
        if u < holes + 0.45:
            L = int(np.clip(round(bh * rng.uniform(0.15, 0.6) ** 1.5 * 1.6), 1, 8))
            s0 = step[y, x] + (1 if lit[y, x] and rng.random() < 0.5 else 0)
            for k in range(1, L + 1):
                if y - k < 0: break
                step[y - k, x] = min(5, s0); mat[y - k, x] = m; band[y - k, x] = band[y, x]
    return step, mat, band

def hero_blades(step, mat, band, buf, lit, rng, density=0.05, min_px=7):
    """The nearest grass carries a few long single blades drawn as strokes (V13's foreground,
    V16's right bank): in light a dark stem with a lit pixel beside it (the paired stroke); in
    shadow an olive stroke on the dark (negative painting). Rasterized one row at a time so the
    lean accumulates into clean 1:1 / 2:1 runs."""
    H, W = step.shape
    ppm = buf['ppm']
    ys, xs = np.nonzero(np.isin(mat, [MI['turf'], MI['meadow']]) & (0.2 * ppm >= min_px))
    if len(ys) == 0:
        return step, mat, band
    n = int(len(ys) * density / max(1.0, 0.2 * ppm[ys].mean()))
    pick = rng.choice(len(ys), size=min(n, len(ys)), replace=False)
    for i in pick:
        y, x = ys[i], xs[i]
        bh = 0.2 * ppm[y, x] * (2.2 if mat[y, x] == MI['meadow'] else 1.0)
        L = int(bh * rng.uniform(1.0, 2.3))
        lean = rng.normal(0, 0.35); bend = rng.choice([-1, 1]) * rng.uniform(0.2, 1.2)
        m0, b0, lt = mat[y, x], band[y, x], lit[y, x]
        fx = float(x)
        for k in range(L):
            t = k / max(L, 1)
            fx += lean + bend * t * t * 1.5
            yy, xx = y - k, int(round(fx))
            if yy < 0 or xx < 0 or xx >= W: break
            if lt:
                step[yy, xx] = 3 if t < 0.8 else 4          # a mid-tone stem, not a black crack
                if xx + 1 < W and rng.random() < 0.5: step[yy, xx + 1] = 5
            else:
                step[yy, xx] = 2 if t < 0.9 else 1
            mat[yy, xx] = m0; band[yy, xx] = b0
    return step, mat, band

def plan_bias(plan, H, W, buf):
    """Screen-space composition of the big shadow masses (Ferrari designs them; V16: the right
    bank in shadow, the left lit, a pool of light on the islet; V09: a lit path in a dark floor).
      None | 'side-left' | 'side-right' | 'foreground' | 'pool' | 'V' | 'far'  """
    if plan is None:
        return None
    yy, xx = np.mgrid[0:H, 0:W]
    u = xx / W - 0.5; t = yy / H
    ground_top = np.argmax(~buf['sky'], axis=0).mean() / H
    tg = np.clip((t - ground_top) / max(1 - ground_top, 0.1), 0, 1)     # 0 at horizon, 1 at bottom
    b = {'side-left': -u * 2, 'side-right': u * 2, 'foreground': (tg - 0.6) * 2.5,
         'far': (0.35 - tg) * 2.5,
         'pool': (np.hypot(u * 1.6, (tg - 0.5) * 2.2) - 0.45) * 2.5,
         'V': np.abs(u) * 2 + (tg - 0.8) * 1.5 - 0.3}[plan]
    return b

def crest_profile(t, accent=0.35, peak=0.25):
    """Measured on V16's far hummocks: value below a crest line, t = 0 at the crest, 1 at the
    base. A thin darker accent on the crest, the lit band ~25% down, falloff to a dark base."""
    up = accent + (1 - accent) * np.clip(t / peak, 0, 1) ** 0.7
    down = np.clip(1 - (t - peak) / (1 - peak), 0, 1) ** 0.9
    return np.where(t < peak, up, down)

def crest_runs(buf, cap):
    """Per pixel: rows below the crest line above it (an occlusion edge or the skyline) and the
    length of that run down to the next crest; runs longer than cap px use cap (near ground has
    no single form reaching the frame bottom)."""
    z = buf['z']; H, W = z.shape
    brk = np.zeros((H, W), bool)
    brk[1:] = ((z[:-1] > z[1:] * 1.25) & (z[:-1] - z[1:] > 2.0)) | buf['sky'][:-1]
    brk[0] = True
    brk &= ~buf['sky']
    k = np.zeros((H, W)); Ln = np.ones((H, W))
    for x in range(W):
        ys = list(np.nonzero(brk[:, x])[0]) + [H]
        for a, b in zip(ys[:-1], ys[1:]):
            k[a:b, x] = np.arange(b - a); Ln[a:b, x] = b - a
    return k, np.minimum(Ln, cap), Ln

def turn_masses(v, lit, buf, crest_band=1.0, mass_turn=1.0, cap=None):
    """Turn the flat masses into forms, inside each family:
      crest band -- every land unit gets the V16 hummock profile below its crest line
      mass turn  -- a lit pool is hottest in its interior and falls off (dithered) toward its
                    edge (the V16 islet); a shadow is darkest along its edge and lifts inside
                    (reflected light), so no mass is a flat colour field with a noisy edge"""
    H, W = v.shape
    out = v.copy()
    if crest_band > 0:
        cap = cap or max(24, H * 0.35)
        k, Lc, Ln = crest_runs(buf, cap)
        t = np.clip(k / Lc, 0, 1)
        f = crest_profile(t)
        # a run cut short by the next crest ends in the dark base (the hummock meets the one in
        # front); a long run (near ground) just relaxes back to neutral after its band
        open_ = Ln > Lc
        relax = 0.45 + (1 - 0.45) * np.exp(-np.clip(t - 0.25, 0, None) / 0.3)
        f = np.where(open_ & (t > 0.25), relax, f)
        f = ndi.gaussian_filter(f, (0.6, 2.0))
        amp = crest_band * np.where(lit, 0.9, 1.3)
        out = out + amp * (f - 0.45)
    if crest_band > 0 and 'convex' in buf:
        # skylight on convex land: mound tops take the sky, the feet sink (V16 hummock bodies)
        out = out + crest_band * np.where(lit, 0.5, 0.9) * np.tanh(2.5 * buf['convex'])
    if mass_turn > 0:
        dl = ndi.distance_transform_edt(lit); ds = ndi.distance_transform_edt(~lit)
        D = max(6.0, 0.06 * min(H, W))
        # lit: the islet recipe, a hot interior and a dithered falloff to the edge.
        # shadow: a dark core just inside the terminator (Ferrari's accent where shade meets
        # light), then reflected light lifting the broad interior
        out = out + mass_turn * np.where(lit, 1.0 * (np.clip(dl / D, 0, 1) - 0.55),
                                         -0.8 * np.exp(-ds / 3.0) + 0.7 * (np.clip(ds / (2.5 * D), 0, 1) - 0.4))
    return np.where(lit, np.clip(out, 3.0, 5.99), np.clip(out, 0, 2.9))

def paint_ground(W=320, H=200, ground='turf', landform='rolling', hour='golden', sun=None,
                 moisture=0.2, ppm=24, pitch=14, cam_h=1.7, canopy=0.0, canopy_scale=3.0,
                 shade=0.0, shade_scale=12.0, vignette=0.35, path=False, relief=1.0, sep=0.9,
                 sheen=0.35, contrast=1.0, jitter=0.5, band_z=(12, 45), seed=0, water=None, lit_base=0.5,
                 shadow_mask=None, blades=0.035, plan=None, crest_band=1.0, mass_turn=1.0,
                 shadow_key=0.3, lit_key=0.8):
    """ground: turf|meadow|earth|gravel|forest   landform: flat|rolling|hummocks|bank|ledges
    hour: noon|morning|golden|dusk|overcast (or sun=(az,el))   moisture 0..1
    ppm: px per metre at screen centre (the distance knob)   pitch/cam_h: camera
    canopy (+canopy_scale m): dapple from off-frame trees; shade (+shade_scale m): big cast
    shadows from off-frame forest/cloud -- Ferrari's compositional shadow masses
    vignette: darkens the bottom corners inside each family (repoussoir)"""
    rng = np.random.default_rng(seed)
    land = Landform(landform, seed, relief, path=path, water=water)
    buf = render(land, W, H, ppm, pitch, cam_h)
    az, el = (sun if sun else HOURS[hour][:2])
    L = light(land, buf, az, el, canopy, canopy_scale, seed, shade, shade_scale, plan_bias(plan, H, W, buf))
    if shadow_mask is not None:
        # cast shadows of objects standing on the ground (a log, a bush, a stone): the ground
        # itself in its shadow family, texture kept inside (READY.md: cast shadows)
        L['sunvis'] = L['sunvis'] * (1 - shadow_mask.astype(float))
    mat, wet = materials(land, buf, ground, moisture, L, seed)
    v, lit, sepm = value_plan(buf, L, mat, sep=sep, sheen=sheen, lit_base=lit_base, overcast=(hour == 'overcast'))
    # moisture darkens the mass (a step at full wet) -- a mass-level change, not a texture
    v = v - 0.9 * wet * (v > 0.5)
    # repoussoir: bottom corners sink, inside each family
    yy, xx = np.mgrid[0:H, 0:W]
    vg = np.clip((yy / H - 0.55) / 0.45, 0, 1) ** 1.5 * (0.45 + 0.55 * np.abs(xx - W / 2) / (W / 2))
    v = v - vignette * 1.6 * vg
    v = np.where(lit, np.maximum(v, 3.0), np.minimum(v, 2.9))   # families never touch
    v = turn_masses(v, lit, buf, crest_band, mass_turn)            # masses become forms
    phi = buf['z'].copy()
    if path:
        pd = land.path_dist(buf['x'], buf['z'])
        phi = np.where(pd < 2.0, pd * 40, phi)
        # the path's mass: a packed, lighter crown; grass-shaded margins drop a step
        onp = (mat == MI['earth']) & (pd < 1.2)
        v = np.where(onp, v + 0.5 * np.exp(-(pd / 0.35) ** 2) - 0.9 * np.clip((pd - 0.45) / 0.3, 0, 1), v)
        v = np.where(lit, np.maximum(v, 3.0), np.minimum(v, 2.9))
    # headroom: keep the mass off the ends of its family so marks can go both ways (a mass
    # pinned at the top step goes flat -- the flat-tone failure)
    # the shadow family sits so that its body is the 1/2 checker and only crest bands reach a
    # solid step 2 (V16 hummocks: L6/L25 checker ~70% of the mass, solid lights rare)
    v = np.where(lit, 3.15 + (v - 3.0) * 0.78, v)
    lm = lit & (mat >= 0) & (mat != MI['water']) & ~buf['sky']
    if lm.sum() > 50 and lit_key > 0:
        # key the light family the same way: whatever turning the land gives (N.L, skylight,
        # crest bands, convexity) is stretched to fill 3..5.9, so a lit meadow shows its
        # swells as sweeps of halftone and top light instead of one flat ochre
        q = np.percentile(v[lm], [3, 50, 97])
        if q[2] - q[0] > 1e-3:
            tgt = np.array([3.05, 4.35, 5.75])
            vk = np.interp(v, q, tgt, left=3.0, right=5.9)
            v = np.where(lm, (1 - lit_key) * v + lit_key * vk, v)
    sm = (~lit) & (mat >= 0) & (mat != MI['water']) & ~buf['sky']
    if sm.sum() > 50:
        # key the shadow family by its own quantiles (a value plan, not a lighting formula):
        # the body of the shadow is the 1/2 checker, the crest bands reach solid 2, and only
        # the feet and cores reach 0 -- V16 hummocks: L2 15% / L6 35% / L25 38%
        q = np.percentile(v[sm], [3, 50, 97])
        tgt = np.array([0.15, 1.0 + shadow_key, 2.3])
        v = np.where(sm, np.interp(v, q, tgt, left=0.2, right=2.6), v)
    off, thr, coh, gap, strk = marks(buf, mat, v, lit, rng, contrast, phi)
    if hour == 'dusk':                 # low light: texture sinks, masses carry the picture (V13)
        off = off * 0.45; strk &= rng.random(strk.shape) < 0.35
    # distance: mark contrast compresses with depth beyond the near band
    fade = np.clip(1.2 - np.log(buf["z"] / band_z[0] + 1e-6) / 2.2, 0.45, 1)
    vv = v + off * fade
    # keep the families apart: marks may not cross the terminator (only a thin fringe may)
    vv = np.where(lit, np.clip(vv, 2.6, 5.99), np.clip(vv, 0, 2.99))
    # run-coherent threshold for strands (a blade is one colour top to bottom), DPaint ordered
    # dither for the mass everywhere else (the V16 islet recipe)
    tb = dither.tile(dither.B2, H, W) + jitter * (rng.random((H, W)) - 0.5)
    t = coh * thr + (1 - coh) * tb
    step = np.clip(np.floor(vv) + ((vv - np.floor(vv)) > t), 0, 5).astype(int)
    step = np.where(lit, np.maximum(step, 3), np.minimum(step, 2))   # families never touch
    wat = mat == MI['water']
    if wat.any():
        # puddles reflect the sky: nearer water sees higher (darker) sky; 1-px ripple rows
        vw = 1.8 + 3.0 * np.clip(np.log(buf['z'] / 2) / 3.5, 0, 1)
        rows = (np.arange(H)[:, None] % 3 == 0) & (rng.random((H, W)) < 0.35)
        vw = vw - 0.6 * rows
        tw = dither.tile(dither.B2, H, W)
        sw = np.clip(np.floor(vw) + ((vw - np.floor(vw)) > tw), 0, 5).astype(int)
        # the far bank reflects as a dark band under the pool's top edge (the channel lesson:
        # far edge dark, near edge light), with stepped bites
        k = np.clip(np.round(0.06 * buf['ppm']), 1, 4).astype(int)
        band_ = np.zeros((H, W), bool)
        for d in range(1, 5):
            above = np.zeros((H, W), bool); above[d:] = ~wat[:-d]
            band_ |= above & (d <= k + (rng.random((H, W)) < 0.3))
        sw = np.where(band_, np.maximum(sw - 3, 0), sw)
        step[wat] = sw[wat]
    step[gap] = 1
    step[strk] = np.minimum(np.maximum(step[strk] + 2, 2), 2)
    # depth bands meet in a dithered seam, never a ruled line
    lz = np.log(buf['z']) + 0.35 * (dither.tile(dither.B4, H, W) - 0.5) + 0.1 * (rng.random((H, W)) - 0.5)
    band = np.digitize(lz, np.log(band_z))
    mat[buf['sky']] = -1
    step, mat, band = crest_fringe(step, mat, band, buf, lit, rng)
    if blades > 0:
        step, mat, band = hero_blades(step, mat, band, buf, lit, rng, density=blades)
    g = Ground()
    g.__dict__.update(mat=mat, band=band, step=step, v=v, vv=vv, lit=lit, wet=wet, buf=buf, L=L,
                      land=land, depth=buf['z'], wx=buf['x'], wz=buf['z'], h=buf['h'], sepm=sepm,
                      hour=hour, moisture=moisture)
    g.relight = lambda hr, g=g: colour(g, hr)
    g.rgb = colour(g, hour)
    return g

def colour(g, hour, sky_rgb=None):
    Hh, W = g.step.shape
    out = np.zeros((Hh, W, 3), np.uint8)
    az, el, sunc, skyc, haze = HOURS[hour]
    for mi, m in enumerate(MATS):
        for b in range(3):
            for wetk, wsel in ((0, g.wet < 0.5), (1, g.wet >= 0.5)):
                sel = (g.mat == mi) & (g.band == b) & wsel
                if sel.any():
                    R = ramp(m, hour, b, wet=0.8 * wetk if g.moisture > 0 else 0)
                    out[sel] = R[g.step[sel]]
    sky = g.mat < 0
    if sky.any():
        out[sky] = (srgb(np.array(haze) * 0.9) * 255).astype(np.uint8)
    return out
