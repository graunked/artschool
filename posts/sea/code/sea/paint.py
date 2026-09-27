"""The sea painter: G-buffer + sea state -> (ramp, step) index canvas. Every mark has a trigger in a field.
Scale-aware: world-space features, and each decides by its size in pixels how it is drawn."""
import numpy as np
from scipy import ndimage as nd
from .waves import bilinear
from .noise import hash2
from .pal import RID, DEPTH_EDGES

BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0 + 1 / 32

def bayer(H, W):
    return np.tile(BAYER4, (H // 4 + 1, W // 4 + 1))[:H, :W]

def vnoise(x, y, cell, seed):
    """value noise in world coords (smooth, 0..1)."""
    gx, gy = x / cell, y / cell
    x0, y0 = np.floor(gx), np.floor(gy); fx, fy = gx - x0, gy - y0
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy)
    a = hash2(x0, y0, seed); b = hash2(x0 + 1, y0, seed); c = hash2(x0, y0 + 1, seed); d = hash2(x0 + 1, y0 + 1, seed)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy

def fbmw(x, y, cell, seed, oct=3):
    s = 0; a = 1; t = 0
    for o in range(oct):
        s = s + a * vnoise(x, y, cell / 2 ** o, seed + o * 17); t += a; a *= 0.5
    return s / t

def worley(x, y, cell, seed, ux=None, uy=None, stretch=1.0):
    """F1, F2 (in cell units). Optional anisotropy: cells stretched `stretch`x along (ux, uy)."""
    if ux is not None and stretch != 1.0:
        a = x * ux + y * uy; b = -x * uy + y * ux
        x, y = a / stretch, b
    gx, gy = x / cell, y / cell
    ix, iy = np.floor(gx), np.floor(gy)
    f1 = np.full(gx.shape, 9.0); f2 = np.full(gx.shape, 9.0)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cx, cy = ix + dx, iy + dy
            px = cx + hash2(cx, cy, seed); py = cy + hash2(cx, cy, seed + 1)
            dd = np.hypot(gx - px, gy - py)
            f2 = np.where(dd < f1, f1, np.minimum(f2, dd)); f1 = np.minimum(f1, dd)
    return f1, f2

def local_foam(parts, G, res, sigma_m, gain):
    """splat foam particles onto a fine grid over the camera footprint (close views)."""
    x0, x1 = G['x'].min() - 2, G['x'].max() + 2; y0, y1 = G['y'].min() - 2, G['y'].max() + 2
    px, py = parts['x'], parts['y']
    m = (px > x0) & (px < x1) & (py > y0) & (py < y1)
    nx_, ny_ = int((x1 - x0) / res) + 1, int((y1 - y0) / res) + 1
    grid = np.zeros((ny_, nx_), np.float32)
    np.add.at(grid, (((py[m] - y0) / res).astype(int), ((px[m] - x0) / res).astype(int)), 1.0)
    grid = nd.gaussian_filter(grid, sigma_m / res) / (res * res) / 2.5 * gain
    return bilinear(grid, (G['x'] - x0) / res, (G['y'] - y0) / res)

def cq(base, mpp, k):
    """feature size: base metres, but never below k pixels; quantised to octaves so LOD bands don't swim."""
    m = np.maximum(1.0, k * np.asarray(mpp, float) / base)
    return base * 2.0 ** np.ceil(np.log2(m))

def sample(F, G, key):
    return bilinear(F[key], G['x'], G['y'])

def paint_sea(G, S, cam, P=None):
    """S: dict of fields on the 1 m grid (depth, rock, kelp, W, F, A, q, Hinst, brk, u, v, T, H, kx, ky, tt).
    P: parameters. Returns ramp, step (int arrays), and a dict of layers for inspection."""
    P = dict(dict(band_jitter=0.35, lace_cell=2.2, lace_stretch=3.0, lump_cell=0.9, kelp_on=0.40,
                  speck_on=0.14, aer_on=0.22, roller_q=0.035, face_q=0.16, chop=0.5, sun_az=205, sun_el=50,
                  glint=0.0, wind_dir=(0.0, 1.0)), **(P or {}))
    H_, W_ = G['x'].shape
    mpp = G['mpp'] if 'mpp' in G else cam.mpp
    x, y = G['x'], G['y']
    sea = G['sea'].copy()
    ramp = np.full((H_, W_), -1, int); step = np.zeros((H_, W_), int)
    by = bayer(H_, W_)
    d = sample(S, G, 'depth'); rock = sample(S, G, 'rock_s') if 'rock_s' in S else sample(S, G, 'rock'); kelp = sample(S, G, 'kelp')
    Wf = sample(S, G, 'W'); Ff = sample(S, G, 'F'); Af = sample(S, G, 'A')
    if 'parts' in S and np.median(mpp) < 0.7:
        Ff = local_foam(S['parts'], G, res=max(float(np.median(mpp)), 0.12), sigma_m=0.45, gain=S.get('part_gain_fine', 0.9))
    hh, ww = S['depth'].shape
    outside = (x < 2) | (y < 2) | (x > ww - 3) | (y > hh - 3)          # beyond the simulated tile: plain sea
    Wf = np.where(outside, 0, Wf); Ff = np.where(outside, 0, Ff); Af = np.where(outside, 0, Af); kelp = np.where(outside, 0, kelp)
    Hs = sample(S, G, 'Hinst'); u = sample(S, G, 'u'); v = sample(S, G, 'v')
    T = S['T']
    # phase: recompute from travel time so it is smooth at any zoom
    q = np.mod((S['t_now'] - sample(S, G, 'tt')) / T, 1.0)
    brk = sample(S, G, 'brkf') > 0.5
    sea &= d > 0.02
    # ---------------- 1. the body: depth bands over bottom type -------------------------------
    jit = (fbmw(x, y, 9.0, 5) - 0.5) * 2 * P['band_jitter']            # irregular band edges, world space
    dsm = sample(S, G, 'depth_s') if 'depth_s' in S else d
    lnd = np.log(np.maximum(dsm, 0.05))
    edges = np.log(DEPTH_EDGES)
    cont = np.interp(lnd + jit * 0.35, edges, np.arange(len(edges)) + 0.5, left=0, right=len(edges))
    # cont: 0 (shallow) .. 6 (deep) continuous; step 6 = shallowest
    lvl = np.floor(cont).astype(int); frac = cont - lvl
    ck = ((np.arange(W_)[None, :] + np.arange(H_)[:, None]) % 2).astype(bool)
    # flat bands; a 50% checker only in a thin seam at each boundary (never a single dotted row:
    # the seam is as wide as the band's local gradient allows, >= 2 px)
    seam_w = 0.10
    lvl = np.where((frac > 1 - seam_w) & ck, lvl + 1, lvl)
    lvl = np.where((frac < seam_w) & ~ck, lvl - 1, lvl)
    lvl = np.clip(lvl, 0, 6)
    body_step = 6 - lvl
    is_rock = rock > 0.5 + 0.25 * (by - 0.5)
    ramp[sea] = np.where(is_rock, RID['sea_rock'], RID['sea_sand'])[sea]
    step[sea] = body_step[sea]
    if 'graze' in G:
        # eye level: Fresnel takes over; the water becomes the sky's ramp (Schlick, n=1.33) + roughness
        th = np.radians(90 - G['graze']); F0 = 0.02
        Fr = F0 + (1 - F0) * (1 - np.cos(th)) ** 5
        Fr = np.clip(Fr * 0.85 + 0.03, 0, 1)
        # Ruskin's 'few dark continuous furrows': swell faces turned toward the eye mirror higher, darker
        # sky; backs turned away mirror the bright horizon. Slope along the view from the crest phase.
        qq = np.mod((S['t_now'] - sample(S, G, 'tt')) / S['T'], 1.0)
        kxs, kys = sample(S, G, 'kx'), sample(S, G, 'ky')
        toward = -(kxs * G['dirx'] + kys * G['diry'])            # >0: wave travels toward the viewer
        slope = -np.sin(2 * np.pi * qq) * np.clip(sample(S, G, 'H') / 1.2, 0, 2)
        Fr = np.clip(Fr * (1 - 0.55 * np.clip(slope * toward, -1, 1)), 0, 1)
        from .pal import MIRROR_F
        mf = np.interp(Fr + (by - 0.5) * 0.04, MIRROR_F, np.arange(len(MIRROR_F)))
        mst = np.round(mf).astype(int)
        mm = sea & (Fr > 0.07)
        ramp[mm] = RID['mirror']; step[mm] = mst[mm]
        G['_mirror'] = mm
    # rock bottom seen through shallow water: mottle (pale rock / dark weed), fading with depth
    vis = np.exp(-d / 3.0) * (rock > 0.4) * (np.asarray(mpp) < 0.5)
    mott = fbmw(x, y, 2.5, 21, 3)
    step = np.where(sea & (vis > 0.15) & (mott > 0.62), np.minimum(step + 1, 6), step)
    step = np.where(sea & (vis > 0.15) & (mott < 0.30), np.maximum(step - 1, 0), step)
    # ---------------- 2. the swell: broad value undulation + the face before a crest -----------
    # the face (shoreward of the crest, about to be reached) turns darker where the wave is steep
    steep = np.clip(Hs / np.maximum(d, 0.3), 0, 1.2)
    face = sea & (q > 1 - P['face_q']) & (steep > 0.35) & ~brk
    step = np.where(face & (by < np.clip((steep - 0.35) * 2.5, 0, 1)), np.maximum(step - 1, 0), step)
    # back of the wave (just passed): slightly lighter (tilted to the sky)
    back = sea & (q < 0.18) & (q > P['roller_q']) & (steep > 0.3)
    step = np.where(back & (by < 0.35), np.minimum(step + 1, 6), step)
    # ---------------- 3. chop: short strokes along the crests of wind ripples -----------------
    # world-space ripple cells ~ 0.6-3 m; a mark only where its length is >= 2 px
    ux, uy = P['wind_dir']
    cellc = cq(1.2, mpp, 2.5)
    f1, f2 = worley(x, y, cellc, 91, ux, uy, 0.35)
    chop_m = sea & (f2 - f1 < 0.10) & (vnoise(x, y, 25, 3) < P['chop']) & (d > 1.0) & (np.asarray(mpp) <= 0.4)
    step = np.where(chop_m & (hash2(np.floor(x / cellc), np.floor(y / cellc), 7) > 0.5), np.minimum(step + 1, 6), step)
    # ---------------- 4. kelp canopy -----------------------------------------------------------
    kn = kelp + (fbmw(x, y, 1.6, 33, 2) - 0.5) * 0.35
    krho0 = sample(S, G, 'kelp_rho') if 'kelp_rho' in S else kelp
    km = sea & (kn > P['kelp_on']) & (krho0 > P.get('mat_rho', 0.22))
    ramp[km] = RID['kelp']
    kst = np.where(kn > P['kelp_on'] + 0.25, 0, np.where(kn > P['kelp_on'] + 0.08, 1, 2))
    step[km] = kst[km]
    # scattered floats and blades beyond the mat: specks, a few px, streaming down-current
    speck_cell = cq(0.9, mpp, 2.0)
    sf1, _ = worley(x, y, speck_cell, 44, None, None, 1.0)
    krho = sample(S, G, 'kelp_rho') if 'kelp_rho' in S else kelp
    grp = fbmw(x, y, 18.0, 45, 2) > 0.55                      # specks come in groups
    sp = sea & ~km & grp & (krho > 0.03) & (sf1 < 0.08 + 0.25 * krho) & (hash2(np.floor(x / speck_cell), np.floor(y / speck_cell), 46) < 0.5)
    ramp[sp] = RID['kelp']; step[sp] = 2
    # ---------------- 5. aeration: jade halo of bubbles ---------------------------------------
    an = Af * (0.75 + 0.5 * fbmw(x, y, 3.0, 55, 2))
    am = sea & (an > P['aer_on'])
    ramp[am] = RID['aer']
    step[am] = np.clip(((an - P['aer_on']) / 0.35 + (by - 0.5) * 0.8).astype(int), 0, 2)[am]
    # swirl threads inside the bubble pool (the photo's pool is laced with white filaments), only when the
    # threads' cells are >= 5 px, so they never collapse into salt
    scell = cq(1.3, mpp, 5.0)
    sf1, sf2 = worley(x, y, scell, 88, None, None, 1.0)
    swirl = am & (an > 0.45) & (sf2 - sf1 < 0.07 + 0.08 * np.clip(an - 0.45, 0, 1)) & (scell / np.asarray(mpp) >= 5.0)
    # (dropped: Voronoi-edge threads read as runes at 0.1 m/px -- the water study's warning, relearned)
    # ---------------- 6+7. foam as COVERAGE: solid shapes with ragged, noise-contour edges ----------
    # (no per-pixel stipple: the threshold field is smooth at >= 3 px, so edges are ragged, never salt)
    un = np.hypot(u, v) + 1e-6
    Fe = 1 - np.exp(-np.clip(Ff, 0, 6) * P.get('foam_gain', 0.3))   # count density -> coverage
    # in the surf zone old foam is densest just behind each bore and thins into lace until the next one
    # (Ruskin: the curdling foam 'subsides' into the thin coating) -> one band per wave, not uniform marbling
    surfz = sample(S, G, 'surf') if 'surf' in S else np.clip((steep - 0.45) / 0.3, 0, 1)
    Fe = np.minimum(Fe, 1 - surfz * np.clip(1.1 * q - 0.05, 0, 0.85))
    # the bubble pool is laced with surfacing foam: part of its aeration shows as coverage (photo 0.3 m/px)
    cov = np.maximum(np.maximum(np.clip(Wf * 1.25, 0, 1.2), Fe), np.clip((Af - 0.2) * 1.6, 0, 1) * P.get('pool_lace', 0.9))
    ecell = cq(3.0, mpp, 4.0)
    # threshold noise stretched along the current: patches and streaks elongate the way the foam drifts
    # foam lines are remnants of broken crests: their edges run along the crest (perpendicular to k)
    # ONE direction for the whole frame (the mean swell heading): rotating the noise per pixel swirled it into
    # wood-grain rings around every rock
    if 'k0' not in S:
        sea_ = S['depth'] > 0.5
        S['k0'] = (float(np.median(S['kx'][sea_])), float(np.median(S['ky'][sea_])))
    tx_, ty_ = -S['k0'][1], S['k0'][0]
    xa = (x * tx_ + y * ty_) / P['lace_stretch']; ya = -x * ty_ + y * tx_
    thr = 0.10 + 0.50 * fbmw(xa, ya, ecell, 68, 2)
    foam = sea & (cov > thr)
    # Ruskin's oval gaps / Gilland's expanding holes: only where the holes are >= 4 px across
    hcell = cq(P['lace_cell'] * 2.2, mpp, 6.0)
    hf1, hf2 = worley(x, y, hcell, 66, tx_, ty_, 1.6)
    hole_r = np.clip(0.75 - cov * 0.7 + surfz * 0.35 * q, 0, 0.8)                  # older, thinner foam -> bigger holes
    holes = foam & (hf1 < hole_r) & (cov < 0.9) & (hcell / np.asarray(mpp) >= 6.0)
    foam &= ~holes
    ramp[foam] = RID['foam']
    # tone: fresh whitewater gets lumps lit toward the sun and shaded away; old foam is the mid tone,
    # its densest cores lit
    lcell = cq(P['lump_cell'], mpp, 2.0)
    lump = fbmw(x, y, lcell, 77, 3)
    saz = np.radians(P['sun_az'])
    sx_, sy_ = np.sin(saz), -np.cos(saz)
    e = 0.35 * lcell
    lg = fbmw(x + sx_ * e, y + sy_ * e, lcell, 77, 3) - lump
    fresh = Wf > 0.35
    tone = np.where(fresh, np.where(lg > 0.02, 3, np.where(lg < -0.035, 2, 3)), np.where(cov > 0.97, 2, 1))
    step[foam] = tone[foam]
    # ---------------- 8. the roller: the breaking crest itself, a thin bright line -------------
    rq = np.maximum(P['roller_q'], 0.6 * mpp / (np.maximum(np.sqrt(9.81 * np.maximum(d, 0.2)), 1) * T))
    roller = sea & brk & (q < rq)
    ramp[roller] = RID['foam']; step[roller] = 3
    # shadow under the lip: the pixel just shoreward of the roller (facing away from the sun)
    # ---------------- 9. sun glitter (Cox & Munk 1954): facets tilted to mirror the sun into the eye --------
    if P.get('sun_el', 50) > -1 and P.get('glitter', 1.0) > 0:
        sel, saz = np.radians(P['sun_el']), np.radians(P['sun_az'])
        sv = np.array([np.sin(saz) * np.cos(sel), -np.cos(saz) * np.cos(sel), np.sin(sel)])
        if 'dirx' in G:
            dxv, dyv, dzv = G['dirx'], G['diry'], G['dz']
        else:
            e = np.radians(cam.elev); fx, fy = -cam.V[0], -cam.V[1]
            dxv = np.full(x.shape, fx * np.cos(e)); dyv = np.full(x.shape, fy * np.cos(e)); dzv = np.full(x.shape, -np.sin(e))
        hx, hy, hz = sv[0] - dxv, sv[1] - dyv, sv[2] - dzv
        hn = np.sqrt(hx * hx + hy * hy + hz * hz) + 1e-9
        cth = np.clip(hz / hn, 1e-3, 1); tan2 = (1 - cth * cth) / (cth * cth)
        U = P.get('wind_speed', 3.0); s2 = 0.003 + 0.00512 * U
        pg = np.exp(-tan2 / s2) * P.get('glitter', 1.0)
        gcell = cq(0.8, mpp, 1.0)
        hsh = hash2(np.floor(x / gcell), np.floor(y / gcell * 2.5), 123)
        gl = sea & (hsh < pg * 0.35) & ~(ramp == RID['foam'])
        ramp[gl] = RID['glint']; step[gl] = 0
        gl2 = sea & (hsh < pg * 0.8) & ~gl & ~(ramp == RID['foam']) & (ramp != RID['kelp'])
        step[gl2] = np.minimum(step[gl2] + 2, 6)
    return ramp, step, dict(q=q, d=d, Wf=Wf, Ff=Ff, Af=Af, kelp=kelp, brk=brk, sea=sea)

def paint_land(G, S, cam, ramp, step, tide=0.45, sun=(205, 50)):
    L = ~G['sea']
    x, y, z = G['x'], G['y'], G['z']
    rough = sample(S, G, 'rough'); chm = sample(S, G, 'chm')
    saz, sel = np.radians(sun[0]), np.radians(sun[1])
    sv = np.array([np.sin(saz) * np.cos(sel), -np.cos(saz) * np.cos(sel), np.sin(sel)])
    ndl = G['nx'] * sv[0] + G['ny'] * sv[1] + G['nz'] * sv[2]
    slope = 1 - G['nz']
    by = bayer(*x.shape)
    zb = z - chm                                   # bare earth under the canopy
    forest = L & (chm > 3)
    wet = L & ~forest & (zb < tide + 1.6)
    rockm = L & ~forest & ((rough > 0.12) | (slope > 0.25))
    sandm = L & ~forest & ~rockm & (zb < 9)
    grav = L & ~forest & ~rockm & ~sandm
    sh = np.clip(ndl * 1.1 + (by - 0.5) * 0.25, 0, 1)
    def put(m, name, n):
        ramp[m] = RID[name]; step[m] = np.clip((sh[m] * n).astype(int), 0, n - 1)
    put(forest, 'forest', 4)
    # forest: crowns as lit caps at their own scale
    cap = fbmw(x, y, 3.5, 101, 2)
    step[forest] = np.clip(step[forest] + (cap[forest] > 0.62) - (cap[forest] < 0.35), 0, 3)
    put(rockm & ~wet, 'rock', 5); put(rockm & wet, 'wetrock', 3)
    put(sandm & ~wet, 'sand', 4); put(sandm & wet, 'wetsand', 3); put(grav, 'gravel', 4)
    return ramp, step
