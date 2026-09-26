"""
A granite boulder in a grass bank at the edge of a tarn, at dusk, in Ferrari's pixel language.
The tutorial version: everything is built in the picture plane, 320x180, 32 colours.

    python boulder.py                      # paints the base cell into ../img/t_final.png
    python figs.py                         # makes every figure in the tutorial
    paint(size='medium', light='left', distance='10m', seed=1, steps=None) -> (rgb, layers)

The order is the order a painter works in:
  1 planes (picture-plane shapes)      2 value plan (one value per plane, check the notan)
  3 the boulder's light plan           4 colour from material ramps
  5 cluster vocabulary per material    6 contrast inside the stroke (blades, cracks)
  7 pixel geometry (clean runs, flat bases, dark lips)
  8 reflection (the frame crops, the water shows it whole)
"""
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage import measure
from PIL import Image, ImageDraw

W, H = 320, 180
TOP = -130                    # the world goes on above the frame: tall things are cropped, not shrunk
HT = H - TOP                  # canvas rows, canvas row r <-> picture row r + TOP
F = 343.0                     # focal length in px (50 deg horizontal field)
PITCH = np.radians(10.0)      # camera looks 10 degrees down (the task's camera)
HC = 1.1                      # camera height above the water, m

# ---------------------------------------------------------------- palette: 32 colours as material ramps
PAL = {
    'sky0': (190, 158, 200), 'sky1': (206, 162, 188), 'sky2': (222, 162, 174),
    'sky3': (236, 168, 160), 'sky4': (246, 188, 160), 'hot': (252, 212, 158),
    'mtn0': (98, 80, 108), 'mtn1': (134, 102, 124), 'mtn2': (238, 144, 116), 'snow0': (176, 152, 192),
    'for0': (34, 30, 42), 'for1': (84, 70, 64),
    'k': (4, 4, 2), 'g0': (26, 32, 8), 'h1': (46, 46, 26), 'g1': (62, 58, 22), 'h2': (88, 84, 48),
    'g2': (116, 90, 36), 'g3': (160, 116, 58), 'g4': (198, 134, 72), 'g5': (232, 174, 102),
    'sedge': (122, 122, 60),
    'r0': (40, 34, 48), 'r1': (78, 68, 90), 'r2': (120, 104, 120), 'r3': (190, 148, 130), 'r4': (236, 192, 160),
    'w0': (52, 44, 58), 'w1': (160, 136, 170), 'w2': (184, 150, 174), 'w3': (206, 162, 164), 'glint': (226, 202, 216),
}
NAMES = list(PAL); I = {n: i for i, n in enumerate(NAMES)}
RGB = np.array([PAL[n] for n in NAMES], np.uint8)


def lab(c):
    c = np.asarray(c, float) / 255
    l = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = l @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


LAB = lab(RGB); LV = LAB[:, 0]
RAMPS = {'sky': ['sky0', 'sky1', 'sky2', 'sky3', 'sky4', 'hot'],
         'grass_far': ['g0', 'g1', 'g2', 'g3', 'g4'], 'grass_far_sh': ['k', 'g0', 'h1', 'g1'],
         'rock_lit': ['r2', 'r3', 'r4'], 'rock_sh': ['r0', 'r1', 'r2']}
RAMPS = {k: np.array([I[n] for n in v]) for k, v in RAMPS.items()}

# direction TO the sun in camera space: x right, y up, z toward the viewer
LIGHTS = {'left': (-0.90, 0.30, 0.30), 'right': (0.90, 0.30, 0.30),
          'front': (-0.35, 0.30, 0.89), 'back': (0.25, 0.35, -0.90)}
SIZES = {'small': 0.55, 'medium': 0.95, 'large': 1.45}          # boulder radius, m
DISTS = {'close': 4.5, '10m': 10.0, '40m': 40.0}


def ground_row(d, pitch=PITCH):
    """picture row of a point on the water plane d metres away"""
    return H / 2 + F * np.tan(np.arctan(HC / d) - pitch)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)


def rank(a):
    """rank-normalise to a uniform [0,1) threshold map: thresholds then equal area fractions"""
    r = np.empty(a.size); r[np.argsort(a, axis=None)] = np.arange(a.size); return (r / a.size).reshape(a.shape)


def ramp_pick(ramp, L, T, w):
    """value L -> two neighbouring ramp colours, mixed by threshold map T (w=0 hard bands, 1 full dither)"""
    idx = RAMPS[ramp]; pos = np.interp(np.nan_to_num(L, nan=30.0), LV[idx], np.arange(len(idx)))
    lo = np.clip(np.floor(pos).astype(int), 0, len(idx) - 1); hi = np.clip(lo + 1, 0, len(idx) - 1)
    return np.where(pos - np.floor(pos) > 0.5 + (T - 0.5) * w, idx[hi], idx[lo])


# ---------------------------------------------------------------- clean pixel geometry
_B = [(1, 0), (5, 1), (4, 1), (3, 1), (2, 1), (1, 1), (1, 2), (1, 3), (1, 4), (0, 1)]
def _dirs(base):
    D = sorted({(dx * sx, dy * sy) for dx, dy in base for sx in (1, -1) for sy in (1, -1)})
    D = np.array(D, float); return D / np.linalg.norm(D, axis=1, keepdims=True)


def clean_mask(mask, tol=0.9, slopes=_B):
    """Re-draw a silhouette as straight runs with clean steps: simplify the contour to a polygon,
    snap each edge to the nearest clean slope, solve the vertices so the polygon still closes."""
    UD = _dirs(slopes); Hh, Ww = mask.shape
    out = Image.new('L', (Ww, Hh), 0); d = ImageDraw.Draw(out)
    for c in measure.find_contours(np.pad(mask.astype(float), 1), 0.5):
        c = c[:, ::-1] - 1
        if len(c) < 8: continue
        P = measure.approximate_polygon(c, tolerance=tol)[:-1]
        if len(P) < 3: continue
        n = len(P); E = np.roll(P, -1, 0) - P
        D = UD[np.argmax((E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)) @ UD.T, 1)]
        A = np.zeros((2 * n + 2, 2 + n)); b = np.zeros(2 * n + 2)
        for i in range(n):
            A[2 * i, 0] = A[2 * i + 1, 1] = 1
            A[2 * i, 2:2 + i] = D[:i, 0]; A[2 * i + 1, 2:2 + i] = D[:i, 1]
            b[2 * i], b[2 * i + 1] = P[i]
        A[2 * n, 2:] = 30 * D[:, 0]; A[2 * n + 1, 2:] = 30 * D[:, 1]      # the polygon must close
        s = np.linalg.lstsq(A, b, rcond=None)[0]
        V = [s[:2]]
        for j in range(n - 1): V.append(V[-1] + s[2 + j] * D[j])
        d.polygon([tuple(v) for v in np.round(V).astype(int)], fill=1)
    return np.array(out) > 0


# ---------------------------------------------------------------- the painter
def paint(size='medium', light='left', distance='10m', seed=1, steps=None, traps=(), pitch_deg=10.0):
    """steps: stop after this step number (1..8) and return what's there. traps: names of mistakes to
    re-introduce for the before/after pictures ('lumpy_terminator', 'checker_water', 'boxy', 'short_strokes',
    'no_grove', 'nearest_colour')."""
    rng = np.random.default_rng(seed)
    Lsun = np.array(LIGHTS[light], float); Lsun /= np.linalg.norm(Lsun)
    sx = -1 if Lsun[0] < -0.2 else (1 if Lsun[0] > 0.2 else 0)        # the screen side the sun is on
    back = light == 'back'
    yy, xx = np.mgrid[TOP:H, 0:W].astype(float)                        # canvas coordinates (picture rows)
    L = np.full((HT, W), 30.0)                                       # target value per pixel (step 2)
    P = np.zeros((HT, W), int)          # planes: 0 sky 1 mountain 2 far forest 3 water 4 bank 5 boulder 6 grove 7 near bank 8 framing trees
    out = {}

    # ===== 1 PLANES, drawn in the picture plane =================================================
    pitch = np.radians(pitch_deg)
    Y_S = int(round(ground_row(80.0, pitch)))          # far shoreline row (a shore ~80 m away)
    d = DISTS[distance]; r_m = SIZES[size]
    y_w = ground_row(d, pitch)                                   # the boulder's waterline row
    rx = F * r_m / d; ry = 0.78 * rx                      # boulder half-width / half-height in px
    xb = W * 0.36 + rng.uniform(-6, 6)                     # boulder a third in from the left
    yc = y_w - 0.45 * ry                                   # ~35% of it sunk below the bank / water line
    # the mountain: a peak taller than the frame; the frame crops it and the water shows it whole
    px = W * 0.58 + rng.uniform(-12, 12)
    knots = np.linspace(-20, W + 20, 18); knots[1:-1] += rng.uniform(-6, 6, 16)
    kv = Y_S - 2 - 60 * np.maximum(0, 1 - np.abs(knots - px) / 115) ** 1.3 - 5 * rng.random(18) * (np.abs(knots - px) < 115)
    ridge = np.interp(np.arange(W), knots, kv)                          # straight segments between knots
    P[(yy < Y_S) & (yy >= ridge[None, :])] = 1
    # far forest: conifers standing on the far shore AT THEIR TRUE SIZE. A 20 m spruce 80 m away is
    # F*20/80 = 86 px tall -- the frame crops it. A notch (a clearing, a farther shore) lets the peak through.
    ftop = np.full(W, Y_S - 3.0); xi = np.arange(W)
    own = np.zeros((HT, W, 2)) + [-99, 0]            # which tree (x0, tip) owns each pixel
    for x0 in np.cumsum(rng.uniform(3, 9, 80)) - 6:
        h_m = rng.uniform(4, 9) if 'toy_forest' in traps else rng.uniform(12, 28)
        notch = np.exp(-((x0 - px) / 55) ** 2)
        h = F * h_m / 80 * (1 - 0.85 * notch) * (0.55 if rng.random() < 0.3 else 1)
        tip = Y_S - 2 - h
        dyt = yy[:, 0] - tip                                       # rows below the tip
        tier = (dyt / (2.5 + 0.06 * np.clip(dyt, 0, None))) % 1
        halfw = np.clip(dyt, 0, None) * 0.22 * (0.55 + 0.7 * tier) + 0.5
        top_here = np.where((np.abs(xi[None, :] - x0) < halfw[:, None]) & (dyt[:, None] > 0), yy, 999).min(0)
        inside = (np.abs(xi[None, :] - x0) < halfw[:, None]) & (dyt[:, None] > 0)
        own[inside & (own[..., 0] == -99)] = (x0, tip)
        ftop = np.minimum(ftop, top_here)
    P[(yy < Y_S) & (yy >= ftop[None, :])] = 2
    # the boulder's bank: land from the far shore down to a waterline that rises gently to the left
    xtip = xb + 0.9 * rx
    yw_x = y_w + 0.10 * np.maximum(0, xb - np.arange(W)) + 1.5 * np.sin(np.arange(W) / 11 + seed)
    tipcut = xtip - (xtip - (xb - rx - 25)) * np.clip((y_w - yy) / max(1, y_w - Y_S), 0, 1) ** 0.7
    bank = (yy >= Y_S) & (yy < yw_x[None, :]) & (xx < tipcut)
    P[bank] = 4
    P[(yy >= Y_S) & ~bank] = 3
    # the boulder: an ellipse with weathered lumps in its outline
    ang = np.arctan2((yy - yc) / ry, (xx - xb) / rx)
    lump = 1 + 0.06 * np.sin(3 * ang + seed) + 0.04 * np.sin(5 * ang + 2 * seed) + 0.03 * np.sin(8 * ang + seed * 3)
    rr = np.hypot((xx - xb) / rx, (yy - yc) / ry)
    boulder = (rr < lump) & (yy < y_w + 0.5)
    if 'boxy' in traps:
        boulder = clean_mask(boulder, tol=2.5, slopes=[(1, 0), (1, 1), (0, 1)])
    else:
        boulder = clean_mask(boulder, tol=0.9)
    boulder &= yy < y_w + 0.5                                           # a dead-flat base at the waterline
    P[boulder] = 5
    # the grove: tall conifers behind the boulder on its bank, cropped by the frame, whole in the water
    trees = []
    if 'no_grove' not in traps:
        for k in range(int(rng.integers(4, 7))):
            tx = xb - rx * rng.uniform(0.8, 1.2) - rng.uniform(5, 110)
            tb = Y_S + (y_w - Y_S) * rng.uniform(0.15, 0.45)             # stands back on the bank
            th = (tb - TOP) * rng.uniform(0.55, 1.0)
            trees.append((tx, tb, th, min(th * rng.uniform(0.09, 0.12), 26), 3))
    tree_id = np.full((HT, W), -1)
    for i, (tx, tb, th, tw, _) in enumerate(trees):
        dy = tb - yy                                                    # height above the tree's base
        fr = np.clip(dy / th, 0, 1)
        tier_len = 3 + 0.05 * dy
        tier = ((th - dy - 0.45 * np.abs(xx - tx)) / tier_len) % 1.0     # 0 at a tier's top, 1 at its drooping skirt
        tj = np.floor((th - dy) / tier_len)
        jag = 0.75 + 0.35 * np.sin(tj * 2.7 + i * 5 + np.sign(xx - tx) * 1.3)
        halfw = tw * (1 - fr) * (0.45 + 0.8 * tier ** 1.3) * jag + 0.6
        inside = (dy > 2) & (dy < th) & (np.abs(xx - tx) < halfw)
        trunk = (dy >= -1) & (dy <= 3) & (np.abs(xx - tx) < 1.0)
        m = (inside | trunk) & (P != 5)
        tree_id[m] = i; P[m] = 6
    # the near banks: a V opening onto the water, designed as two lines in the picture
    k = {'close': 0.55, '10m': 0.85, '40m': 1.0}[distance]
    u = np.arange(W, dtype=float)
    wob = 3 * np.sin(u / 17 + seed) + 2 * np.sin(u / 7.3 + 2 * seed)
    e_left = H * (1 - 0.40 * k) + u / ((0.12 + 0.30 * k) * W) * (H * 0.40 * k)
    e_right = H + 4 - (u - (1 - 0.50 * k) * W) / (0.50 * k * W) * (H * 0.66 * k)
    edge = np.minimum(np.minimum(e_left, e_right), H + 6) + wob
    near = yy >= edge[None, :]
    P[near] = 7
    # framing trees on the near bank opposite the boulder: Ferrari's repoussoir, the frame's dark side
    if 'no_grove' not in traps:
        xs_f = (W - rng.uniform(8, 60, int(rng.integers(2, 4)))) if xb < W / 2 else rng.uniform(8, 60, 2)
        for j, tx in enumerate(xs_f):
            tb = min(H + 10, edge[int(np.clip(tx, 0, W - 1))] + rng.uniform(10, 30))
            th = tb - TOP; tw = rng.uniform(26, 40)
            dy = tb - yy; fr = np.clip(dy / th, 0, 1)
            tier_len = 7 + 0.06 * dy; tier = ((th - dy - 0.45 * np.abs(xx - tx)) / tier_len) % 1.0; tj = np.floor((th - dy) / tier_len)
            jag = 0.75 + 0.35 * np.sin(tj * 2.7 + j * 5 + np.sign(xx - tx) * 1.3)
            m = (dy > 0) & (np.abs(xx - tx) < tw * (1 - fr) * (0.45 + 0.8 * tier ** 1.3) * jag + 1.5)
            trees.append((tx, tb, th, tw, 7)); tree_id[m] = len(trees) - 1; P[m] = 8
    near_shade = near & ((xx > W * 0.5) if sx <= 0 else (xx < W * 0.5))  # the side away from the sun lies in
    if back: near_shade = near.copy()                                    # the shadow of off-frame trees
    if sx == 0 and not back: near_shade = near & False
    # where each water pixel looks in the mirror: standing things mirror about their own waterline,
    # the far world about the far shoreline (a flat bank shows only its lip)
    rows = np.arange(HT)
    ax_bank = np.where(np.arange(W) < tipcut[-1] + 2 * rx, yw_x, y_w) - TOP  # axis for the bank layer (canvas rows)
    src_b = np.clip(np.round(2 * ax_bank[None, :] - 1 - rows[:, None]).astype(int), 0, HT - 1)
    src_f = np.broadcast_to(np.clip(2 * (Y_S - TOP) - 1 - rows[:, None], 0, HT - 1), (HT, W))   # mirror about the shoreline
    colsx = np.broadcast_to(np.arange(W), (HT, W))
    layer_b = (np.isin(P[src_b, colsx], (5, 6)) | ((P[src_b, colsx] == 4) & (ax_bank[None, :] - src_b < 2))) & (src_b < rows[:, None])
    def mirror(A):
        return np.where(layer_b, A[src_b, colsx], A[src_f, colsx])
    out['planes'] = P.copy()
    if steps == 1: return _finish(None, P, out, planes_only=True)

    # ===== 2 VALUE PLAN: one value per plane, from Ferrari's measured groups (L*) =================
    shadow_len = 2.6 * rx
    cast = (P == 4) & (np.abs(yy - (y_w - 0.15 * ry)) < 0.30 * ry * (1 - np.clip(-sx * (xx - xb) / shadow_len, 0, 1))) \
        & (-sx * (xx - xb) > 0) & (-sx * (xx - xb) < shadow_len) if sx else np.zeros_like(bank)
    if back:
        cast = (P == 4) & (np.abs(xx - xb) < rx * (1 + 0.6 * np.clip((y_w - yy) / ry, 0, 2))) & (yy > y_w - 2 * ry) & (yy < y_w)
    plan = {0: 73, 1: 52, 2: 17, 4: 50, 5: 40, 6: 10, 7: 44, 8: 8, 3: 30}
    for p_, v in plan.items(): L[P == p_] = v
    far_fr = np.clip((y_w - yy) / max(1, y_w - Y_S), 0, 1)                # 0 at the waterline, 1 at the far shore
    L[P == 4] = 48 + 8 * far_fr[P == 4]                                   # ground planes brighten as they recede
    L[cast] = 14; L[near_shade] = 14
    if back: L[P == 4] = 24 + 6 * far_fr[P == 4]
    Lm = mirror(L)
    L[P == 3] = (Lm - 0.6 * np.maximum(0, Lm - 68) - 0.06 * np.maximum(0, Lm - 30))[P == 3]
    out['value_plan'] = L.copy()

    # ===== 3 THE BOULDER'S LIGHT PLAN ==========================================================
    nx = (xx - xb) / rx; ny = -(yy - yc) / ry
    nz = np.sqrt(np.clip(1 - nx ** 2 - ny ** 2, 0.02, 1))
    n_big = np.stack([nx, ny, nz], -1); n_big /= np.linalg.norm(n_big, axis=-1, keepdims=True)
    bumps = gaussian_filter(rng.standard_normal((HT, W)), max(1.0, rx / 7))
    bumps = bumps / (bumps.std() + 1e-9)
    gy_, gx_ = np.gradient(bumps)
    n_lump = n_big + 0.9 * np.stack([-gx_, gy_, 0 * gx_], -1)
    n_lump /= np.linalg.norm(n_lump, axis=-1, keepdims=True)
    ndl_big = n_big @ Lsun; ndl = 0.7 * ndl_big + 0.3 * (n_lump @ Lsun)
    n_bad = n_big + 2.5 * np.stack([-gx_, gy_, 0 * gx_], -1); n_bad /= np.linalg.norm(n_bad, axis=-1, keepdims=True)
    term = (n_bad @ Lsun) if 'lumpy_terminator' in traps else ndl_big    # the terminator comes from the BIG form
    lit = term > 0.04
    core = np.exp(-((term + 0.02) / 0.12) ** 2)
    sky_up = np.clip(n_big[..., 1], 0, 1); bounce = np.clip(-n_big[..., 1], 0, 1)
    contact = smoothstep(0, 0.35 * ry, y_w - yy)
    Lb_lit = (44 + 34 * smoothstep(0.05, 0.85, ndl)) * (0.82 + 0.18 * nz)       # half-tone / light / highlight
    Lb_sh = (24 + 10 * sky_up + 6 * bounce - 13 * core) * (0.55 + 0.45 * contact)  # shadow / core / reflected
    bm = P == 5
    L[bm] = np.where(lit, Lb_lit, Lb_sh)[bm]
    out['value_full'] = L.copy()
    if steps in (2, 3): return _finish(None, P, out, value=L)

    # ===== 4 + 5 COLOUR from material ramps, laid in each material's cluster vocabulary ===========
    img = np.zeros((HT, W), int)
    check = ((xx + yy) % 2) * 0.5 + 0.25
    speck = rng.random((HT, W))
    vert = rank(gaussian_filter(rng.random((HT, W)), (4.5 if 'short_strokes' not in traps else 0.8, 0.35)))
    horiz = rank(gaussian_filter(rng.random((HT, W)), (0.5, 6.0)))
    diag = rank(gaussian_filter(rng.random((HT, W)), (0.6, 1.6)) + 0.2 * np.sin((xx - (sx or 1) * yy) / 2.2))
    # sky: flat bands with checker seams; the warm end on the sun's side, near the horizon
    e = np.clip((Y_S - 6 - yy) / 150, 0, 1.5)
    pos = 3.9 - 3.5 * e ** 0.8 + 0.8 * (sx * (xx - W / 2) / W if sx else 0) * np.exp(-e * 2)
    if back: pos = pos + 1.0 * np.exp(-e * 1.2)
    pos = np.clip(pos, 0, 5)
    lo = np.floor(pos).astype(int); fr = pos - lo
    sky = np.where(fr > 0.5 + (check - 0.5) * 0.32, RAMPS['sky'][np.clip(lo + 1, 0, 5)], RAMPS['sky'][lo])
    img[P == 0] = sky[P == 0]
    # mountain: lit flanks warm, shade flanks violet; spurs along the fall line; lifted darks (>= L37)
    wig = 6 * gaussian_filter(rng.standard_normal((HT, W)), 5) / 0.03
    left_flank = xx < px + 0.3 * wig
    fall = np.where(left_flank, xx + yy, xx - yy)                   # spurs run down the fall line
    spur = np.sin(fall / 11 + 0.05 * wig)
    sun_flank = left_flank if sx <= 0 else ~left_flank
    m_lit = np.where(sun_flank, spur < 0.75, spur > 0.9)             # lit flank with shaded gullies, and lit ribs
    if back: m_lit = spur > 0.97
    if light == 'front': m_lit = spur > -0.5
    snow = (yy < ridge[None, :] + 12 + 6 * np.sin(xx / 9) * np.sin(xx / 23)) & (spur < 0.3) & (yy < Y_S - 12)
    mcol = np.where(m_lit, np.where(diag > 0.35, I['mtn2'], I['mtn1']), np.where(diag > 0.7, I['mtn1'], I['mtn0']))
    mcol = np.where(snow, np.where(m_lit, np.where(check > 0.5, I['sky4'], I['sky3']), I['snow0']), mcol)
    crest = (yy - ridge[None, :] < 1.0) & m_lit
    mcol = np.where(crest, I['mtn2'] if not back else I['hot'], mcol)
    img[P == 1] = mcol[P == 1]
    # far forest: black and violet-black in horizontal tier strokes, ochre flecks on the sun side
    # (80 m away: darks lifted to violet-black, flecks duller than the near grove's -- the depth rule)
    fx, ftip = own[..., 0], own[..., 1]
    ftier = ((yy - ftip) - 0.3 * np.abs(xx - fx)) / (2.5 + 0.06 * np.clip(yy - ftip, 0, None)) % 1
    fcol = np.where((ftier < 0.22) | (speck < 0.12), I['k'], I['for0'])      # a dark line under each drooping tier
    ff = ((xx - fx) * (sx if sx else 1) > 0.5) & (ftier > 0.45) & (speck > 0.25)
    if back: ff = (np.abs(xx - fx) > 0.18 * (yy - ftip)) & (ftier > 0.7) & (speck > 0.5)
    fcol = np.where(ff, np.where(speck > 0.9, I['g2'], I['for1']), fcol)
    img[P == 2] = fcol[P == 2]
    # grass: Ferrari's measured recipe -- a base broken by strokes in fixed proportions (lit: base 38 /
    # mid stroke 35 / olive 11 / gap 12 %; shade: black 28 / black-green 31 / olive 40 %), thresholded on a
    # vertically stretched map so the proportions come out as vertical 1-px strokes
    Lg = L
    def grass(Lt, Tv, shade):
        if shade:
            s = np.clip((Lt - 12) / 30, -0.3, 0.3)
            return np.select([Tv < 0.28 - s, Tv < 0.60 - s], [I['k'], I['g0']], np.where(Tv > 0.9 - s / 2, I['h2'], I['g1']))
        s = np.clip((Lt - 50) / 40, -0.3, 0.3)
        return np.select([Tv < 0.12 - s * 0.3, Tv < 0.23 - s * 0.4, Tv < 0.58 - s],
                         [I['g0'], I['g1'], I['g2']], np.where(Lt > 50, I['g4'], I['g3']))
    in_shade = cast | near_shade | ((P == 4) & back)
    g_near = np.where(in_shade, grass(Lg, vert, True), grass(Lg, vert, False))
    # far grass shrinks to a two-colour speckle
    g_far = np.where(in_shade, ramp_pick('grass_far_sh', Lg + 6 * (speck * 2 - 1), check, 0.3),
                     ramp_pick('grass_far', Lg + 6 * (speck * 2 - 1), check, 0.3))
    use_near = (P == 7) | ((P == 4) & (far_fr < 0.45 + 0.3 * speck))
    gm = (P == 4) | (P == 7)
    img[gm] = np.where(use_near, g_near, g_far)[gm]
    # the boulder: warm ramp in light, cool ramp in shade, strokes along the fall line + checker seams
    Tm = 0.35 * diag + 0.65 * check
    spread = 5 * (diag * 2 - 1) * (rx > 20)
    b_col = np.where(lit, ramp_pick('rock_lit', L + spread, Tm, 0.45), ramp_pick('rock_sh', L + 0.8 * spread, Tm, 0.45))
    if 'nearest_colour' in traps:                                        # the whole palette, no ramps
        tgt = np.nan_to_num(L)
        b_col = np.argmin(np.abs(tgt[..., None] - LV[None, None, :]), -1)
    img[bm] = b_col[bm]
    # the grove: black mass, ochre flecks on the sun side of each drooping tier (black 61 / ochre 22 / dark 14 %)
    for i, (tx, tb, th, tw, tl0) in enumerate(trees):
        m = (tree_id == i) & np.isin(P, (6, 8))
        dy = tb - yy; tier = ((th - dy - 0.45 * np.abs(xx - tx)) / (tl0 + 0.05 * dy)) % 1.0
        side = (xx - tx) * (sx if sx else 1)
        fleck = m & (side > 1) & (tier > 0.62) & (tier < 0.9) & (speck > 0.35)
        if back: fleck = m & (np.abs(xx - tx) > (tw * (1 - np.clip(dy / th, 0, 1))) * 0.5) & (tier > 0.7) & (speck > 0.6)
        img[m] = np.where(speck[m] < 0.16, I['for0'], I['k'])
        img[fleck] = np.where(speck[fleck] > 0.85, I['g3'], I['g2'])
    out['colour'] = img.copy()
    if steps in (4, 5): return _finish(img, P, out)

    # ===== 6 CONTRAST INSIDE THE STROKE: cracks on the rock, blades in the grass ===================
    if rx > 14:                                                          # cracks: a dark line, lit lip on the sun side
        for c in range(max(1, int(rx / 25))):
            x0 = xb + rng.uniform(-0.6, 0.6) * rx; y0 = yc + rng.uniform(-0.6, 0.4) * ry
            dx_, dy_ = [(1, 1), (2, 1), (1, 2), (3, 1)][rng.integers(4)]; sgn = rng.choice([-1, 1])
            for t in range(int(rng.uniform(0.12, 0.25) * rx)):
                x = int(round(x0 + sgn * t * dx_ / max(dx_, dy_))); y = int(round(y0 + t * dy_ / max(dx_, dy_))) - TOP
                if 0 <= y < HT and 0 <= x < W and P[y, x] == 5:
                    img[y, x] = I['r0']
                    xl = x - (sx if sx else -1)
                    if 0 <= xl < W and P[y, xl] == 5 and lit[y, xl]: img[y, xl] = I['r4']
    # blades: straight runs with one change of slope, painted back to front, in counterchange
    ys, xs = np.nonzero((P == 4) | (P == 7))
    wy = ys + TOP
    near_water = np.abs(wy - yw_x[xs]) < 5
    near_foot = (np.abs(xs - xb) < rx * 1.4) & (wy > y_w - 0.6 * ry - 3) & (wy < y_w + 2)
    near_top = (P[ys, xs] == 7) & (wy - edge[xs] < 4)
    blen = np.where(P[ys, xs] == 7, 4 + 0.10 * (wy - 100).clip(0), np.where(near_foot, 4 + 0.35 * ry, 2 + 6 * np.clip(rx / 60, 0, 1)))
    dens = np.where(near_foot, 0.45, np.where(near_top, 0.35, np.where(near_water, 0.25, 0.012)))
    pick = rng.random(len(ys)) < dens
    order = np.argsort(wy[pick], kind='stable')
    S = [0.0, 1 / 3, 1 / 2, 1.0]
    for j in np.flatnonzero(pick)[order]:
        y0, x0 = ys[j], xs[j]
        Lb = int(np.clip(blen[j] * rng.uniform(0.6, 1.4), 2, 24))
        lean = rng.normal(0, 0.8); sg = 1 if lean >= 0 else -1
        k0 = int(np.clip(abs(lean) * 1.6, 0, 2)); k1 = min(3, k0 + (Lb > 5))
        split = int(Lb * rng.uniform(0.45, 0.75))
        blit = not in_shade[y0, x0]
        x = float(x0); acc = 0.0
        for t in range(Lb):
            y = y0 - t
            if y < 0: break
            acc += S[k0] if t < split else S[k1]
            if acc >= 1: x += sg * np.floor(acc); acc -= np.floor(acc)
            xi = int(x)
            if not 0 <= xi < W: break
            tip = t / max(1, Lb - 1); over = P[y, xi] not in (4, 7)
            if LV[img[y, xi]] > 42:                       # on light ground: a dark blade with its lit edge
                img[y, xi] = I['g0'] if (over or not blit or rng.random() < 0.35) else I['g2']
                if blit and sx and 0 <= xi + sx < W and not over and tip > 0.3:
                    img[y, xi + sx] = I['g4'] if not back else I['g5']
            else:                                         # on dark ground: a light blade, lighter at the tip
                if blit: c = I['g2'] if tip < 0.4 else (I['g3'] if tip < 0.75 else I['g4'])
                else: c = I['g0'] if tip < 0.3 else (I['h1'] if tip < 0.6 else I['g1'])
                if back and tip > 0.7: c = I['g5']
                if over and not blit: c = I['g1'] if tip < 0.6 else I['sedge']
                img[y, xi] = c
                if sx and blit and 0 <= xi - sx < W and tip > 0.2 and not over:
                    img[y, xi - sx] = I['g0']             # ...with its shadow side
    # a back-lit boulder gets a one-pixel rim
    if back:
        rim = bm & ~(np.roll(bm, 1, 0) & np.roll(bm, 1, 1) & np.roll(bm, -1, 1)) & (n_big[..., 1] > -0.2)
        img[rim] = I['r4']
    out['strokes'] = img.copy()
    if steps == 6: return _finish(img, P, out)

    # ===== 7 PIXEL GEOMETRY: flat dark contact lines where masses meet water =======================
    water = P == 3
    lip = np.zeros_like(water); lip[1:] = water[1:] & ~water[:-1] & ((P[:-1] == 4) | (P[:-1] == 5))
    img[lip & (np.roll(P, 1, 0) == 5)] = I['r0']
    img[lip & (np.roll(P, 1, 0) != 5)] = I['g0']
    out['geometry'] = img.copy()
    if steps == 7: return _finish(img, P, out)

    # ===== 8 REFLECTION: an exact mirror, range compressed; ripples horizontal ====================
    srcimg = mirror(img)
    lab_ = LAB[srcimg].copy(); Lr = lab_[..., 0]
    lab_[..., 0] = Lr - 0.6 * np.maximum(0, Lr - 68) - 0.06 * np.maximum(0, Lr - 30) + 1.0 * (Lr < 25)
    lab_[..., 1] -= 4.0
    allowed = np.array([I[n] for n in ['k', 'for0', 'w0', 'r0', 'r1', 'mtn0', 'mtn1', 'r2', 'h1', 'g0', 'g1', 'g2', 'g3',
                                        'w1', 'w2', 'w3', 'sky1', 'sky2', 'r3', 'mtn2', 'snow0']])
    dd = np.linalg.norm(lab_[..., None, :] - LAB[allowed][None, None], axis=-1)
    o2 = np.argsort(dd, -1)
    c1, c2 = allowed[o2[..., 0]], allowed[o2[..., 1]]
    d1 = np.take_along_axis(dd, o2[..., :1], -1)[..., 0]; d2 = np.take_along_axis(dd, o2[..., 1:2], -1)[..., 0]
    frac = d1 / (d1 + d2 + 1e-9)
    Tr = check if "checker_water" in traps else np.where(np.abs(frac - 0.5) < 0.04, check, horiz)
    wcol = np.where(frac > 0.5 + (Tr - 0.5) * 0.9, c2, c1)
    # sparse 1-px streak rows, denser toward the far shore (the rows Ferrari would colour-cycle)
    rowseed = rng.random(HT)
    streak = (rowseed[:, None] < 0.22 * np.exp(-(yy - Y_S) / 25)) & (np.sin(xx * 0.21 + rowseed[:, None] * 40) > 0.6)
    wcol = np.where(streak & (LV[wcol] > 50), I['w3'], np.where(streak & (LV[wcol] <= 50), I['mtn0'], wcol))
    img[water] = wcol[water]
    img[lip & (np.roll(P, 1, 0) == 5)] = I['r0']
    img[lip & (np.roll(P, 1, 0) != 5)] = I['g0']
    # the near banks go on last, over the water
    nb = P == 7
    img[nb] = out['strokes'][nb]
    return _finish(img, P, out)


def _finish(img, P, out, planes_only=False, value=None):
    crop = slice(-TOP, None)
    out = {k: v[crop] for k, v in out.items()}
    if planes_only or img is None:
        rgb = None
    else:
        rgb = RGB[img[crop]]
    out['P'] = P[crop]
    return rgb, out


# ---------------------------------------------------------------- the check: the same measure as on Ferrari
def notan_shares(rgb, cuts=(28.9, 55.6)):
    """Ferrari's own 3-value cuts on a squinted L*; Ferrari himself is 51 / 27 / 21 % (dark / mid / light)"""
    Lb = gaussian_filter(lab(rgb)[..., 0], 2.5 * rgb.shape[1] / 634)
    q = np.digitize(Lb, cuts)
    return [float((q == i).mean()) for i in range(3)], q


if __name__ == '__main__':
    import os
    os.makedirs('../img', exist_ok=True)
    rgb, lay = paint()
    Image.fromarray(np.repeat(np.repeat(rgb, 3, 0), 3, 1)).save('../img/t_final.png')
    print(notan_shares(rgb)[0])
