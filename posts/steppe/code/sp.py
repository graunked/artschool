"""sp: the steppe painter's shared core.

An index canvas (material, step). Form lives in the steps; light, season and hour live in the
palette. Narrow forms are drawn as clean runs. Everything else in this folder paints into it.

Conventions: arrays are [y, x], y grows downward.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ materials
(SKY, GROUND, CRUST, STONE, BASALT, SAGE, SAGESTEM, GRASS, BITTER, NEEDLE, BARK,
 FAR, SNOW, FLOWER, SHADOWGROUND, HAZE, WATER, FORB) = range(1, 19)
NAMES = {v: k for k, v in dict(SKY=SKY, GROUND=GROUND, CRUST=CRUST, STONE=STONE, BASALT=BASALT,
                               SAGE=SAGE, SAGESTEM=SAGESTEM, GRASS=GRASS, BITTER=BITTER,
                               NEEDLE=NEEDLE, BARK=BARK, FAR=FAR, SNOW=SNOW, FLOWER=FLOWER,
                               HAZE=HAZE, WATER=WATER, FORB=FORB).items()}


class Canvas:
    """Per pixel: material id and step on that material's ramp. Band = depth band (0 near)."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.mat = np.zeros((h, w), np.uint8)
        self.step = np.zeros((h, w), np.int16)
        self.band = np.zeros((h, w), np.uint8)
        self.depth = np.full((h, w), 1e9, np.float32)   # optional z-buffer (smaller = nearer)

    def px(self, x, y, mat, step, band=None, z=None):
        x, y = int(x), int(y)
        if 0 <= x < self.w and 0 <= y < self.h:
            if z is not None:
                if z > self.depth[y, x]:
                    return
                self.depth[y, x] = z
            self.mat[y, x] = mat
            self.step[y, x] = step
            if band is not None:
                self.band[y, x] = band

    def put(self, mask, mat, step, band=None):
        self.mat[mask] = mat
        self.step[mask] = (np.asarray(step)[mask] if np.ndim(step) else step)
        if band is not None:
            self.band[mask] = band

    def blit(self, other, ox, oy, keep=None):
        """Paste another canvas where its material is nonzero."""
        h, w = other.h, other.w
        y0, x0 = max(0, oy), max(0, ox)
        y1, x1 = min(self.h, oy + h), min(self.w, ox + w)
        if y1 <= y0 or x1 <= x0:
            return
        sub = (slice(y0 - oy, y1 - oy), slice(x0 - ox, x1 - ox))
        m = other.mat[sub] > 0
        for a, b in ((self.mat, other.mat), (self.step, other.step), (self.band, other.band)):
            a[y0:y1, x0:x1][m] = b[sub][m]

    def render(self, pal):
        """pal: Palette-like: pal.rgb(mat, step, band) -> (3,) or dict {(mat,step): rgb}."""
        out = np.zeros((self.h, self.w, 3), np.uint8)
        key = (self.mat.astype(np.int64) * 4096 + self.band.astype(np.int64) * 256
               + np.clip(self.step, 0, 255).astype(np.int64))
        for k in np.unique(key):
            m, r = divmod(int(k), 4096)
            b, s = divmod(r, 256)
            out[key == k] = pal.rgb(m, s, b)
        return out


# ------------------------------------------------------------------ helpers
def value_noise(shape, scale, rng, octaves=1):
    h, w = shape
    out = np.zeros(shape)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        s = max(1.0, scale / (2 ** o))
        gh, gw = int(h / s) + 4, int(w / s) + 4
        g = rng.random((gh, gw))
        Y, X = np.meshgrid(np.arange(h) / s, np.arange(w) / s, indexing="ij")
        out += amp * ndi.map_coordinates(g, [Y, X], order=3, mode="reflect")
        tot += amp
        amp *= 0.5
    return out / tot


def upscale(img, k):
    return np.repeat(np.repeat(img, k, 0), k, 1)


def save(img, path, k=1):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(upscale(img, k) if k > 1 else img).save(path)


BAYER2 = np.array([[0, 2], [3, 1]]) / 4.0 + 0.125
BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0 + 1 / 32


def quantize(x, ordered=True, rng=None, bayer=BAYER2):
    """Continuous step coordinate -> integer steps, dithering only between adjacent steps."""
    h, w = x.shape
    f = np.floor(x)
    frac = x - f
    if ordered:
        t = np.tile(bayer, (h // bayer.shape[0] + 1, w // bayer.shape[1] + 1))[:h, :w]
    else:
        t = (rng or np.random.default_rng(0)).random((h, w))
    return (f + (frac > t)).astype(np.int16)


# ------------------------------------------------------------------ clean runs for narrow forms
RATIOS = [0, 0.25, 1 / 3, 0.5, 1.0, 2.0, 3.0, 4.0, 1e9]


def clean_line(x0, y0, x1, y1):
    """A narrow stroke as a clean run: slope snapped to 0,1:4,1:3,1:2,1:1,2:1,3:1,4:1,inf;
    steps are equal-length runs."""
    dx, dy = x1 - x0, y1 - y0
    if max(abs(dx), abs(dy)) < 0.5:
        return [(int(round(x0)), int(round(y0)))]
    sx, sy = (1 if dx >= 0 else -1), (1 if dy >= 0 else -1)
    r = abs(dy) / max(abs(dx), 1e-9)
    r = min(RATIOS, key=lambda q: abs(np.log((q + 0.05) / (r + 0.05))))
    pts = []
    x, y = int(round(x0)), int(round(y0))
    if r >= 1:
        run = int(round(r)) if r < 1e8 else 10 ** 6
        for i in range(int(round(abs(dy))) + 1):
            pts.append((x, y))
            y += sy
            if run < 10 ** 6 and (i + 1) % run == 0:
                x += sx
    else:
        run = int(round(1 / r)) if r > 0 else 10 ** 6
        for i in range(int(round(abs(dx))) + 1):
            pts.append((x, y))
            x += sx
            if run < 10 ** 6 and (i + 1) % run == 0:
                y += sy
    return pts


def polyline_clean(pts):
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        seg = clean_line(*a, *b)
        if out and seg:
            # continue from where the previous segment actually ended
            ex, ey = out[-1]
            ddx, ddy = ex - seg[0][0], ey - seg[0][1]
            seg = [(x + ddx, y + ddy) for x, y in seg][1:]
        out += seg
    return out


def arc_points(x0, y0, ang0, length, bend, nseg=4):
    """A curving narrow form: starts at (x0,y0) heading ang0 (radians from straight up, +right),
    total length in px, bend = total change of angle over the length (radians). Returns
    control points (floats)."""
    pts = [(x0, y0)]
    a = ang0
    seg = length / nseg
    x, y = x0, y0
    for i in range(nseg):
        a_mid = a + bend / nseg * 0.5
        x += np.sin(a_mid) * seg
        y -= np.cos(a_mid) * seg
        a += bend / nseg
        pts.append((x, y))
    return pts


# ------------------------------------------------------------------ OKLab
def srgb_to_lin(c):
    c = np.asarray(c, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.clip(np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055) * 255, 0, 255)


def rgb2lab(rgb):
    l = srgb_to_lin(rgb)
    M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    lms = np.cbrt(l @ M1.T)
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return lms @ M2.T


def lab2rgb(lab):
    lab = np.asarray(lab, float)
    M2i = np.array([[1.0, 0.3963377774, 0.2158037573],
                    [1.0, -0.1055613458, -0.0638541728],
                    [1.0, -0.0894841775, -1.2914855480]])
    lms = (lab @ M2i.T) ** 3
    M1i = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                    [-1.2684380046, 2.6097574011, -0.3413193965],
                    [-0.0041960863, -0.7034186147, 1.7076147010]])
    return lin_to_srgb(lms @ M1i.T)


def mix(a, b, t):
    """Mix two sRGB colours in OKLab."""
    return lab2rgb(rgb2lab(a) * (1 - t) + rgb2lab(b) * t)


# ------------------------------------------------------------------ sheets
def _font(sz=12):
    for p in ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf"]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def grid(cells, cols, rowlabels=None, collabels=None, title=None, labels=None, pad=6,
         bg=(22, 22, 26), fg=(225, 225, 225)):
    h = max(c.shape[0] for c in cells)
    w = max(c.shape[1] for c in cells)
    rows = (len(cells) + cols - 1) // cols
    lw = 120 if rowlabels else 0
    th = 20 if collabels else 0
    tt = 26 if title else 0
    lh = 15 if labels else 0
    W = lw + cols * (w + pad) + pad
    H = tt + th + rows * (h + pad + lh) + pad
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    f = _font(12)
    if title:
        d.text((pad, 5), title, fill=fg, font=_font(14))
    if collabels:
        for c, t in enumerate(collabels):
            d.text((lw + pad + c * (w + pad), tt + 3), t, fill=fg, font=f)
    for i, c in enumerate(cells):
        r, k = divmod(i, cols)
        x = lw + pad + k * (w + pad)
        y = tt + th + pad + r * (h + pad + lh)
        im.paste(Image.fromarray(c), (x, y))
        if labels:
            d.text((x, y + c.shape[0] + 1), labels[i], fill=fg, font=f)
    if rowlabels:
        for r, t in enumerate(rowlabels):
            d.text((4, tt + th + pad + r * (h + pad + lh) + h // 2 - 6), t, fill=fg, font=f)
    return np.array(im)


def fit(img, H):
    """Resize a photo (or anything) to height H with smooth filtering, for side-by-sides."""
    im = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    im = im.convert("RGB")
    return np.array(im.resize((max(1, int(im.width * H / im.height)), H), Image.LANCZOS))


def hstack(panels, labels=None, pad=8, bg=(22, 22, 26)):
    H = max(p.shape[0] for p in panels)
    W = sum(p.shape[1] for p in panels) + pad * (len(panels) + 1)
    lh = 18 if labels else 0
    im = Image.new("RGB", (W, H + pad * 2 + lh), bg)
    d = ImageDraw.Draw(im)
    x = pad
    for i, p in enumerate(panels):
        im.paste(Image.fromarray(p), (x, pad + lh))
        if labels:
            d.text((x, 3), labels[i], fill=(225, 225, 225), font=_font(12))
        x += p.shape[1] + pad
    return np.array(im)
