"""wpaint: willow scrub, painted as an index canvas (material, step).

Stage by stage this file grows. Everything draws into a Canvas that holds, per pixel,
a material id and a step on that material's ramp. The palette (greys first, colour later)
turns (material, step) into RGB. Form lives in the steps; light lives in the palette.

Conventions: arrays are [y, x], y grows downward. 3-D vectors are (x right, y UP, z toward viewer).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------------ materials
SKY, GRAVEL, LEAF, SILVER, STEM, WATER, GRASS, EARTH, FAR = 1, 2, 3, 4, 5, 6, 7, 8, 9
NSTEP = 6   # leaf value plan: 0 core/contact | 1 shadow | 2 reflected || 3 halftone | 4 light | 5 top light

# greyscale ramps (value only), atelier spacing: big jump at the terminator
GREY6 = [0.05, 0.13, 0.22, 0.42, 0.56, 0.70]
# foliage value plan: close base pair, a gap, close mark pair (contrast lives inside the mark)
GREYF = [0.06, 0.14, 0.20, 0.26, 0.46, 0.62]


def grey_palette():
    """(mat, step) -> grey level in [0,1]. Background materials get their own local values."""
    P = {}
    for s in range(NSTEP):
        P[(LEAF, s)] = GREYF[s]
        P[(SILVER, s)] = min(1.0, GREY6[s] + 0.08)
    P.update({(SKY, s): v for s, v in enumerate([0.78, 0.82, 0.86, 0.90])})
    P.update({(GRAVEL, s): v for s, v in enumerate([0.10, 0.30, 0.44, 0.55, 0.63, 0.72])})
    P.update({(STEM, s): v for s, v in enumerate([0.05, 0.18, 0.32, 0.50, 0.70])})
    return P


class Canvas:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.mat = np.zeros((h, w), np.uint16)
        self.step = np.zeros((h, w), np.uint8)

    def put(self, mask, mat, step):
        self.mat[mask] = mat
        self.step[mask] = np.asarray(step, np.int64)[mask] if np.ndim(step) else step

    def px(self, x, y, mat, step):
        if 0 <= x < self.w and 0 <= y < self.h:
            self.mat[y, x] = mat
            self.step[y, x] = step

    def render(self, pal):
        """pal: dict (mat,step)->grey float or RGB tuple."""
        out = np.zeros((self.h, self.w, 3), np.uint8)
        keys = np.unique(self.mat.astype(np.int32) * 64 + self.step)
        for k in keys:
            m, s = divmod(int(k), 64)
            v = pal.get((m, s))
            if v is None:
                # nearest defined step
                cands = [kk for kk in pal if kk[0] == m]
                if not cands:
                    v = (255, 0, 255)
                else:
                    v = pal[min(cands, key=lambda kk: abs(kk[1] - s))]
            if np.isscalar(v):
                v = (int(round(v * 255)),) * 3
            out[(self.mat == m) & (self.step == s)] = v
        return out


# ------------------------------------------------------------------ small helpers
def norm3(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def sun(az_deg, el_deg):
    """az 0: sun at the LEFT; 90: behind the viewer (front light); 180: right; 270: behind subject."""
    a, e = np.radians(az_deg), np.radians(el_deg)
    return norm3([-np.cos(a) * np.cos(e), np.sin(e), np.sin(a) * np.cos(e)])


def value_noise(shape, scale, rng, octaves=1):
    h, w = shape
    out = np.zeros(shape)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        s = max(1.0, scale / (2 ** o))
        gh, gw = int(h / s) + 3, int(w / s) + 3
        g = rng.random((gh, gw))
        yy = np.arange(h) / s
        xx = np.arange(w) / s
        Y, X = np.meshgrid(yy, xx, indexing="ij")
        out += amp * ndi.map_coordinates(g, [Y, X], order=3, mode="reflect")
        tot += amp
        amp *= 0.5
    return out / tot


def upscale(img, k):
    return np.repeat(np.repeat(img, k, 0), k, 1)


def save(img, path, k=1):
    Image.fromarray(upscale(img, k) if k > 1 else img).save(path)


# ------------------------------------------------------------------ clean runs for narrow forms
def clean_line(x0, y0, x1, y1):
    """Pixels of a narrow stroke from (x0,y0) to (x1,y1) as a clean run: the slope is snapped to
    one of 0, 1:4, 1:3, 1:2, 1:1, 2:1, 3:1, 4:1, inf, and steps are equal-length runs."""
    dx, dy = x1 - x0, y1 - y0
    n = max(abs(dx), abs(dy))
    if n < 0.5:
        return [(int(round(x0)), int(round(y0)))]
    ratios = [0, 0.25, 1 / 3, 0.5, 1.0, 2.0, 3.0, 4.0, 1e9]
    sx, sy = (1 if dx >= 0 else -1), (1 if dy >= 0 else -1)
    r = abs(dy) / max(abs(dx), 1e-9)
    r = min(ratios, key=lambda q: abs(np.log((q + 0.05) / (r + 0.05))))
    pts = []
    x, y = int(round(x0)), int(round(y0))
    if r >= 1:   # steep: vertical runs of length r
        run = int(round(r)) if r < 1e8 else 10 ** 6
        steps = int(round(abs(dy)))
        for i in range(steps + 1):
            pts.append((x, y))
            y += sy
            if run < 10 ** 6 and (i + 1) % run == 0:
                x += sx
    else:
        run = int(round(1 / r)) if r > 0 else 10 ** 6
        steps = int(round(abs(dx)))
        for i in range(steps + 1):
            pts.append((x, y))
            x += sx
            if run < 10 ** 6 and (i + 1) % run == 0:
                y += sy
    return pts


def polyline_clean(pts):
    out = []
    for (a, b) in zip(pts[:-1], pts[1:]):
        seg = clean_line(*a, *b)
        if out and seg and out[-1] == seg[0]:
            seg = seg[1:]
        out += seg
    return out


# ------------------------------------------------------------------ fonts / sheets
def _font(sz=12):
    for p in ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf",
              "/Library/Fonts/Arial.ttf"]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def label_grid(cells, cols, labels=None, rowlabels=None, collabels=None, pad=6, bg=(24, 24, 28),
               title=None, fg=(230, 230, 230)):
    """cells: list of HxWx3 uint8 arrays (same size)."""
    h, w = cells[0].shape[:2]
    rows = (len(cells) + cols - 1) // cols
    lw = 110 if rowlabels else 0
    th = 22 if collabels else 0
    tt = 26 if title else 0
    lh = 16 if labels else 0
    W = lw + cols * (w + pad) + pad
    H = tt + th + rows * (h + pad + lh) + pad
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    f = _font(12)
    if title:
        d.text((pad, 5), title, fill=fg, font=_font(14))
    if collabels:
        for c, t in enumerate(collabels):
            d.text((lw + pad + c * (w + pad), tt + 4), t, fill=fg, font=f)
    for i, c in enumerate(cells):
        r, k = divmod(i, cols)
        x = lw + pad + k * (w + pad)
        y = tt + th + pad + r * (h + pad + lh)
        im.paste(Image.fromarray(c), (x, y))
        if labels:
            d.text((x, y + h + 1), labels[i], fill=fg, font=f)
    if rowlabels:
        for r, t in enumerate(rowlabels):
            d.text((4, tt + th + pad + r * (h + pad + lh) + h // 2 - 6), t, fill=fg, font=f)
    return np.array(im)


def side_by_side(panels, labels, height=480, pad=8, bg=(24, 24, 28)):
    """panels: list of uint8 arrays already at display scale (nearest-upscaled pixels or photo)."""
    ims = []
    for p in panels:
        im = Image.fromarray(p)
        if im.height != height:
            im = im.resize((max(1, round(im.width * height / im.height)), height),
                           Image.NEAREST if p.shape[0] < height else Image.LANCZOS)
        ims.append(im)
    W = sum(i.width for i in ims) + pad * (len(ims) + 1)
    out = Image.new("RGB", (W, height + 2 * pad + 18), bg)
    d = ImageDraw.Draw(out)
    x = pad
    for im, t in zip(ims, labels):
        out.paste(im, (x, pad))
        d.text((x, height + pad + 3), t, fill=(230, 230, 230), font=_font(12))
        x += im.width + pad
    return np.array(out)
