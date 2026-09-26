"""pixkit: small shared helpers for the channel study.

Index canvases (int arrays) + palettes (N x 3 uint8). Everything is nearest-neighbour.
"""
import re, json
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

LW = '~/work/pixelart-studies/pedagogy/master-copy/ref/lw/'
BAYER2 = np.array([[0, 2], [3, 1]]) / 4.0 + 0.125
BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0 + 1 / 32


def lw_load(name):
    t = open(LW + name + '.json').read()
    t = re.sub(r'([{,])\s*([A-Za-z_]+)\s*:', r'\1"\2":', t).replace("'", '"')
    d = json.loads(t)
    d['pal'] = np.array(d['colors'], dtype=np.uint8)
    d['idx'] = np.array(d['pixels'], dtype=np.int32).reshape(d['height'], d['width'])
    return d


def tile(m, h, w, ox=0, oy=0):
    my, mx = m.shape
    yy, xx = np.mgrid[0:h, 0:w]
    return m[(yy + oy) % my, (xx + ox) % mx]


def rgb(idx, pal):
    return pal[np.clip(idx, 0, len(pal) - 1)]


def zoom(a, s):
    return np.repeat(np.repeat(a, s, 0), s, 1)


def save(a, path, s=1):
    Image.fromarray(zoom(a.astype(np.uint8), s)).save(path)


# ---------- dithering ----------------------------------------------------------------

def ordered(v, matrix=BAYER2, ox=0, oy=0):
    """continuous step coordinate v -> integer step, mixing only adjacent steps (ordered)."""
    h, w = v.shape
    t = tile(matrix, h, w, ox, oy)
    f = np.floor(v)
    return (f + ((v - f) > t)).astype(int)


def banded(v, width=0.5):
    """squash a continuous coordinate so it holds flat and crosses between integers over `width`."""
    f = np.floor(v); r = v - f
    lo = 0.5 - width / 2
    return f + np.clip((r - lo) / max(width, 1e-6), 0, 1)


def value_noise(h, w, scale, rng, octaves=1, aniso=(1.0, 1.0), persistence=0.5):
    out = np.zeros((h, w)); amp = 1.0; tot = 0
    for o in range(octaves):
        sy = max(1, int(h / (scale * aniso[0]) / 2 ** -o) + 2)
        sx = max(1, int(w / (scale * aniso[1]) / 2 ** -o) + 2)
        g = rng.random((sy + 1, sx + 1))
        yy = np.linspace(0, sy - 1, h); xx = np.linspace(0, sx - 1, w)
        y0 = np.floor(yy).astype(int); x0 = np.floor(xx).astype(int)
        fy = (yy - y0)[:, None]; fx = (xx - x0)[None, :]
        fy = fy * fy * (3 - 2 * fy); fx = fx * fx * (3 - 2 * fx)
        a = g[y0][:, x0]; b = g[y0][:, x0 + 1]; c = g[y0 + 1][:, x0]; d = g[y0 + 1][:, x0 + 1]
        out += amp * (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy)
        tot += amp; amp *= persistence
    return out / tot


# ---------- geometry -----------------------------------------------------------------

def seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy + 1e-9
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0, 1)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy)), t


def polyline_field(h, w, pts, widths, ky=1.0):
    """distance-to-polyline in a space where y is stretched by ky; returns (d - halfwidth) (<0 inside)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    best = np.full((h, w), 1e9)
    for i in range(len(pts) - 1):
        (ax, ay), (bx, by) = pts[i], pts[i + 1]
        d, t = seg_dist(xx, yy * ky, ax, ay * ky, bx, by * ky)
        hw = widths[i] * (1 - t) + widths[i + 1] * t
        best = np.minimum(best, d - hw / 2)
    return best


# ---------- comparison ---------------------------------------------------------------

def srgb2lab(a):
    a = a / 255.0
    a = np.where(a > 0.04045, ((a + 0.055) / 1.055) ** 2.4, a / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = a @ M.T / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def runs(img):
    k = img[..., 0].astype(np.int64) * 65536 + img[..., 1] * 256 + img[..., 2]
    v = (np.diff(k, axis=0) != 0).mean(); hh = (np.diff(k, axis=1) != 0).mean()
    return round(1 / max(v, 1e-6), 2), round(1 / max(hh, 1e-6), 2)


def metrics(copy, ref):
    c = copy.astype(float); r = ref.astype(float)
    bl = np.linalg.norm(gaussian_filter(srgb2lab(c), (3, 3, 0)) - gaussian_filter(srgb2lab(r), (3, 3, 0)), axis=-1).mean()
    return dict(blurLab=round(float(bl), 2), runs_copy=runs(copy), runs_ref=runs(ref))


def panel(items, s, gap=8, label_h=14, bg=(250, 250, 250)):
    """items: list of (rgb array, label). Side by side at scale s."""
    ims = [Image.fromarray(zoom(a.astype(np.uint8), s)) for a, _ in items]
    W = sum(i.width for i in ims) + gap * (len(ims) - 1); H = max(i.height for i in ims) + label_h
    o = Image.new('RGB', (W, H), bg); d = ImageDraw.Draw(o); x = 0
    for im, (_, lab) in zip(ims, items):
        o.paste(im, (x, label_h)); d.text((x + 2, 1), lab, fill=(0, 0, 0)); x += im.width + gap
    return o


def stack(imgs, gap=8, bg=(250, 250, 250)):
    W = max(i.width for i in imgs); H = sum(i.height for i in imgs) + gap * (len(imgs) - 1)
    o = Image.new('RGB', (W, H), bg); y = 0
    for i in imgs:
        o.paste(i, (0, y)); y += i.height + gap
    return o


def compare_sheet(ref, copy, out, details, label='', s_whole=3, s_det=8):
    """ref|copy at s_whole, then each detail window (x,y,w,h) ref|copy at s_det."""
    rows = [panel([(ref, 'FERRARI %dx' % s_whole), (copy, 'COPY %dx %s' % (s_whole, label))], s_whole)]
    for (x, y, w, h) in details:
        rows.append(panel([(ref[y:y + h, x:x + w], 'ref 8x @(%d,%d)' % (x, y)),
                           (copy[y:y + h, x:x + w], 'copy 8x')], s_det))
    stack(rows).save(out)


def edge_profile(lum, mask, y0=0, maxk=10):
    """mean luminance of land pixels by rows to the nearest water below (far bank) / above (near bank)."""
    h, w = mask.shape
    A = [[] for _ in range(maxk + 1)]; B = [[] for _ in range(maxk + 1)]
    for x in range(w):
        col = mask[:, x]
        for y in range(y0, h):
            if col[y]:
                continue
            for k in range(1, maxk + 1):
                if y + k < h and col[y + k]:
                    A[k].append(lum[y, x]); break
            for k in range(1, maxk + 1):
                if y - k >= 0 and col[y - k]:
                    B[k].append(lum[y, x]); break
    f = lambda L: [round(float(np.mean(v)), 0) if v else None for v in L[1:]]
    return f(A), f(B)


def luma(idx, pal):
    p = pal.astype(float)
    return (0.3 * p[:, 0] + 0.59 * p[:, 1] + 0.11 * p[:, 2])[np.clip(idx, 0, len(pal) - 1)]


def col_dist(mask, maxd=64):
    """per pixel: rows to the nearest mask pixel below (db) and above (da) in the same column."""
    h, w = mask.shape
    db = np.full((h, w), maxd, float); da = np.full((h, w), maxd, float)
    cur = np.full(w, maxd, float)
    for y in range(h - 1, -1, -1):
        cur = np.where(mask[y], 0, np.minimum(cur + 1, maxd)); db[y] = cur
    cur = np.full(w, maxd, float)
    for y in range(h):
        cur = np.where(mask[y], 0, np.minimum(cur + 1, maxd)); da[y] = cur
    return db, da


def photo_crop(path, box, size):
    """crop box=(x,y,w,h) of a photo and area-resample to size=(w,h) native pixels."""
    x, y, w, h = box
    im = Image.open(path).convert('RGB').crop((x, y, x + w, y + h)).resize(size, Image.BOX)
    return np.asarray(im)


def triptych(photo, pixels, ferrari, out, s=8, labels=('photo', 'mine', 'Ferrari')):
    items = [(a, l) for a, l in zip((photo, pixels, ferrari), labels) if a is not None]
    panel(items, s).save(out)
