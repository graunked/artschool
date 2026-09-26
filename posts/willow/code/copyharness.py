"""Master-copy harness: decompose a Ferrari crop into (ramp, step) using named ramps of his
indices, build the copyist's sketch (which ramp owns each region, and a smoothed light field),
and compare a copy against the reference (3x whole, 8x detail, metrics)."""
import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
from lw import load, lum


def decompose(idx, pal, ramps):
    """ramps: list of lists of palette indices, dark->light. Returns ramp id map, step map,
    and the RGB table per (ramp, step). Unknown indices snap to nearest ramp colour."""
    table = {}
    lut = {}
    for r, ids in enumerate(ramps):
        for s, i in enumerate(ids):
            table[(r, s)] = tuple(int(v) for v in pal[i])
            lut.setdefault(i, (r, s))
    cols = np.array([table[k] for k in table], float)
    keys = list(table.keys())
    R = np.zeros(idx.shape, int); S = np.zeros(idx.shape, int)
    for i in np.unique(idx):
        if i in lut:
            r, s = lut[i]
        else:
            dd = ((cols - pal[i].astype(float)) ** 2).sum(1)
            r, s = keys[int(dd.argmin())]
        R[idx == i] = r; S[idx == i] = s
    return R, S, table


def sketch(R, S, nramps, maxstep, sigma=2.5, maj=9):
    """Copyist's sketch: majority ramp region (size maj) and smoothed light field in [0,1]."""
    votes = np.stack([ndi.uniform_filter((R == r).astype(float), maj) for r in range(nramps)])
    region = votes.argmax(0)
    light = ndi.gaussian_filter(S / maxstep, sigma)
    return region, light


def render(R, S, table):
    out = np.zeros(R.shape + (3,), np.uint8)
    for (r, s), c in table.items():
        out[(R == r) & (S == s)] = c
    return out


def srgb2lab(a):
    a = a / 255.0
    a = np.where(a > 0.04045, ((a + 0.055) / 1.055) ** 2.4, a / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = a @ M.T / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def metrics(copy, ref):
    c = copy.astype(float); r = ref.astype(float)
    bl = np.linalg.norm(ndi.gaussian_filter(srgb2lab(c), (3, 3, 0)) - ndi.gaussian_filter(srgb2lab(r), (3, 3, 0)), axis=-1).mean()
    def runs(img):
        k = img[..., 0].astype(np.int64) * 65536 + img[..., 1] * 256 + img[..., 2]
        return round(1 / max((np.diff(k, axis=0) != 0).mean(), 1e-6), 2), round(1 / max((np.diff(k, axis=1) != 0).mean(), 1e-6), 2)
    return dict(blurLab=round(float(bl), 2), runs_copy=runs(copy), runs_ref=runs(ref))


def compare_sheet(copy, ref, path, detail=None, label="", k=3):
    h, w = ref.shape[:2]
    a = Image.fromarray(ref).resize((w * k, h * k), Image.NEAREST)
    b = Image.fromarray(copy).resize((w * k, h * k), Image.NEAREST)
    if detail is None:
        detail = (w // 2 - 25, h // 2 - 18, 50, 36)
    x, y, dw, dh = detail
    a8 = Image.fromarray(ref[y:y + dh, x:x + dw]).resize((dw * 8, dh * 8), Image.NEAREST)
    b8 = Image.fromarray(copy[y:y + dh, x:x + dw]).resize((dw * 8, dh * 8), Image.NEAREST)
    W = max(w * k * 2 + 12, dw * 16 + 12); H = h * k + dh * 8 + 44
    o = Image.new("RGB", (W, H), (24, 24, 28)); d = ImageDraw.Draw(o)
    o.paste(a, (0, 16)); o.paste(b, (w * k + 12, 16))
    d.text((2, 2), f"FERRARI {k}x", fill=(230, 230, 230)); d.text((w * k + 14, 2), f"COPY {k}x  " + label, fill=(230, 230, 230))
    y0 = h * k + 38
    o.paste(a8, (0, y0)); o.paste(b8, (dw * 8 + 12, y0))
    d.text((2, y0 - 16), "Ferrari 8x", fill=(230, 230, 230)); d.text((dw * 8 + 14, y0 - 16), "copy 8x", fill=(230, 230, 230))
    for ox in (0, w * k + 12):
        d.rectangle([ox + x * k, 16 + y * k, ox + (x + dw) * k, 16 + (y + dh) * k], outline=(255, 60, 60))
    o.save(path)


def rs_stats(R, S, top):
    """Step shares and shapes of connected lit marks, in (ramp, step) space."""
    from scipy import ndimage as ndi
    o = {"share": np.round(np.bincount(S.ravel(), minlength=top + 1) / S.size, 3).tolist()}
    for k in (top, top - 1):
        m = S >= k
        lab, n = ndi.label(m)
        if n == 0:
            continue
        sl = ndi.find_objects(lab); sz = np.bincount(lab.ravel())[1:]
        hs = np.array([s[0].stop - s[0].start for s in sl]); ws = np.array([s[1].stop - s[1].start for s in sl])
        o[f">={k}"] = dict(per100=round(100 * n / S.size, 2), size=(float(np.median(sz)), float(np.percentile(sz, 90))),
                          h=float(np.mean(hs)), w=float(np.mean(ws)))
    # what lies below the top step
    m = S[:-1] == top
    o["below_top"] = np.round(np.bincount(S[1:][m], minlength=top + 1) / max(1, m.sum()), 2).tolist()
    return o
