"""util.py -- labelled side-by-side boards at integer zoom."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(sz=14):
    for p in ('/System/Library/Fonts/Menlo.ttc', '/System/Library/Fonts/Monaco.ttf'):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def zoom(a, s):
    a = np.asarray(a)
    return np.repeat(np.repeat(a, s, 0), s, 1)


def board(panels, scale=8, pad=6, label_h=20, bg=(24, 24, 28), cols=None, title=None, fsz=13):
    """panels: list of (label, HxWx3 uint8).  Each panel zoomed by scale (or its own scale if a
    3-tuple (label, img, s))."""
    ims = []
    for p in panels:
        lab, im = p[0], np.asarray(p[1])
        s = p[2] if len(p) > 2 else scale
        if im.ndim == 2:
            im = np.stack([im] * 3, -1)
        ims.append((lab, zoom(im.astype(np.uint8), s)))
    cols = cols or len(ims)
    rows = (len(ims) + cols - 1) // cols
    cw = max(i.shape[1] for _, i in ims); ch = max(i.shape[0] for _, i in ims)
    th = 26 if title else 0
    W = cols * (cw + pad) + pad; H = th + rows * (ch + label_h + pad) + pad
    out = Image.new('RGB', (W, H), bg)
    d = ImageDraw.Draw(out); f = font(fsz)
    if title:
        d.text((pad, 4), title, fill=(230, 230, 230), font=font(fsz + 2))
    for n, (lab, im) in enumerate(ims):
        r, c = divmod(n, cols)
        x = pad + c * (cw + pad); y = th + pad + r * (ch + label_h + pad)
        d.text((x, y + 2), lab, fill=(210, 210, 210), font=f)
        out.paste(Image.fromarray(im), (x, y + label_h))
    return out


def fit_photo(path, box, size):
    """crop a photo box (x0,y0,x1,y1) and resize to size (w,h) with area filtering."""
    im = Image.open(path).convert('RGB').crop(box)
    return np.array(im.resize(size, Image.LANCZOS))
