"""Load Ferrari LBM-json scenes (index map + palette)."""
import re, json, os, numpy as np
from PIL import Image
ROOTS = [os.path.expanduser('~/work/pixelart-studies/pedagogy/master-copy/ref/lw'),
         os.path.expanduser('~/work/pixelart-studies/aerial/refs/ferrari_lw')]
def load(name):
    for r in ROOTS:
        p = f'{r}/{name}.json'
        if os.path.exists(p):
            t = open(p).read(); break
    t = re.sub(r'([{,])\s*([A-Za-z_]+)\s*:', r'\1"\2":', t).replace("'", '"')
    d = json.loads(t)
    d['pal'] = np.array(d['colors'], dtype=np.uint8)
    d['idx'] = np.array(d['pixels'], dtype=np.int32).reshape(d['height'], d['width'])
    return d
def rgb(d, box=None):
    a = d['pal'][d['idx']]
    if box: x0, y0, x1, y1 = box; a = a[y0:y1, x0:x1]
    return a
def up(a, k):
    return np.repeat(np.repeat(a, k, 0), k, 1)
def save(a, path, k=1):
    Image.fromarray(up(np.asarray(a, np.uint8), k)).save(path)
