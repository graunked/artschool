"""Texture statistics of a crop: colours for 90%, top-colour share, mean vertical/horizontal runs."""
import sys
import numpy as np
from PIL import Image

def runs(a, axis):
    # axis 0: vertical runs (walk down columns); axis 1: horizontal runs (walk along rows)
    if axis == 0:
        a = a.transpose(1, 0, 2)
    tot, n = 0, 0
    for col in a:
        ch = np.nonzero(np.any(col[1:] != col[:-1], axis=-1))[0]
        L = np.diff(np.concatenate([[-1], ch, [len(col) - 1]]))
        tot += L.sum(); n += len(L)
    return tot / n

def stats(im):
    a = np.array(im.convert('RGB'))
    flat = a.reshape(-1, 3)
    cols, cnt = np.unique(flat, axis=0, return_counts=True)
    o = np.sort(cnt)[::-1] / cnt.sum()
    n90 = int(np.searchsorted(np.cumsum(o), 0.9) + 1)
    return dict(n90=n90, top=round(float(o[0]), 2), vrun=round(runs(a, 0), 2), hrun=round(runs(a, 1), 2))

if __name__ == "__main__":
    f, x, y, w, h = sys.argv[1], *map(int, sys.argv[2:6])
    print(stats(Image.open(f).crop((x, y, x + w, y + h))))
