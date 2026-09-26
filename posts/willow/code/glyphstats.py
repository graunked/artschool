"""Glyph statistics of a foliage crop: value classes by luminance, vertical/horizontal
transitions, and the shapes of connected light marks."""
import numpy as np
from scipy import ndimage as ndi
from lw import lum

def classes(rgb, edges):
    L = lum(rgb)
    return np.digitize(L, edges)

def stats(c, ncls, name=""):
    out = {}
    h, w = c.shape
    share = np.bincount(c.ravel(), minlength=ncls) / c.size
    out["share"] = np.round(share, 3)
    # what is directly BELOW class k (dy=+1) and ABOVE
    below = np.zeros((ncls, ncls)); right = np.zeros((ncls, ncls))
    np.add.at(below, (c[:-1].ravel(), c[1:].ravel()), 1)
    np.add.at(right, (c[:, :-1].ravel(), c[:, 1:].ravel()), 1)
    out["below"] = np.round(below / below.sum(1, keepdims=True).clip(1), 2)
    out["right"] = np.round(right / right.sum(1, keepdims=True).clip(1), 2)
    # connected components of the top class(es)
    for k in range(1, ncls):
        m = c >= k
        lab, n = ndi.label(m, structure=np.ones((3, 3)))
        if n == 0: continue
        sl = ndi.find_objects(lab)
        sizes = np.bincount(lab.ravel())[1:]
        hs = np.array([s[0].stop - s[0].start for s in sl]); ws = np.array([s[1].stop - s[1].start for s in sl])
        out[f"cc>={k}"] = dict(n_per_100px=round(100 * n / c.size, 2), size_med=float(np.median(sizes)),
                               size_p90=float(np.percentile(sizes, 90)), h_med=float(np.median(hs)),
                               w_med=float(np.median(ws)), aspect=round(float(np.median(hs / ws)), 2))
    # mean run lengths per axis
    v = (np.diff(c, axis=0) != 0).mean(); hh = (np.diff(c, axis=1) != 0).mean()
    out["runs_v_h"] = (round(1 / v, 2), round(1 / hh, 2))
    return out

def show(o):
    for k, v in o.items():
        print(" ", k, ":", v if not isinstance(v, np.ndarray) else "\n" + str(v))
