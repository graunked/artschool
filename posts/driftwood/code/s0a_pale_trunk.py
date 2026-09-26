"""Stage 0A: master copy of Ferrari's pale trunk (Living Worlds V09 'Deep Forest' morning, native
crop x 356..396, y 96..252).  The copy is painted from a procedure, not traced:

  * the copyist's sketch: the left and right silhouette columns per row (his long straight runs
    with single 1-px jogs), read from his map and then *simplified* to runs;
  * the form: a 1-D profile across the trunk -- the form classes in fixed spatial order
        rim(sky-cool) | core-dark | halftone | light | edge-falloff
    as a continuous coordinate v(u), u = across position in [0,1];
  * the quantizer: DPaint-style ordered dither between ADJACENT classes only, the dither
    matrix tiled along the axis (rows), so the seams read as 1-px checker columns.

Writes stages/s0a_*.png and prints the class-match rate on the trunk.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.expanduser('~/work/pixelart-studies/pedagogy/master-copy/src'))
import numpy as np
from PIL import Image
import re, json

ROOT = os.path.expanduser('~/work/pixelart-studies/elements/driftwood')
LW = os.path.expanduser('~/work/pixelart-studies/pedagogy/master-copy/ref/lw')


def load(name):
    t = open(f'{LW}/{name}.json').read()
    t = re.sub(r'([{,])\s*([A-Za-z_]+)\s*:', r'\1"\2":', t).replace("'", '"')
    d = json.loads(t)
    d['pal'] = np.array(d['colors'], dtype=np.uint8)
    d['idx'] = np.array(d['pixels'], dtype=np.int32).reshape(d['height'], d['width'])
    return d


CLASSES = [132, 131, 130, 129, 128]          # rim | dark | half | light | edge, in spatial order
BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16 + 1 / 32


def silhouette(idx, x0, x1, y0, y1):
    m = np.isin(idx[y0:y1, x0:x1], CLASSES)
    L = np.full(y1 - y0, -1); R = np.full(y1 - y0, -1)
    for i in range(y1 - y0):
        xs = np.nonzero(m[i])[0]
        if len(xs):
            L[i], R[i] = xs.min(), xs.max()
    return L, R


def paint_trunk(L, R, breaks, mat, shift=0, h=None, w=None):
    """breaks: u positions where v crosses k+0.5 (class boundaries), len = n_classes-1."""
    out = np.full((h, w), -1)
    n = len(breaks) + 1
    for i in range(h):
        if L[i] < 0:
            continue
        xs = np.arange(L[i], R[i] + 1)
        wd = R[i] - L[i] + 1
        u = (xs - L[i] + 0.5) / wd
        # piecewise-linear v(u): v = k + 0.5 at breaks[k]
        bu = np.concatenate([[0.0], breaks, [1.0]])
        bv = np.concatenate([[0.0], np.arange(len(breaks)) + 0.5, [n - 1.0]])
        v = np.interp(u, bu, bv)
        th = mat[(i + shift) % mat.shape[0], xs % mat.shape[1]]
        lo = np.floor(v); fr = v - lo
        k = np.clip(lo + (fr > th), 0, n - 1).astype(int)
        out[i, xs] = k
    return out


def main():
    d = load('V09'); idx = d['idx']; pal = d['pal']
    x0, x1, y0, y1 = 356, 396, 118, 241
    L, R = silhouette(idx, x0, x1, y0, y1)
    ref = idx[y0:y1, x0:x1]
    refk = np.full(ref.shape, -1)
    for k, c in enumerate(CLASSES):
        refk[ref == c] = k
    tm = refk >= 0
    best = None
    mats = {'bayer4': BAYER4, 'bayer2': np.array([[0.125, 0.625], [0.875, 0.375]]),
            'vert2': np.array([[0.25, 0.75], [0.75, 0.25]]),
            'bayer4T': BAYER4.T}
    rng = np.random.default_rng(0)
    grid = np.linspace(0.05, 0.95, 19)
    for name, mat in mats.items():
        for sh in range(mat.shape[0]):
            br = np.array([0.12, 0.40, 0.62, 0.85])
            # coordinate descent over the 4 breaks
            for it in range(4):
                for j in range(4):
                    scores = []
                    for g in grid:
                        b2 = br.copy(); b2[j] = g
                        if not np.all(np.diff(b2) > 0):
                            scores.append(-1); continue
                        cp = paint_trunk(L, R, b2, mat, sh, *ref.shape)
                        scores.append(((cp == refk) & tm).sum() / tm.sum())
                    br[j] = grid[int(np.argmax(scores))]
            cp = paint_trunk(L, R, br, mat, sh, *ref.shape)
            sc = ((cp == refk) & tm).sum() / tm.sum()
            # class-level tolerance: same class within one step
            sc1 = ((np.abs(cp - refk) <= 1) & tm).sum() / tm.sum()
            if best is None or sc > best[0]:
                best = (sc, sc1, name, sh, br.copy(), cp)
            print(f'{name} shift {sh}: exact {sc:.3f} within1 {sc1:.3f} breaks {np.round(br, 2)}')
    sc, sc1, name, sh, br, cp = best
    print('BEST', name, sh, np.round(br, 3), f'{sc:.3f}', f'{sc1:.3f}')
    # render: his background, my trunk
    out = idx[y0:y1, x0:x1].copy()
    mine = out.copy()
    mine[cp >= 0] = np.array(CLASSES)[cp[cp >= 0]]
    A = pal[out]; B = pal[mine]
    s = 8
    gap = np.full((A.shape[0], 2, 3), 255, np.uint8)
    im = np.concatenate([A, gap, B], 1)
    Image.fromarray(im).resize((im.shape[1] * s, im.shape[0] * s), Image.NEAREST).save(f'{ROOT}/stages/s0a_pale_trunk_8x.png')
    np.save(f'{ROOT}/scratch/s0a_best.npy', np.array([*br, sh]))
    return best


if __name__ == '__main__':
    main()
