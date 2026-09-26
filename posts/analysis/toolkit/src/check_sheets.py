"""Apply the SAME analyses used on Ferrari to painted cells, and compare side by side.
usage: python src/check_sheets.py <cell_key> [<cell_key> ...]   (keys as in sheets/cells/*.png)"""
import sys, os, json
sys.path.insert(0, 'src')
from util import *
from scipy.ndimage import gaussian_filter
from collections import Counter
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from clusters import cluster_stats, run_lengths
from dither import dither_stats
from values import analyse
from depth import depth_stats, plot as depth_plot
from planes_names import NAMES as FNAMES

CELLS = 'sheets/cells'
MY = ['sky', 'mountain', 'far forest', 'water', 'bank (boulder side)', 'boulder', 'near bank lit', 'near bank shade', 'grove']
# correspondences: my plane -> Ferrari plane ids
CORR = {'sky': [0], 'mountain': [1], 'far forest': [2], 'water': [6], 'bank (boulder side)': [7, 5],
        'boulder': [1], 'near bank lit': [8], 'near bank shade': [9], 'grove': [3]}


def load_cell(key):
    a = np.array(Image.open(f'{CELLS}/{key}.png').convert('RGB'))
    pl = np.load(f'{CELLS}/{key}_planes.npy')
    # split the near bank into its lit and shaded masses by squinted value
    Lb = gaussian_filter(lum(a), 3)
    p2 = pl.copy()
    near = pl == 6
    p2[near & (Lb < 24)] = 7
    return a, p2


def stats_table(a, P, names, corr_ref=None):
    rows = {}
    lab = rgb2lab(a); L = lab[..., 0]
    same_v = np.all(a[1:] == a[:-1], axis=2); same_h = np.all(a[:, 1:] == a[:, :-1], axis=2)
    ds, _ = dither_stats(a, P, names)
    for i, n in enumerate(names):
        m = P == i
        if m.sum() < 80: continue
        cs = cluster_stats(a, m)
        c = Counter(map(tuple, a[m])); tot = sum(c.values())
        cum = np.cumsum(sorted(c.values(), reverse=True)) / tot
        rows[n] = dict(area=float(m.mean()), L_med=float(np.median(gaussian_filter(L, 2.5)[m])),
                       L_p5=float(np.percentile(L[m], 5)), L_p95=float(np.percentile(L[m], 95)),
                       colours_90=int(np.searchsorted(cum, 0.9) + 1), vsame=float(same_v[m[1:]].mean()),
                       hsame=float(same_h[m[:, 1:]].mean()), cluster_med=float(cs['size_med']),
                       cluster_mean=float(cs['size_mean']), single=float(cs['single']),
                       checker=float(ds.get(n, {}).get('checker', 0)), chroma=float(np.median(np.hypot(lab[..., 1], lab[..., 2])[m])),
                       b=float(np.median(lab[..., 2][m])))
    return rows


if __name__ == '__main__':
    keys = sys.argv[1:]
    ref = load_ref(); Pref = np.load('out/planes.npy')
    R = stats_table(ref, Pref, FNAMES)
    allrows = {'ferrari': R}
    for key in keys:
        a, P = load_cell(key)
        res, _ = analyse(a, P, MY, key, f'chk_{key}')
        M = stats_table(a, P, MY)
        allrows[key] = M
        rows = depth_stats(a, P, [('near bank', [6, 7]), ('bank (boulder side)', [4]), ('boulder', [5]),
                                  ('far forest', [2]), ('mountain', [1]), ('sky', [0])])
        depth_plot(rows, f'img/chk_{key}_depth.png', f'{key}: depth cues, near -> far')
        # side-by-side notan / grey with Ferrari (reference scaled to the same height)
        for kind in ('notan', 'grey'):
            mine = np.array(Image.open(f'img/chk_{key}_{kind}.png').convert('RGB'))
            theirs = np.array(Image.open(f'img/03_{kind}.png').convert('RGB'))
            if kind == 'notan':
                w = min(mine.shape[1], theirs.shape[1])
                th = np.array(Image.fromarray(theirs).resize((w, int(theirs.shape[0] * w / theirs.shape[1])), Image.NEAREST))
                mi = np.array(Image.fromarray(mine).resize((w, int(mine.shape[0] * w / mine.shape[1])), Image.NEAREST))
                out = np.concatenate([th, np.full((10, w, 3), 255, np.uint8), mi], 0)
            else:
                h = min(mine.shape[0], theirs.shape[0])
                th = np.array(Image.fromarray(theirs).resize((int(theirs.shape[1] * h / theirs.shape[0]), h), Image.NEAREST))
                mi = np.array(Image.fromarray(mine).resize((int(mine.shape[1] * h / mine.shape[0]), h), Image.NEAREST))
                out = np.concatenate([th, np.full((h, 10, 3), 255, np.uint8), mi], 1)
            Image.fromarray(out).save(f'img/chk_{key}_{kind}_vs.png')
    json.dump(allrows, open('out/check_stats.json', 'w'), indent=1)
    # print comparison of corresponding planes
    for key in keys:
        print('=' * 20, key)
        for n, ids in CORR.items():
            if n not in allrows[key]: continue
            m = allrows[key][n]; f = allrows['ferrari'][FNAMES[ids[0]]]
            print(f'{n:20s} vs {FNAMES[ids[0]]:18s}| ' + ' '.join(f'{k}={m[k]:.2f}/{f[k]:.2f}' for k in ('L_med', 'L_p5', 'L_p95', 'colours_90', 'vsame', 'hsame', 'cluster_mean', 'single', 'checker', 'b')))
