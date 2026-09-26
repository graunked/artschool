"""Stage 0B: master copy of Ferrari's fallen dead tree (Living Worlds V02, dusk, native crop
x 250..420, y 358..412).  The silhouette is MY sketch (polyline + widths read off his image as a
copyist would, then drawn by my own rasteriser), not his mask.

Procedure found:
  * contre-jour: the whole body is ONE near-black class (69).  No form inside the mass.
  * the only light is a rim (71, dull maroon) on the silhouette facing the glow (up-right):
      - on the trunk's top contour: a run of rim pixels ALONG the contour, 1 px thick
        (2 px where the contour is near-horizontal: the rim sits on top of the run);
      - on each twig (2 px black) the rim is the single right-hand pixel: '##r'.
  * twigs are 2-px strokes stepping in clean runs (1:1, 1:2, 2:1), forking in V/Y shapes, with
    a 1-px lighter-than-sky gap nowhere -- they are pure silhouette.
  * the trunk contour is a sequence of runs that shorten smoothly round the curve (14,10,6,5,4,
    4,4,3,3,3,2,2 ...), never jittering.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from s0a_pale_trunk import load, ROOT
from util import board

X0, Y0, X1, Y1 = 250, 358, 362, 412
BODY, RIM = 69, 71



def sketch_from(ref):
    """the copyist's sketch, made the way a copyist makes one: the big shape reduced to a few
    vertices.  Trunk: centre and width sampled every 8 columns of the thick mass.  Twigs: the
    skeleton of the thin mass, traced into paths and simplified (Douglas-Peucker, 1.2 px), so
    only the gesture survives; the strokes themselves are redrawn by my rules."""
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize
    from skimage.measure import approximate_polygon
    mass = np.isin(ref, [69, 70, 223, 71, 72, 73])
    thick = ndi.binary_opening(mass, np.ones((5, 5)))
    trunk = []
    for x in range(0, ref.shape[1], 8):
        ys = np.nonzero(thick[:, x])[0]
        if len(ys) >= 3:
            trunk.append((x + X0, Y0 + 0.5 * (ys.min() + ys.max() + 1), ys.max() - ys.min() + 1))
    thin = mass & ~ndi.binary_dilation(thick, iterations=1)
    sk = skeletonize(thin)
    lab, n = ndi.label(sk, np.ones((3, 3)))
    twigs = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) < 4:
            continue
        # order the pixels by walking from an endpoint (greedy nearest neighbour)
        pts = list(zip(xs, ys)); path = [pts.pop(int(np.argmax(ys)))]
        while pts:
            d = [abs(p[0] - path[-1][0]) + abs(p[1] - path[-1][1]) for p in pts]
            j = int(np.argmin(d))
            if d[j] > 3:
                break
            path.append(pts.pop(j))
        poly = approximate_polygon(np.array(path, float), tolerance=1.2)
        twigs.append([(p[0] + X0, p[1] + Y0) for p in poly])
    return trunk, twigs


# my sketch, read off his image column by column (a copyist's drawing, ~60 vertices total)
# trunk: top edge as a smooth convex arc (a few control points, spline-sampled), bottom edge a
# straight diagonal; the root end at the lower left
TOP_CTRL = [(253, 406), (259, 396), (268, 390), (280, 386), (296, 383.5), (320, 382.5), (345, 384),
            (361, 385.5)]
BOT_CTRL = [(361, 388.5), (335, 391), (310, 396), (285, 401), (263, 406)]
# twigs: (polyline, base width).  Width tapers to 1 px at the tip; every vertex is a joint that
# gets a 1-px knob on the lit side; long straight spans are broken into kinks every ~4 px.
TWIGS = [
    ([(313, 382), (314, 376), (316, 371), (316, 366), (317, 362), (316, 358)], 3),
    ([(311, 368), (313, 364), (315, 360)], 2),
    ([(319, 382), (321, 378), (323, 375), (324, 370), (326, 366), (326, 362), (327, 358)], 3),
    ([(328, 372), (329, 375), (333, 378), (335, 383)], 2),
    ([(335, 383), (334, 377), (334, 372), (336, 367), (340, 364)], 2),
    ([(345, 384), (345, 380), (346, 376), (346, 371), (348, 369)], 3),
    ([(359, 385), (359, 378), (360, 372), (359, 368)], 2),
    ([(300, 383), (299, 378), (301, 373), (301, 369), (300, 366)], 3),
    ([(291, 388), (290, 381), (290, 375), (291, 371)], 2),
    ([(296, 384), (294, 379), (285, 377), (276, 378), (270, 379), (264, 382), (257, 384), (250, 385)], 3),
    ([(283, 377), (282, 373), (284, 371), (285, 374)], 2),
    ([(277, 378), (275, 376)], 2),
    ([(272, 380), (272, 385), (271, 391)], 2),
    ([(266, 381), (265, 386), (266, 392)], 2),
    ([(258, 384), (258, 390), (259, 398)], 2),
    ([(252, 383), (251, 390), (253, 398), (251, 401)], 2),
]


def spline(pts, n=200):
    pts = np.array(pts, float)
    t = np.linspace(0, 1, len(pts)); tt = np.linspace(0, 1, n)
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(t, pts, bc_type='natural')
    return cs(tt)


def line_px(p, q):
    """clean-run line between two points: a DDA on the major axis, rounded minor axis."""
    (x0, y0), (x1, y1) = p, q
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    t = np.linspace(0, 1, n)
    return np.round(x0 + (x1 - x0) * t).astype(int), np.round(y0 + (y1 - y0) * t).astype(int)


def draw(H, W, snap=True):
    from skimage.draw import polygon as fillpoly
    M = np.zeros((H, W), bool)
    top = spline(TOP_CTRL); bot = np.array(BOT_CTRL, float)
    poly = np.concatenate([top, bot])
    rr, cc = fillpoly(poly[:, 1] - Y0, poly[:, 0] - X0, (H, W))
    M[rr, cc] = True
    for tw, w0 in TWIGS:
        w0 = w0 + 1
        segs = list(zip(tw[:-1], tw[1:]))
        total = sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in segs)
        run = 0.0
        for k, (a, b) in enumerate(segs):
            lx, ly = line_px(a, b)
            L = max(len(lx) - 1, 1)
            for j, (x, y) in enumerate(zip(lx, ly)):
                f = (run + j / L * np.hypot(b[0] - a[0], b[1] - a[1])) / max(total, 1e-6)
                w = w0 if f < 0.65 else int(round(w0 - (w0 - 1) * (f - 0.65) / 0.35))
                for dx in range(max(w, 1)):
                    if 0 <= y - Y0 < H and 0 <= x + dx - X0 < W:
                        M[y - Y0, x + dx - X0] = True
            run += np.hypot(b[0] - a[0], b[1] - a[1])
            # joint knob: one pixel on the outside of the bend at each interior vertex
            if k < len(segs) - 1:
                x, y = b
                if 0 <= y - Y0 < H and 0 <= x - 1 - X0 < W:
                    M[int(y) - Y0, int(x) - 1 - X0] = True
    return M


def rim(M):
    """the back-light wraps the whole silhouette on the glow side: every mass pixel whose right
    neighbour or upper neighbour is open sky gets the rim -- on twigs that is the right-hand
    pixel ('##r'), on the trunk a run along the top contour."""
    P = np.pad(M, 1)
    right_open = M & ~P[1:-1, 2:]
    up_open = M & ~P[:-2, 1:-1]
    return right_open | up_open


def main():
    d = load('V02'); idx = d['idx']; pal = d['pal']
    ref = idx[Y0:Y1, X0:X1]
    H, W = ref.shape
    M = draw(H, W)
    R = rim(M)
    mine = ref.copy()
    refmass = np.isin(ref, [69, 70, 223, 71, 72, 73])
    mine[refmass] = ref[~refmass][0] if (~refmass).any() else ref[0, 0]
    # repaint the background under his mass with the nearest non-mass pixel above (so only my
    # silhouette is shown)
    from scipy import ndimage as ndi
    _, (iy, ix) = ndi.distance_transform_edt(refmass, return_indices=True)
    mine = ref[iy, ix].copy()
    mine[M] = BODY
    mine[R] = RIM
    A = pal[ref]; B = pal[mine]
    iou = (M & refmass).sum() / (M | refmass).sum()
    rimref = np.isin(ref, [71, 72, 73])
    print(f'silhouette IoU {iou:.2f}; rim pixels ref {rimref.sum()} mine {R.sum()}; '
          f'rim on rim {(R & rimref).sum()}')
    board([('Ferrari V02 (native 170x54)', A), ('my copy: sketch + body/rim rules', B)], scale=6, cols=1,
          title='s0b fallen dead tree, contre-jour').save(f'{ROOT}/stages/s0b_fallen_6x.png')


if __name__ == '__main__':
    main()
