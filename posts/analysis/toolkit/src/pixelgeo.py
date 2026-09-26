"""Pixel geometry: contours resolved into short straight runs with clean steps.

Measured on the reference (src/contours.py): Ferrari's mountain crest is 41% 1:1 steps and 53% flat
runs of 3-6 px then a single step; the right shoulder is 43% 1:1 and 29% 2:1.  Silhouettes are
straight segments with CONSISTENT step ratios, not noise.  These helpers impose the same grammar
on procedurally generated shapes: simplify the contour to a polygon, snap every edge to one of the
clean pixel slopes, solve for vertex positions that keep the shape, and re-rasterise.
"""
import numpy as np
from PIL import Image, ImageDraw
from skimage import measure
from scipy.ndimage import distance_transform_edt

# clean slopes as (dx, dy) directions: flat, 3:1, 2:1, 1:1, 1:2, 1:3, vertical (both signs)
_BASE = [(1, 0), (5, 1), (4, 1), (3, 1), (2, 1), (1, 1), (1, 2), (1, 3), (1, 4), (0, 1)]
DIRS = []
for dx, dy in _BASE:
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        v = (dx * sx, dy * sy)
        if v not in DIRS: DIRS.append(v)
DIRS = np.array(DIRS, float)
UDIRS = DIRS / np.linalg.norm(DIRS, axis=1, keepdims=True)


def snap_polygon(P, prefer_flat=0.0):
    """P: (n,2) closed polygon (x,y). Returns a polygon whose edges all have clean slopes."""
    n = len(P)
    E = np.roll(P, -1, 0) - P
    L = np.linalg.norm(E, axis=1) + 1e-9
    U = E / L[:, None]
    cos = U @ UDIRS.T
    # a little bias toward flat runs (ground contacts, crests read level)
    cos = cos + prefer_flat * (np.abs(UDIRS[:, 1]) < 1e-6)[None]
    D = UDIRS[np.argmax(cos, 1)]
    # unknowns: v0 (2) + t_i (n); vertex i = v0 + sum_{j<i} t_j D_j
    A = np.zeros((2 * n + 2, 2 + n)); b = np.zeros(2 * n + 2)
    for i in range(n):
        A[2 * i, 0] = 1; A[2 * i + 1, 1] = 1
        for j in range(i):
            A[2 * i, 2 + j] = D[j, 0]; A[2 * i + 1, 2 + j] = D[j, 1]
        b[2 * i] = P[i, 0]; b[2 * i + 1] = P[i, 1]
    w = 30.0                                             # closure
    A[2 * n, 2:] = w * D[:, 0]; A[2 * n + 1, 2:] = w * D[:, 1]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    v0, t = sol[:2], sol[2:]
    V = [v0]
    for j in range(n - 1):
        V.append(V[-1] + t[j] * D[j])
    return np.array(V)


def clean_mask(mask, tol=1.1, min_area=6, prefer_flat=0.0):
    """re-rasterise a binary mask with clean-slope straight edges"""
    H, W = mask.shape
    pad = np.pad(mask.astype(float), 1)
    out = Image.new('L', (W, H), 0); d = ImageDraw.Draw(out)
    holes = []
    for c in measure.find_contours(pad, 0.5):
        c = c[:, ::-1] - 1                                # (x, y)
        if len(c) < 8: continue
        area = 0.5 * np.abs(np.dot(c[:, 0], np.roll(c[:, 1], 1)) - np.dot(c[:, 1], np.roll(c[:, 0], 1)))
        if area < min_area: continue
        poly = measure.approximate_polygon(c, tolerance=tol)[:-1]
        if len(poly) < 3: continue
        sp = snap_polygon(poly, prefer_flat)
        xy = [tuple(p) for p in np.round(sp).astype(int)]
        rr = measure.grid_points_in_poly((H, W), c[:, ::-1])
        frac = mask[rr].mean() if rr.any() else 1
        if frac > 0.5: d.polygon(xy, fill=1)
        else: holes.append(xy)
    for xy in holes: d.polygon(xy, fill=0)
    return np.array(out) > 0


def refill(img, old, new, fallback=None):
    """pixels gained take the nearest old-inside colour; pixels lost take the nearest outside colour"""
    out = img.copy()
    gain = new & ~old; lose = old & ~new
    if gain.any():
        _, (iy, ix) = distance_transform_edt(~old, return_indices=True)
        out[gain] = img[iy[gain], ix[gain]]
    if lose.any():
        src = fallback if fallback is not None else img
        _, (iy, ix) = distance_transform_edt(old, return_indices=True)
        out[lose] = src[iy[lose], ix[lose]]
    return out


def polyline_profile(f, u0, u1, knots, rng, jitter=0.35):
    """sample f at knots, then join them with clean-slope segments (for ridges / crests).
    Returns y per integer column u in [u0, u1)."""
    us = np.linspace(u0, u1, knots)
    us[1:-1] += rng.uniform(-jitter, jitter, knots - 2) * (us[1] - us[0])
    ys = f(us)
    P = [(us[0], ys[0])]
    for i in range(1, len(us)):
        dx, dy = us[i] - P[-1][0], ys[i] - P[-1][1]
        # snap slope to the clean set (x must advance)
        cand = [(1, 0), (3, 1), (2, 1), (1, 1), (1, 2), (1, 3), (1, 5)]
        best = min(cand, key=lambda c: abs(np.arctan2(abs(dy), dx) - np.arctan2(c[1], c[0])))
        slope = np.sign(dy) * best[1] / best[0]
        P.append((us[i], P[-1][1] + slope * dx))
    P = np.array(P)
    u = np.arange(int(u0), int(u1))
    y = np.interp(u, P[:, 0], P[:, 1])
    return u, np.floor(y + 0.5)
