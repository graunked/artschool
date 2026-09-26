"""tech.py -- the driftwood techniques, each proven against a study image.

cylinder_classes  (spec: stages/s0a2_*; Ferrari V09 pale trunk)
    A round trunk is five FORM CLASSES in fixed spatial order across its width, counted from
    the shadow edge toward the lit edge:
        0 reflected rim | 1 core | 2 halftone | 3 light | 4 lit-edge falloff
    painted on the wood ramp at steps           3 | 2 | 4 | 6 | 5
    The class coordinate is a smooth function of the angle d around the axis, measured from the
    light's peak (d < 0 toward the shadow edge), with class boundaries (Ferrari, measured):
        rim | -69 deg | core | -42 | halftone | -13 | light | +4 | edge
    So the core starts ~20 deg AFTER the geometric terminator, the halftone is a 30 deg band,
    the light only ~27 deg wide at the peak.  No pure zones: the coordinate runs linearly
    through every class and the quantizer mixes ONLY spatially adjacent classes with a 3-level
    checker (pure / 50% / pure), so every seam is a 1-2 px checker running along the axis.
    Dithering follows class adjacency, not value adjacency: core (step 2) dithers straight into
    halftone (step 4), skipping the cool rim colour (step 3).
"""
import numpy as np

CHECK3 = np.array([[0.25, 0.75], [0.75, 0.25]])     # 3-level checker (Ferrari's trunk seams)
CLASS_STEPS = np.array([3, 2, 4, 6, 5])
CYL_BREAKS = np.array([-69.0, -42.0, -13.0, 4.0])   # degrees from the light peak (fitted, s0a2: 89% exact)


def normal_plane_angles(N, V, L, T):
    """angle of N and of the light around the local axis T, both measured in the plane
    perpendicular to T from the view direction; returns d = signed angle from the light peak,
    oriented so that +d points toward the lit silhouette edge."""
    def proj(a):
        a = a - (a * T).sum(-1, keepdims=True) * T
        return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    Vp = proj(np.broadcast_to(V, N.shape))
    B = np.cross(T, Vp)
    Lp = proj(np.broadcast_to(L, N.shape))
    phN = np.degrees(np.arctan2((N * B).sum(-1), (N * Vp).sum(-1)))
    phL = np.degrees(np.arctan2((Lp * B).sum(-1), (Lp * Vp).sum(-1)))
    sgn = np.where(phL >= 0, 1.0, -1.0)
    d = (phN - phL) * sgn
    d = (d + 180) % 360 - 180
    return d, phL


def cylinder_classes(d, phL, breaks=CYL_BREAKS, sh=None):
    """continuous class coordinate v in [0, 4]: piecewise linear through the breaks (v = k+0.5
    at break k), from v = 0 at the shadow silhouette to v = 4 at the lit silhouette."""
    aL = np.abs(phL)
    d0 = np.minimum(-90.0 - aL, breaks[0] - 1)
    d1 = np.maximum(90.0 - aL, breaks[-1] + 1)
    n = len(breaks)
    v = np.zeros_like(d)
    # segment 0: [d0, b0] -> [0, 0.5]; middle segments; last: [b_last, d1] -> [n-0.5, n]
    xs = [d0] + [np.full_like(d, b) for b in breaks] + [d1]
    ys = [0.0] + [k + 0.5 for k in range(n)] + [float(n)]
    for k in range(len(xs) - 1):
        lo, hi = xs[k], xs[k + 1]
        m = (d >= lo) & (d < hi)
        f = (d - lo) / np.maximum(hi - lo, 1e-6)
        v = np.where(m, ys[k] + f * (ys[k + 1] - ys[k]), v)
    v = np.where(d >= d1, n, v)
    v = np.where(d < d0, 0.0, v)
    if sh is not None:
        v = np.where(sh, np.minimum(v, 1.0), v)
    return np.clip(v, 0, n)


def sharpen(v, diam_px, ref_px=13.0, max_gain=4.0):
    """keep seams ~1-2 px wide whatever the trunk width: Ferrari's 13-px trunk mixes over whole
    classes; his 6-px limbs show pure 1-px bands (c4431 2), and wider trunks keep ~2-px seams
    rather than scaling the checker up.  gain = max(ref/diam, diam/ref), clipped."""
    dd = np.maximum(diam_px, 1e-3)
    g = np.clip(np.maximum(ref_px / dd, dd / ref_px), 1.0, max_gain)
    k = np.floor(v); f = v - k
    return k + np.clip((f - 0.5) * g + 0.5, 0, 1)


def quantize_classes(v, mat=CHECK3, phase=(0, 0), class_steps=CLASS_STEPS):
    H, W = v.shape
    th = np.tile(mat, (H // mat.shape[0] + 2, W // mat.shape[1] + 2))
    th = th[phase[0]:phase[0] + H, phase[1]:phase[1] + W]
    lo = np.floor(v)
    k = np.clip(lo + ((v - lo) > th), 0, len(class_steps) - 1).astype(int)
    return class_steps[k], k
