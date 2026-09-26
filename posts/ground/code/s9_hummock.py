"""Stage 0b: V16 far hummocks, copied by procedure.
His silhouettes (the index mask), my fill: per column, a value profile below each crest
(dark accent on the crest line, lit band ~25% down, falloff to a dark base), smoothed
across columns, quantized with the islet's ordered 2x2 + jitter."""
import lw, numpy as np, dither
from scipy import ndimage as ndi
d = lw.load('V16'); I = d['idx']; P = d['pal']
x0, y0, x1, y1 = 300, 215, 640, 280
S = I[y0:y1, x0:x1]
VAL = {69: 3, 235: 3, 72: 2, 59: 2, 70: 1, 120: 1, 126: 1, 71: 0, 127: 0, 60: 0, 61: 0, 62: 0, 52: 0, 53: 0}
RAMP = [71, 70, 69]
m = np.isin(S, list(VAL))
H, W = S.shape

def crest_profile(t, accent=0.35, peak=0.25):
    """0..1 value shape below a crest, t = fraction of the way from crest to base"""
    up = accent + (1 - accent) * np.clip(t / peak, 0, 1) ** 0.7
    down = np.clip(1 - (t - peak) / (1 - peak), 0, 1) ** 0.9
    return np.where(t < peak, up, down)

def crest_t(mask):
    """for each masked pixel: rows below the top of its vertical run, and run length"""
    k = np.zeros(mask.shape); L = np.zeros(mask.shape)
    for x in range(mask.shape[1]):
        col = mask[:, x]; y = 0
        while y < len(col):
            if col[y]:
                y0_ = y
                while y < len(col) and col[y]: y += 1
                k[y0_:y, x] = np.arange(y - y0_); L[y0_:y, x] = y - y0_
            else:
                y += 1
    return k, L

# his crest lines: where his squinted value steps UP going down the column (dark base of the
# hummock behind, lit band of the one in front). These are the silhouettes I must draw myself
# in my own ground; here I take them from him, to test only the fill.
vs = np.vectorize(lambda q: VAL.get(q, 0))(S).astype(float)
sq = ndi.gaussian_filter(vs, (1.2, 3.0))
dy = np.zeros_like(sq); dy[1:] = sq[1:] - sq[:-1]
crest = (dy > 0.12) & (ndi.maximum_filter(dy, size=(5, 1)) == dy) & m
lab, n = ndi.label(ndi.binary_dilation(crest, np.ones((3, 3))))
sz = ndi.sum(crest, lab, range(1, n + 1))
crest = crest & np.isin(lab, np.nonzero(sz >= 25)[0] + 1)
m_cut = m & ~crest
k, L = crest_t(m_cut)
t = k / np.maximum(L, 1)
f = crest_profile(t)
# smooth across columns so the band reads as a form, not per-column noise
num = ndi.gaussian_filter(f * m, (0.6, 2.5)); den = ndi.gaussian_filter(m * 1.0, (0.6, 2.5)) + 1e-6
f = num / den
v = 0.1 + 1.85 * f
st = dither.ordered(v, jitter=0.4, rng=np.random.default_rng(3), n=3)
out = P[S].copy(); out[m] = P[np.array(RAMP)[st]][m]
ref = P[S]
pad = np.zeros((3, W, 3), np.uint8)
dbg = ref.copy(); dbg[crest] = (255, 0, 0)
lw.save(np.concatenate([ref, pad, out, pad, dbg], 0), '../stages/s9_hummocks_5x.png', 5)
