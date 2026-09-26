"""Stage 0a: copy of V16's lit islet by procedure.
A: his value field (blurred step map) + my quantizer   -> isolates the dither procedure
B: my own field (DPaint contour-gradient with highlight point) + my quantizer -> full procedure"""
import lw, numpy as np, dither
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
d = lw.load('V16'); I = d['idx']; P = d['pal']
x0, y0, x1, y1 = 262, 262, 425, 305
S = I[y0:y1, x0:x1]
RAMP = [77, 76, 75, 74]            # step 0..3 (index order: 74 lit ... 77 darkest-cool)
rank = {k: i for i, k in enumerate(RAMP)}
mask = np.isin(S, RAMP)
st = np.vectorize(lambda k: rank.get(k, 0))(S).astype(float)
# his field: normalized blur of the step map inside the mask
sg = 2.5
num = ndi.gaussian_filter(st * mask, sg); den = ndi.gaussian_filter(mask.astype(float), sg) + 1e-6
field = num / den
rng = np.random.default_rng(1)
A = dither.ordered(field, jitter=0.35, rng=rng, n=4)

# B: my own field.  mask = his silhouette (we test fill, not drawing, here)
h, w = S.shape
yy, xx = np.mgrid[0:h, 0:w]
dist_in = ndi.distance_transform_edt(mask)
hx, hy = 8, 14                          # highlight point (upper-left, where the sun pool sits)
r = np.hypot((xx - hx) / 1.0, (yy - hy) * 2.2)   # oval falloff, squashed vertically (a flat-ish mound)
vB = 3.35 - r / 34.0                    # linear falloff from the highlight
vB = vB - 0.6 * np.exp(-dist_in / 2.0) * (yy > h * 0.55)   # darker foot rim at the waterline
vB = np.clip(vB, 0, 3)
B = dither.ordered(vB, jitter=0.35, rng=np.random.default_rng(2), n=4)

def paint(steps):
    out = P[S].copy()
    out[mask] = P[np.array(RAMP)[steps]][mask]
    return out
ref = P[S]
row = lambda *ims: np.concatenate([np.pad(i, ((0, 0), (0, 3), (0, 0))) for i in ims], 1)
lw.save(row(ref, paint(A), paint(B)), '../stages/s0a_islet_6x.png', 6)
# field diagnostic
g = lambda f: np.stack([(f / 3 * 255).astype(np.uint8)] * 3, -1)
lw.save(row(g(field), g(vB)), '../stages/s0a_islet_fields_4x.png', 4)
