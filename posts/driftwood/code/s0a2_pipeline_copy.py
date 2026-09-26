"""Stage 0A-2: the same pale trunk re-painted by the general pipeline (3-D cylinder, designed
light, cylinder_steps, ordered dither), with his palette, and compared to his crop.
This proves the technique is geometry-driven (so it can be rotated to lie down) and still
matches the master."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from s0a_pale_trunk import load, silhouette, CLASSES, ROOT
import geo, tech
from paint import quantize, BAYER2

d = load('V09'); idx = d['idx']; pal = d['pal']
x0, x1, y0, y1 = 356, 396, 126, 241
ref = idx[y0:y1, x0:x1]
L_, R_ = silhouette(idx, x0, x1, y0, y1)
H, W = ref.shape
# his value-ordered ramp mapped onto wood steps 2..6
STEP2IDX = {2: 131, 3: 132, 4: 130, 5: 128, 6: 129}


def paint(az_deg=30.0, breaks=None, phase=(0, 0)):
    out = np.full((H, W), -1)
    V = np.array([0, -1.0, 0]); T = np.array([0, 0, 1.0])
    a = np.radians(az_deg)
    Lv = np.array([np.sin(a), -np.cos(a), 0.3]); Lv /= np.linalg.norm(Lv)
    Vv = np.zeros((H, W)); M = np.zeros((H, W), bool)
    for i in range(H):
        if L_[i] < 0:
            continue
        xs = np.arange(L_[i], R_[i] + 1)
        c = 0.5 * (L_[i] + R_[i] + 1); r = 0.5 * (R_[i] - L_[i] + 1)
        sx = np.clip((xs + 0.5 - c) / r, -0.999, 0.999)
        N = np.stack([sx, -np.sqrt(1 - sx ** 2), np.zeros_like(sx)], -1)
        d, phL = tech.normal_plane_angles(N, V, Lv, T)
        Vv[i, xs] = tech.cylinder_classes(d, phL, breaks if breaks is not None else tech.CYL_BREAKS)
        M[i, xs] = True
    st, k = tech.quantize_classes(Vv, tech.CHECK3, phase)
    out[M] = np.vectorize(STEP2IDX.get)(st[M])
    return out, M


def score(out, M):
    return (out[M] == ref[M]).mean()


br = tech.CYL_BREAKS.copy(); az = 30.0; ph = (0, 0)
for it in range(3):
    for j in range(5):
        best = None
        grid = np.arange(10, 70, 2) if j == 4 else br[j] + np.arange(-20, 21, 2)
        for g in grid:
            b2 = br.copy(); a2 = az
            if j == 4: a2 = g
            else: b2[j] = g
            if not np.all(np.diff(b2) > 0): continue
            for phx in ((0, 0), (1, 0), (0, 1), (1, 1)):
                o, M = paint(a2, b2, phx); sc = score(o, M)
                if best is None or sc > best[0]: best = (sc, g, phx)
        if j == 4: az = best[1]
        else: br[j] = best[1]
        ph = best[2]
    print(it, round(best[0], 3), az, br, ph)
P = br
o, M = paint(az, br, ph)
print('FINAL', score(o, M))
open(os.path.join(ROOT, 'scratch', 's0a2_params.txt'), 'w').write(repr((az, P, ph)))
mine = ref.copy(); mine[M] = o[M]
A = pal[ref]; B = pal[mine]
gap = np.full((H, 2, 3), 255, np.uint8)
im = np.concatenate([A, gap, B], 1)
Image.fromarray(im).resize((im.shape[1] * 8, H * 8), Image.NEAREST).save(f'{ROOT}/stages/s0a2_pipeline_copy_8x.png')
