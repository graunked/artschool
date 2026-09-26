"""churn.py: measure per-frame churn of the lit meadow at near zoom under experiments."""
import sys, numpy as np
from PIL import Image
from zoom import lum, blur
from grasslod.world import World
from grasslod.render import Camera
from grasslod import painter, marks
from grasslod import palette as P

def run(label, p0=24.7, p1=21.0, n=4, patch=None):
    w = World(seed=1); cy, cz = painter.centre_on_bank(w)
    fr = []; ppms = np.geomspace(p0, p1, n)
    for q in ppms:
        cam = Camera(ppm=q, theta=30, W=320, H=180, cx=0, cy=cy, cz=cz)
        idx, pal = painter.paint(w, cam, "golden")
        fr.append(lum(P.to_rgb(idx, pal)))
    out = []
    for k in range(n - 1):
        A, B = fr[k], fr[k + 1]; r = ppms[k + 1] / ppms[k]; H, W = A.shape
        As = np.array(Image.fromarray(A.astype(np.float32)).resize((int(round(W * r)), int(round(H * r))), Image.BOX))
        h, w_ = As.shape; y0, x0 = (H - h) // 2, (W - w_) // 2; Bc = B[y0:y0 + h, x0:x0 + w_]
        out.append(np.abs(blur(As) - blur(Bc))[h // 2 + 10:-6].mean())
    print(label, np.round(out, 2), flush=True)

if __name__ == "__main__":
    exp = sys.argv[1] if len(sys.argv) > 1 else "base"
    if exp == "fixkeep":
        marks.keep_for = lambda *a, **k: 0.15
        painter.keep_for = marks.keep_for
    if exp == "nolock":
        orig = marks.MarkParams.DEFAULT
        marks.MarkParams.DEFAULT = dict(orig, tone_lock=False)
    if exp == "nomarks":
        op = painter.paint
        painter_paint = op
        def p2(*a, **k):
            k["marks"] = False
            return op(*a, **k)
        painter.paint = p2
    if exp == "noconv":
        from grasslod import render
        ov = render.value_design
        def vd(R, sun, params=None):
            out = ov(R, sun, params); R.conv = R.conv * 0; return out
        painter.value_design = vd
    run(exp)
