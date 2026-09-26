"""vocab.py: the mark vocabulary as a function of projected blade length ell.

A flat lit patch and a flat shadowed patch, painted at a ladder of ell, each at 4x, plus a
'handoff strip': one tuft population drawn at 24 ell steps so you can follow single tufts
shrinking from blades to ticks to dots to tone.
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw
from grasslod.world import World
from grasslod.render import Camera
from grasslod.marks import tuft_population, paint_tufts, glyph_area, MarkParams
from grasslod.world import SPECIES
from grasslod.calibrate import correction, _Flat
from grasslod.noise import stable_threshold
from grasslod import palette as P


def flat_patch(ell, v, litfam, hour="golden", W=64, H=48, theta=30.0, mp=None, world=None):
    w = world or World(seed=1)
    import copy
    fw = copy.copy(w)
    fw.height = lambda x, y, min_wl=0.0: np.zeros(np.shape(x))
    fw.water_level = -10
    mp = mp or MarkParams()
    sp = SPECIES[w.species]
    hmean = sp[0] * (1 - 0.75 * w.grazing)
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    ppm = ell / (hmean * np.sqrt(c * c + (mp["lean_top"] * s ** 3) ** 2))
    cam = Camera(ppm=ppm, theta=theta, W=W, H=H)
    R = _Flat(); R.cam = cam; R.min_wl = 1 / ppm; R.sun_screen_x = -1
    xs = (np.arange(W) + 0.5 - W / 2) / ppm
    ys = ((H / 2 - (np.arange(H) + 0.5)) / ppm) / s
    R.X, R.Y = np.meshgrid(xs, ys)
    R.depth = R.Y.copy(); R.mat = np.ones((H, W), np.int8)
    L = np.full((H, W), litfam)
    V0 = np.full((H, W), v)
    Vd = V0 - mp["lock"] * correction(fw, mp, ell, V0, L, mp["ell_ref"])
    from grasslod.marks import keep_for
    keep = keep_for(w, mp, ell, ppm, s)
    m = 3 * sp[0] + 3 / ppm
    pop = tuft_population(fw, xs.min() - m, xs.max() + m, ys.min() - m, ys.max() + 3 * m, min(1, keep * 1.8), seed=7)
    from grasslod.marks import quantize_base
    bw = mp["band_w"] if litfam else mp["band_w_shadow"]
    iy = np.broadcast_to(np.arange(H)[:, None], (H, W)).astype(np.int64)
    R.qbase = quantize_base(Vd, R.X, R.Y, ppm, s, bw, iy=iy)
    canvas = np.full((H, W), np.nan); cm = np.zeros((H, W), np.int8); pr = np.full((H, W), -1, np.int64)
    if pop is not None:
        pop["keep"] = keep
        paint_tufts(canvas, cm, pr, R, Vd, L, fw, pop, mp, fam_top=(2.9, 6.0), fam_bot=(0, 0))
    qb = R.qbase
    q = np.where(np.isnan(canvas), qb, np.floor(np.nan_to_num(canvas) + 0.5))
    q = np.clip(q, 0, 6).astype(int)
    pal, _ = P.build(hour)
    return P.to_rgb(P.GREEN0 + q, pal)


def main(hour="golden"):
    ells = [14, 9, 6, 4, 2.8, 2.0, 1.4, 1.0, 0.7, 0.45, 0.25, 0.12]
    k = 4
    W, H = 64, 48
    rows = [("lit, V=4.8", 4.8, True), ("lit, V=4.1", 4.1, True), ("shadow, V=1.0", 1.0, False)]
    im = Image.new("RGB", (100 + len(ells) * (W * k + 4), 24 + len(rows) * (H * k + 4)), (24, 24, 28))
    d = ImageDraw.Draw(im)
    for j, e in enumerate(ells):
        d.text((100 + j * (W * k + 4) + 4, 6), f"ell {e:g} px", fill=(220, 220, 220))
    for i, (lab, v, lf) in enumerate(rows):
        d.text((6, 24 + i * (H * k + 4) + 6), lab, fill=(220, 220, 220))
        for j, e in enumerate(ells):
            t = flat_patch(e, v, lf, hour, W, H)
            im.paste(Image.fromarray(t).resize((W * k, H * k), Image.NEAREST), (100 + j * (W * k + 4), 24 + i * (H * k + 4)))
    out = os.path.join(os.path.dirname(__file__), "..", "img", f"vocab_{hour}.png")
    im.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["golden"]))
