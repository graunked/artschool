"""Figures for the conifer tutorials, made with the study's real painter.

Nothing here paints: every image comes from the study folder's own modules (grow, paint, palette,
conifer), called with its real parameters. Where a figure is an *ablation*, one painter parameter is
switched off so you can see what that technique does. Run:

    cd ~/work/artschool/posts/conifers/code
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 python3 figs.py [name ...]

Outputs go to ../img/.  (The study folder is read, never written.)
"""
import os, sys, time
sys.dont_write_bytecode = True        # never write into the study folder
import numpy as np
from PIL import Image, ImageDraw

STUDY = os.path.expanduser("~/work/pixelart-studies/elements/conifers")
sys.path.insert(0, STUDY)
os.chdir(STUDY)                     # the study's scripts use relative refs/ paths
import grow, paint, palette, conifer, lw, stats, s1_branch, s1_grey   # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "img")


def up(a, k):
    return np.repeat(np.repeat(np.asarray(a)[..., :3], k, 0), k, 1)


def panel(imgs, labels, k=4, gap=6, name=None):
    ims = [up(i, k) for i in imgs]
    H = max(i.shape[0] for i in ims) + 16
    W = sum(i.shape[1] for i in ims) + gap * (len(ims) - 1)
    out = np.full((H, W, 3), 250, np.uint8); x = 0
    for i in ims:
        out[16:16 + i.shape[0], x:x + i.shape[1]] = i; x += i.shape[1] + gap
    im = Image.fromarray(out); d = ImageDraw.Draw(im); x = 0
    for i, l in zip(ims, labels):
        d.text((x + 2, 2), l, fill=(0, 0, 0)); x += i.shape[1] + gap
    if name:
        im.save(os.path.join(OUT, name)); print("wrote", name, flush=True)
    return im


def v16(x0, y0, x1, y1):
    d = lw.load("V16")
    return d["pal"][d["idx"][y0:y1, x0:x1]]


def grey(G, st):
    H, W = st.shape
    return paint.compose(st, G["mat"], s1_grey.PAL, np.full((H, W, 3), 200, np.uint8))


# ------------------------------------------------------------------ 1. reading his conifer
def fig_v16_classes():
    """Ferrari's mid-distance spire, and the same pixels painted by class."""
    x0, y0, x1, y1 = 420, 30, 480, 130
    d = lw.load("V16"); sub = d["idx"][y0:y1, x0:x1]
    c = stats.classify_idx(sub)
    cols = np.array([[0, 0, 0], [0, 90, 60], [150, 110, 20], [240, 230, 80]], np.uint8)
    cls = np.full(sub.shape + (3,), 235, np.uint8)
    cls[c >= 0] = cols[c[c >= 0]]
    panel([d["pal"][sub], cls], ["Ferrari V16, native (420,30)-(480,130)", "by class: black / dk green / dk olive / lit"],
          k=5, name="t1_v16_classes.png")


# ------------------------------------------------------------------ 2. tier-top light
HEM = dict(sp="hemlock", age="mature", seed=2, mpp=0.12, W=160, H=420)
ABL = [
    ("A  crown normal 0.5, no edge rules", dict(wc=0.5, edge_light=False, vgrain=0, elem_coh=0.9, _user={"elem_coh": 1})),
    ("B  crown normal 0.18 (tier tops)", dict(wc=0.18, edge_light=False, vgrain=0, elem_coh=0.9, _user={"elem_coh": 1})),
    ("C  + thin edge lights, dark texture", dict(wc=0.18, strand_p=0.0, vgrain=0)),
    ("D  + hanging strands, vertical grain (final)", dict()),
]


def fig_tier_top():
    tr = grow.grow(HEM["sp"], HEM["age"], seed=HEM["seed"], mpp=HEM["mpp"])
    full, zoom, notan = [], [], []
    for lab, P in ABL:
        rgb, info = conifer.tree(HEM["sp"], HEM["age"], seed=HEM["seed"], mpp=HEM["mpp"], W=HEM["W"], H=HEM["H"], tr=tr, P=P)
        full.append(rgb[20:330]); zoom.append(rgb[60:150, 40:120])
        st = info["step"]; m = info["G"]["mat"]
        n = np.full(st.shape + (3,), 205, np.uint8)
        n[(m == paint.FOL) & (st < 4)] = 20; n[(m == paint.FOL) & (st >= 4)] = 150
        notan.append(n[20:330])
    labs = [a for a, _ in ABL]
    panel(full, [l[:2] for l in labs], k=2, name="t2_ablation_full.png")
    panel(notan, [l[:2] + " notan" for l in labs], k=2, name="t2_ablation_notan.png")
    panel([v16(560, 0, 640, 90)] + zoom, ["Ferrari V16 near"] + [l[:2] for l in labs], k=4, name="t2_ablation_zoom.png")


# ------------------------------------------------------------------ 3. growing the species
def fig_limbs():
    ims, labs = [], []
    for sp in ["douglas", "hemlock", "cedar", "spruce"]:
        ims.append(s1_branch.paint_limb(sp, 0.02, 170, 110)); labs.append(sp)
    panel(ims, labs, k=2, name="t3_limbs.png")


def fig_kinds():
    """What the foliage is made of: each pixel coloured by the element kind that drew it."""
    KC = np.array([[70, 170, 90], [230, 120, 40], [120, 90, 200], [240, 220, 40], [255, 255, 255]], np.uint8)
    ims, cols, labs = [], [], []
    for sp in ["douglas", "hemlock", "cedar", "spruce"]:
        rgb, info = conifer.tree(sp, "young", seed=1, mpp=0.05, W=180, H=280)
        G = info["G"]; k = G["kind"]
        c = np.full(rgb.shape, 30, np.uint8)
        c[G["mat"] > 1] = (120, 80, 60)
        m = (G["mat"] == paint.FOL) & (k >= 0)
        c[m] = KC[k[m]]
        ims += [rgb, c]; labs += [f"{sp} young 0.05", "kind"]
    panel(ims, labs, k=1, gap=4, name="t3_kinds.png")


def fig_leader():
    ims, labs = [], []
    for sp in ["douglas", "hemlock", "cedar", "spruce"]:
        tr = grow.grow(sp, "mature", seed=2, mpp=0.08)
        rgb, _ = conifer.tree(sp, "mature", seed=2, mpp=0.08, W=120, H=int((tr.H + 1) / 0.08) + 4, tr=tr)
        ims.append(rgb[:150]); labs.append(sp + " top")
    panel(ims, labs, k=3, name="t3_tops.png")


# ------------------------------------------------------------------ 4. every distance
def fig_fill():
    """Coarse-scale area fill on/off: without it, sprays are 1-px skeletons = flags on poles."""
    ims, labs = [], []
    for sp in ["douglas", "cedar"]:
        for mode in ["slow-lines", "fill"]:
            tr = grow.grow(sp, "old", seed=1, mpp=0.28)
            if mode == "slow-lines":
                import foliage_fast
                saved = foliage_fast._fill
                foliage_fast._fill = lambda kind, P, N, D, ek, e, R, TAP, sp_, rng, step: (P, N, D, ek, e)
                tr = grow.grow(sp, "old", seed=1, mpp=0.28)
                foliage_fast._fill = saved
            rgb, _ = conifer.tree(sp, "old", seed=1, mpp=0.28, W=110, H=260, tr=tr)
            ims.append(rgb); labs.append(f"{sp} {'no area fill' if mode != 'fill' else 'area fill'}")
    panel(ims, labs, k=2, name="t4_fill.png")


def fig_branch_coh():
    ims, labs = [], []
    for m in [0.4, 1.0]:
        for bc in [0.0, None]:
            P = {} if bc is None else dict(branch_coh=0.0, _user={"branch_coh": 1})
            tr = grow.grow("spruce", "mature", seed=2, mpp=m)
            Hp = int((tr.H + 1) / m) + 4
            rgb, _ = conifer.tree("spruce", "mature", seed=2, mpp=m, W=max(30, int(Hp * 0.5)), H=Hp, tr=tr, P=P)
            ims.append(rgb); labs.append(f"{m} m/px " + ("per spray" if bc == 0.0 else "per branch"))
    k = [6, 6, 12, 12]
    out = [up(i, kk) for i, kk in zip(ims, k)]
    Hm = max(o.shape[0] for o in out)
    padded = []
    for o in out:
        p = np.full((Hm, o.shape[1], 3), 250, np.uint8); p[Hm - o.shape[0]:] = o; padded.append(p)
    panel(padded, labs, k=1, name="t4_branch_coh.png")


def fig_ladder():
    MPPS = [0.1, 0.16, 0.25, 0.4, 0.65, 1.0, 1.6, 2.5]
    tr0 = grow.grow("hemlock", "mature", seed=2, mpp=1.0)
    cells = []
    for m in MPPS:
        Hp = int(tr0.H * 1.05 / m) + 4; Wp = max(10, int(Hp * 0.45))
        rgb, _ = conifer.tree("hemlock", "mature", seed=2, mpp=m, W=Wp, H=Hp)
        cells.append(rgb)
    Hm = max(c.shape[0] for c in cells)
    pad = []
    for c in cells:
        p = np.full((Hm, c.shape[1], 3), 250, np.uint8); p[Hm - c.shape[0]:] = c; pad.append(p)
    panel(pad, [str(m) for m in MPPS], k=2, gap=3, name="t4_ladder_2x.png")


# ------------------------------------------------------------------ 5. stands
def fig_stand_shadow():
    import sheets
    T = sheets.patch(None, 34, 110, 70, 11, mix=sheets.SPECIES)
    ims, labs = [], []
    for ss in [False, True]:
        rgb, _ = conifer.stand([dict(t) for t in T], 0.35, 300, 170, hour="golden", origin=(150, 166),
                               ground="floor", stand_shadow=ss)
        ims.append(rgb); labs.append("golden, " + ("shared shadow map" if ss else "each tree alone"))
    panel(ims, labs, k=2, name="t5_shared_shadow.png")


def fig_depth():
    import sheets
    T = sheets.patch(None, 34, 110, 70, 11, mix=sheets.SPECIES)
    ims, labs = [], []
    for far in [300.0, 2000.0]:
        rgb, _ = conifer.stand([dict(t) for t in T], 0.35, 300, 170, hour="morning", origin=(150, 166),
                               ground="floor", far=far)
        ims.append(rgb); labs.append(f"aerial depth keyed to far={far:.0f} m")
    panel(ims, labs, k=2, name="t5_depth.png")


def fig_ramps():
    """The palette is the light: each hour's 8-step foliage ramp per species."""
    hrs = list(palette.HOURS)
    rows = []
    for sp in ["douglas", "hemlock", "cedar", "spruce"]:
        row = np.concatenate([np.repeat(palette.foliage_ramp(sp, h)[None], 3, 0) for h in hrs], 1)
        rows.append(row)
        rows.append(np.full((1, row.shape[1], 3), 250, np.uint8))
    img = up(np.concatenate(rows, 0), 12)
    out = np.full((img.shape[0] + 16, img.shape[1] + 70, 3), 250, np.uint8); out[16:, 70:] = img
    im = Image.fromarray(out); d = ImageDraw.Draw(im)
    for i, h in enumerate(hrs):
        d.text((70 + i * 8 * 12 + 2, 2), h, fill=(0, 0, 0))
    for j, sp in enumerate(["douglas", "hemlock", "cedar", "spruce"]):
        d.text((2, 16 + j * 4 * 12 + 10), sp, fill=(0, 0, 0))
    im.save(os.path.join(OUT, "t5_ramps.png")); print("wrote t5_ramps.png")


# ------------------------------------------------------------------ 6. cedar buttress
def fig_buttress():
    ims, labs = [], []
    for lab, over in [("flutes=7 (final)", {}), ("flutes=0", dict(flutes=0)), ("buttress=0", dict(flutes=0, buttress=0.0))]:
        tr = grow.grow("cedar", "old", seed=1, mpp=0.03, **over)
        G, st = paint.paint_tree(tr, 0.03, 200, 270, (100, 264), sun=(20, 30), P=dict(seed=1, lit_frac=0.42))
        img = paint.compose(st, G["mat"], palette.palette("cedar", "morning"), palette.sky("morning", 270, 200))
        img[264:] = palette.ground_ramp("morning")[1]
        ims.append(img); labs.append(lab)
    panel(ims, labs, k=2, name="t6_buttress.png")


FIGS = {k[4:]: v for k, v in globals().items() if k.startswith("fig_")}

if __name__ == "__main__":
    for n in (sys.argv[1:] or list(FIGS)):
        t0 = time.time(); FIGS[n](); print(n, f"{time.time() - t0:.0f}s", flush=True)
