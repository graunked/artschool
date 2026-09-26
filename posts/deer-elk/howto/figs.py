"""Figures for the deer-elk tutorial pages, made by running the study's own code.

Nothing here is a new painter: every animal is painted by `paint.paint_index` / `herd.scene` from
~/work/pixelart-studies/elements/deer-elk/src, exactly as in the study. Where a figure shows a
step switched off (fringe off, checker instead of hair dashes), that is done by swapping one of
the study's own functions for another of its own functions for the length of one render.

    cd ~/work/artschool/posts/deer-elk/howto && nice -n 10 python3 figs.py [name ...]

Writes only into ./fig/. Takes about 3 minutes.
"""
import os, sys
sys.dont_write_bytecode = True                       # leave the study folder untouched
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
STUDY = "~/work/pixelart-studies/elements/deer-elk/src"
sys.path.insert(0, STUDY)
os.chdir(STUDY)                                       # some study modules use relative ref paths
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import rig, paint, render, hide, fringe, palette as P, herd
from render import View

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
os.makedirs(OUT, exist_ok=True)


def font(sz=13):
    for p in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def up(a, z):
    return a.repeat(z, 0).repeat(z, 1)


def row(cells, labels=None, z=6, gap=4, bg=(24, 24, 28), label_h=18, bottom=True):
    """cells: list of HxWx3 (or HxW) uint8, pasted bottom-aligned, magnified by z, labelled."""
    cells = [np.stack([c] * 3, -1) if c.ndim == 2 else c for c in cells]
    H = max(c.shape[0] for c in cells) * z
    W = sum(c.shape[1] * z + gap for c in cells) - gap
    lh = label_h if labels else 0
    im = Image.new("RGB", (W, H + lh), bg)
    d = ImageDraw.Draw(im)
    x = 0
    for i, c in enumerate(cells):
        t = up(c, z)
        im.paste(Image.fromarray(t), (x, lh + (H - t.shape[0] if bottom else 0)))
        if labels:
            d.text((x + 2, 2), labels[i], fill=(220, 220, 225), font=font(12))
        x += t.shape[1] + gap
    return im


def stack(ims, gap=6, bg=(24, 24, 28)):
    W = max(i.width for i in ims)
    out = Image.new("RGB", (W, sum(i.height + gap for i in ims) - gap), bg)
    y = 0
    for i in ims:
        out.paste(i, (0, y)); y += i.height + gap
    return out


def save(im, name):
    if isinstance(im, np.ndarray):
        im = Image.fromarray(im)
    im.save(os.path.join(OUT, name))
    print(name, im.size)


SIDE = dict(yaw=-14, elev=8, sun_az=-150, sun_el=35)

# material false colours for the region map
MATCOL = {rig.BODY: (210, 180, 130), rig.DARK: (90, 60, 40), rig.PALE: (245, 235, 200), rig.BELLY: (130, 90, 60),
          rig.LEG: (70, 50, 40), rig.HOOF: (20, 20, 20), rig.ANTLER: (120, 80, 50), rig.TIP: (255, 255, 240),
          rig.NOSE: (0, 0, 0), rig.EARIN: (230, 160, 160), rig.THROAT: (255, 255, 255), rig.TAILTOP: (10, 10, 10),
          rig.SPOT: (255, 230, 120), rig.VELVET: (150, 120, 100), rig.EYE: (0, 0, 0), rig.FACE: (170, 110, 70), 19: (255, 150, 60)}


def matmap(R, bg=(60, 90, 60)):
    img = np.zeros(R.mat.shape + (3,), np.uint8); img[:] = bg
    for m, c in MATCOL.items():
        img[R.mat == m] = c
    return img


# ------------------------------------------------------------------ how-to 1: boards

def f_boards():
    import board
    for notan in (True, False):
        board.NOTAN = notan
        ps = [board.panel(i) for i in range(len(board.BOARD))]
        W = max(p.shape[1] for p in ps)
        out = np.full((sum(p.shape[0] for p in ps), W), 255, np.uint8)
        y = 0
        for p in ps:
            out[y:y + p.shape[0], :p.shape[1]] = p; y += p.shape[0]
        save(out, "board_final_%s.png" % ("notan" if notan else "grey"))


def f_sizes():
    """Photo area-reduced beside the final rig at 60/30/16/8 px, colour (sizes_c.py's layout)."""
    from board import BOARD
    rows = []
    for (f, crop, gr, wr, (spn, sex), g, ph, yaw) in BOARD:
        im = Image.open(f).convert("RGB").crop(crop)
        cells = []
        for L in (60, 30, 16, 8):
            s = (L / 1.62) / (gr - wr)
            p = np.array(im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BOX))
            ind = rig.Individual(spn, sex, "adult", "summer" if spn == "deer" else "autumn", 1)
            R, _, _ = paint.paint_index(ind, g, ph, View(yaw=yaw + (-14 if yaw == 0 else 14), elev=6, sun_el=45, sun_az=-155 if yaw == 0 else -25), size_px=L)
            cells += [p, paint.colour(R, ind, "morning")]
        rows.append(row(cells, z=4, gap=6))
    save(stack(rows), "sizes_final_colour.png")


# ------------------------------------------------------------------ how-to 2: the rig

def f_skeleton():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (spn, sex, g) in zip(axs, [("elk", "f", "stand"), ("elk", "m", "stand"), ("deer", "f", "alert")]):
        ind = rig.Individual(spn, sex, "adult", "autumn", 1)
        sp = ind.spec()
        tubes, _ = rig.build(ind, rig.gait_pose(sp, g, 0))
        H = sp.H
        for t in tubes:
            if t.c[:, 1].mean() < -0.01 * H:                       # far side: skip for clarity
                continue
            c = t.c / H
            if t.name == "torso":
                st = np.array(sp.torso)
                ax.plot(st[:, 0], st[:, 1], "o-", c="#c0392b", ms=4, lw=1.5, label="torso stations: back")
                ax.plot(st[:, 0], st[:, 2], "o-", c="#2471a3", ms=4, lw=1.5, label="torso stations: belly")
                continue
            # outline of the tube in the side view: centreline +- the up-radius
            ax.plot(c[:, 0], c[:, 2], "-", c="0.35", lw=0.8)
            if not t.thin:
                ax.fill_between(c[:, 0], c[:, 2] - t.a / H, c[:, 2] + t.a / H, color="0.8", alpha=0.5, lw=0)
        ax.axhline(0, c="k", lw=0.6)
        ax.set_aspect("equal"); ax.set_xlim(-0.9, 1.25); ax.set_ylim(-0.03, 1.75)
        ax.set_title(f"{spn} {'bull' if (spn=='elk' and sex=='m') else ('cow' if spn=='elk' else 'doe')} ({g}); units = shoulder height H")
        ax.grid(alpha=0.25)
    axs[0].legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "skeleton.png"), dpi=90)
    print("skeleton.png")


def f_ages():
    cells, labels = [], []
    for spn, sex, age, season in [("elk", "m", "adult", "autumn"), ("elk", "f", "adult", "autumn"), ("elk", "m", "yearling", "autumn"),
                                  ("elk", "f", "young", "summer"), ("deer", "m", "adult", "autumn"), ("deer", "f", "adult", "summer"),
                                  ("deer", "f", "young", "summer")]:
        ind = rig.Individual(spn, sex, age, season, 4)
        # every individual at its own true size relative to the bull (k fixed)
        k = 60 / (1.62 * rig.Individual("elk", "m", "adult").spec().H)
        R, _, _ = paint.paint_index(ind, "stand", 0, View(**SIDE), size_px=1.62 * ind.spec().H * k)
        cells.append(paint.colour(R, ind, "morning", ground=season)); labels.append(f"{spn} {sex} {age}")
    save(row(cells, labels, z=3), "ages_true_scale.png")


# ------------------------------------------------------------------ how-to 3: gaits

def f_footfalls():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sp = rig.Individual("elk", "f").spec()
    gaits = ["walk", "trot", "gallop", "bound", "stot"]
    fig, axs = plt.subplots(len(gaits), 1, figsize=(9, 7.5), sharex=True)
    ph = np.linspace(0, 1, 200, endpoint=False)
    for ax, g in zip(axs, gaits):
        for j, L in enumerate(rig.LEG_NAMES):
            z = np.array([rig.gait_pose(sp, g, p).legs[L][1][1] for p in ph])
            bz = np.array([rig.gait_pose(sp, g, p).body_z for p in ph])
            onground = (z + (bz if g == "stot" else 0)) <= 1e-6 if g != "stot" else (np.array([max(rig.gait_pose(sp, g, p).body_z, 0) for p in ph]) <= 1e-6)
            y = len(rig.LEG_NAMES) - j
            ax.fill_between(ph, y - 0.35, y + 0.35, where=onground, color="#6e5a3a", step="mid")
        ax.set_yticks(range(1, 5)); ax.set_yticklabels(rig.LEG_NAMES[::-1])
        ax.set_ylabel(g, rotation=0, ha="right", va="center", fontsize=11)
        ax.set_ylim(0.4, 4.6)
    axs[-1].set_xlabel("phase of the stride (bars = hoof on the ground)")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "footfalls.png"), dpi=90)
    print("footfalls.png")


def f_stot_check():
    ind = rig.Individual("deer", "m", "adult", "autumn", 3)
    cells, labels = [], []
    for ph in (0.0, 0.1, 0.3, 0.55, 0.8):
        R, tubes, k = paint.paint_index(ind, "stot", ph, View(yaw=0, elev=6, sun_az=-150, sun_el=35), size_px=48)
        on = R.mat >= 0
        lowest = np.nonzero(on.any(1))[0].max()
        cells.append(paint.colour(R, ind, "morning"))
        labels.append(f"phase {ph}: hooves {R.ground_row - lowest:+d} px")
    save(row(cells, labels, z=4), "stot_check.png")


# ------------------------------------------------------------------ how-to 4: the hide

def f_pipeline():
    ind = rig.Individual("elk", "m", "adult", "autumn", 2)
    R, tubes, k = paint.paint_index(ind, "stand", 0, View(**SIDE), size_px=60)
    xg = (np.where(R.mat >= 0, R.x / 5.0, 0.85) * 255).astype(np.uint8)
    cells = [paint.notan(R), matmap(R), xg, paint.grey(R, bg=0.85), paint.colour(R, ind, "morning")]
    save(row(cells, ["1 silhouette", "2 hide regions", "3 value field x (0..5)", "4 quantised steps", "5 palette: morning"], z=5), "pipeline_bull60.png")


def with_swap(mod, name, fn, call):
    old = getattr(mod, name)
    setattr(mod, name, fn)
    try:
        return call()
    finally:
        setattr(mod, name, old)


def f_dither():
    ind = rig.Individual("elk", "f", "adult", "autumn", 1)

    def go():
        R, _, _ = paint.paint_index(ind, "stand", 0, View(**SIDE), size_px=60)
        return paint.colour(R, ind, "morning")
    a = with_swap(hide, "hair_quant", lambda x, seed=0: hide.bayer_quant(x), go)
    b = go()
    save(row([a, b], ["2x2 Bayer (rock)", "1x2 hair dashes (hide)"], z=7), "dither_rock_vs_hair.png")


def f_fringe():
    ind = rig.Individual("elk", "m", "adult", "autumn", 2)

    def go():
        R, _, _ = paint.paint_index(ind, "stand", 0, View(**SIDE), size_px=60)
        return paint.colour(R, ind, "morning")
    a = with_swap(fringe, "hang", lambda R, *a, **k: R, lambda: with_swap(fringe, "interlock", lambda R, *a, **k: R, go))
    b = go()
    save(row([a, b], ["fur edges off", "interlock + hanging mane"], z=7), "fringe_off_on.png")


def f_flatten():
    ind = rig.Individual("elk", "f", "adult", "autumn", 1)
    cells = []
    for fl in (0.5, 0.2):
        orig = hide.apply

        def ap(R, view, tubes, size_px, flatten=0.5, dark_mats=None, hair=True, _fl=fl):
            return orig(R, view, tubes, size_px, flatten=_fl, dark_mats=dark_mats, hair=hair)
        cells.append(with_swap(hide, "apply", ap, lambda: paint.colour(paint.paint_index(ind, "stand", 0, View(**SIDE), size_px=60)[0], ind, "morning")))
    save(row(cells, ["flatten 0.5 (the boulder's)", "flatten 0.2 (hide)"], z=7), "flatten.png")


# ------------------------------------------------------------------ how-to 5: light

def f_rim():
    cells, labels = [], []
    for spn, sex in (("elk", "m"), ("deer", "f")):
        ind = rig.Individual(spn, sex, "adult", "autumn", 2)
        for az, lab in ((-150, "golden, sun in front"), (150, "golden, sun behind")):
            R, _, _ = paint.paint_index(ind, "stand" if spn == "elk" else "alert", 0, View(yaw=-20, elev=8, sun_az=az, sun_el=12), size_px=56)
            cells.append(paint.colour(R, ind, "golden", ground="autumn")); labels.append(f"{spn}: {lab}")
    save(row(cells, labels, z=5), "rim_front_vs_back.png")


def f_swatches():
    hours = list(P.HOURS)
    ind = rig.Individual("elk", "m", "adult", "autumn", 1)
    mats = [(rig.BODY, "body"), (rig.DARK, "mane"), (rig.PALE, "rump"), (rig.ANTLER, "antler"), (P.RIM, "rim"), (P.GROUND, "grass"), (P.GSHADOW, "shadow")]
    sw = 14
    im = Image.new("RGB", (90 + len(mats) * (6 * sw + 10), 24 + len(hours) * (sw + 6)), (24, 24, 28))
    d = ImageDraw.Draw(im)
    for j, (m, lab) in enumerate(mats):
        d.text((90 + j * (6 * sw + 10), 4), lab, fill=(220, 220, 225), font=font(11))
    for i, h in enumerate(hours):
        tab = P.palette(ind, h, 0, "autumn")
        y = 22 + i * (sw + 6)
        d.text((6, y), h, fill=(220, 220, 225), font=font(12))
        for j, (m, lab) in enumerate(mats):
            for s in range(6):
                x = 90 + j * (6 * sw + 10) + s * sw
                d.rectangle([x, y, x + sw - 1, y + sw - 1], fill=tuple(int(v) for v in tab[m][s]))
    save(im, "hour_ramps.png")


# ------------------------------------------------------------------ how-to 6: small figures

def f_handoff():
    ind = rig.Individual("elk", "m", "adult", "autumn", 3)
    tab = P.palette(ind, "morning", 0, "summer")
    cells, labels = [], []
    for L in (18, 16, 14, 12, 11, 10, 9, 8, 7, 6, 5, 4):
        R, _, _ = paint.paint_index(ind, "walk", 0.3, View(**SIDE), size_px=L)
        img = np.zeros(R.mat.shape + (3,), np.uint8); img[:] = tab[P.GROUND][3]
        img[R.ground_shadow > 0.45] = tab[P.GSHADOW][2]
        on = R.mat >= 0
        img[on] = tab[R.mat[on], R.step[on]]
        cells.append(img); labels.append(f"{L}" + (" glyph" if L < paint.MICRO else ""))
    save(row(cells, labels, z=9), "handoff_bull.png")


# ------------------------------------------------------------------ how-to 7: herds

def f_herdplan():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, beh in zip(axs, ["graze", "alert", "file", "bedded"]):
        mem = herd.elk_herd(beh, 26, seed=7, heading=12 if beh != "file" else 6)
        for m in mem:
            c = {"young": "#e67e22", "yearling": "#7f8c8d"}.get(m.ind.age, "#8e44ad" if m.ind.sex == "m" else "#6e5a3a")
            hd = np.radians(m.heading)
            ax.arrow(m.x, m.y, 2.0 * np.cos(hd), 2.0 * np.sin(hd), head_width=0.9, color=c, length_includes_head=True)
            if m.gait == "alert":
                ax.plot(m.x, m.y, "o", mfc="none", mec="r", ms=9)
            if m.gait == "lie":
                ax.plot(m.x, m.y, "s", c=c, ms=4)
        ax.set_aspect("equal"); ax.set_title(beh + " (metres; red ring = head up)"); ax.grid(alpha=0.25)
    axs[0].plot([], [], c="#6e5a3a", label="cow"); axs[0].plot([], [], c="#e67e22", label="calf"); axs[0].plot([], [], c="#7f8c8d", label="yearling")
    axs[0].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "herd_plans.png"), dpi=85)
    print("herd_plans.png")


def f_herd_scenes():
    from multiprocessing import Pool
    specs = [("graze_iso_10", "graze", 10, 30, 340, 200), ("file_iso_3", "file", 3, 30, 320, 120),
             ("far_iso_1.6", "graze", 1.6, 30, 140, 80), ("flee_land_5", "flee", 5, 8, 320, 100)]
    with Pool(4) as pool:
        for name, beh, k, elev, W, H in specs:
            mem = herd.elk_herd(beh, 26, seed=7, heading=12 if beh not in ("file", "flee") else 6)
            img = herd.scene(mem, k, W, H, elev=elev, hour="morning", seed=3, pool=pool)
            save(up(img, 3), f"herd_{name}.png")


ALL = [f_boards, f_sizes, f_skeleton, f_ages, f_footfalls, f_stot_check, f_pipeline, f_dither, f_fringe, f_flatten,
       f_rim, f_swatches, f_handoff, f_herdplan, f_herd_scenes]

if __name__ == "__main__":
    names = sys.argv[1:]
    for f in ALL:
        if not names or f.__name__[2:] in names:
            f()
