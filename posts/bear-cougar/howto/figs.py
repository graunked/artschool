"""The few figures on these pages that the study never saved, made by importing the study's own
code unchanged from ~/work/pixelart-studies/elements/bear-cougar/src (nothing is written there).

    cd ~/work/artschool/posts/bear-cougar/howto
    PYTHONDONTWRITEBYTECODE=1 nice -n 10 ~/miniconda3/bin/python3 figs.py
"""
import os, sys, io, contextlib
sys.dont_write_bytecode = True
SRC = os.path.expanduser("~/work/pixelart-studies/elements/bear-cougar/src")
sys.path.insert(0, SRC)
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from PIL import Image, ImageDraw
import rig, paint, palette as P, render
from render import View

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "fig")
os.makedirs(FIG, exist_ok=True)


def up(a, z):
    return a.repeat(z, 0).repeat(z, 1)


def legs():
    """Side view of the rigs at a standing pose: every leg segment's centreline, joints as dots,
    in shoulder heights; the metapodial (the stance angle that IS the species) in red."""
    tiles = []
    for spn in ("bear", "cougar"):
        ind = rig.Individual(spn, "f", "adult", "summer", 1)
        sp = ind.spec()
        tubes, _ = rig.build(ind, rig.gait_pose(sp, "stand"))
        S = 260
        W, H = int(2.4 * S), int(1.35 * S)
        im = Image.new("RGB", (W, H), (244, 242, 236))
        d = ImageDraw.Draw(im)
        X = lambda p: (W * 0.55 + p[0] / sp.H * S, H - 12 - p[2] / sp.H * S)
        d.line([(0, H - 12), (W, H - 12)], fill=(150, 150, 150))
        d.line([(0, H - 12 - S), (W, H - 12 - S)], fill=(200, 200, 230))
        for t in tubes:
            if t.c[:, 1].mean() < -0.01 * sp.H and t.name[:2] in ("LF", "RF", "LH", "RH"):
                continue                      # near side only
            col = (60, 60, 60)
            if t.name[2:] == "3":
                col = (200, 40, 40)
            elif t.name[2:] in ("1", "2", "pw", "hk", "fl", "cl"):
                col = (40, 90, 160)
            elif t.name == "tail":
                col = (120, 90, 40)
            pts = [X(p) for p in t.c]
            d.line(pts, fill=col, width=3 if col != (60, 60, 60) else 1)
            if t.name[2:] in ("1", "2", "3"):
                for q in (pts[0], pts[-1]):
                    d.ellipse([q[0] - 3, q[1] - 3, q[0] + 3, q[1] + 3], fill=(0, 0, 0))
        d.text((6, 6), f"{spn}: fore metapodial {sp.fore_meta:.0f} deg, hind {sp.hind_meta:.0f} deg from vertical (red)", fill=(0, 0, 0))
        d.text((6, 20), "blue: upper bone, lower bone, paw   grey: torso, neck, head   lines at 0 and 1 H", fill=(60, 60, 60))
        tiles.append(np.array(im))
    out = np.concatenate(tiles, 0)
    Image.fromarray(out).save(os.path.join(FIG, "legs.png"))


def tail_sweep():
    """The cougar's tail chain: droop (rows) x lift (columns), notan at 40 px, with the numbers
    I tuned it by: the lowest point and the tip height in shoulder heights."""
    rows = []
    lines = []
    for droop in (0.3, 0.62, 0.9):
        cells = []
        for lift in (0.2, 0.6, 0.9):
            ind = rig.Individual("cougar", "f", "adult", "summer", 1)
            sp = ind.spec()
            pose = rig.gait_pose(sp, "stand")
            pose.tail = (-38, droop, lift)
            tubes, _ = rig.build(ind, pose)
            tl = [t for t in tubes if t.name == "tail"][0]
            z = tl.c[:, 2] / sp.H
            R, _, _ = paint.paint_index(ind, "stand", 0, View(yaw=0, elev=0, sun_az=-150, sun_el=40), size_px=40, pose=pose)
            n = np.stack([paint.notan(R)] * 3, -1)
            im = Image.fromarray(up(n, 3)); dr = ImageDraw.Draw(im)
            dr.text((4, 2), f"droop {droop} lift {lift}", fill=(0, 0, 0))
            dr.text((4, 14), f"low {z.min():.2f} H  tip {z[-1]:.2f} H", fill=(160, 0, 0))
            cells.append(np.array(im))
            lines.append(f"droop {droop:4} lift {lift:3}  lowest {z.min():.2f} H  tip {z[-1]:.2f} H")
        h = max(c.shape[0] for c in cells); w = sum(c.shape[1] + 8 for c in cells)
        r = np.full((h, w, 3), 255, np.uint8); x = 0
        for c in cells:
            r[h - c.shape[0]:, x:x + c.shape[1]] = c; x += c.shape[1] + 8
        rows.append(r)
    W = max(r.shape[1] for r in rows)
    out = np.full((sum(r.shape[0] + 8 for r in rows), W, 3), 255, np.uint8); y = 0
    for r in rows:
        out[y:y + r.shape[0], :r.shape[1]] = r; y += r.shape[0] + 8
    Image.fromarray(out).save(os.path.join(FIG, "tail_sweep.png"))
    open(os.path.join(FIG, "tail_sweep.txt"), "w").write("\n".join(lines) + "\n")


def ramps():
    """The black bear's body ramp at each hour: with the hide defaults every other animal uses
    (warm and cool pulls as given) and with the black-fur rule (cool x 0.25, warm x 0.6)."""
    d, l = P.BEAR_BLACK[rig.BODY]
    Z = 28
    rows = []
    for hour in ("dawn", "morning", "noon", "golden", "dusk", "overcast"):
        h = P.HOURS[hour]
        a = P.ramp(d, l, 6, h["sun"], h["amb"], h["warm"], h["cool"], h["key"])
        b = P.ramp(d, l, 6, h["sun"], h["amb"], h["warm"] * 0.6, h["cool"] * 0.25, h["key"])
        row = np.full((Z, Z * 13, 3), 255, np.uint8)
        for i, c in enumerate(a):
            row[:, i * Z:(i + 1) * Z] = c
        for i, c in enumerate(b):
            row[:, (7 + i) * Z:(8 + i) * Z] = c
        rows.append(row)
    out = np.concatenate([np.pad(r, ((0, 4), (0, 0), (0, 0)), constant_values=255) for r in rows], 0)
    im = Image.fromarray(np.pad(out, ((18, 0), (70, 0), (0, 0)), constant_values=255)); dr = ImageDraw.Draw(im)
    dr.text((70, 3), "hide defaults (steps 0-5)", fill=(0, 0, 0)); dr.text((70 + 7 * Z, 3), "black-fur rule", fill=(0, 0, 0))
    for i, hour in enumerate(("dawn", "morning", "noon", "golden", "dusk", "overcast")):
        dr.text((4, 18 + i * (Z + 4) + 8), hour, fill=(0, 0, 0))
    im.save(os.path.join(FIG, "bear_ramps.png"))


def values_same_region():
    """The value board, re-run with one fix I found while writing these pages: valboard.py
    quantised the photo over its whole crop (grass included) but my render over the animal
    only, which stretched my animal across all six steps. Here both are quantised over the
    whole cell, background included, exactly as the eye compares them."""
    import valboard as V
    from board import entries
    rows = []
    for e in entries():
        f, crop, gr, wr, (spn, sex), g, ph, yaw, lab = e[:9]
        elev = e[9] if len(e) > 9 else 0
        im = Image.open(f).convert("RGB").crop(crop)
        ind = rig.Individual(spn, sex, "adult", "summer", 1)
        sp = ind.spec()
        hbH = paint.hb_m(ind) / sp.H
        s = (60 / hbH) / (gr - wr)
        p = np.array(im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BOX))
        R, _, _ = paint.paint_index(ind, g, ph, View(yaw=yaw, elev=max(elev, 4), sun_el=45, sun_az=-145), size_px=60)
        mine = paint.colour(R, ind, "morning", rng=np.random.default_rng(2))
        qp = V.quant(V.lum(p), np.ones(p.shape[:2], bool))
        qm = V.quant(V.lum(mine), np.ones(mine.shape[:2], bool))
        hh = max(p.shape[0], mine.shape[0])
        r = np.full((hh, p.shape[1] * 2 + mine.shape[1] * 2 + 18, 3), 255, np.uint8); x = 0
        for a in (p, np.stack([qp] * 3, -1), mine, np.stack([qm] * 3, -1)):
            r[hh - a.shape[0]:, x:x + a.shape[1]] = a; x += a.shape[1] + 6
        rows.append(r)
    W = max(r.shape[1] for r in rows)
    out = np.full((sum(r.shape[0] + 4 for r in rows), W, 3), 255, np.uint8); y = 0
    for r in rows:
        out[y:y + r.shape[0], :r.shape[1]] = r; y += r.shape[0] + 4
    Image.fromarray(up(out, 4)).save(os.path.join(FIG, "values_same_region.png"))


def glyph_prints():
    """The character prints I debugged the glyph painter by (glyphdbg.py's format), for the
    final code: bear and cougar walking at 8, 6 and 4 px."""
    out = []
    ch = {-1: ".", 0: "B", 1: "D", 2: "p", 3: "b", 11: "T", 12: "M", 16: "k"}
    for spn, sex in (("bear", "m"), ("cougar", "f")):
        for L in (8, 6, 4):
            ind = rig.Individual(spn, sex, "adult", "summer", 3)
            R, tubes, k = paint.paint_index(ind, "walk", 0.3, View(yaw=-12, elev=6, sun_az=-150, sun_el=35), size_px=L)
            rows = np.nonzero((R.mat >= 0).any(1))[0]; cols = np.nonzero((R.mat >= 0).any(0))[0]
            out.append(f"{spn} walk, {L} px")
            for y in range(rows.min(), rows.max() + 1):
                out.append("".join((ch.get(int(R.mat[y, x]), "?") + str(R.step[y, x])) if R.mat[y, x] >= 0 else " ." for x in range(cols.min(), cols.max() + 1)))
            out.append("")
    open(os.path.join(FIG, "glyph_prints.txt"), "w").write("\n".join(out))


def gif_strip(name, n=6, z=None):
    """Frames of one of the study's GIFs side by side, for the page."""
    im = Image.open(os.path.expanduser(f"~/work/pixelart-studies/elements/bear-cougar/sheets/anim/{name}.gif"))
    fr = []
    for i in range(0, im.n_frames, max(1, im.n_frames // n)):
        im.seek(i); fr.append(im.convert("RGB").copy())
    W = sum(f.width + 4 for f in fr); H = max(f.height for f in fr)
    s = Image.new("RGB", (W, H), (24, 24, 28)); x = 0
    for f in fr:
        s.paste(f, (x, 0)); x += f.width + 4
    s.save(os.path.join(FIG, f"strip_{name}.png"))


if __name__ == "__main__":
    which = sys.argv[1:] or ["legs", "tail_sweep", "ramps", "values_same_region", "glyph_prints", "strips"]
    for w in which:
        if w == "strips":
            for g in ("bear_walk_48", "bear_pace_48", "cougar_walk_48", "cougar_run_48", "cougar_pounce_48", "bear_climb_40"):
                gif_strip(g)
        else:
            globals()[w]()
        print(w, flush=True)
