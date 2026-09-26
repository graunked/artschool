"""Labelled parameter sheets.   nice -n 10 python3 sheets.py [A B C D E F]

A  hour x distance        B  species x distance     C  moisture x grazing
D  wind                   E  camera angle x distance F  seeds
Each cell is the same machinery (grasslod.paint) with one or two parameters changed.
"""
import os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from PIL import Image, ImageDraw
from multiprocessing import Pool

OUT = os.path.join(os.path.dirname(__file__), "..", "sheets")
CW, CH, SC = 240, 150, 2


def cell(args):
    wkw, ckw, hour, mkw = args
    from grasslod.world import World
    from grasslod.render import Camera
    from grasslod.painter import paint, centre_on_bank
    from grasslod import palette as P
    w = World(**wkw)
    cy, cz = centre_on_bank(w)
    cam = Camera(W=CW, H=CH, cx=0, cy=cy, cz=cz, **ckw)
    idx, pal = paint(w, cam, hour, mparams=mkw)
    return P.to_rgb(idx, pal)


def sheet(name, rows, cols, make, title):
    """make(r, c) -> (world kw, camera kw, hour, mark kw)"""
    jobs = [make(r, c) for r in rows for c in cols]
    with Pool(4) as p:
        tiles = p.map(cell, jobs)
    lw, th = 110, 26
    W = lw + len(cols) * (CW * SC + 6)
    H = th + 18 + len(rows) * (CH * SC + 6)
    im = Image.new("RGB", (W, H), (24, 24, 28))
    d = ImageDraw.Draw(im)
    d.text((8, 6), title, fill=(235, 235, 235))
    for j, c in enumerate(cols):
        d.text((lw + j * (CW * SC + 6) + 4, th), str(c[0] if isinstance(c, tuple) else c), fill=(200, 200, 200))
    k = 0
    for i, r in enumerate(rows):
        y = th + 18 + i * (CH * SC + 6)
        d.text((6, y + 4), str(r[0] if isinstance(r, tuple) else r), fill=(200, 200, 200))
        for j, c in enumerate(cols):
            t = Image.fromarray(tiles[k]).resize((CW * SC, CH * SC), Image.NEAREST)
            im.paste(t, (lw + j * (CW * SC + 6), y))
            k += 1
    os.makedirs(OUT, exist_ok=True)
    im.save(os.path.join(OUT, f"sheet_{name}.png"))
    print("wrote", name)


DIST = [("30 px/m", 30), ("10 px/m", 10), ("3 px/m", 3), ("0.8 px/m", 0.8)]


def main(which):
    if "A" in which:
        hours = ["day", "golden", "dusk", "overcast"]
        sheet("A_hour_x_distance", hours, DIST,
              lambda r, c: (dict(seed=1), dict(ppm=c[1], theta=30), r, None),
              "A  hour (rows) x distance (cols); meadow, theta 30")
    if "B" in which:
        sp = ["turf", "meadow", "hay", "tussock", "sedge"]
        sheet("B_species_x_distance", sp, DIST[:3],
              lambda r, c: (dict(seed=1, species=r), dict(ppm=c[1], theta=30), "golden", None),
              "B  species (rows) x distance (cols); golden hour")
    if "C" in which:
        mo = [("dry 0.1", 0.1), ("mesic 0.5", 0.5), ("wet 0.9", 0.9)]
        gr = [("ungrazed 0", 0.0), ("grazed 0.5", 0.5), ("cropped 0.9", 0.9)]
        sheet("C_moisture_x_grazing", mo, gr,
              lambda r, c: (dict(seed=1, moisture=r[1], grazing=c[1]), dict(ppm=12, theta=30), "day", None),
              "C  moisture (rows) x grazing (cols); day, 12 px/m")
    if "D" in which:
        wd = [("calm 0", 0.0), ("breeze 0.5", 0.5), ("wind 1.0", 1.0)]
        sheet("D_wind", [("20 px/m", 20), ("6 px/m", 6)], wd,
              lambda r, c: (dict(seed=1, wind=c[1], species="hay"), dict(ppm=r[1], theta=30), "golden", None),
              "D  wind (cols): lean and combing sheen; hay, golden")
    if "E" in which:
        th = [("theta 12", 12), ("theta 30", 30), ("theta 55", 55), ("theta 90 (top-down)", 90)]
        sheet("E_camera_x_distance", th, DIST,
              lambda r, c: (dict(seed=1), dict(ppm=c[1], theta=r[1]), "golden", None),
              "E  camera elevation (rows) x distance (cols); golden")
    if "F" in which:
        sd = [("seed 1", 1), ("seed 2", 2), ("seed 3", 3)]
        sheet("F_seeds", sd, DIST[:3],
              lambda r, c: (dict(seed=r[1]), dict(ppm=c[1], theta=30), "day", None),
              "F  seeds (rows) x distance (cols); day")


if __name__ == "__main__":
    main(sys.argv[1:] or list("ABCDEF"))
