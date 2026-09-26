"""figs.py: the stage figures for LESSONS.md (reference board, 8x comparisons)."""
import os
import numpy as np
from PIL import Image, ImageDraw
HERE = os.path.dirname(__file__)
ST = os.path.join(HERE, "..", "..", "..")
IMG = os.path.join(HERE, "..", "img")
LW = os.path.join(ST, "pedagogy", "master-copy", "ref", "lw")
NAT = os.path.join(ST, "pedagogy", "master-copy", "ref", "native.png")
V04 = os.path.join(ST, "grass", "refs", "ferrari_V04.png")


def crop(path, x, y, w, h, k):
    im = Image.open(path).convert("RGB").crop((x, y, x + w, y + h))
    return im.resize((w * k, h * k), Image.NEAREST)


def board(items, cols, out, title, bg=(24, 24, 28)):
    ims = [(lab, im) for lab, im in items]
    cw = max(im.width for _, im in ims); ch = max(im.height for _, im in ims)
    rows = (len(ims) + cols - 1) // cols
    B = Image.new("RGB", (cols * (cw + 8) + 8, 24 + rows * (ch + 22)), bg)
    d = ImageDraw.Draw(B); d.text((8, 6), title, fill=(230, 230, 230))
    for i, (lab, im) in enumerate(ims):
        x = 8 + (i % cols) * (cw + 8); y = 24 + (i // cols) * (ch + 22)
        d.text((x, y), lab, fill=(200, 200, 200)); B.paste(im, (x, y + 14))
    B.save(out)


def refs():
    items = [
        ("Ferrari V16: near shadow bank, olive strands on black (4x)", crop(f"{LW}/V16.png", 470, 360, 100, 70, 4)),
        ("Ferrari tarn: near lit bank, dark strands on ochre (4x)", crop(NAT, 10, 320, 100, 70, 4)),
        ("Ferrari tarn: crest, isolated dark holes -> ticks (4x)", crop(NAT, 10, 252, 100, 70, 4)),
        ("Ferrari V16: lit bank crest, picket of lit ticks (4x)", crop(f"{LW}/V16.png", 0, 232, 100, 70, 4)),
        ("Ferrari V16: mid bank, 3-green checker domes (4x)", crop(f"{LW}/V16.png", 520, 225, 100, 70, 4)),
        ("Ferrari V16: islet, 2-colour checker, fringed crest (4x)", crop(f"{LW}/V16.png", 255, 255, 100, 70, 4)),
        ("Ferrari V04: far slopes, flat planes, fine texture (4x)", crop(V04, 300, 250, 100, 70, 4)),
        ("Ferrari V04: near blades, curved, 2px (4x)", crop(V04, 350, 350, 100, 70, 4)),
    ]
    board(items, 4, os.path.join(IMG, "stage0_ferrari_distance_ladder.png"),
          "Stage 0: how Ferrari paints grass at each distance (crops from his native pixels)")


if __name__ == "__main__":
    refs()


def perspective_bank(hour="golden", v=4.8):
    """Emulate a bank receding in perspective: horizontal bands whose ell grows downward
    (0.5 px at the top -> 9 px at the bottom), next to Ferrari's tarn bank, crest to near."""
    import vocab
    W, H, band = 100, 150, 10
    ells = np.geomspace(0.5, 9.0, H // band)
    rows = []
    for i, e in enumerate(ells):
        t = vocab.flat_patch(float(e), v, True, hour, W=W, H=band + 24)
        rows.append(t[12:12 + band])
    mine = Image.fromarray(np.concatenate(rows, 0)).resize((W * 4, H * 4), Image.NEAREST)
    ref = crop(NAT, 10, 252, W, H, 4)
    board([("mine: bands of growing ell (0.5 -> 9 px), golden", mine), ("Ferrari tarn bank: crest -> near", ref)], 2,
          os.path.join(IMG, "stage2_perspective_bank.png"), "Stage 2: the handoff inside one receding bank, 4x")
