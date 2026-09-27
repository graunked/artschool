"""A hand copy of Genge's Fig 2.9b (Rough Tor) at 0.1 m/px, done the way his book says to draw:
a quadrant grid laid over the plate, the outline and the sheeting lines blocked in as straight
segments read off the grid, then his shading labels applied as value steps.

Every coordinate below was read by eye from the plate with a 10 px grid over it (grid.png).
No geometry, no renderer, no light model: the plate decides where the dark goes.
"""
import sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, '~/work/pixelart-studies/books/genge/src')
import palette  # read-only: the study's Lab-anchored granite ramp

W, H = 120, 76

# 1. outline (blocking-in: straight segments, clockwise from lower left)
OUTLINE = [(10, 75), (10, 56), (9, 50), (9, 46), (12, 44), (16, 43), (17, 39), (21, 36), (25, 34),
           (27, 30), (25, 27), (30, 24), (38, 23), (37, 19), (44, 20), (49, 22), (54, 21), (57, 20),
           (65, 21), (68, 23), (72, 24), (77, 28), (76, 31), (80, 34), (83, 39), (82, 42), (86, 44),
           (90, 46), (91, 50), (92, 45), (99, 44), (107, 46), (109, 49), (107, 53), (111, 56),
           (112, 58), (110, 62), (112, 66), (110, 70), (110, 75)]

# 2. the sheeting joints: the dark line under each slab (the plate's heaviest marks)
SHEETS = [
    [(26, 27), (38, 28), (50, 27)],
    [(27, 31), (45, 30), (60, 31), (77, 31)],
    [(21, 36), (35, 36), (50, 35), (66, 36), (80, 35)],
    [(17, 40), (30, 40), (40, 39), (58, 41), (82, 42)],
    [(12, 45), (25, 44), (40, 45), (60, 45), (80, 46), (90, 47)],
    [(10, 51), (28, 50), (45, 50), (66, 51), (90, 51)],
    [(10, 56), (25, 56), (40, 55)],
    [(40, 59), (60, 58), (72, 60)],
    [(10, 61), (22, 61), (40, 63)],
    [(66, 64), (80, 63), (90, 64)],
    [(10, 67), (30, 67), (50, 68)],
    [(56, 70), (75, 69), (90, 71)],
    # right stack
    [(92, 50), (100, 50), (109, 50)],
    [(92, 54), (100, 55), (108, 54)],
    [(91, 58), (101, 58), (111, 58)],
    [(91, 62), (100, 62), (110, 62)],
    [(91, 66), (101, 67), (111, 66)],
    [(91, 70), (100, 70), (110, 70)],
]
# 3. vertical joints (fewer, thinner; 'deep fractures are dark' only where they are open)
JOINTS = [((49, 22), (47, 36), True), ((40, 39), (38, 75), True), ((66, 45), (62, 75), False),
          ((82, 44), (80, 71), False), ((90, 47), (91, 75), True), ((20, 56), (21, 75), False),
          ((30, 60), (29, 75), False), ((58, 21), (57, 30), False)]


def rasterize():
    mask = Image.new('L', (W, H), 0)
    ImageDraw.Draw(mask).polygon(OUTLINE, fill=1)
    rock = np.array(mask, bool)
    sheet = np.zeros((H, W), bool)
    img = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(img)
    for line in SHEETS:
        d.line(line, fill=1, width=1)
    sheet = np.array(img, bool) & rock
    img = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(img)
    for a, b, open_ in JOINTS:
        d.line([a, b], fill=2 if open_ else 1, width=1)
    joint = np.array(img) * rock
    return rock, sheet, joint


def paint():
    rock, sheet, joint = rasterize()
    step = np.where(rock, 5, -1)
    # slab bodies: lit tops, rounding darker toward each slab's lower edge (a lens in section)
    below = np.zeros((H, W), int) + 99   # px from the sheet line below
    for y in range(H - 2, -1, -1):
        below[y] = np.where(sheet[y + 1], 1, np.minimum(below[y + 1] + 1, 99))
    above = np.zeros((H, W), int) + 99   # px below the sheet line above
    for y in range(1, H):
        above[y] = np.where(sheet[y - 1], 1, np.minimum(above[y - 1] + 1, 99))
    step = np.where(rock & (above == 1), 6, step)              # top of each slab catches the sky
    step = np.where(rock & (above == 2), 6, step)
    step = np.where(rock & (below == 1), 4, step)              # the slab turns under
    # 'shadow indicates top of overhang': the line itself and the pixel under it are darkest
    step = np.where(sheet, 0, step)
    step = np.where(rock & (above == 1) & np.roll(sheet, 1, 0), 1, step)
    # 'inset areas shadowed': the left flank sits back in the plate; the whole lower-left block
    # and the slot between the two stacks are the shadow family
    yy, xx = np.mgrid[0:H, 0:W]
    inset = rock & (((xx < 17) & (yy > 50)) | ((xx >= 86) & (xx <= 92) & (yy > 46)))
    step = np.where(inset & (step > 3), step - 3, step)
    # 'darkest shadow in most inset areas': where a sheet line meets an open joint
    step = np.where(joint == 2, 0, step)
    step = np.where((joint == 1) & (step > 3), 3, step)
    # slab ends round off: each sheet line ends in a notch in the silhouette (the plate's lumpy
    # outline is made of slab ends), and the line is heavy near its ends, lighter across the
    # middle of a big slab (Genge's impersistent line, p. 28)
    rng = np.random.default_rng(7)
    for line in SHEETS:
        (x0, y0), (x1, y1) = line[0], line[-1]
        for (x, y), sgn in (((x0, y0), 1), ((x1, y1), -1)):
            if all(0 <= x - sgn * k < W and rock[y, x - sgn * k] for k in (1, 2, 3)):
                continue   # this line ends inside the rock (at a joint), not at the outline
            for dx, dy in ((0, 0), (sgn, 0), (0, -1), (0, 1), (2 * sgn, 0)):
                xx_, yy_ = x + dx, y + dy
                if 0 <= xx_ < W and 0 <= yy_ < H and (dx, dy) in ((0, 0), (sgn, 0)) or (0 <= xx_ < W and 0 <= yy_ < H and abs(dx) + abs(dy) == 1 and rng.random() < 0.6):
                    step[yy_, xx_] = -1
            # heavy end: second dark pixel under the line for ~5 px
            for k in range(1, 6):
                xx_ = x + sgn * k
                yy_ = int(round(y + (y1 - y0) * (k / max(abs(x1 - x0), 1)) * sgn)) + 1
                if 0 <= xx_ < W and 0 < yy_ < H and step[yy_, xx_] >= 0:
                    step[yy_, xx_] = 0
        n = abs(x1 - x0)
        if n > 20:
            xs = np.arange(min(x0, x1) + 7, max(x0, x1) - 7)
            gaps = xs[(np.floor((xs + rng.integers(0, 6)) / 5) % 3) == 0]
            for x in gaps:
                col = np.flatnonzero(sheet[:, x])
                for y in col:
                    if step[y, x] == 0:
                        step[y, x] = 3
    rock = rock & (step >= 0)
    # grass at the foot (Genge: 'quickly drawn irregular zigzag lines')
    g = np.zeros((H, W), bool)
    rng = np.random.default_rng(3)
    for x in range(W):
        h = 2 + int(rng.integers(0, 3)) + (2 if x % 7 in (0, 1) else 0)
        g[H - h:, x] = True
    return step, rock & (step >= 0), g


if __name__ == '__main__':
    step, rock, grass = paint()
    ramp = palette.ramp('tor_granite')
    gr = palette.ramp('grass_dry')
    sky = palette.sky_rgb(H, W)
    out = sky.copy()
    out[rock] = ramp[np.clip(step[rock], 0, 7)]
    gs = np.where(np.roll(grass, -1, 0) | ~grass, 3, 5)
    out[grass] = gr[gs[grass]]
    Image.fromarray(out).save('copy_tor_1x.png')
    Image.fromarray(out).resize((W * 4, H * 4), Image.NEAREST).save('copy_tor_4x.png')
