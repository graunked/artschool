"""Palettes: the light lives here, the form lives in the index canvas.

Grass has one 7-step ramp per (hour, material):
  0 core black   1 deep shadow   2 shadow   3 reflected/olive
  4 halftone     5 lit           6 sun-struck tip / rim
The shadow family uses steps 0-3, the lit family 2-6 (lit strokes' dark gaps reach 0-1).
Two grass materials share the index layout: green (live) and straw (dead).
Colours sampled from Ferrari's Mirror Pond (day), the tarn at dusk (golden/dusk),
and the rain scene (overcast), then adjusted by eye.
"""
import numpy as np

HOURS = {
    # green ramp, straw ramp, sky (top, horizon), water (light, dark), soil (dark, lit)
    "day": dict(
        green=[(2, 8, 8), (10, 30, 20), (26, 50, 30), (50, 72, 38), (86, 100, 46), (132, 134, 64), (196, 180, 96)],
        straw=[(8, 10, 4), (30, 30, 14), (56, 54, 26), (92, 84, 40), (138, 122, 58), (190, 170, 84), (236, 214, 124)],
        sky=[(20, 110, 220), (120, 190, 240)], water=[(60, 150, 220), (14, 58, 96)],
        soil=[(30, 20, 14), (96, 70, 44)], sun=(-0.55, 0.45, 0.70)),
    "golden": dict(
        green=[(8, 18, 8), (12, 34, 12), (30, 40, 20), (62, 60, 24), (112, 90, 36), (196, 132, 70), (238, 178, 104)],
        straw=[(16, 14, 8), (40, 32, 16), (74, 58, 30), (116, 88, 42), (164, 118, 58), (214, 156, 86), (246, 200, 132)],
        sky=[(206, 168, 190), (242, 180, 150)], water=[(196, 150, 170), (70, 56, 80)],
        soil=[(34, 22, 18), (110, 70, 44)], sun=(-0.75, 0.35, 0.35)),
    "dusk": dict(
        green=[(10, 10, 16), (18, 22, 26), (32, 36, 36), (52, 54, 44), (86, 72, 52), (150, 100, 76), (206, 140, 104)],
        straw=[(14, 12, 16), (34, 28, 30), (58, 48, 44), (88, 70, 56), (126, 94, 70), (176, 124, 90), (220, 164, 120)],
        sky=[(92, 80, 140), (220, 140, 130)], water=[(150, 110, 150), (40, 34, 60)],
        soil=[(24, 18, 22), (80, 56, 50)], sun=(-0.85, 0.2, 0.18)),
    "overcast": dict(
        green=[(6, 14, 14), (14, 32, 26), (26, 50, 38), (42, 70, 50), (62, 92, 62), (88, 116, 76), (122, 144, 96)],
        straw=[(14, 16, 14), (34, 36, 28), (58, 58, 44), (84, 82, 60), (110, 104, 76), (138, 128, 94), (168, 158, 120)],
        sky=[(110, 110, 150), (160, 160, 180)], water=[(120, 124, 150), (40, 48, 66)],
        soil=[(26, 22, 22), (70, 60, 54)], sun=(-0.3, 0.3, 0.9)),
}

# index layout of the output canvas
SKY0 = 0          # 0..7  sky band
WATER0 = 8        # 8..11 water
SOIL0 = 12        # 12..13
GREEN0 = 16       # 16..22
STRAW0 = 24       # 24..30
NIDX = 32


def lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def build(hour="day", wet=0.5):
    """Return (palette[NIDX] of RGB, sun_dir).  `wet` greens the live ramp slightly."""
    H = HOURS[hour]
    pal = [(255, 0, 255)] * NIDX
    s0, s1 = H["sky"]
    for i in range(8):
        pal[SKY0 + i] = lerp(s0, s1, i / 7)
    w0, w1 = H["water"]
    for i in range(4):
        pal[WATER0 + i] = lerp(w1, w0, i / 3)
    pal[SOIL0], pal[SOIL0 + 1] = H["soil"]
    for i in range(7):
        pal[GREEN0 + i] = H["green"][i]
        pal[STRAW0 + i] = H["straw"][i]
    sun = np.array(H["sun"], float)
    return pal, sun / np.linalg.norm(sun)


def to_rgb(idx, pal):
    lut = np.array(pal, np.uint8)
    return lut[np.clip(idx, 0, NIDX - 1)]
