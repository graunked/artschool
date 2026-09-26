"""palette.py -- the light, as palettes.  The index canvas (material, step) never changes with hour
or wetness; only these ramps do (Ferrari: the same index map serves morning and afternoon).

Each ramp has 8 steps in the value plan proved in greyscale:
    0 contact | 1 deep shadow | 2 core | 3 reflected rim || 4 halftone | 5 lit edge | 6 light | 7 face/glint
Rules (from the photos and Ferrari's trunks):
  * silvered wood is nearly neutral; the HUE comes from the light: lit steps take the sun's warmth,
    the shadow steps the sky's coolness -- a hue walk along one ramp, not added white/black;
  * the reflected rim (3) is the coolest step (sky and pale gravel bounce), like Ferrari's
    sky-blue trunk rim (V09 class 132), and sits between core and halftone in value;
  * wet wood is darker by about a third and turns warm brown (tannin); its light steps compress,
    but a specular glint (7) stays near white -- wet things are dark with sharp highlights;
  * gravel is a warm beige-grey held a step below the lit wood; its shadow steps go blue-violet;
  * the root-wad mass is a dark umber-grey (grit and old bark), the bark ramp a warm dark brown.
"""
import numpy as np

WOOD, GRAVEL, RAW, BARK, SAND, ROOTM, WETW = 0, 1, 2, 3, 4, 5, 6

# light colour and sky colour per hour (multipliers), and the black point / haze
HOURS = {   # (critic r9: my first shadow shift was ~3x too strong; these are a third of it)
    'noon':     dict(sun=(1.00, 0.99, 0.96), sky=(0.98, 0.99, 1.01), lift=0.0),
    'morning':  dict(sun=(1.02, 0.98, 0.92), sky=(0.97, 0.98, 1.02), lift=0.02),
    'golden':   dict(sun=(1.10, 0.93, 0.74), sky=(0.93, 0.91, 1.00), lift=0.02),
    'overcast': dict(sun=(0.98, 0.99, 1.00), sky=(0.98, 0.99, 1.00), lift=0.03),
}

# base (neutral) ramps, value-first.  Restraint: the core is a WARM grey (Ferrari's trunk core
# 131,123,123), only the reflected rim leans cool, and not far.
BASE = {
    WOOD:   [(38, 36, 40), (74, 72, 76), (122, 118, 116), (132, 133, 136), (154, 152, 148),
             (178, 175, 168), (204, 201, 193), (228, 226, 217)],
    RAW:    [(38, 32, 32), (74, 66, 62), (122, 112, 102), (134, 130, 130), (168, 156, 140),
             (194, 182, 162), (220, 208, 184), (242, 234, 214)],
    GRAVEL: [(30, 28, 32), (58, 56, 64), (84, 80, 84), (106, 100, 94), (126, 118, 106),
             (144, 136, 120), (162, 154, 136), (180, 172, 152)],
    BARK:   [(20, 14, 12), (36, 26, 22), (54, 40, 32), (66, 52, 46), (84, 64, 48),
             (104, 80, 60), (124, 98, 74), (146, 118, 90)],
    7:      [(30, 30, 34), (56, 57, 64), (82, 83, 88), (102, 102, 104), (120, 118, 116),
             (138, 136, 132), (156, 154, 148), (174, 172, 164)],          # grey stones (quiet)
    8:      [(32, 27, 26), (60, 52, 50), (88, 76, 68), (108, 94, 84), (128, 110, 94),
             (146, 126, 106), (164, 144, 120), (182, 162, 136)],          # rusty stones (quiet)
    9:      [(30, 28, 34), (56, 54, 62), (86, 84, 88), (96, 100, 108), (114, 114, 114),
             (134, 134, 130), (156, 154, 148), (196, 194, 186)],          # less-bleached grey wood
    ROOTM:  [(18, 15, 16), (34, 29, 30), (52, 45, 42), (66, 60, 60), (84, 74, 66),
             (104, 92, 80), (126, 112, 96), (150, 136, 116)],
}
# which steps are 'lit' (take the sun colour) vs 'shadow' (take the sky colour)
LIT = np.array([0, 0, 0, 0, 1, 1, 1, 1], float)
# wet wood: darker, warmer, glint kept
WET_WOOD = [(28, 22, 20), (48, 38, 34), (74, 60, 50), (82, 74, 72), (98, 80, 64),
            (116, 96, 76), (136, 114, 90), (226, 222, 212)]


def ramps(hour='noon', wet=0.0):
    h = HOURS[hour]
    sun = np.array(h['sun']); sky = np.array(h['sky'])
    out = {}
    for m, ramp in BASE.items():
        r = np.array(ramp, float)
        if m in (WOOD, RAW) and wet > 0:
            r = r * (1 - wet) + np.array(WET_WOOD, float) * wet
        mul = LIT[:, None] * sun + (1 - LIT[:, None]) * sky
        if hour == 'overcast':
            # no sun: compress the lit steps toward the halftone, everything takes the sky
            r[4:7] = r[4:7] * 0.92 + r[4] * 0.08
            mul = np.tile(sky, (8, 1))
        r = r * mul
        # the reflected-rim step is the coolest (sky + bounce)
        r = r + h['lift'] * 255 * (1 - r / 255)
        out[m] = [tuple(int(v) for v in np.clip(c, 0, 255)) for c in r]
    # the wet-wood material: the wood ramp fully wet
    r = np.array(WET_WOOD, float) * (LIT[:, None] * sun + (1 - LIT[:, None]) * sky)
    if hour == 'overcast':
        r = np.array(WET_WOOD, float) * sky
    r = r + h['lift'] * 255 * (1 - r / 255)
    out[WETW] = [tuple(int(v) for v in np.clip(c, 0, 255)) for c in r]
    return out
