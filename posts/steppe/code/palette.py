"""palette: the light. (material, step, depth band) -> sRGB, for a season and an hour.

A ramp is built physically-ish in linear RGB and then pushed toward Ferrari's habits:
  colour(step) = albedo * (ambient(step) * sky + sunlit(step) * sun)
- steps below the terminator get only sky light, scaled by occlusion (cool, violet at dusk);
- steps above it add the sun (warm at golden hour);
- depth bands lift the black point and pull everything toward the haze colour; the lit colour
  moves least (measured by the master-copy student: lit stays, shadow changes with depth).

Seasons change albedos (the steppe greens in spring, cures to straw in July, goes grey-tan and
snowy in winter); hours change sun/sky colours and strength.
"""
import numpy as np
import sp
from sp import (SKY, GROUND, CRUST, STONE, BASALT, SAGE, SAGESTEM, GRASS, BITTER, NEEDLE, BARK,
                FAR, SNOW, FLOWER, HAZE, WATER, FORB)

# ---- step light plans (ambient occlusion, sun fraction) per ramp length
PLAN = {
    6: ([0.28, 0.55, 0.85, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.40, 0.66, 0.92]),
    5: ([0.32, 0.62, 0.95, 1.0, 1.0], [0.0, 0.0, 0.0, 0.58, 0.9]),
    4: ([0.3, 0.7, 1.0, 1.0], [0.0, 0.0, 0.55, 1.0]),
    3: ([0.35, 1.0, 1.0], [0.0, 0.5, 1.0]),
    2: ([0.5, 1.0], [0.0, 1.0]),
    1: ([1.0], [0.8]),
}

# ---- hours: sun colour (linear-ish multiplier), sky/ambient colour, haze colour, sky gradient
HOURS = {
    "morning": dict(sun=(1.00, 0.92, 0.80), sky=(0.42, 0.48, 0.64), haze=(196, 206, 222),
                    zenith=(92, 142, 205), horizon=(214, 222, 228), k=1.0),
    "noon":    dict(sun=(1.00, 0.97, 0.90), sky=(0.46, 0.52, 0.66), haze=(190, 205, 225),
                    zenith=(70, 130, 210), horizon=(190, 212, 232), k=1.08),
    "golden":  dict(sun=(1.05, 0.78, 0.50), sky=(0.34, 0.36, 0.54), haze=(214, 186, 170),
                    zenith=(96, 120, 180), horizon=(240, 196, 140), k=1.0),
    "dusk":    dict(sun=(0.80, 0.42, 0.34), sky=(0.28, 0.24, 0.44), haze=(150, 118, 150),
                    zenith=(58, 58, 118), horizon=(236, 140, 118), k=0.85),
    "overcast": dict(sun=(0.30, 0.30, 0.30), sky=(0.66, 0.68, 0.72), haze=(186, 188, 192),
                     zenith=(160, 164, 172), horizon=(206, 208, 210), k=0.95),
}

# ---- albedos by season: lit albedo (sRGB, as if under white light) per material
BASE = {
    GROUND:   (192, 162, 124),   # loess / sandy loam, warm tan (a shade below the straw)
    CRUST:    (150, 128, 102),    # biological soil crust: dark lumpy skin
    STONE:    (150, 140, 128),
    BASALT:   (132, 100, 80),    # weathered basalt: brown-black, rust-orange in the light
    SAGE:     (160, 166, 142),   # grey-green, silver
    SAGESTEM: (122, 108, 92),
    GRASS:    (222, 202, 146),   # cured bunchgrass (summer): pale straw, less orange than the loess
    BITTER:   (112, 122, 84),    # bitterbrush: darker, greener, still dusty
    NEEDLE:   (84, 118, 64),     # ponderosa needles
    BARK:     (196, 112, 62),    # cinnamon-orange plates
    FLOWER:   (190, 168, 120),   # sage bloom (tan)
    SNOW:     (240, 242, 246),
    FORB:     (232, 196, 40),    # balsamroot yellow
    FAR:      (170, 164, 140),
}

SEASONS = {
    "spring": {GRASS: (132, 160, 96), SAGE: (150, 166, 130), GROUND: (184, 160, 124),
               BITTER: (108, 132, 78), FLOWER: (236, 200, 50), NEEDLE: (92, 128, 66),
               FAR: (140, 158, 112)},
    "summer": {},
    "autumn": {GRASS: (200, 182, 138), FLOWER: (214, 188, 120), SAGE: (160, 162, 140),
               BITTER: (132, 132, 90), FAR: (176, 160, 130)},
    "winter": {GRASS: (170, 150, 118), SAGE: (140, 148, 136), GROUND: (160, 140, 118),
               BITTER: (110, 106, 88), NEEDLE: (70, 98, 60), FAR: (150, 146, 136)},
}

NSTEPS = {SKY: 6, GROUND: 5, CRUST: 3, STONE: 4, BASALT: 5, SAGE: 6, SAGESTEM: 4, GRASS: 5,
          BITTER: 6, NEEDLE: 6, BARK: 5, FAR: 4, SNOW: 3, FLOWER: 2, HAZE: 3, WATER: 4, FORB: 2}

# per-material hue shifts of the shadow end (OKLab a, b offsets): Ferrari's cool shadows
SHADOW_SHIFT = {SAGE: (0.006, 0.014), GROUND: (0.006, -0.018), GRASS: (0.006, 0.012),
                NEEDLE: (-0.010, -0.020), BARK: (0.020, -0.040), BASALT: (0.012, -0.036),
                SAGESTEM: (0.010, -0.030), BITTER: (-0.008, -0.022), STONE: (0.0, -0.03)}


# per-material shift of the LIT end (OKLab dL, da, db) applied to the top steps: lit needles
# go yellow-green, lit sage toward straw-silver (the warm sun rim on grey foliage)
LIGHT_SHIFT = {BASALT: (0.02, 0.02, 0.03), NEEDLE: (0.05, -0.012, 0.06), SAGE: (0.02, -0.004, 0.018), BITTER: (0.03, -0.01, 0.04),
               GRASS: (0.015, 0.0, 0.0)}


class Palette:
    def __init__(self, season="summer", hour="noon", bands=4, grey=False):
        self.season, self.hour, self.grey = season, hour, grey
        self.h = HOURS[hour]
        self.alb = dict(BASE)
        self.alb.update(SEASONS[season])
        self.bands = bands
        self.cache = {}

    def ramp(self, m, band=0):
        key = (m, band)
        if key in self.cache:
            return self.cache[key]
        if m == SKY:
            r = self._sky(NSTEPS[SKY])
        else:
            n = NSTEPS.get(m, 4)
            occ, sunf = PLAN[n]
            alb = sp.srgb_to_lin(self.alb.get(m, (128, 128, 128)))
            sun = np.array(self.h["sun"]) * self.h["k"]
            sky = np.array(self.h["sky"])
            cols = []
            for i in range(n):
                lin = alb * (occ[i] * sky + sunf[i] * sun)
                if m == SNOW:
                    lin = alb * (occ[i] * sky * 1.25 + sunf[i] * sun)
                cols.append(sp.lin_to_srgb(lin))
            cols = np.array(cols)
            lab = sp.rgb2lab(cols)
            sh = SHADOW_SHIFT.get(m, (0.0, -0.02))
            for i in range(n):
                t = 1 - i / max(1, n - 1)
                lab[i, 1] += sh[0] * t
                lab[i, 2] += sh[1] * t
            ls = LIGHT_SHIFT.get(m)
            if ls:
                sun_warm = max(0.0, (self.h["sun"][0] - self.h["sun"][2]))   # warm hours push more
                for i in range(n):
                    t = max(0.0, (i - (n - 1) * 0.5) / ((n - 1) * 0.5))
                    lab[i, 0] += ls[0] * t
                    lab[i, 1] += ls[1] * t
                    lab[i, 2] += ls[2] * t * (1 + sun_warm)
            # value contrast: Ferrari keys darks deep
            lab[:, 0] = np.clip(lab[:, 0] ** 1.06, 0, 1)
            r = sp.lab2rgb(lab)
        if band > 0:
            haze = np.array(self.h["haze"], float)
            t = min(0.85, band / (self.bands + 0.5))
            lab = sp.rgb2lab(r)
            hl = sp.rgb2lab(haze)
            n = len(r)
            for i in range(n):
                # darks move a lot, lights little
                wgt = t * (1.0 - 0.55 * i / max(1, n - 1))
                lab[i] = lab[i] * (1 - wgt) + hl * wgt
            r = sp.lab2rgb(lab)
        if self.grey:
            lab = sp.rgb2lab(r)
            lab[:, 1:] = 0
            r = sp.lab2rgb(lab)
        r = np.clip(np.round(r), 0, 255).astype(np.uint8)
        self.cache[key] = r
        return r

    def _sky(self, n):
        z, hz = np.array(self.h["zenith"], float), np.array(self.h["horizon"], float)
        return np.array([sp.mix(z, hz, i / (n - 1)) for i in range(n)])

    def rgb(self, m, s, b=0):
        if m == 0:
            return (255, 0, 255)
        r = self.ramp(m, b)
        return tuple(int(v) for v in r[int(np.clip(s, 0, len(r) - 1))])

    def swatch(self, mats=None, k=12):
        mats = mats or [SKY, GROUND, CRUST, BASALT, SAGE, SAGESTEM, GRASS, BITTER, NEEDLE, BARK,
                        SNOW, FLOWER, FORB]
        rows = []
        for m in mats:
            r = self.ramp(m)
            row = np.zeros((k, 6 * k, 3), np.uint8)
            for i, c in enumerate(r):
                row[:, i * k:(i + 1) * k] = c
            rows.append(row)
        return np.vstack(rows)
