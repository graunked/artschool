"""Willow palettes: (material, step) -> RGB for a season, an hour and a depth.

Light as hue shift: every ramp walks from a cool dark (tinted by the sky's ambient colour) to a warm
light (tinted by the sun). Depth lifts and cools the darks and leaves the lights nearly alone
(measured on Ferrari by the master-copy study). Silver is its own ramp: the undersides of willow
leaves, grey-green to pale silver, cooler than the green.
"""
import numpy as np
from wpaint import SKY, GRAVEL, LEAF, SILVER, STEM, WATER, GRASS, EARTH, FAR

LEAFS = 10


def _lin(c):
    c = np.asarray(c, float) / 255
    return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)


def _srgb(l):
    l = np.clip(l, 0, 1)
    return np.where(l > 0.0031308, 1.055 * l ** (1 / 2.4) - 0.055, 12.92 * l) * 255


def to_lab(c):
    l = _lin(c)
    M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    lms = np.cbrt(M1 @ l)
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return M2 @ lms


def to_rgb(lab):
    M2i = np.array([[1, 0.3963377774, 0.2158037573], [1, -0.1055613458, -0.0638541728], [1, -0.0894841775, -1.2914855480]])
    lms = (M2i @ lab) ** 3
    M1i = np.array([[4.0767416621, -3.3077115913, 0.2309699292], [-1.2684380046, 2.6097574011, -0.3413193965],
                    [-0.0041960863, -0.7034186147, 1.7076147010]])
    return tuple(int(round(v)) for v in _srgb(M1i @ lms))


def mix(a, b, t):
    return to_rgb(to_lab(a) * (1 - t) + to_lab(b) * t)


def ramp(dark, light, n, sun, amb, warm=0.25, cool=0.3):
    """n steps from dark to light in OKLab; lights pulled toward the sun colour, darks toward ambient."""
    out = []
    for i in range(n):
        t = i / max(n - 1, 1)
        c = to_lab(dark) * (1 - t) + to_lab(light) * t
        c = c * (1 - warm * t ** 1.5) + to_lab(sun) * warm * t ** 1.5 if warm else c
        c = c * (1 - cool * (1 - t) ** 2) + to_lab(amb) * cool * (1 - t) ** 2 if cool else c
        out.append(to_rgb(c))
    return out


SEASONS = {
    #          lit base (dark, light)          lit leaves (mid, top)          silver (lo, hi)            twig (dark, light)       shadow (dark, mark)
    "summer": dict(base=((14, 36, 18), (30, 62, 30)), leaf=((92, 132, 56), (168, 200, 96)), silver=((120, 142, 128), (196, 214, 200)),
                   twig=((50, 36, 30), (150, 120, 84)), shadow=((6, 24, 14), (36, 66, 42))),
    "spring": dict(base=((20, 42, 16), (42, 74, 28)), leaf=((118, 158, 48), (196, 222, 98)), silver=((140, 160, 132), (208, 224, 196)),
                   twig=((70, 40, 32), (170, 110, 70)), shadow=((10, 26, 14), (56, 82, 50))),
    "silvery": dict(base=((16, 32, 26), (36, 58, 46)), leaf=((104, 128, 96), (170, 190, 150)), silver=((136, 156, 150), (216, 228, 222)),
                    twig=((54, 40, 34), (150, 126, 96)), shadow=((10, 22, 22), (54, 72, 68))),
    "autumn": dict(base=((36, 26, 12), (70, 52, 22)), leaf=((170, 128, 40), (236, 196, 72)), silver=((170, 156, 110), (232, 220, 170)),
                   twig=((56, 30, 22), (150, 84, 48)), shadow=((22, 16, 12), (78, 60, 34))),
    # winter: no leaves; the twig ramp carries the colour (the leaf ramps are kept for mixed scenes)
    "winter_red": dict(base=((20, 30, 22), (40, 52, 40)), leaf=((90, 104, 70), (150, 160, 110)), silver=((130, 130, 130), (200, 200, 200)),
                       twig=((48, 16, 18), (206, 76, 52)), shadow=((10, 16, 14), (40, 46, 42))),
    "winter_yellow": dict(base=((20, 30, 22), (40, 52, 40)), leaf=((90, 104, 70), (150, 160, 110)), silver=((130, 130, 130), (200, 200, 200)),
                          twig=((58, 44, 18), (226, 196, 96)), shadow=((10, 16, 14), (40, 46, 42))),
    "winter_grey": dict(base=((20, 30, 22), (40, 52, 40)), leaf=((90, 104, 70), (150, 160, 110)), silver=((130, 130, 130), (200, 200, 200)),
                        twig=((40, 32, 42), (156, 136, 146)), shadow=((10, 16, 14), (40, 46, 42))),
    # alpine: dwarf willow mats, cold bright greens, catkins silver-white
    "alpine": dict(base=((18, 34, 20), (36, 60, 34)), leaf=((96, 136, 62), (176, 204, 110)), silver=((170, 176, 160), (236, 238, 226)),
                   twig=((60, 40, 34), (150, 104, 80)), shadow=((8, 20, 14), (40, 60, 42))),
    # alder: darker, bluer green than willow, no silver sheen
    "alder": dict(base=((12, 32, 20), (26, 56, 34)), leaf=((70, 116, 54), (150, 186, 92)), silver=((96, 130, 82), (160, 190, 120)),
                  twig=((50, 36, 34), (140, 116, 100)), shadow=((6, 20, 14), (32, 56, 40))),
}

SEASONS["alder_autumn"] = dict(base=((20, 30, 16), (40, 50, 26)), leaf=((104, 110, 52), (170, 162, 84)),
                               silver=((110, 120, 80), (170, 170, 120)), twig=((50, 36, 34), (140, 116, 100)),
                               shadow=((10, 18, 12), (40, 48, 32)))   # alders drop their leaves green-brown
SEASONS["winter_dogwood"] = dict(SEASONS["winter_red"], twig=((56, 10, 20), (214, 44, 44)))
SEASONS["dogwood_autumn"] = dict(base=((30, 14, 22), (56, 26, 36)), leaf=((140, 50, 70), (204, 96, 104)),
                                 silver=((200, 200, 196), (244, 244, 238)), twig=((56, 10, 20), (190, 50, 50)),
                                 shadow=((16, 8, 14), (52, 26, 36)))       # purple-red leaves, white berries
SEASONS["dogwood"] = dict(base=((10, 26, 16), (22, 48, 28)), leaf=((62, 104, 50), (132, 170, 84)),
                          silver=((210, 210, 204), (246, 246, 240)), twig=((70, 12, 22), (200, 50, 50)),
                          shadow=((6, 16, 10), (28, 46, 30)))          # dark oval leaves, red twigs showing
SEASONS["salmonberry"] = dict(base=((22, 44, 16), (44, 76, 26)), leaf=((120, 160, 50), (200, 224, 104)),
                              silver=((140, 160, 110), (200, 214, 160)), twig=((70, 44, 24), (176, 116, 70)),
                              shadow=((10, 24, 10), (48, 74, 34)))     # big pale yellow-green leaves
SEASONS["salmonberry_winter"] = dict(SEASONS["winter_red"], twig=((52, 30, 20), (190, 124, 76)))
SEASONS["cottonwood"] = dict(base=((10, 28, 18), (24, 50, 32)), leaf=((64, 110, 56), (148, 180, 96)),
                             silver=((150, 170, 150), (222, 232, 220)), twig=((50, 46, 44), (150, 144, 136)),
                             shadow=((6, 18, 12), (30, 50, 36)))
SEASONS["cottonwood_autumn"] = dict(base=((48, 40, 14), (84, 70, 22)), leaf=((196, 164, 44), (246, 222, 96)),
                                    silver=((200, 196, 150), (240, 236, 196)), twig=((50, 46, 44), (150, 144, 136)),
                                    shadow=((24, 20, 10), (70, 60, 28)))
BARKRAMP = ((48, 46, 50), (226, 224, 214))     # red alder trunk: grey to lichen white

HOURS = {
    #          sun colour      ambient (sky)    sky top / horizon          gravel lit
    "morning": dict(key=0.95, sun=(255, 236, 200), amb=(120, 150, 200), sky=((150, 190, 235), (225, 232, 238))),
    "noon": dict(key=1.0, sun=(255, 250, 235), amb=(130, 165, 215), sky=((120, 170, 230), (210, 228, 242))),
    "golden": dict(key=0.88, sun=(255, 196, 120), amb=(110, 110, 170), sky=((120, 150, 210), (250, 214, 170))),
    "dusk": dict(key=0.62, sun=(250, 150, 140), amb=(80, 70, 130), sky=((70, 70, 140), (236, 150, 140))),
    "overcast": dict(key=0.9, sun=(225, 228, 228), amb=(150, 160, 170), sky=((180, 186, 192), (214, 216, 218))),
}


def willow_palette(season="summer", hour="noon", depth=0.0, haze=None):
    S = SEASONS[season]; Hh = HOURS[hour]
    sun, amb = Hh["sun"], Hh["amb"]
    hz = haze or mix(Hh["sky"][1], amb, 0.3)
    warm = 0.12 if hour in ("noon", "overcast") else 0.3
    P = {}
    base = ramp(S["base"][0], S["base"][1], 2, sun, amb, warm=0, cool=0.12)
    leaf = ramp(S["leaf"][0], S["leaf"][1], 3, sun, amb, warm=warm, cool=0.0)
    lit = [base[0], base[1], mix(base[1], leaf[0], 0.55), leaf[1], leaf[2]]
    sil = ramp(mix(S["silver"][0], S["base"][1], 0.5), S["silver"][1], 5, sun, amb, warm=warm * 0.6, cool=0.2)
    shd = ramp(S["shadow"][0], S["shadow"][1], 4, sun, amb, warm=0, cool=0.15)
    twig = ramp(S["twig"][0], S["twig"][1], 4, sun, amb, warm=warm, cool=0.3)
    # depth: darks lift toward haze strongly, lights barely (Ferrari's depth palettes)
    key = Hh.get("key", 1.0)
    def far(c, k):
        lab = to_lab(c); lab[0] *= key
        return mix(to_rgb(lab), hz, np.clip(depth * k, 0, 0.95))
    for i, c in enumerate(lit):
        P[(LEAF, i)] = far(c, 0.9 - 0.12 * i)
    for i, c in enumerate(sil):
        P[(SILVER, i)] = far(c, 0.85 - 0.12 * i)
    for i, c in enumerate(shd):
        P[(LEAFS, i)] = far(c, 1.0 - 0.1 * i)
    for i, c in enumerate(twig):
        P[(STEM, i)] = far(c, 0.9 - 0.1 * i)
    sk = [mix(Hh["sky"][0], Hh["sky"][1], t) for t in (0.0, 0.35, 0.7, 1.0)]
    for i, c in enumerate(sk):
        P[(SKY, i)] = c
    for i, c in enumerate(ramp(BARKRAMP[0], BARKRAMP[1], 6, sun, amb, warm=warm, cool=0.3)):
        P[(13, i)] = far(c, 0.8 - 0.1 * i)
    for i, c in enumerate(ramp((34, 32, 32), (176, 170, 158), 6, sun, amb, warm=warm, cool=0.3)):
        P[(14, i)] = far(c, 0.8 - 0.1 * i)          # cottonwood bark: dark furrowed grey
    for i, c in enumerate(ramp((120, 20, 90), (236, 110, 190), 2, sun, amb, warm=0.1, cool=0.1)):
        P[(16, i)] = far(c, 0.5)                   # salmonberry flower magenta
    for i, c in enumerate(ramp((180, 60, 20), (250, 150, 50), 2, sun, amb, warm=0.1, cool=0.1)):
        P[(17, i)] = far(c, 0.5)                   # salmonberry berry orange
    gr = ramp((52, 50, 50), (214, 206, 190), 6, sun, amb, warm=warm, cool=0.3)
    for i, c in enumerate(gr):
        P[(GRAVEL, i)] = far(c, 0)
    return P
