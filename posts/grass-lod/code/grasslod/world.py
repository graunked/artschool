"""The world the painter looks at: a procedural, multi-scale grassy landscape.

Units are metres.  x runs to screen right, y away from the viewer, z up.
Every field is a pure function of world position, evaluated band-limited at the
spacing the current zoom needs (min_wl), so the same world can be looked at from
30 px/m down to 0.3 px/m without anything changing identity.

The scene: a small stream winding along a valley floor, a grassy cut bank on its far
side with hummocks along the lip, a meadow slope above, and rolling grassland hills
beyond.  Wetness collects in low ground; grazing shortens the sward; tussock
colonies form where it is wet and lightly grazed.
"""
import numpy as np
from dataclasses import dataclass, field
from .noise import fbm, value_noise


# grass species: height of the sward (m), tuft spacing (m), tuft width (m),
# tussock fraction, stiffness (how straight blades stand), seedhead fraction
SPECIES = {
    #            h     spacing width tussock stiff  seedheads
    "turf":    (0.08, 0.07, 0.06, 0.00, 0.8, 0.00),
    "meadow":  (0.35, 0.10, 0.10, 0.05, 0.6, 0.10),
    "hay":     (0.60, 0.10, 0.10, 0.02, 0.5, 0.25),
    "tussock": (0.45, 0.16, 0.22, 0.45, 0.4, 0.05),
    "sedge":   (0.50, 0.12, 0.16, 0.30, 0.7, 0.02),
}


@dataclass
class World:
    seed: int = 1
    species: str = "meadow"
    moisture: float = 0.5      # 0 dry .. 1 wet  (shifts hue green<->straw, sedge in hollows)
    grazing: float = 0.2       # 0 ungrazed .. 1 close-cropped
    wind: float = 0.3          # 0 calm .. 1 strong (lean + combing)
    wind_dir: float = 1.0      # +1 blows to screen right, -1 to the left
    season_dead: float = 0.05  # base fraction of straw (dead) blades
    relief: float = 1.0        # scales regional hills
    hummock: float = 0.3       # amplitude (m) of the 2-6 m hummocks
    water_level: float = 0.0

    # ----- geometry --------------------------------------------------------------
    def channel(self, x):
        """Stream centre line y_c(x) and half width."""
        s = self.seed
        yc = (5.0 * np.sin(x / 37.0 + s) + 2.0 * np.sin(x / 11.0 + 2 * s)
              + 60.0 * np.sin(x / 420.0 + 0.3 * s))
        return yc, 1.6

    def height(self, x, y, min_wl=0.0):
        s = self.seed
        yc, hw = self.channel(x)
        d = y - yc                       # signed distance-ish to the stream (m); >0 = far side
        ad = np.abs(d)
        # regional hills, rising away from the valley floor
        reg = 38.0 * self.relief * fbm(x, y, 900.0, 4, 0.5, seed=s + 3, min_wl=min_wl)
        valley = 0.035 * self.relief * np.minimum(ad, 900.0) + 12.0 * self.relief * (1 - np.exp(-ad / 150.0))
        z = 1.2 + valley + reg * np.clip(ad / 120.0, 0, 1)
        # spurs and gullies (100-300 m), ridged so hill flanks read as planes from afar
        rid = 1.0 - np.abs(fbm(x, y, 260.0, 3, 0.5, seed=s + 5, min_wl=min_wl))
        z = z + 8.0 * self.relief * (rid - 0.5) * np.clip(ad / 60.0, 0, 1)
        # meadow undulation (swells of a few tens of metres)
        z = z + 1.6 * fbm(x, y, 40.0, 3, 0.5, seed=s + 7, min_wl=min_wl)
        # hummocks, 2-6 m
        hum = fbm(x, y, 6.0, 3, 0.55, seed=s + 11, min_wl=min_wl)
        z = z + self.hummock * hum
        # tussock micro-relief (0.4-1 m), only where tussocks grow
        tus = self.tussockness(x, y)
        mic = fbm(x, y, 1.2, 2, 0.5, seed=s + 13, min_wl=min_wl)
        z = z + 0.12 * tus * np.maximum(mic, -0.2)
        # the cut bank: a terrace step at the channel edge, ragged along its length
        rag = 0.5 * fbm(x, y, 3.0, 3, 0.5, seed=s + 17, min_wl=min_wl)
        bank_w = 1.6 + 0.8 * value_noise(x / 9.0, 0.5, s + 19)
        t = np.clip((ad - hw - rag * 0.6) / bank_w, 0, 1)
        step = t * t * (3 - 2 * t)
        bed = self.water_level - 0.35
        z = bed + (z - bed) * step
        return z

    def normal(self, x, y, min_wl=0.0, eps=None):
        if eps is None:
            eps = max(0.02, min_wl * 0.5)
        z0 = self.height(x, y, min_wl)
        zx = (self.height(x + eps, y, min_wl) - z0) / eps
        zy = (self.height(x, y + eps, min_wl) - z0) / eps
        n = np.stack([-zx, -zy, np.ones_like(zx)])
        return n / np.linalg.norm(n, axis=0)

    # ----- ecology ---------------------------------------------------------------
    def wetness(self, x, y):
        """0..1: wet in low ground near the stream and in patchy flushes."""
        yc, hw = self.channel(x)
        ad = np.abs(y - yc)
        near = np.exp(-np.maximum(ad - hw, 0) / 12.0)
        patch = 0.5 + 0.5 * fbm(x, y, 30.0, 3, 0.5, seed=self.seed + 31)
        return np.clip((0.15 + 0.35 * self.moisture) * near + 0.5 * (patch - 0.5) + 1.1 * (self.moisture - 0.5) + 0.45, 0, 1)

    def tussockness(self, x, y):
        sp = SPECIES[self.species]
        colony = 0.5 + 0.5 * fbm(x, y, 12.0, 3, 0.5, seed=self.seed + 37)
        yc, hw = self.channel(x)
        wet = np.exp(-np.maximum(np.abs(y - yc) - hw, 0) / 15.0)
        v = sp[3] + 0.5 * wet * (1 - self.grazing) * (colony - 0.35)
        return np.clip(v * (1.0 - 0.6 * self.grazing), 0, 1)

    def dead(self, x, y):
        """Fraction of straw blades: dry ridges, grazing, season."""
        w = self.wetness(x, y)
        patch = fbm(x, y, 18.0, 3, 0.5, seed=self.seed + 41)
        return np.clip(self.season_dead + 1.0 * (0.5 - w) + 0.35 * patch + 0.2 * self.grazing, 0, 1)

    def clump(self, x, y):
        """0..1 clumpiness at 0.3-2 m: grass grows in clumps with low gaps between; the
        taller crowns catch the light.  At mid distance the clump is the mark."""
        c = fbm(x, y, 1.6, 3, 0.55, seed=self.seed + 47)
        return np.clip(0.5 + 0.9 * c, 0, 1)

    def sheen(self, x, y):
        """Wind combing: long bands across the wind, lit where blades bend toward you."""
        return fbm(x * 0.35, y * 1.4, 8.0, 3, 0.5, seed=self.seed + 53)

    def material(self, x, y):
        """-1..1 ground masses at 10-80 m, independent of light: drier knolls and
        trampled/grazed patches read lighter, wet flushes and lush growth darker."""
        big = fbm(x, y, 45.0, 3, 0.5, seed=self.seed + 71)
        dry = 0.5 - self.wetness(x, y)
        return np.clip(0.8 * big + 0.8 * dry, -1, 1)

    def sward_height(self, x, y):
        """Local blade height (m)."""
        sp = SPECIES[self.species]
        h = sp[0] * (1.0 - 0.75 * self.grazing)
        patch = fbm(x, y, 5.0, 3, 0.5, seed=self.seed + 43)
        tus = self.tussockness(x, y)
        cl = self.clump(x, y)
        return np.maximum(0.03, h * (1 + 0.35 * patch) * (1 + 0.6 * tus) * (0.75 + 0.5 * cl))
