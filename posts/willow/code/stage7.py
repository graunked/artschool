"""Stage 7: more species on the willow pipeline. A species x season sheet, then two scenes."""
import numpy as np, os, sys
from wpaint import *
from species import paint_species
from context import gravel, grass, water, scene_palette, cast_shadow, WATERB, SPECIES_GROUP, GROUP_SEASON
from stage4 import sky_ground
from sheets import pad_to, label_grid
OUT = "stage7"; os.makedirs(OUT, exist_ok=True)
L0 = sun(25, 40)


def cell(species, season, W_, H_, seed=1, cw=170, ch=None, p=None, ground="gravel"):
    ch = ch or int(max(150, H_ * 1.15 + 30))
    cv = sky_ground(cw, ch, ch - 22); gy = ch - 12
    rng = np.random.default_rng(seed)
    gm = np.zeros((ch, cw), bool); gm[ch - 22:] = True
    if ground == "grass":
        grass(cv, gm, rng, L0, blade=(1, 2), dens=0.4, crest=False)
    else:
        gravel(cv, gm, rng, L0, y_near=ch, y_far=ch - 22, stone=(0.8, 2.0), dens=0.3)
    cast_shadow(cv, cw / 2, gy, W_, H_, L0, fore=0.15, rng=rng)
    paint_species(cv, species, cw / 2, gy, W_, H_, L0, rng, p=p, foot=ground)
    return cv.render(scene_palette(season, "noon"))


ROWS = [
    ("willow", 110, 80, dict(), ["summer", "silvery", "autumn"], None),
    ("winter_willow", 110, 80, dict(), ["winter_red", "winter_yellow", "winter_grey"], None),
    ("sitka_alder", 120, 70, dict(), ["alder", "summer", "autumn"], None),
    ("dwarf_willow", 130, 14, dict(catkin_p=0.12), ["alpine", "summer", "autumn"], "grass"),
    ("red_alder", 70, 125, dict(), ["alder", "alder_autumn", "winter_grey"], None),
    ("black_cottonwood", 110, 150, dict(), ["cottonwood", "cottonwood_autumn", "winter_grey"], None),
    ("dogwood", 100, 75, dict(flowers=0.3), ["dogwood", "dogwood_autumn", "winter_dogwood"], None),
    ("salmonberry", 110, 80, dict(flowers=1.0), ["salmonberry", "salmonberry", "salmonberry_winter"], None),
]

if __name__ == "__main__":
    cells, rowlab = [], []
    for sp, W_, H_, p, seasons, gr in ROWS:
        for i, s in enumerate(seasons):
            pp = dict(p)
            if s.startswith("winter") or s.endswith("_winter"):
                pp["bare"] = True
            if sp == "dogwood" and s == "dogwood_autumn":
                pp = dict(berries=0.35)
            if sp == "salmonberry" and s == "summer":
                pp = dict(berries=1.0)
            cells.append(upscale(cell(sp, s, W_, H_, seed=3 + i, p=pp, ground=gr or "gravel"), 2))
        rowlab.append(sp.replace("_", " "))
    Hc = max(c.shape[0] for c in cells); Wc = max(c.shape[1] for c in cells)
    cells = [pad_to(c, Wc, Hc) for c in cells]
    save(label_grid(cells, 3, rowlabels=rowlab, collabels=["season 1", "season 2", "season 3"],
                    title="stage 7: species on one pipeline (columns: willow summer/silvery/autumn; winter red/yellow/grey; alder: summer/summer-willow palette/autumn; dwarf: alpine/summer/autumn; red alder: summer/autumn (green-brown)/winter)"),
         f"{OUT}/s7_species_x_season.png")


def clustered(rng, x0, x1, gap=(20, 70), size=(1, 5), spread=12.0):
    """Stand positions that clump: cluster centres with uneven gaps between them, 1-5 plants per
    cluster packed within `spread` px. Even spacing reads as a colonnade (critic 2's studio-wide
    note); real alder and willow stands come in groups with openings between."""
    xs = []
    x = x0 + rng.uniform(0, gap[0])
    while x < x1:
        n = int(rng.integers(size[0], size[1] + 1))
        for _ in range(n):
            xs.append(x + rng.normal(0, spread))
        x += rng.uniform(*gap) * (0.6 + 0.25 * n)
    return sorted(xs)


def riparian(season="summer", seed=21, w=360, h=200):
    """A west-side gravel-bed river: a band of red alder behind (pale trunks), willow scrub on the
    bar in front of it, the channel with its reflection, the near gravel. Winter: the same
    geometry with bare alders (mauve-grey haze over white trunks) and leafless willow."""
    rng = np.random.default_rng(seed)
    winter = season.startswith("winter")
    cv = sky_ground(w, h, 70)
    h0 = h
    bank = np.zeros((h, w), bool); bank[70:112] = True
    gravel(cv, bank, rng, L0, y_near=112, y_far=70, stone=(0.6, 1.6), dens=0.3)
    for x in rng.uniform(20, w - 20, 3):                  # cottonwoods stand above the alders, ragged
        W_ = rng.uniform(50, 64)
        paint_species(cv, "black_cottonwood", x, int(90 + rng.uniform(-2, 2)), W_, rng.uniform(86, 92), L0, rng,
                      band=1, p=dict(bare=winter), group=2)
    for x in clustered(rng, -20, w + 20, gap=(45, 110), size=(2, 5), spread=9):   # alder band, in groves
        W_ = rng.uniform(28, 46); H_ = min(72, W_ * rng.uniform(1.3, 1.7))
        paint_species(cv, "red_alder", x, int(94 + rng.uniform(-4, 4)), W_, H_, L0, rng, band=1,
                      p=dict(bare=winter, lean=rng.normal(0, 0.12)), group=1)
    x = -10
    while x < w + 10:                                    # dogwood and salmonberry along the bank toe
        W_ = rng.uniform(20, 36)
        sp = "dogwood" if rng.random() < 0.5 else "salmonberry"
        paint_species(cv, sp, x, int(99 + rng.uniform(-2, 2)), W_, W_ * 0.7, L0, rng, band=1, p=dict(bare=winter),
                      group=SPECIES_GROUP[sp])
        x += W_ * rng.uniform(0.8, 1.6)
    for x in clustered(rng, -20, w + 20, gap=(30, 90), size=(1, 4), spread=14):  # willow scrub on the bar
        W_ = rng.uniform(22, 60)
        sp = "winter_willow" if winter else "willow"
        paint_species(cv, sp, x, int(110 + rng.uniform(-2, 2)), W_, W_ * rng.uniform(0.55, 0.8), L0, rng, band=0, foot="gravel")
    wm = np.zeros((h, w), bool); wm[112:160] = True
    water(cv, wm, 112, rng, dash_p=0.06)
    near = np.zeros((h, w), bool); near[160:] = True
    gravel(cv, near, rng, L0, y_near=h, y_far=160, stone=(1.5, 5.0), dens=0.5)
    return cv


def run_scenes():
    for s, pal in (("summer", "summer"), ("autumn", "autumn"), ("winter", "winter_grey")):
        cv = riparian("winter" if s == "winter" else s)
        img = cv.render(scene_palette(pal, "golden" if s == "autumn" else "noon", groups=GROUP_SEASON[s]))
        save(img, f"{OUT}/s7_riparian_{s}_1x.png"); save(img, f"{OUT}/s7_riparian_{s}_3x.png", 3)
