"""Stage 5 settings for the willow: gravel bars, water with reflections, grass edges.
Small painters that follow the studio's measured rules (SYNTHESIS.md): stones with a light top row
and dark bottom row; masses meet level ground in a flat run with a darker contact line; reflection
is an exact mirror with compressed range, darks kept, lights dropped, horizontal dash rows; grass
as paired blade strokes whose tips break the crest and overlap the foot of what stands in it.
"""
import numpy as np
from scipy import ndimage as ndi
from wpaint import *
from foliage import poisson, stroke_run, snap_dir
from palette import mix, to_lab, to_rgb, willow_palette
from willow import BAND, FOLIAGE, LEAFS

REFL = 1000         # reflected materials: m + REFL
WATERB = 12         # water body ramp


def gravel(cv, mask, rng, L, y_near, y_far, stone=(1.2, 5.0), dens=0.55):
    """A braided-river gravel bar: a pale matrix with stones whose size grows toward the viewer.
    Each stone: light top row (sky + sun), mid body, dark bottom row, a one-pixel cast shadow
    on the side away from the sun."""
    h, w = cv.h, cv.w
    n = rng.random((h, w))
    cv.mat[mask] = GRAVEL
    cv.step[mask] = np.where(n[mask] < 0.25, 2, np.where(n[mask] < 0.9, 3, 4))
    lx = -np.sign(L[0]) if abs(L[0]) > 0.05 else 1
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return
    for (px, py) in poisson(h, w, 2.2, rng, mask=mask):
        t = np.clip((py - y_far) / max(y_near - y_far, 1), 0, 1)
        if rng.random() > dens * (0.4 + 0.6 * t):
            continue
        r = stone[0] + (stone[1] - stone[0]) * t * rng.uniform(0.35, 1.0)
        a, b = r * rng.uniform(0.9, 1.4), max(0.7, r * rng.uniform(0.55, 0.85))
        tone = rng.choice([0, 0, 1, -1])            # stones differ: pale quartz, grey, dark greywacke
        x0, x1 = int(px - a - 1), int(px + a + 2); y0, y1 = int(py - b - 1), int(py + b + 2)
        for y in range(max(0, y0), min(h, y1)):
            for x in range(max(0, x0), min(w, x1)):
                if not mask[y, x]:
                    continue
                d = ((x - px) / a) ** 2 + ((y - py) / b) ** 2
                if d < 1:
                    v = (y - py) / b
                    if b < 1.1:                       # a pebble: one lit pixel over one dark
                        s = 4 if v < 0 else 2
                    else:
                        s = 5 if v < -0.4 else (4 if v < 0.35 else 2)
                        if v > 0.7 and d > 0.5:
                            s = 1                     # the underside row only on real stones
                    s = int(np.clip(s + tone, 1, 5))
                    cv.mat[y, x] = GRAVEL; cv.step[y, x] = s
        # cast shadow: one pixel beyond the stone, away from the sun, on its lower half
        sx = int(round(px + lx * (a + 1))); sy = int(round(py + b * 0.3))
        if 0 <= sx < w and 0 <= sy < h and mask[sy, sx]:
            cv.mat[sy, sx] = GRAVEL; cv.step[sy, sx] = 1


def contact_shadow(cv, foliage_mask, gy_map, L, length=4):
    """The darker ground under and beside a bush, away from the sun: a flat run of the gravel's
    darkest steps along the contact, longer on the shadow side."""
    h, w = cv.h, cv.w
    lx = -np.sign(L[0]) if abs(L[0]) > 0.05 else 1
    for x in range(w):
        col = np.nonzero(foliage_mask[:, x])[0]
        if not len(col):
            continue
        yb = col.max() + 1
        for k in range(2):
            if yb + k < h and cv.mat[yb + k, x] == GRAVEL:
                cv.step[yb + k, x] = 0 if k == 0 else min(cv.step[yb + k, x], 1)
        for k in range(1, length + 1):
            xx = x + int(lx * k)
            if 0 <= xx < w and yb < h and cv.mat[yb, xx] == GRAVEL and not foliage_mask[yb, xx]:
                cv.step[yb, xx] = min(cv.step[yb, xx], 1)


def water(cv, mask, y_line, rng, dash_p=0.05, sheen=3, shallow=None):
    """Water: an exact mirror of what stands above y_line, in reflected (compressed) materials;
    where nothing is reflected, the sky's ramp one step down; horizontal dash rows; a pale sheen
    band just under reflected objects; dark shallows near the shore (shallow mask)."""
    h, w = cv.h, cv.w
    src_m = cv.mat.copy(); src_s = cv.step.copy()
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            yy = 2 * y_line - y - 1
            if 0 <= yy < h:
                m, s = src_m[yy, x], src_s[yy, x]
            else:
                m, s = SKY, 0
            if m == SKY:
                cv.mat[y, x] = WATERB; cv.step[y, x] = max(0, 3 - s)   # sky reflects lighter far, darker near
            else:
                cv.mat[y, x] = m + REFL; cv.step[y, x] = s
    # sheen just below reflected objects (grazing reflection is the brightest)
    refl = mask & (cv.mat >= REFL)
    below = mask & ~refl & ndi.binary_dilation(refl, structure=np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]]), iterations=sheen)
    cv.mat[below] = WATERB; cv.step[below] = 4
    # horizontal dash rows: short runs one step lighter
    for y in range(h):
        if not mask[y].any():
            continue
        x = 0
        while x < w:
            if rng.random() < dash_p and mask[y, x]:
                L_ = int(rng.integers(2, 7))
                for k in range(L_):
                    if x + k < w and mask[y, x + k]:
                        if cv.mat[y, x + k] == WATERB:
                            cv.step[y, x + k] = min(4, cv.step[y, x + k] + 1)
                        else:
                            cv.step[y, x + k] = min(5, cv.step[y, x + k] + 1)
                x += L_ + 3
            x += 1
    if shallow is not None:
        sh = mask & shallow
        cv.mat[sh] = WATERB; cv.step[sh] = np.where(rng.random(sh.sum()) < 0.5, 0, 1)


def grass(cv, mask, rng, L, blade=(2, 6), dens=0.6, lean=0.3, crest=True, over=None):
    """Grass as paired blade strokes: [dark gap][lit blade][stem], mostly vertical, leaning
    slightly; tips break above the mask top; `over` lets blades overlap a bush's foot."""
    h, w = cv.h, cv.w
    n = rng.random((h, w))
    cv.mat[mask] = GRASS; cv.step[mask] = np.where(n[mask] < 0.45, 1, 2)
    allowed = mask.copy()
    if crest:
        allowed |= ndi.binary_dilation(mask, structure=np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]), iterations=blade[1])
    if over is not None:
        allowed |= over
    pts = poisson(h, w, 1.6, rng, mask=mask)
    order = np.argsort(pts[:, 1]) if len(pts) else []
    for i in order:
        px, py = pts[i]
        if rng.random() > dens:
            continue
        L_ = int(rng.integers(blade[0], blade[1] + 1))
        a = -np.pi / 2 + rng.normal(0, lean)
        run = stroke_run(int(px), int(py), snap_dir(np.cos(a), np.sin(a)), L_)
        hi = 4 if rng.random() < 0.55 else 3
        for k, (x, y) in enumerate(run):
            if not (0 <= x < w and 0 <= y < h) or not allowed[y, x]:
                break
            cv.mat[y, x] = GRASS; cv.step[y, x] = hi if k >= L_ - 2 else 3 if k >= L_ // 2 else 2
            # paired: a stem pixel beside, one step darker; a dark gap on the other side
            if 0 <= x + 1 < w and allowed[y, x + 1] and rng.random() < 0.5:
                cv.mat[y, x + 1] = GRASS; cv.step[y, x + 1] = max(1, cv.step[y, x] - 1)
            if 0 <= x - 1 < w and mask[y, x - 1] and rng.random() < 0.35:
                cv.mat[y, x - 1] = GRASS; cv.step[y, x - 1] = 0


GROUP = 210          # material offset per species group (each group has its own palette)
SPECIES_GROUP = {"willow": 0, "winter_willow": 0, "dwarf_willow": 0, "sitka_alder": 1, "red_alder": 1,
                 "black_cottonwood": 2, "dogwood": 3, "salmonberry": 4}
# what each group wears in a scene's season (alders drop their leaves green-brown; dogwood turns purple-red)
GROUP_SEASON = {
    "summer": {0: "summer", 1: "alder", 2: "cottonwood", 3: "dogwood", 4: "salmonberry"},
    "autumn": {0: "autumn", 1: "alder_autumn", 2: "cottonwood_autumn", 3: "dogwood_autumn", 4: "autumn"},
    "winter": {0: "winter_grey", 1: "winter_grey", 2: "winter_grey", 3: "winter_dogwood", 4: "salmonberry_winter"},
    "spring": {0: "spring", 1: "alder", 2: "spring", 3: "spring", 4: "spring"},
}


def scene_palette(season="summer", hour="noon", depths=(0.0, 0.3, 0.55, 0.8), water_col=(62, 78, 74), groups=None):
    """Willow bands + grass + water + reflected versions of everything. groups: {group: season} gives
    each species group its own palette (materials offset by GROUP * group)."""
    from willow import band_palette
    P = band_palette(season, hour, depths)
    for g, sg in (groups or {}).items():
        if g == 0:
            continue
        for (m, s), col in band_palette(sg, hour, depths).items():
            if (m % BAND in FOLIAGE and m > 0 and m < 130) or m in (16, 17):
                P[(m + GROUP * g, s)] = col
    Hh = __import__("palette").HOURS[hour]
    sun, amb = Hh["sun"], Hh["amb"]
    gr_lo = {"autumn": (60, 52, 24), "spring": (30, 50, 18)}.get(season, (26, 44, 20))
    gr_hi = {"autumn": (214, 180, 96), "spring": (170, 206, 80)}.get(season, (150, 184, 76))
    from palette import ramp
    for i, c in enumerate(ramp(gr_lo, gr_hi, 5, sun, amb, warm=0.2, cool=0.2)):
        P[(GRASS, i)] = c
    sky_lo, sky_hi = Hh["sky"]
    wat = [mix(water_col, amb, 0.2), mix(sky_lo, water_col, 0.55), mix(sky_lo, water_col, 0.35),
           mix(sky_hi, sky_lo, 0.5), mix(sky_hi, (255, 255, 255), 0.2)]
    wat = [to_rgb(to_lab(c) * np.array([Hh.get("key", 1.0), 1, 1])) for c in wat]
    for i, c in enumerate(wat):
        P[(WATERB, i)] = c
    # reflections: exact colours, lights dropped ~13 L*, darks kept, slight cool tint
    for (m, s), c in list(P.items()):
        if m >= REFL or m == WATERB:
            continue
        lab = to_lab(c)
        drop = 0.16 * np.clip((lab[0] - 0.3) / 0.5, 0, 1)
        lab[0] -= drop
        lab[1:] *= 0.8                 # a little less chroma under water
        rc = mix(to_rgb(lab), water_col, 0.28)
        P[(m + REFL, s)] = rc
    return P


def cast_shadow(cv, cx, gy, W_, H_, L, fore=0.18, rng=None):
    """A bush's shadow on the ground: the ground itself in its shadow family (two steps down), the
    stones keep their marks. An ellipse from the foot toward the side away from the sun, longer as
    the sun drops; foreshortened by the camera (fore: low camera ~0.15, 30 deg oblique ~0.35)."""
    h, w = cv.h, cv.w
    el = np.degrees(np.arcsin(np.clip(L[1], -1, 1)))
    lxz = np.hypot(L[0], L[2]) + 1e-6
    reach = H_ * 0.55 / max(np.tan(np.radians(max(el, 8))), 0.25)
    dx = -L[0] / lxz * reach * 0.8
    dy = -L[2] / lxz * reach * fore * 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    ax = W_ * 0.5 + abs(dx) * 0.45; ay = max(1.5, W_ * fore * 0.5 + abs(dy) * 0.45)
    ecx, ecy = cx + dx * 0.5, gy - 1 + dy * 0.5
    d = ((xx - ecx) / ax) ** 2 + ((yy - ecy) / ay) ** 2
    edge = 1.0 if rng is None else 1.0 + 0.25 * (rng.random((h, w)) - 0.5)
    sh = (d < edge) & (yy >= gy - 1 - max(0, -dy)) & np.isin(cv.mat, (GRAVEL, GRASS))
    cv.step[sh] = np.maximum(cv.step[sh].astype(int) - 2, 0)
    return sh
