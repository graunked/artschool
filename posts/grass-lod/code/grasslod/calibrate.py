"""Tone lock: keep the squinted tone of a patch constant across zoom.

The design value V means "what this grass looks like at the reference distance" (strokes
about ell_ref px long).  At other distances the mark vocabulary changes (strokes, ticks,
dots, bare dither) and each vocabulary has a different mean offset from V (dark gaps pull
down, clipped highlights can't push up).  We measure that offset once, on a flat patch,
for a grid of stroke lengths and design values, and paint every frame with
V' = V - (offset(ell, V) - offset(ell_ref, V)).  So a patch's squinted tone at 0.3 px/m is
the squinted tone it had at 30 px/m, and a slow zoom never brightens or darkens.
"""
import json, os, hashlib
import numpy as np

_CACHE = {}
ELLS = np.array([0.08, 0.15, 0.25, 0.4, 0.6, 0.9, 1.3, 1.8, 2.5, 3.5, 5.0, 7.0, 10.0, 14.0, 20.0, 28.0])
VLIT = np.array([3.6, 4.3, 5.0, 5.6])
VSH = np.array([0.5, 1.1, 1.7, 2.4])
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "cache")


def _key(world, mp):
    d = dict(species=world.species, grazing=round(world.grazing, 3), wind=round(world.wind, 2),
             mp={k: v for k, v in mp.items() if k != "tone_lock"})
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def table(world, mp):
    k = _key(world, mp)
    if k in _CACHE:
        return _CACHE[k]
    os.makedirs(CACHE_DIR, exist_ok=True)
    fn = os.path.join(CACHE_DIR, f"tone_{k}.npz")
    if os.path.exists(fn):
        z = np.load(fn)
        _CACHE[k] = (z["lit"], z["sh"])
        return _CACHE[k]
    lit, sh = _measure(world, mp)
    np.savez(fn, lit=lit, sh=sh)
    _CACHE[k] = (lit, sh)
    return lit, sh


class _Flat:
    """Stand-in raster of a flat, uniformly lit patch."""


def _flat_setup(world, mp, ell, W=160, H=120, theta=30.0):
    from .render import Camera
    import copy
    fw = copy.copy(world)
    fw.height = lambda x, y, min_wl=0.0: np.zeros(np.shape(x))
    fw.water_level = -10.0
    from .world import SPECIES
    sp = SPECIES[world.species]
    hmean = sp[0] * (1 - 0.75 * world.grazing)
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    cam = Camera(ppm=ell / (hmean * np.sqrt(c * c + (mp["lean_top"] * s ** 3) ** 2)), theta=theta, W=W, H=H)
    R = _Flat()
    R.cam = cam
    R.min_wl = 1 / cam.ppm
    R.sun_screen_x = -1
    xs = (np.arange(W) + 0.5 - W / 2) / cam.ppm
    ys = ((H / 2 - (np.arange(H) + 0.5)) / cam.ppm) / cam.s
    R.X, R.Y = np.meshgrid(xs, ys)
    R.depth = R.Y.copy()
    R.mat = np.ones((H, W), np.int8)
    m = 3 * sp[0] + 3 / cam.ppm
    return fw, cam, R, (xs.min() - m, xs.max() + m, ys.min() - m, ys.max() + m * 3)


_COV = {}


def coverage_ratio(world, mp, ell):
    """Coverage rate a(ell): painted coverage = 1 - exp(-a * keep), measured on a flat
    patch (glyph_area is only a guess: blades of one tuft overlap, marks overlap...)."""
    k = "cov" + _key(world, mp)
    if k not in _COV:
        fn = os.path.join(CACHE_DIR, f"cova_{_key(world, mp)}.npy")
        if os.path.exists(fn):
            _COV[k] = np.load(fn)
        else:
            from .marks import tuft_population, paint_tufts, keep_for, target_coverage
            r = np.ones(len(ELLS))
            for ie, e in enumerate(ELLS):
                fw, cam, R, box = _flat_setup(world, mp, e)
                keep = keep_for(world, mp, e, cam.ppm, cam.s, ratio=1.0)
                pop = tuft_population(fw, *box, min(1.0, keep * 1.8), seed=7)
                H, W = R.X.shape
                canvas = np.full((H, W), np.nan)
                R.qbase = None
                if pop is not None:
                    pop["keep"] = keep
                    paint_tufts(canvas, np.zeros((H, W), np.int8), np.full((H, W), -1, np.int64), R,
                                np.full((H, W), 4.3), np.ones((H, W), bool), fw, pop, mp, fam_top=(2.9, 6.0), fam_bot=(0, 0))
                meas = (~np.isnan(canvas[10:-4, 8:-8])).mean()
                # overlapping marks saturate: coverage = 1 - exp(-a * keep).  Store the
                # rate a, so keep_for can invert it exactly for any target.
                r[ie] = -np.log(max(1e-4, 1 - min(meas, 0.995))) / max(keep, 1e-9)
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.save(fn, r)
            _COV[k] = r
    le = np.log(np.clip(ell, ELLS[0], ELLS[-1]))
    return float(np.exp(np.interp(le, np.log(ELLS), np.log(_COV[k]))))


def _measure(world, mp):
    from .render import Camera
    from .marks import tuft_population, paint_tufts, glyph_area
    from .world import SPECIES
    from dataclasses import replace
    import copy
    fw = copy.copy(world)
    fw.height = lambda x, y, min_wl=0.0: np.zeros(np.shape(x))
    fw.water_level = -10.0
    sp = SPECIES[world.species]
    hmean = sp[0] * (1 - 0.75 * world.grazing)
    W, H = 160, 120
    theta = 30.0
    out = {True: np.zeros((len(ELLS), len(VLIT))), False: np.zeros((len(ELLS), len(VSH)))}
    for ie, ell in enumerate(ELLS):
        cam = Camera(ppm=ell / (hmean * np.sqrt(np.cos(np.radians(theta))**2 + (mp['lean_top'] * np.sin(np.radians(theta))**3)**2)), theta=theta, W=W, H=H)
        R = _Flat()
        R.cam = cam
        R.min_wl = 1 / cam.ppm
        R.sun_screen_x = -1
        xs = (np.arange(W) + 0.5 - W / 2) / cam.ppm
        rows = np.arange(H)
        ys = ((H / 2 - (rows + 0.5)) / cam.ppm) / cam.s
        R.X, R.Y = np.meshgrid(xs, ys)
        R.depth = R.Y.copy()
        R.mat = np.ones((H, W), np.int8)
        from .marks import keep_for
        keep = keep_for(world, mp, ell, cam.ppm, cam.s)
        m = 3 * sp[0] + 3 / cam.ppm
        pop = tuft_population(fw, xs.min() - m, xs.max() + m, ys.min() - m, ys.max() + m * 3, min(1.0, keep * 1.8), seed=7)
        if pop is not None:
            pop['keep'] = keep
        for litfam, vals in ((True, VLIT), (False, VSH)):
            for iv, v in enumerate(vals):
                V = np.full((H, W), v)
                L = np.full((H, W), litfam)
                from .marks import quantize_base
                iy = np.broadcast_to(np.arange(H)[:, None], (H, W)).astype(np.int64)
                R.qbase = quantize_base(V, R.X, R.Y, cam.ppm, cam.s,
                                        mp["band_w"] if litfam else mp["band_w_shadow"], iy=iy)
                canvas = np.full((H, W), np.nan)
                cm = np.zeros((H, W), np.int8)
                pr = np.full((H, W), -1, np.int64)
                if pop is not None:
                    paint_tufts(canvas, cm, pr, R, V, L, fw, pop, mp, fam_top=(2.9, 6.0), fam_bot=(0, 0))
                inner = (slice(10, H - 4), slice(8, W - 8))
                c = canvas[inner]
                mk = ~np.isnan(c)
                est = np.where(mk, np.clip(np.floor(np.nan_to_num(c) + 0.5), 0, 6), R.qbase[inner])
                out[litfam][ie, iv] = est.mean() - v
    return out[True], out[False]


def offset(world, mp, ell, V, lit):
    """Mean offset (steps) of the painted patch relative to V, at stroke length ell."""
    tl, ts = table(world, mp)
    le = np.log(np.clip(ell, ELLS[0], ELLS[-1]))
    lE = np.log(ELLS)
    i = np.clip(np.searchsorted(lE, le) - 1, 0, len(ELLS) - 2)
    t = (le - lE[i]) / (lE[i + 1] - lE[i])

    def interp(tab, vals, Vv):
        row = tab[i] * (1 - t) + tab[i + 1] * t
        return np.interp(Vv, vals, row)
    return np.where(lit, interp(tl, VLIT, V), interp(ts, VSH, V))


def correction(world, mp, ell, V, lit, ell_ref=8.0):
    return offset(world, mp, ell, V, lit) - offset(world, mp, ell_ref, V, lit)
