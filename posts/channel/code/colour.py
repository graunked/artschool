"""colour.py -- the palette is the light (Ferrari). The index canvas painted in grey is kept as is;
an hour swaps every ramp's colours. Each colour keeps the luminance its index proved in grey
(scaled by the hour's key), and takes its hue from a walk between the ambient light (shadows) and
the sun (lights), times the material's albedo. The sky is its own hand-set ramp per hour, and the
water reuses it (plus a few water-only tints)."""
import numpy as np
import channel as ch

HOURS = {
    'midday': dict(sun=(255, 246, 226), amb=(126, 146, 232), key=1.0, lift=0.0,
                   sky=['#e6eef2', '#d6e6f0', '#c4dcee', '#b2d0ea', '#a0c4e6', '#8eb8e0', '#7eacda', '#709fd2'],
                   milk=(150, 176, 178), bed=(96, 88, 76)),
    'overcast': dict(sun=(214, 216, 222), amb=(170, 176, 188), key=0.92, lift=0.06,
                     sky=['#dcdee2', '#d4d6dc', '#cdd0d6', '#c6c9d0', '#bfc2ca', '#b8bbc4', '#b1b4be', '#aaaeb8'],
                     milk=(158, 170, 172), bed=(92, 90, 86)),
    'dusk': dict(sun=(255, 160, 104), amb=(120, 100, 170), key=0.78, land_key=0.55, lift=-0.04,
                 sky=['#f6c09a', '#f0ae94', '#e8a09a', '#dc96a2', '#cc90aa', '#bc8cb0', '#aa88b4', '#9a84b6'],
                 milk=(170, 140, 150), bed=(80, 62, 70)),
}

ALBEDO = {
    'grav': (148, 140, 128), 'grass': (176, 150, 92), 'earth': (110, 88, 70), 'bould': (140, 136, 132),
    'wil': (110, 128, 96), 'ridge': (120, 124, 140), 'scrub': (80, 96, 76),
}


def _lin(c):
    c = np.asarray(c, float) / 255
    return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)


def _srgb(l):
    l = np.clip(l, 0, 1)
    return np.clip(255 * np.where(l > 0.0031308, 1.055 * l ** (1 / 2.4) - 0.055, 12.92 * l), 0, 255)


def _Y(lin):
    return lin @ np.array([0.2126, 0.7152, 0.0722])


def fit_lum(rgb_lin, target_Y):
    y = max(_Y(rgb_lin), 1e-6)
    out = rgb_lin * (target_Y / y)
    if out.max() > 1:                                  # desaturate toward white rather than clip hue
        m = out.max()
        out = out / m
        yy = _Y(out)
        t = (target_Y - yy) / max(1 - yy, 1e-6)
        out = out + (1 - out) * np.clip(t, 0, 1)
    return out


def hexrgb(h):
    h = h.lstrip('#'); return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)


def palette(Pgrey, hour):
    """a colour Pal with exactly the grey Pal's layout."""
    L = HOURS[hour]
    sun, amb = _lin(L['sun']), _lin(L['amb'])
    cols = np.zeros_like(Pgrey.arr, dtype=float)
    for name, idx in Pgrey.ramps.items():
        greys = Pgrey.arr[idx][:, 0].astype(float)
        n = len(idx)
        if name == 'sky':
            sk = [hexrgb(c) for c in L['sky']]
            for k, i in enumerate(idx):
                cols[i] = sk[min(k, len(sk) - 1)]
            continue
        if name == 'wmir':      # the milky mirror: the sky's own steps clouded by the water's rock flour
            sk = [hexrgb(c) for c in L['sky']]
            milk = _lin(L['milk'])
            for k, i in enumerate(idx):
                c = _lin(sk[min(k, len(sk) - 1)])
                Yt = _Y(c)
                cols[i] = _srgb(fit_lum(c * (1 - L.get('turb', 0.5)) + milk * L.get('turb', 0.5), Yt))
            continue
        if name == 'wat':
            milk, bed = _lin(L['milk']), _lin(L['bed'])
            for k, i in enumerate(idx):
                t = k / max(n - 1, 1)
                c = bed * (1 - t) + milk * t
                Yt = _Y(_lin((greys[k],) * 3)) * L['key']
                cols[i] = _srgb(fit_lum(c, Yt))
            continue
        alb = _lin(ALBEDO.get(name, (128, 128, 128)))
        g0, g1 = greys.min(), greys.max()
        for k, i in enumerate(idx):
            t = (greys[k] - g0) / max(g1 - g0, 1)
            light = amb * (1 - t ** 1.3) + sun * t ** 1.3          # shadows to ambient, lights to sun
            c = alb * light
            Yt = _Y(_lin((greys[k],) * 3)) * L.get('land_key', L['key']) * (1 + L['lift'] * (1 - t))
            if name in ('ridge', 'scrub'):                 # aerial perspective: toward the horizon sky
                hz = _lin(hexrgb(L['sky'][0]))
                c2 = fit_lum(c, Yt)
                w_ = 0.6 if name == 'ridge' else 0.3
                cols[i] = _srgb(c2 * (1 - w_) + fit_lum(hz, Yt) * w_)
                continue
            cols[i] = _srgb(fit_lum(c, Yt))
    P = ch.Pal([])
    P.ramps = Pgrey.ramps
    P.arr = np.clip(cols, 0, 255).astype(np.uint8)
    P.cols = [tuple(c) for c in P.arr]
    return P
