"""Load a sim result (Modal npz) + place into the painter's field dict."""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .place import PLACES, load
from dem import window
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'cache')
def scene(place, result, tide=0.45, T=None):
    P = load(place, tide)
    d = dict(np.load(os.path.join(CACHE, result) if not os.path.isabs(result) else result))
    pack, X0, Y1, (c0, r0, c1, r1) = PLACES[place]
    chm = np.nan_to_num(window(pack, 'chm', X0 + c0, Y1 - r1, X0 + c1, Y1 - r0, dtype=np.uint8))
    f = lambda k: d[k].astype(np.float32)
    S = dict(depth=np.nan_to_num(P['depth'], nan=30.0), rock=f('rock'), rough=f('rough'), kelp=f('kelp'),
             kelp_rho=f('kelp_rho'), W=f('W'), F=f('F'), A=f('A'), Hinst=f('Hinst'), u=f('u'), v=f('v'),
             tt=d['t'].astype(np.float32), brkf=d['brk'].astype(np.float32), T=float(T or 8.3),
             t_now=float(d['t_now']), chm=chm, H=f('H'), kx=f('kx'), ky=f('ky'),
             parts=dict(x=d['px'], y=d['py']))
    from scipy import ndimage as nd
    S['rock_s'] = nd.gaussian_filter(S['rock'], 3.0)
    S['depth_s'] = nd.gaussian_filter(S['depth'], 6.0)
    S['surf'] = np.clip(nd.gaussian_filter(S['brkf'], 10.0) * 2.5, 0, 1).astype(np.float32)   # breaking-zone indicator
    S['eta'] = eta_field(S)
    zland = np.nan_to_num(P['z'], nan=-40) + chm
    return P, S, zland

def eta_field(S):
    """sea-surface elevation now (m, about still water), 1 m grid: each crest's own height, a Stokes-sharpened
    profile, flattened where broken (bores are lower and flat-topped)."""
    T = S['T']; q = np.mod((S['t_now'] - S['tt']) / T, 1.0)
    a = 0.5 * S['Hinst']
    ph = 2 * np.pi * q
    prof = np.cos(ph) + 0.35 * (np.cos(2 * ph) - 0.3)             # peaked crests, long flat troughs
    brk = S['brkf'] > 0.5
    prof = np.where(brk, np.clip(prof, -1, 0.6), prof)
    d = S['depth']
    return (a * prof * np.clip(d / 0.8, 0, 1)).astype(np.float32)
