"""Palette as light. Each material ramp is computed from a light state; the index canvas never changes.
Sea anchors are MEASURED in the sun state (LESSONS 1.2) and relit physically for other states:
   water colour = body * irradiance * sun tint  +  F_eff * sky   (in linear RGB)."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lab import srgb_to_lab, lab_to_srgb

def lab2lin(lab):
    c = lab_to_srgb(np.asarray(lab, float)) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def lin2srgb(lin):
    c = np.clip(lin, 0, 1)
    return np.clip(np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055) * 255, 0, 255)

LIGHTS = {
    # E: irradiance on the water relative to the calibrated sunny afternoon; tint: colour of that light;
    # sky: sky colour reflected at the 30 deg view; Feff: effective reflectance (Fresnel ~6% + roughness);
    # sun_el/az: for foam lit sides and glitter; haze: veil (colour, amount) for fog/smoke
    'sun':      dict(E=1.00, tint=(1.0, 1.0, 1.0),   sky=(72, -4, -24), Feff=0.06, sun_el=50, sun_az=205, direct=0.75, haze=None),
    'overcast': dict(E=0.40, tint=(0.97, 1.0, 1.03), sky=(80, -1, -3),  Feff=0.09, sun_el=50, sun_az=205, direct=0.0, haze=None),
    'golden':   dict(E=0.42, tint=(1.10, 0.86, 0.62), sky=(74, 2, -8),  Feff=0.07, sun_el=9, sun_az=285, direct=0.8, haze=None),
    'dusk':     dict(E=0.10, tint=(1.0, 0.62, 0.62), sky=(52, 12, -22), Feff=0.09, sun_el=-2, sun_az=295, direct=0.0, haze=None),
    'fog':      dict(E=0.45, tint=(1.0, 1.0, 1.0),   sky=(84, -1, -2),  Feff=0.10, sun_el=40, sun_az=205, direct=0.0, haze=((82, -1, -2), 0.45)),
}

# measured sun-state anchors (L*, a*, b*)
SEA_DEEP = (36, -13, -17)
SEA_SAND_SHALLOW = (64, -15, -5)
SEA_ROCK_SHALLOW = (58, -11, -13)
D0 = 8.0           # m: e-folding depth of the bottom's contribution (fit: sand 4 m -> L* 49, 17 m -> 36)
KELP = [(25, 1, -10), (30, 3, -14), (37, -6, -17)]          # core warmed toward the brown floats; mid/edge measured          # canopy core / mid / edge-speck (measured 23-45)
AER = [(62, -16, -7), (70, -17, -5), (77, -16, -4)]          # photo's churned pool ~ #8cc8c0          # bubble plume (measured 64, sd 18)
FOAM_ALBEDO = 0.82

DEPTH_EDGES = np.array([0.7, 1.6, 3.0, 5.0, 8.0, 13.0])      # step 6 (shallowest) .. 0 (deep)
STEP_DEPTH = np.array([18.0, 10.0, 6.5, 4.0, 2.2, 1.1, 0.4])   # representative depth of each step

def sea_body_lab(d, rock):
    w = 1 - np.exp(-np.asarray(d, float) / D0)
    sh = np.array(SEA_SAND_SHALLOW) * (1 - rock) + np.array(SEA_ROCK_SHALLOW) * rock
    return sh * (1 - w)[..., None] + np.array(SEA_DEEP) * w[..., None] if np.ndim(d) else sh * (1 - w) + np.array(SEA_DEEP) * w

def relight(lab_sun, L):
    """sun-state measured colour -> this light state."""
    ref = LIGHTS['sun']
    lin = lab2lin(lab_sun)
    sky_s = lab2lin(ref['sky']) * ref['Feff']
    body = np.maximum(lin - sky_s, 0)                 # remove the sunny sky reflection
    out = body * L['E'] * np.array(L['tint']) + lab2lin(L['sky']) * L['Feff']
    if L['haze']:
        hc, ha = L['haze']; out = out * (1 - ha) + lab2lin(hc) * ha
    return out

def foam_colours(L):
    """foam: 0 shadow (sky only), 1 turned away, 2 lit, 3 highlight."""
    sky = lab2lin(L['sky']); sunc = np.array(L['tint']) * L['E']
    amb = sky * 0.55 * max(L['E'], 0.25) / 0.55 * 0.9 + 0.08 * sunc
    dir_ = sunc * L['direct']
    cols = [FOAM_ALBEDO * (amb * 0.75), FOAM_ALBEDO * (amb + 0.35 * dir_), FOAM_ALBEDO * (amb + 0.8 * dir_),
            FOAM_ALBEDO * (amb + 1.05 * dir_) + 0.03]
    if L['direct'] == 0:     # flat light: foam is the brightest thing, but only slightly above grey
        base = FOAM_ALBEDO * (lab2lin(L['sky']) * 0.95) * max(L['E'] / 0.4, 0.3)
        cols = [base * 0.62, base * 0.78, base * 0.92, base * 1.0]
    if L['haze']:
        hc, ha = L['haze']; cols = [c * (1 - ha * 0.6) + lab2lin(hc) * ha * 0.6 for c in cols]
    return [np.clip(c, 0, 1) for c in cols]

# ---- land (context only; rock/forest are other studies' subjects) ----
LAND = {
    'rock':   [(24, 2, 4), (33, 2, 6), (44, 3, 9), (56, 3, 11), (66, 2, 10)],
    'wetrock':[(18, 0, 0), (26, 0, 2), (34, 0, 3)],
    'sand':   [(62, 1, 8), (72, 1, 10), (80, 1, 11), (87, 1, 10)],
    'wetsand':[(46, 0, 3), (54, 0, 4), (62, -1, 2)],
    'gravel': [(40, 0, 3), (52, 0, 4), (62, 0, 5), (70, 0, 4)],
    'forest': [(10, -6, 4), (17, -12, 8), (26, -16, 14), (36, -18, 20)],
}

def build(light='sun'):
    """returns dict ramp -> array of sRGB colours (steps)."""
    L = LIGHTS[light]
    P = {}
    P['sea_sand'] = np.array([lin2srgb(relight(sea_body_lab(d, 0.0), L)) for d in STEP_DEPTH])
    P['sea_rock'] = np.array([lin2srgb(relight(sea_body_lab(d, 1.0), L)) for d in STEP_DEPTH])
    P['kelp'] = np.array([lin2srgb(relight(c, L)) for c in KELP])
    P['aer'] = np.array([lin2srgb(relight(c, L)) for c in AER])
    P['foam'] = np.array([lin2srgb(c) for c in foam_colours(L)])
    sunE = L['E']; tint = np.array(L['tint'])
    for k, cols in LAND.items():
        out = []
        for c in cols:
            lin = lab2lin(c) * (0.35 + 0.65 * sunE) * (tint * 0.5 + 0.5)
            if L['haze']:
                hc, ha = L['haze']; lin = lin * (1 - ha) + lab2lin(hc) * ha
            out.append(lin2srgb(lin))
        P[k] = np.array(out)
    # glint: the sun mirrored
    P['glint'] = np.array([lin2srgb(np.clip(np.array(L['tint']) * 0.9 + 0.1, 0, 1)), lin2srgb(np.clip(lab2lin(L['sky']) * 1.1, 0, 1))])
    return build_extra(light, P)

HOR = {'sun': (86, -4, -10), 'overcast': (86, 0, -2), 'golden': (84, 6, 18), 'dusk': (66, 18, 8), 'fog': (86, -1, -2)}
MIRROR_F = [0.05, 0.10, 0.17, 0.27, 0.40, 0.58, 0.80]

def build_extra(light, P):
    L = LIGHTS[light]
    deep = relight(SEA_DEEP, L)
    hor = lab2lin(HOR[light]); zen = lab2lin(L['sky'])
    P['mirror'] = np.array([lin2srgb(deep * (1 - f) + (hor * f ** 0.5 + zen * (1 - f ** 0.5)) * f * 1.0) for f in MIRROR_F])
    P['sky'] = np.array([lin2srgb(zen * (1 - a) + hor * a) for a in np.linspace(0, 1, 6)])
    return P

RAMPS = ['sea_sand', 'sea_rock', 'kelp', 'aer', 'foam', 'rock', 'wetrock', 'sand', 'wetsand', 'gravel', 'forest', 'glint', 'mirror', 'sky']
RID = {k: i for i, k in enumerate(RAMPS)}

def to_rgb(ramp_idx, step_idx, light='sun'):
    P = build(light)
    out = np.zeros(ramp_idx.shape + (3,), np.uint8)
    for k, i in RID.items():
        m = ramp_idx == i
        if m.any():
            cols = P[k]; s = np.clip(step_idx[m], 0, len(cols) - 1)
            out[m] = cols[s].astype(np.uint8)
    return out
