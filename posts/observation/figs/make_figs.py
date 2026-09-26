"""New figures for the observation lessons, made with the REAL study functions from
pixelart-studies/pedagogy/observation/translate_studies.py (imported, not copied).
Writes only into this folder. Run: PYTHONDONTWRITEBYTECODE=1 nice -n 10 python3 make_figs.py"""
import os, sys, numpy as np
from PIL import Image, ImageDraw
OUT = os.path.dirname(os.path.abspath(__file__)) + "/.."
STUDY = os.path.expanduser("~/work/pixelart-studies/pedagogy/observation")
sys.dont_write_bytecode = True
os.chdir(STUDY); sys.path.insert(0, STUDY)
import translate_studies as T

CM = {6: "rim", 5: "lit", 4: "half", 3: "shade", 2: "core", 1: "contact"}
def rock_img(z, h, w, horizon=None):
    img = np.zeros((h, w, 3), np.uint8); img[:] = T.C["g_ochre"]
    img[:horizon if horizon is not None else int(h * 0.47)] = T.C["sky1"]
    for k, c in CM.items(): img[z == k] = T.C[c]
    return img
def up(img, s=8): return Image.fromarray(img).resize((img.shape[1] * s, img.shape[0] * s), Image.NEAREST)
def row(panels, caps, fn, pad=10):
    h = max(p.size[1] for p in panels); w = sum(p.size[0] for p in panels) + pad * (len(panels) - 1)
    out = Image.new("RGB", (w, h + 18), (245, 245, 240)); d = ImageDraw.Draw(out); x = 0
    for p, c in zip(panels, caps):
        out.paste(p, (x, 0)); d.text((x + 2, h + 4), c, fill=(0, 0, 0)); x += p.size[0] + pad
    out.save(os.path.join(OUT, fn)); print("wrote", fn)

w, h = 44, 30; half, rr = (15, 11, 11), 8.0
# 1. cel zones (the trap) vs the ramp coordinate (the fix), same boulder, same light
zc, ins, _ = T.boulder_zones(w, h, half, rr)
zp, _ = T.boulder_paint(w, h, half, rr)
n, inside, P, ground = T.raycast_boulder(w, h, np.array(half), rr, yaw=0.55, seed=1)
L = np.array([-0.9, 0.12, -0.35]); L /= np.linalg.norm(L); B = np.array([0.5, -0.8, -0.2]); B /= np.linalg.norm(B)
v, ndl = T.boulder_value(n, P, ground, half[1], L, B)
vm = np.zeros((h, w, 3), np.uint8); vm[:] = 245
q = np.clip((v - 1) / 5, 0, 1); vm[inside] = (np.stack([q, q, q], -1)[inside] * 235 + 10).astype(np.uint8)
lit_side = inside & (ndl > 0.15)
vm[inside & ~lit_side] = (vm[inside & ~lit_side] * np.array([0.75, 0.8, 1.15])).clip(0, 255).astype(np.uint8)
row([up(rock_img(zc, h, w)), up(vm), up(rock_img(zp, h, w))],
    ["trap: zones painted flat (cel shading)", "the ramp coordinate v (blue tint = shadow range)", "fix: v dithered, planes kept flat"],
    "fig_d_zones_vs_ramp.png")
# 2. four lights, same rock
lights = {"left": ((-0.9, 0.12, -0.35), (0.5, -0.8, -0.2)), "front": ((0.15, 0.15, -1.0), (0.0, -1.0, -0.3)),
          "right": ((0.9, 0.12, -0.35), (-0.5, -0.8, -0.2)), "back": ((0.2, 0.15, 1.0), (0.0, -1.0, -0.2))}
ps = []
for k, (Lk, Bk) in lights.items():
    z, _ = T.boulder_paint(w, h, half, rr, L=Lk, B=Bk)
    ps.append(up(rock_img(z, h, w), 6))
row(ps, list(lights), "fig_d_four_lights.png")
# 3. sizes on screen: what survives as the rock shrinks (10 m .. 40 m)
ps = []; caps = []
for s in (1.0, 0.55, 0.3, 0.16):
    hh = (np.array(half) * s); ww_, h_ = max(8, int(w * s) + 4), max(6, int(h * s) + 4)
    z, _ = T.boulder_paint(ww_, h_, tuple(hh), rr * s, crack=s > 0.5, grain=0.35 if s > 0.5 else 0.0)
    ps.append(up(rock_img(z, h_, ww_), 8)); caps.append(f"{int(2*hh[0])} px wide")
row(ps, caps, "fig_d_sizes.png")
