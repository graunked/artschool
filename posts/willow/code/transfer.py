"""Stage 0 -> willow: carry the proven hedge copy (C2) onto the willow form, one change at a time.
T1: same technique, same palette (Ferrari's hedge ramps B lit / C shadow), willow form.
"""
import numpy as np, sys
from scipy import ndimage as ndi
from wpaint import *
from bush import WillowBush, wand_dirs
from foliage import mid_speckle, rim_arcs, fringe, stroke_marks, wand_tips
from lw import load
from PIL import Image

V09 = load("V09")
PAL = V09["pal"]
RB = [42, 41, 50, 40, 49]      # hedge lit ramp
RC = [3, 43, 42, 86]           # shadow ramp: the near darks of the same scene
SKYC = tuple(PAL[252]); GRAVC = (150, 146, 136)
W, H, GY = 128, 100, 88
G = (('dot', 4), ('h2', 3), ('v2', 1), ('d2', 2), ('a2', 2), ('h3', 1), ('cap', 1), ('hook', 1), ('d3', 2), ('s3', 2), ('arc', 2))
KB = dict(hi=(0.35, 0.8), clump=0.8, glyphs=G)
KC = dict(base=(0.2, 0.8), base_steps=(0, 1), dens=(0.03, 0.35), hi=(0.6, 1.0), mid_p=0.0)


def t1(seed, L, bw=90, bh=68, bkw=None, kb=None, kc=None, fr=dict(), arcs=dict(min_light=0.15, dens=(0.3, 1.0), r_sep=6),
       fam_split=2.9):
    rng = np.random.default_rng(seed)
    b = WillowBush(64, GY, bw, bh, rng, **(bkw or dict(nw=30, lobe_r=(0.07, 0.12), lobe_gap=0.7, skirt_n=4)))
    F = b.fields(W, H, L)
    x = np.nan_to_num(F["x"]); m = F["mask"]
    lit = m & (x >= fam_split); sh = m & ~lit
    ll = np.clip((x - 3) / 2.4, 0, 1); ls = np.clip(x / 2.45, 0, 1)
    SB = mid_speckle(None, ll, rng, top=4, mask=lit, **(KB | (kb or {})))
    SB = rim_arcs(SB, lit, ll, rng, 4, **arcs)
    SC = mid_speckle(None, ls, rng, top=3, mask=sh, **(KC | (kc or {})))
    region = np.where(lit, 0, 1)
    S = np.where(lit, SB, SC)
    if fr is not None:
        SB2, MB = fringe(S, lit, ll, rng, 4, **fr)
        SC2, MC = fringe(S, sh, ls * 0.3, rng, 3, **fr)
        newB = MB & ~m; newC = MC & ~m & ~newB
        region = np.where(newB, 0, np.where(newC, 1, region))
        S = np.where(region == 0, SB2, SC2)
        m = (MB | MC); m[GY:] = False
    img = np.zeros((H, W, 3), np.uint8); img[:GY] = SKYC; img[GY:] = GRAVC
    for r, ramp in ((0, RB), (1, RC)):
        for s, i in enumerate(ramp):
            img[m & (region == r) & (S == s)] = PAL[i]
    img[GY - 1][m[GY - 1]] = PAL[RC[0]] // 2
    return img, b, F


def hedge_crop():
    im = PAL[V09["idx"]]
    return im[222:282, 265:395]


if __name__ == "__main__":
    img, b, F = t1(1, sun(30, 45))
    img2, b, F = t1(2, sun(30, 45))
    ph = np.array(Image.open('refs/crop_nzbush.png').convert('RGB').resize((480, 390)))
    save(np.concatenate([ph, upscale(img, 4)[5:395, :], upscale(img2, 4)[5:395, :]], 1), 'stage0/t1_whole.png')
    # 8x: willow detail beside Ferrari hedge detail
    a = upscale(hedge_crop()[0:40, 40:100], 8); bb = upscale(img[25:65, 30:90], 8)
    save(side_by_side([a, bb], ["Ferrari hedge (V09) 8x", "T1: willow form, hedge technique 8x"], height=320), 'stage0/t1_8x.png')


def t2(seed, L, bw=90, bh=68, bkw=None, ks=None, kc=None, fr=dict(), arcs=None, tips=dict(p=0.25, lens=(1, 4)),
       fam_split=2.9, speck_mix=0.0):
    """T2: the lit family's marks become strokes along the wands (leaves on upright twigs);
    shadow family keeps the hedge speckle. Wand tips run out of the crest."""
    rng = np.random.default_rng(seed)
    b = WillowBush(64, GY, bw, bh, rng, **(bkw or dict(nw=30, lobe_r=(0.07, 0.12), lobe_gap=0.7, skirt_n=4)))
    F = b.fields(W, H, L)
    x = np.nan_to_num(F["x"]); m = F["mask"]
    dx, dy = wand_dirs(b, F)
    lit = m & (x >= fam_split); sh = m & ~lit
    ll = np.clip((x - 3) / 2.4, 0, 1); ls = np.clip(x / 2.45, 0, 1)
    lsd = (L[0], -L[1])
    k = dict(top=4, base_steps=(0, 1), base=(0.3, 0.7), dens=(0.25, 0.9), hi=(0.35, 0.8), lens=(2, 3),
             splay=30, pair_p=0.3, gap_p=0.4, mid_p=0.08, spacing=1.7, clump=0.7)
    SB = stroke_marks(lit, ll, dx, dy, rng, light_side=lsd, **(k | (ks or {})))
    SB = np.where(lit, np.maximum(SB, 0), 0)
    if arcs:
        SB = rim_arcs(SB, lit, ll, rng, 4, **arcs)
    SC = mid_speckle(None, ls, rng, top=3, mask=sh, **(KC | (kc or {})))
    region = np.where(lit, 0, 1)
    S = np.where(lit, SB, SC)
    if tips is not None:
        lf = np.where(lit, 0.4 + 0.6 * ll, 0.0)
        S2, M2 = wand_tips(S, m, dx, dy, lf, rng, 4, lo=1, **tips)
        new = M2 & ~m
        region = np.where(new, np.where(ndi.grey_dilation(lit, size=3), 0, 1), region)
        S = S2; m0 = m; m = M2
    if fr is not None:
        litm = m & (region == 0); shm = m & (region == 1)
        SB2, MB = fringe(S, litm, np.where(litm, 0.4 + 0.6 * ll, 0), rng, 4, **fr)
        SC2, MC = fringe(S, shm, ls * 0.3, rng, 3, **fr)
        newB = MB & ~m; newC = MC & ~m & ~newB
        region = np.where(newB, 0, np.where(newC, 1, region))
        S = np.where(region == 0, SB2, SC2)
        m = (MB | MC)
    m[GY:] = False
    img = np.zeros((H, W, 3), np.uint8); img[:GY] = SKYC; img[GY:] = GRAVC
    for r, ramp in ((0, RB), (1, RC)):
        for s, i in enumerate(ramp):
            img[m & (region == r) & (S == s)] = PAL[i]
    img[GY - 1][m[GY - 1]] = PAL[RC[0]] // 2
    return img, b, F


def run_t2(tag="t2", **kw):
    img, b, F = t2(1, sun(30, 45), **kw)
    img2, b, F = t2(2, sun(30, 45), **kw)
    ph = np.array(Image.open('refs/crop_nzbush.png').convert('RGB').resize((480, 390)))
    save(np.concatenate([ph, upscale(img, 4)[5:395, :], upscale(img2, 4)[5:395, :]], 1), f'stage0/{tag}_whole.png')
    a = upscale(hedge_crop()[0:40, 40:100], 8); bb = upscale(img[25:65, 30:90], 8)
    save(side_by_side([a, bb], ["Ferrari hedge (V09) 8x", f"{tag} 8x"], height=320), f'stage0/{tag}_8x.png')
