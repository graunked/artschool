"""Stage 0: master copies of Ferrari's leafy masses (Living Worlds 'Deep Forest', V09).
C1: the sunlit foreground bush (near-scale leaf glyphs).  C2: the middle-distance hedge band."""
import sys, os
import numpy as np
from lw import load, lum
from copyharness import decompose, sketch, render, metrics, compare_sheet, rs_stats
from foliage import near_leaves, near_leaves2

OUT = "stage0"
os.makedirs(OUT, exist_ok=True)
V09 = load("V09")


def copy_c1(seed=1, tag="", **kw):
    y0, y1, x0, x1 = 322, 380, 0, 140
    idx = V09["idx"][y0:y1, x0:x1]; pal = V09["pal"]
    ramps = [[13, 11, 10, 9], [26, 47, 57, 45], [26, 59, 2, 25]]
    R, S, table = decompose(idx, pal, ramps)
    region, light = sketch(R, S, len(ramps), 3, sigma=kw.pop("sigma", 2.5))
    ref = render(R, S, table)
    rng = np.random.default_rng(seed)
    fn = kw.pop("fn", near_leaves2)
    S2 = fn(region, light, rng, **kw)
    cp = render(region, S2, table)
    m = metrics(cp, ref)
    print('ref ', rs_stats(R, S, 3)); print('copy', rs_stats(region, S2, 3))
    compare_sheet(cp, ref, f"{OUT}/c1_fgbush{tag}.png", detail=(40, 8, 60, 40), label=str(m))
    return m


if __name__ == "__main__":
    print(copy_c1())


def copy_c2(seed=1, tag="", fnB=None, kwB=None, kwC=None, fringe=None, sigma=2.0, detail=(40, 0, 60, 36), arcs=None):
    """The middle-distance hedge: B (main band) and C (far, bluer) ramps generated, everything
    else (sky, pale far trunks, dark trunks) kept from the reference as the setting."""
    from scipy import ndimage as ndi
    from foliage import near_sprays
    y0, y1, x0, x1 = 222, 282, 265, 395
    idx = V09["idx"][y0:y1, x0:x1]; pal = V09["pal"]
    ramps = [[42, 41, 50, 40, 49], [86, 83, 84, 85]]
    known = np.isin(idx, sum(ramps, []))
    R, S, table = decompose(idx, pal, ramps)
    ref = pal[idx].copy()
    # sketch: smoothed foliage silhouette and region, light field
    fol = ndi.uniform_filter(known.astype(float), 5) > 0.5
    region, light = sketch(np.where(known, R, 0), np.where(known, S, 0), 2, 4, sigma=sigma)
    light = np.where(known, ndi.gaussian_filter(np.where(known, S / 4.0, 0), sigma) /
                     np.maximum(ndi.gaussian_filter(known.astype(float), sigma), 1e-3), 0)
    light = ndi.grey_dilation(light, size=3) * (1 - fol) + light * fol
    lo, hi_ = np.percentile(light[fol], [5, 97]); light = np.clip((light - lo) / (hi_ - lo), 0, 1)
    # background: reference pixels outside foliage; under the old foliage use nearest background
    bgm = ~known
    ii = ndi.distance_transform_edt(~bgm, return_distances=False, return_indices=True)
    bg = ref[ii[0], ii[1]]
    rng = np.random.default_rng(seed)
    fnB = fnB or near_sprays
    SB = fnB(region, light, rng, top=4, mask=fol, **(kwB or {}))
    SC = fnB(region, light, rng, top=3, mask=fol, **(kwC or kwB or {}))
    if arcs is not None:
        from foliage import rim_arcs
        SB = rim_arcs(SB, fol & (region == 0), light, rng, 4, **arcs)
    Sg = np.where(region == 0, SB, SC)
    if fringe is not None:
        from foliage import fringe as fr
        SgB, MB = fr(Sg, fol & (region == 0), light, rng, 4, **fringe)
        SgC, MC = fr(Sg, fol & (region == 1), light, rng, 3, **fringe)
        newB = MB & ~fol; newC = MC & ~fol
        region = np.where(newB, 0, np.where(newC, 1, region))
        Sg = np.where(region == 0, SgB, SgC)
        fol = MB | MC
    cp = bg.copy()
    fl = render(region, Sg, table)
    cp[fol] = fl[fol]
    m = metrics(cp, ref)
    compare_sheet(cp, ref, f"{OUT}/c2_hedge{tag}.png", detail=detail, label=str(m), k=4)
    return m, cp, ref


def copy_c2c(seed=1, tag="", kB=None, kC=None, sigma=2.0, detail=(40, 0, 60, 36), k=4, base_fill=True):
    """C2 with the clump grammar."""
    from scipy import ndimage as ndi
    from foliage import clump_foliage
    y0, y1, x0, x1 = 222, 282, 265, 395
    idx = V09["idx"][y0:y1, x0:x1]; pal = V09["pal"]
    ramps = [[42, 41, 50, 40, 49], [86, 83, 84, 85]]
    known = np.isin(idx, sum(ramps, []))
    R, S, table = decompose(idx, pal, ramps)
    ref = pal[idx].copy()
    fol = ndi.uniform_filter(known.astype(float), 5) > 0.5
    region, _ = sketch(np.where(known, R, 0), np.where(known, S, 0), 2, 4, sigma=sigma)
    light = ndi.gaussian_filter(np.where(known, S / 4.0, 0), sigma) / np.maximum(ndi.gaussian_filter(known.astype(float), sigma), 1e-3)
    lo, hi_ = np.percentile(light[fol], [5, 97]); light = np.clip((light - lo) / (hi_ - lo), 0, 1)
    bgm = ~known
    ii = ndi.distance_transform_edt(~bgm, return_distances=False, return_indices=True)
    bg = ref[ii[0], ii[1]]
    rng = np.random.default_rng(seed)
    Sg = -np.ones(idx.shape, int); Mg = np.zeros(idx.shape, bool); Rg = np.zeros(idx.shape, int)
    # fill under the clumps with the dark base so no background shows inside the mass
    if base_fill:
        Sg[fol] = 0; Mg[fol] = True; Rg[fol] = region[fol]
    for r_, kk, top in ((1, kC, 3), (0, kB, 4)):
        mm = fol & (region == r_)
        Sx, Mx = clump_foliage(mm, light, rng, top, **(kk or {}))
        Sg = np.where(Mx, Sx, Sg); Rg = np.where(Mx, r_, Rg); Mg |= Mx
    cp = bg.copy()
    fl = render(Rg, np.maximum(Sg, 0), table)
    cp[Mg] = fl[Mg]
    m = metrics(cp, ref)
    compare_sheet(cp, ref, f"{OUT}/c2c_hedge{tag}.png", detail=detail, label=str(m), k=k)
    return m, cp, ref


def copy_c2f(seed=1, tag="", kB=None, kC=None, mB=None, mC=None, sigma=2.0, detail=(40, 0, 60, 36), k=4, arcs=None):
    """C2: the clump grammar lays out a clump light field; the hedge speckle (dark base,
    lit marks) is drawn from it."""
    from scipy import ndimage as ndi
    from foliage import clump_foliage, mid_speckle, rim_arcs
    y0, y1, x0, x1 = 222, 282, 265, 395
    idx = V09["idx"][y0:y1, x0:x1]; pal = V09["pal"]
    ramps = [[42, 41, 50, 40, 49], [86, 83, 84, 85]]
    known = np.isin(idx, sum(ramps, []))
    R, S, table = decompose(idx, pal, ramps)
    ref = pal[idx].copy()
    fol = ndi.uniform_filter(known.astype(float), 5) > 0.5
    skyish = lum(pal[idx]) > 140
    fol &= known | skyish       # trunks and other things stay the setting
    region, _ = sketch(np.where(known, R, 0), np.where(known, S, 0), 2, 4, sigma=sigma)
    light = ndi.gaussian_filter(np.where(known, S / 4.0, 0), sigma) / np.maximum(ndi.gaussian_filter(known.astype(float), sigma), 1e-3)
    lo, hi_ = np.percentile(light[fol], [5, 97]); light = np.clip((light - lo) / (hi_ - lo), 0, 1)
    ii = ndi.distance_transform_edt(known, return_distances=False, return_indices=True)
    bg = ref[ii[0], ii[1]]
    rng = np.random.default_rng(seed)
    Sg = np.zeros(idx.shape, int); Mg = np.zeros(idx.shape, bool); Rg = np.zeros(idx.shape, int)
    for r_, kk, mk, top in ((1, kC, mC, 3), (0, kB, mB, 4)):
        mm = fol & (region == r_)
        CL = np.full(idx.shape, -1.0)
        _, Mx = clump_foliage(mm, light, rng, top, field=CL, **(kk or {}))
        Mx |= mm
        CL = np.where(CL < 0, light * 0.5, CL)
        Sx = mid_speckle(None, CL, rng, top=top, mask=Mx, **(mk or {}))
        if arcs and r_ == 0:
            Sx = rim_arcs(Sx, Mx, CL, rng, top, **arcs)
        # the crest catches the light: open-above pixels take the lit steps, caps and specks above
        from foliage import fringe
        Sx2, Mx2 = fringe(Sx, Mx, np.clip(CL * 1.3, 0, 1), rng, top, hole_p=0.12, bump_p=0.35, bump_w=(1, 3), peak_p=0.4, rim_p=1.0)
        new = Mx2 & ~Mx & ~known & ~(Rg == (1 - r_)) if False else Mx2 & ~Mg
        Sx = np.where(Mx2, Sx2, Sx); Mx = Mx2
        Sg = np.where(Mx, Sx, Sg); Rg = np.where(Mx, r_, Rg); Mg |= Mx
    cp = bg.copy()
    fl = render(Rg, Sg, table)
    cp[Mg] = fl[Mg]
    m = metrics(cp, ref)
    compare_sheet(cp, ref, f"{OUT}/c2f_hedge{tag}.png", detail=detail, label=str(m), k=k)
    return m, cp, ref


def copy_c1f(seed=1, tag="", kc=None, kl=None, detail=(40, 8, 60, 40)):
    """C1 with the clump grammar: clumps (6-14 px) lay out a light field (lit upper-left cap,
    dark underside); leaf lozenges are drawn from it."""
    from foliage import clump_foliage, near_leaves2
    y0, y1, x0, x1 = 322, 380, 0, 140
    idx = V09["idx"][y0:y1, x0:x1]; pal = V09["pal"]
    ramps = [[13, 11, 10, 9], [26, 47, 57, 45], [26, 59, 2, 25]]
    R, S, table = decompose(idx, pal, ramps)
    region, light = sketch(R, S, len(ramps), 3, sigma=2.5)
    lo, hi_ = np.percentile(light, [3, 97]); light = np.clip((light - lo) / (hi_ - lo), 0, 1)
    ref = render(R, S, table)
    rng = np.random.default_rng(seed)
    CL = np.full(idx.shape, -1.0)
    clump_foliage(np.ones(idx.shape, bool), light, rng, 3, field=CL, **(dict(r=(3.0, 7.0), vo=-0.3, vg=2.4, vl=1.6, vb=2.4) | (kc or {})))
    CL = np.where(CL < 0, light * 0.4, CL)
    S2 = near_leaves2(region, CL, rng, **(dict(bright=0.6, sd=0.4, dens=(0.05, 1.0), base1=(0.0, 0.3)) | (kl or {})))
    cp = render(region, S2, table)
    m = metrics(cp, ref)
    compare_sheet(cp, ref, f"{OUT}/c1f_fgbush{tag}.png", detail=detail, label=str(m))
    return m
