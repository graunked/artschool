"""Lit share of the foliage on the sun side vs the far side of the stem, as the crown-normal
weight wc falls. The study's painter, hemlock mature, 0.12 m/px, sun from the left."""
import sys; sys.dont_write_bytecode = True
import figs, numpy as np, grow, conifer, paint
tr = grow.grow("hemlock", "mature", seed=2, mpp=0.12)
ims = []
for wc in [0.9, 0.5, 0.18, 0.0]:
    rgb, info = conifer.tree("hemlock", "mature", seed=2, mpp=0.12, W=160, H=420, tr=tr, P=dict(wc=wc))
    st, m = info["step"], info["G"]["mat"]
    fol = m == paint.FOL
    xs = np.arange(160)[None, :].repeat(420, 0)
    L = fol & (xs < 80); R = fol & (xs >= 80)
    print(f"wc={wc:4}: lit share sun side {((st>=4)&L).sum()/L.sum():.2f}   far side {((st>=4)&R).sum()/R.sum():.2f}")
    ims.append(rgb[20:330])
figs.panel(ims, ["wc 0.9", "wc 0.5", "wc 0.18 (final)", "wc 0"], k=2, name="t2_wc_sweep.png")
