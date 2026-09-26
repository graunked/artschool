import sys, numpy as np, sp, backdrop, palette
tag = sys.argv[1] if len(sys.argv) > 1 else "a"
cells = []
for hour, band in [("golden", 1), ("noon", 2), ("dusk", 3)]:
    cv = sp.Canvas(240, 100)
    cv.mat[:] = sp.SKY
    cv.step[:] = np.clip(np.arange(100)[:, None] // 12, 0, 5)
    cv.mat[80:] = sp.GROUND; cv.step[80:] = 3
    backdrop.paint_rim(cv, 22, 80, band=band, sun_dx=1, seed=3)
    cells.append(sp.upscale(cv.render(palette.Palette("summer", hour)), 3))
sp.save(sp.hstack(cells, ["golden band1", "noon band2", "dusk band3"]), f"stage4/s4_rim_{tag}.png")
