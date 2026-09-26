"""Stage 1: one sagebrush in greyscale. Photo | ours 4x | ours 1x."""
import sys, numpy as np
from PIL import Image
import sp, sage

GREY = {
    sp.SAGE: [0.07, 0.15, 0.24, 0.43, 0.57, 0.70],
    sp.SAGESTEM: [0.04, 0.13, 0.25, 0.38],
    sp.GROUND: [0.18, 0.30, 0.52, 0.60, 0.66],
    sp.SKY: [0.86],
    sp.FLOWER: [0.5, 0.65],
    sp.SNOW: [0.8, 0.9, 0.97],
}


class Grey:
    def rgb(self, m, s, b):
        r = GREY.get(m, [0.86])
        v = r[min(s, len(r) - 1)]
        return (int(v * 255),) * 3


def bush_on_ground(p, seed, W=None, H=None):
    rng = np.random.default_rng(seed)
    cv, bx, by, _ = sage.paint_bush(p, rng)
    W = W or cv.w + 10
    H = H or cv.h + 6
    out = sp.Canvas(W, H)
    out.mat[:] = sp.GROUND
    rng2 = np.random.default_rng(seed + 100)
    out.step[:] = np.where(rng2.random((H, W)) < 0.12, 2, 3)
    gy = H - 5
    ox, oy = W // 2 - bx, gy - by
    # cast shadow: the bush mask, flattened onto the ground away from the sun, ground 2 steps down
    m = cv.mat > 0
    ys, xs = np.nonzero(m)
    sx = 1 if p["az"] < 90 else -1
    for y, x in zip(ys, xs):
        hgt = by - y
        X = x + ox + sx * int(hgt * 0.9) + sx * 2
        Y = gy - int(hgt * 0.22)
        for dx in (0, 1):
            if 0 <= Y < H and 0 <= X + dx < W and out.mat[Y, X + dx] == sp.GROUND:
                out.step[Y, X + dx] = min(out.step[Y, X + dx], 1)
    out.blit(cv, ox, oy)
    # contact: darkest line along the foot
    for x in range(W):
        col = np.nonzero(cv.mat[:, x - ox] > 0)[0] if 0 <= x - ox < cv.w else []
        if len(col) and col.max() + oy >= gy - 3:
            out.px(x, gy + 1, sp.GROUND, 0)
    return out


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "a"
    cells = []
    for seed in range(4):
        p = sage.SageParams(w=56, h=38)
        cv = bush_on_ground(p, seed)
        cells.append(sp.upscale(cv.render(Grey()), 4))
    sp.save(sp.grid(cells, 4, title="stage 1: one sagebrush, greyscale, 4x (seeds 0-3)"),
            f"stage1/s1_sage_{tag}.png")
    # photo (box-downsampled to the same scale, greyscale) | ours, 8x
    from PIL import Image
    ph = Image.open(sp.HERE + "/refs/sage_ds_2.png").convert("L")
    ph = np.array(ph)[26:, 264:]
    ph = np.stack([ph] * 3, -1)[:, :, :]
    ph = ph[::4, ::4]
    p = sage.SageParams(w=56, h=38)
    ours = bush_on_ground(p, 0).render(Grey())
    sp.save(sp.hstack([sp.upscale(ph, 8), sp.upscale(ours, 8)], ["photo, 64px box, grey, 8x", "ours 8x"]),
            f"stage1/s1_sage_{tag}_8x.png")
