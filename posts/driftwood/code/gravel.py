"""gravel.py -- the gravel bar under the logs.

Observed (braided_driftwood_bar.jpg, near and mid ground): the bar is SHINGLE -- flat plates of
grey-beige stone lying on their faces, packed and overlapping, each a light top face with a thin
dark lip on the side away from the sun (its own cast shadow on the stone below); between them
patches of smoother, darker sand/silt.

Painted (Ferrari's stone rule: light top, dark bottom row, flat ends; contrast inside the mark):
  * plates are world-space flat ellipses (so they foreshorten with the camera pitch), stacked by
    a random height; every pixel knows which plate it is on;
  * a plate's face is ONE pure step (5 lit, a few 4/6 by tilt/albedo) -- no dither on a plate;
  * the pixel(s) just past a plate's edge on the anti-sun side, not covered by a higher plate,
    are its shadow lip: step 2 (the mark's dark partner);
  * gaps and sand: step 3, a calm mid-dark field; sand patches (low-frequency) drop the plate
    density so the sand reads as a mass;
  * in a log's cast shadow everything drops by the family gap: faces 2 (3 on sun-tilted plates),
    lips 1, gaps 1.
Scale: plate size in px decides the glyph automatically (a 2-px plate is a light dash over a dark
pixel, a 6-px plate a lozenge with a lit rim).  Far away (< 1 px) it becomes a speckle of the
same three steps.
"""
import numpy as np
from paint import hash2


def _world_noise(u, v, seed):
    iu, iv = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    fu, fv = u - iu, v - iv
    fu = fu * fu * (3 - 2 * fu); fv = fv * fv * (3 - 2 * fv)
    a = hash2(iu, iv, seed); b = hash2(iu + 1, iv, seed)
    c = hash2(iu, iv + 1, seed); d = hash2(iu + 1, iv + 1, seed)
    return a * (1 - fu) * (1 - fv) + b * fu * (1 - fv) + c * (1 - fu) * fv + d * fu * fv


def plates(X, Y, stone_m, seed, cover=0.85, sand_scale=6.0, sand=0.35, aspect=(0.45, 0.85)):
    """per point: plate id (-1 none), its height, local (u,v) in the plate."""
    c = stone_m
    gx, gy = np.floor(X / c).astype(np.int64), np.floor(Y / c).astype(np.int64)
    pid = np.full(X.shape, -1, np.int64)
    ph = np.full(X.shape, -1.0)
    pu = np.zeros_like(X); pv = np.zeros_like(X)
    sandf = _world_noise(X / (c * sand_scale), Y / (c * sand_scale), seed + 77)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cx, cy = gx + dx, gy + dy
            exists = hash2(cx, cy, seed + 11) < cover * (1 - sand * np.clip((sandf - 0.45) * 3, 0, 1))
            jx = hash2(cx, cy, seed + 1); jy = hash2(cx, cy, seed + 2)
            ra = c * (0.38 + 0.4 * hash2(cx, cy, seed + 3))
            asp = aspect[0] + (aspect[1] - aspect[0]) * hash2(cx, cy, seed + 5)
            ang = hash2(cx, cy, seed + 4) * np.pi
            px = (cx + jx) * c; py = (cy + jy) * c
            vx, vy = X - px, Y - py
            ca, sa = np.cos(ang), np.sin(ang)
            u = (vx * ca + vy * sa) / ra
            v = (-vx * sa + vy * ca) / (ra * asp)
            ins = exists & (u * u + v * v < 1.0)
            h = hash2(cx, cy, seed + 6)
            m = ins & (h > ph)
            pid = np.where(m, (cx + 1000000) * 2000003 + (cy + 1000000), pid)
            ph = np.where(m, h, ph)
            pu = np.where(m, u, pu); pv = np.where(m, v, pv)
    return pid, ph, pu, pv


def _shift(a, dy, dx, fill):
    out = np.full_like(a, fill)
    H, W = a.shape
    ys = slice(max(0, dy), H + min(0, dy)); yd = slice(max(0, -dy), H + min(0, -dy))
    xs = slice(max(0, dx), W + min(0, dx)); xd = slice(max(0, -dx), W + min(0, -dx))
    out[yd, xd] = a[ys, xs]
    return out


def paint_gravel(buf, cam, light, sh, seed=0, stone_m=0.05, density=None, cover=0.7,
                 sand=0.35, gap_step=3, lip_step=3):
    P = buf['P']
    X, Y = P[..., 0], P[..., 1]
    pid, ph, pu, pv = plates(X, Y, stone_m, seed, cover=cover, sand=sand)
    on = pid >= 0
    t = hash2(pid, 3, seed + 9)
    # quiet ground (critic, rounds 3-10: the ground was louder than the logs): faces on ONE step
    # with a few a step up; gaps one step down; lips only where a plate is big enough to cast
    face = np.where(t > 0.8, 5, 4)
    step = np.where(on, face, gap_step)
    lx = light.L @ cam.r; ly = -(light.L @ cam.u)
    sdx = int(np.sign(lx)) if abs(lx) > 0.3 else 0
    sdy = int(np.sign(ly)) if abs(ly) > 0.3 else 0
    if sdy == 0 and sdx == 0:
        sdy = -1
    spx = stone_m * cam.scale
    # value at pixel p+ (toward the sun) / p- (away)
    tow_id = _shift(pid, sdy, sdx, -1); tow_h = _shift(ph, sdy, sdx, -1.0)
    if spx >= 4:
        away_id = _shift(pid, -sdy, -sdx, -1); away_h = _shift(ph, -sdy, -sdx, -1.0)
        rim = on & (tow_id != pid) & (tow_h < ph)
        step = np.where(rim, np.minimum(face + 1, 6), step)
    lip = (tow_id >= 0) & (tow_id != pid) & (tow_h > ph)
    step = np.where(lip, lip_step, step)
    if spx >= 7:
        t2_id = _shift(pid, 2 * sdy, 2 * sdx, -1); t2_h = _shift(ph, 2 * sdy, 2 * sdx, -1.0)
        lip2 = (t2_id >= 0) & (t2_id != pid) & (t2_h > ph) & ~on
        step = np.where(lip2, lip_step, step)
    if light.overcast:
        step = np.where(on, np.where(t < 0.22, 3, np.where(t > 0.88, 5, 4)), 3)
        step = np.where(lip, 2, step)
    shs = np.where(on, np.where(t > 0.88, 3, 2), 1)
    shs = np.where(lip, 1, shs)
    step = np.where(sh, shs, step)
    # stone variety: a plate's rock type by its own hash (river shingle is mixed): most beige,
    # a quarter blue-grey, a few rusty -- same steps, different ramps
    rt = hash2(pid, 5, seed + 21)
    small = spx < 3.5          # tiny plates: variety is noise at this size, keep it rare
    mat = np.where(on & (rt < (0.12 if small else 0.25)), 7,
                   np.where(on & (rt > (0.97 if small else 0.9)), 8, 1))
    paint_gravel.last_mat = mat
    return step.astype(int)
