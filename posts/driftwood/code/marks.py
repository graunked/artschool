"""marks.py -- the wood's material, drawn as strokes on the log's surface.

Observed (Cliffs_and_driftwood, Driftwood_on_Gravel_bar, braided bar):
  * silvered driftwood is smooth and pale; its grain is a set of long, gently wavy dark lines
    running ALONG the axis, broken into dashes, a few px apart; they bunch toward the silhouette
    because the cylinder turns away (foreshortening);
  * checks are the few long splits along the grain: darker and longer than grain, often with a
    lighter lip on the sun side (the split's upper edge catches the light);
  * knots are eyes: a dark centre, grain lines deflecting around them (bulging past, converging
    behind);
  * fresh bark is the INVERSE: a dark body crossed by a braided network of lighter ridges
    (Ferrari's snow-forest trunks: lit ridge lines interwoven along the axis on dark bark).

Painted (Ferrari: narrow forms are clean 1-px runs; contrast lives inside the mark):
  * every mark is a curve on the 3-D surface -- theta(s) with a low-frequency wobble -- sampled at
    half-pixel spacing, projected, and rasterised to 1-px runs; hidden where the surface faces
    away or something is in front;
  * grain = one value step darker than the pixel's own form class;
  * check = two to three steps darker, with a 1-px lip one step lighter on its sun side;
  * knot = an elongated dark eye with a lit rim on the sun side; nearby grain bends round it;
  * density scales with the projected diameter: no grain below ~7 px, checks only down to ~4 px.
"""
import numpy as np
import geo

# one value step darker / lighter than a wood step (value ramp order 0<1<2<3<4<5<6<7)
DARKER = np.array([0, 0, 1, 1, 2, 4, 4, 6])
DARKER2 = np.array([0, 0, 0, 1, 1, 2, 2, 4])
LIGHTER = np.array([1, 2, 3, 4, 5, 6, 7, 7])
TIP = np.array([0, 1, 2, 3, 3, 5, 5, 6])      # a half step: the tapering end of a grain dash
DENSITY = np.array([0.2, 0.5, 0.85, 0.7, 0.95, 0.75, 0.5, 0.4])   # by the step under the dash


def surf_point(lg, s, th, r_scale=1.0):
    p, ax, r, (e1, e2) = lg.frame_at(s)
    n = np.cos(th) * e2 + np.sin(th) * e1
    return p + n * r * r_scale, n


def _visible(buf, cam, k, P, N, tol_px=1.2):
    x, y = cam.project(P)
    xi = np.floor(x).astype(int); yi = np.floor(y).astype(int)
    H, W = buf['obj'].shape
    ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    V = -cam.f
    ok &= (N @ V) > 0.05
    xi = np.clip(xi, 0, W - 1); yi = np.clip(yi, 0, H - 1)
    ok &= buf['obj'][yi, xi] == k
    t = (P - cam.look) @ cam.f + 60.0
    ok &= np.abs(buf['t'][yi, xi] - t) < tol_px / cam.scale + 0.02
    return xi, yi, ok


def _snap_runs(xi, yi, own, seg=5):
    """re-draw a projected dash as clean runs: vertices every ~seg px along it, each span a
    constant-run line (1:1, 2:1, 3:1, 4:1, flat) -- no zigzag diagonals."""
    from render import clean_line
    if len(xi) < 2:
        return xi, yi
    keep = np.concatenate([[True], (np.abs(np.diff(xi)) + np.abs(np.diff(yi))) > 0])
    xi, yi = xi[keep], yi[keep]
    if len(xi) < 2:
        return xi, yi
    idx = list(range(0, len(xi), seg))
    if idx[-1] != len(xi) - 1:
        idx.append(len(xi) - 1)
    X, Y = [], []
    for a, b in zip(idx[:-1], idx[1:]):
        lx, ly = clean_line(xi[a], yi[a], xi[b], yi[b])
        X.append(lx[:-1] if b != idx[-1] else lx); Y.append(ly[:-1] if b != idx[-1] else ly)
    X = np.concatenate(X); Y = np.concatenate(Y)
    H, W = own.shape
    ok = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
    X, Y = X[ok], Y[ok]
    ok = own[Y, X]
    return X[ok], Y[ok]


def _curve(lg, s0, s1, th0, wob_amp, wob_len, ph, step_m, knots=()):
    s = np.arange(s0, s1, step_m)
    th = th0 + wob_amp * np.sin(2 * np.pi * s / wob_len + ph) + 0.5 * wob_amp * np.sin(2 * np.pi * s / (wob_len * 0.37) + 2 * ph)
    # knots push grain around them: a smooth bump in theta, away from the knot centre
    for (ks, kth, kr_s, kr_th) in knots:
        dth = np.angle(np.exp(1j * (th - kth)))
        ds = (s - ks) / (kr_s * 3.0)
        w = np.exp(-ds * ds) * np.exp(-(dth / (kr_th * 3.0)) ** 2)
        th = th + np.sign(dth + 1e-6) * w * kr_th * 2.2
    return s, th


def wood_marks(cv, buf, logs, cam, light, sh, weather=1.0, seed=0, mat_ids=(0,)):
    """draw grain, checks and knots onto cv (in place)."""
    V = -cam.f
    lx = light.L @ cam.r; ly = -(light.L @ cam.u)
    sdx = int(np.sign(lx)) if abs(lx) > 0.35 else 0
    sdy = int(np.sign(ly)) if abs(ly) > 0.35 else 0
    H, W = cv.step.shape
    for k, lg in enumerate(logs):
        rng = np.random.default_rng(lg.seed * 7 + seed)
        dpx = 2 * lg.rad.mean() * cam.scale
        if dpx < 4:
            continue
        step_m = 0.45 / cam.scale
        own = (buf['obj'] == k)
        base = cv.step.copy()
        # the outline row is protected: marks never break the silhouette's clean run
        edge = own & ~(np.roll(own, 1, 0) & np.roll(own, -1, 0) & np.roll(own, 1, 1) & np.roll(own, -1, 1))
        base_edge = cv.step.copy()
        bark = lg.kind == 'bark'
        if bark:
            # the bark body sits two steps down: the voids between ridges are the body itself
            inner = own & ~edge
            cv.step[inner] = np.maximum(cv.step[inner] - 2, 0)
            base = cv.step.copy()
        # ---- knots come from the geometry (the outline bulges there); grain bends round them
        knots = list(lg.knots) if dpx >= 9 else []
        # ---- grain
        if dpx >= 7:
            n_lines = int(np.clip(np.pi * dpx / (2.6 if not bark else 2.2), 6, 80))
            for i in range(n_lines):
                th0 = (i + rng.uniform(-0.3, 0.3)) / n_lines * 2 * np.pi
                wob = rng.uniform(0.06, 0.18) if not bark else rng.uniform(0.25, 0.45)
                wl = (2 * lg.rad.mean() * rng.uniform(2.0, 6.0)) if not bark else lg.rad.mean() * rng.uniform(4, 8)
                ph = rng.uniform(0, 2 * np.pi)
                s = 0.0
                while s < lg.L:
                    dash = rng.uniform(6, 30) / cam.scale * (1.6 if bark else 1.0)
                    gap = rng.uniform(3, 14) / cam.scale * (0.5 if bark else 1.0) / max(weather, 0.3)
                    ss, th = _curve(lg, s, min(s + dash, lg.L), th0, wob, wl, ph, step_m, knots)
                    s += dash + gap
                    if len(ss) < 2:
                        continue
                    P = np.array([surf_point(lg, a, b)[0] for a, b in zip(ss, th)])
                    N = np.array([surf_point(lg, a, b)[1] for a, b in zip(ss, th)])
                    xi, yi, ok = _visible(buf, cam, k, P, N)
                    xi, yi = xi[ok], yi[ok]
                    if len(xi) == 0:
                        continue
                    xi, yi = _snap_runs(xi, yi, own & ~edge)
                    if len(xi) == 0:
                        continue
                    # form turned by mark density: grain shows most in the halftone and core,
                    # least in the light (Ferrari's grey trunk: the lit band stays clean)
                    keep = rng.random() < DENSITY[base[yi[len(yi) // 2], xi[len(xi) // 2]]]
                    if not keep:
                        continue
                    if bark:
                        # Ferrari's bark: bright ridges braided through a dark body, black voids
                        # between them (critic r6: contrast and voids are the language)
                        cv.step[yi, xi] = np.minimum(base[yi, xi] + 3, 7)
                    else:
                        # tapering ends: the first and last pixels of a dash are a half-step
                        cv.step[yi, xi] = DARKER[base[yi, xi]]
                        _, first = np.unique(np.stack([yi, xi], 1), axis=0, return_index=True)
                        order = np.sort(first)
                        if len(order) >= 4:
                            tips = np.concatenate([order[:1], order[-1:]]) if len(order) < 10 else np.concatenate([order[:2], order[-2:]])
                            cv.step[yi[tips], xi[tips]] = TIP[base[yi[tips], xi[tips]]]
        # ---- checks: long splits that wander a little, open to 2 px in their middle third, and
        # close to a half-step at both ends; a lighter lip on the sun side where lit
        if dpx >= 4 and not bark:
            nchk = rng.integers(1, 3) + int(weather > 0.7)
            for _ in range(nchk):
                th0 = rng.uniform(-1.6, 1.6)
                ln = lg.L * rng.uniform(0.2, 0.55)
                s0 = rng.uniform(0, lg.L - ln)
                ss, th = _curve(lg, s0, s0 + ln, th0, rng.uniform(0.1, 0.18), 2 * lg.rad.mean() * rng.uniform(1.5, 4),
                                rng.uniform(0, 6), step_m, ())
                P = np.array([surf_point(lg, a_, b_)[0] for a_, b_ in zip(ss, th)])
                N = np.array([surf_point(lg, a_, b_)[1] for a_, b_ in zip(ss, th)])
                xi, yi, ok = _visible(buf, cam, k, P, N)
                f = np.linspace(0, 1, len(ss))
                xi, yi, f = xi[ok], yi[ok], f[ok]
                if len(xi) == 0:
                    continue
                deep = (f > 0.12) & (f < 0.88)
                cv.step[yi[deep], xi[deep]] = DARKER2[base[yi[deep], xi[deep]]]
                cv.step[yi[~deep], xi[~deep]] = DARKER[base[yi[~deep], xi[~deep]]]
                chk = np.zeros((H, W), bool); chk[yi, xi] = True
                wide = (f > 0.35) & (f < 0.65) & (dpx >= 10)
                if wide.any():
                    y2, x2 = yi[wide] - sdy, xi[wide] - sdx     # opens on the anti-sun side
                    okw = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
                    y2, x2 = y2[okw], x2[okw]
                    okw = own[y2, x2] & ~edge[y2, x2]
                    cv.step[y2[okw], x2[okw]] = DARKER[base[y2[okw], x2[okw]]]
                    chk[y2[okw], x2[okw]] = True
                if (sdx or sdy) and not light.overcast:
                    ly2, lx2 = yi + sdy, xi + sdx
                    okl = (ly2 >= 0) & (ly2 < H) & (lx2 >= 0) & (lx2 < W)
                    ly2, lx2 = ly2[okl], lx2[okl]
                    okl = own[ly2, lx2] & (base[ly2, lx2] >= 4) & ~chk[ly2, lx2]
                    cv.step[ly2[okl], lx2[okl]] = LIGHTER[base[ly2[okl], lx2[okl]]]
        # ---- knots drawn last (they may sit on the outline: a knot bump is a real silhouette event)
        cv.step[edge] = base_edge[edge]
        for (ks, kth, kr_s, kr_th) in knots:
            pts = []
            for a in np.linspace(-1, 1, 9):
                for b in np.linspace(-1, 1, 9):
                    if a * a + b * b <= 1:
                        pts.append((ks + a * kr_s, kth + b * kr_th, a, b))
            P = np.array([surf_point(lg, s_, t_, 1.12)[0] for s_, t_, _, _ in pts])
            N = np.array([surf_point(lg, s_, t_)[1] for s_, t_, _, _ in pts])
            xi, yi, ok = _visible(buf, cam, k, P, N, tol_px=3.0)
            ab = np.array([(a, b) for _, _, a, b in pts])
            rr = np.hypot(ab[:, 0], ab[:, 1])
            for j in np.nonzero(ok)[0]:
                y, x = yi[j], xi[j]
                if rr[j] < 0.55:
                    cv.step[y, x] = 1 if not bark else 0
                elif rr[j] < 1.0:
                    cv.step[y, x] = DARKER[base[y, x]]
            # lit rim of the knot's swelling on the sun side
            if (sdx or sdy) and not light.overcast:
                cy_, cx_ = yi[ok], xi[ok]
                if len(cy_):
                    cy0, cx0 = int(np.median(cy_)), int(np.median(cx_))
                    for (y, x) in zip(cy_, cx_):
                        if (y - cy0) * sdy + (x - cx0) * sdx >= 1:
                            yy, xx = y + sdy, x + sdx
                            if 0 <= yy < H and 0 <= xx < W and own[yy, xx] and base[yy, xx] >= 4:
                                if cv.step[yy, xx] == base[yy, xx]:
                                    cv.step[yy, xx] = LIGHTER[base[yy, xx]]
