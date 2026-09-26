"""The willow bush as a form: geometry (lobes on a dome shell, stems from a root crown)
and the form field (normals, occlusion, the two-family value plan).

A bush is built in pixels at its final size. Geometry is expressed relative to its ground
contact (cx, gy), width W and height H (pixels).
"""
import numpy as np
from scipy import ndimage as ndi
from wpaint import norm3, value_noise


class Bush:
    def __init__(self, cx, gy, W, H, rng, lift=0.28, nlobes=None, lobe=(0.10, 0.20),
                 lean=0.0, flat=1.0, skirt=0.5, arch_w=0.32):
        """lift: how high the foliage underside arches above the ground in the middle (fraction of H),
        revealing the stem crown.  lean: shifts the crown sideways with height (fraction of W).
        flat: >1 flattens the dome top.  skirt: how far the lower lobes spread out and droop."""
        self.cx, self.gy, self.W, self.H = cx, gy, W, H
        self.rng = rng
        self.lift, self.lean, self.arch_w = lift, lean, arch_w
        self.nsig = max(1.0, W * 0.02)
        self.crease_cross, self.ao_lit = 0.55, 1.8
        self.base_fill = 0.0
        n = nlobes or int(10 + 0.18 * W)
        L = []
        # shell lobes: angle over the upper half-ellipse, radius near 1 (the visible outer leaf shell)
        for i in range(n):
            phi = rng.uniform(0.02, np.pi - 0.02)
            rho = rng.uniform(0, 1) ** 0.4
            u = np.cos(phi) * rho
            v = np.sin(phi) * rho
            v = v ** flat if flat != 1 else v
            r = W * rng.uniform(*lobe)
            L.append((u, v, r))
        # skirt lobes: sit low at both sides, drooping toward the ground
        for side in (-1, 1):
            for k in range(int(1 + 2 * skirt)):
                u = side * rng.uniform(0.62, 0.95)
                v = rng.uniform(0.08, 0.35)
                r = W * rng.uniform(lobe[0], lobe[1] * 0.9)
                L.append((u, v, r))
        self.lobes = []
        for (u, v, r) in L:
            x = cx + u * (W / 2 - r * 0.6) + lean * W * v
            y = gy - (H * lift * 0.4) - v * (H - r * 0.9 - H * lift * 0.4)
            if v < 0.35 and abs(u) > 0.55:
                y = min(gy - r * 0.55, y + r * 0.3)   # skirt lobes droop to the gravel
            rho = min(1.0, np.hypot(u, v))
            z = np.sqrt(max(0.0, 1 - rho ** 2)) * W * 0.32 + rng.uniform(0, 0.08) * W
            self.lobes.append(dict(x=x, y=y, r=r, z=z, u=u, v=v))
        # root crown: a short line of stem bases at the ground
        self.root = (cx + lean * W * 0.1, gy + 0.10 * H)

    # ------------------------------------------------------------ fields
    def fields(self, w, h, L, ground_bounce=0.8, env_mix=0.7, term=0.10, shadow_len=None, cast=False):
        """Per-pixel fields over a w x h canvas: mask, height z, normal n, occlusion occ,
        lit-family flag, continuous step x in [0,5.4]."""
        L = np.asarray(L, float)
        yy, xx = np.mgrid[0:h, 0:w].astype(float)
        zmap = np.full((h, w), -1e9)
        own = np.full((h, w), -1, int)
        nx = np.zeros((h, w)); ny = np.zeros((h, w))
        for i, lb in enumerate(self.lobes):
            ax, ay = lb["r"] * 1.15, lb["r"]           # lobes slightly wider than tall
            ux, uy = (xx - lb["x"]) / ax, (yy - lb["y"]) / ay
            d2 = ux ** 2 + uy ** 2
            inside = d2 < 1
            zz = lb["z"] + lb["r"] * np.sqrt(np.clip(1 - d2, 0, 1))
            upd = inside & (zz > zmap)
            zmap[upd] = zz[upd]; own[upd] = i
            nx[upd] = ux[upd]; ny[upd] = -uy[upd]
        mask = own >= 0
        nx0, ny0 = nx.copy(), ny.copy()
        dxa = (xx - self.cx) / (self.W * self.arch_w)
        arch = self.gy - self.lift * self.H * np.clip(1 - dxa ** 2, 0, 1)
        mask &= (yy < arch) & ((yy < self.gy) | (not getattr(self, 'clip_ground', True)))
        mk = own >= 0
        wsum = np.maximum(ndi.gaussian_filter(mk.astype(float), self.nsig), 1e-3)
        nx = ndi.gaussian_filter(nx, self.nsig) / wsum; ny = ndi.gaussian_filter(ny, self.nsig) / wsum
        nzl = np.sqrt(np.clip(1 - nx ** 2 - ny ** 2, 0.0, 1))
        nl = norm3(np.stack([nx, ny, nzl + 0.05], -1))
        # the big form: an egg fitted to the actual mass (top lit, underside turning down)
        ys_, xs_ = np.nonzero(mask)
        if len(xs_):
            ex0, ex1, ey0, ey1 = xs_.min(), xs_.max(), ys_.min(), ys_.max()
        else:
            ex0, ex1, ey0, ey1 = 0, w, 0, h
        ecx, ecy = (ex0 + ex1) / 2, ey0 + (ey1 - ey0) * self.egg_c
        ea, eb = max((ex1 - ex0) / 2, 1), max((ey1 - ey0) * max(self.egg_c, 1 - self.egg_c), 1)
        u = (xx - ecx) / ea
        v = (ecy - yy) / eb
        rr = np.clip(u ** 2 + v ** 2, 0, 0.97)
        ne = norm3(np.stack([u, v, np.sqrt(1 - rr)], -1))
        n = norm3(env_mix * ne + (1 - env_mix) * nl)
        cm = getattr(self, "clump_mix", 0.0)
        if cm > 0:
            cid = np.full((h, w), -1)
            for i, lb in enumerate(self.lobes):
                cid[own == i] = lb.get("clump", -1)
            nc = ne.copy()
            for c in range(getattr(self, "nclumps", 0)):
                mm = (cid == c) & mask
                if mm.sum() < 6:
                    continue
                yy_, xx_ = np.nonzero(mm)
                a_ = max((xx_.max() - xx_.min()) / 2, 1); b_ = max((yy_.max() - yy_.min()) / 2, 1)
                cx_, cy_ = (xx_.max() + xx_.min()) / 2, (yy_.max() + yy_.min()) / 2 + b_ * 0.1
                uu = (xx - cx_) / a_; vv = (cy_ - yy) / b_
                r2 = np.clip(uu ** 2 + vv ** 2, 0, 0.97)
                ncl = norm3(np.stack([uu, vv, np.sqrt(1 - r2)], -1))
                nc[mm] = ncl[mm]
            wts = np.maximum(ndi.gaussian_filter(mask.astype(float), 1.2), 1e-3)
            for k in range(3):
                nc[..., k] = ndi.gaussian_filter(nc[..., k] * mask, 1.2) / wts
            n = norm3((1 - cm) * n + cm * norm3(nc))
        z = np.where(mask, zmap, -1e9)
        # self shadow: march the height field toward the sun
        lxy = np.hypot(L[0], L[1])
        shadow = np.zeros((h, w), bool)
        if cast and lxy > 1e-3:
            sx, sy = L[0] / lxy, -L[1] / lxy
            rate = L[2] / lxy if L[2] > 0 else -0.3
            T = int(shadow_len or self.W * 0.5)
            zb = np.where(mask, ndi.gaussian_filter(np.where(mask, zmap, 0), 1.5) /
                          np.maximum(ndi.gaussian_filter(mask.astype(float), 1.5), 1e-3), -1e9)
            for t in range(1, T, 1):
                zs = ndi.shift(zb, (-sy * t, -sx * t), order=1, cval=-1e9)
                shadow |= zs > zb + t * rate + 1.0
        shadow &= mask
        shadow = ndi.binary_closing(ndi.binary_opening(shadow, np.ones((3, 3))), np.ones((3, 3))) & mask
        # ambient occlusion: pixels under a taller neighbour (lobe tucks) + the underside of the bush
        k = max(3, int(self.W * 0.08))
        zmax = ndi.maximum_filter(z, size=k)
        ao = np.clip((zmax - z - self.W * 0.03) / (self.W * 0.2), 0, 1)
        ao[~mask] = 0
        under = np.clip((yy - (self.gy - 0.5 * self.H)) / (0.5 * self.H), 0, 1) ** 1.5
        occ = np.clip(0.9 * ao + 0.6 * under * (1 - n[..., 1].clip(0, 1)), 0, 1)
        d = n @ L
        sky = 0.5 + 0.5 * n[..., 1]
        bounce = np.clip(-n[..., 1], 0, 1) * ground_bounce + 0.3 * np.clip(-d, 0, 1)
        # lobe caps: a lobe whose own top faces the sun catches light even near/past the big terminator
        nz0 = np.sqrt(np.clip(1 - nx0 ** 2 - ny0 ** 2, 0, 1))
        dl = nx0 * L[0] + ny0 * L[1] + nz0 * L[2]
        cap = (dl > self.cap_t) & (d > term - self.cap_reach) & mask & (ao < 0.5)
        d = np.where(cap & (d <= term), term + 0.02 + 0.3 * (dl - self.cap_t), d)
        lit = (d > term) & mask & ~shadow & ~(ao > self.crease_cross)
        x_lit = 3.0 + 2.4 * np.clip((d - term) / (1 - term), 0, 1) - self.ao_lit * occ
        x_lit = np.clip(x_lit, 3.0, 5.4)
        x_sh = 1.0 + 1.0 * bounce + 0.6 * (sky - 0.5) - 1.8 * occ
        x_sh = np.clip(x_sh, 0.0, 2.45)
        x = np.where(lit, x_lit, x_sh)
        # the dark interior under the canopy: columns within the crown footprint, from the lowest
        # foliage down to the gravel, are the shadowed hollow where the stems stand
        interior = np.zeros_like(mask)
        if self.base_fill > 0:
            half = self.W * self.base_fill
            for xc in range(w):
                if abs(xc - self.cx) > half:
                    continue
                col = np.nonzero(mask[:, xc])[0]
                if len(col):
                    top = col.min()
                    interior[top:int(self.gy), xc] = ~mask[top:int(self.gy), xc]
        x = np.where(interior, 0.35, x)
        mask = mask | interior
        # the skirt: outer wands droop to the gravel, so no foliage shelf hangs over bare ground.
        # Columns whose foliage stops well above the ground get a drooping curtain down to it.
        curtain = np.zeros_like(mask)
        if getattr(self, "curtain", 0):
            cols = np.nonzero(mask.any(0))[0]
            for xc in cols:
                col = np.nonzero(mask[:, xc])[0]
                bot = col.max()
                # the hem rises toward the sides, so the skirt's outline curves down to the ground
                # instead of dropping as a vertical wall
                ax = abs(xc - self.cx)
                hem = self.H * 0.4 * np.clip((ax - 0.28 * self.W) / (0.22 * self.W), 0, 1) ** 1.3
                ybot = int(self.gy - hem)
                if ybot - bot > 1:
                    curtain[bot + 1:ybot, xc] = True
            # ragged lower hem where the curtain ends: leave the arch/hollow in the middle alone
            hollow = np.abs(xx - self.cx) < self.W * 0.22
            curtain &= ~hollow
            x = np.where(curtain, 0.9 + 0.6 * np.clip((self.gy - yy) / (0.3 * self.H), 0, 1), x)
            mask = mask | curtain
        edge = ndi.distance_transform_edt(mask)
        # back light: thin leaves glow at the silhouette (rim), strongest where the edge faces the sun
        if L[2] < -0.05 and self.rim > 0:
            gy2, gx2 = np.gradient(ndi.gaussian_filter(mask.astype(float), 2.0))
            onx, ony = -gx2, gy2                       # outward normal, y up
            onn = np.hypot(onx, ony) + 1e-9
            face = np.clip((onx * L[0] + ony * L[1]) / onn / max(np.hypot(L[0], L[1]), 1e-6), 0, 1)
            rn = value_noise((h, w), 2.0, self.rng); rn = (rn - rn.min()) / (np.ptp(rn) + 1e-9)
            rim = np.clip(1 - (edge - 1) / self.rim_w, 0, 1) * (-L[2]) * face ** 1.5 * self.rim * (0.5 + rn)
            rl = mask & (rim > 0.22) & ~interior
            x = np.where(rl, np.maximum(x, 3.0 + 2.2 * np.clip(rim, 0, 1)), x)
            lit = lit | rl
            # transmitted light: willow leaves are thin, so with the sun behind the whole crown glows
            # wherever it is thin: strongest near the silhouette and in the upper crown, fading into
            # the thick core and the base. (Photo: backlit willow = pale glowing mass, dark twigs.)
            back = np.clip((-L[2] - 0.15) / 0.6, 0, 1) * self.glow
            if back > 0:
                lam = max(4.0, self.W * 0.24)
                upper = np.clip((self.gy - yy) / self.H, 0, 1)
                t = np.exp(-(edge - 1) / lam) * (0.5 + 0.5 * upper) + 0.45 * upper ** 1.5
                t = t * (0.7 + 0.6 * rn) * back
                gl = mask & (t > 0.22) & ~interior
                x = np.where(gl, np.maximum(x, 3.0 + 2.4 * np.clip((t - 0.22) / 0.45, 0, 1)), x)
                lit = lit | gl
        x[~mask] = np.nan
        lit = lit & ~interior
        return dict(mask=mask, z=zmap, n=n, occ=occ, ao=ao, lit=lit, shadow=shadow, x=x, own=own, interior=interior,
                    d=d, edge=edge, under=under)


def dither_steps(x, mask, rng, mode="clump", flat_snap=0.0, grain=1.0):
    """Continuous step coordinate -> integer steps, mixing only adjacent steps.
    mode 'check': ordered 2x2; 'clump': noise threshold (organic, clumped in light);
    'random': white noise."""
    h, w = x.shape
    if mode == "check":
        t = (np.indices((h, w)).sum(0) % 2) * 0.5 + 0.25
    elif mode == "random":
        t = rng.random((h, w))
    else:
        n1 = value_noise((h, w), 1.6 * grain, rng)
        t = np.clip(0.55 * rng.random((h, w)) + 0.45 * (n1 - n1.min()) / (np.ptp(n1) + 1e-9), 0, 1)
    fl = np.floor(np.nan_to_num(x))
    fr = np.nan_to_num(x) - fl
    s = fl + (fr > t)
    s = np.where(mask, s, 0)
    return s.astype(int)


class WillowBush(Bush):
    """A willow: a fan of wands from a root crown, each arching out and drooping at the tip.
    Leaf clusters (lobes) sit along the outer part of each wand, so the silhouette is the sum of
    wand tips, and stems show where clusters are sparse (low down, between wands)."""

    def __init__(self, cx, gy, W, H, rng, nw=None, spread=75, droop=0.45, leafy_from=0.12,
                 lobe_r=(0.055, 0.11), lobe_gap=0.9, lift=0.0, lean=0.0, crown_w=0.12, arch_w=0.32, skirt_n=3, nclumps=None, clump_sd=(9.0, 0.35), clump_mix=0.35, cap_t=0.55, cap_reach=0.35, diverge=0.32, rim=1.4, rim_w=3.5, glow=1.0, curtain=1, egg_c=0.55, top_flat=0.0, view_tilt=0.0, clip_ground=True, base_rise=0.0):
        self.cx, self.gy, self.W, self.H = cx, gy, W, H
        self.rng = rng
        self.lift, self.lean, self.arch_w = lift, lean, arch_w
        self.nsig = max(1.0, W * 0.02)
        self.crease_cross, self.ao_lit = 9.0, 1.2
        self.base_fill = 0.0
        self.egg_c = egg_c
        self.top_flat = top_flat
        self.view_tilt = view_tilt
        self.clip_ground = clip_ground
        self.base_rise = base_rise
        self.clump_mix = clump_mix
        self.diverge = diverge
        self.cap_t, self.cap_reach = cap_t, cap_reach
        self.curtain = curtain
        self.rim, self.rim_w = rim, rim_w
        self.glow = glow
        nw = nw or int(10 + W * 0.12)
        self.wands, self.lobes = [], []
        K = nclumps or int(rng.integers(4, 8))
        cl = [(np.radians(spread) * np.sqrt(rng.uniform(0.05, 1.0)), rng.uniform(0, 2 * np.pi)) for _ in range(K)]
        self.nclumps = K
        for i in range(nw):
            # 3-D fan, wands grouped in clumps (sub-domes): tilt from vertical, azimuth around it
            ci = i % K
            tilt = np.clip(cl[ci][0] + np.radians(rng.normal(0, clump_sd[0])), 0.03, np.radians(spread) * 1.05)
            az = cl[ci][1] + rng.normal(0, clump_sd[1])
            base = np.array([cx + rng.uniform(-1, 1) * crown_w * W / 2, gy, rng.uniform(-1, 1) * crown_w * W / 4])
            f = tilt / np.radians(spread)                     # 0 upright .. 1 splayed
            reach = (W / 2) * (0.25 + 0.75 * f) * rng.uniform(0.75, 1.0)
            height = H * (1.0 - 0.45 * (1 - self.top_flat) * f ** 2) * rng.uniform(0.8 - 0.25 * self.top_flat, 1.0)
            if self.base_rise > 0:
                # a tree crown: branches leave the trunk at different heights; low ones reach out
                # and droop, high ones are short and upright, so the crown is an oval, not a brush
                u = rng.uniform(0, 1)
                base[1] = gy - u * self.base_rise * H
                height = height * (1 - u * self.base_rise) * (0.6 + 0.4 * (1 - f))
                reach = reach * (0.55 + 0.9 * u * (1 - u) * 1.8)
            Lw = np.hypot(reach, height)
            d = np.array([np.cos(az), np.sin(az)])
            P = [np.array([0.0, 0.0]), np.array([self.diverge * reach, 0.75 * height]),
                 np.array([0.65 * reach, 1.08 * height * (1 - 0.25 * f)]),
                 np.array([reach, height * (1 - droop * 0.3) * (1 - 0.8 * f ** 1.3)])]
            pts = []
            n = 14
            for k in range(n + 1):
                t = k / n
                q = ((1 - t) ** 3) * P[0] + 3 * t * (1 - t) ** 2 * P[1] + 3 * t * t * (1 - t) * P[2] + t ** 3 * P[3]
                pts.append(base + np.array([q[0] * d[0] + lean * W * t, -q[1], q[0] * d[1]]))
            pts = np.array(pts)
            if self.view_tilt:
                # camera above: heights foreshorten, and depth (z toward viewer) moves down the screen
                ca, sa = np.cos(np.radians(self.view_tilt)), np.sin(np.radians(self.view_tilt))
                pts[:, 1] = gy + (pts[:, 1] - gy) * ca + pts[:, 2] * sa
            self.wands.append(dict(pts=pts, az=az, tilt=tilt))
            # leaf clusters along the outer part
            t = leafy_from + rng.uniform(0, 0.1)
            while t <= 1.0:
                k = t * n
                j = min(int(k), n - 1); f = k - j
                p = pts[j] * (1 - f) + pts[j + 1] * f
                r = W * rng.uniform(*lobe_r) * (0.85 + 0.3 * t)
                self.lobes.append(dict(x=p[0], y=p[1], r=r, z=p[2] + W * 0.35, u=0, v=0, wand=i, t=t, clump=ci))
                t += lobe_gap * r / max(Lw, 1) * 1.0 + 0.02
        # skirt: foliage of the lowest, most splayed wands rests on the gravel
        for side in (-1, 1):
            for k in range(int(skirt_n)):
                u = side * rng.uniform(0.35, 0.95)
                r = W * rng.uniform(*lobe_r) * 1.1
                self.lobes.append(dict(x=cx + u * (W / 2 - r), y=gy - r * rng.uniform(0.3, 0.8), r=r,
                                       z=W * 0.35 + rng.uniform(-0.3, 0.3) * W, u=u, v=0, wand=-1, t=1, clump=-1))
        self.root = (cx, gy)


def wand_dirs(b, F):
    """Per-pixel wand direction (screen, unit) from lobe ownership; smoothed; skirt lobes and
    the interior radiate from a point under the root."""
    own = F["own"]; h, w = own.shape
    dx = np.zeros((h, w)); dy = np.zeros((h, w))
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    # default: radial fan from below the root, pulled upward
    rx, ry = xx - b.cx, yy - (b.gy + 0.5 * b.H)
    rn = np.hypot(rx, ry) + 1e-6
    dx[:] = rx / rn; dy[:] = ry / rn
    for i, lb in enumerate(b.lobes):
        wi = lb.get("wand", -1)
        if wi is None or wi < 0 or not hasattr(b, "wands"):
            continue
        pts = b.wands[wi]["pts"]; n = len(pts) - 1
        k = min(int(lb["t"] * n), n - 1)
        tx, ty = pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]
        tn = np.hypot(tx, ty) + 1e-6
        m = own == i
        dx[m] = tx / tn; dy[m] = ty / tn
    dx = ndi.gaussian_filter(dx, 1.0); dy = ndi.gaussian_filter(dy, 1.0)
    nn = np.hypot(dx, dy) + 1e-6
    return dx / nn, dy / nn
