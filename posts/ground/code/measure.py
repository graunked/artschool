"""The measurements from the study, collected so they can be rerun.
    cd code && python3 measure.py
region_stats: commonest palette indices in a box, with L* and share (the V16 table)
step_shares:  share of each ramp step inside a family of one of my paintings
crest_profile_of: mean value by depth below each crest, on Ferrari's index map (the hummock profile)"""
import numpy as np, lw
from analyze import Lstar

def region_stats(name, box, top=6):
    d = lw.load(name); I, P = d['idx'], d['pal']
    x0, y0, x1, y1 = box
    s = I[y0:y1, x0:x1]; u, c = np.unique(s, return_counts=True); o = np.argsort(-c)
    return [(int(u[k]), tuple(int(t) for t in P[u[k]]), round(float(Lstar(P[u[k]][None])[0])), round(100 * c[k] / s.size))
            for k in o[:top]]

def step_shares(g, family='shadow'):
    m = (g.mat >= 0) & (g.step < 3 if family == 'shadow' else g.step >= 3)
    return np.round(np.bincount(g.step[m], minlength=6) / max(m.sum(), 1), 2)

# V16 far hummocks: index -> step (0 darkest .. 3 lit), read off the palette by L*
VAL = {69: 3, 235: 3, 72: 2, 59: 2, 70: 1, 120: 1, 126: 1, 71: 0, 127: 0, 60: 0, 61: 0, 62: 0, 52: 0, 53: 0}

def crest_profile_of(name='V16', box=(300, 215, 640, 280), every=2, maxk=40):
    """runs of hummock pixels below a non-hummock pixel, bucketed by run length; mean step by depth"""
    d = lw.load(name); x0, y0, x1, y1 = box
    v = np.vectorize(lambda k: VAL.get(k, -1))(d['idx'][y0:y1, x0:x1])
    prof = {}
    H, W = v.shape
    for x in range(W):
        col = v[:, x]; y = 1
        while y < H:
            if col[y] >= 0 and col[y - 1] < 0:
                y0_ = y
                while y < H and col[y] >= 0: y += 1
                L = y - y0_
                if L >= 6:
                    for k in range(L): prof.setdefault((min(L, maxk) // 10, k), []).append(col[y0_ + k])
            y += 1
    return {Lb: [round(float(np.mean(prof[(Lb, k)])), 1) for k in range(0, maxk, every) if (Lb, k) in prof]
            for Lb in range(5)}

if __name__ == '__main__':
    for n, b in [('islet', (265, 270, 420, 300)), ('far hummocks', (340, 225, 640, 260)),
                 ('right bank', (420, 330, 640, 480)), ('near left bank', (0, 330, 200, 480))]:
        print(n, [(L, s) for _, _, L, s in region_stats('V16', b)])
    for Lb, row in crest_profile_of().items(): print(f'runs ~{Lb}0 px:', ' '.join(map(str, row)))
    import ground
    g = ground.paint_ground(310, 160, ground='turf', landform='hummocks', hour='noon', sun=(80, -2),
                            ppm=8, pitch=5, cam_h=3.0, relief=1.5, seed=4)
    print('my hummocks, shadow step shares 0..2:', step_shares(g)[:3])
