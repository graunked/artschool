"""Stage 2: the edge. Cells: far/near edges against gravel (sloping and cut banks) and grass."""
import sys, time
import numpy as np
import pixkit as pk
import world, painter

P = painter.grey_palette()
W, H = 128, 80


def cell(name, pitch, terrain, prm=None, seed=1, f=420):
    cam = world.Cam(W, H, f=f, h=1.6, pitch_deg=pitch)
    t0 = time.time()
    G = world.raycast(cam, terrain)
    cv = painter.paint(G, cam, P, np.random.default_rng(seed), prm)
    return P.rgb(cv), G, cam


CELLS = {
    'a_far_slope': (12.0, dict(z=(5.0, 7.0), far=('slope', 0.2, 0.6), near=('slope', 0.12, 1.0)), None),
    'b_far_cut': (12.0, dict(z=(5.0, 7.0), far=('cut', 0.45), near=('slope', 0.12, 1.0)), None),
    'c_near_gravel': (17.0, dict(z=(4.4, 6.5), far=('slope', 0.2, 0.6), near=('slope', 0.15, 0.8)), None),
    'd_far_grass': (12.0, dict(z=(5.0, 7.0), far=('cut', 0.18), near=('slope', 0.12, 1.0)),
                    dict(grass=dict(side='far', min_y=0.08, blade_m=0.35, density=0.4, lean=0.3, lit=0.4))),
    'e_near_grass': (17.0, dict(z=(4.4, 6.5), far=('slope', 0.2, 0.6), near=('cut', 0.3)),
                     dict(grass=dict(side='near', min_y=0.15, blade_m=0.4, density=0.4, lean=-0.3, lit=0.4))),
}


def render(name, seed=1):
    pitch, tp, prm = CELLS[name]
    t = world.StraightChannel(tp['z'][0], tp['z'][1], far=tp['far'], near=tp['near'], seed=seed + 2)
    return cell(name, pitch, t, prm, seed)


if __name__ == '__main__':
    names = sys.argv[1:] or list(CELLS)
    for n in names:
        rgb, G, cam = render(n)
        pk.save(rgb, '../study/s2_%s.png' % n, 6)
