"""stage 6: parameter sheets. python3 sheets.py [all|1|2|3|4|5]"""
import sys
import numpy as np, pixkit as pk, technique as T


def sheet(name, rows, cols, fn, title_fn, s=2):
    out = []
    for r in rows:
        items = []
        for c in cols:
            rgb, _ = T.render(fn(r, c))
            items.append((rgb, title_fn(r, c)))
        out.append(pk.panel(items, s))
    pk.stack(out).save('../out/sheet_%s.png' % name)


def main(which):
    if which in ('all', '1'):
        sheet('1_width_x_depth', [0.08, 0.25, 0.6], [1.5, 3.5, 7.0],
              lambda d, w: dict(width=w, depth=d, hour='overcast'),
              lambda d, w: 'width %.1f m, depth %.2f m' % (w, d))
    if which in ('all', '2'):
        sheet('2_flow_x_hour', ['midday', 'overcast', 'dusk'], [0.0, 0.5, 1.0],
              lambda h, f: dict(flow=f, hour=h, width=3.5),
              lambda h, f: '%s, flow %.1f' % (h, f))
    if which in ('all', '3'):
        sheet('3_bank_x_camera', ['low', 'oblique'], ['gravel', 'cut', 'grass', 'tussock_near'],
              lambda cam, b: dict(bank=b, camera=cam, hour='midday'),
              lambda cam, b: '%s bank, %s camera' % (b, cam))
    if which in ('all', '4'):
        sheet('4_distance_x_hour', ['midday', 'dusk'], [4.0, 9.0, 25.0, 70.0],
              lambda h, d: dict(distance=d, hour=h, width=3.5, bend=0.4),
              lambda h, d: '%s, near edge at %d m' % (h, d))
    if which in ('all', '5'):
        sheet('5_seeds_milky', [True, False], [1, 2, 3],
              lambda m, sd: dict(seed=sd, milky=m, hour='midday', flow=0.4 * (sd % 2), bend=0.5),
              lambda m, sd: 'seed %d, %s' % (sd, 'glacial milk' if m else 'clear water'))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'all')
