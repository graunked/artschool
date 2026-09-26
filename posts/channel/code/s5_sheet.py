"""stage 5 sheet: the channel as one form (designed: bends, narrows, splits, rejoins) and a braid
plain, at a low angle and at 30 degrees; plus a distance ladder of the same channel."""
import numpy as np, pixkit as pk, s5_braid as S
rows = []
a, _ = S.render_designed('low', 'dusk')
b, _ = S.render_designed('low', 'overcast')
c, _ = S.render_designed('oblique', 'dusk', z0=40, length=60)
rows.append(pk.panel([(a, 'one channel, low angle, dusk'), (b, 'low angle, overcast'), (c, '30 deg oblique, dusk')], 2))
d, _, _ = S.render('low', seed=12, hour='midday')
e, _, _ = S.render('low', seed=21, hour='overcast')
f, _, _ = S.render('oblique', seed=12, hour='midday')
rows.append(pk.panel([(d, 'braid plain, low, midday'), (e, 'braid plain, low, overcast'), (f, 'braid plain, 30 deg, midday')], 2))
lad = [(S.render_designed('low', 'dusk', z0=z0, length=60 * max(1, z0 / 30), eye=4.0, width=None)[0] if False else S.render_designed('low', 'dusk', z0=z0, length=60 * max(1, z0 / 30), eye=4.0)[0], 'eye 4 m, the channel from %d m' % z0) for z0 in (8, 40, 160)]
rows.append(pk.panel(lad, 2))
R = '~/work/pixelart-studies/transfer/refs/'
d2 = pk.lw_load('V02'); fr = pk.rgb(d2['idx'], d2['pal'])
rows.append(pk.panel([(pk.photo_crop(R + 'braided_lowangle_overcast_nz.jpg', (0, 200, 960, 300), (256, 80)), 'photo: NZ, low'),
                      (pk.photo_crop(R + 'braided_aerial_nz.jpg', (100, 500, 800, 250), (256, 80)), 'photo: NZ aerial'),
                      (fr[270:350, 0:256], 'Ferrari V02')], 2))
pk.stack(rows).save('../out/s5_channel.png')
pk.save(a, '../out/s5_low_dusk_4x.png', 4)
pk.save(d, '../out/s5_braid_low_midday_4x.png', 4)
pk.save(f, '../out/s5_braid_oblique_4x.png', 4)
pk.save(a[40:100, 60:200], '../out/s5_low_dusk_8x.png', 8)
