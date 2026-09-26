import numpy as np, pixkit as pk, channel as ch, painter
P = painter.grey_palette()
gi, gv = P.r('grav'), painter.vals(P, 'grav')
W, H = 96, 60
f, h = 420, 1.6
hz = -49.0
rows = np.arange(H) + 0.5
z = f * h / (rows - hz)
variants = [dict(sm=(0.03, 0.16), den=0.6, base=104, a=0.6), dict(sm=(0.05, 0.25), den=0.45, base=100, a=0.7),
            dict(sm=(0.04, 0.2), den=0.35, base=96, a=0.8), dict(sm=(0.04, 0.12), den=0.8, base=104, a=0.55)]
ims = []
for i, v in enumerate(variants):
    rng = np.random.default_rng(i)
    base = np.full((H, W), float(v['base'])) + (pk.value_noise(H, W, 6, rng, 2) - 0.5) * 18
    g = ch.gravel(np.ones((H, W), bool), base, z, f, rng, gi, gv, v['sm'], v['den'], aspect=v['a'])
    ims.append((P.rgb(g), str(v['sm'])))
ph = pk.photo_crop('~/work/pixelart-studies/transfer/refs/braided_hoh_gravel_bar.jpg', (60, 420, 420, 262), (W, H))
ims.append((ph, 'photo Hoh'))
pk.panel(ims, 5).save('../study/gravel_swatch.png')
