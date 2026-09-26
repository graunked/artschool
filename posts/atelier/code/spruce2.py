import numpy as np

def spruce2(h, rng, lean=0):
    """A distant spruce as a SPIRE, not a cone (after Ferrari's, measured at native scale):
      - a 1-px leader the full height;
      - near the tip, a herringbone: 1-px ticks alternating sides every row;
      - below, tiers of branch strokes on alternating sides, with sky between the tiers:
        a flat run that droops one pixel at its tip;
      - a core only 1 px wide at the top, widening to ~a quarter of the spread at the foot;
      - overall width about a third of the height.
    Returns (dy, dx, kind): kind 0 core, 1 branch, 2 tip. dy counts up from the foot."""
    px = []
    half_max = max(1.0, h * rng.uniform(0.16, 0.22))
    tip = max(2, int(round(h * rng.uniform(0.12, 0.2))))
    for dy in range(h):
        px.append((dy, 0, 2 if dy >= h - tip else 0))
    # herringbone ticks at the tip
    side = 1 if rng.random() < 0.5 else -1
    for dy in range(h - 2, h - tip - 1, -1):
        px.append((dy, side, 2)); side = -side
    # tiers below: strokes on BOTH sides, the two sides a row apart (a herringbone), so the
    # silhouette zigzags; now and then a tier is skipped and the sky shows through
    dy = h - tip - 1
    while dy > 0:
        t = (h - dy) / h
        env = half_max * t ** 0.85
        if not (t > 0.3 and rng.random() < 0.18):          # an occasional gap
            for s_, off in ((side, 0), (-side, -1)):
                L = int(round(env * rng.uniform(0.6, 1.2)))
                for k in range(1, L + 1):
                    ddy = -1 if (L >= 3 and k == L) else 0  # droop one pixel at the tip
                    px.append((dy + off + ddy, s_ * k, 1))
        core = int(round(env * 0.28))
        for c in range(-core, core + 1):
            px.append((dy, c, 0))
        side = -side
        dy -= 2 if rng.random() < 0.7 else 3
    return px
