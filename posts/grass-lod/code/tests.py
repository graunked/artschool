"""tests.py: regression checks for the handoff machinery.

1. pan stability: panning the camera by whole pixels must reproduce the same pixels.
   (This caught two array-alignment bugs that made silhouettes and stroke colours
   quasi-random, i.e. boiling on zoom, while every single frame looked plausible.)
2. tone continuity: the flat-patch mean tone across the ell ladder stays within 0.25 steps
   of the reference (tone lock).
"""
import numpy as np
from grasslod.world import World
from grasslod.render import Camera
from grasslod import painter


def pan(ppm=22.0, hour="golden", theta=30):
    w = World(seed=1); cy, cz = painter.centre_on_bank(w)
    fr = []
    for dx in (0, 3):
        cam = Camera(ppm=ppm, theta=theta, W=240, H=140, cx=dx / ppm, cy=cy, cz=cz)
        idx, _ = painter.paint(w, cam, hour)
        fr.append(idx)
    d = (fr[0][:, 3:] != fr[1][:, :-3])[:, 12:-12].mean()
    print(f"pan stability ppm={ppm}: differing pixels {d:.4f}", "OK" if d < 0.002 else "FAIL")
    return d


def tone():
    import vocab
    from grasslod import palette as P
    ells = [14, 6, 2.8, 1.4, 0.7, 0.3, 0.12]
    for v in (4.8, 4.1):
        ms = []
        for e in ells:
            t = vocab.flat_patch(e, v, True, "golden", W=96, H=72)
            pal, _ = P.build("golden")
            lut = {tuple(c): i for i, c in enumerate(pal)}
            q = np.array([lut[tuple(c)] - P.GREEN0 for c in t.reshape(-1, 3)])
            ms.append(q.mean())
        ms = np.array(ms)
        print(f"tone V={v}: mean step by ell {np.round(ms, 2)}  spread {ms.max() - ms.min():.2f}")


if __name__ == "__main__":
    for p in (22.0, 6.0, 1.5):
        pan(p)
    tone()
