"""quick: render a short run of frames around a ppm and print pop metrics"""
import sys, numpy as np
from multiprocessing import Pool
import zoom

if __name__ == "__main__":
    p0, p1, n, hour = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    import os
    d = "../img/zt"; os.makedirs(d, exist_ok=True)
    ppms = np.geomspace(p0, p1, n)
    o = dict(theta=30, W=320, H=180, hour=hour, dir=d, world=dict(seed=1))
    with Pool(4) as p:
        list(p.imap_unordered(zoom.render, [(i, q, o) for i, q in enumerate(ppms)]))
    from PIL import Image
    fr = [np.array(Image.open(f"{d}/f{i:03d}.png")) for i in range(n)]
    t, s, hf = zoom.metrics(fr, ppms)
    print("tone", np.round(t, 2)); print("struct", np.round(s, 2))
