"""zoom.py: continuous zoom of one grassy bank, 30 -> 0.3 px/m, plus pop metrics.

  nice -n 10 python3 zoom.py NAME [--frames 96] [--hour day] [--theta 30] [--W 320 --H 180]

Writes ../img/zoom_NAME/fNNN.png, ../img/zoom_NAME.gif (2x), ../img/zoom_NAME_metrics.png.

Pop metrics between consecutive frames (A at ppm p, B at ppm p*r, r<1):
  A is shrunk by r about the centre (area average) and compared with B's centre over the
  same world area:  tone = |mean lum difference|,  struct = mean |blur2(A') - blur2(B)|.
  A handoff a viewer never notices has no spikes in either curve.
"""
import os, sys, time, argparse
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from PIL import Image
from multiprocessing import Pool


def render(args):
    i, ppm, o = args
    from grasslod.world import World
    from grasslod.render import Camera
    from grasslod.painter import paint, centre_on_bank
    from grasslod import palette as P
    w = World(**o["world"])
    cy, cz = centre_on_bank(w)
    cam = Camera(ppm=ppm, theta=o["theta"], W=o["W"], H=o["H"], cx=0, cy=cy, cz=cz)
    idx, pal = paint(w, cam, o["hour"], mparams=o.get("mparams"), vparams=o.get("vparams"))
    rgb = P.to_rgb(idx, pal)
    Image.fromarray(rgb).save(os.path.join(o["dir"], f"f{i:03d}.png"))
    return i


def lum(a):
    a = a.astype(float)
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def blur(a, s=2.0):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(a, s)


def metrics(frames, ppms):
    tone, struct, hf = [], [], []
    for k in range(len(frames)):
        L = lum(frames[k])
        hf.append(np.abs(L - blur(L, 1.5)).mean())
    for k in range(len(frames) - 1):
        A, B = lum(frames[k]), lum(frames[k + 1])
        r = ppms[k + 1] / ppms[k]
        H, W = A.shape
        # shrink A by r about centre
        As = np.array(Image.fromarray(A.astype(np.float32)).resize(
            (max(1, int(round(W * r))), max(1, int(round(H * r)))), Image.BOX))
        h, w = As.shape
        y0, x0 = (H - h) // 2, (W - w) // 2
        Bc = B[y0:y0 + h, x0:x0 + w]
        m = 6
        tone.append(abs(As[m:-m, m:-m].mean() - Bc[m:-m, m:-m].mean()))
        struct.append(np.abs(blur(As)[m:-m, m:-m] - blur(Bc)[m:-m, m:-m]).mean())
    return np.array(tone), np.array(struct), np.array(hf)


def plot(path, ppms, tone, struct, hf):
    from PIL import ImageDraw
    Wp, Hp = 720, 300
    im = Image.new("RGB", (Wp, Hp), (250, 250, 248))
    d = ImageDraw.Draw(im)
    n = len(ppms)
    def X(k):
        return 40 + (Wp - 60) * k / max(1, n - 1)
    for arr, col, sc, lab in ((tone, (200, 60, 40), 4.0, "tone pop (lum)"),
                              (struct, (40, 90, 200), 2.0, "structure pop (lum, blur2)"),
                              (hf, (60, 150, 60), 2.0, "grain energy")):
        pts = [(X(k + (0.5 if len(arr) < n else 0)), Hp - 30 - min(250, arr[k] * sc * 4)) for k in range(len(arr))]
        d.line(pts, fill=col, width=2)
        d.text((Wp - 230, 10 + 14 * ["tone pop (lum)", "structure pop (lum, blur2)", "grain energy"].index(lab)), lab, fill=col)
    for k in range(0, n, max(1, n // 8)):
        d.text((X(k) - 10, Hp - 20), f"{ppms[k]:.2g}", fill=(0, 0, 0))
    d.text((5, Hp - 20), "px/m", fill=(0, 0, 0))
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--hour", default="day")
    ap.add_argument("--theta", type=float, default=30)
    ap.add_argument("--W", type=int, default=320)
    ap.add_argument("--H", type=int, default=180)
    ap.add_argument("--p0", type=float, default=30)
    ap.add_argument("--p1", type=float, default=0.3)
    ap.add_argument("--species", default="meadow")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    d = os.path.join(os.path.dirname(__file__), "..", "img", f"zoom_{a.name}")
    os.makedirs(d, exist_ok=True)
    ppms = np.geomspace(a.p0, a.p1, a.frames)
    o = dict(theta=a.theta, W=a.W, H=a.H, hour=a.hour, dir=d, world=dict(seed=1, species=a.species))
    t = time.time()
    with Pool(a.procs) as pool:
        for i in pool.imap_unordered(render, [(i, p, o) for i, p in enumerate(ppms)]):
            pass
    print("render", round(time.time() - t), "s")
    frames = [np.array(Image.open(os.path.join(d, f"f{i:03d}.png"))) for i in range(len(ppms))]
    tone, struct, hf = metrics(frames, ppms)
    np.savez(os.path.join(d, "metrics.npz"), ppms=ppms, tone=tone, struct=struct, hf=hf)
    plot(os.path.join(d, "..", f"zoom_{a.name}_metrics.png"), ppms, tone, struct, hf)
    print("tone pop  max %.2f mean %.2f  | struct max %.2f mean %.2f" % (tone.max(), tone.mean(), struct.max(), struct.mean()))
    worst = np.argsort(-struct)[:5]
    print("worst struct pairs:", [(int(k), round(float(ppms[k]), 2), round(float(struct[k]), 2)) for k in worst])
    gif = os.path.join(d, "..", f"zoom_{a.name}.gif")
    os.system(f"ffmpeg -y -loglevel error -framerate 12 -i {d}/f%03d.png "
              f"-vf \"scale=iw*2:ih*2:flags=neighbor,split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse=dither=none\" {gif}")
    # filmstrip: 8 frames
    sel = np.linspace(0, len(frames) - 1, 8).astype(int)
    strip = np.concatenate([np.pad(frames[k], ((0, 2), (0, 0), (0, 0))) for k in sel], 0)
    Image.fromarray(strip).resize((strip.shape[1] * 2, strip.shape[0] * 2), Image.NEAREST).save(
        os.path.join(d, "..", f"zoom_{a.name}_strip.png"))


if __name__ == "__main__":
    main()
