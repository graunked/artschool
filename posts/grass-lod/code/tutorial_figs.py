"""tutorial_figs.py: the few figures in the tutorials that weren't already made during the study.

All of them call the study's own code (grasslod/, vocab.py); nothing here is a new painter.

  nice -n 10 python3 tutorial_figs.py        # writes ../img/t_*.png
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from PIL import Image, ImageDraw
import vocab
from grasslod.world import World
from grasslod.marks import tuft_population, MarkParams
from grasslod.noise import stable_threshold

OUT = os.path.join(os.path.dirname(__file__), "..", "img")
BG = (24, 24, 28)


def label_row(tiles, labels, title, pad=6):
    w = sum(t.width for t in tiles) + pad * (len(tiles) + 1)
    h = max(t.height for t in tiles) + 40
    im = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(im)
    d.text((pad, 4), title, fill=(230, 230, 230))
    x = pad
    for t, lab in zip(tiles, labels):
        d.text((x, 20), lab, fill=(200, 200, 200))
        im.paste(t, (x, 34))
        x += t.width + pad
    return im


def tuft_life(hour="golden", v=4.8, win_m=(2.4, 1.6), disp=256):
    """The same patch of world ground, the same tufts, painted at shrinking stroke size.
    Each tile is rendered at its true pixel size and then scaled up (nearest) to a common
    display size, so you can follow individual tufts from strands to ticks to holes."""
    ells = [14, 9, 6, 4, 2.8, 2.0, 1.4, 1.0]
    from grasslod.world import SPECIES
    w = World(seed=1)
    mp = MarkParams()
    hmean = SPECIES[w.species][0] * (1 - 0.75 * w.grazing)
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    tiles = []
    for e in ells:
        ppm = e / (hmean * np.sqrt(c * c + (mp["lean_top"] * s ** 3) ** 2))
        W = max(8, int(round(win_m[0] * ppm)))
        H = max(6, int(round(win_m[1] * ppm * s)))
        t = vocab.flat_patch(e, v, True, hour, W=W, H=H)
        tiles.append(Image.fromarray(t).resize((disp, int(disp * H / W)), Image.NEAREST))
    im = label_row(tiles, [f"ell {e:g} px ({t.size[0] and int(round(win_m[0] * e / (hmean * np.sqrt(c*c + (mp['lean_top']*s**3)**2))))} px wide)" for e, t in zip(ells, tiles)],
                   f"One {win_m[0]} m x {win_m[1]} m patch of lit meadow, the same tufts at every zoom, each scaled to the same display width")
    im.save(os.path.join(OUT, "t_tuft_life.png"))


def nested_population(disp=240):
    """Tuft positions kept at three coverage levels: every kept set contains the next."""
    w = World(seed=1)
    box = (0, 3, 0, 3)
    tiles = []
    labs = []
    for keep in (1.0, 0.25, 0.0625, 0.0156):
        pop = tuft_population(w, *box, keep, seed=7)
        im = Image.new("RGB", (disp, disp), (240, 236, 226))
        d = ImageDraw.Draw(im)
        full = tuft_population(w, *box, 1.0, seed=7)
        for x, y in zip(full["x"], full["y"]):
            d.point((x / 3 * disp, disp - y / 3 * disp), fill=(200, 196, 186))
        for x, y, r in zip(pop["x"], pop["y"], pop["rank"]):
            X, Y = x / 3 * disp, disp - y / 3 * disp
            rr = 1.5 if keep > 0.2 else 2.5
            d.ellipse((X - rr, Y - rr, X + rr, Y + rr), fill=(40, 60, 30))
        tiles.append(im)
        labs.append(f"keep {keep:g}: {len(pop['x'])} tufts")
    label_row(tiles, labs, "The nested tuft population on 3 m x 3 m: grey = all tufts, dark = kept. Each kept set is a subset of the one before.").save(
        os.path.join(OUT, "t_nested_population.png"))


def threshold_zoom(disp=200):
    """stable_threshold over one fixed world window as the zoom changes by 1.19x a step,
    next to per-frame white noise. The world-anchored pattern grows and divides; the
    white noise is a different picture every frame."""
    tiles, labs = [], []
    rng = np.random.default_rng(0)
    for ppm in np.geomspace(8, 2, 9):
        W = int(round(6 * ppm))
        xs = (np.arange(W) + 0.5) / ppm
        X, Y = np.meshgrid(xs, xs)
        t = stable_threshold(X, Y, ppm, px=1.0)
        g = (t > 0.5).astype(np.uint8) * 200 + 30
        tiles.append(Image.fromarray(g).resize((disp // 2, disp // 2), Image.NEAREST))
        labs.append(f"{ppm:.1f} px/m")
    top = label_row(tiles, labs, "stable_threshold(X, Y, ppm) > 0.5 on the same 6 m x 6 m of ground, zooming out")
    tiles2 = []
    for ppm in np.geomspace(8, 2, 9):
        W = int(round(6 * ppm))
        g = (rng.random((W, W)) > 0.5).astype(np.uint8) * 200 + 30
        tiles2.append(Image.fromarray(g).resize((disp // 2, disp // 2), Image.NEAREST))
    bot = label_row(tiles2, labs, "screen-space white noise > 0.5, same frames: a new pattern every frame (shimmer)")
    im = Image.new("RGB", (top.width, top.height + bot.height), BG)
    im.paste(top, (0, 0)); im.paste(bot, (0, top.height))
    im.save(os.path.join(OUT, "t_threshold_zoom.png"))


if __name__ == "__main__":
    nested_population()
    threshold_zoom()
    tuft_life()
    print("ok")
