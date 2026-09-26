import sys; sys.path.insert(0, '~/work/pixelart-studies/pedagogy/atelier')
from core import *
import scene
from scene import Params, render_buffers
import drawing
from paint import Painter, colour_palette, grey_palette
from backdrop_util import sun_side_of
import numpy as np

_orig = scene.Backdrop.mountains
def crag_mountains(self, az, detail=True):
    em = _orig(self, az, detail)
    if not hasattr(self, "crags"):
        rpp = 2 * self.cam.tanh / self.cam.W                # radians per pixel
        rng = np.random.default_rng(self.seed + 777)
        self.crags = []; a = -0.8
        while a < 0.8:
            e0 = _orig(self, np.array([a]))[0]
            h = rng.integers(1, 5) * rpp
            sl, sr = rng.choice([1.0, 0.5, 2.0], 2)          # clean slopes, in px per px
            self.crags.append((a, e0, h, sl, sr, rng.random() < 0.75))
            a += rng.integers(7, 21) * rpp
        _orig(self, az, detail)                              # restore _own for this az
    for a, e0, h, sl, sr, up in self.crags:
        d = az - a
        s = np.where(d < 0, sl, sr)
        if up:
            em = np.maximum(em, e0 + h - np.abs(d) * s)
        else:
            near = np.abs(d) * s < h
            em = np.where(near, np.minimum(em, e0 - h + np.abs(d) * s), em)
    return em
if len(sys.argv) > 1 and sys.argv[1].startswith("crag"):
    scene.Backdrop.mountains = crag_mountains
if len(sys.argv) > 1 and sys.argv[1].endswith("nosnap"):
    # built from clean slopes already: don't blur-and-resnap the skyline
    from scene import MTN, SNOW
    _cr = drawing.clean_region
    def clean_region(buf, mask, keys, within=None, sigma=1.0, tol=0.75):
        if sigma == 0.8:                                    # the three mountain/snow calls
            return 0, 0
        return _cr(buf, mask, keys, within, sigma, tol)
    drawing.clean_region = clean_region
if len(sys.argv) > 2 and sys.argv[2] == "spire":
    from spruce2 import spruce2
    drawing.spruce = spruce2
if len(sys.argv) > 3 and sys.argv[3] == "dense":
    # spires are lacy, so plant twice as many: overlapping, they make the dark mass again
    import inspect, textwrap
    src = inspect.getsource(drawing.draw_trees).replace("[(9.0, 0.8, 0.20), (13.0, 1.0, 0.13)]", "[(4.5, 0.8, 0.20), (6.0, 1.0, 0.13)]")
    ns = dict(drawing.__dict__); exec(src, ns); drawing.draw_trees = ns["draw_trees"]
tag = "_".join(sys.argv[1:]) if len(sys.argv) > 1 else "plain"
P = Params('medium', 'left', '10m', 0)
buf = drawing.finish(render_buffers(P))
idx = Painter(buf, seed=0).paint(buf["st"].boulder, buf["cam"].f, sun_side_of(buf), P.bsize, cam=buf["cam"])
np.save(f"cs_{tag}.npy", idx)
to_image(idx, colour_palette(), 3).save(f"cs_{tag}_full.png")
im = to_image(idx, colour_palette(), 1).crop((0, 0, 160, 45)); im.resize((960, 270), Image.NEAREST).save(f"cs_{tag}_mtn.png")
print(tag, len(np.unique(idx)))
