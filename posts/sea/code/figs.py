"""Figures for the art-school pages. Paints nothing of its own: it calls the study's functions (this folder is a
snapshot of elements/sea/src), sometimes with one of their parameters changed, and lays out the results.

    cd ~/work/artschool/posts/sea/code
    OPENBLAS_NUM_THREADS=1 nice -n 10 python3 figs.py [name ...]      # writes ../img/fig_*.png

Needs the study's sim results (elements/sea/cache/*.npz, read-only) and the olycoast lidar pack."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as nd
import sea.scene, sea.paint
STUDY = os.path.expanduser('~/work/pixelart-studies/elements/sea')
sea.scene.CACHE = os.path.join(STUDY, 'cache')
from sea.scene import scene
from sea.render import Cam, gbuffer
from sea.pal import build
from stage1 import render, photo_view
from lab import srgb_to_lab
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'img')
PHOTO = os.path.expanduser('~/work/voxsim-data/places/olycoast/obliques/point_of_arches__060819_15405_2016.jpg')
HM = np.load(os.path.join(STUDY, 'notes', 'H_poa.npy'))
BG = (18, 18, 22)

def up(a, k): return Image.fromarray(np.asarray(a, np.uint8)).resize((a.shape[1] * k, a.shape[0] * k), Image.NEAREST)

def strip(panels, labels, k, title, out):
    w = sum(p.shape[1] * k + 8 for p in panels); h = max(p.shape[0] for p in panels) * k + 40
    im = Image.new('RGB', (w, h), BG); d = ImageDraw.Draw(im); d.text((6, 4), title, fill=(235, 235, 235))
    x = 0
    for p, l in zip(panels, labels):
        im.paste(up(p, k), (x, 36)); d.text((x + 4, 22), l, fill=(210, 210, 210)); x += p.shape[1] * k + 8
    im.save(os.path.join(OUT, out)); print('wrote', out)

def photo_at(cx, cy, mpp, W, H, ss=4):
    c4 = Cam(cx=cx, cy=cy, mpp=mpp / ss, W=W * ss, H=H * ss, yaw=90, elev=30, zc=0.45)
    G4 = gbuffer(c4, np.full((10, 10), -40.0), sea_level=0.45)
    ph4, _ = photo_view(G4, HM, PHOTO)
    return np.asarray(Image.fromarray(ph4).resize((W, H), Image.BOX))

def fig_pinnacle():
    """Why the reef heads break: the same crest tested against the smoothed 1 m depth and against the
    pinnacle depth (shallowest within 5 m) that SeaState uses."""
    d = np.load(os.path.join(sea.scene.CACHE, 'poa_summer.npz'))
    P, S, zl = scene('poa', 'poa_summer.npz')
    dep = np.maximum(S['depth'], 0.02); Hi = S['Hinst']; land = d['land'].astype(bool)
    d_plain = nd.gaussian_filter(dep, 1.2)
    d_pin = nd.gaussian_filter(-nd.maximum_filter(-dep, 5), 1.0)
    sl = (slice(380, 880), slice(520, 1020))
    base = np.clip(1 - dep[sl] / 14, 0, 1)
    def pan(brk):
        rgb = np.stack([base * 0.35 + 0.1, base * 0.5 + 0.15, base * 0.45 + 0.3], -1)
        rgb[brk[sl] & ~land[sl]] = [1.0, 0.25, 0.2]; rgb[land[sl]] = [0.5, 0.47, 0.42]
        return (rgb * 255).astype(np.uint8)
    a = Hi > 0.62 * d_plain; b = Hi > 0.62 * d_pin
    print('breaking cells in crop: plain', int((a[sl] & ~land[sl]).sum()), 'pinnacle', int((b[sl] & ~land[sl]).sum()))
    strip([pan(a), pan(b)], ['H_inst > 0.62 x depth (lidar, smoothed 1.2 m)', 'H_inst > 0.62 x pinnacle depth (min over 5 m)'], 1,
          'One instant, Point of the Arches cove (500 m square, north up). Red: this crest is breaking here. Blue: lighter = shallower.',
          'fig_pinnacle.png')

def fig_three_foams():
    """The sim's three foam quantities over the cove, beside the photo warped to the same map."""
    d = np.load(os.path.join(sea.scene.CACHE, 'poa_summer.npz'))
    land = d['land'].astype(bool)
    sl = (slice(200, 1200), slice(200, 1150))            # = the ortho box (550,450,1500,1450) in the place frame
    photo = np.asarray(Image.open(os.path.join(STUDY, 'notes', 'poa_ortho.png')).convert('RGB'))
    def mono(f, col, gain):
        v = np.clip(f.astype(np.float32)[sl] * gain, 0, 1)[..., None]
        rgb = np.array([0.08, 0.12, 0.2]) * (1 - v) + np.array(col) * v
        rgb[land[sl]] = [0.4, 0.38, 0.35]; return (rgb * 255).astype(np.uint8)
    F = np.zeros(land.shape, np.float32); np.add.at(F, (d['py'].astype(int), d['px'].astype(int)), 1.0)
    panels = [photo, mono(d['W'], (1, 1, 1), 1.5), mono(d['A'], (0.55, 0.85, 0.8), 1.2), mono(nd.gaussian_filter(F, 0.7), (1, 0.95, 0.6), 0.6)]
    panels = [np.asarray(Image.fromarray(p).resize((475, 500), Image.BOX)) for p in panels]
    strip(panels, ['photo, warped onto the lidar map', 'W: whitewater (tau 6 s)', 'A: aeration (tau 14 s)', 'F: foam particles (25 s / 450 s lives)'], 1,
          'Point of the Arches, summer swell Hs 1.6 m, 8.3 s from 281 deg, 60 wave periods simulated. North up, 1 m/px.',
          'fig_three_foams.png')

def fig_cq():
    """The octave rule: the same foam at 3 m/px with every texture size fixed in metres, and with cq()."""
    P, S, zl = scene('poa', 'poa_summer.npz')
    cam = Cam(cx=720, cy=750, mpp=3.0, W=240, H=150, yaw=90, elev=30, zc=0.45)
    real = sea.paint.cq
    sea.paint.cq = lambda base, mpp, k: base + 0 * np.asarray(mpp, float)
    a, _ = render(S, zl, cam)
    sea.paint.cq = real
    b, _ = render(S, zl, cam)
    ph = photo_at(720, 750, 3.0, 240, 150)
    strip([ph, a, b], ['photo, 3 m/px', 'textures fixed in metres (no cq)', 'cq(base, mpp, k): >= k px, octave steps'], 3,
          'The whole cove at region scale. Without the rule, foam edges and holes are 1 px cells: ragged speckle along every edge.',
          'fig_cq.png')

def fig_calib():
    """Foam coverage calibrated against the photo's white fraction at the same view and pixel size."""
    P, S, zl = scene('poa', 'poa_summer.npz')
    cam = Cam(cx=720, cy=750, mpp=1.0, W=480, H=300, yaw=90, elev=30, zc=0.45)
    ph = photo_at(720, 750, 1.0, 480, 300)
    ctr = np.zeros(ph.shape[:2], bool); ctr[55:245, 90:390] = True      # the open-water middle of the cove
    Lp = srgb_to_lab(ph.astype(float))[..., 0]; v = ph.sum(2) > 0
    panels, labels = [ph], []
    for g in (0.15, 0.3, 0.6):
        rgb, G = render(S, zl, cam, params=dict(foam_gain=g))
        Lm = srgb_to_lab(rgb.astype(float))[..., 0]; m = v & G['sea']
        panels.append(rgb); labels.append(f'foam_gain {g}: white {100 * (Lm[m] > 75).mean():.1f}% (middle {100 * (Lm[m & ctr] > 75).mean():.1f}%)')
    labels = [f'photo: white (L* > 75) {100 * (Lp[v] > 75).mean():.1f}% (middle {100 * (Lp[v & ctr] > 75).mean():.1f}%)'] + labels
    strip(panels, labels, 1, 'The fraction of sea pixels whiter than L* 75, whole view and open-water middle. The total can match while the placement is wrong.', 'fig_calib.png')

def fig_ramps():
    """The palette as light: every sea ramp in each of the five light states (one canvas, five palettes)."""
    rows = []
    names = ['sea_sand', 'sea_rock', 'kelp', 'aer', 'foam', 'mirror']
    for light in ['sun', 'overcast', 'golden', 'dusk', 'fog']:
        P = build(light); row = []
        for n in names:
            cols = P[n]; row += [np.tile(c, (1, 1)) for c in cols] + [np.array([BG])]
        rows.append(np.concatenate(row, 0)[None])
    a = np.concatenate(rows, 0).astype(np.uint8)
    im = up(a, 22); canvas = Image.new('RGB', (im.width + 80, im.height + 40), BG); canvas.paste(im, (80, 30))
    dr = ImageDraw.Draw(canvas)
    for i, l in enumerate(['sun', 'overcast', 'golden', 'dusk', 'fog']): dr.text((6, 36 + i * 22), l, fill=(220, 220, 220))
    x = 80
    for n in names:
        k = len(build('sun')[n]); dr.text((x + 2, 12), n, fill=(220, 220, 220)); x += (k + 1) * 22
    canvas.save(os.path.join(OUT, 'fig_ramps.png')); print('wrote fig_ramps.png')

if __name__ == '__main__':
    names = sys.argv[1:] or ['pinnacle', 'three_foams', 'cq', 'calib', 'ramps']
    for n in names: globals()['fig_' + n]()
