"""Figures for the Oga pages. Everything here is either a copy of an image I looked at while working
(the notebook's img/NNN, copied to ../img/) or a crop / zoom of the study's own outputs, some re-rendered
by importing the study's own paint() functions. Nothing is repainted with simpler code.

    ~/miniconda3/bin/python3 howto/figs.py      (from ~/work/artschool/posts/oga)

Writes only into this post folder; the study folder is read, never written (no bytecode, no saves)."""
import sys, os, shutil, glob
sys.dont_write_bytecode = True
import numpy as np
from PIL import Image

STUDY = os.path.expanduser('~/work/pixelart-studies/books/oga')
NB = os.path.expanduser('~/work/artschool/notebooks/oga/img')
HERE = os.path.dirname(os.path.abspath(__file__))
POST = os.path.dirname(HERE)
FIG = os.path.join(HERE, 'fig')
IMG = os.path.join(POST, 'img')
sys.path.insert(0, STUDY + '/src'); sys.path.insert(0, STUDY + '/copies'); sys.path.insert(0, STUDY + '/places')
from sheet import zoom, grid, fit, area

# ---- 1. notebook images used on the pages -------------------------------------------------------
NB_USED = [14, 20, 30, 45, 53, 157, 160, 169, 171, 172, 174, 176, 177, 178, 184, 195, 205, 207, 208, 210, 214, 215, 216, 220, 221, 222, 231, 235, 236, 240, 241, 242, 245, 248, 250, 254, 256, 258, 259, 263, 265, 266, 268]
for i in NB_USED:
    f = glob.glob(f'{NB}/{i:03d}.*')[0]
    shutil.copy(f, os.path.join(IMG, 'nb' + os.path.basename(f)))

# ---- 2. study sheets (final versions) -----------------------------------------------------------
for n in ['00_survives', '00_plate_swatches', 'ex2_ramps', 'ex3b_greys_by_place', 'ex1b_plan', 'p1_foliage', 'p1_foliage_2x',
          'p2_peak', 'p3_rocks', 'p4_meadow', 'p5_gorge', 'place_skykomish', 'place_skykomish_stages', 'place_enchant',
          'place_olycoast', 'place_olycoast_stages', 'place_yakima', 'place_yakima_stages', 'place_sanjuan',
          'place_sanjuan_stages', 'lane_skykomish', '99_places', 'copy1_cover', 'copy2_c0007']:
    shutil.copy(f'{STUDY}/sheets/{n}.png', os.path.join(FIG, n + '.png'))

# ---- 3. re-rendered passage copies: his | mine at 8x, crops ------------------------------------
def pair(ref, mine, k, name, labels):
    grid([zoom(ref, k), zoom(mine, k)], 2, labels).save(os.path.join(FIG, name))

import p1_foliage, p2_peak, p3_rocks, p4_meadow, p5_gorge
cv, ref = p1_foliage.paint()
pair(ref.crop((20, 10, 60, 40)), Image.fromarray((cv.rgb()[10:40, 20:60] * 255 + .5).astype(np.uint8)), 10,
     'p1_detail_8x.png', ['his C-0007, area-reduced, 10x', 'my clumps, 10x'])
cv, ref = p2_peak.paint()
pair(ref, Image.fromarray((cv.rgb() * 255 + .5).astype(np.uint8)), 6, 'p2_final_6x.png', ['his cover, reduced, 6x', 'my copy, 6x'])
cv = p3_rocks.paint(); ref = Image.open(f'{STUDY}/copies/p3_ref.png')
pair(ref, Image.fromarray((cv.rgb() * 255 + .5).astype(np.uint8)), 6, 'p3_final_6x.png', ['his cover rocks, reduced, 6x', 'my copy, 6x'])
cv, ref = p4_meadow.paint()
pair(ref, Image.fromarray((cv.rgb() * 255 + .5).astype(np.uint8)), 6, 'p4_final_6x.png', ['his C-0064 slope, reduced, 6x', 'my copy, 6x'])
cv = p5_gorge.paint(); ref = Image.open(f'{STUDY}/copies/p5_ref.png')
pair(ref, Image.fromarray((cv.rgb() * 255 + .5).astype(np.uint8)), 5, 'p5_final_5x.png', ['his C-0792, reduced, 5x', 'my copy, 5x'])

# ---- 4. places: finals at 2x, crops at 6x ------------------------------------------------------
for n in ['sanjuan', 'skykomish', 'enchant', 'olycoast', 'yakima', 'lane_skykomish']:
    im = Image.open(f'{STUDY}/places/out/{n}.png')
    zoom(im, 2).save(os.path.join(FIG, f'{n}_2x.png'))
crops = {'sanjuan_crowns': ('sanjuan', (90, 0, 190, 60), 6), 'sanjuan_camas': ('sanjuan', (140, 110, 240, 180), 6),
         'skykomish_edge': ('skykomish', (150, 40, 250, 110), 6), 'olycoast_forest': ('olycoast', (0, 20, 110, 100), 5),
         'enchant_water': ('enchant', (40, 130, 150, 230), 5), 'yakima_band': ('yakima', (100, 60, 220, 130), 5),
         'olycoast_root': ('olycoast', (170, 70, 290, 170), 5)}
for name, (n, box, k) in crops.items():
    zoom(Image.open(f'{STUDY}/places/out/{n}.png').crop(box), k).save(os.path.join(FIG, name + '.png'))

# ---- 5. book plates (levels-corrected crops from the study), small, for reference ---------------
for n in ['C0007_bg_in', 'C0064_bg', 'C0792_boulders', 'cover_final', 'C39-5_greys', 'C0081_greens']:
    fit(Image.open(f'{STUDY}/plates/lv/{n}.png'), w=720).save(os.path.join(FIG, 'plate_' + n + '.png'))
print('ok')
