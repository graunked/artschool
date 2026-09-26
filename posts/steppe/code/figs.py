"""figs: the teaching figures for the steppe tutorial, made from the study's real code.

Every function here calls the snapshot of the study's own modules (sage, grass, pine, world,
scene, palette, backdrop), with layers switched on or off through their real parameters or by
replacing one real function with a no-op for one call. Nothing here is a new painter.

    cd code && nice -n 10 python3 figs.py           # all figures -> ../img/t_*.png
    nice -n 10 python3 figs.py sage_layers palette  # just some
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from PIL import Image
import sp, sage, grass, pine, world, scene, palette, backdrop
from s1_sage import Grey, bush_on_ground

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "img")
NAT = os.path.expanduser("~/work/naturalist/plants/photos/")
STUDY = os.path.expanduser("~/work/pixelart-studies/elements/steppe/")


def save(img, name, k=1):
    sp.save(img, os.path.join(OUT, name), k)
    print(name, img.shape[1] * k, "x", img.shape[0] * k, flush=True)


# ---------------------------------------------------------------- 1. box-downsample the photo
def box_downsample(path, box, width=64):
    """The instrument: crop the photo to one plant, shrink it with a BOX filter (every output
    pixel is the plain mean of the photo pixels under it) to the width we paint at."""
    im = Image.open(path).convert("RGB").crop(box)
    small = im.resize((width, max(1, int(width * im.height / im.width))), Image.BOX)
    return np.array(im), np.array(small)


def fig_downsample():
    crops = [
        (NAT + "sagebrush_steppe_Washington/commons_Sagebrush_Sea_adjacent_to_Seedskadee_National_Wi_e3282a3f.jpg",
         (230, 555, 350, 645), "Seedskadee, one sage"),
        (NAT + "sagebrush_steppe_Washington/commons_Sagebrush_Sea_adjacent_to_Seedskadee_National_Wi_e3282a3f.jpg",
         (430, 600, 560, 700), "Seedskadee, sage over its stems"),
        (NAT + "Moses_Coulee_basalt/openverse_Hungate_Canyon_Moses_Coulee_Columbia_River_Plate_35216bab.jpg",
         (830, 380, 1023, 600), "Hungate Canyon, low sun"),
    ]
    rows = []
    for path, box, lab in crops:
        big, small = box_downsample(path, box)
        g = np.array(Image.fromarray(small).convert("L"))
        g = np.stack([g] * 3, -1)
        H = small.shape[0] * 4
        rows.append(sp.hstack([sp.fit(big, H), sp.upscale(small, 4), sp.upscale(g, 4)],
                              [lab, "box 64 px, 4x", "grey, 4x"]))
    W = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0)), constant_values=22) for r in rows]
    save(np.vstack(rows), "t_downsample.png")


# ---------------------------------------------------------------- 2. the four sage models
def fig_sage_models():
    """Same seed, same size, greyscale: the three painters still in sage.py (mode= 'spray',
    'field', 'sprig'), beside the photo downsampled to the same scale."""
    cells, labs = [], []
    _, small = box_downsample(NAT + "sagebrush_steppe_Washington/commons_Sagebrush_Sea_adjacent_to_Seedskadee_National_Wi_e3282a3f.jpg",
                              (430, 600, 560, 700))
    g = np.array(Image.fromarray(small).convert("L"))
    cells.append(np.stack([g] * 3, -1)); labs.append("photo, box 64 px")
    for mode in ["spray", "field", "sprig"]:
        cv = bush_on_ground(sage.SageParams(w=56, h=38, mode=mode), 1)
        cells.append(cv.render(Grey())); labs.append(f'mode="{mode}"')
    H = max(c.shape[0] for c in cells)
    cells = [np.pad(c, ((H - c.shape[0], 0), (0, 0), (0, 0)), constant_values=22) for c in cells]
    save(sp.hstack([sp.upscale(c, 5) for c in cells], labs), "t_sage_models.png")


# ---------------------------------------------------------------- 3. the sprig bush, layer by layer
def _paint_variant(p, seed, open_crest=True):
    real = sage._open_crest
    if not open_crest:
        sage._open_crest = lambda cv, rounds=2: None
    try:
        return bush_on_ground(p, seed)
    finally:
        sage._open_crest = real


def fig_sage_layers():
    base = dict(w=64, h=44, az=35, el=40)
    seed = 3
    steps = [
        ("1 interior only (sprig=0)", dict(sprig=0.0, tips=0.0), False),
        ("2 + sprigs", dict(tips=0.0), False),
        ("3 + open crest", dict(tips=0.0), True),
        ("4 + twig tips (final)", dict(), True),
    ]
    cells, labs = [], []
    # 0: the light field itself, as grey (what the sprigs read their value from)
    p = sage.SageParams(**base)
    rng = np.random.default_rng(seed)
    cv, bx, by, info = sage.paint_bush(p, rng)
    rng = np.random.default_rng(seed)
    f = sage._light_field(info["clumps"], p, rng, cv.w / 2.0, cv.h - 6, 64.0, 44.0,
                          sage.sun_vec(35, 40), cv.h, cv.w)
    img = np.where(f > -9, np.clip(f / 5.4, 0, 1) * 200 + 30, 235).astype(np.uint8)
    img = np.stack([img] * 3, -1)
    pad = bush_on_ground(p, seed).render(Grey()).shape
    img = np.pad(img, ((max(0, pad[0] - img.shape[0]) // 2 + 1,) * 2,
                       (max(0, pad[1] - img.shape[1]) // 2 + 1,) * 2, (0, 0)), constant_values=235)[:pad[0], :pad[1]]
    cells.append(img); labs.append("0 the light field")
    for lab, extra, oc in steps:
        q = dict(base); q.update(extra)
        cells.append(_paint_variant(sage.SageParams(**q), seed, oc).render(Grey())); labs.append(lab)
    save(sp.hstack([sp.upscale(c, 5) for c in cells], labs), "t_sage_layers.png")
    # the same bush in colour, 8x, final
    pal = palette.Palette("summer", "morning")
    cv = bush_on_ground(sage.SageParams(**base), seed)
    save(sp.upscale(cv.render(pal), 8), "t_sage_final_8x.png")


# ---------------------------------------------------------------- 4. sage at every size
def fig_sage_sizes():
    pal = palette.Palette("summer", "morning")
    L = scene.view_light(scene.sun_world(25, 28), 4)
    cells, labs = [], []
    for w in [110, 64, 32, 16, 8, 4, 2]:
        spr, bx, by = scene.sage_sprite(w, w * 0.68, np.random.default_rng(1), L)
        W, H = max(spr.w + 4, 12), max(spr.h + 4, 12)
        cv = sp.Canvas(W, H)
        cv.mat[:] = sp.GROUND; cv.step[:] = 3
        cv.blit(spr, W // 2 - bx, H - 3 - by)
        cells.append(cv.render(pal)); labs.append(f"{w} px")
    Hm = max(c.shape[0] for c in cells)
    k = 3
    out = []
    for c in cells:
        c = np.pad(c, ((Hm - c.shape[0], 0), (0, 0), (0, 0)), mode="edge")
        out.append(sp.upscale(c, k))
    save(sp.hstack(out, labs), "t_sage_sizes.png")


# ---------------------------------------------------------------- 5. the tussock
def fig_tussock():
    pal = palette.Palette("summer", "noon")
    cells, labs = [], []
    for df, lab in [(0.0, "dark_frac=0"), (0.22, "dark_frac=0.22 (used)"), (0.6, "dark_frac=0.6")]:
        W, H = 70, 60
        cv = sp.Canvas(W, H)
        cam = world.Iso(W, H, el=30, ppm=24)
        world.paint_ground(cv, cam, world.GroundParams(seed=3, litter=0.0, stones=0.0, crust=0.0), 1)
        spr, bx, by = grass.paint_tussock(grass.TussockParams(w=30, h=38, az=20, dark_frac=df),
                                          np.random.default_rng(40))
        cv.blit(spr, W // 2 - bx, H - 6 - by)
        cells.append(sp.upscale(cv.render(pal), 5)); labs.append(lab)
    save(sp.hstack(cells, labs), "t_tussock_dark.png")


# ---------------------------------------------------------------- 6. one canvas, many lights
def fig_palette():
    """ONE index canvas, painted once, rendered under five hours and four seasons. Only the
    palette changes. (The cast shadows were painted for one sun, which is the method's price.)"""
    W, H = 160, 100
    cam = world.Persp(W, H, eye=1.6, f=150, hy=30)
    cv = scene.paint_scene(W, H, cam, "summer", "morning", seed=8, sun_az=25, sun_el=25,
                           plants=dict(sage=18, grass=45, backdrop="rim"))
    hours = ["morning", "noon", "golden", "dusk", "overcast"]
    cells = [sp.upscale(cv.render(palette.Palette("summer", h)), 2) for h in hours]
    save(sp.grid(cells, 5, collabels=hours,
                 title="one index canvas (summer, painted once) under five palettes. 2x"), "t_one_canvas.png")
    rows = []
    for s in ["spring", "summer", "autumn", "winter"]:
        rows.append(palette.Palette(s, "golden").swatch(k=10))
    names = ["SKY", "GROUND", "CRUST", "BASALT", "SAGE", "STEM", "GRASS", "BITTER", "NEEDLE", "BARK", "SNOW", "FLOWER", "FORB"]
    save(sp.grid(rows, 4, collabels=["spring", "summer", "autumn", "winter"],
                 title="golden-hour ramps"), "t_ramps.png", 2)


# ---------------------------------------------------------------- 7. the ground, layer by layer
def fig_ground():
    W, H = 110, 70
    pal = palette.Palette("summer", "noon")
    cam = world.Iso(W, H, el=30, ppm=24)
    layers = [
        ("undulation only", dict(crust=0.0, stones=0.0, litter=0.0)),
        ("+ soil crust", dict(crust=0.4, stones=0.0, litter=0.0)),
        ("+ basalt pebbles", dict(crust=0.4, stones=0.3, litter=0.0)),
        ("+ small grasses", dict(crust=0.4, stones=0.3, litter=0.6)),
    ]
    cells, labs = [], []
    for lab, g in layers:
        cv = sp.Canvas(W, H)
        world.paint_ground(cv, cam, world.GroundParams(seed=4, **g), 1)
        cells.append(sp.upscale(cv.render(pal), 3)); labs.append(lab)
    # the same ground with a sage, its cast shadow (the ground two steps down) and contact
    cv = scene.paint_scene(W, H, cam, "summer", "noon", seed=4, sun_az=25, sun_el=40,
                           plants=dict(sage=6, grass=0, pine=0, backdrop=None),
                           ground=dict(crust=0.4, stones=0.3, litter=0.6))
    cells.append(sp.upscale(cv.render(pal), 3)); labs.append("+ plants, shadows, contact")
    save(sp.hstack(cells, labs), "t_ground_layers.png")


# ---------------------------------------------------------------- 8. distance
def fig_distance():
    W, H = 240, 110
    cam = world.Persp(W, H, eye=1.6, f=220, hy=34)
    pal = palette.Palette("summer", "golden")
    kw = dict(seed=21, sun_az=15, sun_el=11, plants=dict(sage=22, grass=45, backdrop="rim"),
              ground=dict(crust=0.4))
    real = scene._far_sea
    scene._far_sea = lambda *a, **k: None
    try:
        off = scene.paint_scene(W, H, cam, "summer", "golden", **kw).render(pal)
    finally:
        scene._far_sea = real
    on = scene.paint_scene(W, H, cam, "summer", "golden", **kw).render(pal)
    nohaze = scene.paint_scene(W, H, cam, "summer", "golden", **kw).render(
        palette.Palette("summer", "golden", bands=10 ** 6))
    save(sp.grid([sp.upscale(nohaze, 2), sp.upscale(off, 2), sp.upscale(on, 2)], 1,
                 labels=["no haze in the palette (bands=1e6)", "haze, but no far sage sea",
                         "haze + far sea (used)"],
                 title="distance: the same world three ways, 2x"), "t_distance.png")


# ---------------------------------------------------------------- 9. bark
def fig_bark():
    """The trunk that reads as ponderosa: the near tree in the parkland hero (painted by
    pine._trunk at ~12 px wide), beside the Dixie Mountain photo's trunk box-downsampled to the
    same width."""
    hero = np.array(Image.open(STUDY + "hero/parkland_morning_1x.png").convert("RGB"))
    crop = hero[0:150, 60:100]
    _, small = box_downsample(NAT + "ponderosa_pine_forest_eastern_Washington/commons_Ponderosa_Pine_Western_Larch_forest_Dixie_Mounta_8beca028.jpg",
                              (520, 350, 660, 1450), width=40)
    big = np.array(Image.open(NAT + "ponderosa_pine_forest_eastern_Washington/commons_Ponderosa_Pine_Western_Larch_forest_Dixie_Mounta_8beca028.jpg").convert("RGB").crop((520, 350, 660, 1450)))
    H = crop.shape[0] * 4
    save(sp.hstack([sp.fit(big, H), sp.upscale(small, 4)[:H], sp.upscale(crop, 4)],
                   ["Dixie Mtn photo", "photo, box 40 px, 4x", "ours (parkland hero), 4x"]), "t_bark.png")


FIGS = dict(downsample=fig_downsample, sage_models=fig_sage_models, sage_layers=fig_sage_layers,
            sage_sizes=fig_sage_sizes, tussock=fig_tussock, palette=fig_palette, ground=fig_ground,
            distance=fig_distance, bark=fig_bark)

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGS):
        FIGS[name]()
