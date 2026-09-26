"""Load Ferrari's Living Worlds scenes as index maps (format from the Canvas Cycle demo)."""
import re, json, numpy as np
REF = "~/work/pixelart-studies/pedagogy/master-copy/ref/lw"
def load(name):
    t = open(f"{REF}/{name}.json").read()
    t = re.sub(r'([{,])\s*([A-Za-z_]+)\s*:', r'\1"\2":', t).replace("'", '"')
    d = json.loads(t)
    d["pal"] = np.array(d["colors"], dtype=np.uint8)
    d["idx"] = np.array(d["pixels"], dtype=np.int32).reshape(d["height"], d["width"])
    return d
def lum(rgb):
    rgb = np.asarray(rgb, float)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
