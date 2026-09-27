"""Read the olycoast/sanjuan 1 m pack tiles (float32 1000x1000, NaN nodata, EPSG:26910, NAVD88 m)."""
import numpy as np, zstandard, os, pyproj
ROOT = os.path.expanduser('~/work/voxsim-data/places')
_T = pyproj.Transformer.from_crs(4326, 26910, always_xy=True)
def lonlat_to_utm(lon, lat): return _T.transform(lon, lat)
def tile(place, layer, tx, ty, dtype=np.float32):
    f = f'{ROOT}/{place}/pack/{layer}/{tx}_{ty}.zst'
    if not os.path.exists(f): return None
    b = zstandard.ZstdDecompressor().decompress(open(f, 'rb').read(), max_output_size=1 << 28)
    a = np.frombuffer(b, dtype=dtype)
    n = int(round(np.sqrt(a.size))); return a.reshape(n, n)
def window(place, layer, x0, y0, x1, y1, dtype=np.float32):
    """Mosaic [x0,x1) x [y0,y1) in UTM metres; returns array with row 0 = north (y1)."""
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    W, H = x1 - x0, y1 - y0
    out = np.full((H, W), np.nan, np.float32)
    for tx in range(int(x0 // 1000), int((x1 - 1) // 1000) + 1):
        for ty in range(int(y0 // 1000), int((y1 - 1) // 1000) + 1):
            t = tile(place, layer, tx, ty, dtype)
            if t is None: continue
            # guess orientation: row 0 = north edge of tile
            ox, oy = tx * 1000, ty * 1000
            ax0, ax1 = max(x0, ox), min(x1, ox + 1000); ay0, ay1 = max(y0, oy), min(y1, oy + 1000)
            if ax0 >= ax1 or ay0 >= ay1: continue
            sub = t[int(oy + 1000 - ay1):int(oy + 1000 - ay0), int(ax0 - ox):int(ax1 - ox)]
            out[int(y1 - ay1):int(y1 - ay0), int(ax0 - x0):int(ax1 - x0)] = sub
    return out
