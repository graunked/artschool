"""Real places from the voxsim-data packs (1 m topo-bathy lidar)."""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dem import window, lonlat_to_utm
PLACES = {
    # name: (pack, NW origin UTM x, y, box cols0,rows0,cols1,rows1 in that frame, tide NAVD m at photo)
    'poa': ('olycoast', 372347, 5346016, (350, 250, 1650, 1650)),
    'alava': ('olycoast', 370145, 5337716, (100, 300, 1500, 1700)),
    'kala': ('olycoast', 395977, 5274211, (0, 300, 900, 1500)),
}
def load(name, tide=0.45):
    pack, X0, Y1, (c0, r0, c1, r1) = PLACES[name]
    z = window(pack, 'elev', X0 + c0, Y1 - r1, X0 + c1, Y1 - r0)
    return dict(z=z, depth=tide - z, tide=tide, origin=(X0 + c0, Y1 - r0), frame=(c0, r0))
