import sys
from PIL import Image
# crop.py src out cx cy w h [scale]  (coords in FULL-res pixels)
s,o=sys.argv[1],sys.argv[2];cx,cy,w,h=map(int,sys.argv[3:7]);sc=float(sys.argv[7]) if len(sys.argv)>7 else 1
im=Image.open(s).convert('RGB');c=im.crop((cx-w//2,cy-h//2,cx+w//2,cy+h//2))
if sc!=1:c=c.resize((int(c.width*sc),int(c.height*sc)),Image.LANCZOS)
c.save(o)
