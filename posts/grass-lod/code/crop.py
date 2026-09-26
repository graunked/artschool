import sys
from PIL import Image
# usage: crop.py src x y w h scale out
s,x,y,w,h,k,o=sys.argv[1:8]
x,y,w,h,k=map(int,(x,y,w,h,k))
im=Image.open(s).convert('RGB').crop((x,y,x+w,y+h))
im.resize((w*k,h*k),Image.NEAREST).save(o)
