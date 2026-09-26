import sys; sys.path.insert(0,'src'); from util import *
a=load_ref(); s=2; g=up(a,s).copy()
im=Image.fromarray(g); d=ImageDraw.Draw(im)
f=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',11)
for x in range(0,a.shape[1],20):
    d.line([(x*s,0),(x*s,g.shape[0])],fill=(0,255,255) if x%100==0 else (0,120,120),width=1)
    if x%40==0: d.text((x*s+2,2),str(x),fill=(255,255,0),font=f)
for y in range(0,a.shape[0],20):
    d.line([(0,y*s),(g.shape[1],y*s)],fill=(0,255,255) if y%100==0 else (0,120,120),width=1)
    d.text((2,y*s+2),str(y),fill=(255,255,0),font=f)
im.save('img/01_grid.png')
