import numpy as np, sys
from PIL import Image, ImageDraw
from dem import *
from hill import hillshade
x0,y1=372347,5346016
def crop(c0,r0,c1,r1,out,sc=1,lvl=(-3,0,2)):
    a=window('olycoast','elev',x0+c0,y1-r1,x0+c1,y1-r0)
    h=hillshade(np.nan_to_num(a,nan=0))
    rgb=np.stack([h*0.5+0.2]*3,-1)
    rgb[a<lvl[0]]*=np.array([0.5,0.7,1.0])
    m=(a>=lvl[0])&(a<lvl[1]); rgb[m]=rgb[m]*0.5+np.array([0.1,0.5,0.4])
    m=(a>=lvl[1])&(a<lvl[2]); rgb[m]=rgb[m]*0.5+np.array([0.5,0.4,0.1])
    m=a>=lvl[2]; rgb[m]=rgb[m]*0.6+0.4
    im=Image.fromarray((np.clip(rgb,0,1)*255).astype(np.uint8)).resize((int(a.shape[1]*sc),int(a.shape[0]*sc)),Image.NEAREST)
    d=ImageDraw.Draw(im)
    for g in range((c0//50+1)*50,c1,50): X=(g-c0)*sc; d.line([(X,0),(X,im.height)],fill=(255,0,0) if g%100==0 else (120,0,0)); d.text((X+2,2),str(g),fill=(255,255,0))
    for g in range((r0//50+1)*50,r1,50): Y=(g-r0)*sc; d.line([(0,Y),(im.width,Y)],fill=(255,0,0) if g%100==0 else (120,0,0)); d.text((2,Y+2),str(g),fill=(255,255,0))
    im.save(out)
if __name__=='__main__':
    a=list(map(int,sys.argv[1:5])); crop(*a,sys.argv[5],float(sys.argv[6]) if len(sys.argv)>6 else 1)
