import sys,glob,os
from PIL import Image,ImageDraw
# usage: contact.py out.png cols thumbw files...
out,cols,tw=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]);fs=sys.argv[4:]
th=int(tw*2/3);rows=(len(fs)+cols-1)//cols
S=Image.new('RGB',(cols*tw,rows*(th+12)),(20,20,24));d=ImageDraw.Draw(S)
for i,f in enumerate(fs):
    im=Image.open(f).convert('RGB');im.thumbnail((tw,th))
    x,y=(i%cols)*tw,(i//cols)*(th+12);S.paste(im,(x,y+12));d.text((x+2,y),os.path.basename(f)[:40],fill=(220,220,220))
S.save(out)
