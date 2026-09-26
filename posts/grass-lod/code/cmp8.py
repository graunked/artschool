"""side-by-side 8x crops: mine vs Ferrari"""
import sys
from PIL import Image
a,ax,ay,b,bx,by,w,h,out=sys.argv[1:10]
ax,ay,bx,by,w,h=map(int,(ax,ay,bx,by,w,h))
A=Image.open(a).convert('RGB'); B=Image.open(b).convert('RGB')
k=8
ca=A.crop((ax,ay,ax+w,ay+h)).resize((w*k,h*k),Image.NEAREST)
cb=B.crop((bx,by,bx+w,by+h)).resize((w*k,h*k),Image.NEAREST)
S=Image.new('RGB',(w*k*2+8,h*k),'white'); S.paste(ca,(0,0)); S.paste(cb,(w*k+8,0)); S.save(out)
