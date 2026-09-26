import sys; sys.path.insert(0,'src'); from util import *
a=load_ref(); x,y,w,h,s,name=sys.argv[1:7]
x,y,w,h,s=map(int,(x,y,w,h,s)); save(a[y:y+h,x:x+w],name,s)
