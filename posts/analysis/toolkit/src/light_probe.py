import sys; sys.path.insert(0,'src'); from util import *
a=load_ref(); P=np.load('out/planes.npy'); lab=rgb2lab(a); L=lab[...,0]
sky=P==0
# sky gradient: by row and by column
for y in range(0,240,20):
    m=sky[y:y+10]; 
    if m.sum(): print('row',y,'L=%.1f a=%.1f b=%.1f'%tuple(lab[y:y+10][m].mean(0)))
H,W=L.shape
for x0 in range(0,W,80):
    m=sky[60:200,x0:x0+80]
    if m.sum()>100: print('col',x0,'L=%.1f a=%.1f b=%.1f'%tuple(lab[60:200,x0:x0+80][m].mean(0)))
# which side of dark are the warm lit pixels, per plane
for pi,name in [(3,'framing trees'),(2,'far forest'),(1,'mountain'),(4,'far banks'),(5,'island')]:
    m=P==pi; lit=m&(L>np.percentile(L[m],70)); dark=m&(L<=np.percentile(L[m],40)+0.5)
    left=np.zeros_like(lit); right=np.zeros_like(lit); upn=np.zeros_like(lit); dn=np.zeros_like(lit)
    # for lit pixels, is the dark neighbour at distance 1..3 to the right (=> lit on left side of form)?
    r=l=u=d=0
    for k in (1,2,3):
        r+=(lit[:,:-k]&dark[:,k:]).sum(); l+=(lit[:,k:]&dark[:,:-k]).sum()
        d+=(lit[:-k]&dark[k:]).sum(); u+=(lit[k:]&dark[:-k]).sum()
    print(f'{name:14s} lit->dark: right {r} left {l} below {d} above {u}  => lit edge faces {"LEFT" if r>l else "RIGHT"}, {"UP" if d>u else "DOWN"}')
# water vs source brightness: reflected sky
w=P==6
print('water pixels L>55 mean', L[w&(L>55)].mean(), ' sky mean', L[sky].mean())
