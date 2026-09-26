import sys; sys.path.insert(0,'src'); from util import *
from scipy.ndimage import gaussian_filter, uniform_filter1d
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
a=load_ref(); P=np.load('out/planes.npy'); L=lum(a)
Lh=uniform_filter1d(gaussian_filter(L,1.5),15,axis=1)  # average across 15 px horizontally
fig,ax=plt.subplots(1,4,figsize=(15,3.8))
specs=[('island x=340',340,255,302,5),('far hummock x=560',560,212,262,4),('left bank x=60',60,238,469,None),('right bank x=560',560,288,469,9)]
for axx,(t,x,y0,y1,pi) in zip(ax,specs):
    ys=np.arange(y0,y1); axx.plot(Lh[ys,x],ys,'k'); axx.invert_yaxis(); axx.set_title(t); axx.set_xlabel('L* (15px avg)'); axx.set_ylabel('y (row)')
    axx.set_xlim(0,70)
plt.tight_layout(); plt.savefig('img/07_profiles.png',dpi=80)
# horizontal profile along island top & right bank near water
for t,y,x0,x1 in [('island row 272',272,262,415),('left meadow row 270',270,0,200)]:
    print(t, np.round(Lh[y,x0:x1:15],0))
