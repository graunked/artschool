import sys; sys.path.insert(0,'src'); from util import *
from scipy.ndimage import gaussian_filter
a=load_ref(); lab=rgb2lab(a); L=lab[...,0]; Lb=gaussian_filter(L,1.5)
# find mirror axis y0 (possibly half-integer) and vertical scale s: water row y maps to source row y0-(y-y0)*s
best=None
xs=slice(250,420)
for y0 in np.arange(225,262,0.5):
  for s in (0.8,0.85,0.9,0.95,1.0,1.05,1.1):
    ys=np.arange(300,400); src=np.round(y0-(ys-y0)*s).astype(int)
    if src.min()<0: continue
    e=np.mean(np.abs(Lb[ys][:,xs]-Lb[src][:,xs]))
    if best is None or e<best[0]: best=(e,y0,s)
print('best axis',best)
e,y0,s=best
ys=np.arange(300,400); src=np.round(y0-(ys-y0)*s).astype(int)
dL=Lb[ys][:,xs]-Lb[src][:,xs]
print('mean dL refl-src %.2f'%dL.mean())
for lo,hi in [(0,30),(30,55),(55,100)]:
    m=(Lb[src][:,xs]>=lo)&(Lb[src][:,xs]<hi); print(f' src L {lo}-{hi}: refl-src dL={dL[m].mean():.2f}  (n={m.sum()})')
labb=np.stack([gaussian_filter(lab[...,i],1.5) for i in range(3)],-1)
d=labb[ys][:,xs]-labb[src][:,xs]; print('mean dLab',d.reshape(-1,3).mean(0).round(2))
# side by side image: source (flipped) vs reflection
flip=a[src][:,250:420]; refl=a[ys][:,250:420]
save(np.concatenate([flip,np.full((100,4,3),255,np.uint8),refl],1),'06_reflection_compare.png',4)
