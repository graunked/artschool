import sys, time; sys.path.insert(0,'.')
import numpy as np
from sea.place import load
from sea.waves import swell, eikonal
from PIL import Image
P=load('poa'); t=time.time()
F=swell(P["depth"],Hs=1.57,T=8.3,dir_from=281,smooth=8.0,alpha=0.35,gamma=0.8)
te,kx,ky=eikonal(P['depth'],T=8.3,dir_from=281)
print('time',time.time()-t)
F['t']=te;F['kx']=kx;F['ky']=ky
np.savez_compressed('../cache/poa_waves.npz',**{k:v for k,v in F.items() if isinstance(v,np.ndarray)})
H=F['H'];Q=F['Q'];ph=2*np.pi*te/8.3
rgb=np.zeros(H.shape+(3,))
rgb[...,2]=np.clip(H/1.5,0,1); rgb[...,0]=np.clip(Q*3,0,1); rgb[...,1]=0.35*(np.cos(ph)>0.92)
rgb[F['land']]=0.5
Image.fromarray((rgb*255).astype(np.uint8)).save('../notes/poa_waves.png')
