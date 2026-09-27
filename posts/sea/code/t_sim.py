import sys, time; sys.path.insert(0,'.')
import numpy as np
from sea.place import load
from sea.sim import SeaState
from PIL import Image
P=load('poa'); Wv=dict(np.load('../cache/poa_waves.npz'))
Wv['land']=Wv['land'].astype(bool)
t=time.time()
S=SeaState(P['depth'],P['z'],Wv,8.3,current=(0.05,-0.25),kelp_amount=0.25,n_periods=60,steps_per_T=8).run(verbose=True)
print('sim',time.time()-t)
np.savez_compressed('../cache/poa_parts.npz',**S.parts)
np.savez_compressed('../cache/poa_state.npz',W=S.W,F=S.F,A=S.A,q=S.q,Hinst=S.Hinst,brk=S.brk,kelp=S.kelp,kelp_rho=S.kelp_rho,rock=S.rock,u=S.u,v=S.v)
rgb=np.zeros(S.d.shape+(3,))
dd=np.clip(S.d/20,0,1)
rgb[...]=np.stack([0.1+0.2*(1-dd),0.3+0.3*(1-dd),0.5+0.1*(1-dd)],-1)
rgb[S.kelp>0.3]=[0.15,0.12,0.2]
aer=np.clip(S.A,0,1)[...,None]; rgb=rgb*(1-0.4*aer)+0.4*aer*np.array([0.5,0.8,0.75])
f=np.clip(S.F*1.5,0,1)[...,None]; rgb=rgb*(1-f)+f
wv=np.clip(S.W*1.5,0,1)[...,None]; rgb=rgb*(1-wv)+wv
rgb[S.land]=[0.45,0.4,0.35]
Image.fromarray((rgb*255).astype(np.uint8)).save('../notes/poa_state.png')
