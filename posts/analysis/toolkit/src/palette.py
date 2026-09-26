import sys; sys.path.insert(0,'src'); from util import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from planes_names import NAMES
a=load_ref(); P=np.load('out/planes.npy')
flat=a.reshape(-1,3); u,inv,cnt=np.unique(flat,axis=0,return_inverse=True,return_counts=True); inv=inv.ravel()
pl=P.ravel()
M=np.zeros((len(u),10))
np.add.at(M,(inv,pl),1)
lab=rgb2lab(u.astype(float)); L=lab[:,0]; hue=np.degrees(np.arctan2(lab[:,2],lab[:,1]))%360; C=np.hypot(lab[:,1],lab[:,2])
# material groups (merge planes)
groups={'sky':[0],'mountain (rock+snow)':[1],'conifers':[2,3],'grass lit (left)':[7,8],'grass shade / banks':[4,9],'island':[5],'water-only':[6]}
home=np.argmax(M,1)
own={}
for g,ps in groups.items():
    share=M[:,ps].sum(1)/M.sum(1)
    if g=='water-only': sel=(M[:,6]/M.sum(1)>0.9)
    else: sel=(share>0.45)&(M[:,6]/M.sum(1)<=0.9)
    own[g]=np.flatnonzero(sel)
# assign each colour to a single group (highest share excl water), ignore duplicates
assigned={}
for i in range(len(u)):
    best=None
    for g,ps in groups.items():
        if g=='water-only': continue
        s=M[i,ps].sum()
        if best is None or s>best[1]: best=(g,s)
    if M[i,6]/M[i].sum()>0.9: best=('water-only',0)
    assigned.setdefault(best[0],[]).append(i)
sw=18; rows=[]
for g in groups:
    ids=sorted(assigned.get(g,[]),key=lambda i:L[i])
    tot=cnt[ids].sum()
    strip=np.full((sw+20,max(1,len(ids))*sw+260,3),255,np.uint8)
    for k,i in enumerate(ids):
        h=int(np.clip(4+np.log1p(cnt[i])*1.5,4,sw))
        strip[20+sw-h:20+sw,260+k*sw:260+(k+1)*sw-1]=u[i]
    strip=label(strip,f'{g} ({len(ids)} cols)',(2,20),fill=(0,0,0),bg=(255,255,255))
    rows.append(strip)
W=max(r.shape[1] for r in rows); rows=[np.pad(r,((0,4),(0,W-r.shape[1]),(0,0)),constant_values=255) for r in rows]
Image.fromarray(np.concatenate(rows,0)).save('img/10_palette_groups.png')
# chart: L vs hue per group, point size by count
fig,ax=plt.subplots(1,2,figsize=(14,5.5))
for g in groups:
    ids=assigned.get(g,[])
    if not ids: continue
    ids=np.array(ids)
    ax[0].scatter(lab[ids,1],lab[ids,2],s=np.sqrt(cnt[ids])*1.2,c=u[ids]/255,edgecolors='k',linewidths=.3)
    ax[1].scatter(C[ids],L[ids],s=np.sqrt(cnt[ids])*1.2,c=u[ids]/255,edgecolors='k',linewidths=.3)
ax[0].set_xlabel('a*  (green - / red +)'); ax[0].set_ylabel('b*  (blue - / yellow +)'); ax[0].axhline(0,c='gray',lw=.5); ax[0].axvline(0,c='gray',lw=.5); ax[0].set_title('all 172 colours in a*b* (size ~ usage)')
ax[1].set_xlabel('chroma'); ax[1].set_ylabel('L*'); ax[1].set_title('value vs chroma')
plt.tight_layout(); plt.savefig('img/10_palette_ab.png',dpi=80)
# coverage: how many colours cover 90% of each plane
for i,n in enumerate(NAMES):
    c=np.sort(M[:,i])[::-1]; cs=np.cumsum(c)/c.sum(); print(f'{n:18s} colours for 80%: {np.searchsorted(cs,.8)+1:3d}  90%: {np.searchsorted(cs,.9)+1:3d}  95%: {np.searchsorted(cs,.95)+1:3d}')
# print the dominant ramps
for g in ['grass lit (left)','grass shade / banks','conifers','mountain (rock+snow)','sky']:
    ids=sorted(assigned[g],key=lambda i:-cnt[i])[:10]; ids=sorted(ids,key=lambda i:L[i])
    print(g,[ (tuple(u[i]),int(L[i]),int(hue[i]),int(C[i])) for i in ids])
