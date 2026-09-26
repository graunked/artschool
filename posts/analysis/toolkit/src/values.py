import sys; sys.path.insert(0,'src'); from util import *
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
def analyse(a,planes,names,tag,prefix):
    L=lum(a); Lb=gaussian_filter(L,2.5)
    g=(L/100*255); gb=(Lb/100*255)
    save(np.concatenate([g,np.full((a.shape[0],6),255),gb],1),f'{prefix}_grey.png',2 if a.shape[1]>400 else 3)
    res={}
    rows=[]
    for k in (2,3,5):
        km=KMeans(k,n_init=6,random_state=0).fit(Lb.reshape(-1,1)[::5]); c=np.sort(km.cluster_centers_.ravel())
        th=(c[1:]+c[:-1])/2; q=np.digitize(Lb,th); lv=np.linspace(0,255,k) if k>2 else np.array([30,235])
        if k==3: lv=np.array([25,130,240])
        rows.append(lv[q]); res[k]=(th,c)
        np.save(f'out/{prefix}_notan{k}.npy',q)
    sep=np.full((a.shape[0],6),255)
    save(np.concatenate([rows[0],sep,rows[1],sep,rows[2]],1),f'{prefix}_notan.png',1 if a.shape[1]>400 else 2)
    # per-plane stats
    stats=[]
    for i,n in enumerate(names):
        m=planes==i
        if m.sum()<50: stats.append(None); continue
        stats.append((n,np.percentile(Lb[m],[5,25,50,75,95]),L[m].std(),m.mean()))
    fig,ax=plt.subplots(1,2,figsize=(13,4.6))
    data=[Lb[planes==i] for i in range(len(names)) if (planes==i).sum()>=50]
    nm=[n for i,n in enumerate(names) if (planes==i).sum()>=50]
    ax[0].boxplot(data,whis=(5,95),showfliers=False,vert=False); ax[0].set_yticks(range(1,len(nm)+1),nm); ax[0].set_xlabel('L* (squint-blurred)'); ax[0].set_xlim(0,100); ax[0].set_title(f'{tag}: value range per plane')
    for t in res[3][0]: ax[0].axvline(t,color='r',ls=':')
    ax[1].hist(L.ravel(),bins=100,range=(0,100),color='k',alpha=.5,label='pixel L*'); ax[1].hist(Lb.ravel(),bins=100,range=(0,100),color='r',alpha=.4,label='blurred L*'); ax[1].legend(); ax[1].set_title('value histogram (red dotted = 3-notan cuts)')
    for t in res[3][0]: ax[1].axvline(t,color='r',ls=':')
    plt.tight_layout(); plt.savefig(f'img/{prefix}_valueplanes.png',dpi=80); plt.close()
    # flat value plan: each plane filled with its median blurred L
    flat=np.zeros_like(L)
    for i in range(len(names)):
        m=planes==i
        if m.any(): flat[m]=np.median(Lb[m])
    save(flat/100*255,f'{prefix}_valueplan.png',1 if a.shape[1]>400 else 2)
    return res,stats
if __name__=='__main__':
    a=load_ref(); planes=np.load('out/planes.npy')
    from planes_names import NAMES
    res,stats=analyse(a,planes,NAMES,'Ferrari','03')
    for k,(th,c) in res.items(): print(k,'centers',c.round(1),'cuts',th.round(1))
    for s in stats:
        if s: print(f'{s[0]:20s} p5/25/50/75/95={s[1].round(0)} pixstd={s[2]:.1f} area={s[3]:.3f}')
