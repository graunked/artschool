import sys; sys.path.insert(0,'src'); from util import *
from scipy.ndimage import gaussian_filter, laplace
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
def depth_stats(a,P,order):
    lab=rgb2lab(a); L=lab[...,0]; C=np.hypot(lab[...,1],lab[...,2])
    Lb=gaussian_filter(L,2.5); hf=np.abs(L-gaussian_filter(L,1.0))
    rows=[]
    for name,ids in order:
        m=np.isin(P,ids)
        rows.append(dict(name=name,L=np.median(Lb[m]),Lrange=np.percentile(Lb[m],95)-np.percentile(Lb[m],5),
            C=np.median(C[m]),b=np.median(lab[...,2][m]),a=np.median(lab[...,1][m]),hf=hf[m].mean(),darkest=np.percentile(L[m],3),lightest=np.percentile(L[m],97)))
    return rows
ORDER=[('near: fore banks',[8,9]),('near: framing trees',[3]),('mid: left meadow',[7]),('mid: island',[5]),('far: banks',[4]),('far: forest',[2]),('distant: mountain',[1]),('sky',[0])]
def plot(rows,fn,title):
    fig,ax=plt.subplots(1,4,figsize=(16,3.6)); x=np.arange(len(rows)); nm=[r['name'] for r in rows]
    ax[0].plot(x,[r['L'] for r in rows],'ko-',label='median L*'); ax[0].fill_between(x,[r['darkest'] for r in rows],[r['lightest'] for r in rows],alpha=.2,label='3-97% pixel L*'); ax[0].set_title('value & value range'); ax[0].legend(fontsize=7)
    ax[1].plot(x,[r['Lrange'] for r in rows],'ro-'); ax[1].set_title('squint contrast (p95-p5)')
    ax[2].plot(x,[r['C'] for r in rows],'go-',label='chroma'); ax[2].plot(x,[r['b'] for r in rows],'y^-',label='b* (warm +)'); ax[2].plot(x,[r['a'] for r in rows],'m^-',label='a* (pink +)'); ax[2].legend(fontsize=7); ax[2].set_title('colour')
    ax[3].plot(x,[r['hf'] for r in rows],'bo-'); ax[3].set_title('detail (hi-freq energy)')
    for axx in ax: axx.set_xticks(x,nm,rotation=60,fontsize=7)
    fig.suptitle(title); plt.tight_layout(); plt.savefig(fn,dpi=80); plt.close()
if __name__=='__main__':
    a=load_ref(); P=np.load('out/planes.npy'); rows=depth_stats(a,P,ORDER)
    for r in rows: print({k:(round(v,1) if not isinstance(v,str) else v) for k,v in r.items()})
    plot(rows,'img/12_depth.png','Ferrari: depth cues, near -> far')
