import sys; sys.path.insert(0,'src'); from util import *
from scipy.ndimage import label as cclabel, find_objects
def colour_index(a):
    flat=a.reshape(-1,3); u,inv=np.unique(flat,axis=0,return_inverse=True); return inv.reshape(a.shape[:2]),u
def cluster_stats(a,mask):
    """connected same-colour runs (4-conn) inside mask: size, h, w, aspect (h/w), vertical fraction"""
    idx,u=colour_index(a)
    S=[];Hs=[];Ws=[];
    for c in np.unique(idx[mask]):
        m=(idx==c)&mask
        lab,n=cclabel(m)
        if n==0: continue
        sizes=np.bincount(lab.ravel())[1:]
        for sl,sz in zip(find_objects(lab),sizes):
            S.append(sz); Hs.append(sl[0].stop-sl[0].start); Ws.append(sl[1].stop-sl[1].start)
    S=np.array(S);Hs=np.array(Hs);Ws=np.array(Ws)
    return dict(n=len(S),size_med=np.median(S),size_mean=S.mean(),size_p90=np.percentile(S,90),
                single=(S==1).mean(),vert=np.mean(Hs>Ws),horiz=np.mean(Ws>Hs),aspect_med=np.median(Hs/Ws),
                h_p90=np.percentile(Hs,90), w_p90=np.percentile(Ws,90),
                colours=len(np.unique(idx[mask])), per100=len(S)/mask.sum()*100)
def run_lengths(a,mask,axis):
    """mean run length of identical colour along axis inside mask"""
    idx,_=colour_index(a); b=idx if axis==1 else idx.T; mk=mask if axis==1 else mask.T
    runs=[]
    for r,m in zip(b,mk):
        if m.sum()<4: continue
        v=r[m]; ch=np.flatnonzero(np.diff(v)!=0); pts=np.r_[-1,ch,len(v)-1]; runs+=list(np.diff(pts))
    return np.mean(runs)
if __name__=='__main__':
    a=load_ref(); P=np.load('out/planes.npy')
    from planes_names import NAMES
    import json; out={}
    for i,n in enumerate(NAMES):
        m=P==i; s=cluster_stats(a,m); s['runH']=run_lengths(a,m,1); s['runV']=run_lengths(a,m,0)
        out[n]={k:float(v) for k,v in s.items()}
        print(f"{n:18s} cols={s['colours']:3.0f} clusters/100px={s['per100']:5.1f} size med={s['size_med']:.0f} mean={s['size_mean']:5.1f} p90={s['size_p90']:4.0f} single={s['single']:.2f} vert={s['vert']:.2f} horiz={s['horiz']:.2f} runH={s['runH']:.2f} runV={s['runV']:.2f}")
    json.dump(out,open('out/ref_cluster_stats.json','w'),indent=1)
