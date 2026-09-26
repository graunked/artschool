import sys; sys.path.insert(0,'src'); from util import *
def dither_stats(a,P,names):
    i00=a[:-1,:-1];i01=a[:-1,1:];i10=a[1:,:-1];i11=a[1:,1:]
    eq=lambda p,q: np.all(p==q,axis=2)
    checker=eq(i00,i11)&eq(i01,i10)&~eq(i00,i01)
    flat=eq(i00,i01)&eq(i00,i10)&eq(i00,i11)
    vstripe=eq(i00,i10)&eq(i01,i11)&~eq(i00,i01)
    hstripe=eq(i00,i01)&eq(i10,i11)&~eq(i00,i10)
    Pc=P[:-1,:-1]; res={}
    for i,n in enumerate(names):
        m=Pc==i
        if m.sum()<50: continue
        res[n]=dict(checker=checker[m].mean(),flat=flat[m].mean(),vstripe=vstripe[m].mean(),hstripe=hstripe[m].mean())
    return res,checker
if __name__=='__main__':
    from planes_names import NAMES
    a=load_ref(); P=np.load('out/planes.npy'); res,ch=dither_stats(a,P,NAMES)
    for n,r in res.items(): print(f"{n:18s} "+' '.join(f'{k}={v:.3f}' for k,v in r.items()))
    out=(a*0.3+120).astype(np.uint8); out[:-1,:-1][ch]=(255,0,200); save(out,'11_checker_map.png',2)
