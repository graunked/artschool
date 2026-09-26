import sys; sys.path.insert(0,'src'); from util import *
from scipy.ndimage import gaussian_filter, median_filter, map_coordinates, binary_dilation
from skimage.feature import canny
def edge_widths(a,SIG=3.5):
    L=lum(a); Lm=gaussian_filter(median_filter(L,3),0.7)
    E=canny(median_filter(L,3),sigma=SIG,low_threshold=4,high_threshold=8)
    gy,gx=np.gradient(gaussian_filter(Lm,2.0))
    ys,xs=np.nonzero(E); n=np.hypot(gx[ys,xs],gy[ys,xs])+1e-9; ux,uy=gx[ys,xs]/n,gy[ys,xs]/n
    t=np.linspace(-8,8,33)
    py=ys[:,None]+uy[:,None]*t[None]; px=xs[:,None]+ux[:,None]*t[None]
    prof=map_coordinates(Lm,[py,px],order=1,mode='nearest')
    lo=np.percentile(prof,10,axis=1); hi=np.percentile(prof,90,axis=1); amp=hi-lo
    p=(prof-lo[:,None])/(amp[:,None]+1e-9)
    # rise width: distance between where profile crosses .2 and .8 nearest centre
    w=np.zeros(len(ys))
    for i in range(len(ys)):
        c=16; q=p[i]
        a20=np.argmin(np.where(np.abs(np.arange(33)-c)<=16,np.abs(q-0.2),9)); a80=np.argmin(np.abs(q-0.8))
        w[i]=abs(t[a80]-t[a20])
    return ys,xs,w,amp,L
def draw(a,prefix,scale):
    ys,xs,w,amp,L=edge_widths(a)
    base=(a*0.35+150).clip(0,255)
    keep=amp>18
    out=base.copy()
    col=np.zeros((len(ys),3))
    col[w<=2.0]=(230,20,20); col[(w>2.0)&(w<=4.5)]=(250,200,0); col[w>4.5]=(30,120,255)
    out[ys[keep],xs[keep]]=col[keep]
    save(out.astype(np.uint8),f'{prefix}_edges.png' if not prefix.endswith('mine') else f'{prefix}.png',scale)
    return ys[keep],xs[keep],w[keep],amp[keep]
if __name__=='__main__':
    a=load_ref(); P=np.load('out/planes.npy')
    from planes_names import NAMES
    ys,xs,w,amp=draw(a,'08',2)
    print('edge px',len(w),'hard(<=2) %.2f mid %.2f soft(>4.5) %.2f'%((w<=2).mean(),((w>2)&(w<=4.5)).mean(),(w>4.5).mean()))
    for i,n in enumerate(NAMES):
        m=P[ys,xs]==i
        if m.sum()>30: print(f'{n:18s} n={m.sum():5d} hard {np.mean(w[m]<=2):.2f} soft {np.mean(w[m]>4.5):.2f} median amp {np.median(amp[m]):.1f}')
