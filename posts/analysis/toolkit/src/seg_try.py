import sys; sys.path.insert(0,'src'); from util import *
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
a=load_ref(); lab=rgb2lab(a)
bl=np.stack([gaussian_filter(lab[...,i],3) for i in range(3)],-1)
H,W=a.shape[:2]; yy,xx=np.mgrid[0:H,0:W]
F=np.concatenate([bl.reshape(-1,3),(yy.reshape(-1,1)*0.12),(xx.reshape(-1,1)*0.03)],1)
km=KMeans(16,n_init=4,random_state=0).fit(F[::7]); lab_=km.predict(F).reshape(H,W)
rng=np.random.default_rng(1); cols=rng.integers(40,255,(16,3))
out=cols[lab_].astype(np.uint8)
im=Image.fromarray(up(out,2)); d=ImageDraw.Draw(im); f=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',16)
for k in range(16):
    ys,xs=np.nonzero(lab_==k); d.text((int(np.median(xs))*2,int(np.median(ys))*2),str(k),fill=(0,0,0),font=f)
im.save('img/tmp_kmeans.png'); np.save('out/km.npy',lab_)
