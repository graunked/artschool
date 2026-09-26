import lw, numpy as np, sys
from PIL import Image
from scipy import ndimage as ndi
def Lstar(rgb):
    c = rgb.astype(float)/255; c = np.where(c>0.04045, ((c+0.055)/1.055)**2.4, c/12.92)
    Y = c@np.array([0.2126,0.7152,0.0722]); return np.where(Y>0.008856, 116*np.cbrt(Y)-16, 903.3*Y)
def squint(rgb, s=3): return ndi.gaussian_filter(Lstar(rgb), s)
if __name__=='__main__':
    for n in sys.argv[1:]:
        d=lw.load(n); a=lw.rgb(d); L=squint(a,3)
        q=np.digitize(L,np.percentile(L[240:],[25,50,75]))  # quartiles of the lower half
        g=(np.array([30,90,160,230])[q]).astype(np.uint8)
        out=np.concatenate([a,np.stack([g]*3,-1)],1)
        lw.save(out,f'../ref/mass_{n}.png',1)
