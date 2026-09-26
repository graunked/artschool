import sys; sys.path.insert(0,'src'); from util import *
from scipy.ndimage import gaussian_filter
from skimage import measure
a=load_ref(); P=np.load('out/planes.npy'); L=lum(a)
s=3; img=Image.fromarray(up((0.55*a+0.45*np.stack([L*2.55]*3,-1)).astype(np.uint8),s)); d=ImageDraw.Draw(img)
Lb=gaussian_filter(L,4); gy,gx=np.gradient(gaussian_filter(L,7))
cols={1:(255,120,255),4:(120,255,120),5:(255,255,0),7:(255,160,0),8:(255,80,0),9:(0,255,255),6:(160,160,255)}
for pi,c in cols.items():
    m=P==pi
    Lm=np.where(m,Lb,np.nan)
    # isophotes of the squinted value within the mass
    lv=np.nanpercentile(Lm,[20,40,60,80])
    for v in lv:
        for cn in measure.find_contours(np.nan_to_num(Lm,nan=-99),v):
            pts=[(p[1]*s,p[0]*s) for p in cn[::2]]
            if len(pts)>6: d.line(pts,fill=c,width=1)
    # brightness gradient arrows (points toward the lit side of the form)
    for y in range(8,a.shape[0],18):
        for x in range(8,a.shape[1],18):
            if m[y,x] and m[max(0,y-6):y+6,max(0,x-6):x+6].mean()>0.9:
                vx,vy=gx[y,x],gy[y,x]; n=np.hypot(vx,vy)
                if n<0.15: d.ellipse([x*s-2,y*s-2,x*s+2,y*s+2],outline=c); continue
                k=min(9,3+n*6)/n; x2,y2=x+vx*k,y+vy*k
                d.line([(x*s,y*s),(x2*s,y2*s)],fill=c,width=2); d.ellipse([x2*s-3,y2*s-3,x2*s+3,y2*s+3],fill=c)
img.save('img/07_form.png')
img.crop((180*s,190*s,634*s,330*s)).save('img/07_form_mid.png')
