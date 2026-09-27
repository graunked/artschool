import numpy as np, sys, json; sys.path.insert(0,'.')
from lab import srgb_to_lab
from PIL import Image
O='~/work/voxsim-data/places/olycoast/obliques/'
S='~/work/voxsim-data/places/sanjuan/obliques/'
def samp(f, rects, vw=1600):
    im=Image.open(f).convert('RGB'); s=im.width/vw; a=np.asarray(im).astype(float)
    out={}
    for name,(x0,y0,x1,y1) in rects.items():
        p=a[int(y0*s):int(y1*s),int(x0*s):int(x1*s)].reshape(-1,3); L=srgb_to_lab(p)
        med=np.median(L,0); rgb=np.median(p,0); out[name]=dict(lab=med.round(1).tolist(),rgb=rgb.round().tolist(),Lsd=float(L[:,0].std().round(1)),p10=float(np.percentile(L[:,0],10).round(1)),p90=float(np.percentile(L[:,0],90).round(1)))
        print(f'{name:38s} L*{med[0]:5.1f} a*{med[1]:6.1f} b*{med[2]:6.1f}  rgb{tuple(int(v) for v in rgb)}  Lsd {L[:,0].std():4.1f} p10-p90 {np.percentile(L[:,0],10):4.1f}-{np.percentile(L[:,0],90):4.1f}')
    return out
if __name__=='__main__':
    R={}
    print('== Cape Alava 2016 (sun, 14:24, summer)')
    R['ca']=samp(O+'cape_alava__060819_15434_2016.jpg',{
      'sand shoal turquoise (~2-4 m)':(150,600,350,700),'sand deeper teal (~6-10 m)':(40,960,200,1050),
      'kelp mat (dense)':(560,925,640,975),'rock reef through clear water':(1300,380,1450,450),
      'deep reef water right':(1450,560,1590,640),'beach surf foam':(160,470,260,500)})
    print('== Point of Arches 2016 (sun 14:27)')
    R['poa']=samp(O+'point_of_arches__060819_15405_2016.jpg',{
      'offshore ~15-20 m':(1100,960,1300,1050),'cove 3-5 m':(900,600,1000,660),'aerated jade by stack':(640,640,690,680),
      'foam streak core':(800,730,830,745),'kelp speck field':(1200,780,1300,830)})
    print('== La Push 2016 (sun 13:27)')
    R['lp']=samp(O+'la_push__0208131107_153_2016.jpg',{
      'open sea':(1100,380,1400,500),'sheltered cove green':(100,600,400,800),'estuary turbid':(500,950,800,1050),'surf zone foam':(1200,640,1400,665)})
    print('== Shi Shi 2006 (big swell, sun 13:44)')
    R['ss']=samp(O+'shi_shi__160811_07663_2006.jpg',{'jade between foam':(620,880,720,940),'foam sheet':(100,810,300,850),'wet sand':(700,720,900,760),'dry sand':(300,620,500,660)})
    json.dump(R,open('../notes/colour_samples.json','w'),indent=1)
