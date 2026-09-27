"""Eye-level view from the beach, looking out to sea (perspective)."""
import sys; sys.path.insert(0,'.')
import numpy as np
from PIL import Image
from sea.scene import scene
from sea.render import PerspCam, persp_gbuffer
from sea.paint import paint_sea, paint_land, bayer
from sea.pal import to_rgb, RID
def find_stand(z, x0, y0, r=150, lo=1.0, hi=1.8):
    best=None
    for yy in range(max(0,int(y0-r)),min(z.shape[0],int(y0+r))):
        for xx in range(max(0,int(x0-r)),min(z.shape[1],int(x0+r))):
            v=z[yy,xx]
            if lo<v<hi:
                dd=(xx-x0)**2+(yy-y0)**2
                if best is None or dd<best[0]: best=(dd,xx,yy,v)
    return best
def render_shore(S,zl,cam,light='sun',tide=0.45):
    G=persp_gbuffer(cam,zl,sea_level=tide,eta=S.get('eta'),depth=S['depth'])
    class C: pass
    c=C(); c.mpp=G['mpp']
    ramp,step,lay=paint_sea(G,S,c,dict(sun_az=205))
    G2=dict(G); G2['sea']=~G['land']            # paint_land paints where ~sea
    ramp,step=paint_land(G2,S,c,ramp,step,tide)
    sky=G['sky']; H,W=sky.shape
    # sky: flat bands by elevation with a checker seam (the sky study owns the real sky)
    el=np.degrees(np.arcsin(np.clip(np.nan_to_num(G['dz']),-1,1)))
    band=np.clip((el/12.0*6).astype(int),0,5)
    ramp[sky]=RID['sky']; step[sky]=(5-band)[sky]
    return to_rgb(ramp,step,light),G
if __name__=='__main__':
    place=sys.argv[1]; res=sys.argv[2]; px,py,heading=map(float,sys.argv[3:6]); out=sys.argv[6]
    light=sys.argv[7] if len(sys.argv)>7 else 'sun'
    P,S,zl=scene(place,res)
    z=np.nan_to_num(P['z'],nan=-40)
    b=find_stand(z,px,py); print('stand',b)
    cam=PerspCam(b[1],b[2],b[3]+1.7,heading,pitch=-3.0,hfov=55,W=480,H=270)
    rgb,G=render_shore(S,zl,cam,light)
    Image.fromarray(rgb).resize((1440,810),Image.NEAREST).save(out)
