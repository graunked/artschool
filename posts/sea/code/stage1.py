"""Stage 1: the sea body, kelp and surf over the real Point of Arches, iso 1 m/px, beside the photo
warped into the very same view through the sea-plane homography."""
import sys, time; sys.path.insert(0,'.')
import numpy as np
from PIL import Image, ImageDraw
from sea.place import load
from sea.render import Cam, gbuffer
from sea.paint import paint_sea, paint_land
from sea.pal import to_rgb
from sea.sim import bottom_type
from dem import window
from reg import apply
def fields(tag='poa'):
    P=load('poa'); Wv=dict(np.load('../cache/poa_waves.npz')); St=dict(np.load('../cache/poa_state.npz'))
    chm=window('olycoast','chm',372347+350,5346016-1650,372347+1650,5346016-250,dtype=np.uint8)
    rock,rough=bottom_type(P['z'],P['depth'])
    S=dict(depth=np.nan_to_num(P['depth'],nan=30.0),rock=rock,rough=rough,kelp=St['kelp'],kelp_rho=St['kelp_rho'],W=St['W'],F=St['F'],A=St['A'],
           Hinst=St['Hinst'],u=St['u'],v=St['v'],tt=Wv['t'],brkf=St['brk'].astype(np.float32),T=8.3,t_now=60*8.3,chm=np.nan_to_num(chm))
    zland=np.nan_to_num(P['z'],nan=-40)+np.nan_to_num(chm)
    return P,S,zland
def render(S,zland,cam,light='sun',tide=0.45,params=None,G=None):
    G=G if G is not None else gbuffer(cam,zland,sea_level=tide)
    ramp,step,lay=paint_sea(G,S,cam,params)
    ramp,step=paint_land(G,S,cam,ramp,step,tide)
    return to_rgb(ramp,step,light),G
def photo_view(G,Hm,photo,frame=(350,250)):
    im=np.asarray(Image.open(photo).convert('RGB'))
    u,v=apply(Hm,(G['x']+frame[0]).ravel(),(G['y']+frame[1]).ravel())
    u=u.reshape(G['x'].shape); v=v.reshape(G['x'].shape)
    ok=(u>=0)&(u<im.shape[1]-1)&(v>=0)&(v<im.shape[0]-1)&G['sea']
    out=np.zeros(G['x'].shape+(3,),np.uint8); out[ok]=im[v[ok].astype(int),u[ok].astype(int)]
    return out,ok
if __name__=='__main__':
    t=time.time()
    from sea.scene import scene
    P,S,zl=scene('poa',sys.argv[3] if len(sys.argv)>3 else 'poa_summer.npz')
    Hm=np.load('../notes/H_poa.npy')
    O='~/work/voxsim-data/places/olycoast/obliques/point_of_arches__060819_15405_2016.jpg'
    mpp=float(sys.argv[1]) if len(sys.argv)>1 else 1.0
    cam=Cam(cx=720,cy=750,mpp=mpp,W=480,H=300,yaw=90,elev=30,zc=0.45)
    rgb,G=render(S,zl,cam)
    ph,ok=photo_view(G,Hm,O)
    # photo at pixel size: box-average the photo footprint -> sample at 4x then box down
    cam4=Cam(cx=720,cy=750,mpp=mpp/4,W=480*4,H=300*4,yaw=90,elev=30,zc=0.45)
    G4=gbuffer(cam4,np.full((10,10),-40.0),sea_level=0.45)
    ph4,_=photo_view(G4,Hm,O)
    ph=np.asarray(Image.fromarray(ph4).resize((480,300),Image.BOX))
    k=3; W,H=480,300
    sheet=Image.new('RGB',(W*k*2+10,H*k+20),(20,20,24)); d=ImageDraw.Draw(sheet)
    sheet.paste(Image.fromarray(ph).resize((W*k,H*k),Image.NEAREST),(0,20))
    sheet.paste(Image.fromarray(rgb).resize((W*k,H*k),Image.NEAREST),(W*k+10,20))
    d.text((4,4),f'photo 060819_15405 warped to iso 30deg, box-averaged to {mpp} m/px (sea plane only)',fill=(230,230,230))
    d.text((W*k+14,4),f'mine, same view, same place (lidar), summer swell Hs 1.57 m 8.3 s, sun',fill=(230,230,230))
    out=sys.argv[2] if len(sys.argv)>2 else '../stages/s1_poa_iso.png'
    sheet.save(out); Image.fromarray(rgb).save(out.replace('.png','_1x.png')); print('t',time.time()-t)
