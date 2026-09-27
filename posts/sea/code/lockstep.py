"""Lockstep: the same place, fields and palette at 0.1, 0.3, 1 and 3 m/px, each beside the photo
warped into the same view (box-averaged to the same pixel size)."""
import sys, time; sys.path.insert(0,'.')
import numpy as np
from PIL import Image, ImageDraw
from stage1 import fields, render, photo_view
from sea.render import Cam, gbuffer
O='~/work/voxsim-data/places/olycoast/obliques/point_of_arches__060819_15405_2016.jpg'
def photo_at(Hm,cx,cy,mpp,W,H,ss=4):
    c4=Cam(cx=cx,cy=cy,mpp=mpp/ss,W=W*ss,H=H*ss,yaw=90,elev=30,zc=0.45)
    G4=gbuffer(c4,np.full((10,10),-40.0),sea_level=0.45)
    ph4,_=photo_view(G4,Hm,O)
    return np.asarray(Image.fromarray(ph4).resize((W,H),Image.BOX))
if __name__=='__main__':
    from sea.scene import scene
    P,S,zl=scene('poa','poa_summer.npz')
    Hm=np.load('../notes/H_poa.npy')
    cx,cy=float(sys.argv[1]),float(sys.argv[2]); out=sys.argv[3]
    scales=[0.1,0.3,1.0,3.0]; W,H=240,150; k=3
    sheet=Image.new('RGB',(2*W*k+10,len(scales)*(H*k+16)),(20,20,24)); d=ImageDraw.Draw(sheet)
    for i,m in enumerate(scales):
        cam=Cam(cx=cx,cy=cy,mpp=m,W=W,H=H,yaw=90,elev=30,zc=0.45)
        rgb,G=render(S,zl,cam)
        ph=photo_at(Hm,cx,cy,m,W,H)
        y=i*(H*k+16)
        d.text((4,y+2),f'photo, {m} m/px',fill=(220,220,220)); d.text((W*k+14,y+2),f'mine, {m} m/px',fill=(220,220,220))
        sheet.paste(Image.fromarray(ph).resize((W*k,H*k),Image.NEAREST),(0,y+16))
        sheet.paste(Image.fromarray(rgb).resize((W*k,H*k),Image.NEAREST),(W*k+10,y+16))
    sheet.save(out)
