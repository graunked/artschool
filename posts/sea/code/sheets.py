"""Labelled sweep sheets. Each panel = one (place, sea state, light, view)."""
import sys, os; sys.path.insert(0,'.')
import numpy as np
from PIL import Image, ImageDraw
from sea.scene import scene
from sea.render import Cam
from sea.pal import LIGHTS
from stage1 import render
from shore import render_shore, find_stand
from sea.render import PerspCam
T = {'calm':7.0,'swell':8.3,'storm':14.0,'summer':8.3}
WIND = {'calm':1.0,'swell':3.0,'storm':18.0,'summer':3.0}
ISO_C = {'poa':(720,750),'alava':(640,430),'kala':(380,800)}
STAND = {'poa':(1020,760,270),'alava':(900,420,300),'kala':(560,820,262)}
_cache = {}
def get(place, state):
    k=(place,state)
    if k not in _cache:
        _cache.clear(); _cache[k]=scene(place,f'{place}_{state}.npz',T=T[state])
    return _cache[k]
def params(light, state):
    L=LIGHTS[light]
    return dict(sun_az=L['sun_az'], sun_el=L['sun_el'], glitter=1.0 if L['direct']>0 else 0.0, wind_speed=WIND[state])
def panel(place, state, light, view, W=240, H=150):
    P,S,zl=get(place,state)
    if view.startswith('iso'):
        mpp=float(view[3:]); cx,cy=ISO_C[place]
        cam=Cam(cx=cx,cy=cy,mpp=mpp,W=W,H=H,yaw=90,elev=30,zc=0.45)
        rgb,G=render(S,zl,cam,light=light,params=params(light,state))
    else:
        x,y,hd=STAND[place]; z=np.nan_to_num(P['z'],nan=-40); b=find_stand(z,x,y,r=250,lo=0.5,hi=0.75)
        cam=PerspCam(b[1],b[2],b[3]+1.7,hd,pitch=-4.0,hfov=55,W=W,H=H)
        rgb,G=render_shore(S,zl,cam,light)
    return rgb
def sheet(rows, cols, cellfn, rlab, clab, title, out, k=2, W=240, H=150):
    lw=110; th=24
    im=Image.new('RGB',(lw+len(cols)*(W*k+6),th+len(rows)*(H*k+18)),(18,18,22)); d=ImageDraw.Draw(im)
    d.text((6,6),title,fill=(235,235,235))
    for j,c in enumerate(cols): d.text((lw+j*(W*k+6)+4,th-2),clab(c),fill=(200,200,200))
    for i,r in enumerate(rows):
        y=th+10+i*(H*k+18); d.text((4,y+H*k//2),rlab(r),fill=(220,220,220))
        for j,c in enumerate(cols):
            rgb=cellfn(r,c)
            im.paste(Image.fromarray(rgb).resize((W*k,H*k),Image.NEAREST),(lw+j*(W*k+6),y))
        im.save(out)
    im.save(out)
if __name__=='__main__':
    which=sys.argv[1]
    if which=='A':   # sea state x place (depth/bottom) in the iso 1 m/px view, sun
        places=['kala','poa','alava']; names={'kala':'Kalaloch: sand','poa':'P. of Arches: rock reef','alava':'Cape Alava: kelp'}
        states=['calm','swell','storm']
        for view in (sys.argv[2:] or ['iso1','iso3','shore']):
            order=[(p,s) for p in places for s in states]
            sheet(states, places, lambda s,p: panel(p,s,'sun',view), lambda s:s, lambda p:names[p],
                  f'Sheet A: sea state x bottom, view {view}, sun (rows: calm Hs 0.6 m/7 s, swell 1.6 m/8.3 s, storm 5.5 m/14 s + 18 m/s wind)',
                  f'../sheets/A_state_x_bottom_{view}.png')
    if which=='B':   # light x view at swell
        lights=['sun','overcast','golden','dusk','fog']; views=['iso1','iso3','shore']
        for place in sys.argv[2:] or ['poa']:
            sheet(lights, views, lambda l,v: panel(place,'swell',l,v), lambda l:l, lambda v:v,
                  f'Sheet B: weather and hour x view, {place}, swell (same index canvas, palette relit)', f'../sheets/B_light_x_view_{place}.png')
