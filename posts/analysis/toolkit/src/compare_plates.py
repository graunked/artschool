"""Comparison plates: the same analyses on Ferrari and on painted cells, side by side."""
import sys; sys.path.insert(0,'src'); from util import *
from scipy.ndimage import gaussian_filter
from atlas import atlas
from edges import draw as edge_draw
CELLS='sheets/cells'
def notan_img(a, cuts=(28.9,55.6)):
    Lb=gaussian_filter(lum(a),2.5*a.shape[1]/634)   # same squint relative to picture width
    q=np.digitize(Lb,cuts); lv=np.array([25,130,240]); g=lv[q].astype(np.uint8)
    return np.stack([g]*3,-1)
def fit(a,w):
    return np.array(Image.fromarray(a).resize((w,int(a.shape[0]*w/a.shape[1])),Image.NEAREST))
ref=load_ref()
keys=['medium_left_10m_s1','medium_front_10m_s1','medium_back_10m_s1','medium_left_close_s1','medium_left_40m_s1','medium_left_10m_s4']
W0=420
tiles=[label(fit(ref,W0),'Ferrari'),label(fit(notan_img(ref),W0),'Ferrari 3-notan')]
for k in keys:
    a=np.array(Image.open(f'{CELLS}/{k}.png').convert('RGB'))
    tiles+= [label(fit(a,W0),k),label(fit(notan_img(a),W0),'3-notan (same cuts)')]
rows=[]
for i in range(0,len(tiles),4):
    r=tiles[i:i+4]
    h=max(t.shape[0] for t in r)
    r=[np.pad(t,((0,h-t.shape[0]),(0,6),(0,0)),constant_values=255) for t in r]
    while len(r)<4: r.append(np.full_like(r[0],255))
    rows.append(np.pad(np.concatenate(r,1),((0,6),(0,0),(0,0)),constant_values=255))
Image.fromarray(np.concatenate(rows,0)).save('img/20_notan_compare.png')
# cluster atlas for my base cell (same windows by material)
a=np.array(Image.open(f'{CELLS}/medium_left_10m_s1.png').convert('RGB'))
P=np.load(f'{CELLS}/medium_left_10m_s1_planes.npy')
def win(pl,cond=None):
    ys,xs=np.nonzero(P==pl)
    ok=(ys>2)&(ys<a.shape[0]-34)&(xs<a.shape[1]-34)
    if cond is not None: ok&=cond(ys,xs)
    i=np.flatnonzero(ok)
    j=i[len(i)//2] if len(i) else 0
    return (int(xs[j]),int(ys[j]))
wins=[('near grass, lit',win(6,lambda y,x: x<120)),('near grass, shade',win(6,lambda y,x: x>220)),('bank by boulder',win(4)),
      ('boulder',win(5)),('grove (conifer)',win(8)),('water',win(3,lambda y,x:(y>120)&(x>120)&(x<200))),
      ('far forest',win(2)),('mountain',win(1)),('sky',win(0))]
atlas(a,[(t,(max(0,x-4),max(0,y-4))) for t,(x,y) in wins],'21_cluster_atlas_mine.png',s=7,sz=32)
edge_draw(a,'22_edges_mine',3)
