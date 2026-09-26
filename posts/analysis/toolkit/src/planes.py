"""Hand-traced planes of the reference (native 634x469 coords) + colour-based sky/tree split.
Output: out/planes.npy (int labels), img/02_planes.png"""
import sys; sys.path.insert(0,'src'); from util import *
a=load_ref(); H,W=a.shape[:2]; lab=rgb2lab(a); L=lab[...,0]
NAMES=['sky','mountain','far forest','framing trees','far banks','island','water','left meadow (mid)','left bank (fore)','right bank (fore)']
COLS=[(214,170,196),(150,110,160),(90,80,50),(40,40,20),(120,120,60),(180,160,70),(120,150,210),(230,180,90),(200,120,50),(60,70,30)]
def poly(pts):
    im=Image.new('L',(W,H),0); ImageDraw.Draw(im).polygon(pts,fill=1); return np.array(im)>0
mountain=poly([(150,135),(205,108),(250,100),(285,90),(307,66),(330,84),(355,95),(395,118),(470,128),(505,135),(505,235),(150,235)])
water=poly([(205,250),(560,250),(634,262),(634,285),(560,290),(520,300),(490,320),(450,345),(420,370),(390,395),(360,420),(345,448),(332,469),(262,469),(250,448),(235,420),(215,390),(190,360),(165,335),(140,318),(160,300),(205,288)])
island=poly([(257,285),(275,268),(310,258),(360,257),(400,265),(418,285),(400,298),(330,300),(270,296)])
leftmeadow=poly([(0,238),(60,240),(130,248),(205,258),(207,288),(160,300),(140,318),(0,318)])
leftfore=poly([(0,318),(140,318),(165,335),(190,360),(215,390),(235,420),(250,448),(262,469),(0,469)])
rightfore=poly([(634,285),(560,290),(520,300),(490,320),(450,345),(420,370),(390,395),(360,420),(345,448),(330,469),(634,469)])
farbanks=poly([(80,212),(200,215),(300,225),(420,222),(500,215),(634,215),(634,262),(560,250),(205,250),(205,258),(130,248),(60,240),(0,238),(0,225)])
framing=poly([(0,0),(175,0),(175,245),(0,245)])|poly([(490,0),(634,0),(634,235),(490,235)])
skyish=(L>50)&(lab[...,1]>3)&(lab[...,2]<30)   # pink/violet high value
yy,xx=np.mgrid[0:H,0:W]
lab_=np.full((H,W),2)                      # default far forest (upper middle)
lab_[framing]=3
lab_[mountain & ~framing]=1
# inside mountain zone, conifer colours stay forest: dark or yellow-ochre
coniferish=(L<35)|((lab[...,2]>25)&(lab[...,1]<25))
lab_[mountain & ~framing & coniferish]=2
lab_[(yy<245)&skyish&~mountain]=0
lab_[(yy<140)&skyish&~mountain]=0
lab_[farbanks]=4; lab_[water]=6; lab_[island]=5
lab_[leftmeadow]=7; lab_[leftfore]=8; lab_[rightfore]=9
np.save('out/planes.npy',lab_)
C=np.array(COLS,np.uint8); out=C[lab_]
mix=(0.45*a+0.55*out).astype(np.uint8)
img=up(mix,2)
im=Image.fromarray(img); d=ImageDraw.Draw(im); f=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',15)
for i,n in enumerate(NAMES):
    d.rectangle([8,8+i*20,24,22+i*20],fill=COLS[i],outline=(0,0,0)); d.text((30,8+i*20),f'{i} {n}',fill=(255,255,255),font=f,stroke_width=2,stroke_fill=(0,0,0))
im.save('img/02_planes.png')
for i,n in enumerate(NAMES): print(i,n,(lab_==i).mean().round(3))
