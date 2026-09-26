import sys; sys.path.insert(0,'src'); from util import *
a=load_ref(); P=np.load('out/planes.npy'); lab=rgb2lab(a); L,A,B=lab[...,0],lab[...,1],lab[...,2]
# light classes: 0 deep shadow/black, 1 shadow (cool / dark), 2 half-light, 3 lit (warm), 4 highlight
cls=np.zeros(L.shape,int)
warm=(B>12)
cls[(L>=8)]=1
cls[(L>=30)&~warm]=2
cls[(L>=25)&warm]=3
cls[(L>=62)&warm]=4
cls[(L>=62)&~warm]=2   # lavender snow / sky in shade-light
C=np.array([(0,0,60),(60,60,160),(150,150,200),(250,170,40),(255,255,160)],np.uint8)
out=C[cls]; out[P==0]=(235,235,235)
img=up(out,2)
im=Image.fromarray(img); d=ImageDraw.Draw(im); f=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',15)
for i,t in enumerate(['core/black accents','shadow (dark, cool or olive)','cool light (lavender snow, sky refl.)','warm lit (sunlit ochre/salmon)','warm highlight']):
    d.rectangle([8,8+i*20,24,22+i*20],fill=tuple(C[i]),outline=(0,0,0)); d.text((30,8+i*20),t,fill=(0,0,0),font=f)
im.save('img/05_lightmap.png')
# magnified mountain + a tree
save(np.concatenate([up(a[70:200,200:400],3),up(out[70:200,200:400],3)],1),'05_light_mountain.png',1)
save(np.concatenate([up(a[20:200,30:130],3),up(out[20:200,30:130],3)],1),'05_light_tree.png',1)
