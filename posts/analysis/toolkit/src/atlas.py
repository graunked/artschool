import sys; sys.path.insert(0,'src'); from util import *
from collections import Counter
def atlas(a,wins,name,s=7,sz=32):
    tiles=[]
    for title,(x,y) in wins:
        w=a[y:y+sz,x:x+sz]
        cnt=Counter(map(tuple,w.reshape(-1,3))).most_common(3)
        row=[up(w,s)]
        for col,_ in cnt:   # isolate each of the 3 dominant colours
            m=np.all(w==col,axis=2); t=np.full_like(w,128); t[m]=col; t[~m]=(40,40,40) if np.mean(col)>90 else (200,200,200)
            row.append(up(t,s//2+1)[:sz*s//2+sz//2*0+ sz*(s//2+1)][:,:])
        # compose: big tile + 3 small isolation tiles stacked
        big=row[0]; smalls=[r[:sz*(s//2+1),:sz*(s//2+1)] for r in row[1:]]
        colh=big.shape[0]; sw=smalls[0].shape[1]
        side=np.full((colh,sw,3),255,np.uint8); yy=0
        for sm in smalls:
            h=min(sm.shape[0],colh-yy); side[yy:yy+h]=sm[:h]; yy+=h+2
            if yy>=colh: break
        t=np.concatenate([big,np.full((colh,3,3),255,np.uint8),side],1)
        t=np.concatenate([np.full((22,t.shape[1],3),255,np.uint8),t],0)
        t=label(t,title,(2,3),fill=(0,0,0),bg=(255,255,255))
        tiles.append(t)
    W=max(t.shape[1] for t in tiles); rows=[]
    for i in range(0,len(tiles),3):
        r=tiles[i:i+3]; r=[np.pad(t,((0,0),(0,W-t.shape[1]+6),(0,0)),constant_values=255) for t in r]
        while len(r)<3: r.append(np.full_like(r[0],255))
        rows.append(np.concatenate(r,1))
    Image.fromarray(np.concatenate(rows,0)).save(f'img/{name}')
if __name__=='__main__':
    a=load_ref()
    wins=[('fore grass, lit (left)',(40,400)),('fore grass, shade (right)',(560,400)),('mid meadow, lit',(60,262)),
          ('far bank / hummock',(520,228)),('island top (rim light)',(340,258)),('conifer, near',(40,120)),
          ('conifer edge vs sky',(95,60)),('mountain rock + light',(215,110)),('snow, shade/light',(318,120)),
          ('water: sky reflection',(300,370)),('water: tree reflection',(290,310)),('shore stones',(250,405))]
    atlas(a,wins,'09_cluster_atlas.png')
