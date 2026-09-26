"""Pixel-geometry of contours: how silhouettes resolve into runs (flat, 45deg, 2:1 ...)."""
import sys; sys.path.insert(0,'src'); from util import *
from collections import Counter
def profile_runs(top):
    """top: y per column (int). Returns list of (run_len_x, step_y) segments: a horizontal run then a step."""
    segs=[]; run=1
    for i in range(1,len(top)):
        dy=int(top[i]-top[i-1])
        if dy==0: run+=1
        else: segs.append((run,dy)); run=1
    segs.append((run,0)); return segs
def classify(segs):
    c=Counter()
    for run,dy in segs:
        a=abs(dy)
        if dy==0: continue
        if run==1 and a==1: c['45deg (1:1)']+=1
        elif run==2 and a==1: c['2:1 (shallow)']+=1
        elif run==1 and a==2: c['1:2 (steep)']+=1
        elif run>=3 and a==1: c['long flat run then 1 step']+=1
        elif run==1 and a>=3: c['vertical drop >=3']+=1
        else: c['other']+=1
    tot=sum(c.values()); return {k:round(v/tot,3) for k,v in c.most_common()},tot
def top_profile(mask,x0,x1):
    out=[]
    for x in range(x0,x1):
        ys=np.flatnonzero(mask[:,x]); out.append(ys[0] if len(ys) else -1)
    return np.array(out)
if __name__=='__main__':
    a=load_ref(); P=np.load('out/planes.npy')
    nonsky=P!=0
    for name,(x0,x1) in {'mountain crest':(205,300),'mountain right shoulder':(318,405),'tree line (left group)':(0,160)}.items():
        t=top_profile(nonsky,x0,x1); segs=profile_runs(t); cl,n=classify(segs)
        print(name,n,cl)
        print('   runs(x) hist', Counter([r for r,d in segs]).most_common(8))
