import sys;sys.path.insert(0,'.')
import numpy as np
from reg import *
from dem import window
from PIL import Image
s=2.625
pairs=[((1045,1005),(705*s-55,665*s+65)),((1265,900),(455*s,548*s)),((870,1320),(1265*s,690*s)),((775,660),(200*s,1010*s)),((575,1380),(1550*s,800*s))]
H=homog([p[0] for p in pairs],[p[1] for p in pairs])
np.save('../notes/H_poa.npy',H)
O='~/work/voxsim-data/places/olycoast/obliques/point_of_arches__060819_15405_2016.jpg'
X0,Y1=372347,5346016
box=(550,450,1500,1450)
img,ok=warp(O,H,box,1.0,'../notes/poa_ortho.png')
a=window('olycoast','elev',X0+box[0],Y1-box[3],X0+box[2],Y1-box[1])
np.save('../notes/poa_ortho_elev.npy',a)
