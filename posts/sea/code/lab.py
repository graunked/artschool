import numpy as np
def srgb_to_lab(rgb):
    c=np.asarray(rgb,float)/255; c=np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
    M=np.array([[0.4124,0.3576,0.1805],[0.2126,0.7152,0.0722],[0.0193,0.1192,0.9505]])
    xyz=c@M.T/np.array([0.95047,1.0,1.08883])
    f=np.where(xyz>0.008856,np.cbrt(xyz),7.787*xyz+16/116)
    return np.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],-1)
def lab_to_srgb(lab):
    lab=np.asarray(lab,float); fy=(lab[...,0]+16)/116; fx=fy+lab[...,1]/500; fz=fy-lab[...,2]/200
    f=np.stack([fx,fy,fz],-1); xyz=np.where(f**3>0.008856,f**3,(f-16/116)/7.787)*np.array([0.95047,1,1.08883])
    Mi=np.linalg.inv(np.array([[0.4124,0.3576,0.1805],[0.2126,0.7152,0.0722],[0.0193,0.1192,0.9505]]))
    c=xyz@Mi.T; c=np.where(c<=0.0031308,12.92*c,1.055*np.clip(c,0,None)**(1/2.4)-0.055)
    return np.clip(c*255,0,255)
