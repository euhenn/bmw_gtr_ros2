'''
TRANSFORMS VALUES OF TIME DOMAIN INTO THE SPATIAL DOMAIN VIA INTERPOLATION BETWEEN POINTS
NEEDED IN ORDER TO HAVE SIMULATION AND CONTROL IN DIFFERENT DOMAINS 

this works very well unless we have a track crossing or tracks close to each other
in this case the part of the reference that is picked is NOT CORRECT
'''

import numpy as np

def time2spatial(x,y,psi, sref,y_ref):
    #xref = y_ref[:, 0] 
    #yref = y_ref[:, 1]
    #psiref = y_ref[:, 2]
    xref = y_ref[:, 0] 
    yref = y_ref[:, 1]
    psiref = y_ref[:, 2]

    #s_idx = ClosestS(ssim,sref)
    idxmin=ClosestPoint(x,y,xref,yref)
    idxmin2=NextClosestPoint(x,y,xref,yref,idxmin)
    p=StateProjection(x,y,xref,yref,sref,idxmin,idxmin2)

    s0=(1-p)*sref[idxmin]+p*sref[idxmin2]
    x0=(1-p)*xref[idxmin]+p*xref[idxmin2]
    y0=(1-p)*yref[idxmin]+p*yref[idxmin2]
    psi0=(1-p)*psiref[idxmin]+p*psiref[idxmin2]

    s=s0
    ey=np.cos(psi0)*(y-y0)-np.sin(psi0)*(x-x0)
    epsi=psi-psi0

    return s,epsi,ey

def StateProjection(x,y,xref,yref,sref,idxmin,idxmin2): # linear interpolation between 2 points
    ds=abs(sref[idxmin]-sref[idxmin2]) # ds
    dxref=xref[idxmin2]-xref[idxmin]
    dyref=yref[idxmin2]-yref[idxmin]
    dx=x-xref[idxmin]
    dy=y-yref[idxmin]
    
    p =(dxref*dx+dyref*dy)/ds/ds
    return p

def ClosestPoint(x,y,xref,yref):
    min_dist=1
    idxmin=0
    for i in range(xref.size):
        dist=Eucl_dist(x,xref[i],y,yref[i])
        if (dist<min_dist):
            min_dist=dist
            idxmin=i
    return idxmin

def NextClosestPoint(x,y,xref,yref,idxmin):
    prev_dist=Eucl_dist(x,xref[idxmin-1],y,yref[idxmin-1])
    next_dist=Eucl_dist(x,xref[idxmin+1],y,yref[idxmin+1])

    if(prev_dist<next_dist):
        idxmin2=idxmin-1
    else:
        idxmin2=idxmin+1

    if(idxmin2<0):
        idxmin2=xref.size-1
    elif(idxmin==xref.size):
        idxmin2=0
    
    return idxmin2


def time2spatialS(x,y,psi,s,sref,y_ref):
    xref = y_ref[:, 0] 
    yref = y_ref[:, 1]
    psiref = y_ref[:, 2]

    idxmin=ClosestS(s,sref)
    idxmin2=NextClosestS(s,sref, idxmin)
    p=StateProjectionS(x,y,xref,yref,sref,idxmin,idxmin2)

    s0=(1-p)*sref[idxmin]+p*sref[idxmin2]
    x0=(1-p)*xref[idxmin]+p*xref[idxmin2]
    y0=(1-p)*yref[idxmin]+p*yref[idxmin2]
    psi0=(1-p)*psiref[idxmin]+p*psiref[idxmin2]

    s=s0
    ey=np.cos(psi0)*(y-y0)-np.sin(psi0)*(x-x0)
    epsi=psi-psi0

    return epsi,ey

def StateProjectionS(x,y,xref,yref,sref,idxmin,idxmin2): # linear interpolation between 2 points
    ds=abs(sref[idxmin]-sref[idxmin2]) # ds
    dxref=xref[idxmin2]-xref[idxmin]
    dyref=yref[idxmin2]-yref[idxmin]
    dx=x-xref[idxmin]
    dy=y-yref[idxmin]
    
    p =(dxref*dx+dyref*dy)/ds/ds
    return p

def ClosestS(s,sref):
    min_dist=1
    idxmin=0
    for i in range(sref.size):
        dist= np.sqrt((s-sref[i])**2) 
        if dist<min_dist:
            min_dist=dist
            idxmin=i
    return idxmin

def NextClosestS(s,sref,idxmin):
    prev_dist=np.sqrt((s-sref[idxmin-1])**2) 
    next_dist=np.sqrt((s-sref[idxmin+1])**2) 

    if(prev_dist<next_dist):
        idxmin2=idxmin-1
    else:
        idxmin2=idxmin+1

    if(idxmin2<0):
        idxmin2=sref.size-1
    elif(idxmin==sref.size):
        idxmin2=0
    
    return idxmin2



def Eucl_dist(x1,x2,y1,y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)


