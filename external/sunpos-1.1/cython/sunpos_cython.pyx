import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport sqrt as csqrt

ctypedef np.int64_t ITYPE_t
ctypedef np.double_t FTYPE_t

ITYPE   = np.int64
FTYPE   = np.double
empty   = np.empty
asarray = np.asarray

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
def sunpos_cython(FTYPE_t xs, FTYPE_t ys, FTYPE_t xe, FTYPE_t ye):
    """
    find_pixel_trace(xs,ys,xe,ye)

    Compute 2D shortest path between two pixel coordinates, return x,y and
    cost of traversal
    
    Arguments:
    - xs, ys: start pixel
    - xe, ye: end pixel
    
    Returns:
    - xpix,ypix, s: x,y pixel coordinates, traversal distances
    """
    cdef ITYPE_t i, ixs,ixe,iys,iye, nh,nv,uhv, nc,hc, sgnx,sgny
    cdef FTYPE_t m,b, dx,dy, dxy,dyx, xc,yc
    cdef np.ndarray[FTYPE_t, ndim=2] xys

    ixs = ITYPE(xs)
    ixe = ITYPE(xe)
    iys = ITYPE(ys)
    iye = ITYPE(ye)

    dx  = FTYPE(xe-xs)
    dy  = FTYPE(ye-ys)
    sgnx = ((dx > 0) - (dx < 0)) 
    sgny = ((dy > 0) - (dy < 0))
    
    # number of (horizontal=iye-iys) and (vertical=ixe-ixs) crossings
    nh = ixe-ixs
    nv = iye-iys
    nc = ITYPE(sgnx*nh + sgny*nv)

    xys = empty(dtype=FTYPE,shape=(nc+1,3))
    # init start/end points
    xys[0, 0] = FTYPE(ixs)
    xys[0, 1] = FTYPE(iys)
    xys[0, 2] = 0.0
    xys[nc,0] = FTYPE(ixe)   
    xys[nc,1] = FTYPE(iye)
    xys[nc,2] = csqrt(dx*dx + dy*dy)
    if nc > 1:
        hc = ITYPE(nc/2)
        uhv = nh==1 or nv==1
        # find path between (xs,ys), (xe,ye)
        if sgnx*sgny != 0: # both sgnx/sgny are nonzero
            m  = dy/dx
            b  = ye-m*xe            
            for i in range(1,nc):
                xc = xys[i-1,0]+sgnx
                yc = xys[i-1,1]+sgny
                dx = xc-xs
                dy = yc-ys
                dxy = ((yc-b)/m)-xs
                dyx = ((m*xc+b))-ys                
                if i==hc and uhv:
                    # force a step at the midpoint if nh or nv == 1
                    if nv==1:
                        yc -= sgny
                        dy  = yc-ys
                    else:
                        xc -= sgnx
                        dx  = xc-xs
                elif (dx*dx + dyx*dyx) < (dy*dy + dxy*dxy): # x-step
                    yc -= sgny
                    dy  = yc-ys
                else: # y-step
                    xc -= sgnx
                    dx  = xc-xs
                                    
                xys[i,0] = xc
                xys[i,1] = yc                    
                xys[i,2] = csqrt(dx*dx + dy*dy)
                
        else: # either sgnx or sgny == 0 (i.e., horizontal or vertical line)
            for i in range(1,nc):
                xc = xys[i-1,0]+sgnx
                yc = xys[i-1,1]+sgny
                dx = xc-xs
                dy = yc-ys
                xys[i,0] = xc
                xys[i,1] = yc
                xys[i,2] = csqrt(dx*dx + dy*dy)

    return xys

