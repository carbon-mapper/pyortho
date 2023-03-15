import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport sqrt as csqrt

DTYPE = np.double
ctypedef np.double_t DTYPE_t

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def find_pixel_trace_cython(DTYPE_t xs,DTYPE_t ys,DTYPE_t xe,DTYPE_t ye):
    """
    find_pixel_trace(xs_in,ys_in,xe_in,ye_in)

    Compute 2D shortest path between two pixel coordinates, optionally return
    cost of traversal
    
    Arguments:
    - xs_in, ys_in: start pixel
    - xe_in, ye_in: end pixel
    
    Returns:
    - xpix,ypix: x,y pixel coordinates, s
    """
    cdef int i, sgnx,sgny, xmin,ymin
    cdef int ixs,ixe,iys,iye, n_vert,n_hori,n_cross
    cdef DTYPE_t m,b, ix,iy, dx,dy
    cdef np.ndarray[DTYPE_t, ndim=2] xys
    cdef np.ndarray[DTYPE_t, ndim=2] cross_dtype

    ixs = int(xs)
    ixe = int(xe)
    iys = int(ys)
    iye = int(ye)

    n_vert  = int(abs(ixe-ixs))
    n_hori  = int(abs(iye-iys))
    n_cross = n_vert+n_hori
    
    xys = np.zeros([n_cross+1,3],dtype=DTYPE)

    dx = DTYPE(xe-xs)
    dy = DTYPE(ye-ys)
    
    xys[ 0,0]      = DTYPE(ixs)
    xys[ 0,1]      = DTYPE(iys)
    xys[n_cross,0] = DTYPE(ixe)
    xys[n_cross,1] = DTYPE(iye)
    xys[n_cross,2] = DTYPE(csqrt(dx**2+dy**2))

    if n_cross > 1:
        cross_dtype = np.zeros([n_cross,2],dtype=DTYPE)
        # otherwise find path between (xs,ys), (xe,ye)
        sgnx = DTYPE((dx > 0) - (dx < 0)) 
        sgny = DTYPE((dy > 0) - (dy < 0))
        if dx != 0:
            m = DTYPE(dy/dx)
        elif dy > 0:
            m = DTYPE(1e+30)
        else:
            m = DTYPE(-1e+30)

        b = DTYPE(ye-m*xe)

        # if n_vert > 0:
        #     vert_x = np.arange(2,n_vert+1)+min(ixs,ixe)
        #     cross_dtype[1:n_vert,0] = np.sqrt((vert_x-xs)**2+((m*vert_x+b)-ys)**2)
        #     cross_dtype[1:n_vert,1] = 1
            
        # if n_hori > 0:
        #     hori_y = np.arange(2,n_hori+1)+min(iys,iye)    
        #     hori_x = (hori_y-b)/m if (dx != 0) else xs*(hori_y/hori_y)
        #     cross_dtype[n_vert+1:,0] = np.sqrt((hori_x-xs)**2+(hori_y-ys)**2)
        
        xmin = DTYPE(min(ixs,ixe))
        ymin = DTYPE(min(iys,iye))
        for i in range(max(n_hori,n_vert)):
            if i < n_vert:
                ix = i+1
                cross_dtype[i,0] = csqrt((ix)**2 + ((m*ix))**2)
                cross_dtype[i,1] = 1
            if i < n_hori:
                iy = i+1                
                ix = (iy-b)/m if dx != 0 else xs
                cross_dtype[n_vert+i,0] = csqrt((ix)**2 + (iy)**2)
                #cross_dtype[n_vert+i,1] = 0

        print cross_dtype
        # sort by distance then type to ensure correct ordering
        #cross_dtype = np.sort(cross_dtype,kind='quicksort',axis=0)
        cross_dtype.sort(kind='mergesort',axis=0)

        #xys[1:-1,0] += ixs+sgnx*np.cumsum((cross_dtype[:-1,1]==1))
        #xys[1:-1,1] += iys+sgny*np.cumsum((cross_dtype[:-1,1]==0))
        #xys[1:-1,2] = (cross_dtype[:-1,0]+cross_dtype[1:,0])/2

        
        for i in range(1,n_cross):
            ix = cross_dtype[i-1,1]
            xys[i,0] = xys[i-1,0] + sgnx*ix
            xys[i,1] = xys[i-1,1] + sgny*(ix==0)
            xys[i,2] = (cross_dtype[i-1,0]+cross_dtype[i,0])/2
            
        del cross_dtype

    return np.asarray(xys)
