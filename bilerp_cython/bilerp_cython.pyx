import numpy as np
cimport numpy as np
cimport cython

ITYPE = np.int64
ctypedef np.int64_t ITYPE_t

FTYPE = np.double
ctypedef np.double_t FTYPE_t

empty = np.empty
asarray = np.asarray

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def bilerp_cython(np.ndarray[FTYPE_t,ndim=2] gridxy,
                  np.ndarray[FTYPE_t,ndim=1] gridxf,
                  np.ndarray[FTYPE_t,ndim=1] gridyf):
    """
    bilerp(gridxy,gridxf,gridyf)

    Bilinear interpolation on a regularly-spaced grid
    
    Arguments:
    - gridxy: n x m grid of sample values to interpolate
    - gridxf: nx float64 x-coordinates within grid extent [0,m-1]
    - gridyf: ny float64 y-coordinates within grid extent [0,n-1]
    
    Keyword Arguments:
    None
    
    Returns:
    - interpolated gridxy values at points (gridxf,gridyf)
    """
    cdef np.ndarray[ITYPE_t, ndim=1] gridxi
    cdef np.ndarray[ITYPE_t, ndim=1] gridyi    
    cdef np.ndarray[FTYPE_t, ndim=1] dx
    cdef np.ndarray[FTYPE_t, ndim=1] dy    
    
    # gridxf,gridyf float indices (0-based)
    gridxi = asarray(gridxf,ITYPE)
    gridyi = asarray(gridyf,ITYPE)
    dx     = gridxf-gridxi
    dy     = gridyf-gridyi
    # gridx0y0,gridx1y0 = gridxy[gridyi,gridxi], gridxy[gridyi,gridxi+1]
    # gridx0y1,gridx1y1 = gridxy[gridyi+1,gridxi], gridxy[gridyi+1,gridxi+1]
    # return gridx0y0*(1.0-dx)*(1.0-dy) + gridx1y0*dx*(1.0-dy) + \
    #        gridx0y1*(1.0-dx)*dy       + gridx1y1*dx*dy
    return     gridxy[gridyi,   gridxi  ] * (1.0-dx) * (1.0-dy) + \
               gridxy[gridyi,   gridxi+1] * dx  * (1.0-dy) + \
               gridxy[gridyi+1, gridxi  ] * (1.0-dx) * dy + \
               gridxy[gridyi+1, gridxi+1] * dx  * dy
    
    # gridxyf  = gridxy[gridyi,   gridxi  ] * (1.0-dx) * (1.0-dy)
    # gridxyf += gridxy[gridyi,   gridxi+1] * dx  * (1.0-dy)
    # gridxyf += gridxy[gridyi+1, gridxi  ] * (1.0-dx) * dy
    # gridxyf += gridxy[gridyi+1, gridxi+1] * dx  * dy
    # return gridxyf


