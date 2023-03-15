from __future__ import division, print_function
import sys, os, datetime, time, json, warnings, signal

from os import SEEK_CUR,SEEK_SET
from os.path import join as pathjoin, exists as pathexists, split as pathsplit
from os.path import splitext, abspath, getsize, realpath, relpath, isabs, expandvars

# (importing individual functions used to be faster than 'from numpy import *')

from numpy import r_, c_, array, asarray, asmatrix, arange, dot, sqrt, ravel
from numpy import float32, float64, count_nonzero
from numpy import bool8, uint8, uint16, uint32, uint64, int8, int16, int32, int64
from numpy import bitwise_and, meshgrid, mgrid, apply_along_axis, dstack
from numpy import fromfile, where, sign, floor, ceil, ones, zeros, round
from numpy import argsort, sort, searchsorted, unique, cumsum, polyfit, polyval
from numpy import pi, tan, sin, cos, arccos, arcsin, arctan, arctan2, hypot 
from numpy import roll as np_roll, sum as np_sum, max as np_max, min as np_min
from numpy import nonzero, gradient, memmap, zeros_like, lexsort, linspace
from numpy import inf, nan, isnan, nanpercentile
from numpy import ones_like, zeros_like, newaxis, power
from numpy import mean, median, diff, setdiff1d, mgrid, percentile, clip
from numpy.linalg import inv
from numpy import set_printoptions

from skimage.segmentation import find_boundaries
from skimage.measure import label as bwlabel
from scipy.ndimage.interpolation import map_coordinates

from spectral.io.envi import read_envi_header, write_envi_header, \
    open as envi_open, create_image as envi_create_image

try:
    from numba import jit as nbjit
except Exception as e:
    pass

# print debug output?
DEBUG = 0

# exit codes
SUCCESS     = 0
FAILURE     = 1

dtime_now = datetime.datetime.now
time_elapsed = lambda start_time: str(dtime_now()-start_time)[2:-4]
time_sleep = time.sleep

def formatwarning(*args):
    message, category, filepath, lineno = args[:4]
    filedir,filename = pathsplit(filepath)
    return "%s (%s:%s): %s\n" % (category.__name__, filename, lineno, message)

warnings.formatwarning = formatwarning
warn = warnings.warn

set_printoptions(suppress=True,precision=6)

# set up functions necessary to load local modules first
def wait_exit(retval,msg=''):
    if msg != '':
        print(msg)
    raw_input()
    sys.exit(retval)
    
def get_env(var_name, default=None):
    # fall back to default when available without waiting
    env_value = os.getenv(var_name) or default
    if env_value is None:
        # error and wait if value not found and no default provided
        wait_exit(FAILURE,'Error (get_env): %s environment variable not defined'%var_name)
    return env_value

def runcmd(cmd,verbose=0):
    from shlex import split as shsplit
    from subprocess import Popen, PIPE
    cmdstr = ' '.join(cmd) if isinstance(cmd,list) else cmd
    cmdspl = shsplit(cmdstr)
    if len(cmdspl)==0:
        cmdspl.append("")
    if verbose:
        print('runcmd: executing command "%s"'%cmdspl[0],
              'with args',cmdspl[1:])
    cmdout = PIPE
    for rstr in ['>>','>&','>']:
        if rstr in cmdstr:
            cmdstr,cmdout = map(lambda s:s.strip(),cmdstr.split(rstr))
            mode = 'w' if rstr!='>>' else 'a'
            cmdout = open(cmdout,mode)

    stime = dtime_now()
    p = Popen(cmdstr.split(), stdout=cmdout, stderr=cmdout)
    out, err = p.communicate()
    retcode = p.returncode

    if cmdout != PIPE:
        cmdout.close()

    if verbose:
        cmdtime = dtime_now()-stime
        print("runcmd: finished executing command\n"
              "cputime=%s seconds, return code=%s."%(str(cmdtime),str(retcode)))

    return out,err,retcode

# global path variables (inferred with respect to this file)
PYORT_ROOT    = pathsplit(realpath(__file__))[0] # (i.e., directory of this file)
PYEXT_ROOT    = pathjoin(PYORT_ROOT,'external')
ORTHO_ROOT    = realpath(pathjoin(PYORT_ROOT,'..'))
PLATFORM_ROOT = realpath(pathjoin(PYORT_ROOT,'platform'))
CONFIG_FILE   = realpath(pathjoin(PYORT_ROOT,'config/pyorthorc.offline'))

# update python paths
FPT_ROOT = pathjoin(PYORT_ROOT,'find_pixel_trace_cython')
FPT_LIB = pathjoin(FPT_ROOT,'find_pixel_trace_cython.so')
SUNPOS_ROOT = pathjoin(PYEXT_ROOT,'sunpos-1.1')
SUNPOS_LIB = pathjoin(SUNPOS_ROOT,'_sunpos.so')

sys.path.extend([PYORT_ROOT,PYEXT_ROOT,FPT_ROOT,SUNPOS_ROOT])

PYEXT_PKG = ['LatLongUTMconversion','sun_dist','sunpos-1.1']
for pkg_path in PYEXT_PKG:
    sys.path.append(pathjoin(PYEXT_ROOT,pkg_path))

# local, external, and compiled imports
from LatLongUTMconversion import LLtoUTM, UTMtoLL
from sun_dist import sun_dist_au

# run build_link_cython.sh if FPT_LIB or SUNPOS_LIB not found
if not (pathexists(FPT_LIB) and pathexists(SUNPOS_LIB)):
    print('Building find_pixel_trace and sunpos libraries')
    curdir = os.getcwd()
    os.chdir(PYORT_ROOT)
    out,err,retcode = runcmd(pathjoin(PYORT_ROOT,'build_link_cython.sh')+' '+PYORT_ROOT)
    if retcode!=0:
        print('Error building the find_pixel_trace and sunpos libraries')
        print('stdout:\n',out,'\n')
        print('stderr:\n',err,'\n')
        sys.exit(retcode)
    os.chdir(curdir)
    print('Build successful')
    
try:
    import find_pixel_trace_cython.find_pixel_trace_cython as _fptlib

    def find_pixel_trace(xs,ys,xe,ye):
        """
        find_pixel_trace(xs,ys,xe,ye)
        
        Summary: computes 2d shortest path + l2 cost from (xs,ys) to (xe,ye)
        
        Arguments:
        - xs: xstart
        - ys: ystart
        - xe: xend
        - ye: yend
        
        Keyword Arguments:
        None
        
        Output:
        - output

        Examples:
        >>> find_pixel_trace(1,1,3,3)
        array([[1.      , 1.      , 0.      ],
               [1.      , 2.      , 1.      ],
               [2.      , 2.      , 1.414214],
               [2.      , 3.      , 2.236068],
               [3.      , 3.      , 2.828427]])
        """
        
        return _fptlib.find_pixel_trace_cython(xs,ys,xe,ye)
        
except Exception as e:
    msg = 'Error importing find_pixel_trace_cython (did you run build_link_cython.sh?)'
    warn(msg)
    sys.exit(FAILURE)


try:
    from sunpos import sunpos, cLocation, cTime, cSunCoordinates
except Exception as e:
    msg = 'Error importing sunpos library (did you run build_link_cython.sh?)'
    warn(msg)
    sys.exit(FAILURE)

    
double        = float64

# sentinel empty return value
EMPTY         = zeros(0)

# flag to indicate search failure (e.g., no science frames found in chunk)
NOTFOUND      = -1 

# georeferencing/gps/pps/frame constants
DATUM_WGS84   = 23
GPS_EPOCHJD   = 2444245
MAXINT14      = 2**14-1
g_14bit_mask  = uint32(MAXINT14)


OBC_DARK1     = 2
OBC_SCIENCE   = 3
OBC_DARK2     = 4
OBC_BRIGHTMED = 5
OBC_BRIGHTHI  = 6
OBC_LASER     = 7

OBC_STATUS_MSG = {OBC_DARK1:     "OBC_DARK1",
                  OBC_SCIENCE:   "OBC_SCIENCE",
                  OBC_DARK2:     "OBC_DARK2",
                  OBC_BRIGHTMED: "OBC_BRIGHTMED",
                  OBC_BRIGHTHI:  "OBC_BRIGHTHI",
                  OBC_LASER:     "OBC_LASER"}

# gps status messages in frame header
HxBABE       = uint32(47806) # gps aligned to pps
HxDEAD       = uint32(57005) # gps not aligned with pps (interpolate)
HxBAD        = uint32(2989) # gps with no pps

GPS_STATUS_MSG = {HxBABE:'HxBABE',HxDEAD:'HxDEAD',HxBAD:'HxBAD'}

# pixel-level ortho error codes
# IMPORTANT: all subsequent error codes must be less than PIX_ERROR_UNDEF
PIX_ERROR_UNDEF       = -9900 
PIX_ERROR_OUTSIDE_PB  = -9901
PIX_ERROR_NO_LOC      = -9902
PIX_ERROR_DEM_EXTENT  = -9903
PIX_ERROR_DEM_NOZINT  = -9904
PIX_DATA_IGNORE_VALUE = -9999

PIX_ERROR_MSG = {PIX_ERROR_UNDEF:       'uninitialized',
                 PIX_ERROR_OUTSIDE_PB:  'outside pushbroom bounds',
                 PIX_ERROR_NO_LOC:      'cannot determine ground location from PPS/GPS data',
                 PIX_ERROR_DEM_EXTENT:  'outside DEM extent',
                 PIX_ERROR_DEM_NOZINT:  'did not intersect DEM',
                 PIX_DATA_IGNORE_VALUE: 'nodata value'}

# downtrack averaging error codes
DT_ERROR_UNDEF        = PIX_ERROR_UNDEF

# lat/lon bounds for geoid interpolation
LATMIN,LATMAX = -90.0,  90.0
LONMIN,LONMAX =   0.0, 360.0        

DEG2RAD       = double(pi/180.0)
RAD2DEG       = 1.0/DEG2RAD

def basename(path):
    '''
    /path/to/file.ext -> file
    '''
    return splitext(pathsplit(path)[1])[0]

def filename2flightid(filename,check_prefix=True):
    '''
    get flight id from filename
    ang20160922t184215_cmf_v1g_img -> ang20160922t184215
    '''
    imgid = basename(filename)
    prefix = ('ang','prm','AVng','PRISM')
    if check_prefix and any([imgid.startswith(p) for p in prefix]):
        imgid = imgid.split('_')[0]
    
    return imgid

def datestr(fmt='%m%d%y',date=datetime.datetime.now()):
    return date.strftime(fmt)
    
def input_timeout(prompt, defstr='y', timeout=5):
    def interrupt_input(signum, frame):
        raise KeyboardInterrupt

    astring = defstr # return defstr if no input entered by timeout        
    signal.signal(signal.SIGALRM, interrupt_input)
    signal.alarm(timeout)
    try:
        astring = raw_input(prompt)
    except KeyboardInterrupt:
        pass
    signal.alarm(0) # disable the signal after we've gotten valid input
    return astring.strip()

def timeit(func):
    """
    timeit(func)

    Decorator to time the invocation of a function

    Arguments:
    - func: function to time
    
    Keyword Arguments:
    None
    
    Returns:
    - return value(s) of function func
    """
    
    from time import time
    outstr = '%s.%s elapsed time: %0.3f seconds'
    def wrapper(*args,**kwargs):        
        starttime  = time()        
        res = func(*args,**kwargs)        
        print(outstr%(func.__module__,func.func_name, time()-starttime))
        return res
    
    return wrapper

def envi_mapinfo(img,astype=dict):
    maplist = img.metadata.get('map info',None)
    if maplist is None:
        return None
    elif astype==list:
        return maplist
    
    if astype==str:
        mapstr = '{ %s }'%(', '.join(maplist))    
        return mapstr 
    elif astype==dict:
        from collections import OrderedDict
        if maplist is None:
            return {}
        
        mapinfo = OrderedDict()
        mapinfo['proj'] = maplist[0]
        mapinfo['xtie'] = float(maplist[1])
        mapinfo['ytie'] = float(maplist[2])
        mapinfo['ulx']  = float(maplist[3])
        mapinfo['uly']  = float(maplist[4])
        mapinfo['xps']  = float(maplist[5])
        mapinfo['yps']  = float(maplist[6])

        if mapinfo['proj'] == 'UTM':
            mapinfo['zone']  = maplist[7]
            mapinfo['hemi']  = maplist[8]
            mapinfo['datum'] = maplist[9]
        elif mapinfo['proj'] == 'Geographic Lat/Lon':
            mapinfo['ref'] = maplist[7]
            
        mapmeta = []
        for mapitem in maplist[len(mapinfo):]:
            if '=' in mapitem:
                key,val = map(lambda s: s.strip(),mapitem.split('='))
                mapinfo[key] = val
            else:
                mapmeta.append(mapitem)

        mapinfo['rotation'] = float(mapinfo.get('rotation','0'))
        if len(mapmeta)!=0:
            mapinfo['metadata'] = mapmeta

    return mapinfo

def float2byte(I,nodata,cliprange=0.99):
    Idata = I!=nodata
    Imin,Imax = percentile(I[Idata],[1.0-cliprange,cliprange])
    Inorm = clip(I,Imin,Imax)
    Inorm = uint8(255*(Inorm-Imin)/(Imax-Imin))    
    Inorm[~Idata] = 0
    return Inorm

def envi2jpeg(img_hdrf,img_jpgf,bands=[0]):
    from skimage.io import imsave
    img = envi_open(img_hdrf,image=img_hdrf.replace('.hdr',''))
    meta = img.metadata
    interleave = meta['interleave']
    nodata = float32(meta.get('data ignore value',-9999))
    imgmm  = img.open_memmap(interleave='source',writable=False)
    imgbands = []
    for band in bands:
        if interleave == 'bip':
            imgband = imgmm[:,:,band]
        elif interleave == 'bil':
            imgband = imgmm[:,band,:]

        scband = float2byte(array(imgband,dtype=float32),nodata)

        imgbands = scband[:,:,newaxis] if len(imgbands)==0 else dstack((imgbands,scband))
        
    imsave(img_jpgf,imgbands.squeeze())

def mask_undef(A):
    return A==PIX_ERROR_UNDEF

def mask_errors(A):
    # return boolean mask where all errors AND undefined pixels flagged
    return A<=PIX_ERROR_UNDEF
    
def any_errors(A,axis=None):    
    return mask_errors(A).any(axis=axis)

def all_errors(A,axis=None):    
    return mask_errors(A).all(axis=axis)
    
def get_projection(imgf):
    img = envi_open(imgf+'.hdr',image=imgf)
    map_info = img.metadata['map info'] 
    return map_info[0]

def get_igm_zone(igm):
    description = igm.metadata.get('description','')
    zstr_idx = description.find('zone')
    if zstr_idx == -1:
        warn('utm zone undefined in IGM file %s'%igm_hdrf)
        return '',''
    utm_toks = description[zstr_idx:].split()
    utm_zone = int(utm_toks[1])
    utm_hemi = 'North' if 'North' in utm_toks[2] else 'South'
    return utm_zone, utm_hemi

def rebin(a,f):
    """
    rebin(a,f) 

    Python version of IDL rebin: http://www.exelisvis.com/docs/REBIN.html
    Limitations: Handles 1-d case only, does NOT upsample.
    
    Arguments:
    - a: 1d input array length n
    - f: bin factor (>1; note: this differs from IDL argument "m" = a.size/f)
    
    Keyword Arguments:
    None
    
    Returns:
    - rebinned array of length m = len(a)/f
    
    Examples:
    >>> list(rebin(array([0, 10, 20, 30]),2))
    [5.0, 25.0]
    >>> list(rebin(array([1690014621, 1690015217, 1690015813, 1690016409, 1690017005, 1690017601, 1690018196, 1690018792, 1690019388, 1690019984, 1690020580, 1690021176, 1690021772, 1690022368, 1690022964, 1690023560, 1690024156, 1690024751],dtype=int),9))
    [1690017004.6666667, 1690022367.8888888]
    """

    if f>1:        
        n = len(a)
        if n % f != 0:
            warn('f must be an integer multiple of n')
            return a

        ff   = double(f)
        m    = int(n*(1.0/ff))
        b    = zeros(m)
        b[:] = [(a[f*i:f*(i+1)].sum())/ff for i in range(m)]

        #b = b-(b.min()-a.min())
        
        return b
    elif f<1:
        warn('rebin cannot upsample, returning original, unsampled array')
    # no resampling, return a
    return a

def extrema(a,**kwargs):
    '''
    extrema(a,**kwargs) -> float,float

    Computes the extrema of a list/array

    Arguments:
    - a: array like

    Keyword Arguments:
    - p: return (1-p),p percentiles instead of min/max (default=1.0)
    - keywards arguments passed to np.min/np.max

    Examples:
    >>> extrema([-0.538577, 2.501624, -1.227843,  1.469152])
    (-1.227843, 2.501624)
    '''
    p = kwargs.pop('p',1.0)
    if p==1.0:
        return np_min(a,**kwargs),np_max(a,**kwargs)
    assert(p>0.0 and p<1.0)
    axis = kwargs.pop('axis',None)
    apercent = lambda q: nanpercentile(a,axis=axis,q=q,interpolation='nearest')
    return apercent((1-p)*100),apercent(p*100)


def julday(month,day,year):
    '''
    given mm/dd/yy, return julian day
    '''
    a = (14 - month)//12
    y = year + 4800 - a
    m = month + 12*a - 3
    return day + ((153*m + 2)//5) + 365*y + y//4 - y//100 + y//400 - 32045

def gps2hour(gpstime):
    #sec_per_hour=3600 # 60**2
    #sec_per_day=86400 # 24*sec_per_hour
    # Examples:
    # >>> gpstime = 323562.00916666654
    # >>> gps2hour(gpstime)
    # 17.878335879629596
    
    return (gpstime % 86400)/3600.0

def gps2time(gpstime):
    return gps2hour(gpstime)

def file2date(rawf):
    import re
    m = re.match(r'^.*([0-9]{8}t[0-9]{6}).*', rawf)
    if m is None:
        warn('Malformed filename %s, cannot extract date'%rawf)
        return EMPTY

    sdate = m.group(1)
    year,month,day = map(int,[sdate[:4],sdate[4:6],sdate[6:8]])
    file_jd=julday(month,day,year)
    gps_week=int((file_jd-GPS_EPOCHJD)/7)    
    hour,minute,sec = map(double,[sdate[9:11],sdate[11:13],sdate[13:]])
    return year,month,day,gps_week,hour,minute,sec

def lsq1d(x,y):
    """
    lsq1d(x,y)
    
    Summary: given a pair of 1D vectors x,y, return f_{lsq}(x)->y
    
    Arguments:
    - x: 1xn independent variable vector
    - y: 1xn dependend variable
    
    Keyword Arguments:
    None
    
    Output:
    - linear lsq fit f_{lsq}(x)->y    
    """    
    from scipy.linalg import lstsq
    m, b = lstsq(np.vstack([x,np.ones_like(x)]).T,y)[0]
    flsq = lambda xi: m*xi + b
    return flsq

def compute_geoidtrace(geoid,lon,lat,dps,interp=True):
    '''
    '''
    geoidlat,geoidlon  = (lat-LATMAX)/(-dps),(lon-LONMIN)/dps
    if interp:
        # note: return scalar rather than 1x1 array
        return bilerp(geoid,geoidlat,geoidlon)[0] 
    return geoid[int(geoidlat),int(geoidlon)]

def bilerp(gridxy, gridxf, gridyf):
    """
    bilinear interpolation on a regular grid
    Examples:
    >>> bilerp(array([[0,2],[4,8]]),0.5,0)
    array([1.])
    >>> bilerp(array([[0,2],[4,8]]),0.0,0.5)
    array([2.])
    >>> bilerp(array([[0,2],[4,8]]),[0.5,0.5],[0,0.5])
    array([1. , 3.5])
    >>> bilerp(array([[0,2],[4,8]]),1,1)
    array([8.])
    """    
    return map_coordinates(gridxy,[[gridyf],[gridxf]],order=1,prefilter=False,
                           output=double).ravel()

def rotxy(x,y,adeg,xc,yc):
    """
    rotxy(x,y,adeg,xc,yc)

    Summary: rotate point x,y about xc,yc by adeg degrees

    Arguments:
    - x: x coord to rotate
    - y: y coord to rotate
    - adeg: angle of rotation in degrees
    - xc: center x coord
    - yc: center y coord

    Output:
    rotated x,y point
    """
    arad = DEG2RAD*adeg
    sinr,cosr = sin(arad),cos(arad)
    rotm = [[cosr,-sinr],[sinr,cosr]]
    xp,yp = dot(rotm,[x-xc,y-yc])+[xc,yc]
    return xp,yp

def lonlat2utm(lon,lat,zone=None):
    '''
    Examples:
    >>> lon,lat,zone = -90.0, 17.7, 16
    >>> x,y,zone,zonealpha=lonlat2utm(lon,lat,zone=zone)
    >>> '%.3f, %.3f, %d, %s'%(x,y,zone,zonealpha)
    '181760.100, 1959529.923, 16, Q'
    >>> lon,lat = utm2lonlat(y,x,zone,zonealpha)
    >>> '%.1f, %.1f'%(lon,lat)
    '-90.0, 17.7'
    >>> lon,lat,zone = 147.8413733, -16.9661409, 55
    >>> x,y,zone,zonealpha=lonlat2utm(lon,lat,zone=zone)
    >>> '%.3f, %.3f, %d, %s'%(x,y,zone,zonealpha)
    '589577.252, 8123998.687, 55, K'
    >>> lon,lat = utm2lonlat(y,x,zone,zonealpha)
    >>> '%.1f, %.1f'%(lon,lat)
    '147.8, -17.0'
    '''
    UTMZone, UTMEasting, UTMNorthing = LLtoUTM(DATUM_WGS84,lat,lon,ZoneNumber=zone)
    return UTMEasting, UTMNorthing, int(UTMZone[:-1]), UTMZone[-1]

def utm2lonlat(y,x,utm_zone,utm_alpha):
    '''
    Examples:
    >>> x, y, zone, zonealpha = 181760.100,1959529.923,16,'Q'
    >>> lon,lat = utm2lonlat(y,x,zone,zonealpha)
    >>> '%.1f, %.1f'%(lon,lat)
    '-90.0, 17.7'
    >>> x,y,zone,zonealpha = lonlat2utm(lon,lat,zone=zone)
    >>> '%.3f, %.3f, %d, %s'%(x,y,zone,zonealpha)
    '181760.100, 1959529.923, 16, Q'
    '''
    lat,lon = UTMtoLL(DATUM_WGS84,double(y),double(x),str(utm_zone)+utm_alpha)
    return lon,lat

def map2sl(x, y, ulx, uly, ps):
    """
    map2sl(x,y,ulx,uly,ps) 

    Given a defined grid find the s,l values for a given x,y
    
    Arguments:
    - x,y: map coordinates
    - ulx,uly: upper left map coordinate 
    - ps: map pixel size
    
    Keyword Arguments:
    None
    
    Returns:
    - s,l: sample line coordinates of x,y
    
    Examples:
    >>> ulx,uly = 0,1000.0
    >>> lrx,lry = 1000.0,0.0
    >>> map2sl(lrx,lry,ulx,uly,1)
    (1000.0, 1000.0)
    >>> map2sl(ulx,lry,ulx,uly,1)
    (0.0, 1000.0)
    >>> map2sl(lrx,uly,ulx,uly,1)
    (1000.0, 0.0)
    >>> s,l = map2sl(ulx,uly,ulx,uly,1)
    >>> sl2map(s,l,ulx,uly,1)
    (0.0, 1000.0)
    """
    return (x-ulx)/ps, (uly-y)/ps

def sl2map(s,l,ulx,uly,ps):
    """
    sl2map(s,l,ulx,uly,ps) 

    Given pixel coordinates (s,l) convert to UTM map coordinates mapX,mapY
    
    Arguments:
    - s,l: sample, line indices
    - ulx,uly: upper left map coordinate 
    - ps: map pixel size 
    
    Keyword Arguments:
    None
    
    Returns:
    - mapX,mapY: s,l in map coordinates

    Examples:
    >>> ulx,uly = 0,1000.0
    >>> s,l = map2sl(ulx,uly,ulx,uly,1)
    >>> s,l
    (0.0, 0.0)
    >>> sl2map(s,l,ulx,uly,1)
    (0.0, 1000.0)
    
    """
    return ulx+ps*s, uly-ps*l

def subset_igm(igmf,lon,lat,nbb):
    """
    subset_igm(igm,lon,lat,bbrad)
    
    Summary: get the bounding box in pixel coordinates to extract a
    [mh x mw] region surrounding coordinate (lon,lat) from an igm file
    
    Arguments:
    - igm: igm file
    - lon: longitude of bbox center
    - lat: latitude of bbox center
    - nbb: dim of bbox in pixels
    
    Keyword Arguments:
    None
    
    Output:
    - [nbb*nbb x 4] matrix of [row,col,utmx,utmy] values
    '''
    Examples:
    >>> lon,lat = -117.336488,33.965377
    >>> igmf = '/Users/bbue/Research/range/watch_out/pyortho-latest/ang20140612t204858_rdn_igm'
    >>> igmout = subset_igm(igmf,lon,lat,2)
    """
    igm_img = envi_open(igmf+'.hdr',image=igmf)
    utm_zone,utm_hemi = get_igm_zone(igm_img)
    bin_factor = int(igm_img.metadata['line averaging'])
    print(bin_factor)
    igm_mm  = igm_img.open_memmap(interleave='source')
    utmx,utmy,utm_zone,utm_alpha = lonlat2utm(lon,lat)
    nbr,nbc = nbb,nbb
    nrows,ncols = igm_mm.shape[:2]
    dist = sqrt((igm_mm[:,:,0]-utmx)**2+(igm_mm[:,:,1]-utmy)**2)
    cy,cx = map(lambda idx: idx[0],where(dist==dist.min()))
    rowmin,rowmax = max(0,cy-nbr),min(nrows-1,cy+nbr+1)
    colmin,colmax = max(0,cx-nbc),min(ncols-1,cx+nbc+1)
    h,w = rowmax-rowmin,colmax-colmin
    rows = arange(rowmin,rowmax)
    cols = arange(colmin,colmax)
    rr,cc = meshgrid(rows,cols)
    print(igm_mm.shape)
    utmxv = igm_mm[rr,cc,0].ravel()
    utmyv = igm_mm[rr,cc,1].ravel()
    print(len(rr),utmxv.shape)
    print(lon,lat,utmx,utmy,cx,cy,utm_zone,len(utmxv),h,w)

    return rr,cc,utmxv,utmyv

def array2rgba(a,cmap=None,**kwargs):
    import pylab as pl
    from numpy import clip,uint8
    vmin = float(kwargs.pop('vmin',a.min()))
    vmax = float(kwargs.pop('vmax',a.max()))
    an = clip(((a-vmin)/(vmax-vmin)),0.,1.)
    if cmap is None:
        cmap = pl.get_cmap(pl.rcParams['image.cmap'])
    elif isinstance(cmap,str):
        cmap = pl.get_cmap(cmap)
    return cmap(an)

def rotate_camera(camera_model,rot=0,reversed=False):
    det_ids = arange(camera_model.shape[1])
    new_model = (camera_model[:,det_ids[::-1]] \
                 if reversed else camera_model).copy()
    if rot==0:
        return new_model        
    ar    = DEG2RAD*rot
    sinr  = sin(ar)
    cosr  = cos(ar)
    rotm = float64([[cosr,-sinr, 0.],
                    [sinr, cosr, 0.],
                    [0.,   0.,   1.]])
    new_model = dot(rotm.T,new_model)
    #print('camera_model.shape',camera_model.shape)
    #print('new_model.shape',new_model.shape)

    return new_model

def plot_trajectory(frame_pos,frame_clock,fig=None,ax=None,cmap=None):
    import pylab as pl
    from mpl_toolkits.mplot3d import Axes3D
    fig = fig or pl.figure()
    ax3 = ax or fig.add_subplot(111, projection='3d')
    lat,lon,alt = frame_pos[:,:3].T
    print(lat,lon,alt,len(lon),len(alt))
    x,y = zeros(len(lat)),zeros(len(lon))
    x[0], y[0], ul_zone, ul_alpha = lonlat2utm(lon[0], lat[0])    
    for i,(loni,lati) in enumerate(zip(lon,lat)):
        if i==0:
            continue
        x[i], y[i], ul_zone, ul_alpha = lonlat2utm(lon[i], lat[i], zone=ul_zone)
    pitch,roll,head = frame_pos[3:6]

    scale = 1.0 #111120.0
    plot3 = ax3.scatter(x,y,alt,c=array2rgba(linspace(0,1,len(lon)),cmap=cmap))
    
    
def plot_camera(camera_model,reversed=False,fig=None,ax=None,c=None):
    import pylab as pl
    from mpl_toolkits.mplot3d import Axes3D
    pb_len = int(camera_model.shape[1])
    det_ids = arange(camera_model.shape[1])
    if reversed:
        det_ids = det_ids[::-1]
    s = 30
    pb_s = pb_len//s
    x = arange(1,pb_len+1,s)
    y,z = zeros(len(x)),zeros(len(x))
    u,v,w = camera_model[:,::s]
    _wmin,_wmax = extrema(w)
    _wdif = (_wmax-_wmin)
    _wsep = _wdif*0.05
    if c is None:
        c = det_ids/camera_model.shape[1]
        c = r_[c,c_[c,c].ravel()]
        cmap = pl.get_cmap(pl.rcParams['image.cmap'])    
        c = array2rgba(c,cmap=cmap)
        
        print(c[::12][:,0])
    # repeat to color arrow heads as well

    print('plotting camera model with %d samples/frame'%pb_len)
    fig = fig or pl.figure()
    ax3 = ax or fig.add_subplot(111, projection='3d')
    qout3 = ax3.quiver(x,y,z,-u,-v,-w,pivot='tail',length=_wdif/32,
                       arrow_length_ratio=0.1,
                       linewidths=1,colors=c)
    ax3.set_xlabel('pushbroom')
    ax3.set_ylabel('v')
    ax3.set_zlabel('w')    
    ax3.set_xlim(-2,pb_len+3)
    _vmin,_vmax = extrema(v)
    _vsep = (_vmax-_vmin)*0.005
    ax3.set_ylim(-_vsep,_vsep) #_vmin-_vsep,_vmax+_vsep)
    #ax3.set_yticks([-0.1,-0.05,0,0.05,0.1])
    xt = ['1']+(['']*2)+['%4.1f'%(pb_len/2)]+(['']*2)+['%d'%pb_len]
    ax3.set_xticklabels(xt)
    ax3.set_yticklabels(['','','','',''])
    ax3.set_zlim(-_wsep,_wsep)
    ax3.set_zticklabels(['','','','',''])

def plane2world_camera_model(pitch,roll,heading,camera,flip=False):
    """
    plane2world_camera_model(pitch,roll,heading,camera) 

    Given the current (pitch,roll,heading), map camera model into
    east north up (enu) frame
    
    Arguments:
    - pitch roll heading: in radians                  
    - camera:    [3 x pushbroom_length] frame offsets
    
    Keyword Arguments:
    None
    
    Returns:
    [3 x pushbroom length] camera model
    """
    
    # C matrix converts from aero to enu frame
    C = [[0.,  1.,  0.],
         [1.,  0.,  0.],
         [0.,  0., -1.]]

    cos_roll,sin_roll = cos(roll),sin(roll)
    R_r = [[cos_roll, 0., -sin_roll], 
           [0.,       1.,  0.], 
           [sin_roll, 0.,  cos_roll]]
    
    cos_pitch,sin_pitch = cos(pitch),sin(pitch)
    R_p = [[1.,  0.,         0.], 
           [0.,  cos_pitch,  sin_pitch], 
           [0., -sin_pitch,  cos_pitch]]    
    
    cos_heading,sin_heading = cos(heading),sin(heading)
    R_h = [[cos_heading, -sin_heading, 0.],
           [sin_heading,  cos_heading, 0.], 
           [0.,           0.,          1.]]
    
    if flip:
       orient = DEG2RAD*180.0 
       cos_orient,sin_orient = cos(orient),sin(orient)
       R_o = [[cos_orient, -sin_orient, 0.],
              [sin_orient,  cos_orient, 0.], 
              [0.,           0.,          1.]]
       return dot(dot(dot(dot(dot(R_r,R_p),R_h),R_o).T,C),camera)
       
    # M= R_r*R_p*R_h = mout of navout_mout
    #return asarray((((asmatrix(R_r)*R_p)*R_h).T*C)*camera)
    return dot(dot(dot(dot(R_r,R_p),R_h).T,C),camera)

def gc_rad(dlat):
    a,b = 6378.1370,6356.7523
    a2,b2 = a*a,b*b
    dlatr = DEG2RAD*dlat
    cosr,sinr = cos(dlatr),sin(dlatr)
    return sqrt( ( (power(a2*cosr,2.) + power(b2*sinr,2.)) /
                   (power(a*cosr,2.)) + power(b*sinr,2.)) )

def gc_distance(dlon1, dlat1, dlon2, dlat2):
    """
    gc_distance(dlon1, dlat1, dlon2, dlat2)

    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    Input:
    - dlon1,dlat1: first lat/lon coordinate
    - dlon2,dlat2: second lat/lon coordinate

    Returns:
    - distance between (dlon1,dlat1) and (dlon2,dlat2)
    """
    # convert decimal degrees to radians haversine formula
    lon1,lat1,lon2,lat2 = DEG2RAD*array([dlon1,dlat1,dlon2,dlat2])    
    a = power(sin((lat2-lat1)/2.),2.) + cos(lat1) * cos(lat2) * \
        power(sin((lon2-lon1)/2.),2.)
    return arcsin(sqrt(a))*12742000.0 # earthrad=6371km*1000m*2

def format_clock_words(msw,lsw):
    return (int64(msw)<<16)+lsw

def extract_fc(word):
    # applies 14 bit mask to extract frame count from 16-bit pps/gps word
    return bitwise_and(g_14bit_mask, int32(word))

def read_frames_meta_bands(img_path, platform, start_line=0, num_lines=99999,
                           alloc_lines=1000, bands=[]):
    if not pathexists(img_path):
        warn('image file %s not found!'%img_path)
        return EMPTY,EMPTY,0

    if len(bands) == 0:
        bands = arange(NC)

    NC = platform.NC
    NS = platform.NS
    NRAW = platform.NRAW
    NFRAME = platform.NFRAME    

    img_size = getsize(img_path)
    num_read = 0
    num_bands = len(bands)
    
    alloc_meta = zeros([alloc_lines,2],dtype=int64)
    meta = alloc_meta.copy()

    
    read_bands = False if num_bands==0 else True
    if read_bands:
        img_bands = zeros([alloc_lines,NS,num_bands])
        alloc_bands = img_bands.copy()
        band_buf = zeros([NS,num_bands])

    obc_start = NOTFOUND # odbc start frame
    obcv_prev = NOTFOUND
    done = False
    
    def collect_bands(f_ptr,b_buf,band_idx,band_type='<u2'):
        # parse band_idx bands starting after frame pointer f_ptr metadata
        assert(list(band_idx)==sorted(band_idx))
        f_start = f_ptr.tell()
        b_prev = 0
        for jb,b in enumerate(band_idx):
            f_ptr.seek((b-b_prev)*NS*2,SEEK_CUR)
            bj = fromfile(f_ptr, count=NS, dtype=band_type)
            if len(bj) < NS:                            
                break

            b_buf[jb,:] = bj
            b_prev = b        
        f_ptr.seek(f_start) # rewind to frame index to advance
        return jb
    
    with open(img_path, 'r') as f:
        if start_line > 0:
            f.seek(start_line*NRAW*2,SEEK_CUR)
            if f.tell() >= img_size:
                done=True

        while not done and num_read < num_lines:
            # get frame header
            buf = fromfile(f, count=NS, dtype='<u2')
            if len(buf) > 0:
                clock = format_clock_words(buf[0],buf[1])
                count = extract_fc(buf[160])
                #int64(bitwise_and(uint32(g_14bit_mask), uint32(buf[160])))
                obcv  = buf[321] >> 8
                if obcv == OBC_SCIENCE and obc_start == NOTFOUND:
                    obc_start = num_read

                b_read = collect_bands(f,b_buf,bands)
                
                if b_read == num_bands:
                    if num_read+1 == meta.shape[0]:
                        meta  = r_[meta, alloc_meta.copy()]
                        img_bands = r_[img_bands, alloc_bands.copy()]
                       
                    meta[num_read,:]  = [clock, count]
                    img_bands[num_read,:,:] = b_buf

                    obcv_prev = obcv
                    num_read += 1
                else:
                    done = True
                f.seek(NC*NS*2,SEEK_CUR)
            else:
                done = True

        img_bands = img_bands[:num_read,:]
        meta = meta[:num_read,:]
    #print("read_frames_meta_bands: start_line=%d, num_read=%d"%(start_line,num_read))
    return img_bands, meta, num_read, obc_start

def read_frames_meta(img_path, platform, start_line=0, num_lines=inf,
                     alloc_lines=1000, obc_start=NOTFOUND,
                     return_obcv=False, transition_warnings=True,
                     verbose=False):
    NC = platform.NC
    NS = platform.NS
    RAWBYTES = platform.NRAW*2
    FRAMEBYTES = platform.NFRAME*2

    num_read = 0

    obc_start = NOTFOUND # first obc science header, offset by start_line        
    obcv_prev = NOTFOUND # to keep track of state transitions
    
    if not pathexists(img_path):
        warn('image file %s not found!'%img_path)
        return EMPTY, num_read, obc_start

    img_size = getsize(img_path)
    if img_size % RAWBYTES != 0:
        warn('file "%s" contains truncated frames'%pathsplit(img_path)[1])
    
    nl_max = int(float(img_size)/RAWBYTES)
    if start_line < 0: # indexing from EOF
        start_line = nl_max+start_line

    if start_line < 100:
        warn('obc_start value may be incorrect for small start_line offsets')
    
    if start_line > nl_max:
        warn('start_line > max frames in file, nothing to read')
        return EMPTY, num_read, obc_start

    frame_obcv = zeros(alloc_lines,dtype=int32)
    frame_meta = zeros([alloc_lines,2],dtype=int64)
    with open(img_path, 'r') as f:
        if start_line > 0:
            f.seek(start_line*RAWBYTES,SEEK_CUR)
            nl_max = nl_max-start_line
            
        nl_max = min(nl_max,num_lines) # don't read beyond EOF
        while num_read < nl_max:
            # read frame header
            buf = fromfile(f, count=NS, dtype='<u2')
            if len(buf) > 0:
                obcv  = buf[321] >> 8                
                if verbose:
                    print('num_read:',num_read,
                          'start_line+num_read:',start_line+num_read,
                          'obcv:',obcv,
                          'obcv_prev:',obcv_prev)

                if obc_start == NOTFOUND:
                    if obcv == OBC_SCIENCE:
                        if transition_warnings and obcv_prev != OBC_DARK1:
                            warn('obc_start not at OBC_DARK1->OBC_SCIENCE transition')
                        obc_start = num_read
                elif obcv == OBC_DARK2 and obcv_prev == OBC_SCIENCE:
                    tn_idx = start_line+num_read-1
                    if transition_warnings:
                        warn('OBC_SCIENCE->OBC_DARK2 transition detected at index %d'%tn_idx)
                    break
                obcv_prev = obcv

                #gps_state = buf[340]
                #print(GPS_STATUS_MSG[gps_state])
                #if gps_state==HxBAD:
                #    warn('Bad GPS msg found')
                clock = format_clock_words(buf[0],buf[1])
                count,countN = extract_fc(buf[160]),extract_fc(buf[319])
                if verbose:
                    print('clock:',clock,'count:',count,'countN:',countN)
                    
                if num_read+1 == frame_meta.shape[0]:
                    frame_meta = r_[frame_meta, zeros([alloc_lines,2],dtype=int64)]
                    frame_obcv = r_[frame_obcv, zeros(alloc_lines,dtype=int32)]
                    
                frame_meta[num_read,:]  = [clock, count]
                frame_obcv[num_read] = obcv
                num_read = num_read+1
                f.seek(FRAMEBYTES,SEEK_CUR)                
            else:
                break

        frame_meta = frame_meta[:num_read,:]
        frame_obcv = frame_obcv[:num_read]
        
    if transition_warnings and obc_start==NOTFOUND:
        warn('no OBC_SCIENCE frames read in range [%d,%d]'%(start_line,start_line+num_read))
        
    #print("read_frames_meta: start_line=%d,num_read=%d"%(start_line,num_read))
    if return_obcv:        
        return frame_meta, num_read, obc_start, frame_obcv
    
    return frame_meta, num_read, obc_start

def find_science_frames(rawf,platform,start_line,num_lines=500,collect=0,verbose=0):
    print('Finding science frames for "%s" (this may take awhile)'%(pathsplit(rawf)[1]))
    img_sl = img_nl = NOTFOUND
    chunk_nl = chunk_sl = 0
    fc0 = NOTFOUND
    raw_nl = start_line # number of raw lines parsed
    n_buf = 0 # number of frame buffer chunks of size num_lines parsed
    frames = EMPTY
    while True:
        #print(n_buf,img_sl,img_nl,chunk_nl,chunk_sl,raw_nl)
        frame_chunk = read_frames_meta(rawf,platform,start_line=raw_nl,num_lines=num_lines)
        chunk_meta,chunk_nl,chunk_sl = frame_chunk
        if chunk_nl == 0:            
            if img_sl == NOTFOUND:
                # no science frames found or EOF
                print('No science frames found in raw file')
                break
            else: # EOF
                # all done!
                print('Reached EOF')                
                break                
        elif img_sl == NOTFOUND and chunk_sl != NOTFOUND:
            # found first science frame
            img_sl = raw_nl+chunk_sl # keep track of starting frame index
            img_sl += platform.shutter_offset # drop a few frames in case the shutter isn't fully open
            if verbose:
                print('Science frame found at index %d'%img_sl)
                print('Rewinding frame pointer at %d by %d frames to %d'%(raw_nl+chunk_nl,
                                                                          raw_nl-img_sl,
                                                                          img_sl))
            raw_nl = img_sl # move pointer to index of science frame
            chunk_sl = chunk_nl = 0
            continue # rewind to populate buffer from first science frame
        elif chunk_sl == NOTFOUND:
            if img_sl != NOTFOUND: 
                print('All science frames found')
                img_nl = raw_nl
                break
            elif img_sl == NOTFOUND:
                # all frames in chunk non-science frames
                if verbose:
                    print('No science frames found in buffer starting at frame %d'%raw_nl)            
                raw_nl += chunk_nl            
                continue
 
        if collect:
            if fc0 == NOTFOUND:
                fc0 = chunk_meta[0,1]
            frame_meta = chunk_meta[:chunk_nl]
            frames = r_[frames,frame_meta] if len(frames)!=0 else frame_meta
        else:
            # since we're not collecting frames, just find first/last index
            pass
        
        raw_nl += chunk_nl
        n_buf += 1

    if collect:
        fcmod = MAXINT14+1
        frames[:,1] = (frames[:,1] - fc0) % fcmod
        zp = where(frames[:,1]==0)[0]        
        for z in zp[zp!=0]:
            frames[z:,1] += fcmod

        frames[:,1] += img_sl
        
    return img_sl, img_nl, frames

def interp_xymask(data,mask,lx,ty,ps,nbuf=10,method='nearest'):
    """
    interp_xymask(data,mask,lx,ty,ps,nbuf=10,method='nearest')

    interpolates masked coordinates in data wrt top left coordinate (lx,ty)

    Arguments:
    - data: n x m array with values to interpolate
    - mask: n x m binary mask specifying which data values to interpolate
    - lx: leftmost coordinate x-value
    - ty: topmost coordinate y-value
    - ps: pixel size
    
    Keyword Arguments:
    - nbuf: number of pixels to use to buffer mask bounding box (default=10)

    Returns:
    - values of interpolated coordinates where mask==True
    """
    from scipy.interpolate import griddata

    mask_rows,mask_cols = where(mask)
    rmin,rmax = extrema(mask_rows)
    cmin,cmax = extrema(mask_cols)

    rmin,rmax = max(0,rmin-nbuf),min(data.shape[0],rmax+nbuf+1)
    cmin,cmax = max(0,cmin-nbuf),min(data.shape[1],cmax+nbuf+1)

    data_rc = uint64(c_[map(ravel,mgrid[rmin:rmax,cmin:cmax])].T)
    mask_rc = uint64(c_[mask_rows,mask_cols])

    data_rows,data_cols = setdiff2d(data_rc,mask_rc).T
    data_xy = c_[sl2map(data_cols,data_rows,lx,ty,ps)]
    # provide data_vals in x,y (not r,c) order
    data_vals = data.T[data_cols,data_rows] 
    mask_xy = c_[sl2map(mask_cols,mask_rows,lx,ty,ps)]
    intp_vals = griddata(data_xy, data_vals, mask_xy, method=method)
    
    return intp_vals

def subset_latlon_reproject_orig(imgf,subset_extent=[],utm_ps=None,
                                 interp_nodata=True,verbose=0):
    '''
    extract a bounding box subset_extent (format=[ullon,ullat,lrlon,lrlat]) of imgf
    reproject from wgs84 to UTM with pixel size utm_ps
    - if subset_extent=[] and  imgf_ps == utm_ps and imgf_rot == 0 -> loads imgf
    - assumes imgf in wgs84 lat/lon, does not check!
    - assumes subset_extent contained entirely within imgf
    - does not handle images that overlap two utm zones!
    '''

    try:
        from osgeo import gdal, osr
    except Exception as e:
        warn('subset_latlon_reproject requires GDAL')
        return zeros(0),zeros(0)

    g = gdal.Open(imgf, gdal.GA_ReadOnly)

    # due to GDAL bug, g doesn't contain any metadata!
    metadata = read_envi_header(imgf+'.hdr')
    nodata_val = metadata['data ignore value']
    
    nl_in,ns_in = g.RasterYSize,g.RasterXSize
    geo_t = g.GetGeoTransform()
    geo_p = g.GetProjectionRef() 
    l_in, t_in, rot_in = geo_t[0],geo_t[3],geo_t[2]
    xps_deg, yps_deg = geo_t[1],abs(geo_t[5])    

    if utm_ps is None:
        # get x,y pixel sizes in meters and use the largest for resampling
        xps_m = gc_distance(l_in,t_in,l_in+xps_deg,t_in        )
        yps_m = gc_distance(l_in,t_in,l_in        ,t_in-yps_deg)
        utm_ps = max(xps_m,yps_m)

    if len(subset_extent) == 0:
        # reproject full image
        r_in,b_in = l_in+xps_deg*ns_in, t_in-yps_deg*nl_in
        ullon,ullat,lrlon,lrlat = l_in,t_in,r_in,b_in
    elif len(subset_extent) == 4:
        ullon,ullat,lrlon,lrlat = subset_extent
    else:
        warn('subset_extent should be in the form [ullon,ullat,lrlon,lrlat]')
        return EMPTY,EMPTY

    if verbose:
        print('-> Lat/Lon [ullon,ullat,lrlon,lrlat] extent=',array(subset_extent))    
            
    ul_hemi = 'North' if ullat >= 0 else 'South'        
    lx, ty, ul_zone, ul_alpha = lonlat2utm(ullon, ullat)
    rx, by, lr_zone, lr_alpha = lonlat2utm(lrlon, lrlat)
    if lr_zone != ul_zone:
        # need to recompute (rx, by) wrt UL zone
        warn('UL UTM zone (%d) != LR UTM zone (%d), assuming zone of UL pixel.'%(ul_zone,lr_zone))
        rx, by, lr_zone, lr_alpha = lonlat2utm(lrlon, lrlat, zone=ul_zone)
    
    imgdir,imgfile = pathsplit(imgf)
    print('Extracting UTM subset from Geographic Lat/Lon input file %s'%imgfile)
    starttime = dtime_now()

    rot = 0
    if geo_t[2] != 0 or geo_t[4] != 0:
        warn('image %s is not in a north-up coordinate system'%imgf)
        if geo_t[2] == geo_t[4]:
            rot = geo_t[2]
        else:
            warn('ill-defined rotation')
            return EMPTY,EMPTY
        
    # assume a single zone / spatial reference
    utm_zone = ul_zone
    utm_sr   = osr.SpatialReference()
    epsg_id  = int((32600 if ul_hemi=='North' else 32700)+utm_zone)
    utm_sr.ImportFromEPSG(epsg_id)
    utm_wkt = utm_sr.ExportToWkt()

    if verbose:
        print('-> UTM ps=%f, [minx,miny,maxx,maxy] extent='%utm_ps,array([lx,by,rx,ty]))
        print('->     zone %d (alpha=%s, hemisphere=%s)'%(ul_zone,ul_alpha,ul_hemi))
    
    wgs84_sr = osr.SpatialReference()
    wgs84_sr.ImportFromEPSG(4326)
    wgs84_wkt = wgs84_sr.ExportToWkt()

    # The size of the raster is given the new projection and pixel spacing
    # Using the values we calculated above. 
    ns = int(((rx-lx)/utm_ps))+1
    nl = int(((ty-by)/utm_ps))+1

    if ns<0 or nl<0: 
        warn('illegal subset dimensions %d x %d'%(ns,nl))
        return EMPTY,EMPTY

    # update rx,by based on rounded coordinates
    rx,by = sl2map(ns,nl,lx,ty,utm_ps)
    if verbose:
        print('->     subset ns=%d, nl=%d'%(ns,nl))

    # use in-memory driver for the quickness
    img_drv = gdal.GetDriverByName('MEM')
    dest = img_drv.Create('', ns, nl, 1, gdal.GDT_Float32)

    # reproject to uniform pixel size here
    dest.SetGeoTransform((lx, utm_ps, rot_in, ty, rot_in, -utm_ps))
    dest.SetProjection(utm_wkt)
    
    # Perform the projection/resampling to uniform pixel size 
    res = gdal.ReprojectImage(g, dest, wgs84_wkt, utm_wkt, gdal.GRA_Bilinear)

    dest_img = float64(dest.ReadAsArray().squeeze())
    nodata_mask = dest_img==float64(nodata_val)

    if nodata_mask.any():
        n_nodata = npnodata_mask.sum()
        if interp_nodata:
            #  TODO (BDB, 08/19/16): conncomp would probably make more sense 
            warn('interpolating %d NODATA (=%d) pixels'%(n_nodata,nodata_val))
            ccomp = bwlabel(int8(nodata_mask),background=0)
            ucomp = unique(ccomp[ccomp!=0])
            print('extrema(dest_img) before interpolation: "%s"'%str(extrema(dest_img)))                
            for l in ucomp:
                maskl = ccomp==l
                dest_img[maskl] = interp_xymask(dest_img,maskl,lx,ty,utm_ps)
            print('extrema(dest_img) after interpolation: "%s"'%str(extrema(dest_img)))
        else:
            # zero out nodata values
            warn('zeroing %d NODATA (=%d) pixels'%(n_nodata,nodata_val))
            dest_img[nodata_mask] = 0

    dest_extrema = extrema(dest_img)
    dest_extent  = rx,by,lx,ty 
    dest_size    = nl,ns
    dest_meta    = dict(extent=dest_extent,extrema=dest_extrema,dims=dest_size,
                        zone=ul_zone,alpha=ul_alpha,hemi=ul_hemi,rot=rot,
                        ps=utm_ps)    

    if verbose:
        print('->     extrema=[%f,%f]'%(dest_extrema[0],dest_extrema[1]))
        print('-> Output extent=', str(array(dest_extent)).replace(' ',''))

    print('Extraction CPUtime (MM:SS.ms):      %s'%time_elapsed(starttime))
    return dest_img, dest_meta

def gdal_tx(epsg_to,epsg_from=4326):
    from osgeo import gdal,osr
    sr_to = osr.SpatialReference()
    sr_to.ImportFromEPSG (epsg_to)
    sr_from = osr.SpatialReference()
    sr_from.ImportFromEPSG (epsg_from)
    tx = osr.CoordinateTransformation(sr_from,sr_to)
    return tx,sr_to,sr_from

def gdal_save(outf,data,geoproj,geotrans,cols,rows):
    from osgeo import gdal
    print('saving',outf)
    envi_drv = gdal.GetDriverByName('ENVI')
    sub_ds = envi_drv.Create(outf, cols, rows, 1, gdal.GDT_Float32)
    sub_ds.SetGeoTransform(geotrans)
    sub_ds.SetProjection(geoproj)
    sub_ds.GetRasterBand(1).WriteArray(data)

def gdal_reproject(g, pixel_spacing, epsg_to, epsg_from=4326, outf=None):
    """
    A sample function to reproject and resample a GDAL dataset from within 
    Python. The idea here is to reproject from one system to another, as well
    as to change the pixel size. The procedure is slightly long-winded, but
    goes like this:
    
    1. Set up the two Spatial Reference systems.
    2. Open the original dataset, and get the geotransform
    3. Calculate bounds of new geotransform by projecting the UL corners 
    4. Calculate the number of pixels with the new projection & spacing
    5. Create an in-memory raster dataset
    6. Perform the projection
    """
    from osgeo import gdal

    tx,sr_to,sr_from = gdal_tx(epsg_to,epsg_from)
    
    geo_t = g.GetGeoTransform()
    x_size = g.RasterXSize # Raster xsize
    y_size = g.RasterYSize # Raster ysize

    # Work out the boundaries of the new dataset in the target projection
    (ulx,uly,ulz) = tx.TransformPoint(geo_t[0], geo_t[3])
    (lrx,lry,lrz) = tx.TransformPoint(geo_t[0] + geo_t[1]*x_size,
                                      geo_t[3] + geo_t[5]*y_size)

    # See how using 27700 and WGS84 introduces a z-value!
    # Now, we create an in-memory raster
    img_drv = gdal.GetDriverByName('MEM')
    # The size of the raster is given the new projection and pixel spacing
    # Using the values we calculated above. Also, setting it to store one band
    # and to use Float32 data type.
    x_out=int64((lrx-ulx)/pixel_spacing)
    y_out=int64((uly-lry)/pixel_spacing)

    dest = img_drv.Create('', x_out, y_out, 1, gdal.GDT_Float32)
    # Calculate the new geotransform
    new_geo = (ulx, pixel_spacing, geo_t[2],
               uly, geo_t[4], -pixel_spacing)
    # Set the geotransform
    dest.SetGeoTransform(new_geo)
    dest.SetProjection(sr_to.ExportToWkt())
    # Perform the projection/resampling 
    res = gdal.ReprojectImage(g, dest,
                              sr_from.ExportToWkt(),
                              sr_to.ExportToWkt(),
                              gdal.GRA_Bilinear)

    if outf:
        data = float32(dest.ReadAsArray().squeeze())
        gdal_save(subf,data,sr_to.ExportToWkt(),new_geo,x_out,y_out)
    
    return dest, (ulx,uly), (lrx,lry), new_geo

def subset_latlon_reproject(imgf,subset_extent=[],utm_ps=None,
                            interp_nodata=True,verbose=0):
    '''
    extract a bounding box subset_extent (format=[ullon,ullat,lrlon,lrlat]) of imgf
    reproject from wgs84 to UTM with pixel size utm_ps
    - if subset_extent=[] and  imgf_ps == utm_ps and imgf_rot == 0
      -> loads entire imgf
    - assumes imgf in wgs84 lat/lon, does not check!
    - assumes subset_extent contained entirely within imgf
    '''

    try:
        from osgeo import gdal, osr
    except Exception as e:
        warn('subset_latlon_reproject requires GDAL')
        return zeros(0),zeros(0)
    
    imgdir,imgfile = pathsplit(imgf)
    print('Extracting UTM subset from Geographic Lat/Lon input file %s'%imgfile)
    starttime = dtime_now()

    wgs84_epsg = 4326
    wgs84_sr = osr.SpatialReference()
    wgs84_sr.ImportFromEPSG(wgs84_epsg)
    wgs84_wkt = wgs84_sr.ExportToWkt()
    
    g = gdal.Open(imgf, gdal.GA_ReadOnly)

    # due to GDAL bug, g doesn't contain any metadata!
    metadata = read_envi_header(imgf+'.hdr')
    nodata_val = float64(metadata['data ignore value'])
    
    nl_in,ns_in = g.RasterYSize,g.RasterXSize
    geo_t = g.GetGeoTransform()
    geo_p = g.GetProjectionRef() 
    l_in, t_in, rot_in = geo_t[0],geo_t[3],geo_t[2]
    xps_deg, yps_deg = geo_t[1],abs(geo_t[5])    

    if len(subset_extent) == 0:
        # reproject full image
        r_in,b_in = l_in+xps_deg*ns_in, t_in-yps_deg*nl_in
        ullon,ullat,lrlon,lrlat = l_in,t_in,r_in,b_in
    elif len(subset_extent) == 4:
        ullon,ullat,lrlon,lrlat = subset_extent
    else:
        warn('subset_extent should be in the form [ullon,ullat,lrlon,lrlat]')
        return EMPTY,EMPTY

    ul_hemi = 'North' if ullat >= 0 else 'South'

    lx, ty, ul_zone, ul_alpha = lonlat2utm(ullon, ullat)
    rx, by, lr_zone, lr_alpha = lonlat2utm(lrlon, lrlat)
    if lr_zone != ul_zone:
        # need to recompute (rx, by) wrt UL zone
        warn('UL UTM zone (%d) != LR UTM zone (%d), assuming zone of UL pixel.'%(ul_zone,lr_zone))
        rx, by, lr_zone, lr_alpha = lonlat2utm(lrlon, lrlat, zone=ul_zone)

    if utm_ps is None:
        # get x,y pixel sizes in meters and use the largest for resampling
        xps_m = gc_distance(l_in,t_in,l_in+xps_deg,t_in        )
        yps_m = gc_distance(l_in,t_in,l_in        ,t_in-yps_deg)
        utm_ps = max(xps_m,yps_m)

    # assume a single zone / spatial reference
    utm_zone = ul_zone
    utm_epsg  = int((32600 if ul_hemi=='North' else 32700)+utm_zone)

    # subset lat/lon data
    ul_samp,ul_line = map2sl(ullon, ullat, l_in, t_in, xps_deg)
    lr_samp,lr_line = map2sl(lrlon, lrlat, l_in, t_in, xps_deg)

    # map truncated (sample,line) coords back into DEM
    ullon,ullat = sl2map(int(ul_samp),int(ul_line),l_in,t_in,xps_deg)
    lrlon,lrlat = sl2map(int(ceil(lr_samp)),int(ceil(lr_line)),l_in,t_in,xps_deg)

    ul_samp,ul_line = map2sl(ullon, ullat, l_in, t_in, xps_deg)
    lr_samp,lr_line = map2sl(lrlon, lrlat, l_in, t_in, xps_deg)
    
    nl_deg = int((round(lr_line)-round(ul_line)))+1
    ns_deg = int((round(lr_samp)-round(ul_samp)))+1

    # use in-memory driver for the quickness
    img_drv = gdal.GetDriverByName('MEM')
    gsub = img_drv.Create('', ns_deg, nl_deg, 1, gdal.GDT_Float32)

    gsub.SetGeoTransform((ullon, geo_t[1], geo_t[2], ullat, geo_t[4], geo_t[5]))
    gsub.SetProjection(wgs84_wkt)
    
    # subset the original image
    res = gdal.ReprojectImage(g, gsub, wgs84_wkt, wgs84_wkt,
                              gdal.GRA_NearestNeighbour)

        
    dest,(ulx,uly),(lrx,lry),dest_geo = gdal_reproject(gsub,utm_ps,utm_epsg)

    utm_sr   = osr.SpatialReference()
    utm_sr.ImportFromEPSG(utm_epsg)
    utm_wkt = utm_sr.ExportToWkt()
    
    if verbose:
        print('-> Lat/Lon [ullon,ullat,lrlon,lrlat] extent=',
              array([ullon,ullat,lrlon,lrlat]))

    rot = 0
    if geo_t[2] != 0 or geo_t[4] != 0:
        warn('image %s is not in a north-up coordinate system'%imgf)
        if geo_t[2] == geo_t[4]:
            rot = geo_t[2]
        else:
            warn('ill-defined rotation')
            return EMPTY,EMPTY


    dest_img = float64(dest.ReadAsArray().squeeze())
    nodata_mask = dest_img==float64(nodata_val)

    nl,ns = dest_img.shape[0],dest_img.shape[1]
    lx,ty = ulx,uly
    rx,by = lrx,lry
    
    if nodata_mask.any():
        n_nodata = nodata_mask.sum()
        if interp_nodata:
            #  TODO (BDB, 08/19/16): conncomp would probably make more sense 
            warn('interpolating %d NODATA (=%d) pixels'%(n_nodata,nodata_val))
            ccomp = bwlabel(int8(nodata_mask),background=0)
            ucomp = unique(ccomp[ccomp!=0])
            print('extrema(dest_img) before interpolation: "%s"'%str(extrema(dest_img)))                
            for l in ucomp:
                maskl = ccomp==l
                dest_img[maskl] = interp_xymask(dest_img,maskl,lx,ty,utm_ps)
            print('extrema(dest_img) after interpolation: "%s"'%str(extrema(dest_img)))
        else:
            # zero out nodata values
            warn('zeroing %d NODATA (=%d) pixels'%(n_nodata,nodata_val))
            dest_img[nodata_mask] = 0

    dest_extrema = extrema(dest_img)
    dest_size    = nl,ns
    dest_extent  = rx,by,lx,ty 
    dest_size    = nl,ns
    dest_meta    = dict(extent=dest_extent,extrema=dest_extrema,dims=dest_size,
                        zone=ul_zone,alpha=ul_alpha,hemi=ul_hemi,rot=rot,
                        ps=utm_ps,geotransform=dest_geo,projection=utm_wkt)    

    if verbose:
        print('->     extrema=[%f,%f]'%(dest_extrema[0],dest_extrema[1]))
        print('-> Output extent=', array(dest_extent))

    print('Extraction CPUtime (MM:SS.ms):      %s'%time_elapsed(starttime))
    return dest_img, dest_meta

def subset_utm(imgbase,subset_extent,interp_nodata=True,verbose=0):
    '''
    extract a bounding box subset_extent (format=[ullon,ullat,lrlon,lrlat]) of imgf    
    - assumes imgf in UTM x/y (does *not* reproject from wgs84 to UTM)
    - assumes subset_extent contained entirely within imgf
    - does not handle images that overlap two utm zones (assumes zone of ullon)!
    '''
    dest_img = EMPTY
    dest_meta = {}
    if len(subset_extent)!=4:
        warn('subset_utm requires [ullon,ullat,lrlon,lrlat] bounding extent')
        return dest_img,dest_meta

    ullon,ullat,lrlon,lrlat = subset_extent
    
    ul_hemi = 'North' if ullat >= 0 else 'South'
    lr_hemi = 'North' if lrlat >= 0 else 'South'
    if ul_hemi != lr_hemi: # bounding box overlaps both N/S hemispheres
        warn('bounding box overlaps Northern and Southern hemispheres')
        # FIXME (BDB, 03/11/16): test this!
        
    # convert lon/lat bbox into utm, get sample indices, and subset the image
    lx,ty,ul_zone,ul_alpha = lonlat2utm(ullon, ullat)
    rx,ly,lr_zone,lr_alpha = lonlat2utm(lrlon, lrlat)
    if lr_zone != ul_zone:
        warn('UL and LR in different UTM zones, assuming zone of UL pixel.')
        rx,ly,lr_zone,lr_alpha = lonlat2utm(lrlon, lrlat, zone=ul_zone)

    imgf     = imgbase if pathexists(imgbase) else imgbase+str(ul_zone)
    img      = envi_open(imgf+'.hdr',image=imgf)
    img_hdr  = img.metadata
    map_info = img_hdr['map info']
    imgdir,imgfile = pathsplit(imgf)

    nl_in, ns_in = int(img_hdr['lines']),int(img_hdr['samples'])
    nodata_val = img_hdr.get('data ignore value',-9999)
    rot = 0.0 # assume rotation = 0

    if map_info[0] != 'UTM':
        warn('invalid projection "%s", UTM required'%(pathsplit(imgf)[1],
                                                      map_info[0]))
        return dest_img,dest_meta

    # get UTM image bounding box from map info
    l_in, t_in = double(map_info[3]),double(map_info[4])    
    xps_in, yps_in = double(map_info[5]),double(map_info[6])
    r_in,b_in = sl2map(ns_in,nl_in,l_in,t_in,xps_in)
    
    if abs(xps_in) != abs(yps_in):
        warn('subset_utm cannot process DEMs with different x and y pixel sizes')
        return dest_img,dest_meta

    # x/y pixel size equal, pick one
    utm_ps = xps_in

    print('Extracting UTM subset from UTM input file %s'%imgfile)
    starttime = dtime_now()
    if verbose:
        print('-> Lat/Lon [ullon,ullat,lrlon,lrlat] extent=', array(subset_extent))
        print('-> UTM ps=%f, [minx,maxx,miny,maxy] extent='%utm_ps,array([lx,rx,ly,ty]))
        print('->     zone %d (alpha=%s, hemisphere=%s)'%(ul_zone, ul_alpha, ul_hemi))
    
    # upper left sample,line
    min_samp, min_line = map2sl(lx, ty, l_in, t_in, utm_ps)  
    min_samp, min_line = int(min_samp),int(min_line)
    
    # lower right sample,line
    max_samp, max_line = map2sl(rx, ly, l_in, t_in, utm_ps)
    max_samp, max_line = int(ceil(max_samp)),int(ceil(max_line))
    
    if max_samp<min_samp:
        warn('subset_utm max_samp should be >= min_samp! Bad extent?')
        return dest_img,dest_meta
    elif max_line<min_line:
        warn('subset_utm max_line should be >= min_line! Bad extent?')
        return dest_img,dest_meta

    if min_samp < 0:
        warn('subset_utm min_samp < 0, clipping min_samp=0.')
        min_samp = 0
    elif max_samp > ns_in-1:
        warn('subset_utm max_samp > ns (%d), clipping max_samp=ns.'%ns_in)
        max_samp = ns_in-1

    if min_line < 0:
        warn('subset_utm min_line < 0, clipping min_line=0.')
        min_line = 0
    elif max_line > nl_in-1:
        warn('subset_utm max_line > nl (%d), clipping max_line=nl.'%nl_in)
        max_line = nl_in-1          
    
    lx_new,ty_new = sl2map(min_samp,min_line,l_in,t_in,utm_ps)
    rx_new,ly_new = sl2map(max_samp,max_line,l_in,t_in,utm_ps)

    nl_new = max_line-min_line+1
    ns_new = max_samp-min_samp+1

    #print('Original bbox (l, r, b, t): %8.6f, %8.6f, %8.6f, %8.6f'%(l_in,r_in,b_in,t_in))
    #print('Original dims (lines, samples): %d, %d'%(nl_in, ns_in))
    #print('Subset bbox (l, r, b, t): %8.6f, %8.6f, %8.6f, %8.6f'%(lx,rx,ly,ty))
    #print('Subset dims (lines, samples): %d, %d'%(nl_new, ns_new))

    dest_img = img.read_subregion([min_line,max_line],[min_samp,max_samp],
                                  use_memmap=False)
    print('->\tsubset ns=%d, nl=%d'%(ns_new,nl_new))
    if dest_img is None:
        warn('unable to read', imgf)
        return EMPTY,dest_meta
        
    dest_img = asarray(dest_img.squeeze(),dtype=double)
    
    nodata_mask = dest_img==double(nodata_val)
    n_nodata = nodata_mask.sum()
    if n_nodata != 0:
        if interp_nodata:
            warn('interpolating %d NODATA (=%d) pixels'%(n_nodata,nodata_val))
            ccomp = bwlabel(int8(nodata_mask),background=0)
            ucomp = unique(ccomp[ccomp!=0])
            print('extrema(dest_img) before interpolation: "%s"'%str(extrema(dest_img)))                
            for l in ucomp:
                maskl = ccomp==l
                dest_img[maskl] = interp_xymask(dest_img,maskl,lx_new,ty_new,utm_ps)
            print('extrema(dest_img) after interpolation: "%s"'%str(extrema(dest_img)))
        else:
            # zero out nodata values
            warn('zeroing %d NODATA (=%d) pixels'%(n_nodata,nodata_val))
            dest_img[nodata_mask] = 0
    
    dest_extrema = extrema(dest_img)      
    dest_extent  = rx_new,ly_new,lx_new,ty_new
    dest_size    = nl_new,ns_new
    if verbose:
        print('-> Output extent=', array(dest_extent))
    
    dest_meta    = dict(extent=dest_extent,extrema=dest_extrema,dims=dest_size,
                        zone=ul_zone,alpha=ul_alpha,hemi=ul_hemi,rot=rot,ps=utm_ps)
    
    print('Extraction CPUtime (MM:SS.ms):      %s'%time_elapsed(starttime))
    return dest_img, dest_meta

#@profile
def geolocate(lines, samples, clock, nav, dem, alt_delta, rs_ps=4.0,
              return_lonlat=True, interp_geoid=True, check_bounds=False,
              cache_max_elev=False, offset_latlon=[], frame_meta=[],
              verbose=0):
    """
    geolocate(lines, samples, clock, nav, dem,  rs_ps=4.0,
              interp_geoid=True, return_lonlat=True,
              check_bounds=False, offset_latlon=[], verbose=0) 

    Given a list of n_ls ([line],[sample]) pixel locations, along with nav and DEM info
    return array ground_loc of the utm (x,y,z) coordinates of the pixel locations
    Bad coordinates indicated with ground_loc[:,2] < 0

    Arguments:
    - lines:          [n_ls x 1] line number of each coordinate
    - samples:        [n_ls x 1] sample number of each coordinate
    - clock:          [n_ls x 1] frame clock values for each coordinate
    - nav:            navigation class for parsing pps/gps information
    - dem:            DEM with georeferenced UTM elevation map
    
    Keyword Arguments:
    - rs_ps:          resampled pixel size in meters (default=4.0)
    - return_lonlat:  return wgs84 lon/lat instead of utm x/y (default=True)
    - interp_geoid:   interpolate when computing geoid correction (default=True)
    - offset_latlon:  [2 x 1] array of offsets to apply to lat/lon coordinates
    - verbose:        verbosity level (default=0)
    - frame_meta:     array populated with values
                      [frame_clock, utm_x, utm_y, altitude, pitch, roll, heading]
                      NOTE: returned in place if frame_meta kwarg present
    Returns:
    - ground_loc:     [n_ls x {(3|5)}] array of ground locations of each (line,sample) pair
                      Coordinate format: (x,y,z) if return_lonlat is False
                                         (x,y,z,lon,lat) otherwise

    """
    
    if len(nav.geoid) == 0 or len(nav.platform.camera_model) == 0:
        warn("world / camera models not initialized")
        return EMPTY

    if len(nav.pps_table) == 0 or len(nav.gps_table) == 0:
        warn("empty gps/pps tables")
        return EMPTY

    if len(dem.data_utm) == 0:
        warn("DEM not initialized")
        return EMPTY
    
    n_ls = len(lines)
    if n_ls == 0:
        warn('no coordinates to geolocate')
        return EMPTY    

    if len(samples) != n_ls:
        warn("len(samples) != len(lines)")
        return EMPTY
   
    if len(clock) != n_ls:
        warn("len(clock) != len(lines)")
    
    dem_extent = dem.meta['extent']
    dem_r = dem.meta['rot']
    dem_ps = dem.meta['ps']
    dem_size, dem_extrema = dem.meta['dims'],dem.meta['extrema']
    dem_zone,dem_alpha,dem_hemi = dem.meta['zone'],dem.meta['alpha'],dem.meta['hemi']
    dem_utm         = dem.data_utm
    dem_nl,dem_ns   = dem_size
    dem_min,dem_max = dem.extrema
    dem_xns,dem_yns,dem_x0,dem_y0 = dem_extent

    dem_bbox = [0,0,dem_ns,dem_nl]

    scalef_ps = rs_ps/dem_ps
    if scalef_ps != 1.0:
        # if our pixel size rs_ps isn't the same as the dem, bilerp
        if verbose:
            warn('Pixel size scale factor < 1.0 (%f with image ps=%.3f, dem ps=%.3f), resampling'%(scalef_ps,rs_ps,dem_ps))
        interpf = bilerp
        eff_ps  = rs_ps # effective pixel size of image wrt DEM
        dem_ns_ps = dem_ns/scalef_ps
        dem_nl_ps = dem_nl/scalef_ps
        dem_bbox_ps = [0,0,dem_ns_ps,dem_nl_ps]
    else:
        # otherwise just use the exact values in the dem
        interpf   = lambda img,xp,yp: img[int(yp),int(xp)].ravel()
        eff_ps    = dem_ps
        dem_ns_ps = dem_ns
        dem_nl_ps = dem_nl
        dem_bbox_ps = dem_bbox

    def outside_dem(xl,xh,yl,yh):
        return min(xl,xh)<dem_bbox_ps[0] or max(xl,xh)>=dem_bbox_ps[2] or \
            min(yl,yh)<dem_bbox_ps[1] or max(yl,yh)>=dem_bbox_ps[3]

    # each frame is associated with a unique clock measurement
    uframes,uidx = unique(lines,return_index=True)
    framec       = len(uframes)

    # output dimensions = [utmx,utmy,z] or [utmx,utmy,z,lon,lat] 
    out_dim      = 5 if return_lonlat else 3 
    ground_loc   = PIX_ERROR_UNDEF*ones([n_ls,out_dim]) 
    no_loc       = 0

    # nav variable/function references
    cam_model          = nav.platform.camera_model
    pb_len             = nav.platform.pb_len
    pb_off             = nav.platform.pb_off
    geoid              = nav.geoid
    geoid_dps          = nav.geoid_dps

    max_at_dist        = sqrt(2)*rs_ps
    
    ls_offset          = 0 # offset into ground_loc for the first sample in each frame
    frame_prev         = NOTFOUND
    
    for frame_idx in uframes:
        frame_clock = double(clock[frame_idx])
        #print('clock[frame_idx]: "%s"'%str((clock[frame_idx])))
        frame_mask  = lines==frame_idx
        samp_idx    = samples[frame_mask] # zero indexed offsets into frame
        ls_nsamp    = len(samp_idx) # number of samples at this frame index
        bad_mask    = samp_idx>(pb_len-1)
        if bad_mask.any():
            warn('no frame offset for %d samples in line %d'%(bad_mask.sum(),
                                                              frame_idx))
                                                                     
            badloc_idx  = ls_offset+where(bad_mask)[0]
            ground_loc[badloc_idx,:]   = PIX_ERROR_OUTSIDE_PB
            # note: do not want to stop processing here,
            # since we're just skipping some of the samples

        # get aircraft coordinates, convert to utm
        clock_loc = nav.clock2location(frame_clock)
        if clock_loc[0] == PIX_ERROR_NO_LOC:
            no_loc += 1
            ground_loc[ls_offset:ls_offset+ls_nsamp,:] = PIX_ERROR_NO_LOC
            ls_offset += ls_nsamp
            continue                

        # get aircraft position, translate to UTM (wrt DEM zone)
        lat,lon,altitude = clock_loc[:3]

        if len(offset_latlon) != 0:            
            lat,lon = lat+offset_latlon[0],lon+offset_latlon[1]
        
        utm_x,utm_y,utm_zone,utm_alpha = lonlat2utm(lon,lat,zone=dem_zone)

        # get orientation, apply gridnorth correction 
        pitch,roll       = DEG2RAD*clock_loc[3:5]

        # zone central meridian = (abs(utm_zone)-31)*6+3            
        gridnorth        = sin(DEG2RAD*lat)*(((abs(utm_zone)-31.)*6.+3.)-lon)
        heading          = DEG2RAD*(clock_loc[5]+gridnorth)

        # map lon to [0,360] (*after* utm/gridnorth correction!)
        if lon < 0.0:         
            lon = 360.0+lon

        # apply geoid correction
        geoidtrace = compute_geoidtrace(geoid,lon,lat,geoid_dps,
                                        interp=interp_geoid)
        altitude  = altitude-geoidtrace

        # update maximum elevation if we're at lower altitude than max(dem)
        if dem_max > altitude:
            msg  = 'dem_max (%4.1f) > altitude, clipping to '%dem_max
            msg += 'altitude-alt_delta.'
            dem_max  = altitude-alt_delta
            warn(msg)
            if cache_max_elev:
                dem.maxv = dem_max

        if DEBUG:
            print('frame,clock: %d %d'%(frame_idx,int(frame_clock)))
            print('altitude,utm_x,utm_y: %f %f %f'%(altitude,utm_x,utm_y))

        # keep track of frame metadata for obs files
        frame_info = [frame_clock,utm_x,utm_y,altitude,pitch,roll,heading]
        # only keep track of metadata for frames we keep
        frame_meta.append(frame_info)
        
        # get ray directions for selected samples
        if DEBUG:
            print("==============")
            print("samp_idx {}".format(samp_idx))
            print("pitch {},roll {},heading {}".format(pitch,roll,heading))

        frame_camera = cam_model[:,samp_idx] # 3 x 606
        frame_xyz = plane2world_camera_model(pitch,roll,heading,frame_camera)

        if DEBUG:
            print("frame_xyz {}".format(frame_xyz))
            print("==============")
            #input()

            
        # TODO (BDB, 07/21/15): VECTORIZE THIS LOOP 
        # project down to min/max DEM elevation 
        f_x,f_y,f_z = frame_xyz[0,:],frame_xyz[1,:],frame_xyz[2,:]
        
        d_min,d_max = (dem_min-altitude)/f_z, (dem_max-altitude)/f_z

        # get x,y utm coords of min/max intersections
        xo_lo_mapa,yo_lo_mapa = utm_x+d_min*f_x, utm_y+d_min*f_y
        xo_hi_mapa,yo_hi_mapa = utm_x+d_max*f_x, utm_y+d_max*f_y

        # get sample/line coords wrt 
        ixo_loa,iyo_loa = dem.map2sl(xo_lo_mapa,yo_lo_mapa,eff_ps=eff_ps)
        ixo_hia,iyo_hia = dem.map2sl(xo_hi_mapa,yo_hi_mapa,eff_ps=eff_ps)

        # trace ray through intersected pixels, get lo/hi intersections
        z_minmax = zeros([2,frame_xyz.shape[1]])
        for j,sj in enumerate(samp_idx):
            ls_idx = ls_offset+j # offset into ground_loc array for this sample
            if ground_loc[ls_idx,2] < PIX_ERROR_UNDEF:
                # only skip if pixel flagged with error other than PIX_ERROR_UNDEF
                continue
            ixo_hi,ixo_lo = ixo_hia[j],ixo_loa[j]
            iyo_hi,iyo_lo = iyo_hia[j],iyo_loa[j]
            if check_bounds and outside_dem(ixo_lo,ixo_hi,iyo_lo,iyo_hi):
                warn('pixels traced outside DEM extent')
                if DEBUG:
                    f_xj,f_yj,f_zj = f_x[j],f_y[j],f_z[j]
                    d_minj,d_maxj = d_min[j],d_max[j]
                    yo_lo_map,xo_lo_map = yo_lo_mapa[j],xo_lo_mapa[j]
                    yo_hi_map,xo_hi_map = yo_hi_mapa[j],xo_hi_mapa[j]                    
                    lono_lo,lato_lo = dem.utm2lonlat(yo_lo_map,xo_lo_map)
                    lono_hi,lato_hi = dem.utm2lonlat(yo_hi_map,xo_hi_map)
                    print('frame=%d, sample=%d (ls_idx=%d): '%(frame_idx,sj,ls_idx))
                    print('index,clock:',array([frame_idx,int(frame_clock)]))
                    print('altitude,utm_x,utm_y:',array([altitude,utm_x,utm_y]))                   
                    print('pitch,roll:',array([pitch,roll]))
                    print('gridnorth,heading:',array([gridnorth,heading]))
                    print('frame xyz: '+str(array([f_xj,f_yj,f_zj])))
                    print('dem_bbox:'+str(dem_bbox_ps))
                    print('[dem_min, dem_max]:'+str([dem_min,dem_max]))
                    print('[d_min, d_max]:'+str(array([d_minj,d_maxj])))                    
                    print('[ixo_lo, iyo_lo]:'+str(scalef_ps*array([ixo_lo,iyo_lo])))
                    print('[ixo_hi, iyo_hi]:'+str(scalef_ps*array([ixo_hi,iyo_hi])))
                    print('[xo_lo_map, yo_lo_map]:'+str(array([xo_lo_map,yo_lo_map])))
                    print('[xo_hi_map, yo_hi_map]:'+str(array([xo_hi_map,yo_hi_map])))
                    print('[lono_lo, lato_lo]:'+str([lono_lo,lato_lo]))
                    print('[lono_hi, lato_hi]:'+str([lono_hi,lato_hi]))                    
#                    raw_input()

                ground_loc[ls_idx,:] = PIX_ERROR_DEM_EXTENT
                continue
            
            xys_tr = find_pixel_trace(ixo_hi,iyo_hi,ixo_lo,iyo_lo)
            z_minmax[:,j] = extrema(interpf(dem_utm,
                                            xys_tr[:,0]*scalef_ps,
                                            xys_tr[:,1]*scalef_ps))

                
        d_lo_map,d_hi_map   = (z_minmax-altitude)/f_z    
        x_lo_mapa,y_lo_mapa = utm_x+d_lo_map*f_x, utm_y+d_lo_map*f_y
        x_hi_mapa,y_hi_mapa = utm_x+d_hi_map*f_x, utm_y+d_hi_map*f_y

        ix_loa,iy_loa = dem.map2sl(x_lo_mapa,y_lo_mapa,eff_ps=eff_ps)
        ix_hia,iy_hia = dem.map2sl(x_hi_mapa,y_hi_mapa,eff_ps=eff_ps)

        for j,sj in enumerate(samp_idx):
            ls_idx = ls_offset+j
            if ground_loc[ls_idx,2] < PIX_ERROR_UNDEF:
                # only skip if pixel flagged with error other than PIX_ERROR_UNDEF                
                continue
            
            f_xj,f_yj,f_zj = f_x[j],f_y[j],f_z[j]
            d_lo_mapj,d_hi_mapj = d_lo_map[j],d_hi_map[j]

            ix_hi,iy_hi,ix_lo,iy_lo = ix_hia[j],iy_hia[j],ix_loa[j],iy_loa[j]

            if check_bounds and outside_dem(ix_lo,ix_hi,iy_lo,iy_hi):
                warn('pixels traced outside DEM extent')
                if DEBUG:
                    y_lo_map,x_lo_map = y_lo_mapa[j],x_lo_mapa[j]
                    y_hi_map,x_hi_map = y_hi_mapa[j],x_hi_mapa[j]
                    lon_lo,lat_lo = dem.utm2lonlat(y_lo_map,x_lo_map)
                    lon_hi,lat_hi = dem.utm2lonlat(y_hi_map,x_hi_map)
                    print('%d (frame=%d, sample=%d): '%(ls_idx,frame_idx,sj))
                    print('index,clock: %d %d'%(frame_idx,int(frame_clock)))
                    print('altitude,utm_x,utm_y:',array([altitude,utm_x,utm_y]))
                    print('pitch,roll:',array([pitch,roll]))
                    print('gridnorth,heading:',array([gridnorth,heading]))
                    print('frame xyz: '+str(array([f_xj,f_yj,f_zj])))
                    print('dem_bbox:'+str(dem_bbox_ps))
                    print('[dem_min, dem_max]:'+str([dem_min,dem_max]))                    
                    print('[d_lo_mapj, d_hi_mapj]:'+str(array([d_lo_mapj,d_hi_mapj])))
                    print('[ix_lo, iy_lo]:'+str(scalef_ps*array([ix_lo,iy_lo])))
                    print('[ix_hi, iy_hi]:'+str(scalef_ps*array([ix_hi,iy_hi])))
                    print('[x_lo_map, y_lo_map]:'+str(array([x_lo_map,y_lo_map])))
                    print('[x_hi_map, y_hi_map]:'+str(array([x_hi_map,y_hi_map]))) 
                    print('[lon_lo, lat_lo]:'+str([lon_lo,lat_lo]))
                    print('[lon_hi, lat_hi]:'+str([lon_hi,lat_hi]))
                    print('[z_min, z_max]:%s'%(str(z_minmax[:,j])))
                    raw_input()
                ground_loc[ls_idx,:] = PIX_ERROR_DEM_EXTENT
                continue                   

            # determine intersection from extrema
            if int(ix_lo*scalef_ps) != int(ix_hi*scalef_ps) or int(iy_lo*scalef_ps) != int(iy_hi*scalef_ps):
            #if int(ix_lo) != int(ix_hi) or int(iy_lo) != int(iy_hi):
                xys = find_pixel_trace(ix_hi,iy_hi,ix_lo,iy_lo)
                z_trace = interpf(dem_utm,xys[:,0]*scalef_ps,xys[:,1]*scalef_ps)
                if len(xys) == 1:
                    z_int = z_trace[0]
                else:
                    x_hi_d,y_hi_d = utm_x-x_hi_mapa[j],utm_y-y_hi_mapa[j]
                    s_trace = sqrt(x_hi_d*x_hi_d + y_hi_d*y_hi_d)+(xys[:,2]*eff_ps)
                    z_diff  = z_trace-(altitude+s_trace/(sqrt(1.0-(f_zj*f_zj))/f_zj))

                    # find first point that sticks above ray
                    # which then gives us the intercept segment
                    #pos_diff = z_diff>=-(dem_ps*scalef_ps)
                    pos_diff = z_diff>=0.0
                    n_int = count_nonzero(pos_diff)                        
                    if n_int != 0:
                        idx_best = min(where(pos_diff)[0])
                        if idx_best==0 or n_int==1:
                            # if first point is above zero it must be the dem subset high point
                            z_int = z_trace[0]
                        elif n_int > 1:
                            # otherwise find the segment id and interpolate z_int
                            zd_best,zd_next = z_diff[[idx_best,idx_best-1]]
                            zt_best,zt_next = z_trace[[idx_best,idx_best-1]]
                            zd_frac = 1.0-zd_best/abs(zd_best-zd_next)
                            z_int  = zt_next+zd_frac*(zt_best-zt_next)
                        else:
                            print('no positive signed z-intersection for traced pixel %d (frame=%d, sample=%d), skipping'%(ls_idx,frame_idx,sj),1)
                            ground_loc[ls_idx,:] = PIX_ERROR_DEM_NOZINT
                            continue
                    else:
                        # hit the last pixel in the trace
                        z_int = z_trace[-1]

            else:
                z_int = interpf(dem_utm,ix_lo*scalef_ps,iy_lo*scalef_ps)

            
            # found intersection, compute (x,y) ground location
            d_int = (z_int-altitude)/f_zj
            ground_x, ground_y = utm_x+d_int*f_xj, utm_y+d_int*f_yj
            
            # if ls_idx != 0 and frame_idx==frame_prev and ground_loc[ls_idx-1,2]>PIX_ERROR_UNDEF:
            #     last_x,last_y = ground_loc[ls_idx-1,:2]
            #     at_dist = sqrt((ground_x-prev_x)**2 - (ground_y-prev_y)**2)
            #     if eucd > max_at_dist:
            #         warn('frame %d sample %d too far (%f vs. %f) from neighbors'%(frame_idx,sj,
            #                                                                      at_dist,
            ground_loc[ls_idx,:3] = [ground_x, ground_y, z_int]
            if return_lonlat:
                ground_lonlat = dem.utm2lonlat(ground_y,ground_x)
                ground_loc[ls_idx,3:] = ground_lonlat

            if verbose>2:
                if not return_lonlat:
                    ground_lonlat = dem.utm2lonlat(ground_y,ground_x)
                print('d_int: %f ground_xyz: %f %f %f'%(d_int,ground_x,ground_y,z_int))
                print('ground_lon,ground_lat: %f %f'%(ground_lonlat[0],ground_lonlat[1]))


        # move offset past the number of (valid) samples in this frame
        ls_offset += ls_nsamp            
        frame_prev = frame_idx

    if no_loc != 0:
        warn('unable to determine ground locations for %d of %d frames'%(no_loc,framec))

    #input()
    
    return ground_loc

def write_igm(igmf,igm_nl,igm_ns,bin_factor,zone,hemi):
    '''
    creates header for igm file 
    '''
    if not pathexists(igmf):
        warn('IGM file %s does not exist'%igmf)
        return FAILURE

    description  = 'ANG AIG VSWIR RT-Ortho IGM (easting, northing, elevation)\n'
    description += 'UTM zone %d %s'%(zone,hemi)

    igm_hdrf             = igmf+'.hdr'
    igm_hdr              = {'lines':igm_nl,'samples':igm_ns}
    igm_hdr['bands']     = 3 # bands = [mapX,mapY,alt]
    igm_hdr['data type'] = 5 # 2 = int16, 3 = int32, 12 = uint16, 5 = double

    igm_hdr['header offset']  = 0
    igm_hdr['byte order']     = 0
    igm_hdr['interleave']     = 'bip'
    igm_hdr['line averaging'] = bin_factor
    igm_hdr['description']    = description
    igm_hdr['band names']     = '{Easting (m), Northing (m), Elevation (m)}'

    write_envi_header(igm_hdrf,igm_hdr)

    return SUCCESS

def rotate_glt(bbox_xy,snap=5.0):
    '''
    computes rotation matrix minimizing *width* of GLT bounding box
    '''
    rot_xy = bbox_xy
    rot_mat = [[1.0,0.0],[0.0,1.0]]
    rot_deg = 0.0
    if snap==0:
        return rot_xy, rot_mat, rot_deg

    xr = extrema(bbox_xy[0,:])
    rot_min = abs(xr[1]-xr[0])
    for r in arange(-90,91,snap):
        ar    = DEG2RAD*r
        cosr  = cos(ar)
        sinr  = sin(ar)
        r_ar   = [[cosr,-sinr], [sinr,cosr]]
        r_xy   = dot(r_ar,bbox_xy)
        xr     = extrema(r_xy[0,:])
        r_diff = abs(xr[1]-xr[0])
        # NOTE (BDB, 09/01/15): code below finds min size (not min width) bbox
        # yr = extrema(rxy[1,:])
        # r_diff = np_min([xr[1]-xr[0],yr[1]-yr[0]]) 
        if r_diff < rot_min:
            rot_min = r_diff
            rot_xy,rot_mat,rot_deg  = r_xy,r_ar,r

    return rot_xy, rot_mat, rot_deg

def setdiff2d(a1,a2):
    '''
    >>> a1=array([[-0.004247, -1.671015],
    ...           [ 0.403241,  0.952984],
    ...           [-0.145735,  0.032906],
    ...           [ 1.256124,  0.072584],
    ...           [-0.385722, -1.261825]])
    >>> a2=array([[ 0.888407, -2.613242],
    ...           [ 0.403241,  0.952984],
    ...           [-0.145735,  0.032906]])
    >>> setdiff2d(a1,a2)
    array([[-0.385722, -1.261825],
           [-0.004247, -1.671015],
           [ 1.256124,  0.072584]])
    '''
    a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
    a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
    return setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1])   


def erode_boundary(img,ictr,jctr,bufrad):
    """
    erode_boundary(img,ictr,jctr,bufrad)
    
    Summary: two pass inplace erosion of provided contour with a square filter
    
    Arguments:
    - img: M X N x K array of int32 image data
    - ictr: boundary row indices
    - jctr: boundary column indices  
    Keyword Arguments:
    None
    
    Output:
    image with eroded boundary
    """
    
    R,C,_ = img.shape
    for idx in range(len(ictr)):
        i,j = ictr[idx],jctr[idx]
        imin,imax = max(0,i-bufrad),min(R,i+bufrad+1)
        jmin,jmax = max(0,j-bufrad),min(C,j+bufrad+1)
        if img[imin:imax,jmin:jmax,0].max() <= 0:
            # zero all negative values in this window
            img[imin:imax,jmin:jmax,:] = 0


try:
    _erode_boundary_sig='void(i4[:,:,:],i4[:],i4[:],i4)'
    _erode_boundary = nbjit(_erode_boundary_sig,nopython=True)(erode_boundary)
except:
    warn('Could not run numba JIT for function "erode_boundary"; GLT generation runtime will increase')
    _erode_boundary = erode_boundary

            
def fillnn(img,nn_sorted):
    '''
    inplace replacement of zero pixels with their nearest positive neighbor
    img:  M x N array of 2d pixels
    nn_sorted: NN x 2 array of (row,column) neighbor offsets
               (sorted by distance to origin, then ccw angle)
    '''
    R,C,_ = img.shape
    NN,_ = nn_sorted.shape

    for i in range(1,R-1):
        for j in range(1,C-1): 
            if img[i,j,0]==0:
                # found a pixel to fill, find nearest positive neighbor
                for n in range(NN):
                    ki,lj = i+nn_sorted[n,0],j+nn_sorted[n,1]
                    if ki>0 and ki<R-1 and lj>0 and lj<C-1 and img[ki,lj,0]>0: 
                        img[i,j,:] = -img[ki,lj,:]
                        break # first match = nearest, no more searching

try:
    _fillnn_sig = 'void(i4[:,:,:],i4[:,:])'
    _fillnn = nbjit(_fillnn_sig,nopython=True)(fillnn)
except Exception as e:
    warn('Could not run numba JIT for function "fillnn"; GLT generation runtime will increase')
    _fillnn = fillnn
    
def infillnn(img,winr,winb,order='ccw'):
    # construct window for nn search to populate missing pixels
    winr  = int32(winr)
    wind  = 2*winr+1
    winv  = arange(wind)-winr
    nn    = array(map(lambda w: w.ravel(),meshgrid(winv,winv)),dtype=int32).T
    dists = (nn*nn).sum(axis=1)
    wkeep = dists < winr**2
    nn    = nn[wkeep]
    angls = arctan2(nn[:,0],nn[:,1])*RAD2DEG
    dists = dists[wkeep]
    
    # sort neighbors starting from 3:00    
    #angls = -((where(angls<0,angls+360,angls)+270) % 360)

    # sort neighbors starting from 12:00
    angls = (where(angls<0,angls+360,angls)+270) % 360

    if order == 'cw':
        angls = -angls
    
    # get/apply sort indices, removing sidx[0] (offset=[0,0])
    sorta = array(zip(dists,angls*1000),dtype=[('d','i4'),('a','i4')])
    nn = nn[argsort(sorta,order=['d','a'])[1:]]

    # fill missing pixels in-place
    _fillnn(img,int32(nn))
    
    bufrad = max(0,winr-(2+winb))
    if bufrad>0: # values outside this range have no effect
        ictr,jctr = where(find_boundaries((img[:,:,0]<0),mode='thick'))
        _erode_boundary(img,int32(ictr),int32(jctr),int32(bufrad))
        
    # also remove any interpolated pixels adjacent to image borders
    edgerad = max(1,bufrad/2+0.5)
    nrows,ncols = img.shape[0],img.shape[1]        
    rowidx,colidx = arange(nrows),arange(ncols)
    ictr,jctr = rowidx,zeros(nrows)
    ictr,jctr = r_[ictr,rowidx],r_[jctr,ones(nrows)*(ncols-1)]
    ictr,jctr = r_[ictr,zeros(ncols)],r_[jctr,colidx]
    ictr,jctr = r_[ictr,ones(ncols)*(nrows-1)],r_[jctr,colidx]
    _erode_boundary(img,int32(ictr),int32(jctr),int32(edgerad))

def dt_distance(xyz):
    #dists = sqrt(((xyz[1:,1,:-1]-xyz[:-1,1,:-1])**2).reshape([-1,2]).sum(axis=1))
    dists = sqrt(((xyz[1:,:,:-1]-xyz[:-1,:,:-1])**2).reshape([-1,2]).sum(axis=1))
    return dists

def at_distance(xyz):
    dists = sqrt(((xyz[:,1:,:-1]-xyz[:,:-1,:-1])**2).reshape([-1,2]).sum(axis=1))
    #dists = sqrt(((xyz[:,0,:-1] -xyz[:,1,:-1])**2).reshape([-1,2]).sum(axis=1))
    return dists

    
def xyz2ps(xyz,ps_avg_fn,min_ps,max_ps,bin_factor_min=1,verbose=1):
    assert((xyz.shape[0]>1) and (xyz.shape[1]==3) and (xyz.shape[2]>=3))

    # xyz.shape = (nline,nsamp,3), nsamp~=3 for nadir samples
    dt_dist = dt_distance(xyz)
    at_dist = at_distance(xyz)

    dt_ps = ps_avg_fn(dt_dist)
    at_ps = ps_avg_fn(at_dist)
    
    _dt_ps = min(max_ps,max(min_ps,dt_ps))
    _at_ps = min(max_ps,max(min_ps,at_ps))

    if _dt_ps != dt_ps:
        warn('clipped dt_ps (%f) to (%f)'%(dt_ps,_dt_ps))
        dt_ps = _dt_ps

    if _at_ps != at_ps:
        warn('clipped at_ps (%f) to (%f)'%(at_ps,_at_ps))
        at_ps = _at_ps

    dt_min,dt_max = extrema(dt_dist)
    at_min,at_max = extrema(at_dist)
    ort_ps        = int(at_ps*10)/10.0    
    
    _ort_ps = min(max_ps,max(min_ps,ort_ps))
    if _ort_ps != ort_ps:
        warn('clipped ort_ps (%f) to (%f)'%(ort_ps,_ort_ps))
        ort_ps = min_ps

    bin_factor  = int(round(ort_ps/dt_ps))
    if bin_factor == 0:
        msg  = "estimated bin_factor<%d (at_ps=%f, dt_ps=%f)"%(bin_factor_min,
                                                               at_ps,dt_ps)
        msg += ', using default (%d)'%bin_factor_min
        warn(msg)
        bin_factor=bin_factor_min

    if verbose:
        print('dt_ps: %8.6f'%dt_ps,'dt_min: %8.6f'%dt_min,'dt_max: %8.6f'%dt_max)
        print('at_ps: %8.6f'%at_ps,'at_min: %8.6f'%at_min,'at_max: %8.6f'%at_max)
        print('ort_ps: %3.1f'%ort_ps,'ps_avg_fn:',ps_avg_fn.func_name)
        print('bin_factor: %d'%bin_factor)
        
    return dict(ort_ps=ort_ps, bin_factor=bin_factor,
                dt_ps=dt_ps, dt_min=dt_min, dt_max=dt_max,
                at_ps=at_ps, at_min=at_min, at_max=at_max)

def write_glt(gltf,igm_xyz,igm_nl,igm_sidx,igm_cidx,ulx,uly,ort_ps,zone,hemi,
              img_sl,img_ss,bin_factor,nn_rad,nn_buf,ps_avg_fn,min_ps,max_ps,
              GLT_ROT_SNAP,verbose=0):
    '''
    update_glt offsets s,l coords to fit within the (xs,ys), (xe,ye) bounding
    box, where the original coordinates are defined with respect to the DEM
    (ulx,uly) and pixel size ps.
    '''
    starttime  = dtime_now()
    
    igm_ns = len(igm_sidx)
    # todo: validate this reshape
    igm_xyz3 = igm_xyz.reshape([igm_nl,-1,3])
    eff_ps = ort_ps
    # recompute effective ps using nadir-pointing center samples
    ps_dict = xyz2ps(igm_xyz3[:,igm_cidx,:],ps_avg_fn,min_ps,max_ps,verbose=0)
    est_ps = ps_dict['ort_ps']   
    if est_ps > ort_ps: # ort_ps too small, use new estimate instead
        print('Increasing ort_ps (%3.2f) to %3.2f'%(ort_ps,est_ps))
        eff_ps = est_ps

    # generate s,l indices for igm, filter out bad pixels
    igm_keep = igm_xyz[:,2]>PIX_ERROR_UNDEF 
    igm_sl = c_[map(ravel,meshgrid(igm_sidx+1,arange(igm_nl)+1))].T
    igm_sl = igm_sl[igm_keep,:]
    
    # select (good) x/y coords at image bounding box to compute rotation
    igm_xyz3 = r_[igm_xyz3[0,:,:].squeeze(), igm_xyz3[:,-1,:].squeeze(),
                  igm_xyz3[:,0,:].squeeze(), igm_xyz3[-1,:,:].squeeze()]

    # exclude pixels flagged as errors
    igm_xyz3 = igm_xyz3[igm_xyz3[:,2]>PIX_ERROR_UNDEF,:2].T 

    # rotate glt about upper right coordinate
    rigm_xyz3,rotm,rota = rotate_glt(igm_xyz3,snap=GLT_ROT_SNAP)
    
    # rotate (good) pixels if necessary
    if rota != 0.0:
        rx, ry     = dot(rotm,igm_xyz[igm_keep,:2].T)
        rulx, ruly = dot(rotm,[ulx,uly])
        rotstr     = ', rotation=%10.7f'%(-rota) 
    else:
        rx, ry     = igm_xyz[igm_keep,:2].T
        rulx, ruly = ulx, uly
        rotstr     = ', rotation=0.0'

    print("=====================\n")
    print("rulx {}, ruly {}".format(rulx,ruly))

    # get s,l coords with respect to rotated (DEM) ulx/uly
    if verbose:
        print('GLT rotation angle:                ',rota)
    glt_sl = asarray(map2sl(rx,ry,rulx,ruly,eff_ps),dtype=int).T    
    # compute bounding box extent in pixel coords
    xs,xe = extrema(glt_sl[:,0])
    ys,ye = extrema(glt_sl[:,1])
    glt_nl,glt_ns = ye-ys+1,xe-xs+1

    # the GLT cannot contain more pixels than max(IGM.shape)**2
    glt_num_pix = glt_nl*glt_ns
    glt_max_pix = int(1.25*max(igm_nl,igm_ns))**2
    if glt_num_pix > glt_max_pix:
        # if it is, something has gone wrong
        #igm_ns = int(len(glt_sl[:,0])/igm_nl)
        msg = 'Error computing GLT dimensions: '
        msg += 'total GLT pixels ({glt_num_pix}) larger than max(IGM.shape)**2 ({glt_max_pix})'
        if verbose:
            msg += '\nxs: {xs}, ys: {ys}, xe: {xe}, ye: {ye}'
            msg += '\nigm_nl: {igm_nl}, igm_ns: {igm_ns}, '
            msg += 'glt_nl: {glt_nl}, glt_ns: {glt_ns}'
        warn(msg.format(**locals()))
        sys.exit(FAILURE)

    # undo rotation for UL coordinate to get glt_ulx,glt_uly
    rxs,rys               = dot([xs,ys],inv(rotm))
    print("xs {}, ys {}, rotm {}".format(xs, ys, rotm))
    glt_ulx,glt_uly       = sl2map(rxs,rys,ulx,uly,eff_ps) #####
    print("rxs {}, rys {}, ulx {}, uly {}, eff_ps {}\n".format(rxs,rys,ulx,uly,eff_ps))
    # different: xs, ys, rxs, rys
    # between original_model and updated_cam_model
    print("=====================\n")
    
    # initialize and populate GLT image
    glt_hdrf              = gltf+'.hdr'
    glt_description       = 'ANG AIG VSWIR RT-Ortho GLT (IGM sample, IGM line)'
    glt_map_info          = '{glt_ulx}, {glt_uly}, {eff_ps}, {eff_ps}, {zone}'.format(**locals())
    glt_hdr               = {'lines':glt_nl,'samples':glt_ns,'bands':2}
    glt_hdr['band names'] = "{GLT Sample Lookup, GLT Line Lookup}"
    glt_hdr['data type']  = 3 # 2 = int16, 3 = int32, 12 = uint16, 5 = double
    glt_hdr['map info']   = '{UTM, 1, 1, %s, %s, WGS-84, units=Meters%s}'%(glt_map_info,
                                                                           hemi,rotstr)
    glt_hdr['header offset']       = 0
    glt_hdr['byte order']          = 0
    glt_hdr['interleave']          = 'bip'
    glt_hdr['line averaging']      = bin_factor
    glt_hdr['raw starting line']   = img_sl+1 # convert 0-indexed to 1-indexed
    glt_hdr['raw starting sample'] = img_ss+1
    glt_hdr['description']         = glt_description
    # TODO (BDB, 08/20/15): what about nodata values? 

    if verbose:
        print('GLT coordinate offsets:             ({xs},{ys}), ({xe},{ye})'.format(**locals())    )
    glt_img = envi_create_image(glt_hdrf,glt_hdr,force=True,ext='')
    glt_mm  = glt_img.open_memmap(interleave='source',writable=True)

    # shift glt_sl to fit bbox extent, fill memmap with igm sample/line values
    glt_mm[:,:,:] = 0 
    glt_mm[glt_sl[:,1]-ys,glt_sl[:,0]-xs,:] = igm_sl

    glt_mm = infillnn(glt_mm,nn_rad,nn_buf)
    glt_mm = None # flush to disk
    print('CPUtime (MM:SS.ms):                 %s'%time_elapsed(starttime))
    return SUCCESS

def generate_obs_loc(igmf,dem,frame_meta,gps_table):
    '''
    obs layers:
    1  path length (sensor-to-ground in meters)
    2  to-sensor-azimuth (0 to 360 degrees cw from N)
    3  to-sensor-zenith (0 to 90 degrees from zenith)
    4  to-sun-azimuth
    5  to-sun-zenith
    6  phase (degrees between to-sensor and to-sun vectors in principal plane)
    7  slope (local surface slope as derived from DEM in degrees)
    8  aspect (local surface aspect 0 to 360 degrees based on DEM slope and aspect and to sun vector, -1 to 1)
    10 UTC time (decimal hours for mid-line pixels).
    11 Earth to sun distance (in AU)
    
    loc layers:
    1 WGS-84 longitude (decimal degrees)
    2 WGS-84 latitude (decimal degrees)
    3 Estimated ground elevation at each pixel center (meters)
    '''

    obsf,locf = igmf.replace('igm','obs'),igmf.replace('igm','loc')
    obs_hdrf,loc_hdrf = obsf+'.hdr', locf+'.hdr'

    starttime   = dtime_now()    
    igm_img = envi_open(igmf+'.hdr',image=igmf)
    igm_mm  = igm_img.open_memmap(interleave='source',writable=False)

    loc_hdr  = igm_img.metadata.copy()
    loc_hdr['description'] = 'ANG AIG VSWIR RT-Ortho LOC'
    loc_hdr['band names'] = '{Longitude (WGS-84), Latitude (WGS-84), Elevation (m)}'
    loc_img = envi_create_image(loc_hdrf,loc_hdr,force=True,ext='')
    loc_mm  = loc_img.open_memmap(interleave='source',writable=True)
    
    obs_hdr  = igm_img.metadata.copy()
    obs_hdr['band names'] = '{Path length (m), To-sensor azimuth (0 to 360 degrees cw from N), To-sensor zenith (0 to 90 degrees from zenith) , To-sun azimuth (0 to 360 degrees cw from N), To-sun zenith (0 to 90 degrees from zenith) , Solar phase, Slope, Aspect, Cosine(i), UTC Time, Earth-sun distance (AU)}'
    obs_hdr['description'] = 'ANG AIG VSWIR RT-Ortho OBS'
    obs_hdr['bands'] = 11
    obs_img = envi_create_image(obs_hdrf,obs_hdr,force=True,ext='')
    obs_mm  = obs_img.open_memmap(interleave='source',writable=True)

    # FIXME (BDB, 09/10/15): get this info from the gps data? 
    yy,mm,dd,gpsw,hh,nn,ss = file2date(igmf)
    
    sutc = cTime()
    sloc = cLocation()
    sunc = cSunCoordinates() 
    
    sutc.iYear    = yy #int, year
    sutc.iMonth   = mm #int, month
    sutc.iDay     = dd #int, day    
    sutc.dHours   = hh
    sutc.dMinutes = nn
    sutc.dSeconds = ss
    esd = sun_dist_au(yy, mm, dd)  

    # Earth/sun distance in astronomical units
    obs_mm[:,:,10] = esd

    igm_nl, igm_ns = igm_mm.shape[:2]

    # local aliases for dem
    dem_utm = dem.data_utm
    dem_ps  = dem.ps
    datum   = dem.datum
    ulx,uly = dem.ulx,dem.uly

    #dxdy   = ones(dem_utm.shape,dtype=float)*dem_ps # x/y deltas for gradient
    #dx,dy  = gradient(dem_utm,dxdy,dxdy)
    dx,dy  = gradient(dem_utm,dem_ps)
    slope  = 0.5*pi-arctan(hypot(dx,dy)) 
    aspect = arctan2(dx,dy)

    zone_alpha = str(dem.utm_zone)+dem.utm_alpha
    frame_meta = asarray(frame_meta)

    n_step = 10
    l_step = int(igm_nl/n_step)

    #gpstime = frame_meta[:,0]
    # compute interpolated UTC time from gps file
    #from ortho_nav import read_gps
    #gpsfile=igmf.replace("rdn_igm","gps")
    #gps_table,gps_velo=read_gps(gpsfile)
    gpstime=gps_table.T[0]

    print('gpstime: "%s"'%str((gpstime)))
    print('frame_meta: "%s"'%str((frame_meta)))
    
    from scipy.interpolate import interp1d 
    interp_gps = interp1d(arange(gpstime.size),gpstime)
    gpstime = interp_gps(linspace(0,gpstime.size-1,arange(igm_nl).size)) 
    # change each line's UTC time to mid line UTC decimal hour
    for i in range(igm_ns):
        obs_mm[:,i,9]=gps2hour(gpstime)

    gpsdiff = median(diff(gpstime)) # diff(gpstime) should be constant
    gpsoff = ones([l_step,igm_ns])*(gpsdiff*arange(igm_ns)/float(igm_ns))
    sol_azimuth = zeros([l_step,igm_ns])
    sol_zenith  = zeros([l_step,igm_ns])
    for lj,l_min in enumerate(range(0,igm_nl,l_step)):
        obs_per = int(100*lj/n_step)
        print('Processing line %i of %i (%3d%%)'%(l_min,igm_nl,obs_per))
        l_max  = min(l_min + l_step, igm_nl)
        l_step = min(l_step,l_max-l_min)
        l_range = arange(l_min,l_max)
        interp_size = [igm_ns,l_step]

        # sensor to ground geometry + deltas
        mapx,mapy,mapz = igm_mm[l_range,:,:].T
        airx,airy,airz = frame_meta[l_range,1:4].T
        dx,dy,dz = airx-mapx, airy-mapy, airz-mapz 
        ds = sqrt(dx*dx + dy*dy + dz*dz)
        dcz_sensor = dz/ds
                
        # generate location file
        lon,lat = dem.utm2lonlat(mapy,mapx)
        loc_mm[l_range,:,0] = lon.T
        loc_mm[l_range,:,1] = lat.T
        loc_mm[l_range,:,2] = mapz.T

        # path length (sensor to ground) in meters)
        obs_mm[l_range,:,0] = ds.T 

        # to-sensor azimuth (0 to 360 degrees cw from N)
        azimuth_sensor = arctan2(dx,dy)
        azimuth_sensor += (azimuth_sensor<0)*2*pi
        obs_mm[l_range,:,1] = azimuth_sensor.T * RAD2DEG

        # to-sensor-zenith (0 to 90 degrees from zenith)
        obs_mm[l_range,:,2] = arccos(dcz_sensor.T) * RAD2DEG

        # solar zenith and azimuth
        #ltime = frame_meta[l_range,0]
        for l in xrange(l_step):
            for s in xrange(igm_ns):                                
                sloc.dLatitude  = lat[s,l]
                sloc.dLongitude = lon[s,l]
                sunpos(sutc,sloc,sunc)
                sol_zenith[l,s]  = sunc.dZenithAngle
                sol_azimuth[l,s] = sunc.dAzimuth

        sol_zenith = sol_zenith[:l_step,:]
        sol_azimuth = sol_azimuth[:l_step,:]
        obs_mm[l_range,:,3] = sol_azimuth
        obs_mm[l_range,:,4] = sol_zenith

        # solar phase 
        sin_zenith = sin(sol_zenith*DEG2RAD)
        dcx_sun = sin(sol_azimuth*DEG2RAD) * sin_zenith
        dcy_sun = cos(sol_azimuth*DEG2RAD) * sin_zenith
        dcz_sun = cos(sol_zenith*DEG2RAD) 

        # get slope, aspect by mapping mapx,mapy into DEM coordinate frame
        dem_xf,dem_yf = map2sl(mapx.ravel(),mapy.ravel(),ulx,uly,dem_ps)
        slope_interp  = bilerp(slope,dem_xf,dem_yf).reshape(interp_size).T
        aspect_interp = bilerp(aspect,dem_xf,dem_yf).reshape(interp_size).T

        obs_mm[l_range,:,5] = arccos((dx.T*dcx_sun + dy.T*dcy_sun)/ds.T \
                                     + dcz_sensor.T*dcz_sun)*RAD2DEG 
        obs_mm[l_range,:,6] = slope_interp*RAD2DEG
        obs_mm[l_range,:,7] = aspect_interp*RAD2DEG
        
        # cosine of sun relative to surface normal
        obs_mm[l_range,:,8] = sin(slope_interp)*(sin(aspect_interp)*dcx_sun \
                                                 + cos(aspect_interp)*dcy_sun) \
                                                 + cos(slope_interp)*dcz_sun        

        # compute interpolated gpstime for each sample in frame # removed to fix UTC time in obs 
        #obs_mm[l_range,:,9] = (gpstime[l_range] + gpsoff[:l_step,:].T).T

    if obs_per < 100:
        print('Processing line %i of %i   \t100%%'%(igm_nl,igm_nl))

    # close memmaps to flush to disk
    obs_mm = None
    loc_mm = None
    igm_mm = None

    print('CPUtime (MM:SS.ms):                 %s'%time_elapsed(starttime))
    return obsf, locf

def generate_obs_ort(obsf,gltf):
    from apply_glt import apply_glt
    starttime   = dtime_now()    
    ort_obsf = obsf.replace('_obs','_obs_ort')
    apply_glt(obsf,gltf,ort_obsf)
    print('CPUtime (MM:SS.ms):                 %s'%time_elapsed(starttime))

def generate_landmask(loc_hdrf,z_eps=1e-4):
    starttime   = dtime_now()
    locf    = loc_hdrf.replace('.hdr','')
    loc_img = envi_open(loc_hdrf,image=locf)
    loc_mm  = loc_img.open_memmap(interleave='source',writable=False)

    land_hdrf = loc_hdrf.replace('_loc','_land')
    land_hdr  = loc_img.metadata.copy()
    land_hdr['description'] = 'ANG AIG VSWIR RT-Ortho LAND'
    land_hdr['band names']  = '{Land Mask}'
    land_hdr['bands']       = 1
    land_hdr['data type']   = 1
    land_img = envi_create_image(land_hdrf,land_hdr,force=True,ext='')
    land_mm  = land_img.open_memmap(interleave='source',writable=True)

    land_mm[:] = array(abs(loc_mm[:,:,0])<z_eps,dtype=uint8)
    land_mm = None
    print('CPUtime (MM:SS.ms):                 %s'%time_elapsed(starttime))
    return land_hdrf

def generate_ql(igm_hdrf,glt_hdrf,ql_imgf,bands=[0]):
    from apply_glt_dev import apply_glt as apply_glt_dev
    igm_imgf = igm_hdrf.replace('.hdr','')
    glt_imgf = glt_hdrf.replace('.hdr','')
    apply_glt_dev(igm_imgf,glt_imgf,ql_imgf,bands=bands,verbose=False)
    envi2jpeg(ql_imgf+'.hdr',ql_imgf+'.jpg',bands=bands)    

def generate_kml(glt_hdrf,ql_jpgf=None,templatef='template.kml',verbose=1):
    """
    """
    with open(templatef) as fid:
        kml_template = fid.read()
    
    glt_imgf = glt_hdrf.replace('.hdr','')
    glt_img  = envi_open(glt_hdrf,image=glt_imgf)
    glt_meta = glt_img.metadata
    glt_mm   = glt_img.open_memmap(interleave='source',writable=False)
    glt_map  = glt_meta['map info']    

    nl,ns = int(glt_meta['lines']),int(glt_meta['samples'])
    raw_sl = int(glt_meta['raw starting line'])
    raw_ss = int(glt_meta['raw starting sample'])    
    ps,utm_zone,utm_hemi = float(glt_map[6]),glt_map[7],glt_map[8]

    ulcol,ulrow = 0,0
    lrcol,lrrow = ns-1,nl-1
    urcol,urrow = ns-1,0
    llcol,llrow = 0,nl-1   
    
    ulx,uly = map(float,glt_map[3:5])
    lrx,lry = sl2map(lrcol,lrrow,ulx,uly,ps)
    urx,ury = sl2map(urcol,urrow,ulx,uly,ps)    
    llx,lly = sl2map(llcol,llrow,ulx,uly,ps)

    rot = float(glt_map[-1].replace('rotation=',''))
    urx,ury = rotxy(urx,ury,rot,ulx,uly)
    llx,lly = rotxy(llx,lly,rot,ulx,uly)
    lrx,lry = rotxy(lrx,lry,rot,ulx,uly)

    print('bounding box (ulx,uly)=',(ulx,uly),'(lrx,lry)=',(lrx,lry))
    
    # assign an appropriate alpha value for given hemisphere
    # alpha >= 'N' -> Northern hemisphere, else Southern
    utm_alpha = 'N' if utm_hemi == 'North' else 'M'
    
    ullon,ullat = utm2lonlat(uly,ulx,utm_zone,utm_alpha)

    # get true alpha designation by mapping ul coord back to specified zone
    _ulx,_uly,_utm_zone,utm_alpha = lonlat2utm(ullon,ullat,zone=utm_zone)

    # make sure our remapped x,y is within 2px of original x,y
    assert(abs(ulx-_ulx)<2*ps and abs(uly-_uly)<2*ps)
    
    lrlon,lrlat = utm2lonlat(lry,lrx,utm_zone,utm_alpha)
    urlon,urlat = utm2lonlat(ury,urx,utm_zone,utm_alpha)
    lllon,lllat = utm2lonlat(lly,llx,utm_zone,utm_alpha)

    if verbose:
        print('llx,lly: "%s"'%str((llx,lly)),
              'lrx,lry: "%s"'%str((lrx,lry)))
        print('urx,ury: "%s"'%str((urx,ury)),
              'ulx,uly: "%s"'%str((ulx,uly)))

        print('lllon,lllat: "%s"'%str((lllon,lllat)),
              'lrlon,lrlat: "%s"'%str((lrlon,lrlat)))
        print('urlon,urlat: "%s"'%str((urlon,urlat)),
              'ullon,ullat: "%s"'%str((ullon,ullat)))

    outkmlf = glt_hdrf.replace('glt.hdr','overlay.kml')
    llz,urz,ulz,lrz=0,0,0,0
    filebase = glt_hdrf.replace('_glt.hdr','')
    filedate,fileyear = '[date]','[year]'
    solarelev,solarazi = '[solarelev]','[solarazi]'
    ortps = '[ortps]'
    imagefile = ql_jpgf or filebase+'_RGB-W200.jpg'
    outkml = kml_template.format(**locals())
    with open(outkmlf,'w') as fid:
        fid.write(outkml)
        
def plot_igm(igm_xyz):
    import pylab as pl
    pl.ioff()    
    fig = pl.figure(1)
    pl.subplot(3,1,1)
    pl.imshow(igm_xyz[:,:,0],interpolation='none')
    pl.ylabel('UTM X')
    pl.subplot(3,1,2)
    pl.imshow(igm_xyz[:,:,1],interpolation='none')
    pl.ylabel('UTM Y')
    pl.subplot(3,1,3)
    pl.imshow(igm_xyz[:,:,2],interpolation='none')
    pl.ylabel('Elevation')

def plot_glt(glt_xyz,ort_rgb,plot_frames=True,titlestr='GLT',
             imgmax=1500,rgb_nl=1000):
    import pylab as pl
    pl.ioff()    
    glt_keep = glt_xyz[:,2]>=0
    
    glt_sel = glt_xyz[glt_keep,:]
    rgb_sel = ort_rgb[glt_keep,:]
    if len(rgb) == 0:
        imgmin = rgb_sel[:buf_nl,:].min(axis=0)
        if plot_frames:
            imgdif = imgmax-imgmin 
        else:
            imgdif = rgb_sel[:rgb_nl,:].max(axis=0)-imgmin 
    
    rgb = clip((rgb_sel-imgmin)/imgdif,0.0,1.0)
    
    fig = pl.figure(2,figsize=(10,10),dpi=100)
    pl.hold('on')
    ax = fig.add_subplot(111)
    ax.set_rasterization_zorder(1)
    ax.scatter(glt_sel[:,1],glt_sel[:,0],s=10,
               marker='s',c=rgb,zorder=0)                
    left, top = .01, .99
    ax.text(left,top,titlestr,color='red',
            horizontalalignment='left',
            verticalalignment='top',fontsize=10,
            rotation=0,family='monospace',
            transform=ax.transAxes)    

if __name__ == '__main__':
    import doctest
    doctest.testmod()
