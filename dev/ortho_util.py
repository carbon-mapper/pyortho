from __future__ import division
import pylab as pl
import sys, os, datetime, json

from os import SEEK_CUR,SEEK_SET
from os.path import join as pathjoin, exists as pathexists, split as pathsplit
from os.path import splitext, abspath, getsize, realpath, isabs

import warnings

def formatwarning(*args):
    message, category, filepath, lineno = args[:4]
    filedir,filename = pathsplit(filepath)
    return "%s (%s:%s): %s\n" % (category.__name__, filename, lineno, message)

warnings.formatwarning = formatwarning
warn = warnings.warn

# set up functions necessary to load local modules first
def wait_exit(retval,msg=''):
    if msg != '':
        print >> sys.stderr, msg
    raw_input()
    sys.exit(retval)

def get_env(var_name, default=None):
    # fall back to default when available without waiting
    env_value = os.getenv(var_name) or default
    if env_value is None:
        # error and wait if value not found and no default provided
        wait_exit(FAILURE,'Error (get_env): %s environment variable not defined'%var_name)

    return env_value

# global path variables (inferred with respect to this file)
PYORT_ROOT = realpath(pathsplit(__file__)[0]) # (i.e., directory of this script)
PYEXT_ROOT = pathjoin(PYORT_ROOT,'external')
ORTHO_ROOT = realpath(pathjoin(PYORT_ROOT,'..'))
PLATFORM_ROOT = pathjoin(PYORT_ROOT,'platform')

# update python paths
PYEXT_PKG = ['sunpos-1.1']
sys.path.extend([PYORT_ROOT,PYEXT_ROOT])
for pkg_path in PYEXT_PKG:
    sys.path.append(pathjoin(PYEXT_ROOT,pkg_path))

# default paths
GEO_FILE  = pathjoin(PYORT_ROOT,'geoid/geoid.json')
DEM_PREFIX  = get_env('DEM_PREFIX',realpath(pathjoin(ORTHO_ROOT,'data/dem')))
OUTPUT_PATH = get_env('WATCH_OUTPUT','/tmp')
    

dtime_now = datetime.datetime.now
time_elapsed = lambda start_time: str(dtime_now()-start_time)[:-4]

import signal

from numpy import r_, c_, array, asarray, asmatrix, arange, dot, sqrt, ravel
from numpy import float32, uint64, uint32, int32, int8, bool8, float64 as double
from numpy import bitwise_and, meshgrid, apply_along_axis, tensordot, dstack
from numpy import fromfile, where, sign, floor, ceil, ones, zeros, round
from numpy import argsort, sort, searchsorted, unique, cumsum, polyfit, polyval
from numpy import pi, tan, sin, cos, arccos, arcsin, arctan, arctan2, hypot 
from numpy import roll as np_roll, sum as np_sum, max as np_max, min as np_min
from numpy import nonzero, gradient, memmap, zeros_like, lexsort, linspace
from numpy import inf, nan, isnan, ones_like, zeros_like
from numpy import mean, median, diff, setdiff1d, mgrid
from numpy.linalg import inv

from numpy import set_printoptions
set_printoptions(suppress=True,precision=6)

from scipy.ndimage.interpolation import map_coordinates
from skimage.segmentation import find_boundaries

from spectral.io.envi import read_envi_header, write_envi_header, \
    open as envi_open, create_image as envi_create_image

from numexpr import evaluate as neval
from numba import autojit

# local and external imports
import sun_dist
from LatLongUTMconversion import LLtoUTM, UTMtoLL
from sunpos import sunpos, cLocation, cTime, cSunCoordinates
from find_pixel_trace_cython import find_pixel_trace_cython
#from bilerp_cython import bilerp_cython

use_numpy = False
use_numba = False
use_cython = True

# allow negative elevation values (below sea-level)
allow_negative_elevation=False 

# allow empty dems (elev_{i,j} \in {0,NODATA} \forall rows i, cols j) 
allow_empty_dem=True     

# downtrack averaging params
ORT_PS        = 1 # default output pixel resolution (meters)
BIN_FACTOR    = 1 # default downtrack binning factor (number of raw lines/frames)

# pixel size, frame binning, scaling and conversion factors
MIN_PS        = 0.1   # min pixel size (meters)
MAX_PS        = 60.0  # max pixel size (meters)

ALT_DELTA     = 25 # reduce dem_max by this value (in meters) if alt>dem_max
GLT_MAX_NPIX  = 40000**2 # output warning if GLT larger than this

ps_avg_fn     = median # function used to estimate dt/at ps (either mean or median)

# exit codes
SUCCESS     = 0
FAILURE     = 1

# pps/gps and other georeferencing constants
DATUM_WGS84 = 23
OBC_BYTE    = 641
OBC_SCIENCE = 3
SYNC_MSG    = 33279
DENOM32     = 2147483648.0
DENOM64     = 9.22337203685e+18 * 2**20

PPS_COEF    = 65536**arange(4)
PPS_COLS    = 'gpstime,counter,frame_count'.split(',')
GPS_COEF    = 256**asarray([4,5,6,7,0,1,2,3],int)
GPS_COLS    = 'gpstime,lat,lon,alt,pitch,roll,heading'.split(',')
GPS_EPOCHJD = 2444245

# pixel-level ortho error codes
PIX_ERROR_UNDEF       = -9900 # NOTE: all error codes must be < PIX_ERROR_UNDEF
PIX_ERROR_OUTSIDE_PB  = -9901
PIX_ERROR_NO_LOC      = -9902
PIX_ERROR_DEM_EXTENT  = -9903
PIX_ERROR_DEM_NOZINT  = -9904

PIX_ERROR_MSG = {PIX_ERROR_UNDEF:      'uninitialized pixel',
                 PIX_ERROR_OUTSIDE_PB: 'pixel outside pushbroom bounds',
                 PIX_ERROR_NO_LOC:     'cannot determine pixel ground location from pps/gps',
                 PIX_ERROR_DEM_EXTENT: 'pixel outside DEM extent',
                 PIX_ERROR_DEM_NOZINT: 'no positive z intersection'}

# geoid bounds
LATMIN,LATMAX = -90.0,  90.0
LONMIN,LONMAX =   0.0, 360.0        

DEG2RAD       = double(pi/180.0)


# logging and utility functions
log_prefs = {'verbose': 0}
def log(msg, priority=1, outfile=sys.stdout):
    """ print the given message iff the verbose level is high enough """
    global log_prefs
    if log_prefs['verbose'] >= priority:
        print >> outfile, msg

def wait_exit(retval,msg=''):
    if msg != '':
        print >> sys.stderr, msg
    raw_input()
    sys.exit(retval)

def datestr(fmt='%m%d%y',date=datetime.datetime.now()):
    return date.strftime(fmt)
    
def get_env(var_name, default=None):
    env_value = os.getenv(var_name)
    if env_value is None:
        if default is None:
            wait_exit(FAILURE,'Error (get_env): %s environment variable not defined'%var_name)
        else:
            env_value = default
    return env_value

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

def functime(func):
    """
    functime(func)

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

def get_projection(imgf):
    img = envi_open(imgf+'.hdr',image=imgf)
    map_info = img.metadata['map info'] 
    return map_info[0]

def valid_platforms():
    import glob
    platform_files = glob.glob(pathjoin(PLATFORM_ROOT,'*.json'))
    platforms = {}
    for pf in platform_files:
        platform_id = pf.replace('.json','').upper()
        platforms[platform_id] = load_platform(pf)
    return platforms

def identify_platform(fname):
    platforms = valid_platforms()
    fbase = pathsplit(fname)[1]
    for p_id, p_vars in platforms.iteritems():
        prefixes = p_vars.get('filename_prefix',[])
        if len(prefixes)==0:
            warn('WARNING: no filename prefixes provided for platform "%s", cannot identify from file'%p_id)
            break 
        for val in prefixes: 
            if fbase.startswith(val):
                return load_platform(p_id)
    return 'UNKNOWN'

def load_platform(platform_id,imgf=None,camf=None,geof=None):
    platform_json = pathjoin(PLATFORM_ROOT,platform_id.upper()+'.json')
    with open(platform_json,'r') as fid:
        parms = json.load(fid)
        platform_id = parms.pop('platform_id').upper()

        # update camera/geoid paths if necessary
        if camf is None:
            camera_file = parms.get('camera_file') 
        else:
            camera_file = realpath(camf) 
            
        if not isabs(camera_file):
            parms['camera_file'] = realpath(pathjoin(PYORT_ROOT,camera_file))
        else:
            parms['camera_file'] = camera_file        

        if geof is None:
            geoid_file = parms.get('geoid_file') 
        else:
            geoid_file = realpath(geof) 
            
        if not isabs(geoid_file):
            parms['geoid_file'] = realpath(pathjoin(PYORT_ROOT,geoid_file))
        else:
            parms['geoid_file'] = geoid_file
            
        # init/validate platform
        platform = PLATFORM(platform_id,parms)
        if imgf is not None and platform.checkfileprefix(imgf) == FAILURE:
            warn('WARNING: unexpected filename prefix for platform %s'%platform.platform_id)
        return platform


class PLATFORM():
    """
    PLATFORM

    Sensor platform constants and metadata for frame parsing

    Keyword Arguments:
    - platform_id: either 'AVIRIS-NG' or 'PRISM' (default='AVIRIS-NG') 
    
    """
    def __init__(self, platform_id, platform_vars):
        self.platform_id = platform_id
        self.prefixes    = platform_vars['filename_prefix']
        self.camf        = platform_vars['camera_file']
        self.geof        = platform_vars['geoid_file']
        self.NS          = platform_vars['NS']
        self.NC          = platform_vars['NC']
        self.PPS_MSG     = platform_vars['PPS_MSG']                
        self.NFRAME      = self.NS*self.NC
        self.NRAW        = self.NFRAME + self.NS # (1 extra channel)

    def __str__(self):
        outstr = self.platform_id
        outstr += ' camera=%s'%self.camf
        outstr += ' (NC=%d, NS=%d, NFRAME=%d, NRAW=%d)'%(self.NC,self.NS,
                                                        self.NFRAME,self.NRAW)
        return outstr
        
    def checkfileprefix(self, fpath):
        if self.platform_id == 'unknown':
            return FAILURE

        _,fname = pathsplit(fpath)
        for p in self.prefixes:
            if fname.startswith(p):
                return SUCCESS

        print 'ERROR: invalid filename prefix for platform %s: %s'%(self.platform_id, fname) 
        return FAILURE


    
class NAV():
    def __init__(self,platform,ppsf,gpsf):
        """
        NAV(platform,ppsf,gpsf)

        Initializes ortho nav parameters for a given platform
        (typically pps,gps,geoid and camera)

        Arguments:
        - platform: platform class instance (contains camera,geoid params) 
        - ppsf: path to pps table file
        - gpsf: path to gps table file

        Keyword Arguments:
        None        

        Returns:
        geo geoids, camera model 
        """
        
        self.initialized   = False
        self.platform      = platform
        
        if self.platform.platform_id == 'unknown':
            print 'ERROR: cannot load constants, bad platform_id "%s"'%platform_id
            return        
        
        if not pathexists(ppsf):
            print 'WARNING: pps file %s not found!'%ppsf

        if not pathexists(gpsf):
            print 'WARNING: gps file %s not found!'%gpsf
        
        self.ppsf          = ppsf
        self.gpsf          = gpsf
        self.pps_table     = []
        self.gps_table     = []
        self.gps_nl        = 0
        self.pps_nl        = 0
        self.gps_size_prev = 0
        self.pps_size_prev = 0        

        geof = self.platform.geof
        if not pathexists(geof):
            print 'ERROR: geoid file %s not found!'%geof
            return

        with open(geof,'r') as geo_fid:
            geoid_parms = json.load(geo_fid)
            geoidf     = geoid_parms['geoid_file']
            geoid_dims = geoid_parms['geoid_dims']            
            geoid_dps  = geoid_parms['geoid_dps']            
            geoid_LONMIN  = geoid_parms['geoid_LONMIN']            
            geoid_LATMAX  = geoid_parms['geoid_LATMAX']            
        
        camf = self.platform.camf
        if not pathexists(camf):
            print 'ERROR: camera file %s not found!'%camf
            return

        with open(camf,"rb") as cam_fid, open(geoidf, "rb") as geo_fid:
            cam_model = fromfile(cam_fid,dtype='<d').reshape([-1,3]).T
            geoid = fromfile(geo_fid,dtype='<f').reshape(geoid_dims).T

        self.geoid       = geoid
        self.geoid_dps   = geoid_dps
        self.geoid_LATMAX  = geoid_LATMAX
        self.geoid_LONMIN  = geoid_LONMIN
                
        
        self.cam_model   = cam_model
        self.pb_len      = int(cam_model.shape[1])
        self.pb_cen      = int(self.pb_len/2)
        self.initialized = True        

    def compute_geoidtrace(self,lon,lat,interp=True):
        geo_dps = self.geoid_dps
        geoidlat  = (lat-self.geoid_LATMAX)/(-geo_dps)
        geoidlon  = (lon-self.geoid_LONMIN)/geo_dps
        if interp:
            return bilerp(self.geoid,geoidlat,geoidlon)
        else:            
            return self.geoid[int(geoidlat),int(geoidlon)]
            
    def collect_tables(self,update_mode='reload',check_size=False,pps_nappend=10):
        """
        init / dynamically update pps/gps tables as they're written to disk
        """

        return_code=SUCCESS
        if update_mode == None: # don't update, just return tables
            return self.pps_table, self.gps_table, return_code
        
        if check_size:
            # NOTE (BDB, 08/31/15): enabling these can potentially cause a
            # race condition if the pps/gps files are corrupted or don't change
            pps_size = 0
            if pathexists(self.ppsf):
                pps_size = getsize(self.ppsf)

            gps_size = 0
            if pathexists(self.gpsf):
                gps_size = getsize(self.gpsf)

            update_tables = False
            if update_mode == 'append':
                # TODO (BDB, 09/13/15): check to see if we've read enough lines to not update in append mode 
                update_tables = True 

            if self.pps_size_prev == 0 or self.gps_size_prev == 0:
                update_tables = True # always update if either size == 0
            
            if pps_size > self.pps_size_prev:
                self.pps_size_prev = pps_size
                update_tables = True

            if gps_size > self.gps_size_prev:
                self.gps_size_prev = gps_size
                update_tables = True

            if update_tables == False: # tables unchanged and not empty, return early
                return self.pps_table, self.gps_table, return_code

        PPS_MSG = self.platform.PPS_MSG
        if update_mode=='reload': # reload tables from scratch
            print 'Reloading PPS/GPS tables'

            try:
                pps_table,pps_nl = read_pps(self.ppsf,PPS_MSG)
                self.pps_table = pps_table
                self.pps_nl    = pps_nl
            except Exception, e:
                print 'An unexpected error occurred updating the PPS table:', e
                pass
            
            try:
                # parse 10 gps lines for each pps line
                gps_table,gps_nl,return_code = read_gps(self.gpsf)
                self.gps_table = gps_table
                self.gps_nl    = gps_nl                                
            except Exception, e:
                print 'An unexpected error occurred updating the GPS table:', e
                pass                    
            
        elif update_mode=='append': # parse  tables from file            
            print 'Appending new lines to PPS/GPS tables'
            try:
                pps_chunk,pps_nc = read_pps(self.ppsf,PPS_MSG,sl=self.pps_nl,
                                            nl=pps_nappend)
                if pps_nc > 0:
                    if self.pps_nl > 0:
                        self.pps_table = r_[self.pps_table,pps_chunk]
                    else:
                        self.pps_table = pps_chunk
                    self.pps_nl        += pps_nc
                  
            except Exception, e:
                print 'An unexpected error occurred updating the PPS table:', e
                pass

            try:
                # parse 10 gps lines for each pps line
                gps_chunk,gps_nc,return_code = read_gps(self.gpsf,sl=self.gps_nl,
                                                        nl=10*pps_nappend)
                if gps_nc > 0:
                    if self.gps_nl > 0:
                        self.gps_table = r_[self.gps_table,gps_chunk]
                    else:
                        self.gps_table = gps_chunk
                    self.gps_nl        += gps_nc                
            except Exception, e:
                print 'An unexpected error occurred updating the GPS table:', e
                pass    
        else:
            print 'Undefined update mode "%s", no table updates performed'%update_mode

        print 'PPS table num_lines:', self.pps_nl
        print 'GPS table num_lines:', self.gps_nl            
        return self.pps_table, self.gps_table, return_code

    def clock2location(self,clock):
        """
        clock2location(clock,pps_table,gps_table)

        Given clock time, determine location via linear interpolation in
        PPS and GPS tables

        Arguments:
        - clock: clock time from frame header
        - pps_table: table of Precise Positioning Service pulses
                     # pps layout [pps clock count]
        - gps_table: [num_gps x 7] table of PPS-indexed GPS positions
                     # gps layout [pps,lat,lon,alt,pitch,roll,heading]

        Keyword Arguments:
        None

        Returns: 
        - lat,lon,altitude,pitch,roll,heading
        """
        pps_table, gps_table, ignore_code = self.collect_tables(update_mode=None)
        # David's mod
	if pps_table is None or len(pps_table) < 2:
            return []

        if clock < pps_table[0,1] or clock > pps_table[-1,1]:
            return []

        # make sure we're working with doubles
        # clock = double(clock)
        # pps_table = double(pps_table)
        # gps_table = double(gps_table)

        # given our clock entry, lerp the nearest pps_table entries
        approx_idx = searchsorted(pps_table[:,1],clock,'right')
        pps_lower,clock_lower = pps_table[approx_idx-1,:2]
        pps_upper,clock_upper = pps_table[approx_idx,:2]
        pps_offset = (pps_upper-pps_lower) * (clock-clock_lower) / \
            (clock_upper-clock_lower)
        pps = pps_lower + pps_offset

        # exit if our pps entry is outside the gps table
        if pps < gps_table[0,0] or pps > gps_table[-1,0]:
            return []

        # now lerp the gps entry given the interpolated pps value
        approx_idx = searchsorted(gps_table[:,0],pps,'right')
        gps_lower = gps_table[approx_idx-1,:]
        gps_upper = gps_table[approx_idx,:]
        pps_delta = (pps-gps_lower[0]) / (gps_upper[0]-gps_lower[0])
        location = (1.0-pps_delta)*gps_lower[1:7] + pps_delta*gps_upper[1:7]    
        return location.squeeze()    
    
class DEM():
    def __init__(self,demf,init_lonlat,subset_width=0.5):
        """
        __init__(demf,subset=True,init_lonlat,subset_width=0.5) 

        Initializes DEM, optionally subsetting a subset_width bounding box

        Arguments:
        - demf: dem file path

        Keyword Arguments:
        - init_lonlat: center coordinate of DEM subset (default=[])
        - subset_width: DEM bounding box subset width in degrees (default=0.5)

        Returns:
        - data_utm: DEM data (or subset thereof) in square UTM coords
        - meta: associated metadata
        """

        self.initialized = False
        if len(init_lonlat) == 0:
            print 'ERROR: init_lonlat not defined'
            return
        else:
            init_lon, init_lat = init_lonlat
            # take a subset bbox \pm subset_width centered on (init_lat,init_lon)
            shift_lon = subset_width if init_lon >= 0 else -subset_width        
            
            ullon,ullat = init_lon+shift_lon,init_lat+subset_width
            lrlon,lrlat = init_lon-shift_lon,init_lat-subset_width

            subset_ll = [ullon,ullat,lrlon,lrlat]        

        sref = get_projection(demf)            
        if sref=='Geographic Lat/Lon': # necessary to reproject to utm
            data_utm, meta = subset_latlon_reproject(demf,subset_ll)
        elif sref=='UTM':
            data_utm, meta = subset_utm(demf,subset_ll)

        if len(meta) == 0:
            print 'ERROR: unable to load DEM from %s'%demf
            return
            
        self.data_utm    = data_utm
        self.meta        = meta
        self.lrx         = meta[0][0]
        self.lry         = meta[0][1]
        self.ulx         = meta[0][2]
        self.uly         = meta[0][3]
        self.utm_zone    = meta[-3]
        self.utm_alpha   = meta[-2]
        self.utm_hemi    = meta[-1] 
        self.datum       = DATUM_WGS84
        self.extrema     = meta[-4]
        self.minv        = self.extrema[0]
        self.maxv        = self.extrema[1]
        self.ps          = self.meta[2]
        self.initialized = True

    
def rebin(a,f):
    """
    rebin(a,m) 

    Python version of IDL rebin: http://www.exelisvis.com/docs/REBIN.html
    Limitations: Handles 1-d case only, does NOT upsample.
    
    Arguments:
    - a: 1d input array
    - f: bin factor (>1)
    
    Keyword Arguments:
    None
    
    Returns:    
    """

    if f>1:        
        n = len(a)
        if n % f != 0:
            print 'f must be an integer multiple of n'
            return a

        ff   = float(f)
        m    = int(n*(1/ff))
        b    = zeros(m)
        b[:] = [np_sum(a[f*i:f*(i+1)])/ff for i in range(m)]

        return b
    elif f<1:
        print 'rebin: f must be greater than 1, cannot upsample'
    return a

def extrema(x):
    '''
    extrema(x)

    Computes the extrema of a list/array

    Arguments:
    - x: array like

    Keyword Arguments:
    None

    Returns:
    - min(x), max(x)
    '''
    return np_min(x),np_max(x)

def julday(month,day,year):
    a = (14 - month)//12
    y = year + 4800 - a
    m = month + 12*a - 3
    return day + ((153*m + 2)//5) + 365*y + y//4 - y//100 + y//400 - 32045

def gps2hour(gpstime):
    return (gpstime % 86400)/3600.0

def file2date(rawf):
    import re
    m = re.match(r'^.*([0-9]{8}t[0-9]{6}).*', rawf)
    if m is None:
        print 'Malformed filename %s, cannot extract date'%rawf
        return []

    sdate = m.group(1)
    year,month,day = map(int,[sdate[:4],sdate[4:6],sdate[6:8]])
    file_jd=julday(month,day,year)
    gps_week=int((file_jd-GPS_EPOCHJD)/7)    
    hour,min,sec = map(double,[sdate[9:11],sdate[11:13],sdate[13:]])
    return year,month,day,gps_week,hour,min,sec
    

def bilerp(gridxy, gridxf, gridyf):
    """
    bilinear interpolation on a regular grid
    """    
    return map_coordinates(gridxy,[[gridyf],[gridxf]],order=1,prefilter=False,
                           output=double).ravel()

def interp_nn(in_xy, in_val, out_xy):
    """
    2D nearest-neighbor interpolation on a regular grid
    """
    nn = NearestNDInterpolator(in_xy,in_val)
    return nn(out_xy)
    
def bilerp_numpy(gridxy, gridxf, gridyf):
    """
    bilerp_numpy(gridxy,gridxf,gridyf)

    Bilinear interpolation on a regularly-spaced grid
    
    Arguments:
    - gridxy: n x m grid of sample values to interpolate
    - gridxf: nx double x-coordinates within grid extent [0,m-1]
    - gridyf: ny double y-coordinates within grid extent [0,n-1]
    
    Keyword Arguments:
    None
    
    Returns:
    - interpolated gridxy values at points (gridxf,gridyf)
    """
    
    # gridxf,gridyf float indices (0-based)
    gridxi = gridxf.astype(int)
    gridyi = gridyf.astype(int)
    dx     = gridxf-gridxi
    dy     = gridyf-gridyi
    # gridx0y0,gridx1y0 = gridxy[gridyi,gridxi], gridxy[gridyi,gridxi+1]
    # gridx0y1,gridx1y1 = gridxy[gridyi+1,gridxi], gridxy[gridyi+1,gridxi+1]
    # return gridx0y0*(1.0-dx)*(1.0-dy) + gridx1y0*dx*(1.0-dy) + \
    #        gridx0y1*(1.0-dx)*dy       + gridx1y1*dx*dy
    # return     gridxy[gridyi,   gridxi  ] * (1.0-dx) * (1.0-dy) + \
    #            gridxy[gridyi,   gridxi+1] * dx  * (1.0-dy) + \
    #            gridxy[gridyi+1, gridxi  ] * (1.0-dx) * dy + \
    #            gridxy[gridyi+1, gridxi+1] * dx  * dy
    
    gridxyf  = gridxy[gridyi,   gridxi  ] * (1.0-dx) * (1.0-dy)
    gridxyf += gridxy[gridyi,   gridxi+1] * dx  * (1.0-dy)
    gridxyf += gridxy[gridyi+1, gridxi  ] * (1.0-dx) * dy
    gridxyf += gridxy[gridyi+1, gridxi+1] * dx  * dy
    return gridxyf

def lonlat2utm(lon,lat,zone=None):
    UTMZone, UTMEasting, UTMNorthing = LLtoUTM(23,lat,lon,ZoneNumber=zone)
    return UTMEasting, UTMNorthing, int(UTMZone[:-1]), UTMZone[-1]

def map2sl(x, y, ulx, uly, ps):
    """
    map2sl(x,y,ulx,uly,ps) 

    Given a defined grid find the s,l values for a given x,y
    
    Arguments:
    - x,y: map coordinates
    - ulx,uly: upper left map coordinate 
    - xps: x map pixel size
    
    Keyword Arguments:
    - yps: y map pixel size (default=xps)
    - rot: map rotation in degrees (default=0)
    
    Returns:
    - s,l: sample line coordinates of x,y
    """
    return (x-ulx)/ps, (uly-y)/ps

def map2sl_rot(x, y, ulx, uly, xps, yps=0, rot=0):
    """
    map2sl_rot(x,y,ulx,uly,xps,yps=xps,rot=0) 

    Given a defined grid find the s,l values for a given x,y
    
    Arguments:
    - x,y: map coordinates
    - ulx,uly: upper left map coordinate 
    - xps: x map pixel size
    
    Keyword Arguments:
    - yps: y map pixel size (default=xps)
    - rot: map rotation in degrees (default=0)
    
    Returns:
    - s,l: sample line coordinates of x,y
    """
    if yps == 0:
        yps = xps
    
    if rot==0:
        return (x-ulx)/xps, (uly-y)/yps
    
    ar = DEG2RAD*rot
    cos_ar,sin_ar = cos(ar), sin(ar)
    rotm = [[cos_ar,-sin_ar], [sin_ar,cos_ar]]
    rp,p0 = dot(rotm,r_[x, y]), dot(rotm,r_[ulx,uly])
    return (rp[0,:]-p0[0])/xps, (p0[1]-rp[1,:])/xps

def sl2map(s,l,ulx,uly,ps):
    """
    sl2map_rot(s,l,ulx,uly,ps) 

    Given pixel coordinates (s,l) convert to UTM map coordinates mapX,mapY
    
    Arguments:
    - s,l: sample, line indices
    - ulx,uly: upper left map coordinate 
    - ps: map pixel size 
    
    Keyword Arguments:
    None
    
    Returns:
    - mapX,mapY: s,l in map coordinates
    """
    return ulx+ps*s, uly-ps*l

def sl2map_rot(s,l,ulx,uly,xps,yps=0,rot=0):
    """
    sl2map_rot(s,l,ulx,uly,xps,yps=None,rot=0) 

    Given pixel coordinates (s,l) convert to UTM map coordinates mapX,mapY
    
    Arguments:
    - s,l: sample, line indices
    - ulx,uly: upper left map coordinate 
    - xps: x map pixel size 
    
    Keyword Arguments:
    - yps: y map pixel size (default=xps)
    - rot: map rotation in degrees (default=0)
    
    Returns:
    - mapX,mapY: s,l in map coordinates
    """
    if yps == 0:
        yps = xps

    if rot == 0:
        return ulx+xps*s, uly-yps*l
    
    #(ulx,xps_in,rotxps,uly,rotxps,yps_in) = geotransform (yps_in often < 0)
    X = ulx + xps * s + rot * l
    Y = uly + rot * s - yps * l    
    return X, Y

def plane2world_camera_model(pitch,roll,heading,camera):
    """
    plane2world_camera_model(orientdeg,camera) 

    Given the current (pitch,roll,heading), map camera model into enu frame
    
    Arguments:
    - pitch roll heading: in radians                  
    - camera:    [3 x pushbroom_length] frame offsets
    
    Keyword Arguments:
    None
    
    Returns:
    [3 x pushbroom length] world camera model
    """
    
    # C matrix converts from aero to enu frame
    C = [[0,  1,  0],
         [1,  0,  0],
         [0,  0, -1]]

    cos_roll,sin_roll = cos(roll),sin(roll)
    R_r = [[cos_roll, 0, -sin_roll], 
           [0,        1,  0], 
           [sin_roll, 0,  cos_roll]]
    
    cos_pitch,sin_pitch = cos(pitch),sin(pitch)
    R_p = [[1,  0,          0], 
           [0,  cos_pitch,  sin_pitch], 
           [0, -sin_pitch,  cos_pitch]]    
    
    cos_heading,sin_heading = cos(heading),sin(heading)
    R_h = [[cos_heading, -sin_heading, 0],
           [sin_heading,  cos_heading, 0], 
           [0,            0,           1]]

    # M= R_r*R_p*R_h = mout of navout_mout
    #return asarray((((asmatrix(R_r)*R_p)*R_h).T*C)*camera)
    return dot(dot(dot(dot(R_r,R_p),R_h).T,C),camera)

def find_pixel_trace_numpy(xs,ys,xe,ye):
    """
    find_pixel_trace(xs_in,ys_in,xe_in,ye_in,compute_s=True)

    Compute 2D shortest path between two pixel coordinates, optionally return
    cost of traversal
    
    Arguments:
    - xs_in, ys_in: start pixel
    - xe_in, ye_in: end pixel
    
    Keyword Arguments:
    - compute_s:  compute and return pixelwise traverse distances (default=True)
    
    Returns:
    - xpix,ypix: x,y pixel coordinates, optionally s
    """
    ixs,ixe=int(xs),int(xe)
    iys,iye=int(ys),int(ye)
    n_vert,n_hori = abs(ixe-ixs),abs(iye-iys)
    n_cross     = int(n_vert+n_hori)
    xys         = zeros([3,n_cross+1])
    xys[:2, 0]  = ixs,iys
    xys[:2,-1]  = ixe,iye
      
    dx,dy = xe-xs,ye-ys
    if n_cross > 1:
        cross_dtype = asarray(zeros(n_cross),dtype=[('d',double),('t',bool8)])

        # otherwise find path between (xs,ys), (xe,ye)
        if dx != 0:
            m = dy/dx
            sdx = dx/abs(dx)
        else:
            m = 1e+30 if dy > 0 else -1e+30
            sdx = 0
            
        sdy = dy/abs(dy) if dy != 0 else 0
            
        b = ye - m * xe
            
        if n_hori > 0:
            hori_y = arange(2,n_hori+1)+np_min([iys,iye])
            hori_x = (hori_y-b)/m if (dx != 0) else xs*ones(n_hori-1)
            cross_dtype['d'][n_vert+1:] = (hori_x-xs)**2+(hori_y-ys)**2
            #cross_dist[n_vert:] = distance(r_[xs,ys],c_[hori_x,hori_y])
            #cross_type[n_vert:] = type_hori # noop since type_hori==1

        if n_vert > 0:
            vert_x = arange(2,n_vert+1)+np_min([ixs,ixe])
            cross_dtype['d'][1:n_vert] = (vert_x-xs)**2+((m*vert_x+b)-ys)**2
            cross_dtype['t'][1:n_vert] = 1 # prioritize horiz over vert crossings

        # sort by distance then type to ensure correct ordering
        cross_dtype.sort(kind='quicksort',order='d')
        
        xys[0,1:-1] += ixs+sdx*cumsum((cross_dtype['t'][:-1]==1))
        xys[1,1:-1] += iys+sdy*cumsum((cross_dtype['t'][:-1]==0))
        xys[2,1:-1] = sqrt((cross_dtype['d'][:-1]+cross_dtype['d'][1:])/2)

    xys[2,-1] = sqrt(dx**2+dy**2)
    return xys


def find_pixel_trace_python(xs_in,ys_in,xe_in,ye_in,compute_s=True):
    """
    find_pixel_trace_python(xs_in,ys_in,xe_in,ye_in,compute_s=True)

    Compute 2D shortest path between two pixel coordinates, optionally return
    cost of traversal (mostly pure python version)
    
    Arguments:
    - xs_in, ys_in: start pixel
    - xe_in, ye_in: end pixel
    
    Keyword Arguments:
    - compute_s:  compute and return pixelwise traverse distances (default=True)
    
    Returns:
    - xpix,ypix: x,y pixel coordinates
    """
    # type_* prioritizes hori vs vert in sort below
    type_hori,type_vert=1,2     
    xs,ys=double(xs_in),double(ys_in)
    xe,ye=double(xe_in),double(ye_in)
    
    dx,dy=xe-xs,ye-ys
    ds=dx**2 + dy**2
    
    sgn_x,sgn_y=sign(dx),sign(dy)
    m=dy/dx if (dx != 0) else (1e+30 if (dy > 0) else -1e+30)
    b=ye-m*xe

    ixs,ixe=int32(xs),int32(xe)
    iys,iye=int32(ys),int32(ye)
    n_vert,n_hori=abs(ixe-ixs),abs(iye-iys)
    n_cross=n_vert+n_hori
    if n_cross > 1:
        cross_dist = zeros(n_cross)
        cross_type = ones(n_cross,dtype=int8)
        if n_vert > 0:
            vert_x = arange(1,n_vert+1)+min(ixs,ixe)
            cross_dist[:n_vert] = (vert_x-xs)**2+((m*vert_x+b)-ys)**2
            cross_type[:n_vert] = type_vert
            
        if n_hori > 0:
            hori_y = arange(1,n_hori+1)+min(iys,iye)    
            hori_x = (hori_y-b)/m if (dx != 0) else xs*ones(n_hori)
            cross_dist[n_vert:] = (hori_x-xs)**2+(hori_y-ys)**2
            #cross_type[n_vert:] = type_hori
            
        # vert_dist_sq = zeros(n_vert,dtype=int32)
        # hori_dist_sq = zeros(n_hori,dtype=int32)
        # if n_vert > 0:
        #     vert_x=arange(1,n_vert+1,dtype=int32)+min(ixs,ixe)
        #     vert_y=m*vert_x+b
        #     vert_dist_sq=(vert_x-ixs)**2+(vert_y-iys)**2

        # if n_hori > 0:
        #     hori_y=arange(1,n_hori+1,dtype=int32)+min(iys,iye)    
        #     hori_x=(hori_y-b)/m if (dx != 0) else ixs*ones(n_hori,dtype=int32)
        #     hori_dist_sq=(hori_x-ixs)**2+(hori_y-iys)**2


        # cross_type = r_[type_vert*ones(n_vert,dtype=int32),
        #                  type_hori*ones(n_hori,dtype=int32)]        
        # cross_dist = r_[vert_dist_sq,hori_dist_sq]
        
        # sort by distance then order to ensure correct ordering
        sorder=argsort(cross_dist)
        #sorder=argsort(c_[cross_dist,cross_order],axis=0)[:,0]
        cross_type=cross_type[sorder]

        xpix=ixs+r_[0,cumsum((cross_type==type_vert)*sgn_x)]
        ypix=iys+r_[0,cumsum((cross_type==type_hori)*sgn_y)]
        
        if compute_s:            
            cross_dist = sqrt(cross_dist[sorder])
            s=r_[0,(cross_dist+np_roll(cross_dist,-1))/2]
            s[n_cross]=sqrt(ds)
            return xpix,ypix,double(s)
    else:
        xpix,ypix=r_[ixs,ixe],r_[iys,iye]
        if compute_s:
            return xpix,ypix,double([0,sqrt(ds)])
    
    return xpix,ypix


def outside_bbox(x,y,maxx,maxy,minx=0,miny=0):
    """
    outside_bbox(x,y,maxx,maxy,minx=0,miny=0)

    Check if (x,y) are inside the bounding box [minx,miny,maxx,maxy]
    
    Arguments:
    - x,y: input x,y location or arrays of x,y locations (assumed int)
    - maxx,maxy: largest bounding box location
    
    Keyword Arguments:
    - minx,miny: smallest bounding box location (default=0,0)
    
    Returns:
    True or boolean mask if (x,y) outside bbox, False otherwise
    """
    ix,iy = int(x),int(y)
    return ((ix<minx) or (ix>maxx-1) or (iy<miny) or (iy>maxy-1))

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
    a = sin((lat2-lat1)/2.0)**2 + cos(lat1) * cos(lat2) * sin((lon2-lon1)/2.0)**2
    return arcsin(sqrt(a)) * 12742000.0 # meters 6371*1000*2

def read_frames_meta_bands(img_path, platform, sl=0, nl=99999,
                           alloc_lines=1000, bands=[]):
    if not pathexists(img_path):
        print 'ERROR: image file %s not found!'%img_path
        return array([]),array([]),0

    if len(bands) == 0:
        bands = arange(NC)

    NC = platform.NC
    NS = platform.NS
    NRAW = platform.NRAW
    NFRAME = platform.NFRAME    
        
    img_size = getsize(img_path)
    num_read = 0
    num_chan = len(bands)

    img_bands = zeros([alloc_lines,NS,num_chan])
    meta  = zeros([alloc_lines,2],dtype=uint64)
    g_14bit_mask = uint32(16383)
    obc_start = -1 # odbc start frame
    done = False    
    with open(img_path, 'r') as f:        
        if sl > 0:
            f.seek(sl*NRAW*2,SEEK_CUR)
            if f.tell() >= img_size:
                done=True        

        while not done and num_read < nl:
            # get frame header
            buf = fromfile(f, count=NS, dtype='<u2')
            if len(buf) == 0:
                done = True
            else:
                clock = 65536*buf[0]+buf[1]
                count = bitwise_and(g_14bit_mask, uint32(buf[160]))
                obcv  = buf[321] >> 8
                if obcv == OBC_SCIENCE and obc_start == -1:
                    obc_start = sl+num_read
                    
                # get image bands, compute intensity
                band_prev=0
                for jb,b in enumerate(bands):
                    f.seek((b-band_prev-1)*NS*2,SEEK_CUR)
                    bbuf = fromfile(f, count=NS,  dtype='<u2')
                    if len(bbuf) == 0:
                        buf = []
                        break
                                        
                    buf = dstack([buf, bbuf]) if jb>0 else bbuf
                    band_prev = b

                f.seek((NC-band_prev)*NS*2,SEEK_CUR)
                    
                if len(buf) == 0:
                    done = True
                else:                    
                    if num_read+1 == img_bands.shape[0]:
                        img_bands = r_[img_bands, zeros([alloc_lines,NS,num_chan])]
                        meta  = r_[meta,  zeros([alloc_lines, 2],dtype=uint64)]
                    img_bands[num_read,:,:] = buf                      
                    meta[num_read,:]  = [clock, count]
                    num_read += 1

        img_bands = img_bands[:num_read,:]
        meta = meta[:num_read,:]
    #print "read_frames_meta_bands: sl=%d, num_read=%d"%(sl,num_read)
    return img_bands, meta, num_read, obc_start

def read_frames_meta(img_path, platform, sl=0, nl=99999, alloc_lines=1000):
    if not pathexists(img_path):
        print 'ERROR: image file %s not found!'%img_path
        return array([]),0

    NC = platform.NC
    NS = platform.NS
    NRAW = platform.NRAW
    NFRAME = platform.NFRAME
    
    img_size = getsize(img_path)
    num_read = 0
    meta = zeros([alloc_lines,2],dtype=uint64)
    
    g_14bit_mask = uint32(16383)
    done = False
    obc_start = -1 # first obc science header, offset by sl
    with open(img_path, 'r') as f:
        if sl > 0:
            f.seek(sl*NRAW*2,SEEK_CUR)
            if f.tell() >= img_size:
                done=True        
        while not done and num_read < nl:
            # read frame header
            buf = fromfile(f, count=NS, dtype='<u2')
            if len(buf) == 0:
                done = True
            else:
                clock = 65536*buf[0]+buf[1]
                count = bitwise_and(g_14bit_mask, uint32(buf[160]))
                obcv  = buf[321] >> 8
                if obcv == OBC_SCIENCE and obc_start == -1:
                    obc_start = sl+num_read
                #elif obc_start != -1 and obcv != OBC_SCIENCE:
                #    done = True
                #    break # finished reading science frames
                
                f.seek(NFRAME*2,SEEK_CUR)
                if f.tell() >= img_size: # moved to/past EOF
                    #print 'tell %d>=nbytes %d (EOF)'%(f.tell(),img_size)
                    done = True
                else:
                    if num_read+1 == meta.shape[0]:                        
                        meta  = r_[meta, zeros([alloc_lines,2],dtype=uint64)]                
                    meta[num_read,:]  = [clock, count]
                    num_read += 1
        
        meta  = meta[:num_read,:]
    #print "read_frames_meta: num_read=%d"%num_read
    return meta, num_read, obc_start

def read_pps(pps_path, msg_size, sl=0, nl=99999):
    """
    read_pps(pps_path, sl=0) 
    
    Arguments:
    - pps_path: path to pps table file
    
    Keyword Arguments:
    - sl: number of lines to skip (default=0)
    - nl: maximum number of lines to read beyond 'sl' (default=99999)
    
    Returns:
    - pps_table: msg_read x 3 pps table file
    - msg_read: number of pps messages read
    """
    
    msg_read = 0
    time_table = array([],dtype=double)
    if not pathexists(pps_path):
        print 'WARNING: pps file %s not found!'%pps_path
        return time_table,msg_read

    # minimum possible size to get a valid pps frame = 15 bytes
    pps_size = getsize(pps_path)
    if pps_size < msg_size+2: 
        print 'WARNING: pps file %s size less than %d bytes'%(pps_path,msg_size+2)
        return time_table,msg_read

    with open(pps_path,'r') as f:
        # traverse file with sliding window to find sync message

        bytec = 0
        while fromfile(f,count=1,dtype='<u2') != SYNC_MSG:
            if bytec >= pps_size:
                print 'PPS file contains no sync messages'
                return [],[]
            # back up one byte to search every 2-byte string for sync_msg
            f.seek(-1,SEEK_CUR) 
            bytec += 1
            continue

        # found a sync_msg, back up to sync position
        f.seek(-2,SEEK_CUR) 
        if sl > 0:
            f.seek(sl*msg_size*2,SEEK_CUR)
            if f.tell() >= pps_size:
                return time_table, msg_read

        # read messages until we get a sync header or an empty buffer
        while msg_read < nl:
            if f.tell()+msg_size >= pps_size: # truncated file, return what we have
                break
            buf=fromfile(f,count=msg_size,dtype='<u2')
            if len(buf) < msg_size-1:
                print 'PPS message length too short'
                break
            elif buf[0] != SYNC_MSG:
                print 'Expected PPS synchronization word not found'
                break
            else:
                gpstime     = dot(buf[[7,8,5,6]],PPS_COEF) / DENOM64
                counter     = uint64(buf[10])*65536 + buf[11]
                frame_count = buf[9]
                if msg_read>0:
                    time_table=r_[time_table,c_[gpstime,counter,frame_count]]
                else:
                    time_table=c_[gpstime,counter,frame_count]
                msg_read+=1                

    return time_table, msg_read

def read_gps(gps_path, sl=0, nl=99999, recovery_bytes=10):
    """
    read_gps(gps_path,sl=0,nl=99999)

    Reads table of GPS values from file, optionally skipping 'sl' initial entries
    
    Arguments:
    - gps_path: path to gps table file
    
    Keyword Arguments:
    - sl: number of lines to skip (default=0)
    - nl: maximum number of lines to read beyond 'sl' (default=99999)
    - recovery_bytes: number of bytes to read if sync_msg not found (default=0)
    
    Returns:
    - gps_table: msg_read x 7 gps table
    - msg_read: number of lines read
    """
    
    return_code = SUCCESS
    msg_read = 0
    msg_skipped = 0
    locations = array([],dtype=double)
    if not pathexists(gps_path):
        print 'WARNING: GPS file %s not found!'%gps_path
        return locations,msg_read,return_code

    gps_size = getsize(gps_path)
    if gps_size == 0: 
        print 'WARNING: GPS file %s empty!'%gps_path
        return locations,msg_read,return_code

    file_done=False
    with open(gps_path,'r') as f:
        while not file_done:
            header=fromfile(f,count=5,dtype='<u2')
            if len(header)<5:
                return locations,msg_read,return_code

            if header[0] != SYNC_MSG:
                f.seek(-9,SEEK_CUR) # backup by (5*2)-1 to read even/odd byte msgs
                continue            
            
            msg_bytes=2*(header[2]+1)
            if f.tell()+msg_bytes > gps_size:
                # truncated message, return everything up until now
                print 'Truncated GPS message encountered, returning valid messages'                
                return locations,msg_read,return_code
            if header[1] == 3501:                    
                if msg_skipped < sl:
                    f.seek(msg_bytes,SEEK_CUR)
                    msg_skipped+=1
                else: 
                    gpstime=dot(fromfile(f,count=8,dtype='<u1'),GPS_COEF) / DENOM64
                    lat,lon=fromfile(f,count=2,dtype='<i4') / DENOM32 * 180
                    alt=fromfile(f,count=1,dtype='<i4') / DENOM32 * 32768
                    other=fromfile(f,count=6,dtype='<u2')
                    pitch,roll,heading=fromfile(f,count=3,dtype='<i4') / DENOM32 * 180
                    data_checksum=fromfile(f,count=1,dtype='<u2')
                    if len(data_checksum)==0:
                        file_done=True
                    elif abs(lat) <= 90.0 and abs(lon) <= 180.0:
                        if msg_read != 0:
                            locations = r_[locations,c_[gpstime,lat,lon,alt,pitch,roll,heading]]
                        else:
                            locations = c_[gpstime,lat,lon,alt,pitch,roll,heading]
                        msg_read+=1
                        if msg_read >= nl:
                            return locations,msg_read,return_code
                    else:
                        print 'ERROR: failed GPS checksum'
                        return locations,msg_read,return_code
            elif msg_bytes > 0:
                f.seek(msg_bytes,SEEK_CUR)
    return locations, msg_read, return_code



#@functime
def subset_latlon_reproject(imgf,subset_extent=[],utm_ps=None):
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
    except Exception, e:
        print 'ERROR: subset_latlon_reproject requires GDAL'
        return [],[]

    g = gdal.Open(imgf, gdal.GA_ReadOnly)
    nodata_val = g.GetMetadataItem('data ignore value')
    
    nl_in,ns_in = g.RasterYSize,g.RasterXSize
    geo_t = g.GetGeoTransform()
    geo_p = g.GetProjectionRef() 
    l_in, t_in, rot_in = geo_t[0],geo_t[3],geo_t[2]
    xps_deg, yps_deg = geo_t[1],abs(geo_t[5])    

    if utm_ps is None:
        # get x,y pixel sizes in meters and use the largest for resampling
        xps_m = gc_distance(t_in,l_in,t_in,l_in+xps_deg)
        yps_m = gc_distance(t_in,l_in,t_in+yps_deg,l_in)
        utm_ps = np_max([xps_m,yps_m])
    print 'utm_ps: ',utm_ps

    if len(subset_extent) == 0:
        # reproject full image
        r_in,b_in = l_in+xps_deg*ns_in, t_in-yps_deg*nl_in
        ullon,ullat,lrlon,lrlat = l_in,t_in,r_in,b_in
    elif len(subset_extent) == 4:
        ullon,ullat,lrlon,lrlat = subset_extent
    else:
        print 'ERROR: subset_extent should be in the form [ullon,ullat,lrlon,lrlat]'
        return [],[]
        
    ul_hemi = 'North' if ullat >= 0 else 'South'
    lr_hemi = 'North' if lrlat >= 0 else 'South'
    if ul_hemi != lr_hemi: # bounding box overlaps both N/S hemispheres
        print 'WARNING: bounding box overlaps Northern and Southern hemispheres'        
        # FIXME (BDB, 03/11/16): test this!
        
    ulx, uly, ul_zone, ul_alpha = lonlat2utm(ullon, ullat)
    lrx, lry, lr_zone, lr_alpha = lonlat2utm(lrlon, lrlat)

    imgdir,imgfile = pathsplit(imgf)
    print 'Subset Geographic Lat/Lon image %s'%imgfile
    print '-> Extent [ullon,ullat,lrlon,lrlat]', array(subset_extent)
    print '-> UTM zone %d (alpha=%s, hemisphere=%s)'%(ul_zone, ul_alpha, ul_hemi)
    print '-> UTM [minx,maxx,miny,maxy]',array([ulx,lrx,lry,uly])
    
    if lr_zone != ul_zone:
        print 'ERROR: UL and LR pixels in different UTM zones!'
        return [],[]

    rot = 0
    if geo_t[2] != 0 or geo_t[4] != 0:
        print 'WARNING: image %s is not in a north-up coordinate system'%imgf
        raw_input()
        if geo_t[2] == geo_t[4]:
            rot = geo_t[2]
        else:
            print 'ERROR: ill-defined rotation'
            return [],[]
        
    # assume a single zone / spatial reference
    utm_zone = ul_zone
    utm_sr   = osr.SpatialReference()
    epsg_id  = int((32600 if ul_hemi=='North' else 32700)+utm_zone)
    utm_sr.ImportFromEPSG(epsg_id)
    utm_wkt = utm_sr.ExportToWkt()

    wgs84_sr = osr.SpatialReference()
    wgs84_sr.ImportFromEPSG(4326)
    wgs84_wkt = wgs84_sr.ExportToWkt()

    # The size of the raster is given the new projection and pixel spacing
    # Using the values we calculated above. 
    ns = int((lrx-ulx)/utm_ps) 
    nl = int((uly-lry)/utm_ps)

    # use in-memory driver for the quickness
    mem_drv = gdal.GetDriverByName('MEM')
    dest = mem_drv.Create('', ns, nl, 1, gdal.GDT_Float32)

    # reproject to uniform pixel size here
    dest.SetGeoTransform((ulx, utm_ps, geo_t[2], uly, geo_t[4], -utm_ps))
    dest.SetProjection(utm_wkt)
    
    # Perform the projection/resampling to uniform pixel size 
    res = gdal.ReprojectImage(g, dest, wgs84_wkt, utm_wkt, gdal.GRA_Bilinear)

    dest_img     = asarray(dest.ReadAsArray().squeeze(),dtype=double)

    if nodata_val is not None:
        # zero out nodata values
        dest_img[dest_img==nodata_val] = 0

    dest_extrema = extrema(dest_img)
    print(dest_extrema)
    if not allow_negative_elevation and dest_extrema[0] < 0:
        dest_img[dest_img<0] = 0
        dest_extrema = (0,dest_extrema[1])

    if (dest_img==0).all():
        if allow_empty_dem:
            warn('all DEM elevation values <= zero or NODATA')
        else:
            print 'ERROR: all DEM elevation values <= zero or NODATA'
            return [],[]
        
    dest_extent  = lrx,lry,ulx,uly 
    dest_size    = nl,ns    
    dest_meta    = dest_extent,rot,utm_ps,dest_size,dest_extrema, \
                   ul_zone,ul_alpha,ul_hemi
    
    return dest_img, dest_meta

#@functime
def subset_utm(imgbase,subset_extent):
    '''
    extract a bounding box subset_extent (format=[ullon,ullat,lrlon,lrlat]) of imgf    
    - assumes imgf in UTM x/y (does *not* reproject from wgs84 to UTM)
    - assumes subset_extent contained entirely within imgf
    - does not handle images that overlap two utm zones (assumes zone of ullon)!
    '''
    if len(subset_extent)!=4:
        print 'ERROR: subset_utm requires [ullon,ullat,lrlon,lrlat] bounding extent'
        return [],[]

    ullon,ullat,lrlon,lrlat = subset_extent
    
    ul_hemi = 'North' if ullat >= 0 else 'South'
    lr_hemi = 'North' if lrlat >= 0 else 'South'
    if ul_hemi != lr_hemi: # bounding box overlaps both N/S hemispheres
        print 'WARNING: bounding box overlaps Northern and Southern hemispheres'        
        # FIXME (BDB, 03/11/16): test this!
        
    # convert lon/lat bbox into utm, get sample indices, and subset the image
    lx,uy,ul_zone,ul_alpha = lonlat2utm(ullon, ullat)
    rx,ly,lr_zone,lr_alpha = lonlat2utm(lrlon, lrlat)
    if lr_zone != ul_zone:
        print 'WARNING: UL and LR pixels in different UTM zones,',
        print 'assuming zone of UL pixel.'
        # recompute rx,ly in ul_zone
        rx,ly,lr_zone,lr_alpha = lonlat2utm(lrlon, lrlat, zone=ul_zone)

    imgf     = imgbase+str(ul_zone)
    img      = envi_open(imgf+'.hdr',image=imgf)
    img_hdr  = img.metadata
    map_info = img_hdr['map info']
    imgdir,imgfile = pathsplit(imgf)
    print 'Subset UTM image %s'%imgfile
    print '-> Extent [ullon,ullat,lrlon,lrlat]', array(subset_extent)
    print '-> UTM zone %d (alpha=%s, hemisphere=%s)'%(ul_zone, ul_alpha, ul_hemi)
    print '-> UTM [minx,maxx,miny,maxy]',array([lx,rx,ly,uy])

    nl_in, ns_in = int(img_hdr['lines']),int(img_hdr['samples'])
    nodata_val = float(img_hdr.get('data ignore value',None))
    rot_in = 0.0 # assume rotation = 0

    if map_info[0] != 'UTM':
        print 'ERROR: image %s invalid projection "%s", '%(pathsplit(imgf)[1],
                                                           map_info[0]),
        print 'UTM coordinates required'
        return [],[]

    # get UTM image bounding box from map info
    l_in, t_in = double(map_info[3]),double(map_info[4])    
    xps_in, yps_in = double(map_info[5]),double(map_info[6])
    r_in,b_in = sl2map(ns_in,nl_in,l_in,t_in,xps_in)
    
    if abs(xps_in) != abs(yps_in):
        print 'ERROR: subset_utm cannot process DEMs with different x and y pixel sizes'
        return [],[]            

    # x/y pixel size equal, pick one
    utm_ps = xps_in
    print 'utm_ps: ',utm_ps
    
    # upper left sample,line
    min_samp, min_line = map2sl(lx, uy, l_in, t_in, utm_ps)  
    min_samp, min_line = int(min_samp),int(min_line)
    
    # lower right sample,line
    max_samp, max_line = map2sl(rx, ly, l_in, t_in, utm_ps)
    max_samp, max_line = int(ceil(max_samp)),int(ceil(max_line))

    print 'min_line, max_line, min_samp, max_samp: %d, %d, %d, %d'%(min_line,
                                                                    max_line,
                                                                    min_samp,
                                                                    max_samp)
    
    if max_samp<min_samp:
        print 'ERROR: subset_utm max_samp should be >= min_samp! Bad extent?'
        return [],[]
    elif max_line<min_line:
        print 'ERROR: subset_utm max_line should be >= min_line! Bad extent?'
        return [],[]    

    if min_samp < 0:
        print 'WARNING: subset_utm min_samp < 0, clipping min_samp=0.'
        min_samp = 0
    elif max_samp > ns_in-1:
        print 'WARNING: subset_utm max_samp > ns (%d), clipping max_samp=ns.'%ns_in
        max_samp = ns_in-1

    if min_line < 0:
        print 'WARNING: subset_utm min_line < 0, clipping min_line=0.'
        min_line = 0
    elif max_line > nl_in-1:
        print 'WARNING: subset_utm max_line > nl (%d), clipping max_line=nl.'%nl_in
        max_line = nl_in-1          
    
    lx_new,uy_new = sl2map(min_samp,min_line,l_in,t_in,utm_ps)
    rx_new,ly_new = sl2map(max_samp,max_line,l_in,t_in,utm_ps)

    nl_new = max_line-min_line+1
    ns_new = max_samp-min_samp+1

    
    #print 'Original bbox (l, r, b, t): %8.6f, %8.6f, %8.6f, %8.6f'%(l_in,r_in,b_in,t_in)
    #print 'Original dims (lines, samples): %d, %d'%(nl_in, ns_in)
    #print 'Subset bbox (l, r, b, t): %8.6f, %8.6f, %8.6f, %8.6f'%(lx,rx,ly,uy)
    #print 'Subset dims (lines, samples): %d, %d'%(nl_new, ns_new)
    
    try:
        img_mm   = img.open_memmap(interleave='source', writable=False)
        dest_img = img_mm[0,min_line:max_line,min_samp:max_samp]
    except Exception, e:
        print 'WARNING: cannot read image using memmap interface, trying read_subregion'
        dest_img = img.read_subregion([min_line,max_line],[min_samp,max_samp],
                                      use_memmap=False)
        if dest_img is None:
            print 'ERROR: unable to read', imgf
            return [],[]
        
    dest_img = asarray(dest_img.squeeze(),dtype=double)

    # zero out nodata values
    if nodata_val is not None:
        dest_img[dest_img==nodata_val] = 0
    
    dest_extrema = extrema(dest_img)

    if not allow_negative_elevation and dest_extrema[0] < 0:
        dest_img[dest_img<0] = 0
        dest_extrema = (0,dest_extrema[1])

    if not allow_empty_dem and (dest_img==0).all():
        print 'ERROR: all DEM elevation values <= zero or NODATA (=%s)'%str(nodata_val)
        return [],[]

    dest_extent  = rx_new,ly_new,lx_new,uy_new
    dest_size    = nl_new,ns_new
    dest_meta    = dest_extent,rot_in,utm_ps,dest_size,dest_extrema, \
                   ul_zone,ul_alpha,ul_hemi
    return dest_img, dest_meta

#@profile
def geolocate(lines, samples, clock, nav, dem, rs_ps=4.0, return_lonlat=True,
              z_int_tol=-1, interp_geoid=True, check_bounds=False,
              cache_max_elev=False, offset_latlon=[], frame_meta=[], verbose=0):
    """
    geolocate(lines, samples, clock, nav, dem, z_int_tol=1e-8, rs_ps=4.0,
              z_int_tol=rs_ps/10.0, interp_geoid=True, return_lonlat=True,
              check_bounds=False, offset_latlon=[], verbose=0) 

    Given a list of n_ls ([line],[sample]) pixel locations, along with nav and DEM info
    return array ground_loc of the utm (x,y,z) coordinates of the pixel locations
    Bad coordinates indicated with ground_loc[:,2] < 0

    Arguments:
    - lines:          [n_ls x 1] line number of each coordinate
    - samples:        [n_ls x 1] sample number of each coordinate
    - clock:          [n_ls x 1] frame clock values for each coordinate
    - nav:            navigation class with camera / world model
    - DEM:            DEM with georeferenced UTM elevation map
    
    Keyword Arguments:
    - rs_ps:          resampled pixel size in meters (default=4.0)
    - return_lonlat:  return wgs84 lon/lat instead of utm x/y (default=True)
    - interp_geoid:   interpolate when computing geoid correction (default=True)
    - z_int_tol:      tolerance to accept small negative z-intersections (default=rs_ps/10.0)
                      NOTE: only used when rs_ps and dem_ps are not equal
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
    
    global log_prefs
    log_prefs['verbose'] = verbose

    if len(nav.geo) == 0 or len(nav.cam_model) == 0:
        print "ERROR: world / camera models not initialized (did you run init_ortho?)"
        return []

    if len(nav.pps_table) == 0 or len(nav.gps_table) == 0:
        print "ERROR: empty gps/pps tables"
        return []

    if len(dem.data_utm) == 0:
        print "ERROR: DEM not initialized (did you run init_dem?)"
        return []
    
    n_ls = len(lines)
    if n_ls == 0:
        print 'WARNING: no coordinates to geolocate, exiting.'
        return []    

    if len(samples) != n_ls:
        print "ERROR: len(samples) != len(lines)."
        return []
   
    if len(clock) != n_ls:
        print "ERROR: len(clock) != len(lines)."
        return []  

    
    dem_extent,dem_r,dem_ps,dem_size,dem_extrema = dem.meta[:5]
    dem_zone,dem_alpha,dem_hemi = dem.meta[5:]
    dem_utm         = dem.data_utm
    dem_nl,dem_ns   = dem_size
    dem_min,dem_max = dem.minv,dem.maxv
    dem_xns,dem_yns,dem_x0,dem_y0 = dem_extent

    scalef_ps = rs_ps/dem_ps
    if scalef_ps != 1.0:
        # if our pixel size rs_ps isn't the same as the dem, bilerp
        print 'Pixel size scale factor = %f, resampling'%scalef_ps
        interpf = bilerp
        eff_ps  = rs_ps # effective pixel size of image wrt DEM
        if z_int_tol == -1:
            z_int_tol = 10*scalef_ps #1e-8 #1e-8 #rs_ps/4.0
            #print 'z_int_tol:',z_int_tol
    else:
        # otherwise just use the exact values in the dem
        print 'Pixel size scale factor = 1.0, not resampling'
        interpf   = lambda img,xp,yp: img[int(yp),int(xp)].ravel()
        eff_ps    = dem_ps
        z_int_tol = 0.0

    # each frame is associated with a unique clock measurement
    uframes,uidx = unique(lines,return_index=True)
    framec       = len(uframes)

    # output dimensions = [lon,lat,z,utmx,utmy] or [utmx,utmy,z]
    out_dim      = 5 if return_lonlat else 3 
    ground_loc   = PIX_ERROR_UNDEF*ones([n_ls,out_dim]) 
    no_loc       = 0
    
    print 'Correcting %d frames with %d samples/frame'%(framec,len(lines)/framec)

    # nav local variable/function references
    pb_len             = nav.pb_len
    cam_model          = nav.cam_model
    clock2location     = nav.clock2location
    compute_geoidtrace = nav.compute_geoidtrace
    ls_offset          = 0 # offset into ground_loc for the first sample in each frame
    #for frame_idx,frame_clock in zip(uframes,uclock):
    for frame_idx in uframes:
        frame_clock = double(clock[frame_idx])
        frame_mask  = neval('lines==frame_idx')
        samp_idx    = samples[frame_mask]
        ls_nsamp    = len(samp_idx) # number of samples at this frame index
        bad_mask    = neval('samp_idx>pb_len-1')
        if bad_mask.any():
            print 'WARNING: no frame offset for line %d samples %s, skipping'%(frame_idx,str(bad_idx))
            badloc_idx  = ls_offset+where(bad_mask)[0]
            ground_loc[badloc_idx,:]   = PIX_ERROR_OUTSIDE_PB
            # note: do not need a 'continue' here, since we're just skipping samples

        # get aircraft coordinates, convert to utm
        clock_loc = clock2location(frame_clock)
        if len(clock_loc) == 0:
            no_loc += 1
            ground_loc[ls_offset:ls_offset+ls_nsamp,:]   = PIX_ERROR_NO_LOC
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
        gridnorth        = sin(DEG2RAD*lat)*(((abs(utm_zone)-31)*6+3)-lon)
        heading          = DEG2RAD*(clock_loc[5]+gridnorth)

        # map lon to [0,360] (*after* utm/gridnorth correction!)
        if lon < 0.0:         
            lon = 360.0+lon

        # apply geoid correction
        geoidtrace = compute_geoidtrace(lon,lat,interp=interp_geoid)
        altitude  -= geoidtrace
        
        # update maximum elevation if we're at lower altitude than max(dem)
        if dem_max > altitude:
            msg  = 'Reducing dem_max (%4.1f) to '%dem_max
            dem_max  = altitude-ALT_DELTA
            msg += 'altitude (%4.1f)-alt_delta (%4.1f) = %f.'%(altitude,
                                                               ALT_DELTA,
                                                               dem_max)
            warn(msg)            
            if cache_max_elev:
                dem.maxv = dem_max            

        # keep track of frame metadata for obs files
        frame_info = [frame_clock,utm_x,utm_y,altitude,pitch,roll,heading]
        frame_meta.append(frame_info)
            
        # get ray directions for selected samples
        frame_camera = cam_model[:,samp_idx]
        frame_xyz = plane2world_camera_model(pitch,roll,heading,frame_camera)
        
        # TODO (BDB, 07/21/15): VECTORIZE THIS LOOP 
        # project down to min/max DEM elevation 
        f_x,f_y,f_z = frame_xyz[0,:],frame_xyz[1,:],frame_xyz[2,:]
        
        d_min,d_max = (dem_min-altitude)/f_z, (dem_max-altitude)/f_z
        xout_lo_mapa,yout_lo_mapa = utm_x+d_min*f_x, utm_y+d_min*f_y
        xout_hi_mapa,yout_hi_mapa = utm_x+d_max*f_x, utm_y+d_max*f_y
        
        ixout_loa,iyout_loa = map2sl(xout_lo_mapa,yout_lo_mapa,dem_x0,dem_y0,eff_ps)
        ixout_hia,iyout_hia = map2sl(xout_hi_mapa,yout_hi_mapa,dem_x0,dem_y0,eff_ps)

        # trace ray through intersected pixels, get lo/hi intersections
        z_minmax = zeros([2,frame_xyz.shape[1]])
        for j,sj in enumerate(samp_idx):
            ls_idx = ls_offset+j # offset into ground_loc array for this sample
            if ground_loc[ls_idx,2] < PIX_ERROR_UNDEF:
                # only skip if pixel flagged with error other than PIX_ERROR_UNDEF
                continue
            ixout_hi,ixout_lo = ixout_hia[j],ixout_loa[j]
            iyout_hi,iyout_lo = iyout_hia[j],iyout_loa[j]
            if check_bounds and (outside_bbox(ixout_lo*scalef_ps,
                                              iyout_lo*scalef_ps,
                                              dem_ns,dem_nl) or
                                 outside_bbox(ixout_hi*scalef_ps,
                                              iyout_hi*scalef_ps,
                                              dem_ns,dem_nl)):
                warn('bounds for traced pixel %d (frame=%d, sample=%d) outside DEM extent, skipping'%(ls_idx,frame_idx,sj),1)
                log(str(scalef_ps*array([ixout_hi,iyout_hi,ixout_lo,iyout_lo,
                                         dem_ns/scalef_ps, dem_nl/scalef_ps])),1)
                ground_loc[ls_idx,:] = PIX_ERROR_DEM_EXTENT
                continue
            
            xys_tr = find_pixel_trace(ixout_hi,iyout_hi,ixout_lo,iyout_lo)
            #print xpix_tr, ypix_tr
            #xpix_tr,ypix_tr = skline(ixout_hi,iyout_hi,ixout_lo,iyout_lo,compute_s=False) 
            z_minmax[:,j] = extrema(interpf(dem_utm,xys_tr[:,0]*scalef_ps,
                                            xys_tr[:,1]*scalef_ps))

        # refine intersections with tighter lo/hi bounds
        #d_lohi_map = (z_minmax-altitude)/f_z    
        #x_lo_mapa,x_hi_mapa = utm_x+d_lohi_map*f_x
        #y_lo_mapa,y_hi_mapa = utm_y+d_lohi_map*f_y
                
        d_lo_map,d_hi_map   = (z_minmax-altitude)/f_z    
        x_lo_mapa,y_lo_mapa = utm_x+d_lo_map*f_x, utm_y+d_lo_map*f_y
        x_hi_mapa,y_hi_mapa = utm_x+d_hi_map*f_x, utm_y+d_hi_map*f_y
        
        #map_lohi = utm_xy+tensordot([d_lo_map,d_hi_map],f_xy,0)
        #x_lo_map,y_lo_map,x_hi_map,y_hi_map = map_lohi.ravel()

        ix_loa,iy_loa = map2sl(x_lo_mapa,y_lo_mapa,dem_x0,dem_y0,eff_ps)
        ix_hia,iy_hia = map2sl(x_hi_mapa,y_hi_mapa,dem_x0,dem_y0,eff_ps)

        for j,sj in enumerate(samp_idx):
            ls_idx = ls_offset+j
            if ground_loc[ls_idx,2] < PIX_ERROR_UNDEF:
                # only skip if pixel flagged with error other than PIX_ERROR_UNDEF                
                continue
            
            f_xj,f_yj,f_zj = frame_xyz[:,j]
            ix_hi,iy_hi,ix_lo,iy_lo = ix_hia[j],iy_hia[j],ix_loa[j],iy_loa[j]
            if check_bounds and (outside_bbox(ix_lo*scalef_ps,iy_lo*scalef_ps,dem_ns,dem_nl) or
                                 outside_bbox(ix_hi*scalef_ps,iy_hi*scalef_ps,dem_ns,dem_nl)):
                log('mapped bounds for traced pixel %d (frame=%d, sample=%d) outside DEM extent, skipping'%(ls_idx,frame_idx,sj),1)
                print array([ix_hi, iy_hi, ix_lo, iy_lo])
                print scalef_ps*array([ix_hi, iy_hi, ix_lo, iy_lo])
                print dem_ns/scalef_ps, dem_nl/scalef_ps
                ground_loc[ls_idx,:]   = PIX_ERROR_DEM_EXTENT
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
                    z_diff  = z_trace-(altitude+s_trace/(sqrt(1-(f_zj*f_zj))/f_zj))

                    if (z_diff<0).all(): # apply tolerance theshold if we get a negative intersection
                        if verbose>1:
                            log('z-int (frame=%d, sample=%d)=%f (< z_int_tol)'%(frame_idx,sj, z_diff[-1]))
                        z_diff += z_int_tol
                        
                    idx_int = where(z_diff>PIX_ERROR_UNDEF)[0] # large negative values are error codes
                    n_int   = len(idx_int)
                        
                    if n_int == 0: 
                        log('no positive z-intersection for traced pixel %d (frame=%d, sample=%d), skipping'%(ls_idx,frame_idx,sj),1)
                        print array([ix_hi, iy_hi, ix_lo, iy_lo])
                        print scalef_ps*array([ix_hi, iy_hi, ix_lo, iy_lo])
                        print dem_ns/scalef_ps, dem_nl/scalef_ps
                        print min(abs(z_diff)), scalef_ps, eff_ps, rs_ps, dem_ps, altitude, z_trace, s_trace
                        print xys
                        ground_loc[ls_idx,:]   = PIX_ERROR_DEM_NOZINT
                        continue

                    idx_int = idx_int[0]
                    if n_int == 1 or idx_int == 0:
                        z_int = z_trace[0]
                    elif n_int > 1:
                        z_d,z_tm1 = z_diff[idx_int],z_trace[idx_int-1]
                        z_frac = 1.0-z_d/abs(z_d-z_diff[idx_int-1])
                        z_int  = z_tm1+z_frac*(z_trace[idx_int]-z_tm1)
                   
            else:
                z_int = interpf(dem_utm,ix_lo*scalef_ps,iy_lo*scalef_ps)

            # found intersection, ensure positive, compute (x,y) ground location
            d_int = (z_int-altitude)/f_zj
            ground_x, ground_y = utm_x+d_int*f_xj, utm_y+d_int*f_yj

            ground_loc[ls_idx,:3] = [ground_x, ground_y, z_int]
            if return_lonlat:
                ground_lat,ground_lon = UTMtoLL(dem.datum,ground_y,ground_x,str(utm_zone)+utm_alpha)
                ground_loc[ls_idx,3:] = [ground_lon, ground_lat]
                
            if verbose>1:
                print 'frame,sample,clock: %d %d %d'%(frame_idx,sj,int(frame_clock))
                print 'lat,lon,altitude,utm_x,utm_y: %f %f %f %f %f'%(ground_lat,ground_lon,altitude,utm_x,utm_y)
                print 'frame_xyz: %f %f %f'%(frame_xyz[0,j],frame_xyz[1,j],frame_xyz[2,j])            
                print 'ground_loc: %f %f %f'%(ground_x,ground_y,z_int)

            #except Exception, e:
            #    print 'WARNING: an unexpected error occurred localizing pixel %d (frame=%d, sample=%d):'%(ls_idx,frame_idx,sj),
            #    print e, ', skipping'
            #    pass
            
        if verbose>1:
            print 'completed frame %d\n'%frame_idx

        # move offset past the number of (valid) samples in this frame
        ls_offset += ls_nsamp

    if no_loc > 0:
        print 'WARNING: unable to determine locations for %d of %d frames'%(no_loc,framec)

    return ground_loc        

def update_igm(igmf,igm_nl,igm_ns,bin_factor,zone,hemi):
    '''
    creates header for igm file 
    '''
    if not pathexists(igmf):
        print 'ERROR: IGM file %s does not exist'%igmf
        return FAILURE
    
    description  = 'ANG AIG VSWIR RT-Ortho IGM (easting, northing, elevation)\n'
    description += 'UTM zone %d %s'%(zone,hemi)
    print 'Generating header for IGM file', igmf
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

def rotate_glt(bbox_xy, snap=5.0):
    '''
    computes rotation matrix minimizing *width* of GLT bounding box
    '''
    rotmat = [[1.0,0.0],[0.0,1.0]]
    rotdeg = 0.0
    rotmin = inf
    rotxy  = bbox_xy
    for r in arange(-90,91,snap):
        ar    = DEG2RAD*r
        cosar = cos(ar)
        sinar = sin(ar)
        rar   = [[cosar,-sinar], [sinar,cosar]]
        rxy   = dot(rar,bbox_xy)
        xr,yr = extrema(rxy[0,:]),extrema(rxy[1,:])
        rdiff = abs(xr[1]-xr[0])
        # NOTE (BDB, 09/01/15): code below finds min size (not min width) bbox 
        # rdiff = np_min([xr[1]-xr[0],yr[1]-yr[0]]) 
        if rdiff < rotmin:
            rotmin,rotdeg = rdiff,ar
            rotmat,rotxy  = rar,rxy

    return rotxy, rotmat, rotdeg/DEG2RAD

def setdiff2d(a1,a2):
    a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
    a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
    return setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1])   

@autojit('void(i4[:,:,:],i4[:,:])',nopython=True)
def _infillnn(img,nn_sorted):
    '''
    inplace replacement of zero pixels with their nearest positive neighbor
    img:  M x N array of 2d pixels
    nn_sorted: NN x 2 array of (row,column) neighbor offsets
               (sorted by distance to origin, then ccw angle)
    '''
    R,C,_ = img.shape
    NN,_ = nn_sorted.shape

    for i in range(R):
        for j in range(C): 
            if img[i,j,0]==0:
                # found a pixel to fill, find nearest positive neighbor
                for n in range(NN):
                    ki,lj = i+nn_sorted[n,0],j+nn_sorted[n,1]
                    if ki>=0 and ki<R and lj>=0 and lj<C and img[ki,lj,0]>0: 
                        img[i,j,:] = -img[ki,lj,:]
                        break # first match = nearest, no more searching

@autojit('void(i4[:,:,:],i4[:],i4[:],i4)',nopython=True)
def _erode_contour(img,ictr,jctr,bufrad):
    R,C,_ = img.shape
    for i,j in zip(ictr,jctr):
        imin,imax = max(0,i-bufrad),min(R,i+bufrad)
        jmin,jmax = max(0,j-bufrad),min(C,j+bufrad)
        if img[imin:imax,jmin:jmax,0].max() <= 0:
            # zero all negative values in this window
            img[imin:imax,jmin:jmax,:] = 0
                    
def infillnn(img,winr,winb):
    # construct window for nn search to populate missing pixels
    winr  = int32(winr)
    wind  = 2*winr+1
    winv  = arange(wind)-winr
    nn    = array(map(lambda w: w.ravel(),meshgrid(winv,winv)),dtype=int32).T
    dists = (nn*nn).sum(axis=1)
    wkeep = dists < winr**2
    nn    = nn[wkeep]
    angls = arctan2(nn[:,0],nn[:,1])
    dists = dists[wkeep]
    
    # sort neighbors clockwise starting from 3:00    
    #angls = -((where(angls<0,angls+360,angls)+270) % 360)

    # sort neighbors counterclockwise starting from 12:00
    angls = (where(angls<0,angls+360,angls)+270) % 360

    # get/apply sort indices, removing sidx[0] (offset=[0,0])
    sorta = array(zip(dists,angls*1000),dtype=[('d','i4'),('a','i4')])
    nn = nn[argsort(sorta,order=['d','a'])[1:]]

    # fill missing pixels in-place
    _infillnn(img,int32(nn))
    
    bufrad = max(0,winr-(2+winb))
    if bufrad>0 and bufrad<winr: # values outside this range have no effect
        ictr,jctr = where(find_boundaries(neval('img[:,:,0]!=0'),mode='inner'))
        _erode_contour(img,int32(ictr),int32(jctr),int32(bufrad))
    

def xyz2ps(xyz,verbose=True):
    dt_dist = sqrt(diff(xyz[:,:,0],axis=0)**2)
    at_dist = sqrt(diff(xyz[:,:,1],axis=1)**2)

    dt_ps = ps_avg_fn(dt_dist)
    at_ps = ps_avg_fn(at_dist)

    dt_min,dt_max = extrema(dt_dist)
    at_min,at_max = extrema(at_dist)
    ort_ps        = int(at_ps*10)/10.0

    if ort_ps < MIN_PS:
        warn('increasing ort_ps (%f) to MIN_PS (%f)'%(ort_ps,MIN_PS))
        ort_ps = MIN_PS

    if ort_ps > MAX_PS:
        warn('decreasing ort_ps (%f) to MAX_PS (%f)'%(ort_ps,MAX_PS))
        ort_ps = MAX_PS

    bin_factor  = int(ceil(ort_ps/dt_ps))
    if bin_factor == 0:
        msg  = "estimated bin_factor=0 (at_ps=%f, dt_ps=%f)"%(at_ps,dt_ps)
        msg += ', using default (%d)'%BIN_FACTOR
        warn(msg)
        bin_factor=BIN_FACTOR

    if verbose:
        print 'dt_ps: %8.6f'%dt_ps,'dt_min: %8.6f'%dt_min,'dt_max: %8.6f'%dt_max
        print 'at_ps: %8.6f'%at_ps,'at_min: %8.6f'%at_min,'at_max: %8.6f'%at_max
        print 'ort_ps: %3.1f'%ort_ps,'ps_avg_fn:',ps_avg_fn.func_name
        print 'bin_factor: %d'%bin_factor        
        
    return dict(ort_ps=ort_ps, bin_factor=bin_factor,
                dt_ps=dt_ps, dt_min=dt_min, dt_max=dt_max,
                at_ps=at_ps, at_min=at_min, at_max=at_max)

def update_glt(gltf,igm_xyz,igm_nl,igm_sidx,igm_cidx,ulx,uly,ort_ps,zone,hemi,
               img_sl,bin_factor,nn_rad,nn_buf):
    '''
    update_glt offsets s,l coords to fit within the (xs,ys), (xe,ye) bounding
    box, where the original coordinates are defined with respect to the DEM
    (ulx,uly) and pixel size ps.
    '''
    igm_ns = len(igm_sidx)
    # todo: validate this reshape
    igm_xyz3 = igm_xyz.reshape([igm_nl,-1,3])
    
    eff_ps = ort_ps
    # recompute effective ps using nadir-pointing center samples
    ps_dict = xyz2ps(igm_xyz3[:,igm_cidx,:],verbose=False)
    ps_est = ps_dict['ort_ps']   
    if ps_est > ort_ps: # ort_ps too small, use ps_med instead
        eff_ps = ps_est
        print 'Increasing ort_ps (%3.2f) to %3.2f'%(ort_ps,eff_ps)

    print 'Computing GLT rotation'
    # generate s,l indices for igm, filter out bad pixels
    igm_keep = igm_xyz[:,2]>=0 # z-intersection < 0 -> ort error
    igm_sl = c_[map(ravel,meshgrid(igm_sidx+1,arange(igm_nl)+1))].T
    igm_sl = igm_sl[igm_keep,:]
    
    # select (good) x/y coords at image bounding box to compute rotation
    
    igm_xyz3 = r_[igm_xyz3[0,:,:].squeeze(), igm_xyz3[:,-1,:].squeeze(),
              igm_xyz3[:,0,:].squeeze(), igm_xyz3[-1,:,:].squeeze()]
    igm_xyz3 = igm_xyz3[igm_xyz3[:,2]>=0,:2].T # exclude pixels with error codes
    rigm_xyz3,rotm,rota = rotate_glt(igm_xyz3)    
    
    # rotate (good) pixels if necessary
    if rota != 0.0:        
        rx, ry     = dot(rotm,igm_xyz[igm_keep,:2].T)
        rulx, ruly = dot(rotm,[ulx,uly])
        rotstr     = ', rotation=%10.7f'%(-rota) # NOTE: flip sign of rotation
    else:
        rx, ry     = igm_xyz[igm_keep,:2].T
        rulx, ruly = ulx, uly
        rotstr     = ''
        
    #dx,dy  = diff(rx),diff(ry) # pick pixel size in meters, not from glt_sl
    #ps_dist = sqrt(dx**2 + dy**2 - 2*(dx*dy)) # this considers diagonals
    #ps_dist = sqrt(r_[dx,dy]**2) # this does not
    #ps_med = ps_avg_fn(ps_dist)

        
    # get s,l coords with respect to rotated (DEM) ulx/uly
    print 'GLT rotation angle ',rota        
    glt_sl = asarray(map2sl(rx,ry,rulx,ruly,eff_ps),dtype=int).T    
    # compute bounding box extent in pixel coords
    xs,xe = extrema(glt_sl[:,0])
    ys,ye = extrema(glt_sl[:,1])
    glt_nl,glt_ns = ye-ys+1,xe-xs+1
    if glt_nl*glt_ns > GLT_MAX_NPIX:
        igm_ns = int(len(glt_sl[:,0])/igm_nl)        
        print 'WARNING: GLT size > GLT_MAX_NPIX (%d)!'%GLT_MAX_NPIX
        print 'Processing may be slow!'
        print 'xs: {xs}, ys: {ys}, xe: {xe}, ye: {ye}'.format(**locals())    
        print 'igm_nl: {igm_nl}, igm_ns: {igm_ns}, glt_nl: {glt_nl}, glt_ns: {glt_ns}'.format(**locals())    

    # undo rotation for UL coordinate to get glt_ulx,glt_uly
    rxs,rys               = dot([xs,ys],inv(rotm))
    glt_ulx,glt_uly       = sl2map(rxs,rys,ulx,uly,eff_ps)
    
    # initialize and populate GLT image
    print 'Generating header for GLT file', gltf
    glt_hdrf              = gltf+'.hdr'
    glt_description       = 'ANG AIG VSWIR RT-Ortho GLT (IGM sample, IGM line)'
    glt_map_info          = '{glt_ulx}, {glt_uly}, {eff_ps}, {eff_ps}, {zone}'.format(**locals())
    glt_hdr               = {'lines':glt_nl,'samples':glt_ns,'bands':2}
    glt_hdr['band names'] = "{GLT Sample Lookup, GLT Line Lookup}"
    glt_hdr['data type']  = 3 # 2 = int16, 3 = int32, 12 = uint16, 5 = double
    glt_hdr['map info']   = '{UTM, 1, 1, %s, %s, WGS-84%s}'%(glt_map_info,
                                                             hemi,rotstr)
    
    glt_hdr['header offset']       = 0
    glt_hdr['byte order']          = 0
    glt_hdr['interleave']          = 'bip'
    glt_hdr['line averaging']      = bin_factor
    glt_hdr['raw starting line']   = img_sl
    glt_hdr['raw starting sample'] = 1
    glt_hdr['description']         = glt_description
    # TODO (BDB, 08/20/15): what about nodata values? 

    print 'Writing GLT coordinates with offsets ({xs},{ys}), ({xe},{ye})'.format(**locals())    
    glt_img = envi_create_image(glt_hdrf,glt_hdr,force=True,ext='')
    glt_mm  = glt_img.open_memmap(interleave='source',writable=True)

    # shift glt_sl to fit bbox extent, fill memmap with igm sample/line values
    glt_mm[:,:,:] = 0 
    glt_mm[glt_sl[:,1]-ys,glt_sl[:,0]-xs,:] = igm_sl
    
    print 'Infilling gaps in GLT coordinates'
    starttime  = dtime_now()
    glt_mm = infillnn(glt_mm,nn_rad,nn_buf)
    print 'CPUtime (H:M:S.ms):   %s'%time_elapsed(starttime)
    glt_mm = None # flush to disk
    return SUCCESS

# def map2sun(mapx,mapy,yy,mm,dd,hh,nn,ss,latlon_out,sun_out):
#     sloc = cLocation()
#     sunc = cSunCoordinates()
#     sutc = cTime()
#     sutc.iYear    = yy #int, year
#     sutc.iMonth   = mm #int, month
#     sutc.iDay     = dd #int, day    
#     sutc.dHours   = hh
#     sutc.dMinutes = nn
#     sutc.dSeconds = ss    
#     # solar zenith and azimuth
#     for l in range(mapx.shape[0]):
#         for s in range(mapx.shape[1]):
#             lat,lon = UTMtoLL(datum,mapy[l,s],mapx[l,s],zone_alpha)

#             sloc.dLatitude  = lat
#             sloc.dLongitude = lon

#             #sunc = sunposf(sutc,sloc)
#             sunpos(sutc,sloc,sunc)
#             sun_out[l,s,:] = [sunc.dZenithAngle,sunc.dAzimuth]
#             latlon_out[l,s,:] = [lon,lat]


# def process_chunk(s_offset,s_step,igm_ns,l,mapxl,mapyl,mapzl,datum,zone_alpha,sutc,
#                   obs_mm,loc_mm):
#     sloc = cLocation()
#     sunc = cSunCoordinates()
#     s_max = min(s_offset + s_step, igm_ns)
#     for s in range(s_offset,s_max):
#         lat,lon = UTMtoLL(datum,mapyl[s],mapxl[s],zone_alpha)

#         sloc.dLatitude  = lat
#         sloc.dLongitude = lon

#         sunpos(sutc,sloc,sunc)

#         obs_mm[l,s,[3,4]] = [sunc.dZenithAngle,sunc.dAzimuth]
#         loc_mm[l,s,:] = [lon,lat,mapzl[s]]

def generate_obs_loc(igmf,dem,frame_meta):
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
    print 'Generating OBS and LOC images'
    stime   = dtime_now()
    igm_img = envi_open(igmf+'.hdr',image=igmf)
    igm_mm  = igm_img.open_memmap(interleave='source',writable=False)

    loc_hdrf = igmf.replace('igm','loc')+'.hdr'
    loc_hdr  = igm_img.metadata.copy()
    loc_hdr['description'] = 'ANG AIG VSWIR RT-Ortho LOC'
    loc_hdr['band names'] = '{Longitude (WGS-84), Latitude (WGS-84), Altitude (m)}'
    loc_img = envi_create_image(loc_hdrf,loc_hdr,force=True,ext='')
    loc_mm  = loc_img.open_memmap(interleave='source',writable=True)
    
    obs_hdrf = igmf.replace('igm','obs')+'.hdr'
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
    esd = sun_dist.sun_dist_au(yy, mm, dd)  

    # Earth/sun distance in astronomical units
    obs_mm[:,:,10] = esd

    igm_nl, igm_ns = igm_mm.shape[:2]

    # local aliases for dem
    dem_utm = dem.data_utm
    dem_ps  = dem.ps
    datum   = dem.datum
    ulx,uly = dem.ulx,dem.uly

    dxdy   = ones(dem_utm.shape,dtype=float)*dem_ps # x/y deltas for gradient
    dx, dy = gradient(dem_utm,dxdy,dxdy)
    slope  = 0.5*pi-arctan(hypot(dx,dy)) 
    aspect = arctan2(dx,dy)

    zone_alpha = str(dem.utm_zone)+dem.utm_alpha
    frame_meta = asarray(frame_meta)

    l_step = int(igm_nl/10)
    l_ones = ones([l_step,igm_ns])
    sol_azimuth = zeros([l_step,igm_ns])
    sol_zenith  = zeros([l_step,igm_ns])
    for lj,l_min in enumerate(range(0,igm_nl,l_step)):
        print 'Processing line %i of %i\t%3.f%%'%(l_min,igm_nl,100*lj/10.0)
        l_max  = min(l_min + l_step, igm_nl)
        l_step = min(l_step,l_max-l_min)
        interp_size = [igm_ns,l_step]

        # sensor to ground geometry + deltas
        mapx,mapy,mapz = igm_mm[l_min:l_max,:,:].T
        airx,airy,airz = frame_meta[l_min:l_max,1:4].T
        dx,dy,dz = airx-mapx, airy-mapy, airz-mapz 
        ds = sqrt(dx*dx + dy*dy + dz*dz)
        dcz_sensor = dz/ds
                
        # generate location file
        lat,lon = UTMtoLL(datum,mapy,mapx,zone_alpha)
        loc_mm[l_min:l_max,:,0] = lon.T
        loc_mm[l_min:l_max,:,1] = lat.T
        loc_mm[l_min:l_max,:,2] = mapz.T

        # path length (sensor to ground) in meters)
        obs_mm[l_min:l_max,:,0] = ds.T 

        # to-sensor azimuth (0 to 360 degrees cw from N)
        azimuth_sensor = arctan2(dx,dy)
        azimuth_sensor += (azimuth_sensor<0)*2*pi
        obs_mm[l_min:l_max,:,1] = azimuth_sensor.T / DEG2RAD

        # to-sensor-zenith (0 to 90 degrees from zenith)
        obs_mm[l_min:l_max,:,2] = arccos(dcz_sensor.T) / DEG2RAD

        # solar zenith and azimuth
        #ltime = frame_meta[l_min:l_max,0]
        #sutc.dHours = gps2hour(ltime) #double, hours UTC        
        for l in xrange(l_step):
            for s in xrange(igm_ns):                                
                sloc.dLatitude  = lat[s,l]
                sloc.dLongitude = lon[s,l]
                sunpos(sutc,sloc,sunc)
                sol_zenith[l,s]  = sunc.dZenithAngle
                sol_azimuth[l,s] = sunc.dAzimuth

        sol_zenith = sol_zenith[:l_step,:]
        sol_azimuth = sol_azimuth[:l_step,:]
        obs_mm[l_min:l_max,:,3] = sol_zenith
        obs_mm[l_min:l_max,:,4] = sol_azimuth

        # solar phase 
        sin_zenith = sin(sol_zenith*DEG2RAD)
        dcx_sun = sin(sol_azimuth*DEG2RAD) * sin_zenith
        dcy_sun = cos(sol_azimuth*DEG2RAD) * sin_zenith
        dcz_sun = cos(sol_zenith*DEG2RAD) 
        obs_mm[l_min:l_max,:,5] = arccos((dx.T*dcx_sun + dy.T*dcy_sun)/ds.T \
                                         + dcz_sensor.T*dcz_sun) / DEG2RAD 

        # get slope, aspect by mapping mapx,mapy into DEM coordinate frame
        dem_xf,dem_yf = map2sl(mapx.ravel(),mapy.ravel(),ulx,uly,dem_ps)
        slope_interp  = bilerp(slope,dem_xf,dem_yf).reshape(interp_size).T
        aspect_interp = bilerp(aspect,dem_xf,dem_yf).reshape(interp_size).T
        obs_mm[l_min:l_max,:,6] = slope_interp / DEG2RAD
        obs_mm[l_min:l_max,:,7] = aspect_interp / DEG2RAD
        
        # cosine of sun relative to surface normal
        obs_mm[l_min:l_max,:,8] = sin(slope_interp)*(sin(aspect_interp)*dcx_sun \
                                                     + cos(aspect_interp)*dcy_sun) \
                                  + cos(slope_interp)*dcz_sun        

        # UTC time equal for each sample in the frame
        obs_mm[l_min:l_max,:,9] = dot(frame_meta[l_min:l_max,0],
                                      l_ones[:l_step,:])
    print 'Processing line %i of %i\t%3.f%%'%(igm_nl,igm_nl,100.)        

    obs_mm = None
    igm_mm = None
    print 'CPUtime (H:M:S.ms):   %s'%time_elapsed(stime)
    
# default function aliases
if use_numba:
    find_pixel_trace = find_pixel_trace_numba
elif use_cython:
    find_pixel_trace = find_pixel_trace_cython
elif use_numpy:
    find_pixel_trace = find_pixel_trace_numpy
    bilerp = bilerp_numpy
else: # use (mostly) pure python function
    find_pixel_trace = find_pixel_trace_python
