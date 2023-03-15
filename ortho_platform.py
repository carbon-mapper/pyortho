from __future__ import division, print_function

from ortho_util import *

def valid_platforms():
    from glob import glob
    platform_files = glob(pathjoin(PLATFORM_ROOT,'*.json'))
    platforms = {}
    for pf in platform_files:
        pfpath,pfid = pathsplit(pf.replace('.json',''))
        platforms[pfid.upper()] = load_platform(pfid)
    return platforms

def identify_platform(imgf):
    """
    identify_platform(imgf)
    Use filename of imgf to infer sensor platform id
    
    Arguments:
    - imgf: image filename
    
    Keyword Arguments:
    None
    
    Returns:
    platform_id
    
    Examples:
    >>> identify_platform('ang20140612t204858_raw')
    'AVIRIS-NG'
    >>> identify_platform('ang20170616t184842_raw')
    'AVIRIS-NG_y17'
    >>> identify_platform('prm20151026t173213_raw')
    'PRISM'
    """
    platforms = valid_platforms()
    fbase = pathsplit(imgf)[1]
    p_len = 0
    for p_id in platforms:
        prefixes = platforms[p_id].prefixes
        if len(prefixes)==0:
            warn('WARNING: no filename prefixes provided for platform "%s", cannot identify from file'%p_id)
            break 
        for p_val in prefixes:
            # choose longest matching prefix
            p_val_len = len(p_val)
            if fbase.startswith(p_val) and p_val_len>=p_len:
                p_match = p_id
                p_len = p_val_len

    return p_match if p_len > 0 else 'UNKNOWN'

def load_platform(platform_id,imgf=None,camf=None):
    if platform_id is None:
        # guess the platform by filename prefix if unspecified    
        platform_id = identify_platform(imgf)
        if platform_id == 'UNKNOWN':
            return
    platform_id = platform_id.upper()
    platform_json = pathjoin(PLATFORM_ROOT,platform_id+'.json')
    with open(platform_json,'r') as fid:
        parms = json.load(fid)
        platform_id = parms.pop('platform_id').upper()

        # update camera path if necessary
        camera_file = parms.get('camera_file') if camf is None \
                      else realpath(camf)

        # replace environment variables
        camera_file = expandvars(camera_file)

        if not isabs(camera_file):
            parms['camera_file'] = realpath(pathjoin(PYORT_ROOT,camera_file))
        else:
            parms['camera_file'] = camera_file

        if 'shutter_offset' not in parms.keys():
            parms['shutter_offset'] = 0
            
        # init/validate platform
        platform = PLATFORM(platform_id,parms)
        if imgf is not None and platform.checkfileprefix(imgf) == FAILURE:
            warn('WARNING: unexpected filename prefix for platform %s'%platform.platform_id)
    return platform

def load_camera_model(camf):
    camf_base,camf_ext = splitext(camf)
    if camf_ext.endswith('.txt'):
        from numpy import loadtxt
        camera_model = loadtxt(camf)[:,3:6]
    else:
        with open(camf,"rb") as cam_fid:
            camera_model = fromfile(cam_fid,dtype='<d').reshape([-1,3])
    return camera_model.T

class PLATFORM():
    """
    PLATFORM

    Sensor platform constants and metadata for frame parsing

    Keyword Arguments:
    - platform_id: either 'AVIRIS-NG' or 'PRISM' (default='AVIRIS-NG') 
    
    """
    def __init__(self, platform_id, platform_vars):
        self.platform_id = platform_id        
        self.prefixes       = platform_vars['filename_prefix']
        self.NS             = platform_vars['NS']
        self.NC             = platform_vars['NC']
        self.PPS_MSG        = platform_vars['PPS_MSG']                
        self.PB_OFF         = platform_vars['PB_OFF']        
        self.NFRAME         = self.NS*self.NC
        self.NRAW           = self.NFRAME+self.NS # (1 extra metadata channel)
        self.camf           = platform_vars['camera_file']
        self.shutter_offset = platform_vars['shutter_offset']

    def __str__(self):
        outstr = self.platform_id
        outstr += '\ncamera=%s'%self.camf
        outstr += '\nNC=%d, NS=%d, NFRAME=%d, NRAW=%d'%(self.NC,self.NS,
                                                        self.NFRAME,self.NRAW)
        outstr += '\nPPS_MSG=%d, PB_OFF=%d'%(self.PPS_MSG,self.PB_OFF)
        return outstr
        
    def checkfileprefix(self, fpath):
        if self.platform_id == 'unknown':
            return FAILURE

        _,fname = pathsplit(fpath)
        for p in self.prefixes:
            if fname.startswith(p):
                return SUCCESS

        print('ERROR: invalid filename prefix for platform %s: %s'%(self.platform_id, fname) )
        return FAILURE

    def loadcamera(self,camf=None,doplot=False):
        camf = camf or self.camf
        if not pathexists(camf):
            warn('camera file %s not found!'%camf)
            return FAILURE

        try:
            self.camera_model = load_camera_model(camf)
            self.pb_len       = int(self.camera_model.shape[1])
            self.pb_off       = self.PB_OFF
            self.pb_cen       = int(self.pb_len/2)
        except Exception as e:
            warn('unable to load camera file "%s"'%camf)
            return FAILURE

        if doplot:
            plot_camera(self.camera_model)
        return SUCCESS

if __name__ == '__main__':
    import numpy as np
    import pylab as pl
    from mpl_toolkits.mplot3d import Axes3D
    platform_id='PRISM'
    platform = load_platform(platform_id)
    if platform is None:
        warn('platform unspecified and cannot be identified from file "%s"'%rawf)
        sys.exit(FAILURE)

    prm_camf_old = 'camera_model/vswir_prism_606s_camera_cal3_2014_02_20_18_04_11'
    prm_camf_up1 = 'camera_model/updated_cam_model.prm20170524t174419.prm20170524t175501.moreGCPs.txt'
    prm_camf_up2 = 'camera_model/updated_cam_model.prm20170524t174419.prm20170524t175501.moreGCPs.pb_off18.txt'    

    cam_old = load_camera_model(prm_camf_old)
    cam_up1 = load_camera_model(prm_camf_up1)
    cam_up2 = load_camera_model(prm_camf_up2)

    fig = pl.figure()
    ax3 = fig.add_subplot(111, projection='3d')
    plot_camera(cam_old,fig=fig,ax=ax3,c='k')
    plot_camera(cam_up1,fig=fig,ax=ax3,c='b')
    plot_camera(cam_up2,fig=fig,ax=ax3,c='r')
    pl.show()
    
    platform.loadcamera(prm_camf_old)
    print(platform.camera_model.shape)
    print(platform.camera_model.shape)

    plot_camera(platform.camera_model,reversed=False)
    pl.title('PRISM')

    plot_camera(newmodel,reversed=False)
    pl.title('PRISM (new)')


    pl.show()
    platform_id='AVIRIS-NG'
    platform = load_platform(platform_id)    
    er2_camf = 'camera_model/avng_er2_2017_camera_cal3_2017_03_23_15_19_17'
    platform.loadcamera(er2_camf)
    plot_camera(platform.camera_model,reversed=True)
    pl.title('AVIRIS-NG ER-2 (rotated)')

    er2_to_b200 = rotate_camera(platform.camera_model,180.0)
    plot_camera(er2_to_b200,reversed=False)
    pl.title('AVIRIS-NG ER-2')

    b200_camf = 'camera_model/avng_b200_2017_camera_cal3_2017_03_23_15_19_17'
    save_b200=False
    if save_b200:
        with open(b200_camf,'w') as cam_fid:
            er2_to_b200.T.tofile(cam_fid,format='<d')
            print('wrote',b200_camf)

        with open(b200_camf,'r') as cam_fid:
            loaded_model = fromfile(cam_fid,dtype='<d').reshape([-1,3]).T
            print(((er2_to_b200-loaded_model)**2).sum())
        raw_input()        
 
    twinotter_camf = 'camera_model/avng_2014_camera_cal3_2014_06_20_14_26_06'
    platform.loadcamera(twinotter_camf)
    plot_camera(platform.camera_model,reversed=False)
    pl.title('AVIRIS-NG Twin Otter')
    
    plot_camera(rotate_camera(platform.camera_model,180.0),reversed=True)
    pl.title('AVIRIS-NG Twin Otter (rotated)')    

    pl.show()
