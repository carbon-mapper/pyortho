import argparse, doctest
import warnings
from collections import OrderedDict

from ortho_util import *
from ortho_nav import *
from orthorectify import GEOID_FILE


# example report:
# /Volumes/QuantumSpace/Data/AVIRISNG/20140612_ucr/ang20140612t204858_ort_report.txt

__version__ = 'AVNG PyOrtho Ver 1.0 September 22, 2015'

undef_str = '[undefined]'

from datetime import datetime

# list of (key,description) tuples (todo: add datatype string?)
# list used here (rather than dict) to ensure proper ordering of keys in report
key_info = [
    ('root'      ,	'root file name'),
    ('procdate'  ,	'processing date'),
    ('ver'       ,	'software version'),
    ('ystart'    ,	'raw starting sample (1-based)'),
    ('nsh'       ,	'raw starting line (1-based)'),
    ('nlh'       ,	'number of samples'),
    ('nlo'       ,	'number of input frames'),
    ('nout'      ,	'number of output frames'),
    ('binfac'    ,	'frame bin factor used'),
    ('headwrap'  ,	'cmigits heading wraps'),
    ('year'      ,	'mid line year'),
    ('month'     ,	'mid line month'),
    ('day'       ,	'mid line day'),
    ('hour'      ,	'mid line utc hour'),
    ('minute'    ,	'mid line utc min'),
    ('solele'    ,	'mid solar elevation'),
    ('solazi'    ,	'mid solar azimuth'),
    ('meanele'   ,	'mean scene elevation'),
    ('minele'    ,	'min scene elevation'),
    ('maxele'    ,	'max scene elevation'),
    ('meanatps'  ,	'mean across-track nadir pixel size'),
    ('minatps'   ,	'min across-track nadir pixel size'),
    ('maxatps'   ,	'max across-track nadir pixel size'),
    ('meandtps'  ,	'mean down-track nadir pixel size'),
    ('mindtps'   ,	'min down-track nadir pixel size'),
    ('maxdtps'   ,	'max down-track nadir pixel size'),
    ('finalps'   ,	'final rendering pixel size'),
    ('meangeoid' ,	'mean geoid undulation'),
    ('mingeoid'  ,	'min geoid undulation'),
    ('maxgeoid'  ,	'max geoid undulation'),
    ('zone'      ,	'utm zone'),
    ('datum'     ,	'geographic datum'),
    ('gnadj'     ,	'average grid north adjustment'),
    ('rotsnap'   ,	'rotation snap value'),
    ('finalang'  ,	'final rotation value'),
    ('ulx'       ,	'glt upper left easting'),
    ('uly'       ,	'glt upper left northing'),
    ('gltnso'    ,	'glt nso'),
    ('gltnlo'    ,	'glt nlo'),
    ('timebias'  ,	'time bias added to avngdcs times to match gps time')
]

# ancillary keys in the order they appear in metaf=root.meta
meta_keys = ['meangeoid', 'mingeoid', 'maxgeoid', 'gnadj',
             'headwrap', 'timebias', 'rotsnap']

defaults = {
    'ver':      __version__,
    'procdate': datetime.today().strftime("%a %b %d %H:%M:%S %Y"),
    'headwrap': int(0),
    'timebias': 0.0,
    'rotsnap' : 5.0
}

class Report(OrderedDict):
    def __init__(self,igmf,**kwargs):
        doparse = kwargs.pop('parse',True) # automatically parse by default

        super(Report,self).__init__()

        self.path, self.root = pathsplit(igmf.replace('_igm',''))

        self.igmf = igmf
        self.gltf = igmf.replace('igm','glt')
        self.obsf = igmf.replace('igm','obs')        
        self.locf = igmf.replace('igm','loc')        
        self.metaf = pathjoin(self.path,self.root+'.meta')            
        
        self._info  = dict(key_info)
        self._undef = undef_str

        for key,desc in key_info[1:]:
            self[key] = defaults.get(key,self._undef)
            
        if doparse:
            self.parse()
            
    def __repr__(self):
        return '\n'.join(['%s : %s'%(self._info[key],str(self[key]))
                          for key in self])
    
    def defined(self):
        return [key for key in self if self[key] != self._undef]
                
    def undefined(self):
        return [key for key in self if self[key] == self._undef]

    def check_files(self):
        for f in [self.igmf,self.gltf,self.obsf,self.metaf]:
            if not pathexists(f):
                warnings.warn("Unable to read file %s, using default values"%f)
    
    def parse_igm(self):
        igm_img          = envi_open(self.igmf+'.hdr',image=self.igmf)
        igm_mm           = igm_img.open_memmap(interleave='source',writable=False)
        meta,mapinfo     = igm_img.metadata,envi_mapinfo(igm_img)
        utm_x,utm_y,elev = igm_mm[:,:,0],igm_mm[:,:,1],igm_mm[:,:,2]


        mid = int(igm_mm.shape[1]/2+0.5)
        # FIXME: this is the wrong computation to use
        at_dist = sqrt(((diff(c_[utm_x[:],utm_y[:]],axis=0))**2).sum(axis=0))
        dt_dist = sqrt(diff(utm_x[:,mid-1:mid+2],axis=1)**2)
        
        igm_dict = dict(nlo=igm_mm.shape[0], nlh=igm_mm.shape[1],
                        minele=np_min(elev), maxele=np_max(elev),
                        meanele=mean(elev), mindtps=np_min(dt_dist),
                        maxdtps=np_max(dt_dist), minatps=np_min(at_dist),
                        maxatps=np_max(at_dist), meandtps=mean(dt_dist),
                        meanatps=mean(at_dist))

        if self['binfac'] == self._undef:
            igm_dict['binfac'] = meta.get('line averaging',self._undef)

        self.update(**igm_dict)
               
    def parse_glt(self):
        glt_img      = envi_open(self.gltf+'.hdr',image=self.gltf)
        meta,mapinfo = glt_img.metadata,envi_mapinfo(glt_img)
        ystart = meta.get('raw starting sample',1)
        nsh = meta.get('raw starting line',1)
        rotation = mapinfo.get('rotation',0.0)
        glt_dict = dict(ystart=ystart,nsh=nsh,nout=meta['lines'],
                        gltnso=meta['samples'],gltnlo=meta['lines'],
                        ulx=mapinfo['ulx'],uly=mapinfo['uly'],
                        finalps=mapinfo['xps'],zone=mapinfo['zone'],
                        datum=mapinfo['datum'],finalang=rotation)
        if self['binfac'] == self._undef:
            glt_dict['binfac'] = meta.get('line averaging',self._undef)        
        self.update(**glt_dict)
        
    def parse_obs(self):
        obs_img      = envi_open(self.obsf+'.hdr',image=self.obsf)
        
        obs_mm       = obs_img.open_memmap(interleave='source',writable=False)
        meta,mapinfo = obs_img.metadata,envi_mapinfo(obs_img)
        midl,mids    = map(int,[obs_mm.shape[0]/2-1,obs_mm.shape[1]/2-1])

        obsdate      = file2date(self.obsf)
        Y,M,D,H      = obsdate[:4]
        gpshour      = obs_mm[midl,0,9]
        #gpshour      = gps2hour(gpstime) 
        ihour        = int(gpshour)
        minute       = int((gpshour-ihour)*60)
        hour         = ihour
        
        obs_dict = dict(solazi=obs_mm[midl,mids,3],solele=obs_mm[midl,mids,4],
                        year=Y,month=M,day=D,hour=hour,minute=minute)
        self.update(**obs_dict)

    def parse_loc(self):
        geoid,geoid_dps = parse_geoid(GEOID_FILE)

        loc_img         = envi_open(self.locf+'.hdr',image=self.locf)
        loc_mm          = loc_img.open_memmap(interleave='source',writable=False)
        lon,lat         = loc_mm[:,0,0],loc_mm[:,0,1]

        # need utm zone of ul pixel
        _,_,utm_zone,_  = lonlat2utm(lon[0],lat[0])

        gridnorth = sin(DEG2RAD*lat)*(((abs(utm_zone)-31.0)*6.0+3.0)-lon)        
        lon = where(lon<0.0,360.0+lon,lon)
        
        # apply geoid correction
        geoidtrace = compute_geoidtrace(geoid,lon,lat,geoid_dps,interp=True)
        
        loc_dict = dict(mingeoid=geoidtrace.min(),maxgeoid=geoidtrace.max(),
                        meangeoid=geoidtrace.mean(),gnadj=abs(gridnorth.mean()))
        self.update(**loc_dict)
        
    def parse_meta(self):
        if not pathexists(self.metaf):
            self.parse_loc()
        else:
            with open(self.metaf,'r') as fid:
                for key in meta_keys:
                    self[key] = float(fid.readline())
            
    def parse(self):
        self.check_files()
        self.parse_igm()
        self.parse_glt()
        self.parse_obs()
        self.parse_meta()
        return self
    
    def write(self,outfile=None):
        if outfile is None:
            outfile = (self.path.replace("raw","ortho_report") + "/" +
                          self.root+'_ort_report.txt')
        with open(outfile,'w') as fid:        
            fid.write(str(self)+'\n')
            print 'Report saved to %s'%outfile
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-o','--output_path', help='Path to store output products', 
                        type=str, required=False)
    parser.add_argument('-t','--test', help='Enable test mode', 
                        action='store_true', default=False, required=False) 
    
    parser.add_argument('igm', help='Path to IGM file')
    
    args = vars(parser.parse_args())

    if args['test']:
        import doctest    
        sys.exit(doctest.testmod())

    igmf = args['igm']
    igm_dir = pathsplit(igmf)[0]
    out_dir = args['output_path'] or igm_dir

    
    print(Report(igmf))
