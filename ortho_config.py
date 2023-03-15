from __future__ import print_function
from ortho_util import *
from collections import OrderedDict
from numpy import genfromtxt, mean, median, uint8, bool8, int64, float64

func = lambda args: args

MAX_NL = inf
params = [ # (format = ('name', 'dtype', 'valid values (optional)')
    # general params
    ('verbose'                  , uint8,	(0,1,2,3)),
    ('return_lonlat'            , bool8,	()),
    ('table_updates'            , str,          ('reload','append')),
    ('check_dem_bounds'         , bool8,	()),
    ('check_table_size'         , bool8,	()),
    ('interp_geoid'             , bool8,	()),
    ('read_hdr_nl'              , bool8,	()),

    # DEM params
    ('subset_width'             , float64,	()),
    ('allow_empty_dem'          , bool8,	()),
    ('ps_avg_fn'                , func,         (mean,median)),
    
    # DT averaging/pixel size
    ('DT_CHUNKS'                , int64,	()),
    ('BIN_DELTA'                , int64,	()),
    ('BIN_FACTOR'               , int64,	()),
    ('ORT_PS'                   , float64,	()),
    ('MIN_PS'                   , float64,	()),
    ('MAX_PS'                   , float64,	()),
    
    # Buffer / IO
    ('IMG_SL'                   , int64,	()),
    ('BUF_NL'                   , int64,	()),
    ('MAX_NL'                   , int64,	()),

    ('EMPTY_MAX'                , int64,	()),
    ('EMPTY_RETRY'              , int64,	()),
    ('WAIT_DATA'                , float64,      ()),
    ('WAIT_GPS'                 , float64,      ()),
    ('WAIT_USER'                , float64,      ()),

    # interpolation
    ('S_STEP'                   , int64,	()),
    ('POLY_DEG'                 , int64,	()),
    ('NN_RAD'                   , int64,	()),
    ('NN_BUF'                   , int64,	()),
    
    # pixel size estimation
    ('ALT_DELTA'                , float64,	()),
    ('GLT_ROT_SNAP'             , float64,      ()),
]


class CONFIG(OrderedDict):
    def __init__(self,configf,**kwargs):        
        super(CONFIG,self).__init__()
        
        self['configf'] = configf 

        self.dtype = {}
        self.dvals = {}
        for parm,dtype,dvals in params:
            self[parm] = None
            self.dtype[parm] = dtype
            self.dvals[parm] = dvals
            
        self.parse()
            
    def __repr__(self):
        return '\n'.join(['%s : %s'%(key,str(self[key]))
                          for key in self])

    def parse(self):        
        with open(self['configf'],'r') as fid:
            input_parms = genfromtxt(fid,delimiter=':',unpack=False,dtype=str,
                                      autostrip=True)
            
            for parm,value in input_parms:
                if parm not in self:
                    raise Exception('unrecognized parameter %s'%parm)

                value = value.lower()
                dtype = self.dtype[parm]
                if value in set(["true","false","t","f"]):
                    value = (value[0] == "t")
                elif value == 'nan':
                    value = nan
                elif value == 'inf':
                    value = inf
                elif value == 'mean':
                    value = mean
                elif value == 'median':
                    value = median
                elif dtype == int: # parse as float first to avoid warning
                    value = int64(float64(value))
                elif dtype == float:
                    value = float64(value)
                else:
                    value = dtype(value)

                dvals = self.dvals[parm]
                if len(dvals) != 0 and value not in dvals:
                    raise Exception('unrecognized value %s for %s'%(value,parm))
                
                self[parm] = value

            for parm,_,_ in params:
                if self[parm] is None:
                    raise Exception('required parameter %s undefined'%parm)
                
                
            MAX_NL = self['MAX_NL']
            if self['BUF_NL'] > MAX_NL:
                warn('reduced %s (%d) to MAX_NL (%d)'%('BUF_NL',self['BUF_NL'],MAX_NL))
                self[parm] = MAX_NL
                
if __name__ == '__main__':
    c = CONFIG('./config/pyorthorc.offline')
    print(c)
