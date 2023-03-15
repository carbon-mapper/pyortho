from __future__ import division, print_function

from ortho_util import *

class DEM():
    def __init__(self,demf,init_lonlat,subset_dims,allow_empty_dem=True,
                 verbose=0):
        """
        __init__(demf,init_lonlat,subset_dims,
                 allow_empty_dem=True, verbose=0) 

        Initializes DEM, optionally subsetting a subset_width bounding box

        Arguments:
        - demf: dem file path     
        - init_lonlat: center coordinate of DEM subset
        - subset_dims: either
          - DEM bounding box (half)width in degrees about init lonlat
          - [ullon,ullat,lrlon,lrlat] bbox coords

        Keyword Arguments:
        - allow_empty_dem: allow dem to contain only zero values (default=True)
        - verbose: verbosity level
        
        Returns:
        - data_utm: DEM data (or subset thereof) in square UTM coords
        - meta: associated metadata
        """
        self.initialized = False
        #  TODO (BDB, 04/05/17): move this logic outside of the DEM class 
        if isinstance(subset_dims,list):
            if len(subset_dims)!=4:
                warn('malformed subset dims')
                return
            
            ullon,ullat,lrlon,lrlat = subset_dims
        else: # subset_dims = scalar
            if len(init_lonlat) == 0:
                warn('initial lat/lon undefined')
                return

            subset_width = subset_dims
            init_lon, init_lat = init_lonlat
            # take a subset bbox \pm subset_width centered on (init_{lat,lon})
            
            ullon,ullat = init_lon-subset_width,init_lat+subset_width
            lrlon,lrlat = init_lon+subset_width,init_lat-subset_width

        if lrlon > 180.0:
            lrlon = -360+lrlon
        elif ullon < -180.0:
            ullon = 360+ullon
            
        subset_ll = [ullon,ullat,lrlon,lrlat]        

        sref = get_projection(demf)            
        if sref=='Geographic Lat/Lon': # GDAL necessary to reproject to utm
            data_utm, meta = subset_latlon_reproject(demf,subset_ll,
                                                     verbose=verbose)
            
            save_dem_subset=False
            if save_dem_subset:
                dembase,demext = splitext(pathsplit(demf)[1])
                subf = dembase+'_sub'+demext
                repf = dembase+'_reproj'+demext

                try:                    
                    envi_drv = gdal.GetDriverByName('ENVI')
                    ns_sub = data_utm.shape[1]
                    nl_sub = data_utm.shape[0]
                    sub_ds= envi_drv.Create(subf, ns_sub, nl_sub, 1, gdal.GDT_Float32)
                    sub_ds.SetGeoTransform(meta['geotransform'])
                    sub_ds.SetProjection(meta['projection'])
                    data = float32(gsub.ReadAsArray().squeeze())
                    sub_ds.GetRasterBand(1).WriteArray(data)
                    sub_ds = None
                    print('Saved DEM subset',subf)
                except:
                    print('Unable to save DEM subset',subf)
                    pass
                
        elif sref=='UTM':
            data_utm, meta = subset_utm(demf,subset_ll,verbose=verbose)

            
        if len(meta) == 0:
            warn('unable to load DEM from %s'%demf)
            return

        self.demf        = demf
        self.data_utm    = data_utm
        self.meta        = meta
        self.datum       = DATUM_WGS84

        self.extent      = meta['extent']
        self.extrema     = meta['extrema']
        self.ps          = meta['ps']
        self.utm_zone    = meta['zone']
        self.utm_alpha   = meta['alpha']
        self.utm_hemi    = meta['hemi']
        
        self.lrx         = self.extent[0]
        self.lry         = self.extent[1]
        self.ulx         = self.extent[2]
        self.uly         = self.extent[3]

        self.bbox        = [0,0]+list(meta['dims'])

        #self.init_lon    = init_lon
        #self.init_lat    = init_lat
        
        if self.extrema[1] <= 0.0:
            warn('all DEM elevation values at/below zero or NODATA')
            if self.extrema[0]<=0.0 and not allow_empty_dem:
                return
            
        if verbose:
            print('DEM min: %10.6f, max: %10.6f'%self.extrema)
            
        self.initialized = True

    def map2sl(self,x,y,eff_ps=None):
        eff_ps = eff_ps or self.ps        
        return map2sl(x,y,self.ulx,self.uly,eff_ps)
        
    def sl2map(self,s,l,eff_ps=None):
        eff_ps = eff_ps or self.ps
        return sl2map(s,l,self.ulx,self.uly,eff_ps)
        
    def utm2lonlat(self,utmy,utmx):
        # get lon,lat wrt DEM UTM zone
        return utm2lonlat(utmy,utmx,str(self.utm_zone),self.utm_alpha)

    def lonlat2utm(self,lon,lat):
        # get utmx,utmy for lon,lat with respect ot this utm zone
        return lonlat2utm(lon,lat,zone=self.utm_zone)

    # def outside_bounds(self,xl,xh,yl,yh):
    #     return min(xl,xh)<dem_bbox_ps[0] or max(xl,xh)>=dem_bbox_ps[2] or \
    #         min(yl,yh)<dem_bbox_ps[1] or max(yl,yh)>=dem_bbox_ps[3]
    
    def plot(self,show=True,save=False):
        import pylab as pl

        ll_str = '%8.5f,%8.5f'%(self.init_lon,self.init_lat)
        init_utm = lonlat2utm(self.init_lon,self.init_lat)
        easting, northing, zonenum, zoneletter = init_utm
        init_s, init_l = map2sl(easting,northing,self.ulx,self.uly,self.ps)        
        pl.imshow(self.data_utm)
        pl.scatter([init_s],[init_l],c='r')
        
        pl.text(init_s+1,init_l+1,ll_str,color='white',size=12)
        pl.xticks(int32(linspace(0,self.data_utm.shape[1],10)),
                  int32(linspace(self.ulx,self.lrx,10)),rotation=90)
        pl.yticks(int32(linspace(0,self.data_utm.shape[0],10)),
                  int32(linspace(self.uly,self.lry,10)))
        pl.tight_layout()
        if save:
            outd = pathsplit(self.demf)[0]
            outf = 'dem_subset_%8.5f_%8.5f.png'%(self.init_lon,self.init_lat)
            #outf = pathjoin(outd,outf)
            print('saving',outf)
            pl.savefig(outf)
        if show:
            pl.show()        
