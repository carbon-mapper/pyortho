from __future__ import print_function

from ortho_util import *

from os.path import expanduser

validf = 'ortho_validation.xlsx'
plot_gcps=True

def parse_validation_data(xls,**kwargs):
    from pandas import read_excel
    df = read_excel(xls,header=0)
    df.dropna(subset = ['Sample, Line'])
    df = df.ffill()

def test_igm(test_dir):
    
    # AVIRIS-NG test flightlines
    igm_avng_ucr = [
        #pathjoin(test_dir,'test_ucr','orig','ang20140612t204858_ort_igm_orig'),
        pathjoin(test_dir,'test_ucr','ang20140612t204858_rdn_igm'),
        pathjoin(test_dir,'..','ang20140612t204858_rdn_igm'), # ucr_latest
    ]
    igm_avng_4c = [        
        pathjoin(test_dir,'test_4c','orig','ang20150420t182808_ort_igm'),        
        pathjoin(test_dir,'test_4c','0916','ang20150420t182808_rdn_igm'),
        pathjoin(test_dir,'..','ang20150420t182808_rdn_igm') # 4c_latest
    ]
    igm_avng=igm_avng_ucr+igm_avng_4c


    # PRISM test flightlines
    igm_prism_gbr = [
        pathjoin(test_dir,'test_gbr','prm20160909t023104_rdn_igm'),
        pathjoin(test_dir,'test_gbr','prm20160909t022648_rdn_igm'),
        pathjoin(test_dir,'test_gbr','prm20160909t011917_rdn_igm'),
        # also compare to Boardman et al. IDL output
        pathjoin(test_dir,'test_gbr','prm20160909t011917_idl_ortho',
                 'prm20160909t011917_ort_igm'),
    ]
    igm_prism_cai = [
        pathjoin(test_dir,'test_cai','prm20160910t000701_rdn_igm'),
        pathjoin(test_dir,'test_cai','prm20160910t001136_rdn_igm'),
        pathjoin(test_dir,'test_cai','prm20160910t001706_rdn_igm'),
    ]
    igm_prism_liz = [    
        pathjoin(test_dir,'test_liz','prm20160908t231526_rdn_igm'),
    ]    
    igm_prism_hi = [    
        pathjoin(test_dir,'test_hi_orcas','prm20160620t012318_rdn_igm'),
        pathjoin(test_dir,'test_hi_orcas','prm20160622t014223_rdn_igm'),
        pathjoin(test_dir,'test_hi_orcas','prm20160622t015132_rdn_igm'),
        pathjoin(test_dir,'test_hi_orcas','prm20160623t020816_rdn_igm'),
    ]
    igm_prism_clu = [
        # compare local output to output on cluster
        pathjoin(test_dir,'test_clu','prm20160908t231526_rdn_v1p1_igm'),
        pathjoin(test_dir,'test_clu','prm20160909t011917_rdn_v1p1_igm'),
    ]

    igm_prism = igm_prism_gbr + igm_prism_cai + igm_prism_liz + igm_prism_hi + \
                igm_prism_clu

    igm_files = igm_avng+igm_prism
    
    # NOTE (BDB, 09/14/16): coords below computed according to:
    #   sample = raw_col (1-indexed)
    #   line = raw_row (1-indexed)

    # AVIRIS-NG
    coords_ucr = {(  1,  1004):(470862., 3758620.),
                  (  1, 11096):(467335., 3758664.),
                  ( 58,  6467):(468947., 3758596.),
                  (478,  5132):(469487., 3758216.),
                  (501,  5527):(469361., 3758199.),
                  (598,  1004):(470975., 3758105.),
                  (598, 11096):(467365., 3758076.)
    }
    coords_4c =  {(  1,  30451):(739322., 4073294.),                 
                  (132,   10022):(728819., 4076223.),
                  (239,   10098):(728879., 4076336.),
                  (275,  20219):(734036., 4075005.),
                  (598,   1009):(724633., 4077910.)
    }

    # PRISM
    coords_gbr1 = {(184,   3204):(367594., 8134191.),
                   (189,   3238):(367601., 8134177.),
                   (228,   4102):(367770., 8133794.),
                   (368,   3042):(367636., 8134301.),
                   (420,   3209):(367683., 8134235.), #=gbr2
                   (465,   5526):(368183., 8133217.),
    }
    coords_gbr2 = {(43,    4756):(367683., 8134235.), #=gbr1
                   (83,    4818):(367701., 8134213.), #=gbr1
    }
    coords_gbr3 = {#(633,   2690):(340851.042577, 8360499.4343),
                   (462,   10522):(340851., 8360499.),
                   (442,   10928):(340551., 8360234.), #(-14.827429, 145.518174)
                   #(463,  8845):(340740., 8360471.), #(-14.825284, 145.519955)
    }

    # cairns airport
    coords_cair1 = {(100, 7751):(367640., 8134264.),
                    (78,  7735):(367628., 8134266.)}
    
    coords_cair2 = {(275, 1919):(367640., 8134264.),
                    (247, 1903):(367628., 8134266.)}
    
    coords_cair3 = {(80, 5845):(367640., 8134264.),
                    (62, 5825):(367628., 8134266.)}

    # lizard island, gbr, au
    coords_liz1 = {(489,37468): (333834.0, 8376997.0),
                   (495,37466): (333798.0, 8377033.0)}

    # hawaii: honolulu airport 
    coords_hi1 = {(498, 5257): (611489.0, 2357174.0),
                  (567, 5744): (611693.0, 2357364.0)}
    coords_hi2 = {(446, 15722): (611489.0, 2357174.0),
                  (479, 16223): (611694.0, 2357363.0)}

    # hawaii: molokai 8_3
    coords_hi3 = {(93,6192): (723317.0, 2329185.0),
                  (165,17355): (713132.0, 2330703.0)}

    # hawaii: Kaneohe Bay 1_2
    coords_hi4 = {(119, 13387):(632660.0, 2366953.0), #(21.399948, -157.720191)                  
                  (198, 20357):(628330.0, 2373040.0),} #(21.455253, -157.761486)                  


    # palau 1:
    coords_pal1 = {():(458766.0, 853251.0), #(7.71900278,134.62607500),
                   ():(457601.0, 853111.0), #(7.71772222,134.61551667),
                   ():(441566.0, 834216.0)} #(7.54665417,134.47032048)

    # dummy zones for north/south hemisphere
    utm_north = 'N' 
    utm_south = 'M'
    
    test_params = {
        'ang20150420t182808':dict(location='4C1',rawsl2mapxy=coords_4c,
                                  utm_alpha=utm_north),
        
        'ang20140612t204858':dict(location='UCR1',rawsl2mapxy=coords_ucr,
                                  utm_alpha=utm_north),
        
        'prm20160909t023104':dict(location='GBR1',rawsl2mapxy=coords_gbr1,
                                  utm_alpha='M'),
        'prm20160909t022648':dict(location='GBR2',rawsl2mapxy=coords_gbr2,
                                  utm_alpha=utm_south),         
        'prm20160909t011917':dict(location='GBR3',rawsl2mapxy=coords_gbr3,
                                  utm_alpha=utm_south),
        
        'prm20160910t000701':dict(location='CAI1',rawsl2mapxy=coords_cair1,
                                  utm_alpha=utm_south),
        'prm20160910t001136':dict(location='CAI2',rawsl2mapxy=coords_cair2,
                                  utm_alpha=utm_south),
        'prm20160910t001706':dict(location='CAI3',rawsl2mapxy=coords_cair3,
                                  utm_alpha=utm_south),
        
        'prm20160908t231526':dict(location='LIZ1',rawsl2mapxy=coords_liz1,
                                  utm_alpha=utm_south,high_alt=True),
        
        'prm20160622t014223':dict(location='HI1',rawsl2mapxy=coords_hi1,
                                  utm_alpha=utm_north),
        'prm20160622t015132':dict(location='HI2',rawsl2mapxy=coords_hi2,
                                  utm_alpha=utm_north),        
        'prm20160623t020816':dict(location='HI3',rawsl2mapxy=coords_hi3,
                                  utm_alpha=utm_north,high_alt=True),
        'prm20160620t012318':dict(location='HI4',rawsl2mapxy=coords_hi4,
                                  utm_alpha=utm_north,high_alt=True),

        #'prm20170510t042716':dict(location='PAL1',rawsl2mapxy=coords_pal1,
        #                          utm_alpha=utm_north,high_alt=True)
    }
    
    test_locs = ['4C','UCR','GBR','CAI','LIZ','HI']
    #test_locs = ['GBR','CAI','LIZ','HI']
    #test_locs = ['HI']

    test_locs = set(test_locs)    
    retcode = SUCCESS
    glt_totxy_mse = 0
    glt_totpix_mse = 0
    igm_totxy_mse = 0
    igm_totpix_mse = 0
    for igmf in igm_files:
        igm_dir,igm_file = pathsplit(igmf)
        file_base = igm_file.split('_')[0]
        if file_base not in test_params:
            print('skipping',file_base)
            continue
        file_params = test_params[file_base]
        locstr = file_params['location']
        location,locidx = locstr[:-1],locstr[-1]

        if location not in test_locs:
            print("Location %s not in test_locs, skipping"%location)
            continue

        rawsl2mapxy = file_params['rawsl2mapxy']
        utm_alpha = file_params['utm_alpha']
        high_alt = file_params.get('high_alt',False)

        # allow for greater error for high altitude flightlines
        xythr = 15.0 if high_alt else 5.0
        
        igm_hdrf = igmf+'.hdr'
        if not pathexists(igmf):
            warn('IGM file %s not found, skipping'%igmf)
            continue
        igm  = envi_open(igm_hdrf,igmf)
        if igm is None:
            warn('unable to read %s'%igmf)
            continue

        gltf = igmf.replace('igm','glt')
        glt_hdrf = gltf+'.hdr'
        if not pathexists(gltf):
            warn('GLT file %s not found'%gltf)
            continue
        glt  = envi_open(glt_hdrf,gltf)
        if glt is None:
            warn('unable to read %s'%gltf)
            continue        
        
        bin_factor = igm.metadata.get('line averaging',None)
        bin_factor = bin_factor or igm.metadata.get('bin factor',None)
        if not bin_factor:
            warn('bin_factor undefined in IGM file %s'%igm_hdrf)
            continue
        bin_factor = double(bin_factor)
        
        description = igm.metadata.get('description','')
        zstr_idx = description.find('zone')
        if zstr_idx == -1:
            warn('utm zone undefined in IGM file %s'%igm_hdrf)
            continue            
        utm_zone = int(description[zstr_idx:].split()[1])
            
        igm_mm = igm.open_memmap(writable=False)
        print('igm_file: "%s"'%str((igmf)))
        print('dims:',igm_mm.shape)
        glt_mm = glt.open_memmap(writable=False)
        print('glt_file: "%s"'%str((gltf)))
        print('dims:',glt_mm.shape)

        glt_dir,glt_file = pathsplit(gltf)
        glt_meta = glt.metadata
        glt_map = glt_meta['map info']
        glt_ulx,glt_uly,glt_ps = map(float,glt_map[3:6])
        glt_rot = float(glt_map[-1].split('=')[1])
        raw_sl = int(glt_meta.get('raw starting line','1'))
        raw_ss = int(glt_meta.get('raw starting sample','1'))

        print('igm_file: %s'%igm_file,
              'location: "%s":'%location,'index: %s'%locidx,
              '\n -> ps: %10.6f,'%glt_ps, 'bin_factor: "%s",'%str((bin_factor)),
              'high altitude: %s,'%str(high_alt),'rotation: %10.6f'%glt_rot,
              '\n -> utm_zone: "%s,"'%str((utm_zone)),
              'ulx,uly: %10.6f, %10.6f'%(glt_ulx,glt_uly))
                      
        ns,nl = int(glt_meta['samples']),int(glt_meta['lines'])
        bbox_s = [0, ns-1, ns-1, 0]
        bbox_l = [nl-1, nl-1, 0, 0]
        for s1,l1 in zip(bbox_s,bbox_l):            
            gltx,glty = sl2map(s1,l1,glt_ulx,glt_uly,glt_ps)
            gltxr,gltyr = rotxy(gltx,glty,glt_rot,glt_ulx,glt_uly)
            print((s1,l1),'->',(gltx,glty),'->',(gltxr,gltyr))
            #gltxr,gltyr = sl2map_rot(s,l,glt_ulx,glt_uly,glt_ps,rot=glt_rot)
            #print((s,l),'->',(gltx,glty),'->',(gltxr,gltyr))
            #raw_input()
            

        npts = len(rawsl2mapxy)
        igmxydiffmse = 0
        gltxydiffmse = 0
        # compute gcp error
        for (s1,l1) in sorted(rawsl2mapxy):
            (mapx,mapy) = rawsl2mapxy[(s1,l1)]
            lon,lat = utm2lonlat(mapy,mapx,utm_zone,utm_alpha)
            #s1,l1 = s1+1,l1+1
            igms = s1 # -(raw_ss-1) # note: igm already shifted, no need to offset!
            igml = int(ceil((l1-(raw_sl-1))/bin_factor))
            print((s1,l1),(igms,igml))
            gltls = where((glt_mm[:,:,0]==igms) & \
                          (glt_mm[:,:,1]==igml))
            if len(gltls[0]) == 0:
                print('coordinate %d,%d not found in glt'%(igms,igml))
                input()
            gltl,glts = gltls
            if len(glts) != 1:
                glts,gltl = [(mean(glts))],[(mean(gltl))]
            glts,gltl = int(glts[0]),int(gltl[0])
            gltx,glty = sl2map(glts,gltl,glt_ulx,glt_uly,glt_ps)
            gltx,glty = rotxy(gltx,glty,glt_rot,glt_ulx,glt_uly)
            gltx,glty = round(gltx),round(glty)
                        
            igmx,igmy = igm_mm[igml-1,igms-1,0],igm_mm[igml-1,igms-1,1]
            igmx,igmy = round(igmx),round(igmy)
            igmlon,igmlat = utm2lonlat(igmy,igmx,utm_zone,utm_alpha)

            lldiff = max(abs(array([igmlat-lat,igmlon-lon])))
            igmxydiffmax= max(abs(array([mapx-igmx,mapy-igmy])))
            gltxydiffmax = max(abs(array([mapx-gltx,mapy-glty])))
            igmxydiffmean= mean(abs(array([mapx-igmx,mapy-igmy])))
            gltxydiffmean = mean(abs(array([mapx-gltx,mapy-glty])))
            gltxydiffmse += gltxydiffmean
            igmxydiffmse += igmxydiffmean
            #print('-> (mapx,mapy):',(mapx,mapy),'-> (lat,lon):',(lat,lon))
            #print('\tpredicted:',(igmlatp,igmlonp),(igmxp,igmyp))
            #print('\tactual:',(igmlat,igmlon),(igmx,igmy))
            print('raw (ss,sl): (%3d, %5d)'%(raw_ss,raw_sl),
                  'raw (s,l): (%3d, %5d)'%(s1,l1),
                  '-> igm (s,l): (%3d, %5d)'%(igms,igml),
                  '-> glt (s,l): (%3d, %5d)'%(glts,gltl),
                  '\n-> gmapx,gmapy: %10.6f, %10.6f'%(mapx,mapy),
                  '\n-> igmx,igmy: %10.6f, %10.6f'%(igmx,igmy),
                  '\n   mean(|gmap_xy-igm_xy|): %10.9f'%(igmxydiffmean),
                  '\n   max(|gmap_xy-igm_xy|): %10.9f'%(igmxydiffmax),
                  '\n-> gltx,glty: %10.6f, %10.6f'%(gltx,glty),
                  '\n   mean(|gmap_xy-glt_xy|): %10.9f'%(gltxydiffmean),
                  '\n   max(|gmap_xy-glt_xy|): %10.9f'%(gltxydiffmax),
                  '\n-> lat,lon:   %10.6f, %10.6f'%(lat,lon),
                  '\n   max(|gmap_latlon-igm_latlon|): %10.9f'%(lldiff),
                  )

            igmxoff,igmyoff = (igm_mm[:,:,0]-mapx),(igm_mm[:,:,1]-mapy)
            igmxyoff = (igmxoff*igmxoff)+(igmyoff*igmyoff)
            igmr,igmc = where(igmxyoff==igmxyoff.min())
            igmxdiff,igmydiff = igm_mm[igmr,igmc,0]-mapx,igm_mm[igmr,igmc,1]-mapy
            print('-> nearest igm coordinate: (%d,%d)'%(igmc[0],igmr[0]))
            print('-> nearest raw coordinate: (%d,%d)'%(igmc[0]+(raw_ss-1),(igmr[0]*bin_factor)+(raw_sl-1)))
            print('   offset (igmx,igmy): %10.6f, %10.6f'%(igmxdiff,igmydiff))
            print('   offset (igms,igml): %d, %d'%(igmc[0]-igms,igmr[0]-igml))
            
            #assert((abs(igmx-gltx) <= 2*glt_ps) and (abs(igmy-glty) <= 2*glt_ps))
                        
            # map x,y coords should match to within xythr(=5) meters
            #assert(igmxydiff <= xythr)
        igmxy_mse = igmxydiffmse/npts
        gltxy_mse = gltxydiffmse/npts
        print('\n',igm_file)
        print('   mean abs err (m): %.6f'%(igmxy_mse))                  
        print('   mean abs err (ps=%f): %.6f'%(glt_ps,igmxy_mse/glt_ps))                 
        print('#'*65)
        print('\n',glt_file)
        print('   mean abs err (m): %.6f'%(gltxy_mse))                  
        print('   mean abs err (ps=%f): %.6f'%(glt_ps,gltxy_mse/glt_ps))                 
        print('#'*65)        
        print()
        igm_totxy_mse += igmxy_mse
        igm_totpix_mse += (igmxy_mse / npts)
        glt_totxy_mse += gltxy_mse
        glt_totpix_mse += (gltxy_mse / npts)

        if plot_gcps:
            file_raw = file_base+'_raw'
            pngf = pathjoin(igm_dir,file_raw)+'.png'
            print(pngf)
            if pathexists(pngf):
                import pylab as pl
                #from skimage.io import imread
                from scipy.misc import imread
                rawrgb = imread(pngf)
                gcpx,gcpy = [],[]
                rawx,rawy = [],[]
                fig,ax = pl.subplots(1,1,sharex=True,sharey=True)
                ax.imshow(rawrgb,origin='upper')
                
                for (s1,l1) in sorted(rawsl2mapxy):
                    (mapx,mapy) = rawsl2mapxy[(s1,l1)]
                    #s1,l1 = s1+1,l1+1 #+raw_sl
                    gcpx.append(s1)
                    gcpy.append(l1)
 

                    igmxoff,igmyoff = (igm_mm[:,:,0]-mapx),(igm_mm[:,:,1]-mapy)
                    igmxyoff = (igmxoff*igmxoff)+(igmyoff*igmyoff)
                    igmr,igmc = where(igmxyoff==igmxyoff.min())
                    igmx = igm_mm[igmr,igmc,0]
                    igmy = igm_mm[igmr,igmc,1]
                    print((mapx,mapy),(igmx,igmy))
                    raw_r = ((igmr*bin_factor)+(raw_sl-1))
                    raw_c = igmc+(raw_ss-1)
                    rawx.append(raw_c)
                    rawy.append(raw_r)
                    #pl.text(s,l,'%.6f,%.6f  '%(lat,lon),
                    #        horizontalalignment='right',
                    #        verticalalignment='center',fontsize=24)
                lmin,lmax = extrema(gcpy)
                ax.scatter(gcpx,gcpy,marker='o',c='b',s=40,edgecolors='k')
                ax.scatter(rawx,rawy,marker='o',c='r',s=40,edgecolors='k')

                print('gcpx: "%s"'%str((gcpx)))
                print('gcpy: "%s"'%str((gcpy)))
                print('rawx: "%s"'%str((rawx)))
                print('rawy: "%s"'%str((rawy)))
                pl.xlim(min(gcpx+rawx)-10,max(gcpx+rawx)+10)
                pl.ylim(min(gcpy+rawy)-10,max(gcpy+rawy)+10)
                
                ax.set_title(file_raw)
                pl.show()
        
        igm,igm_mm = None,None
        glt,glt_mm = None,None

    print('igm_tot_mse (m):',igm_totxy_mse)
    print('igm_tot_mse (pix):',igm_totpix_mse)
    print('glt_tot_mse (m):',glt_totxy_mse)
    print('glt_tot_mse (pix):',glt_totpix_mse)
    return retcode

def test_params(test_case):
    valid_tests = set(['ucr','ucr_india','4c', 'shndoa', 'burbank',
                       'prism0','prism1','prism2','india021116','india021316',
                       'india021416','india021716'])
    if test_case not in valid_tests:
        warn('invalid test case "%s"'%test_case)
        return ()
    
    offset_latlon = []
    # check hostname to use appropriate directory paths
    from socket import gethostname as hostname
    ngdcs_host = hostname() == 'avirisdev.jpl.nasa.gov'

    test_india = False
    if test_case == 'ucr_india':
        test_case = 'ucr'
        test_india = True        

    dem_file = 'state' # 'conus' # 
    # Test cases!
    if test_case == 'india021316':
        IMG_BASE='ang20160213t055054'
        IMG_DIR='/Volumes/QuantumSpace/Data/AVIRISNG/'        
        #IMG_DIR='/Volumes/Space/Data/AVIRISNG/'

        DEM_SREF = 'Geographic Lat/Lon' #  'UTM' # 
        dem_file = 'india_srtm_1arcsec'
        dem_prefix = '/Volumes/prism/lustre/shared/dem/india_srtm_1arcsec/india_srtm_1arcsec'
        #dem_file = 'world_dem'
        #dem_prefix = '/Volumes/QuantumSpace/Data/dem'
        rawf = pathjoin(IMG_DIR,IMG_BASE,IMG_BASE+'_raw')

    elif test_case == 'india021116':
        IMG_BASE='ang20160211t071427' # 'ang20160210t061239'
        IMG_DIR='/Users/bbue/Research/data/AVIRISNG'
        DEM_SREF = 'Geographic Lat/Lon' #  'UTM' # 
        dem_file = 'india_srtm_1arcsec'
        dem_prefix = '/Volumes/prism/lustre/shared/dem/india_srtm_1arcsec/india_srtm_1arcsec'
        rawf = pathjoin(IMG_DIR,IMG_BASE,IMG_BASE+'_raw')
        
    elif test_case == 'india021716':
        IMG_BASE='ang20160217t080930'
        #IMG_BASE='ang20160217t080145'
        #IMG_DIR='/Volumes/Space/Data/AVIRISNG/ortho_err/'
        IMG_DIR='/Volumes/prism/lustre/ang/y16/raw/'
        DEM_SREF = 'Geographic Lat/Lon' #  'UTM' # 
        dem_file = 'india_srtm_1arcsec'
        dem_prefix = '/Volumes/prism/lustre/shared/dem/india_srtm_1arcsec/india_srtm_1arcsec'
        rawf = pathjoin(IMG_DIR,IMG_BASE+'_raw')

    elif test_case == 'india021416':
        IMG_BASE='ang20160214t112747'
        IMG_DIR='/Volumes/Space/Data/AVIRISNG/ortho_err/'

        DEM_SREF = 'Geographic Lat/Lon' #  'UTM' # 
        dem_file = 'india_srtm_1arcsec'
        dem_prefix = '/Volumes/prism/lustre/shared/dem/india_srtm_1arcsec/india_srtm_1arcsec'
        rawf = pathjoin(IMG_DIR,IMG_BASE+'_raw')         
        
    elif 'prism' in test_case:
        if test_case[-1] == '1':            
            IMG_BASE='prm20151026t173213'
            IMG_DIR='/Volumes/QuantumSpace/Data/PRISM/'+IMG_BASE
        elif test_case[-1] == '2':
            # (very) low altitude error case
            IMG_BASE='prm20160120t192216'
            IMG_DIR='/Volumes/Space/Data/PRISM/'+IMG_BASE
        else:            
            IMG_BASE='PRISM20160115t153511'
            IMG_DIR='/Volumes/QuantumSpace/Data/PRISM/'+IMG_BASE

        DEM_SREF = 'Geographic Lat/Lon' #  'UTM' # 
        dem_file = 'orcas'
        
        rawf = pathjoin(IMG_DIR,IMG_BASE+'_raw')

    elif test_case == 'ucr':
        if ngdcs_host:
            IMG_DIR='/data/UCR'
            dem_prefix = '/home/ngdcs/src/range/data/dem/dem_ca/dem_ca'
        else:
            IMG_DIR='/Volumes/QuantumSpace/Data/AVIRISNG/20140612_ucr'
            #IMG_DIR='/Volumes/TravelSpace/Data/AVIRISNG/20140612_ucr'
            if test_india:
                dem_prefix='/Users/bbue/Desktop'
            else:
                dem_prefix='/Volumes/QuantumSpace/Data/dem'
        rawf = pathjoin(IMG_DIR,'ang20140612t204858_raw')

    elif test_case == '4c':
        if ngdcs_host:
            IMG_DIR='/data/4C'
            dem_prefix = '/home/ngdcs/src/range/data/dem'
        else:
            #IMG_DIR='/Volumes/Space/Data/AVIRISNG/20150420_4c/ang20150420t182808'
            IMG_DIR='/Volumes/QuantumSpace/Data/AVIRISNG/20150420_4c'
            #dem_prefix = '/Volumes/SpaceTravel/Data/dem'
            dem_prefix='/Users/bbue/Research/data/dem'

        rawf = pathjoin(IMG_DIR,'ang20150420t182808_raw')

    elif test_case == 'shndoa':
        if ngdcs_host:
            IMG_DIR='/data/SHNDOA'
            dem_prefix = '/home/ngdcs/src/range/data/dem'
        else:
            IMG_DIR='/Users/bbue/.sshfs/avng.home/data/SHNDOA'
            dem_prefix='/Users/bbue/.sshfs/avng.home/src/range/data/dem/conus_ned_1arcsec/conus_ned_1arcsec_utm'

        rawf = pathjoin(IMG_DIR,'ang20150727t190800_raw')

    elif test_case == 'burbank':
        if ngdcs_host:
            IMG_DIR='/data/burbank'
            dem_prefix = '/home/ngdcs/src/range/data/dem'
        else:
            #IMG_DIR='/Users/bbue/.sshfs/avng.home/data/burbank'
            #IMG_DIR='/Volumes/Space/Data/AVIRISNG/20150914_burbank'
            IMG_DIR='/Volumes/TravelSpace/Data/AVIRISNG/20150917'
            #dem_prefix='/Volumes/Space/Data/dem'
            dem_prefix='/Users/bbue/Research/data/dem'

        #rawf = pathjoin(IMG_DIR,'ang20150907t223238_raw')    
        #rawf = pathjoin(IMG_DIR,'ang20150907t223943_raw')   # error case 
        #rawf = pathjoin(IMG_DIR,'ang20150907t225344_raw')    
        #rawf = pathjoin(IMG_DIR,'ang20150907t230523_raw')    
        #rawf = pathjoin(IMG_DIR,'ang20150907t231622_raw')
        #rawf = pathjoin(IMG_DIR,'ang20150914t183748_raw')    # missing lines
        #rawf = pathjoin(IMG_DIR,'ang20150914t175900_raw')    # missing lines
        rawf = pathjoin(IMG_DIR,'ang20150917t010641_raw')

    # DEM sources for testcases
    if dem_file == 'orcas':
        if DEM_SREF == 'UTM':
            dem_prefix='/Volumes/QuantumSpace/Data/dem/ORCAS_DEM/ORCAS_DEM_float_utm'
        else:
            dem_prefix='/Volumes/QuantumSpace/Data/dem/ORCAS_DEM/ORCAS_DEM_float'

    elif dem_file == 'world_dem':
        dem_prefix='/Volumes/QuantumSpace/Data/dem/world_dem/world_dem'
            
    elif dem_file=='conus':
        if ngdcs_host:
            dem_prefix = '/home/ngdcs/src/range/data/dem'
        else:
            dem_prefix = '/Volumes/Space/Data'
            if not pathexists(dem_prefix):
                dem_prefix = '/Volumes/SpaceTravel/Data/dem'
                if not pathexists(dem_prefix):
                    warn('conus DEM not found, fallback to state DEM')
                    dem_file='state' # use state dem instead

        dem_prefix = pathjoin(dem_prefix,'conus_ned_1arcsec/conus_ned_1arcsec')
        DEM_SREF='Geographic Lat/Lon'

    elif dem_file=='state':
        DEM_SREF = 'Geographic Lat/Lon' # 'UTM'
        rawfile = pathsplit(rawf)[1]
        if test_case == 'ucr':
            if test_india:
                ucr_latlon = 33.974021, -117.328107
                hyderabad_latlon = 17.416504, 78.507726
                gujarat_latlon = 23.260591, 69.665381
                mumbai_latlon = 19.181725, 72.908110
                newdelhi_latlon = 28.653198, 77.232138
                kanpur_latlon = 26.332842, 80.367058
                kolkata_latlon = 22.599558, 88.372507
                chennai_latlon = 13.083345, 80.281796
                andaman_latlon = 12.583750, 92.779903
                nicobar_latlon = 7.029529, 93.726813
                target_latlon = gujarat_latlon # mumbai_latlon # kanpur_latlon # kolkata_latlon # chennai_latlon #hyderabad_latlon
                offset_latlon = array(target_latlon)-array(ucr_latlon)            
                dem_prefix = pathjoin(dem_prefix,'india_srtm_1arcsec/india_srtm_1arcsec_utm')
            else:
                dem_prefix = pathjoin(dem_prefix,'dem_ca/dem_ca')            
        elif rawfile == 'ang20150420t182808_raw':
            dem_prefix = pathjoin(dem_prefix,'dem_4c/dem_4c')

    if DEM_SREF == 'UTM':
        dem_prefix += '_utm'

    return (rawf, dem_prefix, offset_latlon)

if __name__ == '__main__':
    test_dir='~/Research/AVIRISNG/range/watch_out/pyortho-latest/'    
    test_igm(expanduser(test_dir))
   
