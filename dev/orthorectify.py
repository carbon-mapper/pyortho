#!/usr/bin/env python
import sys, time, datetime, argparse
dtime_now = datetime.datetime.now

plot_rgb      = False
plot_frames   = False

if plot_rgb or plot_frames:
    import pylab as pl
    try:
        # import pretty formatting options if available
        from util.aliases_pylab import *
    except Exception, e:
        pass
    pl.ioff()

from ortho_util import *

# general parameters
verbose       = 1 # verbosity level (max = 2)
return_lonlat = False # compute/return lonlat during realtime processing
table_updates = 'reload' # pps/gps table update mode (reload|append)
check_size    = True # check table size before updating tables

# pixel size/bin factor estimation params
DT_CHUNKS     = 5 # number of buffers to use during downtrack averaging (0 disables and uses defaults)
BIN_DELTA     = 3 # if DT_CHUNKS > 1, ignore bin_factor estimates that differ by more than BIN_DELTA from their previous estimate

# DEM default params
subset_width  = 1     # DEM bounding box half-width (degrees)

# buffer / IO params
IMG_SL        = 800 # number of calibration (dark current) frames to ignore
MAX_NL        = inf # hard limit on the maximum number of raw image frames to process before exiting
BUF_NL        = min(1000,MAX_NL) # lines to use in rolling buffer
PS_NL         = BUF_NL # nlines used to estimate ort_ps,bin_factor
INIT_NL       = BUF_NL # fill buffer with this many frames until we find a science frame

MAX_EMPTY     = 20 # maximum number of empty buffers in a row before querying user to bailing out
EMPTY_RETRY   = 3 # number of attempts to query the user before continuing 
WAIT_DATA     = 1 # sleep time (seconds) while waiting for raw buffer to fill
WAIT_GPS      = 10 # sleep time (seconds) while waiting for gps data
WAIT_USER     = 5 # wait time (seconds) while waiting for user input

# sample interpolation
S_STEP        = 6 # sample step size (1=no interpolation, 6=realtime default)
POLY_DEG      = 3 # polydeg used (for S_STEP>1) to interp line/samp -> mapX/mapY
NN_RAD        = 7 # window size (radius) used for nn infilling of glt gaps (15=envi default)
NN_BUF        = 1 # buffer around glt edges (max=NN_RAD)

#@profile
def compute_igm_glt(rawf,platform,output_path,dem_prefix,max_nl=None,
                    offset_latlon=[]):
    global plot_frames, plot_rgb
    img_dir,img_base = pathsplit(rawf.replace('_raw',''))

    out_dim = 5 if return_lonlat else 3
    max_nl = max_nl or MAX_NL
        
    rgb = []
    if plot_rgb:
        rgbf = None
        irgbf = pathjoin(img_dir,rawf+'_i1_rgb_img')
        krgbf = pathjoin(img_dir,rawf+'_k1_rgb_img')
        if pathexists(irgbf):
            rgbf = irgbf
        elif pathexists(krgbf):
            rgbf = krgbf

        try:
            # note: UCR rgb image already offset by IMG_SL=1003            
            img_rgb  = envi_open(rgbf+'.hdr')
            # rgb_data = img_rgb.open_memmap(interleave='source',writable=False)
            rgb_data = img_rgb.open_memmap(writable=False)
            print 'Loaded RGB image %s'%rgbf
        except Exception, e:
            warn('unable to load rgb image, falling back to raw frames')
            plot_rgb = False
            plot_frames = True
            pass
    
    binf = pathjoin(output_path,img_base+'_raw.binfac')
    igmf = pathjoin(output_path,img_base+'_rdn_igm')
    gltf = pathjoin(output_path,img_base+'_rdn_glt')

    ppsf = pathjoin(img_dir,img_base+'_pps')
    gpsf = pathjoin(img_dir,img_base+'_gps')    
        
    # initialize world, camera and elevation models
    nav  = NAV(platform,ppsf,gpsf)
    if not nav.initialized:
        print 'ERROR: unable to initialize navigation parameters'
        return FAILURE
    
    pb_len  = nav.pb_len # pushbroom length for the above camera
    pb_cen  = nav.pb_cen
    pb_samp = arange(pb_len)

    # local refs to nav data / functions
    platform       = nav.platform
    clock2location = nav.clock2location
    collect_tables = nav.collect_tables
    
    ps_nl = PS_NL
        
    if S_STEP != 1:
        # ensure first/last samples in frame always geolocated
        pb_half = pb_samp[:pb_cen:S_STEP]
        buf_samp = r_[pb_half,(pb_len-pb_half-1)[::-1]]                           
    else:
        buf_samp = pb_samp

    # initialize line/sample indices to geolocate with respect to buffer    
    buf_nl              = INIT_NL # will shrink to (a multiple of) BUF_NL        
    buf_ns              = len(buf_samp)
    buf_lines           = arange(max(INIT_NL,BUF_NL))
    buf_lidx,buf_sidx   = map(ravel,meshgrid(buf_lines,buf_samp))

    # samples to interpolate
    interpolate_samples = buf_ns != pb_len # flag to do interpolation
    intp_samp           = setdiff1d(pb_samp,buf_samp)
    intp_ns             = len(intp_samp)
    intp_lines          = buf_lines.copy()
    intp_lidx,intp_sidx = map(ravel,meshgrid(buf_lines,intp_samp))
    intp_scoef          = c_[intp_samp,intp_samp] # sample coef for interpolation

    # default ps/bin_factors
    ort_ps           = ORT_PS
    bin_factor       = BIN_FACTOR
    binned_nl        = BUF_NL   

    est_ort          = zeros(DT_CHUNKS)
    est_dt           = zeros(DT_CHUNKS)
    est_factor       = zeros(DT_CHUNKS,dtype=int)
    
    # init downtrack and across track pixel size, compute from 3 center samples
    dt_width         = 1
    dt_samp          = arange(pb_cen-dt_width,pb_cen+dt_width+1)
    dt_lidx,dt_sidx  = map(ravel,meshgrid(buf_lines,dt_samp))
    dt_ns            = len(dt_samp)
    dt_ps,at_ps      = ORT_PS, ORT_PS
    dt_min,dt_max    = ORT_PS, ORT_PS
    at_min,at_max    = ORT_PS, ORT_PS
    dt_nbuf          = 0
    
    # bookkeeping / defaults
    igm_nl           = 0 # number of lines written to igm file
    raw_nl           = IMG_SL # number of raw lines parsed
    img_sl           = -1 # "true" start line of science data
    glt_xyz          = []
    ort_rgb          = []
    rgb_xyz          = []
    frame_meta       = []
    init_dem         = True
    init_buf         = True
    num_empty        = 0
    dem_ulx          = 0
    dem_uly          = 0
    dem_zone         = 0
    dem_hemi         = 'North'
    
    with open(igmf, 'wb') as igm:
        while img_sl == -1 or raw_nl < max_nl+img_sl:
            if not plot_frames:
                frame_chunk = read_frames_meta(rawf,platform,sl=raw_nl,nl=buf_nl)
            else:
                frame_chunk = read_frames_meta_bands(rawf,platform,sl=raw_nl,
                                                     nl=buf_nl,bands=frame_rgb)
                chunk_bands = frame_chunk[0]

            # last three arguments same for both read_frames functions
            chunk_meta,chunk_nl,chunk_sl = frame_chunk[-3:]                

            # number of binned lines for this chunk
            binned_chunk_nl = int(chunk_nl/bin_factor) 

            # truncate chunk_nl according to bin size to get raw lines for this chunk         
            raw_chunk_nl    = min(binned_chunk_nl*bin_factor,chunk_nl)  
            print 'Read %d raw frames'%raw_chunk_nl
                
            if raw_chunk_nl == 0: # no data read or EOF
                if img_sl == -1:
                    print 'Waiting for valid science data'
                    time.sleep(WAIT_DATA)
                    continue
                else: # EOF
                    print 'Reached EOF'
                    break
            elif chunk_sl == -1: # all frames in chunk non-science frames
                raw_nl += raw_chunk_nl
                continue
            elif img_sl == -1: # found a science frame, skip to start
                # note: chunk_sl is offset by sl argument to read_frames
                print 'Science frame found at frame index %d'%chunk_sl
                img_sl  = chunk_sl
                raw_nl += img_sl-raw_nl
                # keep these values if we're using >1 chunk to estimate ort_ps
                dt_sl,dt_raw  = img_sl,raw_nl
                continue # rewind to populate buffer from first science frame

            print 'Processing frames %d to %d'%(raw_nl,raw_nl+chunk_nl)

            # retrieve the latest pps/gps tables from the disk
            pps_table,gps_table,return_code = collect_tables(update_mode=table_updates,check_size=check_size) 

            if return_code==FAILURE:
                print 'ERROR: corrupt data in GPS table, unable to proceed'
                return FAILURE
            
            chunk_clock = chunk_meta[:,0]                        
            if init_dem: # initialize dem and other required ortho variables
                init_latlon = clock2location(chunk_clock[0])
		if len(init_latlon)<1:
                    print 'Waiting for valid lat/lon data'
                    time.sleep(WAIT_DATA)
                    continue

                if len(offset_latlon) != 0:
                    init_latlon[0] = init_latlon[0]+offset_latlon[0]
                    init_latlon[1] = init_latlon[1]+offset_latlon[1]
                
                dem = DEM(dem_prefix,init_latlon[[1,0]],subset_width=subset_width)
                if not dem.initialized:
                    print 'ERROR: unable to initialize DEM with prefix "%s"'%dem_prefix
                    return FAILURE

                # dem references
                dem_utm    = dem.data_utm
                dem_ps     = dem.ps
                dem_ulx    = dem.ulx
                dem_uly    = dem.uly
                dem_zone   = dem.utm_zone
                dem_hemi   = dem.utm_hemi
                
                if verbose:
                    print 'DEM min: %10.6f, max: %10.6f'%dem.extrema
                
                init_dem = False

            if dt_nbuf < DT_CHUNKS:
                # compute ps from DT_CHUNKS buffers 
                if chunk_nl < ps_nl:
                    print 'At least %d science frames required to estimate ort_ps, waiting...'%(PS_NL)
                    time.sleep(WAIT_DATA)
                    continue
                elif dt_nbuf == 0:
                    print 'Computing downtrack pixel size using NADIR-pointing',
                    print 'samples %s, frames %d-%d'%(dt_samp,img_sl,
                                                      img_sl+chunk_nl)

                dt_mask  = dt_lidx<chunk_nl
                dt_lidx  = dt_lidx[dt_mask]
                dt_sidx  = dt_sidx[dt_mask]
                dt_clock = chunk_clock[dt_lidx]
                dt_xyz   = geolocate(dt_lidx,dt_sidx,dt_clock,nav,dem,
                                     interp_geoid=True,return_lonlat=False,
                                     verbose=verbose,rs_ps=ort_ps,
                                     offset_latlon=offset_latlon)

                dt_xyz3   = dt_xyz.reshape([chunk_nl,dt_ns,3])

                print 'DT averaging results (chunk %d of %d):'%(dt_nbuf+1,
                                                                DT_CHUNKS)
                ps_dict = xyz2ps(dt_xyz3,verbose=True)
                est_ort[dt_nbuf] = ps_dict['ort_ps']
                est_dt[dt_nbuf] = ps_dict['dt_ps']
                est_factor[dt_nbuf] = ps_dict['bin_factor']

                if dt_nbuf+1 != DT_CHUNKS:
                    raw_nl += raw_chunk_nl
                else:
                    # pick the bin factor most similar to the previous estimate
                    # todo: pick smallest bin_factor? median of smallest?
                    # consider dt/at ps variance during bin_factor selection
                    if DT_CHUNKS > 1:                        
                        est_diff = abs(diff(est_factor))
                        est_fit  = where(est_diff<=BIN_DELTA)[0]+1
                        if len(est_fit) > 1:
                            # average if multiple acceptable values exist
                            ort_ps = ps_avg_fn(est_ort[est_fit])
                            ort_ps = int(ort_ps*10)/10.0
                            dt_ps = ps_avg_fn(est_dt[est_fit])
                            bin_factor  = int(ceil(ort_ps/dt_ps))
                        else:
                            # pick the estimate closest to its predecessor
                            idx = est_diff.argmin()+1
                            ort_ps = est_ort[idx]
                            bin_factor = est_factor[idx]
                        #est_idx = est_factor.argmin()
                    else:
                        ort_ps = est_ort[0]
                        bin_factor = est_factor[0]

                    with open(binf,'w') as binfout:
                        print 'Writing', binf
                        binfout.write('%d\n'%bin_factor)
                    
                    # found our ort_ps and bin_factor, rewind to img_sl,raw_nl
                    img_sl,raw_nl = dt_sl,dt_raw

                dt_nbuf = dt_nbuf+1                    
                continue 
            
            if init_buf:
                # update buf_nl according to bin factor
                binned_nl = int(BUF_NL/bin_factor)
                buf_nl    = binned_nl*bin_factor

                print 'buf_nl: %d'%buf_nl
                print 'binned_nl: %d'%binned_nl

                # subset orig line/samp indices based on binned_nl
                buf_mask   = buf_lidx<binned_nl
                buf_lidx   = buf_lidx[buf_mask]
                buf_sidx   = buf_sidx[buf_mask]
                buf_lines  = buf_lines[buf_lines<binned_nl]

                # also subset interpolated samples
                intp_mask  = intp_lidx<binned_nl
                intp_lidx  = intp_lidx[intp_mask]
                intp_sidx  = intp_sidx[intp_mask]
                intp_lines = intp_lines[intp_lines<binned_nl]

                # buffer for igm coords (initialized to UNDEF)
                igm_xyz    = ones([binned_nl,pb_len,3],dtype=double)*PIX_ERROR_UNDEF
                igm_mask   = zeros(binned_nl,dtype=bool8)

                # make sure we're not outside of (binned) chunk dims
                buf_range  = buf_lines[buf_lines<binned_chunk_nl]            
                buf_mask   = buf_lidx<binned_chunk_nl
                buf_lidx   = buf_lidx[buf_mask]
                buf_sidx   = buf_sidx[buf_mask]
                init_buf   = False
                continue

            # apply bin factor to get binned_chunk_nl clock times
            bin_clock  = rebin(chunk_clock[:raw_chunk_nl],bin_factor)
            buf_clock  = bin_clock[buf_lidx]

            # geolocate samples in buffer
            start_time = dtime_now()
            ground_xyz = geolocate(buf_lidx,buf_sidx,buf_clock,nav,dem,
                                   interp_geoid=True,verbose=verbose,
                                   rs_ps=ort_ps,return_lonlat=return_lonlat,
                                   offset_latlon=offset_latlon,
                                   frame_meta=frame_meta)

            # interpolate out-of-buffer samples
            # binned_chunk_nl always <= binned_nl
            igm_shape = [binned_chunk_nl,buf_ns,3]
            # clear buffers
            igm_xyz[:,:,:] = PIX_ERROR_UNDEF 
            igm_mask[:]  = 0
            if interpolate_samples:
                igm_xyz[:binned_chunk_nl,buf_samp,:] = ground_xyz[:,:3].reshape(igm_shape)
                # lerp in mapX,mapY, then bilerp in elevation
                for l in range(binned_chunk_nl):                    
                    # only fit samples without errors (large negative values=errors)
                    fit_mask = igm_xyz[l,buf_samp,2]>PIX_ERROR_UNDEF
                    if (fit_mask==0).all():
                        continue
                    igm_mask[l] = 1
                    fit_samp    = buf_samp[fit_mask]
                    pcoef       = polyfit(fit_samp,igm_xyz[l,fit_samp,:2],
                                          deg=POLY_DEG)
                    igm_x,igm_y = polyval(pcoef,intp_scoef).T
                    igm_s,igm_l = map2sl(igm_x,igm_y,dem_ulx,dem_uly,dem_ps)
                    igm_z       = bilerp(dem_utm,igm_s,igm_l)
                    igm_xyz[l,intp_samp,:] = c_[igm_x,igm_y,igm_z]
            else:
                igm_xyz[:binned_chunk_nl,:,:] = ground_xyz[:,:3].reshape(igm_shape)
                # check for empty lines
                for l in range(binned_chunk_nl):
                    # as long as we have at least 1 valid sample in a line, we keep it
                    fit_mask = igm_xyz[l,:,2]>PIX_ERROR_UNDEF
                    if (fit_mask==0).all():
                        continue                        
                    igm_mask[l] = 1

            igm_mask_chunk  = igm_mask[:binned_chunk_nl]
            binned_empty_nl = np_sum(igm_mask_chunk==0)

            # if all lines empty, retry 
            if binned_chunk_nl == binned_empty_nl:
                if num_empty < MAX_EMPTY:
                    print 'No valid pixels in buffer, waiting %d seconds for new GPS data'%WAIT_GPS
                    num_empty += 1
                    time.sleep(WAIT_GPS)
                    continue
                else:
                    print 'Observed MAX_EMPTY (%d) empty raw buffers in a row.'%MAX_EMPTY
                    stop_early  = False
                    bad_input = 0
                    while bad_input < EMPTY_RETRY:
                        yn = input_timeout('Continue waiting (y/n)? ','y',WAIT_USER)
                        if yn.lower() in ('n','no'):
                            stop_early = True
                            break
                        else:
                            print 'Invalid input "%s" (please choose "y" or "n")'%yn
                            bad_input += 1
                            
                    if not stop_early:
                        num_empty = 0
                    else:
                        break
                    

            # found >= 1 frame with georeferenced measurements, reset num_empty
            num_empty = 0
                
            # compute start line for next chunk based on empty lines
            if binned_empty_nl > 0:
                # if we found a few empty lines, we'll want to attempt to redo those,
                # so find first empty (binned) frame index, increment raw_nl accordingly

                # FIXME (BDB, 09/15/15): what happens if valid frames exist
                #                        after raw_empty_sl? dups in the igm?
                binned_empty_idx = where(igm_mask_chunk)[0]
                binned_next_sl   = binned_empty_idx.min()
                raw_next_sl      = binned_next_sl*bin_factor
            else:
                # no empty frames, move start line to next chunk
                raw_next_sl    = raw_chunk_nl
                binned_next_sl = binned_chunk_nl

            if binned_next_sl > 0:
                # get rid of empty lines
                ort_xyz = igm_xyz[igm_mask_chunk,:,:]

                # TODO (BDB, 08/20/15): replace error codes with nodata?
                igm.write(double(ort_xyz).tobytes())

                ort_xyz = ort_xyz.reshape([-1,3])
                
                glt_xyz = r_[glt_xyz,ort_xyz] if len(glt_xyz) > 0 else ort_xyz

            raw_nl   += raw_next_sl
            igm_nl   += binned_next_sl
            time_str  = time_elapsed(start_time)

            if raw_next_sl > 0:
                raw_chunk_per = (100.0*raw_chunk_nl/raw_next_sl)
                outstr  = 'Frames processed (%% of %4d):        %3.0f\n'%(raw_next_sl,raw_chunk_per)
                outstr += 'Pixel size (meters):                 %.1f\n'%ort_ps
                outstr += 'Bin factor (raw frames):             %d\n'%bin_factor
                outstr += 'Geolocated samples/frame:            %d\n'%buf_ns
                outstr += 'Interpolated samples/frame:          %d\n'%intp_ns
                outstr += 'Total raw frames processed:          %d\n'%raw_nl
                outstr += 'Total binned frames processed:       %d\n'%igm_nl
                outstr += 'Total CPUtime (H:M:S.ms):            %s'%time_str
            print outstr

            # plot corrected frames
            if raw_nl>=max_nl and (plot_rgb or plot_frames):
                rgb_nl = raw_nl-raw_chunk_nl
                if plot_frames:
                    rgb_range = arange(0,chunk_nl,bin_factor)
                    rgb_chunk = (chunk_bands[rgb_range,:,:][:,pb_samp,:]).reshape([-1,3])                    
                    rgb_xyz = r_[rgb_xyz,rgb_chunk] if len(rgb_xyz) > 0 \
                              else rgb_chunk
                else:
                    rgb_range = arange(0,rgb_nl,bin_factor)+img_sl
                    rgb_xyz = (rgb_data[rgb_range,:,:][:,pb_samp,:]).reshape([-1,3])

                ort_rgb = r_[ort_rgb,rgb_xyz] if len(ort_rgb)>0 \
                          else rgb_xyz

                glt_keep = glt_xyz[:,2]>=0
                
                glt_sel = glt_xyz[glt_keep,:]
                rgb_sel = ort_rgb[glt_keep,:]
                if len(rgb) == 0:
                    imgmin = rgb_sel[:buf_nl,:].min(axis=0)
                    if plot_frames:
                        imgmax = 1500
                        imgdif = imgmax-imgmin 
                    else:
                        imgdif = rgb_sel[:buf_nl,:].max(axis=0)-imgmin 
                
                rgb = (rgb_sel-imgmin)/imgdif
                rgb[(rgb<0)|(rgb!=rgb)] = 0; rgb[rgb>1] = 1; 
                
                fig = pl.figure(1)
                pl.subplot(3,1,1)
                pl.imshow(igm_xyz[:,:,0],interpolation='none')
                pl.ylabel('IGM mapX')
                pl.subplot(3,1,2)
                pl.imshow(igm_xyz[:,:,1],interpolation='none')
                pl.ylabel('IGM mapY')
                pl.subplot(3,1,3)
                pl.imshow(igm_xyz[:,:,2],interpolation='none')
                pl.ylabel('IGM elev')
                
                fig = pl.figure(2,figsize=(10,10),dpi=100)
                pl.hold('on')
                ax = fig.add_subplot(111)
                ax.set_rasterization_zorder(1)
                ax.scatter(glt_sel[:,1],glt_sel[:,0],s=10,
                           marker='s',c=rgb,zorder=0)
                outstr  = img_base+'\n'+outstr
                left, top = .01, .99
                ax.text(left,top,outstr,color='red',
                        horizontalalignment='left',
                        verticalalignment='top',fontsize=10,
                        rotation=0,family='monospace',
                        transform=ax.transAxes)

                pl.axis('equal')
                pl.draw()
                pl.show()

    # update glts/igms and write to disk
    update_igm(igmf,igm_nl,pb_len,bin_factor,dem_zone,dem_hemi)
    update_glt(gltf,glt_xyz,igm_nl,pb_samp,dt_samp,dem_ulx,dem_uly,ort_ps,
               dem_zone,dem_hemi,img_sl,bin_factor,NN_RAD,NN_BUF)

    generate_obs_loc(igmf,dem,frame_meta)

    print('GLT/IGM generation complete for file %s.'%rawf)
    return SUCCESS

def init_test(test_case):
    valid_cases = set(['ucr','ucr_india','4c', 'shndoa', 'burbank',
                       'prism0','prism1','prism2'])
    if test_case not in valid_cases:
        print('Error: invalid test case "%s"'%test_case)
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
    if 'prism' in test_case:
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
            dem_prefix = '/home/ngdcs/src/range/data/dem'
        else:
            IMG_DIR='/Volumes/QuantumSpace/Data/AVIRISNG/20140612_ucr'
            #IMG_DIR='/Volumes/TravelSpace/Data/AVIRISNG/20140612_ucr'
            if test_india:
                dem_prefix='/Users/bbue/Desktop'
            else:
                dem_prefix='/Volumes/QuantumSpace/Data/dem'
        dem_file = 'state'
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--platform', help='Platform (AVIRIS-NG or PRISM)', 
                        type=str, required=False) 
    parser.add_argument('-r','--root_prefix', help='Path where camera/geoid files are located', 
                        type=str, required=True)   
    parser.add_argument('-c','--camera_file', help='Path to camera file for specified platform',
                        type=str, required=False)    
    parser.add_argument('-g','--geoid_file', help='Path to geoid file for specified platform',
                        type=str, required=False)    
    parser.add_argument('-d','--dem_prefix', help='DEM path + filename prefix', 
                        type=str, required=False)
    parser.add_argument('-o','--output_path', help='Path to store output products', 
                        type=str, required=False)    
    parser.add_argument('-t','--test', help='Enable test mode', 
                        action='store_true', default=False, required=False)
    parser.add_argument('-n','--numlines', help='Maximum number of lines to process', 
                        type=int, default=MAX_NL, required=False)
    
    parser.add_argument('raw', help='Path to raw image (or test case id if test_mode enable)')
    args = vars(parser.parse_args())

    root_prefix   = args['root_prefix']
    platform_id   = args['platform']    
    rawf          = args['raw']
    max_nl        = args['numlines']
    test_mode     = args['test']
    camera_file   = args['camera_file']
    geoid_file    = args['geoid_file']
    output_path   = args['output_path'] or OUTPUT_PATH
    offset_latlon = []

    if not pathexists(output_path):
        print 'ERROR: output path %s does not exist'%output_path
        sys.exit(FAILURE)
    
    if test_mode: 
        # in test mode, rawf gives the name of the test case, and we get the
        # dem_prefix and img_dir via the init_test function
        test_case  = rawf.lower()
        test_params = init_test(test_case)
        if len(test_params) == 0:
            print 'ERROR: unrecognized test case'%test_case
            sys.exit(FAILURE)            
        rawf, dem_prefix, offset_latlon = test_params
    else:
        # init dem_prefix using environment variable or defaults 
        dem_prefix = DEM_PREFIX

    if not pathexists(rawf):
        print 'ERROR: raw file %s does not exist'%rawf
        sys.exit(FAILURE)
        
    # override defaults if prefix passed on cmdline
    dem_prefix = args.get('dem_prefix') or dem_prefix
        
    if platform_id is None:
        # guess the platform by filename prefix if unspecified    
        platform = identify_platform(rawf)
    else:            
        platform = load_platform(platform_id,imgf=rawf,camf=camera_file,
                                 geof=geoid_file)
        
    if platform is None:
        print 'ERROR: platform unspecified and cannot be identified from file "%s"'%rawf
        sys.exit(FAILURE)

    # use sanitized platform_id
    platform_id = platform.platform_id

    print 'raw_file:',rawf
    print 'platform_id:',platform_id
    print 'dem_prefix:',dem_prefix
    print 'output_path:',output_path                

    retval = compute_igm_glt(rawf,platform,output_path,dem_prefix,max_nl=max_nl,
                             offset_latlon=offset_latlon)
    #wait_exit(retval)
    sys.exit(retval)
