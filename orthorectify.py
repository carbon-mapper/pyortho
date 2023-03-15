#!/usr/bin/env python
from __future__ import print_function
import sys, argparse

from ortho_util import *
from ortho_config import *
from ortho_platform import *
from ortho_dem import *
from ortho_nav import *

plot_rgb      = False
plot_frames   = False

# default paths
GEOID_FILE  = get_env('GEOID_FILE',pathjoin(PYORT_ROOT,'world_model/egm96/egm96'))
OUTPUT_PATH = get_env('WATCH_OUTPUT','./pyortho_output')
DEM_PREFIX  = get_env('DEM_PREFIX',realpath(pathjoin(ORTHO_ROOT,'data/dem')))

#@profile
def compute_igm_glt(rawf,platform,config,output_path,dem_prefix,max_nl=None,
                    offset_clock=0,offset_latlon=[]):
    global plot_frames, plot_rgb

    verbose       = config['verbose']
    return_lonlat = config['return_lonlat']
    table_updates = config['table_updates']
    check_bounds  = config['check_dem_bounds']
    check_size    = config['check_table_size']
    interp_geoid  = config['interp_geoid']
    subset_width  = config['subset_width']
    read_hdr_nl   = config['read_hdr_nl']

    ORT_PS        = config['ORT_PS']
    BIN_FACTOR    = config['BIN_FACTOR']    
    DT_CHUNKS     = config['DT_CHUNKS']
    BIN_DELTA     = config['BIN_DELTA']
    ALT_DELTA     = config['ALT_DELTA']
    GLT_ROT_SNAP  = config['GLT_ROT_SNAP']

    MIN_PS        = config['MIN_PS']
    MAX_PS        = config['MAX_PS']

    IMG_SL        = config['IMG_SL']
    MAX_NL        = config['MAX_NL']
    BUF_NL        = config['BUF_NL']

    EMPTY_MAX     = config['EMPTY_MAX']
    EMPTY_RETRY   = config['EMPTY_RETRY']
    WAIT_DATA     = config['WAIT_DATA']
    WAIT_GPS      = config['WAIT_GPS']
    WAIT_USER     = config['WAIT_USER']

    S_STEP        = config['S_STEP']
    POLY_DEG      = config['POLY_DEG']
    NN_RAD        = config['NN_RAD']
    NN_BUF        = config['NN_BUF']

    allow_empty   = config['allow_empty_dem']
    ps_avg_fn     = config['ps_avg_fn']

    inittime = dtime_now()

    img_dir,img_base = pathsplit(rawf.replace('_raw',''))

    out_dim = 5 if config['return_lonlat'] else 3

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
            print('Loaded RGB image %s'%rgbf)
        except Exception:
            warn('unable to load rgb image, falling back to raw frames')
            plot_rgb = False
            plot_frames = True
            pass

    if not os.access(output_path,os.W_OK):
        warn('IGM output path "%s" is not writeable'%output_path)
        return FAILURE
        
    binf = pathjoin(output_path,img_base+'_raw.binfac')
    igmf = pathjoin(output_path,img_base+'_rdn_igm')
    gltf = pathjoin(output_path,img_base+'_rdn_glt')

    if pathexists(igmf) and not os.access(igmf,os.W_OK):
        warn('IGM file "%s" is not writeable'%igmf)
        return FAILURE

    ppsf = pathjoin(img_dir,img_base+'_pps')
    gpsf = pathjoin(img_dir,img_base+'_gps')    
    
     # initialize world, camera_model and elevation models
    nav  = NAV(platform,ppsf,gpsf,GEOID_FILE,table_updates=table_updates,
               check_size=check_size,offset_clock=offset_clock,verbose=verbose)
    if not nav.initialized:
        warn('Unable to initialize navigation parameters')
        return FAILURE

    # reload camera_model if exists
    if config['UPDATED_CAM_MODEL'] != None and os.access(config['UPDATED_CAM_MODEL'], os.W_OK):
        print ("before nav.platform.camera_model")
        print (nav.platform.camera_model)
        with open(config['UPDATED_CAM_MODEL'], 'r') as f:
            lines2 = f.readlines()
            if len(lines2) == nav.platform.camera_model.size/3:
                count = 0
                for line in lines2:
                    values = line.split()
                    nav.platform.camera_model[:,count] = values[3:6]
                    count += 1
            f.close()
        print ("after nav.platform.camera_model")
        print (nav.platform.camera_model)
    
    
    pb_len  = nav.platform.pb_len # pushbroom length for the current platform camera
    pb_cen  = nav.platform.pb_cen
    pb_off  = nav.platform.pb_off
    pb_samp = arange(pb_cen-int(pb_len/2),pb_cen+int(pb_len/2))

    assert(diff(pb_samp).max()==1)
    
    if S_STEP != 1:
        # when interpolating, ensure first/last samples in frame are geolocated
        pb_half = pb_samp[:pb_cen:S_STEP]
        buf_samp = r_[pb_half,(pb_len-pb_half-1)[::-1]]
    else:
        buf_samp = pb_samp

    # initialize line/sample indices to geolocate with respect to buffer    
    buf_nl              = BUF_NL # will shrink to (a multiple of) BUF_NL 
    buf_ns              = len(buf_samp)
    buf_per             = int(100.0*buf_ns/pb_len)
    buf_lines           = arange(BUF_NL)
    buf_lidx,buf_sidx   = map(ravel,meshgrid(buf_lines,buf_samp))

    # samples to interpolate
    intp_samp           = setdiff1d(pb_samp,buf_samp)
    intp_ns             = len(intp_samp)
    intp_per            = int(100.0*intp_ns/pb_len)
    intp_lines          = buf_lines.copy()
    intp_lidx,intp_sidx = map(ravel,meshgrid(buf_lines,intp_samp))
    intp_scoef          = c_[intp_samp,intp_samp] # sample coef for interpolation

    # default ps/bin_factors
    ps_nl              = BUF_NL
    ort_ps             = ORT_PS
    bin_factor         = BIN_FACTOR
    bin_nl             = BUF_NL
    empty_nl           = 0
    use_gps_bounds     = False

    compute_max_nl     = True
    dt_nl              = NOTFOUND
    if max_nl is None or max_nl == inf:
        max_nl = MAX_NL # defaults        
        rawhdrf = rawf+'.hdr'
        if read_hdr_nl and pathexists(rawhdrf):
            try:
                rawhdr = read_envi_header(rawhdrf)
                max_nl = int(rawhdr['lines'])
                use_gps_bounds = True
            except Exception as e:
                warn('unable to extract max_nl from hdr file "%s"'%rawhdr)            

        collect = 0
        if compute_max_nl:
            img_sl, img_nl, frames = find_science_frames(rawf,platform,
                                                         start_line=IMG_SL,
                                                         num_lines=BUF_NL,
                                                         collect=collect)
            if collect:
                print('frames.min(axis=0): "%s"'%str((frames.min(axis=0))))
                print('frames.max(axis=0): "%s"'%str((frames.max(axis=0))))
                print('np.diff(frames).min(axis=0): "%s"'%str((diff(frames,axis=0).min(axis=0))))
                print('np.diff(frames).max(axis=0): "%s"'%str((diff(frames,axis=0).max(axis=0))))
                max_nl = min(max_nl,img_nl)
                dt_nl = img_nl-img_sl # img_sl,img_nl = first,last science frames
                print('img_sl: "%s"'%str((img_sl)))
                print('img_nl: "%s"'%str((img_nl)))
                print('max_nl: "%s"'%str((max_nl)))
                print('dt_nl: "%s"'%str((dt_nl)))
                frameticks = frames[:,0]            
                fc2pos = lambda t: nav.clock2location(t[0])
                frame_pos = apply_along_axis(fc2pos,1,frameticks.reshape([-1,1]))
                import pylab as pl
                fig=pl.figure()
                plot_trajectory(frame_pos,frameticks,fig=fig,cmap='Spectral')

                bfig = pl.figure()
                rbinf = 2
                rframeticks = rebin(frames[:rbinf*(len(frames)//rbinf),0],rbinf)
                rframe_pos = apply_along_axis(fc2pos,1,rframeticks.reshape([-1,1]))
                plot_trajectory(rframe_pos,rframeticks,fig=bfig,cmap='gist_earth')
                pl.show()
            
    print('raw max_nl:   %.0f'%max_nl)   
    
    # bookkeeping / defaults
    glt_xyz          = []
    frame_meta       = []
    ort_rgb          = []
    rgb_xyz          = []

    igm_nl           = 0      # number of lines written to igm file
    raw_nl           = IMG_SL # number of raw lines parsed
    img_sl           = NOTFOUND     # index of first science frame
    img_nl           = max_nl # index of last science frame
    img_ss           = pb_off # index of first sample in first science frame
    buf_sl           = 0

    init_dem         = True
    init_dt          = True
    init_buf         = True
    num_empty        = 0
    dem_ulx          = 0
    dem_uly          = 0
    dem_zone         = 0
    dem_hemi         = 'North'
    chunk_n          = 0

    # init downtrack and across track pixel size, compute from dt_width center samples    
    dt_width         = 1
    dt_sl            = NOTFOUND     # index of first science frame
    dt_raw           = NOTFOUND     
    dt_samp          = arange(pb_cen-dt_width,pb_cen+dt_width+1)
    dt_lidx,dt_sidx  = map(ravel,meshgrid(buf_lines,dt_samp))
    dt_ns            = len(dt_samp)
    dt_ps,at_ps      = ORT_PS, ORT_PS
    dt_min,dt_max    = ORT_PS, ORT_PS
    at_min,at_max    = ORT_PS, ORT_PS
    dt_nbuf          = 0

    if dt_nl == NOTFOUND:
        # (conservatively) estimate ort_ps on using half of the raw lines
        dt_nl = int(0.5*(max_nl-IMG_SL))

    # resize DT_CHUNKS to fit in the range of available science frames
    if DT_CHUNKS*buf_nl > dt_nl:
        dt_chunks = max(1,int(dt_nl/buf_nl))
        msg  = 'reduced DT_CHUNKS from %d to %d to permit pixel size'
        msg += ' estimation with max_nl=%d frames'
        warn(msg%(DT_CHUNKS,dt_chunks,max_nl))
    else:        
        dt_chunks = DT_CHUNKS

    # ort_ps,dt_ps,bin_factor estimates
    est_ort = DT_ERROR_UNDEF*ones([dt_chunks,3])

    n_err = 0
    with open(igmf, 'wb') as igm:
        rttime = dtime_now()
        while img_sl == NOTFOUND or raw_nl < max_nl+img_sl:
            n_err_bin = 0
            if not plot_frames:
                frame_chunk = read_frames_meta(rawf,platform,start_line=raw_nl,num_lines=buf_nl)
                chunk_meta,chunk_nl,chunk_sl = frame_chunk
            else:
                frame_chunk = read_frames_meta_bands(rawf,platform,start_line=raw_nl,
                                                     num_lines=buf_nl,bands=frame_rgb)
                chunk_bands,chunk_meta,chunk_nl,chunk_sl = frame_chunk
            chunk_sci = chunk_nl-chunk_sl if chunk_sl!=-1 else 0

            # notes:
            # chunk_nl = number of frames read in this chunk (<= buf_nl)
            # chunk_sl = index of first science frame relative to chunk (-1 if no science frames)
            # chunk_sci = chunk_nl-chunk_sl = number of science frames in chunk
            # chunk_sl+chunk_sci = index of last science frame relative to chunk
            # raw_nl+chunk_sl = index of first science frame relative to file            
            
            # number of binned frames relative to *this* chunk
            bin_chunk_nl = int(chunk_nl/bin_factor) 

            # truncate chunk_nl to bin_factor multiple to get raw lines in chunk
            raw_chunk_nl = min(bin_chunk_nl*bin_factor,chunk_nl)

            if raw_chunk_nl == 0: # no data read or EOF
                if img_sl == NOTFOUND:
                    if verbose:
                        print('No science frames found')
                    if WAIT_DATA > 0:
                        # if running in real time, wait for buffer to fill
                        if verbose:
                            print('waiting %d seconds for new frames'%WAIT_DATA)
                        time_sleep(WAIT_DATA)
                    continue                                
                else: # EOF
                    if verbose:
                        print('Reached EOF')
                    if init_dt: 
                        if dt_nbuf == 0:
                            # ran past EOF while estimating ort_ps
                            warn('Unable to estimate ort_ps, cannot proceed')
                            break
                        elif dt_nbuf < dt_chunks:
                            # not enough science frames to estimate ort_ps
                            # using DT_CHUNKS, truncate to available estimates
                            print('\nEstimaing ort_ps from %d chunks'%dt_nbuf)
                            est_ort = est_ort[:dt_nbuf,:]
                            dt_chunks = dt_nbuf                            
                            continue
                    else:
                        # all done!
                        print('\nGeolocalization CPUtime (MM:SS.ms):        %s'%time_elapsed(rttime))
                        print('Total Buffers processed:       %d\n'%(chunk_n+1))
                        break
            elif chunk_sl != NOTFOUND and img_sl == NOTFOUND:
                # found a science frame
                img_sl = raw_nl+chunk_sl # starting frame index
                img_sl += platform.shutter_offset # drop a few frames in case the shutter isn't fully open
                raw_nl = img_sl # move pointer to index of science frame
                if verbose:
                    print('Science frame found at index %d'%img_sl)
                # save these values if we're using >1 chunk to estimate ort_ps
                dt_sl,dt_raw  = img_sl,raw_nl
                chunk_sl = chunk_nl = 0
                continue # rewind to populate buffer from first science frame
            elif chunk_sl == NOTFOUND and img_sl != NOTFOUND: # all frames in chunk non-science frames
                print('All science frames processed')
                img_nl = raw_nl
                if not init_dt:
                    break
            elif chunk_sl == NOTFOUND and img_sl == NOTFOUND:
                if verbose:
                    print('No science frames found in buffer starting at frame %d'%raw_nl)
                raw_nl += raw_chunk_nl
                continue
            
            raw_clock = double(chunk_meta[:,0])
            if init_dem: # initialize dem and other required ortho variables                
                init_latlon = nav.clock2location(raw_clock[0])
                                                 
                if init_latlon[0] == PIX_ERROR_NO_LOC:
                    if verbose:
                        print('No valid lat/lon data in buffer starting at frame %d'%raw_nl)
                    if WAIT_DATA > 0:
                        if verbose:
                            print('waiting %d seconds for updated tables and new frames'%WAIT_DATA)
                        time_sleep(WAIT_DATA)
                    continue                

                if not use_gps_bounds:
                    subset_dims = subset_width
                else:
                    subset_width = 0.25
                    lrlon,lrlat = nav.maxlon+subset_width,nav.minlat-subset_width
                    ullon,ullat = nav.minlon-subset_width,nav.maxlat+subset_width

                    subset_dims = [ullon,ullat,lrlon,lrlat]
                
                if len(offset_latlon) != 0:
                    init_latlon[0] = init_latlon[0]+offset_latlon[0]
                    init_latlon[1] = init_latlon[1]+offset_latlon[1]
                
                dem = DEM(dem_prefix,init_latlon[[1,0]],subset_dims=subset_dims,
                          allow_empty_dem=allow_empty,verbose=verbose)
                if not dem.initialized:
                    warn('unable to initialize DEM with prefix "%s"'%dem_prefix)
                    return FAILURE

                # dem references
                dem_utm    = dem.data_utm
                dem_ps     = dem.ps
                dem_ulx    = dem.ulx
                dem_uly    = dem.uly
                dem_zone   = dem.utm_zone   
                dem_hemi   = dem.utm_hemi
                dem_alpha  = dem.meta['alpha']

                init_dem = False

            if init_dt:
                # initialize downtrack pixel averaging parameters
                if dt_nbuf < dt_chunks:
                    # compute ps from dt_chunks buffers 
                    if chunk_nl < ps_nl:
                        print('At least %d science frames required to estimate ort_ps'%(ps_nl))
                        if WAIT_DATA > 0:
                            if verbose:
                                print('waiting %d seconds for new frames'%WAIT_DATA)
                            time_sleep(WAIT_DATA)
                        continue
                    elif dt_nbuf == 0:
                        ptup = (dt_chunks,raw_chunk_nl,str(dt_samp))
                        msg  = 'Estimating downtrack pixel size using (up to)'
                        msg += ' %d chunks of %d frames/chunk, sample indices %s'
                        print(msg%ptup)

                    dt_mask  = dt_lidx<chunk_nl
                    dt_lidx  = dt_lidx[dt_mask]
                    dt_sidx  = dt_sidx[dt_mask]
                    dt_clock = raw_clock[dt_lidx]
                    dt_xyz   = geolocate(dt_lidx,dt_sidx,dt_clock,nav,dem,
                                         ALT_DELTA,interp_geoid=interp_geoid,
                                         verbose=verbose,rs_ps=ort_ps,
                                         offset_latlon=offset_latlon,
                                         check_bounds=check_bounds,
                                         return_lonlat=False)


                    if len(dt_xyz) == 0:
                        print('Unable to geolocate any pixels in current downtrack buffer')
                        raw_nl += chunk_nl
                        continue

                    dt_xyz   = dt_xyz.reshape([chunk_nl,dt_ns,3])

                    ptup = (dt_nbuf+1,dt_chunks,raw_nl,raw_nl+chunk_nl)
                    print('Buffer %d of %d, frames %d-%d'%ptup)

                    ps_dict = xyz2ps(dt_xyz,ps_avg_fn,MIN_PS,MAX_PS,verbose=verbose)
                    est_ort[dt_nbuf,:] = [ps_dict[v] for v in ['at_ps','dt_ps','bin_factor']]
                    dt_nbuf += 1

                if dt_nbuf != dt_chunks:
                    raw_nl += chunk_nl
                    continue

                print('\nComputing downtrack pixel size from %d chunks'%dt_nbuf)
                # exclude any undefined estimates
                dt_keep = est_ort[:,0] != DT_ERROR_UNDEF
                if not dt_keep.all():
                    print('Excluded %d undefined ort_ps estimates'%((dt_keep==0).sum()))
                    est_ort = est_ort[dt_keep,:]

                est_factor = int64(est_ort[:,2])
                #print('est_ort: "%s"'%str((est_ort)))
                #print('est_factor: "%s"'%str((est_factor)))
                # NOTE (BDB, 08/30/16): pick smallest bin_factor?
                #   median of smallest? consider dt/at ps variance during
                #   bin_factor selection?
                if len(est_ort) > 1:                        
                    est_diff = abs(diff(est_factor))
                    est_fit  = est_diff<=BIN_DELTA
                    if est_fit.any():
                        est_fit = where(est_fit)[0]+1
                        nfit = len(est_fit)
                        if nfit > 1:
                            # average if multiple estimates available
                            ort_ps = int(ps_avg_fn(est_ort[est_fit,0])*10)/10.0
                            dt_ps = ps_avg_fn(est_ort[est_fit,1])
                            bin_factor  = round(ort_ps/dt_ps)
                        elif nfit == 1:
                            # pick the estimate closest to its predecessor
                            est_idx = est_diff.argmin()
                            ort_ps = int(est_ort[est_idx,0]*10)/10.0
                            bin_factor = est_factor[est_idx]                            
                            print('est_idx: "%s"'%str((est_idx)))
                            print('est_ort[est_idx,0]: "%s"'%str((est_ort[est_idx,0])))
                            print('ort_ps: "%s"'%str((ort_ps)))
                    else:
                        ps_std = std(est_ort[:,0])
                        msg  = 'high variance in pixel size estimates'
                        msg += ' (stddev=%5.3g), using best estimate'%ps_std
                        warn(msg)
                        est_idx = est_diff.argmin()+1
                        ort_ps = int(est_ort[est_idx,0]*10)/10.0
                        bin_factor = est_factor[est_idx]
                else:
                    ort_ps = int(est_ort[0,0]*10)/10.0
                    bin_factor = est_factor[0]

                ort_ps = double(ort_ps)
                bin_factor = int64(max(1,bin_factor))
                    
                with open(binf,'w') as binfout:
                    binfout.write('%d\n'%bin_factor)
                    if verbose:
                        print('Wrote bin_factor=%d to %s'%(bin_factor, binf))
                
                #print('\nRewinding to first science frame at index %d'%img_sl)
                img_sl,raw_nl = dt_sl,dt_raw

                # found our ort_ps and bin_factor, rewind to img_sl,raw_nl
                init_dt = False
                # need to rewind to recompute proper chunk sizes
                continue

            if init_buf:
                # update buf_nl according to bin factor

                # bin_nl = # of binned lines in a generic chunk
                # buf_nl = # of raw lines read in a generic chunk
                bin_nl     = int(BUF_NL/bin_factor) 
                buf_nl     = bin_nl*bin_factor

                if bin_factor != 1:
                    # subset orig line/samp indices based on bin_nl
                    buf_mask   = buf_lidx<bin_nl
                    buf_lidx   = buf_lidx[buf_mask]
                    buf_sidx   = buf_sidx[buf_mask]
                    buf_lines  = buf_lines[buf_lines<bin_nl]

                    # also subset interpolated samples
                    intp_mask  = intp_lidx<bin_nl
                    intp_lidx  = intp_lidx[intp_mask]
                    intp_sidx  = intp_sidx[intp_mask]
                    intp_lines = intp_lines[intp_lines<bin_nl]

                # buffer for igm coords (initialized to UNDEF)
                igm_xyz    = PIX_ERROR_UNDEF*ones([bin_nl,pb_len,3],dtype=double)
                igm_mask   = zeros(bin_nl,dtype=bool8)
                    
                init_buf   = False

                print('Initialization CPUtime (MM:SS.ms):  %s'%time_elapsed(inittime))

                print('\nSample localization/interpolation:')
                print('# Samples/frame:                    %d'%pb_len)
                print('# Geolocated samples/frame (%%):     %d (%d%%)'%(buf_ns,buf_per))
                print('# Interpolated samples/frame (%%):   %d (%d%%)'%(intp_ns,intp_per))

                print('\nPixel size + downtrack frame averaging summary:')
                print('Pixel size (meters):                %3.1f'%ort_ps)
                print('# Raw frames/buffer:                %d'%buf_nl)
                print('# Raw frames/bin:                   %d'%bin_factor)
                print('# Binned frames/buffer:             %d'%bin_nl)
                print()                
                
                continue

            if bin_chunk_nl < bin_nl:
                # this must occur outside of init_buf block
                # make sure we're not outside of (binned) chunk dims
                buf_mask   = buf_lidx<bin_chunk_nl
                buf_lidx   = buf_lidx[buf_mask]
                buf_sidx   = buf_sidx[buf_mask]
                buf_lines  = buf_lines[buf_lines<bin_chunk_nl]

                # also subset interpolated samples
                intp_mask  = intp_lidx<bin_chunk_nl
                intp_lidx  = intp_lidx[intp_mask]
                intp_sidx  = intp_sidx[intp_mask]
                intp_lines = intp_lines[intp_lines<bin_chunk_nl]
            
            ptup = (chunk_n,raw_nl,raw_nl+chunk_nl)
            print('Processing frame buffer %d (raw frames %d-%d)'%ptup)
            
            # apply bin factor to get binned_chunk_nl clock times
            bin_clock  = rebin(raw_clock[:raw_chunk_nl],bin_factor)            
            buf_clock  = bin_clock[buf_lidx]
            #print('extrema(bin_clock): "%s"'%str((extrema(bin_clock))))

            # geolocate samples in buffer
            start_time = dtime_now()
            ##########
            if DEBUG:
                print("######################################")
                print("ort_ps {}, return_lonlat {}, offset_latlon {}, check_bounds {}, frame_meta {}".format(ort_ps, return_lonlat, offset_latlon, check_bounds, frame_meta))
                print("######################################")
            ground_xyz = geolocate(buf_lidx,buf_sidx,buf_clock,nav,dem,ALT_DELTA,
                                   interp_geoid=interp_geoid,verbose=verbose,
                                   rs_ps=ort_ps,return_lonlat=return_lonlat,
                                   offset_latlon=offset_latlon,
                                   check_bounds=check_bounds,
                                   frame_meta=frame_meta)
            #input()

            if len(ground_xyz) == 0:
                print('Unable to geolocate any pixels in current buffer')
                continue
            
            xyz_error_mask = mask_errors(ground_xyz)
            n_err_bin = count_nonzero(xyz_error_mask)
            if n_err_bin != 0:                
                xyz_errors = ground_xyz[xyz_error_mask]
                xyz_error_types = int32(unique(xyz_errors))
                summary=['%d errors occurred during geolocation'%n_err_bin,'error summary:']
                for et_id in xyz_error_types:
                    n_et = count_nonzero(xyz_errors==et_id)
                    et_msg = PIX_ERROR_MSG[et_id] 
                    summary.append('\terror %d (%s): %d pixels'%(et_id,et_msg,n_et))
                warn('\n'.join(summary))

                if DEBUG:
                    err_lidx, err_sidx = where(xyz_error_mask)
                    err_clock = buf_clock[err_lidx]
                    debug_meta = []
                    debug_xyzll = geolocate(buf_lidx[err_lidx],buf_sidx[err_sidx],err_clock,nav,dem,ALT_DELTA,
                                            interp_geoid=interp_geoid,verbose=3,
                                            rs_ps=ort_ps,return_lonlat=True,
                                            offset_latlon=offset_latlon,
                                            check_bounds=False,
                                            frame_meta=debug_meta)
                    xyzll_shape = [bin_chunk_nl,buf_ns,5]
                    debug_xyzll = debug_xyzll.reshape(xyzll_shape)
                    debug_mask = mask_errors(debug_xyzll)
                    print('xyz errors at buf indices:')
                    debug_idx = where(debug_mask)
                    debug_lidx = unique(debug_idx[0])
                    debug_sidx = unique(debug_idx[1])
                    print(debug_idx)
                    print('values:')
                    print(debug_xyzll[debug_mask])
                    print('xyz output:')
                    set_printoptions(suppress=True,precision=3)
                    cmin,cmax = map(int,extrema(debug_idx[1]))
                    for i in debug_lidx:
                        print('binned line, raw lines')
                        blinei=((raw_nl-img_sl)//bin_factor)+i
                        rlinei=raw_nl+(i*bin_factor)
                        print(blinei,(rlinei,rlinei+bin_factor))
                        print('binned clock, raw clock')
                        ibf = i*bin_factor
                        print(buf_clock[i],raw_clock[ibf:ibf+bin_factor])
                        print('xyzll[%d:%d]:'%(cmin,cmax))
                        print(debug_xyzll[i,cmin:cmax,:])
                    set_printoptions(suppress=True,precision=12)
                
                        
            # bin_chunk_nl always <= bin_nl
            igm_shape = [bin_chunk_nl,buf_ns,3]

            # clear buffers
            igm_xyz[:,:,:] = PIX_ERROR_UNDEF 
            igm_mask[:]    = 0
            ground_xyz3 = ground_xyz[:,:3].reshape(igm_shape)

            #  TODO (BDB, 04/05/17): this could be better streamlined 
            # interpolate out-of-buffer samples
            if buf_ns != pb_len: # only samples in buf_samp geolocated
                igm_xyz[:bin_chunk_nl,buf_samp,:] = ground_xyz3
                for l in range(bin_chunk_nl):                    
                    # only fit samples without errors
                    lbad_mask = mask_errors(igm_xyz[l,buf_samp,2])
                    lbad_count = count_nonzero(lbad_mask)
                    if lbad_count!=0:
                        warn('bad pixels detected and zero-filled')
                        igm_xyz[l,lbad_mask,:] = 0
                        if lbad_count == buf_ns:
                            continue
                    
                    igm_mask[l] = 1                    
                    fit_samp    = buf_samp[~lbad_mask]

                    # lerp in mapX,mapY, then bilerp in elevation
                    pcoef       = polyfit(fit_samp,igm_xyz[l,fit_samp,:2],
                                          deg=POLY_DEG)
                    igm_x,igm_y = polyval(pcoef,intp_scoef).T
                    igm_s,igm_l = map2sl(igm_x,igm_y,dem_ulx,dem_uly,dem_ps)
                    igm_z       = bilerp(dem_utm,igm_s,igm_l)
                    igm_xyz[l,intp_samp,:] = c_[igm_x,igm_y,igm_z]
            else: # all samples geolocated
                igm_xyz[:bin_chunk_nl,:,:] = ground_xyz3
                # check for empty lines
                for l in range(bin_chunk_nl):
                    # as long as we have at least 1 valid sample in a line, we keep it
                    lbad_mask = mask_errors(igm_xyz[l,:,2])                    
                    lbad_count = count_nonzero(lbad_mask)
                    if lbad_count!=0:
                        warn('bad pixels detected and zero-filled')
                        igm_xyz[l,lbad_mask,:] = 0
                        if lbad_count==buf_ns:
                            continue
                        
                    igm_mask[l] = 1

            igm_mask_chunk  = igm_mask[:igm_xyz.shape[0]]
            bin_empty_nl = count_nonzero(igm_mask_chunk==0)

            # if all lines empty, wait and retry 
            if bin_chunk_nl == bin_empty_nl:
                if num_empty < EMPTY_MAX:
                    num_empty += 1
                    if verbose:
                        print('No valid pixels in buffer')
                    if WAIT_GPS > 0:
                        if verbose:
                            print('waiting %d seconds for new GPS data'%WAIT_GPS)
                        time_sleep(WAIT_GPS)
                    continue
                else:
                    print('Observed EMPTY_MAX (%d) empty raw buffers in a row.'%EMPTY_MAX)
                    stop_early = False
                    bad_input  = 0
                    while bad_input < EMPTY_RETRY:
                        yn = 'n'
                        if WAIT_USER > 0:
                            yn = input_timeout('Continue waiting (y/n)? ','y',
                                               WAIT_USER)
                            
                        if yn.lower() in ('n','no'):
                            stop_early = True
                            break
                        else:
                            print('Invalid input "%s" (choose "y" or "n")'%yn)
                            bad_input += 1
                            
                    if not stop_early:
                        num_empty = 0
                    else:
                        break

            # found >= 1 frame with georeferenced measurements, reset num_empty
            num_empty = 0
                
            # compute offset for next chunk based on empty lines
            if bin_empty_nl > 0:
                # if we found a few empty lines, attempt to redo those, so find
                # first empty (binned) frame index, increment raw_nl accordingly

                # NOTE (BDB, 09/15/15): what happens if valid frames exist
                #                       after raw_empty_sl? dups in the igm?
                bin_empty_idx = where(igm_mask_chunk)[0]
                bin_inc_sl    = bin_empty_idx.max()+1
                raw_inc_sl    = bin_inc_sl*bin_factor
                empty_nl     += bin_empty_nl
            else:
                # no empty frames, move start line to next chunk
                bin_inc_sl = bin_chunk_nl
                raw_inc_sl = raw_chunk_nl

            #print('bin_inc_sl: "%s"'%str((bin_inc_sl)))
            #print('raw_inc_sl: "%s"'%str((raw_inc_sl)))
                
            if bin_inc_sl > 0:
                # get rid of empty lines, handle truncated chunks
                ort_xyz = igm_xyz[igm_mask_chunk,:,:]

                #bad_x = dt_distance(ort_xyz)
                #bad_y = at_distance(ort_xyz)
                #print(bad_x.shape,extrema(bad_x))
                #print(bad_y.shape,extrema(bad_y))
                
                # TODO (BDB, 08/20/15): replace error codes with nodata?
                igm.write(double(ort_xyz).tobytes())

                # TODO (BDB, 04/27/16): get rid of glt_xyz, use igm memmap 
                ort_xyz = ort_xyz.reshape([-1,3])
                glt_xyz = r_[glt_xyz,ort_xyz] if len(glt_xyz) > 0 else ort_xyz

            raw_nl += raw_inc_sl
            igm_nl += bin_inc_sl
            n_err  += n_err_bin
            if raw_inc_sl > 0:
                ort_ullr = ort_xyz[[0,-1]]
                lon_range,lat_range = dem.utm2lonlat(ort_ullr[:,1],
                                                     ort_ullr[:,0])
                zfmt_str = lambda val: '%-9.6f'%val
                llfmt_str = lambda val: '%-7.4f'%val
                lat_str = ', '.join(map(llfmt_str,sorted(lat_range)))
                lon_str = ', '.join(map(llfmt_str,sorted(lon_range)))
                z_str = ', '.join(map(zfmt_str,sorted(ort_ullr[:,2])))
                
                raw_chunk_per = (100.0*raw_chunk_nl/raw_inc_sl)
                outstr  = 'Latitude range:                     %s\n'%lat_str
                outstr += 'Longitude range:                    %s\n'%lon_str
                outstr += 'Elevation range:                    %s\n'%z_str
                outstr += '# Raw frames processed:             %d\n'%raw_nl
                outstr += '# Binned frames processed:          %d\n'%igm_nl
                outstr += '# Empty frames:                     %d\n'%empty_nl
                outstr += '# Pixel errors:                     %d\n'%n_err
                outstr += 'CPUtime (MM:SS.ms):                 %s'%time_elapsed(start_time)
                print(outstr)

                chunk_n += 1

            # plot corrected frames
            if (plot_rgb or plot_frames) and raw_nl>=max_nl:
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

                plot_igm(igm_xyz)
                plot_glt(glt_xyz,ort_rgb,plot_frames=True,titlestr='')

    if len(glt_xyz) == 0:
        warn('Unable to geocorrect any image pixels')
        return FAILURE

    # if we dropped any frames, update igm_nl
    #if empty_nl!=0:
    #    print('Reduced igm_nl=%d by %d since %d frames dropped'%(igm_nl,empty_nl))
    #    igm_nl = igm_nl-empty_nl
    
    # update/write igm/glt/obs/loc products
    print('Finalizing IGM %s'%pathsplit(igmf)[1])    
    print('           GLT %s'%pathsplit(gltf)[1])
    write_igm(igmf,igm_nl,pb_len,bin_factor,dem_zone,dem_hemi)
    write_glt(gltf,glt_xyz,igm_nl,pb_samp,dt_samp,dem_ulx,dem_uly,ort_ps,
              dem_zone,dem_hemi,img_sl,img_ss,bin_factor,NN_RAD,NN_BUF,
              ps_avg_fn,MIN_PS,MAX_PS,GLT_ROT_SNAP,verbose=verbose)
    print('IGM/GLT generation complete.\n')

    print('Generating OBS %s'%pathsplit(igmf.replace('igm','obs'))[1])
    print('           LOC %s'%pathsplit(igmf.replace('igm','loc'))[1])        
    obsf, locf = generate_obs_loc(igmf,dem,frame_meta,nav.gps_table)
    print('OBS/LOC generation complete.\n')

    print('Generating orthorectified OBS %s'%pathsplit(igmf.replace('igm','obs_ort'))[1])
    generate_obs_ort(obsf,gltf)
    print('Orthorectified OBS generation complete.\n')
    
    # print('Generating Land Mask %s'%pathsplit(igmf.replace('igm','land'))[1])
    # land_hdr = generate_landmask(loc_hdr)
    # print('Land Mask generation complete.\n')

    print('All products successfully generated for raw file:\n%s'%rawf)
    print('\nOutput directory:\n%s\n'%output_path)
    print('Total CPUtime (MM:SS.ms):           %s'%time_elapsed(inittime))
    return SUCCESS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--config_file', help='Path to configuration file (default=%s)'%(relpath(CONFIG_FILE)),
                        type=str, required=False)    
    parser.add_argument('-p','--platform', help='Platform (AVIRIS-NG or PRISM)', 
                        type=str, required=False)
    parser.add_argument('-c','--camera_file', help='Path to camera file for specified platform',
                        type=str, required=False)    
    parser.add_argument('-g','--geoid_file', help='Path to geoid file (default=%s)'%(relpath(GEOID_FILE)),
                        type=str, required=False)    
    parser.add_argument('-d','--dem_prefix', help='DEM path + filename prefix', 
                        type=str, required=False)    
    parser.add_argument('-w','--subset_width', help='DEM subset width in degrees lat/lon',
                        type=float, required=False)
    parser.add_argument('-o','--output_path', help='Path to store output products (default=%s)'%(relpath(OUTPUT_PATH)), 
                        type=str, required=False)
    parser.add_argument('-s','--skiplines', help='Initial lines (raw frames) to skip (default=0)',
                        type=int, required=False)
    parser.add_argument('-n','--numlines', help='Maximum number of lines (raw frames) to process (default=all frames)', 
                        type=int, default=inf, required=False)
    parser.add_argument('--offset_clock', help='Offset frame clock ticks by this value in seconds (default=0)', 
                        type=float, default=0.0, required=False)
    parser.add_argument('-u','--updated_camera', help='Path to updated camera model',
                        type=str, required=False)    

    
    parser.add_argument('raw', help='Path to raw image')
    args = vars(parser.parse_args())

    platform_id   = args['platform']    
    max_nl        = args['numlines']
    img_sl        = args['skiplines']
    subset_width  = args['subset_width']
    camera_file   = args['camera_file']
    geoid_file    = args['geoid_file'] or GEOID_FILE
    config_file   = args['config_file'] or CONFIG_FILE
    output_path   = args['output_path'] or OUTPUT_PATH
    dem_prefix    = args['dem_prefix']
    offset_latlon = []
    offset_clock  = args['offset_clock']
    rawf          = args['raw']
    updated_camera_file = args['updated_camera']

    #  TODO (BDB, 03/26/17): update test mode args 
    test_mode = False
    
    config_file = realpath(config_file)
    if not pathexists(config_file):
        warn('config_file %s does not exist'%config_file)
        sys.exit(FAILURE)

    if not pathexists(output_path):
        warn('output path %s does not exist'%output_path)
        sys.exit(FAILURE)
        
    if test_mode:
        from ortho_tests import *
        
        # in test mode, the first argument is the name of the test case,        
        # and we get the dem_prefix and img_dir via the test_params function
        test_case  = rawf.lower()
        
        if test_case == 'igm':
            sys.exit(test_igm())
            
        params = test_params(test_case)
        if len(params) == 0:
            warn('unrecognized test case'%test_case)
            sys.exit(FAILURE)            
        rawf, test_dem_prefix, offset_latlon = params

        # cmdline dem_prefix overrides test_params prefix
        dem_prefix = dem_prefix or test_dem_prefix

    dem_prefix = dem_prefix or DEM_PREFIX

    if not pathexists(rawf):
        warn('raw file %s does not exist'%rawf)
        sys.exit(FAILURE)

    config = CONFIG(config_file)

    # override config img_sl,max_nl if -s/-n cmdline arguments passed
    config['IMG_SL'] = img_sl or config['IMG_SL']
    config['MAX_NL'] = max_nl or config['MAX_NL']
    if subset_width:
        config['subset_width'] = subset_width*0.5 
    config['UPDATED_CAM_MODEL'] = updated_camera_file

    platform = load_platform(platform_id,imgf=rawf,camf=camera_file)
    if platform is None:
        warn('platform unspecified and cannot be identified from file "%s"'%rawf)
        sys.exit(FAILURE)

    # use sanitized platform_id
    platform_id = platform.platform_id

    # camera file from platform used if not provided
    camera_file = camera_file or platform.camf

    print('pyortho parameters:')
    print('config_file: ',   config_file)
    print('platform:    ',   platform_id)
    print('raw_file:    ',   rawf)
    print('dem_prefix:  ',   dem_prefix)
    print('camera_file: ',   platform.camf)
    print('output_path: ',   output_path)
    print('offset_clock:',   offset_clock)
    print('platform:\n')
    print(platform)

    if len(offset_latlon)!=0:
        print('offset_latlon: ',offset_latlon)
    
    retval = compute_igm_glt(rawf,platform,config,output_path,dem_prefix,
                             offset_clock=offset_clock,
                             offset_latlon=offset_latlon)
    #wait_exit(retval)
    sys.exit(retval)
