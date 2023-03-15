from __future__ import division, print_function

import matplotlib
matplotlib.use('TkAgg') # use TkAgg so maximize() works

from ortho_util import *
from ortho_platform import *

# pps/gps and other georeferencing constants
SYNC_MSG    = 33279
NAV_MSG     = 3501
DENOM16     = 32768.0 # 2**15
DENOM32     = 2147483648.0 # 2*31
DENOM64     = 9.22337203685e+18*(2**20)

PPS_COEF    = 65536**asarray([0,1,2,3],dtype=uint64)
PPS_COLS    = 'gpstime,counter,frame_count'.split(',')
GPS_COEF    = 256**asarray([4,5,6,7,0,1,2,3],dtype=uint64)
GPS_COLS    = 'gpstime,lat,lon,alt,pitch,roll,heading'.split(',')

# return array of correct dims when clock2location encounters an error
C2L_ERR     = array([PIX_ERROR_NO_LOC]*(len(GPS_COLS)-1))

def parse_geoid(geoidf):
    geoid_img = envi_open(geoidf+'.hdr',image=geoidf)
    geoid = geoid_img.read_band(0).T
    geoid_mapinfo = envi_mapinfo(geoid_img)
    geoid_dps = geoid_mapinfo['xps']

    return geoid, geoid_dps

def sgsmooth(y,**kwargs):
    from savitzky_golay import savitzky_golay
    winsize=kwargs.pop('winsize',21)
    order=kwargs.pop('order',1)
    deriv=kwargs.pop('deriv',0)
    rate=kwargs.pop('rate',5)
    assert(order<winsize-1)
    ys = savitzky_golay(y, winsize, order, deriv, rate)
    return ys

def medsmooth(y,**kwargs):
    ksize=kwargs.pop('kernel_size',21)
    from scipy.signal import medfilt
    #med = median(y)
    #mad = median(abs(y-med))
    ys = medfilt(y,kernel_size=ksize)
    return ys

def smoothaxis(A,axis=0,**kwargs):
    #print('smoothing!')
    return apply_along_axis(lambda y: sgsmooth(y,**kwargs),axis,A)
    #return apply_along_axis(lambda y: medsmooth(y,**kwargs),axis,A)

def location2clock(lat,lon,pps_table,gps_table):
    #lat,lon = 7.724684,134.620242
    from numpy import argmax
    n_gps = len(gps_table)
    print(n_gps)
    minlat,maxlat = extrema(gps_table[:,1])
    minlon,maxlon = extrema(gps_table[:,2])
    minv = c_[minlat,minlon]
    maxv = c_[maxlat,maxlon]
    dv = maxv-minv
    print(minv,maxv)
    gps_normed = gps_table[:,[1,2]].copy()
    gps_normed = gps_normed - minv
    gps_normed = gps_normed / dv
    llv = (c_[lat,lon]-minv) / dv
    diffv = float32(map(lambda v: (llv-v),gps_normed)).squeeze()
    print(diffv.shape)
    dvec = (diffv[:,0]*diffv[:,0])#.sum(axis=1)
    print(dvec.shape)
    approx_idx = argmax(-dvec)
    pps = gps_table[approx_idx,0]
    idxr = arange(approx_idx-5,approx_idx+5)
    print(pps,lat,lon,approx_idx,diffv[idxr],dvec[idxr])
    print(gps_table[idxr][:,[1,2]])

        
#@nbjit('f8[:](f8,f8[:,:],f8[:,:],i8)',nopython=True)
def clock2location(clock,pps_table,gps_table,verbose=0):
    """
    clock2location(clock,pps_table,gps_table)

    Given clock time, determine location via linear interpolation in
    PPS and GPS tables

    Arguments:
    - clock:       gps clock time (from frame header)
    - pps_table:   [num_pps x 3] table of Precise Positioning Service pulses
                   count=num_pps, layout=[pps clock count]
    - gps_table:   [num_gps x 7] table of PPS-indexed GPS positions
                   count=num_gps (usually 10*num_pps),
                   layout=[pps,lat,lon,alt,pitch,roll,heading]

    Keyword Arguments:
    None

    Returns: 
    - lat,lon,altitude,pitch,roll,heading
    """

    if verbose:
        print('clock:',clock)

    n_gps = len(gps_table)
    n_pps = len(pps_table)
    if n_pps < 2 or n_gps < 2:
        return C2L_ERR

    # exit if our clock entry is outside the pps table
    if clock < pps_table[0,1] or clock > pps_table[-1,1]:
        return C2L_ERR
    
    sortkw = {'side':'right'}
    # given our clock entry, lerp the nearest pps_table entries
    approx_idx = searchsorted(pps_table[:,1],clock,**sortkw)
    if approx_idx < n_pps:
        pps_lower = pps_table[approx_idx-1,:2]
        pps_upper = pps_table[approx_idx,:2]
        clock_delta = (clock-pps_lower[1]) / (pps_upper[1]-pps_lower[1])
        pps = (1.0-clock_delta)*pps_lower[0] + clock_delta*pps_upper[0]
    elif approx_idx == n_pps:
        pps = pps_table[-1,0]

    if verbose:
        print('pps(clock):',pps)
        print('pps_lower:',pps_lower)
        print('pps_upper:',pps_upper)

    # exit if our pps entry is outside the gps table
    if pps < gps_table[0,0] or pps > gps_table[-1,0]:
        if verbose:
            print('pps entry',pps,'outside valid gps range',
                  gps_table[[0,-1],0])
        return C2L_ERR

    # now lerp the gps entry given the interpolated pps value
    approx_idx = searchsorted(gps_table[:,0],pps,**sortkw)
    if approx_idx < n_gps:
        gps_lower = gps_table[approx_idx-1,:]
        gps_upper = gps_table[approx_idx,:]

        pps_delta = (pps-gps_lower[0]) / (gps_upper[0]-gps_lower[0])
        gps = (1.0-pps_delta)*gps_lower[1:7] + pps_delta*gps_upper[1:7]
    elif approx_idx == n_gps:
        gps = gps_table[-1,1:7]

    if verbose:
        print('gps(pps):',gps)
        print('gps_lower:',gps_lower)
        print('gps_upper:',gps_upper)
    
    return gps

def format_cmigits_words(words,scale,verbose=0):
    # map words into an nbits-sized bitmask in reverse bsig order
    nw    = len(words)
    nbits = nw*16
    bits  = zeros(nbits,dtype=int8)
    mask  = uint32(2**arange(16))    
    bsig  = [1,0,3,2] if nw==4 else [1,0]
    for i in range(nw):
        bits[i*16:(i+1)*16] = int8((mask & uint32(words[bsig[-(i+1)]]))>0)
    
    # flip MSB and accumulate integer value
    bits[-1] = -bits[-1]
    int_exp  = scale-(nbits-1)
    int_out  = (float64(bits)*(power(2.,arange(nbits)))).sum()
    value    = int_out*power(2.,int_exp)
    
    if verbose:
        print('raw_words',words)
        print('bits',bits)
        print('n_elements(bits)',len(bits))
        print('integer',int_out)
        print('exponent',int_exp)
        print('value',value)

    return value

def read_pps(pps_path, msg_words, start_line=0, num_lines=99999, smooth=False):
    """
    read_pps(pps_path, msg_words, start_line=0) 
    
    Arguments:
    - pps_path: path to pps table file
    - msg_words: number of words in message (13 or 14)
    
    Keyword Arguments:
    - start_line: start line = number of lines to skip (default=0)
    - num_lines: maximum number of lines to read beyond 'start_line' (default=99999)
    
    Returns:
    - pps_table: msg_read x 3 pps table file
    - msg_read: number of pps messages read
    """
    
    msg_read = 0
    time_table = array([],dtype=double)
    if not pathexists(pps_path):
        warn('pps file %s not found!'%pps_path)
        return time_table

    # minimum possible size to get a valid pps frame = msg_words+2 bytes
    pps_size = getsize(pps_path)
    if pps_size < msg_words+2: 
        warn('pps file %s size less than minimum (%d) bytes'%(pps_path,msg_words+2))
        return time_table

    msg_bytes = 2*msg_words
    nr = floor(pps_size/msg_bytes)
    nl_max = min(nr,num_lines)

    with open(pps_path,'r') as f:
        # traverse file with sliding window to find sync message
        bytec = 0
        while fromfile(f,count=1,dtype='<u2') != SYNC_MSG:
            if bytec >= pps_size:
                warn('PPS file contains no sync messages')
                return time_table
            # back up one byte to search every 2-byte string for sync_msg
            f.seek(-1,SEEK_CUR) 
            bytec += 1

        # found first sync_msg, back up to sync position
        f.seek(-2,SEEK_CUR)

        if start_line > 0:
            f.seek(start_line*msg_bytes,SEEK_CUR)
            if f.tell() >= pps_size:
                return time_table
        
        # read messages until we get a sync header or an empty buffer
        time_table = []
        while msg_read < nl_max:
            if f.tell()+msg_bytes > pps_size: # truncated file, return what we have
                warn('PPS table contains truncated messages')
                break
            buf=fromfile(f,count=msg_words,dtype='<u2')
            if len(buf) < msg_words:
                warn('PPS message %d truncated'%(msg_read+1))
                break
            elif buf[0] != SYNC_MSG:
                warn('Expected PPS synchronization word not found')
                break            
            elif int16(buf[:4]).sum()+buf[4] != 0:
                warn('PPS message %d checksum failed'%(msg_read+1))
                break

            #gpstime     = dot(float64(buf[[7,8,5,6]]),PPS_COEF) / DENOM64
            #gpstime     = dot(uint32(buf[[6,5,8,7]]),uint32(PPS_COEF)) #/ DENOM64
            gpsdata = int64(buf[5:9]) #buf[[5,6,7,8]]
            gpstime = format_cmigits_words(gpsdata,20)
            #count      = uint32(bitwise_and(uint32(g_14bit_mask), buf[9].astype(int16)))
            count   = float64(extract_fc(buf[9])) #int64(bitwise_and(uint32(g_14bit_mask), uint32(buf[9])))
            frtc    = float64(format_clock_words(buf[10],buf[11])) #int64(uint64(buf[10])*65536)+uint32(buf[11])
            l_time  = c_[gpstime,frtc,count]
            
            if len(time_table)>0:
                time_table=r_[time_table,l_time]
            else:
                time_table=l_time
            msg_read+=1
            
    if smooth:
        time_table = smoothaxis(time_table,axis=0)
            
    return time_table

def read_gps(gps_path, start_line=0, num_lines=99999, smooth=False):
    """
    read_gps(gps_path, start_line=0, num_lines=99999)

    Reads table of GPS values from file, optionally skipping 'start_line' initial entries
    
    Arguments:
    - gps_path: path to gps table file
    
    Keyword Arguments:
    - start_line: number of lines to skip (default=0)
    - num_lines: maximum number of lines to read beyond 'start_line' (default=99999)
    
    Returns:
    - gps_table: msg_read x 7 gps table
    - msg_read: number of lines read
    """
    
    msg_read = 0
    msg_skipped = 0
    locations = array([],dtype=double)
    velocities = array([],dtype=double)
    if not pathexists(gps_path):
        warn('GPS file %s not found!'%gps_path)
        return locations,velocities

    gps_size = getsize(gps_path)
    if gps_size == 0: 
        warn('GPS file %s empty!'%gps_path)
        return locations,velocities

    file_done=False
    with open(gps_path,'r') as f:
	###heading = 0.0
        while not file_done:
            header=fromfile(f,count=5,dtype='<u2')
            if len(header)<5:
                return locations,velocities

            if header[0] != SYNC_MSG:
                f.seek(-9,SEEK_CUR) # backup by (5*2)-1 to read even/odd byte msgs
                continue            
            
            msg_bytes=2*(header[2]+1)
            if f.tell()+msg_bytes > gps_size:
                # truncated message, return everything up until now
                warn('Truncated GPS message encountered, returning valid messages')
                return locations,velocities
            if header[1] == NAV_MSG:
		import pdb
                if msg_skipped < start_line:
                    f.seek(msg_bytes,SEEK_CUR)
                    msg_skipped+=1
                else:                    
                    gpsdata = fromfile(f,count=4,dtype='<u2')
                    gpstime = format_cmigits_words(gpsdata,20)

                    posdata = fromfile(f,count=6,dtype='<u2')
		    #try:
    			#heading
		    #except NameError:
    			#pass
		    #else:
    			#heading_old = heading
			#pitch_old = pitch
			#roll_old = roll
			#alt_old = alt
			#lat_old = lat
			#lon_old = lon
                    lat     = format_cmigits_words(posdata[0:2],0)*180.0
                    lon     = format_cmigits_words(posdata[2:4],0)*180.0
                    alt     = format_cmigits_words(posdata[4:6],15)

                    vecdata = fromfile(f,count=6,dtype='<u2')
                    vnorth  = format_cmigits_words(vecdata[0:2],10)
                    veast   = format_cmigits_words(vecdata[2:4],10)
                    vup     = format_cmigits_words(vecdata[4:6],10)

                    ortdata = fromfile(f,count=6,dtype='<u2')
                    pitch   = format_cmigits_words(ortdata[0:2],0)*180.0
                    roll    = format_cmigits_words(ortdata[2:4],0)*180.0
                    #heading = 63.08435255661607#-110.80147756263614#
                    heading = format_cmigits_words(ortdata[4:6],0)*180.0
		    #try:
    			#heading_old
		    #except NameError:
    			#pass
		    #else:
		        #print('Error found on nav continuity. Correcting.')
		        #if (heading_old/heading) < 0.90 or (heading_old/heading) > 1.1:
			#    heading = heading_old
		        #if (pitch_old/pitch) < 0.90 or (pitch_old/pitch) > 1.1:
			#    pitch = pitch_old
		        #if (roll_old/roll) < 0.90 or (roll_old/roll) > 1.1:
			#    roll = roll_old
		        #if (alt_old/alt) < 0.90 or (alt_old/alt) > 1.1:
			#    alt = alt_old
		        #if (lat_old/lat) < 0.90 or (lat_old/lat) > 1.1:
			#    lat = lat_old
		        #if (lon_old/lon) < 0.90 or (lon_old/lon) > 1.1:
			#    lon = lon_old
		    #pdb.set_trace()
		    #print(heading)
                    #gpstime=dot(fromfile(f,count=8,dtype='<u1'),GPS_COEF) / DENOM64
                    #f.seek(-8,SEEK_CUR) # back up by (5*2)-1 to read even/odd byte msgs

                    #navdata = fromfile(f,count=9,dtype='<i4')/DENOM32
                    #lat,lon,pitch,roll,heading = 180.0*navdata[[0,1,6,7,8]]
                    #alt,vnorth,veast,vup = DENOM16*navdata[[2,3,4,5]]
                    #print(lat,lon,pitch,roll,heading,alt),raw_input()
                    #lat,lon=fromfile(f,count=2,dtype='<i4') / DENOM32 * 180.0
                    #alt=fromfile(f,count=1,dtype='<i4') / DENOM32 * 32768
                    #vnorth,veast,vup=fromfile(f,count=3,dtype='<i4') / DENOM32 * 32768
                    #pitch,roll,heading=fromfile(f,count=3,dtype='<i4') / DENOM32 * 180                    
                    data_checksum=fromfile(f,count=1,dtype='<u2')
                    #print('data_checksum: "%s"'%str((data_checksum)))
                    if len(data_checksum)==0:
                        file_done=True
                    elif (abs(float32([lat,lon,pitch,roll,heading])) <= 180.0).all():
                        l_location = float64(c_[gpstime,lat,lon,alt,pitch,roll,heading])
                        l_velocity = float64(c_[vnorth,veast,vup])
                        if msg_read != 0:
                            locations = r_[locations,l_location]
                            velocities = r_[velocities,l_velocity]
                        else:
                            locations = l_location
                            velocities = l_velocity
                        msg_read+=1
                        if msg_read >= num_lines:
                            return locations,velocities
                    else:
                        warn('bad GPS data at msg %d'%msg_read)
                        return locations,velocities
            elif msg_bytes > 0:
                f.seek(msg_bytes,SEEK_CUR)

    if smooth:
        locations = smoothaxis(locations,axis=0)
              
    return locations,velocities

def validate_pps_gps(pps_table,gps_table,**kwargs):
    verbose = kwargs.pop('verbose',0)
    npps = len(pps_table)
    ngps = len(gps_table)

    pps_time,pps_clock,frame_count = pps_table.T
    gps_time = gps_table[:,0]

    count_diff = abs(diff(diff(frame_count)%((2**16))))
    count_max = count_diff.max()
    if count_max > 1:
        warn('adjacent PPS frame counts differ by > 1 (max diff=%d)'%count_max)
        if verbose>1:
            bad_diffs = where(count_diff>1)[0]
            print('frame diffs:')
            print(bad_diffs)
            print('frame counts:')
            print([count_diff[i:i+3] for i in bad_diffs])

    clock_diff = abs(diff(diff(pps_clock)))
    clock_max = clock_diff.max()
    if clock_max > 1:
        warn('adjacent PPS clock ticks differ by > 1 tick (max diff=%d)'%clock_max)
        if verbose>1:
            print('clock ticks:')
            print(pps_clock)

    gps_min,pps_min = gps_time[0],pps_time[0]
    gps_max,pps_max = gps_time[-1],pps_time[-1]
    min_time = min(pps_min,gps_min)
    max_time = max(gps_max,pps_max)
    min_clock = kwargs.pop('min_clock',inf)
    max_clock = kwargs.pop('max_clock',-inf)

    #print(min_clock,pps_clock[0])
    #print(max_clock,pps_clock[-1])
    if min_clock < pps_clock[0]:
        warn('PPS table missing heading entries')
    if max_clock > pps_clock[-1]:
        warn('PPS table missing trailing entries')
    
    pps_below = (pps_time<gps_time[0]).sum()
    pps_above = (pps_time>gps_time[-1]).sum()
    gps_below = (gps_time<pps_time[0]).sum()
    gps_above = (gps_time>pps_time[-1]).sum() 

    # 1 leading entry is acceptable
    if pps_below>1:
        warn('PPS table contains %d extra leading entries'%pps_below)
    if pps_above>0:
        warn('PPS table contains %d extra trailing entries'%pps_above)

    if gps_below>0:
        warn('GPS table contains %d entries < PPS minimum'%gps_below)

    # need at least 1 gps time > the max pps time to get location
    if gps_above==0:
        warn('GPS table does not contain entries for the %d largest PPS clock ticks'%pps_above)

def summarize_pps_gps(pps_table,gps_table,gps_velo,doplot=False,saveplots=False,
                      saveprefix=None,plot_velocity=False):
    gps_ticks = gps_table[:,0]
    pps_ticks = pps_table[:,0]

    print()
    print('GPS table')    
    print('total lines:    ', gps_table.shape[0])
    print('latitude range: ', extrema(gps_table[:,1]))
    print('longitude range:', extrema(gps_table[:,2]))
    print('altitude range: ', extrema(gps_table[:,3]))
    print('start location: ', gps_table[0,[1,2,3]])
    print('end location:   ', gps_table[-1,[1,2,3]])
    print('start tick:     ', gps_table[0,0])
    print('end tick:       ', gps_table[-1,0])
    print('tick range:     ', extrema(gps_ticks))
    print()
    
    print('PPS table')
    print('total lines:    ', pps_table.shape[0])
    print('start counter:  ', pps_table[0,1])
    print('end counter:    ', pps_table[-1,1])
    print('counter range:  ', extrema(pps_table[:,1]))
    print('start fc:       ', pps_table[0,-1])
    print('end fc:         ', pps_table[-1,-1])
    print('fc range:       ', extrema(pps_table[:,-1]))
    print('start tick:     ', pps_table[0,0])
    print('end tick:       ', pps_table[-1,0])
    print('tick range:     ', extrema(pps_ticks))
    print()
    
    if doplot:
        gpsprefix = 'gps' if saveprefix is None else saveprefix+'_gps'
        ppsprefix = 'pps' if saveprefix is None else saveprefix+'_pps'
        
        from numpy import cumsum, vectorize
        print("Plotting gps_table locations (%d entries)"%gps_table.shape[0])
        pos_fig,pos_ax = pl.subplots(3,1,sharex=True,sharey=False,num=1)
        pos_ax[0].plot(gps_ticks,gps_table[:,1])
        pos_ax[0].set_ylabel('lat')
        
        pos_ax[1].plot(gps_ticks,gps_table[:,2])
        pos_ax[1].set_ylabel('lon')
        
        pos_ax[2].plot(gps_ticks,gps_table[:,3])
        pos_ax[2].set_ylabel('alt')
        pos_ax[2].set_xlabel('clock')
        pl.suptitle('GPS position')

        if saveplots:
            posfig=gpsprefix+'_pos.pdf'
            pl.savefig(posfig)
            print('saved',posfig)                        
        
        ort_fig,ort_ax = pl.subplots(3,1,sharex=True,sharey=False,num=2)
        ort_ax[0].plot(gps_ticks,gps_table[:,4])
        ort_ax[0].set_ylabel('pitch')
        
        ort_ax[1].plot(gps_ticks,gps_table[:,5])
        ort_ax[1].set_ylabel('roll')
        
        ort_ax[2].plot(gps_ticks,gps_table[:,6])
        ort_ax[2].set_ylabel('heading')
        ort_ax[2].set_xlabel('clock')
        pl.suptitle('GPS orientation')

        if saveplots:            
            orientfig=gpsprefix+'_orient.pdf'
            pl.savefig(orientfig)
            print('saved',orientfig)            
        
        if plot_velocity:
            vel_fig,vel_ax = pl.subplots(4,1,sharex=True,sharey=False,num=3)
            vel_ax[0].plot(gps_ticks,gps_velo[:,0])
            vel_ax[0].set_ylabel('vx')

            vel_ax[1].plot(gps_ticks,gps_velo[:,1])
            vel_ax[1].set_ylabel('vy')

            vel_ax[2].plot(gps_ticks,gps_velo[:,2])
            vel_ax[2].set_ylabel('vz')

            vel_ax[3].plot(gps_ticks,sqrt((gps_velo*gps_velo).sum(axis=1)))
            vel_ax[3].set_ylabel('total')        
            vel_ax[3].set_xlabel('clock')
            pl.suptitle('GPS velocity')
            if saveplots:
                maximize()
                velofig=gpsprefix+'_velo.pdf'
                pl.savefig(velofig)
                print('saved',velofig)

        # render pps table counter and frame counter
        print("Plotting pps_table counters (%d entries)"%pps_table.shape[0])
                
        pps_fig,pps_ax = pl.subplots(2,1,sharex=True,sharey=False,num=4)
        pps_ax[0].plot(pps_ticks,pps_table[:, 1])
        pps_ax[0].set_ylabel('counter')
        
        pps_ax[1].plot(pps_ticks,pps_table[:,-1])
        pps_ax[1].set_ylabel('frame counter')
        pps_ax[1].set_xlabel('clock')
        pl.suptitle('PPS counters')
        if saveplots:
            maximize()
            pl.savefig(ppsprefix+'.pdf')
        
        if 0:
            loc_diffs = -diff(pps_ticks)/2.0
            loc_ticks = pps_ticks[1:-1]+loc_diffs[1:] # shift + drop last entry for plotting
            vfunc = lambda c: clock2location(c[0],pps_table,gps_table,verbose=1)
            pps_loc = apply_along_axis(vfunc,1,pps_ticks.reshape([-1,1])+0.01)
            pps_ticks = pps_ticks[1:-1]

            loc_fig,loc_ax = pl.subplots(3,1,sharex=True,sharey=False,num=5)
            #for i in range(3):
            #    minv,maxv=extrema(pps_loc[:,i])
            #    loc_ax[i].axvline(348487076,ymin=minv,ymax=maxv)        
            loc_ax[0].plot(pps_ticks,pps_loc[:,0])
            loc_ax[0].set_ylabel('lat')

            loc_ax[1].plot(pps_ticks,pps_loc[:,1])
            loc_ax[1].set_ylabel('lon')

            loc_ax[2].plot(pps_ticks,pps_loc[:,2])
            loc_ax[2].set_ylabel('alt')
            loc_ax[2].set_xlabel('pps ticks')

            pl.suptitle('PPS Locations')

            ort_fig,ort_ax = pl.subplots(3,1,sharex=True,sharey=False,num=6)
            #for i in range(3):
            #    minv,maxv=extrema(pps_loc[:,i+3])
            #    ort_ax[i].axvline(348487076,ymin=minv,ymax=maxv)        
            ort_ax[0].plot(pps_ticks,pps_loc[:,3])
            ort_ax[0].set_ylabel('pitch')

            ort_ax[1].plot(pps_ticks,pps_loc[:,4])
            ort_ax[1].set_ylabel('roll')

            ort_ax[2].plot(pps_ticks,pps_loc[:,5])
            ort_ax[2].set_ylabel('heading')
            ort_ax[2].set_xlabel('pps ticks')

            pl.suptitle('PPS Orientation')                      


class NAV():
    def __init__(self,platform,ppsf,gpsf,geoidf,table_updates='reload',
                 check_size=False,offset_clock=0,verbose=0):
        """
        NAV(ppsf,gpsf,pps_msg,geoidf,verbose=0)

        Initializes ortho navigation parameters 
        (typically pps,gps,geoid and camera)

        Arguments:
        - platform: platform class containing camera/world model
        - ppsf: path to pps table file
        - gpsf: path to gps table file
        - geoidf: path to geoid file

        Keyword Arguments:
        - offset_ticks: shift incoming clock ticks by this value (in seconds)
        - verbose: generate verbose output (default=0)

        Returns:
        geoids, camera model 
        """
        
        self.initialized   = False
        self.verbose       = verbose
        
        if not pathexists(ppsf):
            warn('pps file %s not found!'%ppsf)
            return

        if not pathexists(gpsf):
            warn('WARNING: gps file %s not found!'%gpsf)
            return

        if not pathexists(geoidf):
            warn('geoid file %s not found!'%geoidf)
            return

        if platform.loadcamera() == FAILURE:
            warn('unlable to load platform camera model')
            return
        
        self.ppsf          = ppsf
        self.gpsf          = gpsf
        self.pps_msg       = platform.PPS_MSG
        self.pps_table     = []
        self.gps_table     = []
        self.gps_velo      = []
        self.gps_nl        = 0
        self.pps_nl        = 0
        self.gps_size_prev = 0
        self.pps_size_prev = 0
        self.table_updates = table_updates
        self.clock_offset  = offset_clock
        self.clock_freq    = 10000 # 10000hz sampling frequency
        self.check_size    = check_size

        geoid_data         = parse_geoid(geoidf)
        self.geoid         = geoid_data[0]
        self.geoid_dps     = geoid_data[1]

        self.minlat        = None
        self.maxlat        = None
        self.minlon        = None
        self.maxlon        = None
        
        self.platform      = platform
        self.initialized   = True

        
    def collect_tables(self,pps_nappend=10,**kwargs):
        """
        init / dynamically update pps/gps tables as they're written to disk
        """
        update_mode = kwargs.pop('update_mode',self.table_updates)
        check_size = kwargs.pop('check_size',self.check_size)
        smooth_gps = kwargs.pop('smooth_gps',False)        
        smooth_pps = kwargs.pop('smooth_pps',False)
        
        statemsg = 'Initializing'
        if update_mode == None: # don't update, just return tables
            return self.pps_table, self.gps_table
        
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
                return self.pps_table, self.gps_table

        if update_mode=='reload': # reload tables from scratch
            if self.verbose:
                if self.pps_nl>0 and self.gps_nl>0:
                    statemsg = 'Reloading' 
                print('%s PPS/GPS tables'%statemsg)

            try:
                pps_table = read_pps(self.ppsf,self.pps_msg,smooth=smooth_pps)
                self.pps_table = pps_table
                self.pps_nl    = pps_table.shape[0]
            except Exception as e:
                warn('An unexpected error occurred updating the PPS table:', e)
                pass
            
            try:
                # parse 10 gps lines for each pps line
                gps_table,gps_velo = read_gps(self.gpsf,smooth=smooth_gps)
                self.gps_table = gps_table
                self.gps_velo  = gps_velo
                self.gps_nl    = gps_table.shape[0]
            except Exception as e:
                warn('An unexpected error occurred updating the GPS table:', e)
                pass                    
            
        elif update_mode=='append': # parse  tables from file            
            if verbose:
                if self.pps_nl>0 and self.gps_nl>0:
                   statemsg = 'Appending new measuments to'
                print('%s PPS/GPS tables'%msg)                
            try:
                pps_chunk = read_pps(self.ppsf,self.pps_msg,start_line=self.pps_nl,
                                     num_lines=pps_nappend,smooth=smooth_pps)
                pps_nc = pps_chunk.shape[0]
                if pps_nc > 0:
                    if self.pps_nl > 0:
                        self.pps_table = r_[self.pps_table,pps_chunk]
                    else:
                        self.pps_table = pps_chunk
                    self.pps_nl += pps_nc
                  
            except Exception as e:
                warn('An unexpected error occurred updating the PPS table:', e)
                pass

            try:
                # parse 10 gps lines for each pps line
                gps_chunk,gps_chunk_velo = read_gps(self.gpsf,start_line=self.gps_nl,
                                                    num_lines=10*pps_nappend,
                                                    smooth=smooth_gps)
                gps_nc = gps_chunk.shape[0]
                if gps_nc > 0:
                    if self.gps_nl > 0:
                        self.gps_table = r_[self.gps_table,gps_chunk]
                        self.gps_velo = r_[self.gps_velo,gps_chunk_velo]
                    else:
                        self.gps_table = gps_chunk
                        self.gps_velo = gps_chunk_velo
                    self.gps_nl += gps_nc                
            except Exception as e:
                warn('An unexpected error occurred updating the GPS table:', e)
                pass    
        else:
            warn('Unknown PPS/GPS table update mode "%s"'%update_mode)

        if self.gps_nl>0:
            self.minlat,self.maxlat = extrema(self.gps_table[:,1])
            self.minlon,self.maxlon = extrema(self.gps_table[:,2])

        validate_pps_gps(self.pps_table,self.gps_table)
        if self.verbose > 1:
            summarize_pps_gps(pps_table,gps_table,gps_velo)
            
        return self.pps_table, self.gps_table

    def clock2location(self,clock,**kwargs):
        # wrapper for clock2location function that updates pps/gps tables
        pps_table,gps_table = self.collect_tables(**kwargs)
        if pps_table.shape[0]==0:
            warn('PPS table empty or corrupt, unable to compute surface coordinates')
            return EMPTY
        
        elif gps_table.shape[0]==0:
            warn('GPS table empty or corrupt, unable to compute surface coordinates')
            return EMPTY

        clock_off = clock + (self.clock_offset*self.clock_freq)
        return clock2location(clock_off,pps_table,gps_table,**kwargs)

def maximize():
    mng = pl.get_current_fig_manager()
    backend = matplotlib.get_backend().lower()
    if backend.startswith('wx'):        
        mng.frame.Maximize(True)
    elif backend.startswith('qt4'):
        mng.window.showMaximized()
    elif backend.startswith('qt5'):
        mng.window.showMaximized()        
    elif backend.startswith('tk'):
        #mng.window.state('zoomed')
        #mng.full_screen_toggle()
        mng.resize(*mng.window.maxsize())
    
if __name__ == '__main__':
    import pylab as pl
    import argparse
    pl.ioff()
    set_printoptions(suppress=True,precision=6)

    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--frame_range', help='Frame range (default=(0,-1))', 
                        type=int, nargs=2, default=(0,-1), metavar='FRAME')
    parser.add_argument('-o','--outdir', help="Figure outdir (optional, default='.')",
                        type=str, default='.', metavar='OUTDIR')    
    parser.add_argument('--plot_tables', help='Plot PPS/GPS table coordinates and clock ticks', 
                        action='store_true')
    parser.add_argument('--plot_frames', help='Plot frame coordinates and clock ticks', 
                        action='store_true')
    parser.add_argument('-v','--verbose', help='Verbose output ([0,1,2])',
                        action='store_true')
    
    parser.add_argument('rawf', help='Path to raw image file')
        
    args        = parser.parse_args()
    
    imgf        = args.rawf
    outdir      = args.outdir
    frange      = args.frame_range
    verbose     = args.verbose
    
    plot_tables = args.plot_tables
    plot_frames = args.plot_frames

    
    ppsf = splitext(imgf)[0].replace('_raw','_pps')
    gpsf = splitext(imgf)[0].replace('_raw','_gps')
    platform_id = identify_platform(imgf)

    if not pathexists(outdir):
        print('outdir "%s" does not exist, exiting'%outdir)
        sys.exit(1)
    
    outpre = pathjoin(outdir,pathsplit(imgf)[1].replace('_raw',''))

    outsuf = '.pdf'
    if frange != (0,-1):
        outsuf = '_%d_%d.pdf'%(frange[0],frange[1])
    
    print('Loading PPS/GPS tables for %s, platform %s'%(outpre,platform_id))    

    platform = load_platform(platform_id)
    smooth_pps = False
    smooth_gps = False
    gps,gps_velo = read_gps(gpsf,smooth=smooth_gps)    
    pps = read_pps(ppsf,platform.PPS_MSG,smooth=smooth_pps)

    #lonq,latq = 134.62607,7.71900
    #print(latq,lonq,location2clock(latq,lonq,pps,gps))
    #raw_input()
    pps_nr = pps.shape[0]
    gps_nr = gps.shape[0]

    validate_pps_gps(pps,gps,verbose=0)

    if not pathexists(imgf):
        print(imgf,'not found')
        sys.exit(0)
        
    img = envi_open(imgf+'.hdr',image=imgf)
    img_nl = img.shape[0]

    fi,fj=int(frange[0]),int(frange[1])
    fj = min(fj,img_nl) if fj>0 else img_nl+(fj+1)

    assert(fi<fj)

    print('\nCollecting frames %d through %d\n'%(fi,fj))
    fn = (fj-fi)
    fdelt = max(1,int(fn/10))
    fclock,floc,fobc = [],[],[]
    i = fi
    while i < fj:
        if i % fdelt == 0:
            print('Frame %d (of %d, %.2f%% complete)'%(i+1,fn,(i*100.0)/fn))
        frameI,nrI,obc_startI,obcvI = read_frames_meta(imgf, platform,
                                                       start_line=i,
                                                       num_lines=fdelt,
                                                       return_obcv=True,
                                                       transition_warnings=False,
                                                       verbose=0)
        # read_frames will return < fdelt frames when we exit science region
        iend = i+len(frameI)
        
        locI = float32([clock2location(frameIj,pps,gps,verbose=0)
                        for frameIj in frameI[:,0]])
        if len(fclock)>0:
            fclock = r_[fclock,frameI[:,0]]
            floc =   r_[floc,locI]
            fobc =   r_[fobc,obcvI]
        else:
            fclock = frameI[:,0]
            floc =   locI
            fobc =   obcvI
        #fclock.append(frameI[0,0])
        #floc.append(locI)
        #fobc.append(obcvI[0])

        if verbose:
            obcv_strI = [OBC_STATUS_MSG.get(obcvIj,'OBC_%d'%obcvIj)
                         for obcvIj in obcvI]

            print('frame[%d] type:'%i,obcv_strI)
            print('clock,count:',frameI[0])
            #print('obc_start,nr:',obc_startI,nrI)
            print('lat,lon,alt:',locI[:3])
            print('pitch,roll,heading:',locI[3:6])
            print()

            print('i',i,'iend',iend,'img_nl',img_nl,'fdelt',fdelt,
                  'len(frameI)',len(frameI),
                  'len(fclock)',len(fclock),
                  'locI.shape',locI.shape)
            
        i = iend

    print('done, collected %d frames\n'%len(fclock))

    summarize_pps_gps(pps,gps,gps_velo,doplot=plot_tables,saveplots=True,
                      saveprefix=outpre)

    ptile = 0.01
    fclock = float32(fclock)
    if plot_frames:
        print("Plotting frames (%d entries)"%fclock.shape[0])
        from numpy import nanpercentile
        fclocki = int32(fclock)
        floc = float32(floc)
        fobcv = int32(fobc)
        ftrans = diff(r_[fobcv,fobcv[-1]])
        ftransi = where(ftrans!=0)[0]
        fmask = fobcv == OBC_SCIENCE
        fsci = where(fmask)[0]
        minsci = [ftj for ftj in ftransi if ((fobcv[ftj]==OBC_DARK1) and
                                             (fobcv[ftj+1]==OBC_SCIENCE))]
        if len(minsci)==0 and fobcv[0]==OBC_SCIENCE:
            minsci = [0]
        minsci = minsci[0]+fi
        maxsci = fsci.max()+fi+1
        
        print('Plotting',len(fclock),'frames')
        fig,ax = pl.subplots(3,1,sharex=True,sharey=False)
        ax[0].plot(fclocki,floc[:,0]); ax[0].set_ylabel('lat')
        ax[1].plot(fclocki,floc[:,1]); ax[1].set_ylabel('lon')
        ax[2].plot(fclocki,floc[:,2]); ax[2].set_ylabel('alt')
        ax[2].set_xlabel('clock')
        pl.suptitle('Position\nSelected frames [%d,%d], science frames [%d,%d]'%(fi,fj,
                                                                                 minsci,
                                                                                 maxsci))        
        pig,bx = pl.subplots(3,1,sharex=True,sharey=False)
        bx[0].plot(fclocki,floc[:,3]); bx[0].set_ylabel('pitch'); 
        bx[1].plot(fclocki,floc[:,4]); bx[1].set_ylabel('roll');  
        bx[2].plot(fclocki,floc[:,5]); bx[2].set_ylabel('heading')
        bx[2].set_xlabel('clock')
        
        clockmin,clockmax = extrema(fclocki)
        clockdelt = (clockmax-clockmin)*0.005
        clockmin,clockmax = clockmin-clockdelt,clockmax+clockdelt
        ax[0].set_xlim(clockmin,clockmax)
        bx[0].set_xlim(clockmin,clockmax)
        for i in range(6):
            (plotax,ploti) = (ax,i) if i<3 else (bx,i-3)
            ymin = nanpercentile(floc[fmask,i],q=ptile*100)
            ymax = nanpercentile(floc[fmask,i],q=(1-ptile)*100)
            ydelt = (ymax-ymin)*0.005
            ymin,ymax = ymin-ydelt,ymax+ydelt
            plotax[ploti].set_ylim(ymin,ymax)
            plotax[ploti].axvline(fclocki[0],color='k',ls='--',label='Start/End')
            plotax[ploti].axvline(fclocki[-1],color='k',ls='--')
            flabj,ftrij = [],[]
            fseenj = set([])
            for ftri,ftj in enumerate(ftransi):
                fclockij = fclocki[ftj]
                if (((fobcv[ftj]==OBC_DARK1) and (fobcv[ftj+1]==OBC_SCIENCE)) or
                    ((fobcv[ftj+1]==OBC_DARK2) and (fobcv[ftj]==OBC_SCIENCE))):
                    # initial dark -> science transition or final science -> dark
                    fcolorj = 'g'
                    flabelj = 'Science'
                elif ftrans[ftj]>0:                    
                    fcolorj = 'b'
                    flabelj = 'Valid'
                else:
                    fcolorj = 'r'
                    flabelj = 'Invalid'
                if flabelj in fseenj:
                    flabelj = None
                else:
                    fseenj.add(flabelj)
                fplotj = plotax[ploti].axvline(fclockij,color=fcolorj,ls='--',
                                               label=flabelj)

            if ploti==0:
                plotax[ploti].legend(fontsize=10,loc='upper right')
                    
        pl.suptitle('Orientation\nSelected frames [%d,%d], science frames [%d,%d]'%(fi,
                                                                          fj,
                                                                          minsci,
                                                                          maxsci))

        gpsfigf = outpre+'_raw_pos'+outsuf
        pl.figure(fig.number)
        maximize()
        pl.savefig(gpsfigf)
        print('saved',gpsfigf)
        ppsfigf = outpre+'_raw_orient'+outsuf
        pl.figure(pig.number)
        maximize()
        pl.savefig(ppsfigf)
        print('saved',ppsfigf)
        
        #pl.show()
        
