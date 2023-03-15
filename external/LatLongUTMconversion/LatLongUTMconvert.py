#!/usr/bin/env python
# command line interface to LatLongUTMconversion.py

import sys, getopt
from LatLongUTMconversion import *    

def deg2dms(dd):
    m,s = divmod(abs(dd)*3600,60)
    d,m = divmod(m,60)
    return (d if dd>=0 else -d),m,s

def main():
    usagestr = 'usage: LatLongUTMconvert.py x y [-u/--utm] [-d/--datum datum] [-z/--zone zone]'
    argv = sys.argv
    x,y = float(argv[1]),float(argv[2])

    argv = argv[3:]
    try:
        options       = ['help','verbose','utm','dms']
        optionsparam  = ['zone']
        optionsabrv   = ''.join([o[0] for o in options])
        optionsabrv  += ''.join([o[0]+':' for o in optionsparam])
        opts, args = getopt.getopt(argv, optionsabrv, options)                                       
    except getopt.error, msg:
        raise RuntimeError('\n'.join([str(msg),usagestr]))

    dms = False
    utm = False
    zone = None
    datum = 23 # 23=wgs84
    for opt, val in opts:
        if opt in ('--help','-h'):
            print __doc__
            return 0
        elif opt in ('--verbose','-v'):
            verbose = True
        elif opt in ('--utm','-u'):
            utm=True
        elif opt in ('--zone','-z'):
            zone = val
        elif opt in ('--dms','-d'):
            dms = True

    if not utm:                
        lat,lon = x,y
        zone,east,north = LLtoUTM(datum, lat, lon, int(zone))
        print east,north,zone[:-1],zone[-1]
    else:
        east,north = x,y
        lat,lon = UTMtoLL(datum, north, east, zone)
        if not dms:
            print lat,lon
        else:
            print deg2dms(lat),deg2dms(lon)

    return 0

if __name__ == '__main__':
    sys.exit(main())
