#!/usr/bin/env bash
TILEDIR=/mnt/usb1/india_srtm_1arcsec/SRTM_tiles
OUTFILE=india_srtm_1arcsec
NODATA=-9999
#gdal_merge=/Users/bbue/Research/AVIRISNG/range/ort/python/external/gdal_merge.py
gdal_merge=$(which gdal_merge.py)
$gdal_merge -of ENVI -init $NODATA -n $NODATA -o $OUTFILE $TILEDIR/*.hgt 
