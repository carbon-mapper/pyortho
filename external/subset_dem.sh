#!/bin/bash

gdalhome=/usr/local/bin
gdaltranslate=$gdalhome/gdal_translate
gdalwarp=$gdalhome/gdalwarp
gdalinfo=$gdalhome/gdalinfo
gdalsrsinfo=$gdalhome/gdalsrsinfo

python='python' # use anaconda path here 
ll2utm="$python LatLongUTMconvert.py"

# input dem in wgs84 lat/lon coords
dem_dir=/Volumes/Space/Data/dem/conus_ned_1arcsec
dem_file=conus_ned_1arcsec 
dem_path=$dem_dir/$dem_file

# the following should output "WGS84" for the projection
#$gdalsrsinfo /Volumes/Space/Data/dem/conus_ned_1arcsec/conus_ned_1arcsec

# utm zone to use for extracted dem
sub_zone=11

# output subset DEM 
sub_dir=/Users/bbue/Desktop/
sub_file=${dem_file}_sub_utm${sub_zone}
sub_path=$sub_dir/$sub_file

# delete the output file if it already exists
rm -f $sub_path

# lat/lon extent to extract from dem_file ul_lon ul_lat lr_lon lr_lat (ul=upper left, lr=lower right)
ul_lon="-117.359214"
ul_lat="33.974528"
lr_lon="-117.333847"
lr_lat="33.965533"
sub_extent="$ul_lon $ul_lat $lr_lon $lr_lat"

# temporary file to save dem subset
sub_tmp=/tmp/dem_subset 

# reprojection string
reproj="-t_srs EPSG:326${sub_zone}"

# resampling string (uncomment to resample to pixel size different than dem_file)
#sub_xps=30 # x-pixel size in meters
#sub_yps=30 # y-pixel size in meters
#resamp="-tr $sub_xps $sub_yps" 

# extract the subset
$gdaltranslate -of ENVI -projwin $sub_extent $dem_path $sub_tmp

# reproject (and resample if necessary)
$gdalwarp -of ENVI $reproj $resamp $sub_tmp $sub_path

# utm coords should be close to gdalinfo output
echo "UTM ul (x,y,zone):" $($ll2utm $ul_lat $ul_lon -z $sub_zone)
echo "UTM lr (x,y,zone):" $($ll2utm $lr_lat $lr_lon -z $sub_zone)
$gdalinfo $sub_path

