## pyortho: Python (Real Time) Orthorectification Toolbox

### Required libraries

* numpy/scipy: http://numpy.org / http://scipy.org
* scikit-image: http://scikit-image.org
* spectral python: http://spectralpython.sourceforge.net
* numba: http://numba.pydata.org

### Optional libraries

* gdal: required to subset map projected DEMs (http://gdal.org/python/)
* matplotlib: required to display/export orthocorrected quicklook images (http://matplotlib.sourceforge.net)

### Intro

The pyortho package generates the Input Geometry (IGM) and Geometry Lookup Table (GLT) images to orthorectify raw AVIRIS-NG or PRISM images, provided tables of associated frame timings (PPS) and associated aircraft location/orientation measurements per frame (GPS).

The primary components of the package are:

* orthorectify.py: command line script which parses raw image data, user parameters, platform-specific constants and navigational measurements and generates orthoproessed products, particularly IGM and GLT images, used to orthorectify the input image and measure geospatial observations associated with the image.
* ortho\_util.py - contains the geolocate function called by orthorectify.py that raytraces pixels from the aircraft to the surface, and other helper functions
* ortho\_platform.py - parses/interprets platform-specific parameters provided in pyortho/platform/{AVIRIS-NG,PRISM}.json, stores camera model parameters 
* ortho\_nav.py - functions to compute aircraft position using C-MIGITS III PPS/GPS tables 
* ortho\_config.py - parser to read configuration parameters provided in pyortho/config, generally used to switch between offline and realtime processing modes
* ortho\_dem.py - stores Digital Elevation Model data and parameters

### Command Line Usage
```
usage: orthorectify.py [-h] [-f CONFIG_FILE] [-p PLATFORM] [-c CAMERA_FILE]
                       [-g GEOID_FILE] [-d DEM_PREFIX] [-w SUBSET_WIDTH]
                       [-o OUTPUT_PATH] [-s SKIPLINES] [-n NUMLINES]
                       [--offset_clock OFFSET_CLOCK]
                       raw

positional arguments:
  raw                   Path to raw image

optional arguments:
  -h, --help            show this help message and exit
  -f CONFIG_FILE, --config_file CONFIG_FILE
                        Path to configuration file
                        (default=config/pyorthorc.offline)
  -p PLATFORM, --platform PLATFORM
                        Platform (AVIRIS-NG or PRISM)
  -c CAMERA_FILE, --camera_file CAMERA_FILE
                        Path to camera file for specified platform
  -g GEOID_FILE, --geoid_file GEOID_FILE
                        Path to geoid file (default=world_model/egm96/egm96)
  -d DEM_PREFIX, --dem_prefix DEM_PREFIX
                        DEM path + filename prefix
  -w SUBSET_WIDTH, --subset_width SUBSET_WIDTH
                        DEM subset width in degrees lat/lon
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path to store output products
                        (default=pyortho_output)
  -s SKIPLINES, --skiplines SKIPLINES
                        Initial lines (raw frames) to skip (default=0)
  -n NUMLINES, --numlines NUMLINES
                        Maximum number of lines (raw frames) to process
                        (default=all frames)
  --offset_clock OFFSET_CLOCK
                        Offset frame clock ticks by this value in seconds
                        (default=0)
```                  

### Example 

The following command...

```
$ python orthorectify.py -o ./pyortho_output/ -d ./dem/conus_ned_1arcsec ./ang20140612t204858_raw
```
...will orthorectify the AVIRIS-NG ang20140612t204858\_raw image (captured over UC-Riverside in '14) with corresponding PPS/GPS tables ang20140612t204858\_{gps,pps} using the (SRTM) Continental US DEM conus\_ned\_1arcsec. The output products are saved in /lustre/bbue/pyortho\_output. The corresponding GPS/PPS tables must be in the same directory as the raw image. 

Orthorectifying a PRISM image uses the same syntax as AVIRIS-NG.

```
$ python orthorectify.py -o ./pyortho_output/ -d ./dem/HI_DEM_ASTER ./prm20160622t014223_raw
```

### Output Files

Running the orthorectify.py script on the image "flightline_raw" will generate the following output files:

* flightline\_rdn\_igm: contains downtrack-binned surface coordinates (UTM\_x, UTM\_y, elevation) for each science frame in the \_raw, dims [num\_science\_frames/bin\_factor x num\_detector\_elts x 3], where bin\_factor is the number of raw frames averaged to match the cross track pixel size.
* flightline\_rdn\_glt: orthocorrected image file that contains the geometric lookup table entries IGM\_row & IGM\_col, dims determined by the pixel size estimated for the flightline. Negative values indicate locations populated via nearest-neighbor interpolation. The corresponding glt.hdr contains the "map info" string that ENVI/GDAL/(insert GIS software du jour here) parses to compute surface locations for each pixel in the flightline.
* flightline\_rdn\_loc: contains {lat, lon, elevation} values for each pixel, same dims as corresponding igm.
* flightline\_rdn\_obs: same dims as corresponding igm, contains 11 bands of observational data 
	* Path length (m) 
	* To-sensor azimuth (0 to 360 degrees cw from N) 
	* To-sensor zenith (0 to 90 degrees from zenith) 
	* To-sun azimuth (0 to 360 degrees cw from N)
	* To-sun zenith (0 to 90 degrees from zenith) 
	* Solar phase
	* Slope
	* Aspect
	* Cosine(i)
	* UTC Time
	* Earth-sun distance (AU)
* flightline\_rdn\_obs\_ort: orthorectified version of obs file, same dims as the corresponding glt.

### Processing Steps

For a user-specified \_raw image and an associated DEM, orthorectify.py performs the following steps: 

1. Parse user-specified command line parameters
2. Parse system constants in config file
3. Identify platform from raw image filename, parse platform-specific constants and camera model
4. Geolocate first sample in first science frame in raw image
5. Subset DEM based on geolocated coordinate
6. Geolocate 3 center pixels in all available science frames
7. Estimate along track (at) and down track (dt) pixel size based on surface coordinates from central pixels
8. Compute downtrack binning factor based on pixel size
9. Repeat until the last science frame is found:

	* Select a chunk of raw science frames
	* Average sequential frame ticks according to bin factor
	* Geolocate all samples in each downtrack-binned frame

13. Save IGM and GLT products to disk
	
	


### Copyright statement
License: Apache 2.0 (http://www.apache.org/licenses/)
