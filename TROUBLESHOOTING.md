## Troubleshooting Orthorectfication Issues

### General Troubleshooting

* Use the pyortho/ortho_nav.py diagnostic functions to view the contents of the PPS/GPS tables and the frame headers for a given raw image. 
* If there are no obvious issues in the ortho_nav.py output, visually inspect the DEM to see if there are NODATA values that aren't specified in the .hdr
* Make sure the DEM is large enough to cover the raw data you're processing


### Common Issues

#### NODATA values in DEM

NODATA values are typically assigned very small negative values (e.g., -9999, -32768). When present, the 'data ignore value' field must be defined in the .hdr file for the DEM. 

A missing or incorrect 'data ignore value' in a DEM will result in (potentially significant) geolocalization errors, as the ortho software will treat the NODATA values as valid surface elevations. 

<b>Resolution:</b>

* Add the 'data ignore value' to the .hdr for the DEM.
* Make sure the DEM contains values matching the 'data ignore value' present in the .hdr. 
* If no values matching 'data ignore value' are present in the DEM, make sure the provided 'data ignore value' entry is correct via visual inspection or by computing image statistics for the DEM.  

#### Raw file begins with science frames instead of dark frames

A typical AVIRIS-NG / PRISM raw image should contain (in order):

1) a fixed length sequence of dark calibration frames 
2) an arbitrary length sequence of science frames
3) another set of dark calibration frames
4) additional non-science frames (not used in orthoprocessing)

However, sometimes the raw images will contain a handful of extra science frames at the beginning of the file before entering the dark calibration region. Why exactly this occurs is currently under investigation, but it appears to be a result of the framebuffer dumping frames from the most recent previous acquisition into the current raw. 

<b>Resolution:</b>

* To work around this issue, we need to skip those initial science so we start processing within the first dark calibration region. 
* One way to do this is to change the value of the IMG_SL parameter to a value that starts the orthoprocessing at any frame in the first dark calibration region. 
* Another is to use the '-s' parameter of orthorectify.py to manually set the start line from the command line.

#### Bad surface coords from corrupt clock timings

The clock entries in the GPS table, PPS table, or the raw frames headers may occasionally contain corrupt clock timing values. When these bad entries are present, the surface coordinates for pixels estimated from those clock values will be inaccurate, and may rapidly diverge from the coordinates of their neighboring pixels. 

If the errors are significant, the following error may occur: 

"Error computing GLT dimensions: total GLT pixels (...) larger than max(IGM.shape)**2 (...)"

The above error occurs if any error in computing surface coordinates produces outliers that are spatially distant from their true coordinates. If the bounding box of the surface (x,y,z) values produces a GLT larger than twice the number of IGM pixels (a solution that is not physically realizable), the GLT generation will bail out with the error above.

<b>Resolution:</b>

* These errors are quite rare, and are currently handled on a case-by-case basis since diagnosing the source of the corruption is essential to make sure the hardware / software interface is functional. 
* File an issue on the pyortho github site and/or contact bbue@jpl.nasa.gov with info. on which file caused the problem. 



#### Indexing outside DEM extent

If the raw image contains coordinates that image outside of the spatial extent of the DEM subset, we cannot determine the surface coordinates for that location. This may also crash the orthorectification process if we index out of the memory allocated for the DEM.

<b>Resolution:</b>
 
* Inspect the ortho\_nav.py output to make sure the lat/lon values are within the bounds of the DEM.
* If the lat,lon appear to fit into the DEM, increase subset\_width to extract a larger region of the DEM during ortho processing.

#### DEM subsetting errors 

The following errors might occur in the initial orthoprocessing stage when attempting to subset the DEM:

* "Attempt to create dataset is illegal"
* "ERROR 2: memdataset.cpp, ...: cannot allocate ... bytes"

These will typically occur because:

* the DEM contains NODATA values that are not flagged with a 'data ignore value' field in its corresponding .hdr
* the spatial extent of the flightline is not contained in the extent of the DEM,
* the GPS/PPS tables contain corrupt values, or
* a subset of the first few science frames in the raw image contain corrupt values

<b>Resolution:</b>

* Check the DEM extent + nodata values first
* Next, check the frame coordinates with ortho_nav.py to make sure no outliers are present
* If coordinate outliers are present, the error is probably a result of bad clock timings or some other form of data corruption







