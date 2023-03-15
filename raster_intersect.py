from __future__ import absolute_import, division, print_function

import sys,os
import gdal
import numpy as np

from ortho_util import *

gdalwarp  = 'gdalwarp' # '/shared/anaconda2/bin/gdalwarp'
gdalwarp += ' -r {resample} -of ENVI'
gdalwarp += ' -srcnodata {nodata} -dstnodata {nodata} -wt Float32 -ot Float32'
gdalwarp += ' -te {left} {bottom} {right} {top} -tr {ps} {ps} {imagef} {outf}'
resample  = 'bilinear' #

NODATA = -9999

def filename(path):
    '''
    /path/to/file.ext -> file.ext
    '''
    return pathsplit(path)[1]

def basename(path):
    '''
    /path/to/file.ext -> file
    '''
    return splitext(filename(path))[0]

def bands2intensity(img,p=0.95):
    """
    bands2intensity(img,p=0.95)

    Summary: computes mean intensity of percentile-clipped bands

    Arguments:
    - img: n x m x b image data

    Keyword Arguments:
    - p: clip percentile (default=0.95)

    Output:
    - n x m mean intensity image with values in [0,1] range
    """
    assert(img.ndim==3)
    inten = np.zeros([img.shape[0],img.shape[1]])
    for bi in range(img.shape[2]):
        bandi = img[...,bi]
        bmin,bmax = extrema(bandi,p=p)
        inten = inten + np.clip((bandi-bmin)/(bmax-bmin),0,1)
    return inten/img.shape[2]

def findRasterIntersect(image1,image2,bands=[1],outdir='.',overwrite=False,resample=resample):
    """
    findRasterIntersect(image1,image2,band=1,outdir='.',overwrite=False,resample=resample)
    
    Summary: generates cropped + resampled images capturing the pixels where image1+image2 overlap

    based on some code via: http://sciience.tumblr.com/post/101722591382/finding-the-georeferenced-intersection-between-two
    
    Arguments:
    - image1: image1 path
    - image2: image2 path
    
    Keyword Arguments:
    - bands: bands to use in difference measurement (default=[1])
    - outdir: output directory (default='.')
    - overwrite: overwrite existing output files (default=False)
    - resample: resampling method (default='bilinear')
    
    Output:
    - sub1: overlap image from image1
    - sub2: overlap image from image2
    - mask: set membership mask (0=disjoint, {1,2}={image1,image2} only, 3=overlap)
    
    """
    print('image1:',image1)
    print('image2:',image2)
    
    flightid1 = basename(image1).split('_')[0]
    flightid2 = basename(image2).split('_')[0]
    outdir = pathjoin(outdir,'_'.join([flightid1,flightid2]))
    
    if not pathexists(outdir):
        print('created directory',outdir)
        os.makedirs(outdir)

    img1 = envi_open(image1+'.hdr')
    img2 = envi_open(image2+'.hdr')
    mapdict1 = envi_mapinfo(img1)
    mapdict2 = envi_mapinfo(img2)

    def ct(s,l,mapdict):
        ulx,uly,xps = [mapdict[k] for k in ('ulx','uly','xps')]
        rot = mapdict.get('rotation',0)
        x,y = sl2map(s,l,ulx,uly,xps)
        if rot==0:
            return x,y
        return rotxy(x,y,rot,ulx,uly)

    def bbox(p):
        minx,maxx = extrema([pi[0] for pi in p])
        miny,maxy = extrema([pi[1] for pi in p])
        return minx,maxy,maxx,miny

    ct1 = lambda p: ct(p[0],p[1],mapdict1)
    rows1,cols1 = img1.shape[0],img1.shape[1]
    r1 = bbox(map(ct1,[(0,0),(cols1,0),(cols1,rows1),(0,rows1)]))

    ct2 = lambda p: ct(p[0],p[1],mapdict2)
    rows2,cols2 = img2.shape[0],img2.shape[1]
    r2 = bbox(map(ct2,[(0,0),(cols2,0),(cols2,rows2),(0,rows2)]))

    nodata1 = int(img1.metadata.get('data ignore value',NODATA))
    nodata2 = int(img2.metadata.get('data ignore value',NODATA))
    xps1 = float(mapdict1['xps'])
    xps2 = float(mapdict2['xps'])

    print('image1 rows,cols:',rows1,cols1)
    print('image2 rows,cols:',rows2,cols2)
    print('image1 bounding box: %s' % str(['%.6f'%p for p in r1]))
    print('image2 bounding box: %s' % str(['%.6f'%p for p in r2]))

    #  TODO (BDB, 01/29/18): resample to min ps? max? mean?
    print('image1 ps: "%s"'%str((xps1)))
    print('image2 ps: "%s"'%str((xps2)))
    ps = np.min([xps1,xps2])    
    if ps != xps1 or ps != xps2:
        print('resampling to ps: "%s"'%str((ps)))
    # find left,top,right,bottom of intersection between bounding boxes
    intersection = [max(r1[0], r2[0]), min(r1[1], r2[1]),
                    min(r1[2], r2[2]), max(r1[3], r2[3])]    
    print('intersection:',['%.6f'%p for p in intersection])
    if r1 != r2:
        left,top,right,bottom = intersection

        print('** different bounding boxes **')
        # check for any overlap at all...
        if (right < left) or (top < bottom):
            intersection = None
            print('*** ERROR *** IMAGE1 AND IMAGE2 DO NOT OVERLAP ***')
            return ()
        else:            
            base1,ext1 = splitext(image1)
            out1 = pathjoin(outdir,basename(base1)+'_sub'+ext1)
            if overwrite or not pathexists(out1):
                if pathexists(out1):
                    os.unlink(out1)

                imagef,outf,nodata = image1,out1,nodata1
                stdout1,stderr1,ret1 = runcmd(gdalwarp.format(**locals()),verbose=1)
                if ret1 != 0:
                    print(stderr1)
                    return ()
            print('loading',out1)                
            raster1 = gdal.Open(out1)
            col1 = raster1.RasterXSize
            row1 = raster1.RasterYSize
                        
            base2,ext2 = splitext(image2)
            out2 = pathjoin(outdir,basename(base2)+'_sub'+ext2)
            if overwrite or not pathexists(out2):
                if pathexists(out2):
                    os.unlink(out2)             

                imagef,outf,nodata = image2,out2,nodata2
                stdout2,stderr2,ret2 = runcmd(gdalwarp.format(**locals()),verbose=1)
                if ret2 != 0:
                    print(stderr2)
                    return ()
            print('loading',out2)
            raster2 = gdal.Open(out2)
            col2 = raster2.RasterXSize
            row2 = raster2.RasterYSize
            
            print('sub1 rows,cols:',row1,col1)
            print('sub2 rows,cols:',row2,col2)
            if col1 != col2 or row1 != row2:
                print("*** ERROR *** INTERSECTION COLS and ROWS DO NOT MATCH ***")
                return ()
            
            # these arrays should now have the same spatial geometry though NaNs may differ

    else: # same dimensions from the get go
        raster1 = gdal.Open(image1)
        raster2 = gdal.Open(image2)
        col1 = raster1.RasterXSize # = col2
        row1 = raster1.RasterYSize # = row2

    bands1 = [raster1.GetRasterBand(band+1) for band in bands]
    bands2 = [raster2.GetRasterBand(band+1) for band in bands]
        
    # only use first band in masking operation
    array1 = bands1[0].ReadAsArray()
    array2 = bands2[0].ReadAsArray()

    # create overlap mask with nodata region
    nodata1 = array1 == nodata1
    nodata2 = array2 == nodata2
    mask_array =  np.int16(~nodata1) - np.int16(~nodata2)

    # 1,2 = image1,image2 only
    mask_array[mask_array==-1] = 2

    # 3 = overlap 
    mask_array[(~nodata1) & (~nodata2)] = 3 

    # 0 = no overlap
    mask_array[nodata1 & nodata2] = 0

    mask_out = pathjoin(outdir,'mask_array')
    if overwrite and pathexists(mask_out):
        os.unlink(mask_out)
    envi_drv = gdal.GetDriverByName('ENVI')
    mask_ds = envi_drv.Create(mask_out, col1, row1, 1, gdal.GDT_Byte)

    # inherit the transform + projection
    mask_ds.SetGeoTransform(raster1.GetGeoTransform())
    mask_ds.SetProjection(raster1.GetProjection()) 
    mask_band = mask_ds.GetRasterBand(1)
    mask_band.WriteArray(mask_array)

    if len(bands)>1:
        data1 = [array1]+[b.ReadAsArray() for b in bands1[1:]]
        data2 = [array2]+[b.ReadAsArray() for b in bands2[1:]]

        extrema1 = [extrema(b[~nodata1]) for b in data1]
        extrema2 = [extrema(b[~nodata2]) for b in data2]
    
        array1 = np.dstack(data1)
        array2 = np.dstack(data2)
    elif len(bands)==1:
        extrema1 = [extrema(array1[~nodata1])]
        extrema2 = [extrema(array2[~nodata2])]

        array1 = array1[...,np.newaxis]
        array2 = array2[...,np.newaxis]

    print('image1 extrema: "%s"'%str(extrema1))
    print('image2 extrema: "%s"'%str(extrema2))

    print('array1.shape: "%s"'%str((array1.shape)))
    print('array2.shape: "%s"'%str((array2.shape)))
    
    return (array1, array2, mask_array, intersection, ps)

def evalIntersection(image1,image2,outdir):    
    nbands = envi_open(image1+'.hdr').nbands

    assert(nbands==envi_open(image2+'.hdr').nbands)
    
    bands = [int(b) for b in range(25,nbands,25)]
    print('bands (nbands=%d):'%nbands,bands)
    
    isect = findRasterIntersect(image1,image2,bands=bands,outdir=outdir,
                                overwrite=overwrite)
    if len(isect)==0:
        print('an error occurred intersecting %s and %s'%(basename(image1),
                                                          basename(image2)))
        sys.exit(1)
        
    isect_array1, isect_array2, isect_mask, isect_bb, isect_ps = isect
    
    isect_mask   = np.float32(isect_mask)
    isect_array1 = np.float32(isect_array1)
    isect_array2 = np.float32(isect_array2)

    isect_inten1 = bands2intensity(isect_array1)
    isect_inten2 = bands2intensity(isect_array2)

    # isect_mask == 0 = no overlap between image1 & image2
    isect_disjoint = (isect_mask == 0)

    # isect_mask == {1,2} = {image1,image2} only
    image1_only    = (isect_mask == 1)
    image2_only    = (isect_mask == 2)

    # isect_mask == 3 = image1 & image2 overlap
    isect_overlap  = (isect_mask == 3)

    # isect_diff = sum of intensity images
    isect_diff = isect_inten1-isect_inten2
    isect_dmax = np.abs(isect_diff[isect_overlap]).max()
    isect_diff[~isect_overlap] = np.nan

    # mask the empty regions in each image
    isect_inten1[~(image1_only | isect_overlap)] = np.nan
    isect_inten2[~(image2_only | isect_overlap)] = np.nan
    isect_mask[isect_disjoint] = np.nan

    # get rid of empty rows,cols for visualization
    dropcols = isect_disjoint.all(axis=0)
    droprows = isect_disjoint.all(axis=1)

    isect_inten1 = isect_inten1[~droprows][:,~dropcols]
    isect_inten2 = isect_inten2[~droprows][:,~dropcols]
    isect_mask = isect_mask[~droprows][:,~dropcols]
    isect_diff = isect_diff[~droprows][:,~dropcols]
    isect_overlap = isect_overlap[~droprows][:,~dropcols]
    vmin,vmax = extrema(np.r_[isect_inten1[isect_overlap].ravel(),
                              isect_inten2[isect_overlap].ravel()],p=0.95)
    
    fig,ax = pl.subplots(1,4,sharex=True,sharey=True,figsize=(12,4))
    ax[0].imshow(isect_inten1,vmin=vmin,vmax=vmax); ax[0].set_title('isect_inten1')
    ax[1].imshow(isect_inten2,vmin=vmin,vmax=vmax); ax[1].set_title('isect_inten2')
    ax[2].imshow(isect_mask,vmin= 0,vmax=3); ax[2].set_title('isect_mask')
    ax[3].imshow(isect_diff,vmin=-isect_dmax,vmax=isect_dmax); ax[3].set_title('isect_diff')

    for i in range(4):
        ax[i].set_xticks([])
    ax[0].set_yticks([])
    
    pl.tight_layout()
    pl.ioff()
    pl.show()

if __name__ == '__main__':
    import pylab as pl
    import sys,argparse,socket
    testcase = 'bakersfield'
    
    parser = argparse.ArgumentParser(description="Intersect two rasters")
    parser.add_argument("--overwrite", action='store_true', help="overwrite existing subimages")
    parser.add_argument("testcase", type=str,  help="testcase (bakersfield, newport, or palmdale)", default=testcase)

    args = parser.parse_args(sys.argv[1:])

    testcase = args.testcase
    overwrite = args.overwrite

    indir = '/lustre/bbue/prism_ortho/multiview'
    if testcase=='palmdale':    
        image1 = 'prm20170524t180401_rdn_v1t1_img' # good ortho
        image2 = 'prm20170524t175501_rdn_v1t1_img' # bad (but non-null) ortho
        #image3 = 'prm20170524t182017_rdn_v1t1_img' # bad (null) ortho
        #image4 = 'prm20170524t174419_rdn_v1t1_img' # bad (null) ortho
    elif testcase=='bakersfield':    
        image1 = 'prm20170523t182254_rdn_v1t1_img' # good ortho
        image2 = 'prm20170523t183245_rdn_v1t1_img' # good ortho
    elif testcase=='newport':
        image1 = 'prm20170529t160259_rdn_v1t1_img' # good ortho
        image2 = 'prm20170529t161011_rdn_v1t1_img' # good ortho
        #image3 = 'prm20170529t161447_rdn_v1t1_img' # bad (null) ortho        
    else:
        print('unknown testcase "%s"'%testcase)
        sys.exit(1)

    
    indir = pathjoin(indir,testcase)
    outdir = indir

    images = [image1,image2]    
    images = [pathjoin(indir,imagef) for imagef in images]
    evalIntersection(images[0],images[1],outdir)
