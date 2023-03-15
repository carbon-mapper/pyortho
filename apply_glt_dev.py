#!/usr/bin/env python
# David R Thompson
# Clip the dark current segments at the start and end of an image

import os
import sys
import spectral
import argparse
from scipy import *
from scipy import linalg


chunksize = 500  # process this many lines at once

# Return the header associated with an image file
def find_header(imgfile):
  if os.path.exists(imgfile+'.hdr'):
    return imgfile+'.hdr'
  ind = imgfile.rfind('.raw')
  if ind >= 0:
    return imgfile[0:ind]+'.hdr'
  ind = imgfile.rfind('.img')
  if ind >= 0:
    return imgfile[0:ind]+'.hdr'
  raise IOError('No header found for file {0}'.format(imgfile));

def apply_glt(in_file,glt_file,out_file,bands=[],verbose=True):
  in_hdr = find_header(in_file)
  if not os.path.exists(in_hdr):
    raise IOError('cannot find a header');
  img = spectral.io.envi.open(in_hdr, in_file)
  
  glt_hdr = find_header(glt_file)
  if not os.path.exists(glt_hdr):
    raise IOError('cannot find a header');
  glt = spectral.io.envi.open(glt_hdr, glt_file)

  out_hdr = out_file+'.hdr'

  # Get metadata
  metadata = img.metadata.copy()
  interleave = img.metadata['interleave']
  if interleave not in ('bip','bil'):
    raise ValueError('cannot use %s interleave'%interleave)
  
  # Define band subset if necessary
  nbands = len(bands)
  if nbands == 0: # apply glt to all bands
    nbands = int(metadata['bands'])
    bands = arange(nbands)

  bands = array(bands,dtype=int)
  band_names = metadata.get('band names',map(str,bands))
  band_names = [n for i,n in enumerate(band_names) if i in bands]
   
  for field in ['lines','samples','map info']:
    metadata[field] = glt.metadata[field]

  metadata['bands'] = nbands  
  metadata['band names'] = band_names
  metadata['data ignore value'] = -9999
  metadata['map info'] = glt.metadata['map info']
  metadata['data type'] = 4 # float32
  
  out = spectral.io.envi.create_image(out_hdr, metadata, ext='', force=True)

  imgmm = img.open_memmap(interleave='source', writable=False)
  outmm = out.open_memmap(interleave='source', writable=True)

  nl = glt.shape[0]

  bad = glt.read_band(0)==0
  cols = abs(glt.read_band(0))-1
  rows = abs(glt.read_band(1))-1
  
  nstep = 10
  step = int(nl/nstep)
  for i in range(nl):
    if verbose and (i % step == 0 or i==nl):
      iper = 0 if i==0 else iper+1
      print('Processing line %6d of %6d   \t%3d%%'%(i,nl,100*float(iper)/nstep))
      
    outmm[i,:,:] = 0

    coli,rowi,badi = cols[i,:],rows[i,:],bad[i,:]
    # spatial subset
    if interleave == 'bil':
      for j,(r,c,b) in enumerate(zip(rowi, coli, badi)):
        if not b:
          outmm[i,:,j] = array(imgmm[r,bands,c],dtype=float32)
        else:
          outmm[i,:,j] = -9999

    elif interleave == 'bip': 
      for j,(r,c,b) in enumerate(zip(rowi, coli, badi)):
        if not b:
          outmm[i,j,:] = array(imgmm[r,c,bands],dtype=float32)
        else:
          outmm[i,j,:] = -9999    
  
def main():
  # parse the command line (perform the correction on all command line arguments)
  parser = argparse.ArgumentParser(description="Clip dark lines from file")
  parser.add_argument("in_file", metavar = "IMAGE", type=str, 
      help="Radiance image")
  parser.add_argument("glt_file", metavar = "GLT", type=str, 
      help="GLT image")
  parser.add_argument("out_file", metavar = 'OUT', type=str, 
      help="output")
  parser.add_argument('bands', type=int, nargs='*',default=[], 
                      help='Indices of bands to correct (default=all)',
                      metavar='BAND_INDEX')  
 
  args = parser.parse_args(sys.argv[1:])

  in_file = args.in_file
  glt_file = args.glt_file
  out_file = args.out_file
  bands = args.bands

  apply_glt(in_file,glt_file,out_file,bands=bands)

if __name__ == "__main__":
  main()

