#!/home/dthompson/src/anaconda/bin/python
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



# parse the command line (perform the correction on all command line arguments)
def main():

  parser = argparse.ArgumentParser(description="Clip dark lines from file")
  parser.add_argument("img_files", metavar = "IMAGE", type=str, 
      nargs='+',help="Radiance images")
  parser.add_argument("--start_sample", type=int, 
      help="One-indexed starting sample number")
  parser.add_argument("--start_line", type=int, 
      help="One-indexed starting line number")
  parser.add_argument("--num_samples", type=int, 
      help="Total number of columns in cropped image")
  parser.add_argument("--out_file", type=str, default="", 
      help="Override output filepath")
  parser.add_argument("--num_lines", type=int, 
      help="Radiance images")
  parser.add_argument("--start_band", type=int, 
      help="One-indexed starting band number")
  parser.add_argument("--num_bands", type=int, 
      help="Total number of bands")
  parser.add_argument("--binning_factor", type=int, 
      help="Bin factor")
  parser.add_argument("--smoothing", type=str, 
      help="Load smoothing factors from a file")
  parser.add_argument("--ort_hdr_file", type=str, 
      help="Get params from GLT or Boardman 3 channel ortho fastlook image")
  args = parser.parse_args(sys.argv[1:])


  # optionally, read parameters from GLT header (or RB rgb hdr)
  if args.ort_hdr_file is not None:
    hdrfile = find_header(args.ort_hdr_file)
    with open(hdrfile,'r') as f:
      for line in f:
        toks = line.split('=')
        if len(toks) > 1:
          if 'raw starting sample' in toks[0]: 
            args.start_sample = int(toks[1].strip())
            print('starting sample: %i'%args.start_sample)
          if 'raw starting line' in toks[0]: 
            args.start_line = int(toks[1].strip())
            print('starting line: %i'%args.start_line)
          if 'line averaging' in toks[0]: 
            args.binning_factor = int(toks[1].strip())
            print('line averaging: %i'%args.binning_factor)
          if 'lines' in toks[0]: 
            args.num_lines = int(toks[1].strip())
            print('number of lines: %i'%args.num_lines)
          if 'samples' in toks[0]: 
            args.num_samples = int(toks[1].strip())
            print('number of samples: %i'%args.num_samples)

  if args.start_sample is None:
    raise ValueError('I need a start sample')
  if args.num_samples is None:
    raise ValueError('I need the number of samples')
  if args.start_line is None:
    raise ValueError('I need a start line')
  if args.num_lines is None:
    raise ValueError('I need an ending line')
  for in_file in args.img_files:
    in_hdr = find_header(in_file)
    if not os.path.exists(in_hdr):
      raise IOError('cannot find a header');
    
    # assume image is bil, get ns from metadata
    img = spectral.io.envi.open(in_hdr, in_file)
    if img.metadata['interleave'] != 'bil':
      raise ValueError('cannot use %s interleave'%img.metadata['interleave'])

    # translate to zero-based indexing
    sl = args.start_line-1
    nl = args.num_lines
    ss = args.start_sample-1
    ns = args.num_samples
    if args.start_band is None and args.num_bands is None:
        sb = 0
        nb = img.nbands
    else:
        sb = args.start_band-1
        nb = args.num_bands
    bin = args.binning_factor

    if args.smoothing is None:
       smooth = ones((int(img.metadata['bands']),))
    else:
       smooth = loadtxt(args.smoothing,comments = '#')

    print('zero-indexed starting line: '+str(sl))
    print('zero-indexed starting sample: '+str(ss))
    print('zero-indexed starting band: '+str(sb))

    out_file = in_file+'_clip'
    out_hdr = in_file+'_clip.hdr'
    if len(args.out_file)>1:
      out_file = args.out_file
      out_hdr = args.out_file + '.hdr'
    #tmp_out_file = in_file.split("/")[-1]
    #print "tmp_out_file {}".format(tmp_out_file)
    #tmp_out_file = "/lustre/iwashita/pyortho_output/" + tmp_out_file
    #out_file = tmp_out_file+'_clip'
    #out_hdr = tmp_out_file+'_clip.hdr'

    # process the image in chunks 
    metadata = img.metadata.copy()
    metadata['lines'] = nl
    metadata['samples'] = ns
    metadata['bands'] = nb
    metadata['smoothing factors'] = '{'+','.join([str(s) for s in smooth[sb:(sb+nb)]])+'}'
    metadata['wavelength'] = '{'+','.join(metadata['wavelength'][sb:(sb+nb)])+'}'
    metadata['fwhm'] = '{'+','.join(metadata['fwhm'][sb:(sb+nb)])+'}'
    print("metadata {}".format(metadata))
    out = spectral.io.envi.create_image(out_hdr, metadata, ext='', 
                force=True)
    intervals = [(i, min(i+chunksize, img.nrows)) \
                  for i in range(0, out.nrows, chunksize)]
    
    for i in range(nl):
      if i==0 or i%100==0:
        del img
        del out
        img = spectral.io.envi.open(in_hdr, in_file)
        inmm = img.open_memmap(interleave='source', writable=False)
        out = spectral.io.envi.open(out_hdr, out_file)
        outmm = out.open_memmap(interleave='source', writable=True)

      #print ("1 inmm.shape {}".format(inmm.shape))
      #print ("1 outmm.shape {}".format(outmm.shape))
      # spatial subset
      chunk = array(inmm[(sl+(i*bin)):(sl+(i+1)*bin), :, ss:(ss+ns)])
      #print ("1 chunk.shape {}".format(chunk.shape))

      # apply smoothing
      bip = chunk.transpose((0,2,1))
      bip = bip * smooth
      bil = bip.transpose((0,2,1))

      # spectral subset
      chunk = bil[:, sb:(sb+nb), :]

      # average and write to file
      outmm[i,:,:] = mean(chunk,axis=0)

    # write smoothing factors to header
    del out 
    out_hdr

if __name__ == "__main__":
  main()

