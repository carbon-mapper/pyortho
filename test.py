from find_pixel_trace_cython import find_pixel_trace_cython
'''
ENVI output (incorrect...)
ENVI> ang_aig_find_pixel_trace,3,3,1,1,xpix,ypix,s    
ENVI> print,xpix,ypix,s
           3           2           2           1           1
           3           3           2           2           1
      0.00000      0.00000     0.707107      1.41421      2.82843

ENVI> ang_aig_find_pixel_trace,1,1,3,3,xpix,ypix,s
ENVI> print,xpix,ypix,s                           
           1           2           2           3           3
           1           1           2           2           3
      0.00000      1.41421      2.12132      2.82843      2.82843

ENVI> ang_aig_find_pixel_trace,3,3,3,1,xpix,ypix,s
ENVI> print,xpix,ypix,s                           
           3           3           3
           3           2           1
      0.00000     0.500000      2.00000

ENVI> ang_aig_find_pixel_trace,3,3,1,3,xpix,ypix,s 
ENVI> print,xpix,ypix,s                           
           3           2           1
           3           3           3
      0.00000     0.500000      2.00000

ENVI> ang_aig_find_pixel_trace,5,3,1,1,xpix,ypix,s
ENVI> print,xpix,ypix,s                           
           5           4           3           2           1           1           1
           3           3           3           3           3           2           1
      0.00000      2.11803      2.53225      3.21699          Inf          Inf      4.47214      

ENVI> ang_aig_find_pixel_trace,1,1,3,5,xpix,ypix,s
ENVI> print,xpix,ypix,s                           
           1           1           2           2           2           3           3
           1           2           2           3           4           4           5
      0.00000      1.67705      2.23607      2.79508      3.91312      4.47214      4.47214
'''      
if __name__ == '__main__':
    print 'diag-'
    print find_pixel_trace_cython(3,3,1,1)    
    print 'diag+'
    print find_pixel_trace_cython(1,1,3,3) 
    print 'dx==0'
    print find_pixel_trace_cython(3,3,3,1)
    print 'dy==0'
    print find_pixel_trace_cython(3,3,1,3)
    print 'rect-'
    print find_pixel_trace_cython(5,3,1,1)
    print 'rect+'
    print find_pixel_trace_cython(1,1,3,5)
