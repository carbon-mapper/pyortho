from ortho_util import *

if __name__ == '__main__':
    gen_zint_ql=True
    gen_rgb_ql=False
    gen_rdn_ql=False

    
    src_dir = '/Users/bbue/Research/data/AVIRISNG/20150420_4c/ang20150420t182808/'
    out_dir = '/Users/bbue/Research/AVIRISNG/range/watch_out/'
    
    igm_hdrf = 'ang20150420t182808_rdn_igm.hdr'
    glt_hdrf = 'ang20150420t182808_rdn_glt.hdr'

    zint_imgf = igm_hdrf.replace('igm.hdr','zint')

    rgb_hdrf = pathjoin(src_dir,'ang20150420t182808_raw_k1_rgb_img.hdr')
    rgb_ql_imgf = rgb_hdrf.replace('img.hdr','img_ort_ql')
    
    rdn_hdrf = pathjoin(src_dir,'ang20150420t182808_rdn.hdr')
    rdn_ql_imgf = rdn_hdrf.replace('rdn.hdr','rdn_ort_ql')

    if gen_zint_ql:
        zint_bands = [2]
        generate_ql(pathjoin(out_dir,igm_hdrf),pathjoin(out_dir,glt_hdrf),
                    pathjoin(out_dir,zint_imgf),bands=zint_bands)
        generate_ql(pathjoin(src_dir,igm_hdrf.replace('rdn','ort')),
                    pathjoin(src_dir,glt_hdrf.replace('rdn','ort')),
                    pathjoin(src_dir,zint_imgf.replace('rdn','ort')),
                    bands=zint_bands)

    if gen_rgb_ql:
        rgb_ql_bands = [0,1,2]
        envi2jpeg(rgb_hdrf,rgb_ql_imgf+'.jpg',rgb_ql_bands)
        #generate_ql(rgb_hdrf,pathjoin(out_dir,glt_hdrf),
        #            pathjoin(out_dir,rgb_ql_imgf),bands=rgb_ql_bands)
        #generate_ql(rgb_hdrf,pathjoin(src_dir,glt_hdrf.replace('rdn','ort')),
        #            pathjoin(src_dir,rgb_ql_imgf.replace('img_ort','ort')),
        #            bands=rgb_ql_bands)
        
    if gen_rdn_ql:
        rdn_ql_bands = [59,41,23]
        generate_ql(rdn_hdrf,pathjoin(out_dir,glt_hdrf),
                    pathjoin(out_dir,rdn_ql_imgf),bands=rdn_ql_bands)
        generate_ql(rdn_hdrf,pathjoin(src_dir,glt_hdrf.replace('rdn','ort')),
                    pathjoin(src_dir,rdn_ql_imgf.replace('rdn_ort','ort')),
                    bands=rdn_ql_bands)        
        

    generate_kml(pathjoin(out_dir,glt_hdrf),pathjoin(out_dir,zint_imgf))
    generate_kml(pathjoin(src_dir,glt_hdrf.replace('rdn','ort')),
                 pathjoin(src_dir,zint_imgf.replace('rdn','ort')))
