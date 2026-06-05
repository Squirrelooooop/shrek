import shrek_hek_cellpose_py310 as cp

# convert ome-tiff to multipage tiff (for the inverted only)
#cp.ometotiff(input_dir = "/Users/sunny/Desktop/20260427_ARPA_Demo/")

# downsample the multipage tiff
#cp.downsample(input_dir = "/Users/sunny/Desktop/20260427_ARPA_Demo/multipage_tiff", msPerFrame=60, ds_factor = 10)

# do the motion correction in sunny_caiman_motion_correction.py and remember to change the conda env to caiman

# cellpose masks
cp.tiff_to_mask(input_dir = "/Users/sunny/Desktop/20260427_ARPA_Demo/multipage_tiff/ds/motion_corrected", snap_dir = "")
