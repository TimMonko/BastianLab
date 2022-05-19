# -*- coding: utf-8 -*-
"""
Batch Macro
Created on Thu May 19 15:32:14 2022

@author: Tim M
"""

#%% Get Image Paths

import os, fnmatch
import time

start = time.time()

# See Current Directory structure
for root, subdirs, files in os.walk("."):
    path = root.split(os.sep) # split at separator (/ or \)
    print((len(path) -1) * '---', os.path.basename(root))
    for file in files:
        print(len(path) * '---', file)   

# Use a path for the files of interest, can be a parent folder or a specific subdirectory
path = "20220517-PSD95-PunctaProjectAnalysis/"
search_str = "*.czi*"
filepath_list = []

for path, subdirs, files in os.walk(path):
        for file in fnmatch.filter(files, search_str):
            filepath = os.path.join(path, file)
            print(filepath)
            filepath_list.append(filepath)

print("Directory Access: ", time.time() - start)

#%%
import napari
from aicsimageio import AICSImage 
import pyclesperanto_prototype as cle
import napari_simpleitk_image_processing as nsitk

if 'viewer' not in globals():
    viewer = napari.Viewer()

blob_number = []

for images in filepath_list:
    start = time.time()
    img = AICSImage(images)
    PSD95 = img.get_image_dask_data("YX", channel_names = 'EGFP')
    
    PSD95_ms = cle.median_sphere(PSD95, None, 1.0, 1.0, 0.0) # good, 2 maybe ok
    PSD95_ths = cle.top_hat_sphere(PSD95_ms, None, 5.0, 5.0, 0.0) # good

    PSD95_logf = nsitk.laplacian_of_gaussian_filter(PSD95_ths, 1.5)
    PSD95_logf_in = nsitk.invert_intensity(PSD95_logf)
    PSD95_hmax = nsitk.h_maxima(PSD95_logf_in, 300.0)
    PSD95_blobs = cle.voronoi_otsu_labeling(PSD95_hmax, None, 1.0, 1.0)
    PSD95_total = cle.maximum_of_all_pixels(PSD95_blobs)
    
    blob_number.append(PSD95_total)

    # viewer.add_image(PSD95_hmax, name='LoG-H-Max')
    # viewer.add_labels(PSD95_blobs)
    
    print(images + " Blob Filtering: ",time.time() - start)
    
    
    
    
    