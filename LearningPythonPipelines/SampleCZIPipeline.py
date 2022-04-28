"""
Sample Pipeline for Bastian Lab
Created on Mon Mar  7 15:16:53 2022
Began after Tutorial 30 of APEER Micro Python tutorials
@author: Tim M
"""
#%% Get Image Paths

import os, fnmatch

# See Current Directory structure
for root, subdirs, files in os.walk("."):
    path = root.split(os.sep) # split at separator (/ or \)
    print((len(path) -1) * '---', os.path.basename(root))
    for file in files:
        print(len(path) * '---', file)   

# Use a path for the files of interest, can be a parent folder or a specific subdirectory
path = "PSD95-Images/"
search_str = "*.ti*"
filepath_list = []

for path, subdirs, files in os.walk(path):
        for file in fnmatch.filter(files, search_str):
            filepath = os.path.join(path, file)
            print(filepath)
            filepath_list.append(filepath)

#%% Import Image Single Image to Napari
# pip install napari[all]
# pip install pyqt5==5.12.3        
# pip install pyqtwebengine==5.12.1    
import napari
from aicsimageio import AICSImage
img = AICSImage(filepath_list[2])
print(img.dims)

#https://allencellmodeling.github.io/aicsimageio/_static/v3/
#if 'viewer' not in globals():
viewer = napari.Viewer()
    
#viewer.add_image(img.data, name = 'raw') # or raw.dask_data. Remember that the attribute is needed because the AICSimage class incorporates many things in the meta data 
# napari.view_image(raw.dask_data)
# viewer.add_image(img.data, channel_axis = 1, name = img.channel_names, gamma = 0.45)

#%% Blob Filtering
import pyclesperanto_prototype as cle
C0 = img.get_image_data("YX",C=0)
viewer.add_image(C0)

# median sphere
C0_ms = cle.median_sphere(C0, None, 1.0, 1.0, 0.0)
#viewer.add_image(C0_ms, name='Result of median_sphere (clesperanto)')

# top hat sphere
C0_ths = cle.top_hat_sphere(C0_ms, None, 5.0, 5.0, 0.0)
#viewer.add_image(C0_ths, name='Result of top_hat_sphere (clesperanto)')

# gaussian blur

C0_gb = cle.gaussian_blur(C0_ths, None, 1.0, 1.0, 0.0)
#viewer.add_image(C0_gb, name='Result of gaussian_blur (clesperanto)')

# laplace box
C0_lb = cle.laplace_box(C0_gb)
viewer.add_image(C0_lb, name='Result of laplace_box (clesperanto) [1]')

#%% Blob Detection
#viewer.add_image(filters.difference_of_gaussians(C0_lb,low_sigma = 1, high_sigma = 2))
#DoG = feature.blob_dog(C0_lb, max_sigma=30, threshold=.1)
#above uses peak_local_max

C0_max = cle.detect_maxima_box(C0_lb, radius_x = 2, radius_y = 2, radius_z = 0)
#viewer.add_labels(C0_max)
C0_otsu = cle.threshold_otsu(C0_lb)
#viewer.add_labels(C0_otsu)

C0_spots = cle.binary_and(C0_max, C0_otsu)
#viewer.add_labels(C0_spots)

C0_voronoi = cle.masked_voronoi_labeling(C0_spots, C0_otsu)
viewer.add_labels(C0_voronoi)

#%% Ridge Detection

from skimage.filters import meijering #, sato, frangi, hessian

# Gaussian Blur 
C0s_gb = cle.gaussian_blur(C0, None, 1.0, 1.0, 0.0)
#viewer.add_image(C0s_gb)
# Gaussian Background Subtraction
C0s_sgb = cle.subtract_gaussian_background(C0s_gb, None, 10.0, 10.0, 0.0)
#viewer.add_image(C0s_sgb)

# Meijering Ridge Filter
C0s_mj = meijering(C0s_sgb, sigmas = range(6,12,2), black_ridges = False)
viewer.add_image(C0s_mj) #does not work with pycl.OCLAArray

# Otsu Threshold
C0s_mj_to = cle.threshold_otsu(C0s_mj)
#viewer.add_labels(C0s_mj_to, name='Result of threshold_otsu (clesperanto) [1]')

# Closing Labels
C0s_mj_cl = cle.closing_labels(C0s_mj_to, None, 3.0)
#viewer.add_labels(C0s_mj_cl)

# Connected Components Labeling

C0s_neurites = cle.connected_components_labeling_box(C0s_mj_cl)
viewer.add_labels(C0s_neurites)

#%% Only Blobs on Ridges
