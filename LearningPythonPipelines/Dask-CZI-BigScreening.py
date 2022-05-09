"""
Sample Pipeline for Bastian Lab
Created on Mon Mar  7 15:16:53 2022
Began after Tutorial 30 of APEER Micro Python tutorials
@author: Tim M
"""


#https://github.com/haesleinhuepf/BioImageAnalysisNotebooks/blob/main/docs/32_tiled_image_processing/tiled_nuclei_counting.ipynb 
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
path = "BigDAPIMap2Images/"
search_str = "*.czi*"
filepath_list = []

for path, subdirs, files in os.walk(path):
        for file in fnmatch.filter(files, search_str):
            filepath = os.path.join(path, file)
            print(filepath)
            filepath_list.append(filepath)

print("Directory Access: ", time.time() - start)

#%% Import Image  and Display
import napari
from aicsimageio import AICSImage 

img = AICSImage(filepath_list[0])
#img_m = AICSImage(filepath_list[0], reconstruct_mosaic = False)
#C1 = img.get_image_dask_data("YX", C=1)
#C1_m = img_m.get_image_dask_data("MYX", C=1)


print(img.dims)

import pyclesperanto_prototype as cle
cle.get_device()

print("Import: ", time.time() - start)

if 'viewer' not in globals():
    viewer = napari.Viewer()
    
#!nvidia-smi --query-gpu=memory.used --format=csv

#%% Ridge Filtering

C0 = img.get_image_dask_data("YX", C=0)
cle.push(C0)

C0_crop = cle.crop(C0, width = 1000, height = 1000, start_x = 7000, start_y = 7000)
viewer.add_image(C0_crop)

#C0_blur = cle.gaussian_blur(C0_crop, None, 1, 1, 0)
#viewer.add_image(C0_blur)

from skimage.filters import meijering #, sato, frangi, hessian

C0_mj = meijering(C0_crop, sigmas = range(1,5,1), black_ridges = False) # Meijering Ridge Filter
viewer.add_image(C0_mj)

C0_neurites = cle.greater_constant(C0_mj, None, constant = 0.04) #second order image threshold
viewer.add_labels(C0_neurites)

print("Neurite Filtering: ", time.time() - start)

#%% DAPI Filtering
C1 = img.get_image_dask_data("YX", C=1)
viewer.add_image(C1)
C1_crop = cle.crop(C1, width = 1000, height = 1000, start_x = 7000, start_y = 7000)
viewer.add_image(C1_crop)

cle.set_wait_for_kernel_finish(True) #???
C1_DAPI = cle.voronoi_otsu_labeling(C1, spot_sigma = 10)
#!nvidia-smi --query-gpu=memory.used --format=csv
viewer.add_labels(C1_DAPI)

print("Blob Filtering: ",time.time() - start)

#%% Blob Labelling - redo this to work with Big imagE? 

C0_max = cle.detect_maxima_box(C0_lb, radius_x = 2, radius_y = 2, radius_z = 0) #Local Maxima, automatic
C0_otsu = cle.threshold_otsu(C0_lb)
C0_spots = cle.binary_and(C0_max, C0_otsu) # Create binary where local maxima AND Otsu blobs exist
C0_blobs = cle.masked_voronoi_labeling(C0_spots, C0_otsu) # Convert binary map to label map and watershed with voronoi

viewer.add_labels(C0_blobs)
print("Blob Labeling: ", time.time() - start)
