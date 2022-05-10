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
start = time.time()

import napari
from aicsimageio import AICSImage 
#from aicsimageio.readers.czi_reader import CziReader

img = AICSImage(filepath_list[0])
print(img.dims)


if 'viewer' not in globals():
    viewer = napari.Viewer()
    
!nvidia-smi --query-gpu=memory.used --format=csv

C1 = img.get_image_dask_data("YX", C=1)
C1_chunk = C1.rechunk(1000)
# viewer.add_image(C1_chunk)

print("Import: ", time.time() - start)

#%% Cle
# cle.set_wait_for_kernel_finish(True) #???
import pyclesperanto_prototype as cle
print(cle.get_device())

start = time.time()
cle.set_wait_for_kernel_finish(True)
C1_blur = cle.gaussian_blur(C1, None, 5, 5, 0)
print("Cle: ",time.time() - start)

!nvidia-smi --query-gpu=memory.used --format=csv

start = time.time()
C1_tophat = cle.top_hat_sphere(C1, None, 20, 20, 0)
print("Cle: ",time.time() - start)
#%% Skimage
from skimage import filters, morphology

start = time.time()
C1_skiblur = filters.gaussian(C1_chunk, sigma = 5)
print("Ski: ",time.time() - start)

start = time.time()
footprint = morphology.disk(20)
C1_skitophat = morphology.white_tophat(C1_skiblur, footprint)
print("Ski: ",time.time() - start)

#cle.set_wait_for_kernel_finish(False) #???


viewer.add_image(C1_blur)

cle.set_wait_for_kernel_finish(True) #???

viewer.add_image(C1_tophat)
    
#!nvidia-smi --query-gpu=memory.used --format=csv

#%% Ridge Filtering
C0 = img.get_image_dask_data("YX", C=0)
viewer.add_image(C0)
cle.push(C0)

#C0_crop = cle.crop(C0, width = 1000, height = 1000, start_x = 7000, start_y = 7000)
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

cle.set_wait_for_kernel_finish(True) #???
C1_blur = cle.gaussian_blur(C1, None, 5, 5, 0)
C1_tophat = cle.top_hat_sphere(C1_blur, None, 20, 20, 0)
viewer.add_image(C1_tophat)
C1_DAPI = cle.voronoi_otsu_labeling(C1_tophat, spot_sigma = 10, outline_sigma = 5)
viewer.add_labels(C1_DAPI)
!nvidia-smi --query-gpu=memory.used --format=csv

print("Blob Filtering: ",time.time() - start)

#%% Blob Labelling - redo this to work with Big imagE? 

C0_max = cle.detect_maxima_box(C0_lb, radius_x = 2, radius_y = 2, radius_z = 0) #Local Maxima, automatic
C0_otsu = cle.threshold_otsu(C0_lb)
C0_spots = cle.binary_and(C0_max, C0_otsu) # Create binary where local maxima AND Otsu blobs exist
C0_blobs = cle.masked_voronoi_labeling(C0_spots, C0_otsu) # Convert binary map to label map and watershed with voronoi

viewer.add_labels(C0_blobs)
print("Blob Labeling: ", time.time() - start)
