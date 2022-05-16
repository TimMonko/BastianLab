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
path = "2022-04-14/"
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

img = AICSImage(filepath_list[8])
print(img.dims)

if 'viewer' not in globals():
    viewer = napari.Viewer()
    



C1_chunk = C1.rechunk(1000)
# viewer.add_image(C1_chunk)

print("Import: ", time.time() - start)

#%% Cle Blob Filtering
import pyclesperanto_prototype as cle
print(cle.get_device())

C1 = img.get_image_dask_data("YX", C=1)


start = time.time()
C1_blur = cle.gaussian_blur(C1, None, 5, 5, 0)
print("Cle: ",time.time() - start) # 32s

#!nvidia-smi --query-gpu=memory.used --format=csv

start = time.time()
C1_tophat = cle.top_hat_sphere(C1_blur, None, 20, 20, 0)
print("Cle: ",time.time() - start) # 2.2s


start = time.time()
C1_DAPI = cle.voronoi_otsu_labeling(C1_tophat, spot_sigma = 10, outline_sigma = 3)
print("Cle: ",time.time() - start) # 2.2s

# viewer.add_image(C1)
# viewer.add_image(C1_blur)
# viewer.add_image(C1_tophat)
viewer.add_labels(C1_DAPI)


#%% Skimage Blob filtering
# from skimage import filters, morphology

# start = time.time()
# C1_skiblur = filters.gaussian(C1, sigma = 5)
# print("Ski: ",time.time() - start) # 44s

# start = time.time()
# footprint = morphology.disk(20)
# C1_skitophat = morphology.white_tophat(C1_skiblur, footprint)
# print("Ski: ",time.time() - start) # 772s

    
#%% Ridge Filtering
C0 = img.get_image_dask_data("YX", C=0)

#C0_crop = cle.crop(C0, width = 1000, height = 1000, start_x = 7000, start_y = 7000)
#viewer.add_image(C0_crop)

C0_blur = cle.mean_sphere(C0, None, 1, 1, 0)
viewer.add_image(C0_blur)

C0_neurites = cle.greater_constant(C0_blur, None, constant = 250)
viewer.add_labels(C0_neurites)

print("Neurite Filtering: ", time.time() - start)

overlap = cle.label_nonzero_pixel_count_ratio_map(C1_DAPI, C0_neurites)
viewer.add_image(overlap)

DAPI_MAP2 = cle.greater_or_equal_constant(overlap, None, constant = 0.5)
viewer.add_labels(DAPI_MAP2)

#%% Blob Labelling - redo this to work with Big imagE? 

C0_max = cle.detect_maxima_box(C0_lb, radius_x = 2, radius_y = 2, radius_z = 0) #Local Maxima, automatic
C0_otsu = cle.threshold_otsu(C0_lb)
C0_spots = cle.binary_and(C0_max, C0_otsu) # Create binary where local maxima AND Otsu blobs exist
C0_blobs = cle.masked_voronoi_labeling(C0_spots, C0_otsu) # Convert binary map to label map and watershed with voronoi

viewer.add_labels(C0_blobs)
print("Blob Labeling: ", time.time() - start)
