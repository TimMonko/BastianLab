"""
Sample Pipeline for Bastian Lab
Created on Mon Mar  7 15:16:53 2022
Began after Tutorial 30 of APEER Micro Python tutorials
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
print(img.dims)

if 'viewer' not in globals():
    viewer = napari.Viewer()

import pyclesperanto_prototype as cle
cle.get_device()

print("Import: ", time.time() - start)

#%% Ridge Filtering

C0 = img.get_image_dask_data("YX", C=0)
cle.push(C0)

C0_crop = cle.crop(C0, width = 1000, height = 1000, start_x = 7000, start_y = 7000)
viewer.add_image(C0_crop)

C0_blur = cle.gaussian_blur(C0_crop, None, 1, 1, 0)
viewer.add_image(C0_blur)

from skimage.filters import meijering #, sato, frangi, hessian

C0_sgb = cle.subtract_gaussian_background(C0_blur, None, 10.0, 10.0, 0.0)
viewer.add_image(C0_sgb)
C0_mj = meijering(C0_sgb, sigmas = range(6,12,2), black_ridges = False) # Meijering Ridge Filter
viewer.add_image(C0_mj)

#%% DAPI Filtering
C1 = img.get_image_dask_data("YX", C=1)

C1_crop = cle.crop(C1, width = 1000, height = 1000, start_x = 7000, start_y = 7000)

C1_blur = cle.gaussian_blur(C1_crop, None, 3.0, 3.0, 0.0)

C1_ths = cle.top_hat_sphere(C1_blur, None, 20.0, 20.0, 0.0)

C1_DAPI = cle.voronoi_otsu_labeling(C1_ths, spot_sigma = 10)
viewer.add_labels(C1_DAPI)


print("Blob Filtering: ",time.time() - start)

#%% Blob Labelling

C0_max = cle.detect_maxima_box(C0_lb, radius_x = 2, radius_y = 2, radius_z = 0) #Local Maxima, automatic
C0_otsu = cle.threshold_otsu(C0_lb)
C0_spots = cle.binary_and(C0_max, C0_otsu) # Create binary where local maxima AND Otsu blobs exist
C0_blobs = cle.masked_voronoi_labeling(C0_spots, C0_otsu) # Convert binary map to label map and watershed with voronoi

viewer.add_labels(C0_blobs)
print("Blob Labeling: ", time.time() - start)

#%% Ridge Filtering

from skimage.filters import meijering #, sato, frangi, hessian

C0s_gb = cle.gaussian_blur(C0, None, 1.0, 1.0, 0.0)# Gaussian Blur 
C0s_sgb = cle.subtract_gaussian_background(C0s_gb, None, 10.0, 10.0, 0.0) # Gaussian Background Subtraction
C0s_mj = meijering(C0s_sgb, sigmas = range(6,12,2), black_ridges = False) # Meijering Ridge Filter

viewer.add_image(C0s_mj)
print("Ridge Filtering: ", time.time() - start)

#%% Ridge Labelling

C0s_mj_to = cle.threshold_otsu(C0s_mj) # Otsu of Ridges, binary
C0s_mj_cl = cle.closing_labels(C0s_mj_to, None, 3.0) # Closing Labels, binary
C0s_neurites = cle.connected_components_labeling_box(C0s_mj_cl) # Connected Components Labeling, CCM

viewer.add_labels(C0s_neurites)
print("Ridge Labelling: ", time.time() - start)

#%% Ridge Signed Maurer Distance Map 

import napari_simpleitk_image_processing as nsitk

not_neurite = cle.binary_not(C0s_mj_to)
distance_from_neurite = nsitk.signed_maurer_distance_map(not_neurite)

mean_distance_map = cle.mean_intensity_map(distance_from_neurite, C0_blobs)

viewer.add_image(mean_distance_map, colormap = 'turbo')
print("Signed Maurer Distance Map: ", time.time() - start)
#%% Filter Blobs by SMD Map

objects_close_by_neurite = cle.exclude_labels_with_map_values_out_of_range(
    mean_distance_map,
    C0_blobs,
    minimum_value_range=-100,
    maximum_value_range=5)

viewer.add_image(objects_close_by_neurite, colormap= 'turbo')
print("Filter Blobs by SMD Map: ", time.time() - start)