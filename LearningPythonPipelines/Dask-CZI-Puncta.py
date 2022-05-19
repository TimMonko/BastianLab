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
path = "20220517-PSD95-PunctaProjectAnalysis/"
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

img = AICSImage(filepath_list[125])
print(img.dims)

if 'viewer' not in globals():
    viewer = napari.Viewer()

print("Import: ", time.time() - start)

#%% Blob Filtering
import pyclesperanto_prototype as cle
import napari_simpleitk_image_processing as nsitk
cle.get_device()

start = time.time()

PSD95 = img.get_image_dask_data("YX", channel_names = 'EGFP')
viewer.add_image(PSD95)

PSD95_ms = cle.median_sphere(PSD95, None, 1.0, 1.0, 0.0) # good, 2 maybe ok
PSD95_ths = cle.top_hat_sphere(PSD95_ms, None, 5.0, 5.0, 0.0) # good

PSD95_log = cle.laplace_box(cle.gaussian_blur(PSD95_ths, None, 1.5, 1.5, 0))
PSD95_blobs = cle.voronoi_otsu_labeling(PSD95_log, None, 1.0, 1.0)


                               
# PSD95_logf = nsitk.laplacian_of_gaussian_filter(PSD95_ths, 1.5)
# PSD95_logf_in = nsitk.invert_intensity(PSD95_logf)

# PSD95_clemax = nsitk.h_maxima(PSD95_clelog, 300)
# PSD95_hmax = nsitk.h_maxima(PSD95_logf_in, 300.0)
# PSD95_blobs = cle.voronoi_otsu_labeling(PSD95_hmax, None, 1.0, 1.0)


# viewer.add_image(PSD95_hmax, name='LoG-H-Max')
# viewer.add_labels(PSD95_blobs)
print("Blob Filtering: ",time.time() - start)

#%% Blob Labelling

# start = time.time()

# PSD95_max = cle.detect_maxima_box(PSD95, radius_x = 2, radius_y = 2, radius_z = 0) #Local Maxima, automatic
# PSD95_otsu = cle.threshold_otsu(PSD95_lb)
# PSD95_spots = cle.binary_and(PSD95_max, PSD95_otsu) # Create binary where local maxima AND Otsu blobs exist
# PSD95_blobs = cle.masked_voronoi_labeling(PSD95_spots, PSD95_otsu) # Convert binary map to label map and watershed with voronoi

# viewer.add_labels(PSD95_blobs)
# print("Blob Labeling: ", time.time() - start)

#%% Ridge Filtering

from skimage.filters import meijering #, sato, frangi, hessian

PSD95_gbridge = cle.gaussian_blur(PSD95, None, 1.0, 1.0, 0.0)# Gaussian Blur 
PSD95_sgbridge = cle.subtract_gaussian_background(PSD95_gbridge, None, 10.0, 10.0, 0.0) # Gaussian Background Subtraction # Use for puncta?
PSD95_mjridge = meijering(PSD95_sgbridge, sigmas = range(6,12,2), black_ridges = False) # Meijering Ridge Filter

viewer.add_image(PSD95_mjridge)
print("Ridge Filtering: ", time.time() - start)

#%% Ridge Labelling

PSD95_mj_to = cle.threshold_otsu(PSD95_mjridge) # Otsu of Ridges, binary
PSD95_mj_cl = cle.closing_labels(PSD95_mj_to, None, 3.0) # Closing Labels, binary
PSD95_neurites = cle.connected_components_labeling_box(PSD95_mj_cl) # Connected Components Labeling, CCM

viewer.add_labels(PSD95_neurites)
print("Ridge Labelling: ", time.time() - start)

#%% Ridge Signed Maurer Distance Map 

import napari_simpleitk_image_processing as nsitk

not_neurite = cle.binary_not(PSD95_mj_to)
viewer.add_labels(not_neurite)
distance_from_neurite = nsitk.signed_maurer_distance_map(not_neurite)

mean_distance_map = cle.mean_intensity_map(distance_from_neurite, PSD95_blobs)

viewer.add_image(mean_distance_map, colormap = 'turbo')
print("Signed Maurer Distance Map: ", time.time() - start)
#%% Filter Blobs by SMD Map

objects_close_by_neurite = cle.exclude_labels_with_map_values_out_of_range(
    mean_distance_map,
    PSD95_blobs,
    minimum_value_range=-100,
    maximum_value_range=5)

viewer.add_image(objects_close_by_neurite, colormap= 'turbo')
print("Filter Blobs by SMD Map: ", time.time() - start)

#%% Blob math

num_labels = cle.maximum_of_all_pixels(objects_close_by_neurite)
label_excl_edges = cle.exclude_labels_on_edges(objects_close_by_neurite)

mean_intensity_map = cle.mean_intensity_map(PSD95, objects_close_by_neurite)
viewer.add_image(mean_intensity_map)

statistics = cle.statistics_of_labelled_pixels(PSD95, objects_close_by_neurite)
