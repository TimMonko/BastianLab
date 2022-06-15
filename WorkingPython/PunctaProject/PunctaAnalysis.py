# -*- coding: utf-8 -*-
"""
Puncta Analysis For PSD-95
Created on Wed May 25 09:41:55 2022

@author: TimMonko
"""
#%% Images Paths
import glob
import os 

file_type = "*.tif"
file_directory = "D:\Bastian Lab\Sophie\Date-20220517-PSD95-PunctaProjectAnalysis\Cropped_Cleared_Output"
os.chdir(file_directory)

# Create subdirectory to save label images to, use try except in case label folder already exists (as in instance of reruns or restarts)
label_directory = file_directory + "/Labels/"
try: 
    os.mkdir(label_directory)
except OSError:
    pass

filenames = glob.glob(file_type)

#%% Function Def
from aicsimageio import AICSImage
import pyclesperanto_prototype as cle
from skimage.filters import meijering
import napari_simpleitk_image_processing as nsitk
import time
import numpy as np
import pandas as pd
from skimage.io import imsave

def blob_filter(blob_image):
    PSD95_ms = cle.median_sphere(blob_image, None, 1.0, 1.0, 0.0)
    PSD95_ths = cle.top_hat_sphere(PSD95_ms, None, 5.0, 5.0, 0.0) #lower toightens up puncta
    PSD95_gb = cle.gaussian_blur(PSD95_ths, None, 1.0, 1.0, 0.0) # lower gauss seems important for not blurring too much
    PSD95_lb = cle.laplace_box(PSD95_gb)
    return(PSD95_lb)

def blob_label(LoG_image):
    PSD95_max = cle.detect_maxima_box(LoG_image, radius_x = 2, radius_y = 2, radius_z = 0) #Local Maxima, automatic
    PSD95_otsu = cle.threshold_otsu(LoG_image)
    PSD95_spots = cle.binary_and(PSD95_max, PSD95_otsu) # Create binary where local maxima AND Otsu blobs exist
    PSD95_blobs = cle.masked_voronoi_labeling(PSD95_spots, PSD95_otsu) # Convert binary map to label map and watershed with voronoi
    return(PSD95_blobs)
    
def ridge_filter(ridge_image):
    PSD95_gbridge = cle.gaussian_blur(ridge_image, None, 1.0, 1.0, 0.0)# Gaussian Blur 
    PSD95_sgbridge = cle.subtract_gaussian_background(PSD95_gbridge, None, 10.0, 10.0, 0.0) # Gaussian Background Subtraction
    PSD95_mjridge = meijering(PSD95_sgbridge, sigmas = range(6,12,2), black_ridges = False) # Meijering Ridge Filter
    return(PSD95_mjridge)

def ridge_label(ridge_image):
    PSD95_mj_to = cle.threshold_otsu(ridge_image) # Otsu of Ridges, binary
    PSD95_mj_cl = cle.closing_labels(PSD95_mj_to, None, 3.0) # Closing Labels, binary
    # PSD95_neurites = cle.connected_components_labeling_box(PSD95_mj_cl) # VERY TIME CONSUMING Connected Components Labeling, CCM
    return(PSD95_mj_cl)

def blobs_on_ridges(blob_label, ridge_label):
    not_neurite = cle.binary_not(ridge_label)
    distance_from_neurite = nsitk.signed_maurer_distance_map(not_neurite)
    mean_distance_map = cle.mean_intensity_map(distance_from_neurite, blob_label)
    blobs_close_by_neurite = cle.exclude_labels_with_map_values_out_of_range(
        mean_distance_map,
        blob_label,
        minimum_value_range=-100,
        maximum_value_range=8)
    return(blobs_close_by_neurite)
    
    
    
#%% Image Processing 

start_overall = time.time()
label_ratio_list = []
stats_blob_list = []

# Seems necessary for challenging image analyses - prevents GPU from trying to do math beyond VRAM capability, resulting in a slightly slower process but one that does not abort
cle.set_wait_for_kernel_finish(True)

for file in filenames:
    start = time.time()
    
    # Reset these variables in case image is skipped, so that summary table includes NaNs
    blob_ratio = np.nan
    ridge_ratio = np.nan
    total_blobs = np.nan
    ridge_area = np.nan
    
    # Import Image and extract specific channel
    img = AICSImage(file)
    #PSD95 = img.get_image_data("YX", C = img.channel_names.index('EGFP'))
    PSD95 = img.get_image_data("YX")
    
    # Filter grayscale image blobs, then create labels for the blobs
    PSD95_filter_blobs = blob_filter(PSD95)
    PSD95_blobs = blob_label(PSD95_filter_blobs)
    
    # Determine area of image covered by labels (non_zero) compared to background. This can be used to filter for images which extract too many labels, indicative of poor S/B.
    blob_ratio = np.count_nonzero(PSD95_blobs) / PSD95_blobs.size  
    
    # Arbitrary blob_area cut-off, but seems useful for PSD95-Puncta based images
    if blob_ratio <= 0.05: 
        
        # Filter grayscale image ridges, then create labels for the ridges
        PSD95_filter_ridges = ridge_filter(PSD95)
        PSD95_ridges = ridge_label(PSD95_filter_ridges)
        
        # Determine area of image covered by labels (non_zero) compared to background. This can be used to filter for images which extract too many labels, indicative of poor S/B. 
        ridge_area = np.count_nonzero(PSD95_ridges)
        ridge_ratio = ridge_area / PSD95_ridges.size
        
        # Arbitrary ridge_ratio cut-off, but seems useful for PSD95-extracted ridges
        if ridge_ratio <= 0.10:        
            
            # Function compares two label images, and extracts blobs if in proximity to ridges
            PSD95_blobs_on_PSD95_ridges = blobs_on_ridges(PSD95_blobs, PSD95_ridges)
            
            # Saving stack of resulting images
            fsave = label_directory + file + "_labels.tif"        
            stacked_results = np.stack((PSD95, PSD95_ridges, PSD95_blobs_on_PSD95_ridges), axis = 0).astype(np.uint16)
            imsave(fsave, stacked_results, check_contrast = False)
            
            # Get statistics for the background (0) and all labels (1:n), then append to list to later concatenate
            stats_blobs = (pd.DataFrame(cle.statistics_of_background_and_labelled_pixels(PSD95, PSD95_blobs_on_PSD95_ridges))
                         .assign(filename = file)
                         )
            stats_blob_list.append(stats_blobs)
            
            stats_ridges = (pd.DataFrame(cle.statistics_of_background_and_labelled_pixels(PSD95, PSD95_ridges))
                         .assign(filename = file)
                         )
            stats_blob_list.append(stats_blobs) 
            
            # Maximum label refers to total number of extracted blobs
            total_blobs = stats_blobs['original_label'].max()
            

    # Create a summary table of area ratio and total number of blobs, for later referencing
    scale = 0.182 # img.physical_pixel_sizes.X # 0.182um x 0.182um / pixel
    ridge_squm = ridge_area * scale * scale
    
    label_summary = (pd.DataFrame(
        {'filename' : file,
         'blob_ratio' : blob_ratio,
         'ridge_ratio' : ridge_ratio,
         'total_blobs' : total_blobs,
         'ridge_pixel_area' : ridge_area, 
         'pixel_size' : scale,
         'ridge_squm' : ridge_squm,
         'blob_per_squm' : total_blobs / ridge_squm},
        index = [file]))
    print(label_summary)
    label_ratio_list.append(label_summary)
        
    print("took ", time.time() - start, " seconds")
    
print("Overall Time: ", time.time() - start_overall)

# Using pandas, concatenate list and then save to csv
stats_all = pd.concat(stats_blob_list)
stats_all.to_csv('stats.csv')

label_summary_all = pd.concat(label_ratio_list)
label_summary_all.to_csv('label_summary.csv')

#%% Napari Viewing
# import napari
# viewer = napari.Viewer()

