# -*- coding: utf-8 -*-
"""
DAPI / MAP2 Screening

Created on Wed Jun 15 13:07:31 2022

@author: Bastian Lab (Tim M and Daniel M)
"""

#%% Prepare directories

import glob
import os

file_type = "*.czi"
file_directory = "D:\Bastian Lab\Daniel\DAPI_MAP2_testing"
os.chdir(file_directory)
filenames = glob.glob(file_type)

output_directory = file_directory + "/Output/"

try:
    os.mkdir(output_directory)
except OSError:
    pass

#%% Import Modules

# Image Management
from aicsimageio import AICSImage
import pyclesperanto_prototype as cle
from skimage.io import imsave

# Data Management
import time
import numpy as np
import pandas as pd

img = AICSImage(filenames[0])

#%% Process Directory

start_overall = time.time()
summary_list = []

cle.set_wait_for_kernel_finish(True)

for file in filenames:
    
    img = AICSImage(file)
    scenes = img.scenes
    
    for scene in scenes:
        start = time.time()
        img.set_scene(scene)
        
        DAPI = img.get_image_dask_data("YX", C = img.channel_names.index("DAPI"))
        # DAPI_crop = cle.crop(DAPI, start_x = 5000, start_y = 5000, width = 3000, height = 3000)
        
        DAPI_rp = cle.replace_intensity(DAPI, value_to_replace = 0, value_replacement = 300)
        
        DAPI_maxima_blur = cle.gaussian_blur(DAPI_rp, sigma_x = 10, sigma_y = 10, sigma_z = 0)
        DAPI_maxima_spots = cle.detect_maxima_box(DAPI_maxima_blur, radius_x = 0, radius_y = 0, radius_z = 0)
        
        DAPI_threshold_blur = cle.gaussian_blur(DAPI_rp, None, 2, 2, 0)
        DAPI_threshold = cle.threshold(DAPI_threshold_blur, None, 500)
        
        DAPI_selected_spots = cle.binary_and(DAPI_threshold, DAPI_maxima_spots)
        DAPI_labels = cle.masked_voronoi_labeling(DAPI_selected_spots, DAPI_threshold)    
        
        DAPI_labels_filter = cle.exclude_labels_outside_size_range(DAPI_labels, None, 3, 2000)
        
        # Could Filter out oblong DAPI particles 
        
        print("DAPI took", time.time() - start, "seconds")
        
        DAPI_rp = None
        DAPI_labels = None
        
        MAP2_start = time.time()
        MAP2 = img.get_image_dask_data("YX", C = img.channel_names.index("AF647"))
        # MAP2_crop = cle.crop(MAP2, start_x = 5000, start_y = 5000, width = 3000, height = 3000)
        ## REPLACE ZERO WITH MEAN INTENSITY
        #MAP2_mi = cle.mean_of_all_pixels(MAP2)
        MAP2_rp = cle.replace_intensity(MAP2, value_to_replace = 0, value_replacement = 70)
        
        MAP2_th = cle.top_hat_sphere(MAP2_rp, None, 5, 5, 0)
        MAP2_blur = cle.gaussian_blur(MAP2_th, None, 2, 2, 0)
        MAP2_threshold = cle.threshold(MAP2_blur, None, 35)
        print("MAP2_threshold", time.time() - MAP2_start, "seconds")
        
        MAP2_rp = None
        MAP2_th = None
        MAP2_blur = None
        
        MAP2_start_label = time.time()
        MAP2_labels = cle.connected_components_labeling_box(MAP2_threshold)
        MAP2_threshold = None
        print("MAP2_labels", time.time() - MAP2_start_label, "seconds")
        
        ### THIS IS THE SPOT TO FIX ###
        # CONSIDER COMPARING RECONSTRUCTED DAPI LABELS BACKWARDS TO THE CONNECTED COMPONENT MAP? Even then, still requires CCL map, but at least would not require math on the entire image???)
        #MAP2_labels_filter = cle.exclude_small_labels(MAP2_labels, None, 500)
        
        MAP2_labels_filter = MAP2_labels
        print("MAP2_labels", time.time() - MAP2_start_label, "seconds")

    
        overlap_count =  cle.label_overlap_count_map(DAPI_labels_filter, MAP2_labels_filter)
        overlap_labels =  cle.connected_components_labeling_box(overlap_count)
        
        total_neurons = overlap_labels.max()
        
        MAP2_area = np.count_nonzero(MAP2_labels_filter)
        scale = img.physical_pixel_sizes.X # um / pixel
        MAP2_um2 = MAP2_area * scale * scale
        MAP2_um2_per_neuron = MAP2_um2 / total_neurons
    
        summary = (pd.DataFrame(
            {'scene' : img.current_scene,
             'total_neurons' : overlap_labels.max(),
             'total_DAPI' : DAPI_labels_filter.max(),
             'MAP2_um2' : MAP2_um2,
             'MAP2_um2_per_neuron' : MAP2_um2_per_neuron},
            index = [file]))
        
        summary_list.append(summary)
        print(summary)
        
        fsave = output_directory + file + img.current_scene + "_labels.tif"
        stacked_results = np.stack((MAP2_labels_filter, DAPI_labels_filter, overlap_labels), axis = 0).astype(np.uint16)
        imsave(fsave, stacked_results[:, 5000:8000, 5000:8000], check_contrast = False)
        
        print("took ", time.time() - start, "seconds")
    
print("Overall Time: ", time.time() - start_overall)

summary_all = pd.concat(summary_list)
summary_all.to_csv('summary.csv')


#%%

import napari
viewer = napari.Viewer()

# viewer.add_image(MAP2)
# viewer.add_image(MAP2_crop)
# viewer.add_image(DAPI_crop)
# viewer.add_labels(MAP2_labels)
# viewer.add_labels(DAPI_labels)
# viewer.add_labels(overlap_labels)

# viewer.add_image(MAP2_bg)
# viewer.add_image(MAP2_blur)
# viewer.add_image(MAP2_rp)

#%%
# from cellpose import models
# model = models.Cellpose(gpu=False, model_type = 'nuclei')

# model.eval(DAPI_crop, diameter = None, channels = [0,0])

