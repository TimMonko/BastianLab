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

def gaussian_blur_label(image, gaussian_sigma):
    bg = cle.subtract_gaussian_background(image, None, 20.0, 20.0, 0.0)
    blur = cle.gaussian_blur(bg, None, gaussian_sigma, gaussian_sigma, 0)
    threshold = cle.threshold_otsu(blur)
    labels = cle.connected_components_labeling_box(threshold)
    return(labels)

#%%
img = AICSImage(filenames[0])
scenes = img.scenes

MAP2 = img.get_image_dask_data("YX", C = img.channel_names.index("AF647"))

MAP2_crop = cle.crop(MAP2, start_x = 5000, start_y = 5000, width = 3000, height = 3000)

MAP2_labels = gaussian_blur_label(MAP2_crop, 8)


DAPI = img.get_image_dask_data("YX", C = img.channel_names.index("DAPI"))
DAPI_crop = cle.crop(DAPI, start_x = 5000, start_y = 5000, width = 3000, height = 3000)

DAPI_labels = gaussian_blur_label(DAPI_crop, 2)

overlap_count =  cle.label_overlap_count_map(DAPI_labels, MAP2_labels)
overlap_labels =  cle.connected_components_labeling_box(overlap_count)


#stats = (pd.DataFrame(cle.statistics_of_labelled_pixels(overlap_labels, DAPI_crop))
#         .assign(filename = filenames[0])
#         )
total_neurons = overlap_labels.max()

summary = (pd.DataFrame(
    {'scene' : img.current_scene,
     'total_neurons' : overlap_labels.max(),
     'total_DAPI' : DAPI_labels.max()},
    index = [filenames[0]]))

print(summary)

summary.to_csv('stats.csv')

fsave = output_directory + filenames[0] + "_labels.tif"
stacked_results = np.stack((MAP2_crop, DAPI_crop, MAP2_labels, DAPI_labels, overlap_labels), axis = 0).astype(np.uint16)
imsave(fsave, stacked_results, check_contrast = False)

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
     
        MAP2 = img.get_image_dask_data("YX", C = img.channel_names.index("AF647"))
        ##
        ## REPLACE ZERO WITH MEAN INTENSITY
        ##
        MAP2_mi = cle.mean_of_all_pixels(MAP2)
        MAP2_rp = cle.replace_intensity(MAP2, value_to_replace = 0, value_replacement = MAP2_mi)
        MAP2_bg = cle.subtract_gaussian_background(MAP2, None, 20.0, 20.0, 0.0)

        MAP2_blur = cle.gaussian_blur(MAP2_bg, None, 8, 8, 0)
        MAP2_threshold = cle.threshold_otsu(MAP2_blur)
        MAP2_labels = cle.connected_components_labeling_box(MAP2_threshold)

        # #MAP2_crop = cle.crop(MAP2, start_x = 5000, start_y = 5000, width = 3000, height = 3000)
        # MAP2_labels = gaussian_blur_label(MAP2, 8)
    
    
        DAPI = img.get_image_dask_data("YX", C = img.channel_names.index("DAPI"))
        #DAPI_crop = cle.crop(DAPI, start_x = 5000, start_y = 5000, width = 3000, height = 3000)
        DAPI_bg = cle.subtract_gaussian_background(DAPI, None, 20.0, 20.0, 0.0)
        DAPI_blur = cle.gaussian_blur(DAPI_bg, None, 2, 2, 0)
        DAPI_threshold = cle.threshold_otsu(DAPI_blur)
        DAPI_labels = cle.connected_components_labeling_box(DAPI_threshold)
        # DAPI_labels = gaussian_blur_label(DAPI, 2)
    
        overlap_count =  cle.label_overlap_count_map(DAPI_labels, MAP2_labels)
        overlap_labels =  cle.connected_components_labeling_box(overlap_count)
        
        total_neurons = overlap_labels.max()
        
        MAP2_area = np.count_nonzero(MAP2_labels)
        scale = img.physical_pixel_sizes.X # um / pixel
        MAP2_um2 = MAP2_area * scale * scale
        MAP2_um2_per_neuron = MAP2_um2 / total_neurons

    
        summary = (pd.DataFrame(
            {'scene' : img.current_scene,
             'total_neurons' : overlap_labels.max(),
             'total_DAPI' : DAPI_labels.max(),
             'MAP2_um2' : MAP2_um2,
             'MAP2_um2_per_neuron' : MAP2_um2_per_neuron},
            index = [file]))
        
        summary_list.append(summary)
        print(summary)
        
        fsave = output_directory + file + img.current_scene + "_labels.tif"
        stacked_results = np.stack((MAP2, DAPI, MAP2_labels, DAPI_labels, overlap_labels), axis = 0).astype(np.uint16)
        imsave(fsave, stacked_results, check_contrast = False)
        
        print("took ", time.time() - start, "seconds")
    
print("Overall Time: ", time.time() - start_overall)

summary_all = pd.concat(summary_list)
summary_all.to_csv('summary.csv')


#%%

import napari

viewer = napari.Viewer()

viewer.add_image(MAP2)
viewer.add_image(DAPI)
viewer.add_labels(MAP2_labels)
viewer.add_labels(DAPI_labels)
viewer.add_labels(overlap_labels)

viewer.add_image(MAP2_bg)
viewer.add_image(MAP2_blur)
