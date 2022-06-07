# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 16:19:32 2022

@author: Tim M + Daniel M
"""

#%% Image Paths

import glob
import os

file_type = "*.czi"
### YOU ONLY NEED TO CHANGE THE DIRECTORY ###
file_directory = "C:\\Users\Tim M\Desktop\DM_Cropping_Input"

os.chdir(file_directory)

filenames = glob.glob(file_type)

# Create subdirectory to save label images to, use try except in case label folder already exists (as in instance of reruns or restarts)
output_directory = file_directory + "/Output/"
try: 
    os.mkdir(output_directory)
except OSError:
    pass



#%%

from aicsimageio import AICSImage
from skimage.io import imsave

for file in filenames:

    img = AICSImage(file)
    dask_img = img.get_image_dask_data("CYX")
    
    # CHANGE THESE DIMENSIONS TO CROP DIFFERENTLY
    crop_img = dask_img[0:2,5000:8000, 5000:8000]
    
    fsave = output_directory + file + "_crop.tif"  
    imsave(fsave, crop_img, check_contrast = False)
    
#%% Napari
# import napari

# viewer = napari.Viewer()
# viewer.add_image(crop_img)
