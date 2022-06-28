# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 16:27:18 2022

@author: Bastian Lab
"""

#%% Images Paths
import glob
import os 

file_type = "*.czi"
file_directory = "D:\Bastian Lab\Tim\CD7 Troubleshooting"
os.chdir(file_directory)

# Create subdirectory to save label images to
label_directory = file_directory + "/Labels/"
try: 
    os.mkdir(label_directory)
except OSError:
    pass

filenames = glob.glob(file_type)

#%% Function Def
from aicsimageio import AICSImage

img = AICSImage(filenames[0])
img2 = AICSImage(filenames[1])


img.set_scene(1)
img2.set_scene(1)

img.xarray_data
img2.xarray_data

img.dims
img2.dims

img.physical_pixel_sizes
img2.physical_pixel_sizes