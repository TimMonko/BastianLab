# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 16:27:18 2022

@author: Bastian Lab
"""

#%% Images Paths
import glob
import os 

file_type = "*.czi"
file_directory = "D:\Bastian Lab\Tommy\D5_27_22_HCC"
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

img = AICSImage(filenames[2])
img2 = AICSImage(filenames[4])


img.set_scene(1)
img2.set_scene(1)
