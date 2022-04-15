"""
Sample Pipeline for Bastian Lab
Created on Mon Mar  7 15:16:53 2022
Began after Tutorial 30 of APEER Micro Python tutorials
@author: Tim M
"""

import os, fnmatch

# See Current Directory structure
for root, subdirs, files in os.walk("."):
    path = root.split(os.sep) # split at separator (/ or \)
    print((len(path) -1) * '---', os.path.basename(root))
    for file in files:
        print(len(path) * '---', file)   

# Use a path for the files of interest, can be a parent folder or a specific subdirectory
path = "BatchInput/"
search_str = "*.ti*"
filepath_list = []

for path, subdirs, files in os.walk(path):
        for file in fnmatch.filter(files, search_str):
            filepath = os.path.join(path, file)
            print(filepath)
            filepath_list.append(filepath)
            
# At this point, I believe it would be possible to continue the for loop, but for cleaner code could also separate out and then iterate on the new list. I believe that creating the list of files is a good stopping point, but maybe processing them first would be good so as not to fill up the whole memory. Could also use functions and then place in above code, but I daresay this is less readable (although more similar to imageJ-scripting)


# Pre-processing here. 1) load in image 2) process 3) save output to list. This should save memory compared to bringing in all images and printing. UNLESS there is a need to see all the raw images in python, which there may be, but that can be saved into it's own list
from matplotlib import pyplot as plt
from skimage import io
from skimage.filters import gaussian

rawimage_list = []
preprocess_list = []

for image in filepath_list:
    raw = io.imread(image, as_gray = True)
    plt.imshow(raw)
    rawimage_list.append(raw)
    gaussian_img = gaussian(raw, sigma = 1, mode = 'constant', cval = 0.0)
    plt.imshow(gaussian_img)
    preprocess_list.append(gaussian_img)

   
plt.imshow(rawimage_list[1])
plt.imshow(preprocess_list[1])
