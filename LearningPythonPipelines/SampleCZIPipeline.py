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
path = "PSD95-Images/"
search_str = "*.ti*"
filepath_list = []

for path, subdirs, files in os.walk(path):
        for file in fnmatch.filter(files, search_str):
            filepath = os.path.join(path, file)
            print(filepath)
            filepath_list.append(filepath)
            
import napari
from aicsimageio import AICSImage
img = AICSImage(filepath_list[2])
print(img.dims)


#https://allencellmodeling.github.io/aicsimageio/_static/v3/
viewer = napari.Viewer()
#viewer.add_image(img.data, name = 'raw') # or raw.dask_data. Remember that the attribute is needed because the AICSimage class incorporates many things in the meta data 
# napari.view_image(raw.dask_data)
viewer.add_image(img.data, channel_axis = 1, name = img.channel_names)

# Import scikit-image's filtering module
from skimage import filters
from skimage import exposure
from skimage import morphology
from skimage import feature
from skimage import measure
from skimage import segmentation
from skimage import restoration
from scipy import ndimage
import numpy as np

#The .data and .xarray_data properties will load the whole scene into memory. The .get_image_data function will load the whole scene into memory and then retrieve the specified chunk.
C0 = img.get_image_data("YX",C=0) #keep only the data in the parantheses, select by channel type)
viewer.add_image(C0)

LoG = ndimage.gaussian_laplace(C0, sigma = 2)
viewer.add_image(LoG, name = "LoG")
viewer.add_image(filters.difference_of_gaussians(C0, low_sigma = 1, high_sigma = 2), name = 'DoG')

foreground = C0 >= filters.threshold_otsu(C0)
viewer.add_labels(foreground)

## THIS IS IT HURRAY https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html

viewer.add_image(exposure.equalize_hist(C0))
blobs = feature.blob_log(C0, threshold=.1)
viewer.add_image(blobs)
# I understand all code before this. I'm now at the processing stage hurray!



foreground_processed = morphology.remove_small_holes(foreground, 60)
foreground_processed = morphology.remove_small_objects(foreground_processed, min_size=50)
viewer.layers['foreground'].data = foreground_processed
distance = ndimage.distance_transform_edt(foreground_processed)
viewer.add_image(distance);


smoothed_distance = filters.gaussian(distance, 10)
viewer.layers['distance'].data = smoothed_distance

peak_local_max = feature.peak_local_max(
    smoothed_distance,
    footprint=np.ones((7, 7), dtype=np.bool),
    indices=False,
    labels=measure.label(foreground_processed)
)
peaks = np.nonzero(peak_local_max)

viewer.add_points(np.array(peaks).T, name='peaks', size=5, face_color='red')



new_peaks = np.round(viewer.layers['peaks'].data).astype(int).T
seeds = np.zeros(C0.shape, dtype=bool)
seeds[(new_peaks[0], new_peaks[1])] = 1

markers = measure.label(seeds)
nuclei_segmentation = segmentation.watershed(
    -smoothed_distance, 
    markers, 
    mask=foreground_processed
)

viewer.add_labels(nuclei_segmentation);

viewer.layers['nuclei_segmentation'].save('nuclei-automated-segmentation.tif', plugin='builtins');

