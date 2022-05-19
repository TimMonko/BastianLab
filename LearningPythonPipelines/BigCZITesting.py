"""
Sample Pipeline for Bastian Lab
Created on Mon Mar  7 15:16:53 2022
Began after Tutorial 30 of APEER Micro Python tutorials
@author: Tim M
"""


#https://github.com/haesleinhuepf/BioImageAnalysisNotebooks/blob/main/docs/32_tiled_image_processing/tiled_nuclei_counting.ipynb 
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

#%% IMPORT AND DISPLAY
import napari
from aicsimageio import AICSImage 
from aicsimageio.readers import CziReader

img = AICSImage("BigDAPIMap2Images/2.25.22 HCC_MAP2_ICC Imaging_10x_0.95NA_AF647 + DAPI_4.14.22_RowE_G.czi")

img = AICSImage(filepath_list[0])
dask_aics = img.dask_data
img_m = AICSImage(filepath_list[0], reconstruct_mosaic = False)
dask_aicsrm = img_m.dask_data
img_czi = CziReader(filepath_list[0])


print(img.dims)

import pyclesperanto_prototype as cle
cle.get_device()

print("Import: ", time.time() - start)

if 'viewer' not in globals():
    viewer = napari.Viewer()
    
#!nvidia-smi --query-gpu=memory.used --format=csv


#%% pylibCZIrw
"""
At the moment, pylibCZIrw completely abstracts away the subblock concept, both in the reading and in the writing APIs.
If pylibCZIrw is extended in the future to support subblock-based access (e.g. accessing acquisition tiles), this API must not be altered.
The core concept of pylibCZIrw is focussing on reading and writing 2D image planes by specifying the dimension indices and its location in order to only read or write what is really needed.
"""
from pylibCZIrw import czi as pyczi
import json
from matplotlib import pyplot as plt
import matplotlib.cm as cm

with pyczi.open_czi(filepath_list[0]) as czidoc:
    md_xml = czidoc.raw_metadata
    print(md_xml)
    md_dict = czidoc.metadata

    # show some parts of it
    print(json.dumps(md_dict["ImageDocument"]["Metadata"]["Information"]["Image"], sort_keys=False, indent=4))
    
my_roi = (200, 400, 800, 600)

with pyczi.open_czi(filepath_list[0]) as czidoc:
    ch0 = czidoc.read(roi=my_roi, plane={'C': 0})
    ch1 = czidoc.read(roi=my_roi, plane={'C': 1})

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(ch0[...,0], cmap=cm.inferno, vmin=100, vmax=4000)
ax[0].set_title("ch0")
ax[1].imshow(ch1[...,0], cmap=cm.Greens_r, vmin=100, vmax=4000)
ax[1].set_title("ch1")


#%%CZTILE IMPORT

# import the "tiling strategy" from the cztile package
from cztile.fixed_total_area_strategy import AlmostEqualBorderFixedTotalAreaStrategy2D
from cztile.tiling_strategy import Rectangle as czrect

# create the filename for the new CZI image file
newczi_tile = os.path.join(os.getcwd(), "newczi_tilewise.czi")

# specify the original CZI file to be read
czifile_orig = os.path.join(filepath_list[0])

# create a "tile" by specifying the desired tile dimension and the
# minimum required overlap between tiles (depends on the processing)
tiler = AlmostEqualBorderFixedTotalAreaStrategy2D(total_tile_width=1600,
                                                  total_tile_height=1400,
                                                  min_border_width=128)

# create CZI instance to read some metadata 
with pyczi.open_czi(czifile_orig) as czidoc_r:
    
    # get the size of the bounding rectange for the scence
    tiles = tiler.tile_rectangle(czidoc_r.scenes_bounding_rectangle[0])
    
# show the created tile locations
for tile in tiles:
    print(tile.roi.x, tile.roi.y, tile.roi.w, tile.roi.h)
    
#%% CZTILE RW IMPORT
from skimage.filters import gaussian
import numpy as np


# define a generic 2D processing function - could by a DL segmentation as well
def process2d(image2d: np.ndarray, **kwargs: int) -> np.ndarray:

    # insert or modify the desired processing function here
    image2d = gaussian(image2d, sigma=kwargs["sigma"],
                       preserve_range=True,
                       mode='nearest').astype(image2d.dtype)

    return image2d


# open an existing CZI file for reading planes and a new CZI to insert
# the newly processed chunks into the new CZI file

# create to figures
fig1, ax1 = plt.subplots(1, 4, figsize=(16, 8))
fig2, ax2 = plt.subplots(1, 4, figsize=(16, 8))

# counter figures
a = 0

# open a CZI instance to read and in parallel one to write
with pyczi.open_czi(czifile_orig) as czidoc_r:
    with pyczi.create_czi(newczi_tile, exist_ok=True) as czidoc_w:

        # loop over all tiles created by the "tiler"
        for tile in tqdm(tiles):
            
            # read a specific tile from the CZI using the roi parameter
            tile2d = czidoc_r.read(plane={"C": 0},
                                   roi=(tile.roi.x,
                                        tile.roi.y,
                                        tile.roi.w,
                                        tile.roi.h)
                                  )
            
            # process the current tile using a function
            tile2d_processed = process2d(tile2d, sigma=11)
            
            # show sthe tiles to illustrate the idea
            ax1[a].imshow(tile2d[...,0], interpolation="nearest", cmap="gray",vmin=100, vmax=300)
            ax1[a].set_title("Tile: " + str(a))
            ax2[a].imshow(tile2d_processed[...,0], interpolation="nearest", cmap="gray", vmin=100, vmax=4000)
            ax2[a].set_title("Tile: " + str(a))
                                                                          
            # write the new CZI file using the processed data
            czidoc_w.write(tile2d_processed,
                           plane={"C": 0},
                           location=(tile.roi.x,
                                     tile.roi.y)
                          )
            
            a += 1    
    
    

#%% CZTILE STRATEGY
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.filters import gaussian

from cztile.fixed_total_area_strategy import AlmostEqualBorderFixedTotalAreaStrategy2D
from cztile.tiling_strategy import Rectangle as czrect

# define a generic 2D processing function - could by a DL segmentation as well
def process2d(image2d: np.ndarray, **kwargs: int) -> np.ndarray:

    # insert or modify the desired processing function here
    image2d = gaussian(image2d, sigma=kwargs["sigma"],
                       preserve_range=True,
                       mode='nearest').astype(image2d.dtype)

    return image2d



# create the "tiler"
tiler = AlmostEqualBorderFixedTotalAreaStrategy2D(total_tile_width=1200,
                                                  total_tile_height=1000,
                                                  min_border_width=128)

# create the tiles
tiles = tiler.tile_rectangle(czrect(x=0, y=0, w=img.shape[1], h=img.shape[0]))

# show the tile locations
for tile in tiles:
    print(tile.roi.x, tile.roi.y, tile.roi.w, tile.roi.h)

fig1, ax1 = plt.subplots(1, 4, figsize=(16, 8))
fig2, ax2 = plt.subplots(1, 4, figsize=(16, 8))
a = 0

# iterate over all tiles and apply the processing
for tile in tqdm(tiles):
    
    #print(tile.roi.x, tile.roi.y, tile.roi.w, tile.roi.h)

    # get a single frame based on the tile coordinates and size
    tile2d = img[tile.roi.y:tile.roi.y + tile.roi.h, tile.roi.x:tile.roi.x + tile.roi.w]

    # do some processing here
    tile2d_processed = process2d(tile2d, sigma=15)
    
    ax1[a].imshow(tile2d, interpolation="nearest", cmap="gray",vmin=100, vmax=2000)
    ax1[a].set_title("Tile: " + str(a))
    ax2[a].imshow(tile2d_processed, interpolation="nearest", cmap="gray",vmin=100, vmax=2000)
    ax2[a].set_title("Tile: " + str(a))

    # place frame inside the new image
    processed_img[tile.roi.y:tile.roi.y + tile.roi.h, tile.roi.x:tile.roi.x + tile.roi.w] = tile2d_processed
    
    a += 1


#%% Ridge Filtering

C0 = img.get_image_dask_data("YX", C=0)
cle.push(C0)

C0_crop = cle.crop(C0, width = 1000, height = 1000, start_x = 7000, start_y = 7000)
viewer.add_image(C0_crop)

#C0_blur = cle.gaussian_blur(C0_crop, None, 1, 1, 0)
#viewer.add_image(C0_blur)

from skimage.filters import meijering #, sato, frangi, hessian

C0_mj = meijering(C0_crop, sigmas = range(1,5,1), black_ridges = False) # Meijering Ridge Filter
viewer.add_image(C0_mj)

C0_neurites = cle.greater_constant(C0_mj, None, constant = 0.04) #second order image threshold
viewer.add_labels(C0_neurites)

print("Neurite Filtering: ", time.time() - start)

#%% DAPI Filtering
C1 = img.get_image_dask_data("YX", C=1)
viewer.add_image(C1)
C1_crop = cle.crop(C1, width = 1000, height = 1000, start_x = 7000, start_y = 7000)
viewer.add_image(C1_crop)

cle.set_wait_for_kernel_finish(True) #???
C1_DAPI = cle.voronoi_otsu_labeling(C1, spot_sigma = 10)
#!nvidia-smi --query-gpu=memory.used --format=csv
viewer.add_labels(C1_DAPI)

print("Blob Filtering: ",time.time() - start)

#%% Blob Labelling - redo this to work with Big imagE? 

C0_max = cle.detect_maxima_box(C0_lb, radius_x = 2, radius_y = 2, radius_z = 0) #Local Maxima, automatic
C0_otsu = cle.threshold_otsu(C0_lb)
C0_spots = cle.binary_and(C0_max, C0_otsu) # Create binary where local maxima AND Otsu blobs exist
C0_blobs = cle.masked_voronoi_labeling(C0_spots, C0_otsu) # Convert binary map to label map and watershed with voronoi

viewer.add_labels(C0_blobs)
print("Blob Labeling: ", time.time() - start)
