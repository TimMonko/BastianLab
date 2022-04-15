"""
SUMMARY OF APEER-MICRO TUTORIALS
Created on Fri Feb 18 12:30:22 2022

@author: Tim M

Written starting with tutorial 25 from Sreeni's series https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG
Intended to summarize basic principles of image analysis in python

"""

"""
READING IMAGES
"""

#SCIKITIMAGE
#pip install scikitimage
#scikit image use typical RGB loading format
from skimage import io
img_jpg = io.imread("Images/P123.jpg")
img_tiff = io.imread("Images/Snap-85.ome.tiff") #skimage appears to be able to load without tiffile package imported?

#OPEN-CV
#pip install opencv-python
# opencv uses BRG for images. If switching between spaces (like scikit to opencv), then will look incorrect. If kept within open cv then things look ok.
import cv2
img_cv = cv2.imread("Images/P123.jpg")

#TIFFFILE
#pip install tifffile
import tifffile
#RGB Images
img_tifffile = tifffile.imread("Images/Snap-85.ome.tiff") # produces same message as with SKIMAGE, likely uses same loa

#AICSImageIO - Seems to be a well made proprietary image reader from Allen Brain Institute
#https://allencellmodeling.github.io/aicsimageio/
#pip install aicsimageio
#pip install aicspylibczi #required to install format dependency for new proprietary file types 
from aicsimageio import AICSImage
img_AICS_czi = AICSImage("Images/weakPSD95mito.czi") # Interestingly, AICS will give an error that it cannot open a proprietary image format if the path is wrong, even though it does support the file type (after pip install aicsimageio[czi])

# ... From AICS github example. img.set_scene probably very important if we don't split scenes
# Get an AICSImage object
img = AICSImage("my_file.tiff")  # selects the first scene found
img.data  # returns 5D TCZYX numpy array
img.xarray_data  # returns 5D TCZYX xarray data array backed by numpy
img.dims  # returns a Dimensions object
img.dims.order  # returns string "TCZYX"
img.dims.X  # returns size of X dimension
img.shape  # returns tuple of dimension sizes in TCZYX order
img.get_image_data("CZYX", T=0)  # returns 4D CZYX numpy array

# Get the id of the current operating scene
img.current_scene

# Get a list valid scene ids
img.scenes

# Change scene using name
img.set_scene("Image:1")
# Or by scene index
img.set_scene(1)

# Use the same operations on a different scene

from aicsimageio import AICSImage

AICSImage("my_file.czi").save("my_file.ome.tiff")
# ...







#CZIFILE - NOT WHAT I WANT TO USE
#pip install czifile
import czifile
czi = czifile.CziFile("Images/weakPSD95mito.czi")
print(czi.shape) # WHERE IS THE EXTRA PRINT NUMBER COMING FROM?
#time series, scenes, channels, x,y,z, RGB - doesn't seem correct
print(czi.axes)

"""
Information CZI Dimension Characters:
- '0': 'Sample',  # e.g. RGBA
- 'X': 'Width',
- 'Y': 'Height',
- 'C': 'Channel',
- 'Z': 'Slice',  # depth
- 'T': 'Time',
- 'R': 'Rotation',
- 'S': 'Scene',  # contiguous regions of interest in a mosaic image
- 'I': 'Illumination',  # direction
- 'B': 'Block',  # acquisition
- 'M': 'Mosaic',  # index of tile for compositing a scene
- 'H': 'Phase',  # e.g. Airy detector fibers
- 'V': 'View',  # e.g. for SPIM
"""
#%%
"""
Example filter and plot
"""
from skimage import filters, img_as_ubyte # this syntax allows loading only a specific library from a package. Commas at end allow multiple loads
img_jpg_gaussian = filters.gaussian(img_jpg, sigma = 1) # gaussian filter will of course save as 
img_jpg_gaussian_8bit = img_as_ubyte(img_jpg_gaussian)

""" 
Display Images
"""
#SKIMAGE
io.imshow(img_jpg_gaussian_8bit)
#MATPLOTLIB
#pip install matplotlib
import matplotlib.pyplot as plt
plt.imshow(img_tiff)
img_gray = io.imread("Images/Snap-85.ome.tiff", as_gray= True)

fig = plt.figure(figsize = (10, 10))

ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img_gray, cmap = "hot")
ax1.title.set_text('1st')

ax2 = fig.add_subplot(2,2,2)
ax2.imshow(img_gray, cmap = "jet")
ax2.title.set_text('2nd')

plt.show()

#OPENCV sucks, don't use it it's annoying
#cv2.imshow("title", var)
#cv2.waitKey(0) # displays image until closed
#cv2.destroyAllWindows() # supposed to be necessary but is broken

#%%
"""
Saving Images
"""
#SCIKITIMAGE
io.imsave("Exported/gaus-P123-8bit.tif", img_jpg_gaussian_8bit)
#MATPLOTLIB
plt.imsave("Exported/gaus-P123-plt.jpg", img_jpg_gaussian_8bit)
#OPEN-CV
cv2.imwrite("Exported/P123-cv-fromski.tif", img_jpg_gaussian_8bit)
cv2.imwrite("Exported/P123-cv2.tif", img_cv)
#AICSIMAGE FOR CZI this does not seem to be working well, which is a problem that needs solved 
# from aicsimageio.writers import OmeTiffWriter
# with OmeTiffWriter("mito.ome.tiff") as writer:
#     writer.save(
#         img_AICS,
#         pixels_physical_size=img_AICS.get_physical_pixel_size(),
# dimension_order="ZYX",        
#         )
# OmeTiffWriter.save(img_AICS, "Exported/mito.ome.tiff")
# AICSIMAGE for TIFF
img_AICS_tiff.save("Exported/Snap-85.ome.tiff") # this is definitely better than tiff format
#TIFFFILE
tifffile.imwrite("Exported/Snap-85.tiff", img_tifffile)

#%%
"""
Napari - Displaying images
"""
#pip install napari
#pip install napari-aicsimageio

#This appears to be default napari, which may be ok, documentation on all this is terrible
# import napari
# viewer = napari.Viewer()
# viewer.open("weakPSD95mito.czi")

#%%
"""
Batch Reading from Tutorial 27/28
Alternative to os.listdir is to use glob.glob (import glob), however, I like this format and the below for loop gives flexibilty in visualization as well as overall data structure. both can be used for equivalent purposes
"""
# GLOB
import glob 
#save each file into a list
glob_list = []

from skimage import io
path = "BatchInput/*.ti*"
print(glob.glob(path))
for file in glob.glob(path):
    print(file)
    a = io.imread(file)
    glob_list.append(a)


# OS.LISTDIR
# CANNOT BE USED FOR PATTERN MATCHING (such as *.ti*) AS GLOB CAN
import os
os_list = [] # create list to store images from folder

path = "./BatchInput/" # Not possible for pattern match
print(os.listdir(path))
for image in os.listdir(path):
    print(image)    


#OS.LISTDIR for displaying file tree (fun) and then do 
import os

path = "BatchInput/"
print(os.walk(".")) # nothing to see here as this is just a generator
for root, dirs, files in os.walk("."):
    #print(root) # prints root directory names (i.e. folders)
    path = root.split(os.sep) # split at separator (/ or \)
    #print(path) # gives names of directories for easy location of files 
    #print(files) # prints all file names in all directories, useful after 
    print((len(path) -1) * '---', os.path.basename(root))
    for file in files:
        print(len(path) * '---', file)

    # not my favorite visualization, as there is no sense of the images in the parent folder. does not visualize it in any smaller sense
    # for name in dirs: 
    #     print (os.path.join(root,name))
        
    # for name in files: 
    #     print (os.path.join(root, name))
    
# STACKOVERFLOW COMBO https://stackoverflow.com/questions/8931099/quicker-to-os-walk-or-glob
# Because this can "os.walk" through subdirectories it can be used on a master folder input to detect multiple subfolders. This is because os.walk is recursive therefore fnmatch.filter will glob pull files 
import os, glob, fnmatch 
# import shutil
path = "Tim Monko - Bastian Lab/"
search_str = "*.ti*"

for path, subdirs, files in os.walk(path):
        for name in fnmatch.filter(files, search_str):
            print(name)
            # shutil.copy(os.path.join(path,name), dest) # why is this here??? it is to copy/move files
# as such can also use an os.walk through a current directory with

# for path, subdirs, files in os.walk("."):
#         for name in fnmatch.filter(files, search_str):
#             print(name)
#%%
"""
Basic Processing with Sci-Kit Image (Tutorial 29)
"""

from matplotlib import pyplot as plt
from skimage import io, color
from skimage.transform import rescale, resize, downscale_local_mean

img = io.imread("Images/Snap-85.ome.tiff", as_gray = True) # converts to floating point
io.imshow(img)

img_rescaled = rescale(img, 1.0/4.0, anti_aliasing = False) # makes the image entirely smaller - but this is the ONLY one that should be used
img_resized = resize(img, (200,200), anti_aliasing=True) # force image to these dimensions, causes compression
plt.imshow(img_rescaled)
plt.imshow(img_resized)

img_downscaled = downscale_local_mean(img, (4,3))
plt.imshow(img_downscaled) #change/compress shape 

from skimage.filters import gaussian

gaussian_img = gaussian(img, sigma = 1, mode = 'constant', cval = 0.0)
plt.imshow(gaussian_img)

#%% 
"""
Basic Processing with OpenCV (Tutorial 30)
"""
import cv2

plt.imshow(img)

img_cvresized = cv2.resize(img, None, fx=2, fy=2)
cv2.imshow(img_cvresized)
# extract the three images, X, Y, C, but requires a different image. can of course do same thing with SKIMAGE subsettign
blue = img[:,:, 0] # show only blue
green = img[:,:,1] # show only green
red = img[:,:,2] # red
# can also use
b,g,r = cv2.split(img) # use this if 3 channel
cv2.merge((b,g,r)) # merge together channels. This is the key can I do with skimage? 
#cv2.Canny # this will be really good!!!!!! edge detect mitochondria
cv2.waitKey(0)
cv2.destroyAllWindows()

#%% 
""" 
Tutorial 31, 32, 33- Unsharp Mask, Gaussian, and Median - 3/7/22
https://www.youtube.com/watch?v=_p_36DIJMIw&list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG&index=32
a form of convolution
padding images handles border pixels by either giving a raw pixel value, or matching to the outside value, for example it can be constant with cval = 0.0
sharpening images with an unsharp image
Unsharpened image = original + amount * (original - blurred)
Median filter is best for salt and pepper noise
"""

from skimage import io, img_as_float
from skimage.filters import unsharp_mask
from skimage.filters import gaussian,median

#manual unsharp mask and showing of gaussian
img = img_as_float(io.imread("Images/Snap-85.ome.tiff", as_gray = True))
gaussian_img = gaussian(img, sigma=3, mode = 'constant', cval = 0.0)
img2 = (img - gaussian_img)*2
img3 = img + img2

#median 
median_img = median(img)

#skimage unsharp
unsharped_img = unsharp_mask(img, radius = 3, amount = 2)

#image display
from matplotlib import pyplot as plt
plt.imshow(img, cmap = "gray")
plt.imshow(img2, cmap = "gray")
plt.imshow(img3, cmap = "gray")
plt.imshow(gaussian_img, cmap = "gray")
plt.imshow(median_img, cmap = "gray")
plt.imshow(unsharped_img, cmap = "gray")


