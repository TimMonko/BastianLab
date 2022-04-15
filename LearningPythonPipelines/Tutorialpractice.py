"""
APEER TUTORIAL PRACTICE
Began 2/24/22 with tutorial 25

Created on Thu Feb 24 12:35:26 2022

@author: Tim M
"""

from skimage import io
img_ski = io.imread("Images/Snap-85.ome.tiff")
img_tiff = io.imread("Images/Snap-85.ome.tiff", as_gray = True)

io.imshow(img_ski)
io.imshow(img_tiff)

import matplotlib.pyplot as plt
#pyplot is nice because it can be rearranged
plt.imshow(img_tiff)

fig = plt.figure(figsize = (10, 10))

ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img_tiff, cmap = "hot")
ax1.title.set_text('1st')

ax2 = fig.add_subplot(2,2,2)
ax2.imshow(img_tiff, cmap = "jet")
ax2.title.set_text('2nd')

plt.show()

import cv2

gray_img = cv2.imread("Images/Snap-85.ome.tiff", 0)
color_img = cv2.imread("Images/Snap-85.ome.tiff", 1)

#this does not work
cv2.imshow("skimage import", img_ski)
cv2.imshow("color opencv", color_img)
cv2.imshow("gray opencv", gray_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
"""
Tutorial 26
plotting in python using matplotlib.pyplot
Wed 3/2/22
Lots to learn here, reminds me of R ggplot, in terms of adding, subplotting, and labelling other aesthetic changes. Though, this does not seem to use the confusing aes format, it may in fact make itless flexible because there will be a lot of redundant referencing
"""

from matplotlib import pyplot as plt

x = [1,2,3,4,5]
y = [1,4,9,16,25]

plt.plot(x,y)

import numpy as np
a = np.array(x)
b = np.array(y)
plt.plot(a,b)

plt.imshow(img_tiff, cmap= "gray")
plt.hist(img_tiff.flat, bins = 100, range = (0,250))

plt.plot(a,b)
plt.plot(a,b, 'bo')

plt.bar
# all these are saved on the same axis 

# lets make a plot with multiproperties

wells = [1,2,3,4,5]
cells = [80,62,88,110,90]

plt.figure(figsize = (8,8))
plt.bar(wells, cells)
plt.xlabel
plt.ylabel

plt.savefig("fig.jpg")
plt.show()


fig = plt.plot(a,b)
plt.setp(fig, color = 'r', linewidth = 4.0)
plt.show

#%%
"""
tutorial 27 + 28
glob for multiple files **** this is the main concern
then for 28 is os.listdir (does mostly the same function, apparently)
I think I like os.listdir better because the flexibility seems familiar to stringr, though I am quite unsure the purpose of os.walk besides that I think it is a generator of the current working director. It supposedly "walks" through each instance, such as in a for loop it will look at the new structure for each attempt
Wed 3/2/22
"""
import cv2
import glob
file_list = glob.glob("BatchInput/*.*")
print(file_list)

# save each file into a list
my_list=[] #empty list to store images from folder


from skimage import io

path = "BatchInput/*.*"
for file in glob.glob(path):
    print(file)
    a = io.imread(file)
    my_list.append(a)

from matplotlib import pyplot as plt
plt.imshow(my_list[0])
####
####
import os 
path = "BatchInput/*.*"
print(os.walk(".")) # nothing to see here as this is just a generator
for root, dirs, files in os.walk("."):
    #print(root) # prints root directory names (i.e. folders)
    path = root.split(os.sep) # split at separator (/ or \)
    #print(path) # gives names of directories for easy location of files 
    #print(files) # prints all file names in all directories, useful after 
    a = io.imread(files)
    my_list.append(a)
    
    print((len(path) -1) * '---', os.path.basename(root))
    for file in files:
        print(len(path) * '---', file)
     
    # not my favorite visualization, as there is no sense of the images in the parent folder. does not visualize it in any smaller sense
    # for name in dirs: 
    #     print (os.path.join(root,name))
        
    # for name in files: 
    #     print (os.path.join(root, name))

#%%
"""
Tutorial 29 - 3/2/22
Basic processing with SCIKIT-image WOOHOO
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

sobel_img = sobel(img)
plt.imshow(sobel_img)

#%%
"""
Tutorial 30 - 3/2/22
Basic processing with OPENCV - good for machine learning especially in realtime video 
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
Tutorial 31 - Unsharp Mask - 3/7/22
https://www.youtube.com/watch?v=_p_36DIJMIw&list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG&index=32
sharpening images with an unsharp image
Unsharpened image = original + amount * (original - blurred)
"""

from skimage import io, img_as_float
from skimage.filters import unsharp_mask
from skimage.filters import gaussian

#manual unsharp mask
img = img_as_float(io.imread("Images/Snap-85.ome.tiff", as_gray = True))
gaussian_img = gaussian(img, sigma=2, mode = 'constant', cval = 0.0)
img2 = (img - gaussian_img)*2
img3 = img + img2

#skimage unsharp
unsharped_img = unsharp_mask(img, radius = 3, amount = 2)
from matplotlib import pyplot as plt
plt.imshow(img, cmap = "gray")
plt.imshow(img2, cmap = "gray")
plt.imshow(img3, cmap = "gray")
plt.imshow(unsharped_img, cmap = "gray")




