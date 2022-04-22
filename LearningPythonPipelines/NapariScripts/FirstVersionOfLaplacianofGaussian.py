from skimage.io import imread
import napari_simpleitk_image_processing as nsitk
import napari_segment_blobs_and_things_with_membranes as nsbatwm
import napari
if 'viewer' not in globals():
    viewer = napari.Viewer()

image0_I = imread("C:/Users/TimMonko/Desktop/dendrite-Cs_28_29_HCC_2_18_22-04-Stitching-11-Scene-04-TR9-Deconvolutio...4-Stitching-11-Scene-04-TR9-Deconvolution (defaults)-15.czi #1-4.tif")
viewer.add_image(image0_I, name="Image:0")

# binominal blur filter

image1_B = nsitk.binominal_blur_filter(image0_I, 1)
viewer.add_image(image1_B, name='Result of Binominal blur (n-SimpleITK)')

# white top hat

image2_W = nsitk.white_top_hat(image1_B, 4, 4, 0)
viewer.add_image(image2_W, name='Result of White top-hat (n-SimpleITK)')

# laplacian of gaussian filter

image3_L = nsitk.laplacian_of_gaussian_filter(image2_W, 1.0)
viewer.add_image(
    image3_L, name='Result of Laplacian of Gaussian (n-SimpleITK)')

# invert intensity

image4_ii = nsitk.invert_intensity(image3_L)
viewer.add_image(image4_ii, name='Result of invert_intensity')

# voronoi otsu labeling

image5_V = nsbatwm.voronoi_otsu_labeling(image4_ii, 1.0, 1.0)
viewer.add_labels(image5_V, name='Result of Voronoi-Otsu-labeling (nsbatwm)')
