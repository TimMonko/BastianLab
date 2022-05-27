# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:29:14 2022

@author: TimMonko
"""
#%% Images Paths
import glob
import os

file_type = ".tiff"
file_directory = "C:/Users/TimMonko/Documents/GitHub/BastianLab/WorkingPython/Redox"

glob_path = file_directory + "/*" + file_type
print(glob_path)

filenames = glob.glob(glob_path)

#%% Function Def

from aicsimageio import AICSImage
import pyclesperanto_prototype as cle
import pandas as pd

def label_threshold(img, median_rad):
    median_filter = cle.median_box(img, None, radius_x = median_rad, radius_y = median_rad, radius_z = median_rad)
    otsu = cle.threshold_otsu(median_filter)
    return(otsu)

def label_stats(channel, label, column_name):
    stats = (pd.DataFrame(cle.statistics_of_background_and_labelled_pixels(channel, label))
             .assign(channel = column_name)
             )
    return(stats)

stats_list = []

for file in filenames:
    img = AICSImage(file)
    Ex385 = img.get_image_data("TYX", C=1)
    Ex470 = img.get_image_data("TYX", C=2)
    labels_Ex470 = label_threshold(Ex470, median_rad = 1)
    
    #Below combines all T points in data, which is ok, but also need to check for consistency across T, perhaps another time
    stats_Ex470 = label_stats(Ex470, labels_Ex470, "Ex470")
    stats_Ex385 = label_stats(Ex385, labels_Ex470, "Ex385")    
    
    merge_stats = (pd.concat([stats_Ex470, stats_Ex385])
                   .assign(filename = os.path.basename(file))
                   )
    stats_list.append(merge_stats)

stats_all = pd.concat(stats_list)

stats_intensity_difference = (stats_all.pivot(values = 'mean_intensity', index = ['filename','channel'], columns = 'original_label')
               .rename(columns = {0 : 'background',
                                  1 : 'signal'})
               .assign(intensity_difference =  lambda stats_all : stats_all.signal - stats_all.background)
               )

stats_intensity_ratio = (stats_intensity_difference.reset_index()
               .pivot(values = 'intensity_difference', index = 'filename', columns = 'channel')
               .assign(ratio = lambda stats_intensity_difference : stats_intensity_difference.Ex470 / stats_intensity_difference.Ex385)
               )

stats_intensity_difference.plot.bar()
stats_intensity_ratio.plot.bar(y = 'ratio')


stats_intensity_difference.to_csv(os.path.join(file_directory, 'stats_intensity_difference.csv'))
stats_intensity_ratio.to_csv(os.path.join(file_directory, 'stats_intensity_ratio.csv'))
