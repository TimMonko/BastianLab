# -*- coding: utf-8 -*-
"""
DAPI MAP2 statistics

Created on Wed Jun 15 16:45:47 2022

@author: Bastian Lab
"""

#%% Import and Wrangle Data

import os
import pandas as pd

file_directory = "D:\Bastian Lab\Daniel\DAPI_MAP2_testing"
os.chdir(file_directory)

filepath = "D:\Bastian Lab\Daniel\DAPI_MAP2_testing\summary.csv"
raw_data = pd.read_csv(filepath)

data = (raw_data
        )

data_describe = data.describe()

#%% Plots

import seaborn as sns

sns.set_theme(
    style = "darkgrid",
    context = "notebook",
    palette = "colorblind",
    ) # explicility invoked else uses matplotlib defaults
# plt.style.use() # 


ax = sns.scatterplot(x = "total_neurons", y = "MAP2_um2", data = data)

r = sns.lmplot(x = 'total_neurons', y = 'MAP2_um2', data = data)

r.set_xlabels("Number of Neurons")
r.set_ylabels("MAP2 area, um2")
r.figure
r.figure.savefig('LmPlot.svg')
