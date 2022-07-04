# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 08:51:02 2022

@author: TimMonko
"""

#%% Import and Wrangle Data
import os
import pandas as pd
import glob
import numpy as np

file_directory = r"C:\Users\TimMonko\Documents\GitHub\BastianLab\WorkingPython\AxonMitoMotility_TM 2\A"
os.chdir(file_directory)

file_type = "*.csv"
filenames = glob.glob(file_type)

data_list = []
for file in filenames:
    val_name = file[:len(file) - 4]
    raw_data = pd.read_csv(file)
    melt_data = (raw_data
                 .melt(value_name = val_name, var_name = 'Tx')
                 .set_index('Tx'))
    data_list.append(melt_data)

data = pd.concat(data_list, axis = 1, ignore_index = False).dropna(axis = 0).reset_index()
data = data.assign(Log = lambda data: np.log(data['Average Speed (per Mito)']))
data = data.assign(Log2 = lambda data: np.log(data['Percent Time Moving']))

data = data.assign(Per = lambda data: data['Percent Time Moving'] / 100)
data_describe = data.groupby("Tx").describe()

#%% Pinguoin Stats
# https://pingouin-stats.org/index.html
# https://pingouin-stats.org/guidelines.html
import pingouin as pg

def dv_tests(df, dv, iv):
    norm = pg.normality(data = df, dv = dv, group = iv) # Normal distriubtion of population assumed true in ANOVA/T
    
    hs = pg.homoscedasticity(data = df, dv = dv, group = iv) # Homogeneity of Variances (ANOVA/T assumption is true)

    lr = pg.anova(data = df, dv = dv, between = iv)
    print(dv + ' x ' + iv, norm, hs, lr, sep = '\n')
   
for dvs in data.columns[data.columns != 'Tx']:
    dv_tests(data, dvs, 'Tx')

# mitoc density, retro velocity, perc time retro motion, perc pausing, perc time moving, perc time anterograde motion  

#%% Seaborn Settings for Plotting
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator # https://github.com/trevismd/statannotations
#statsannotations for the future https://levelup.gitconnected.com/statistics-on-seaborn-plots-with-statannotations-2bfce0394c00

# must explicitely invoke set_them else uses matplotlib defaults. Overrides all matplotlib based plots
sns.set_theme(style = "darkgrid", context = "notebook", palette = "colorblind") 

#%% Discrete plotting

def box_swarm_plot(x, y, data, ylabel):
    ax = sns.boxplot(x = x, y = y, data = data,
                     width = 0.6, palette = 'pastel')

    ax = sns.swarmplot(x = x , y = y, data = data,
                       edgecolor = 'gray', linewidth = None)

    ax.set_xlabel(None)
    ax.set_ylabel(ylabel)
    annotator = Annotator(ax, pairs = [("IS", "ID")], data = data, x = x, y = y)
    annotator.configure(test = 'Mann-Whitney', text_format = 'star', loc = 'inside')
    annotator.apply_and_annotate()
    return ax

plots = []

for dvs in data.columns[data.columns != 'Tx']:
    plot = box_swarm_plot('Tx', dvs, data, dvs)
    plt.show()
    plots.append(plot)

# plots[0].figure.savefig('box.svg') # to call from list

#%% PairGrid relationship plot
#quick and dirty, but caution on amount of variables

pp = sns.pairplot(data, hue = "Tx", kind = 'reg') # default is scatter

#%% Residual plotting to test for linearity of data
a = sns.residplot(x = 'Log', y = 'Percent Time Moving', data = data)

#%%

r = sns.lmplot(
    x = 'Average Speed (per Mito)',
    y = 'Per',
    data = data,
    hue = 'Tx',
    logistic = True)
#%% Seaborn Relational Plotting


r = sns.lmplot(
    x = 'Pause Frequency (per Mito)', 
    y = 'Log', 
    hue = 'Tx', 
    data = data)

r = sns.lmplot(
    x = 'Average Speed (per Mito)', 
    y = 'Percent Time Moving', 
    hue = 'Tx', 
    data = data,
    logx = True)

r = sns.lmplot(
    x = 'Log', 
    y = 'Percent Time Moving', 
    hue = 'Tx', 
    data = data)

r = sns.lmplot(
    x = 'Log', 
    y = 'Log2', 
    hue = 'Tx', 
    data = data,
    robust = True)

r = sns.lmplot(
    x = 'Average_Branch_Length', 
    y = 'Num_Branches', 
    hue = 'Tx', 
    data = data)

r.legend.set_title("Treatment")
#r.set_xlabels("Avg. Branch Length")
#r.set_ylabels("Num. Branches")

plt.show()
r.figure.savefig('LmPlot.svg')
#%% Kde plot

j = sns.jointplot(
    data = data,
    x = 'Average Speed (per Mito)',
    y = 'Percent Time Moving',
    hue = 'Tx')

j = sns.jointplot(
    data = data,
    x = 'Average Speed (per Mito)',
    y = 'Percent Time Moving',
    hue = 'Tx',
    kind = "kde")

#%% Tidy Data statsmodels stats 
# R like syntax, much more robust, but more complex 
# https://www.statsmodels.org/stable/gettingstarted.html
# Consider the importance of bounded data (i.e. percents) when modeling https://stats.stackexchange.com/questions/103731/what-are-the-issues-with-using-percentage-outcome-in-linear-regression



import statsmodels.api as sm 
from statsmodels.formula.api import ols

# OLS model -- a "complicated" look at ANOVA stats, C() forces categorical
model = ols('Q("Average Speed (per Mito)") ~ Q("Percent Time Moving") *  C(Tx)', data = data).fit()
model = ols('Total_Branch_Length ~ Num_Branches * C(Tx)', data = data).fit()
model = ols('Num_Branches ~ Average_Branch_Length * C(Tx)', data = data).fit()

#print(model.summary()) # very overwhelming output
# So, push the model through the ANOVA to simplify the output 
anova_table = sm.stats.anova_lm(model, typ = 2)
anova_table

# Outliers with statsmodels
#test = model.outlier_test()
#outliers_test = test[test['bonf(p)'] < 0.05]




