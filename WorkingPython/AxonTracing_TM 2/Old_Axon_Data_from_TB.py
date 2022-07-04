# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 08:51:02 2022

@author: TimMonko
"""

#%% Import and Wrangle Data
import os
import pandas as pd

file_directory = r"C:\Users\TimMonko\Documents\GitHub\BastianLab\WorkingPython\AxonTracing_TM 2"
os.chdir(file_directory)

filepath = r"C:\Users\TimMonko\Documents\GitHub\BastianLab\WorkingPython\AxonTracing_TM 2\Combined_TM.csv"
raw_data = pd.read_csv(filepath)

data = raw_data
data = data[data.Primary_Axon_Length < 1200]
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
    annotator = Annotator(ax, pairs = [("Control", "DFO")], data = data, x = x, y = y)
    annotator.configure(test = 'Mann-Whitney', text_format = 'star', loc = 'inside')
    annotator.apply_and_annotate()
    return ax

plots = []

#box_swarm_plot('Tx', 'Num_Branches', data, 'Num_Branches')

for dvs in data.columns[data.columns != 'Tx']:
    plot = box_swarm_plot('Tx', dvs, data, dvs)
    plt.show()
    plots.append(plot)

# plots[0].figure.savefig('box.svg') # to call from list

#%% PairGrid relationship plot
#quick and dirty

pp = sns.pairplot(data, hue = "Tx", kind = 'reg') # default is scatter

#%% Seaborn Relational Plotting

def lm_plot(x, y, data = data, hue = 'Tx'):
    r = sns.lmplot(x = x, y = y, data = data, hue = hue)
    r.legend.set_title("Treatment")
    return r

r = lm_plot("Primary_Axon_Length", "Num_Branches")
r.axes[0,0].set_xlabel("Primary Axon Length, um")
r.axes[0,0].set_ylabel("Number of Branches")
r = lm_plot(x = "Primary_Axon_Length", y= "Average_Branch_Length")
r = lm_plot("Num_Branches", "Total_Branch_Length")
r = lm_plot(x = "Num_Branches", y = "Average_Branch_Length")

plt.show()
r.figure.savefig('LmPlot.svg')
#%% Kde plot

j = sns.jointplot(
    data = data,
    x = 'Primary_Axon_Length',
    y = 'Num_Branches',
    hue = 'Tx',
    kind = "kde")

#%% Tidy Data statsmodels stats 
# R like syntax, much more robust, but more complex 
# https://www.statsmodels.org/stable/gettingstarted.html

import statsmodels.api as sm 
from statsmodels.formula.api import ols

# OLS model -- a "complicated" look at ANOVA stats, C() forces categorical


model = ols('Num_Branches ~ Primary_Axon_Length * C(Tx)', data = data).fit()
model = ols('Primary_Axon_Length ~ Num_Branches * C(Tx)', data = data).fit()
model = ols('Num_Branches ~ Average_Branch_Length * C(Tx)', data = data).fit()

#print(model.summary()) # very overwhelming output
# So, push the model through the ANOVA to simplify the output 
anova_table = sm.stats.anova_lm(model, typ = 2) # Type 2 Sum iof Squares if no intereaction b/w independent variables, Type 3 if there is interaction. Never use Type 1. Type 3 appears to be same as regression
anova_table

# Outliers with statsmodels
#test = model.outlier_test()
#outliers_test = test[test['bonf(p)'] < 0.05]
#%% Multivariate statsmodels

from statsmodels.multivariate.manova import MANOVA

fit = MANOVA.from_formula('Num_Branches * Primary_Axon_Length ~ C(Tx)', data = data)
print(fit.mv_test())

pg.box_m(data, dvs = ['Num_Branches', 'Primary_Axon_Length'], group = 'Tx')
