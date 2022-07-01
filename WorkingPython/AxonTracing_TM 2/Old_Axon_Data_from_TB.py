# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 08:51:02 2022

@author: TimMonko
"""

#%% Import and Wrangle Data
import os
import pandas as pd

file_directory = r"C:/Users/Tim M/Documents/GitHub/BastianLab/WorkingPython/AxonTracing_TM 2"
os.chdir(file_directory)

filepath = r"C:/Users/Tim M/Documents/GitHub/BastianLab/WorkingPython/AxonTracing_TM 2\Combined_TM.csv"
raw_data = pd.read_csv(filepath)

data = raw_data

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
#statsannotations for the future https://levelup.gitconnected.com/statistics-on-seaborn-plots-with-statannotations-2bfce0394c00

# must explicitely invoke set_them else uses matplotlib defaults. Overrides all matplotlib based plots
sns.set_theme(
    style = "darkgrid",
    context = "notebook",
    palette = "colorblind",
    ) # explicility invoked else uses matplotlib defaults
# plt.style.use() # 

#%% Discrete plotting

def box_swarm_plot(x, y, data, ylabel):
    ax = sns.boxplot(x = x, y = y, data = data,
                     width = 0.6, palette = 'pastel')

    ax = sns.swarmplot(x = x , y = y, data = data,
                       edgecolor = 'gray', linewidth = None)

    ax.set_xlabel(None)
    ax.set_ylabel(ylabel)
    return ax

plots = []

for dvs in data.columns[data.columns != 'Tx']:
    plot = box_swarm_plot('Tx', dvs, data, dvs)
    plt.show()
    plots.append(plot)

# plots[0].figure.savefig('box.svg') # to call from list

#%% PairGrid relationship plot
#quick and dirty

pp = sns.pairplot(data, hue = "Tx", kind = 'reg') # default is scatter

#%% Seaborn Relational Plotting

r = sns.lmplot(
    x = 'Primary_Axon_Length', 
    y = 'Num_Branches', 
    hue = 'Tx', 
    data = data)


r = sns.lmplot(
    y = 'Num_Branches', 
    x = 'Total_Branch_Length', 
    hue = 'Tx', 
    data = data)

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
    x = 'Primary_Axon_Length',
    y = 'Num_Branches',
    hue = 'Tx')

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
model = ols('Total_Branch_Length ~ Num_Branches * C(Tx)', data = data).fit()
model = ols('Num_Branches ~ Average_Branch_Length * C(Tx)', data = data).fit()

#print(model.summary()) # very overwhelming output
# So, push the model through the ANOVA to simplify the output 
anova_table = sm.stats.anova_lm(model, typ = 2)
anova_table

# Outliers with statsmodels
#test = model.outlier_test()
#outliers_test = test[test['bonf(p)'] < 0.05]

from statsmodels.multivariate.manova import MANOVA

fit = MANOVA.from_formula('Num_Branches + Average_Branch_Length ~ C(Tx)', data = data)

print(fit.mv_test())


