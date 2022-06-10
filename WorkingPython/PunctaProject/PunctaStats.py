# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 11:12:02 2022

@author: TimMonko

https://www.utc.fr/~jlaforet/Suppl/python-cheatsheets.pdf
https://stackoverflow.com/questions/44065573/anova-for-groups-within-a-dataframe-using-scipy
"""
#%% Import and Manipulate Data
import os
import pandas as pd
import matplotlib.pyplot as plt

file_directory = "C:\\Users\TimMonko\Downloads"
os.chdir(file_directory)

filepath = "C:\\Users\TimMonko\Downloads\label_summary.csv"
raw_data = pd.read_csv(filepath)


data = (raw_data
        .assign(treatment = lambda x: raw_data.filename.str.extract(r'(Group\d)'))
        .replace({'Group1': 'Iron Sufficient', 'Group2': 'Iron Repleted', 'Group3': 'Iron Deficient'})
        .dropna() # drop rows with any column having NA/null, can index
        ) 

print("number of files without full analysis:", len(raw_data)-len(data))

data_grp = data.groupby("treatment")
data_describe = data_grp.describe()
print(data_describe)

#print("number of files for each group: \n", data.groupby('treatment').size())


#%% Seaborn Settings
import seaborn as sns

# must explicitely invoke set_them else uses matplotlib defaults. Overrides all matplotlib based plots
sns.set_theme(
    style = "darkgrid",
    context = "notebook",
    palette = "colorblind",
    ) # explicility invoked else uses matplotlib defaults
# plt.style.use() # 

#%% Discrete plotting

ax = sns.boxplot(x = 'treatment',
                 y = 'blob_per_squm',
                 data = data,
                 width = 0.6,
                 palette = 'pastel'
                 )

ax = sns.swarmplot(x = 'treatment',
                   y = 'blob_per_squm',
                   data = data,
                   edgecolor = 'gray',
                   linewidth = None
                   )

ax.set_xlabel(None)
ax.set_ylabel("Puncta / Sq.um")

ax.figure # call figure without plt.show()
ax.figure.savefig('BoxPlot.svg') # save figure object from ax class


#%% Seaborn Relational Plotting

r = sns.lmplot(x = 'ridge_squm', 
               y = 'total_blobs', 
               hue = 'treatment', 
               data = data)

r.legend.set_title("Condition")
r.set_xlabels("Neurite, Sq.um")
r.set_ylabels("Puncta")

r.figure
r.figure.savefig('LmPlot.svg')


#%% Tiday Day statsmodels stats https://www.reneshbedre.com/blog/anova.html

import statsmodels.api as sm 
from statsmodels.formula.api import ols

model = ols('blob_per_squm ~ C(treatment)', data = data).fit()
anova_table = sm.stats.anova_lm(model, typ = 2)
anova_table
#%% Tidy Data bioinfokit stats https://www.reneshbedre.com/blog/anova.html

from bioinfokit.analys import stat

# ANOVA One-way
res = stat()
res.anova_stat(df = data, res_var='blob_per_squm', anova_model='blob_per_squm ~ C(treatment)')
res.anova_summary

# Tukey HSD post-hoc
res.tukey_hsd(df=data, res_var='blob_per_squm', xfac_var='treatment', anova_model='blob_per_squm ~ C(treatment)')
res.tukey_summary

#%% Testing ANOVA assumptions
# QQ-plot for testing anova assumptions
# res.anova_std_residuals are standardized residuals obtained from ANOVA (check above)
sm.qqplot(res.anova_std_residuals, line='45')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")
plt.show()

import scipy.stats as stats

# Test normality 
w, pvalue = stats.shapiro(model.resid)
print(w, pvalue)

# Levene's Non-normal Homogeneity of Variance Test
from bioinfokit.analys import stat 
res = stat()
res.levene(df=data, res_var='blob_per_squm', xfac_var='treatment')
res.levene_summary

