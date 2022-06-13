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

file_directory = "C:\\Users\Tim M\Downloads"
os.chdir(file_directory)

filepath = "C:\\Users\Tim M\Downloads\label_summary.csv"
raw_data = pd.read_csv(filepath)


data = (raw_data
        .assign(treatment = lambda x: raw_data.filename.str.extract(r'(Group\d)'))
        .replace({'Group1': 'Iron Sufficient', 'Group2': 'Iron Repleted', 'Group3': 'Iron Deficient'})
        .dropna() # drop rows with any column having NA/null, can index
        ) 

print("number of files without full analysis:", len(raw_data)-len(data))

data_grp = data.groupby("treatment")
data_describe = data_grp.describe()
data_describe

#%% Removes outliers, but not with grouped data
# Remove outliers with IQR method https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba

# FULL data outliers identificationm, very different use than Grouped function below - detects overall anomolies
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

outliers = data[((data < (Q1 - 1.5* IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis =1)]

data_rm = data[~data.isin(outliers)].dropna()

# GROUPED data, 1 column only -- seems very similar to statsmodels outlier_test
def is_outlier(s):
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5* IQR
    upper_limit = Q3 + 1.5* IQR
    return ~s.between(lower_limit, upper_limit)

grouped_outliers = data[data_grp['blob_per_squm'].apply(is_outlier)]

#%% Pingouin Stats 
# https://pingouin-stats.org/index.html
# https://pingouin-stats.org/guidelines.html
# https://g0rella.github.io/gorella_mwn/intro_statistics.html
# https://towardsdatascience.com/the-new-kid-on-the-statistics-in-python-block-pingouin-6b353a1db57c

# #Pinguoin also adds methods to pd.DataFrame, so can use data.anova() for example
import pingouin as pg

# levene's but other settings available 
hs = pg.homoscedasticity(data = data, dv = 'blob_per_squm', group = 'treatment')
print(hs)

lr = pg.anova(dv = 'blob_per_squm', between = 'treatment', data = data)
print(lr)

# posthoc
lr_posthoc = pg.pairwise_tukey(dv = 'blob_per_squm', between = 'treatment', data = data)
print(lr_posthoc)


#%% Seaborn Settings for Plotting
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

#%% Tidy Data statsmodels stats 
# R like syntax, much more robust, but more complex 
# https://python.cogsci.nl/numerical/statistics/
# https://www.reneshbedre.com/blog/anova.html
# somewhat helpful but mostly confusing https://g0rella.github.io/gorella_mwn/intro_statistics.html
# https://www.statsmodels.org/stable/gettingstarted.html

import statsmodels.api as sm 
from statsmodels.formula.api import ols

# OLS model -- a "complicated" look at ANOVA stats, C() forces categorical
model = ols('blob_per_squm ~ C(treatment)', data = data).fit()
print(model.summary()) # very overwhelming output

# Outliers with statsmodels
test = model.outlier_test()
outliers_test = test[test['bonf(p)'] < 0.05]

# So, push the model through the ANOVA to simplify the output 
anova_table = sm.stats.anova_lm(model, typ = 2)
anova_table