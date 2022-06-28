# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 13:33:22 2022

@author: Tim M
"""

import pandas as pd

def paste_96well_to_dataframe(value_type = 'tx'):
    df = (pd.read_clipboard()
          .reset_index()
          .rename(columns = {'index' : 'row'})
          .melt(id_vars = 'row', var_name = 'col', value_name = value_type)
          )
    df['id'] = df['row'].astype(str) + df['col'].astype(str)
    return(df)

genotype = paste_96well_to_dataframe(value_type = 'genotype')

DFO = paste_96well_to_dataframe(value_type = 'tx')

groups = pd.concat([genotype, DFO], axis = 1)
groups = groups.loc[:, ~groups.columns.duplicated()]
