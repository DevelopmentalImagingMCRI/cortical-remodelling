
# coding: utf-8

import numpy as np
import pandas as pd

import os

from sklearn.metrics import r2_score, mean_squared_error

from pls import PLSR
from functions import z_score, run_plsr

# load gene expression data
gene_data = np.loadtxt('data/gene_expression_data.csv', delimiter=',')[:,1:]
gene_names = np.loadtxt('data/gene_list.csv', dtype=str)

print('number of regions: %i' % np.shape(gene_data)[0])
print('number of genes: %i \n' % np.shape(gene_data)[1])

# load imaging phenotype and Zscore
relative_change = np.loadtxt('data/relative_change_cust250.csv')

# remove any parcels without expression data
n_region = len(relative_change)
nan_indices = np.where(~np.isnan(gene_data[:,0]))[0]

print('%i regions removed \n' % (n_region-len(nan_indices)))

# PLS
n_comps = 3
X = gene_data[nan_indices,:]
Y = relative_change[nan_indices]

print('running PLS with %i components' % n_comps)
z_X = z_score(X)
z_Y = z_score(Y)
# requires 2D array
z_Y = z_Y.reshape(-1,1)

gene_loadings, component_scores, component_loadings, explained_variance, m_s_e = run_plsr(z_X, z_Y,  n_comps=n_comps)

## SAVE #################
os.makedirs('Results', exist_ok=True)

# 1. model fit
outdf = pd.DataFrame([m_s_e, explained_variance]).T
outdf.columns = ['mse', 'explained_variance']
outdf.to_csv('Results/model_fit.csv', index=False)

# 2. component loadings (square these to get % explained variance)
outdf=pd.DataFrame([np.arange(n_comps)+1, component_loadings[0]]).T
outdf.columns = ['component', 'loading']
outdf.to_csv('Results/component_loadings.csv', index=False)

# 3. component_scores (PLS component maps)
# account for missing regions
out_data = []
for i in np.arange(n_comps):
    out_vec = np.zeros((n_region))
    out_vec[nan_indices] = component_scores[:,i]
    out_data.append(out_vec)
out_data = pd.DataFrame(np.vstack(out_data).T)
out_data.columns = ['PLS1','PLS2','PLS3']
out_data.to_csv('Results/PLS_component_scores.csv', index=False)

# 4. gene loadings
genes_out = pd.concat((pd.DataFrame(gene_names), pd.DataFrame(gene_loadings)), axis=1)
genes_out.to_csv('Results/gene_component_loadings.csv', index=False, header=False)

print("see: \n Results/model_fit.csv \n Results/component_loadings.csv \n Results/PLS_component_scores.csv \n Results/gene_component_loadings.csv")

