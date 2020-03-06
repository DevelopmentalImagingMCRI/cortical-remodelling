# -*- coding: utf-8 -*-
"""
Helper functions for PLS analysis

"""
import numpy as np

from sklearn.metrics import r2_score, mean_squared_error
from pls import PLSR


def z_score(x):
    z_x = (x - np.mean(x,0)) / np.std(x,0)
    
    return z_x

def run_plsr(X, Y, n_comps):
    
    plsr = PLSR()
    plsr.fit(X, Y, ncomps=n_comps)

    #outputs
    # 1. LOADINGS  #################
    # output loadings for each gene (correlation between regional gene expression and PLS component weight)
    # loadings are generally given as dot(X, Xscores)
    # normalising X first means that dot(normX, Xscores) is same as correlation
    normXs = X / np.linalg.norm(X, axis=0)
    x_loadings = np.dot(normXs.T, plsr.Xscores)

    # correlation between each component score and Y
    normYs = Y / np.linalg.norm(Y)
    y_loadings = np.dot(normYs.T, plsr.Xscores)

    # 2. LATENT FACTORS #################
    # output component scores
    x_scores = plsr.Xscores

    # 3. EXPLAINED VARIANCE #################
    explained_variance = r2_score(Y, plsr.predicted_values)
    # mse
    m_s_e = mean_squared_error(Y, plsr.predicted_values)

    return x_loadings, x_scores, y_loadings, explained_variance, m_s_e
