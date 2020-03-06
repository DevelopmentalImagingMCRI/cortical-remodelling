# -*- coding: utf-8 -*-
"""
PLS code modified from @harrymatthews50
https://github.com/harrymatthews50/Modelling_Craniofacial_Growth_Trajectories/blob/master/Modules/ShapeStats/multivariate_statistics.py

"""
import pandas as pa
import numpy as np
import copy

class Linear_regression(object):
    """Abstract base class for linear regressions """
    def __init__(self):
        super(Linear_regression,self).__init__()

    def predict(self, X, original_space=False):
        """Predict block of Y-variables, given X
        INPUTS:
            X - an n (observations) x p(variables) matrix of predictor variables
            original_space - if True it will assume that X has not been centered and will center it
                            and will add the mean of Y back onto the predicted Y-block, so that input from the original space of X variables results in a prediction in the o$
        OUTPUTS:
            Y - an n (observations) x o(variables matrix of outcome variables


        """
        if isinstance(X,float):
            X = np.array([[X]])

        if X.ndim == 1:
            X = np.atleast_2d(X).T

        if original_space==True:
            X = X-self.Xmean



        if any([item>1 for item in self.coefs.shape]): # if any axis has only one element

            Y = np.dot(X, self.coefs)
        else:
            Y = X*self.coefs.flatten()


        if original_space==True:
            Y = Y+self.Ymean

        return Y




class PLSR(Linear_regression):
    """
    Implements a partial least-squares regression by SIMPLS, with observation weights added
        REFERENCES:  De Jong, S. (1993). SIMPLS: an alternative approach to partial least squares regression. Chemometrics and intelligent laboratory systems, 18(3), 251-263.
    """
    
    def __init__(self):

        super(PLSR,self).__init__()
        self.X0 = None
        self.Y0 = None
    
    def fit(self,X, Y,ncomps='full_rank'): 
        """
        Fits the regression using the SIMPLS algorithm - columns are mean-centred internally but not scaled to have unit variance
        
        INPUTS:
            X - a numpy array where columns correspond to predictor variables and rows to observations
            Y - a numpy array  where columns correspond to outocome variables and rows to observations
            ncomps - the number of latent components to include in the regression if ncomps = 'full_rank' (default) this will include all components, whihc is equal to equal to np.min([len(X.index), len(X.columns)])

        NOTES:
            X and Y must have corresponding indices in their Index, and contain only numeric values the regression will treat all variables as though they were continuous 
            
        """
        
        if ncomps=='full_rank': # fit with all components
            ncomps = np.min([np.shape(X)[0]-1, np.shape(X)[1]])
        elif isinstance(ncomps, int)==False:
            raise TypeError('PLSR:  ncomps must either be an integer or \'full_rank\' but value ' + str(ncomps) + 'of type '+ str(type(ncomps))+'was specified')
    
        
        if ncomps>np.min([np.shape(X)[0]-1, np.shape(X)[1]]):
            raise ValueError('Number of components in a PLS-regression should be less than or equal to the smaller of (number of observations in x-1, number of variables in x) i.e. should be less than '+ str(np.min([len(X.index)-1, len(X.columns)]))+' but value '+str(ncomps)+'was specified') 
        
        
        # mean centre
        Xmean = np.zeros(X.shape[1])
        Ymean = np.zeros(Y.shape[1])

        X0=X-Xmean
        Y0=Y-Ymean
        

        # The following implementation of simpls is copied from MATLAB PLSREGRESS 
        n,dx=X0.shape
        dy = Y0.shape[1]

        # initialise arrays
        Xloadings = np.zeros([dx,ncomps])
        Yloadings = np.zeros([dy,ncomps])

        Xscores = np.zeros([n,ncomps])
        Yscores = np.zeros([n,ncomps])

        Weights = np.zeros([dx,ncomps])
        
        V = np.zeros([dx,ncomps])

        Cov = np.dot(np.transpose(X0),Y0)  # cross(dot)-product matrix - information on variance in X, Y and covariance(X,Y)

        for i in range(ncomps):
            u,s,v = np.linalg.svd(Cov,full_matrices = 0) # u and v -> left and right singular values

            ri = u[:,0]  # such that t = X0*ri
            ci = v[0,:]  # such that u = Y0*ci, aiming for max(u'*t) or ri.T*X0.T*Y0*ci or ci*(Y0*X0')*ri etc
            si = s[0]

            ti = np.dot(X0,ri) # projection onto ri
            normti = np.linalg.norm(ti)  # normalise t 
            ti = ti/normti
            Xloadings[:,i] = np.dot(np.transpose(X0),ti)  # regress x on normalised t

            qi = si*ci/normti	  # si*ci equiv. to Y0*ti i.e.: regress y on normalised t
            Yloadings[:,i] = qi

            Xscores[:,i] = ti            # **normalised** Xscores
            Yscores[:,i]= np.dot(Y0,qi)  # Yscores = Y0 * Yloadings

            Weights[:,i] = ri/normti    # scale weights accordingly

            vi = Xloadings[:,i]  # stabilise Xloadings - see Matlab code

            for repeat in range(2):
                for j in range(i):
                    vj = V[:,j]
                    vi = vi - np.dot(vj,vi)*vj


            vi = vi/np.linalg.norm(vi)
            V[:,i] = vi



            Cov = Cov-np.outer(vi,np.dot(vi,Cov))  # deflate cross-product matrix
            Vi = V[:,0:(i+1)]
            if i==0: 
                Cov = Cov-np.outer(Vi,np.dot(np.transpose(Vi),Cov))# Vi will be a single column, numpy will only do this operation using np.outer
            else:
                Cov = Cov-np.dot(Vi,np.dot(np.transpose(Vi),Cov))



        # Orthogonalise Y-scores to previous Xscore - gives only unique contribution of PLS comp to Y
        for i in range(ncomps):    
            ui = Yscores[:,i]
            for repeat in range(2):
                for j in range(i):
                    tj = Xscores[:,j]
                    ui = ui - np.dot(tj,ui)*tj

            Yscores[:,i] = ui

        self.Xloadings = Xloadings
        self.Yloadings = Yloadings
        self.Xscores = Xscores
        
        self.Yscores = Yscores
        self.Weights = Weights
        self.Yresiduals = Y0 - np.dot(self.Xscores,np.transpose(self.Yloadings))
        self.Xresiduals = X0 - np.dot(self.Xscores,np.transpose(self.Xloadings))
        self.predicted_values = np.dot(self.Xscores,np.transpose(self.Yloadings))

        self.coefs = np.atleast_2d(np.dot(self.Weights,np.transpose(self.Yloadings)))
  
       
