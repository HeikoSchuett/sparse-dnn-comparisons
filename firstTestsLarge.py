#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 14:48:30 2018
First attemts at sparse statistics
@author: heiko
"""

import numpy as np
import scipy
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

def sparse_testset(n=10000,p=5,q=10000,beta=1):
    """
    generates a sparse linear regression dataset with 
    n datapoints 
    p true predictors and 
    q possible predictors
    """
    assert p<=q, 'need at least as many possible as true predictors'
    X = np.random.randn(n,q)
    y = beta*np.matmul(X[:,0:p],np.ones((p,1)))+np.random.randn(n,1)
    y = np.ravel(y)
    return (y,X)

data = sparse_testset()

lasso = lm.Lasso(alpha=0.1)
lasso.fit(data[1],data[0])

lassoCV = lm.LassoCV(cv = 10)
lassoCV.fit(data[1],data[0])

#lP = lm.lasso_path(data[1],data[0])
alphas_lasso, coefs_lasso, _ = lm.lasso_path(data[1], data[0], 0.01, fit_intercept=True)

colors = ['b', 'r', 'g', 'c', 'k']
neg_log_alphas_lasso = -np.log10(alphas_lasso)
for coef_l, c in zip(coefs_lasso, colors):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)