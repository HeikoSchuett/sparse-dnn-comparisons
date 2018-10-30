# My implementations of coordinate descent
# based on the definitions from Statistical learning with sparsity
# by Hastie, Tibshirani & Wainwright

import numpy as np
import numpy.linalg as linalg


def lasso_loss(beta,X,y,lamb):
    # assume centered X,y, i.e. mean(X)=mean(y) = 0
    return np.sum((y-np.matmul(X,beta))**2)/2/len(y)-lamb*np.sum(np.abs(beta))

def lasso_loss_derivative(beta,X,y,lamb):
    # assume centered X,y, i.e. mean(X)=mean(y) = 0
    g = np.matmul(X.transpose(),(y-np.matmul(X,beta)))
    g[(beta==0) & (np.abs(g)<lamb)] = 0
    g[(beta!=0) | (np.abs(g)>lamb)] = g[(beta!=0) | (np.abs(g)>lamb)]-lamb*np.sign(beta) 
    return (np.sum((y-np.matmul(X,beta))**2)/2/len(y)-lamb*np.sum(np.abs(beta)),g)


def soft_threshold(x,lamb):
    return np.sign(x)*np.maximum(np.abs(x)-lamb,0)

#def lasso_cd_step(r,betaj,xj,lamb,eps):
#    betaj_new = soft_threshold(betaj+np.sum(xj*r)/len(xj),lamb)
#    changed = False
#    if np.abs(betaj_new-betaj)>eps:
#        changed = True
#    else:
#        changed = False
#    if betaj_new!=betaj:
#        return (betaj_new,r-xj*(betaj_new-betaj),changed)
#    else:
#        return (betaj_new,r,changed)
def lasso_cd_step(r,betaj,xj,lamb,eps):
    betaj_new = soft_threshold(betaj+np.sum(xj*r)/len(xj),lamb)
    return (betaj_new,r-xj*(betaj_new-betaj),np.abs(betaj_new-betaj)>eps)



def lasso_cd(X,y,lamb,beta0=None,eps=10**-10):
    if X.shape[1]==len(y):
        X=X.transpose()
    if beta0 is None:
        beta0 = np.zeros(X.shape[1])
    changed = True
    beta = beta0
    r = y-np.matmul(X,beta0)
    while changed:
        changed = False
        for j in range(len(beta0)):
            if np.abs(betaj+np.sum(xj*r)/len(xj))>lamb:
                (beta[j],r,c) = lasso_cd_step(r,beta[j],X[:,j],lamb,eps)
            else:
                c = False
            if c:
                changed = True
    return beta

def lasso_path(X,y,lambs,beta0=None,eps=10**-10):
    beta_path = np.zeros((X.shape[1],len(lambs)))
    for iL in range(len(lambs)):
        beta0=lasso_cd(X,y,lambs[iL],beta0=beta0,eps=10**-10)
        beta_path[:,iL]=beta0
    return beta_path
        