# -*- coding: utf-8 -*-

import numpy as np

def hermitian(A, **kwargs):
    return np.transpose(A, **kwargs).conj()
# Make some shortcuts for transpose,hermitian:
#    np.transpose(A) --> T(A)
#    hermitian(A) --> H(A)
T = np.transpose
C = np.conj
H = hermitian

def compute_tyler_shape_estimator(X, Sigma_init, max_iter_fp = 40):
    # Computes tyler's estimators for mu and Sigma with the data of the cluster (hard assigment)
    
    K, N = X.shape
    
    Sigma_fixed_point = Sigma_init.copy()
    sq_maha = np.empty((K, ))
    convergence_fp = False
    ite_fp = 1
    
    while not(convergence_fp) and ite_fp < max_iter_fp:
            
        inv_Sigma_fixed_point = np.linalg.inv(Sigma_fixed_point)
        sq_maha = ( (C(X) @ inv_Sigma_fixed_point) * X).sum(1)
        
        Sigma_fixed_point_new = X.T @ (N/sq_maha * H(X)).T / K 
        Sigma_fixed_point_new = Sigma_fixed_point_new/Sigma_fixed_point_new[0,0]

        convergence_fp = True
        convergence_fp = convergence_fp and (np.linalg.norm(Sigma_fixed_point_new-Sigma_fixed_point, ord='fro')/N) < 10**(-6)

        Sigma_fixed_point = Sigma_fixed_point_new.copy() 

        ite_fp += 1
        
    return Sigma_fixed_point