# -*- coding: utf-8 -*-

import numpy as np

def compute_tyler_shape_estimator(X, Sigma_init, max_iter_fp = 40):
    
    K, N = X.shape
    
    Sigma_fixed_point = Sigma_init.copy()
    sq_maha = np.empty((K, ))
    convergence_fp = False
    ite_fp = 1
    
    while not(convergence_fp) and ite_fp < max_iter_fp:
            
        inv_Sigma_fixed_point = np.linalg.inv(Sigma_fixed_point)
        sq_maha = ( (X @ inv_Sigma_fixed_point) * X).sum(1)
        
        Sigma_fixed_point_new = X.T @ (N/sq_maha * X.T).T / K 
        Sigma_fixed_point_new = Sigma_fixed_point_new/Sigma_fixed_point_new[0,0]

        convergence_fp = True
        convergence_fp = convergence_fp and (np.linalg.norm(Sigma_fixed_point_new-Sigma_fixed_point, ord='fro')/N) < 10**(-6)

        Sigma_fixed_point = Sigma_fixed_point_new.copy() 

        ite_fp += 1
        
    return Sigma_fixed_point