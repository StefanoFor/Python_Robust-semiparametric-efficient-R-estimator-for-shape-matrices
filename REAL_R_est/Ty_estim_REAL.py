# -*- coding: utf-8 -*-

import numpy as np

def compute_tyler_shape_estimator(X, Sigma_init, max_iter_fp = 40):
    
    K, N = X.shape
    
    Sigma_fixed_point = Sigma_init.copy()
    s_maha = np.empty((K, ))
    convergence_fp = False
    ite_fp = 1
    
    while not(convergence_fp) and ite_fp < max_iter_fp:
            
        inv_Sigma_fixed_point = np.linalg.inv(Sigma_fixed_point)
        s_maha = ( (X @ inv_Sigma_fixed_point) * X).sum(1)
        
        Sigma_fixed_point_new = X.T @ (N/s_maha * X.T).T / K 
        Sigma_fixed_point_new = Sigma_fixed_point_new/Sigma_fixed_point_new[0,0]

        convergence_fp = True
        convergence_fp = convergence_fp and (np.linalg.norm(Sigma_fixed_point_new-Sigma_fixed_point, ord='fro')/N) < 10**(-6)

        Sigma_fixed_point = Sigma_fixed_point_new.copy() 

        ite_fp += 1
        
    return Sigma_fixed_point


def compute_tyler_joint_estimator(X, mu_init, Sigma_init, max_iter_fp = 40 ):
    
    K, N = X.shape
    
    mu_fixed_point = mu_init.copy()
    Sigma_fixed_point = Sigma_init.copy()
    s_maha = np.empty((K, ))
    convergence_fp = False
    ite_fp = 1
    
    while not(convergence_fp) and ite_fp < max_iter_fp:
        
        inv_Sigma_fixed_point = np.linalg.inv(Sigma_fixed_point)
        X0 = X - mu_fixed_point
        
        s_maha = ( (X0 @ inv_Sigma_fixed_point) * X0).sum(1)
        
        r_inv = s_maha**(-1/2)
        mu_fixed_point_new = np.sum(X * r_inv[:,np.newaxis], 0)/np.sum(r_inv);
        
        Sigma_fixed_point_new = X0.T @ (N/s_maha * X0.T).T / K 
        Sigma_fixed_point_new = Sigma_fixed_point_new/Sigma_fixed_point_new[0,0]

        convergence_fp = True
        convergence_fp = convergence_fp and (math.sqrt(np.inner(mu_fixed_point - mu_fixed_point_new, mu_fixed_point - mu_fixed_point_new)/N) < 10**(-6))
        convergence_fp = convergence_fp and (np.linalg.norm(Sigma_fixed_point_new-Sigma_fixed_point, ord='fro')/N) < 10**(-6)

        mu_fixed_point = mu_fixed_point_new.copy()
        Sigma_fixed_point = Sigma_fixed_point_new.copy() 

        ite_fp += 1
        
    return mu_fixed_point, Sigma_fixed_point
