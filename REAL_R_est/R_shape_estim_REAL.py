# -*- coding: utf-8 -*-
import math
import numpy as np
import scipy as sp
from scipy.stats.distributions import chi2
import routines

def R_estimator_VdW_score(y, mu, S0, pert):
    
    """
    # -----------------------------------------------------------
    # This function implement the R-estimator for shape matrices
    
    # Input:
    #   y: (N, K)-dim real data array where N is the dimension of each vector and K is the number of available data
    #   mu: N-dim array containing a preliminary estimate of the location
    #   S0: (N, N)-dim array containing a preliminary estimator of the scatter matrix
    #   pert: perturbation parameter
    
    # Output:
    # S_est: Estimated shape matrix with the normalization [S_est]_{1,1} = 1
    # -----------------------------------------------------------
    """
    
    N, K = y.shape 
    
    S0 = S0/S0[0,0]
    y = T(T(y) - mu)
    
    D_n = routines.Dup(N)
    M_n = T(D_n[:,1:])
    
    E_n = routines.L(N)
    N_n = E_n[1:,:]

    # Generation of the perturbation matrix
    V = pert * np.random.randn(N, N)
    V = (V + T(V))/2
    V[0,0]=0
    
    alpha_est, Delta_S, Psi_S = alpha_estimator_sub( y, S0, V, M_n, N_n)

    beta_est = 1/alpha_est
    v_appo = beta_est*(sp.linalg.inv(Psi_S) @ Delta_S)/math.sqrt(K)
    N_VDW = E_n @ routines.vec(S0) + np.append([0], v_appo)
    
    N_VDW_mat = np.reshape(D_n @ N_VDW,(N,N), order='F' )
    
    return N_VDW_mat

# -----------------------------------------------------------
# Functions that will be used in the estimator  

# Make some shortcuts for transpose,hermitian:
#    np.transpose(A) --> T(A)
#    hermitian(A) --> H(A)
T = np.transpose


def kernel_rank_sign(y,S0):
    N, K = y.shape

    IN_S = sp.linalg.inv(S0)
    SR_IN_S = sp.linalg.sqrtm(IN_S)
    A_appo = SR_IN_S @ y
    Rq = np.linalg.norm(A_appo , axis=0)
    u = A_appo /Rq
    temp = Rq.argsort()
    ranks = np.arange(len(Rq))[temp.argsort()] + 1
    
    kernel_vect = chi2.ppf(ranks/(K+1), df=N)/2

    return kernel_vect, u, SR_IN_S
    
def  Delta_Psi_eval(y, S, M_n):
    N, K = y.shape

    kernel_vect, u, sr_S = kernel_rank_sign(y,S)
    
    #sr_S = sp.linalg.inv(sp.linalg.sqrtm(S))
    inv_sr_S2 = np.kron(sr_S,sr_S)

    I_N = np.eye(N)
    J_n_per = np.eye(N**2) - np.outer(routines.vec(I_N), routines.vec(I_N))/N

    K_V = M_n @ (inv_sr_S2 @ J_n_per)
    
    Kernel_appo = np.zeros((N,N))
    for n1 in range(N):
        Kernel_appo[n1,n1:N] = ((u[n1,:] * kernel_vect) * u[n1:N,:]).sum(axis=1)
    Kernel_appo = np.triu(Kernel_appo, 1) + np.tril(Kernel_appo.transpose(), 0)
    Kernel_appo = np.ravel(Kernel_appo, order='C')

    #Kernel_appo = np.zeros((N**2,))
    #for k in range(K):
    #    uk = u[:,k]
    #    Mat_appo = np.outer(uk,uk)
    #    Kernel_appo = Kernel_appo  + kernel_vect[k] * vec(Mat_appo)

    Delta_S = (K_V @ Kernel_appo) / math.sqrt(K)
    Psi_S = K_V @ T(K_V)
    
    return Delta_S, Psi_S

def  Delta_only_eval(y, S, M_n):
    N, K = y.shape

    kernel_vect, u, sr_S = kernel_rank_sign(y,S)
    
    #sr_S = sp.linalg.inv(sp.linalg.sqrtm(S))
    inv_sr_S2 = np.kron(sr_S,sr_S)

    I_N = np.eye(N)
    J_n_per = np.eye(N**2) - np.outer(routines.vec(I_N), routines.vec(I_N))/N
    K_V = M_n @ (inv_sr_S2 @ J_n_per)
    
    Kernel_appo = np.zeros((N,N))
    for n1 in range(N):
        Kernel_appo[n1,n1:N] = ((u[n1,:] * kernel_vect) * u[n1:N,:]).sum(axis=1)
    Kernel_appo = np.triu(Kernel_appo, 1) + np.tril(Kernel_appo.transpose(), 0)
    Kernel_appo = np.ravel(Kernel_appo, order='C')

    #Kernel_appo = np.zeros((N**2,))
    #for k in range(K):
    #    uk = u[:,k]
    #    Mat_appo = np.outer(uk,uk)
    #    Kernel_appo = Kernel_appo  + kernel_vect[k] * vec(Mat_appo)

    Delta_S = ( K_V @ Kernel_appo ) / math.sqrt(K)

    return Delta_S

def alpha_estimator_sub( y, S0, V, M_n, N_n):
    
    N, K = y.shape 
    
    Delta_S, Psi_S = Delta_Psi_eval(y, S0, M_n)
    S_pert = S0 + V/math.sqrt(K)
    Delta_S_pert = Delta_only_eval(y, S_pert, M_n)
    V_1 = routines.vec(V)
    alpha_est = np.linalg.norm(Delta_S_pert-Delta_S)/np.linalg.norm(Psi_S @ (N_n @ V_1))
    
    return alpha_est, Delta_S, Psi_S


# -----------------------------------------------------------


