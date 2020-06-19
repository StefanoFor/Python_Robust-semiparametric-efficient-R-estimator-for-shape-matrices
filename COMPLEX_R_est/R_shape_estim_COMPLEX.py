# -*- coding: utf-8 -*-
import math
import numpy as np
import scipy as sp
from scipy.stats.distributions import gamma
import routines

def R_estimator_VdW_score(y, mu, S0, pert):
    
    """
    # -----------------------------------------------------------
    # This function implement the R-estimator for shape matrices
    
    # Input:
    #   y: (N, K)-dim complex data array where N is the dimension of each vector and K is the number of available data
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

    # Generation of the perturbation matrix
    V = pert * (np.random.randn(N, N) + 1j*np.random.randn(N, N))
    V = (V + H(V))/2
    V[0,0]=0
    
    alpha_est, Delta_S, Psi_S = alpha_estimator_sub( y, S0, V)

    beta_est = 1/alpha_est
    v_appo = beta_est*(sp.linalg.inv(Psi_S) @ Delta_S)/math.sqrt(K)
    N_VDW = routines.vec(S0) + np.append([0], v_appo)
    
    N_VDW_mat = np.reshape( N_VDW, (N,N), order='F' )
    
    return N_VDW_mat

# -----------------------------------------------------------
# Functions that will be used in the estimator  

def hermitian(A, **kwargs):
    return np.transpose(A, **kwargs).conj()
# Make some shortcuts for transpose,hermitian:
#    np.transpose(A) --> T(A)
#    hermitian(A) --> H(A)
T = np.transpose
C = np.conj
H = hermitian


def kernel_rank_sign(y,S0):
    N, K = y.shape

    IN_S = sp.linalg.inv(S0)
    SR_IN_S = sp.linalg.sqrtm(IN_S)
    A_appo = SR_IN_S @ y
    Rq = np.real(np.linalg.norm(A_appo , axis=0))
    u = A_appo /Rq
    temp = Rq.argsort()
    ranks = np.arange(len(Rq))[temp.argsort()] + 1
    
    # Alternative method to calculate the ranks
    #ranks = np.empty_like(temp)
    #ranks[temp] = np.arange(len(Rq)) + 1
    
    kernel_vect = gamma.ppf(ranks/(K+1), N, scale = 1)

    return kernel_vect, u, SR_IN_S
    
def  Delta_Psi_eval(y, S):
    N, K = y.shape

    kernel_vect, u, sr_S = kernel_rank_sign(y,S)
    
    # sr_S = sp.linalg.inv(sp.linalg.sqrtm(S))
    inv_sr_S2 = np.kron(T(sr_S),sr_S)

    I_N = np.eye(N)
    J_n_per = np.eye(N**2) - np.outer(routines.vec(I_N), routines.vec(I_N))/N

    K_V = inv_sr_S2 @ J_n_per

    # Kernel_appo = np.zeros((N**2,), dtype=complex)
    # for k in range(K):
    #     uk = u[:,k]
    #     Mat_appo = np.outer(uk,C(uk))
    #     Kernel_appo = Kernel_appo  + kernel_vect[k] * routines.vec(Mat_appo)
            
    Kernel_appo = np.zeros((N,N), dtype=complex)
    for n1 in range(N):
        Kernel_appo[n1,n1:N] = ((u[n1,:].conj() * kernel_vect) * u[n1:N,:]).sum(axis=1)
    Kernel_appo = np.triu(Kernel_appo, 1) + np.tril(Kernel_appo.transpose().conj(), 0)
    Kernel_appo = np.ravel(Kernel_appo, order='C')

    Delta_S = (K_V @ Kernel_appo) / math.sqrt(K)
    Delta_S = np.delete(Delta_S, [0])
    
    Kc = K_V @ H(K_V)
    Psi_S = np.delete(np.delete(Kc, 0, 0), 0 , 1)
    
    return Delta_S, Psi_S

def  Delta_only_eval(y, S):
    N, K = y.shape

    kernel_vect, u, sr_S = kernel_rank_sign(y,S)
    
    #sr_S = sp.linalg.inv(sp.linalg.sqrtm(S))
    inv_sr_S2 = np.kron(T(sr_S),sr_S)

    I_N = np.eye(N)
    J_n_per = np.eye(N**2) - np.outer(routines.vec(I_N), routines.vec(I_N))/N
    K_V = inv_sr_S2 @ J_n_per
    
    # Kernel_appo = np.zeros((N**2,), dtype=complex)
    # for k in range(K):
    #     uk = u[:,k]
    #     Mat_appo = np.outer(uk,C(uk))
    #     Kernel_appo = Kernel_appo  + kernel_vect[k] * routines.vec(Mat_appo)
        
    Kernel_appo = np.zeros((N,N), dtype=complex)
    for n1 in range(N):
        Kernel_appo[n1,n1:N] = ((u[n1,:].conj() * kernel_vect) * u[n1:N,:]).sum(axis=1)
    Kernel_appo = np.triu(Kernel_appo, 1) + np.tril(Kernel_appo.transpose().conj(), 0)
    Kernel_appo = np.ravel(Kernel_appo, order='C')

    Delta_S = (K_V @ Kernel_appo) / math.sqrt(K)
    Delta_S = np.delete(Delta_S, [0])

    return Delta_S

def alpha_estimator_sub( y, S0, V):
    
    N, K = y.shape 
    
    Delta_S, Psi_S = Delta_Psi_eval(y, S0)
    S_pert = S0 + V/math.sqrt(K)
    Delta_S_pert = Delta_only_eval(y, S_pert)
    V_1 = routines.vec(V)
    V_1 = np.delete(V_1, [0])
    alpha_est = np.linalg.norm(Delta_S_pert-Delta_S)/np.linalg.norm(Psi_S @ V_1)
    
    return alpha_est, Delta_S, Psi_S


# -----------------------------------------------------------


