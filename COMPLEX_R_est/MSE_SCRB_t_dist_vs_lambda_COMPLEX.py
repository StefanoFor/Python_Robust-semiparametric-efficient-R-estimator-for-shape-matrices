# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 17:26:09 2020

@author: Utente
"""

import math
import cmath
import numpy as np
import scipy as sp
from scipy.special import gamma
import matplotlib.pyplot as plt
import numpy.matlib


def hermitian(A, **kwargs):
    return np.transpose(A, **kwargs).conj()
# Make some shortcuts for transpose,hermitian:
#    np.transpose(A) --> T(A)
#    hermitian(A) --> H(A)
T = np.transpose
C = np.conj
H = hermitian

import routines
import Ty_estim_COMPLEX
import R_shape_estim_COMPLEX


#from scipy.linalg import toeplitz

Ns = 10 ** 3
N = 8
perturbation_par = 10**(-2)

rho = 0.8*cmath.exp( 1j*2*math.pi/5 )
sigma2 = 4

lambdavect = np.linspace(1.1, 21.1, 20)

Nl = len(lambdavect)

K = 5 * N
n = np.arange(0, N, 1)

rx = rho ** n
Sigma = C(sp.linalg.toeplitz(rx))
Ls = H(sp.linalg.cholesky(Sigma))
Shape_S = N*Sigma/np.trace(Sigma)

Inv_Shape_S = sp.linalg.inv(Shape_S)

DIM = int(N**2)

J_phi = routines.vec(np.eye(N)).reshape(-1, 1)
U = sp.linalg.null_space(T(J_phi))

Fro_MSE_SCM = np.empty(Nl)
Fro_MSE_Ty = np.empty(Nl)
Fro_MSE_R = np.empty(Nl)
SCRBn = np.empty(Nl)


for il in range(Nl):

    lambdap = lambdavect[il]
    print(lambdap)
    eta = lambdap/(sigma2*(lambdap-1))
    scale=eta/lambdap

    MSE_SCM = np.zeros((DIM,DIM))
    MSE_Ty = np.zeros((DIM,DIM))
    MSE_R = np.zeros((DIM,DIM))
    
    for i in range(Ns):
        # Generation of the t-distributed data
        w = (np.random.randn(N, K) + 1j*np.random.randn(N, K))/math.sqrt(2)
        x = Ls @ w
        R = np.random.gamma(lambdap, scale, size=K)
        y = (1/R) ** (1/2) * x
        
        # Sample Mean and Sample Covariance Matrix (SCM)
        SCM = y @ H(y) / K
        Scatter_SCM = N*SCM/np.trace(SCM)
        err_s = routines.vec(Scatter_SCM-Shape_S)
        err_SCM = np.outer(err_s, C(err_s))
        MSE_SCM = MSE_SCM + err_SCM/Ns
        
        Ty1 = Ty_estim_COMPLEX.compute_tyler_shape_estimator(T(y), np.eye(N))
        Ty = N * Ty1  / np.trace(Ty1)
        
        # MSE mismatch on sigma
        err_v = routines.vec(Ty-Shape_S)
        err_Ty = np.outer(err_v, C(err_v))
        MSE_Ty = MSE_Ty + err_Ty/Ns
        
        mu0 = np.zeros((N,))
        Rm1 = R_shape_estim_COMPLEX.R_estimator_VdW_score(y, mu0, Ty1, perturbation_par)
        Rm = N * Rm1  / np.trace(Rm1)
        
        # MSE mismatch on sigma
        err_rm = routines.vec(Rm-Shape_S)
        err_RM = np.outer(err_rm, C(err_rm))
        MSE_R = MSE_R + err_RM/Ns
        
    # Semiparametric CRB
    a2 = (lambdap + N)/(N + lambdap + 1)
    SFIM_Sigma = K * a2 * (np.kron(T(Inv_Shape_S),Inv_Shape_S) - (1/N) * np.outer(routines.vec(Inv_Shape_S), C(routines.vec(Inv_Shape_S)))) 
    
    SCRB = U @ sp.linalg.inv(H(U) @ SFIM_Sigma @ U) @ H(U)
    SCRBn[il] = np.linalg.norm(SCRB, ord='fro')

    Fro_MSE_SCM[il] = np.linalg.norm(MSE_SCM, ord='fro')
    Fro_MSE_Ty[il] = np.linalg.norm(MSE_Ty, ord='fro')
    Fro_MSE_R[il] = np.linalg.norm(MSE_R, ord='fro')
    



plt.plot(lambdavect, Fro_MSE_SCM, lambdavect, Fro_MSE_Ty, lambdavect, Fro_MSE_R, lambdavect, SCRBn)
plt.ylim(0.3, 0.7)
plt.legend(['SCM','Ty','R','SCRB'])
plt.ylabel('MSE')
plt.xlabel('Shape parameter: lambda')
plt.show()