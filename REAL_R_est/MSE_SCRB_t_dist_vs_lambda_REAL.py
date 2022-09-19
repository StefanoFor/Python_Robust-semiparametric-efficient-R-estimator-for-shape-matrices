# -*- coding: utf-8 -*-

import math
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
import Ty_estim_REAL
import R_shape_estim_REAL


#from scipy.linalg import toeplitz

Ns = 10 ** 6
N = 8
perturbation_par = 10**(-2)

rho = 0.8
sigma2 = 4

lambdavect = np.linspace(2.1, 21.1, 20)

Nl = len(lambdavect)

K = 5 * N
n = np.arange(0, N, 1)

rx = rho ** n
Sigma = sp.linalg.toeplitz(rx)
Ls = T(sp.linalg.cholesky(Sigma))
Shape_S = N*Sigma/np.trace(Sigma)
Ln = routines.L(N)
Dn = routines.Dup(N)

Inv_Shape_S = sp.linalg.inv(Shape_S)

DIM = int(N*(N+1)/2)
Fro_MSE_SCM = np.empty(Nl)
Fro_MSE_Ty = np.empty(Nl)
Fro_MSE_R = np.empty(Nl)
SCRBn = np.empty(Nl)


J_phi = routines.jacobian_constraint_real(N)

U = sp.linalg.null_space(T(J_phi))

for il in range(Nl):
    lambdap = lambdavect[il]
    print(lambdap)
    
    eta = lambdap/(sigma2*(lambdap-2))
    scale=eta/lambdap;


    MSE_SCM = np.zeros((DIM,DIM))
    MSE_Ty = np.zeros((DIM,DIM))
    MSE_R = np.zeros((DIM,DIM))
    
    for i in range(Ns):
        # Generation of the t-distributed data
        w = np.random.randn(N, K)
        x = Ls @ w
        R = 2 * np.random.gamma(lambdap/2, 2*scale, size=K)
        y = (1/R ** (1/2)) * x
        
        # Sample Mean and Sample Covariance Matrix (SCM)
        SCM = y @ T(y) / K
        Scatter_SCM = N*SCM/np.trace(SCM)
        err_s = routines.vech(Scatter_SCM-Shape_S)
        err_SCM = np.outer(err_s, err_s)
        MSE_SCM = MSE_SCM + err_SCM/Ns
        
        Ty1 = Ty_estim_REAL.compute_tyler_shape_estimator(T(y), np.eye(N))
        Ty = N * Ty1  / np.trace(Ty1)
        
        # MSE mismatch on sigma
        err_v = routines.vech(Ty-Shape_S)
        err_Ty = np.outer(err_v, err_v)
        MSE_Ty = MSE_Ty + err_Ty/Ns
        
        mu0 = np.zeros((N,))
        Rm1 = R_shape_estim_REAL.R_estimator_VdW_score(y, mu0, Ty1, perturbation_par)
        Rm = N * Rm1  / np.trace(Rm1)
        
        # MSE mismatch on sigma
        err_rm = routines.vech(Rm-Shape_S)
        err_RM = np.outer(err_rm, err_rm)
        MSE_R = MSE_R + err_RM/Ns
        
    # Semiparametric CRB
    a2 = (lambdap + N)/(2*(N+2+lambdap))
    SFIM_Sigma = K * a2 * T(Dn) @ (np.kron(Inv_Shape_S,Inv_Shape_S) - (1/N) * np.outer(routines.vec(Inv_Shape_S), routines.vec(Inv_Shape_S)) ) @ Dn
    SCRB = U @ sp.linalg.inv(T(U) @ SFIM_Sigma @ U) @ T(U)
    SCRBn[il] = np.linalg.norm(SCRB, ord='fro')

    Fro_MSE_SCM[il] = np.linalg.norm(MSE_SCM, ord='fro')
    Fro_MSE_Ty[il] = np.linalg.norm(MSE_Ty, ord='fro')
    Fro_MSE_R[il] = np.linalg.norm(MSE_R, ord='fro')
    



plt.plot(lambdavect, Fro_MSE_SCM, lambdavect, Fro_MSE_Ty, lambdavect, Fro_MSE_R, lambdavect, SCRBn)
plt.ylim(0.3, 1)
plt.legend(['SCM','Ty','R','SCRB'])
plt.ylabel('MSE and Bound')
plt.xlabel('Degrees of freedom: lambda')
plt.show()
