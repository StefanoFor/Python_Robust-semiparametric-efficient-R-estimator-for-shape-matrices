
# -*- coding: utf-8 -*-

import numpy as np

def jacobian_constraint_real(N):
    J_phi = np.zeros((int(N*(N+1)/2), 1))

    for jj in range(N):
        index_con = int(N*jj - jj*(jj-1)/2)
        J_phi[(index_con, 0)] = 1

    return J_phi

def vec(x):
    """ravel matrix in fortran order (stacking columns)
    """
    return np.ravel(x, order='F')

def vech(x):
    """ravel lower triangular part of matrix in fortran order (stacking columns)
    behavior for arrays with more than 2 dimensions not checked yet
    """
    if x.ndim == 2:
        idx = np.triu_indices_from(x.T)
        return x.T[idx[0], idx[1]] #, x[idx[1], idx[0]]
    elif x.ndim > 2:
        #try ravel last two indices
        #idx = np.triu_indices(x.shape[-2::-1])
        n_rows, n_cols = x.shape[-2:]
        idr, idc = np.array([[i, j] for j in range(n_cols)
                                    for i in range(j, n_rows)]).T
        return x[..., idr, idc]


def veclow(x):
    """ravel lower triangular part of matrix excluding diagonal
    This is the same as vech after dropping diagonal elements
    """
    if x.ndim == 2:
        idx = np.triu_indices_from(x.T, k=1)
        return x.T[idx[0], idx[1]] #, x[idx[1], idx[0]]
    else:
        raise ValueError('x needs to be 2-dimensional')


def vech_cross_product(x0, x1):
    """vectorized cross product with lower triangel
    TODO: this should require symmetry, and maybe x1 = x0, otherwise dropping
    above diagonal might not make sense
    TODO: we also want resorted diagonal, off-diagonal for use with correlation
    i.e. std first and then correlation coefficients.
    """
    n_rows, n_cols = x0.shape[-1], x1.shape[-1]
    idr, idc = np.array([[i, j] for j in range(n_cols)
                                for i in range(j, n_rows)]).T
    return x0[..., idr] * x1[..., idc]


def unvec(x, n_rows, n_cols=None):
    """create matrix from fortran raveled 1-d array
    """
    if n_cols is None:
        n_cols = n_rows

    return x.reshape(n_rows, n_cols, order='F')


def unvech(x, n_rows, n_cols=None):
    if n_cols is None:
        n_cols = n_rows

    #  we use triu but transpose to get fortran ordered tril
    n_rows, n_cols = n_cols, n_cols
    idx = np.triu_indices(n_rows, m=n_cols)
    x_new = np.zeros((n_rows, n_cols), dtype=x.dtype)
    x_new[idx[0], idx[1]] = x
    return x_new.T


def dg(x):
    """create matrix with off-diagonal elements set to zero
    """
    return np.diag(x.diagonal())


def E(i, j, nr, nc):
    """create unit matrix with 1 in (i,j)th element and zero otherwise
    """
    x = np.zeros((nr, nc), np.int64)
    x[i, j] = 1
    return x


def K(n):
    """selection matrix
    symmetric case only
    """
    k = sum(np.kron(E(i, j, n, n), E(i, j, n, n).T)
            for i in range(n) for j in range(n))
    return k

def Ms(n):
    k = K(n)
    return (np.eye(*k.shape) + k) / 2.

def u(i, n):
    """unit vector
    """
    u_ = np.zeros(n, np.int64)
    u_[i] = 1
    return u_


def L(n):
    """elimination matrix
    symmetric case
    """
    # they use 1-based indexing
    # k = sum(u(int(round((j - 1)*n + i - 0.5* j*(j - 1) -1)), n*(n+1)//2)[:, None].dot(vec(E(i, j, n, n))[None, :])
    k = sum(u(int(np.trunc((j)*n + i - 0.5* (j + 1)*(j))), n*(n+1)//2)[:, None].dot(vec(E(i, j, n, n))[None, :])
            for i in range(n) for j in range(i+1))
    return k


def Md_0(n):
    l = L(n)
    ltl = l.T.dot(l)
    k = K(n)
    md = ltl.dot(k).dot(ltl)
    return md


def Md(n):
    """symmetric case
    """
    md = sum(np.kron(E(i, i, n, n), E(i, i, n, n).T)
            for i in range(n))
    return md


def Dup(n):
    """duplication matrix
    """
    l = L(n)
    ltl = l.T.dot(l)
    k = K(n)
    d = l.T + k.dot(l.T) - ltl.dot(k).dot(l.T)
    return d


def ravel_indices(n_rows, n_cols):
    """indices for ravel in fortran order
    """
    ii, jj = np.meshgrid(np.arange(n_rows), np.arange(n_cols))
    return ii.ravel(order='C'), jj.ravel(order='C')

