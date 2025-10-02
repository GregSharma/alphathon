# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Cython-optimized kernels for maximum performance."""
import numpy as np
cimport numpy as cnp
from libc.math cimport exp, log, sqrt, cos
cimport cython

ctypedef cnp.float64_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef cnp.ndarray[DTYPE_t, ndim=2] rbf_kernel_cy(
    cnp.ndarray[DTYPE_t, ndim=2] X1,
    cnp.ndarray[DTYPE_t, ndim=2] X2,
    double ls,
    double sf2
):
    """Ultra-fast RBF kernel in Cython."""
    cdef int n1 = X1.shape[0]
    cdef int n2 = X2.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=2] K = np.zeros((n1, n2), dtype=np.float64)
    cdef double dist2, ls_inv = 1.0 / (ls * ls)
    cdef int i, j
    
    for i in range(n1):
        for j in range(n2):
            dist2 = (X1[i, 0] - X2[j, 0]) * (X1[i, 0] - X2[j, 0])
            K[i, j] = sf2 * exp(-0.5 * dist2 * ls_inv)
    
    return K


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double trapz_cy(cnp.ndarray[DTYPE_t, ndim=1] y, cnp.ndarray[DTYPE_t, ndim=1] x):
    """Ultra-fast trapezoidal integration in Cython."""
    cdef int n = y.shape[0]
    cdef double result = 0.0
    cdef int i
    
    for i in range(n - 1):
        result += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef cnp.ndarray[DTYPE_t, ndim=1] cumtrapz_cy(
    cnp.ndarray[DTYPE_t, ndim=1] y,
    cnp.ndarray[DTYPE_t, ndim=1] x
):
    """Ultra-fast cumulative trapezoidal integration in Cython."""
    cdef int n = y.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef int i
    
    for i in range(1, n):
        result[i] = result[i-1] + 0.5 * (y[i-1] + y[i]) * (x[i] - x[i-1])
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef cnp.ndarray[DTYPE_t, ndim=1] compute_char_func_cy(
    cnp.ndarray[DTYPE_t, ndim=1] u,
    cnp.ndarray[DTYPE_t, ndim=1] rnd_k,
    cnp.ndarray[DTYPE_t, ndim=1] grid_k
):
    """Ultra-fast characteristic function in Cython."""
    cdef double dk = grid_k[1] - grid_k[0]
    cdef int n_u = u.shape[0]
    cdef int n_k = grid_k.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] phi = np.zeros(n_u, dtype=np.float64)
    cdef int i, j
    
    for i in range(n_u):
        for j in range(n_k):
            phi[i] += cos(u[i] * grid_k[j]) * rnd_k[j]
        phi[i] *= dk
    
    return phi


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef cnp.ndarray[DTYPE_t, ndim=1] gradient_cy(
    cnp.ndarray[DTYPE_t, ndim=1] y,
    cnp.ndarray[DTYPE_t, ndim=1] x
):
    """Ultra-fast numerical gradient in Cython."""
    cdef int n = y.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] grad = np.zeros(n, dtype=np.float64)
    cdef int i
    
    # Forward difference at start
    grad[0] = (y[1] - y[0]) / (x[1] - x[0])
    
    # Central differences in middle
    for i in range(1, n-1):
        grad[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    
    # Backward difference at end
    grad[n-1] = (y[n-1] - y[n-2]) / (x[n-1] - x[n-2])
    
    return grad
