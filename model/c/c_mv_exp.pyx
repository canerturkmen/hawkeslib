import cython
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
# from cython.parallel import prange, parallel
# cimport openmp

cdef extern from "math.h":
    double exp(double x) nogil
    double log(double x) nogil

def mv_log_like(np.ndarray[ndim=1, dtype=np.float64_t] t,
                    np.ndarray[ndim=1, dtype=int] c,
                    np.ndarray[ndim=2, dtype=int] alpha,
                    np.ndarray[ndim=1, dtype=int] beta,
                    np.ndarray[ndim=1, dtype=int] lda0,
                 int K):

    cdef int N = len(t)

    cdef:
        double** last = <double **> malloc(K*sizeof(double*))
        np.ndarray[ndim=2, dtype=long] Z = np.zeros((K, N), dtype=int)
        np.ndarray[ndim=2, dtype=long] B = np.zeros(N, dtype=int)
        double* lci, ll # last c_i, last l
        int ci

    for k in range(K):
        last[k] = NULL

    with nogil:
        for i in range(N):
            ci = c[i]

            for l in range(K):
                if last[l] == NULL:
                    Z[l, i] = 0
                    if l == ci:
                        B[i] = 0
                else:
                    Z[l, i] = (1 + B[last[l]]) * exp( -beta[ci] * (t[i] - t[last[l]]) )
                    if l == ci:
                        B[i] = Z[l, i]

    # np.sum(np.log(Z.sum(0) + lda0))
    pass