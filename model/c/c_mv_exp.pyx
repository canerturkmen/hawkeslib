import cython
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
# from cython.parallel import prange, parallel
# cimport openmp

cdef extern from "math.h":
    double exp(double x) nogil
    double log(double x) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def mv_exp_loglike(np.ndarray[ndim=1, dtype=np.float64_t] t,
                    np.ndarray[ndim=1, dtype=int] c,
                    np.ndarray[ndim=2, dtype=np.float64_t] alpha,
                    np.ndarray[ndim=1, dtype=np.float64_t] beta,
                    np.ndarray[ndim=1, dtype=np.float64_t] lda0):

    cdef:
        int N = len(t)
        int K = len(lda0)
        int* last = <int *> malloc(K*sizeof(int))
        double* B = <double *> malloc(N*sizeof(double))
        double ll_sum, l_temp, nll_sum, T # last c_i, last l
        int ci

    for k in range(K):
        last[k] = -1

    with nogil:
        T = t[N-1]
        ll_sum = 0
        nll_sum = 0

        for l in range(K):
            nll_sum += T * lda0[l]

        for i in range(N):
            ci = c[i]
            l_temp = lda0[ci]

            for l in range(K):
                if last[l] == -1:
                    B[i] = 0
                    last[l] = i
                else:
                    l_temp += alpha[l, ci] * beta[ci] * exp(-beta[ci] * (t[i] - t[last[l]])) * (1 + B[last[l]])
                    if ci == l:
                        B[i] = exp(-beta[l] * (t[i] - t[last[l]])) * (1 + B[last[l]])
                        last[l] = i

                # for nll
                nll_sum -= alpha[l, ci] *  (1 - exp(-beta[ci] * (T - t[i]) ) )

            ll_sum += log(l_temp)

    free(last)
    free(B)
    return ll_sum - nll_sum