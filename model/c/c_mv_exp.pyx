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
        np.ndarray[ndim=1, dtype=np.float64_t] phi = np.zeros(K)
        np.ndarray[ndim=2, dtype=np.float64_t] Kap = np.zeros((K, K))
        double B = 0
        double T, A1, A2 # last c_i, last l
        double tli, temp_lk, tm
        int j

    for k in range(K):
        last[k] = -1

    # todo: repeat without gil?
    T = t[N-1]
    A1 = -T * np.sum(lda0)
    for m in range(N):
        j = c[m]
        tm = t[m]

        # for B
        temp_lk = lda0[j]
        for i in range(K):
            tli = t[last[i]]
            if tli < 0:
                continue
            Kap[i, j] = exp(-beta[j] * (tm - tli)) * (1 + Kap[i, i])
            temp_lk += alpha[i, j] * beta[j] * Kap[i, j]

        last[j] = m

        # for A.2
        phi[j] += (1 - exp(-beta[j] * (T - tm)))

        B += log(temp_lk)

    A2 = -alpha.dot(phi).sum()

    free(last)
    return A1 + A2 + B


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# def mv_exp_jac_loglike(np.ndarray[ndim=1, dtype=np.float64_t] t,
#                     np.ndarray[ndim=1, dtype=int] c,
#                     np.ndarray[ndim=2, dtype=np.float64_t] alpha,
#                     np.ndarray[ndim=1, dtype=np.float64_t] beta,
#                     np.ndarray[ndim=1, dtype=np.float64_t] lda0):
#     # TODO: calculate the jacobian!
#
#     cdef:
#         int N = len(t)
#         int K = len(lda0)
#         int* last = <int *> malloc(K*sizeof(int))
#         double* B = <double *> malloc(N*sizeof(double))
#         double ll_sum, l_temp, nll_sum, T # last c_i, last l
#         int ci
#
#         np.ndarray[ndim=2, dtype=np.float64_t] jac_a = np.zeros(alpha.shape)
#         np.ndarray[ndim=1, dtype=np.float64_t] jac_b = np.zeros(beta.shape)
#         np.ndarray[ndim=1, dtype=np.float64_t] jac_l = np.zeros(lda0.shape)
#
#     for k in range(K):
#         last[k] = -1
#
#     with nogil:
#         T = t[N-1]
#         ll_sum = 0
#         nll_sum = 0
#
#         for l in range(K):
#             # nll_sum += T * lda0[l]
#             jac_l[l] += -T
#
#         for i in range(N):
#             ci = c[i]
#             # l_temp = lda0[ci]
#             jac_l[ci] = jac_l[ci] + 1
#
#             for l in range(K):
#                 if last[l] == -1:
#                     B[i] = 0
#                     last[l] = i
#                 else:
#                     # l_temp += alpha[l, ci] * beta[ci] * exp(-beta[ci] * (t[i] - t[last[l]])) * (1 + B[last[l]])
#
#                     jac_a[l, ci] += beta[ci] * exp(-beta[ci] * (t[i] - t[last[l]])) * (1 + B[last[l]])
#                     # todo: jac_b
#
#                     if ci == l:
#                         B[i] = exp(-beta[l] * (t[i] - t[last[l]])) * (1 + B[last[l]])
#                         last[l] = i
#
#                 # for nll
#                 # nll_sum += alpha[l, ci] *  (1 - exp(-beta[ci] * (T - t[i]) ) )
#                 jac_a[l, ci] += (1 - exp(-beta[ci] * (T - t[i]) ) )
#
#             # ll_sum += log(l_temp)
#
#     free(last)
#     free(B)
#     return jac_a, jac_b, jac_l