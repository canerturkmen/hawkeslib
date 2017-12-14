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
    # todo: check B once more
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

        B += log(temp_lk + 1e-12)
        last[j] = m

        # for A.2
        phi[j] += (1 - exp(-beta[j] * (T - tm)))

    A2 = -alpha.dot(phi).sum()

    free(last)
    print A1, A2, B, A1+A2+B
    return A1 + A2 + B


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def mv_exp_jac_loglike(np.ndarray[ndim=1, dtype=np.float64_t] t,
                    np.ndarray[ndim=1, dtype=int] c,
                    np.ndarray[ndim=2, dtype=np.float64_t] alpha,
                    np.ndarray[ndim=1, dtype=np.float64_t] beta,
                    np.ndarray[ndim=1, dtype=np.float64_t] lda0):

    cdef:
        int N = len(t)
        int K = len(lda0)
        int* last = <int *> malloc(K*sizeof(int))
        np.ndarray[ndim=1, dtype=np.float64_t] phi = np.zeros(K)
        np.ndarray[ndim=1, dtype=np.float64_t] nab_phi = np.zeros(K)
        np.ndarray[ndim=2, dtype=np.float64_t] Kap = np.zeros((K, K))
        np.ndarray[ndim=2, dtype=np.float64_t] n_Kap = np.zeros((K, K))
        double B = 0
        double T, A1, A2 # last c_i, last l
        double tli, temp_lk, temp_lk_inv, tm
        int j

        np.ndarray[ndim=2, dtype=np.float64_t] jac_a = np.zeros((K, K))
        np.ndarray[ndim=1, dtype=np.float64_t] jac_b = np.zeros(K)
        np.ndarray[ndim=1, dtype=np.float64_t] jac_l = np.zeros(K)


    for k in range(K):
        last[k] = -1

    # from A.1
    T = t[N-1]

    jac_l = - np.ones(K) * T

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
            n_Kap[i, j] = exp(-beta[j] * (tm - tli)) * ((tm - tli) + Kap[i, i])
            temp_lk += alpha[i, j] * beta[j] * Kap[i, j]


        # from (B)
        temp_lk_inv = 1 / (temp_lk + 1e-12)
        jac_l[j] += temp_lk_inv
        jac_b[j] += temp_lk_inv * (alpha[:, j].dot(Kap[:, j]) - beta[j] * alpha[:, j].dot(n_Kap[:, j]))
        jac_a[:, j] += temp_lk_inv * beta[j] * Kap[:, j]

        last[j] = m

        # for A.2
        phi[j] += (1 - exp(-beta[j] * (T - tm)))
        nab_phi[j] += (T - tm) *  exp(-beta[j] * (T - tm))

    # print "before A.2"
    # print "a", jac_a
    # print "b", jac_b
    # print "l", jac_l
    # print "phi", phi
    # print "nab_phi", nab_phi

    # from A.2
    jac_a += phi  # broadcast op, \nabla A[i, j] (A.2) = -\phi(j)
    jac_b += alpha.dot(nab_phi)

    # print "after A.2"
    # print "a", jac_a
    # print "b", jac_b
    # print "l", jac_l

    free(last)
    return jac_a, jac_b, jac_l