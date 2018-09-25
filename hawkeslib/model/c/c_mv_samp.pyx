"""
Module containing generic sampling code --as well as
special cases for faster computation-- for multivariate
Hawkes processes with factorized kernels
"""

import cython
import numpy as np
cimport numpy as cnp

cdef extern from "math.h":
    double exp(double x) nogil
    double log(double x) nogil
    bint isnan(double x) nogil

cdef extern from "stdlib.h":
    double rand() nogil
    int RAND_MAX

cdef double uu() nogil:
    return <double> rand() / RAND_MAX

cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        void push_back(T&) nogil except+
        size_t size()
        T& operator[](size_t)

cdef enum DelayDistribution:
    EXPONENTIAL, BETA


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def _get_mv_offspring_exp(double t, cnp.ndarray[ndim=1, dtype=cnp.float64_t] Acp, double theta, double T):
    """
    :param t: time of parent
    :param Acp: A[c_parent, :]
    :param K: int, number of processes
    """
    cdef:
        int Nk, K = Acp.shape[0]
        vector[cnp.float64_t] tos
        vector[long] cos
        int i, j
        long k
        double tt

    for k in range(K):
        Nk = np.random.poisson(Acp[k])
        for j in range(Nk):
            tt = -log(uu()) / theta + t
            tos.push_back(<cnp.float64_t> tt)
            cos.push_back(k)

    cdef cnp.ndarray[cnp.float64_t] tres = np.empty(tos.size(), dtype=np.float)
    cdef cnp.ndarray[long] cres = np.empty(cos.size(), dtype=np.int)
    for i in range(tres.shape[0]):
        tres[i] = tos[i]
        cres[i] = cos[i]

    return tres[tres < T], cres[tres < T]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def _get_mv_offspring_generic(double t, cnp.ndarray[ndim=1, dtype=cnp.float64_t] Acp,
                              double theta1, double theta2, double theta3, double theta4, double T,
                              DelayDistribution distid):
    """
    :param t: time of parent
    :param Acp: A[c_parent, :]
    :param K: int, number of processes
    """
    cdef:
        int Nk, K = Acp.shape[0]
        cnp.ndarray[ndim=1, dtype=cnp.float64_t] tos = np.empty([0])
        cnp.ndarray[ndim=1, dtype=long] cos = np.empty([0], dtype=np.int)
        long k

    for k in range(K):
        Nk = np.random.poisson(Acp[k])
        if distid == EXPONENTIAL:
            tos = np.r_[tos, t + np.random.exponential(scale=1./theta1, size=Nk)]
        elif distid == BETA:
            # in the BETA case, we take theta1 = alpha, theta2 = beta, and theta3 = tmax
            tos = np.r_[tos, t + np.random.beta(theta1, theta2, size=Nk) * theta3]
        cos = np.r_[cos, np.ones(Nk, dtype=np.int) * k]

    return tos[tos < T], cos[tos < T]


@cython.boundscheck(False)
@cython.wraparound(False)
def mv_sample_branching(double T,
                            cnp.ndarray[ndim=1, dtype=cnp.float64_t] mu,
                            cnp.ndarray[ndim=2, dtype=cnp.float64_t] A,
                            double theta1, double theta2, double theta3, double theta4,
                            DelayDistribution distid):
    """
    Implements a generic branching sampler for multivariate Hawkes processes with
    factorized and normalized triggering kernels, taking advantage of the
    cluster process representation.
    """
    cdef:
        cnp.ndarray[cnp.float64_t] P = np.array([])
        cnp.ndarray[long] C = np.array([], dtype=np.int)

        int Nk_0, K = mu.shape[0]
        int i, k
        cnp.ndarray[cnp.float64_t] curr_P = np.array([])
        cnp.ndarray[long] curr_C = np.array([], dtype=np.int)

    for k in range(K):
        Nk0 = np.random.poisson(mu[k] * T)

        Pk0 = np.random.rand(Nk0) * T
        Ck0 = np.ones(Pk0.shape[0], dtype=np.int) * k

        curr_P = np.concatenate([curr_P, Pk0])
        curr_C = np.concatenate([curr_C, Ck0])

    while curr_P.shape[0] > 0:
        P = np.concatenate([P, curr_P])
        C = np.concatenate([C, curr_C])

        os_P = []  # offspring timestamps
        os_C = []  # offspring marks

        for i in range(len(curr_P)):
            ci = curr_C[i]
            if distid == EXPONENTIAL:
                # for the exponential decay case, we sample
                tres, cres = _get_mv_offspring_exp(curr_P[i], A[ci, :], theta1, T)
            elif distid == BETA:
                tres, cres = _get_mv_offspring_generic(curr_P[i], A[ci, :], theta1, theta2, theta3, 0, T, BETA)

            os_P.append(tres)
            os_C.append(cres)

        curr_P = np.concatenate(os_P)
        curr_C = np.concatenate(os_C)

    six = np.argsort(P, kind="mergesort")

    return P[six], C[six]