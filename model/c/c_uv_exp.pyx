"""
Module defining Cython functions for calculating the likelihood and the gradient of
a univariate Hawkes process with exponential decay function. Also implements Ogata's
modified thinning method for sampling.

This pyx module must be compiled through c++!
"""
import cython
import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange

ctypedef cnp.float64_t npfloat

cdef extern from "math.h":
    double exp(double x) nogil
    double log(double x) nogil

cdef extern from "stdlib.h":
    double rand() nogil
    int RAND_MAX

cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        void push_back(T&) nogil except+
        size_t size()
        T& operator[](size_t)


@cython.boundscheck(False)
@cython.wraparound(False)
def uv_exp_ll(cnp.ndarray[ndim=1, dtype=npfloat] t, double mu, double alpha, double theta, double T):
    """
    Likelihood of a univariate Hawkes process with exponential decay.

    :param t: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps
    :param mu: the exogenous intensity
    :param alpha: the infectivity factor alpha
    :param theta: intensity parameter of the delay density
    :param T: the maximum time
    :return: the log likelihood
    """

    cdef:
        double phi = 0.
        double lComp = -mu * T
        double lJ = 0
        int N = len(t)
        double lda, pi, F, r, d
        int j = 0

    with nogil:
        for j in range(N-1):
            d = t[j+1] - t[j]
            r = T - t[j+1]

            ed = exp(-theta * d)  # exp_diff
            F = 1 - exp(-theta * r)

            phi = ed * (1 + phi)
            lda = mu + alpha * theta * phi

            lJ = lJ + log(lda)
            lComp -= alpha * F

    return lJ + lComp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def uv_exp_ll_grad(cnp.ndarray[ndim=1, dtype=npfloat] t, double mu, double alpha, double theta, double T):
    """
    Calculate the gradient of the likelihood function w.r.t. parameters mu, alpha, theta

    :param t: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps
    :param mu: the exogenous intensity
    :param alpha: the infectivity factor alpha
    :param theta: intensity parameter of the delay density
    :param T: the maximum time
    :return: the gradient as a numpy.array of shape (3,). Gradients w.r.t. mu, alpha, theta respectively
    """

    cdef:
        double phi = 0., nphi = 0.
        double Calpha = 0., Ctheta = 0.
        double nmu = 0., nalpha = 0., ntheta = 0.
        int N = len(t), j = 0
        double d = 0., r = 0.

    with nogil:
        for j in range(N-1):
            d = t[j+1] - t[j]
            r = T - t[j+1]

            ed = exp(-theta * d)
            F = 1 - exp(-theta * r)

            nphi = ed * (-d * (1 + phi) + nphi)
            phi = ed * (1 + phi)
            lda = mu + alpha * theta * phi

            nmu = nmu + 1. / lda
            nalpha = nalpha + theta * phi / lda
            ntheta = ntheta + alpha * (phi + theta * nphi) / lda

            Calpha = Calpha + F
            Ctheta = Ctheta + alpha * r * (1 - F)

    return np.array([nmu - T, nalpha - Calpha, ntheta - Ctheta])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def uv_exp_sample_ogata(double T, double mu, double alpha, double theta, double phi=0):
    """
    Implements Ogata's modified thinning algorithm for sampling from a univariate Hawkes process
    with exponential decay.

    :param T: the maximum time
    :param mu: the exogenous intensity
    :param alpha: the infectivity factor alpha
    :param theta: intensity parameter of the delay density
    :param phi: (optionally) the starting phi, the running sum of exponential differences specifying the history until
    a certain time, thus making it possible to take conditional samples

    :return: 1-d numpy ndarray of samples
    """

    cdef:
        double t = 0.
        double ed = 0.
        double lda = 0.
        vector[npfloat] td
        int j

    with nogil:
        while t < T:
            M = mu + alpha * theta * (1 + phi)

            r1 = rand() / RAND_MAX
            r2 = rand() / RAND_MAX

            E = -log(r1) / M
            t = t + E

            ed = exp(-theta * E)
            lda = mu + alpha * theta * ed * (1 + phi)

            if t < T and r2 * M <= lda:
                td.push_back(<npfloat> t)
                phi = ed * (1 + phi)

    cdef cnp.ndarray[npfloat] res = np.empty(td.size(), dtype=np.float)
    for j in prange(res.shape[0], nogil=True):
        res[j] = td[j]

    return res
