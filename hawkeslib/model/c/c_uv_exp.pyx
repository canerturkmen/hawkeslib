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

cdef double uu() nogil:
    return <double> rand() / RAND_MAX


@cython.boundscheck(False)
@cython.wraparound(False)
def uv_exp_phi(cnp.ndarray[ndim=1, dtype=npfloat] t,
               double theta,
               double T):
    """
    Get the ending `phi`, or the "state" of a Hawkes process after a given
    observation

    :param t: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps
    :param mu: the exogenous intensity
    :param alpha: the infectivity factor alpha
    :param theta: intensity parameter of the delay density
    :param T: the maximum time
    :return: the `phi`, or the "state of the hawkes process"
    """

    cdef:
        double phi = 0.
        int N = t.shape[0]
        double d, r
        int j = 0

    with nogil:
        for j in range(N-1):
            d = t[j+1] - t[j]
            ed = exp(-theta * d)  # exp_diff
            phi = ed * (1 + phi)

        phi *= exp(-theta * (T - t[j+1]))

    return phi


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
        int N = t.shape[0]
        double lda, pi, F, r, d
        int j = 0

    if N == 0:
        return lJ + lComp

    with nogil:

        lComp -= alpha * (1 - exp(-theta * (T - t[0])))
        lJ = log(mu)

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

    if N == 0:
        return np.array([1./mu - T, 0., 0.])

    with nogil:

        nmu = 1. / mu
        Calpha = 1 - exp(-theta * (T - t[0]))
        Ctheta = alpha * (T - t[0]) * exp(-theta * (T - t[0]))

        for j in range(N-1):
            d = t[j+1] - t[j]
            r = T - t[j+1]

            ed = exp(-theta * d)
            F = 1 - exp(-theta * r)

            nphi = ed * (d * (1 + phi) + nphi)
            phi = ed * (1 + phi)
            lda = mu + alpha * theta * phi

            nmu = nmu + 1. / lda
            nalpha = nalpha + theta * phi / lda
            ntheta = ntheta + alpha * (phi - theta * nphi) / lda

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
        double d = 0.
        double lda = 0.
        vector[npfloat] td
        int j

    with nogil:
        while t < T:
            M = mu + alpha * theta * (1 + phi)

            r1 = <double>rand() / RAND_MAX
            r2 = <double>rand() / RAND_MAX

            E = -log(r1) / M
            t = t + E

            ed = exp(-theta * (E + d))
            lda = mu + alpha * theta * ed * (1 + phi)

            if t < T and r2 * M <= lda:
                td.push_back(<npfloat> t)
                phi = ed * (1 + phi)
                d = 0
            else:
                d = d + E

    cdef cnp.ndarray[npfloat] res = np.empty(td.size(), dtype=np.float)
    for j in prange(res.shape[0], nogil=True):
        res[j] = td[j]

    return res


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def _get_offspring(double t, double alpha, double theta, double T):
    cdef:
        int N = np.random.poisson(alpha)
        cdef cnp.ndarray[npfloat] os = np.empty(shape=(N,), dtype=np.float)  # offsprings
        int j
        double tt

    with nogil:
        for j in range(N):
            tt = -log(uu()) / theta + t
            os[j] = tt

    return os[os < T]


@cython.boundscheck(False)
@cython.wraparound(False)
def uv_exp_sample_branching(double T, double mu, double alpha, double theta):
    """
    Implement a branching sampler for a univariate exponential HP, taking advantage of the
    cluster process representation. As pointed out by Moller and Rasmussen (2005), this is an approximate sampler
    and suffers from edge effects.
    """
    cdef:
        cnp.ndarray[npfloat] P = np.array([])
        int imm_count
        cnp.ndarray[npfloat] curr_gen

    imm_count = np.random.poisson(mu * T)
    curr_gen = np.random.rand(imm_count) * T

    while len(curr_gen) > 0:
        P = np.concatenate([P, curr_gen])
        offsprings = []
        for k in curr_gen:
            v = _get_offspring(k, alpha, theta, T)
            offsprings.append(v)

        curr_gen = np.concatenate(offsprings)

    P.sort(kind="mergesort")

    return P


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def uv_exp_fit_em_base(cnp.ndarray[ndim=1, dtype=npfloat] t, double T, int maxiter=500, double reltol=1e-5):
    """
    Fit a univariate Hawkes process with exponential decay function using the Expectation-Maximization
    algorithm. The algorithm exploits the memoryless property of the delay density to compute the E-step
    in linear time. Due to the Poisson cluster property of HP, the M-step is in constant time.

    :param t: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps
    :param T: the maximum time
    :param maxiter: int, maximum number of EM iterations
    :param reltol: double, the relative improvement tolerance to stop the algorithm
    :return: tuple, (final log likelihood, (mu, alpha, theta))
    """

    cdef:
        double t0 = t[0], ti, d, r
        double mu, alpha, theta
        double phi, ga, E1, E2, E3, C1, C2
        double ed, er
        double Z, atz, odll = 0., odll_p = 0., relimp
        int i, j = 0, N = len(t)

    mu = N * 0.8 * (1 + (uu() - .5) / 10.) / T
    alpha = 0.2 + (uu() - .5) / 10.
    theta = mu * (1 + (uu() - .5) / 10.)

    odll_p = uv_exp_ll(t, mu, alpha, theta, T)

    for j in range(maxiter):

        # E-step

        # initialize accumulators
        phi, ga = 0, 0

        # initialize ESS
        E1 = 1. / mu
        E2, E3 = 0, 0
        C1, C2 = 0, 0

        with nogil:

            C1 += 1 - exp(-theta * (T - t0))
            C2 += (T - t0) * exp(- theta * (T - t0))

            for i in range(1, N):
                ti = t[i]
                d = ti - t[i-1] + 1e-15
                r = T - ti + 1e-15

                ed = exp(-theta * d)  # log of the exp difference exp(-theta * d)
                er = exp(-theta * r)  # log of the exp difference of time remaining (for the compensator)

                ga = ed * (d * (1 + phi) + ga)
                phi = ed * (1 + phi)

                Z = mu + alpha * theta * phi
                atz = alpha * theta / Z

                # collect ESS

                E1 += 1. / Z
                E2 += atz * phi
                E3 += atz * ga

                C1 += 1 - er
                C2 += r * er

            # M-step

            mu = mu * E1 / T
            theta = E2 / (alpha * C2 + E3)
            alpha = E2 / C1

        # calculate observed data log likelihood

        odll = uv_exp_ll(t, mu, alpha, theta, T)
        relimp = (odll - odll_p) / abs(odll_p)  # relative improvement
        if relimp < -1e-5:
            raise Exception("Convergence problem, the log likelihood did not increase")
        elif relimp < reltol:
            break
        odll_p = odll

    return odll, (mu, alpha, theta), j
