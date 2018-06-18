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
import warnings

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


cdef double lsexp(double a, double b) nogil:
    '''fast logsumexp'''
    if a >= b:
        return a + log(1 + exp(b - a))
    else:
        return b + log(1 + exp(a - b))


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

            r1 = <double>rand() / RAND_MAX
            r2 = <double>rand() / RAND_MAX

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

    return P


@cython.boundscheck(False)
@cython.wraparound(False)
def uv_exp_fit_em(cnp.ndarray[ndim=1, dtype=npfloat] t, double T, int maxiter=500, double reltol=1e-5):
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
        double lnmu, lnalpha = log(.5), lntheta = log(.5), theta
        double lnphi, lnga, lnE1, lnE2, lnE3, lnC1, lnC2, lnath
        double lned, lner, ln1pphi, lnZ, lnatz, odll = -1e15, relimp
        double odll_p
        int i, j = 0, N = len(t)

    lnmu = log(N * 0.8 / T)
    odll_p = uv_exp_ll(t, exp(lnmu), exp(lnalpha), exp(lntheta), T)

    for j in range(maxiter):

        # E-step

        # initialize accumulators
        lnphi = -np.inf
        lnga  = -np.inf

        # initialize ESS
        lnE1 = 0  # log(mu / mu)
        lnE2 = -np.inf
        lnE3 = -np.inf

        with nogil:
            theta = exp(lntheta)

            lnC1 = log(1 - exp(-theta * (T - t0)))
            lnC2 = log(T - t0) - theta * (T - t0)

            lnath = lnalpha + lntheta

            for i in range(1, N):
                ti = t[i]
                d = ti - t[i-1] + 1e-15
                r = T - ti + 1e-15

                lned = - theta * d  # log of the exp difference exp(-theta * d)
                lner = - theta * r  # log of the exp difference of time remaining (for the compensator)
                ln1pphi = lsexp(0, lnphi)  # log(1 + phi(i-1))
                lnga = lned + lsexp(log(d) + ln1pphi, lnga)
                lnphi = lned + ln1pphi

                lnZ = lsexp(lnmu, lnath + lnphi)
                lnatz = lnath - lnZ

                # collect ESS

                lnE1 = lsexp(lnE1, lnmu - lnZ)
                lnE2 = lsexp(lnE2, lnatz + lnphi)
                lnE3 = lsexp(lnE3, lnatz + lnga)

                lnC1 = lsexp(lnC1, log(1 - exp(lner)))
                lnC2 = lsexp(lnC2, log(r) + lner)

            # M-step

            lnmu = lnE1 - log(T)
            lnalpha = lnE2 - lnC1
            lntheta = lnE2 - lsexp(lnE3, lnalpha + lnC2)

        # calculate observed data log likelihood
        odll = uv_exp_ll(t, exp(lnmu), exp(lnalpha), exp(lntheta), T)
        relimp = (odll - odll_p) / abs(odll_p)  # relative improvement
        if relimp < 0:
            raise Exception("Convergence problem, the log likelihood did not increase")
        elif relimp < reltol:
            break
        odll_p = odll

    return odll, (exp(lnmu), exp(lnalpha), exp(lntheta)), j


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

    warnings.warn("This algorithm is currently not working properly and has known issues")

    cdef:
        double t0 = t[0], ti, d, r
        double mu, alpha = 0.5, theta = 0.5
        double phi, ga, E1, E2, E3, C1, C2
        double ed, er
        double Z, atz, odll, odll_p, relimp
        int i, j = 0, N = len(t)

    mu = N * 0.8 / T
    odll_p = uv_exp_ll(t, mu, alpha, theta, T)

    for j in range(maxiter):

        # E-step

        # initialize accumulators
        phi, ga = 0, 0

        # initialize ESS
        E1 = 1
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

                E1 += mu / Z
                E2 += atz * phi
                E3 += atz * ga

                C1 += 1 - er
                C2 += r * er

                # M-step

                mu = E1 / T
                alpha = E2 / C1
                theta = E2 / (E3 + alpha * C2)

        # calculate observed data log likelihood

        odll = uv_exp_ll(t, mu, alpha, theta, T)
        print odll_p, odll
        relimp = (odll - odll_p) / abs(odll_p)  # relative improvement
        if relimp < 0:
            raise Exception("Convergence problem, the log likelihood did not increase")
        elif relimp < reltol:
            break
        odll_p = odll

    return odll, (mu, alpha, theta), j