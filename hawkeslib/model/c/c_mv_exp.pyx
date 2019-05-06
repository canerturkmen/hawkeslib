import cython
import numpy as np
cimport numpy as cnp

from .c_mv_samp import mv_sample_branching

cnp.import_array()

cdef extern from "math.h":
    double exp(double x) nogil
    double log(double x) nogil
    bint isnan(double x) nogil

cdef extern from "stdlib.h":
    double rand() nogil
    int RAND_MAX

cdef double uu() nogil:
    return <double> rand() / RAND_MAX

@cython.boundscheck(False)
@cython.wraparound(False)
def mv_exp_sample_branching(double T,
                            cnp.ndarray[ndim=1, dtype=cnp.float64_t] mu,
                            cnp.ndarray[ndim=2, dtype=cnp.float64_t] A,
                            double theta):
    # wraps the new generic sampler for backward compatibility
    return mv_sample_branching(T, mu, A, theta, 0, 0, 0, 0)


@cython.boundscheck(False)
@cython.wraparound(False)
def mv_exp_ll(cnp.ndarray[ndim=1, dtype=cnp.float64_t] t,
              cnp.ndarray[ndim=1, dtype=long] c,
              cnp.ndarray[ndim=1, dtype=cnp.float64_t] mu,
              cnp.ndarray[ndim=2, dtype=cnp.float64_t] A, double theta, double T):
    """
    Compute log likelihood for a multivariate Hawkes process with exponential decay

    :param t: the timestamps of a finite realization from a multivariate Hawkes process
    :param c: the 'marks' or the process ids for the realization (note that `len(c) == len(t)` must hold)
    :param mu: the background intensities for the processes, array of length K (number of processes)
    :param A: the infectivity matrix, nonnegative matrix of shape (K, K)
    :param theta: the exponential delay parameter theta
    :param T: the maximum time for which an observation could be made
    """
    cdef:
        int N = t.shape[0]
        int K = mu.shape[0]
        int i, k
        long ci
        double ti

        cnp.ndarray[ndim=1, dtype=cnp.float64_t] phi = np.zeros(K)
        cnp.ndarray[ndim=1, dtype=cnp.float64_t] d = np.ones(K) * np.inf
        cnp.ndarray[ndim=1, dtype=cnp.float64_t] ed = np.zeros(K)
        cnp.ndarray[ndim=1, dtype=cnp.float64_t] F = np.zeros(K)
        double lJ = 0., lda = 0., Aphi = 0.

    if N == 0:
        return lJ - np.sum(mu) * T

    with nogil:
        # for t0
        F[c[0]] += 1 - exp(-theta * (T - t[0]))
        lJ = log(mu[c[0]])
        d[c[0]] = 0.

        for i in range(1, N):
            ci = c[i]
            ti = t[i]

            Aphi = 0
            for k in range(K):
                d[k] += ti - t[i-1]
                ed[k] = exp(-theta * d[k])
                Aphi += A[k, ci] * ed[k] * (1 + phi[k])

            lda = mu[ci] + theta * Aphi
            lJ += log(lda)

            F[ci] += 1 - exp(-theta * (T - ti))

            phi[ci] = ed[ci] * (1 + phi[ci])
            d[ci] = 0.

    return lJ + -np.sum(mu * T) - np.sum(A.T.dot(F))


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def mv_exp_fit_em(cnp.ndarray[ndim=1, dtype=cnp.float64_t] t,
                  cnp.ndarray[ndim=1, dtype=long] c,
                  double T,
                  int maxiter=500, double reltol=1e-5):
    """
    Fit a multivariate Hawkes process with exponential decay using the Expectation-Maximization algorithm.

    :param t: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps
    :param c: Marks of the events (must be labels between 0 .. K-1 for a process with K discrete marks)
    :param T: the maximum time (i.e. the process was observed in the interval [0, T) )
    :param maxiter: int, maximum number of EM iterations
    :param reltol: double, the relative improvement tolerance to stop the algorithm
    :return: tuple, (final log likelihood, (mu, A, theta), number of iterations)
    """

    cdef:
        int K = np.unique(c).shape[0]
        int i, j = 0, k = 0, N = t.shape[0], l
        long ci
        double ti
        double Aphi = 0., lda = 0.
        double rate_scale = N / (T * K)

        cnp.ndarray[ndim=1, dtype=cnp.float64_t] mu = np.random.rand(K) * rate_scale
        cnp.ndarray[ndim=2, dtype=cnp.float64_t] A = np.eye(K) * .2 + np.ones((K,K)) * .05
        double theta = rate_scale * .1

        cnp.ndarray[ndim=1, dtype=cnp.float64_t] E1 = np.empty(K)
        cnp.ndarray[ndim=2, dtype=cnp.float64_t] E2 = np.empty((K, K))
        double E3 = 0., E2sum = 0.
        cnp.ndarray[ndim=1, dtype=cnp.float64_t] C1 = np.empty(K)
        cnp.ndarray[ndim=1, dtype=cnp.float64_t] C2 = np.empty(K)

        cnp.ndarray[ndim=1, dtype=cnp.float64_t] phi = np.zeros(K)
        cnp.ndarray[ndim=1, dtype=cnp.float64_t] nphi = np.zeros(K)
        cnp.ndarray[ndim=1, dtype=cnp.float64_t] d = np.ones(K) * np.inf
        cnp.ndarray[ndim=1, dtype=cnp.float64_t] ed = np.zeros(K)
        double er, gamma
        double odll, odll_p, relimp

    odll_p = -np.inf

    for j in range(maxiter):

        # E-step

        ci = c[0]

        cnp.PyArray_FillWithScalar(E1, 0.)
        cnp.PyArray_FillWithScalar(E2, 0.)
        E3 = 0.
        cnp.PyArray_FillWithScalar(C1, 0.)
        cnp.PyArray_FillWithScalar(C2, 0.)

        E1[ci] = 1. / mu[ci]
        C1[ci] = 1 - exp(-theta * (T - t[0]))
        C2[ci] = (T - t[0]) * exp(-theta * (T - t[0]))

        cnp.PyArray_FillWithScalar(phi, 0.)
        cnp.PyArray_FillWithScalar(nphi, 0.)

        for i in range(1, N):
            ci = c[i]
            ti = t[i]

            er = exp(-theta * (T - ti))

            Aphi = 0
            for k in range(K):
                d[k] += ti - t[i-1]
                ed[k] = exp(-theta * d[k])
                Aphi += A[k, ci] * ed[k] * (1 + phi[k])

            lda = mu[ci] + theta * Aphi

            E1[ci] += 1. / lda
            for l in range(K):
                E2[l, ci] += ed[l] * (1 + phi[l]) / lda

                gamma = ed[l] * (d[l] * (1 + phi[l]) + nphi[l])
                if isnan(gamma):
                    gamma = 0

                E3 += A[l, ci] * gamma / lda

            C1[ci] += 1 - er
            C2[ci] += (T - ti) * er

            nphi[ci] = ed[ci] * (nphi[ci] + d[ci] * (1 + phi[ci]))
            phi[ci] = ed[ci] * (1 + phi[ci])
            d[ci] = 0.

        # M-step
        E2 = E2 * theta * A
        theta = E2.sum() / (A.T.dot(C2).sum() + theta * E3) 
        for l in range(K):
            mu[l] = mu[l] * E1[l] / T
            for k in range(K):
                A[l, k] = E2[l,k] / C1[l]

        # calculate observed data log likelihood
        odll = mv_exp_ll(t, c, mu, A, theta, T)
        relimp = (odll - odll_p) / abs(odll_p)  # relative improvement
        if relimp < -1e-5:
            print(odll_p, odll)
            raise Exception("Convergence problem, the log likelihood did not increase")
        elif relimp < reltol:
            break
        odll_p = odll

    return odll, (mu, A, theta), j

