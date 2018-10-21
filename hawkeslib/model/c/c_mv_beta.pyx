import cython
import numpy as np
cimport numpy as cnp
cimport scipy.special.cython_special as csc

from betaincder.c.betaincder cimport digamma, betaincderp, betaincderq
from scipy.optimize import minimize

cnp.import_array()

cdef extern from "math.h":
    double exp(double x) nogil
    double log(double x) nogil
    double lgamma(double x) nogil
    bint isnan(double x) nogil


cpdef double lsexp(double a, double b) nogil:
    '''fast logsumexp'''
    if a >= b:
        return a + log(1 + exp(b - a))
    else:
        return b + log(1 + exp(a - b))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def mv_beta_ll(cnp.ndarray[ndim=1, dtype=cnp.float64_t] t,
              cnp.ndarray[ndim=1, dtype=long] c,
              cnp.ndarray[ndim=1, dtype=cnp.float64_t] mu,
              cnp.ndarray[ndim=2, dtype=cnp.float64_t] A,
              double th1, double th2, double tmax,
              double T):
    """
    beta-delay multivariate hawkes process
    th1, th2, tmax are beta distribution's alpha, beta, and t_max params respectively
    """
    cdef:
        int N = t.shape[0]
        int K = mu.shape[0]
        int i, k
        long ci, cj
        double ti

        cnp.ndarray[ndim=1, dtype=cnp.float64_t] F = np.zeros(K)
        double lJ = 0., lda = 0.
        double lnbeta_denom, d
        int j

        double logd

    lndenom = log(tmax) + lgamma(th1) + lgamma(th2) - lgamma(th1 + th2)

    for i in range(N):
        ci = c[i]
        ti = t[i]

        lda = mu[ci]

        for j in range(i-1, -1, -1):
            d = ti - t[j]
            if d > tmax:
                break

            logd = (th1 - 1) * log(d / tmax) + (th2 - 1) * log(1 - d/tmax) - lndenom
            lda += A[c[j], c[i]] * exp(logd)

        lJ += log(lda)

        F[ci] += csc.betainc(th1, th2, (T - ti)/tmax) if T - ti < tmax else 1

    return lJ - np.sum(A.T.dot(F)) - np.sum(mu) * T


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _comp_theta(cnp.ndarray[ndim=1, dtype=cnp.float64_t] t,
                cnp.ndarray[ndim=1, dtype=long] c,
                cnp.ndarray[ndim=2, dtype=cnp.float64_t] A,
                double th1,
                double th2,
                double tmax):
    cdef:
        int K = A.shape[0]
        int N = t.shape[0]
        double _sum = 0.

    for i in range(N):
        for k in range(K):
            _sum += A[c[i],k] * csc.betainc(th1, th2, t[i] / tmax)

    return _sum

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _g_comp_theta(cnp.ndarray[ndim=1, dtype=cnp.float64_t] t,
                cnp.ndarray[ndim=1, dtype=long] c,
                cnp.ndarray[ndim=2, dtype=cnp.float64_t] A,
                double th1,
                double th2,
                double tmax, int partial):
    """:param int partial: which derivative to take(0: w.r.t. th1, 1: w.r.t. th2)"""
    cdef:
        int K = A.shape[0]
        int N = t.shape[0]
        double _sum = 0.

    for i in range(N):
        for k in range(K):
            _sum += A[c[i],k] * betaincderp(t[i] / tmax, th1, th2) if partial == 0 \
                    else A[c[i],k] * betaincderq(t[i] / tmax, th1, th2)

    return _sum

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def _optimize_th1_th2(A, double th1, double th2, double tmax,
                      double E2sum, double E3, double E4,
                      t_last, c_last):
    """conditional optimization of Beta delay parameters theta 1 and 2"""

    def f(th):
        return - (E3 * th[0] + E4 * th[1] - E2sum * csc.betaln(th[0], th[1]) \
            - _comp_theta(t_last, c_last, A, th[0], th[1], tmax))

    def g(th):
        cdef double g0, g1
        g0 = - E2sum * (digamma(th[0]) - digamma(th[0] + th[1])) + E3 \
                - _g_comp_theta(t_last, c_last, A, th[0], th[1], tmax, 0)

        g1 = - E2sum * (digamma(th[1]) - digamma(th[0] + th[1])) + E4 \
                - _g_comp_theta(t_last, c_last, A, th[0], th[1], tmax, 1)

        return -np.array([g0, g1])

    minres = minimize(f, x0=np.array([th1, th2]), method="L-BFGS-B", jac=g, bounds=[(0, None), (0, None)])

    return minres.x


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def mv_beta_fit_em(cnp.ndarray[ndim=1, dtype=cnp.float64_t] t,
                  cnp.ndarray[ndim=1, dtype=long] c,
                  double T, double tmax,
                  int maxiter=500, double reltol=1e-5):
    """beta-delay em"""

    cdef:
        int K = np.unique(c).shape[0]
        int i, j = 0, k = 0, N = t.shape[0], l, iix = 0
        long ci
        double ti

        # parameter estimates at time tau
        cnp.ndarray[ndim=1, dtype=cnp.float64_t] mu = np.random.rand(K) * N / (T * K)
        cnp.ndarray[ndim=2, dtype=cnp.float64_t] A = np.eye(K) * .2 + np.ones((K,K)) * .05
        double th1 = 1., th2 = 1.

        # expected sufficient statistics and their local counterparts (if necessary)
        cnp.ndarray[ndim=1, dtype=cnp.float64_t] E1 = np.empty(K)
        cnp.ndarray[ndim=2, dtype=cnp.float64_t] E2 = np.empty((K, K))
        cnp.ndarray[ndim=2, dtype=cnp.float64_t] E2_loc = np.empty((K, K))
        double E3 = 0., E3_loc = 0.
        double E4 = 0., E4_loc = 0.
        double E2sum = 0.,

        cnp.ndarray[ndim=1, dtype=cnp.float64_t] C1 = np.empty(K)

        double d, lndenom, lng, lnlda
        double odll = 0, odll_p, relimp

    odll_p = -np.inf
    t_last = T - t[t >= T - tmax]
    c_last = c[t >= T - tmax]

    for iix in range(maxiter):

        # E-step
        ci = c[0]

        cnp.PyArray_FillWithScalar(E1, 0.)
        cnp.PyArray_FillWithScalar(E2, 0.)
        E3 = 0.
        E4 = 0.
        cnp.PyArray_FillWithScalar(C1, 0.)

        E1[ci] = 1. / mu[ci]
        C1[ci] = csc.betainc(th1, th2, (T - t[0]) / tmax) if T - t[0] < tmax else 1.

        lndenom = log(tmax) + lgamma(th1) + lgamma(th2) - lgamma(th1 + th2)

        for i in range(1, N):
            ci = c[i]
            ti = t[i]

            cnp.PyArray_FillWithScalar(E2_loc, -np.inf)
            E3_loc = 0.
            E4_loc = 0.

            lnlda = log(mu[ci])

            for j in range(i-1, -1, -1):
                d = ti - t[j]
                if d > tmax:
                    break

                lng = (th1 - 1)*log(d / tmax) + (th2 - 1)*log(1 - d/tmax) - lndenom

                E2_loc[c[j], ci] = lsexp(E2_loc[c[j], ci], log(A[c[j], ci]) + lng)  # collected in log domain

                E3_loc += exp(log(A[c[j], ci]) + lng) * log(d/tmax)  # collected in base domain
                E4_loc += exp(log(A[c[j], ci]) + lng) * log(1 - d/tmax)  # collected in base domain

                lnlda = lsexp(lnlda, log(A[c[j], ci]) + lng)

            E1[ci] += exp(-lnlda)
            E3 += E3_loc / exp(lnlda)
            E4 += E4_loc / exp(lnlda)

            for k in range(K):
                for l in range(K):
                    E2[l, k] += exp(E2_loc[l, k] - lnlda)

            C1[ci] += (csc.betainc(th1, th2, (T - ti)/tmax) if T-ti < tmax else 1.)

        # M-step
        E2sum = E2.sum()
        th1, th2 = _optimize_th1_th2(A, th1, th2, tmax, E2sum, E3, E4, t_last, c_last)
        for l in range(K):
            mu[l] = mu[l] * E1[l] / T
            for k in range(K):
                A[l, k] = E2[l,k] / C1[l]

        # calculate observed data log likelihood
        odll = mv_beta_ll(t, c, mu, A, th1, th2, tmax, T)
        relimp = (odll - odll_p) / abs(odll_p)  # relative improvement
        if iix % 50 == 0:
            print odll
        if relimp < -1e-5:
            print odll_p, odll
            raise Exception("Convergence problem, the log likelihood did not increase")
        elif relimp < reltol:
            break
        odll_p = odll

    return odll, (mu, A, th1, th2), iix