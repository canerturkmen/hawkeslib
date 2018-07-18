import cython
import numpy as np
cimport numpy as cnp

cdef extern from "math.h":
    double exp(double x) nogil
    double log(double x) nogil

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
        int K = np.unique(c).shape[0]
        int i, k
        long ci
        double ti

        cnp.ndarray[ndim=1, dtype=cnp.float64_t] phi = np.zeros(K)
        cnp.ndarray[ndim=1, dtype=cnp.float64_t] d = np.ones(K) * np.inf
        cnp.ndarray[ndim=1, dtype=cnp.float64_t] ed = np.zeros(K)
        cnp.ndarray[ndim=1, dtype=cnp.float64_t] F = np.zeros(K)
        double lJ = 0., lda = 0., dot = 0.

    with nogil:
        # for t0
        F[c[0]] += 1 - exp(-theta * (T - t[0]))
        lJ = log(mu[c[0]])
        d[c[0]] = 0.

        for i in range(1, N):
            ci = c[i]
            ti = t[i]

            dot = 0
            for k in range(K):
                d[k] += ti - t[i-1]
                ed[k] = exp(-theta * d[k])
                phi[k] = ed[k] * (1 + phi[k])
                dot += A[k, ci] * phi[k]

            lda = mu[ci] + theta * dot

            F[ci] += 1 - exp(-theta * (T - ti))

            lJ += log(lda)
            d[ci] = 0.

    return lJ + -np.sum(mu * T) - np.sum(A.T.dot(F))
