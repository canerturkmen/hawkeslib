
import numpy as np
cimport numpy as np
import cython
from libc.stdlib cimport malloc, free
from cython.parallel import prange


from scipy.special import logsumexp

cdef extern from "/home/caner/res/hawkes_em/fillmem.c":
    ctypedef struct HawkesData:
        unsigned long *times
        unsigned short int *codes
        unsigned int *epar
        int N
        short int maxcode

    HawkesData get_all_data(int tmax)

cdef extern from "math.h":
    double exp(double x) nogil
    double log(double x) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def hawkes_em(int tmax, int MAXITER=100):

    ## retrieve the data

    cdef HawkesData mdata = get_all_data(tmax)
    print "Retrieved data (N, maxcode):", mdata.N, mdata.maxcode

    ## define the necessary fixed terms
    cdef double ecdll, ecdll_prev = -1e10
    cdef int K = mdata.maxcode + 1  # number of marks
    cdef int N = mdata.N            # number of events
    cdef long T = mdata.times[N-1]     # timestamp (in millis) of last event
    cdef np.ndarray[ndim=1, dtype=long] S = np.zeros(K, dtype=int) # cardinalities of each parent

    for i in range(N-1): # fill cardinalities
        S[mdata.codes[i]] = S[mdata.codes[i]] + 1

    ## initialize the sufficient statistics

    cdef np.ndarray[ndim=1, dtype=np.float64_t] zeta_ss
    cdef np.ndarray[ndim=2, dtype=np.float64_t] Gamma_ss
    cdef np.ndarray[ndim=1, dtype=np.float64_t] gamma_lnd_ss

    ## initialize the parameters

    cdef np.ndarray[ndim=1, dtype=np.float64_t] lda  = np.random.rand(K)
    cdef np.ndarray[ndim=2, dtype=np.float64_t] Phi  = np.random.rand(K, K)
    cdef np.ndarray[ndim=1, dtype=np.float64_t] beta = np.random.rand(K)

    ## pure C for loops
    cdef int iter, n, n_par, cn, cm, par0, j, d
    cdef double* _lnpar
    cdef double _lnpar0
    cdef double _lsemax, _lse# max buffer and sum for logsumexp

    for iter in range(MAXITER):

        # E-step
        zeta_ss      = np.zeros(K)
        Gamma_ss     = np.zeros((K, K))
        gamma_lnd_ss = np.zeros(K)

        for n in prange(N, nogil=True):

            # initialize the temp variables
            par0 = mdata.epar[n]
            cn = mdata.codes[n]
            n_par = n - par0

            _lnpar = <double *> malloc(n_par * sizeof(double))

            # read variables into temp arrays
            _lnpar0 = log(lda[cn])
            _lsemax = log(lda[cn])

            for j in range(n_par):
                cm = mdata.codes[par0 + j]
                d  = mdata.times[n] - mdata.times[par0 + j]

                if d == 0 or d == tmax:
                    _lnpar[j] = 1
                else:
                    _lnpar[j] = log(Phi[cm, cn]) + beta[cn] * log(1 - <double>d / tmax) + log(beta[cn]) + log(<double>tmax - d)
                    if _lnpar[j] > _lsemax:
                        _lsemax = _lnpar[j]

            # normalize (Bayes rule)
            # run the logsumexp trick
            _lse = exp(_lnpar0 - _lsemax)
            for j in range(n_par):
                _lse += exp(_lnpar[j] - _lsemax)
            _lse = _lsemax + log(_lse)

            _lnpar0 = _lnpar0 - _lse
            for j in range(n_par):
                _lnpar[j] = _lnpar[j] - _lse

            # sum into sufficient statistics

            zeta_ss[cn] += exp(_lnpar0)

            for j in range(n_par):
                cm = mdata.codes[par0 + j]
                d  = mdata.times[n] - mdata.times[par0 + j]

                if d == 0 or d == tmax:
                    continue

                Gamma_ss[cm, cn] += exp(_lnpar[j])
                gamma_lnd_ss[cn] += exp(_lnpar[j]) * log(1 - <double>d / tmax)

            free(_lnpar)

        # M-step

        lda = zeta_ss / T
        Phi = (Gamma_ss.T / S).T
        beta = -  Gamma_ss.sum(0) / gamma_lnd_ss

        # ECDLL

        ecdll = - np.sum(lda * T) + np.sum(np.log(lda) * zeta_ss)
        ecdll += - np.sum(Phi.T * S) + np.sum(np.log(Phi) * Gamma_ss)
        ecdll += np.sum(beta * gamma_lnd_ss) + np.sum(np.log(beta) * Gamma_ss.sum(0))

        print ecdll
        if ecdll - ecdll_prev < .1:
            break
        ecdll_prev = ecdll

    return lda, Phi, beta
