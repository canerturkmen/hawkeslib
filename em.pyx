import numpy as np
cimport numpy as np
import cython
from libc.stdlib cimport malloc, free
from cython.parallel import prange, parallel
cimport openmp

cdef extern from "fillmem.c":
    ctypedef struct HawkesData:
        unsigned long *times
        unsigned short int *codes
        unsigned int *epar
        int N
        short int maxcode

    HawkesData get_all_data(char* data_filename, int tmax)
    void release_all_data(HawkesData r)

cdef extern from "math.h":
    double exp(double x) nogil
    double log(double x) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double logsumexp(double* a, int N) nogil:
    """
    Pure-C implementation of the log-sum-exp trick
    for an array of doubles.

    This implementation releases the global interpreter lock (GIL).

    :param a: the pointer to the first element of array
    :param N: size of the array
    """
    cdef:
        int i
        double _lsemax = a[0]
        double _lse = 0

    for i in range(N):
        if a[i] > _lsemax:
            _lsemax = a[i]

    for i in range(N):
        _lse += exp(a[i] - _lsemax)

    return _lsemax + log(_lse)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def hawkes_em(str data_filename, int tmax, int maxiter=100, int n_threads=4, lda0=None, Phi0=None, beta0=None):
    """
    Function implementing the E-M algorithm for bounded delay Hawkes processes.
    The function accepts the filename where formatted CSV data is available,
    returns and calculates the parameter vectors lambda, Phi and beta.

    Optionally, starting vectors can be provided.

    The function supports multithreading for significantly faster computation. Furthermore,
    compute-intensive portions of the E-step are compiled to pure C.

    :param data_filename: full absolute path to a data file. Data should be formatted as
        a CSV file of two items: the 'mark' of the event, and the timestamp of the event.
        Both items should be integers. Timestamps are assumed to start from 0. The CSV file
        should have no headers.
    :param tmax: the maximum lookback period, or the delay bound for the Hawkes model
    :param maxiter: maximum number of EM iterations allowed
    :param n_threads: number of threads to compute with
    :param lda0: optional, initial value for the lambda parameter
    :param Phi0: optional, initial value for the Phi parameter
    :param beta0: optional, initial value for the beta parameter

    :returns: Learned parameters. A tuple, (lambda, Phi, beta) all numpy ndarrays of suitable dimension
    """
    ## retrieve the data
    cdef HawkesData mdata = get_all_data(bytes(data_filename), tmax)
    print "Retrieved data (N, maxcode):", mdata.N, mdata.maxcode

    ## define the necessary fixed terms
    cdef double ecdll, ecdll_prev = -1e15
    cdef int K = mdata.maxcode + 1  # number of marks
    cdef int N = mdata.N            # number of events
    cdef long T = mdata.times[N-1]     # timestamp (in millis) of last event
    cdef np.ndarray[ndim=1, dtype=long] S = np.zeros(K, dtype=int) # cardinalities of each parent

    for i in range(N-1): # fill cardinalities
        S[mdata.codes[i]] = S[mdata.codes[i]] + 1

    ## initialize the sufficient statistics

    cdef np.ndarray[ndim=2, dtype=np.float64_t] zeta_ss_loc
    cdef np.ndarray[ndim=3, dtype=np.float64_t] Gamma_ss_loc
    cdef np.ndarray[ndim=2, dtype=np.float64_t] gamma_lnd_ss_loc


    cdef np.ndarray[ndim=1, dtype=np.float64_t] zeta_ss
    cdef np.ndarray[ndim=2, dtype=np.float64_t] Gamma_ss
    cdef np.ndarray[ndim=1, dtype=np.float64_t] gamma_lnd_ss

    ## initialize the parameters

    cdef np.ndarray[ndim=1, dtype=np.float64_t] lda
    cdef np.ndarray[ndim=2, dtype=np.float64_t] Phi
    cdef np.ndarray[ndim=1, dtype=np.float64_t] beta

    lda = np.random.rand(K) if lda0 is None else lda0
    Phi = np.random.rand(K, K) if Phi0 is None else Phi0
    beta = np.random.rand(K) if beta0 is None else beta0

    ## pure C for loops
    cdef int iter, n, n_par, cn, cm, par0, j, d
    cdef double* _lnpar
    cdef double _lnpar0
    cdef double _lsemax, _lse# max buffer and sum for logsumexp

    cdef int tid # for multithreading


    for iter in range(maxiter):

        # E-step
        zeta_ss_loc      = np.zeros((K, n_threads))
        Gamma_ss_loc     = np.zeros((K, K, n_threads))
        gamma_lnd_ss_loc = np.zeros((K, n_threads))

        zeta_ss      = np.zeros(K)
        Gamma_ss     = np.zeros((K, K))
        gamma_lnd_ss = np.zeros(K)

        # MAP
        with nogil, parallel(num_threads=n_threads):
            tid = openmp.omp_get_thread_num()

            for n in prange(N):

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

                zeta_ss_loc[cn,tid] += exp(_lnpar0)

                for j in range(n_par):
                    cm = mdata.codes[par0 + j]
                    d  = mdata.times[n] - mdata.times[par0 + j]

                    if d == 0 or d == tmax:
                        continue

                    Gamma_ss_loc[cm, cn, tid] += exp(_lnpar[j])
                    gamma_lnd_ss_loc[cn, tid] += exp(_lnpar[j]) * log(1 - <double>d / tmax)

                free(_lnpar)

        # REDUCE
        Gamma_ss = Gamma_ss_loc.sum(2)
        zeta_ss  = zeta_ss_loc.sum(1)
        gamma_lnd_ss = gamma_lnd_ss_loc.sum(1)

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

    release_all_data(mdata)

    return lda, Phi, beta


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def hawkes_perplexity(str data_filename, int tmax, np.ndarray[ndim=1, dtype=np.float64_t] lda, np.ndarray[ndim=2, dtype=np.float64_t] Phi,
                                np.ndarray[ndim=1, dtype=np.float64_t] beta, int n_threads=4):
    """
    Function implementing the E-M algorithm for bounded delay Hawkes processes.
    The function accepts the filename where formatted CSV data is available,
    returns and calculates the parameter vectors lambda, Phi and beta.

    Optionally, starting vectors can be provided.

    Supports multithreading as the perplexity calculation can be computationally
    intensive.

    :param data_filename: full absolute path to a data file. Data should be formatted as
        a CSV file of two items: the 'mark' of the event, and the timestamp of the event.
        Both items should be integers. Timestamps are assumed to start from 0. The CSV file
        should have no headers.
    :param tmax: the maximum lookback period, or the delay bound for the Hawkes model
    :param lda: the lambda parameter
    :param Phi: Phi parameter
    :param beta: beta parameter

    :returns: double, the predictive perplexity of the data set under lambda, Phi, beta provided
    """
    cdef HawkesData mdata = get_all_data(bytes(data_filename), tmax)
    print "Retrieved data (N, maxcode):", mdata.N, mdata.maxcode

    cdef:
        int N = mdata.N
        int K = mdata.maxcode + 1
        long T = mdata.times[N-1]     # timestamp (in millis) of last event
        np.ndarray[ndim=1, dtype=long] S = np.zeros(K, dtype=int) # cardinalities of each parent

    for i in range(N-1): # fill cardinalities
        S[mdata.codes[i]] = S[mdata.codes[i]] + 1

    # calculate the log predictive likelihood
    cdef:
        double lpl, d
        double* _lnpar
        int par0, j, n, cn, cm, n_par, tid
        np.ndarray[ndim=1, dtype=np.float64_t] lpl_loc

    lpl_loc = np.zeros(n_threads)

    with nogil, parallel(num_threads=n_threads):
        tid = openmp.omp_get_thread_num()

        for n in prange(N):

            # initialize the temp variables
            par0 = mdata.epar[n]
            cn = mdata.codes[n]
            n_par = n - par0

            _lnpar = <double *> malloc((n_par + 1) * sizeof(double))

            for j in range(n_par):
                cm = mdata.codes[par0 + j]
                d  = mdata.times[n] - mdata.times[par0 + j]

                if d == 0 or d == tmax:
                    _lnpar[j] = 1
                else:
                    _lnpar[j] = log(Phi[cm, cn]) + log(beta[cn]) + (beta[cn] - 1) * log(tmax - d) - beta[cn] * log(tmax)

            _lnpar[n_par] = log(lda[cn])

            lpl_loc[tid] += logsumexp(<double *>_lnpar, n_par + 1)

            free(_lnpar)

    lpl = np.sum(lda * T) + np.sum(Phi.T * S) + np.sum(lpl_loc)

    release_all_data(mdata)

    return exp(lpl / N)
