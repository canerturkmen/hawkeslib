import numpy as np
cimport numpy as np
import cython

cdef extern from "math.h":
    double exp(double x) nogil
    double log(double x) nogil

cdef extern from "stdlib.h":
    double rand() nogil
    int RAND_MAX

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def uv_exp_sample(float T, float _a, float _b, float _l0, int buffer_size=-1):

    cdef:
        float s = 0
        np.ndarray[ndim=1, dtype=np.float64_t] t_n
        int i = 0

    if buffer_size == -1:
        # take the unconditional mean x 1.3 as buffer size
        buffer_size = <int> (T * 1.3 * _l0 / (1 - (_a / _b)))

    t_n = np.zeros(buffer_size)

    while s < T:
        lda_bar = _l0 + _a * np.sum(np.exp(- _b * (s - t_n[:i])))

        u = <float>rand() / RAND_MAX
        w = - log(u) / lda_bar
        s += w

        D = <float>rand() / RAND_MAX
        if D * lda_bar <= _l0 + _a * np.sum(np.exp(- _b * (s - t_n[:i]))):
            i += 1
            t_n[i] = s

    if t_n[i-1] > T:
        return t_n[:i-1]
    else:
        return t_n[:i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def uv_exp_loglike(np.ndarray[ndim=1, dtype=np.float64_t] ser, float alpha, float beta, float lda0):
    cdef:
        np.ndarray[ndim=1, dtype=np.float64_t] d = np.diff(ser)
        np.ndarray[ndim=1, dtype=np.float64_t] exp_diff = np.exp(-beta * d)
        np.ndarray[ndim=1, dtype=np.float64_t] B = np.zeros(len(ser))
        int i = 0
        int len_exp_diff = len(exp_diff)
        float ll = 0
        float T = 0

    T = ser[len(ser)-1]

    with nogil:
        for i in range(len_exp_diff):
            B[i+1] = (1 + B[i]) * exp_diff[i]
            i += 1

        ll = lda0 * T

    return np.sum(np.log(lda0 + alpha * B)) \
                - (np.sum(1 - np.exp(- beta * (T - ser) )) * alpha / beta) \
                - ll