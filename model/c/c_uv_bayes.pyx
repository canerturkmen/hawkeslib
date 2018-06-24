"""
Helper functions for Bayesian inference in univariate HP
"""
import cython
from scipy.special import gammaln, betaln

cdef extern from "math.h":
    double log(double x) nogil


@cython.cdivision(True)
def cmake_gamma_logpdf(float k, float zeta):
    cdef float log_part
    log_part = k * log(zeta) +gammaln(k)

    def f0(float x):
        return (k - 1) * log(x) - x / zeta - log_part

    return f0


@cython.cdivision(True)
def cmake_beta_logpdf(float a, float b):
    cdef float log_part
    log_part = betaln(a, b)

    def f0(float x):
        return (a-1) * log(x) + (b-1) * log(1-x) - log_part
    return f0


