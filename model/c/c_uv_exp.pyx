import numpy as np
cimport numpy as np
import cython
from libc.stdlib cimport malloc, free
from cython.parallel import prange, parallel
cimport openmp


cdef extern from "math.h":
    double exp(double x) nogil
    double log(double x) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double uv_exp_sample():
    pass