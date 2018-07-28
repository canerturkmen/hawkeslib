"""
Multivariate Hawkes process with an exponential decay triggering kernel
"""
import numpy as np
from .model import PointProcess
from .c.c_mv_exp import mv_exp_ll, mv_exp_fit_em, mv_exp_sample_branching


class MultivariateExpHawkesProcess(PointProcess):
    """
    Implements a multivariate Hawkes process with exponential delay density, with the conditional
    intensity function given by

    .. math::
        \lambda^*_k(t) = \mu_k + \sum_{t_i < t} A(c_i, k) \\theta \exp(-\\theta (t - t_i))

    Here, :math:`c_i` refer to a set of discrete "marks". That is, the process is multivariate in the sense that it is
    made up of :math:`K` different processes, which :math:`c_i` and :math:`k` index.
    :math:`A` is a nonnegative matrix with spectral radius less than 1, known as the "infectivity matrix", and governs
    the mutual excitation behavior between these processes. For example, :math:`A_{l, k}` refers to the number
    (in expectation) of further events of mark :math:`k` expected to be caused by each event with mark :math:`l`.
    :math:`\mu_k` is the "base intensity" for each process :math:`k`.

    This class implements methods for evaluating log likelihood, estimating parameters and taking forward samples
    from such a process. As with most other methods in this library, the methods are implemented in Cython.
    """

    _mu = None
    _A = None
    _theta = None

    def __init__(self):
        pass

    @classmethod
    def _prep_t_c_T(cls, t, c, T):
        t, T = cls._prep_t_T(t, T)
        c = c.astype(int)
        t = t.astype(float)
        if len(t) != len(c):
            raise ValueError("timestamp (t) and marks (c) lengths should match!")
        return t, c, T

    @classmethod
    def _check_params(self, mu, A, theta):
        if not max(np.linalg.eigvals(A)) < 1:
            raise ValueError("Infectivity matrix A does not lead to a stationary process")
        if np.min(mu) < 0 or np.min(A) < 0 or theta < 0:
            raise ValueError("Negative parameter values not allowed")
        if mu.shape[0] != A.shape[0]:
            raise ValueError("mu and A dimensions do not match")
        if A.shape[0] != A.shape[1]:
            raise ValueError("A should be a square matrix")
        if mu.ndim != 1 or A.ndim != 2:
            raise ValueError("mu should be a vector, and A a matrix")
        try:
            float(theta)
        except:
            raise ValueError("theta should be a floating point scalar")

    @classmethod
    def log_likelihood_with_params(cls, t, c, mu, A, theta, T=None):
        """
        Compute the log likelihood of a multivariate Hawkes process.

        :param numpy.array[float] t: the timestamps of the observed occurrences, in an interval bounded by 0 and ``T``.
            Must be a 1-dimensional numpy.array in sorted order
        :param numpy.array[int] c: the marks of the observed occurrences. must take values in :math:`\{0, 1, ..., K\}`
            where each value refers to which "process" the occurrence belongs to.
            Must be a 1-dimensional numpy.ndarray with the order
            corresponding to t. That is, if t[i] is the timestamp of an occurrence, c[i] is the corresponding mark.
        :param numpy.array[float] mu: the base intensities :math:`\mu_k`. ``mu[k]`` refers to the background intensity of
            occurrences with mark ``k``. Must be a nonnegative 1-dimensional numpy.array.
        :param numpy.array[float] A: the infectivity matrix :math:`A`. ``A[l, k]`` refers to the mutual excitation behavior,
            *caused* by occurrences of mark ``l``, resulting in occurrences of mark ``k`` (note the **index order**
            here). Must be a nonnegative 2-dimensional numpy.array that corresponds to a matrix of spectral radius less
            than 1. That is, the operator norm (greatest eigenvalue) of the matrix should be strictly less than 1.
        :param float theta: the delay parameter :math:`\\theta`, governs the delay effect of the mutually exciting occurrences.
            Corresponds to the rate (inverse of *scale*) parameter of an exponential distribution. Must be nonnegative.
        :param T: (optional) the upper bound of the observation interval. If not provided, the timestamp of
            the last occurrence will be taken
        :type T: float or None

        :return: log likelihood of the parameters under the data ``t, c, T``.
        :rtype: float
        """
        cls._check_params(mu, A, theta)
        t, c, T = cls._prep_t_c_T(t, c, T)
        return mv_exp_ll(t, c, mu, A, theta, T)

    def _fetch_params(self):
        if self._mu is None or self._A is None or self._theta is None:
            raise ValueError("Some parameters seem to be missing. Did you fit() already?")
        return self._mu, self._A, self._theta

    def set_params(self, mu, A, theta):
        """
        Set parameters ``mu, A, theta``.

        :param numpy.array[float] mu: the base intensities :math:`mu_k`. Must be a nonnegative 1-dimensional numpy.array.
        :param numpy.array[float] A: the infectivity matrix :math:`A`.
            Must be a nonnegative 2-dimensional numpy.array that corresponds to a matrix of spectral radius less
            than 1. That is, the operator norm (greatest eigenvalue) of the matrix should be strictly less than 1.
        :param float theta: the delay parameter, governs the delay effect of the mutually exciting occurrences.
        """
        self._check_params(mu, A, theta)
        self._mu, self._A, self._theta = mu, A, theta

    def get_params(self):
        """
        Retrieve parameters of a fit MultivariateExpHawkesProcess object. Only works if the process has been ``fit``
        before or parameters have been set.

        :returns: 3-tuple, corresponding to ``mu``, ``A``, ``theta``.
        :rtype: tuple[numpy.array, numpy.array, float]
        """
        return self._fetch_params()

    def sample(self, T):
        """
        Take a (unconditional) forward sample of occurrences from the process, in the interval :math:`[0, T)`.
        This routine uses the "branching" sampler, relying on the clustering property of Hawkes processes (and not
        Ogata's thinning method).

        Only works if the process has been ``fit`` before or parameters have been set.

        :param float T: the upper bound of the interval in which to sample.

        :returns: 2-tuple, corresponding to timestamps ``t``, and marks ``c``.
        :rtype: tuple[numpy.array[float], numpy.array[long]]
        """
        assert T > 0, "T should be a positive number"
        mu, A, theta = self._fetch_params()
        return mv_exp_sample_branching(T, mu, A, theta)

    def log_likelihood(self, t, c, T=None):
        """
        Compute the log likelihood of fitted parameters of a  multivariate Hawkes process, under the data provided.
        Only works if the process has been ``fit`` before or parameters have been set.

        :param numpy.array[float] t: the timestamps of the observed occurrences, in an interval bounded by 0 and ``T``.
            Must be a 1-dimensional numpy.array in sorted order
        :param numpy.array[int] c: the marks of the observed occurrences. must take values in :math:`\{0, 1, ..., K\}`
            where each value refers to which "process" the occurrence belongs to.
            Must be a 1-dimensional numpy.ndarray with the order
            corresponding to t. That is, if t[i] is the timestamp of an occurrence, c[i] is the corresponding mark.
        :param T: (optional) the upper bound of the observation interval. If not provided, the timestamp of
            the last occurrence will be taken
        :type T: float or None

        :return: log likelihood of the set parameters under the data ``t, c, T``.
        :rtype: float
        """
        mu, A, th = self._fetch_params()
        t, c, T = self._prep_t_c_T(t, c, T)
        return mv_exp_ll(t, c, mu, A, th, T)

    def fit(self, t, c, T=None, **kwargs):
        """
        Fit the parameters of the process given data ``t``, ``c``, ``T``, and set them. This routine uses the
        Expectation-Maximization (EM) algorithm.

        :param numpy.array[float] t: the timestamps of the observed occurrences, in an interval bounded by 0 and ``T``.
            Must be a 1-dimensional numpy.array in sorted order
        :param numpy.array[int] c: the marks of the observed occurrences. must take values in :math:`\{0, 1, ..., K\}`
            where each value refers to which "process" the occurrence belongs to.
            Must be a 1-dimensional numpy.ndarray with the order
            corresponding to t. That is, if t[i] is the timestamp of an occurrence, c[i] is the corresponding mark.
        :param T: (optional) the upper bound of the observation interval. If not provided, the timestamp of
            the last occurrence will be taken
        :type T: float or None
        :param kwargs: see below.

        :return: likelihood of the process under the fit parameters
        :rtype: float

        **Keyword Arguments**

        * *reltol* (``float``) --
          The relative log likelihood improvement used as a stopping condition. Defaults to ``1e-5``.
        * *maxiter* (``int``) --
          The maximum number of iterations. Defaults to 500.
        """
        t, c, T = self._prep_t_c_T(t, c, T)

        emkwargs = {k: v for k, v in kwargs.items() if k in ["maxiter", "reltol"]}

        ll, params, _ = mv_exp_fit_em(t, c, T, **emkwargs)

        self.set_params(*params)
        return ll