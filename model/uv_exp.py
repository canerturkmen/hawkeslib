"""
Univariate (K=1) Hawkes model with a single exponential delay density.
"""
import numpy as np
from .model import PointProcess
from .c.c_uv_exp import uv_exp_ll, uv_exp_ll_grad, uv_exp_sample_ogata
from scipy.optimize import minimize


class UnivariateExpHawkesProcess(PointProcess):
    """
    Class for the univariate Hawkes process (self-exciting process) with the exponential conditional intensity
    function.

    .. math::

        \lambda^*(t) = \mu + \alpha \theta \sum_{t_i < t} \exp(-\theta (t - t_i))
    """

    _mu = None
    _alpha = None
    _theta = None

    def __init__(self):
        pass

    @classmethod
    def log_likelihood_with_params(cls, t, mu, alpha, theta):
        """
        Calculate the log likelihood of a bounded finite realization, given a set of parameters.

        :param t: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps. must be sorted (asc).
        dtype must be float
        :param mu: the exogenous intensity
        :param alpha: the infectivity factor alpha
        :param theta: intensity parameter of the delay density
        :return:
        """
        assert alpha < 1, "Not stationary!"
        return uv_exp_ll(t, mu, alpha, theta)

    def _fetch_params(self):
        """
        Get the parameters currently in the object.
        :return: 3-tuple, (mu, alpha, theta)
        """
        pars = self._mu, self._alpha, self._theta
        assert None not in pars, "Some parameters seem to be missing. Did you fit() already?"
        return pars

    def set_params(self, mu, alpha, theta):
        """
        Manually set the parameters of process (without fitting).

        :param mu: the exogenous intensity
        :param alpha: the infectivity factor alpha
        :param theta: intensity parameter of the delay density
        :return: 3-tuple, (mu, alpha, theta)
        """
        assert alpha < 1, "Not stationary!"
        assert np.min([mu, alpha, theta]) > 0, "Parameters must be greater than zero!"

        self._mu, self._alpha, self._theta = mu, alpha, theta

    def get_params(self):
        return self._fetch_params()

    def sample(self, T):
        """
        Take an (unconditional) sample from the process using Ogata's modified thinning method.

        :param T: maximum time (samples from :math:`[0, T]`)
        :return: 1-d ndarray of sampled timestamps
        """
        mu, alpha, theta = self._fetch_params()
        return uv_exp_sample_ogata(T, mu, alpha, theta)

    def log_likelihood(self, t, T=None):
        """
        Get the log likelihood of a bounded finite realization on [0,T]. If T is not provided, it is taken
        as the last point in the realization.

        :param t: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps. must be
        sorted (asc). dtype must be float.
        :param T: (optional) maximum time
        :return: the log likelihood
        """
        m, a, th = self._fetch_params()
        if T is None:
            T = t[-1]
        return uv_exp_ll(t, m, a, th, T)

    def fit(self, t, T=None):
        """
        Given a bounded finite realization on [0, T], fit parameters with line search (L-BFGS-B).

        :param t: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps. must be
        sorted (asc). dtype must be float.
        :param T: (optional) maximum time. If None, the last occurrence time will be taken.

        :return: the optimization result
        :rtype: scipy.optimize.optimize.OptimizeResult
        """

        # todo: sometimes converges to a point where the unconditional mean is significantly off
        # todo: from the true value. happens about 5% of the time

        N = len(t)

        if T is None:
            T = t[-1]

        ress = []

        # due to a convergence problem, we reiterate until the unconditional mean starts making sense
        for epoch in range(5):
            # estimate starting mu via the unconditional sample formula assuming
            # $\alpha \approx 0.2$
            mu0 = N * 0.8 / T

            # initialize other parameters randomly
            a0, th0 = np.random.rand(2)
            # todo: initialize th0 better ?

            minres = minimize(lambda x: -uv_exp_ll(t, x[0], x[1], x[2], T),
                              x0=np.array([mu0, a0, th0]),
                              jac=lambda x: -uv_exp_ll_grad(t, x[0], x[1], x[2], T),
                              bounds=[(1e-5, None), (1e-5, 1), (1e-5, None)],
                              method="L-BFGS-B", options={"disp": True, "ftol": 1e-10, "gtol": 1e-8})

            ress.append(minres)
            mu, a, _ = minres.x

            # take the unconditional mean and see if it makes sense
            Napprox = mu * T / (1 - a)
            if abs(Napprox - N)/N < .01:  # if the approximation error is in range, break
                break

        self.set_params(*minres.x)

        return minres
