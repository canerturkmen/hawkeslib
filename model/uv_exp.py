"""
Univariate (K=1) Hawkes model with a single exponential delay density.
"""
import numpy as np
from .model import PointProcess
from .c.c_uv_exp import *

class UnivariateExpHawkesProcess(PointProcess):

    _alpha = None
    _beta = None
    _lda0 = None

    def __init__(self):
        pass

    @classmethod
    def log_likelihood_with_params(cls, t_n, _a, _b, _l0):
        assert _a < _b, "Not stationary!"
        return uv_exp_loglike(t_n, _a, _b, _l0)

    def _fetch_params(self):
        _a, _b, _l0 = self.get_params()
        assert None not in (_a, _b, _l0), "Some parameters seem to be missing. Did you fit() already?"
        return _a, _b, _l0

    def set_params(self, alpha, beta, lda0):
        """
        Manually set the parameters of the Hawkes process.

        :param alpha:
        :param beta:
        :param lda0:
        :return:
        """
        assert alpha < beta, "Not stationary!"
        assert np.min([lda0, alpha, beta]) > 0, "Parameters cannot be below zero!"

        self._alpha, self._beta, self._lda0 = alpha, beta, lda0

    def get_params(self):
        return self._alpha, self._beta, self._lda0

    def sample(self, T):
        return uv_exp_sample(T, *self._fetch_params())

    def log_likelihood(self, t_n):
        return uv_exp_loglike(t_n, *self._fetch_params())

    def fit(self, t_n, method='lbfgs'):

        lda_hat = len(t_n) / (1.2 * t_n[-1])

        if method == "lbfgs":
            # TODO: enforce inequality constraint
            from scipy.optimize import minimize

            x0 = np.random.rand(2)
            x0[0] *= x0[1]  # make sure initial soln is feasible
            x0 = np.concatenate((x0, (lda_hat, )))

            print x0

            minres = minimize(lambda x: -self.log_likelihood_with_params(t_n, x[0], x[1], x[2]),
                              x0=x0,
                              bounds=[(0, 1), (0, 1), (0, None)],
                              method="L-BFGS-B")
            return minres

            # todo: correctly set params!
            # self.set_params(x[0], x[1], x[2])

        elif method == 'pso':  # particle swarm
            # TODO: enforce inequality constraint
            from pyswarm import pso

            xopt, fopt = pso(lambda x: -self.log_likelihood_with_params(t_n, x[0], x[1], x[2]),
                             [0, 0, 0],
                             [1, 1, lda_hat * 2],
                             f_ieqcons=lambda x: [- x[0] + x[1]])

            print xopt, fopt