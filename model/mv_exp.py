"""
Multivariate Hawkes model with exponential delay density.
"""
import numpy as np
from .model import PointProcess
from .c.c_mv_exp import *


class MultivariateExpHawkesProcess(PointProcess):

    _alpha = None
    _beta = None
    _lda0 = None

    def __init__(self):
        pass

    def conditional_sample(self, t_n, c_n, T):
        pass

    def sample(self, T):
        pass

    @classmethod
    def assert_stationarity(cls, alpha):
        assert np.linalg.norm(alpha, ord=2) <= 1, "Not stationary!"

    @classmethod
    def log_likelihood_with_params(cls, t_n, c_n, _a, _b, _l0):
        # cls.assert_stationarity(_a)

        # todo: more asserts

        return mv_exp_loglike(t_n, c_n, _a, _b, _l0)

    @classmethod
    def grad_flat_loglike_with_params(cls, t_n, c_n, _a, _b, _l0):
        ga, gb, gl0 = mv_exp_jac_loglike(t_n, c_n, _a, _b, _l0)
        K = len(gb)
        return np.concatenate((gl0, gb, ga.reshape((K * K,))))

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
        # self.assert_stationarity(alpha)
        assert np.min(map(np.min, [alpha, beta, lda0])) >= 0, "Parameters cannot be below zero!"

        assert alpha.shape[0] == alpha.shape[1], "Matrix must be square!!"
        assert alpha.shape[0] == len(beta), "Alpha and beta sizes are inconsistent"
        assert alpha.shape[0] == len(beta), "lambda and alpha sizes are inconsistent"

        self._alpha, self._beta, self._lda0 = alpha, beta, lda0

    def get_params(self):
        return self._alpha, self._beta, self._lda0

    def log_likelihood(self, t_n, c_n):
        a, b, l = self._fetch_params()
        return mv_exp_loglike(t_n, c_n, a, b, l)

    def fit(self, t_n, c_n):
        from scipy.optimize import minimize

        c_n = c_n.astype(np.int32)
        K = len(np.unique(c_n))
        T = t_n[-1]

        # sensible starting values for lambda
        lda_hat = np.zeros(K)
        for k in range(K):
            lda_hat[k] = np.sum(c_n == k) / (1.2 * T)

        alpha = np.eye(K) * np.random.rand() * .5 + np.random.rand(K, K) * .1
        beta  = np.random.rand(K)

        # we code the vector as
        # x[:K] -> lambda, x[K:2*K] -> beta, reshape(x[2*K:]) -> alpha

        x0 = np.concatenate((lda_hat, beta, alpha.reshape((K*K,))))

        print x0.shape
        print x0

        minres = minimize(lambda x: -self.log_likelihood_with_params(t_n, c_n,
                                                                     np.reshape(x[2*K:], (K, K)),
                                                                     x[K: 2*K],
                                                                     x[:K]),
                              jac=lambda x: -self.grad_flat_loglike_with_params(t_n, c_n,
                                                                     np.reshape(x[2*K:], (K, K)),
                                                                     x[K: 2*K],
                                                                     x[:K]),
                              x0=x0,
                              bounds=[(0, None) for i in range(len(x0))],
                              method="L-BFGS-B",
                              options={"eps": 1e-9, "ftol": 1e-13, "gtol": 1e-12}
                          )
        xopt = minres.x
        self.set_params(np.reshape(xopt[2*K:], (K, K)), xopt[K: 2*K], xopt[:K])

        return minres


