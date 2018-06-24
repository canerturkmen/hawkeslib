"""
Classes implementing a (homogenous) Poisson Process and a Bayesian version using the fasthawkes interface
"""

import numpy as np
from .model import PointProcess


class PoissonProcess(PointProcess):

    _mu = None

    def __init__(self):
        pass

    @classmethod
    def _prep_t_T(cls, t, T):
        if T is None:
            T = t[-1]
        cls._assert_good_t_T(t, T)

        return t, T

    @classmethod
    def log_likelihood_with_params(cls, t, mu, T=None):
        cls._prep_t_T(t, T)
        return -mu * T + len(T) * np.log(mu)

    def get_params(self):
        assert self._mu is not None, "The intensity parameter appears to be missing, did you fit already?"
        return self._mu

    def set_params(self, mu):
        assert mu > 0, "The intensity must be greater than 0"
        self._mu = mu

    def log_likelihood(self, t, T=None):
        mu = self.get_params()
        return self.log_likelihood_with_params(t, mu, T)

    def fit(self, t, T=None):
        t, T = self._prep_t_T(t, T)

        mu = float(len(t)) / T
        self.set_params(mu)

        return self.log_likelihood(t, T)

