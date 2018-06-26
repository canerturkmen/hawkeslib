"""
Classes implementing a (homogenous) Poisson Process and a Bayesian version using the fasthawkes interface
"""

import numpy as np
from scipy.special import gammaln

from .c.c_uv_bayes import cmake_gamma_logpdf
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
        t, T = cls._prep_t_T(t, T)
        return -mu * T + len(t) * np.log(mu)

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


class BayesianPoissonProcess(PoissonProcess):
    """
    Implements a "Bayesian" version of the temporal Poisson process with a conjugate Gamma prior
    """

    def _get_log_posterior_pot(self, t, T, mu_hyp):
        """
        Get the log (unnormalized) posterior as a callable with function
        signature (mu,).

        :param t: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps. must be
        sorted (asc). dtype must be float.
        :param T: (optional) maximum time
        :param mu_hyp: tuple, hyperparameters for the prior for mu. (k, theta) for the shape-scale parameterization of
        the Gamma distribution

        :return: callable, a function with signature (mu, alpha, theta) for evaluating the log unnormalized posterior
        """
        t, T = self._prep_t_T(t, T)

        pr_mu = cmake_gamma_logpdf(*mu_hyp)

        def f0(mu, a, th):
            return self.log_likelihood_with_params(t, mu, T) + pr_mu(mu)

        return f0

    @classmethod
    def _get_marginal_likelihood(cls, t, T, mu_hyp):
        """
        Take the marginal likelihood analytically using the Gamma-Poisson conjugacy
        :param t:
        :param T:
        :param mu_hyp: 2-tuple, hyperparameters for the prior for mu. (k, theta) for the shape-scale parameterization of
        the Gamma distribution
        :return:
        """
        t, T = cls._prep_t_T(t, T)

        N = len(t)
        k, theta = mu_hyp

        return gammaln(N + k) - gammaln(k) \
               - (N + k) * np.log(T + 1. / theta) - k * np.log(theta)

    def fit(self, t, T=None):
        """
        Fit a maximum a posteriori (MAP) estimate
        :param t:
        :param T:
        :return:
        """
        pass
