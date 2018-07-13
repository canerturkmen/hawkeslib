"""
Classes implementing a (homogenous) Poisson Process and a Bayesian version using the fasthawkes interface
"""

import numpy as np
from scipy.special import gammaln

from .c.c_uv_bayes import cmake_gamma_logpdf
from .model import PointProcess, BayesianPointProcessMixin


class PoissonProcess(PointProcess):
    """
    This class implements a basic temporal homogenous (stationary) Poisson process.
    """

    _mu = None

    def __init__(self):
        pass

    @classmethod
    def log_likelihood_with_params(cls, t, mu, T=None):
        """
        Calculate the poisson process log likelihood of a finite realization observed in the interval
        ..math:`[0,T)`

        :param t: finite realization (timestamps) of the process
        :param mu: the constant intensity parameter
        :param T: the maximum time for which the observation was made
        :return: float, log-likelihood
        """
        t, T = cls._prep_t_T(t, T)
        return -mu * T + len(t) * np.log(mu)

    def get_params(self):
        assert self._mu is not None, "The intensity parameter appears to be missing, did you fit already?"
        return self._mu

    def set_params(self, mu):
        assert mu > 0, "The intensity must be greater than 0"
        self._mu = mu

    def log_likelihood(self, t, T=None):
        """
        Log likelihood, given parameters
        :param t: finite realization (timestamps) of the process
        :param T: the maximum time for which the observation was made
        :return: float, log-likelihood
        """
        mu = self.get_params()
        return self.log_likelihood_with_params(t, mu, T)

    def fit(self, t, T=None):
        """
        Fit a poisson process constant intensity given a finite realization of the process observed
        on the bounded interval ..math:`[0, T)`. This function simply sets ..math:`\mu = N / T` where
        N is the number of observed points in the process.

        :param t: finite realization (timestamps) of the process
        :param T: the maximum time for which the observation was made
        :return:
        """
        t, T = self._prep_t_T(t, T)

        mu = float(len(t)) / T
        self.set_params(mu)

        return self.log_likelihood(t, T)


class BayesianPoissonProcess(PoissonProcess, BayesianPointProcessMixin):
    """
    Implements a "Bayesian" version of the temporal Poisson process with a conjugate Gamma prior
    """

    def __init__(self, mu_hyp):
        super(BayesianPoissonProcess, self).__init__()
        self.mu_hyp = mu_hyp

    @classmethod
    def _get_log_posterior_pot(cls, t, T, mu_hyp):
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
        t, T = cls._prep_t_T(t, T)

        pr_mu = cmake_gamma_logpdf(*mu_hyp)

        def f0(mu):
            return cls.log_likelihood_with_params(t, mu, T) + pr_mu(mu)

        return f0

    @classmethod
    def _get_marginal_likelihood(cls, t, T, mu_hyp):
        """
        Take the marginal likelihood (evidence) analytically using the Gamma-Poisson conjugacy. This quantity is
        simply the integral over the posterior potential.

        .. math::
            \int_0^{\infty} p(D | \theta)p(\theta) d\theta

        :param t: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps. must be
        sorted (asc). dtype must be float.
        :param T: (optional) maximum time
        :param mu_hyp: 2-tuple, hyperparameters for the prior for mu. (k, theta) for the shape-scale parameterization of
        the Gamma distribution

        :return: float, the log marginal likelihood
        """
        t, T = cls._prep_t_T(t, T)

        N = len(t)
        k, theta = mu_hyp

        return gammaln(N + k) - gammaln(k) \
               - (N + k) * np.log(T + 1. / theta) - k * np.log(theta)

    def fit(self, t, T=None):
        """
        Fit a maximum a posteriori (MAP) estimate using the closed form maximizer of the Gamma-Poisson
        model.

        :param t: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps. must be
        sorted (asc). dtype must be float.
        :param T: (optional) maximum time

        :return: float, log-unnormalized posterior of the maximizer
        """
        t, T = self._prep_t_T(t, T)
        k, theta = self.mu_hyp

        mustar = (float(len(t)) + k - 1) / (T + 1. / theta)
        self.set_params(mustar)

        logpot = self._get_log_posterior_pot(t, T, self.mu_hyp)

        return logpot(mustar)

    def marginal_likelihood(self, t, T=None):
        t, T = self._prep_t_T(t, T)
        return self._get_marginal_likelihood(t, T, self.mu_hyp)

    def sample_posterior(self, n_samp, t, T=None):
        """
        Take samples from the posterior distribution of parameters

        :param n_samp: number of samples to take
        :param t: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps. must be
        sorted (asc). dtype must be float.
        :param T: (optional) maximum time

        :return: numpy.array, the samples from the posterior
        """
        t, T = self._prep_t_T(t, T)
        N = len(t)
        k, theta = self.mu_hyp

        return np.random.gamma(k + N, 1. / (T + 1. / theta), size=n_samp)
