"""
Classes implementing a (homogenous) Poisson Process and a Bayesian version using the hawkeslib interface
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
        Evaluate log likelihood of parameters given data ``t`` in interval :math:`[0, T`.

        :param numpy.array[float] t: timestamps of observed occurrences of the process, in ascending order
            in a 1-d numpy array.
        :param float mu: the constant intensity (rate) parameter
        :param T: the maximum time for which the observation was made, if not provided it will be set as
            the maximum timestamp in ``t``
        :type T: float or None

        :return: log-likelihood
        :rtype: float
        """
        t, T = cls._prep_t_T(t, T)
        return -mu * T + len(t) * np.log(mu)

    def get_params(self):
        """
        Returns the intensity parameter, ``mu``

        :return: mu, the intensity parameter
        :rtype: float
        """
        assert self._mu is not None, "The intensity parameter appears to be missing, did you fit already?"
        return self._mu

    def set_params(self, mu):
        """
        Sets the intensity parameter, ``mu``

        :param float mu: the (constant) intensity parameter
        """
        assert mu > 0, "The intensity must be greater than 0"
        self._mu = mu

    def log_likelihood(self, t, T=None):
        """
        Evaluate log likelihood of parameters given data ``t`` in interval :math:`[0, T`.

        :param numpy.array[float] t: timestamps of observed occurrences of the process, in ascending order
            in a 1-d numpy array.
        :param T: the maximum time for which the observation was made, if not provided it will be set as
            the maximum timestamp in ``t``
        :type T: float or None

        :return: log-likelihood
        :rtype: float
        """
        mu = self.get_params()
        return self.log_likelihood_with_params(t, mu, T)

    def fit(self, t, T=None):
        """
        Fit a Poisson process intensity given a finite realization (timestamps of observations)
        of the process observed in the bounded interval :math:`[0, T)`.

        The maximum likelihood estimate, in this case, is trivial. This function simply sets :math:`\mu = N / T`
        where :math:`N` is the number of observed points in the process.

        :param numpy.array[float] t: timestamps of observed occurrences of the process, in ascending order
            in a 1-d numpy array.
        :param T: the maximum time for which the observation was made, if not provided it will be set as
            the maximum timestamp in ``t``
        :type T: float or None
        """
        t, T = self._prep_t_T(t, T)

        mu = float(len(t)) / T
        self.set_params(mu)

        return self.log_likelihood(t, T)

    def sample(self, T):
        """
        Generate samples from the Poisson process.

        .. warning::
            Currently not implemented.

        :param float T: the upper bound of the interval to sample for

        :return: sampled timestamps in ascending order
        :rtype: numpy.array[float]
        """
        raise NotImplementedError("The sampler for Poisson processes was not implemented")


class BayesianPoissonProcess(PoissonProcess, BayesianPointProcessMixin):
    """
    Implements a "Bayesian" version of the temporal Poisson process with a (conjugate) Gamma prior,

    .. math::
        \\mu \\sim \\mathcal{G}(k, \\eta)

    where :math:`\\mathcal{G}` denotes the "shape-scale" parameterization of the Gamma distribution.
    The hyperparameters are given during initialization.

    The class implements methods for sampling from the posterior, and computing marginal likelihood. Both
    are trivial due to conjugacy.
    """

    # todo: implement log posterior methods

    def __init__(self, mu_hyp):
        """
        Initialize a BayesianPoissonProcess

        :param tuple[float,float] mu_hyp: hyperparameters for the prior for mu. (k, eta) for
            the shape-scale parameterization of the Gamma distribution
        """
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
        Take the log marginal likelihood (evidence) analytically using the Gamma-Poisson conjugacy. This quantity is
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
        model. This method takes the MAP estimate and sets it as the process parameter.

        :param numpy.array[float] t: timestamps of observed occurrences of the process, in ascending order
            in a 1-d numpy array.
        :param T: the maximum time for which the observation was made, if not provided it will be set as
            the maximum timestamp in ``t``
        :type T: float or None
        :return: log-unnormalized posterior of the maximizing parameter
        :rtype: float
        """
        t, T = self._prep_t_T(t, T)
        k, theta = self.mu_hyp

        mustar = (float(len(t)) + k - 1) / (T + 1. / theta)
        self.set_params(mustar)

        logpot = self._get_log_posterior_pot(t, T, self.mu_hyp)

        return logpot(mustar)

    def marginal_likelihood(self, t, T=None):
        """
        Take the **log** marginal likelihood (evidence) analytically using Gamma-Poisson conjugacy.

        :param numpy.array[float] t: timestamps of observed occurrences of the process, in ascending order
            in a 1-d numpy array.
        :param T: the maximum time for which the observation was made, if not provided it will be set as
            the maximum timestamp in ``t``
        :type T: float or None

        :return: log-marginal likelihood (evidence)
        :rtype: float
        """
        t, T = self._prep_t_T(t, T)
        return self._get_marginal_likelihood(t, T, self.mu_hyp)

    def sample_posterior(self, n_samp, t, T=None):
        """
        Take samples from the posterior distribution of ``mu`` given data ``t``.

        :param int n_samp: number of samples to take
        :param numpy.array[float] t: timestamps of observed occurrences of the process, in ascending order
            in a 1-d numpy array.
        :param T: the maximum time for which the observation was made, if not provided it will be set as
            the maximum timestamp in ``t``
        :type T: float or None

        :rtype: numpy.array[float]
        :return: the samples from the posterior distribution of ``mu``
        """
        t, T = self._prep_t_T(t, T)
        N = len(t)
        k, theta = self.mu_hyp

        return np.random.gamma(k + N, 1. / (T + 1. / theta), size=n_samp)
