import numpy as np

from model.c.c_uv_exp import uv_exp_ll
from .uv_exp import UnivariateExpHawkesProcess
from .c.c_uv_bayes import cmake_gamma_logpdf, cmake_beta_logpdf


class BayesianUVExpHawkesProcess(UnivariateExpHawkesProcess):
    """
    This class implements a "Bayesian" version of the univariate HP model with exponential
    decay. Specifically one with the conditional intensity function

    .. math::
        \lambda^*(t) = \mu + \sum_{t_i < t} \alpha \theta \exp( - \theta (t - t_i))

    where we take Gamma priors for :math:`\mu, \theta`, and a Beta prior for :math:`\alpha`.

    The hyperparameters to the appropriate Gamma and Beta priors are given during initialization, and
    they are not inferred (e.g. through empirical Bayes).
    """

    def _get_log_posterior_pot(self, t, T, mu_hyp, alpha_hyp, theta_hyp):
        """
        Get the log (unnormalized) posterior as a callable with function
        signature (mu, alpha, beta).

        :param t: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps. must be
        sorted (asc). dtype must be float.
        :param T: (optional) maximum time
        :param mu_hyp: tuple, hyperparameters for the prior for mu. (k, theta) for the shape-scale parameterization of
        the Gamma distribution
        :param alpha_hyp: tuple, hyperparameters for the Beta prior for alpha. (a, b)
        :param theta_hyp: tuple, hyperparameters for the prior for theta. (k, theta) for the shape-scale
        parameterization of the Gamma distribution

        :return: callable, a function with signature (mu, alpha, theta) for evaluating the log unnormalized posterior
        """
        t, T = self._prep_t_T(t, T)

        pr_alpha = cmake_beta_logpdf(*alpha_hyp)
        pr_mu, pr_theta = cmake_gamma_logpdf(*mu_hyp), cmake_gamma_logpdf(*theta_hyp)

        def f0(mu, a, th):
            return uv_exp_ll(t, mu, a, th, T) + pr_alpha(a) + pr_mu(mu) + pr_theta(th)

        return f0

    def fit(self):
        """
        Get the MAP estimate via gradient descent
        :return:
        """
        pass

    def run_inference(self):
        """
        Get samples from the posterior via MCMC
        :return:
        """
        pass
