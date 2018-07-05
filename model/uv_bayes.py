import numpy as np

from .c.c_uv_exp import uv_exp_ll, uv_exp_ll_grad
from .uv_exp import UnivariateExpHawkesProcess
from scipy.stats import gamma, beta
from scipy.optimize import minimize


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

    def __init__(self, mu_hyp, alpha_hyp, theta_hyp):
        """
        Initialize a Bayesian HP model
        :param mu_hyp:
        :param alpha_hyp:
        :param theta_hyp:
        """
        super(BayesianUVExpHawkesProcess, self).__init__()

        self.mu_hyp = mu_hyp
        self.alpha_hyp = alpha_hyp
        self.theta_hyp = theta_hyp

        self._get_log_posterior = lambda t, T: self._get_log_posterior_pot_grad_fns(t, T, mu_hyp, alpha_hyp, theta_hyp)[0]
        self._get_log_posterior_grad = lambda t, T: self._get_log_posterior_pot_grad_fns(t, T, mu_hyp, alpha_hyp, theta_hyp)[1]

    @classmethod
    def _get_log_posterior_pot_grad_fns(cls, t, T, mu_hyp, alpha_hyp, theta_hyp):
        """
        Get the log (unnormalized) posterior (and gradient) as a callable with function
        signature (mu, alpha, beta).

        :param t: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps. must be
        sorted (asc). dtype must be float.
        :param T: (optional) maximum time
        :param mu_hyp: tuple, hyperparameters for the prior for mu. (k, theta) for the shape-scale parameterization of
        the Gamma distribution
        :param alpha_hyp: tuple, hyperparameters for the Beta prior for alpha. (a, b)
        :param theta_hyp: tuple, hyperparameters for the prior for theta. (k, theta) for the shape-scale
        parameterization of the Gamma distribution

        :return: tuple, callables, a function with signature (mu, alpha, theta) for evaluating
        the log unnormalized posterior (and its gradient)
        """
        t, T = cls._prep_t_T(t, T)

        def f0(x):
            mu, a, th = x[0], x[1], x[2]
            res = uv_exp_ll(t, mu, a, th, T)

            res += gamma.logpdf(mu, mu_hyp[0], scale=mu_hyp[1]) \
                + gamma.logpdf(th, theta_hyp[0], scale=theta_hyp[1]) \
                + beta.logpdf(a, alpha_hyp[0], alpha_hyp[1])

            return res

        def g0(x):
            mu, a, th = x[0], x[1], x[2]
            res = uv_exp_ll_grad(t, mu, a, th, T)

            res += (mu_hyp[0] - 1) / mu - 1. / mu_hyp[1] \
                   + (theta_hyp[0] - 1) / th - 1. / theta_hyp[1] \
                   + (alpha_hyp[0] - 1) / a - (alpha_hyp[1] - 1) / (1 - a)

            return res

        return f0, g0

    def _fit_grad_desc(self, t, T=None):
        """
        Given a bounded finite realization on [0, T], fit parameters with line search (L-BFGS-B).

        :param t: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps. must be
        sorted (asc). dtype must be float.
        :param T: (optional) maximum time. If None, the last occurrence time will be taken.

        :return: the optimization result
        :rtype: scipy.optimize.optimize.OptimizeResult
        """

        N = len(t)

        ress = []

        f = self._get_log_posterior(t, T)
        g = self._get_log_posterior_grad(t, T)

        # due to a convergence problem, we reiterate until the unconditional mean starts making sense
        for epoch in range(5):
            # estimate starting mu via the unconditional sample formula assuming
            # $\alpha \approx 0.2$
            mu0 = N * 0.8 / T

            # initialize other parameters randomly
            a0, th0 = np.random.rand(2)

            minres = minimize(lambda x: -f(x), x0=np.array([mu0, a0, th0]),
                              jac=lambda x: -g(x),
                              bounds=[(1e-5, None), (1e-5, 1), (1e-5, None)],
                              method="L-BFGS-B", options={"disp": True, "ftol": 1e-10, "gtol": 1e-8})

            ress.append(minres)
            mu, a, _ = minres.x

            # take the unconditional mean and see if it makes sense
            Napprox = mu * T / (1 - a)
            if abs(Napprox - N)/N < .01:  # if the approximation error is in range, break
                break

        return minres

    def log_posterior_with_params(self, t, mu, alpha, theta, T=None):
        """Evaluate the log potential (unnormalized posterior)"""
        t, T = self._prep_t_T(t, T)
        return self._get_log_posterior(t, T)([mu, alpha, theta])

    def fit(self, t, T=None, **kwargs):
        """
        Get the MAP estimate via gradient descent
        :return:
        """
        t, T = self._prep_t_T(t, T)

        minres = self._fit_grad_desc(t, T)
        params = minres.x

        ll = self.log_likelihood_with_params(t, params[0], params[1], params[2], T)

        self.set_params(*params)

        return ll

    def sample_posterior(self):
        """
        Get samples from the posterior via MCMC
        :return:
        """
        pass
