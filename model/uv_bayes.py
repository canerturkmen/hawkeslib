import numpy as np
import numdifftools as nd
from .c.c_uv_exp import uv_exp_ll, uv_exp_ll_grad
from .uv_exp import UnivariateExpHawkesProcess
from scipy.stats import gamma, beta
from scipy.optimize import minimize

import theano as th
from theano import tensor as tt
import pymc3 as pm


class HPLLOp(th.Op):
    """
    Theano op that implements the log-likelihood function of a univariate
    exponential Hawkes Process
    """
    __props__ = ()

    itypes = [tt.dscalar, tt.dscalar, tt.dscalar]
    otypes = [tt.dscalar]

    def __init__(self, t, T):
        super(HPLLOp, self).__init__()
        self.t = t
        self.T = T

    def perform(self, node, inputs, output_storage):
        x = inputs
        m = output_storage[0]
        ll = uv_exp_ll(self.t, x[0], x[1], x[2], self.T)
        m[0] = np.array(ll)


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

            res[0] += (mu_hyp[0] - 1) / mu - 1. / mu_hyp[1]
            res[1] += (alpha_hyp[0] - 1) / a - (alpha_hyp[1] - 1) / (1 - a)
            res[2] += (theta_hyp[0] - 1) / th - 1. / theta_hyp[1]

            return res

        return f0, g0

    def get_marginal_likelihood(self, t, T = None):
        """calculate marginal likelihood using Laplace's approximation"""
        t, T = self._prep_t_T(t, T)

        f = self._get_log_posterior(t, T)
        g = self._get_log_posterior_grad(t, T)

        xopt = np.array(self.get_params())
        # xopt = self._fit_grad_desc(t, T).x
        H = nd.Jacobian(g)(xopt)

        return f(xopt) + 1.5 * np.log(2 * np.pi) - .5 * np.linalg.slogdet(H)[1]

    def _fit_grad_desc(self, t, T=None):
        """
        Given a bounded finite realization on [0, T], fit parameters with line search (L-BFGS-B).

        :param t: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps. must be
        sorted (asc). dtype must be float.
        :param T: (optional) maximum time. If None, the last occurrence time will be taken.

        :return: the optimization result
        :rtype: scipy.optimize.optimize.OptimizeResult
        """
        t, T = self._prep_t_T(t, T)
        N = len(t)

        ress = []

        f = self._get_log_posterior(t, T)
        g = self._get_log_posterior_grad(t, T)

        best_minres = None
        best_ll = np.inf

        # due to problems in conversion, we run multistart
        for epoch in range(10):
            # draw from the priors for initializing the algorithm
            mu0 = np.random.gamma(self.mu_hyp[0], scale=self.mu_hyp[1])
            th0 = np.random.gamma(self.theta_hyp[0], scale=self.theta_hyp[1])
            a0 = np.random.beta(self.alpha_hyp[0], self.alpha_hyp[1])

            minres = minimize(lambda x: -f(x), x0=np.array([mu0, a0, th0]),
                              jac=lambda x: -g(x),
                              bounds=[(1e-5, None), (1e-5, 1), (1e-5, None)],
                              method="L-BFGS-B", options={"disp": True, "ftol": 1e-8, "gtol": 1e-8})

            ress.append(minres)
            mu, a, _ = minres.x

            print "try %d" % epoch, minres.x, minres.fun

            if minres.fun < best_ll:
                best_ll = minres.fun
                best_minres = minres

        return best_minres

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

    def sample_posterior(self, t, T, n_samp, n_burnin=None):
        """
        Get samples from the posterior via MCMC
        :return:
        """

        t, T = self._prep_t_T(t, T)

        if n_burnin is None:
            n_burnin = int(n_samp / 5)

        with pm.Model() as model:
            mu = pm.Gamma("mu", alpha=self.mu_hyp[0], beta=1. / self.mu_hyp[1])
            theta = pm.Gamma("theta", alpha=self.theta_hyp[0], beta=1. / self.theta_hyp[1])
            alpha = pm.Beta("alpha", alpha=self.alpha_hyp[0], beta=self.alpha_hyp[1])

            op = HPLLOp(t, T)

            def uvexpll(v):
                op(mu, alpha, theta)

            a = pm.Deterministic('a', op(mu, alpha, theta))

            llop = pm.Potential('ll', a)

            trace = pm.sample(n_samp, step=pm.Metropolis(), cores=1, nchains=1)
            burned_trace = trace[n_burnin:]

        return burned_trace

    def marginal_likelihood(self, t, T=None):
        return self.get_marginal_likelihood(t, T)