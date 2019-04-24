import numpy as np
import numdifftools as nd

from hawkeslib.util.multitrace import MultiTrace
from .model import BayesianPointProcessMixin
from .c.c_uv_exp import uv_exp_ll, uv_exp_ll_grad
from .uv_exp import UnivariateExpHawkesProcess
from scipy.stats import gamma, beta
from scipy.optimize import minimize


class BayesianUVExpHawkesProcess(UnivariateExpHawkesProcess, BayesianPointProcessMixin):
    """
    This class inherits from :class:`hawkeslib.UnivariateExpHawkesProcess` and implements a
    "Bayesian" univariate HP model with exponential delay density.

    Specifically the model is determined by the conditional intensity function

    .. math::
        \\lambda^*(t) = \mu + \sum_{t_i < t} \\alpha \\theta \exp( - \\theta (t - t_i)),

    and the prior distributions

    .. math::
        \\begin{align}
            \\mu &\sim \\mathcal{G}(k_\mu, \\eta_\mu), \\\\
            \\theta &\sim \\mathcal{G}(k_\\theta, \\eta_\\theta), \\\\
            \\alpha &\sim \\mathcal{B}(a, b).
        \\end{align}

    Here, :math:`\\mathcal{G}` denotes the Gamma distribution in its "shape-scale" parameterization.
    :math:`\\mathcal{B}` denotes the Beta distribution. See :class:`hawkeslib.UnivariateExpHawkesProcess`
    and the tutorial for further details on the model and parameters.

    The hyperparameters to the appropriate Gamma and Beta priors are given during initialization as ``mu_hyp``,
    ``theta_hyp`` and ``alpha_hyp``, and they are not fitted (e.g. through empirical Bayes).

    This class implements methods for sampling from the posterior (i.e. sampling parameters, not observations
    from the posterior predictive --for now), calculating marginal likelihood (e.g., for computing Bayesian
    hypothesis tests), and fitting maximum a posteriori (MAP) estimates.
    """

    def __init__(self, mu_hyp, alpha_hyp, theta_hyp):
        """
        Initialize a Bayesian univariate HP model

        :param tuple[float,float] mu_hyp: hyperparameters for the prior for mu. (k, eta) for the
            shape-scale parameterization of the Gamma distribution
        :param tuple[float,float] alpha_hyp: hyperparameters for the Beta prior for alpha. (a, b)
        :param tuple[float,float] theta_hyp: hyperparameters for the prior for theta. (k, eta) for the shape-scale
            parameterization of the Gamma distribution
        """
        super(BayesianUVExpHawkesProcess, self).__init__()

        self.mu_hyp = mu_hyp
        self.alpha_hyp = alpha_hyp
        self.theta_hyp = theta_hyp

        self._log_posterior = lambda t, T: self._get_log_posterior_pot_grad_fns(t, T, mu_hyp, alpha_hyp, theta_hyp)[0]
        self._log_posterior_grad = lambda t, T: self._get_log_posterior_pot_grad_fns(t, T, mu_hyp, alpha_hyp, theta_hyp)[1]

    @classmethod
    def _get_log_posterior_pot_grad_fns(cls, t, T, mu_hyp, alpha_hyp, theta_hyp):
        """
        Get the log (unnormalized) posterior (and gradient) as functions with
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

    def _fit_grad_desc(self, t, T=None, nr_restarts=5):
        """
        Given a bounded finite realization on [0, T], fit parameters with line search (L-BFGS-B). The procedure
        is usually unstable, which is why we use multiple random restarts. Each restart is a draw from the
        prior.

        :param t: Bounded finite sample of the process up to time T. 1-d ndarray of timestamps. must be
            sorted (asc). dtype must be float.
        :param T: (optional) maximum time. If None, the last occurrence time will be taken.
        :param nr_restarts: int, number of random restarts

        :return: the optimization result
        :rtype: scipy.optimize.optimize.OptimizeResult
        """
        t, T = self._prep_t_T(t, T)
        N = len(t)

        ress = []

        f = self._log_posterior(t, T)
        g = self._log_posterior_grad(t, T)

        best_minres = None
        best_ll = np.inf

        # due to problems in conversion, we run multistart
        for epoch in range(nr_restarts):
            # draw from the priors for initializing the algorithm
            mu0 = np.random.gamma(self.mu_hyp[0], scale=self.mu_hyp[1])
            th0 = np.random.gamma(self.theta_hyp[0], scale=self.theta_hyp[1])
            a0 = np.random.beta(self.alpha_hyp[0], self.alpha_hyp[1])

            minres = minimize(lambda x: -f(x), x0=np.array([mu0, a0, th0]),
                              jac=lambda x: -g(x),
                              bounds=[(1e-5, None), (1e-5, 1), (1e-5, None)],
                              method="L-BFGS-B", options={"disp": False, "ftol": 1e-8, "gtol": 1e-8})

            ress.append(minres)
            mu, a, _ = minres.x

            if minres.fun < best_ll:
                best_ll = minres.fun
                best_minres = minres

        return best_minres

    def marginal_likelihood(self, t, T = None):
        """
        Calculate log marginal likelihood of the process under given data, using Laplace's approximation.
        This method uses a Gaussian approximation around the currently fit parameters (i.e. expects MAP
        parameters to already have been fit, e.g. through :meth:`fit`).

        :param numpy.array[float] t: Observation timestamps of the process up to time T. 1-d array of timestamps.
            must be sorted (asc)
        :param T: (optional) maximum time
        :type T: float or None

        :return: the log marginal likelihood
        :rtype: float
        """
        t, T = self._prep_t_T(t, T)

        f = self._log_posterior(t, T)
        g = self._log_posterior_grad(t, T)

        xopt = np.array(self.get_params())
        # xopt = self._fit_grad_desc(t, T).x
        H = nd.Jacobian(g)(xopt)

        return f(xopt) + 1.5 * np.log(2 * np.pi) - .5 * np.linalg.slogdet(H)[1]

    def log_posterior_with_params(self, t, mu, alpha, theta, T=None):
        """
        Evaluate the log potential (unnormalized posterior) of the process parameters ``mu``, ``alpha``,
        ``theta``, under data ``t``, ``T``.

        :param numpy.array[float] t: Observation timestamps of the process up to time T. 1-d array of timestamps.
            must be sorted (asc)
        :param float mu: the exogenous intensity
        :param float alpha: the infectivity factor alpha
        :param float theta: intensity parameter of the delay density
        :param T: (optional) maximum time
        :type T: float or None

        :rtype: float
        :return: the log unnormalized posterior (log potential)
        """
        t, T = self._prep_t_T(t, T)
        return self._log_posterior(t, T)([mu, alpha, theta])

    def log_posterior(self, t, T=None):
        """
        Get the log unnormalized posterior for parameters already fit, under observed timestamps ``t``.

        :param numpy.array[float] t: Observation timestamps of the process up to time T. 1-d array of timestamps.
            must be sorted (asc)
        :param T: (optional) maximum time
        :type T: float or None

        :rtype: float
        :return: the log unnormalized posterior (log potential)
        """
        t, T = self._prep_t_T(t, T)
        mu, alpha, theta = self.get_params()
        return self._log_posterior(t, T)([mu, alpha, theta])

    def fit(self, t, T=None, **kwargs):
        """
        Get the maximum a posteriori (MAP) estimate via gradient descent, and store it in the
        ``BayesianUVExpHawkesProcess`` object.

        This function uses ``scipy``'s L-BFGS-B routine to fit the parameters. It also takes an optional
        keyword argument ``nr_restarts`` that sets the number of random restarts for the multistart gradient
        descent algorithm.

        :param numpy.array[float] t: Observation timestamps of the process up to time T. 1-d array of timestamps.
            must be sorted (asc)
        :param T: (optional) maximum time
        :type T: float or None
        :param kwargs: see below.
        :return: the resulting log unnormalized posterior (log potential) --for the parameters fit
        :rtype: float

        **Keyword Arguments**

        * *nr_restarts* (``int``) -- Optional, the number of random restarts of the GD algorithm.
          The best log-posterior fit will be returned as the result. Defaults to 5.

        """
        t, T = self._prep_t_T(t, T)

        minres = self._fit_grad_desc(t, T, nr_restarts=kwargs.get("nr_restarts", 5))
        params = minres.x

        lp = self.log_posterior_with_params(t, params[0], params[1], params[2], T)

        self.set_params(*params)

        return lp

    def sample_posterior(self, t, T, n_samp, n_burnin=None, rwm_sigma=0.2):
        """
        Get samples from the posterior, e.g. for posterior inference or computing Bayesian credible intervals.
        This routine samples via the random walk Metropolis (RWM) algorithm.

        The function returns a ``MultiTrace`` object that can be operated on simply like a ``numpy.array``.

        :param numpy.array[float] t: Observation timestamps of the process up to time T. 1-d array of timestamps.
            must be sorted (asc)
        :param T: (optional) maximum time
        :type T: float or None
        :param int n_samp: number of posterior samples to take
        :param int n_burnin: number of samples to discard (as the burn-in samples)
        :param float rwm_sigma: the standard deviation for the proposal of the Random Walk Metropolis algorithm

        :rtype: hawkeslib.util.MultiTrace
        :return: the posterior samples for mu, alpha and theta as a trace object
        """

        t, T = self._prep_t_T(t, T)

        if n_burnin is None:
            n_burnin = int(n_samp / 5)

        samples = np.empty((n_samp, 3))

        log_u = np.log(np.random.rand(n_samp))

        def log_post(par):
            return self.log_posterior_with_params(
                t, par[0], par[1], par[2], T
            )

        x = np.array([
            self.mu_hyp[0],
            self.alpha_hyp[0],
            self.theta_hyp[0]
        ])

        x += np.random.randn(3) * rwm_sigma
        curr_log_post = log_post(x)

        for i in range(n_samp):
            x_n = x + np.random.randn(3) * rwm_sigma

            prop_log_post = log_post(x_n)
            post_diff = prop_log_post - curr_log_post

            if post_diff > 0 or post_diff > log_u[i]:
                x = x_n
                curr_log_post = prop_log_post

            samples[i] = x

        result_mtrace = MultiTrace(
            ["mu", "alpha", "theta"],
            *[samples[:, j] for j in range(3)]
        )

        return result_mtrace[n_burnin:]
