"""
Univariate (K=1) Hawkes model with a single exponential delay density.
"""
import numpy as np
from hawkeslib.model.model import PointProcess
from hawkeslib.model.c.c_uv_exp import (
    uv_exp_ll,
    uv_exp_ll_grad,
    uv_exp_sample_ogata,
    uv_exp_sample_branching,
    uv_exp_fit_em_base,
    uv_exp_phi
)
from scipy.optimize import minimize


class UnivariateExpHawkesProcess(PointProcess):
    """
    Univariate Hawkes process (self-exciting process) with the exponential conditional intensity
    function

    .. math::
        \lambda^*(t) = \mu + \\alpha \\theta \sum_{t_i < t} \exp(-\\theta (t - t_i)),

    where :math:`\mu` (``mu``) is a constant background (exogenous) intensity, :math:`\\alpha` (``alpha``)
    is the "infectivity factor"
    which governs how many (in expectation) further events are caused by a given event, and :math:`\\theta` (``theta``)
    is the
    rate parameter for the "exponential delay" -- the probability distribution of time between events that have been
    caused by one another.

    This class inherits from :class:`hawkeslib.model.PointProcess` and implements several methods that are required
    for evaluating the likelihood, taking forward samples, and fitting parameters for such Hawkes processes.

    For parameter fitting, the :meth:`fit` method implements both a gradient descent algorithm benefiting from
    ``scipy``'s L-BFGS-B implementation, and an Expectation-Maximization algorithm. For sampling, both Ogata's modified
    "thinning" algorithm and a "branching" sampler are made available.

    See the Hawkes process tutorial for further details and references.
    """

    _mu = None
    _alpha = None
    _theta = None

    def __init__(self):
        pass

    @classmethod
    def log_likelihood_with_params(cls, t, mu, alpha, theta, T=None):
        """
        Calculate the log likelihood of parameters, given process realization ``t``.

        :param numpy.array[float] t: Observed event times of the process up to time ``T``. 1-d array of timestamps.
            Event times must be sorted (asc).
        :param float mu: the exogenous intensity
        :param float alpha: the infectivity factor alpha
        :param float theta: intensity parameter of the delay density
        :param T: (optional) the upper bound of the observation period. If not provided, it is taken
            as the maximum timestamp in ``t``.
        :type T: float or None

        :return: the log likelihood
        :rtype: float
        """
        assert alpha < 1, "Not stationary!"
        if T is None:
            T = t[-1]

        cls._assert_good_t_T(t, T)
        return uv_exp_ll(t, mu, alpha, theta, T)

    def _fetch_params(self):
        """
        Get the parameters currently in the object.
        :return: 3-tuple, (mu, alpha, theta)
        """
        pars = self._mu, self._alpha, self._theta
        assert None not in pars, "Some parameters seem to be missing. Did you fit() already?"
        return pars

    def set_params(self, mu, alpha, theta, check_stationary=True):
        """
        Manually set the process parameters.

        :param float mu: the exogenous intensity
        :param float alpha: the infectivity factor alpha
        :param float theta: intensity parameter of the delay density
        """
        if check_stationary:
            assert alpha < 1, "Not stationary!"
        assert np.min([mu, alpha, theta]) > 0, "Parameters must be greater than zero!"

        self._mu, self._alpha, self._theta = mu, alpha, theta

    def get_params(self):
        """
        Get the parameters of the process. The process must have been fit before, or parameters set
        through :meth:`set_params`.

        :return: (mu, alpha, theta)
        :rtype: tuple[float, float, float]
        """
        return self._fetch_params()

    def sample(self, T, method="ogata"):
        """
        Take an (unconditional) sample from the process using Ogata's modified thinning method or
        a "branching" sampler exploiting the Poisson cluster structure of HP.

        Parameters must be set, either by fitting or through :meth:`set_params`.

        :param float T: maximum time (samples in :math:`[0, T)`)
        :param str method: either ``"ogata"`` or ``"branching"``

        :return: sampled timestamps
        :rtype: numpy.array[float]
        """
        mu, alpha, theta = self._fetch_params()
        if method == "branching":
            return uv_exp_sample_branching(T, mu, alpha, theta)
        return uv_exp_sample_ogata(T, mu, alpha, theta)

    def conditional_sample(self, T, tcond, Tcond=None):
        """
        Take a sample from a fitted model, conditioning on a previous interval
        to compute the last state of the process.

        :param T: maximum time (samples in :math:`[0, T)`)
        :param tcond: timestamps of the conditioning interval
        :param Tcond: length of the conditioning interval

        :return: sampled timestamps
        :rtype: numpy.array[float]
        """

        if Tcond is None:
            Tcond = tcond[-1]

        mu, alpha, theta = self._fetch_params()

        phi = uv_exp_phi(tcond, theta, Tcond)

        return uv_exp_sample_ogata(T, mu, alpha, theta, phi=phi)

    def log_likelihood(self, t, T=None):
        """
        Get the log likelihood of parameters currently set in the process (either through :meth:`fit` or
        :meth:`set_params`), given a observations ``t`` in :math:`[0, T)`.

        :param numpy.array[float] t: Occurrence times of the observed process up to time T. 1-d ndarray
            of timestamps. must be sorted (asc)
        :param T: (optional) the upper bound of the observation period. If not provided, it is taken
            as the maximum timestamp in ``t``.
        :type T: float or None

        :return: the log likelihood
        :rtype: float
        """

        m, a, th = self._fetch_params()
        if T is None:
            T = t[-1]
        self._assert_good_t_T(t, T)

        return uv_exp_ll(t, m, a, th, T)

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

        # due to a convergence problem, we reiterate until the unconditional mean starts making sense
        for epoch in range(5):
            # estimate starting mu via the unconditional sample formula assuming
            # $\alpha \approx 0.2$
            mu0 = N * 0.8 / T

            # initialize other parameters randomly
            a0, th0 = np.random.rand(2)
            # todo: initialize th0 better ?

            minres = minimize(lambda x: -uv_exp_ll(t, x[0], x[1], x[2], T),
                              x0=np.array([mu0, a0, th0]),
                              jac=lambda x: -uv_exp_ll_grad(t, x[0], x[1], x[2], T),
                              bounds=[(1e-5, None), (1e-5, 1), (1e-5, None)],
                              method="L-BFGS-B", options={"disp": False, "ftol": 1e-10, "gtol": 1e-8})

            ress.append(minres)
            mu, a, _ = minres.x

            # take the unconditional mean and see if it makes sense
            Napprox = mu * T / (1 - a)
            if abs(Napprox - N)/N < .01:  # if the approximation error is in range, break
                break

        return minres

    def fit(self, t, T=None, method="em", **kwargs):
        """
        Fit parameters of the process, using one of gradient descent or expectation-maximization algorithms
        obtaining maximum likelihood estimates of parameters ``mu``, ``alpha``, and ``theta`` and storing
        them in the object.

        :param numpy.array[float] t: Occurrence times of the observed process up to time T. 1-d ndarray
            of timestamps. must be sorted (asc)
        :param T: (optional) the upper bound of the observation period. If not provided, it is taken
            as the maximum timestamp in ``t``.
        :type T: float or None
        :param str method: specifies which method to use. one of ``"em"`` or ``"gd"``, ``"em"`` by default.
        :param kwargs: specifies options for the EM algorithm. see below.

        :return: likelihood of the process under the fit parameters
        :rtype: float

        :Keyword Arguments:

        * *reltol* (``float``) --
          **For EM only!** -- The relative log likelihood improvement used as a stopping condition.
          Defaults to ``1e-5``.
        * *maxiter* (``int``) -- **For EM only!** --
          The maximum number of iterations. Defaults to 500.
        """

        if T is None:
            T = t[-1]

        self._assert_good_t_T(t, T)

        if method == "em":  # expectation-maximization
            emkwargs = {k: v for k, v in kwargs.items() if k in ["maxiter", "reltol"]}

            ll, params, _ = uv_exp_fit_em_base(t, T, **emkwargs)

        elif method == "gd":  # gradient descent
            minres = self._fit_grad_desc(t, T)

            params = minres.x
            ll = self.log_likelihood_with_params(t, params[0], params[1], params[2])

        else:
            raise ValueError("method must be one of `gd` or `em`")

        self.set_params(*params, check_stationary=False)

        return ll
