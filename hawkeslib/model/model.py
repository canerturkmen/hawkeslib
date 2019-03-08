import abc
import numpy as np


class PointProcess(object):
    """
    Defines common interface for all point process implementations.
    """

    @classmethod
    def _assert_good_t_T(cls, t, T):
        if not np.all(t >= 0):
            raise ValueError("The array t cannot contain negative time values")
        if not np.all(np.diff(t) >= 0):
            raise ValueError("The array t must be in sorted order")
        if not np.all(t <= T):
            raise ValueError("The maximum time T must be greater than all values in array t")

    @classmethod
    def _prep_t_T(cls, t, T):
        if T is None:
            T = t[-1]
        t = np.array(t)

        cls._assert_good_t_T(t, T)

        return t, T

    @classmethod
    @abc.abstractmethod
    def log_likelihood_with_params(cls, *args):
        """
        Calculate log likelihood function under the model for the given parameters
        """
        pass

    @abc.abstractmethod
    def sample(self, T):
        """
        Generate unconditional forward samples from the point process

        :returns: samples
        :rtype: numpy.array
        """
        pass

    @abc.abstractmethod
    def conditional_sample(self, T, t, Tcond=None):
        pass

    @abc.abstractmethod
    def log_likelihood(self, *args):
        """
        Calculate log likelihood function under the model fit
        """
        pass

    @abc.abstractmethod
    def fit(self, *args):
        pass


class BayesianPointProcessMixin:

    @classmethod
    @abc.abstractmethod
    def log_posterior_with_params(self, *args):
        pass

    @abc.abstractmethod
    def log_posterior(self, t, T=None):
        pass

    @abc.abstractmethod
    def marginal_likelihood(self, t, T=None):
        pass

    @abc.abstractmethod
    def sample_posterior(self, t, T, n_samp, n_burnin=None):
        pass


