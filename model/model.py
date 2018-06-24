import abc
import numpy as np


class PointProcess(object):
    """
    Abstract class that defines common interface for all point process
    implementations.
    """

    params = None

    @classmethod
    def _assert_good_t_T(cls, t, T):
        if not np.all(t >= 0):
            raise ValueError("The array t cannot contain negative time values")
        if not np.all(np.diff(t) >= 0):
            raise ValueError("The array t must be in sorted order")
        if not np.all(t <= T):
            raise ValueError("The maximum time T must be greater than all values in array t")

    @abc.abstractmethod
    def sample(self, T):
        """
        Generate unconditional forward samples from the point process

        :returns: numpy arrays for times (t_n) and marks (c_n)
        :rtype: numpy.array, numpy.array
        """
        pass

    @abc.abstractmethod
    def conditional_sample(self, t_n, c_n, T):
        """
        Given a series of observations (t_n, c_n), draw a forward sample from the end of the
        series for the next T time steps.

        :returns: numpy arrays for times (t_n) and marks (c_n)
        :rtype: numpy.array, numpy.array
        """
        pass

    @abc.abstractmethod
    def log_likelihood(self, *args):
        """
        Calculate log likelihood function under the model for the given
        :param t_n:
        :param c_n:
        :return: log likelihood
        :rtype: float
        """
        pass

    @abc.abstractmethod
    def fit(self, *args):
        """

        """
        pass

    def perplexity(self, t_n, c_n):
        ll = self.log_likelihood(t_n, c_n)
        return np.exp(-ll / len(t_n))