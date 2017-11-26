"""
Univariate (K=1) Hawkes model with a single exponential delay density.
"""
import numpy as np
from .model import PointProcess

class UnivariateExpHawkesProcess(PointProcess):

    _alpha = None
    _beta = None
    _lda0 = None

    def __init__(self):
        pass

    def set_params(self, alpha, beta, lda0):
        """
        Manually set the parameters of the Hawkes process.

        :param alpha:
        :param beta:
        :param lda0:
        :return:
        """
        assert alpha < beta, "Not stationary!"
        assert np.min([lda0, alpha, beta]) > 0, "Parameters cannot be below zero!"

        self._alpha, self._beta, self._lda0 = alpha, beta, lda0

    def get_params(self):
        return self._alpha, self._beta, self._lda0

    def sample(self, T):
        _a, _b, _l0 = self.get_params()

        assert None not in (_a, _b, _l0), "Some parameters seem to be missing. Did you fit() already?"

        arr = []
        s, n = 0, 0

        while s < T:
            lda_bar = _l0 + np.sum(_a * np.exp(- _b * (s - np.array(arr))))

            u = np.random.rand()
            w = - np.log(u) / lda_bar
            s += w

            D = np.random.rand()
            if D * lda_bar <= _l0 + np.sum(_a * np.exp(- _b * (s - np.array(arr)))):
                n += 1
                arr.append(s)

        if arr[-1] > T:
            arr = arr[:-1]

        return arr