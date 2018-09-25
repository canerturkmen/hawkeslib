import unittest as ut
import mock
import numpy as np

from ..model.c import c_mv_beta, c_mv_samp
from scipy.stats import beta


class MVBetaLLTests(ut.TestCase):

    def test_ll_correct_shortrange(self):
        arr = np.array([1., 2., 2.5, 3., 6., 7.])

        T = 8.
        tmax = 0.9
        A = np.eye(1) * .2
        th1, th2 = .2, .2
        mu = np.array([.5])

        comp = np.sum(mu * T)
        for t in arr:
            if T - t < tmax:
                comp += A[0,0] * beta.cdf((T - t) / tmax, th1, th2)
            else:
                comp += A[0,0]

        lJ = np.log(mu[0]) * 4
        lJ += np.log(mu[0] + A[0,0] * beta.pdf(.5/.9, th1, th2) / .9) * 2

        ll_base = lJ - comp
        ll_computed = c_mv_beta.mv_beta_ll(arr, np.zeros(len(arr), dtype=int),
                                           mu, A, th1, th2, tmax, T)
        self.assertAlmostEqual(ll_base, ll_computed)

    def test_ll_correct_longrange(self):
        arr = np.array([1., 2., 2.5, 3., 6., 7.])

        T = 8.
        tmax = 100.
        A = np.eye(1) * .2
        th1, th2 = .2, .3
        mu = np.array([.5])

        lJ = 0.
        comp = np.sum(mu * T)
        for i, t in enumerate(arr):
            comp += A[0,0] * beta.cdf((T - t) / tmax, th1, th2)
            lda = mu[0]
            for tj in arr[:i]:
                lda += A[0, 0] * beta.pdf((t - tj)/ tmax, th1, th2) / tmax
            lJ += np.log(lda)

        ll_base = lJ - comp
        ll_computed = c_mv_beta.mv_beta_ll(arr, np.zeros(len(arr), dtype=int),
                                           mu, A, th1, th2, tmax, T)
        self.assertAlmostEqual(ll_base, ll_computed)


class GenericSamplerTests(ut.TestCase):
    # for the Beta delay case

    def test_first_moments_close(self):
        A = np.eye(3) * .2
        mu = np.array([.6, 1.5, .2])
        T = 2000.

        t, c = c_mv_samp.mv_sample_branching(T, mu, A, .3, 23., 5., 0, 1)

        np.testing.assert_allclose(np.bincount(c) / T, mu / .8, rtol=.2)

