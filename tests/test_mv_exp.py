import os
import unittest as ut
import numpy as np

from ..model.c import c_mv_exp
from .. import UnivariateExpHawkesProcess


class MVExpLikelihoodTests(ut.TestCase):

    def test_degenerate_mv_ll_matches_uv_ll(self):

        fpath = os.path.join(os.path.dirname(__file__), 'tfx_fixture.npy')
        arr = np.load(fpath)

        uv = UnivariateExpHawkesProcess()
        uv.set_params(5, .2, 10.)
        uvll = uv.log_likelihood(arr, arr[-1])

        mvll = c_mv_exp.mv_exp_ll(arr, np.zeros(len(arr), dtype=np.int),
                                  np.ones(1) * 5,
                                  np.ones((1, 1)) * .2, 10., arr[-1])

        self.assertAlmostEqual(uvll, mvll, places=3)


class MVExpBranchingSamplerTests(ut.TestCase):

    def setUp(self):
        pass

    def test_mv_sample_diag_kernel_numbers_ok(self):
        K, T = 3, 10000
        mu = np.random.rand(K)
        A = np.eye(K) * .2
        # A[2, 0] = .5
        theta = 1.

        tres, cres = c_mv_exp.mv_exp_sample_branching(T, mu, A, theta)

        # calculate expectation
        I_K = np.eye(A.T.shape[0])
        Endt = np.linalg.pinv(I_K - A.T).dot(mu) * T  # expected number of each mark
        Rndt = np.bincount(cres)  # true outcomes

        devi = np.abs(Endt - Rndt) / Rndt

        self.assertLessEqual(max(devi), .1)  # max deviation should be less than 0.1

    def test_mv_sample_full_kernel_numbers_ok(self):
        K, T = 3, 10000
        mu = np.random.rand(K)
        A = np.eye(K) * .2
        A[2, 0] = .5
        A[0, 1] = .3

        theta = 1.

        tres, cres = c_mv_exp.mv_exp_sample_branching(T, mu, A, theta)

        # calculate expectation
        I_K = np.eye(A.T.shape[0])
        Endt = np.linalg.pinv(I_K - A.T).dot(mu) * T  # expected number of each mark
        Rndt = np.bincount(cres)  # true outcomes

        devi = np.abs(Endt - Rndt) / Rndt

        self.assertLessEqual(max(devi), .1)  # max deviation should be less than 0.1

    def test_uv_sample_matches_numbers(self):
        K, T = 1, 10000
        mu = np.random.rand(K)
        A = np.eye(K) * .2
        theta = 1.

        tres, cres = c_mv_exp.mv_exp_sample_branching(T, mu, A, theta)
        Endt = np.bincount(cres)[0]

        uv = UnivariateExpHawkesProcess()
        uv.set_params(mu[0], .2, 1.)
        Rndt = len(uv.sample(T))

        devi = np.abs(Endt - Rndt) / Rndt

        self.assertLessEqual(devi, .1)


    def test_mv_sample_diag_kernel_uv_matches_numbers(self):
        K, T = 2, 10000
        mu = np.random.rand(K)
        A = np.eye(K) * .3
        theta = 1.

        tres, cres = c_mv_exp.mv_exp_sample_branching(T, mu, A, theta)
        Endt = np.bincount(cres)[0]

        uv = UnivariateExpHawkesProcess()
        uv.set_params(mu[0], .2, 1.)
        Rndt = len(uv.sample(T))

        devi = np.abs(Endt - Rndt) / Rndt

        self.assertLessEqual(devi, .1)
