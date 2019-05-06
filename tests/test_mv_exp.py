from __future__ import division

import os
import unittest as ut
import mock
import numpy as np

from hawkeslib.model.mv_exp import MultivariateExpHawkesProcess
from hawkeslib.model.c import c_mv_exp, c_uv_exp
from hawkeslib import UnivariateExpHawkesProcess

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

    def test_mv_ll_matches_naive(self):
        N = 100
        fpath = os.path.join(os.path.dirname(__file__), 'tfx_fixture.npy')
        t = np.load(fpath)[:N]
        c = np.random.choice([0,1,2], size=N)

        K = 3
        mu = np.random.rand(K)
        A = np.random.rand(K, K) * .05 + np.eye(K) * .3
        theta = 1.

        T = t[-1]

        # compute log likelihood naively
        ll = -np.sum(mu) * T
        F = np.zeros(K)
        for i in range(N):
            z = mu[c[i]]
            for j in range(i):
                z += A[c[j], c[i]] * theta * np.exp(-theta * (t[i] - t[j]))
            ll += np.log(z)
            F[c[i]] += 1 - np.exp(-theta * (T - t[i]))
        ll -= A.T.dot(F).sum()

        ll_lib = c_mv_exp.mv_exp_ll(t, c, mu, A, theta, T)

        self.assertAlmostEqual(ll, ll_lib, places=4)

    def test_mv_empty_array_ok(self):
        arr = np.array([])

        t = np.array([])
        c = np.array([], dtype=np.long)

        K = 3
        mu = np.array([2., 3., 5.])
        A = np.eye(3) * 0.5
        theta = 1.

        T = 1.

        tgt = -10.
        computed = c_mv_exp.mv_exp_ll(t, c, mu, A, theta,T)

        self.assertAlmostEqual(tgt, computed)

class MVExpBranchingSamplerTests(ut.TestCase):

    def setUp(self):
        pass

    def test_mv_sample_diag_kernel_numbers_ok(self):
        K, T = 3, 10000
        mu = np.array([.5, .3, .2])
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
        mu = np.array([.5, .3, .2])
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
        mu = np.array([.5])
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
        mu = np.array([.5, .3])
        A = np.eye(K) * .3
        theta = 1.

        tres, cres = c_mv_exp.mv_exp_sample_branching(T, mu, A, theta)
        Endt = np.bincount(cres)[0]

        uv = UnivariateExpHawkesProcess()
        uv.set_params(mu[0], .3, 1.)
        Rndt = len(uv.sample(T))

        devi = np.abs(Endt - Rndt) / Rndt

        self.assertLessEqual(devi, .1)


class MVEMAlgorithmTests(ut.TestCase):

    def setUp(self):

        self.t = np.load(os.path.join(os.path.dirname(__file__), 'tfx_mvt.npy'))
        self.c = np.load(os.path.join(os.path.dirname(__file__), 'tfx_mvc.npy'))
        self.T = self.t[-1]

    def test_em_runs_no_convergence_issue(self):

        try:
            c_mv_exp.mv_exp_fit_em(self.t, self.c, self.T, maxiter=100)
        except Exception as e:
            self.fail(e)

    def test_em_params_close(self):

        _, p, _ = c_mv_exp.mv_exp_fit_em(self.t, self.c, self.T, maxiter=200, reltol=1e-6)

        assert np.allclose(np.array([.2, .6]), p[0], rtol=0.2), p[0]
        assert np.allclose(np.eye(2) * .4 + np.ones((2,2)) * .1, p[1], rtol=0.2)
        self.assertAlmostEqual(1., p[2], delta=.2)

    def test_em_close_to_uv(self):

        c = np.zeros(len(self.t), dtype=int)

        _, p, _ = c_mv_exp.mv_exp_fit_em(self.t, c, self.T, maxiter=200, reltol=1e-6)
        _, pu, _ = c_uv_exp.uv_exp_fit_em_base(self.t, self.T, maxiter=200, reltol=1e-6)

        self.assertAlmostEqual(p[0][0], pu[0], delta=.05)
        self.assertAlmostEqual(p[1][0][0], pu[1], delta=.05)
        self.assertAlmostEqual(pu[2], p[2], delta=.05)

    def test_em_truefx_does_not_fail(self):

        t = np.load(os.path.join(os.path.dirname(__file__), 'tfx_truefx_t.npy'))[:1000]
        c = np.load(os.path.join(os.path.dirname(__file__), 'tfx_truefx_c.npy'))[:1000]
        
        try:
            _, p, _ = c_mv_exp.mv_exp_fit_em(t, c, t[-1], maxiter=200, reltol=1e-6)
        except Exception as e:
            self.fail(e)

class MVExpClassTests(ut.TestCase):
    """tests the mv exp hp python interface"""

    def setUp(self):

        self.p = MultivariateExpHawkesProcess()
        self.t = np.load(os.path.join(os.path.dirname(__file__), 'tfx_mvt.npy'))
        self.c = np.load(os.path.join(os.path.dirname(__file__), 'tfx_mvc.npy'))
        self.T = self.t[-1]

    def test_llwp_wrong_mu_dim_rejected(self):
        with self.assertRaises(ValueError):
            self.p.log_likelihood_with_params(self.t, self.c, np.array([.2, .3, .4]),
                                              np.eye(2) * .2, 1.)

    def test_llwp_wrong_A_dim_rejected(self):
        with self.assertRaises(ValueError):
            self.p.log_likelihood_with_params(self.t, self.c, np.array([.2, .3]),
                                              np.eye(3) * .2, 1.)

    def test_llwp_nonstationary_A_rejected(self):
        with self.assertRaises(ValueError):
            self.p.log_likelihood_with_params(self.t, self.c, np.array([.2, .3]),
                                              np.eye(2) * 1.2, 1.)

    def test_setpars_nonstationary_A_rejected(self):
        with self.assertRaises(ValueError):
            self.p.set_params(np.array([.2, .3]), np.eye(2) * 1.2, 1.)

    def test_setpars_mismatch_A_mu_rejected(self):
        with self.assertRaises(ValueError):
            self.p.set_params(np.array([.2, .3]), np.eye(3) * .2, 1.)

    # noinspection PyTypeChecker
    def test_llwp_nonfloat_theta_rejected(self):
        with self.assertRaises(ValueError):
            self.p.log_likelihood_with_params(self.t, self.c, np.array([.2, .3]),
                                              np.eye(2) * .2, np.array([1., 1.]))

    @mock.patch('hawkeslib.model.mv_exp.mv_exp_ll')
    def test_llwp_nonlong_c_cast(self, m):
        c = self.c.astype(float)
        self.p.log_likelihood_with_params(self.t, c, np.array([.2, .3]),
                                              np.eye(2) * .2, 1.)
        assert m.call_args[0][1].dtype == np.int64

    @mock.patch('hawkeslib.model.mv_exp.mv_exp_fit_em')
    def test_fit_nonlong_c_cast(self, m):
        mu, A, theta = np.array([.2, .3]), np.eye(2) * .2, 1.
        m.return_value = (1., (mu, A, theta), 10)

        c = self.c.astype(float)
        self.p.fit(self.t, c)
        assert m.call_args[0][1].dtype == np.int64

    def test_ll_mismatch_t_c_len_rejected(self):
        c = self.c[:-2]
        self.p.set_params(np.array([.2, .3]), np.eye(2)*.2, 1.)
        with self.assertRaises(ValueError):
            self.p.log_likelihood(self.t, c)

    def test_llwp_mismatch_t_c_len_rejected(self):
        c = self.c[:-2]
        with self.assertRaises(ValueError):
            self.p.log_likelihood_with_params(self.t, c, np.array([.2, .3]),
                                              np.eye(2) * .2, 1.)

    def test_fit_mismatch_t_c_len_rejected(self):
        c = self.c[:-2]
        with self.assertRaises(ValueError):
            self.p.fit(self.t, c)

    @mock.patch('hawkeslib.model.mv_exp.mv_exp_ll')
    def test_log_likelihood_call_correct(self, m):
        mu, A, theta = np.array([.2, .3]), np.eye(2) * .2, 1.
        self.p.set_params(mu, A, theta)
        self.p.log_likelihood(self.t, self.c)
        m.assert_called_once()

    @mock.patch('hawkeslib.model.mv_exp.mv_exp_ll')
    def test_log_likelihood_with_params_call_correct(self, m):
        mu, A, theta = np.array([.2, .3]), np.eye(2) * .2, 1.
        self.p.log_likelihood_with_params(self.t, self.c, mu, A, theta)
        m.assert_called_once()

    @mock.patch('hawkeslib.model.mv_exp.mv_exp_sample_branching')
    def test_sample_call_correct(self, m):
        mu, A, theta = np.array([.2, .3]), np.eye(2) * .2, 1.
        self.p.set_params(mu, A, theta)
        self.p.sample(1000)
        m.assert_called_once()

    @mock.patch('hawkeslib.model.mv_exp.mv_exp_fit_em')
    def test_fit_call_correct(self, m):
        mu, A, theta = np.array([.2, .3]), np.eye(2) * .2, 1.
        m.return_value = (1., (mu, A, theta), 10)

        self.p.fit(self.t, self.c)
        m.assert_called_once()

    @mock.patch('hawkeslib.model.mv_exp.mv_exp_fit_em')
    def test_fit_call_correct_kwarg(self, m):
        mu, A, theta = np.array([.2, .3]), np.eye(2) * .2, 1.
        m.return_value = (1., (mu, A, theta), 10)

        self.p.fit(self.t, self.c, maxiter=100, reltol=1e-6)
        m.assert_called_once()
