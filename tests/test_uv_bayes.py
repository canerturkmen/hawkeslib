import os
import random

import numdifftools as nd
import numpy as np
import unittest as ut

from scipy.stats import beta, gamma

from hawkeslib.model.c import c_uv_exp
from hawkeslib.model.uv_exp import UnivariateExpHawkesProcess
from hawkeslib.model.uv_bayes import BayesianUVExpHawkesProcess
from hawkeslib.model.c import c_uv_bayes


class TestBayesHelpers(ut.TestCase):

    def test_gamma_prior(self):
        k, scale = 5., 7.
        sp = gamma.logpdf(6., k, scale=scale)
        f0 = c_uv_bayes.cmake_gamma_logpdf(k, scale)

        self.assertAlmostEqual(sp, f0(6.), places=4)

    def test_beta_prior(self):
        a, b = 3., 3.
        sp = beta.logpdf(.45, a, b)
        f0 = c_uv_bayes.cmake_beta_logpdf(a, b)

        self.assertAlmostEqual(sp, f0(.45), places=4)


class TestUVExpBayesLogPosterior(ut.TestCase):

    def setUp(self):
        fpath = os.path.join(os.path.dirname(__file__), 'tfx_fixture.npy')
        self.arr = np.load(fpath)  # test fixture

        self.bhp = BayesianUVExpHawkesProcess((1., 5.), (1, 1), (1., 5.))

    def test_dummy_posterior_correct(self):
        A = self.arr
        logpost = self.bhp.log_posterior_with_params(A, 5., .2, .1, A[-1])

        check = self.bhp.log_likelihood_with_params(A, 5, .2, .1, A[-1]) + \
                gamma.logpdf(5, self.bhp.mu_hyp[0], scale=self.bhp.mu_hyp[1]) + \
                gamma.logpdf(.1, self.bhp.theta_hyp[0], scale=self.bhp.theta_hyp[1]) + \
                beta.logpdf(.2, self.bhp.alpha_hyp[0], self.bhp.alpha_hyp[1])

        self.assertAlmostEqual(logpost, check)

    def test_diffuse_prior_posterior_correct(self):
        A = self.arr
        bhp2 = BayesianUVExpHawkesProcess((1, 10000), (1, 1), (1, 1e5))

        logpost = bhp2.log_posterior_with_params(A, 5., .2, .1, A[-1])

        check = bhp2.log_likelihood_with_params(A, 5, .2, .1, A[-1]) + \
                gamma.logpdf(5, bhp2.mu_hyp[0], scale=bhp2.mu_hyp[1]) + \
                gamma.logpdf(.1, bhp2.theta_hyp[0], scale=bhp2.theta_hyp[1]) + \
                beta.logpdf(.2, bhp2.alpha_hyp[0], bhp2.alpha_hyp[1])

        self.assertAlmostEqual(logpost, check)

    def test_gradient_correct_finite_difference(self):
        A = self.arr
        f = self.bhp._log_posterior(A, A[-1])
        g = self.bhp._log_posterior_grad(A, A[-1])

        gr_numeric = nd.Gradient(f)([.3, .2, 5.])
        gr_manual = g([.3, .2, 5.])

        np.testing.assert_allclose(gr_manual, gr_numeric, rtol=1e-2)

    def test_hyperparam_setter(self):
        self.assertAlmostEqual(self.bhp.theta_hyp[0], 1.)
        self.assertAlmostEqual(self.bhp.theta_hyp[1], 5.)

        self.assertAlmostEqual(self.bhp.mu_hyp[0], 1.)
        self.assertAlmostEqual(self.bhp.mu_hyp[1], 5.)

        self.assertAlmostEqual(self.bhp.alpha_hyp[0], 1.)
        self.assertAlmostEqual(self.bhp.alpha_hyp[1], 1.)

    def test_log_posterior_with_params(self):
        A = self.arr
        f, g = BayesianUVExpHawkesProcess._get_log_posterior_pot_grad_fns(A, A[-1], (1, 5), (1,1), (1, 5))

        ll0 = f([.5, .6, 7.])
        ll1 = self.bhp.log_posterior_with_params(A, .5, .6, 7., A[-1])

        self.assertAlmostEqual(ll0, ll1)


class TestUVExpBayesMAP(ut.TestCase):

    def setUp(self):
        fpath = os.path.join(os.path.dirname(__file__), 'tfx_fixture.npy')
        self.arr = np.load(fpath)  # test fixture

        self.bhp = BayesianUVExpHawkesProcess((1., 1.), (1, 1), (1., 1.))

    def test_small_data_map_differs(self):
        A = self.arr[:10]
        res = self.bhp._fit_grad_desc(A, A[-1])

        hp = UnivariateExpHawkesProcess()
        res2 = hp._fit_grad_desc(A, A[-1])

        error = np.linalg.norm(res.x - res2.x, ord=1)
        assert error > .001, "Error smaller than .001!" + str(res.x) + str(res2.x)

    def test_fit_sets_params(self):
        A = self.arr[:500]

        assert self.bhp._mu is None

        self.bhp.fit(A, A[-1])

        assert self.bhp._mu is not None

    def test_gradient_0_at_map(self):
        A = self.arr[:500]

        x = np.array([0.0099429, 0.59019621, 0.16108526])
        g = self.bhp._log_posterior_grad(A, A[-1])(x)

        assert np.linalg.norm(g, ord=1) < 10, "Gradient not zero!" + str(g)

    def test_marginal_likelihood_ok(self):
        a, T = self.arr, self.arr[-1]
        self.bhp.fit(a, T)

        ml = self.bhp.marginal_likelihood(a, T)
        true = -44507.4

        self.assertAlmostEqual(ml, true, places=1)

    def test_marginal_likelihood_nofit_raises(self):
        with self.assertRaises(AssertionError):
            self.bhp.marginal_likelihood(self.arr, self.arr[-1])

    def test_log_posterior_correct(self):
        a, T = self.arr, self.arr[-1]

        prior2 = c_uv_bayes.cmake_beta_logpdf(1., 1.)
        prior1 = c_uv_bayes.cmake_gamma_logpdf(1., 1.)

        self.bhp.fit(a, T)

        m, alpha, theta = self.bhp.get_params()

        true_post = c_uv_exp.uv_exp_ll(a, m, alpha, theta, T) + prior2(alpha) + prior1(theta) + \
                    prior1(m)

        calc_post = self.bhp.log_posterior(a, T)

        self.assertAlmostEqual(true_post, calc_post, places=2)

    def test_log_posterior_nofit_raises(self):
        with self.assertRaises(AssertionError):
            self.bhp.log_posterior(self.arr, self.arr[-1])

    def test_log_posterior_with_params_correct(self):
        a, T = self.arr, self.arr[-1]

        m, alpha, theta = 3., .2, 10.

        prior2 = c_uv_bayes.cmake_beta_logpdf(1., 1.)
        prior1 = c_uv_bayes.cmake_gamma_logpdf(1., 1.)

        true_post = c_uv_exp.uv_exp_ll(a, m, alpha, theta, T) + prior2(alpha) + prior1(theta) +\
            prior1(m)

        calc_post = self.bhp.log_posterior_with_params(a, m, alpha, theta, T)

        self.assertAlmostEqual(true_post, calc_post, places=2)
