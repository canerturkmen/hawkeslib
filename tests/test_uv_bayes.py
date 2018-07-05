import os

import numdifftools as nd
import numpy as np
import unittest as ut

from scipy.stats import beta, gamma

from ..model.uv_exp import UnivariateExpHawkesProcess
from ..model.uv_bayes import BayesianUVExpHawkesProcess
from ..model.c.c_uv_bayes  import cmake_gamma_logpdf, cmake_beta_logpdf


class TestBayesHelpers(ut.TestCase):

    def test_gamma_prior(self):
        k, scale = 5., 7.

        sp = gamma.logpdf(6., k, scale=scale)
        f0 = cmake_gamma_logpdf(k, scale)

        self.assertAlmostEqual(sp, f0(6.), places=4)

    def test_beta_prior(self):
        a, b = 3., 3.

        sp = beta.logpdf(.45, a, b)

        f0 = cmake_beta_logpdf(a, b)

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
        f = self.bhp._get_log_posterior(A, A[-1])
        g = self.bhp._get_log_posterior_grad(A, A[-1])

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

        self.bhp = BayesianUVExpHawkesProcess((1., 5.), (1, 1), (1., 5.))

    def test_diffuse_prior_map_close_to_mle(self):
        A = self.arr
        bhp2 = BayesianUVExpHawkesProcess((1, 10000), (1, 1), (1, 1e5))
        hp = UnivariateExpHawkesProcess()

        res = bhp2._fit_grad_desc(A, A[-1])
        res2 = hp._fit_grad_desc(A, A[-1])

        np.testing.assert_allclose(res.x, res2.x, rtol=.005)

    def test_small_data_map_differs(self):
        # todo: failing probabilistically!
        A = self.arr[:50]
        res = self.bhp._fit_grad_desc(A, A[-1])

        hp = UnivariateExpHawkesProcess()
        res2 = hp._fit_grad_desc(A, A[-1])

        error = np.linalg.norm(res.x - res2.x, ord=1)
        assert error > .1, "Error smaller than .1!" + str(res.x) + str(res2.x)

    def test_num_gradient_0_at_map(self):
        # todo: failing probabilistically
        A = self.arr[:500]
        res = self.bhp._fit_grad_desc(A, A[-1])

        f = self.bhp._get_log_posterior(A, A[-1])

        # g = nd.Gradient(f)(res.x)
        g = self.bhp._get_log_posterior_grad(A, A[-1])(res.x)

        assert np.linalg.norm(g, ord=2) < 10, "Gradient not zero!" + str(g) + str(res.x)
        np.testing.assert_allclose(np.array([0.0099429, 0.59019621, 0.16108526]), res.x, rtol=.1)

    def test_fit_sets_params(self):
        pass


