"""
Tests for Univariate Hawkes processes
"""

import unittest as ut
import mock
import numpy as np
from ..model.uv_exp import UnivariateExpHawkesProcess as UVHP

# what else to test?
# - for gd and em
# -- algorithm runs without problem
# -- test converges to known values for 3 test fixtures
# -- fitters hit correct methods, C functions
# -- fitters assert stationarity
# - log likelihood of known test fixtures
# - log likelihood of known test fixtures, setting params

# all methods with t refuse non-sorted order
# assert all t must be less than T

class UVExpSamplerTests(ut.TestCase):

    T = 10000

    def setUp(self):
        self.uv = UVHP()
        self.uv.set_params(.5, .2, 10.)

    def test_branching_correct_number_samples(self):

        smp = self.uv.sample(self.T, method="branching")
        EN = .5 * self.T / (1 - self.uv._alpha)
        N = float(len(smp))

        assert abs(N - EN) / N < .05

    @mock.patch('fasthawkes.model.uv_exp.uv_exp_sample_branching')
    def test_branching_calls_correct_cython(self, mock_method):

        smp = self.uv.sample(self.T, method="branching")

        mock_method.assert_called_with(self.T, .5, .2, 10.)

    def test_ogata_correct_number_samples(self):

        smp = self.uv.sample(self.T, method="ogata")
        EN = .5 * self.T / (1 - self.uv._alpha)
        N = float(len(smp))

        assert abs(N - EN) / N < .05

    @mock.patch('fasthawkes.model.uv_exp.uv_exp_sample_ogata')
    def test_ogata_calls_correct_cython(self, mock_method):
        smp = self.uv.sample(self.T, method="ogata")

        mock_method.assert_called_with(self.T, .5, .2, 10.)


class UVExpSetterGetterTests(ut.TestCase):

    def test_params_start_none(self):
        uv = UVHP()
        self.assertIsNone(uv._alpha)
        self.assertIsNone(uv._mu)
        self.assertIsNone(uv._theta)

    def test_getter(self):

        uv = UVHP()
        uv._mu, uv._alpha, uv._theta = .5, .4, .3

        pars1 = np.array(uv.get_params())
        pars2 = np.array([.5, .4, .3])

        np.testing.assert_allclose(pars1, pars2)

    def test_setter(self):
        uv = UVHP()
        uv.set_params(.5, .4, .2)

        pars1 = np.array(uv.get_params())
        pars2 = np.array([.5, .4, .2])

        np.testing.assert_allclose(pars1, pars2)

    def test_fetch(self):
        uv = UVHP()
        uv.set_params(.5, .4, .2)

        pars1 = np.array(uv._fetch_params())
        pars2 = np.array([.5, .4, .2])

        np.testing.assert_allclose(pars1, pars2)


class UVExpLikelihoodTests(ut.TestCase):

    @mock.patch('fasthawkes.model.uv_exp.uv_exp_ll')
    def test_log_likelihood_calls_correct(self, m):
        uv = UVHP()
        a = np.array([2, 3, 6])

        uv.log_likelihood_with_params(a, .3, .2, 10., 1000)
        m.assert_called()

    @mock.patch('fasthawkes.model.uv_exp.uv_exp_ll')
    def test_log_likelihood_passes_all_params(self,m):
        uv = UVHP()
        a = np.array([2, 3, 6])

        uv.log_likelihood_with_params(a, .3, .2, 10., 1000)
        m.assert_called_with(a, .3, .2, 10., 1000)

    @mock.patch('fasthawkes.model.uv_exp.uv_exp_ll')
    def test_log_likelihood_set_params_call_correct(self, m):
        """after set_params, with log_likelihood alone"""
        uv = UVHP()
        uv.set_params(5, .2, 10)
        a = np.array([2, 3, 6])

        uv.log_likelihood(a, 1000)
        m.assert_called_with(a, 5., .2, 10., 1000)

    def test_simple_fixture_ok(self):
        uv = UVHP()
        uv.set_params(5, .2, 10)
        T = 15.

        a = np.array([3, 4, 6, 8.2, 9., 12.])

        mu, alpha, theta = uv.get_params()

        comp = mu * T + alpha * np.sum(1 - np.exp(-theta * (T - a)))

        sum = 0
        for i in range(len(a)):
            s = mu
            for j in range(i):
                s += alpha * theta * np.exp(-theta * (a[i] - a[j]))
            sum += np.log(s)

        computed = uv.log_likelihood(a, T)
        test = sum - comp

        self.assertAlmostEqual(computed, test)

    def test_fixture2_ok(self):
        pass



