"""
Tests for Univariate Hawkes processes
"""

import unittest as ut
import mock
import numpy as np
import os

from hawkeslib.model.uv_exp import UnivariateExpHawkesProcess as UVHP


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

    @mock.patch('hawkeslib.model.uv_exp.uv_exp_sample_branching')
    def test_branching_calls_correct_cython(self, mock_method):

        smp = self.uv.sample(self.T, method="branching")

        mock_method.assert_called_with(self.T, .5, .2, 10.)

    def test_ogata_correct_number_samples(self):

        smp = self.uv.sample(self.T, method="ogata")
        EN = .5 * self.T / (1 - self.uv._alpha)
        N = float(len(smp))

        assert abs(N - EN) / N < .05

    @mock.patch('hawkeslib.model.uv_exp.uv_exp_sample_ogata')
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

    def setUp(self):
        self.uv = UVHP()
        self.uv.set_params(5, .3, 10.)

    @mock.patch('hawkeslib.model.uv_exp.uv_exp_ll')
    def test_log_likelihood_calls_correct(self, m):
        uv = UVHP()
        a = np.array([2, 3, 6])

        uv.log_likelihood_with_params(a, .3, .2, 10., 1000)
        m.assert_called()

    @mock.patch('hawkeslib.model.uv_exp.uv_exp_ll')
    def test_log_likelihood_passes_all_params(self,m):
        uv = UVHP()
        a = np.array([2, 3, 6])

        uv.log_likelihood_with_params(a, .3, .2, 10., 1000)
        m.assert_called_with(a, .3, .2, 10., 1000)

    @mock.patch('hawkeslib.model.uv_exp.uv_exp_ll')
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
        fpath = os.path.join(os.path.dirname(__file__), 'tfx_fixture.npy')

        arr = np.load(fpath)

        tgt = -3628302.7661192594
        computed = self.uv.log_likelihood(arr, arr[-1])

        self.assertAlmostEqual(tgt, computed)

    def test_empty_array_ok(self):
        arr = np.array([])

        tgt = -10.
        computed = self.uv.log_likelihood(arr, 2.)

        self.assertAlmostEqual(tgt, computed)

    def test_ll_methods_refuse_nonsorted(self):
        t = np.array([3., 2., 5.])

        with self.assertRaises(ValueError):
            self.uv.log_likelihood(t, 6.)

        with self.assertRaises(ValueError):
            self.uv.log_likelihood_with_params(t, 10, .5, 10, 6.)

    def test_ll_methods_refuse_invalid_T(self):
        t = np.array([2., 3., 5.])

        with self.assertRaises(ValueError):
            self.uv.log_likelihood(t, 4.)

        with self.assertRaises(ValueError):
            self.uv.log_likelihood_with_params(t, 10, .5, 10, 4.)

    def test_ll_methods_refuse_negative_t(self):
        t = np.array([-2., 3., 5.])

        with self.assertRaises(ValueError):
            self.uv.log_likelihood(t, 7.)

        with self.assertRaises(ValueError):
            self.uv.log_likelihood_with_params(t, 10, .5, 10, 7.)


class UVExpFittingTests(ut.TestCase):

    def setUp(self):
        fpath = os.path.join(os.path.dirname(__file__), 'tfx_fixture.npy')
        self.arr = np.load(fpath)

        self.uv = UVHP()

    @mock.patch('hawkeslib.model.uv_exp.uv_exp_ll_grad')
    def test_fitter_runs_gd(self, m):
        a = self.arr

        try:
            self.uv.fit(a, a[-1], method="gd")
        except:
            pass

        m.assert_called()

    @mock.patch('hawkeslib.model.uv_exp.uv_exp_fit_em_base')
    def test_fitter_runs_em(self, m):
        a = self.arr

        try:
            self.uv.fit(a, a[-1], method="em")
        except:
            pass

        m.assert_called()

    def test_fitter_refuses_nonsorted(self):
        a = np.array([5., 6., 7., 6.5, 8.])

        with self.assertRaises(ValueError):
            self.uv.fit(a, a[-1], method="em")

        with self.assertRaises(ValueError):
            self.uv.fit(a, a[-1], method="gd")

    def test_fitter_refuses_badT(self):
        a = np.array([5., 6., 7., 8.])

        with self.assertRaises(ValueError):
            self.uv.fit(a, 6., method="em")

        with self.assertRaises(ValueError):
            self.uv.fit(a, 6., method="gd")

    def test_em_fixture_correct(self):
        self.uv.fit(self.arr, method="em")
        pars = self.uv.get_params()

        np.testing.assert_allclose(pars, [.006, .555, .1612], rtol=.05)

    def test_gd_fixture_correct(self):
        self.uv.fit(self.arr, method="gd")
        pars = self.uv.get_params()

        np.testing.assert_allclose(pars, [.006, .555, .1612], rtol=.05)

    def test_fitter_no_unpack_error(self):
        a = np.array([5., 6., 7., 8.])

        try:
            self.uv.fit(a, 10., method="em")
        except ValueError as e:
            if "to unpack" in e.message:
                self.fail(e.message)

