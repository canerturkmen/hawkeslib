"""
Tests for Univariate Hawkes processes
"""

import unittest as ut
import mock
from ..model.uv_exp import UnivariateExpHawkesProcess as UVHP


class UVExpSamplerTests(ut.TestCase):

    T = 10000

    def setUp(self):
        self.uv = UVHP()
        self.uv.set_params(.5, .2, 10.)
        print(self.uv)

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

