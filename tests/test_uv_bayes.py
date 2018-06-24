import unittest as ut

from scipy.stats import beta, gamma

from ..model.c.c_uv_bayes import cmake_gamma_logpdf, cmake_beta_logpdf


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
