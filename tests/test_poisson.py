import unittest as ut
import numpy as np
from ..model.poisson import PoissonProcess
from cmath import log


class PoissonTests(ut.TestCase):

    def setUp(self):
        self.pp = PoissonProcess()
        self.a = np.array([1., 2., 3., 5.])  # simple fixture

    def test_likelihood_pars_ok(self):
        # with T
        comp = self.pp.log_likelihood_with_params(self.a, 5., 6.)
        true = - 5 * 6. + 4 * log(5.)

        self.assertAlmostEqual(comp, true)

    def test_likelihood_noT(self):
        comp = self.pp.log_likelihood_with_params(self.a, 5.)
        true = - 5.**2 + 4 * log(5.)

        self.assertAlmostEqual(comp, true)

    def test_fit(self):
        self.pp.fit(self.a)
        self.assertAlmostEqual(self.pp._mu, .8)
