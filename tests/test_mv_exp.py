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