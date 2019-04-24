import unittest as ut
import numpy as np

from hawkeslib.util.multitrace import MultiTrace


class MultiTraceTests(ut.TestCase):

    def setUp(self):

        self.a = np.array(range(10))
        self.b = np.array(range(10, 20))

        self.mt = MultiTrace(["a", "b"], self.a, self.b)

    def test_kv_correct(self):

        assert np.allclose(self.mt["a"], self.a)
        assert np.allclose(self.mt["b"], self.b)

    def test_slice_correct(self):

        mt2 = self.mt[:5]

        assert isinstance(mt2, MultiTrace)
        assert np.allclose(np.array(range(5)), mt2["a"])
        assert np.allclose(np.array(range(10, 15)), mt2["b"])
