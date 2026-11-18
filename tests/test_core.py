import unittest
import numpy as np
from gslat.numerics import clean_mask
from gslat.statistics import certainty, _threshold_lookup

class CoreTests(unittest.TestCase):
    def test_entropy_certainty_has_expected_endpoints(self):
        result=certainty(np.array([.5,.01,.99]))
        self.assertAlmostEqual(result[0],0.0,places=12)
        self.assertGreater(result[1],.9)
        self.assertAlmostEqual(result[1],result[2],places=12)
    def test_empirical_risk_lookup_is_monotone(self):
        edges=np.array([0.,1.,2.,3.])
        water=np.array([.7,.25,.05])
        land=np.array([.05,.25,.7])
        priors,cuts=_threshold_lookup(edges,water,land)
        self.assertEqual(len(priors),1001)
        self.assertTrue(np.all(np.diff(cuts)>=0))
    def test_cleanup_removes_tiny_components(self):
        mask=np.zeros((12,12),dtype=bool)
        mask[2:5,2:5]=True
        mask[8,8:10]=True
        result=clean_mask(mask,np.ones_like(mask),5)
        self.assertTrue(result[3,3])
        self.assertFalse(result[8,8])

if __name__ == '__main__': unittest.main()
