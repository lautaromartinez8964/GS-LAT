import unittest
import numpy as np
from scipy.ndimage import uniform_filter
from gslat.numerics import clean_mask
from gslat.statistics import _threshold_lookup, _update_prior

class CoreTests(unittest.TestCase):
    def test_empirical_risk_lookup_is_monotone(self):
        edges=np.array([0.,1.,2.,3.])
        water=np.array([.7,.25,.05])
        land=np.array([.05,.25,.7])
        priors,cuts=_threshold_lookup(edges,water,land)
        self.assertEqual(len(priors),1001)
        self.assertTrue(np.all(np.diff(cuts)>=0))
    def test_local_prior_uses_valid_posterior_average_without_shrinkage(self):
        indices = np.array([[0, 1, 0]])
        valid = np.array([[True, True, False]])
        initial_prior = 0.2
        prior = np.full(indices.shape, initial_prior, dtype=np.float32)
        denominator = uniform_filter(valid.astype(np.float32), size=3, mode="constant")
        result = _update_prior(prior, indices, np.array([0.8, 0.2]), np.array([0.2, 0.8]),
                               valid, 3, initial_prior, denominator)
        expected = (0.5 + 1 / 17) / 2
        np.testing.assert_allclose(result, [[expected, expected, initial_prior]], rtol=1e-6)
    def test_local_prior_keeps_probability_bounds(self):
        indices = np.array([[0, 1]])
        valid = np.ones_like(indices, dtype=bool)
        prior = np.full(indices.shape, 0.5, dtype=np.float32)
        result = _update_prior(prior, indices, np.array([1., 0.]), np.array([0., 1.]),
                               valid, 1, 0.5, np.ones_like(prior))
        np.testing.assert_allclose(result, [[0.99, 0.01]])
    def test_cleanup_removes_tiny_components(self):
        mask=np.zeros((12,12),dtype=bool)
        mask[2:5,2:5]=True
        mask[8,8:10]=True
        result=clean_mask(mask,np.ones_like(mask),5)
        self.assertTrue(result[3,3])
        self.assertFalse(result[8,8])

if __name__ == '__main__': unittest.main()
