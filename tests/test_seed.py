import unittest
import numpy as np
from gslat.seed import histogram_seed

class SeedTests(unittest.TestCase):
    def test_histogram_seed_stays_in_native_coordinate(self):
        values=np.concatenate([np.full(600,2.0),np.full(400,8.0),np.linspace(2.0,8.0,200)]).reshape(20,60)
        valid=np.ones_like(values,dtype=bool)
        seed,cut,detail=histogram_seed(values,valid)
        self.assertTrue(np.isfinite(cut))
        self.assertGreaterEqual(cut,float(values.min()))
        self.assertLessEqual(cut,float(values.max()))
        np.testing.assert_array_equal(seed,valid & (values<=cut))
        self.assertIn(detail["rule"],{"interpeak-valley","longest-tail-triangle","triangle"})

if __name__ == '__main__': unittest.main()
