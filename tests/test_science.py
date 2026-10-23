import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from science import evaluate_depth,srgb_to_linear,linear_to_srgb,transform

class ScientificChecks(unittest.TestCase):
    def test_perfect_prediction(self):
        g=np.full((480,640),2.,dtype=np.float32)
        r=evaluate_depth(g,g)
        self.assertEqual(r['abs_rel'],0)
        self.assertEqual(r['rmse_m'],0)
        self.assertEqual(r['delta1'],1)
        self.assertEqual(r['valid_pixels'],426*560)
    def test_known_one_meter_error(self):
        g=np.full((480,640),2.)
        r=evaluate_depth(g+1,g)
        self.assertAlmostEqual(r['abs_rel'],.5)
        self.assertAlmostEqual(r['rmse_m'],1.)
        self.assertEqual(r['delta1'],0)
    def test_reference_mask(self):
        g=np.full((480,640),2.);g[50:80,50:80]=0
        r=evaluate_depth(np.full_like(g,2.),g)
        self.assertEqual(r['abs_rel'],0)
        self.assertEqual(r['valid_pixels'],426*560-900)
    def test_nonfinite_prediction_rejected(self):
        p=np.full((480,640),2.);p[50,50]=np.nan
        with self.assertRaises(ValueError):evaluate_depth(p,np.full_like(p,2.))
    def test_srgb_roundtrip(self):
        x=np.linspace(0,1,301,dtype=np.float32)
        np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(x)),x,atol=2e-7)
    def test_identity_and_gray_world_neutralization(self):
        x=np.full((20,30,3),[.6,.4,.2],dtype=np.float32)
        same,_=transform(x,'clean','none')
        np.testing.assert_allclose(same,x,atol=2e-7)
        y,_=transform(x,'clean','gray_world')
        np.testing.assert_allclose(y[:,:,0],y[:,:,1],atol=2e-7)
        np.testing.assert_allclose(y[:,:,1],y[:,:,2],atol=2e-7)

if __name__=='__main__':unittest.main()
