import unittest
import numpy as np
from QFpy.VaR_utils import GetM2 

class TestSecondMomentEstimation(unittest.TestCase):
    def test_small_case(self):
        # Example with a small 2x2 matrix
        sigma = np.array([[1, 0.5], [0.5, 1]])
        alpha = np.array([0.5, 1])
        beta = np.array([[1, 1/3], [1/3, 1]])
        
        # Result computed in Mathematica
        expected_result = 12037/72
        
        # Call the function
        result = GetM2(sigma, alpha, beta)
        
        # Use assertAlmostEqual for floating-point comparisons
        self.assertAlmostEqual(result, expected_result, places=5)
    
    def test_identity_sigma(self):
        # Sigma as identity, should simplify the result
        sigma = np.identity(2)
        alpha = np.array([1, 1])
        beta = np.array([[1, 0], [0, 1]])
        
        # Result computed in Mathematica
        expected_result = 72 
        
        result = GetM2(sigma, alpha, beta)
        
        self.assertAlmostEqual(result, expected_result, places=5)
    
    def test_zeros_in_beta(self):
        # Test case with zeros in beta
        sigma = np.array([[1, 0.5], [0.5, 1]])
        alpha = np.array([0.5, 1])
        beta = np.zeros((2, 2))
        
        # If beta is zero, the result should be zero
        expected_result = 0
        
        result = GetM2(sigma, alpha, beta)
        
        self.assertAlmostEqual(result, expected_result, places=5)
        
if __name__ == "__main__":
    unittest.main()