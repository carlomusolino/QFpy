import unittest
import numpy as np

from QFpy.capm_utils import find_efficient_frontier, compute_volatility, compute_return

class TestFindEfficientFrontier(unittest.TestCase):
    def test_find_efficient_frontier_basic(self):
        # Basic test case with known values
        sigma = np.array([0.1, 0.2, 0.15])
        mu = np.array([0.05, 0.1, 0.07])
        rho = np.array([[1.0, 0.5, 0.3],
                        [0.5, 1.0, 0.6],
                        [0.3, 0.6, 1.0]])
        target_return = 0.08

        # Call the function
        weights = find_efficient_frontier(target_return, sigma, mu, rho)

        # Assert the weights sum to 1
        self.assertAlmostEqual(np.sum(weights), 1.0, places=7)
        
        # Assert the expected return is close to the target return
        actual_return = compute_return(weights,mu)
        self.assertAlmostEqual(actual_return, target_return, places=7)

    def test_find_efficient_frontier_single_asset(self):
        # Test with a single asset (trivial case)
        sigma = np.array([0.2])
        mu = np.array([0.1])
        rho = np.array([[1.0]])
        target_return = 0.1

        # Call the function
        weights = find_efficient_frontier(target_return, sigma, mu, rho)

        # Assert that the single asset receives all the weight
        self.assertAlmostEqual(weights[0], 1.0, places=7)

    def test_find_efficient_frontier_equal_assets(self):
        # Test with two assets with the same characteristics
        sigma = np.array([0.2, 0.2])
        mu = np.array([0.1, 0.1])
        rho = np.array([[1.0, 1.0],
                        [1.0, 1.0]])
        target_return = 0.1

        # Call the function
        weights = find_efficient_frontier(target_return, sigma, mu, rho)

        # Assert that the weights are equal
        self.assertAlmostEqual(weights[0], 0.5, places=7)

    def test_find_efficient_frontier_no_solution(self):
        # Test where no solution should exist (target return too high)
        sigma = np.array([0.1, 0.2])
        mu = np.array([0.05, 0.06])
        rho = np.array([[1.0, 0.5],
                        [0.5, 1.0]])
        target_return = 0.2  # This is higher than any individual asset return

        # Expect an error due to the infeasibility of achieving this return
        
        weights = find_efficient_frontier(target_return, sigma, mu, rho)
        risk = compute_volatility(weights, sigma, rho)
        self.assertAlmostEqual(risk,2.599999999999998, places=7)

if __name__ == '__main__':
    unittest.main()