import numpy as np 
from scipy.optimize import LinearConstraint, SR1, minimize

def compute_return(w,mu):
    return np.dot(w,mu)

def compute_volatility(w, sigma, rho):
    # Step 1: Compute the covariance matrix from the correlation matrix
    cov_matrix = np.outer(sigma, sigma) * rho

    # Step 2: Compute the portfolio variance using matrix multiplication
    portfolio_variance = np.dot(w, np.dot(cov_matrix, w))

    # Step 3: Take the square root to get the portfolio volatility
    return np.sqrt(portfolio_variance)

def compute_volatility_jac(w, sigma, rho):
    N = len(w)
    vol = compute_volatility(w,sigma,rho) 
    cov_matrix = np.outer(sigma, sigma) * rho
    J = np.zeros(N)
    for i in range(N):
        J[i] = np.sum(w * cov_matrix[i,:])
    return J / vol

def find_efficient_frontier(x: float,
                            sigma: np.ndarray,
                            mu: np.ndarray,
                            rho: np.ndarray):
    """Compute a point on the efficient frontier of risk-reward.

    Args:
        x (float): Target return.
        sigma (np.ndarray): Array of volatilities.
        mu (np.ndarray): Array of expected returns.
        rho (np.ndarray): Correlation matrix.
    """
    N = len(sigma)    
    if N == 1:
        return np.array([1])
    # Initialize constraints 
    # Constraint matrix
    A = np.ones((2, N))
    lb = np.zeros(2)
    ub = np.zeros(2)
    
    tol = 1e-04
    # The first constraint: sum of weights equals 1
    A[0, :] = np.ones(N)
    lb[0] = 1 
    ub[0] = 1 
    
    # The second constraint: expected return equals the target return `x`
    A[1, :] = mu
    lb[1] = x 
    ub[1] = x 
    
    # Define linear constraints
    linear_constraints = LinearConstraint(A, lb, ub)
    
    # Define the volatility function (objective function)
    volatility = lambda w: compute_volatility(w, sigma, rho)
    jac = lambda w: compute_volatility_jac(w, sigma, rho)
    # Initial guess: equally weighted portfolio
    w0 = np.ones(N)/N
    
    # Optimize to find the weights that minimize volatility for the target return
    result = minimize(volatility, w0,
                      method="trust-constr", jac=jac,
                      constraints=[linear_constraints])  # Adding bounds to ensure non-negative weights

    if not result.success:
        raise ValueError("Optimization did not converge:", result.message)
    
    # Return the optimal weights
    return result.x