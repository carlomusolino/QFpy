import numpy as np
from numpy.random import normal 

class NormalSampler:
    
    def __init__(self):
        pass
    
    def get(self):
        return normal()
    


def LogNormalStep(X: float, dt: float, sigma: float, mu: float = 0):
    """Perform a log-normal step for the stochastic variable X.

    Evolve the stochastic variable X(t) from t=0 to t=dt.
    Note that volatility, timestep and drift rate must have 
    compatible units for meaningful results to be obtained.
    
    Args:
        X (float): Stochastic variable.
        dt (float): Timestep.
        sigma (float): Volatility.
        mu (float, optional): Drift rate. Defaults to 0.

    Returns:
        float: The stochastic variable at time dt.
    """
    eps = normal() 
    return X * np.exp( (mu-sigma**2*0.5) * dt + sigma * eps * np.sqrt(dt))

def __QuadraticPortfolioReturnOneAsset(dt: float, sigma: float, alpha: float, beta: float, theta: float = None):
    
    dX = normal() * np.sqrt(sigma[0] * dt)
    
    return theta * dt + alpha[0] * dX + beta[0] * dX ** 2 

def QuadraticPortfolioReturn(dt: float, sigma: np.ndarray, alpha: np.ndarray, beta: np.ndarray, theta: float = None, sampler = NormalSampler()):
    
    # Number of underlyings to the portfolio
    N = len(alpha)
    
    if theta is None:
        theta = 0
    
    # Treat the single asset case separately for convenience
    if N == 1 or type(alpha) is float:
        return __QuadraticPortfolioReturnOneAsset(dt,np.array(sigma),np.array(alpha),np.array(beta),np.array(theta))

    dX = np.zeros(N)
    for i in range(N):
        dX[i] = sampler.get() * np.sqrt(sigma[i,i] * dt)
        
    
        
    return theta * dt + np.sum(alpha*dX) + np.sum( dX * [ np.sum(dX * beta[:,i]) for i in range(N) ] )
    
    
    
def QuadraticPortfolioReturnMC(N: int, dt: float, sigma: np.ndarray, alpha: np.ndarray, beta: np.ndarray, theta: float = None, sampler = NormalSampler()):
    dP = [] 
    for i in range(N):
        dP.append(QuadraticPortfolioReturn(dt,sigma,alpha,beta,theta,sampler))
    return dP 