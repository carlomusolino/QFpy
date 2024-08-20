import numpy as np
from typing import Callable

from QFpy.VaR_utils import GetCornishFisherPercentile, GetLinearPercentile
from QFpy.MC_utils import QuadraticPortfolioReturnMC

def __VaRScalar(k: float, dt: float, 
                S: np.ndarray, 
                vols: np.ndarray, 
                deltas: np.ndarray, 
                gammas: np.ndarray = None, 
                theta: float = 0, 
                rho: np.ndarray = None, 
                method: str = "quadratic",
                valuation: Callable = None,
                N: int = 0,
                *args):
    if vols is None or deltas is None or S is None:
        raise ValueError("vols, deltas, and S must not be None")
    
    if rho is None:
        rho = np.ones_like(vols)
    if gammas is None:
        gammas = np.zeros_like(vols)
    
    # Construct sigma alpha and beta
    sigma = rho * vols**2 
    alpha = deltas * S
    beta = 0.5 * gammas * S**2
        
    if method == "linear":
        return abs(GetLinearPercentile(1-k,sigma,alpha)[0]) * np.sqrt(dt)
    elif method == "quadratic":
        return abs(GetCornishFisherPercentile(1-k,sigma,alpha,beta)[0]) * np.sqrt(dt)
    elif method == "MC-quadratic":
        return abs(np.quantile(np.sort(QuadraticPortfolioReturnMC(N,1,sigma,alpha,beta,theta)),1-k)) * np.sqrt(dt)

def VaR(k: float, dt: float, 
        S: np.ndarray, 
        vols: np.ndarray, 
        deltas: np.ndarray, 
        gammas: np.ndarray = None, 
        theta: float = 0, 
        rho: np.ndarray = None, 
        method: str = "quadratic",
        valuation: Callable = None,
        N: int = 0,
        *args):
    if not method in ["quadratic", "linear", "MC-quadratic", "MC-linear", "MC"]:
        raise ValueError(f"Method {method} not supported.")
    if method == "MC" and valuation is None :
        raise ValueError("To use MC, you need to provide a valuation function.")
    if method in ["MC", "MC-linear", "MC-quadratic"] and N<=0:
        raise ValueError("To use MonteCarlo methods please provide a positive number of trials.")
    
    if not type(S) is np.ndarray or len(S) == 1:
        return __VaRScalar(k,dt,np.array([S]),np.array([vols]),
                           np.array([deltas]),
                           np.array([gammas]),
                           np.array([theta]),
                           np.array([1]),
                           np.array([method]),
                           valuation,N,*args)
    
    N = len(S)
    
    if rho is None:
        rho = np.eye(N)
    if gammas is None:
        gammas = np.zeros((N,N))
        
    # Construct sigma alpha and beta
    sigma = rho 
    alpha = deltas * S
    beta = 0.5 * gammas
    # Scale sigma and beta by 
    # vols and S
    for i in range(N):
        sigma[:,i] *= vols
        sigma[i,:] *= vols 
        beta[i,:] *= S 
        beta[:,i] *= S 
        
    if method == "linear":
        return GetLinearPercentile(1-k,sigma,alpha) * np.sqrt(dt)
    elif method == "quadratic":
        return GetCornishFisherPercentile(1-k,sigma,alpha,beta) * np.sqrt(dt)
    elif method == "MC-quadratic":
        return np.quantile(np.sort(QuadraticPortfolioReturnMC(N,1,sigma,alpha,beta,theta))) * np.sqrt(dt)