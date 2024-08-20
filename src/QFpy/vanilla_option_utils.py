import numpy as np
import scipy.stats
from typing import Callable


def __d1(t: np.ndarray,S: np.ndarray,E: float,r: float,D: float,sigma: float,T: float):
    return (np.log(S/E)+(r-D+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
def __d2(t: np.ndarray,S: np.ndarray,E: float,r: float,D: float,sigma: float,T: float):
    return __d1(t,S,E,r,D,sigma,T) - sigma*np.sqrt(T-t)

def __CDF(x):
    return scipy.stats.norm.cdf(x)

def __dCDF(x):
    return 1./np.sqrt(2*np.pi) * np.exp(-0.5*x**2)

def EuropeanCallValue(t: np.ndarray, S: np.ndarray,
                        T: float,
                        E: float,
                        sigma: float, 
                        r: float, 
                        D: float = 0
                        ):
    return S*np.exp(-D*(T-t))*__CDF(__d1(t,S,E,r,D,sigma,T)) - E*np.exp(-r*(T-t))*__CDF(__d2(t,S,E,r,D,sigma,T))

def EuropeanCallDelta(t: np.ndarray, S: np.ndarray,
                        T: float,
                        E: float,
                        sigma: float, 
                        r: float, 
                        D: float = 0
                        ):
    return np.exp(-D*(T-t)) * __CDF(__d1(t,S,E,r,D,sigma,T))

def EuropeanCallGamma(t: np.ndarray, S: np.ndarray,
                        T: float,
                        E: float,
                        sigma: float, 
                        r: float, 
                        D: float = 0
                        ):
    return np.exp(-D*(T-t)) * __dCDF(__d1(t,S,E,r,D,sigma,T)) / ( sigma * S * np.sqrt(T-t) ) 

def EuropeanCallTheta(t: np.ndarray, S: np.ndarray,
                        T: float,
                        E: float,
                        sigma: float, 
                        r: float, 
                        D: float = 0
                        ):
    d1v = __d1(t,S,E,r,D,sigma,T)
    d2v = __d2(t,S,E,r,D,sigma,T)
    return -sigma*S*np.exp(-D*(T-t))*__dCDF(d1v)/(2*np.sqrt(T-t)) + D*S*__CDF(d1v)*np.exp(-D*(T-t)) - r*E*np.exp(-r*(T-t)) * __CDF(d2v)

def EuropeanCallVega(t: np.ndarray, S: np.ndarray,
                        T: float,
                        E: float,
                        sigma: float, 
                        r: float, 
                        D: float = 0
                        ):
    d1v = __d1(t,S,E,r,D,sigma,T)
    return S * np.sqrt(T-t) * np.exp(-D*(T-t)) * __dCDF(d1v)

def EuropeanCallRho(t: np.ndarray, S: np.ndarray,
                        T: float,
                        E: float,
                        sigma: float, 
                        r: float, 
                        D: float = 0
                        ):
    d2v = __d2(t,S,E,r,D,sigma,T)
    return E*(T-t)*np.exp(-r*(T-t))*__CDF(d2v)

def EuropeanCallSpeed(t: np.ndarray, S: np.ndarray,
                        T: float,
                        E: float,
                        sigma: float, 
                        r: float, 
                        D: float = 0
                        ):
    d1v = __d1(t,S,E,r,D,sigma,T)
    return -np.exp(-D*(T-t))*__dCDF(d1v)/(sigma**2*S**2*(T-t)) * ( d1v + sigma * np.sqrt(T-t))

# Puts 

def EuropeanPutValue(t: np.ndarray, S: np.ndarray,
                        T: float,
                        E: float,
                        sigma: float, 
                        r: float, 
                        D: float = 0
                        ):
    return - S*np.exp(-D*(T-t))*__CDF(-__d1(t,S,E,r,D,sigma,T)) + E*np.exp(-r*(T-t))*__CDF(-__d2(t,S,E,r,D,sigma,T))

def EuropeanPutDelta(t: np.ndarray, S: np.ndarray,
                        T: float,
                        E: float,
                        sigma: float, 
                        r: float, 
                        D: float = 0
                        ):
    return np.exp(-D*(T-t)) * (__CDF(__d1(t,S,E,r,D,sigma,T))-1)

def EuropeanPutGamma(t: np.ndarray, S: np.ndarray,
                        T: float,
                        E: float,
                        sigma: float, 
                        r: float, 
                        D: float = 0
                        ):
    return np.exp(-D*(T-t)) * __dCDF(__d1(t,S,E,r,D,sigma,T)) / ( sigma * S * np.sqrt(T-t) ) 

def EuropeanPutTheta(t: np.ndarray, S: np.ndarray,
                        T: float,
                        E: float,
                        sigma: float, 
                        r: float, 
                        D: float = 0
                        ):
    d1v = __d1(t,S,E,r,D,sigma,T)
    d2v = __d2(t,S,E,r,D,sigma,T)
    return -sigma*S*np.exp(-D*(T-t))*__dCDF(-d1v)/(2*np.sqrt(T-t)) - D*S*__CDF(-d1v)*np.exp(-D*(T-t)) + r*E*np.exp(-r*(T-t)) * __CDF(-d2v)

def EuropeanPutVega(t: np.ndarray, S: np.ndarray,
                        T: float,
                        E: float,
                        sigma: float, 
                        r: float, 
                        D: float = 0
                        ):
    d1v = __d1(t,S,E,r,D,sigma,T)
    return S * np.sqrt(T-t) * np.exp(-D*(T-t)) * __dCDF(d1v)

def EuropeanPutRho(t: np.ndarray, S: np.ndarray,
                        T: float,
                        E: float,
                        sigma: float, 
                        r: float, 
                        D: float = 0
                        ):
    d2v = __d2(t,S,E,r,D,sigma,T)
    return -E*(T-t)*np.exp(-r*(T-t))*__CDF(-d2v)

def EuropeanPutSpeed(t: np.ndarray, S: np.ndarray,
                        T: float,
                        E: float,
                        sigma: float, 
                        r: float, 
                        D: float = 0
                        ):
    d1v = __d1(t,S,E,r,D,sigma,T)
    return -np.exp(-D*(T-t))*__dCDF(d1v)/(sigma**2*S**2*(T-t)) * ( d1v + sigma * np.sqrt(T-t))

