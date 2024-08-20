import numpy as np
from abc import ABC, abstractmethod
from typing import Callable 
from scipy.integrate import trapz 

class Option(ABC):
    """Abstract base class for financial options.

    Methods:
        payoff
    """
    def __init__(self):
        pass 
    
    @abstractmethod
    def payoff(self, S: float, path_dependent: bool):
        pass
    
    @abstractmethod
    def asymptotic_value(self, S:float, t: float, rate: Callable):
        pass
    
    @abstractmethod    
    def exercise(self,V: np.ndarray, S: np.ndarray, t: float, *args):
        pass


class EuropeanCallOption(Option):
    
    def __init__(self, K, T):
        self.__K = K
        self.T   = T 
        
    def payoff(self, S: float, path_dependent: bool, *args):
        if path_dependent:
            return np.maximum(0, S[-1] - self.__K)
        else:
            return np.maximum(0, S - self.__K)
    
    def asymptotic_value(self, S: float, t: float, rate: Callable):
        times = np.linspace(t, self.T, 100)
        rdt = trapz(rate(times), times)
        return (0, S[1] - self.__K * np.exp(-rdt))
    
    def exercise(self,V: np.ndarray, S: np.ndarray, t: float, *args):
        return V
    
    def __repr__(self):
        return f"European style call option, strike {self.__K} expiry {self.T}"
    
class EuropeanPutOption(Option):
    
    def __init__(self, K, T):
        self.__K = K
        self.T = T 
        
    def payoff(self, S: float, path_dependent: bool, *args):
        if path_dependent:
            return np.maximum(0, self.__K - S[-1])
        else:
            return np.maximum(0, self.__K - S)
    
    def asymptotic_value(self, S:float, t: float, rate: Callable):
        times = np.linspace(t, self.T, 100)
        rdt = trapz(rate(times), times)
        return (self.__K * np.exp(-rdt), 0)
    
    def exercise(self,V: np.ndarray, S: np.ndarray, t: float, *args):
        return V
    
    def __repr__(self):
        return f"European style put option, strike {self.__K} expiry {self.T}"
    
class AmericanCallOption(Option):
    
    def __init__(self, K, T):
        self.__K = K
        self.T = T 
        
    def payoff(self, S: float, path_dependent: bool, *args):
        if path_dependent:
            return np.maximum(0, S[-1] - self.__K)
        else:
            return np.maximum(0, S - self.__K)
    
    def asymptotic_value(self, S:float, t: float, rate: Callable):
        times = np.linspace(t, self.T, 100)
        rdt = trapz(rate(times), times)
        return (0, S - self.__K * np.exp(-rdt))
    
    def exercise(self,V: np.ndarray, S: np.ndarray, t: float, *args):
        return np.maximum(V,self.payoff(S,False,*args))
    
    def __repr__(self):
        return f"American style call option, strike {self.__K} expiry {self.T}"
    
class AmericanPutOption(Option):
    
    def __init__(self, K, T):
        self.__K = K
        self.T = T 
        
    def payoff(self, S: float, path_dependent: bool, *args):
        if path_dependent:
            return np.maximum(0, self.__K - S[-1])
        else:
            return np.maximum(0, self.__K - S)
    
    def asymptotic_value(self, S:float, t: float, rate: Callable):
        times = np.linspace(t, self.T, 100)
        rdt = trapz(rate(times), times)
        return (self.__K * np.exp(-rdt), 0)
    
    def exercise(self,V: np.ndarray, S: np.ndarray, t: float, *args):
        return np.maximum(V,self.payoff(S,False,*args))
    
    def __repr__(self):
        return f"American style put option, strike {self.__K} expiry {self.T}"

class AsianCallOption(Option):
    
    def __init__(self, K, T, mean_kind="arithmetic"):
        self.__K = K
        self.T = T 
        self.Type = mean_kind
        
    def payoff(self, S: float, path_dependent: bool, *args):
        if not path_dependent:
                raise ValueError("Asian style options require the history of the asset price to be evaluated.")
        if self.Type == "arithmetic":
            Smean = np.mean(np.array(S))
        elif self.Type == "geometrics":
            Smean = 0 
        return np.maximum(0, Smean - self.__K)
    
    #unused
    def asymptotic_value(self, S:float, t: float, rate: Callable):
        return 0
    
    #unused
    def exercise(self,V: np.ndarray, S: np.ndarray, t: float, *args):
        return V
    
    def __repr__(self):
        return f"Asian style call option, strike {self.__K} expiry {self.T}"
    
class AsianPutOption(Option):
    
    def __init__(self, K, T, mean_kind="arithmetic"):
        self.__K = K
        self.T = T 
        self.Type = mean_kind
        
    def payoff(self, S: float, path_dependent: bool, *args):
        if not path_dependent:
                raise ValueError("Asian style options require the history of the asset price to be evaluated.")
        if self.Type == "arithmetic":
            Smean = np.mean(np.array(S))
        elif self.Type == "geometrics":
            Smean = 0 
        return np.maximum(0, Smean - self.__K)
    
    #unused
    def asymptotic_value(self, S:float, t: float, rate: Callable):
        return 0
    
    #unused
    def exercise(self,V: np.ndarray, S: np.ndarray, t: float, *args):
        return V
    
    def __repr__(self):
        return f"Asian style put option, strike {self.__K} expiry {self.T}"
    