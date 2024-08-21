import numpy as np
from abc import ABC, abstractmethod
from typing import Callable 
from scipy.integrate import trapz 
from scipy.stats import gmean

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
            return np.maximum(0, S[-1,:] - self.__K)
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
            return np.maximum(0, self.__K - S[-1,:])
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
            return np.maximum(0, S[-1,:] - self.__K)
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
            return np.maximum(0, self.__K - S[-1,:])
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
            Smean = np.mean(S,axis=0)
        elif self.Type == "geometrics":
            Smean = gmean(S, axis=0) 
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
            Smean = np.mean(S,axis=0)
        elif self.Type == "geometrics":
            Smean = gmean(S, axis=0) 
        return np.maximum(0, Smean - self.__K)
    
    #unused
    def asymptotic_value(self, S:float, t: float, rate: Callable):
        return 0
    
    #unused
    def exercise(self,V: np.ndarray, S: np.ndarray, t: float, *args):
        return V
    
    def __repr__(self):
        return f"Asian style put option, strike {self.__K} expiry {self.T}"

class DownAndInCallOption(Option):
    
    def __init__(self, K, B, T):
        self.__K = K
        self.__B = B
        self.T = T 
        
    def payoff(self, S: float, path_dependent: bool, *args):
        if not path_dependent:
                raise ValueError("Barrier options require the history of the asset price to be evaluated.")
        if ( np.minimum( S ) <  self.__B):
            return np.maximum(0, S[-1,:] - self.__K)
        else:
            return 0
    
    #unused
    def asymptotic_value(self, S:float, t: float, rate: Callable):
        return 0
    
    #unused
    def exercise(self,V: np.ndarray, S: np.ndarray, t: float, *args):
        return V
    
    def __repr__(self):
        return f"Barrier style down and in call option, strike {self.__K} expiry {self.T}, barrier {self.__B}"

class DownAndOutCallOption(Option):
    
    def __init__(self, K, B, T):
        self.__K = K
        self.__B = B
        self.T = T 
        
    def payoff(self, S: float, path_dependent: bool, *args):
        if not path_dependent:
                raise ValueError("Barrier options require the history of the asset price to be evaluated.")
        if ( np.minimum( np.array(S) ) >  self.__B):
            return np.maximum(0, S[-1,:] - self.__K)
        else:
            return 0
    
    #unused
    def asymptotic_value(self, S:float, t: float, rate: Callable):
        return 0
    
    #unused
    def exercise(self,V: np.ndarray, S: np.ndarray, t: float, *args):
        return V
    
    def __repr__(self):
        return f"Barrier style down and out call option, strike {self.__K} expiry {self.T}, barrier {self.__B}"

class DownAndInPutOption(Option):
    
    def __init__(self, K, B, T):
        self.__K = K
        self.__B = B
        self.T = T 
        
    def payoff(self, S: float, path_dependent: bool, *args):
        if not path_dependent:
                raise ValueError("Barrier options require the history of the asset price to be evaluated.")
        if ( np.minimum( np.array(S) ) <  self.__B):
            return np.maximum(0, self.__K - S[-1])
        else:
            return 0
    
    #unused
    def asymptotic_value(self, S:float, t: float, rate: Callable):
        return 0
    
    #unused
    def exercise(self,V: np.ndarray, S: np.ndarray, t: float, *args):
        return V
    
    def __repr__(self):
        return f"Barrier style down and in put option, strike {self.__K} expiry {self.T}, barrier {self.__B}"

class DownAndOutPutOption(Option):
    
    def __init__(self, K, B, T):
        self.__K = K
        self.__B = B
        self.T = T 
        
    def payoff(self, S: float, path_dependent: bool, *args):
        if not path_dependent:
                raise ValueError("Barrier options require the history of the asset price to be evaluated.")
        if ( np.minimum( np.array(S) ) >  self.__B):
            return np.maximum(0, self.__K - S[-1])
        else:
            return 0
    
    #unused
    def asymptotic_value(self, S:float, t: float, rate: Callable):
        return 0
    
    #unused
    def exercise(self,V: np.ndarray, S: np.ndarray, t: float, *args):
        return V
    
    def __repr__(self):
        return f"Barrier style down and out put option, strike {self.__K} expiry {self.T}, barrier {self.__B}"

class UpAndInCallOption(Option):
    
    def __init__(self, K, B, T):
        self.__K = K
        self.__B = B
        self.T = T 
        
    def payoff(self, S: float, path_dependent: bool, *args):
        if not path_dependent:
                raise ValueError("Barrier options require the history of the asset price to be evaluated.")
        if ( np.maximum( np.array(S) ) >  self.__B):
            return np.maximum(0, S[-1] - self.__K)
        else:
            return 0
    
    #unused
    def asymptotic_value(self, S:float, t: float, rate: Callable):
        return 0
    
    #unused
    def exercise(self,V: np.ndarray, S: np.ndarray, t: float, *args):
        return V
    
    def __repr__(self):
        return f"Barrier style up and in call option, strike {self.__K} expiry {self.T}, barrier {self.__B}"

class UpAndOutCallOption(Option):
    
    def __init__(self, K, B, T):
        self.__K = K
        self.__B = B
        self.T = T 
        
    def payoff(self, S: float, path_dependent: bool, *args):
        if not path_dependent:
                raise ValueError("Barrier options require the history of the asset price to be evaluated.")
        if ( np.maximum( np.array(S) ) <  self.__B):
            return np.maximum(0, S[-1] - self.__K)
        else:
            return 0
    
    #unused
    def asymptotic_value(self, S:float, t: float, rate: Callable):
        return 0
    
    #unused
    def exercise(self,V: np.ndarray, S: np.ndarray, t: float, *args):
        return V
    
    def __repr__(self):
        return f"Barrier style up and out call option, strike {self.__K} expiry {self.T}, barrier {self.__B}"

class UpAndInPutOption(Option):
    
    def __init__(self, K, B, T):
        self.__K = K
        self.__B = B
        self.T = T 
        
    def payoff(self, S: float, path_dependent: bool, *args):
        if not path_dependent:
                raise ValueError("Barrier options require the history of the asset price to be evaluated.")
        if ( np.maximum( np.array(S) ) >  self.__B):
            return np.maximum(0, self.__K - S[-1])
        else:
            return 0
    
    #unused
    def asymptotic_value(self, S:float, t: float, rate: Callable):
        return 0
    
    #unused
    def exercise(self,V: np.ndarray, S: np.ndarray, t: float, *args):
        return V
    
    def __repr__(self):
        return f"Barrier style up and out put option, strike {self.__K} expiry {self.T}, barrier {self.__B}"

class UpAndOutPutOption(Option):
    
    def __init__(self, K, B, T):
        self.__K = K
        self.__B = B
        self.T = T 
        
    def payoff(self, S: float, path_dependent: bool, *args):
        if not path_dependent:
                raise ValueError("Barrier options require the history of the asset price to be evaluated.")
        if ( np.maximum( np.array(S) ) <  self.__B):
            return np.maximum(0, self.__K - S[-1])
        else:
            return 0
    
    #unused
    def asymptotic_value(self, S:float, t: float, rate: Callable):
        return 0
    
    #unused
    def exercise(self,V: np.ndarray, S: np.ndarray, t: float, *args):
        return V
    
    def __repr__(self):
        return f"Barrier style up and out put option, strike {self.__K} expiry {self.T}, barrier {self.__B}"

class FixedStrikeLookbackCallOption(Option):
    
    def __init__(self, K, T, kind="max"):
        self.T = T 
        self.__K = K 
        self.__kind = kind
        if not kind in ["max","min"]:
            raise ValueError("Type kind argument of a LookbackOption"+ 
                             "constructor has to be either 'min' or 'max'")
    
    def payoff(self, S: float, path_dependent: bool, *args):
        if not path_dependent:
                raise ValueError("Barrier options require the history of the asset price to be evaluated.")
        if self.__kind == 'min':
            s = np.minimum(np.array(s))
        else:
            se = np.maximum(np.array(S))
        return np.maximum(0, self.__K - s)

    
    #unused
    def asymptotic_value(self, S:float, t: float, rate: Callable):
        return 0
    
    #unused
    def exercise(self,V: np.ndarray, S: np.ndarray, t: float, *args):
        return V
    
    def __repr__(self):
        return f"Fixed strike lookback call option, strike {self.__K} expiry {self.T}, "+\
               f"based on {self.__kind} asset price."

class FixedStrikeLookbackPutOption(Option):
    
    def __init__(self, K, T, kind="max"):
        self.T = T 
        self.__K = K 
        self.__kind = kind
        if not kind in ["max","min"]:
            raise ValueError("Type kind argument of a LookbackOption"+ 
                             "constructor has to be either 'min' or 'max'")
    
    def payoff(self, S: float, path_dependent: bool, *args):
        if not path_dependent:
                raise ValueError("Barrier options require the history of the asset price to be evaluated.")
        if self.__kind == 'min':
            s = np.minimum(np.array(S))
        else:
            s = np.maximum(np.array(S))
        return np.maximum(0, s - self.__K)

    
    #unused
    def asymptotic_value(self, S:float, t: float, rate: Callable):
        return 0
    
    #unused
    def exercise(self,V: np.ndarray, S: np.ndarray, t: float, *args):
        return V
    
    def __repr__(self):
        return f"Fixed strike lookback put option, strike {self.__K} expiry {self.T}, "+\
               f"based on {self.__kind} asset price."
               
class FloatingStrikeLookbackCallOption(Option):
    
    def __init__(self, T, kind="max"):
        self.T = T 
        self.__kind = kind
        if not kind in ["max","min"]:
            raise ValueError("Type kind argument of a LookbackOption"+ 
                             "constructor has to be either 'min' or 'max'")
    
    def payoff(self, S: float, path_dependent: bool, *args):
        if not path_dependent:
                raise ValueError("Barrier options require the history of the asset price to be evaluated.")
        if self.__kind == 'min':
            K = np.minimum(np.array(S))
        else:
            K = np.maximum(np.array(S))
        return np.maximum(0, K - S[-1])

    
    #unused
    def asymptotic_value(self, S:float, t: float, rate: Callable):
        return 0
    
    #unused
    def exercise(self,V: np.ndarray, S: np.ndarray, t: float, *args):
        return V
    
    def __repr__(self):
        return f"Floating strike lookback call option, expiry {self.T}, "+\
               f"strike based on {self.__kind} asset price."

class FloatingStrikeLookbackPutOption(Option):
    
    def __init__(self, T, kind="max"):
        self.T = T 
        self.__kind = kind
        if not kind in ["max","min"]:
            raise ValueError("Type kind argument of a LookbackOption"+ 
                             "constructor has to be either 'min' or 'max'")
    
    def payoff(self, S: float, path_dependent: bool, *args):
        if not path_dependent:
                raise ValueError("Barrier options require the history of the asset price to be evaluated.")
        if self.__kind == 'min':
            K = np.minimum(np.array(S))
        else:
            K = np.maximum(np.array(S))
        return np.maximum(0, S[-1] - K)

    
    #unused
    def asymptotic_value(self, S:float, t: float, rate: Callable):
        return 0
    
    #unused
    def exercise(self,V: np.ndarray, S: np.ndarray, t: float, *args):
        return V
    
    def __repr__(self):
        return f"Floating strike lookback put option, expiry {self.T}, "+\
               f"strike based on {self.__kind} asset price."