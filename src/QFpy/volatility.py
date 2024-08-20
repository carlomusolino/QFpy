import numpy as np 


class ConstantVol:
    
    def __init__(self, vol):
        self.vol = vol 
    
    def __call__(self, S, t):
        return self.vol * np.ones_like(S, dtype=float)
    
    def __repr__(self):
        return f"Constant volatility {self.__vol}"
        
    