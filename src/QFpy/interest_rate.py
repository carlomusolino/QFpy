import numpy as np

class ConstantRate:
    
    def __init__(self, r):
        self.r = r 
        
    def __call__(self, t):
        return self.r * np.ones_like(t, dtype=float) 
    
    def __repr__(self):
        return f"Constant risk-free interest rate {self.__r}."