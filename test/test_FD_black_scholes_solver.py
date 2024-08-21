import unittest
import numpy as np
from QFpy.black_scholes import BlackScholesFDSolver
from QFpy.options import Option, EuropeanCallOption
from QFpy.vanilla_option_utils import EuropeanCallValue, EuropeanCallDelta, EuropeanCallGamma, EuropeanPutTheta
from QFpy.volatility import ConstantVol
from QFpy.interest_rate import ConstantRate 

class TestBlackScholesMCSolver(unittest.TestCase):
    
    def setUp(self): 
        self.vol  = ConstantVol(0.01)
        self.rate = ConstantRate(0.05/365)
        self.K = 100 
        self.T = 100 
        underlying_range = [0, 2*self.K]
        time_range = [0,self.T]
        option = EuropeanCallOption(self.K,self.T)
        self.solver = BlackScholesFDSolver(self.vol,self.rate,option,
                                           underlying_range,time_range,
                                           underlying_npoints=200,
                                           time_npoints=200)
    
    
    def test_value(self):
        S = 80 
        t = 90 
        
        
        val = self.solver.get_value(t,S)
        
        exact = EuropeanCallValue(t,S,self.T,self.K,self.vol.vol,self.rate.r,0)
        
        self.assertAlmostEqual(val,exact,places=3)
        
    def test_delta(self):
        S = 80 
        t = 90 
        

        val = self.solver.get_delta(t,S)
        
        exact = EuropeanCallDelta(t,S,self.T,self.K,self.vol.vol,self.rate.r,0)
        
        self.assertAlmostEqual(val,exact,places=3)
    
    def test_gamma(self):
        S = 80 
        t = 90 

        val = self.solver.get_gamma(t,S)
        
        exact = EuropeanCallGamma(t,S,self.T,self.K,self.vol.vol,self.rate.r,0)
        
        self.assertAlmostEqual(val,exact,places=3)
        
if __name__=="__main__":
    unittest.main() 