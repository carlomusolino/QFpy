import unittest
import numpy as np
from QFpy.black_scholes import BlackScholesMCSolver
from QFpy.options import Option, EuropeanPutOption
from QFpy.vanilla_option_utils import EuropeanPutValue, EuropeanPutDelta, EuropeanPutGamma, EuropeanPutTheta
from QFpy.volatility import ConstantVol
from QFpy.interest_rate import ConstantRate 

class TestBlackScholesMCSolver(unittest.TestCase):
    
    def setUp(self): 
        self.vol  = ConstantVol(0.01)
        self.rate = ConstantRate(0.05/365)
        self.K = 100 
        self.T = 100 
        option = EuropeanPutOption(self.K,self.T)
        self.solver = BlackScholesMCSolver(self.vol,self.rate,option)
    
    
    def test_value(self):
        S = 80 
        t = 90 
        
        dt = 1 
        ntrials = 1000 
        (val, sigma) = self.solver.get_value(t,S,dt,ntrials)
        
        exact = EuropeanPutValue(t,S,self.T,self.K,self.vol.vol,self.rate.r,0)
        
        self.assertAlmostEqual(val,exact,delta=1.96*sigma)
        
    def test_delta(self):
        S = 80 
        t = 90 
        
        dt = 1 
        ntrials = 1000 
        (val, sigma) = self.solver.get_delta(t,S,dt,ntrials)
        
        exact = EuropeanPutDelta(t,S,self.T,self.K,self.vol.vol,self.rate.r,0)
        
        self.assertAlmostEqual(val,exact,delta=1.96*sigma)
    
    def test_gamma(self):
        S = 80 
        t = 90 
        
        dt = 1 
        ntrials = 1000 
        (val, sigma) = self.solver.get_gamma(t,S,dt,ntrials)
        
        exact = EuropeanPutGamma(t,S,self.T,self.K,self.vol.vol,self.rate.r,0)
        
        self.assertAlmostEqual(val,exact,delta=1.96*sigma)
        
if __name__=="__main__":
    unittest.main() 