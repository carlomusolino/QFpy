from scipy.interpolate import RegularGridInterpolator
import numpy as np
from scipy.sparse import diags
from typing import Callable

class BlackScholesFDSolver:
    
    def __init__(self,
                 volatility: Callable,
                 interest_rate: Callable,
                 payoff_function: Callable,
                 asymptotic_value: Callable,
                 underlying_range: np.ndarray,
                 time_range: np.ndarray, 
                 underlying_npoints: int = 1000,
                 time_npoints: int = 100):
        
        # Modelled asset 
        self.__payoff = payoff_function
        self.__bc     = asymptotic_value
        
        T0 = time_range[0]
        T1 = time_range[1]
        
        # Domain ranges 
        self.__Srange = underlying_range
        # Bound check on S, no negative equities 
        self.__Srange[0] = np.max(self.__Srange[0], 0)
        # BC assumes S==0 at the lower end 
        if ( self.__Srange[0] > 1e-2 ): 
            print("WARNING: lower bound {self.__Srange[0]} on S is not zero,"+\
                  " the Black-Scholes BC is imposed at S==0."+\
                  " this might lead to inaccuracies")
        
        # We assume that T1 is maturity or that the payoff is 
        # somehow consistent with T1.
        self.__Trange = np.array([T0, T1])
        
        # Domain spacings
        self.__h = (self.__Srange[1] - self.__Srange[0]) / underlying_npoints 
        self.__dt = ((T1-T0) / time_npoints )
        self.__invh = 1./self.__h
        
        # Domains 
        
        # S domain, add h and subtract -h
        # being careful not to have negative equities
        self.__grid   = np.linspace(np.max(self.__Srange[0]-self.__h, 0)
                                    self.__Srange[1]+self.__h,underlying_range+2)
        
        # T domain no need for gzs 
        self.__times  = np.linspace(self.__Trange[0],self.__Trange[1],time_npoints)
        
        # Derivative price
        self.__V = np.zeros(np.meshgrid(self.__times,self.__grid)[0].shape)
        
        # Greeks (no Vega and Rho, these you need to 
        # compute yourself!)
        self.__Delta = np.zeros(self.__V.shape)
        self.__Gamma = np.zeros(self.__V.shape)
        self.__Theta = np.zeros(self.__V.shape)
        
        # BS evolution matrices
        self.__A,self.__B = self.__get_matrices()
        
        # Risk free interest rate (potentially function of time)
        self.__r = interest_rate(self.__times)
        
        # Volatility (potentially function of time and asset price)
        self.__sigma = volatility(self.__times, self.__grid)
        
        # Has this been solved
        self.__has_solution = False 
    
    def __get_Gamma(self):
        if not self.__has_solution:
            self.__solve() 
        
        N = len(self.__grid)
        invh = self.__invh 
        A = np.zeros((N,N))
        V = self.__V 
        
        for i in range(1,N-1):
            A[i,i+1] = 1 
            A[i,i-1] = 1 
            A[i,i]   = -2 
        A[0,0] = 1 
        A[-1,-1] = 1 
        return (invh ** 2 * A @ V )

    def __get_Delta(self):
        if not self.__has_solution:
            self.__solve() 
            
        N = len(self.__grid)
        invh = self.__invh 
        A = np.zeros((N,N))
        V = self.__V 
        
        for i in range(1,N-1):
            A[i,i+1] = 1 
            A[i,i-1] = -1 
        A[0,0] = 1 
        A[-1,-1] = 1 
        return (0.5 * invh * A @ V)
    
    def __get_Theta(self):
        if not self.__has_solution:
            self.__solve() 
            
        V = self.__V 
        dt = self.__dt
        return (np.diff(V,axis=1) / dt)
        
    def __get_matrices(self):
        """Get matrices for Crank-Nicholson solver.

        Returns:
            np.ndarray: A and B, resp. first and second 
                        order derivative operators appearing 
                        on the RHS of the Black-Scholes equation.
        """
        dt = self.__dt
        sigma = self.__sigma
        r = self.__r
        N = len(self.__grid)
        
        # Coefficients
        alpha = 0.25 * dt * (sigma**2 * (np.arange(N)**2) - r * np.arange(N))
        beta = -0.5 * dt * (sigma**2 * (np.arange(N)**2) + r)
        gamma = 0.25 * dt * (sigma**2 * (np.arange(N)**2) + r * np.arange(N))

        # Matrix setup for implicit scheme
        A = np.zeros((N, N))
        B = np.zeros((N, N))

        # Fill the matrices leaving out 
        # a frame of width 1 for BCs 
        for i in range(1, N-1):
            A[i, i-1] = -alpha[i]
            A[i, i] = 1 - beta[i]
            A[i, i+1] = -gamma[i]

            B[i, i-1] = alpha[i]
            B[i, i] = 1 + beta[i]
            B[i, i+1] = gamma[i]

        # Set BCs (untouched)
        A[0, 0] = 1
        A[N-1, N-1] = 1
        B[0, 0] = 1
        B[N-1, N-1] = 1
        
        return A,B
    
    def initialize(self, *args):
        
        # Set final condition
        for i,S in enumerate(self.__grid):
            self.__V[i,-1] = self.__payoff(S,*args)

        # Set boundaries at S = 0 and S = infty 
        for i,t in enumerate(self.__times):
            boundaries = self.__bc(self.__Srange,t,*args)
            self.__V[0,i] = boundaries[0]
            self.__V[-1,i] = boundaries[1]
    
    def __solve(self):
        if self.__has_solution :
            return 
        
        # Read in vars 
        V = self.__V
        A = self.__A 
        B = self.__B 
        
        # Loop backwards in time
        for i in range(len(self.__times)-2, -1, -1):
            # Apply Crank-Nicholson method (implicit solve of linear PDE)
            self.__V[1:-1,i] = np.linalg.solve(A, B @ V[:,i+1])[1:-1]
        
        # Store the fact that we have the solution 
        self.__has_solution = True 
        
        # Compute Greeks
        self.__Gamma = self.__get_Gamma()
        self.__Delta = self.__get_Delta()
        self.__Theta = self.__get_Theta() 
        
        return
    
    def __interp_array(self,t,S, arr):
        _t = self.__times 
        _S = self.__grid 
        RGI = RegularGridInterpolator((_S,_t), arr)
        return RGI((S,t))
    
    def get_value(self, t: np.ndarray = None, S: np.ndarray = None):
        if not self.__has_solution:
            self.__solve()
        if ( t is None and S is None ) : 
            return (self.__times[1:-1,:], self.__grid[1:-1,:], self.__V[1:-1,:])
        elif t is None:
            return self.__interp_array(self.__times[1:-1,:], S, self.__V)
        elif S is None:
            return self.__interp_array(t, self.__grid[1:-1,:], self.__V)
        else:
            return self.__interp_array(t, S, self.__V)
        
    def get_delta(self, t: np.ndarray = None, S: np.ndarray = None):
        if not self.__has_solution:
            self.__solve()
        if ( t is None and S is None ) : 
            return (self.__times[1:-1,:], self.__grid[1:-1,:], self.__Delta[1:-1,:])
        elif t is None:
            return self.__interp_array(self.__times[1:-1,:], S, self.__Delta)
        elif S is None:
            return self.__interp_array(t, self.__grid[1:-1,:], self.__Delta)
        else:
            return self.__interp_array(t, S, self.__Delta)
        
    def get_gamma(self, t: np.ndarray = None, S: np.ndarray = None):
        if not self.__has_solution:
            self.__solve()
        if ( t is None and S is None ) : 
            return (self.__times[1:-1,:], self.__grid[1:-1,:], self.__Gamma[1:-1,:])
        elif t is None:
            return self.__interp_array(self.__times[1:-1,:], S, self.__Gamma)
        elif S is None:
            return self.__interp_array(t, self.__grid[1:-1,:], self.__Gamma)
        else:
            return self.__interp_array(t, S, self.__Gamma)
        
    def get_theta(self, t: np.ndarray = None, S: np.ndarray = None):
        if not self.__has_solution:
            self.__solve()
        if ( t is None and S is None ) : 
            return (self.__times[1:-1,:], self.__grid[1:-1,:], self.__Theta[1:-1,:])
        elif t is None:
            return self.__interp_array(self.__times[1:-1,:], S, self.__Theta)
        elif S is None:
            return self.__interp_array(t, self.__grid[1:-1,:], self.__Theta)
        else:
            return self.__interp_array(t, S, self.__Theta)
    
    
    def __repr__(self):
        return "Black-Scholes FD solver:\n"+\
              f"   time range                      = [{self.__Trange[0]}, {self.__Trange[1]}]\n"+\
              f"   asset price range               = [{self.__Srange[0]}, {self.__Srange[1]}]\n"+\
              f"   number of points in time        = {len(self.__times)}\n"+\
              f"   number of points in asset price = {len(self.__grid)}\n"+\
              f"   time axis spacing               = {self.__dt}\n"+\
              f"   asset price axis spacing        = {self.__h}\n"
              

