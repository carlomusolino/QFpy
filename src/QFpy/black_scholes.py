from scipy.interpolate import RegularGridInterpolator
import numpy as np
from scipy.sparse import diags
from typing import Callable
from scipy.integrate import cumtrapz 

from QFpy.MC_utils import LogNormalStep
from QFpy.typing_utils import is_scalar 
from QFpy.options import Option, EuropeanCallOption, EuropeanPutOption, AmericanCallOption, AmericanPutOption, AsianCallOption, AsianPutOption

def __divides_exactly(a, b, tol=1e-9):
    if abs(a) < tol or abs(b) < tol:
        return False  # Avoid division by very small numbers
    
    return abs(a % b) < tol 

class BlackScholesFDSolver:
    
    def __init__(self,
                 volatility: Callable,
                 interest_rate: Callable,
                 option: Option,
                 underlying_range: np.ndarray,
                 time_range: np.ndarray, 
                 underlying_npoints: int = 1000,
                 time_npoints: int = 100):
        
        # Modelled asset 
        self.__option = option
        
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
        self.__grid   = np.linspace(np.max(self.__Srange[0]-self.__h, 0),
                                    self.__Srange[1]+self.__h,underlying_npoints+2)
        
        # T domain no need for gzs 
        self.__times  = np.linspace(self.__Trange[0],self.__Trange[1],time_npoints)
        
        # Derivative price
        self.__V = np.zeros(np.meshgrid(self.__times,self.__grid)[0].shape)
        
        # Greeks (no Vega and Rho, these you need to 
        # compute yourself!)
        self.__Delta = np.zeros(self.__V.shape)
        self.__Gamma = np.zeros(self.__V.shape)
        self.__Theta = np.zeros(self.__V.shape)
        
        # Risk free interest rate (potentially function of time)
        self.__r = interest_rate
        
        # Volatility (potentially function of time and asset price)
        self.__sigma = volatility
        
        # Impose boundary and final conditions 
        self.__initialize() 
        
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
        
    def __get_matrices(self, t):
        """Get matrices for Crank-Nicholson solver.

        Returns:
            np.ndarray: A and B, resp. first and second 
                        order derivative operators appearing 
                        on the RHS of the Black-Scholes equation.
        """
        dt    = self.__dt
        sigma = self.__sigma(self.__grid, t)
        r     = self.__r(t)
        N     = len(self.__grid)
        
        # Coefficients
        alpha = 0.25 * dt * (sigma**2 * (np.arange(N)**2) - r * np.arange(N))
        beta  = -0.5 * dt * (sigma**2 * (np.arange(N)**2) + r)
        gamma = 0.25 * dt * (sigma**2 * (np.arange(N)**2) + r * np.arange(N))

        # Matrix setup for implicit scheme
        A = np.zeros((N, N))
        B = np.zeros((N, N))

        # Fill the matrices leaving out 
        # a frame of width 1 for BCs 
        for i in range(1, N-1):
            A[i, i-1] = -alpha[i]
            A[i, i]   = 1 - beta[i]
            A[i, i+1] = -gamma[i]

            B[i, i-1] = alpha[i]
            B[i, i]   = 1 + beta[i]
            B[i, i+1] = gamma[i]

        # Set BCs (untouched)
        A[0, 0]     = 1
        A[N-1, N-1] = 1
        B[0, 0]     = 1
        B[N-1, N-1] = 1
        
        return A,B
    
    def __initialize(self):
        # Set final condition
        for i,S in enumerate(self.__grid):
            self.__V[i,-1] = self.__option.payoff(S,False)

        # Set boundaries at S = 0 and S = infty 
        for i,t in enumerate(self.__times):
            boundaries = self.__option.asymptotic_value(self.__Srange,t,self.__r)
            self.__V[0,i]  = boundaries[0]
            self.__V[-1,i] = boundaries[1]
    
    def __solve(self):
        if self.__has_solution :
            return 
        
        # Read in vars 
        V = self.__V

        # Loop backwards in time
        for i in range(len(self.__times)-2, -1, -1):
            # Get the matrices 
            A,B = self.__get_matrices(self.__times[i])
            # Apply Crank-Nicholson method (implicit solve of linear PDE)
            self.__V[1:-1,i] = self.__option.exercise(np.linalg.solve(A, B @ V[:,i+1])[1:-1], 
                                                          self.__grid[1:-1], 
                                                          self.__times[i])
        
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
              f"   option                          = {self.__option}\n"+\
              f"   underlying vol                  = {self.__sigma}\n"+\
              f"   risk-free interest rate         = {self.__r}\n"+\
              f"   time range                      = [{self.__Trange[0]}, {self.__Trange[1]}]\n"+\
              f"   asset price range               = [{self.__Srange[0]}, {self.__Srange[1]}]\n"+\
              f"   number of points in time        = {len(self.__times)}\n"+\
              f"   number of points in asset price = {len(self.__grid)}\n"+\
              f"   time axis spacing               = {self.__dt}\n"+\
              f"   asset price axis spacing        = {self.__h}\n"

class BlackScholesMCSolver:
    
    def __init__(self,
                 volatility: Callable,
                 interest_rate: Callable,
                 option: Option,
                 n_underlying: int = 1,
                 rho: np.ndarray = np.array([1])):
        self.__vol = volatility 
        self.__r   = interest_rate 
        
        self.__option = option 

        self.__n_underlying = n_underlying 
        
        self.__T = self.__option.T 
        
        
    def __shoot_single_asset(self, t, T, dt, S0):
        N = int((T-t)/dt)
        S = [S0]
        _t = t 
        _rates = [self.__r(t)]
        _vols = [self.__vol(S0,t)]
        _times = [t]
        for i in range(1,N):
            _t += dt 
            vol = self.__vol(S[i-1], _t)
            r   = self.__r(_t)
            S.append(LogNormalStep(S[i-1],dt,vol,r))
            _rates.append(r)
            _times.append(_t)
            _vols.append(vol)
        return (S,_rates,_vols,_times)
    
    def __shoot_multiple_assets(self, t, T, dt, S0):
        N = int((T-t)/dt)
        S = [S0]
        _t = t 
        _rates = [self.__r(t)]
        _vols = [self.__vol(S0,t)]
        _times = [t]
        rng = np.random.default_rng()
        for i in range(1,N):
            _t += dt 
            vol = np.array([self.__vol[j](S[i-1], _t) for j in range(self.__n_underlying)])
            r   = self.__r(_t)
            dX = rng.multivariate_normal(0, self.__rho, size=self.__n_underlying)
            S.append(S[i-1] * np.exp( (r - 0.5*vol**2) * dt + vol * dX * np.sqrt(dt) ) )
            _rates.append(r)
            _times.append(_t)
            _vols.append(vol)
        return (S,_rates,_vols,_times)

    def __discount(self, S, r, t):
        return S * np.exp(-cumtrapz(r,t))
    
    def __get_value_single_asset(self, t: float, S: float, dt: float, ntrials: int):
        Vlist = [] 
        for i in range(ntrials):
            Shist, rhist, volhist, times = self.__shoot_single_asset(t,self.__T,dt,S)
            Vlist.append( self.__discount(self.__option.payoff(Shist, 
                                                                   True,
                                                                   rhist,
                                                                   volhist,
                                                                   times), 
                                          rhist, 
                                          times) )
        return (np.mean(np.array(Vlist)), np.std(np.array(Vlist),ddof=1) )

    def __get_value_multiple_assets(self,
                                    t: float,
                                    S: np.ndarray,
                                    dt: float,
                                    ntrials: int):
        Vlist = []
        for i in range(ntrials):
            Shist, rhist, volhist, times = self.__shoot_multiple_assets(t,self.__T,dt,S)
            Vlist.append(self.__option.payoff(Shist,
                                                  True, 
                                                  rhist, 
                                                  volhist, 
                                                  times), 
                         rhist, 
                         times)
        return (np.mean(np.array(Vlist)), np.std(np.array(Vlist),ddof=1) ) 
    
    def __get_value(self,t,S,dt,ntrials):
        if self.__n_underlying == 1:
            return self.__get_value_single_asset(t,S,dt,ntrials)
        else:
            return self.__get_value_multiple_assets(t,S,ntrials)
    
    def get_value(self, t: np.ndarray, S: np.ndarray, dt: float, ntrials: int ): 
        if is_scalar(t) and is_scalar(S):
            return self.__get_value(t,S,dt,ntrials)
        elif not is_scalar(t) and is_scalar(S):
            if t.ndim > 1:
                raise ValueError("If S is scalar t must be rank 1.")
            mean   = np.zeros(len(t))
            stddev = np.zeros(len(t))
            for i,tt in enumerate(t):
                mu,sigma  = self.__get_value(tt,S,dt,ntrials)
                mean[i]   = mu 
                stddev[i] = sigma
            return mean,stddev  
            
        elif not is_scalar(S) and is_scalar(t):
            if S.ndim > 1:
                raise ValueError("If t is scalar S must be rank 1.")
            mean   = np.zeros(len(S))
            stddev = np.zeros(len(S))
            for i,s in enumerate(S):
                mu,sigma = self.__get_value(t,s,dt,ntrials)
                mean[i] = mu 
                stddev[i] = sigma
            return mean,stddev    
        else:
            if ( not S.ndim == 2 ) or ( not t.ndim == 2):
                raise ValueError("If both S and t are arrays they must be a meshgrid.")
            rows, cols = S.shape
            mean   = np.zeros_like(S,dtype=float)
            stddev = np.zeros_like(S,dtype=float)
            for i in range(rows):
                for j in range(cols):
                    mu,sigma = self.__get_value(t[i,j],S[i,j],dt,ntrials)
                    mean[i,j] = mu 
                    stddev[i,j] = sigma
            return mean,stddev
    
    def __repr__(self):
        return "Black-Scholes MC solver:\n"+\
              f"   option                      = {self.__option}\n"+\
              f"   underlying vol              = {self.__vol}\n"+\
              f"   risk-free interest rate     = {self.__r}\n"+\
              f"   number of underlying assets = {self.__n_underlying}"