import numpy as np 
from scipy.stats import norm

def GetM0(sigma: np.ndarray, beta: np.ndarray):
    """Get 0th moment of the distribution of portfolio returns.
    
    The moment is calculated assuming the return can be expressed 
    at second order as:
    
    .. math::    
    
        \Delta P = \Sum_{i} \alpha_i \Delta x_i + \Sum_i \Sum_j \beta_{i,j} \Delta x_i \Delta x_j
    
    
    With 
    
    .. math::
    
        \beta_{i,j} = \frac{1}{2} S_i S_j~\gamma_{i,j}~\frac{\partial^2 P}{\partial S_i \partial S_j}
    
    Args:
        sigma (np.ndarray): np.ndarray
            Covariance of assets/options forming the portfolio, scaled by the vols. Of shape (N, N).
        beta (np.ndarray): np.ndarray
            Matrix containing cross second derivatives of portfolio value 
            w.r.t. underlying asset values (scaled by asset values), of shape (N, N).

    Returns:
        float: The zeroth-moment of the returns distribution.
    """
    N = beta.shape[0]
    # Treat separately the scalar case
    if type(sigma) is float or N == 1 :
        return beta * sigma  
    # And the full case
    t = 0 
    for i in range(N):
        for j in range(N):
            t += beta[i,j] * sigma[i,j]
    return t

def GetM1(sigma: np.ndarray, alpha: np.ndarray, beta: np.ndarray):
    """
    Calculate the first moment of the distribution of portfolio returns.

    This function computes the first moment using the given covariance matrix, 
    and the coefficients alpha and beta. The first moment is computed as a sum 
    of four terms:

    .. math::
    
        E[\Delta P^2] = \sum_{i,j} \alpha_i \alpha_j \sigma_{ij}
              + \sum_{i,j,k,l} \beta_{ij} \beta_{kl} ( \sigma_{ij} \sigma_{kl}
              + \sigma_{ik} \sigma_{jl} + \sigma_{il} \sigma_{jk} )

    Args:
        sigma (np.ndarray): np.ndarray
            Covariance matrix of the assets/options forming the portfolio, of shape (N, N).
        alpha (np.ndarray): np.ndarray
            Array of portfolio deltas, scaled by the asset price. Of shape (N,).
        beta (np.ndarray): np.ndarray
            Matrix containing cross second derivatives of portfolio value 
            with respect to underlying asset values (scaled by asset values), of shape (N, N).

    Returns:
        float: The first moment (mean) of the returns distribution.
    """
    N = len(alpha)
    # Treat separately the scalar case
    if type(alpha) is float or N == 1 :
        return (alpha**2 + 3 * beta**2 * sigma) * sigma
    # And the full case
    t1 = 0
    t2 = 0 
    t3 = 0 
    t4 = 0
    for i in range(N):
        for j in range(N):
            t1 += alpha[i] * alpha[j] * sigma[i,j]
            for k in range(N):
                for l in range(N):
                    t2 += beta[i,j]*beta[k,l] * sigma[i,j] * sigma[k,l]
                    t3 += beta[i,j]*beta[k,l] * sigma[i,k] * sigma[j,l]
                    t4 += beta[i,j]*beta[k,l] * sigma[i,l] * sigma[j,k]
    return t1 + t2 + t3 + t4

def GetLinearM1(sigma: np.ndarray, alpha: np.ndarray):
    """
    Calculate the first moment of the distribution of portfolio returns.

    This function computes the first moment using the given covariance matrix, 
    and the coefficients alpha and beta. The first moment is computed as a sum 
    of four terms:

    .. math::
    
        E[\Delta P^2] = \sum_{i,j} \alpha_i \alpha_j \sigma_{ij}

    Args:
        sigma (np.ndarray): np.ndarray
            Covariance matrix of the assets/options forming the portfolio, scaled by the vols. Of shape (N, N).
        alpha (np.ndarray): np.ndarray
            Array of portfolio deltas, scaled by the asset price. Of shape (N,).

    Returns:
        float: The first moment (mean) of the returns distribution.
    """
    N = len(alpha)
    # Treat separately the scalar case
    if type(alpha) is float or N == 1 :
        return alpha**2  * sigma
    # And the full case
    t = 0
    for i in range(N):
        for j in range(N):
            t += alpha[i] * alpha[j] * sigma[i,j]

    return t


def GetM2(sigma: np.ndarray, alpha: np.ndarray, beta: np.ndarray):
    """
    Calculate the second moment (variance) of the distribution of portfolio returns.

    This function evaluates the second moment, which is a measure of the 
    variance in the portfolio's returns, based on the covariance matrix, and the 
    coefficients alpha and beta. The second moment is computed as a sum of terms 
    involving both linear and quadratic components of the portfolio's returns.

    .. math::

        E[\Delta P^3] = 3 \sum_{i,j,k,l} \alpha_i \alpha_j \beta_{kl} 
        (\sigma_{ij} \sigma_{kl} + \sigma_{ik} \sigma_{jl} + \sigma_{il} \sigma_{jk}) 
        + \sum_{i1, i2, ..., i6} \beta_{i1i2} \beta_{i3i4} \beta_{i5i6} 
        \sum_{\text{permutations}} \sigma_{k1k2} \sigma_{k3k4} \sigma_{k5k6}

    Args:
        sigma (np.ndarray): np.ndarray
            Covariance matrix of the assets/options forming the portfolio, of shape (N, N).
        alpha (np.ndarray): np.ndarray
            Array of portfolio deltas, scaled by the asset price. Of shape (N,).
        beta (np.ndarray): np.ndarray
            Matrix containing cross second derivatives of portfolio value 
            with respect to underlying asset values (scaled by asset values), of shape (N, N).

    Returns:
        float: The second moment (variance) of the returns distribution.
    """
    N = len(alpha)
    # Treat separately the scalar case
    if type(alpha) is float or N == 1:
        return (9 * alpha**2 * beta + 15 * beta**3 * sigma) * sigma**2
    
    # And the full case
    t1 = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    t1 += alpha[i] * alpha[j] * beta[k,l] * ( sigma[i,j] * sigma[k,l] +\
                                                             sigma[i,k] * sigma[j,l] +\
                                                             sigma[i,l] * sigma[j,k] )
    t4 = 0      
    for i1 in range(N):
        for i2 in range(N):
            for i3 in range(N):
                for i4 in range(N):
                    for i5 in range(N):
                        for i6 in range(N):
                            indices = [i1,i2,i3,i4,i5,i6]
                            
                            # Compute Q 
                            Q = 0
                            terns = (((0,4),(1,3),(2,5)),\
                                    ((0,2),(1,3),(4,5)),\
                                    ((0,3),(1,4),(2,5)),\
                                    ((0,4),(1,2,),(3,5)),\
                                    ((0,5),(1,2),(3,4)),\
                                    ((0,4),(1,5),(2,3)),\
                                     ((0,5),(1,4),(2,3)),\
                                     ((0,1),(2,3),(4,5)),\
                                     ((0,1),(2,4),(3,5)),\
                                     ((0,3),(1,5),(2,4)),\
                                     ((0,1),(2,5),(3,4)),\
                                     ((0,2),(1,4),(3,5)),\
                                     ((0,5),(1,3),(2,4)),\
                                     ((0,3),(1,2),(4,5)),\
                                     ((0,2),(1,5),(3,4)) )
                            for tern in terns:
                                k1 = indices[tern[0][0]]
                                k2 = indices[tern[0][1]]
                                k3 = indices[tern[1][0]]
                                k4 = indices[tern[1][1]]
                                k5 = indices[tern[2][0]]
                                k6 = indices[tern[2][1]]
                                Q += sigma[k1,k2] * sigma[k3,k4] * sigma[k5,k6]
                            t4 += beta[i1,i2] * beta[i3,i4] * beta[i5,i6] * Q
    return 3 * t1 + t4 
    
def GetMu(sigma: np.ndarray, alpha: np.ndarray, beta: np.ndarray, linear: bool = False):
    if linear:
        return 0
    else:
        return GetM0(sigma,beta)

def GetSigmaSq(sigma: np.ndarray, alpha: np.ndarray, beta: np.ndarray, linear: bool = False):
    if linear:
        return GetLinearM1(sigma,alpha)
    else:
        mu = GetMu(sigma,alpha,beta)
        M1 = GetM1(sigma,alpha,beta)
        return M1 - mu**2 

def GetXi(sigma: np.ndarray, alpha: np.ndarray, beta: np.ndarray, linear: bool = False):
    if linear:
        return 0
    else:
        mu = GetMu(sigma,alpha,beta)
        M1 = GetM1(sigma,alpha,beta)
        M2 = GetM2(sigma,alpha,beta)
        
        return ( M2 - 3*M1*mu + 2*mu**3 ) / (M1 - mu**2)**(3/2)

def GetCornishFisherPercentile(k: float, sigma: np.ndarray, alpha: np.ndarray, beta: np.ndarray):
    """Compute VaR using a quadratic model.
    
    The percentile of the returns distribution is evaluated
    according to a Corner-Fisher asymptotic expansion.

    Args:
        k (float): Percentile
        sigma (np.ndarray): Correlation matrix (scaled by vols).
        alpha (np.ndarray): Portfolio deltas (scaled by asset prices).
        beta (np.ndarray): Portfolio gammas (scaled by asset prices and divided by 2).

    Returns:
        float: The (1-k)-percent VaR of the portfolio.
    """
    
    
    if (type(beta) is float and abs(beta) < 1e-12) or ( np.all(np.abs(beta)<1e-12) ):
        sigma = np.sqrt(GetSigmaSq(sigma,alpha,beta,True))
        return norm.ppf(k) * sigma 
    else:
        mu = GetMu(sigma,alpha,beta)
        s = np.sqrt(GetSigmaSq(sigma,alpha,beta))
        xi = GetXi(sigma,alpha,beta)
        
        z = norm.ppf(k)
        
        w = z + 1./6.*(z**2-1)*xi
        
        return mu + w * s

def GetLinearPercentile(k: float, sigma: np.ndarray, alpha: np.ndarray):
    """Compute VaR using a linear model.
    
    The percentile of the returns distribution is evaluated
    assuming (log-)normal returns.

    Args:
        k (float): Percentile
        sigma (np.ndarray): Correlation matrix (scaled by vols).
        alpha (np.ndarray): Portfolio deltas (scaled by asset prices).

    Returns:
        float: The (1-k)-percent VaR of the portfolio.
    """
    sigma = np.sqrt(GetSigmaSq(sigma,alpha,np.ones(sigma.shape),True))
    return norm.ppf(k) * sigma 