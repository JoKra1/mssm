import numpy as np
import scipy as scp
import math
import warnings
from ..python.gamm_solvers import cpp_backsolve_tr

def sample_MVN(n,mu,scale,P,L,LI=None,use=None,seed=None):
    """
    Draw n samples x from multivariate normal with mean mu and covariance matrix Sigma so that Sigma/scale = LI.T@LI, LI = L^{-1}, and
    finally L@L.T = {Sigma/scale}^{-1}. In other words, L*(1/scale)^{0.5} is the cholesky for the precision matrix corresponding to Sigma.
    Notably, L (and LI) have actually be computed for P@[X.T@X+S_\lambda]@P.T (see Wood \& Fasiolo, 2017), hence for sampling we need to correct
    for permutation matrix ``P``. if ``LI`` is provided, then ``P`` can be omitted and is assumed to have been applied to ``LI already``

    Used to sample the uncorrected posterior \beta|y,\lambda ~ N(\boldsymbol{\beta},(X.T@X+S_\lambda)^{-1}\phi) for a GAMM (see Wood, 2017).

    Based on section 7.4 in Gentle (2009), assuming Sigma is p*p and covariance matrix of uncorrected posterior:
        x = mu + P.T@LI.T*scale^{0.5}@z where z_i ~ N(0,1) for all i = 1,...,p

    Notably, we can rely on the fact of equivalence that:
        L.T*(1/scale)^{0.5} @ P@x = z
    
    ...and then first solve for y in:
        L.T*(1/scale)^{0.5} @ y = z
    
    ...followed by computing:
        y = P@x
        x = P.T@y
    
        
    The latter allows to avoid forming L^{-1} (which unlike L might not benefit from the sparsity preserving permutation P). Hence, if ``LI is None``,
    ``L`` will be used for sampling.

    Often we care only about a handfull of elements in mu (usually the first ones corresponding to "fixed effects'" in a GAMM). In that case we
    can generate x only for this sub-set of interest by only using a row-block of L/LI (all columns remain). Argument ``use`` can be a Numpy array
    containg the indices of elements in mu that should be sampled. Because this only works efficiently when ``LI`` is available an error is raised
    when ``not use is None and LI is None``.

    If ``mu`` is set to any integer (i.e., not a Numpy array/list) it is treated as 0. 
    
    References:
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
     - Gentle, J. (2009). Computational Statistics.
    """
    if L is None and LI is None:
        raise ValueError("Either ``L`` or ``LI`` have to be provided.")
    
    if not L is None and not LI is None:
        warnings.warn("Both ``L`` and ``LI`` were provided, will rely on ``LI``.")

    if not L is None and LI is None and P is None:
        raise ValueError("When sampling with ``L`` ``P`` must be provided.")
    
    if not use is None and LI is None:
        raise ValueError("If ``use`` is not None ``LI`` must be provided.")
    
    # Correct for scale
    if not LI is None:
        Cs = LI.T*math.sqrt(scale)
    else:
        Cs = L.T*math.sqrt(1/scale)
    
    # Sample from N(0,1)
    z = scp.stats.norm.rvs(size=Cs.shape[1]*n,random_state=seed).reshape(Cs.shape[1],n)

    # Sample with L
    if LI is None:
        z = cpp_backsolve_tr(Cs.tocsc(),scp.sparse.csc_array(z)).toarray() # actually y

        if isinstance(mu,int):
            return P.T@z
        
        return mu[:,None] + P.T@z
    
    else:
        # Sample with LI
        if not P is None:
            Cs = P.T@Cs

        if not use is None:
            Cs = Cs[use,:]
        
        if isinstance(mu,int):
            return Cs@z
        
        if not use is None:
            mus = mu[use,None]
        else:
            mus = mu[:,None]

        return mus + Cs@z

