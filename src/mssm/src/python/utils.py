import numpy as np
import scipy as scp
import math
import warnings
from ..python.gamm_solvers import cpp_backsolve_tr,compute_S_emb_pinv_det,cpp_chol,cpp_solve_coef,update_scale_edf
from ..python.formula import reparam

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


def compute_reml_candidate_GAMM(family,y,X,penalties):
   """
   Allows to evaluate REML criterion (e.g., Wood, 2011; Wood, 2016) efficiently for
   a set of \lambda values.

   Used for computing the correction applied to the edf for the GLRT - based on Wood (2017) and Wood et al., (2016).

   See REML function below for morre details.
   """
   S_emb,_,_ = compute_S_emb_pinv_det(X.shape[1],penalties,"svd")
   LP, Pr, coef, code = cpp_solve_coef(y,X,S_emb)

   if code != 0:
       raise ValueError("Forming coefficients for specified penalties was not possible.")
   
   eta = (X @ coef).reshape(-1,1)
   
   # Optionally estimate scale
   if family.twopar:
        _,_,_,_,_,scale = update_scale_edf(y,None,eta,None,X.shape[0],X.shape[1],LP,None,Pr,None,family,penalties,10)

        llk = family.llk(y,eta,scale)
   else:
        scale = family.scale
        llk = family.llk(y,eta)

   # Now compute REML for candidate
   reml = REML(llk,(X.T@X).tocsc(),coef,scale,penalties)
   return reml,LP,Pr,coef,scale
   

def REML(llk,H,coef,scale,penalties):
   """
   Based on Wood (2011). Exact REML for Gaussian GAM, Laplace approximate (Wood, 2016) for everything else.
   Evaluated after applying stabilizing reparameterization discussed by Wood (2011).
   """ 

   # Compute S_\lambda before any re-parameterization
   S_emb,_,_ = compute_S_emb_pinv_det(len(coef),penalties,"svd")

   # Re-parameterize as shown in Wood (2011) to enable stable computation of log(|S_\lambda|+)
   Sj_reps,S_reps,SJ_term_idx,S_idx,S_coefs,Q_reps,_,Mp = reparam(None,penalties,None,option=4)
   
   #if not X is None:
   # S_emb = None
   # Q_emb = None
   # c_idx = S_idx[0]
   # for Si,(S_rep,S_coef) in enumerate(zip(S_reps,S_coefs)):
   #     for _ in range(Sj_reps[SJ_term_idx[Si][0]].rep_sj):
   #         Q_emb,_ = embed_in_S_sparse(*translate_sparse(Q_reps[Si]),Q_emb,len(coef),S_coef,c_idx)
   #         S_emb,c_idx = embed_in_S_sparse(*translate_sparse(S_rep),S_emb,len(coef),S_coef,c_idx)
   # #print(Q_emb @ S_emb @ QT_emb - S_emb1)
   # Xq = X@Q_emb
   # Hq = (Xq.T@Xq).tocsc()
   # coef = coef@Q_emb

   # Now we can start evaluating the first terms of 2l_r in Wood (2011). We divide everything by 2
   # to get l_r.
   reml = llk - (coef.T @ S_emb @ coef)/scale/2
   
   # Now we need to compute log(|S_\lambda|+), Wood shows that after the re-parameterization log(|S_\lambda|)
   # can be computed separately from the diagonal or R if Q@R=S_reps[i] for all terms i. Below we compute from
   # the diagonal of the cholesky of the term specific S_reps[i], applying conditioning as shown in Appendix B of Wood (2011).
   lgdetS = 0
   for Si,S_rep in enumerate(S_reps):

        Sdiag = np.power(np.abs(S_rep.diagonal()),0.5)
        PI = scp.sparse.diags(1/Sdiag,format='csc')
        P = scp.sparse.diags(Sdiag,format='csc')

        L,code = cpp_chol(PI@S_rep@PI)
        
        if code == 0:
            ldetSI = np.log((L@P).power(2).diagonal()).sum()*Sj_reps[SJ_term_idx[Si][0]].rep_sj
        else:
            warnings.warn("Cholesky for log-determinant to compute REML failed. Falling back on QR.")
            R = np.linalg.qr(S_rep.toarray(),mode='r')
            ldetSI = np.log(np.abs(R.diagonal())).sum()*Sj_reps[SJ_term_idx[Si][0]].rep_sj
        
        lgdetS += ldetSI
        
   # Now log(|H+S_\lambda|)... Wood (2011) shows stable computation based on QR decomposition, but
   # we will generally not be able to compute a QR decomposition of X so that X.T@X=H efficiently.
   # Hence, we simply rely on the cholesky (again pre-conditioned) used for fitting (based on S_\lambda before
   # re-parameterization).
   H_pen = H/scale + S_emb/scale

   Sdiag = np.power(np.abs(H_pen.diagonal()),0.5)
   PI = scp.sparse.diags(1/Sdiag,format='csc')
   P = scp.sparse.diags(Sdiag,format='csc')
   L,code = cpp_chol(PI@H_pen@PI)

   if code != 0:
       raise ValueError("Failed to compute REML.")
  
   lgdetXXS = np.log((L@P).power(2).diagonal()).sum()

   # Done
   return reml + lgdetS/2 - lgdetXXS/2 + Mp/2*np.log(2*np.pi)