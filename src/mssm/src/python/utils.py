import numpy as np
import scipy as scp
import math
import warnings
from itertools import permutations,product,repeat
import copy
from ..python.gamm_solvers import cpp_backsolve_tr,compute_S_emb_pinv_det,cpp_chol,cpp_solve_coef,update_scale_edf,compute_Linv,apply_eigen_perm,tqdm,managers,shared_memory,cpp_solve_coefXX,update_PIRLS,correct_coef_step
from ..python.formula import reparam,map_csc_to_eigen,mp
from ..python.exp_fam import Family,Gaussian, Identity

def sample_MVN(n,mu,scale,P,L,LI=None,use=None,seed=None):
    """
    Draw n samples x from multivariate normal with mean ``mu`` and covariance matrix Sigma so that Sigma/scale = LI.T@LI, LI = L^{-1}, and
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

   See REML function below for more details.
   """

   S_emb,_,_ = compute_S_emb_pinv_det(X.shape[1],penalties,"svd")

   # Need pseudo-data only in case of GAM
   z = None
   Wr = None

   if isinstance(family,Gaussian) and isinstance(family.link,Identity):
        # AMM - directly solve for coef
        LP, Pr, coef, code = cpp_solve_coef(y,X,S_emb)

        if code != 0:
            raise ValueError("Forming coefficients for specified penalties was not possible.")
        
        eta = (X @ coef).reshape(-1,1)
        mu = eta
        nH = (X.T@X).tocsc()
   else:
       # GAMM - have to repeat Newton step
       yb = y
       Xb = X

       mu = family.init_mu(y)
       eta = family.link.f(mu)
       
       # First pseudo-dat iteration
       yb,Xb,z,Wr = update_PIRLS(y,yb,mu,eta,X,Xb,family)

       # Solve coef
       LP, Pr, coef, code = cpp_solve_coef(yb,Xb,S_emb)

       if code != 0:
            raise ValueError("Forming coefficients for specified penalties was not possible.")

       # Update eta & mu
       eta = (X @ coef).reshape(-1,1)
       mu = family.link.fi(eta)

       # Compute deviance
       dev = family.deviance(y,mu)

       # And penalized deviance term
       c_pen_dev = dev + coef.T @ S_emb @ coef
       pen_dev = c_pen_dev + 1e7

       # Now repeat until convergence
       for newt_iter in range(50):
        
           # Update pseudo-dat
           yb,Xb,z,Wr = update_PIRLS(y,yb,mu,eta,X,Xb,family)

           LP, Pr, n_coef, code = cpp_solve_coef(yb,Xb,S_emb)
           
           if code != 0:
                raise ValueError("Forming coefficients for specified penalties was not possible.")

           # Update eta & mu
           eta = (X @ n_coef).reshape(-1,1)
           mu = family.link.fi(eta)

           # Update deviance
           dev = family.deviance(y,mu)

           pen_dev = dev + n_coef.T @ S_emb @ n_coef

           # Step-size control:
           dev,pen_dev,mu,eta,coef = correct_coef_step(coef,n_coef,dev,pen_dev,c_pen_dev,family,eta,mu,y,X,len(penalties),S_emb,None,1)

           # Convergence control
           if np.abs(pen_dev - c_pen_dev) < 1e-7*pen_dev:
                break
           
           # Prepare next step
           c_pen_dev = pen_dev

       W = Wr@Wr
       nH = (X.T@W@X).tocsc() 

   # Get edf and optionally estimate scale
   _,_,edf,_,_,scale = update_scale_edf(y,z,eta,Wr,X.shape[0],X.shape[1],LP,None,Pr,None,family,penalties,10)

   if family.twopar:
        llk = family.llk(y,mu,scale)
   else:
        llk = family.llk(y,mu)

   # Now compute REML for candidate
   reml = REML(llk,nH,coef,scale,penalties)

   return reml,LP,Pr,coef,scale,edf,llk
   

def REML(llk,H,coef,scale,penalties):
   """
   Based on Wood (2011). Exact REML for Gaussian GAM, Laplace approximate (Wood, 2016) for everything else.
   Evaluated after applying stabilizing reparameterization discussed by Wood (2011).

   References:

    - Wood, S. N., (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models.
    - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models
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
        # We need to evaluate log(|S_\lambda/\phi|+) after re-parameterization of S_\lambda (so this will be a regular determinant).
        # We have that (https://en.wikipedia.org/wiki/Determinant):
        #   det(S_\lambda * 1/\phi) = (1/\phi)^p * det(S_\lambda)
        # taking logs:
        #    log(det(S_\lambda * 1/\phi)) = log((1/\phi)^p) + log(det(S_\lambda))
        # We know that log(det(S_\lambda)) is insensitive to whether or not we re-parameterize, so
        # we can simply take S_rep/scale and compute log(det()) for that.
        Sdiag = np.power(np.abs((S_rep/scale).diagonal()),0.5)
        PI = scp.sparse.diags(1/Sdiag,format='csc')
        P = scp.sparse.diags(Sdiag,format='csc')

        L,code = cpp_chol(PI@(S_rep/scale)@PI)

        if code == 0:
            ldetSI = (2*np.log((L@P).diagonal()).sum())*Sj_reps[SJ_term_idx[Si][0]].rep_sj
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
  
   lgdetXXS = 2*np.log((L@P).diagonal()).sum()

   # Done
   return reml + lgdetS/2 - lgdetXXS/2 + (Mp*np.log(2*np.pi))/2

def estVp(ep,remls,rGrid):
    """Estimate covariance matrix of log(\lambda). REML scores are used to
    approximate expectation, similar to what was suggested by Greven & Scheipl (2016).

    References:
     - https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices
     - Greven, S., & Scheipl, F. (2016). Comment on: Smoothing Parameter and Model Selection for General Smooth Models

    :param ep: Model estimate log(\lambda), i.e., the expectation over rGrid
    :type ep: [float]
    :param remls: REML score associated with each \lambda candidate in rGrid
    :type remls: [float]
    :param rGrid: A 2d array, holding all \lambda samples considered so far. Each row is one sample
    :type rGrid: [float]
    :return: An estimate of the covariance matrix of log(\lambda) - 2d array of shape len(mp)*len(mp).
    :rtype: [float]
    """
    ws = scp.special.softmax(remls)
    wp = (np.log(rGrid[0]).reshape(-1,1) - ep)
    # Vp = E[(wp-ep)(wp-ep)^T] - see Wikipedia
    Vp = ws[0]*(wp @ wp.T)
    for ridx,r in enumerate(rGrid):
        wp = (np.log(r).reshape(-1,1) - ep)
        Vp += ws[ridx]*(wp @ wp.T)
    
    return Vp

def _compute_VB_corr_terms_MP(family,address_y,address_dat,address_ptr,address_idx,address_datXX,address_ptrXX,address_idxXX,shape_y,shape_dat,shape_ptr,shape_datXX,shape_ptrXX,rows,cols,rPen,r):
   """
   Multi-processing code for Grevel & Scheipl correction for Gaussian additive model - see ``correct_VB`` for details.
   """
   dat_shared = shared_memory.SharedMemory(name=address_dat,create=False)
   ptr_shared = shared_memory.SharedMemory(name=address_ptr,create=False)
   idx_shared = shared_memory.SharedMemory(name=address_idx,create=False)
   dat_sharedXX = shared_memory.SharedMemory(name=address_datXX,create=False)
   ptr_sharedXX = shared_memory.SharedMemory(name=address_ptrXX,create=False)
   idx_sharedXX = shared_memory.SharedMemory(name=address_idxXX,create=False)
   y_shared = shared_memory.SharedMemory(name=address_y,create=False)

   data = np.ndarray(shape_dat,dtype=np.double,buffer=dat_shared.buf)
   indptr = np.ndarray(shape_ptr,dtype=np.int64,buffer=ptr_shared.buf)
   indices = np.ndarray(shape_dat,dtype=np.int64,buffer=idx_shared.buf)
   dataXX = np.ndarray(shape_datXX,dtype=np.double,buffer=dat_sharedXX.buf)
   indptrXX = np.ndarray(shape_ptrXX,dtype=np.int64,buffer=ptr_sharedXX.buf)
   indicesXX = np.ndarray(shape_datXX,dtype=np.int64,buffer=idx_sharedXX.buf)
   y = np.ndarray(shape_y,dtype=np.double,buffer=y_shared.buf)

   X = scp.sparse.csc_array((data,indices,indptr),shape=(rows,cols),copy=False)
   XX = scp.sparse.csc_array((dataXX,indicesXX,indptrXX),shape=(cols,cols),copy=False)

   # Prepare penalties with current lambda candidate r
   for ridx,rc in enumerate(r):
        rPen[ridx].lam = rc

   # Now compute REML - and all other terms needed for correction proposed by Greven & Scheipl (2017)
   S_emb,_,_ = compute_S_emb_pinv_det(X.shape[1],rPen,"svd")
   LP, Pr, coef, code = cpp_solve_coef(y,X,S_emb)

   if code != 0:
       raise ValueError("Forming coefficients for specified penalties was not possible.")
   
   eta = (X @ coef).reshape(-1,1)
   
   # Optionally estimate scale
   if family.twopar:
        _,_,edf,_,_,scale = update_scale_edf(y,None,eta,None,X.shape[0],X.shape[1],LP,None,Pr,None,family,rPen,10)

        llk = family.llk(y,eta,scale)
   else:
        scale = family.scale
        llk = family.llk(y,eta)

   # Now compute REML for candidate
   reml = REML(llk,XX,coef,scale,rPen)
   coef = coef.reshape(-1,1)

   # Form VB, first solve LP{^-1}
   LPinv = compute_Linv(LP,1)
   Linv = apply_eigen_perm(Pr,LPinv)

   # Now collect what we need for the remaining terms
   return Linv,coef,reml,scale,edf,llk

def correct_VB(model,nR = 11,lR = 20,grid_type = 'JJJ',n_c=10,form_t=True,form_t1=False,verbose=False):
    """
    Wood et al. (2016) and Wood (2017) show that when basing conditional versions of model selection criteria or hypothesis
    tests on Vb, which is the co-variance matrix for the conditional posterior of \boldsymbol{\beta} so that
    \boldsymbol{\beta} | y, \lambda ~ N(\hat{\boldsymbol{\beta}},Vb), the tests are severely biased. To correct for this they
    show that uncertainty in \lambda needs to be accounted for. Hence they suggest to base these tests on V, the covariance matrix
    of the unconditional posterior \boldsymbol{\beta} | y ~ N(\hat{\boldsymbol{\beta}},V). They show how to obtain an estimate of V,
    but this requires V_p - an estimate of the covariance matrix of log(\lambda). V_p requires derivatives that are not available
    when using the efs update.

    Greven & Scheipl in their comment to the paper by Wood et al. (2016) show another option to estimate V that does not require V_p,
    based either on forming a mixture approximation or on the total variance property. The latter is implemented below, based on the
    equations for the expectations outlined in their response. A problem of this estimate is that a grid of \lambda values needs to be
    provided covering the prior on \lambda (see Wood 2011 for the relation between smoothness penalties and this prior). For ``mssm`` the
    limits on this are 1e-7 and 1e7. However, as the authors already conclude, covering the entire prior range is not that efficient.
    Hence we provide an alternative way to set-up the grid, based on first forming marginal grids for each \lambda that contain nR equally-spaced
    samples from \lambda/lr to \lambda*lr, while all other lambda values are set to random samples between the prior limits. This neglects quite a bit
    of the prior space. So we use these initial samples to estimate V_p, so that log(\lambda)|y ~ N(log(\hat{\lambda}),V_p,) - see Wood et al. (2016).
    We then repeatedly sample new \lambda vectors from this normal, followed by updating out estimate of the normal (i.e., V_p) given these samples (
    using the REML weights to approximate the expectation for estimating V_p, based on Greven & Scheipl; 2016). Note that until the last sampling step we
    add small values to the diagonal of V_p to promote exploration. The idea is that this should help us to better explore and with less samples, locally
    around \hat{\lambda}, the uncertainty than if we would just sample from a grid. We then also later re-compute the REML weights from this normal and
    then follow the steps outlined by (Greven & Scheipl; 2016) to compute V, rather than computing the approximation suggested by Wood et al. (2016).
    
    This is done when argument grid_type = 'JJJ'. Otherwise, the G&S strategy is employed - forming a grid over the full space (from 1e-7 to 1e7,
    again evaluated for nR equally-spaced values and then permuted for the number of \lambda parameters). Note that
    the latter can get very expensive quite quickly.

    References:

     - Wood, S. N., (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models.
     - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models
     - Greven, S., & Scheipl, F. (2016). Comment on: Smoothing Parameter and Model Selection for General Smooth Models
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
    """

    nPen = len(model.formula.penalties)
    rPen = copy.deepcopy(model.formula.penalties)

    if grid_type == 'GS':
        # Build Full prior grid as discussed by Greven & Scheipl in their comment on Wood et al. (2016)
        rGrid = [np.exp(np.linspace(np.log(1e-7),np.log(1e7),nR)) for _ in range(nPen)]
        rGrid = np.array(list(product(*rGrid)))
    else:
        # Set up grid of nR equidistant values based on marginal grids that cover range from \lambda/lr to \lambda*lr
        # conditional on all estimated penalty values except the current one.
        
        rGrid = []
        for pi,pen in enumerate(rPen):
            
            # Set up marginal grid from \lambda/lr to \lambda*lr
            mGrid = np.exp(np.linspace(np.log(max([1e-7,pen.lam/lR])),np.log(min([1e7,pen.lam*lR])),nR))
            
            # Now create penalty candidates conditional on estimates for all other penalties except current one
            for val in mGrid:
                if abs(val - pen.lam) <= 1e-7:
                    continue
                
                rGrid.append(np.array([val if pii == pi else np.random.choice(np.exp(np.linspace(np.log(max([1e-7,pen2.lam/lR])),np.log(min([1e7,pen2.lam*lR])),nR)),size=1)[0] for pii,pen2 in enumerate(rPen)]))
        
        # Make sure actual estimate is included once.
        rGrid.append(np.array([pen2.lam for pen2 in rPen]))
        rGrid = np.array(rGrid)
            
    y = model.formula.y_flat[model.formula.NOT_NA_flat]
    X = model.get_mmat()
    family = model.family

    orig_scale = family.scale
    if family.twopar:
        _,orig_scale = model.get_pars()

    remls = []
    Vs = []
    coefs = []
    edfs = []
    llks = []
    aics = []

    if X.shape[1] < 2000 and isinstance(family,Gaussian) and isinstance(family.link,Identity) and n_c > 1: # Parallelize grid search
        with managers.SharedMemoryManager() as manager, mp.Pool(processes=n_c) as pool:
            # Create shared memory copies of data, indptr, and indices for X, XX, and y

            # X
            rows, cols, _, data, indptr, indices = map_csc_to_eigen(X)
            shape_dat = data.shape
            shape_ptr = indptr.shape
            shape_y = y.shape

            dat_mem = manager.SharedMemory(data.nbytes)
            dat_shared = np.ndarray(shape_dat, dtype=np.double, buffer=dat_mem.buf)
            dat_shared[:] = data[:]

            ptr_mem = manager.SharedMemory(indptr.nbytes)
            ptr_shared = np.ndarray(shape_ptr, dtype=np.int64, buffer=ptr_mem.buf)
            ptr_shared[:] = indptr[:]

            idx_mem = manager.SharedMemory(indices.nbytes)
            idx_shared = np.ndarray(shape_dat, dtype=np.int64, buffer=idx_mem.buf)
            idx_shared[:] = indices[:]

            #XX
            _, _, _, dataXX, indptrXX, indicesXX = map_csc_to_eigen((X.T@X).tocsc())
            shape_datXX = dataXX.shape
            shape_ptrXX = indptrXX.shape

            dat_memXX = manager.SharedMemory(dataXX.nbytes)
            dat_sharedXX = np.ndarray(shape_datXX, dtype=np.double, buffer=dat_memXX.buf)
            dat_sharedXX[:] = dataXX[:]

            ptr_memXX = manager.SharedMemory(indptrXX.nbytes)
            ptr_sharedXX = np.ndarray(shape_ptrXX, dtype=np.int64, buffer=ptr_memXX.buf)
            ptr_sharedXX[:] = indptrXX[:]

            idx_memXX = manager.SharedMemory(indicesXX.nbytes)
            idx_sharedXX = np.ndarray(shape_datXX, dtype=np.int64, buffer=idx_memXX.buf)
            idx_sharedXX[:] = indicesXX[:]

            # y
            y_mem = manager.SharedMemory(y.nbytes)
            y_shared = np.ndarray(shape_y, dtype=np.double, buffer=y_mem.buf)
            y_shared[:] = y[:]

            args = zip(repeat(family),repeat(y_mem.name),repeat(dat_mem.name),
                       repeat(ptr_mem.name),repeat(idx_mem.name),repeat(dat_memXX.name),
                       repeat(ptr_memXX.name),repeat(idx_memXX.name),repeat(shape_y),
                       repeat(shape_dat),repeat(shape_ptr),repeat(shape_datXX),repeat(shape_ptrXX),
                       repeat(rows),repeat(cols),repeat(rPen),rGrid)
            
            Linvs, coefs, remls, scales, edfs, llks = zip(*pool.starmap(_compute_VB_corr_terms_MP,args))
            aics = -2*np.array(llks) + 2*np.array(edfs)

            if grid_type == "JJJ":
                Linvs = list(Linvs)
                coefs = list(coefs)
                remls = list(remls)
                scales = list(scales)
                edfs = list(edfs)
                llks = list(llks)
                aics = list(aics)

    else: # Better to parallelize inverse computation necessary to obtain Vb
        enumerator = rGrid
        if verbose:
            enumerator = tqdm(rGrid)
        for r in enumerator:

            # Prepare penalties with current lambda candidate r
            for ridx,rc in enumerate(r):
                rPen[ridx].lam = rc
            
            # Now compute REML - and all other terms needed for correction proposed by Greven & Scheipl (2017)
            reml,LP,Pr,coef,scale,edf,llk = compute_reml_candidate_GAMM(family,y,X,rPen)
            coef = coef.reshape(-1,1)

            # Form VB, first solve LP{^-1}
            LPinv = compute_Linv(LP,n_c)
            Linv = apply_eigen_perm(Pr,LPinv)

            # Can already compute first term from correction
            Vb = Linv.T@Linv*scale
            Vb += coef@coef.T

            # and aic under current penalty
            aic = -2*llk + 2*edf

            # Now collect what we need for the remaining terms
            Vs.append(Vb)
            coefs.append(coef)
            remls.append(reml)
            edfs.append(edf)
            llks.append(llk)
            aics.append(aic)

    if grid_type == "JJJ":
        # Iteratively estimate Vp - covariance matrix of log(\lambda) to guide further REML grid sampling
        id_weight = 0.1
        ep = np.log(np.array([pen.lam for pen in model.formula.penalties]).reshape(-1,1))

        # Get first estimate for Vp based on samples collected so far
        Vp = estVp(ep,remls,rGrid) + id_weight*np.identity(len(ep))

        # Now continuously update Vp and generate more REML samples in the process
        n_est = nR
        
        if X.shape[1] < 2000 and isinstance(family,Gaussian) and isinstance(family.link,Identity) and n_c > 1: # Parallelize grid search
            with managers.SharedMemoryManager() as manager, mp.Pool(processes=n_c) as pool:
                # Create shared memory copies of data, indptr, and indices for X, XX, and y

                # X
                rows, cols, _, data, indptr, indices = map_csc_to_eigen(X)
                shape_dat = data.shape
                shape_ptr = indptr.shape
                shape_y = y.shape

                dat_mem = manager.SharedMemory(data.nbytes)
                dat_shared = np.ndarray(shape_dat, dtype=np.double, buffer=dat_mem.buf)
                dat_shared[:] = data[:]

                ptr_mem = manager.SharedMemory(indptr.nbytes)
                ptr_shared = np.ndarray(shape_ptr, dtype=np.int64, buffer=ptr_mem.buf)
                ptr_shared[:] = indptr[:]

                idx_mem = manager.SharedMemory(indices.nbytes)
                idx_shared = np.ndarray(shape_dat, dtype=np.int64, buffer=idx_mem.buf)
                idx_shared[:] = indices[:]

                #XX
                _, _, _, dataXX, indptrXX, indicesXX = map_csc_to_eigen((X.T@X).tocsc())
                shape_datXX = dataXX.shape
                shape_ptrXX = indptrXX.shape

                dat_memXX = manager.SharedMemory(dataXX.nbytes)
                dat_sharedXX = np.ndarray(shape_datXX, dtype=np.double, buffer=dat_memXX.buf)
                dat_sharedXX[:] = dataXX[:]

                ptr_memXX = manager.SharedMemory(indptrXX.nbytes)
                ptr_sharedXX = np.ndarray(shape_ptrXX, dtype=np.int64, buffer=ptr_memXX.buf)
                ptr_sharedXX[:] = indptrXX[:]

                idx_memXX = manager.SharedMemory(indicesXX.nbytes)
                idx_sharedXX = np.ndarray(shape_datXX, dtype=np.int64, buffer=idx_memXX.buf)
                idx_sharedXX[:] = indicesXX[:]

                # y
                y_mem = manager.SharedMemory(y.nbytes)
                y_shared = np.ndarray(shape_y, dtype=np.double, buffer=y_mem.buf)
                y_shared[:] = y[:]
                
                enumerator = range(nR*len(ep))
                if verbose:
                    enumerator = tqdm(enumerator)
                for sp in enumerator:
                    # Generate next \lambda values for which to compute REML, and Vb
                    p_sample = scp.stats.multivariate_normal.rvs(mean=np.ndarray.flatten(ep),cov=Vp,size=n_est)
                    p_sample = np.exp(p_sample)
                    p_sample = p_sample[[np.any(np.all(rGrid==lam,axis=1))==False for lam in p_sample]]

                    if n_est == 1:
                        p_sample = [p_sample]
                    
                    # Now compute reml for new candidates in parallel
                    args = zip(repeat(family),repeat(y_mem.name),repeat(dat_mem.name),
                            repeat(ptr_mem.name),repeat(idx_mem.name),repeat(dat_memXX.name),
                            repeat(ptr_memXX.name),repeat(idx_memXX.name),repeat(shape_y),
                            repeat(shape_dat),repeat(shape_ptr),repeat(shape_datXX),repeat(shape_ptrXX),
                            repeat(rows),repeat(cols),repeat(rPen),p_sample)
                    
                    sample_Linvs, sample_coefs, sample_remls, sample_scales, sample_edfs, sample_llks = zip(*pool.starmap(_compute_VB_corr_terms_MP,args))
                    sample_aics = -2*np.array(sample_llks) + 2*np.array(sample_edfs)

                    Linvs.extend(list(sample_Linvs))
                    scales.extend(list(sample_scales))
                    coefs.extend(list(sample_coefs))
                    remls.extend(list(sample_remls))
                    edfs.extend(list(sample_edfs))
                    llks.extend(list(sample_llks))
                    aics.extend(list(sample_aics))

                    rGrid = np.concatenate((rGrid,p_sample),axis=0)

                    # Update Vp - based on additional REML scores available now
                    id_weight *= 0.99

                    # Last step should not involve identity at all.
                    if sp == (nR-1):
                        id_weight = 0

                    Vp = estVp(ep,remls,rGrid) + id_weight*np.identity(len(ep))
            
        else:
            enumerator = range(nR*len(ep))
            if verbose:
                enumerator = tqdm(enumerator)
            for sp in enumerator:

                # Generate next \lambda values for which to compute REML, and Vb
                p_sample = scp.stats.multivariate_normal.rvs(mean=np.ndarray.flatten(ep),cov=Vp,size=n_est)
                p_sample = np.exp(p_sample)
                p_sample = p_sample[[np.any(np.all(rGrid==lam,axis=1))==False for lam in p_sample]]

                if n_est == 1:
                    p_sample = [p_sample]

                for ps in p_sample:
                    for ridx,rc in enumerate(ps):
                        rPen[ridx].lam = rc
                    
                    reml,LP,Pr,coef,scale,edf,llk = compute_reml_candidate_GAMM(family,y,X,rPen)
                    coef = coef.reshape(-1,1)

                    # Form VB, first solve LP{^-1}
                    LPinv = compute_Linv(LP,n_c)
                    Linv = apply_eigen_perm(Pr,LPinv)

                    # Can already compute first term from correction
                    Vb = Linv.T@Linv*scale
                    Vb += coef@coef.T

                    # and aic under current penalty
                    aic = -2*llk + 2*edf

                    # Collect all necessary objects for G&S correction.
                    Vs.append(Vb)
                    coefs.append(coef)
                    remls.append(reml)
                    edfs.append(edf)
                    llks.append(llk)
                    aics.append(aic)
                    rGrid = np.concatenate((rGrid,ps.reshape(1,-1)),axis=0)

                # Update Vp - based on additional REML scores available now
                id_weight *= 0.99

                # Last step should not involve identity at all.
                if sp == (nR-1):
                    id_weight = 0

                Vp = estVp(ep,remls,rGrid) + id_weight*np.identity(len(ep))

    # Compute weights proposed by Greven & Scheipl (2017)
    ws = scp.special.softmax(remls)
    if grid_type == "JJJ":
        # Use the estimated normal to compute weights instead.
        ws2 = scp.stats.multivariate_normal.logpdf(np.log(rGrid),mean=np.ndarray.flatten(ep),cov=Vp)
        ws = scp.special.softmax(ws2)

    # And "Expected aic - over lambda uncertainty"
    expected_aic = np.sum(ws*aics)

    # Now compute \hat{cov(\boldsymbol{\beta}|y)}
    if X.shape[1] < 2000 and isinstance(family,Gaussian) and isinstance(family.link,Identity) and n_c > 1:
        # E_{p|y}[V_\boldsymbol{\beta}(\lambda)] + E_{p|y}[\boldsymbol{\beta}\boldsymbol{\beta}^T] in Greven & Scheipl (2017)
        Vr1 = ws[0]* ((Linvs[0].T@Linvs[0]*scales[0]) + (coefs[0]@coefs[0].T))

        # E_{p|y}[\boldsymbol{\beta}] in Greven & Scheipl (2017)
        Vr2 = ws[0]*coefs[0] 
        
        # Now sum over remaining r
        for ri in range(1,len(rGrid)):
            Vr1 += ws[ri]*((Linvs[ri].T@Linvs[ri]*scales[ri]) + (coefs[ri]@coefs[ri].T))
            Vr2 += ws[ri]*coefs[ri]

    else:
        Vr1 = ws[0]*Vs[0] 
        Vr2 = ws[0]*coefs[0] 

        for ri in range(1,len(rGrid)):
            Vr1 += ws[ri]*Vs[ri]
            Vr2 += ws[ri]*coefs[ri]

    # Now, Greven & Scheipl provide final estimate =
    # E_{p|y}[V_\boldsymbol{\beta}(\lambda)] + E_{p|y}[\boldsymbol{\beta}\boldsymbol{\beta}^T] - E_{p|y}[\boldsymbol{\beta}] E_{p|y}[\boldsymbol{\beta}]^T
    V = Vr1 - (Vr2@Vr2.T)

    # Check V is full rank - can use LV for sampling as well..
    LV,code = cpp_chol(scp.sparse.csc_array(V))
    if code != 0:
        raise ValueError("Failed to estimate unconditional covariance matrix.")

    # Compute corrected edf (e.g., for AIC; Wood, Pya, & Saefken, 2016)
    total_edf = None
    if form_t or form_t1:
        F = V@((X.T@X)/orig_scale)
        total_edf = F.trace()

    # Compute corrected smoothness bias corrected edf (t1 in section 6.1.2 of Wood, 2017)
    total_edf2 = None
    if form_t1:
        total_edf2 = 2*total_edf - (F@F).trace()

    return V,LV,total_edf,total_edf2,expected_aic