import numpy as np
import scipy as scp
import math
import warnings
from itertools import permutations,product,repeat
import copy
from ..python.gamm_solvers import cpp_backsolve_tr,compute_S_emb_pinv_det,cpp_chol,cpp_solve_coef,update_scale_edf,compute_Linv,apply_eigen_perm,tqdm,managers,shared_memory,cpp_solve_coefXX,update_PIRLS,correct_coef_step,update_coef_gen_smooth,cpp_cholP,update_coef_gammlss,compute_lgdetD_bsb,calculate_edf,compute_eigen_perm,cpp_dChol
from ..python.formula import reparam,map_csc_to_eigen,mp
from ..python.exp_fam import Family,Gaussian, Identity,GAMLSSFamily,GENSMOOTHFamily

def sample_MVN(n,mu,scale,P,L,LI=None,use=None,seed=None):
    """
    Draw ``n`` samples from multivariate normal with mean :math:`\\boldsymbol{\mu}` (``mu``) and covariance matrix :math:`\\boldsymbol{\Sigma}`.
    
    :math:`\\boldsymbol{\Sigma}` does not need to be provided. Rather the function expects either ``L`` (:math:`\mathbf{L}` in what follows) or ``LI`` (:math:`\mathbf{L}^{-1}` in what follows) and ``scale`` (:math:`\phi` in what follows).
    These relate to :math:`\\boldsymbol{\Sigma}` so that :math:`\\boldsymbol{\Sigma}/\phi = \mathbf{L}^{-T}\mathbf{L}^{-1}` or :math:`\mathbf{L}\mathbf{L}^T = [\\boldsymbol{\Sigma}/\phi]^{-1}`
    so that :math:`\mathbf{L}*(1/\phi)^{0.5}` is the Cholesky of the precision matrix of :math:`\\boldsymbol{\Sigma}`.

    Notably, for models available in ``mssm`` ``L`` (and ``LI``) have usually be computed for a permuted matrix, e.g., :math:`\mathbf{P}[\mathbf{X}^T\mathbf{X} + \mathbf{S}_{\lambda}]\mathbf{P}^T` (see Wood \& Fasiolo, 2017).
    Hence for sampling we often need to correct for permutation matrix :math:`\mathbf{P}` (``P``). if ``LI`` is provided, then ``P`` can be omitted and is assumed to have been used to un-pivot ``LI`` already.

    Used for example sample the uncorrected posterior :math:`\\boldsymbol{\\beta} | \mathbf{y}, \\boldsymbol{\lambda} \sim N(\\boldsymbol{\\mu} = \hat{\\boldsymbol{\\beta}},[\mathbf{X}^T\mathbf{X} + \mathbf{S}_{\lambda}]^{-1}\phi)` for a GAMM (see Wood, 2017).
    Based on section 7.4 in Gentle (2009), assuming :math:`\\boldsymbol{\Sigma}` is :math:`p*p` and covariance matrix of uncorrected posterior, samples :math:`\\boldsymbol{\\beta}` are then obtained by computing:

    .. math::

        \\boldsymbol{\\beta} = \hat{\\boldsymbol{\\beta}} + [\mathbf{P}^T \mathbf{L}^{-T}*\phi^{0.5}]\mathbf{z}\ \\text{where}\ z_i \sim N(0,1)\ \\forall i = 1,...,p

    Alternatively, relying on the fact of equivalence that:

    .. math::

        [\mathbf{L}^T*(1/\phi)^{0.5}]\mathbf{P}[\\boldsymbol{\\beta} - \hat{\\boldsymbol{\\beta}}] = \mathbf{z}
    
    we can first solve for :math:`\mathbf{y}` in:

    .. math::

        [\mathbf{L}^T*(1/\phi)^{0.5}] \mathbf{y} = \mathbf{z}
    
    followed by computing:

    .. math::

        \mathbf{y} = \mathbf{P}[\\boldsymbol{\\beta} - \hat{\\boldsymbol{\\beta}}]

        \\boldsymbol{\\beta} = \hat{\\boldsymbol{\\beta}} + \mathbf{P}^T\mathbf{y}
    
        
    The latter avoids forming :math:`\mathbf{L}^{-1}` (which unlike :math:`\mathbf{L}` might not benefit from the sparsity preserving permutation :math:`\mathbf{P}`). If ``LI is None``,
    ``L`` will thus be used for sampling as outlined in these alternative steps.

    Often we care only about a handfull of elements in ``mu`` (e.g., the first ones corresponding to "fixed effects'" in a GAMM). In that case we
    can generate samles only for this sub-set of interest by only using a sub-block of rows of :math:`\mathbf{L}` or :math:`\mathbf{L}^{-1}` (all columns remain). Argument ``use`` can be a ``np.array``
    containg the indices of elements in ``mu`` that should be sampled. Because this only works efficiently when ``LI`` is available an error is raised when ``not use is None and LI is None``.

    If ``mu`` is set to **any integer** (i.e., not a Numpy array/list) it is automatically treated as 0. For :class:`mssm.models.GAMMLSS` or :class:`mssm.models.GSMM` models, ``scale`` can be set to 1.
    
    References:

     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
     - Gentle, J. (2009). Computational Statistics.

    :param n: Number of samples to generate
    :type n: int
    :param mu: mean of normal distribution as described above
    :type mu: np.array
    :param scale: scaling parameter of covariance matrix as described above
    :type scale: float
    :param P: Permutation matrix, optional.
    :type P: scp.sparse.csc_array
    :param L: Cholesky of precision of scaled covariance matrix as described above.
    :type L: scp.sparse.csc_array
    :param LI: Inverse of cholesky factor of precision of scaled covariance matrix as described above.
    :type LI: scp.sparse.csc_array, optional
    :param use: Indices of parameters in ``mu`` for which to generate samples, defaults to None in which case all parameters will be sampled
    :type use: [int], optional
    :param seed: Seed to use for random sample generation, defaults to None
    :type seed: int, optional
    :raises ValueError: In case neither ``LI`` nor ``L`` are provided.
    :raises ValueError: In case ``L`` is provided but ``P`` is not.
    :raises ValueError: In case ``use`` is provided but ``LI`` is not. 
    :return: Samples from multi-variate normal distribution. In case ``use`` is not provided, the returned array will be of shape ``(p,n)`` where ``p==LI.shape[1]``. Otherwise, the returned array will be of shape ``(len(use),n)``.
    :rtype: np.array
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

def forward_hessian(coef,llkfun,*llkgargs,**llkkwargs):
    """Generic function to approximate the hessian of ``llkfun`` at ``coef``.

    ``llkfun`` is called as ``llkfun(coef,*llkargs)`` - so all additional arguments have to be passed via the latter.

    Uses finite differences with fixed value for epsilon, based on chapter 5.5.2 in Wood (2015).

    References:
     - S. Wood (2015). Core Statistics

    :param coef: Current estimate of coefficients of which ``llkfun`` is some function.
    :type coef: numpy.array
    :param llkfun: log-likelihood function to optimize.
    :type llkfun: Callable
    :return: An approximation of the Hessian as a numpy array.
    :rtype: numpy.array
    """
    
    f0 = llkfun(coef,*llkgargs,**llkkwargs)

    eps1 = np.zeros_like(coef,dtype=float)
    eps2 = np.zeros_like(coef,dtype=float)
    hes = np.zeros((len(coef),len(coef)),dtype=float)

    epsilon = np.power(np.finfo(float).eps,0.125)*2

    fis = []
    for ci in range(len(coef)):
        eps1[ci] = epsilon
        fi = llkfun(coef + eps1,*llkgargs,**llkkwargs)
        fis.append(fi)
        eps1[ci] = 0
    
    for ci in range(len(coef)):

        for cj in range(ci,len(coef)):

            eps1[ci] = epsilon
            eps2[cj] = epsilon

            fb = llkfun(coef + eps1 + eps2,*llkgargs,**llkkwargs)
            fd = (fb - fis[ci] - fis[cj] + f0)/np.power(epsilon,2)

            hes[ci,cj] = fd
            if ci != cj:
                hes[cj,ci] = fd

            eps1[ci] = 0
            eps2[cj] = 0
    
    return hes

def central_hessian(coef,llkfun,*llkgargs,**llkkwargs):
    """Generic function to approximate the hessian of ``llkfun`` at ``coef``.

    ``llkfun`` is called as ``llkfun(coef,*llkargs)`` - so all additional arguments have to be passed via the latter.

    Uses central finite differences with fixed value for epsilon, based on eq. 8 and following paragraphs in Ridout (2009) (also used similarly in numdifftools).

    References:
     - S. Wood (2015). Core Statistics
     - M.S. Ridout (2009). Statistical Applications of the Complex-step Method of Numerical Differentiation
     - P. Brodtkorb (2014). numdifftools. see https://numdifftools.readthedocs.io/en/latest/reference/generated/numdifftools.core.Hessian.html#equation-9

    :param coef: Current estimate of coefficients of which ``llkfun`` is some function.
    :type coef: numpy.array
    :param llkfun: log-likelihood function to optimize.
    :type llkfun: Callable
    :return: An approximation of the Hessian as a numpy array.
    :rtype: numpy.array
    """
    f0 = llkfun(coef,*llkgargs,**llkkwargs)

    eps1 = np.zeros_like(coef,dtype=float)
    eps2 = np.zeros_like(coef,dtype=float)
    hes = np.zeros((len(coef),len(coef)),dtype=float)

    epsilon = np.power(np.finfo(float).eps,0.125)*2

    fps =[]
    fms = []

    for ci in range(len(coef)):
        eps1[ci] = epsilon
        fp = llkfun(coef + eps1,*llkgargs,**llkkwargs)
        fm = llkfun(coef - eps1,*llkgargs,**llkkwargs)
        fps.append(fp)
        fms.append(fm)
        eps1[ci] = 0
    
    for ci in range(len(coef)):

        for cj in range(ci,len(coef)):
            
            eps1[ci] = epsilon
            eps2[cj] = epsilon
            

            f1 = llkfun(coef + eps1 + eps2,*llkgargs,**llkkwargs)
            f2 = llkfun(coef - eps1 - eps2,*llkgargs,**llkkwargs)
        
            fd = ((f1 - fps[ci]) - (fps[cj] - f0) + (f2 - fms[ci]) - (fms[cj] - f0))/(2*np.power(epsilon,2))

            hes[ci,cj] = fd

            if ci != cj:
                hes[cj,ci] = fd

            eps1[ci] = 0
            eps2[cj] = 0

        for cj in range(ci):
            eps1[ci] = epsilon
            eps2[cj] = epsilon
            

            f1 = llkfun(coef + eps1 + eps2,*llkgargs,**llkkwargs)
            f2 = llkfun(coef - eps1 - eps2,*llkgargs,**llkkwargs)
        
            fd = ((f1 - fps[ci]) - (fps[cj] - f0) + (f2 - fms[ci]) - (fms[cj] - f0))/(2*np.power(epsilon,2))

            hes[ci,cj] = (hes[ci,cj] + fd)/2

            if ci != cj:
                hes[cj,ci] = (hes[cj,ci] + fd)/2

            eps1[ci] = 0
            eps2[cj] = 0


    return hes

def adjust_CI(model,n_ps,b,predi_mat,use_terms,alpha,seed):
        """
        Internal function to adjust point-wise CI to behave like whole-function interval (based on Wood, 2017; section 6.10.2 and Simpson, 2016):

        ``model.coef +- b`` gives point-wise interval, and for the interval to cover the whole-function, ``1-alpha`` % of posterior samples should
        be expected to fall completely within these boundaries.

        From section 6.10 in Wood (2017) we have that :math:`\\boldsymbol{\\beta} | \mathbf{y}, \\boldsymbol{\lambda} \sim N(\hat{\\boldsymbol{\\beta}},\mathbf{V})`.
        :math:`\mathbf{V}` is the covariance matrix of this conditional posterior, and can be obtained by evaluating ``model.lvi.T @ model.lvi * model.scale`` (``model.scale`` should be
        set to 1 for :class:`msssm.models.GAMMLSS` and :class:`msssm.models.GSMM`).

        The implication of this result is that we can also expect the deviations :math:`\\boldsymbol{\\beta} - \hat{\\boldsymbol{\\beta}}`  to follow
        :math:`\\boldsymbol{\\beta} - \hat{\\boldsymbol{\\beta}} | \mathbf{y}, \\boldsymbol{\lambda} \sim N(0,\mathbf{V})`. In line with the whole-function interval definition above, ``1-alpha`` % of
        ``predi_mat@[*coef - coef]`` (where ``[*coef - coef]`` representes the deviations :math:`\\boldsymbol{\\beta} - \hat{\\boldsymbol{\\beta}}`) should fall within ``[b,-b]``.
        Wood (2017) suggests to find ``a`` so that ``[a*b,a*-b]`` achieves this.

        To do this, we find ``a`` for every ``predi_mat@[*coef - coef]`` and then select the final one so that ``1-alpha`` % of samples had an equal or lower
        one. The consequence: ``1-alpha`` % of samples drawn should fall completely within the modified boundaries.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
         - Simpson, G. (2016). Simultaneous intervals for smooths revisited.

        :param model: GAMM,GAMLSS, or GSMM model (which has been fitted) for which to estimate :math:`\mathbf{V}`
        :type model: GAMM or GAMLSS or GSMM
        :param n_ps: Number of samples to obtain from posterior.
        :type n_ps: int
        :param b: Ci boundary of point-wise CI.
        :type b: [float]
        :param predi_mat: Model matrix for a particular smooth term or additive combination of parameters evaluated usually at a representative sample of predictor variables.
        :type predi_mat: scipy.sparse.csc_array
        :param use_terms: The indices corresponding to the terms that should be used to obtain the prediction or ``None`` in which case all terms will be used.
        :type use_terms: list[int] or None
        :param alpha: The alpha level to use for the whole-function interval adjustment calculation as outlined above.
        :type alpha: float
        :param seed: Can be used to provide a seed for the posterior sampling.
        :type seed: int or None
        """
        use_post = None
        if not use_terms is None:
            # If we have many random factor levels, but want to make predictions only
            # for fixed effects, it's wasteful to sample all coefficients from posterior.
            # The code below performs a selection of the coefficients to be sampled.
            use_post = predi_mat.sum(axis=0) != 0
            use_post = np.arange(0,predi_mat.shape[1])[use_post]

        # Sample deviations [*coef - coef] from posterior of model
        post = model.sample_post(n_ps,use_post,deviations=True,seed=seed)

        # To make computations easier take the abs of predi_mat@[*coef - coef], because [b,-b] is symmetric we can
        # simply check whether abs(predi_mat@[*coef - coef]) < b by computing abs(predi_mat@[*coef - coef])/b. The max of
        # this ratio, over rows of predi_mat, is a for this sample. If a<=1, no extension is necessary for this series.
        if use_post is None:
            fpost = np.abs(predi_mat@post)
        else:
            fpost = np.abs(predi_mat[:,use_post]@post)

        # Compute ratio between abs(predi_mat@[*coef - coef])/b for every sample.
        fpost = fpost / b[:,None]

        # Then compute max of this ratio, over rows of predi_mat, for every sample. np.max(fpost,axis=0) now is a vector
        # with n_ps elements, holding for each sample the multiplicative adjustment a, necessary to ensure that predi_mat@[*coef - coef]
        # falls completely between [a*b,a*-b].
        # The final multiplicative adjustment bmadq is selected from this vector to be high enough so that in 1-(alpha) simulations
        # we have an equal or lower a.
        bmadq = np.quantile(np.max(fpost,axis=0),1-alpha)

        # Then adjust b
        b *= bmadq

        return b

def compute_reml_candidate_GAMM(family,y,X,penalties,n_c=10):
   """
   Allows to evaluate REML criterion (e.g., Wood, 2011; Wood, 2016) efficiently for
   a set of \lambda values.

   Internal function used for computing the correction applied to the edf for the GLRT - based on Wood (2017) and Wood et al., (2016).

   See :func:`REML` function for more details.
   """

   S_emb,_,_,_ = compute_S_emb_pinv_det(X.shape[1],penalties,"svd")

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

   # Get edf and optionally estimate scale (scale will be kept at fixed (e.g., 1) for Generalized case)
   _,_,edf,_,_,scale = update_scale_edf(y,z,eta,Wr,X.shape[0],X.shape[1],LP,None,Pr,None,family,penalties,n_c)

   if family.twopar:
        llk = family.llk(y,mu,scale)
   else:
        llk = family.llk(y,mu)

   # Now compute REML for candidate
   reml = REML(llk,nH/scale,coef,scale,penalties)

   return reml,LP,Pr,coef,scale,edf,llk

def compute_REML_candidate_GSMM(family,y,Xs,penalties,coef,n_coef,coef_split_idx,method="Newton",conv_tol=1e-7,n_c=10,**bfgs_options):
    """
    Allows to evaluate REML criterion (e.g., Wood, 2011; Wood, 2016) efficiently for
    a set of \lambda values for a GSMM or GAMMLSS.

    Internal function used for computing the correction applied to the edf for the GLRT - based on Wood (2017) and Wood et al., (2016).

    See :func:`REML` function for more details.
   """

    # Build current penalties
    S_emb,S_pinv,_,FS_use_rank = compute_S_emb_pinv_det(n_coef,penalties,"svd")

    if isinstance(family,GENSMOOTHFamily): # GSMM
        
        # Compute likelihood for current estimate
        c_llk = family.llk(coef,coef_split_idx,y,Xs)

        if method != "Newton":
            # Define negative penalized likelihood function to be minimized via BFGS
            def __neg_pen_llk(coef,coef_split_idx,y,Xs,family,S_emb):
                neg_llk = -1 * family.llk(coef,coef_split_idx,y,Xs)
                return neg_llk + coef.T@S_emb@coef
        
        # Estimate coefficients:
        if method == "Newton":
            coef,H,L,LV,c_llk,_,_,_,_ = update_coef_gen_smooth(family,y,Xs,coef,
                                                               coef_split_idx,S_emb,None,
                                                               c_llk,0,100,
                                                               100,conv_tol,"Chol",None,None)
            
            V = LV.T @ LV # inverse of hessian of penalized likelihood
            nH = -1*H # negative hessian of likelihood
        else:
            opt = scp.optimize.minimize(__neg_pen_llk,
                                        np.ndarray.flatten(coef),
                                        args=(coef_split_idx,y,Xs,family,S_emb),
                                        method=method,
                                        options={"maxiter":100,
                                                 **bfgs_options})
            
            # Get coefficient estimate
            coef = opt["x"].reshape(-1,1)

            # Compute penalized likelihood for current estimate
            c_llk = family.llk(coef,coef_split_idx,y,Xs)

            # Get inverse of Hessian of penalized likelihood
            if method == "BFGS":
               V = scp.sparse.csc_array(opt["hess_inv"])
            elif method == "L-BFGS-B":
               V = scp.sparse.csc_array(opt.hess_inv.todense())
            V.eliminate_zeros()

            # Get Cholesky factor needed for (accelerated) EFS
            LVPT, P, code = cpp_cholP(V)
            LVT = apply_eigen_perm(P,LVPT)
            LV = LVT.T

            # Get an apparoximation of the Hessian of the likelihood
            LHPT = compute_Linv(LVPT)
            LHT = apply_eigen_perm(P,LHPT)
            nH = LHT.T@LHT # approximately: negative Hessian of llk + S_emb
            nH -= S_emb # approximately: negative Hessian of llk 

    else: # GAMMLSS
        split_coef = np.split(coef,coef_split_idx)

        # Initialize etas and mus
        etas = [Xs[i]@split_coef[i] for i in range(family.n_par)]
        mus = [family.links[i].fi(etas[i]) for i in range(family.n_par)]

        c_llk = family.llk(y,*mus)

        # Estimate coefficients
        coef,split_coef,mus,etas,H,L,LV,c_llk,_,_,_,_ = update_coef_gammlss(family,mus,y,Xs,coef,
                                                                            coef_split_idx,S_emb,None,
                                                                            c_llk,0,100,
                                                                            100,conv_tol,"Chol",None,None)
        
        V = LV.T@LV
        nH = -1*H

    # Remaining computations are shared for GAMMLSS and GSMM
    
    # Compute reml
    reml = REML(c_llk,nH,coef,1,penalties)[0,0]

    # Compute edf
    lgdetDs = []
    for lti,lTerm in enumerate(penalties):

        lt_rank = None
        if FS_use_rank[lti]:
            lt_rank = lTerm.rank

        lgdetD,_ = compute_lgdetD_bsb(lt_rank,lTerm.lam,S_pinv,lTerm.S_J_emb,coef)
        lgdetDs.append(lgdetD)

    total_edf,_, _ = calculate_edf(None,None,LV,penalties,lgdetDs,n_coef,n_c)

    return reml,V,coef,total_edf,c_llk


def REML(llk,nH,coef,scale,penalties):
   """
   Based on Wood (2011). Exact REML for Gaussian GAM, Laplace approximate (Wood, 2016) for everything else.
   Evaluated after applying stabilizing reparameterization discussed by Wood (2011).

   References:
    - Wood, S. N., (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models.
    - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models

   :param llk: log-likelihood of model
   :type llk: float
   :param nH: negative hessian of log-likelihood of model
   :type nH: scipy.sparse.csc_array
   :param coef: Estimated vector of coefficients of shape (-1,1)
   :type coef: numpy.array
   :param scale: (Estimated) scale parameter - can be set to 1 for GAMLSS or GSMMs.
   :type scale: float
   :param penalties: List of penalties that were part of the model.
   :type penalties: [LambdaTerm]
   :return: (Approximate) REML score
   :rtype: float
   """ 

   # Compute S_\lambda before any re-parameterization
   S_emb,_,_,_ = compute_S_emb_pinv_det(len(coef),penalties,"svd")

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
        
   # Now log(|nH+S_\lambda|)... Wood (2011) shows stable computation based on QR decomposition, but
   # we will generally not be able to compute a QR decomposition of X so that X.T@X=H efficiently.
   # Hence, we simply rely on the pivoted cholesky (again pre-conditioned) used for fitting (based on S_\lambda before
   # re-parameterization).
   H_pen = nH + S_emb/scale

   Sdiag = np.power(np.abs(H_pen.diagonal()),0.5)
   PI = scp.sparse.diags(1/Sdiag,format='csc')
   P = scp.sparse.diags(Sdiag,format='csc')
   L,Pr,code = cpp_cholP(PI@H_pen@PI)
   Pr = compute_eigen_perm(Pr)

   if code != 0:
       raise ValueError("Failed to compute REML.")
  
   #print(((P@Pr.T@L) @ (P@Pr.T@L).T - H_pen).max())
   # Can ignore Pr for det computation, because det(Pr)*det(Pr.T)=1
   lgdetXXS = 2*np.log((P@L).diagonal()).sum()

   # Done
   return reml + lgdetS/2 - lgdetXXS/2 + (Mp*np.log(2*np.pi))/2

def estimateVp(model,nR = 20,lR = 100,n_c=10,a=1e-7,b=1e7,verbose=False,drop_NA=True,optimizer="Newton",seed=None,conv_tol=1e-7,df=40,strategy="JJJ3",**bfgs_options):
    """Estimate covariance matrix :math:`\mathbf{V}_{\\boldsymbol{p}}` of posterior for :math:`\mathbf{p} = log(\\boldsymbol{\lambda})`. REML scores are used to
    approximate expectation, similar to what was suggested by Greven & Scheipl (2016).

    References:
     - https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices
     - Greven, S., & Scheipl, F. (2016). Comment on: Smoothing Parameter and Model Selection for General Smooth Models

    :param model: GAMM,GAMLSS, or GSMM model (which has been fitted) for which to estimate :math:`\mathbf{V}`
    :type model: GAMM or GAMLSS or GSMM
    :param nR: Initial :math:`\lambda`  Grid is based on `nR` equally-spaced samples from :math:`\lambda/lr` to :math:`\lambda*lr`. In addition, ``nR*len(model.formula.penalties)`` updates to :math:`\mathbf{V}_{\\boldsymbol{p}}` are performed during each of which additional `nR` :math:`\lambda` samples/reml scores are generated/computed, defaults to 20
    :type nR: int, optional
    :param lR: Initial :math:`\lambda`  Grid is based on `nR` equally-spaced samples from :math:`\lambda/lr` to :math:`\lambda*lr`, defaults to 100
    :type lR: int, optional
    :param n_c: Number of cores to use to compute the estimate, defaults to 10
    :type n_c: int, optional
    :param a: Minimum :math:`\lambda` value that is included when forming the initial grid. In addition, any of the :math:`\lambda` estimates obtained from ``model`` (used to define the mean for the posterior of :math:`\\boldsymbol{p}|y \sim N(log(\hat{\\boldsymbol{p}}),\mathbf{V}_{\\boldsymbol{p}})`) which are smaller than this are set to this value as well, defaults to 1e-7 the minimum possible estimate
    :type a: float, optional
    :param b: Maximum :math:`\lambda` value that is included when forming the initial grid. In addition, any of the :math:`\lambda` estimates obtained from ``model`` (used to define the mean for the posterior of :math:`\\boldsymbol{p}|y \sim N(log(\hat{\\boldsymbol{p}}),\mathbf{V}_{\\boldsymbol{p}})`) which are larger than this are set to this value as well, defaults to 1e7 the maximum possible estimate
    :type b: float, optional
    :param verbose: Whether to print progress information or not, defaults to False
    :type verbose: bool, optional
    :param drop_NA: Whether to drop rows in the **model matrices** corresponding to NAs in the dependent variable vector. Defaults to True.
    :type drop_NA: bool,optional
    :param optimizer: Which optimizer to use to estimate the coefficients for more general smooth models - supports "Newton", "BFGS", and "L-BFGS-B". In case of the former, ``self.family`` needs to implement :func:`gradient` and :func:`hessian`. Defaults to "Newton"
    :type optimizer: str,optional
    :param seed: Seed to use for random parts of the estimate. Defaults to None
    :type seed: int,optional
    :param conv_tol: Tolerance used to determine whether estimate has converged. The Frobenius norm of the **difference** in the estimate after the update (``||Vp_next - Vp||``) is compared to ``conv_tol*||Vp_next||``, defaults to 1e-7
    :type conv_tol: float, optional
    :param df: Degrees of freedom used for the multivariate t distribution used to sample the next set of candidates. Setting this to ``np.inf`` means a multivariate normal is used for sampling, defaults to 40
    :type df: int, optional
    :param bfgs_options: Any additional keyword arguments that should be passed on to the call of :func:`scipy.optimize.minimize`. If none are provided, the ``gtol`` argument will be initialized to 1e-3. Note also, that in any case the ``maxiter`` argument is automatically set to 100. Defaults to None.
    :type bfgs_options: key=value,optional
    :raises ValueError: Will throw an error when ``optimizer`` is not one of 'Newton', 'BFGS', 'L-BFGS-B' and a :class:`mssm.models.GSMM` is to be estimated.
    :return: An estimate of the covariance matrix of the posterior for :math:`\mathbf{p} = log(\\boldsymbol{\lambda})`
    :rtype: numpy.array
    """
    np_gen = np.random.default_rng(seed)

    family = model.family

    if isinstance(family,GENSMOOTHFamily):
        if not bfgs_options:
            bfgs_options = {"gtol":1e-3}

        if not optimizer in ["Newton", "BFGS", "L-BFGS-B"]:
            raise ValueError("'optimizer' needs to be set to one of 'Newton', 'BFGS', 'L-BFGS-B'.")

    if isinstance(family,Family):
        rPen = copy.deepcopy(model.formula.penalties)
    else: # GAMMLSS and GSMM case
        rPen = copy.deepcopy(model.overall_penalties)

    if isinstance(family,Family):
        y = model.formula.y_flat[model.formula.NOT_NA_flat]
        X = model.get_mmat()

    else:
        if drop_NA:
            y = model.formulas[0].y_flat[model.formulas[0].NOT_NA_flat]
        else:
            y = model.formulas[0].y_flat
        Xs = model.get_mmat(drop_NA=drop_NA)
        init_coef = copy.deepcopy(model.overall_coef)
    
    if strategy == "JJJ1" or strategy=="JJJ2":
        # Approximate Vp via finite differencing of negative REML.

        # Set up mean log-smoothing penalty vector - ignoring any a and b limits provided.
        if isinstance(family,Family):
            ep = np.log(np.array([pen.lam for pen in model.formula.penalties]).reshape(-1,1))
        else:
            ep = np.log(np.array([pen.lam for pen in model.overall_penalties]).reshape(-1,1))

        #from numdifftools import Hessian
        def reml_wrapper(rho,family,y,X,rPen,*reml_args,**reml_kwargs):
            
            for peni in range(len(rho)):
                rPen[peni].lam = np.exp(rho[peni])
            
            if isinstance(family,Family):
                reml,_,_,_,_,_,_ = compute_reml_candidate_GAMM(family,y,X,rPen,*reml_args,**reml_kwargs)
            else:
                reml,_,_,_,_ = compute_REML_candidate_GSMM(family,y,X,rPen,*reml_args,**reml_kwargs)
            
            return -reml
        
        if isinstance(family,Family):
            nHp = central_hessian(ep.flatten(),reml_wrapper,family,y,X,rPen,n_c)
            #nHp = Hessian(reml_wrapper)(ep.flatten(),family,y,X,rPen,n_c)
        else:
            nHp = central_hessian(ep.flatten(),reml_wrapper,family,y,Xs,rPen,init_coef,len(init_coef),model.coef_split_idx,n_c=n_c,method=optimizer,**bfgs_options)
        
        # Vp is now simply [nHp]^{-1}
        # but we should rely on an eigen decomposition so that we can naturally produce a generalized inverse as discussed by
        # WPS (2016).

        eig, U =scp.linalg.eigh(nHp)
        ire = np.zeros_like(eig)
        ire[eig > 0] = 1/np.sqrt(eig[eig > 0]) # Only compute inverse for eig values larger than zero, setting rem. ones to zero yields generalized invserse. 
        Ri = np.diag(ire)@U.T # Root of Vp

        Vp = Ri.T@Ri

        # Now, in mgcv a regularized version is computed as well, which essentially sets all positive eigenvalues
        # to a positive minimum. This regularized version is utilized in the smoothness uncertainty correction, so we compute it
        # as well.
        # See: https://github.com/cran/mgcv/blob/aff4560d187dfd7d98c7bd367f5a0076faf129b7/R/gam.fit3.r#L1010
        ire2 = np.zeros_like(eig)
        ire2[eig > 0] = 1/np.sqrt(eig[eig > 0] + 0.1)
        Rir = np.diag(ire2)@U.T # Root of regularized Vp

        Vpr = Rir.T@Rir
        #print(eig,1/eig)
        if strategy == "JJJ1":
            return Vp,Vpr,Ri,Rir
            
    if strategy == "JJJ2":
        orig_Vp = copy.deepcopy(Vp)
        Vp = Vpr

    # Set up grid of nR equidistant values based on marginal grids that cover range from \lambda/lr to \lambda*lr
    # conditional on all estimated penalty values except the current one.
    rGrid = []
    if strategy == "JJJ3":
        for pi,pen in enumerate(rPen):
            
            # Set up marginal grid from \lambda/lr to \lambda*lr
            mGrid = np.exp(np.linspace(np.log(max([a,pen.lam/lR])),np.log(min([b,pen.lam*lR])),nR))
            
            # Now create penalty candidates conditional on estimates for all other penalties except current one
            for val in mGrid:
                if abs(val - pen.lam) <= 1e-7:
                    continue
                
                rGrid.append(np.array([val if pii == pi else np_gen.choice(np.exp(np.linspace(np.log(max([a,pen2.lam/lR])),np.log(min([b,pen2.lam*lR])),nR)),size=None) for pii,pen2 in enumerate(rPen)]))
        
    # Make sure actual estimate is included once.
    rGrid.append(np.array([pen2.lam for pen2 in rPen]))
    rGrid = np.array(rGrid)


    if isinstance(family,Family):
        y = model.formula.y_flat[model.formula.NOT_NA_flat]
        X = model.get_mmat()

        orig_scale = family.scale
        if family.twopar:
            _,orig_scale = model.get_pars()

    else:
        if drop_NA:
            y = model.formulas[0].y_flat[model.formulas[0].NOT_NA_flat]
        else:
            y = model.formulas[0].y_flat
        Xs = model.get_mmat(drop_NA=drop_NA)
        orig_scale = 1
        init_coef = copy.deepcopy(model.overall_coef)
    
    
    if isinstance(family,Family):
        y = model.formula.y_flat[model.formula.NOT_NA_flat]
        X = model.get_mmat()

        orig_scale = family.scale
        if family.twopar:
            _,orig_scale = model.get_pars()

    else:
        if drop_NA:
            y = model.formulas[0].y_flat[model.formulas[0].NOT_NA_flat]
        else:
            y = model.formulas[0].y_flat
        Xs = model.get_mmat(drop_NA=drop_NA)
        orig_scale = 1
        init_coef = copy.deepcopy(model.overall_coef)
    
    remls = []

    enumerator = rGrid
    if verbose and strategy == "JJJ3":
        enumerator = tqdm(rGrid)
    for r in enumerator:

        # Prepare penalties with current lambda candidate r
        for ridx,rc in enumerate(r):
            rPen[ridx].lam = rc
        
        # Now compute REML - and all other terms needed for correction proposed by Greven & Scheipl (2017)
        if isinstance(family,Family):
            reml,_,_,_,_,_,_ = compute_reml_candidate_GAMM(family,y,X,rPen,n_c)
        else:
            reml,_,_,_,_ = compute_REML_candidate_GSMM(family,y,Xs,rPen,init_coef,len(init_coef),model.coef_split_idx,n_c=n_c,method=optimizer,**bfgs_options)

        # Now collect what we need for updating Vp
        remls.append(reml)
    
    # Iteratively estimate Vp - covariance matrix of log(\lambda) to guide further REML grid sampling

    # (Re-)create mean of log-smoothing parameter vector, this time adhering to limits.
    if isinstance(family,Family):
        ep = np.log(np.array([min(b,max(a,pen.lam)) for pen in model.formula.penalties]).reshape(-1,1))
    else:
        ep = np.log(np.array([min(b,max(a,pen.lam)) for pen in model.overall_penalties]).reshape(-1,1))

    # Get first estimate for Vp based on samples collected so far
    if strategy == "JJJ3":
        Vp = updateVp(ep,remls,rGrid)

    # Now continuously update Vp and generate more REML samples in the process
    n_est = nR
    enumerator = range(nR*len(ep))
    if verbose and strategy == "JJJ3":
        enumerator = tqdm(enumerator,desc="Estimating Vp",leave=True)
    elif verbose:
        enumerator = tqdm(enumerator,desc="Refining Vp",leave=True)

    for sp in enumerator:

        # Generate next \lambda values for which to compute REML, and Vb
        p_sample = []
        while len(p_sample) == 0:
            p_sample = scp.stats.multivariate_t.rvs(loc=np.ndarray.flatten(ep),shape=Vp,df=df,size=n_est,random_state=seed)
            p_sample = np.exp(p_sample)

            if len(np.ndarray.flatten(ep)) == 1: # Single lambda parameter in model
                
                if n_est == 1: # and single sample (n_est==1) - so p_sample needs to be shape (1,1)
                    p_sample = np.array([p_sample])

                p_sample = p_sample.reshape(n_est,1) # p_sample needs to be shape (n_est,1)
                
            elif n_est == 1: # multiple lambdas - so p_sample needs to be shape (1,n_lambda)
                p_sample = np.array([p_sample]).reshape(1,-1)
            
            # Re-sample values we have encountered before.
            minDiag = 0.1*min(np.sqrt(Vp.diagonal()))
            for lami in range(p_sample.shape[0]):
                while np.any(np.max(np.abs(rGrid - p_sample[lami]),axis=1) < minDiag) or np.any(p_sample[lami] < 1e-9) or np.any(p_sample[lami] > 1e12):
                    if not seed is None:
                        seed += 1
                    p_sample[lami] = np.exp(scp.stats.multivariate_t.rvs(loc=np.ndarray.flatten(ep),shape=Vp,df=df,size=1,random_state=seed))

            if not seed is None:
                seed += 1

        for ps in p_sample:
            for ridx,rc in enumerate(ps):
                rPen[ridx].lam = rc
            
            if isinstance(family,Family):
                #print([penx.lam for penx in rPen])
                reml,_,_,_,_,_,_ = compute_reml_candidate_GAMM(family,y,X,rPen,n_c)

            else:
                reml,_,_,_,_ = compute_REML_candidate_GSMM(family,y,Xs,rPen,init_coef,len(init_coef),model.coef_split_idx,n_c=n_c,method=optimizer,**bfgs_options)

            # Collect new remls and update grid of log(lambdas)
            remls.append(reml)
            rGrid = np.concatenate((rGrid,ps.reshape(1,-1)),axis=0)

        Vp_next = updateVp(ep,remls,rGrid)

        # Update
        if strategy == "JJJ3":
            Vp = Vp_next
        else:
            Vp = 0.99*Vp + 0.01*Vp_next
        
    #print(len(np.unique(rGrid,axis=0)),rGrid.shape)
    if strategy == "JJJ3":
        return Vp
    
    else:
        # Re-compute root of regularized VP (at this point Vp actually holds refined Vpr)
        eig, U =scp.linalg.eigh(Vp)
        ire = np.zeros_like(eig)
        ire[eig > 0] = np.sqrt(eig[eig > 0])
        Rir = np.diag(ire)@U.T # Root of refined regularized Vp

        # Make sure Vp is original
        Vpr = copy.deepcopy(Vp)
        Vp = orig_Vp

        return Vp, Vpr, Ri, Rir

def updateVp(ep,remls,rGrid):
    """Update covariance matrix of posterior for :math:`\mathbf{p} = log(\\boldsymbol{\lambda})`. REML scores are used to
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
   S_emb,_,_,_ = compute_S_emb_pinv_det(X.shape[1],rPen,"svd")
   LP, Pr, coef, code = cpp_solve_coef(y,X,S_emb)

   if code != 0:
       raise ValueError("Forming coefficients for specified penalties was not possible.")
   
   eta = (X @ coef).reshape(-1,1)
   
   # Compute scale
   _,_,edf,_,_,scale = update_scale_edf(y,None,eta,None,X.shape[0],X.shape[1],LP,None,Pr,None,family,rPen,10)

   llk = family.llk(y,eta,scale)

   # Now compute REML for candidate
   reml = REML(llk,XX/scale,coef,scale,rPen)
   coef = coef.reshape(-1,1)

   # Form VB, first solve LP{^-1}
   LPinv = compute_Linv(LP,1)
   Linv = apply_eigen_perm(Pr,LPinv)

   # Now collect what we need for the remaining terms
   return Linv,coef,reml,scale,edf,llk

def compute_Vb_corr_WPS(Vbr,Vpr,Vr,H,S_emb,penalties,coef,scale=1):
    """Computes both correction terms for ``Vb`` or :math:`\mathbf{V}_{\\boldsymbol{\\beta}}`, which is the co-variance matrix for the conditional posterior of :math:`\\boldsymbol{\\beta}` so that
    :math:`\\boldsymbol{\\beta} | y, \\boldsymbol{\lambda} \sim N(\hat{\\boldsymbol{\\beta}},\mathbf{V}_{\\boldsymbol{\\beta}})`, described by Wood, Pya, & Säfken (2016).

    References:
     - Wood, S. N., (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models.
     - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

    :param Vbr: Transpose of root for the estimate for the (unscaled) covariance matrix of :math:`\\boldsymbol{\\beta} | y, \\boldsymbol{\lambda}` - the coefficients estimated by the model.
    :type Vbr: scipy.sparse.csc_array
    :param Vpr: A (regularized) estimate of the covariance matrix of :math:`\\boldsymbol{\\rho}` - the log smoothing penalties.
    :type Vpr: numpy.array
    :param Vr: Transpose of root of **un-regularized** covariance matrix of :math:`\\boldsymbol{\\rho}` - the log smoothing penalties.
    :type Vr: numpy.array
    :param H: The Hessian of the log-likelihood
    :type H: scipy.sparse.csc_array
    :param S_emb: The weighted penalty matrix.
    :type S_emb: scipy.sparse.csc_array
    :param penalties: A list holding the :class:`Lambdaterm`s estimated for the model.
    :type penalties: [LambdaTerm]
    :param penalties: An array holding the estimated regression coefficients. Has to be of shape (-1,1)
    :type penalties: numpy.array
    :param scale: Any scale parameter estimated as part of the model. Can be omitted for more generic models beyond GAMMs. Defaults to 1.
    :type scale: float
    :raises ArithmeticError: Will throw an error when the negative Hessian of the penalized likelihood is ill-scaled so that a Cholesky decomposition fails.
    :return: A tuple containing: ``Vc`` and ``Vcc``. ``Vbr.T@Vbr*scale`` + ``Vc`` + ``Vcc`` is then approximately the correction devised by WPS (2016).
    :rtype: (numpy.array, numpy.array)
    """

    # Get (unscaled) negative Hessian of the penalized likelihood.
    # For a GAMM, this will thus be X.T@W@X + S_emb - since H = (X.T@W@X)\phi
    nH = (-1*H)*scale + S_emb

    # We need un-pivoted transpose of cholesky of this, but can pre-condition
    Sdiag = np.power(np.abs(nH.diagonal()),0.5)
    PI = scp.sparse.diags(1/Sdiag,format='csc')
    P = scp.sparse.diags(Sdiag,format='csc')
    LP,code = cpp_chol(PI@nH@PI)
    R = (P@LP).T
    R.sort_indices()

    if code != 0:
        raise ArithmeticError("Failed to compute Cholesky of negative Hessian of penalized likelihood.")

    #print((R.T@R - nH).max())

    # Get partial derivatives of beta with respect to \rho, the log smoothing penalties
    dBetadRhos = np.zeros((len(coef),len(penalties)))

    # Vbr.T@Vbr = nH^{-1} - Vbr is in principle available in model.lvi or model.overall_lvi after fitting,
    # but we need Rinv anyway..
    Rinv = compute_Linv(R.T).T

    for peni,pen in enumerate(penalties):
        # Given in Wood (2017)
        #print((-pen.lam * (Vbr.T @ (Vbr @ (pen.S_J_emb @ coef)))).shape)
        dBetadRhos[:,[peni]] = -pen.lam * (Rinv @ Rinv.T) @ (pen.S_J_emb @ coef)
    
    #dBetadRhos = np.array(dBetadRhos) # Should be of shape (nCoef,nLambda)
    #print(dBetadRhos.shape)

    # Can now compute first correction term
    Vcr = Vr @ dBetadRhos.T
    Vc = Vcr.T @ Vcr

    # Now second correction term. First need partial derivatives of elements in R^{-1} with respect to each \rho
    # Supplementary materials D in WPS (2016) show how to obtain those from partial derivatives of elements in R 
    # with respect to each \rho, obtaining the latter is implemented in cpp_dchol.
    dRdRhos = []
    for pen in penalties:

        #dDat, dRow, dcol = cpp_dChol(R, pen.lam*pen.S_J_emb.tocsr())

        #dDat = [d for r in dDat for d in r]
        #dcol = [d for r in dcol for d in r]

        #dRdRhos.append(scp.sparse.csr_array((dDat,(dRow,dcol)),shape=Vc.shape))


        if True:
            dChol2 = np.zeros(H.shape)
            A = pen.lam*pen.S_J_emb.toarray()
            R2 = R.toarray()

            for i in range(H.shape[1]):
                Rii = 0
                dRii = 0
                for j in range(i,H.shape[1]):
                    
                    Bij = A[i,j]
                    if i > 0:
                        k = 0
                        while k < i:
                            Bij -= ((dChol2[k,i]*R2[k,j]) + (R2[k,i]*dChol2[k,j]))
                            k += 1
                        
                    if i == j:
                        Rii = R2[i,i]
                        dRii = 0.5*Bij/Rii
                        dChol2[i,j] = dRii
                    elif j > i:
                        dChol2[i,j] = (Bij - (R2[i,j] * dRii))/Rii
            
            
            dRdRhos.append(dChol2)
            
    
    # Now inverse computations (see sup. materials D in WPS, 2016)
    # Let's review this:
    # The un-numbered equation between eq. (6) and (7) - the Taylor expansion - 
    # suggests that we need dR'.Td\rho where R'.T@R' = Vb.
    # we have: R.T@R = nH = Vb^{-1} (assuming Vb/nH are unscaled)
    # and thus: R^{-1}@R^{-T} = Vb.
    # Hence: R' = R^{-T}
    # and:   R'.T = R^{-1}
    # So: dR'.Td\rho = dR^{-1}d\rho = R^{-1}@dRd\rho@R^{-1}
    # and we either have to take the transpose or change the flip the indexing in the 5 term sum!

    #print((R@Rinv).max())

    for dRi in range(len(dRdRhos)):
        dRdRhos[dRi] = (Rinv@dRdRhos[dRi])@Rinv
    
    # Now final sum
    Vcc = np.zeros_like(Vc)
    for j in range(Vc.shape[0]):
        for m in range(j,Vc.shape[1]):

            for i in range(len(coef)):
                for l in range(len(penalties)):
                    for k in range(len(penalties)):

                        Vcc[j,m] += dRdRhos[k][j,i] * Vpr[k,l] * dRdRhos[l][m,i]

                        if m > j:
                            Vcc[m,j] += dRdRhos[k][m,i] * Vpr[k,l] * dRdRhos[l][j,i]
    
    # Done, don't forget to scale Vcc since nH was unscaled!
    return Vc, scale*Vcc


def correct_VB(model,nR = 20,lR = 100,grid_type = 'JJJ3',a=1e-7,b=1e7,df=40,n_c=10,form_t=True,form_t1=False,verbose=False,drop_NA=True,method="Newton",V_shrinkage_weight=0.75,only_expected_edf=False,seed=None,**bfgs_options):
    """Estimate :math:`\mathbf{V}`, the covariance matrix of the unconditional posterior :math:`\\boldsymbol{\\beta} | y \sim N(\hat{\\boldsymbol{\\beta}},\\mathbf{V})` to account for smoothness uncertainty.
    
    Wood et al. (2016) and Wood (2017) show that when basing conditional versions of model selection criteria or hypothesis
    tests on :math:`\mathbf{V}_{\\boldsymbol{\\beta}}`, which is the co-variance matrix for the conditional posterior of :math:`\\boldsymbol{\\beta}` so that
    :math:`\\boldsymbol{\\beta} | y, \\boldsymbol{\lambda} \sim N(\hat{\\boldsymbol{\\beta}},\mathbf{V}_{\\boldsymbol{\\beta}})`, the tests are severely biased. To correct for this they
    show that uncertainty in :math:`\\boldsymbol{\lambda}` needs to be accounted for. Hence they suggest to base these tests on :math:`\mathbf{V}`, the covariance matrix
    of the **unconditional posterior** :math:`\\boldsymbol{\\beta} | y \sim N(\hat{\\boldsymbol{\\beta}},\\mathbf{V})`. They show how to obtain an estimate of :math:`\mathbf{V}`,
    but this requires :math:`\mathbf{V}_{\\boldsymbol{p}}` - an estimate of the covariance matrix of :math:`\\boldsymbol{p}=log(\\boldsymbol{\lambda})`. :math:`\mathbf{V}_{\\boldsymbol{p}}` requires derivatives that are not available
    when using the efs update.

    Greven & Scheipl in their comment to the paper by Wood et al. (2016) show another option to estimate :math:`\mathbf{V}` that does not require :math:`\mathbf{V}_{\\boldsymbol{p}}`,
    based either on forming a mixture approximation or on the total variance property. The latter is implemented below, based on the
    equations for the expectations outlined in their response. A problem of this estimate is that a grid of :math:`\\boldsymbol{\lambda}` values needs to be
    provided covering the prior on :math:`\\boldsymbol{\lambda}` (see Wood 2011 for the relation between smoothness penalties and this prior). For ``mssm`` the
    default limits on this are 1e-7 and 1e7.
    
    However, as the authors already conclude, covering the entire prior range is not that efficient in case more than one :math:`\lambda` is to be estimated.
    Hence we provide an alternative way to set-up the grid, based on first forming marginal grids for each :math:`\lambda` in :math:`\\boldsymbol{\lambda}` that contain nR equally-spaced
    samples from :math:`\lambda/lr` to :math:`\lambda*lr`, while all other :math:`\lambda` values are set to random samples between the prior limits. This neglects quite a bit
    of the prior space. So we use these initial samples to estimate :math:`\mathbf{V}_{\\boldsymbol{p}}` (using the REML weights to approximate the expectation based on Greven & Scheipl; 2016), so that :math:`\\boldsymbol{p}|y \sim N(log(\hat{\\boldsymbol{p}}),\mathbf{V}_{\\boldsymbol{p}})` - see Wood et al. (2016).
    
    We then repeatedly sample new :math:`\\boldsymbol{p}` vectors from this normal, followed by updating out estimate of the normal (i.e., :math:`\mathbf{V}_{\\boldsymbol{p}}`) given these samples. Note that until the last sampling step we
    add small values to the diagonal of :math:`\mathbf{V}_{\\boldsymbol{p}}` to promote exploration. The idea is that this should help us to better explore and with less samples, locally
    around :math:`\hat{\\boldsymbol{p}}`, the uncertainty than if we would just sample from a grid. We then also later re-compute the REML weights from this normal and
    then follow the steps outlined by (Greven & Scheipl; 2016) to compute :math:`\mathbf{V}`, rather than computing the approximation suggested by Wood et al. (2016).
    
    This is done when argument grid_type = 'JJJ'. Otherwise, the G&S strategy is employed - forming a grid over the full space (from 1e-7 to 1e7,
    again evaluated for nR equally-spaced values and then permuted for the number of \lambda parameters). Note that
    the latter can get very expensive quite quickly.

    References:
     - Wood, S. N., (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models.
     - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models
     - Greven, S., & Scheipl, F. (2016). Comment on: Smoothing Parameter and Model Selection for General Smooth Models
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

    :param model: GAMM, GAMMLSS, or GSMM model (which has been fitted) for which to estimate :math:`\mathbf{V}`
    :type model: GAMM or GAMMLSS or GSMM
    :param nR: :math:`\lambda`  Grid (at least the initial one, in case grid_type=="JJJ") is based on `nR` equally-spaced samples from :math:`\lambda/lr` to :math:`\lambda*lr`. In case grid_type=="JJJ", ``nR*len(model.formula.penalties)`` updates to :math:`\mathbf{V}_{\\boldsymbol{p}}` are performed during each of which additional `nR` :math:`\lambda` samples/reml scores are generated/computed, defaults to 20
    :type nR: int, optional
    :param lR: :math:`\lambda`  Grid (at least the initial one, in case grid_type=="JJJ") is based on `nR` equally-spaced samples from :math:`\lambda/lr` to :math:`\lambda*lr`, defaults to 100
    :type lR: int, optional
    :param grid_type: How to define the grid of :math:`\lambda` values on which to base the correction - see above for details, defaults to 'JJJ'
    :type grid_type: str, optional
    :param a: Minimum :math:`\lambda` value that is included when forming the initial grid. In addition, any of the :math:`\lambda` estimates obtained from ``model`` (used to define the mean for the posterior of :math:`\\boldsymbol{p}|y \sim N(log(\hat{\\boldsymbol{p}}),\mathbf{V}_{\\boldsymbol{p}})`) which are smaller than this are set to this value as well, defaults to 1e-7 the minimum possible estimate
    :type a: float, optional
    :param b: Maximum :math:`\lambda` value that is included when forming the initial grid. In addition, any of the :math:`\lambda` estimates obtained from ``model`` (used to define the mean for the posterior of :math:`\\boldsymbol{p}|y \sim N(log(\hat{\\boldsymbol{p}}),\mathbf{V}_{\\boldsymbol{p}})`) which are larger than this are set to this value as well, defaults to 1e7 the maximum possible estimate
    :type b: float, optional
    :param df: Degrees of freedom used for the multivariate t distribution used to sample the next set of candidates. Setting this to ``np.inf`` means a multivariate normal is used for sampling, defaults to 40
    :type df: int, optional
    :param n_c: Number of cores to use to compute the correction, defaults to 10
    :type n_c: int, optional
    :param form_t: Whether or not the smoothness uncertainty corrected edf should be computed, defaults to True
    :type form_t: bool, optional
    :param form_t1: Whether or not the smoothness uncertainty + smoothness bias corrected edf should be computed, defaults to False
    :type form_t1: bool, optional
    :param verbose: Whether to print progress information or not, defaults to False
    :type verbose: bool, optional
    :param drop_NA: Whether to drop rows in the **model matrices** corresponding to NAs in the dependent variable vector. Defaults to True.
    :type drop_NA: bool,optional
    :param method: Which method to use to estimate the coefficients - supports "Newton", "BFGS", and "L-BFGS-B". In case of the former, ``self.family`` needs to implement :func:`gradient` and :func:`hessian`. Defaults to "Newton"
    :type method: str,optional
    :param seed: Seed to use for random parts of the correction. Defaults to None
    :type seed: int,optional
    :param bfgs_options: Any additional keyword arguments that should be passed on to the call of :func:`scipy.optimize.minimize`. If none are provided, the ``gtol`` argument will be initialized to 1e-3. Note also, that in any case the ``maxiter`` argument is automatically set to 100. Defaults to None.
    :type bfgs_options: key=value,optional
    :raises ValueError: Will throw an error when ``method`` is not one of 'Newton', 'BFGS', 'L-BFGS-B' and a :class:`mssm.models.GSMM` is to be estimated.
    :return: A tuple containing: V - an estimate of the unconditional covariance matrix, LV - the Cholesky of the former, total_edf - smoothness uncertainty corrected edf, total_edf2 - smoothness uncertainty + smoothness bias corrected edf, expected_aic - an "expected version of the aic" based on a weighted sum (using the reml weights) over all computed aics
    :rtype: (scipy.sparse.csc_array,scipy.sparse.csc_array,float,float,float) 
    """
    np_gen = np.random.default_rng(seed)

    family = model.family

    if not grid_type in ["GS","JJJ1","JJJ2","JJJ3"]:
        raise ValueError("'grid_type' has to be set to one of 'GS', 'JJJ1', 'JJJ2', or 'JJJ3'.")

    if isinstance(family,GENSMOOTHFamily):
        if not bfgs_options:
            bfgs_options = {"gtol":1e-3}

        if not method in ["Newton", "BFGS", "L-BFGS-B"]:
            raise ValueError("'method' needs to be set to one of 'Newton', 'BFGS', 'L-BFGS-B'.")

    if isinstance(family,Family):
        nPen = len(model.formula.penalties)
        rPen = copy.deepcopy(model.formula.penalties)
        S_emb,_,_,_ = compute_S_emb_pinv_det(model.hessian.shape[1],model.formula.penalties,"svd")

    else: # GAMMLSS and GSMM case
        nPen = len(model.overall_penalties)
        rPen = copy.deepcopy(model.overall_penalties)
        S_emb,_,_,_ = compute_S_emb_pinv_det(model.hessian.shape[1],model.overall_penalties,"svd")
    
    if isinstance(family,Family):
        y = model.formula.y_flat[model.formula.NOT_NA_flat]
        X = model.get_mmat()

        orig_scale = family.scale
        if family.twopar:
            _,orig_scale = model.get_pars()
    else:
        if drop_NA:
            y = model.formulas[0].y_flat[model.formulas[0].NOT_NA_flat]
        else:
            y = model.formulas[0].y_flat
        Xs = model.get_mmat(drop_NA=drop_NA)
        orig_scale = 1
        init_coef = copy.deepcopy(model.overall_coef)

    Vp = None
    Vpr = None
    if grid_type == "JJJ1" or grid_type == "JJJ2":
        # Approximate Vp via finitie differencing
        Vp, Vpr, Vr, Vrr = estimateVp(model,n_c=n_c,strategy="JJJ1")

        # Compute approximate WPS (2016) correction
        if grid_type == "JJJ1":
            if isinstance(family,Family):
                Vc,Vcc = compute_Vb_corr_WPS(model.lvi,Vpr,Vr,model.hessian,S_emb,model.formula.penalties,model.coef.reshape(-1,1),scale=orig_scale)
            else:
                Vc,Vcc = compute_Vb_corr_WPS(model.overall_lvi,Vpr,Vr,model.hessian,S_emb,model.overall_penalties,model.overall_coef.reshape(-1,1))
                
            if isinstance(family,Family):
                V = Vc + Vcc + ((model.lvi.T@model.lvi)*orig_scale)
            else:
                V = Vc + Vcc + model.lvi.T@model.lvi
        
        if grid_type == "JJJ2": # Refine Vpr estimate
            orig_Vp = copy.deepcopy(Vp)
            Vp = Vpr

    rGrid = []
    if grid_type == 'GS':
        # Build Full prior grid as discussed by Greven & Scheipl in their comment on Wood et al. (2016)
        rGrid = [np.exp(np.linspace(np.log(a),np.log(b),nR)) for _ in range(nPen)]
        rGrid = np.array(list(product(*rGrid)))

    elif grid_type == 'JJJ3':
        # Set up grid of nR equidistant values based on marginal grids that cover range from \lambda/lr to \lambda*lr
        # conditional on all estimated penalty values except the current one.
        
        for pi,pen in enumerate(rPen):
            
            # Set up marginal grid from \lambda/lr to \lambda*lr
            mGrid = np.exp(np.linspace(np.log(max([a,pen.lam/lR])),np.log(min([b,pen.lam*lR])),nR))
            
            # Now create penalty candidates conditional on estimates for all other penalties except current one
            for val in mGrid:
                if abs(val - pen.lam) <= 1e-7:
                    continue
                
                rGrid.append(np.array([val if pii == pi else np_gen.choice(np.exp(np.linspace(np.log(max([a,pen2.lam/lR])),np.log(min([b,pen2.lam*lR])),nR)),size=None) for pii,pen2 in enumerate(rPen)]))
    
    if grid_type != 'GS' and grid_type != "JJJ1":
        # Make sure actual estimate is included once.
        rGrid.append(np.array([pen2.lam for pen2 in rPen]))
        rGrid = np.array(rGrid)
    
    remls = []
    Vs = []
    coefs = []
    edfs = []
    llks = []
    aics = []

    if isinstance(family,Gaussian) and X.shape[1] < 2000 and isinstance(family.link,Identity) and n_c > 1 and grid_type != "JJJ1": # Parallelize grid search
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

            if grid_type == "JJJ2" or grid_type == "JJJ3":
                Linvs = list(Linvs)
                coefs = list(coefs)
                remls = list(remls)
                scales = list(scales)
                edfs = list(edfs)
                llks = list(llks)
                aics = list(aics)

    elif grid_type != "JJJ1": # Better to parallelize inverse computation necessary to obtain Vb
        enumerator = rGrid
        if verbose and grid_type == "JJJ3":
            enumerator = tqdm(rGrid)
        
        rGridx = 0
        for r in enumerator:

            # Prepare penalties with current lambda candidate r
            for ridx,rc in enumerate(r):
                rPen[ridx].lam = rc
            
            # Now compute REML - and all other terms needed for correction proposed by Greven & Scheipl (2017)
            if isinstance(family,Family):
                reml,LP,Pr,coef,scale,edf,llk = compute_reml_candidate_GAMM(family,y,X,rPen,n_c)
                coef = coef.reshape(-1,1)

                # Form VB, first solve LP{^-1}
                LPinv = compute_Linv(LP,n_c)
                Linv = apply_eigen_perm(Pr,LPinv)

                # Can already compute first term from correction
                Vb = Linv.T@Linv*scale
                Vb += coef@coef.T
            else:
                try:
                    reml,V,coef,edf,llk = compute_REML_candidate_GSMM(family,y,Xs,rPen,init_coef,len(init_coef),model.coef_split_idx,n_c=n_c,method=method,**bfgs_options)
                    coef = coef.reshape(-1,1)
                except:
                    warnings.warn(f"Unable to compute REML score for sample {r}. Skipping.")
                    rGrid = np.delete(rGrid,rGridx,0)
                    continue

                # Can already compute first term from correction
                Vb = V + coef@coef.T

            rGridx += 1 # Only if we actually reach this point..

            # and aic under current penalty
            aic = -2*llk + 2*edf

            # Now collect what we need for the remaining terms
            Vs.append(Vb)
            coefs.append(coef)
            remls.append(reml)
            edfs.append(edf)
            llks.append(llk)
            aics.append(aic)

    if grid_type != "GS" and grid_type != "JJJ1":
        # Iteratively estimate/refine (JJJ3 vs JJJ2) Vp - covariance matrix of log(\lambda) to guide further REML grid sampling
        if isinstance(family,Family):
            ep = np.log(np.array([min(b,max(a,pen.lam)) for pen in model.formula.penalties]).reshape(-1,1))
        else:
            ep = np.log(np.array([min(b,max(a,pen.lam)) for pen in model.overall_penalties]).reshape(-1,1))
        #print(ep)
        # Get first estimate for Vp based on samples collected so far
        if grid_type == "JJJ3":
            Vp = updateVp(ep,remls,rGrid)

        # Now continuously update Vp and generate more REML samples in the process
        n_est = nR
        
        if isinstance(family,Gaussian) and X.shape[1] < 2000 and isinstance(family.link,Identity) and n_c > 1: # Parallelize grid search
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
                    p_sample = []
                    while len(p_sample) == 0:
                        p_sample = scp.stats.multivariate_t.rvs(loc=np.ndarray.flatten(ep),shape=Vp,df=df,size=n_est,random_state=seed)
                        p_sample = np.exp(p_sample)

                        if len(np.ndarray.flatten(ep)) == 1: # Single lambda parameter in model
                        
                            if n_est == 1: # and single sample (n_est==1) - so p_sample needs to be shape (1,1)
                                p_sample = np.array([p_sample])

                            p_sample = p_sample.reshape(n_est,1) # p_sample needs to be shape (n_est,1)
                            
                        elif n_est == 1: # multiple lambdas - so p_sample needs to be shape (1,n_lambda)
                            p_sample = np.array([p_sample]).reshape(1,-1)

                        # Re-sample values we have encountered before.
                        minDiag = 0.1*min(np.sqrt(Vp.diagonal()))
                        for lami in range(p_sample.shape[0]):
                            while np.any(np.max(np.abs(rGrid - p_sample[lami]),axis=1) < minDiag) or np.any(p_sample[lami] < 1e-9) or np.any(p_sample[lami] > 1e12):
                                if not seed is None:
                                    seed += 1
                                p_sample[lami] = np.exp(scp.stats.multivariate_t.rvs(loc=np.ndarray.flatten(ep),shape=Vp,df=df,size=1,random_state=seed))

                        if not seed is None:
                            seed += 1

                    
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
                    Vp_next = updateVp(ep,remls,rGrid)
                    if grid_type == "JJJ3":
                        Vp = Vp_next
                    elif grid_type == "JJJ2":
                        Vp = 0.99*Vp + 0.01*Vp_next
            
        else:
            enumerator = range(nR*len(ep))
            if verbose:
                enumerator = tqdm(enumerator)
            for sp in enumerator:

                # Generate next \lambda values for which to compute REML, and Vb
                p_sample = []
                while len(p_sample) == 0:
                    p_sample = scp.stats.multivariate_t.rvs(loc=np.ndarray.flatten(ep),shape=Vp,df=df,size=n_est,random_state=seed)
                    p_sample = np.exp(p_sample)

                    if len(np.ndarray.flatten(ep)) == 1: # Single lambda parameter in model
                        
                        if n_est == 1: # and single sample (n_est==1) - so p_sample needs to be shape (1,1)
                            p_sample = np.array([p_sample])

                        p_sample = p_sample.reshape(n_est,1) # p_sample needs to be shape (n_est,1)
                        
                    elif n_est == 1: # multiple lambdas - so p_sample needs to be shape (1,n_lambda)
                        p_sample = np.array([p_sample]).reshape(1,-1)
                    
                    # Re-sample values we have encountered before.
                    minDiag = 0.1*min(np.sqrt(Vp.diagonal()))
                    for lami in range(p_sample.shape[0]):
                        
                        while np.any(np.max(np.abs(rGrid - p_sample[lami]),axis=1) < minDiag) or np.any(p_sample[lami] < 1e-9) or np.any(p_sample[lami] > 1e12):
                            if not seed is None:
                                seed += 1
                            p_sample[lami] = np.exp(scp.stats.multivariate_t.rvs(loc=np.ndarray.flatten(ep),shape=Vp,df=df,size=1,random_state=seed))

                    if not seed is None:
                        seed += 1

                for ps in p_sample:
                    for ridx,rc in enumerate(ps):
                        rPen[ridx].lam = rc
                    
                    if isinstance(family,Family):
                        reml,LP,Pr,coef,scale,edf,llk = compute_reml_candidate_GAMM(family,y,X,rPen,n_c)
                        coef = coef.reshape(-1,1)

                        # Form VB, first solve LP{^-1}
                        LPinv = compute_Linv(LP,n_c)
                        Linv = apply_eigen_perm(Pr,LPinv)

                        # Can already compute first term from correction
                        Vb = Linv.T@Linv*scale
                        Vb += coef@coef.T
                    else:
                        try:
                            reml,V,coef,edf,llk = compute_REML_candidate_GSMM(family,y,Xs,rPen,init_coef,len(init_coef),model.coef_split_idx,n_c=n_c,method=method,**bfgs_options)
                        except:
                            warnings.warn(f"Unable to compute REML score for sample {np.exp(ps)}. Skipping.")
                            continue

                        coef = coef.reshape(-1,1)

                        # Can already compute first term from correction
                        Vb = V + coef@coef.T

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
                Vp_next = updateVp(ep,remls,rGrid)
                if grid_type == "JJJ3":
                    Vp = Vp_next
                elif grid_type == "JJJ2":
                    Vp = 0.99*Vp + 0.01*Vp_next
    
    if grid_type == "JJJ2":
        # Re-compute root of regularized VP (at this point Vp actually holds refined Vpr)
        eig, U =scp.linalg.eigh(Vp)
        ire = np.zeros_like(eig)
        ire[eig > 0] = np.sqrt(eig[eig > 0])
        Vrr = np.diag(ire)@U.T # Root of refined regularized Vp

        # Make sure Vp is original
        Vpr = copy.deepcopy(Vp)
        Vp = orig_Vp

        # Compute approximate WPS (2016) correction
        if V_shrinkage_weight > 0:
            if isinstance(family,Family):
                Vc,Vcc = compute_Vb_corr_WPS(model.lvi,Vpr,Vr,model.hessian,S_emb,model.formula.penalties,model.coef.reshape(-1,1),scale=orig_scale)
            else:
                Vc,Vcc = compute_Vb_corr_WPS(model.overall_lvi,Vpr,Vr,model.hessian,S_emb,model.overall_penalties,model.overall_coef.reshape(-1,1))

    if grid_type != "JJJ1":
        # Compute weights proposed by Greven & Scheipl (2017)
        ws = scp.special.softmax(remls)
        #if grid_type != "GS":
            # Use the estimated normal to compute weights instead.
            #ws2 = scp.stats.multivariate_normal.logpdf(np.log(rGrid),mean=np.ndarray.flatten(ep),cov=Vp)
            #ws = scp.special.softmax(ws2)

        # And simple correction of the edf - essentially taking an upper bound of
        # the edf that could be expected, assuming that the uncertainty around edf is approximately
        # normal.
        upper_edf = model.edf + 2.33*np.sqrt(np.sum(ws*np.power(edfs-np.sum(ws*edfs),2)))
        
        if only_expected_edf:
            if verbose and grid_type != "JJJ1":
                print(f"Correction was based on {rGrid.shape[0]} samples in total.")

            return None,None,None,None,None,None,None,None,upper_edf

        # Now compute \hat{cov(\boldsymbol{\beta}|y)}
        if isinstance(family,Gaussian) and X.shape[1] < 2000 and isinstance(family.link,Identity) and n_c > 1:
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

        if grid_type == "JJJ2" and V_shrinkage_weight > 0:
            # Have WPS correction terms, can compute V based on those and then refine this via V computed above
            # weights are essentially chosen arbitrarily...
            if isinstance(family,Family):
                V = V_shrinkage_weight*(Vc + Vcc + ((model.lvi.T@model.lvi)*orig_scale)) + (1-V_shrinkage_weight)*V
            else:
                V = V_shrinkage_weight*(Vc + Vcc + model.lvi.T@model.lvi) + (1-V_shrinkage_weight)*V
            
    else:
        upper_edf = None

    # Check V is full rank - can use LV for sampling as well..
    LV,code = cpp_chol(scp.sparse.csc_array(V))
    if code != 0:
        raise ValueError("Failed to estimate unconditional covariance matrix.")

    # Compute corrected edf (e.g., for AIC; Wood, Pya, & Saefken, 2016)
    total_edf = None
    if form_t or form_t1:
        if isinstance(family,Family):
            if isinstance(family,Gaussian) and isinstance(family.link,Identity): # Strictly additive case
                F = V@((X.T@X)/orig_scale)
            else: # Generalized case
                W = model.Wr@model.Wr
                F = V@((X.T@W@X)/orig_scale)
        else: # GSMM/GAMLSS case
            F = V@(-1*model.hessian)

        edf = F.diagonal()
        total_edf = F.trace()
    
    # In mgcv, an upper limit is enforced on edf and total_edf when they are uncertainty corrected - based on t1 in section 6.1.2 of Wood (2017)
    # so the same is done here.
    if grid_type == "JJJ1" or grid_type == "JJJ2":
        if isinstance(family,Family):
            if isinstance(family,Gaussian) and isinstance(family.link,Identity): # Strictly additive case
                ucF = (model.lvi.T@model.lvi)@((X.T@X))
            else: # Generalized case
                W = model.Wr@model.Wr
                ucF = (model.lvi.T@model.lvi)@((X.T@W@X))
        else: # GSMM/GAMLSS case
            ucF = (model.lvi.T@model.lvi)@(-1*model.hessian)

        total_edf2 = 2*model.edf - (ucF@ucF).trace()
        if total_edf > total_edf2:
            #print(edf)
            total_edf = total_edf2
            edf = None

    # Compute uncertainty corrected smoothness bias corrected edf (t1 in section 6.1.2 of Wood, 2017)
    edf2 = None
    total_edf2 = None
    if form_t1:
        edf2 = 2*edf - (F@F).diagonal()
        total_edf2 = 2*total_edf - (F@F).trace()

    if verbose and grid_type != "JJJ1":
        print(f"Correction was based on {rGrid.shape[0]} samples in total.")    

    return V,LV,Vp,Vpr,edf,total_edf,edf2,total_edf2,upper_edf