import numpy as np
import scipy as scp
import math
import warnings
from itertools import permutations,product,repeat
import copy
from ..python.gamm_solvers import cpp_backsolve_tr,compute_S_emb_pinv_det,cpp_chol,cpp_solve_coef,update_scale_edf,compute_Linv,apply_eigen_perm,tqdm,managers,shared_memory,cpp_solve_coefXX,update_PIRLS,PIRLS_pdat_weights,correct_coef_step,update_coef_gen_smooth,cpp_cholP,update_coef_gammlss,compute_lgdetD_bsb,calculate_edf,compute_eigen_perm,cpp_dChol,computeHSR1,computeH,update_coef,deriv_transform_mu_eta,deriv_transform_eta_beta
from ..python.formula import reparam,map_csc_to_eigen,mp,translate_sparse
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

def compute_reml_candidate_GAMM(family,y,X,penalties,n_c=10,offset=0,init_eta=None,method='Chol',compute_inv=False,origNH=None):
    """
    Allows to evaluate REML criterion (e.g., Wood, 2011; Wood, 2016) efficiently for
    a set of \lambda values.

    Internal function used for computing the correction applied to the edf for the GLRT - based on Wood (2017) and Wood et al., (2016).

    See :func:`REML` function for more details.
    """

    S_emb,_,S_root,_ = compute_S_emb_pinv_det(X.shape[1],penalties,"svd",method != 'Chol')

    # Need pseudo-data only in case of GAM
    z = None
    Wr = None
    mu_inval = None
   
    try:
        if isinstance(family,Gaussian) and isinstance(family.link,Identity):
            # AMM - directly solve for coef
            eta,mu,coef,Pr,_,LP,keep,drop = update_coef(y-offset,X,X,family,S_emb,S_root,n_c,None,0)
            nH = (X.T@X).tocsc()
        else:
            # GAMM - have to repeat Newton step
            yb = y
            Xb = X
            
            if init_eta is None:
                mu = family.init_mu(y)
                eta = family.link.f(mu)
            else:
                eta = init_eta
                mu = family.link.fi(eta)

            # First pseudo-dat iteration
            yb,Xb,z,Wr = update_PIRLS(y,yb,mu,eta-offset,X,Xb,family)

            # Solve coef
            eta,mu,coef,Pr,_,LP,keep,drop = update_coef(yb,X,Xb,family,S_emb,S_root,n_c,None,offset)

            # Now repeat until convergence
            inval = np.isnan(mu).flatten()
            dev = family.deviance(y[inval == False],mu[inval == False])
            pen_dev = dev + coef.T @ S_emb @ coef

            for newt_iter in range(500):
                    
                # Perform step-length control for the coefficients (repeat step 3 in Wood, 2017)
                if newt_iter > 0:
                    dev,pen_dev,mu,eta,coef = correct_coef_step(coef,n_coef,dev,pen_dev,c_dev_prev,family,eta,mu,y,X,len(penalties),S_emb,None,n_c,offset)

                # Update PIRLS weights
                yb,Xb,z,Wr = update_PIRLS(y,yb,mu,eta-offset,X,Xb,family)

                # Convergence check for inner loop
                if newt_iter > 0:
                    dev_diff_inner = abs(pen_dev - c_dev_prev)
                    if dev_diff_inner < 1e-9*pen_dev or newt_iter == 499:
                        break

                c_dev_prev = pen_dev

                # Now propose next set of coefficients
                eta,mu,n_coef,Pr,_,LP,keep,drop = update_coef(yb,X,Xb,family,S_emb,S_root,n_c,None,offset)

                # Update deviance & penalized deviance
                inval = np.isnan(mu).flatten()
                dev = family.deviance(y[inval == False],mu[inval == False])
                pen_dev = dev + n_coef.T @ S_emb @ n_coef
            
            # At this point, Wr/z might not match the dimensions of y and X, because observations might be
            # excluded at convergence. eta and mu are of correct dimension, so we need to re-compute Wr - this
            # time with a weight of zero for any dropped obs.
            inval_check =  np.any(np.isnan(z))

            if inval_check:
                _, w, inval = PIRLS_pdat_weights(y,mu,eta-offset,family)
                w[inval] = 0

                # Re-compute weight matrix
                Wr_fix = scp.sparse.spdiags([np.sqrt(np.ndarray.flatten(w))],[0])
            else:
                Wr_fix = Wr
                
            W = Wr_fix@Wr_fix
            nH = (X.T@W@X).tocsc() 
        
        # Dropped some coef, needs to be reflected in nH
        if drop is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nH[:,drop] = 0
                nH[drop,:] = 0

        # Get edf and optionally estimate scale (scale will be kept at fixed (e.g., 1) for Generalized case)
        #InvCholXXSP = compute_Linv(LP,n_c)
        _,_,edf,_,_,scale = update_scale_edf(y,z,eta,Wr,X.shape[0],X.shape[1],LP,None,Pr,None,family,penalties,keep,drop,n_c)
        #print(edf-(InvCholXXS.T@InvCholXXS@nH).trace())
        #edf = (InvCholXXS.T@InvCholXXS@nH).trace()

        if family.twopar:
            if mu_inval is None:
                llk = family.llk(y,mu,scale)
            else:
                llk = family.llk(y[mu_inval == False],mu[mu_inval == False],scale)
        else:
            if mu_inval is None:
                llk = family.llk(y,mu)
            else:
                llk = family.llk(y[mu_inval == False],mu[mu_inval == False])

        # Now compute REML for candidate
        reml = REML(llk,nH/scale,coef,scale,penalties)

        Linv = None
        if compute_inv or origNH is not None:
            LPinv = compute_Linv(LP,n_c)
            Linv = apply_eigen_perm(Pr,LPinv)

            # Dropped some terms, need to insert zero columns and rows for dropped coefficients
            if Linv.shape[1] < X.shape[1]:
                Linvdat,Linvrow,Linvcol = translate_sparse(Linv)
            
                Linvrow = keep[Linvrow]
                Linvcol = keep[Linvcol]

                Linv = scp.sparse.csc_array((Linvdat,(Linvrow,Linvcol)),shape=(X.shape[1],X.shape[1]))
            
        if origNH is not None:
            # Compute trace for tau2
            if isinstance(family,Gaussian) and isinstance(family.link,Identity):
                edf *= (scale/origNH)
            else:
                edf = (Linv@origNH@Linv.T).trace()*scale
   
    except:
        return -np.inf,scp.sparse.csc_matrix((len(coef), len(coef))),scp.sparse.csc_matrix((len(coef), len(coef))),list(range(len(coef))),coef.reshape(-1,1),scale,edf,-np.inf

    return reml,Linv,LP,Pr,coef.reshape(-1,1),scale,edf,llk

def compute_REML_candidate_GSMM(family,y,Xs,penalties,coef,n_coef,coef_split_idx,method="Chol",conv_tol=1e-7,n_c=10,bfgs_options={},origNH=None):
    """
    Allows to evaluate REML criterion (e.g., Wood, 2011; Wood, 2016) efficiently for
    a set of \lambda values for a GSMM or GAMMLSS.

    Internal function used for computing the correction applied to the edf for the GLRT - based on Wood (2017) and Wood et al., (2016).

    See :func:`REML` function for more details.
   """

    # Build current penalties
    S_emb,S_pinv,_,FS_use_rank = compute_S_emb_pinv_det(n_coef,penalties,"svd")

    try:
        if isinstance(family,GENSMOOTHFamily): # GSMM
            
            # Compute likelihood for current estimate
            c_llk = family.llk(coef,coef_split_idx,y,Xs)

            __old_opt = None
            if method == "qEFS":
                __old_opt = scp.optimize.LbfgsInvHessProduct(np.array([1]).reshape(1,1),np.array([1]).reshape(1,1))
                __old_opt.init = False
                __old_opt.omega = 1
                __old_opt.method = 'qEFS'
                __old_opt.form = 'SR1'
                __old_opt.bfgs_options = bfgs_options
            
            coef,H,L,LV,c_llk,_,_,_,_ = update_coef_gen_smooth(family,y,Xs,coef,
                                                                coef_split_idx,S_emb,None,None,None,None,
                                                                c_llk,0,1000,
                                                                1000,conv_tol,method,None,None,__old_opt)
            
            if method == 'qEFS':
                
                # Get an approximation of the Hessian of the likelihood
                if LV.form == 'SR1':
                    H = -1*computeHSR1(LV.sk,LV.yk,LV.rho,scp.sparse.identity(len(coef),format='csc')*LV.omega,omega=LV.omega,make_psd=True)
                else:
                    H = -1*computeH(LV.sk,LV.yk,LV.rho,scp.sparse.identity(len(coef),format='csc')*LV.omega,omega=LV.omega,make_psd=True)

                # Get Cholesky factor of approximate inverse of penalized hessian (needed for CIs)
                pH = scp.sparse.csc_array((-1*H) + S_emb)
                Lp, Pr, _ = cpp_cholP(pH)
                LVp0 = compute_Linv(Lp,10)
                LV = apply_eigen_perm(Pr,LVp0)
            
            V = LV.T @ LV # inverse of hessian of penalized likelihood
            nH = -1*H # negative hessian of likelihood

        else: # GAMMLSS
            split_coef = np.split(coef,coef_split_idx)

            # Initialize etas and mus
            etas = [Xs[i]@split_coef[i] for i in range(family.n_par)]
            mus = [family.links[i].fi(etas[i]) for i in range(family.n_par)]

            c_llk = family.llk(y,*mus)

            # Estimate coefficients
            coef,split_coef,mus,etas,H,L,LV,c_llk,_,_,_,_ = update_coef_gammlss(family,mus,y,Xs,coef,
                                                                                coef_split_idx,S_emb,None,None,None,None,
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

        total_edf,_, _ = calculate_edf(None,None,LV,penalties,lgdetDs,n_coef,n_c,None,S_emb)

        if origNH is not None:
            # Compute trace for tau2
            total_edf = (LV@origNH@LV.T).trace()
    
    except:
        return -np.inf,scp.sparse.csc_matrix((len(coef), len(coef))),scp.sparse.csc_matrix((len(coef), len(coef))),coef.reshape(-1,1),total_edf,-np.inf

    return reml,V,LV,coef.reshape(-1,1),total_edf,c_llk


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
       raise ValueError(f"Failed to compute REML for lambda: {[pen.lam for pen in penalties]}.")
  
   #print(((P@Pr.T@L) @ (P@Pr.T@L).T - H_pen).max())
   # Can ignore Pr for det computation, because det(Pr)*det(Pr.T)=1
   lgdetXXS = 2*np.log((P@L).diagonal()).sum()

   # Done
   return reml + lgdetS/2 - lgdetXXS/2 + (Mp*np.log(2*np.pi))/2

def estimateVp(model,nR = 250,grid_type = 'JJJ1',a=1e-7,b=1e7,df=40,n_c=10,drop_NA=True,method="Chol",Vp_fidiff=False,use_importance_weights=True,prior=None,seed=None,**bfgs_options):
    """Estimate covariance matrix :math:`\mathbf{V}^{\\boldsymbol{\\rho}}` of posterior for :math:`\mathbf{p} = log(\\boldsymbol{\lambda})`.
    
    Either :math:`\mathbf{V}^{\\boldsymbol{\\rho}}` is based on finite difference approximation or on a PQL approximation (see ``grid_type`` parameter), or it is estimated via numerical
    integration similar to what is done in the :func:`correct_VB` function (this is done when ``grid_type=='JJJ2'``; see the aforementioned function for details).

    Example::

        # Simulate some data for a Gaussian model
        sim_fit_dat = sim3(n=500,scale=2,c=1,family=Gaussian(),seed=21)

        # Now fit nested models
        sim_fit_formula = Formula(lhs("y"),
                                    [i(),f(["x0"],nk=20,rp=0),f(["x1"],nk=20,rp=0),f(["x2"],nk=20,rp=0),f(["x3"],nk=20,rp=0)],
                                    data=sim_fit_dat,
                                    print_warn=False)

        model = GAMM(sim_fit_formula,Gaussian())
        model.fit(exclude_lambda=False,progress_bar=False,max_outer=100)

        # Compute correction from Wood et al. (2016) - will be approximate for more generic models
        # Vp is approximate covariance matrix of log regularization parameters
        # Vpr is regularized version of the former
        # Ri is a root of covariance matrix of log regularization parameters
        # Rir is a root of regularized version of covariance matrix of log regularization parameters
        # ep will be an estimate of the mean of the marginal posterior of log regularization parameters (for ``grid_type="JJJ1"`` this will simply be the log of the estimated regularization parameters)
        Vp, Vpr, Ri, Rir, ep = estimateVp(model,grid_type="JJJ1",verbose=True,seed=20)


        # Compute MC estimate for generic model and given prior
        prior = DummyRhoPrior(b=np.log(1e12)) # Set up uniform prior
        Vp_MC, Vpr_MC, Ri_MC, Rir_MC, ep_MC = estimateVp(model,strategy="JJJ2",verbose=True,seed=20,use_importance_weights=True,prior=prior)


    References:
     - https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices
     - Greven, S., & Scheipl, F. (2016). Comment on: Smoothing Parameter and Model Selection for General Smooth Models

    :param model: GAMM, GAMMLSS, or GSMM model (which has been fitted) for which to estimate :math:`\mathbf{V}`
    :type model: GAMM or GAMMLSS or GSMM
    :param nR: In case ``grid!="JJJ1"``, ``nR`` samples/reml scores are generated/computed to numerically evaluate the expectations necessary for the uncertainty correction, defaults to 250
    :type nR: int, optional
    :param grid_type: How to compute the smoothness uncertainty correction. Setting ``grid_type="JJJ1"`` means a PQL or finite difference approximation is obtained. Setting ``grid_type="JJJ2"`` means numerical integration is performed - see :func:`correct_VB` for details , defaults to 'JJJ1'
    :type grid_type: str, optional
    :param a: Any of the :math:`\lambda` estimates obtained from ``model`` (used to define the mean for the posterior of :math:`\\boldsymbol{\\rho}|y \sim N(log(\hat{\\boldsymbol{\\rho}}),\mathbf{V}^{\\boldsymbol{\\rho}})` used to sample ``nR`` candidates) which are smaller than this are set to this value as well, defaults to 1e-7 the minimum possible estimate
    :type a: float, optional
    :param b: Any of the :math:`\lambda` estimates obtained from ``model`` (used to define the mean for the posterior of :math:`\\boldsymbol{\\rho}|y \sim N(log(\hat{\\boldsymbol{\\rho}}),\mathbf{V}^{\\boldsymbol{\\rho}})` used to sample ``nR`` candidates) which are larger than this are set to this value as well, defaults to 1e7 the maximum possible estimate
    :type b: float, optional
    :param df: Degrees of freedom used for the multivariate t distribution used to sample the next set of candidates. Setting this to ``np.inf`` means a multivariate normal is used for sampling, defaults to 40
    :type df: int, optional
    :param n_c: Number of cores to use during parallel parts of the correction, defaults to 10
    :type n_c: int, optional
    :param drop_NA: Whether to drop rows in the **model matrices** corresponding to NAs in the dependent variable vector. Defaults to True.
    :type drop_NA: bool,optional
    :param method: Which method to use to solve for the coefficients (and smoothing parameters). The default ("Chol") relies on Cholesky decomposition. This is extremely efficient but in principle less stable, numerically speaking. For a maximum of numerical stability set this to "QR/Chol". In that case a QR decomposition is used - which is first pivoted to maximize sparsity in the resulting decomposition but also pivots for stability in order to get an estimate of rank defficiency. A Cholesky is than used using the combined pivoting strategy obtained from the QR. This takes substantially longer. If this is set to ``'qEFS'``, then the coefficients are estimated via quasi netwon and the smoothing penalties are estimated from the quasi newton approximation to the hessian. This only requieres first derviative information. Defaults to "Chol".
    :type method: str,optional
    :param Vp_fidiff: Whether to rely on a finite difference approximation to compute :math:`\mathbf{V}^{\\boldsymbol{\\rho}}` or on a PQL approximation. The latter is exact for Gaussian and canonical GAMs and far cheaper if many penalties are to be estimated. Defaults to False (PQL approximation)
    :type Vp_fidiff: bool,optional
    :param use_importance_weights: Whether to rely importance weights to compute the numerical integration when ``grid_type != 'JJJ1'`` or on the log-densities of :math:`\mathbf{V}^{\\boldsymbol{\\rho}}` - the latter assumes that the unconditional posterior is normal. Defaults to True (Importance weights are used)
    :type use_importance_weights: bool,optional
    :param prior: An (optional) instance of an arbitrary class that has a ``.logpdf()`` method to compute the prior log density of a sampled candidate. If this is set to ``None``, the prior is assumed to coincide with the proposal distribution, simplifying the importance weight computation. Ignored when ``use_importance_weights=False``. Defaults to None
    :type prior: any, optional
    :param recompute_H: Whether or not to re-compute the Hessian of the log-likelihood at an estimate of the mean of the Bayesian posterior :math:`\\boldsymbol{\\beta}|y` before computing the (uncertainty/bias corrected) edf. Defaults to False
    :type recompute_H: bool, optional
    :param seed: Seed to use for random parts of the correction. Defaults to None
    :type seed: int,optional
    :param bfgs_options: Any additional keyword arguments that should be passed on to the call of ``scipy.optimize.minimize``. If none are provided, the ``gtol`` argument will be initialized to 1e-3. Note also, that in any case the ``maxiter`` argument is automatically set to 100. Defaults to None.
    :type bfgs_options: key=value,optional
    :return: A tuple with 4 elements: an estimate of the covariance matrix of the posterior for :math:`\mathbf{p} = log(\\boldsymbol{\lambda})`, a regularized version of the former, a root of the covariance matrix, a root of the regularized covariance matrix, and an estimate of the mean of the posterior
    :rtype: (numpy.array, numpy.array, numpy.array, numpy.array)
    """
    np_gen = np.random.default_rng(seed)

    family = model.family

    if not grid_type in ["JJJ1","JJJ2"]:
        raise ValueError("'grid_type' has to be set to one of 'JJJ1', 'JJJ2'.")

    if isinstance(family,GENSMOOTHFamily):
        if not bfgs_options:
            bfgs_options = {"ftol":1e-9,
                            "maxcor":30,
                            "maxls":100}

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
        X = Xs[0]
        orig_scale = 1
        init_coef = copy.deepcopy(model.overall_coef)
    
    if grid_type == "JJJ1"  or grid_type == "JJJ2":
        # Approximate Vp via finite differencing or PQL approximation of negative REML.

        # Set up mean log-smoothing penalty vector - ignoring any a and b limits provided.
        if isinstance(family,Family):
            ep = np.log(np.array([pen.lam for pen in model.formula.penalties]).reshape(-1,1))
        else:
            ep = np.log(np.array([pen.lam for pen in model.overall_penalties]).reshape(-1,1))

        #from numdifftools import Hessian
        if Vp_fidiff:
            def reml_wrapper(rho,family,y,X,rPen,*reml_args,**reml_kwargs):
                
                for peni in range(len(rho)):
                    rPen[peni].lam = np.exp(rho[peni])
                
                if isinstance(family,Family):
                    reml,_,_,_,_,_,_,_ = compute_reml_candidate_GAMM(family,y,X,rPen,*reml_args,**reml_kwargs)
                else:
                    reml,_,_,_,_,_ = compute_REML_candidate_GSMM(family,y,X,rPen,*reml_args,**reml_kwargs)
                
                return -reml
            
            if isinstance(family,Family):
                nHp = central_hessian(ep.flatten(),reml_wrapper,family,y,X,rPen,n_c,model.offset,model.pred,method)
                #nHp = Hessian(reml_wrapper)(ep.flatten(),family,y,X,rPen,n_c,model.offset,model.pred)
            else:
                nHp = central_hessian(ep.flatten(),reml_wrapper,family,y,Xs,rPen,init_coef,len(init_coef),model.coef_split_idx,n_c=n_c,method=method,bfgs_options=bfgs_options)
            
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
        else:
            # Take PQL approximation instead
            Vp, Vpr, Ri, Rir, _, _ = compute_Vp_WPS(model.lvi if isinstance(family,Family) else model.overall_lvi,
                                                 model.hessian,
                                                 S_emb,
                                                 model.formula.penalties if isinstance(family,Family) else model.overall_penalties,
                                                 model.coef.reshape(-1,1) if isinstance(family,Family) else model.overall_coef.reshape(-1,1),
                                                 scale=orig_scale if isinstance(family,Family) else 1)
        
        if grid_type == "JJJ1":
            return Vp,Vpr,Ri,Rir,ep
            
    # Strategies JJJ2
    rGrid = np.array([])
    remls = []
    # Generate \lambda values from Vpr for which to compute REML, and Vb

    # First recompute mean, this time accepting limits imposed by a and b
    if isinstance(family,Family):
        ep = np.log(np.array([min(b,max(a,pen.lam)) for pen in model.formula.penalties]).reshape(-1,1))
    else:
        ep = np.log(np.array([min(b,max(a,pen.lam)) for pen in model.overall_penalties]).reshape(-1,1))
    #print(ep)

    n_est = nR
    
    if X.shape[0] < 1e5 and X.shape[1] < 2000 and n_c > 1: # Parallelize grid search
        # Generate next \lambda values for which to compute REML, and Vb
        p_sample = scp.stats.multivariate_t.rvs(loc=np.ndarray.flatten(ep),shape=Vpr,df=df,size=n_est,random_state=seed)
        p_sample = np.exp(p_sample)

        if len(np.ndarray.flatten(ep)) == 1: # Single lambda parameter in model
        
            if n_est == 1: # and single sample (n_est==1) - so p_sample needs to be shape (1,1)
                p_sample = np.array([p_sample])

            p_sample = p_sample.reshape(n_est,1) # p_sample needs to be shape (n_est,1)
            
        elif n_est == 1: # multiple lambdas - so p_sample needs to be shape (1,n_lambda)
            p_sample = np.array([p_sample]).reshape(1,-1)

        minDiag = 0.1*min(np.sqrt(Vpr.diagonal()))
        # Make sure actual estimate is included once.
        if np.any(np.max(np.abs(p_sample - np.array([pen2.lam for pen2 in rPen])),axis=1) < minDiag) == False:
            p_sample = np.concatenate((np.array([pen2.lam for pen2 in rPen]).reshape(1,-1),p_sample),axis=0)
        
        if isinstance(family,Gaussian) and isinstance(family.link,Identity): # Strictly additive case
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

                # Now compute reml for new candidates in parallel
                args = zip(repeat(family),repeat(y_mem.name),repeat(dat_mem.name),
                        repeat(ptr_mem.name),repeat(idx_mem.name),repeat(dat_memXX.name),
                        repeat(ptr_memXX.name),repeat(idx_memXX.name),repeat(shape_y),
                        repeat(shape_dat),repeat(shape_ptr),repeat(shape_datXX),repeat(shape_ptrXX),
                        repeat(rows),repeat(cols),repeat(rPen),repeat(model.offset),p_sample)
                
                sample_Linvs, sample_coefs, sample_remls, sample_scales, sample_edfs, sample_llks = zip(*pool.starmap(_compute_VB_corr_terms_MP,args))
                
                remls.extend(list(sample_remls))
                rGrid = p_sample
        else: # all other models

            rPens = []
            for ps in p_sample:
                rcPen = copy.deepcopy(rPen)
                for ridx,rc in enumerate(ps):
                    rcPen[ridx].lam = rc
                rPens.append(rcPen)

            if isinstance(family,Family):
                origNH = None
                args = zip(repeat(family),repeat(y),repeat(X),rPens,
                            repeat(1),repeat(model.offset),repeat(None),
                            repeat(method),repeat(True),repeat(origNH))
                with mp.Pool(processes=n_c) as pool:
                    sample_remls, sample_Linvs, _, _,sample_coefs, sample_scales, sample_edfs, sample_llks = zip(*pool.starmap(compute_reml_candidate_GAMM,args))
            else:
                origNH = None
                args = zip(repeat(family),repeat(y),repeat(Xs),rPens,
                            repeat(init_coef),repeat(len(init_coef)),
                            repeat(model.coef_split_idx),repeat(method),
                            repeat(1e-7),repeat(1),repeat(bfgs_options),
                            repeat(origNH))
                with mp.Pool(processes=n_c) as pool:
                    sample_remls, _, sample_Linvs, sample_coefs, sample_edfs, sample_llks = zip(*pool.starmap(compute_REML_candidate_GSMM,args))
                sample_scales = np.ones(p_sample.shape[0])


            remls.extend(list(sample_remls))
            rGrid = p_sample
            rPens = None
        
    else:
        # Generate \lambda values for which to compute REML, and Vb
        p_sample = scp.stats.multivariate_t.rvs(loc=np.ndarray.flatten(ep),shape=Vpr,df=df,size=n_est,random_state=seed)
        p_sample = np.exp(p_sample)

        if len(np.ndarray.flatten(ep)) == 1: # Single lambda parameter in model
            
            if n_est == 1: # and single sample (n_est==1) - so p_sample needs to be shape (1,1)
                p_sample = np.array([p_sample])

            p_sample = p_sample.reshape(n_est,1) # p_sample needs to be shape (n_est,1)
            
        elif n_est == 1: # multiple lambdas - so p_sample needs to be shape (1,n_lambda)
            p_sample = np.array([p_sample]).reshape(1,-1)
        
        # Make sure actual estimate is included once.
        minDiag = 0.1*min(np.sqrt(Vpr.diagonal()))
        if np.any(np.max(np.abs(p_sample - np.array([pen2.lam for pen2 in rPen])),axis=1) < minDiag) == False:
            p_sample = np.concatenate((np.array([pen2.lam for pen2 in rPen]).reshape(1,-1),p_sample),axis=0)

        for ps in p_sample:
            for ridx,rc in enumerate(ps):
                rPen[ridx].lam = rc
            
            if isinstance(family,Family):
                reml,Linv,LP,Pr,coef,scale,edf,llk = compute_reml_candidate_GAMM(family,y,X,rPen,n_c,model.offset,method=method,compute_inv=True)

            else:
                try:
                    reml,V,_,coef,edf,llk = compute_REML_candidate_GSMM(family,y,Xs,rPen,init_coef,len(init_coef),model.coef_split_idx,n_c=n_c,method=method,bfgs_options=bfgs_options)
                except:
                    warnings.warn(f"Unable to compute REML score for sample {np.exp(ps)}. Skipping.")
                    continue

            remls.append(reml)
            
            if len(rGrid) == 0:
                rGrid = ps.reshape(1,-1)
            else:
                rGrid = np.concatenate((rGrid,ps.reshape(1,-1)),axis=0)

    # Compute weights
    # Compute weights proposed by Greven & Scheipl (2017) - still work under importance sampling case instead of grid case if we assume Vp
    # is prior for \rho|\mathbf{y}.
    if use_importance_weights and prior is None:
        ws = scp.special.softmax(remls)

    elif use_importance_weights and prior is not None: # Standard importance weights (e.g., Branchini & Elvira, 2024)
        logp = prior.logpdf(np.log(rGrid))
        q = scp.stats.multivariate_t(loc=np.ndarray.flatten(ep),shape=Vpr,df=df,allow_singular=True)
        logq = q.logpdf(np.log(rGrid))

        ws = scp.special.softmax(remls + logp - logq)

    else:
        # Normal weights result in cancellation, i.e., just average
        ws = scp.special.softmax(np.ones_like(remls))

    # Now estimare Vp from weights

    # First update mean
    ep = ws[0]* np.log(rGrid[0]).reshape(-1,1)
    for ridx in range(1,rGrid.shape[0]):
        ep += ws[ridx]* np.log(rGrid[ridx]).reshape(-1,1)

    # Now Vp
    Vp = updateVp(ep,ws,rGrid)

    # Compute root of VP
    eig, U =scp.linalg.eigh(Vp)
    ire = np.zeros_like(eig)
    ire[eig > 0] = np.sqrt(eig[eig > 0])
    Ri = np.diag(ire)@U.T # Root of Vp

    Vp = Ri.T@Ri # Make sure Vp is PSD

    # Now, in mgcv a regularized version is computed as well, which essentially sets all positive eigenvalues
    # to a positive minimum. This regularized version is utilized in the smoothness uncertainty correction, so we compute it
    # as well.
    # See: https://github.com/cran/mgcv/blob/aff4560d187dfd7d98c7bd367f5a0076faf129b7/R/gam.fit3.r#L1010
    ire2 = np.zeros_like(eig)
    ire2[eig > 0] = 1/((1/np.sqrt(eig[eig > 0])) + 0.1)
    Rir = np.diag(ire2)@U.T # Root of regularized Vp

    Vpr = Rir.T@Rir

    return Vp, Vpr, Ri, Rir,ep

def updateVp(ep,ws,rGrid):
    """Update covariance matrix of posterior for :math:`\mathbf{p} = log(\\boldsymbol{\lambda})`. REML scores are used to
    approximate expectation, similar to what was suggested by Greven & Scheipl (2016).

    References:
     - https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices
     - Greven, S., & Scheipl, F. (2016). Comment on: Smoothing Parameter and Model Selection for General Smooth Models

    :param ep: Model estimate log(\lambda), i.e., the expectation over rGrid
    :type ep: [float]
    :param ws: weight associated with each log(\lambda) value used for numerical integration
    :type ws: [float]
    :param rGrid: A 2d array, holding all \lambda samples considered so far. Each row is one sample
    :type rGrid: [float]
    :return: An estimate of the covariance matrix of log(\lambda) - 2d array of shape len(mp)*len(mp).
    :rtype: [float]
    """

    wp = (np.log(rGrid[0]).reshape(-1,1) - ep)
    # Vp = E[(wp-ep)(wp-ep)^T] - see Wikipedia
    Vp = ws[0]*(wp @ wp.T)
    for ridx in range(1,rGrid.shape[0]):
        wp = (np.log(rGrid[ridx]).reshape(-1,1) - ep)
        Vp += ws[ridx]*(wp @ wp.T)
    
    return Vp

def _compute_VB_corr_terms_MP(family,address_y,address_dat,address_ptr,address_idx,address_datXX,address_ptrXX,address_idxXX,shape_y,shape_dat,shape_ptr,shape_datXX,shape_ptrXX,rows,cols,rPen,offset,r):
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
   LP, Pr, coef, code = cpp_solve_coef(y-offset,X,S_emb)

   if code != 0:
       raise ValueError("Forming coefficients for specified penalties was not possible.")
   
   eta = (X @ coef).reshape(-1,1) + offset
   
   # Compute scale
   _,_,edf,_,_,scale = update_scale_edf(y,None,eta,None,X.shape[0],X.shape[1],LP,None,Pr,None,family,rPen,None,None,10)

   llk = family.llk(y,eta,scale)

   # Now compute REML for candidate
   reml = REML(llk,XX/scale,coef,scale,rPen)
   coef = coef.reshape(-1,1)

   # Form VB, first solve LP{^-1}
   LPinv = compute_Linv(LP,1)
   Linv = apply_eigen_perm(Pr,LPinv)

   # Now collect what we need for the remaining terms
   return Linv,coef,reml,scale,edf,llk

def compute_Vp_WPS(Vbr,H,S_emb,penalties,coef,scale=1):
    """ Computes the inverse of what is approximately the negative Hessian of the Laplace approximate REML criterion with respect to the log smoothing penalties.

    The derivatives computed are only exact for Gaussian additive models and canonical generalized additive models. For all other models they are in-exact in that they
    assume that the hessian of the log-likelihood does not depend on :math:`\lambda` (or :math:`log(\lambda)`), so they are essentially the PQL derivatives of Wood et al. (2017).
    The inverse computed here acts as an approximation to the covariance matrix of the log smoothing parameters.

    References:
     - Wood, S. N., (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models.
     - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
     - Wood, S. N., Li, Z., Shaddick, G., & Augustin, N. H. (2017). Generalized Additive Models for Gigadata: Modeling the U.K. Black Smoke Network Daily Data.

    :param Vbr: Transpose of root for the estimate for the (unscaled) covariance matrix of :math:`\\boldsymbol{\\beta} | y, \\boldsymbol{\lambda}` - the coefficients estimated by the model.
    :type Vbr: scipy.sparse.csc_array
    :param H: The Hessian of the log-likelihood
    :type H: scipy.sparse.csc_array
    :param S_emb: The weighted penalty matrix.
    :type S_emb: scipy.sparse.csc_array
    :param penalties: A list holding the Lambdaterms estimated for the model.
    :type penalties: [LambdaTerm]
    :param coef: An array holding the estimated regression coefficients. Has to be of shape (-1,1)
    :type coef: numpy.array
    :param scale: Any scale parameter estimated as part of the model. Can be omitted for more generic models beyond GAMMs. Defaults to 1.
    :type scale: float
    """
    # Form nH - the negative hessian of the penalized llk - note, H is scaled by \phi for GAMMs, so have to do the same for S_emb:
    nH = (-1*H) + S_emb/scale

    # Get partial derivatives of coef with respect to log(lambda) - see Wood (2017):
    dBetadRhos = np.zeros((len(coef),len(penalties)))

    # Vbr.T@Vbr*scale = nH^{-1} - Vbr is available in model.lvi or model.overall_lvi after fitting
    for peni,pen in enumerate(penalties):
        # Given in Wood (2017)
        dBetadRhos[:,[peni]] = -pen.lam *  (Vbr.T @ (Vbr @ (pen.S_J_emb @ coef)))

    # Also need to apply re-parameterization from Wood (2011) to the penalties and S_emb
    Sj_reps,S_reps,SJ_term_idx,SJ_idx,S_coefs,Q_reps,_,Mp = reparam(None,penalties,None,option=4)

    # Need the inverse of each re-parameterized S_rep
    S_inv_reps = []
    for S_rep in S_reps:
        LS,code = cpp_chol(S_rep.tocsc())
        if code != 0:
            raise ArithmeticError("Could not compute Cholesky of reparameterized S_rep.")
        LSinv = compute_Linv(LS)
        S_inv_rep = LSinv.T@LSinv
        S_inv_reps.append(S_inv_rep)

    Vp = np.zeros((len(penalties),len(penalties)))

    # Now can accumulate hessian of reml with respect to log(lambda)
    grp_idx = 0
    for peni in range(len(penalties)):
        
        # Keep track of which S_rep/group this S_i belongs to
        for grp_i,grp in enumerate(SJ_term_idx):
            if peni in grp:
                grp_idx = grp_i
                break
        
        for penj in range(peni,len(penalties)):

            gamma = peni == penj

            # Compute first term:
            t1 = -1*dBetadRhos[:,[peni]].T @ (nH @ dBetadRhos[:,[penj]])

            # Derivatives of BSB (Wood et al., 2017 has the correct ones)
            t2 = ((penalties[peni].lam/(scale) * coef.T@penalties[peni].S_J_emb@dBetadRhos[:,[penj]]) + (penalties[penj].lam/(scale) * coef.T@penalties[penj].S_J_emb@dBetadRhos[:,[peni]]))
            
            if gamma:
                t2 += penalties[peni].lam/(2*scale) * coef.T@penalties[peni].S_J_emb@coef
            
            # Now derivative of the log-determinant of S_emb. Only defined if peni==penj or both are in the same group
            t3 = 0
            if gamma or penj in SJ_term_idx[grp_idx]:
                t3 = -1* penalties[peni].lam * penalties[penj].lam * penalties[peni].rep_sj * (S_inv_reps[grp_idx]@Sj_reps[penj].S_J@S_inv_reps[grp_idx]@Sj_reps[peni].S_J).trace()

                if gamma:
                    t3 += penalties[peni].lam * penalties[peni].rep_sj * (S_inv_reps[grp_idx]@Sj_reps[peni].S_J).trace()

                t3 = 0.5*t3
            
            # Now second partial derivative of hessian of negative penalized likelihood with respect to log(lambda) - assuming that H does not
            # depend on log(lambda)
            t4 = 0.5 * ((penalties[peni].S_J_emb*penalties[peni].lam) @ Vbr.T @ Vbr @ (penalties[penj].S_J_emb*penalties[penj].lam) @ Vbr.T @ Vbr).trace()

            # And first
            t5 = 0
            if gamma:
                t5 = 0.5 * (Vbr.T @ Vbr @ (penalties[peni].S_J_emb*penalties[peni].lam)).trace()
            
            # Collect result
            Vpij = t1 - t2 + t3 + t4 - t5
            Vp[peni,penj] = Vpij

            if peni != penj:
                Vp[penj,peni] = Vpij

    # Vp is now simply inv(-Vp) but we should rely on an eigen decomposition so that we can naturally produce a generalized inverse as discussed by
    # WPS (2016).
    Hp = copy.deepcopy(Vp)

    eig, U =scp.linalg.eigh(-Vp)
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
            
    return Vp, Vpr, Ri, Rir, Hp, dBetadRhos


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
    :param penalties: A list holding the Lambdaterms estimated for the model.
    :type penalties: [LambdaTerm]
    :param coef: An array holding the estimated regression coefficients. Has to be of shape (-1,1)
    :type coef: numpy.array
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

class RhoPrior:
    """
    Base class to demonstrate the functionlaity that any prior passed to the correct_VB function has to implement.
    """

    def __init__(self,*args,**kwargs):
        self.args = args
        self.kwargs = kwargs
    
    def logpdf(self, rho:np.ndarray):
        """Compute log density for log smoothing penalty parameters included in rho under this prior.

        :param rho: Numpy array of shape (nR,nrho) containing nR proposed candidate vectors for the nrho log-smoothing parameters.
        :type rho: np.ndarray
        """
        pass


class DummyRhoPrior(RhoPrior):
    """
    Simple uniform prior for rho - the log-smoothing penalty parameters
    """

    def __init__(self, a=np.log(1e-7),b=np.log(1e7)):
        super().__init__(a=a,b=b)
    
    def logpdf(self, rho):
        a = self.kwargs["a"]
        b = self.kwargs["b"]
        
        ld = np.zeros(rho.shape[0])
        ld[(np.min(rho,axis=1) < a) | (np.max(rho,axis=1) > b)] = -np.inf
        return ld


def correct_VB(model,nR = 250,grid_type = 'JJJ1',a=1e-7,b=1e7,df=40,n_c=10,form_t=True,form_t1=False,verbose=False,drop_NA=True,method="Chol",only_expected_edf=False,Vp_fidiff=False,use_importance_weights=True,prior=None,recompute_H=False,seed=None,**bfgs_options):
    """Estimate :math:`\\tilde{\mathbf{V}}`, the covariance matrix of the marginal posterior :math:`\\boldsymbol{\\beta} | y` to account for smoothness uncertainty.
    
    Wood et al. (2016) and Wood (2017) show that when basing conditional versions of model selection criteria or hypothesis
    tests on :math:`\mathbf{V}`, which is the co-variance matrix for the normal approximation to the conditional posterior of :math:`\\boldsymbol{\\beta}` so that
    :math:`\\boldsymbol{\\beta} | y, \\boldsymbol{\lambda} \sim N(\hat{\\boldsymbol{\\beta}},\mathbf{V})`, the tests are severely biased. To correct for this they
    show that uncertainty in :math:`\\boldsymbol{\lambda}` needs to be accounted for. Hence they suggest to base these tests on :math:`\\tilde{\mathbf{V}}`, the covariance matrix
    of the normal approximation to the **marginal posterior** :math:`\\boldsymbol{\\beta} | y`. They show how to obtain an estimate of :math:`\\tilde{\mathbf{V}}`,
    but this requires :math:`\mathbf{V}^{\\boldsymbol{\\rho}}` - an estimate of the covariance matrix of the normal approximation to the posterior of :math:`\\boldsymbol{\\rho}=log(\\boldsymbol{\lambda})`. Computing :math:`\mathbf{V}^{\\boldsymbol{\\rho}}` requires derivatives that are not available
    when using the efs update.

    This function implements multiple strategies to approximately correct for smoothing parameter uncertainty, based on the proposals by  Wood et al. (2016) and Greven & Scheipl (2017). The most straightforward strategy
    (``grid_type = 'JJJ1'``) is to obtain a PQL or finite difference approximation for :math:`\mathbf{V}^{\\boldsymbol{\\rho}}` and to then compute approximately the Wood et al. (2016) correction assuming that higher-order derivatives of the llk are zero (this will be exact
    for Gaussian additive or canonical Generalized models). This is too costly for large sparse multi-level models and not exact for more generic models. The MC based alternative available via ``grid_type = 'JJJ2'`` addresses the first problem (**Important**, set: ``use_importance_weights=False`` and ``only_expected_edf=True``.). The second MC based alternative
    available via ``grid_type = 'JJJ3'`` is most appropriate for more generic models (The ``prior`` argument can be used to specify any prior to be placed on :math:`\\boldsymbol{\\rho}` also you will need to set: ``use_importance_weights=True`` and ``only_expected_edf=False``).
    Both strategies use a PQL or finite difference approximation to :math:`\mathbf{V}^{\\boldsymbol{\\rho}}` to obtain ``nR`` samples from the (normal approximation) to the posterior of :math:`\\boldsymbol{\\rho}`.
    From these samples mssm then estimates :math:`\\tilde{\mathbf{V}}` as described in more detail by Krause et al. (in preparation).

    Example::

        # Simulate some data for a Gaussian model
        sim_fit_dat = sim3(n=500,scale=2,c=1,family=Gaussian(),seed=21)

        # Now fit nested models
        sim_fit_formula = Formula(lhs("y"),
                                    [i(),f(["x0"],nk=20,rp=0),f(["x1"],nk=20,rp=0),f(["x2"],nk=20,rp=0),f(["x3"],nk=20,rp=0)],
                                    data=sim_fit_dat,
                                    print_warn=False)

        model = GAMM(sim_fit_formula,Gaussian())
        model.fit(exclude_lambda=False,progress_bar=False,max_outer=100)


        # Compute correction from Wood et al. (2016) - will be approximate for more generic models
        # V will be approximate covariance matrix of marginal posterior of coefficients
        # LV is Cholesky of the former
        # Vp is approximate covariance matrix of log regularization parameters
        # Vpr is regularized version of the former
        # edf is vector of estimated degrees of freedom (uncertainty corrected) per coefficient
        # total_edf is sum of former (but subjected to upper bounds so this might not be exactly the same as sum(edf))
        # ed2 is optionally smoothness bias corrected version of edf
        # total_edf2 is optionally bias corrected version of total_edf (again subjected to upper bounds)
        # expected_edf is None here but for MC strategies (i.e., ``grid!=1``) will be an estimate of total_edf (**without being subjected to upper bounds**) that does not require forming V (only computed when ``only_expected_edf=True``). 
        # mean_coef is None here but for MC strategies will be an estimate of the mean of the marginal posterior of coefficients, only computed when setting ``recompute_H=True``
        V,LV,Vp,Vpr,edf,total_edf,edf2,total_edf2,expected_edf,mean_coef = correct_VB(model,grid_type="JJJ1",verbose=True,seed=20)


        # Compute MC estimate for generic model and given prior
        prior = DummyRhoPrior(b=np.log(1e12)) # Set up uniform prior
        V_MC,LV_MC,Vp_MC,Vpr_MC,edf_MC,total_edf_MC,edf2_MC,total_edf2_MC,expected_edf_MC,mean_coef_MC = correct_VB(model2,grid_type="JJJ3",verbose=True,seed=20,df=10,prior=prior,recompute_H=True)

    References:
     - Wood, S. N., (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models.
     - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models
     - Greven, S., & Scheipl, F. (2016). Comment on: Smoothing Parameter and Model Selection for General Smooth Models
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

    :param model: GAMM, GAMMLSS, or GSMM model (which has been fitted) for which to estimate :math:`\mathbf{V}`
    :type model: GAMM or GAMMLSS or GSMM
    :param nR: In case ``grid!="JJJ1"``, ``nR`` samples/reml scores are generated/computed to numerically evaluate the expectations necessary for the uncertainty correction, defaults to 250
    :type nR: int, optional
    :param grid_type: How to compute the smoothness uncertainty correction - see above for details, defaults to 'JJJ1'
    :type grid_type: str, optional
    :param a: Any of the :math:`\lambda` estimates obtained from ``model`` (used to define the mean for the posterior of :math:`\\boldsymbol{\\rho}|y \sim N(log(\hat{\\boldsymbol{\\rho}}),\mathbf{V}^{\\boldsymbol{\\rho}})` used to sample ``nR`` candidates) which are smaller than this are set to this value as well, defaults to 1e-7 the minimum possible estimate
    :type a: float, optional
    :param b: Any of the :math:`\lambda` estimates obtained from ``model`` (used to define the mean for the posterior of :math:`\\boldsymbol{\\rho}|y \sim N(log(\hat{\\boldsymbol{\\rho}}),\mathbf{V}^{\\boldsymbol{\\rho}})` used to sample ``nR`` candidates) which are larger than this are set to this value as well, defaults to 1e7 the maximum possible estimate
    :type b: float, optional
    :param df: Degrees of freedom used for the multivariate t distribution used to sample the next set of candidates. Setting this to ``np.inf`` means a multivariate normal is used for sampling, defaults to 40
    :type df: int, optional
    :param n_c: Number of cores to use during parallel parts of the correction, defaults to 10
    :type n_c: int, optional
    :param form_t: Whether or not the smoothness uncertainty corrected edf should be computed, defaults to True
    :type form_t: bool, optional
    :param form_t1: Whether or not the smoothness uncertainty + smoothness bias corrected edf should be computed, defaults to False
    :type form_t1: bool, optional
    :param verbose: Whether to print progress information or not, defaults to False
    :type verbose: bool, optional
    :param drop_NA: Whether to drop rows in the **model matrices** corresponding to NAs in the dependent variable vector. Defaults to True.
    :type drop_NA: bool,optional
    :param method: Which method to use to solve for the coefficients (and smoothing parameters). The default ("Chol") relies on Cholesky decomposition. This is extremely efficient but in principle less stable, numerically speaking. For a maximum of numerical stability set this to "QR/Chol". In that case a QR decomposition is used - which is first pivoted to maximize sparsity in the resulting decomposition but also pivots for stability in order to get an estimate of rank defficiency. A Cholesky is than used using the combined pivoting strategy obtained from the QR. This takes substantially longer. If this is set to ``'qEFS'``, then the coefficients are estimated via quasi netwon and the smoothing penalties are estimated from the quasi newton approximation to the hessian. This only requieres first derviative information. Defaults to "Chol".
    :type method: str,optional
    :param only_expected_edf: Whether to compute edf. by explicitly forming covariance matrix (``only_expected_edf=False``) or not. The latter is much more efficient for sparse models at the cost of access to the covariance matrix and the ability to compute an upper bound on the smoothness uncertainty corrected edf. Only makes sense when ``grid_type!='JJJ1'``. Defaults to False
    :type only_expected_edf: bool,optional
    :param Vp_fidiff: Whether to rely on a finite difference approximation to compute :math:`\mathbf{V}^{\\boldsymbol{\\rho}}` or on a PQL approximation. The latter is exact for Gaussian and canonical GAMs and far cheaper if many penalties are to be estimated. Defaults to False (PQL approximation)
    :type Vp_fidiff: bool,optional
    :param use_importance_weights: Whether to rely importance weights to compute the numerical integration when ``grid_type != 'JJJ1'`` or on the log-densities of :math:`\mathbf{V}^{\\boldsymbol{\\rho}}` - the latter assumes that the unconditional posterior is normal. Defaults to True (Importance weights are used)
    :type use_importance_weights: bool,optional
    :param prior: An (optional) instance of an arbitrary class that has a ``.logpdf()`` method to compute the prior log density of a sampled candidate. If this is set to ``None``, the prior is assumed to coincide with the proposal distribution, simplifying the importance weight computation. Ignored when ``use_importance_weights=False``. Defaults to None
    :type prior: any, optional
    :param recompute_H: Whether or not to re-compute the Hessian of the log-likelihood at an estimate of the mean of the Bayesian posterior :math:`\\boldsymbol{\\beta}|y` before computing the (uncertainty/bias corrected) edf. Defaults to False
    :type recompute_H: bool, optional
    :param seed: Seed to use for random parts of the correction. Defaults to None
    :type seed: int,optional
    :param bfgs_options: Any additional keyword arguments that should be passed on to the call of :func:`scipy.optimize.minimize`. If none are provided, the ``gtol`` argument will be initialized to 1e-3. Note also, that in any case the ``maxiter`` argument is automatically set to 100. Defaults to None.
    :type bfgs_options: key=value,optional
    :return: A tuple containing: V - an estimate of the unconditional covariance matrix, LV - the Cholesky of the former, Vp - an estimate of the covariance matrix for :math:`\\boldsymbol{\\rho}`, Vpr - a regularized version of the former, edf - smoothness uncertainty corrected coefficient-wise edf, total_edf - smoothness uncertainty corrected total (i.e., model) edf, edf2 - smoothness uncertainty + smoothness bias corrected coefficient-wise edf, total_edf2 - smoothness uncertainty + smoothness bias corrected total (i.e., model) edf, expected_edf - an optional estimate of total_edf that does not require forming V, mean_coef - an optional estimate of the mean of the posterior of the coefficients
    :rtype: (scipy.sparse.csc_array,scipy.sparse.csc_array,numpyp.array,numpy.array,numpy.array,float,numpy.array,float,float,numpy.array) 
    """
    np_gen = np.random.default_rng(seed)

    family = model.family

    if not grid_type in ["JJJ1","JJJ2","JJJ3"]:
        raise ValueError("'grid_type' has to be set to one of 'JJJ1', 'JJJ2', or 'JJJ3'.")

    if isinstance(family,GENSMOOTHFamily):
        if not bfgs_options:
            bfgs_options = {"ftol":1e-9,
                            "maxcor":30,
                            "maxls":100}

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
        X = Xs[0]
        orig_scale = 1
        init_coef = copy.deepcopy(model.overall_coef)

    Vp = None
    Vpr = None
    if grid_type == "JJJ1" or grid_type == "JJJ2" or grid_type == "JJJ3":
        # Approximate Vp via finitie differencing
        if Vp_fidiff:
            Vp, Vpr, Vr, Vrr, _ = estimateVp(model,n_c=n_c,grid_type="JJJ1",Vp_fidiff=True)
        else:
            # Take PQL approximation instead
            Vp, Vpr, Vr, Vrr, _, dBetadRhos = compute_Vp_WPS(model.lvi if isinstance(family,Family) else model.overall_lvi,
                                                 model.hessian,
                                                 S_emb,
                                                 model.formula.penalties if isinstance(family,Family) else model.overall_penalties,
                                                 model.coef.reshape(-1,1) if isinstance(family,Family) else model.overall_coef.reshape(-1,1),
                                                 scale=orig_scale if isinstance(family,Family) else 1)

        # Compute approximate WPS (2016) correction
        if grid_type == "JJJ1" or (grid_type == "JJJ3" and only_expected_edf == False):
            if isinstance(family,Family):
                Vc,Vcc = compute_Vb_corr_WPS(model.lvi,Vpr,Vr,model.hessian,S_emb,model.formula.penalties,model.coef.reshape(-1,1),scale=orig_scale)
            else:
                Vc,Vcc = compute_Vb_corr_WPS(model.overall_lvi,Vpr,Vr,model.hessian,S_emb,model.overall_penalties,model.overall_coef.reshape(-1,1))
                
            if isinstance(family,Family):
                V = Vc + Vcc + ((model.lvi.T@model.lvi)*orig_scale)
            else:
                V = Vc + Vcc + model.overall_lvi.T@model.overall_lvi
            
            if grid_type == "JJJ3":
                # Can enforce lower bound of JJJ1 here
                Vlb = copy.deepcopy(V)

    rGrid = np.array([])
    remls = []
    Vs = []
    coefs = []
    edfs = []
    llks = []
    Linvs = []
    scales = []

    if grid_type != "JJJ1":
        # Generate \lambda values from Vpr for which to compute REML, and Vb
        if isinstance(family,Family):
            ep = np.log(np.array([min(b,max(a,pen.lam)) for pen in model.formula.penalties]).reshape(-1,1))
        else:
            ep = np.log(np.array([min(b,max(a,pen.lam)) for pen in model.overall_penalties]).reshape(-1,1))
        #print(ep)

        n_est = nR
        
        if X.shape[0] < 1e5 and X.shape[1] < 2000 and n_c > 1: # Parallelize grid search
            # Generate next \lambda values for which to compute REML, and Vb
            p_sample = scp.stats.multivariate_t.rvs(loc=np.ndarray.flatten(ep),shape=Vpr,df=df,size=n_est,random_state=seed)
            p_sample = np.exp(p_sample)

            if len(np.ndarray.flatten(ep)) == 1: # Single lambda parameter in model
            
                if n_est == 1: # and single sample (n_est==1) - so p_sample needs to be shape (1,1)
                    p_sample = np.array([p_sample])

                p_sample = p_sample.reshape(n_est,1) # p_sample needs to be shape (n_est,1)
                
            elif n_est == 1: # multiple lambdas - so p_sample needs to be shape (1,n_lambda)
                p_sample = np.array([p_sample]).reshape(1,-1)

            minDiag = 0.1*min(np.sqrt(Vpr.diagonal()))
            # Make sure actual estimate is included once.
            if np.any(np.max(np.abs(p_sample - np.array([pen2.lam for pen2 in rPen])),axis=1) < minDiag) == False:
                p_sample = np.concatenate((np.array([pen2.lam for pen2 in rPen]).reshape(1,-1),p_sample),axis=0)
            
            if isinstance(family,Gaussian) and isinstance(family.link,Identity) and method == "Chol": # Fast Strictly additive case
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

                    # Now compute reml for new candidates in parallel
                    args = zip(repeat(family),repeat(y_mem.name),repeat(dat_mem.name),
                            repeat(ptr_mem.name),repeat(idx_mem.name),repeat(dat_memXX.name),
                            repeat(ptr_memXX.name),repeat(idx_memXX.name),repeat(shape_y),
                            repeat(shape_dat),repeat(shape_ptr),repeat(shape_datXX),repeat(shape_ptrXX),
                            repeat(rows),repeat(cols),repeat(rPen),repeat(model.offset),p_sample)
                    
                    sample_Linvs, sample_coefs, sample_remls, sample_scales, sample_edfs, sample_llks = zip(*pool.starmap(_compute_VB_corr_terms_MP,args))
                    
                    if only_expected_edf == False:
                        Linvs.extend(list(sample_Linvs))
                    scales.extend(list(sample_scales))
                    coefs.extend(list(sample_coefs))
                    remls.extend(list(sample_remls))
                    edfs.extend(list(sample_edfs))
                    llks.extend(list(sample_llks))
                    rGrid = p_sample
            else: # all other models

                rPens = []
                for ps in p_sample:
                    rcPen = copy.deepcopy(rPen)
                    for ridx,rc in enumerate(ps):
                        rcPen[ridx].lam = rc
                    rPens.append(rcPen)

                if isinstance(family,Family):
                    origNH = None
                    if only_expected_edf:
                        if isinstance(family,Gaussian) and isinstance(family.link,Identity):
                            origNH = orig_scale
                        else:
                            origNH = -1*model.hessian
                    args = zip(repeat(family),repeat(y),repeat(X),rPens,
                               repeat(1),repeat(model.offset),repeat(None),
                               repeat(method),repeat(True),repeat(origNH))
                    with mp.Pool(processes=n_c) as pool:
                        sample_remls, sample_Linvs, _, _,sample_coefs, sample_scales, sample_edfs, sample_llks = zip(*pool.starmap(compute_reml_candidate_GAMM,args))
                else:
                    origNH = None
                    if only_expected_edf:
                        origNH = -1*model.hessian
                    args = zip(repeat(family),repeat(y),repeat(Xs),rPens,
                               repeat(init_coef),repeat(len(init_coef)),
                               repeat(model.coef_split_idx),repeat(method),
                               repeat(1e-7),repeat(1),repeat(bfgs_options),
                               repeat(origNH))
                    with mp.Pool(processes=n_c) as pool:
                        sample_remls, _, sample_Linvs, sample_coefs, sample_edfs, sample_llks = zip(*pool.starmap(compute_REML_candidate_GSMM,args))
                    sample_scales = np.ones(p_sample.shape[0])

                if only_expected_edf == False:
                    Linvs.extend(list(sample_Linvs))
                scales.extend(list(sample_scales))
                coefs.extend(list(sample_coefs))
                remls.extend(list(sample_remls))
                edfs.extend(list(sample_edfs))
                llks.extend(list(sample_llks))
                rGrid = p_sample
                rPens = None
            
        else:
            # Generate \lambda values for which to compute REML, and Vb
            p_sample = scp.stats.multivariate_t.rvs(loc=np.ndarray.flatten(ep),shape=Vpr,df=df,size=n_est,random_state=seed)
            p_sample = np.exp(p_sample)

            if len(np.ndarray.flatten(ep)) == 1: # Single lambda parameter in model
                
                if n_est == 1: # and single sample (n_est==1) - so p_sample needs to be shape (1,1)
                    p_sample = np.array([p_sample])

                p_sample = p_sample.reshape(n_est,1) # p_sample needs to be shape (n_est,1)
                
            elif n_est == 1: # multiple lambdas - so p_sample needs to be shape (1,n_lambda)
                p_sample = np.array([p_sample]).reshape(1,-1)
            
            # Make sure actual estimate is included once.
            minDiag = 0.1*min(np.sqrt(Vpr.diagonal()))
            if np.any(np.max(np.abs(p_sample - np.array([pen2.lam for pen2 in rPen])),axis=1) < minDiag) == False:
                p_sample = np.concatenate((np.array([pen2.lam for pen2 in rPen]).reshape(1,-1),p_sample),axis=0)

            for ps in p_sample:
                for ridx,rc in enumerate(ps):
                    rPen[ridx].lam = rc
                
                if isinstance(family,Family):
                    reml,Linv,LP,Pr,coef,scale,edf,llk = compute_reml_candidate_GAMM(family,y,X,rPen,n_c,model.offset,method=method,compute_inv=True)
                    #coef = coef.reshape(-1,1)

                    # Form VB, first solve LP{^-1}
                    #LPinv = compute_Linv(LP,n_c)
                    #Linv = apply_eigen_perm(Pr,LPinv)

                    # Collect conditional posterior covariance matrix for this set of coef
                    Vb = Linv.T@Linv*scale
                    #Vb += coef@coef.T
                else:
                    try:
                        reml,V,_,coef,edf,llk = compute_REML_candidate_GSMM(family,y,Xs,rPen,init_coef,len(init_coef),model.coef_split_idx,n_c=n_c,method=method,bfgs_options=bfgs_options)
                    except:
                        warnings.warn(f"Unable to compute REML score for sample {np.exp(ps)}. Skipping.")
                        continue

                    #coef = coef.reshape(-1,1)

                    # Collect conditional posterior covariance matrix for this set of coef
                    Vb = V #+ coef@coef.T

                # Collect all necessary objects for G&S correction.
                coefs.append(coef)
                remls.append(reml)
                edfs.append(edf)
                llks.append(llk)
                if only_expected_edf == False:
                    Vs.append(Vb)
                
                if len(rGrid) == 0:
                    rGrid = ps.reshape(1,-1)
                else:
                    rGrid = np.concatenate((rGrid,ps.reshape(1,-1)),axis=0)
    
    ###################################################### Prepare computation of tau ###################################################### 

    if grid_type != "JJJ1":
        # Compute weights proposed by Greven & Scheipl (2017) - still work under importance sampling case instead of grid case if we assume Vp
        # is prior for \rho|\mathbf{y}.
        if use_importance_weights and prior is None:
            ws = scp.special.softmax(remls)

        elif use_importance_weights and prior is not None: # Standard importance weights (e.g., Branchini & Elvira, 2024)
            logp = prior.logpdf(np.log(rGrid))
            q = scp.stats.multivariate_t(loc=np.ndarray.flatten(ep),shape=Vpr,df=df,allow_singular=True)
            logq = q.logpdf(np.log(rGrid))

            ws = scp.special.softmax(remls + logp - logq)

        else:
            # Normal weights result in cancellation, i.e., just average
            ws = scp.special.softmax(np.ones_like(remls))

        Vcr = Vr @ dBetadRhos.T

        # Optionally re-compute negative Hessian at posterior mean for coef.
        if recompute_H and (only_expected_edf == False):

            # Estimate mean of posterior beta|y
            mean_coef = ws[0]*coefs[0]
            for ri in range(1,len(rGrid)):
                mean_coef += ws[ri]*coefs[ri]
            
            # Recompute Hessian at mean
            if isinstance(family,Family):
                yb = y
                Xb = X

                S_emb,_,S_root,_ = compute_S_emb_pinv_det(X.shape[1],model.formula.penalties,"svd",method != 'Chol')

                if isinstance(family,Gaussian) and isinstance(family.link,Identity): # strictly additive case
                    nH = (-1*model.hessian)*orig_scale

                else: # Generalized case
                    eta = (X @ mean_coef).reshape(-1,1) + model.offset
                    mu = family.link.fi(eta)
                    
                    # Compute pseudo-dat and weights for mean coef
                    yb,Xb,z,Wr = update_PIRLS(y,yb,mu,eta-model.offset,X,Xb,family)

                    inval_check =  np.any(np.isnan(z))

                    if inval_check:
                        _, w, inval = PIRLS_pdat_weights(y,mu,eta-model.offset,family)
                        w[inval] = 0

                        # Re-compute weight matrix
                        Wr_fix = scp.sparse.spdiags([np.sqrt(np.ndarray.flatten(w))],[0])
                    else:
                        Wr_fix = Wr
                        
                    W = Wr_fix@Wr_fix
                    nH = (X.T@W@X).tocsc() 

                # Solve for coef to get Cholesky needed to re-compute scale
                _,_,_,Pr,_,LP,keep,drop = update_coef(yb,X,Xb,family,S_emb,S_root,n_c,None,model.offset)

                # Re-compute scale
                _,_,_,_,_,scale = update_scale_edf(y,z,eta,Wr,X.shape[0],X.shape[1],LP,None,Pr,None,family,model.formula.penalties,keep,drop,n_c)
                
                # And negative hessian
                nH /= scale

                if drop is not None:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        nH[:,drop] = 0
                        nH[drop,:] = 0

            else: # GSMM/GAMLSS case
                if isinstance(Family,GAMLSSFamily): #GAMLSS case
                    split_coef = np.split(mean_coef,model.coef_split_idx)

                    # Update etas and mus
                    etas = [Xs[i]@split_coef[i] for i in range(family.n_par)]
                    mus = [family.links[i].fi(etas[i]) for i in range(family.n_par)]

                    # Get derivatives with respect to eta
                    if family.d_eta == False:
                        d1eta,d2eta,d2meta = deriv_transform_mu_eta(y,mus,family)
                    else:
                        d1eta = [fd1(y,*mus) for fd1 in family.d1]
                        d2eta = [fd2(y,*mus) for fd2 in family.d2]
                        d2meta = [fd2m(y,*mus) for fd2m in family.d2m]

                    # Get derivatives with respect to coef
                    _,H = deriv_transform_eta_beta(d1eta,d2eta,d2meta,Xs,only_grad=False)

                else: # GSMM
                    H = family.hessian(mean_coef,model.coef_split_idx,y,Xs)

                nH = -1 * H

            if verbose:
                print(f"Recomputed negative Hessian. 2 Norm of coef. difference: {np.linalg.norm(mean_coef-model.coef.reshape(-1,1))}. F. Norm of n. Hessian difference: {scp.sparse.linalg.norm(nH + model.hessian)}")

        else:
            mean_coef = None
            nH = -1*model.hessian

        ###################################################### Compute tau ###################################################### 

        if only_expected_edf:
            # Compute correction of edf directly..
            expected_edf = max(model.edf,np.sum(ws*edfs))

            if grid_type == 'JJJ2':
                # Can now add remaining correction term
                
                # Now have Vc = Vcr.T @ Vcr, so:
                # tr(Vc@(-1*model.hessian)) = tr(Vcr.T @ Vcr@(-1*model.hessian)) = tr(Vcr@ (-1*model.hessian)@Vcr.T)
                #expected_edf += (Vc@(-1*model.hessian)).trace()
                expected_edf += (Vcr@ (nH)@Vcr.T).trace()

            elif grid_type == 'JJJ3':
                # Correct based on G&S expectations instead
                tr1 = ws[0]* (coefs[0].T@(nH)@coefs[0])[0,0]

                # E_{p|y}[\boldsymbol{\beta}] in Greven & Scheipl (2017)
                tr2 = ws[0]*coefs[0] 
                
                # Now sum over remaining r
                for ri in range(1,len(rGrid)):
                    tr1 += ws[ri]* (coefs[ri].T@(nH)@coefs[ri])[0,0]
                    tr2 += ws[ri]*coefs[ri]
                
                # Enforce lower bound of JJJ2
                if (Vcr@ (nH)@Vcr.T).trace() > (tr1 - (tr2.T@(nH)@tr2)[0,0]):
                    expected_edf += (Vcr@ (nH)@Vcr.T).trace()
                else:
                    expected_edf += tr1 - (tr2.T@(nH)@tr2)[0,0]
                    
            if verbose:
                print(f"Correction was based on {rGrid.shape[0]} samples in total.")

            return None,None,None,None,None,None,None,None,expected_edf,mean_coef
        else:
            expected_edf = None
        
        ###################################################### Compute tau and full covariance matrix ###################################################### 

        if grid_type == 'JJJ2':
            
            if X.shape[0] < 1e5 and X.shape[1] < 2000 and n_c > 1:
                # E_{p|y}[V_\boldsymbol{\beta}(\lambda)] + E_{p|y}[\boldsymbol{\beta}\boldsymbol{\beta}^T] in Greven & Scheipl (2017)
                Vr1 = ws[0]* (Linvs[0].T@Linvs[0]*scales[0])

                # Now sum over remaining r
                for ri in range(1,len(rGrid)):
                    Vr1 += ws[ri]*((Linvs[ri].T@Linvs[ri]*scales[ri]))

            else:
                Vr1 = ws[0]*Vs[0]

                for ri in range(1,len(rGrid)):
                    Vr1 += ws[ri]*Vs[ri]
            
            if (Vr1@(nH)).trace() < model.edf:
                nH = -1 * model.hessian # Reset nH
                
                if isinstance(family,Family):
                    Vr1 = model.lvi.T@model.lvi*orig_scale
                else:
                    Vr1 = model.overall_lvi.T@model.overall_lvi

            V = Vr1 + Vcr.T @ Vcr

        elif grid_type == 'JJJ3':

            # Now compute \hat{cov(\boldsymbol{\beta}|y)}
            if X.shape[0] < 1e5 and X.shape[1] < 2000 and n_c > 1:
                # E_{p|y}[V_\boldsymbol{\beta}(\lambda)] + E_{p|y}[\boldsymbol{\beta}\boldsymbol{\beta}^T] in Greven & Scheipl (2017)
                Vr1 = ws[0]* (Linvs[0].T@Linvs[0]*scales[0]) 
                Vr2 = ws[0]* (coefs[0]@coefs[0].T)
                # E_{p|y}[\boldsymbol{\beta}] in Greven & Scheipl (2017)
                Vr3 = ws[0]*coefs[0] 
                
                # Now sum over remaining r
                for ri in range(1,len(rGrid)):
                    Vr1 += ws[ri]*(Linvs[ri].T@Linvs[ri]*scales[ri])
                    Vr2 += ws[ri]* (coefs[ri]@coefs[ri].T)
                    Vr3 += ws[ri]*coefs[ri]

            else:
                Vr1 = ws[0]*Vs[0]
                Vr2 = ws[0]*(coefs[0]@coefs[0].T)
                Vr3 = ws[0]*coefs[0] 

                for ri in range(1,len(rGrid)):
                    Vr1 += ws[ri]*Vs[ri]
                    Vr2 += ws[ri]*(coefs[ri]@coefs[ri].T)
                    Vr3 += ws[ri]*coefs[ri]
            
            #if (Vr1@(nH)).trace() < model.edf:
            #    nH = -1 * model.hessian # Reset nH
            #
            #    if isinstance(family,Family):
            #        Vr1 = model.lvi.T@model.lvi*orig_scale
            #    else:
            #        Vr1 = model.overall_lvi.T@model.overall_lvi

            # Now, Greven & Scheipl provide final estimate =
            # E_{p|y}[V_\boldsymbol{\beta}(\lambda)] + E_{p|y}[\boldsymbol{\beta}\boldsymbol{\beta}^T] - E_{p|y}[\boldsymbol{\beta}] E_{p|y}[\boldsymbol{\beta}]^T
            # Enforce lower bound of JJJ2 again
            #if (Vcr @ (nH) @ Vcr.T).trace() > ((Vr2 - (Vr3@Vr3.T)) @ (nH)).trace():
            #    V = Vr1 + Vcr.T @ Vcr
            #else:
            V = Vr1 + Vr2 - (Vr3@Vr3.T)
            
            # Enforce lower bound of JJJ1
            if (Vlb@(-1 * model.hessian)).trace() > (V@nH).trace():
                nH = -1 * model.hessian
                V = Vlb
            
    else:
        mean_coef = None
        expected_edf = None
        nH = -1 * model.hessian

    # Check V is full rank - can use LV for sampling as well..
    LV,code = cpp_chol(scp.sparse.csc_array(V))
    if code != 0:
        raise ValueError("Failed to estimate marginal covariance matrix for ecoefficients.")

    # Compute corrected edf (e.g., for AIC; Wood, Pya, & Saefken, 2016)
    total_edf = None
    if form_t or form_t1:
        F = V@(nH)

        edf = F.diagonal()
        total_edf = F.trace()

    # In mgcv, an upper limit is enforced on edf and total_edf when they are uncertainty corrected - based on t1 in section 6.1.2 of Wood (2017)
    # so the same is done here.
    total_edf2 = None
    if grid_type == "JJJ1" or grid_type == "JJJ2" or grid_type == "JJJ3":
        if isinstance(family,Family):
            if isinstance(family,Gaussian) and isinstance(family.link,Identity): # Strictly additive case
                ucF = (model.lvi.T@model.lvi)@((X.T@X))
            else: # Generalized case
                W = model.Wr@model.Wr
                ucF = (model.lvi.T@model.lvi)@((X.T@W@X))
        else: # GSMM/GAMLSS case
            ucF = (model.overall_lvi.T@model.overall_lvi)@(-1*model.hessian)

        total_edf2 = 2*model.edf - (ucF@ucF).trace()
        if total_edf > total_edf2:
            #print(edf)
            total_edf = total_edf2
            #edf = None

    # Compute uncertainty corrected smoothness bias corrected edf (t1 in section 6.1.2 of Wood, 2017)
    edf2 = None
    if form_t1:
        edf2 = 2*edf - (F@F).diagonal()
        if total_edf2 is None: # Otherwise respect upper bound.
            total_edf2 = 2*total_edf - (F@F).trace()

    if verbose and grid_type != "JJJ1":
        print(f"Correction was based on {rGrid.shape[0]} samples in total.")    

    return V,LV,Vp,Vpr,edf,total_edf,edf2,total_edf2,expected_edf,mean_coef