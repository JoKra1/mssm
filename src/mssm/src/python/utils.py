import numpy as np
import scipy as scp
import math
import warnings
from itertools import permutations,product,repeat
import copy
from ..python.gamm_solvers import cpp_backsolve_tr,compute_S_emb_pinv_det,cpp_chol,cpp_solve_coef,update_scale_edf,compute_Linv,apply_eigen_perm,tqdm,managers,shared_memory,cpp_solve_coefXX,update_PIRLS,PIRLS_pdat_weights,correct_coef_step,update_coef_gen_smooth,cpp_cholP,update_coef_gammlss,compute_lgdetD_bsb,calculate_edf,compute_eigen_perm,computeHSR1,computeH,update_coef,deriv_transform_mu_eta,deriv_transform_eta_beta,translate_sparse,map_csc_to_eigen
from ..python.terms import fs, rs
from .file_loading import mp
from .repara import reparam
from ..python.exp_fam import Family,Gaussian, Identity,GAMLSSFamily,GSMMFamily,Link
from ..python.formula import Formula,LambdaTerm
import davies
import dChol
from collections.abc import Callable

def computeAr1Chol(formula:Formula,rho:float) -> tuple[scp.sparse.csc_array,float]:
    """Computes the inverse of the cholesky of the (scaled) variance matrix of an ar1 model.

    :param formula: Formula of the model
    :type formula: Formula
    :param rho: ar1 weight.
    :type rho: float
    :return: Tuple, containing banded inverse Cholesky as a scipy array and the correction needed to get the likelihood of the ar1 model.
    :rtype: tuple[scp.sparse.csc_array,float]
    """

    y_flat = formula.y_flat[formula.NOT_NA_flat]
    n_y = len(y_flat)
    d0 = np.tile(1/np.sqrt(1-np.power(rho,2)),n_y)
    d1 = np.tile(-rho/np.sqrt(1-np.power(rho,2)),n_y-1)

    # Correct ar1 process for individual (time) series
    if formula.series_id is not None:
        start = np.tile(False,len(formula.y_flat))
        start[formula.sid] = True
        start = start[formula.NOT_NA_flat]
        sid0 = np.where(start)[0]
        sid1 = sid0[1:] - 1
        
        d0[sid0] = 1
        d1[sid1] = 0

    Lrhoi = scp.sparse.diags_array([d0,d1],format='csr',offsets=[0,1])

    # Likelihood correction computed as done by mgcv. see: https://github.com/cran/mgcv/blob/fb7e8e718377513e78ba6c6bf7e60757fc6a32a9/R/bam.r#L2761
    llc = (n_y - (np.sum(start) if formula.series_id is not None else 1))*np.log(1/np.sqrt(1-np.power(rho,2)))

    return Lrhoi,llc

class GAMLSSGSMMFamily(GSMMFamily):
    """Implementation of the ``GSMMFamily`` class that uses only information about the likelihood to estimate any implemented GAMMLSS model.
    
    Allows to estimate any GAMMLSS as a GSMM via the L-qEFS & Newton update. Example::

        # Simulate 500 data points
        sim_dat = sim3(500,2,c=1,seed=0,family=Gaussian(),binom_offset = 0, correlate=False)

        # We need to model the mean: mu_i
        formula_m = Formula(lhs("y"),
                            [i(),f(["x0"]),f(["x1"]),f(["x2"]),f(["x3"])],
                            data=sim_dat)

        # And for sd - here constant
        formula_sd = Formula(lhs("y"),
                            [i()],
                            data=sim_dat)

        # Collect both formulas
        formulas = [formula_m,formula_sd]
        links = [Identity(),LOG()]

        # Now define the general family + model
        gsmm_fam = GAMLSSGSMMFamily(2,GAUMLSS(links))
        model = GSMM(formulas=formulas,family=gsmm_fam)

        # Fit with SR1
        bfgs_opt={"gtol":1e-9,
                "ftol":1e-9,
                "maxcor":30,
                "maxls":200,
                "maxfun":1e7}
                        
        model.fit(init_coef=None,method='qEFS',extend_lambda=False,
                control_lambda=0,max_outer=200,max_inner=500,min_inner=500,
                seed=0,qEFSH='SR1',max_restarts=5,overwrite_coef=False,qEFS_init_converge=False,prefit_grad=True,
                progress_bar=True,**bfgs_opt)
        
        ################### Or for a multinomial model: ###################
        
        formulas = [Formula(lhs("y"),
                        [i(),f(["x0"])],
                        data=sim5(1000,seed=91)) for k in range(4)]

        # Create family - again specifying K-1 pars - here 4!
        family = MULNOMLSS(4)

        # Collect both formulas
        links = family.links

        # Now again define the general family + model
        gsmm_fam = GAMLSSGSMMFamily(4,family)
        model = GSMM(formulas=formulas,family=gsmm_fam)

        # And fit with SR1
        bfgs_opt={"gtol":1e-9,
                "ftol":1e-9,
                "maxcor":30,
                "maxls":200,
                "maxfun":1e7}
                        
        model.fit(init_coef=None,method='qEFS',extend_lambda=False,
                control_lambda=0,max_outer=200,max_inner=500,min_inner=500,
                seed=0,qEFSH='SR1',max_restarts=0,overwrite_coef=False,qEFS_init_converge=False,prefit_grad=True,
                progress_bar=True,**bfgs_opt)

    References:
        - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
        - Nocedal & Wright (2006). Numerical Optimization. Springer New York.

    :param pars: Number of parameters of the likelihood.
    :type pars: int
    :param gammlss_family: Any implemented member of the :class:`GAMLSSFamily` class. Available in ``self.llkargs[0]``.
    :type gammlss_family: GAMLSSFamily
    """

    def __init__(self, pars: int, gammlss_family:GAMLSSFamily) -> None:
        super().__init__(pars, gammlss_family.links, gammlss_family)
    
    def llk(self, coef:np.ndarray,coef_split_idx:list[int],ys:list[np.ndarray],Xs:list[scp.sparse.csc_array]) -> float:
        """
        Function to evaluate log-likelihood of GAMM(LSS) model when estimated via GSMM.

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the sub-sets associated with each paramter of the llk.
        :type coef_split_idx: [int]
        :param ys: List containing the vectors of observations passed as ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the actual observed data is passed along via the first formula (so it is stored in ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula, then ``ys`` contains ``None`` at their indices to save memory.
        :type ys: [np.ndarray or None]
        :param Xs: A list of sparse model matrices per likelihood parameter.
        :type Xs: [scp.sparse.csc_array]
        :return: The log-likelihood evaluated at ``coef``.
        :rtype: float
        """
        y = ys[0]
        gammlss_family = self.llkargs[0]
        split_coef = np.split(coef,coef_split_idx)
        etas = [Xs[ei]@split_coef[ei] for ei in range(len(Xs))]
        mus = [self.links[ei].fi(etas[ei]) for ei in range(len(Xs))]
        
        llk = gammlss_family.llk(y,*mus)

        if np.isnan(llk):
            return -np.inf
        
        return llk
    
    def gradient(self, coef:np.ndarray,coef_split_idx:list[int],ys:list[np.ndarray],Xs:list[scp.sparse.csc_array]) -> np.ndarray:
        """
        Function to evaluate gradient of GAMM(LSS) model when estimated via GSMM.

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the sub-sets associated with each paramter of the llk.
        :type coef_split_idx: [int]
        :param ys: List containing the vectors of observations passed as ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the actual observed data is passed along via the first formula (so it is stored in ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula, then ``ys`` contains ``None`` at their indices to save memory.
        :type ys: [np.ndarray or None]
        :param Xs: A list of sparse model matrices per likelihood parameter.
        :type Xs: [scp.sparse.csc_array]
        :return: The Gradient of the log-likelihood evaluated at ``coef`` as numpy array) of shape (-1,1).
        :rtype: np.ndarray
        """
        y = ys[0]
        split_coef = np.split(coef,coef_split_idx)
        etas = [Xs[ei]@split_coef[ei] for ei in range(len(Xs))]
        mus = [self.links[ei].fi(etas[ei]) for ei in range(len(Xs))]
        
        # Get the Gamlss family
        gammlss_family = self.llkargs[0]
        
        if gammlss_family.d_eta == False:
         
            d1eta,d2eta,d2meta = deriv_transform_mu_eta(y,mus,gammlss_family)
        else:
            d1eta = [fd1(y,*mus) for fd1 in gammlss_family.d1]
            d2eta = [fd2(y,*mus) for fd2 in gammlss_family.d2]
            d2meta = [fd2m(y,*mus) for fd2m in gammlss_family.d2m]
            
        # Get gradient
        grad,_ = deriv_transform_eta_beta(d1eta,d2eta,d2meta,Xs,only_grad=True)

        return grad.reshape(-1,1)
    
    def hessian(self, coef:np.ndarray,coef_split_idx:list[int],ys:list[np.ndarray],Xs:list[scp.sparse.csc_array]) -> scp.sparse.csc_array:
        """
        Function to evaluate Hessian of GAMM(LSS) model when estimated via GSMM.

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the sub-sets associated with each paramter of the llk.
        :type coef_split_idx: [int]
        :param ys: List containing the vectors of observations passed as ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the actual observed data is passed along via the first formula (so it is stored in ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula, then ``ys`` contains ``None`` at their indices to save memory.
        :type ys: [np.ndarray or None]
        :param Xs: A list of sparse model matrices per likelihood parameter.
        :type Xs: [scp.sparse.csc_array]
        :return: The Hessian of the log-likelihood evaluated at ``coef``.
        :rtype: scp.sparse.csc_array
        """
        y = ys[0]
        split_coef = np.split(coef,coef_split_idx)
        etas = [Xs[ei]@split_coef[ei] for ei in range(len(Xs))]
        mus = [self.links[ei].fi(etas[ei]) for ei in range(len(Xs))]
        
        # Get the Gamlss family
        gammlss_family = self.llkargs[0]
        
        if gammlss_family.d_eta == False:
         
            d1eta,d2eta,d2meta = deriv_transform_mu_eta(y,mus,gammlss_family)
        else:
            d1eta = [fd1(y,*mus) for fd1 in gammlss_family.d1]
            d2eta = [fd2(y,*mus) for fd2 in gammlss_family.d2]
            d2meta = [fd2m(y,*mus) for fd2m in gammlss_family.d2m]
            
        # Get Hessian
        _,H = deriv_transform_eta_beta(d1eta,d2eta,d2meta,Xs,only_grad=False)

        return H


def sample_MVN(n:int,mu:int|np.ndarray,scale:float,P:scp.sparse.csc_array|None,L:scp.sparse.csc_array|None,LI:scp.sparse.csc_array|None=None,use:list[int]|None=None,seed:int|None=None) -> np.ndarray:
    """Draw ``n`` samples from multivariate normal with mean :math:`\\boldsymbol{\\mu}` (``mu``) and covariance matrix :math:`\\boldsymbol{\\Sigma}`.
    
    :math:`\\boldsymbol{\\Sigma}` does not need to be provided. Rather the function expects either ``L`` (:math:`\\mathbf{L}` in what follows) or ``LI`` (:math:`\\mathbf{L}^{-1}` in what follows) and ``scale`` (:math:`\\phi` in what follows).
    These relate to :math:`\\boldsymbol{\\Sigma}` so that :math:`\\boldsymbol{\\Sigma}/\\phi = \\mathbf{L}^{-T}\\mathbf{L}^{-1}` or :math:`\\mathbf{L}\\mathbf{L}^T = [\\boldsymbol{\\Sigma}/\\phi]^{-1}`
    so that :math:`\\mathbf{L}*(1/\\phi)^{0.5}` is the Cholesky of the precision matrix of :math:`\\boldsymbol{\\Sigma}`.

    Notably, for models available in ``mssm`` ``L`` (and ``LI``) have usually be computed for a permuted matrix, e.g., :math:`\\mathbf{P}[\\mathbf{X}^T\\mathbf{X} + \\mathbf{S}_{\\lambda}]\\mathbf{P}^T` (see Wood \\& Fasiolo, 2017).
    Hence for sampling we often need to correct for permutation matrix :math:`\\mathbf{P}` (``P``). if ``LI`` is provided, then ``P`` can be omitted and is assumed to have been used to un-pivot ``LI`` already.

    Used for example sample the uncorrected posterior :math:`\\boldsymbol{\\beta} | \\mathbf{y}, \\boldsymbol{\\lambda} \\sim N(\\boldsymbol{\\mu} = \\hat{\\boldsymbol{\\beta}},[\\mathbf{X}^T\\mathbf{X} + \\mathbf{S}_{\\lambda}]^{-1}\\phi)` for a GAMM (see Wood, 2017).
    Based on section 7.4 in Gentle (2009), assuming :math:`\\boldsymbol{\\Sigma}` is :math:`p*p` and covariance matrix of uncorrected posterior, samples :math:`\\boldsymbol{\\beta}` are then obtained by computing:

    .. math::

        \\boldsymbol{\\beta} = \\hat{\\boldsymbol{\\beta}} + [\\mathbf{P}^T \\mathbf{L}^{-T}*\\phi^{0.5}]\\mathbf{z}\\ \\text{where}\\ z_i \\sim N(0,1)\\ \\forall i = 1,...,p

    Alternatively, relying on the fact of equivalence that:

    .. math::

        [\\mathbf{L}^T*(1/\\phi)^{0.5}]\\mathbf{P}[\\boldsymbol{\\beta} - \\hat{\\boldsymbol{\\beta}}] = \\mathbf{z}
    
    we can first solve for :math:`\\mathbf{y}` in:

    .. math::

        [\\mathbf{L}^T*(1/\\phi)^{0.5}] \\mathbf{y} = \\mathbf{z}
    
    followed by computing:

    .. math::

        \\mathbf{y} = \\mathbf{P}[\\boldsymbol{\\beta} - \\hat{\\boldsymbol{\\beta}}]

        \\boldsymbol{\\beta} = \\hat{\\boldsymbol{\\beta}} + \\mathbf{P}^T\\mathbf{y}
    
        
    The latter avoids forming :math:`\\mathbf{L}^{-1}` (which unlike :math:`\\mathbf{L}` might not benefit from the sparsity preserving permutation :math:`\\mathbf{P}`). If ``LI is None``,
    ``L`` will thus be used for sampling as outlined in these alternative steps.

    Often we care only about a handfull of elements in ``mu`` (e.g., the first ones corresponding to "fixed effects'" in a GAMM). In that case we
    can generate samles only for this sub-set of interest by only using a sub-block of rows of :math:`\\mathbf{L}` or :math:`\\mathbf{L}^{-1}` (all columns remain). Argument ``use`` can be a ``np.array``
    containg the indices of elements in ``mu`` that should be sampled. Because this only works efficiently when ``LI`` is available an error is raised when ``not use is None and LI is None``.

    If ``mu`` is set to **any integer** (i.e., not a Numpy array/list) it is automatically treated as 0. For :class:`mssm.models.GAMMLSS` or :class:`mssm.models.GSMM` models, ``scale`` can be set to 1.
    
    References:

     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
     - Gentle, J. (2009). Computational Statistics.

    :param n: Number of samples to generate
    :type n: int
    :param mu: mean of normal distribution as described above
    :type mu: int | np.ndarray
    :param scale: scaling parameter of covariance matrix as described above
    :type scale: float
    :param P: Permutation matrix or None.
    :type P: scp.sparse.csc_array | None
    :param L: Cholesky of precision of scaled covariance matrix as described above.
    :type L: scp.sparse.csc_array | None
    :param LI: Inverse of cholesky factor of precision of scaled covariance matrix as described above.
    :type LI: scp.sparse.csc_array | None, optional
    :param use: Indices of parameters in ``mu`` for which to generate samples, defaults to None in which case all parameters will be sampled
    :type use: list[int] | None, optional
    :param seed: Seed to use for random sample generation, defaults to None
    :type seed: int | None, optional
    :return: Samples from multi-variate normal distribution. In case ``use`` is not provided, the returned array will be of shape ``(p,n)`` where ``p==LI.shape[1]``. Otherwise, the returned array will be of shape ``(len(use),n)``.
    :rtype: np.ndarray
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
    
def print_parametric_terms(model,par:int=0) -> None:
        """Prints summary output for linear/parametric terms in the model of a specific parameter, not unlike the one returned in R when using the ``summary`` function for ``mgcv`` models.
        
        If the model has not been estimated yet, it prints the term names instead.
        
        For each coefficient, the named identifier and estimated value are returned. In addition, for each coefficient a p-value is returned, testing
        the null-hypothesis that the corresponding coefficient :math:`\\beta=0`. Under the assumption that this is true, the Null distribution follows
        a t-distribution for models in which an additional scale parameter was estimated (e.g., Gaussian, Gamma) and a standardized normal distribution for
        models in which the scale parameter is known or was fixed (e.g., Binomial). For the former case, the t-statistic, Degrees of freedom of the Null
        distribution (DoF.), and the p-value are printed as well. For the latter case, only the z-statistic and the p-value are printed.
        See Wood (2017) section 6.12 and 1.3.3 for more details.

        Note that, un-penalized coefficients that are part of a smooth function are not covered by this function.

        References:
            - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

        :param model: GSMM, GAMMLSS, or GAMM model
        :type model: mssm.models.GSMM | mssm.models.GAMMLSS | mssm.models.GAMM 
        :param par: Parameter of the likelihood/family for which to print terms, defaults to 0
        :type par: int, optional
        :raises NotImplementedError: Will throw an error when called for a model for which the model matrix was never former completely.
        :rtype: None
        """
        # Wood (2017) section 6.12 defines that b_j.T@[V_{b_j}]^{-1}@b_j is the test-statistic following either
        # an F distribution or a Chi-square distribution (the latter if we have a known scale parameter).
        # Here we want to report single parameter tests for all un-penalized coefficients, so V_b_j is actually the
        # diagonal elements of the initial n_j_coef*n_j_coef sub-block of V_b. Where n_j_coef is the number of un-penalized
        # coefficients present in the model. We can form this block efficiently from L, where L.T@L = V, by taking only the
        # first n_j_coef columns of L. 1 over the diagonal of the resulting sub-block then gives the desired inverse to compute
        # the test-statistic for a single test. Note that for a single test we need to take the root of the F/Chi-square statistic
        # to go to a t/N statistic. As a consequence, sqrt(b_j.T@[V_{b_j}]^{-1}@b_j) is actually abs(b_j/sqrt(V_{b_j})) as shown in
        # section 1.3.3 of Wood (2017). So we can just compute that directly.

        if isinstance(model.family,Family): # GAMM case
            form = model.formulas[0]

            if model.coef is not None:
                coef = model.coef.flatten()

                lvi = model.lvi
                scale = model.scale
            else:
                coef = None
        else:
            # GAMMLSS or GSMM case
            form = model.formulas[par]
            
            if model.coef is not None:
                # Get coef and lvi for selected formula
                split_coef = np.split(model.coef,model.coef_split_idx)
                split_idx = np.ndarray.flatten(np.split(np.arange(len(model.coef)),model.coef_split_idx)[par])
                coef = np.ndarray.flatten(split_coef[par])
                lvi = model.lvi[:,split_idx]
                scale = 1
            else:
                coef = None

        term_names = np.array(form.get_term_names())
        linear_names = term_names[form.get_linear_term_idx()]
        
        if coef is None:
            for term in linear_names:
                print(term)
        else:
            if len(form.file_paths) != 0:
                raise NotImplementedError("Cannot return p-value for parametric terms if X.T@X was formed iteratively.")

            # Number of linear terms
            n_j_coef = sum(form.coef_per_term[form.get_linear_term_idx()])

            # Corresponding coef
            coef_j = coef[:n_j_coef]

            # and names...
            coef_j_names = form.coef_names[:n_j_coef]

            # Form initial n_j_coef*n_j_coef sub-block of V_b
            V_b_j = (lvi[:,:n_j_coef].T@lvi[:,:n_j_coef])*scale

            # Now get the inverse of the diagonal for the test-statistic (ts)
            V_b_inv_j = V_b_j.diagonal()

            # Actual ts (all positive, later we should return * sign(coef_j)):
            ts = np.abs(coef_j/np.sqrt(V_b_inv_j))

            # Compute p(abs(T/N) > abs(t/n))
            if isinstance(model.family,Family) and model.family.twopar:
                ps = 1 - scp.stats.t.cdf(ts,df = len(form.y_flat[form.NOT_NA_flat]) - form.n_coef)
            else:
                ps = 1 - scp.stats.norm.cdf(ts)

            ps *= 2 # Correct for abs

            for coef_name,coef,t,p in zip(coef_j_names,coef_j,ts,ps):
                t_str = coef_name + f": {round(coef,ndigits=3)}, "

                if isinstance(model.family,Family) and model.family.twopar:
                    t_str += f"t: {round(np.sign(coef)*t,ndigits=3)}, DoF.: {int(len(form.y_flat[form.NOT_NA_flat]) - form.n_coef)}, P(|T| > |t|): "
                else:
                    t_str += f"z: {round(np.sign(coef)*t,ndigits=3)}, P(|Z| > |z|): "

                if p < 0.001:
                    t_str += "{:.3e}".format(p,ndigits=3)
                else:
                    t_str += f"{round(p,ndigits=5)}"

                if p < 0.001:
                    t_str += " ***"
                elif p < 0.01:
                    t_str += " **"
                elif p < 0.05:
                    t_str += " *"
                elif p < 0.1:
                    t_str += " ."
                print(t_str)

            print("\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: .")
    
def print_smooth_terms(model,par:int=0,pen_cutoff:float=0.2,ps:list[float]|None=None,Trs:list[float]|None=None) -> None:
        """Prints the name of the smooth terms included in the model of a given parameter.
        
        After fitting, the estimated degrees of freedom per term are printed as well.
        Smooth terms with edf. < ``pen_cutoff`` will be highlighted. This only makes sense when extra Kernel penalties are placed on smooth terms to enable
        penalizing them to a constant zero. In that case edf. < ``pen_cutoff`` can then be taken as evidence that the smooth has all but notationally disappeared
        from the model, i.e., it does not contribute meaningfully to the model fit. This can be used as an alternative form of model selection - see Marra & Wood (2011).

        References:
            - Marra & Wood (2011). Practical variable selection for generalized additive models.

        :param model: GSMM, GAMMLSS, or GAMM model
        :type model: mssm.models.GSMM | mssm.models.GAMMLSS | mssm.models.GAMM 
        :param par: Distribution parameter for which to compute p-values. Ignored when ``model`` is a GAMM. Defaults to 0
        :type par: int, optional
        :param pen_cutoff: At which edf. cut-off smooth terms should be marked as "effectively removed", defaults to None
        :type pen_cutoff: float, optional
        :param ps: Optional list of p-values per smooth term if these should be printed, defaults to None
        :type ps: [float], optional
        :param Trs: Optional list of test statistics (based on which the ``ps`` were computed) per smooth term if these should be printed, defaults to None
        :type Trs: [float], optional
        :rtype: None
        """

        if ps is not None and Trs is None:
            raise ValueError("To display p-values, both ``ps`` and ``Trs`` needs to be provided.")
        
        if ps is not None and len(ps) != len(Trs):
            raise ValueError("``ps`` and ``Trs`` must have the same length but do not.")

        if isinstance(model.family,Family): # GAMM case
            form = model.formulas[0]

            if model.coef is None:
                term_edf = None
            else:
                term_edf = model.term_edf
        else:
            # GAMMLSS or GSMM case
            form = model.formulas[par]

            # Get term edf for par
            if model.coef is None:
                term_edf = None
            else:
                term_edf = []
                prev_start_idx = 0
                prev_edf_idx = 0
                for pen in model.overall_penalties:

                    if pen.dist_param > par:
                        break
                    
                    if pen.start_index > prev_start_idx:
                        if pen.dist_param == par:
                            term_edf.append(model.term_edf[prev_edf_idx])
                        
                        # Keep track of start idx (coef idx) and edf idx.
                        prev_start_idx = pen.start_index
                        prev_edf_idx += 1

        term_names = np.array(form.get_term_names())
        smooth_names = [*term_names[form.get_smooth_term_idx()],
                        *term_names[form.get_random_term_idx()]]
        
        if term_edf is None:
            for term in smooth_names:
                print(term)
        else:
            terms = form.terms
            coding_factors = form.get_coding_factors()
            name_idx = 0
            edf_idx = 0
            p_idx = 0
            pen_out = 0
            for sti in form.get_smooth_term_idx():
                sterm = terms[sti]
                if not sterm.by is None and sterm.id is None:
                    for li in range(len(form.get_factor_levels()[sterm.by])):
                        t_edf = round(term_edf[edf_idx],ndigits=3)
                        e_str = smooth_names[name_idx] + f": {coding_factors[sterm.by][li]}; edf: {t_edf}"
                        if t_edf < pen_cutoff:
                            # Term has effectively been removed from the model
                            e_str += " *"
                            pen_out += 1
                        if ps is not None and (isinstance(sterm,fs) == False):
                            if isinstance(model.family,Family) and model.family.twopar:
                                e_str += f" f: {round(Trs[p_idx],ndigits=3)} P(F > f) = "
                            else:
                                e_str += f" chi^2: {round(Trs[p_idx],ndigits=3)} P(Chi^2 > chi^2) = "
                            
                            if ps[p_idx] < 0.001:
                                e_str += "{:.3e}".format(ps[p_idx],ndigits=3)
                            else:
                                e_str += f"{round(ps[p_idx],ndigits=5)}"
                            
                            if ps[p_idx] < 0.001:
                                e_str += " ***"
                            elif ps[p_idx] < 0.01:
                                e_str += " **"
                            elif ps[p_idx] < 0.05:
                                e_str += " *"
                            elif ps[p_idx] < 0.1:
                                e_str += " ."

                            p_idx += 1

                        print(e_str)
                        edf_idx += 1
                else:
                    t_edf = round(term_edf[edf_idx],ndigits=3)
                    e_str = smooth_names[name_idx] + f"; edf: {t_edf}"
                    if t_edf < pen_cutoff:
                        # Term has effectively been removed from the model
                        e_str += " *"
                        pen_out += 1
                    if ps is not None and (isinstance(sterm,fs) == False):
                        if isinstance(model.family,Family) and model.family.twopar:
                            e_str += f" f: {round(Trs[p_idx],ndigits=3)} P(F > f) = "
                        else:
                            e_str += f" chi^2: {round(Trs[p_idx],ndigits=3)} P(Chi^2 > chi^2) = "
                        
                        if ps[p_idx] < 0.001:
                            e_str += "{:.3e}".format(ps[p_idx],ndigits=3)
                        else:
                            e_str += f"{round(ps[p_idx],ndigits=5)}"
                        
                        if ps[p_idx] < 0.001:
                            e_str += " ***"
                        elif ps[p_idx] < 0.01:
                            e_str += " **"
                        elif ps[p_idx] < 0.05:
                            e_str += " *"
                        elif ps[p_idx] < 0.1:
                            e_str += " ."
                        
                        p_idx += 1

                    print(e_str)
                    edf_idx += 1
                
                name_idx += 1
            
            for rti in form.get_random_term_idx():
                rterm = terms[rti]
                if isinstance(rterm,rs):
                    if rterm.var_coef > 1 and len(rterm.variables) > 1:
                        for li in range(rterm.var_coef):
                            print(smooth_names[name_idx] + f":{li}; edf: {round(term_edf[edf_idx],ndigits=3)}")
                            edf_idx += 1
                    else:
                        print(smooth_names[name_idx] + f"; edf: {round(term_edf[edf_idx],ndigits=3)}")
                        edf_idx += 1
                else:
                    print(smooth_names[name_idx] + f"; edf: {round(term_edf[edf_idx],ndigits=3)}")
                    edf_idx += 1
                name_idx += 1
            
            if ps is not None:
                print("\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!")

            if pen_out == 1:
                print("\nOne term has been effectively penalized to zero and is marked with a '*'")
            elif pen_out > 1:
                print(f"\n{pen_out} terms have been effectively penalized to zero and are marked with a '*'")

def compute_bias_corrected_edf(model,overwrite:bool=False) -> None:
    """This function computes and assigns smoothing bias corrected (term-wise) estimated degrees of freedom.

    For a definition of smoothing bias-corrected estimated degrees of freedom see Wood (2017).

    **Note:** This function modifies ``model``, setting ``edf1`` and ``term_edf1`` attributes.

    References:
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

    :param model: Model for which to compute p values.
    :type model: mssm.models.GSMM | mssm.models.GAMMLSS | mssm.models.GAMM 
    :param overwrite: Whether previously computed bias corrected edf should be overwritten. Otherwise this function immediately terminates if ``model.edf1 is not None``, defaults to False
    :type overwrite: bool, optional
    :rtype: None
    """

    if model.edf1 is None or overwrite:
        term_edf1 = []
        edf1 = 0
        
        if isinstance(model.family,Family): # GAMM case
            scale = model.scale

            # Start with forming F matrix
            nH = -1*model.hessian
            F = model.lvi.T@model.lvi@(nH*scale)

            edf1 += model.formulas[0].unpenalized_coef
            
        else:

            # Start with forming F matrix
            nH = -1*model.hessian
            F = model.lvi.T@model.lvi@nH

            edf1 += np.sum([form.unpenalized_coef for form in model.formulas])
        
        F_diag = F.diagonal()

        prev_start_idx = 0
        for pen in model.overall_penalties:
            if pen.start_index > prev_start_idx:
                S_start = pen.start_index
                S_len = pen.rep_sj * pen.S_J.shape[1]
                S_end = pen.start_index + S_len

                # Now compute sum of diagonal
                Fjc = F[S_start:S_end,:]
                Fjr = F[:,S_start:S_end]
                Fjd = F_diag[S_start:S_end]

                Fjtrace = Fjc.multiply(Fjr.T).sum()
                t_edf1 = 2*np.sum(Fjd) - Fjtrace
                term_edf1.append(t_edf1)
                edf1 += t_edf1

                # Update current start index
                prev_start_idx = pen.start_index
        
        model.edf1 = edf1
        model.term_edf1 = term_edf1

def approx_smooth_p_values(model,par:int=0,n_sel:int=1e5,edf1:bool=True,force_approx:bool=False,seed:int=0) -> tuple[list[float],list[float]]:
        """ Function to compute approximate p-values for smooth terms, testing whether :math:`\\mathbf{f}=\\mathbf{X}\\boldsymbol{\\beta} = \\mathbf{0}` based on the algorithm by Wood (2013).

        Wood (2013, 2017) generalize the :math:`\\boldsymbol{\\beta}_j^T\\mathbf{V}_{\\boldsymbol{\\beta}_j}^{-1}\\boldsymbol{\\beta}_j` test-statistic for parametric terms
        (computed by function :func:`mssm.models.print_parametric_terms`) to the coefficient vector :math:`\\boldsymbol{\\beta}_j` parameterizing smooth functions. :math:`\\mathbf{V}` here is the
        covariance matrix of the posterior distribution for :math:`\\boldsymbol{\\beta}` (see Wood, 2017). The idea is to replace
        :math:`\\mathbf{V}_{\\boldsymbol{\\beta}_j}^{-1}` with a rank :math:`r` pseudo-inverse (smooth blocks in :math:`\\mathbf{V}` are usually
        rank deficient). Wood (2013, 2017) suggest to base :math:`r` on the estimated degrees of freedom for the smooth term in question - but that :math:`r`  is usually not integer.

        They provide a generalization that addresses the realness of :math:`r`, resulting in a test statistic :math:`T_r`, which follows a weighted
        Chi-square distribution under the Null. Following the recommendation in Wood (2013) we here approximate the reference distribution under the Null by means of the computations outlined in
        the paper by Davies (1980). If this fails, we fall back on a Gamma distribution with :math:`\\alpha=r/2` and :math:`\\phi=2`.
        
        In case of a two-parameter distribution (i.e., estimated scale parameter :math:`\\phi`), the Chi-square reference distribution needs to be corrected, again resulting in a
        weighted chi-square distribution which should behave something like a F distribution with DoF1 = :math:`r` and DoF2 = :math:`\\epsilon_{DoF}` (i.e., the residual degrees of freedom),
        which would be the reference distribution for :math:`T_r/r` if :math:`r` were integer and :math:`\\mathbf{V}_{\\boldsymbol{\\beta}_j}` full rank. We again follow the recommendations by Wood (2013)
        and rely on the methods by Davies (1980) to compute the p-value under this reference distribution. If this fails, we approximate the reference distribution for :math:`T_r/r` with a Beta distribution, with
        :math:`\\alpha=r/2` and :math:`\\beta=\\epsilon_{DoF}/2` (see Wikipedia for the specific transformation applied to :math:`T_r/r` so that the resulting transformation is approximately beta
        distributed) - which is similar to the Gamma approximation used for the Chi-square distribution in the no-scale parameter case.

        **Warning:** The resulting p-values are **approximate**. They should only be treated as indicative.

        **Note:** Just like in ``mgcv``, the returned p-value is an average: two p-values are computed because of an ambiguity in forming :math:`T_r` and averaged to get the final one. For :math:`T_r` we return the max of the two
        alternatives.

        References:
         - Davies, R. B. (1980). Algorithm AS 155: The Distribution of a Linear Combination of χ2 Random Variables.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
         - Wood, S. N. (2013). On p-values for smooth components of an extended generalized additive model.
         - ``testStat`` function in mgcv, see: https://github.com/cran/mgcv/blob/master/R/mgcv.r#L3780
        
        :param model: Model for which to compute p values.
        :type model: mssm.models.GSMM | mssm.models.GAMMLSS | mssm.models.GAMM 
        :param par: Distribution parameter for which to compute p-values. Ignored when ``model`` is a GAMM. Defaults to 0
        :type par: int, optional
        :param n_sel: Maximum number of rows of model matrix. For models with more observations a random sample of ``n_sel`` rows is obtained. Defaults to 1e5 
        :type n_sel: int, optional
        :param edf1: Whether or not the estimated degrees of freedom should be corrected for smoothnes bias. Doing so results in more accurate p-values but can be expensive for large models for which the difference is anyway likely to be marginal. Defaults to True
        :type edf1: bool, optional
        :param force_approx: Whether or not the p-value should be forced to be approximated based on a Gamma/Beta distribution. Only use for testing - in practice you want to keep this at ``False``. Defaults to False
        :type force_approx: bool, optional
        :param seed: Random seed determining the random sample computation. Defaults to 0
        :type seed: int, optional
        :return: Tuple conatining two lists: first list holds approximate p-values for all smooth terms, second list holds test statistic.
        :rtype: tuple[list[float],list[float]]
        """

        np_gen = np.random.default_rng(seed)

        if edf1: # Use smoothing bias corrected edf as discussed by Wood (2017)
            compute_bias_corrected_edf(model)

        if isinstance(model.family,Family): # GAMM case
            form = model.formulas[0]
            X = model.get_mmat()
            coef = model.coef.flatten()
            lvi = model.lvi
            scale = model.scale
            term_edf = model.term_edf1 if edf1 else model.term_edf
            rs_df = X.shape[0] - model.edf

        else:
            # GAMMLSS or GSMM case
            form = model.formulas[par]
            X = model.get_mmat()[par]

            # Get coef and lvi for selected formula
            split_coef = np.split(model.coef,model.coef_split_idx)
            split_idx = np.ndarray.flatten(np.split(np.arange(len(model.coef)),model.coef_split_idx)[par])
            coef = np.ndarray.flatten(split_coef[par])
            lvi = model.lvi[:,split_idx]
            scale = 1
            term_edf = []
            rs_df = None

            prev_start_idx = 0
            prev_edf_idx = 0
            for pen in model.overall_penalties:
                if pen.start_index > prev_start_idx:
                    
                    if pen.dist_param > par:
                        break

                    if pen.dist_param == par: # Collect only those belonging to this par
                        if edf1:
                            term_edf.append(min(model.term_edf1[prev_edf_idx], pen.rep_sj * pen.S_J.shape[1]))
                        else:
                            term_edf.append(min(model.term_edf[prev_edf_idx], pen.rep_sj * pen.S_J.shape[1]))
                    
                    # Keep track of start idx (coef idx) and edf idx.
                    prev_start_idx = pen.start_index
                    prev_edf_idx += 1

        terms = form.get_terms()

        # Find smooth terms in formula
        st_idx = form.get_smooth_term_idx()

        # Set-up storage
        ps = []
        Trs = []

        # Loop over smooth terms
        start_coef = form.unpenalized_coef # Start indexing V_b after un-penalized coef
        edf_idx = 0
        for sti in st_idx:
            if isinstance(terms[sti],fs) == False:
                
                n_s_coef = form.coef_per_term[sti]
                enumerator = range(1)
                if not terms[sti].by is None and terms[sti].id is None:
                    n_levels = len(form.get_factor_levels()[terms[sti].by])
                    enumerator = range(n_levels)
                    n_s_coef = int(n_s_coef / n_levels)

                for _ in enumerator:
                    # Extract coefficients corresponding to smooth
                    end_coef = start_coef+n_s_coef
                    s_coef = coef[start_coef:end_coef].reshape(-1,1)

                    # Extract sub-block of V_{b_j}
                    V_b_j = ((lvi[:,start_coef:end_coef].T@lvi[:,start_coef:end_coef])*scale).toarray()

                    # Form QR of sub-block of X associated with current smooth
                    X_b_j = X[:,start_coef:end_coef]

                    if X_b_j.shape[0] > n_sel:
                        # Select subset of rows randomly as suggested by Wood (2013)
                        sel = np_gen.choice(X_b_j.shape[0],size=n_sel,replace=False)
                        X_b_j = X_b_j[sel,:]

                    R = np.linalg.qr(X_b_j.toarray(),mode='r')

                    # Form generalized inverse of V_f (see Wood, 2017; section 6.12.1)
                    RVR = R@V_b_j@R.T

                    # Eigen-decomposition:
                    s, U =scp.linalg.eigh(RVR)
                    s = np.flip(s)
                    U = np.flip(U,axis=1)

                    # get edf for this term and compute r,k,v, and p from Wood (2017)
                    r = term_edf[edf_idx]
                    k = int(r)
                    if k > 0:
                        v = r-k

                        if v == 0: # Integer case - standard F-test
                            
                            if isinstance(model.family,Family) and model.family.twopar:
                                Tr1 /= k
                                Tr2 /= k

                                p1 = 1 - scp.stats.f.cdf(Tr1,k,rs_df)
                                p2 = 1 - scp.stats.f.cdf(Tr2,k,rs_df)

                            else: # Integer case - standard Chi-square test
                                p1 = 1 - scp.stats.chi2.cdf(Tr1,k)
                                p2 = 1 - scp.stats.chi2.cdf(Tr2,k)

                        else:
                            p = np.power(v*(1-v)/2,0.5)

                            #print(k,s,s[:k-1],s[k-1],s[k])
                            # Take only eigen-vectors need for the actual product for the test-statistic
                            U = U[:,:k+1]

                            # Fix sign of Eigen-vectors to sign of first row (based on testStat function in mgcv)
                            sign = np.sign(U[0,:])
                            U *= sign

                            # Now we can reform the diagonal matrix needed for inverting RVR (computation follows Wood, 2012)
                            S = np.zeros((U.shape[1],U.shape[1]))
                            for ilam,lam in enumerate(s[:k-1]):
                                S[ilam,ilam] = 1/lam

                            Lb = np.array([[np.power(s[k-1],-0.5),0],
                                        [0,np.power(s[k],-0.5)]])
                            
                            Bb = np.array([[1,p],
                                        [p,v]])
                            
                            B = Lb@Bb@Lb.T
                            
                            S[k-1:k+1,k-1:k+1] = B

                            # And finally compute the inverse
                            RVRI1 = U@S@U.T

                            # Also compute inverse for alternative version of Eigen-vectors (see Wood, 2017):
                            U *= sign
                            RVRI2 = U@S@U.T

                            # And the test statistic defined in Wood (2012)
                            Tr1 = (s_coef.T@R.T@RVRI1@R@s_coef)[0,0]
                            Tr2 = (s_coef.T@R.T@RVRI2@R@s_coef)[0,0]

                            # And the weights for the chi-square distributions..
                            v1 = (v + 1 + np.pow(1 - np.pow(v,2),0.5))/2
                            v2 = v + 1 - v1
                            
                            # Now we need the p-value.
                            if isinstance(model.family,Family) and model.family.twopar:
                                # First try Davies as discussed by Wood (2013)

                                # We have: v > 0. Based on Wood (2013), if k == 1 we need only 2 weighted chi-square variables.
                                # When k > 1, we need 3 chi-square variables where the first one is unweighted with k-2+1 dof.

                                # So start with unweighted variable
                                n = []
                                lb = []
                                if k > 1:
                                    n.append(k-2+1) 
                                    lb.append(1)
                                
                                # Now the weighted variables
                                n.extend([1,1,rs_df])
                                lb.extend([v1,v2])

                                # cast to np.array - not lb since we still need to add weight for last variable
                                nc = np.array([0 for _ in range(len(n))],dtype=np.float64)
                                n = np.array(n)
                                
                                # Compute as discussed by Wood (2017)
                                p1,code1,_ = davies.daviesQF(np.array([*lb,-Tr1/rs_df]),nc,n,0,0,2e-5,len(n),20000)
                                p2,code2,_ = davies.daviesQF(np.array([*lb,-Tr2/rs_df]),nc,n,0,0,2e-5,len(n),20000)
                                p1 = 1 - p1
                                p2 = 1 - p2

                                # Since this approximates an F statistic we need to adjust Tr (see below):
                                Tr1 /= r
                                Tr2 /= r

                                if code1 > 0 or code2 > 0 or force_approx:

                                    if code1 == 2 or code2 == 2:
                                        warnings.warn("Round-off error in p-value computation might be problematic. Proceed with caution.")
                                
                                    if code1 != 2 or code2 != 2:
                                        warnings.warn(f"Falling back to approximate p-value computation. Error codes: {code1}, {code2}")
                                        # Davies failed... Now, in case of an estimated scale parameter and integer r: Tr/r \\sim F(r,rs_df)
                                        # So to approximate the case where r is real, we can use a Beta (see Wikipedia):
                                        # if X \\sim F(d1,d2) then (d1*X/d2) / (1 + (d1*X/d2)) \\sim Beta(d1/2,d2/2)
                                        
                                        p1 = 1 - scp.stats.beta.cdf((r*Tr1/rs_df) / (1 + (r*Tr1/rs_df)),a=r/2,b=rs_df/2)
                                        p2 = 1 - scp.stats.beta.cdf((r*Tr2/rs_df) / (1 + (r*Tr2/rs_df)),a=r/2,b=rs_df/2)
                                
                            else:
                                # First try Davies as discussed by Wood (2013)

                                # First handle un-weighted variable as described above
                                n = []
                                lb = []
                                if k > 1:
                                    n.append(k-2+1) 
                                    lb.append(1)
                                
                                # Now the weighted variables
                                n.extend([1,1])
                                lb.extend([v1,v2])
    
                                # cast to np.array
                                nc = np.array([0. for _ in range(len(n))],dtype=np.float64)
                                n = np.array(n)
                                lb = np.array(lb)
                                
                                p1,code1,_ = davies.daviesQF(lb,nc,n,Tr1,0,2e-5,len(n),20000)
                                p2,code2,_ = davies.daviesQF(lb,nc,n,Tr2,0,2e-5,len(n),20000)
                                p1 = 1 - p1
                                p2 = 1 - p2

                                if code1 > 0 or code2 > 0 or force_approx:

                                    if code1 == 2 or code2 == 2:
                                        warnings.warn("Round-off error in p-value computation might be problematic. Proceed with caution.")
                                
                                    if code1 != 2 or code2 != 2:
                                        warnings.warn(f"Falling back to approximate p-value computation. Error codes: {code1}, {code2}")
                                        # Davies failed... Now, Wood (2013) suggest that the Chi-square distribution of
                                        # the Null can be approximated with a gamma with alpha=r/2 and scale=2:
                                        
                                        p1 = 1-scp.stats.gamma.cdf(Tr1,a=r/2,scale=2)
                                        p2 = 1-scp.stats.gamma.cdf(Tr2,a=r/2,scale=2)

                        p = (p1 + p2)/2
                        Tr = max(Tr1,Tr2)
                    
                    else:
                        warnings.warn(f"Function {sti} appears to be fully penalized. This function does not support such terms. Setting p=1 and Tr=-1.")
                        p = 1
                        Tr = -1
                
                    ps.append(p)
                    Trs.append(Tr)

                    # Prepare for next term
                    edf_idx += 1
                    start_coef += n_s_coef

            else: # Random smooth terms are fully penalized
                start_coef += form.coef_per_term[sti]
                edf_idx += 1
        
        return ps,Trs

def adjust_CI(model,n_ps:int,b:np.ndarray,predi_mat:scp.sparse.csc_array,use_terms:list[int]|None,alpha:float,seed:int|None,par:int=0) -> np.ndarray:
        """Internal function to adjust point-wise CI to behave like whole-function interval (based on Wood, 2017; section 6.10.2 and Simpson, 2016):

        ``model.coef +- b`` gives point-wise interval, and for the interval to cover the whole-function, ``1-alpha`` % of posterior samples should
        be expected to fall completely within these boundaries.

        From section 6.10 in Wood (2017) we have that :math:`\\boldsymbol{\\beta} | \\mathbf{y}, \\boldsymbol{\\lambda} \\sim N(\\hat{\\boldsymbol{\\beta}},\\mathbf{V})`.
        :math:`\\mathbf{V}` is the covariance matrix of this conditional posterior, and can be obtained by evaluating ``model.lvi.T @ model.lvi * model.scale`` (``model.scale`` should be
        set to 1 for :class:`msssm.models.GAMMLSS` and :class:`msssm.models.GSMM`).

        The implication of this result is that we can also expect the deviations :math:`\\boldsymbol{\\beta} - \\hat{\\boldsymbol{\\beta}}`  to follow
        :math:`\\boldsymbol{\\beta} - \\hat{\\boldsymbol{\\beta}} | \\mathbf{y}, \\boldsymbol{\\lambda} \\sim N(0,\\mathbf{V})`. In line with the whole-function interval definition above, ``1-alpha`` % of
        ``predi_mat@[*coef - coef]`` (where ``[*coef - coef]`` representes the deviations :math:`\\boldsymbol{\\beta} - \\hat{\\boldsymbol{\\beta}}`) should fall within ``[b,-b]``.
        Wood (2017) suggests to find ``a`` so that ``[a*b,a*-b]`` achieves this.

        To do this, we find ``a`` for every ``predi_mat@[*coef - coef]`` and then select the final one so that ``1-alpha`` % of samples had an equal or lower
        one. The consequence: ``1-alpha`` % of samples drawn should fall completely within the modified boundaries.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
         - Simpson, G. (2016). Simultaneous intervals for smooths revisited.

        :param model: Model for which to compute p values.
        :type model: mssm.models.GSMM | mssm.models.GAMMLSS | mssm.models.GAMM 
        :param n_ps: Number of samples to obtain from posterior.
        :type n_ps: int
        :param b: Ci boundary of point-wise CI.
        :type b: np.ndarray
        :param predi_mat: Model matrix for a particular smooth term or additive combination of parameters evaluated usually at a representative sample of predictor variables.
        :type predi_mat: scp.sparse.csc_array
        :param use_terms: The indices corresponding to the terms that should be used to obtain the prediction or ``None`` in which case all terms will be used.
        :type use_terms: list[int] | None
        :param alpha: The alpha level to use for the whole-function interval adjustment calculation as outlined above.
        :type alpha: float
        :param seed: Can be used to provide a seed for the posterior sampling.
        :type seed: int | None
        :param par: The index corresponding to the parameter of the log-likelihood for which samples are to be obtained for the coefficients, defaults to 0.
        :type par: int, optional
        :return: The adjusted vector ``b``
        :rtype: np.ndarray
        """

        use_post = None
        if not use_terms is None:
            # If we have many random factor levels, but want to make predictions only
            # for fixed effects, it's wasteful to sample all coefficients from posterior.
            # The code below performs a selection of the coefficients to be sampled.
            use_post = predi_mat.sum(axis=0) != 0
            use_post = np.arange(0,predi_mat.shape[1])[use_post]

        # Sample deviations [*coef - coef] from posterior of model
        post = model.sample_post(n_ps,use_post,deviations=True,seed=seed,par=par)

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

def compute_reml_candidate_GAMM(family:Family,y:np.ndarray,X:scp.sparse.csc_array,penalties:list[LambdaTerm],n_c:int=10,offset:float|np.ndarray=0,init_eta:np.ndarray|None=None,method:str='Chol',compute_inv:bool=False,origNH:float|None=None) -> tuple[float,scp.sparse.csc_array|None,scp.sparse.csc_array,list[int],np.ndarray,float,float,float]:
    """Allows to evaluate REML criterion (e.g., Wood, 2011; Wood, 2016) efficiently for a set of \\lambda values for a GAMM model.

    Internal function used for computing the correction applied to the edf for the GLRT - based on Wood (2017) and Wood et al., (2016).

    See :func:`REML` function for more details.

    :param family: Family of the model
    :type family: Family
    :param y: vector of observations
    :type y: np.ndarray
    :param X: Model matrix
    :type X: scp.sparse.csc_array
    :param penalties: List of penalties
    :type penalties: list[LambdaTerm]
    :param n_c: Number of cores to use, defaults to 10
    :type n_c: int, optional
    :param offset: Fixed offset to add to eta, defaults to 0
    :type offset: float | np.ndarray, optional
    :param init_eta: Initial vector for linear predictor, defaults to None
    :type init_eta: np.ndarray | None, optional
    :param method: Method to use to solve for coefficients, defaults to 'Chol'
    :type method: str, optional
    :param compute_inv: Whether to compute the inverse of the pivoted Cholesky of the negative hessian of the penalized llk, defaults to False
    :type compute_inv: bool, optional
    :param origNH: Optional external scale parameter, defaults to None
    :type origNH: float | None, optional
    :return: reml criterion, un-pivoted inverse of the pivoted Cholesky of the negative hessian of the penalized llk, pivoted Cholesky, pivot column indices, coefficients, estimated scale, total edf, llk
    :rtype: tuple[float, scp.sparse.csc_array|None, scp.sparse.csc_array, list[int], np.ndarray, float, float, float]
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
            yb,Xb,z,Wr = update_PIRLS(y,yb,mu,eta-offset,X,Xb,family,None)

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
                yb,Xb,z,Wr = update_PIRLS(y,yb,mu,eta-offset,X,Xb,family,None)

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
        _,_,edf,_,_,scale = update_scale_edf(y,z,eta,Wr,X.shape[0],X.shape[1],LP,None,Pr,None,None,family,penalties,keep,drop,n_c)
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

def compute_REML_candidate_GSMM(family:GAMLSSFamily|GSMMFamily,y:np.ndarray|list[np.ndarray],Xs:list[scp.sparse.csc_array],penalties:list[LambdaTerm],coef:np.ndarray,n_coef:int,coef_split_idx:list[int],method:str="Chol",conv_tol:float=1e-7,n_c:int=10,bfgs_options:dict={},origNH:scp.sparse.csc_array|None=None) -> tuple[float,scp.sparse.csc_array,scp.sparse.csc_array,np.ndarray,float,float]:
    """Allows to evaluate REML criterion (e.g., Wood, 2011; Wood, 2016) efficiently for a set of \\lambda values for a GSMM or GAMMLSS.

    Internal function used for computing the correction applied to the edf for the GLRT - based on Wood (2017) and Wood et al., (2016).

    See :func:`REML` function for more details.
    
    :param family: Model Family
    :type family: GAMLSSFamily | GSMMFamily
    :param y: Vector of observations or list of vectors (for GSMM)
    :type y: np.ndarray | list[np.ndarray]
    :param Xs: List of model matrices
    :type Xs: list[scp.sparse.csc_array]
    :param penalties: List of penalties
    :type penalties: list[LambdaTerm]
    :param coef: Final coefficient estimate obtained from estimation - used to initialize
    :type coef: np.ndarray
    :param n_coef: Number of coefficients
    :type n_coef: int
    :param coef_split_idx: The indices at which to split the overall coefficient vector into separate lists - one per parameter.
    :type coef_split_idx: list[int]
    :param method: Method to use to solve for the coefficients (lambda parameters in case this is set to 'qEFS'), defaults to "Chol"
    :type method: str, optional
    :param conv_tol: Tolerance, defaults to 1e-7
    :type conv_tol: float, optional
    :param n_c: Number of cores to use, defaults to 10
    :type n_c: int, optional
    :param bfgs_options: An optional dictionary holding arguments that should be passed on to the call of :func:`scipy.optimize.minimize` if ``method=='qEFS'``, defaults to {}
    :type bfgs_options: dict, optional
    :param origNH: Optional external hessian matrix, defaults to None
    :type origNH: scp.sparse.csc_array | None, optional
    :return: reml criterion,conditional covariance matrix of coefficients for this lambda, un-pivoted inverse of the pivoted Cholesky of the negative hessian of the penalized llk, coefficients, total edf, llk
    :rtype: tuple[float, scp.sparse.csc_array, scp.sparse.csc_array, np.ndarray, float, float]
    """

    # Build current penalties
    S_emb,S_pinv,_,FS_use_rank = compute_S_emb_pinv_det(n_coef,penalties,"svd")

    try:
        if isinstance(family,GSMMFamily): # GSMM
            
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


def REML(llk:float,nH:scp.sparse.csc_array,coef:np.ndarray,scale:float,penalties:list[LambdaTerm],keep:list[int]|None=None) -> float|np.ndarray:
   """
   Based on Wood (2011). Exact REML for Gaussian GAM, Laplace approximate (Wood, 2016) for everything else.
   Evaluated after applying stabilizing reparameterization discussed by Wood (2011).

   **Important**: the dimension of the output depend on the shape of ``coef``. If ``coef`` is flattened, then the output will be a float. If ``coef`` is of shape (-1,1), the output will be [[float]].

   References:
    - Wood, S. N., (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models.
    - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models

   :param llk: log-likelihood of model
   :type llk: float
   :param nH: negative hessian of log-likelihood of model
   :type nH: scp.sparse.csc_array
   :param coef: Estimated vector of coefficients of shape (-1,1)
   :type coef: np.ndarray
   :param scale: (Estimated) scale parameter - can be set to 1 for GAMLSS or GSMMs.
   :type scale: float
   :param penalties: List of penalties that were part of the model.
   :type penalties: [LambdaTerm]
   :param keep: Optional List of indices corresponding to identifiable coefficients. Coefficients not in this list (not identifiable) are dropped from the negative hessian of the penalized log-likelihood. Can also be set to ``None`` (default) in which case all coefficients are treated as identifiable.
   :type keep: list[int]|None, optional
   :return: (Approximate) REML score
   :rtype: float|np.ndarray
   """ 

   # Compute S_\\lambda before any re-parameterization
   S_emb,_,_,_ = compute_S_emb_pinv_det(len(coef),penalties,"svd")

   # Re-parameterize as shown in Wood (2011) to enable stable computation of log(|S_\\lambda|+)
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
   
   # Now we need to compute log(|S_\\lambda|+), Wood shows that after the re-parameterization log(|S_\\lambda|)
   # can be computed separately from the diagonal or R if Q@R=S_reps[i] for all terms i. Below we compute from
   # the diagonal of the cholesky of the term specific S_reps[i], applying conditioning as shown in Appendix B of Wood (2011).
   lgdetS = 0
   for Si,S_rep in enumerate(S_reps):
        # We need to evaluate log(|S_\\lambda/\\phi|+) after re-parameterization of S_\\lambda (so this will be a regular determinant).
        # We have that (https://en.wikipedia.org/wiki/Determinant):
        #   det(S_\\lambda * 1/\\phi) = (1/\\phi)^p * det(S_\\lambda)
        # taking logs:
        #    log(det(S_\\lambda * 1/\\phi)) = log((1/\\phi)^p) + log(det(S_\\lambda))
        # We know that log(det(S_\\lambda)) is insensitive to whether or not we re-parameterize, so
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
        
   # Now log(|nH+S_\\lambda|)... Wood (2011) shows stable computation based on QR decomposition, but
   # we will generally not be able to compute a QR decomposition of X so that X.T@X=H efficiently.
   # Hence, we simply rely on the pivoted cholesky (again pre-conditioned) used for fitting (based on S_\\lambda before
   # re-parameterization).
   H_pen = nH + S_emb/scale

   # Drop unidentifiable parameters
   if keep is not None:
       H_pen = H_pen[:,keep]
       H_pen = H_pen[keep,:]

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

def estimateVp(model,nR:int = 250,grid_type:str = 'JJJ1',a:float=1e-7,b:float=1e7,df:int=40,n_c:int=10,drop_NA:bool=True,method:str="Chol",Vp_fidiff:bool=False,use_importance_weights:bool=True,prior:Callable|None=None,seed:int|None=None,**bfgs_options) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate covariance matrix :math:`\\mathbf{V}^{\\boldsymbol{\\rho}}` of posterior for :math:`\\boldsymbol{\\rho} = log(\\boldsymbol{\\lambda})`.
    
    Either :math:`\\mathbf{V}^{\\boldsymbol{\\rho}}` is based on finite difference approximation or on a PQL approximation (see ``grid_type`` parameter), or it is estimated via numerical
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

    :param model: GAMM, GAMMLSS, or GSMM model (which has been fitted) for which to estimate :math:`\\mathbf{V}`
    :type model: mssm.models.GSMM | mssm.models.GAMMLSS | mssm.models.GAMM 
    :param nR: In case ``grid!="JJJ1"``, ``nR`` samples/reml scores are generated/computed to numerically evaluate the expectations necessary for the uncertainty correction, defaults to 250
    :type nR: int, optional
    :param grid_type: How to compute the smoothness uncertainty correction. Setting ``grid_type="JJJ1"`` means a PQL or finite difference approximation is obtained. Setting ``grid_type="JJJ2"`` means numerical integration is performed - see :func:`correct_VB` for details , defaults to 'JJJ1'
    :type grid_type: str, optional
    :param a: Any of the :math:`\\lambda` estimates obtained from ``model`` (used to define the mean for the posterior of :math:`\\boldsymbol{\\rho}|y \\sim N(log(\\hat{\\boldsymbol{\\rho}}),\\mathbf{V}^{\\boldsymbol{\\rho}})` used to sample ``nR`` candidates) which are smaller than this are set to this value as well, defaults to 1e-7 the minimum possible estimate
    :type a: float, optional
    :param b: Any of the :math:`\\lambda` estimates obtained from ``model`` (used to define the mean for the posterior of :math:`\\boldsymbol{\\rho}|y \\sim N(log(\\hat{\\boldsymbol{\\rho}}),\\mathbf{V}^{\\boldsymbol{\\rho}})` used to sample ``nR`` candidates) which are larger than this are set to this value as well, defaults to 1e7 the maximum possible estimate
    :type b: float, optional
    :param df: Degrees of freedom used for the multivariate t distribution used to sample the next set of candidates. Setting this to ``np.inf`` means a multivariate normal is used for sampling, defaults to 40
    :type df: int, optional
    :param n_c: Number of cores to use during parallel parts of the correction, defaults to 10
    :type n_c: int, optional
    :param drop_NA: Whether to drop rows in the **model matrices** corresponding to NAs in the dependent variable vector. Defaults to True.
    :type drop_NA: bool,optional
    :param method: Which method to use to solve for the coefficients (and smoothing parameters). The default ("Chol") relies on Cholesky decomposition. This is extremely efficient but in principle less stable, numerically speaking. For a maximum of numerical stability set this to "QR/Chol". In that case a QR decomposition is used - which is first pivoted to maximize sparsity in the resulting decomposition but also pivots for stability in order to get an estimate of rank defficiency. A Cholesky is than used using the combined pivoting strategy obtained from the QR. This takes substantially longer. If this is set to ``'qEFS'``, then the coefficients are estimated via quasi netwon and the smoothing penalties are estimated from the quasi newton approximation to the hessian. This only requieres first derviative information. Defaults to "Chol".
    :type method: str,optional
    :param Vp_fidiff: Whether to rely on a finite difference approximation to compute :math:`\\mathbf{V}^{\\boldsymbol{\\rho}}` or on a PQL approximation. The latter is exact for Gaussian and canonical GAMs and far cheaper if many penalties are to be estimated. Defaults to False (PQL approximation)
    :type Vp_fidiff: bool,optional
    :param use_importance_weights: Whether to rely importance weights to compute the numerical integration when ``grid_type != 'JJJ1'`` or on the log-densities of :math:`\\mathbf{V}^{\\boldsymbol{\\rho}}` - the latter assumes that the unconditional posterior is normal. Defaults to True (Importance weights are used)
    :type use_importance_weights: bool,optional
    :param prior: An (optional) instance of an arbitrary class that has a ``.logpdf()`` method to compute the prior log density of a sampled candidate. If this is set to ``None``, the prior is assumed to coincide with the proposal distribution, simplifying the importance weight computation. Ignored when ``use_importance_weights=False``. Defaults to None
    :type prior: Callable|None, optional
    :param recompute_H: Whether or not to re-compute the Hessian of the log-likelihood at an estimate of the mean of the Bayesian posterior :math:`\\boldsymbol{\\beta}|y` before computing the (uncertainty/bias corrected) edf. Defaults to False
    :type recompute_H: bool, optional
    :param seed: Seed to use for random parts of the correction. Defaults to None
    :type seed: int|None,optional
    :param bfgs_options: Any additional keyword arguments that should be passed on to the call of ``scipy.optimize.minimize``. If none are provided, the ``gtol`` argument will be initialized to 1e-3. Note also, that in any case the ``maxiter`` argument is automatically set to 100. Defaults to None.
    :type bfgs_options: key=value,optional
    :return: A tuple with 5 elements: an estimate of the covariance matrix of the posterior for :math:`\\boldsymbol{\\rho} = log(\\boldsymbol{\\lambda})`, a regularized version of the former, a root of the covariance matrix, a root of the regularized covariance matrix, and an estimate of the mean of the posterior
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    np_gen = np.random.default_rng(seed)

    family = model.family

    if not grid_type in ["JJJ1","JJJ2"]:
        raise ValueError("'grid_type' has to be set to one of 'JJJ1', 'JJJ2'.")
    
    if isinstance(family,Family) and model.rho is not None and grid_type != "JJJ1":
        raise ValueError("For models with an ar1 model only grid_type='JJJ1' is supported.")

    if isinstance(family,GSMMFamily):
        if not bfgs_options:
            bfgs_options = {"ftol":1e-9,
                            "maxcor":30,
                            "maxls":100}

    nPen = len(model.overall_penalties)
    rPen = copy.deepcopy(model.overall_penalties)
    S_emb,_,_,_ = compute_S_emb_pinv_det(model.hessian.shape[1],model.overall_penalties,"svd")
    
    if isinstance(family,Family):
        y = model.formulas[0].y_flat[model.formulas[0].NOT_NA_flat]

        if not model.formulas[0].get_lhs().f is None:
            # Optionally apply function to dep. var. before fitting.
            y = model.formulas[0].get_lhs().f(y)

        X = model.get_mmat()

        orig_scale = family.scale
        if family.twopar:
            _,orig_scale = model.get_pars()
    else:
        if isinstance(family,GAMLSSFamily):
            y = model.formulas[0].y_flat[model.formulas[0].NOT_NA_flat]

            if not model.formulas[0].get_lhs().f is None:
                # Optionally apply function to dep. var. before fitting. Not sure why that would be desirable for this model class...
                y = model.formulas[0].get_lhs().f(y)
            
            Xs = model.get_mmat()

        else: # Need all y vectors in y, i.e., y is actually ys
            ys = []
            for fi,form in enumerate(model.formulas):
                
                # Repeated y-variable - don't have to pass all of them
                if fi > 0 and form.get_lhs().variable == model.formulas[0].get_lhs().variable:
                    ys.append(None)
                    continue

                # New y-variable
                if drop_NA:
                    y = form.y_flat[form.NOT_NA_flat]
                else:
                    y = form.y_flat

                # Optionally apply function to dep. var. before fitting. Not sure why that would be desirable for this model class...
                if not form.get_lhs().f is None:
                    y = form.get_lhs().f(y)
                
                # And collect
                ys.append(y)
            
            y = ys
            Xs = model.get_mmat(drop_NA=drop_NA)

        X = Xs[0]
        orig_scale = 1
        init_coef = copy.deepcopy(model.coef)
    
    if grid_type == "JJJ1"  or grid_type == "JJJ2":
        # Approximate Vp via finite differencing or PQL approximation of negative REML.

        # Set up mean log-smoothing penalty vector - ignoring any a and b limits provided.
        ep = np.log(np.array([pen.lam for pen in model.overall_penalties]).reshape(-1,1))

        if Vp_fidiff:
            def reml_wrapper(rho):
                
                for peni in range(len(rho)):
                    rPen[peni].lam = np.exp(rho[peni])
                
                if isinstance(family,Family):
                    reml,_,_,_,_,_,_,_ = compute_reml_candidate_GAMM(family,y,X,rPen,n_c,model.offset,model.preds[0],method)
                else:
                    reml,_,_,_,_,_ = compute_REML_candidate_GSMM(family,y,Xs,rPen,init_coef,len(init_coef),model.coef_split_idx,n_c=n_c,method=method,bfgs_options=bfgs_options)
                
                return -reml
            
            nHp = scp.differentiate.hessian(lambda r: np.apply_along_axis(reml_wrapper, axis=0, arr=r),ep.flatten(),order=4,maxiter=3,tolerances={"atol":1e-7,"rtol":1e-5})

            if False in nHp.success.flatten():
                warnings.warn(f"Finite differencing step failed to reach tolerance. Consider setting ``Vp_fidiff=False`. Element-wise error:\n {nHp.error}.")
            if -3 in nHp.status.flatten():
                raise ValueError("Finite differencing step failed. Set ``Vp_fidiff=False``.")
            nHp = nHp.ddf
            
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
            Vp, Vpr, Ri, Rir, _, _ = compute_Vp_WPS(model.lvi,
                                                 model.hessian,
                                                 S_emb,
                                                 model.overall_penalties,
                                                 model.coef,
                                                 scale=orig_scale if isinstance(family,Family) else 1)
        
        if grid_type == "JJJ1":
            return Vp,Vpr,Ri,Rir,ep
            
    # Strategies JJJ2
    rGrid = np.array([])
    remls = []
    # Generate \\lambda values from Vpr for which to compute REML, and Vb

    # First recompute mean, this time accepting limits imposed by a and b
    ep = np.log(np.array([min(b,max(a,pen.lam)) for pen in model.overall_penalties]).reshape(-1,1))
    #print(ep)

    n_est = nR
    
    if n_c > 1: # Parallelize grid search
        # Generate next \\lambda values for which to compute REML, and Vb
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
        # Generate \\lambda values for which to compute REML, and Vb
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

def updateVp(ep:np.ndarray,ws:np.ndarray,rGrid:np.ndarray) -> np.ndarray:
    """Update covariance matrix of posterior for :math:`\\boldsymbol{\\rho} = log(\\boldsymbol{\\lambda})`. REML scores are used to
    approximate expectation, similar to what was suggested by Greven & Scheipl (2016).

    References:
     - https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices
     - Greven, S., & Scheipl, F. (2016). Comment on: Smoothing Parameter and Model Selection for General Smooth Models

    :param ep: Model estimate log(\\lambda), i.e., the expectation over rGrid
    :type ep: np.ndarray
    :param ws: weight associated with each log(\\lambda) value used for numerical integration
    :type ws: np.ndarray
    :param rGrid: A 2d array, holding all \\lambda samples considered so far. Each row is one sample
    :type rGrid: np.ndarray
    :return: An estimate of the covariance matrix of log(\\lambda) - 2d array of shape len(mp)*len(mp).
    :rtype: np.ndarray
    """

    wp = (np.log(rGrid[0]).reshape(-1,1) - ep)
    # Vp = E[(wp-ep)(wp-ep)^T] - see Wikipedia
    Vp = ws[0]*(wp @ wp.T)
    for ridx in range(1,rGrid.shape[0]):
        wp = (np.log(rGrid[ridx]).reshape(-1,1) - ep)
        Vp += ws[ridx]*(wp @ wp.T)
    
    return Vp

def _compute_VB_corr_terms_MP(family:Gaussian,address_y:str,address_dat:str,address_ptr:str,address_idx:str,address_datXX:str,address_ptrXX:str,address_idxXX:str,shape_y:tuple,shape_dat:tuple,shape_ptr:tuple,shape_datXX:tuple,shape_ptrXX:tuple,rows:int,cols:int,rPen:list[LambdaTerm],offset:float|np.ndarray,r:np.ndarray) -> tuple[scp.sparse.csc_array,np.ndarray,float,float,float,float]:
   """Multi-processing code for Grevel & Scheipl correction for Gaussian additive model - see ``correct_VB`` for details.

    :param family: Family of model
    :type family: Gaussian
    :param address_y: Memory address for y vector
    :type address_y: str
    :param address_dat: Memory address for X data
    :type address_dat: str
    :param address_ptr: Memory address for X pointers
    :type address_ptr: str
    :param address_idx: Memory address for X indices
    :type address_idx: str
    :param address_datXX: Memory address for X.T@X data
    :type address_datXX: str
    :param address_ptrXX: Memory address for X.T@X pointers
    :type address_ptrXX: str
    :param address_idxXX: Memory address for X.T@X indices
    :type address_idxXX: str
    :param shape_y: shape of y data array
    :type shape_y: tuple
    :param shape_dat: shape of X data array
    :type shape_dat: tuple
    :param shape_ptr: shape of X pointers array
    :type shape_ptr: tuple
    :param shape_datXX: shape of X.T@X data array
    :type shape_datXX: tuple
    :param shape_ptrXX: shape of X data array
    :type shape_ptrXX: tuple
    :param rows: Rows of X
    :type rows: int
    :param cols: Columns of X
    :type cols: int
    :param rPen: List of penalties
    :type rPen: list[LambdaTerm]
    :param offset: Any fixed offset to add to the linear predictor/mean
    :type offset: float | np.ndarray
    :param r: List of log(lambda) values for which to evaluate the reml score.
    :type r: np.ndarray
    :return: Un-pivoted inverse of pivoted Cholesky of negative penalized hessian,coefficients,reml score,scale etimate,total edf,llk
    :rtype: tuple[scp.sparse.csc_array,np.ndarray,float,float,float,float]
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
   _,_,edf,_,_,scale = update_scale_edf(y,None,eta,None,X.shape[0],X.shape[1],LP,None,Pr,None,None,family,rPen,None,None,1)

   llk = family.llk(y,eta,scale)

   # Now compute REML for candidate
   reml = REML(llk,XX/scale,coef,scale,rPen)
   coef = coef.reshape(-1,1)

   # Form VB, first solve LP{^-1}
   LPinv = compute_Linv(LP,1)
   Linv = apply_eigen_perm(Pr,LPinv)

   # Now collect what we need for the remaining terms
   return Linv,coef,reml,scale,edf,llk

def compute_Vp_WPS(Vbr:scp.sparse.csc_array,H:scp.sparse.csc_array,S_emb:scp.sparse.csc_array,penalties:list[LambdaTerm],coef:np.ndarray,scale:float=1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Computes the inverse of what is approximately the negative Hessian of the Laplace approximate REML criterion with respect to the log smoothing penalties.

    The derivatives computed are only exact for Gaussian additive models and canonical generalized additive models. For all other models they are in-exact in that they
    assume that the hessian of the log-likelihood does not depend on :math:`\\lambda` (or :math:`log(\\lambda)`), so they are essentially the PQL derivatives of Wood et al. (2017).
    The inverse computed here acts as an approximation to the covariance matrix of the log smoothing parameters.

    References:
     - Wood, S. N., (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models.
     - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
     - Wood, S. N., Li, Z., Shaddick, G., & Augustin, N. H. (2017). Generalized Additive Models for Gigadata: Modeling the U.K. Black Smoke Network Daily Data.

    :param Vbr: Transpose of root for the estimate for the (unscaled) covariance matrix of :math:`\\boldsymbol{\\beta} | y, \\boldsymbol{\\lambda}` - the coefficients estimated by the model.
    :type Vbr: scp.sparse.csc_array
    :param H: The Hessian of the log-likelihood
    :type H: scp.sparse.csc_array
    :param S_emb: The weighted penalty matrix.
    :type S_emb: scp.sparse.csc_array
    :param penalties: A list holding the Lambdaterms estimated for the model.
    :type penalties: [LambdaTerm]
    :param coef: An array holding the estimated regression coefficients. Has to be of shape (-1,1)
    :type coef: np.ndarray
    :param scale: Any scale parameter estimated as part of the model. Can be omitted for more generic models beyond GAMMs. Defaults to 1.
    :type scale: float
    :return: Generalized inverse of negative hessian of approximate REML criterion, regularized version of the former, root of generalized inverse, root of regularized generalized inverse, hessian of approximate REML criterion, np.array of shape ((len(coef),len(penalties))) containing in each row the partial derivative of the coefficients with respect to an individual lambda parameter
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    # Form nH - the negative hessian of the penalized llk - note, H is scaled by \\phi for GAMMs, so have to do the same for S_emb:
    nH = (-1*H) + S_emb/scale

    # Get partial derivatives of coef with respect to log(lambda) - see Wood (2017):
    dBetadRhos = np.zeros((len(coef),len(penalties)))

    # Vbr.T@Vbr = nH^{-1} - Vbr is available in model.lvi or model.lvi after fitting
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
            t4 = 0.5 * (penalties[peni].D_J_emb.T @ Vbr.T @ Vbr @ penalties[penj].D_J_emb).power(2).sum()*penalties[peni].lam*penalties[penj].lam

            # And first
            t5 = 0
            if gamma:
                t5 = 0.5 * (Vbr @ penalties[peni].D_J_emb).power(2).sum()*penalties[peni].lam
            
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


def compute_Vb_corr_WPS(Vbr:scp.sparse.csc_array,Vpr,Vr,H:scp.sparse.csc_array,S_emb:scp.sparse.csc_array,penalties:list[LambdaTerm],coef:np.ndarray,scale:float=1) -> tuple[np.ndarray,np.ndarray]:
    """Computes both correction terms for ``Vb`` or :math:`\\mathbf{V}_{\\boldsymbol{\\beta}}`, which is the co-variance matrix for the conditional posterior of :math:`\\boldsymbol{\\beta}` so that
    :math:`\\boldsymbol{\\beta} | y, \\boldsymbol{\\lambda} \\sim N(\\hat{\\boldsymbol{\\beta}},\\mathbf{V}_{\\boldsymbol{\\beta}})`, described by Wood, Pya, & Säfken (2016).

    References:
     - Wood, S. N., (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models.
     - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

    :param Vbr: Transpose of root for the estimate for the (unscaled) covariance matrix of :math:`\\boldsymbol{\\beta} | y, \\boldsymbol{\\lambda}` - the coefficients estimated by the model.
    :type Vbr: scp.sparse.csc_array
    :param Vpr: A (regularized) estimate of the covariance matrix of :math:`\\boldsymbol{\\rho}` - the log smoothing penalties.
    :type Vpr: np.ndarray
    :param Vr: Transpose of root of **un-regularized** covariance matrix of :math:`\\boldsymbol{\\rho}` - the log smoothing penalties.
    :type Vr: np.ndarray
    :param H: The Hessian of the log-likelihood
    :type H: scp.sparse.csc_array
    :param S_emb: The weighted penalty matrix.
    :type S_emb: scp.sparse.csc_array
    :param penalties: A list holding the Lambdaterms estimated for the model.
    :type penalties: [LambdaTerm]
    :param coef: An array holding the estimated regression coefficients. Has to be of shape (-1,1)
    :type coef: np.ndarray
    :param scale: Any scale parameter estimated as part of the model. Can be omitted for more generic models beyond GAMMs. Defaults to 1.
    :type scale: float
    :raises ArithmeticError: Will throw an error when the negative Hessian of the penalized likelihood is ill-scaled so that a Cholesky decomposition fails.
    :return: A tuple containing: ``Vc`` and ``Vcc``. ``Vbr.T@Vbr*scale`` + ``Vc`` + ``Vcc`` is then approximately the correction devised by WPS (2016).
    :rtype: tuple[np.ndarray, np.ndarray]
    """

    # Get (unscaled) negative Hessian of the penalized likelihood.
    # For a GAMM, this will thus be X.T@W@X + S_emb - since H = (X.T@W@X)\\phi
    nH = (-1*H)*scale + S_emb

    # We need un-pivoted transpose of cholesky of this, but can pre-condition
    Sdiag = np.power(np.abs(nH.diagonal()),0.5)
    PI = scp.sparse.diags(1/Sdiag,format='csc')
    P = scp.sparse.diags(Sdiag,format='csc')
    LP,code = cpp_chol(PI@nH@PI)
    R = (P@LP).T.toarray()
    #R.sort_indices()

    if code != 0:
        raise ArithmeticError("Failed to compute Cholesky of negative Hessian of penalized likelihood.")

    #print((R.T@R - nH).max())

    # Get partial derivatives of beta with respect to \rho, the log smoothing penalties
    dBetadRhos = np.zeros((len(coef),len(penalties)))

    # Vbr.T@Vbr = nH^{-1} - Vbr is in principle available in model.lvi or model.lvi after fitting,
    # but we need Rinv anyway..
    Rinv = scp.linalg.solve_triangular(R,np.identity(nH.shape[1]))                      
    V = Rinv @ Rinv.T

    for peni,pen in enumerate(penalties):
        # Given in Wood (2017)
        #print((-pen.lam * (Vbr.T @ (Vbr @ (pen.S_J_emb @ coef)))).shape)
        dBetadRhos[:,[peni]] = -pen.lam * V @ (pen.S_J_emb @ coef)
    
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

        A = pen.lam*pen.S_J_emb.toarray()
        dRdRho = dChol.dChol(R,A)

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
        dRdRhos.append(np.asfortranarray((Rinv@dRdRho)@Rinv)) # Must make sure this is Fortran array order, as expected by dChol.computeV2

    # Now final sum
    Vcc = dChol.computeV2(Vpr,dRdRhos,Vc.shape[1])

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

    def __init__(self, a=np.log(1e-7),b=np.log(1e7)) -> None:
        super().__init__(a=a,b=b)
    
    def logpdf(self, rho:np.ndarray) -> np.ndarray:
        """Returns an array holding zeroes for all log(lambda) parameters within ``self.a`` and ``self.b``, otherwise ``-np.inf``.

        :param rho: Array of log(lambda) parameters
        :type rho: np.ndarray
        :return: Log-density array as described above
        :rtype: np.ndarray
        """
        a = self.kwargs["a"]
        b = self.kwargs["b"]
        
        ld = np.zeros(rho.shape[0])
        ld[(np.min(rho,axis=1) < a) | (np.max(rho,axis=1) > b)] = -np.inf
        return ld


def correct_VB(model,nR:int = 250,grid_type:str = 'JJJ1',a:float=1e-7,b:float=1e7,df:int=40,n_c:int=10,form_t1:bool=False,verbose:bool=False,drop_NA:bool=True,method:str="Chol",only_expected_edf:bool=False,Vp_fidiff:bool=False,use_importance_weights:bool=True,prior:Callable|None=None,recompute_H:bool=False,seed:int|None=None,compute_Vcc:bool=True,**bfgs_options) -> tuple[scp.sparse.csc_array|None, scp.sparse.csc_array|None, np.ndarray|None ,np.ndarray|None, np.ndarray|None, float|None, np.ndarray|None, float|None, float, np.ndarray]:
    """Estimate :math:`\\tilde{\\mathbf{V}}`, the covariance matrix of the marginal posterior :math:`\\boldsymbol{\\beta} | y` to account for smoothness uncertainty.
    
    Wood et al. (2016) and Wood (2017) show that when basing conditional versions of model selection criteria or hypothesis
    tests on :math:`\\mathbf{V}`, which is the co-variance matrix for the normal approximation to the conditional posterior of :math:`\\boldsymbol{\\beta}` so that
    :math:`\\boldsymbol{\\beta} | y, \\boldsymbol{\\lambda} \\sim N(\\hat{\\boldsymbol{\\beta}},\\mathbf{V})`, the tests are severely biased. To correct for this they
    show that uncertainty in :math:`\\boldsymbol{\\lambda}` needs to be accounted for. Hence they suggest to base these tests on :math:`\\tilde{\\mathbf{V}}`, the covariance matrix
    of the normal approximation to the **marginal posterior** :math:`\\boldsymbol{\\beta} | y`. They show how to obtain an estimate of :math:`\\tilde{\\mathbf{V}}`,
    but this requires :math:`\\mathbf{V}^{\\boldsymbol{\\rho}}` - an estimate of the covariance matrix of the normal approximation to the posterior of :math:`\\boldsymbol{\\rho}=log(\\boldsymbol{\\lambda})`. Computing :math:`\\mathbf{V}^{\\boldsymbol{\\rho}}` requires derivatives that are not available
    when using the efs update.

    This function implements multiple strategies to approximately correct for smoothing parameter uncertainty, based on the proposals by  Wood et al. (2016) and Greven & Scheipl (2017). The most straightforward strategy
    (``grid_type = 'JJJ1'``) is to obtain a PQL or finite difference approximation for :math:`\\mathbf{V}^{\\boldsymbol{\\rho}}` and to then compute approximately the Wood et al. (2016) correction assuming that higher-order derivatives of the llk are zero (this will be exact
    for Gaussian additive or canonical Generalized models). This is too costly for large sparse multi-level models and not exact for more generic models. The MC based alternative available via ``grid_type = 'JJJ2'`` addresses the first problem (**Important**, set: ``use_importance_weights=False`` and ``only_expected_edf=True``.). The second MC based alternative
    available via ``grid_type = 'JJJ3'`` is most appropriate for more generic models (The ``prior`` argument can be used to specify any prior to be placed on :math:`\\boldsymbol{\\rho}` also you will need to set: ``use_importance_weights=True`` and ``only_expected_edf=False``).
    Both strategies use a PQL or finite difference approximation to :math:`\\mathbf{V}^{\\boldsymbol{\\rho}}` to obtain ``nR`` samples from the (normal approximation) to the posterior of :math:`\\boldsymbol{\\rho}`.
    From these samples mssm then estimates :math:`\\tilde{\\mathbf{V}}` as described in more detail by Krause et al. (in preparation).

    **Note:** If you set ``only_expected_edf=True``, only the last two output arguments will be non-zero.

    Example::

        # Simulate some data for a Gaussian model
        sim_fit_dat = sim3(n=500,scale=2,c=1,family=Gaussian(),seed=21)

        # Now fit nested models
        sim_fit_formula = Formula(lhs("y"),
                                    [i(),
                                     f(["x0"],nk=20),
                                     f(["x1"],nk=20),
                                     f(["x2"],nk=20),
                                     f(["x3"],nk=20)],
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
        # total_edf is sum of former (but subjected to upper bounds so might not be exactly the same)
        # ed2 is optionally smoothness bias corrected version of edf
        # total_edf2 is optionally bias corrected version of total_edf (subjected to upper bounds)
        # expected_edf is None here but for MC strategies (i.e., ``grid!=1``) will be an estimate
        # of total_edf (**without being subjected to upper bounds**) that does not require forming
        # V (only computed when ``only_expected_edf=True``). 
        # mean_coef is None here but for MC strategies will be an estimate of the mean of the
        # marginal posterior of coefficients, only computed when setting ``recompute_H=True``

        V,LV,Vp,Vpr,edf,total_edf,edf2,total_edf2,expected_edf,mean_coef = correct_VB(model,
                                                                                      grid_type="JJJ1",
                                                                                      verbose=True,
                                                                                      seed=20)

        # Compute MC estimate for generic model and given prior
        prior = DummyRhoPrior(b=np.log(1e12)) # Set up uniform prior
        V_MC,LV_MC,Vp_MC,Vpr_MC,edf_MC,\
        total_edf_MC,edf2_MC,total_edf2_MC,expected_edf_MC,mean_coef_MC = correct_VB(model2,
                                                                                     grid_type="JJJ3",
                                                                                     verbose=True,
                                                                                     seed=20,
                                                                                     df=10,
                                                                                     prior=prior,
                                                                                     recompute_H=True)

    References:
     - Wood, S. N., (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models.
     - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models
     - Greven, S., & Scheipl, F. (2016). Comment on: Smoothing Parameter and Model Selection for General Smooth Models
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

    :param model: GAMM, GAMMLSS, or GSMM model (which has been fitted) for which to estimate :math:`\\mathbf{V}`
    :type model: mssm.models.GSMM | mssm.models.GAMMLSS | mssm.models.GAMM 
    :param nR: In case ``grid!="JJJ1"``, ``nR`` samples/reml scores are generated/computed to numerically evaluate the expectations necessary for the uncertainty correction, defaults to 250
    :type nR: int, optional
    :param grid_type: How to compute the smoothness uncertainty correction - see above for details, defaults to 'JJJ1'
    :type grid_type: str, optional
    :param a: Any of the :math:`\\lambda` estimates obtained from ``model`` (used to define the mean for the posterior of :math:`\\boldsymbol{\\rho}|y \\sim N(log(\\hat{\\boldsymbol{\\rho}}),\\mathbf{V}^{\\boldsymbol{\\rho}})` used to sample ``nR`` candidates) which are smaller than this are set to this value as well, defaults to 1e-7 the minimum possible estimate
    :type a: float, optional
    :param b: Any of the :math:`\\lambda` estimates obtained from ``model`` (used to define the mean for the posterior of :math:`\\boldsymbol{\\rho}|y \\sim N(log(\\hat{\\boldsymbol{\\rho}}),\\mathbf{V}^{\\boldsymbol{\\rho}})` used to sample ``nR`` candidates) which are larger than this are set to this value as well, defaults to 1e7 the maximum possible estimate
    :type b: float, optional
    :param df: Degrees of freedom used for the multivariate t distribution used to sample the next set of candidates. Setting this to ``np.inf`` means a multivariate normal is used for sampling, defaults to 40
    :type df: int, optional
    :param n_c: Number of cores to use during parallel parts of the correction, defaults to 10
    :type n_c: int, optional
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
    :param Vp_fidiff: Whether to rely on a finite difference approximation to compute :math:`\\mathbf{V}^{\\boldsymbol{\\rho}}` or on a PQL approximation. The latter is exact for Gaussian and canonical GAMs and far cheaper if many penalties are to be estimated. Defaults to False (PQL approximation)
    :type Vp_fidiff: bool,optional
    :param use_importance_weights: Whether to rely importance weights to compute the numerical integration when ``grid_type != 'JJJ1'`` or on the log-densities of :math:`\\mathbf{V}^{\\boldsymbol{\\rho}}` - the latter assumes that the unconditional posterior is normal. Defaults to True (Importance weights are used)
    :type use_importance_weights: bool,optional
    :param prior: An (optional) instance of an arbitrary class that has a ``.logpdf()`` method to compute the prior log density of a sampled candidate. If this is set to ``None``, the prior is assumed to coincide with the proposal distribution, simplifying the importance weight computation. Ignored when ``use_importance_weights=False``. Defaults to None
    :type prior: Callable|None, optional
    :param recompute_H: Whether or not to re-compute the Hessian of the log-likelihood at an estimate of the mean of the Bayesian posterior :math:`\\boldsymbol{\\beta}|y` before computing the (uncertainty/bias corrected) edf. Defaults to False
    :type recompute_H: bool, optional
    :param compute_Vcc: Whether to compute the second correction term when `strategy='JJJ1'` (or when computing the lower-bound for the remaining strategies) or only the first one. In contrast to the second one, the first correction term is substantially cheaper to compute - so setting this to False for larger models will speed up the correction considerably. Defaults to True
    :type compute_Vcc: bool, optional
    :param seed: Seed to use for random parts of the correction. Defaults to None
    :type seed: int|None,optional
    :param bfgs_options: Any additional keyword arguments that should be passed on to the call of :func:`scipy.optimize.minimize`. If none are provided, the ``gtol`` argument will be initialized to 1e-3. Note also, that in any case the ``maxiter`` argument is automatically set to 100. Defaults to None.
    :type bfgs_options: key=value,optional
    :return: A tuple containing: ``V`` - an estimate of the unconditional covariance matrix, ``LV`` - the Cholesky of the former, ``Vp`` - an estimate of the covariance matrix for :math:`\\boldsymbol{\\rho}`, ``Vpr`` - a regularized version of the former, ``edf`` - smoothness uncertainty corrected coefficient-wise edf, ``total_edf`` - smoothness uncertainty corrected total (i.e., model) edf, ``edf2`` - smoothness uncertainty + smoothness bias corrected coefficient-wise edf, ``total_edf2`` - smoothness uncertainty + smoothness bias corrected total (i.e., model) edf, ``expected_edf`` - an optional estimate of total_edf that does not require forming ``V``, ``mean_coef`` - an optional estimate of the mean of the posterior of the coefficients
    :rtype: tuple[scp.sparse.csc_array|None, scp.sparse.csc_array|None, np.ndarray|None ,np.ndarray|None, np.ndarray|None, float|None, np.ndarray|None, float|None, float, np.ndarray]
    """
    np_gen = np.random.default_rng(seed)

    family = model.family

    if not grid_type in ["JJJ1","JJJ2","JJJ3"]:
        raise ValueError("'grid_type' has to be set to one of 'JJJ1', 'JJJ2', or 'JJJ3'.")
    
    if isinstance(family,Family) and model.rho is not None and grid_type != "JJJ1":
        raise ValueError("For models with an ar1 model only grid_type='JJJ1' is supported.")
    
    if compute_Vcc == False and Vp_fidiff:
        warnings.warn("Ignoring request to rely on finite differencing to obtain covariance matrix of $\\rho$, because `compute_Vcc` was set to False.")
        Vp_fidiff = False

    if isinstance(family,GSMMFamily):
        if not bfgs_options:
            bfgs_options = {"ftol":1e-9,
                            "maxcor":30,
                            "maxls":100}

    nPen = len(model.overall_penalties)
    rPen = copy.deepcopy(model.overall_penalties)
    S_emb,_,_,_ = compute_S_emb_pinv_det(model.hessian.shape[1],model.overall_penalties,"svd")
    
    if isinstance(family,Family):
        y = model.formulas[0].y_flat[model.formulas[0].NOT_NA_flat]

        if not model.formulas[0].get_lhs().f is None:
            # Optionally apply function to dep. var. before fitting.
            y = model.formulas[0].get_lhs().f(y)

        X = model.get_mmat()

        orig_scale = family.scale
        if family.twopar:
            _,orig_scale = model.get_pars()
    else:
        if isinstance(family,GAMLSSFamily):
            y = model.formulas[0].y_flat[model.formulas[0].NOT_NA_flat]

            if not model.formulas[0].get_lhs().f is None:
                # Optionally apply function to dep. var. before fitting. Not sure why that would be desirable for this model class...
                y = model.formulas[0].get_lhs().f(y)
            
            Xs = model.get_mmat()

        else: # Need all y vectors in y, i.e., y is actually ys
            ys = []
            for fi,form in enumerate(model.formulas):
                
                # Repeated y-variable - don't have to pass all of them
                if fi > 0 and form.get_lhs().variable == model.formulas[0].get_lhs().variable:
                    ys.append(None)
                    continue

                # New y-variable
                if drop_NA:
                    y = form.y_flat[form.NOT_NA_flat]
                else:
                    y = form.y_flat

                # Optionally apply function to dep. var. before fitting. Not sure why that would be desirable for this model class...
                if not form.get_lhs().f is None:
                    y = form.get_lhs().f(y)
                
                # And collect
                ys.append(y)
            
            y = ys
            Xs = model.get_mmat(drop_NA=drop_NA)

        X = Xs[0]
        orig_scale = 1
        init_coef = copy.deepcopy(model.coef)

    Vp = None
    Vpr = None
    if grid_type == "JJJ1" or grid_type == "JJJ2" or grid_type == "JJJ3":
        # Approximate Vp via finitie differencing
        if Vp_fidiff:
            Vp, Vpr, Vr, Vrr, _ = estimateVp(model,n_c=n_c,grid_type="JJJ1",Vp_fidiff=True)
        else:
            # Take PQL approximation instead
            Vp, Vpr, Vr, Vrr, _, dBetadRhos = compute_Vp_WPS(model.lvi,
                                                 model.hessian,
                                                 S_emb,
                                                 model.overall_penalties,
                                                 model.coef,
                                                 scale=orig_scale if isinstance(family,Family) else 1)

        # Compute approximate WPS (2016) correction
        if grid_type == "JJJ1" or (grid_type == "JJJ3" and only_expected_edf == False):

            if compute_Vcc:
                if isinstance(family,Family):
                    Vc,Vcc = compute_Vb_corr_WPS(model.lvi,Vpr,Vr,model.hessian,S_emb,model.overall_penalties,model.coef,scale=orig_scale)
                else:
                    Vc,Vcc = compute_Vb_corr_WPS(model.lvi,Vpr,Vr,model.hessian,S_emb,model.overall_penalties,model.coef)
            else:
                # Only compute first correction term
                Vc = Vr @ dBetadRhos.T
                Vc = Vc.T @ Vc
                Vcc = 0
                
            if isinstance(family,Family):
                V = Vc + Vcc + ((model.lvi.T@model.lvi)*orig_scale)
            else:
                V = Vc + Vcc + model.lvi.T@model.lvi
            
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
        # Generate \\lambda values from Vpr for which to compute REML, and Vb
        ep = np.log(np.array([min(b,max(a,pen.lam)) for pen in model.overall_penalties]).reshape(-1,1))
        #print(ep)

        n_est = nR
        
        if n_c > 1: # Parallelize grid search
            # Generate next \\lambda values for which to compute REML, and Vb
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
            # Generate \\lambda values for which to compute REML, and Vb
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

                S_emb,_,S_root,_ = compute_S_emb_pinv_det(X.shape[1],model.overall_penalties,"svd",method != 'Chol')

                if isinstance(family,Gaussian) and isinstance(family.link,Identity): # strictly additive case
                    nH = (-1*model.hessian)*orig_scale

                else: # Generalized case
                    eta = (X @ mean_coef).reshape(-1,1) + model.offset
                    mu = family.link.fi(eta)
                    
                    # Compute pseudo-dat and weights for mean coef
                    yb,Xb,z,Wr = update_PIRLS(y,yb,mu,eta-model.offset,X,Xb,family,None)

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
                _,_,_,_,_,scale = update_scale_edf(y,z,eta,Wr,X.shape[0],X.shape[1],LP,None,Pr,None,None,family,model.overall_penalties,keep,drop,n_c)
                
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
                print(f"Recomputed negative Hessian. 2 Norm of coef. difference: {np.linalg.norm(mean_coef-model.coef)}. F. Norm of n. Hessian difference: {scp.sparse.linalg.norm(nH + model.hessian)}")

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
            
            if n_c > 1:
                # E_{p|y}[V_\boldsymbol{\beta}(\\lambda)] + E_{p|y}[\boldsymbol{\beta}\boldsymbol{\beta}^T] in Greven & Scheipl (2017)
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
                    Vr1 = model.lvi.T@model.lvi

            V = Vr1 + Vcr.T @ Vcr

        elif grid_type == 'JJJ3':

            # Now compute \\hat{cov(\boldsymbol{\beta}|y)}
            if n_c > 1:
                # E_{p|y}[V_\boldsymbol{\beta}(\\lambda)] + E_{p|y}[\boldsymbol{\beta}\boldsymbol{\beta}^T] in Greven & Scheipl (2017)
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
            #        Vr1 = model.lvi.T@model.lvi

            # Now, Greven & Scheipl provide final estimate =
            # E_{p|y}[V_\boldsymbol{\beta}(\\lambda)] + E_{p|y}[\boldsymbol{\beta}\boldsymbol{\beta}^T] - E_{p|y}[\boldsymbol{\beta}] E_{p|y}[\boldsymbol{\beta}]^T
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
    F = V@(nH)

    edf = F.diagonal()
    total_edf = F.trace()

    # In mgcv, an upper limit is enforced on edf and total_edf when they are uncertainty corrected - based on t1 in section 6.1.2 of Wood (2017)
    # so the same is done here.
    if isinstance(family,Family):
        ucF = model.lvi.T@model.lvi@((-1*model.hessian)*orig_scale)
    else: # GSMM/GAMLSS case
        ucF = (model.lvi.T@model.lvi)@(-1*model.hessian)

    # Compute upper bound
    ucFFd = ucF.multiply(ucF.T).sum(axis=0)
    total_edf2 = 2*model.edf - np.sum(ucFFd)
    edf2 = 2*ucF.diagonal() - ucFFd

    # Enforce upper bound
    if total_edf > total_edf2:
        total_edf = total_edf2
        edf = edf2

    # Compute uncertainty corrected smoothness bias corrected edf (t1 in section 6.1.2 of Wood, 2017)
    if form_t1:
        total_edf2 += (total_edf - model.edf)
        edf2 += (edf - ucF.diagonal())

    if verbose and grid_type != "JJJ1":
        print(f"Correction was based on {rGrid.shape[0]} samples in total.")    

    return V,LV,Vp,Vpr,edf,total_edf,edf2,total_edf2,expected_edf,mean_coef