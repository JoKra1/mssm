import numpy as np
import scipy as scp
import copy
from collections.abc import Callable
from .src.python.formula import Formula,build_sparse_matrix_from_formula,lhs,pd,warnings,build_penalties
from .src.python.exp_fam import Link,Logit,Identity,LOG,LOGb,Family,Binomial,Gaussian,GAMLSSFamily,GAUMLSS,Gamma,InvGauss,MULNOMLSS,GAMMALS,GSMMFamily,PropHaz,Poisson
from .src.python.gamm_solvers import solve_gamm_sparse,mp,repeat,tqdm,cpp_cholP,apply_eigen_perm,compute_Linv,solve_gamm_sparse2,solve_gammlss_sparse,solve_generalSmooth_sparse
from .src.python.terms import TermType,GammTerm,i,f,fs,irf,l,li,ri,rs
from .src.python.penalties import embed_shared_penalties,IdentityPenalty,DifferencePenalty
from .src.python.utils import sample_MVN,REML,adjust_CI,print_smooth_terms,print_parametric_terms,approx_smooth_p_values,compute_bias_corrected_edf,GAMLSSGSMMFamily,computeAr1Chol
from .src.python.custom_types import VarType,ConstType,Constraint,PenType,LambdaTerm,Fit_info

##################################### GSMM class #####################################

class GSMM():
    """
    Class to fit General Smooth/Mixed Models (see Wood, Pya, & Säfken; 2016). Estimation is possible via exact Newton method for coefficients of via L-qEFS update (see Krause et al., (submitted) and example below).

    Examples::

        from mssm.models import *
        from mssmViz.sim import *
        from mssmViz.plot import *
        import matplotlib.pyplot as plt

        class NUMDIFFGENSMOOTHFamily(GSMMFamily):
            # Implementation of the ``GSMMFamily`` class that uses finite differencing to obtain the
            # gradient of the likelihood to estimate a Gaussian GAMLSS via the general smooth code and
            # the L-qEFS update by Krause et al. (in preparation).

            # References:
            #    - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
            #    - Nocedal & Wright (2006). Numerical Optimization. Springer New York.
            

            def __init__(self, pars: int, links:[Link]) -> None:
                super().__init__(pars, links)
            
            def llk(self, coef, coef_split_idx, ys, Xs):
                # Likelihood for a Gaussian GAM(LSS) - implemented so
                # that the model can be estimated using the general smooth code.
                y = ys[0]
                split_coef = np.split(coef,coef_split_idx)
                eta_mu = Xs[0]@split_coef[0]
                eta_sd = Xs[1]@split_coef[1]
                
                mu_mu = self.links[0].fi(eta_mu)
                mu_sd = self.links[1].fi(eta_sd)

                family = GAUMLSS(self.links)
                llk = family.llk(y,mu_mu,mu_sd)
                return llk

        # Simulate 500 data points
        sim_dat = sim3(500,2,c=1,seed=0,family=Gaussian(),binom_offset = 0, correlate=False)

        # We need to model the mean: \\mu_i
        formula_m = Formula(lhs("y"),
                            [i(),f(["x0"]),f(["x1"]),f(["x2"]),f(["x3"])],
                            data=sim_dat)

        # And for sd - here constant
        formula_sd = Formula(lhs("y"),
                            [i()],
                            data=sim_dat)

        # Collect both formulas
        formulas = [formula_m,formula_sd]
        links = [Identity(),LOGb(-0.001)]

        # Now define the general family + model and fit!
        gsmm_fam = NUMDIFFGENSMOOTHFamily(2,links)
        model = GSMM(formulas=formulas,family=gsmm_fam)

        # Fit with SR1
        bfgs_opt={"gtol":1e-9,
                "ftol":1e-9,
                "maxcor":30,
                "maxls":200,
                "maxfun":1e7}
                        
        model.fit(method='qEFS',bfgs_options=bfgs_opt)

        # Extract all coef
        coef = model.coef

        # Now split them to get separate lists per parameter of the log-likelihood (here mean and scale)
        # split_coef[0] then holds the coef associated with the first parameter (here the mean) and so on
        split_coef = np.split(coef,model.coef_split_idx)


    References:
     - Wood, S. N., & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. https://doi.org/10.1111/biom.12666
     - Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models: Estimation of Semiparametric Generalized Linear Models. https://doi.org/10.1111/j.1467-9868.2010.00749.x
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
     - Nocedal & Wright (2006). Numerical Optimization. Springer New York.
     - Krause et al. (submitted). The Mixed-Sparse-Smooth-Model Toolbox (MSSM): Efficient Estimation and Selection of Large Multi-Level Statistical Models. https://doi.org/10.48550/arXiv.2506.13132
    
    :param formulas: A list of formulas, one per parameter of the likelihood that is to be modeled as a smooth model
    :type formulas: [Formula]
    :param family: A GSMMFamily family.
    :type family: GSMMFamily
    :ivar [Formula] formulas: The list of formulas passed to the constructor.
    :ivar scp.sparse.csc_array | None lvi: The inverse of the Cholesky factor of the conditional model coefficient covariance matrix - or None, in case the ``L-BFGS-B`` optimizer was used and ``form_VH`` was set to False when calling ``model.fit()``. Initialized with ``None``.
    :ivar scp.sparse.linalg.LinearOperator lvi_linop: A :class:`scipy.sparse.linalg.LinearOperator` of the conditional model coefficient covariance matrix (**not the root**) - or None. Only available in case the ``L-BFGS-B`` optimizer was used and ``form_VH`` was set to False when calling ``model.fit()``.
    :ivar np.ndarray coef:  Contains all coefficients estimated for the model. Shape of the array is (-1,1). Initialized with ``None``.
    :ivar [[float]] preds: The linear predictors for every parameter of ``family`` evaluated for each observation in the training data (after removing NaNs). Initialized with ``None``.
    :ivar [[float]] mus: The predicted means for every parameter of ``family`` evaluated for each observation in the training data (after removing NaNs). Initialized with ``None``.
    :ivar scp.sparse.csc_array hessian:  Estimated hessian of the log-likelihood (will correspond to ``hessian - diag*eps`` if ``self.info.eps > 0`` after fitting). Initialized with ``None``.
    :ivar float edf: The model estimated degrees of freedom as a float. Initialized with ``None``.
    :ivar float edf1: The model estimated degrees of freedom as a float corrected for smoothness bias. Set by the :func:`approx_smooth_p_values` function, the first time it is called. Initialized with ``None``.
    :ivar [float] term_edf: The estimated degrees of freedom per smooth term. Initialized with ``None``.
    :ivar [float] term_edf1: The estimated degrees of freedom per smooth term corrected for smoothness bias. Set by the :func:`approx_smooth_p_values` function, the first time it is called. Initialized with ``None``.
    :ivar float penalty: The total penalty applied to the model deviance after fitting as a float. Initialized with ``None``.
    :ivar [int] coef_split_idx: The index at which to split the overall coefficient vector into separate lists - one per parameter of ``family``. See the examples. Initialized after fitting!
    :ivar [LambdaTerm] overall_penalties:  Contains all penalties estimated for the model. Initialized with ``None``.
    :ivar Fit_info info: A :class:`Fit_info` instance, with information about convergence (speed) of the model.
    """

    def __init__(self, formulas: list[Formula], family: GSMMFamily):
        
        self.family = family
        self.formulas:list[Formula] = formulas
        self.lvi:scp.sparse.csc_array = None
        self.lvi_linop:scp.sparse.linalg.LinearOperator = None
        self.coef:np.ndarray = None
        self.preds:list[np.ndarray] = None # Linear predictors
        self.mus:list[np.ndarray] = None # Estimated parameters of log-likelihood
        self.hessian:scp.sparse.csc_array = None
        self.scale = 1

        self.edf:float = None
        self.edf1:float = None
        self.term_edf:list[float] = None
        self.term_edf1:list[float] = None

        self.penalty = 0
        self.overall_penalties:list[LambdaTerm] = None
        self.info:Fit_info = None
    
    ##################################### Getters #####################################
    
    def get_pars(self) -> np.ndarray:
        """
        Returns a list containing all coefficients estimated for the model. Use ``self.coef_split_idx`` to split the vector into separate subsets per parameter of the log-likelihood.

        Will return None if called before fitting was completed.
        
        :return: Model coefficients - before splitting!
        :rtype: [float] or None
        """
        return self.coef
    
    def get_mmat(self,use_terms:list[int]|None=None,drop_NA:bool=True,par:int|None=None) -> list[scp.sparse.csc_array] | scp.sparse.csc_array:
        """
        By default, returns a list containing exactly the model matrices used for fitting as a ``scipy.sparse.csc_array``. Will raise an error when fitting was not completed before calling this function.

        Optionally, the model matrix associated with a specific parameter of the log-likelihood can be obtained by setting ``par`` to the desired index, instead of ``None``.
        Additionally, all columns not corresponding to terms for which the indices are provided via ``use_terms`` are zeroed in case ``use_terms is not None``.

        :param use_terms: Optionally provide indices of terms in the formual that should be created. If this argument is provided columns corresponding to any term not included in this list will be zeroed, defaults to None
        :type use_terms: [int], optional
        :param drop_NA: Whether rows in the model matrix corresponding to NAs in the dependent variable vector should be dropped, defaults to True
        :type drop_NA: bool, optional
        :param par: The index corresponding to the parameter of the log-likelihood for which to obtain the model matrix. Setting this to ``None`` means all matrices are returned in a list, defaults to None.
        :type par: int or None, optional
        :raises ValueError: Will throw an error when called before the model was fitted/before model penalties were formed.
        :return: Model matrices :math:`\\mathbf{X}` used for fitting - one per parameter of ``self.family`` or a single model matrix for a specific parameter.
        :rtype: [scp.sparse.csc_array] or scp.sparse.csc_array
        """

        iterator = [par] if par is not None else [fidx for fidx in range(len(self.formulas))]
        mmat = []

        for fidx in iterator:
            form = self.formulas[fidx]

            if form.built_penalties == False:
                raise ValueError("Model matrix cannot be returned if penalties have not been initialized. Call model.fit() first.")

            terms = form.terms
            has_intercept = form.has_intercept
            ltx = form.get_linear_term_idx()
            irstx = form.get_ir_smooth_term_idx()
            stx = form.get_smooth_term_idx()
            rtx = form.get_random_term_idx()
            var_types = form.get_var_types()
            var_map = form.get_var_map()
            var_mins = form.get_var_mins()
            var_maxs = form.get_var_maxs()
            factor_levels = form.get_factor_levels()

            if drop_NA:
                cov_flat = form.cov_flat[form.NOT_NA_flat]
            else:
                cov_flat = form.cov_flat

            if len(irstx) > 0:
                cov_flat = form.cov_flat # Need to drop NA rows **after** building!
                cov = form.cov
            else:
                cov = None

            # Build the model matrix with all information from the formula
            model_mat = build_sparse_matrix_from_formula(terms,has_intercept,
                                                         ltx,irstx,stx,rtx,var_types,var_map,
                                                         var_mins,var_maxs,factor_levels,
                                                         cov_flat,cov,use_only=use_terms)
            
            if len(irstx) > 0 and drop_NA:
                model_mat = model_mat[form.NOT_NA_flat,:]
            
            mmat.append(model_mat)
        
        # Return desired matrix / list of matrices
        if par is not None:
            return mmat[0]
        else:
            return mmat
    
    def get_llk(self,penalized:bool=True,drop_NA:bool=True) -> float | None:
        """
        Get the (penalized) log-likelihood of the estimated model (float or None) given the trainings data.
        
        Will instead return ``None`` if called before fitting.
        
        :param penalized: Whether the penalized log-likelihood should be returned or the regular log-likelihood, defaults to True
        :type penalized: bool, optional
        :param drop_NA: Whether rows in the model matrices corresponding to NAs in the dependent variable vector should be dropped, defaults to True
        :type drop_NA: bool, optional
        :return: llk score
        :rtype: float or None
        """

        pen = 0
        if penalized:
            pen = 0.5*self.penalty
        if self.coef is not None:

            ys = []
            for fi,form in enumerate(self.formulas):
                
                # Repeated y-variable - don't have to pass all of them
                if fi > 0 and form.get_lhs().variable == self.formulas[0].get_lhs().variable:
                    ys.append(None)
                    continue

                # New y-variable
                if drop_NA:
                    y = form.y_flat[form.NOT_NA_flat]
                else:
                    y = form.y_flat

                # Optionally apply function to dep. var.
                if not form.get_lhs().f is None:
                    y = form.get_lhs().f(y)
                
                # And collect
                ys.append(y)

            # Build model matrices for all formulas
            Xs = self.get_mmat(drop_NA=drop_NA)

            return self.family.llk(self.coef,self.coef_split_idx,ys,Xs) - pen

        return None

    def get_reml(self,drop_NA:bool=True) -> float:
        """
        Get's the Laplcae approximate REML (Restrcited Maximum Likelihood) score for the estimated lambda values (see Wood, 2011).

        References:

         - Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models: Estimation of Semiparametric Generalized Linear Models.
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.

        :param drop_NA: Whether rows in the model matrices corresponding to NAs in the dependent variable vector should be dropped when computing the log-likelihood, defaults to True
        :type drop_NA: bool, optional
        :raises ValueError: Will throw an error when called before the model was fitted/before model penalties were formed.
        :return: REML score
        :rtype: float
        """

        if self.coef is None or self.hessian is None:
            raise ValueError("Model needs to be estimated before evaluating the REML score. Call model.fit()")
        
        llk = self.get_llk(False,drop_NA=drop_NA)

        keep = None
        if self.info.dropped is not None:
            keep = [cidx for cidx in range(self.hessian.shape[1]) if cidx not in self.info.dropped]
        
        reml = REML(llk,-1*self.hessian,self.coef,1,self.overall_penalties,keep)[0,0]
        return reml
    
    def get_resid(self,drop_NA:bool=True,**kwargs) -> np.ndarray:
        """The computation of the residual vector will differ between different :class:`GSMM` models and is thus implemented
        as a method by each :class:`GSMMFamily` family. These should be consulted to get more details. In general, if the model is specified correctly,
        the returned vector should approximately look like what could be expected from taking independent samples from :math:`N(0,1)`.

        Additional arguments required by the specific :func:`GSMMFamily.get_resid` method can be passed along via ``kwargs``.

        **Note**: Families for which no residuals are available can return None.

        References:
            - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, Scale and Shape.
            - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
        
        :param drop_NA: Whether rows in the model matrices corresponding to NAs in the dependent variable vector should be dropped from the model matrices, defaults to True
        :type drop_NA: bool, optional
        :raises ValueError: An error is raised in case the residuals are requested before the model has been fit.
        :return: vector of standardized residuals of shape (-1,1). **Note**, the first axis will not necessarily match the dimension of any of the response vectors (this will depend on the specific Family's implementation).
        :rtype: np.ndarray
        """

        if self.coef is None is None:
            raise ValueError("Model needs to be estimated before evaluating the residuals. Call model.fit()")
        
        # Get observation vectors
        ys = []
        for fi,form in enumerate(self.formulas):
            
            # Repeated y-variable - don't have to pass all of them
            if fi > 0 and form.get_lhs().variable == self.formulas[0].get_lhs().variable:
                ys.append(None)
                continue

            # New y-variable
            if drop_NA:
                y = form.y_flat[form.NOT_NA_flat]
            else:
                y = form.y_flat

            # Optionally apply function to dep. var.
            if not form.get_lhs().f is None:
                y = form.get_lhs().f(y)
            
            # And collect
            ys.append(y)

        # Get model matrices
        Xs = self.get_mmat(drop_NA=drop_NA)

        return self.family.get_resid(self.coef,self.coef_split_idx,ys,Xs,**kwargs)
    
    ##################################### Summary #####################################
    
    def print_parametric_terms(self):
        """Prints summary output for linear/parametric terms in the model, separately for each parameter of the family's distribution.
        
        For each coefficient, the named identifier and estimated value are returned. In addition, for each coefficient a p-value is returned, testing
        the null-hypothesis that the corresponding coefficient :math:`\\beta=0`. Under the assumption that this is true, the Null distribution follows
        approximately a standardized normal distribution. The corresponding z-statistic and the p-value are printed.
        See Wood (2017) section 6.12 and 1.3.3 for more details.

        Note that, un-penalized coefficients that are part of a smooth function are not covered by this function.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

        :raises NotImplementedError: Will throw an error when called for a model for which the model matrix was never former completely.
        """

        for formi, _ in enumerate(self.formulas):
            print(f"\nDistribution parameter: {formi + 1}\n")
            print_parametric_terms(self,par=formi)
    
    def print_smooth_terms(self, pen_cutoff:float=0.2, p_values:bool=False, edf1:bool=True):
        """Prints the name of the smooth terms included in the model. After fitting, the estimated degrees of freedom per term are printed as well.
        Smooth terms with edf. < ``pen_cutoff`` will be highlighted. This only makes sense when extra Kernel penalties are placed on smooth terms to enable
        penalizing them to a constant zero. In that case edf. < ``pen_cutoff`` can then be taken as evidence that the smooth has all but notationally disappeared
        from the model, i.e., it does not contribute meaningfully to the model fit. This can be used as an alternative form of model selection - see Marra & Wood (2011).

        References:

         - Marra & Wood (2011). Practical variable selection for generalized additive models.

        :param pen_cutoff: At which edf. cut-off smooth terms should be marked as "effectively removed", defaults to None
        :type pen_cutoff: float, optional
        :param p_values: Whether approximate p-values should be printed for the smooth terms, defaults to False
        :type p_values: bool, optional
        :param edf1: Whether or not the estimated degrees of freedom should be corrected for smoothnes bias. Doing so results in more accurate p-values but can be expensive for large models for which the difference is anyway likely to be marginal, defaults to False
        :type edf1: bool, optional
        """
        ps = None
        Trs = None
        for formi, _ in enumerate(self.formulas):
            print(f"\nDistribution parameter: {formi + 1}\n")
            
            if p_values:
                ps, Trs = approx_smooth_p_values(self,par=formi,edf1=edf1)

            print_smooth_terms(self,par=formi,pen_cutoff=pen_cutoff,ps=ps,Trs=Trs)
    
    ##################################### Fitting #####################################
    
    def fit(self,init_coef:np.ndarray|None=None,max_outer:int=200,max_inner:int=500,min_inner:int|None=None,
            conv_tol:float=1e-7,extend_lambda:bool=False,extension_method_lam:str="nesterov2",
            control_lambda:int|None=None,restart:bool=False,optimizer:str="Newton",method:str="QR/Chol",
            check_cond:int=1,piv_tol:float=np.power(np.finfo(float).eps,0.04),progress_bar:bool=True,
            n_cores:int=10,seed:int=0,drop_NA:bool=True,init_lambda:list[float]|None=None,form_VH:bool=True,
            use_grad:bool=False,build_mat:list[bool]|None=None,should_keep_drop:bool=True,gamma:float=1,
            qEFSH:str='SR1',overwrite_coef:bool=True,max_restarts:int=0,qEFS_init_converge:bool=False,
            prefit_grad:bool=True,repara:bool=None,init_bfgs_options:dict|None=None,
            bfgs_options:dict|None=None):
        """
        Fit the specified model.

        **Note**: Keyword arguments are initialized to maximise stability. For faster configurations (necessary for larger models) see examples below.
        
        :param init_coef: An initial estimate for the coefficients. Must be a numpy array of shape (-1,1). Defaults to None.
        :type init_coef: np.ndarray,optional
        :param max_outer: The maximum number of fitting iterations.
        :type max_outer: int,optional
        :param max_inner: The maximum number of fitting iterations to use by the inner Newton step for coefficients.
        :type max_inner: int,optional
        :param min_inner: The minimum number of fitting iterations to use by the inner Newton step for coefficients. By default set to ``max_inner``.
        :type min_inner: int,optional
        :param conv_tol: The relative (change in penalized deviance is compared against ``conv_tol`` * previous penalized deviance) criterion used to determine convergence.
        :type conv_tol: float,optional
        :param extend_lambda: Whether lambda proposals should be accelerated or not. Can lower the number of new smoothing penalty proposals necessary for models with heavily penalized functions. Disabled by default.
        :type extend_lambda: bool,optional
        :param extension_method_lam: **Experimental - do not change!** Which method to use to extend lambda proposals. Set to 'nesterov2' by default.
        :type extension_method_lam: str,optional
        :param control_lambda: Whether lambda proposals should be checked (and if necessary decreased) for whether or not they (approxiately) increase the Laplace approximate restricted maximum likelihood of the model. For ``method != 'qEFS'`` the following options are available: setting this to 0 disables control. Setting it to 1 means the step will never be smaller than the original EFS update but extensions will be removed in case the objective was exceeded (only has an effect when setting ``extend_lambda=True``). Setting it to 2 means that steps will generally be halved when they fail to increase the aproximate REML criterion. For ``method=='qEFS'`` the following options are available: setting this to 0 disables control. Setting it to 1 means the check described by Krause et al. (submitted) will be performed to control updates to lambda. Setting it to 2 means that steps will generally be halved when they fail to increase the aproximate REML criterion (note, that the gradient is based on quasi-newton approximations as well and thus less accurate). Setting it to 3 means both checks (i.e., 1 and 2) are performed. Set to 2 by default if ``method != 'qEFS'`` and otherwise to 1.
        :type control_lambda: int,optional
        :param restart: Whether fitting should be resumed. Only possible if the same model has previously completed at least one fitting iteration.
        :type restart: bool,optional
        :param optimizer: Deprecated. Defaults to "Newton"
        :type optimizer: str,optional
        :param method: Which method to use to solve for the coefficients (and smoothing parameters). "Chol" relies on Cholesky decomposition. This is extremely efficient but in principle less stable, numerically speaking. For a maximum of numerical stability set this to "QR/Chol" or "LU/Chol". In that case the coefficients are still obtained via a Cholesky decomposition but a QR/LU decomposition is formed afterwards to check for rank deficiencies and to drop coefficients that cannot be estimated given the current smoothing parameter values. This takes substantially longer. If this is set to ``'qEFS'``, then the coefficients are estimated via quasi netwon and the smoothing penalties are estimated from the quasi newton approximation to the hessian. This only requieres first derviative information. Defaults to "QR/Chol".
        :type method: str,optional
        :param check_cond: Whether to obtain an estimate of the condition number for the linear system that is solved. When ``check_cond=0``, no check will be performed. When ``check_cond=1``, an estimate of the condition number for the final system (at convergence) will be computed and warnings will be issued based on the outcome (see :func:`mssm.src.python.gamm_solvers.est_condition`). Defaults to 1.
        :type check_cond: int,optional
        :param piv_tol: Deprecated.
        :type piv_tol: float,optional
        :param progress_bar: Whether progress should be displayed (convergence info and time estimate). Defaults to True.
        :type progress_bar: bool,optional
        :param n_cores: Number of cores to use during parts of the estimation that can be done in parallel. Defaults to 10.
        :type n_cores: int,optional
        :param seed: Seed to use for random parameter initialization. Defaults to 0
        :type seed: int,optional
        :param drop_NA: Whether to drop rows in the **model matrices** and observations vectors corresponding to NAs in the observation vectors. Set this to False if you want to handle NAs yourself in the likelihood function. Defaults to True.
        :type drop_NA: bool,optional
        :param init_lambda: A set of initial :math:`\\lambda` parameters to use by the model. Length of list must match number of parameters to be estimated. Defaults to None
        :type init_lambda: [float],optional
        :param form_VH: Whether to explicitly form matrix ``V`` - the estimated inverse of the negative Hessian of the penalized likelihood - and ``H`` - the estimate of the Hessian of the log-likelihood - when using the ``qEFS`` method. If set to False, only ``V`` is returned - as a :class:`scipy.sparse.linalg.LinearOperator` - and available in ``self.lvi``. Additionally, ``self.hessian`` will then be equal to ``None``. **Note**, that this will break default prediction/confidence interval methods - so do not call them. Defaults to True
        :type form_VH: bool,optional
        :param use_grad: Deprecated.
        :type use_grad: bool,optional
        :param build_mat: An (optional) list, containing one bool per :class:`mssm.src.python.formula.Formula` in ``self.formulas`` - indicating whether the corresponding model matrix should be built. Useful if multiple formulas specify the same model matrix, in which case only one needs to be built. Only the matrices actually built are then passed down to the likelihood/gradient/hessian function in ``Xs``. Defaults to None, which means all model matrices are built.
        :type build_mat: [bool], optional
        :param should_keep_drop: Only used when ``method in ["QR/Chol","LU/Chol","Direct/Chol"]``. If set to True, any coefficients that are dropped during fitting - are permanently excluded from all subsequent iterations. If set to False, this is determined anew at every iteration - **costly**! Defaults to True.
        :type should_keep_drop: bool,optional
        :param gamma: Setting this to a value larger than 1 promotes more complex (less smooth) models. Setting this to a value smaller than 1 (but must be > 0) promotes smoother models! Defaults to 1.
        :type gamma: float,optional
        :param qEFSH: Should the hessian approximation use a symmetric rank 1 update (``qEFSH='SR1'``) that is forced to result in positive semi-definiteness of the approximation or the standard bfgs update (``qEFSH='BFGS'``) . Defaults to 'SR1'.
        :type qEFSH: str,optional
        :param overwrite_coef: Whether the initial coefficients passed to the optimization routine should be over-written by the solution obtained for the un-penalized version of the problem when ``method='qEFS'``. Setting this to False will be useful when passing coefficients from a simpler model to initialize a more complex one. Only has an effect when ``qEFS_init_converge=True``. Defaults to True.
        :type overwrite_coef: bool,optional
        :param max_restarts: How often to shrink the coefficient estimate back to a random vector when convergence is reached and when ``method='qEFS'``. The optimizer might get stuck in local minima so it can be helpful to set this to 1-3. What happens is that if we converge, we shrink the coefficients back to a random vector and then continue optimizing once more. Defaults to 0.
        :type max_restarts: int,optional
        :param qEFS_init_converge: Whether to optimize the un-penalzied version of the model and to use the hessian (and optionally coefficients, if ``overwrite_coef=True``) to initialize the q-EFS solver. Ignored if ``method!='qEFS'``. Defaults to False.
        :type qEFS_init_converge: bool,optional
        :param prefit_grad: Whether to rely on Gradient Descent to improve the initial starting estimate for coefficients. Defaults to True.
        :type prefit_grad: bool,optional
        :param repara: Whether to re-parameterize the model (for every proposed update to the regularization parameters) via the steps outlined in Appendix B of Wood (2011) and suggested by Wood et al., (2016). This greatly increases the stability of the fitting iteration. Defaults to True if ``method != 'qEFS'`` else False.
        :type repara: bool,optional
        :param init_bfgs_options: An optional dictionary holding the same key:value pairs that can be passed to ``bfgs_options`` but pased to the optimizer of the un-penalized problem. If this is None, it will be set to a copy of ``bfgs_options``. Only has an effect when ``qEFS_init_converge=True``. Defaults to None.
        :type init_bfgs_options: dict,optional
        :param bfgs_options: An optional dictionary holding arguments that should be passed on to the call of :func:`scipy.optimize.minimize` if ``method=='qEFS'``. If none are provided, the ``gtol`` argument will be initialized to ``conv_tol``. Note also, that in any case the ``maxiter`` argument is automatically set to ``max_inner``. Defaults to None.
        :type bfgs_options: dict,optional
        :raises ValueError: Will throw an error when ``optimizer`` is not 'Newton'.
        """

        # Initialize remaining arguments to defaults
        if bfgs_options is None:
            bfgs_options = {"gtol":1.1*conv_tol,
                            "ftol":1.1*conv_tol,
                            "maxcor":30,
                            "maxls":20,
                            "maxfun":500}
        
        if control_lambda is None:
            control_lambda = 2 if method != 'qEFS' else 1
            
        if min_inner is None:
            min_inner = max_inner

        if repara is None:
            repara = True if method != 'qEFS' else False
        
        if init_bfgs_options is None:
            init_bfgs_options = copy.deepcopy(bfgs_options)

        # Some checks
        if not optimizer in ["Newton"]:
            raise ValueError("'optimizer' needs to be set to 'Newton'.")
        
        if self.overall_penalties is None and restart == True:
            raise ValueError("Penalties were not initialized. ``Restart`` must be set to False.")

        if extend_lambda and method == 'qEFS':
            warnings.warn("Ignoring argument ``extend_lambda``, which is not supported for ``method='qEFS'``.")
            extend_lambda = False
        
        # Get ys
        ys = []
        for fi,form in enumerate(self.formulas):
            
            # Repeated y-variable - don't have to pass all of them
            if fi > 0 and form.get_lhs().variable == self.formulas[0].get_lhs().variable:
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

        # Build penalties and model matrices for all formulas
        Xs = []
        ind_penalties = []
        for fi,form in enumerate(self.formulas):
            
            if build_mat is None or build_mat[fi]:
                if restart == False:
                    ind_penalties.append(build_penalties(form))
                Xs.append(self.get_mmat(drop_NA=drop_NA,par=fi))

        # Get all penalties
        if restart == False:
            shared_penalties = embed_shared_penalties(ind_penalties,self.formulas,self.family.extra_coef)
            shared_penalties = [sp for sp in shared_penalties if len(sp) > 0]

            smooth_pen = [pen for pens in shared_penalties for pen in pens]
            self.overall_penalties = smooth_pen

            # Clean up
            shared_penalties = None
            ind_penalties = None
        
            # Check for family-wide initialization of lambda values
            if init_lambda is None:
                init_lambda = self.family.init_lambda(smooth_pen)
            
            # Otherwise initialize with provided values or simply with somewhat stronger penalty than for GAMs
            for pen_i in range(len(smooth_pen)):
                if init_lambda is None:
                    smooth_pen[pen_i].lam = 10 if method != 'qEFS' else 1
                else:
                    smooth_pen[pen_i].lam = init_lambda[pen_i]

        else:
            smooth_pen = self.overall_penalties

        # Initialize overall coefficients
        form_n_coef = [form.n_coef for form in self.formulas]
        form_up_coef = [form.unpenalized_coef for form in self.formulas]
        n_coef = np.sum(form_n_coef)

        if self.family.extra_coef is not None:
            form_n_coef.append(self.family.extra_coef)
            form_up_coef.append(self.family.extra_coef)
            n_coef += self.family.extra_coef

        # Again check first for family wide initialization
        if init_coef is None:
            init_coef = self.family.init_coef([GAMM(form,family=Gaussian()) for form in self.formulas])
        
        # Otherwise again initialize with provided values or randomly
        if not init_coef is None:
            coef = np.array(init_coef).reshape(-1,1)
            
            if self.family.extra_coef is not None:
                coef = np.concatenate((coef,np.ones(self.family.extra_coef).reshape(-1,1)),axis=0)
        else:
            coef = scp.stats.norm.rvs(size=n_coef,random_state=seed).reshape(-1,1)

        coef_split_idx = form_n_coef[:-1]

        if len(self.formulas) > 1:
            for coef_i in range(1,len(coef_split_idx)):
                coef_split_idx[coef_i] += coef_split_idx[coef_i-1]
        
        # Now fit model
        coef,H,LV,LV_linop,total_edf,term_edfs,penalty,smooth_pen,fit_info = solve_generalSmooth_sparse(self.family,ys,Xs,form_n_coef,form_up_coef,coef,coef_split_idx,smooth_pen,
                                                                                    max_outer,max_inner,min_inner,conv_tol,extend_lambda,extension_method_lam,
                                                                                    control_lambda,optimizer,method,check_cond,piv_tol,repara,should_keep_drop,form_VH,
                                                                                    use_grad,gamma,qEFSH,overwrite_coef,max_restarts,qEFS_init_converge,prefit_grad,
                                                                                    progress_bar,n_cores,init_bfgs_options,bfgs_options)
        
        self.overall_penalties = smooth_pen
        self.coef = coef
        self.edf = total_edf
        self.term_edf = term_edfs
        self.penalty = penalty
        self.coef_split_idx = coef_split_idx
        self.lvi = LV
        self.lvi_linop = LV_linop
        self.hessian = H
        if fit_info.eps > 0: # Make sure -H + S_emb is invertible
            warnings.warn(f"model.info.eps > 0 ({np.round(fit_info.eps,decimals=2)}). Perturbing Hessian of log-likelihood to ensure that negative Hessian of penalized log-likelihood is invertible.")
            self.hessian -= fit_info.eps*scp.sparse.identity(H.shape[1],format='csc')
        self.info = fit_info

        # Assign predictions and parameter estimates
        if build_mat is None:
            split_coef = np.split(coef,coef_split_idx)
            self.preds = [X@spcoef for X,spcoef in zip(Xs,split_coef)]
            self.mus = [link.fi(pred) for link,pred in zip(self.family.links,self.preds)]
    
    ##################################### Prediction #####################################

    def sample_post(self, n_ps:int, use_post:list[int]|None=None, deviations:bool=False, seed:int|None=None, par:int=0) -> np.ndarray:
        """
        Obtain ``n_ps`` samples from posterior :math:`[\\boldsymbol{\\beta}_m - \\hat{\\boldsymbol{\\beta}}_m] | \\mathbf{y},\\boldsymbol{\\lambda} \\sim N(0,\\mathbf{V})`,
        where :math:`\\mathbf{V}=[-\\mathbf{H} + \\mathbf{S}_{\\lambda}]^{-1}` (see Wood et al., 2016; Wood 2017, section 6.10), :math:`\\boldsymbol{\\beta}_m` is the set of
        coefficients in the model of parameter :math:`m` of the log-likelihood (see argument ``par``), and :math:`\\mathbf{H}` is the hessian of
        the log-likelihood (Wood et al., 2016;). To obtain samples for :math:`\\boldsymbol{\\beta}_m`, set ``deviations`` to false.

        see :func:`sample_MVN` for more details and the :func:`GAMMLSS.sample_post` function for code examples.

        References:

         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

        :param n_ps: Number of samples to obtain from posterior.
        :type n_ps: int,optional
        :param use_post: The indices corresponding to coefficients for which to actually obtain samples. **Note**: an index of 0 indexes the first coefficient in the model of parameter ``par``, that is indices have to correspond to columns in the parameter-specific model matrix. By default all coefficients are sampled.
        :type use_post: [int],optional
        :param deviations: Whether to return samples of **deviations** from the estimated coefficients (i.e., :math:`\\boldsymbol{\\beta} - \\hat{\\boldsymbol{\\beta}}`) or actual samples of coefficients (i.e., :math:`\\boldsymbol{\\beta}`), defaults to False
        :type deviations: bool,optional
        :param seed: A seed to use for the sampling, defaults to None
        :type seed: int,optional
        :param par: The index corresponding to the parameter of the log-likelihood for which samples are to be obtained for the coefficients, defaults to 0.
        :type par: int, optional
        :returns: An np.ndarray of dimension ``[len(use_post),n_ps]`` containing the posterior samples. If ``use_post is None``, ``len(use_post)`` will match the number of coefficients associated with parameter ``par`` of the log-likelihood instead. Can simply be post-multiplied with (the subset of columns indicated by ``use_post`` of) the model matrix :math:`\\mathbf{X}^m` associated with the parameter :math:`m` of the log-likelihood to generate posterior **sample curves**.
        :rtype: np.ndarray
        """
        
        # Extract coef and cols of lvi associated with par
        if len(self.formulas) > 1:
            split_coef = np.split(self.coef,self.coef_split_idx)
            split_idx = np.ndarray.flatten(np.split(np.arange(len(self.coef)),self.coef_split_idx)[par])
            coef = np.ndarray.flatten(split_coef[par])
            lvi = self.lvi[:,split_idx]
        else:
            coef = self.coef.flatten()
            lvi = self.lvi

        # Now sample
        if deviations:
            post = sample_MVN(n_ps,0,self.scale,P=None,L=None,LI=lvi,use=use_post,seed=seed)
        else:
            post = sample_MVN(n_ps,coef,self.scale,P=None,L=None,LI=lvi,use=use_post,seed=seed)

        return post

    def predict(self, use_terms:list[int]|None, n_dat:pd.DataFrame, alpha:float=0.05, ci:bool=False, whole_interval:bool=False, n_ps:int=10000, seed:int|None=None, par:int=0) -> tuple[np.ndarray,scp.sparse.csc_array,np.ndarray|None]:
        """
        Make a prediction using the fitted model for new data ``n_dat`` using only the terms indexed by ``use_terms`` and for parameter ``par`` of the log-likelihood.

        Importantly, predictions and standard errors are always returned on the scale of the linear predictor.

        See the :func:`GAMMLSS.predict` function for code examples.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
         - Simpson, G. (2016). Simultaneous intervals for smooths revisited.

        :param use_terms: The indices corresponding to the terms that should be used to obtain the prediction or ``None`` in which case all terms will be used.
        :type use_terms: list[int] or None
        :param n_dat: A pandas DataFrame containing new data for which to make the prediction. Importantly, all variables present in the data used to fit the model also need to be present in this DataFrame. Additionally, factor variables must only include levels also present in the data used to fit the model. If you want to exclude a specific factor from the prediction (for example the factor subject) don't include the terms that involve it in the ``use_terms`` argument.
        :type n_dat: pd.DataFrame
        :param alpha: The alpha level to use for the standard error calculation. Specifically, 1 - (``alpha``/2) will be used to determine the critical cut-off value according to a N(0,1).
        :type alpha: float, optional
        :param ci: Whether the standard error ``se`` for credible interval (CI; see  Wood, 2017) calculation should be returned. The CI is then [``pred`` - ``se``, ``pred`` + ``se``]
        :type ci: bool, optional
        :param whole_interval: Whether or not to adjuste the point-wise CI to behave like whole-function interval (based on Wood, 2017; section 6.10.2 and Simpson, 2016). Defaults to False. The CI is then [``pred`` - ``se``, ``pred`` + ``se``]
        :type whole_interval: bool, optional
        :param n_ps: How many samples to draw from the posterior in case the point-wise CI is adjusted to behave like a whole-function interval CI.
        :type n_ps: int, optional
        :param seed: Can be used to provide a seed for the posterior sampling step in case the point-wise CI is adjusted to behave like a whole-function interval CI.
        :type seed: int or None, optional
        :param par: The index corresponding to the parameter of the log-likelihood for which to make the prediction, defaults to 0
        :type par: int, optional
        :raises ValueError: An error is raised in case the standard error is to be computed for a Multinomial GAMMLSS model, which is currently not supported.
        :return: A tuple with 3 entries. The first entry is the prediction ``pred`` based on the new data ``n_dat``. The second entry is the model matrix built for ``n_dat`` that was post-multiplied with the model coefficients to obtain ``pred``. The third entry is ``None`` if ``ci``==``False`` else the standard error ``se`` in the prediction.
        :rtype: (np.ndarray,scp.sparse.csc_array,np.ndarray or None)
        """

        # Extract desired formula and perform some checks
        form = self.formulas[par]
        var_map = form.get_var_map()
        var_keys = var_map.keys()
        sub_group_vars = form.get_subgroup_variables()

        for k in var_keys:
            if k in sub_group_vars:
                if k.split(":")[0] not in n_dat.columns:
                    raise IndexError(f"Variable {k.split(':')[0]} is missing in new data.")
            else:
                if k not in n_dat.columns:
                    raise IndexError(f"Variable {k} is missing in new data.")
        
        # Extract coef and cols of lvi associated with par  
        if len(self.formulas) > 1:
            split_coef = np.split(self.coef,self.coef_split_idx)
            split_idx = np.ndarray.flatten(np.split(np.arange(len(self.coef)),self.coef_split_idx)[par])
            coef = np.ndarray.flatten(split_coef[par])
            lvi = self.lvi[:,split_idx]
        else:
            coef = self.coef.flatten()
            lvi = self.lvi
        
        # Encode test data
        _,pred_cov_flat,_,_,pred_cov,_,_ = form.encode_data(n_dat,prediction=True)

        # Then, we need to build the model matrix - but only for the terms which should
        # be included in the prediction!
        terms = form.terms
        has_intercept = form.has_intercept
        ltx = form.get_linear_term_idx()
        irstx = form.get_ir_smooth_term_idx()
        stx = form.get_smooth_term_idx()
        rtx = form.get_random_term_idx()
        var_types = form.get_var_types()
        var_mins = form.get_var_mins()
        var_maxs = form.get_var_maxs()
        factor_levels = form.get_factor_levels()

        if len(irstx) == 0:
            pred_cov = None

        # So we pass the desired terms to the use_only argument
        predi_mat = build_sparse_matrix_from_formula(terms,has_intercept,
                                                     ltx,irstx,stx,rtx,var_types,var_map,
                                                     var_mins,var_maxs,factor_levels,
                                                     pred_cov_flat,pred_cov,
                                                     use_only=use_terms)
        
        # Now we calculate the prediction
        pred = predi_mat @ coef

        # Optionally calculate the boundary for a 1-alpha CI
        if ci:
            # Wood (2017) 6.10
            c = predi_mat @ lvi.T @ lvi * self.scale @ predi_mat.T
            c = c.diagonal()
            b = scp.stats.norm.ppf(1-(alpha/2)) * np.sqrt(c)

            # Whole-interval CI (section 6.10.2 in Wood, 2017), the same idea was also
            # explored by Simpson (2016) who performs very similar computations to compute
            # such intervals. See adjust_CI function.
            if whole_interval:
                b = adjust_CI(self,n_ps,b,predi_mat,use_terms,alpha,seed,par)

            return pred,predi_mat,b

        return pred,predi_mat,None
    
    def predict_diff(self, dat1:pd.DataFrame, dat2:pd.DataFrame, use_terms:list[int]|None, alpha:float=0.05, whole_interval:bool=False, n_ps:int=10000, seed:int|None=None, par:int=0) -> tuple[np.ndarray,np.ndarray]:
        """
        Get the difference in the predictions for two datasets and for parameter ``par`` of the log-likelihood. Useful to compare a smooth estimated for
        one level of a factor to the smooth estimated for another level of a factor. In that case, ``dat1`` and ``dat2`` should only differ in the level of
        said factor. Importantly, predictions and standard errors are again always returned on the scale of the linear predictor - 
        see the :func:`predict` method for details.

        See the :func:`GAMMLSS.predict_diff` function for code examples.

        References:

         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
         - Simpson, G. (2016). Simultaneous intervals for smooths revisited.
         - ``get_difference`` function from ``itsadug`` R-package: https://rdrr.io/cran/itsadug/man/get_difference.html

        :param dat1: A pandas DataFrame containing new data for which to make the prediction. Importantly, all variables present in the data used to fit the model also need to be present in this DataFrame. Additionally, factor variables must only include levels also present in the data used to fit the model. If you want to exclude a specific factor from the prediction (for example the factor subject) don't include the terms that involve it in the `use_terms` argument.
        :type dat1: pd.DataFrame
        :param dat2: A second pandas DataFrame for which to also make a prediction. The difference in the prediction between this `dat1` will be returned.
        :type dat2: pd.DataFrame
        :param use_terms: The indices corresponding to the terms that should be used to obtain the prediction or ``None`` in which case all terms will be used.
        :type use_terms: list[int] or None
        :param alpha: The alpha level to use for the standard error calculation. Specifically, 1 - (`alpha`/2) will be used to determine the critical cut-off value according to a N(0,1).
        :type alpha: float, optional
        :param whole_interval: Whether or not to adjuste the point-wise CI to behave like whole-function interval (based on Wood, 2017; section 6.10.2 and Simpson, 2016). Defaults to False.
        :type whole_interval: bool, optional
        :param n_ps: How many samples to draw from the posterior in case the point-wise CI is adjusted to behave like a whole-function interval CI.
        :type n_ps: int, optional
        :param seed: Can be used to provide a seed for the posterior sampling step in case the point-wise CI is adjusted to behave like a whole-function interval CI.
        :type seed: int or None, optional
        :param par: The index corresponding to the parameter of the log-likelihood for which to make the prediction, defaults to 0
        :type par: int, optional
        :raises ValueError: An error is raised in case the predicted difference is to be computed for a Multinomial GAMMLSS model, which is currently not supported.
        :return: A tuple with 2 entries. The first entry is the predicted difference (between the two data sets ``dat1`` & ``dat2``) ``diff``. The second entry is the standard error ``se`` of the predicted difference. The difference CI is then [``diff`` - ``se``, ``diff`` + ``se``]
        :rtype: (np.ndarray,np.ndarray)
        """
        
        _,pmat1,_ = self.predict(use_terms,dat1,par=par)
        _,pmat2,_ = self.predict(use_terms,dat2,par=par)

        pmat_diff = pmat1 - pmat2

        # Extract coef and cols of lvi associated with par  
        if len(self.formulas) > 1:
            split_coef = np.split(self.coef,self.coef_split_idx)
            split_idx = np.ndarray.flatten(np.split(np.arange(len(self.coef)),self.coef_split_idx)[par])
            coef = np.ndarray.flatten(split_coef[par])
            lvi = self.lvi[:,split_idx]
        else:
            coef = self.coef.flatten()
            lvi = self.lvi

        
        # Predicted difference
        diff = pmat_diff @ coef
        
        # Difference CI
        c = pmat_diff @ lvi.T @ lvi * self.scale @ pmat_diff.T
        c = c.diagonal()
        b = scp.stats.norm.ppf(1-(alpha/2)) * np.sqrt(c)

        # Whole-interval CI (section 6.10.2 in Wood, 2017), the same idea was also
        # explored by Simpson (2016) who performs very similar computations to compute
        # such intervals. See adjust_CI function.
        if whole_interval:
            b = adjust_CI(self,n_ps,b,pmat_diff,use_terms,alpha,seed,par)

        return diff,b

##################################### GAMMLSS class #####################################

class GAMMLSS(GSMM):
    """
    Class to fit Generalized Additive Mixed Models of Location Scale and Shape (see Rigby & Stasinopoulos, 2005).

    Examples::

        from mssm.models import *
        from mssmViz.sim import *
        from mssmViz.plot import *
        import matplotlib.pyplot as plt

        # Simulate 500 data points
        GAUMLSSDat = sim6(500,seed=20)

        # We need to model the mean: \\mu_i = \\alpha + f(x0)
        formula_m = Formula(lhs("y"),
                            [i(),f(["x0"],nk=10)],
                            data=GAUMLSSDat)

        # and the standard deviation as well: log(\\sigma_i) = \\alpha + f(x0)
        formula_sd = Formula(lhs("y"),
                            [i(),f(["x0"],nk=10)],
                            data=GAUMLSSDat)

        # Collect both formulas
        formulas = [formula_m,formula_sd]

        # Create Gaussian GAMMLSS family with identity link for mean
        # and log link for sigma
        family = GAUMLSS([Identity(),LOG()])

        # Now define the model and fit!
        model = GAMMLSS(formulas,family)
        model.fit()

        # Get total coef vector & split them
        coef = model.coef
        split_coef = np.split(coef,model.coef_split_idx)

        # Get coef associated with the mean
        coef_m = split_coef[0]
        # and with the scale parameter
        coef_s = split_coef[1]

        # Similarly, `preds` holds linear predictions for m & s
        pred_m = model.preds[0]
        pred_s = model.preds[1]

        # While `mu` holds the estimated fitted parameters
        # (i.e., `preds` after applying the inverse of the link function of each parameter)
        mu_m = model.mus[0]
        mu_s = model.mus[1]

    References:
     - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, Scale and Shape.
     - Wood, S. N., & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. https://doi.org/10.1111/biom.12666
     - Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models: Estimation of Semiparametric Generalized Linear Models. https://doi.org/10.1111/j.1467-9868.2010.00749.x
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
    
    :param formulas: A list of formulas for the GAMMLS model
    :type formulas: [Formula]
    :param family: A :class:`GAMLSSFamily`. Currently :class:`GAUMLSS`, :class:`MULNOMLSS`, and :class:`GAMMALS` are supported.
    :type family: GAMLSSFamily
    :ivar [Formula] formulas: The list of formulas passed to the constructor.
    :ivar scp.sparse.csc_array lvi: The inverse of the Cholesky factor of the conditional model coefficient covariance matrix. Initialized with ``None``.
    :ivar np.ndarray coef:  Contains all coefficients estimated for the model. Shape of the array is (-1,1). Initialized with ``None``.
    :ivar [[float]] preds: The linear predictors for every parameter of ``family`` evaluated for each observation in the training data (after removing NaNs). Initialized with ``None``.
    :ivar [[float]] mus: The predicted means for every parameter of ``family`` evaluated for each observation in the training data (after removing NaNs). Initialized with ``None``.
    :ivar scp.sparse.csc_array hessian:  Estimated hessian of the log-likelihood (will correspond to ``hessian - diag*eps`` if ``self.info.eps > 0`` after fitting). Initialized with ``None``.
    :ivar float edf: The model estimated degrees of freedom as a float. Initialized with ``None``.
    :ivar float edf1: The model estimated degrees of freedom as a float corrected for smoothness bias. Set by the :func:`approx_smooth_p_values` function, the first time it is called. Initialized with ``None``.
    :ivar [float] term_edf: The estimated degrees of freedom per smooth term. Initialized with ``None``.
    :ivar [float] term_edf1: The estimated degrees of freedom per smooth term corrected for smoothness bias. Set by the :func:`approx_smooth_p_values` function, the first time it is called. Initialized with ``None``.
    :ivar float penalty: The total penalty applied to the model deviance after fitting as a float. Initialized with ``None``.
    :ivar [int] coef_split_idx: The index at which to split the overall coefficient vector into separate lists - one per parameter of ``family``. See the examples. Initialized after fitting!
    :ivar [LambdaTerm] overall_penalties:  Contains all penalties estimated for the model. Initialized with ``None``.
    :ivar Fit_info info: A :class:`Fit_info` instance, with information about convergence (speed) of the model.
    :ivar np.ndarray res: The working residuals of the model (If applicable). Initialized with ``None``.
    """
    def __init__(self, formulas: list[Formula], family: GAMLSSFamily):
        super().__init__(formulas, family)
        self.res:np.ndarray = None
    
    ##################################### Getters #####################################
    
    def get_pars(self) -> np.ndarray:
        """
        Returns a list containing all coefficients estimated for the model. Use ``self.coef_split_idx`` to split the vector into separate subsets per distribution parameter.

        Will return None if called before fitting was completed.
        
        :return: Model coefficients - before splitting!
        :rtype: [float] or None
        """
        return super.get_pars()
    
    def get_mmat(self,use_terms:list[int]|None=None,par:int|None=None) -> list[scp.sparse.csc_array]|scp.sparse.csc_array:
        """
        Returns a list containing exaclty the model matrices used for fitting as a ``scipy.sparse.csc_array``. Will raise an error when fitting was not completed before calling this function.

        Optionally, the model matrix associated with a specific parameter of the log-likelihood can be obtained by setting ``par`` to the desired index, instead of ``None``.
        Additionally, all columns not corresponding to terms for which the indices are provided via ``use_terms`` can optionally be zeroed.

        :param use_terms: Optionally provide indices of terms in the formual that should be created. If this argument is provided columns corresponding to any term not included in this list will be zeroed, defaults to None
        :type use_terms: [int], optional
        :param par: The index corresponding to the parameter of the distribution for which to obtain the model matrix. Setting this to ``None`` means all matrices are returned in a list, defaults to None.
        :type par: int or None, optional
        :raises ValueError: Will throw an error when called before the model was fitted/before model penalties were formed.
        :return: Model matrices :math:`\\mathbf{X}` used for fitting - one per parameter of ``self.family`` or a single model matrix for a specific parameter.
        :rtype: [scp.sparse.csc_array] or scp.sparse.csc_array
        """
        return super().get_mmat(use_terms,True,par)

    def get_llk(self,penalized:bool=True) -> float|None:
        """
        Get the (penalized) log-likelihood of the estimated model (float or None) given the trainings data.
        
        Will instead return ``None`` if called before fitting.
        
        :param penalized: Whether the penalized log-likelihood should be returned or the regular log-likelihood, defaults to True
        :type penalized: bool, optional
        :return: llk score
        :rtype: float or None
        """

        pen = 0
        if penalized:
            pen = 0.5*self.penalty
        if self.preds is not None:
            mus = [self.family.links[i].fi(self.preds[i]) for i in range(self.family.n_par)]

            y = self.formulas[0].y_flat[self.formulas[0].NOT_NA_flat]
            if not self.formulas[0].get_lhs().f is None:
                y = self.formulas[0].get_lhs().f(y)

            return self.family.llk(y,*mus) - pen

        return None
                        
    def get_reml(self) -> float:
        """
        Get's the Laplcae approximate REML (Restrcited Maximum Likelihood) score for the estimated lambda values (see Wood, 2011).

        References:

         - Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models: Estimation of Semiparametric Generalized Linear Models.
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.

        :raises ValueError: Will throw an error when called before the model was fitted/before model penalties were formed.
        :return: REML score
        :rtype: float
        """

        if self.coef is None or self.hessian is None:
            raise ValueError("Model needs to be estimated before evaluating the REML score. Call model.fit()")
        
        llk = self.get_llk(False)

        keep = None
        if self.info.dropped is not None:
            keep = [cidx for cidx in range(self.hessian.shape[1]) if cidx not in self.info.dropped]
        
        reml = REML(llk,-1*self.hessian,self.coef,1,self.overall_penalties,keep)[0,0]
        return reml
    
    def get_resid(self,**kwargs) -> np.ndarray:
        """ Returns standarized residuals for GAMMLSS models (Rigby & Stasinopoulos, 2005).

        The computation of the residual vector will differ between different GAMMLSS models and is thus implemented
        as a method by each GAMMLSS family. These should be consulted to get more details. In general, if the
        model is specified correctly, the returned vector should approximately look like what could be expected from
        taking :math:`N` independent samples from :math:`N(0,1)`.

        Additional arguments required by the specific :func:`GAMLSSFamily.get_resid` method can be passed along via ``kwargs``.

        **Note**: Families for which no residuals are available can return None.

        References:
         - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, Scale and Shape.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

        :raises NotImplementedError: An error is raised in case the residuals are to be computed for a Multinomial GAMMLSS model, which is currently not supported.
        :raises ValueError: An error is raised in case the residuals are requested before the model has been fit.
        :return: A np.ndarray of standardized residuals that should be :math:`\\sim N(0,1)` if the model is correct.
        :return: Standardized residual vector as array of shape (-1,1)
        :rtype: np.ndarray
        """

        if self.coef is None or self.hessian is None:
            raise ValueError("Model needs to be estimated before evaluating the residuals. Call model.fit()")

        if isinstance(self.family,MULNOMLSS):
            raise NotImplementedError("Residual computation for Multinomial model is not currently supported.")
        
        return self.family.get_resid(self.formulas[0].y_flat[self.formulas[0].NOT_NA_flat],*self.mus,**kwargs)

    ##################################### Summary #####################################
    
    def print_parametric_terms(self):
        """Prints summary output for linear/parametric terms in the model, separately for each parameter of the family's distribution.
        
        For each coefficient, the named identifier and estimated value are returned. In addition, for each coefficient a p-value is returned, testing
        the null-hypothesis that the corresponding coefficient :math:`\\beta=0`. Under the assumption that this is true, the Null distribution follows
        approximately a standardized normal distribution. The corresponding z-statistic and the p-value are printed.
        See Wood (2017) section 6.12 and 1.3.3 for more details.

        Note that, un-penalized coefficients that are part of a smooth function are not covered by this function.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

        :raises NotImplementedError: Will throw an error when called for a model for which the model matrix was never former completely.
        """
        super().print_parametric_terms()
    
    def print_smooth_terms(self, pen_cutoff:float=0.2, p_values:bool=False, edf1:bool = True):
        """Prints the name of the smooth terms included in the model. After fitting, the estimated degrees of freedom per term are printed as well.
        Smooth terms with edf. < ``pen_cutoff`` will be highlighted. This only makes sense when extra Kernel penalties are placed on smooth terms to enable
        penalizing them to a constant zero. In that case edf. < ``pen_cutoff`` can then be taken as evidence that the smooth has all but notationally disappeared
        from the model, i.e., it does not contribute meaningfully to the model fit. This can be used as an alternative form of model selection - see Marra & Wood (2011).

        References:

         - Marra & Wood (2011). Practical variable selection for generalized additive models.

        :param pen_cutoff: At which edf. cut-off smooth terms should be marked as "effectively removed", defaults to None
        :type pen_cutoff: float, optional
        :param p_values: Whether approximate p-values should be printed for the smooth terms, defaults to False
        :type p_values: bool, optional
        :param edf1: Whether or not the estimated degrees of freedom should be corrected for smoothnes bias. Doing so results in more accurate p-values but can be expensive for large models for which the difference is anyway likely to be marginal, defaults to False
        :type edf1: bool, optional
        """
        super().print_smooth_terms(pen_cutoff,p_values,edf1)
    
    ##################################### Fitting #####################################
        
    def fit(self,max_outer:int=200,max_inner:int=500,min_inner:int|None=None,conv_tol:float=1e-7,extend_lambda:bool=False,
            extension_method_lam:str="nesterov2",control_lambda:int=2,restart:bool=False,method:str="QR/Chol",check_cond:int=1,
            piv_tol:float=np.power(np.finfo(float).eps,0.04),should_keep_drop:bool=True,prefit_grad:bool=True,repara:bool=True,
            progress_bar:bool=True,n_cores:int=10,seed:int=0,init_lambda:list[float]|None=None):
        """
        Fit the specified model.

        **Note**: Keyword arguments are initialized to maximise stability. For faster estimation set ``method='Chol'``.

        Examples::

            from mssm.models import *
            from mssmViz.sim import *
            from mssmViz.plot import *
            import matplotlib.pyplot as plt

            # Simulate 500 data points
            GAUMLSSDat = sim6(500,seed=20)

            # We need to model the mean: \\mu_i = \\alpha + f(x0)
            formula_m = Formula(lhs("y"),
                                [i(),f(["x0"],nk=10)],
                                data=GAUMLSSDat)

            # and the standard deviation as well: log(\\sigma_i) = \\alpha + f(x0)
            formula_sd = Formula(lhs("y"),
                                [i(),f(["x0"],nk=10)],
                                data=GAUMLSSDat)

            # Collect both formulas
            formulas = [formula_m,formula_sd]

            # Create Gaussian GAMMLSS family with identity link for mean
            # and log link for sigma
            family = GAUMLSS([Identity(),LOG()])

            # Now define the model and fit!
            model = GAMMLSS(formulas,family)
            model.fit()

            # Now fit again via Cholesky
            model.fit(method="Chol")

        :param max_outer: The maximum number of fitting iterations.
        :type max_outer: int,optional
        :param max_inner: The maximum number of fitting iterations to use by the inner Newton step for coefficients.
        :type max_inner: int,optional
        :param min_inner: The minimum number of fitting iterations to use by the inner Newton step for coefficients. By default set to ``max_inner``.
        :type min_inner: int,optional
        :param conv_tol: The relative (change in penalized deviance is compared against ``conv_tol`` * previous penalized deviance) criterion used to determine convergence.
        :type conv_tol: float,optional
        :param extend_lambda: Whether lambda proposals should be accelerated or not. Can lower the number of new smoothing penalty proposals necessary for models involving heavily penalized functions. Disabled by default.
        :type extend_lambda: bool,optional
        :param extension_method_lam: **Experimental - do not change!** Which method to use to extend lambda proposals. Set to 'nesterov2' by default.
        :type extension_method_lam: str,optional
        :param control_lambda: Whether lambda proposals should be checked (and if necessary decreased) for whether or not they (approxiately) increase the Laplace approximate restricted maximum likelihood of the model. Setting this to 0 disables control. Setting it to 1 means the step will never be smaller than the original EFS update but extensions will be removed in case the objective was exceeded. Setting it to 2 means that steps will be halved if it fails to increase the approximate REML. Set to 2 by default.
        :type control_lambda: int,optional
        :param restart: Whether fitting should be resumed. Only possible if the same model has previously completed at least one fitting iteration.
        :type restart: bool,optional
        :param method: Which method to use to solve for the coefficients (and smoothing parameters). "Chol" relies on Cholesky decomposition. This is extremely efficient but in principle less stable, numerically speaking. For a maximum of numerical stability set this to "QR/Chol" or "LU/Chol". In that case the coefficients are still obtained via a Cholesky decomposition but a QR/LU decomposition is formed afterwards to check for rank deficiencies and to drop coefficients that cannot be estimated given the current smoothing parameter values. This takes substantially longer. Defaults to "QR/Chol".
        :type method: str,optional
        :param check_cond: Whether to obtain an estimate of the condition number for the linear system that is solved. When ``check_cond=0``, no check will be performed. When ``check_cond=1``, an estimate of the condition number for the final system (at convergence) will be computed and warnings will be issued based on the outcome (see :func:`mssm.src.python.gamm_solvers.est_condition`). Defaults to 1.
        :type check_cond: int,optional
        :param piv_tol: Deprecated.
        :type piv_tol: float,optional
        :param should_keep_drop: Only used when ``method in ["QR/Chol","LU/Chol","Direct/Chol"]``. If set to True, any coefficients that are dropped during fitting - are permanently excluded from all subsequent iterations. If set to False, this is determined anew at every iteration - **costly**! Defaults to True.
        :type should_keep_drop: bool,optional
        :param prefit_grad: Whether to rely on Gradient Descent to improve the initial starting estimate for coefficients. Defaults to True.
        :type prefit_grad: bool,optional
        :param repara: Whether to re-parameterize the model (for every proposed update to the regularization parameters) via the steps outlined in Appendix B of Wood (2011) and suggested by Wood et al., (2016). This greatly increases the stability of the fitting iteration. Defaults to True.
        :type repara: bool,optional
        :param progress_bar: Whether progress should be displayed (convergence info and time estimate). Defaults to True.
        :type progress_bar: bool,optional
        :param n_cores: Number of cores to use during parts of the estimation that can be done in parallel. Defaults to 10.
        :type n_cores: int,optional
        :param seed: Seed to use for random parameter initialization. Defaults to 0
        :type seed: int,optional
        :param init_lambda: A set of initial :math:`\\lambda` parameters to use by the model. Length of list must match number of parameters to be estimated. Defaults to None
        :type init_lambda: [float],optional
        """

        # Initialize remaining arguments to defaults
        if min_inner is None:
            min_inner = max_inner

        # Some checks
        if self.overall_penalties is None and restart == True:
            raise ValueError("Penalties were not initialized. ``Restart`` must be set to False.")
        
        # Get y
        y = self.formulas[0].y_flat[self.formulas[0].NOT_NA_flat]

        if not self.formulas[0].get_lhs().f is None:
            # Optionally apply function to dep. var. before fitting. Not sure why that would be desirable for this model class...
            y = self.formulas[0].get_lhs().f(y)

        # Build penalties and model matrices for all formulas
        Xs = []
        ind_penalties = []
        for fi,form in enumerate(self.formulas):
            if restart == False:
                ind_penalties.append(build_penalties(form))
            Xs.append(self.get_mmat(par=fi))

        # Initialize coef from family
        coef = self.family.init_coef([GAMM(form,family=Gaussian()) for form in self.formulas])

        # Get GAMMLSS penalties
        if restart == False:
            shared_penalties = embed_shared_penalties(ind_penalties,self.formulas,None)
            gamlss_pen = [pen for pens in shared_penalties for pen in pens]
            self.overall_penalties = gamlss_pen

            # Clean up
            shared_penalties = None
            ind_penalties = None
        
            # Check for family-wide initialization of lambda values
            if init_lambda is None:
                init_lambda = self.family.init_lambda(gamlss_pen)

            # Else start with provided values or simply with much weaker penalty than for GAMs
            for pen_i in range(len(gamlss_pen)):
                if init_lambda is None:
                    gamlss_pen[pen_i].lam = 10
                else:
                    gamlss_pen[pen_i].lam = init_lambda[pen_i]

        else:
            gamlss_pen = self.overall_penalties

        # Initialize overall coefficients if not done by family
        form_n_coef = [form.n_coef for form in self.formulas]
        if coef is None:
            coef = scp.stats.norm.rvs(size=sum(form_n_coef),random_state=seed).reshape(-1,1)

        form_up_coef = [form.unpenalized_coef for form in self.formulas]
        
        # Now get split index
        coef_split_idx = form_n_coef[:-1]

        if self.family.n_par > 1:
            for coef_i in range(1,len(coef_split_idx)):
                coef_split_idx[coef_i] += coef_split_idx[coef_i-1]

        coef,etas,mus,wres,H,LV,total_edf,term_edfs,penalty,gamlss_pen,fit_info = solve_gammlss_sparse(self.family,y,Xs,form_n_coef,form_up_coef,coef,coef_split_idx,
                                                                                            gamlss_pen,max_outer,max_inner,min_inner,conv_tol,
                                                                                            extend_lambda,extension_method_lam,control_lambda,
                                                                                            method,check_cond,piv_tol,repara,should_keep_drop,prefit_grad,progress_bar,n_cores)
        
        self.overall_penalties = gamlss_pen
        self.coef = coef
        self.preds = etas
        self.mus = mus
        self.res = wres
        self.edf = total_edf
        self.term_edf = term_edfs
        self.penalty = penalty
        self.coef_split_idx = coef_split_idx
        self.lvi = LV
        self.hessian = H
        if fit_info.eps > 0: # Make sure -H + S_emb is invertible
            warnings.warn(f"model.info.eps > 0 ({np.round(fit_info.eps,decimals=2)}). Perturbing Hessian of log-likelihood to ensure that negative Hessian of penalized log-likelihood is invertible.")
            self.hessian -= fit_info.eps*scp.sparse.identity(H.shape[1],format='csc')
        self.info = fit_info
    
    ##################################### Prediction #####################################
    
    def sample_post(self, n_ps:int, use_post:list[int]|None=None, deviations:bool=False, seed:int|None=None, par:int=0) -> np.ndarray:
        """
        Obtain ``n_ps`` samples from posterior :math:`[\\boldsymbol{\\beta}_m - \\hat{\\boldsymbol{\\beta}}_m] | \\mathbf{y},\\boldsymbol{\\lambda} \\sim N(0,\\mathbf{V})`,
        where :math:`\\mathbf{V}=[-\\mathbf{H} + \\mathbf{S}_{\\lambda}]^{-1}` (see Wood et al., 2016; Wood 2017, section 6.10), :math:`\\boldsymbol{\\beta}_m` is the set of
        coefficients in the model of parameter :math:`m` of the distribution (see argument ``par``), and :math:`\\mathbf{H}` is the hessian of
        the log-likelihood (Wood et al., 2016;). To obtain samples for :math:`\\boldsymbol{\\beta}`, set ``deviations`` to false.

        see :func:`sample_MVN` for more details.

        Examples::

            from mssm.models import *
            from mssmViz.sim import *
            from mssmViz.plot import *
            import matplotlib.pyplot as plt

            # Simulate 500 data points
            GAUMLSSDat = sim6(500,seed=20)

            # We need to model the mean: \\mu_i = \\alpha + f(x0)
            formula_m = Formula(lhs("y"),
                                [i(),f(["x0"],nk=10)],
                                data=GAUMLSSDat)

            # and the standard deviation as well: log(\\sigma_i) = \\alpha + f(x0)
            formula_sd = Formula(lhs("y"),
                                [i(),f(["x0"],nk=10)],
                                data=GAUMLSSDat)

            # Collect both formulas
            formulas = [formula_m,formula_sd]

            # Create Gaussian GAMMLSS family with identity link for mean
            # and log link for sigma
            family = GAUMLSS([Identity(),LOG()])

            # Now fit
            model = GAMMLSS(formulas,family)
            model.fit()

            new_dat = pd.DataFrame({"x0":np.linspace(0,1,30)})

            # Now obtain the estimate for `f(["x0"],nk=10)` and the model matrix corresponding to it!
            # Note, that we set `use_terms = [1]` - so all columns in X_f not belonging to `f(["x0"],nk=10)`
            # (e.g., the first one, belonging to the offset) are zeroed.
            mu_f,X_f,_ = model.predict([1],new_dat,ci=True)

            # Now we can sample from the posterior of `f(["x0"],nk=10)` in the model of the mean:
            post = model.sample_post(10000,None,deviations=False,seed=0,par=0)

            # Since we set deviations to false post has coefficient samples and can simply be post-multiplied to
            # get samples of `f(["x0"],nk=10)` 
            post_f = X_f @ post

            # Plot the estimated effect and 50 posterior samples
            plt.plot(new_dat["x0"],mu_f,color="black",linewidth=2)

            for sidx in range(50):
                plt.plot(new_dat["x0"],post_f[:,sidx],alpha=0.2)

            plt.show()

            # In this case, we are not interested in the offset, so we can omit it during the sampling step (i.e., to not sample coefficients
            # for it):

            # `use_post` identifies only coefficients related to `f(["x0"],nk=10)`
            use_post = X_f.sum(axis=0) != 0
            use_post = np.arange(0,X_f.shape[1])[use_post]
            print(use_post)

            # `use_post` can now be passed to `sample_post`:
            post2 = model.sample_post(10000,use_post,deviations=False,seed=0,par=0)

            # Importantly, post2 now has a different shape - which we have to take into account when multiplying.
            post_f2 = X_f[:,use_post] @ post2

            plt.plot(new_dat["x0"],mu_f,color="black",linewidth=2)

            for sidx in range(50):
                plt.plot(new_dat["x0"],post_f2[:,sidx],alpha=0.2)

            plt.show()

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

        :param n_ps: Number of samples to obtain from posterior.
        :type n_ps: int,optional
        :param use_post: The indices corresponding to coefficients for which to actually obtain samples. **Note**: an index of 0 indexes the first coefficient in the model of parameter ``par``, that is indices have to correspond to columns in the parameter-specific model matrix. By default all coefficients are sampled.
        :type use_post: [int],optional
        :param deviations: Whether to return samples of **deviations** from the estimated coefficients (i.e., :math:`\\boldsymbol{\\beta} - \\hat{\\boldsymbol{\\beta}}`) or actual samples of coefficients (i.e., :math:`\\boldsymbol{\\beta}`), defaults to False
        :type deviations: bool,optional
        :param seed: A seed to use for the sampling, defaults to None
        :type seed: int,optional
        :param par: The index corresponding to the distribution parameter for which to make the prediction (e.g., 0 = mean)
        :type par: int
        :returns: An np.ndarray of dimension ``[len(use_post),n_ps]`` containing the posterior samples. Can simply be post-multiplied with model matrix :math:`\\mathbf{X}` to generate posterior **sample curves**.
        :rtype: np.ndarray
        """
        return super().sample_post(n_ps,use_post,deviations,seed,par)

    def predict(self, use_terms:list[int]|None, n_dat:pd.DataFrame, alpha:float=0.05, ci:bool=False, whole_interval:bool=False, n_ps:int=10000, seed:int|None=None, par:int=0) -> tuple[np.ndarray,scp.sparse.csc_array,np.ndarray|None]:
        """
        Make a prediction using the fitted model for new data ``n_dat`` using only the terms indexed by ``use_terms`` and for distribution parameter ``par``.

        Importantly, predictions and standard errors are always returned on the scale of the linear predictor. For the Gaussian GAMMLSS model, the 
        predictions for the standard deviation will for example usually (i.e., for the default link choices) reflect the log of the standard deviation.
        To get the predictions on the standard deviation scale, one could then apply the inverse log-link function to the predictions and the CI-bounds
        on the scale of the respective linear predictor. See the examples below.

        Examples::

            # Simulate 500 data points
            GAUMLSSDat = sim6(500,seed=20)

            # We need to model the mean: \\mu_i = \\alpha + f(x0)
            formula_m = Formula(lhs("y"),
                                [i(),f(["x0"],nk=10)],
                                data=GAUMLSSDat)

            # and the standard deviation as well: log(\\sigma_i) = \\alpha + f(x0)
            formula_sd = Formula(lhs("y"),
                                [i(),f(["x0"],nk=10)],
                                data=GAUMLSSDat)

            # Collect both formulas
            formulas = [formula_m,formula_sd]

            # Create Gaussian GAMMLSS family with identity link for mean
            # and log link for sigma
            family = GAUMLSS([Identity(),LOG()])

            # Now fit
            model = GAMMLSS(formulas,family)
            model.fit()

            new_dat = pd.DataFrame({"x0":np.linspace(0,1,30)})

            # Mean predictions don't have to be transformed since the Identity link is used for this predictor.
            mu_mean,_,b_mean = model.predict(None,new_dat,ci=True)

            # These can be used for confidence intervals:
            mean_upper_CI = mu_mean + b_mean
            mean_lower_CI = mu_mean - b_mean

            # Standard deviation predictions do have to be transformed - by default they are on the log-scale.
            eta_sd,_,b_sd = model.predict(None,new_dat,ci=True,par=1)
            mu_sd = model.family.links[1].fi(eta_sd) # Index to `links` is 1 because the sd is the second parameter!

            # These can be used for approximate confidence intervals:
            sd_upper_CI = model.family.links[1].fi(eta_sd + b_sd)
            sd_lower_CI = model.family.links[1].fi(eta_sd - b_sd)


        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
         - Simpson, G. (2016). Simultaneous intervals for smooths revisited.

        :param use_terms: The indices corresponding to the terms in the formula of the parameter that should be used to obtain the prediction or ``None`` in which case all terms will be used.
        :type use_terms: list[int] or None
        :param n_dat: A pandas DataFrame containing new data for which to make the prediction. Importantly, all variables present in the data used to fit the model also need to be present in this DataFrame. Additionally, factor variables must only include levels also present in the data used to fit the model. If you want to exclude a specific factor from the prediction (for example the factor subject) don't include the terms that involve it in the ``use_terms`` argument.
        :type n_dat: pd.DataFrame
        :param alpha: The alpha level to use for the standard error calculation. Specifically, 1 - (``alpha``/2) will be used to determine the critical cut-off value according to a N(0,1).
        :type alpha: float, optional
        :param ci: Whether the standard error ``se`` for credible interval (CI; see  Wood, 2017) calculation should be returned. The CI is then [``pred`` - ``se``, ``pred`` + ``se``]
        :type ci: bool, optional
        :param whole_interval: Whether or not to adjuste the point-wise CI to behave like whole-function interval (based on Wood, 2017; section 6.10.2 and Simpson, 2016). Defaults to False. The CI is then [``pred`` - ``se``, ``pred`` + ``se``]
        :type whole_interval: bool, optional
        :param n_ps: How many samples to draw from the posterior in case the point-wise CI is adjusted to behave like a whole-function interval CI.
        :type n_ps: int, optional
        :param seed: Can be used to provide a seed for the posterior sampling step in case the point-wise CI is adjusted to behave like a whole-function interval CI.
        :type seed: int or None, optional
        :param par: The index corresponding to the parameter for which to make the prediction (e.g., 0 = mean), defaults to 0
        :type par: int, optional
        :raises ValueError: An error is raised in case the standard error is to be computed for a Multinomial GAMMLSS model, which is currently not supported.
        :return: A tuple with 3 entries. The first entry is the prediction ``pred`` based on the new data ``n_dat``. The second entry is the model matrix built for ``n_dat`` that was post-multiplied with the model coefficients to obtain ``pred``. The third entry is ``None`` if ``ci``==``False`` else the standard error ``se`` in the prediction.
        :rtype: (np.ndarray,scp.sparse.csc_array,np.ndarray or None)
        """
        return super().predict(use_terms,n_dat,alpha,ci,whole_interval,n_ps,seed,par)
    
    def predict_diff(self, dat1:pd.DataFrame, dat2:pd.DataFrame, use_terms:list[int]|None, alpha:float=0.05, whole_interval:bool=False, n_ps:int=10000, seed:int|None=None, par:int=0) -> tuple[np.ndarray,np.ndarray]:
        """
        Get the difference in the predictions for two datasets and for distribution parameter ``par``. Useful to compare a smooth estimated for
        one level of a factor to the smooth estimated for another level of a factor. In that case, ``dat1`` and
        ``dat2`` should only differ in the level of said factor. Importantly, predictions and standard errors are again always returned on the scale of the linear predictor - 
        see the :func:`predict` method for details.

        Examples::

            from mssm.models import *
            from mssmViz.sim import *
            from mssmViz.plot import *
            import matplotlib.pyplot as plt

            # Simulate 500 data points
            GAUMLSSDat = sim9(500,1,seed=20)

            # We include a tensor smooth in the model of the mean
            formula_m = Formula(lhs("y"),
                                [i(),f(["x0","x1"],te=True)],
                                data=GAUMLSSDat)

            # The model of the standard deviation remains the same
            formula_sd = Formula(lhs("y"),
                                [i(),f(["x0"])],
                                data=GAUMLSSDat)

            # Collect both formulas
            formulas = [formula_m,formula_sd]

            # Create Gaussian GAMMLSS family with identity link for mean
            # and log link for sigma
            family = GAUMLSS([Identity(),LOG()])

            # Now fit
            model = GAMMLSS(formulas,family)
            model.fit()

            # Now we want to know whether the effect of x0 is different for two values of x1:
            new_dat1 = pd.DataFrame({"x0":np.linspace(0,1,30),
                                    "x1":[0.25 for _ in range(30)]})

            new_dat2 = pd.DataFrame({"x0":np.linspace(0,1,30),
                                    "x1":[0.75 for _ in range(30)]})

            # Now we can get the predicted difference of the effect of x0 for the two values of x1:
            pred_diff,se = model.predict_diff(new_dat1,new_dat2,use_terms=[1],par=0)

            # mssmViz also has a convenience function to visualize it:
            plot_diff(new_dat1,new_dat2,["x0"],model,use=[1],response_scale=False)

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
         - Simpson, G. (2016). Simultaneous intervals for smooths revisited.
         - ``get_difference`` function from ``itsadug`` R-package: https://rdrr.io/cran/itsadug/man/get_difference.html

        :param dat1: A pandas DataFrame containing new data for which to make the prediction. Importantly, all variables present in the data used to fit the model also need to be present in this DataFrame. Additionally, factor variables must only include levels also present in the data used to fit the model. If you want to exclude a specific factor from the prediction (for example the factor subject) don't include the terms that involve it in the `use_terms` argument.
        :type dat1: pd.DataFrame
        :param dat2: A second pandas DataFrame for which to also make a prediction. The difference in the prediction between this `dat1` will be returned.
        :type dat2: pd.DataFrame
        :param use_terms: The indices corresponding to the terms in the formula of the parameter that should be used to obtain the prediction or ``None`` in which case all terms will be used.
        :type use_terms: list[int] or None
        :param alpha: The alpha level to use for the standard error calculation. Specifically, 1 - (`alpha`/2) will be used to determine the critical cut-off value according to a N(0,1).
        :type alpha: float, optional
        :param whole_interval: Whether or not to adjuste the point-wise CI to behave like whole-function interval (based on Wood, 2017; section 6.10.2 and Simpson, 2016). Defaults to False.
        :type whole_interval: bool, optional
        :param n_ps: How many samples to draw from the posterior in case the point-wise CI is adjusted to behave like a whole-function interval CI.
        :type n_ps: int, optional
        :param seed: Can be used to provide a seed for the posterior sampling step in case the point-wise CI is adjusted to behave like a whole-function interval CI.
        :type seed: int or None, optional
        :param par: The index corresponding to the parameter for which to make the prediction (e.g., 0 = mean), defaults to 0
        :type par: int, optional
        :raises ValueError: An error is raised in case the predicted difference is to be computed for a Multinomial GAMMLSS model, which is currently not supported.
        :return: A tuple with 2 entries. The first entry is the predicted difference (between the two data sets ``dat1`` & ``dat2``) ``diff``. The second entry is the standard error ``se`` of the predicted difference. The difference CI is then [``diff`` - ``se``, ``diff`` + ``se``]
        :rtype: (np.ndarray,np.ndarray)
        """
        return super().predict_diff(dat1,dat2,use_terms,alpha,whole_interval,n_ps,seed,par)
    
##################################### GAMM class #####################################

class GAMM(GAMMLSS):
    """Class to fit Generalized Additive Mixed Models.

    Examples::

        from mssm.models import *
        from mssmViz.sim import *
        from mssmViz.plot import *
        import matplotlib.pyplot as plt

        #### Binomial model example ####
        Binomdat = sim3(10000,0.1,family=Binomial(),seed=20)

        formula = Formula(lhs("y"),[i(),f(["x0"]),f(["x1"]),f(["x2"]),f(["x3"])],data=Binomdat)

        # By default, the Binomial family assumes binary data and uses the logit link.
        # Count data is also possible though - see the `Binomial` family.
        model = GAMM(formula,Binomial())
        model.fit()

        # Plot estimated effects on scale of the log-odds
        plot(model)

        #### Gaussian model with tensor smooth and p-values ####
        sim_dat = sim3(n=500,scale=2,c=0,seed=20)

        formula = Formula(lhs("y"),[i(),f(["x0","x3"],te=True,nk=9),f(["x1"]),f(["x2"])],data=sim_dat)
        model = GAMM(formula,Gaussian())

        model.fit()
        model.print_smooth_terms(p_values=True)


        #### Standard linear (mixed) models are also possible ####
        # *li() with three variables: three-way interaction
        sim_dat,_ = sim1(100,random_seed=100)

        # Specify formula with three-way linear interaction and random intercept term
        formula = Formula(lhs("y"),[i(),*li(["fact","x","time"]),ri("sub")],data=sim_dat)

        # ... and model
        model = GAMM(formula,Gaussian())

        # then fit
        model.fit()

        # get estimates for linear terms
        model.print_parametric_terms()

    References:
     - Wood, S. N., & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. https://doi.org/10.1111/biom.12666
     - Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models: Estimation of Semiparametric Generalized Linear Models. https://doi.org/10.1111/j.1467-9868.2010.00749.x
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

    :param formula: A formula for the GAMM model
    :type formula: Formula
    :param family: A distribution implementing the :class:`Family` class. Currently :class:`Gaussian`, :class:`Gamma`, and :class:`Binomial` are implemented.
    :type family: Family
    :ivar [Formula] formulas: A list including the formula passed to the constructor.
    :ivar scp.sparse.csc_array lvi: The inverse of the Cholesky factor of the conditional model coefficient covariance matrix. Initialized with ``None``.
    :ivar np.ndarray coef:  Contains all coefficients estimated for the model. Shape of the array is (-1,1). Initialized with ``None``.
    :ivar [[float]] preds: The first index corresponds to the linear predictors for the mean of the family evaluated for each observation in the training data (after removing NaNs). Initialized with ``None``.
    :ivar [[float]] mus: The first index corresponds to the estimated value of the mean of the family evaluated for each observation in the training data (after removing NaNs). Initialized with ``None``.
    :ivar scp.sparse.csc_array hessian: Estimated hessian of the log-likelihood used during fitting - will be the expected hessian for non-canonical models. Initialized with ``None``.
     :ivar float edf: The model estimated degrees of freedom as a float. Initialized with ``None``.
    :ivar float edf1: The model estimated degrees of freedom as a float corrected for smoothness bias. Set by the :func:`approx_smooth_p_values` function, the first time it is called. Initialized with ``None``.
    :ivar [float] term_edf: The estimated degrees of freedom per smooth term. Initialized with ``None``.
    :ivar [float] term_edf1: The estimated degrees of freedom per smooth term corrected for smoothness bias. Set by the :func:`approx_smooth_p_values` function, the first time it is called. Initialized with ``None``.
    :ivar float penalty: The total penalty applied to the model deviance after fitting as a float. Initialized with ``None``.
    :ivar [LambdaTerm] overall_penalties:  Contains all penalties estimated for the model. Initialized with ``None``.
    :ivar Fit_info info: A :class:`Fit_info` instance, with information about convergence (speed) of the model.
    :ivar np.ndarray res: The working residuals of the model (If applicable). Initialized with ``None``.
    :ivar scp.sparse.csc_array Wr: For generalized models a diagonal matrix holding the root of the Fisher weights at convergence. Initialized with ``None``.
    :ivar scp.sparse.csc_array WN: For generalized models a diagonal matrix holding the Newton weights at convergence. Initialized with ``None``.
    :ivar scp.sparse.csc_array hessian_obs: Observed hessian of the log-likelihood at final coefficient estimate. Not updated for strictly additive models (i.e., Gaussian with identity link). Initialized with ``None``.
    :ivar float rho: Optional auto-correlation at lag 1 parameter used during estimation. Initialized with ``None``.
    :ivar np.ndarray res_ar: Holding the working residuals of the model corrected for any auto-correlation parameter used during estimation. Initialized with ``None``.
    """

    def __init__(self,
                 formula: Formula,
                 family: Family):

        super().__init__([formula],family)

        self.Wr = None
        self.WN = None
        self.hessian_obs = None
        self.rho = None
        self.res_ar = None


    ##################################### Getters #####################################

    def get_pars(self) -> tuple[np.ndarray|None,float|None]:
        """
        Returns a tuple. The first entry is a np.ndarray with all estimated coefficients. The second entry is the estimated scale parameter.
        
        Will instead return ``(None,None)`` if called before fitting.

        :return: Model coefficients and scale parameter that were estimated
        :rtype: (np.ndarray,float) or (None, None)
        """
        return self.coef,self.scale

    def get_mmat(self,use_terms:list[int]|None=None) -> scp.sparse.csc_array:
        """
        Returns exaclty the model matrix used for fitting as a scipy.sparse.csc_array. Will throw an error when called for a model for which the model
        matrix was never former completely - i.e., when :math:`\\mathbf{X}^T\\mathbf{X}` was formed iteratively for estimation, by setting the ``file_paths`` argument of the ``Formula`` to
        a non-empty list.
        
        Optionally, all columns not corresponding to terms for which the indices are provided via ``use_terms`` can be zeroed.

        :param use_terms: Optionally provide indices of terms in the formual that should be created. If this argument is provided columns corresponding to any term not included in this list will be zeroed, defaults to None
        :type use_terms: [int], optional
        :raises ValueError: Will throw an error when called before the model was fitted/before model penalties were formed.
        :raises NotImplementedError: Will throw an error when called for a model for which the model matrix was never former completely
        :return: Model matrix :math:`\\mathbf{X}` used for fitting.
        :rtype: scp.sparse.csc_array
        """
        
        if len(self.formulas[0].file_paths) != 0:
            raise NotImplementedError("Cannot return the model-matrix if X.T@X was formed iteratively.")
        
        return super().get_mmat(use_terms,par=0)
    
    def get_llk(self,penalized:bool=True,ext_scale:float|None=None) -> float|None:
        """
        Get the (penalized) log-likelihood of the estimated model (float or None) given the trainings data. LLK can optionally be evaluated for an external scale parameter ``ext_scale``.
        
        Will instead return ``None`` if called before fitting.
        
        :param penalized: Whether the penalized log-likelihood should be returned or the regular log-likelihood, defaults to True
        :type penalized: bool, optional
        :param ext_scale: Optionally provide an external scale parameter at which to evaluate the log-likelihood, defaults to None
        :type ext_scale: float, optional
        :raises NotImplementedError: Will throw an error when called for a model for which the model matrix was never former completely.
        :return: llk score
        :rtype: float or None
        """

        if len(self.formulas[0].file_paths) != 0:
            raise NotImplementedError("Cannot return the log-likelihood if X.T@X was formed iteratively.")

        pen = 0
        if penalized:
            pen = 0.5*self.penalty
        if self.preds is not None:
            mu = self.preds[0]
            if isinstance(self.family,Gaussian) == False or isinstance(self.family.link,Identity) == False:
                mu = self.family.link.fi(self.preds[0])
            
            y = self.formulas[0].y_flat[self.formulas[0].NOT_NA_flat]
            if not self.formulas[0].get_lhs().f is None:
                y = self.formulas[0].get_lhs().f(y)

            if self.rho is not None:
                # Need to correct for the "weights" of the covariance matrix of the ar1 model, as done in the bam function in
                # mgcv. see: https://github.com/cran/mgcv/blob/fb7e8e718377513e78ba6c6bf7e60757fc6a32a9/R/bam.r#L2761
                _,ar_cor = computeAr1Chol(self.formulas[0],self.rho)
                pen -= ar_cor

            if self.family.twopar:
                scale = self.scale
                if not ext_scale is None:
                    scale = ext_scale

                if self.rho is not None and isinstance(self.family,Gaussian) and isinstance(self.family.link,Identity):
                    y = self.res_ar
                    mu = 0

                return self.family.llk(y,mu,scale) - pen
            else:
                return self.family.llk(y,mu) - pen
        return None

    def get_reml(self) -> float:
        """
        Get's the (Laplace approximate) REML (Restricted Maximum Likelihood) score (as a float) for the estimated lambda values (see Wood, 2011).

        References:

         - Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models: Estimation of Semiparametric Generalized Linear Models.
        
        :raises ValueError: Will throw an error when called before the model was fitted/before model penalties were formed.
        :raises NotImplementedError: Will throw an error when called for a model for which the model matrix was never former completely.
        :raises TypeError: Will throw an error when called before the model was fitted/before model penalties were formed.
        :return: REML score
        :rtype: float
        """
        if (not isinstance(self.family,Gaussian) or isinstance(self.family.link,Identity) == False) and self.Wr is None:
            raise TypeError("Model is not Normal and pseudo-dat weights are not avilable. Call model.fit() first!")
        
        if self.coef is None or self.overall_penalties is None:
            raise ValueError("Model needs to be estimated before evaluating the REML score. Call model.fit()")
        
        scale = self.family.scale
        if self.family.twopar:
            scale = self.scale # Estimated scale
        
        llk = self.get_llk(False)

        # Compute negative Hessian of llk (Wood, 2011)
        nH = -1 * self.hessian

        keep = None
        if self.info.dropped is not None:
            keep = [cidx for cidx in range(self.hessian.shape[1]) if cidx not in self.info.dropped]
            
        reml = REML(llk,nH,self.coef,scale,self.overall_penalties,keep)[0,0]
        
        return reml
    
    def get_resid(self,type:str='Pearson') -> np.ndarray:
        """
        Get different types of residuals from the estimated model.

        By default (``type='Pearson'``) this returns the residuals :math:`e_i = y_i - \\mu_i` for additive models and the pearson/working residuals :math:`w_i^{0.5}*(z_i - \\eta_i)` (see Wood, 2017 sections 3.1.5 & 3.1.7) for
        generalized additive models. Here :math:`w_i` are the Fisher scoring weights, :math:`z_i` the pseudo-data point for each observation, and :math:`\\eta_i` is the linear prediction (i.e., :math:`g(\\mu_i)` - where :math:`g()`
        is the link function) for each observation.

        If ``type= "Deviance"``, the deviance residuals are returned, which are equivalent to :math:`sign(y_i - \\mu_i)*D_i^{0.5}`, where :math:`\\sum_{i=1,...N} D_i` equals the model deviance (see Wood 2017, section 3.1.7). Additionally,
        if the model was estimated with ``rho!=None``, ``type="ar1"`` returns the standardized working residuals corrected for lag1 auto-correlation. These are best compared to the standard working residuals.

        Throws an error if called before model was fitted, when requesting an unsupported type, or when requesting 'ar1' residuals for a model for which ``model.rho==None``.

        References:

         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

    
        :param type: The type of residual to return for a Generalized model, "Pearson" by default, but can be set to "Deviance" and (for some models) to "ar1" as well.
        :type type: str,optional
        :raises ValueError: Will throw an error when called before the model was fitted/before model penalties were formed, when requesting an unsupported type, or when requesting 'ar1' residuals for a model for which ``model.rho==None``.
        :return: Empirical residual vector in a numpy array
        :rtype: np.ndarray
        """
        if self.res is None or self.preds is None:
            raise ValueError("Model needs to be estimated before evaluating the residuals. Call model.fit()")
        
        if type not in ["Pearson", "Deviance", "ar1", "family"]:
            raise ValueError("Type must be one of 'Pearson','Deviance', or 'ar1'.")
        
        if type == "ar1" and self.rho is None:
            raise ValueError("ar1 residuals are only available if the model was estimated with ``model.fit(..., rho=<some value>)``.")
        
        if type == "Pearson":
            return self.res
        
        elif type == "ar1":
            return self.res_ar
        
        elif type == "Deviance":
            # Deviance residual requires computing quantity D_i, which is the amount each data-point contributes to
            # overall deviance. Implemented by the family members.
            mu = self.preds[0]
            y = self.formulas[0].y_flat[self.formulas[0].NOT_NA_flat]

            if not isinstance(self.family,Gaussian) or isinstance(self.family.link,Identity) == False:
                mu = self.family.link.fi(mu)

            return np.sign(y - mu) * np.sqrt(self.family.D(y,mu))
    
    ##################################### Summary #####################################
        
    def print_parametric_terms(self):
        """Prints summary output for linear/parametric terms in the model, not unlike the one returned in R when using the ``summary`` function
        for ``mgcv`` models.
        
        For each coefficient, the named identifier and estimated value are returned. In addition, for each coefficient a p-value is returned, testing
        the null-hypothesis that the corresponding coefficient :math:`\\beta=0`. Under the assumption that this is true, the Null distribution follows
        a t-distribution for models in which an additional scale parameter was estimated (e.g., Gaussian, Gamma) and a standardized normal distribution for
        models in which the scale parameter is known or was fixed (e.g., Binomial). For the former case, the t-statistic, Degrees of freedom of the Null
        distribution (DoF.), and the p-value are printed as well. For the latter case, only the z-statistic and the p-value are printed.
        See Wood (2017) section 6.12 and 1.3.3 for more details.

        Note that, un-penalized coefficients that are part of a smooth function are not covered by this function.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

        :raises NotImplementedError: Will throw an error when called for a model for which the model matrix was never former completely.
        """
        print_parametric_terms(self)
    
    def print_smooth_terms(self, pen_cutoff:float=0.2, p_values:bool=False, edf1:bool=True):
        """Prints the name of the smooth terms included in the model. After fitting, the estimated degrees of freedom per term are printed as well.
        Smooth terms with edf. < ``pen_cutoff`` will be highlighted. This only makes sense when extra Kernel penalties are placed on smooth terms to enable
        penalizing them to a constant zero. In that case edf. < ``pen_cutoff`` can then be taken as evidence that the smooth has all but notationally disappeared
        from the model, i.e., it does not contribute meaningfully to the model fit. This can be used as an alternative form of model selection - see Marra & Wood (2011).

        References:

         - Marra & Wood (2011). Practical variable selection for generalized additive models.

        :param pen_cutoff: At which edf. cut-off smooth terms should be marked as "effectively removed", defaults to None
        :type pen_cutoff: float, optional
        :param p_values: Whether approximate p-values should be printed for the smooth terms, defaults to False
        :type p_values: bool, optional
        :param edf1: Whether or not the estimated degrees of freedom should be corrected for smoothnes bias. Doing so results in more accurate p-values but can be expensive for large models for which the difference is anyway likely to be marginal, defaults to False
        :type edf1: bool, optional
        """
        ps = None
        Trs = None
        if p_values:
            ps, Trs = approx_smooth_p_values(self,edf1=edf1)

        print_smooth_terms(self,pen_cutoff=pen_cutoff,ps=ps,Trs=Trs)
                
    ##################################### Fitting #####################################
    
    def fit(self,max_outer:int=200,max_inner:int=None,conv_tol:float=1e-7,extend_lambda:bool=False,control_lambda:int=2,exclude_lambda:bool=False,
            extension_method_lam:str = "nesterov",restart:bool=False,method:str="QR",check_cond:int=1,progress_bar:bool=True,n_cores:int=10,
            offset:float|np.ndarray|None = None,rho:float|None=None):
        """
        Fit the specified model.
        
        **Note**: Keyword arguments are initialized to maximise stability. For faster configurations (necessary for larger models) see the 'Big model' example below.

        Examples::

            from mssm.models import *
            from mssmViz.sim import *
            from mssmViz.plot import *

            ########## Big Model ##########
            dat = pd.read_csv('https://raw.githubusercontent.com/JoKra1/mssmViz/main/data/GAMM/sim_dat.csv')

            # mssm requires that the data-type for variables used as factors is 'O'=object
            dat = dat.astype({'series': 'O',
                            'cond':'O',
                            'sub':'O',
                            'series':'O'})

            formula = Formula(lhs=lhs("y"), # The dependent variable - here y!
                                terms=[i(), # The intercept, a
                                        l(["cond"]), # For cond='b'
                                        f(["time"],by="cond",constraint=ConstType.QR), # to-way interaction between time and cond; one smooth over time per cond level
                                        f(["x"],by="cond",constraint=ConstType.QR), # to-way interaction between x and cond; one smooth over x per cond level
                                        f(["time","x"],by="cond",constraint=ConstType.QR,nk=9), # three-way interaction
                                        fs(["time"],rf="sub")], # Random non-linear effect of time - one smooth per level of factor sub
                                data=dat,
                                print_warn=False,find_nested=False)
                
            model = GAMM(formula,Gaussian())

            # To speed up estimation, use the following key-word arguments:
            model.fit(method="Chol",max_inner=1) # max_inner only matters for Generalized models (i.e., non-Gaussian) - but for those will often be much faster

            ########## ar1 model (without resets per time-series) ##########
            formula = Formula(lhs=lhs("y"),
                                terms=[i(),
                                        l(["cond"]),
                                        f(["time"],by="cond"),
                                        f(["x"],by="cond"),
                                        f(["time","x"],by="cond")],
                                data=dat,
                                print_warn=False,
                                series_id=None) # No series identifier passed to formula -> ar1 model does not reset!

            model = GAMM(formula,Gaussian())

            model.fit(rho=0.99)

            # Visualize the un-corrected residuals:
            plot_val(model,resid_type="Pearson")

            # And the corrected residuals:
            plot_val(model,resid_type="ar1")

            ########## ar1 model (with resets per time-series) ##########
            formula = Formula(lhs=lhs("y"),
                                terms=[i(),
                                        l(["cond"]),
                                        f(["time"],by="cond"),
                                        f(["x"],by="cond"),
                                        f(["time","x"],by="cond")],
                                data=dat,
                                print_warn=False,
                                series_id='series') # 'series' variable identifies individual time-series -> ar1 model resets per series!

            model = GAMM(formula,Gaussian())

            model.fit(rho=0.99)

            # Visualize the un-corrected residuals:
            plot_val(model,resid_type="Pearson")

            # And the corrected residuals:
            plot_val(model,resid_type="ar1")

        :param max_outer: The maximum number of fitting iterations. Defaults to 200.
        :type max_outer: int,optional
        :param max_inner: The maximum number of fitting iterations to use by the inner Newton step updating the coefficients for Generalized models. Defaults to 500 for non ar1 models.
        :type max_inner: int,optional
        :param conv_tol: The relative (change in penalized deviance is compared against ``conv_tol`` * previous penalized deviance) criterion used to determine convergence.
        :type conv_tol: float,optional
        :param extend_lambda: Whether lambda proposals should be accelerated or not. Can lower the number of new smoothing penalty proposals necessary. Disabled by default.
        :type extend_lambda: bool,optional
        :param control_lambda: Whether lambda proposals should be checked (and if necessary decreased) for whether or not they (approxiately) increase the Laplace approximate restricted maximum likelihood of the model. Setting this to 0 disables control. Setting it to 1 means the step will never be smaller than the original EFS update but extensions will be removed in case the objective was exceeded. Setting it to 2 means that steps will be halved if it fails to increase the approximate REML. Set to 2 by default.
        :type control_lambda: int,optional
        :param exclude_lambda: Whether selective lambda terms should be excluded heuristically from updates. Can make each iteration a bit cheaper but is problematic when using additional Kernel penalties on terms. Thus, disabled by default.
        :type exclude_lambda: bool,optional
        :param extension_method_lam: **Experimental - do not change!** Which method to use to extend lambda proposals. Set to 'nesterov' by default.
        :type extension_method_lam: str,optional
        :param restart: Whether fitting should be resumed. Only possible if the same model has previously completed at least one fitting iteration.
        :type restart: bool,optional
        :param method: Which method to use to solve for the coefficients. ("Chol") relies on Cholesky decomposition. This is extremely efficient but in principle less stable, numerically speaking. For a maximum of numerical stability set this to "QR". In that case a QR decomposition is used - which is first pivoted to maximize sparsity in the resulting decomposition but then also pivots for stability in order to get an estimate of rank defficiency. This takes substantially longer. This argument is ignored if ``len(self.formulas[0].file_paths)>0`` that is, if :math:`\\mathbf{X}^T\\mathbf{X}` and :math:`\\mathbf{X}^T\\mathbf{y}` should be created iteratively. Defaults to "QR".
        :type method: str,optional
        :param check_cond: Whether to obtain an estimate of the condition number for the linear system that is solved. When ``check_cond=0``, no check will be performed. When ``check_cond=1``, an estimate of the condition number for the final system (at convergence) will be computed and warnings will be issued based on the outcome (see :func:`mssm.src.python.gamm_solvers.est_condition`). When ``check_cond=2``, an estimate of the condition number will be performed for each new system (at each iteration of the algorithm) and an error will be raised if the condition number is estimated as too high given the chosen ``method``. Is ignored, if :math:`\\mathbf{X}^T\\mathbf{X}` and :math:`\\mathbf{X}^T\\mathbf{y}` should be created iteratively. Defaults to 1.
        :type check_cond: int,optional
        :param progress_bar: Whether progress should be displayed (convergence info and time estimate). Defaults to True.
        :type progress_bar: bool,optional
        :param n_cores: Number of cores to use during parts of the estimation that can be done in parallel. Defaults to 10.
        :type n_cores: int,optional
        :param offset: Mimics the behavior of the ``offset`` argument for ``gam`` in ``mgcv`` in R. If a value is provided here (can either be a float or a numpy.array of shape (-1,1) - if it is an array, then the first dimension has to match the number of observations in the data. NANs present in the dependent variable will be excluded from the offset vector.) then it is consistently added to the linear predictor during estimation. It will **not** be used by any other function of the :class:`GAMM` class (e.g., for prediction). This argument is ignored if ``len(self.formulas[0].file_paths)>0`` that is, if :math:`\\mathbf{X}^T\\mathbf{X}` and :math:`\\mathbf{X}^T\\mathbf{y}` should be created iteratively. Defaults to None.
        :type offset: float or np.ndarray,optional
        :param rho: Optional correlation parameter for an "ar1 residual model". Essentially mimics the behavior of the ``rho`` paramter for the ``bam`` function in ``mgcv``. **Note**, if you want to re-start the ar1 process multiple times (for example because you work with time-series data and have multiple time-series) then you must pass the ``series.id`` argument to the :class:`Formula` used for this model. Defaults to None.
        :type rho: float,optional
        """

        # Initialize remaining arguments to defaults
        if max_inner is None:
            if rho is not None:
                max_inner = 1
            else:
                max_inner = 500

        # We need to initialize penalties
        if not restart:
            if self.overall_penalties is not None:
                warnings.warn("Resetting penalties. If you don't want that, set ``restart=True``.")
            self.overall_penalties = build_penalties(self.formulas[0])
        penalties = self.overall_penalties
        
        # Some checks
        if penalties is None and restart:
            raise ValueError("Penalties were not initialized. ``Restart`` must be set to False.")
        
        if len(self.formulas[0].discretize) != 0 and method != "Chol":
            raise ValueError("When discretizing derivative code, the method argument must be set to 'Chol'.")
        
        if len(self.formulas[0].file_paths) != 0 and rho is not None:
            raise ValueError("ar1 model of the residuals is not supported when iteratviely building the model matrix.")
        
        if max_inner > 1 and rho is not None:
            raise ValueError("ar1 model of the residuals only supported for ``max_inner=1``.")
        
        self.offset = 0

        if len(self.formulas[0].file_paths) == 0:
            # We need to build the model matrix once
            terms = self.formulas[0].terms
            has_intercept = self.formulas[0].has_intercept
            ltx = self.formulas[0].get_linear_term_idx()
            irstx = self.formulas[0].get_ir_smooth_term_idx()
            stx = self.formulas[0].get_smooth_term_idx()
            rtx = self.formulas[0].get_random_term_idx()
            var_types = self.formulas[0].get_var_types()
            var_map = self.formulas[0].get_var_map()
            var_mins = self.formulas[0].get_var_mins()
            var_maxs = self.formulas[0].get_var_maxs()
            factor_levels = self.formulas[0].get_factor_levels()

            cov_flat = self.formulas[0].cov_flat[self.formulas[0].NOT_NA_flat]
            
            if len(irstx) > 0:
                cov_flat = self.formulas[0].cov_flat # Need to drop NA rows **after** building!
                cov = self.formulas[0].cov
            else:
                cov = None

            y_flat = self.formulas[0].y_flat[self.formulas[0].NOT_NA_flat]

            # Offset handling - for strictly additive model just subtract from y and then pass zero
            # via self.offset (default initialization)
            if offset is not None:

                # First drop NANs
                if isinstance(offset,np.ndarray):
                    offset = offset[self.formulas[0].NOT_NA_flat]

                if not isinstance(self.family,Gaussian) or isinstance(self.family.link,Identity) == False:
                    self.offset = offset
                else:
                    y_flat -= offset

            if not self.formulas[0].get_lhs().f is None:
                # Optionally apply function to dep. var. before fitting.
                y_flat = self.formulas[0].get_lhs().f(y_flat)

            if y_flat.shape[0] != self.formulas[0].y_flat.shape[0] and progress_bar:
                print("NAs were excluded for fitting.")

            # Build the model matrix with all information from the formula
            if self.formulas[0].file_loading_nc == 1:
                model_mat = build_sparse_matrix_from_formula(terms,has_intercept,
                                                            ltx,irstx,stx,rtx,var_types,var_map,
                                                            var_mins,var_maxs,factor_levels,
                                                            cov_flat,cov)
            
            else:
                # Build row sets of model matrix in parallel:
                
                cov_split = np.array_split(cov_flat,self.formulas[0].file_loading_nc,axis=0)
                with mp.Pool(processes=self.formulas[0].file_loading_nc) as pool:
                    # Build the model matrix with all information from the formula - but only for sub-set of rows
                    Xs = pool.starmap(build_sparse_matrix_from_formula,zip(repeat(terms),repeat(has_intercept),
                                                                           repeat(ltx),repeat(irstx),repeat(stx),repeat(rtx),
                                                                           repeat(var_types),repeat(var_map),repeat(var_mins),
                                                                           repeat(var_maxs),repeat(factor_levels),cov_split,
                                                                           repeat(cov)))
                    
                    model_mat = scp.sparse.vstack(Xs,format='csc')

            if len(irstx) > 0:
                model_mat = model_mat[self.formulas[0].NOT_NA_flat,:]

            # Get initial estimate of mu based on family:
            init_mu_flat = self.family.init_mu(y_flat)

            # Optionally set up ar1 model
            Lrhoi = None
            if rho is not None:
                self.rho = rho
                Lrhoi,_ = computeAr1Chol(self.formulas[0],rho)
            
            # Now we have to estimate the model
            coef,eta,wres,Wr,WN,scale,LVI,edf,term_edf,penalty,fit_info = solve_gamm_sparse(init_mu_flat,y_flat,
                                                                                      model_mat,penalties,self.formulas[0].n_coef,
                                                                                      self.family,max_outer,max_inner,"svd",
                                                                                      conv_tol,extend_lambda,control_lambda,
                                                                                      exclude_lambda,extension_method_lam,
                                                                                      len(self.formulas[0].discretize) == 0,
                                                                                      method,check_cond,progress_bar,n_cores,
                                                                                      self.offset,Lrhoi)
            
            self.Wr = Wr
            self.WN = WN

            if fit_info.dropped is not None: # Make sure hessian has zero columns/rows for unidentifiable coef.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model_mat[:,fit_info.dropped] = 0

            # Compute (expected) Hessian of llk (Wood, 2011)
            if not isinstance(self.family,Gaussian) or isinstance(self.family.link,Identity) == False:

                if rho is not None:
                    self.res_ar = wres[0]
                    wres = wres[1]

                    self.hessian = -1 * ((model_mat.T@(Wr@Lrhoi@Lrhoi.T@Wr)@model_mat).tocsc()/scale)

                else:
                    self.hessian = -1 * ((model_mat.T@(Wr@Wr)@model_mat).tocsc()/scale)

                    # Compute observed Hessian of llk
                    self.hessian_obs = -1 * ((model_mat.T@(WN)@model_mat).tocsc()/scale)

            else:
                if rho is not None:
                    self.hessian = -1 * ((model_mat.T@Lrhoi@Lrhoi.T@model_mat).tocsc()/scale)
                    eta = model_mat@coef.reshape(-1,1)
                    self.res_ar = wres
                    wres = y_flat.reshape(-1,1) - eta

                else:
                    self.hessian = -1 * ((model_mat.T@model_mat).tocsc()/scale)

                if offset is not None:
                    # Assign correct offset and re-adjust y_flat + eta
                    self.offset = offset
                    y_flat += offset
                    eta += offset
        
        else:
            # Iteratively build model matrix.
            # Follows steps in "Generalized additive models for large data sets" (2015) by Wood, Goude, and Shaw
            if not self.formulas[0].get_lhs().f is None:
                raise ValueError("Cannot apply function to dep. var. when building model matrix iteratively. Consider creating a modified variable in the data-frame.")
            
            if isinstance(self.family,Gaussian) == False or isinstance(self.family.link,Identity) == False:
                raise ValueError("Iteratively building the model matrix is currently only supported for Normal models.")
            
            coef,eta,wres,XX,scale,LVI,edf,term_edf,penalty,fit_info = solve_gamm_sparse2(self.formulas[0],penalties,self.formulas[0].n_coef,
                                                                                          self.family,max_outer,"svd",
                                                                                          conv_tol,extend_lambda,control_lambda,
                                                                                          exclude_lambda,extension_method_lam,
                                                                                          len(self.formulas[0].discretize) == 0,
                                                                                          progress_bar,n_cores)
            
            self.hessian = -1*(XX/scale)
        
        self.coef = coef.reshape(-1,1)
        self.scale = scale
        self.preds = [eta]
        self.mus = [self.family.link.fi(eta)]
        self.res = wres
        self.edf = edf
        self.term_edf = term_edf
        self.penalty = penalty
        self.info = fit_info
        self.lvi = LVI
    
    ##################################### Prediction #####################################

    def sample_post(self,n_ps:int,use_post:list[int]|None=None,deviations:bool=False,seed:int|None=None,par:int=0) -> np.ndarray:
        """
        Obtain ``n_ps`` samples from posterior :math:`[\\boldsymbol{\\beta} - \\hat{\\boldsymbol{\\beta}}] | \\mathbf{y},\\boldsymbol{\\lambda} \\sim N(0,\\mathbf{V})`,
        where V is :math:`[\\mathbf{X}^T\\mathbf{X} + \\mathbf{S}_{\\lambda}]^{-1}*/\\phi` (see Wood, 2017; section 6.10). To obtain samples for :math:`\\boldsymbol{\\beta}`,
        set ``deviations`` to false.

        see :func:`sample_MVN` for more details.

        Examples::

            from mssm.models import *
            from mssmViz.sim import *
            from mssmViz.plot import *
            import matplotlib.pyplot as plt

            # Fit a Gamma Gam
            Gammadat = sim3(500,2,family=Gamma(),seed=0)

            formula = Formula(lhs("y"),[i(),f(["x0"]),f(["x1"]),f(["x2"]),f(["x3"])],data=Gammadat)

            # By default, the Gamma family assumes that the model predictions match log(\\mu_i), i.e., a log-link is used.
            model = GAMM(formula,Gamma())
            model.fit()

            # Now get model matrix for a couple of example covariates
            new_dat = pd.DataFrame({"x0":np.linspace(0,1,30),
                                    "x1":np.linspace(0,1,30),
                                    "x2":np.linspace(0,1,30),
                                    "x3":np.linspace(0,1,30)})

            f0,X_f,ci = model.predict([1],new_dat,ci=True)

            # Get `use_post` to only identify coefficients related to `f(["x0"])` - that way we can efficiently sample the
            # posterior only for `f(["x0"])`. If you want to sample all coefficients, simply set `use_post=None`.
            use_post = X_f.sum(axis=0) != 0
            use_post = np.arange(0,X_f.shape[1])[use_post]
            print(use_post)

            # `use_post` can now be passed to `sample_post`:
            post = model.sample_post(10000,use_post,deviations=False,seed=0,par=0)

            # Since we set deviations to false post has coefficient samples and can simply be post-multiplied to
            # get samples of `f(["x0"])` - importantly, post has a different shape than X_f, so we need to account for that
            post_f = X_f[:,use_post] @ post

            # Note: samples are also on scale of linear predictor!
            plt.plot(new_dat["x0"],f0,color="black",linewidth=2)

            for sidx in range(50):
                plt.plot(new_dat["x0"],post_f[:,sidx],alpha=0.2)

            plt.show()


        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

        :param n_ps: Number of samples to obtain from posterior.
        :type n_ps: int,optional
        :param use_post: The indices corresponding to coefficients for which to actually obtain samples. By default all coefficients are sampled.
        :type use_post: [int],optional
        :param deviations: Whether to return samples of **deviations** from the estimated coefficients (i.e., :math:`\\boldsymbol{\\beta} - \\hat{\\boldsymbol{\\beta}}`) or actual samples of coefficients (i.e., :math:`\\boldsymbol{\\beta}`), defaults to False
        :type deviations: bool,optional
        :param seed: A seed to use for the sampling, defaults to None
        :type seed: int,optional
        :returns: An np.ndarray of dimension ``[len(use_post),n_ps]`` containing the posterior samples. Can simply be post-multiplied with model matrix :math:`\\mathbf{X}` to generate posterior **sample curves/predictions**.
        :rtype: np.ndarray
        """
        return super().sample_post(n_ps,use_post,deviations,seed,par=0)

    def predict(self, use_terms:list[int]|None, n_dat:pd.DataFrame,alpha:float=0.05,ci:bool=False,whole_interval:bool=False,n_ps:int=10000,seed:int|None=None,par:int=0) -> tuple[np.ndarray,scp.sparse.csc_array,np.ndarray|None]:
        """Make a prediction using the fitted model for new data ``n_dat``.
         
        But only using the terms indexed by ``use_terms``. Importantly, predictions and standard errors are always returned on the scale of the linear predictor.
        When estimating a Generalized Additive Model, the mean predictions and standard errors (often referred to as the 'response'-scale predictions) can be obtained
        by applying the link inverse function to the predictions and the CI-bounds on the linear predictor scale (DO NOT transform the standard error first and then add it to the
        transformed predictions - only on the scale of the linear predictor is the standard error additive). See examples below.

        Examples::

            from mssm.models import *
            from mssmViz.sim import *
            from mssmViz.plot import *
            import matplotlib.pyplot as plt

            # Fit a Gamma Gam
            Gammadat = sim3(500,2,family=Gamma(),seed=0)

            formula = Formula(lhs("y"),[i(),f(["x0"]),f(["x1"]),f(["x2"]),f(["x3"])],data=Gammadat)

            # By default, the Gamma family assumes that the model predictions match log(\\mu_i), i.e., a log-link is used.
            model = GAMM(formula,Gamma())
            model.fit()

            # Now make prediction for `f["x0"]`
            new_dat = pd.DataFrame({"x0":np.linspace(0,1,30),
                                    "x1":np.linspace(0,1,30),
                                    "x2":np.linspace(0,1,30),
                                    "x3":np.linspace(0,1,30)})

            f0,X_f,ci = model.predict([1],new_dat,ci=True)

            # Can also use the plot function from mssmViz
            plot(model,which=[1])

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
         - Simpson, G. (2016). Simultaneous intervals for smooths revisited.

        :param use_terms: The indices corresponding to the terms that should be used to obtain the prediction or ``None`` in which case all terms will be used.
        :type use_terms: list[int] or None
        :param n_dat: A pandas DataFrame containing new data for which to make the prediction. Importantly, all variables present in the data used to fit the model also need to be present in this DataFrame. Additionally, factor variables must only include levels also present in the data used to fit the model. If you want to exclude a specific factor from the prediction (for example the factor subject) don't include the terms that involve it in the ``use_terms`` argument.
        :type n_dat: pd.DataFrame
        :param alpha: The alpha level to use for the standard error calculation. Specifically, 1 - (``alpha``/2) will be used to determine the critical cut-off value according to a N(0,1).
        :type alpha: float, optional
        :param ci: Whether the standard error ``se`` for credible interval (CI; see  Wood, 2017) calculation should be returned. The CI is then [``pred`` - ``se``, ``pred`` + ``se``]
        :type ci: bool, optional
        :param whole_interval: Whether or not to adjuste the point-wise CI to behave like whole-function interval (based on Wood, 2017; section 6.10.2 and Simpson, 2016). Defaults to False. The CI is then [``pred`` - ``se``, ``pred`` + ``se``]
        :type whole_interval: bool, optional
        :param n_ps: How many samples to draw from the posterior in case the point-wise CI is adjusted to behave like a whole-function interval CI.
        :type n_ps: int, optional
        :param seed: Can be used to provide a seed for the posterior sampling step in case the point-wise CI is adjusted to behave like a whole-function interval CI.
        :type seed: int or None, optional
        :return: A tuple with 3 entries. The first entry is the prediction ``pred`` based on the new data ``n_dat``. The second entry is the model matrix built for ``n_dat`` that was post-multiplied with the model coefficients to obtain ``pred``. The third entry is ``None`` if ``ci``==``False`` else the standard error ``se`` in the prediction.
        :rtype: (np.ndarray,scp.sparse.csc_array,np.ndarray or None)
        """
        return super().predict(use_terms,n_dat,alpha,ci,whole_interval,n_ps,seed,0)
        
    def predict_diff(self,dat1:pd.DataFrame,dat2:pd.DataFrame,use_terms:list[int]|None,alpha:float=0.05,whole_interval:bool=False,n_ps:int=10000,seed:int|None=None,par:int=0) -> tuple[np.ndarray,np.ndarray]:
        """Get the difference in the predictions for two datasets.
        
        Useful to compare a smooth estimated for one level of a factor to the smooth estimated for another
        level of a factor. In that case, ``dat1`` and ``dat2`` should only differ in the level of said factor.
        Importantly, predictions and standard errors are again always returned on the scale of the linear predictor - 
        see the :func:`predict` method for details.

        Examples::

            from mssm.models import *
            from mssmViz.sim import *
            from mssmViz.plot import *
            import matplotlib.pyplot as plt

            # Fit a Gamma Gam
            Gammadat = sim3(500,2,family=Gamma(),seed=0)

            # Include tensor smooth in model of log(mean)
            formula = Formula(lhs("y"),[i(),f(["x0","x1"],te=True),f(["x2"]),f(["x3"])],data=Gammadat)

            # By default, the Gamma family assumes that the model predictions match log(\\mu_i), i.e., a log-link is used.
            model = GAMM(formula,Gamma())
            model.fit()

            # Now we want to know whether the effect of x0 is different for two values of x1:
            new_dat1 = pd.DataFrame({"x0":np.linspace(0,1,30),
                                    "x1":[0.25 for _ in range(30)],
                                    "x2":np.linspace(0,1,30),
                                    "x3":np.linspace(0,1,30)})

            new_dat2 = pd.DataFrame({"x0":np.linspace(0,1,30),
                                    "x1":[0.75 for _ in range(30)],
                                    "x2":np.linspace(0,1,30),
                                    "x3":np.linspace(0,1,30)})

            # Now we can get the predicted difference of the effect of x0 for the two values of x1:
            pred_diff,se = model.predict_diff(new_dat1,new_dat2,use_terms=[1],par=0)

            # mssmViz also has a convenience function to visualize it:
            plot_diff(new_dat1,new_dat2,["x0"],model,use=[1],response_scale=False)

        References:

         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
         - Simpson, G. (2016). Simultaneous intervals for smooths revisited.
         - ``get_difference`` function from ``itsadug`` R-package: https://rdrr.io/cran/itsadug/man/get_difference.html

        :param dat1: A pandas DataFrame containing new data for which to make the prediction. Importantly, all variables present in the data used to fit the model also need to be present in this DataFrame. Additionally, factor variables must only include levels also present in the data used to fit the model. If you want to exclude a specific factor from the prediction (for example the factor subject) don't include the terms that involve it in the ``use_terms`` argument.
        :type n_dat: pd.DataFrame
        :param dat2: A second pandas DataFrame for which to also make a prediction. The difference in the prediction between this ``dat1`` will be returned.
        :type dat2: pd.DataFrame
        :param use_terms: The indices corresponding to the terms that should be used to obtain the prediction or ``None`` in which case all terms will be used.
        :type use_terms: list[int] or None
        :param alpha: The alpha level to use for the standard error calculation. Specifically, 1 - (``alpha``/2) will be used to determine the critical cut-off value according to a N(0,1).
        :type alpha: float, optional
        :param whole_interval: Whether or not to adjuste the point-wise CI to behave like whole-function interval (based on Wood, 2017; section 6.10.2 and Simpson, 2016). Defaults to False.
        :type whole_interval: bool, optional
        :param n_ps: How many samples to draw from the posterior in case the point-wise CI is adjusted to behave like a whole-function interval CI.
        :type n_ps: int, optional
        :param seed: Can be used to provide a seed for the posterior sampling step in case the point-wise CI is adjusted to behave like a whole-function interval CI.
        :type seed: int or None, optional
        :return: A tuple with 2 entries. The first entry is the predicted difference (between the two data sets ``dat1`` & ``dat2``) ``diff``. The second entry is the standard error ``se`` of the predicted difference. The difference CI is then [``diff`` - ``se``, ``diff`` + ``se``]
        :rtype: (np.ndarray,np.ndarray)
        """
        return super().predict_diff(dat1,dat2,use_terms,alpha,whole_interval,n_ps,seed,0)