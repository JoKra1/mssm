import numpy as np
import scipy as scp
import copy
from collections.abc import Callable
from .src.python.formula import Formula,build_sparse_matrix_from_formula,lhs,pd,warnings,build_penalties
from .src.python.exp_fam import Link,Logit,Identity,LOG,LOGb,Family,Binomial,Gaussian,GAMLSSFamily,GAUMLSS,Gamma,InvGauss,Binomial2,MULNOMLSS,GAMMALS,GSMMFamily,PropHaz,Poisson
from .src.python.gamm_solvers import solve_gamm_sparse,mp,repeat,tqdm,cpp_cholP,apply_eigen_perm,compute_Linv,solve_gamm_sparse2,solve_gammlss_sparse,solve_generalSmooth_sparse
from .src.python.terms import TermType,GammTerm,i,f,fs,irf,l,li,ri,rs
from .src.python.penalties import embed_shared_penalties
from .src.python.utils import sample_MVN,REML,adjust_CI,print_smooth_terms,print_parametric_terms,approx_smooth_p_values
from .src.python.custom_types import VarType,ConstType,Constraint,PenType,LambdaTerm
import davies

##################################### GAMM class #####################################

class GAMM:
    """Class to fit Generalized Additive Mixed Models.

    References:
     - Wood, S. N., & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. https://doi.org/10.1111/biom.12666
     - Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models: Estimation of Semiparametric Generalized Linear Models. https://doi.org/10.1111/j.1467-9868.2010.00749.x
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

    
    :param formula: A formula for the GAMM model
    :type formula: Formula
    :param family: A distribution implementing the :class:`Family` class. Currently :class:`Gaussian`, :class:`Gamma`, and :class:`Binomial` are implemented.
    :type family: Family
    :ivar [float] pred: The model prediction for the training data. Of the same dimension as ``self.formula.__lhs``. Initialized with ``None``.
    :ivar [float] res: The working residuals for the training data. Of the same dimension as ``self.formula.__lhs``.Initialized with ``None``.
    :ivar float edf: The model estimated degrees of freedom as a float. Initialized with ``None``.
    :ivar [float] term_edf: The estimated degrees of freedom per smooth term. Initialized with ``None``.
    :ivar scipy.sparse.csc_array lvi: The inverse of the Cholesky factor of the conditional model coefficient covariance matrix. Initialized with ``None``.
    :ivar float penalty: The total penalty applied to the model deviance after fitting as a float. Initialized with ``None``.
    :ivar [LambdaTerm] overall_penalties:  Contains all penalties estimated for the model. Initialized with ``None``.
    :ivar scipy.sparse.csc_array hessian: Estimated hessian of the log-likelihood used during fitting - will be the expected hessian for non-canonical models. Initialized with ``None``.
    :ivar scipy.sparse.csc_array hessian_obs: Observed hessian of the log-likelihood at final coefficient estimate. Not updated for strictly additive models (i.e., Gaussian with identity link). Initialized with ``None``.
    :ivar Fit_info info: A :class:`Fit_info` instance, with information about convergence (speed) of the model.
    """

    def __init__(self,
                 formula: Formula,
                 family: Family):

        # Formula associated with model
        self.formula = formula

        # Family of model
        self.family = family

        self.coef = None
        self.scale = None
        self.pred = None
        self.res = None
        self.edf = None
        self.term_edf = None
        self.Wr = None
        self.WN = None
        self.lvi = None
        self.penalty = 0
        self.hessian = None
        self.hessian_obs = None
        self.overall_penalties = None

    ##################################### Getters #####################################

    def get_pars(self):
        """
        Returns a tuple. The first entry is a np.array with all estimated coefficients. The second entry is the estimated scale parameter.
        
        Will instead return ``(None,None)`` if called before fitting.

        :return: Model coefficients and scale parameter that were estimated
        :rtype: (np.array,float) or (None, None)
        """
        return self.coef,self.scale
    
    def get_llk(self,penalized:bool=True,ext_scale:float or None=None):
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

        if len(self.formula.file_paths) != 0:
            raise NotImplementedError("Cannot return the log-likelihood if X.T@X was formed iteratively.")

        pen = 0
        if penalized:
            pen = 0.5*self.penalty
        if self.pred is not None:
            mu = self.pred
            if isinstance(self.family,Gaussian) == False or isinstance(self.family.link,Identity) == False:
                mu = self.family.link.fi(self.pred)
            if self.family.twopar:
                scale = self.scale
                if not ext_scale is None:
                    scale = ext_scale
                return self.family.llk(self.formula.y_flat[self.formula.NOT_NA_flat],mu,scale) - pen
            else:
                return self.family.llk(self.formula.y_flat[self.formula.NOT_NA_flat],mu) - pen
        return None

    def get_mmat(self,use_terms=None,drop_NA=True):
        """
        Returns exaclty the model matrix used for fitting as a scipy.sparse.csc_array. Will throw an error when called for a model for which the model
        matrix was never former completely - i.e., when :math:`\mathbf{X}^T\mathbf{X}` was formed iteratively for estimation, by setting the ``file_paths`` argument of the ``Formula`` to
        a non-empty list.
        
        Optionally, all columns not corresponding to terms for which the indices are provided via ``use_terms`` can be zeroed.

        :param use_terms: Optionally provide indices of terms in the formual that should be created. If this argument is provided columns corresponding to any term not included in this list will be zeroed, defaults to None
        :type use_terms: [int], optional
        :param drop_NA: Whether rows in the model matrix corresponding to NAs in the dependent variable vector should be dropped, defaults to True
        :type drop_NA: bool, optional
        :raises ValueError: Will throw an error when called before the model was fitted/before model penalties were formed.
        :raises NotImplementedError: Will throw an error when called for a model for which the model matrix was never former completely
        :return: Model matrix :math:`\mathbf{X}` used for fitting.
        :rtype: scp.sparse.csc_array
        """
        if self.formula.built_penalties == False:
            raise ValueError("Model matrix cannot be returned if penalties have not been initialized. Call model.fit() first.")
        elif len(self.formula.file_paths) != 0:
            raise NotImplementedError("Cannot return the model-matrix if X.T@X was formed iteratively.")
        else:
            terms = self.formula.terms
            has_intercept = self.formula.has_intercept
            ltx = self.formula.get_linear_term_idx()
            irstx = self.formula.get_ir_smooth_term_idx()
            stx = self.formula.get_smooth_term_idx()
            rtx = self.formula.get_random_term_idx()
            var_types = self.formula.get_var_types()
            var_map = self.formula.get_var_map()
            var_mins = self.formula.get_var_mins()
            var_maxs = self.formula.get_var_maxs()
            factor_levels = self.formula.get_factor_levels()
            if drop_NA:
                cov_flat = self.formula.cov_flat[self.formula.NOT_NA_flat]
            else:
                cov_flat = self.formula.cov_flat

            if len(irstx) > 0:
                cov_flat = self.formula.cov_flat # Need to drop NA rows **after** building!
                cov = self.formula.cov
            else:
                cov = None

            # Build the model matrix with all information from the formula
            model_mat = build_sparse_matrix_from_formula(terms,has_intercept,
                                                        ltx,irstx,stx,rtx,var_types,var_map,
                                                        var_mins,var_maxs,factor_levels,
                                                        cov_flat,cov,use_only=use_terms)
            
            if len(irstx) > 0 and drop_NA:
                model_mat = model_mat[self.formula.NOT_NA_flat,:]
            
            return model_mat
        
    
    def print_smooth_terms(self,pen_cutoff=0.2,p_values=False):
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
        """
        ps = None
        Trs = None
        if p_values:
            ps, Trs = approx_smooth_p_values(self)

        print_smooth_terms(self,pen_cutoff=pen_cutoff,ps=ps,Trs=Trs)
                        
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

    def get_reml(self):
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
            
        reml = REML(llk,nH,self.coef,scale,self.overall_penalties,keep)
        
        return reml
    
    def get_resid(self,type='Pearson'):
        """
        Returns the residuals :math:`e_i = y_i - \mu_i` for additive models and (by default) the Pearson residuals :math:`w_i^{0.5}*(z_i - \eta_i)` (see Wood, 2017 sections 3.1.5 & 3.1.7) for
        generalized additive models. Here :math:`w_i` are the Fisher scoring weights, :math:`z_i` the pseudo-data point for each observation, and :math:`\eta_i` is the linear prediction (i.e., :math:`g(\mu_i)` - where :math:`g()`
        is the link function) for each observation.

        If ``type= "Deviance"``, the deviance residuals are returned, which are equivalent to :math:`sign(y_i - \mu_i)*D_i^{0.5}`, where :math:`\sum_{i=1,...N} D_i` equals the model deviance (see Wood 2017, section 3.1.7).

        Throws an error if called before model was fitted.

        References:

         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

    
        :param type: The type of residual to return for a Generalized model, "Pearson" by default, but can be set to "Deviance" as well. Ignorred for additive models with identity link.
        :type type: str,optional
        :raises ValueError: Will throw an error when called before the model was fitted/before model penalties were formed.
        :return: Empirical residual vector
        :rtype: [float]
        """
        if self.res is None or self.pred is None:
            raise ValueError("Model needs to be estimated before evaluating the residuals. Call model.fit()")
        
        if type == "Pearson" or (isinstance(self.family,Gaussian) == True and isinstance(self.family.link,Identity) == True):
            return self.res
        else:
            # Deviance residual requires computing quantity D_i, which is the amount each data-point contributes to
            # overall deviance. Implemented by the family members.
            mu = self.pred
            y = self.formula.y_flat[self.formula.NOT_NA_flat]

            if not isinstance(self.family,Gaussian) or isinstance(self.family.link,Identity) == False:
                mu = self.family.link.fi(mu)

            return np.sign(y - mu) * np.sqrt(self.family.D(y,mu))
                
    ##################################### Fitting #####################################
    
    def fit(self,max_outer=50,max_inner=100,conv_tol=1e-7,extend_lambda=True,control_lambda=True,exclude_lambda=False,extension_method_lam = "nesterov",restart=False,method="Chol",check_cond=1,progress_bar=True,n_cores=10,offset = None):
        """
        Fit the specified model. Additional keyword arguments not listed below should not be modified unless you really know what you are doing.

        :param max_outer: The maximum number of fitting iterations. Defaults to 50.
        :type max_outer: int,optional
        :param max_inner: The maximum number of fitting iterations to use by the inner Newton step updating the coefficients for Generalized models. Defaults to 100.
        :type max_inner: int,optional
        :param conv_tol: The relative (change in penalized deviance is compared against ``conv_tol`` * previous penalized deviance) criterion used to determine convergence.
        :type conv_tol: float,optional
        :param extend_lambda: Whether lambda proposals should be accelerated or not. Can lower the number of new smoothing penalty proposals necessary. Enabled by default.
        :type extend_lambda: bool,optional
        :param control_lambda: Whether lambda proposals should be checked (and if necessary decreased) for actually improving the Restricted maximum likelihood of the model. Can lower the number of new smoothing penalty proposals necessary. Enabled by default.
        :type control_lambda: bool,optional
        :param exclude_lambda: Whether selective lambda terms should be excluded heuristically from updates. Can make each iteration a bit cheaper but is problematic when using additional Kernel penalties on terms. Thus, disabled by default.
        :type exclude_lambda: bool,optional
        :param restart: Whether fitting should be resumed. Only possible if the same model has previously completed at least one fitting iteration.
        :type restart: bool,optional
        :param method: Which method to use to solve for the coefficients. The default ("Chol") relies on Cholesky decomposition. This is extremely efficient but in principle less stable, numerically speaking. For a maximum of numerical stability set this to "QR". In that case a QR decomposition is used - which is first pivoted to maximize sparsity in the resulting decomposition but then also pivots for stability in order to get an estimate of rank defficiency. This takes substantially longer. This argument is ignored if ``len(self.formula.file_paths)>0`` that is, if :math:`\mathbf{X}^T\mathbf{X}` and :math:`\mathbf{X}^T\mathbf{y}` should be created iteratively. Defaults to "Chol".
        :type method: str,optional
        :param check_cond: Whether to obtain an estimate of the condition number for the linear system that is solved. When ``check_cond=0``, no check will be performed. When ``check_cond=1``, an estimate of the condition number for the final system (at convergence) will be computed and warnings will be issued based on the outcome (see :func:`mssm.src.python.gamm_solvers.est_condition`). When ``check_cond=2``, an estimate of the condition number will be performed for each new system (at each iteration of the algorithm) and an error will be raised if the condition number is estimated as too high given the chosen ``method``. Is ignored, if :math:`\mathbf{X}^T\mathbf{X}` and :math:`\mathbf{X}^T\mathbf{y}` should be created iteratively. Defaults to 1.
        :type check_cond: int,optional
        :param progress_bar: Whether progress should be displayed (convergence info and time estimate). Defaults to True.
        :type progress_bar: bool,optional
        :param n_cores: Number of cores to use during parts of the estimation that can be done in parallel. Defaults to 10.
        :type n_cores: int,optional
        :param offset: Mimics the behavior of the ``offset`` argument for ``gam`` in ``mgcv`` in R. If a value is provided here (can either be a float or a ``numpy.array`` of shape (-1,1) - if it is an array, then the first dimension has to match the number of observations in the data. NANs present in the dependent variable will be excluded from the offset vector.) then it is consistently added to the linear predictor during estimation. It will **not** be used by any other function of the :class:`GAMM` class (e.g., for prediction). This argument is ignored if ``len(self.formula.file_paths)>0`` that is, if :math:`\mathbf{X}^T\mathbf{X}` and :math:`\mathbf{X}^T\mathbf{y}` should be created iteratively. Defaults to None.
        :type offset: float or [float],optional
        """
        # We need to initialize penalties
        if not restart:
            if self.overall_penalties is not None:
                warnings.warn("Resetting penalties. If you don't want that set ``restart=True``.")
            self.overall_penalties = build_penalties(self.formula)
        penalties = self.overall_penalties

        if penalties is None and restart:
            raise ValueError("Penalties were not initialized. ``Restart`` must be set to False.")
        
        if len(self.formula.discretize) != 0 and method != "Chol":
            raise ValueError("When discretizing derivative code, the method argument must be set to 'Chol'.")
        
        self.offset = 0

        if len(self.formula.file_paths) == 0:
            # We need to build the model matrix once
            terms = self.formula.terms
            has_intercept = self.formula.has_intercept
            ltx = self.formula.get_linear_term_idx()
            irstx = self.formula.get_ir_smooth_term_idx()
            stx = self.formula.get_smooth_term_idx()
            rtx = self.formula.get_random_term_idx()
            var_types = self.formula.get_var_types()
            var_map = self.formula.get_var_map()
            var_mins = self.formula.get_var_mins()
            var_maxs = self.formula.get_var_maxs()
            factor_levels = self.formula.get_factor_levels()

            cov_flat = self.formula.cov_flat[self.formula.NOT_NA_flat]
            
            if len(irstx) > 0:
                cov_flat = self.formula.cov_flat # Need to drop NA rows **after** building!
                cov = self.formula.cov
            else:
                cov = None

            y_flat = self.formula.y_flat[self.formula.NOT_NA_flat]

            # Offset handling - for strictly additive model just subtract from y and then pass zero
            # via self.offset (default initialization)
            if offset is not None:

                # First drop NANs
                if isinstance(offset,np.ndarray):
                    offset = offset[self.formula.NOT_NA_flat]

                if not isinstance(self.family,Gaussian) or isinstance(self.family.link,Identity) == False:
                    self.offset = offset
                else:
                    y_flat -= offset

            if not self.formula.get_lhs().f is None:
                # Optionally apply function to dep. var. before fitting.
                y_flat = self.formula.get_lhs().f(y_flat)

            if y_flat.shape[0] != self.formula.y_flat.shape[0] and progress_bar:
                print("NAs were excluded for fitting.")

            # Build the model matrix with all information from the formula
            if self.formula.file_loading_nc == 1:
                model_mat = build_sparse_matrix_from_formula(terms,has_intercept,
                                                            ltx,irstx,stx,rtx,var_types,var_map,
                                                            var_mins,var_maxs,factor_levels,
                                                            cov_flat,cov)
            
            else:
                # Build row sets of model matrix in parallel:
                rpXs = []
                rpcovs = []
                for sti in stx:
                    if terms[sti].should_rp:
                        for rpi in range(len(terms[sti].RP)):
                            # Don't need to pass those down to the processes.
                            rpXs.append(terms[sti].RP[rpi].X)
                            rpcovs.append(terms[sti].RP[rpi].cov)
                            terms[sti].RP[rpi].X = None
                            terms[sti].RP[rpi].cov = None
                
                cov_split = np.array_split(cov_flat,self.formula.file_loading_nc,axis=0)
                with mp.Pool(processes=self.formula.file_loading_nc) as pool:
                    # Build the model matrix with all information from the formula - but only for sub-set of rows
                    Xs = pool.starmap(build_sparse_matrix_from_formula,zip(repeat(terms),repeat(has_intercept),
                                                                           repeat(ltx),repeat(irstx),repeat(stx),repeat(rtx),
                                                                           repeat(var_types),repeat(var_map),repeat(var_mins),
                                                                           repeat(var_maxs),repeat(factor_levels),cov_split,
                                                                           repeat(cov)))
                    
                    model_mat = scp.sparse.vstack(Xs,format='csc')
                
                # Re-assign rpXs and covs
                rpidx = 0
                for sti in stx:
                    if terms[sti].should_rp:
                        for rpi in range(len(terms[sti].RP)):
                            terms[sti].RP[rpi].X = rpXs[rpidx]
                            terms[sti].RP[rpi].cov = rpcovs[rpidx]
                            rpidx += 1

                rpXs = None
                rpcovs = None

            if len(irstx) > 0:
                # Scipy 1.15.0 does not like indexing via pd.series object, bug?
                # anyway, getting values first is fine.
                model_mat = model_mat[self.formula.NOT_NA_flat.values,:]

            # Get initial estimate of mu based on family:
            init_mu_flat = self.family.init_mu(y_flat)

            # Now we have to estimate the model
            coef,eta,wres,Wr,WN,scale,LVI,edf,term_edf,penalty,fit_info = solve_gamm_sparse(init_mu_flat,y_flat,
                                                                                      model_mat,penalties,self.formula.n_coef,
                                                                                      self.family,max_outer,max_inner,"svd",
                                                                                      conv_tol,extend_lambda,control_lambda,
                                                                                      exclude_lambda,extension_method_lam,
                                                                                      len(self.formula.discretize) == 0,
                                                                                      method,check_cond,progress_bar,n_cores,
                                                                                      self.offset)
            
            self.Wr = Wr
            self.WN = WN

            if fit_info.dropped is not None: # Make sure hessian has zero columns/rows for unidentifiable coef.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model_mat[:,fit_info.dropped] = 0

            # Compute (expected) Hessian of llk (Wood, 2011)
            if not isinstance(self.family,Gaussian) or isinstance(self.family.link,Identity) == False:
                self.hessian = -1 * ((model_mat.T@(Wr@Wr)@model_mat).tocsc()/scale)
                # Compute observed Hessian of llk
                self.hessian_obs = -1 * ((model_mat.T@(WN)@model_mat).tocsc()/scale)
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
            if not self.formula.get_lhs().f is None:
                raise ValueError("Cannot apply function to dep. var. when building model matrix iteratively. Consider creating a modified variable in the data-frame.")
            
            if isinstance(self.family,Gaussian) == False or isinstance(self.family.link,Identity) == False:
                raise ValueError("Iteratively building the model matrix is currently only supported for Normal models.")
            
            coef,eta,wres,XX,scale,LVI,edf,term_edf,penalty,fit_info = solve_gamm_sparse2(self.formula,penalties,self.formula.n_coef,
                                                                                          self.family,max_outer,"svd",
                                                                                          conv_tol,extend_lambda,control_lambda,
                                                                                          exclude_lambda,extension_method_lam,
                                                                                          len(self.formula.discretize) == 0,
                                                                                          progress_bar,n_cores)
            
            self.hessian = -1*(XX/scale)
        
        self.coef = coef
        self.scale = scale # ToDo: scale name is used in another context for more general mssm..
        self.pred = eta
        self.res = wres
        self.edf = edf
        self.term_edf = term_edf
        self.penalty = penalty
        self.info = fit_info
        self.lvi = LVI
    
    ##################################### Prediction #####################################

    def sample_post(self,n_ps,use_post=None,deviations=False,seed=None):
        """
        Obtain ``n_ps`` samples from posterior :math:`[\\boldsymbol{\\beta} - \hat{\\boldsymbol{\\beta}}] | \mathbf{y},\\boldsymbol{\lambda} \sim N(0,\mathbf{V})`,
        where V is :math:`[\mathbf{X}^T\mathbf{X} + \mathbf{S}_{\lambda}]^{-1}*/\phi` (see Wood, 2017; section 6.10). To obtain samples for :math:`\\boldsymbol{\\beta}`,
        set ``deviations`` to false.

        see :func:`sample_MVN` for more details.

        References:

         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

        :param n_ps: Number of samples to obtain from posterior.
        :type n_ps: int,optional
        :param use_post: The indices corresponding to coefficients for which to actually obtain samples. By default all coefficients are sampled.
        :type use_post: [int],optional
        :param deviations: Whether to return samples of **deviations** from the estimated coefficients (i.e., :math:`\\boldsymbol{\\beta} - \hat{\\boldsymbol{\\beta}}`) or actual samples of coefficients (i.e., :math:`\\boldsymbol{\\beta}`), defaults to False
        :type deviations: bool,optional
        :param seed: A seed to use for the sampling, defaults to None
        :type seed: int,optional
        :returns: An np.array of dimension ``[len(use_post),n_ps]`` containing the posterior samples. Can simply be post-multiplied with model matrix :math:`\mathbf{X}` to generate posterior **sample curves/predictions**.
        :rtype: [float]
        """
        if deviations:
            post = sample_MVN(n_ps,0,self.scale,P=None,L=None,LI=self.lvi,use=use_post,seed=seed)
        else:
            post = sample_MVN(n_ps,self.coef,self.scale,P=None,L=None,LI=self.lvi,use=use_post,seed=seed)
        
        return post

    def predict(self, use_terms, n_dat,alpha=0.05,ci=False,whole_interval=False,n_ps=10000,seed=None):
        """Make a prediction using the fitted model for new data ``n_dat``.
         
        But only using the terms indexed by ``use_terms``. Importantly, predictions and standard errors are always returned on the scale of the linear predictor.
        When estimating a Generalized Additive Model, the mean predictions and standard errors (often referred to as the 'response'-scale predictions) can be obtained
        by applying the link inverse function to the predictions and the CI-bounds on the linear predictor scale (DO NOT transform the standard error first and then add it to the
        transformed predictions - only on the scale of the linear predictor is the standard error additive)::

            gamma_model = GAMM(gamma_formula,Gamma()) # A true GAM
            gamma_model.fit()
            # Now get predictions on the scale of the linear predictor
            pred,_,b = gamma_model.predict(None,new_dat,ci=True)
            # Then transform to the response scale
            mu_pred = gamma_model.family.link.fi(pred)
            mu_upper_CI = gamma_model.family.link.fi(pred + b)
            mu_lower_CI = gamma_model.family.link.fi(pred - b)

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
        :rtype: (np.array,scp.sparse.csc_array,np.array or None)
        """
        var_map = self.formula.get_var_map()
        var_keys = var_map.keys()
        sub_group_vars = self.formula.get_subgroup_variables()

        for k in var_keys:
            if k in sub_group_vars:
                if k.split(":")[0] not in n_dat.columns:
                    raise IndexError(f"Variable {k.split(':')[0]} is missing in new data.")
            else:
                if k not in n_dat.columns:
                    raise IndexError(f"Variable {k} is missing in new data.")
        
        # Encode test data
        _,pred_cov_flat,_,_,pred_cov,_,_ = self.formula.encode_data(n_dat,prediction=True)

        # Then, we need to build the model matrix - but only for the terms which should
        # be included in the prediction!
        terms = self.formula.terms
        has_intercept = self.formula.has_intercept
        ltx = self.formula.get_linear_term_idx()
        irstx = self.formula.get_ir_smooth_term_idx()
        stx = self.formula.get_smooth_term_idx()
        rtx = self.formula.get_random_term_idx()
        var_types = self.formula.get_var_types()
        var_mins = self.formula.get_var_mins()
        var_maxs = self.formula.get_var_maxs()
        factor_levels = self.formula.get_factor_levels()

        if len(irstx) == 0:
            pred_cov = None

        # So we pass the desired terms to the use_only argument
        predi_mat = build_sparse_matrix_from_formula(terms,has_intercept,
                                                     ltx,irstx,stx,rtx,var_types,var_map,
                                                     var_mins,var_maxs,factor_levels,
                                                     pred_cov_flat,pred_cov,
                                                     use_only=use_terms)
        
        # Now we calculate the prediction
        pred = predi_mat @ self.coef

        # Optionally calculate the boundary for a 1-alpha CI
        if ci:
            # Wood (2017) 6.10
            c = predi_mat @ self.lvi.T @ self.lvi * self.scale @ predi_mat.T
            c = c.diagonal()
            b = scp.stats.norm.ppf(1-(alpha/2)) * np.sqrt(c)

            # Whole-interval CI (section 6.10.2 in Wood, 2017), the same idea was also
            # explored by Simpson (2016) who performs very similar computations to compute
            # such intervals. See adjust_CI function.
            if whole_interval:
                b = adjust_CI(self,n_ps,b,predi_mat,use_terms,alpha,seed)

            return pred,predi_mat,b

        return pred,predi_mat,None
    
    def predict_diff(self,dat1,dat2,use_terms,alpha=0.05,whole_interval=False,n_ps=10000,seed=None):
        """Get the difference in the predictions for two datasets.
        
        Useful to compare a smooth estimated for one level of a factor to the smooth estimated for another
        level of a factor. In that case, ``dat1`` and ``dat2`` should only differ in the level of said factor.
        Importantly, predictions and standard errors are again always returned on the scale of the linear predictor - 
        see the :func:`predict` method for details.

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
        :rtype: (np.array,np.array)
        """
        _,pmat1,_ = self.predict(use_terms,dat1)
        _,pmat2,_ = self.predict(use_terms,dat2)

        pmat_diff = pmat1 - pmat2
        
        # Predicted difference
        diff = pmat_diff @ self.coef
        
        # Difference CI
        c = pmat_diff @ self.lvi.T @ self.lvi * self.scale @ pmat_diff.T
        c = c.diagonal()
        b = scp.stats.norm.ppf(1-(alpha/2)) * np.sqrt(c)

        # Whole-interval CI (section 6.10.2 in Wood, 2017), the same idea was also
        # explored by Simpson (2016) who performs very similar computations to compute
        # such intervals. See adjust_CI function.
        if whole_interval:
            b = adjust_CI(self,n_ps,b,pmat_diff,use_terms,alpha,seed)

        return diff,b

class GAMMLSS(GAMM):
    """
    Class to fit Generalized Additive Mixed Models of Location Scale and Shape (see Rigby & Stasinopoulos, 2005).

    Example::

        # Simulate 500 data points
        GAUMLSSDat = sim6(500,seed=20)

        # We need to model the mean: \mu_i = \\alpha + f(x0)
        formula_m = Formula(lhs("y"),
                            [i(),f(["x0"],nk=10)],
                            data=GAUMLSSDat)

        # and the standard deviation as well: log(\sigma_i) = \\alpha + f(x0)
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
    :ivar [[float]] overall_preds: The predicted means for every parameter of ``family`` evaluated for each observation in the training data. Initialized with ``None``.
    :ivar float edf: The model estimated degrees of freedom as a float. Initialized with ``None``.
    :ivar [float] overall_term_edf: The estimated degrees of freedom per smooth term. Initialized with ``None``.
    :ivar scipy.sparse.csc_array lvi: The inverse of the Cholesky factor of the conditional model coefficient covariance matrix. Initialized with ``None``.
    :ivar float penalty: The total penalty applied to the model deviance after fitting as a float. Initialized with ``None``.
    :ivar [int] overall_coef:  Contains all coefficients estimated for the model. Initialized with ``None``.
    :ivar [int] coef_split_idx: The index at which to split the overall coefficient vector into separate lists - one per parameter of ``family``. Initialized after fitting!
    :ivar scp.sparse.csc_array hessian:  Estimated hessian of the log-likelihood (will correspond to ``hessian - diag*eps`` if ``self.info.eps > 0`` after fitting). Initialized with ``None``.
    :ivar [LambdaTerm] overall_penalties:  Contains all penalties estimated for the model. Initialized with ``None``.
    :ivar Fit_info info: A :class:`Fit_info` instance, with information about convergence (speed) of the model.
    """
    def __init__(self, formulas: [Formula], family: GAMLSSFamily):
        super().__init__(None, family)
        self.formulas = copy.deepcopy(formulas) # self.formula can hold formula for single parameter later on for predictions.
        self.overall_lvi = None
        self.overall_coef = None
        self.overall_preds = None # etas
        self.overall_mus = None # Expected values for each parameter of response distribution
        self.hessian = None

    
    def get_pars(self):
        """
        Returns a list containing all coefficients estimated for the model. Use ``self.coef_split_idx`` to split the vector into separate subsets per distribution parameter.

        Will return None if called before fitting was completed.
        
        :return: Model coefficients - before splitting!
        :rtype: [float] or None
        """
        return self.overall_coef
    

    def get_llk(self,penalized:bool=True):
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
        if self.overall_preds is not None:
            mus = [self.family.links[i].fi(self.overall_preds[i]) for i in range(self.family.n_par)]
            return self.family.llk(self.formulas[0].y_flat[self.formulas[0].NOT_NA_flat],*mus) - pen

        return None

    def get_mmat(self,use_terms=None,drop_NA=True):
        """
        Returns a list containing exaclty the model matrices used for fitting as a ``scipy.sparse.csc_array``. Will raise an error when fitting was not completed before calling this function.

        Optionally, all columns not corresponding to terms for which the indices are provided via ``use_terms`` can be zeroed.

        :param use_terms: Optionally provide indices of terms in the formual that should be created. If this argument is provided columns corresponding to any term not included in this list will be zeroed, defaults to None
        :type use_terms: [int], optional
        :param drop_NA: Whether rows in the model matrix corresponding to NAs in the dependent variable vector should be dropped, defaults to True
        :type drop_NA: bool, optional
        :raises ValueError: Will throw an error when called before the model was fitted/before model penalties were formed.
        :return: Model matrices :math:`\mathbf{X}` used for fitting - one per parameter of ``self.family``.
        :rtype: [scp.sparse.csc_array]
        """
        if self.formula is None: # Prevent problems when this is called from .print_smooth_terms()
            Xs = []
            for form in self.formulas:
                if form.built_penalties == False:
                    raise ValueError("Model matrices cannot be returned if penalties have not been initialized. Call model.fit() first.")
                
                mod = GAMM(form,family=Gaussian())
                Xs.append(mod.get_mmat(use_terms=use_terms,drop_NA=drop_NA))
            return Xs
        else:
            mod = GAMM(self.formula,family=Gaussian())
            return mod.get_mmat(use_terms=use_terms,drop_NA=drop_NA)
    
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
    
    def print_smooth_terms(self, pen_cutoff=0.2,p_values=False):
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
        """
        ps = None
        Trs = None
        for formi, _ in enumerate(self.formulas):
            print(f"\nDistribution parameter: {formi + 1}\n")
            
            if p_values:
                ps, Trs = approx_smooth_p_values(self,par=formi)

            print_smooth_terms(self,par=formi,pen_cutoff=pen_cutoff,ps=ps,Trs=Trs)
                        
    def get_reml(self):
        """
        Get's the Laplcae approximate REML (Restrcited Maximum Likelihood) score for the estimated lambda values (see Wood, 2011).

        References:

         - Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models: Estimation of Semiparametric Generalized Linear Models.
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.

        :raises ValueError: Will throw an error when called before the model was fitted/before model penalties were formed.
        :return: REML score
        :rtype: float
        """

        if self.overall_coef is None or self.hessian is None:
            raise ValueError("Model needs to be estimated before evaluating the REML score. Call model.fit()")
        
        llk = self.get_llk(False)

        keep = None
        if self.info.dropped is not None:
            keep = [cidx for cidx in range(self.hessian.shape[1]) if cidx not in self.info.dropped]
        
        reml = REML(llk,-1*self.hessian,self.overall_coef,1,self.overall_penalties,keep)[0,0]
        return reml
    
    def get_resid(self):
        """ Returns standarized residuals for GAMMLSS models (Rigby & Stasinopoulos, 2005).

        The computation of the residual vector will differ a lot between different GAMMLSS models and is thus implemented
        as a method by each GAMMLSS family. These should be consulted to get more details. In general, if the
        model is specified correctly, the returned vector should approximately look like what could be expected from
        taking :math:`N` independent samples from :math:`N(0,1)`.

        References:
         
         - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, Scale and Shape.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

        :raises NotImplementedError: An error is raised in case the residuals are to be computed for a Multinomial GAMMLSS model, which is currently not supported.
        :raises ValueError: An error is raised in case the residuals are requested before the model has been fit.
        :return: A list of standardized residuals that should be :math:`\sim N(0,1)` if the model is correct.
        :return: Empirical residual vector
        :rtype: [float]
        """

        if self.overall_coef is None or self.hessian is None:
            raise ValueError("Model needs to be estimated before evaluating the residuals. Call model.fit()")

        if isinstance(self.family,MULNOMLSS):
            raise NotImplementedError("Residual computation for Multinomial model is not currently supported.")
        
        return self.family.get_resid(self.formulas[0].y_flat[self.formulas[0].NOT_NA_flat],*self.overall_mus)
        

    def fit(self,max_outer=50,max_inner=200,min_inner=200,conv_tol=1e-7,extend_lambda=True,extension_method_lam="nesterov2",control_lambda=1,restart=False,method="Chol",check_cond=1,piv_tol=np.power(np.finfo(float).eps,0.04),should_keep_drop=True,prefit_grad=False,repara=False,progress_bar=True,n_cores=10,seed=0,init_lambda=None):
        """
        Fit the specified model. Additional keyword arguments not listed below should not be modified unless you really know what you are doing.

        :param max_outer: The maximum number of fitting iterations.
        :type max_outer: int,optional
        :param max_inner: The maximum number of fitting iterations to use by the inner Newton step for coefficients.
        :type max_inner: int,optional
        :param min_inner: The minimum number of fitting iterations to use by the inner Newton step for coefficients.
        :type min_inner: int,optional
        :param conv_tol: The relative (change in penalized deviance is compared against ``conv_tol`` * previous penalized deviance) criterion used to determine convergence.
        :type conv_tol: float,optional
        :param extend_lambda: Whether lambda proposals should be accelerated or not. Can lower the number of new smoothing penalty proposals necessary. Enabled by default.
        :type extend_lambda: bool,optional
        :param control_lambda: Whether lambda proposals should be checked (and if necessary decreased) for whether or not they (approxiately) increase the Laplace approximate restricted maximum likelihood of the model. Setting this to 0 disables control. Setting it to 1 means the step will never be smaller than the original EFS update but extensions will be removed in case the objective was exceeded. Setting it to 2 means that steps will be halved. Set to 1 by default.
        :type control_lambda: int,optional
        :param restart: Whether fitting should be resumed. Only possible if the same model has previously completed at least one fitting iteration.
        :type restart: bool,optional
        :param method: Which method to use to solve for the coefficients. The default ("Chol") relies on Cholesky decomposition. This is extremely efficient but in principle less stable, numerically speaking. For a maximum of numerical stability set this to "QR/Chol" or "LU/Chol". In that case the coefficients are still obtained via a Cholesky decomposition but a QR/LU decomposition is formed afterwards to check for rank deficiencies and to drop coefficients that cannot be estimated given the current smoothing parameter values. This takes substantially longer. Defaults to "Chol".
        :type method: str,optional
        :param check_cond: Whether to obtain an estimate of the condition number for the linear system that is solved. When ``check_cond=0``, no check will be performed. When ``check_cond=1``, an estimate of the condition number for the final system (at convergence) will be computed and warnings will be issued based on the outcome (see :func:`mssm.src.python.gamm_solvers.est_condition`). Defaults to 1.
        :type check_cond: int,optional
        :param piv_tol: Deprecated.
        :type piv_tol: float,optional
        :param should_keep_drop: Only used when ``method in ["QR/Chol","LU/Chol","Direct/Chol"]``. If set to True, any coefficients that are dropped during fitting - are permanently excluded from all subsequent iterations. If set to False, this is determined anew at every iteration - **costly**! Defaults to True.
        :type should_keep_drop: bool,optional
        :param prefit_grad: Whether to rely on Gradient Descent to improve the initial starting estimate for coefficients. Defaults to False.
        :type prefit_grad: bool,optional
        :param repara: Whether to re-parameterize the model (for every proposed update to the regularization parameters) via the steps outlined in Appendix B of Wood (2011) and suggested by Wood et al., (2016). This greatly increases the stability of the fitting iteration. Defaults to False.
        :type repara: bool,optional
        :param progress_bar: Whether progress should be displayed (convergence info and time estimate). Defaults to True.
        :type progress_bar: bool,optional
        :param n_cores: Number of cores to use during parts of the estimation that can be done in parallel. Defaults to 10.
        :type n_cores: int,optional
        :param seed: Seed to use for random parameter initialization. Defaults to 0
        :type seed: int,optional
        :param init_lambda: A set of initial :math:`\lambda` parameters to use by the model. Length of list must match number of parameters to be estimated. Defaults to None
        :type init_lambda: [float],optional
        """

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
        for form in self.formulas:
            mod = GAMM(form,family=Gaussian())
            if restart == False:
                ind_penalties.append(build_penalties(form))
            Xs.append(mod.get_mmat())

        # Initialize coef from family
        coef = self.family.init_coef([GAMM(form,family=Gaussian()) for form in self.formulas])

        # Get GAMMLSS penalties
        if restart == False:
            shared_penalties = embed_shared_penalties(ind_penalties,self.formulas)
            gamlss_pen = [pen for pens in shared_penalties for pen in pens]
            self.overall_penalties = gamlss_pen

            # Clean up
            shared_penalties = None
            ind_penalties = None
        
            # Check for family-wide initialization of lambda values
            if init_lambda is None:
                init_lambda = self.family.init_lambda(self.formulas)

            # Else start with provided values or simply with much weaker penalty than for GAMs
            for pen_i in range(len(gamlss_pen)):
                if init_lambda is None:
                    gamlss_pen[pen_i].lam = 0.01
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
        self.overall_coef = coef
        self.overall_preds = etas
        self.overall_mus = mus
        self.res = wres
        self.edf = total_edf
        self.overall_term_edf = term_edfs
        self.penalty = penalty
        self.coef_split_idx = coef_split_idx
        self.overall_lvi = LV
        self.hessian = H
        if fit_info.eps > 0: # Make sure -H + S_emb is invertible
            warnings.warn(f"model.info.eps > 0 ({np.round(fit_info.eps,decimals=2)}). Perturbing Hessian of log-likelihood to ensure that negative Hessian of penalized log-likelihood is invertible.")
            self.hessian -= fit_info.eps*scp.sparse.identity(H.shape[1],format='csc')
        self.info = fit_info
    
    def sample_post(self, n_ps, use_post=None, deviations=False, seed=None, par=0):
        """
        Obtain ``n_ps`` samples from posterior :math:`[\\boldsymbol{\\beta} - \hat{\\boldsymbol{\\beta}}] | \mathbf{y},\\boldsymbol{\lambda} \sim N(0,\mathbf{V})`,
        where :math:`\mathbf{V}=[-\mathbf{H} + \mathbf{S}_{\lambda}]^{-1}` (see Wood et al., 2016; Wood 2017, section 6.10). :math:`\mathbf{H}` here is the hessian of
        the log-likelihood (Wood et al., 2016;). To obtain samples for :math:`\\boldsymbol{\\beta}`, set ``deviations`` to false.

        see :func:`sample_MVN` for more details.

        References:

         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

        :param n_ps: Number of samples to obtain from posterior.
        :type n_ps: int,optional
        :param use_post: The indices corresponding to coefficients for which to actually obtain samples. By default all coefficients are sampled.
        :type use_post: [int],optional
        :param deviations: Whether to return samples of **deviations** from the estimated coefficients (i.e., :math:`\\boldsymbol{\\beta} - \hat{\\boldsymbol{\\beta}}`) or actual samples of coefficients (i.e., :math:`\\boldsymbol{\\beta}`), defaults to False
        :type deviations: bool,optional
        :param seed: A seed to use for the sampling, defaults to None
        :type seed: int,optional
        :param par: The index corresponding to the distribution parameter for which to make the prediction (e.g., 0 = mean)
        :type par: int
        :returns: An np.array of dimension ``[len(use_post),n_ps]`` containing the posterior samples. Can simply be post-multiplied with model matrix :math:`\mathbf{X}` to generate posterior **sample curves**.
        :rtype: [float]
        """
        # Prepare so that we can just call gamm.sample_post()
        if self.coef is None: # Prevent problems when this is called from .predict()
            self.formula = self.formulas[par]
            split_coef = np.split(self.overall_coef,self.coef_split_idx)
            self.coef = np.ndarray.flatten(split_coef[par])
            self.scale=1
            start = 0
            
            if len(self.coef_split_idx) == 0:
                end = self.formula.n_coef
            else:
                end = self.coef_split_idx[0]
                for pari in range(1,par+1):
                    start = end
                    end += self.formulas[pari].n_coef
            self.lvi = self.overall_lvi[:,start:end]
        
            post = super().sample_post(n_ps, use_post, deviations, seed)

            # Clean up
            self.formula = None
            self.coef = None
            self.scale = None
            self.lvi = None
        else:
            post = super().sample_post(n_ps, use_post, deviations, seed)

        return post

    def predict(self, par, use_terms, n_dat, alpha=0.05, ci=False, whole_interval=False, n_ps=10000, seed=None):
        """
        Make a prediction using the fitted model for new data ``n_dat`` using only the terms indexed by ``use_terms`` and for distribution parameter ``par``.

        Importantly, predictions and standard errors are always returned on the scale of the linear predictor. For the Gaussian GAMMLSS model, the 
        predictions for the standard deviation will thus reflect the log of the standard deviation. To get the predictions on the standard deviation scale,
        one can apply the inverse log-link function to the predictions and the CI-bounds on the scale of the respective linear predictor.::

            model = GAMMLSS(formulas,GAUMLSS([Identity(),LOG()])) # Fit a Gaussian GAMMLSS model
            model.fit()
            # Mean predictions don't have to be transformed since the Identity link is used for this predictor.
            mu_mean,_,b_mean = model.predict(0,None,new_dat,ci=True)
            mean_upper_CI = mu_mean + b_mean
            mean_lower_CI = mu_mean - b_mean
            # Standard deviation predictions do have to be transformed - by default they are on the log-scale.
            eta_sd,_,b_sd = model.predict(1,None,new_dat,ci=True)
            mu_sd = model.family.links[1].fi(eta_sd) # Index to `links` is 1 because the sd is the second parameter!
            sd_upper_CI = model.family.links[1].fi(eta_sd + b_sd)
            sd_lower_CI = model.family.links[1].fi(eta_sd - b_sd)


        References:

         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
         - Simpson, G. (2016). Simultaneous intervals for smooths revisited.

        :param par: The index corresponding to the parameter for which to make the prediction (e.g., 0 = mean)
        :type par: int
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
        :raises ValueError: An error is raised in case the standard error is to be computed for a Multinomial GAMMLSS model, which is currently not supported.
        :return: A tuple with 3 entries. The first entry is the prediction ``pred`` based on the new data ``n_dat``. The second entry is the model matrix built for ``n_dat`` that was post-multiplied with the model coefficients to obtain ``pred``. The third entry is ``None`` if ``ci``==``False`` else the standard error ``se`` in the prediction.
        :rtype: (np.array,scp.sparse.csc_array,np.array or None)
        """
        
        # Prepare so that we can just call gamm.predict()    
        self.formula = self.formulas[par]
        split_coef = np.split(self.overall_coef,self.coef_split_idx)
        self.coef = np.ndarray.flatten(split_coef[par])
        self.scale=1
        start = 0

        if len(self.coef_split_idx) == 0:
            end = self.formula.n_coef
        else:
            end = self.coef_split_idx[0]
            for pari in range(1,par+1):
                start = end
                end += self.formulas[pari].n_coef

        self.lvi = self.overall_lvi[:,start:end]

        pred = super().predict(use_terms, n_dat, alpha, ci, whole_interval, n_ps, seed)

        # Clean up
        self.formula = None
        self.coef = None
        self.scale = None
        self.lvi = None

        return pred
    
    def predict_diff(self, dat1, dat2, par, use_terms, alpha=0.05, whole_interval=False, n_ps=10000, seed=None):
        """
        Get the difference in the predictions for two datasets and for distribution parameter ``par``. Useful to compare a smooth estimated for
        one level of a factor to the smooth estimated for another level of a factor. In that case, ``dat1`` and
        ``dat2`` should only differ in the level of said factor. Importantly, predictions and standard errors are again always returned on the scale of the linear predictor - 
        see the :func:`predict` method for details.

        References:

         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
         - Simpson, G. (2016). Simultaneous intervals for smooths revisited.
         - ``get_difference`` function from ``itsadug`` R-package: https://rdrr.io/cran/itsadug/man/get_difference.html

        :param dat1: A pandas DataFrame containing new data for which to make the prediction. Importantly, all variables present in the data used to fit the model also need to be present in this DataFrame. Additionally, factor variables must only include levels also present in the data used to fit the model. If you want to exclude a specific factor from the prediction (for example the factor subject) don't include the terms that involve it in the `use_terms` argument.
        :type dat1: pd.DataFrame
        :param dat2: A second pandas DataFrame for which to also make a prediction. The difference in the prediction between this `dat1` will be returned.
        :type dat2: pd.DataFrame
        :param par: The index corresponding to the parameter for which to make the prediction (e.g., 0 = mean)
        :type par: int
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
        :raises ValueError: An error is raised in case the predicted difference is to be computed for a Multinomial GAMMLSS model, which is currently not supported.
        :return: A tuple with 2 entries. The first entry is the predicted difference (between the two data sets ``dat1`` & ``dat2``) ``diff``. The second entry is the standard error ``se`` of the predicted difference. The difference CI is then [``diff`` - ``se``, ``diff`` + ``se``]
        :rtype: (np.array,np.array)
        """
        
        _,pmat1,_ = self.predict(par,use_terms,dat1)
        _,pmat2,_ = self.predict(par,use_terms,dat2)

        pmat_diff = pmat1 - pmat2

        # Now prepare formula, coef, scale, and lvi in case sample_post get's called:
        self.formula = self.formulas[par]
        split_coef = np.split(self.overall_coef,self.coef_split_idx)
        self.coef = np.ndarray.flatten(split_coef[par])
        self.scale=1

        start = 0
        if len(self.coef_split_idx) == 0:
            end = self.formula.n_coef
        else:
            end = self.coef_split_idx[0]
            for pari in range(1,par+1):
                start = end
                end += self.formulas[pari].n_coef

        self.lvi = self.overall_lvi[:,start:end]
        
        # Predicted difference
        diff = pmat_diff @ self.coef
        
        # Difference CI
        c = pmat_diff @ self.lvi.T @ self.lvi * self.scale @ pmat_diff.T
        c = c.diagonal()
        b = scp.stats.norm.ppf(1-(alpha/2)) * np.sqrt(c)

        # Whole-interval CI (section 6.10.2 in Wood, 2017), the same idea was also
        # explored by Simpson (2016) who performs very similar computations to compute
        # such intervals. See adjust_CI function.
        if whole_interval:
            b = adjust_CI(self,n_ps,b,pmat_diff,use_terms,alpha,seed)

        # Clean up
        self.formula = None
        self.coef = None
        self.scale = None
        self.lvi = None

        return diff,b
    

class GSMM(GAMMLSS):
    """
    Class to fit General Smooth/Mixed Models (see Wood, Pya, & Säfken; 2016). Estimation is possible via exact Newton method for coefficients of via L-qEFS update (see Krause et al., in preparation and example below).

    Example::

        class NUMDIFFGENSMOOTHFamily(GSMMFamily):
            # Implementation of the ``GSMMFamily`` class that uses finite differencing to obtain the
            # gradient of the likelihood to estimate a Gaussian GAMLSS via the general smooth code and the L-qEFS update by Krause et al. (in preparation).

            # References:
            #    - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
            #    - Nocedal & Wright (2006). Numerical Optimization. Springer New York.
            

            def __init__(self, pars: int, links:[Link], llkfun:Callable, *llkargs) -> None:
                super().__init__(pars, links, *llkargs)
                self.llkfun = llkfun
            
            def llk(self, coef, coef_split_idx, y, Xs):
                return self.llkfun(coef, coef_split_idx, self.links, y, Xs,*self.llkargs)

        def llk_gamm_fun(coef,coef_split_idx,links,y,Xs):
            # Likelihood for a Gaussian GAM(LSS) - implemented so
            # that the model can be estimated using the general smooth code.

            coef = coef.reshape(-1,1)
            split_coef = np.split(coef,coef_split_idx)
            eta_mu = Xs[0]@split_coef[0]
            eta_sd = Xs[1]@split_coef[1]
            
            mu_mu = links[0].fi(eta_mu)
            mu_sd = links[1].fi(eta_sd)
            
            family = GAUMLSS([Identity(),LOG()])
            llk = family.llk(y,mu_mu,mu_sd)
            return llk

        # Simulate 500 data points
        sim_dat = sim3(500,2,c=1,seed=0,family=Gaussian(),binom_offset = 0, correlate=False)

        # We need to model the mean: \mu_i
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

        # Now define the general family + model and fit!
        gsmm_fam = GAMLSSGENSMOOTHFamily(2,links,llk_gamm_fun,GAUMLSS(links))
        model = GSMM(formulas=formulas,family=gsmm_fam)

        # Fit with SR1
        bfgs_opt={"gtol":1e-9,
                "ftol":1e-9,
                "maxcor":30,
                "maxls":200,
                "maxfun":1e7}
                        
        model.fit(init_coef=None,method='qEFS',extend_lambda=False,
                control_lambda=False,max_outer=200,max_inner=500,min_inner=500,
                seed=0,qEFSH='SR1',max_restarts=0,overwrite_coef=False,
                qEFS_init_converge=False,prefit_grad=True,
                progress_bar=True,**bfgs_opt)


    References:
     - Wood, S. N., & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. https://doi.org/10.1111/biom.12666
     - Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models: Estimation of Semiparametric Generalized Linear Models. https://doi.org/10.1111/j.1467-9868.2010.00749.x
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
     - Nocedal & Wright (2006). Numerical Optimization. Springer New York.
    
    
    :param formulas: A list of formulas, one per parameter of the likelihood that is to be modeled as a smooth model
    :type formulas: [Formula]
    :param family: A GSMMFamily family.
    :type family: GSMMFamily
    :ivar float edf: The model estimated degrees of freedom as a float. Initialized with ``None``.
    :ivar [float] overall_term_edf: The estimated degrees of freedom per smooth term. Initialized with ``None``.
    :ivar scipy.sparse.csc_array or scipy.sparse.linalg.LinearOperator lvi: Either the inverse of the Cholesky factor of the conditional model coefficient covariance matrix - or (in case the ``L-BFGS-B`` optimizer was used and ``form_VH`` was set to False when calling ``model.fit()``) a :class:`scipy.sparse.linalg.LinearOperator` of the covariance matrix **not the root**. Initialized with ``None``.
    :ivar float penalty: The total penalty applied to the model deviance after fitting as a float. Initialized with ``None``.
    :ivar [int] overall_coef:  Contains all coefficients estimated for the model. Initialized with ``None``.
    :ivar [int] coef_split_idx: The index at which to split the overall coefficient vector into separate lists - one per parameter of ``family``. Initialized after fitting!
    :ivar scp.sparse.csc_array hessian:  Estimated hessian of the log-likelihood (will correspond to ``hessian - diag*eps`` if ``self.info.eps > 0`` after fitting). Initialized with ``None``.
    :ivar [LambdaTerm] overall_penalties:  Contains all penalties estimated for the model. Initialized with ``None``.
    :ivar Fit_info info: A :class:`Fit_info` instance, with information about convergence (speed) of the model.
    """

    def __init__(self, formulas: [Formula], family: GSMMFamily):
        super().__init__(formulas, family)
    
    def get_llk(self,penalized:bool=True,drop_NA=True):
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
        if self.overall_coef is not None:

            y = self.formulas[0].y_flat[self.formulas[0].NOT_NA_flat]
            # Build penalties and model matrices for all formulas
            Xs = []
            for form in self.formulas:
                mod = GAMM(form,family=Gaussian())
                Xs.append(mod.get_mmat(drop_NA=drop_NA))

            return self.family.llk(self.overall_coef,self.coef_split_idx,y,Xs) - pen

        return None

    def get_reml(self,drop_NA=True):
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

        if self.overall_coef is None or self.hessian is None:
            raise ValueError("Model needs to be estimated before evaluating the REML score. Call model.fit()")
        
        llk = self.get_llk(False,drop_NA=drop_NA)

        keep = None
        if self.info.dropped is not None:
            keep = [cidx for cidx in range(self.hessian.shape[1]) if cidx not in self.info.dropped]
        
        reml = REML(llk,-1*self.hessian,self.overall_coef,1,self.overall_penalties,keep)[0,0]
        return reml
    
    def get_resid(self):
        """What qualifies as "residual" will differ vastly between different implementations of this class, so this method simply returns ``None``.
        """
        return None
    
    def fit(self,init_coef=None,max_outer=50,max_inner=200,min_inner=200,conv_tol=1e-7,extend_lambda=True,extension_method_lam="nesterov2",control_lambda=1,restart=False,optimizer="Newton",method="Chol",check_cond=1,piv_tol=np.power(np.finfo(float).eps,0.04),progress_bar=True,n_cores=10,seed=0,drop_NA=True,init_lambda=None,form_VH=True,use_grad=False,build_mat=None,should_keep_drop=True,gamma=1,qEFSH='SR1',overwrite_coef=True,max_restarts=0,qEFS_init_converge=True,prefit_grad=False,repara=False,init_bfgs_options=None,**bfgs_options):
        """
        Fit the specified model. Additional keyword arguments not listed below should not be modified unless you really know what you are doing.

        :param max_outer: The maximum number of fitting iterations.
        :type max_outer: int,optional
        :param max_inner: The maximum number of fitting iterations to use by the inner Newton step for coefficients.
        :type max_inner: int,optional
        :param min_inner: The minimum number of fitting iterations to use by the inner Newton step for coefficients.
        :type min_inner: int,optional
        :param conv_tol: The relative (change in penalized deviance is compared against ``conv_tol`` * previous penalized deviance) criterion used to determine convergence.
        :type conv_tol: float,optional
        :param extend_lambda: Whether lambda proposals should be accelerated or not. Can lower the number of new smoothing penalty proposals necessary. Enabled by default.
        :type extend_lambda: bool,optional
        :param control_lambda: Whether lambda proposals should be checked (and if necessary decreased) for whether or not they (approxiately) increase the Laplace approximate restricted maximum likelihood of the model. Setting this to 0 disables control. Setting it to 1 means the step will never be smaller than the original EFS update but extensions will be removed in case the objective was exceeded. Setting it to 2 means that steps will be halved. Set to 1 by default.
        :type control_lambda: int,optional
        :param restart: Whether fitting should be resumed. Only possible if the same model has previously completed at least one fitting iteration.
        :type restart: bool,optional
        :param optimizer: Deprecated. Defaults to "Newton"
        :type optimizer: str,optional
        :param method: Which method to use to solve for the coefficients (and smoothing parameters). The default ("Chol") relies on Cholesky decomposition. This is extremely efficient but in principle less stable, numerically speaking. For a maximum of numerical stability set this to "QR/Chol" or "LU/Chol". In that case the coefficients are still obtained via a Cholesky decomposition but a QR/LU decomposition is formed afterwards to check for rank deficiencies and to drop coefficients that cannot be estimated given the current smoothing parameter values. This takes substantially longer. If this is set to ``'qEFS'``, then the coefficients are estimated via quasi netwon and the smoothing penalties are estimated from the quasi newton approximation to the hessian. This only requieres first derviative information. Defaults to "Chol".
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
        :param drop_NA: Whether to drop rows in the **model matrices** corresponding to NAs in the dependent variable vector. Defaults to True.
        :type drop_NA: bool,optional
        :param init_lambda: A set of initial :math:`\lambda` parameters to use by the model. Length of list must match number of parameters to be estimated. Defaults to None
        :type init_lambda: [float],optional
        :param form_VH: Whether to explicitly form matrix ``V`` - the estimated inverse of the negative Hessian of the penalized likelihood - and ``H`` - the estimate of said Hessian - when using the ``qEFS`` method. If set to False, only ``V`` is returned - as a :class:`scipy.sparse.linalg.LinearOperator` - and available in ``self.overall_lvi``. Additionally, ``self.hessian`` will then be equal to ``None``. Note, that this will break default prediction/confidence interval methods - so do not call them. Defaults to True
        :type form_VH: bool,optional
        :param use_grad: Deprecated.
        :type use_grad: bool,optional
        :param build_mat: An (optional) list, containing one bool per :class:`mssm.src.python.formula.Formula` in ``self.formulas`` - indicating whether the corresponding model matrix should be built. Useful if multiple formulas specify the same model matrix, in which case only one needs to be built. Defaults to None, which means all model matrices are built.
        :type build_mat: [bool], optional
        :param should_keep_drop: Only used when ``method in ["QR/Chol","LU/Chol","Direct/Chol"]``. If set to True, any coefficients that are dropped during fitting - are permanently excluded from all subsequent iterations. If set to False, this is determined anew at every iteration - **costly**! Defaults to True.
        :type should_keep_drop: bool,optional
        :param gamma: Setting this to a value larger than 1 promotes more complex (less smooth) models. Setting this to a value smaller than 1 (but must be > 0) promotes smoother models! Defaults to 1.
        :type gamma: float,optional
        :param qEFSH: Should the hessian approximation use a symmetric rank 1 update (``qEFSH='SR1'``) that is forced to result in positive definiteness of the approximation or the standard bfgs update (``qEFSH='BFGS'``) . Defaults to 'SR1'.
        :type qEFSH: str,optional
        :param overwrite_coef: Whether the initial coefficients passed to the optimization routine should be over-written by the solution obtained for the un-penalized version of the problem when ``method='qEFS'``. Setting this to False will be useful when passing coefficients from a simpler model to initialize a more complex one. Only has an effect when ``qEFS_init_converge=True``. Defaults to True.
        :type overwrite_coef: bool,optional
        :param max_restarts: How often to shrink the coefficient estimate back to a random vector when convergence is reached and when ``method='qEFS'``. The optimizer might get stuck in local minima so it can be helpful to set this to 1-3. What happens is that if we converge, we shrink the coefficients back to a random vector and then continue optimizing once more. Defaults to 0.
        :type max_restarts: int,optional
        :param qEFS_init_converge: Whether to optimize the un-penalzied version of the model and to use the hessian (and optionally coefficients, if ``overwrite_coef=True``) to initialize the q-EFS solver. Ignored if ``method!='qEFS'``. Defaults to True.
        :type qEFS_init_converge: bool,optional
        :param prefit_grad: Whether to rely on Gradient Descent to improve the initial starting estimate for coefficients. Defaults to False.
        :type prefit_grad: bool,optional
        :param repara: Whether to re-parameterize the model (for every proposed update to the regularization parameters) via the steps outlined in Appendix B of Wood (2011) and suggested by Wood et al., (2016). This greatly increases the stability of the fitting iteration. Defaults to False.
        :type repara: bool,optional
        :param init_bfgs_options: An optional dictionary holding the same key:value pairs that can be passed to ``bfgs_options`` but pased to the optimizer of the un-penalized problem. If this is None, it will be set to a copy of ``bfgs_options``. Defaults to None.
        :type init_bfgs_options: dict,optional
        :param bfgs_options: Any additional keyword arguments that should be passed on to the call of :func:`scipy.optimize.minimize` if ``method=='qEFS'``. If none are provided, the ``gtol`` argument will be initialized to ``conv_tol``. Note also, that in any case the ``maxiter`` argument is automatically set to ``max_inner``. Defaults to None.
        :type bfgs_options: key=value,optional
        :raises ValueError: Will throw an error when ``optimizer`` is not 'Newton'.
        """

        if not bfgs_options:
            bfgs_options = {"gtol":conv_tol,
                            "ftol":1e-9,
                            "maxcor":30,
                            "maxls":100,
                            "maxfun":1e7}
        
        if init_bfgs_options is None:
            init_bfgs_options = copy.deepcopy(bfgs_options)

        if not optimizer in ["Newton"]:
            raise ValueError("'optimizer' needs to be set to 'Newton'.")
        
        if self.overall_penalties is None and restart == True:
            raise ValueError("Penalties were not initialized. ``Restart`` must be set to False.")
        
        # Get y
        if drop_NA:
            y = self.formulas[0].y_flat[self.formulas[0].NOT_NA_flat]
        else:
            y = self.formulas[0].y_flat

        if not self.formulas[0].get_lhs().f is None:
            # Optionally apply function to dep. var. before fitting. Not sure why that would be desirable for this model class...
            y = self.formulas[0].get_lhs().f(y)

        # Build penalties and model matrices for all formulas
        Xs = []
        ind_penalties = []
        for fi,form in enumerate(self.formulas):
            mod = GAMM(form,family=Gaussian())
            if build_mat is None or build_mat[fi]:
                if restart == False:
                    ind_penalties.append(build_penalties(form))
                Xs.append(mod.get_mmat(drop_NA=drop_NA))

        # Get all penalties
        if restart == False:
            shared_penalties = embed_shared_penalties(ind_penalties,self.formulas)
            shared_penalties = [sp for sp in shared_penalties if len(sp) > 0]

            smooth_pen = [pen for pens in shared_penalties for pen in pens]
            self.overall_penalties = smooth_pen

            # Clean up
            shared_penalties = None
            ind_penalties = None
        
            # Check for family-wide initialization of lambda values
            if init_lambda is None:
                init_lambda = self.family.init_lambda(self.formulas)
            
            # Otherwise initialize with provided values or simply with much weaker penalty than for GAMs
            for pen_i in range(len(smooth_pen)):
                if init_lambda is None:
                    smooth_pen[pen_i].lam = 0.001
                else:
                    smooth_pen[pen_i].lam = init_lambda[pen_i]

        else:
            smooth_pen = self.overall_penalties

        # Initialize overall coefficients
        form_n_coef = [form.n_coef for form in self.formulas]
        form_up_coef = [form.unpenalized_coef for form in self.formulas]
        n_coef = np.sum(form_n_coef)

        # Again check first for family wide initialization
        if init_coef is None:
            init_coef = self.family.init_coef([GAMM(form,family=Gaussian()) for form in self.formulas])
        
        # Otherwise again initialize with provided values or randomly
        if not init_coef is None:
            coef = np.array(init_coef).reshape(-1,1)
        else:
            coef = scp.stats.norm.rvs(size=n_coef,random_state=seed).reshape(-1,1)

        coef_split_idx = form_n_coef[:-1]

        if len(self.formulas) > 1:
            for coef_i in range(1,len(coef_split_idx)):
                coef_split_idx[coef_i] += coef_split_idx[coef_i-1]
        
        # Now fit model
        coef,H,LV,total_edf,term_edfs,penalty,smooth_pen,fit_info = solve_generalSmooth_sparse(self.family,y,Xs,form_n_coef,form_up_coef,coef,coef_split_idx,smooth_pen,
                                                                                    max_outer,max_inner,min_inner,conv_tol,extend_lambda,extension_method_lam,
                                                                                    control_lambda,optimizer,method,check_cond,piv_tol,repara,should_keep_drop,form_VH,
                                                                                    use_grad,gamma,qEFSH,overwrite_coef,max_restarts,qEFS_init_converge,prefit_grad,
                                                                                    progress_bar,n_cores,init_bfgs_options,bfgs_options)
        
        self.overall_penalties = smooth_pen
        self.overall_coef = coef
        self.edf = total_edf
        self.overall_term_edf = term_edfs
        self.penalty = penalty
        self.coef_split_idx = coef_split_idx
        self.overall_lvi = LV
        self.hessian = H
        if fit_info.eps > 0: # Make sure -H + S_emb is invertible
            warnings.warn(f"model.info.eps > 0 ({np.round(fit_info.eps,decimals=2)}). Perturbing Hessian of log-likelihood to ensure that negative Hessian of penalized log-likelihood is invertible.")
            self.hessian -= fit_info.eps*scp.sparse.identity(H.shape[1],format='csc')
        self.info = fit_info