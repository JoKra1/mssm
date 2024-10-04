import numpy as np
import scipy as scp
import copy
from collections.abc import Callable
from .src.python.formula import Formula,build_sparse_matrix_from_formula,VarType,lhs,ConstType,Constraint,pd,embed_shared_penalties
from .src.python.exp_fam import Link,Logit,Identity,LOG,Family,Binomial,Gaussian,GAMLSSFamily,GAUMLSS,Gamma,MULNOMLSS,GAMMALS,GENSMOOTHFamily
from .src.python.gamm_solvers import solve_gamm_sparse,mp,repeat,tqdm,cpp_cholP,apply_eigen_perm,compute_Linv,solve_gamm_sparse2,solve_gammlss_sparse,solve_generalSmooth_sparse
from .src.python.terms import TermType,GammTerm,i,f,fs,irf,l,li,ri,rs
from .src.python.penalties import PenType,LambdaTerm
from .src.python.utils import sample_MVN,REML

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
        self.lvi = None
        self.penalty = 0

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
        :return: llk score
        :rtype: float or None
        """

        pen = 0
        if penalized:
            pen = self.penalty
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
        if self.formula.penalties is None:
            raise ValueError("Model matrix cannot be returned if penalties have not been initialized. Call model.fit() or model.formula.build_penalties() first.")
        elif len(self.formula.file_paths) != 0:
            raise NotImplementedError("Cannot return the model-matrix if X.T@X was formed iteratively.")
        else:
            terms = self.formula.get_terms()
            has_intercept = self.formula.has_intercept()
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
    
    def print_smooth_terms(self,pen_cutoff=0.2):
        """Prints the name of the smooth terms included in the model. After fitting, the estimated degrees of freedom per term are printed as well.
        Smooth terms with edf. < ``pen_cutoff`` will be highlighted. This only makes sense when extra Kernel penalties are placed on smooth terms to enable
        penalizing them to a constant zero. In that case edf. < ``pen_cutoff`` can then be taken as evidence that the smooth has all but notationally disappeared
        from the model, i.e., it does not contribute meaningfully to the model fit. This can be used as an alternative form of model selection - see Marra & Wood (2011).

        References:

         - Marra & Wood (2011). Practical variable selection for generalized additive models.

        :param pen_cutoff: At which edf. cut-off smooth terms should be marked as "effectively removed", defaults to None
        :type pen_cutoff: float, optional
        """
        term_names = np.array(self.formula.get_term_names())
        smooth_names = [*term_names[self.formula.get_smooth_term_idx()],
                        *term_names[self.formula.get_random_term_idx()]]
        
        if self.term_edf is None:
            for term in smooth_names:
                print(term)
        else:
            terms = self.formula.get_terms()
            coding_factors = self.formula.get_coding_factors()
            name_idx = 0
            edf_idx = 0
            pen_out = 0
            for sti in self.formula.get_smooth_term_idx():
                sterm = terms[sti]
                if not sterm.by is None and sterm.id is None:
                    for li in range(len(self.formula.get_factor_levels()[sterm.by])):
                        t_edf = round(self.term_edf[edf_idx],ndigits=3)
                        e_str = smooth_names[name_idx] + f": {coding_factors[sterm.by][li]}; edf: {t_edf}"
                        if t_edf < pen_cutoff:
                            # Term has effectively been removed from the model
                            e_str += " *"
                            pen_out += 1
                        print(e_str)
                        edf_idx += 1
                else:
                    t_edf = round(self.term_edf[edf_idx],ndigits=3)
                    e_str = smooth_names[name_idx] + f"; edf: {t_edf}"
                    if t_edf < pen_cutoff:
                        # Term has effectively been removed from the model
                        e_str += " *"
                        pen_out += 1
                    print(e_str)
                    edf_idx += 1
                
                name_idx += 1
            
            for rti in self.formula.get_random_term_idx():
                rterm = terms[rti]
                if isinstance(rterm,rs):
                    if rterm.var_coef > 1 and len(rterm.variables) > 1:
                        for li in range(rterm.var_coef):
                            print(smooth_names[name_idx] + f":{li}; edf: {round(self.term_edf[edf_idx],ndigits=3)}")
                            edf_idx += 1
                    else:
                        print(smooth_names[name_idx] + f"; edf: {round(self.term_edf[edf_idx],ndigits=3)}")
                        edf_idx += 1
                else:
                    print(smooth_names[name_idx] + f"; edf: {round(self.term_edf[edf_idx],ndigits=3)}")
                    edf_idx += 1
                name_idx += 1
            
            if pen_out == 1:
                print("\nOne term has been effectively penalized to zero and is marked with a '*'")
            elif pen_out > 1:
                print(f"\n{pen_out} terms have been effectively penalized to zero and are marked with a '*'")
                        
    def get_reml(self):
        """
        Get's the (Laplace approximate) REML (Restricted Maximum Likelihood) score (as a float) for the estimated lambda values (see Wood, 2011).

        References:

         - Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models: Estimation of Semiparametric Generalized Linear Models.
        
        :raises ValueError: Will throw an error when called before the model was fitted/before model penalties were formed.
        :raises TypeError: Will throw an error when called before the model was fitted/before model penalties were formed.
        :return: REML score
        :rtype: float
        """
        if (not isinstance(self.family,Gaussian) or isinstance(self.family.link,Identity) == False) and self.Wr is None:
            raise TypeError("Model is not Normal and pseudo-dat weights are not avilable. Call model.fit() first!")
        
        if self.coef is None or self.formula.penalties is None:
            raise ValueError("Model needs to be estimated before evaluating the REML score. Call model.fit()")
        
        scale = self.family.scale
        if self.family.twopar:
            scale = self.scale # Estimated scale
        
        X = self.get_mmat()
        llk = self.get_llk(False)

        # Compute negative Hessian of llk (Wood, 2011)
        if not isinstance(self.family,Gaussian) or isinstance(self.family.link,Identity) == False:
            W = self.Wr@self.Wr
            nH = (X.T@W@X).tocsc() 
        else:
            nH = (X.T@X).tocsc() 
            

        reml = REML(llk,nH,self.coef,scale,self.formula.penalties)
        
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
        :type maxiter: str,optional
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
    
    def fit(self,maxiter=50,conv_tol=1e-7,extend_lambda=True,control_lambda=True,exclude_lambda=False,extension_method_lam = "nesterov",restart=False,progress_bar=True,n_cores=10):
        """
        Fit the specified model. Additional keyword arguments not listed below should not be modified unless you really know what you are doing.

        :param maxiter: The maximum number of fitting iterations.
        :type maxiter: int,optional
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
        :param progress_bar: Whether progress should be displayed (convergence info and time estimate). Defaults to True.
        :type progress_bar: bool,optional
        :param n_cores: Number of cores to use during parts of the estimation that can be done in parallel. Defaults to 10.
        :type n_cores: int,optional
        """
        # We need to initialize penalties
        if not restart:
            self.formula.build_penalties()
        penalties = self.formula.penalties

        if penalties is None and restart:
            raise ValueError("Penalties were not initialized. Restart must be set to False.")

        if len(self.formula.file_paths) == 0:
            # We need to build the model matrix once
            terms = self.formula.get_terms()
            has_intercept = self.formula.has_intercept()
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
                for sti in stx:
                    if terms[sti].should_rp:
                        for rpi in range(len(terms[sti].RP)):
                            # Don't need to pass those down to the processes.
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

            if len(irstx) > 0:
                model_mat = model_mat[self.formula.NOT_NA_flat,:]

            # Get initial estimate of mu based on family:
            init_mu_flat = self.family.init_mu(y_flat)

            # Now we have to estimate the model
            coef,eta,wres,Wr,scale,LVI,edf,term_edf,penalty,fit_info = solve_gamm_sparse(init_mu_flat,y_flat,
                                                                                      model_mat,penalties,self.formula.n_coef,
                                                                                      self.family,maxiter,"svd",
                                                                                      conv_tol,extend_lambda,control_lambda,
                                                                                      exclude_lambda,extension_method_lam,
                                                                                      len(self.formula.discretize) == 0,
                                                                                      progress_bar,n_cores)
            
            self.Wr = Wr
        
        else:
            # Iteratively build model matrix.
            # Follows steps in "Generalized additive models for large data sets" (2015) by Wood, Goude, and Shaw
            if not self.formula.get_lhs().f is None:
                raise ValueError("Cannot apply function to dep. var. when building model matrix iteratively. Consider creating a modified variable in the data-frame.")
            
            if isinstance(self.family,Gaussian) == False or isinstance(self.family.link,Identity) == False:
                raise ValueError("Iteratively building the model matrix is currently only supported for Normal models.")
            
            coef,eta,wres,scale,LVI,edf,term_edf,penalty,fit_info = solve_gamm_sparse2(self.formula,penalties,self.formula.n_coef,
                                                                                       self.family,maxiter,"svd",
                                                                                       conv_tol,extend_lambda,control_lambda,
                                                                                       exclude_lambda,extension_method_lam,
                                                                                       len(self.formula.discretize) == 0,
                                                                                       progress_bar,n_cores)
        
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
    
    def __adjust_CI(self,n_ps,b,predi_mat,use_terms,alpha,seed):
        """
        Internal function to adjust point-wise CI to behave like whole-interval (based on Wood, 2017; section 6.10.2 and Simpson, 2016):

        self.coef +- b gives point-wise interval, so for interval to be whole-interval
        1-alpha% of posterior samples should fall completely within these boundaries.

        From section 6.10 in Wood (2017) we have that *coef | y, lambda ~ N(coef,V), where V is self.lvi.T @ self.lvi * self.scale.
        Implication is that deviations [*coef - coef] | y, lambda ~ N(0,V). In line with definition above, 1-alpha% of
        predi_mat@[*coef - coef] should fall within [b,-b]. Wood (2017) suggests to find a so that [a*b,a*-b] achieves this.

        To do this, we find a for every predi_mat@[*coef - coef] and then select the final one so that 1-alpha% of samples had an equal or lower
        one. The consequence: 1-alpha% of samples drawn should fall completely within the modified boundaries.

        References:

         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
         - Simpson, G. (2016). Simultaneous intervals for smooths revisited.
        """
        use_post = None
        if not use_terms is None:
            # If we have many random factor levels, but want to make predictions only
            # for fixed effects, it's wasteful to sample all coefficients from posterior.
            # The code below performs a selection of the coefficients to be sampled.
            use_post = predi_mat.sum(axis=0) != 0
            use_post = np.arange(0,predi_mat.shape[1])[use_post]

        # Sample deviations [*coef - coef] from posterior of GAMM
        post = self.sample_post(n_ps,use_post,deviations=True,seed=seed)

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
        terms = self.formula.get_terms()
        has_intercept = self.formula.has_intercept()
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
            # such intervals. See __adjust_CI function.
            if whole_interval:
                b = self.__adjust_CI(n_ps,b,predi_mat,use_terms,alpha,seed)

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
        # such intervals. See __adjust_CI function.
        if whole_interval:
            b = self.__adjust_CI(n_ps,b,pmat_diff,use_terms,alpha,seed)

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
    :ivar [float] term_edf: The estimated degrees of freedom per smooth term. Initialized with ``None``.
    :ivar scipy.sparse.csc_array lvi: The inverse of the Cholesky factor of the conditional model coefficient covariance matrix. Initialized with ``None``.
    :ivar float penalty: The total penalty applied to the model deviance after fitting as a float. Initialized with ``None``.
    :ivar [int] overall_coef:  Contains all coefficients estimated for the model. Initialized with ``None``.
    :ivar [int] coef_split_idx: The index at which to split the overall coefficient vector into separate lists - one per parameter of ``family``. Initialized after fitting!
    :ivar scp.sparse.csc_array hessian:  Estimated hessian of the log-likelihood. Initialized with ``None``.
    :ivar [LambdaTerm] overall_penalties:  Contains all penalties estimated for the model. Initialized with ``None``.
    """
    def __init__(self, formulas: [Formula], family: GAMLSSFamily):
        super().__init__(formulas[0], family)
        self.formulas = copy.deepcopy(formulas) # self.formula can hold formula for single parameter later on for predictions.
        self.overall_lvi = None
        self.overall_coef = None
        self.overall_preds = None # etas
        self.overall_penalties = None
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
            pen = self.penalty
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
        Xs = []
        for form in self.formulas:
            if form.penalties is None:
                raise ValueError("Model matrices cannot be returned if penalties have not been initialized. Call model.fit() first.")
            
            mod = GAMM(form,family=Gaussian())
            Xs.append(mod.get_mmat(use_terms=use_terms,drop_NA=drop_NA))
        return Xs
    
    def print_smooth_terms(self, pen_cutoff=0.2):
        """Prints the name of the smooth terms included in the model. After fitting, the estimated degrees of freedom per term are printed as well.
        Smooth terms with edf. < ``pen_cutoff`` will be highlighted. This only makes sense when extra Kernel penalties are placed on smooth terms to enable
        penalizing them to a constant zero. In that case edf. < ``pen_cutoff`` can then be taken as evidence that the smooth has all but notationally disappeared
        from the model, i.e., it does not contribute meaningfully to the model fit. This can be used as an alternative form of model selection - see Marra & Wood (2011).

        References:

         - Marra & Wood (2011). Practical variable selection for generalized additive models.

        :param pen_cutoff: At which edf. cut-off smooth terms should be marked as "effectively removed", defaults to None
        :type pen_cutoff: float, optional
        """
        idx = 0
        start_idx = -1
        pen_idx = 0
        n_coef = 0
        for form in self.formulas:
            # Prepare formula, term_edf so that we can just call GAMM.print_smooth_terms()
            n_coef += form.n_coef
            self.term_edf = []
            for peni in range(pen_idx,len(self.overall_penalties)):
                if self.overall_penalties[peni].start_index >= n_coef:
                    break
                elif self.overall_penalties[peni].start_index > start_idx:
                    self.term_edf.append(self.overall_term_edf[idx])
                    idx += 1
                    start_idx = self.overall_penalties[peni].start_index
                pen_idx += 1
            self.formula = form
            super().print_smooth_terms(pen_cutoff)
                        
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
        
        reml = REML(llk,-1*self.hessian,self.overall_coef,1,self.overall_penalties)[0,0]
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
        

    def fit(self,max_outer=50,max_inner=50,min_inner=50,conv_tol=1e-7,extend_lambda=True,control_lambda=True,progress_bar=True,n_cores=10,seed=0):
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
        :param control_lambda: Whether lambda proposals should be checked (and if necessary decreased) for whether or not they (approxiately) increase the Laplace approximate restricted maximum likelihood of the model. Enabled by default.
        :type control_lambda: bool,optional
        :param progress_bar: Whether progress should be displayed (convergence info and time estimate). Defaults to True.
        :type progress_bar: bool,optional
        :param n_cores: Number of cores to use during parts of the estimation that can be done in parallel. Defaults to 10.
        :type n_cores: int,optional
        :param seed: Seed to use for random parameter initialization. Defaults to 0
        :type seed: int,optional
        """
        
        # Get y
        y = self.formulas[0].y_flat[self.formulas[0].NOT_NA_flat]

        if not self.formulas[0].get_lhs().f is None:
            # Optionally apply function to dep. var. before fitting. Not sure why that would be desirable for this model class...
            y = self.formulas[0].get_lhs().f(y)

        # Build penalties and model matrices for all formulas
        Xs = []
        for form in self.formulas:
            mod = GAMM(form,family=Gaussian())
            form.build_penalties()
            Xs.append(mod.get_mmat())

        # Fit mean model
        if self.family.mean_init_fam is not None:
            mean_model = GAMM(self.formulas[0],family=self.family.mean_init_fam)
            mean_model.fit(progress_bar=False,restart=True)
            m_coef,_ = mean_model.get_pars()
        else:
            m_coef = scp.stats.norm.rvs(size=self.formulas[0].n_coef,random_state=seed).reshape(-1,1)

        # Get GAMMLSS penalties
        shared_penalties = embed_shared_penalties(self.formulas)
        gamlss_pen = [pen for pens in shared_penalties for pen in pens]
        self.overall_penalties = gamlss_pen

        # Start with much weaker penalty than for GAMs
        for pen_i in range(len(gamlss_pen)):
            gamlss_pen[pen_i].lam = 0.01

        # Initialize overall coefficients
        form_n_coef = [form.n_coef for form in self.formulas]
        coef = np.concatenate((m_coef.reshape(-1,1),
                            *[np.ones((self.formulas[ix].n_coef)).reshape(-1,1) for ix in range(1,self.family.n_par)]))
        coef_split_idx = form_n_coef[:-1]

        for coef_i in range(1,len(coef_split_idx)):
            coef_split_idx[coef_i] += coef_split_idx[coef_i-1]

        coef,etas,mus,wres,H,LV,total_edf,term_edfs,penalty = solve_gammlss_sparse(self.family,y,Xs,form_n_coef,coef,coef_split_idx,
                                                                                 gamlss_pen,max_outer,max_inner,min_inner,conv_tol,
                                                                                 extend_lambda,control_lambda,progress_bar,n_cores)
        
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
            
            end = self.coef_split_idx[0]
            for pari in range(1,par+1):
                start = end
                end += self.formulas[pari].n_coef
            self.lvi = self.overall_lvi[:,start:end]
        
        post = super().sample_post(n_ps, use_post, deviations, seed)

        # Clean up
        self.coef = None
        self.scale = None
        self.lvi = None
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

        Standard errors cannot currently be computed for Multinomial GAMMLSS models - attempting to set ``ci=True`` for such a model will result in an error.

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
        if isinstance(self.family,MULNOMLSS) and ci == True:
            raise ValueError("Standard error computation for Multinomial model is not currently supported.")
        
        # Prepare so that we can just call gamm.predict()    
        self.formula = self.formulas[par]
        split_coef = np.split(self.overall_coef,self.coef_split_idx)
        self.coef = np.ndarray.flatten(split_coef[par])
        self.scale=1
        start = 0
        
        end = self.coef_split_idx[0]
        for pari in range(1,par+1):
            start = end
            end += self.formulas[pari].n_coef
        self.lvi = self.overall_lvi[:,start:end]

        pred = super().predict(use_terms, n_dat, alpha, ci, whole_interval, n_ps, seed)

        # Clean up
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
        if isinstance(self.family,MULNOMLSS):
            raise ValueError("Standard error computation for Multinomial model is not currently supported.")
        
        # Prepare so that we can just call gamm.predict_diff()
        self.formula = self.formulas[par]
        split_coef = np.split(self.overall_coef,self.coef_split_idx)
        self.coef = np.ndarray.flatten(split_coef[par])
        self.scale=1
        start = 0
        
        end = self.coef_split_idx[0]
        for pari in range(1,par+1):
            start = end
            end += self.formulas[pari].n_coef
        self.lvi = self.overall_lvi[:,start:end]

        pred_diff = super().predict_diff(dat1, dat2, use_terms, alpha, whole_interval, n_ps, seed)

        # Clean up
        self.coef = None
        self.scale = None
        self.lvi = None

        return pred_diff
    

class GSMM(GAMMLSS):
    """
    Class to fit General Smooth/Mixed Models (see Wood, Pya, & Säfken; 2016). Estimation is possible via exact Newton method for coefficients of via BFGS (see example below).

    Example::

        class NUMDIFFGENSMOOTHFamily(GENSMOOTHFamily):
            # Implementation of the ``GENSMOOTHFamily`` class that uses ``numdifftools`` to obtain the
            # gradient and hessian of the likelihood to estimate a Gaussian GAMLSS via the general smooth code.

            # For BFGS :func:``gradient`` and :func:``hessian`` can also just return None.

            # References:

            #    - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
            #    - P. Brodtkorb (2014). numdifftools. see https://github.com/pbrod/numdifftools
            #    - Nocedal & Wright (2006). Numerical Optimization. Springer New York.
            

            def __init__(self, pars: int, links:[Link], llkfun:Callable, *llkargs) -> None:
                super().__init__(pars, links, *llkargs)
                self.llkfun = llkfun
            
            def llk(self, coef, coef_split_idx, y, Xs):
                return self.llkfun(coef, coef_split_idx, self.links, y, Xs,*self.llkargs)
            
            def gradient(self, coef, coef_split_idx, y, Xs):
                return Gradient(self.llkfun)(np.ndarray.flatten(coef),coef_split_idx,self.links,y,Xs,*self.llkargs)
            
            def hessian(self, coef, coef_split_idx, y, Xs):
                return scp.sparse.csc_array(Hessian(self.llkfun)(np.ndarray.flatten(coef),coef_split_idx,self.links,y,Xs))
                

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
        GAUMLSSDat = sim6(500,seed=20)

        # We need to model the mean: \mu_i = \alpha + f(x0)
        formula_m = Formula(lhs("y"),
                        [i(),f(["x0"],nk=10)],
                        data=GAUMLSSDat)

        # and the standard deviation as well: log(\sigma_i) = \alpha + f(x0)
        formula_sd = Formula(lhs("y"),
                        [i(),f(["x0"],nk=10)],
                        data=GAUMLSSDat)

        # Collect both formulas
        formulas = [formula_m,formula_sd]
        links = [Identity(),LOG()]

        # Now define the general family + model and fit!
        gsmm_fam = NUMDIFFGENSMOOTHFamily(2,links,llk_gamm_fun)
        model = GSMM(formulas=formulas,family=gsmm_fam)

        # First fit with bfgs and, to speed things up, only then with Newton
        model.fit(init_coef=None,method="BFGS",extend_lambda=False,max_outer=100,seed=10,conv_tol=1e-3)

        # Use BFGS estimate as initial estimate for Newton model
        coef = model.overall_coef

        # Now re-fit with full Newton
        model.fit(init_coef=coef,method="Newton",extend_lambda=False,max_outer=100,seed=10,conv_tol=1e-7,restart=True)

    References:

     - Wood, S. N., & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. https://doi.org/10.1111/biom.12666
     - Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models: Estimation of Semiparametric Generalized Linear Models. https://doi.org/10.1111/j.1467-9868.2010.00749.x
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
     - Nocedal & Wright (2006). Numerical Optimization. Springer New York.
    
    
    :param formulas: A list of formulas, one per parameter of the likelihood that is to be modeled as a smooth model
    :type formulas: [Formula]
    :param family: A GENSMOOTHFamily family.
    :type family: GENSMOOTHFamily
    :ivar float edf: The model estimated degrees of freedom as a float. Initialized with ``None``.
    :ivar [float] term_edf: The estimated degrees of freedom per smooth term. Initialized with ``None``.
    :ivar scipy.sparse.csc_array lvi: The inverse of the Cholesky factor of the conditional model coefficient covariance matrix. Initialized with ``None``.
    :ivar float penalty: The total penalty applied to the model deviance after fitting as a float. Initialized with ``None``.
    :ivar [int] overall_coef:  Contains all coefficients estimated for the model. Initialized with ``None``.
    :ivar [int] coef_split_idx: The index at which to split the overall coefficient vector into separate lists - one per parameter of ``family``. Initialized after fitting!
    :ivar scp.sparse.csc_array hessian:  Estimated hessian of the log-likelihood. Initialized with ``None``.
    :ivar [LambdaTerm] overall_penalties:  Contains all penalties estimated for the model. Initialized with ``None``.
    """

    def __init__(self, formulas: [Formula], family: GENSMOOTHFamily):
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
            pen = self.penalty
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
        
        reml = REML(llk,-1*self.hessian,self.overall_coef,1,self.overall_penalties)[0,0]
        return reml
    
    def get_resid(self):
        """What qualifies as "residual" will differ vastly between different implementations of this class, so this method simply returns ``None``.
        """
        return None
    
    def fit(self,init_coef=None,max_outer=50,max_inner=50,min_inner=50,conv_tol=1e-7,extend_lambda=True,control_lambda=True,restart=False,progress_bar=True,n_cores=10,seed=0,method="Newton",drop_NA=True):
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
        :param control_lambda: Whether lambda proposals should be checked (and if necessary decreased) for whether or not they (approxiately) increase the Laplace approximate restricted maximum likelihood of the model. Enabled by default.
        :type control_lambda: bool,optional
        :param restart: Whether fitting should be resumed. Only possible if the same model has previously completed at least one fitting iteration.
        :type restart: bool,optional
        :param progress_bar: Whether progress should be displayed (convergence info and time estimate). Defaults to True.
        :type progress_bar: bool,optional
        :param n_cores: Number of cores to use during parts of the estimation that can be done in parallel. Defaults to 10.
        :type n_cores: int,optional
        :param seed: Seed to use for random parameter initialization. Defaults to 0
        :type seed: int,optional
        :param method: Which method to use to estimate the coefficients - supports "Newton" and "BFGS". In case of the former, ``self.family`` needs to implement :func:``gradient`` and :func:``hessian``. Defaults to "Newton"
        :type method: str,optional
        :param drop_NA: Whether to drop rows in the **model matrices** corresponding to NAs in the dependent variable vector. Defaults to True.
        :type drop_NA: bool,optional
        """
        
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
        for form in self.formulas:
            mod = GAMM(form,family=Gaussian())
            if self.overall_penalties is None or restart == False:
                form.build_penalties()
            Xs.append(mod.get_mmat(drop_NA=drop_NA))

        # Get all penalties
        shared_penalties = embed_shared_penalties(self.formulas)
        shared_penalties = [sp for sp in shared_penalties if len(sp) > 0]

        smooth_pen = [pen for pens in shared_penalties for pen in pens]
        self.overall_penalties = smooth_pen

        # Start with much weaker penalty than for GAMs
        for pen_i in range(len(smooth_pen)):
            smooth_pen[pen_i].lam = 0.001

        # Optionally Initialize overall coefficients
        form_n_coef = [form.n_coef for form in self.formulas]
        n_coef = np.sum(form_n_coef)

        if not init_coef is None:
            coef = np.array(init_coef).reshape(-1,1)
        else:
            coef = scp.stats.norm.rvs(size=n_coef,random_state=seed).reshape(-1,1)

        coef_split_idx = form_n_coef[:-1]
        for coef_i in range(1,len(coef_split_idx)):
            coef_split_idx[coef_i] += coef_split_idx[coef_i-1]
        
        # Now fit model
        coef,H,LV,total_edf,term_edfs,penalty = solve_generalSmooth_sparse(self.family,y,Xs,form_n_coef,coef,coef_split_idx,smooth_pen,
                                                                           max_outer,max_inner,min_inner,conv_tol,extend_lambda,
                                                                           control_lambda,progress_bar,n_cores,method)
        
        self.overall_coef = coef
        self.edf = total_edf
        self.term_edf = term_edfs
        self.penalty = penalty
        self.coef_split_idx = coef_split_idx
        self.overall_lvi = LV
        self.hessian = H