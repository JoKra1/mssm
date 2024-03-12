import numpy as np
import scipy as scp
import copy
from collections.abc import Callable
from .src.python.formula import Formula,PFormula,PTerm,build_sparse_matrix_from_formula,VarType,lhs,ConstType,Constraint,pd
from .src.python.exp_fam import Link,Logit,Family,Binomial,Gaussian
from .src.python.sem import anneal_temps_zero,const_temps,compute_log_probs,pre_ll_sms_gamm,se_step_sms_gamm,decode_local,se_step_sms_dc_gamm,pre_ll_sms_IR_gamm,init_states_IR,compute_hsmm_probabilities
from .src.python.gamm_solvers import solve_gamm_sparse,mp,repeat,tqdm,cpp_cholP,apply_eigen_perm,compute_Linv,solve_gamm_sparse2
from .src.python.terms import TermType,GammTerm,i,f,fs,irf,l,li,ri,rs
from .src.python.penalties import PenType

##################################### Base Class #####################################

class MSSM:

    def __init__(self,
                 formula:Formula,
                 family:Family,
                 pre_llk_fun:Callable = None,
                 estimate_pi:bool=False,
                 estimate_TR:bool=False,
                 cpus:int=1):
        
        # Formulas associated with model
        self.formula = formula # For coefficients
        self.p_formula = self.formula.p_formula # For sojourn time distributions

        # Family of model
        self.family = family
        
        ## "prior" Log-likelihood functions
        self.pre_llk_fun = pre_llk_fun

        ## Should transition matrices and initial distribution be estimated?
        self.estimate_pi = estimate_pi
        self.estimate_TR = estimate_TR

        self.cpus=cpus

        # Current estimates
        self.__coef = None
        self.__scale = None
        self.__TR = None
        self.__pi = None
    
    ##################################### Getters #####################################
    
    def get_pars(self):
        pass
    
    def get_llk(self):
        pass
    
    ##################################### Fitting #####################################

    def fit(self):
        pass

    ##################################### Prediction #####################################

    def predict(self,terms,n_dat):
        pass

##################################### GAMM class #####################################

class GAMM(MSSM):
    """
    Class to fit Generalized Additive Mixed Models.

    References:
    - Wood, S. N., & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. https://doi.org/10.1111/biom.12666
    - Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models: Estimation of Semiparametric Generalized Linear Models. https://doi.org/10.1111/j.1467-9868.2010.00749.x
    - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

    Parameters:
    :param formula: A formula for the GAMM model
    :type variables: Formula
    :param family: An exponential family. Currently only ``Gaussian`` or ``Binomial`` are implemented.
    :type Family
    :param pred: A np.array holding the model prediction for the training data. Of the same dimension as ``self.formula.__lhs``.
    :type pred: np.array,optional
    :param res: A np.array holding the working residuals for the training data. Of the same dimension as ``self.formula.__lhs``.
    :type res: np.array,optional
    :param edf: The model estimated degrees of freedom.
    :type edf: float,optional
    :param term_edf: The estimated degrees of freedom per smooth term.
    :type term_edf: list[float],optional
    :param lvi: The inverse of the Cholesky factor of the model coefficient covariance matrix. 
    :type lvi: scipy.sparse.csc_array,optional
    :param penalty: The total penalty applied to the model deviance after fitting.
    :type penalty: float,optional
    """

    def __init__(self,
                 formula: Formula,
                 family: Family):
        super().__init__(formula,family)

        self.pred = None
        self.res = None
        self.edf = None
        self.term_edf = None

        self.lvi = None
        self.penalty = 0

    ##################################### Getters #####################################

    def get_pars(self):
        """ Returns a tuple. The first entry is a np.array with all estimated coefficients. The second entry is the estimated scale parameter. Will contain Nones before fitting."""
        return self.__coef,self.__scale
    
    def get_llk(self,penalized:bool=True):
        """Get the (penalized) log-likelihood of the estimated model given the trainings data."""

        pen = 0
        if penalized:
            pen = self.penalty
        if self.pred is not None:
            mu = self.pred
            if isinstance(self.family,Gaussian) == False:
                mu = self.family.link.fi(self.pred)
            if self.family.twopar:
                return self.family.llk(self.formula.y_flat[self.formula.NOT_NA_flat],mu,self.__scale) - pen
            else:
                return self.family.llk(self.formula.y_flat[self.formula.NOT_NA_flat],mu) - pen
        return None

    def get_mmat(self):
        """
        Returns exaclty the model matrix used for fitting as a scipy.sparse.csc_array.
        """
        if self.formula.penalties is None:
            raise ValueError("Model matrix cannot be returned if penalties have not been initialized. Call model.fit() or model.formula.build_penalties() first.")
        else:
            terms = self.formula.get_terms()
            has_intercept = self.formula.has_intercept()
            has_scale_split = False
            ltx = self.formula.get_linear_term_idx()
            irstx = []
            stx = self.formula.get_smooth_term_idx()
            rtx = self.formula.get_random_term_idx()
            var_types = self.formula.get_var_types()
            var_map = self.formula.get_var_map()
            var_mins = self.formula.get_var_mins()
            var_maxs = self.formula.get_var_maxs()
            factor_levels = self.formula.get_factor_levels()
            cov_flat = self.formula.cov_flat[self.formula.NOT_NA_flat]

            cov = None
            n_j = None
            state_est_flat = None
            state_est = None

            # Build the model matrix with all information from the formula
            model_mat = build_sparse_matrix_from_formula(terms,has_intercept,has_scale_split,
                                                        ltx,irstx,stx,rtx,var_types,var_map,
                                                        var_mins,var_maxs,factor_levels,
                                                        cov_flat,cov,n_j,state_est_flat,state_est)
            
            return model_mat
    
    def print_smooth_terms(self,pen_cutoff=0.2):
        """Prints the name of the smooth terms included in the model. After fitting, the estimated degrees of freedom per term are printed as well."""
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
            
            for _ in self.formula.get_random_term_idx():
                print(smooth_names[name_idx] + f"; edf: {self.term_edf[edf_idx]}")
                edf_idx += 1
                name_idx += 1
            
            if pen_out == 1:
                print("\nOne term has been effectively penalized to zero and is marked with a '*'")
            elif pen_out > 1:
                print(f"\n{pen_out} terms have been effectively penalized to zero and are marked with a '*'")
                        

                
    ##################################### Fitting #####################################
    
    def fit(self,maxiter=50,conv_tol=1e-7,extend_lambda=True,control_lambda=True,exclude_lambda=True,restart=False,progress_bar=True,n_cores=10):
        """
        Fit the specified model.

        Parameters:

        :param maxiter: The maximum number of fitting iterations.
        :type maxiter: int,optional
        :param conv_tol: The relative (change in penalized deviance is compared against ``conv_tol`` * previous penalized deviance) criterion used to determine convergence.
        :type conv_tol: float,optional
        :param extend_lambda: Whether lambda proposals should be accelerated or not. Can lower the number of new smoothing penalty proposals necessary. Enabled by default.
        :type extend_lambda: bool,optional
        :param control_lambda: Whether lambda proposals should be checked (and if necessary decreased) for actually improving the Restricted maximum likelihood of the model. Can lower the number of new smoothing penalty proposals necessary. Enabled by default.
        :type extend_lambda: bool,optional
        :param restart: Whether fitting should be resumed. Only possible if the same model has previously completed at least one fitting iteration.
        :type restart: bool,optional
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
            has_scale_split = False
            ltx = self.formula.get_linear_term_idx()
            irstx = []
            stx = self.formula.get_smooth_term_idx()
            rtx = self.formula.get_random_term_idx()
            var_types = self.formula.get_var_types()
            var_map = self.formula.get_var_map()
            var_mins = self.formula.get_var_mins()
            var_maxs = self.formula.get_var_maxs()
            factor_levels = self.formula.get_factor_levels()

            cov = None
            n_j = None
            state_est_flat = None
            state_est = None
            cov_flat = self.formula.cov_flat[self.formula.NOT_NA_flat]
            y_flat = self.formula.y_flat[self.formula.NOT_NA_flat]

            if y_flat.shape[0] != self.formula.y_flat.shape[0] and progress_bar:
                print("NAs were excluded for fitting.")

            # Build the model matrix with all information from the formula
            model_mat = build_sparse_matrix_from_formula(terms,has_intercept,has_scale_split,
                                                         ltx,irstx,stx,rtx,var_types,var_map,
                                                         var_mins,var_maxs,factor_levels,
                                                         cov_flat,cov,n_j,state_est_flat,state_est)

            # Get initial estimate of mu based on family:
            init_mu_flat = self.family.init_mu(y_flat)

            # Now we have to estimate the model
            coef,eta,wres,scale,LVI,edf,term_edf,penalty = solve_gamm_sparse(init_mu_flat,y_flat,
                                                                             model_mat,penalties,self.formula.n_coef,
                                                                             self.family,maxiter,"svd",
                                                                             conv_tol,extend_lambda,control_lambda,
                                                                             exclude_lambda,progress_bar,n_cores)
        
        else:
            # Iteratively build model matrix.
            # Follows steps in "Generalized additive models for large data sets" (2015) by Wood, Goude, and Shaw
            if isinstance(self.family,Gaussian) == False:
                raise ValueError("Iteratively building the model matrix is currently only supported for Normal models.")
            
            coef,eta,wres,scale,LVI,edf,term_edf,penalty = solve_gamm_sparse2(self.formula,penalties,self.formula.n_coef,
                                                                              self.family,maxiter,"svd",
                                                                              conv_tol,extend_lambda,control_lambda,
                                                                              exclude_lambda,progress_bar,n_cores)
        
        self.__coef = coef
        self.__scale = scale # ToDo: scale name is used in another context for more general mssm..
        self.pred = eta
        self.res = wres
        self.edf = edf
        self.term_edf = term_edf
        self.penalty = penalty

        self.lvi = LVI
    
    ##################################### Prediction #####################################

    def predict(self, use_terms, n_dat,alpha=0.05,ci=False):
        """
        Make a prediction using the fitted model for new data ``n_dat`` using only the terms indexed by ``use_terms``.

        References:
        - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

        Parameters:

        :param use_terms: The indices corresponding to the terms that should be used to obtain the prediction or ``None``
        in which case all terms will be used.
        :type use_terms: list[int] or None
        :param n_dat: A pandas DataFrame containing new data for which to make the prediction. Importantly, all variables present in
        the data used to fit the model also need to be present in this DataFrame. Additionally, factor variables must only include levels
        also present in the data used to fit the model. If you want to exclude a specific factor from the prediction (for example the factor
        subject) don't include the terms that involve it in the ``use_terms`` argument.
        :type n_dat: pd.DataFrame
        :param alpha: The alpha level to use for the standard error calculation. Specifically, 1 - (``alpha``/2) will be used to determine the critical cut-off value according to a N(0,1).
        :type alpha: float, optional
        :param ci: Whether the standard error ``se`` for credible interval (CI; see  Wood, 2017) calculation should be returned. The CI is then [``pred`` - ``se``, ``pred`` + ``se``]
        :type alpha: bool, optional

        Returns:
        :return: A tuple with 3 entries. The first entry is the prediction ``pred`` based on the new data ``n_dat``. The second entry is the model
        matrix built for ``n_dat`` that was post-multiplied with the model coefficients to obtain ``pred``. The third entry is ``None`` if ``ci``==``False`` else
        the standard error ``se`` in the prediction.
        :rtype: tuple
        
        """
        var_map = self.formula.get_var_map()
        var_keys = var_map.keys()

        for k in var_keys:
            if k not in n_dat.columns:
                raise IndexError(f"Variable {k} is missing in new data.")
        
        # Encode test data
        _,pred_cov_flat,_,_,_,_,_ = self.formula.encode_data(n_dat,prediction=True)

        # Then, we need to build the model matrix - but only for the terms which should
        # be included in the prediction!
        terms = self.formula.get_terms()
        has_intercept = self.formula.has_intercept()
        has_scale_split = False
        ltx = self.formula.get_linear_term_idx()
        irstx = []
        stx = self.formula.get_smooth_term_idx()
        rtx = self.formula.get_random_term_idx()
        var_types = self.formula.get_var_types()
        var_mins = self.formula.get_var_mins()
        var_maxs = self.formula.get_var_maxs()
        factor_levels = self.formula.get_factor_levels()
        n_j = None
        state_est_flat = None
        state_est = None

        # So we pass the desired terms to the use_only argument
        predi_mat = build_sparse_matrix_from_formula(terms,has_intercept,has_scale_split,
                                                     ltx,irstx,stx,rtx,var_types,var_map,
                                                     var_mins,var_maxs,factor_levels,
                                                     pred_cov_flat,None,n_j,state_est_flat,
                                                     state_est,use_only=use_terms)
        
        # Now we calculate the prediction
        pred = predi_mat @ self.__coef

        # Optionally calculate the boundary for a 1-alpha CI
        if ci:
            # Wood (2017) 6.10
            c = predi_mat @ self.lvi.T @ self.lvi * self.__scale @ predi_mat.T
            c = c.diagonal()
            b = scp.stats.norm.ppf(1-(alpha/2)) * np.sqrt(c)
            return pred,predi_mat,b

        return pred,predi_mat,None
    
    def predict_diff(self,dat1,dat2,use_terms,alpha=0.05):
        """
        Get the difference in the predictions for two datasets. Useful to compare a smooth estimated for
        one level of a factor to the smooth estimated for another level of a factor. In that case, ``dat1`` and
        ``dat2`` should only differ in the level of said factor.

        References:
        - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
        - ``get_difference`` function from ``itsadug`` R-package: https://rdrr.io/cran/itsadug/man/get_difference.html

        Parameters:

        :param dat1: A pandas DataFrame containing new data for which to make the prediction. Importantly, all variables present in
        the data used to fit the model also need to be present in this DataFrame. Additionally, factor variables must only include levels
        also present in the data used to fit the model. If you want to exclude a specific factor from the prediction (for example the factor
        subject) don't include the terms that involve it in the ``use_terms`` argument.
        :type n_dat: pd.DataFrame
        :param dat2: A second pandas DataFrame for which to also make a prediction. The difference in the prediction between this ``dat1`` will be returned.
        :type dat2: pd.DataFrame
        :param use_terms: The indices corresponding to the terms that should be used to obtain the prediction or ``None``
        in which case all terms will be used.
        :type use_terms: list[int] or None
        :param alpha: The alpha level to use for the standard error calculation. Specifically, 1 - (``alpha``/2) will be used to determine the critical cut-off value according to a N(0,1).
        :type alpha: float, optional

        Returns:
        :return: A tuple with 2 entries. The first entry is the predicted difference (between the two data sets ``dat1`` & ``dat2``) ``pred``. The second entry is the standard error ``se`` of the predicted difference..
        :rtype: tuple
        """
        _,pmat1,_ = self.predict(use_terms,dat1)
        _,pmat2,_ = self.predict(use_terms,dat2)

        pmat_diff = pmat1 - pmat2
        
        # Predicted difference
        diff = pmat_diff @ self.__coef
        
        # Difference CI
        c = pmat_diff @ self.lvi.T @ self.lvi * self.__scale @ pmat_diff.T
        c = c.diagonal()
        b = scp.stats.norm.ppf(1-(alpha/2)) * np.sqrt(c)

        return diff,b

class sMsGAMM(MSSM):
    # Class to fit semi-Markov-switching Generalized Additive Mixed Models.
    # see: Langrock, R., Kneib, T., Glennie, R., & Michelot, T. (2017). Markov-switching generalized additive models. https://doi.org/10.1007/s11222-015-9620-3
    # and: Haji-Maghsoudi, S., Bulla, J., Sadeghifar, M., Roshanaei, G., & Mahjub, H. (2021). Generalized linear mixed hidden semi-Markov models in longitudinal settings: A Bayesian approach. https://doi.org/10.1002/sim.8908
    # for an introduction to similar models.

    def __init__(self,
                 formula: Formula,
                 family: Family,
                 end_points:list,
                 pre_llk_fun=pre_ll_sms_gamm,
                 estimate_pi: bool = True,
                 estimate_TR: bool = True,
                 pi=None,
                 TR=None,
                 mvar_by=None,
                 cpus: int = 1):
        
        super().__init__(formula,
                         family,
                         pre_llk_fun,
                         estimate_pi,
                         estimate_TR,
                         cpus)
        
        # Multivariate indicator factor.
        self.mvar_by = mvar_by
        if not self.mvar_by is None:
            if not self.mvar_by in self.formula.get_var_types():
                raise KeyError(f"Multivariate factor {self.mvar_by} does not exist in the data.")
            if self.formula.get_var_types()[self.mvar_by] != VarType.FACTOR:
                raise ValueError(f"Multivariate variable {self.mvar_by} is not a factor.")
        
        self.end_points= end_points
        self.n_j = formula.get_nj()
        
        # Check the sojourn time distributions provided
        self.pds:list[PTerm] = None
        self.__check_pds()

        # Create starting TR & pi
        if estimate_pi == False and pi is None:
            raise ValueError("If estimate_pi==False, pi needs to be provided.")
        if estimate_TR == False and TR is None:
            raise ValueError("If estimate_TR==False, TR needs to be provided.")
        
        self.__init_tr_pi(pi,TR)
        self.__coef = None
        self.__scale = None

    ##################################### Getters #####################################

    def get_pars(self):
        return self.__coef,self.__scale,self.__TR,self.__pi
    
    ##################################### Inits & Checks #####################################

    def __init_tr_pi(self,prov_pi,prov_TR):
        # Initialize initial distribution of states to
        # equal probability to be in any state
        # and transition matrix to have equal likelihood to transition
        # anywhere.
        if self.estimate_pi:
            self.__pi = np.zeros(self.n_j) + 1 / self.n_j
        else:
            self.__pi = prov_pi

        if self.estimate_TR:
            self.__TR = np.zeros((self.n_j,self.n_j)) + 1 / (self.n_j - 1)
            for j in range(self.n_j):
                self.__TR[j,j] = 0
        else:
            self.__TR = prov_TR
    
    def __check_pds(self):
        # Perform some check on the sojourn time
        # distributions passed to the constructor.
        # Also set the number of levels associated with
        # by_split factor variables!
        self.pds = self.p_formula.get_terms()
        var_types = self.formula.get_var_types()
        var_levels = self.formula.get_factor_levels()

        if len(self.pds) != self.n_j:
            raise ValueError("A sojourn time distribution for every state needs to be provided.")

        # Check that splt_by vars exist in the variable structure of formula.
        for pTerm in self.pds:
            if not pTerm.split_by is None:
                if pTerm.split_by not in var_types:
                    raise KeyError(f"Variable {pTerm.split_by} used as split_by argument does not exist in data.")
                if var_types[pTerm.split_by] != VarType.FACTOR:
                    raise ValueError(f"Variable {pTerm.split_by} used as split_by argument is not a factor variable.")
                pTerm.n_by = len(var_levels[pTerm.split_by])
    
    def get_mmat_full(self):
        """
        Returns the full model matrix for all collected observations as a scipy.sparse.csc_array.
        """
        has_scale_split = self.formula.has_scale_split()

        if self.formula.penalties is None:
            raise ValueError("Model matrix cannot be returned if penalties have not been initialized. Call model.fit() or model.formula.build_penalties() first.")
        else:
            # Model matrix parameters that remain constant are specified.
            # And then we need to build the model matrix once for the entire
            # data so that we can later get observation probabilities for the
            # entire data.
            terms = self.formula.get_terms()
            has_intercept = self.formula.has_intercept()
            ltx = self.formula.get_linear_term_idx()
            irstx = []
            stx = self.formula.get_smooth_term_idx()
            rtx = self.formula.get_random_term_idx()
            var_types = self.formula.get_var_types()
            var_map = self.formula.get_var_map()
            var_mins = self.formula.get_var_mins()
            var_maxs = self.formula.get_var_maxs()
            factor_levels = self.formula.get_factor_levels()
            NOT_NA_flat = self.formula.NOT_NA_flat
            cov_flat = self.formula.cov_flat

            model_mat_full = build_sparse_matrix_from_formula(terms,has_intercept,has_scale_split,
                                                            ltx,irstx,stx,rtx,var_types,var_map,
                                                            var_mins,var_maxs,factor_levels,
                                                            cov_flat[NOT_NA_flat],None,self.n_j,
                                                            None,None)
            
            return model_mat_full

    ##################################### Fitting #####################################
    
    def __propose_all_states(self,pool,cov,temp,pi,TR,log_o_probs,log_dur_probs,var_map):
        # MP code to propose states for every series
        args = zip(repeat(self.n_j),repeat(temp),cov,self.end_points,repeat(pi),
                   repeat(TR),log_o_probs,repeat(log_dur_probs),repeat(self.pds),
                   repeat(self.pre_llk_fun),repeat(var_map))
        
        state_durs_new, states_new, llks = zip(*pool.starmap(se_step_sms_gamm,args))
        return list(state_durs_new),list(states_new),list(llks)
    
    def __decode_all_states(self,pool,cov,pi,TR,log_o_probs,log_dur_probs,var_map):
        # MP code to decode final states for every series
        args = zip(repeat(self.n_j),cov,repeat(pi),repeat(TR),log_o_probs,
                   repeat(log_dur_probs),repeat(self.pds),repeat(var_map))
        
        state_durs_decoded, states_decoded, llks_decoded = zip(*pool.starmap(decode_local,args))
        return list(state_durs_decoded),list(states_decoded),list(llks_decoded)
    
    def compute_all_probs(self,pool,cov,pi,TR,log_o_probs,log_dur_probs,var_map):
        # MP code to decode final states for every series
        args = zip(repeat(self.n_j),cov,repeat(pi),repeat(TR),log_o_probs,
                   repeat(log_dur_probs),repeat(self.pds),repeat(var_map),repeat(True))
        
        llks_decoded, etas, gammas, smoothed = zip(*pool.starmap(compute_hsmm_probabilities,args))
        return list(etas),list(gammas),list(smoothed),list(llks_decoded)
    
    def fit(self,burn_in=100,maxiter_inner=30,m_avg=15,conv_tol=1e-7,extend_lambda=True,control_lambda=True,exclude_lambda=False,init_scale=100,t0=0.25,r=0.925,progress_bar=True):
        # Performs Stochastic Expectation maiximization based on Nielsen (2002) see also the sem.py file for
        # more details as well as:
        # Ref:
        # 1. Allassonnière, S., & Chevallier, J. (2021). A new class of stochastic EM algorithms. Escaping local maxima and handling intractable sampling. https://doi.org/10.1016/j.csda.2020.107159
        # 2. Celeux, G., Chauveau, D., & Diebolt, J. (1992). On Stochastic Versions of the EM Algorithm. https://doi.org/10.1177/075910639203700105
        # 3. Delyon, B., Lavielle, M., & Moulines, E. (1999). Convergence of a stochastic approximation version of the EM algorithm. https://doi.org/10.1214/aos/1018031103
        
        has_scale_split = self.formula.has_scale_split()
        
        # Penalties need to be initialized and copied for each stage
        self.formula.build_penalties()
        
        if has_scale_split:
            penalties = [copy.deepcopy(self.formula.penalties) for j in range(self.n_j)]
        else:
            penalties = self.formula.penalties

        self.state_penalties = penalties

        # Model matrix parameters that remain constant are specified.
        # And then we need to build the model matrix once for the entire
        # data so that we can later get observation probabilities for the
        # entire data.
        terms = self.formula.get_terms()
        has_intercept = self.formula.has_intercept()
        ltx = self.formula.get_linear_term_idx()
        irstx = []
        stx = self.formula.get_smooth_term_idx()
        rtx = self.formula.get_random_term_idx()
        var_types = self.formula.get_var_types()
        var_map = self.formula.get_var_map()
        var_mins = self.formula.get_var_mins()
        var_maxs = self.formula.get_var_maxs()
        factor_levels = self.formula.get_factor_levels()
        NOT_NA_flat = self.formula.NOT_NA_flat
        cov_flat = self.formula.cov_flat
        y_flat = self.formula.y_flat
        cov = self.formula.cov
        n_series = len(self.formula.y)
        n_obs = len(y_flat)

        model_mat_full = build_sparse_matrix_from_formula(terms,has_intercept,has_scale_split,
                                                          ltx,irstx,stx,rtx,var_types,var_map,
                                                          var_mins,var_maxs,factor_levels,
                                                          cov_flat[NOT_NA_flat],None,self.n_j,
                                                          None,None)

        # Now we need to iteratively improve the model estimates (GAMM parameters and
        # sojourn distribution parameters). So we start by maximizing the latter given
        # the current state & state dur estimates and then obtain the next set of state
        # & state dur parameters given these new model parameters.
        state_coef = [scp.stats.norm.rvs(size=model_mat_full.shape[1]) for j in range(self.n_j)]
        state_scales = [init_scale for j in range(self.n_j)]
        state_penalties = []
        state_LVIs = []
        llk_hist = []

        n_pi = self.__pi
        n_TR = self.__TR
        
        # For state proposals we utilize a temparature schedule. This is similar to the idea of simulated annealing proposed
        # by Kirkpatrick, Gelatt and Vecchi (1983). However, the mechanism here is closer to simulated
        # tempering than annealing (Marinari & Parisi 1992; Allassonnière & Chevallier, 2021). Specifically,
        # at every iteration iter we sample noise from a normal distribution with sd=temp_schedule[iter] and add
        # that to the smoothed probabilities used to propose new steps. The idea is that as long as
        # sd=temp_schedule[iter] > 0 we promote exploring new state sequence candidates, so that we (hopefully)
        # further reduce the chance of ending up with a local maximum. Of course, if we set sd=temp_schedule[iter]
        # too extreme, we will not get anywhere since the noise dominates the smoothed probabilities. So this
        # likely requires some tuning.
        # see: Marinari, E., & Parisi, G. (1992). Simulated Tempering: A New Monte Carlo Scheme. https://doi.org/10.1209/0295-5075/19/6/002
        temp_schedule = anneal_temps_zero(burn_in,t0,r)

        iterator = range(burn_in + m_avg)
        if progress_bar:
            iterator = tqdm(iterator,desc="Fitting",leave=True)
        
        for iter in iterator:

            ### Stochastic Expectation ###

            # Propose new states based on all updated parameters.
            # First we need the probability of stage durations for every stage (i.e., under every sojourn dist.).
            s_log_o_probs,dur_log_probs = compute_log_probs(self.n_j,n_obs,has_scale_split,
                                                            model_mat_full,state_coef,
                                                            state_scales,self.pds,y_flat,
                                                            NOT_NA_flat,self.formula.sid,
                                                            self.family,factor_levels,
                                                            self.mvar_by)
            
            # Now we can propose a new set of states and state_durs for every series.
            with mp.Pool(processes=self.cpus) as pool:
                # After the burn-in period, we do not add noise to the state probabilities, since
                # we no longer want to explore unnecessarily.
                temp = 0
                if iter < burn_in:
                    temp = temp_schedule[iter]

                durs,states,llks = self.__propose_all_states(pool,cov,temp,n_pi,n_TR,s_log_o_probs,dur_log_probs,var_map)

                if not self.mvar_by is None:
                    states_flat = np.array([st for s in states for _ in range(len(factor_levels[self.mvar_by])) for st in s],dtype=int)
                else:
                    states_flat = np.array([st for s in states for st in s],dtype=int)
            
            ### Convergence control ###

            # Convergence control is best based on plotting the change in penalized likelihood over iterations.
            # Once we start oscillating around a constant value the chain has likely converged ~ see Nielsen (2002).
            if iter > 0:
                pen_llk = np.sum(llks) - np.sum(state_penalties)
                llk_hist.append(pen_llk)

            ### Maximization ###

            # First update all GAMM parameters - if scale_split==True, do so for every state separately
            if has_scale_split:
                for j in range(self.n_j):
                    
                    state_NOT_NA = NOT_NA_flat[states_flat == j]
                    state_y = y_flat[states_flat == j][state_NOT_NA]
                    state_cov = cov_flat[states_flat == j,:][state_NOT_NA]
                    
                    # Build the model matrix with all information from the formula for state j
                    model_mat = build_sparse_matrix_from_formula(terms,has_intercept,has_scale_split,
                                                                ltx,irstx,stx,rtx,var_types,var_map,
                                                                var_mins,var_maxs,factor_levels,
                                                                state_cov,None,self.n_j,None,None)
                    
                    # Get initial estimate of mu based on family:
                    if isinstance(self.family,Gaussian): #or iter == 0
                        #init_mu_flat = self.family.init_mu(state_y)
                        init_mu_flat = self.family.init_mu(model_mat @ state_coef[j])
                    else: # Use last coefficient set for mu estimate. Penalties carry over as well.
                        init_mu_flat = self.family.link.fi(model_mat @ state_coef[j])

                    # Now we have to estimate the model for the current state j
                    coef,eta,wres,scale,LVI,edf,term_edf,penalty = solve_gamm_sparse(init_mu_flat,state_y,
                                                                                    model_mat,penalties[j],self.formula.n_coef,
                                                                                    self.family,maxiter_inner,"svd",
                                                                                    conv_tol,extend_lambda,control_lambda,
                                                                                    exclude_lambda,False,self.cpus)
                    
                    
                    
                    # Collect coefficients, penalties, Invs and scale parameter 
                    if iter == 0:
                        state_penalties.append(penalty)
                        state_LVIs.append(LVI)
                    else:
                        state_penalties[j] = penalty
                        state_LVIs[j] = LVI

                    state_coef[j] = coef
                    state_scales[j] = scale
            else:
                raise NotImplementedError("has_scale_split==False is not yet implemented.")

            # Next update all sojourn time distribution parameters as well as TR and pi.
            if self.estimate_pi:
                n_pi = np.zeros(self.n_j)
            
            if self.estimate_TR:
                n_TR = np.zeros((self.n_j,self.n_j))

            # Iterate over every series to get the durations from every state in that series
            j_dur = [[] for j in range(self.n_j)]
            j_cov = [[] for j in range(self.n_j)]
            for s in range(n_series):
                if self.estimate_pi:
                    # For given state sequences optimal estimate for pi
                    # is the percentage of being in state j at time 0 over
                    # all series. Follows direclty from sub-equation 1 in Yu (2011)
                    # for complete data case.
                    n_pi[durs[s][0,0]] += 1
                
                if self.estimate_TR:
                    # For given state sequences, optimal estimate
                    # for the transition probability from i -> j
                    # is: n(i -> j)/n(i -> any not i)
                    # where n() gives the number of occurences across
                    # all sequences. Follows direclty from sub-equation 2 in Yu (2011)
                    # for complete data case.
                    for tr in range(1,durs[s].shape[0]):
                        n_TR[durs[s][tr-1,0],durs[s][tr,0]] += 1
                
                # Durations for every state put in a list
                sd = durs[s]
                s_durs = [sd[sd[:,0] == j,1] for j in range(self.n_j)]
                for j in range(self.n_j):
                    j_dur[j].extend(s_durs[j])
                    pd_split = self.pds[j].split_by
                    if pd_split is not None:
                        c_s = cov[s] # Encoded variables from this series
                        c_s_j = c_s[0,var_map[pd_split]] # Factor variables are assumed constant so we can take first element.
                        j_cov[j].extend([c_s_j for _ in s_durs[j]])

            # Maximize sojourn time distribution parameters by obtaining MLE
            # for the proposed stage durations.
            for j in range(self.n_j):
                j_dur[j] = np.array(j_dur[j])
                pd_j = self.pds[j]

                if pd_j.split_by is not None:
                    j_cov[j] = np.array(j_cov[j])
                    for ci in range(pd_j.n_by):
                        pd_j.fit(j_dur[j][j_cov[j] == ci],ci)
                else:
                    pd_j.fit(j_dur[j])

            # Counts -> Probs for pi and TR
            if self.estimate_pi:
                n_pi /= n_series
            
            if self.estimate_TR:
                for j in range(self.n_j):
                    n_TR[j,:] /= np.sum(n_TR[j,:])
            
            # We average over the last m parameter sets after convergence to
            # get the final estimate (Nielsen, 2002).
            if iter == burn_in:
                self.__TR = n_TR
                self.__pi = n_pi
                self.__scale = [state_scales[j] for j in range(self.n_j)]
                self.__coef = [state_coef[j] for j in range(self.n_j)]
                pd_params = [np.array(pd_j.params) for pd_j in self.pds]
                self.lvi = [state_LVIs[j] for j in range(self.n_j)]

            elif iter > burn_in:
                self.__TR += n_TR
                self.__pi += n_pi
                for j in range(self.n_j):
                    self.__scale[j] += state_scales[j]
                    self.__coef[j] += state_coef[j]
                    self.lvi[j] += state_LVIs[j]
                    pd_params[j] += self.pds[j].params

        # Now we have to finalize the average over the last m parameters and to decode
        # the state sequence given those parameters.
        self.__TR /= m_avg
        self.__pi /= m_avg

        # We have to normalize since the rows will not necessarily sum to one after averaging
        self.__TR /= np.sum(self.__TR,axis=1)
        self.__pi /= np.sum(self.__pi)

        for j in range(self.n_j):
            self.__scale[j] /= m_avg
            self.__coef[j] /= m_avg
            self.lvi[j] /= m_avg
            pd_params[j] /= m_avg
            # Overwrite duration dist. parameters with average ones
            self.pds[j].params = pd_params[j]

        # We need to compute the log observation and duration probs one last time for decoding.
        # This time we use the max parameters we obtained for calculation stored in self.
        s_log_o_probs,dur_log_probs = compute_log_probs(self.n_j,n_obs,has_scale_split,
                                                            model_mat_full,self.__coef,
                                                            self.__scale,self.pds,y_flat,
                                                            NOT_NA_flat,self.formula.sid,
                                                            self.family,factor_levels,
                                                            self.mvar_by)
        
        # Now we can decode.
        with mp.Pool(processes=self.cpus) as pool:
            _,states_max,_ = self.__decode_all_states(pool,cov,self.__pi,self.__TR,s_log_o_probs,dur_log_probs,var_map)

            if not self.mvar_by is None:
                max_states_flat = np.array([st for s in states_max for _ in range(len(factor_levels[self.mvar_by])) for st in s],dtype=int)
            else:
                max_states_flat = np.array([st for s in states_max for st in s],dtype=int)
        
        return llk_hist,max_states_flat
            
##################################### Prediction #####################################

    def predict(self, j, use_terms, n_dat,alpha=0.05,ci=False):
        # Basically GAMM.predict() but with an aditional j argument, based on
        # which the coefficients and scale parameters are selected.
        var_map = self.formula.get_var_map()
        var_keys = var_map.keys()

        for k in var_keys:
            if k not in n_dat.columns:
                raise IndexError(f"Variable {k} is missing in new data.")
        
        # Encode test data
        _,pred_cov_flat,_,_,pred_cov,_,_ = self.formula.encode_data(n_dat,prediction=True)

        # Then, we need to build the model matrix - but only for the terms which should
        # be included in the prediction!
        terms = self.formula.get_terms()
        has_intercept = self.formula.has_intercept()
        has_scale_split = False
        ltx = self.formula.get_linear_term_idx()
        irstx = []
        stx = self.formula.get_smooth_term_idx()
        rtx = self.formula.get_random_term_idx()
        var_types = self.formula.get_var_types()
        var_mins = self.formula.get_var_mins()
        var_maxs = self.formula.get_var_maxs()
        factor_levels = self.formula.get_factor_levels()
        n_j = None
        state_est_flat = None
        state_est = None

        # So we pass the desired terms to the use_only argument
        predi_mat = build_sparse_matrix_from_formula(terms,has_intercept,has_scale_split,
                                                     ltx,irstx,stx,rtx,var_types,var_map,
                                                     var_mins,var_maxs,factor_levels,
                                                     pred_cov_flat,pred_cov,n_j,state_est_flat,
                                                     state_est,use_only=use_terms)
        
        # Now we calculate the prediction
        pred = predi_mat @ self.__coef[j]

        # Optionally calculate the boundary for a 1-alpha CI
        if ci:
            # Wood (2017) 6.10
            c = predi_mat @ self.lvi[j].T @ self.lvi[j] * self.__scale[j] @ predi_mat.T
            c = c.diagonal()
            b = scp.stats.norm.ppf(1-(alpha/2)) * np.sqrt(c)
            return pred,predi_mat,b

        return pred,predi_mat,None


class sMsIRGAMM(sMsGAMM):

    def __init__(self,
                 formula: Formula,
                 family: Family,
                 end_points: list,
                 fix:None or list[list[int,int]] = None,
                 pre_llk_fun=pre_ll_sms_IR_gamm,
                 cpus: int = 1):

        super().__init__(formula,
                         family,
                         end_points,
                         pre_llk_fun,
                         False,
                         False,
                         np.zeros(formula.get_nj()),
                         np.zeros((formula.get_nj(),
                                   formula.get_nj())),
                         None,
                         cpus)
        
        # Define which events should be fixed and at which sample.
        if not fix is None:
            self.fix = [event[0] for event in fix]
            self.fix_at = [event[1] for event in fix]
        else:
            self.fix = None
            self.fix_at = None

        self.__coef = None
        self.__scale = None
        self.penalty = 0

    ##################################### Getters #####################################

    def get_pars(self):
        return self.__coef,self.__scale
    
    ##################################### Fitting #####################################

    def __init_all_states(self,pool,end_points):

        # MP code to propose initial state for every series
        args = zip(end_points,repeat(self.n_j),repeat(self.pre_llk_fun),
                   repeat(self.fix),repeat(self.fix_at))
        
        state_durs_new, states_new = zip(*pool.starmap(init_states_IR,args))
        return list(state_durs_new),list(states_new)
    
    def __propose_all_states(self,pool,temp,y,NOT_NAs,end_points,cov,state_durs,states,coef,scale,
                        log_o_probs,var_map,terms,has_intercept,ltx,irstx,
                        stx,rtx,var_types,var_mins,var_maxs,factor_levels,
                        prop_sd,n_prop,use_only):
        
        # MP code to propose states for every series
        args = zip(repeat(self.n_j),repeat(temp),y,NOT_NAs,end_points,cov,state_durs,states,
                   repeat(coef),repeat(scale),log_o_probs,repeat(self.pds),
                   repeat(self.pre_llk_fun),repeat(var_map),repeat(self.family),repeat(terms),
                   repeat(has_intercept),repeat(ltx),repeat(irstx),repeat(stx),
                   repeat(rtx),repeat(var_types),repeat(var_mins),repeat(var_maxs),
                   repeat(factor_levels),repeat(self.fix),repeat(self.fix_at),repeat(prop_sd),
                   repeat(n_prop),repeat(use_only))
        
        state_durs_new, states_new, llks = zip(*pool.starmap(se_step_sms_dc_gamm,args))
        return list(state_durs_new),list(states_new), list(llks)
    
    def fit(self,maxiter_outer=100,maxiter_inner=30,conv_tol=1e-6,extend_lambda=True,control_lambda=True,exclude_lambda=True,t0=1,r=0.925,schedule="anneal",n_prop=None,prop_sd=2,progress_bar=True,mmat_MP=True):
        # Performs something like Stochastic Expectation maiximization (e.g., Nielsen, 2002) see the sem.py file for
        # more details.
        
        # Penalties need to be initialized
        self.formula.build_penalties()
        
        penalties = self.formula.penalties

        # Propose an initial set of states and state_durs for every series.
        with mp.Pool(processes=self.cpus) as pool:
            durs,states = self.__init_all_states(pool,self.end_points)

        # Model matrix parameters that remain constant are specified.
        # And then we need to build the model matrix for the start estimates to get start coefficients.
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
        NOT_NA_flat = self.formula.NOT_NA_flat
        cov_flat = self.formula.cov_flat
        y_flat = self.formula.y_flat
        n_series = len(self.formula.y)
        n_obs = len(y_flat)

        # We use a heuristic to determine the number of samples that should be drawn when proposing new states.
        if n_prop is None:
            n_prop = int(n_series*0.05)

        # For the IR GAMM we need the cov object split by series id
        # Importantly, we must not exclude any rows for with the dependent variable is
        # NA at this point, to make sure that the convolution is calculated accurately.
        cov = self.formula.cov
        
        if mmat_MP:
            with mp.Pool(processes=self.cpus) as pool:
                model_mat_full = build_sparse_matrix_from_formula(terms,has_intercept,False,
                                                                ltx,irstx,stx,rtx,var_types,var_map,
                                                                var_mins,var_maxs,factor_levels,
                                                                cov_flat,cov,None,
                                                                None,states,pool=pool)
        else:
            model_mat_full = build_sparse_matrix_from_formula(terms,has_intercept,False,
                                                                ltx,irstx,stx,rtx,var_types,var_map,
                                                                var_mins,var_maxs,factor_levels,
                                                                cov_flat,cov,None,
                                                                None,states,pool=None)
        
        # Only now can we remove the NAs
        model_mat_full = model_mat_full[NOT_NA_flat,]
        
        # And estimate the model

        # Get initial estimate of mu based on family:
        init_mu_flat = self.family.init_mu(y_flat[NOT_NA_flat])

        coef,eta,wres,scale,LVI,edf,term_edf,penalty = solve_gamm_sparse(init_mu_flat,y_flat[NOT_NA_flat],
                                                                        model_mat_full,penalties,self.formula.n_coef,
                                                                        self.family,maxiter_inner,"svd",
                                                                        conv_tol,extend_lambda,control_lambda,
                                                                        exclude_lambda,False,self.cpus)

        # For state proposals we can utilize a temparature schedule. See sMsGamm.fit().
        if schedule == "anneal":
            temp_schedule = anneal_temps_zero(maxiter_outer,t0,r)
        else:
            temp_schedule = const_temps(maxiter_outer)

        last_llk = None
        llk_hist = []

        iterator = range(maxiter_outer)
        if progress_bar:
            iterator = tqdm(iterator,desc="Fitting",leave=True)

        for iter in iterator:
            ### Stochastic Expectation ###

            # Propose new states based on all updated parameters.

            # For IR GAMM we only need the probability of observing every
            # series under the model.
            log_o_probs = np.zeros(n_obs)

            
            # Handle observation probabilities
            mu = (model_mat_full @ coef).reshape(-1,1)

            if not isinstance(self.family,Gaussian):
                mu = self.family.link.fi(mu)

            if not self.family.twopar:
                log_o_probs[NOT_NA_flat] = np.ndarray.flatten(self.family.lp(y_flat[NOT_NA_flat],mu))
            else:
                log_o_probs[NOT_NA_flat] = np.ndarray.flatten(self.family.lp(y_flat[NOT_NA_flat],mu,scale))
            log_o_probs[NOT_NA_flat == False] = np.nan

            # We need to split the observation probabilities by series
            s_log_o_probs = np.split(log_o_probs,self.formula.sid[1:],axis=0)
            
            # Now we can propose a new set of states and state_durs for every series.
            with mp.Pool(processes=self.cpus) as pool:
                durs,states,llks = self.__propose_all_states(pool,temp_schedule[iter],self.formula.y,
                                                             self.formula.NOT_NA,self.end_points,
                                                             cov,durs,states,coef,scale,s_log_o_probs,
                                                             var_map,terms,has_intercept,ltx,irstx,
                                                             stx,rtx,var_types,var_mins,var_maxs,
                                                             factor_levels,prop_sd,n_prop,None)
            
            ### Convergence control ###

            # Convergence control is based on the change in penalized complete data likelihood
            if iter > 0:
                pen_llk = np.sum(llks) - penalty

                if iter > 1:
                    # Also check convergence
                    llk_diff = abs(pen_llk - last_llk)

                    if progress_bar:
                        iterator.set_description_str(desc="Fitting - Conv.: " + "{:.2e}".format(llk_diff - conv_tol*abs(pen_llk)), refresh=True)

                    if llk_diff < conv_tol*abs(pen_llk):
                        if progress_bar:
                            iterator.set_description_str(desc="Converged!", refresh=True)
                            iterator.close()
                        break

                last_llk = pen_llk
                llk_hist.append(pen_llk)

            ### Maximization ###

            # First update all GAMM parameters
            if mmat_MP:
                with mp.Pool(processes=self.cpus) as pool:
                    model_mat_full = build_sparse_matrix_from_formula(terms,has_intercept,False,
                                                                    ltx,irstx,stx,rtx,var_types,var_map,
                                                                    var_mins,var_maxs,factor_levels,
                                                                    cov_flat,cov,None,
                                                                    None,states,pool=pool)
            else:
                model_mat_full = build_sparse_matrix_from_formula(terms,has_intercept,False,
                                                                    ltx,irstx,stx,rtx,var_types,var_map,
                                                                    var_mins,var_maxs,factor_levels,
                                                                    cov_flat,cov,None,
                                                                    None,states,pool=None)
                
            model_mat_full = model_mat_full[NOT_NA_flat,]

            # Use last coefficient set for mu estimate. Penalties carry over as well.
            if isinstance(self.family,Gaussian):
                init_mu_flat = model_mat_full @ coef
            else: 
                init_mu_flat = self.family.link.fi(model_mat_full @ coef)

            # Now fit the model again.
            coef,eta,wres,scale,LVI,edf,term_edf,penalty = solve_gamm_sparse(init_mu_flat,y_flat[NOT_NA_flat],
                                                                             model_mat_full,penalties,self.formula.n_coef,
                                                                             self.family,maxiter_inner,"svd",
                                                                             conv_tol,extend_lambda,control_lambda,
                                                                             exclude_lambda,False,self.cpus)

            # Next update all sojourn time distribution parameters

            # Iterate over every series to get the durations from every state in that series
            j_dur = [[] for j in range(self.n_j)]
            j_cov = [[] for j in range(self.n_j)]
            for s in range(n_series):

                # Durations for every state put in a list
                sd = durs[s]
                s_durs = [sd[sd[:,0] == j,1] for j in range(self.n_j)]
                for j in range(self.n_j):
                    j_dur[j].extend(s_durs[j])
                    pd_split = self.pds[j].split_by
                    if pd_split is not None:
                        c_s = cov[s] # Encoded variables from this series
                        c_s_j = c_s[0,var_map[pd_split]] # Factor variables are assumed constant so we can take first element.
                        j_cov[j].extend([c_s_j for _ in s_durs[j]])

            # Maximize sojourn time distribution parameters by obtaining MLE
            # for the proposed stage durations.
            for j in range(self.n_j):

                # We do not need to maximize for states that have a fixed event.
                if not self.fix is None and j in self.fix:
                    continue
                j_dur[j] = np.array(j_dur[j])
                pd_j = self.pds[j]

                if pd_j.split_by is not None:
                    j_cov[j] = np.array(j_cov[j])
                    for ci in range(pd_j.n_by):
                        pd_j.fit(j_dur[j][j_cov[j] == ci],ci)
                else:
                    pd_j.fit(j_dur[j])
        
        # Collect final state sequence in the same format returned by sMsGamm
        # and in trial-level format needed for prediction.
        states_flat = []
        for s in range(n_series):
            sd = durs[s]
            s_states_flat = []
            for st in range(sd.shape[0]):
                if sd[st,1] != 0:
                    for d in range(sd[st,1]):
                        s_states_flat.append(st)
            states_flat.append(s_states_flat)

        # Save final coefficients
        self.__scale = scale
        self.__coef = coef
        self.lvi = LVI
        self.penalty = penalty
        self.pred = eta
        self.res = wres
        self.edf = edf
        self.term_edf = term_edf

        return llk_hist,states_flat,states

    def predict(self, states, use_terms, n_dat,alpha=0.05,ci=False):
        # Basically GAMM.predict() but with states.
        var_map = self.formula.get_var_map()
        var_keys = var_map.keys()

        for k in var_keys:
            if k not in n_dat.columns:
                raise IndexError(f"Variable {k} is missing in new data.")
        
        # Encode test data
        _,pred_cov_flat,_,_,pred_cov,_,_ = self.formula.encode_data(n_dat,prediction=True)

        # Then, we need to build the model matrix - but only for the terms which should
        # be included in the prediction!
        terms = self.formula.get_terms()
        has_intercept = self.formula.has_intercept()
        has_scale_split = False
        ltx = self.formula.get_linear_term_idx()
        irstx = self.formula.get_ir_smooth_term_idx()
        stx = self.formula.get_smooth_term_idx()
        rtx = self.formula.get_random_term_idx()
        var_types = self.formula.get_var_types()
        var_mins = self.formula.get_var_mins()
        var_maxs = self.formula.get_var_maxs()
        factor_levels = self.formula.get_factor_levels()
        n_j = None
        state_est_flat = None

        # So we pass the desired terms to the use_only argument
        predi_mat = build_sparse_matrix_from_formula(terms,has_intercept,has_scale_split,
                                                     ltx,irstx,stx,rtx,var_types,var_map,
                                                     var_mins,var_maxs,factor_levels,
                                                     pred_cov_flat,pred_cov,n_j,state_est_flat,
                                                     [states],use_only=use_terms)
        
        # Now we calculate the prediction
        pred = predi_mat @ self.__coef

        # Optionally calculate the boundary for a 1-alpha CI
        if ci:
            # Wood (2017) 6.10
            c = predi_mat @ self.lvi.T @ self.lvi * self.__scale @ predi_mat.T
            c = c.diagonal()
            b = scp.stats.norm.ppf(1-(alpha/2)) * np.sqrt(c)
            return pred,predi_mat,b

        return pred,predi_mat,None
    
    def predict_llk(self, use_terms, n_dat, n_endpoints,conv_tol=1e-6,t0=1,r=0.925,n_prop=500,prop_sd=2):
        # Estimates best state sequence given model for new data and returns the CDL under this
        # state sequence, the model, and given the new data.
        var_map = self.formula.get_var_map()
        var_keys = var_map.keys()

        for k in var_keys:
            if k not in n_dat.columns:
                raise IndexError(f"Variable {k} is missing in new data.")
        
        # Encode test data
        pred_y_flat,pred_cov_flat,pred_NOT_NAs_flat,pred_y,pred_cov,pred_NOT_NAs,pred_sid = self.formula.encode_data(n_dat,prediction=False)

        # Propose an initial set of states and state_durs for every new series.
        with mp.Pool(processes=self.cpus) as pool:
            durs,states = self.__init_all_states(pool,n_endpoints)


        # Then, we need to build the model matrix - but only for the terms which should
        # be included in the prediction!
        terms = self.formula.get_terms()
        has_intercept = self.formula.has_intercept()
        has_scale_split = False
        ltx = self.formula.get_linear_term_idx()
        irstx = self.formula.get_ir_smooth_term_idx()
        stx = self.formula.get_smooth_term_idx()
        rtx = self.formula.get_random_term_idx()
        var_types = self.formula.get_var_types()
        var_mins = self.formula.get_var_mins()
        var_maxs = self.formula.get_var_maxs()
        factor_levels = self.formula.get_factor_levels()
        n_j = None
        state_est_flat = None
        n_obs = len(pred_y_flat)

        # Setup temp schedule.
        temp_schedule = anneal_temps_zero(n_prop,t0,r)
        
        last_llk = None
        llk_hist = []
        for iter in range(n_prop):

            # We pass the desired terms to the use_only argument
            predi_mat = build_sparse_matrix_from_formula(terms,has_intercept,has_scale_split,
                                                        ltx,irstx,stx,rtx,var_types,var_map,
                                                        var_mins,var_maxs,factor_levels,
                                                        pred_cov_flat,pred_cov,n_j,state_est_flat,
                                                        states,use_only=use_terms)
            
            # Only now can we remove the NAs
            predi_mat = predi_mat[pred_NOT_NAs_flat,]

            # Propose new states, basically copied from IR GAMM.fit()

            # Handle observation probabilities
            log_o_probs = np.zeros(n_obs)
            
            mu = (predi_mat @ self.__coef).reshape(-1,1)

            if not isinstance(self.family,Gaussian):
                mu = self.family.link.fi(mu)

            if not self.family.twopar:
                log_o_probs[pred_NOT_NAs_flat] = np.ndarray.flatten(self.family.lp(pred_y_flat[pred_NOT_NAs_flat],mu))
            else:
                log_o_probs[pred_NOT_NAs_flat] = np.ndarray.flatten(self.family.lp(pred_y_flat[pred_NOT_NAs_flat],mu,self.__scale))
            log_o_probs[pred_NOT_NAs_flat == False] = np.nan

            # We need to split the observation probabilities by the predicted series
            s_log_o_probs = np.split(log_o_probs,pred_sid[1:],axis=0)
            
            # Now we can propose a new set of states and state_durs for every series.
            # Basically - we propose only one new candidate, which is either accepted or not.
            # In the long run we should find the best state sequence for the new data given the
            # models parameters. We return the CDL of these sequences.
            with mp.Pool(processes=self.cpus) as pool:
                durs,states,llks = self.__propose_all_states(pool,temp_schedule[iter],pred_y,
                                                                pred_NOT_NAs,n_endpoints,
                                                                pred_cov,durs,states,self.__coef,self.__scale,s_log_o_probs,
                                                                var_map,terms,has_intercept,ltx,irstx,
                                                                stx,rtx,var_types,var_mins,var_maxs,
                                                                factor_levels,prop_sd,1,use_terms)
            
            # Held-out LLK
            LO_llk = sum(llks)
            if iter > 0:
                # Also check convergence
                if abs(LO_llk - last_llk) < conv_tol*abs(LO_llk):
                    print("Converged",iter)
                    break

            last_llk = LO_llk
            llk_hist.append(LO_llk)

        return LO_llk

