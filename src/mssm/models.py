import numpy as np
import scipy as scp
from tqdm import tqdm
import warnings
from . import utils
import warnings
import multiprocessing as mp
from itertools import repeat
from matplotlib import pyplot as plt
import re
from copy import deepcopy
from collections.abc import Callable
from .src.python.formula import *
from .src.python.terms import *
from .src.python.penalties import *
from .src.python.exp_fam import *
from .src.python.gamm_solvers import solve_gamm_sparse

##################################### Base Class #####################################

class MSSM:

    def __init__(self,
                 formula:Formula,
                 family:Family,
                 p_formula=None,
                 pre_llk_fun:Callable = None,
                 estimate_pi:bool=False,
                 estimate_TR:bool=False,
                 cpus:int=1):
        
        # Formulas associated with model
        self.formula = formula # For coefficients
        self.p_formula = p_formula # For sojourn time distributions

        # Family of model
        self.family = family
        
        ## "prior" Log-likelihood functions
        self.pre_llk_fun = pre_llk_fun

        ## Should transition matrices and initial distribution be estimated?
        self.estimate_pi = estimate_pi
        self.estimate_TR = estimate_TR

        self.cpus=cpus

        # Temperature schedule
        self.__temp = None

        # History containers
        self.__coef_hist = None
        self.__state_dur_hist = None
        self.__state_hist = None
        self.__phi_hist = None
        self.__scale_hist = None
        self.__TR_hist = None
        self.__pi_hist = None

        # Current estimates
        self.__coef = None
        self.__state_dur = None
        self.__state = None
        self.__phi = None
        self.__scale = None
        self.__TR = None
        self.__pi = None

        # Function storage
        self.__sample_fun = None
        self.__sample_fun_kwargs = None
        self.m_pi = None
        self.m_TR = None
        self.m_ps = None
        self.par_ps = None
        self.e_bs = None
        self.__e_bs_kwargs = None

    ##################################### Setters #####################################

    def set_sample_fun(self,fun,**kwargs):
        self.__sample_fun = fun
        self.__sample_fun_kwargs = kwargs
    
    def set_par_ps(self,fun):
        self.par_ps = fun
    
    def set_e_bs(self,fun, **kwargs):
        self.e_bs = fun
        self.__e_bs_kwargs = kwargs

    def set_m_pi(self,fun):
        self.m_pi = fun

    def set_m_TR(self,fun):
        self.m_TR = fun

    def set_m_ps(self,fun):
        self.m_ps = fun

    def set_temp(self,fun,iter=500,**kwargs):
        self.__temp = fun(iter,**kwargs)
    
    ##################################### Getters #####################################
    
    def get_pars(self):
        pass
    
    def get_llk(self):
        pass
    
    def get_coef_hist(self):
        return self.__coef_hist
    
    def get_state_hist(self):
        return self.__state_hist
    
    def get_phi_hist(self):
        return self.__phi_hist
    
    ##################################### Fitting #####################################

    def fit(self):
        pass

    ##################################### Prediction #####################################

    def predict(self,terms,n_dat):
        pass

    ##################################### Plotting #####################################

    def plot(self):
        pass

    def val_plot(self):
        pass

##################################### GAMM class #####################################

class GAMM(MSSM):

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
        return self.__coef,self.__scale
    
    def get_llk(self,penalized=True):
        # Get (Penalized) log-likelihood of estimated model.
        pen = 0
        if penalized:
            pen = self.penalty
        if self.pred is not None:
            mu = self.pred
            if isinstance(self.family,Gaussian) == False:
                mu = self.family.link.fi(self.pred)
            if self.family.twopar:
                return self.family.llk(self.formula.y_flat,mu,self.__scale) - pen
            else:
                return self.family.llk(self.formula.y_flat,mu) - pen
        return None
    
    ##################################### Fitting #####################################
    
    def fit(self,maxiter=30,conv_tol=1e-7,extend_lambda=True,restart=False):

        # We need to initialize penalties
        if not restart:
            self.formula.build_penalties()
        penalties = self.formula.penalties

        if penalties is None and restart:
            raise ValueError("Penalties were not initialized. Restart must be set to False.")

        # And then we need to build the model matrix once
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
        y_flat = self.formula.y_flat[self.formula.NOT_NA_flat]

        if y_flat.shape[0] != self.formula.y_flat.shape[0]:
            print("NAs were excluded for fitting.")

        cov = None
        n_j = None
        state_est_flat = None
        state_est = None

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
                                                                         conv_tol,extend_lambda)
        
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
        # Based on itsadug get_difference function
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

    def __init__(self, n_j, series, time,
                 end_points, llk_fun, pre_llk_fun,
                 covariates=None, is_DC=True,
                 sep_per_j=utils.j_split,
                 split_p_by_cov = None,
                 estimate_pi=True,
                 estimate_TR=True, cpus=1):
        super().__init__(n_j, series, time,
                         end_points, llk_fun, pre_llk_fun,
                         covariates, is_DC, sep_per_j,
                         split_p_by_cov,estimate_pi,
                         estimate_TR, cpus)

        if sep_per_j is not None:
            self.set_e_bs(utils.get_log_o_prob_mat)
            self.set_sample_fun(utils.se_step_sms_gamm)
        else:
            warnings.warn("Don't forget to call set_e_bs() and set_sample_fun().")

        self.set_m_pi(utils.m_pi_sms_gamm)
        self.set_m_TR(utils.m_TR_sms_gamm)
        self.set_m_ps(utils.m_gamma2s_sms_gamm)
        self.set_par_ps(utils.par_gamma2s)

class sMsDCGAMM(MSSM):

    def __init__(self, n_j, series, time,
                 end_points, llk_fun, pre_llk_fun,
                 covariates=None, is_DC=True,
                 sep_per_j=None, split_p_by_cov = None,
                 estimate_pi=False,
                 estimate_TR=False, cpus=1):
        super().__init__(n_j, series, time,
                         end_points, llk_fun, pre_llk_fun,
                         covariates, is_DC, sep_per_j,
                         split_p_by_cov,estimate_pi,
                         estimate_TR, cpus)

        self.set_e_bs(utils.get_log_o_prob_mat)
        self.set_m_ps(utils.m_gamma2s_sms_dc_gamm)
        self.set_par_ps(utils.par_gamma2s)
        self.set_sample_fun(utils.se_step_sms_dc_gamm)
    
    ##################################### MP handlers #####################################
    
    def __propose_all_states(self,pool,temp,pi,TR,state_durs_est,state_est,ps,coef,scale):
        args = zip(repeat(self.n_j),repeat(temp),self.series,self.end_points,self.time,
                   self.covariates,repeat(pi),repeat(TR),state_durs_est,state_est,repeat(ps),
                   repeat(coef),repeat(scale),repeat(self.__build_mat_fun), repeat(self.pre_llk_fun),
                   repeat(self.llk_fun),repeat(self.e_bs),repeat(self.sep_per_j is not None),
                   repeat(self.split_p_by_cov),repeat(self.__build_mat_kwargs),
                   repeat(self.__e_bs_kwargs))
        
        mapping = zip(repeat(self.__sample_fun),args,repeat(self.__sample_fun_kwargs))

        state_durs_new, states_new = zip(*pool.starmap(utils.map_unwrap_args_kwargs,mapping))
        return list(state_durs_new),list(states_new)
    
    def __calc_llk_all_events(self,pool,pi,TR,state_dur_est,state_est,ps,logprobs):
        args = zip(repeat(self.n_j),repeat(pi),repeat(TR),state_dur_est,state_est,repeat(ps),logprobs,self.covariates,repeat(self.split_p_by_cov))
        llks = pool.starmap(self.llk_fun,args)
        return llks
    
    ##################################### SEM - Estimation #####################################

    def __advance_chain(self,chain,pool,temp,c_pi,c_TR,c_p_pars,c_coef,c_scale,c_state_durs_est,c_state_est):
        # Performs One Stochastic Expectation maiximization iteration (or something quite like it if is_DC=True).
        # See utils.py for details but also Nielsen (2002).

        # Propose new candidates

        ## Parameterize duration distributions based on current/previous parameters
        ps = self.par_ps(c_p_pars)
        
        ## Sample next latent state estimate (Stochastic E-step)
        n_state_dur_est, n_state_est = self.__propose_all_states(pool,temp,c_pi,c_TR,
                                                                 c_state_durs_est,
                                                                 c_state_est,ps,
                                                                 c_coef,c_scale)

        # Calculate M-steps based on new latent state estimate

        ## First calculate design matrix for every series and combine.
        n_mat = self.__build_all_mod_mat(pool,n_state_est)

        ## Now actually update coefficients based on proposal
        if self.sep_per_j is not None and not self.is_DC:
            ### If we have truely separate models per latent state, we update the coefs and scales separately
            n_logprobs = np.zeros(len(self.series_fl))
            
            n_state_est_fl = np.array([st for s in n_state_est for st in s],dtype=int)
            y_split, x_split = self.sep_per_j(self.n_j,n_state_est_fl,self.series_fl, n_mat)
            
            n_coef, self.__penalties[chain][0], n_scale, j_embS = utils.solve_am(x_split[0],
                                                                                 y_split[0],
                                                                                 self.__penalties[chain][0],
                                                                                 0,maxiter=10)
            n_scale = [n_scale]
            

            ### Get probabilities of observing series under new model for this state
            n_logprobs[n_state_est_fl == 0] = self.e_bs(self.n_j,y_split[0], x_split[0], n_coef,
                                                        n_scale[0], False, **self.__e_bs_kwargs)

            tot_penalty = n_coef.T @ j_embS @ n_coef
            
            #### Repeat for remaining states
            for j in range(1,self.n_j):
                nj_coef, self.__penalties[chain][j], nj_scale, j_embS = utils.solve_am(x_split[j],
                                                                                       y_split[j],
                                                                                       self.__penalties[chain][j],
                                                                                       0,maxiter=10)
                n_coef = np.concatenate((n_coef,nj_coef))
                
                n_scale.append(nj_scale)
                tot_penalty += nj_coef.T @ j_embS @ nj_coef
                
                n_logprobs[n_state_est_fl == j] = self.e_bs(self.n_j,y_split[j], x_split[j],
                                                            nj_coef, nj_scale, False,
                                                            **self.__e_bs_kwargs)

            n_scale = np.array(n_scale)

        else:
            ### Otherwise we can fit a shared model.
            n_coef, self.__penalties[chain], n_scale, embS = utils.solve_am(n_mat,self.series_fl,
                                                                            self.__penalties[chain],
                                                                            0,maxiter=10)
            tot_penalty = n_coef.T @ embS @ n_coef

            ### Get probabilities of observing series under new model for all states at once.
            n_logprobs = self.e_bs(self.n_j,self.series_fl,n_mat, n_coef,n_scale, False, mask_by_j=None)


        ## M-step sojourn distributions (see utils)
        n_p_pars = self.m_ps(self.n_j,self.end_points,c_p_pars,n_state_dur_est,n_state_est,self.covariates,self.split_p_by_cov)

        ## M-steps for initial and transition distributions can be completed optionally (see utils)
        if self.estimate_pi:
            n_pi = self.m_pi(self.n_j,c_pi,n_state_dur_est,self.covariates)
        else:
            n_pi = np.copy(c_pi)
        
        if self.estimate_TR:
            n_TR = self.m_TR(self.n_j,c_TR,n_state_dur_est,self.covariates)
        else:
            n_TR = np.copy(c_TR)

        # Now calculate likelihood of new model

        ## Parameterize the sojourn distributions using current parameters
        ps = self.par_ps(n_p_pars)

        ## Then calculate log likelihood of new model
        llks_new = self.__calc_llk_all_events(pool,n_pi,n_TR,n_state_dur_est,n_state_est,ps,n_logprobs)

        return np.sum(llks_new) - tot_penalty, n_state_dur_est, n_state_est, n_coef, n_p_pars,n_pi, n_TR,n_scale

    def fit(self,i_pi,i_TR,i_coef,i_scale,i_p_pars,i_state_durs,i_states,n_chains=10,collect_hist=False):
        
        ### Initialize
        iter = len(self.__temp)
        self.__state_dur = deepcopy(i_state_durs)
        self.__state = deepcopy(i_states)

        self.__phi = deepcopy(i_p_pars)
        self.__coef = deepcopy(i_coef)
        self.__scale = deepcopy(i_scale)
        
        self.__pi = deepcopy(i_pi)
        self.__TR = deepcopy(i_TR)
        
        # Deepcopy penalties to multiple chains
        self.__penalties = [deepcopy(self.__penalties) for _ in range(n_chains)]

        if collect_hist:
            self.__state_dur_hist = [[deepcopy(i_state_dur)] for i_state_dur in i_state_durs]
            self.__state_hist = [[deepcopy(i_st)] for i_st in i_states]

            self.__phi_hist = [[deepcopy(i_pp)] for i_pp in i_p_pars]
            self.__coef_hist = [[deepcopy(i_cf)] for i_cf in i_coef]
            self.__scale_hist = [[deepcopy(i_sig)] for i_sig in i_scale]
            
            self.__pi_hist = [[deepcopy(i_pii)] for i_pii in i_pi] 
            self.__TR_hist = [[deepcopy(i_tr)] for i_tr in i_TR] 
        
        # Always collect llk changes
        self.__llk_hist = np.zeros((n_chains,iter+1))
        self.__llk_hist[:,0] = - np.Inf

        with mp.Pool(processes=self.cpus) as pool:
            for i in tqdm(range(iter)):

                for ic in range(n_chains):

                    # Perform SEM step for current chain.
                    llk, n_states_dur, n_states, \
                    n_coef, n_p_pars, n_pi, \
                    n_TR, n_scale  = self.__advance_chain(ic,pool,
                                                          self.__temp[i],
                                                          self.__pi[ic],
                                                          self.__TR[ic],
                                                          self.__phi[ic],
                                                          self.__coef[ic],
                                                          self.__scale[ic],
                                                          self.__state_dur[ic],
                                                          self.__state[ic])
                    
                    self.__pi[ic] = n_pi
                    self.__TR[ic] = n_TR
                    self.__phi[ic] = n_p_pars
                    self.__coef[ic] = n_coef
                    self.__scale[ic] = n_scale
                    self.__state_dur[ic] = n_states_dur
                    self.__state[ic] = n_states
                    
                    # Store new parameters
                    if collect_hist:
                        self.__state_dur_hist[ic].append(deepcopy(n_states_dur))
                        self.__state_hist[ic].append(deepcopy(n_states))

                        self.__coef_hist[ic].append(deepcopy(n_coef))
                        self.__scale_hist[ic].append(deepcopy(n_scale))

                        self.__phi_hist[ic].append(deepcopy(n_p_pars))

                        self.__pi_hist[ic].append(deepcopy(n_pi))
                        self.__TR_hist[ic].append(deepcopy(n_TR))
                    self.__llk_hist[ic,i+1] = llk
                    
        # Plot log-likelihood for all chains
        for ic in range(n_chains):
            plt.plot(self.__llk_hist[ic,:])
        plt.show()
