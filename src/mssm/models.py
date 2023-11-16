import numpy as np
import scipy as scp
from tqdm import tqdm
import warnings
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
from .src.python.sem import *
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

    def get_mmat(self,):
        # Returns the model-matrix used for fitting.
        if self.formula.penalties is None:
            raise ValueError("Model matrix cannot be returned if penalties have not been initialized. Call model.fit() first.")
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

    ##################################### Fitting #####################################
    
    def fit(self,maxiter=30,conv_tol=1e-7,extend_lambda=True,control_lambda=True,restart=False):

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
                                                                         conv_tol,extend_lambda,control_lambda)
        
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

    def __init__(self,
                 formula: Formula,
                 family: Family,
                 p_formula: PFormula,
                 end_points:list,
                 pre_llk_fun=pre_ll_sms_gamm,
                 estimate_pi: bool = True,
                 estimate_TR: bool = True,
                 pi=None,
                 TR=None,
                 cpus: int = 1):
        
        super().__init__(formula,
                         family,
                         p_formula,
                         pre_llk_fun,
                         estimate_pi,
                         estimate_TR,
                         cpus)
        
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

    ##################################### Fitting #####################################
    
    def __propose_all_states(self,pool,y,cov,temp,pi,TR,log_o_probs,log_dur_probs,var_map):
        # MP code to propose states for every series
        args = zip(repeat(self.n_j),repeat(temp),y,cov,self.end_points,repeat(pi),
                   repeat(TR),log_o_probs,repeat(log_dur_probs),repeat(self.pds),
                   repeat(self.pre_llk_fun),repeat(var_map))
        
        state_durs_new, states_new, llks = zip(*pool.starmap(se_step_sms_gamm,args))
        return list(state_durs_new),list(states_new),list(llks)
    
    def fit(self,maxiter_outer=100,maxiter_inner=30,max_no_improv=15,conv_tol=1e-4,extend_lambda=True,control_lambda=True,init_scale=100,t0=0.25,r=0.925):
        # Performs Stochastic Expectation maiximization based on Nielsen (2002) see also the sem.py file for
        # more details.
        
        has_scale_split = self.formula.has_scale_split()
        
        # Penalties need to be initialized and copied for each stage
        self.formula.build_penalties()
        
        if has_scale_split:
            penalties = [copy.deepcopy(self.formula.penalties) for j in range(self.n_j)]
        else:
            penalties = self.formula.penalties

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
        temp_schedule = anneal_temps_zero(maxiter_outer,t0,r)

        max_states_flat = None
        no_improv_iter = 0
        
        for iter in range(maxiter_outer):

            ### Stochastic Expectation ###

            # Propose new states based on all updated parameters.
            # First we need the probability of stage durations for every stage (i.e., under every sojourn dist.).
            # Since we usually care only about a small number of states, we
            # can neglect their impact on the forward and backward time complexity.
            # However, even ignoring this both algorithms still take at least number_of_obs*
            # maximum_duration steps (for n_j=1). Now in principle, a state could take as long as the
            # series lasts. But that would lead to an n_t*n_t complexity, which is
            # just not feasible. One way to constrain this is to just consider the
            # most likely durations under the current parameter set. We use the quantile
            # function to determine the highest 99% cut-off (only 1% of duration values are more
            # extreme than this one), across states which we then set as the max_dur to be considered.
            max_dur = int(max([round(pd.max_ppf(0.99)) for pd in self.pds]))

            c_durs = np.arange(1,max_dur)

            # So we need to collect the log-probabilities of state j lasting for duration d according to
            # state j's sojourn time distribution(s).
            dur_log_probs = []

            # For the forward and backward probabilities computed in the sem step
            # we also need for every time point the probability of observing the
            # series at that time-point according to the GAMM from EVERY state.
            log_o_probs = np.zeros((self.n_j,n_obs))

            for j in range(self.n_j):
                # Handle duration probabilities
                pd_j = self.pds[j]

                if pd_j.split_by is not None:
                    for ci in range(pd_j.n_by):
                        dur_log_probs.append(pd_j.log_prob(c_durs,ci))
                else:
                    dur_log_probs.append(pd_j.log_prob(c_durs))
                
                if has_scale_split:
                    # Handle observation probabilities
                    j_mu = (model_mat_full @ state_coef[j]).reshape(-1,1)

                    if not isinstance(self.family,Gaussian):
                        j_mu = self.family.link.fi(j_mu)

                    if not self.family.twopar:
                        log_o_probs[j,NOT_NA_flat] = np.ndarray.flatten(self.family.lp(y_flat[NOT_NA_flat],j_mu))
                    else:
                        log_o_probs[j,NOT_NA_flat] = np.ndarray.flatten(self.family.lp(y_flat[NOT_NA_flat],j_mu,state_scales[j]))
                    log_o_probs[j,NOT_NA_flat == False] = np.nan

                else:
                    raise NotImplementedError("has_scale_split==False is not yet implemented.")
                
            dur_log_probs = np.array(dur_log_probs)
            
            # We need to split the observation probabilities by series
            s_log_o_probs = np.split(log_o_probs,self.formula.sid[1:],axis=1)
            
            # Now we can propose a new set of states and state_durs for every series.
            with mp.Pool(processes=self.cpus) as pool:
                durs,states,llks = self.__propose_all_states(pool,self.formula.y,cov,temp_schedule[iter],n_pi,n_TR,s_log_o_probs,dur_log_probs,var_map)
                states_flat = np.array([st for s in states for st in s],dtype=int)
            
            ### Convergence control ###

            # Convergence control is based on the change in penalized likelihood between
            # two subsequent local maxima or the inability to find a new maximum for
            # a long time.
            if iter > 0:
                pen_llk = np.sum(llks) - np.sum(state_penalties)

                if iter == 1: # Initialize max pen llk
                    max_pen_llk = pen_llk
                if iter > 1:

                    if pen_llk >= max_pen_llk: # Collect new best parameters
                        self.__TR = n_TR
                        self.__pi = n_pi
                        self.__scale = state_scales
                        self.__coef = state_coef
                        self.lvi = state_LVIs
                        max_states_flat = states_flat
                        
                        # Also check convergence
                        if (pen_llk - max_pen_llk) < conv_tol*abs(pen_llk):
                            print("Converged",iter)
                            break

                        max_pen_llk = pen_llk
                        no_improv_iter = -1 # reset!

                    no_improv_iter += 1

                    if no_improv_iter >= max_no_improv:
                        print("Converged",iter)
                        break

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
                                                                                    conv_tol,extend_lambda,control_lambda)
                    
                    
                    
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
                 p_formula: PFormula,
                 end_points: list,
                 fix:None or list[list[int,int]] = None,
                 pre_llk_fun=pre_ll_sms_IR_gamm,
                 cpus: int = 1):

        super().__init__(formula,
                         family,
                         p_formula,
                         end_points,
                         pre_llk_fun,
                         False,
                         False,
                         np.zeros(formula.get_nj()),
                         np.zeros((formula.get_nj(),
                                   formula.get_nj())),
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
    
    def fit(self,maxiter_outer=100,maxiter_inner=30,conv_tol=1e-6,extend_lambda=True,control_lambda=True,t0=1,r=0.925,schedule="anneal",n_prop=None,prop_sd=2):
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

        model_mat_full = build_sparse_matrix_from_formula(terms,has_intercept,False,
                                                          ltx,irstx,stx,rtx,var_types,var_map,
                                                          var_mins,var_maxs,factor_levels,
                                                          cov_flat,cov,None,
                                                          None,states)
        
        # Only now can we remove the NAs
        model_mat_full = model_mat_full[NOT_NA_flat,]
        
        # And estimate the model

        # Get initial estimate of mu based on family:
        init_mu_flat = self.family.init_mu(y_flat[NOT_NA_flat])

        coef,eta,wres,scale,LVI,edf,term_edf,penalty = solve_gamm_sparse(init_mu_flat,y_flat[NOT_NA_flat],
                                                                        model_mat_full,penalties,self.formula.n_coef,
                                                                        self.family,maxiter_inner,"svd",
                                                                        conv_tol,extend_lambda,control_lambda)

        # For state proposals we can utilize a temparature schedule. See sMsGamm.fit().
        if schedule == "anneal":
            temp_schedule = anneal_temps_zero(maxiter_outer,t0,r)
        else:
            temp_schedule = const_temps(maxiter_outer)

        last_llk = None
        llk_hist = []
        for iter in range(maxiter_outer):
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
                    if abs(pen_llk - last_llk) < conv_tol*abs(pen_llk):
                        print("Converged",iter)
                        break

                last_llk = pen_llk
                llk_hist.append(pen_llk)

            ### Maximization ###

            # First update all GAMM parameters
            model_mat_full = build_sparse_matrix_from_formula(terms,has_intercept,False,
                                                              ltx,irstx,stx,rtx,var_types,var_map,
                                                              var_mins,var_maxs,factor_levels,
                                                              cov_flat,cov,None,
                                                              None,states)
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
                                                                             conv_tol,extend_lambda,control_lambda)

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

        plt.plot(llk_hist)
        plt.show()
        return LO_llk

