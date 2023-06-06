import numpy as np
from tqdm import tqdm
import warnings
from . import utils
import warnings
import multiprocessing as mp
from itertools import repeat
from matplotlib import pyplot as plt
from copy import deepcopy


class sMsGAMMBase:

    def __init__(self,n_j,series,time,
                 end_points,llk_fun,pre_llk_fun,
                 covariates=None,is_DC=True,
                 sep_per_j=None,
                 estimate_pi=False,
                 estimate_TR=False,
                 cpus=1):
        
        self.n_j = n_j # Number of latent states
        self.series = series 
        self.n_s = len(series)
        self.series_fl = np.array([o for s in series for o in s])
        self.time = time

        ## Log-likelihood and "prior" Log-likelihood functions
        self.llk_fun = llk_fun
        self.pre_llk_fun = pre_llk_fun

        self.end_points = end_points
        self.covariates = covariates
        
        self.is_DC = is_DC
        self.sep_per_j = sep_per_j
        self.estimate_pi = estimate_pi
        self.estimate_TR = estimate_TR

        self.cpus=cpus

        # Build NAN index
        self.is_NA_fl = np.isnan(self.series_fl)

        # Temperature schedule
        self.__temp = None

        # Containers
        self.__coef_hist = None
        self.__state_dur_hist = None
        self.__state_hist = None
        self.__scale_hist = None
        self.__sigma_hist = None
        self.__TR_hist = None
        self.__pi_hist = None

        # Function storage
        self.__sample_fun = None
        self.__sample_fun_kwargs = None
        self.__build_mat_fun = None
        self.__build_mat_kwargs = None
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
    
    def set_mat_fun(self,fun,**kwargs):
        self.__build_mat_fun = fun
        self.__build_mat_kwargs = kwargs
    
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

    def set_penalties(self,penalties):
        self.__penalties = penalties
    
    ##################################### Getters #####################################

    def get_last_pars_chain(self,chain=0):
        return self.__state_hist[chain][-1], \
               self.__coef_hist[chain][-1], \
               self.__sigma_hist[chain][-1], \
               self.__scale_hist[chain][-1], \
               self.__pi_hist[chain][-1], \
               self.__TR_hist[chain][-1]
    
    def get_last_pars_max(self):
        chain_last_llks = self.__llk_hist[:,-1]
        idx = np.array(list(range(len(chain_last_llks))))
        idx = idx[chain_last_llks == max(chain_last_llks)][0]
        return idx,self.__state_hist[idx][-1], \
               self.__coef_hist[idx][-1], \
               self.__sigma_hist[idx][-1], \
               self.__scale_hist[idx][-1], \
               self.__pi_hist[idx][-1], \
               self.__TR_hist[idx][-1] \
    
    def get_last_llk_max(self):
        chain_last_llks = self.__llk_hist[:,-1]
        idx = np.array(list(range(len(chain_last_llks))))
        idx = idx[chain_last_llks == max(chain_last_llks)][0]
        return idx, chain_last_llks[idx]

    def get_gcv_max(self):
        if self.sep_per_j is not None and not self.is_DC:
            warnings.warn("GCV currently only supported for DC GAMM")
            return None

        chain_last_llks = self.__llk_hist[:,-1]
        idx = np.array(list(range(len(chain_last_llks))))
        idx = idx[chain_last_llks == max(chain_last_llks)][0]

        best_states = self.__state_hist[idx][-1]
        best_coef = self.__coef_hist[idx][-1]
        best_penalties = self.__penalties[idx]
        with mp.Pool(processes=self.cpus) as pool:
            n_mat = self.__build_all_mod_mat(pool,best_states)
        n_mat = n_mat[np.isnan(self.series_fl) == False,:]
        y = self.series_fl[np.isnan(self.series_fl) == False]
        gcv = utils.compute_gcv(y,n_mat,best_coef,best_penalties)
        
        return gcv

    def get_penalties(self):
        return self.__penalties
    
    def get_coef_hist(self):
        return self.__coef_hist
    
    def get_state_hist(self):
        return self.__state_hist
    
    def get_scale_hist(self):
        return self.__scale_hist
    
    ##################################### MP handlers #####################################
    
    def __build_all_mod_mat(self,pool,state_est):
        args = zip(self.time, self.covariates, state_est)
        mapping = zip(repeat(self.__build_mat_fun), args, repeat(self.__build_mat_kwargs))
        

        all_mat = pool.starmap(utils.map_unwrap_args_kwargs,mapping)
        all_mat = np.concatenate(all_mat,axis=0)
        return all_mat
    
    def __propose_all_states(self,pool,temp,pi,TR,state_durs_est,state_est,ps,coef,sigma):
        args = zip(repeat(self.n_j),repeat(temp),self.series,self.end_points,self.time,
                   self.covariates,repeat(pi),repeat(TR),state_durs_est,state_est,repeat(ps),
                   repeat(coef),repeat(sigma),repeat(self.__build_mat_fun), repeat(self.pre_llk_fun),
                   repeat(self.llk_fun),repeat(self.e_bs),repeat(self.sep_per_j is not None),
                   repeat(self.__build_mat_kwargs),repeat(self.__e_bs_kwargs))
        
        mapping = zip(repeat(self.__sample_fun),args,repeat(self.__sample_fun_kwargs))

        state_durs_new, states_new = zip(*pool.starmap(utils.map_unwrap_args_kwargs,mapping))
        return list(state_durs_new),list(states_new)
    
    def __calc_llk_all_events(self,pool,pi,TR,state_dur_est,state_est,ps,logprobs):
        args = zip(repeat(self.n_j),repeat(pi),repeat(TR),state_dur_est,state_est,repeat(ps),logprobs,self.covariates)
        llks = pool.starmap(self.llk_fun,args)
        return llks
    
    ##################################### SEM - Estimation #####################################

    def __advance_chain(self,chain,pool,temp,c_pi,c_TR,c_p_pars,c_coef,c_sigma,c_state_durs_est,c_state_est):
        # Performs One Stochastic Expectation maiximization iteration (or something quite like it if is_DC=True).
        # See utils.py for details but also Nielsen (2002).

        # Propose new candidates

        ## Parameterize duration distributions based on current/previous parameters
        ps = self.par_ps(c_p_pars)
        
        ## Sample next latent state estimate (Stochastic E-step)
        n_state_dur_est, n_state_est = self.__propose_all_states(pool,temp,c_pi,c_TR,
                                                                 c_state_durs_est,
                                                                 c_state_est,ps,
                                                                 c_coef,c_sigma)

        # Calculate M-steps based on new latent state estimate

        ## First calculate design matrix for every series and combine.
        n_mat = self.__build_all_mod_mat(pool,n_state_est)

        ## Now actually update coefficients based on proposal
        if self.sep_per_j is not None and not self.is_DC:
            ### If we have truely separate models per latent state, we update the coefs and sigmas separately
            n_logprobs = np.zeros(len(self.series_fl))
            
            n_state_est_fl = np.array([st for s in n_state_est for st in s],dtype=int)
            y_split, x_split = self.sep_per_j(self.n_j,n_state_est_fl,self.series_fl, n_mat)
            
            n_coef, self.__penalties[chain][0], n_sigma, j_embS = utils.solve_am(x_split[0],
                                                                                 y_split[0],
                                                                                 self.__penalties[chain][0],
                                                                                 0,maxiter=10)
            n_sigma = [n_sigma]
            

            ### Get probabilities of observing series under new model for this state
            n_logprobs[n_state_est_fl == 0] = self.e_bs(self.n_j,y_split[0], x_split[0], n_coef,
                                                        n_sigma[0], False, **self.__e_bs_kwargs)

            tot_penalty = n_coef.T @ j_embS @ n_coef
            
            #### Repeat for remaining states
            for j in range(1,self.n_j):
                nj_coef, self.__penalties[chain][j], nj_sigma, j_embS = utils.solve_am(x_split[j],
                                                                                       y_split[j],
                                                                                       self.__penalties[chain][j],
                                                                                       0,maxiter=10)
                n_coef = np.concatenate((n_coef,nj_coef))
                
                n_sigma.append(nj_sigma)
                tot_penalty += nj_coef.T @ j_embS @ nj_coef
                
                n_logprobs[n_state_est_fl == j] = self.e_bs(self.n_j,y_split[j], x_split[j],
                                                            nj_coef, nj_sigma, False,
                                                            **self.__e_bs_kwargs)

            n_sigma = np.array(n_sigma)

        else:
            ### Otherwise we can fit a shared model.
            n_coef, self.__penalties[chain], n_sigma, embS = utils.solve_am(n_mat,self.series_fl,
                                                                            self.__penalties[chain],
                                                                            0,maxiter=10)
            tot_penalty = n_coef.T @ embS @ n_coef

            ### Get probabilities of observing series under new model for all states at once.
            n_logprobs = self.e_bs(self.n_j,self.series_fl,n_mat, n_coef,n_sigma, False, mask_by_j=None)


        ## M-step sojourn distributions (see utils)
        n_p_pars = self.m_ps(self.n_j,self.end_points,c_p_pars,n_state_dur_est,n_state_est,self.covariates)

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

        return np.sum(llks_new) - tot_penalty, n_state_dur_est, n_state_est, n_coef, n_p_pars,n_pi, n_TR,n_sigma

    def fit(self,i_pi,i_TR,i_coef,i_sigma,i_p_pars,i_state_durs,i_states,n_chains=10):

        # Deepcopy penalties to multiple chains
        self.__penalties = [deepcopy(self.__penalties) for _ in range(n_chains)]

        iter = len(self.__temp)
        self.__state_dur_hist = deepcopy(i_state_durs)
        self.__state_hist = deepcopy(i_states)

        self.__scale_hist = deepcopy(i_p_pars)
        self.__coef_hist = deepcopy(i_coef)
        self.__sigma_hist = deepcopy(i_sigma)
        
        self.__pi_hist = deepcopy(i_pi)
        self.__TR_hist = deepcopy(i_TR)

        self.__llk_hist = np.zeros((n_chains,iter+1))
        self.__llk_hist[:,0] = - np.Inf

        with mp.Pool(processes=self.cpus) as pool:
            for i in tqdm(range(iter)):

                for ic in range(n_chains):

                    # Perform SEM step for current chain.
                    llk, n_states_dur, n_states, \
                    n_coef, n_p_pars, n_pi, \
                    n_TR, n_sigma  = self.__advance_chain(ic,pool,self.__temp[i],
                                                          self.__pi_hist[ic][i],
                                                          self.__TR_hist[ic][i],
                                                          self.__scale_hist[ic][i],
                                                          self.__coef_hist[ic][i],
                                                          self.__sigma_hist[ic][i],
                                                          self.__state_dur_hist[ic][i],
                                                          self.__state_hist[ic][i])
                    
                    # Store new parameters
                    self.__state_dur_hist[ic].append(deepcopy(n_states_dur))
                    self.__state_hist[ic].append(deepcopy(n_states))

                    self.__coef_hist[ic].append(deepcopy(n_coef))
                    self.__sigma_hist[ic].append(deepcopy(n_sigma))

                    self.__scale_hist[ic].append(deepcopy(n_p_pars))

                    self.__pi_hist[ic].append(deepcopy(n_pi))
                    self.__TR_hist[ic].append(deepcopy(n_TR))
                    self.__llk_hist[ic,i+1] = llk
                    
        # Plot log-likelihood for all chains
        for ic in range(n_chains):
            plt.plot(self.__llk_hist[ic,:])
        plt.show()

class sMsGAMM(sMsGAMMBase):

    def __init__(self, n_j, series, time,
                 end_points, llk_fun, pre_llk_fun,
                 covariates=None, is_DC=True,
                 sep_per_j=utils.j_split,
                 estimate_pi=True,
                 estimate_TR=True, cpus=1):
        super().__init__(n_j, series, time,
                         end_points, llk_fun, pre_llk_fun,
                         covariates, is_DC, sep_per_j,
                         estimate_pi, estimate_TR, cpus)

        if sep_per_j is not None:
            self.set_e_bs(utils.get_log_o_prob_mat)
            self.set_sample_fun(utils.se_step_sms_gamm)
        else:
            warnings.warn("Don't forget to call set_e_bs() and set_sample_fun().")

        self.set_m_pi(utils.m_pi_sms_gamm)
        self.set_m_TR(utils.m_TR_sms_gamm)
        self.set_m_ps(utils.m_gamma2s_sms_gamm)
        self.set_par_ps(utils.par_gamma2s)

class sMsDCGAMM(sMsGAMMBase):

    def __init__(self, n_j, series, time,
                 end_points, llk_fun, pre_llk_fun,
                 covariates=None, is_DC=True,
                 sep_per_j=None, estimate_pi=False,
                 estimate_TR=False, cpus=1):
        super().__init__(n_j, series, time,
                         end_points, llk_fun, pre_llk_fun,
                         covariates, is_DC, sep_per_j,
                         estimate_pi, estimate_TR, cpus)

        self.set_e_bs(utils.get_log_o_prob_mat)
        self.set_m_ps(utils.m_gamma2s_sms_dc_gamm)
        self.set_par_ps(utils.par_gamma2s)
        self.set_sample_fun(utils.se_step_sms_dc_gamm)