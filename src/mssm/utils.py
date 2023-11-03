import numpy as np
import pandas as pd
import scipy as scp
from dataclasses import dataclass
import math
from enum import Enum
from collections.abc import Callable
import re
import warnings
import copy

##################################### Complete data log-likelihood and Pre-likelihood functions #####################################

def ll_sms_dc_gamm(n_j,pi,TR,state_dur_est, state_est,ps,logprobs,cov,split_p_by_cov):
    # Complete data likelihood function for left-right de-convolving sMs GAMM.
    # Yu (2011): alpha_t(j) = Probability of state j terminating at time t
    # AND observing values o[1:t] given current parameters.

    # In the left-right sms dc gamm model we further assume last state ends at last time
    # point. Thus, we don't need to marginalize out j - we simply assume no other state
    # can terminate at last time-point. Thus our complete data likelihood of observing values o[1:last_timepoint]
    # given current parameters is simply alpha_last_timepoint(n_j).

    # Because of the left-right assumptions calculation of the complete data likelihood
    # is trivial (essentially a simplification of equation 15 in Yu, 2011).
    # We simply add the log probabilities of every state lasting until the onset of the
    # subsequent one.
    # We then add the sum of the log-probabilities of observing y_t at any t, given our
    # GAMM based on the known state onsets. This works because we assume that given
    # stage onsets, given the GAMM parameters, given the gamm predictions Y_hat_t:
    # Y_t ~ N(Y_hat_t,sigma) so Y is i.i.d.

    # Log-probs can include NAs, so we need to handle those
    # For the de-convolving sMS GAMM we can simply drop them.
    c_logprobs = logprobs[np.isnan(logprobs) == False]


    js = np.arange(n_j+1) # This has been changed for varying first and last events
    alpha = 0

    # Duration probability for every state
    for j in js:
        
        # Check for zero scale Gammas belonging to fixed events.
        if split_p_by_cov is None:
           j_s = j
        elif j not in split_p_by_cov["split"]:
           j_s = j
        else:
           j_s = j + cov[0,split_p_by_cov["cov"]]

        if not np.isnan(ps[j_s].mean()):
         alpha +=  ps[j_s].logpdf(state_dur_est[j,1])

    # observation probabilities
    alpha += np.sum(c_logprobs)

    # Complete data likelihood.
    return alpha

def pre_ll_sms_dc_gamm_fixed_full(n_j, end_point, state_dur_est, state_est):
    # Prior likelihood check for left-right de-convolving sMs GAMM.
    # Checks whether likelihood should be set to - inf based
    # on any knowledge about the support of the likelihood function.

    # Assumes a fixed event at to and ent_point!

    # Cannot have state onset before zero
    if np.any(state_est < 0):
       return True
    
    # Cannot have state onset after trial is over
    if np.any(state_est > end_point):
      return True
    
    # Cannot violate left-right ordering
    if np.any(state_est[1:len(state_est)] - state_est[0:-1] < 1):
       return True
    
    return False

def pre_ll_sms_dc_gamm_fixed_first(n_j, end_point, state_dur_est, state_est):
    # Prior likelihood check for left-right de-convolving sMs GAMM.
    # Checks whether likelihood should be set to - inf based
    # on any knowledge about the support of the likelihood function.

    # Assumes a fixed event at t0!

    # Cannot have state onset before zero
    if np.any(state_est < 0):
       return True
    
    # Cannot have state onset after end_point - 1
    if np.any(state_est >= end_point):
      return True
    
    # Cannot violate left-right ordering
    if np.any(state_est[1:len(state_est)] - state_est[0:-1] < 1):
       return True
    
    return False

def pre_ll_sms_dc_gamm(n_j, end_point, state_dur_est, state_est):
    # Prior likelihood check for left-right de-convolving sMs GAMM.
    # Checks whether likelihood should be set to - inf based
    # on any knowledge about the support of the likelihood function.

    # Assumes no fixed event at all!

    # Cannot have state onset before 1
    if np.any(state_est <= 0):
       return True
    
    # Cannot have state onset after end_point - 1
    if np.any(state_est >= end_point):
       return True
    
    # Cannot violate left-right ordering
    if np.any(state_est[1:len(state_est)] - state_est[0:-1] < 1):
       return True
    
    return False

##################################### Stochastic E-step proposal algorithms #####################################

def prop_norm(end_point,c_state_est,sd,fix):
    # Standard proposal for random walk MH sampler, acts as proposal for
    # left-right (de-convolving) sMs GAMM: We simply propose a neighbor
    # for every state onset according to an onset centered Gaussian.

    # This has been changed to allow for varying first and last event
    n_state_est = np.round(scp.stats.norm.rvs(loc=c_state_est,scale=sd)).astype(int)

    if fix is not None:
       for event in fix:
          # event[0] holds event location that should be fixed
          # event[1] holds the sample at which it should be fixed.
          if event[1] == "last":
            n_state_est[event[0]] = end_point
          else:
            n_state_est[event[0]] = event[1]

    n_state_dur_est = []
    last_state_est = 0
    for j in range(len(n_state_est)):
       n_state_dur_est.append([j, n_state_est[j] - last_state_est])
       last_state_est = n_state_est[j]
    n_state_dur_est.append([len(n_state_est), end_point - n_state_est[-1]])
    
    return np.array(n_state_dur_est), n_state_est

##################################### Stochastic E-step algorithms #####################################

def se_step_sms_dc_gamm(n_j,temp,series,end_point,time,cov,pi,TR,
                        state_durs_est,state_est,ps,coef,sigma,
                        create_matrix_fn, pre_lln_fn, ll_fn,e_bs_fun,
                        repeat_by_j,split_p_by_cov,build_mat_kwargs,e_bs_kwargs,
                        sd=2,fix=None,n_prop = 50):
    
    # Proposes a new candidate solution for a left-right DC Gamm via a random walk step. The proposed
    # state is accepted according to a modified Metropolis Hastings acceptance ratio. In fact, the decision
    # is closer to a simulated annealing decision since we have a temparature schedule as well. However, we
    # never let the temparature drop to zero since our complete data likelihood around candidates drops rapidly
    # so if we would drop the temperature to zero (and thus fall back to steepest ascent on the likelihood surface)
    # we would immediately get stuck on whatever maximum we currently are. By ensuring that temp >= 1 we keep exploring
    # new candidates. This is actually closer to simulated tempering than annealing (Marinari & Parisi 1992;
    # Allassonnière & Chevallier, 2021). But we use a continuously decreasing temperature function to make sure that we
    # still explore less with time which is more like annealing again... So the decision is a mixture of both.
    #
    # Some not yet organized thoughts:
    # This proposal is actually not a sample from P(State | obs, current_par) as suggested in the classical stochastic expectation step (Nielsen, 2000).
    # In principle if we increase n_prop state_dist should however become a close representation of this distribution. However, in
    # practice there is no benefit to performance since the MH acceptance step makes sure that in the long-run we are likely to improve
    # our estimates. Not sampling directly from P(State | obs, current_par) also seems to be a feature of Stochastic approximate Expectation Maximization or
    # Stochastic Simulated Annealing Expectation Maximization. However, these approaches seem to form an iterative approximation of Q: Q_{t+1} = Q_{t-1} + lambda * Q_{t},
    # where Q_{t} is based on the stochastic/approximate E step, and then maximize Q_{t+1} (see Celeux, Chauveau & Diebolt, 1995; Delyon, 1999).
    # We simply maximize the CDL based on our candidate - which IS again closer to traditional SEM.. So it's again a combination of a couple things that work well in practice.
    # The sampler for the most general case (se_step_sms_gamm) behaves more like traditional SEM and the acceptance step should in principle not be necessary.

    #n_prop = 50 # Depending on the proposal function it can be useful to sample a lot of candidates and then select from all those at random
    n_cand = 1 # I considered allowing to return multiple samples, which might help with performance but I haven't implemented that yet.

    c_state_est = np.copy(state_est)
    c_mat = create_matrix_fn(time,cov,c_state_est,**build_mat_kwargs)
    c_logprobs = e_bs_fun(n_j,series,c_mat,coef,sigma,repeat_by_j,**e_bs_kwargs)
    c_state_durs_est = np.copy(state_durs_est)
    c_llk = ll_fn(n_j,pi,TR,c_state_durs_est,state_est,ps,c_logprobs,cov,split_p_by_cov)

    cutoffs = scp.stats.uniform.rvs(size=n_prop)

    state_dist = np.zeros((n_prop,n_j),dtype=int)
    states_durs_dist = []
    acc = 0

    for i in range(n_prop):
      
      # Propose new onset
      # This has been changed to handle varying first and last events
      n_state_durs_est,n_state_est = prop_norm(end_point,c_state_est,sd,fix)

      # Pre-check new proposal
      rejection = pre_lln_fn(n_j, end_point, n_state_durs_est, n_state_est)

      if not rejection:
        # Calculate likelihood of proposal given observations and current parameters
        n_mat = create_matrix_fn(time,cov,n_state_est,**build_mat_kwargs)
        n_logprobs = e_bs_fun(n_j,series,n_mat,coef,sigma,repeat_by_j,**e_bs_kwargs)
        n_llk = ll_fn(n_j,pi,TR,n_state_durs_est,n_state_est,ps,n_logprobs,cov,split_p_by_cov)

        # Simulated Annealing/Metropolis acceptance
        if (n_llk > c_llk) or (np.exp((n_llk - c_llk)/temp) >= cutoffs[i]):
          c_state_est = np.copy(n_state_est)
          c_state_durs_est = np.copy(n_state_durs_est)
          c_mat = np.copy(n_mat)
          c_llk = n_llk
          acc += 1
      
      # Update onset distribution estimate
      state_dist[i,:] = np.copy(c_state_est)
      states_durs_dist.append(np.copy(c_state_durs_est))
    

    # Sample new onset according to ~ P(onset | series, current parameters). The latter only
    # holds with n_prop = LARGE.
    sample = np.random.randint(n_prop,size=n_cand)[0]
    
    return states_durs_dist[sample],state_dist[sample,:]

##################################### Other M-steps #####################################

def m_gamma2s_sms_dc_gamm(n_j,end_points,p_pars,state_dur_est,state_est,cov,split_p_by_cov):
   # For the deconvolving left-right case, we know every series has the same
   # number of states and transitions - so we can cast to numpy array to make
   # indexing easier.
   # Optimal estimate for scales is then simply the mean duration per state divided by
   # our fixed shape, i.e., 2. See Anderson et al., 2016

   # This has been changed to support varying first and last events
   # Assuming end at 30:
   # Example fixed at begin and end: [0,3,12,30]
   # Expand: [0,0,3,12,30,30]
   # Durs: [0-0==0,3-0==3,12-3==9,30-12==18,30-30==0]
   # Example varying at begin and end: [1,3,12,28]
   # Expand: [0,1,3,12,28,30]
   # Durs: [1-0==1,3-1==2,12-3==9,28-12==16,30-28==2]
   np_state_est = np.array(state_est)
   np_state_est = np.insert(np_state_est,0,0,axis=1)
   np_state_est = np.insert(np_state_est,np_state_est.shape[1],end_points,axis=1)
   durs = np.array([np_state_est[:,j] - np_state_est[:,j-1] for j in range(1,np_state_est.shape[1])])

   if split_p_by_cov is None:
      scales = [dur/2 for dur in np.mean(durs,axis=1)]
   else:
      cov_split = np.array([c[0,split_p_by_cov["cov"]] for c in cov]) # cov[:,split_p_by_cov["cov"]] should be categorical so identical over dim 0
      scales = []
      for j in range(durs.shape[0]):
         if j not in split_p_by_cov["split"]:
            scales.append(np.mean(durs[j,:])/2)
         else:
            for c in range(max(cov_split)+1):
               scales.append(np.mean(durs[j,cov_split == c])/2)
   
   return scales


##################################### Other #####################################

def map_unwrap_args_kwargs(fn, args, kwargs):
    # Taken from isarandi's answer to: https://stackoverflow.com/questions/45718523/
    # Basis for entire MP code used in this project.
    return fn(*args, **kwargs)
