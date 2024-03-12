import numpy as np
import scipy as scp
import math
from .formula import build_sparse_matrix_from_formula
from .exp_fam import Gaussian

##################################### Temperature functions #####################################

def anneal_temps_zero(iter,t0=0.25,ratio=0.925):
   # Annealing schedule as proposed by Kirkpatrick, Gelatt and Vecchi (1983).
   # see Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by Simulated Annealing. https://doi.org/10.1126/science.220.4598.671
   ts = np.array(list(range(iter)))

   temp = np.array([t0*ratio**t for t in ts])
   return temp

def const_temps(iter):
   # Don't use temperature schedule.
   ts = np.array(list(range(iter)))

   return np.array([1 for t in ts])

##################################### HsMM functions #####################################

def forward_eta(n_j,n_t,pi,TR,log_dur_mat,log_obs_mat):
   # Forward pass for HsMM based on math in Yu (2011) and inspired by
   # implementation in edhsmm (https://github.com/poypoyan/edhsmm).
   # see: Yu, S.-Z. (2010). Hidden semi-Markov models. https://doi.org/10.1016/j.artint.2009.11.011

   # Our sampler requires the probabilities P(S_t = j| obs, pars).
   # So we need only compute eta(t,j,d) to ultimately compute gamma(t,j).
   # For the etas we have to compute the forward and backward pass so we
   # get alphas and betas anyway.
   # Reminder: summing over alpha(last_t,j) gives P(obs|pars), so the
   # likelihood of the model. To obtain P(S_t = j | obs, pars)
   # we compute gamma(t,j) / sum(alpha(last_t,j)).

   # For forward pass we follow equations 15, 16, and 19 in Yu (2011)
   # But we work with logs to prevent precision loss.
   with np.errstate(divide="ignore"):
    log_pi = np.log(pi)
    log_TR = np.log(TR)

  # Number of durations to be considered:
   n_d = log_dur_mat.shape[1]

   # Storage
   etas = np.zeros((n_t,n_j,n_d)) - np.Inf
   u = np.zeros((n_t,n_j,n_d))

   alpha_stars = np.zeros((n_t,n_j))  - np.Inf # Storage equation 16
   for t in range(n_t):
      alpha_t = np.zeros(n_j) - np.Inf # Storage equation 15

      for j in range(n_j):

        for d in range(n_d):
           # Recursion for probabilities of values under our GAMM u(t,j,d)
           # defined in equation 20 from Yu (2011)
           u_next = log_obs_mat[j,t]

           if np.isnan(u_next):
              # If we have a missing observation, then we simply indicate that
              # this missing value could equally likely be encountered under every state.
              u_next = np.log(1/n_j)

           if t == 0 or d == 0:
              # Initialization as described in paragraph below eq 20
              u[t,j,d] = u_next
           else:
              u[t,j,d] = u[t-1,j,d-1] + u_next
           
           if t-d == 0:
              # Initialization based on simplifying assumption that a
              # state begins with observing the series.
              etas[t,j,d] = log_pi[j] + log_dur_mat[j,d] + u[t,j,d]
           elif t-d > 0:
              etas[t,j,d] = alpha_stars[t-d,j] + log_dur_mat[j,d] + u[t,j,d]

        alpha_t[j] = scp.special.logsumexp(etas[t,j,:]) # Calculating the sum in eq 15

      # Now we have all a(t,j) so we can complete eq 16.
      # The indexing is a bit odd here, makes it look like we are computing alpha_star(t=1,j)
      # that's not the case though. In Yu (2011) t = 1, ..., n_t - here we start at t = 0
      # for which we have the initialization. And since no new state can begin after the series is
      # over we stop at t = n_t - 1
      if t < n_t - 1:
        for j in range(n_j):
          for i in range(n_j):
              if i == j:
                continue # No self transitions in hsmms
              alpha_stars[t+1,j] = np.logaddexp(alpha_stars[t+1,j], alpha_t[i] + log_TR[i,j])
    
    # We have the llk in alpha, and the etas that we care about!
   return scp.special.logsumexp(alpha_t), etas, u

def backward_eta(n_j,n_t,TR,log_dur_mat,etas,u):
   # Backward pass based on math in Yu (2011) and inspired by
   # implementation in edhsmm (https://github.com/poypoyan/edhsmm).

   # Gets the partially computed etas from the forward_eta() function. Since we have already
   # computed the u term from equation 20, we pass that one along as well!
   # Computes gammas(t,j) on the fly.

   # For backward pass we rely on equations 17, 18, and also 19 but again we
   # work with logs.
   with np.errstate(divide="ignore"):
    log_TR = np.log(TR)

   # Computations of eta are based on a(t,j,d)*b(t,j,d), see eq  6 in Yu. We have already filled eta(t,j,d)
   # with the a's. However, based on the simplifying independence assumptions that we make,
   # bt,j,d) == b(t,j) - see paragraph before eq 15 in Yu (2011). So we need to store the b's
   # We also need to compute the gammas afterwards so we need to store those as well!

   # Number of durations to be considered:
   n_d = log_dur_mat.shape[1]

   # Storage
   betas = np.zeros((n_t,n_j)) - np.Inf
   betas[-1,:] = 0 # SImplifying assumption, see below!
   gammas = np.zeros((n_t,n_j)) - np.Inf

   # Now we loop backwards over the series.
   for t in range(n_t-1,-1,-1):
      beta_stars_t = np.zeros(n_j) - np.Inf

      if t > 0:
         for j in range(n_j):
            
            for d in range(n_d):
               if t + d < n_t:
                  # Initializing recursion of beta star i.e., eq 17 is not necessary (or rather has already
                  # happened) since we have initialized the betas with zeros (log(1)==0). This is based on simplifying
                  # assumption that b(last_time-point) = 1 (log(1) == 0)
                  beta_stars_t[j] = np.logaddexp(beta_stars_t[j], betas[t+d,j] + u[t+d,j,d] + log_dur_mat[j,d])
      
      # With the beta_stars calculated we can complete equation 18 for beta. Indexing here again makes this
      # look a bit odd.
      for j in range(n_j):
        if t > 0:
         for i in range(n_j):
            if i == j:
               continue # No self transitions in hsmms

            betas[t-1,j] = np.logaddexp(betas[t-1,j],log_TR[j,i] + beta_stars_t[i])

         # Now we can update etas as well - eq 6!
         etas[t-1,j,:] +=  betas[t-1,j]

        # Finally, we can calculate gammas, according to eq 8.
        for tau_add in range(min(n_d,n_t - t)):
          tau = t + tau_add
          for d in range(tau_add,n_d): # tau_add is difference between tau and t, so the term below the second sum in eq 8
              gammas[t,j] = np.logaddexp(gammas[t,j], etas[tau,j,d])
          
   return etas, gammas

##################################### sms GAMM SEM functions #####################################

def compute_log_probs(n_j,n_obs,has_scale_split,
                      model_mat,state_coef,
                      state_scales,pds,y_flat,
                      NOT_NA_flat,series_id,
                      family,factor_levels,
                      mvar_by):
   # We need the probability of stage durations for every stage (i.e., under every sojourn dist.).
   # Since we usually care only about a small number of states, we
   # can neglect their impact on the forward and backward time complexity.
   # However, even ignoring this both algorithms still take at least number_of_obs*
   # maximum_duration steps (for n_j=1). Now in principle, a state could take as long as the
   # series lasts. But that would lead to an n_t*n_t complexity, which is
   # just not feasible. One way to constrain this is to just consider the
   # most likely durations under the current parameter set. We use the quantile
   # function to determine the highest 99% cut-off (only 1% of duration values are expected more
   # extreme than this one), across states which we then set as the max_dur to be considered.
   max_dur = int(max([round(pd.max_ppf(0.99)) for pd in pds]))
   
   c_durs = np.arange(1,max_dur)

   # So we need to collect the log-probabilities of state j lasting for duration d according to
   # state j's sojourn time distribution(s).
   dur_log_probs = []

   # For the forward and backward probabilities computed in the sem step
   # we also need for every time point the probability of observing the
   # series at that time-point according to the GAMM from EVERY state.
   log_o_probs = np.zeros((n_j,n_obs))

   for j in range(n_j):
      # Handle duration probabilities
      pd_j = pds[j]

      if pd_j.split_by is not None:
         for ci in range(pd_j.n_by):
            dur_log_probs.append(pd_j.log_prob(c_durs,ci))
      else:
         dur_log_probs.append(pd_j.log_prob(c_durs))
      
      if has_scale_split:
         # Handle observation probabilities
         j_mu = (model_mat @ state_coef[j]).reshape(-1,1)

         if not isinstance(family,Gaussian):
            j_mu = family.link.fi(j_mu)

         # Prediction for y series according to state specific GAMM
         if not family.twopar:
            log_o_probs[j,NOT_NA_flat] = np.ndarray.flatten(family.lp(y_flat[NOT_NA_flat],j_mu))
         else:
            log_o_probs[j,NOT_NA_flat] = np.ndarray.flatten(family.lp(y_flat[NOT_NA_flat],j_mu,state_scales[j]))
         log_o_probs[j,NOT_NA_flat == False] = np.nan

      else:
            raise NotImplementedError("has_scale_split==False is not yet implemented.")
      
   dur_log_probs = np.array(dur_log_probs)
   
   # We need to split the observation probabilities by series
   s_log_o_probs = np.split(log_o_probs,series_id[1:],axis=1)

   if not mvar_by is None:
      # We need to split the s_log_o_probs from every series by the multivariate factor
      # and then sum the log-probs together. This is a strong independence assumption (see Langrock, 2021)
      n_by_mvar = len(factor_levels[mvar_by])
      s_log_o_probs = [s_prob.reshape(n_j,n_by_mvar,-1).sum(axis=1) for s_prob in s_log_o_probs]
   
   return s_log_o_probs,dur_log_probs

def prop_smoothed(n_j,n_t,smoothed):
   # Simple proposal for sMs-GAMM with state re-entries, based on
   # the smoothed probabilities P(State_t == j|parameters).
   # We simply select a state at every time-point t based on those
   # probabilities.
   # This is different from previous MCMC sampling approaches in the HsMM
   # literature (see Guedon, 2003; Guedon, 2005; Guedon, 2007 for alternative approaches)
   # but seems to work quite well.
   # Ref:
   # Guédon, Y. (2003). Estimating Hidden Semi-Markov Chains From Discrete Sequences https://doi.org/10.1198/1061860032030
   # Guédon, Y. (2005). Hidden hybrid Markov/semi-Markov chains. https://doi.org/10.1016/j.csda.2004.05.033
   # Guédon, Y. (2007). Exploring the state sequence space for hidden Markov and semi-Markov chains. https://doi.org/10.1016/j.csda.2006.03.015

   js = np.arange(n_j)

   n_state_est = np.zeros(n_t)
   for idx in range(len(n_state_est)):
    n_state_est[idx] = np.random.choice(js,p=smoothed[:,idx])
   
   n_state_dur_est = []
   c_state = n_state_est[0]
   c_dur = 1
   for s in range(1,len(n_state_est)):
      if n_state_est[s] != c_state:
         n_state_dur_est.append([c_state,c_dur])
         c_state = n_state_est[s]
         c_dur = 1
      c_dur += 1
    
   #print(np.array(state_durs),new_states)
   return np.array(n_state_dur_est,dtype=int), n_state_est

def pre_ll_sms_gamm(n_j, end_point, state_dur_est, state_est):
   # Prior likelihood check for sMs GAMM WITH state re-entries.
  
   # Cannot marginalize out one or more states - model matrix will become
   # unidentifiable.
   js = set(state_est)
   #print(js)
   if len(js) < n_j:
      return True

   return False

def compute_hsmm_probabilities(n_j,cov,pi,TR,log_o_probs,
                              log_dur_probs,pds,var_map,
                              compute_smoothed):
   
   # Computes bunch of probabilities defined by Yu (2011)
   s_log_dur_probs = np.zeros((n_j,log_dur_probs.shape[1]))
   j = 0
   j_split = 0
   while j < n_j:
      pd_j = pds[j]
      if pd_j.split_by is not None:
         #print(j,j_split+int(cov[0,var_map[pd_j.split_by]]),cov[0,var_map[pd_j.split_by]])
         s_log_dur_probs[j,:] = log_dur_probs[j_split+int(cov[0,var_map[pd_j.split_by]]),:]
         j_split += pd_j.n_by
      else:
         #print(j,j_split)
         s_log_dur_probs[j,:] =log_dur_probs[j_split,:]
         j_split += 1
      j += 1

   # Now we can perform the regular forward and backward pass + some additional calculations...
   llk_fwd, etas_c, u = forward_eta(n_j,log_o_probs.shape[1],pi,TR,s_log_dur_probs,log_o_probs)
   etas_c, gammas_c = backward_eta(n_j,log_o_probs.shape[1],TR,s_log_dur_probs,etas_c,u)

   # Now the gammas_c are log-probs. We could convert them to probs
   # via exp() but that is not going to guarantee that every columns sums
   # up to 1 because of numerical precision (or lack of). So we use a softmax
   # to ensure this.
   smoothed = None
   if compute_smoothed:
      smoothed = gammas_c - llk_fwd
      smoothed = scp.special.softmax(smoothed ,axis=1).T

   return llk_fwd,etas_c,gammas_c,smoothed

def se_step_sms_gamm(n_j,temp,cov,end_point,pi,TR,
                     log_o_probs,log_dur_probs,pds,
                     pre_lln_fn,var_map):
    # Proposes next set of latent states - see Nielsen, S. F. (2000). The stochastic EM algorithm: Estimation and asymptotic results. Bernoulli, 6(3), 457–489.
    
    # We need to pick the correct state duration distributions
    # for this particular trial - i.e., in case of a by_split we need
    # to select the distribution corresponding to the factor level on
    # this series!
    llk_fwd,_,gammas_c,_ = compute_hsmm_probabilities(n_j,cov,pi,TR,log_o_probs,
                                                    log_dur_probs,pds,var_map,False)

    # Now the gammas_c are log-probs. We could convert them to probs
    # via exp() but that is not going to guarantee that every columns sums
    # up to 1 because of numerical precision (or lack of). So we use a softmax
    # to ensure this.
    smoothed = gammas_c - llk_fwd
    if temp > 0:
        noise_blanket = scp.stats.norm.rvs(size=smoothed.shape[0]*smoothed.shape[1],scale=temp).reshape(smoothed.shape)
        smoothed = smoothed + noise_blanket
    smoothed = scp.special.softmax(smoothed ,axis=1).T

    rejected = True
    while rejected:
      
      # Propose new state sequence
      n_state_durs_est,n_state_est = prop_smoothed(n_j,log_o_probs.shape[1],smoothed)

      # Pre-check new proposal
      rejected = pre_lln_fn(n_j, end_point, n_state_durs_est, n_state_est)
    
    return n_state_durs_est,n_state_est,llk_fwd

def decode_local(n_j,cov,pi,TR,log_o_probs,log_dur_probs,pds,var_map):
    # Decoding from the smoothed probabilities - selecting j according to argmax(smoothed,axis=0)
    # See, Langrock (2021)
    llk_fwd,_,_,smoothed = compute_hsmm_probabilities(n_j,cov,pi,TR,log_o_probs,
                                                    log_dur_probs,pds,var_map,True)

    # Up until here this was just like setting up for sampling, but now we select more optimally to decode!
    states_est = np.argmax(smoothed,axis=0)

    state_dur_est = []
    c_state = states_est[0]
    c_dur = 1
    for s in range(1,len(states_est)):
      if states_est[s] != c_state:
         state_dur_est.append([c_state,c_dur])
         c_state = states_est[s]
         c_dur = 1
      c_dur += 1
    
    return np.array(state_dur_est,dtype=int), states_est, llk_fwd

##################################### sMs IR GAMM SEM functions #####################################

def prop_norm(end_point,c_state_est,sd,fix,fix_at):
    # Standard proposal for random walk MH sampler, acts as proposal for
    # left-right impulse response GAMM: We simply propose a neighbor
    # for every state onset according to an onset centered Gaussian.
    # see: https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm

    n_state_est = np.round(scp.stats.norm.rvs(loc=c_state_est,scale=sd)).astype(int)

    if fix is not None:
       for ev,at in zip(fix,fix_at):
          # fix holds event that should be fixed
          # fix_at holds the sample at which it should be fixed.
          if at == -1:
            n_state_est[ev] = end_point
          else:
            n_state_est[ev] = at

    # Collect duration of every state
    n_state_dur_est = []
    last_state_est = 0
    # State lasts from prev event (or series start) to next event
    for j in range(len(n_state_est)):
       n_state_dur_est.append([j, n_state_est[j] - last_state_est])
       last_state_est = n_state_est[j]
   
    # Last state lasts from last event to series end
    n_state_dur_est.append([len(n_state_est), end_point - n_state_est[-1]])
    
    return np.array(n_state_dur_est,dtype=int), n_state_est

def pre_ll_sms_IR_gamm(n_j, end_point, state_dur_est, state_est,fix,fix_at):
    # Prior likelihood check for left-right impulse_response sMs GAMM.
    # Checks whether likelihood should be set to - inf based
    # on any knowledge about the support of the likelihood function.

    if not fix is None and 0 in fix:
      # Cannot have state onset before 0
      if np.any(state_est < 0):
         return True
    # Cannot have state onset before 1
    elif np.any(state_est <= 0): 
       return True
    
    if not fix is None and -1 in fix_at:
      # Cannot have state onset after trial is over
      if np.any(state_est > end_point):
         return True
    # Cannot have state onset after end_point - 1
    elif np.any(state_est >= end_point):
       return True
    
    # Cannot violate left-right ordering
    if np.any(state_est[1:len(state_est)] - state_est[0:-1] < 1):
       return True
    
    return False

def init_states_IR(end_point,n_j,pre_llk_fun,fix,fix_at):
   # Number of events == number of states - 1
   n_event = n_j - 1
   # Function to initialize state estimate for sms IR GAMM.
   # Start with equally spaced states
   if n_event == 1:
      start = np.array([0])
   elif n_event == 2:
      start = np.array([0,end_point])
   else:
      start = np.array(range(0,
                                    end_point,
                                    round(end_point/n_event)))[1:(n_event - 1)]

      start = np.insert(start,0,0)
      start = np.insert(start,len(start),end_point)

   # Then make large random steps away from equally spaced states.
   prop_dur_state, prop_state = prop_norm(end_point,start,5,fix,fix_at)
   
   rejection = pre_llk_fun(n_j,end_point,prop_dur_state,prop_state,fix,fix_at)

   # Repeat until we find a suitable candidate
   while rejection:
      prop_dur_state, prop_state = prop_norm(end_point,start,5,fix,fix_at)
      
      rejection = pre_llk_fun(n_j,end_point,prop_dur_state,prop_state,fix,fix_at)

   return prop_dur_state, prop_state

def ll_sms_dc_gamm(n_j,pds,cov,var_map,state_dur_est,log_o_probs,fix):
   # Complete data likelihood function for left-right impulse response GAMM.
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

   js = np.arange(n_j)
   alpha = 0

   # Duration probability for every state
   for j in js:
      # Skip fixed states!
      if fix is None or not j in fix: 
         # We again need to pick the correct state duration distributions
         # for this particular trial. Because of the required acceptance step
         # we cannot pre-compute probabilities for specific durations for all series - since under
         # the new proposal we might have a new max duration. So we have to do that here.
         pd_j = pds[j]
         if pd_j.split_by is not None:
            # Cast to int is necessary since we index by cov and valid since factor codings are always free of decimals.
            alpha += pd_j.log_prob(state_dur_est[j,1],int(cov[0,var_map[pd_j.split_by]]))
         else:
            alpha += pd_j.log_prob(state_dur_est[j,1])

   # observation probabilities
   alpha += np.sum(log_o_probs[np.isnan(log_o_probs) == False])

   # Complete data likelihood.
   return alpha

def se_step_sms_dc_gamm(n_j,temp,series,NOT_NAs,end_point,cov,
                        state_durs,states,coef,scale,
                        log_o_probs,pds,pre_lln_fn,var_map,
                        family,terms,has_intercept,ltx,irstx,
                        stx,rtx,var_types,var_mins,
                        var_maxs,factor_levels,
                        fix,fix_at,sd,
                        n_prop,use_only):
    
    # Proposes a new candidate solution for a left-right Impulse response Gamm via a random walk step. The proposed
    # state is accepted according to a modified Metropolis Hastings acceptance ratio used for simulated annealing
    # (Kirkpatrick, Gelatt and Vecchi, 1983).
    #
    # Some thoughts:
    # The resulting proposal is not a sample from P(State | obs, current_par) as done in the classical stochastic expectation step (Nielsen, 2000).
    # In principle if we increase n_prop and leave temp==1, state_dist should however become a close representation of this distribution (assuming we
    # monitor burn in and correlation between candidates). Not sampling directly from P(State | obs, current_par) also seems to be a feature of
    # Stochastic approximate Expectation Maximization or Stochastic Simulated Annealing Expectation Maximization.
    # However, these approaches seem to form an iterative approximation of Q: Q_{t+1} = Q_{t-1} + lambda * Q_{t},
    # where Q_{t} is based on the stochastic/approximate E step, and then maximize Q_{t+1} (see Celeux, Chauveau & Diebolt, 1995; Delyon, 1999).
    # We simply maximize the CDL based on our candidate - which IS again closer to traditional SEM.
    # The sampler for the most general case (se_step_sms_gamm) behaves more like traditional SEM so the acceptance step is not necessary (might however improve convergence?).
    # For temp != 1 we really use simulated annealing and do not form a sample to approximate P(State | obs, current_par) any longer. Why? The idea
    # is that in later iterations (temp < 1) we should have a relatively good estimate so approximating the entire P(State | obs, current_par) is not that
    # important, rather we want to look more and more (the more temp < 1) at similarly likely new candidates so that we don't oscillate widely around a
    # good estimate.

    n_cand = 1 # ToDo: I considered allowing to return multiple samples, which might help with performance but I haven't implemented that yet.

    # Get complete data likelihood under current state series and parameters.
    c_llk = ll_sms_dc_gamm(n_j,pds,cov,var_map,state_durs,log_o_probs,fix)

    cutoffs = scp.stats.uniform.rvs(size=n_prop)

    state_dist = np.zeros((n_prop,n_j - 1),dtype=int)
    states_durs_dist = []
    state_llks = []
    
    # Build model matrix once for all effects that do not differ over iterations
    use_fixed = [t for t in [*ltx,*stx,*rtx] if use_only is None or t in use_only]
    use_irs = [t for t in irstx if use_only is None or t in use_only]

    model_mat_s_fix = build_sparse_matrix_from_formula(terms,has_intercept,False,
                                                        ltx,irstx,stx,rtx,var_types,var_map,
                                                        var_mins,var_maxs,factor_levels,
                                                        cov,[cov],n_j,None,[states],
                                                        use_only=use_fixed)

    model_mat_s_fix = model_mat_s_fix[NOT_NAs,]

    for i in range(n_prop):
      
      # Propose new onset
      n_state_durs,n_states = prop_norm(end_point,states,sd,fix,fix_at)

      # Pre-check new proposal
      rejection = pre_lln_fn(n_j, end_point, n_state_durs, n_states, fix, fix_at)

      # Calculate complete data likelihood under new state and current parameters
      if not rejection:
         # Start with re-building model-matrix for this series according to the proposed state.
         # Re-build only the parts that differ during sampling.
         model_mat_s = build_sparse_matrix_from_formula(terms,has_intercept,False,
                                                        ltx,irstx,stx,rtx,var_types,var_map,
                                                        var_mins,var_maxs,factor_levels,
                                                        cov,[cov],n_j,None,[n_states],
                                                        use_only=use_irs)
         
         model_mat_s = model_mat_s[NOT_NAs,]
         model_mat_s += model_mat_s_fix

         # Re-collect observation probabilities under new state sequence
         log_o_probs_n = np.zeros(len(series))

         mu = (model_mat_s @ coef).reshape(-1,1)

         if not isinstance(family,Gaussian):
            mu = family.link.fi(mu)

         if not family.twopar:
            log_o_probs_n[NOT_NAs] = np.ndarray.flatten(family.lp(series[NOT_NAs],mu))
         else:
            log_o_probs_n[NOT_NAs] = np.ndarray.flatten(family.lp(series[NOT_NAs],mu,scale))
         log_o_probs_n[NOT_NAs == False] = np.nan

         # Now finally calculate complete data likelihood under new states.
         n_llk = ll_sms_dc_gamm(n_j,pds,cov,var_map,n_state_durs,log_o_probs_n,fix)

         # Simulated Annealing/Metropolis acceptance (Kirkpatrick, Gelatt and Vecchi, 1983)
         # Kirkpatrick et al.: accept new candidate if delta_E <= 0 or otherwise with prob exp(-delta_E / T) where T = temp.
         # We want to accept new candidate if delta_llk >= 0 or otherwise with prob exp(delta_llk / T):
         # For fixed T and the case where n_llk < c_llk, delta_llk and exp(delta_llk / T) increase/approache 1
         # the closer n_llk gets to c_llk.
         # For fixed delta_llk and the case where n_llk < c_llk, as T increases > 1, exp(delta_llk / T) increases
         # (higher probability of accepting new state). As T decreases < 1, exp(delta_llk / T) decreases
         # (lower probability of accepting new state).
         delta_llk = n_llk - c_llk # delta E in Kirkpatrick et al.
         if (delta_llk >= 0) or (np.exp(delta_llk/temp) >= cutoffs[i]):
            states = n_states
            state_durs = n_state_durs
            c_llk = n_llk
      
      # Update onset distribution estimate
      state_dist[i,:] = states
      states_durs_dist.append(state_durs)
      state_llks.append(c_llk)
    
    # Sample new onset according to ~ P(onset | series, current parameters). The latter only
    # holds with n_prop = LARGE.
    sample = np.random.randint(n_prop,size=n_cand)[0]
    
    return states_durs_dist[sample],state_dist[sample,:],state_llks[sample]