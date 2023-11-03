import numpy as np
import scipy as scp
import math

##################################### Temperature functions #####################################

def anneal_temps_zero(iter,b=4):
   # Annealing schedule as proposed by Kirkpatrick, Gelatt and Vecchi (1983).
   ts = np.array(list(range(iter)))

   temp = np.array([1/(b*math.sqrt(t+1)) for t in ts])
   return temp

##################################### HsMM functions #####################################

def forward_eta(n_j,n_t,pi,TR,log_dur_mat,log_obs_mat):
   # Forward pass for HsMM based on math in Yu (2011) and inspired by
   # implementation in edhsmm (https://github.com/poypoyan/edhsmm).

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
            betas[t-1,j] = np.logaddexp(betas[t-1,j],log_TR[j,i] + beta_stars_t[i])

            # Now we can update etas as well - eq 6!
         etas[t-1,j,:] +=  betas[t-1,j]

        # Finally, we can calculate gammas, according to eq 8.
        for tau_add in range(min(n_d,n_t - t)):
          tau = t + tau_add
          for d in range(tau_add,n_d): # tau_add is difference between tau and t, so the term below the second sum in eq 8
              gammas[t,j] = np.logaddexp(gammas[t,j], etas[tau,j,d])
          
   return etas, gammas

def prop_smoothed(n_j,n_t,smoothed):
   # Simple proposal for sMs-GAMM with state re-entries, based on
   # the smoothed probabilities P(State_t == j|parameters).
   # We simply select a state at every time-point t based on those
   # probabilities.
   # This is different from previous MCMC sampling approaches in the HsMM
   # literature (see Guedon, 2003; Guedon, 2005; Guedon, 2007) for alternative approaches
   # but seems to work quite well. 
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

def se_step_sms_gamm(n_j,temp,series,cov,end_point,pi,TR,
                     log_o_probs,log_dur_probs,pds,
                     pre_lln_fn,var_map):
    # Proposes next set of latent states - see Nielsen (2002).
    
    # We need to pick the correct state duration distributions
    # for this particular trial - i.e., in case of a by_split we need
    # to select the distribution corresponding to the factor level on
    # this series!
    s_log_dur_probs = np.zeros((n_j,log_dur_probs.shape[1]))
    j = 0
    while j < n_j:
       pd_j = pds[j]
       if pd_j.split_by is not None:
          s_log_dur_probs[j,:] = log_dur_probs[j+cov[0,var_map[pd_j.split_by]],:]
          j += cov[0,var_map[pd_j.split_by]] + 1
       else:
          s_log_dur_probs[j,:] =log_dur_probs[j,:]
          j += 1
    
    # Now we can perform the regular forward and backward pass + some additional calculations...
    llk_fwd, etas_c, u = forward_eta(n_j,series.shape[0],pi,TR,s_log_dur_probs,log_o_probs)
    etas_c, gammas_c = backward_eta(n_j,series.shape[0],TR,s_log_dur_probs,etas_c,u)

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
      n_state_durs_est,n_state_est = prop_smoothed(n_j,series.shape[0],smoothed)

      # Pre-check new proposal
      rejected = pre_lln_fn(n_j, end_point, n_state_durs_est, n_state_est)
    
    return n_state_durs_est,n_state_est,llk_fwd