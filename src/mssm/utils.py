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

##################################### Conventional Pupil basis  #####################################

def convolve_event(f,pulse_locations,i):
  # Convolution of function f with dirac delta spike centered around
  # sample pulse_locations[i].
  # Based on code by Wierda et al. 2012
  
  # Create spike
  spike = np.array([0 for _ in range(max(pulse_locations)+1)])
  spike[pulse_locations[i]] = 1
  
  # Convolve "spike" with function template f
  o = scp.signal.fftconvolve(f,spike,mode="full")
  return o


def h_basis(i,time,pulse_locations,n=10.1,t_max=930,f=1e-24):
  # Response function from Hoeks and Levelt
  # + scale parameter introduced by Wierda et al. 2012
  # Based on code by Wierda et al. 2012
  # n+1 = number of laters
  # t_max = response maximum
  # f = scaling factor
  h = f*(time**n)*np.exp(-n*time/t_max)
  
  # Convolve "spike" defined by peaks with h
  o = convolve_event(h,pulse_locations,i)
  
  # Keep only the realization of the response function
  # within the un-expanded time window
  o_restr = o[0:len(time)]
  return o_restr

##################################### B-spline functions #####################################

def tpower(x, t, p):
  # Truncated p-th power function
  # Function taken from "Splines, Knots, and Penalties" by Eilers & Marx (2010)
  return (x - t) ** (p * (x > t))

def bbase(x, knots, dx, deg):
   # Function taken from "Splines, Knots, and Penalties" by Eilers & Marx (2010)
   P = P = tpower(x[:,None],knots,deg)
   n = P.shape[1]
   D = np.diff(np.identity(n),n=deg+1) / (scp.special.gamma(deg + 1) * dx ** deg)
   B = (-1) ** (deg + 1) * P.dot(D)
   return B

def constant_basis(i,cov,state_est,convolve,max_c):
   # Can be used to add an intercept term to the model matrix
   # Keyword argument behavior is experimental...
   offset = np.ones(len(cov))

   if convolve:
    offset[0:state_est[i]] = 0
   if max_c is not None:
      offset[cov > max_c] = 0
   return offset.reshape(-1,1)

def slope_basis(i,cov,state_est,convolve,max_c):
   # Can be used to add a slope term to the model matrix
   # Keyword argument behavior is experimental...
   slope = np.copy(cov)

   if convolve:
    slope[0:state_est[i]] = 0
   if max_c is not None:
      slope[cov > max_c] = 0
   return slope.reshape(-1,1)

def B_spline_basis(i, cov, state_est, identifiable, drop_outer_k, convolve, min_c, max_c, nk, deg):
  # Setup basis with even knot locations.
  # Code based on "Splines, Knots, and Penalties" by Eilers & Marx (2010)

  # Identifiability constraint reduces dimensionality of matrix by one (Wood, 2017)
  # so we add one extra dimension to make sure that everything still matches...
  # ToDo: The identifiability behavior is odd, look into the reparameterization discussed in Wood, 2017
  if identifiable:
     nk += 1

  xl = min(cov)
  xr = max(cov)
  if not max_c is None:
     xr = max_c

  if not min_c is None:
     xl = min_c

  rg = xr - xl

  if drop_outer_k:
    ndx = (nk - deg + 2*deg)
     
  else:
    ndx = nk - deg

  dx = rg / (ndx)
  knots = np.arange(xl - deg * dx, xr + deg * dx + (0.99*dx),dx)

  B = bbase(cov,knots,dx,deg)

  if identifiable:
     B = B[:,0:-1]
     nk -= 1
     
  if drop_outer_k:
     B = B[:,deg:-deg]

  if convolve:
    o_restr = np.zeros(B.shape)

    for nki in range(nk):
      o = convolve_event(B[:,nki],state_est,i)
      o_restr[:,nki] = o[0:len(cov)]

    B = o_restr
  
  if identifiable:
     B -= np.mean(B,axis=0,keepdims=True)
  
  return B

def split_term_cov(term,cov_val,cov,cov_coding):
   rowsMat, colsMat = term.shape
   exp_term = np.zeros((rowsMat,colsMat * len(cov_coding[cov])))
   exp_term[:,cov_val*colsMat:(cov_val+1)*colsMat] = term

   return exp_term

##################################### Model-matrix setup functions #####################################
def create_event_matrix_time(time,cov,state_est, identifiable = False, drop_outer_k=False, convolve=True, min_c=0, max_c=2500, nk=10, deg=2):
  # Setup a model matrix for a left-right mssm, where every
  # state entry elicits an impluse response that affects the
  # observed signal in some way: the effect might differ between
  # states. Also estimates random intercepts for every level of cov.

  # Create intercept for series
  inter = constant_basis(None,time,state_est,convolve=False,max_c=None)

  # Create matrix for first onset because depending on the
  # basis there might be different dimensions!
  matrix_first = B_spline_basis(0,time,state_est, identifiable, drop_outer_k, convolve, min_c, max_c, nk, deg)

  rowsMat, colsMat = matrix_first.shape

  # Now that dimensions are known expand for number of
  # event locations.
  event_matrix = np.zeros((rowsMat,colsMat * len(state_est)))
  event_matrix[:,0:colsMat] = matrix_first

  # And fill with the remaining design matrix blocks.
  cIndex = colsMat
  for ci in range(1,len(state_est)):
    event_matrix[:,cIndex:cIndex+colsMat] = B_spline_basis(ci,time,state_est, identifiable, drop_outer_k, convolve, min_c, max_c, nk, deg)
    cIndex += colsMat

  #return np.concatenate((inter,event_matrix),axis=1)
  return event_matrix

def create_event_matrix_time_split(time,cov,state_est,split_k=None,rand_int=None,identifiable = False, drop_outer_k=False, convolve=True, min_c=0, max_c=2500, nk=10, deg=2,predict_all_fact = False):
  # Setup a model matrix for a left-right mssm, where every
  # state entry elicits an impluse response that affects the
  # observed signal in some way: the effect might differ between
  # states. Also estimates random intercepts for every level of cov.

  # Create intercept for series (Not in use)
  inter = constant_basis(None,time,state_est,convolve=False,max_c=None)

  # Create matrix for first onset because depending on the
  # basis there might be different dimensions!
  matrix_first = B_spline_basis(0,time,state_est, identifiable, drop_outer_k, convolve, min_c, max_c, nk, deg)

  rowsMat, colsMat = matrix_first.shape

  # Now that dimensions are known expand for number of
  # event locations.
  event_matrix = np.zeros((rowsMat,colsMat * len(state_est) + (len(split_k["split"]) * (split_k["n_levels"] - 1) * colsMat)))

  # And fill with the design matrix blocks.
  cIndex = 0
  cov_s = cov[0,split_k["cov"]]
  for j in range(len(state_est)):
    if j in split_k["split"]:
       for ci in range(split_k["n_levels"]):
          if ci == cov_s or predict_all_fact:
            event_matrix[:,cIndex:cIndex+colsMat] = B_spline_basis(j,time,state_est, identifiable, drop_outer_k, convolve, min_c, max_c, nk, deg)
          cIndex += colsMat
    else:
      event_matrix[:,cIndex:cIndex+colsMat] = B_spline_basis(j,time,state_est, identifiable, drop_outer_k, convolve, min_c, max_c, nk, deg)
      cIndex += colsMat

  if not rand_int is None:
     cov_r = cov[0,rand_int["cov"]]
     rand_matrix = np.zeros((rowsMat,rand_int["n_levels"]))
     rand_matrix[:,cov_r] = 1
     if predict_all_fact:
        rand_matrix[:,:] = 1
     event_matrix = np.concatenate((event_matrix,rand_matrix),axis=1)
     
  #return np.concatenate((inter,event_matrix),axis=1)
  return event_matrix

def create_event_matrix_cov_pupil_FS(time,cov,state_est, identifiable = False, drop_outer_k=False, convolve=True, min_c=0, max_c=1500, nk=5, deg=2, n_s=10):
   # Create event matrix based on pupil response assumed by Hoeks & Levelt
   B = np.zeros((len(time), len(state_est)))

   for ci in range(B.shape[1]):
      B[:,ci] = h_basis(ci,time,state_est)
   
   return B

def create_event_matrix_time2(time,cov,state_est, identifiable = False, drop_outer_k=False, convolve=True, min_c=0, max_c=2500, nk=10, deg=2, n_s=10):
  # Setup a model matrix for a left-right mssm, where every
  # state entry elicits an impluse response that affects the
  # observed signal in some way: the effect might differ between
  # states. Also estimates random intercepts for every level of cov.

  # Create intercept for series
  inter = constant_basis(None,time,state_est,convolve=False,max_c=None)

  # Create matrix for first onset because depending on the
  # basis there might be different dimensions!
  matrix_first = B_spline_basis(0,time,state_est, identifiable, drop_outer_k, convolve, min_c, max_c, nk, deg)

  rowsMat, colsMat = matrix_first.shape

  # Now that dimensions are known expand for number of
  # event locations.
  event_matrix = np.zeros((rowsMat,colsMat * len(state_est)))
  event_matrix[:,0:colsMat] = matrix_first

  # And fill with the remaining design matrix blocks.
  cIndex = colsMat
  for ci in range(1,len(state_est)):
    event_matrix[:,cIndex:cIndex+colsMat] = B_spline_basis(ci,time,state_est, identifiable, drop_outer_k, convolve, min_c, max_c, nk, deg)
    cIndex += colsMat
  
  # Random intercepts for every covariate level of series
  s = list(set(cov))[0]
  rand_int_s = np.zeros((rowsMat,n_s))
  rand_int_s[:,s] = 1

  return np.concatenate((inter,event_matrix,rand_int_s),axis=1)

def create_event_matrix_cov4(time,cov,state_est, identifiable = True, drop_outer_k=False, convolve=True, min_c=0, max_c=1500, nk=5, deg=2, n_s=10):
   # Create model matrix for a mssm. Here we do not assume the same
   # sigma parameter for all states - so we return the whole matrix
   # and then split it later into separate ones for every state (
   # based on the current state_est).

   # This one also has an intercept and works with two covariates
   # and random intercepts.
   
   B1 = B_spline_basis(None,cov[:,0],state_est, identifiable, drop_outer_k, convolve, min_c, max_c, nk, deg)
   B2 = B_spline_basis(None,cov[:,1],state_est, identifiable, drop_outer_k, convolve, min_c, max_c, nk, deg)
   inter = constant_basis(None,cov[:,0],state_est,convolve=False,max_c=None)

   # Define random intercept for all series - i.e., simply set the s column to 1
   s = list(set(cov[:,2]))[0]
   rand_int_s = np.zeros((B1.shape[0],n_s))
   rand_int_s[:,s] = 1

   return np.concatenate((inter,B1,B2,rand_int_s),axis=1)

def create_event_matrix_cov4_merge(time,cov,state_est, identifiable = True, drop_outer_k=False, convolve=True, min_c=0, max_c=1500, nk=5, deg=2, n_s=10):
   # This is equivalent to the create_event_matrix_cov4() function
   # but the naming highlights that it is to be used for a shared model.
   # Specifically, it is used to create the same model matrix for all states.
   # This is the opposite of what create_event_matrix_cov4_split()
   # does - which creates a block-shifted matrix to incorporate all states into
   # a single model.
   
   return create_event_matrix_cov4(time,cov,state_est, identifiable, drop_outer_k, convolve, min_c, max_c, nk, deg, n_s=10)

def create_event_matrix_cov4_split(time,cov,state_est, identifiable = True, drop_outer_k=False, convolve=True, min_c=0, max_c=1500, nk=5, deg=2, n_s=10):
   # Create model matrix for a mssm. Here we do assume the same
   # sigma parameter for all states - so instead of splitting the
   # matrix per j we block-shift the shared matrix by j

   # This one also has an intercept and works with two covariates
   # and random intercepts.
   
   B1 = B_spline_basis(None,cov[:,0],state_est, identifiable, drop_outer_k, convolve, min_c, max_c, nk, deg)
   B2 = B_spline_basis(None,cov[:,1],state_est, identifiable, drop_outer_k, convolve, min_c, max_c, nk, deg)
   inter_j = constant_basis(None,cov[:,0],state_est,convolve=False,max_c=None)

   n_j = len(set(state_est))

   inter_by = np.zeros((B1.shape[0],n_j))
   B1_by = np.zeros((B1.shape[0],B1.shape[1]*n_j))
   B2_by = np.zeros((B2.shape[0],B2.shape[1]*n_j))

   # If a single sigma parameter is assumed, then the different relationships
   # between the cov and the signal for every state can be estimated via
   # a "by-factor smooth" setup as utilized in mgcv (Wood, 2016).
   start_b1 = 0
   start_b2 = 0
   for state in range(n_j):
        B1_by[state_est==state,start_b1:(start_b1+B1.shape[1])] = B1[state_est==state,:]
        B2_by[state_est==state,start_b2:(start_b2+B2.shape[1])] = B2[state_est==state,:]
        inter_by[state_est==state,state] = inter_j[state_est==state,0]
        start_b1 += B1.shape[1]
        start_b2 += B2.shape[1]

   # First define random intercept for all series - i.e., simply set the s column to 1
   s = list(set(cov[:,2]))[0]
   rand_int_s = np.zeros((B1.shape[0],n_s))
   rand_int_s[:,s] = 1
   
   # Now block-shift per state to allow for random intercept not just by s but also by j
   rand_int_sj = np.zeros((B1.shape[0],n_s*n_j))
   start = 0
   for state in range(n_j):
        rand_int_sj[state_est==state,start:(start+rand_int_s.shape[1])] = rand_int_s[state_est==state,:]
        start += rand_int_s.shape[1]

   # Shifted intercepts in first n_j columns then shifted B matrix blocks for every j.
   # The intercepts are not penalized and can be skipped via the start_index parameter from
   # the LambdaTerm class. Then random intercepts - which are penalized.
   X_by = np.concatenate((inter_by,B1_by,B2_by,rand_int_sj),axis=1)
   return X_by

##################################### Penalty functions #####################################

def diff_pen(n,m=2,identifiable=True):
  # Creates difference (order=m) n*n penalty matrix
  # Based on code in Eilers & Marx (1996) and Wood (2017)
  if identifiable:
     n += 1
  D = np.diff(np.identity(n),m)
  if identifiable:
     D = D[0:-1,:]
  S = D @ D.T
  return S,D

def id_dist_pen(n,f=lambda x: 1):
  # Creates identity matrix penalty in case f(i) = 1
  # Can be used to create event-distance weighted penalty matrices for deconvolving sms GAMMs
  S = np.identity(n)
  for i in range(n):
    S[i,i] = f(i+1)
  return S

def embed_in_S(penalty,embS,cIndex,lam=None):
  # Embeds penalty into embS from upper left index cIndex.
  # Optionally embeds embS*lam
  # Based on Wood (2017) and Wood & Fasiolo (2016).
  pen_dim = penalty.shape[1]
  
  if lam is None:
    embS[cIndex:cIndex+pen_dim,cIndex:cIndex+pen_dim] += penalty
  else:
    embS[cIndex:cIndex+pen_dim,cIndex:cIndex+pen_dim] += penalty*lam

  cIndex += pen_dim
  return embS, cIndex

@dataclass
class LambdaTerm:
  # Lambda term storage. Can hold multiple penalties associated with a single lambda
  # value!
  # start_index can be useful in case we want to have multiple penalties on some
  # coefficients (see Wood, 2017; Wood & Fasiolo, 2017).
  penalties:np.ndarray
  lam:float = 1.1
  start_index:int = None
  frozen:bool = False

##################################### GAMM functions #####################################

def step_fellner_schall(penalties,embS,cIndex,cCoef,cLam,Inv,gInv,sigma):
  # Perform a generalized Fellner Schall update step for a lambda term. This update rule is
  # discussed in Wood & Fasiolo (2016) and used here because of it's efficiency.
  embJ = np.zeros(embS.shape)
  for pen in penalties:
    embJ, cIndex = embed_in_S(pen,embJ,cIndex)
  
  # ToDo: There is a bug somewhere in here that leads to negative
  # denom values. This should not happen based on Theorem 1 in Wood & Fasiolo (2017)
  # so I am doing something wrong. This ugly max hack for now prevents this but of course this is no
  # solution...
  num = np.trace(gInv @ embJ) - np.trace(Inv @ embJ)
  denom = cCoef.T @ embJ @ cCoef
  
  cLam = sigma * max(num / denom,1e-10) * cLam
  cLam = max(cLam,1e-10) # Prevent Lambda going to zero
  cLam = min(cLam,1e+10) # Prevent overflow

  return cLam, cIndex

def pen_lm(X,y,S):
  # Based on Wood & Fasiolo (2016)
  # ToDo: Stop solving directly for b. I literally just plugged
  # the equation from the paper in, but since we are dealing with
  # triangular matrices here, there has to be a more efficient way via
  # back/forward solving.
  # ToDo: Swapped to solve here, seems to work?
  #L, d, perm = scp.linalg.ldl(X.T @ X + S)
  #P = np.identity(L.shape[0])
  #P = P[:,perm]
  #invL = scp.linalg.lapack.dtrtri(P @ L,lower=1)[0] @ P.T
  #invd = np.linalg.inv(d)
  #Inv = invL.T @ invd @ invL
  Inv = scp.linalg.solve(X.T @ X + S, np.identity(X.shape[1]))
  b = Inv @ X.T @ y
  pred = X @ b
  return pred, b, Inv

def solve_am(X_r,y_r,lTerms,cIndex,maxiter=10):
  # Solve penalized additive (mixed) model by iteratively updating coefficient
  # weights and then lambda penalty terms.
  # Based on Wood (2017) and Wood & Fasiolo (2016)

  # Handle within trial NAs

  X = X_r[np.isnan(y_r) == False,:]
  y = y_r[np.isnan(y_r) == False]

  rowsX, colsX = X.shape
  if rowsX < colsX:
     warnings.warn("Model is not identifiable!")

  start_index = cIndex
  cCoef = np.zeros(colsX)
  H = X.T @ X

  for i in range(maxiter):
    
    # Start by building up the embedded S matrix containing
    # all individual penalties multiplied by their respective lambda
    # parameter.
    embS = np.zeros((colsX,colsX))

    for lTerm in lTerms:
      if lTerm.start_index is not None:
        cIndex = lTerm.start_index
      for pen in lTerm.penalties:
        embS, cIndex = embed_in_S(pen,embS,cIndex,lam=lTerm.lam)
    cIndex = start_index
    #print(embS)
    #print(embS)

    # Solve for coefficients an inverse of X^T*X+S
    pred, cCoef, Inv = pen_lm(X,y,embS)

    # Calculate residuals
    res = y - pred
    resDot = res.dot(res)

    # Generalized inverse of S
    gInv = np.linalg.pinv(embS)

    # Sigma estimate from Wood & Fasiolo (2016)
    
    sigma = resDot / (rowsX - np.trace(Inv @ H))

    # Now perform the Fellner Schall update discussed by Wood & Fasiolo (2016)
    # Essentially we get new lambda values, which in the next iteration then
    # are used to update the coefficients.
    for lTerm in lTerms:
      if lTerm.start_index is not None:
        cIndex = lTerm.start_index
      cLam, cIndex = step_fellner_schall(lTerm.penalties,embS,cIndex,cCoef,lTerm.lam,Inv,gInv,sigma)
      if cLam < 0:
        warnings.warn(f"Resetting Lambda Reason:\nPrevious Lambda = {lTerm.lam}, Next Lambda = {cLam}, Sigma = {sigma}")
        cLam = 1.1
      lTerm.lam = cLam
    cIndex = start_index
    #print([lTerm.lam for lTerm in lTerms])
  
  return cCoef, lTerms, sigma, embS

def compute_gcv(y,X,coef,lTerms):
   # Follows Wood (2017), computes the generalized
   # cross-validation error for a GAMM.
   pred = X @ coef
   res = y - pred
   resDot = res.dot(res)
   n,m = X.shape

   embS = np.zeros((m,m))

   cIndex = 0
   for lTerm in lTerms:
      if lTerm.start_index is not None:
         cIndex = lTerm.start_index
      for pen in lTerm.penalties:
         embS, cIndex = embed_in_S(pen,embS,cIndex,lam=lTerm.lam)
      
   Inv = scp.linalg.solve(X.T @ X + embS, np.identity(m))

   A = X @ Inv @ X.T
   gcv = (n * resDot) / ((n - np.trace(A))**2)
   return gcv

@dataclass
class GAMMResults:
   lambdaTerms: list = None
   sigma: list = None
   mat: np.ndarray = None
   S: np.ndarray = None
   Inv: np.ndarray = None
   coef: np.ndarray = None


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

def ll_sms_gamm(n_j,pi,TR,state_dur_est,state_est,ps,logprobs,cov,split_p_by_cov):
   # Complete data likelihood function for sMs GAMM WITH state re-entries, so not left-right.
   # Again, based on equation 15 in Yu (2011) but adapted for the complete likelihood
   # case, since we have a state_est available. We are again interested
   # in the last_time_point alpha of the state we predicted to end there (i.e., the last
   # candidate state). We simply initialize alpha based on the current estimates of pi (
   # initial state probabilities) given our first state and it's duration.
   # For every transition according to the candidate state sequence we then evaluate the
   # probability of the transition and the subsequent stay in that state via the current
   # transition probability matrix (TR) estimate.

   # In the end we again add the log-probabilities of observing y_t at any t - since the
   # GAMM setup ensures that the predicted y_hat_t at time-point t is based on our current
   # estimated state for that t (see alignment code in se_step_sms_gamm).

   # Log-probs can include NAs, so we need to handle those.
   # Again we can just exclude them here.
   c_logprobs = logprobs[np.isnan(logprobs) == False]

   with np.errstate(divide="ignore"):
    # See: https://stackoverflow.com/questions/21752989/
    log_pi = np.log(pi)
    log_TR = np.log(TR)

   t = 0
   alpha = log_pi[state_dur_est[0,0]] + ps[state_dur_est[0,0]].logpdf(state_dur_est[0,1])
   t += state_dur_est[0,1]

   for tr in range(1,state_dur_est.shape[0]):
      alpha += (log_TR[state_dur_est[tr-1,0], state_dur_est[tr,0]] +
                ps[state_dur_est[tr,0]].logpdf(state_dur_est[tr,1]))
      t += state_dur_est[0,1]

   alpha += np.sum(c_logprobs)
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

def pre_ll_sms_gamm(n_j, end_point, state_dur_est, state_est):
   # Prior likelihood check for sMs GAMM WITH state re-entries.
  
   # Cannot marginalize out one or more states - model matrix will become
   # unidentifiable.
   js = set(state_est)
   #print(js)
   if len(js) < n_j:
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

def prop_smoothed(n_j,c_state_est,smoothed):
   # Simple proposal for sMs-GAMM with state re-entries, based on
   # the smoothed probabilities P(State_t == j|parameters).
   # We simply select a state at every time-point t based on those
   # probabilities.
   # This is different from previous MCMC sampling approaches in the HsMM
   # literature (see Guedon, 2003; Guedon, 2005; Guedon, 2007) for alternative approaches
   # but seems to work quite well. 
   
   n_state_est = np.copy(c_state_est)
   for idx in range(len(n_state_est)):
    n_state_est[idx] = np.random.choice(list(range(n_j)),p=smoothed[:,idx])
   
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

def se_step_sms_gamm(n_j,temp,series,end_point,time,cov,pi,TR,
                     state_durs_est,state_est,ps,coef,sigma,
                     create_matrix_fn, pre_lln_fn, ll_fn,e_bs_fun,
                     repeat_by_j,split_p_by_cov,build_mat_kwargs,
                     e_bs_kwargs,merge_mat_fun=None):

    n_prop = 1 
    n_cand = 1 
    
    # Since we usually care only about a small number of states, we
    # can neglect their impact on the forward and backward time complexity.
    # However, even ignoring this both still take at least number_of_obs*
    # maximum_duration steps (for n_j=1). Now in principle, a state could take as long as the
    # series lasts. But that would lead to an n_t*n_t complexity, which is
    # just not feasible. One way to constrain this is to just consider the
    # most likely durations under the current parameter set. We use the quantile
    # function to determine the highest 99% cut-off (only 1% of duration values are more
    # extreme than this one), across states which we then set as the max_dur to be considered.
    max_dur = int(round(max([p.ppf(q=0.99) for p in ps])))
    
    c_state_est = np.copy(state_est)
    c_state_durs_est = np.copy(state_durs_est)
    
    # For forward and backward probabilities we need for every time
    # point the probability of observing the series at that time-point
    # according to the GAMM from EVERY state.
    # For separate models (repeat_by_j) this is easy since we have the
    # same model matrix for every state. For shared models we have a
    # block-shifted shared model matrix. However we need the same matrix
    # for every state to get the aforementioned probabilities. So for those
    # models an extra function is called here.
    if repeat_by_j == False:
      c_mat = merge_mat_fun(time,cov,c_state_est,**build_mat_kwargs)
    else:
      c_mat = create_matrix_fn(time,cov,c_state_est,**build_mat_kwargs)

    c_logprobs = e_bs_fun(n_j,series,c_mat,coef,sigma,repeat_by_j,**e_bs_kwargs)

    # In both cases we then need to align the observation probabilities
    # for the complete data likelihood function.
    aligned_logprobs = np.copy(c_logprobs[0,:])

    for j in range(1,n_j):
      aligned_logprobs[c_state_est==j] = c_logprobs[j,c_state_est==j]
    
    c_llk = ll_fn(n_j,pi,TR,c_state_durs_est,c_state_est,ps,aligned_logprobs,cov,split_p_by_cov)
    
    cutoffs = scp.stats.uniform.rvs(size=n_prop)

    state_dist = np.zeros((n_prop,len(time)),dtype=int)
    states_durs_dist = []
    acc = 0

    # ... we also need the probability of every possible duration for every state.
    log_dur_mat = get_log_dur_prob_mat(n_j,max_dur,ps,cov)
    
    # Now we can perform the regular forward and backward pass + some additional calculations...
    llk_fwd, etas_c, u = forward_eta(n_j,series.shape[0],pi,TR,log_dur_mat,c_logprobs)
    etas_c, gammas_c = backward_eta(n_j,series.shape[0],TR,log_dur_mat,etas_c,u)

    # Now the gammas_c are un-normalized log-probs. So we hack them
    # to normalized over(j) probabilities so that we can sample from them.
    smoothed = scp.special.softmax(gammas_c - llk_fwd,axis=1).T

    for i in range(n_prop):
      
      # Propose new state sequence
      n_state_durs_est,n_state_est = prop_smoothed(n_j,c_state_est,smoothed)

      # Pre-check new proposal
      rejection = pre_lln_fn(n_j, end_point, n_state_durs_est, n_state_est)

      if not rejection:
        
        # Calculate likelihood of proposal given observations and current parameters
        if repeat_by_j == False:
          n_mat = merge_mat_fun(time,cov,c_state_est,**build_mat_kwargs)
        else:
          n_mat = create_matrix_fn(time,cov,c_state_est,**build_mat_kwargs)
        
        n_logprobs = e_bs_fun(n_j,series,n_mat,coef,sigma,repeat_by_j,**e_bs_kwargs)

        aligned_logprobs = np.copy(n_logprobs[0,:])

        for j in range(1,n_j):
            aligned_logprobs[n_state_est==j] = n_logprobs[j,n_state_est==j]

        n_llk = ll_fn(n_j,pi,TR,n_state_durs_est,n_state_est,ps,aligned_logprobs,cov,split_p_by_cov)

        # Simulated Annealing/Metropolis acceptance
        if np.exp((n_llk - c_llk)/temp) >= cutoffs[i]:
          c_state_est = np.copy(n_state_est)
          c_state_durs_est = np.copy(n_state_durs_est)
          c_mat = np.copy(n_mat)
          c_llk = n_llk
          acc += 1
      
      # Update state distribution estimate
      state_dist[i,:] = np.copy(c_state_est)
      states_durs_dist.append(np.copy(c_state_durs_est))
    
    # Sample new state sequence from all collected candidates
    sample = np.random.randint(n_prop,size=n_cand)[0]
    
    return states_durs_dist[sample],state_dist[sample,:]

##################################### Temperature schedulers #####################################

def anneal_temps(iter,b=0.005):
   # Annealing schedule as proposed by Kirkpatrick, Gelatt and Vecchi (1983).
   # However, by using this schedule we never converge to a steepest ascent algorithm
   # since temp -> 1 in the limit instead of temp -> 0
   ts = np.array(list(range(iter)))

   temp = np.array([1 + 1/(b*math.sqrt(t+1)) for t in ts])
   return temp

def anneal_temps_zero(iter,b=0.005):
   # Annealing schedule as proposed by Kirkpatrick, Gelatt and Vecchi (1983).
   ts = np.array(list(range(iter)))

   temp = np.array([1/(b*math.sqrt(t+1)) for t in ts])
   return temp

def const_temp(iter,a=1.0):
   # Temperature remains constant throughout all
   # iterations. Not really ever useful, except for the se_step_sms_gamm()
   # step, to show that the acceptance is not necessary...
   temp = np.array([a for _ in range(iter)])
   return temp

##################################### Parameterization functions #####################################

def par_gamma2s(scales):
   # We use GAMMA2 distributions to model sojourn times. This is based on work
   # by Anderson et al., (2016) but absolutely not required. Can be replaced by any
   # pdf or pmf for which an m-step can be implemented as well.
   return [scp.stats.gamma(a=2, scale=scales[j]) for j in range(len(scales))]

##################################### HsMM functions #####################################

def forward_eta(n_j,n_t,pi,TR,log_dur_mat,log_obs_mat):
   # Forward pass for HsMM as described in Yu (2011). Our
   # sampler requires the probabilities P(S_t = j| obs, pars).
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
   # Backward pass for HsMM as described in Yu (2011). Gets the partially
   # computed etas from the forward_eta() function. Since we have already
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

def m_gamma2s_sms_gamm(n_j,end_points,p_pars,state_dur_est,state_est,cov,split_p_by_cov):
   # M step for gammas for sMs Gamm with state re-entries. A bit messy
   # but all it does is to again calculate per state the average duration
   # over all time points and also series.
   durs = [[] for j in range(n_j)]

   for s in state_dur_est:
    s_durs = [s[s[:,0] == j,1] for j in range(n_j)]
    for j in range(n_j):
      durs[j].extend(s_durs[j])
   
   scales = [np.mean(ds)/2 for ds in durs]
   return scales

def m_pi_sms_gamm(n_j,pi,state_dur_est,cov):
   # For given state sequences optimal estimate for pi
   # is the percentage of being in state j at time 0 over
   # all series. Follows direclty from sub-equation 1 in Yu (2011)
   # for complete data case.
   n_pi = np.zeros(n_j)
   n_s = len(state_dur_est)
   for s in range(n_s):
      n_pi[state_dur_est[s][0,0]] += 1
   n_pi /= n_s
   return n_pi
   
def m_TR_sms_gamm(n_j,TR,state_dur_est,cov):
   # For given state sequences, optimal estimate
   # for the transition probability from i -> j
   # is: n(i -> j)/n(i -> any not i)
   # where n() gives the number of occurences across
   # all sequences. Follows direclty from sub-equation 2 in Yu (2011)
   # for complete data case.
   n_TR = np.zeros(TR.shape)
   n_s = len(state_dur_est)
   for s in range(n_s):
    for tr in range(1,state_dur_est[s].shape[0]):
        n_TR[state_dur_est[s][tr-1,0],state_dur_est[s][tr,0]] += 1
  
   for j in range(n_j):
    n_TR[j,:] /= np.sum(n_TR[j,:])

   return n_TR

##################################### Other #####################################

def map_unwrap_args_kwargs(fn, args, kwargs):
    # Taken from isarandi's answer to: https://stackoverflow.com/questions/45718523/
    # Basis for entire MP code used in this project.
    return fn(*args, **kwargs)

def j_split(n_j,state_est,y,X):
   # Split flattened state sequence, flattend Y fector, and shared matrix X into j
   # parts, one for every state.
   y_split = []
   X_split = []

   for j in range(n_j):
      y_split.append(y[state_est == j])
      X_split.append(X[state_est == j,:])

   return y_split, X_split

def get_log_o_prob_mat(n_j,series,X,coef,sigma,repeat_by_j,mask_by_j=None):
   # Function to get the probability of every value in the observed
   # series under the parameterized GAMM (or under the different GAMMs)
   n_k = X.shape[1]
   n_t = X.shape[0]
   NAs = np.isnan(series)

   if repeat_by_j:
    # Separate models per j + separate sigmas
    # If this is True, then X should be of dim n_k but coef of dim n_j*n_k
    #
    # ToDo: This (and the mask_by_j case) assumes the same number of coefficients per n_j. In principle this
    # makes sense, since the point of sms GAMMs is to assume that the shape of
    # the relationships depends on states. But it could in principle also 
    # be desirable to allow for different relationships (i.e., with different variables)
    # depending on states..
    log_probs = np.zeros((n_j,n_t))
    idx = 0
    for j in range(n_j):
      log_probs[j,NAs==False] = scp.stats.norm.logpdf(series[NAs==False],
                                                      loc=X[NAs==False,:] @ coef[idx:idx+n_k],
                                                      scale=math.sqrt(sigma[j]))
      log_probs[j,NAs==True] == np.nan
      idx += n_k

   elif mask_by_j is not None and repeat_by_j == False:
      # Separate models per j but shared sigma.
      log_probs = np.zeros((n_j,n_t))
      for j in range(n_j):
        log_probs[j,NAs==False] = scp.stats.norm.logpdf(series[NAs==False],
                                                        loc=X[NAs==False,:] @ coef[mask_by_j == j],
                                                        scale=math.sqrt(sigma))
        log_probs[j,NAs==True] == np.nan

   else:
    # The de-convolving case + separate models per j but shared sigma in full matrix form.
    log_probs = np.zeros(n_t)
    log_probs[NAs==False] = scp.stats.norm.logpdf(series[NAs==False],
                                                  loc=X[NAs==False,:] @ coef,
                                                  scale=math.sqrt(sigma))
    log_probs[NAs==True] == np.nan

   return log_probs

def get_log_dur_prob_mat(n_j,max_dur,ps,cov):
    # Build a n_j*max_dur matrix, containing in every cell (j,d)
    # the probability of state j lasting for duration d according to
    # state j's sojourn time distribution.
    durs = np.arange(max_dur)
    js = np.arange(n_j)
    # Now we collect all density values i.e., the probabilities for all durations per state
    dur_log_probs = np.zeros((n_j,durs.shape[0]))

    for j in js:
        dur_log_probs[j,:] = ps[j].logpdf(durs + 1)
    
    return dur_log_probs

class TermType(Enum):
    LSMOOTH = 1
    SMOOTH = 2
    LINEAR = 3
    RANDINT = 4
    RANDSLOPE = 5

class VarType(Enum):
    NUMERIC = 1
    FACTOR = 2

class PenType(Enum):
    IDENTITY = 1
    DIFFERENCE = 2
    DISTANCE = 3

class GammTerm():
   
   def __init__(self,variables:list[str],
                type:TermType,
                is_penalized:bool,
                penalty:list,
                pen_args:list) -> None:
        
        self.variables = variables
        self.type = type
        self.penalized = is_penalized
        self.penalty = penalty
        self.pen_args = pen_args

class i(GammTerm):
    def __init__(self) -> None:
        super().__init__(["1"], TermType.LINEAR, False, [], [])

class f(GammTerm):
    def __init__(self,variables:list,
                by:str=None,
                nk:int=10,
                is_penalized:bool = True,
                penalty:list = [PenType.DIFFERENCE],
                pen_args:list = [{"m":2}]) -> None:
        
        # Initialization
        super().__init__(variables, TermType.SMOOTH, is_penalized, penalty, pen_args)
        self.by = by
        self.nk = nk

class fl(GammTerm):
    def __init__(self,variable:str,
                stage:str,
                by:str=None,
                nk:int=10,
                is_penalized:bool = True,
                penalty:list = [PenType.DIFFERENCE],
                pen_args:list = [{"m":2}]) -> None:
        
        # Initialization
        super().__init__([variable], TermType.LSMOOTH, is_penalized, penalty, pen_args)
        self.stage = stage
        self.by = by
        self.nk = nk

class l(GammTerm):
    def __init__(self,variables:list,
                is_penalized:bool = False,
                penalty:list = [],
                pen_args:list = []) -> None:
        
        # Initialization
        super().__init__(variables, TermType.LINEAR, is_penalized, penalty, pen_args)

class ri(GammTerm):
    def __init__(self,variable:str,
                is_penalized:bool = True,
                penalty:list = [PenType.IDENTITY],
                pen_args:list = [None]) -> None:
        
        # Initialization
        super().__init__([variable], TermType.RANDINT, is_penalized, penalty, pen_args)

class rs(GammTerm):
    def __init__(self,variable:str,
                by:str,
                is_penalized:bool = True,
                penalty:list = [PenType.IDENTITY],
                pen_args:list = [None]) -> None:
        
        # Initialization
        super().__init__([variable], TermType.RANDSLOPE, is_penalized, penalty, pen_args)
        self.by = by

class lhs():
    def __init__(self,variable:str,f:Callable=None) -> None:
        self.variable = variable
        self.f=f

class Formula():
    def __init__(self,lhs:lhs,terms:list[GammTerm],data:pd.DataFrame) -> None:
        self.lhs = lhs
        self.terms = terms
        self.data = data
        self.__factor_codings = {}
        self.__coding_factors = {}
        self.__factor_levels = {}
        self.__var_to_cov = {}
        self.__var_types = {}
        self.__var_mins = {}
        self.__var_maxs = {}
        self.__linear_terms = []
        self.__smooth_terms = []
        self.__random_terms = []
        self.__has_intercept = False
        cvi = 0

        # Perform input checks
        if self.lhs.variable not in self.data.columns:
            raise IndexError(f"Column '{self.lhs.variable}' does not exist in Dataframe.")

        for ti, term in enumerate(self.terms):
            
            if isinstance(term,i):
                self.__has_intercept = True
                self.__linear_terms.append(ti)
                continue
            
            # All variables must exist in data
            for var in term.variables:

                if not var in data.columns:
                    raise KeyError(f"Variable '{var}' of term {ti} does not exist in dataframe.")
                
                vartype = data[var].dtype

                # Store information for all variables once.
                if not var in self.__var_to_cov:
                    self.__var_to_cov[var] = cvi

                    # Assign vartype enum and calculate mins/maxs for continuous variables
                    if vartype in ['float64','int64']:
                        # ToDo: these can be properties of the formula.
                        self.__var_types[var] = VarType.NUMERIC
                        self.__var_mins[var] = np.min(data[var])
                        self.__var_maxs[var] = np.max(data[var])
                    else:
                        self.__var_types[var] = VarType.FACTOR
                        self.__var_mins[var] = None
                        self.__var_maxs[var] = None

                        # Code factor variables into integers for example for easy dummy coding
                        levels = np.unique(data[var])

                        self.__factor_codings[var] = {}
                        self.__coding_factors[var] = {}
                        self.__factor_levels[var] = levels
                        
                        for ci,c in enumerate(levels):
                           self.__factor_codings[var][c] = ci
                           self.__coding_factors[var][ci] = c

                    cvi += 1
                
                # Smooth-term variables must all be continuous
                if isinstance(term, f) or isinstance(term, fl):
                    if not vartype in ['float64','int64']:
                        raise TypeError(f"Variable '{var}' attributed to smooth/latent smooth term {ti} must be numeric and is not.")
                    
                # Random intercept variable must be categorical
                if isinstance(term, ri):
                    if vartype in ['float64','int64']:
                        raise TypeError(f"Variable '{var}' attributed to random intercept term {ti} must not be numeric but is.")
                
            # by-variables must be categorical
            if isinstance(term, f) or isinstance(term, fl) or isinstance(term, rs):
                if not term.by is None:
                    if not term.by in data.columns:
                        raise KeyError(f"By-variable '{term.by}' attributed to term {ti} does not exist in dataframe.")
                    
                    if data[term.by].dtype in ['float64','int64']:
                        raise KeyError(f"Data-type of By-variable '{term.by}' attributed to term {ti} must not be numeric but is. E.g., Make sure the pandas dtype is 'object'.")
            
            # Remaining term allocation.
            if isinstance(term, f) or isinstance(term, fl):
               self.__smooth_terms.append(ti)

            if isinstance(term, ri) or isinstance(term,rs):
               self.__random_terms.append(ti)

    def get_factor_codings(self) -> dict:
        return copy.deepcopy(self.__factor_codings)
    
    def get_coding_factors(self) -> dict:
        return copy.deepcopy(self.__coding_factors)
    
    def get_var_map(self) -> dict:
        return copy.deepcopy(self.__var_to_cov)
    
    def get_factor_levels(self) -> dict:
       return copy.deepycopy(self.__factor_levels)
    
    def get_var_types(self) -> dict:
       return copy.deepcopy(self.__var_types)
    
    def get_var_mins_maxs(self) -> (dict,dict):
       return (copy.deepcopy(self.__var_mins),copy.deepcopy(self.__var_maxs))
    
    def get_linear_term_idx(self) -> list[int]:
       return(copy.deepcopy(self.__linear_terms))
    
    def get_smooth_term_idx(self) -> list[int]:
       return(copy.deepcopy(self.__smooth_terms))
    
    def get_random_term_idx(self) -> list[int]:
       return(copy.deepcopy(self.__random_terms))
    
    def compute_n_coef(self) -> None:
       n_coef = 0
       if self.__has_intercept:
          n_coef += 1

       for lti in self.__linear_terms:
          ti = self.terms[lti]

          if isinstance(ti,i):
             continue
          
          t_coef = []
          for var in ti.variables:
             # Assuming dummy coding: ...
             # Assuming no dummy coding: ...
             continue

def build_mat_for_series(formula,data,series_id:str):
   id_col = np.array(data[series_id])
   var_map = formula.get_var_map()
   n_var = len(var_map)
   var_keys = var_map.keys()
   var_types = formula.get_var_types()
   factor_coding = formula.get_factor_codings()
   
   # Collect every series from data frame, make sure to maintain the
   # order of the data frame.
   # Based on: https://stackoverflow.com/questions/12926898
   _, id = np.unique(id_col,return_index=True)
   sid = np.sort(id)

   # Collect entire y column
   y_flat = np.array(data[formula.lhs.variable]).reshape(-1,1)
   n_y = y_flat.shape[0]

   # Then split by seried id
   y = np.split(y_flat,sid[1:])

   # Now all predictor variables
   cov_flat = np.zeros((n_y,n_var),dtype=float) # Treating all covariates as floats has important implications for factors and requires special care!

   for c in var_keys:
      c_raw = np.array(data[c])

      if var_types[c] == VarType.FACTOR:
         c_code = np.zeros(n_y,dtype=float)

         for code_key in factor_coding[c].keys():
            c_code[c_raw == code_key] = factor_coding[c][code_key]

         cov_flat[:,var_map[c]] = c_code

      else:
         cov_flat[:,var_map[c]] = c_raw
   
   # Now split cov by series id as well
   cov = np.split(cov_flat,sid[1:],axis=0)

   return y_flat,cov_flat,y,cov

def build_series_matrix_from_formula(formula,cov,state_est):
   pass