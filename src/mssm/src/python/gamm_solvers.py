import numpy as np
import scipy as scp
import warnings
from .exp_fam import Family,Gaussian,est_scale
from .penalties import PenType,id_dist_pen,translate_sparse
from .formula import build_sparse_matrix_from_formula,setup_cache,clear_cache,cpp_solvers,pd,Formula,mp,repeat,os,map_csc_to_eigen
from tqdm import tqdm
from functools import reduce

CACHE_DIR = './.db'
SHOULD_CACHE = False
MP_SPLIT_SIZE=1000

def cpp_chol(A):
   return cpp_solvers.chol(*map_csc_to_eigen(A))

def cpp_cholP(A):
   return cpp_solvers.cholP(*map_csc_to_eigen(A))

def cpp_qr(A):
   return cpp_solvers.pqr(*map_csc_to_eigen(A))

def cpp_solve_am(y,X,S):
   return cpp_solvers.solve_am(y,*map_csc_to_eigen(X),*map_csc_to_eigen(S))

def cpp_solve_coef(y,X,S):
   return cpp_solvers.solve_coef(y,*map_csc_to_eigen(X),*map_csc_to_eigen(S))

def cpp_solve_coefXX(Xy,XXS):
   return cpp_solvers.solve_coefXX(Xy,*map_csc_to_eigen(XXS))

def cpp_solve_L(X,S):
   return cpp_solvers.solve_L(*map_csc_to_eigen(X),*map_csc_to_eigen(S))

def cpp_solve_LXX(XXS):
   return cpp_solvers.solve_L(*map_csc_to_eigen(XXS))

def cpp_solve_tr(A,C):
   return cpp_solvers.solve_tr(*map_csc_to_eigen(A),C)

def cpp_backsolve_tr(A,C):
   return cpp_solvers.backsolve_tr(*map_csc_to_eigen(A),C)

def compute_lgdetD_bsb(rank,cLam,gInv,emb_SJ,cCoef):
   # Derivative of log(|S_lambda|+), the log of the "Generalized determinant", with respect to lambda see Wood, Shaddick, & Augustin, (2017)
   # and Wood & Fasiolo (2016), and Wood (2017), and Wood (2020)
   if not rank is None:
      # (gInv @ emb_SJ).trace() should be equal to rank(S_J)/cLam for single penalty terms (Wood, 2020)
      lgdet_deriv = rank/cLam
   else:
      lgdet_deriv = (gInv @ emb_SJ).trace()

   # Derivative of log(|XX + S_lambda|) is computed elsewhere, but we need the remaining part from the LLK (Wood & Fasiolo, 2016):
   bSb = cCoef.T @ emb_SJ @ cCoef
   return lgdet_deriv,bSb

def step_fellner_schall_sparse(lgdet_deriv,ldet_deriv,bSb,cLam,scale):
  # Compute a generalized Fellner Schall update step for a lambda term. This update rule is
  # discussed in Wood & Fasiolo (2016) and used here because of it's efficiency.
  
  num = max(0,lgdet_deriv - ldet_deriv)

  denom = max(0,bSb)

  # Especially when using Null-penalties denom can realisitically become
  # equal to zero: every coefficient of a term is penalized away. In that
  # case num /denom is not defined so we set nLam to nLam_max.
  if denom <= 0: # Prevent overflow
     nLam = 1e+7
  else:
     nLam = scale * (num / denom) * cLam

     nLam = max(nLam,1e-7) # Prevent Lambda going to zero
     nLam = min(nLam,1e+7) # Prevent overflow

  # Compute lambda delta
  delta_lam = nLam-cLam

  return delta_lam

def grad_lambda(lgdet_deriv,ldet_deriv,bSb,scale):
   # P. Deriv of restricted likelihood with respect to lambda.
   # From Wood & Fasiolo (2016)
   return lgdet_deriv/2 - ldet_deriv/2 - bSb / (2*scale)

def compute_S_emb_pinv_det(col_S,penalties,pinv):
   # Computes final S multiplied with lambda
   # and the pseudo-inverse of this term.
   S_emb = None

   # We need to compute the pseudo-inverse on the penalty block (so sum of all
   # penalties weighted by lambda) for every term so we first collect and sum
   # all term penalties together.
   SJs = [] # Summed SJs for every term
   SJ_terms = [] # How many penalties were summed
   SJ_types = [] # None if summation otherwise penalty type
   SJ_lams = [] # None if summation otherwise penalty lambda
   SJ_reps = [] # How often should pinv(SJ) be stacked (for terms with id)
   SJ_idx = [] # Starting index of every SJ
   SJ_idx_max = 0 # Max starting index - used to decide whether a new SJ should be added.
   SJ_idx_len = 0 # Number of separate SJ blocks.

   # Build S_emb and collect every block for pinv(S_emb)
   for lti,lTerm in enumerate(penalties):
      #print(f"pen: {lti}")

      # Add S_J_emb * lambda to the overall S_emb
      if lti == 0:
         S_emb = lTerm.S_J_emb * lTerm.lam
      else:
         S_emb += lTerm.S_J_emb * lTerm.lam

      # Now collect the S_J for the pinv calculation in the next step.
      if lti == 0 or lTerm.start_index > SJ_idx_max:
         SJs.append(lTerm.S_J*lTerm.lam)
         SJ_terms.append(1)
         SJ_types.append(lTerm.type)
         SJ_lams.append(lTerm.lam)
         SJ_reps.append(lTerm.rep_sj)
         SJ_idx.append(lTerm.start_index)
         SJ_idx_max = lTerm.start_index
         SJ_idx_len += 1

      else: # A term with the same starting index exists already - so sum the SJs
         idx_match = [idx for idx in range(SJ_idx_len) if SJ_idx[idx] == lTerm.start_index]
         #print(idx_match,lTerm.start_index)
         if len(idx_match) > 1:
            raise ValueError("Penalty index matches multiple previous locations.")
         SJs[idx_match[0]] += lTerm.S_J*lTerm.lam
         SJ_terms[idx_match[0]] += 1
         SJ_types[idx_match[0]] = None
         SJ_lams[idx_match[0]] = None
         if SJ_reps[idx_match[0]] != lTerm.rep_sj:
            raise ValueError("Repeat Number for penalty does not match previous penalties repeat number.")

   #print(SJ_idx_len)
   #print(SJ_reps,SJ_lams,SJ_idx)
   S_pinv_elements = []
   S_pinv_rows = []
   S_pinv_cols = []
   cIndexPinv = SJ_idx[0]

   FS_use_rank = []
   for SJi in range(SJ_idx_len):
      # Now handle all pinv calculations because all penalties
      # associated with a term have been collected in SJ

      if SJ_terms[SJi] == 1:
         cIndexPinv += (SJs[SJi].shape[1]*SJ_reps[SJi])
         FS_use_rank.append(True)

      else:
         # Compute pinv(SJ) via cholesky factor L so that L @ L' = SJ' @ SJ.
         # If SJ is full rank, then pinv(SJ) = inv(L)' @ inv(L) @ SJ'.
         # However, SJ' @ SJ will usually be rank deficient so we add e=1e-7 to the main diagonal, because:
         # As e -> 0 we get closer to pinv(SJ) if we base it on L_e in L_e @ L_e' = SJ' @ SJ + e*I
         # where I is the identity.
         if pinv != "svd":
            t = scp.sparse.identity(SJs[SJi].shape[1],format="csc")
            L,code = cpp_chol(SJs[SJi].T @ SJs[SJi] + 1e-10*t)

            if code != 0:
               warnings.warn(f"Cholesky factor computation failed with code {code}")

            LI = scp.sparse.linalg.spsolve(L,t)
            SJ_pinv = LI.T @ LI @ SJs[SJi].T
            #print(np.max(abs((SJ.T @ SJ @ SJ_pinv - SJ).toarray())))

         # Compute pinv via SVD
         else:
            # ToDo: Find a better svd implementation. The sparse svds commonly fails with propack
            # but propack is the only one for which we can set k=SJ.shape[1]. This is requried
            # since especially the null-penalty terms require all vectors to be recoverd to
            # compute an accurate pseudo-inverse.
            # casting to an array and then using svd works but I would rather avoid that to benefit
            # from the smaller memory footprint of the sparse variants.
            #u, Sig, vt = scp.sparse.linalg.svds(SJ,SJ.shape[1],solver='propack',maxiter=20*SJ.shape[1])
            u, Sig, vt = scp.linalg.svd(SJs[SJi].toarray())
            rs = [1/s if s > 1e-7 else 0 for s in Sig]
            SJ_pinv = vt.T @ np.diag(rs) @ u.T
            SJ_pinv = scp.sparse.csc_array(SJ_pinv)

         SJ_pinv_elements,SJ_pinv_rows,SJ_pinv_cols = translate_sparse(SJ_pinv)

         SJ_pinv_rows = np.array(SJ_pinv_rows)
         SJ_pinv_cols = np.array(SJ_pinv_cols)

         SJ_pinv_shape = SJs[SJi].shape[1]
         for _ in range(SJ_reps[SJi]):
            S_pinv_elements.extend(SJ_pinv_elements)
            S_pinv_rows.extend(SJ_pinv_rows + cIndexPinv)
            S_pinv_cols.extend(SJ_pinv_cols + cIndexPinv)
            cIndexPinv += SJ_pinv_shape
         
         for _ in range(SJ_terms[SJi]):
            FS_use_rank.append(False)
         
   S_pinv = scp.sparse.csc_array((S_pinv_elements,(S_pinv_rows,S_pinv_cols)),shape=(col_S,col_S))

   if len(FS_use_rank) != len(penalties):
      raise IndexError("An incorrect number of rank decisions were made.")
   
   return S_emb, S_pinv, FS_use_rank

def PIRLS_pdat_weights(y,mu,eta,family:Family):
   # Compute pseudo-data and weights for Penalized Reweighted Least Squares iteration (Wood, 2017, 6.1.1)
   # Calculation is based on a(mu) = 1, so reflects Fisher scoring!
   dy1 = family.link.dy1(mu)
   z = dy1 * (y - mu) + eta
   w = 1 / (dy1**2 * family.V(mu))
   return z, w

def update_PIRLS(y,yb,mu,eta,X,Xb,family):
   # Update the PIRLS weights and data (if the model is not Gaussian)
   # and update the fitting matrices yb & Xb
   z = None
   Wr = None

   if isinstance(family,Gaussian) == False:
      # Compute weights and pseudo-dat
      z, w = PIRLS_pdat_weights(y,mu,eta,family)

      Wr = scp.sparse.spdiags([np.sqrt(np.ndarray.flatten(w))],[0])

      # Update yb and Xb
      yb = Wr @ z
      Xb = (Wr @ X).tocsc()
      
   
   return yb,Xb,z,Wr

def compute_eigen_perm(Pr):
   nP = len(Pr)
   P = [1 for _ in range(nP)]
   Pc = [c for c in range(nP)]
   Perm = scp.sparse.csc_array((P,(Pr,Pc)),shape=(nP,nP))
   return Perm

def apply_eigen_perm(Pr,InvCholXXSP):
   Perm = compute_eigen_perm(Pr)
   InvCholXXS = InvCholXXSP @ Perm
   return InvCholXXS

def compute_B_mp(L,PD):
   B = cpp_solve_tr(L,PD)
   return B.power(2).sum()
   
def compute_B(L,P,lTerm,n_c=10):
   # Solves L @ B = P @ D for B, parallelizing over column
   # blocks of D if int(D.shape[1]/1000) > 1

   # D is extremely sparse and P only shuffles rows, so we
   # can take only the columns which we know contain non-zero elements
   D_start = lTerm.start_index
   D_len = lTerm.rep_sj * lTerm.S_J.shape[1]
   D_end = lTerm.start_index + D_len

   D_r = int(D_len/1000)
   if D_r > 1 and n_c > 1:
      # Parallelize
      n_c = min(D_r,n_c)
      split = np.array_split(range(D_start,D_end),n_c)
      PD = P @ lTerm.D_J_emb
      PDs = [PD[:,split[i]] for i in range(n_c)]

      with mp.Pool(processes=n_c) as pool:
         args = zip(repeat(L),PDs)
        
         pow_sums = pool.starmap(compute_B_mp,args)
      return sum(pow_sums)

   B = cpp_solve_tr(L,P @ lTerm.D_J_emb[:,D_start:D_end])
   return B.power(2).sum()

def compute_Linv(L,n_c=10):
   # Solves L @ inv(L) = I for Binv(L) parallelizing over column
   # blocks of I if int(I.shape[1]/2000) > 1
   
   n_col = L.shape[1]
   r = int(n_col/2000)
   T = scp.sparse.eye(n_col,format='csc')
   if r > 1 and n_c > 1:
      # Parallelize over column blocks of I
      # Can speed up computations considerably and is feasible memory-wise
      # since L itself is super sparse.
      
      n_c = min(r,n_c)
      split = np.array_split(range(n_col),n_c)
      LBs = [T[:,split[i]] for i in range(n_c)]

      with mp.Pool(processes=n_c) as pool:
         args = zip(repeat(L),LBs)
        
         LBinvs = pool.starmap(cpp_solve_tr,args)
      
      return scp.sparse.hstack(LBinvs)

   return cpp_solve_tr(L,T)


def calculate_edf(InvCholXXS,penalties,colsX):
   total_edf = colsX
   Bs = []
   term_edfs = []

   for lTerm in penalties:
      B = InvCholXXS @ lTerm.D_J_emb # Needed for Fellner Schall update (Wood & Fasiolo, 2016)
      Bps = B.power(2).sum()
      pen_params = lTerm.lam * Bps
      total_edf -= pen_params
      Bs.append(Bps)
      term_edfs.append(pen_params) # Not actually edf yet - rather the amount of parameters penalized away by individual penalties.
   
   return total_edf,term_edfs,Bs

def calculate_term_edf(penalties,param_penalized):
   # We need to correclty subtract all parameters penalized by
   # individual penalties from the number of coefficients of each term
   # to get the term-wise edf.
   term_pen_params = []
   term_n_coef = []
   term_idx = []
   SJ_idx_max = 0
   SJ_idx_len = 0

   for lti,lTerm in enumerate(penalties):

      # Collect the n_coef for this term, the n_pen_coef and the idx
      if lti == 0 or lTerm.start_index > SJ_idx_max:
         term_n_coef.append(lTerm.S_J.shape[1]*lTerm.rep_sj)
         term_pen_params.append(param_penalized[lti])
         term_idx.append(lTerm.start_index)
         SJ_idx_max = lTerm.start_index
         SJ_idx_len += 1

      else: # A term with the same starting index exists already - so sum the pen_params
         idx_match = [idx for idx in range(SJ_idx_len) if term_idx[idx] == lTerm.start_index]
         #print(idx_match,lTerm.start_index)
         if len(idx_match) > 1:
            raise ValueError("Penalty index matches multiple previous locations.")
         
         if term_n_coef[idx_match[0]] != lTerm.S_J.shape[1]*lTerm.rep_sj:
            raise ValueError("Penalty dimensions do not match!")
         
         term_pen_params[idx_match[0]] += param_penalized[lti]
   
   # Now we can compute the final term-wise edf.
   term_edf = []
   for lti in range(SJ_idx_len):
      term_edf.append(term_n_coef[lti] - term_pen_params[lti])
   
   return term_edf

def update_scale_edf(y,z,eta,Wr,rowsX,colsX,InvCholXXSP,Pr,family,penalties):
   # Updates the scale of the model. For this the edf
   # are computed as well - they are returned because they are needed for the
   # lambda step proposal anyway.
   
   # Calculate Pearson residuals for GAMM (Wood, 3.1.7)
   # Standard residuals for AMM
   if isinstance(family,Gaussian) == False:
      wres = Wr @ (z - eta)
   else:
      wres = y - eta

   # Calculate total and term wise edf
   InvCholXXS = apply_eigen_perm(Pr,InvCholXXSP)

   # If there are penalized terms we need to adjust the total_edf
   if len(penalties) > 0:
      total_edf, term_edfs, Bs = calculate_edf(InvCholXXS,penalties,colsX)
   else:
      total_edf = colsX
      term_edfs = None
      Bs = None

   # Optionally estimate scale parameter
   if family.scale is None:
      scale = est_scale(wres,rowsX,total_edf)
   else:
      scale = family.scale
   
   return wres,InvCholXXS,total_edf,term_edfs,Bs,scale

def update_coef_and_scale(y,yb,z,Wr,rowsX,colsX,X,Xb,family,S_emb,penalties,n_c,formula):
   # Solves the additive model for a given set of weights and penalty
   if formula is None:
      LP, Pr, coef, code = cpp_solve_coef(yb,Xb,S_emb)
   else:
      #yb is X.T@y and Xb is X.T@X
      LP, Pr, coef, code = cpp_solve_coefXX(yb,Xb + S_emb)

   if code != 0:
      raise ArithmeticError(f"Solving for coef failed with code {code}. Model is likely unidentifiable.")
   
   # Solve for inverse of Chol factor of XX+S
   InvCholXXSP = compute_Linv(LP,n_c)
   
   # Update mu & eta
   if formula is None:
      eta = (X @ coef).reshape(-1,1)
   else:
      eta = []
      for file in formula.file_paths:
         eta_file = read_eta(file,formula,coef,n_c)
         eta.extend(eta_file)
      eta = np.array(eta)

   mu = eta

   if isinstance(family,Gaussian) == False:
      mu = family.link.fi(eta)

   # Update scale parameter
   wres,InvCholXXS,total_edf,term_edfs,Bs,scale = update_scale_edf(y,z,eta,Wr,rowsX,colsX,InvCholXXSP,Pr,family,penalties)
   return eta,mu,coef,InvCholXXS,total_edf,term_edfs,Bs,scale,wres

def init_step_gam(y,yb,mu,eta,rowsX,colsX,X,Xb,
                  family,col_S,penalties,
                  pinv,n_c,formula):
   # Initial fitting iteration without step-length control for gam.

   # Compute starting estimate S_emb and S_pinv
   if len(penalties) > 0:
      S_emb,S_pinv,FS_use_rank = compute_S_emb_pinv_det(col_S,penalties,pinv)
   else:
      S_emb = scp.sparse.csc_array((colsX, colsX), dtype=np.float64)

   # Estimate coefficients for starting lambda
   # We just accept those here - no step control, since
   # there are no previous coefficients/deviance that we can
   # compare the result to.

   # First (optionally, only in the non Gaussian case) compute pseudo-dat and weights:
   yb,Xb,z,Wr = update_PIRLS(y,yb,mu,eta,X,Xb,family)
   
   # Solve additive model
   eta,mu,coef,\
   InvCholXXS,\
   total_edf,\
   term_edfs,\
   Bs,scale,wres = update_coef_and_scale(y,yb,z,Wr,rowsX,colsX,
                                         X,Xb,family,S_emb,
                                         penalties,n_c,formula)
   
   # Deviance under these starting coefficients
   # As well as penalized deviance
   dev = family.deviance(y,mu)
   pen_dev = dev

   if len(penalties) > 0:
      pen_dev += coef.T @ S_emb @ coef

   # Now propose first lambda extension via the Fellner Schall method
   # by Wood & Fasiolo (2016). Simply don't use an extension term (see Wood & Fasiolo; 2016) for
   # this first update.
   lam_delta = []
   if len(penalties) > 0:
      for lti,lTerm in enumerate(penalties):

         lt_rank = None
         if FS_use_rank[lti]:
            lt_rank = lTerm.rank
         
         lgdetD,bsb = compute_lgdetD_bsb(lt_rank,lTerm.lam,S_pinv,lTerm.S_J_emb,coef)
         dLam = step_fellner_schall_sparse(lgdetD,Bs[lti],bsb,lTerm.lam,scale)
         lam_delta.append(dLam)

      lam_delta = np.array(lam_delta).reshape(-1,1)
   
   return dev,pen_dev,eta,mu,coef,InvCholXXS,total_edf,term_edfs,scale,wres,lam_delta,S_emb


def correct_coef_step(coef,n_coef,dev,pen_dev,c_dev_prev,family,eta,mu,y,X,n_pen,S_emb,formula,n_c):
   # Perform step-length control for the coefficients (Step 3 in Wood, 2017)
   corrections = 0
   while pen_dev > c_dev_prev:
      # Coefficient step did not improve deviance - so correction
      # is necessary.

      if corrections > 30:
         # If we could not find a better coefficient set simply accept
         # previous coefficient
         n_coef = coef

      n_coef = (coef + n_coef)/2

      # Update mu & eta for correction
      if formula is None:
         eta = (X @ n_coef).reshape(-1,1)
      else:
         eta = []
         for file in formula.file_paths:
            eta_file = read_eta(file,formula,n_coef,n_c)
            eta.extend(eta_file)
         eta = np.array(eta)

      mu = eta

      if isinstance(family,Gaussian) == False:
         mu = family.link.fi(eta)
      
      # Update deviance
      dev = family.deviance(y,mu)

      # And penalized deviance term
      if n_pen > 0:
         pen_dev = dev + n_coef.T @ S_emb @ n_coef
      corrections += 1
   
   # Collect accepted coefficient
   coef = n_coef
   return dev,pen_dev,mu,eta,coef

def correct_lambda_step(y,yb,z,Wr,rowsX,colsX,X,Xb,
                        family,col_S,S_emb,penalties,
                        pinv,lam_delta,extend_by,o_iter,
                        dev_check,n_c,control_lambda,
                        extend_lambda,exclude_lambda,
                        formula):
   # Propose & perform step-length control for the lambda parameters via the Fellner Schall method
   # by Wood & Fasiolo (2016)
   lam_accepted = False
   lam_checks = 0
   while not lam_accepted:

      # Re-compute S_emb and S_pinv
      S_emb,S_pinv,FS_use_rank = compute_S_emb_pinv_det(col_S,penalties,pinv)

      # Update coefficients
      eta,mu,n_coef,\
      InvCholXXS,\
      total_edf,\
      term_edfs,\
      Bs,scale,wres = update_coef_and_scale(y,yb,z,Wr,rowsX,colsX,
                                             X,Xb,family,S_emb,
                                             penalties,n_c,formula)
      
      # Compute gradient of REML with respect to lambda
      # to check if step size needs to be reduced.
      lgdetDs = []
      bsbs = []
      for lti,lTerm in enumerate(penalties):

         lt_rank = None
         if FS_use_rank[lti]:
            lt_rank = lTerm.rank

         lgdetD,bsb = compute_lgdetD_bsb(lt_rank,lTerm.lam,S_pinv,lTerm.S_J_emb,n_coef)
         lgdetDs.append(lgdetD)
         bsbs.append(bsb)

      lam_grad = [grad_lambda(lgdetDs[lti],Bs[lti],bsbs[lti],scale) for lti in range(len(penalties))]
      lam_grad = np.array(lam_grad).reshape(-1,1) 
      check = lam_grad.T @ lam_delta

      if check[0,0] < 0 and control_lambda: # because of minimization in Wood (2017) they use a different check.
         # Reset extension or cut the step taken in half
         for lti,lTerm in enumerate(penalties):
            if extend_lambda and extend_by > 1:
               # I experimented with just iteratively reducing the step-size but it just takes too many
               # wasted iterations then. Thus, I now just reset the extension factor below. It can then build up again
               # if needed.
               lTerm.lam -= lam_delta[lti][0]
               lam_delta[lti] /= extend_by
               lTerm.lam += lam_delta[lti][0]
            else: # If the step size extension is already at the minimum, fall back to the strategy by Wood (2017) to just half the step
               lam_delta[lti] = lam_delta[lti]/2
               lTerm.lam -= lam_delta[lti][0]

         if extend_lambda and extend_by > 1:
            extend_by = 1
      else:
         if extend_lambda and lam_checks == 0 and extend_by < 2 and o_iter > 0 and dev_check: # Try longer step next time.
            extend_by += 0.5

         # Accept the step and propose a new one as well!
         lam_accepted = True
         lam_delta = []
         for lti,(lGrad,lTerm) in enumerate(zip(lam_grad,penalties)):
            
            if np.abs(lGrad[0]) >= 1e-8*np.sum(np.abs(lam_grad)) or exclude_lambda == False:

               dLam = step_fellner_schall_sparse(lgdetDs[lti],Bs[lti],bsbs[lti],lTerm.lam,scale)

               if extend_lambda:
                  extension = lTerm.lam  + dLam*extend_by
                  if extension < 1e7 and extension > 1e-7:
                     dLam *= extend_by
            else: # ReLikelihood is insensitive to further changes in this smoothing penalty, so set change to 0.
               dLam = 0

            lam_delta.append(dLam)

         lam_delta = np.array(lam_delta).reshape(-1,1)

      lam_checks += 1

   return eta,mu,n_coef,InvCholXXS,total_edf,term_edfs,scale,wres,lam_delta,extend_by,penalties,S_emb

def solve_gamm_sparse(mu_init,y,X,penalties,col_S,family:Family,
                      maxiter=10,pinv="svd",conv_tol=1e-7,
                      extend_lambda=True,control_lambda=True,
                      exclude_lambda=False,
                      progress_bar=False,n_c=10):
   # Estimates a penalized Generalized additive mixed model, following the steps outlined in Wood (2017)
   # "Generalized Additive Models for Gigadata"

   n_c = min(mp.cpu_count(),n_c)
   rowsX,colsX = X.shape
   coef = None
   n_coef = None
   extend_by = 1 # Extension factor for lambda update for the Fellner Schall method by Wood & Fasiolo (2016)

   # Additive mixed model can simply be fit on y and X
   # Generalized mixed model needs to be fit on weighted X and pseudo-dat
   # but the same routine can be used (Wood, 2017) so both should end
   # up in the same variables passed down:
   yb = y
   Xb = X

   # mu and eta (start estimates in case the family is not Gaussian)
   mu = mu_init
   eta = mu

   if isinstance(family,Gaussian) == False:
      eta = family.link.f(mu)

   # Compute starting estimates
   dev,pen_dev,eta,mu,coef,InvCholXXS,total_edf,term_edfs,scale,wres,lam_delta,S_emb = init_step_gam(y,yb,mu,eta,rowsX,colsX,X,Xb,
                                                                                                     family,col_S,penalties,
                                                                                                     pinv,n_c,None)
   
   # Loop to optimize smoothing parameter (see Wood, 2017)
   iterator = range(maxiter)
   if progress_bar:
      iterator = tqdm(iterator,desc="Fitting",leave=True)

   for o_iter in iterator:

      # We need the previous deviance and penalized deviance
      # for step control and convergence control respectively
      prev_dev = dev
      prev_pen_dev = pen_dev
      
      if o_iter > 0:

         # Obtain deviance and penalized deviance terms
         # under current lambda for proposed coef (n_coef)
         # and current coef.
         dev = family.deviance(y,mu) 
         pen_dev = dev
         c_dev_prev = prev_dev

         if len(penalties) > 0:
            pen_dev += n_coef.T @ S_emb @ n_coef
            c_dev_prev += coef.T @ S_emb @ coef

         # Perform step-length control for the coefficients (Step 3 in Wood, 2017)
         dev,pen_dev,mu,eta,coef = correct_coef_step(coef,n_coef,dev,pen_dev,c_dev_prev,family,eta,mu,y,X,len(penalties),S_emb,None,n_c)

         # Test for convergence (Step 2 in Wood, 2017)
         dev_diff = abs(pen_dev - prev_pen_dev)

         if progress_bar:
            iterator.set_description_str(desc="Fitting - Conv.: " + "{:.2e}".format(dev_diff - conv_tol*pen_dev), refresh=True)
            
         if dev_diff < conv_tol*pen_dev:
            if progress_bar:
               iterator.set_description_str(desc="Converged!", refresh=True)
               iterator.close()
            break

      # Update pseudo-dat weights for next coefficient step
      yb,Xb,z,Wr = update_PIRLS(y,yb,mu,eta,X,Xb,family)
         
      # Step length control for proposed lambda change
      if len(penalties) > 0: 
         
         # Test the lambda update
         for lti,lTerm in enumerate(penalties):
            lTerm.lam += lam_delta[lti][0]
            #print(lTerm.lam,lam_delta[lti][0])

         # Now check step length and compute lambda + coef update.
         dev_check = None
         if o_iter > 0:
            dev_check = dev_diff < 1e-3*pen_dev

         eta,mu,n_coef,InvCholXXS,total_edf,term_edfs,scale,wres,lam_delta,extend_by,penalties,S_emb = correct_lambda_step(y,yb,z,Wr,rowsX,colsX,X,Xb,
                                                                                                                           family,col_S,S_emb,penalties,
                                                                                                                           pinv,lam_delta,extend_by,o_iter,
                                                                                                                           dev_check,n_c,control_lambda,
                                                                                                                           extend_lambda,exclude_lambda,None)
      
      else:
         # If there are no penalties simply perform a newton step
         # for the coefficients only
         eta,mu,n_coef,\
         InvCholXXS,\
         total_edf,\
         term_edfs,\
         _,scale,wres = update_coef_and_scale(y,yb,z,Wr,rowsX,colsX,
                                              X,Xb,family,S_emb,
                                              penalties,n_c,None)

   # Final penalty
   if len(penalties) > 0:
      penalty = coef.T @ S_emb @ coef
   else:
      penalty = 0

   # Final term edf
   if not term_edfs is None:
      term_edfs = calculate_term_edf(penalties,term_edfs)

   return coef,eta,wres,scale,InvCholXXS,total_edf,term_edfs,penalty

################################################ Iterative GAMM building code ################################################

def read_mmat(should_cache,cache_dir,file,fi,terms,has_intercept,has_scale_split,
              ltx,irstx,stx,rtx,var_types,var_map,var_mins,
              var_maxs,factor_levels,cov_flat_file,cov,n_j,
              state_est_flat,state_est):
   """
   Creates model matrix for that dataset. The model-matrix is either cached or not. If the former is the case,
   the matrix is read in on subsequent calls to this function
   """

   target = file.split("/")[-1].split(".csv")[0] + f"_{fi}.npz"
   
   if should_cache == False:
         mmat = build_sparse_matrix_from_formula(terms,has_intercept,has_scale_split,
                                       ltx,irstx,stx,rtx,var_types,var_map,var_mins,
                                       var_maxs,factor_levels,cov_flat_file,cov,n_j,
                                       state_est_flat,state_est)
         
   elif should_cache == True and target not in os.listdir(cache_dir):
         mmat = build_sparse_matrix_from_formula(terms,has_intercept,has_scale_split,
                                       ltx,irstx,stx,rtx,var_types,var_map,var_mins,
                                       var_maxs,factor_levels,cov_flat_file,cov,n_j,
                                       state_est_flat,state_est)
         
         scp.sparse.save_npz(f"{cache_dir}/" + target,mmat)
   else:
         mmat = scp.sparse.load_npz(f"{cache_dir}/" + target)

   return mmat

def form_cross_prod_mp(should_cache,cache_dir,file,fi,y_flat,terms,has_intercept,has_scale_split,
                       ltx,irstx,stx,rtx,var_types,var_map,var_mins,
                       var_maxs,factor_levels,cov_flat_file,cov,n_j,
                       state_est_flat,state_est):
   
   model_mat = read_mmat(should_cache,cache_dir,file,fi,terms,has_intercept,has_scale_split,
                         ltx,irstx,stx,rtx,var_types,var_map,var_mins,
                         var_maxs,factor_levels,cov_flat_file,cov,n_j,
                         state_est_flat,state_est)
   
   Xy = model_mat.T @ y_flat
   XX = (model_mat.T @ model_mat).tocsc()

   return XX,Xy

def read_mmat_cross(file,formula,nc):
   """
   Reads subset of data and creates cross-product of model matrix for that dataset.
   """

   terms = formula.get_terms()
   has_intercept = formula.has_intercept()
   has_scale_split = False
   ltx = formula.get_linear_term_idx()
   irstx = []
   stx = formula.get_smooth_term_idx()
   rtx = formula.get_random_term_idx()
   var_types = formula.get_var_types()
   var_map = formula.get_var_map()
   var_mins = formula.get_var_mins()
   var_maxs = formula.get_var_maxs()
   factor_levels = formula.get_factor_levels()

   for sti in stx:
      if terms[sti].should_rp:
         for rpi in range(len(terms[sti].RP)):
            # Don't need to pass those down to the processes.
            terms[sti].RP[rpi].X = None
            terms[sti].RP[rpi].cov = None

   cov = None
   n_j = None
   state_est_flat = None
   state_est = None

   # Read file
   file_dat = pd.read_csv(file)

   # Encode data in this file
   y_flat_file,cov_flat_file,NAs_flat_file,_,_,_,_ = formula.encode_data(file_dat)
   cov_flat_file = cov_flat_file[NAs_flat_file,:]
   y_flat_file = y_flat_file[NAs_flat_file]

   # Parallelize over sub-sets of this file
   rows,_ = cov_flat_file.shape
   split = np.arange(0,rows,max(1,MP_SPLIT_SIZE),dtype=int)[1:]
   cov_flat_files = np.vsplit(cov_flat_file,split)
   y_flat_files = np.split(y_flat_file,split)
   subsets = [fi for fi in range(len(split) + 1)]

   with mp.Pool(processes=nc) as pool:
      # Build the model matrix with all information from the formula - but only for sub-set of rows in this file
      XX,Xy = zip(*pool.starmap(form_cross_prod_mp,zip(repeat(SHOULD_CACHE),repeat(CACHE_DIR),repeat(file),subsets,y_flat_files,repeat(terms),repeat(has_intercept),repeat(has_scale_split),
                                                       repeat(ltx),repeat(irstx),repeat(stx),repeat(rtx),
                                                       repeat(var_types),repeat(var_map),
                                                       repeat(var_mins),repeat(var_maxs),repeat(factor_levels),
                                                       cov_flat_files,repeat(cov),repeat(n_j),
                                                       repeat(state_est_flat),repeat(state_est))))
   
   XX = reduce(lambda xx1,xx2: xx1+xx2,XX)
   Xy = reduce(lambda xy1,xy2: xy1+xy2,Xy)
   return XX,Xy,len(y_flat_file)

def form_eta_mp(should_cache,cache_dir,file,fi,coef,terms,has_intercept,has_scale_split,
                ltx,irstx,stx,rtx,var_types,var_map,var_mins,
                var_maxs,factor_levels,cov_flat_file,cov,n_j,
                state_est_flat,state_est):
   
   model_mat = read_mmat(should_cache,cache_dir,file,fi,terms,has_intercept,has_scale_split,
                         ltx,irstx,stx,rtx,var_types,var_map,var_mins,
                         var_maxs,factor_levels,cov_flat_file,cov,n_j,
                         state_est_flat,state_est)
   
   eta_file = (model_mat @ coef).reshape(-1,1)
   return eta_file

def read_eta(file,formula,coef,nc):
   """
   Reads subset of data and creates model matrix for that dataset.
   """

   terms = formula.get_terms()
   has_intercept = formula.has_intercept()
   has_scale_split = False
   ltx = formula.get_linear_term_idx()
   irstx = []
   stx = formula.get_smooth_term_idx()
   rtx = formula.get_random_term_idx()
   var_types = formula.get_var_types()
   var_map = formula.get_var_map()
   var_mins = formula.get_var_mins()
   var_maxs = formula.get_var_maxs()
   factor_levels = formula.get_factor_levels()

   for sti in stx:
      if terms[sti].should_rp:
         for rpi in range(len(terms[sti].RP)):
            # Don't need to pass those down to the processes.
            terms[sti].RP[rpi].X = None
            terms[sti].RP[rpi].cov = None

   cov = None
   n_j = None
   state_est_flat = None
   state_est = None

   # Read file
   file_dat = pd.read_csv(file)

   # Encode data in this file
   _,cov_flat_file,NAs_flat_file,_,_,_,_ = formula.encode_data(file_dat)
   cov_flat_file = cov_flat_file[NAs_flat_file,:]

   # Parallelize over sub-sets of this file
   rows,_ = cov_flat_file.shape
   split = np.arange(0,rows,max(1,MP_SPLIT_SIZE),dtype=int)[1:]
   cov_flat_files = np.vsplit(cov_flat_file,split)
   subsets = [fi for fi in range(len(split) + 1)]

   with mp.Pool(processes=nc) as pool:
      # Build eta with all information from the formula - but only for sub-set of rows in this file
      etas = pool.starmap(form_eta_mp,zip(repeat(SHOULD_CACHE),repeat(CACHE_DIR),repeat(file),subsets,repeat(coef),repeat(terms),repeat(has_intercept),repeat(has_scale_split),
                                         repeat(ltx),repeat(irstx),repeat(stx),repeat(rtx),
                                         repeat(var_types),repeat(var_map),
                                         repeat(var_mins),repeat(var_maxs),repeat(factor_levels),
                                         cov_flat_files,repeat(cov),repeat(n_j),
                                         repeat(state_est_flat),repeat(state_est)))

   eta = []
   for eta_file in etas:
      eta.extend(eta_file)

   return eta

def solve_gamm_sparse2(formula:Formula,penalties,col_S,family:Family,
                       maxiter=10,pinv="svd",conv_tol=1e-7,
                       extend_lambda=True,control_lambda=True,
                       exclude_lambda=False,
                       progress_bar=False,n_c=10):
   # Estimates a penalized additive mixed model, following the steps outlined in Wood (2017)
   # "Generalized Additive Models for Gigadata" but builds X.T @ X, and X.T @ y iteratively - and only once.
   setup_cache(CACHE_DIR,SHOULD_CACHE)
   n_c = min(mp.cpu_count(),n_c)

   y_flat = []
   rowsX = 0

   iterator = formula.file_paths
   if progress_bar:
      iterator = tqdm(iterator,desc="Accumulating X.T @ X",leave=True)

   for fi,file in enumerate(iterator):
      # Read file
      file_dat = pd.read_csv(file)

      # Encode data in this file
      y_flat_file,_,NAs_flat_file,_,_,_,_ = formula.encode_data(file_dat)
      y_flat.extend(y_flat_file[NAs_flat_file])

      # Build the model matrix cross-products with all information from the formula - but only for sub-set of rows in this file
      XX0,Xy0,rowsX0 = read_mmat_cross(file,formula,n_c)
      rowsX += rowsX0
      # Compute cross-product
      if fi == 0:
         XX = XX0
         Xy = Xy0
      else:
         XX += XX0
         Xy += Xy0
      
   colsX = XX.shape[1]
   coef = None
   n_coef = None
   extend_by = 1 # Extension factor for lambda update for the Fellner Schall method by Wood & Fasiolo (2016)

   # mu and eta (start estimates in case the family is not Gaussian)
   y = np.array(y_flat)
   mu = np.array(y_flat)
   eta = mu

   # Compute starting estimates
   dev,pen_dev,eta,mu,coef,InvCholXXS,total_edf,term_edfs,scale,wres,lam_delta,S_emb = init_step_gam(y,Xy,mu,eta,rowsX,colsX,None,XX,
                                                                                                     family,col_S,penalties,
                                                                                                     pinv,n_c,formula)
   
   # Loop to optimize smoothing parameter (see Wood, 2017)
   iterator = range(maxiter)
   if progress_bar:
      iterator = tqdm(iterator,desc="Fitting",leave=True)

   for o_iter in iterator:

      # We need the previous deviance and penalized deviance
      # for step control and convergence control respectively
      prev_dev = dev
      prev_pen_dev = pen_dev
      
      if o_iter > 0:

         # Obtain deviance and penalized deviance terms
         # under current lambda for proposed coef (n_coef)
         # and current coef.
         dev = family.deviance(y,mu) 
         pen_dev = dev
         c_dev_prev = prev_dev

         if len(penalties) > 0:
            pen_dev += n_coef.T @ S_emb @ n_coef
            c_dev_prev += coef.T @ S_emb @ coef

         # Perform step-length control for the coefficients (Step 3 in Wood, 2017)
         dev,pen_dev,mu,eta,coef = correct_coef_step(coef,n_coef,dev,pen_dev,c_dev_prev,family,eta,mu,y,None,len(penalties),S_emb,formula,n_c)

         # Test for convergence (Step 2 in Wood, 2017)
         dev_diff = abs(pen_dev - prev_pen_dev)

         if progress_bar:
            iterator.set_description_str(desc="Fitting - Conv.: " + "{:.2e}".format(dev_diff - conv_tol*pen_dev), refresh=True)
            
         if dev_diff < conv_tol*pen_dev:
            if progress_bar:
               iterator.set_description_str(desc="Converged!", refresh=True)
               iterator.close()
            break
         
      # Step length control for proposed lambda change
      if len(penalties) > 0: 
         
         # Test the lambda update
         for lti,lTerm in enumerate(penalties):
            lTerm.lam += lam_delta[lti][0]
            #print(lTerm.lam,lam_delta[lti][0])

         # Now check step length and compute lambda + coef update.
         dev_check = None
         if o_iter > 0:
            dev_check = dev_diff < 1e-3*pen_dev

         eta,mu,n_coef,InvCholXXS,total_edf,term_edfs,scale,wres,lam_delta,extend_by,penalties,S_emb = correct_lambda_step(y,Xy,None,None,rowsX,colsX,None,XX,
                                                                                                                           family,col_S,S_emb,penalties,
                                                                                                                           pinv,lam_delta,extend_by,o_iter,
                                                                                                                           dev_check,n_c,control_lambda,
                                                                                                                           extend_lambda,exclude_lambda,
                                                                                                                           formula)
      
      else:
         # If there are no penalties simply perform a newton step
         # for the coefficients only
         eta,mu,n_coef,\
         InvCholXXS,\
         total_edf,\
         term_edfs,\
         _,scale,wres = update_coef_and_scale(y,Xy,None,None,rowsX,colsX,
                                              None,XX,family,S_emb,penalties,n_c,formula)

   # Final penalty
   if len(penalties) > 0:
      penalty = coef.T @ S_emb @ coef
   else:
      penalty = 0

   # Final term edf
   if not term_edfs is None:
      term_edfs = calculate_term_edf(penalties,term_edfs)

   clear_cache(CACHE_DIR,SHOULD_CACHE)

   return coef,eta,wres,scale,InvCholXXS,total_edf,term_edfs,penalty
