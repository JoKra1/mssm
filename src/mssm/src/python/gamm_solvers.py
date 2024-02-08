import numpy as np
import scipy as scp
import multiprocessing as mp
from itertools import repeat
import warnings
from .exp_fam import Family,Gaussian
from .penalties import PenType,id_dist_pen,translate_sparse
import cpp_solvers
from tqdm import tqdm
import math
import copy

def cpp_chol(A):
   return cpp_solvers.chol(A)

def cpp_qr(A):
   return cpp_solvers.pqr(A)

def cpp_solve_am(y,X,S):
   return cpp_solvers.solve_am(y,X,S)

def cpp_solve_coef(y,X,S):
   return cpp_solvers.solve_coef(y,X,S)

def cpp_solve_L(X,S):
   return cpp_solvers.solve_L(X,S)

def cpp_solve_tr(A,B):
   return cpp_solvers.solve_tr(A,B)

def cpp_backsolve_tr(A,B):
   return cpp_solvers.backsolve_tr(A,B)

def reparam(X,S,cov,option=1,n_bins=30,QR=False,identity=False,scale=False):
   """
    Options 1 - 3 are natural reparameterization discussed in Wood (2017; 5.4.2)
    with different strategies for the QR computation of X.

       1. Form complete matrix X based on entire covariate.
       2. Form matrix X only based on unique covariate values.
       3. Form matrix X on a sample of values making up covariate. Covariate
       is split up into ``n_bins`` equally wide bins. The number of covariate values
       per bin is then calculated. Subsequently, the ratio relative to minimum bin size is
       computed and each ratio is rounded to the nearest integer. Then ``ratio`` samples
       are obtained from each bin. That way, imbalance in the covariate is approximately preserved when
       forming the QR.
    
    If ``QR==True`` then X is decomposed into Q @ R directly via QR decomposition. Alternatively, we first
    form X.T @ X and then compute the cholesky L of this product - note that L.T = R. Overall the latter
    strategy is much faster (in particular if ``option==1``), but the increased loss of precision in L/R
    might not be ok for some.

    After transformation S only contains elements on it's diagonal and X the transformed functions. As discussed
    in Wood (2017), the transformed functions are decreasingly flexible - so the elements on S diagonal become smaller
    and eventually zero, for elements that are in the kernel of the original S (un-penalized == not flexible).

    For a similar transformation (based solely on S), Wood et al. (2013) show how to further reduce the diagonally
    transformed S to an even simpler identity penalty. As discussed also in Wood (2017) the same behavior of decreasing
    flexibility if all entries on the diagonal of S are 1 can only be maintained if the transformed functions are
    multiplied by a weight related to their wiggliness. Specifically, more flexible functions need to become smaller in
    amplitude - so that for the same level of penalization they are removed earlier than less flexible ones. To achieve this
    Wood further post-multiply the transformed matrix 'X with a matrix that contains on it's diagonal the reciprocal of the
    square root of the transformed penalty matrix (and 1s in the last cells corresponding to the kernel). This is done here
    if ``identity=True``.

    In ``mgcv`` the transformed model matrix and penalty can optionally be scaled by the root mean square value of the transformed
    model matrix (see the nat.param function in mgcv). This is done here if ``scale=True``.

    References:
      - Wood, S. N., Scheipl, F., & Faraway, J. J. (2013). Straightforward intermediate rank tensor product smoothing in mixed models.
      - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
      - mgcv source code (accessed 2024). smooth.R file, nat.param function.
   """

   if option < 4:

      # For option 1 just use provided basis matrix
      if option != 1:
         unq,idx,c = np.unique(cov,return_counts=True,return_index=True)
         if option == 2:
            # Form basis based on unique values in cov
            sample = idx
         elif option == 3:
            # Form basis based on re-sampled values of cov to keep row number small but hopefully imbalance
            # in the data preserved.
            weights,values = np.histogram(cov,bins=n_bins)
            ratio = np.round(weights/min(weights),decimals=0).astype(int)

            sample = []
            for bi in range(n_bins-1):
               sample_val = np.random.choice(unq[(unq >= values[bi]) & (unq < values[bi+1])],size=ratio[bi],replace=True)
               sample_idx = [idx[unq == sample_val[svi]][0] for svi in range(ratio[bi])]
               sample.extend(sample_idx)
            sample.append(idx[-1])
            sample = np.array(sample)
            
         # Now re-form basis
         X = X[sample,:]
      
      # Now decompose X = Q @ R
      if QR:
         _,R = scp.linalg.qr(X.toarray(),mode='economic')
         R = scp.sparse.csc_array(R)
         
      else:
         XX = X.T @ X
         L,code = cpp_chol(XX)

         if code != 0:
            raise ValueError("Cholesky failed during reparameterization.")

         R = L.T

      # Now form B and proceed with eigen decomposition of it (see Wood, 2017)
      # see also smooth.R nat.param function in mgcv.
      # R.T @ A = S.T
      # A = Rinv.T @ S.T
      # R.T @ B = A.T
      # A.T = S @ Rinv ## Transpose order reverses!
      # B = Rinv.T @ A.T
      # B = Rinv.T @ S @ Rinv
      B = cpp_solve_tr(R.T,cpp_solve_tr(R.T,S.T).T)

      s, U =scp.linalg.eigh(B.toarray())

      # Decreasing magnitude for ease of indexing..
      s = np.flip(s)
      U = scp.sparse.csc_array(np.flip(U,axis=1))

      rank = len(s[s > 1e-7])

      # First rank elements are non-zero - corresponding to penalized functions, last S.shape[1] - rank
      # are zero corresponding to dimensionality of kernel of S
      Srp = scp.sparse.diags([s[i] if s[i] > 1e-7 else 0 for i in range(S.shape[1])],offsets=0,format='csc')
      Drp = scp.sparse.diags([s[i]**0.5 if s[i] > 1e-7 else 0 for i in range(S.shape[1])],offsets=0,format='csc')

      # Now compute matrix to transform basis functions. The transformed functions are decreasingly flexible. I.e.,
      # Xrp[:,i] is more flexible than Xrp[:,i+1]. According to Wood (2017) Xrp = Q @ U. Now we want to be able to
      # evaluate the basis for new data resulting in Xpred. So we also have to transform Xpred. Following Wood (2017),
      # based on QR decomposition we have X = Q @ R, so we form matrix C so that R @ C = U to have Xrp = Q @ R @ C = Q @ U.
      # Then Xpred_rp = X_pred @ C can similarly be obtained.
      # see smooth.R nat.param function in mgcv.

      C = cpp_backsolve_tr(R,U)

      IRrp = None
      if identity:
         # Transform S to identity as described in Wood et al. (2013). Form inverse of root of transformed S for
         # all cells not covering a kernel function. For those simply insert 1. Then post-multiply transformed X (or equivalently C) by it.
         IRrp = [1/s[i]**0.5 if s[i] > 1e-7 else 1 for i in range(S.shape[1])]
         Srp = scp.sparse.diags([1 if s[i] > 1e-7 else 0 for i in range(S.shape[1])],offsets=0,format='csc')
         Drp = copy.deepcopy(Srp)

         C = C @ scp.sparse.diags(IRrp,offsets=0,format='csc')

      rms1 = rms2 = None
      if scale:
         # mgcv optionally scales the transformed model & penalty matrices (separately per range and kernel space columns of S) by the root mean square of the model matrix.
         # see smooth.R nat.param function in mgcv.
         Xrp = X @ C
         rms1 = math.sqrt((Xrp[:,:rank]).power(2).mean())

         # Scale transformation matrix
         C[:,:rank] /= rms1
         
         # Now apply the separate scaling for Kernel of S as done by mgcv
         if X.shape[1] - rank > 0:
            rms2 = math.sqrt((Xrp[:,rank:]).power(2).mean())
            C[:,rank:] /= rms2
         
         # Scale penalty
         Srp /= rms1**2
         Drp /= rms1

      # Done, return
      return C, Srp, Drp, IRrp, rms1, rms2, rank

   else:
      raise NotImplementedError(f"Requested option {option} for reparameterization is not implemented.")

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
      Xb = Wr @ X
   
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
      scale = family.est_scale(wres,rowsX,total_edf)
   else:
      scale = family.scale
   
   return wres,InvCholXXS,total_edf,term_edfs,Bs,scale

def update_coef_and_scale(y,yb,z,Wr,rowsX,colsX,X,Xb,family,S_emb,penalties,n_c):
   # Solves the additive model for a given set of weights and penalty
   LP, Pr, coef, code = cpp_solve_coef(yb,Xb,S_emb)

   if code != 0:
      raise ArithmeticError(f"Solving for coef failed with code {code}. Model is likely unidentifiable.")
   
   # Solve for inverse of Chol factor of XX+S
   InvCholXXSP = compute_Linv(LP,n_c)
   
   # Update mu & eta
   eta = (X @ coef).reshape(-1,1)
   mu = eta

   if isinstance(family,Gaussian) == False:
      mu = family.link.fi(eta)

   # Update scale parameter
   wres,InvCholXXS,total_edf,term_edfs,Bs,scale = update_scale_edf(y,z,eta,Wr,rowsX,colsX,InvCholXXSP,Pr,family,penalties)
   return eta,mu,coef,InvCholXXS,total_edf,term_edfs,Bs,scale,wres
   
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
                                         X,Xb,family,S_emb,penalties,n_c)
   
   # Deviance under these starting coefficients
   # As well as penalized deviance
   dev = family.deviance(y,mu)
   pen_dev = dev

   if len(penalties) > 0:
      pen_dev += coef.T @ S_emb @ coef

   # Now we propose a lambda extension via the Fellner Schall method
   # by Wood & Fasiolo (2016)
   # We also consider an extension term as reccomended by Wood & Fasiolo (2016)
   extend_by = 1
   if len(penalties) > 0:
      lam_delta = []
      for lti,lTerm in enumerate(penalties):

         lt_rank = None
         if FS_use_rank[lti]:
            lt_rank = lTerm.rank
         
         lgdetD,bsb = compute_lgdetD_bsb(lt_rank,lTerm.lam,S_pinv,lTerm.S_J_emb,coef)
         dLam = step_fellner_schall_sparse(lgdetD,Bs[lti],bsb,lTerm.lam,scale)

         if extend_lambda:
            extension = lTerm.lam  + dLam*extend_by
            if extension < 1e7 and extension > 1e-7: # Keep lambda in correct space
               dLam *= extend_by

         lam_delta.append(dLam)

      lam_delta = np.array(lam_delta).reshape(-1,1)
   
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
         corrections = 0
         while pen_dev > c_dev_prev:
            # Newton step did not improve deviance - so correction
            # is necessary.

            if corrections > 30:
               # If we could not find a better coefficient set simply accept
               # previous coefficient
               n_coef = coef[:]
         
            n_coef = (coef + n_coef)/2

            # Update mu & eta for correction
            eta = (X @ n_coef).reshape(-1,1)
            mu = eta

            if isinstance(family,Gaussian) == False:
               mu = family.link.fi(eta)
            
            # Update deviance
            dev = family.deviance(y,mu)

            # And penalized deviance term
            if len(penalties) > 0:
               pen_dev = dev + n_coef.T @ S_emb @ n_coef
            corrections += 1
         
         # Collect accepted coefficient
         coef = n_coef[:]

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
                                                  X,Xb,family,S_emb,penalties,n_c)
            
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
               if extend_lambda and lam_checks == 0 and extend_by < 2 and o_iter > 0 and dev_diff < 1e-3*pen_dev: # Try longer step next time.
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
      
      else:
         # If there are no penalties simply perform a newton step
         # for the coefficients only
         eta,mu,n_coef,\
         InvCholXXS,\
         total_edf,\
         term_edfs,\
         Bs,scale,wres = update_coef_and_scale(y,yb,z,Wr,rowsX,colsX,
                                               X,Xb,family,S_emb,penalties,n_c)

   # Final penalty
   if len(penalties) > 0:
      penalty = coef.T @ S_emb @ coef
   else:
      penalty = 0

   # Final term edf
   if not term_edfs is None:
      term_edfs = calculate_term_edf(penalties,term_edfs)

   return coef,eta,wres,scale,InvCholXXS,total_edf,term_edfs,penalty