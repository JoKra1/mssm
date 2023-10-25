import numpy as np
import scipy as scp
import warnings
from .exp_fam import Family,Gaussian
from .penalties import PenType,id_dist_pen,translate_sparse
from ..cpp import cpp_solvers

def cpp_chol(A):
   return cpp_solvers.chol(A)

def cpp_qr(A):
   return cpp_solvers.pqr(A)

def cpp_solve_am(y,X,S):
   return cpp_solvers.solve_am(y,X,S)

def step_fellner_schall_sparse(gInv,emb_SJ,Bps,cCoef,cLam,sigma,verbose=True):
  # Perform a generalized Fellner Schall update step for a lambda term. This update rule is
  # discussed in Wood & Fasiolo (2016) and used here because of it's efficiency.
  
  num = max(0,(gInv @ emb_SJ).trace() - Bps)
  denom = max(0,cCoef.T @ emb_SJ @ cCoef)
  nLam = sigma * (num / denom) * cLam
  nLam = max(nLam,1e-7) # Prevent Lambda going to zero
  nLam = min(nLam,1e+7) # Prevent overflow

  if verbose:
   print(f"Num = {(gInv @ emb_SJ).trace()} - {Bps} == {num}\nDenom = {denom}; Lambda = {nLam}")

  return nLam-cLam

def compute_S_emb_pinv_det(col_S,penalties,pinv):
   # Computes final S multiplied with lambda
   # and the pseudo-inverse of this term.
   cIndexPinv = 0
   if penalties[0].start_index is not None:
      cIndexPinv = penalties[0].start_index
      
   n_lterms = len(penalties)
   S_emb = None
   S_pinv_elements = []
   S_pinv_rows = []
   S_pinv_cols = []

   SJ = None
   SJ_terms = 0
   
   # Build S_emb and pinv(S_emb)
   for lti,lTerm in enumerate(penalties):
      #print(f"pen: {lti}")

      # Collect number of penalties on the term
      SJ_terms += 1

      # We need to compute the pseudo-inverse on the penalty block (so sum of all
      # penalties weighted by lambda) for the term so we first add all penalties
      # on this term together.
      if SJ is None:
         SJ = lTerm.S_J*lTerm.lam
      else:
         SJ += lTerm.S_J*lTerm.lam

      # Add S_J_emb * lambda to the overall S_emb
      if lti == 0:
         S_emb = lTerm.S_J_emb * lTerm.lam
      else:
         S_emb += lTerm.S_J_emb * lTerm.lam

      if lti < (n_lterms - 1) and penalties[lti + 1].start_index is not None:
         #print("Skip")
         continue
      else: # Now handle all pinv calculations because all penalties associated with a term have been collected in SJ
         #print(f"repeat: {lTerm.rep_sj}")
         #print(f"number of penalties on term: {SJ_terms}")
         if SJ_terms == 1 and lTerm.type == PenType.IDENTITY:
            #print("Identity shortcut")
            SJ_pinv_elements,SJ_pinv_rows,SJ_pinv_cols,_,_,_ = id_dist_pen(SJ.shape[1],lambda x: 1/lTerm.lam)
         else:
            # Compute pinv(SJ) via cholesky factor L so that L @ L' = SJ' @ SJ.
            # If SJ is full rank, then pinv(SJ) = inv(L)' @ inv(L) @ SJ'.
            # However, SJ' @ SJ will usually be rank deficient so we add e=1e-7 to the main diagonal, because:
            # As e -> 0 we get closer to pinv(SJ) if we base it on L_e in L_e @ L_e' = SJ' @ SJ + e*I
            # where I is the identity.
            if pinv != "svd":
               t = scp.sparse.identity(SJ.shape[1],format="csc")
               L,code = cpp_chol(SJ.T @ SJ + 1e-10*t)

               if code != 0:
                  warnings.warn(f"Cholesky factor computation failed with code {code}")

               LI = scp.sparse.linalg.spsolve(L,t)
               SJ_pinv = LI.T @ LI @ SJ.T
               #print(np.max(abs((SJ.T @ SJ @ SJ_pinv - SJ).toarray())))

            # Compute pinv via SVD
            else:
               u, Sig, vt = scp.sparse.linalg.svds(SJ,SJ.shape[1]-1)
               rs = [1/s if s > 1e-7 else 0 for s in Sig]
               SJ_pinv = vt.T @ np.diag(rs) @ u.T
               SJ_pinv = scp.sparse.csc_array(SJ_pinv)

            SJ_pinv_elements,SJ_pinv_rows,SJ_pinv_cols = translate_sparse(SJ_pinv)
         
         SJ_pinv_rows = np.array(SJ_pinv_rows)
         SJ_pinv_cols = np.array(SJ_pinv_cols)

         for _ in range(lTerm.rep_sj):
            S_pinv_elements.extend(SJ_pinv_elements)
            S_pinv_rows.extend(SJ_pinv_rows + cIndexPinv)
            S_pinv_cols.extend(SJ_pinv_cols + cIndexPinv)
            cIndexPinv += (SJ_pinv_cols[-1] + 1)
         
         # Reset number of penalties and SJ
         SJ = None
         SJ_terms = 0

   S_pinv = scp.sparse.csc_array((S_pinv_elements,(S_pinv_rows,S_pinv_cols)),shape=(col_S,col_S))
   return S_emb, S_pinv

def PIRLS_pdat_weights(y,mu,eta,family:Family):
   # Compute pseudo-data and weights for Penalized Reweighted Least Squares iteration (Wood, 2017, 6.1.1)
   # Calculation is based on a(mu) = 1, so reflects Fisher scoring!
   dy1 = family.link.dy1(mu)
   z = dy1 * (y - mu) + eta
   w = 1 / (dy1**2 * family.V(mu))
   return z, w

def PIRLS(y,yb,mu,eta,X,Xb,S_emb,family,maxiter_inner):
   # Perform Penalized Iterative Reweighted Least Squares for current lambda parameter (Wood, 2017,6.1.1)
   prev_coef = None
   z = None
   Wr = None
   for i_iter in range(maxiter_inner):

      if isinstance(family,Gaussian) == False:
         # Compute weights and pseudo-dat
         z, w = PIRLS_pdat_weights(y,mu,eta,family)

         Wr = scp.sparse.spdiags([np.sqrt(np.ndarray.flatten(w))],[0])

         # Update yb and Xb
         yb = Wr @ z
         Xb = Wr @ X
      
      # Solve additive model
      InvCholXXSP, Pr, coef, code = cpp_solve_am(yb,Xb,S_emb)

      if code != 0:
         raise ArithmeticError(f"Solving for coef failed with code {code}. Model is likely unidentifiable.")
      
      # Update mu & eta
      eta = (X @ coef).reshape(-1,1)
      mu = eta

      if isinstance(family,Gaussian) == False:
         mu = family.link.fi(eta)
      
   return y,yb,mu,eta,X,Xb,z,Wr,InvCholXXSP,Pr,coef

def solve_gamm_sparse(mu_init,y,X,penalties,col_S,family:Family,maxiter_outer=10,maxiter_inner=1,pinv="svd"):
   # Estimates a penalized Generalized additive mixed model, following the steps outlined in Wood (2017)
   rowsX,colsX = X.shape
   coef = None
   prev_lam_delta = None

   # Additive mixed model can simply be fit on y and X
   # Generalized mixedl model needs to be fit on weighted X and pseudo-dat
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
      S_emb,S_pinv = compute_S_emb_pinv_det(col_S,penalties,pinv)
   else:
      S_emb = scp.sparse.csc_array((colsX, colsX), dtype=np.float64)

   # Outer Loop - to optimize smoothing parameter
   for o_iter in range(maxiter_outer):
      
      # Inner Loop to optionally update pseudo data and weights i.e.,
      # to perform PIRLS (Wood, 2017) or to simply calculate an AMM
      y,yb,mu,eta,X,Xb,z,Wr,InvCholXXSP,Pr,coef = PIRLS(y,yb,mu,eta,X,Xb,S_emb,family,maxiter_inner)

      # Calculate Pearson residuals for GAMM (Wood, 3.1.7)
      # Standard residuals for AMM
      if isinstance(family,Gaussian) == False:
         wres = Wr @ (z - eta)
      else:
         wres = y - eta

      # Calculate total and term wise edf
      nP = len(Pr)
      P = [1 for _ in range(nP)]
      Pc = [c for c in range(nP)]
      Perm = scp.sparse.csc_array((P,(Pr,Pc)),shape=(nP,nP))
      InvCholXXS = InvCholXXSP @ Perm

      total_edf = colsX

      # If there are penalized terms we need to adjust the total_edf
      if len(penalties) > 0:
         Bs = []
         term_edfs = []

         for lTerm in penalties:
            B = InvCholXXS @ lTerm.D_J_emb # Needed for Fellner Schall update (Wood & Fasiolo, 2016)
            Bps = B.power(2).sum()
            pen_params = lTerm.lam * Bps
            total_edf -= pen_params
            Bs.append(Bps)
            term_edfs.append(lTerm.S_J.shape[1] - pen_params)
      else:
         term_edfs = None

      # Optionally estimate scale parameter
      if family.scale is None:
         scale = family.est_scale(wres,rowsX,total_edf)
      else:
         scale = family.scale

      # Perform Fellner Schall Update if there are penalized terms:
      if len(penalties) > 0:
         lam_delta = []
         for lti,lTerm in enumerate(penalties):
            dLam = step_fellner_schall_sparse(S_pinv,lTerm.S_J_emb,Bs[lti],coef,lTerm.lam,scale)
            lam_delta.append(dLam)

         # ToDo: Optionally test lambda step - right now just apply
         for lti,lTerm in enumerate(penalties):
            lTerm.lam += lam_delta[lti]
         
         # ToDo: Convergence control
         lam_delta = np.array(lam_delta).reshape(-1,1)
         prev_lam_delta = lam_delta[:]

         # Re-compute S_emb and S_pinv
         S_emb,S_pinv = compute_S_emb_pinv_det(col_S,penalties,pinv)

   # Final penalty
   if len(penalties) > 0:
      penalty = coef.T @ S_emb @ coef
   else:
      penalty = 0

   return coef,eta,wres,scale,InvCholXXS,total_edf,term_edfs,penalty