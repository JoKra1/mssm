import numpy as np
import scipy as scp
from .exp_fam import Family,Gaussian,est_scale,GAMLSSFamily,GSMMFamily,Identity,warnings
from .penalties import embed_in_S_sparse
from .formula import build_sparse_matrix_from_formula,setup_cache,clear_cache,pd,Formula,mp,repeat,os,math,tqdm,sys,copy
from .compact_rep import computeH, computeHSR1, computeV, computeVSR1
from .repara import reparam_model
from functools import reduce
from .custom_types import Fit_info,LambdaTerm
from .terms import GammTerm
from .matrix_solvers import *
from collections.abc import Callable

CACHE_DIR = './.db'
SHOULD_CACHE = False
MP_SPLIT_SIZE = 2000


def compute_lgdetD_bsb(rank:int|None,cLam:float,gInv:scp.sparse.csc_array,emb_SJ:scp.sparse.csc_array,cCoef:np.ndarray) -> tuple[float,float]:
   """Internal function. Computes derivative of :math:`log(|\\mathbf{S}_\\lambda|_+)`, the log of the "Generalized determinant", with respect to lambda.
   
   See Wood, Shaddick, & Augustin, (2017) and Wood & Fasiolo (2017), and Wood (2017), and Wood (2011)

   References:
      - Wood, S. N., Li, Z., Shaddick, G., & Augustin, N. H. (2017). Generalized Additive Models for Gigadata: Modeling the U.K. Black Smoke Network Daily Data. https://doi.org/10.1080/01621459.2016.1195744
      - Wood, S. N., & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. https://doi.org/10.1111/biom.12666
      - Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models: Estimation of Semiparametric Generalized Linear Models. https://doi.org/10.1111/j.1467-9868.2010.00749.x
      - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

   :param rank: Known rank of penalty matrix or None (should only be set to int for single penalty terms)
   :type rank: int | None
   :param cLam: Current lambda value
   :type cLam: float
   :param gInv: Generalized inverse of total penalty matrix
   :type gInv: scp.sparse.csc_array
   :param emb_SJ: Embedded penalty matrix
   :type emb_SJ: scp.sparse.csc_array
   :param cCoef: coefficient vector
   :type cCoef: np.ndarray
   :return: Tuple, first element is aforementioned derivative, second is ``cCoef.T@emb_SJ@cCoef``
   :rtype: tuple[float,float]
   """
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

def step_fellner_schall_sparse(lgdet_deriv:float,ldet_deriv:float,bSb:float,cLam:float,scale:float) -> float:
  """Internal function. Compute a generalized Fellner Schall update step for a lambda term. This update rule is discussed in Wood & Fasiolo (2017).

   References:
      - Wood, S. N., & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. https://doi.org/10.1111/biom.12666
      - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

   :param lgdet_deriv: Derivative of :math:`log(|\\mathbf{S}_\\lambda|_+)`, the log of the "Generalized determinant", with respect to lambda.
   :type lgdet_deriv: float
   :param ldet_deriv: Derivative of :math:`log(|\\mathbf{H} + S_\\lambda|)` (:math:`\\mathbf{X}` is negative hessian of penalized llk) with respect to lambda.
   :type ldet_deriv: float
   :param bSb: ``cCoef.T@emb_SJ@cCoef`` where ``cCoef`` is current coefficient estimate
   :type bSb: float
   :param cLam: Current lambda value
   :type cLam: float
   :param scale: Optional scale parameter (or 1)
   :type scale: float
   :return: The additive update to ``cLam``
   :rtype: float
  """
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

def grad_lambda(lgdet_deriv:float,ldet_deriv:float,bSb:float,scale:float) -> np.ndarray:
   """Internal function. Computes gradient of REML criterion with respect to all lambda paraemters.

   References:
      - Wood, S. N., & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. https://doi.org/10.1111/biom.12666
      - Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models: Estimation of Semiparametric Generalized Linear Models. https://doi.org/10.1111/j.1467-9868.2010.00749.x
      - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

   :param lgdet_deriv: Derivative of :math:`log(|\\mathbf{S}_\\lambda|_+)`, the log of the "Generalized determinant", with respect to lambda.
   :type lgdet_deriv: float
   :param ldet_deriv: Derivative of :math:`log(|\\mathbf{H} + S_\\lambda|)` (:math:`\\mathbf{X}` is negative hessian of penalized llk) with respect to lambda.
   :type ldet_deriv: float
   :param bSb: ``cCoef.T@emb_SJ@cCoef`` where ``cCoef`` is current coefficient estimate
   :type bSb: float
   :param scale: Optional scale parameter (or 1)
   :type scale: float
   :return: The gradient of the reml criterion
   :rtype: np.ndarray
   """
   # P. Deriv of restricted likelihood with respect to lambda.
   # From Wood & Fasiolo (2016)
   return lgdet_deriv/2 - ldet_deriv/2 - bSb / (2*scale)

def compute_S_emb_pinv_det(col_S:int,penalties:list[LambdaTerm],pinv:str,root:bool=False) -> tuple[scp.sparse.csc_array, scp.sparse.csc_array, scp.sparse.csc_array|None, list[bool]]:
   """Internal function. Compute the total embedded penalty matrix, a generalized inverse of the former, optionally a root of the total penalty matrix, and determines for which EFS updates the rank rather than the generalized inverse should be used.


   :param col_S: Number of columns of total penalty matrix
   :type col_S: int
   :param penalties: List of penalties
   :type penalties: list[LambdaTerm]
   :param pinv: Strategy to use to compute the generalized inverse. Set this to 'svd'.
   :type pinv: str
   :param root: Whther to compute a root of the generalized inverse, defaults to False
   :type root: bool, optional
   :return: A tuple holding total embedded penalty matrix, a generalized inverse of the former, optionally a root of the total penalty matrix, and a list of bools indicating for which EFS updates the rank rather than the generalized inverse should be used
   :rtype: tuple[scp.sparse.csc_array, scp.sparse.csc_array, scp.sparse.csc_array|None, list[bool]]
   """
   # Computes final S multiplied with lambda
   # and the pseudo-inverse of this term. Optionally, the matrix
   # root of this term can be computed as well.
   S_emb = None

   # We need to compute the pseudo-inverse on the penalty block (so sum of all
   # penalties weighted by lambda) for every term so we first collect and sum
   # all term penalties together.
   SJs = [] # Summed SJs for every term
   DJs = [] # None if summation, otherwise embedded DJ*sqrt(\\lambda) - root of corresponding SJ*\\lambda
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
         DJs.append(lTerm.D_J_emb*math.sqrt(lTerm.lam))
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
         DJs[idx_match[0]] = None
         if SJ_reps[idx_match[0]] != lTerm.rep_sj:
            raise ValueError("Repeat Number for penalty does not match previous penalties repeat number.")

   #print(SJ_idx_len)
   #print(SJ_reps,SJ_lams,SJ_idx)
   S_pinv_elements = []
   S_pinv_rows = []
   S_pinv_cols = []
   cIndexPinv = SJ_idx[0]

   # Optionally compute root of S_\\lambda
   S_root_singles = None
   S_root_elements = []
   S_root_rows = []
   S_root_cols = []

   FS_use_rank = []
   for SJi in range(SJ_idx_len):
      # Now handle all pinv calculations because all penalties
      # associated with a term have been collected in SJ

      if SJ_terms[SJi] == 1:

         if root:
            if S_root_singles is None:
               S_root_singles = copy.deepcopy(DJs[SJi])
            else:
               S_root_singles += DJs[SJi]

         # No overlap between penalties so we can just leave this block in the
         # pseudo-inverse empty.
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
            #print((SJs[SJi]@SJ_pinv@SJs[SJi] - SJs[SJi]).min(),(SJs[SJi]@SJ_pinv@SJs[SJi] - SJs[SJi]).max())
         
         if root:
            eig, U =scp.linalg.eigh(SJs[SJi].toarray())
            D_root = scp.sparse.csc_array(U@np.diag([e**0.5 if e > sys.float_info.epsilon**0.7 else 0 for e in eig]))
            
            Dl_dat, Dl_rows, Dl_cols = translate_sparse(D_root)
            Dl_rows = np.array(Dl_rows)
            Dl_cols = np.array(Dl_cols)

         SJ_pinv_elements,SJ_pinv_rows,SJ_pinv_cols = translate_sparse(SJ_pinv)

         SJ_pinv_rows = np.array(SJ_pinv_rows)
         SJ_pinv_cols = np.array(SJ_pinv_cols)

         SJ_pinv_shape = SJs[SJi].shape[1]
         for _ in range(SJ_reps[SJi]):
            S_pinv_elements.extend(SJ_pinv_elements)
            S_pinv_rows.extend(SJ_pinv_rows + cIndexPinv)
            S_pinv_cols.extend(SJ_pinv_cols + cIndexPinv)

            if root:
               S_root_elements.extend(Dl_dat)
               S_root_rows.extend(Dl_rows + cIndexPinv)
               S_root_cols.extend(Dl_cols + cIndexPinv)

            cIndexPinv += SJ_pinv_shape
         
         for _ in range(SJ_terms[SJi]):
            FS_use_rank.append(False)
         
   S_pinv = scp.sparse.csc_array((S_pinv_elements,(S_pinv_rows,S_pinv_cols)),shape=(col_S,col_S))

   S_root = None
   if root:
      S_root = scp.sparse.csc_array((S_root_elements,(S_root_rows,S_root_cols)),shape=(col_S,col_S))
      if not S_root_singles is None:
         S_root += S_root_singles

   if len(FS_use_rank) != len(penalties):
      raise IndexError("An incorrect number of rank decisions were made.")
   
   return S_emb, S_pinv, S_root, FS_use_rank

def PIRLS_pdat_weights(y:np.ndarray,mu:np.ndarray,eta:np.ndarray,family:Family) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
   """Internal function. Compute pseudo-data and weights for Penalized Reweighted Least Squares iteration (Wood, 2017, 6.1.1)

   Calculation is based on a(mu) = 1, so reflects Fisher scoring!

   References:
      - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

   :param y: vector of observations
   :type y: np.ndarray
   :param mu: vector of mean estimates
   :type mu: np.ndarray
   :param eta: vector of linear predictors
   :type eta: np.ndarray
   :param family: Family of model
   :type family: Family
   :raises ValueError: If not a single observation provided information for Fisher weights.
   :return: the pesudo-data, weights, and a boolean array indicating invalid weights/pseudo-observations
   :rtype: tuple[np.ndarray,np.ndarray,np.ndarray]
   """

   with warnings.catch_warnings(): # Catch divide by 0 in w and errors in dy1 computation
      warnings.simplefilter("ignore")
      dy1 = family.link.dy1(mu)
      z = dy1 * (y - mu) + eta
      w = 1 / (np.power(dy1,2) * family.V(mu))

   # Prepare to take steps, if any of the weights or pseudo-data become nan or inf
   invalid_idx = np.isnan(w) | np.isnan(z) | np.isinf(w) | np.isinf(z)

   if np.sum(invalid_idx) == len(y):
      raise ValueError("Not a single observation provided information for Fisher weights.")

   z[invalid_idx] = np.nan # Make sure this is consistent, for scale computation

   return z, w, invalid_idx.flatten()

def PIRLS_newton_weights(y:np.ndarray,mu:np.ndarray,eta:np.ndarray,family:Family)  -> tuple[np.ndarray,np.ndarray,np.ndarray]:
   """Internal function. Compute pseudo-data and newton weights for Penalized Reweighted Least Squares iteration (Wood, 2017, 6.1.1 and 3.1.2)

   Calculation reflects full Newton scoring!

   References:
      - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

   :param y: vector of observations
   :type y: np.ndarray
   :param mu: vector of mean estimates
   :type mu: np.ndarray
   :param eta: vector of linear predictors
   :type eta: np.ndarray
   :param family: Family of model
   :type family: Family
   :raises ValueError: If not a single observation provided information for newton weights.
   :return: the pesudo-data, weights, and a boolean array indicating invalid weights/pseudo-observations
   :rtype: tuple[np.ndarray,np.ndarray,np.ndarray]
   """

   with warnings.catch_warnings(): # Catch divide by 0 in w and errors in dy1 computation
      warnings.simplefilter("ignore")
      dy1 = family.link.dy1(mu)
      dy2 = family.link.dy2(mu)
      V = family.V(mu)
      dVy1 = family.dVy1(mu)

      # Compute a(\mu) as shown in section 3.1.2 of Wood (2017)
      a = 1 + (y - mu) * (dVy1/V + dy2/dy1)

      z = (dy1 * (y - mu)/a) + eta
      w = a / (np.power(dy1,2) * V)

   # Prepare to take steps, if any of the weights or pseudo-data become nan or inf
   invalid_idx = np.isnan(w) | np.isnan(z) | np.isinf(w) | np.isinf(z)

   if np.sum(invalid_idx) == len(y):
      raise ValueError("Not a single observation provided information for Newton weights.")

   z[invalid_idx] = np.nan # Make sure this is consistent, for scale computation

   return z, w, invalid_idx.flatten()

def update_PIRLS(y:np.ndarray,yb:np.ndarray,mu:np.ndarray,eta:np.ndarray,X:scp.sparse.csc_array,Xb:scp.sparse.csc_array,family:Family,Lrhoi:scp.sparse.csc_array|None) -> tuple[np.ndarray,scp.sparse.csc_array,np.ndarray|None,scp.sparse.csc_array|None]:
   """Internal function. Updates the pseudo-weights and observation vector ``yb`` and model matrix ``Xb`` of the working model.

   **Note**: Dimensions of ``yb`` and ``Xb`` might not match those of ``y`` and ``X`` since rows of invalid pseudo-data observations are dropped here.

   :param y: vector of observations
   :type y: np.ndarray
   :param yb: vector of observations of the working model
   :type yb: np.ndarray
   :param mu: vector of mean estimates
   :type mu: np.ndarray
   :param eta: vector of linear predictors
   :type eta: np.ndarray
   :param X: Model matrix
   :type X: scp.sparse.csc_array
   :param Xb: Model matrix of working model
   :type Xb: scp.sparse.csc_array
   :param family: Family of model
   :type family: Family
   :param Lrhoi: Optional covariance matrix of an ar1 model
   :type Lrhoi: scp.sparse.csc_array | None
   :return: Updated observation vector ``yb`` and model matrix ``Xb`` of the working model, pseudo-weights, and a diagonal sparse matrix holding the **root** of the Fisher weights. Latter two are None for strictly additive models.
   :rtype: tuple[np.ndarray,scp.sparse.csc_array,np.ndarray|None,scp.sparse.csc_array|None]
   """
   # Update the PIRLS weights and data (if the model is not Gaussian)
   # and update the fitting matrices yb & Xb
   z = None
   Wr = None

   if isinstance(family,Gaussian) == False or isinstance(family.link,Identity) == False:
      # Compute weights and pseudo-dat - drop any rows for which z or w is not defined.
      z, w, inval = PIRLS_pdat_weights(y,mu,eta,family)

      Wr = scp.sparse.spdiags([np.sqrt(np.ndarray.flatten(w[inval == False]))],[0])

      # Update yb and Xb
      yb = Wr @ z[inval == False]
      Xb = (Wr @ X[inval == False,:]).tocsc()

      if Lrhoi is not None:
         # Apply ar1 model to working linear model.
         # Basically z_i ~ N(X@b,W^-1*phi), where W=Wr@Wr
         # Now: Wr@z_i = Wr@X@b + e and instead of assuming e ~ N(0,\\phi)
         # We Assume e ~ N(0,V\\phi) where V=Lrho^T@Lrho is covariance matrix of ar1 model.
         # Let Lhroi = Lrho^-1. Then: Lhroi^T@Wr@z_i = Lhroi^T@Wr@X@b + eps
         # with eps ~ N(0,\\phi)
         # see section 1.8.4 of Wood (2017)
         
         Lrhoiv = Lrhoi[inval == False,:]
         Lrhoiv = Lrhoiv[:,inval == False]

         yb = Lrhoiv.T@yb # Lhroi^T@Wr@z_i
         Xb = Lrhoiv.T@Xb #Lhroi^T@Wr@X
         Xb.sort_indices()
      
   return yb,Xb,z,Wr

def compute_eigen_perm(Pr:list[int]) -> scp.sparse.csc_array:
   """Internal function. Computes column permutation matrix obtained from Eigen.

   :param Pr: List of column indices
   :type Pr: list[int]
   :return: Permutation matrix as sparse array
   :rtype: scp.sparse.csc_array
   """

   nP = len(Pr)
   P = [1 for _ in range(nP)]
   Pc = [c for c in range(nP)]
   Perm = scp.sparse.csc_array((P,(Pr,Pc)),shape=(nP,nP))
   return Perm

def apply_eigen_perm(Pr:list[int],InvCholXXSP:scp.sparse.csc_array) -> scp.sparse.csc_array:
   """Internal function. Unpivots columns of ``InvCholXXSP`` (usually the inverse of a Cholesky factor) and returns the unpivoted version.

   :param Pr: List of column indices
   :type Pr: list[int]
   :param InvCholXXSP: Pivoted matrix
   :type InvCholXXSP: scp.sparse.csc_array
   :return: Unpivoted matrix
   :rtype: scp.sparse.csc_array
   """
   Perm = compute_eigen_perm(Pr)
   InvCholXXS = InvCholXXSP @ Perm
   return InvCholXXS

def computetrVS3(t1:np.ndarray|None,t2:np.ndarray|None,t3:np.ndarray|None,lTerm:LambdaTerm,V0:scp.sparse.csc_array) -> float:
   """Internal function. Compute ``tr(V@lTerm.S_j)`` from linear operator of ``V`` obtained from L-BFGS-B optimizer.

   Relies on equation 3.13 in Byrd, Nocdeal & Schnabel (1992). Adapted to ensure positive semi-definitiness required
   by EFS update.

   References:
      - Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994). Representations of quasi-Newton matrices and their use in limited memory methods. Mathematical Programming, 63(1), 129–156. https://doi.org/10.1007/BF01582063

   :param t1: ``nCoef*2m``  matrix from Byrd, Nocdeal & Schnabel (1992). If ``t2 is None``, then ``V`` is treated like an identity matrix.
   :type t1: np.ndarray or None
   :param t2: ``2m*2m``  matrix from Byrd, Nocdeal & Schnabel (1992). If ``t2 is None``, then ``V`` is treated like an identity matrix.
   :type t2: np.ndarray or None
   :param t3: ``2m*nCoef``  matrix from Byrd, Nocdeal & Schnabel (1992). If ``t2 is None``, then ``t1`` is treated like an identity matrix.
   :type t3: np.ndarray or None
   :param lTerm: Current lambda term for which to compute the trace.
   :type lTerm: LambdaTerm
   :param V0: Initial estimate for the inverse of the hessian fo the negative penalized likelihood.
   :type V0: scipy.sparse.csc_array
   :return: trace
   :rtype: float
   """

   tr = 0

   S_start = lTerm.start_index
   S_len = lTerm.rep_sj * lTerm.S_J.shape[1]
   S_end = S_start + S_len
   
   for cidx in range(S_start,S_end):
      S_c = lTerm.S_J_emb[:,[cidx]] # Can remain sparse

      # Now compute product with vector in compact form
      if not t2 is None:
         #print((V0@S_c)[[cidx],[0]][0],lTerm.S_J_emb[[cidx],[cidx]])
         VS_c = (V0@S_c)[[cidx],[0]][0] - t1[[cidx],:]@t2@t3@S_c
      
      else:
         VS_c = np.array((V0@S_c)[[cidx],[0]][0])

      tr += VS_c[0,0]

   return tr

def calculate_edf(LP:scp.sparse.csc_array|None,Pr:list[int],InvCholXXS:scp.sparse.csc_array|scp.sparse.linalg.LinearOperator|None,
                  penalties:list[LambdaTerm],lgdetDs:list[float]|None,colsX:int,n_c:int,drop:list[int]|None,S_emb:scp.sparse.csc_array) -> tuple[float,list[float],list[scp.sparse.csc_array]]:
   """Internal function. Follows steps outlined by Wood & Fasiolo (2017) to compute total degrees of freedom by the model.
   
   Generates the B matrix also required for the derivative of the log-determinant of X.T@X+S_lambda. This
   is either done exactly - as described by Wood & Fasiolo (2017) - or approximately. The latter is much faster.

   Also implements the L-qEFS trace computations described by Krause et al. (submitted) based on a quasi-newton approximation to the negative hessian of the log-likelihood.

   References:
      - Wood, S. N., & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. https://doi.org/10.1111/biom.12666
      - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
      - Krause et al. (submitted). The Mixed-Sparse-Smooth-Model Toolbox (MSSM): Efficient Estimation and Selection of Large Multi-Level Statistical Models. https://doi.org/10.48550/arXiv.2506.13132

   :param LP: Pivoted Cholesky of negative penalzied hessian or None
   :type LP: scp.sparse.csc_array | None
   :param Pr: Permutation list of ``LP``
   :type Pr: list[int]
   :param InvCholXXS: Unpivoted Inverse of ``LP``, or a quasi-newton approximation of it (for the L-qEFS update), or None
   :type InvCholXXS: scp.sparse.csc_array | scp.sparse.linalg.LinearOperator | None
   :param penalties: list of penalties
   :type penalties: list[LambdaTerm]
   :param lgdetDs: list of Derivatives of :math:`log(|\\mathbf{H} + S_\\lambda|)` (:math:`\\mathbf{X}` is negative hessian of penalized llk) with respect to lambda.
   :type lgdetDs: list[float]
   :param colsX: Number of columns of model matrix
   :type colsX: int
   :param n_c: Number of cores to use for computations
   :type n_c: int
   :param drop: List of dropped coefficients - can be None
   :type drop: list[int]
   :param S_emb: Total penalty matrix
   :type S_emb: scp.sparse.csc_array
   :return: A tuple containing the total estimated degrees of freedom, the amount of parameters penalized away by individual penalties in a list, and a list of the aforementioned B matrices
   :rtype: tuple[float,list[float],list[scp.sparse.csc_array]]
   """

   total_edf = colsX
   Bs = []
   term_edfs = []

   if (not InvCholXXS is None) and isinstance(InvCholXXS,scp.sparse.linalg.LinearOperator):
      # Following prepares trace computation via equation 3.13 in Byrd, Nocdeal & Schnabel (1992).
      # First form initial matrices H0/V0.
      # And then get m (number of implicit hessian updates), yk, sk, and rho from V

      #print("omega",max(1,(1/InvCholXXS.omega)/(colsX)))
      omega = InvCholXXS.omega

      # Form S_emb
      #S_emb,_,_,_ = compute_S_emb_pinv_det(colsX,penalties,"svd")

      # And initial approximation for the hessian...
      form = InvCholXXS.form

      H0 = (scp.sparse.identity(colsX,format='csc')*omega) + S_emb
      Lp, Pr, _ = cpp_cholP(H0)
      P = compute_eigen_perm(Pr)

      # ... and the corresponding inverse.
      LVp0 = compute_Linv(Lp,n_c)
      LV0 = apply_eigen_perm(Pr,LVp0)
      V0 = LV0.T @ LV0

      # Reset H0 to just scaled identity
      H0 = (scp.sparse.identity(colsX,format='csc')*omega)
      
      # Now get all the update vectors
      s, y, rho, m = InvCholXXS.sk, InvCholXXS.yk, InvCholXXS.rho, InvCholXXS.n_corrs

      if m == 0: # L-BFGS routine converged after first step

         t1 = None
         t2 = None
         t3 = None

      else:
         
         # Now compute compact representation of V.
         # Below we get int2, with the shifted Eigenvalues so that
         # H = H0 + nt1 @ int2 @ nt3 - where H0 = I*omega + S_emb and I*omega + nt1 @ int2 @ nt3
         # is PSD.
         # Now, using the Woodbury identity:
         # V = (H)^-1 = V0 - V0@nt1@ (int2^-1 + nt3@V0@nt1)^-1 @ nt3@V0
         #
         # since nt3=nt1.T, and int2^-1 = nt2 we have:
         #
         # V = V0 - V0@nt1@ (nt2 + nt1.T@V0@nt1)^-1 @ nt1.t@V0
         #

         # Compute inverse:
         if form != 'SR1':
            nt1,nt2,int2,nt3 = computeH(s,y,rho,H0,explicit=False)

            invt2 = nt2 + nt3@V0@nt1

            U,sv_invt2,VT = scp.linalg.svd(invt2,lapack_driver='gesvd')

            # Nowe we can compute all parts for the Woodbury identy to obtain V
            t2 = VT.T @ np.diag(1/sv_invt2)  @  U.T

            t1 = V0@nt1
            t3 = nt3@V0
         else:
            nt1, int2, nt3 = computeHSR1(s,y,rho,H0,omega,make_psd=True,explicit=False)

            # When using SR1 int2 is potentially singular, so we need a modified Woodbury that accounts for that.
            # This is given by eq. 23 in Henderson & Searle (1981):
            invt2 = np.identity(int2.shape[1]) + int2@nt3@V0@nt1

            U,sv_invt2,VT = scp.linalg.svd(invt2,lapack_driver='gesvd')

            # Nowe we can again compute all parts for the modified Woodbury identy to obtain V
            t2 = VT.T @ np.diag(1/sv_invt2)  @  U.T

            t1 = V0@nt1
            t3 = int2@nt3@V0

         # We now have: V = V0 - V0@nt1@t2@nt3@V0, but don't have to form that explcitly!
         #V3 = V0 - V0@nt1@t2@nt3@V0
         #LVPT, P, code = cpp_cholP(scp.sparse.csc_array(V3))
         #LVT = apply_eigen_perm(P,LVPT)
         #LV = LVT.T


   for lti,lTerm in enumerate(penalties):
      if not InvCholXXS is None:
         # Compute B, needed for Fellner Schall update (Wood & Fasiolo, 2017)
         if isinstance(InvCholXXS,scp.sparse.linalg.LinearOperator):
               Bps = computetrVS3(t1,t2,t3,lTerm,V0)
         else:
            B = InvCholXXS @ lTerm.D_J_emb 
            Bps = B.power(2).sum()
      else:
         Bps = compute_B(LP,compute_eigen_perm(Pr),lTerm,n_c,drop)

         if not lTerm.clust_series is None:
            if Bps[1] < lgdetDs[lti]:
               Bps = Bps[1]
            elif (Bps[1]*0.7 + Bps[0]*0.3) < lgdetDs[lti]:
               Bps = Bps[1]*0.7 + Bps[0]*0.3
            elif (Bps[1] + Bps[0])/2 < lgdetDs[lti]:
               Bps = (Bps[1] + Bps[0])/2
            elif Bps[0] < lgdetDs[lti]:
               Bps = Bps[0]
            else:
               Bps = lgdetDs[lti] - 1e-7

      pen_params = lTerm.lam * Bps
      total_edf -= pen_params
      Bs.append(Bps)
      term_edfs.append(pen_params) # Not actually edf yet - rather the amount of parameters penalized away by individual penalties.
   
   return total_edf,term_edfs,Bs

def calculate_term_edf(penalties:list[LambdaTerm],param_penalized:list[float]) -> list[float]:
   """Internal function. Computes the smooth-term (and random term) specific estimated degrees of freedom.

   See Wood (2017) for a definition and Wood, S. N., & Fasiolo, M. (2017). for the computations.

   References:
      - Wood, S. N., & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. https://doi.org/10.1111/biom.12666
      - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

   :param penalties: List of penalties
   :type penalties: list[LambdaTerm]
   :param param_penalized: List holding the amount of parameters penalized away by individual penalties - obtained from :func:`calculate_edf`.
   :type param_penalized: list[float]
   :return: A list holding the estimated degrees of freedom per smooth/random term in the model
   :rtype: list[float]
   """
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

def update_scale_edf(y:np.ndarray,z:np.ndarray,eta:np.ndarray,Wr:scp.sparse.csc_array,rowsX:int,colsX:int,LP:scp.sparse.csc_array|None,
                     InvCholXXSP:scp.sparse.csc_array|None,Pr:list[int],lgdetDs:list[float],Lrhoi:scp.sparse.csc_array|None,family:Family,
                     penalties:list[LambdaTerm],keep:list[int]|None,drop:list[int],n_c:int) -> tuple[np.ndarray,scp.sparse.csc_array|None,float,list[float],list[scp.sparse.csc_array],float]:
   """Internal function. Updates the scale of the model. For this the edf are computed as well - they are returned as well because they are needed for the lambda step.

   References:
      - Wood, S. N., & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. https://doi.org/10.1111/biom.12666
      - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

   :param y: vector of observations
   :type y: np.ndarray
   :param z: vector of pseudo-data (can contain NaNs for invalid observations)
   :type z: np.ndarray
   :param eta: vector of linear predictors
   :type eta: np.ndarray
   :param Wr: diagonal sparse matrix holding the **root** of the Fisher weights
   :type Wr: scp.sparse.csc_array
   :param rowsX: Rows of model matrix
   :type rowsX: int
   :param colsX: Cols of model matrix
   :type colsX: int
   :param LP: Pivoted Cholesky of negative penalzied hessian or None
   :type LP: scp.sparse.csc_array | None
   :param InvCholXXSP: Inverse of ``LP``, or None
   :type InvCholXXSP: scp.sparse.csc_array | None
   :param Pr: Permutation list of ``LP``
   :type Pr: list[int]
   :param lgdetDs: List of derivatives of :math:`log(|\\mathbf{S}_\\lambda|_+)`, the log of the "Generalized determinant", with respect to lambdas.
   :type lgdetDs: list[float]
   :param Lrhoi: Optional covariance matrix of an ar1 model
   :type Lrhoi: scp.sparse.csc_array | None
   :param family: Family of model
   :type family: Family
   :param penalties: List of penalties
   :type penalties: list[LambdaTerm]
   :param keep: List of coefficients to keep, can be None -> keep all
   :type keep: list[int] | None
   :param drop: List of coefficients to drop
   :type drop: list[int]
   :param n_c: Number of cores to use
   :type n_c: int
   :return: a tuple containing the working residuals, optionally the unpivoted inverse of ``LP``, total edf, term-wise edf, Bs, scale estimate
   :rtype: tuple[np.ndarray, scp.sparse.csc_array|None, float, list[float], list[scp.sparse.csc_array], float]
   """
   # Updates the scale of the model. For this the edf
   # are computed as well - they are returned because they are needed for the
   # lambda step proposal anyway.
   
   # Calculate Pearson/working residuals for GAMM (Wood, 3.1.5 & 3.1.7)
   # Standard residuals for AMM
   dropped = 0
   if isinstance(family,Gaussian) == False or isinstance(family.link,Identity) == False:
      inval = np.isnan(z)
      dropped = np.sum(inval) # Make sure to only take valid z/eta/w here and for computing the scale

      # Apply ar1 model to working residuals 
      if Lrhoi is not None:
         inval_f = (inval == False).flatten()
         Lrhoiv = Lrhoi[inval_f,:]
         Lrhoiv = Lrhoiv[:,inval_f]

         wres = Lrhoiv.T@ (Wr @ (z[inval == False] - eta[inval == False]).reshape(-1,1))
      else:
         wres = Wr @ (z[inval == False] - eta[inval == False]).reshape(-1,1)
   else:
      wres = y - eta

   # Calculate total and term wise edf
   InvCholXXS = None
   if not InvCholXXSP is None:
      InvCholXXS = apply_eigen_perm(Pr,InvCholXXSP)

      # Dropped some terms, need to insert zero columns and rows for dropped coefficients
      if InvCholXXS.shape[1] < colsX:
         Linvdat,Linvrow,Linvcol = translate_sparse(InvCholXXS)
      
         Linvrow = keep[Linvrow]
         Linvcol = keep[Linvcol]

         InvCholXXS = scp.sparse.csc_array((Linvdat,(Linvrow,Linvcol)),shape=(colsX,colsX))

   # If there are penalized terms we need to adjust the total_edf - make sure to subtract dropped coef from colsX
   if len(penalties) > 0:
      total_edf, term_edfs, Bs = calculate_edf(LP,Pr,InvCholXXS,penalties,lgdetDs,colsX-(len(drop) if drop is not None else 0),n_c,drop,None)
   else:
      total_edf = colsX
      term_edfs = None
      Bs = None

   # Optionally estimate scale parameter
   if family.scale is None:
      scale = est_scale(wres,rowsX - dropped,total_edf)
   else:
      scale = family.scale
   
   return wres,InvCholXXS,total_edf,term_edfs,Bs,scale

def update_coef(yb:np.ndarray,X:scp.sparse.csc_array,Xb:scp.sparse.csc_array,family:Family,
                S_emb:scp.sparse.csc_array,S_root:scp.sparse.csc_array|None,n_c:int,
                formula:Formula|None,offset:float|np.ndarray) ->tuple[np.ndarray,np.ndarray,np.ndarray,list[int],scp.sparse.csc_array,scp.sparse.csc_array,list[int]|None,list[int]|None]:
   """Internal function. Estimates the coefficients of the model and updates the linear predictor and mean estimates.

   References:
      - Wood, S. N., & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. https://doi.org/10.1111/biom.12666
      - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.). 

   :param yb: vector of observations of the working model
   :type yb: np.ndarray
   :param X: Model matrix
   :type X: scp.sparse.csc_array
   :param Xb: Model matrix of working model
   :type Xb: scp.sparse.csc_array
   :param family: Family of Model
   :type family: Family
   :param S_emb: Total penalty matrix
   :type S_emb: scp.sparse.csc_array
   :param S_root: Root of total penalty matrix or None
   :type S_root: scp.sparse.csc_array | None
   :param n_c: Number of cores
   :type n_c: int
   :param formula: Formula of model or None
   :type formula: Formula | None
   :param offset: Offset (fixed effect) to add to ``eta``
   :type offset: float | np.ndarray
   :return: A tuple containing the linear predictor ``eta``, the estimated means ``mu``, the estimated coefficients, the column permutation indices ``Pr``, the column permutation matirx ``P``, the cholesky of the pivoted penalized negative hessian, an optional list of the coefficients to keep, an optional list of the estimated coefficients to drop
   :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, list[int], scp.sparse.csc_array, scp.sparse.csc_array, list[int]|None, list[int]|None]
   """
   # Solves the coefficients of an additive model, given weights and penalty.
   keep = None
   drop = None
   if formula is None:
      if S_root is None:
         LP, Pr, coef, code = cpp_solve_coef(yb,Xb,S_emb)
         P = compute_eigen_perm(Pr)

      else: # Qr-based
         RP,Pr1,Pr2,coef,rank,code = cpp_solve_coef_pqr(yb,Xb,S_root.T.tocsc())

         # Need to get overall pivot...
         P1 = compute_eigen_perm(Pr1)
         P2 = compute_eigen_perm(Pr2)
         P = P2.T@P1.T

         # Need to insert zeroes in case of rank deficiency - first insert nans to that we
         # can then easily find dropped coefs.
         if rank < S_emb.shape[1]:
            coef = np.concatenate((coef,[np.nan for _ in range(S_emb.shape[1]-rank)]))
   
         # Can now unpivot coef
         coef = coef @ P

         # And identify which coef was dropped
         idx = np.arange(len(coef))
         drop = idx[np.isnan(coef)]
         keep = idx[np.isnan(coef)==False]

         # Now actually set dropped ones to zero
         coef[drop] = 0

         # Convert R so that rest of code can just continue as with Chol (i.e., L)
         LP = RP.T.tocsc()

         # Keep only columns of Pr/P that belong to identifiable params. So P.T@LP is Cholesky of negative penalized Hessian
         # of model without unidentifiable coef. Important: LP and Pr/P no longer match dimensions of embedded penalties
         # after this! So we need to keep track of that in the appropriate functions (i.e., `calculate_edf` which calls
         # `compute_B` when called with only LP and not Linv).
         P = P[:,keep]
         _,Pr,_ = translate_sparse(P.tocsc())
         P = compute_eigen_perm(Pr)

   else:
      #yb is X.T@y and Xb is X.T@X
      LP, Pr, coef, code = cpp_solve_coefXX(yb,Xb + S_emb)
      P = compute_eigen_perm(Pr)
   
   # Update mu & eta
   if formula is None:
      eta = (X @ coef).reshape(-1,1) + offset
   else:
      if formula.keep_cov:
         eta = keep_eta(formula,coef,n_c)
      else:
         eta = []
         for file in formula.file_paths:
            eta_file = read_eta(file,formula,coef,n_c)
            eta.extend(eta_file)
      eta = np.array(eta)

   mu = eta

   if isinstance(family,Gaussian) == False or isinstance(family.link,Identity) == False:
      with warnings.catch_warnings(): # Catch errors with mean computation (e.g., overflow)
         warnings.simplefilter("ignore")
         mu = family.link.fi(eta)
   
   return eta,mu,coef,Pr,P,LP,keep,drop

def update_coef_and_scale(y:np.ndarray,yb:np.ndarray,z:np.ndarray,Wr:scp.sparse.csc_array,rowsX:int,colsX:int,
                          X:scp.sparse.csc_array,Xb:scp.sparse.csc_array,Lrhoi:scp.sparse.csc_array|None,family,
                          S_emb:scp.sparse.csc_array,S_root:scp.sparse.csc_array|None,S_pinv:scp.sparse.csc_array,
                          FS_use_rank:list[bool],penalties:list[LambdaTerm],n_c:int,formula:Formula,
                          form_Linv:bool,offset:float|np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray,scp.sparse.csc_array|None,list[float],list[float],float,list[float],list[scp.sparse.csc_array],float,np.ndarray,list[int]|None,list[int]|None]:
   """Internal function to update the coefficients and (optionally) scale parameter of the model.

   References:
      - Wood, S. N., & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. https://doi.org/10.1111/biom.12666
      - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.). 

   :param y: vector of observations
   :type y: np.ndarray
   :param yb: vector of observations of the working model
   :type yb: np.ndarray
   :param z: vector of pseudo-data (can contain NaNs for invalid observations)
   :type z: np.ndarray
   :param Wr: diagonal sparse matrix holding the **root** of the Fisher weights
   :type Wr: scp.sparse.csc_array
   :param rowsX: Rows of model matrix
   :type rowsX: int
   :param colsX: Cols of model matrix
   :type colsX: int
   :param X: Model matrix
   :type X: scp.sparse.csc_array
   :param Xb: Model matrix of working model
   :type Xb: scp.sparse.csc_array
   :param Lrhoi: Optional covariance matrix of an ar1 model
   :type Lrhoi: scp.sparse.csc_array | None
   :param family: Family of model
   :type family: Family
   :param S_emb: Total penalty matrix
   :type S_emb: scp.sparse.csc_array
   :param S_root: Root of total penalty matrix or None
   :type S_root: scp.sparse.csc_array | None
   :param S_pinv: Generalized inverse of total penalty matrix
   :type S_pinv: scp.sparse.csc_array
   :param FS_use_rank: A list of bools indicating for which EFS updates the rank rather than the generalized inverse should be used
   :type FS_use_rank: list[bool]
   :param penalties: List of penalties
   :type penalties: list[LambdaTerm]
   :param n_c: Number of cores
   :type n_c: int
   :param formula: Formula of the model
   :type formula: Formula
   :param form_Linv: Whether to form the inverse of the cholesky of the negative penalzied hessian or not
   :type form_Linv: bool
   :param offset: Offset (fixed effect) to add to ``eta``
   :type offset: float | np.ndarray
   :return: A tuple containing the linear predictor ``eta``, the estimated means ``mu``, the estimated coefficients, the unpivoted cholesky of the penalized negative hessian, the inverse of the former (optional), derivative of :math:`log(|\\mathbf{S}_\\lambda|_+)` with respect to lambdas, cCoef.T@emb_SJ@cCoef for each SJ, total edf, termwise edf, Bs, scale estimate, working residuals, an optional list of the coefficients to keep, an optional list of the estimated coefficients to drop
   :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, scp.sparse.csc_array|None, list[float], list[float], float, list[float], list[scp.sparse.csc_array], float, np.ndarray, list[int]|None, list[int]|None]
   """
   # Solves the additive model for a given set of weights and penalty
   eta,mu,coef,Pr,P,LP,keep,drop = update_coef(yb,X,Xb,family,S_emb,S_root,n_c,formula,offset)
   
   # Given new coefficients compute lgdetDs and bsbs - needed for REML gradient and EFS step
   lgdetDs = None
   bsbs = None
   if len(penalties) > 0:
      lgdetDs = []
      bsbs = []
      for lti,lTerm in enumerate(penalties):

         lt_rank = None
         if FS_use_rank[lti]:
            lt_rank = lTerm.rank

         lgdetD,bsb = compute_lgdetD_bsb(lt_rank,lTerm.lam,S_pinv,lTerm.S_J_emb,coef)
         lgdetDs.append(lgdetD)
         bsbs.append(bsb)
   
   # Solve for inverse of Chol factor of XX+S
   InvCholXXSP = None
   if form_Linv:
      InvCholXXSP = compute_Linv(LP,n_c)

   # Un-pivot L
   L = P.T@LP

   # Makes sense to insert zero rows and columns here in case of unidentifiable params since L is not used for any solves.
   # InvCholXXSP might also be of reduced size but is not un-pivoted here so we don't need to worry about padding here.
   if LP.shape[1] < S_emb.shape[1]:
      
      Ldat,Lrow,Lcol = translate_sparse(L.tocsc()) # L@L.T = H_pen
      
      Lrow = keep[Lrow]
      Lcol = keep[Lcol]

      L = scp.sparse.csc_array((Ldat,(Lrow,Lcol)),shape=(S_emb.shape[1],S_emb.shape[1]))

   # Update scale parameter - and un-pivot + optionally pad InvCholXXSP
   wres,InvCholXXS,total_edf,term_edfs,Bs,scale = update_scale_edf(y,z,eta,Wr,rowsX,colsX,LP,InvCholXXSP,Pr,lgdetDs,Lrhoi,family,penalties,keep,drop,n_c)
   return eta,mu,coef,L,InvCholXXS,lgdetDs,bsbs,total_edf,term_edfs,Bs,scale,wres,keep,drop

def init_step_gam(y:np.ndarray,yb:np.ndarray,mu:np.ndarray,eta:np.ndarray,rowsX:int,
                  colsX:int,X:scp.sparse.csc_array,Xb:scp.sparse.csc_array,
                  family:Family,col_S:int,penalties:list[LambdaTerm],
                  pinv:str,n_c:int,formula:Formula,form_Linv:bool,
                  method:str,offset:float|np.ndarray,Lrhoi:scp.sparse.csc_array|None) -> tuple[float,float,np.ndarray,np.ndarray,np.ndarray,scp.sparse.csc_array,scp.sparse.csc_array,float,list[float],float,np.ndarray,np.ndarray,scp.sparse.csc_array]:
   """Internal function. Gets initial estimates for a GAM model for coefficients and proposes first lambda update.

   References:
      - Wood, S. N., & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. https://doi.org/10.1111/biom.12666
      - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
      - Krause et al. (submitted). The Mixed-Sparse-Smooth-Model Toolbox (MSSM): Efficient Estimation and Selection of Large Multi-Level Statistical Models. https://doi.org/10.48550/arXiv.2506.13132

   :param y: vector of observations
   :type y: np.ndarray
   :param yb: vector of observations of the working model
   :type yb: np.ndarray
   :param mu: vector of mean estimates
   :type mu: np.ndarray
   :param eta: vector of linear predictors
   :type eta: np.ndarray
   :param rowsX: Rows of model matrix
   :type rowsX: int
   :param colsX: Cols of model matrix
   :type colsX: int
   :param X: Model matrix
   :type X: scp.sparse.csc_array
   :param Xb: Model matrix of working model
   :type Xb: scp.sparse.csc_array
   :param family: Family of model
   :type family: Family
   :param col_S: Cols of penalty matrix
   :type col_S: int
   :param penalties: List of penalties
   :type penalties: list[LambdaTerm]
   :param pinv: Method to use to compute generalzied inverse of total penalty, set to 'svd'!
   :type pinv: str
   :param n_c: Number of cores to use
   :type n_c: int
   :param formula: Formula of the model
   :type formula: Formula
   :param form_Linv: Whether to form the inverse of the cholesky of the negative penalzied hessian or not
   :type form_Linv: bool
   :param method: Which method to use to solve for the coefficients ("Chol" or "Qr")
   :type method: str
   :param offset: Offset (fixed effect) to add to ``eta``
   :type offset: float | np.ndarray
   :param Lrhoi: Optional covariance matrix of an ar1 model
   :type Lrhoi: scp.sparse.csc_array | None
   :return: A tuple containing the deviance ``dev``, penalized deviance ``pen_dev``,eta, mu, coef, CholXXS, InvCholXXS, total_edf, term_edfs, scale, wres, lam_delta, S_emb
   :rtype: tuple[float, float, np.ndarray, np.ndarray, np.ndarray, scp.sparse.csc_array, scp.sparse.csc_array, float, list[float], float, np.ndarray, np.ndarray, scp.sparse.csc_array]
   """
   # Initial fitting iteration without step-length control for gam.

   # Compute starting estimate S_emb and S_pinv
   if len(penalties) > 0:
      S_emb,S_pinv,S_root,FS_use_rank = compute_S_emb_pinv_det(col_S,penalties,pinv,root=method=="QR")
   else:
      S_emb = scp.sparse.csc_array((colsX, colsX), dtype=np.float64)
      S_pinv = None
      S_root = None
      FS_use_rank = None

   # Estimate coefficients for starting lambda
   # We just accept those here - no step control, since
   # there are no previous coefficients/deviance that we can
   # compare the result to.

   # First (optionally, only in the non Gaussian case) compute pseudo-dat and weights:
   yb,Xb,z,Wr = update_PIRLS(y,yb,mu,eta - offset,X,Xb,family,Lrhoi)
   
   # Solve additive model
   eta,mu,coef,\
   CholXXS,\
   InvCholXXS,\
   lgdetDs,\
   bsbs,\
   total_edf,\
   term_edfs,\
   Bs,scale,wres,\
   keep, drop = update_coef_and_scale(y,yb,z,Wr,rowsX,colsX,
                                         X,Xb,Lrhoi,family,S_emb,S_root,S_pinv,
                                         FS_use_rank,penalties,n_c,
                                         formula,form_Linv,offset)
   
   # Deviance under these starting coefficients
   # As well as penalized deviance
   inval = np.isnan(mu).flatten()
   dev = family.deviance(y[inval == False],mu[inval == False])
   pen_dev = dev

   if len(penalties) > 0:
      pen_dev += coef.T @ S_emb @ coef

   # Now propose first lambda extension via the Fellner Schall method
   # by Wood & Fasiolo (2016). Simply don't use an extension term (see Wood & Fasiolo; 2016) for
   # this first update.
   lam_delta = []
   if len(penalties) > 0:
      for lti,lTerm in enumerate(penalties):

         dLam = step_fellner_schall_sparse(lgdetDs[lti],Bs[lti],bsbs[lti],lTerm.lam,scale)
         lam_delta.append(dLam)

      lam_delta = np.array(lam_delta).reshape(-1,1)
   
   return dev,pen_dev,eta,mu,coef,CholXXS,InvCholXXS,total_edf,term_edfs,scale,wres,lam_delta,S_emb


def correct_coef_step(coef:np.ndarray,n_coef:np.ndarray,dev:float,pen_dev:float,
                      c_dev_prev:float,family:Family,eta:np.ndarray,mu:np.ndarray,
                      y:np.ndarray,X:scp.sparse.csc_array,n_pen:float,
                      S_emb:scp.sparse.csc_array,formula:Formula,n_c:int,offset:float|np.ndarray) -> tuple[float,float,np.ndarray,np.ndarray,np.ndarray]:
   """Internal function. Performs step-length control on the coefficient vector.

   References:
      - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
      - Krause et al. (submitted). The Mixed-Sparse-Smooth-Model Toolbox (MSSM): Efficient Estimation and Selection of Large Multi-Level Statistical Models. https://doi.org/10.48550/arXiv.2506.13132


   :param coef: Current coefficient estimate
   :type coef: np.ndarray
   :param n_coef: New coefficient estimate
   :type n_coef: np.ndarray
   :param dev: new deviance
   :type dev: float
   :param pen_dev: new penalized deviance
   :type pen_dev: float
   :param c_dev_prev: previous penalized deviance
   :type c_dev_prev: float
   :param family: Family of model
   :type family: Family
   :param eta: vector of linear predictors - under new coefficient estimate
   :type eta: np.ndarray
   :param mu: vector of mean estimates - under new coefficient estimate
   :type mu: np.ndarray
   :param y: vector of observations of the working model
   :type y: np.ndarray
   :param X: Model matrix of working model
   :type X: scp.sparse.csc_array
   :param n_pen: total penalty under new coefficient estimate
   :type n_pen: float
   :param S_emb: Total penalty matrix
   :type S_emb: scp.sparse.csc_array
   :param formula: Formula of model
   :type formula: Formula
   :param n_c: Number of cores
   :type n_c: int
   :param offset: Offset (fixed effect) to add to ``eta``
   :type offset: float | np.ndarray
   :return: Updated versions of dev,pen_dev,mu,eta,coef
   :rtype: tuple[float,float,np.ndarray,np.ndarray,np.ndarray]
   """
   # Perform step-length control for the coefficients (Step 3 in Wood, 2017)
   corrections = 0
   while pen_dev > c_dev_prev or  (np.isinf(pen_dev) or np.isnan(pen_dev)):
      # Coefficient step did not improve deviance - so correction
      # is necessary.

      if corrections > 30:
         # If we could not find a better coefficient set simply accept
         # previous coefficient
         n_coef = coef
      
      # Make sure to definitely exit
      if corrections > 31:
         break

      # Step halving
      n_coef = (coef + n_coef)/2

      # Update mu & eta for correction
      # Note, Wood (2017) show pseudo-data and weight computation in
      # step 1 - which should be re-visited after the correction, but because
      # mu and eta can change during the correction (due to step halving) and neither
      # the pseudo-data nor the weights are necessary to compute the deviance it makes
      # sense to only compute these once **after** we have completed the coef corrections.
      if formula is None:
         eta = (X @ n_coef).reshape(-1,1) + offset
      else:
         if formula.keep_cov:
            eta = keep_eta(formula,n_coef,n_c)
         else:
            eta = []
            for file in formula.file_paths:
               eta_file = read_eta(file,formula,n_coef,n_c)
               eta.extend(eta_file)
         eta = np.array(eta)

      mu = eta

      if isinstance(family,Gaussian) == False or isinstance(family.link,Identity) == False:
         with warnings.catch_warnings(): # Catch errors with mean computation (e.g., overflow)
            warnings.simplefilter("ignore")
            mu = family.link.fi(eta)
      
      # Update deviance
      inval = np.isnan(mu).flatten()
      dev = family.deviance(y[inval == False],mu[inval == False])

      # And penalized deviance term
      if n_pen > 0:
         pen_dev = dev + n_coef.T @ S_emb @ n_coef
      corrections += 1
   
   # Collect accepted coefficient
   coef = n_coef
   return dev,pen_dev,mu,eta,coef

def initialize_extension(method:str,penalties:list[LambdaTerm]) -> dict:
   """Internal function. Initializes a dictionary holding all the necessary information to compute the lambda extensions at every iteration of the fitting iteration.

   :param method: Which extension method to use
   :type method: str
   :param penalties: List of penalties
   :type penalties: list[LambdaTerm]
   :return: extension info dictionary
   :rtype: dict
   """

   if method == "nesterov" or method == "nesterov2":
      extend_by = {"prev_lam":[lterm.lam for lterm in penalties],
                   "acc":[0 for _ in penalties]}

   return extend_by

def extend_lambda_step(lti:int,lam:float,dLam:float,extend_by:dict,was_extended:list[bool], method:str) -> tuple[float,dict,bool]:
   """Internal function. Performs an update to the lambda parameter, ideally extending the step aken without overshooting the objective.

   :param lti: Penalty index
   :type lti: int
   :param lam: Current lamda value
   :type lam: float
   :param dLam: The lambda update
   :type dLam: float
   :param extend_by: Extension info dictionary
   :type extend_by: dict
   :param was_extended: List holding indication per lambda parameter whether it was extended or not
   :type was_extended: bool
   :param method: Extension method to use.
   :type method: str
   :raises ValueError: If requested method is not implemented
   :return: Updated values for dLam,extend_by,was_extended
   :rtype: tuple[float,dict,bool]
   """

   if method == "nesterov" or method == "nesterov2":
      # The idea for the correction is based on the derivations in the supplementary materials
      # of Sutskever et al. (2013) - but adapted to use efs_step**2 / |lam_t - lam_{t-1}| for
      # the correction to lambda. So that the next efs update will be calculated from
      # lam_t + efs_step + (efs_step**2 / |lam_t - lam_{t-1}|) instead of just lam_t + efs_step.

      # Essentially, until corrected increase in lambda reaches unit size a fraction of the quadratic
      # efs_step is added. At unit size the quadratic efs_step is added in its entirety. As corrected update
      # step-length decreases further the extension will become increasingly larger than just the quadratic
      # of the efs_step.
   
      diff_lam = lam - extend_by["prev_lam"][lti]

      if method == "nesterov2":
         acc = dLam * min(0.99,abs(dLam)/max(sys.float_info.epsilon,2*abs(diff_lam)))

      else:
         acc = np.sign(dLam)*(dLam**2/max(sys.float_info.epsilon,abs(diff_lam)))

      extend_by["prev_lam"][lti] = lam

      if dLam>1 and diff_lam<1:
         acc = 0
      
      extend_by["acc"][lti] = acc

      extension = lam + dLam + acc

      if extension < 1e7 and extension > 1e-7 and np.sign(diff_lam) == np.sign(dLam) and abs(acc) > 0:
         dLam += acc
         was_extended[lti] = True
      else:
         was_extended[lti] = False
   else:
      raise ValueError(f"Lambda extension method '{method}' is not implemented.")
   
   return dLam,extend_by,was_extended

def undo_extension_lambda_step(lti:int,lam:float,dLam:float,extend_by:dict,was_extended:list[bool], method:str, family:Family) -> tuple[float,float]:
   """Internal function. Deals with resetting any extension terms.

   :param lti: Penalty index
   :type lti: int
   :param lam: Current lamda value
   :type lam: float
   :param dLam: The lambda update
   :type dLam: float
   :param extend_by: Extension info dictionary
   :type extend_by: dict
   :param was_extended: List holding indication per lambda parameter whether it was extended or not
   :type was_extended: bool
   :param method: Extension method to use.
   :type method: str
   :param family: model family
   :type family: Family
   :raises ValueError: If requested method is not implemented
   :return: Updated values for lam and dlam
   :rtype: tuple[float,float]
   """
   if method == "nesterov"  or method == "nesterov2":
      # We can simply reset lam by the extension factor computed earlier. If we repeatedly have to half we
      # can fall back to the strategy by Wood & Fasiolo (2016) to just half the step.
      dLam-= extend_by["acc"][lti]
      lam -= extend_by["acc"][lti]
      extend_by["acc"][lti] = 0
      was_extended[lti] = False

   else:
      raise ValueError(f"Lambda extension method '{method}' is not implemented.")

   return lam, dLam

def correct_lambda_step(y:np.ndarray,yb:np.ndarray,z:np.ndarray,Wr:scp.sparse.csc_array,rowsX:int,colsX:int,X:scp.sparse.csc_array,Xb:scp.sparse.csc_array,coef:np.ndarray,
                        Lrhoi:scp.sparse.csc_array|None,family:Family,col_S:int,S_emb:scp.sparse.csc_array,penalties:list[LambdaTerm],
                        was_extended:list[bool],pinv:str,lam_delta:np.ndarray,
                        extend_by:dict,o_iter:int,dev_check:float,n_c:int,
                        control_lambda:int,extend_lambda:bool,
                        exclude_lambda:bool,extension_method_lam:str,
                        formula:Formula,form_Linv:bool,method:str,offset:float|np.ndarray,max_inner:int) -> tuple[np.ndarray,scp.sparse.csc_array,np.ndarray,scp.sparse.csc_array,np.ndarray,np.ndarray,np.ndarray,scp.sparse.csc_array,scp.sparse.csc_array|None,float,list[float],float,np.ndarray,np.ndarray,dict,list[LambdaTerm],list[bool],scp.sparse.csc_array,int,list[int]|None,list[int]|None]:
   """Performs step-length control for lambda.

   Lambda update is based on EFS update by Wood & Fasiolo (2017), step-length control is partially based on Wood et al. (2017) - Krause et al. (submitted) has the specific implementation.

   References:
      - Wood, S. N., Li, Z., Shaddick, G., & Augustin, N. H. (2017). Generalized Additive Models for Gigadata: Modeling the U.K. Black Smoke Network Daily Data. https://doi.org/10.1080/01621459.2016.1195744
      - Wood, S. N., & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. https://doi.org/10.1111/biom.12666
      - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).


   :param y: vector of observations
   :type y: np.ndarray
   :param yb: vector of observations of the working model
   :type yb: np.ndarray
   :param z: pseudo-data (can have NaNs for invalid observations)
   :type z: np.ndarray
   :param Wr: diagonal sparse matrix holding the **root** of the Fisher weights
   :type Wr: scp.sparse.csc_array
   :param rowsX: Rows of model matrix
   :type rowsX: int
   :param colsX: Cols of model matrix
   :type colsX: int
   :param X: Model matrix
   :type X: scp.sparse.csc_array
   :param Xb: Model matrix of working model
   :type Xb: scp.sparse.csc_array
   :param coef: Current coefficient estimate
   :type coef: np.ndarray
   :param Lrhoi: Optional covariance matrix of an ar1 model
   :type Lrhoi: scp.sparse.csc_array | None
   :param family: Model family
   :type family: Family
   :param col_S: Columns of total penalty matrix
   :type col_S: int
   :param S_emb: Total penalty matrix
   :type S_emb: scp.sparse.csc_array
   :param penalties: List of penalties
   :type penalties: list[LambdaTerm]
   :param was_extended: List holding indication per lambda parameter whether it was extended or not
   :type was_extended: bool
   :param pinv: Method to use to compute generalzied inverse of total penalty, set to 'svd'!
   :type pinv: str
   :param lam_delta: Proposed update to lambda parameters
   :type lam_delta: np.ndarray
   :param extend_by: Extension info dictionary
   :type extend_by: dict
   :param o_iter: Outer iteration index
   :type o_iter: int
   :param dev_check: Multiple of previous deviance used for convergence check
   :type dev_check: float
   :param n_c: Number of cores to use
   :type n_c: int
   :param control_lambda: Whether lambda proposals should be checked (and if necessary decreased) for whether or not they (approxiately) increase the Laplace approximate restricted maximum likelihood of the model. Setting this to 0 disables control. Setting it to 1 means the step will never be smaller than the original EFS update but extensions will be removed in case the objective was exceeded. Setting it to 2 means that steps will be halved if it fails to increase the approximate REML. Set to 2 by default.
   :type control_lambda: int
   :param extend_lambda: Whether lambda proposals should be accelerated or not. Can lower the number of new smoothing penalty proposals necessary. Disabled by default.
   :type extend_lambda: bool
   :param exclude_lambda: Whether selective lambda terms should be excluded heuristically from updates. Can make each iteration a bit cheaper but is problematic when using additional Kernel penalties on terms. Thus, disabled by default.
   :type exclude_lambda: bool
   :param extension_method_lam: **Experimental - do not change!** Which method to use to extend lambda proposals. Set to 'nesterov' by default.
   :type extension_method_lam: str
   :param formula: Formula of model
   :type formula: Formula
   :param form_Linv: Whether to form the inverse of the cholesky of the negative penalzied hessian or not
   :type form_Linv: bool
   :param method: Which method to use to solve for the coefficients ("Chol" or "Qr")
   :type method: str
   :param offset: Offset (fixed effect) to add to ``eta``
   :type offset: float | np.ndarray
   :param max_inner: Maximum number of iterations to use to update the coefficient estimate
   :type max_inner: int
   :return: Tuple containing updated values for yb, Xb, z, Wr, eta, mu, n_coef, the Cholesky fo the penalzied hessian ``CholXXS``, the inverse of the former ``InvCholXXS``, total edf, term-wse edfs, updated scale, working residuals, accepted update to lambda, extend_by, penalties, was_extended, updated S_emb, number of lambda updates, an optional list of the coefficients to keep, an optional list of the estimated coefficients to drop
   :rtype: tuple[np.ndarray, scp.sparse.csc_array, np.ndarray, scp.sparse.csc_array, np.ndarray, np.ndarray, np.ndarray, scp.sparse.csc_array, scp.sparse.csc_array|None, float, list[float], float, np.ndarray, np.ndarray, dict, list[LambdaTerm], list[bool], scp.sparse.csc_array, int, list[int]|None, list[int]|None]
   """
   # Propose & perform step-length control for the lambda parameters via the Fellner Schall method
   # by Wood & Fasiolo (2016)
   lam_accepted = False
   lam_checks = 0

   # dev_check holds current deviance - but need penalized for lambda step-length (see Wood et al., 2017)
   pen_dev_check = dev_check + coef.T @ S_emb @ coef 

   while not lam_accepted:

      # Re-compute S_emb and S_pinv
      S_emb,S_pinv,S_root,FS_use_rank = compute_S_emb_pinv_det(col_S,penalties,pinv,method=="QR")

      # Update coefficients
      eta,mu,n_coef,\
      CholXXS,\
      InvCholXXS,\
      lgdetDs,\
      bsbs,\
      total_edf,\
      term_edfs,\
      Bs,scale,wres,\
      keep, drop = update_coef_and_scale(y,yb,z,Wr,rowsX,colsX,
                                            X,Xb,Lrhoi,family,S_emb,S_root,S_pinv,
                                            FS_use_rank,penalties,n_c,
                                            formula,form_Linv,offset)
      
      # Optionally repeat PIRLS iteration until convergence - this is no longer PQL/Wood, 2017 but will generally be more stable (Wood & Fasiolo, 2017) 
      if max_inner > 1 and ((isinstance(family,Gaussian) == False) or (isinstance(family.link,Identity) == False)):
         
         # If coef. are updated to convergence, we have to check first iter against incoming/old coef
         # but under current penalty! (see Wood et al., 2017)
         c_dev_prev = dev_check + coef.T @ S_emb @ coef # dev check holds current deviance
         n_coef2 = copy.deepcopy(n_coef)
         n_coef = copy.deepcopy(coef)

         # Update deviance & penalized deviance
         inval = np.isnan(mu).flatten()
         dev = family.deviance(y[inval == False],mu[inval == False])
         pen_dev = dev + n_coef2.T @ S_emb @ n_coef2

         for i_iter in range(max_inner - 1):
            
            # Perform step-length control for the coefficients (repeat step 3 in Wood, 2017)
            dev,pen_dev,mu,eta,n_coef = correct_coef_step(n_coef,n_coef2,dev,pen_dev,c_dev_prev,family,eta,mu,y,X,len(penalties),S_emb,None,n_c,offset)

            # Update PIRLS weights
            yb,Xb,z,Wr = update_PIRLS(y,yb,mu,eta-offset,X,Xb,family,None)

            # Convergence check for inner loop
            if i_iter > 0:
               dev_diff_inner = abs(pen_dev - c_dev_prev)
               if dev_diff_inner < 1e-9*pen_dev or i_iter == max_inner - 2:
                  break

            c_dev_prev = pen_dev

            # Now propose next set of coefficients
            eta,mu,n_coef2,Pr,P,LP,keep,drop = update_coef(yb,X,Xb,family,S_emb,S_root,n_c,None,offset)

            # Update deviance & penalized deviance
            inval = np.isnan(mu).flatten()
            dev = family.deviance(y[inval == False],mu[inval == False])
            pen_dev = dev + n_coef2.T @ S_emb @ n_coef2

         
         # Now re-compute scale and Linv
         # Given new coefficients compute lgdetDs and bsbs - needed for REML gradient and EFS step
         lgdetDs = None
         bsbs = None
         if len(penalties) > 0:
            lgdetDs = []
            bsbs = []
            for lti,lTerm in enumerate(penalties):

               lt_rank = None
               if FS_use_rank[lti]:
                  lt_rank = lTerm.rank

               lgdetD,bsb = compute_lgdetD_bsb(lt_rank,lTerm.lam,S_pinv,lTerm.S_J_emb,coef)
               lgdetDs.append(lgdetD)
               bsbs.append(bsb)
         
         # Solve for inverse of Chol factor of XX+S
         InvCholXXSP = None
         if form_Linv:
            InvCholXXSP = compute_Linv(LP,n_c)
         
         wres,InvCholXXS,total_edf,term_edfs,_,scale = update_scale_edf(y,z,eta,Wr,rowsX,colsX,LP,InvCholXXSP,Pr,lgdetDs,None,family,penalties,keep,drop,n_c)

         # Unpivot Cholesky of negative penalized hessian - InvCholXXS has already been un-pivoted (and padded) by `update_scale_edf`!
         CholXXS = P.T@LP

         # Again need to make sure here to insert zero rows and columns after un-pivoting for parameters that were dropped.
         if LP.shape[1] < S_emb.shape[1]:
            CholXXSdat,CholXXSrow,CholXXScol = translate_sparse(CholXXS.tocsc())
      
            CholXXSrow = keep[CholXXSrow]
            CholXXScol = keep[CholXXScol]

            CholXXS = scp.sparse.csc_array((CholXXSdat,(CholXXSrow,CholXXScol)),shape=(colsX,colsX))
      
      # Compute gradient of REML with respect to lambda
      # to check if step size needs to be reduced (part of step 6 in Wood, 2017).
      lam_grad = [grad_lambda(lgdetDs[lti],Bs[lti],bsbs[lti],scale) for lti in range(len(penalties))]
      lam_grad = np.array(lam_grad).reshape(-1,1) 
      check = lam_grad.T @ lam_delta

      # For Generalized models we theoretically should not reduce step beyond original EFS update since
      # criterion maximized is approximate REML. But we can probably relax the criterion a
      # bit, since check will quite often be < 0. Additionally, performing step length control after all
      # can be motivated (see Krause et al., submitted). So if control_lambda > 1 we control anyway.
      check_criterion = 0
      if ((isinstance(family,Gaussian) == False) or (isinstance(family.link,Identity) == False)) and control_lambda < 2:
            
            if o_iter > 0:
               check_criterion = 1e-7*-abs(pen_dev_check)
            
            if check[0,0] < check_criterion: 
               # Now check whether we extend lambda - and if we do so whether any extension was actually applied.
               # If not, we still "pass" the check.
               if (extend_lambda == False) or (np.any(was_extended) == False):
                  check[0,0] = check_criterion + 1

      # Now check whether we have to correct lambda.
      # Because of minimization in Wood (2017) they use a different check (step 7) but idea is the same.
      if check[0,0] < check_criterion and control_lambda > 0: 
         # Reset extension or cut the step taken in half (for additive models)
         lam_changes = 0
         for lti,lTerm in enumerate(penalties):

            # Reset extension factor for all terms that were extended.
            if extend_lambda and was_extended[lti]:
               lam, dLam = undo_extension_lambda_step(lti,lTerm.lam,lam_delta[lti][0],extend_by,was_extended, extension_method_lam, family)
               lTerm.lam = lam
               lam_delta[lti][0] = dLam
               lam_changes += 1

            # Otherwise, rely on the strategy by Wood & Fasiolo (2016) to just half the step.
            # For strictly additive models we can always do this. For all others we only do it if control_lambda > 1
            elif (isinstance(family,Gaussian) and isinstance(family.link,Identity)) or control_lambda == 2:
                  lam_delta[lti] = lam_delta[lti]/2
                  lTerm.lam -= lam_delta[lti][0]
                  lam_changes += 1
         
         # If step becomes extremely small, accept step
         if lam_changes == 0 or np.linalg.norm(lam_delta) < 1e-7:
            lam_accepted = True
         
         # Also have to reset eta, mu, z, and Wr
         if max_inner > 1 and ((isinstance(family,Gaussian) == False) or (isinstance(family.link,Identity) == False)):
            eta = (X @ coef).reshape(-1,1) + offset
            mu = eta

            if isinstance(family,Gaussian) == False or isinstance(family.link,Identity) == False:
               with warnings.catch_warnings(): # Catch errors with mean computation (e.g., overflow)
                  warnings.simplefilter("ignore")
                  mu = family.link.fi(eta)
            
            # Reset PIRLS weights
            yb,Xb,z,Wr = update_PIRLS(y,yb,mu,eta-offset,X,Xb,family,None)
            
      if check[0,0] >= check_criterion or (control_lambda == 0) or lam_accepted:
         # Accept the step and propose a new one as well! (part of step 6 in Wood, 2017; here uses efs from Wood & Fasiolo, 2017 to propose new lambda delta)
         lam_accepted = True
         lam_delta = []
         for lti,(lGrad,lTerm) in enumerate(zip(lam_grad,penalties)):
            # Heuristic check to determine whether smooth penalty should be excluded from further updates.
            # For penalties that -> inf, there is an inverse relationship between the gradient^2 and the lambda term:
            # the gradient^2 diminishes, the lambda term explodes
            # If the ratio between those two gets close to zero, we drop the corresponding term from the next update.
            if (not ((lTerm.lam > 1e5) and (((lGrad[0]**2)/lTerm.lam) < 1e-8))) or exclude_lambda == False:

               dLam = step_fellner_schall_sparse(lgdetDs[lti],Bs[lti],bsbs[lti],lTerm.lam,scale)
               #print("Theorem 1:",lgdetDs[lti]-Bs[lti],bsbs[lti])

               if extend_lambda:
                  dLam,extend_by,was_extended = extend_lambda_step(lti,lTerm.lam,dLam,extend_by,was_extended,extension_method_lam)

            else: # ReLikelihood is probably insensitive to further changes in this smoothing penalty, so set change to 0.
               dLam = 0
               was_extended[lti] = False

            lam_delta.append(dLam)

         lam_delta = np.array(lam_delta).reshape(-1,1)

      lam_checks += 1

   return yb,Xb,z,Wr,eta,mu,n_coef,CholXXS,InvCholXXS,total_edf,term_edfs,scale,wres,lam_delta,extend_by,penalties,was_extended,S_emb,lam_checks,keep,drop

################################################ Main solver ################################################

def solve_gamm_sparse(mu_init:np.ndarray,y:np.ndarray,X:scp.sparse.csc_array,penalties:list[LambdaTerm],col_S:int,family:Family,
                      maxiter:int=10,max_inner:int = 100,pinv:str="svd",conv_tol:float=1e-7,
                      extend_lambda:bool=False,control_lambda:int=1,
                      exclude_lambda:bool=False,extension_method_lam:str = "nesterov",
                      form_Linv:bool=True,method:str="Chol",check_cond:int=2,progress_bar:bool=False,
                      n_c:int=10,offset:int=0,Lrhoi:scp.sparse.csc_array|None=None) -> tuple[np.ndarray,np.ndarray,np.ndarray,scp.sparse.csc_array,scp.sparse.csc_array,float,scp.sparse.csc_array,float,list[float],float,Fit_info]:
   """Estimates a Generalized Additive Mixed model. Implements the algorithms discussed in section 3.2 of the paper by Krause et al. (submitted).

   Relies on methods proposed by Wood et al. (2017), Wood & Fasiolo (2017), Wood (2011), and Wood (2017).

   References:
      - Wood, S. N., Li, Z., Shaddick, G., & Augustin, N. H. (2017). Generalized Additive Models for Gigadata: Modeling the U.K. Black Smoke Network Daily Data. https://doi.org/10.1080/01621459.2016.1195744
      - Wood, S. N., & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. https://doi.org/10.1111/biom.12666
      - Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models: Estimation of Semiparametric Generalized Linear Models. https://doi.org/10.1111/j.1467-9868.2010.00749.x
      - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
      - Krause et al. (submitted). The Mixed-Sparse-Smooth-Model Toolbox (MSSM): Efficient Estimation and Selection of Large Multi-Level Statistical Models. https://doi.org/10.48550/arXiv.2506.13132

   :param mu_init: Initial values for means
   :type mu_init: np.ndarray
   :param y: vector of observations
   :type y: np.ndarray
   :param X: Model matrix
   :type X: scp.sparse.csc_array
   :param penalties: List of penalties
   :type penalties: list[LambdaTerm]
   :param col_S: Columns of total penalty matrix
   :type col_S: int
   :param family: Family of model
   :type family: Family
   :param maxiter: Maximum number of iterations for outer algorithm updating lambda, defaults to 10
   :type maxiter: int, optional
   :param max_inner: Maximum number of iterations for inner algorithm updating coefficients, defaults to 100
   :type max_inner: int, optional
   :param pinv: Method to use to compute generalzied inverse of total penalty,, defaults to "svd"
   :type pinv: str, optional
   :param conv_tol: Convergence tolerance, defaults to 1e-7
   :type conv_tol: float, optional
   :param extend_lambda: Whether lambda proposals should be accelerated or not. Can lower the number of new smoothing penalty proposals necessary. Disabled by default.
   :type extend_lambda: bool
   :param control_lambda: Whether lambda proposals should be checked (and if necessary decreased) for whether or not they (approxiately) increase the Laplace approximate restricted maximum likelihood of the model. Setting this to 0 disables control. Setting it to 1 means the step will never be smaller than the original EFS update but extensions will be removed in case the objective was exceeded. Setting it to 2 means that steps will be halved if it fails to increase the approximate REML. Set to 1 by default.
   :type control_lambda: int
   :param exclude_lambda: Whether selective lambda terms should be excluded heuristically from updates. Can make each iteration a bit cheaper but is problematic when using additional Kernel penalties on terms. Thus, disabled by default.
   :type exclude_lambda: bool
   :param extension_method_lam: _description_, defaults to "nesterov"
   :type extension_method_lam: str, optional
   :param form_Linv: Whether to form the inverse of the cholesky of the negative penalzied hessian or not, defaults to True
   :type form_Linv: bool, optional
   :param method:  Which method to use to solve for the coefficients ("Chol" or "Qr"), defaults to "Chol"
   :type method: str, optional
   :param check_cond: Whether to obtain an estimate of the condition number for the linear system that is solved. When ``check_cond=0``, no check will be performed. When ``check_cond=1``, an estimate of the condition number for the final system (at convergence) will be computed and warnings will be issued based on the outcome (see :func:`mssm.src.python.gamm_solvers.est_condition`). When ``check_cond=2``, an estimate of the condition number will be performed for each new system (at each iteration of the algorithm) and an error will be raised if the condition number is estimated as too high given the chosen ``method``., defaults to 2
   :type check_cond: int, optional
   :param progress_bar: Whether to print progress or not, defaults to False
   :type progress_bar: bool, optional
   :param n_c: Number of cores to use, defaults to 10
   :type n_c: int, optional
   :param offset: Offset (fixed effect) to add to ``eta``, defaults to 0
   :type offset: int, optional
   :param Lrhoi: Optional covariance matrix of an ar1 model, defaults to None
   :type Lrhoi: scp.sparse.csc_array | None, optional
   :raises ArithmeticError: _description_
   :raises ArithmeticError: _description_
   :raises ArithmeticError: _description_
   :raises ArithmeticError: _description_
   :raises warnings.warn: _description_
   :return: An estimate of the coefficients coef,the linear predictor eta, the working residuals wres, the root of the Fisher weights as matrix Wr, the matrix with Newton weights at convergence WN, an estimate of the scale parameter, an inverse of the cholesky of the penalized negative hessian InvCholXXS, total edf, term-wise edf, total penalty, a :class:`Fit_info` object
   :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, scp.sparse.csc_array, scp.sparse.csc_array, float, scp.sparse.csc_array, float, list[float], float, Fit_info]
   """
   # Estimates a penalized Generalized additive mixed model, following the steps outlined in Wood, Li, Shaddick, & Augustin (2017)
   # "Generalized Additive Models for Gigadata" referred to as Wood (2017) below.

   n_c = min(mp.cpu_count(),n_c)
   rowsX,colsX = X.shape
   coef = None
   n_coef = None
   K2 = None

   if Lrhoi is not None and isinstance(family,Gaussian) and isinstance(family.link,Identity):
      # Can simply apply ar1 model at start, then we estimate Lrhoi.T@y ~ N(Lrhoi.T@X@coef,scale)
      # Need to fix eta and resid after convergence.
      y = Lrhoi.T@y
      X = (Lrhoi.T@X).tocsc()
      X.sort_indices()

   # Additive mixed model can simply be fit on y and X
   # Generalized mixed model needs to be fit on weighted X and pseudo-dat
   # but the same routine can be used (Wood, 2017) so both should end
   # up in the same variables passed down:
   yb = y
   Xb = X

   # mu and eta (start estimates in case the family is not Gaussian)
   mu = mu_init
   eta = mu

   if isinstance(family,Gaussian) == False or isinstance(family.link,Identity) == False:
      eta = family.link.f(mu)

   # Compute starting estimates
   dev,pen_dev,eta,mu,coef,CholXXS,InvCholXXS,total_edf,term_edfs,scale,wres,lam_delta,S_emb = init_step_gam(y,yb,mu,eta,rowsX,colsX,X,Xb,
                                                                                                     family,col_S,penalties,
                                                                                                     pinv,n_c,None,form_Linv,method,offset,Lrhoi)
   
   yb,Xb,z,Wr = update_PIRLS(y,yb,mu,eta-offset,X,Xb,family,Lrhoi)
   
   if check_cond == 2:
      K2,_,_,Kcode = est_condition(CholXXS,InvCholXXS,verbose=False)
      if method == "Chol" and Kcode == 1:
         raise ArithmeticError(f"Condition number ({K2}) of matrix A, where A.T@A=X.T@X + S_\\lambda, is larger than 1/sqrt(u), where u is half the machine precision. Try calling ``model.fit()`` with ``method='QR'``.")
      if method != "Chol" and Kcode == 2:
         raise ArithmeticError(f"Condition number ({K2}) of matrix A, where A.T@A=X.T@X + S_\\lambda, is larger than 1/u, where u is half the machine precision. The model estimates are likely inaccurate.")

   # Initialize extension variable
   extend_by = initialize_extension(extension_method_lam,penalties)
   was_extended = [False for _ in enumerate(penalties)]
   
   # Loop to optimize smoothing parameter (see Wood, 2017)
   iterator = range(maxiter)
   if progress_bar:
      iterator = tqdm(iterator,desc="Fitting",leave=True)

   # Keep track on some useful info
   fit_info = Fit_info()
   for o_iter in iterator:

      if o_iter > 0:

         # Obtain deviance and penalized deviance terms
         # under **current** lambda for proposed coef (n_coef)
         # and current coef. (see Step 3 in Wood, 2017)
         inval = np.isnan(mu).flatten()
         dev = family.deviance(y[inval == False],mu[inval == False])
         pen_dev = dev
         c_dev_prev = prev_dev

         if len(penalties) > 0:
            pen_dev += n_coef.T @ S_emb @ n_coef
            c_dev_prev += coef.T @ S_emb @ coef

         # For Gaussian/PQL, perform step-length control for the coefficients here (Step 3 in Wood, 2017)
         if max_inner <= 1 or (isinstance(family,Gaussian) and isinstance(family.link,Identity)):

            dev,pen_dev,mu,eta,coef = correct_coef_step(coef,n_coef,dev,pen_dev,c_dev_prev,family,eta,mu,y,X,len(penalties),S_emb,None,n_c,offset)

            # Update pseudo-dat weights for next coefficient step (step 1 in Wood, 2017; but moved after the coef correction because z and Wr depend on
            # mu and eta, which change during the correction but anything that needs to be computed during the correction (deviance) does not depend on
            # z and Wr).
            yb,Xb,z,Wr = update_PIRLS(y,yb,mu,eta - offset,X,Xb,family,Lrhoi)
         else:
            coef = n_coef

         # Test for convergence (Step 2 in Wood, 2017), implemented based on step 4 in Wood, Goude, & Shaw (2016): Generalized
         # additive models for large data-sets. They reccomend inspecting the change in deviance after a PQL iteration to monitor
         # convergence. Wood (2017) check the REML gradient against a fraction of the current deviance so to determine whether the change
         # in deviance is "small" enough, it is also compared to a fraction of the current deviance. mgcv's bam function also considers this
         # for convergence decisions but as part of a larger composite criterion, also involving checks on the scale parameter for example.
         # From simulations, I get the impression that the simple criterion proposed by WGS seems to suffice.
         dev_diff = abs(pen_dev - prev_pen_dev)

         if progress_bar:
            iterator.set_description_str(desc="Fitting - Conv.: " + "{:.2e}".format(dev_diff - conv_tol*pen_dev), refresh=True)
            
         if dev_diff < conv_tol*pen_dev:
            if progress_bar:
               iterator.set_description_str(desc="Converged!", refresh=True)
               iterator.close()
            fit_info.code = 0
            break

      # We need the deviance and penalized deviance of the model at this point (before completing steps 5-7 (dev_{old} in WGS used for convergence control)
      # for coef step control (step 3 in Wood, 2017) and convergence control (step 2 in Wood, 2017 based on step 4 in Wood, Goude, & Shaw, 2016) respectively
      prev_dev = dev
      prev_pen_dev = pen_dev
         
      # Step length control for proposed lambda change (steps 5-7 in Wood, 2017) adjusted to make use of EFS from Wood & Fasiolo, 2017
      if len(penalties) > 0: 
         
         # Test the lambda update (step 5 in Wood, 2017)
         for lti,lTerm in enumerate(penalties):
            lTerm.lam += lam_delta[lti][0]
            #print(lTerm.lam,lam_delta[lti][0])

         # Now check step length and compute lambda + coef update. (steps 6-7 in Wood, 2017)
         yb,Xb,z,Wr,eta,mu,n_coef,CholXXS,InvCholXXS,total_edf,term_edfs,scale,wres,lam_delta,extend_by,penalties,was_extended,S_emb,lam_checks,keep,drop = correct_lambda_step(y,yb,z,Wr,rowsX,colsX,X,Xb,coef,
                                                                                                                                                                     Lrhoi,family,col_S,S_emb,penalties,
                                                                                                                                                                     was_extended,pinv,lam_delta,
                                                                                                                                                                     extend_by,o_iter,dev,n_c,
                                                                                                                                                                     control_lambda,extend_lambda,
                                                                                                                                                                     exclude_lambda,extension_method_lam,
                                                                                                                                                                     None,form_Linv,method,offset,max_inner)
         
         fit_info.lambda_updates += lam_checks
         
      else:
         # If there are no penalties simply perform a newton step
         # for the coefficients only
         eta,mu,n_coef,\
         CholXXS,\
         InvCholXXS,\
         _,\
         _,\
         total_edf,\
         term_edfs,\
         _,scale,wres,\
         keep, drop = update_coef_and_scale(y,yb,z,Wr,rowsX,colsX,
                                            X,Xb,Lrhoi,family,S_emb,None,None,None,
                                            penalties,n_c,None,form_Linv,offset)
         
      fit_info.dropped = drop

      # Check condition number of current system. 
      if check_cond == 2:
         K2,_,_,Kcode = est_condition(CholXXS,InvCholXXS,verbose=progress_bar)
         if method == "Chol" and Kcode == 1:
            raise ArithmeticError(f"Condition number ({K2}) of matrix A, where A.T@A=X.T@X + S_\\lambda, is larger than 1/sqrt(u), where u is half the machine precision. Try calling ``model.fit()`` with ``method='QR'``.")
         if method != "Chol" and Kcode == 2:
            raise ArithmeticError(f"Condition number ({K2}) of matrix A, where A.T@A=X.T@X + S_\\lambda, is larger than 1/u, where u is half the machine precision. The model estimates are likely inaccurate.")
      
      # At this point we:
      #  - have corrected & accepted the lam_deltas added above (step 5)
      #  - have proposed new coefficients (n_coef)
      #  - have updated eta and mu to reflect these new coef
      #  - have assigned the deviance before completing steps 5-7 to prev_dev
      #  - have proposed new lambda deltas (lam_delta)
      #
      # This completes step 8 in Wood (2017)!
      fit_info.iter += 1

   # Final penalty
   if len(penalties) > 0:
      penalty = coef.T @ S_emb @ coef
   else:
      penalty = 0

   # Final term edf
   if not term_edfs is None:
      term_edfs = calculate_term_edf(penalties,term_edfs)

   # At this point, Wr/z might not match the dimensions of y and X, because observations might be
   # excluded at convergence. eta and mu are of correct dimension, so we need to re-compute Wr - this
   # time with a weight of zero for any dropped obs.
   WN = None
   if Wr is not None:
      inval = np.isnan(z)
      inval_check =  np.any(inval)

      wres2 = None
      if Lrhoi is not None: # Need to return ar-corrected and "normal" working residuals
         wres2 = Wr @ (z[inval == False] - eta[inval == False]).reshape(-1,1)

      if inval_check:
         w_full = np.zeros_like(eta)
         w_full[inval==False] = Wr.diagonal()
         w = w_full

         # Re-compute weight matrix
         Wr = scp.sparse.spdiags([np.sqrt(np.ndarray.flatten(w))],[0])

         # Adjust working residuals
         wres_full = np.zeros_like(eta)
         wres_full[inval==False] = wres.flatten()
         wres = wres_full

         if Lrhoi is not None:
            wres_full2 = np.zeros_like(eta)
            wres_full2[inval==False] = wres2.flatten()
            wres2 = wres_full2

      if wres2 is not None:
         wres = [wres,wres2]
   
      # Compute Newton weights once to enable computation of observed hessian
      _, wN, inval = PIRLS_newton_weights(y,mu,eta-offset,family)
      wN[inval] = 0
      WN = scp.sparse.spdiags([np.ndarray.flatten(wN)],[0])

   if InvCholXXS is None:
      if method != "Chol":
         warnings.warn("Final re-computation of inverse of Cholesky of penalized Hessian might not account for dropped coefficients. Check `model.info` and inspect returned inverse carefully.")
      Lp, Pr, _ = cpp_cholP((Xb.T @ Xb + S_emb).tocsc())
      InvCholXXSP = compute_Linv(Lp,n_c)
      InvCholXXS = apply_eigen_perm(Pr,InvCholXXSP)
   
   # Check condition number of current system but only after convergence - and warn only. 
   if check_cond == 1:
      K2,_,_,Kcode = est_condition(CholXXS,InvCholXXS,verbose=False)

      if fit_info.code == 0: # Convergence was reached but Knumber might suggest instable system.
         fit_info.code = Kcode

      if method == "Chol" and Kcode == 1:
         warnings.warn(f"Condition number ({K2}) of matrix A, where A.T@A=X.T@X + S_\\lambda, is larger than 1/sqrt(u), where u is half the machine precision. Try calling ``model.fit()`` with ``method='QR'``.")
      if method != "Chol" and Kcode == 2:
         raise warnings.warn(f"Condition number ({K2}) of matrix A, where A.T@A=X.T@X + S_\\lambda, is larger than 1/u, where u is half the machine precision. The model estimates are likely inaccurate.")

   fit_info.K2 = K2

   return coef,eta,wres,Wr,WN,scale,InvCholXXS,total_edf,term_edfs,penalty,fit_info

################################################ Iterative GAMM building code ################################################

def read_mmat(should_cache:bool,cache_dir:str,file:str,fi:int,terms:list[GammTerm],has_intercept:bool,
              ltx:list[int],irstx:list[int],stx:list[int],rtx:list[int],var_types:dict,var_map:dict,var_mins:dict,
              var_maxs:dict,factor_levels:dict,cov_flat_file:np.ndarray,cov:list[np.ndarray]) -> scp.sparse.csc_array:
   """Creates model matrix for that dataset. The model-matrix is either cached or not. If the former is the case,
   the matrix is read in on subsequent calls to this function.

   :param should_cache: whether or not the directory should actually be created
   :type should_cache: bool
   :param cache_dir: path to cache directory
   :type cache_dir: str
   :param file: File name
   :type file: str
   :param fi: File index in all files
   :type fi: int
   :param terms: List of terms in model formula
   :type terms: list[GammTerm]
   :param has_intercept: Whether the formula has an intercept or not
   :type has_intercept: bool
   :param ltx: Linear term indices
   :type ltx: list[int]
   :param irstx: Impulse response function term indices
   :type irstx: list[int]
   :param stx: Smooth term indices
   :type stx: list[int]
   :param rtx: Random term indices
   :type rtx: list[int]
   :param var_types: Dictionary holding variable types
   :type var_types: dict
   :param var_map: Dictionary mapping variable names to column indices in the encoded data
   :type var_map: dict
   :param var_mins: Dictionary with variable minimums 
   :type var_mins: dict
   :param var_maxs: Dictionary with variable maximums 
   :type var_maxs: dict
   :param factor_levels: Dictionary with levels associated with each factor
   :type factor_levels: dict
   :param cov_flat_file: Encoded data based on ``file``
   :type cov_flat_file: np.ndarray
   :param cov: Essentially ``[cov_flat_file]``
   :type cov: list[np.ndarray]
   :return: model matrix associated with this file
   :rtype: scp.sparse.csc_array
   """

   target = file.split("/")[-1].split(".csv")[0] + f"_{fi}.npz"
   
   if should_cache == False:
         mmat = build_sparse_matrix_from_formula(terms,has_intercept,
                                       ltx,irstx,stx,rtx,var_types,var_map,var_mins,
                                       var_maxs,factor_levels,cov_flat_file,cov)
         
   elif should_cache == True and target not in os.listdir(cache_dir):
         mmat = build_sparse_matrix_from_formula(terms,has_intercept,
                                       ltx,irstx,stx,rtx,var_types,var_map,var_mins,
                                       var_maxs,factor_levels,cov_flat_file,cov)
         
         scp.sparse.save_npz(f"{cache_dir}/" + target,mmat)
   else:
         mmat = scp.sparse.load_npz(f"{cache_dir}/" + target)

   return mmat

def form_cross_prod_mp(should_cache:bool,cache_dir:str,file:str,fi:int,y_flat:np.ndarray,terms:list[GammTerm],has_intercept:bool,
                       ltx:list[int],irstx:list[int],stx:list[int],rtx:list[int],var_types:dict,var_map:dict,var_mins:dict,
                       var_maxs:dict,factor_levels:dict,cov_flat_file:np.ndarray,cov:list[np.ndarray]) -> tuple[scp.sparse.csc_array,np.ndarray]:
   """Computes X.T@X and X.T@y based on the data in ``file``.

   :param should_cache: whether or not the directory should actually be created
   :type should_cache: bool
   :param cache_dir: path to cache directory
   :type cache_dir: str
   :param file: File name
   :type file: str
   :param fi: File index in all files
   :type fi: int
   :param y_flat: Observation vector
   :type y_flat: np.ndarray
   :param terms: List of terms in model formula
   :type terms: list[GammTerm]
   :param has_intercept: Whether the formula has an intercept or not
   :type has_intercept: bool
   :param ltx: Linear term indices
   :type ltx: list[int]
   :param irstx: Impulse response function term indices
   :type irstx: list[int]
   :param stx: Smooth term indices
   :type stx: list[int]
   :param rtx: Random term indices
   :type rtx: list[int]
   :param var_types: Dictionary holding variable types
   :type var_types: dict
   :param var_map: Dictionary mapping variable names to column indices in the encoded data
   :type var_map: dict
   :param var_mins: Dictionary with variable minimums 
   :type var_mins: dict
   :param var_maxs: Dictionary with variable maximums 
   :type var_maxs: dict
   :param factor_levels: Dictionary with levels associated with each factor
   :type factor_levels: dict
   :param cov_flat_file: Encoded data based on ``file``
   :type cov_flat_file: np.ndarray
   :param cov: Essentially ``[cov_flat_file]``
   :type cov: list[np.ndarray]
   :return: X.T@X, X.T@y
   :rtype: tuple[scp.sparse.csc_array,np.ndarray]
   """
   
   model_mat = read_mmat(should_cache,cache_dir,file,fi,terms,has_intercept,
                         ltx,irstx,stx,rtx,var_types,var_map,var_mins,
                         var_maxs,factor_levels,cov_flat_file,cov)
   
   Xy = model_mat.T @ y_flat
   XX = (model_mat.T @ model_mat).tocsc()

   return XX,Xy

def read_XTX(file:str,formula:Formula,nc:int) -> tuple[scp.sparse.csc_array,np.ndarray,int]:
   """Computes X.T@X and X.T@y for this file in parallel, reading data from file.

   :param file: File name
   :type file: str
   :param formula: Formula of model
   :type formula: Formula
   :param nc: Number of cores to use
   :type nc: int
   :return: X.T@X, X.T@y
   :rtype: tuple[scp.sparse.csc_array,np.ndarray,int]
   """

   terms = formula.terms
   has_intercept = formula.has_intercept
   ltx = formula.get_linear_term_idx()
   irstx = []
   stx = formula.get_smooth_term_idx()
   rtx = formula.get_random_term_idx()
   var_types = formula.get_var_types()
   var_map = formula.get_var_map()
   var_mins = formula.get_var_mins()
   var_maxs = formula.get_var_maxs()
   factor_levels = formula.get_factor_levels()

   cov = None

   # Read file
   file_dat = pd.read_csv(file)

   # Encode data in this file
   y_flat_file,cov_flat_file,NAs_flat_file,_,_,_,_ = formula.encode_data(file_dat)
   cov_flat_file = cov_flat_file[NAs_flat_file,:]
   y_flat_file = y_flat_file[NAs_flat_file]

   # Parallelize over sub-sets of this file
   rows,_ = cov_flat_file.shape
   cov_flat_files = np.array_split(cov_flat_file,min(nc,rows),axis=0)
   y_flat_files = np.array_split(y_flat_file,min(nc,rows))
   subsets = [i for i in range(len(cov_flat_files))]

   with mp.Pool(processes=nc) as pool:
      # Build the model matrix with all information from the formula - but only for sub-set of rows in this file
      XX,Xy = zip(*pool.starmap(form_cross_prod_mp,zip(repeat(SHOULD_CACHE),repeat(CACHE_DIR),repeat(file),
                                                       subsets,y_flat_files,repeat(terms),repeat(has_intercept),
                                                       repeat(ltx),repeat(irstx),repeat(stx),repeat(rtx),
                                                       repeat(var_types),repeat(var_map),
                                                       repeat(var_mins),repeat(var_maxs),repeat(factor_levels),
                                                       cov_flat_files,repeat(cov))))
   
   XX = reduce(lambda xx1,xx2: xx1+xx2,XX)
   Xy = reduce(lambda xy1,xy2: xy1+xy2,Xy)

   return XX,Xy,len(y_flat_file)

def keep_XTX(cov_flat:np.ndarray,y_flat:np.ndarray,formula:Formula,nc:int,progress_bar:bool) -> tuple[scp.sparse.csc_array,np.ndarray]:
   """Computes X.T@X and X.T@y in blocks.

   :param cov_flat: Encoded data as np.array
   :type cov_flat: np.ndarray
   :param y_flat: vector of observations
   :type y_flat: np.ndarray
   :param formula: Formula of model
   :type formula: Formula
   :param nc: Number of cores to use
   :type nc: int
   :param progress_bar: Whether to print progress or not
   :type progress_bar: bool
   :return: X.T@X, X.T@y
   :rtype: tuple[scp.sparse.csc_array,np.ndarray]
   """

   terms = formula.terms
   has_intercept = formula.has_intercept
   ltx = formula.get_linear_term_idx()
   irstx = []
   stx = formula.get_smooth_term_idx()
   rtx = formula.get_random_term_idx()
   var_types = formula.get_var_types()
   var_map = formula.get_var_map()
   var_mins = formula.get_var_mins()
   var_maxs = formula.get_var_maxs()
   factor_levels = formula.get_factor_levels()

   cov = None

   
   cov_split = np.array_split(cov_flat,len(formula.file_paths),axis=0)
   y_split = np.array_split(y_flat,len(formula.file_paths))

   iterator = cov_split
   if progress_bar:
      iterator = tqdm(iterator,desc="Accumulating X.T @ X",leave=True)

   for fi,_ in enumerate(iterator):

      rows,_ = cov_split[fi].shape
      cov_flat_files = np.array_split(cov_split[fi],min(nc,rows),axis=0)
      y_flat_files = np.array_split(y_split[fi],min(nc,rows))
      subsets = [i for i in range(len(cov_flat_files))]

      with mp.Pool(processes=min(rows,nc)) as pool:
         # Build the model matrix with all information from the formula - but only for sub-set of rows at a time
         XX0,Xy0 = zip(*pool.starmap(form_cross_prod_mp,zip(repeat(SHOULD_CACHE),repeat(CACHE_DIR),repeat(f"/outer_split_{fi}.csv"),
                                                         subsets,y_flat_files,repeat(terms),repeat(has_intercept),
                                                         repeat(ltx),repeat(irstx),repeat(stx),repeat(rtx),
                                                         repeat(var_types),repeat(var_map),
                                                         repeat(var_mins),repeat(var_maxs),repeat(factor_levels),
                                                         cov_flat_files,repeat(cov))))
      
      XX0 = reduce(lambda xx1,xx2: xx1+xx2,XX0)
      Xy0 = reduce(lambda xy1,xy2: xy1+xy2,Xy0)

      # Compute X.T@X and X.T@y
      if fi == 0:
         XX = XX0
         Xy = Xy0
      else:
         XX += XX0
         Xy += Xy0

   return XX,Xy

def form_eta_mp(should_cache:bool,cache_dir:str,file:str,fi:int,coef:np.ndarray,terms:list[GammTerm],has_intercept:bool,
                       ltx:list[int],irstx:list[int],stx:list[int],rtx:list[int],var_types:dict,var_map:dict,var_mins:dict,
                       var_maxs:dict,factor_levels:dict,cov_flat_file:np.ndarray,cov:list[np.ndarray]) -> np.ndarray:
   """Computed ``X@coef``, where ``X`` is model matrix for ``file``.

   :param should_cache: whether or not the directory should actually be created
   :type should_cache: bool
   :param cache_dir: path to cache directory
   :type cache_dir: str
   :param file: File name
   :type file: str
   :param fi: File index in all files
   :type fi: int
   :param coef: Current coefficient estimate
   :type coef: np.ndarray
   :param terms: _description_
   :param terms: List of terms in model formula
   :type terms: list[GammTerm]
   :param has_intercept: Whether the formula has an intercept or not
   :type has_intercept: bool
   :param ltx: Linear term indices
   :type ltx: list[int]
   :param irstx: Impulse response function term indices
   :type irstx: list[int]
   :param stx: Smooth term indices
   :type stx: list[int]
   :param rtx: Random term indices
   :type rtx: list[int]
   :param var_types: Dictionary holding variable types
   :type var_types: dict
   :param var_map: Dictionary mapping variable names to column indices in the encoded data
   :type var_map: dict
   :param var_mins: Dictionary with variable minimums 
   :type var_mins: dict
   :param var_maxs: Dictionary with variable maximums 
   :type var_maxs: dict
   :param factor_levels: Dictionary with levels associated with each factor
   :type factor_levels: dict
   :param cov_flat_file: Encoded data based on ``file``
   :type cov_flat_file: np.ndarray
   :param cov: Essentially ``[cov_flat_file]``
   :type cov: list[np.ndarray]
   :return: X@coef for this file
   :rtype: np.ndarray
   """
   
   model_mat = read_mmat(should_cache,cache_dir,file,fi,terms,has_intercept,
                         ltx,irstx,stx,rtx,var_types,var_map,var_mins,
                         var_maxs,factor_levels,cov_flat_file,cov)
   
   eta_file = (model_mat @ coef).reshape(-1,1)
   return eta_file

def read_eta(file,formula:Formula,coef:np.ndarray,nc:int) -> np.ndarray:
   """Computes ``X@coef`` in parallel, where ``X`` is the model matrix based on this ``file`` and ``coef`` is the current coefficient estimate.

   :param file: File name
   :type file: str
   :param formula: Formula of model
   :type formula: Formula
   :param coef: Current coefficient estimate
   :type coef: np.ndarray
   :param nc: Number of cores to use
   :type nc: int
   :return: X@coef
   :rtype: np.ndarray
   """

   terms = formula.terms
   has_intercept = formula.has_intercept
   ltx = formula.get_linear_term_idx()
   irstx = []
   stx = formula.get_smooth_term_idx()
   rtx = formula.get_random_term_idx()
   var_types = formula.get_var_types()
   var_map = formula.get_var_map()
   var_mins = formula.get_var_mins()
   var_maxs = formula.get_var_maxs()
   factor_levels = formula.get_factor_levels()

   cov = None

   # Read file
   file_dat = pd.read_csv(file)

   # Encode data in this file
   _,cov_flat_file,NAs_flat_file,_,_,_,_ = formula.encode_data(file_dat)
   cov_flat_file = cov_flat_file[NAs_flat_file,:]

   # Parallelize over sub-sets of this file
   rows,_ = cov_flat_file.shape
   cov_flat_files = np.array_split(cov_flat_file,min(nc,rows),axis=0)
   subsets = [i for i in range(len(cov_flat_files))]

   with mp.Pool(processes=nc) as pool:
      # Build eta with all information from the formula - but only for sub-set of rows in this file
      etas = pool.starmap(form_eta_mp,zip(repeat(SHOULD_CACHE),repeat(CACHE_DIR),repeat(file),subsets,
                                          repeat(coef),repeat(terms),repeat(has_intercept),
                                          repeat(ltx),repeat(irstx),repeat(stx),
                                          repeat(rtx),repeat(var_types),repeat(var_map),
                                          repeat(var_mins),repeat(var_maxs),repeat(factor_levels),
                                          cov_flat_files,repeat(cov)))

   eta = []
   for eta_file in etas:
      eta.extend(eta_file)

   return eta

def keep_eta(formula:Formula,coef:np.ndarray,nc:int) -> np.ndarray:
   """Computes ``X@coef`` in parallel, where ``X`` is the overall model matrix and ``coef`` is current coefficient estimate.

   :param formula: Formula of model
   :type formula: Formula
   :param coef: Current coefficient estimate
   :type coef: np.ndarray
   :param nc: Number of cores to use
   :type nc: int
   :return: X@coef
   :rtype: np.ndarray
   """

   terms = formula.terms
   has_intercept = formula.has_intercept
   ltx = formula.get_linear_term_idx()
   irstx = []
   stx = formula.get_smooth_term_idx()
   rtx = formula.get_random_term_idx()
   var_types = formula.get_var_types()
   var_map = formula.get_var_map()
   var_mins = formula.get_var_mins()
   var_maxs = formula.get_var_maxs()
   factor_levels = formula.get_factor_levels()

   cov = None

   cov_split = np.array_split(formula.cov_flat[formula.NOT_NA_flat],len(formula.file_paths),axis=0)
   eta = []
   for fi,_ in enumerate(cov_split):
      rows,_ = cov_split[fi].shape
      cov_flat_files = np.array_split(cov_split[fi],min(nc,rows),axis=0)
      subsets = [i for i in range(len(cov_flat_files))]

      with mp.Pool(processes=min(rows,nc)) as pool:
         # Build eta with all information from the formula - but only for sub-set of rows
         etas = pool.starmap(form_eta_mp,zip(repeat(SHOULD_CACHE),repeat(CACHE_DIR),repeat(f"/outer_split_{fi}.csv"),subsets,
                                             repeat(coef),repeat(terms),repeat(has_intercept),
                                             repeat(ltx),repeat(irstx),repeat(stx),
                                             repeat(rtx),repeat(var_types),repeat(var_map),
                                             repeat(var_mins),repeat(var_maxs),repeat(factor_levels),
                                             cov_flat_files,repeat(cov)))


      for eta_split in etas:
         eta.extend(eta_split)

   return eta


def solve_gamm_sparse2(formula:Formula,penalties:list[LambdaTerm],col_S:int,family:Family,
                       maxiter:int=10,pinv:str="svd",conv_tol:float=1e-7,
                       extend_lambda:bool=False,control_lambda:int=1,
                       exclude_lambda:bool=False,extension_method_lam:str = "nesterov",
                       form_Linv:bool=True,progress_bar:bool=False,n_c:int=10) -> tuple[np.ndarray,np.ndarray,np.ndarray,scp.sparse.csc_array,float,scp.sparse.csc_array|None,float,list[float],float,Fit_info]:
   """Estimates an Additive Mixed model. Implements the algorithms discussed in section 3.1 of the paper by Krause et al. (submitted).

   Relies on methods proposed by Wood et al. (2017), Wood & Fasiolo (2017), Wood (2011), and Wood (2017). In addition, this function builds the products involving the model matrix only once (iteratively) as described
   by Wood et al. (2015).

   References:
      - Wood, S. N., Li, Z., Shaddick, G., & Augustin, N. H. (2017). Generalized Additive Models for Gigadata: Modeling the U.K. Black Smoke Network Daily Data. https://doi.org/10.1080/01621459.2016.1195744
      - Wood, S. N., & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. https://doi.org/10.1111/biom.12666
      - Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models: Estimation of Semiparametric Generalized Linear Models. https://doi.org/10.1111/j.1467-9868.2010.00749.x
      - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
      - Wood, S. N., Goude, Y., & Shaw, S. (2015). Generalized additive models for large data sets. Journal of the Royal Statistical Society: Series C (Applied Statistics), 64(1), 139–155. https://doi.org/10.1111/rssc.12068
      - Krause et al. (submitted). The Mixed-Sparse-Smooth-Model Toolbox (MSSM): Efficient Estimation and Selection of Large Multi-Level Statistical Models. https://doi.org/10.48550/arXiv.2506.13132

   :param formula: Formula of the model
   :type formula: Formula
   :param penalties: List of penalties
   :type penalties: list[LambdaTerm]:param penalties: _description_
   :type penalties: list[LambdaTerm]
   :param col_S: Columns of total penalty matrix
   :type col_S: int
   :param family: Family of model
   :type family: Family
   :param maxiter: Maximum number of iterations for outer algorithm updating lambda, defaults to 10
   :type maxiter: int, optional
   :param pinv: Method to use to compute generalzied inverse of total penalty,, defaults to "svd"
   :type pinv: str, optional
   :param conv_tol: Convergence tolerance, defaults to 1e-7
   :type conv_tol: float, optional
   :param extend_lambda: Whether lambda proposals should be accelerated or not. Can lower the number of new smoothing penalty proposals necessary. Disabled by default.
   :type extend_lambda: bool
   :param control_lambda: Whether lambda proposals should be checked (and if necessary decreased) for whether or not they (approxiately) increase the Laplace approximate restricted maximum likelihood of the model. Setting this to 0 disables control. Setting it to 1 means the step will never be smaller than the original EFS update but extensions will be removed in case the objective was exceeded. Setting it to 2 means that steps will be halved if it fails to increase the approximate REML. Set to 1 by default.
   :type control_lambda: int
   :param exclude_lambda: Whether selective lambda terms should be excluded heuristically from updates. Can make each iteration a bit cheaper but is problematic when using additional Kernel penalties on terms. Thus, disabled by default.
   :type exclude_lambda: bool
   :param extension_method_lam: Which method to use to extend lambda proposals., defaults to "nesterov"
   :type extension_method_lam: str, optional
   :param form_Linv: Whether to form the inverse of the cholesky of the negative penalzied hessian or not, defaults to True
   :type form_Linv: bool, optional
   :param progress_bar: Whether to print progress or not, defaults to False
   :type progress_bar: bool, optional
   :param n_c: Number of cores to use, defaults to 10
   :type n_c: int, optional
   :return: An estimate of the coefficients coef, the linear predictor eta, the working residuals wres, the negative hessian, the estimated scale, an inverse of the cholesky of the negative penalized hessian, total edf, term-wise edfs, total penalty, a :class:`Fit_info` object
   :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, scp.sparse.csc_array, float, scp.sparse.csc_array|None, float, list[float], float, Fit_info]
   """
   # Estimates a penalized additive mixed model, following the steps outlined in Wood (2017)
   # "Generalized Additive Models for Gigadata" but builds X.T @ X, and X.T @ y iteratively - and only once.
   setup_cache(CACHE_DIR,SHOULD_CACHE)
   n_c = min(mp.cpu_count(),n_c)

   y_flat = []
   rowsX = 0


   if formula.keep_cov:
      y_flat_np = formula.y_flat[formula.NOT_NA_flat].reshape(-1,1)
      rowsX = len(y_flat_np)

      # Compute X.T@X and X.T@y
      XX,Xy = keep_XTX(formula.cov_flat[formula.NOT_NA_flat,:],y_flat_np,formula,n_c,progress_bar)
      y_flat.extend(y_flat_np)
      y_flat_np = None
   else:
      iterator = formula.file_paths
      if progress_bar:
         iterator = tqdm(iterator,desc="Accumulating X.T @ X",leave=True)

      for fi,file in enumerate(iterator):
         # Read file
         file_dat = pd.read_csv(file)

         # Encode data in this file
         y_flat_file,_,NAs_flat_file,_,_,_,_ = formula.encode_data(file_dat)
         y_flat.extend(y_flat_file[NAs_flat_file])

         # Build the model matrix products with all information from the formula - but only for sub-set of rows in this file
         XX0,Xy0,rowsX0 = read_XTX(file,formula,n_c)
         rowsX += rowsX0
         # Compute X.T@X and X.T@y
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
   was_extended = [False for _ in enumerate(penalties)]

   # mu and eta (start estimates in case the family is not Gaussian)
   y = np.array(y_flat)
   mu = np.array(y_flat)
   eta = mu

   # Compute starting estimates
   dev,pen_dev,eta,mu,coef,CholXXS,InvCholXXS,total_edf,term_edfs,scale,wres,lam_delta,S_emb = init_step_gam(y,Xy,mu,eta,rowsX,colsX,None,XX,
                                                                                                     family,col_S,penalties,
                                                                                                     pinv,n_c,formula,form_Linv,"Chol",0,None)
   
   # Initialize extension variable
   extend_by = initialize_extension(extension_method_lam,penalties)
   was_extended = [False for _ in enumerate(penalties)]

   # Loop to optimize smoothing parameter (see Wood, 2017)
   iterator = range(maxiter)
   if progress_bar:
      iterator = tqdm(iterator,desc="Fitting",leave=True)

   # Keep track on some useful info
   fit_info = Fit_info()
   for o_iter in iterator:

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
         dev,pen_dev,mu,eta,coef = correct_coef_step(coef,n_coef,dev,pen_dev,c_dev_prev,family,eta,mu,y,None,len(penalties),S_emb,formula,n_c,0)

         # Test for convergence (Step 2 in Wood, 2017)
         dev_diff = abs(pen_dev - prev_pen_dev)

         if progress_bar:
            iterator.set_description_str(desc="Fitting - Conv.: " + "{:.2e}".format(dev_diff - conv_tol*pen_dev), refresh=True)
            
         if dev_diff < conv_tol*pen_dev:
            if progress_bar:
               iterator.set_description_str(desc="Converged!", refresh=True)
               iterator.close()
            fit_info.code = 0
            break
      
      # We need the deviance and penalized deviance of the model at this point (before completing steps 5-7 (dev_{old} in WGS used for convergence control)
      # for coef step control (step 3 in Wood, 2017) and convergence control (step 2 in Wood, 2017 based on step 4 in Wood, Goude, & Shaw, 2016) respectively
      prev_dev = dev
      prev_pen_dev = pen_dev
         
      # Step length control for proposed lambda change
      if len(penalties) > 0: 
         
         # Test the lambda update
         for lti,lTerm in enumerate(penalties):
            lTerm.lam += lam_delta[lti][0]
            #print(lTerm.lam,lam_delta[lti][0])

         # Now check step length and compute lambda + coef update.
         _,_,_,_,eta,mu,n_coef,CholXXS,InvCholXXS,total_edf,term_edfs,scale,wres,lam_delta,extend_by,penalties,was_extended,S_emb,lam_checks,_,_ = correct_lambda_step(y,Xy,None,None,rowsX,colsX,None,XX,coef,
                                                                                                                                                                        None,family,col_S,S_emb,penalties,
                                                                                                                                                                        was_extended,pinv,lam_delta,
                                                                                                                                                                        extend_by,o_iter,dev,
                                                                                                                                                                        n_c,control_lambda,extend_lambda,
                                                                                                                                                                        exclude_lambda,extension_method_lam,
                                                                                                                                                                        formula,form_Linv,"Chol",0,1)
         fit_info.lambda_updates += lam_checks
      else:
         # If there are no penalties simply perform a newton step
         # for the coefficients only
         eta,mu,n_coef,\
         CHOLXXS,\
         InvCholXXS,\
         _,\
         _,\
         total_edf,\
         term_edfs,\
         _,scale,wres,\
         _, _ = update_coef_and_scale(y,Xy,None,None,rowsX,colsX,
                                            None,XX,None,family,S_emb,None,None,None,
                                            penalties,n_c,formula,form_Linv,0)

      fit_info.iter += 1

   # Final penalty
   if len(penalties) > 0:
      penalty = coef.T @ S_emb @ coef
   else:
      penalty = 0

   # Final term edf
   if not term_edfs is None:
      term_edfs = calculate_term_edf(penalties,term_edfs)

   clear_cache(CACHE_DIR,SHOULD_CACHE)

   return coef,eta,wres,XX,scale,InvCholXXS,total_edf,term_edfs,penalty,fit_info

################################################ GAMMLSS code ################################################

def deriv_transform_mu_eta(y:np.ndarray,means:list[np.ndarray],family:GAMLSSFamily) -> tuple[list[np.ndarray],list[np.ndarray],list[np.ndarray]]:
   """Compute derivatives (first and second order) of llk with respect to each linear predictor based on their respective mean for all observations following steps outlined by Wood, Pya, & Säfken (2016)

    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.

   :param y: Vector of observations
   :type y: np.ndarray
   :param means: List holding vectors of mean estimates
   :type means: list[np.ndarray]
   :param family: Family of the model
   :type family: GAMLSSFamily
   :return: A tuple containing a list containing the first order partial derivatives with respect to each parameter, the same for pure second derivatives, and a list containing mixed derivatives
   :rtype: tuple[list[np.ndarray],list[np.ndarray],list[np.ndarray]]
   """ 
   with warnings.catch_warnings(): # Catch warnings associated with derivative evaluation, we handle those below
      warnings.simplefilter("ignore")

      d1 = [fd1(y,*means) for fd1 in family.d1]
      d2 = [fd2(y,*means) for fd2 in family.d2]
      d2m = [fd2m(y,*means) for fd2m in family.d2m]

      # Link derivatives
      ld1 = [family.links[mui].dy1(means[mui]) for mui in range(len(means))]
      ld2 = [family.links[mui].dy2(means[mui]) for mui in range(len(means))]

   # Transform first order derivatives via A.1 in Wood, Pya, & Säfken (2016)
   """
   WPS (2016) provide that $l_{\\eta}$ is obtained as $l^i_{\\mu}/h'(\\mu^i)$ - where $h'$ is the derivative of the link function $h$.
   This follows from applying the chain rule and the inversion rule of derivatives
   $\frac{\\partial llk(h^{-1}(\\eta))}{\\partial \\eta} = \frac{\\partial llk(\\mu)}{\\partial \\mu} \frac{\\partial h^{-1}(\\eta)}{\\partial \\eta} = \frac{\\partial llk(\\mu)}{\\partial \\mu}\frac{1}{\frac{\\partial h(\\mu)}{\\mu}}$.
   """
   # d1eta = [d1[mui]/ld1[mui] for mui in range(len(means))]
   d1eta = []
   for mui in range(len(means)):
      
      with warnings.catch_warnings(): # Divide by 0
         warnings.simplefilter("ignore")
         de = d1[mui]/ld1[mui]
         #d1eta.append(d1[mui]/ld1[mui])

      de[np.isnan(de) | np.isinf(de)] = 0 
      d1eta.append(de)

   # Pure second order derivatives are transformed via A.2 in WPS (2016)
   """
   For second derivatives we need pure and mixed. Computation of $l^\\mathbf{i}_{\\eta^l,\\eta^m}$ in general is again obtained by applying the steps outlined for first order and provided by WPS (2016)
   for the pure case. For the mixed case it is even simpler: We need $\frac{\\partial^2 llk(h_1^{-1}(\\eta^1),h_2^{-1}(\\eta^2))}{\\partial \\eta^1 \\partial \\eta^2}$,
   which is $\frac{\\partial llk / \\partial \\eta^1}{\\partial \\eta^2}$. With the first partial being equal to
   $\frac{\\partial llk}{\\partial \\mu^1}\frac{1}{\frac{\\partial h_1(\\mu^1)}{\\mu^1}}$ (see comment above for first order) we now have
   $\frac{\\partial \frac{\\partial llk}{\\partial \\mu^1}\frac{1}{\frac{\\partial h_1(\\mu^1)}{\\mu^1}}}{\\partial \\eta^2}$.
   
   We now apply the product rule (the second term in the sum disappears, because
   $\\partial \frac{1}{\frac{\\partial h_1(\\mu^1)}{\\mu^1}} / \\partial \\eta^2 = 0$, this is not the case for pure second derivatives as shown in WPS, 2016)
   to get $\frac{\\partial \frac{\\partial llk}{\\partial \\mu^1}}{\\partial \\eta^2} \frac{1}{\frac{\\partial h_1(\\mu^1)}{\\mu^1}}$.
   We can now again rely on the same steps taken to get the first derivatives (chain rule + inversion rule) to get
   $\frac{\\partial \frac{\\partial llk}{\\partial \\mu^1}}{\\partial \\eta^2} =
   \frac{\\partial^2 llk}{\\partial \\mu^1 \\partial \\mu^2}\frac{1}{\frac{\\partial h_2(\\mu^2)}{\\mu^2}}$.
   
   Thus, $\frac{\\partial llk / \\partial \\eta^1}{\\partial \\eta^2} =
   \frac{\\partial^2 llk}{\\partial \\mu^1 \\partial \\mu^2}\frac{1}{\frac{\\partial h_2(\\mu^2)}{\\mu^2}}\frac{1}{\frac{\\partial h_1(\\mu^1)}{\\mu^1}}$.
   """
   # d2eta = [d2[mui]/np.power(ld1[mui],2) - d1[mui]*ld2[mui]/np.power(ld1[mui],3) for mui in range(len(means))]
   d2eta = []
   for mui in range(len(means)):
      
      with warnings.catch_warnings(): # Divide by 0
         warnings.simplefilter("ignore")
         d2e = d2[mui]/np.power(ld1[mui],2) - d1[mui]*ld2[mui]/np.power(ld1[mui],3)
         #d2eta.append(d2[mui]/np.power(ld1[mui],2) - d1[mui]*ld2[mui]/np.power(ld1[mui],3))
   
      d2e[np.isnan(d2e) | np.isinf(d2e)] = 0
      d2eta.append(d2e)

   # Mixed second derivatives thus also are transformed as proposed by WPS (2016)
   d2meta = []
   mixed_idx = 0
   for mui in range(len(means)):
      for muj in range(mui+1,len(means)):
         
         with warnings.catch_warnings(): # Divide by 0
            warnings.simplefilter("ignore")
            d2em = d2m[mixed_idx] * (1/ld1[mui]) * (1/ld1[muj])
            #d2meta.append(d2m[mixed_idx] * (1/ld1[mui]) * (1/ld1[muj]))
         
         d2em[np.isnan(d2em) | np.isinf(d2em)] = 0
         d2meta.append(d2em)

         mixed_idx += 1

   return d1eta,d2eta,d2meta

def deriv_transform_eta_beta(d1eta:list[np.ndarray],d2eta:list[np.ndarray],d2meta:list[np.ndarray],Xs,only_grad=False):
   """
   Further transforms derivatives of llk with respect to eta to get derivatives of llk with respect to coefficients
   Based on section 3.2 and Appendix A in Wood, Pya, & Säfken (2016)

   References:
   - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
   """

   # Gradient: First order partial derivatives of llk with respect to coefficients
   """
   WPS (2016) provide $l^j_{\beta^l} = l^\\mathbf{i}_{\\eta^l}\\mathbf{X}^l_{\\mathbf{i},j}$. See ```deriv_transform_mu_eta```.
   """
   grad = []

   for etai in range(len(d1eta)):
      # Naive:
      # for j in range(Xs[etai].shape[1]):
         # grad.append(np.sum([d1eta[etai][i]*Xs[etai][i,j] for i in range(Xs[etai].shape[0])]))
         # but this is just product of d1eta[etai][:] and Xs[etai], so we can skip inner loop
      #val_idx = ((np.isnan(d1eta[etai]) | np.isinf(d1eta[etai])) == False).flatten()
      #grad.extend(((d1eta[etai][val_idx]).T @ Xs[etai][val_idx,:]).T)
      grad.extend((d1eta[etai].T @ Xs[etai]).T)
   
   if only_grad:
      return np.array(grad).reshape(-1,1),None
   # Hessian: Second order partial derivatives of llk with respect to coefficients
   """
   WPS (2016) provide in general:
   $l^{j,k}_{\beta^l,\beta^m} = l^\\mathbf{i}_{\\eta^l,\\eta^m}\\mathbf{X}^l_{\\mathbf{i},j}\\mathbf{X}^m_{\\mathbf{i},k}$.
   See ```deriv_transform_mu_eta```.
   """

   mixed_idx = 0
   
   hr_idx = 0
   h_rows = []
   h_cols = []
   h_vals = []
   for etai in range(len(d1eta)):
      hc_idx = 0
      for etaj in range(len(d1eta)):

         if etaj < etai:
            hc_idx += Xs[etaj].shape[1]
            continue

         if etai == etaj:
            # Pure 2nd
            d2 = d2eta[etai]
         else:
            # Mixed partial
            d2 = d2meta[mixed_idx]
            mixed_idx += 1

         val_idx = ((np.isnan(d2) | np.isinf(d2)) == False).flatten()

         for coefi in range(Xs[etai].shape[1]):
               # More efficient computation now, no longer an additional nested loop over coefj..
               #d2beta = (d2[val_idx]*Xs[etai][:,[coefi]][val_idx]).T @ Xs[etaj][val_idx,coefi:Xs[etaj].shape[1]]
               d2beta = (d2*Xs[etai][:,[coefi]]).T @ Xs[etaj][:,(coefi if etai == etaj else 0):Xs[etaj].shape[1]]
               
               if d2beta.nnz > 0:
               
                  # Sort, to make symmetric deriv extraction easier
                  if not d2beta.has_sorted_indices:
                     d2beta.sort_indices()
                  
                  # Get non-zero column entries for current row
                  cols = d2beta.indices[d2beta.indptr[0]:d2beta.indptr[1]] + (coefi if etai == etaj else 0)
                  
                  # Get non-zero values for current row in sorted order
                  vals = d2beta.data[d2beta.indptr[0]:d2beta.indptr[1]]

                  h_rows.extend(np.tile(coefi,d2beta.nnz) + hr_idx)
                  h_cols.extend(cols + hc_idx)
                  h_vals.extend(vals)

                  # Symmetric 2nd deriv..
                  if etai == etaj and ((cols[0] + hc_idx) == (coefi + hr_idx)): # For diagonal block need to skip diagonal element if present (i.e., non-zero)
                     h_rows.extend(cols[1:] + hc_idx)
                     h_cols.extend(np.tile(coefi,d2beta.nnz-1) + hr_idx)
                     h_vals.extend(vals[1:])
                  else: # For off-diagonal block can assign everything
                     h_rows.extend(cols + hc_idx)
                     h_cols.extend(np.tile(coefi,d2beta.nnz) + hr_idx)
                     h_vals.extend(vals)

         hc_idx += Xs[etaj].shape[1]
      hr_idx += Xs[etai].shape[1]
    
   hessian = scp.sparse.csc_array((h_vals,(h_rows,h_cols)))
   return np.array(grad).reshape(-1,1),hessian

def newton_coef_smooth(coef:np.ndarray,grad:np.ndarray,H:scp.sparse.csc_array,S_emb:scp.sparse.csc_array) -> tuple[np.ndarray,scp.sparse.csc_array,scp.sparse.csc_array,float]:
   """ Follows sections 3.1.2 and 3.14 in Wood, Pya, & Säfken (2016) to update the coefficients of a GAMLSS/GSMM model via a newton step.

   1) Computes gradient of the penalized likelihood (grad - S_emb@coef)
   2) Computes negative Hessian of the penalized likelihood (-1*H + S_emb) and it's inverse.
   3) Uses these two to compute the Netwon step.
   4) Step size control - happens outside

   References:
      - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
      - mgcv source code, in particular: https://github.com/cran/mgcv/blob/master/R/gam.fit4.r
   
   :param coef: Current coefficient estimate
   :type coef: np.ndarray
   :param grad: gradient of llk with respect to coef
   :type grad: np.ndarray
   :param H: hessian of the llk
   :type H: scp.sparse.csc_array
   :param S_emb: Total penalty matrix
   :type S_emb: scp.sparse.csc_array
   :return: A tuple containing an estimate of the coefficients, the un-pivoted cholesky of the penalized negative hessian, the inverse of the former, the multiple (float) added to the diagonal of the negative penalized hessian to make it invertible
   :rtype: tuple[np.ndarray,scp.sparse.csc_array,scp.sparse.csc_array,float]
   """   
    
   pgrad = np.array([grad[i] - (S_emb[[i],:]@coef)[0] for i in range(len(grad))])
   nH = -1*H + S_emb

   # Below tries diagonal pre-conditioning as implemented in mgcv's gam.fit5 function,
   # see: https://github.com/cran/mgcv/blob/master/R/gam.fit4.r#L1028. However, this does not work well for me, so I must be
   # doing something wrong.. The diagonal pre-conditioning as suggested by WPS (2016) does work. So for
   # now this is what happens below the comment.
   """
   nHdgr = nH.diagonal()
   mD = np.min(nHdgr)
   ill_def = False
   if mD <= 0:
      mAcc = np.max(nHdgr)*np.power(np.finfo(float).eps,0.5)
      if (-mD) < mAcc:
         nHdgr[nHdgr < mAcc] = mAcc
      else:
         ill_def = True
   
   if ill_def:
      nH += np.abs(mD)*scp.sparse.identity(nH.shape[1],format='csc') + np.abs(mAcc)*scp.sparse.identity(nH.shape[1],format='csc')
      D = scp.sparse.diags(np.ones_like(nHdgr))
      DI = D
   else:
   D = scp.sparse.diags(np.power(nHdgr,-0.5))
   DI = scp.sparse.diags(1/np.power(nHdgr,-0.5)) # For cholesky
   """
   
   # Diagonal pre-conditioning as suggested by WPS (2016)
   nHdgr = nH.diagonal()
   nHdgr = np.power(np.abs(nHdgr),-0.5)
   D = scp.sparse.diags(nHdgr)
   DI = scp.sparse.diags(1/nHdgr)

   nH2 = (D@nH@D).tocsc()
   #print(max(np.abs(nH.diagonal())),max(np.abs(nH2.diagonal())))

   # Compute V, inverse of nH
   eps = 0
   code = 1
   while code != 0:
      
      Lp, Pr, code = cpp_cholP(nH2+eps*scp.sparse.identity(nH2.shape[1],format='csc'))
      P = compute_eigen_perm(Pr)

      if code != 0:
         # Adjust identity added to nH
         if eps == 0:
            eps += 1e-14
         else:
            eps *= 2
         continue
      
      LVp = compute_Linv(Lp,10)
      LV = apply_eigen_perm(Pr,LVp)
      V = LV.T @ LV

   # Undo conditioning.
   V = D@V@D
   LV = LV@D

   # Update coef
   n_coef = coef + (V@pgrad)

   """
   if ill_def and eps == 0: # Initial fix was enough
      eps = np.abs(mD) + np.abs(mAcc)
   """
   return n_coef,DI@P.T@Lp,LV,eps

def gd_coef_smooth(coef:np.ndarray,grad:np.ndarray,S_emb:scp.sparse.csc_array,a:float) -> np.ndarray:
   """
   Follows sections 3.1.2 and 3.14 in WPS (2016) to update the coefficients of a GAMLSS/GSMM model via a Gradient descent (ascent actually) step.

   1) Computes gradient of the penalized likelihood (grad - S_emb@coef)
   3) Uses this to compute update
   4) Step size control - happens outside

   References:
      - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.

   :param coef: Current coefficient estimate
   :type coef: np.ndarray
   :param grad: gradient of llk with respect to coef
   :type grad: np.ndarray
   :param S_emb: Total penalty matrix
   :type S_emb: scp.sparse.csc_array
   :param a: Step length for gradient descent update
   :type a: float
   :return: An updated estimate of the coefficients
   :rtype: np.ndarray
   """
    
   # Compute penalized gradient
   pgrad = np.array([grad[i] - (S_emb[[i],:]@coef)[0] for i in range(len(grad))])
   
   # Update coef
   n_coef = coef + a * pgrad
   
   return n_coef

def correct_coef_step_gammlss(family:GAMLSSFamily,y:np.ndarray,Xs:list[scp.sparse.csc_array],coef:np.ndarray,next_coef:np.ndarray,coef_split_idx:list[int],c_llk:float,S_emb:scp.sparse.csc_array,a:float) -> tuple[np.ndarray,list[np.ndarray],list[np.ndarray],list[np.ndarray],float,float,float]:
   """
   Apply step size correction to Newton update for GAMLSS models, as discussed by WPS (2016).

   References:
   - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.

   :param family: Family of model
   :type family: GAMLSSFamily
   :param y: Vector of observations
   :type y: np.ndarray
   :param Xs: List of model matrices
   :type Xs: list[scp.sparse.csc_array]
   :param coef: Current coefficient estimate
   :type coef: np.ndarray
   :param next_coef: Updated coefficient estimate
   :type next_coef: np.ndarray
   :param coef_split_idx: The indices at which to split the overall coefficient vector into separate lists - one per parameter.
   :type coef_split_idx: list[int]
   :param c_llk: Current log likelihood
   :type c_llk: float
   :param S_emb: Total penalty matrix
   :type S_emb: scp.sparse.csc_array
   :param a: Step length for gradient descent update
   :type a: float
   :return: A tuple containing the corrected coefficient estimate ``next_coef``,``next_coef`` split via ``coef_split_idx``,next mus,next etas,next llk,nex penalized llk, updated step length fro next gradient update
   :rtype: tuple[np.ndarray,list[np.ndarray],list[np.ndarray],list[np.ndarray],float,float,float]
   """
   # Update etas and mus
   next_split_coef = np.split(next_coef,coef_split_idx)
   next_etas = [Xs[i]@next_split_coef[i] for i in range(family.n_par)]

   with warnings.catch_warnings(): # Catch warnings associated with mean transformation
      warnings.simplefilter("ignore")
      next_mus = [family.links[i].fi(next_etas[i]) for i in range(family.n_par)]

   # Find and exclude invalid indices before evaluating llk
   inval = np.isnan(next_mus[0])

   if len(coef_split_idx) != 0:
      for mi in range(1,len(next_mus)):
         inval = inval |  np.isnan(next_mus[mi])
   inval = inval.flatten()
   
   # Step size control for newton step.
   next_llk = family.llk(y[inval == False],*[nmu[inval == False] for nmu in next_mus])
   
   # Evaluate improvement of penalized llk under new and old coef - but in both
   # cases for current lambda (see Wood, Li, Shaddick, & Augustin; 2017)
   next_pen_llk = next_llk - 0.5*next_coef.T@S_emb@next_coef
   prev_llk_cur_pen = c_llk - 0.5*coef.T@S_emb@coef
    
   for n_checks in range(32):
      
      if next_pen_llk >= prev_llk_cur_pen and (np.isinf(next_pen_llk[0,0]) == False and np.isnan(next_pen_llk[0,0]) == False):
         break
      
      if n_checks > 30:
         next_coef = coef
         
      # Half it if we do not observe an increase in penalized likelihood (WPS, 2016)
      next_coef = (coef + next_coef)/2
      next_split_coef = np.split(next_coef,coef_split_idx)

      # Update etas and mus again
      next_etas = [Xs[i]@next_split_coef[i] for i in range(family.n_par)]
      
      with warnings.catch_warnings(): # Catch warnings associated with mean transformation
         warnings.simplefilter("ignore")
         next_mus = [family.links[i].fi(next_etas[i]) for i in range(family.n_par)]

      # Find and exclude invalid indices before evaluating llk
      inval = np.isnan(next_mus[0])

      if len(coef_split_idx) != 0:
         for mi in range(1,len(next_mus)):
            inval = inval |  np.isnan(next_mus[mi])
      inval = inval.flatten()
      
      # Re-evaluate penalized likelihood
      next_llk = family.llk(y[inval == False],*[nmu[inval == False] for nmu in next_mus])
      next_pen_llk = next_llk - 0.5*next_coef.T@S_emb@next_coef
      n_checks += 1

   # Update step-size for gradient
   if n_checks > 0 and a > 1e-9:
      a /= 2
   elif n_checks == 0 and a < 1:
      a *= 2
   
   return next_coef,next_split_coef,next_mus,next_etas,next_llk,next_pen_llk,a

def identify_drop(H:scp.sparse.csc_array,S_scaled:scp.sparse.csc_array,method:str='QR') -> tuple[list[int]|None,list[int]|None]:
   """
   Routine to (approximately) identify the rank of the scaled negative hessian of the penalized likelihood based on a rank revealing QR decomposition or the methods by Foster (1986) and Gotsman & Toledo (2008).
   
   If ``method=="QR"``, a rank revealing QR decomposition is performed for the scaled penalized Hessian. The latter has to be transformed to a dense matrix for this.
   This is essentially the approach by Wood et al. (2016) and is the most accurate. Alternatively, we can rely on a variant of Foster's method.
   This is done when ``method=="LU"`` or ``method=="Direct"``. ``method=="LU"`` requires ``p`` LU decompositions - where ``p`` is approximately the Kernel size of the matrix.
   Essentially continues to find vectors forming a basis of the Kernel of the balanced penalzied Hessian from the upper matrix of the LU decomposition and successively drops columns
   corresponding to the maximum absolute value of the Kernel vectors (see Foster, 1986). This is repeated until we can form a cholesky of the scaled penalized hessian which as an acceptable condition number.
   If ``method=="Direct"``, the same procedure is completed, but Kernel vectors are found directly based on the balanced penalized Hessian, which can be less precise. 

   References:
   - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
   - Foster (1986). Rank and null space calculations using matrix decomposition without column interchanges.
   - Gotsman & Toledo (2008). On the Computation of Null Spaces of Sparse Rectangular Matrices.
   - mgcv source code, in particular: https://github.com/cran/mgcv/blob/master/R/gam.fit4.r
   
   :param H: Estimate of the hessian of the log-likelihood.
   :type H: scp.sparse.csc_array
   :param S_scaled: Scaled version of the penalty matrix (i.e., unweighted total penalty divided by it's norm).
   :type S_scaled: scp.sparse.csc_array
   :param method: Which method to use to check for rank deficiency, defaults to 'QR'
   :type method: str, optional
   :return: A tuple containing lists of the coefficients to keep and to drop, both of which are None when we don't need to drop any.
   :rtype: tuple[list[int]|None,list[int]|None]
   """

   rank = H.shape[1]
   
   # Form scaled negative hessian of penalized likelihood to check for rank deficincy, as
   # reccomended by Wood et al. (2016).
   H_scaled = H / scp.sparse.linalg.norm(H,ord=None)

   nH_scaled = -1*H_scaled + S_scaled
   
   # again perform pre-conditioning as done by mgcv, see
   # https://github.com/cran/mgcv/blob/master/R/gam.fit4.r#L1167
   nHdgr = np.abs(nH_scaled.diagonal())
   nHdgr[nHdgr < np.power(np.finfo(float).eps,2)] = 1
   nHdgr = np.power(nHdgr,-0.5)
   D = scp.sparse.diags(nHdgr)
   nH_scaled = (D@nH_scaled@D).tocsc()

   keep = [cidx for cidx in range(H.shape[1])]
   drop = []

   if method == 'QR':
      # Perform dense QR decomposition with pivoting to estimate rank
      Pr, rank = cpp_dqrr(nH_scaled.toarray())

      if rank < nH_scaled.shape[1]:
         drop = Pr[rank:]
         keep = [cidx for cidx in range(H.shape[1]) if cidx not in drop]

      drop = np.sort(drop)
      keep = np.sort(keep)
   
      return keep,drop
    
   # Verify that we actually have a problem..
   Lp, Pr, code = cpp_cholP(nH_scaled)

   if code == 0:
      # Check condition number
      P = compute_eigen_perm(Pr)
      LVp = compute_Linv(Lp,10)
      LV = apply_eigen_perm(Pr,LVp)

      # Undo conditioning.
      LV = LV
      L = P.T@Lp

      K2,_,_,Kcode = est_condition(L,LV,verbose=False)
      if Kcode == 0:
         return keep, drop
    
   # Negative penalized hessian is not of full rank. Need to fix that.
   nH_drop = copy.deepcopy(nH_scaled)

   # Follow steps outlined in algorithm 1 of Foster:
   # - Optionally form LU decomposition of current penalized hessian matrix
   # - find approximate singular value (from nH or U) + vector -> vector is approximate Kernel vector
   # - drop column of current (here also row since nH is symmetric) matrix corresponding to maximum of approximate Kernel vector
   # - check if cholesky works now, otherwise continue

   while True:

      # Find Null-vector of U (can ignore row pivoting for LU, see: Gotsman & Toledo; 2008)
      if method == "LU": # Original proposal by Foster
         lu = scp.sparse.linalg.splu(nH_drop,permc_spec='COLAMD',diag_pivot_thresh=1,options=dict(SymmetricMode=False,IterRefine='Double',Equil=True))
         R = lu.U
         P = scp.sparse.csc_matrix((np.ones(nH_drop.shape[1]), (np.arange(nH_drop.shape[1]), lu.perm_c)))
      else:
         R = nH_drop

      # Find approximate Null-vector w of nH (See Foster, 1986)
      try:
         _,_,vh = scp.sparse.linalg.svds(R,k=1,return_singular_vectors=True,random_state=20,which='SM',maxiter=10000)
      except:
         try:
            _,_,vh = scp.sparse.linalg.svds(R,k=1,return_singular_vectors=True,random_state=20,which='SM',solver='lobpcg',maxiter=10000)
         except:
            # Solver failed.. get out and try again later.
            return keep, drop

      # w needs to be of original shape!
      w = np.zeros(H.shape[1])

      if method == "LU":
         wk = (vh@P.T).T # Need to undo column-pivoting here - depends on method used to compute pivot, here COLAMD.
      else:
         wk = vh # No pivoting if directly extracting vector from nH
      w[keep] = wk.flatten()

      # Drop next col + row and update keep list
      drop_c = np.argmax(np.abs(w))
      drop.append(drop_c)
      keep = [cidx for cidx in range(H.shape[1]) if cidx not in drop]
      #print(drop_c)

      nH_drop = nH_scaled[keep,:]
      nH_drop = nH_drop[:,keep]

      # Check if Cholesky works now and has good conditioning - otherwise continue
      Lp, Pr2, code = cpp_cholP(nH_drop)
      if code == 0:
         # Check condition number
         P2 = compute_eigen_perm(Pr2)
         LVp = compute_Linv(Lp,10)
         LV = apply_eigen_perm(Pr2,LVp)

         # Undo conditioning.
         LV = LV
         L = P2.T@Lp

         K2,_,_,Kcode = est_condition(L,LV,verbose=False)

         if Kcode == 0:
            break

   drop = np.sort(drop)
   keep = np.sort(keep)
    
   return keep,drop

def drop_terms_S(penalties:list[LambdaTerm],keep:list[int]) -> list[LambdaTerm]:
   """Zeros out rows and cols of penalty matrices corresponding to dropped terms. Roots are re-computed as well.

   :param penalties: List of Lambda terms included in the model formula
   :type penalties: list[LambdaTerm]
   :param keep: List of columns/rows to keep.
   :type keep: list[int]
   :return: List of updated penalties - a copy is made.
   :rtype: list[LambdaTerm]
   """
   # Don´t actually drop, just zero
   
   drop_pen = copy.deepcopy(penalties)

   for peni,pen in enumerate(drop_pen):
      start_idx = pen.start_index
      end_idx = start_idx + pen.rep_sj*pen.S_J.shape[1]
      # Identify coefficients kept for this penalty matrix. Take care to look
      # out for rep_sj...
      keep_pen = [cf-start_idx for cf in keep if cf >= start_idx and cf < end_idx]

      needs_drop = len(keep_pen) < (end_idx - start_idx)

      if needs_drop:
         # If we have repetitions then the easiest solution is the one below - but this
         # is far from efficient...
         if pen.rep_sj > 1:
            pen.S_J = pen.S_J_emb[start_idx:end_idx,start_idx:end_idx]
            pen.rep_sj = 1
         
         # Compute new reduced penalty
         rS_J = pen.S_J[keep_pen,:]
         rS_J = rS_J[:,keep_pen]

         # Re-compute root & rank - this will now be very costly for shared penalties.
         eig, U =scp.linalg.eigh(rS_J.toarray())
         pen.rank = sum([1 for e in eig if e >  sys.float_info.epsilon**0.7])

         rD_J = scp.sparse.csc_array(U@np.diag([e**0.5 if e > sys.float_info.epsilon**0.7 else 0 for e in eig]))

         # Re-embed in orig shape blocks
         keep_pen = np.array(keep_pen)

         Sdat,Srow,Scol = translate_sparse(rS_J)
         Ddat,Drow,Dcol = translate_sparse(rD_J)
         Srow = keep_pen[Srow]
         Scol = keep_pen[Scol]
         Drow = keep_pen[Drow]
         Dcol = keep_pen[Dcol]

         pen.S_J = scp.sparse.csc_array((Sdat,(Srow,Scol)),shape=penalties[peni].S_J.shape)
         #print(pen.S_J.toarray())
         #D_J = scp.sparse.csc_array((Ddat,(Drow,Dcol)),shape=penalties[peni].S_J.shape)

         #print("rPen",(D_J@D_J.T-pen.S_J).min(),(D_J@D_J.T-pen.S_J).max())

         # Now Re-embed in overall zero blocks again
         pen.D_J_emb,_ = embed_in_S_sparse(Ddat,Drow,Dcol,None,penalties[peni].S_J_emb.shape[1],penalties[peni].S_J.shape[1],start_idx)
         pen.S_J_emb,_ = embed_in_S_sparse(Sdat,Srow,Scol,None,penalties[peni].S_J_emb.shape[1],penalties[peni].S_J.shape[1],start_idx)
         #print("rPen2",(pen.D_J_emb@pen.D_J_emb.T-pen.S_J_emb).min(),(pen.D_J_emb@pen.D_J_emb.T-pen.S_J_emb).max())
   #print(drop_pen)
   return drop_pen


def drop_terms_X(Xs:list[scp.sparse.csc_array],keep:list[int]) -> tuple[list[scp.sparse.csc_array],list[int]]:
   """Drops cols of model matrices corresponding to dropped terms.

   :param Xs: List of model matrices included in the model formula.
   :type Xs: list[scp.sparse.csc_array]
   :param keep: List of columns to keep.
   :type keep: list[int]
   :return: Tuple, containing a list of updated model matrices - a copy is made - and a new list conatining the indices by which to split the coefficient vector.
   :rtype: tuple[list[scp.sparse.csc_array],list[int]]
   """
   # Drop from model matrices
   start_idx = 0
   split_idx = []
   new_Xs = []
   for Xi,X in enumerate(Xs):
      end_idx = start_idx + X.shape[1]

      keep_X = [cf-start_idx for cf in keep if cf >= start_idx and cf < end_idx]

      start_idx += X.shape[1]
      X = X[:,keep_X]
      #print(X.shape)
      new_Xs.append(X)

      if Xi == 0:
         split_idx.append(X.shape[1])
      elif Xi < len(Xs)-1:
         split_idx.append(X.shape[1] + split_idx[-1])
   
   return new_Xs,split_idx

def check_drop_valid_gammlss(y:np.ndarray,coef:np.ndarray,coef_split_idx:list[int],Xs:list[scp.sparse.csc_array],S_emb:scp.sparse.csc_array,keep:list[int],family:GAMLSSFamily) -> tuple[bool,float]:
   """Checks whether an identified set of coefficients to be dropped from the model results in a valid log-likelihood.

   :param y: Vector of response variable
   :type y: np.ndarray
   :param coef: Vector of coefficientss
   :type coef: np.ndarray
   :param coef_split_idx: List with indices to split coef - one per parameter of response distribution
   :type coef_split_idx: list[int]
   :param Xs: List of model matrices - one per parameter of response distribution
   :type Xs: list[scp.sparse.csc_array]
   :param S_emb: Total penalty matrix
   :type S_emb: scp.sparse.csc_array
   :param keep: List of coefficients to retain
   :type keep: list[int]
   :param family: Model family
   :type family: GAMLSSFamily
   :return: tuple holding bool indicating if likelihood is valid and penalized log-likelihood under dropped set.
   :rtype: tuple[bool,float]
   """
   # Drop from coef
   dropcoef = coef[keep]

   # ... from Xs...
   rXs,rcoef_split_idx = drop_terms_X(Xs,keep)
   #print(rXs)

   # ... and from S_emb
   rS_emb = S_emb[keep,:]
   rS_emb = rS_emb[:,keep]

   # Now re-compute split coef, etas, and mus
   rsplit_coef = np.split(dropcoef,rcoef_split_idx)
   etas = [rXs[i]@rsplit_coef[i] for i in range(family.n_par)]

   with warnings.catch_warnings(): # Catch warnings associated with mean transformation
      warnings.simplefilter("ignore")
      mus = [family.links[i].fi(etas[i]) for i in range(family.n_par)]

   # Find and exclude invalid indices before evaluating llk
   inval = np.isnan(mus[0])

   if len(coef_split_idx) != 0:
      for mi in range(1,len(mus)):
         inval = inval |  np.isnan(mus[mi])
   inval = inval.flatten()

   # Re-compute llk
   c_llk = family.llk(y[inval == False],*[mu[inval == False] for mu in mus])
   c_pen_llk = c_llk - 0.5*dropcoef.T@rS_emb@dropcoef

   if (np.isinf(c_pen_llk[0,0]) or np.isnan(c_pen_llk[0,0])):
      return False,None
   
   return True,c_pen_llk

def handle_drop_gammlss(family:GAMLSSFamily, y:np.ndarray, coef:np.ndarray, keep:list[int], Xs:list[scp.sparse.csc_array], S_emb:scp.sparse.csc_array) -> tuple[np.ndarray, list[np.ndarray], list[int], list[scp.sparse.csc_array], scp.sparse.csc_array, list[np.ndarray], list[np.ndarray], float, float]:
   """Drop coefficients and make sure this is reflected in the model matrices, total penalty, llk, and penalized llk.

   :param family: Model family
   :type family: GAMLSSFamily
   :param y: Vector of observations
   :type y: np.ndarray
   :param coef: Vector of coefficients
   :type coef: np.ndarray
   :param keep: List of parameter indices to keep.
   :type keep: list[int]
   :param Xs: List of model matrices
   :type Xs: list[scp.sparse.csc_array]
   :param S_emb: Total penalty matrix.
   :type S_emb: scp.sparse.csc_array
   :return: A tuple holding: reduced coef vector, split version of the reduced coef vector, a new list of indices determining where to split the reduced coef vector, list with reduced model matrices, reduced total penalty matrix, updated etas, mus, llk, and penalzied llk
   :rtype: tuple[np.ndarray, list[np.ndarray], list[int], list[scp.sparse.csc_array], scp.sparse.csc_array, list[np.ndarray], list[np.ndarray], float, float]
   """
   # Drop from coef
   coef = coef[keep]

   # ... from Xs...
   rXs,rcoef_split_idx = drop_terms_X(Xs,keep)
   #print(rXs)

   # ... and from S_emb
   rS_emb = S_emb[keep,:]
   rS_emb = rS_emb[:,keep]

   # Now re-compute split coef, etas, and mus
   rsplit_coef = np.split(coef,rcoef_split_idx)
   etas = [rXs[i]@rsplit_coef[i] for i in range(family.n_par)]

   with warnings.catch_warnings(): # Catch warnings associated with mean transformation
      warnings.simplefilter("ignore")
      mus = [family.links[i].fi(etas[i]) for i in range(family.n_par)]

   # Find and exclude invalid indices before evaluating llk
   inval = np.isnan(mus[0])

   if len(rcoef_split_idx) != 0:
      for mi in range(1,len(mus)):
         inval = inval |  np.isnan(mus[mi])
   inval = inval.flatten()

   # Re-compute llk
   c_llk = family.llk(y[inval == False],*[mu[inval == False] for mu in mus])
   c_pen_llk = c_llk - 0.5*coef.T@rS_emb@coef

   return coef, rsplit_coef, rcoef_split_idx, rXs, rS_emb, etas, mus, c_llk, c_pen_llk

def restart_coef_gammlss(coef:np.ndarray,split_coef:list[np.ndarray],c_llk:float,c_pen_llk:float,
                         etas:list[np.ndarray],mus:list[np.ndarray],n_coef:int,coef_split_idx:list[int],
                         y:np.ndarray,Xs:list[scp.sparse.csc_array],S_emb:scp.sparse.csc_array,
                         family:GAMLSSFamily,outer:int,restart_counter:int) -> tuple[np.ndarray, list[np.ndarray], float, float, list[np.ndarray], list[np.ndarray]]:
   """Shrink coef towards random vector to restart algorithm if it get's stuck.

   :param coef: Coefficient estimate
   :type coef: np.ndarray
   :param split_coef: Split of coefficient estimate
   :type split_coef: list[np.ndarray]
   :param c_llk: Current llk
   :type c_llk: float
   :param c_pen_llk: Current penalized llk
   :type c_pen_llk: float
   :param etas: List of linear predictors
   :type etas: list[np.ndarray]
   :param mus: List of estimated means
   :type mus: list[np.ndarray]
   :param n_coef: Number of coefficients
   :type n_coef: int
   :param coef_split_idx: List with indices to split coef - one per parameter of response distribution
   :type coef_split_idx: list[int]
   :param y: Vector of observations
   :type y: np.ndarray
   :param Xs: List of model matrices - one per parameter of response distribution
   :type Xs: [scp.sparse.csc_array]
   :param S_emb: Total penalty matrix
   :type S_emb: scp.sparse.csc_array
   :param family: Model family
   :type family: GAMLSSFamily
   :param outer: Outer iteration index
   :type outer: int
   :param restart_counter: Number of restarts already handled previously
   :type restart_counter: int
   :return: Updates for coef, split_coef, c_llk, c_pen_llk, etas, mus
   :rtype: tuple[np.ndarray, list[np.ndarray], float, float, list[np.ndarray], list[np.ndarray]]
   """
   res_checks = 0
   res_scale = 0.5 if outer <= 10 else 1/(restart_counter+2)
   #print("resetting conv.",res_scale)

   while res_checks < 30:
      res_coef = ((1-res_scale)*coef + res_scale*scp.stats.norm.rvs(size=n_coef,random_state=outer+res_checks).reshape(-1,1))

      # Re-compute llk
      with warnings.catch_warnings():
         warnings.simplefilter("ignore")

         # Update etas and mus
         res_split_coef = np.split(res_coef,coef_split_idx)
         res_etas = [Xs[i]@res_split_coef[i] for i in range(family.n_par)]

         with warnings.catch_warnings(): # Catch warnings associated with mean transformation
               warnings.simplefilter("ignore")
               res_mus = [family.links[i].fi(res_etas[i]) for i in range(family.n_par)]

         # Find and exclude invalid indices before evaluating llk
         inval = np.isnan(res_mus[0])

         if len(coef_split_idx) != 0:
            for mi in range(1,len(res_mus)):
               inval = inval |  np.isnan(res_mus[mi])
         inval = inval.flatten()
         
         # Step size control for newton step.
         res_llk = family.llk(y[inval == False],*[nmu[inval == False] for nmu in res_mus])
         
         # Evaluate improvement of penalized llk under new and old coef - but in both
         # cases for current lambda (see Wood, Li, Shaddick, & Augustin; 2017)
         res_pen_llk = res_llk - 0.5*res_coef.T@S_emb@res_coef

      if (np.isinf(res_pen_llk[0,0]) or np.isnan(res_pen_llk[0,0])):
         res_checks += 1
         continue
      
      coef = res_coef
      split_coef = res_split_coef
      c_llk = res_llk
      c_pen_llk = res_pen_llk
      etas = res_etas
      mus= res_mus
      break
   
   return coef, split_coef, c_llk, c_pen_llk, etas, mus
    
def update_coef_gammlss(family:GAMLSSFamily,mus:list[np.ndarray],y:np.ndarray,Xs,coef:np.ndarray,
                        coef_split_idx:list[int],S_emb:scp.sparse.csc_array,S_norm:scp.sparse.csc_array,
                        S_pinv:scp.sparse.csc_array,FS_use_rank:list[bool],gammlss_penalties:list[LambdaTerm],
                        c_llk:float,outer:int,max_inner:int,min_inner:int,conv_tol:float,method:str,
                        piv_tol:float,keep_drop:list[list[int],list[int]]|None) -> tuple[np.ndarray,list[np.ndarray],list[np.ndarray],list[np.ndarray],scp.sparse.csc_array,scp.sparse.csc_array,scp.sparse.csc_array,float,float,float,list[int] | None,list[int] | None]:
   """Repeatedly perform Newton update with step length control to the coefficient vector - essentially implements algorithm 3 from the paper by Krause et al. (submitted).
   
   Based on steps outlined by Wood, Pya, & Säfken (2016). Checks for rank deficiency when ``method != "Chol"``.

   References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
     - Krause et al. (submitted). The Mixed-Sparse-Smooth-Model Toolbox (MSSM): Efficient Estimation and Selection of Large Multi-Level Statistical Models. https://doi.org/10.48550/arXiv.2506.13132

   :param family: Family of model
   :type family: GAMLSSFamily
   :param mus: List of estimated means
   :type mus: list[np.ndarray]
   :param y: Vector of observations
   :type y: np.ndarray
   :param Xs: List of model matrices - one per parameter of response distribution
   :type Xs: [scp.sparse.csc_array]
   :param coef: Coefficient estimate
   :type coef: np.ndarray
   :param coef_split_idx: List with indices to split coef - one per parameter of response distribution
   :type coef_split_idx: list[int]
   :param S_emb: Total penalty matrix
   :type S_emb: scp.sparse.csc_array
   :param S_emb: Total penalty matrix - normalized/scaled for rank checks
   :type S_emb: scp.sparse.csc_array
   :param S_pinv: Generalized inverse of total penalty matrix
   :type S_pinv: scp.sparse.csc_array
   :param FS_use_rank: A list of bools indicating for which EFS updates the rank rather than the generalized inverse should be used
   :type FS_use_rank: list[bool]
   :param gammlss_penalties: List of penalties
   :type gammlss_penalties: list[LambdaTerm]
   :param c_llk: Current llk
   :type c_llk: float
   :param outer: Index of outer iteration
   :type outer: int
   :param max_inner: Maximum number of inner iterations
   :type max_inner: int
   :param min_inner: Minimum number of inner iterations
   :type min_inner: int
   :param conv_tol: Convergence tolerance
   :type conv_tol: float
   :param method: Method to use to estimate coefficients
   :type method: str
   :param piv_tol: Deprecated
   :type piv_tol: float
   :param keep_drop: Set of previously dropped coeeficients or None
   :type keep_drop: list[list[int],list[int]] | None
   :return: A tuple containing an estimate of all coefficients, a split version of the former, updated values for mus, etas, the negative hessian of the log-likelihood, cholesky of negative hessian of the penalized log-likelihood, inverse of the former, new llk, new penalized llk, the multiple (float) added to the diagonal of the negative penalized hessian to make it invertible, an optional list of the coefficients to keep, an optional list of the estimated coefficients to drop
   :rtype: tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray], scp.sparse.csc_array, scp.sparse.csc_array, scp.sparse.csc_array, float, float, float, list[int] | None, list[int] | None]
   """
   grad_only = method == "Grad"
   a = 0.1 # Step-size for gradient only

   if grad_only:
      H = None
      L = None
      LV = None
      eps = 0

   full_coef = None
   # Update coefficients:
   if keep_drop is None:
      keep = None
      drop = None
   else:
      keep = keep_drop[0]
      drop = keep_drop[1]

      # Handle previous drop
      full_coef = np.zeros_like(coef) # Prepare zeroed full coef vector
      full_coef_split_idx = copy.deepcopy(coef_split_idx) # Original split index
      coef, split_coef, coef_split_idx, Xs, S_emb, etas, mus, c_llk, c_pen_llk = handle_drop_gammlss(family, y, coef, keep, Xs, S_emb) # Drop

   converged = False
   checked_identifiable = False
   inner = 0
   while converged == False:

      if inner >= max_inner:
         break
      
      # Get derivatives with respect to eta
      if family.d_eta == False:
         d1eta,d2eta,d2meta = deriv_transform_mu_eta(y,mus,family)
      else:
         d1eta = [fd1(y,*mus) for fd1 in family.d1]
         d2eta = [fd2(y,*mus) for fd2 in family.d2]
         d2meta = [fd2m(y,*mus) for fd2m in family.d2m]

      # Get derivatives with respect to coef
      grad,H = deriv_transform_eta_beta(d1eta,d2eta,d2meta,Xs,only_grad=grad_only)

      # Update coef and perform step size control
      if grad_only:
         next_coef = gd_coef_smooth(coef,grad,S_emb,a)
      else:
         next_coef,L,LV,eps = newton_coef_smooth(coef,grad,H,S_emb)

      # Prepare to check convergence
      prev_llk_cur_pen = c_llk - 0.5*coef.T@S_emb@coef

      # Perform step length control
      coef,split_coef,mus,etas,c_llk,c_pen_llk,a = correct_coef_step_gammlss(family,y,Xs,coef,next_coef,coef_split_idx,c_llk,S_emb,a)

      # Very poor start estimate, restart
      if grad_only and outer == 0 and inner <= 20 and np.abs(c_pen_llk - prev_llk_cur_pen) < conv_tol*np.abs(c_pen_llk):
         coef, split_coef, c_llk, c_pen_llk, etas, mus = restart_coef_gammlss(coef,split_coef,c_llk, c_pen_llk,etas, mus,len(coef),coef_split_idx,y,Xs,S_emb,family,inner,0)
         a = 0.1

      if np.abs(c_pen_llk - prev_llk_cur_pen) < conv_tol*np.abs(c_pen_llk):
         converged = True
         
         # Check for drop
         if (keep_drop is None) and (checked_identifiable == False) and (method in ["QR/Chol","LU/Chol","Direct/Chol"]):
            keep,drop = identify_drop(H,S_norm,method.split("/")[0])

            # No drop necessary -> converged
            if len(drop) == 0:
               keep = None
               drop = None
               break
            else:
               # Found drop, but need to check whether it is safe
               drop_valid,drop_pen_llk = check_drop_valid_gammlss(y,coef,coef_split_idx,Xs,S_emb,keep,family)

               # Skip if likelihood becomes invalid
               if drop_valid == False:
                  keep = None
                  drop = None
                  break
               # If we identify all or all but one coefficients to drop also skip
               elif drop_valid and len(drop) >= len(coef) - 1:
                  keep = None
                  drop = None
                  break

               # At this point: found & accepted drop -> adjust parameters
               full_coef = np.zeros_like(coef) # Prepare zeroed full coef vector
               full_coef_split_idx = copy.deepcopy(coef_split_idx)
               coef,split_coef, coef_split_idx, Xs, S_emb, etas, mus, c_llk, c_pen_llk = handle_drop_gammlss(family, y, coef, keep, Xs, S_emb) # Drop
               converged = False # Re-iterate until convergence
               inner = -1 # Reset fitting iterations
               checked_identifiable = True
         
         # Simply converge
         else:
            break

      if eps <= 0 and outer > 0 and inner >= (min_inner-1):
         break # end inner loop and immediately optimize lambda again.

      inner += 1

   # Need to fill full_coef at this point and pad LV if we dropped
   if drop is not None:
      full_coef[keep] = coef
      
      LVdat,LVrow,LVcol = translate_sparse(LV) # LV.T@LV = V
      LVrow = keep[LVrow]
      LVcol = keep[LVcol]
      LV = scp.sparse.csc_array((LVdat,(LVrow,LVcol)),shape=(len(full_coef),len(full_coef)))
   else:
      # Full coef is simply coef
      full_coef = coef
      full_coef_split_idx = coef_split_idx

   # Make sure at convergence negative Hessian of llk is at least positive semi-definite as well
   checkH = True
   checkHc = 0

   # Compute lgdetDs once
   if gammlss_penalties is not None:
      lgdetDs = []

      for lti,lTerm in enumerate(gammlss_penalties):
         lt_rank = None
         if FS_use_rank[lti]:
            lt_rank = lTerm.rank

         lgdetD,_ = compute_lgdetD_bsb(lt_rank,lTerm.lam,S_pinv,lTerm.S_J_emb,full_coef)
         lgdetDs.append(lgdetD)

   while checkH and (gammlss_penalties is not None):

      # Re-compute ldetHS
      _,_, ldetHSs = calculate_edf(None,None,LV,gammlss_penalties,lgdetDs,len(full_coef),10,None,None)

      # And check whether Theorem 1 now holds
      checkH = np.any([lgdetDs[lti] - ldetHSs[lti] < 0 for lti in range(len(gammlss_penalties))])

      if checkH:
         if eps == 0:
            eps += 1e-14
         else:
            eps *= 2
         _,L,LV,_ = newton_coef_smooth(coef,grad,H - eps*scp.sparse.identity(H.shape[1],format='csc'),S_emb)

         # Pad new LV in case of drop again:
         if drop is not None:
            LVdat,LVrow,LVcol = translate_sparse(LV) # LV.T@LV = V
            LVrow = keep[LVrow]
            LVcol = keep[LVcol]
            LV = scp.sparse.csc_array((LVdat,(LVrow,LVcol)),shape=(len(full_coef),len(full_coef)))

      if checkHc > 30:
         break

      checkHc += 1

   # Done except when we have a drop -> still need to pad H and L
   if drop is not None:
      #print(coef)
      split_coef = np.split(full_coef,full_coef_split_idx)

      # Now H, L
      Hdat,Hrow,Hcol = translate_sparse(H)
      Ldat,Lrow,Lcol = translate_sparse(L.tocsc()) # L@L.T = H_pen
      
      Hrow = keep[Hrow]
      Hcol = keep[Hcol]
      Lrow = keep[Lrow]
      Lcol = keep[Lcol]

      H = scp.sparse.csc_array((Hdat,(Hrow,Hcol)),shape=(len(full_coef),len(full_coef)))
      L = scp.sparse.csc_array((Ldat,(Lrow,Lcol)),shape=(len(full_coef),len(full_coef)))
      
   return full_coef,split_coef,mus,etas,H,L,LV,c_llk,c_pen_llk,eps,keep,drop

def correct_lambda_step_gamlss(family:GAMLSSFamily,mus:list[np.ndarray],y:np.ndarray,Xs:list[scp.sparse.csc_array],S_norm:scp.sparse.csc_array,
                               n_coef:int,form_n_coef:list[int],form_up_coef:list[int],coef:np.ndarray,coef_split_idx:list[int],
                               gamlss_pen:list[LambdaTerm],lam_delta:np.ndarray,extend_by:dict,was_extended:list[bool],c_llk:float,
                               fit_info:Fit_info,outer:int,max_inner:int,min_inner:int,conv_tol:float,method:str,
                               piv_tol:float,keep_drop:list[list[int],list[int]]|None,extend_lambda:bool,
                               extension_method_lam:str,control_lambda:int,repara:bool,n_c:int) -> tuple[np.ndarray,list[np.ndarray],list[np.ndarray],list[np.ndarray],scp.sparse.csc_array,scp.sparse.csc_array,scp.sparse.csc_array,float,float,float,list[int],list[int],scp.sparse.csc_array,list[LambdaTerm],float,list[float],np.ndarray]:
   """Updates and performs step-length control for the vector of lambda parameters of a GAMMLSS model. Essentially completes the steps described in section 3.3 of the paper by Krause et al. (submitted).

   Based on steps outlined by Wood, Pya, & Säfken (2016).

   References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
     - Krause et al. (submitted). The Mixed-Sparse-Smooth-Model Toolbox (MSSM): Efficient Estimation and Selection of Large Multi-Level Statistical Models. https://doi.org/10.48550/arXiv.2506.13132

   :param family: Family of model
   :type family: GAMLSSFamily
   :param mus: List of estimated means
   :type mus: list[np.ndarray]
   :param y: Vector of observations
   :type y: np.ndarray
   :param Xs: List of model matrices - one per parameter of response distribution
   :type Xs: [scp.sparse.csc_array]
   :param S_norm: Scaled version of the penalty matrix (i.e., unweighted total penalty divided by it's norm).
   :type S_norm: scp.sparse.csc_array
   :param n_coef: Number of coefficients
   :type n_coef: int
   :param form_n_coef: List of number of coefficients per formula
   :type form_n_coef: list[int]
   :param form_up_coef: List of un-penalized number of coefficients per formula
   :type form_up_coef: list[int]
   :param coef: Coefficient estimate
   :type coef: np.ndarray
   :param coef_split_idx: The indices at which to split the overall coefficient vector into separate lists - one per parameter.
   :type coef_split_idx: list[int]
   :param gamlss_pen: List of penalties
   :type gamlss_pen: list[LambdaTerm]
   :param lam_delta: Update to vector of lambda parameters
   :type lam_delta: np.ndarray
   :param extend_by: Extension info dictionary
   :type extend_by: dict
   :param was_extended: List holding indication per lambda parameter whether it was extended or not
   :type was_extended: list[bool]
   :param c_llk: Current llk
   :type c_llk: float
   :param fit_info: A :class:`Fit_info` object
   :type fit_info: Fit_info
   :param outer: Index of outer iteration
   :type outer: int
   :param max_inner: Maximum number of inner iterations
   :type max_inner: int
   :param min_inner: Minimum number of inner iterations
   :type min_inner: int
   :param conv_tol: Convergence tolerance
   :type conv_tol: float
   :param method: Method to use to estimate coefficients
   :type method: str
   :param piv_tol: Deprecated
   :type piv_tol: float
   :param keep_drop: Set of previously dropped coeeficients or None
   :type keep_drop: list[list[int],list[int]] | None
   :param extend_lambda: Whether lambda proposals should be accelerated or not. Can lower the number of new smoothing penalty proposals necessary
   :type extend_lambda: bool
   :param extension_method_lam: Which method to use to extend lambda proposals.
   :type extension_method_lam: str
   :param control_lambda: Whether lambda proposals should be checked (and if necessary decreased) for whether or not they (approxiately) increase the Laplace approximate restricted maximum likelihood of the model. Setting this to 0 disables control. Setting it to 1 means the step will never be smaller than the original EFS update but extensions will be removed in case the objective was exceeded. Setting it to 2 means that steps will be halved if it fails to increase the approximate REML.
   :type control_lambda: int
   :param repara: Whether to apply a stabilizing re-parameterization to the model
   :type repara: bool
   :param n_c: Number of cores to use
   :type n_c: int
   :return: coef estimate under corrected lambda, split version of next coef estimate, next mus, next etas, the negative hessian of the log-likelihood, cholesky of negative hessian of the penalized log-likelihood, inverse of the former, new llk, new penalized llk, the multiple (float) added to the diagonal of the negative penalized hessian to make it invertible, an optional list of the coefficients to keep, an optional list of the estimated coefficients to drop, the new total penalty matrix, the new list of penalties, total edf, term-wise edfs, the update to the lambda vector
   :rtype: tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray], scp.sparse.csc_array, scp.sparse.csc_array, scp.sparse.csc_array, float, float, float, list[int], list[int], scp.sparse.csc_array, list[LambdaTerm], float, list[float], np.ndarray]
   """
   
   # Fitting routine with step size control for smoothing parameters of GAMMLSS models. Not obvious - because we have only approximate REMl
   # and approximate derivative, because we drop the last term involving the derivative of the negative penalized
   # Hessian with respect to the smoothing parameters (see section 4 in Wood & Fasiolo, 2017). However, Krause et al. (in preparation) motivate
   # step-legnth control based on these approximate information and step-length control is thus performed here if ``control_lambda > 0``.
   # Setting ``control_lambda=1`` means any acceleration of the lambda step is undone if we overshoot the approximate criterion. Setting ``control_lambda=2``
   # means we perform step-length reduction as if the approximate information were accurate.

   lam_accepted = False
   
   #prev_drop = fit_info.dropped
   #reductions = 0
   while lam_accepted == False:

      # Build new penalties
      S_emb,S_pinv,_,FS_use_rank = compute_S_emb_pinv_det(n_coef,gamlss_pen,"svd")

      # Re-parameterize
      if repara:
         coef_rp,Xs_rp,gamlss_pen_rp,S_emb,S_norm,S_root_rp,S_pinv,Q_emb,Qs = reparam_model(form_n_coef, form_up_coef, coef, coef_split_idx, Xs,
                                                                                           gamlss_pen, form_inverse=True,
                                                                                           form_root=False, form_balanced=True, n_c=n_c)
      else:
         coef_rp = coef
         Xs_rp = Xs
         gamlss_pen_rp = gamlss_pen
   
      # First re-compute coef
      next_coef,split_coef,next_mus,next_etas,H,L,LV,next_llk,next_pen_llk,eps,keep,drop  = update_coef_gammlss(family,mus,y,Xs_rp,coef_rp,
                                                                                                                coef_split_idx,S_emb,S_norm,
                                                                                                                S_pinv,FS_use_rank,gamlss_pen_rp,
                                                                                                                c_llk,outer,max_inner,
                                                                                                                min_inner,conv_tol,
                                                                                                                method,piv_tol,keep_drop)
      

      # Now re-compute lgdetDs, ldetHS, and bsbs
      lgdetDs = []
      bsbs = []
      for lti,lTerm in enumerate(gamlss_pen_rp):

            lt_rank = None
            if FS_use_rank[lti]:
               lt_rank = lTerm.rank

            lgdetD,bsb = compute_lgdetD_bsb(lt_rank,lTerm.lam,S_pinv,lTerm.S_J_emb,next_coef)
            lgdetDs.append(lgdetD)
            bsbs.append(bsb)

      total_edf,term_edfs, ldetHSs = calculate_edf(None,None,LV,gamlss_pen_rp,lgdetDs,n_coef,n_c,None,None)
      #print([l1-l2 for l1,l2 in zip(lgdetDs,ldetHSs)])
      #print(total_edf)
      fit_info.lambda_updates += 1


      # Can exit loop here, no extension and no control
      if outer == 0  or (control_lambda < 1) or (extend_lambda == False and control_lambda < 2):
         lam_accepted = True
         continue

      # Compute approximate!!! gradient of REML with respect to lambda
      # to check if step size needs to be reduced (part of step 6 in Wood, 2017).
      lam_grad = [grad_lambda(lgdetDs[lti],ldetHSs[lti],bsbs[lti],1) for lti in range(len(gamlss_pen_rp))]
      lam_grad = np.array(lam_grad).reshape(-1,1) 
      check = lam_grad.T @ lam_delta

      # Now undo the acceleration if overall direction is **very** off - don't just check against 0 because
      # our criterion is approximate, so we can be more lenient (see Wood et al., 2017).
      lam_changes = 0 
      if check[0] < 1e-7*-abs(next_pen_llk):
         for lti,lTerm in enumerate(gamlss_pen): # Make sure to work with original penalty object here, so that step-length control is reflected in lambda values

            # For extended terms undo extension
            if extend_lambda and was_extended[lti]:
               lam, dLam = undo_extension_lambda_step(lti,lTerm.lam,lam_delta[lti],extend_by,was_extended, extension_method_lam, None)

               lTerm.lam = lam
               lam_delta[lti] = dLam
               lam_changes += 1
            
            elif control_lambda == 2:
               # Continue to half step - this is not necessarily because we overshoot the REML
               # (because we only have approximate gradient), but can help preventing that models
               # oscillate around estimates.
               lam_delta[lti] = lam_delta[lti]/2
               lTerm.lam -= lam_delta[lti]
               lam_changes += 1
         
         # If we did not change anything, simply accept
         if lam_changes == 0 or np.linalg.norm(lam_delta) < 1e-7:
            lam_accepted = True
      
      # If we pass the check we can simply accept.
      else:
         lam_accepted = True
            
   # At this point we have accepted the step - so we can propose a new one
   if drop is not None:
      fit_info.dropped = drop
   else:
      fit_info.dropped = None

   step_norm = np.linalg.norm(lam_delta)
   lam_delta = []
   for lti,lTerm in enumerate(gamlss_pen):

      lgdetD = lgdetDs[lti]
      ldetHS = ldetHSs[lti]
      bsb = bsbs[lti]
      
      #print(lgdetD-ldetHS)
      dLam = step_fellner_schall_sparse(lgdetD,ldetHS,bsb[0,0],lTerm.lam,1)
      #print("Theorem 1:",lgdetD-ldetHS,bsb,lTerm.lam)

      # For poorly scaled/ill-identifiable problems we cannot rely on the theorems by Wood
      # & Fasiolo (2017) - so the condition below will be met, in which case we just want to
      # take very small steps until it hopefully gets more stable (due to term dropping or better lambda value).
      if lgdetD - ldetHS < 0:
         dLam = np.sign(dLam) * min(abs(lTerm.lam)*0.001,abs(dLam))
         dLam = 0 if lTerm.lam+dLam < 0 else dLam

      if extend_lambda:
         dLam,extend_by,was_extended = extend_lambda_step(lti,lTerm.lam,dLam,extend_by,was_extended,extension_method_lam)
         if (((outer > 100 and step_norm < 5e-3) or outer > 250) or (max(eps,fit_info.eps) > 0 and outer > 50)) and was_extended[lti]:
            if max(eps,fit_info.eps) > 0 and outer > 50:
               damp = 1*np.power(0.9,outer-50)
            elif step_norm < 5e-3 and outer <= 250 and max(eps,fit_info.eps) <= 0:
               damp = 1*np.power(0.9,outer-100)
            elif outer > 250 and max(eps,fit_info.eps) <= 0:
               damp = 1*np.power(0.9,outer-250)

            extend_by["acc"][lti]*=damp

      # Dampen steps taken if Hessian continues to be modified to prevent erratic jumps or if progress is slow
      if (outer > 100 and step_norm < 5e-3) or outer > 250 or (max(eps,fit_info.eps) > 0 and outer > 50):
         if max(eps,fit_info.eps) > 0 and outer > 50:
            damp = 1*np.power(0.9,outer-50)
         elif step_norm < 5e-3 and outer <= 250 and max(eps,fit_info.eps) <= 0:
            damp = 1*np.power(0.9,outer-100)
         elif outer > 250 and max(eps,fit_info.eps) <= 0:
            damp = 1*np.power(0.9,outer-250)
         #print(damp,1*np.power(0.9,outer-50))
         dLam = damp*dLam

      lam_delta.append(dLam)
   
   fit_info.eps = eps
   
   # And un-do the latest re-parameterization
   if repara:
         
      # Transform S_emb (which is currently S_emb_rp)
      S_emb = Q_emb @ S_emb @ Q_emb.T

      # Transform coef
      for qi,Q in enumerate(Qs):
         split_coef[qi] = Q@split_coef[qi]
      next_coef = np.concatenate(split_coef).reshape(-1,1)

      # Transform H, L, LV
      H = Q_emb @ H @ Q_emb.T
      L = Q_emb @ L
      LV = LV @ Q_emb.T

   return next_coef,split_coef,next_mus,next_etas,H,L,LV,next_llk,next_pen_llk,eps,keep,drop,S_emb,gamlss_pen,total_edf,term_edfs,lam_delta

    
def solve_gammlss_sparse(family:GAMLSSFamily,y:np.ndarray,Xs:list[scp.sparse.csc_array],form_n_coef:list[int],form_up_coef:list[int],coef:np.ndarray,
                         coef_split_idx:list[int],gamlss_pen:list[LambdaTerm],
                         max_outer:int=50,max_inner:int=30,min_inner:int=1,conv_tol:float=1e-7,
                         extend_lambda:bool=True,extension_method_lam:str = "nesterov2",
                         control_lambda:int=1,method:str="Chol",check_cond:int=1,piv_tol:float=0.175,
                         repara:bool=True,should_keep_drop:bool=True,prefit_grad:bool=False,progress_bar:bool=True,n_c:int=10) -> tuple[np.ndarray,list[np.ndarray],list[np.ndarray],np.ndarray,scp.sparse.csc_array,scp.sparse.csc_array,float,list[float],float,list[LambdaTerm],Fit_info]:
   """ Fits a GAMLSS model - essentially completes the steps discussed in section 3.3 of the paper by Krause et al. (submitted).

   Based on steps outlined by Wood, Pya, & Säfken (2016)

   References:
      - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
      - Krause et al. (submitted). The Mixed-Sparse-Smooth-Model Toolbox (MSSM): Efficient Estimation and Selection of Large Multi-Level Statistical Models. https://doi.org/10.48550/arXiv.2506.13132


   :param family: Model family
   :type family: GAMLSSFamily
   :param y: Vector of observations
   :type y: np.ndarray
   :param Xs: List of model matrices - one per parameter of response distribution
   :type Xs: [scp.sparse.csc_array]
   :param form_n_coef: List of number of coefficients per formula
   :type form_n_coef: list[int]
   :param form_up_coef: List of un-penalized number of coefficients per formula
   :type form_up_coef: list[int]
   :param coef: Coefficient estimate
   :type coef: np.ndarray
   :param coef_split_idx: The indices at which to split the overall coefficient vector into separate lists - one per parameter.
   :type coef_split_idx: list[int]
   :param gamlss_pen: List of penalties
   :type gamlss_pen: list[LambdaTerm]
   :param max_outer: Maximum number of outer iterations, defaults to 50
   :type max_outer: int, optional
   :param max_inner: Maximum number of inner iterations, defaults to 30
   :type max_inner: int, optional
   :param min_inner: Minimum number of inner iterations, defaults to 1
   :type min_inner: int, optional
   :param conv_tol: Convergence tolerance, defaults to 1e-7
   :type conv_tol: float, optional
   :param extend_lambda: Whether lambda proposals should be accelerated or not. Can lower the number of new smoothing penalty proposals necessary, defaults to True
   :type extend_lambda: bool, optional
   :param extension_method_lam: Which method to use to extend lambda proposals, defaults to "nesterov2"
   :type extension_method_lam: str, optional
   :param control_lambda: Whether lambda proposals should be checked (and if necessary decreased) for whether or not they (approxiately) increase the Laplace approximate restricted maximum likelihood of the model. Setting this to 0 disables control. Setting it to 1 means the step will never be smaller than the original EFS update but extensions will be removed in case the objective was exceeded. Setting it to 2 means that steps will be halved if it fails to increase the approximate REML., defaults to 1
   :type control_lambda: int, optional
   :param method: Method to use to estimate coefficients, defaults to "Chol"
   :type method: str, optional
   :param check_cond: Whether to obtain an estimate of the condition number for the linear system that is solved. When ``check_cond=0``, no check will be performed. When ``check_cond=1``, an estimate of the condition number for the final system (at convergence) will be computed and warnings will be issued based on the outcome (see :func:`mssm.src.python.gamm_solvers.est_condition`)., defaults to 1
   :type check_cond: int, optional
   :param piv_tol: Deprecated, defaults to 0.175
   :type piv_tol: float, optional
   :param repara: Whether to apply a stabilizing re-parameterization to the model, defaults to True
   :type repara: bool, optional
   :param should_keep_drop: If set to True, any coefficients that are dropped during fitting - are permanently excluded from all subsequent iterations, defaults to True
   :type should_keep_drop: bool, optional
   :param prefit_grad: Whether to rely on Gradient Descent to improve the initial starting estimate for coefficients., defaults to False
   :type prefit_grad: bool, optional
   :param progress_bar: Whether progress should be displayed, defaults to True
   :type progress_bar: bool, optional
   :param n_c: Number of cores to use, defaults to 10
   :type n_c: int, optional
   :return: coef estimate, etas, mus, working residuals, the negative hessian of the log-likelihood, inverse of cholesky of negative hessian of the penalized log-likelihood, total edf, term-wise edfs, total penalty, final list of penalties, a :class:`Fit_info` object
   :rtype: tuple[np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray, scp.sparse.csc_array, scp.sparse.csc_array, float, list[float], float, list[LambdaTerm], Fit_info]
   """
   # total number of coefficients
   n_coef = np.sum(form_n_coef)

   extend_by = initialize_extension(extension_method_lam,gamlss_pen)
   was_extended = [False for _ in enumerate(gamlss_pen)]

   split_coef = np.split(coef,coef_split_idx)

   # Initialize etas and mus
   etas = [Xs[i]@split_coef[i] for i in range(family.n_par)]
   mus = [family.links[i].fi(etas[i]) for i in range(family.n_par)]

   # Find and exclude invalid indices before evaluating llk
   inval = np.isnan(mus[0])

   if len(coef_split_idx) != 0:
      for mi in range(1,len(mus)):
         inval = inval |  np.isnan(mus[mi])

   if np.sum(inval) > 0:
      raise ValueError("The initial coefficients result in invalid value for at least one predictor. Provide a different set via the family's ``init_coef`` function.")

   # Build current penalties
   S_emb,_,_,_ = compute_S_emb_pinv_det(n_coef,gamlss_pen,"svd")
   c_llk = family.llk(y,*mus)
   c_pen_llk = c_llk - 0.5*coef.T@S_emb@coef

   # Build normalized penalty for rank checks (e.g., Wood, Pya & Saefken, 2016)
   keep_drop = None

   S_norm = copy.deepcopy(gamlss_pen[0].S_J_emb)/scp.sparse.linalg.norm(gamlss_pen[0].S_J_emb,ord=None)
   for peni in range(1,len(gamlss_pen)):
      S_norm += gamlss_pen[peni].S_J_emb/scp.sparse.linalg.norm(gamlss_pen[peni].S_J_emb,ord=None)
   
   S_norm /= scp.sparse.linalg.norm(S_norm,ord=None)

    # Try improving start estimate via Gradient only
   if prefit_grad:
      # Re-parameterize
      if repara:
         coef_rp,Xs_rp,_,S_emb_rp,S_norm_rp,_,_,_,Qs = reparam_model(form_n_coef, form_up_coef, coef, coef_split_idx, Xs,
                                                                     gamlss_pen, form_inverse=False,
                                                                     form_root=False, form_balanced=False, n_c=n_c)
      else:
         coef_rp = coef
         Xs_rp = Xs
         S_emb_rp = S_emb
         S_norm_rp = S_norm

      coef,split_coef,mus,etas,_,_,_,c_llk,c_pen_llk,_,_,_ = update_coef_gammlss(family,mus,y,Xs_rp,coef_rp,
                                                                        coef_split_idx,S_emb_rp,
                                                                        S_norm_rp,None,None,None,
                                                                        c_llk,0,max_inner,
                                                                        min_inner,conv_tol,
                                                                        "Grad",piv_tol,None)
      
      if repara:
         split_coef = np.split(coef,coef_split_idx)

         # Transform coef
         for qi,Q in enumerate(Qs):
            split_coef[qi] = Q@split_coef[qi]
         coef = np.concatenate(split_coef).reshape(-1,1)

   iterator = range(max_outer)
   if progress_bar:
      iterator = tqdm(iterator,desc="Fitting",leave=True)
   
   fit_info = Fit_info()
   fit_info.eps = 0
   lam_delta = []
   prev_llk_hist = []
   for outer in iterator:
      
      # 1) Update coef for given lambda
      # 2) Check lambda -> repeat 1 if necessary
      # 3) Propose new lambda
      coef,split_coef,mus,etas,H,L,LV,c_llk,\
      c_pen_llk,eps,keep,drop,S_emb,gamlss_pen,\
      total_edf,term_edfs,lam_delta = correct_lambda_step_gamlss(family,mus,y,Xs,S_norm,n_coef,form_n_coef,form_up_coef,coef,
                                                                  coef_split_idx,gamlss_pen,lam_delta,
                                                                  extend_by,was_extended,c_llk,fit_info,outer,
                                                                  max_inner,min_inner,conv_tol,method,
                                                                  piv_tol,keep_drop,extend_lambda,
                                                                  extension_method_lam,control_lambda,repara,n_c)
      
      if drop is not None:
         if should_keep_drop:
            keep_drop = [keep,drop]

      fit_info.iter += 1

      # Check convergence
      if outer > 0:
         if progress_bar:
               iterator.set_description_str(desc="Fitting - Conv.: " + "{:.2e}".format((np.abs(prev_pen_llk - c_pen_llk) - conv_tol*np.abs(c_pen_llk))[0,0]), refresh=True)
         #print(drop,lam_delta,np.max(np.abs([prev_lam[lti] - gamlss_pen[lti].lam for lti in range(len(gamlss_pen))])))
         # generally converge if change in pen llk is small - but if we have dropped something also permit small change in lambda after correction or - if we have been at it for some time - a very small proposal
         if np.any([np.abs(prev_llk_hist[hi] - c_pen_llk[0,0])  < conv_tol*np.abs(c_pen_llk) for hi in range(len(prev_llk_hist))]) or\
            ((drop is not None) and (np.max(np.abs([prev_lam[lti] - gamlss_pen[lti].lam for lti in range(len(gamlss_pen))])) < 0.01)) or\
            (outer > 100 and (np.max(np.abs(lam_delta)) < 0.01)):
               if progress_bar:
                  iterator.set_description_str(desc="Converged!", refresh=True)
                  iterator.close()
               fit_info.code = 0
               break
            
      # We need the penalized likelihood of the model at this point for convergence control (step 2 in Wood, 2017 based on step 4 in Wood, Goude, & Shaw, 2016)
      prev_pen_llk = c_pen_llk
      prev_llk_hist.append(prev_pen_llk[0,0])
      if len(prev_llk_hist) > 5:
         del prev_llk_hist[0]

      prev_lam = copy.deepcopy([lterm.lam for lterm in gamlss_pen])
      #prev_coef = copy.deepcopy(coef)
      #print("lambda before step:",[lterm.lam for lterm in gamlss_pen])
      # And ultimately apply new proposed lambda step
      for lti,lTerm in enumerate(gamlss_pen):
         lTerm.lam += lam_delta[lti]

      # At this point we have:
      # 1) Estimated coef given lambda
      # 2) Performed checks on lambda and re-estimated coefficients if necessary
      # 3) Proposed lambda updates
      # 4) Checked for convergence
      # 5) Applied the new update lambdas so that we can go to the next iteration
      #print("lambda after step:",[lterm.lam for lterm in gamlss_pen])
    
   if check_cond == 1:
      K2,_,_,Kcode = est_condition(L,LV,verbose=False)

      fit_info.K2 = K2

      if fit_info.code == 0: # Convergence was reached but Knumber might still suggest instable system.
         fit_info.code = Kcode

      if Kcode > 0:
         warnings.warn(f"Condition number ({K2}) of matrix A, where A.T@A=H and H is the Hessian of the negative penalized likelihood, is larger than 1/sqrt(u), where u is half the machine precision. Call ``model.fit()`` with ``method='QR/Chol'``, but note that even then estimates are likely to be inaccurate.")
    
   # "Residuals"
   wres = y - Xs[0]@split_coef[0]

   # Total penalty
   penalty = coef.T@S_emb@coef

   # Calculate actual term-specific edf
   term_edfs = calculate_term_edf(gamlss_pen,term_edfs)
    
   return coef,etas,mus,wres,H,LV,total_edf,term_edfs,penalty[0,0],gamlss_pen,fit_info

################################################ General Smooth model code ################################################

def correct_coef_step_gen_smooth(family:GSMMFamily,ys:list[np.ndarray],Xs:list[scp.sparse.csc_array],coef:np.ndarray,next_coef:np.ndarray,coef_split_idx:list[int],c_llk:float,S_emb:scp.sparse.csc_array,a:float) -> tuple[np.ndarray,float,float,float]:
   """Apply step size correction to Newton update for general smooth models, as discussed by Wood, Pya, & Säfken (2016).

    References:
      - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.

   :param family: Model family
   :type family: GSMMFamily
   :param ys: List of vectors of observations
   :type ys: list[np.ndarray]
   :param Xs: List of model matrices
   :type Xs: list[scp.sparse.csc_array]
   :param coef: Coefficient estimate
   :type coef: np.ndarray
   :param next_coef: Proposed next coefficient estimate
   :type next_coef: np.ndarray
   :param coef_split_idx: The indices at which to split the overall coefficient vector into separate lists - one per parameter.
   :type coef_split_idx: list[int]
   :param c_llk: Current log likelihood
   :type c_llk: float
   :param S_emb: Total penalty matrix
   :type S_emb: scp.sparse.csc_array
   :param a: Step length for gradient descent update
   :type a: float
   :return: A tuple containing the corrected coefficient estimate ``next_coef``,next llk, next penalized llk, updated step length for next gradient update
   :rtype: tuple[np.ndarray,float,float,float]
   """
    
   # Step size control for newton step.
   next_llk = family.llk(next_coef,coef_split_idx,ys,Xs)
   
   # Evaluate improvement of penalized llk under new and old coef - but in both
   # cases for current lambda (see Wood, Li, Shaddick, & Augustin; 2017)
   next_pen_llk = next_llk - 0.5*next_coef.T@S_emb@next_coef
   prev_llk_cur_pen = c_llk - 0.5*coef.T@S_emb@coef
    
   for n_checks in range(32):
        
      if next_pen_llk >= prev_llk_cur_pen and (np.isinf(next_pen_llk[0,0]) == False and np.isnan(next_pen_llk[0,0]) == False):
         break
      
      if n_checks > 30:
         next_coef = coef
      
      # Half it if we do not observe an increase in penalized likelihood (WPS, 2016)
      next_coef = (coef + next_coef)/2
      
      # Update pen_llk
      next_llk = family.llk(next_coef,coef_split_idx,ys,Xs)
      next_pen_llk = next_llk - 0.5*next_coef.T@S_emb@next_coef
        
   # Update step-size for gradient
   if n_checks > 0 and a > 1e-9:
      a /= 2
   elif n_checks == 0 and a < 1:
      a *= 2
    
   return next_coef,next_llk,next_pen_llk,a

def back_track_alpha(coef:np.ndarray,step:np.ndarray,llk_fun:Callable,grad_fun:Callable,*llk_args,alpha_max:float=1,c1:float=1e-4,max_iter:int=100) -> float | None:
   """Simple step-size backtracking function that enforces Armijo condition (Nocedal & Wright, 2004)

   References:
      - Nocedal & Wright (2006). Numerical Optimization. Springer New York.

   :param coef: coefficient estimate
   :type coef: np.ndarray
   :param step: step to take to update coefficients
   :type step: np.ndarray
   :param llk_fun: llk function
   :type llk_fun: Callable
   :param grad_fun: function to evaluate gradient of llk
   :type grad_fun: Callable
   :param alpha_max: Parameter by Nocedal & Wright, defaults to 1
   :type alpha_max: float, optional
   :param c1: 2nd Parameter by Nocedal & Wright, defaults to 1e-4
   :type c1: float, optional
   :param max_iter: Number of maximum iterations, defaults to 100
   :type max_iter: int, optional
   :return: The step-length meeting the Armijo condition or None in case none such was found
   :rtype: float | None
   """
   # 
   c_llk = llk_fun(coef.flatten(),*llk_args)
   c_grad = grad_fun(coef.flatten(),*llk_args).reshape(-1,1)

   c_alpha = alpha_max

   for _ in range(max_iter):
      # Test Armijo condition
      with warnings.catch_warnings():
         warnings.simplefilter("ignore")
         n_llk = llk_fun((coef + c_alpha*step).flatten(),*llk_args)

      armijo = n_llk <= c_llk + c_alpha*c1*c_grad.T@step

      if armijo and (np.isnan(n_llk[0,0]) == False and np.isinf(n_llk[0,0]) == False):
         return c_alpha
      
      c_alpha /= 2
   
   return None

def check_drop_valid_gensmooth(ys:list[np.ndarray],coef:np.ndarray,Xs:list[scp.sparse.csc_array],S_emb:scp.sparse.csc_array,keep:list[int],family:GSMMFamily) -> tuple[bool,float|None]:
   """Checks whether an identified set of coefficients to be dropped from the model results in a valid log-likelihood.

   :param ys: List holding vectors of observations
   :type ys: list[np.ndarray]
   :param coef: Vector of coefficients
   :type coef: np.ndarray
   :param Xs: List of model matrices - one per parameter
   :type Xs: list[scp.sparse.csc_array]
   :param S_emb: Total Penalty matrix
   :type S_emb: scp.sparse.csc_array
   :param keep: List of coefficients to retain
   :type keep: list[int]
   :param family: Model family
   :type family: GSMMFamily
   :return: tuple holding bool indicating if likelihood is valid and penalized log-likelihood under dropped set.
   :rtype: tuple[bool,float|None]
   """
   # Drop from coef
   dropcoef = coef[keep]

   # ... from Xs...
   rXs,rcoef_split_idx = drop_terms_X(Xs,keep)

   # ... and from S_emb
   rS_emb = S_emb[keep,:]
   rS_emb = rS_emb[:,keep]

   # Re-compute llk
   c_llk = family.llk(dropcoef,rcoef_split_idx,ys,rXs)
   c_pen_llk = c_llk - 0.5*dropcoef.T@rS_emb@dropcoef

   if (np.isinf(c_pen_llk[0,0]) or np.isnan(c_pen_llk[0,0])):
      return False,None
   
   return True,c_pen_llk

def restart_coef(coef:np.ndarray,c_llk:float,c_pen_llk:float,n_coef:np.ndarray,coef_split_idx:list[int],ys:list[np.ndarray],Xs:list[scp.sparse.csc_array],S_emb:scp.sparse.csc_array,family:GSMMFamily,outer:int,restart_counter:int) -> tuple[np.ndarray, float, float]:
   """Shrink coef towards random vector to restart algorithm if it get's stuck.

   :param coef: Coefficient estimate
   :type coef: np.ndarray
   :param c_llk: Current llk
   :type c_llk: float
   :param c_pen_llk: Current penalized llk
   :type c_pen_llk: float
   :param n_coef: Number of coefficients
   :type n_coef: np.ndarray
   :param coef_split_idx: List with indices to split coef - one per parameter of response distribution
   :type coef_split_idx: list[int]
   :param ys: List of observation vectors
   :type ys: list[np.ndarray]
   :param Xs: List of model matrices
   :type Xs: list[scp.sparse.csc_array]
   :param S_emb: Total penalty matrix
   :type S_emb: scp.sparse.csc_array
   :param family: Model family
   :type family: GSMMFamily
   :param outer: Outer iteration index
   :type outer: int
   :param restart_counter: Number of restarts already handled previously
   :type restart_counter: int
   :return: Updates for coef, c_llk, c_pen_llk
   :rtype: tuple[np.ndarray, float, float]
   """

   res_checks = 0
   res_scale = 0.5 if outer <= 10 else 1/(restart_counter+2)
   #print("resetting conv.",res_scale)

   while res_checks < 30:
      res_coef = ((1-res_scale)*coef + res_scale*scp.stats.norm.rvs(size=n_coef,random_state=outer+res_checks).reshape(-1,1))

      # Re-compute llk
      with warnings.catch_warnings():
         warnings.simplefilter("ignore")
         res_llk = family.llk(res_coef,coef_split_idx,ys,Xs)
         res_pen_llk = res_llk - 0.5*res_coef.T@S_emb@res_coef

      if (np.isinf(res_pen_llk[0,0]) or np.isnan(res_pen_llk[0,0])):
         res_checks += 1
         continue
      
      coef = res_coef
      c_llk = res_llk
      c_pen_llk = res_pen_llk
      break
   
   return coef, c_llk, c_pen_llk

def test_SR1(sk:np.ndarray,yk:np.ndarray,rho:np.ndarray,sks:np.ndarray,yks:np.ndarray,rhos:np.ndarray) -> bool:
   """Test whether SR1 update is well-defined for both V and H.

   Relies on steps discussed by Byrd, Nocdeal & Schnabel (1992).

   References:
    - Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994). Representations of quasi-Newton matrices and their use in limited memory methods. Mathematical Programming, 63(1), 129–156. https://doi.org/10.1007/BF01582063

   :param sk: New update vector sk
   :type sk: np.ndarray
   :param yk: New update vector yk
   :type yk: np.ndarray
   :param rho: New rho
   :type rho: np.ndarray
   :param sks: Previous update vectors sk
   :type sks: np.ndarray
   :param yks: Previous update vector sks
   :type yks: np.ndarray
   :param rhos: Previous rhos
   :type rhos: np.ndarray
   :return: Check whether SR1 update is well-defined for both V and H.
   :rtype: bool
   """
   # Conditionally accept sk, yk, and rho
   if len(sks) == 0:
      sks = sk.T
      yks = yk.T
      rhos = np.array([rho[0,0]])
   else:
      sks = np.append(sks,sk.T,axis=0)
      yks = np.append(yks,yk.T,axis=0)
      rhos = np.append(rhos,rho[0,0])

   # Compute new omega to better scale hessian (again see Nocedal & Wright, 2004)
   if sks.shape[0] > 0 and yk.T@sk > 0:
      omega = np.dot(yks[-1],yks[-1])/np.dot(yks[-1],sks[-1])
      if omega < 0 or (sks.shape[0] == 1):
         # Cannot use scaling for first update vector since inverse of hessian will result in
         # zero t2 term due to cancellation
         omega = 1
   else:
      omega = 1

   # Update H0/V0 (rather H0(k)/V0(k))
   H0 = scp.sparse.identity(sk.shape[0],format='csc')*omega
   V0 = scp.sparse.identity(sk.shape[0],format='csc')*(1/omega)
   
   # Now try both SR1 updates
   Fail = False
   try:
      _,_,_ = computeHSR1(sks,yks,rhos,H0,omega=omega,make_psd=False,explicit=False)
      _,_,_ = computeVSR1(sks,yks,rhos,V0,omega=1/omega,make_psd=False,explicit=False)
      #t12,t22,t32 = computeHSR1(yks,sks,rhos,V0,omega=1/omega,make_psd=False,explicit=False)
      #print(np.max(np.abs((t11@t21@t31)-(t12@t22@t32))))
   except:
      Fail = True

   # Undo conditional assignment
   sks = np.delete(sks,0,axis=0)
   yks = np.delete(yks,0,axis=0)
   rhos = np.delete(rhos,0)
   return Fail

def handle_drop_gsmm(family:GSMMFamily, ys:list[np.ndarray], coef:np.ndarray, keep:list[int], Xs:list[scp.sparse.csc_array], S_emb:scp.sparse.csc_array) -> tuple[np.ndarray, list[int], list[scp.sparse.csc_array], scp.sparse.csc_array, float, float]:
   """Drop coefficients and make sure this is reflected in the model matrices, total penalty, llk, and penalized llk.

   :param family: Model family
   :type family: GSMMFamily
   :param ys: List with vector of observations
   :type ys: list[np.ndarray]
   :param coef: Vector of coefficients
   :type coef: np.ndarray
   :param keep: List of parameter indices to keep.
   :type keep: list[int]
   :param Xs: List of model matrices
   :type Xs: list[scp.sparse.csc_array]
   :param S_emb: Total penalty matrix.
   :type S_emb: scp.sparse.csc_array
   :return: A tuple holding: reduced coef vector, a new list of indices determining where to split the reduced coef vector, list with reduced model matrices, reduced total penalty matrix, updated llk, and penalized llk
   :rtype: tuple[np.ndarray, list[int], list[scp.sparse.csc_array], scp.sparse.csc_array, float, float]
   """
   # Drop from coef
   coef = coef[keep]

   # ... from Xs...
   rXs,rcoef_split_idx = drop_terms_X(Xs,keep)
   #print(rXs)

   # ... and from S_emb
   rS_emb = S_emb[keep,:]
   rS_emb = rS_emb[:,keep]

   # Re-compute llk
   c_llk = family.llk(coef,rcoef_split_idx,ys,rXs)
   c_pen_llk = c_llk - 0.5*coef.T@rS_emb@coef

   return coef, rcoef_split_idx, rXs, rS_emb, c_llk, c_pen_llk
    
def update_coef_gen_smooth(family:GSMMFamily,ys:list[np.ndarray],Xs:list[scp.sparse.csc_array],coef:np.ndarray,coef_split_idx:list[int],
                           S_emb:scp.sparse.csc_array,S_norm:scp.sparse.csc_array,S_pinv:scp.sparse.csc_array,FS_use_rank:list[bool],smooth_pen:list[LambdaTerm],
                           c_llk:float,outer:int,max_inner:int,min_inner:int,conv_tol:float,method:str,piv_tol:float,keep_drop:list[list[int],list[int]]|None,
                           opt_raw:scp.sparse.linalg.LinearOperator|None) -> tuple[np.ndarray,scp.sparse.csc_array|None,scp.sparse.csc_array|None,scp.sparse.csc_array|scp.sparse.linalg.LinearOperator,float,float,float,list[int]|None,list[int]|None]:
   """Repeatedly perform Newton/Gradient/L-qEFS update with step length control to the coefficient vector - essentially completes the steps discussed in sections 3.3 and 4 of the paper by Krause et al. (submitted).

   Based on steps outlined by Wood, Pya, & Säfken (2016).

   References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
     - Krause et al. (submitted). The Mixed-Sparse-Smooth-Model Toolbox (MSSM): Efficient Estimation and Selection of Large Multi-Level Statistical Models. https://doi.org/10.48550/arXiv.2506.13132

   :param family: Model family
   :type family: GSMMFamily
   :param ys: List of observation vectors
   :type ys: list[np.ndarray]
   :param Xs: List of model matrices
   :type Xs: list[scp.sparse.csc_array]
   :param coef: Coefficient estimate
   :type coef: np.ndarray
   :param coef_split_idx: The indices at which to split the overall coefficient vector into separate lists - one per parameter of llk.
   :type coef_split_idx: list[int]
   :param S_emb: Total penalty matrix
   :type S_emb: scp.sparse.csc_array
   :param S_norm: Scaled version of the penalty matrix (i.e., unweighted total penalty divided by it's norm).
   :type S_norm: scp.sparse.csc_array
   :param S_pinv: Generalized inverse of total penalty matrix
   :type S_pinv: scp.sparse.csc_array
   :param FS_use_rank: A list of bools indicating for which EFS updates the rank rather than the generalized inverse should be used
   :type FS_use_rank: list[bool]
   :param smooth_pen: List of penalties
   :type smooth_pen: list[LambdaTerm]
   :param c_llk: Current llk
   :type c_llk: float
   :param outer: Index of outer iteration
   :type outer: int
   :param max_inner: Maximum number of inner iterations
   :type max_inner: int
   :param min_inner: Minimum number of inner iterations
   :type min_inner: int
   :param conv_tol: Convergence tolerance
   :type conv_tol: float
   :param method: Method to use to estimate coefficients
   :type method: str
   :param piv_tol: Deprecated
   :type piv_tol: float
   :param keep_drop: Set of previously dropped coeeficients or None
   :type keep_drop: list[list[int],list[int]] | None
   :param opt_raw: If the L-qEFS update is used to estimate coefficients/lambda parameters, then this is the previous state of the quasi-Newton approximations to the (inverse) of the hessian of the log-likelihood
   :type opt_raw: scp.sparse.linalg.LinearOperator | None
   :return: A tuple containing an estimate of all coefficients, the negative hessian of the log-likelihood,cholesky of negative hessian of the penalized log-likelihood,inverse of the former (or another instance of :class:`scp.sparse.linalg.LinearOperator` representing the new quasi-newton approximation), new llk, new penalized llk, the multiple (float) added to the diagonal of the negative penalized hessian to make it invertible, an optional list of the coefficients to keep, an optional list of the estimated coefficients to drop
   :rtype: tuple[np.ndarray, scp.sparse.csc_array|None, scp.sparse.csc_array|None, scp.sparse.csc_array|scp.sparse.linalg.LinearOperator, float, float, float, list[int]|None, list[int]|None]
   """
   grad_only = method == "Grad"
   a = 0.1 # Step-size for gradient only
   #print(c_llk)

   if grad_only:
      H = None
      L = None
      LV = None
      eps = 0
   
   if method == "qEFS":
      # Define wrapper for negative (penalized) likelihood function plus function to evaluate negative gradient of the former likelihood
      # to compute line-search.
      def __neg_llk(coef,coef_split_idx,ys,Xs,family,S_emb):
         coef = coef.reshape(-1,1)
         neg_llk = -1 * family.llk(coef,coef_split_idx,ys,Xs)
         return neg_llk + 0.5*coef.T@S_emb@coef
      
      def __neg_grad(coef,coef_split_idx,ys,Xs,family,S_emb):
         # see Wood, Pya & Saefken (2016)
         coef = coef.reshape(-1,1)
         grad = family.gradient(coef,coef_split_idx,ys,Xs)
         pgrad = np.array([grad[i] - (S_emb[[i],:]@coef)[0] for i in range(len(grad))])
         return -1*pgrad.flatten()
      
      shrinkage = False
      form = opt_raw.form

      H = None
      L = None
      eps = 0

      # Now optimize penalized problem
      opt = scp.optimize.minimize(__neg_llk,
                                 np.ndarray.flatten(coef),
                                 args=(coef_split_idx,ys,Xs,family,S_emb),
                                 method="L-BFGS-B",
                                 jac = __neg_grad,
                                 options={"maxiter":max_inner,
                                          **opt_raw.bfgs_options})

      coef = opt["x"].reshape(-1,1)
      c_llk = family.llk(coef,coef_split_idx,ys,Xs)

      # Now approximate hessian of llk at penalized solution:

      # Initialize from previous solution
      maxcor = opt_raw.bfgs_options["maxcor"]

      if outer > 0 or opt_raw.init:
         sks = opt_raw.sk
         yks = opt_raw.yk
         rhos = opt_raw.rho
      else:
         sks = np.array([])
         yks = np.array([])
         rhos = np.array([])
      
      # see Nocedal & Wright, 2004 for sacling factor
      if outer > 0 or opt_raw.init:
         omega = np.dot(opt_raw.yk[-1],opt_raw.yk[-1])/np.dot(opt_raw.yk[-1],opt_raw.sk[-1])
      else:
         omega = 1
      omegas = [omega] # omega can fluctuate between updates.. keep track of smallest one within an update series and use that for scaling efs update later
      updates = 0

      # Initialize hessian + inverse
      H0 = scp.sparse.identity(len(coef),format='csc')*omega
      V0 = scp.sparse.identity(len(coef),format='csc')*(1/omega)

      if len(sks) > 0:
         if form == 'SR1':
            t1_llk, t2_llk, t3_llk = computeVSR1(opt_raw.sk,opt_raw.yk,opt_raw.rho,V0,1/omega,make_psd=True,explicit=False)
         else:
            t1_llk, t2_llk, t3_llk = computeV(opt_raw.sk,opt_raw.yk,opt_raw.rho,V0,explicit=False)

      # Also compute H0 and V0 for penalized problem.
      pH0 = scp.sparse.identity(len(coef),format='csc')*omega
      Lp, Pr, _ = cpp_cholP(pH0 + S_emb)
      LVp0 = compute_Linv(Lp,10)
      LV0 = apply_eigen_perm(Pr,LVp0)
      pV0 = LV0.T @ LV0

      # Should hessian be estimated on gradients from completely un-penalized problem or on problem with a little shrinkage?
      # Latter might be more appropriate for random effects where hessian is far from PD!
      S_up = scp.sparse.csc_matrix((len(coef), len(coef)))
      if shrinkage:
         S_up = scp.sparse.identity(len(coef),format='csc')*1e-3

   # Update coefficients:
   full_coef = None

   if keep_drop is None:
      keep = None
      drop = None
   else:
      keep = keep_drop[0]
      drop = keep_drop[1]

      # Handle previous drop
      full_coef = np.zeros_like(coef) # Prepare zeroed full coef vector
      full_coef_split_idx = copy.deepcopy(coef_split_idx) # Original split index
      coef, coef_split_idx, Xs, S_emb, c_llk, c_pen_llk = handle_drop_gsmm(family, ys, coef, keep, Xs, S_emb) # Drop

   converged = False
   checked_identifiable = False
   inner = 0
   while converged == False:

      if inner >= max_inner:
         break
      
      # Get llk derivatives with respect to coef
      grad = family.gradient(coef,coef_split_idx,ys,Xs)

      if grad_only == False:
         if method == "qEFS":
            # Update limited memory representation of the negative hessian of the llk
            if shrinkage == False:
               grad_up = grad
            else:
               grad_up = np.array([grad[i] - (S_up[[i],:]@coef)[0] for i in range(len(grad))]).reshape(-1,1)

            # First compute current direction on "un-penalized" llk
            if len(sks) > 0:
               step = V0@grad_up + t1_llk@t2_llk@t3_llk@grad_up
            else:
               step = V0@grad_up

            if form == 'SR1':
               # Find step satisfying armijo condition
               alpha = back_track_alpha(coef,step,__neg_llk,__neg_grad,coef_split_idx,ys,Xs,family,S_up,alpha_max=1)
               new_slope = 1
            else:
               # Find a step that meets the Wolfe conditions (Nocedal & Wright, 2004)
               alpha,_,_,_,_,new_slope = scp.optimize.line_search(__neg_llk,__neg_grad,coef.flatten(),step.flatten(),
                                                                  args=(coef_split_idx,ys,Xs,family,S_up),
                                                                  maxiter=100,amax=1)
            
            if alpha is None:
               new_slope = None
               alpha = 1e-7

            # Compute gradient at new point
            next_grad_up = family.gradient(coef + alpha*step,coef_split_idx,ys,Xs)
            if shrinkage:
               next_grad_up = np.array([next_grad_up[i] - (S_up[[i],:]@(coef + alpha*step))[0] for i in range(len(next_grad_up))]).reshape(-1,1)

            # Form update vectors for limited memory representation of hessian of negative llk
            yk = (-1*next_grad_up) - (-1*grad_up)
            sk = alpha*step
            rhok = 1/(yk.T@sk)

            if form == 'SR1':
               # Check if SR1 update is defined (see Nocedal & Wright, 2004)
               skip = test_SR1(sk,yk,rhok,sks,yks,rhos)

               if outer == 0 and len(sks) == 0 and (skip or new_slope is None):
                  # Potentially very bad start estimate, try find a better one
                  coef, _, _ = restart_coef(coef,None,None,len(coef),coef_split_idx,ys,Xs,S_emb,family,inner,0)
                  c_llk = family.llk(coef,coef_split_idx,ys,Xs)
                  grad = family.gradient(coef,coef_split_idx,ys,Xs)

            if new_slope is not None and (form != 'SR1' or (skip == False)):
               # Wolfe/Armijo met, can collect update vectors

               if form != 'SR1' and len(sks) > 0:
                  # But first dampen for BFGS update - see Nocedal & Wright (2004):
                  Ht1, _, Ht2, Ht3 = computeH(sks,yks,rhos,H0,explicit=False)
                  Bs = H0@sk + Ht1@Ht2@Ht3@sk
                  sBs = sk.T@Bs
                  
                  if sk.T@yk < 0.2*sBs:
                     theta = (0.8*sBs)/(sBs - sk.T@yk)
                     yk = theta*yk + (1-theta)*Bs
               
               if len(sks) == 0:
                  sks = sk.T
                  yks = yk.T
                  rhos = np.array([rhok[0,0]])
               else:
                  sks = np.append(sks,sk.T,axis=0)
                  yks = np.append(yks,yk.T,axis=0)
                  rhos = np.append(rhos,rhok[0,0])

               # Discard oldest update vector
               if sks.shape[0] > maxcor:
                  sks = np.delete(sks,0,axis=0)
                  yks = np.delete(yks,0,axis=0)
                  rhos = np.delete(rhos,0)
            
            # Update omega to better scale hessian (again see Nocedal & Wright, 2004)
            if sks.shape[0] > 0 and yk.T@sk > 0:
               omega = np.dot(yks[-1],yks[-1])/np.dot(yks[-1],sks[-1])
               if omega < 0 or (sks.shape[0] == 1 and form == 'SR1'):
                  # Cannot use scaling for first update vector since inverse of hessian will result in
                  # zero t2 term due to cancellation
                  omega = 1
            else:
               omega = 1

            omegas.append(omega)

            if len(omegas) > maxcor:
               del omegas[0]
            
            # Update H0/V0 (rather H0(k)/V0(k))
            H0 = scp.sparse.identity(len(coef),format='csc')*omega
            V0 = scp.sparse.identity(len(coef),format='csc')*(1/omega)

            # Also for penalized problem!
            pH0 = scp.sparse.identity(len(coef),format='csc')*omega
            Lp, Pr, _ = cpp_cholP(pH0 + S_emb)
            LVp0 = compute_Linv(Lp,10)
            LV0 = apply_eigen_perm(Pr,LVp0)
            pV0 = LV0.T @ LV0

            if sks.shape[0] > 0:

               # Compute updated estimate of inverse of negative hessian of llk (implicitly)
               if form == 'SR1':
                  t1_llk, t2_llk, t3_llk = computeVSR1(sks,yks,rhos,V0,1/omega,make_psd=True,explicit=False) #
               else:
                  t1_llk, t2_llk, t3_llk = computeV(sks,yks,rhos,V0,explicit=False)

               # Can now form penalized gradient and approximate penalized hessian to update penalized
               # coefficients. Important: pass un-penalized hessian here!
               # Because # Below we get int2
               # H = H0 + nt1 @ int2 @ nt3; that is the estimate of the negative un-penalized hessian of llk
               # Now we replace H0 with pH0 - which is really: H0 + S_emb.
               # Now H is our estimate of the negative hessian of the penalized llk.
               # Now, using the Woodbury identity:
               # pV = (H)^-1 = pV0 - pV0@nt1@ (int2^-1 + nt3@pV0@nt1)^-1 @ nt3@pV0
               #
               # since nt3=nt1.T, and int2^-1 = nt2 we have:
               #
               # pV = pV0 - pV0@nt1@ (nt2 + nt1.T@pV0@nt1)^-1 @ nt1.t@pV0

               # Compute inverse:
               if form != 'SR1':
                  nt1,nt2,int2,nt3 = computeH(sks,yks,rhos,H0,explicit=False)

                  invt2 = nt2 + nt3@pV0@nt1

                  U,sv_invt2,VT = scp.linalg.svd(invt2,lapack_driver='gesvd')

                  # Nowe we can compute all parts for the Woodbury identy to obtain pV
                  t2 = VT.T @ np.diag(1/sv_invt2)  @  U.T

                  t1 = pV0@nt1
                  t3 = nt3@pV0
               else:
                  nt1, int2, nt3 = computeHSR1(sks,yks,rhos,H0,omega,make_psd=True,explicit=False)

                  # When using SR1 int2 is potentially singular, so we need a modified Woodbury inverse that accounts for that.
                  # This is given by eq. 23 in Henderson & Searle (1981):
                  invt2 = np.identity(int2.shape[1]) + int2@nt3@pV0@nt1

                  U,sv_invt2,VT = scp.linalg.svd(invt2,lapack_driver='gesvd')

                  # Nowe we can compute all parts for the modified Woodbury identy to obtain V
                  t2 = VT.T @ np.diag(1/sv_invt2)  @  U.T

                  t1 = pV0@nt1
                  t3 = int2@nt3@pV0
               # We now have: pV = pV0 - pV0@nt1@t2@nt3@pV0, but don't have to form that explcitly!

            # All we need at this point is to form the penalized gradient. # And then we can compute
            # next_coef via quasi newton step as well (handled below).
            pgrad = np.array([grad[i] - (S_emb[[i],:]@coef)[0] for i in range(len(grad))]).reshape(-1,1)

         else:
            H = family.hessian(coef,coef_split_idx,ys,Xs)

      ##################################### Update coef and perform step size control #####################################

      # Update Coefficients
      if grad_only:
         next_coef = gd_coef_smooth(coef,grad,S_emb,a)

      elif method == "qEFS":
         
         # Update coefficients via implicit quasi newton step
         if sks.shape[0] > 0:
            pen_step = pV0@pgrad - t1@t2@t3@pgrad
         else:
            pen_step = pV0@pgrad               
         
         # Line search on penalized problem to ensure we make some progress.
         with warnings.catch_warnings(): # Line search might fail, but we handle that.
            warnings.simplefilter("ignore")
            # Find step satisfying armijo condition
            alpha_pen = back_track_alpha(coef,pen_step,__neg_llk,__neg_grad,coef_split_idx,ys,Xs,family,S_emb,alpha_max=10)

         # Just backtrack in step size control
         if alpha_pen is None:
            alpha_pen = 1
         
         next_coef = coef + alpha_pen*pen_step
         updates += 1
         
      else:
         next_coef,L,LV,eps = newton_coef_smooth(coef,grad,H,S_emb)

      # Prepare to check convergence
      prev_llk_cur_pen = c_llk - 0.5*coef.T@S_emb@coef

      # Store previous coef and llk in case we fall-back to gd
      if method == 'qEFS':
         prev_coef = copy.deepcopy(coef)
         prev_llk = c_llk

      # Perform step length control - will immediately pass if line search was succesful.
      coef,c_llk,c_pen_llk,a = correct_coef_step_gen_smooth(family,ys,Xs,coef,next_coef,coef_split_idx,c_llk,S_emb,a)

      # Very poor start estimate, restart
      if grad_only and outer == 0 and inner <= 20 and np.abs(c_pen_llk - prev_llk_cur_pen) < conv_tol*np.abs(c_pen_llk):
         coef, _, _ = restart_coef(coef,None,None,len(coef),coef_split_idx,ys,Xs,S_emb,family,inner,0)
         c_llk = family.llk(coef,coef_split_idx,ys,Xs)
         c_pen_llk = c_llk - 0.5*coef.T@S_emb@coef
         a = 0.1

      # Check if this step would converge, if that is the case try gradient first
      if method == 'qEFS' and np.abs(c_pen_llk - prev_llk_cur_pen) < conv_tol*np.abs(c_pen_llk):

         alpha_pen_grad = back_track_alpha(prev_coef,pgrad,__neg_llk,__neg_grad,coef_split_idx,ys,Xs,family,S_emb,alpha_max=1)

         if alpha_pen_grad is not None:

            # Test llk for improvement over quasi newton step
            grad_pen_llk = family.llk(prev_coef + alpha_pen_grad*pgrad,coef_split_idx,ys,Xs) - 0.5*(prev_coef + alpha_pen_grad*pgrad).T@S_emb@(prev_coef + alpha_pen_grad*pgrad)

            if grad_pen_llk > c_pen_llk:
               next_coef = prev_coef + alpha_pen_grad*pgrad
               coef,c_llk,c_pen_llk,a = correct_coef_step_gen_smooth(family,ys,Xs,prev_coef,next_coef,coef_split_idx,prev_llk,S_emb,a)

      if np.abs(c_pen_llk - prev_llk_cur_pen) < conv_tol*np.abs(c_pen_llk) and (method != "qEFS" or (opt["nit"] == 1 and outer > 0) or (updates >= maxcor)):
         converged = True

         # Check for drop
         if (keep_drop is None) and (checked_identifiable == False) and (method in ["QR/Chol","LU/Chol","Direct/Chol"]):
            keep,drop = identify_drop(H,S_norm,method.split("/")[0])

            # No drop necessary -> converged
            if len(drop) == 0:
               keep = None
               drop = None
               break
            else:
               # Found drop, but need to check whether it is safe
               drop_valid,drop_pen_llk = check_drop_valid_gensmooth(ys,coef,Xs,S_emb,keep,family)

               # Skip if likelihood becomes invalid
               if drop_valid == False:
                  keep = None
                  drop = None
                  break
               # If we identify all or all but one coefficients to drop also skip
               elif drop_valid and len(drop) >= len(coef) - 1:
                  keep = None
                  drop = None
                  break

               # At this point: found & accepted drop -> adjust parameters
               full_coef = np.zeros_like(coef) # Prepare zeroed full coef vector
               full_coef_split_idx = copy.deepcopy(coef_split_idx)
               coef, coef_split_idx, Xs, S_emb, c_llk, c_pen_llk = handle_drop_gsmm(family, ys, coef, keep, Xs, S_emb) # Drop
               converged = False # Re-iterate until convergence
               inner = -1 # Reset fitting iterations
               checked_identifiable = True
         
         # Simply converge
         else:
            break

      if eps <= 0 and outer > 0 and inner >= (min_inner-1):
         break # end inner loop and immediately optimize lambda again.

      inner += 1
   
   if method == 'qEFS':
      # Store update to S and Y in scipy LbfgsInvHess and keep that in LV, since this is what we need later
      LV = scp.optimize.LbfgsInvHessProduct(sks,yks)
      LV.nit = opt["nit"]
      LV.omega = omega#np.min(omegas)
      LV.method = "qEFS"
      LV.updates = updates
      LV.form = form
   
   # Need to fill full_coef at this point and pad LV if we dropped
   if drop is not None:
      full_coef[keep] = coef
      
      LVdat,LVrow,LVcol = translate_sparse(LV) # LV.T@LV = V
      LVrow = keep[LVrow]
      LVcol = keep[LVcol]
      LV = scp.sparse.csc_array((LVdat,(LVrow,LVcol)),shape=(len(full_coef),len(full_coef)))
   else:
      # Full coef is simply coef
      full_coef = coef
      full_coef_split_idx = coef_split_idx


   if method != 'qEFS':
      # Make sure at convergence negative Hessian of llk is at least positive semi-definite as well
      checkH = True
      checkHc = 0

      # Compute lgdetDs once
      if smooth_pen is not None:
         lgdetDs = []

         for lti,lTerm in enumerate(smooth_pen):

            lt_rank = None
            if FS_use_rank[lti]:
               lt_rank = lTerm.rank

            lgdetD,_ = compute_lgdetD_bsb(lt_rank,lTerm.lam,S_pinv,lTerm.S_J_emb,full_coef)
            lgdetDs.append(lgdetD)
      
      while checkH and (smooth_pen is not None):
         #_,S_pinv,_,FS_use_rank = compute_S_emb_pinv_det(len(coef),smooth_pen,"svd")

         # Re-compute ldetHS
         _,_, ldetHSs = calculate_edf(None,None,LV,smooth_pen,lgdetDs,len(full_coef),10,None,None)

         # And check whether Theorem 1 now holds
         checkH = np.any([lgdetDs[lti] - ldetHSs[lti] < 0 for lti in range(len(smooth_pen))])

         if checkH:
            if eps == 0:
               eps += 1e-14
            else:
               eps *= 2
            _,L,LV,_ = newton_coef_smooth(coef,grad,H - eps*scp.sparse.identity(H.shape[1],format='csc'),S_emb)

            # Pad new LV in case of drop again:
            if drop is not None:
               LVdat,LVrow,LVcol = translate_sparse(LV) # LV.T@LV = V
               LVrow = keep[LVrow]
               LVcol = keep[LVcol]
               LV = scp.sparse.csc_array((LVdat,(LVrow,LVcol)),shape=(len(full_coef),len(full_coef)))

         if checkHc > 30:
            break

         checkHc += 1
   
   # Done except when we have a drop -> still need to pad H and L
   if drop is not None:
      #print(coef)
      split_coef = np.split(full_coef,full_coef_split_idx)

      # Now H, L
      Hdat,Hrow,Hcol = translate_sparse(H)
      Ldat,Lrow,Lcol = translate_sparse(L.tocsc()) # L@L.T = H_pen
      
      Hrow = keep[Hrow]
      Hcol = keep[Hcol]
      Lrow = keep[Lrow]
      Lcol = keep[Lcol]

      H = scp.sparse.csc_array((Hdat,(Hrow,Hcol)),shape=(len(full_coef),len(full_coef)))
      L = scp.sparse.csc_array((Ldat,(Lrow,Lcol)),shape=(len(full_coef),len(full_coef)))
   
   return full_coef,H,L,LV,c_llk,c_pen_llk,eps,keep,drop

def correct_lambda_step_gen_smooth(family:GSMMFamily,ys:list[np.ndarray],Xs:list[scp.sparse.csc_array],S_norm:scp.sparse.csc_array,n_coef:int,form_n_coef:list[int],form_up_coef:list[int],coef:np.ndarray,
                                    coef_split_idx:list[int],smooth_pen:list[LambdaTerm],lam_delta:np.ndarray,
                                    extend_by:dict,was_extended:list[bool],c_llk:float,fit_info:Fit_info,outer:int,
                                    max_inner:int,min_inner:int,conv_tol:float,gamma:float,method:str,qEFSH:str,overwrite_coef:bool,qEFS_init_converge:bool,optimizer:str,
                                    __old_opt:scp.sparse.linalg.LinearOperator|None,use_grad:bool,__neg_pen_llk:Callable,__neg_pen_grad:Callable,piv_tol:float,keep_drop:list[list[int],list[int]]|None,extend_lambda:bool,
                                    extension_method_lam:str,control_lambda:int,repara:bool,n_c:int,
                                    init_bfgs_options:dict,bfgs_options:dict) -> tuple[np.ndarray,scp.sparse.csc_array|None,scp.sparse.csc_array|None,scp.sparse.csc_array|scp.sparse.linalg.LinearOperator,scp.sparse.csc_array|None,float,float,scp.sparse.linalg.LinearOperator|None,list[int],list[int],scp.sparse.csc_array,list[LambdaTerm],float,list[float],np.ndarray]:
   """Updates and performs step-length control for the vector of lambda parameters of a GSMM model. Essentially completes the steps discussed in sections 3.3 and 4 of the paper by Krause et al. (submitted).

   Based on steps outlined by Wood, Pya, & Säfken (2016).

   References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
     - Krause et al. (submitted). The Mixed-Sparse-Smooth-Model Toolbox (MSSM): Efficient Estimation and Selection of Large Multi-Level Statistical Models. https://doi.org/10.48550/arXiv.2506.13132

   :param family: Model family
   :type family: GSMMFamily
   :param ys: List of observation vectors
   :type ys: list[np.ndarray]
   :param Xs: List of model matrices
   :type Xs: list[scp.sparse.csc_array]
   :param S_norm: Scaled version of the penalty matrix (i.e., unweighted total penalty divided by it's norm).
   :type S_norm: scp.sparse.csc_array
   :param n_coef: Number of coefficients
   :type n_coef: int
   :param form_n_coef: List of number of coefficients per formula
   :type form_n_coef: list[int]
   :param form_up_coef: List of un-penalized number of coefficients per formula
   :type form_up_coef: list[int]
   :param coef: Coefficient estimate
   :type coef: np.ndarray
   :param coef_split_idx: The indices at which to split the overall coefficient vector into separate lists - one per parameter.
   :type coef_split_idx: list[int]
   :param smooth_pen: List of penalties
   :type smooth_pen: list[LambdaTerm]
   :param lam_delta: Update to vector of lambda parameters
   :type lam_delta: np.ndarray
   :param extend_by: Extension info dictionary
   :type extend_by: dict
   :param was_extended: List holding indication per lambda parameter whether it was extended or not
   :type was_extended: list[bool]
   :param c_llk: Current llk
   :type c_llk: float
   :param fit_info: A :class:`Fit_info` object
   :type fit_info: Fit_info
   :param outer: Index of outer iteration
   :type outer: int
   :param max_inner: Maximum number of inner iterations
   :type max_inner: int
   :param min_inner: Minimum number of inner iterations
   :type min_inner: int
   :param conv_tol: Convergence tolerance
   :type conv_tol: float
   :param gamma: Weight factor determining whether we should look for smoother or less smooth models
   :type gamma: float
   :param method: Method to use to estimate coefficients (and lambda parameter)
   :type method: str
   :param qEFSH: Should the hessian approximation use a symmetric rank 1 update (``qEFSH='SR1'``) that is forced to result in positive semi-definiteness of the approximation or the standard bfgs update (``qEFSH='BFGS'``) 
   :type qEFSH: str
   :param overwrite_coef: Whether the initial coefficients passed to the optimization routine should be over-written by the solution obtained for the un-penalized version of the problem when ``method='qEFS'``. Setting this to False will be useful when passing coefficients from a simpler model to initialize a more complex one. Only has an effect when ``qEFS_init_converge=True``.
   :type overwrite_coef: bool
   :param qEFS_init_converge: Whether to optimize the un-penalzied version of the model and to use the hessian (and optionally coefficients, if ``overwrite_coef=True``) to initialize the q-EFS solver. Ignored if ``method!='qEFS'``.
   :type qEFS_init_converge: bool
   :param optimizer: Deprecated
   :type optimizer: str
   :param __old_opt: If the L-qEFS update is used to estimate coefficients/lambda parameters, then this is the previous state of the quasi-Newton approximations to the (inverse) of the hessian of the log-likelihood
   :type __old_opt: scp.sparse.linalg.LinearOperator | None
   :param use_grad: Deprecated
   :type use_grad: bool
   :param __neg_pen_llk: Function to evaluate negative penalized log-likelihood
   :type __neg_pen_llk: Callable
   :param __neg_pen_grad: Function to evaluate gradient of negative penalized log-likelihood
   :type __neg_pen_grad: Callable
   :param piv_tol: Deprecated
   :type piv_tol: float
   :param keep_drop: Set of previously dropped coeeficients or None
   :type keep_drop: list[list[int],list[int]] | None
   :param extend_lambda: Whether lambda proposals should be accelerated or not. Can lower the number of new smoothing penalty proposals necessary
   :type extend_lambda: bool
   :param extension_method_lam:  Which method to use to extend lambda proposals.
   :type extension_method_lam: str
   :param control_lambda: Whether lambda proposals should be checked (and if necessary decreased) for whether or not they (approxiately) increase the Laplace approximate restricted maximum likelihood of the model. For ``method != 'qEFS'`` the following options are available: setting this to 0 disables control. Setting it to 1 means the step will never be smaller than the original EFS update but extensions will be removed in case the objective was exceeded (only has an effect when setting ``extend_lambda=True``). Setting it to 2 means that steps will generally be halved when they fail to increase the aproximate REML criterion. For ``method=='qEFS'`` the following options are available: setting this to 0 disables control. Setting it to 1 means the check described by Krause et al. (submitted) will be performed to control updates to lambda. Setting it to 2 means that steps will generally be halved when they fail to increase the aproximate REML criterion (note, that the gradient is based on quasi-newton approximations as well and thus less accurate). Setting it to 3 means both checks (i.e., 1 and 2) are performed.
   :type control_lambda: int
   :param repara: Whether to apply a stabilizing re-parameterization to the model
   :type repara: bool
   :param n_c: Number of cores to use
   :type n_c: int
   :param init_bfgs_options: An optional dictionary holding the same key:value pairs that can be passed to ``bfgs_options`` but pased to the optimizer of the un-penalized problem. Only has an effect when ``qEFS_init_converge=True``. 
   :type init_bfgs_options: dict
   :param bfgs_options: An optional dictionary holding arguments that should be passed on to the call of :func:`scipy.optimize.minimize` if ``method=='qEFS'``.
   :type bfgs_options: dict
   :return: coef estimate under corrected lambda, the negative hessian of the log-likelihood, cholesky of negative hessian of the penalized log-likelihood, inverse of the former (or another instance of :class:`scp.sparse.linalg.LinearOperator` representing the new quasi-newton approximation), covariance matrix of coefficients, next llk, next penalized llk, if the L-qEFS update is used to estimate coefficients/lambda parameters a ``scp.sparse.linalg.LinearOperator`` holding the previous quasi-Newton approximations to the (inverse) of the hessian of the log-likelihood, an optional list of the coefficients to keep, an optional list of the estimated coefficients to drop, new total penalty matrix, new list of penalties, total edf, term-wise edfs, the update to the lambda vector
   :rtype: tuple[np.ndarray, scp.sparse.csc_array|None, scp.sparse.csc_array|None, scp.sparse.csc_array|scp.sparse.linalg.LinearOperator, scp.sparse.csc_array|None, float, float, scp.sparse.linalg.LinearOperator|None, list[int], list[int], scp.sparse.csc_array, list[LambdaTerm], float, list[float], np.ndarray]
   """
   # Fitting iteration and step size control for smoothing parameters of general smooth model.
   # Basically a more general copy of the function for gammlss. Again, step-size control is not obvious - because we have only approximate REMl
   # and approximate derivative, because we drop the last term involving the derivative of the negative penalized
   # Hessian with respect to the smoothing parameters (see section 4 in Wood & Fasiolo, 2017). However, what we
   # can do is at least undo the acceleration if we over-shoot the approximate derivative...

   lam_accepted = False
   while lam_accepted == False:
      
      # Build new penalties
      S_emb,S_pinv,_,FS_use_rank = compute_S_emb_pinv_det(n_coef,smooth_pen,"svd")

      # Re-parameterize
      if repara:
         coef_rp,Xs_rp,smooth_pen_rp,S_emb,S_norm,S_root_rp,S_pinv,Q_emb,Qs = reparam_model(form_n_coef, form_up_coef, coef, coef_split_idx, Xs,
                                                                                            smooth_pen, form_inverse=True,
                                                                                            form_root=False, form_balanced=True, n_c=n_c)
         
         if __old_opt is not None:
            # sk and yk of previous optimizer are in state before re-param and will have to be transformed here
            sk = __old_opt.sk
            yk = __old_opt.yk

            if len(sk) > 0 and sk.shape[1] == n_coef:
               sk = sk @ Q_emb
               yk = yk @ Q_emb

               __old_opt.yk = yk
               __old_opt.sk = sk

      else:
         coef_rp = coef
         Xs_rp = Xs
         smooth_pen_rp = smooth_pen

      # Then re-compute coef
      if optimizer == "Newton":
         
         # Optimize un-penalized problem first to get a good starting estimate for hessian.
         if method == "qEFS" and outer == 0:
            if qEFS_init_converge:
               opt_raw = scp.optimize.minimize(__neg_pen_llk,
                                             np.ndarray.flatten(coef_rp),
                                             args=(coef_split_idx,ys,Xs_rp,family,scp.sparse.csc_matrix((len(coef_rp), len(coef_rp)))),
                                             method="L-BFGS-B",
                                             jac = __neg_pen_grad,
                                             options={"maxiter":max_inner,
                                                      **init_bfgs_options})
               #print(opt_raw)
               if opt_raw["nit"] > 1:
                  # Set "initial" coefficients to solution found for un-penalized problem.
                  if overwrite_coef:
                     coef_rp = opt_raw["x"].reshape(-1,1)
                     c_llk = family.llk(coef_rp,coef_split_idx,ys,Xs_rp)

                  __old_opt = opt_raw.hess_inv
                  __old_opt.init = True
                  __old_opt.nit = opt_raw["nit"]

                  H_y, H_s = opt_raw.hess_inv.yk, opt_raw.hess_inv.sk
                  
                  # Get scaling for hessian from Nocedal & Wright, 2004:
                  omega_Raw = np.dot(H_y[-1],H_y[-1])/np.dot(H_y[-1],H_s[-1])
                  __old_opt.omega = omega_Raw#np.min(H_ev)
               else:
                  __old_opt = scp.optimize.LbfgsInvHessProduct(np.array([1]).reshape(1,1),np.array([1]).reshape(1,1))
                  __old_opt.init = False
                  __old_opt.omega = 1
            else:
               __old_opt = scp.optimize.LbfgsInvHessProduct(np.array([1]).reshape(1,1),np.array([1]).reshape(1,1))
               __old_opt.init = False
               __old_opt.omega = 1

            __old_opt.method = 'qEFS'
            __old_opt.form = qEFSH
            __old_opt.bfgs_options = bfgs_options

            
         next_coef,H,L,LV,next_llk,next_pen_llk,eps,keep,drop = update_coef_gen_smooth(family,ys,Xs_rp,coef_rp,
                                                                                       coef_split_idx,S_emb,
                                                                                       S_norm,S_pinv,FS_use_rank,smooth_pen_rp,
                                                                                       c_llk,outer,max_inner,
                                                                                       min_inner,conv_tol,
                                                                                       method,piv_tol,keep_drop,
                                                                                       __old_opt)
         
         if method == "qEFS" and outer == 0 and __old_opt.init == False:
            __old_opt = LV
            __old_opt.bfgs_options = bfgs_options
            __old_opt.init = True


         V = None

      else:
         raise DeprecationWarning("Non-Newton optimizers are deprecated.")
      
      # Now re-compute lgdetDs, ldetHS, and bsbs
      lgdetDs = []
      bsbs = []
      for lti,lTerm in enumerate(smooth_pen_rp):

            lt_rank = None
            if FS_use_rank[lti]:
               lt_rank = lTerm.rank

            lgdetD,bsb = compute_lgdetD_bsb(lt_rank,lTerm.lam,S_pinv,lTerm.S_J_emb,next_coef)
            lgdetDs.append(lgdetD)
            bsbs.append(bsb*gamma)

      total_edf,term_edfs, ldetHSs = calculate_edf(None,None,LV,smooth_pen_rp,lgdetDs,n_coef,n_c,None,S_emb)
      fit_info.lambda_updates += 1

      # Can exit loop here, no extension and no control
      if outer == 0  or (control_lambda < 1) or (extend_lambda == False and control_lambda < 2):
         lam_accepted = True
         continue
      
      # Compute approximate!!! gradient of REML with respect to lambda
      # to check if step size needs to be reduced (part of step 6 in Wood, 2017).
      lam_grad = [grad_lambda(lgdetDs[lti],ldetHSs[lti],bsbs[lti],1) for lti in range(len(smooth_pen_rp))]
      lam_grad = np.array(lam_grad).reshape(-1,1) 
      check = lam_grad.T @ lam_delta

      # Now undo the acceleration if overall direction is **very** off - don't just check against 0 because
      # our criterion is approximate, so we can be more lenient (see Wood et al., 2017).
      lam_changes = 0 
      if check[0] < 1e-7*-abs(next_pen_llk):
         for lti,lTerm in enumerate(smooth_pen):

            # For extended terms undo extension
            if extend_lambda and was_extended[lti]:
               lam, dLam = undo_extension_lambda_step(lti,lTerm.lam,lam_delta[lti],extend_by,was_extended, extension_method_lam, None)

               lTerm.lam = lam
               lam_delta[lti] = dLam
               lam_changes += 1
            
            elif control_lambda >= 2:
               # Continue to half step - this is not necessarily because we overshoot the REML
               # (because we only have approximate gradient), but can help preventing that models
               # oscillate around estimates.
               lam_delta[lti] = lam_delta[lti]/2
               lTerm.lam -= lam_delta[lti]
               lam_changes += 1
         
         # If we did not change anything, simply accept
         if lam_changes == 0 or np.linalg.norm(lam_delta) < 1e-7:
            lam_accepted = True
      
      # If we pass the check we can simply accept.
      else:
         lam_accepted = True
   
   # At this point we have accepted the lambda step and can propose a new one!
   if drop is not None:
      fit_info.dropped = drop
   else:
      fit_info.dropped = None

   # For qEFS we check whether new approximation results in worse balance of efs update - then we fall back to previous approximation
   # control_lambda = 0 -> No checks for qEFS
   # control_lambda = 1 -> balance check (below)
   # control_lambda = 2 -> approximate gradient check
   # control_lambda = 3 -> both checks
   if method == "qEFS":
      if control_lambda in [1,3]:
         total_edf2,term_edfs2, ldetHSs2 = calculate_edf(None,None,__old_opt,smooth_pen_rp,lgdetDs,n_coef,n_c,None,S_emb)
         diff1 = [np.abs((lgdetDs[lti] - ldetHSs[lti]) - bsbs[lti]) for lti in range(len(smooth_pen_rp))]
         diff2 = [np.abs((lgdetDs[lti] - ldetHSs2[lti]) - bsbs[lti]) for lti in range(len(smooth_pen_rp))]
         #print([(lgdetDs[lti] - ldetHSs[lti]) - bsbs[lti] for lti in range(len(smooth_pen_rp))])
         #print([(lgdetDs[lti] - ldetHSs2[lti]) - bsbs[lti] for lti in range(len(smooth_pen_rp))])
         #print(np.mean(diff2),np.mean(diff1))

         if np.mean(diff2) < np.mean(diff1):
            # Reset approximation to previous one
            nit = LV.nit
            LV = copy.deepcopy(__old_opt)
            LV.nit = nit
            ldetHSs = ldetHSs2
            total_edf = total_edf2
            term_edfs = term_edfs2

      # Propagate current implicit approximation to hessian of negative llk
      LV.bfgs_options = bfgs_options
      __old_opt = copy.deepcopy(LV)

   step_norm = np.linalg.norm(lam_delta)
   lam_delta = []
   for lti,lTerm in enumerate(smooth_pen):

      lgdetD = lgdetDs[lti]
      ldetHS = ldetHSs[lti]
      bsb = bsbs[lti]
      
      dLam = step_fellner_schall_sparse(lgdetD,ldetHS,bsb[0,0],lTerm.lam,1)
      #print("Theorem 1:",lgdetD-ldetHS,bsb)

      # For poorly scaled/ill-identifiable problems we cannot rely on the theorems by Wood
      # & Fasiolo (2017) - so the condition below will ocasionally be met, in which case we just want to
      # take very small steps until it hopefully gets more stable (due to term dropping or better lambda value).
      if lgdetD - ldetHS < 0:
         dLam = np.sign(dLam) * min(abs(lTerm.lam)*0.001,abs(dLam))
         dLam = 0 if lTerm.lam+dLam < 0 else dLam

      if extend_lambda:
         dLam,extend_by,was_extended = extend_lambda_step(lti,lTerm.lam,dLam,extend_by,was_extended,extension_method_lam)
         if method != "qEFS"  and (((outer > 100 and step_norm < 5e-3) or outer > 250 or (max(eps,fit_info.eps) > 0 and outer > 50)) and was_extended[lti]):
            if max(eps,fit_info.eps) > 0 and outer > 50:
               damp = 1*np.power(0.9,outer-50)
            elif step_norm < 5e-3 and outer <= 250 and max(eps,fit_info.eps) <= 0:
               damp = 1*np.power(0.9,outer-100)
            elif outer > 250 and max(eps,fit_info.eps) <= 0:
               damp = 1*np.power(0.9,outer-250)

            extend_by["acc"][lti]*=damp

      # Dampen steps taken if Hessian continues to be modified to prevent erratic jumps
      if method != "qEFS" and ((outer > 100 and step_norm < 5e-3) or outer > 250 or (max(eps,fit_info.eps) > 0 and outer > 50)):
         if max(eps,fit_info.eps) > 0 and outer > 50:
            damp = 1*np.power(0.9,outer-50)
         elif step_norm < 5e-3 and outer <= 250 and max(eps,fit_info.eps) <= 0:
            damp = 1*np.power(0.9,outer-100)
         elif outer > 250 and max(eps,fit_info.eps) <= 0:
            damp = 1*np.power(0.9,outer-250)

         dLam = damp*dLam

      lam_delta.append(dLam)
   
   fit_info.eps = eps

   # And un-do the latest re-parameterization
   if repara:
         
      # Transform S_emb (which is currently S_emb_rp)
      S_emb = Q_emb @ S_emb @ Q_emb.T

      split_coef = np.split(next_coef,coef_split_idx)

      # Transform coef
      for qi,Q in enumerate(Qs):
         split_coef[qi] = Q@split_coef[qi]
      next_coef = np.concatenate(split_coef).reshape(-1,1)

      # Transform H, L, LV
      if method != 'qEFS':
         H = Q_emb @ H @ Q_emb.T
         L = Q_emb @ L
         LV = LV @ Q_emb.T
      else:
         # sk and yk of previous and current optimizer are in state after re-param and will have to be transformed here
         sk1 = __old_opt.sk
         yk1 = __old_opt.yk
         sk2 = LV.sk
         yk2 = LV.yk

         if len(sk1) > 0 and sk1.shape[1] == n_coef:
            sk1 = sk1 @ Q_emb.T
            yk1 = yk1 @ Q_emb.T
            sk2 = sk2 @ Q_emb.T
            yk2 = yk2 @ Q_emb.T

            __old_opt.yk = yk1
            __old_opt.sk = sk1
            LV.yk = yk2
            LV.sk = sk2

   return next_coef,H,L,LV,V,next_llk,next_pen_llk,__old_opt,keep,drop,S_emb,smooth_pen,total_edf,term_edfs,lam_delta

def solve_generalSmooth_sparse(family:GSMMFamily,ys:list[np.ndarray],Xs:list[scp.sparse.csc_array],form_n_coef:list[int],form_up_coef:list[int],coef:np.ndarray,coef_split_idx:list[int],smooth_pen:list[LambdaTerm],
                               max_outer:int=50,max_inner:int=50,min_inner:int=50,conv_tol:float=1e-7,extend_lambda:bool=True,extension_method_lam:str = "nesterov2",control_lambda:int=1,optimizer:str="Newton",method:str="Chol",
                               check_cond:int=1,piv_tol:float=0.175,repara:bool=True,should_keep_drop:bool=True,form_VH:bool=True,use_grad:bool=False,gamma:float=1,qEFSH:str='SR1',overwrite_coef:bool=True,max_restarts:int=0,
                               qEFS_init_converge:bool=True,prefit_grad:bool=False,progress_bar:bool=True,n_c:int=10,init_bfgs_options:dict={"gtol":1e-9,"ftol":1e-9,"maxcor":30,"maxls":100,"maxfun":1e7},
                               bfgs_options:dict={"gtol":1e-9,"ftol":1e-9,"maxcor":30,"maxls":100,"maxfun":1e7}) -> tuple[np.ndarray,scp.sparse.csc_array|None,scp.sparse.csc_array|scp.sparse.linalg.LinearOperator,scp.sparse.linalg.LinearOperator|None,float,list[float],float,list[LambdaTerm],Fit_info]:
   """Fits a general smooth model. Essentially completes the steps discussed in sections 3.3 and 4 of the paper by Krause et al. (submitted).
   
   Based on steps outlined by Wood, Pya, & Säfken (2016). An even more general version of :func:``solve_gammlss_sparse`` that can use the L-qEFS update by Krause et al. (submitted) to estimate the coefficients and lambda parameters.
   The update requires only a function to compute the log-likelihood and a function to compute the gradient of said likelihood with respect to the coefficients. Alternatively full Newton can be used - requiring a function to compute the hessian as well.

   References:

      - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
      - Nocedal & Wright (2006). Numerical Optimization. Springer New York.
      - Krause et al. (submitted). The Mixed-Sparse-Smooth-Model Toolbox (MSSM): Efficient Estimation and Selection of Large Multi-Level Statistical Models. https://doi.org/10.48550/arXiv.2506.13132

   :param family: Model family
   :type family: GSMMFamily
   :param ys: List of observation vectors
   :type ys: list[np.ndarray]
   :param Xs: List of model matrices
   :type Xs: list[scp.sparse.csc_array]
   :param form_n_coef: List of number of coefficients per formula
   :type form_n_coef: list[int]
   :param form_up_coef: List of un-penalized number of coefficients per formula
   :type form_up_coef: list[int]
   :param coef: Coefficient estimate
   :type coef: np.ndarray
   :param coef_split_idx: The indices at which to split the overall coefficient vector into separate lists - one per parameter.
   :type coef_split_idx: list[int]
   :param smooth_pen: List of penalties
   :type smooth_pen: list[LambdaTerm]
   :param max_outer: Maximum number of outer iterations, defaults to 50
   :type max_outer: int, optional
   :param max_inner: Maximum number of inner iterations, defaults to 50
   :type max_inner: int, optional
   :param min_inner: Minimum number of inner iterations, defaults to 50
   :type min_inner: int, optional
   :param conv_tol: Convergence tolerance, defaults to 1e-7
   :type conv_tol: float, optional
   :param extend_lambda: Whether lambda proposals should be accelerated or not. Can lower the number of new smoothing penalty proposals necessary, defaults to True
   :type extend_lambda: bool, optional
   :param extension_method_lam: Which method to use to extend lambda proposals, defaults to "nesterov2"
   :type extension_method_lam: str, optional
   :param control_lambda: Whether lambda proposals should be checked (and if necessary decreased) for whether or not they (approxiately) increase the Laplace approximate restricted maximum likelihood of the model. For ``method != 'qEFS'`` the following options are available: setting this to 0 disables control. Setting it to 1 means the step will never be smaller than the original EFS update but extensions will be removed in case the objective was exceeded (only has an effect when setting ``extend_lambda=True``). Setting it to 2 means that steps will generally be halved when they fail to increase the aproximate REML criterion. For ``method=='qEFS'`` the following options are available: setting this to 0 disables control. Setting it to 1 means the check described by Krause et al. (submitted) will be performed to control updates to lambda. Setting it to 2 means that steps will generally be halved when they fail to increase the aproximate REML criterion (note, that the gradient is based on quasi-newton approximations as well and thus less accurate). Setting it to 3 means both checks (i.e., 1 and 2) are performed, defaults to 1
   :type control_lambda: int, optional
   :param optimizer: Deprecated, defaults to "Newton"
   :type optimizer: str, optional
   :param method: Which method to use to estimate the coefficients (and lambda parameters), defaults to "Chol"
   :type method: str, optional
   :param check_cond: Whether to obtain an estimate of the condition number for the linear system that is solved. When ``check_cond=0``, no check will be performed. When ``check_cond=1``, an estimate of the condition number for the final system (at convergence) will be computed and warnings will be issued based on the outcome (see :func:`mssm.src.python.gamm_solvers.est_condition`), defaults to 1
   :type check_cond: int, optional
   :param piv_tol: Deprecated, defaults to 0.175
   :type piv_tol: float, optional
   :param repara: Whether to apply a stabilizing re-parameterization to the model, defaults to True
   :type repara: bool, optional
   :param should_keep_drop: If set to True, any coefficients that are dropped during fitting - are permanently excluded from all subsequent iterations, defaults to True
   :type should_keep_drop: bool, optional
   :param form_VH: Whether to explicitly form matrix ``V`` - the estimated inverse of the negative Hessian of the penalized likelihood - and ``H`` - the estimate of the Hessian of the log-likelihood - when using the ``qEFS`` method, defaults to True
   :type form_VH: bool, optional
   :param use_grad: Deprecated, defaults to False
   :type use_grad: bool, optional
   :param gamma: Setting this to a value larger than 1 promotes more complex (less smooth) models. Setting this to a value smaller than 1 (but must be > 0) promotes smoother models, defaults to 1
   :type gamma: float, optional
   :param qEFSH: Should the hessian approximation use a symmetric rank 1 update (``qEFSH='SR1'``) that is forced to result in positive semi-definiteness of the approximation or the standard bfgs update (``qEFSH='BFGS'``), defaults to 'SR1'
   :type qEFSH: str, optional
   :param overwrite_coef: Whether the initial coefficients passed to the optimization routine should be over-written by the solution obtained for the un-penalized version of the problem when ``method='qEFS'``, defaults to True
   :type overwrite_coef: bool, optional
   :param max_restarts: How often to shrink the coefficient estimate back to a random vector when convergence is reached and when ``method='qEFS'``. The optimizer might get stuck in local minima so it can be helpful to set this to 1-3. What happens is that if we converge, we shrink the coefficients back to a random vector and then continue optimizing once more, defaults to 0
   :type max_restarts: int, optional
   :param qEFS_init_converge: Whether to optimize the un-penalzied version of the model and to use the hessian (and optionally coefficients, if ``overwrite_coef=True``) to initialize the q-EFS solver. Ignored if ``method!='qEFS'``, defaults to True
   :type qEFS_init_converge: bool, optional
   :param prefit_grad: Whether to rely on Gradient Descent to improve the initial starting estimate for coefficients, defaults to False
   :type prefit_grad: bool, optional
   :param progress_bar: Whether progress should be printed or not, defaults to True
   :type progress_bar: bool, optional
   :param n_c: Number of cores to use, defaults to 10
   :type n_c: int, optional
   :param init_bfgs_options: An optional dictionary holding the same key:value pairs that can be passed to ``bfgs_options`` but pased to the optimizer of the un-penalized problem, defaults to {"gtol":1e-9,"ftol":1e-9,"maxcor":30,"maxls":100,"maxfun":1e7}
   :type init_bfgs_options: _type_, optional
   :param bfgs_options: An optional dictionary holding arguments that should be passed on to the call of :func:`scipy.optimize.minimize` if ``method=='qEFS'``, defaults to {"gtol":1e-9,"ftol":1e-9,"maxcor":30,"maxls":100,"maxfun":1e7}
   :type bfgs_options: _type_, optional
   :return: coef estimate, the negative hessian of the log-likelihood, inverse of cholesky of negative hessian of the penalized log-likelihood, if ``method=='qEFS'`` an instance of :class:`scp.sparse.linalg.LinearOperator` representing the new quasi-newton approximation, total edf, term-wise edfs, total penalty, final list of penalties, a :class:`Fit_info` object
   :rtype: tuple[np.ndarray, scp.sparse.csc_array|None, scp.sparse.csc_array|scp.sparse.linalg.LinearOperator, scp.sparse.linalg.LinearOperator|None, float, list[float], float, list[LambdaTerm], Fit_info]
   """

   # total number of coefficients
   n_coef = np.sum(form_n_coef)

   extend_by = initialize_extension(extension_method_lam,smooth_pen)
   was_extended = [False for _ in enumerate(smooth_pen)]

   # Build current penalties
   S_emb,S_pinv,_,FS_use_rank = compute_S_emb_pinv_det(n_coef,smooth_pen,"svd")

   # Build normalized penalty for rank checks (e.g., Wood, Pya & Saefken, 2016)
   keep_drop = None

   S_norm = copy.deepcopy(smooth_pen[0].S_J_emb) / scp.sparse.linalg.norm(smooth_pen[0].S_J_emb,ord=None)
   for peni in range(1,len(smooth_pen)):
      S_norm += smooth_pen[peni].S_J_emb / scp.sparse.linalg.norm(smooth_pen[peni].S_J_emb,ord=None)
   
   S_norm /= scp.sparse.linalg.norm(S_norm,ord=None)

   # Compute penalized likelihood for current estimate
   c_llk = family.llk(coef,coef_split_idx,ys,Xs)
   c_pen_llk = c_llk - 0.5*coef.T@S_emb@coef

   __neg_pen_llk = None
   __neg_pen_grad = None
   if method == "qEFS":
      # Define negative penalized likelihood function to be minimized via BFGS
      # plus function to evaluate negative gradient of penalized likelihood.
      def __neg_pen_llk(coef,coef_split_idx,ys,Xs,family,S_emb):
         coef = coef.reshape(-1,1)
         neg_llk = -1 * family.llk(coef,coef_split_idx,ys,Xs)
         return neg_llk + 0.5*coef.T@S_emb@coef
      
      def __neg_pen_grad(coef,coef_split_idx,ys,Xs,family,S_emb):
         # see Wood, Pya & Saefken (2016)
         coef = coef.reshape(-1,1)
         grad = family.gradient(coef,coef_split_idx,ys,Xs)
         pgrad = np.array([grad[i] - (S_emb[[i],:]@coef)[0] for i in range(len(grad))])
         return -1*pgrad.flatten()

    # Try improving start estimate via Gradient only
   if prefit_grad:
      # Re-parameterize
      if repara:
         coef_rp,Xs_rp,_,S_emb_rp,S_norm_rp,_,_,_,Qs = reparam_model(form_n_coef, form_up_coef, coef, coef_split_idx, Xs,
                                                                                            smooth_pen, form_inverse=False,
                                                                                            form_root=False, form_balanced=False, n_c=n_c)
      else:
         coef_rp = coef
         Xs_rp = Xs
         S_emb_rp = S_emb
         S_norm_rp = S_norm

      coef,_,_,_,c_llk,c_pen_llk,_,_,_ = update_coef_gen_smooth(family,ys,Xs_rp,coef_rp,
                                                                  coef_split_idx,S_emb_rp,
                                                                  S_norm_rp,None,None,None,
                                                                  c_llk,0,max_inner,
                                                                  min_inner,conv_tol,
                                                                  "Grad",piv_tol,None,
                                                                  None)
      
      if repara:
         split_coef = np.split(coef,coef_split_idx)

         # Transform coef
         for qi,Q in enumerate(Qs):
            split_coef[qi] = Q@split_coef[qi]
         coef = np.concatenate(split_coef).reshape(-1,1)

         
   iterator = range(max_outer)
   if progress_bar:
      iterator = tqdm(iterator,desc="Fitting",leave=True)
        
   fit_info = Fit_info()
   fit_info.eps = 0
   __old_opt = None
   one_update_counter = 0
   restart_counter = 0
   lam_delta = []
   prev_llk_hist = []
   for outer in iterator:

      # 1) Update coef for given lambda
      # 2) Check lambda -> repeat 1 if necessary
      # 3) Propose new lambda
      coef,H,L,LV,V,c_llk,c_pen_llk,\
      __old_opt,keep,drop,S_emb,smooth_pen,\
      total_edf,term_edfs,lam_delta = correct_lambda_step_gen_smooth(family,ys,Xs,S_norm,n_coef,form_n_coef,form_up_coef,coef,
                                                                           coef_split_idx,smooth_pen,lam_delta,
                                                                           extend_by,was_extended,c_llk,fit_info,outer,
                                                                           max_inner,min_inner,conv_tol,gamma,method,qEFSH,overwrite_coef,
                                                                           qEFS_init_converge,optimizer,
                                                                           __old_opt,use_grad,__neg_pen_llk,__neg_pen_grad,
                                                                           piv_tol,keep_drop,extend_lambda,
                                                                           extension_method_lam,control_lambda,repara,n_c,
                                                                           init_bfgs_options,bfgs_options)
      
      # Monitor whether we keep changing lambda but not actually updating coefficients..
      if method == 'qEFS':
         #print(LV.nit)
         if LV.nit == 1:
            one_update_counter += 1
         else:
            one_update_counter = 0
         
         if one_update_counter > 3 and ((outer <= 10) or (restart_counter < max_restarts)):
            # Prevent getting stuck in local optimum (early on). Shrink coef towards
            # random vector that still results in valid llk
            coef, c_llk, c_pen_llk = restart_coef(coef,c_llk,c_pen_llk,n_coef,coef_split_idx,ys,Xs,S_emb,family,outer,restart_counter)
            restart_counter += 1
            #print(res_checks)
         
         elif one_update_counter > 3:
            # Force convergence
            #print("force conv.")
            prev_pen_llk = c_pen_llk
            prev_llk_hist[-1] = c_pen_llk[0,0]

      if drop is not None:
         if should_keep_drop:
            keep_drop = [keep,drop]

      # Check overall convergence
      fit_info.iter += 1
      if outer > 0:
         #print((np.abs(prev_pen_llk - c_pen_llk) - conv_tol*np.abs(c_pen_llk))[0,0], (np.abs(prev_pen_llk - c_pen_llk) < conv_tol*np.abs(c_pen_llk)),np.any([np.abs(prev_llk_hist[hi] - c_pen_llk[0,0])  < conv_tol*np.abs(c_pen_llk) for hi in range(len(prev_llk_hist))]),[np.abs(prev_llk_hist[hi] - c_pen_llk[0,0])  - conv_tol*np.abs(c_pen_llk) for hi in range(len(prev_llk_hist))])
         if progress_bar:
               iterator.set_description_str(desc="Fitting - Conv.: " + "{:.2e}".format((np.abs(prev_pen_llk - c_pen_llk) - conv_tol*np.abs(c_pen_llk))[0,0]), refresh=True)

         if np.any([np.abs(prev_llk_hist[hi] - c_pen_llk[0,0])  < conv_tol*np.abs(c_pen_llk) for hi in range(len(prev_llk_hist))]) or\
            ((drop is not None) and (np.max(np.abs([prev_lam[lti] - smooth_pen[lti].lam for lti in range(len(smooth_pen))])) < 0.01)) or\
            (outer > 100 and (np.max(np.abs(lam_delta)) < 0.01)):
               # Optionally trigger re-start for efs method. Shrink coef towards
               # random vector that still results in valid llk
               if method=="qEFS" and (restart_counter < max_restarts):
                  coef, c_llk, c_pen_llk = restart_coef(coef,c_llk,c_pen_llk,n_coef,coef_split_idx,ys,Xs,S_emb,family,outer,restart_counter)
                  restart_counter += 1
               else:
                  if progress_bar:
                     iterator.set_description_str(desc="Converged!", refresh=True)
                     iterator.close()
                  fit_info.code = 0
                  break
        
      # We need the penalized likelihood of the model at this point for convergence control (step 2 in Wood, 2017 based on step 4 in Wood, Goude, & Shaw, 2016)
      prev_pen_llk = c_pen_llk
      prev_llk_hist.append(prev_pen_llk[0,0])
      if len(prev_llk_hist) > 5:
         del prev_llk_hist[0]
      prev_llk = c_llk
      prev_coef = copy.deepcopy(coef)
      prev_lam = copy.deepcopy([lterm.lam for lterm in smooth_pen])

      #print("lambda before step:",[lterm.lam for lterm in smooth_pen])

      # And ultimately apply new proposed lambda step
      for lti,lTerm in enumerate(smooth_pen):
         lTerm.lam += lam_delta[lti]

      # At this point we have:
      # 1) Estimated coef given lambda
      # 2) Performed checks on lambda and re-estimated coefficients if necessary
      # 3) Proposed lambda updates
      # 4) Checked for convergence
      # 5) Applied the new update lambdas so that we can go to the next iteration

      #print("lambda after step:",[lterm.lam for lterm in smooth_pen])
        
   # Total penalty
   penalty = coef.T@S_emb@coef

   # Calculate actual term-specific edf
   term_edfs = calculate_term_edf(smooth_pen,term_edfs)
   
   LV_linop = None
   if method == "qEFS":
      LV_linop = LV
      # Optionally form last V + Chol explicitly during last iteration
      # when working with qEFS update
      if form_VH:
         
         # Get an approximation of the Hessian of the likelihood
         if LV.form == 'SR1':
            H = -1*computeHSR1(LV.sk,LV.yk,LV.rho,scp.sparse.identity(len(coef),format='csc')*LV.omega,omega=LV.omega,make_psd=True,make_pd=True)
         else:
            H = -1*computeH(LV.sk,LV.yk,LV.rho,scp.sparse.identity(len(coef),format='csc')*LV.omega)
         
         H = scp.sparse.csc_array(H)

         # Get Cholesky factor of inverse of penalized hessian (needed for CIs)
         pH = scp.sparse.csc_array((-1*H) + S_emb)
         Lp, Pr, _ = cpp_cholP(pH)
         P = compute_eigen_perm(Pr)
         LVp0 = compute_Linv(Lp,10)
         LV = apply_eigen_perm(Pr,LVp0)
         L = P.T@Lp

      else:
         LV = None
         H = None # Do not approximate H.
       
   if check_cond == 1 and (method != "qEFS" or form_VH):
      K2,_,_,Kcode = est_condition(L,LV,verbose=False)

      fit_info.K2 = K2

      if fit_info.code == 0: # Convergence was reached but Knumber might suggest instable system.
         fit_info.code = Kcode

      if Kcode > 0:
         warnings.warn(f"Condition number ({K2}) of matrix A, where A.T@A=H and H is the Hessian of the negative penalized likelihood, is larger than 1/sqrt(u), where u is half the machine precision. Call ``model.fit()`` with ``method='QR/Chol'``, but note that even then estimates are likely to be inaccurate.")

   return coef,H,LV,LV_linop,total_edf,term_edfs,penalty[0,0],smooth_pen,fit_info
