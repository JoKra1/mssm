import numpy as np
import scipy as scp
from .exp_fam import Family,Gaussian,est_scale,GAMLSSFamily,Identity,warnings
from .penalties import PenType,id_dist_pen,translate_sparse,dataclass
from .formula import build_sparse_matrix_from_formula,setup_cache,clear_cache,cpp_solvers,pd,Formula,mp,repeat,os,map_csc_to_eigen,map_csr_to_eigen,math,tqdm,sys,copy,embed_in_S_sparse,reparam
from functools import reduce
from multiprocessing import managers,shared_memory

CACHE_DIR = './.db'
SHOULD_CACHE = False
MP_SPLIT_SIZE = 2000

@dataclass
class Fit_info:
   """Holds information related to convergence (speed) for GAMMs, GAMMLSS, and GSMMs.

   :ivar int lambda_updates: The total number of lambda updates computed during estimation. Initialized with 0.
   :ivar int iter: The number of outer iterations (a single outer iteration can involve multiple lambda updates) completed during estimation. Initialized with 0.
   :ivar int code: Convergence status. Anything above 0 indicates that the model did not converge and estimates should be considered carefully. Initialized with 1.
   :ivar float eps: The fraction added to the last estimate of the negative Hessian of the penalized likelihood during GAMMLSS or GSMM estimation. If this is not 0 - the model should not be considered as converged, irrespective of what ``code`` indicates. This most likely implies that the model is not identifiable. Initialized with ``None`` and ignored for GAMM estimation.
   :ivar float K2: An estimate for the condition number of matrix ``A``, where ``A.T@A=H`` and ``H`` is the final estimate of the negative Hessian of the penalized likelihood. Only available if ``check_cond>0`` when ``model.fit()`` is called for any model (i.e., GAMM, GAMMLSS, GSMM). Initialized with ``None``.
   :ivar [int] dropped: The final set of coefficients dropped during GAMMLSS/GSMM estimation when using ``method="QR/Chol"`` or ``None`` in which case no coefficients were dropped. Initialized with 0.
   """
   lambda_updates:int=0
   iter:int=0
   code:int=1
   eps:float or None = None
   K2:float or None = None
   dropped:[int] or None = None

def cpp_dChol(R,A):
   return cpp_solvers.dCholdRho(*map_csr_to_eigen(R),*map_csr_to_eigen(A))

def cpp_chol(A):
   return cpp_solvers.chol(*map_csc_to_eigen(A))

def cpp_cholP(A):
   return cpp_solvers.cholP(*map_csc_to_eigen(A))

def cpp_qr(A):
   return cpp_solvers.pqr(*map_csc_to_eigen(A))

def cpp_qrr(A):
   return cpp_solvers.pqrr(*map_csc_to_eigen(A))

def cpp_dqrr(A):
   return cpp_solvers.dpqrr(A)

def cpp_symqr(A,tol):
   return cpp_solvers.spqr(*map_csc_to_eigen(A),tol)

def cpp_solve_qr(A):
   return cpp_solvers.solve_pqr(*map_csc_to_eigen(A))

def cpp_solve_am(y,X,S):
   return cpp_solvers.solve_am(y,*map_csc_to_eigen(X),*map_csc_to_eigen(S))

def cpp_solve_coef(y,X,S):
   return cpp_solvers.solve_coef(y,*map_csc_to_eigen(X),*map_csc_to_eigen(S))

def cpp_solve_coef_pqr(y,X,E):
   return cpp_solvers.solve_coef_pqr(y,*map_csc_to_eigen(X),*map_csc_to_eigen(E))

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

def est_condition(L,Linv,seed=0,verbose=True):
   """Estimate the condition number ``K`` - the ratio of the largest to smallest singular values - of matrix ``A``, where ``A.T@A = L@L.T``.

   ``L`` and ``Linv`` can either be obtained by Cholesky decomposition, i.e., ``A.T@A = L@L.T`` or
   by QR decomposition ``A=Q@R`` where ``R=L.T``.

   If ``verbose=True`` (default), separate warnings will be issued in case ``K>(1/(0.5*sqrt(epsilon)))`` and ``K>(1/(0.5*epsilon))``.
   If the former warning is raised, this indicates that computing ``L`` via a Cholesky decomposition is likely unstable
   and should be avoided. If the second warning is raised as well, obtaining ``L`` via QR decomposition (of ``A``) is also likely
   to be unstable (see Golub & Van Loan, 2013).

   References:
     - Cline et al. (1979). An Estimate for the Condition Number of a Matrix.
     - Golub & Van Loan (2013). Matrix computations, 4th edition.

   :param L: Cholesky or any other root of ``A.T@A`` as a sparse matrix.
   :type L: scipy.sparse.csc_array
   :param Linv: Inverse of Choleksy (or any other root) of ``A.T@A``.
   :type Linv: scipy.sparse.csc_array
   :param seed: The seed to use for the random parts of the singular value decomposition. Defaults to 0.
   :type seed: int or None or numpy.random.Generator
   :param verbose: Whether or not warnings should be printed. Defaults to True.
   :type verbose: bool
   :return: A tuple, containing the estimate of condition number ``K``, an estimate of the largest singular value of ``A``, an estimate of the smallest singular value of ``A``, and a ``code``. The latter will be zero in case no warning was raised, 1 in case the first warning described above was raised, and 2 if the second warning was raised as well.
   :rtype: (float,float,float,int)
   """

   # Get unit round-off (Golub & Van Loan, 2013)
   u = 0.5*np.finfo(float).eps

   # Now get estimates of largest and smallest singular values of A
   # from norms of L and Linv (Cline et al. 1979)
   try:
      min_sing = scp.sparse.linalg.svds(Linv,k=1,return_singular_vectors=False,random_state=seed)[0]
      max_sing = scp.sparse.linalg.svds(L,k=1,return_singular_vectors=False,random_state=seed)[0]
   except:
      try:
         min_sing = scp.sparse.linalg.svds(Linv,k=1,return_singular_vectors=False,random_state=seed,solver='lobpcg')[0]
         max_sing = scp.sparse.linalg.svds(L,k=1,return_singular_vectors=False,random_state=seed,solver='lobpcg')[0]
      except:
         # Solver failed.. get out
         warnings.warn("Estimating the condition number of matrix A, where A.T@A=L.T@L failed. This can happen but might indicate that something is wrong. Consider estimates carefully!")
         return np.inf,np.inf,-np.inf,1
          
   K = max_sing*min_sing
   code = 0
   
   if K > 1/np.sqrt(u):
      if verbose:
         warnings.warn("Condition number of matrix A, where A.T@A=L.T@L, is larger than 1/sqrt(u), where u is half the machine precision.")
      code = 1
   
   if K > 1/u:
      if verbose:
         warnings.warn("Condition number of matrix A, where A.T@A=L.T@L, is larger than 1/u, where u is half the machine precision.")
      code = 2
   
   return K, max_sing, 1/min_sing, code

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

def compute_S_emb_pinv_det(col_S,penalties,pinv,root=False):
   # Computes final S multiplied with lambda
   # and the pseudo-inverse of this term. Optionally, the matrix
   # root of this term can be computed as well.
   S_emb = None

   # We need to compute the pseudo-inverse on the penalty block (so sum of all
   # penalties weighted by lambda) for every term so we first collect and sum
   # all term penalties together.
   SJs = [] # Summed SJs for every term
   DJs = [] # None if summation, otherwise embedded DJ*sqrt(\lambda) - root of corresponding SJ*\lambda
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

   # Optionally compute root of S_\lambda
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

def PIRLS_pdat_weights(y,mu,eta,family:Family):
   # Compute pseudo-data and weights for Penalized Reweighted Least Squares iteration (Wood, 2017, 6.1.1)
   # Calculation is based on a(mu) = 1, so reflects Fisher scoring!
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

def PIRLS_newton_weights(y,mu,eta,family:Family):
   # Compute newton pseudo-data and weights for Penalized Reweighted Least Squares iteration (Wood, 2017, 6.1.1 and 3.1.2)
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
      raise ValueError("Not a single observation provided information for Fisher weights.")

   z[invalid_idx] = np.nan # Make sure this is consistent, for scale computation

   return z, w, invalid_idx.flatten()

def update_PIRLS(y,yb,mu,eta,X,Xb,family):
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

def compute_block_B_shared(address_dat,address_ptr,address_idx,shape_dat,shape_ptr,rows,cols,nnz,T):
   BB = compute_block_linv_shared(address_dat,address_ptr,address_idx,shape_dat,shape_ptr,rows,cols,nnz,T)
   return BB.power(2).sum()

def compute_block_B_shared_cluster(address_dat,address_ptr,address_idx,shape_dat,shape_ptr,rows,cols,nnz,T,cluster_weights):
   BB = compute_block_linv_shared(address_dat,address_ptr,address_idx,shape_dat,shape_ptr,rows,cols,nnz,T)
   BBps = BB.power(2).sum()
   return np.sum(cluster_weights*BBps),len(cluster_weights)*BBps
   
def compute_B(L,P,lTerm,n_c=10,drop=None):
   # Solves L @ B = P @ D for B, parallelizing over column
   # blocks of D if int(D.shape[1]/2000) > 1

   # Also allows for approximate B computation for very big factor smooths.
   D_start = lTerm.start_index
   idx = np.arange(lTerm.S_J_emb.shape[1])
   if drop is None:
      drop = []
   keep = idx[np.isin(idx,drop)==False]
   
   if lTerm.clust_series is None:
      
      col_sums = lTerm.S_J.sum(axis=0)
      if lTerm.type == PenType.NULL and sum(col_sums[col_sums > 0]) == 1:
         # Null penalty for factor smooth has usually only non-zero element in a single colum,
         # so we only need to solve one linear system per level of the factor smooth.
         NULL_idx = np.argmax(col_sums)

         D_idx = np.arange(lTerm.start_index+NULL_idx,
                                lTerm.S_J.shape[1]*(lTerm.rep_sj+1),
                                lTerm.S_J.shape[1])

         D_len = len(D_idx)
         PD = P @ lTerm.D_J_emb[:,D_idx][keep,:]
      else:
         # First get columns associated to penalty
         D_len = lTerm.rep_sj * lTerm.S_J.shape[1]
         D_end = lTerm.start_index + D_len
         D_idx = idx[D_start:D_end]

         # Now check if dropped column is included, if so remove and update length.
         D_idx = D_idx[np.isin(D_idx,drop)==False]
         D_len = len(D_idx)
         PD = P @ lTerm.D_J_emb[:,D_idx][keep,:]

      D_r = int(D_len/2000)

      if D_r > 1 and n_c > 1:
         # Parallelize over column blocks of P @ D
         # Can speed up computations considerably and is feasible memory-wise
         # since L itself is super sparse.
         n_c = min(D_r,n_c)
         split = np.array_split(range(D_len),n_c)
         PD = P @ lTerm.D_J_emb[:,D_idx][keep,:]
         PDs = [PD[:,split[i]] for i in range(n_c)]

         with managers.SharedMemoryManager() as manager, mp.Pool(processes=n_c) as pool:
            # Create shared memory copies of data, indptr, and indices
            rows, cols, nnz, data, indptr, indices = map_csc_to_eigen(L)
            shape_dat = data.shape
            shape_ptr = indptr.shape

            dat_mem = manager.SharedMemory(data.nbytes)
            dat_shared = np.ndarray(shape_dat, dtype=np.double, buffer=dat_mem.buf)
            dat_shared[:] = data[:]

            ptr_mem = manager.SharedMemory(indptr.nbytes)
            ptr_shared = np.ndarray(shape_ptr, dtype=np.int64, buffer=ptr_mem.buf)
            ptr_shared[:] = indptr[:]

            idx_mem = manager.SharedMemory(indices.nbytes)
            idx_shared = np.ndarray(shape_dat, dtype=np.int64, buffer=idx_mem.buf)
            idx_shared[:] = indices[:]

            args = zip(repeat(dat_mem.name),repeat(ptr_mem.name),repeat(idx_mem.name),repeat(shape_dat),repeat(shape_ptr),repeat(rows),repeat(cols),repeat(nnz),PDs)
         
            pow_sums = pool.starmap(compute_block_B_shared,args)

         return sum(pow_sums)
      
      # Not worth parallelizing, solve directly
      B = cpp_solve_tr(L,PD)
      return B.power(2).sum()
   
   # Approximate the derivative based just on the columns in D_J that belong to the
   # maximum series identified for each cluster. Use the size of the cluster and the weights to
   # correct for the fact that all series in the cluster are slightly different after all.
   if len(drop) > 0:
      raise ValueError("Approximate derivative computation cannot currently handle unidentifiable terms.")

   n_coef = lTerm.S_J.shape[0]
   rank = int(lTerm.rank/lTerm.rep_sj)

   sum_bs_lw = 0
   sum_bs_up = 0

   targets = [P@lTerm.D_J_emb[:,(D_start + (s*n_coef)):(D_start + ((s+1)*n_coef) - (n_coef-rank))] for s in lTerm.clust_series]

   if len(targets) < 20*n_c:

      for weights,target in zip(lTerm.clust_weights,targets):

         BB = cpp_solve_tr(L,target)
         BBps = BB.power(2).sum()
         sum_bs_lw += np.sum(weights*BBps)
         sum_bs_up += len(weights)*BBps
   
   else:
      # Parallelize
      with managers.SharedMemoryManager() as manager, mp.Pool(processes=n_c) as pool:
         # Create shared memory copies of data, indptr, and indices
         rows, cols, nnz, data, indptr, indices = map_csc_to_eigen(L)
         shape_dat = data.shape
         shape_ptr = indptr.shape

         dat_mem = manager.SharedMemory(data.nbytes)
         dat_shared = np.ndarray(shape_dat, dtype=np.double, buffer=dat_mem.buf)
         dat_shared[:] = data[:]

         ptr_mem = manager.SharedMemory(indptr.nbytes)
         ptr_shared = np.ndarray(shape_ptr, dtype=np.int64, buffer=ptr_mem.buf)
         ptr_shared[:] = indptr[:]

         idx_mem = manager.SharedMemory(indices.nbytes)
         idx_shared = np.ndarray(shape_dat, dtype=np.int64, buffer=idx_mem.buf)
         idx_shared[:] = indices[:]

         args = zip(repeat(dat_mem.name),repeat(ptr_mem.name),repeat(idx_mem.name),
                    repeat(shape_dat),repeat(shape_ptr),repeat(rows),repeat(cols),
                    repeat(nnz),targets,lTerm.clust_weights)
         
         sum_bs_lw_all,sum_bs_up_all = zip(*pool.starmap(compute_block_B_shared_cluster,args))

         sum_bs_lw = np.sum(sum_bs_lw_all)
         sum_bs_up = np.sum(sum_bs_up_all)


   return sum_bs_lw, sum_bs_up

def compute_block_linv_shared(address_dat,address_ptr,address_idx,shape_dat,shape_ptr,rows,cols,nnz,T):
   dat_shared = shared_memory.SharedMemory(name=address_dat,create=False)
   ptr_shared = shared_memory.SharedMemory(name=address_ptr,create=False)
   idx_shared = shared_memory.SharedMemory(name=address_idx,create=False)

   data = np.ndarray(shape_dat,dtype=np.double,buffer=dat_shared.buf)
   indptr = np.ndarray(shape_ptr,dtype=np.int64,buffer=ptr_shared.buf)
   indices = np.ndarray(shape_dat,dtype=np.int64,buffer=idx_shared.buf)

   L = cpp_solvers.solve_tr(rows, cols, nnz, data, indptr, indices, T)

   return L

def compute_Linv(L,n_c=10):
   # Solves L @ inv(L) = I for inv(L) parallelizing over column
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
      Ts = [T[:,split[i]] for i in range(n_c)]

      with managers.SharedMemoryManager() as manager, mp.Pool(processes=n_c) as pool:
         # Create shared memory copies of data, indptr, and indices
         rows, cols, nnz, data, indptr, indices = map_csc_to_eigen(L)
         shape_dat = data.shape
         shape_ptr = indptr.shape

         dat_mem = manager.SharedMemory(data.nbytes)
         dat_shared = np.ndarray(shape_dat, dtype=np.double, buffer=dat_mem.buf)
         dat_shared[:] = data[:]

         ptr_mem = manager.SharedMemory(indptr.nbytes)
         ptr_shared = np.ndarray(shape_ptr, dtype=np.int64, buffer=ptr_mem.buf)
         ptr_shared[:] = indptr[:]

         idx_mem = manager.SharedMemory(indices.nbytes)
         idx_shared = np.ndarray(shape_dat, dtype=np.int64, buffer=idx_mem.buf)
         idx_shared[:] = indices[:]

         args = zip(repeat(dat_mem.name),repeat(ptr_mem.name),repeat(idx_mem.name),repeat(shape_dat),repeat(shape_ptr),repeat(rows),repeat(cols),repeat(nnz),Ts)
        
         LBinvs = pool.starmap(compute_block_linv_shared,args)
      
      return scp.sparse.hstack(LBinvs)

   return cpp_solve_tr(L,T)

def computeH_Brust(s,y,rho,H0):
   """Computes explicitly the negative Hessian of the penalized likelihood :math:`\mathbf{H}` from the L-BFGS-B optimizer info.

   Relies on equations 2.6 in Brust (2024).

   References:
    - Brust, J. J. (2024). Useful Compact Representations for Data-Fitting (arXiv:2403.12206). arXiv. https://doi.org/10.48550/arXiv.2403.12206
    - Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994). Representations of quasi-Newton matrices and their use in limited memory methods. Mathematical Programming, 63(1), 129–156. https://doi.org/10.1007/BF01582063

   :param s: List holding the first set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type s: [numpy.array]
   :param y: List holding the second set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type y: [numpy.array]
   :param rho: List holding element-wise ```1/y.T@s`` from Byrd, Nocdeal & Schnabel (1992).
   :type rho: [numpy.array]
   :param H0: Initial estimate for the hessian fo the negative penalized likelihood. Here some multiple of the identity (multiplied by ``omega``) plus the embedded penalty matrix.
   :type H0: scipy.sparse.csc_array
   :param make_psd: Whether to enforce PSD as mentioned in the description. By default set to False.
   :type make_psd: bool, optional
   :param omega: Multiple used to get initial estimate for the hessian fo the negative penalized likelihood. Defaults to 1.
   :type omega: float, optional
   """
   # Number of updates?
   m = len(y)
   S = np.array(s).T
   Y = np.array(y).T
   C = S

   # Compute D & R
   D = np.identity(m)
   D[0,0] *= np.dot(s[0],y[0])

   # Now use eq. 2.5 of Brust (2024) to compute R.
   # This is the same as in Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994) essentially.
   R0 = np.dot(s[0], y[0]).reshape(1,1)
   R = R0
   for k in range(1,m):
   
      D[k,k] *= np.dot(s[k],y[k])
      
      R = np.concatenate((np.concatenate((R0,S[:,:k].T@Y[:,[k]]),axis=1),
                          np.concatenate((np.zeros((1,R0.shape[1])),
                                             np.array([1/rho[k]]).reshape(1,1)),axis=1)),axis=0)
      
      R0 = R
   
   # Compute C.T@S and extract the upper triangular part from that matrix as shown by Brust (2024)
   CTS = C.T@S
   RCS = np.triu(CTS)

   # We now need inverse of RCS
   RCS_inv = scp.linalg.solve_triangular(RCS, np.identity(m),lower=False)

   # Can now form inverse of middle block from Brust (2024)
   t2inv = np.zeros((2*m,2*m))
   t2inv[:m,:m] = (-1*RCS_inv.T) @ (R + R.T - (D + S.T@H0@S)) @ RCS_inv # Upper left
   t2inv[:m,m:] = RCS_inv.T # Upper right
   t2inv[m:,:m] = RCS_inv # Lower left
   #t2inv[m:,m:] = 0 # Lower right remains empty

   t2 = np.zeros((2*m,2*m))
   t2[:m,m:] = RCS # Upper right
   t2[m:,:m] = RCS.T # Lower left
   t2[m:,m:] = (R + R.T - (D + S.T@H0@S)) # Lower right

   # Can now compute remaining terms to compute H as shown by Brust (2024)
   t1 = np.concatenate((C,Y - H0@S),axis=1)
   t3 = np.concatenate((C.T, (Y - H0@S).T),axis=0)

   H = H0 + t1@t2inv@t3
   
   return H

def computeH(s,y,rho,H0,make_psd=False,omega=1,explicit=True):
   """Computes explicitly the negative Hessian of the penalized likelihood :math:`\mathbf{H}` from the L-BFGS-B optimizer info.

   Relies on equations 2.16 in Byrd, Nocdeal & Schnabel (1992). Adapted here to work for the case where ``H0``:math:`=\mathbf{I}*\omega + \mathbf{S}_{\lambda}` and
   we need :math:`\mathbf{I}*\omega + \mathbf{U}\mathbf{D}\mathbf{U}^T` to be PSD. :math:`\mathbf{U}\mathbf{D}\mathbf{U}^T` is the update matrix for the
   negative Hessian of the penalized likelihood, **not** the inverse (:math:`\mathbf{V}`)! For this the implicit eigenvalue decomposition of Brust (2024) is used.

   References:
    - Brust, J. J. (2025). Useful Compact Representations for Data-Fitting (arXiv:2403.12206). arXiv. https://doi.org/10.48550/arXiv.2403.12206
    - Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994). Representations of quasi-Newton matrices and their use in limited memory methods. Mathematical Programming, 63(1), 129–156. https://doi.org/10.1007/BF01582063


   :param s: List holding the first set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type s: [numpy.array]
   :param y: List holding the second set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type y: [numpy.array]
   :param rho: List holding element-wise ```1/y.T@s`` from Byrd, Nocdeal & Schnabel (1992).
   :type rho: [numpy.array]
   :param H0: Initial estimate for the hessian fo the negative penalized likelihood. Here some multiple of the identity (multiplied by ``omega``) plus the embedded penalty matrix.
   :type H0: scipy.sparse.csc_array
   :param make_psd: Whether to enforce PSD as mentioned in the description. By default set to False.
   :type make_psd: bool, optional
   :param omega: Multiple used to get initial estimate for the hessian fo the negative penalized likelihood. Defaults to 1.
   :type omega: float, optional
   """
   # Number of updates?
   m = len(y)

   # First form S,Y, and D
   S = np.array(s).T
   Y = np.array(y).T
   
   STS = S.T@H0@S
   DK = np.identity(m)
   DK[0,0] *= np.dot(s[0],y[0])

   # Now use eq. 2.5 to compute R - only have to do this once
   R0 = np.dot(s[0], y[0]).reshape(1,1)
   R = R0
   for k in range(1,m):
   
      DK[k,k] *= np.dot(s[k],y[k])
      
      R = np.concatenate((np.concatenate((R0,S[:,:k].T@Y[:,[k]]),axis=1),
                          np.concatenate((np.zeros((1,R0.shape[1])),
                                             np.array([1/rho[k]]).reshape(1,1)),axis=1)),axis=0)
      
      R0 = R
   
   # Eq 2.22
   L = S.T@Y - R

   # Now compute term 2 in 3.13 of Byrd, Nocdeal & Schnabel (1992)
   t2 = np.zeros((2*m,2*m))
   t2[:m,:m] = STS
   t2[:m,m:] = L
   t2[m:,:m] = L.T
   t2[m:,m:] = -1*DK

   # We actually need the inverse to compute H

   # Eq 2.26 of Byrd, Nocdeal & Schnabel (1992)
   Dinv = copy.deepcopy(DK)
   Dpow = copy.deepcopy(DK)
   Dnpow = copy.deepcopy(DK)
   for k in range(m):
      Dinv[k,k] = 1/Dinv[k,k]
      Dpow[k,k] = np.power(Dpow[k,k],0.5)
      Dnpow[k,k] = np.power(Dnpow[k,k],-0.5)

   JJT = STS + L@Dinv@L.T
   J = scp.linalg.cholesky(JJT, lower=True)

   t2L = np.zeros((2*m,2*m))
   t2L[:m,:m] = Dpow
   t2L[m:,:m] = (-1*L)@Dnpow
   t2L[m:,m:] = J

   t2U = np.zeros((2*m,2*m))
   t2U[:m,:m] = -1*Dpow
   t2U[:m:,m:] = Dnpow@L.T
   t2U[m:,m:] = J.T

   t2_flip = t2L@t2U

   invt2L = scp.linalg.inv(t2L)
   invT2U = scp.linalg.inv(t2U)
   invt2 = invt2L.T@invT2U.T


   t2_sort = np.zeros((2*m,2*m))
   # top left <- bottom right
   t2_sort[:m,:m] = t2_flip[m:,m:]
   # top right <- bottom left
   t2_sort[:m,m:] = t2_flip[m:,:m]
   # bottom left <- top right
   t2_sort[m:,:m] = t2_flip[:m,m:]
   # bottom right <- top left
   t2_sort[m:,m:] = t2_flip[:m,:m]
   

   invt2_sort = np.zeros((2*m,2*m))
   # top left <- bottom right
   invt2_sort[:m,:m] = invt2[m:,m:]
   # top right <- bottom left
   invt2_sort[:m,m:] = invt2[m:,:m]
   # bottom left <- top right
   invt2_sort[m:,:m] = invt2[:m,m:]
   # bottom right <- top left
   invt2_sort[m:,m:] = invt2[:m,:m]


   # And terms 1 and 2
   t1 = np.concatenate((H0@S,Y),axis=1)
   t3 = np.concatenate((S.T@H0,Y.T),axis=0)


   # We have H0 + U@D@U.T with H0 = I*omega + S_emb and U@D@U.T=t1@(-t2)@t1.T
   # Now enforce that I*omega + t1@(-t2)@t1.T is psd
   if make_psd:
      correction = t1@(-1*invt2_sort)@t3

      # Compute implicit eigen decomposition as shown by Burst (2024)
      Q,R = scp.linalg.qr(t1,mode='economic')
      Rit2R = R@(-1*invt2_sort)@R.T

      # ev holds non-zero eigenvalues of U@D@U.T (e.g., Brust, 2024)
      ev, P = scp.linalg.eigh(Rit2R,driver='ev')

      # Now find closest PSD
      fix_idx = (ev + omega) <= 0
      if np.sum(fix_idx) > 0:
         ev[fix_idx] = (-1*omega)

         # Re-compute correction
         if explicit == False:
            return Q @ P,np.diag(ev),P.T @ Q.T
         
         correction = Q @ P @ np.diag(ev) @ P.T @ Q.T
         
      H = H0 + correction

   else:
      if explicit == False:
         return t1, (-1*invt2_sort), t3
      
      H = H0 + t1@(-1*invt2_sort)@t3
   
   return H

def computeV(s,y,rho,V0,explicit=True):
   """Computes, explicitly (or implicitly) the inverse of the negative Hessian of the penalized likelihood :math:`\mathbf{V}` from the L-BFGS-B optimizer info.

   Relies on equations 2.16 and 3.13 in Byrd, Nocdeal & Schnabel (1992).

   References:
    - Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994). Representations of quasi-Newton matrices and their use in limited memory methods. Mathematical Programming, 63(1), 129–156. https://doi.org/10.1007/BF01582063


   :param s: List holding the first set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type s: [numpy.array]
   :param y: List holding the second set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type y: [numpy.array]
   :param rho: List holding element-wise ```1/y.T@s`` from Byrd, Nocdeal & Schnabel (1992).
   :type rho: [numpy.array]
   :param V0: Initial estimate for the inverse of the hessian fo the negative penalized likelihood.
   :type V0: scipy.sparse.csc_array
   :param explicit: Whether or not to return the approximate matrix explicitly or implicitly in form of the three update vectors.
   :type explicit: bool
   """
   m = len(y)
   # First form S,Y, and D
   S = np.array(s).T
   Y = np.array(y).T
   
   DYTY = Y.T@V0@Y

   DYTY[0,0] += np.dot(s[0],y[0])

   # Now use eq. 2.5 to compute R^{-1} - only have to do this once
   Rinv0 = 1/np.dot(s[0], y[0]).reshape(1,1)
   Rinv = Rinv0
   for k in range(1,m):
   
      DYTY[k,k] += np.dot(s[k],y[k])
      
      Rinv = np.concatenate((np.concatenate((Rinv0,(-rho[k])*Rinv0@S[:,:k].T@Y[:,[k]]),axis=1),
                           np.concatenate((np.zeros((1,Rinv0.shape[1])),
                                             np.array([rho[k]]).reshape(1,1)),axis=1)),axis=0)
      
      Rinv0 = Rinv
   
   # Now compute term 2 in 3.13 used for all S_j
   t2 = np.zeros((2*m,2*m))
   t2[:m,:m] = Rinv.T@DYTY@Rinv
   t2[:m,m:] = -Rinv.T
   t2[m:,:m] = -Rinv

   # And terms 1 and 2
   t1 = np.concatenate((S,V0@Y),axis=1)
   t3 = np.concatenate((S.T,Y.T@V0),axis=0)

   if explicit:
      V = V0 + t1@t2@t3
      return V
   else:
      return t1, t2, t3
   
def computeVSR1(s,y,rho,V0,omega=1,make_psd=False,explicit=True):
   """Computes, explicitly (or implicitly) the symmetric rank one (SR1) approximation of the inverse of the negative Hessian of the penalized likelihood :math:`\mathbf{V}`.

   Can ensure positive (semi) definiteness of the approximation via an eigen decomposition as shown by Brust (2024). This is enforced via the ``make_psd`` argument.

   Relies on equations 2.16 and 3.13 in Byrd, Nocdeal & Schnabel (1992).

   References:
    - Brust, J. J. (2025). Useful Compact Representations for Data-Fitting (arXiv:2403.12206). arXiv. https://doi.org/10.48550/arXiv.2403.12206
    - Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994). Representations of quasi-Newton matrices and their use in limited memory methods. Mathematical Programming, 63(1), 129–156. https://doi.org/10.1007/BF01582063

   :param s: List holding the first set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type s: [numpy.array]
   :param y: List holding the second set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type y: [numpy.array]
   :param rho: List holding element-wise ```1/y.T@s`` from Byrd, Nocdeal & Schnabel (1992).
   :type rho: [numpy.array]
   :param V0: Initial estimate for the inverse of the hessian fo the negative penalized likelihood.
   :type V0: scipy.sparse.csc_array
   :param make_psd: Whether to enforce PSD as mentioned in the description. By default set to False.
   :type make_psd: bool, optional
   :param explicit: Whether or not to return the approximate matrix explicitly or implicitly in form of the three update vectors.
   :type explicit: bool
   """
   m = len(y)
   # First form S,Y, and D
   S = np.array(s).T
   Y = np.array(y).T
   
   YTY = Y.T@V0@Y
   DK = np.identity(m)
   DK[0,0] *= np.dot(s[0],y[0])

   # Now use eq. 2.5 to compute R - only have to do this once
   R0 = np.dot(s[0], y[0]).reshape(1,1)
   R = R0
   for k in range(1,m):
   
      DK[k,k] *= np.dot(s[k],y[k])
      
      R = np.concatenate((np.concatenate((R0,S[:,:k].T@Y[:,[k]]),axis=1),
                          np.concatenate((np.zeros((1,R0.shape[1])),
                                             np.array([1/rho[k]]).reshape(1,1)),axis=1)),axis=0)
      
      R0 = R
   
   # Eq 2.22
   L = S.T@Y - R
   
   # Now compute term 2 in eq. 5.2
   t2 = scp.linalg.inv(R + R.T - DK - YTY)

   # And terms 1 and 2
   t1 = S - V0@Y
   t3 = t1.T

   if make_psd:
      # Compute implicit eigen decomposition as shown by Brust (2024)
      Q,R = scp.linalg.qr(t1,mode='economic')
      Rit2R = R@(t2)@R.T

      # ev holds non-zero eigenvalues of U@D@U.T (e.g., Brust, 2024)
      ev, P = scp.linalg.eigh(Rit2R,driver='ev')
      
      # Now find closest PSD.
      fix_idx = (ev + omega) <= 0
      
      if np.sum(fix_idx) > 0:
         #print("fix VSR1",np.sum(fix_idx),omega,1/omega)
         ev[fix_idx] =  (-1*omega) #+ np.power(np.finfo(float).eps,0.9)

         #while np.any(np.abs(ev) < 1e-7):
         #   ev[np.abs(ev) < 1e-7] += np.power(np.finfo(float).eps,0.7)

         #print(f"implicit ev post shift. abs min: {np.min(np.abs(ev))}, min: {np.min(ev)}, min + 1: {np.min(ev+omega)}, max + 1: {np.max(ev+omega)}",ev[:10]+omega)

         # Now Q @ P @ np.diag(ev) @ P.T @ Q.T = shifted PSD version of U@D@U.T
         # so we can set:
         # shifted_invt2=np.diag(ev)
         # shifted_t2 = np.diag(1/ev)
         # t1 = Q @ P
         # t3 = t1.T = P.T @ Q.T
         t1 = Q @ P
         t3 = P.T@Q.T
         t2 = np.diag(ev)

   if explicit:
      V = V0 + t1@t2@t3
      return V
   else:
      return t1, t2, t3
   
def computeHSR1(s,y,rho,H0,omega=1,make_psd=False,make_pd=False,explicit=True):
   """Computes, explicitly (or implicitly) the symmetric rank one (SR1) approximation of the negative Hessian of the penalized likelihood :math:`\mathbf{H}`.

   Can ensure positive (semi) definiteness of the approximation via an eigen decomposition as shown by Brust (2024). This is enforced via the ``make_psd`` argument.

   Relies on equations 2.16 and 3.13 in Byrd, Nocdeal & Schnabel (1992).

   References:
    - Brust, J. J. (2025). Useful Compact Representations for Data-Fitting (arXiv:2403.12206). arXiv. https://doi.org/10.48550/arXiv.2403.12206
    - Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994). Representations of quasi-Newton matrices and their use in limited memory methods. Mathematical Programming, 63(1), 129–156. https://doi.org/10.1007/BF01582063

   :param s: List holding the first set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type s: [numpy.array]
   :param y: List holding the second set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type y: [numpy.array]
   :param rho: List holding element-wise ```1/y.T@s`` from Byrd, Nocdeal & Schnabel (1992).
   :type rho: [numpy.array]
   :param H0: Initial estimate for the hessian fo the negative penalized likelihood.
   :type H0: scipy.sparse.csc_array
   :param make_psd: Whether to enforce PSD as mentioned in the description. By default set to False.
   :type make_psd: bool, optional
   :param make_pd: Whether to enforce numeric positive definiteness, not just PSD. Ignored if ``make_psd=False``. By default set to False.
   :type make_pd: bool, optional
   :param explicit: Whether or not to return the approximate matrix explicitly or implicitly in form of the three update vectors.
   :type explicit: bool
   """
   m = len(y)
   # First form S,Y, and D
   S = np.array(s).T
   Y = np.array(y).T
   
   STS = S.T@H0@S
   DK = np.identity(m)
   DK[0,0] *= np.dot(s[0],y[0])

   # Now use eq. 2.5 to compute R - only have to do this once
   R0 = np.dot(s[0], y[0]).reshape(1,1)
   R = R0
   for k in range(1,m):
   
      DK[k,k] *= np.dot(s[k],y[k])
      
      R = np.concatenate((np.concatenate((R0,S[:,:k].T@Y[:,[k]]),axis=1),
                        np.concatenate((np.zeros((1,R0.shape[1])),
                                             np.array([1/rho[k]]).reshape(1,1)),axis=1)),axis=0)
      
      R0 = R
   
   # Eq 2.22
   L = S.T@Y - R
   
   # Now compute term 2 in eq. 5.2
   t2 = scp.linalg.inv(DK + L + L.T - STS)

   # And terms 1 and 2
   t1 = Y - H0@S
   t3 = t1.T

   if make_psd:
      # Compute implicit eigen decomposition as shown by Brust (2024)
      Q,R = scp.linalg.qr(t1,mode='economic')
      Rit2R = R@(t2)@R.T

      # ev holds non-zero eigenvalues of U@D@U.T (e.g., Brust, 2024)
      ev, P = scp.linalg.eigh(Rit2R,driver='ev')
      
      # Now find closest PSD.
      fix_idx = (ev + omega) <= 0
      
      if np.sum(fix_idx) > 0:
         #print("fix VSR1",np.sum(fix_idx),omega,1/omega)
         ev[fix_idx] =  (-1*omega) #+ np.power(np.finfo(float).eps,0.9)

         # Useful to guarantee that penalized hessian is pd at convergence
         if make_pd:
            ev[fix_idx] += np.power(np.finfo(float).eps,0.9)

         #while np.any(np.abs(ev) < 1e-7):
         #   ev[np.abs(ev) < 1e-7] += np.power(np.finfo(float).eps,0.7)

         #print(f"implicit ev post shift. abs min: {np.min(np.abs(ev))}, min: {np.min(ev)}, min + 1: {np.min(ev+omega)}, max + 1: {np.max(ev+omega)}",ev[:10]+omega)

         # Now Q @ P @ np.diag(ev) @ P.T @ Q.T = shifted PSD version of U@D@U.T
         # so we can set:
         # shifted_invt2=np.diag(ev)
         # shifted_t2 = np.diag(1/ev)
         # t1 = Q @ P
         # t3 = t1.T = P.T @ Q.T
         t1 = Q @ P
         t3 = P.T@Q.T
         t2 = np.diag(ev)

   if explicit:
      H = H0 + t1@t2@t3
      return H
   else:
      return t1, t2, t3

def compute_t1_shifted_t2_t3(s,y,rho,H0,omega=1,form='Byrd'):
   """Computes the compact update to get the inverse of the negative Hessian of the penalized likelihood :math:`\mathbf{V}` from the L-BFGS-B optimizer info.

   Relies on equations 2.16 and 3.13 in Byrd, Nocdeal & Schnabel (1992) or on 2.6 in Brust (2024). Adapted here to work for the case where ``H0``:math:`=\mathbf{I}*\omega + \mathbf{S}_{\lambda}` and
   we need :math:`\mathbf{I}*\omega + \mathbf{U}\mathbf{D}\mathbf{U}^T` to be PSD. :math:`\mathbf{U}\mathbf{D}\mathbf{U}^T` is the update matrix for the
   negative Hessian of the penalized likelihood, **not** the inverse (:math:`\mathbf{V}`)! For this the implicit eigenvalue decomposition of Brust (2024) is used.

   References:
    - Brust, J. J. (2025). Useful Compact Representations for Data-Fitting (arXiv:2403.12206). arXiv. https://doi.org/10.48550/arXiv.2403.12206
    - Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994). Representations of quasi-Newton matrices and their use in limited memory methods. Mathematical Programming, 63(1), 129–156. https://doi.org/10.1007/BF01582063

   :param s: List holding the first set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type s: [numpy.array]
   :param y: List holding the second set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type y: [numpy.array]
   :param rho: List holding element-wise ```1/y.T@s`` from Byrd, Nocdeal & Schnabel (1992).
   :type rho: [numpy.array]
   :param H0: Initial estimate for the hessian fo the negative penalized likelihood. Here some multiple of the identity (multiplied by ``omega``) plus the embedded penalty matrix.
   :type H0: scipy.sparse.csc_array
   :param omega: Multiple used to get initial estimate for the hessian fo the negative penalized likelihood. Defaults to 1.
   :type omega: float, optional
   :param form: Which compact form to compute - the one from Byrd et al. (1992) or the one from Brust (2024). Defaults to "Byrd".
   :type form: float, optional
   """

   # Number of updates?
   m = len(y)

   # Now form S,Y, and D
   S = np.array(s).T
   Y = np.array(y).T

   if form == "Byrd": # Compact representation from Byrd, Nocdeal & Schnabel (1992)
   
      STS = S.T@H0@S
      DK = np.identity(m)
      DK[0,0] *= np.dot(s[0],y[0])

      # Now use eq. 2.5 to compute R - only have to do this once
      R0 = np.dot(s[0], y[0]).reshape(1,1)
      R = R0
      for k in range(1,m):
      
         DK[k,k] *= np.dot(s[k],y[k])
         
         R = np.concatenate((np.concatenate((R0,S[:,:k].T@Y[:,[k]]),axis=1),
                           np.concatenate((np.zeros((1,R0.shape[1])),
                                                np.array([1/rho[k]]).reshape(1,1)),axis=1)),axis=0)
         
         R0 = R
      
      # Eq 2.22
      L = S.T@Y - R

      # Now compute term 2 in 3.13 of Byrd, Nocdeal & Schnabel (1992)
      t2 = np.zeros((2*m,2*m))
      t2[:m,:m] = STS
      t2[:m,m:] = L
      t2[m:,:m] = L.T
      t2[m:,m:] = -1*DK

      # We actually need the inverse to compute H

      # Eq 2.26 of Byrd, Nocdeal & Schnabel (1992)
      Dinv = copy.deepcopy(DK)
      Dpow = copy.deepcopy(DK)
      Dnpow = copy.deepcopy(DK)
      for k in range(m):
         Dinv[k,k] = 1/Dinv[k,k]
         Dpow[k,k] = np.power(Dpow[k,k],0.5)
         Dnpow[k,k] = np.power(Dnpow[k,k],-0.5)

      JJT = STS + L@Dinv@L.T
      J = scp.linalg.cholesky(JJT, lower=True)

      t2L = np.zeros((2*m,2*m))
      t2L[:m,:m] = Dpow
      t2L[m:,:m] = (-1*L)@Dnpow
      t2L[m:,m:] = J

      t2U = np.zeros((2*m,2*m))
      t2U[:m,:m] = -1*Dpow
      t2U[:m:,m:] = Dnpow@L.T
      t2U[m:,m:] = J.T

      t2_flip = t2L@t2U

      invt2L = scp.linalg.inv(t2L)
      invT2U = scp.linalg.inv(t2U)
      invt2 = invt2L.T@invT2U.T

      t2_sort = np.zeros((2*m,2*m))
      # top left <- bottom right
      t2_sort[:m,:m] = t2_flip[m:,m:]
      # top right <- bottom left
      t2_sort[:m,m:] = t2_flip[m:,:m]
      # bottom left <- top right
      t2_sort[m:,:m] = t2_flip[:m,m:]
      # bottom right <- top left
      t2_sort[m:,m:] = t2_flip[:m,:m]

      invt2_sort = np.zeros((2*m,2*m))
      # top left <- bottom right
      invt2_sort[:m,:m] = invt2[m:,m:]
      # top right <- bottom left
      invt2_sort[:m,m:] = invt2[m:,:m]
      # bottom left <- top right
      invt2_sort[m:,:m] = invt2[:m,m:]
      # bottom right <- top left
      invt2_sort[m:,m:] = invt2[:m,:m]

      # And t1 and t3
      t1 = np.concatenate((H0@S,Y),axis=1)
      t3 = np.concatenate((S.T@H0,Y.T),axis=0)

      shifted_invt2 = -1*invt2_sort
      shifted_t2 = -1*t2_sort
   elif form == "ByrdSR1":
      # SR1 approximation      
      STS = S.T@H0@S
      DK = np.identity(m)
      DK[0,0] *= np.dot(s[0],y[0])

      # Now use eq. 2.5 to compute R - only have to do this once
      R0 = np.dot(s[0], y[0]).reshape(1,1)
      R = R0
      for k in range(1,m):
      
         DK[k,k] *= np.dot(s[k],y[k])
         
         R = np.concatenate((np.concatenate((R0,S[:,:k].T@Y[:,[k]]),axis=1),
                           np.concatenate((np.zeros((1,R0.shape[1])),
                                                np.array([1/rho[k]]).reshape(1,1)),axis=1)),axis=0)
         
         R0 = R
      
      # Eq 2.22
      L = S.T@Y - R
      
      # Now compute term 2 in eq. 5.2
      shifted_t2 = DK + L + L.T - STS
      shifted_invt2 = scp.linalg.inv(shifted_t2)

      # And terms 1 and 2
      t1 = Y - H0@S
      t3 = t1.T
   else: #  Brust (2024)
      C = S

      # Compute D & R
      D = np.identity(m)
      D[0,0] *= np.dot(s[0],y[0])

      # Now use eq. 2.5 of Brust (2024) to compute R.
      # This is the same as in Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994) essentially.
      R0 = np.dot(s[0], y[0]).reshape(1,1)
      R = R0
      for k in range(1,m):
      
         D[k,k] *= np.dot(s[k],y[k])
         
         R = np.concatenate((np.concatenate((R0,S[:,:k].T@Y[:,[k]]),axis=1),
                           np.concatenate((np.zeros((1,R0.shape[1])),
                                                np.array([1/rho[k]]).reshape(1,1)),axis=1)),axis=0)
         
         R0 = R
      
      # Compute C.T@S and extract the upper triangular part from that matrix as shown by Brust (2024)
      CTS = C.T@S
      RCS = np.triu(CTS)

      # We now need inverse of RCS
      RCS_inv = scp.linalg.solve_triangular(RCS, np.identity(m),lower=False)

      # Can now form inverse of middle block from Brust (2024)
      t2inv = np.zeros((2*m,2*m))
      t2inv[:m,:m] = (-1*RCS_inv.T) @ (R + R.T - (D + S.T@H0@S)) @ RCS_inv # Upper left
      t2inv[:m,m:] = RCS_inv.T # Upper right
      t2inv[m:,:m] = RCS_inv # Lower left
      #t2inv[m:,m:] = 0 # Lower right remains empty

      t2 = np.zeros((2*m,2*m))
      t2[:m,m:] = RCS # Upper right
      t2[m:,:m] = RCS.T # Lower left
      t2[m:,m:] = (R + R.T - (D + S.T@H0@S)) # Lower right

      # Can now compute remaining terms to compute H as shown by Brust (2024)
      t1 = np.concatenate((C,Y - H0@S),axis=1)
      t3 = np.concatenate((C.T, (Y - H0@S).T),axis=0)

      # H = H0 + t1@t2inv@t3
      shifted_invt2 = t2inv
      shifted_t2 = t2


   # We have H0 + U@D@U.T with H0 = I*omega + S_emb and U@D@U.T=t1@(-t2)@t1.T
   # Now enforce that I*omega + t1@(-t2)@t1.T is psd

   # Compute implicit eigen decomposition as shown by Brust (2024)
   Q,R = scp.linalg.qr(t1,mode='economic')
   Rit2R = R@(shifted_invt2)@R.T

   # ev holds non-zero eigenvalues of U@D@U.T (e.g., Brust, 2024)
   ev, P = scp.linalg.eigh(Rit2R,driver='ev')
   
   # Now find closest PSD.
   fix_idx = (ev + omega) <= 0
   
   if np.sum(fix_idx) > 0:
      #print("fix",np.sum(fix_idx),omega)
      ev[fix_idx] =  (-1*omega)
      
      if form != "ByrdSR1":
         ev[fix_idx] += np.power(np.finfo(float).eps,0.9)

         while np.any(np.abs(ev) < 1e-7):
            ev[np.abs(ev) < 1e-7] += np.power(np.finfo(float).eps,0.7)

      #print(f"implicit ev post shift. abs min: {np.min(np.abs(ev))}, min: {np.min(ev)}, min + 1: {np.min(ev+omega)}, max + 1: {np.max(ev+omega)}",ev[:10]+omega)

      # Now Q @ P @ np.diag(ev) @ P.T @ Q.T = shifted PSD version of U@D@U.T
      # so we can set:
      # shifted_invt2=np.diag(ev)
      # shifted_t2 = np.diag(1/ev)
      # t1 = Q @ P
      # t3 = t1.T = P.T @ Q.T
      t1 = Q @ P
      t3 = P.T@Q.T
      shifted_invt2=np.diag(ev)
      if form != "ByrdSR1":
         shifted_t2 = np.diag(1/ev)
      else:
         shifted_t2 = np.diag([1/evi if np.abs(evi) != 0 else 0 for evi in ev])

   return t1, shifted_t2, shifted_invt2, t3, 0


def compute_H_adjust_ev(s,y,rho,H0,omega=1,form='Byrd'):
   """Computes the non-zero eigenvalues of the update to get the negative Hessian of the penalized likelihood :math:`\mathcal{H}` from the L-BFGS-B optimizer info.

   Relies on equations 2.16 and 3.13 in Byrd, Nocdeal & Schnabel (1992) or on 2.6 in Brust (2024). Adapted here to work for the case where ``H0``:math:`=\mathbf{I}*\omega + \mathbf{S}_{\lambda}` and
   we need the eigenvalues for :math:`\mathbf{I}*\omega + \mathbf{U}\mathbf{D}\mathbf{U}^T`. To get those, we simply add ``omega`` to the evs of :math:`\mathbf{U}\mathbf{D}\mathbf{U}^T`

   References:
    - Brust, J. J. (2025). Useful Compact Representations for Data-Fitting (arXiv:2403.12206). arXiv. https://doi.org/10.48550/arXiv.2403.12206
    - Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994). Representations of quasi-Newton matrices and their use in limited memory methods. Mathematical Programming, 63(1), 129–156. https://doi.org/10.1007/BF01582063

   :param s: List holding the first set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type s: [numpy.array]
   :param y: List holding the second set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type y: [numpy.array]
   :param rho: List holding element-wise ```1/y.T@s`` from Byrd, Nocdeal & Schnabel (1992).
   :type rho: [numpy.array]
   :param H0: Initial estimate for the hessian fo the negative penalized likelihood. Here some multiple of the identity (multiplied by ``omega``) plus the embedded penalty matrix.
   :type H0: scipy.sparse.csc_array
   :param omega: Multiple used to get initial estimate for the hessian fo the negative penalized likelihood. Defaults to 1.
   :type omega: float, optional
   :param form: Which compact form to compute - the one from Byrd et al. (1992) or the one from Brust (2024). Defaults to "Byrd".
   :type form: float, optional
   """

   # Number of updates?
   m = len(y)

   # Now form S,Y, and D
   S = np.array(s).T
   Y = np.array(y).T

   if form == "Byrd": # Compact representation from Byrd, Nocdeal & Schnabel (1992)
   
      STS = S.T@H0@S
      DK = np.identity(m)
      DK[0,0] *= np.dot(s[0],y[0])

      # Now use eq. 2.5 to compute R - only have to do this once
      R0 = np.dot(s[0], y[0]).reshape(1,1)
      R = R0
      for k in range(1,m):
      
         DK[k,k] *= np.dot(s[k],y[k])
         
         R = np.concatenate((np.concatenate((R0,S[:,:k].T@Y[:,[k]]),axis=1),
                           np.concatenate((np.zeros((1,R0.shape[1])),
                                                np.array([1/rho[k]]).reshape(1,1)),axis=1)),axis=0)
         
         R0 = R
      
      # Eq 2.22
      L = S.T@Y - R

      # Now compute term 2 in 3.13 of Byrd, Nocdeal & Schnabel (1992)
      t2 = np.zeros((2*m,2*m))
      t2[:m,:m] = STS
      t2[:m,m:] = L
      t2[m:,:m] = L.T
      t2[m:,m:] = -1*DK

      # We actually need the inverse to compute H

      # Eq 2.26 of Byrd, Nocdeal & Schnabel (1992)
      Dinv = copy.deepcopy(DK)
      Dpow = copy.deepcopy(DK)
      Dnpow = copy.deepcopy(DK)
      for k in range(m):
         Dinv[k,k] = 1/Dinv[k,k]
         Dpow[k,k] = np.power(Dpow[k,k],0.5)
         Dnpow[k,k] = np.power(Dnpow[k,k],-0.5)

      JJT = STS + L@Dinv@L.T
      J = scp.linalg.cholesky(JJT, lower=True)

      t2L = np.zeros((2*m,2*m))
      t2L[:m,:m] = Dpow
      t2L[m:,:m] = (-1*L)@Dnpow
      t2L[m:,m:] = J

      t2U = np.zeros((2*m,2*m))
      t2U[:m,:m] = -1*Dpow
      t2U[:m:,m:] = Dnpow@L.T
      t2U[m:,m:] = J.T

      t2_flip = t2L@t2U

      invt2L = scp.linalg.inv(t2L)
      invT2U = scp.linalg.inv(t2U)
      invt2 = invt2L.T@invT2U.T

      t2_sort = np.zeros((2*m,2*m))
      # top left <- bottom right
      t2_sort[:m,:m] = t2_flip[m:,m:]
      # top right <- bottom left
      t2_sort[:m,m:] = t2_flip[m:,:m]
      # bottom left <- top right
      t2_sort[m:,:m] = t2_flip[:m,m:]
      # bottom right <- top left
      t2_sort[m:,m:] = t2_flip[:m,:m]

      invt2_sort = np.zeros((2*m,2*m))
      # top left <- bottom right
      invt2_sort[:m,:m] = invt2[m:,m:]
      # top right <- bottom left
      invt2_sort[:m,m:] = invt2[m:,:m]
      # bottom left <- top right
      invt2_sort[m:,:m] = invt2[:m,m:]
      # bottom right <- top left
      invt2_sort[m:,m:] = invt2[:m,:m]

      # And t1 and t3
      t1 = np.concatenate((H0@S,Y),axis=1)
      t3 = np.concatenate((S.T@H0,Y.T),axis=0)

      shifted_invt2 = -1*invt2_sort
      shifted_t2 = -1*t2_sort
   else: #  Brust (2024)
      C = S

      # Compute D & R
      D = np.identity(m)
      D[0,0] *= np.dot(s[0],y[0])

      # Now use eq. 2.5 of Brust (2024) to compute R.
      # This is the same as in Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994) essentially.
      R0 = np.dot(s[0], y[0]).reshape(1,1)
      R = R0
      for k in range(1,m):
      
         D[k,k] *= np.dot(s[k],y[k])
         
         R = np.concatenate((np.concatenate((R0,S[:,:k].T@Y[:,[k]]),axis=1),
                           np.concatenate((np.zeros((1,R0.shape[1])),
                                                np.array([1/rho[k]]).reshape(1,1)),axis=1)),axis=0)
         
         R0 = R
      
      # Compute C.T@S and extract the upper triangular part from that matrix as shown by Brust (2024)
      CTS = C.T@S
      RCS = np.triu(CTS)

      # We now need inverse of RCS
      RCS_inv = scp.linalg.solve_triangular(RCS, np.identity(m),lower=False)

      # Can now form inverse of middle block from Brust (2024)
      t2inv = np.zeros((2*m,2*m))
      t2inv[:m,:m] = (-1*RCS_inv.T) @ (R + R.T - (D + S.T@H0@S)) @ RCS_inv # Upper left
      t2inv[:m,m:] = RCS_inv.T # Upper right
      t2inv[m:,:m] = RCS_inv # Lower left
      #t2inv[m:,m:] = 0 # Lower right remains empty

      t2 = np.zeros((2*m,2*m))
      t2[:m,m:] = RCS # Upper right
      t2[m:,:m] = RCS.T # Lower left
      t2[m:,m:] = (R + R.T - (D + S.T@H0@S)) # Lower right

      # Can now compute remaining terms to compute H as shown by Brust (2024)
      t1 = np.concatenate((C,Y - H0@S),axis=1)
      t3 = np.concatenate((C.T, (Y - H0@S).T),axis=0)

      # H = H0 + t1@t2inv@t3
      shifted_invt2 = t2inv
      shifted_t2 = t2


   # We have H0 + U@D@U.T with H0 = I*omega + S_emb and U@D@U.T=t1@(-t2)@t1.T
   # Now enforce that I*omega + t1@(-t2)@t1.T is psd

   # Compute implicit eigen decomposition as shown by Brust (2024)
   Q,R = scp.linalg.qr(t1,mode='economic')
   Rit2R = R@(shifted_invt2)@R.T

   # ev holds non-zero eigenvalues of U@D@U.T (e.g., Brust, 2024)
   ev, P = scp.linalg.eigh(Rit2R,driver='ev')
   
   return ev + omega

def computetrVS3(t1,t2,t3,lTerm,V0):
   """Compute ``tr(V@lTerm.S_j)`` from linear operator of ``V`` obtained from L-BFGS-B optimizer.

   Relies on equation 3.13 in Byrd, Nocdeal & Schnabel (1992). Adapted to ensure positive semi-definitiness required
   by EFS update.

   References:
    - Brust, J. J. (2025). Useful Compact Representations for Data-Fitting (arXiv:2403.12206). arXiv. https://doi.org/10.48550/arXiv.2403.12206

   :param t1: ``nCoef*2m``  matrix from Byrd, Nocdeal & Schnabel (1992). If ``t2 is None``, then ``V`` is treated like an identity matrix.
   :type t1: numpy.array or None
   :param t2: ``2m*2m``  matrix from Byrd, Nocdeal & Schnabel (1992). If ``t2 is None``, then ``V`` is treated like an identity matrix.
   :type t2: numpy.array or None
   :param t3: ``2m*nCoef``  matrix from Byrd, Nocdeal & Schnabel (1992). If ``t2 is None``, then ``t1`` is treated like an identity matrix.
   :type t3: numpy.array or None
   :param lTerm: Current lambda term for which to compute the trace.
   :type lTerm: mssm.src.python.penalties.LambdaTerm
   :param V0: Initial estimate for the inverse of the hessian fo the negative penalized likelihood.
   :type V0: scipy.sparse.csc_array
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

def computetrVS2(t2,S,Y,lTerm):
   """Compute ``tr(V@lTerm.S_j)`` from linear operator of ``V`` obtained from L-BFGS-B optimizer.

   Relies on equation 3.13 in Byrd, Nocdeal & Schnabel (1992).

   :param t2: ``2m*2m``  matrix from Byrd, Nocdeal & Schnabel (1992). If ``t2 is None``, then ``V`` is treated like an identity matrix.
   :type t2: numpy.array or None
   :param S: ``n*k`` matrix holding first set of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type S: numpy.array
   :param Y: ``n*k`` matrix holding second set of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type Y: numpy.array
   :param lTerm: Current lambda term for which to compute the trace.
   :type lTerm: mssm.src.python.penalties.LambdaTerm
   """

   tr = 0

   S_start = lTerm.start_index
   S_len = lTerm.rep_sj * lTerm.S_J.shape[1]
   S_end = S_start + S_len
   
   for cidx in range(S_start,S_end):
      S_c = lTerm.S_J_emb[:,[cidx]] # Can remain sparse
      
      # Now compute remaining terms 1 and 2 for equation 3.13
      if not t2 is None:
         t1 = np.concatenate((S[[cidx],:],Y[[cidx],:]),axis=1)
         t3 = np.concatenate((S.T@S_c,Y.T@S_c),axis=0)

         VS_c = lTerm.S_J_emb[[cidx],[cidx]] + t1@t2@t3
      else:
         VS_c = np.array([lTerm.S_J_emb[[cidx],[cidx]]])

      tr += VS_c[0,0]

   return tr


def computetrVS(V,lTerm):
   """Compute ``tr(V@lTerm.S_j)`` from linear operator of ``V`` obtained from L-BFGS-B optimizer.

   Relies on original "dual loop" strategy from Nocedal (1980). Not as efficient as ``computetrVS``.

   :param V: Linear operator of ``V``, which is the current estimate for the inverse of the negative Hessian of the penalized likelihood.
   :type V: scipy.sparse.linalg.LinearOperator
   :param lTerm: Current lambda term for which to compute the trace.
   :type lTerm: mssm.src.python.penalties.LambdaTerm
   """
   tr = 0

   S_start = lTerm.start_index
   S_len = lTerm.rep_sj * lTerm.S_J.shape[1]
   S_end = S_start + S_len
   
   for cidx in range(S_start,S_end):
      S_c = lTerm.S_J_emb[:,[cidx]].toarray()
      VS_c = V.matvec(S_c)
      tr += VS_c[cidx,0]
   
   return tr
   

def calculate_edf(LP,Pr,InvCholXXS,penalties,lgdetDs,colsX,n_c,drop):
   # Follows steps outlined by Wood & Fasiolo (2017) to compute total degrees of freedom by the model.
   # Generates the B matrix also required for the derivative of the log-determinant of X.T@X+S_\lambda. This
   # is either done exactly - as described by Wood & Fasiolo (2017) - or approximately. The latter is much faster
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
      S_emb,_,_,_ = compute_S_emb_pinv_det(colsX,penalties,"svd")

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

         nt1,nt2,int2,nt3,_ = compute_t1_shifted_t2_t3(s,y,rho,H0,omega,"ByrdSR1" if form == 'SR1' else "Byrd")

         # Compute inverse:
         if form != 'SR1':
            invt2 = nt2 + nt3@V0@nt1

            U,sv_invt2,VT = scp.linalg.svd(invt2,lapack_driver='gesvd')

            # Nowe we can compute all parts for the Woodbury identy to obtain V
            t2 = VT.T @ np.diag(1/sv_invt2)  @  U.T

            t1 = V0@nt1
            t3 = nt3@V0
         else:
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

def update_scale_edf(y,z,eta,Wr,rowsX,colsX,LP,InvCholXXSP,Pr,lgdetDs,family,penalties,keep,drop,n_c):
   # Updates the scale of the model. For this the edf
   # are computed as well - they are returned because they are needed for the
   # lambda step proposal anyway.
   
   # Calculate Pearson residuals for GAMM (Wood, 3.1.5 & 3.1.7)
   # Standard residuals for AMM
   dropped = 0
   if isinstance(family,Gaussian) == False or isinstance(family.link,Identity) == False:
      inval = np.isnan(z)
      dropped = np.sum(inval) # Make sure to only take valid z/eta/w here and for computing the scale
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
      total_edf, term_edfs, Bs = calculate_edf(LP,Pr,InvCholXXS,penalties,lgdetDs,colsX-(len(drop) if drop is not None else 0),n_c,drop)
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

def update_coef(yb,X,Xb,family,S_emb,S_root,n_c,formula,offset):
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

def update_coef_and_scale(y,yb,z,Wr,rowsX,colsX,X,Xb,family,S_emb,S_root,S_pinv,FS_use_rank,penalties,n_c,formula,form_Linv,offset):
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
   wres,InvCholXXS,total_edf,term_edfs,Bs,scale = update_scale_edf(y,z,eta,Wr,rowsX,colsX,LP,InvCholXXSP,Pr,lgdetDs,family,penalties,keep,drop,n_c)
   return eta,mu,coef,L,InvCholXXS,lgdetDs,bsbs,total_edf,term_edfs,Bs,scale,wres,keep,drop

def init_step_gam(y,yb,mu,eta,rowsX,colsX,X,Xb,
                  family,col_S,penalties,
                  pinv,n_c,formula,form_Linv,
                  method,offset):
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
   yb,Xb,z,Wr = update_PIRLS(y,yb,mu,eta - offset,X,Xb,family)
   
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
                                         X,Xb,family,S_emb,S_root,S_pinv,
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


def correct_coef_step(coef,n_coef,dev,pen_dev,c_dev_prev,family,eta,mu,y,X,n_pen,S_emb,formula,n_c,offset):
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

def initialize_extension(method,penalties):
   """
   Initializes a variable holding all the necessary information to
   compute the lambda extensions at every iteration of the fitting iteration.
   """

   if method == "nesterov" or method == "nesterov2":
      extend_by = {"prev_lam":[lterm.lam for lterm in penalties],
                   "acc":[0 for _ in penalties]}

   return extend_by

def extend_lambda_step(lti,lam,dLam,extend_by,was_extended, method):
   """
   Performs an update to the lambda parameter, ideally extending the step
   taken without overshooting the objective.
   
   If method is set to "nesterov", a nesterov-like acceleration update is applied to the
   lambda parameter. We fall back to the default startegy by Wood & Fasiolo (2016) to just
   half the step taken in case the extension was not succesful (for additive models).
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

def undo_extension_lambda_step(lti,lam,dLam,extend_by, was_extended, method, family):
   """
   Deals with resetting
   any extension terms.
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

def correct_lambda_step(y,yb,z,Wr,rowsX,colsX,X,Xb,coef,
                        family,col_S,S_emb,penalties,
                        was_extended,pinv,lam_delta,
                        extend_by,o_iter,dev_check,n_c,
                        control_lambda,extend_lambda,
                        exclude_lambda,extension_method_lam,
                        formula,form_Linv,method,offset,max_inner):
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
                                            X,Xb,family,S_emb,S_root,S_pinv,
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
            yb,Xb,z,Wr = update_PIRLS(y,yb,mu,eta-offset,X,Xb,family)

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
         
         wres,InvCholXXS,total_edf,term_edfs,_,scale = update_scale_edf(y,z,eta,Wr,rowsX,colsX,LP,InvCholXXSP,Pr,lgdetDs,family,penalties,keep,drop,n_c)

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

      # For Generalized models we should not reduce step beyond original EFS update since
      # criterion maximized is approximate REML. Also, we can probably relax the criterion a
      # bit, since check will quite often be < 0.
      check_criterion = 0
      if family.is_canonical == False: #(isinstance(family,Gaussian) == False) or (isinstance(family.link,Identity) == False):
            
            if o_iter > 0:
               check_criterion = 1e-7*-abs(pen_dev_check)
            
            if check[0,0] < check_criterion: 
               # Now check whether we extend lambda - and if we do so whether any extension was actually applied.
               # If not, we still "pass" the check.
               if (extend_lambda == False) or (np.any(was_extended) == False):
                  check[0,0] = check_criterion + 1

      # Now check whether we have to correct lambda.
      # Because of minimization in Wood (2017) they use a different check (step 7) but idea is the same.
      if check[0,0] < check_criterion and control_lambda: 
         # Reset extension or cut the step taken in half (for additive models)
         lam_changes = 0
         for lti,lTerm in enumerate(penalties):

            # Reset extension factor for all terms that were extended.
            if extend_lambda and was_extended[lti]:
               lam, dLam = undo_extension_lambda_step(lti,lTerm.lam,lam_delta[lti][0],extend_by,was_extended, extension_method_lam, family)
               lTerm.lam = lam
               lam_delta[lti][0] = dLam
               lam_changes += 1

            # Otherwise, rely on the strategy by Wood & Fasiolo (2016) to just half the step for canonical models.
            elif family.is_canonical:
                  lam_delta[lti] = lam_delta[lti]/2
                  lTerm.lam -= lam_delta[lti][0]
                  lam_changes += 1
         
         # For non-canonical models or if step becomes extremely small, accept step
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
            yb,Xb,z,Wr = update_PIRLS(y,yb,mu,eta-offset,X,Xb,family)
            
      if check[0,0] >= check_criterion or (control_lambda == False) or lam_accepted:
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

def solve_gamm_sparse(mu_init,y,X,penalties,col_S,family:Family,
                      maxiter=10,max_inner = 100,pinv="svd",conv_tol=1e-7,
                      extend_lambda=True,control_lambda=True,
                      exclude_lambda=False,extension_method_lam = "nesterov",
                      form_Linv=True,method="Chol",check_cond=2,progress_bar=False,
                      n_c=10,offset=0):
   # Estimates a penalized Generalized additive mixed model, following the steps outlined in Wood, Li, Shaddick, & Augustin (2017)
   # "Generalized Additive Models for Gigadata" referred to as Wood (2017) below.

   n_c = min(mp.cpu_count(),n_c)
   rowsX,colsX = X.shape
   coef = None
   n_coef = None
   K2 = None

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
                                                                                                     pinv,n_c,None,form_Linv,method,offset)
   
   yb,Xb,z,Wr = update_PIRLS(y,yb,mu,eta-offset,X,Xb,family)
   
   if check_cond == 2:
      K2,_,_,Kcode = est_condition(CholXXS,InvCholXXS,verbose=False)
      if method == "Chol" and Kcode == 1:
         raise ArithmeticError(f"Condition number ({K2}) of matrix A, where A.T@A=X.T@X + S_\lambda, is larger than 1/sqrt(u), where u is half the machine precision. Try calling ``model.fit()`` with ``method='QR'``.")
      if method != "Chol" and Kcode == 2:
         raise ArithmeticError(f"Condition number ({K2}) of matrix A, where A.T@A=X.T@X + S_\lambda, is larger than 1/u, where u is half the machine precision. The model estimates are likely inaccurate.")

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
            yb,Xb,z,Wr = update_PIRLS(y,yb,mu,eta - offset,X,Xb,family)
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
                                                                                                                                                                     family,col_S,S_emb,penalties,
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
                                            X,Xb,family,S_emb,None,None,None,
                                            penalties,n_c,None,form_Linv,offset)
         
      fit_info.dropped = drop

      # Check condition number of current system. 
      if check_cond == 2:
         K2,_,_,Kcode = est_condition(CholXXS,InvCholXXS,verbose=progress_bar)
         if method == "Chol" and Kcode == 1:
            raise ArithmeticError(f"Condition number ({K2}) of matrix A, where A.T@A=X.T@X + S_\lambda, is larger than 1/sqrt(u), where u is half the machine precision. Try calling ``model.fit()`` with ``method='QR'``.")
         if method != "Chol" and Kcode == 2:
            raise ArithmeticError(f"Condition number ({K2}) of matrix A, where A.T@A=X.T@X + S_\lambda, is larger than 1/u, where u is half the machine precision. The model estimates are likely inaccurate.")
      
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
      inval_check =  np.any(np.isnan(z))

      if inval_check:
         _, w, inval = PIRLS_pdat_weights(y,mu,eta-offset,family)
         w[inval] = 0

         # Re-compute weight matrix
         Wr = scp.sparse.spdiags([np.sqrt(np.ndarray.flatten(w))],[0])
   
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
         warnings.warn(f"Condition number ({K2}) of matrix A, where A.T@A=X.T@X + S_\lambda, is larger than 1/sqrt(u), where u is half the machine precision. Try calling ``model.fit()`` with ``method='QR'``.")
      if method != "Chol" and Kcode == 2:
         raise warnings.warn(f"Condition number ({K2}) of matrix A, where A.T@A=X.T@X + S_\lambda, is larger than 1/u, where u is half the machine precision. The model estimates are likely inaccurate.")

   fit_info.K2 = K2

   return coef,eta,wres,Wr,WN,scale,InvCholXXS,total_edf,term_edfs,penalty,fit_info

################################################ Iterative GAMM building code ################################################

def read_mmat(should_cache,cache_dir,file,fi,terms,has_intercept,
              ltx,irstx,stx,rtx,var_types,var_map,var_mins,
              var_maxs,factor_levels,cov_flat_file,cov):
   """
   Creates model matrix for that dataset. The model-matrix is either cached or not. If the former is the case,
   the matrix is read in on subsequent calls to this function
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

def form_cross_prod_mp(should_cache,cache_dir,file,fi,y_flat,terms,has_intercept,
                       ltx,irstx,stx,rtx,var_types,var_map,var_mins,
                       var_maxs,factor_levels,cov_flat_file,cov):
   
   model_mat = read_mmat(should_cache,cache_dir,file,fi,terms,has_intercept,
                         ltx,irstx,stx,rtx,var_types,var_map,var_mins,
                         var_maxs,factor_levels,cov_flat_file,cov)
   
   Xy = model_mat.T @ y_flat
   XX = (model_mat.T @ model_mat).tocsc()

   return XX,Xy

def read_XTX(file,formula,nc):
   """
   Reads subset of data and creates X.T@X with X = model matrix for that dataset.
   """

   terms = formula.get_terms()
   has_intercept = formula.has_intercept()
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

def keep_XTX(cov_flat,y_flat,formula,nc,progress_bar):
   """
   Takes subsets of data and creates X.T@X with X = model matrix iteratively over these subsets.
   """

   terms = formula.get_terms()
   has_intercept = formula.has_intercept()
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

def form_eta_mp(should_cache,cache_dir,file,fi,coef,terms,has_intercept,
                ltx,irstx,stx,rtx,var_types,var_map,var_mins,
                var_maxs,factor_levels,cov_flat_file,cov):
   
   model_mat = read_mmat(should_cache,cache_dir,file,fi,terms,has_intercept,
                         ltx,irstx,stx,rtx,var_types,var_map,var_mins,
                         var_maxs,factor_levels,cov_flat_file,cov)
   
   eta_file = (model_mat @ coef).reshape(-1,1)
   return eta_file

def read_eta(file,formula,coef,nc):
   """
   Reads subset of data and creates model prediction for that dataset.
   """

   terms = formula.get_terms()
   has_intercept = formula.has_intercept()
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

def keep_eta(formula,coef,nc):
   """
   Forms subset of data and creates model prediction for that dataset.
   """

   terms = formula.get_terms()
   has_intercept = formula.has_intercept()
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


def solve_gamm_sparse2(formula:Formula,penalties,col_S,family:Family,
                       maxiter=10,pinv="svd",conv_tol=1e-7,
                       extend_lambda=True,control_lambda=True,
                       exclude_lambda=False,extension_method_lam = "nesterov",
                       form_Linv=True,progress_bar=False,n_c=10):
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
                                                                                                     pinv,n_c,formula,form_Linv,"Chol",0)
   
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
                                                                                                                                                                        family,col_S,S_emb,penalties,
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
                                            None,XX,family,S_emb,None,None,None,
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

def reparam_model(dist_coef, dist_up_coef, coef, split_coef_idx, Xs, penalties, form_inverse=True, form_root=True, form_balanced=True, n_c=1):
    """Relies on the transformation strategy from Appendix B of Wood (2011) to re-parameterize the model.

    Coefficients, model matrices, and penalties are all transformed. The transformation is applied to each term separately as explained by
    Wood et al., (2016).

    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.

    :param dist_coef: List of number of coefficients per formula/linear predictor/distribution parameter of model.
    :type dist_coef: [int]
    :param dist_up_coef: List of number of **unpenalized** (i.e., fixed effects, linear predictors/parameters) coefficients per formula/linear predictor/distribution parameter of model.
    :type dist_up_coef: [int]
    :param coef: Vector of coefficients (numpy.array of dim (-1,1)).
    :type coef: numpy.array
    :param split_coef_idx: List with indices to split ``coef`` vector into separate versions per linear predictor.
    :type split_coef_idx: [int]
    :param Xs: List of model matrices obtained for example via ``model.get_mmat()``.
    :type Xs: [scipy.sparse.csc_array]
    :param penalties: List of penalties for model.
    :type penalties: [LambdaTerm]
    :param form_inverse: Whether or not an inverse of the transformed penalty matrices should be formed. Useful for computing the EFS update, defaults to True
    :type form_inverse: bool, optional
    :param form_root: Whether or not to form a root of the total penalty, defaults to True
    :type form_root: bool, optional
    :param form_balanced: Whether or not to form the "balanced" penalty as described by Wood et al. (2016) after the re-parameterization, defaults to True
    :type form_balanced: bool, optional
    :param n_c: Number of cores to use to ocmpute the inverse when ``form_inverse=True``, defaults to 1
    :type n_c: int, optional
    :raises ValueError: Raises a value error if one of the inverse computations fails. 
    :return: A tuple with 9 elements: the re-parameterized coefficient vector, a list with the re-parameterized model matrices, a list of the penalties after re-parameterization, the total re-parameterized penalty matrix, optionally the balanced version of the former, optionally a root of the re-parameterized total penalty matrix, optionally the inverse of the re-parameterized total penalty matrix, the transformation matrix ``Q`` so that ``Q.T@S_emb@Q = S_emb_rp`` where ``S_emb`` and ``S_emb_rp`` are the total penalty matrix before and after re-parameterization, a list of transformation matrices ``QD`` so that ``XD@QD=XD_rp`` where ``XD`` and XD_rp`` are the model matrix of the Dth linear predictor before and after re-parameterization.
    coef_rp,Xs_rp,Sj_reps,S_emb_rp,S_norm_rp,S_root_rp,S_inv_rp,Q_emb,Qs
    :rtype: (numpy.array, [scipy.sparse.csc_array], [Lambdaterm], scp.sparse.csc_array, scp.sparse.csc_array or None, scp.sparse.csc_array or None, scp.sparse.csc_array or None, scp.sparse.csc_array, [scp.sparse.csc_array])
    """

    # Apply reparam from Wood (2011) Appendix B
    Sj_reps,S_reps,SJ_term_idx,_,S_coefs,Q_reps,_,_ = reparam(Xs[0],penalties,None,option=4)

    S_emb_rp = None
    S_norm_rp = None
    S_root_rp = None
    S_inv_rp = None
    Q_emb = None

    dist_idx = Sj_reps[0].dist_param
    Qs = []

    # Create transformation matrix for first dist. parameter/linear predictor and overall
    Q_emb,_ = embed_in_S_sparse(*translate_sparse(scp.sparse.identity(dist_up_coef[0],format='csc')),None,len(coef),dist_up_coef[0],0)
    Qd_emb,c_idx = embed_in_S_sparse(*translate_sparse(scp.sparse.identity(dist_up_coef[0],format='csc')),None,dist_coef[0],dist_up_coef[0],0)
    cd_idx = c_idx

    for Si,(S_rep,S_coef) in enumerate(zip(S_reps,S_coefs)):
        
        # Create new Q if we move to a new parameter
        if Sj_reps[SJ_term_idx[Si][0]].dist_param != dist_idx:
            dist_idx += 1
            Qs.append(Qd_emb)

            # Make sure to update c_idx and cdidx here
            cd_idx = 0
            Qd_emb,cd_idx = embed_in_S_sparse(*translate_sparse(scp.sparse.identity(dist_up_coef[dist_idx],format='csc')),None,dist_coef[dist_idx],dist_up_coef[dist_idx],cd_idx)
            Q_emb,c_idx = embed_in_S_sparse(*translate_sparse(scp.sparse.identity(dist_up_coef[dist_idx],format='csc')),Q_emb,len(coef),dist_up_coef[dist_idx],c_idx)
            #c_idx += dist_up_coef[dist_idx]

        # Compute inverse of S_rep
        if form_inverse or form_root:
            L,code = cpp_chol(S_rep.tocsc())
            if code != 0:
                raise ValueError("Inverse of transformed penalty could not be computed.")
        
        if form_inverse:
            Linv = compute_Linv(L,n_c)
            S_inv = Linv.T@Linv

        # Embed transformed unweighted SJ as well
        for sjidx in SJ_term_idx[Si]:
            S_J_emb_rp,c_idx_j = embed_in_S_sparse(*translate_sparse(Sj_reps[sjidx].S_J),None,len(coef),S_coef,c_idx)
            for _ in range(1,Sj_reps[SJ_term_idx[Si][0]].rep_sj):
                S_J_emb_rp,c_idx_j = embed_in_S_sparse(*translate_sparse(Sj_reps[sjidx].S_J),S_J_emb_rp,len(coef),S_coef,c_idx_j)
            
            Sj_reps[sjidx].S_J_emb = S_J_emb_rp
            
            # Also pad S_J (not embedded) with zeros so that shapes remain consistent compared to original parameterization
            S_J_rp,_ = embed_in_S_sparse(*translate_sparse(Sj_reps[sjidx].S_J),None,S_coef,S_coef,0)
            Sj_reps[sjidx].S_J = S_J_rp

        # Continue to fill overall penalty matrix and current Q
        for _ in range(Sj_reps[SJ_term_idx[Si][0]].rep_sj):

            # Transformation matrix for current linear predictor
            Qd_emb,cd_idx = embed_in_S_sparse(*translate_sparse(Q_reps[Si]),Qd_emb,dist_coef[dist_idx],S_coef,cd_idx)

            # Overall transformation matrix for penalties
            Q_emb,_ = embed_in_S_sparse(*translate_sparse(Q_reps[Si]),Q_emb,len(coef),S_coef,c_idx)

            # Inverse of total weighted penalty
            if form_inverse:
                S_inv_rp,_ = embed_in_S_sparse(*translate_sparse(S_inv),S_inv_rp,len(coef),S_coef,c_idx)
            
            # Root of total weighted penalty
            if form_root:
                S_root_rp,_ = embed_in_S_sparse(*translate_sparse(L),S_root_rp,len(coef),S_coef,c_idx)
            
            # Total weighted penalty
            S_emb_rp,c_idx = embed_in_S_sparse(*translate_sparse(S_rep),S_emb_rp,len(coef),S_coef,c_idx)
   
    # Fill remaining cells of Q_emb in case final dist. parameter/linear predictor has only unpenalized coef
    if c_idx != len(coef):
       Q_emb,c_idx = embed_in_S_sparse(*translate_sparse(scp.sparse.identity(len(coef)-c_idx,format='csc')),Q_emb,len(coef),len(coef)-c_idx,c_idx)

    
    # Collect final Q
    Qs.append(Qd_emb)

    # Transform roots of S_J_emb
    for pidx in range(len(Sj_reps)):
        Sj_reps[pidx].D_J_emb = Q_emb.T @ penalties[pidx].D_J_emb

    if form_balanced:
        # Compute balanced penalty in reparam
        S_norm_rp = copy.deepcopy(Sj_reps[0].S_J_emb)/scp.sparse.linalg.norm(Sj_reps[0].S_J_emb,ord=None)
        for peni in range(1,len(Sj_reps)):
            S_norm_rp += Sj_reps[peni].S_J_emb/scp.sparse.linalg.norm(Sj_reps[peni].S_J_emb,ord=None)

        S_norm_rp /= scp.sparse.linalg.norm(S_norm_rp,ord=None)

    # Transform model matrices
    Xs_rp = copy.deepcopy(Xs)
    for qi,Q in enumerate(Qs):
        #print(Xs[qi].shape,Q.shape)
        Xs_rp[qi] = Xs[qi]@Q

    # Transform coef
    coef_rp = copy.deepcopy(coef)
    split_coef_rp = np.split(coef_rp,split_coef_idx)

    for qi,Q in enumerate(Qs):
        split_coef_rp[qi] = Q.T@split_coef_rp[qi]

    coef_rp = np.concatenate(split_coef_rp).reshape(-1,1)

    return coef_rp,Xs_rp,Sj_reps,S_emb_rp,S_norm_rp,S_root_rp,S_inv_rp,Q_emb,Qs

def deriv_transform_mu_eta(y,means,family:GAMLSSFamily):
    """
    Compute derivatives (first and second order) of llk with respect to each mean for all observations following steps outlined by Wood, Pya, & Säfken (2016)

    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
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
    WPS (2016) provide that $l_{\eta}$ is obtained as $l^i_{\mu}/h'(\mu^i)$ - where $h'$ is the derivative of the link function $h$.
    This follows from applying the chain rule and the inversion rule of derivatives
    $\frac{\partial llk(h^{-1}(\eta))}{\partial \eta} = \frac{\partial llk(\mu)}{\partial \mu} \frac{\partial h^{-1}(\eta)}{\partial \eta} = \frac{\partial llk(\mu)}{\partial \mu}\frac{1}{\frac{\partial h(\mu)}{\mu}}$.
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
    For second derivatives we need pure and mixed. Computation of $l^\mathbf{i}_{\eta^l,\eta^m}$ in general is again obtained by applying the steps outlined for first order and provided by WPS (2016)
    for the pure case. For the mixed case it is even simpler: We need $\frac{\partial^2 llk(h_1^{-1}(\eta^1),h_2^{-1}(\eta^2))}{\partial \eta^1 \partial \eta^2}$,
    which is $\frac{\partial llk /\ \partial \eta^1}{\partial \eta^2}$. With the first partial being equal to
    $\frac{\partial llk}{\partial \mu^1}\frac{1}{\frac{\partial h_1(\mu^1)}{\mu^1}}$ (see comment above for first order) we now have
    $\frac{\partial \frac{\partial llk}{\partial \mu^1}\frac{1}{\frac{\partial h_1(\mu^1)}{\mu^1}}}{\partial \eta^2}$.
    
    We now apply the product rule (the second term in the sum disappears, because
    $\partial \frac{1}{\frac{\partial h_1(\mu^1)}{\mu^1}} /\ \partial \eta^2 = 0$, this is not the case for pure second derivatives as shown in WPS, 2016)
    to get $\frac{\partial \frac{\partial llk}{\partial \mu^1}}{\partial \eta^2} \frac{1}{\frac{\partial h_1(\mu^1)}{\mu^1}}$.
    We can now again rely on the same steps taken to get the first derivatives (chain rule + inversion rule) to get
    $\frac{\partial \frac{\partial llk}{\partial \mu^1}}{\partial \eta^2} =
    \frac{\partial^2 llk}{\partial \mu^1 \partial \mu^2}\frac{1}{\frac{\partial h_2(\mu^2)}{\mu^2}}$.
    
    Thus, $\frac{\partial llk /\ \partial \eta^1}{\partial \eta^2} =
    \frac{\partial^2 llk}{\partial \mu^1 \partial \mu^2}\frac{1}{\frac{\partial h_2(\mu^2)}{\mu^2}}\frac{1}{\frac{\partial h_1(\mu^1)}{\mu^1}}$.
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

def deriv_transform_eta_beta(d1eta,d2eta,d2meta,Xs,only_grad=False):
    """
    Further transforms derivatives of llk with respect to eta to get derivatives of llk with respect to coefficients
    Based on section 3.2 and Appendix A in Wood, Pya, & Säfken (2016)

    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
    """

    # Gradient: First order partial derivatives of llk with respect to coefficients
    """
    WPS (2016) provide $l^j_{\beta^l} = l^\mathbf{i}_{\eta^l}\mathbf{X}^l_{\mathbf{i},j}$. See ```deriv_transform_mu_eta```.
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
    $l^{j,k}_{\beta^l,\beta^m} = l^\mathbf{i}_{\eta^l,\eta^m}\mathbf{X}^l_{\mathbf{i},j}\mathbf{X}^m_{\mathbf{i},k}$.
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

def newton_coef_smooth(coef,grad,H,S_emb):
    """
    Follows sections 3.1.2 and 3.14 in WPS (2016) to update the coefficients of the GAMLSS model via a
    newton step.
    1) Computes gradient of the penalized likelihood (grad - S_emb@coef)
    2) Computes negative Hessian of the penalized likelihood (-1*H + S_emb) and it's inverse.
    3) Uses these two to compute the Netwon step.
    4) Step size control - happens outside

    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
     - mgcv source code, in particular: https://github.com/cran/mgcv/blob/master/R/gam.fit4.r
    """
    
    pgrad = np.array([grad[i] - (S_emb[[i],:]@coef)[0] for i in range(len(grad))])
    nH = -1*H + S_emb

    # Diagonal pre-conditioning as suggested by WPS (2016) and implemented in mgcv's gam.fit5 function,
    # see: https://github.com/cran/mgcv/blob/master/R/gam.fit4.r#L1028
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
       nH += mD*scp.sparse.identity(nH.shape[1],format='csc') + mAcc*scp.sparse.identity(nH.shape[1],format='csc')
       D = scp.sparse.diags(np.ones_like(nHdgr))
       DI = D
    else:
      D = scp.sparse.diags(np.power(nHdgr,-0.5))
      DI = scp.sparse.diags(1/np.power(nHdgr,-0.5)) # For cholesky

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

    return n_coef,DI@P.T@Lp,LV,eps

def gd_coef_smooth(coef,grad,S_emb,a):
    """
    Follows sections 3.1.2 and 3.14 in WPS (2016) to update the coefficients of the GAMLSS model via a
    Gradient descent (ascent actually) step.
    1) Computes gradient of the penalized likelihood (grad - S_emb@coef)
    3) Uses this to compute update
    4) Step size control - happens outside

    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
    """
    
    # Compute penalized gradient
    pgrad = np.array([grad[i] - (S_emb[[i],:]@coef)[0] for i in range(len(grad))])
    
    # Update coef
    n_coef = coef + a * pgrad
    
    return n_coef

def correct_coef_step_gammlss(family,y,Xs,coef,next_coef,coef_split_idx,c_llk,S_emb):
    """
    Apply step size correction to Newton update for GAMLSS models, as discussed by WPS (2016).

    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
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
    n_checks = 0
    while next_pen_llk < prev_llk_cur_pen or (np.isinf(next_pen_llk[0,0]) or np.isnan(next_pen_llk[0,0])):
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
    
    return next_coef,next_split_coef,next_mus,next_etas,next_llk,next_pen_llk

def identify_drop(H,S_scaled,method='QR'):
    """
    Routine to (approximately) identify the rank of the scaled negative hessian of the penalized likelihood based on a rank revealing QR decomposition or the methods by Foster (1986) and Gotsman & Toledo (2008).
    
    If ``method=="QR"``, a rank revealing QR decomposition is performed for the scaled penalized Hessian. The latter has to be transformed to a dense matrix for this.
    This is essentially the approach by Wood et al. (2016) and is the most accurate. Alternatively, we can rely on a variant of Foster's method.
    This is done when ``method=="LU"`` or ``method=="direct"``. ``method=="LU"`` requires ``p`` LU decompositions - where ``p`` is approximately the Kernel size of the matrix.
    Essentially continues to find vectors forming a basis of the Kernel of the balanced penalzied Hessian from the upper matrix of the LU decomposition and successively drops columns
    corresponding to the maximum absolute value of the Kernel vectors (see Foster, 1986). This is repeated until we can form a cholesky of the scaled penalized hessian which as an acceptable condition number.
    If ``method=="direct"``, the same procedure is completed, but Kernel vectors are found directly based on the balanced penalized Hessian, which can be less precise. 

    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
     - Foster (1986). Rank and null space calculations using matrix decomposition without column interchanges.
     - Gotsman & Toledo (2008). On the Computation of Null Spaces of Sparse Rectangular Matrices.
     - mgcv source code, in particular: https://github.com/cran/mgcv/blob/master/R/gam.fit4.r
    
    :param H: Estimate of the hessian of the log-likelihood.
    :type H: scipy.sparse.csc_array
    :param S_scaled: Scaled version of the penalty matrix (i.e., unweighted total penalty divided by it's norm).
    :type S_scaled: scipy.sparse.csc_array
    :param method: Which method to use to check for rank deficiency, defaults to 'QR'
    :type method: str, optional
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

def drop_terms_S(penalties,keep):
   """Zeros out rows and cols of penalty matrices corresponding to dropped terms. Roots are re-computed as well.

   :param penalties: List of Lambda terms included in the model formula
   :type penalties: [:class:`mssm.src.python.penalties.LambdaTerm`]
   :param keep: List of columns/rows to keep.
   :type keep: [int]
   :return: List of updated penalties - a copy is made.
   :rtype: [:class:`mssm.src.python.penalties.LambdaTerm`]
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


def drop_terms_X(Xs,keep):
    """Drops cols of model matrices corresponding to dropped terms.

    :param penalties: List of model matrices included in the model formula.
    :type penalties: [scipy.sparse.csc_array]
    :param keep: List of columns to keep.
    :type keep: [int]
    :return: List of updated model matrices - a copy is made.
    :rtype: [scipy.sparse.csc_array]
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

def check_drop_valid_gammlss(y,coef,coef_split_idx,Xs,S_emb,keep,family):
   """Checks whether an identified set of coefficients to be dropped from the model results in a valid log-likelihood.

   :param y: Vector of response variable
   :type y: numpy.array
   :param coef: Vector of coefficientss
   :type coef: numpy.array
   :param coef_split_idx: List with indices to split coef - one per parameter of response distribution
   :type coef_split_idx: [int]
   :param Xs: List of model matrices - one per parameter of response distribution
   :type Xs: [scipy.sprase.csc_array]
   :param S_emb: Penalty matrix
   :type S_emb: scipy.sparse.csc-array
   :param keep: List of coefficients to retain
   :type keep: [int]
   :param family: Model family
   :type family: mssm.src.python.exp_fam.GAMLSSFamily
   :return: tuple holding check if likelihood is valid and penalized log-likelihood under dropped set.
   :rtype: (bool,[[float]] or None)
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

    
def update_coef_gammlss(family,mus,y,Xs,coef,coef_split_idx,S_emb,S_norm,S_pinv,FS_use_rank,gammlss_penalties,c_llk,outer,max_inner,min_inner,conv_tol,method,piv_tol,keep_drop):
   """
   Repeatedly perform Newton update with step length control to the coefficient vector - based on
   steps outlined by WPS (2016). Checks for rank deficiency when ``method != "Chol"``.

   References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
   """
   # Update coefficients:
   if keep_drop is None:
      converged = False
      for inner in range(max_inner):
         
         # Get derivatives with respect to eta
         if family.d_eta == False:
            d1eta,d2eta,d2meta = deriv_transform_mu_eta(y,mus,family)
         else:
            d1eta = [fd1(y,*mus) for fd1 in family.d1]
            d2eta = [fd2(y,*mus) for fd2 in family.d2]
            d2meta = [fd2m(y,*mus) for fd2m in family.d2m]

         # Get derivatives with respect to coef
         grad,H = deriv_transform_eta_beta(d1eta,d2eta,d2meta,Xs,only_grad=False)

         # Update coef and perform step size control
         next_coef,L,LV,eps = newton_coef_smooth(coef,grad,H,S_emb)

         # Prepare to check convergence
         prev_llk_cur_pen = c_llk - 0.5*coef.T@S_emb@coef

         # Perform step length control
         coef,split_coef,mus,etas,c_llk,c_pen_llk = correct_coef_step_gammlss(family,y,Xs,coef,next_coef,coef_split_idx,c_llk,S_emb)

         if np.abs(c_pen_llk - prev_llk_cur_pen) < conv_tol*np.abs(c_pen_llk):
            converged = True
            #if eps <= 0 or np.linalg.norm(coef-next_coef) < conv_tol*np.linalg.norm(coef):
            break

         if eps <= 0 and outer > 0 and inner >= (min_inner-1):
            break # end inner loop and immediately optimize lambda again.

      #print(converged,eps)
   
      # Make sure at convergence negative Hessian of llk is at least positive semi-definite as well
      checkH = True
      checkHc = 0
      
      while checkH and (gammlss_penalties is not None):
         #_,S_pinv,_,FS_use_rank = compute_S_emb_pinv_det(len(coef),gammlss_penalties,"svd")

         # Re-compute lgdetDs, ldetHS
         lgdetDs = []

         for lti,lTerm in enumerate(gammlss_penalties):

               lt_rank = None
               if FS_use_rank[lti]:
                  lt_rank = lTerm.rank

               lgdetD,_ = compute_lgdetD_bsb(lt_rank,lTerm.lam,S_pinv,lTerm.S_J_emb,coef)
               lgdetDs.append(lgdetD)

         _,_, ldetHSs = calculate_edf(None,None,LV,gammlss_penalties,lgdetDs,len(coef),10,None)

         # And check whether Theorem 1 now holds
         checkH = np.any([lgdetDs[lti] - ldetHSs[lti] < 0 for lti in range(len(gammlss_penalties))])

         if checkH:
            if eps == 0:
               eps += 1e-14
            else:
               eps *= 2
            _,L,LV,_ = newton_coef_smooth(coef,grad,H - eps*scp.sparse.identity(H.shape[1],format='csc'),S_emb)

         if checkHc > 30:
            break

         checkHc += 1
      
   
   # In case we coverged check for unidentifiable parameters, as reccomended by Wood. et al (2016)
   keep = None
   drop = None
   if keep_drop is not None or (method == "QR/Chol" and converged):
      
      if keep_drop is not None:
         keep = keep_drop[0]
         drop = keep_drop[1]
      else:
         # Check for drop
         keep,drop = identify_drop(H,S_norm)

      if len(drop) == 0:
         keep = None
         drop = None
      elif keep_drop is None:
         # Found drop, but need to check whether it is safe
         drop_valid,drop_pen_llk = check_drop_valid_gammlss(y,coef,coef_split_idx,Xs,S_emb,keep,family)

         # Skip if likelihood becomes invalid
         if drop_valid == False:
            keep = None
            drop = None
         # If we identify all or all but one coefficients to drop also skip
         elif drop_valid and len(drop) >= len(coef) - 1:
            keep = None
            drop = None
      
      #print(drop)
      # Now we need to continue iterating in smaller problem until convergence
      if drop is not None:
         # Prepare zeroed full coef vector
         full_coef = np.zeros_like(coef)

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

         if len(coef_split_idx) != 0:
            for mi in range(1,len(mus)):
               inval = inval |  np.isnan(mus[mi])
         inval = inval.flatten()

         # Re-compute llk
         c_llk = family.llk(y[inval == False],*[mu[inval == False] for mu in mus])
         c_pen_llk = c_llk - 0.5*coef.T@rS_emb@coef
         
         # and now repeat Newton iteration
         for inner in range(max_inner):
      
            # Get derivatives with respect to eta
            if family.d_eta == False:
               d1eta,d2eta,d2meta = deriv_transform_mu_eta(y,mus,family)
            else:
               d1eta = [fd1(y,*mus) for fd1 in family.d1]
               d2eta = [fd2(y,*mus) for fd2 in family.d2]
               d2meta = [fd2m(y,*mus) for fd2m in family.d2m]

            # Get derivatives with respect to coef
            grad,H = deriv_transform_eta_beta(d1eta,d2eta,d2meta,rXs,only_grad=False)

            # Update Coefficients
            next_coef,L,LV,eps = newton_coef_smooth(coef,grad,H,rS_emb)

            # Prepare to check convergence
            prev_llk_cur_pen = c_llk - 0.5*coef.T@rS_emb@coef

            # Perform step length control
            coef,rsplit_coef,mus,etas,c_llk,c_pen_llk = correct_coef_step_gammlss(family,y,rXs,coef,next_coef,rcoef_split_idx,c_llk,rS_emb)

            if np.abs(c_pen_llk - prev_llk_cur_pen) < conv_tol*np.abs(c_pen_llk):
               break

         #print("eps",eps)
         # converged on smaller problem now adjust return objects for dropped coef

         # start with coef
         full_coef[keep] = coef

         # Again make sure negative Hessian of llk is at least positive semi-definite as well
         checkH = True
         if gammlss_penalties is not None:
            drop_pen = copy.deepcopy(gammlss_penalties)
         LVdat,LVrow,LVcol = translate_sparse(LV) # LV.T@LV = V
         LVrow = keep[LVrow]
         LVcol = keep[LVcol]
         LV = scp.sparse.csc_array((LVdat,(LVrow,LVcol)),shape=(len(full_coef),len(full_coef)))

         checkHc = 0
         
         while checkH and (gammlss_penalties is not None):
            #_,S_pinv,_,FS_use_rank = compute_S_emb_pinv_det(len(full_coef),drop_pen,"svd")

            # Now re-compute lgdetDs, ldetHS
            lgdetDs = []

            for lti,lTerm in enumerate(drop_pen):

                  lt_rank = None
                  if FS_use_rank[lti]:
                     lt_rank = lTerm.rank

                  lgdetD,_ = compute_lgdetD_bsb(lt_rank,lTerm.lam,S_pinv,lTerm.S_J_emb,full_coef)
                  lgdetDs.append(lgdetD)

            _,_, ldetHSs = calculate_edf(None,None,LV,drop_pen,lgdetDs,len(full_coef),10,None)

            checkH = np.any([lgdetDs[lti] - ldetHSs[lti] < 0 for lti in range(len(drop_pen))])

            if checkH:
               if eps == 0:
                  eps += 1e-14
               else:
                  eps *= 2
               _,L,LV,_ = newton_coef_smooth(coef,grad,H - eps*scp.sparse.identity(H.shape[1],format='csc'),rS_emb)
               LVdat,LVrow,LVcol = translate_sparse(LV) # LV.T@LV = V
               LVrow = keep[LVrow]
               LVcol = keep[LVcol]
               LV = scp.sparse.csc_array((LVdat,(LVrow,LVcol)),shape=(len(full_coef),len(full_coef)))
            
            if checkHc > 30:
               break

            checkHc += 1
         

         coef = full_coef
         #print(coef)
         split_coef = np.split(coef,coef_split_idx)

         # Now H, L
         Hdat,Hrow,Hcol = translate_sparse(H)
         Ldat,Lrow,Lcol = translate_sparse(L.tocsc()) # L@L.T = H_pen
         
         Hrow = keep[Hrow]
         Hcol = keep[Hcol]
         Lrow = keep[Lrow]
         Lcol = keep[Lcol]

         H = scp.sparse.csc_array((Hdat,(Hrow,Hcol)),shape=(len(full_coef),len(full_coef)))
         L = scp.sparse.csc_array((Ldat,(Lrow,Lcol)),shape=(len(full_coef),len(full_coef)))
         #print((LV.T@LV - V).max(),(LV.T@LV - V).min())
         #print((L@L.T - nH).max(),(L@L.T - nH).min())

   return coef,split_coef,mus,etas,H,L,LV,c_llk,c_pen_llk,eps,keep,drop

def correct_lambda_step_gamlss(family,mus,y,Xs,S_norm,n_coef,form_n_coef,form_up_coef,coef,
                               coef_split_idx,gamlss_pen,lam_delta,
                               extend_by,was_extended,c_llk,fit_info,outer,
                               max_inner,min_inner,conv_tol,method,
                               piv_tol,keep_drop,extend_lambda,
                               extension_method_lam,control_lambda,repara,n_c):
   
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

      total_edf,term_edfs, ldetHSs = calculate_edf(None,None,LV,gamlss_pen_rp,lgdetDs,n_coef,n_c,None)
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
         for lti,lTerm in enumerate(gamlss_pen_rp):

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
   for lti,lTerm in enumerate(gamlss_pen_rp):

      lgdetD = lgdetDs[lti]
      ldetHS = ldetHSs[lti]
      bsb = bsbs[lti]
      
      #print(lgdetD-ldetHS)
      dLam = step_fellner_schall_sparse(lgdetD,ldetHS,bsb[0,0],lTerm.lam,1)
      #print("Theorem 1:",lgdetD-ldetHS,bsb)

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

      for lti,lTerm in enumerate(gamlss_pen_rp): # Re-write accepted lambda to original penalties
         gamlss_pen[lti].lam = lTerm.lam
         
      # Transform S_emb (which is currently S_emb_rp)
      S_emb = Q_emb @ S_emb @ Q_emb.T

      # Transform coef
      for qi,Q in enumerate(Qs):
         split_coef[qi] = Q@split_coef[qi]
      next_coef = np.concatenate(split_coef).reshape(-1,1)

      # Transform H, L, LV
      H = Q_emb.T @ H @ Q_emb
      L = Q_emb.T @ L
      LV = LV @ Q_emb.T

   return next_coef,split_coef,next_mus,next_etas,H,L,LV,next_llk,next_pen_llk,eps,keep,drop,S_emb,gamlss_pen,total_edf,term_edfs,lam_delta

    
def solve_gammlss_sparse(family,y,Xs,form_n_coef,form_up_coef,coef,coef_split_idx,gamlss_pen,
                         max_outer=50,max_inner=30,min_inner=1,conv_tol=1e-7,
                         extend_lambda=True,extension_method_lam = "nesterov2",
                         control_lambda=True,method="Chol",check_cond=1,piv_tol=0.175,
                         repara=True,should_keep_drop=True,progress_bar=True,n_c=10):
    """
    Fits a GAMLSS model, following steps outlined by Wood, Pya, & Säfken (2016).

    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
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

def correct_coef_step_gen_smooth(family,y,Xs,coef,next_coef,coef_split_idx,c_llk,S_emb,a):
    """
    Apply step size correction to Newton update for general smooth models, as discussed by WPS (2016).

    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
    """
    
    # Step size control for newton step.
    next_llk = family.llk(next_coef,coef_split_idx,y,Xs)
    
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
        next_llk = family.llk(next_coef,coef_split_idx,y,Xs)
        next_pen_llk = next_llk - 0.5*next_coef.T@S_emb@next_coef
        
    # Update step-size for gradient
    if n_checks > 0 and a > 1e-9:
       a /= 2
    elif n_checks == 0 and a < 1:
       a *= 2
    
    return next_coef,next_llk,next_pen_llk,a

def back_track_alpha(coef,step,llk_fun,grad_fun,*llk_args,alpha_max=1,c1=1e-4,max_iter=100):
   # Simple step-size backtracking function that enforces Armijo condition (Nocedal & Wright, 2004)
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

def check_drop_valid_gensmooth(y,coef,Xs,S_emb,keep,family):
   """Checks whether an identified set of coefficients to be dropped from the model results in a valid log-likelihood.

   :param y: Vector of response variable
   :type y: numpy.array
   :param coef: Vector of coefficientss
   :type coef: numpy.array
   :param coef_split_idx: List with indices to split coef - one per parameter of response distribution
   :type coef_split_idx: [int]
   :param Xs: List of model matrices - one per parameter of response distribution
   :type Xs: [scipy.sprase.csc_array]
   :param S_emb: Penalty matrix
   :type S_emb: scipy.sparse.csc-array
   :param keep: List of coefficients to retain
   :type keep: [int]
   :param family: Model family
   :type family: mssm.src.python.exp_fam.GAMLSSFamily
   :return: tuple holding check if likelihood is valid and penalized log-likelihood under dropped set.
   :rtype: (bool,[[float]] or None)
   """
   # Drop from coef
   dropcoef = coef[keep]

   # ... from Xs...
   rXs,rcoef_split_idx = drop_terms_X(Xs,keep)

   # ... and from S_emb
   rS_emb = S_emb[keep,:]
   rS_emb = rS_emb[:,keep]

   # Re-compute llk
   c_llk = family.llk(dropcoef,rcoef_split_idx,y,rXs)
   c_pen_llk = c_llk - 0.5*dropcoef.T@rS_emb@dropcoef

   if (np.isinf(c_pen_llk[0,0]) or np.isnan(c_pen_llk[0,0])):
      return False,None
   
   return True,c_pen_llk

def restart_coef(coef,c_llk,c_pen_llk,n_coef,coef_split_idx,y,Xs,S_emb,family,outer,restart_counter):
   """Shrink coef towards random vector to restart algorithm if it get's stuck.

   :param coef: _description_
   :type coef: _type_
   :param n_coef: _description_
   :type n_coef: _type_
   :param coef_split_idx: _description_
   :type coef_split_idx: _type_
   :param y: _description_
   :type y: _type_
   :param Xs: _description_
   :type Xs: _type_
   :param S_emb: _description_
   :type S_emb: _type_
   :param family: _description_
   :type family: _type_
   :param outer: _description_
   :type outer: _type_
   :param restart_counter: _description_
   :type restart_counter: _type_
   :return: _description_
   :rtype: _type_
   """
   res_checks = 0
   res_scale = 0.5 if outer <= 10 else 1/(restart_counter+2)
   #print("resetting conv.",res_scale)

   while res_checks < 30:
      res_coef = ((1-res_scale)*coef + res_scale*scp.stats.norm.rvs(size=n_coef,random_state=outer+res_checks).reshape(-1,1))

      # Re-compute llk
      with warnings.catch_warnings():
         warnings.simplefilter("ignore")
         res_llk = family.llk(res_coef,coef_split_idx,y,Xs)
         res_pen_llk = res_llk - 0.5*res_coef.T@S_emb@res_coef

      if (np.isinf(res_pen_llk[0,0]) or np.isnan(res_pen_llk[0,0])):
         res_checks += 1
         continue
      
      coef = res_coef
      c_llk = res_llk
      c_pen_llk = res_pen_llk
      break
   
   return coef, c_llk, c_pen_llk

def test_SR1(sk,yk,rho,sks,yks,rhos):
   """Test whether SR1 update is well-defined for both V and H.

   :param sk: _description_
   :type sk: _type_
   :param yk: _description_
   :type yk: _type_
   :param rho: _description_
   :type rho: _type_
   :param sks: _description_
   :type sks: _type_
   :param yks: _description_
   :type yks: _type_
   :param rhos: _description_
   :type rhos: _type_
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

    
def update_coef_gen_smooth(family,y,Xs,coef,coef_split_idx,S_emb,S_norm,smooth_pen,c_llk,outer,max_inner,min_inner,conv_tol,method,piv_tol,keep_drop,opt_raw):
   """
   Repeatedly perform Newton/Graidnet update with step length control to the coefficient vector - based on
   steps outlined by WPS (2016).

   References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
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
      def __neg_llk(coef,coef_split_idx,y,Xs,family,S_emb):
         coef = coef.reshape(-1,1)
         neg_llk = -1 * family.llk(coef,coef_split_idx,y,Xs)
         return neg_llk + 0.5*coef.T@S_emb@coef
      
      def __neg_grad(coef,coef_split_idx,y,Xs,family,S_emb):
         # see Wood, Pya & Saefken (2016)
         coef = coef.reshape(-1,1)
         grad = family.gradient(coef,coef_split_idx,y,Xs)
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
                                 args=(coef_split_idx,y,Xs,family,S_emb),
                                 method="L-BFGS-B",
                                 jac = __neg_grad,
                                 options={"maxiter":max_inner,
                                          **opt_raw.bfgs_options})

      coef = opt["x"].reshape(-1,1)
      c_llk = family.llk(coef,coef_split_idx,y,Xs)

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
   if keep_drop is None:
      converged = False
      for inner in range(max_inner):
         
         # Get llk derivatives with respect to coef
         grad = family.gradient(coef,coef_split_idx,y,Xs)

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
                  alpha = back_track_alpha(coef,step,__neg_llk,__neg_grad,coef_split_idx,y,Xs,family,S_up,alpha_max=1)
                  new_slope = 1
               else:
                  # Find a step that meets the Wolfe conditions (Nocedal & Wright, 2004)
                  alpha,_,_,_,_,new_slope = scp.optimize.line_search(__neg_llk,__neg_grad,coef.flatten(),step.flatten(),
                                                                     args=(coef_split_idx,y,Xs,family,S_up),
                                                                     maxiter=100,amax=1)
               
               if alpha is None:
                  new_slope = None
                  alpha = 1e-7

               # Compute gradient at new point
               next_grad_up = family.gradient(coef + alpha*step,coef_split_idx,y,Xs)
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
                     coef, _, _ = restart_coef(coef,None,None,len(coef),coef_split_idx,y,Xs,S_emb,family,inner,0)
                     c_llk = family.llk(coef,coef_split_idx,y,Xs)
                     grad = family.gradient(coef,coef_split_idx,y,Xs)

               if new_slope is not None and (form != 'SR1' or (skip == False)):
                  # Wolfe/Armijo met, can collect update vectors

                  if form != 'SR1' and len(sks) > 0:
                     # But first dampen for BFGS update - see Nocedal & Wright (2004):
                     Ht1, Ht2, Ht3 = computeH(sks,yks,rhos,H0,make_psd=False,omega=omega,explicit=False)
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

                  # Store update to S and Y in scipy LbfgsInvHess..
                  V_raw = scp.optimize.LbfgsInvHessProduct(sks,yks)

                  # And keep that in LV, since this is what we want to return later
                  LV = V_raw
                  LV.nit = opt["nit"]
                  LV.omega = np.min(omegas)
                  LV.method = "qEFS"
                  LV.updates = updates
                  LV.form = form

                  # Compute updated estimate of inverse of negative hessian of llk (implicitly)
                  if form == 'SR1':
                     t1_llk, t2_llk, t3_llk = computeVSR1(V_raw.sk,V_raw.yk,V_raw.rho,V0,1/omega,make_psd=True,explicit=False) #
                  else:
                     t1_llk, t2_llk, t3_llk = computeV(V_raw.sk,V_raw.yk,V_raw.rho,V0,explicit=False)

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
                  nt1,nt2,int2,nt3,_ = compute_t1_shifted_t2_t3(V_raw.sk,V_raw.yk,V_raw.rho,H0,omega,"ByrdSR1" if form == 'SR1' else "Byrd")

                  # Compute inverse:
                  if form != 'SR1':
                     invt2 = nt2 + nt3@pV0@nt1

                     U,sv_invt2,VT = scp.linalg.svd(invt2,lapack_driver='gesvd')

                     # Nowe we can compute all parts for the Woodbury identy to obtain pV
                     t2 = VT.T @ np.diag(1/sv_invt2)  @  U.T

                     t1 = pV0@nt1
                     t3 = nt3@pV0
                  else:
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
               H = family.hessian(coef,coef_split_idx,y,Xs)

         # Update coef and perform step size control
         if True:

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
                  alpha_pen = back_track_alpha(coef,pen_step,__neg_llk,__neg_grad,coef_split_idx,y,Xs,family,S_emb,alpha_max=10)

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
            coef,c_llk,c_pen_llk,a = correct_coef_step_gen_smooth(family,y,Xs,coef,next_coef,coef_split_idx,c_llk,S_emb,a)

            # Very poor start estimate, restart
            if grad_only and outer == 0 and inner <= 20 and np.abs(c_pen_llk - prev_llk_cur_pen) < conv_tol*np.abs(c_pen_llk):
               coef, _, _ = restart_coef(coef,None,None,len(coef),coef_split_idx,y,Xs,S_emb,family,inner,0)
               c_llk = family.llk(coef,coef_split_idx,y,Xs)
               c_pen_llk = c_llk - 0.5*coef.T@S_emb@coef
               a = 0.1

            # Check if this step would converge, if that is the case try gradient first
            if method == 'qEFS' and np.abs(c_pen_llk - prev_llk_cur_pen) < conv_tol*np.abs(c_pen_llk):

               alpha_pen_grad = back_track_alpha(prev_coef,pgrad,__neg_llk,__neg_grad,coef_split_idx,y,Xs,family,S_emb,alpha_max=1)

               if alpha_pen_grad is not None:

                  # Test llk for improvement over quasi newton step
                  grad_pen_llk = family.llk(prev_coef + alpha_pen_grad*pgrad,coef_split_idx,y,Xs) - 0.5*(prev_coef + alpha_pen_grad*pgrad).T@S_emb@(prev_coef + alpha_pen_grad*pgrad)

                  if grad_pen_llk > c_pen_llk:
                     next_coef = prev_coef + alpha_pen_grad*pgrad
                     coef,c_llk,c_pen_llk,a = correct_coef_step_gen_smooth(family,y,Xs,prev_coef,next_coef,coef_split_idx,prev_llk,S_emb,a)

            if np.abs(c_pen_llk - prev_llk_cur_pen) < conv_tol*np.abs(c_pen_llk) and (method != "qEFS" or (opt["nit"] == 1 and outer > 0) or (updates >= maxcor)):
               converged = True
               break

            if eps <= 0 and outer > 0 and inner >= (min_inner-1):
               break # end inner loop and immediately optimize lambda again.
         else:
            # Simply accept next coef step on first iteration
            if grad_only:
               coef = gd_coef_smooth(coef,grad,S_emb,a)
            elif method == "qEFS":
               # Update coefficients via implicit quasi newton step
               if sks.shape[0] > 0:
                  pen_step = pV0@pgrad - t1@t2@t3@pgrad
               else:
                  pen_step = pV0@pgrad
               coef = coef + pen_step
               updates += 1
            else:
               coef,L,LV,eps = newton_coef_smooth(coef,grad,H,S_emb)

            c_llk = family.llk(coef,coef_split_idx,y,Xs)
            c_pen_llk = c_llk - 0.5*coef.T@S_emb@coef
   
      if method != 'qEFS':
         # Make sure at convergence negative Hessian of llk is at least positive semi-definite as well
         checkH = True
         checkHc = 0
         
         while checkH and (smooth_pen is not None):
            _,S_pinv,_,FS_use_rank = compute_S_emb_pinv_det(len(coef),smooth_pen,"svd")

            # Re-compute lgdetDs, ldetHS
            lgdetDs = []

            for lti,lTerm in enumerate(smooth_pen):

                  lt_rank = None
                  if FS_use_rank[lti]:
                     lt_rank = lTerm.rank

                  lgdetD,_ = compute_lgdetD_bsb(lt_rank,lTerm.lam,S_pinv,lTerm.S_J_emb,coef)
                  lgdetDs.append(lgdetD)

            _,_, ldetHSs = calculate_edf(None,None,LV,smooth_pen,lgdetDs,len(coef),10,None)

            # And check whether Theorem 1 now holds
            checkH = np.any([lgdetDs[lti] - ldetHSs[lti] < 0 for lti in range(len(smooth_pen))])

            if checkH:
               if eps == 0:
                  eps += 1e-14
               else:
                  eps *= 2
               _,L,LV,_ = newton_coef_smooth(coef,grad,H - eps*scp.sparse.identity(H.shape[1],format='csc'),S_emb)

            if checkHc > 30:
               break

            checkHc += 1
         

   # In case we converged check for unidentifiable parameters, as reccomended by Wood. et al (2016)
   keep = None
   drop = None
   if (grad_only == False) and keep_drop is not None or (method == "QR/Chol" and eps > 0 and converged):
      
      if keep_drop is not None:
         keep = keep_drop[0]
         drop = keep_drop[1]
      else:
         # Check for drop
         keep,drop = identify_drop(H,S_norm)

      if len(drop) == 0:
         keep = None
         drop = None
      elif keep_drop is None:
         # Found drop, but need to check whether it is safe
         drop_valid,drop_pen_llk = check_drop_valid_gensmooth(y,coef,Xs,S_emb,keep,family)

         # Skip if likelihood becomes invalid
         if drop_valid == False:
            keep = None
            drop = None
         # If the drop makes pen llk worse by a lot also skip it
         elif drop_valid and np.abs(drop_pen_llk[0,0]) > 10*np.abs(c_pen_llk[0,0]):
            keep = None
            drop = None
         # If we identify all or all but one coefficients to drop also skip
         elif drop_valid and len(drop) >= len(coef) - 1:
            keep = None
            drop = None
      
      #print(drop)
      # Now we need to continue iterating in smaller problem until convergence
      if drop is not None:
         # Prepare zeroed full coef vector
         full_coef = np.zeros_like(coef)

         # Drop from coef
         coef = coef[keep]

         # ... from Xs...
         rXs,rcoef_split_idx = drop_terms_X(Xs,keep)
         #print(rXs)

         # ... and from S_emb
         rS_emb = S_emb[keep,:]
         rS_emb = rS_emb[:,keep]

         # Re-compute llk
         c_llk = family.llk(coef,rcoef_split_idx,y,rXs)
         c_pen_llk = c_llk - 0.5*coef.T@rS_emb@coef
         
         # and now repeat Newton iteration
         for inner in range(max_inner):
      
            # Get llk derivatives with respect to coef
            grad = family.gradient(coef,rcoef_split_idx,y,rXs)
            H = family.hessian(coef,rcoef_split_idx,y,rXs)

            # Update Coefficients
            next_coef,L,LV,eps = newton_coef_smooth(coef,grad,H,rS_emb)

            # Prepare to check convergence
            prev_llk_cur_pen = c_llk - 0.5*coef.T@rS_emb@coef

            # Perform step length control
            coef,c_llk,c_pen_llk,a = correct_coef_step_gen_smooth(family,y,rXs,coef,next_coef,rcoef_split_idx,c_llk,rS_emb,a)

            if np.abs(c_pen_llk - prev_llk_cur_pen) < conv_tol*np.abs(c_pen_llk):
               break

         #print("eps",eps)
         # converged on smaller problem now adjust return objects for dropped coef

         # start with coef
         full_coef[keep] = coef

         # Again make sure negative Hessian of llk is at least positive semi-definite as well
         checkH = True
         if smooth_pen is not None:
            drop_pen = drop_terms_S(copy.deepcopy(smooth_pen),keep)
         LVdat,LVrow,LVcol = translate_sparse(LV) # LV.T@LV = V
         LVrow = keep[LVrow]
         LVcol = keep[LVcol]
         LV = scp.sparse.csc_array((LVdat,(LVrow,LVcol)),shape=(len(full_coef),len(full_coef)))

         checkHc = 0
         
         while checkH and (smooth_pen is not None):
            _,S_pinv,_,FS_use_rank = compute_S_emb_pinv_det(len(full_coef),drop_pen,"svd")

            # Now re-compute lgdetDs, ldetHS
            lgdetDs = []

            for lti,lTerm in enumerate(drop_pen):

                  lt_rank = None
                  if FS_use_rank[lti]:
                     lt_rank = lTerm.rank

                  lgdetD,_ = compute_lgdetD_bsb(lt_rank,lTerm.lam,S_pinv,lTerm.S_J_emb,full_coef)
                  lgdetDs.append(lgdetD)

            _,_, ldetHSs = calculate_edf(None,None,LV,drop_pen,lgdetDs,len(full_coef),10,None)

            checkH = np.any([lgdetDs[lti] - ldetHSs[lti] < 0 for lti in range(len(drop_pen))])

            if checkH:
               if eps == 0:
                  eps += 1e-14
               else:
                  eps *= 2
               _,L,LV,_ = newton_coef_smooth(coef,grad,H - eps*scp.sparse.identity(H.shape[1],format='csc'),rS_emb)
               LVdat,LVrow,LVcol = translate_sparse(LV) # LV.T@LV = V
               LVrow = keep[LVrow]
               LVcol = keep[LVcol]
               LV = scp.sparse.csc_array((LVdat,(LVrow,LVcol)),shape=(len(full_coef),len(full_coef)))
            
            if checkHc > 30:
               break

            checkHc += 1
         

         coef = full_coef

         # Now H, L
         Hdat,Hrow,Hcol = translate_sparse(H)
         Ldat,Lrow,Lcol = translate_sparse(L.tocsc()) # L@L.T = H_pen

         Hrow = keep[Hrow]
         Hcol = keep[Hcol]
         Lrow = keep[Lrow]
         Lcol = keep[Lcol]

         H = scp.sparse.csc_array((Hdat,(Hrow,Hcol)),shape=(len(full_coef),len(full_coef)))
         L = scp.sparse.csc_array((Ldat,(Lrow,Lcol)),shape=(len(full_coef),len(full_coef)))
         #print((LV.T@LV - V).max(),(LV.T@LV - V).min())
         #print((L@L.T - nH).max(),(L@L.T - nH).min())
   
   return coef,H,L,LV,c_llk,c_pen_llk,eps,keep,drop

def correct_lambda_step_gen_smooth(family,y,Xs,S_norm,n_coef,coef,
                                    coef_split_idx,smooth_pen,lam_delta,
                                    extend_by,was_extended,c_llk,fit_info,outer,
                                    max_inner,min_inner,conv_tol,gamma,method,qEFSH,overwrite_coef,qEFS_init_converge,optimizer,
                                    __old_opt,use_grad,__neg_pen_llk,__neg_pen_grad,piv_tol,keep_drop,extend_lambda,
                                    extension_method_lam,control_lambda,n_c,
                                    init_bfgs_options,bfgs_options):
   # Fitting iteration and step size control for smoothing parameters of general smooth model.
   # Basically a more general copy of the function for gammlss. Again, step-size control is not obvious - because we have only approximate REMl
   # and approximate derivative, because we drop the last term involving the derivative of the negative penalized
   # Hessian with respect to the smoothing parameters (see section 4 in Wood & Fasiolo, 2017). However, what we
   # can do is at least undo the acceleration if we over-shoot the approximate derivative...

   lam_accepted = False
   while lam_accepted == False:
      
      # Build new penalties
      S_emb,S_pinv,_,FS_use_rank = compute_S_emb_pinv_det(n_coef,smooth_pen,"svd")

      # Then re-compute coef
      if optimizer == "Newton":
         
         # Optimize un-penalized problem first to get a good starting estimate for hessian.
         if method == "qEFS" and outer == 0:
            if qEFS_init_converge:
               opt_raw = scp.optimize.minimize(__neg_pen_llk,
                                             np.ndarray.flatten(coef),
                                             args=(coef_split_idx,y,Xs,family,scp.sparse.csc_matrix((len(coef), len(coef)))),
                                             method="L-BFGS-B",
                                             jac = __neg_pen_grad if use_grad else None,
                                             options={"maxiter":max_inner,
                                                      **init_bfgs_options})
               #print(opt_raw)
               if opt_raw["nit"] > 1:
                  # Set "initial" coefficients to solution found for un-penalized problem.
                  if overwrite_coef:
                     coef = opt_raw["x"].reshape(-1,1)
                     c_llk = family.llk(coef,coef_split_idx,y,Xs)

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

            
         next_coef,H,L,LV,next_llk,next_pen_llk,eps,keep,drop = update_coef_gen_smooth(family,y,Xs,coef,
                                                                                       coef_split_idx,S_emb,
                                                                                       S_norm,smooth_pen,
                                                                                       c_llk,outer,max_inner,
                                                                                       min_inner,conv_tol,
                                                                                       method,piv_tol,keep_drop,
                                                                                       __old_opt)
         
         if method == "qEFS" and outer == 0 and __old_opt.init == False:
            __old_opt = LV
            __old_opt.bfgs_options = bfgs_options
            __old_opt.init = True


         V = None
            
         if drop is not None:
            
            # Re-compute penalty matrices in smaller problem space.
            old_pen = copy.deepcopy(smooth_pen)
            smooth_pen = drop_terms_S(smooth_pen,keep)

            S_emb,S_pinv,_,FS_use_rank = compute_S_emb_pinv_det(n_coef,smooth_pen,"svd")

      else:
         raise DeprecationWarning("Non-Newton optimizers are deprecated.")
      
      # Now re-compute lgdetDs, ldetHS, and bsbs
      lgdetDs = []
      bsbs = []
      for lti,lTerm in enumerate(smooth_pen):

            lt_rank = None
            if FS_use_rank[lti]:
               lt_rank = lTerm.rank

            lgdetD,bsb = compute_lgdetD_bsb(lt_rank,lTerm.lam,S_pinv,lTerm.S_J_emb,next_coef)
            lgdetDs.append(lgdetD)
            bsbs.append(bsb*gamma)

      total_edf,term_edfs, ldetHSs = calculate_edf(None,None,LV,smooth_pen,lgdetDs,n_coef,n_c,None)
      fit_info.lambda_updates += 1

      if drop is not None:
         smooth_pen = old_pen

      # Can exit loop here, no extension and no control
      if outer == 0  or (control_lambda < 1) or (extend_lambda == False and control_lambda < 2):
         lam_accepted = True
         continue
      
      # Compute approximate!!! gradient of REML with respect to lambda
      # to check if step size needs to be reduced (part of step 6 in Wood, 2017).
      lam_grad = [grad_lambda(lgdetDs[lti],ldetHSs[lti],bsbs[lti],1) for lti in range(len(smooth_pen))]
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
   
   # At this point we have accepted the lambda step and can propose a new one!
   if drop is not None:
      fit_info.dropped = drop
   else:
      fit_info.dropped = None

   # For qEFS we check whether new approximation results in worse balance of efs update - then we fall back to previous approximation
   if method == "qEFS":
      #if outer > 0:
      total_edf2,term_edfs2, ldetHSs2 = calculate_edf(None,None,__old_opt,smooth_pen,lgdetDs,n_coef,n_c,None)
      diff1 = [np.abs((lgdetDs[lti] - ldetHSs[lti]) - bsbs[lti]) for lti in range(len(smooth_pen))]
      diff2 = [np.abs((lgdetDs[lti] - ldetHSs2[lti]) - bsbs[lti]) for lti in range(len(smooth_pen))]
      #print([(lgdetDs[lti] - ldetHSs[lti]) - bsbs[lti] for lti in range(len(smooth_pen))])
      #print([(lgdetDs[lti] - ldetHSs2[lti]) - bsbs[lti] for lti in range(len(smooth_pen))])
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

   return next_coef,H,L,LV,V,next_llk,next_pen_llk,__old_opt,keep,drop,S_emb,smooth_pen,total_edf,term_edfs,lam_delta

def solve_generalSmooth_sparse(family,y,Xs,form_n_coef,coef,coef_split_idx,smooth_pen,
                              max_outer=50,max_inner=50,min_inner=50,conv_tol=1e-7,
                              extend_lambda=True,extension_method_lam = "nesterov2",
                              control_lambda=True,optimizer="Newton",method="Chol",
                              check_cond=1,piv_tol=0.175,should_keep_drop=True,
                              form_VH=True,use_grad=False,gamma=1,qEFSH='SR1',
                              overwrite_coef=True,max_restarts=0,qEFS_init_converge=True,prefit_grad=False,progress_bar=True,
                              n_c=10,init_bfgs_options={"gtol":1e-9,"ftol":1e-9,"maxcor":30,"maxls":100,"maxfun":1e7},
                              bfgs_options={"gtol":1e-9,"ftol":1e-9,"maxcor":30,"maxls":100,"maxfun":1e7}):
    """
    Fits a general smooth model, following steps outlined by Wood, Pya, & Säfken (2016). Essentially,
    an even more general version of :func:``solve_gammlss_sparse`` that requires only a function to compute
    the log-likelihood, a function to compute the gradient of said likelihood with respect to the coefficients,
    and a function to compute the hessian of said likelihood with respect to the coefficients. In case computation
    of the hessian is too expensive, BFGS ("Broyden, Fletcher, Goldfarb, and Shanno algorithm", see; Nocedal & Wright; 2006)
    estimation can be substituted for the full Newton step. Note that even though the estimate of the inverse of the Hessian
    obtained from BFGS could be used for confidence interval computations (and model comparisons) this estimate will not
    always be close to the actual inverse of the Hessian - resulting in very poor coverage of the ground truth.

    References:

      - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
      - Nocedal & Wright (2006). Numerical Optimization. Springer New York.
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
    c_llk = family.llk(coef,coef_split_idx,y,Xs)
    c_pen_llk = c_llk - 0.5*coef.T@S_emb@coef

    __neg_pen_llk = None
    __neg_pen_grad = None
    if method == "qEFS":
      # Define negative penalized likelihood function to be minimized via BFGS
      # plus function to evaluate negative gradient of penalized likelihood - the
      # latter is only used if use_grad=True.
      def __neg_pen_llk(coef,coef_split_idx,y,Xs,family,S_emb):
         coef = coef.reshape(-1,1)
         neg_llk = -1 * family.llk(coef,coef_split_idx,y,Xs)
         return neg_llk + 0.5*coef.T@S_emb@coef
      
      def __neg_pen_grad(coef,coef_split_idx,y,Xs,family,S_emb):
         # see Wood, Pya & Saefken (2016)
         coef = coef.reshape(-1,1)
         grad = family.gradient(coef,coef_split_idx,y,Xs)
         pgrad = np.array([grad[i] - (S_emb[[i],:]@coef)[0] for i in range(len(grad))])
         return -1*pgrad.flatten()

    # Try improving start estimate via Gradient only
    if prefit_grad:
      coef,_,_,_,c_llk,c_pen_llk,_,_,_ = update_coef_gen_smooth(family,y,Xs,coef,
                                                                  coef_split_idx,S_emb,
                                                                  S_norm,None,
                                                                  c_llk,0,max_inner,
                                                                  min_inner,conv_tol,
                                                                  "Grad",piv_tol,None,
                                                                  None)

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
      total_edf,term_edfs,lam_delta = correct_lambda_step_gen_smooth(family,y,Xs,S_norm,n_coef,coef,
                                                                           coef_split_idx,smooth_pen,lam_delta,
                                                                           extend_by,was_extended,c_llk,fit_info,outer,
                                                                           max_inner,min_inner,conv_tol,gamma,method,qEFSH,overwrite_coef,
                                                                           qEFS_init_converge,optimizer,
                                                                           __old_opt,use_grad,__neg_pen_llk,__neg_pen_grad,
                                                                           piv_tol,keep_drop,extend_lambda,
                                                                           extension_method_lam,control_lambda,n_c,
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
            coef, c_llk, c_pen_llk = restart_coef(coef,c_llk,c_pen_llk,n_coef,coef_split_idx,y,Xs,S_emb,family,outer,restart_counter)
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
                  coef, c_llk, c_pen_llk = restart_coef(coef,c_llk,c_pen_llk,n_coef,coef_split_idx,y,Xs,S_emb,family,outer,restart_counter)
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

    if method == "qEFS":
        
        # Optionally form last V + Chol explicitly during last iteration
        # when working with qEFS update
         if form_VH:
            
            # Get an approximation of the Hessian of the likelihood
            if LV.form == 'SR1':
               H = -1*computeHSR1(LV.sk,LV.yk,LV.rho,scp.sparse.identity(len(coef),format='csc')*LV.omega,omega=LV.omega,make_psd=True,make_pd=True)
            else:
               H = -1*computeH(LV.sk,LV.yk,LV.rho,scp.sparse.identity(len(coef),format='csc')*LV.omega,omega=LV.omega,make_psd=True)
            
            H = scp.sparse.csc_array(H)

            # Get Cholesky factor of inverse of penalized hessian (needed for CIs)
            pH = scp.sparse.csc_array((-1*H) + S_emb)
            Lp, Pr, _ = cpp_cholP(pH)
            P = compute_eigen_perm(Pr)
            LVp0 = compute_Linv(Lp,10)
            LV = apply_eigen_perm(Pr,LVp0)
            L = P.T@Lp

         else:
            H = None # Do not approximate H.
       
    if check_cond == 1 and (method != "qEFS" or form_VH):
      K2,_,_,Kcode = est_condition(L,LV,verbose=False)

      fit_info.K2 = K2

      if fit_info.code == 0: # Convergence was reached but Knumber might suggest instable system.
         fit_info.code = Kcode

      if Kcode > 0:
         warnings.warn(f"Condition number ({K2}) of matrix A, where A.T@A=H and H is the Hessian of the negative penalized likelihood, is larger than 1/sqrt(u), where u is half the machine precision. Call ``model.fit()`` with ``method='QR/Chol'``, but note that even then estimates are likely to be inaccurate.")

    return coef,H,LV,total_edf,term_edfs,penalty[0,0],smooth_pen,fit_info
