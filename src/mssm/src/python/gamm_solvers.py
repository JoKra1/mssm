import numpy as np
import scipy as scp
import warnings
from .exp_fam import Family,Gaussian,est_scale,GAMLSSFamily,Identity
from .penalties import PenType,id_dist_pen,translate_sparse,dataclass
from .formula import build_sparse_matrix_from_formula,setup_cache,clear_cache,cpp_solvers,pd,Formula,mp,repeat,os,map_csc_to_eigen,map_csr_to_eigen,math,tqdm,sys,copy,embed_in_S_sparse
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
   min_sing = scp.sparse.linalg.svds(Linv,k=1,return_singular_vectors=False,random_state=seed)[0]
   max_sing = scp.sparse.linalg.svds(L,k=1,return_singular_vectors=False,random_state=seed)[0]
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
   dy1 = family.link.dy1(mu)
   z = dy1 * (y - mu) + eta
   w = 1 / (dy1**2 * family.V(mu))
   return z, w

def update_PIRLS(y,yb,mu,eta,X,Xb,family):
   # Update the PIRLS weights and data (if the model is not Gaussian)
   # and update the fitting matrices yb & Xb
   z = None
   Wr = None

   if isinstance(family,Gaussian) == False or isinstance(family.link,Identity) == False:
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

def compute_block_B_shared(address_dat,address_ptr,address_idx,shape_dat,shape_ptr,rows,cols,nnz,T):
   BB = compute_block_linv_shared(address_dat,address_ptr,address_idx,shape_dat,shape_ptr,rows,cols,nnz,T)
   return BB.power(2).sum()

def compute_block_B_shared_cluster(address_dat,address_ptr,address_idx,shape_dat,shape_ptr,rows,cols,nnz,T,cluster_weights):
   BB = compute_block_linv_shared(address_dat,address_ptr,address_idx,shape_dat,shape_ptr,rows,cols,nnz,T)
   BBps = BB.power(2).sum()
   return np.sum(cluster_weights*BBps),len(cluster_weights)*BBps
   
def compute_B(L,P,lTerm,n_c=10):
   # Solves L @ B = P @ D for B, parallelizing over column
   # blocks of D if int(D.shape[1]/2000) > 1

   # Also allows for approximate B computation for very big factor smooths.
   D_start = lTerm.start_index

   if lTerm.clust_series is None:
      
      col_sums = lTerm.S_J.sum(axis=0)
      if lTerm.type == PenType.NULL and sum(col_sums[col_sums > 0]) == 1:
         # Null penalty for factor smooth has usually only non-zero element in a single colum,
         # so we only need to solve one linear system per level of the factor smooth.
         NULL_idx = np.argmax(col_sums)

         D_NULL_idx = np.arange(lTerm.start_index+NULL_idx,
                                lTerm.S_J.shape[1]*(lTerm.rep_sj+1),
                                lTerm.S_J.shape[1])

         D_len = len(D_NULL_idx)
         PD = P @ lTerm.D_J_emb[:,D_NULL_idx]
      else:
         D_len = lTerm.rep_sj * lTerm.S_J.shape[1]
         D_end = lTerm.start_index + D_len
         PD = P @ lTerm.D_J_emb[:,D_start:D_end]

      D_r = int(D_len/2000)

      if D_r > 1 and n_c > 1:
         # Parallelize over column blocks of P @ D
         # Can speed up computations considerably and is feasible memory-wise
         # since L itself is super sparse.
         n_c = min(D_r,n_c)
         split = np.array_split(range(D_len),n_c)
         PD = P @ lTerm.D_J_emb
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
   

def calculate_edf(LP,Pr,InvCholXXS,penalties,lgdetDs,colsX,n_c):
   # Follows steps outlined by Wood & Fasiolo (2017) to compute total degrees of freedom by the model.
   # Generates the B matrix also required for the derivative of the log-determinant of X.T@X+S_\lambda. This
   # is either done exactly - as described by Wood & Fasiolo (2017) - or approximately. The latter is much faster
   total_edf = colsX
   Bs = []
   term_edfs = []

   if (not InvCholXXS is None) and isinstance(InvCholXXS,scp.sparse.linalg.LinearOperator):
      # Following prepares trace computation via equation 3.13 in Byrd, Nocdeal & Schnabel (1992).
      # First get m (number of implicit hessian updates), yk, sk, and rho from V
      s, y, rho, m = InvCholXXS.sk, InvCholXXS.yk, InvCholXXS.rho, InvCholXXS.n_corrs

      if m == 0: # L-BFGS routine converged after first step

         S = None
         Y = None
         t2 = None

      else:

         # Now form S,Y, and D - only have to do this once
         S = np.array(s).T
         Y = np.array(y).T
         DYTY = Y.T@Y
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
         
         # Now pre-compute term 2 in 3.13 used for all S_j
         t2 = np.zeros((2*m,2*m))
         t2[:m,:m] = Rinv.T@DYTY@Rinv
         t2[:m,m:] = -Rinv.T
         t2[m:,:m] = -Rinv


   for lti,lTerm in enumerate(penalties):
      if not InvCholXXS is None:
         # Compute B, needed for Fellner Schall update (Wood & Fasiolo, 2017)
         if isinstance(InvCholXXS,scp.sparse.linalg.LinearOperator):
            Bps = computetrVS2(t2,S,Y,lTerm)
         else:
            B = InvCholXXS @ lTerm.D_J_emb 
            Bps = B.power(2).sum()
      else:
         Bps = compute_B(LP,compute_eigen_perm(Pr),lTerm,n_c)

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

def update_scale_edf(y,z,eta,Wr,rowsX,colsX,LP,InvCholXXSP,Pr,lgdetDs,family,penalties,n_c):
   # Updates the scale of the model. For this the edf
   # are computed as well - they are returned because they are needed for the
   # lambda step proposal anyway.
   
   # Calculate Pearson residuals for GAMM (Wood, 3.1.5 & 3.1.7)
   # Standard residuals for AMM
   if isinstance(family,Gaussian) == False or isinstance(family.link,Identity) == False:
      wres = Wr @ (z - eta)
   else:
      wres = y - eta

   # Calculate total and term wise edf
   InvCholXXS = None
   if not InvCholXXSP is None:
      InvCholXXS = apply_eigen_perm(Pr,InvCholXXSP)

   # If there are penalized terms we need to adjust the total_edf
   if len(penalties) > 0:
      total_edf, term_edfs, Bs = calculate_edf(LP,Pr,InvCholXXS,penalties,lgdetDs,colsX,n_c)
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

def update_coef_and_scale(y,yb,z,Wr,rowsX,colsX,X,Xb,family,S_emb,S_root,S_pinv,FS_use_rank,penalties,n_c,formula,form_Linv):
   # Solves the additive model for a given set of weights and penalty
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

         # Can now unpivot coef
         coef = coef @ P

         # Now convert so that rest of code can just continue as with Chol
         LP = RP.T.tocsc()
         _,Pr,_ = translate_sparse(P.tocsc())

         if rank < S_emb.shape[1]:
            # Rank defficiency detected during numerical factorization.
            # Set code > 0
            warnings.warn(f"Rank deficiency detected. Most likely a result of coefficients {Pr[rank:]} being unidentifiable. Check 'model.formula.coef_names' at the corresponding indices to identify the problematic terms and consider dropping (or penalizing) them.")
            code = 1

   else:
      #yb is X.T@y and Xb is X.T@X
      LP, Pr, coef, code = cpp_solve_coefXX(yb,Xb + S_emb)
      P = compute_eigen_perm(Pr)

   if code != 0:
      raise ArithmeticError(f"Solving for coef failed with code {code}. Model is likely unidentifiable.")
   
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
   
   # Update mu & eta
   if formula is None:
      eta = (X @ coef).reshape(-1,1)
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
      mu = family.link.fi(eta)

   # Update scale parameter
   wres,InvCholXXS,total_edf,term_edfs,Bs,scale = update_scale_edf(y,z,eta,Wr,rowsX,colsX,LP,InvCholXXSP,Pr,lgdetDs,family,penalties,n_c)
   return eta,mu,coef,P.T@LP,InvCholXXS,lgdetDs,bsbs,total_edf,term_edfs,Bs,scale,wres

def init_step_gam(y,yb,mu,eta,rowsX,colsX,X,Xb,
                  family,col_S,penalties,
                  pinv,n_c,formula,form_Linv,
                  method):
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
   yb,Xb,z,Wr = update_PIRLS(y,yb,mu,eta,X,Xb,family)
   
   # Solve additive model
   eta,mu,coef,\
   CholXXS,\
   InvCholXXS,\
   lgdetDs,\
   bsbs,\
   total_edf,\
   term_edfs,\
   Bs,scale,wres = update_coef_and_scale(y,yb,z,Wr,rowsX,colsX,
                                         X,Xb,family,S_emb,S_root,S_pinv,
                                         FS_use_rank,penalties,n_c,
                                         formula,form_Linv)
   
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

         dLam = step_fellner_schall_sparse(lgdetDs[lti],Bs[lti],bsbs[lti],lTerm.lam,scale)
         lam_delta.append(dLam)

      lam_delta = np.array(lam_delta).reshape(-1,1)
   
   return dev,pen_dev,eta,mu,coef,CholXXS,InvCholXXS,total_edf,term_edfs,scale,wres,lam_delta,S_emb


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

      # Step halving
      n_coef = (coef + n_coef)/2

      # Update mu & eta for correction
      # Note, Wood (2017) show pseudo-data and weight computation in
      # step 1 - which should be re-visited after the correction, but because
      # mu and eta can change during the correction (due to step halving) and neither
      # the pseudo-data nor the weights are necessary to compute the deviance it makes
      # sense to only compute these once **after** we have completed the coef corrections.
      if formula is None:
         eta = (X @ n_coef).reshape(-1,1)
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

def correct_lambda_step(y,yb,z,Wr,rowsX,colsX,X,Xb,
                        family,col_S,S_emb,penalties,
                        was_extended,pinv,lam_delta,
                        extend_by,o_iter,dev_check,n_c,
                        control_lambda,extend_lambda,
                        exclude_lambda,extension_method_lam,
                        formula,form_Linv,method):
   # Propose & perform step-length control for the lambda parameters via the Fellner Schall method
   # by Wood & Fasiolo (2016)
   lam_accepted = False
   lam_checks = 0
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
      Bs,scale,wres = update_coef_and_scale(y,yb,z,Wr,rowsX,colsX,
                                             X,Xb,family,S_emb,S_root,S_pinv,
                                             FS_use_rank,penalties,n_c,
                                             formula,form_Linv)
      
      # Compute gradient of REML with respect to lambda
      # to check if step size needs to be reduced (part of step 6 in Wood, 2017).
      lam_grad = [grad_lambda(lgdetDs[lti],Bs[lti],bsbs[lti],scale) for lti in range(len(penalties))]
      lam_grad = np.array(lam_grad).reshape(-1,1) 
      check = lam_grad.T @ lam_delta

      # For Generalized models we should not reduce step beyond original EFS update since
      # criterion maximized is approximate REML. Also, we can probably relax the criterion a
      # bit, since check will quite often be < 0.
      check_criterion = 0
      if (isinstance(family,Gaussian) == False) or (isinstance(family.link,Identity) == False):
            
            if dev_check is not None:
               check_criterion = 1e-7*-abs(dev_check)
            
            if check[0,0] < check_criterion: 
               # Now check whether we extend lambda - and if we do so whether any extension was actually applied.
               # If not, we still "pass" the check.
               if (extend_lambda == False) or (np.any(was_extended) == False):
                  check[0,0] = check_criterion + 1

      # Now check whether we have to correct lambda.
      # Because of minimization in Wood (2017) they use a different check (step 7) but idea is the same.
      if check[0,0] < check_criterion and control_lambda: 
         # Reset extension or cut the step taken in half (for additive models)
         for lti,lTerm in enumerate(penalties):

            # Reset extension factor for all terms that were extended.
            if extend_lambda and was_extended[lti]:
               lam, dLam = undo_extension_lambda_step(lti,lTerm.lam,lam_delta[lti][0],extend_by,was_extended, extension_method_lam, family)
               lTerm.lam = lam
               lam_delta[lti][0] = dLam

            # For Gaussian models only, rely on the strategy by Wood & Fasiolo (2016) to just half the step for additive models 
            elif isinstance(family,Gaussian) and isinstance(family.link,Identity):
               lam_delta[lti] = lam_delta[lti]/2
               lTerm.lam -= lam_delta[lti][0]
         
      else:
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

               if extend_lambda:
                  dLam,extend_by,was_extended = extend_lambda_step(lti,lTerm.lam,dLam,extend_by,was_extended,extension_method_lam)

            else: # ReLikelihood is probably insensitive to further changes in this smoothing penalty, so set change to 0.
               dLam = 0
               was_extended[lti] = False

            lam_delta.append(dLam)

         lam_delta = np.array(lam_delta).reshape(-1,1)

      lam_checks += 1

   return eta,mu,n_coef,CholXXS,InvCholXXS,total_edf,term_edfs,scale,wres,lam_delta,extend_by,penalties,was_extended,S_emb,lam_checks

################################################ Main solver ################################################

def solve_gamm_sparse(mu_init,y,X,penalties,col_S,family:Family,
                      maxiter=10,pinv="svd",conv_tol=1e-7,
                      extend_lambda=True,control_lambda=True,
                      exclude_lambda=False,extension_method_lam = "nesterov",
                      form_Linv=True,method="Chol",check_cond=2,progress_bar=False,n_c=10):
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
                                                                                                     pinv,n_c,None,form_Linv,method)
   
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
         dev = family.deviance(y,mu) 
         pen_dev = dev
         c_dev_prev = prev_dev

         if len(penalties) > 0:
            pen_dev += n_coef.T @ S_emb @ n_coef
            c_dev_prev += coef.T @ S_emb @ coef

         # Perform step-length control for the coefficients (Step 3 in Wood, 2017)
         dev,pen_dev,mu,eta,coef = correct_coef_step(coef,n_coef,dev,pen_dev,c_dev_prev,family,eta,mu,y,X,len(penalties),S_emb,None,n_c)

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

      # Update pseudo-dat weights for next coefficient step (step 1 in Wood, 2017; but moved after the coef correction because z and Wr depend on
      # mu and eta, which change during the correction but anything that needs to be computed during the correction (deviance) does not depend on
      # z and Wr).
      yb,Xb,z,Wr = update_PIRLS(y,yb,mu,eta,X,Xb,family)

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
         dev_check = None
         if o_iter > 0:
            dev_check = pen_dev

         eta,mu,n_coef,CholXXS,InvCholXXS,total_edf,term_edfs,scale,wres,lam_delta,extend_by,penalties,was_extended,S_emb,lam_checks = correct_lambda_step(y,yb,z,Wr,rowsX,colsX,X,Xb,
                                                                                                                                                   family,col_S,S_emb,penalties,
                                                                                                                                                   was_extended,pinv,lam_delta,
                                                                                                                                                   extend_by,o_iter,dev_check,n_c,
                                                                                                                                                   control_lambda,extend_lambda,
                                                                                                                                                   exclude_lambda,extension_method_lam,
                                                                                                                                                   None,form_Linv,method)
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
         _,scale,wres = update_coef_and_scale(y,yb,z,Wr,rowsX,colsX,
                                              X,Xb,family,S_emb,None,None,None,
                                              penalties,n_c,None,form_Linv)

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

   if InvCholXXS is None:
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

   return coef,eta,wres,Wr,scale,InvCholXXS,total_edf,term_edfs,penalty,fit_info

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
                                                                                                     pinv,n_c,formula,form_Linv,"Chol")
   
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
         dev,pen_dev,mu,eta,coef = correct_coef_step(coef,n_coef,dev,pen_dev,c_dev_prev,family,eta,mu,y,None,len(penalties),S_emb,formula,n_c)

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
         dev_check = None
         if o_iter > 0:
            dev_check = pen_dev

         eta,mu,n_coef,CholXXS,InvCholXXS,total_edf,term_edfs,scale,wres,lam_delta,extend_by,penalties,was_extended,S_emb,lam_checks = correct_lambda_step(y,Xy,None,None,rowsX,colsX,None,XX,
                                                                                                                                                   family,col_S,S_emb,penalties,
                                                                                                                                                   was_extended,pinv,lam_delta,
                                                                                                                                                   extend_by,o_iter,dev_check,
                                                                                                                                                   n_c,control_lambda,extend_lambda,
                                                                                                                                                   exclude_lambda,extension_method_lam,
                                                                                                                                                   formula,form_Linv,"Chol")
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
         _,scale,wres = update_coef_and_scale(y,Xy,None,None,rowsX,colsX,
                                              None,XX,family,S_emb,None,None,None,
                                              penalties,n_c,formula,form_Linv)

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

def deriv_transform_mu_eta(y,means,family:GAMLSSFamily):
    """
    Compute derivatives (first and second order) of llk with respect to each mean for all observations following steps outlined by Wood, Pya, & Säfken (2016)

    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
    """
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
    d1eta = [d1[mui]/ld1[mui] for mui in range(len(means))]

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
    d2eta = [d2[mui]/np.power(ld1[mui],2) - d1[mui]*ld2[mui]/np.power(ld1[mui],3) for mui in range(len(means))]

    # Mixed second derivatives thus also are transformed as proposed by WPS (2016)
    d2meta = []
    mixed_idx = 0
    for mui in range(len(means)):
        for muj in range(len(means)):
            if muj <= mui:
                continue
            
            d2meta.append(d2m[mixed_idx] * (1/ld1[mui]) * (1/ld1[muj]))
            mixed_idx += 1

    return d1eta,d2eta,d2meta

def deriv_transform_eta_beta(d1eta,d2eta,d2meta,Xs):
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
        grad.extend((d1eta[etai].T @ Xs[etai]).T)

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

            for coefi in range(Xs[etai].shape[1]):
                # More efficient computation now, no longer an additional nested loop over coefj..
                d2beta = (d2*Xs[etai][:,[coefi]]).T @ Xs[etaj][:,coefi:Xs[etaj].shape[1]]
                
                if d2beta.nnz > 0:
                  
                  # Sort, to make symmetric deriv extraction easier
                  if not d2beta.has_sorted_indices:
                     d2beta.sort_indices()
                  
                  # Get non-zero column entries for current row
                  cols = d2beta.indices[d2beta.indptr[0]:d2beta.indptr[1]] + coefi
                  
                  # Get non-zero values for current row in sorted order
                  vals = d2beta.data[d2beta.indptr[0]:d2beta.indptr[1]]

                  h_rows.extend(np.tile(coefi,d2beta.nnz) + hr_idx)
                  h_cols.extend(cols + hc_idx)
                  h_vals.extend(vals)

                  # Symmetric 2nd deriv..
                  if (cols[0] + hc_idx) == (coefi + hr_idx):
                     h_rows.extend(cols[1:] + hc_idx)
                     h_cols.extend(np.tile(coefi,d2beta.nnz-1) + hr_idx)
                     h_vals.extend(vals[1:])
                  else:
                     h_rows.extend(cols + hc_idx)
                     h_cols.extend(np.tile(coefi,d2beta.nnz) + hr_idx)
                     h_vals.extend(vals)

            hc_idx += Xs[etaj].shape[1]
        hr_idx += Xs[etai].shape[1]
    
    hessian = scp.sparse.csc_array((h_vals,(h_rows,h_cols)))
    return np.array(grad).reshape(-1,1),hessian

def newton_coef_smooth(coef,grad,H,S_emb,method,piv_tol):
    """
    Follows sections 3.1.2 and 3.14 in WPS (2016) to update the coefficients of the GAMLSS model via a
    newton step.
    1) Computes gradient of the penalized likelihood (grad - S_emb@coef)
    2) Computes negative Hessian of the penalized likelihood (-1*H + S_emb) and it's inverse.
    3) Uses these two to compute the Netwon step.
    4) Step size control - happens outside

    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
    """
    
    pgrad = np.array([grad[i] - (S_emb[[i],:]@coef)[0] for i in range(len(grad))])
    nH = -1*H + S_emb

    # Diagonal pre-conditioning as suggested by WPS (2016)
    nHdgr = nH.diagonal()
    nHdgr = np.power(np.abs(nHdgr),-0.5)
    D = scp.sparse.diags(nHdgr)
    DI = scp.sparse.diags(1/nHdgr) # For cholesky
    #D = scp.sparse.identity(nH.shape[1],format='csc')
    #DI = scp.sparse.identity(nH.shape[1],format='csc')
    nH2 = (D@nH@D).tocsc()
    #print(max(np.abs(nH.diagonal())),max(np.abs(nH2.diagonal())))

    if method != "Chol":
         # First perform QR decomposition with aggressive pivot tolerance
         
         _,Pr1,Pr2,rank,code = cpp_symqr(nH2,piv_tol)
         
         P1 = compute_eigen_perm(Pr1)
         P2 = compute_eigen_perm(Pr2)
         P = P2.T@P1.T

         _,Pr,_ = translate_sparse(P.tocsc())

         if code != 0:
            raise ArithmeticError("Computation of pre-cholesky QR decomposition failed. Model is likely miss-specified.")

    # Compute V, inverse of nH
    eps = 0
    code = 1
    while code != 0:
        
        if method == "Chol":
            Lp, Pr, code = cpp_cholP(nH2+eps*scp.sparse.identity(nH2.shape[1],format='csc'))
            P = compute_eigen_perm(Pr)
        else:
            # Now compute cholesky based on pivoting strategy inferred earlier from QR.
            pNH2 = (P@(nH2+eps*scp.sparse.identity(nH2.shape[1],format='csc'))@P.T).tocsc()
            Lp, code = cpp_chol(pNH2)

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

def correct_coef_step_gammlss(family,y,Xs,coef,next_coef,coef_split_idx,c_llk,S_emb):
    """
    Apply step size correction to Newton update for GAMLSS models, as discussed by WPS (2016).

    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
    """
    # Update etas and mus
    next_split_coef = np.split(next_coef,coef_split_idx)
    next_etas = [Xs[i]@next_split_coef[i] for i in range(family.n_par)]
    next_mus = [family.links[i].fi(next_etas[i]) for i in range(family.n_par)]
    
    # Step size control for newton step.
    next_llk = family.llk(y,*next_mus)
    
    # Evaluate improvement of penalized llk under new and old coef - but in both
    # cases for current lambda (see Wood, Li, Shaddick, & Augustin; 2017)
    next_pen_llk = next_llk - next_coef.T@S_emb@next_coef
    prev_llk_cur_pen = c_llk - coef.T@S_emb@coef
    n_checks = 0
    while next_pen_llk < prev_llk_cur_pen:
        if n_checks > 30:
            next_coef = coef
            
        # Half it if we do not observe an increase in penalized likelihood (WPS, 2016)
        next_coef = (coef + next_coef)/2
        next_split_coef = np.split(next_coef,coef_split_idx)

        # Update etas and mus again
        next_etas = [Xs[i]@next_split_coef[i] for i in range(family.n_par)]

        next_mus = [family.links[i].fi(next_etas[i]) for i in range(family.n_par)]
        
        # Re-evaluate penalized likelihood
        next_llk = family.llk(y,*next_mus)
        next_pen_llk = next_llk - next_coef.T@S_emb@next_coef
        n_checks += 1
    
    return next_coef,next_split_coef,next_mus,next_etas,next_llk,next_pen_llk

def identify_drop(H,S_scaled,method='LU'):
    """
    Routine to approximately identify the rank of the scaled negative hessian of the penalized likelihood based on Foster (1986) and Gotsman & Toledo (2008).

    Requires ``p`` QR/LU decompositions - where ``p`` is approximately the Kernel size of the matrix. Essentially continues to find
    vectors forming a basis of the Kernel of the matrix and successively drops columns corresponding to the maximum absolute value of the
    Kernel vectors. This is repeated until we can form a cholesky of the scaled hessian.

    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
     - Foster (1986). Rank and null space calculations using matrix decomposition without column interchanges.
     - Gotsman & Toledo (2008). On the Computation of Null Spaces of Sparse Rectangular Matrices.
    
    :param H: Estimate of the hessian of the log-likelihood.
    :type H: scipy.sparse.csc_array
    :param S_scaled: Scaled version of the penalty matrix (i.e., unweighted total penalty divided by it's norm).
    :type S_scaled: scipy.sparse.csc_array
    """

    rank = H.shape[1]
    
    # Form scaled negative hessian of penalized likelihood to check for rank deficincy, as
    # reccomended by Wood et al. (2016).
    H_scaled = H / scp.sparse.linalg.norm(H,ord=None)

    nH_scaled = -1*H_scaled + S_scaled

    nHdgr = nH_scaled.diagonal()
    nHdgr = np.power(np.abs(nHdgr),-0.5)
    D = scp.sparse.diags(nHdgr)
    nH_scaled = (D@nH_scaled@D).tocsc()

    # Verify that we actually have a problem..
    keep = [cidx for cidx in range(H.shape[1])]
    drop = []
    _, _, code = cpp_cholP(nH_scaled)

    if code == 0:
       return keep, drop
    
    nH_drop = copy.deepcopy(nH_scaled)

    # Now follows steps outlined in algorithm 1 of Foster:
    # - Form QR (or LU) decomposition of current matrix
    # - find approximate singular value (from R or U) + vector -> vector is approximate Kernel vector
    # - drop column of current (here also row since nH is symmetric) matrix corresponding to maximum of approximate Kernel vector
    # - check if cholesky works now, otherwise continue
    while True:

      # Find Null-vector of R (U) -> which is Nullvector of A@Pc (can ignore row pivoting for LU, see: Gotsman & Toledo; 2008)
      if method == "QR": # Original proposal by Foster
         R,Pr,rank,_  = cpp_qrr(nH_drop)
         P = compute_eigen_perm(Pr)

         # Deal with drops during factorization.
         if rank < nH_drop.shape[1]:
            drop.extend(Pr[rank:])
            keep = [cidx for cidx in range(H.shape[1]) if cidx not in drop]

      else: # Faster strategy motivated by Gotsman & Toledo
         lu = scp.sparse.linalg.splu(nH_drop,permc_spec='COLAMD',diag_pivot_thresh=1,options=dict(SymmetricMode=False,IterRefine='Double',Equil=True))
         R = lu.U
         P = scp.sparse.csc_matrix((np.ones(nH_drop.shape[1]), (np.arange(nH_drop.shape[1]), lu.perm_c)))

      # Find approximate Null-vector w of nH (See Foster, 1986)
      _,_,vh = scp.sparse.linalg.svds(R,k=1,return_singular_vectors=True,random_state=20,which='SM')
      
      # w needs to be of original shape!
      w = np.zeros(H.shape[1])
      wk = (vh@P.T).T # Need to undo column-pivoting here - depends on method used to compute pivot, here COLAMD for both.
      w[keep] = wk.flatten()

      # Drop next col + row and update keep list
      drop_c = np.argmax(np.abs(w))
      drop.append(drop_c)
      keep = [cidx for cidx in range(H.shape[1]) if cidx not in drop]
      #print(drop_c)

      nH_drop = nH_scaled[keep,:]
      nH_drop = nH_drop[:,keep]

      # Check if Cholesky works now - otherwise continue
      _, _, code = cpp_cholP(nH_drop)
      if code == 0:
         break

    drop = np.sort(drop)
    keep = np.sort(keep)

    #print(drop,rank)
    
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
    
def update_coef_gammlss(family,mus,y,Xs,coef,coef_split_idx,S_emb,S_norm,c_llk,outer,max_inner,min_inner,conv_tol,method,piv_tol,keep_drop):
   """
   Repeatedly perform Newton update with step length control to the coefficient vector - based on
   steps outlined by WPS (2016).

   References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
   """
   # Update coefficients:
   if keep_drop is None:
      converged = False
      for inner in range(max_inner):
         
         # Get derivatives with respect to eta
         d1eta,d2eta,d2meta = deriv_transform_mu_eta(y,mus,family)

         # Get derivatives with respect to coef
         grad,H = deriv_transform_eta_beta(d1eta,d2eta,d2meta,Xs)

         # Update coef and perform step size control
         if outer > 0 or inner > 0:
            # Update Coefficients
            next_coef,L,LV,eps = newton_coef_smooth(coef,grad,H,S_emb,method,piv_tol)

            # Prepare to check convergence
            prev_llk_cur_pen = c_llk - coef.T@S_emb@coef

            # Perform step length control
            coef,split_coef,mus,etas,c_llk,c_pen_llk = correct_coef_step_gammlss(family,y,Xs,coef,next_coef,coef_split_idx,c_llk,S_emb)

            if np.abs(c_pen_llk - prev_llk_cur_pen) < conv_tol*np.abs(c_pen_llk):
               converged = True
               if eps <= 0 or method == 'Chol':
                  break

            if eps <= 0 and outer > 0 and inner >= (min_inner-1):
               break # end inner loop and immediately optimize lambda again.
         else:
            # Simply accept next coef step on first iteration
            coef,L,LV,eps = newton_coef_smooth(coef,grad,H,S_emb,method,piv_tol)
            split_coef = np.split(coef,coef_split_idx)
            etas = [Xs[i]@split_coef[i] for i in range(family.n_par)]
            mus = [family.links[i].fi(etas[i]) for i in range(family.n_par)]
            c_llk = family.llk(y,*mus)
            c_pen_llk = c_llk - coef.T@S_emb@coef
      
      #print(converged,eps)
   
   # In case we coverged check for unidentifiable parameters, as reccomended by Wood. et al (2016)
   keep = None
   drop = None
   if keep_drop is not None or (method == "QR/Chol" and eps > 0 and converged):
      
      if keep_drop is not None:
         keep = keep_drop[0]
         drop = keep_drop[1]
      else:
         # Check for drop
         keep,drop = identify_drop(H,S_norm)

      if len(drop) == 0:
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
         mus = [family.links[i].fi(etas[i]) for i in range(family.n_par)]

         # Re-compute llk
         c_llk = family.llk(y,*mus)
         c_pen_llk = c_llk - coef.T@rS_emb@coef
         
         # and now repeat Newton iteration
         for inner in range(max_inner):
      
            # Get derivatives with respect to eta
            d1eta,d2eta,d2meta = deriv_transform_mu_eta(y,mus,family)

            # Get derivatives with respect to coef
            grad,H = deriv_transform_eta_beta(d1eta,d2eta,d2meta,rXs)

            # Update Coefficients
            next_coef,L,LV,eps = newton_coef_smooth(coef,grad,H,rS_emb,method,piv_tol)

            # Prepare to check convergence
            prev_llk_cur_pen = c_llk - coef.T@rS_emb@coef

            # Perform step length control
            coef,rsplit_coef,mus,etas,c_llk,c_pen_llk = correct_coef_step_gammlss(family,y,rXs,coef,next_coef,rcoef_split_idx,c_llk,rS_emb)

            if np.abs(c_pen_llk - prev_llk_cur_pen) < conv_tol*np.abs(c_pen_llk):
               break

         #print("eps",eps)
         # converged on smaller problem - now adjust return objects for dropped coef

         # start with coef
         full_coef[keep] = coef
         coef = full_coef
         #print(coef)
         split_coef = np.split(coef,coef_split_idx)

         # Now H, L, LV
         Hdat,Hrow,Hcol = translate_sparse(H)
         Ldat,Lrow,Lcol = translate_sparse(L.tocsc()) # L@L.T = H_pen
         LVdat,LVrow,LVcol = translate_sparse(LV) # LV.T@LV = V

         Hrow = keep[Hrow]
         Hcol = keep[Hcol]
         Lrow = keep[Lrow]
         Lcol = keep[Lcol]
         LVrow = keep[LVrow]
         LVcol = keep[LVcol]

         H = scp.sparse.csc_array((Hdat,(Hrow,Hcol)),shape=(len(full_coef),len(full_coef)))
         L = scp.sparse.csc_array((Ldat,(Lrow,Lcol)),shape=(len(full_coef),len(full_coef)))
         LV = scp.sparse.csc_array((LVdat,(LVrow,LVcol)),shape=(len(full_coef),len(full_coef)))
         #print((LV.T@LV - V).max(),(LV.T@LV - V).min())
         #print((L@L.T - nH).max(),(L@L.T - nH).min())

   return coef,split_coef,mus,etas,H,L,LV,c_llk,c_pen_llk,eps,keep,drop
    
def solve_gammlss_sparse(family,y,Xs,form_n_coef,coef,coef_split_idx,gamlss_pen,
                         max_outer=50,max_inner=30,min_inner=1,conv_tol=1e-7,
                         extend_lambda=True,extension_method_lam = "nesterov2",
                         control_lambda=True,method="Chol",check_cond=1,piv_tol=0.175,
                         should_keep_drop=True,progress_bar=True,n_c=10):
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

    # Build current penalties
    S_emb,S_pinv,_,FS_use_rank = compute_S_emb_pinv_det(n_coef,gamlss_pen,"svd")
    c_llk = family.llk(y,*mus)
    c_pen_llk = c_llk - coef.T@S_emb@coef

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
    for outer in iterator:

        # Update coefficients:
        if outer == 0 or extend_lambda == False or control_lambda==False or (control_lambda and refit):
            coef,split_coef,mus,etas,H,L,LV,c_llk,c_pen_llk,eps,keep,drop = update_coef_gammlss(family,mus,y,Xs,coef,
                                                                                 coef_split_idx,S_emb,S_norm,
                                                                                 c_llk,outer,max_inner,
                                                                                 min_inner,conv_tol,
                                                                                 method,piv_tol,keep_drop)
            
            if drop is not None:

               fit_info.dropped = drop
               if should_keep_drop:
                  keep_drop = [keep,drop]

               # Re-compute penalty matrices in smaller problem space.
               old_pen = copy.deepcopy(gamlss_pen)
               
               # Should we re-build penalties here to match reduced space? Not sure that's necessary, because
               # V and LV have zeroes in rows and columns for un-identifiable parameters.
               gamlss_pen = drop_terms_S(gamlss_pen,keep)

               S_emb,S_pinv,_,FS_use_rank = compute_S_emb_pinv_det(n_coef,gamlss_pen,"svd")

            # Given new coefficients compute lgdetDs, ldetHS, and bsbs - needed for efs step
            lgdetDs = []
            bsbs = []
            for lti,lTerm in enumerate(gamlss_pen):

               lt_rank = None
               if FS_use_rank[lti]:
                  lt_rank = lTerm.rank

               lgdetD,bsb = compute_lgdetD_bsb(lt_rank,lTerm.lam,S_pinv,lTerm.S_J_emb,coef)
               lgdetDs.append(lgdetD)
               bsbs.append(bsb)

            total_edf,term_edfs, ldetHSs = calculate_edf(None,None,LV,gamlss_pen,lgdetDs,n_coef,n_c)
            #print([l1-l2 for l1,l2 in zip(lgdetDs,ldetHSs)])
            fit_info.lambda_updates += 1

            if drop is not None:
               gamlss_pen = old_pen

        # Check overall convergence
        if outer > 0:

            if progress_bar:
                iterator.set_description_str(desc="Fitting - Conv.: " + "{:.2e}".format((np.abs(prev_pen_llk - c_pen_llk) - conv_tol*np.abs(c_pen_llk))[0,0]), refresh=True)

            if np.abs(prev_pen_llk - c_pen_llk) < conv_tol*np.abs(c_pen_llk):
                if progress_bar:
                    iterator.set_description_str(desc="Converged!", refresh=True)
                    iterator.close()
                fit_info.code = 0
                break
            
        # We need the penalized likelihood of the model at this point for convergence control (step 2 in Wood, 2017 based on step 4 in Wood, Goude, & Shaw, 2016)
        prev_pen_llk = c_pen_llk
        
        # Now compute EFS step
        lam_delta = []
        for lti,lTerm in enumerate(gamlss_pen):

            lgdetD = lgdetDs[lti]
            ldetHS = ldetHSs[lti]
            bsb = bsbs[lti]
            
            #print(lgdetD-ldetHS)
            dLam = step_fellner_schall_sparse(lgdetD,ldetHS,bsb[0,0],lTerm.lam,1)

            # For poorly scaled/ill-identifiable problems we cannot rely on the theorems by Wood
            # & Fasiolo (2017) - so the condition below will be met, in which case we just want to
            # take very small steps until it hopefully gets more stable (due to term dropping or better lambda value).
            if lgdetD - ldetHS < 0:
               dLam = np.sign(dLam) * min(abs(lTerm.lam)*0.001,abs(dLam))

            if extend_lambda:
                dLam,extend_by,was_extended = extend_lambda_step(lti,lTerm.lam,dLam,extend_by,was_extended,extension_method_lam)
            lTerm.lam += dLam

            lam_delta.append(dLam)

        # Build new penalties
        S_emb,S_pinv,_,FS_use_rank = compute_S_emb_pinv_det(n_coef,gamlss_pen,"svd")

        if extend_lambda and control_lambda:
            # Step size control for smoothing parameters. Not obvious - because we have only approximate REMl
            # and approximate derivative, because we drop the last term involving the derivative of the negative penalized
            # Hessian with respect to the smoothing parameters (see section 4 in Wood & Fasiolo, 2017). However, what we
            # can do is at least undo the acceleration if we over-shoot the approximate derivative...

            # First re-compute coef
            next_coef,split_coef,next_mus,next_etas,H,L,LV,next_llk,next_pen_llk,eps,keep,drop  = update_coef_gammlss(family,mus,y,Xs,coef,
                                                                                                      coef_split_idx,S_emb,S_norm,
                                                                                                      c_llk,outer,max_inner,
                                                                                                      min_inner,conv_tol,
                                                                                                      method,piv_tol,keep_drop)
            
            if drop is not None:

               fit_info.dropped = drop
               if should_keep_drop:
                  keep_drop = [keep,drop]

               # Re-compute penalty matrices in smaller problem space.
               old_pen = copy.deepcopy(gamlss_pen)
               gamlss_pen = drop_terms_S(gamlss_pen,keep)

               S_emb,S_pinv,_,FS_use_rank = compute_S_emb_pinv_det(n_coef,gamlss_pen,"svd")

            # Now re-compute lgdetDs, ldetHS, and bsbs
            lgdetDs = []
            bsbs = []
            for lti,lTerm in enumerate(gamlss_pen):

                  lt_rank = None
                  if FS_use_rank[lti]:
                     lt_rank = lTerm.rank

                  lgdetD,bsb = compute_lgdetD_bsb(lt_rank,lTerm.lam,S_pinv,lTerm.S_J_emb,next_coef)
                  lgdetDs.append(lgdetD)
                  bsbs.append(bsb)

            total_edf,term_edfs, ldetHSs = calculate_edf(None,None,LV,gamlss_pen,lgdetDs,n_coef,n_c)
            #print([l1-l2 for l1,l2 in zip(lgdetDs,ldetHSs)])
            fit_info.lambda_updates += 1

            if drop is not None:
               gamlss_pen = old_pen
            
            # Compute approximate!!! gradient of REML with respect to lambda
            # to check if step size needs to be reduced (part of step 6 in Wood, 2017).
            lam_grad = [grad_lambda(lgdetDs[lti],ldetHSs[lti],bsbs[lti],1) for lti in range(len(gamlss_pen))]
            lam_grad = np.array(lam_grad).reshape(-1,1) 
            check = lam_grad.T @ lam_delta

            # Now undo the acceleration if overall direction is **very** off - don't just check against 0 because
            # our criterion is approximate, so we can be more lenient (see Wood et al., 2017). 
            
            refit = False
            if check[0] < 1e-7*-abs(next_pen_llk):
               refit = True
               for lti,lTerm in enumerate(gamlss_pen):
                  if was_extended[lti]:

                     lTerm.lam -= extend_by["acc"][lti]

               # Rebuild penalties
               S_emb,S_pinv,_,FS_use_rank = compute_S_emb_pinv_det(n_coef,gamlss_pen,"svd")
            else:
               # Can re-use estimate for next iteration
               coef=next_coef
               mus=next_mus
               etas=next_etas
               c_llk=next_llk
               c_pen_llk=next_pen_llk

        #print(outer,[lterm.lam for lterm in gamlss_pen])
        fit_info.iter += 1
    
    fit_info.eps = eps

    if check_cond == 1:

      if drop is not None:
         # Make sure cond. estimate happens in reduced space.
         L_drop_K2 = L[keep,:]
         L_drop_K2 = L_drop_K2[:,keep]

         LV_drop_K2 = LV[keep,:]
         LV_drop_K2 = LV_drop_K2[:,keep]
         K2,_,_,Kcode = est_condition(L_drop_K2,LV_drop_K2,verbose=False)
      else:
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
    
    return coef,etas,mus,wres,H,LV,total_edf,term_edfs,penalty[0,0],fit_info

################################################ General Smooth model code ################################################

def correct_coef_step_gen_smooth(family,y,Xs,coef,next_coef,coef_split_idx,c_llk,S_emb):
    """
    Apply step size correction to Newton update for general smooth models, as discussed by WPS (2016).

    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
    """
    
    # Step size control for newton step.
    next_llk = family.llk(next_coef,coef_split_idx,y,Xs)
    
    # Evaluate improvement of penalized llk under new and old coef - but in both
    # cases for current lambda (see Wood, Li, Shaddick, & Augustin; 2017)
    next_pen_llk = next_llk - next_coef.T@S_emb@next_coef
    prev_llk_cur_pen = c_llk - coef.T@S_emb@coef
    n_checks = 0
    while next_pen_llk < prev_llk_cur_pen:
        if n_checks > 30:
            next_coef = coef
            
        # Half it if we do not observe an increase in penalized likelihood (WPS, 2016)
        next_coef = (coef + next_coef)/2
        
        # Update pen_llk
        next_llk = family.llk(next_coef,coef_split_idx,y,Xs)
        next_pen_llk = next_llk - next_coef.T@S_emb@next_coef
        n_checks += 1
    
    return next_coef,next_llk,next_pen_llk
    
def update_coef_gen_smooth(family,y,Xs,coef,coef_split_idx,S_emb,S_norm,c_llk,outer,max_inner,min_inner,conv_tol,method,piv_tol,keep_drop):
   """
   Repeatedly perform Newton update with step length control to the coefficient vector - based on
   steps outlined by WPS (2016).

   References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
   """
   # Update coefficients:
   if keep_drop is None:
      converged = False
      for inner in range(max_inner):
         
         # Get llk derivatives with respect to coef
         grad = family.gradient(coef,coef_split_idx,y,Xs)
         H = family.hessian(coef,coef_split_idx,y,Xs)

         # Update coef and perform step size control
         if outer > 0 or inner > 0:
            # Update Coefficients
            next_coef,L,LV,eps = newton_coef_smooth(coef,grad,H,S_emb,method,piv_tol)

            # Prepare to check convergence
            prev_llk_cur_pen = c_llk - coef.T@S_emb@coef

            # Perform step length control
            coef,c_llk,c_pen_llk = correct_coef_step_gen_smooth(family,y,Xs,coef,next_coef,coef_split_idx,c_llk,S_emb)

            if np.abs(c_pen_llk - prev_llk_cur_pen) < conv_tol*np.abs(c_pen_llk):
               converged = True
               if eps <= 0 or method == 'Chol':
                  break

            if eps <= 0 and outer > 0 and inner >= (min_inner-1):
               break # end inner loop and immediately optimize lambda again.
         else:
            # Simply accept next coef step on first iteration
            coef,L,LV,eps = newton_coef_smooth(coef,grad,H,S_emb,method,piv_tol)
            c_llk = family.llk(coef,coef_split_idx,y,Xs)
            c_pen_llk = c_llk - coef.T@S_emb@coef
   
   # In case we coverged check for unidentifiable parameters, as reccomended by Wood. et al (2016)
   keep = None
   drop = None
   if keep_drop is not None or (method == "QR/Chol" and eps > 0 and converged):
      
      if keep_drop is not None:
         keep = keep_drop[0]
         drop = keep_drop[1]
      else:
         # Check for drop
         keep,drop = identify_drop(H,S_norm)

      if len(drop) == 0:
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
         c_pen_llk = c_llk - coef.T@rS_emb@coef
         
         # and now repeat Newton iteration
         for inner in range(max_inner):
      
            # Get llk derivatives with respect to coef
            grad = family.gradient(coef,rcoef_split_idx,y,rXs)
            H = family.hessian(coef,rcoef_split_idx,y,rXs)

            # Update Coefficients
            next_coef,L,LV,eps = newton_coef_smooth(coef,grad,H,rS_emb,method,piv_tol)

            # Prepare to check convergence
            prev_llk_cur_pen = c_llk - coef.T@rS_emb@coef

            # Perform step length control
            coef,c_llk,c_pen_llk = correct_coef_step_gen_smooth(family,y,rXs,coef,next_coef,rcoef_split_idx,c_llk,rS_emb)

            if np.abs(c_pen_llk - prev_llk_cur_pen) < conv_tol*np.abs(c_pen_llk):
                  break

         #print("eps",eps)
         # converged on smaller problem - now adjust return objects for dropped coef

         # start with coef
         full_coef[keep] = coef
         coef = full_coef

         # Now H, L, LV
         Hdat,Hrow,Hcol = translate_sparse(H)
         Ldat,Lrow,Lcol = translate_sparse(L.tocsc()) # L@L.T = H_pen
         LVdat,LVrow,LVcol = translate_sparse(LV) # LV.T@LV = V

         Hrow = keep[Hrow]
         Hcol = keep[Hcol]
         Lrow = keep[Lrow]
         Lcol = keep[Lcol]
         LVrow = keep[LVrow]
         LVcol = keep[LVcol]

         H = scp.sparse.csc_array((Hdat,(Hrow,Hcol)),shape=(len(full_coef),len(full_coef)))
         L = scp.sparse.csc_array((Ldat,(Lrow,Lcol)),shape=(len(full_coef),len(full_coef)))
         LV = scp.sparse.csc_array((LVdat,(LVrow,LVcol)),shape=(len(full_coef),len(full_coef)))
         #print((LV.T@LV - V).max(),(LV.T@LV - V).min())
         #print((L@L.T - nH).max(),(L@L.T - nH).min())
   
   return coef,H,L,LV,c_llk,c_pen_llk,eps,keep,drop


def solve_generalSmooth_sparse(family,y,Xs,form_n_coef,coef,coef_split_idx,smooth_pen,
                              max_outer=50,max_inner=50,min_inner=50,conv_tol=1e-7,
                              extend_lambda=True,extension_method_lam = "nesterov2",
                              control_lambda=True,optimizer="Newton",method="Chol",
                              check_cond=1,piv_tol=0.175,should_keep_drop=True,
                              form_VH=True,use_grad=False,progress_bar=True,
                              n_c=10,**bfgs_options):
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
    c_pen_llk = c_llk - coef.T@S_emb@coef
    
    iterator = range(max_outer)
    if progress_bar:
        iterator = tqdm(iterator,desc="Fitting",leave=True)

    if optimizer != "Newton":
        # Define negative penalized likelihood function to be minimized via BFGS
        # plus function to evaluate negative gradient of penalized likelihood - the
        # latter is only used if use_grad=True.
        def __neg_pen_llk(coef,coef_split_idx,y,Xs,family,S_emb):
            neg_llk = -1 * family.llk(coef,coef_split_idx,y,Xs)
            return neg_llk + coef.T@S_emb@coef
        
        def __neg_pen_grad(coef,coef_split_idx,y,Xs,family,S_emb):
           # see Wood, Pya & Saefken (2016)
           grad = family.gradient(coef,coef_split_idx,y,Xs)
           pgrad = np.array([grad[i] - (S_emb[[i],:]@coef)[0] for i in range(len(grad))])
           return -1*pgrad.flatten()
           
        
    fit_info = Fit_info()
    __old_opt = None
    for outer in iterator:

        # Update coefficients:
        if outer == 0 or extend_lambda == False or control_lambda==False or (control_lambda and refit):
            if optimizer == "Newton":
               coef,H,L,LV,c_llk,c_pen_llk,eps,keep,drop = update_coef_gen_smooth(family,y,Xs,coef,
                                                                  coef_split_idx,S_emb,S_norm,
                                                                  c_llk,outer,max_inner,
                                                                  min_inner,conv_tol,
                                                                  method,piv_tol,keep_drop)
               
               fit_info.eps = eps
               if drop is not None:

                  fit_info.dropped = drop
                  if should_keep_drop:
                     keep_drop = [keep,drop]

                  # Re-compute penalty matrices in smaller problem space.
                  old_pen = copy.deepcopy(smooth_pen)
                  smooth_pen = drop_terms_S(smooth_pen,keep)

                  S_emb,S_pinv,_,FS_use_rank = compute_S_emb_pinv_det(n_coef,smooth_pen,"svd")

            else:
               opt = scp.optimize.minimize(__neg_pen_llk,
                                          np.ndarray.flatten(coef),
                                          args=(coef_split_idx,y,Xs,family,S_emb),
                                          method=optimizer,
                                          jac = __neg_pen_grad if use_grad else None,
                                          options={"maxiter":max_inner,
                                                   **bfgs_options})
               
               drop = None
               eps = 0
               # Get coefficient estimate
               coef = opt["x"].reshape(-1,1)

               # Compute penalized likelihood for current estimate
               c_llk = family.llk(coef,coef_split_idx,y,Xs)
               c_pen_llk = c_llk - coef.T@S_emb@coef

               # Get inverse of Hessian of penalized likelihood
               if optimizer == "BFGS":
                  V = scp.sparse.csc_array(opt["hess_inv"])
                  V.eliminate_zeros()

                  # Get Cholesky factor needed for (accelerated) EFS
                  LVPT, P, code = cpp_cholP(V)
                  LVT = apply_eigen_perm(P,LVPT)
                  LV = LVT.T
               elif optimizer == "L-BFGS-B":
                  # Get linear operator. No need for Cholesky
                  V = opt.hess_inv

                  if __old_opt is not None and V.n_corrs < __old_opt.n_corrs:
                     # L-BFGS converged quickly, so (inverse) of Hessian might in worst case simply be set to identity
                     # but we can re-use last approximation of inverse to fill up
                     for cori in range(__old_opt.n_corrs-1,V.n_corrs-1,-1):
                        V.sk = np.insert(V.sk,0,__old_opt.sk[cori],axis=0)
                        V.yk = np.insert(V.yk,0,__old_opt.yk[cori],axis=0)
                        V.rho = np.insert(V.rho,0,__old_opt.rho[cori],axis=0)
                     
                     V.n_corrs = __old_opt.n_corrs
                  
                  __old_opt = copy.deepcopy(V)

            # Given new coefficients compute lgdetDs, ldetHS, and bsbs - needed for efs step
            lgdetDs = []
            bsbs = []
            for lti,lTerm in enumerate(smooth_pen):

               lt_rank = None
               if FS_use_rank[lti]:
                  lt_rank = lTerm.rank

               lgdetD,bsb = compute_lgdetD_bsb(lt_rank,lTerm.lam,S_pinv,lTerm.S_J_emb,coef)
               lgdetDs.append(lgdetD)
               bsbs.append(bsb)

            total_edf,term_edfs, ldetHSs = calculate_edf(None,None,V if optimizer=="L-BFGS-B" else LV,smooth_pen,lgdetDs,n_coef,n_c)
            fit_info.lambda_updates += 1

            if drop is not None:
               smooth_pen = old_pen

        # Check overall convergence
        if outer > 0:

            if progress_bar:
                iterator.set_description_str(desc="Fitting - Conv.: " + "{:.2e}".format((np.abs(prev_pen_llk - c_pen_llk) - conv_tol*np.abs(c_pen_llk))[0,0]), refresh=True)

            if np.abs(prev_pen_llk - c_pen_llk) < conv_tol*np.abs(c_pen_llk):
                if progress_bar:
                    iterator.set_description_str(desc="Converged!", refresh=True)
                    iterator.close()
                fit_info.code = 0
                break
        
        # We need the penalized likelihood of the model at this point for convergence control (step 2 in Wood, 2017 based on step 4 in Wood, Goude, & Shaw, 2016)
        prev_pen_llk = c_pen_llk

        # Now compute EFS step
        lam_delta = []
        for lti,lTerm in enumerate(smooth_pen):

            lgdetD = lgdetDs[lti]
            ldetHS = ldetHSs[lti]
            bsb = bsbs[lti]
            
            dLam = step_fellner_schall_sparse(lgdetD,ldetHS,bsb[0,0],lTerm.lam,1)
            #print(lgdetD-ldetHS,dLam)

            # For poorly scaled/ill-identifiable problems we cannot rely on the theorems by Wood
            # & Fasiolo (2017) - so the condition below will ocasionally be met, in which case we just want to
            # take very small steps until it hopefully gets more stable (due to term dropping or better lambda value).
            if lgdetD - ldetHS < 0:
               dLam = np.sign(dLam) * min(abs(lTerm.lam)*0.001,abs(dLam))

            if extend_lambda:
                dLam,extend_by,was_extended = extend_lambda_step(lti,lTerm.lam,dLam,extend_by,was_extended,extension_method_lam)
            lTerm.lam += dLam

            lam_delta.append(dLam)

        # Build new penalties
        S_emb,S_pinv,_,FS_use_rank = compute_S_emb_pinv_det(n_coef,smooth_pen,"svd")

        if extend_lambda and control_lambda:
            # Step size control for smoothing parameters. Not obvious - because we have only approximate REMl
            # and approximate derivative, because we drop the last term involving the derivative of the negative penalized
            # Hessian with respect to the smoothing parameters (see section 4 in Wood & Fasiolo, 2017). However, what we
            # can do is at least undo the acceleration if we over-shoot the approximate derivative...

            # First re-compute coef
            if optimizer == "Newton":
                next_coef,H,L,LV,next_llk,next_pen_llk,eps,keep,drop = update_coef_gen_smooth(family,y,Xs,coef,
                                                                coef_split_idx,S_emb,S_norm,
                                                                c_llk,outer,max_inner,
                                                                min_inner,conv_tol,
                                                                method,piv_tol,keep_drop)
                
                fit_info.eps = eps
                if drop is not None:

                  fit_info.dropped = drop
                  if should_keep_drop:
                     keep_drop = [keep,drop]

                  # Re-compute penalty matrices in smaller problem space.
                  old_pen = copy.deepcopy(smooth_pen)
                  smooth_pen = drop_terms_S(smooth_pen,keep)

                  S_emb,S_pinv,_,FS_use_rank = compute_S_emb_pinv_det(n_coef,smooth_pen,"svd")

            else:
                opt = scp.optimize.minimize(__neg_pen_llk,
                                            np.ndarray.flatten(coef),
                                            args=(coef_split_idx,y,Xs,family,S_emb),
                                            method=optimizer,
                                            jac = __neg_pen_grad if use_grad else None,
                                            options={"maxiter":max_inner,
                                                     **bfgs_options})
                
                drop = None
                eps = 0
                # Get next coefficient estimate
                next_coef = opt["x"].reshape(-1,1)

                # Compute penalized likelihood for next estimate
                next_llk = family.llk(next_coef,coef_split_idx,y,Xs)
                next_pen_llk = next_llk - next_coef.T@S_emb@next_coef

                # Get inverse of Hessian of penalized likelihood
                if optimizer == "BFGS":
                  V = scp.sparse.csc_array(opt["hess_inv"])
                  V.eliminate_zeros()

                  # Get Cholesky factor needed for (accelerated) EFS
                  LVPT, P, code = cpp_cholP(V)
                  LVT = apply_eigen_perm(P,LVPT)
                  LV = LVT.T
                elif optimizer == "L-BFGS-B":
                  # Get linear operator. No need for Cholesky
                  V = opt.hess_inv

                  if __old_opt is not None and V.n_corrs < __old_opt.n_corrs:
                     # L-BFGS converged quickly, so (inverse) of Hessian might in worst case simply be set to identity
                     # but we can re-use last approximation of inverse to fill up
                     for cori in range(__old_opt.n_corrs-1,V.n_corrs-1,-1):
                        V.sk = np.insert(V.sk,0,__old_opt.sk[cori],axis=0)
                        V.yk = np.insert(V.yk,0,__old_opt.yk[cori],axis=0)
                        V.rho = np.insert(V.rho,0,__old_opt.rho[cori],axis=0)

                     V.n_corrs = __old_opt.n_corrs
                  
                  __old_opt = copy.deepcopy(V)
            
            # Now re-compute lgdetDs, ldetHS, and bsbs
            lgdetDs = []
            bsbs = []
            for lti,lTerm in enumerate(smooth_pen):

                lt_rank = None
                if FS_use_rank[lti]:
                    lt_rank = lTerm.rank

                lgdetD,bsb = compute_lgdetD_bsb(lt_rank,lTerm.lam,S_pinv,lTerm.S_J_emb,next_coef)
                lgdetDs.append(lgdetD)
                bsbs.append(bsb)

            total_edf,term_edfs, ldetHSs = calculate_edf(None,None,V if optimizer=="L-BFGS-B" else LV,smooth_pen,lgdetDs,n_coef,n_c)
            fit_info.lambda_updates += 1

            if drop is not None:
               smooth_pen = old_pen
            
            # Compute approximate!!! gradient of REML with respect to lambda
            # to check if step size needs to be reduced (part of step 6 in Wood, 2017).
            lam_grad = [grad_lambda(lgdetDs[lti],ldetHSs[lti],bsbs[lti],1) for lti in range(len(smooth_pen))]
            lam_grad = np.array(lam_grad).reshape(-1,1) 
            check = lam_grad.T @ lam_delta

            # Now undo the acceleration if overall direction is **very** off - don't just check against 0 because
            # our criterion is approximate, so we can be more lenient (see Wood et al., 2017).
            refit = False 
            if check[0] < 1e-7*-abs(next_pen_llk):
                refit = True
                for lti,lTerm in enumerate(smooth_pen):
                    if was_extended[lti]:

                        lTerm.lam -= extend_by["acc"][lti]

                # Rebuild penalties
                S_emb,S_pinv,_,FS_use_rank = compute_S_emb_pinv_det(n_coef,smooth_pen,"svd")
            else: # Can re-use estimate for next iteration
               coef = next_coef
               c_llk = next_llk
               c_pen_llk = next_pen_llk

        #print([lterm.lam for lterm in smooth_pen])
        fit_info.iter += 1
    
    # Total penalty
    penalty = coef.T@S_emb@coef

    # Calculate actual term-specific edf
    term_edfs = calculate_term_edf(smooth_pen,term_edfs)

    if optimizer != "Newton":
        
        if optimizer == "L-BFGS-B":
           
           if form_VH:
               # Optionally form last V + Chol explicitly during last iteration
               V = scp.sparse.csc_array(V.todense())
               V.eliminate_zeros()

               # Get Cholesky factor needed for (accelerated) EFS
               LVPT, P, code = cpp_cholP(V)
               LVT = apply_eigen_perm(P,LVPT)
               LV = LVT.T
           else:
               LV = V # Return operator directly.

        if (optimizer != "L-BFGS-B") or form_VH:
            # Get an approximation of the Hessian of the likelihood
            LHPT = compute_Linv(LVPT)
            LHT = apply_eigen_perm(P,LHPT)
            L = LHT.T
            H = L@LHT # approximately: negative Hessian of llk + S_emb
            H -= S_emb # approximately: negative Hessian of llk 
            H *= -1 # approximately: Hessian of llk
        else:
           H = None # Do not approximate H
       
    if check_cond == 1 and ((optimizer != "L-BFGS-B") or form_VH):

      if drop is not None:
         # Make sure cond. estimate happens in reduced space.
         L_drop_K2 = L[keep,:]
         L_drop_K2 = L_drop_K2[:,keep]

         LV_drop_K2 = LV[keep,:]
         LV_drop_K2 = LV_drop_K2[:,keep]
         K2,_,_,Kcode = est_condition(L_drop_K2,LV_drop_K2,verbose=False)
      else:
         K2,_,_,Kcode = est_condition(L,LV,verbose=False)

      fit_info.K2 = K2

      if fit_info.code == 0: # Convergence was reached but Knumber might suggest instable system.
         fit_info.code = Kcode

      if Kcode > 0:
         warnings.warn(f"Condition number ({K2}) of matrix A, where A.T@A=H and H is the Hessian of the negative penalized likelihood, is larger than 1/sqrt(u), where u is half the machine precision. Call ``model.fit()`` with ``method='QR/Chol'``, but note that even then estimates are likely to be inaccurate.")

    return coef,H,LV,total_edf,term_edfs,penalty[0,0],fit_info