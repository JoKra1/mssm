import numpy as np
import scipy as scp
import warnings
from .exp_fam import Family,Gaussian,est_scale,GAMLSSFamily,Identity
from .penalties import PenType,id_dist_pen,translate_sparse,dataclass
from .formula import build_sparse_matrix_from_formula,setup_cache,clear_cache,cpp_solvers,pd,Formula,mp,repeat,os,map_csc_to_eigen,math,tqdm,sys
from functools import reduce
from multiprocessing import managers,shared_memory

CACHE_DIR = './.db'
SHOULD_CACHE = False
MP_SPLIT_SIZE = 2000

@dataclass
class Fit_info:
   lambda_updates:int=0
   iter:int=0
   code:int=1

def cpp_chol(A):
   return cpp_solvers.chol(*map_csc_to_eigen(A))

def cpp_cholP(A):
   return cpp_solvers.cholP(*map_csc_to_eigen(A))

def cpp_qr(A):
   return cpp_solvers.pqr(*map_csc_to_eigen(A))

def cpp_solve_qr(A):
   return cpp_solvers.solve_pqr(*map_csc_to_eigen(A))

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


def calculate_edf(LP,Pr,InvCholXXS,penalties,lgdetDs,colsX,n_c):
   # Follows steps outlined by Wood & Fasiolo (2017) to compute total degrees of freedom by the model.
   # Generates the B matrix also required for the derivative of the log-determinant of X.T@X+S_\lambda. This
   # is either done exactly - as described by Wood & Fasiolo (2017) - or approximately. The latter is much faster
   total_edf = colsX
   Bs = []
   term_edfs = []

   for lti,lTerm in enumerate(penalties):
      if not InvCholXXS is None:
         B = InvCholXXS @ lTerm.D_J_emb # Needed for Fellner Schall update (Wood & Fasiolo, 2017)
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

def update_coef_and_scale(y,yb,z,Wr,rowsX,colsX,X,Xb,family,S_emb,S_pinv,FS_use_rank,penalties,n_c,formula,form_Linv):
   # Solves the additive model for a given set of weights and penalty
   if formula is None:
      LP, Pr, coef, code = cpp_solve_coef(yb,Xb,S_emb)
   else:
      #yb is X.T@y and Xb is X.T@X
      LP, Pr, coef, code = cpp_solve_coefXX(yb,Xb + S_emb)

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
   return eta,mu,coef,InvCholXXS,lgdetDs,bsbs,total_edf,term_edfs,Bs,scale,wres

def init_step_gam(y,yb,mu,eta,rowsX,colsX,X,Xb,
                  family,col_S,penalties,
                  pinv,n_c,formula,form_Linv):
   # Initial fitting iteration without step-length control for gam.

   # Compute starting estimate S_emb and S_pinv
   if len(penalties) > 0:
      S_emb,S_pinv,FS_use_rank = compute_S_emb_pinv_det(col_S,penalties,pinv)
   else:
      S_emb = scp.sparse.csc_array((colsX, colsX), dtype=np.float64)
      S_pinv = None
      FS_use_rank = None

   # Estimate coefficients for starting lambda
   # We just accept those here - no step control, since
   # there are no previous coefficients/deviance that we can
   # compare the result to.

   # First (optionally, only in the non Gaussian case) compute pseudo-dat and weights:
   yb,Xb,z,Wr = update_PIRLS(y,yb,mu,eta,X,Xb,family)
   
   # Solve additive model
   eta,mu,coef,\
   InvCholXXS,\
   lgdetDs,\
   bsbs,\
   total_edf,\
   term_edfs,\
   Bs,scale,wres = update_coef_and_scale(y,yb,z,Wr,rowsX,colsX,
                                         X,Xb,family,S_emb,S_pinv,
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

   extend_by = 1 # Extension factor for lambda update for the Fellner Schall method by Wood & Fasiolo (2016)

   if method == "nesterov":
      extend_by = {"prev_lam":[lterm.lam for lterm in penalties],
                   "acc":[0 for _ in penalties]}

   return extend_by

def extend_lambda_step(lti,lam,dLam,extend_by,was_extended, method):
   """
   Performs an update to the lambda parameter, ideally extending the step
   taken without overshooting the objective. Two options are supported.
   Setting method to "mult" will simply multiply lambda by an extension factor.
   This one will be increased whenever an extension was successful (extended step
   did not overshoot) and reset to 1 (no extension), whenever an extension was not
   succesful. Wood (2017) and Wood & Fasiolo (2016) suggest that such a simple
   strategy can lead to improved convergence speed.

   If method is set to "nesterov", a nesterov-like acceleration update is applied to the
   lambda parameter. We fall back to the default startegy by Wood & Fasiolo (2016) to just
   half the step taken in case the extension was not succesful.
   """

   if method == "mult":
      extension = lam  + dLam*extend_by
      if extension < 1e7 and extension > 1e-7:
         dLam *= extend_by
         was_extended[lti] = True
      else:
         was_extended[lti] = False

   elif method == "nesterov":
      # The idea for the correction is based on the derivations in the supplementary materials
      # of Sutskever et al. (2013) - but adapted to use efs_step**2 / |lam_t - lam_{t-1}| for
      # the correction to lambda. So that the next efs update will be calculated from
      # lam_t + efs_step + (efs_step**2 / |lam_t - lam_{t-1}|) instead of just lam_t + efs_step.

      # Essentially, until corrected increase in lambda reaches unit size a fraction of the quadratic
      # efs_step is added. At unit size the quadratic efs_step is added in its entirety. As corrected update
      # step-length decreases further the extension will become increasingly larger than just the quadratic
      # of the efs_step.
   
      diff_lam = lam - extend_by["prev_lam"][lti]
      extend_by["prev_lam"][lti] = lam
      acc = np.sign(dLam)*(dLam**2/max(sys.float_info.epsilon,abs(diff_lam)))
      extend_by["acc"][lti] = acc

      extension = lam + dLam + acc

      if extension < 1e7 and extension > 1e-7 and np.sign(diff_lam) == np.sign(dLam):
         dLam += acc
         was_extended[lti] = True
      else:
         was_extended[lti] = False
   else:
      raise ValueError(f"Lambda extension method '{method}' is not implemented.")
   
   return dLam,extend_by,was_extended

def reduce_lambda_step(lti,lam,dLam,extend_by,was_extended, method):
   """
   Corrects the lambda step taken if it overshot the objective. Deals with resetting
   any extension terms.
   """
   if method == "mult":
      if extend_by > 1 and was_extended[lti]:
         # I experimented with just iteratively reducing the step-size but it just takes too many
         # wasted iterations then. Thus, I now just reset the extension factor below. It can then build up again
         # if needed.
         
         # Make sure correction by extend_by is only applied if an extension was actually used.
         lam -= dLam
         dLam /= extend_by
         lam += dLam

      else: # fall back to the strategy by Wood & Fasiolo (2016) to just half the step.
         dLam/=2
         lam -= dLam
   
   elif method == "nesterov":
      # We can simply reset lam by the extension factor computed earlier. If we repeatedly have to half we
      # can fall back to the strategy by Wood & Fasiolo (2016) to just half the step.
      if was_extended[lti] and extend_by["acc"][lti] != 0:
         dLam-= extend_by["acc"][lti]
         lam -= extend_by["acc"][lti]
         extend_by["acc"][lti] = 0
      else:
         dLam/=2
         lam -= dLam

   else:
      raise ValueError(f"Lambda extension method '{method}' is not implemented.")

   return lam, dLam

def adapt_extension_strategy(extend_by,reduced_step,dev_check,method):
   """
   Adjusts the step length (currently only for multiplication strategy) depending on whether the previous extension
   was succesful or not.
   """
   if method == "mult":
      if reduced_step and extend_by > 1:
         extend_by = 1
   
      elif not reduced_step and extend_by < 2 and not dev_check is None and dev_check:
         # Try longer step next time.
         extend_by += 0.5

   return extend_by

def correct_lambda_step(y,yb,z,Wr,rowsX,colsX,X,Xb,
                        family,col_S,S_emb,penalties,
                        was_extended,pinv,lam_delta,
                        extend_by,o_iter,dev_check,n_c,
                        control_lambda,extend_lambda,
                        exclude_lambda,extension_method_lam,
                        formula,form_Linv):
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
      lgdetDs,\
      bsbs,\
      total_edf,\
      term_edfs,\
      Bs,scale,wres = update_coef_and_scale(y,yb,z,Wr,rowsX,colsX,
                                             X,Xb,family,S_emb,S_pinv,
                                             FS_use_rank,penalties,n_c,
                                             formula,form_Linv)
      
      # Compute gradient of REML with respect to lambda
      # to check if step size needs to be reduced (part of step 6 in Wood, 2017).
      lam_grad = [grad_lambda(lgdetDs[lti],Bs[lti],bsbs[lti],scale) for lti in range(len(penalties))]
      lam_grad = np.array(lam_grad).reshape(-1,1) 
      check = lam_grad.T @ lam_delta

      if check[0,0] < 0 and control_lambda: # because of minimization in Wood (2017) they use a different check (step 7) but idea is the same.
         # Reset extension or cut the step taken in half
         for lti,lTerm in enumerate(penalties):
            if extend_lambda:
               lam, dLam = reduce_lambda_step(lti,lTerm.lam,lam_delta[lti][0],extend_by,was_extended, extension_method_lam)
               lTerm.lam = lam
               lam_delta[lti][0] = dLam

            else: # If no extension is to be used rely on the strategy by Wood & Fasiolo (2016) to just half the step
               lam_delta[lti] = lam_delta[lti]/2
               lTerm.lam -= lam_delta[lti][0]

         if extend_lambda:
            extend_by = adapt_extension_strategy(extend_by,True,dev_check,extension_method_lam)
         
      else:
         if extend_lambda and lam_checks == 0: 
            extend_by = adapt_extension_strategy(extend_by,False,dev_check,extension_method_lam)

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

   return eta,mu,n_coef,InvCholXXS,total_edf,term_edfs,scale,wres,lam_delta,extend_by,penalties,was_extended,S_emb,lam_checks

################################################ Main solver ################################################

def solve_gamm_sparse(mu_init,y,X,penalties,col_S,family:Family,
                      maxiter=10,pinv="svd",conv_tol=1e-7,
                      extend_lambda=True,control_lambda=True,
                      exclude_lambda=False,extension_method_lam = "nesterov",
                      form_Linv=True,progress_bar=False,n_c=10):
   # Estimates a penalized Generalized additive mixed model, following the steps outlined in Wood, Li, Shaddick, & Augustin (2017)
   # "Generalized Additive Models for Gigadata" referred to as Wood (2017) below.

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

   if isinstance(family,Gaussian) == False or isinstance(family.link,Identity) == False:
      eta = family.link.f(mu)

   # Compute starting estimates
   dev,pen_dev,eta,mu,coef,InvCholXXS,total_edf,term_edfs,scale,wres,lam_delta,S_emb = init_step_gam(y,yb,mu,eta,rowsX,colsX,X,Xb,
                                                                                                     family,col_S,penalties,
                                                                                                     pinv,n_c,None,form_Linv)
      
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
            dev_check = dev_diff < 1e-3*pen_dev

         eta,mu,n_coef,InvCholXXS,total_edf,term_edfs,scale,wres,lam_delta,extend_by,penalties,was_extended,S_emb,lam_checks = correct_lambda_step(y,yb,z,Wr,rowsX,colsX,X,Xb,
                                                                                                                                                   family,col_S,S_emb,penalties,
                                                                                                                                                   was_extended,pinv,lam_delta,
                                                                                                                                                   extend_by,o_iter,dev_check,n_c,
                                                                                                                                                   control_lambda,extend_lambda,
                                                                                                                                                   exclude_lambda,extension_method_lam,
                                                                                                                                                   None,form_Linv)
         fit_info.lambda_updates += lam_checks
         
      else:
         # If there are no penalties simply perform a newton step
         # for the coefficients only
         eta,mu,n_coef,\
         InvCholXXS,\
         _,\
         _,\
         total_edf,\
         term_edfs,\
         _,scale,wres = update_coef_and_scale(y,yb,z,Wr,rowsX,colsX,
                                              X,Xb,family,S_emb,None,None,
                                              penalties,n_c,None,form_Linv)
      
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

   return coef,eta,wres,Wr,scale,InvCholXXS,total_edf,term_edfs,penalty,fit_info

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

def read_XTX(file,formula,nc):
   """
   Reads subset of data and creates X.T@X with X = model matrix for that dataset.
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
   cov_flat_files = np.array_split(cov_flat_file,min(nc,rows),axis=0)
   y_flat_files = np.array_split(y_flat_file,min(nc,rows))
   subsets = [i for i in range(len(cov_flat_files))]

   with mp.Pool(processes=nc) as pool:
      # Build the model matrix with all information from the formula - but only for sub-set of rows in this file
      XX,Xy = zip(*pool.starmap(form_cross_prod_mp,zip(repeat(SHOULD_CACHE),repeat(CACHE_DIR),repeat(file),
                                                       subsets,y_flat_files,repeat(terms),repeat(has_intercept),repeat(has_scale_split),
                                                       repeat(ltx),repeat(irstx),repeat(stx),repeat(rtx),
                                                       repeat(var_types),repeat(var_map),
                                                       repeat(var_mins),repeat(var_maxs),repeat(factor_levels),
                                                       cov_flat_files,repeat(cov),repeat(n_j),
                                                       repeat(state_est_flat),repeat(state_est))))
   
   XX = reduce(lambda xx1,xx2: xx1+xx2,XX)
   Xy = reduce(lambda xy1,xy2: xy1+xy2,Xy)
   return XX,Xy,len(y_flat_file)

def keep_XTX(cov_flat,y_flat,formula,nc,progress_bar):
   """
   Takes subsets of data and creates X.T@X with X = model matrix iteratively over these subsets.
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
                                                         subsets,y_flat_files,repeat(terms),repeat(has_intercept),repeat(has_scale_split),
                                                         repeat(ltx),repeat(irstx),repeat(stx),repeat(rtx),
                                                         repeat(var_types),repeat(var_map),
                                                         repeat(var_mins),repeat(var_maxs),repeat(factor_levels),
                                                         cov_flat_files,repeat(cov),repeat(n_j),
                                                         repeat(state_est_flat),repeat(state_est))))
      
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
   Reads subset of data and creates model prediction for that dataset.
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
   cov_flat_files = np.array_split(cov_flat_file,min(nc,rows),axis=0)
   subsets = [i for i in range(len(cov_flat_files))]

   with mp.Pool(processes=nc) as pool:
      # Build eta with all information from the formula - but only for sub-set of rows in this file
      etas = pool.starmap(form_eta_mp,zip(repeat(SHOULD_CACHE),repeat(CACHE_DIR),repeat(file),subsets,
                                          repeat(coef),repeat(terms),repeat(has_intercept),
                                          repeat(has_scale_split),repeat(ltx),repeat(irstx),repeat(stx),
                                          repeat(rtx),repeat(var_types),repeat(var_map),
                                          repeat(var_mins),repeat(var_maxs),repeat(factor_levels),
                                          cov_flat_files,repeat(cov),repeat(n_j),
                                          repeat(state_est_flat),repeat(state_est)))

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
                                             repeat(has_scale_split),repeat(ltx),repeat(irstx),repeat(stx),
                                             repeat(rtx),repeat(var_types),repeat(var_map),
                                             repeat(var_mins),repeat(var_maxs),repeat(factor_levels),
                                             cov_flat_files,repeat(cov),repeat(n_j),
                                             repeat(state_est_flat),repeat(state_est)))


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
   dev,pen_dev,eta,mu,coef,InvCholXXS,total_edf,term_edfs,scale,wres,lam_delta,S_emb = init_step_gam(y,Xy,mu,eta,rowsX,colsX,None,XX,
                                                                                                     family,col_S,penalties,
                                                                                                     pinv,n_c,formula,form_Linv)
   
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
            dev_check = dev_diff < 1e-3*pen_dev

         eta,mu,n_coef,InvCholXXS,total_edf,term_edfs,scale,wres,lam_delta,extend_by,penalties,was_extended,S_emb,lam_checks = correct_lambda_step(y,Xy,None,None,rowsX,colsX,None,XX,
                                                                                                                                                   family,col_S,S_emb,penalties,
                                                                                                                                                   was_extended,pinv,lam_delta,
                                                                                                                                                   extend_by,o_iter,dev_check,
                                                                                                                                                   n_c,control_lambda,extend_lambda,
                                                                                                                                                   exclude_lambda,extension_method_lam,
                                                                                                                                                   formula,form_Linv)
         fit_info.lambda_updates += lam_checks
      else:
         # If there are no penalties simply perform a newton step
         # for the coefficients only
         eta,mu,n_coef,\
         InvCholXXS,\
         _,\
         _,\
         total_edf,\
         term_edfs,\
         _,scale,wres = update_coef_and_scale(y,Xy,None,None,rowsX,colsX,
                                              None,XX,family,S_emb,None,None,
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

   return coef,eta,wres,scale,InvCholXXS,total_edf,term_edfs,penalty,fit_info


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

            #print(etai,etaj,mixed_idx)
                
            for coefi in range(Xs[etai].shape[1]):
                for coefj in range(Xs[etaj].shape[1]):

                    if hc_idx+coefj < hr_idx+coefi:
                        continue

                    # Naive:
                    # d2beta = np.sum([d2[i]*Xs[etai][i,coefi]*Xs[etaj][i,coefj] for i in range(Xs[etai].shape[0])])
                    # But this is again just a dot product, preceded by element wise multiplication. In principle we
                    # could even skip these loops but that might get a bit tricky with sparse matrix set up- for now
                    # I just leave it like this...
                    d2beta = ((d2*Xs[etai][:,[coefi]]).T @ Xs[etaj][:,[coefj]])[0,0]
                    #print(hr_idx+coefi,hc_idx+coefj)
                    h_rows.append(hr_idx+coefi)
                    h_cols.append(hc_idx+coefj)
                    h_vals.append(d2beta)
                    if hr_idx+coefi != hc_idx+coefj: # Symmetric 2nd deriv..
                        h_rows.append(hc_idx+coefj)
                        h_cols.append(hr_idx+coefi)
                        h_vals.append(d2beta)

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
    """
    
    pgrad = np.array([grad[i] - (S_emb[[i],:]@coef)[0] for i in range(len(grad))])
    nH = -1*H + S_emb

    # Diagonal pre-conditioning as suggested by WPS (2016)
    D = scp.sparse.diags(np.abs(nH.diagonal())**0.5)
    nH2 = (D@nH@D).tocsc()

    # Compute V, inverse of nH
    eps = 0
    code = 1
    while code != 0:

        Lp, Pr, code = cpp_cholP(nH2+eps*scp.sparse.identity(nH2.shape[1],format='csc'))

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

    return n_coef,LV,eps

def correct_coef_step_gen_smooth(family,y,Xs,coef,next_coef,coef_split_idx,c_llk,S_emb):
    """
    Apply step size correction to Newton update for general smooth
    models and GAMLSS models, as discussed by WPS (2016).

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
        
        # Step size control for newton step. Half it if we do not
        # observe an increase in penalized likelihood (WPS, 2016)
        next_llk = family.llk(y,*next_mus)
        next_pen_llk = next_llk - next_coef.T@S_emb@next_coef
        n_checks += 1
    
    return next_coef,next_split_coef,next_mus,next_etas,next_llk,next_pen_llk
    
def update_coef_gen_smooth(family,mus,y,Xs,coef,coef_split_idx,S_emb,c_llk,outer,max_inner,min_inner,conv_tol):
   """
   Repeatedly perform Newton update with step length control to the coefficient vector - based on
   steps outlined by WPS (2016).

   References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
   """
   # Update coefficients:
   for inner in range(max_inner):
      
      # Get derivatives with respect to eta
      d1eta,d2eta,d2meta = deriv_transform_mu_eta(y,mus,family)

      # Get derivatives with respect to coef
      grad,H = deriv_transform_eta_beta(d1eta,d2eta,d2meta,Xs)

      # Update coef and perform step size control
      if outer > 0 or inner > 0:
         # Update Coefficients
         next_coef,LV,eps = newton_coef_smooth(coef,grad,H,S_emb)

         # Prepare to check convergence
         prev_llk_cur_pen = c_llk - coef.T@S_emb@coef

         # Perform step length control
         coef,split_coef,mus,etas,c_llk,c_pen_llk = correct_coef_step_gen_smooth(family,y,Xs,coef,next_coef,coef_split_idx,c_llk,S_emb)

         if np.abs(c_pen_llk - prev_llk_cur_pen) < conv_tol*np.abs(c_pen_llk):
            break

         if eps <= 0 and outer > 0 and inner >= (min_inner-1):
            break # end inner loop and immediately optimize lambda again.
      else:
         # Simply accept next coef step on first iteration
         coef,LV,_ = newton_coef_smooth(coef,grad,H,S_emb)
         split_coef = np.split(coef,coef_split_idx)
         etas = [Xs[i]@split_coef[i] for i in range(family.n_par)]
         mus = [family.links[i].fi(etas[i]) for i in range(family.n_par)]
         c_llk = family.llk(y,*mus)
         c_pen_llk = c_llk - coef.T@S_emb@coef
   
   return coef,split_coef,mus,etas,H,LV,c_llk,c_pen_llk
    
def solve_gammlss_sparse(family,y,Xs,form_n_coef,coef,coef_split_idx,gamlss_pen,
                         max_outer=50,max_inner=30,min_inner=1,conv_tol=1e-7,
                         extend_lambda=True,control_lambda=True,progress_bar=True,n_c=10):
    """
    Fits a GAMLSS model, following steps outlined by Wood, Pya, & Säfken (2016).

    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
    """
    # total number of coefficients
    n_coef = np.sum(form_n_coef)

    extend_by = initialize_extension("nesterov",gamlss_pen)
    was_extended = [False for _ in enumerate(gamlss_pen)]

    split_coef = np.split(coef,coef_split_idx)

    # Initialize etas and mus
    etas = [Xs[i]@split_coef[i] for i in range(family.n_par)]
    mus = [family.links[i].fi(etas[i]) for i in range(family.n_par)]

    # Build current penalties
    S_emb,S_pinv,FS_use_rank = compute_S_emb_pinv_det(n_coef,gamlss_pen,"svd")
    c_llk = family.llk(y,*mus)
    c_pen_llk = c_llk - coef.T@S_emb@coef

    iterator = range(max_outer)
    if progress_bar:
        iterator = tqdm(iterator,desc="Fitting",leave=True)

    for outer in iterator:

        # Update coefficients:
        coef,split_coef,mus,etas,H,LV,c_llk,c_pen_llk = update_coef_gen_smooth(family,mus,y,Xs,coef,
                                                                             coef_split_idx,S_emb,
                                                                             c_llk,outer,max_inner,
                                                                             min_inner,conv_tol)
        
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

        # Check overall convergence
        if outer > 0:

            if progress_bar:
                iterator.set_description_str(desc="Fitting - Conv.: " + "{:.2e}".format((np.abs(prev_pen_llk - c_pen_llk) - conv_tol*np.abs(c_pen_llk))[0,0]), refresh=True)

            if np.abs(prev_pen_llk - c_pen_llk) < conv_tol*np.abs(c_pen_llk):
                if progress_bar:
                    iterator.set_description_str(desc="Converged!", refresh=True)
                    iterator.close()
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
            if extend_lambda:
                dLam,extend_by,was_extended = extend_lambda_step(lti,lTerm.lam,dLam,extend_by,was_extended,"nesterov")
            lTerm.lam += dLam

            lam_delta.append(dLam)

        # Build new penalties
        S_emb,S_pinv,FS_use_rank = compute_S_emb_pinv_det(n_coef,gamlss_pen,"svd")

        if extend_lambda and control_lambda:
            # Step size control for smoothing parameters. Not obvious - because we have only approximate REMl
            # and approximate derivative, because we drop the last term involving the derivative of the negative penalized
            # Hessian with respect to the smoothing parameters (see section 4 in Wood & Fasiolo, 2017). However, what we
            # can do is at least undo the acceleration if we over-shoot the approximate derivative...

            # First re-compute coef
            next_coef,_,_,_,_,LV,_,next_pen_llk = update_coef_gen_smooth(family,mus,y,Xs,coef,
                                                            coef_split_idx,S_emb,
                                                            c_llk,outer,max_inner,
                                                            min_inner,conv_tol)
            
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
            
            # Compute approximate!!! gradient of REML with respect to lambda
            # to check if step size needs to be reduced (part of step 6 in Wood, 2017).
            lam_grad = [grad_lambda(lgdetDs[lti],ldetHSs[lti],bsbs[lti],1) for lti in range(len(gamlss_pen))]
            lam_grad = np.array(lam_grad).reshape(-1,1) 
            check = lam_grad.T @ lam_delta

            # Now undo the acceleration if overall direction is **very** off - don't just check against 0 because
            # our criterion is approximate, so we can be more lenient (see Wood et al., 2017). 
            
            if check[0] < 1e-3*-abs(prev_pen_llk):
               for lti,lTerm in enumerate(gamlss_pen):
                  if was_extended[lti]:

                     lTerm.lam -= extend_by["acc"][lti]

               # Rebuild penalties
               S_emb,S_pinv,FS_use_rank = compute_S_emb_pinv_det(n_coef,gamlss_pen,"svd")

        #print([lterm.lam for lterm in gamlss_pen])
    
    # "Residuals"
    wres = y - Xs[0]@split_coef[0]

    # Total penalty
    penalty = coef.T@S_emb@coef

    # Calculate actual term-specific edf
    term_edfs = calculate_term_edf(gamlss_pen,term_edfs)
    
    return coef,etas,mus,wres,H,LV,total_edf,term_edfs,penalty[0,0]