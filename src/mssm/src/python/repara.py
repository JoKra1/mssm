import math
import numpy as np
import scipy as scp
from .matrix_solvers import eigen_solvers,map_csc_to_eigen,translate_sparse,cpp_chol,compute_Linv
from .penalties import embed_in_S_sparse
from .custom_types import LambdaTerm
import copy
import sys

def reparam(X:scp.sparse.csc_array|None,S:list[LambdaTerm],cov:np.ndarray|None,option:int=1,n_bins:int=30,QR:bool=False,identity:bool=False,scale:bool=False) -> tuple:
   """Options 1 - 3 are natural reparameterization discussed in Wood (2017; 5.4.2)
   with different strategies for the QR computation of :math:`\\mathbf{X}`. Option 4 helps with stabilizing the REML computation
   and is from Appendix B of Wood (2011) and section 6.2.7 in Wood (2017):

      1. Form complete matrix :math:`\\mathbf{X}` based on entire covariate.
      2. Form matrix :math:`\\mathbf{X}` only based on unique covariate values.
      3. Form matrix :math:`\\mathbf{X}` on a sample of values making up covariate. Covariate
         is split up into ``n_bins`` equally wide bins. The number of covariate values
         per bin is then calculated. Subsequently, the ratio relative to minimum bin size is
         computed and each ratio is rounded to the nearest integer. Then ``ratio`` samples
         are obtained from each bin. That way, imbalance in the covariate is approximately preserved when
         forming the QR.
      4. Transform term-specific :math:`\\mathbf{S}_{\\boldsymbol{\\lambda}}` based on Appendix B of Wood (2011) and section 6.2.7 in Wood (2017)
         so that they are full-rank and their log-determinant can be computed safely. In that case, only ``S`` needs
         to be provided and has to be a list holding the penalties to be transformed. If the transformation is to be applied to
         model matrices, coefficients, hessian, and covariance matrices X should be set to something other than ``None`` (does not matter what, can
         for example be the first model matrix.) The :func:`mssm.src.python.gamm_solvers.reparam_model` function can be used to apply the transformation and also
         returns the required transformation matrices to reverse it.
   
   For Options 1-3:

      If ``QR==True`` then :math:`\\mathbf{X}` is decomposed into :math:`\\mathbf{Q}\\mathbf{R}` directly via QR decomposition. Alternatively, we first
      form :math:`\\mathbf{X}^T\\mathbf{X}` and then compute the cholesky :math:`\\mathbf{L}` of this product - note that :math:`\\mathbf{L}^T = \\mathbf{R}`. Overall the latter
      strategy is much faster (in particular if ``option==1``), but the increased loss of precision in :math:`\\mathbf{L}^T = \\mathbf{R}` might not be ok for some.

      After transformation S only contains elements on it's diagonal and :math:`\\mathbf{X}` the transformed functions. As discussed
      in Wood (2017), the transformed functions are decreasingly flexible - so the elements on :math:`\\mathbf{S}` diagonal become smaller
      and eventually zero, for elements that are in the kernel of the original :math:`\\mathbf{S}` (un-penalized == not flexible).

      For a similar transformation (based solely on :math:`\\mathbf{S}`), Wood et al. (2013) show how to further reduce the diagonally
      transformed :math:`\\mathbf{S}` to an even simpler identity penalty. As discussed also in Wood (2017) the same behavior of decreasing
      flexibility if all entries on the diagonal of :math:`\\mathbf{S}` are 1 can only be maintained if the transformed functions are
      multiplied by a weight related to their wiggliness. Specifically, more flexible functions need to become smaller in
      amplitude - so that for the same level of penalization they are removed earlier than less flexible ones. To achieve this
      Wood further post-multiply the transformed matrix :math:`\\mathbf{X}'` with a matrix that contains on it's diagonal the reciprocal of the
      square root of the transformed penalty matrix (and 1s in the last cells corresponding to the kernel). This is done here
      if ``identity=True``.

      In ``mgcv`` the transformed model matrix and penalty can optionally be scaled by the root mean square value of the transformed
      model matrix (see the nat.param function in mgcv). This is done here if ``scale=True``.

   For Option 4:

      Option 4 enforces re-parameterization of term-specific :math:`\\mathbf{S}_{\\boldsymbol{\\lambda}}` based on section Wood (2011) and section 6.2.7 in Wood (2017).
      In ``mssm`` multiple penalties can be placed on individual terms (i.e., tensor terms, random smooths, Kernel penalty) but
      it is not always the case that the term-specific :math:`\\mathbf{S}_{\\boldsymbol{\\lambda}}` - i.e., the sum over all those individual penalties multiplied with
      their :math:`\\lambda` parameters, is of full rank. If we need to form the inverse of the term-specific :math:`\\mathbf{S}_{\\boldsymbol{\\lambda}}` this is problematic.
      It is also problematic, as discussed by Wood (2011), if the different :math:`\\lambda` are all of different magnitude in which case forming
      the term-specific :math:`log(|\\mathbf{S}_{\\boldsymbol{\\lambda}}|+)` becomes numerically difficult.

      The re-parameterization implemented by option 4, based on Appendix B in Wood (2011), solves these issues. After this re-parameterization a
      term-specific :math:`\\mathbf{S}_{\\boldsymbol{\\lambda}}` has been formed that is full rank. And :math:`log(|\\mathbf{S}_{\\boldsymbol{\\lambda}}|)` - no longer just a generalized determinant - can be
      computed without running into numerical problems.

      The strategy by Wood (2011) could be applied to form an overall - not just term-specific - :math:`\\mathbf{S}_{\\boldsymbol{\\lambda}}` with these properties. However, this
      does not work for general smooth models as defined by Wood et al. (2016). Hence, mssm opts for the blockwise strategy.
      However, in ``mssm`` penalties currently cannot overlap, so this is not necessary at the moment.

   References:
      - Wood, S. N., (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models.
      - Wood, S. N., Scheipl, F., & Faraway, J. J. (2013). Straightforward intermediate rank tensor product smoothing in mixed models.
      - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
      - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
      - mgcv source code (accessed 2024). smooth.R file, nat.param function.

   :param X: Model/Term matrix or None
   :type X: scp.sparse.csc_array | None
   :param S: List of penalties
   :type S: list[LambdaTerm]
   :param cov: covariate array associated with a specific term or None
   :type cov: np.ndarray | None
   :param option: Which re-parameterization to compute, defaults to 1
   :type option: int, optional
   :param n_bins: Number of bins to use as part of option 3, defaults to 30
   :type n_bins: int, optional
   :param QR: Whether to rely on a QR decomposition or not (then a Cholesky is used) as part of options 1-3, defaults to False
   :type QR: bool, optional
   :param identity: Whether the penalty matrix should be transformed to identity as part of options 1-3, defaults to False
   :type identity: bool, optional
   :param scale: Whether the penalty matrix and term matrix should be scaled as part of options 1-3, defaults to False
   :type scale: bool, optional
   :return: Return object content depends on ``option`` but will usually hold informations to apply/undo the required re-parameterization as well as already re-parameterized objects.
   :rtype: tuple
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
         R = scp.sparse.csr_array(R)
         
      else:
         XX = (X.T @ X).tocsc()
         
         L,code = eigen_solvers.chol(*map_csc_to_eigen(XX))

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
      
      B = eigen_solvers.solve_tr(*map_csc_to_eigen(R.T),eigen_solvers.solve_tr(*map_csc_to_eigen(R.T),S.T).T)

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
      
      C = eigen_solvers.backsolve_tr(*map_csc_to_eigen(R.tocsc()),U)

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
   
   elif option == 4:
      # Reparameterize S_\\lambda for safe REML evaluation - based on section 6.2.7 in Wood (2017) and Appendix B of Wood (2011).
      # S needs to be list holding penalties.

      # We first sort S into term-specific groups of penalties
      SJs = [] # SJs groups per term
      ljs = [] # Lambda groups per term
      SJ_reps = [] # How often should SJ be stacked
      SJ_term_idx = [] # Penalty index of every SJ
      SJ_idx = [] # start index of every SJ
      SJ_coef = [] # Number of coef in the term penalized by a (sum of) SJ
      SJ_idx_max = 0 # Max starting index - used to decide whether a new SJ should be added.
      SJ_idx_len = 0 # Number of separate SJ blocks.

      # Build S_emb and collect every block for pinv(S_emb)
      for lti,lTerm in enumerate(S):
         #print(f"pen: {lti}")

         # Now collect the S_J for the pinv calculation in the next step.
         if lti == 0 or lTerm.start_index > SJ_idx_max:
            SJs.append([lTerm.S_J])
            ljs.append([lTerm.lam])
            SJ_term_idx.append([lti])
            SJ_reps.append(lTerm.rep_sj)
            SJ_coef.append(lTerm.S_J.shape[1])
            SJ_idx.append(lTerm.start_index)
            SJ_idx_max = lTerm.start_index
            SJ_idx_len += 1

         else: # A term with the same starting index exists already - so add the Sjs to the corresponding list
            idx_match = [idx for idx in range(SJ_idx_len) if SJ_idx[idx] == lTerm.start_index]
            #print(idx_match,lTerm.start_index)
            if len(idx_match) > 1:
               raise ValueError("Penalty index matches multiple previous locations.")
            
            SJs[idx_match[0]].append(lTerm.S_J)
            ljs[idx_match[0]].append(lTerm.lam)
            SJ_term_idx[idx_match[0]].append(lti)

            if SJ_reps[idx_match[0]] != lTerm.rep_sj:
               raise ValueError("Repeat Number for penalty does not match previous penalties repeat number.")
      
      # Now we can re-parameterize SJ groups of length > 1
      Sj_reps = [] # Will hold transformed S_J
      Q_reps = []
      QT_reps = []
      for pen in S:
         Sj_reps.append(LambdaTerm(S_J=copy.deepcopy(pen.S_J),rep_sj=pen.rep_sj,lam=pen.lam,type=pen.type,rank=pen.rank,term=pen.term,start_index=pen.start_index,dist_param=pen.dist_param))

      S_reps = [] # Term specific S_\\lambda
      eps = sys.float_info.epsilon**0.7
      Mp = Sj_reps[0].start_index # Number of un-penalized dimensions (Kernel space dimension of total S_\\lambda; Wood, 2011)
      for grp_idx,SJgroup,LJgroup in zip(SJ_term_idx,SJs,ljs):

         for rpidx in range(len(SJgroup)):
            Sj_reps[grp_idx[rpidx]].rp_idx = len(S_reps)

         normed_SJ = [SJ_mat/scp.sparse.linalg.norm(SJ_mat,ord='fro') for SJ_mat in SJgroup]
         normed_S = normed_SJ[0]
         for j in range(1,len(normed_SJ)):
            normed_S += normed_SJ[j]

         D, U = scp.linalg.eigh(normed_S.toarray())

         # Decreasing magnitude for ease of indexing..
         D = np.flip(D)
         U = scp.sparse.csc_array(np.flip(U,axis=1))

         Ur = U[:,D > max(D)*eps]
         r = Ur.shape[1] # Range dimension
         Mp += (normed_S.shape[1] - r)

         # Initial re-parameterization as discussed by Wood (2011)
         if r < normed_S.shape[1]:
            SJbar = [Ur.T@SJ_mat@Ur for SJ_mat in SJgroup]

            if not X is None:
               Q_rep0 = copy.deepcopy(U) # Need to account for this when computing Q matrix.
         else:
            SJbar = copy.deepcopy(SJgroup)
            Q_rep0 = None
            
         if len(SJgroup) == 1: # No further re-parameterization necessary
            Sj_reps[grp_idx[0]].S_J = SJbar[0]
            S_reps.append(SJbar[0]*LJgroup[0])

            if not X is None:
               if not Q_rep0 is None:
                  Q_reps.append(scp.sparse.csc_array(Q_rep0))
                  QT_reps.append(scp.sparse.csc_array(Q_rep0.T))
               else:
                  Q_reps.append(scp.sparse.eye(r,format='csc'))
                  QT_reps.append(scp.sparse.eye(r,format='csc'))
            continue

         S_Jrep = copy.deepcopy(SJbar) # Original S_J transformed 
         S_rep = None
         Q_rep = None

         # Initialization as described by Wood (2011)
         K = 0
         Q = r
         rem_idx = np.arange(0,len(SJbar),dtype=int)

         while True:
            # Find max norm
            norm_group = [scp.sparse.linalg.norm(SJ_mat*SJ_lam,ord='fro') for SJ_mat,SJ_lam in zip([SJbar[idx] for idx in rem_idx],[LJgroup[idx] for idx in rem_idx])]
            
            am_norm = np.argmax(norm_group)

            # Find Sk = S_J with max norm, in Wood (2011) this can be multiple but we just use the max. as discussed in Wood (2017)
            Sk = SJbar[rem_idx[am_norm]]*LJgroup[rem_idx[am_norm]]

            # De-compose Sk - form Ur and Un
            D, U =scp.linalg.eigh(Sk.toarray())

            # Decreasing magnitude for ease of indexing..
            D = np.flip(D)
            U = scp.sparse.csc_array(np.flip(U,axis=1))
            r = len(D[D > max(D)*eps]) # r is known in advance here almost every-time but we need to de-compose anyway..
            
            if r == Q:
               break
            
            # Seperate U into kernel and rank space of Sk
            n = U.shape[1] - r
            Un = U[:,r:]
            Un = Un.reshape(Sk.shape[1],-1)

            Ur = U[:,:r]
            Ur = Ur.reshape(Sk.shape[1],-1)

            Dr = np.diag(D[:r])
            
            # Sum up Sjk - remaining S_J not Sk
            Sjk = None
            for j in rem_idx:
                  if j == rem_idx[am_norm]:
                     continue

                  if Sjk is None:
                     Sjk = SJbar[j]*LJgroup[j]
                  else:
                     Sjk += SJbar[j]*LJgroup[j]
            
            if not Sjk is None:
                  Sjk = Sjk.toarray()
            else:
                  raise ValueError("Problem during re-parameterization.")
                  
            # Compute C' in Wood (2011)

            # Transform Sk
            Sk = Dr + Ur.T@Sjk@Ur
            
            # Fill C' in Wood (2011)
            #print(Sk.shape,(Ur.T@Sjk@Un).shape,(Un.T@Sjk@Ur).shape,(Un.T@Sjk@Un).shape)

            C = np.concatenate((np.concatenate((Sk,Un.T@Sjk@Ur),axis=0),
                                np.concatenate((Ur.T@Sjk@Un,Un.T@Sjk@Un),axis=0)),axis=1)
            
            #print(C.shape)

            # Form S' in Wood (2011)
            if S_rep is None:
                  S_rep = C
                  Ta = np.concatenate((Ur.toarray(),np.zeros((Ur.shape[0],n))),axis=1)
                  Tg = U.toarray()
                  
                  if not X is None:
                     if Q_rep0 is None:
                        Q_rep = U.toarray()
                     else:
                        init_drop = Q_rep0.shape[1] - U.shape[1]
                        
                        Q_rep = np.concatenate((np.concatenate((U.toarray(),np.zeros((U.shape[0],init_drop))),axis=1),
                                                np.concatenate((np.zeros((init_drop,U.shape[1])),np.identity(init_drop)),axis=1)),axis=0)
            else:
                  A = S_rep[:K,:K] # From partitioning step 6 in Wood, 2011
                  B = S_rep[:K,K:] # From partitioning step 6 in Wood, 2011
                  BU = B @ U
                  S_rep = np.concatenate((np.concatenate((A,BU.T),axis=0),
                                          np.concatenate((BU,C),axis=0)),axis=1)
                  
                  Ta = np.concatenate((np.concatenate((np.identity(K),np.zeros((K,r+n))),axis=1),
                                       np.concatenate((np.zeros((Ur.shape[0],K)),Ur.toarray(),np.zeros((Ur.shape[0],n))),axis=1)),axis=0)
                  
                  Tg = np.concatenate((np.concatenate((np.identity(K),np.zeros((K,r+n))),axis=1),
                                       np.concatenate((np.zeros((U.shape[0],K)),U.toarray()),axis=1)),axis=0)
                  
                  if not X is None:
                     if Q_rep0 is None:
                        Q_rep = Q_rep @ Tg
                     else:
                        init_drop = Q_rep0.shape[1] - Tg.shape[1]
                        Q_rep = Q_rep @ np.concatenate((np.concatenate((Tg,np.zeros((Tg.shape[0],init_drop))),axis=1),
                                                        np.concatenate((np.zeros((init_drop,Tg.shape[1])),np.identity(init_drop)),axis=1)),axis=0)

            #print(Ta.shape,Tg.shape)
            # Transform remaining terms that made up Sjk
            for j in rem_idx:
                  if j == rem_idx[am_norm]:
                     S_Jrep[j] = Ta.T@S_Jrep[j]@Ta
                     continue
                  SJbar[j] = scp.sparse.csc_array(Un.T@SJbar[j]@Un)
                  S_Jrep[j] = Tg.T@S_Jrep[j]@Tg

            K += r
            Q -= r
            rem_idx = np.delete(rem_idx,am_norm)


         for j in range(len(grp_idx)):
            
            Sj_reps[grp_idx[j]].S_J = scp.sparse.csc_array(S_Jrep[j])

         if S_rep is None: # Fix r==Q in first iteration
            S_rep = SJbar[0]*LJgroup[0]
            if not X is None:
               Q_rep = scp.sparse.eye(r,format='csc')

         S_reps.append(scp.sparse.csc_array(S_rep))

         if not X is None:
            if not Q_rep0 is None:
               Q_reps.append(scp.sparse.csc_array(Q_rep0@Q_rep))
               QT_reps.append(scp.sparse.csc_array(Q_rep.T@Q_rep0.T))
            else:
               Q_reps.append(scp.sparse.csc_array(Q_rep))
               QT_reps.append(scp.sparse.csc_array(Q_rep.T))
         
      return Sj_reps,S_reps,SJ_term_idx,SJ_idx,SJ_coef,Q_reps,QT_reps,Mp
         
   else:
      raise NotImplementedError(f"Requested option {option} for reparameterization is not implemented.")
   
def reparam_model(dist_coef:list[int], dist_up_coef:list[int], coef:np.ndarray, split_coef_idx:list[int], Xs:list[scp.sparse.csc_array], penalties:list[LambdaTerm], form_inverse:bool=True, form_root:bool=True, form_balanced:bool=True, n_c:int=1) -> tuple[np.ndarray, list[scp.sparse.csc_array], list[LambdaTerm], scp.sparse.csc_array, scp.sparse.csc_array | None, scp.sparse.csc_array | None, scp.sparse.csc_array | None, scp.sparse.csc_array, list[scp.sparse.csc_array]]:
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
    :type Xs: [scp.sparse.csc_array]
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
    :return: A tuple with 9 elements: the re-parameterized coefficient vector, a list with the re-parameterized model matrices, a list of the penalties after re-parameterization, the total re-parameterized penalty matrix, optionally the balanced version of the former, optionally a root of the re-parameterized total penalty matrix, optionally the inverse of the re-parameterized total penalty matrix, the transformation matrix ``Q`` so that ``Q.T@S_emb@Q = S_emb_rp`` where ``S_emb`` and ``S_emb_rp`` are the total penalty matrix before and after re-parameterization, a list of transformation matrices ``QD`` so that ``XD@QD=XD_rp`` where ``XD`` and ``XD_rp`` are the model matrix of the Dth linear predictor before and after re-parameterization.
    :rtype: tuple[np.ndarray, list[scp.sparse.csc_array], list[LambdaTerm], scp.sparse.csc_array, scp.sparse.csc_array | None, scp.sparse.csc_array | None, scp.sparse.csc_array | None, scp.sparse.csc_array, list[scp.sparse.csc_array]]
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
        if (form_inverse and len(SJ_term_idx[Si]) > 1) or form_root:
            L,code = cpp_chol(S_rep.tocsc())
            if code != 0:
                raise ValueError("Inverse of transformed penalty could not be computed.")
        
        # Form inverse - only if this is not a single penalty term
        if form_inverse and len(SJ_term_idx[Si]) > 1:
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
            if form_inverse and len(SJ_term_idx[Si]) > 1:
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