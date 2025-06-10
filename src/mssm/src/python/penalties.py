import warnings
import numpy as np
import scipy as scp
from .custom_types import ConstType
from .matrix_solvers import translate_sparse
import copy

##################################### Penalty functions #####################################

def embed_in_S_sparse(pen_data,pen_rows,pen_cols,S_emb,S_col,SJ_col,cIndex):
   """Embed a term-specific penalty matrix (provided as elements, row and col indices) into the across-term penalty matrix (see Wood, 2017) """

   embedding = np.array(pen_data)
   r_embedding = np.array(pen_rows) + cIndex
   c_embedding = np.array(pen_cols) + cIndex

   if S_emb is None:
      S_emb = scp.sparse.csc_array((embedding,(r_embedding,c_embedding)),shape=(S_col,S_col))
   else:
      S_emb += scp.sparse.csc_array((embedding,(r_embedding,c_embedding)),shape=(S_col,S_col))

   return S_emb,cIndex+SJ_col

def embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,Sj,SJ_col):
   """Parameterize a term-specific penalty matrix (provided as elements, row and col indices)"""
   embedding = np.array(pen_data)

   if Sj is None:
      Sj = scp.sparse.csc_array((embedding,(pen_rows,pen_cols)),shape=(SJ_col,SJ_col))
   else:
      Sj += scp.sparse.csc_array((embedding,(pen_rows,pen_cols)),shape=(SJ_col,SJ_col))
      
   return Sj

def embed_shared_penalties(formulas):
   """
   Embed penalties from individual model into overall penalties for GAMLSS models.
   """

   shared_penalties = [copy.deepcopy(form.penalties) for form in formulas]

   # Assign original formula index to each penalty
   for fi in range(len(shared_penalties)):
      for lterm in shared_penalties[fi]:
         lterm.dist_param = fi

   for fi,form in enumerate(formulas):
      for ofi,other_form in enumerate(formulas):
         if fi == ofi:
               continue

         if ofi < fi:
               for lterm in shared_penalties[fi]:
                  lterm.S_J_emb = scp.sparse.vstack([scp.sparse.csc_array((other_form.n_coef,lterm.S_J_emb.shape[1])),
                                                   lterm.S_J_emb]).tocsc()
                  lterm.D_J_emb = scp.sparse.vstack([scp.sparse.csc_array((other_form.n_coef,lterm.S_J_emb.shape[1])),
                                                   lterm.D_J_emb]).tocsc()
                  
                  lterm.S_J_emb = scp.sparse.hstack([scp.sparse.csc_array((lterm.S_J_emb.shape[0],other_form.n_coef)),
                                                   lterm.S_J_emb]).tocsc()
                  
                  lterm.D_J_emb = scp.sparse.hstack([scp.sparse.csc_array((lterm.S_J_emb.shape[0],other_form.n_coef)),
                                                   lterm.D_J_emb]).tocsc()
                  
                  lterm.start_index += other_form.n_coef
         
         elif ofi > fi:
               for lterm in shared_penalties[fi]:
                  lterm.S_J_emb = scp.sparse.vstack([lterm.S_J_emb,
                                                   scp.sparse.csc_array((other_form.n_coef,lterm.S_J_emb.shape[1]))]).tocsc()
                  
                  lterm.D_J_emb = scp.sparse.vstack([lterm.D_J_emb,
                                                   scp.sparse.csc_array((other_form.n_coef,lterm.S_J_emb.shape[1]))]).tocsc()
                  
                  lterm.S_J_emb = scp.sparse.hstack([lterm.S_J_emb,
                                                   scp.sparse.csc_array((lterm.S_J_emb.shape[0],other_form.n_coef))]).tocsc()
                  
                  lterm.D_J_emb = scp.sparse.hstack([lterm.D_J_emb,
                                                   scp.sparse.csc_array((lterm.S_J_emb.shape[0],other_form.n_coef))]).tocsc()
               
   return shared_penalties

def diff_pen(n,constraint,m=2):
  # Creates difference (order=m) n*n penalty matrix
  # Based on code in Eilers & Marx (1996) and Wood (2017)

  D = np.diff(np.identity(n),m)
  S = D @ D.T
  rank = n - m # Eilers & Marx (1996): P-spline penalties consider m-degree polynomial as smooth, i.e., un-penalized!

  # ToDo: mgcv scales penalties - I wanted to do something
  # similar, but the approach below does not work.
  #FS = np.linalg.norm(S,1)
  #S = S / FS
  #D = D / FS**0.5

  # Absorb any identifiability constraints
  if constraint is not None:
     Z = constraint.Z
     if constraint.type == ConstType.QR:
       S = Z.T @ S @ Z
       D = Z.T @ D
     elif constraint.type == ConstType.DROP:
        S = np.delete(np.delete(S,Z,axis=1),Z,axis=0)
        D = np.delete(D,Z,axis=0)
     elif constraint.type == ConstType.DIFF:
        D = np.diff(np.concatenate((D[Z:D.shape[0],:],D[:Z,:]),axis=0),axis=0) # Correct for column differencing applied to X! See smoothCon help for mgcv (Wood, 2017)
        D = np.concatenate((D[D.shape[0]-Z:,:],D[:D.shape[0]-Z,:]),axis=0) 
        S = D @ D.T

        
  S = scp.sparse.csc_array(S)
  D = scp.sparse.csc_array(D)

  # Data in S and D is in canonical format, for competability this is translated to data, rows, columns
  pen_data,pen_rows,pen_cols = translate_sparse(S)
  chol_data,chol_rows,chol_cols = translate_sparse(D)

  return pen_data,pen_rows,pen_cols,chol_data,chol_rows,chol_cols,rank

def id_dist_pen(n,constraint,f=None):
  # Creates identity matrix penalty in case f(i) = 1
  # Can be used to create event-distance weighted penalty matrices for deconvolving sms GAMMs
  elements = [0.0 for _ in range(n)]
  idx = [0.0 for _ in range(n)]

  for i in range(n):
    if f is None:
      elements[i] = 1.0
    else:
      elements[i] = f(i+1)
    idx[i] = i

  return elements,idx,idx,elements,idx,idx,n # I' @ I = I; also identity is full rank

def TP_pen(S_j,D_j,j,ks,constraint):
   # Tensor smooth penalty - not including the reparameterization of Wood (2017) 5.6.2
   # but reflecting Eilers & Marx (2003) instead
   if j == 0:
      S_TP = S_j
      D_TP = D_j
   else:
      S_TP = scp.sparse.identity(ks[0])
      D_TP = scp.sparse.identity(ks[0])
      #m_rank *= ks[0] # Modify rank of marginal - identities are full-rank.
   
   for i in range(1,len(ks)):
      if j == i:
         S_TP = scp.sparse.kron(S_TP,S_j,format='csc')
         D_TP = scp.sparse.kron(D_TP,D_j,format='csc')
      else:
         S_TP = scp.sparse.kron(S_TP,scp.sparse.identity(ks[i]),format='csc')
         D_TP = scp.sparse.kron(D_TP,scp.sparse.identity(ks[i]),format='csc')
         #m_rank *= ks[i]
   
   if constraint is not None:
     Z = constraint.Z
     if constraint.type == ConstType.QR:
       S_TP = Z.T @ S_TP @ Z
       D_TP = Z.T @ D_TP
       S_TP = scp.sparse.csc_array(S_TP)
       D_TP = scp.sparse.csc_array(D_TP)
     elif constraint.type == ConstType.DROP:
        S_TP = scp.sparse.csc_array(np.delete(np.delete(S_TP.toarray(),Z,axis=1),Z,axis=0))
        D_TP = scp.sparse.csc_array(np.delete(D_TP.toarray(),Z,axis=0))
     elif constraint.type == ConstType.DIFF:
        D_TP = D_TP.toarray()
        D_TP = np.diff(np.concatenate((D_TP[Z:D_TP.shape[0],:],D_TP[:Z,:]),axis=0),axis=0) # Correct for column differencing applied to X! See smoothCon help for mgcv (Wood, 2017)
        D_TP = np.concatenate((D_TP[D_TP.shape[0]-Z:,:],D_TP[:D_TP.shape[0]-Z,:]),axis=0) 
        S_TP = D_TP @ D_TP.T
        S_TP = scp.sparse.csc_array(S_TP)
        D_TP = scp.sparse.csc_array(D_TP)

   pen_data,pen_rows,pen_cols = translate_sparse(S_TP)
   chol_data,chol_rows,chol_cols = translate_sparse(D_TP)
   return pen_data,pen_rows,pen_cols,chol_data,chol_rows,chol_cols
