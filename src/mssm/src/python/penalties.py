import warnings
import numpy as np
import scipy as scp
from dataclasses import dataclass
from enum import Enum

class PenType(Enum):
    IDENTITY = 1
    DIFFERENCE = 2
    DISTANCE = 3

##################################### Penalty functions #####################################

def translate_sparse(mat):
  # Translate canonical sparse csc matrix representation into data, row, col representation
  # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.html#scipy.sparse.csc_array
  elements = mat.data
  idx = mat.indices
  iptr = mat.indptr

  data = []
  rows = []
  cols = []

  for ci in range(mat.shape[1]):
     
     c_data = elements[iptr[ci]:iptr[ci+1]]
     c_rows = idx[iptr[ci]:iptr[ci+1]]

     data.extend(c_data)
     rows.extend(c_rows)
     cols.extend([ci for _ in range(len(c_rows))])

  return data, rows, cols

def diff_pen(n,m=2,Z=None):
  # Creates difference (order=m) n*n penalty matrix
  # Based on code in Eilers & Marx (1996) and Wood (2017)

  D = np.diff(np.identity(n),m)
  S = D @ D.T

  # ToDo: mgcv scales penalties - I wanted to do something
  # similar, but the approach below does not work.
  #FS = np.linalg.norm(S,1)
  #S = S / FS
  #D = D / FS**0.5

  # Absorb any identifiability constraints
  if Z is not None:
     S = Z.T @ S @ Z
     D = Z.T @ D
  
  S = scp.sparse.csc_array(S)
  D = scp.sparse.csc_array(D)

  # Data in S and D is in canonical format, for competability this is translated to data, rows, columns
  pen_data,pen_rows,pen_cols = translate_sparse(S)
  chol_data,chol_rows,chol_cols = translate_sparse(D)

  return pen_data,pen_rows,pen_cols,chol_data,chol_rows,chol_cols

def id_dist_pen(n,f=None):
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

  return elements,idx,idx,elements,idx,idx # I' @ I = I

def TP_pen(S_j,D_j,j,ks):
   # Tensor smooth penalty - not including the reparameterization of Wood (2017) 5.6.2
   # but reflecting Eilers & Marx (2003) instead
   if j == 0:
      S_TP = S_j
      D_TP = D_j
   else:
      S_TP = scp.sparse.identity(ks[0])
      D_TP = scp.sparse.identity(ks[0])
   
   for i in range(1,len(ks)):
      if j == i:
         S_TP = scp.sparse.kron(S_TP,S_j,format='csc')
         D_TP = scp.sparse.kron(D_TP,D_j,format='csc')
      else:
         S_TP = scp.sparse.kron(S_TP,scp.sparse.identity(ks[i]),format='csc')
         D_TP = scp.sparse.kron(D_TP,scp.sparse.identity(ks[i]),format='csc')
   
   pen_data,pen_rows,pen_cols = translate_sparse(S_TP)
   chol_data,chol_rows,chol_cols = translate_sparse(D_TP)
      
   return pen_data,pen_rows,pen_cols,chol_data,chol_rows,chol_cols

@dataclass
class LambdaTerm:
  # Lambda term storage. Can hold multiple penalties associated with a single lambda
  # value!
  # start_index can be useful in case we want to have multiple penalties on some
  # coefficients (see Wood, 2017; Wood & Fasiolo, 2017).
  S_J:scp.sparse.csc_array=None
  S_J_emb:scp.sparse.csc_array=None
  D_J_emb:scp.sparse.csc_array=None
  rep_sj:int=1
  lam:float = 1.1
  start_index:int = None
  frozen:bool = False
  type:PenType = None