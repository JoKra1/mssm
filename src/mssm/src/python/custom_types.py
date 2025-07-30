from dataclasses import dataclass
from enum import Enum
import scipy as scp
import numpy as np

################################################################## Enums/Types ##################################################################
class PenType(Enum):
    """Custom Penalty data type used by internal functions.
    """
    IDENTITY = 1
    DIFFERENCE = 2
    DISTANCE = 3
    REPARAM1 = 4 # Approximate Demmler & Reinsch (1975) parameterization for univariate smooths; see section 5.4.2 of Wood (2017)
    NULL = 5
    DERIVATIVE = 6
    COEFFICIENTS = 7
    CUSTOM = 8

class ConstType(Enum):
    """Custom Constraint data type used by internal functions.
    """
    DROP = 1
    QR = 2
    DIFF = 3

class VarType(Enum):
    """Custom variable data type used by internal functions.
    """
    NUMERIC = 1
    FACTOR = 2

class TermType(Enum):
  """Custom Term data type used by internal functions.
    """
  IRSMOOTH = 1
  SMOOTH = 2
  LINEAR = 3
  RANDINT = 4
  RANDSLOPE = 5


################################################################## Data-classes ##################################################################

@dataclass
class LambdaTerm:
  """:math:`\\lambda` storage term.

  Usually ``model.overall_penalties`` holds a list of these.

  :ivar scp.sparse.csc_array S_J: The penalty matrix associated with this lambda term. Note, in case multiple penalty matrices share the same lambda value, the ``rep_sj`` argument determines how many diagonal blocks we need to fill with this penalty matrix to get ``S_J_emb``. Initialized with ``None``.
  :ivar scp.sparse.csc_array S_J_emb: A zero-embedded version of the penalty matrix associated with this lambda term. Note, this matrix contains ``rep_sj`` diagonal sub-blocks each filled with ``S_J``. Initialized with ``None``.
  :ivar scp.sparse.csc_array D_J_emb: Root of ``S_J_emb``, so that ``D_J_emb@D_J_emb.T=S_J_emb``. Initialized with ``None``.
  :ivar int rep_sj: How many sequential sub-blocks of ``S_J_emb`` need to be filled with ``S_J``. Useful if all levels of a categorical variable for which a separate smooth is to be estimated are assumed to share the same lambda value. Initialized with 1.
  :ivar float lam: The current estimate for :math:`\\lambda`. Initialized with 1.1.
  :ivar int start_index: The first row and column in the overall penalty matrix taken up by ``S_J``. Initialized with ``None``.
  :ivar PenType type: The type of this penalty term. Initialized with ``None``.
  :ivar int rank: The rank of ``S_J``. Initialized with ``None``.
  :ivar int term: The index of the term in a :class:`mssm.src.python.formula.Formula` with which this penalty is associated. Initialized with ``None``.
  """
  # Lambda term storage. Can hold multiple penalties associated with a single lambda
  # value!
  # start_index can be useful in case we want to have multiple penalties on some
  # coefficients (see Wood, 2017; Wood & Fasiolo, 2017).
  S_J:scp.sparse.csc_array|None=None
  S_J_emb:scp.sparse.csc_array|None=None
  D_J_emb:scp.sparse.csc_array|None=None
  rep_sj:int=1
  lam:float = 1.1
  start_index:int|None = None
  frozen:bool = False
  type:PenType|None = None
  rank:int | None = None
  term:int | None = None
  clust_series: list[int] | None = None
  clust_weights:list[list[float]] | None = None
  dist_param: int = 0
  rp_idx: int | None = None
  S_J_lam:scp.sparse.csc_array | None=None

@dataclass
class Reparameterization:
   """Holds information necessary to re-parameterize a smooth term.

   :ivar scp.sparse.csc_array Srp: The transformed penalty matrix
   :ivar scp.sparse.csc_array Drp: The root of the transformed penalty matrix
   :ivar scp.sparse.csc_array C: Transformation matrix for model matrix and/or penalty.
   """
   # Holds all information necessary to transform model matrix & penalty via various re-parameterization strategies as discussed in Wood (2017).
   Srp:scp.sparse.csc_array|None = None # Transformed penalty
   Drp:scp.sparse.csc_array|None = None # Transformed root of penalty
   C:scp.sparse.csc_array|None= None
   scale:float|None = None
   IRrp:scp.sparse.csc_array|None = None
   rms1:float|None = None
   rms2:float|None = None
   rank:int|None = None

@dataclass
class Fit_info:
   """Holds information related to convergence (speed) for GAMMs, GAMMLSS, and GSMMs.

   :ivar int lambda_updates: The total number of lambda updates computed during estimation. Initialized with 0.
   :ivar int iter: The number of outer iterations (a single outer iteration can involve multiple lambda updates) completed during estimation. Initialized with 0.
   :ivar int code: Convergence status. Anything above 0 indicates that the model did not converge and estimates should be considered carefully. Initialized with 1.
   :ivar float eps: The fraction added to the last estimate of the negative Hessian of the penalized likelihood during GAMMLSS or GSMM estimation. If this is not 0 - the model should not be considered as converged, irrespective of what ``code`` indicates. This most likely implies that the model is not identifiable. Initialized with ``None`` and ignored for GAMM estimation.
   :ivar float K2: An estimate for the condition number of matrix ``A``, where ``A.T@A=H`` and ``H`` is the final estimate of the negative Hessian of the penalized likelihood. Only available if ``check_cond>0`` when ``model.fit()`` is called for any model (i.e., GAMM, GAMMLSS, GSMM). Initialized with ``None``.
   :ivar [int] dropped: The final set of coefficients dropped during GAMMLSS/GSMM estimation when using ``method in ["QR/Chol","LU/Chol","Direct/Chol"]`` or ``None`` in which case no coefficients were dropped. Initialized with 0.
   """
   lambda_updates:int=0
   iter:int=0
   code:int=1
   eps:float | None = None
   K2:float | None = None
   dropped:list[int] | None = None

@dataclass
class Constraint:
  """
  Constraint storage. ``Z``, either holds the Qr-based correction matrix that needs to be multiplied with :math:`\\mathbf{X}`, :math:`\\mathbf{S}`,
  and :math:`\\mathbf{D}` (where :math:`\\mathbf{D}\\mathbf{D}^T = \\mathbf{S}`) to make terms subject to the conventional sum-to-zero constraints
  applied also in mgcv (Wood, 2017), the column/row that should be dropped from those - then :math:`\\mathbf{X}` can also no longer take
  on a constant, or ``None`` indicating that the model should be "difference re-coded" to enable sparse sum-to-zero
  constraints. The latter two are available in mgcv's ``smoothCon`` function by setting the ``sparse.cons`` argument to 1 or
  2 respectively.

  The QR-based approach is described in detail by Wood (2017) and is similar to just mean centering every basis function involved in the
  smooth and then dropping one column from the corresponding centered model matrix. The column-dropping approach is self-explanatory. The
  difference re-coding re-codes bases functions to correspond to differences of bases functions. The resulting basis remains sparser than the
  alternatives, but this is not a true centering constraint: :math:`f(x)` will not necessarily be orthogonal to the intercept,
  i.e., :math:`\\mathbf{1}^T \\mathbf{f(x)}` will not necessarily be 0. Hence, confidence intervals will usually be wider when using
  ConstType.DIFF (also when using ConstType.DROP, for the same reason) instead of ``ConstType.QR`` (see Wood; 2017,2020)!

  A final note regards the use of tensor smooths when ``te==False``. Since the value of any constant estimated for a smooth depends on the type of constraint used, the marginal functions estimated
  for the "main effects" (:math:`f(x)`, :math:`f(z)`) and "interaction effect" (:math:`f(x,z)`) in a model: :math:`y = a + f(x) + f(z) + f(x,z)` will differ depending on the type of constraint used. The "Anova-like" decomposition
  described in detail in Wood (2017) is achievable only when using ``ConstType.QR``.

  Thus, ``ConstType.QR`` is the default by all ``mssm`` functions, and the other two options should be considered experimental.

  References:
    - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
    - Wood, S. N. (2020). Inference and computation with generalized additive models and their extensions. TEST, 29(2), 307–339. https://doi.org/10.1007/s11749-020-00711-5
    - Eilers, P. H. C., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. Statistical Science, 11(2), 89–121. https://doi.org/10.1214/ss/1038425655

  """
  Z:np.ndarray|int|None = None
  type:ConstType|None = None