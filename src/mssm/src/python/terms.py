from collections.abc import Callable
from enum import Enum
from itertools import combinations
import copy
import sys
import numpy as np
import scipy as scp
from itertools import repeat
from . import smooths
from . import penalties
from .custom_types import PenType, LambdaTerm,Constraint,ConstType,TermType,VarType,Reparameterization
from .repara import reparam
from .matrix_solvers import translate_sparse,warnings
from .penalties import embed_in_S_sparse,embed_in_Sj_sparse,TP_pen,adjust_pen_drop,Penalty,DifferencePenalty,IdentityPenalty
from .smooths import TP_basis_calc

class GammTerm():
   """Base-class implemented by the terms passed to :class:`mssm.src.python.formula.Formula`.

   :param variables: List of variables as strings.
   :type variables: [str]
   :param type: Type of term as enum
   :type type: TermType
   :param is_penalized: Whether the term is penalized/can be penalized or not
   :type is_penalized: bool
   :param penalty: The default penalties associated with a term.
   :type penalty: [Penalty]
   :param pen_kwargs: A list of dictionaries, each with key-word arguments passed to the construction of the corresponding  :class:`Penalty` in ``penalty``.
   :type pen_kwargs: [dict]
   """
   
   def __init__(self,variables:list[str],
                type:TermType,
                is_penalized:bool,
                penalty:list[Penalty],
                pen_kwargs:list[dict]) -> None:
        
        self.variables = variables
        self.type = type
        self.is_penalized = is_penalized
        self.penalty = penalty
        self.pen_kwargs = pen_kwargs
        self.name = None

   def build_penalty(self,penalties:list[LambdaTerm],cur_pen_idx:int,*args,**kwargs) -> tuple[list[LambdaTerm],int]:
      """Builds a penalty matrix associated with this term and returns an updated ``penalties`` list including it.

      This method is implemented by most implementations of the :class:`GammTerm` class.
      Two arguments need to be returned: the updated ``penalties`` list including the new penalty implemented as a :class:`LambdaTerm` and the updated ``cur_pen_idx``.
      The latter simply needs to be incremented for every penalty added to ``penalties``.

      :param penalties: List of previosly created penalties.
      :type penalties: [LambdaTerm]
      :param cur_pen_idx: Index of the last element in ``penalties``.
      :type cur_pen_idx: int
      :return: Updated ``penalties`` list including the new penalty implemented as a :class:`LambdaTerm` and the updated ``cur_pen_idx``
      :rtype: tuple[list[LambdaTerm],int]
      """
      return penalties, cur_pen_idx
   
   def build_matrix(self,*args,**kwargs):
      """Builds the design/term/model matrix associated with this term and returns it represented as a list of values, a list of row indices, and a list of column indices.

      This method is implemented by every implementation of the :class:`GammTerm` class.
      The returned lists can then be used to create a sparse matrix for this term. Also returns the number of additional columnsthat would be added to the total model matrix by this term.
      """
      pass

   def get_coef_info(self,*args,**kwargs):
      """Returns the total number of coefficients associated with this term, the number of unpenalized coefficients associated with this term, and a list with names for each of the coefficients associated with this term.

      This method is implemented by every implementation of the :class:`GammTerm` class.
      """
      pass     

class i(GammTerm):
    """
    An intercept/offset term. In a model
    
    .. math::
       \\mu_i = a + f(x_i)
       
    it reflects :math:`a`.

    """

    def __init__(self) -> None:
        super().__init__(["1"], TermType.LINEAR, False, [], [])
        self.name = "Intercept"
    
    def build_matrix(self, ci:int, ti:int, ridx:np.ndarray, use_only:list[int]) -> tuple[list[float],list[int],list[int],int]:
      """Builds the design/term/model matrix for an intercept term.

      :param ci: Current column index.
      :type ci: int
      :param ti: Index corresponding to the position the current term (i.e., self) takes on in the list of terms of the Formula.
      :type ti: int
      :param ridx: Array of non NAN rows in the data.
      :type ridx: np.ndarray
      :param use_only: A list holding term indices for which the matrix should be formed. For terms not included in this list a zero matrix will be returned. Can be set to ``None`` so that no terms are excluded.
      :type use_only: [int]
      :return: matrix data, matrix row indices, matrix column indices, added columns
      :rtype: tuple[list[float],list[int],list[int],int]
      """
      n_y = len(ridx)
      offset = np.ones(n_y)

      if use_only is None or ti in use_only:
        return offset, ridx, [ci for _ in range(n_y)], 1
      
      return [], [], [], 1
    
    def get_coef_info(self) -> tuple[int,int,list[str]]:
      """Returns the total number of coefficients associated with this term, the number of unpenalized coefficients associated with this term, and a list with names for each of the coefficients associated with this term.

      :return: Number of coefficients associated with term, number of un-penalized coefficients associated with term, coef names
      :rtype: tuple[int,int,list[str]]
      """
      return 1, 1, ["Intercept"]
       

class f(GammTerm):
    """
    A univariate or tensor interaction smooth term. If ``variables`` only contains a
    single variable :math:`x`, this term will represent a univariate :math:`f(x)` in a model:
    
    .. math::

      \\mu_i = a + f(x_i)
   
    For example, the model below in ``mgcv``:

    ::

      bam(y ~ s(x,k=10) + s(z,k=20))

    would be expressed as follows in ``mssm``:

    ::

      GAMM(Formula(lhs("y"),[i(),f(["x"],nk=9),f(["z"],nk=19)]),Gaussian())
      
    If ``variables`` contains two variables :math:`x` and :math:`z`, then this term will either represent
    the tensor interaction :math:`f(x,z)` in model:
     
    .. math::

      \\mu_i = a + f(x_i) + f(z_i) + f(x_i,z_i)
    
    or in model:
    
    .. math::

      \\mu_i = a + f(x_i,z_i)

    The first behavior is achieved by setting ``te=False``. In that case it is necessary
    to add 'main effect' ``f`` terms for :math:`x` and :math:`y`. In other words, the behavior then mimicks
    the ``ti()`` term available in ``mgcv`` (Wood, 2017). If ``te=True``, the term instead behaves like
    a ``te()`` term in ``mgcv``, so no separate smooth effects for the main effects need to be included.

    For example, the model below in ``mgcv``:

    ::

      bam(y ~ te(x,z,k=10))

    would be expressed as follows in ``mssm``:

    ::

      GAMM(Formula(lhs("y"),[i(),f(["x","z"],nk=9,te=True)]),Gaussian())

    In addition, the model below in ``mgcv``:

    ::

      bam(y ~ s(x,k=10) + s(z,k=20) + ti(x,z,k=10))

    would be expressed as follows in ``mssm``:

    ::

      GAMM(Formula(lhs("y"),[i(),f(["x"],nk=9),f(["z"],nk=19),f(["x","z"],nk=9,te=False)]),Gaussian())

    By default a B-spline basis is used with ``nk=9`` basis functions (after removing identifiability
    constrains). This is equivalent to ``mgcv``'s default behavior of using 10 basis functions
    (before removing identifiability constrains). In case ``variables`` contains more then one variable
    ``nk`` can either bet set to a single value or to a list containing the number of basis functions
    that should be used to setup the spline matrix for every variable. The former implies that the same
    number of coefficients should be used for all variables. Keyword arguments that change the computation of
    the spline basis can be passed along via a dictionary to the ``basis_kwargs`` argument. Importantly, if
    multiple variables are present and a list is passed to ``nk``, a list of dictionaries with keyword arguments
    of the same length needs to be passed to ``basis_kwargs`` as well.

    Multiple penalties can be placed on every term by adding ``Penalty`` to the ``penalties``
    argument. In case ``variables`` contains multiple variables a separate tensor penalty (see Wood, 2017) will
    be created for every penalty included in ``penalties``. Again, key-word arguments that alter the behavior of
    the penalty creation need to be passed as dictionaries to ``pen_kwargs`` for every penalty included in ``penalties``.
    By default, a univariate term is penalized with a difference penalty of order 2 (Eilers & Marx, 2010). 

    References:

     - Eilers, P., & Marx, B. (2010). Splines, knots, and penalties. https://doi.org/10.1002/WICS.125
     - Marra, G., & Wood, S. N. (2011). Practical variable selection for generalized additive models. Computational Statistics & Data Analysis, 55(7), 2372–2387. https://doi.org/10.1016/j.csda.2011.02.004
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

    :param variables: A list of the variables (strings) of which the term is a function. Need to exist in ``data`` passed to ``Formula``. Need to be continuous.
    :type variables: list[str]
    :param by: A string corresponding to a factor in ``data`` passed to ``Formula``. Separate f(``variables``) (and smoothness penalties) will be estimated per level of ``by``.
    :type by: str, optional
    :param by_cont: A string corresponding to a numerical variable in ``data`` passed to ``Formula``. The model matrix for the estimated smooth term f(``variables``) will be multiplied by the column of this variable. Can be used to estimate 'varying coefficient' models but also to set up binary smooths or to only estimate a smooth term for specific levels of a factor (i.e., what is possible for ordered factors in R & mgcv).
    :type by_cont: str, optional
    :param binary: A list containing two strings. The first string corresponds to a factor in ``data`` passed to ``Formula``. A separate f(``variables``) will be estimated for the level of this factor corresponding to the second string.
    :type binary: [str,str], optional
    :param id: Only useful in combination with specifying a ``by`` variable. If ``id`` is set to any integer the penalties placed on the separate f(``variables``) will share a single smoothness penalty.
    :type id: int, optional
    :param nk: Number of basis functions to use. Even if ``identifiable`` is true, this number will reflect the final number of basis functions for this term (i.e., mssm acts like you would have asked for 10 basis functions if ``nk=9`` and identifiable=True; the default).
    :type nk: int or list[int], optional
    :param te: For tensor interaction terms only. If set to false, the term mimics the behavior of ``ti()`` in mgcv (Wood, 2017). Otherwise, the term behaves like a ``te()`` term in mgcv - i.e., the marginal basis functions are not removed from the interaction.
    :type te: bool, optional
    :param rp: Experimental - will currently break for tensor smooths or in case ``by`` is provided. Whether or not to re-parameterize the term - see :func:`mssm.src.python.formula.reparam` for details. Defaults to no re-parameterization.
    :type rp: int, optional
    :param constraint: What kind of identifiability constraints should be absorbed by the terms (if they are to be identifiable). Either QR-based constraints (default, well-behaved), by means of column-dropping (no infill, not so well-behaved), or by means of difference re-coding (little infill, not so well behaved either).
    :type constraint: mssm.src.constraints.ConstType, optional
    :param identifiable: Whether or not the constant should be removed from the space of functions this term can fit. Achieved by enforcing that :math:`\\mathbf{1}^T \\mathbf{X} = 0` (:math:`\\mathbf{X}` here is the spline matrix computed for the observed data; see Wood, 2017 for details). Necessary in most cases to keep the model identifiable.
    :type identifiable: bool, optional
    :param basis: The basis functions to use to construct the spline matrix. By default a B-spline basis (Eilers & Marx, 2010) implemented in :func:`mssm.src.smooths.B_spline_basis`.
    :type basis: Callable, optional
    :param basis_kwargs: A list containing one or multiple dictionaries specifying how the basis should be computed. Consult the docstring of the function computing the basis you want. For the default B-spline basis for example see the  :func:`mss.src.smooths.B_spline_basis` function. The default arguments set by any basis function, should work for most cases though.
    :type basis_kwargs: dict, optional
    :param is_penalized: Should the term be left unpenalized or not. There are rarely good reasons to set this to False.
    :type is_penalized: bool, optional
    :param penalize_null: Should a separate Null-space penalty (Marra & Wood, 2011) be placed on the term. By default, the term here will leave a linear f(`variables`) un-penalized! Thus, there is no option for the penalty to achieve f(`variables`) = 0 even if that would be supported by the data. Adding a Null-space penalty provides the penalty with that power. This can be used for model selection instead of Hypothesis testing and is the preferred way in ``mssm`` (see Marra & Wood, 2011 for details).
    :type penalize_null: bool, optional
    :param penalty: A list of penalty types to be placed on the term.
    :type penalty: list[Penalty], optional
    :param pen_kwargs: A list containing one or multiple dictionaries specifying how the penalty should be created. Consult the docstring of the :func:`Penalty.constructor` method of the specific :class:`Penalty` you want to use for details.
    :type pen_kwargs: list[dict], optional
    """

    def __init__(self,variables:list,
                by:str=None,
                by_cont:str=None,
                binary:tuple[str,str] | None = None,
                id:int=None,
                nk:int | list[int] = None,
                te: bool = False,
                rp:int = 0,
                constraint:ConstType=ConstType.QR,
                identifiable:bool=True,
                basis:Callable=smooths.B_spline_basis,
                basis_kwargs:dict={},
                is_penalized:bool = True,
                penalize_null:bool = False,
                penalty:list[Penalty] | None = None,
                pen_kwargs:list[dict] | None = None) -> None:
        
        if not binary is None and not by is None:
           raise ValueError("Binary smooths cannot also have a by-keyword.")
        
        if te==True and len(variables) == 1:
           raise ValueError("``te`` can only be set to true in case multiple variables are provided via ``variables``.")
        
        if rp != 0 and penalty is not None and (len(penalty) > len(variables)):
          raise ValueError("Re-parameterization only supports a single penalty (per maginal).")
        
        if nk is None:
          # Set mgcv-like defaults
          if len(variables) == 1:
            nk = 9
          else:
            if te:
              nk = 5
            else:
              nk = 4
        
        if not binary is None and identifiable:
           # Remove identifiability constraint for
           # binary difference smooths.
           identifiable = False
           # For te terms this can be skipped for these one coef is simply
           # subtracted after identifiability constraints have been observed,
           # while for all other terms mssm acts as if one additional coef was requested.
           if te is False:
            nk = nk + 1

        # Default penalty setup
        if penalty is None:
           penalty = [DifferencePenalty()]
           pen_kwargs = [{"m":2}]

        # For tensor product smooths we need to for every penalty in
        # penalty (and kwargs as well) repeat the penalty (and kwargs) for len(variables)
        if len(variables) > 1:
           tp_pens = []
           tp_pen_kwargs = []
           for pen,pen_kwarg in zip(penalty,pen_kwargs):
              for _ in range(len(variables)):
               tp_pens.append(copy.deepcopy(pen))
               tp_pen_kwargs.append(copy.deepcopy(pen_kwarg))
         
           penalty = tp_pens
           pen_kwargs = tp_pen_kwargs
        
        # Initialization: ToDo: the deepcopy can be dropped now.
        super().__init__(variables, TermType.SMOOTH, is_penalized, copy.deepcopy(penalty), copy.deepcopy(pen_kwargs))
        self.is_identifiable = identifiable
        self.Z = constraint
        self.should_rp = rp
        if rp:
          self.RP = [Reparameterization() for _ in range(len(variables))]
        else:
          self.RP = []
        self.basis = basis
        self.basis_kwargs = basis_kwargs
        self.by = by
        self.binary = binary
        self.binary_level = None
        self.id = id
        self.has_null_penalty = penalize_null
        self.by_cont = by_cont
        self.drop_coef = None

        # Tensor bases can each have different number of basis functions
        if len(variables) == 1 or isinstance(nk,list):
         self.nk = nk
        else:
         self.nk = [nk for _ in range(len(variables))]

        self.te = te

        # Term name
        self.name = f"f({variables}"
        if by is not None:
           self.name += f",by={by})"
        elif binary is not None:
           self.name += f",by={binary[1]})"
        else:
           self.name += ")"
        if by_cont is not None:
           self.name += f",by_c={by_cont})"

    def absorb_repara(self,rpidx,X,cov):
      """Computes all terms necessary to absorb a re-parameterization into the term and penalty matrix.

      :param rpidx: Index to specific reparam. obejct. There must be a 1 to 1 relationship between reparam. objects and the number of marginals required by this smooth (i.e., the number of variables).
      :type rpidx: int
      :param X: Design matrix associated with this term.
      :type X: scipy.sparse.csc_array
      :param cov: The covariate this term is a function of as a flattened numpy array.
      :type cov: np.ndarray
      :raises ValueError: If this method is called with ``rpidx`` exceeding the number of this term's RP objects (i.e., when ``rpidx > (len(self.RP) - 1)``) or if ``self.rp`` is equal to a value for which no reparameterisation is implemented.
      """

      if len(self.RP) == 0:
        warnings.warn("RP method called for term that should not be re-parameterized. Skipping RP attempt.")
        return
      
      if rpidx > (len(self.RP) - 1):
        raise ValueError(f"rpidx {rpidx} exceeds the {len(self.RP)} RP objects associated with term.")
      
      # Now safe to proceed with building RP object
      vars = self.variables

      if len(vars) > 1:
        id_k = self.nk[rpidx]
      else:
        id_k = self.nk

      if self.should_rp == 1:
        # Demmler & Reinsch (1975) re-parameterization was requested - need penalty for this.
        pen_kwargs = self.pen_kwargs[rpidx]
        penalty = self.penalty[rpidx]

        # Extract penalty generator
        pen_generator = penalty.constructor
      
        constraint = None
        # Pass one extra coef to penalty constructor if term is identifiable
        if self.is_identifiable:
          id_k += 1
          constraint = self.Z[rpidx]

        # Again get penalty elements used by this term.
        pen_data,pen_rows,pen_cols,_,_,_,rank = pen_generator(id_k,constraint,**pen_kwargs)

        # Make sure nk matches right dimension again
        if self.is_identifiable:
          id_k -= 1
        
        S_J = scp.sparse.csc_array((pen_data,(pen_rows,pen_cols)),shape=(id_k,id_k))

        # Now can compute the actual re-parameterization
        C, Srp, Drp, IRrp, rms1, rms2, _ = reparam(X,S_J,cov,QR=False,option=1,scale=True,identity=True)

        self.RP[rpidx].C = C
        self.RP[rpidx].Srp = Srp # Transformed penalty
        self.RP[rpidx].Drp = Drp # Transformed root of penalty
        self.RP[rpidx].IRrp = IRrp
        self.RP[rpidx].rms1 = rms1
        self.RP[rpidx].rms2 = rms2
        self.RP[rpidx].rank = rank

      else:
        raise ValueError(f"Requested a reparameterisation {self.should_rp} that is not supported")

    def build_penalty(self,ti:int,penalties:list[LambdaTerm],cur_pen_idx:int,penid:int,factor_levels:dict,col_S:int) -> tuple[list[LambdaTerm],int]:
      """Builds a penalty matrix associated with this smooth term and returns an updated ``penalties`` list including it.

      This method is implemented by most implementations of the :class:`GammTerm` class.
      Two arguments need to be returned: the updated ``penalties`` list including the new penalty implemented as a :class:`LambdaTerm` and the updated ``cur_pen_idx``.
      The latter simply needs to be incremented for every penalty added to ``penalties``.

      :param ti: Index corresponding to the position the current term (i.e., self) takes on in the list of terms of the Formula.
      :type ti: int
      :param penalties: List of previosly created penalties.
      :type penalties: [LambdaTerm]
      :param cur_pen_idx: Index of the last element in ``penalties``.
      :type cur_pen_idx: int
      :param penid: If a term is subjected to multipe penalties, then ``penid`` indexes which of those penalties is currently implemented. Otherwise can be set to zero.
      :type penid: int
      :param factor_levels: Factor levels dictionary. Keys are factor variables in the data, values are np.arrays holding the unique levels (as str) of the corresponding factor.
      :type factor_levels: dict
      :param n_coef: Number of coefficients associated with this term.
      :type n_coef: int
      :param col_S: Number of columns of the total penalty matrix.
      :type col_S: int
      :return: Updated ``penalties`` list including the new penalties implemented as a :class:`LambdaTerm` and the updated ``cur_pen_idx``
      :rtype: tuple[list[LambdaTerm],int]
      """
      # We again have to deal with potential identifiable constraints!
      # Then we again act as if n_k was n_k+1 for difference penalties

      # penid % len(vars) because it will just go from 0-(len(vars)-1) and
      # reset if penid >= len(vars) which might happen in case of multiple penalties on
      # every tp basis
      vars = self.variables

      if len(vars) > 1:
        id_k = self.nk[penid % len(vars)]
      else:
        id_k = self.nk

      # Extract penalty, penalty type, and constructor
      pen_kwargs = self.pen_kwargs[penid]
      penalty = self.penalty[penid]
      pen_generator = penalty.constructor
      pen = penalty.type
      
      # Determine penalty generator
      constraint = None

      # Pass one extra coef to penalty constructor if term is identifiable so that we end up with self.nk identifiable coef - delay for te terms
      if self.is_identifiable and self.te == False:
        id_k += 1
        constraint = self.Z[penid % len(vars)]

      # Again get penalty elements used by this term.
      pen_data,pen_rows,pen_cols,chol_data,chol_rows,chol_cols,rank = pen_generator(id_k,constraint,**pen_kwargs)

      # Make sure nk matches right dimension again
      if self.is_identifiable and self.te == False:
        id_k -= 1

      if self.should_rp == 1:
        # Demmler & Reinsch (1975) Re-parameterization was requested

        # Only supported for univariate term, so check
        if len(vars) > 1:
          raise ValueError(f"Demmler and Reinsch reparameterisation only supported for univariate smooth terms - but requested for term {ti}.")

        # Find correct transformation index
        if len(vars) > 1:
          rp_idx = penid
        else:
          rp_idx = 0
        
        # Extract transformed penalty + root
        Srp = self.RP[rp_idx].Srp
        Drp = self.RP[rp_idx].Drp

        # Update penalty and chol factor
        pen_data,pen_rows,pen_cols = translate_sparse(Srp)
        chol_data,chol_rows,chol_cols = translate_sparse(Drp)

        # Assign reparam type
        pen = PenType.REPARAM1

      # Create lambda term
      lTerm = LambdaTerm(start_index=cur_pen_idx,
                                   type = pen,
                                   term=ti)

      # For tensor product smooths we first have to recalculate:
      # pen_data,pen_rows,pen_cols,chol_data,chol_rows,chol_cols via TP_pen()
      # Then they can just be embedded via the calls below.

      if len(vars) > 1:
        # Absorb the identifiability constraint for te terms only after the tensor basis has been computed.
        if self.te and self.is_identifiable:
          constraint = self.Z[0] # Zero-index because a single set of identifiability constraints exists: one for the entire Tp basis.
        else:
          constraint = None
        
        pen_data,\
        pen_rows,\
        pen_cols,\
        chol_data,\
        chol_rows,\
        chol_cols = TP_pen(scp.sparse.csc_array((pen_data,(pen_rows,pen_cols)),shape=(id_k,id_k)),
                          scp.sparse.csc_array((chol_data,(chol_rows,chol_cols)),shape=(id_k,id_k)),
                          penid % len(vars),self.nk,constraint)
        
        # For te/ti terms, penalty dim are nk_1 * nk_2 * ... * nk_j over all j variables
        id_k = np.prod(self.nk)
        
        # For te terms we need to subtract one if term was made identifiable.
        if self.te and self.is_identifiable:
          id_k -= 1

      # Final penalty matrix
      lTerm.S_J = embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J,id_k)

      # Compute rank for TP penalty
      if len(vars) > 1:
        D = scp.linalg.eigh(lTerm.S_J.toarray(),eigvals_only=True)
        rank = len(D[D > max(D)*sys.float_info.epsilon**0.7])

      lTerm.rank = rank

      # Embed first penalty - if the term has a by-keyword more are added below.
      if self.drop_coef is None:
        lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,id_k,cur_pen_idx)
        lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,id_k,cur_pen_idx)
        
      else:
        # Adjust penalty for dropped coef
        pen_data_d,pen_rows_d,pen_cols_d,dropped = adjust_pen_drop(pen_data,pen_rows,pen_cols,self.drop_coef)

        lTerm.S_J = embed_in_Sj_sparse(pen_data_d,pen_rows_d,pen_cols_d,None,id_k-dropped)

        # Re-compute root & rank
        eig, U =scp.linalg.eigh(lTerm.S_J.toarray())
        rrank = sum([1 for e in eig if e >  np.power(sys.float_info.epsilon,0.7)])
        rD_J = scp.sparse.csc_array(U@np.diag([np.power(e,0.5) if e > np.power(sys.float_info.epsilon,0.7) else 0 for e in eig]))
        chol_data_d,chol_rows_d,chol_cols_d = translate_sparse(rD_J)

        # And embed
        lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data_d,chol_rows_d,chol_cols_d,lTerm.D_J_emb,col_S,id_k-dropped,cur_pen_idx)
        lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data_d,pen_rows_d,pen_cols_d,lTerm.S_J_emb,col_S,id_k-dropped,cur_pen_idx)

        # Adjust rank for dropped
        lTerm.rank = rrank
      
      if self.by is not None:
        by_levels = factor_levels[self.by]
          
        if self.id is not None:

          pen_iter = len(by_levels) - 1

          #for _ in range(pen_iter):
          #    lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,id_k,cur_pen_idx)
          #    lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,id_k,cur_pen_idx)
          if self.drop_coef is None:
            chol_rep = np.tile(chol_data,pen_iter)
            idx_row_rep = np.repeat(np.arange(pen_iter),len(chol_rows))*id_k
            idx_col_rep = np.repeat(np.arange(pen_iter),len(chol_cols))*id_k
            chol_rep_row = np.tile(chol_rows,pen_iter) + idx_row_rep
            chol_rep_cols = np.tile(chol_cols,pen_iter) + idx_col_rep
            
            lTerm.D_J_emb, _ = embed_in_S_sparse(chol_rep,chol_rep_row,chol_rep_cols,lTerm.D_J_emb,col_S,id_k*pen_iter,cur_pen_idx)

          pen_rep = np.tile(pen_data,pen_iter)
          idx_row_rep = np.repeat(np.arange(pen_iter),len(pen_rows))*id_k
          idx_col_rep = np.repeat(np.arange(pen_iter),len(pen_cols))*id_k
          pen_rep_row = np.tile(pen_rows,pen_iter) + idx_row_rep
          pren_rep_cols = np.tile(pen_cols,pen_iter) + idx_col_rep

          if self.drop_coef is None:
            lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_rep,pen_rep_row,pren_rep_cols,lTerm.S_J_emb,col_S,id_k*pen_iter,cur_pen_idx)
          else:
            # Adjust penalty for dropped coef
            pen_data_d,pen_rows_d,pen_cols_d,dropped = adjust_pen_drop(pen_rep,pen_rep_row,pren_rep_cols,self.drop_coef,offset=id_k)

            lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data_d,pen_rows_d,pen_cols_d,lTerm.S_J_emb,col_S,(id_k*pen_iter)-dropped,cur_pen_idx)

            lTerm.S_J = lTerm.S_J_emb[lTerm.start_index:cur_pen_idx,lTerm.start_index:cur_pen_idx]

            ########### Re-compute root & rank blockwise ###########
            rrank = 0
            lTerm.D_J_emb = None
            by_cols = np.arange(id_k*len(by_levels))
            tmp_pen_idx_by = lTerm.start_index
            lvl_drop = np.array(self.drop_coef)
            for by_lvl in range(len(by_levels)):
              
              # First pick set of original columns associated with current level.
              # Then Figure out whether a column needs to be dropped
              lvl_cols = by_cols[by_lvl*id_k:(by_lvl+1)*id_k]
              keep_col = ~np.isin(lvl_cols,lvl_drop)

              # Now re-align columns, based on **all coefficients dropped** (i.e., across levels)
              # To ensure that cols/rows dropped in lTerm.S_J from a previous level are also correctly removed/
              # accounted for in the alignment.
              lvl_cols_realign = np.zeros_like(lvl_cols)
              lvl_cols_realign[:] = lvl_cols
              for d in lvl_drop:
                lvl_cols_realign[lvl_cols > d] -= 1
              
              # Now drop and select block of lTerm.S_J associated with current by-level
              lvl_cols_realign = lvl_cols_realign[keep_col]
              lvl_SJ = lTerm.S_J[lvl_cols_realign,:]
              lvl_SJ = lvl_SJ[:,lvl_cols_realign]

              # Now compute root & rank for this block and adjust overall rank accordingly.
              eig, U =scp.linalg.eigh(lvl_SJ.toarray())
              rrankJ = sum([1 for e in eig if e >  np.power(sys.float_info.epsilon,0.7)])
              
              rD_J = scp.sparse.csc_array(U@np.diag([np.power(e,0.5) if e > np.power(sys.float_info.epsilon,0.7) else 0 for e in eig]))
              chol_data_dJ,chol_rows_dJ,chol_cols_dJ = translate_sparse(rD_J)

              lTerm.D_J_emb, tmp_pen_idx_by = embed_in_S_sparse(chol_data_dJ,chol_rows_dJ,chol_cols_dJ,lTerm.D_J_emb,col_S,lvl_SJ.shape[1],tmp_pen_idx_by)
              rrank += rrankJ

          # For pinv calculation during model fitting.
          if self.drop_coef is None:
            lTerm.rep_sj = pen_iter + 1
            lTerm.rank = rank * (pen_iter + 1)
          else:
            lTerm.rep_sj = 1
            lTerm.rank = rrank

          penalties.append(lTerm)

        else:
          # In case all levels get their own smoothing penalty - append first lterm then create new ones for
          # remaining levels.
          penalties.append(lTerm)

          pen_iter = len(by_levels) - 1

          for by_lvl in range(pen_iter):

              # Create lambda term
              lTerm = LambdaTerm(start_index=cur_pen_idx,
                                           type = pen,
                                           term=ti)

              # Embed penalties
              lTerm.rank = rank

              if self.drop_coef is None:
                lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,id_k,cur_pen_idx)
                lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,id_k,cur_pen_idx)
                lTerm.S_J = embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J,id_k)
              else:
                # Adjust penalty for dropped coef
                pen_data_d,pen_rows_d,pen_cols_d,dropped = adjust_pen_drop(pen_data,pen_rows,pen_cols,self.drop_coef,offset=(by_lvl+1)*id_k)

                lTerm.S_J = embed_in_Sj_sparse(pen_data_d,pen_rows_d,pen_cols_d,lTerm.S_J,id_k-dropped)

                # Re-compute root & rank
                eig, U =scp.linalg.eigh(lTerm.S_J.toarray())
                rrank = sum([1 for e in eig if e >  np.power(sys.float_info.epsilon,0.7)])
                rD_J = scp.sparse.csc_array(U@np.diag([np.power(e,0.5) if e > np.power(sys.float_info.epsilon,0.7) else 0 for e in eig]))
                chol_data_d,chol_rows_d,chol_cols_d = translate_sparse(rD_J)
                
                # And embed
                lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data_d,chol_rows_d,chol_cols_d,lTerm.D_J_emb,col_S,id_k-dropped,cur_pen_idx)
                lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data_d,pen_rows_d,pen_cols_d,lTerm.S_J_emb,col_S,id_k-dropped,cur_pen_idx)

                # Adjust rank for dropped
                lTerm.rank = rrank
              
              penalties.append(lTerm)

      else:
          penalties.append(lTerm)

      return penalties,cur_pen_idx
    
    def build_matrix(self,ci:int,ti:int,var_map:dict,var_mins:dict,var_maxs:dict,factor_levels:dict,ridx:list[int],cov_flat:np.ndarray,use_only:list[int],tol:int=0) -> tuple[list[float],list[int],list[int],int]:
      """Builds the design/term/model matrix for this smooth term.

      References:
        - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param ci: Current column index.
      :type ci: int
      :param ti: Index corresponding to the position the current term (i.e., self) takes on in the list of terms of the Formula.
      :type ti: int
      :param var_map: Var map dictionary. Keys are variables in the data, values their column index in the encoded predictor matrix.
      :type var_map: dict
      :param var_mins: Var mins dictionary. Keys are variables in the data, values are either the minimum value the variable takes on for continuous variables or ``None`` for categorical variables.
      :type var_mins: dict
      :param var_maxs: Var maxs dictionary. Keys are variables in the data, values are either the maximum value the variable takes on in for continuous variables or ``None`` for categorical variables.
      :type var_maxs: dict
      :param factor_levels: Factor levels dictionary. Keys are factor variables in the data, values are np.arrays holding the unique levels (as str) of the corresponding factor.
      :type factor_levels: dict
      :param ridx: Array of non NAN rows in the data.
      :type ridx: np.ndarray
      :param cov_flat: An array, containing all (encoded, in case of categorical predictors) values on each predictor (each columns of ``cov_flat`` corresponds to a different predictor) variable included in any of the terms in order of the data-frame passed to the Formula.
      :type cov_flat: np.ndarray
      :param use_only: A list holding term indices for which the matrix should be formed. For terms not included in this list a zero matrix will be returned. Can be set to ``None`` so that no terms are excluded.
      :type use_only: [int]
      :param tol: A tolerance that can be used to prune the term matrix from values close to zero rather than absolutely zero. Defaults to strictly zero.
      :type tol: int, optional
      :return: matrix data, matrix row indices, matrix column indices, added columns
      :rtype: tuple[list[float],list[int],list[int],int]
      """
      vars = self.variables
      term_ridx = []

      new_elements = []
      new_rows = []
      new_cols = []
      new_ci = 0

      # Calculate Coef number for control checks
      if len(vars) > 1:
        n_coef = np.prod(self.nk)
        if self.te and self.is_identifiable:
          n_coef -= 1
      else:
        n_coef = self.nk
      #print(n_coef)

      if self.by is not None:
        by_levels = factor_levels[self.by]
        n_coef *= len(by_levels)

      # Adjust for dropped coef.
      if self.drop_coef is not None:
        n_coef -= len(self.drop_coef)
          
      # Calculate smooth term for corresponding covariate

      # Handle identifiability constraints for every basis and
      # optionally update tensor surface.
      for vi in range(len(vars)):

        if len(vars) > 1:
          id_nk = self.nk[vi]
        else:
          id_nk = self.nk

        if self.is_identifiable and self.te == False:
          id_nk += 1

        #print(var_mins[vars[0]],var_maxs[vars[0]])
        matrix_term_v = self.basis(cov_flat[:,var_map[vars[vi]]],
                                    None, id_nk, min_c=var_mins[vars[vi]],
                                    max_c=var_maxs[vars[vi]], **self.basis_kwargs)

        if self.is_identifiable and self.te == False:
          if self.Z[vi].type == ConstType.QR:
              matrix_term_v = matrix_term_v @ self.Z[vi].Z

          elif self.Z[vi].type == ConstType.DROP:
              matrix_term_v = np.delete(matrix_term_v,self.Z[vi].Z,axis=1)

          elif self.Z[vi].type == ConstType.DIFF:
              # Applies difference re-coding for sum-to-zero coefficients.
              # Based on smoothCon in mgcv(2017). See constraints.py
              # for more details.
              matrix_term_v = np.diff(np.concatenate((matrix_term_v[:,self.Z[vi].Z:matrix_term_v.shape[1]],matrix_term_v[:,:self.Z[vi].Z]),axis=1))
              matrix_term_v = np.concatenate((matrix_term_v[:,matrix_term_v.shape[1]-self.Z[vi].Z:],matrix_term_v[:,:matrix_term_v.shape[1]-self.Z[vi].Z]),axis=1)
        
        if self.should_rp > 0:
          # Reparameterization of marginals was requested - at this point it can be easily evaluated.
          matrix_term_v = matrix_term_v @ self.RP[vi].C

        if vi == 0:
          matrix_term = matrix_term_v
        else:
          matrix_term = TP_basis_calc(matrix_term,matrix_term_v)
      
      if self.is_identifiable and self.te:
          if self.Z[0].type == ConstType.QR:
            matrix_term = matrix_term @ self.Z[0].Z

          elif self.Z[0].type == ConstType.DROP:
            matrix_term = np.delete(matrix_term,self.Z[0].Z,axis=1)

          elif self.Z[0].type == ConstType.DIFF:
            matrix_term = np.diff(np.concatenate((matrix_term[:,self.Z[0].Z:matrix_term.shape[1]],matrix_term[:,:self.Z[0].Z]),axis=1))
            matrix_term = np.concatenate((matrix_term[:,matrix_term.shape[1]-self.Z[0].Z:],matrix_term[:,:matrix_term.shape[1]-self.Z[0].Z]),axis=1)

      m_rows, m_cols = matrix_term.shape
      #print(m_cols)

      # Multiply each row of model matrix by value in by_cont
      if self.by_cont is not None:
        by_cont_cov = cov_flat[:,var_map[self.by_cont]]
        matrix_term *= by_cont_cov.reshape(-1,1)
      
      # Handle optional by keyword
      if self.by is not None:
        term_ridx = []

        by_cov = cov_flat[:,var_map[self.by]]
        
        # Split by cov and update rows with elements in columns
        m_coli_by = 0
        for by_level in range(len(by_levels)):
          by_cidx = by_cov == by_level
          for m_coli in range(m_cols):
              if self.drop_coef is None or m_coli_by not in self.drop_coef:
                term_ridx.append(ridx[by_cidx,])
              m_coli_by += 1
      
      # Handle optional binary keyword
      elif self.binary is not None:

        by_cov = cov_flat[:,var_map[self.binary[0]]]
        by_cidx = by_cov == self.binary_level

        if self.drop_coef is None:
          term_ridx = [ridx[by_cidx,] for _ in range(m_cols)]
        else:
          term_ridx = [ridx[by_cidx,] for m_coli in range(m_cols) if m_coli not in self.drop_coef]

      # No by or binary just use rows/cols as they are
      else:
        if self.drop_coef is None:
          term_ridx = [ridx[:] for _ in range(m_cols)]
        else:
          term_ridx = [ridx[:] for m_coli in range(m_cols) if m_coli not in self.drop_coef]

      f_cols = len(term_ridx)

      if n_coef != f_cols:
        raise KeyError("Not all model matrix columns were created.")

      # Find basis elements > 0 and collect correspondings elements and row indices
      for m_coli in range(f_cols):
        final_ridx = term_ridx[m_coli]
        final_col = matrix_term[final_ridx,m_coli%m_cols]

        # Tolerance row index for this columns
        cidx = abs(final_col) > tol
        if use_only is None or ti in use_only:
          new_elements.extend(final_col[cidx])
          new_rows.extend(final_ridx[cidx])
          new_cols.extend([ci for _ in range(len(final_ridx[cidx]))])
        new_ci += 1
        ci += 1
        term_ridx[m_coli] = None
      
      return new_elements,new_rows,new_cols,new_ci
  
    def get_coef_info(self, factor_levels:dict) -> tuple[int,int,list[str]]:
      """Returns the total number of coefficients associated with this smooth term, the number of unpenalized coefficients associated with this smooth term, and a list with names for each of the coefficients associated with this smooth term.

      :param factor_levels: Factor levels dictionary. Keys are factor variables in the data, values are np.arrays holding the unique levels (as str) of the corresponding factor.
      :type factor_levels: dict
      :return: Number of coefficients associated with term, number of un-penalized coefficients associated with term, coef names
      :rtype: tuple[int,int,list[str]]
      """
      coef_names = []

      vars = self.variables
      # Calculate Coef names
      if len(vars) > 1:
          term_n_coef = np.prod(self.nk)
          if self.te and self.is_identifiable:
            # identifiable te() terms loose one coefficient since the identifiability constraint
            # is computed after the tensor product calculation. So this term behaves
            # different than all other terms in mssm, which is a bit annoying. But there is
            # no easy solution - we could add 1 coefficient to the marginal basis for one variable
            # but then we will always favor one direction.
            term_n_coef -= 1        
      else:
          term_n_coef = self.nk

      # Total coef accounting for potential by keywords.
      n_coef = term_n_coef

      # var label
      var_label = vars[0]
      if len(vars) > 1:
          var_label = "_".join(vars)
    
      if self.binary is not None:
          var_label += self.binary[0]

      if self.by is not None:
          by_levels = factor_levels[self.by]
          n_coef *= len(by_levels)

          for by_level in by_levels:
              coef_names.extend([f"f_{var_label}_{ink}_{by_level}" for ink in range(term_n_coef)])
          
      else:
          coef_names.extend([f"f_{var_label}_{ink}" for ink in range(term_n_coef)])
      
      # Adjust for drop
      if self.drop_coef is not None:
        n_coef -= len(self.drop_coef)
        n_idx = np.arange(len(coef_names))
        keep = ~np.isin(n_idx,self.drop_coef)
        coef_names = [coef_names[ni] for ni in n_idx[keep]]
      
      # Check if term is penalized - if not we need to return the correct number of unpenalized coefficients
      n_nopen = 0
      if not self.is_penalized:
        n_nopen = n_coef
      return n_coef,n_nopen,coef_names

class fs(f):
   """
    Essentially a :class:`f` term with ``by=rf``, ``id != None``, ``penalize_null= True``, ``pen_kwargs = [{"m":1}]``, and ``rp=1``.
    
    This term approximates the "factor-smooth interaction" basis "fs" with ``m= 1`` available in ``mgcv`` (Wood, 2017). For example,
    the term below from ``mgcv``:

    ::

      s(x,sub,bs="fs"))
   
    would approximately correspond to the following term in ``mssm``:

    ::

      fs(["x"],rf="sub")

    They are however not equivalent (mgcv by default uses a different basis for which the ``m`` key-word has a different functionality).
    
    Specifically, here ``m= 1`` implies that the only function left unpenalized by the default (difference) penalty is the constant (Eilers & Marx, 2010). Thus,
    a linear basis is penalized by the same default penalty that also penalizes smoothness (and not by a separate penalty as
    is the case in ``mgcv`` when ``m=1`` for the default basis)! Any constant basis is penalized by the null-space penalty (in both ``mgcv`` and ``mssm``;
    see Marra & Wood, 2011) - the term thus shrinks towards zero (Wood, 2017).

    The factor smooth basis in mgcv allows to let the penalty be different for different levels of an additional factor (by additionally specifying
    the ``by`` argument for a smooth with basis "fs"). I.e.,
   
    ::

      s(Time,Subject,by='condition',bs='fs')
   
    in ``mgcv`` would estimate a non-linear random smooth of "time" per level of the "subject" & "condition" interaction - with the same penalty being placed on all
    random smooth terms within the same "condition" level.
    
    This can be achieved in ``mssm`` by adding multiple :class:`fs` terms to the :class:`Formula` and utilising the ``by_subgroup`` argument. This needs to be set to a list
    where the first element identifies the additional factor variable (e.g., "condition") and the second element corresponds to a level of said factor variable. E.g., to approximate
    the aforementioned ``mgcv`` term we have to add:
     
    ::
    
      *[fs(["Time"],rf="subject_cond",by_subgroup=["cond",cl]) for cl in np.unique(dat["cond"])]
      
    to the :class:`Formula` ``terms`` list. Importantly, "subject_cond" is the interaction of "subject" and "condition" - not just the "subject variable in the data.

    Model estimation can become quite expensive for :class:`fs` terms, when the factor variable for ``rf`` has many levels. (> 10000) In that case, approximate derivative
    evaluation can speed things up considerably. To enforce this, the ``approx_deriv`` argument needs to be specified with a dict, having the following structure:
    ``{"no_disc":[str],"excl":[str],"split_by":[str],"restarts":int,"seed":None or int}``. "no_disc" should usually be set to an empty list, and should in general only contain names of
    continuous variables included in the formula. Any variable mentioned here will not be discretized before clustering - this will make the approximation a bit more
    accurate but also require more time. Similarly, "excl" specifies any continuous variables that should be excluded for clustering. "split_by" should generally be set
    to a list containing all categorical variables present in the formula. "restarts" indicates the number of times to re-produce the clustering (40 seems to be a good number). "seed" can either be
    set to None or to an integer - in the latter case, the random cluster initialization will use that seed, ensuring that the clustering outcome (and hence model fit) is replicable.

    References:

     - Eilers, P., & Marx, B. (2010). Splines, knots, and penalties. https://doi.org/10.1002/WICS.125
     - Marra, G., & Wood, S. N. (2011). Practical variable selection for generalized additive models.Computational Statistics & Data Analysis, 55(7), 2372–2387. https://doi.org/10.1016/j.csda.2011.02.004
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.). Chapman and Hall/CRC.

    :param variables: A list of the variables (strings) of which the term is a function. Need to exist in ``data`` passed to ``Formula``. Need to be continuous.
    :type variables: list[str]
    :param rf: A string corresponding to a (random) factor in ``data`` passed to ``Formula``. Separate f(``variables``) (but a shared smoothness penalty!) will be estimated per level of ``rf``.
    :type rf: str, optional
    :param nk: Number of basis functions -1 to use. I.e., if ``nk=9`` (the default), the term will use 10 basis functions. By default ``f()`` has identifiability constraints applied and we act as if ``nk``+ 1 coefficients were requested. The ``fs()`` term needs no identifiability constrains so if the same number of coefficients used for a ``f()`` term is requested (the desired approach), one coefficient is added to compensate for the lack of identifiability constraints. This is the opposite to how this is handled in mgcv: specifying ``nk=10`` for "fixed" univariate smooths results in 9 basis functions being available. However, for a smooth in mgcv with basis='fs', 10 basis functions will remain available.
    :type nk: int or list[int], optional
    :param basis: The basis functions to use to construct the spline matrix. By default a B-spline basis (Eilers & Marx, 2010) implemented in :func:`mssm.src.smooths.B_spline_basis`.
    :type basis: Callable, optional
    :param basis_kwargs: A list containing one or multiple dictionaries specifying how the basis should be computed. For the B-spline basis the following arguments (with default values) are available: ``convolve``=``False``, ``min_c``=``None``, ``max_c``=``None``, ``deg``=``3``. See :func:`mssm.src.smooths.B_spline_basis` for details.
    :type basis_kwargs: dict, optional
    :param by_cont: A string corresponding to a numerical variable in ``data`` passed to ``Formula``. The model matrix for the estimated smooth term will be multiplied by the column of this variable. Can be used as an alternative to estimate separate random smooth terms per level of another factor (wich is also possible with `by_subgroup`).
    :type by_cont: str, optional
    :param by_subgroup: List including a factor variable and specific level of said variable. Allows for separate penalties as described above.
    :type by_subgroup: [str,str], optional
    :param approx_deriv: Dict holding important info for the clustering algorithm. Structure: ``{"no_disc":[str],"excl":[str],"split_by":[str],"restarts":int}``
    :type approx_deriv: dict, optional
    """
   
   def __init__(self,
                variables: list,
                rf: str = None,
                nk: int = 9,
                m: int = 1,
                rp:int = 1,
                by_cont:str|None = None,
                by_subgroup:tuple[str,str] | None = None,
                approx_deriv:dict | None = None,
                basis: Callable = smooths.B_spline_basis,
                basis_kwargs: dict = {}):

      penalty = [DifferencePenalty()]
      pen_kwargs = [{"m":m}]
      super().__init__(variables, rf, by_cont, None, 99, nk+1, False, rp, ConstType.QR, False,
                       basis, basis_kwargs,
                       True, True, penalty, pen_kwargs)
      
      self.approx_deriv=approx_deriv
      self.by_subgroup = by_subgroup

      if not self.by_subgroup is None:

         self.name +=  ": " + self.by_subgroup[1]
    
   def build_penalty(self,ti:int,penalties:list[LambdaTerm],cur_pen_idx:int,penid:int,factor_levels:dict,col_S:int) -> tuple[list[LambdaTerm],int]:
      """Builds a penalty matrix associated with this factor smooth term and returns an updated ``penalties`` list including it.

      This method is implemented by most implementations of the :class:`GammTerm` class.
      Two arguments need to be returned: the updated ``penalties`` list including the new penalty implemented as a :class:`LambdaTerm` and the updated ``cur_pen_idx``.
      The latter simply needs to be incremented for every penalty added to ``penalties``.

      :param ti: Index corresponding to the position the current term (i.e., self) takes on in the list of terms of the Formula.
      :type ti: int
      :param penalties: List of previosly created penalties.
      :type penalties: [LambdaTerm]
      :param cur_pen_idx: Index of the last element in ``penalties``.
      :type cur_pen_idx: int
      :param penid: If a term is subjected to multipe penalties, then ``penid`` indexes which of those penalties is currently implemented. Otherwise can be set to zero.
      :type penid: int
      :param factor_levels: Factor levels dictionary. Keys are factor variables in the data, values are np.arrays holding the unique levels (as str) of the corresponding factor.
      :type factor_levels: dict
      :param col_S: Number of columns of the total penalty matrix.
      :type col_S: int
      :return: Updated ``penalties`` list including the new penalties implemented as a :class:`LambdaTerm` and the updated ``cur_pen_idx``
      :rtype: tuple[list[LambdaTerm],int]
      """
      return super().build_penalty(ti, penalties, cur_pen_idx, penid, factor_levels, col_S)
   
   def build_matrix(self,ci:int,ti:int,var_map:dict,var_mins:dict,var_maxs:dict,factor_levels:dict,ridx:np.ndarray,cov_flat:np.ndarray,use_only:list[int],tol:int=0) -> tuple[list[float],list[int],list[int],int]:
      """Builds the design/term/model matrix for this factor smooth term.

      References:
        - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param ci: Current column index.
      :type ci: int
      :param ti: Index corresponding to the position the current term (i.e., self) takes on in the list of terms of the Formula.
      :type ti: int
      :param var_map: Var map dictionary. Keys are variables in the data, values their column index in the encoded predictor matrix.
      :type var_map: dict
      :param var_mins: Var mins dictionary. Keys are variables in the data, values are either the minimum value the variable takes on for continuous variables or ``None`` for categorical variables.
      :type var_mins: dict
      :param var_maxs: Var maxs dictionary. Keys are variables in the data, values are either the maximum value the variable takes on in for continuous variables or ``None`` for categorical variables.
      :type var_maxs: dict
      :param factor_levels: Factor levels dictionary. Keys are factor variables in the data, values are np.arrays holding the unique levels (as str) of the corresponding factor.
      :type factor_levels: dict
      :param ridx: Array of non NAN rows in the data.
      :type ridx: np.ndarray
      :param cov_flat: An array, containing all (encoded, in case of categorical predictors) values on each predictor (each columns of ``cov_flat`` corresponds to a different predictor) variable included in any of the terms in order of the data-frame passed to the Formula.
      :type cov_flat: np.ndarray
      :param use_only: A list holding term indices for which the matrix should be formed. For terms not included in this list a zero matrix will be returned. Can be set to ``None`` so that no terms are excluded.
      :type use_only: [int]
      :param tol: A tolerance that can be used to prune the term matrix from values close to zero rather than absolutely zero. Defaults to strictly zero.
      :type tol: int, optional
      :return: matrix data, matrix row indices, matrix column indices, added columns
      :rtype: tuple[list[float],list[int],list[int],int]
      """
      return super().build_matrix(ci, ti, var_map, var_mins, var_maxs, factor_levels, ridx, cov_flat, use_only, tol)

   def get_coef_info(self, factor_levels:dict) -> tuple[int,int,list[str]]:
      """Returns the total number of coefficients associated with this factor smooth term, the number of unpenalized coefficients associated with this factor smooth term, and a list with names for each of the coefficients associated with this factor smooth term.

      :param factor_levels: Factor levels dictionary. Keys are factor variables in the data, values are np.arrays holding the unique levels (as str) of the corresponding factor.
      :type factor_levels: dict
      :return: Number of coefficients associated with term, number of un-penalized coefficients associated with term, coef names
      :rtype: tuple[int,int,list[str]]
      """
      return super().get_coef_info(factor_levels)
        
class irf(GammTerm):
    """A simple impulse response term, designed to correct for events with overlapping responses in multi-level time-series modeling.

       The idea (see Ehinger & Dimigen; 2019 for a detailed introduction to this kind of deconvolution analysis) is that some kind of event happens during each recorded time-series
       (e.g., stimulus onset, distractor display, mask onset, etc.) which is assumed to affect the recorded signal in the next X ms in some way. The moment of event onset can
       differ between recorded time-series. In other words, the event is believed to act like an impulse which triggers a delayed response on the signal. This term class can be
       used to estimate the shape of this impulse response. Multiple ``irf`` terms can be included in a ``Formula`` if multiple events happen, potentially with overlapping responses.

       Example::

         # Simulate time-series based on two events that elicit responses which vary in their overlap.
         # The summed responses + a random intercept + noise is then the signal.
         overlap_dat,onsets1,onsets2 = sim7(100,1,2,seed=20)

         # Model below tries to recover the shape of the two responses in the 200 ms after event onset (max_c=200) + the random intercepts:
         overlap_formula = Formula(lhs("y"),[irf(["time"],onsets1,nk=15,basis_kwargs=[{"max_c":200,"min_c":0,"convolve":True}]),
                                             irf(["time"],onsets2,nk=15,basis_kwargs=[{"max_c":200,"min_c":0,"convolve":True}]),
                                             ri("factor")],
                                             data=overlap_dat,
                                             series_id="series") # For models with irf terms, the column in the data identifying unique series need to be specified.

         model = GAMM(overlap_formula,Gaussian())
         model.fit()

       Note, that care needs to be taken when predicting for models including ``irf`` terms, because the onset of events can differ between time-series. Hence, model
       predictions + standard errors should first be obtained for the entire data-set used also to train the model and then extract series-specific predictions from the
       model-matrix as follows::

         # Get model matrix for entire data-set but only based on the estimated shape for first irf term:
         _,pred_mat,ci_b = model.predict([0],overlap_dat,ci=True)

         # Now extract the prediction + approximate ci boundaries for a single series:
         s = 8
         s_pred = pred_mat[overlap_dat["series"] == s,:]@model.coef
         s_ci = ci_b[overlap_dat["series"] == s]

         # Now the estimated response following the onset of the first event can be visualized + an approximate CI:
         from matplotlib import pyplot as plt
         plt.plot(overlap_dat["time"][overlap_dat["series"] == s],s_pred,color='blue')
         plt.plot(overlap_dat["time"][overlap_dat["series"] == s],s_pred+s_ci,color='blue',linestyle='dashed')
         plt.plot(overlap_dat["time"][overlap_dat["series"] == s],s_pred-s_ci,color='blue',linestyle='dashed')

       References:

       - Ehinger, B. V., & Dimingen, O. (2019). Unfold: an integrated toolbox for overlap correction, non-linear modeling, and regression-based EEG analysis. https://doi.org/10.7717/peerj.7838

       :param variables: A list of the variables (strings) of which the term is a function. Need to exist in ``data`` passed to ``Formula``. Need to be continuous.
       :type variables: list[str]
       :param event_onset: A ``np.array`` containing, for each individual time-series, the index corresponding to the sample/time-point at which the event eliciting the response to be estimate by this term happened.
       :type event_onset: [int]
       :param basis_kwargs: A list containing one or multiple dictionaries specifying how the basis should be computed. For ``irf`` terms, the ``convolve`` argument has to be set to True! Also, ``min_c`` and ``max_c`` must be specified. ``min_c`` corresponds to the assumed min. delay of the response after event onset and can usually be set to 0. ``max_c`` corresponds to the assumed max. delay of the response (in ms) after which the response is believed to have returned to a zero base-line.
       :type basis_kwargs: dict
       :param by: A string corresponding to a factor in ``data`` passed to ``Formula``. Separate irf(``variables``) (and smoothness penalties) will be estimated per level of ``by``.
       :type by: str, optional
       :param id: Only useful in combination with specifying a ``by`` variable. If ``id`` is set to any integer the penalties placed on the separate irff(``variables``) will share a single smoothness penalty.
       :type id: int, optional
       :param nk: Number of basis functions to use. I.e., if ``nk=10`` (the default), the term will use 10 basis functions (Note that these terms are not made identifiable by absorbing any kind of constraint). 
       :type nk: int, optional
       :param basis: The basis functions to use to construct the spline matrix. By default a B-spline basis (Eilers & Marx, 2010) implemented in ``src.smooths.B_spline_basis``.
       :type basis: Callable, optional
       :param is_penalized: Should the term be left unpenalized or not. There are rarely good reasons to set this to False.
       :type is_penalized: bool, optional
       :param penalty: A list of penalty types to be placed on the term.
       :type penalty: list[Penalty], optional
       :param pen_kwargs: A list containing one or multiple dictionaries specifying how the penalty should be created. For the default difference penalty (Eilers & Marx, 2010) the only keyword argument (with default value) available is: ``m=2``. This reflects the order of the difference penalty. Note, that while a higher ``m`` permits penalizing towards smoother functions it also leads to an increased dimensionality of the penalty Kernel (the set of f[``variables``] which will not be penalized). In other words, increasingly more complex functions will be left un-penalized for higher ``m`` (except if ``penalize_null`` is set to True). ``m=2`` is usually a good choice and thus the default but see Eilers & Marx (2010) for details.
       :type pen_kwargs: list[dict], optional      
       """
    
    def __init__(self,variables:list[str],
                event_onset:list[int],
                basis_kwargs:list[dict],
                by:str=None,
                id:int=None,
                nk:int=10,
                basis:Callable=smooths.B_spline_basis,
                is_penalized:bool = True,
                penalty:list[Penalty] | None = None,
                pen_kwargs:list[dict] | None = None) -> None:
        
        # Default penalty setup
        if penalty is None:
           penalty = [DifferencePenalty()]
           pen_kwargs = [{"m":2}]

        # For impulse response tensor product smooths we need to for every penalty in
        # penalty (and kwargs as well) repeat the penalty (and kwargs) for len(variables)
        if len(variables) > 1:
           tp_pens = []
           tp_pen_kwargs = []
           for pen,pen_kwarg in zip(penalty,pen_kwargs):
              for _ in range(len(variables)):
               tp_pens.append(copy.deepcopy(pen))
               tp_pen_kwargs.append(copy.deepcopy(pen_kwarg))
         
           penalty = tp_pens
           pen_kwargs = tp_pen_kwargs
        
        # Initialization: ToDo: the deepcopy can be dropped now.
        super().__init__(variables, TermType.IRSMOOTH, is_penalized, copy.deepcopy(penalty), copy.deepcopy(pen_kwargs))
        self.basis = basis
        self.basis_kwargs = basis_kwargs
        self.event_onset = event_onset
        self.by = by
        self.id = id
        self.by_cont = None

        # nk can also be a list for irf smooths.
        if len(variables) == 1 or isinstance(nk,list):
         self.nk = nk
        else:
         self.nk = [nk for _ in range(len(variables))]

        # Term name
        self.name = f"f({'_'.join(variables)}"
        if by is not None:
           self.name += f",by={by})"
  
    def build_penalty(self,ti:int,penalties:list[LambdaTerm],cur_pen_idx:int,penid:int,factor_levels:dict,col_S:int) -> tuple[list[LambdaTerm],int]:
      """Builds a penalty matrix associated with this impulse response smooth term and returns an updated ``penalties`` list including it.

      This method is implemented by most implementations of the :class:`GammTerm` class.
      Two arguments need to be returned: the updated ``penalties`` list including the new penalty implemented as a :class:`LambdaTerm` and the updated ``cur_pen_idx``.
      The latter simply needs to be incremented for every penalty added to ``penalties``.

      :param ti: Index corresponding to the position the current term (i.e., self) takes on in the list of terms of the Formula.
      :type ti: int
      :param penalties: List of previosly created penalties.
      :type penalties: [LambdaTerm]
      :param cur_pen_idx: Index of the last element in ``penalties``.
      :type cur_pen_idx: int
      :param penid: If a term is subjected to multipe penalties, then ``penid`` indexes which of those penalties is currently implemented. Otherwise can be set to zero.
      :type penid: int
      :param factor_levels: Factor levels dictionary. Keys are factor variables in the data, values are np.arrays holding the unique levels (as str) of the corresponding factor.
      :type factor_levels: dict
      :param col_S: Number of columns of the total penalty matrix.
      :type col_S: int
      :return: Updated ``penalties`` list including the new penalties implemented as a :class:`LambdaTerm` and the updated ``cur_pen_idx``
      :rtype: tuple[list[LambdaTerm],int]
      """
      vars = self.variables

      if len(vars) > 1:
        id_k = self.nk[penid % len(vars)]
      else:
        id_k = self.nk

      # Extract penalty, penalty type, and constructor
      penalty = self.penalty[penid]
      pen_generator = penalty.constructor
      pen = penalty.type

      # Get non-zero elements and indices for the penalty used by this term.
      pen_data,pen_rows,pen_cols,chol_data,chol_rows,chol_cols,rank = pen_generator(id_k,None,**self.pen_kwargs[penid])

      # For tensor product smooths we first have to recalculate:
      # pen_data,pen_rows,pen_cols,chol_data,chol_rows,chol_cols via TP_pen()
      # Then they can just be embedded via the calls below.

      if len(vars) > 1:
        constraint = None
        
        pen_data,\
        pen_rows,\
        pen_cols,\
        chol_data,\
        chol_rows,\
        chol_cols = TP_pen(scp.sparse.csc_array((pen_data,(pen_rows,pen_cols)),shape=(id_k,id_k)),
                          scp.sparse.csc_array((chol_data,(chol_rows,chol_cols)),shape=(id_k,id_k)),
                          penid % len(vars),self.nk,constraint)
        
        # For te terms, penalty dim are nk_1 * nk_2 * ... * nk_j over all j variables
        id_k = np.prod(self.nk)

      # Create lambda term
      lTerm = LambdaTerm(start_index=cur_pen_idx,
                                   type = pen,
                                   term=ti)

      # Embed first penalty - if the term has a by-keyword more are added below.
      lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,id_k,cur_pen_idx)
      lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,id_k,cur_pen_idx)
      lTerm.S_J = embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J,id_k)

      # Compute rank for TP penalty
      if len(vars) > 1:
        D = scp.linalg.eigh(lTerm.S_J.toarray(),eigvals_only=True)
        rank = len(D[D > max(D)*sys.float_info.epsilon**0.7])
      lTerm.rank = rank
          
      if self.by is not None:
        by_levels = factor_levels[self.by]
        if self.id is not None:

          for _ in range(len(by_levels)-1):
              lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,id_k,cur_pen_idx)
              lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,id_k,cur_pen_idx)

          # For pinv calculation during model fitting.
          lTerm.rep_sj = len(by_levels)
          lTerm.rank = rank * len(by_levels)
          penalties.append(lTerm)
        else:
          # In case all levels get their own smoothing penalty - append first lterm then create new ones for
          # remaining levels.
          penalties.append(lTerm)

          for _ in range(len(by_levels)-1):
            # Create lambda term
            lTerm = LambdaTerm(start_index=cur_pen_idx,
                                         type = pen,
                                         term=ti)

            # Embed penalties
            lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,id_k,cur_pen_idx)
            lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,id_k,cur_pen_idx)
            lTerm.S_J = embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J,id_k)
            lTerm.rank = rank
            penalties.append(lTerm)
      else:
        penalties.append(lTerm)
      
      return penalties,cur_pen_idx
   
    def build_matrix(self,ci:int,ti:int,var_map:dict,var_mins:dict,var_maxs:dict,factor_levels:dict,ridx:np.ndarray,cov:list[np.ndarray],use_only:list[int],pool,tol:int=0) -> tuple[list[float],list[int],list[int],int]:
      """Builds the design/term/model matrix associated with this impulse response smooth term and returns it represented as a list of values, a list of row indices, and a list of column indices.

      This method is implemented by every implementation of the :class:`GammTerm` class.
      The returned lists can then be used to create a sparse matrix for this term. Also returns an updated ``ci`` column index, reflecting how many additional columns would be added
      to the total model matrix.

      :param ci: Current column index.
      :type ci: int
      :param ti: Index corresponding to the position the current term (i.e., self) takes on in the list of terms of the Formula.
      :type ti: int
      :param var_map: Var map dictionary. Keys are variables in the data, values their column index in the encoded predictor matrix.
      :type var_map: dict
      :param var_mins: Var mins dictionary. Keys are variables in the data, values are either the minimum value the variable takes on for continuous variables or ``None`` for categorical variables.
      :type var_mins: dict
      :param var_maxs: Var maxs dictionary. Keys are variables in the data, values are either the maximum value the variable takes on in for continuous variables or ``None`` for categorical variables.
      :type var_maxs: dict
      :param factor_levels: Factor levels dictionary. Keys are factor variables in the data, values are np.arrays holding the unique levels (as str) of the corresponding factor.
      :type factor_levels: dict
      :param ridx: Array of non NAN rows in the data.
      :type ridx: np.ndarray
      :param cov: A list containing a separate array per time-series included in the data and indicated to the formula. The array contains, for the particular time-seriers, all (encoded, in case of categorical predictors) values on each predictor (each columns of the array corresponds to a different predictor) variable included in any of the terms in order of the data-frame passed to the Formula.
      :type cov: [np.ndarray]
      :param use_only: A list holding term indices for which the matrix should be formed. For terms not included in this list a zero matrix will be returned. Can be set to ``None`` so that no terms are excluded.
      :type use_only: [int]
      :param pool: A multiprocessing pool for parallel matrix construction parts
      :type pool: Any
      :param tol: A tolerance that can be used to prune the term matrix from values close to zero but not absolutely zero. Defaults to strictly zero.
      :type tol: int, optional
      :return: matrix data, matrix row indices, matrix column indices, added columns
      :rtype: tuple[list[float],list[int],list[int],int]
      """
      vars = self.variables
      term_elements = []
      term_idx = []

      new_elements = []
      new_rows = []
      new_cols = []
      new_ci = 0

      # Calculate number of coefficients
      n_coef = self.nk

      if len(vars) > 1:
          n_coef = np.prod(self.nk)

      by_levels = None
      if self.by is not None:
        by_levels = factor_levels[self.by]
        n_coef *= len(by_levels)

      if pool is None:
        for s_cov,s_event in zip(cov,self.event_onset):
          
          final_term = build_ir_smooth_series(self,s_cov,s_event,var_map,var_mins,var_maxs,by_levels)

          m_rows,m_cols = final_term.shape

          # Find basis elements > 0
          if len(term_idx) < 1:
            for m_coli in range(m_cols):
              term_elements.append([])
              term_idx.append([])

          for m_coli in range(m_cols):
            final_col = final_term[:,m_coli]
            cidx = abs(final_col) > tol
            term_elements[m_coli].extend(final_col[cidx])
            term_idx[m_coli].extend(cidx)

        if n_coef != len(term_elements):
          raise KeyError("Not all model matrix columns were created.")
        
        # Now collect actual row indices
        for m_coli in range(len(term_elements)):

          if use_only is None or ti in use_only:
            new_elements.extend(term_elements[m_coli])
            new_rows.extend(ridx[term_idx[m_coli]])
            new_cols.extend([ci for _ in range(len(term_elements[m_coli]))])
          ci += 1
          new_ci += 1

      else:
          
        args = zip(repeat(self),cov,self.event_onset,repeat(var_map),repeat(var_mins),repeat(var_maxs),repeat(by_levels))
          
        final_terms = pool.starmap(build_ir_smooth_series,args)
        final_term = np.vstack(final_terms)
        m_rows,m_cols = final_term.shape

        for m_coli in range(m_cols):
          if use_only is None or ti in use_only:
            final_col = final_term[:,m_coli]
            cidx = abs(final_col) > tol
            new_elements.extend(final_col[cidx])
            new_rows.extend(ridx[cidx])
            new_cols.extend([ci for _ in range(len(ridx[cidx]))])
          ci += 1
          new_ci += 1
      
      return new_elements,new_rows,new_cols,new_ci

    def get_coef_info(self,ti:int,factor_levels:dict) -> tuple[int,int,list[str]]:
      """Returns the total number of coefficients associated with this impulse response smooth term, the number of unpenalized coefficients associated with this term, and a list with names for each of the coefficients associated with this term.

      :param ti: Index corresponding to the position the current term (i.e., self) takes on in the list of terms of the Formula.
      :type ti: int
      :param factor_levels: Factor levels dictionary. Keys are factor variables in the data, values are np.arrays holding the unique levels (as str) of the corresponding factor.
      :type factor_levels: dict
      :return: Number of coefficients associated with term, number of un-penalized coefficients associated with term, coef names
      :rtype: tuple[int,int,list[str]]
      """
      # Calculate Coef names for impulse response terms
      vars = self.variables
      n_coef = self.nk
      coef_names = []

      if len(vars) > 1:
        n_coef = np.prod(self.nk)

      # var label
      var_label = vars[0]
      if len(vars) > 1:
        var_label = "_".join(vars)

      if self.by is not None:
        by_levels = factor_levels[self.by]
        n_coef *= len(by_levels)

        for by_level in by_levels:
          coef_names.extend([f"irf_{ti}_{var_label}_{ink}_{by_level}" for ink in range(n_coef)])
      
      else:
        coef_names.extend([f"irf_{ti}_{var_label}_{ink}" for ink in range(n_coef)])
      
      # Check if term is penalized - if not we need to return the correct number of unpenalized coefficients
      n_nopen = 0
      if not self.is_penalized:
        n_nopen = n_coef

      return n_coef, n_nopen, coef_names

def build_ir_smooth_series(irsterm:irf,s_cov:np.ndarray,s_event:int,var_map:dict,var_mins:dict,var_maxs:dict,by_levels:np.ndarray|None) -> np.ndarray:
      """Function to build the impulse response martrix for a single time-series.

      :param irsterm: Impulse response smooth term
      :type irsterm: irf
      :param s_cov: covariate array associated with ``irsterm``
      :type s_cov: np.ndarray
      :param s_event: Onset of impulse response function
      :type s_event: int
      :param var_map: Var map dictionary. Keys are variables in the data, values their column index in the encoded predictor matrix.
      :type var_map: dict
      :param var_mins: Var mins dictionary. Keys are variables in the data, values are either the minimum value the variable takes on for continuous variables or ``None`` for categorical variables.
      :type var_mins: dict
      :param var_maxs: Var maxs dictionary. Keys are variables in the data, values are either the maximum value the variable takes on in for continuous variables or ``None`` for categorical variables.
      :type var_maxs: dict
      :param by_levels: Numpy array holding the levels of the factor associated with the ``irsterm`` term (via ``irsterm.by``) or None
      :type by_levels: np.ndarray | None
      :return: The term matrix associated with the particular event at ``s_event``
      :rtype: np.ndarray
      """
      vars = irsterm.variables
      for vi in range(len(vars)):

        if len(vars) > 1:
          id_nk = irsterm.nk[vi]
        else:
          id_nk = irsterm.nk

        # Create matrix for event corresponding to term.
        # ToDo: For Multivariate case, the matrix term needs to be build iteratively for
        # every level of the multivariate factor to make sure that the convolution operation
        # works as intended. The splitting can happen later via by.
        basis_kwargs_v = irsterm.basis_kwargs[vi]

        if "max_c" in basis_kwargs_v and "min_c" in basis_kwargs_v:
          matrix_term_v = irsterm.basis(s_cov[:,var_map[vars[vi]]],s_event, id_nk, **basis_kwargs_v)
        else:
          matrix_term_v = irsterm.basis(s_cov[:,var_map[vars[vi]]],s_event, id_nk,min_c=var_mins[vars[vi]],max_c=var_maxs[vars[vi]], **basis_kwargs_v)

        if vi == 0:
          matrix_term = matrix_term_v
        else:
          matrix_term = TP_basis_calc(matrix_term,matrix_term_v)
      
      
      m_rows,m_cols = matrix_term.shape

      # Handle optional by keyword
      if irsterm.by is not None:
          
        by_matrix_term = np.zeros((m_rows,m_cols*len(by_levels)),dtype=float)

        by_cov = s_cov[:,var_map[irsterm.by]]

        # ToDo: For MV case this check will be true.
        if len(np.unique(by_cov)) > 1:
          raise ValueError(f"By-variable {irsterm.by} has varying levels on series level. This should not be the case.")
        
        # Fill the by matrix blocks.
        cByIndex = 0
        for by_level in range(len(by_levels)):
          if by_level == by_cov[0]:
              by_matrix_term[:,cByIndex:cByIndex+m_cols] = matrix_term
          cByIndex += m_cols # Update column range associated with current level.
        
        final_term = by_matrix_term
      else:
        final_term = matrix_term
      
      return final_term

class ri(GammTerm):
    """
    Adds a random intercept for the factor ``variable`` to the model. The random intercepts :math:`b_i` are assumed
    to be i.i.d :math:`b_i \\sim N(0,\\sigma_b)` i.e., normally distributed around zero - the simplest random effect supported by ``mssm``.

    Thus, this term achieves exactly what is achieved in ``mgcv`` by adding the term::

      s(variable,bs="re")
    
    The ``variable`` needs to identify a factor-variable in the data (i.e., the .dtype of the variable has to be equal to 'O'). If you want to
    add more complex random effects to the model (e.g., random slopes for continuous variable "x" per level of factor ``variable``) use the :class:`rs` class.

    :param variable: The name (string) of a factor variable. For every level of this factor a random intercept will be estimated. The random intercepts are assumed to follow a normal distribution centered around zero.
    :type variable: str
    """
    def __init__(self,
                 variable:str) -> None:
        
        # Initialization
        super().__init__([variable], TermType.RANDINT, True, [IdentityPenalty(PenType.IDENTITY)], [{}])

        # Term name
        self.name = f"ri({variable})"

    def build_penalty(self,ti:int,penalties:list[LambdaTerm],cur_pen_idx:int,factor_levels:dict,col_S:int) -> tuple[list[LambdaTerm],int]:
      """Builds a penalty matrix associated with this random intercept term and returns an updated ``penalties`` list including it.

      This method is implemented by most implementations of the :class:`GammTerm` class.
      Two arguments need to be returned: the updated ``penalties`` list including the new penalty implemented as a :class:`LambdaTerm` and the updated ``cur_pen_idx``.
      The latter simply needs to be incremented for every penalty added to ``penalties``.

      :param ti: Index corresponding to the position the current term (i.e., self) takes on in the list of terms of the Formula.
      :type ti: int
      :param penalties: List of previosly created penalties.
      :type penalties: [LambdaTerm]
      :param cur_pen_idx: Index of the last element in ``penalties``.
      :type cur_pen_idx: int
      :param factor_levels: Factor levels dictionary. Keys are factor variables in the data, values are np.arrays holding the unique levels (as str) of the corresponding factor.
      :type factor_levels: dict
      :param col_S: Number of columns of the total penalty matrix.
      :type col_S: int
      :return: Updated ``penalties`` list including the new penalties implemented as a :class:`LambdaTerm` and the updated ``cur_pen_idx``
      :rtype: tuple[list[LambdaTerm],int]
      """
      vars = self.variables
      idk = len(factor_levels[vars[0]])

      pen_data,pen_rows,pen_cols,chol_data,chol_rows,chol_cols,rank = self.penalty[0].constructor(idk,None)

      lTerm = LambdaTerm(start_index=cur_pen_idx,
                                   type = PenType.IDENTITY,
                                   term = ti)
      
      lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,idk,cur_pen_idx)
      lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,idk,cur_pen_idx)
      lTerm.S_J = embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J,idk)
      lTerm.rank = rank
      penalties.append(lTerm)

      return penalties, cur_pen_idx
   
    def build_matrix(self,ci:int,ti:int,var_map:dict,factor_levels:dict,ridx:np.ndarray,cov_flat:np.ndarray,use_only:list[int]) -> tuple[list[float],list[int],list[int],int]:
      """Builds the design/term/model matrix associated with this random intercept term and returns it represented as a list of values, a list of row indices, and a list of column indices.

      This method is implemented by every implementation of the :class:`GammTerm` class.
      The returned lists can then be used to create a sparse matrix for this term. Also returns an updated ``ci`` column index, reflecting how many additional columns would be added
      to the total model matrix.

      :param ci: Current column index.
      :type ci: int
      :param ti: Index corresponding to the position the current term (i.e., self) takes on in the list of terms of the Formula.
      :type ti: int
      :param var_map: Var map dictionary. Keys are variables in the data, values their column index in the encoded predictor matrix.
      :type var_map: dict
      :param factor_levels: Factor levels dictionary. Keys are factor variables in the data, values are np.arrays holding the unique levels (as str) of the corresponding factor.
      :type factor_levels: dict
      :param ridx: Array of non NAN rows in the data.
      :type ridx: np.ndarray
      :param cov_flat: An array, containing all (encoded, in case of categorical predictors) values on each predictor (each columns of ``cov_flat`` corresponds to a different predictor) variable included in any of the terms in order of the data-frame passed to the Formula.
      :type cov_flat: np.ndarray
      :param use_only: A list holding term indices for which the matrix should be formed. For terms not included in this list a zero matrix will be returned. Can be set to ``None`` so that no terms are excluded.
      :type use_only: [int]
      :return: matrix data, matrix row indices, matrix column indices, added columns
      :rtype: tuple[list[float],list[int],list[int],int]
      """
      vars = self.variables
      n_y = len(ridx)
      offset = np.ones(n_y)
      by_cov = cov_flat[:,var_map[vars[0]]]

      new_elements = []
      new_rows = []
      new_cols = []
      new_ci = 0

      for fl in range(len(factor_levels[vars[0]])):
        fl_idx = by_cov == fl
        if use_only is None or ti in use_only:
          new_elements.extend(offset[fl_idx])
          new_rows.extend(ridx[fl_idx])
          new_cols.extend([ci for _ in range(len(offset[fl_idx]))])
        new_ci += 1
        ci += 1

      return new_elements,new_rows,new_cols,new_ci

    def get_coef_info(self,factor_levels:dict,coding_factors:dict) -> tuple[int,int,list[str]]:
      """Returns the total number of coefficients associated with this random intercept term, the number of unpenalized coefficients associated with this term, and a list with names for each of the coefficients associated with this term.

      :param factor_levels: Factor levels dictionary. Keys are factor variables in the data, values are np.arrays holding the unique levels (as str) of the corresponding factor.
      :type factor_levels: dict
      :param coding_factors: Factor coding dictionary. Keys are factor variables in the data, values are dictionaries, where the keys correspond to the encoded levels (int) of the factor and the values to their levels (str).
      :type coding_factors: dict
      :return: Number of coefficients associated with term, number of un-penalized coefficients associated with term, coef names
      :rtype: tuple[int,int,list[str]]
      """
      vars = self.variables
      by_code_factors = coding_factors[vars[0]]
      n_coef = 0
      coef_names = []

      for fl in range(len(factor_levels[vars[0]])):
        coef_names.append(f"ri_{vars[0]}_{by_code_factors[fl]}")
        n_coef += 1

      return n_coef, 0, coef_names

class l(GammTerm):
    """
    Adds a parametric (linear) term to the model formula. The model :math:`\\mu_i = a + b*x_i` can for example be achieved
    by adding ``[i(), l(['x'])]`` to the ``term`` argument of a ``Formula``. The coefficient :math:`b` estimated for
    the term will then correspond to the slope of :math:`x`. This class can also be used to add predictors for
    categorical variables. If the formula includes an intercept, binary coding will be utilized to
    add reference-level adjustment coefficients for the remaining k-1 levels of any additional factor variable.

    If more than one variable is included in ``variables`` the model will only add the the len(``variables``)-interaction
    to the model! Lower order interactions and main effects will not be included by default (see :func:`li` function instead, which
    automatically includes all lower-order interactions and main effects).

    Example: The interaction effect of factor variable "cond", with two levels "1" and "2", and acontinuous variable "x"
    on the dependent variable "y" are of interest. To estimate such a model, the following formula can be used::

      formula = Formula(lhs("y"),terms=[i(),l(["cond"]),l(["x"]),l(["cond","x"])])
   
    This formula will estimate the following model:

    .. math::

      \\mu_i = a + b_1*c_i + b_2*x_i + b_3*c_i*x_i

    Here, :math:`c` is a binary predictor variable created so that it is 1 if "cond"=2 else 0 and :math:`b_3` is the coefficient that is added
    because ``l(["cond","x"])`` is included in the terms (i.e., the interaction effect).

    To get a model with only main effects for "cond" and "x", the following formula could be used::

      formula = Formula(lhs("y"),terms=[i(),l(["cond"]),l(["x"])])

    This formula will estimate:

    .. math::
    
      \\mu_i = a + b_1*c_i + b_2*x_i

    :param variables: A list of the variables (strings) for which linear predictors should be included
    :type variables: [str]
    """
    def __init__(self,
                 variables:list) -> None:
        
        # Initialization
        super().__init__(variables, TermType.LINEAR, False, [], [])

        # Term name
        self.name = f"l({variables})"
   
    def build_matrix(self,has_intercept:bool,ci:int,ti:int,var_map:dict,var_types:dict,factor_levels:dict,ridx:np.ndarray,cov_flat:np.ndarray,use_only:list[int]) -> tuple[list[float],list[int],list[int],int]:
      """Builds the design/term/model matrix associated with this linear term and returns it represented as a list of values, a list of row indices, and a list of column indices.

      This method is implemented by every implementation of the :class:`GammTerm` class.
      The returned lists can then be used to create a sparse matrix for this term. Also returns an updated ``ci`` column index, reflecting how many additional columns would be added
      to the total model matrix.

      :param has_intercept: Whether or not the formula of which this term is part includes an intercept term.
      :type has_intercept: bool
      :param ci: Current column index.
      :type ci: int
      :param ti: Index corresponding to the position the current term (i.e., self) takes on in the list of terms of the Formula.
      :type ti: int
      :param var_map: Var map dictionary. Keys are variables in the data, values their column index in the encoded predictor matrix.
      :type var_map: dict
      :param var_types: Var types dictionary. Keys are variables in the data, values are either ``VarType.NUMERIC`` for continuous variables or ``VarType.FACTOR`` for categorical variables.
      :type var_types: dict
      :param factor_levels: Factor levels dictionary. Keys are factor variables in the data, values are np.arrays holding the unique levels (as str) of the corresponding factor.
      :type factor_levels: dict
      :param ridx: Array of non NAN rows in the data.
      :type ridx: np.ndarray
      :param cov_flat: An array, containing all (encoded, in case of categorical predictors) values on each predictor (each columns of ``cov_flat`` corresponds to a different predictor) variable included in any of the terms in order of the data-frame passed to the Formula.
      :type cov_flat: np.ndarray
      :param use_only: A list holding term indices for which the matrix should be formed. For terms not included in this list a zero matrix will be returned. Can be set to ``None`` so that no terms are excluded.
      :type use_only: [int]
      :return: matrix data, matrix row indices, matrix column indices, added columns
      :rtype: tuple[list[float],list[int],list[int],int]
      """
      return build_linear_term(self,has_intercept,ci,ti,var_map,var_types,factor_levels,ridx,cov_flat,use_only)

    def get_coef_info(self,has_intercept:bool,var_types:dict,factor_levels:dict,coding_factors:dict) -> tuple[int,int,list[str]]:
      """Returns the total number of coefficients associated with this linear term, the number of unpenalized coefficients associated with this term, and a list with names for each of the coefficients associated with this term.

      :param has_intercept: Whether or not the formula of which this term is part includes an intercept term.
      :type has_intercept: bool
      :param var_types: Var types dictionary. Keys are variables in the data, values are either ``VarType.NUMERIC`` for continuous variables or ``VarType.FACTOR`` for categorical variables.
      :type var_types: dict
      :param factor_levels: Factor levels dictionary. Keys are factor variables in the data, values are np.arrays holding the unique levels (as str) of the corresponding factor.
      :type factor_levels: dict
      :param coding_factors: Factor coding dictionary. Keys are factor variables in the data, values are dictionaries, where the keys correspond to the encoded levels (int) of the factor and the values to their levels (str).
      :type coding_factors: dict
      :return: Number of coefficients associated with term, number of un-penalized coefficients associated with term, coef names
      :rtype: tuple[int,int,list[str]]
      """
      return get_linear_coef_info(self,has_intercept,var_types,factor_levels,coding_factors)

def li(variables:list[str]):
   """
    Behaves like the :class:`l` class but automatically includes all lower-order interactions and main effects.

    Example: The interaction effect of factor variable "cond", with two levels "1" and "2", and acontinuous variable "x"
    on the dependent variable "y" are of interest. To estimate such a model, the following formula can be used::

      formula = Formula(lhs("y"),terms=[i(),*li(["cond","x"])])

    Note, the use of the ``*`` operator to unpack the individual terms returned from li!
   
    This formula will still (see :class:`l`) estimate the following model:
    
    .. math::

      \\mu = a + b_1*c_i + b_2*x_i + b_3*c_i*x_i

    with: :math:`c` corresponding to a binary predictor variable created so that it is 1 if "cond"=2 else 0.

    To get a model with only main effects for "cond" and "x" :class:`li` **cannot be used** and :class:`l` needs to be used instead::

      formula = Formula(lhs("y"),terms=[i(),l(["cond"]),l(["x"])])

    This formula will estimate:

    .. math::
    
      \\mu_i = a + b_1*c_i + b_2*x_i

    :param variables: A list of the variables (strings) for which linear predictors should be included
    :type variables: list[str]
    """
   
   # Create len(variables)-way interaction, all lower
   # order interactions and main effects (order=1)
   full_order = []
   for order in range(1,len(variables)+1):
      full_order.extend(combinations(variables,order))
   
   order_terms = [l(list(term)) for term in full_order]

   return order_terms

class rs(GammTerm):
    """
    Adds random slopes for the effects of ``variables`` for each level of the
    random factor ``rf``. The type of random slope created depends on the content of ``variables``.
    
    If ``len(variables)==1``, and the string in ``variables`` identifies a categorical variable in the data, then
    a random offset adjustment (for every level of the categorical variable, so without binary coding!) will be
    estimated for every level of the random factor ``rf``.

    Example: The factor variable "cond", with two levels "1" and "2" is assumed to have a general effect on the DV "y".
    However, data was collected from multiple subjects (random factor ``rf`` = "subject") and it is reasonable to assume
    that the effect of "cond" is slightly different for every subject (it is also assumed that all subjects took part
    in both conditions identified by "cond"). A model that accounts for this is estimated via::

      formula = Formula(lhs("y"),terms=[i(),l(["cond"]),rs(["cond"],rf="subject")])
   
    This formula will estimate the following model:

    .. math::

      \\mu = a + b_1*c_i + a_{j(i),cc(i)}

    Here, :math:`c` is again a binary predictor variable created so that it is 1 if "cond"=2 for observation i else 0, :math:`cc(i)` indexes the level of "cond" at observation :math:`i`,
    :math:`j(i)` indexes the level of "subject" at observation :math:`i`, and :math:`a_{j,cc(i)}` identifies the random offset estimated for subject :math:`j` at the level of "cond"
    indicated by :math:`cc(i)`. The :math:`a_{j,cc(i)}` are assumed to be i.i.d :math:`\\sim N(0,\\sigma_a)`. Note that the fixed effect sturcture uses binary coding but the random effect structure does not!
    
    Hence, ``rs(["cond"],rf="subject")`` in ``mssm`` corresponds to adding the term below to a ``mgcv`` model::

      s(cond,subject,bs="re")
   
    If all the strings in ``variables`` identify continuous variables in the data, then a random slope for the
    len(``variables``)-way interaction (will simplify to a slope for a single continuous variable if len(``variables``) == 1)
    will be estimated for every level of the random factor ``rf``.

    Example: The continuous variable "x" is assumed to have a general effect on the DV "y".
    However, data was collected from multiple subjects (random factor ``rf`` ="subject") and it is reasonable to assume
    that the effect of "x" is slightly different for every subject. A model that accounts for this is estimated via::

      formula = Formula(lhs("y"),terms=[i(),l(["x"]),rs(["x"],rf="subject")])
   
    This formula will estimate the following model:

    .. math::
      
      \\mu = a + b*x_i + b_{j(i)} * x_i
    
    Where, :math:`j(i)` again indexes the level of "subject" at observation :math:`i`, :math:`b_j(i)` identifies the random slope (the subject-specific slope adjustment for :math:`b`)
    for variable "x" estimated for subject :math:`j` and the :math:`b_{j(i)}` are again assumed to be i.i.d from a **single** :math:`\\sim N(0,\\sigma_b)`
   
    Note, lower-order interaction slopes (as well as main effects) are **not pulled in by default**! Consider the following formula::

      formula = Formula(lhs("y"),terms=[i(),*li(["x","z"]),rs(["x","z"],rf="subject")])
   
    with another continuous variable "z". This corresponds to the model:

    .. math::
      
      \\mu = a + b_1*x_i + b_2*z_i + b_3*x_i*z_i + b_{j(i)}*x_i*z_i

    With :math:`j(i)` again indexing the level of "subject" at observation i, :math:`b_{j(i)}` identifying the random slope (the subject-specific slope adjustment for :math:`b_3`) for the interaction of
    variables :math:`x` and :math:`z` estimated for subject :math:`j`. The :math:`b_{j(i)}` are again assumed to be i.i.d from a **single** :math:`\\sim N(0,\\sigma_b)`.
    
    To add random slopes for the main effects of either :math:`x` or :math:`z` as well as an additional random intercept, additional :class:`rs`
    and a :class:`ri` terms would have to be added to the formula::

      formula = Formula(lhs("y"),terms=[i(),*li(["x","z"]),
                                       ri("subject"),
                                       rs(["x"],rf="subject"),
                                       rs(["z"],rf="subject"),
                                       rs(["x","z"],rf="subject")])

    If ``len(variables) > 1`` and at least one string in ``variables`` identifies a categorical variable in the data then random slopes for the
    len(``variables``)-way interaction will be estimated for every level of the random factor ``rf``. Separate distribution parameters (the :math:`\\sigma` of
    the Normal) will be estimated for every level of the resulting interaction.

    Example: The continuous variable "x" and the factor variable "cond", with two levels "1" and "2" are assumed to have a general interaction effect
    on the DV "y". However, data was collected from multiple subjects (random factor ``rf`` ="subject") and it is reasonable to assume
    that their interaction effect is slightly different for every subject. A model that accounts for this is estimated via::

      formula = Formula(lhs("y"),terms=[i(),*li(["x","cond"]),rs(["x","cond"],rf="subject")])

    This formula will estimate the following model:

    .. math::
      
      \\mu = a + b_1*c_i + b_2*x_i + b_3*x_i*c_i + b_{j(i),cc(i)}*x_i
    
    With, :math:`c` corresponding to a binary predictor variable created so that it is 1 if "cond"=2 for observation :math:`i` else 0, :math:`cc(i)` corresponds to the level of "cond" at observation :math:`i`,
    :math:`j(i)` corresponds to the level of "subject" at observation :math:`i`, and :math:`b_{j(i),cc(i)}` identifies the random slope for variable :math:`x` at "cond" = :math:`cc(i)` estimated for subject :math:`j`.
    That is: the :math:`b_{j,cc(i)}` where :math:`cc(i)=1` are assumed to be i.i.d realizations from normal distribution :math:`N(0,\\sigma_{b_1})` and the :math:`b_{j,cc(i)}` where :math:`cc(i)=2` are assumed to be
    i.i.d realizations from a **separate normal distribution** :math:`N(0,\\sigma_{b_2})`.

    Hence, adding ``rs(["x","cond"],rf="subject")`` to a ``mssm`` model, is equivalent to adding the term below to a ``mgcv`` model::

      s(x,subject,by=cond,bs="re")

    Correlations between random effects cannot be taken into account by means of parameters (this is possible for example in ``lme4``).

    :param variables: A list of variables. Can point to continuous and categorical variables.
    :type variables: [str]
    :param rf: A factor variable. Identifies the random factor in the data.
    :type rf: str
    """
    def __init__(self,
                 variables:list[str],
                 rf:str) -> None:
        
        # Initialization
        super().__init__(variables, TermType.RANDSLOPE, True, [IdentityPenalty(PenType.IDENTITY)], [{}])
        self.var_coef = None
        self.by = rf
        self.by_cont = None

        # Term name
        self.name = f"rs({variables},{rf})"
    
    def build_penalty(self,ti:int,penalties:list[LambdaTerm],cur_pen_idx:int,factor_levels:dict,col_S:int) -> tuple[list[LambdaTerm],int]:
      """Builds a penalty matrix associated with this random slope term and returns an updated ``penalties`` list including it.

      This method is implemented by most implementations of the :class:`GammTerm` class.
      Two arguments need to be returned: the updated ``penalties`` list including the new penalty implemented as a :class:`LambdaTerm` and the updated ``cur_pen_idx``.
      The latter simply needs to be incremented for every penalty added to ``penalties``.

      :param ti: Index corresponding to the position the current term (i.e., self) takes on in the list of terms of the Formula.
      :type ti: int
      :param penalties: List of previosly created penalties.
      :type penalties: [LambdaTerm]
      :param cur_pen_idx: Index of the last element in ``penalties``.
      :type cur_pen_idx: int
      :param factor_levels: Factor levels dictionary. Keys are factor variables in the data, values are np.arrays holding the unique levels (as str) of the corresponding factor.
      :type factor_levels: dict
      :param col_S: Number of columns of the total penalty matrix.
      :type col_S: int
      :return: Updated ``penalties`` list including the new penalties implemented as a :class:`LambdaTerm` and the updated ``cur_pen_idx``
      :rtype: tuple[list[LambdaTerm],int]
      """
      vars = self.variables

      if self.var_coef is None:
            raise ValueError("Number of coefficients for random slope were not initialized.")
      
      if len(vars) > 1 and self.var_coef > 1:
        # Separate penalties for interactions involving at least one categorical factor.
        # In that case, a separate penalty will describe the random coefficients for the random factor (rterm.by)
        # per level of the (interaction of) categorical factor(s) involved in the interaction.
        # For interactions involving only continuous variables this condition will be false and a single
        # penalty will be estimated.
        idk = len(factor_levels[self.by])
        pen_data,pen_rows,pen_cols,chol_data,chol_rows,chol_cols,rank = self.penalty[0].constructor(idk,None)
        for _ in range(self.var_coef):
          lTerm = LambdaTerm(start_index=cur_pen_idx,
                                       type = PenType.IDENTITY,
                                       term = ti)
    
          lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,idk,cur_pen_idx)
          lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,idk,cur_pen_idx)
          lTerm.S_J = embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J,idk)
          lTerm.rank = rank
          penalties.append(lTerm)

      else:
        # Single penalty for random coefficients of a single variable (categorical or continuous) or an
        # interaction of only continuous variables.
        idk = len(factor_levels[self.by])*self.var_coef
        pen_data,pen_rows,pen_cols,chol_data,chol_rows,chol_cols,rank = self.penalty[0].constructor(idk,None)


        lTerm = LambdaTerm(start_index=cur_pen_idx,
                                     type = PenType.IDENTITY,
                                     term=ti)
    
        lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,idk,cur_pen_idx)
        lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,idk,cur_pen_idx)
        lTerm.S_J = embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J,idk)
        lTerm.rank = rank
        penalties.append(lTerm)

      return penalties, cur_pen_idx
   
    def build_matrix(self,ci:int,ti:int,var_map:dict,var_types:dict,factor_levels:dict,ridx:np.ndarray,cov_flat:np.ndarray,use_only:list[int]) -> tuple[list[float],list[int],list[int],int]:
      """Builds the design/term/model matrix associated with this random slope term and returns it represented as a list of values, a list of row indices, and a list of column indices.

      This method is implemented by every implementation of the :class:`GammTerm` class.
      The returned lists can then be used to create a sparse matrix for this term. Also returns an updated ``ci`` column index, reflecting how many additional columns would be added
      to the total model matrix.

      :param ci: Current column index.
      :type ci: int
      :param ti: Index corresponding to the position the current term (i.e., self) takes on in the list of terms of the Formula.
      :type ti: int
      :param var_map: Var map dictionary. Keys are variables in the data, values their column index in the encoded predictor matrix.
      :type var_map: dict
      :param var_types: Var types dictionary. Keys are variables in the data, values are either ``VarType.NUMERIC`` for continuous variables or ``VarType.FACTOR`` for categorical variables.
      :type var_types: dict
      :param factor_levels: Factor levels dictionary. Keys are factor variables in the data, values are np.arrays holding the unique levels (as str) of the corresponding factor.
      :type factor_levels: dict
      :param ridx: Array of non NAN rows in the data.
      :type ridx: np.ndarray
      :param cov_flat: An array, containing all (encoded, in case of categorical predictors) values on each predictor (each columns of ``cov_flat`` corresponds to a different predictor) variable included in any of the terms in order of the data-frame passed to the Formula.
      :type cov_flat: np.ndarray
      :param use_only: A list holding term indices for which the matrix should be formed. For terms not included in this list a zero matrix will be returned. Can be set to ``None`` so that no terms are excluded.
      :type use_only: [int]
      :return: matrix data, matrix row indices, matrix column indices, added columns
      :rtype: tuple[list[float],list[int],list[int],int]
      """
      by_cov = cov_flat[:,var_map[self.by]]
      by_levels = factor_levels[self.by]
      old_ci = ci

      # First get all columns for all linear predictors associated with this
      # term - might involve interactions!
      lin_elements,\
      lin_rows,\
      lin_cols,\
      lin_ci = build_linear_term(self,False,ci,ti,var_map,
                                 var_types,factor_levels,
                                 ridx,cov_flat,None)
      
      # Need to cast to np.array for indexing
      lin_elements = np.array(lin_elements)
      lin_rows = np.array(lin_rows)
      lin_cols = np.array(lin_cols)

      new_elements = []
      new_rows = []
      new_cols = []
      new_ci = 0
      
      # For every column
      for coef_i in range(lin_ci): 
          # Collect the coefficinet column and row index
          inter_i = lin_elements[lin_cols == old_ci]
          rdx_i = lin_rows[lin_cols == old_ci]
          # split the column over len(by_levels) columns for every level of the random factor
          for fl in range(len(by_levels)): 
            # First check which of the remaining rows correspond to current level of random factor
            fl_idx = by_cov == fl
            # Then adjust to the rows actually present in the interaction column
            fl_idx = fl_idx[rdx_i]
            # Now collect
            if use_only is None or ti in use_only:
                new_elements.extend(inter_i[fl_idx])
                new_rows.extend(rdx_i[fl_idx])
                new_cols.extend([ci for _ in range(len(inter_i[fl_idx]))])
            new_ci += 1
            ci += 1
          old_ci += 1

      # Matrix returned here holds for every linear coefficient one column for every level of the random
      # factor. So: coef1_1, coef_1_2, coef1_3, ... coef_n_1, coef_n,2, coef_n_3

      return new_elements,new_rows,new_cols,new_ci

    def get_coef_info(self,var_types:dict,factor_levels:dict,coding_factors:dict) -> tuple[int,int,list[str]]:
      """Returns the total number of coefficients associated with this random slope term, the number of unpenalized coefficients associated with this term, and a list with names for each of the coefficients associated with this term.

      :param var_types: Var types dictionary. Keys are variables in the data, values are either ``VarType.NUMERIC`` for continuous variables or ``VarType.FACTOR`` for categorical variables.
      :type var_types: dict
      :param factor_levels: Factor levels dictionary. Keys are factor variables in the data, values are np.arrays holding the unique levels (as str) of the corresponding factor.
      :type factor_levels: dict
      :param coding_factors: Factor coding dictionary. Keys are factor variables in the data, values are dictionaries, where the keys correspond to the encoded levels (int) of the factor and the values to their levels (str).
      :type coding_factors: dict
      :return: Number of coefficients associated with term, number of un-penalized coefficients associated with term, coef names
      :rtype: tuple[int,int,list[str]]
      """
      t_total_coef,\
      _,\
      t_coef_names = get_linear_coef_info(self,False,
                                var_types,
                                factor_levels,
                                coding_factors)

      self.var_coef = t_total_coef # We need t_total_coef penalties for this term later.
      by_code_factors = coding_factors[self.by]
      by_code_levels = factor_levels[self.by]
      
      rf_coef_names = []
      for cname in t_coef_names:
        rf_coef_names.extend([f"{cname}_{by_code_factors[fl]}" for fl in range(len(by_code_levels))])
      
      t_ncoef = len(rf_coef_names)

      return t_ncoef,0,rf_coef_names

def build_linear_term(lTerm:l|rs,has_intercept:bool,ci:int,ti:int,var_map:dict,var_types:dict,factor_levels:dict,ridx:np.ndarray,cov_flat:np.ndarray,use_only:list[int]) -> tuple[list[float],list[int],list[int],int]:
  """Builds the design/term/model matrix associated with a linear/random term and returns it represented as a list of values, a list of row indices, and a list of column indices.

  :param lTerm: Linear or random slope term
  :type LTerm: l | rs
  :param has_intercept: Whether or not the formula of which this term is part includes an intercept term.
  :type has_intercept: bool
  :param ci: Current column index.
  :type ci: int
  :param ti: Index corresponding to the position the current term (i.e., self) takes on in the list of terms of the Formula.
  :type ti: int
  :param var_map: Var map dictionary. Keys are variables in the data, values their column index in the encoded predictor matrix.
  :type var_map: dict
  :param var_types: Var types dictionary. Keys are variables in the data, values are either ``VarType.NUMERIC`` for continuous variables or ``VarType.FACTOR`` for categorical variables.
  :type var_types: dict
  :param factor_levels: Factor levels dictionary. Keys are factor variables in the data, values are np.arrays holding the unique levels (as str) of the corresponding factor.
  :type factor_levels: dict
  :param ridx: Array of non NAN rows in the data.
  :type ridx: np.ndarray
  :param cov_flat: An array, containing all (encoded, in case of categorical predictors) values on each predictor (each columns of ``cov_flat`` corresponds to a different predictor) variable included in any of the terms in order of the data-frame passed to the Formula.
  :type cov_flat: np.ndarray
  :param use_only: A list holding term indices for which the matrix should be formed. For terms not included in this list a zero matrix will be returned. Can be set to ``None`` so that no terms are excluded.
  :type use_only: [int]
  :return: matrix data, matrix row indices, matrix column indices, added columns
  :rtype: tuple[list[float],list[int],list[int],int]
  """
  new_elements = []
  new_rows = []
  new_cols = []
  new_ci = 0
  n_y = len(ridx)

  # Main effects
  if len(lTerm.variables) == 1:
    var = lTerm.variables[0]
    if var_types[var] == VarType.FACTOR:
      offset = np.ones(n_y)
      
      fl_start = 0

      if has_intercept: # Dummy coding when intercept is added.
        fl_start = 1

      for fl in range(fl_start,len(factor_levels[var])):
        fridx = ridx[cov_flat[:,var_map[var]] == fl]
        if use_only is None or ti in use_only:
          new_elements.extend(offset[fridx])
          new_rows.extend(fridx)
          new_cols.extend([ci for _ in range(len(fridx))])
        ci += 1
        new_ci += 1

    else: # Continuous predictor
      slope = cov_flat[:,var_map[var]]
      if use_only is None or ti in use_only:
        new_elements.extend(slope)
        new_rows.extend(ridx)
        new_cols.extend([ci for _ in range(n_y)])
      ci += 1
      new_ci += 1

  else: # Interactions
    interactions = []
    inter_idx = []

    for var in lTerm.variables:
      new_interactions = []
      new_inter_idx = []

      # Interaction with categorical predictor as start
      if var_types[var] == VarType.FACTOR:
        fl_start = 0

        if has_intercept: # Dummy coding when intercept is added.
          fl_start = 1

        if len(interactions) == 0:
          for fl in range(fl_start,len(factor_levels[var])):
            new_interactions.append(np.ones(n_y))
            new_inter_idx.append(cov_flat[:,var_map[var]] == fl)

        else:
          for old_inter,old_idx in zip(interactions,inter_idx):
            for fl in range(fl_start,len(factor_levels[var])):
              new_interactions.append(old_inter)
              new_idx = cov_flat[:,var_map[var]] == fl
              new_inter_idx.append(old_idx == new_idx)

      else: # Interaction with continuous predictor as start
        if len(interactions) == 0:
          new_interactions.append(cov_flat[:,var_map[var]])
          new_inter_idx.append(np.array([True for _ in range(n_y)]))

        else:
          for old_inter,old_idx in zip(interactions,inter_idx):
            new_interactions.append(old_inter * cov_flat[:,var_map[var]]) # handle continuous * continuous case.
            new_inter_idx.append(old_idx)

      
      interactions = copy.deepcopy(new_interactions)
      inter_idx = copy.deepcopy(new_inter_idx)

    # Now write interaction terms into model matrix
    for inter,inter_idx in zip(interactions,inter_idx):
      if use_only is None or ti in use_only:
        new_elements.extend(inter[ridx[inter_idx]])
        new_rows.extend(ridx[inter_idx])
        new_cols.extend([ci for _ in range(len(ridx[inter_idx]))])
      ci += 1
      new_ci += 1
  
  return new_elements,new_rows,new_cols,new_ci

def get_linear_coef_info(lTerm:l | rs,has_intercept:bool,var_types:dict,factor_levels:dict,coding_factors:dict) -> tuple[int,int,list[str]]:
  """Returns the total number of coefficients associated with a linear or random term, the number of unpenalized coefficients associated with a linear or random and a list with names for each of the coefficients associated with a linear or random.

  :param lTerm: Linear or random slope term
  :type LTerm: l | rs
  :param has_intercept: Whether or not the formula of which this term is part includes an intercept term.
  :type has_intercept: bool
  :param var_types: Var types dictionary. Keys are variables in the data, values are either ``VarType.NUMERIC`` for continuous variables or ``VarType.FACTOR`` for categorical variables.
  :type var_types: dict
  :param factor_levels: Factor levels dictionary. Keys are factor variables in the data, values are np.arrays holding the unique levels (as str) of the corresponding factor.
  :type factor_levels: dict
  :param coding_factors: Factor coding dictionary. Keys are factor variables in the data, values are dictionaries, where the keys correspond to the encoded levels (int) of the factor and the values to their levels (str).
  :type coding_factors: dict
  :return: Number of coefficients associated with term, number of un-penalized coefficients associated with term, coef names
  :rtype: tuple[int,int,list[str]]
  """
  unpenalized_coef = 0
  coef_names = []
  total_coef = 0

  # Main effects
  if len(lTerm.variables) == 1:
    var = lTerm.variables[0]
    if var_types[var] == VarType.FACTOR:
        
      fl_start = 0

      if has_intercept: # Dummy coding when intercept is added.
          fl_start = 1

      for fl in range(fl_start,len(factor_levels[var])):
          coef_names.append(f"{var}_{coding_factors[var][fl]}")
          unpenalized_coef += 1
          total_coef += 1

    else: # Continuous predictor
      coef_names.append(f"{var}")
      unpenalized_coef += 1
      total_coef += 1

  else: # Interactions
    inter_coef_names = []

    for var in lTerm.variables:
      new_inter_coef_names = []

      # Interaction with categorical predictor as start
      if var_types[var] == VarType.FACTOR:
        fl_start = 0

        if has_intercept: # Dummy coding when intercept is added.
          fl_start = 1

        if len(inter_coef_names) == 0:
          for fl in range(fl_start,len(factor_levels[var])):
            new_inter_coef_names.append(f"{var}_{coding_factors[var][fl]}")
        else:
          for old_name in inter_coef_names:
              for fl in range(fl_start,len(factor_levels[var])):
                new_inter_coef_names.append(old_name + f"_{var}_{coding_factors[var][fl]}")

      else: # Interaction with continuous predictor as start
        if len(inter_coef_names) == 0:
          new_inter_coef_names.append(var)
        else:
            for old_name in inter_coef_names:
              new_inter_coef_names.append(old_name + f"_{var}")
      
      inter_coef_names = copy.deepcopy(new_inter_coef_names)

    # Now add interaction term names
    for name in inter_coef_names:
      coef_names.append(name)
      unpenalized_coef += 1
      total_coef += 1

  return total_coef,unpenalized_coef,coef_names