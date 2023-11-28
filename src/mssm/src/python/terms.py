from collections.abc import Callable
from enum import Enum
import copy
from . import smooths
from . import penalties

class TermType(Enum):
    LSMOOTH = 1
    SMOOTH = 2
    LINEAR = 3
    RANDINT = 4
    RANDSLOPE = 5

class GammTerm():
   
   def __init__(self,variables:list[str],
                type:TermType,
                is_penalized:bool,
                penalty:list[penalties.PenType],
                pen_kwargs:list[dict]) -> None:
        
        self.variables = variables
        self.type = type
        self.is_penalized = is_penalized
        self.penalty = penalty
        self.pen_kwargs = pen_kwargs     

class i(GammTerm):
    """
    An intercept/offset term. In a model y = a + f(x) it reflects a.

    Parameters:

    :param by_latent: Should an overall intercept be added or one "by_latent" stage
    :type by_latent: bool, optional
    """

    def __init__(self,
                 by_latent:bool=False) -> None:
        super().__init__(["1"], TermType.LINEAR, False, [], [])
        self.by_latent = by_latent

class f(GammTerm):
    """
    A univariate or tensor interaction smooth term. If ``variables`` only contains a
    single variable 'x', this term will represent a univariate f(x) in a model y = a + f(x). If
    ``variables`` contains two variables 'x' and 'y', then this term will represent
    the tensor interaction f(x,y) in a model a + f(x) + f(y) + f(x,y). In that case it is thus necessary
    to add 'main effect' ``f()`` terms for 'x' and 'y'. In other words, the behavior for multiple
    variables here mimicks the ``ti()`` term available in mgcv (Wood, 2017).

    By default a B-spline basis is used with ``nk``=9 basis functions (after removing identifiability
    constrains). This is equivalent to mgcv's default behavior of using 10 basis functions
    (before removing identifiability constrains). In case ``variables`` contains more then one variable
    ``nk`` can either bet set to a single value or to a list containing the number of basis functions
    that should be used to setup the spline matrix for every variable. The former implies that the same
    number of coefficients should be used for all variables. Keyword arguments that change the computation of
    the spline basis can be passed via a dictionary to the ``basis_kwargs`` argument. Importantly, if
    multiple variables are present and a list is passed to ``nk``, a list of dictionaries with keyword arguments
    of the same length needs to be passed to ``basis_kwargs`` as well.

    Multiple penalties can be placed on every term by adding ``penalties.PenType`` to the ``penalties``
    argument. In case ``variables`` contains multiple variables a separate tensor penalty (see Wood, 2017) will
    be created for every penalty included in ``penalties``. Again, key-word arguments that alter the behavior of
    the penalty creation need to be passed as dictionaries to ``pen_kwargs`` for every penalty included in ``penalties``.
    By default, a univariate term is penalized with a difference penalty of order 2 (Eilers & Marx, 2010). 

    References:
    - Eilers, P., & Marx, B. (2010). Splines, knots, and penalties. https://doi.org/10.1002/WICS.125
    - Marra, G., & Wood, S. N. (2011). Practical variable selection for generalized additive models.
    Computational Statistics & Data Analysis, 55(7), 2372–2387. https://doi.org/10.1016/j.csda.2011.02.004
    - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.). Chapman and Hall/CRC.

    Parameters:

    :param variables: A list of the variables (strings) of which the term is a function.
    Need to exist in ``data`` passed to ``Formula``. Need to be continuous.
    :type variables: list[str]
    :param by: A string corresponding to a factor in ``data`` passed to ``Formula``. Separate f(``variables``)
    (and smoothness penalties) will be estimated per level of ``by``.
    :type by: str, optional
    :param binary: A list containing two strings. The first string corresponds to a factor in ``data`` passed to
    ``Formula``. A separate f(``variables``) will be estimated for the level of this factor corresponding to the second string.
    :type binary:list[str,str], optional
    :param id: Only useful in combination with specifying a ``by`` variable. If id is set to any integer the
    penalties placed on the separate f(``variables``) will share a single smoothness penalty.
    :type id: int, optional
    :param nk: Number of basis functions to use. Even if ``identifiable`` is true, this number will reflect the
    final number of basis functions for this term (i.e., mssm acts like you would have asked for 10 basis
    functions if ``nk``=9 and identifiable=True; the default).
    :type nk: int or list[int], optional
    :param identifiable: Whether or not the constant should be removed from the space of functions this term can
    fit. Achieved by enforcing that 1.T @ X = 0 (X here is the spline matrix computed for the observed data;
    see Wood, 2017 for details). Necessary in most cases to keep the model identifiable.
    :type identifiable: bool, optional
    :param basis: The basis functions to use to construct the spline matrix. By default a B-spline basis
    (Eilers & Marx, 2010) implemented in ``src.smooths.B_spline_basis``.
    :type basis: Callable, optional
    :param basis_kwargs: A list containing one or multiple dictionaries specifying how the basis should be computed.
    For the B-spline basis the following arguments (with default values) are available: ``convolve``=``False``,
    ``min_c``=``None``, ``max_c``=``None``, ``deg``=``3``. See ``src.smooths.B_spline_basis`` for details.
    :type basis_kwargs: dict, optional
    :param by_latent: Should an overall f(``variables``) be added or one "by_latent" stage
    :type by_latent: bool, optional
    :param is_penalized: Should the term be left unpenalized or not. There are rarely good reasons to set this to False.
    :type is_penalized: bool, optional
    :param penalize_null: Should a separate Null-space penalty (Marra & Wood, 2011) be placed on the term. By default,
    the term here will leave a linear f(`variables`) un-penalized! Thus, there is no option for the penalty to achieve
    f(`variables`) = 0 even if that would be supported by the data. Adding a Null-space penalty provides the penalty
    with that power. This can be used for model selection instead of Hypothesis testing and is the preferred way in mssm
    (see Marra & Wood, 2011 for details).
    :type penalize_null: bool, optional
    :param penalty: A list of penalty types to be placed on the term.
    :type penalty: list[penalties.PenType], optional
    :param pen_kwargs: A list containing one or multiple dictionaries specifying how the penalty should be created. For the
    default difference penalty (Eilers & Marx, 2010) the only keyword argument (with default value) available is: ``m``=2.
    This reflects the order of the difference penalty. Note, that while a higher ``m`` permits penalizing towards smoother
    functions it also leads to an increased dimensionality of the penalty Kernel (the set of f[``variables``] which will
    not be penalized). In other words, increasingly more complex functions will be left un-penalized for higher ``m``
    (except if ``penalize_null`` is set to True). ``m``=2 is usually a good choice and thus the default but see
    Eilers & Marx (2010) for details.
    :type pen_kwargs: list[dict], optional
    """

    def __init__(self,variables:list,
                by:str=None,
                binary:list[str,str] or None = None,
                id:int=None,
                nk:int or list[int] = 9,
                identifiable:bool=True,
                basis:Callable=smooths.B_spline_basis,
                basis_kwargs:dict={"convolve":False},
                by_latent:bool=False,
                is_penalized:bool = True,
                penalize_null:bool = False,
                penalty:list[penalties.PenType] or None = None,
                pen_kwargs:list[dict] or None = None) -> None:
        
        if not binary is None and not by is None:
           raise ValueError("Binary smooths cannot also have a by-keyword.")
        
        if not binary is None and identifiable:
           # Remove identifiability constrain for
           # binary difference smooths.
           identifiable = False
           nk = nk + 1

        # Default penalty setup
        if penalty is None:
           penalty = [penalties.PenType.DIFFERENCE]
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
        self.Z = None
        self.basis = basis
        self.basis_kwargs = basis_kwargs
        self.by = by
        self.binary = binary
        self.binary_level = None
        self.id = id
        self.has_null_penalty = penalize_null

        # Tensor bases can each have different number of basis functions
        if len(variables) == 1 or isinstance(nk,list):
         self.nk = nk
        else:
         self.nk = [nk for _ in range(len(variables))]
              
        self.by_latent = by_latent

class fs(f):
   """
    Essentially a ``f()`` term with ``by``=``rf``, ``id`` != None, ``penalize_null`` = True, and ``pen_kwargs`` = ``[{"m":1}]``.
    This term approximates the "factor-smooth interaction" basis "fs" with ``m``= 1 available in mgcv (Wood, 2017). It is however
    not equivalent (mgcv by default uses a very different basis and the ``m`` key-word has a different functionality).
    
    Specifically, here ``m``= 1 implies that the only function left unpenalized by the default (difference) penalty is the constant. Thus,
    a linear ``f(``variables``)`` is penalized by the same default penalty that also penalizes smoothness (and not by a separate penalty as
    is the case in mgcv when ``m``=1)! Any constant ``f(``variables``)`` is penalized by the null-space penalty (in both mgcv and mssm;
    see Marra & Wood, 2011) - the term thus shrinks towards zero (Wood, 2017).

    References:
    - Eilers, P., & Marx, B. (2010). Splines, knots, and penalties. https://doi.org/10.1002/WICS.125
    - Marra, G., & Wood, S. N. (2011). Practical variable selection for generalized additive models.
    Computational Statistics & Data Analysis, 55(7), 2372–2387. https://doi.org/10.1016/j.csda.2011.02.004
    - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.). Chapman and Hall/CRC.

    Parameters:

    :param variables: A list of the variables (strings) of which the term is a function.
    Need to exist in ``data`` passed to ``Formula``. Need to be continuous.
    :type variables: list[str]
    :param rf: A string corresponding to a (random) factor in ``data`` passed to ``Formula``. Separate f(``variables``)
    (but a shared smoothness penalty!) will be estimated per level of ``rf``.
    :type rf: str, optional
    :param nk: Number of basis functions -1 to use. I.e., if ``nk``=9 (the default), the term will use 10 basis functions.
    By default ``f()`` has identifiability constraints applied and we act as if ``nk``+ 1 coefficients were
    requested. The ``fs()`` term needs no identifiability constrains so if the same number of coefficients used for
    a ``f()`` term is requested (the desired approach), one coefficient is added to compensate for the lack of
    identifiability constraints. This is the opposite to how this is handled in mgcv: specifying nk=10 for "fixed" univariate smooths
    results in 9 basis functions being available. However, for a smooth in mgcv with basis='fs', 10 basis functions will remain available.
    :type nk: int or list[int], optional
    :param basis: The basis functions to use to construct the spline matrix. By default a B-spline basis
    (Eilers & Marx, 2010) implemented in ``src.smooths.B_spline_basis``.
    :type basis: Callable, optional
    :param basis_kwargs: A list containing one or multiple dictionaries specifying how the basis should be computed.
    For the B-spline basis the following arguments (with default values) are available: ``convolve``=``False``,
    ``min_c``=``None``, ``max_c``=``None``, ``deg``=``3``. See ``src.smooths.B_spline_basis`` for details.
    :type basis_kwargs: dict, optional
    :param by_latent: Should an overall f(``variables``) be added or one "by_latent" stage
    :type by_latent: bool, optional
    """
   
   def __init__(self,
                variables: list,
                rf: str = None,
                nk: int = 9,
                basis: Callable = smooths.B_spline_basis,
                basis_kwargs: dict = {},
                by_latent: bool = False):

      penalty = [penalties.PenType.DIFFERENCE]
      pen_kwargs = [{"m":1}]
      super().__init__(variables, rf, None, 99, nk+1, False,
                       basis, basis_kwargs, by_latent,
                       True, True, penalty, pen_kwargs)
        
class irf(GammTerm):
    def __init__(self,variables:[str],
                event:int,
                by:str=None,
                id:int=None,
                nk:int=10,
                basis:Callable=smooths.B_spline_basis,
                basis_kwargs:dict={"convolve":True},
                is_penalized:bool = True,
                penalty:list[penalties.PenType] or None = None,
                pen_kwargs:list[dict] or None = None) -> None:
        
        # Default penalty setup
        if penalty is None:
           penalty = [penalties.PenType.DIFFERENCE]
           pen_kwargs = [{"m":2}]
        
        # Initialization: ToDo: the deepcopy can be dropped now.
        super().__init__(variables, TermType.LSMOOTH, is_penalized, copy.deepcopy(penalty), copy.deepcopy(pen_kwargs))
        self.basis = basis
        self.basis_kwargs = basis_kwargs
        self.event = event
        self.by = by
        self.id = id
        self.nk = nk

class l(GammTerm):
    def __init__(self,
                 variables:list,
                 by_latent:bool=False) -> None:
        
        # Initialization
        super().__init__(variables, TermType.LINEAR, False, [], [])
        self.by_latent = by_latent

class ri(GammTerm):
    def __init__(self,
                 variable:str,
                 by_latent:bool=False) -> None:
        
        # Initialization
        super().__init__([variable], TermType.RANDINT, True, [penalties.PenType.IDENTITY], [{}])
        self.by_latent = by_latent

class rs(GammTerm):
    def __init__(self,
                 variables:list,
                 rf:str,
                 by_latent:bool=False) -> None:
        
        # Initialization
        super().__init__(variables, TermType.RANDSLOPE, True, [penalties.PenType.IDENTITY], [{}])
        self.var_coef = None
        self.by = rf
        self.by_latent = by_latent