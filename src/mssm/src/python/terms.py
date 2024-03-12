from collections.abc import Callable
from enum import Enum
from itertools import combinations
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
        self.name = None     

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
        self.name = "Intercept"

class f(GammTerm):
    """
    A univariate or tensor interaction smooth term. If ``variables`` only contains a
    single variable 'x', this term will represent a univariate f(x) in a model y = a + f(x). If
    ``variables`` contains two variables 'x' and 'y', then this term will either represent
    the tensor interaction f(x,y) in a model a + f(x) + f(y) + f(x,y) or in a model a + f(x,y).
    The first behavior is achieved by setting ``te=False``. In that case it is thus necessary
    to add 'main effect' ``f()`` terms for 'x' and 'y'. In other words, the behavior then mimicks
    the ``ti()`` term available in mgcv (Wood, 2017). If ``te=True``, the term instead behaves like
    a ``te()`` term in mgcv, so no separate smooth effects for the main effects need to be included.

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
    :param te: For tensor interaction terms only. If set to false, the term mimics the behavior of ti() in mgcv (Wood, 2017).
    Otherwise, the term behaves like a te() term in mgcv - i.e., the marginal basis functions are not removed from the interaction.
    :type te: bool, optional
    :param constraint: What kind of identifiability constraints should be absorbed by the terms (if they are to be identifiable). Either QR-based
    constraints (default, well-behaved, expensive infill), by means of column-dropping (no infill, not so well-behaved for large k), or by means of
    difference re-coding (little infill, not so well behaved for small k).
    :type constraint: mssm.src.constraints.ConstType, optional
    :param identifiable: Whether or not the constant should be removed from the space of functions this term can
    fit. Achieved by enforcing that 1.T @ X = 0 (X here is the spline matrix computed for the observed data;
    see Wood, 2017 for details). Necessary in most cases to keep the model identifiable.
    :type identifiable: bool, optional
    :param basis: The basis functions to use to construct the spline matrix. By default a B-spline basis
    (Eilers & Marx, 2010) implemented in ``src.smooths.B_spline_basis``.
    :type basis: Callable, optional
    :param basis_kwargs: A list containing one or multiple dictionaries specifying how the basis should be computed.
    For the B-spline basis the following arguments (with default values) are available: ``convolve``=``False``,
    ``min_c``=``None``, ``max_c``=``None``, ``deg``=``3``. See ``src.smooths.B_spline_basis`` for details, but the default should work for most cases.
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
                te: bool = False,
                rp:int = 0,
                constraint:penalties.ConstType=penalties.ConstType.DROP,
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
        
        if te==True and len(variables) == 1:
           raise ValueError("``te`` can only be set to true in case multiple variables are provided via ``variables``.")
        
        if not binary is None and identifiable:
           # Remove identifiability constrain for
           # binary difference smooths.
           identifiable = False
           # For te terms this can be skipped for these one coef is simply
           # subtracted after identifiability constraints have been observed,
           # while for all other terms mssm acts as if one additional coef was requested.
           if te is False:
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
        self.Z = constraint
        self.should_rp = rp
        self.RP = []
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

        self.te = te
              
        self.by_latent = by_latent

        # Term name
        self.name = f"f({variables}"
        if by is not None:
           self.name += f",by={by})"
        elif binary is not None:
           self.name += f",by={binary[1]})"

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
    :param constraint: What kind of identifiability constraints should be absorbed by the terms (if they are to be identifiable). Either QR-based
    constraints (default, well-behaved, expensive infill), by means of column-dropping (no infill, not so well-behaved for large k), or by means of
    difference re-coding (little infill, not so well behaved for small k).
    :type constraint: mssm.src.constraints.ConstType, optional
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
                m: int = 1,
                rp:int = 1,
                constraint:penalties.ConstType=penalties.ConstType.DROP,
                basis: Callable = smooths.B_spline_basis,
                basis_kwargs: dict = {},
                by_latent: bool = False):

      penalty = [penalties.PenType.DIFFERENCE]
      pen_kwargs = [{"m":m}]
      super().__init__(variables, rf, None, 99, nk+1, False, rp, constraint, False,
                       basis, basis_kwargs, by_latent,
                       True, True, penalty, pen_kwargs)
        
class irf(GammTerm):
    def __init__(self,variables:[str],
                event:int,
                basis_kwargs:list[dict],
                by:str=None,
                id:int=None,
                nk:int=10,
                basis:Callable=smooths.B_spline_basis,
                is_penalized:bool = True,
                penalty:list[penalties.PenType] or None = None,
                pen_kwargs:list[dict] or None = None) -> None:
        
        # Default penalty setup
        if penalty is None:
           penalty = [penalties.PenType.DIFFERENCE]
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
        super().__init__(variables, TermType.LSMOOTH, is_penalized, copy.deepcopy(penalty), copy.deepcopy(pen_kwargs))
        self.basis = basis
        self.basis_kwargs = basis_kwargs
        self.event = event
        self.by = by
        self.id = id

        # nk can also be a list for irf smooths.
        if len(variables) == 1 or isinstance(nk,list):
         self.nk = nk
        else:
         self.nk = [nk for _ in range(len(variables))]

        # Term name
        self.name = f"f({'_'.join(variables)}"
        if by is not None:
           self.name += f",by={by})"

class l(GammTerm):
    """
    Adds a parametric (linear) term to the model formula. The model y = a + b*x can for example be achieved
    by adding [i(), l(['x'])] to the ``term`` argument of a ``Formula``. The coefficient "b" estimated for
    the term will then correspond to the slope of "x". This class can also be used to add predictors for
    categorical variables. If the formula includes an intercept, binary coding will be utilized for to
    add reference-level adjustment coefficients for the remaining k-1 levels of the factor variable.

    If more than variable is included in ``variables`` the model will only add the the len(``variables``)-interaction
    to the model! Lower order interactions and main effects will not be included by default (see li() function instead, which
    automatically includes all lower-order interactions and main effects).

    Example: The interaction effect of factor variable "cond", with two levels "1" and "2", and acontinuous variable "x"
    on the dependent variable "y" are of interest. To estimate such a model, the following formula can be used:
         formula = Formula(lhs("y),terms=[i(),l(["cond"]),l(["x"]),l(["cond","x"])])
   
    This formula will estimate the following model:
         y_hat = a + b1*c + b2*x + b3*c*x
         with: c = binary predictor variable created so that it is 1 if "cond"=2 else 0
         b3 is the coefficient that is added because l(["cond","x"]) is included in the terms.

    To get a model with only main effects for "cond" and "x", the following formula could be used:
         formula = Formula(lhs("y),terms=[i(),l(["cond"]),l(["x"])])

    This formula will estimate:
         y_hat = a + b1*c + b2*x

    Parameters:

    :param variables: A list of the variables (strings) for which linear predictors should be included
    :type variables: list[str]
    :param by_latent: Should linear terms be added separately "by_latent" stage or not
    :type by_latent: bool, optional
    """
    def __init__(self,
                 variables:list,
                 by_latent:bool=False) -> None:
        
        # Initialization
        super().__init__(variables, TermType.LINEAR, False, [], [])
        self.by_latent = by_latent

        # Term name
        self.name = f"l({variables})"

def li(variables:list[str],by_latent:bool=False):
   """
    Behaves like the l() class but li() automatically includes all lower-order interactions and main effects.

    Example: The interaction effect of factor variable "cond", with two levels "1" and "2", and acontinuous variable "x"
    on the dependent variable "y" are of interest. To estimate such a model, the following formula can be used:
         formula = Formula(lhs("y),terms=[i(),*li(["cond","x"])])

    Note, the use of the "*" operator to unpack the individual terms returned from li!
   
    This formula will still estimate the following model:
         y_hat = a + b1*c + b2*x + b3*c*x
         with: c = binary predictor variable created so that it is 1 if "cond"=2 else 0

    To get a model with only main effects for "cond" and "x" ``li()`` cannot be used and ``l()`` needs to be used instead:
         formula = Formula(lhs("y),terms=[i(),l(["cond"]),l(["x"])])

    This formula will estimate:
         y_hat = a + b1*c + b2*x

    Parameters:

    :param variables: A list of the variables (strings) for which linear predictors should be included
    :type variables: list[str]
    :param by_latent: Should linear terms be added separately "by_latent" stage or not
    :type by_latent: bool, optional
    """
   
   # Create len(variables)-way interaction, all lower
   # order interactions and main effects (order=1)
   full_order = []
   for order in range(1,len(variables)+1):
      full_order.extend(combinations(variables,order))
   
   order_terms = [l(list(term),by_latent) for term in full_order]

   return order_terms

class ri(GammTerm):
    """
    Adds a random intercept for the factor ``variable`` to the model. The random intercepts "b" are assumed
    to be "b ~ N(0,sigma_b)" i.e., normally distributed around zero - the simplest random effect supported by ``mssm``.

    The ``variable`` needs to identify a factor-variable in the data (dat[''variable''].dtype == 'O'). If you want to
    add more complex random effects to the model (e.g., random slopes for continuous variable "x" per level of factor
    ``variable``) use the ``rs()`` class.

    Parameters:

    :param variable: A factor variable. For every level of this factor a random intercept will be estimated. The random
    intercepts are assumed to follow a normal distribution centered around zero.
    :type variable: str
    :param by_latent: Should random intercepts be added separately "by_latent" stage or not
    :type by_latent: bool, optional
    """
    def __init__(self,
                 variable:str,
                 by_latent:bool=False) -> None:
        
        # Initialization
        super().__init__([variable], TermType.RANDINT, True, [penalties.PenType.IDENTITY], [{}])
        self.by_latent = by_latent

        # Term name
        self.name = f"ri({variable})"

class rs(GammTerm):
    """
    Adds random slopes for the effects of the term encoded by the ``variables`` for each level of the
    random factor ``rf``. The type of random slope created depends on the ``variables``.
    
    If len(``variables``)==1, and the str in ``variables`` identifies a categorical variable in the data, then
    a random offset adjustment (for every level of the categorical variable, so without binary coding!) will be
    estimated for every level of the random factor ``rf``.

    Example: The factor variable "cond", with two levels "1" and "2" is assumed to have a general effect on the DV "y".
    However, data was collected from multiple subjects (random factor ``rf``="subject") and it is reasonable to assume
    that the effect of "cond" is slightly different for every subject (it is also assumed that all subjects took part
    in both conditions identified by "cond"). A model that accounts for this is estimated via:
      formula = Formula(lhs("y),terms=[i(),l(["cond"]),rs(["cond"],rf="subject")])
   
    This formula will estimate the following model:
         y^hat_i = a + b1*c_i + a_{j(i),cc(i)}
         with: c = binary predictor variable created so that it is 1 if "cond"=2 for observation i else 0
         and:  cc(i) corresponding to the level of "cond" at observation i
         and:  j(i) corresponding to the level of "subject" at observation i
         and:  a_{j,cc(i)} identifying the random offset estimated for subject j and the level of "cond"
               indicated by cc(i). The a_{j,cc(i)} are assumed to be from a **single** normal distribution N(0,sigma_a)
   
    Note that the fixed effect sturcture uses binary coding but the random effect structure does not.

    If all the str in ``variables`` identify continuous variables in the data, then a random slope for the
    len(``variables``)-way interaction (will simplify to a slope for a single continuous variable if len(``variables``) == 1)
    will be estimated for every level of the random factor ``rf``.

    Example: The continuous variable "x" is assumed to have a general effect on the DV "y".
    However, data was collected from multiple subjects (random factor ``rf``="subject") and it is reasonable to assume
    that the effect of "x" is slightly different for every subject. A model that accounts for this is estimated via:
      formula = Formula(lhs("y),terms=[i(),l(["x"]),rs(["x"],rf="subject")])
   
    This formula will estimate the following model:
         y^hat_i = a + b*x_i + b_j(i) * x_i
         with: j(i) corresponding to the level of "subject" at observation i
         and:  b_j(i) identifying the random slope (the subject-specific slope adjustment for "b") for variable "x" estimated
         for subject j. The b_j(i) are assumed to be from a **single** normal distribution N(0,sigma_b)
   
    Note, lower-order interaction slopes (as well as main effects) are not pulled in by default! Consider the following formula:
      formula = Formula(lhs("y),terms=[i(),*li(["x","z"]),rs(["x","z"],rf="subject")])
      with: another continuous variable "z"
    
    This corresponds to the model:
      y^hat_i = a + b1*x_i + b2*z_i + b3*x_i*z_i + b_j(i)*x_i*z_i
      with: j(i) corresponding to the level of "subject" at observation i
      and:  b_j(i) identifying the random slope (the subject-specific slope adjustment for "b3") for the interaction of
      variables "x" and "z" estimated for subject j. The b_j(i) are assumed to be from a **single** normal distribution N(0,sigma_b)
    
    To add random slopes for the main effects of either "x" or "z" as well as an additional random intercept, additional ``rs``
    and a ``ri`` would have to be added to the formula:
      formula = Formula(lhs("y),terms=[i(),*li(["x","z"]),
                                       ri("subject"),
                                       rs(["x"],rf="subject"),
                                       rs(["z"],rf="subject"),
                                       rs(["x","z"],rf="subject")])

    If len(``variables``) > 1 and at least one str in ``variables`` identifies a categorical variable in the data then random slopes for the
    len(``variables``)-way interaction will be estimated for every level of the random factor ``rf``. Separate distribution parameters (the sigma of
    the Normal) will be estimated for every level of the resulting interaction.

    Example: The continuous variable "x" and the factor variable "cond", with two levels "1" and "2" are assumed to have a general interaction effect
    on the DV "y". However, data was collected from multiple subjects (random factor ``rf``="subject") and it is reasonable to assume
    that the interaction effect is slightly different for every subject. A model that accounts for this is estimated via:
      formula = Formula(lhs("y),terms=[i(),*li(["x","cond"]),rs(["x","cond"],rf="subject")])

    This formula will estimate the following model:
         y^hat_i = a + b1*c_i + b2*x_i + b3*x_i*c_i + b_{j(i),cc(i)}*x_i
         with: c = binary predictor variable created so that it is 1 if "cond"=2 for observation i else 0
         and:  cc(i) corresponding to the level of "cond" at observation i
         and:  j(i) corresponding to the level of "subject" at observation i
         and:  b_{j(i),cc(i)} identifying the random slope for variable "x" and "cond"=cc(i) estimated for subject j.
         The b_{j,cc(i)} where cc(i)=1 are assumed to be from a normal distribution N(0,sigma_b1) and the b_{j,cc(i)} where cc(i)=2
         are assumed to be from a separate normal distribution N(0,sigma_b2).

    
    Correlations between random effects cannot be taken into account by means of parameters (this is possible for example in lme4).

    Parameters:

    :param variables: A list of variables. Can point to continuous and categorical variables.
    :type variables: list[str]
    :param rf: A factor variable. Identifies the random factor in the data.
    :type rf: str
    :param by_latent: Should random slopes be added separately "by_latent" stage or not
    :type by_latent: bool, optional
    """
    def __init__(self,
                 variables:list[str],
                 rf:str,
                 by_latent:bool=False) -> None:
        
        # Initialization
        super().__init__(variables, TermType.RANDSLOPE, True, [penalties.PenType.IDENTITY], [{}])
        self.var_coef = None
        self.by = rf
        self.by_latent = by_latent

        # Term name
        self.name = f"rs({variables},{rf})"