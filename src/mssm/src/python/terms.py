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
   """Base-class implemented by the terms passed to :class:`mssm.src.python.formula.Formula`.

   :param variables: List of variables as strings.
   :type variables: [str]
   :param type: Type of term as enum
   :type type: TermType
   :param is_penalized: Whether the term is penalized/can be penalized or not
   :type is_penalized: bool
   :param penalty: The default penalties associated with a term.
   :type penalty: [penalties.PenType]
   :param pen_kwargs: A list of dictionaries, each with key-word arguments passed to the construction of the corresponding  :class:`penalties.PenType` in ``penalty``.
   :type pen_kwargs: [dict]
   """
   
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
    An intercept/offset term. In a model
    
    .. math::
       \mu_i = a + f(x_i)
       
    it reflects :math:`a`.

    """

    def __init__(self) -> None:
        super().__init__(["1"], TermType.LINEAR, False, [], [])
        self.name = "Intercept"

class f(GammTerm):
    """
    A univariate or tensor interaction smooth term. If ``variables`` only contains a
    single variable :math:`x`, this term will represent a univariate :math:`f(x)` in a model:
    
    .. math::

      \mu_i = a + f(x_i)
   
    For example, the model below in ``mgcv``:

    ::

      bam(y ~ s(x,k=10) + s(z,k=20))

    would be expressed as follows in ``mssm``:

    ::

      GAMM(Formula(lhs("y"),[i(),f(["x"],nk=9),f(["z"],nk=19)]),Gaussian())
      
    If ``variables`` contains two variables :math:`x` and :math:`z`, then this term will either represent
    the tensor interaction :math:`f(x,z)` in model:
     
    .. math::

      \mu_i = a + f(x_i) + f(z_i) + f(x_i,z_i)
    
    or in model:
    
    .. math::

      \mu_i = a + f(x_i,z_i)

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

    Multiple penalties can be placed on every term by adding ``penalties.PenType`` to the ``penalties``
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
    :param identifiable: Whether or not the constant should be removed from the space of functions this term can fit. Achieved by enforcing that :math:`\mathbf{1}^T \mathbf{X} = 0` (:math:`\mathbf{X}` here is the spline matrix computed for the observed data; see Wood, 2017 for details). Necessary in most cases to keep the model identifiable.
    :type identifiable: bool, optional
    :param basis: The basis functions to use to construct the spline matrix. By default a B-spline basis (Eilers & Marx, 2010) implemented in :func:`mssm.src.smooths.B_spline_basis`.
    :type basis: Callable, optional
    :param basis_kwargs: A list containing one or multiple dictionaries specifying how the basis should be computed. For the B-spline basis the following arguments (with default values) are available: ``convolve``=``False``, ``min_c``=``None``, ``max_c``=``None``, ``deg``=``3``. See :func:`mss.src.smooths.B_spline_basis` for details, but the default should work for most cases.
    :type basis_kwargs: dict, optional
    :param is_penalized: Should the term be left unpenalized or not. There are rarely good reasons to set this to False.
    :type is_penalized: bool, optional
    :param penalize_null: Should a separate Null-space penalty (Marra & Wood, 2011) be placed on the term. By default, the term here will leave a linear f(`variables`) un-penalized! Thus, there is no option for the penalty to achieve f(`variables`) = 0 even if that would be supported by the data. Adding a Null-space penalty provides the penalty with that power. This can be used for model selection instead of Hypothesis testing and is the preferred way in ``mssm`` (see Marra & Wood, 2011 for details).
    :type penalize_null: bool, optional
    :param penalty: A list of penalty types to be placed on the term.
    :type penalty: list[penalties.PenType], optional
    :param pen_kwargs: A list containing one or multiple dictionaries specifying how the penalty should be created. For the default difference penalty (Eilers & Marx, 2010) the only keyword argument (with default value) available is: ``m=2``. This reflects the order of the difference penalty. Note, that while a higher ``m`` permits penalizing towards smoother functions it also leads to an increased dimensionality of the penalty Kernel (the set of bases functions which will not be penalized). In other words, increasingly more complex functions will be left un-penalized for higher ``m`` (except if ``penalize_null`` is set to True). ``m=2`` is usually a good choice and thus the default but see Eilers & Marx (2010) for details.
    :type pen_kwargs: list[dict], optional
    """

    def __init__(self,variables:list,
                by:str=None,
                binary:list[str,str] or None = None,
                id:int=None,
                nk:int or list[int] = 9,
                te: bool = False,
                rp:int = 0,
                constraint:penalties.ConstType=penalties.ConstType.QR,
                identifiable:bool=True,
                basis:Callable=smooths.B_spline_basis,
                basis_kwargs:dict={},
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

        # Term name
        self.name = f"f({variables}"
        if by is not None:
           self.name += f",by={by})"
        elif binary is not None:
           self.name += f",by={binary[1]})"
        else:
           self.name += ")"

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
                by_subgroup:[str,str]or None = None,
                approx_deriv:dict or None = None,
                basis: Callable = smooths.B_spline_basis,
                basis_kwargs: dict = {}):

      penalty = [penalties.PenType.DIFFERENCE]
      pen_kwargs = [{"m":m}]
      super().__init__(variables, rf, None, 99, nk+1, False, rp, penalties.ConstType.QR, False,
                       basis, basis_kwargs,
                       True, True, penalty, pen_kwargs)
      
      self.approx_deriv=approx_deriv
      self.by_subgroup = by_subgroup

      if not self.by_subgroup is None:

         self.name +=  ": " + self.by_subgroup[1]
        
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
       :type penalty: list[penalties.PenType], optional
       :param pen_kwargs: A list containing one or multiple dictionaries specifying how the penalty should be created. For the default difference penalty (Eilers & Marx, 2010) the only keyword argument (with default value) available is: ``m=2``. This reflects the order of the difference penalty. Note, that while a higher ``m`` permits penalizing towards smoother functions it also leads to an increased dimensionality of the penalty Kernel (the set of f[``variables``] which will not be penalized). In other words, increasingly more complex functions will be left un-penalized for higher ``m`` (except if ``penalize_null`` is set to True). ``m=2`` is usually a good choice and thus the default but see Eilers & Marx (2010) for details.
       :type pen_kwargs: list[dict], optional      
       """
    
    def __init__(self,variables:[str],
                event_onset:[int],
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
        self.event_onset = event_onset
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
    Adds a parametric (linear) term to the model formula. The model :math:`\mu_i = a + b*x_i` can for example be achieved
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

      \mu_i = a + b_1*c_i + b_2*x_i + b_3*c_i*x_i

    Here, :math:`c` is a binary predictor variable created so that it is 1 if "cond"=2 else 0 and :math:`b_3` is the coefficient that is added
    because ``l(["cond","x"])`` is included in the terms (i.e., the interaction effect).

    To get a model with only main effects for "cond" and "x", the following formula could be used::

      formula = Formula(lhs("y"),terms=[i(),l(["cond"]),l(["x"])])

    This formula will estimate:

    .. math::
    
      \mu_i = a + b_1*c_i + b_2*x_i

    :param variables: A list of the variables (strings) for which linear predictors should be included
    :type variables: [str]
    """
    def __init__(self,
                 variables:list) -> None:
        
        # Initialization
        super().__init__(variables, TermType.LINEAR, False, [], [])

        # Term name
        self.name = f"l({variables})"

def li(variables:list[str]):
   """
    Behaves like the :class:`l` class but automatically includes all lower-order interactions and main effects.

    Example: The interaction effect of factor variable "cond", with two levels "1" and "2", and acontinuous variable "x"
    on the dependent variable "y" are of interest. To estimate such a model, the following formula can be used::

      formula = Formula(lhs("y"),terms=[i(),*li(["cond","x"])])

    Note, the use of the ``*`` operator to unpack the individual terms returned from li!
   
    This formula will still (see :class:`l`) estimate the following model:
    
    .. math::

      \mu = a + b_1*c_i + b_2*x_i + b_3*c_i*x_i

    with: :math:`c` corresponding to a binary predictor variable created so that it is 1 if "cond"=2 else 0.

    To get a model with only main effects for "cond" and "x" :class:`li` **cannot be used** and :class:`l` needs to be used instead::

      formula = Formula(lhs("y"),terms=[i(),l(["cond"]),l(["x"])])

    This formula will estimate:

    .. math::
    
      \mu_i = a + b_1*c_i + b_2*x_i

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

class ri(GammTerm):
    """
    Adds a random intercept for the factor ``variable`` to the model. The random intercepts :math:`b_i` are assumed
    to be i.i.d :math:`b_i \sim N(0,\sigma_b)` i.e., normally distributed around zero - the simplest random effect supported by ``mssm``.

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
        super().__init__([variable], TermType.RANDINT, True, [penalties.PenType.IDENTITY], [{}])

        # Term name
        self.name = f"ri({variable})"

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

      \mu = a + b_1*c_i + a_{j(i),cc(i)}

    Here, :math:`c` is again a binary predictor variable created so that it is 1 if "cond"=2 for observation i else 0, :math:`cc(i)` indexes the level of "cond" at observation :math:`i`,
    :math:`j(i)` indexes the level of "subject" at observation :math:`i`, and :math:`a_{j,cc(i)}` identifies the random offset estimated for subject :math:`j` at the level of "cond"
    indicated by :math:`cc(i)`. The :math:`a_{j,cc(i)}` are assumed to be i.i.d :math:`\sim N(0,\sigma_a)`. Note that the fixed effect sturcture uses binary coding but the random effect structure does not!
    
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
      
      \mu = a + b*x_i + b_{j(i)} * x_i
    
    Where, :math:`j(i)` again indexes the level of "subject" at observation :math:`i`, :math:`b_j(i)` identifies the random slope (the subject-specific slope adjustment for :math:`b`)
    for variable "x" estimated for subject :math:`j` and the :math:`b_{j(i)}` are again assumed to be i.i.d from a **single** :math:`\sim N(0,\sigma_b)`
   
    Note, lower-order interaction slopes (as well as main effects) are **not pulled in by default**! Consider the following formula::

      formula = Formula(lhs("y"),terms=[i(),*li(["x","z"]),rs(["x","z"],rf="subject")])
   
    with another continuous variable "z". This corresponds to the model:

    .. math::
      
      \mu = a + b_1*x_i + b_2*z_i + b_3*x_i*z_i + b_{j(i)}*x_i*z_i

    With :math:`j(i)` again indexing the level of "subject" at observation i, :math:`b_{j(i)}` identifying the random slope (the subject-specific slope adjustment for :math:`b_3`) for the interaction of
    variables :math:`x` and :math:`z` estimated for subject :math:`j`. The :math:`b_{j(i)}` are again assumed to be i.i.d from a **single** :math:`\sim N(0,\sigma_b)`.
    
    To add random slopes for the main effects of either :math:`x` or :math:`z` as well as an additional random intercept, additional :class:`rs`
    and a :class:`ri` terms would have to be added to the formula::

      formula = Formula(lhs("y"),terms=[i(),*li(["x","z"]),
                                       ri("subject"),
                                       rs(["x"],rf="subject"),
                                       rs(["z"],rf="subject"),
                                       rs(["x","z"],rf="subject")])

    If ``len(variables) > 1`` and at least one string in ``variables`` identifies a categorical variable in the data then random slopes for the
    len(``variables``)-way interaction will be estimated for every level of the random factor ``rf``. Separate distribution parameters (the :math:`\sigma` of
    the Normal) will be estimated for every level of the resulting interaction.

    Example: The continuous variable "x" and the factor variable "cond", with two levels "1" and "2" are assumed to have a general interaction effect
    on the DV "y". However, data was collected from multiple subjects (random factor ``rf`` ="subject") and it is reasonable to assume
    that their interaction effect is slightly different for every subject. A model that accounts for this is estimated via::

      formula = Formula(lhs("y"),terms=[i(),*li(["x","cond"]),rs(["x","cond"],rf="subject")])

    This formula will estimate the following model:

    .. math::
      
      \mu = a + b_1*c_i + b_2*x_i + b_3*x_i*c_i + b_{j(i),cc(i)}*x_i
    
    With, :math:`c` corresponding to a binary predictor variable created so that it is 1 if "cond"=2 for observation :math:`i` else 0, :math:`cc(i)` corresponds to the level of "cond" at observation :math:`i`,
    :math:`j(i)` corresponds to the level of "subject" at observation :math:`i`, and :math:`b_{j(i),cc(i)}` identifies the random slope for variable :math:`x` at "cond" = :math:`cc(i)` estimated for subject :math:`j`.
    That is: the :math:`b_{j,cc(i)}` where :math:`cc(i)=1` are assumed to be i.i.d realizations from normal distribution :math:`N(0,\sigma_{b_1})` and the :math:`b_{j,cc(i)}` where :math:`cc(i)=2` are assumed to be
    i.i.d realizations from a **separate normal distribution** :math:`N(0,\sigma_{b_2})`.

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
        super().__init__(variables, TermType.RANDSLOPE, True, [penalties.PenType.IDENTITY], [{}])
        self.var_coef = None
        self.by = rf

        # Term name
        self.name = f"rs({variables},{rf})"