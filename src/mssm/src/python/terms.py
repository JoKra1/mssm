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
    def __init__(self,
                 by_latent:bool=False) -> None:
        super().__init__(["1"], TermType.LINEAR, False, [], [])
        self.by_latent = by_latent

class f(GammTerm):
    def __init__(self,variables:list,
                by:str=None,
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
        self.id = id
        self.has_null_penalty = penalize_null

        # Tensor bases can each have different number of basis functions
        if len(variables) == 1 or isinstance(nk,list):
         self.nk = nk
        else:
         self.nk = [nk for _ in range(len(variables))]
              
        self.by_latent = by_latent

class fs(f): # Approximates bs='fs' in mgcv
   def __init__(self,
                variables: list,
                rf: str = None,
                nk: int = 9,
                basis: Callable = smooths.B_spline_basis,
                basis_kwargs: dict = {},
                by_latent: bool = False):

      penalty = [penalties.PenType.DIFFERENCE]
      pen_kwargs = [{"m":1}]
      super().__init__(variables, rf, 99, nk+1, False,
                       basis, basis_kwargs, by_latent,
                       True, True, penalty, pen_kwargs)
        
class irf(GammTerm):
    def __init__(self,variable:str,
                state:int,
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
        super().__init__([variable], TermType.LSMOOTH, is_penalized, copy.deepcopy(penalty), copy.deepcopy(pen_kwargs))
        self.basis = basis
        self.basis_kwargs = basis_kwargs
        self.state = state
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