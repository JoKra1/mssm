import warnings
import copy
from collections.abc import Callable
import numpy as np
import scipy as scp
import pandas as pd
from tqdm import tqdm
from .smooths import TP_basis_calc
from .terms import GammTerm,i,l,f,irf,ri,rs,fs
from .penalties import embed_in_S_sparse,embed_in_Sj_sparse
from .file_loading import read_cov, read_cor_cov_single, read_cov_no_cor ,read_unique,read_dtype,mp,repeat,setup_cache,clear_cache,os
from .custom_types import PenType,LambdaTerm,Reparameterization,ConstType,Constraint,VarType
from .matrix_solvers import translate_sparse,eigen_solvers
import math
import sys

class lhs():
    """
    The Left-hand side of a regression equation.

    See the :class:`Formula` class for examples.

    :param variable: The name of the dependent/response variable in the dataframe passed to a :class:`Formula`. Can point to continuous and categorical variables. For :class:`mssm..models.GSMM` models, the variable can also be set to any placeholder variable in the data, since not every :class:`Formula` will be associated with a particular response variable.
    :type variable: str
    :param f: A function that will be applied to the ``variable`` before fitting. For example: np.log(). By default no function is applied to the ``variable``.
    :type f: Callable, optional
    """
    def __init__(self,variable:str,f:Callable=None) -> None:
        self.variable = variable
        self.f=f

def _compute_constraint_single_MP(sterm:f,vars:list[str],lhs_var:str,file:str,var_mins:dict,var_maxs:dict,file_loading_kwargs:dict) -> np.ndarray:
   """Internal function to compute QR identifiability constraint based on reading model matrix data from file

   Wood (2017) has an overview over identifiability constraintsfor smooth terms.

   References:
      - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

   :param sterm: Smooth term to make identifiable
   :type sterm: f
   :param vars: List of variables
   :type vars: list[str]
   :param lhs_var: name of dependent variable
   :type lhs_var: str
   :param file: file name
   :type file: str
   :param var_mins: Dictionary holding covariate minimums
   :type var_mins: dict
   :param var_maxs: Dictionary holding covariate maximums
   :type var_maxs: dict
   :param file_loading_kwargs: Any optional file loading key-word arguments.
   :type file_loading_kwargs: dict
   :return: Constraint vector ``C``
   :rtype: np.ndarray
   """

   C = 0

   if len(vars) > 1 and sterm.te == False:
      C = [0 for _ in range(len(vars))]

   matrix_term = None # for Te basis
   for vi in range(len(vars)):
      # If a smooth term needs to be identifiable I act as if you would have asked for nk+1!
      # so that the identifiable term is of the dimension expected.
      
      if len(vars) > 1:
         id_nk = sterm.nk[vi]
      else:
         id_nk = sterm.nk
      
      if sterm.te == False:
         id_nk += 1

      var_cov_flat = read_cor_cov_single(lhs_var,vars[vi],file,file_loading_kwargs)

      matrix_term_v = sterm.basis(var_cov_flat,
                                    None,id_nk,min_c=var_mins[vars[vi]],
                                    max_c=var_maxs[vars[vi]], **sterm.basis_kwargs)

      if sterm.te == False:
         if len(vars) > 1:
            C[vi] += np.sum(matrix_term_v,axis=0).reshape(-1,1)
         else:
            C += np.sum(matrix_term_v,axis=0).reshape(-1,1)
      else:
         if vi == 0:
            matrix_term = matrix_term_v
         else:
            matrix_term = TP_basis_calc(matrix_term,matrix_term_v)

   # Now deal with te basis
   if sterm.te:
      C += np.sum(matrix_term,axis=0).reshape(-1,1)

   return C

class Formula():
    """The formula of a regression equation.
   
    **Note:** The class implements multiple ``get_*`` functions to access attributes stored in instance variables. The get functions always return a copy of the
    instance variable and the results are thus safe to manipulate.

    Examples::

      from mssm.models import *
      from mssmViz.sim import *

      from mssm.src.python.formula import build_penalties,build_model_matrix

      # Get some data and formula
      Binomdat = sim3(10000,0.1,family=Binomial(),seed=20)
      formula = Formula(lhs("y"),[i(),f(["x0"]),f(["x1"]),f(["x2"]),f(["x3"])],data=Binomdat)

      # Now with a tensor smooth
      formula = Formula(lhs("y"),[i(),f(["x0","x1"],te=True),f(["x2"]),f(["x3"])],data=Binomdat)

      # Now with a tensor smooth anova style
      formula = Formula(lhs("y"),[i(),f(["x0"]),f(["x1"]),f(["x0","x1"]),f(["x2"]),f(["x3"])],data=Binomdat)


      ######## Stream data from file and set up custom codebook #########

      file_paths = [f'https://raw.githubusercontent.com/JoKra1/mssmViz/main/data/GAMM/sim_dat_cond_{cond}.csv' for cond in ["a","b"]]

      # Set up specific coding for factor 'cond'
      codebook = {'cond':{'a': 0, 'b': 1}}

      formula = Formula(lhs=lhs("y"), # The dependent variable - here y!
                        terms=[i(), # The intercept, a
                                 l(["cond"]), # For cond='b'
                                 f(["time"],by="cond"), # to-way interaction between time and cond; one smooth over time per cond level
                                 f(["x"],by="cond"), # to-way interaction between x and cond; one smooth over x per cond level
                                 f(["time","x"],by="cond"), # three-way interaction
                                 fs(["time"],rf="sub")], # Random non-linear effect of time - one smooth per level of factor sub
                        data=None, # No data frame!
                        file_paths=file_paths, # Just a list with paths to files.
                        print_warn=False,
                        codebook=codebook)

      # Alternative:
      formula = Formula(lhs=lhs("y"),
                              terms=[i(),
                                    l(["cond"]),
                                    f(["time"],by="cond"),
                                    f(["x"],by="cond"),
                                    f(["time","x"],by="cond"),
                                    fs(["time"],rf="sub")],
                              data=None,
                              file_paths=file_paths,
                              print_warn=False,
                              keep_cov=True, # Keep encoded data structure in memory
                              codebook=codebook)

      ########## preparing for ar1 model (with resets per time-series) and data type requirements ##########

      dat = pd.read_csv('https://raw.githubusercontent.com/JoKra1/mssmViz/main/data/GAMM/sim_dat.csv')

      # mssm requires that the data-type for variables used as factors is 'O'=object
      dat = dat.astype({'series': 'O',
                        'cond':'O',
                        'sub':'O',
                        'series':'O'})

      formula = Formula(lhs=lhs("y"),
                        terms=[i(),
                                 l(["cond"]),
                                 f(["time"],by="cond"),
                                 f(["x"],by="cond"),
                                 f(["time","x"],by="cond")],
                        data=dat,
                        print_warn=False,
                        series_id='series') # 'series' variable identifies individual time-series

    :param lhs: The lhs object defining the dependent variable.
    :type variable: lhs
    :param terms: A list of the terms which should be added to the model. See :py:mod:`mssm.src.python.terms` for info on which terms can be added.
    :type terms: [GammTerm]
    :param data: A pandas dataframe (with header!) of the data which should be used to estimate the model. The variable specified for ``lhs`` as well as all variables included for a ``term`` in ``terms`` need to be present in the data, otherwise the call to Formula will throw an error.
    :type data: pd.DataFrame or None
    :param series_id: A string identifying the individual experimental units. Usually a unique trial identifier. Only necessary if approximate derivative computations are to be utilized for random smooth terms or if you need to estimate an 'ar1' model for multiple time-series data.
    :type series_id: str, optional
    :param codebook: Codebook - keys should correspond to factor variable names specified in terms. Values should again be a ``dict``, with keys for each of K levels of the factor and value corresponding to an integer in {0,K}.
    :type codebook: dict or None
    :param print_warn: Whether warnings should be printed. Useful when fitting models from terminal. Defaults to True.
    :type print_warn: bool,optional
    :param keep_cov: Whether or not the internal encoding structure of all predictor variables should be created when forming :math:`\\mathbf{X}^T\\mathbf{X}` iteratively instead of forming :math:`\\mathbf{X}` directly. Can speed up estimation but increases memory footprint. Defaults to True.
    :type keep_cov: bool,optional
    :param find_nested: Whether or not to check for nested smooth terms. This only has an effect if you include at least one smooth term with more than two variables. Additionally, this check is often not necessary if you correctly use the ``te`` key-word of smooth terms and ensure that the marginals used to construct ti smooth terms have far fewer basis functions than the "main effect" univariate smooths. Thus, if you know what you're doing and you're working with large models, you might want to disable this (i.e., set to False) because this check can get quite expensive for larger models. Defaults to True.
    :type find_nested: bool,optional
    :param file_paths: A list of paths to .csv files from which :math:`\\mathbf{X}^T\\mathbf{X}` and :math:`\\mathbf{X}^T\\mathbf{y}` should be created iteratively. Setting this to a non-empty list will prevent fitting X as a whole. ``data`` should then be set to ``None``. Defaults to an empty list.
    :type file_paths: [str],optional
    :param file_loading_nc: How many cores to use to a) accumulate :math:`\\mathbf{X}` in parallel (if ``data`` is not ``None`` and ``file_paths`` is an empty list) or b) to accumulate :math:`\\mathbf{X}^T\\mathbf{X}` and :math:`\\mathbf{X}^T\\mathbf{y}` (and :math:`\\mathbf{\\eta}` during estimation) (if ``data`` is ``None`` and ``file_paths`` is a non-empty list). For case b, this should really be set to the maimum number of cores available. For a this only really speeds up accumulating :math:`\\mathbf{X}` if :math:`\\mathbf{X}` has many many columns and/or rows. Defaults to 1.
    :type file_loading_nc: int,optional
    :param file_loading_kwargs: Any key-word arguments to pass to pandas.read_csv when :math:`\\mathbf{X}^T\\mathbf{X}` and :math:`\\mathbf{X}^T\\mathbf{y}` should be created iteratively (if ``data`` is ``None`` and ``file_paths`` is a non-empty list). Defaults to ``{"header":0,"index_col":False}``.
    :type file_loading_kwargs: dict,optional
    :ivar lhs lhs: The left-hand side object of the regression formula passed to the constructor. Initialized at construction.
    :ivar [GammTerm] terms: The list of terms passed to the constructor. Initialized at construction.
    :ivar pd.DataFrame data: The dataframe passed to the constructor. Initialized at construction.
    :ivar [int] coef_per_term: A list containing the number of coefficients corresponding to each term included in ``terms``. Initialized at construction.
    :ivar [str] coef_names: A list containing a named identifier (e.g., "Intercept") for each coefficient estimated by the model. Initialized at construction.
    :ivar int n_coef: The number of coefficients estimated by the model in total. Initialized at construction.
    :ivar int unpenalized_coef: The number of un-penalized coefficients estimated by the model. Initialized at construction.
    :ivar np.ndarray or None y_flat: An array, containing all values on the dependent variable (i.e., specified by ``lhs.variable``) in order of the data-frame passed to ``data``. This variable will be initialized at construction but only if ``file_paths=None``, i.e., in case :math:`\\mathbf{X}^T\\mathbf{X}` and :math:`\\mathbf{X}^T\\mathbf{y}` are **not** created iteratively.
    :ivar np.ndarray or None cov_flat: An array, containing all (encoded, in case of categorical predictors) values on each predictor (each columns of ``cov_flat`` corresponds to a different predictor) variable included in any of the ``terms`` in order of the data-frame passed to ``data``. This variable will be initialized at construction but only if ``file_paths=None``, i.e., in case :math:`\\mathbf{X}^T\\mathbf{X}` and :math:`\\mathbf{X}^T\\mathbf{y}` are **not** created iteratively.
    :ivar np.ndarray or None NOT_NA_flat: An array, containing an indication (as bool) for each value on the dependent variable (i.e., specified by ``lhs.variable``) whether the corresponding value is not a number ("NA") or not. In order of the data-frame passed to ``data``. This variable will be initialized at construction but only if ``file_paths=None``, i.e., in case :math:`\\mathbf{X}^T\\mathbf{X}` and :math:`\\mathbf{X}^T\\mathbf{y}` are **not** created iteratively.
    """
    def __init__(self,
                 lhs:lhs,
                 terms:list[GammTerm],
                 data:pd.DataFrame,
                 series_id:str | None=None,
                 codebook:dict | None=None,
                 print_warn:bool=True,
                 keep_cov:bool = False,
                 find_nested:bool = True,
                 file_paths:list[str] = [],
                 file_loading_nc:int = 1,
                 file_loading_kwargs: dict = {"header":0,"index_col":False}) -> None:
        
        self.lhs = lhs
        self.terms = terms
        self.data = data
        self.series_id = series_id
        self.print_warn = print_warn
        self.keep_cov = keep_cov # For iterative X.T@X building, whether the encoded data should be kept or read in from file again during every iteration.
        self.file_paths = file_paths # If this will not be empty, we accumulate t(X)@X directly without forming X. Only useful if model is normal.
        self.file_loading_nc = file_loading_nc
        self.file_loading_kwargs = file_loading_kwargs
        self.factor_codings = {}
        self.coding_factors = {}
        self.factor_levels = {}
        self.var_to_cov = {}
        self.var_types = {}
        self.var_mins = {}
        self.var_maxs = {}
        self.subgroup_variables = []
        self.term_names:list[str] = []
        self.linear_terms:list[int] = []
        self.smooth_terms:list[int] = []
        self.ir_smooth_terms:list[int] = []
        self.random_terms:list[int] = []
        self.has_intercept = False
        self.has_irf = False
        self.n_irf = 0
        self.unpenalized_coef:int|None = None
        self.coef_names:list[str]|None = None
        self.n_coef:int|None = None # Number of total coefficients in formula.
        self.coef_per_term:list[int]|None = None # Number of coefficients associated with each term
        self.built_penalties = False
        self.find_nested = find_nested
        cvi = 0 # Number of variables included in some way as predictors

        # Encoding from data frame to series-level dependent values + predictor values (in cov)
        # sid holds series end indices for quick splitting.
        self.y_flat:np.ndarray|None = None
        self.cov_flat:np.ndarray|None = None
        self.NOT_NA_flat:np.ndarray|None = None
        self.y:list[np.ndarray]|None = None
        self.cov:list[np.ndarray]|None = None
        self.NOT_NA:list[np.ndarray]|None = None
        self.sid:np.ndarray|None = None

        # Discretization?
        self.discretize = {}
        
        # Perform input checks first for LHS/Dependent variable.
        if len(self.file_paths) == 0 and self.lhs.variable not in self.data.columns:
            raise IndexError(f"Column '{self.lhs.variable}' does not exist in Dataframe.")
        
        if len(self.file_paths) != 0 and self.keep_cov == False and self.find_nested:
           if print_warn:
               warnings.warn("``find_nested=True`` is not supported when iteratively building the model matrix and not keeping the encoded data (i.e., when ``keep_cov=False``). Setting ``find_nested=False``")
           self.find_nested = False

        # Now some checks on the terms - some problems might only be caught later when the 
        # penalties are built.
        for ti, term in enumerate(self.terms):
            
            # Collect term name
            self.term_names.append(term.name)

            # Term allocation.
            if isinstance(term,i):
                self.has_intercept = True
                self.linear_terms.append(ti)
                continue
            
            if isinstance(term,l):
               self.linear_terms.append(ti)

            if isinstance(term, f):
               self.smooth_terms.append(ti)

            if isinstance(term,irf):
               self.ir_smooth_terms.append(ti)
               self.n_irf += 1
            
            if isinstance(term, ri) or isinstance(term,rs):
               self.random_terms.append(ti)
            
            if isinstance(term,fs):
               if not term.approx_deriv is None:
                  self.discretize[ti] = term.approx_deriv

                  # Make sure all categorical split variables end up being encoded since
                  # they do not necessarily have to be in the formula in case of
                  # sub-groups.
                  for split_by_fac in self.discretize[ti]["split_by"]:
                     cvi = self.__encode_var(split_by_fac,'O',cvi,codebook)
            
            # All variables must exist in data
            for var in term.variables:

                if len(self.file_paths) == 0 and not var in self.data.columns:
                    raise KeyError(f"Variable '{var}' of term {ti} does not exist in dataframe.")
                
                if len(self.file_paths) == 0:
                     vartype = data[var].dtype
                else:
                     vartype = read_dtype(var,self.file_paths[0],self.file_loading_kwargs)

                # Store information for all variables once.
                cvi = self.__encode_var(var,vartype,cvi,codebook)
                
                # Smooth-term variables must all be continuous
                if isinstance(term, f) or isinstance(term, irf):
                    if not vartype in ['float64','int64']:
                        raise TypeError(f"Variable '{var}' attributed to smooth/impulse response smooth term {ti} must be numeric and is not.")
                    
                # Random intercept variable must be categorical
                if isinstance(term, ri):
                    if vartype in ['float64','int64']:
                        raise TypeError(f"Variable '{var}' attributed to random intercept term {ti} must not be numeric but is.")
                
            # by-variables must be categorical
            if isinstance(term, f) or isinstance(term, irf) or isinstance(term, rs):
                if not term.by is None or (isinstance(term, f) and not term.binary is None):
                    
                    t_by = term.by
                    if t_by is None:
                       t_by = term.binary[0]

                    if len(self.file_paths) == 0 and not t_by in self.data.columns:
                        raise KeyError(f"By-variable '{t_by}' attributed to term {ti} does not exist in dataframe.")
                    
                    if len(self.file_paths) == 0 and data[t_by].dtype in ['float64','int64']:
                        raise KeyError(f"Data-type of By-variable '{t_by}' attributed to term {ti} must not be numeric but is. E.g., Make sure the pandas dtype is 'object'.")
                    
                    if len(self.file_paths) > 0 and read_dtype(t_by,self.file_paths[0],self.file_loading_kwargs) in ['float64','int64']:
                        raise KeyError(f"Data-type of By-variable '{t_by}' attributed to term {ti} must not be numeric but is. E.g., Make sure the pandas dtype is 'object'.")
                    
                    
                    t_by_subgroup = None
                    # Handle sub-cluster for factor smooth term.
                    if isinstance(term, fs) and not term.by_subgroup is None:

                       t_by_subgroup = term.by_subgroup

                       if len(self.file_paths) == 0 and not t_by_subgroup[0] in self.data.columns:
                           raise KeyError(f"Sub-group by-variable '{t_by_subgroup}' attributed to term {ti} does not exist in dataframe.")
                        
                       if len(self.file_paths) == 0 and data[t_by_subgroup[0]].dtype in ['float64','int64']:
                           raise KeyError(f"Data-type of sub-group by-variable '{t_by_subgroup[0]}' attributed to term {ti} must not be numeric but is. E.g., Make sure the pandas dtype is 'object'.")
                        
                       if len(self.file_paths) > 0 and read_dtype(t_by_subgroup[0],self.file_paths[0],self.file_loading_kwargs) in ['float64','int64']:
                           raise KeyError(f"Data-type of sub-group by-variable '{t_by_subgroup[0]}' attributed to term {ti} must not be numeric but is. E.g., Make sure the pandas dtype is 'object'.")
                       
                       # Make sure sub-group variable is also encoded.
                       cvi = self.__encode_var(t_by_subgroup[0],'O',cvi,codebook)

                     # Store information for by variables as well.
                    cvi = self.__encode_var(t_by,'O',cvi,codebook,by_subgroup=t_by_subgroup)
                    
                    # If no error was raised we can now over-write the by variable for the factor smooth
                    if isinstance(term, fs) and  not t_by_subgroup is None:
                        term.by += ":" + t_by_subgroup[1]
                    
                    if isinstance(term, f) and not term.binary is None:
                        term.binary_level = self.factor_codings[t_by][term.binary[1]]
                
                if not term.by_cont is None: # Continuous variable to be multiplied with model matrix for this term.
                   t_by_cont = term.by_cont

                   if len(self.file_paths) == 0 and not t_by_cont in self.data.columns:
                        raise KeyError(f"By_cont-variable '{t_by_cont}' attributed to term {ti} does not exist in dataframe.")
                   
                   if len(self.file_paths) == 0:
                        vartype = data[t_by_cont].dtype
                   else:
                        vartype = read_dtype(t_by_cont,self.file_paths[0],self.file_loading_kwargs)

                   if vartype not in ['float64','int64']:
                     raise TypeError(f"Variable '{t_by_cont}' attributed to term {ti} must be numeric but is not.")
                   
                   cvi = self.__encode_var(t_by_cont,vartype,cvi,codebook)
                   
    
        if self.n_irf > 0:
           self.has_irf = True
        
        if self.has_irf and (len(self.file_paths) != 0 or self.data is None):
           raise NotImplementedError("Building X.T@X iteratively does not support Impulse Response Terms (i.e., ``irf``) in the formula.")
        
        # Encode data into columns usable by the model
        if len(self.file_paths) == 0 or self.keep_cov:
            y_flat,cov_flat,NAs_flat,y,cov,NAs,sid = self.encode_data(self.data)

            # Store encoding
            self.y_flat = y_flat
            self.cov_flat = cov_flat
            self.NOT_NA_flat = NAs_flat
            self.y = y
            self.cov = cov
            self.NOT_NA = NAs
            self.sid = sid
        
        if len(self.discretize) > 0:
           if self.series_id is None:
               raise ValueError(f"The identifier column for unique series must be provided when requesting to approximate the derivative of one or more factor smooth terms.")
    
           if len(self.file_paths) != 0 and self.keep_cov == False:
              # Need to create cov_flat (or at least figure out the correct dimensions) and sid after all
              #sid_var_cov_flat = read_cov(self.lhs.variable,self.series_id,self.file_paths,self.file_loading_nc,self.file_loading_kwargs)
              #self.cov_flat = np.zeros((len(sid_var_cov_flat),len(self.var_to_cov.keys())),dtype=int)

              #_, id = np.unique(sid_var_cov_flat,return_index=True)
              #self.sid = np.sort(id)
              raise ValueError("``Formula.keep.cov`` must be set to ``True`` when reading data from file AND approximating the derivative for factor smooths.")
           
           for sti in self.discretize.keys():
               self.__cluster_discretize(*self.__split_discretize(self.__discretize(sti),sti),sti)
           
           #if len(self.file_paths) != 0 and self.keep_cov == False:
           #   # Clean up
           #   self.cov_flat = None
           #   self.sid = None
           #   self.series_id = None

        # Absorb any constraints for model terms
        if len(self.file_paths) == 0:
            self.__absorb_constraints()
        else:
            self.__absorb_constraints2()

        # Can now check nesting
        if self.find_nested and len(self.smooth_terms) > 0:
         self.__fix_nested()
        
        # Compute (final) number of coef and coef names
        self.__get_coef_info()

        #print(self.n_coef,len(self.coef_names))
   
    def __encode_var(self,var:str,vartype:np.dtype,cvi:int,codebook:dict,by_subgroup:tuple[str,str]|None=None) -> int:
      """Internal function that does bookkeeping on variables, encoding them into the ``codebook``.

      :param var: name of variable
      :type var: str
      :param vartype: type of variable
      :type vartype: np.dtype
      :param cvi: variable index
      :type cvi: int
      :param codebook: codebook dictionary
      :type codebook: dict
      :param by_subgroup: Either None or a tuple with two strings: first corresponding to a factor, second to a level fo that factor, defaults to None
      :type by_subgroup: tuple[str,str] | None, optional
      :return: Updated cvi
      :rtype: int
      """
      # Store information for all variables once.
      if not by_subgroup is None:
         _org_var = var
         var += ":" + by_subgroup[1]
         self.subgroup_variables.append(var)

      if not var in self.var_to_cov:
         self.var_to_cov[var] = cvi

         # Assign vartype enum and calculate mins/maxs for continuous variables
         if vartype in ['float64','int64']:
            # ToDo: these can be properties of the formula.
            self.var_types[var] = VarType.NUMERIC
            if len(self.file_paths) == 0:
               self.var_mins[var] = np.min(self.data[var])
               self.var_maxs[var] = np.max(self.data[var])
            else:
               unique_var = read_unique(var,self.file_paths,self.file_loading_nc,self.file_loading_kwargs)
               self.var_mins[var] = np.min(unique_var)
               self.var_maxs[var] = np.max(unique_var)
         else:
            self.var_types[var] = VarType.FACTOR
            self.var_mins[var] = None
            self.var_maxs[var] = None

            # Code factor variables into integers for easy dummy coding
            if len(self.file_paths) == 0:
               if by_subgroup is None:
                  levels = np.unique(self.data[var])
               else:
                  levels = np.unique(self.data.loc[self.data[by_subgroup[0]] == by_subgroup[1],_org_var])
            else:
               if by_subgroup is None:
                  levels = read_unique(var,self.file_paths,self.file_loading_nc,self.file_loading_kwargs)
               else:
                  rf_fac = read_cov(self.lhs.variable,_org_var,self.file_paths,self.file_loading_nc,self.file_loading_kwargs)
                  #print(len(rf_fac))
                  sub_fac = read_cov(self.lhs.variable,by_subgroup[0],self.file_paths,self.file_loading_nc,self.file_loading_kwargs)
                  #print(len(rf_fac[sub_fac == by_subgroup[1]]))
                  levels = np.unique(rf_fac[sub_fac == by_subgroup[1]])
                  

            self.factor_codings[var] = {}
            self.coding_factors[var] = {}
            self.factor_levels[var] = levels
            
            for ci,c in enumerate(levels):
               if not codebook is None and var in codebook:
                  self.factor_codings[var][c] = codebook[var][c]
                  self.coding_factors[var][codebook[var][c]] = c
               else:
                  self.factor_codings[var][c] = ci
                  self.coding_factors[var][ci] = c

         cvi += 1

      return cvi
  
    def __get_coef_info(self) -> None:
      """Get's information about the number of coefficients from each term + the names of these coefficients. Terms also provide info here about the number of unpenalized coefficients that come with the term.
      """
      var_types = self.var_types
      factor_levels = self.factor_levels
      coding_factors = self.coding_factors

      terms = self.terms
      self.unpenalized_coef = 0
      self.n_coef = 0
      self.coef_names = []
      self.coef_per_term = np.zeros(len(terms),dtype=int)

      for lti in self.get_linear_term_idx():
         # Calculate Coef names for linear terms
         lterm = terms[lti]

         if isinstance(lterm,i):
            # Intercept
            t_total_coef,\
            t_unpenalized_coef,\
            t_coef_names = lterm.get_coef_info()
         
         else:
            # Linear effects
            t_total_coef,\
            t_unpenalized_coef,\
            t_coef_names = lterm.get_coef_info(self.has_intercept,
                                               var_types,
                                               factor_levels,
                                               coding_factors)
         self.coef_names.extend(t_coef_names)
         self.coef_per_term[lti] = t_total_coef
         self.n_coef += t_total_coef
         self.unpenalized_coef += t_unpenalized_coef
      
      for irsti in self.get_ir_smooth_term_idx():
         # Calculate Coef names for impulse response terms
         irsterm = terms[irsti]

         t_total_coef,\
         t_unpenalized_coef,\
         t_coef_names = irsterm.get_coef_info(self.has_intercept,
                                              factor_levels)
         
         self.coef_names.extend(t_coef_names)
         self.coef_per_term[irsti] = t_total_coef
         self.n_coef += t_total_coef
         self.unpenalized_coef += t_unpenalized_coef

      for sti in self.get_smooth_term_idx():
         # Calculate Coef names for smooth terms
         sterm = terms[sti]

         t_total_coef,\
         t_unpenalized_coef,\
         t_coef_names = sterm.get_coef_info(factor_levels)
         
         self.coef_names.extend(t_coef_names)
         self.coef_per_term[sti] = t_total_coef
         self.n_coef += t_total_coef
         self.unpenalized_coef += t_unpenalized_coef

      for rti in self.get_random_term_idx():
         # Calculate Coef names for random terms
         rterm = terms[rti]

         if isinstance(rterm,ri):
            t_total_coef,\
            t_unpenalized_coef,\
            t_coef_names = rterm.get_coef_info(factor_levels,coding_factors)

         elif isinstance(rterm,rs):
            t_total_coef,\
            t_unpenalized_coef,\
            t_coef_names = rterm.get_coef_info(var_types,
                                               factor_levels,
                                               coding_factors)

         self.coef_names.extend(t_coef_names)
         self.coef_per_term[rti] = t_total_coef
         self.n_coef += t_total_coef
         self.unpenalized_coef += t_unpenalized_coef
            
    def encode_data(self,data:pd.DataFrame,prediction:bool=False) -> tuple[np.ndarray|None,np.ndarray,np.ndarray|None,list[np.ndarray]|None,list[np.ndarray]|None,list[np.ndarray]|None,np.ndarray|None]:
      """
      Encodes ``data``, which needs to be a ``pd.DataFrame`` and by default (if ``prediction==False``) builds an index
      of which rows in ``data`` are NA in the column of the dependent variable described by ``self.lhs``.

      :param data: The data to encode.
      :type data: pd.DataFrame
      :param prediction: Whether or not a NA index and a column for the dependent variable should be generated.
      :type prediction: bool, optional
      :return: A tuple with 7 (optional) entries: the dependent variable described by ``self.lhs``, the encoded predictor variables as a (N,k) array (number of rows matches the number of rows of the first entry returned, the number of columns matches the number of k variables present in the formula), an indication for each row whether the dependent variable described by ``self.lhs`` is NA, like the first entry but split into a list of lists by ``self.series_id``, like the second entry but split into a list of lists by ``self.series_id``, ike the third entry but split into a list of lists by ``self.series_id``, start and end points for the splits used to split the previous three elements (identifying the start and end point of every level of ``self.series_id``).
      :rtype: (np.ndarray|None, np.ndarray, np.ndarray|None, list[np.ndarray]|None, list[np.ndarray]|None, list[np.ndarray]|None, np.ndarray|None)
      """
      # Build NA index
      if prediction:
         NAs = None
         NAs_flat = None
      else:
         if data is None:
            # read in dep var - without NA correction!
            y_flat = read_cov_no_cor(self.lhs.variable,self.file_paths,self.file_loading_nc,self.file_loading_kwargs)
            NAs_flat = np.isnan(y_flat) == False
         else:
            NAs_flat = np.isnan(data[self.lhs.variable].values) == False

      if not data is None:
         if not prediction and data.shape[0] != data[NAs_flat].shape[0] and self.print_warn:
            warnings.warn(f"{data.shape[0] - data[NAs_flat].shape[0]} {self.lhs.variable} values ({round((data.shape[0] - data[NAs_flat].shape[0]) / data.shape[0] * 100,ndigits=2)}%) are NA.")
         n_y = data.shape[0]
      else:
         n_y = len(y_flat)

      id_col = None
      if not self.series_id is None:
         if data is None:
            id_col = read_cov_no_cor(self.series_id,self.file_paths,self.file_loading_nc,self.file_loading_kwargs)
         else:
            id_col = np.array(data[self.series_id])

      var_map = self.get_var_map()
      n_var = len(var_map)
      var_keys = var_map.keys()
      var_types = self.var_types
      factor_coding = self.factor_codings
      
      # Collect every series from data frame, make sure to maintain the
      # order of the data frame.
      # Based on: https://stackoverflow.com/questions/12926898
      sid = None
      if not self.series_id is None:
         _, id = np.unique(id_col,return_index=True)
         sid = np.sort(id)

      if prediction: # For encoding new data
         y_flat = None
         y = None
      else:
         # Collect entire y column
         if not data is None:
            y_flat = np.array(data[self.lhs.variable]).reshape(-1,1)
         
         # Then split by seried id
         y = None
         NAs = None
         if not self.series_id is None:
            y = np.split(y_flat,sid[1:])

            # Also split NA index
            NAs = np.split(NAs_flat,sid[1:])

      # Now all predictor variables
      cov_flat = np.zeros((n_y,n_var),dtype=float) # Treating all predictors as floats has important implications for factors and requires special care!

      for c in var_keys:
         if data is None:
            if c in self.subgroup_variables:
               c_raw = read_cov_no_cor(c.split(":")[0],self.file_paths,self.file_loading_nc,self.file_loading_kwargs)
            else:
               c_raw = read_cov_no_cor(c,self.file_paths,self.file_loading_nc,self.file_loading_kwargs)
         else:
            if c in self.subgroup_variables:
               c_raw = np.array(data[c.split(":")[0]])
            else:
               c_raw = np.array(data[c])

         if var_types[c] == VarType.FACTOR:

            c_coding = factor_coding[c]

            # Code factor variable
            if c in self.subgroup_variables:
               # Set level to -1 which will be ignored later when building the factor smooth.
               c_code = [c_coding[cr] if cr in c_coding else -1 for cr in c_raw]
            else:
               c_code = [c_coding[cr] for cr in c_raw]

            cov_flat[:,var_map[c]] = c_code

         else:
            cov_flat[:,var_map[c]] = c_raw
      
      # Now split cov by series id as well
      cov = None
      if not self.series_id is None:
         cov = np.split(cov_flat,sid[1:],axis=0)

      return y_flat,cov_flat,NAs_flat,y,cov,NAs,sid
    
    def __discretize(self,sti:int) -> np.ndarray:
      """Internal function to discretize covariates.

      :param sti: Smooth term index pointing to smooth holding a discretization dict
      :type sti: int
      :return: np.ndarray holding discretized covariate values of the data set required by ``self.discretize[sti]``.
      :rtype: np.ndarray
      """
      dig_cov_flat = np.zeros_like(self.cov_flat)
      var_types = self.var_types
      var_map = self.get_var_map()
      
      collected = []

      for var in var_types.keys():
         # Skip variables that should be ignored all together.
         # Useful if one or more continuous variables can be split into
         # categorical factor passed along via "split_by"
         if var in self.discretize[sti]["excl"] or var_types[var] == VarType.FACTOR:
            continue

         if var not in self.discretize[sti]["no_disc"]:
            # Discretize continuous variable into k**0.5 bins, where k is the number of unique values this variable
            # took on in the training set (based on, Wood et al., 2017 "Gams for Gigadata").
            values = np.linspace(min(self.cov_flat[:,var_map[var]]),
                                 max(self.cov_flat[:,var_map[var]]),
                                 int(len(np.unique(self.cov_flat[:,var_map[var]]))**0.5))
            dig_cov_flat[:,var_map[var]] = np.digitize(self.cov_flat[:,var_map[var]],values)
            collected.append(var_map[var])

         # Also collect continuous variables that should not be discretized
         else:
            dig_cov_flat[:,var_map[var]] = self.cov_flat[:,var_map[var]]
            collected.append(var_map[var])

      return dig_cov_flat[:,collected]

    
    def __split_discretize(self,dig_cov_flat_all:np.ndarray,sti:int) -> tuple[list[np.ndarray],list[np.ndarray]]:
      """Internal function to split dscretized covariate objects.

      :param dig_cov_flat_all: np.ndarray holding discretized covariate values of the data set required by ``self.discretize[sti]``.
      :type dig_cov_flat_all: np.ndarray
      :param sti: Smooth term index pointing to smooth holding a discretization dict
      :type sti: int
      :return: tuple holding two lists of np.ndarray. First list has split discretized covariate objects per series, second has series id
      :rtype: tuple[list[np.ndarray],list[np.ndarray]]
      """
      var_map = self.get_var_map()
      factor_codings = self.factor_codings

      # Create seried id column in ascending order:
      id_col = np.zeros(dig_cov_flat_all.shape[0],dtype=int)
      id_splits = np.split(id_col,self.sid[1:])
      id_splits = [split + i for i,split in enumerate(id_splits)]
      id_col = np.concatenate(id_splits)

      if not self.terms[sti].by_subgroup is None:
         # Adjust for fact that this factor smooth is fitted for separate level of sub-group.
         sub_group_fact = self.terms[sti].by_subgroup[0]
         sub_group_lvl = self.terms[sti].by_subgroup[1]

         # Build index vector corresponding only to series of sub-group level
         sub_lvl_idx = self.cov_flat[:,var_map[sub_group_fact]] == factor_codings[sub_group_fact][sub_group_lvl]

         # Just take what belongs to the sub-group from the disceretized matrix
         dig_cov_flat_all = dig_cov_flat_all[sub_lvl_idx,:]

         # For id col a bit more work is necessary..
         # First take again what belongs to sub-group. Now the problem is that series will no longer go from 0-S. So we
         # have to reset the values in id_col to start from zero and then increment towards the number of series included
         # in this sub-group.
         id_col = id_col[sub_lvl_idx]
         # Split based on indices
         _, id = np.unique(id_col,return_index=True)
         sub_sid = np.sort(id)

         # Reset index
         id_splits = np.split(id_col,sub_sid[1:])
         for i,_ in enumerate(id_splits):
            id_splits[i][:] = i

         # And merge again
         id_col = np.concatenate(id_splits)

      if len(self.discretize[sti]["split_by"]) == 0:
         # Don't actually split
         return [dig_cov_flat_all],[id_col]

      # Now split dig_cov_flat_all per level of combination of all factor variables used for splitting, again correcting for any
      # potential sub-grouping
      if self.terms[sti].by_subgroup is None:
         unq_fact_comb,unq_fact_comb_memb = np.unique(self.cov_flat[:,[var_map[fact] for fact in self.discretize[sti]["split_by"]]],
                                                      axis=0,return_inverse=True)
      else:
         unq_fact_comb,unq_fact_comb_memb = np.unique(self.cov_flat[sub_lvl_idx,[var_map[fact] for fact in self.discretize[sti]["split_by"]]],
                                                      axis=0,return_inverse=True)
      # Split series id column per level
      fact_series = [id_col[unq_fact_comb_memb == fact] for fact in range(len(unq_fact_comb))]

      # Split dig_cov_flat_all per level
      dig_cov_flats = [dig_cov_flat_all[unq_fact_comb_memb == fact,:] for fact in range(len(unq_fact_comb))]

      return dig_cov_flats, fact_series
    
    def __cluster_discretize(self,dig_cov_flats:list[np.ndarray], fact_series:list[np.ndarray], sti:int) -> None:
      """Internal function that clusters on the discretized covariate objects.

      Takes input from :func:`__split_discretize`. Sets ``self.discretize[sti]["clust_series"]`` and ``self.discretize[sti]["clust_weights"]``.

      :param dig_cov_flats: split discretized covariate objects per series
      :type dig_cov_flats: list[np.ndarray]
      :param fact_series: series id array per series
      :type fact_series: list[np.ndarray]
      :param sti: Smooth term index pointing to smooth holding a discretization dict
      :type sti: int
      """
      best_series = None
      best_weights = None
      best_error = None

      iterator = range(self.discretize[sti]["restarts"])
      seed = self.discretize[sti]["seed"]

      if self.print_warn:
         iterator = tqdm(iterator,desc="Clustering",leave=True)

      for rep in iterator:
         clust_max_series = []
         weights = []
         error = 0

         for dig_cov_flat,fact_s in zip(dig_cov_flats,fact_series):

            # Create a simple index vector for each unique series in this factor split.
            _,fact_s_idx = np.unique(fact_s,return_index=True)
            sid_idx = fact_s[fact_s_idx]

            # Now compute the number of unique rows across the discretized matrix.
            dig_cov_flat_unq = np.unique(dig_cov_flat,axis=0)

            # Also compute, for each column, the inverse - telling us for each row in the discretized data
            # to which unique value on the corresponding **variable** it belongs.
            dig_cov_flat_unq_memb = np.zeros_like(dig_cov_flat,dtype=int)

            dig_cov_unq_counts = []

            for vari in range(dig_cov_flat.shape[1]):
               dig_var_flat_unq,dig_var_flat_unq_memb = np.unique(dig_cov_flat[:,vari],return_inverse=True)

               if vari > 0:
                  dig_var_flat_unq_memb += dig_cov_unq_counts[-1]

               dig_cov_unq_counts.append(len(dig_var_flat_unq))
               dig_cov_flat_unq_memb[:,vari] = dig_var_flat_unq_memb[:]

            # Now we prepare the cluster structure:
            # Every series now gets represented by a row vector with sum(dig_cov_unq_counts) entries
            # Every column in these vectors corresponds to a unique value of an individual discretized
            # (or not) co-variate. The first dig_cov_unq_counts[0] columns correspond to the unique values
            # of covariate 1, the next dig_cov_unq_counts[1] columns correspond to covariate 2, and so on.
            # Each column gets assigned the number of times the corresponding unique value exists for the series to
            # which the vector belongs.
            clust = np.zeros((len(sid_idx),sum(dig_cov_unq_counts)))

            # To compute this we just split the inverse per unique series
            # s_split_unq_memb is a list of 2d arrays, each array corresponding to a series
            # with the number of rows matching the number of observations collected for that series.
            # the number of columns matches the number of collected (and potentially discretized) covariates.
            s_split_unq_memb = np.split(dig_cov_flat_unq_memb,fact_s_idx[1:])

            # Now we flatten that 2d array and collect the unique values and how often they occur. This gives us
            # for every series the indices in the cluster structure that will be non-zero and the value we need to
            # store in each non-zero cell.
            s_split_unq_cnts = [np.unique(s_dig,return_counts=True) for s_dig in s_split_unq_memb]

            # Then loop over each series and add the counts of the corresponding rows to the cluster structure
            for sidx,(udr,cnts) in enumerate(s_split_unq_cnts):
               clust[sidx,udr] += cnts

            # Use heuristic to determine the number of clusters also used to discretize individual covariates
            # Then cluster - for estimation this only has to do once before starting the actual fitting routine
            clust_centroids,clust_lab = scp.cluster.vq.kmeans2(clust,int((dig_cov_flat_unq.shape[1]*dig_cov_flat_unq.shape[0])**0.5),minit='++',seed=seed)

            if seed is not None:
               seed += 1

            # Compute clustering loss, according to scipy docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans2.html
            # Simply pick the cluster set out of all repetitions that minimizes the loss
            for k in range(clust_centroids.shape[0]):
               error += np.power(clust[clust_lab == k,:] - clust_centroids[k,:],2).sum()

            # Find ordering of series ids based on assigned cluster labels
            arg_sort_clust = np.argsort(clust_lab)

            # Sort cluster labels in ascending order for easy split of cluster ordered
            # ids into cluster groups
            sort_clust = clust_lab[arg_sort_clust]
            idx_clust_sort = np.arange(0,len(clust_lab),dtype=int)[arg_sort_clust]

            # Find cluster split points
            _, cid = np.unique(sort_clust,return_index=True)
            csid = np.sort(cid)

            # Now collect all series ids of a particular cluster into a separate array
            idx_grouped = np.split(idx_clust_sort,csid[1:])

            # Find the (ideally, this is done heuristically after all) most complex series in every cluster:
            # Compute the number of unique rows for each series in a cluster, then pick the maximum.
            # Compute complexity weights for all series in the cluster relative to that maximum.
            # These act as a a proxy of how similar each series is to the cluster prototype/maximum.
            
            for k,clu in enumerate(idx_grouped):
               clu_sums = np.sum(clust[clu,:],axis=1)
               clust_max_series.append(sid_idx[clu[np.argmax(clu_sums)]])
               weights.append(clu_sums/np.max(clu_sums))

               #
               #clust_distances = np.power(clust[clu,:] - clust_centroids[k,:],2).sum(axis=1) + 1
            
               #clust_max_series.append(sid_idx[clu[np.argmin(clust_distances)]])
               #clust_rel_distances = clu_sums/clu_sums[np.argmin(clust_distances)]
               #weights.append(clust_rel_distances/np.max(clust_rel_distances))

         if (rep == 0) or (error < best_error):
            best_series = np.array(clust_max_series)
            best_weights = weights
            best_error = error

      self.discretize[sti]["clust_series"] = best_series
      self.discretize[sti]["clust_weights"] = best_weights

    def __absorb_constraints2(self) -> None:
      """Internal function to absorb identifiability constraints when data is read from file.

      Wood (2017) has an overview over identifiability constraints and different reparameterizations for smooth terms.

      References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :raises ValueError: When re-parameterization is requested for identifiable terms.
      :raises NotImplementedError: When a constraint type other than ``QR`` is requested.
      """
      
      for sti in self.get_smooth_term_idx():

         sterm = self.terms[sti]
         vars = sterm.variables

         if not sterm.is_identifiable:
            if sterm.should_rp > 0:
               # Reparameterization of marginals was requested - but no identifiability constraints necessary.
               for vi in range(len(vars)):
                  
                  if len(vars) > 1:
                     id_nk = sterm.nk[vi]
                  else:
                     id_nk = sterm.nk
                  
                  var_cov_flat = read_cov(self.lhs.variable,vars[vi],self.file_paths,self.file_loading_nc,self.file_loading_kwargs)

                  matrix_term_v = sterm.basis(var_cov_flat,
                                              None,id_nk,min_c=self.var_mins[vars[vi]],
                                              max_c=self.var_maxs[vars[vi]], **sterm.basis_kwargs)

                  sterm.absorb_repara(vi,scp.sparse.csc_array(matrix_term_v),var_cov_flat)

            continue

         term_constraint = sterm.Z
         sterm.Z = []

         if sterm.should_rp > 0:
            raise ValueError("Re-parameterizing identifiable terms is currently not supported when files are loaded in to build X.T@X incrementally.")
         
         if term_constraint == ConstType.QR:

            with mp.Pool(processes=10) as pool:
               C = pool.starmap(_compute_constraint_single_MP,zip(repeat(sterm),repeat(vars),
                                                                 repeat(self.lhs.variable),
                                                                 self.file_paths,
                                                                 repeat(self.var_mins),
                                                                 repeat(self.var_maxs),
                                                                 repeat(self.file_loading_kwargs)))

            if sterm.te or len(vars) == 1:
               C = np.sum(np.array(C),axis=0)
               Q,_ = scp.linalg.qr(C,pivoting=False,mode='full')
               sterm.Z.append(Constraint(Q[:,1:],ConstType.QR))
            else:
               for vi in range(len(vars)):
                  CVI = np.sum(np.array([cvi[vi] for cvi in C]),axis=0)
                  Q,_ = scp.linalg.qr(CVI,pivoting=False,mode='full')
                  sterm.Z.append(Constraint(Q[:,1:],ConstType.QR))
         else:
            raise NotImplementedError("Only QR constraints are currently supported when files are loaded in to build X.T@X incrementally.")

    def __absorb_constraints(self) -> None:
      """Internal function to absorb identifiability constraints and apply selected reparameterizations to individual smooth terms.

      Wood (2017) has an overview over identifiability constraints and different reparameterizations for smooth terms.

      References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
      """
      var_map = self.get_var_map()

      for sti in self.get_smooth_term_idx():

         sterm = self.terms[sti]
         vars = sterm.variables

         if not sterm.is_identifiable:
            if sterm.should_rp > 0:
               # Reparameterization of marginals was requested - but no identifiability constraints are needed.
               for vi in range(len(vars)):
                  
                  if len(vars) > 1:
                     id_nk = sterm.nk[vi]
                  else:
                     id_nk = sterm.nk
                  
                  var_cov_flat = self.cov_flat[self.NOT_NA_flat,var_map[vars[vi]]]

                  matrix_term_v = sterm.basis(var_cov_flat,
                                              None,id_nk,min_c=self.var_mins[vars[vi]],
                                              max_c=self.var_maxs[vars[vi]], **sterm.basis_kwargs)

                  sterm.absorb_repara(vi,scp.sparse.csc_array(matrix_term_v),var_cov_flat)

            continue

         term_constraint = sterm.Z
         sterm.Z = []
         matrix_term = None # for Te basis
         for vi in range(len(vars)):
            # If a smooth term needs to be identifiable I act as if you would have asked for nk+1!
            # so that the identifiable term is of the dimension expected.
            
            if len(vars) > 1:
               id_nk = sterm.nk[vi]
            else:
               id_nk = sterm.nk
            
            if sterm.te == False:
               id_nk += 1

            var_cov_flat = self.cov_flat[self.NOT_NA_flat,var_map[vars[vi]]]

            matrix_term_v = sterm.basis(var_cov_flat,
                                      None,id_nk,min_c=self.var_mins[vars[vi]],
                                      max_c=self.var_maxs[vars[vi]], **sterm.basis_kwargs)

            if sterm.te == False:

               if term_constraint == ConstType.QR:
                  # Wood (2017) 5.4.1 Identifiability constraints via QR. ToDo: Replace with mean subtraction method.
                  C = np.sum(matrix_term_v,axis=0).reshape(-1,1)
                  Q,_ = scp.linalg.qr(C,pivoting=False,mode='full')
                  sterm.Z.append(Constraint(Q[:,1:],ConstType.QR))
               elif term_constraint == ConstType.DROP:
                  sterm.Z.append(Constraint(int(matrix_term_v.shape[1]/2),ConstType.DROP))
               elif term_constraint == ConstType.DIFF:
                  sterm.Z.append(Constraint(int(matrix_term_v.shape[1]/2),ConstType.DIFF))

            if sterm.should_rp > 0:
               # Reparameterization of marginals was requested.

               if sterm.Z[vi].type == ConstType.QR:
                  XPb = matrix_term_v @ sterm.Z[vi].Z
               elif sterm.Z[vi].type == ConstType.DROP:
                  XPb = np.delete(matrix_term_v,sterm.Z[vi].Z,axis=1)
               elif sterm.Z[vi].type == ConstType.DIFF:
                  # Applies difference re-coding for sum-to-zero coefficients.
                  # Based on smoothCon in mgcv(2017). See constraints.py
                  # for more details.
                  XPb = np.diff(np.concatenate((matrix_term_v[:,sterm.Z[vi].Z:matrix_term_v.shape[1]],matrix_term_v[:,:sterm.Z[vi].Z]),axis=1))
                  XPb = np.concatenate((XPb[:,XPb.shape[1]-sterm.Z[vi].Z:],XPb[:,:XPb.shape[1]-sterm.Z[vi].Z]),axis=1)

               sterm.absorb_repara(vi,scp.sparse.csc_array(XPb),self.cov_flat[self.NOT_NA_flat,var_map[vars[vi]]])

            if sterm.te:
               if vi == 0:
                  matrix_term = matrix_term_v
               else:
                  matrix_term = TP_basis_calc(matrix_term,matrix_term_v)
         
         # Now deal with te basis
         if sterm.te:
            if term_constraint == ConstType.QR:
               C = np.sum(matrix_term,axis=0).reshape(-1,1)
               Q,_ = scp.linalg.qr(C,pivoting=False,mode='full')
               sterm.Z.append(Constraint(Q[:,1:],ConstType.QR))
            elif term_constraint == ConstType.DROP:
               sterm.Z.append(Constraint(int(matrix_term.shape[1]/2),ConstType.DROP))
            elif term_constraint == ConstType.DIFF:
               sterm.Z.append(Constraint(int(matrix_term.shape[1]/2),ConstType.DIFF))
   
    def __fix_nested(self) -> None:
      """Internal function that identifies nested terms in the Formula, based on section 5.6.3 of Wood (2017) and what is implemented in the ``gam.side`` function in ``mgcv``.

      Automatically figures out which coefficients have to be dropped from individual smooth terms and assigns the corresponding indices to ``term.drop_coef``.
      These are then automatically removed from the model matrix and penalty built for the particular term (i.e., when calling ``term.build_matrix`` and ``term.build_penalty``
      for example via :func:`build_sparse_matrix_from_formula` and :func:`build_penalties`.)

      References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
         - mgcv gam.side function, see: https://github.com/cran/mgcv/blob/fb7e8e718377513e78ba6c6bf7e60757fc6a32a9/R/mgcv.r#L563
      """
      sti = self.get_smooth_term_idx()
      sti = [ti for ti in sti if isinstance(self.terms[ti],fs) == False]


      term_var = [self.terms[ti].variables for ti in sti]
      n_term_var = [len(tvar) for tvar in term_var]
      max_var = np.max(n_term_var)

      # Check if we have terms with multiple variables
      if max_var == 1:
         return

      # Have smooth terms and potentially some nesting. Now need an extended list of smooth terms,
      # their variable names (taking into account by, binary, and by_cont keywords), and the number of variables
      # to check more carefully for nesting and to then also do something about it.
      term_var = [[] for _ in sti]
      unlist_var = []
      for tii, ti in enumerate(sti):
         sterm = self.terms[ti]
         vars = sterm.variables
         
         for var in vars:
            ext_vars = []

            if sterm.by is not None:
                  
                  by_levels = self.factor_levels[sterm.by]
                  ext_vars.extend([var + "_" + str(lvl) for lvl in by_levels])
            else:
                  ext_vars.append(var)
            
            if sterm.binary is not None:
                  ext_vars = [ext_var + "_" + sterm.binary[0] + "_"  + sterm.binary[1] for ext_var in ext_vars]
            
            if sterm.by_cont is not None:
                  ext_vars = [ext_var + "_" + sterm.by_cont for ext_var in ext_vars]

            term_var[tii].extend(ext_vars)
            unlist_var.extend(ext_vars)

      # Can now safely check for duplicates
      if len(unlist_var) == len(np.unique(unlist_var)):
         return

      # Identify nested depenndencies as discussed in section 5.6.3 of Wood (2017)

      # First build model matrices for terms, making sure to create multiple ones per level of by variable
      # and to account for binary/by_cont keywords.
      Xs = [[] for _ in sti]
      n_term_by = []
      var_map = self.get_var_map()
      cov_flat = self.cov_flat
      for tii, ti in enumerate(sti):
         sterm = self.terms[ti]
         vars = sterm.variables
         for vi in range(len(vars)):
                        
            if len(vars) > 1:
                  id_nk = sterm.nk[vi]
            else:
                  id_nk = sterm.nk
            
            if sterm.is_identifiable and sterm.te == False:
                  id_nk += 1
            
            var_cov_flat = cov_flat[self.NOT_NA_flat,var_map[vars[vi]]]

            matrix_term_v = sterm.basis(var_cov_flat,
                                          None,id_nk,min_c=self.var_mins[vars[vi]],
                                          max_c=self.var_maxs[vars[vi]], **sterm.basis_kwargs)

            if sterm.te == False:
                  # Absorb identifiability constraints into marginals
                  if sterm.is_identifiable:
                     if sterm.Z[vi].type == ConstType.QR:
                        matrix_term_v = matrix_term_v @ sterm.Z[vi].Z
                     elif sterm.Z[vi].type == ConstType.DROP:
                        matrix_term_v = np.delete(matrix_term_v,sterm.Z[vi].Z,axis=1)
                     elif sterm.Z[vi].type == ConstType.DIFF:
                        # Applies difference re-coding for sum-to-zero coefficients.
                        # Based on smoothCon in mgcv(2017). See constraints.py
                        # for more details.
                        matrix_term_v = np.diff(np.concatenate((matrix_term_v[:,sterm.Z[vi].Z:matrix_term_v.shape[1]],matrix_term_v[:,:sterm.Z[vi].Z]),axis=1))
                        matrix_term_v = np.concatenate((matrix_term_v[:,matrix_term_v.shape[1]-sterm.Z[vi].Z:],matrix_term_v[:,:matrix_term_v.shape[1]-sterm.Z[vi].Z]),axis=1)
                  
                  # Absorb reparam into marginal
                  if sterm.should_rp > 0:
                     matrix_term_v = matrix_term_v @ sterm.RP[vi].C

            # Prepare te/ti
            if vi == 0:
                  matrix_term = matrix_term_v
            else:
                  matrix_term = TP_basis_calc(matrix_term,matrix_term_v)

         # Now can handle te - ti and univariate are ready at this point
         if sterm.te:
            if sterm.is_identifiable:
               if sterm.Z[0].type == ConstType.QR:
                  matrix_term = matrix_term @ sterm.Z[0].Z

               elif sterm.Z[0].type == ConstType.DROP:
                  matrix_term = np.delete(matrix_term,sterm.Z[0].Z,axis=1)

               elif sterm.Z[0].type == ConstType.DIFF:
                  matrix_term = np.diff(np.concatenate((matrix_term[:,sterm.Z[0].Z:matrix_term.shape[1]],matrix_term[:,:sterm.Z[0].Z]),axis=1))
                  matrix_term = np.concatenate((matrix_term[:,matrix_term.shape[1]-sterm.Z[0].Z:],matrix_term[:,:matrix_term.shape[1]-sterm.Z[0].Z]),axis=1)

            # Currently no further reparam for te implemented
         

         # Now deal with by, binary, or by_cont
         # Multiply each row of model matrix by value in by_cont
         if sterm.by_cont is not None:
            by_cont_cov = cov_flat[self.NOT_NA_flat,var_map[sterm.by_cont]]
            matrix_term *= by_cont_cov.reshape(-1,1)
         
         # Handle optional by keyword
         n_by = 1
         if sterm.by is not None:
            by_levels = self.factor_levels[sterm.by]
            by_cov = cov_flat[self.NOT_NA_flat,var_map[sterm.by]]
            n_by = len(by_levels)
            
            # Split by cov
            for by_level in range(len(by_levels)):
                  by_cidx = (by_cov == by_level).astype(int)
                  Xs[tii].append(matrix_term * by_cidx.reshape(-1,1))
         
         # Handle optional binary keyword
         elif sterm.binary is not None:
            by_cov = cov_flat[self.NOT_NA_flat,var_map[sterm.binary[0]]]
            by_cidx = (by_cov == sterm.binary_level).astype(int)

            Xs[tii].append(matrix_term * by_cidx.reshape(-1,1))

         else:
            Xs[tii].append(matrix_term)
            
         n_term_by.append(n_by)

      # At this point we have all the model matrices and can work on actually identifying
      # nested terms.
      osize = 2
      drop = {}
      while osize <= max_var:

         otis = [ti for ti in range(len(sti)) if n_term_var[ti] == osize]
         
         for otii,oti in enumerate(otis):

            for by_idx in range(n_term_by[oti]):
                  
                  # Get variables of this term
                  otis_var = np.array(term_var[oti])
                  otis_n_var = n_term_var[oti]
                  otis_n_by = n_term_by[oti]

                  if otis_n_by > 1:
                     otis_var = otis_var[np.arange(by_idx,otis_n_var*otis_n_by,otis_n_by)]

                  # Now first find all terms with less than osize variables...
                  itis = []
                  X1 = []
                  for ti in range(len(sti)):
                     if n_term_var[ti] <= (osize-1) and np.any([t_var in otis_var for t_var in term_var[ti]]):
                        itis.append(ti)

                        X1.append(Xs[ti][by_idx])

                  # and then all previous terms with exactly osize variables
                  if otii > 0 and len(otis) > 1:
                     for ti in otis[:otii]:
                        if np.any([t_var in otis_var for t_var in term_var[ti]]):
                              itis.append(ti)

                              X1.append(Xs[ti][by_idx])
                  
                  # Build combined model matrix
                  if len(X1) == 0:
                     continue

                  X1 = np.concatenate(X1,axis=1)

                  if self.has_intercept:
                     X1 = np.concatenate([np.ones(X1.shape[0]).reshape(-1,1),X1],axis=1)

                  # And figure out identifiability issues
                  nid,rank = eigen_solvers.id_dependencies(X1,Xs[oti][by_idx],np.pow(np.finfo(float).eps,0.5))
                  if rank < Xs[oti][by_idx].shape[1]:
                     
                     # Delete dropped coef from Xs
                     Xs[oti][by_idx] = np.delete(Xs[oti][by_idx],nid[rank:],axis=1)

                     # and collect them for later model matrix building
                     nid += by_idx*len(nid)
                     
                     if not sti[oti] in drop:
                        drop[sti[oti]] = list(np.sort(nid[rank:]))
                     else:
                        drop[sti[oti]].extend(np.sort(nid[rank:]))

         osize += 1

      for ti in sti:
         if ti in drop:
            self.terms[ti].drop_coef = drop[ti]

    
    #### Getters ####

    def get_lhs(self) -> lhs:
       """Get a copy of the ``lhs`` specified for this formula."""
       return copy.deepcopy(self.lhs)
    
    def get_terms(self) -> list[GammTerm]:
       """Get a copy of the ``terms`` specified for this formula."""
       return copy.deepcopy(self.terms)
    
    def get_data(self) -> pd.DataFrame:
       """Get a copy of the ``data`` specified for this formula."""
       return copy.deepcopy(self.data)

    def get_factor_codings(self) -> dict:
        """Get a copy of the factor coding dictionary. Keys are factor variables in the data, values are dictionaries, where the keys correspond to the levels (str) of the factor and the values to their encoded levels (int)."""
        return copy.deepcopy(self.factor_codings)
    
    def get_coding_factors(self) -> dict:
        """Get a copy of the factor coding dictionary. Keys are factor variables in the data, values are dictionaries, where the keys correspond to the encoded levels (int) of the factor and the values to their levels (str)."""
        return copy.deepcopy(self.coding_factors)
    
    def get_var_map(self) -> dict:
        """Get a copy of the var map dictionary. Keys are variables in the data, values their column index in the encoded predictor matrix returned by ``self.encode_data``."""
        return copy.deepcopy(self.var_to_cov)
    
    def get_factor_levels(self) -> dict:
       """Get a copy of the factor levels dictionary. Keys are factor variables in the data, values are np.arrays holding the unique levels (as str) of the corresponding factor."""
       return copy.deepcopy(self.factor_levels)
    
    def get_var_types(self) -> dict:
       """Get a copy of the var types dictionary. Keys are variables in the data, values are either ``VarType.NUMERIC`` for continuous variables or ``VarType.FACTOR`` for categorical variables."""
       return copy.deepcopy(self.var_types)
    
    def get_var_mins(self) -> dict:
       """Get a copy of the var mins dictionary. Keys are variables in the data, values are either the minimum value the variable takes on in ``self.data`` for continuous variables or ``None`` for categorical variables."""
       return copy.deepcopy(self.var_mins)
    
    def get_var_maxs(self) -> dict:
       """Get a copy of the var maxs dictionary. Keys are variables in the data, values are either the maximum value the variable takes on in ``self.data`` for continuous variables or ``None`` for categorical variables."""
       return copy.deepcopy(self.var_maxs)
    
    def get_var_mins_maxs(self) -> tuple[dict,dict]:
       """Get a tuple containing copies of both the mins and maxs directory. See ``self.get_var_mins`` and ``self.get_var_maxs``."""
       return (copy.deepcopy(self.var_mins),copy.deepcopy(self.var_maxs))
    
    def get_linear_term_idx(self) -> list[int]:
       """Get a copy of the list of indices that identify linear terms in ``self.terms``."""
       return(copy.deepcopy(self.linear_terms))
    
    def get_smooth_term_idx(self) -> list[int]:
       """Get a copy of the list of indices that identify smooth terms in ``self.terms``."""
       return(copy.deepcopy(self.smooth_terms))
    
    def get_ir_smooth_term_idx(self) -> list[int]:
       """Get a copy of the list of indices that identify impulse response terms in ``self.terms``."""
       return(copy.deepcopy(self.ir_smooth_terms))
    
    def get_random_term_idx(self) -> list[int]:
       """Get a copy of the list of indices that identify random terms in ``self.terms``."""
       return(copy.deepcopy(self.random_terms))
    
    def get_n_coef(self) -> int:
       """Get the number of coefficients that are implied by the formula."""
       return self.n_coef
    
    def get_depvar(self) -> np.ndarray:
       """Get a copy of the encoded dependent variable (defined via ``self.lhs``)."""
       return copy.deepcopy(self.y_flat)
    
    def get_notNA(self) -> np.ndarray:
       """Get a copy of the encoded 'not a NA' vector for the dependent variable (defined via ``self.lhs``)."""
       return copy.deepcopy(self.NOT_NA_flat)
    
    def get_has_intercept(self) -> bool:
       """Does this formula include an intercept or not."""
       return self.has_intercept
    
    def has_ir_terms(self) -> bool:
       """Does this formula include impulse response terms or not."""
       return self.has_irf
    
    def get_term_names(self) -> list[str]:
       """Returns a copy of the list with the names of the terms specified for this formula."""
       return copy.deepcopy(self.term_names)
   
    def get_subgroup_variables(self) -> list:
       """Returns a copy of sub-group variables for factor smooths."""
       return copy.deepcopy(self.subgroup_variables)
    
def build_penalties(formula) -> list[LambdaTerm]:
      """Function to build all penalty matrices required by a :class:`Formula`.

      The function is called whenever it is needed, but the example below shows you how to use it in case you want to extract the penalties directly.

      Examples::

         from mssm.models import *
         from mssmViz.sim import *
         from mssm.src.python.formula import build_penalties

         # Get some data and formula
         Binomdat = sim3(10000,0.1,family=Binomial(),seed=20)
         formula = Formula(lhs("y"),[i(),f(["x0"]),f(["x1"]),f(["x2"]),f(["x3"])],data=Binomdat)

         # Now extract the penalties
         penalties = build_penalties(formula)

         print(penalties)

      :param formula: A Formula
      :type formula: Formula
      :raises KeyError: If an un-penalized irf term is included in the formula after penalized terms.
      :raises KeyError:  If an un-penalized smooth term is included in the formula after penalized terms.
      :raises ValueError: If no start index has been defined by the formula. For testing only.
      :return: A list of all penalties (encoded as :class:`LambdaTerm`) required by the formula
      :rtype: list[LambdaTerm]
      """

      col_S = formula.n_coef
      factor_levels = formula.factor_levels
      terms = formula.terms
      penalties = []
      start_idx = formula.unpenalized_coef

      if start_idx is None:
         ValueError("Penalty start index is ill-defined.")

      cur_pen_idx = start_idx
      prev_pen_idx = start_idx

      for irsti in formula.get_ir_smooth_term_idx():

         irsterm = terms[irsti]
         vars = irsterm.variables

         # Calculate nCoef 
         n_coef = irsterm.nk

         if len(vars) > 1:
            n_coef = np.prod(irsterm.nk)
         
         by_levels = None
         if irsterm.by is not None:
            by_levels = factor_levels[irsterm.by]
         
         if not irsterm.is_penalized:
            if len(penalties) == 0:

               added_not_penalized = n_coef
               if irsterm.by is not None:
                  added_not_penalized *= len(by_levels)
               start_idx += added_not_penalized
               cur_pen_idx = start_idx

               if formula.print_warn:
                  warnings.warn(f"Impulse response smooth {irsti} is not penalized. Smoothing terms should generally be penalized.")

            else:
               raise KeyError(f"Impulse response smooth {irsti} is not penalized and placed in the formula after penalized terms. Unpenalized terms should be moved to the beginning of the formula, ideally behind any linear terms.")
         
         else:

            for penid in range(len(irsterm.penalty)):
               
               # Smooth terms can have multiple penalties.
               # In that case the starting index of every subsequent
               # penalty needs to be reset.
               if penid > 0:
                  cur_pen_idx = prev_pen_idx

               penalties,cur_pen_idx = irsterm.build_penalty(irsti,penalties,cur_pen_idx,
                                                             penid,factor_levels,col_S)
         
         # Keep track of previous penalty starting index
         prev_pen_idx = cur_pen_idx
                        
      for sti in formula.get_smooth_term_idx():

         sterm = terms[sti]
         vars = sterm.variables

         # Calculate nCoef
         if len(vars) > 1:
            n_coef = np.prod(sterm.nk)
         else:
            n_coef = sterm.nk

         by_levels = None
         if sterm.by is not None:
            by_levels = factor_levels[sterm.by]

         if not sterm.is_penalized:
            if len(penalties) == 0:

               added_not_penalized = n_coef
               if sterm.by is not None:
                  added_not_penalized *= len(by_levels)

               start_idx += added_not_penalized

               if formula.print_warn:
                  warnings.warn(f"Smooth {sti} is not penalized. Smoothing terms should generally be penalized.")

            else:
               raise KeyError(f"Smooth {sti} is not penalized and placed in the formula after penalized terms. Unpenalized terms should be moved to the beginning of the formula, ideally behind any linear terms.")
         
         else:
            S_j_TP_last = None
            TP_last_n = 0
            for penid in range(len(sterm.penalty)):
            
               # Smooth terms can have multiple penalties.
               # In that case the starting index of every subsequent
               # penalty needs to be reset.
               if penid > 0:
                  cur_pen_idx = prev_pen_idx
               
               prev_n_pen = len(penalties)
               penalties,cur_pen_idx = sterm.build_penalty(sti,penalties,cur_pen_idx,
                                                           penid,factor_levels,col_S)

               
               # Add necessary info for derivative approx. for factor smooth penalties
               if isinstance(sterm,fs):
                  if not sterm.approx_deriv is None:
                     penalties[-1].clust_series = formula.discretize[sti]["clust_series"]
                     penalties[-1].clust_weights = formula.discretize[sti]["clust_weights"]
               
               # Optionally include a Null-space penalty - an extra penalty on the
               # function space not regularized by the penalty we just created:
               if sterm.has_null_penalty:

                  is_reparam1 = False
                  n_pen = len(penalties)

                  # Below we distinguish between TP smooths of multiple variables and single variable smooths.
                  # For TP smooths Marra & Wood (2011) suggest to first sum over the penalties for individual
                  # variables and then computing the null-space for that summed penalty.
                  if len(vars) > 1:

                     # For multivariate smooths, we need to chek for the possibility of separate penalties
                     # for different bylevels. These might not be identical - in case different coefficients might have been dropped
                     # for each level. Thus we need to keep track of a list of sums; one sum per by-level
                     # and compute the null-space penalty separately per penalty in those lists.
                     if (sterm.drop_coef) is not None and (sterm.by is not None) and (sterm.id is None):
                        
                        if penid == 0:
                           S_j_TP_last = [penalties[-(pidx+1)].S_J.toarray() for pidx in range(len(by_levels)-1,-1,-1)]
                        else:
                           new_TP_last = [penalties[-(pidx+1)].S_J.toarray() for pidx in range(len(by_levels)-1,-1,-1)]
                           S_j_TP_last = [S_j_TP_last[pidx] + new_TP_last[pidx] for pidx in range(len(new_TP_last))]

                     # Here all potential penalties belonging to different bylevels are equivalent. Or we have no bylevels. OR we have an id.
                     # Any of those means that penalties from different variables are equivalent and we can just maintain a single penalty sum.
                     else:
                        S_j_last = penalties[-1].S_J.toarray()

                        if penid == 0:
                           S_j_TP_last =  [S_j_last]
                        else:
                           S_j_TP_last[0] +=  S_j_last
                     
                     # Keep track of penalties added for tensor smooths, note that (n_pen - prev_n_pen) will be > 1 for by level smooths
                     TP_last_n += (n_pen - prev_n_pen)

                     if penid < (len(sterm.penalty) - 1): # Not done yet - more penalties to build
                        continue

                     # This holds the final list of penalty sums
                     S_j_last = S_j_TP_last
                  
                     # In the end, update the number of new penalties based on the number of variables
                     # involed for TPs
                     added_pen = int(TP_last_n / len(vars))
                     
                  else:
                     # check for Demmler & Reinsch reparameterization for univariate smooths
                     S_j_last = [penalties[-1].S_J.toarray()]
                     is_reparam1 = penalties[-1].type == PenType.REPARAM1 # Only for univariate smooths should this be true.
                     added_pen = n_pen - prev_n_pen

                  # Number of coefficients depends on which type S_j_last is: list with multiple elements or not?
                  if len(S_j_last) == 1:
                     idk = [S_j_last[0].shape[1]]
                     last_pen_rep = penalties[-1].rep_sj # Might have repetition in case of id keyword
                  else:
                     idk = [SJ.shape[1] for SJ in S_j_last]
                     last_pen_rep = 1 # No repetition for independent by-level penalties!
                  
                  # Now compute the desired Null-space penalties

                  NULL_DIMs = [] # Rank of Nullspace-penalty
                  NULL_idks = [] # Dimension of penalty
                  NULL_S = [] # Nullspace penalty in [dat,row,col]
                  NULL_D = [] # Root of Nullspace penalty in [dat, row, col]
                  
                  if is_reparam1 == False:
                     # Based on: Marra & Wood (2011) and: https://rdrr.io/cran/mgcv/man/gam.selection.html
                     # and: https://eric-pedersen.github.io/mgcv-esa-workshop/slides/03-model-selection.pdf

                     for sji,S_ji_last in enumerate(S_j_last):
                        s, U =scp.linalg.eigh(S_ji_last)
                        DNULL = U[:,s <= 1e-7]
                        NULL_DIM = DNULL.shape[1] # Null-space dimension
                        DNULL = DNULL.reshape(S_ji_last.shape[1],-1)

                        SNULL = DNULL @ DNULL.T

                        SNULL = scp.sparse.csc_array(SNULL)
                        DNULL = scp.sparse.csc_array(DNULL)
                        

                        # Data in S and D is in canonical format, for competability this is translated to data, rows, columns
                        pen_data,pen_rows,pen_cols = translate_sparse(SNULL)
                        chol_data,chol_rows,chol_cols = translate_sparse(DNULL)

                        NULL_S.append([pen_data,pen_rows,pen_cols])
                        NULL_D.append([chol_data,chol_rows,chol_cols])
                        NULL_DIMs.append(NULL_DIM)
                        NULL_idks.append(idk[sji])

                  else:
                     # Under the Demmler & Reinsch (1975) re-parameterization the last S.shape[1] - S.rank cols/rows correspond to functions in the kernel.
                     # Hence we can simply place identity penalties on those to shrink them to zero. In this form we can also readily
                     # have separate penalties on different null-space functions! This is how mgcv implements factor smooths as well.

                     # Can only happen to univariate smooth, so indexing S_j_last is safe here.

                     NULL_DIM = S_j_last[0].shape[1] - sterm.RP[0].rank # Null-space dimension
                     
                     for nci in range(S_j_last[0].shape[1] - NULL_DIM, S_j_last[0].shape[1]):
                        
                        NULL_S.append([[1],[nci],[nci]])
                        NULL_D.append([[1],[nci],[nci]])
                        NULL_DIMs.append(1)
                        NULL_idks.append(idk[0])
                     
                  # Now iterate over Null-space penalties
                  for nri in range(len(NULL_DIMs)):
                     
                     # Get current Nullspace penalty
                     nri_rank = NULL_DIMs[nri]
                     pen_data,pen_rows,pen_cols = NULL_S[nri]
                     chol_data,chol_rows,chol_cols = NULL_D[nri]
                     idk = NULL_idks[nri]

                     cur_pen_idx = prev_pen_idx

                     lTerm = LambdaTerm(start_index=cur_pen_idx,
                                          type = PenType.NULL,
                                          term = sti)
                     
                     # Embed first penalty
                     lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,idk,cur_pen_idx)
                     lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,idk,cur_pen_idx)
                     lTerm.S_J = embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J,idk)
                     lTerm.rank = nri_rank

                     # Update prev_pen_idx for split by penalties and append current one
                     if len(S_j_last) > 1:
                        penalties.append(lTerm)
                        prev_pen_idx = cur_pen_idx
                     
                     # Handle equal penalties per by-level with id keyword
                     else:
                        # Single penalty added - but could involve by keyword
                        if (added_pen) == 1:
                           
                           # Handle any By-keyword
                           if last_pen_rep > 1:
                              pen_iter = last_pen_rep - 1
                              #for _ in range(pen_iter):
                              #   lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,idk,cur_pen_idx)
                              #   lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,idk,cur_pen_idx)
                              
                              chol_rep = np.tile(chol_data,pen_iter)
                              idx_row_rep = np.repeat(np.arange(pen_iter),len(chol_rows))*idk
                              idx_col_rep = np.repeat(np.arange(pen_iter),len(chol_cols))*idk
                              chol_rep_row = np.tile(chol_rows,pen_iter) + idx_row_rep
                              chol_rep_cols = np.tile(chol_cols,pen_iter) + idx_col_rep
                              
                              lTerm.D_J_emb, _ = embed_in_S_sparse(chol_rep,chol_rep_row,chol_rep_cols,lTerm.D_J_emb,col_S,idk*pen_iter,cur_pen_idx)

                              pen_rep = np.tile(pen_data,pen_iter)
                              idx_row_rep = np.repeat(np.arange(pen_iter),len(pen_rows))*idk
                              idx_col_rep = np.repeat(np.arange(pen_iter),len(pen_cols))*idk
                              pen_rep_row = np.tile(pen_rows,pen_iter) + idx_row_rep
                              pren_rep_cols = np.tile(pen_cols,pen_iter) + idx_col_rep

                              lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_rep,pen_rep_row,pren_rep_cols,lTerm.S_J_emb,col_S,idk*pen_iter,cur_pen_idx)
                              
                              lTerm.rep_sj = last_pen_rep

                           # In any case, term can be appended here.
                           lTerm.rank = nri_rank*last_pen_rep
                           penalties.append(lTerm)
                        else:
                           # Independent penalties via by
                           # Append penalty for first level
                           penalties.append(lTerm)

                           # And add the penalties again for the remaining levels as separate terms
                           for _ in range((added_pen) - 1):
                              lTerm = LambdaTerm(start_index=cur_pen_idx,
                                             type = PenType.NULL,
                                             term = sti)
                        
                              # Embed penalties
                              lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,idk,cur_pen_idx)
                              lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,idk,cur_pen_idx)
                              lTerm.S_J = embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J,idk)
                              lTerm.rank = nri_rank
                              penalties.append(lTerm)

         # Keep track of previous penalty starting index
         prev_pen_idx = cur_pen_idx

      for rti in formula.get_random_term_idx():
         # Build penalties for random terms
         rterm = terms[rti]

         penalties,cur_pen_idx = rterm.build_penalty(rti,penalties,cur_pen_idx,
                                                     factor_levels,col_S)
            
      if cur_pen_idx != col_S:
         raise ValueError(f"Penalty dimension {cur_pen_idx},{cur_pen_idx} does not match outer model matrix dimension {col_S}")

      formula.built_penalties = True
      return penalties

def build_sparse_matrix_from_formula(terms:list[GammTerm],has_intercept:bool,
                                     ltx:list[int],irstx:list[int],stx:list[int],rtx:list[int],
                                     var_types:dict,var_map:dict,
                                     var_mins:dict,var_maxs:dict,
                                     factor_levels:dict,cov_flat:np.ndarray,
                                     cov:np.ndarray|None,pool:mp.pool.Pool|None=None,use_only:list[int]|None=None,
                                     tol:float=0) -> scp.sparse.csc_array:
   """Build model matrix from formula properties.

   This function is used internally to construct model matrices from :class:`Formula` objects. For greater convenience see the
   :func:`build_model_matrix` function.

   **Important**, make sure to only ever call this when ``formula.built_penalties==True`` - see the :func:`build_model_matrix` function description.

   :param terms: List of terms of a :class:`Formula` 
   :type terms: list[GammTerm]
   :param has_intercept: Indicator of whether the Formula has an intercept or not
   :type has_intercept: bool
   :param ltx: Linear term indices
   :type ltx: list[int]
   :param irstx: Impulse response function term indices
   :type irstx: list[int]
   :param stx: Smooth term indices
   :type stx: list[int]
   :param rtx: Random term indices
   :type rtx: list[int]
   :param var_types: Dictionary holding variable types
   :type var_types: dict
   :param var_map: Dictionary mapping variable names to column indices in the encoded data
   :type var_map: dict
   :param var_mins: Dictionary with variable minimums 
   :type var_mins: dict
   :param var_maxs: Dictionary with variable maximums 
   :type var_maxs: dict
   :param factor_levels: Dictionary with levels associated with each factor
   :type factor_levels: dict
   :param cov_flat: Encoded data
   :type cov_flat: np.ndarray
   :param cov: Encoded data split by levels of the factor in ``Formula.series_id``
   :type cov: np.ndarray | None, optional
   :param pool: An instance of a multiprocessing pool, defaults to None
   :type pool: mp.pool.Pool | None, optional
   :param use_only: A list of indices corresponding to which terms should actually be built. If ``None``, then all terms are build. Terms not built are set to zero columns, defaults to None
   :type use_only: list[int] | None, optional
   :param tol: Optional tolerance. Absolute values in the model matrix smaller than this are set to actual zeroes, defaults to 0
   :type tol: float, optional
   :return: The model matrix implied by a :class:`Formula`  and ``cov_flat``.
   :rtype: scp.sparse.csc_array
   """
   n_y = cov_flat.shape[0]
   elements = []
   rows = []
   cols = []
   ridx = np.array([ri for ri in range(n_y)]) #ToDo: dtype=int?

   ci = 0
   for lti in ltx:
      # Build matrix for linear terms
      lterm = terms[lti]

      if isinstance(lterm,i):
         # Intercept
         new_elements,\
         new_rows,\
         new_cols,\
         new_ci = lterm.build_matrix(ci,lti,ridx,use_only)
         
         elements.extend(new_elements)
         rows.extend(new_rows)
         cols.extend(new_cols)
         ci += new_ci
      
      else:
         # Linear term
         new_elements,\
         new_rows,\
         new_cols,\
         new_ci = lterm.build_matrix(has_intercept,ci,lti,var_map,
                                     var_types,factor_levels,ridx,
                                     cov_flat,use_only)
         
         elements.extend(new_elements)
         rows.extend(new_rows)
         cols.extend(new_cols)
         ci += new_ci
   
   for irsti in irstx:
      # Impulse response terms need to be calculate for every series individually - costly
      irsterm = terms[irsti]
      
      new_elements,\
      new_rows,\
      new_cols,\
      new_ci = irsterm.build_matrix(ci,irsti,var_map,
                                    var_mins,var_maxs,
                                    factor_levels,ridx,cov,
                                    use_only,pool,tol)
      
      elements.extend(new_elements)
      rows.extend(new_rows)
      cols.extend(new_cols)
      ci += new_ci

   for sti in stx:

      sterm = terms[sti]

      new_elements,\
      new_rows,\
      new_cols,\
      new_ci = sterm.build_matrix(ci,sti,var_map,
                                  var_mins,var_maxs,
                                  factor_levels,ridx,cov_flat,
                                  use_only,tol)
      
      elements.extend(new_elements)
      rows.extend(new_rows)
      cols.extend(new_cols)
      ci += new_ci
      
   for rti in rtx:
      rterm = terms[rti]

      if isinstance(rterm,ri):
         new_elements,\
         new_rows,\
         new_cols,\
         new_ci = rterm.build_matrix(ci,rti,var_map,factor_levels,
                                     ridx,cov_flat,use_only)
         
      elif isinstance(rterm,rs):
         new_elements,\
         new_rows,\
         new_cols,\
         new_ci = rterm.build_matrix(ci,rti,var_map,var_types,
                                     factor_levels,ridx,cov_flat,use_only)
         
      elements.extend(new_elements)
      rows.extend(new_rows)
      cols.extend(new_cols)
      ci += new_ci

   mat = scp.sparse.csc_array((elements,(rows,cols)),shape=(n_y,ci))

   return mat

def build_model_matrix(formula:Formula,pool:mp.pool.Pool|None=None,use_only:list[int]|None=None,tol:float=0) -> scp.sparse.csc_array:
   """Function to build the model matrix implied by ``formula``.

   **Important:** A small selection of smooth terms, requires that the penalty matrices are built at least once before the model matrix can be build.
   For this reason, you generally must call ``build_penalties(formula)`` before calling ``build_model_matrix(formula)`` (interally, mssm checks whether
   ``formula.built_penalties==True``.). See the example below.

   Examples::

      from mssm.models import *
      from mssmViz.sim import *
      from mssmViz.plot import *
      import matplotlib.pyplot as plt

      from mssm.src.python.formula import build_penalties,build_model_matrix

      # Get some data and formula
      Binomdat = sim3(10000,0.1,family=Binomial(),seed=20)
      formula = Formula(lhs("y"),[i(),f(["x0"]),f(["x1"]),f(["x2"]),f(["x3"])],data=Binomdat)

      # First extract the penalties
      penalties = build_penalties(formula)

      # Then the model matrix:
      X = build_model_matrix(formula)

   :param formula: A Formula
   :type formula: Formula
   :param pool: An instance of a multiprocessing pool, defaults to None
   :type pool: mp.pool.Pool | None, optional
   :param use_only: A list of indices corresponding to which terms should actually be built. If ``None``, then all terms are build. Terms not built are set to zero columns, defaults to None
   :type use_only: list[int] | None, optional
   :param tol: Optional tolerance. Absolute values in the model matrix smaller than this are set to actual zeroes, defaults to 0
   :type tol: float, optional
   :raises ValueError: If ``formula.built_penalties == False`` - i.e., it is required that ``build_penalties(formula)`` was called before calling ``build_model_matrix(formula)``.
   :raises NotImplementedError: If the ``formula`` was set up to read data from file, rather than from a pd.Dataframe.
   :return: The model matrix implied by a :class:`Formula`  and ``cov_flat``.
   :rtype: scp.sparse.csc_array
   """

   if formula.built_penalties == False:
      raise ValueError("You must call ``build_penalties(formula)`` once before calling ``build_model_matrix(formula)``.")

   if len(formula.file_paths) != 0:
      raise NotImplementedError("Cannot return the model-matrix if data was read directly from file, rather than provided as a pd.DataFrame.")

   # Get all the objects from formula required
   terms = formula.terms
   has_intercept = formula.has_intercept
   ltx = formula.get_linear_term_idx()
   irstx = formula.get_ir_smooth_term_idx()
   stx = formula.get_smooth_term_idx()
   rtx = formula.get_random_term_idx()
   var_types = formula.get_var_types()
   var_map = formula.get_var_map()
   var_mins = formula.get_var_mins()
   var_maxs = formula.get_var_maxs()
   factor_levels = formula.get_factor_levels()

   cov_flat = formula.cov_flat[formula.NOT_NA_flat]
            
   if len(irstx) > 0:
         cov_flat = formula.cov_flat # Need to drop NA rows **after** building!
         cov = formula.cov
   else:
         cov = None

   mat = build_sparse_matrix_from_formula(terms,has_intercept,
                                          ltx,irstx,stx,rtx,var_types,var_map,
                                          var_mins,var_maxs,factor_levels,
                                          cov_flat,cov,pool,use_only,tol)
   
   if len(irstx) > 0:
      mat = mat[formula.NOT_NA_flat,:]

   return mat