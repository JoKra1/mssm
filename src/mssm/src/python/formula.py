import warnings
import copy
from collections.abc import Callable
import numpy as np
import scipy as scp
import pandas as pd
from enum import Enum
from .smooths import TP_basis_calc
from .terms import GammTerm,i,l,f,irf,ri,rs
from .penalties import PenType,id_dist_pen,diff_pen,TP_pen,LambdaTerm,translate_sparse

class VarType(Enum):
    NUMERIC = 1
    FACTOR = 2

class lhs():
    """
    The Left-hand side of a regression equation.

    Parameters:

    :param variable: The dependent variable. Can point to continuous and categorical variables.
    :type variable: str
    :param f: A function that will be applied to the ``variable`` before fitting. For example: np.log().
    By default no function is applied to the ``variable``.
    :type f: Callable, optional
    """
    def __init__(self,variable:str,f:Callable=None) -> None:
        self.variable = variable
        self.f=f


def get_coef_info_linear(has_intercept,lterm,var_types,coding_factors,factor_levels):
    unpenalized_coef = 0
    coef_names = []
    total_coef = 0
    coef_per_term = []
    # Main effects
    if len(lterm.variables) == 1:
        var = lterm.variables[0]
        if var_types[var] == VarType.FACTOR:
            
            fl_start = 0

            if has_intercept: # Dummy coding when intercept is added.
                fl_start = 1

            for fl in range(fl_start,len(factor_levels[var])):
                coef_names.append(f"{var}_{coding_factors[var][fl]}")
                unpenalized_coef += 1
                total_coef += 1

            coef_per_term.append(len(factor_levels[var]) - fl_start)

        else: # Continuous predictor
            coef_names.append(f"{var}")
            unpenalized_coef += 1
            total_coef += 1
            coef_per_term.append(1)

    else: # Interactions
        inter_coef_names = []

        for var in lterm.variables:
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

        coef_per_term.append(len(inter_coef_names))
    return total_coef,unpenalized_coef,coef_names,coef_per_term

def get_coef_info_smooth(has_scale_split,n_j,sterm,factor_levels):
    coef_names = []
    total_coef = 0
    coef_per_term = []

    vars = sterm.variables
    # Calculate Coef names
    if len(vars) > 1:
        term_n_coef = np.prod(sterm.nk)
    else:
        term_n_coef = sterm.nk

    # Total coef accounting for potential by keywords.
    n_coef = term_n_coef

    # var label
    var_label = vars[0]
    if len(vars) > 1:
        var_label = "_".join(vars)
   
    if sterm.binary is not None:
        var_label += sterm.binary[0]

    if sterm.by is not None:
        by_levels = factor_levels[sterm.by]
        n_coef *= len(by_levels)

        if sterm.by_latent is not False and has_scale_split is False:
            n_coef *= n_j
            for by_state in range(n_j):
                for by_level in by_levels:
                    coef_names.extend([f"f_{var_label}_{ink}_{by_level}_{by_state}" for ink in range(term_n_coef)])
        else:
            for by_level in by_levels:
                coef_names.extend([f"f_{var_label}_{ink}_{by_level}" for ink in range(term_n_coef)])
         
    else:
        if sterm.by_latent is not False and has_scale_split is False:
            for by_state in range(n_j):
                coef_names.extend([f"f_{var_label}_{ink}_{by_state}" for ink in range(term_n_coef)])
        else:
            coef_names.extend([f"f_{var_label}_{ink}" for ink in range(term_n_coef)])
         
    total_coef += n_coef
    coef_per_term.append(n_coef)
    return total_coef,coef_names,coef_per_term

def build_smooth_penalties(has_scale_split,n_j,penalties,cur_pen_idx,
                           pen,penid,sterm,
                           vars,by_levels,n_coef,col_S):
    # We again have to deal with potential identifiable constraints!
    # Then we again act as if n_k was n_k+1 for difference penalties

    # penid % len(vars) because it will just go from 0-(len(vars)-1) and
    # reset if penid >= len(vars) which might happen in case of multiple penalties on
    # every tp basis

    if len(vars) > 1:
        id_k = sterm.nk[penid % len(vars)]
    else:
        id_k = n_coef

    pen_kwargs = sterm.pen_kwargs[penid]
    
    # Determine penalty generator
    if pen == PenType.DIFFERENCE:
        pen_generator = diff_pen
        if sterm.is_identifiable:
            id_k += 1
            pen_kwargs["Z"] = sterm.Z[penid % len(vars)]

    else:
        pen_generator = id_dist_pen

    # Again get penalty elements used by this term.
    pen_data,pen_rows,pen_cols,chol_data,chol_rows,chol_cols = pen_generator(id_k,**pen_kwargs)

    # Create lambda term
    lTerm = LambdaTerm(start_index=cur_pen_idx,
                       type = pen)

    # For tensor product smooths we first have to recalculate:
    # pen_data,pen_rows,pen_cols,chol_data,chol_rows,chol_cols via TP_pen()
    # Then they can just be embedded via the calls below.

    if len(vars) > 1:
        pen_data,\
        pen_rows,\
        pen_cols,\
        chol_data,\
        chol_rows,\
        chol_cols = TP_pen(scp.sparse.csc_array((pen_data,(pen_rows,pen_cols)),shape=(pen_cols[-1]+1,pen_cols[-1]+1)),
                           scp.sparse.csc_array((chol_data,(chol_rows,chol_cols)),shape=(pen_cols[-1]+1,pen_cols[-1]+1)),
                           penid % len(vars),sterm.nk)

    # Embed first penalty - if the term has a by-keyword more are added below.
    lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,cur_pen_idx)
    lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,cur_pen_idx)
    lTerm.S_J = embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J)
        
    if sterm.by is not None:
        
        if sterm.id is not None:

            pen_iter = len(by_levels) - 1

            if sterm.by_latent is not False and has_scale_split is False:
                pen_iter = (len(by_levels)*n_j)-1

            for _ in range(pen_iter):
                lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,cur_pen_idx)
                lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,cur_pen_idx)

            # For pinv calculation during model fitting.
            lTerm.rep_sj = pen_iter + 1
            penalties.append(lTerm)

        else:
            # In case all levels get their own smoothing penalty - append first lterm then create new ones for
            # remaining levels.
            penalties.append(lTerm)

            pen_iter = len(by_levels) - 1

            if sterm.by_latent is not False and has_scale_split is False:
                pen_iter = (len(by_levels) * n_j)-1

            for _ in range(pen_iter):

                # Create lambda term
                lTerm = LambdaTerm(start_index=cur_pen_idx,
                                type = pen)

                # Embed penalties
                lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,cur_pen_idx)
                lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,cur_pen_idx)
                lTerm.S_J = embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J)
                penalties.append(lTerm)

    else:
        if sterm.by_latent is not False and has_scale_split is False:
            # Handle by latent split - all latent levels get unique id
            penalties.append(lTerm)

            for _ in range(n_j-1):
                # Create lambda term
                lTerm = LambdaTerm(start_index=cur_pen_idx,
                                type = pen)

                # Embed penalties
                lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,cur_pen_idx)
                lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,cur_pen_idx)
                lTerm.S_J = embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J)
                penalties.append(lTerm)
        else:
            penalties.append(lTerm)

    return penalties,cur_pen_idx

def build_irf_penalties(penalties,cur_pen_idx,
                        pen,penid,irsterm,by_levels,n_coef,col_S):
    # Determine penalty generator
    if pen == PenType.DIFFERENCE:
        pen_generator = diff_pen
    else:
        pen_generator = id_dist_pen

    # Get non-zero elements and indices for the penalty used by this term.
    pen_data,pen_rows,pen_cols,chol_data,chol_rows,chol_cols = pen_generator(n_coef,**irsterm.pen_kwargs[penid])

    # Create lambda term
    lTerm = LambdaTerm(start_index=cur_pen_idx,
                        type = pen)

    # Embed first penalty - if the term has a by-keyword more are added below.
    lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,cur_pen_idx)
    lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,cur_pen_idx)
    lTerm.S_J = embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J)
        
    if irsterm.by is not None:
        if irsterm.id is not None:

            for _ in range(len(by_levels)-1):
                lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,cur_pen_idx)
                lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,cur_pen_idx)

            # For pinv calculation during model fitting.
            lTerm.rep_sj = len(by_levels)
            penalties.append(lTerm)
        else:
            # In case all levels get their own smoothing penalty - append first lterm then create new ones for
            # remaining levels.
            penalties.append(lTerm)

            for _ in range(len(by_levels)-1):
                # Create lambda term
                lTerm = LambdaTerm(start_index=cur_pen_idx,
                                type = pen)

                # Embed penalties
                lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,cur_pen_idx)
                lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,cur_pen_idx)
                lTerm.S_J = embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J)
                penalties.append(lTerm)
    else:
        penalties.append(lTerm)
    
    return penalties,cur_pen_idx

class Formula():
    """
    The formula of a regression equation.

    Parameters:

    :param lhs: The lhs object defining the dependent variable.
    :type variable: lhs
    :param terms: A list of the terms which should be added to the model. See ``mssm.src.python.terms`` for info on which terms can be added.
    :type terms: list[GammTerm]
    :param data: A pandas dataframe (with header!) of the data which should be used to estimate the model. The variable specified for ``lhs`` as
    well as all variables included for a ``term`` in ``terms`` need to be present in the data, otherwise the call to Formula will throw an error.
    :type data: pd.DataFrame
    :param series_id: A tring identifying the individual experimental units. Usually a unique trial identifier. Can only be ignored if a
   ``mssm.models.GAMM`` is to be estimated.
    :type series_id: str, optional
    :param split_scale: Whether or not a separate Gamm (including sseparate scale parameters) should be estimated per latent state. Only relevant
    if a ``mssm.models.sMsGAMM`` is to be estimated.
    :type split_scale: bool, optional
    :param n_j: Number of latent states to estimate. Only relevant if a ``mssm.models.sMsGAMM`` is to be estimated.
    :type n_j: int, optional
    """
    def __init__(self,
                 lhs:lhs,
                 terms:list[GammTerm],
                 data:pd.DataFrame,
                 series_id:str or None=None,
                 split_scale:bool=False,
                 n_j:int=3) -> None:
        
        self.__lhs = lhs
        self.__terms = terms
        self.__data = data
        self.series_id = series_id
        self.__split_scale = split_scale # Separate scale parameters per state, if true then formula counts for individual state.
        self.__n_j = n_j # Number of latent states to estimate - not for irf terms but for f terms!
        if self.__split_scale:
           warnings.warn("split_scale==True! All terms will be estimted per latent stage, independent of terms' by_latent status.")
        self.__factor_codings = {}
        self.__coding_factors = {}
        self.__factor_levels = {}
        self.__var_to_cov = {}
        self.__var_types = {}
        self.__var_mins = {}
        self.__var_maxs = {}
        self.__term_names = []
        self.__linear_terms = []
        self.__smooth_terms = []
        self.__ir_smooth_terms = []
        self.__random_terms = []
        self.__has_intercept = False
        self.__has_irf = False
        self.__has_by_latent = False
        self.__n_irf = 0
        self.unpenalized_coef = None
        self.coef_names = None
        self.n_coef = None # Number of total coefficients in formula.
        self.ordered_coef_per_term = None # Number of coefficients associated with each term - order: linear terms, irf terms, f terms, random terms
        cvi = 0 # Number of variables included in some way as predictors

        # Encoding from data frame to series-level dependent values + predictor values (in cov)
        # sid holds series end indices for quick splitting.
        self.y_flat = None
        self.cov_flat = None
        self.NOT_NA_flat = None
        self.y = None
        self.cov = None
        self.NOT_NA = None
        self.sid = None
        # Penalties
        self.penalties = None
        
        # Perform input checks first for LHS/Dependent variable.
        if self.__lhs.variable not in self.__data.columns:
            raise IndexError(f"Column '{self.__lhs.variable}' does not exist in Dataframe.")

        # Now some checks on the terms - some problems might only be caught later when the 
        # penalties are built.
        for ti, term in enumerate(self.__terms):
            
            # Collect term name
            self.__term_names.append(term.name)

            # Term allocation.
            if isinstance(term,i):
                self.__has_intercept = True
                self.__linear_terms.append(ti)
                continue
            
            if isinstance(term,l):
               self.__linear_terms.append(ti)

            if isinstance(term, f):
               self.__smooth_terms.append(ti)

            if isinstance(term,irf):
               if len(term.variables) > 1:
                  raise NotImplementedError("Multiple variables for impulse response terms have not been implemented yet.")
               
               self.__ir_smooth_terms.append(ti)
               self.__n_irf += 1
            
            if isinstance(term, ri) or isinstance(term,rs):
               self.__random_terms.append(ti)

            if not isinstance(term,irf):
               if term.by_latent:
                  self.__has_by_latent = True
            
            # All variables must exist in data
            for var in term.variables:

                if not var in self.__data.columns:
                    raise KeyError(f"Variable '{var}' of term {ti} does not exist in dataframe.")
                
                vartype = data[var].dtype

                # Store information for all variables once.
                if not var in self.__var_to_cov:
                    self.__var_to_cov[var] = cvi

                    # Assign vartype enum and calculate mins/maxs for continuous variables
                    if vartype in ['float64','int64']:
                        # ToDo: these can be properties of the formula.
                        self.__var_types[var] = VarType.NUMERIC
                        self.__var_mins[var] = np.min(self.__data[var])
                        self.__var_maxs[var] = np.max(self.__data[var])
                    else:
                        self.__var_types[var] = VarType.FACTOR
                        self.__var_mins[var] = None
                        self.__var_maxs[var] = None

                        # Code factor variables into integers for example for easy dummy coding
                        levels = np.unique(self.__data[var])

                        self.__factor_codings[var] = {}
                        self.__coding_factors[var] = {}
                        self.__factor_levels[var] = levels
                        
                        for ci,c in enumerate(levels):
                           self.__factor_codings[var][c] = ci
                           self.__coding_factors[var][ci] = c

                    cvi += 1
                
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

                    if not t_by in self.__data.columns:
                        raise KeyError(f"By-variable '{t_by}' attributed to term {ti} does not exist in dataframe.")
                    
                    if data[t_by].dtype in ['float64','int64']:
                        raise KeyError(f"Data-type of By-variable '{t_by}' attributed to term {ti} must not be numeric but is. E.g., Make sure the pandas dtype is 'object'.")
                    
                     # Store information for by variables as well.
                    if not t_by in self.__var_to_cov:
                        self.__var_to_cov[t_by] = cvi

                        # Assign vartype enum
                        self.__var_types[t_by] = VarType.FACTOR
                        self.__var_mins[t_by] = None
                        self.__var_maxs[t_by] = None

                        # Code factor variables into integers for example for easy dummy coding
                        levels = np.unique(self.__data[t_by])

                        self.__factor_codings[t_by] = {}
                        self.__coding_factors[t_by] = {}
                        self.__factor_levels[t_by] = levels
                           
                        for ci,c in enumerate(levels):
                              self.__factor_codings[t_by][c] = ci
                              self.__coding_factors[t_by][ci] = c

                        cvi += 1

                    if isinstance(term, f) and not term.binary is None:
                        term.binary_level = self.__factor_codings[t_by][term.binary[1]]
            
        if self.__n_irf > 0:
           self.__has_irf = True

        if self.__has_irf and self.__split_scale:
           raise ValueError("Formula includes an impulse response term. split_scale must be set to False!")
        
        if self.__has_irf and self.__has_by_latent:
           raise NotImplementedError("Formula includes an impulse response term. Having regular smooth terms differ by latent stages is currently not supported.")
        
        # Compute number of coef and coef names
        self.__get_coef_info()
        # Encode data into columns usable by the model
        y_flat,cov_flat,NAs_flat,y,cov,NAs,sid = self.encode_data(self.__data)
        # Store encoding
        self.y_flat = y_flat
        self.cov_flat = cov_flat
        self.NOT_NA_flat = NAs_flat
        self.y = y
        self.cov = cov
        self.NOT_NA = NAs
        self.sid = sid
        # Absorb any constraints for model terms
        self.__absorb_constraints()

        #print(self.n_coef,len(self.coef_names))
  
    def __get_coef_info(self):
      var_types = self.get_var_types()
      factor_levels = self.get_factor_levels()
      coding_factors = self.get_coding_factors()

      terms = self.__terms
      self.unpenalized_coef = 0
      self.n_coef = 0
      self.coef_names = []
      self.ordered_coef_per_term = []

      for lti in self.get_linear_term_idx():
         lterm = terms[lti]

         if isinstance(lterm,i):
            self.coef_names.append("Intercept")
            self.unpenalized_coef += 1
            self.n_coef += 1
            self.ordered_coef_per_term.append(1)
         
         else:
            # Linear effects
            t_total_coef,\
            t_unpenalized_coef,\
            t_coef_names,\
            t_coef_per_term = get_coef_info_linear(self.has_intercept(),
                                                   lterm,var_types,
                                                   coding_factors,
                                                   factor_levels)
            self.coef_names.extend(t_coef_names)
            self.ordered_coef_per_term.extend(t_coef_per_term)
            self.n_coef += t_total_coef
            self.unpenalized_coef += t_unpenalized_coef
      
      for irsti in self.get_ir_smooth_term_idx():
         # Calculate Coef names for impulse response terms
         irsterm = terms[irsti]
         var = irsterm.variables[0]
         n_coef = irsterm.nk

         if irsterm.by is not None:
            by_levels = factor_levels[irsterm.by]
            n_coef *= len(by_levels)

            for by_level in by_levels:
               self.coef_names.extend([f"irf_{irsterm.event}_{ink}_{by_level}" for ink in range(irsterm.nk)])
         
         else:
            self.coef_names.extend([f"irf_{irsterm.event}_{ink}" for ink in range(irsterm.nk)])
         
         self.n_coef += n_coef
         self.ordered_coef_per_term.append(n_coef)

      for sti in self.get_smooth_term_idx():

         sterm = terms[sti]
         s_total_coef,\
         s_coef_names,\
         s_coef_per_term = get_coef_info_smooth(self.has_scale_split(),
                                                self.__n_j,sterm,
                                                factor_levels)
         self.coef_names.extend(s_coef_names)
         self.ordered_coef_per_term.extend(s_coef_per_term)
         self.n_coef += s_total_coef

      for rti in self.get_random_term_idx():
         rterm = terms[rti]
         vars = rterm.variables

         if isinstance(rterm,ri):
            by_code_factors = coding_factors[vars[0]]

            for fl in range(len(factor_levels[vars[0]])):
               self.coef_names.append(f"ri_{vars[0]}_{by_code_factors[fl]}")
               self.n_coef += 1

            self.ordered_coef_per_term.append(len(factor_levels[vars[0]]))

         elif isinstance(rterm,rs):
            t_total_coef,\
            _,\
            t_coef_names,\
            _ = get_coef_info_linear(False,
                                     rterm,var_types,
                                     coding_factors,
                                     factor_levels)

            rterm.var_coef = t_total_coef # We need t_total_coef penalties for this term later.
            by_code_factors = coding_factors[rterm.by]
            by_code_levels = factor_levels[rterm.by]
            
            rf_coef_names = []
            for cname in t_coef_names:
               rf_coef_names.extend([f"{cname}_{by_code_factors[fl]}" for fl in range(len(by_code_levels))])
            
            t_ncoef = len(rf_coef_names)
            self.coef_names.extend(rf_coef_names)
            self.ordered_coef_per_term.append(t_ncoef)
            self.n_coef += t_ncoef
            
               
    
    def encode_data(self,data,prediction=False):
      """
      Encodes ``data``, which needs to be a ``pd.DataFrame`` and by default (if ``prediction==False``) builds an index
      of which rows in ``data`` are NA in the column of the dependent variable described by ``self.lhs``.

      Parameters:

      :param data: The data to encode.
      :type data: pd.DataFrame
      :param prediction: Whether or not a NA index and a column for the dependent variable should be generated.
      :type prediction: bool, optional

      Returns:
      :return: A tuple with 7 entries: a ``np.array`` of the dependent variable described by ``self.__lhs`` or ``None``, a ``np.array`` with as many columns
      as there are predictor variables specified in ``self.__terms``, holding the encoded predictor variables (number of rows matches the number of rows of the first entry returned),
      either a ``np.array`` indicating for each row whether the dependent variable described by ``self.__lhs`` is NA or ``None``, either like the first entry but split into a list of lists by ``self.series_id`` or ``None``,
      either like the second entry but split into a list of lists by ``self.series_id`` or ``None``, either like the third entry but split into a list of lists by ``self.series_id`` or ``None``, either a
      ``np.array`` indicating the start and end point for the splits used to split the previous three elements (identifying the start and end point of every level of ``self.series_id``) or ``None``.
      :rtype: tuple
      """
      # Build NA index
      if prediction:
         NAs = None
         NAs_flat = None
      else:
         NAs_flat = np.isnan(data[self.get_lhs().variable]) == False

      if not prediction and data.shape[0] != data[NAs_flat].shape[0]:
         warnings.warn(f"{data.shape[0] - data[NAs_flat].shape[0]} {self.get_lhs().variable} values ({round((data.shape[0] - data[NAs_flat].shape[0]) / data.shape[0] * 100,ndigits=2)}%) are NA.")

      n_y = data.shape[0]

      id_col = None
      if not self.series_id is None:
         id_col = np.array(data[self.series_id])

      var_map = self.get_var_map()
      n_var = len(var_map)
      var_keys = var_map.keys()
      var_types = self.get_var_types()
      factor_coding = self.get_factor_codings()
      
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
         y_flat = np.array(data[self.get_lhs().variable]).reshape(-1,1)
         
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
         c_raw = np.array(data[c])

         if var_types[c] == VarType.FACTOR:

            c_coding = factor_coding[c]

            # Code factor variable
            c_code = [c_coding[cr] for cr in c_raw]

            cov_flat[:,var_map[c]] = c_code

         else:
            cov_flat[:,var_map[c]] = c_raw
      
      # Now split cov by series id as well
      cov = None
      if not self.series_id is None:
         cov = np.split(cov_flat,sid[1:],axis=0)

      return y_flat,cov_flat,NAs_flat,y,cov,NAs,sid

    def __absorb_constraints(self):
      var_map = self.get_var_map()

      for sti in self.get_smooth_term_idx():

         sterm = self.__terms[sti]

         if not sterm.is_identifiable:
            continue
         
         sterm.Z = []
         vars = sterm.variables

         for vi in range(len(vars)):
            # If a smooth term needs to be identifiable I act as if you would have asked for nk+1!
            # so that the identifiable term is of the dimension expected.
            
            if len(vars) > 1:
               id_nk = sterm.nk[vi] + 1
            else:
               id_nk = sterm.nk + 1

            matrix_term = sterm.basis(None,self.cov_flat[self.NOT_NA_flat,var_map[vars[vi]]],
                                      None,id_nk,min_c=self.__var_mins[vars[vi]],
                                      max_c=self.__var_maxs[vars[vi]], **sterm.basis_kwargs)

            # Wood (2017) 5.4.1 Identifiability constraints via QR. ToDo: Replace with cheaper reflection method.
            C = np.sum(matrix_term,axis=0).reshape(-1,1)
            Q,_ = scp.linalg.qr(C,pivoting=False,mode='full')
            sterm.Z.append(Q[:,1:])

    def build_penalties(self):
      """Builds the penalties required by ``self.__terms``. Called automatically whenever needed. Call manually only for testing."""

      if self.penalties is not None:
         warnings.warn("Penalties were already initialized. Resetting them.")
         self.__get_coef_info() # Because previous initialization might have over-written n_coef or unpenalized _coef
         self.penalties = None

      col_S = self.n_coef
      factor_levels = self.get_factor_levels()
      terms = self.__terms
      penalties = []
      start_idx = self.unpenalized_coef

      if start_idx is None:
         ValueError("Penalty start index is ill-defined. Make sure to call 'formula.__get_coef_info' before calling this function.")

      cur_pen_idx = start_idx
      prev_pen_idx = start_idx

      for irsti in self.get_ir_smooth_term_idx():

         irsterm = terms[irsti]

         # Calculate nCoef 
         n_coef = irsterm.nk
         
         by_levels = None
         if irsterm.by is not None:
            by_levels = factor_levels[irsterm.by]
         
         if not irsterm.is_penalized:
            if len(penalties) == 0:

               added_not_penalized = n_coef
               if irsterm.by is not None:
                  added_not_penalized *= len(by_levels)
               start_idx += added_not_penalized
               self.unpenalized_coef += added_not_penalized
               cur_pen_idx = start_idx

               warnings.warn(f"Impulse response smooth {irsti} is not penalized. Smoothing terms should generally be penalized.")

            else:
               raise KeyError(f"Impulse response smooth {irsti} is not penalized and placed in the formula after penalized terms. Unpenalized terms should be moved to the beginning of the formula, ideally behind any linear terms.")
         
         else:

            for penid,pen in enumerate(irsterm.penalty):
               
               # Smooth terms can have multiple penalties.
               # In that case the starting index of every subsequent
               # penalty needs to be reset.
               if penid > 0:
                  cur_pen_idx = prev_pen_idx

               penalties,cur_pen_idx = build_irf_penalties(penalties,cur_pen_idx,
                                                           pen,penid,irsterm,
                                                           by_levels,n_coef,col_S)
         
         # Keep track of previous penalty starting index
         prev_pen_idx = cur_pen_idx
                        
      for sti in self.get_smooth_term_idx():

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

               if sterm.by_latent is not False and self.has_scale_split() is False:
                  added_not_penalized *= self.__n_j

               start_idx += added_not_penalized
               self.unpenalized_coef += added_not_penalized

               warnings.warn(f"Smooth {sti} is not penalized. Smoothing terms should generally be penalized.")

            else:
               raise KeyError(f"Smooth {sti} is not penalized and placed in the formula after penalized terms. Unpenalized terms should be moved to the beginning of the formula, ideally behind any linear terms.")
         
         else:
            S_j_TP_last = None
            TP_last_n = 0
            for penid,pen in enumerate(sterm.penalty):
            
               # Smooth terms can have multiple penalties.
               # In that case the starting index of every subsequent
               # penalty needs to be reset.
               if penid > 0:
                  cur_pen_idx = prev_pen_idx
               
               prev_n_pen = len(penalties)
               penalties,cur_pen_idx = build_smooth_penalties(self.has_scale_split(),self.__n_j,
                                                              penalties,cur_pen_idx,
                                                              pen,penid,sterm,vars,
                                                              by_levels,n_coef,col_S)

               if sterm.has_null_penalty:

                  n_pen = len(penalties)
                  # Optionally include a Null-space penalty - an extra penalty on the
                  # function space not regularized by the penalty we just created:

                  S_j_last = penalties[-1].S_J.toarray()
                  last_pen_rep = penalties[-1].rep_sj

                  if len(vars) > 1:
                     # Distinguish between TP smooths of multiple variables and
                     # single variable smooths. For TP smooths Marra & Wood (2011) suggest to first
                     # sum over the penalties for individual variables and then computing the null-space
                     # for that summed penalty.

                     # First sum over the first len(vars) penalties that were recently added. If there
                     # are more then these are identical - just corresponding to different by levels.
                     # Therefore, last_pen_rep also does not have to be updated.

                     if penid == 0:
                        S_j_TP_last =  S_j_last
                     else:
                        S_j_TP_last +=  S_j_last
                     
                     TP_last_n += (n_pen - prev_n_pen)

                     if penid < (len(sterm.penalty) - 1):
                        continue

                     # In the end update the number of new penalties based on the number of variables
                     # involed in the TP.
                     S_j_last = S_j_TP_last
                     n_pen = prev_n_pen + int(TP_last_n / len(vars))
                  
                  # Based on: Marra & Wood (2011) and: https://rdrr.io/cran/mgcv/man/gam.selection.html
                  # and: https://eric-pedersen.github.io/mgcv-esa-workshop/slides/03-model-selection.pdf
                  s, U =scp.linalg.eigh(S_j_last)
                  DNULL = U[:,s <= 1e-7]
                  DNULL = DNULL.reshape(S_j_last.shape[1],-1)

                  SNULL = DNULL @ DNULL.T

                  SNULL = scp.sparse.csc_array(SNULL)
                  DNULL = scp.sparse.csc_array(DNULL)

                  # Data in S and D is in canonical format, for competability this is translated to data, rows, columns
                  pen_data,pen_rows,pen_cols = translate_sparse(SNULL)
                  chol_data,chol_rows,chol_cols = translate_sparse(DNULL)

                  cur_pen_idx = prev_pen_idx

                  lTerm = LambdaTerm(start_index=cur_pen_idx,
                                       type = pen)
                  
                  # Embed first penalty - if the term has a by-keyword more are added below.
                  lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,cur_pen_idx)
                  lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,cur_pen_idx)
                  lTerm.S_J = embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J)
                  
                  # Single penalty added - but could involve by keyword
                  if (n_pen - prev_n_pen) == 1:
                     
                     # Handle any By-keyword
                     if last_pen_rep > 1:
                        for _ in range(last_pen_rep - 1):
                           lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,cur_pen_idx)
                           lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,cur_pen_idx)
                     
                        lTerm.rep_sj = last_pen_rep

                     # In any case, term can be appended here.
                     penalties.append(lTerm)
                  else:
                     # Independent penalties via by
                     # Append penalty for first level
                     penalties.append(lTerm)

                     # And add the penalties again for the remaining levels as separate terms
                     for _ in range((n_pen - prev_n_pen) - 1):
                        lTerm = LambdaTerm(start_index=cur_pen_idx,
                                       type = pen)
                  
                        # Embed penalties
                        lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,cur_pen_idx)
                        lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,cur_pen_idx)
                        lTerm.S_J = embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J)
                        penalties.append(lTerm)

         # Keep track of previous penalty starting index
         prev_pen_idx = cur_pen_idx

      for rti in self.get_random_term_idx():

         rterm = terms[rti]
         vars = rterm.variables

         if isinstance(rterm,ri):
            pen_data,pen_rows,pen_cols,chol_data,chol_rows,chol_cols = id_dist_pen(len(factor_levels[vars[0]]))

            lTerm = LambdaTerm(start_index=cur_pen_idx,
                                             type = PenType.IDENTITY)
            
            lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,cur_pen_idx)
            lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,cur_pen_idx)
            lTerm.S_J = embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J)
            penalties.append(lTerm)

         else:
            if rterm.var_coef is None:
               raise ValueError("Number of coefficients for random slope were not initialized.")
            if len(vars) > 1 and rterm.var_coef > 1:
               # Separate penalties for interactions involving at least one categorical factor.
               # In that case, a separate penalty will describe the random coefficients for the random factor (rterm.by)
               # per level of the (interaction of) categorical factor(s) involved in the interaction.
               # For interactions involving only continuous variables this condition will be false and a single
               # penalty will be estimated.
               pen_data,pen_rows,pen_cols,chol_data,chol_rows,chol_cols = id_dist_pen(len(factor_levels[rterm.by]))
               for _ in range(rterm.var_coef):
                  lTerm = LambdaTerm(start_index=cur_pen_idx,
                                             type = PenType.IDENTITY)
            
                  lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,cur_pen_idx)
                  lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,cur_pen_idx)
                  lTerm.S_J = embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J)
                  penalties.append(lTerm)

            else:
               # Single penalty for random coefficients of a single variable (categorical or continuous) or an
               # interaction of only continuous variables.
               pen_data,pen_rows,pen_cols,chol_data,chol_rows,chol_cols = id_dist_pen(len(factor_levels[rterm.by])*rterm.var_coef)


               lTerm = LambdaTerm(start_index=cur_pen_idx,
                                             type = PenType.IDENTITY)
            
               lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,cur_pen_idx)
               lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,cur_pen_idx)
               lTerm.S_J = embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J)
               penalties.append(lTerm)
            
      if cur_pen_idx != col_S:
         raise ValueError("Penalty dimension {cur_pen_idx},{cur_pen_idx} does not match outer model matrix dimension {col_S}")

      self.penalties = penalties
    
    #### Getters ####

    def get_lhs(self) -> lhs:
       """Get a copy of the ``lhs`` specified for this formula."""
       return copy.deepcopy(self.__lhs)
    
    def get_terms(self) -> list[GammTerm]:
       """Get a copy of the ``terms`` specified for this formula."""
       return copy.deepcopy(self.__terms)
    
    def get_data(self) -> pd.DataFrame:
       """Get a copy of the ``data`` specified for this formula."""
       return copy.deepcopy(self.__data)

    def get_factor_codings(self) -> dict:
        """Get a copy of the factor coding dictionary. Keys are factor variables in the data, values are dictionaries, where the keys correspond to the levels (str) of the factor and the values to their encoded levels (int)."""
        return copy.deepcopy(self.__factor_codings)
    
    def get_coding_factors(self) -> dict:
        """Get a copy of the factor coding dictionary. Keys are factor variables in the data, values are dictionaries, where the keys correspond to the encoded levels (int) of the factor and the values to their levels (str)."""
        return copy.deepcopy(self.__coding_factors)
    
    def get_var_map(self) -> dict:
        """Get a copy of the var map dictionary. Keys are variables in the data, values their column index in the encoded predictor matrix returned by ``self.encode_data``."""
        return copy.deepcopy(self.__var_to_cov)
    
    def get_factor_levels(self) -> dict:
       """Get a copy of the factor levels dictionary. Keys are factor variables in the data, values are np.arrays holding the unique levels (as str) of the corresponding factor."""
       return copy.deepcopy(self.__factor_levels)
    
    def get_var_types(self) -> dict:
       """Get a copy of the var types dictionary. Keys are variables in the data, values are either ``VarType.NUMERIC`` for continuous variables or ``VarType.FACTOR`` for categorical variables."""
       return copy.deepcopy(self.__var_types)
    
    def get_var_mins(self) -> dict:
       """Get a copy of the var mins dictionary. Keys are variables in the data, values are either the minimum value the variable takes on in ``self.__data`` for continuous variables or ``None` for categorical variables."""
       return copy.deepcopy(self.__var_mins)
    
    def get_var_maxs(self) -> dict:
       """Get a copy of the var maxs dictionary. Keys are variables in the data, values are either the maximum value the variable takes on in ``self.__data`` for continuous variables or ``None` for categorical variables."""
       return copy.deepcopy(self.__var_maxs)
    
    def get_var_mins_maxs(self) -> (dict,dict):
       """Get a tuple containing copies of both the mins and maxs directory. See ``self.get_var_mins`` and ``self.get_var_maxs``."""
       return (copy.deepcopy(self.__var_mins),copy.deepcopy(self.__var_maxs))
    
    def get_linear_term_idx(self) -> list[int]:
       """Get a copy of the list of indices that identify linear terms in ``self.__terms``."""
       return(copy.deepcopy(self.__linear_terms))
    
    def get_smooth_term_idx(self) -> list[int]:
       """Get a copy of the list of indices that identify smooth terms in ``self.__terms``."""
       return(copy.deepcopy(self.__smooth_terms))
    
    def get_ir_smooth_term_idx(self) -> list[int]:
       """Get a copy of the list of indices that identify impulse response terms in ``self.__terms``."""
       return(copy.deepcopy(self.__ir_smooth_terms))
    
    def get_random_term_idx(self) -> list[int]:
       """Get a copy of the list of indices that identify random terms in ``self.__terms``."""
       return(copy.deepcopy(self.__random_terms))
    
    def get_nj(self) -> int:
       """Get the number of latent states assumed by this formula."""
       if self.__has_irf:
          # Every event has an irf and there are always
          # n_event + 1 states.
          return self.__n_irf + 1
       return self.__n_j
    
    def get_n_coef(self) -> int:
       """Get the number of coefficients that are implied by the formula."""
       return self.n_coef
    
    def get_penalties(self) -> list:
       """Get a copy of the penalties implied by the formula. Will be None if the penalties have not been initizlized yet."""
       return copy.deepcopy(self.penalties)
    
    def get_depvar(self) -> list:
       """Get a copy of the encoded dependent variable (defined via ``self.__lhs``)."""
       return copy.deepcopy(self.y_flat)
    
    def get_notNA(self) -> list:
       """Get a copy of the encoded 'not a NA' vector for the dependent variable (defined via ``self.__lhs``)."""
       return copy.deepcopy(self.NOT_NA_flat)
    
    def has_intercept(self) -> bool:
       """Does this formula include an intercept or not."""
       return self.__has_intercept
    
    def has_ir_terms(self) -> bool:
       """Does this formula include impulse response terms or not."""
       return self.__has_irf
    
    def has_scale_split(self) -> bool:
       """Does this formula include a scale split or not."""
       return self.__split_scale
    
    def get_term_names(self) -> list:
       """Returns a copy of the list with the names of the terms specified for this formula."""
       return copy.deepcopy(self.__term_names)

def embed_in_S_sparse(pen_data,pen_rows,pen_cols,S_emb,S_col,cIndex):
   """Embed a term-specific penalty matrix (provided as elements, row and col indices) into the across-term penalty matrix (see Wood, 2017) """

   embedding = np.array(pen_data)
   r_embedding = np.array(pen_rows) + cIndex
   c_embedding = np.array(pen_cols) + cIndex

   if S_emb is None:
      S_emb = scp.sparse.csc_array((embedding,(r_embedding,c_embedding)),shape=(S_col,S_col))
   else:
      S_emb += scp.sparse.csc_array((embedding,(r_embedding,c_embedding)),shape=(S_col,S_col))

   return S_emb,cIndex+(pen_cols[-1]+1)

def embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,Sj):
   """Parameterize a term-specific penalty matrix (provided as elements, row and col indices)"""
   embedding = np.array(pen_data)
   pen_col = pen_cols[-1]+1

   if Sj is None:
      Sj = scp.sparse.csc_array((embedding,(pen_rows,pen_cols)),shape=(pen_col,pen_col))
   else:
      Sj += scp.sparse.csc_array((embedding,(pen_rows,pen_cols)),shape=(pen_col,pen_col))
      
   return Sj


def build_linear_term_matrix(ci,n_y,has_intercept,lti,lterm,var_types,var_map,factor_levels,ridx,cov_flat,use_only):
   """Parameterize model matrix for a linear term."""
   new_elements = []
   new_rows = []
   new_cols = []
   new_ci = 0
   # Main effects
   if len(lterm.variables) == 1:
      var = lterm.variables[0]
      if var_types[var] == VarType.FACTOR:
         offset = np.ones(n_y)
         
         fl_start = 0

         if has_intercept: # Dummy coding when intercept is added.
            fl_start = 1

         for fl in range(fl_start,len(factor_levels[var])):
            fridx = ridx[cov_flat[:,var_map[var]] == fl]
            if use_only is None or lti in use_only:
               new_elements.extend(offset[fridx])
               new_rows.extend(fridx)
               new_cols.extend([ci for _ in range(len(fridx))])
            ci += 1
            new_ci += 1

      else: # Continuous predictor
         slope = cov_flat[:,var_map[var]]
         if use_only is None or lti in use_only:
            new_elements.extend(slope)
            new_rows.extend(ridx)
            new_cols.extend([ci for _ in range(n_y)])
         ci += 1
         new_ci += 1

   else: # Interactions
      interactions = []
      inter_idx = []

      for var in lterm.variables:
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
         if use_only is None or lti in use_only:
            new_elements.extend(inter[ridx[inter_idx]])
            new_rows.extend(ridx[inter_idx])
            new_cols.extend([ci for _ in range(len(ridx[inter_idx]))])
         ci += 1
         new_ci += 1
   
   return new_elements,new_rows,new_cols,new_ci

def build_ir_smooth_term_matrix(ci,irsti,irsterm,var_map,factor_levels,ridx,cov,state_est,use_only,pool,tol):
   """Parameterize model matrix for an impulse response term."""
   var = irsterm.variables[0]
   term_elements = []
   term_idx = []

   new_elements = []
   new_rows = []
   new_cols = []
   new_ci = 0

   # Calculate number of coefficients
   n_coef = irsterm.nk

   if irsterm.by is not None:
      by_levels = factor_levels[irsterm.by]
      n_coef *= len(by_levels)

   if pool is None:
      for s_cov,s_state in zip(cov,state_est):
         
         # Create matrix for state corresponding to term.
         # ToDo: For Multivariate case, the matrix term needs to be build iteratively for
         # every level of the multivariate factor to make sure that the convolution operation
         # works as intended. The splitting can happen later via by.
         matrix_term = irsterm.basis(irsterm.event,s_cov[:,var_map[var]],s_state, irsterm.nk, **irsterm.basis_kwargs)
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

         if use_only is None or irsti in use_only:
            new_elements.extend(term_elements[m_coli])
            new_rows.extend(ridx[term_idx[m_coli]])
            new_cols.extend([ci for _ in range(len(term_elements[m_coli]))])
         ci += 1
         new_ci += 1

   else:
      raise NotImplementedError("Multiprocessing code for impulse response terms is not yet implemented in new formula api.")
   
   return new_elements,new_rows,new_cols,new_ci

def build_smooth_term_matrix(ci,n_j,has_scale_split,sti,sterm,var_map,var_mins,var_maxs,factor_levels,ridx,cov_flat,state_est_flat,use_only,tol):
   """Parameterize model matrix for a smooth term."""
   vars = sterm.variables
   term_ridx = []

   new_elements = []
   new_rows = []
   new_cols = []
   new_ci = 0

   # Calculate Coef number for control checks
   if len(vars) > 1:
      n_coef = np.prod(sterm.nk)
   else:
      n_coef = sterm.nk
   #print(n_coef)

   if sterm.by is not None:
      by_levels = factor_levels[sterm.by]
      n_coef *= len(by_levels)

      if sterm.by_latent is not False and has_scale_split is False:
         n_coef *= n_j
      
   # Calculate smooth term for corresponding covariate

   # Handle identifiability constraints for every basis and
   # optionally update tensor surface.
   for vi in range(len(vars)):

      if len(vars) > 1:
         id_nk = sterm.nk[vi]
      else:
         id_nk = sterm.nk

      if sterm.is_identifiable:
         id_nk += 1

      #print(var_mins[vars[0]],var_maxs[vars[0]])
      matrix_term_v = sterm.basis(None,cov_flat[:,var_map[vars[vi]]],
                                  None, id_nk, min_c=var_mins[vars[vi]],
                                  max_c=var_maxs[vars[vi]], **sterm.basis_kwargs)

      if sterm.is_identifiable:
         matrix_term_v = matrix_term_v @ sterm.Z[vi]
      
      if vi == 0:
         matrix_term = matrix_term_v
      else:
         matrix_term = TP_basis_calc(matrix_term,matrix_term_v)

   m_rows, m_cols = matrix_term.shape
   #print(m_cols)
   term_ridx = [ridx[:] for _ in range(m_cols)]

   # Handle optional by keyword
   if sterm.by is not None:
      new_term_ridx = []

      by_cov = cov_flat[:,var_map[sterm.by]]
      
      # Split by cov and update rows with elements in columns
      for by_level in range(len(by_levels)):
         by_cidx = by_cov == by_level
         for m_coli in range(m_cols):
            new_term_ridx.append(term_ridx[m_coli][by_cidx,])

      term_ridx = new_term_ridx
   
   # Handle optional binary keyword
   if sterm.binary is not None:
      new_term_ridx = []

      by_cov = cov_flat[:,var_map[sterm.binary[0]]]
      by_cidx = by_cov == sterm.binary_level

      for m_coli in range(m_cols):
         new_term_ridx.append(term_ridx[m_coli][by_cidx,])

      term_ridx = new_term_ridx

   # Handle split by latent variable if a shared scale term across latent stages is assumed.
   if sterm.by_latent is not False and has_scale_split is False:
      new_term_ridx = []

      # Split by state and update rows with elements in columns
      for by_state in range(n_j):
         for m_coli in range(len(term_ridx)):
            # Adjust state estimate for potential by split earlier.
            col_cor_state_est = state_est_flat[term_ridx[m_coli]]
            new_term_ridx.append(term_ridx[m_coli][col_cor_state_est == by_state,])

      term_ridx = new_term_ridx

   f_cols = len(term_ridx)

   if n_coef != f_cols:
      raise KeyError("Not all model matrix columns were created.")

   # Find basis elements > 0 and collect correspondings elements and row indices
   for m_coli in range(f_cols):
      final_ridx = term_ridx[m_coli]
      final_col = matrix_term[final_ridx,m_coli%m_cols]

      # Tolerance row index for this columns
      cidx = abs(final_col) > tol
      if use_only is None or sti in use_only:
         new_elements.extend(final_col[cidx])
         new_rows.extend(final_ridx[cidx])
         new_cols.extend([ci for _ in range(len(final_ridx[cidx]))])
      new_ci += 1
      ci += 1
   
   return new_elements,new_rows,new_cols,new_ci

def build_ri_term_matrix(ci,n_y,rti,rterm,var_map,factor_levels,ridx,cov_flat,use_only):
   """Parameterize model matrix for a random intercept term."""
   vars = rterm.variables
   offset = np.ones(n_y)
   by_cov = cov_flat[:,var_map[vars[0]]]

   new_elements = []
   new_rows = []
   new_cols = []
   new_ci = 0

   for fl in range(len(factor_levels[vars[0]])):
      fl_idx = by_cov == fl
      if use_only is None or rti in use_only:
         new_elements.extend(offset[fl_idx])
         new_rows.extend(ridx[fl_idx])
         new_cols.extend([ci for _ in range(len(offset[fl_idx]))])
      new_ci += 1
      ci += 1

   return new_elements,new_rows,new_cols,new_ci

def build_rs_term_matrix(ci,n_y,rti,rterm,var_types,var_map,factor_levels,ridx,cov_flat,use_only):
   """Parameterize model matrix for a random slope term."""

   by_cov = cov_flat[:,var_map[rterm.by]]
   by_levels = factor_levels[rterm.by]
   old_ci = ci

   # First get all columns for all linear predictors associated with this
   # term - might involve interactions!
   lin_elements,\
   lin_rows,\
   lin_cols,\
   lin_ci = build_linear_term_matrix(ci,n_y,False,rti,rterm,
                                     var_types,var_map,factor_levels,
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
         if use_only is None or rti in use_only:
            new_elements.extend(inter_i[fl_idx])
            new_rows.extend(rdx_i[fl_idx])
            new_cols.extend([ci for _ in range(len(inter_i[fl_idx]))])
         new_ci += 1
         ci += 1
      old_ci += 1

   # Matrix returned here holds for every linear coefficient one column for every level of the random
   # factor. So: coef1_1, coef_1_2, coef1_3, ... coef_n_1, coef_n,2, coef_n_3

   return new_elements,new_rows,new_cols,new_ci


def build_sparse_matrix_from_formula(terms,has_intercept,
                                     has_scale_split,
                                     ltx,irstx,stx,rtx,
                                     var_types,var_map,
                                     var_mins,var_maxs,
                                     factor_levels,cov_flat,
                                     cov,n_j,state_est_flat,
                                     state_est,pool=None,
                                     use_only=None,tol=1e-10):
   
   """Builds the entire model-matrix specified by a formula."""
   n_y = cov_flat.shape[0]
   elements = []
   rows = []
   cols = []
   ridx = np.array([ri for ri in range(n_y)])

   ci = 0
   for lti in ltx:
      lterm = terms[lti]

      if isinstance(lterm,i):
         offset = np.ones(n_y)
         if use_only is None or lti in use_only:
            elements.extend(offset)
            rows.extend(ridx)
            cols.extend([ci for _ in range(n_y)])
         ci += 1
      
      else:
         
         new_elements,\
         new_rows,\
         new_cols,\
         new_ci = build_linear_term_matrix(ci,n_y,has_intercept,lti,lterm,
                                           var_types,var_map,factor_levels,
                                           ridx,cov_flat,use_only)
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
      new_ci = build_ir_smooth_term_matrix(ci,irsti,irsterm,var_map,
                                           factor_levels,ridx,cov,
                                           state_est,use_only,pool,tol)
      elements.extend(new_elements)
      rows.extend(new_rows)
      cols.extend(new_cols)
      ci += new_ci

   for sti in stx:

      sterm = terms[sti]

      new_elements,\
      new_rows,\
      new_cols,\
      new_ci = build_smooth_term_matrix(ci,n_j,has_scale_split,sti,sterm,
                                        var_map,var_mins,var_maxs,
                                        factor_levels,ridx,cov_flat,
                                        state_est_flat,use_only,tol)
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
         new_ci = build_ri_term_matrix(ci,n_y,rti,rterm,var_map,factor_levels,
                                       ridx,cov_flat,use_only)
         elements.extend(new_elements)
         rows.extend(new_rows)
         cols.extend(new_cols)
         ci += new_ci

      elif isinstance(rterm,rs):
         new_elements,\
         new_rows,\
         new_cols,\
         new_ci = build_rs_term_matrix(ci,n_y,rti,rterm,var_types,var_map,
                                       factor_levels,ridx,cov_flat,use_only)
         elements.extend(new_elements)
         rows.extend(new_rows)
         cols.extend(new_cols)
         ci += new_ci

   mat = scp.sparse.csc_array((elements,(rows,cols)),shape=(n_y,ci))

   return mat


class PTerm():
   # Storage for sojourn time distribution
   def __init__(self,distribution:callable,
                init_kwargs:dict or None=None,
                fit_kwargs:dict or None=None,
                split_by:str or None=None) -> None:
      self.distribution = distribution
      self.kwargs = init_kwargs # Any parameters required to use distribution.
      if self.kwargs is None:
         self.kwargs = {}
      self.split_by = split_by
      self.fit_kwargs = fit_kwargs
      if self.fit_kwargs is None:
         self.fit_kwargs = {}
      self.n_by = None
      self.params = None

   def log_prob(self,d,by_i=None):
      # Get log-probability of durations d under current
      # sojourn distribution
      if self.params is None:
         return self.distribution.logpdf(d,**self.kwargs)

      if self.split_by is None:
         return self.distribution.logpdf(d,*self.params)
      
      # Optionally use distribution associated with a particular variable
      return self.distribution.logpdf(d,*self.params[by_i,:])

   def sample(self,N,by_i=None):
      # Sample N values from current sojourn time distribution
      if self.split_by is None:

         if not self.params is None:
            return self.distribution.rvs(*self.params,size=N)

      if not self.params is None:
         # Optionally again pick distribution parameters associated with
         # specific by variable
         return self.distribution.rvs(*self.params[by_i,:],size=N)
      
      # Initial sampling might be based on distributions default parameters
      # as provided by scipy and any necessary parameter specified in kwargs.
      return self.distribution.rvs(**self.kwargs,size=N)
   
   def fit(self,d,by_i=None):
      # Update parameters of distribution(s)
      if self.split_by is None:
         self.params = self.distribution.fit(d,**self.fit_kwargs)
      else:
         fit = self.distribution.fit(d,**self.fit_kwargs)
         if self.params is None:
            self.params = np.zeros((self.n_by,len(fit)))
         self.params[by_i,:] = fit
   
   def max_ppf(self,q):
      # Return the criticial value for quantile q.
      # In case split_by is true, return the max critical
      # value taken over all splits
      if self.params is None:
         return self.distribution.ppf(q,**self.kwargs)
      
      if not self.split_by is None:
         return max([self.distribution.ppf(q,*self.params[by_i,:]) for by_i in range(self.n_by)])
      
      return self.distribution.ppf(q,*self.params)
      
class PFormula():
   def __init__(self,terms:list[PTerm]) -> None:
      self.__terms = terms
   
   def get_terms(self):
      return copy.deepcopy(self.__terms)