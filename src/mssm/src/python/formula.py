import warnings
import copy
from collections.abc import Callable
import numpy as np
import scipy as scp
import pandas as pd
from enum import Enum
from .smooths import TP_basis_calc
from .terms import GammTerm,i,l,f,irf,ri,rs
from .penalties import PenType,id_dist_pen,diff_pen,TP_pen,LambdaTerm

class VarType(Enum):
    NUMERIC = 1
    FACTOR = 2

class lhs():
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
        interactions = []
        inter_coef_names = []

        for var in lterm.variables:
            new_inter_coef_names = []

            # Interaction with categorical predictor as start
            if var_types[var] == VarType.FACTOR:
                fl_start = 0

                if has_intercept: # Dummy coding when intercept is added.
                    fl_start = 1

                if len(interactions) == 0:
                    for fl in range(fl_start,len(factor_levels[var])):
                        new_inter_coef_names.append(f"{var}_{coding_factors[var][fl]}")
                else:
                    for old_name in inter_coef_names:
                        for fl in range(fl_start,len(factor_levels[var])):
                            new_inter_coef_names.append(old_name + f"_{var}_{coding_factors[var][fl]}")

            else: # Interaction with continuous predictor as start
                if len(interactions) == 0:
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

def get_coef_info_smooth(has_sigma_split,n_j,sterm,factor_levels):
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

    if sterm.by is not None:
        by_levels = factor_levels[sterm.by]
        n_coef *= len(by_levels)

        if sterm.by_latent is not False and has_sigma_split is False:
            n_coef *= n_j
            for by_state in range(n_j):
                for by_level in by_levels:
                    coef_names.extend([f"f_{var_label}_{ink}_{by_level}_{by_state}" for ink in range(term_n_coef)])
        else:
            for by_level in by_levels:
                coef_names.extend([f"f_{var_label}_{ink}_{by_level}" for ink in range(term_n_coef)])
         
    else:
        if sterm.by_latent is not False and has_sigma_split is False:
            for by_state in range(n_j):
                coef_names.extend([f"f_{var_label}_{ink}_{by_state}" for ink in range(term_n_coef)])
        else:
            coef_names.extend([f"f_{var_label}_{ink}" for ink in range(term_n_coef)])
         
    total_coef += n_coef
    coef_per_term.append(n_coef)
    return total_coef,coef_names,coef_per_term

def build_smooth_penalties(has_sigma_split,n_j,penalties,cur_pen_idx,
                           pen_i_s_idx,pen_start_idx,pen,penid,sterm,
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
    lTerm = LambdaTerm(start_index=pen_i_s_idx,
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

            if sterm.by_latent is not False and has_sigma_split is False:
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

                # Make sure that this is set to None in case the lambda term for the first level received a start_idx reset.
            pen_start_idx = None 
            pen_i_s_idx = None

            pen_iter = len(by_levels) - 1

            if sterm.by_latent is not False and has_sigma_split is False:
                pen_iter = (len(by_levels) * n_j)-1

            for _ in range(pen_iter):

                # Create lambda term
                lTerm = LambdaTerm(start_index=pen_i_s_idx,
                                type = pen)

                # Embed penalties
                lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,cur_pen_idx)
                lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,cur_pen_idx)
                lTerm.S_J = embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J)
                penalties.append(lTerm)

    else:
        if sterm.by_latent is not False and has_sigma_split is False:
            # Handle by latent split - all latent levels get unique id
            penalties.append(lTerm)

            # Make sure that this is set to None in case the lambda term for the first level received a start_idx reset.
            pen_start_idx = None 
            pen_i_s_idx = None

            for _ in range(n_j-1):
                # Create lambda term
                lTerm = LambdaTerm(start_index=pen_i_s_idx,
                                type = pen)

                # Embed penalties
                lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,cur_pen_idx)
                lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,cur_pen_idx)
                lTerm.S_J = embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J)
                penalties.append(lTerm)
        else:
            penalties.append(lTerm)

    return penalties,cur_pen_idx,pen_start_idx

def build_irf_penalties(penalties,cur_pen_idx,pen_i_s_idx,pen_start_idx,
                        pen,penid,irsterm,by_levels,n_coef,col_S):
    # Determine penalty generator
    if pen == PenType.DIFFERENCE:
        pen_generator = diff_pen
    else:
        pen_generator = id_dist_pen

    # Get non-zero elements and indices for the penalty used by this term.
    pen_data,pen_rows,pen_cols,chol_data,chol_rows,chol_cols = pen_generator(n_coef,**irsterm.pen_kwargs[penid])

    # Create lambda term
    lTerm = LambdaTerm(start_index=pen_i_s_idx,
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

            # Make sure that this is set to None in case the lambda term for the first level received a start_idx reset.
            pen_start_idx = None 
            pen_i_s_idx = None
            for _ in range(len(by_levels)-1):
                # Create lambda term
                lTerm = LambdaTerm(start_index=pen_i_s_idx,
                                type = pen)

                # Embed penalties
                lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,cur_pen_idx)
                lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,cur_pen_idx)
                lTerm.S_J = embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J)
                penalties.append(lTerm)
    else:
        penalties.append(lTerm)
    
    return penalties,cur_pen_idx,pen_start_idx

class Formula():
    def __init__(self,
                 lhs:lhs,
                 terms:list[GammTerm],
                 data:pd.DataFrame,
                 series_id:str,
                 split_sigma:bool=False,
                 n_j:int=3) -> None:
        
        self.__lhs = lhs
        self.__terms = terms
        self.__data = data
        self.series_id = series_id
        self.__split_sigma = split_sigma # Separate sigma parameters per state, if true then formula counts for individual state.
        self.__n_j = n_j # Number of latent states to estimate - not for irf terms but for f terms!
        if self.__split_sigma:
           warnings.warn("split_sigma==True! All terms will be estimted per latent stage, independent of terms' by_latent status.")
        self.__factor_codings = {}
        self.__coding_factors = {}
        self.__factor_levels = {}
        self.__var_to_cov = {}
        self.__var_types = {}
        self.__var_mins = {}
        self.__var_maxs = {}
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
        self.y = None
        self.cov = None
        self.sid = None
        # Penalties
        self.penalties = None
        
        # Perform input checks first for LHS/Dependent variable.
        if self.__lhs.variable not in self.__data.columns:
            raise IndexError(f"Column '{self.__lhs.variable}' does not exist in Dataframe.")

        # Now some checks on the terms - some problems might only be caught later when the 
        # penalties are built.
        for ti, term in enumerate(self.__terms):
            
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
               if len(term.variables > 1):
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
                if not term.by is None:
                    if not term.by in self.__data.columns:
                        raise KeyError(f"By-variable '{term.by}' attributed to term {ti} does not exist in dataframe.")
                    
                    if data[term.by].dtype in ['float64','int64']:
                        raise KeyError(f"Data-type of By-variable '{term.by}' attributed to term {ti} must not be numeric but is. E.g., Make sure the pandas dtype is 'object'.")
                    
                     # Store information for by variables as well.
                    if not term.by in self.__var_to_cov:
                        self.__var_to_cov[term.by] = cvi

                        # Assign vartype enum
                        self.__var_types[term.by] = VarType.FACTOR
                        self.__var_mins[term.by] = None
                        self.__var_maxs[term.by] = None

                        # Code factor variables into integers for example for easy dummy coding
                        levels = np.unique(self.__data[term.by])

                        self.__factor_codings[term.by] = {}
                        self.__coding_factors[term.by] = {}
                        self.__factor_levels[term.by] = levels
                           
                        for ci,c in enumerate(levels):
                              self.__factor_codings[term.by][c] = ci
                              self.__coding_factors[term.by][ci] = c

                        cvi += 1
            
        if self.__n_irf > 0:
           self.__has_irf = True

        if self.__has_irf and self.__split_sigma:
           raise ValueError("Formula includes an impulse response term. split_sigma must be set to False!")
        
        if self.__has_irf and self.__has_by_latent:
           raise NotImplementedError("Formula includes an impulse response term. Having regular smooth terms differ by latent stages is currently not supported.")
        
        # Compute number of coef and coef names
        self.__get_coef_info()
        # Encode data into columns usable by the model
        y_flat,cov_flat,y,cov,sid = self.encode_data(self.__data)
        # Store encoding
        self.y_flat = y_flat
        self.cov_flat = cov_flat
        self.y = y
        self.cov = cov
        self.sid = sid
        # Absorb any constraints for model terms
        self.__absorb_constraints()
        # Compute penalties
        self.__build_penalties()

        #print(self.n_coef,len(self.coef_names))
  
    def __get_coef_info(self):
      var_types = self.get_var_types()
      factor_levels = self.get_factor_levels()
      coding_factors = self.get_coding_factors()

      terms = self.get_terms()
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
            # Main effects
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
               self.coef_names.extend([f"irf_{irsterm.state}_{ink}_{by_level}" for ink in range(irsterm.nk)])
         
         else:
            self.coef_names.extend([f"irf_{irsterm.state}_{ink}" for ink in range(irsterm.nk)])
         
         self.n_coef += n_coef
         self.ordered_coef_per_term.append(n_coef)

      for sti in self.get_smooth_term_idx():

         sterm = terms[sti]
         s_total_coef,\
         s_coef_names,\
         s_coef_per_term = get_coef_info_smooth(self.has_sigma_split(),
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
            raise NotImplementedError("Random slope terms are not yet supported.")
    
    def encode_data(self,r_data,prediction=False):

      # Drop NAs for fitting
      if prediction:
         data = r_data
      else:
         data = r_data[np.isnan(r_data[self.get_lhs().variable]) == False]

      if data.shape[0] != r_data.shape[0]:
         warnings.warn(f"{r_data.shape[0] - data.shape[0]} {self.get_lhs().variable} values ({round((r_data.shape[0] - data.shape[0]) / r_data.shape[0] * 100,ndigits=2)}%) are NA.")

      n_y = data.shape[0]

      id_col = np.array(data[self.series_id])
      var_map = self.get_var_map()
      n_var = len(var_map)
      var_keys = var_map.keys()
      var_types = self.get_var_types()
      factor_coding = self.get_factor_codings()
      
      # Collect every series from data frame, make sure to maintain the
      # order of the data frame.
      # Based on: https://stackoverflow.com/questions/12926898
      _, id = np.unique(id_col,return_index=True)
      sid = np.sort(id)

      if prediction: # For encoding new data
         y_flat = None
         y = None
      else:
         # Collect entire y column
         y_flat = np.array(data[self.get_lhs().variable]).reshape(-1,1)
         
         # Then split by seried id
         y = np.split(y_flat,sid[1:])

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
      cov = np.split(cov_flat,sid[1:],axis=0)

      return y_flat,cov_flat,y,cov,sid

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

            matrix_term = sterm.basis(None,self.cov_flat[:,var_map[vars[vi]]], None, id_nk,min_c=self.__var_mins[vars[vi]],max_c=self.__var_maxs[vars[vi]], **sterm.basis_kwargs)

            # Wood (2017) 5.4.1 Identifiability constraints via QR. ToDo: Replace with cheaper reflection method.
            C = np.sum(matrix_term,axis=0).reshape(-1,1)
            Q,_ = scp.linalg.qr(C,pivoting=False,mode='full')
            sterm.Z.append(Q[:,1:])

    def __build_penalties(self):
      col_S = self.n_coef
      factor_levels = self.get_factor_levels()
      terms = self.__terms
      penalties = []
      start_idx = self.unpenalized_coef

      if start_idx is None:
         ValueError("Penalty start index is ill-defined. Make sure to call 'formula.__get_coef_info' before calling this function.")

      pen_start_idx = start_idx
      cur_pen_idx = start_idx
      prev_pen_idx = start_idx

      for irsti in self.get_ir_smooth_term_idx():
         
         if len(penalties) > 0:
            pen_start_idx = None

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
               pen_start_idx = start_idx
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
                  pen_i_s_idx = cur_pen_idx
               else:
                  pen_i_s_idx = pen_start_idx

               penalties,cur_pen_idx,pen_start_idx = build_irf_penalties(penalties,cur_pen_idx,pen_i_s_idx,
                                                                         pen_start_idx,pen,penid,irsterm,
                                                                         by_levels,n_coef,col_S)
         
         # Keep track of previous penalty starting index
         prev_pen_idx = cur_pen_idx
                        
      for sti in self.get_smooth_term_idx():

         if len(penalties) > 0:
            pen_start_idx = None

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

               if sterm.by_latent is not False and self.has_sigma_split() is False:
                  added_not_penalized *= self.__n_j

               start_idx += added_not_penalized
               self.unpenalized_coef += added_not_penalized
               pen_start_idx = start_idx

               warnings.warn(f"Smooth {sti} is not penalized. Smoothing terms should generally be penalized.")

            else:
               raise KeyError(f"Smooth {sti} is not penalized and placed in the formula after penalized terms. Unpenalized terms should be moved to the beginning of the formula, ideally behind any linear terms.")
         
         else:
            
            for penid,pen in enumerate(sterm.penalty):
            
               # Smooth terms can have multiple penalties.
               # In that case the starting index of every subsequent
               # penalty needs to be reset.
               if penid > 0:
                  cur_pen_idx = prev_pen_idx
                  pen_i_s_idx = cur_pen_idx
               else:
                  pen_i_s_idx = pen_start_idx
               
               penalties,cur_pen_idx,pen_start_idx = build_smooth_penalties(self.has_sigma_split(),self.__n_j,
                                                                            penalties,cur_pen_idx,pen_i_s_idx,
                                                                            pen_start_idx,pen,penid,sterm,vars,
                                                                            by_levels,n_coef,col_S)

         # Keep track of previous penalty starting index
         prev_pen_idx = cur_pen_idx

      for rti in self.get_random_term_idx():

         if len(penalties) > 0:
            pen_start_idx = None

         rterm = terms[rti]
         vars = rterm.variables

         if isinstance(rterm,ri):
            pen_data,pen_rows,pen_cols,chol_data,chol_rows,chol_cols = id_dist_pen(len(factor_levels[vars[0]]))

            lTerm = LambdaTerm(start_index=pen_start_idx,
                                             type = PenType.IDENTITY)
            
            lTerm.D_J_emb, _ = embed_in_S_sparse(chol_data,chol_rows,chol_cols,lTerm.D_J_emb,col_S,cur_pen_idx)
            lTerm.S_J_emb, cur_pen_idx = embed_in_S_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J_emb,col_S,cur_pen_idx)
            lTerm.S_J = embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,lTerm.S_J)
            penalties.append(lTerm)

         else:
            NotImplementedError("Penalties for random slopes are not yet supported.")
      
      if cur_pen_idx != col_S:
         raise ValueError("Penalty dimension {cur_pen_idx},{cur_pen_idx} does not match outer model matrix dimension {col_S}")

      self.penalties = penalties
    
    #### Getters ####

    def get_lhs(self) -> lhs:
       return copy.deepcopy(self.__lhs)
    
    def get_terms(self) -> list[GammTerm]:
       return copy.deepcopy(self.__terms)
    
    def get_data(self) -> pd.DataFrame:
       return copy.deepcopy(self.__data)

    def get_factor_codings(self) -> dict:
        return copy.deepcopy(self.__factor_codings)
    
    def get_coding_factors(self) -> dict:
        return copy.deepcopy(self.__coding_factors)
    
    def get_var_map(self) -> dict:
        return copy.deepcopy(self.__var_to_cov)
    
    def get_factor_levels(self) -> dict:
       return copy.deepcopy(self.__factor_levels)
    
    def get_var_types(self) -> dict:
       return copy.deepcopy(self.__var_types)
    
    def get_var_mins(self) -> dict:
       return copy.deepcopy(self.__var_mins)
    
    def get_var_maxs(self) -> dict:
       return copy.deepcopy(self.__var_maxs)
    
    def get_var_mins_maxs(self) -> (dict,dict):
       return (copy.deepcopy(self.__var_mins),copy.deepcopy(self.__var_maxs))
    
    def get_linear_term_idx(self) -> list[int]:
       return(copy.deepcopy(self.__linear_terms))
    
    def get_smooth_term_idx(self) -> list[int]:
       return(copy.deepcopy(self.__smooth_terms))
    
    def get_ir_smooth_term_idx(self) -> list[int]:
       return(copy.deepcopy(self.__ir_smooth_terms))
    
    def get_random_term_idx(self) -> list[int]:
       return(copy.deepcopy(self.__random_terms))
    
    def has_intercept(self) -> bool:
       return self.__has_intercept
    
    def has_ir_terms(self) -> bool:
       return self.__has_irf
    
    def has_sigma_split(self) -> bool:
       return self.__split_sigma

def embed_in_S_sparse(pen_data,pen_rows,pen_cols,S_emb,S_col,cIndex):

   embedding = np.array(pen_data)
   r_embedding = np.array(pen_rows) + cIndex
   c_embedding = np.array(pen_cols) + cIndex

   if S_emb is None:
      S_emb = scp.sparse.csc_array((embedding,(r_embedding,c_embedding)),shape=(S_col,S_col))
   else:
      S_emb += scp.sparse.csc_array((embedding,(r_embedding,c_embedding)),shape=(S_col,S_col))

   return S_emb,cIndex+(pen_cols[-1]+1)

def embed_in_Sj_sparse(pen_data,pen_rows,pen_cols,Sj):

   embedding = np.array(pen_data)
   pen_col = pen_cols[-1]+1

   if Sj is None:
      Sj = scp.sparse.csc_array((embedding,(pen_rows,pen_cols)),shape=(pen_col,pen_col))
   else:
      Sj += scp.sparse.csc_array((embedding,(pen_rows,pen_cols)),shape=(pen_col,pen_col))
      
   return Sj

def build_sparse_matrix_from_formula(terms,has_intercept,
                                     has_sigma_split,
                                     ltx,irstx,stx,rtx,
                                     var_types,var_map,
                                     var_mins,var_maxs,
                                     factor_levels,cov_flat,
                                     cov,n_j,state_est_flat,
                                     state_est,pool=None,
                                     use_only=None,tol=1e-10):

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
                     elements.extend(offset[fridx])
                     rows.extend(fridx)
                     cols.extend([ci for _ in range(len(fridx))])
                  ci += 1

            else: # Continuous predictor
               slope = cov_flat[:,var_map[var]]
               if use_only is None or lti in use_only:
                  elements.extend(slope)
                  rows.extend(ridx)
                  cols.extend([ci for _ in range(n_y)])
               ci += 1

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
                  elements.extend(inter[ridx[inter_idx]])
                  rows.extend(ridx[inter_idx])
                  cols.extend([ci for _ in range(len(ridx[inter_idx]))])
               ci += 1
   
   for irsti in irstx:
      # Impulse response terms need to be calculate for every series individually - costly
      irsterm = terms[irsti]
      var = irsterm.variables[0]

      term_elements = []
      term_idx = []

      # Calculate Coef names
      n_coef = irsterm.nk

      if irsterm.by is not None:
         by_levels = factor_levels[irsterm.by]
         n_coef *= len(by_levels)

      if pool is None:
         for s_cov,s_state in zip(cov,state_est):
            
            # Create matrix for state corresponding to term.
            matrix_term = irsterm.basis(irsterm.state,s_cov[:,var_map[var]],s_state, irsterm.nk, **irsterm.basis_kwargs)
            m_rows,m_cols = matrix_term.shape

            # Handle optional by keyword
            if irsterm.by is not None:
               
               by_matrix_term = np.zeros((m_rows,m_cols*len(by_levels)),dtype=float)

               by_cov = s_cov[:,var_map[irsterm.by]]

               if len(np.unique(by_cov)) > 1:
                  raise ValueError(f"By-variable {irsterm.by} has varying levels on series level. This should not be the case.")
               
               # Fill the by matrix blocks.
               cByIndex = 0
               for by_level in range(len(by_levels)):
                  if by_level == by_cov[0]:
                     by_matrix_term[:,cByIndex:cByIndex+m_cols] = matrix_term
                     cByIndex += m_cols
               
               final_term = by_matrix_term
            else:
               final_term = matrix_term

            m_rows,m_cols = final_term.shape

            # Find basis elements > 0
            if len(term_idx) < 1:
               for m_coli in range(m_cols):
                  final_col = final_term[:,m_coli]
                  term_elements.append(final_col[abs(final_col) > tol])
                  term_idx.append(abs(final_col) > tol)
            else:
               for m_coli in range(m_cols):
                  final_col = final_term[:,m_coli]
                  term_elements[m_coli] = np.append(term_elements[m_coli],final_col[abs(final_col) > tol])
                  term_idx[m_coli] = np.append(term_idx[m_coli], abs(final_col) > tol)

         if n_coef != len(term_elements):
            raise KeyError("Not all model matrix columns were created.")
         
         # Now collect actual row indices
         for m_coli in range(len(term_elements)):
            if use_only is None or irsti in use_only:
               elements.extend(term_elements[:,m_coli])
               rows.extend(ridx[term_idx[m_coli]])
               cols.extend([ci for _ in range(len(term_elements[:,m_coli]))])
            ci += 1

      else:
         raise NotImplementedError("Multiprocessing code for impulse response terms is not yet implemented in new formula api.")

   for sti in stx:

      sterm = terms[sti]
      vars = sterm.variables

      term_ridx = []

      # Calculate Coef number for control checks
      if len(vars) > 1:
         n_coef = np.prod(sterm.nk)
      else:
         n_coef = sterm.nk
      #print(n_coef)

      if sterm.by is not None:
         by_levels = factor_levels[sterm.by]
         n_coef *= len(by_levels)

         if sterm.by_latent is not False and has_sigma_split is False:
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
         matrix_term_v = sterm.basis(None,cov_flat[:,var_map[vars[vi]]], None, id_nk, min_c=var_mins[vars[vi]],max_c=var_maxs[vars[vi]], **sterm.basis_kwargs)

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
      
      # Handle split by latent variable if a shared sigma term across latent stages is assumed.
      if sterm.by_latent is not False and has_sigma_split is False:
         new_term_ridx = []

         # Split by state and update rows with elements in columns
         for by_state in range(n_j):
            for m_coli in range(len(term_elements)):
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
            elements.extend(final_col[cidx])
            rows.extend(final_ridx[cidx])
            cols.extend([ci for _ in range(len(final_ridx[cidx]))])
         ci += 1

   for rti in rtx:
      rterm = terms[rti]
      vars = rterm.variables

      if isinstance(rterm,ri):
         offset = np.ones(n_y)

         by_cov = cov_flat[:,var_map[vars[0]]]

         for fl in range(len(factor_levels[vars[0]])):
            fl_idx = by_cov == fl
            if use_only is None or rti in use_only:
               elements.extend(offset[fl_idx])
               rows.extend(ridx[fl_idx])
               cols.extend([ci for _ in range(len(offset[fl_idx]))])
            ci += 1

         
      elif isinstance(rterm,rs):
         raise NotImplementedError("Random slope terms are not yet supported.")

   mat = scp.sparse.csc_array((elements,(rows,cols)),shape=(n_y,ci))

   return mat