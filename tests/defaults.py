import numpy as np
from mssm.models import *
import copy
max_atol = 100 #0
max_rtol = 100 #0.001

default_compare_test_kwargs = {"correct_V":True,
                               "correct_t1":False,
                               "perform_GLRT":False,
                               "nR":250,
                               "n_c":10,
                               "alpha":0.05,
                               "grid":'JJJ1',
                               "a":1e-7,
                               "b":1e7,
                               "df":40,
                               "verbose":False,
                               "drop_NA":True,
                               "method":"Chol",
                               "seed":None,
                               "only_expected_edf":False,
                               "Vp_fidiff":False,
                               "use_importance_weights":True,
                               "prior":None,
                               "recompute_H":False,
                               "compute_Vcc":True}

default_gamm_test_kwargs = {"max_outer":50,
                            "max_inner":100,
                            "conv_tol":1e-7,
                            "extend_lambda":True,
                            "control_lambda":True,
                            "exclude_lambda":False,
                            "extension_method_lam" : "nesterov",
                            "restart":False,
                            "method":"Chol",
                            "check_cond":1,
                            "progress_bar":True,
                            "n_cores":10,
                            "offset" : None}

default_gammlss_test_kwargs = {"max_outer":50,
                               "max_inner":200,
                               "min_inner":200,
                               "conv_tol":1e-7,
                               "extend_lambda":True,
                               "extension_method_lam":"nesterov2",
                               "control_lambda":1,
                               "restart":False,
                               "method":"Chol",
                               "check_cond":1,
                               "piv_tol":np.power(np.finfo(float).eps,0.04),
                               "should_keep_drop":True,
                               "prefit_grad":False,
                               "repara":False,
                               "progress_bar":True,
                               "n_cores":10,
                               "seed":0,
                               "init_lambda":None}

default_gsmm_test_kwargs = {"init_coef":None,
                            "max_outer":50,
                            "max_inner":200,
                            "min_inner":200,
                            "conv_tol":1e-7,
                            "extend_lambda":True,
                            "extension_method_lam":"nesterov2",
                            "control_lambda":1,
                            "restart":False,
                            "optimizer":"Newton",
                            "method":"Chol",
                            "check_cond":1,
                            "piv_tol":np.power(np.finfo(float).eps,0.04),
                            "progress_bar":False,
                            "n_cores":10,
                            "seed":0,
                            "drop_NA":True,
                            "init_lambda":None,
                            "form_VH":True,
                            "use_grad":False,
                            "build_mat":None,
                            "should_keep_drop":True,
                            "gamma":1,
                            "qEFSH":'SR1',
                            "overwrite_coef":True,
                            "max_restarts":0,
                            "qEFS_init_converge":True,
                            "prefit_grad":False,
                            "repara":False,
                            "init_bfgs_options":None}

def init_coef_gaumlss_tests(self, models):
      """Function to initialize the coefficients of the model.

      Fits a GAMM for the mean and initializes all coef. for the standard deviation to 1.

      :param models: A list of :class:`mssm.models.GAMM`'s, - each based on one of the formulas provided to a model.
      :type models: [mssm.models.GAMM]
      :return: A ``numpy.array`` of shape (-1,1), holding initial values for all model coefficients.
      :rtype: numpy array
      """
      
      mean_model = models[0]
      mean_model.family = Gaussian(self.links[0])
      test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
      test_kwargs["progress_bar"] = False
      mean_model.fit(**test_kwargs)

      m_coef,_ = mean_model.get_pars()
      coef = np.concatenate((m_coef.reshape(-1,1),np.ones((models[1].formulas[0].n_coef)).reshape(-1,1)))

      return coef

def init_coef_gammals_tests(self, models):
      """Function to initialize the coefficients of the model.

      Fits a GAMM for the mean and initializes all coef. for the scale parameter to 1.

      :param models: A list of :class:`mssm.models.GAMM`'s, - each based on one of the formulas provided to a model.
      :type models: [mssm.models.GAMM]
      :return: A ``numpy.array`` of shape (-1,1), holding initial values for all model coefficients.
      :rtype: numpy array
      """
      
      mean_model = models[0]
      mean_model.family = Gamma(self.links[0])
      test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
      test_kwargs["progress_bar"] = False
      test_kwargs["max_inner"] = 1
      mean_model.fit(**test_kwargs)

      m_coef,_ = mean_model.get_pars()
      coef = np.concatenate((m_coef.reshape(-1,1),np.ones((models[1].formulas[0].n_coef)).reshape(-1,1)))
      return coef

def init_penalties_tests_gammlss(self, penalties):
    return [0.01 for _ in range(len(penalties))]

def init_penalties_tests_gsmm(self, penalties):
    return [0.001 for _ in range(len(penalties))]