import mssm
from mssm.models import *
from mssm.src.python.compare import compare_CDL
from mssm.src.python.utils import estimateVp,correct_VB,DummyRhoPrior
import numpy as np
import os
from mssmViz.sim import*
from .defaults import default_compare_test_kwargs,default_gamm_test_kwargs,default_gammlss_test_kwargs,max_atol,max_rtol,init_coef_gaumlss_tests,init_coef_gammals_tests,init_penalties_tests_gammlss

mssm.src.python.exp_fam.GAUMLSS.init_coef = init_coef_gaumlss_tests
mssm.src.python.exp_fam.GAMMALS.init_coef = init_coef_gammals_tests
mssm.src.python.exp_fam.GAUMLSS.init_lambda = init_penalties_tests_gammlss
mssm.src.python.exp_fam.GAMMALS.init_lambda = init_penalties_tests_gammlss
mssm.src.python.exp_fam.MULNOMLSS.init_lambda = init_penalties_tests_gammlss

################################################################## Tests ##################################################################

class Test_model_comparisons1:

    # Model comparison and smoothness uncertainty correction tests

    # Simulate some data
    sim_fit_dat = sim3(n=500,scale=2,c=0.1,family=Gaussian(),seed=21)

    # Now fit nested models
    sim_fit_formula = Formula(lhs("y"),
                                [i(),f(["x0"],nk=20,rp=1),f(["x1"],nk=20,rp=1),f(["x2"],nk=20,rp=1),f(["x3"],nk=20,rp=1)],
                                data=sim_fit_dat,
                                print_warn=False)

    sim_fit_model = GAMM(sim_fit_formula,Gaussian())

    test_kwargs_model = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs_model["exclude_lambda"] = False
    test_kwargs_model["max_outer"] = 100

    sim_fit_model.fit(**test_kwargs_model)

    sim_fit_formula2 = Formula(lhs("y"),
                                [i(),f(["x1"],nk=20,rp=1),f(["x2"],nk=20,rp=1),f(["x3"],nk=20,rp=1)],
                                data=sim_fit_dat,
                                print_warn=False)

    sim_fit_model2 = GAMM(sim_fit_formula2,Gaussian())
    sim_fit_model2.fit(**test_kwargs_model)

    prior = DummyRhoPrior(b=np.log(1e12))

    # And perform a couple of bias corrected / smoothness uncertainty corrected / or not / comparisons...
    test_kwargs = copy.deepcopy(default_compare_test_kwargs)
    test_kwargs["correct_V"] = False
    test_kwargs["correct_t1"] = False
    test_kwargs["seed"] = 22
    uncor_result = compare_CDL(sim_fit_model,sim_fit_model2,**test_kwargs)

    test_kwargs = copy.deepcopy(default_compare_test_kwargs)
    test_kwargs["correct_t1"] = False
    test_kwargs["n_c"] = 1
    test_kwargs["grid"] = "JJJ1"
    test_kwargs["seed"] = 22
    test_kwargs["Vp_fidiff"] = True
    cor_result1 = compare_CDL(sim_fit_model,sim_fit_model2,**test_kwargs)

    test_kwargs = copy.deepcopy(default_compare_test_kwargs)
    test_kwargs["correct_t1"] = False
    test_kwargs["n_c"] = 1
    test_kwargs["grid"] = "JJJ2"
    test_kwargs["seed"] = 22
    test_kwargs["only_expected_edf"] = True
    test_kwargs["use_importance_weights"] = False
    test_kwargs["Vp_fidiff"] = False
    cor_result2 = compare_CDL(sim_fit_model,sim_fit_model2,**test_kwargs)

    test_kwargs = copy.deepcopy(default_compare_test_kwargs)
    test_kwargs["correct_t1"] = False
    test_kwargs["n_c"] = 10
    test_kwargs["grid"] = "JJJ2"
    test_kwargs["seed"] = 22
    test_kwargs["use_importance_weights"] = False
    test_kwargs["Vp_fidiff"] = False
    cor_result3 = compare_CDL(sim_fit_model,sim_fit_model2,**test_kwargs)

    test_kwargs = copy.deepcopy(default_compare_test_kwargs)
    test_kwargs["correct_t1"] = False
    test_kwargs["n_c"] = 1
    test_kwargs["grid"] = "JJJ3"
    test_kwargs["seed"] = 22
    test_kwargs["use_importance_weights"] = True
    test_kwargs["Vp_fidiff"] = False
    cor_result4 = compare_CDL(sim_fit_model,sim_fit_model2,**test_kwargs)

    test_kwargs = copy.deepcopy(default_compare_test_kwargs)
    test_kwargs["correct_t1"] = False
    test_kwargs["n_c"] = 1
    test_kwargs["grid"] = "JJJ3"
    test_kwargs["seed"] = 22
    test_kwargs["use_importance_weights"] = True
    test_kwargs["Vp_fidiff"] = False
    test_kwargs["prior"] = prior
    cor_result5 = compare_CDL(sim_fit_model,sim_fit_model2,**test_kwargs)


    test_kwargs = copy.deepcopy(default_compare_test_kwargs)
    test_kwargs["correct_V"] = False
    test_kwargs["correct_t1"] = True
    test_kwargs["n_c"] = 1
    test_kwargs["grid"] = "JJJ3"
    test_kwargs["seed"] = 22
    unbiased_uncor_result = compare_CDL(sim_fit_model,sim_fit_model2,**test_kwargs)

    test_kwargs = copy.deepcopy(default_compare_test_kwargs)
    test_kwargs["correct_t1"] = True
    test_kwargs["n_c"] = 1
    test_kwargs["grid"] = "JJJ3"
    test_kwargs["seed"] = 22
    test_kwargs["use_importance_weights"] = True
    test_kwargs["prior"] = prior
    unbiased_cor_result1 = compare_CDL(sim_fit_model,sim_fit_model2,**test_kwargs)

    def test_comp1(self):
        assert round(self.uncor_result['aic_diff'],ndigits=3) == 1.98

    def test_comp2(self):
        assert round(self.cor_result1['aic_diff'],ndigits=3) == 1.981

    def test_comp3(self):
        assert round(self.cor_result2['aic_diff'],ndigits=3) == 1.703

    def test_comp4(self):
        assert round(self.cor_result3['aic_diff'],ndigits=3) == 1.736

    def test_comp5(self):
        assert round(self.cor_result4['aic_diff'],ndigits=3) == 1.981

    def test_comp6(self):
        assert round(self.cor_result5['aic_diff'],ndigits=3) == 2.977

    def test_comp7(self):
        assert round(self.unbiased_uncor_result['aic_diff'],ndigits=3) == 1.98

    def test_comp8(self):
        assert round(self.unbiased_cor_result1['aic_diff'],ndigits=3) == 2.977
    
    def test_edf1(self):
        assert round(self.unbiased_cor_result1['Res. DOF'],ndigits=3) == 1.48


class Test_model_comparisons2:

    # Model comparison and smoothness uncertainty correction tests

    # Simulate some data
    sim_fit_dat = sim3(n=500,scale=2,c=0.1,family=Gamma(),seed=21)

    # Now fit nested models
    sim_fit_formula = Formula(lhs("y"),
                                [i(),f(["x0"],nk=20,rp=1),f(["x1"],nk=20,rp=1),f(["x2"],nk=20,rp=1),f(["x3"],nk=20,rp=1)],
                                data=sim_fit_dat,
                                print_warn=False)

    sim_fit_model = GAMM(sim_fit_formula,Gamma())

    test_kwargs_model = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs_model["exclude_lambda"] = False
    test_kwargs_model["max_outer"] = 100

    sim_fit_model.fit(**test_kwargs_model)

    sim_fit_formula2 = Formula(lhs("y"),
                                [i(),f(["x1"],nk=20,rp=1),f(["x2"],nk=20,rp=1),f(["x3"],nk=20,rp=1)],
                                data=sim_fit_dat,
                                print_warn=False)

    sim_fit_model2 = GAMM(sim_fit_formula2,Gamma())
    sim_fit_model2.fit(**test_kwargs_model)

    prior = DummyRhoPrior(b=np.log(1e12))
    # And perform a couple of bias corrected / smoothness uncertainty corrected / or not / comparisons...
    test_kwargs = copy.deepcopy(default_compare_test_kwargs)
    test_kwargs["correct_V"] = False
    test_kwargs["correct_t1"] = False
    test_kwargs["seed"] = 22
    uncor_result = compare_CDL(sim_fit_model,sim_fit_model2,**test_kwargs)

    test_kwargs = copy.deepcopy(default_compare_test_kwargs)
    test_kwargs["correct_t1"] = False
    test_kwargs["n_c"] = 1
    test_kwargs["grid"] = "JJJ1"
    test_kwargs["seed"] = 22
    test_kwargs["Vp_fidiff"] = True
    cor_result1 = compare_CDL(sim_fit_model,sim_fit_model2,**test_kwargs)

    test_kwargs = copy.deepcopy(default_compare_test_kwargs)
    test_kwargs["correct_t1"] = False
    test_kwargs["n_c"] = 1
    test_kwargs["grid"] = "JJJ2"
    test_kwargs["seed"] = 22
    test_kwargs["only_expected_edf"] = True
    test_kwargs["use_importance_weights"] = False
    test_kwargs["Vp_fidiff"] = False
    cor_result2 = compare_CDL(sim_fit_model,sim_fit_model2,**test_kwargs)

    test_kwargs = copy.deepcopy(default_compare_test_kwargs)
    test_kwargs["correct_t1"] = False
    test_kwargs["n_c"] = 10
    test_kwargs["grid"] = "JJJ2"
    test_kwargs["seed"] = 22
    test_kwargs["use_importance_weights"] = False
    test_kwargs["Vp_fidiff"] = False
    cor_result3 = compare_CDL(sim_fit_model,sim_fit_model2,**test_kwargs)

    test_kwargs = copy.deepcopy(default_compare_test_kwargs)
    test_kwargs["correct_t1"] = False
    test_kwargs["n_c"] = 1
    test_kwargs["grid"] = "JJJ3"
    test_kwargs["seed"] = 22
    test_kwargs["use_importance_weights"] = True
    test_kwargs["Vp_fidiff"] = False
    cor_result4 = compare_CDL(sim_fit_model,sim_fit_model2,**test_kwargs)

    test_kwargs = copy.deepcopy(default_compare_test_kwargs)
    test_kwargs["correct_t1"] = False
    test_kwargs["n_c"] = 10
    test_kwargs["grid"] = "JJJ3"
    test_kwargs["seed"] = 22
    test_kwargs["use_importance_weights"] = True
    test_kwargs["Vp_fidiff"] = False
    test_kwargs["prior"] = prior
    cor_result5 = compare_CDL(sim_fit_model,sim_fit_model2,**test_kwargs)


    test_kwargs = copy.deepcopy(default_compare_test_kwargs)
    test_kwargs["correct_V"] = False
    test_kwargs["correct_t1"] = True
    test_kwargs["n_c"] = 10
    test_kwargs["grid"] = "JJJ3"
    test_kwargs["seed"] = 22
    unbiased_uncor_result = compare_CDL(sim_fit_model,sim_fit_model2,**test_kwargs)

    test_kwargs = copy.deepcopy(default_compare_test_kwargs)
    test_kwargs["correct_t1"] = True
    test_kwargs["n_c"] = 10
    test_kwargs["grid"] = "JJJ3"
    test_kwargs["seed"] = 22
    test_kwargs["use_importance_weights"] = True
    test_kwargs["prior"] = prior
    unbiased_cor_result1 = compare_CDL(sim_fit_model,sim_fit_model2,**test_kwargs)

    def test_comp1(self):
        assert round(self.uncor_result['aic_diff'],ndigits=3) == 25.03

    def test_comp2(self):
        assert round(self.cor_result1['aic_diff'],ndigits=3) == 24.522

    def test_comp3(self):
        assert round(self.cor_result2['aic_diff'],ndigits=3) == 24.545

    def test_comp4(self):
        assert round(self.cor_result3['aic_diff'],ndigits=3) == 24.526

    def test_comp5(self):
        assert round(self.cor_result4['aic_diff'],ndigits=3) == 24.765

    def test_comp6(self):
        assert round(self.cor_result5['aic_diff'],ndigits=3) == 24.765

    def test_comp7(self):
        assert round(self.unbiased_uncor_result['aic_diff'],ndigits=3) == 25.03

    def test_comp8(self):
        assert round(self.unbiased_cor_result1['aic_diff'],ndigits=3) == 24.765
    
    def test_edf1(self):
        assert round(self.unbiased_cor_result1['Res. DOF'],ndigits=3) == 21.959

class Test_model_comparison3_hard:

    # Model comparison and smoothness uncertainty correction tests

    # Simulate some data
    sim_fit_dat = sim3(n=500,scale=2,c=0.1,family=Gamma(),seed=21)

    # Now fit nested models
    sim_fit_formula = Formula(lhs("y"),
                                [i(),f(["x0"],nk=20,rp=1),f(["x1"],nk=20,rp=1),f(["x2"],nk=20,rp=1),f(["x3"],nk=20,rp=1)],
                                data=sim_fit_dat,
                                print_warn=False)

    sim_fit_formula_sd = Formula(lhs("y"),
                                [i()],
                                data=sim_fit_dat,
                                print_warn=False)

    sim_fit_model = GAMMLSS([sim_fit_formula,copy.deepcopy(sim_fit_formula_sd)],family = GAMMALS([LOG(),LOGb(-0.01)]))

    test_kwargs_model = copy.deepcopy(default_gammlss_test_kwargs)
    test_kwargs_model["extend_lambda"] = False
    test_kwargs_model["control_lambda"] = 2
    test_kwargs_model["max_outer"] = 200
    test_kwargs_model["max_inner"] = 500

    sim_fit_model.fit(**test_kwargs_model)

    sim_fit_formula2 = Formula(lhs("y"),
                                [i(),f(["x1"],nk=20,rp=1),f(["x2"],nk=20,rp=1),f(["x3"],nk=20,rp=1)],
                                data=sim_fit_dat,
                                print_warn=False)

    sim_fit_model2 = GAMMLSS([sim_fit_formula2,copy.deepcopy(sim_fit_formula_sd)],family = GAMMALS([LOG(),LOGb(-0.01)]))
    sim_fit_model2.fit(**test_kwargs_model)

    prior = DummyRhoPrior(b=np.log(1e12))
    # And perform a couple of bias corrected / smoothness uncertainty corrected / or not / comparisons...
    test_kwargs = copy.deepcopy(default_compare_test_kwargs)
    test_kwargs["correct_V"] = False
    test_kwargs["correct_t1"] = False
    test_kwargs["seed"] = 22
    uncor_result = compare_CDL(sim_fit_model,sim_fit_model2,**test_kwargs)

    test_kwargs = copy.deepcopy(default_compare_test_kwargs)
    test_kwargs["correct_t1"] = False
    test_kwargs["n_c"] = 1
    test_kwargs["grid"] = "JJJ1"
    test_kwargs["seed"] = 22
    test_kwargs["Vp_fidiff"] = True
    cor_result1 = compare_CDL(sim_fit_model,sim_fit_model2,**test_kwargs)

    test_kwargs = copy.deepcopy(default_compare_test_kwargs)
    test_kwargs["correct_t1"] = False
    test_kwargs["n_c"] = 1
    test_kwargs["grid"] = "JJJ2"
    test_kwargs["seed"] = 22
    test_kwargs["only_expected_edf"] = True
    test_kwargs["use_importance_weights"] = False
    test_kwargs["Vp_fidiff"] = False
    cor_result2 = compare_CDL(sim_fit_model,sim_fit_model2,**test_kwargs)

    test_kwargs = copy.deepcopy(default_compare_test_kwargs)
    test_kwargs["correct_t1"] = False
    test_kwargs["n_c"] = 1
    test_kwargs["grid"] = "JJJ2"
    test_kwargs["seed"] = 22
    test_kwargs["use_importance_weights"] = False
    test_kwargs["Vp_fidiff"] = False
    cor_result3 = compare_CDL(sim_fit_model,sim_fit_model2,**test_kwargs)

    test_kwargs = copy.deepcopy(default_compare_test_kwargs)
    test_kwargs["correct_t1"] = False
    test_kwargs["n_c"] = 1
    test_kwargs["grid"] = "JJJ3"
    test_kwargs["seed"] = 22
    test_kwargs["use_importance_weights"] = True
    test_kwargs["Vp_fidiff"] = False
    cor_result4 = compare_CDL(sim_fit_model,sim_fit_model2,**test_kwargs)

    test_kwargs = copy.deepcopy(default_compare_test_kwargs)
    test_kwargs["correct_t1"] = False
    test_kwargs["n_c"] = 1
    test_kwargs["grid"] = "JJJ3"
    test_kwargs["seed"] = 22
    test_kwargs["use_importance_weights"] = True
    test_kwargs["Vp_fidiff"] = False
    test_kwargs["prior"] = prior
    cor_result5 = compare_CDL(sim_fit_model,sim_fit_model2,**test_kwargs)


    test_kwargs = copy.deepcopy(default_compare_test_kwargs)
    test_kwargs["correct_V"] = False
    test_kwargs["correct_t1"] = True
    test_kwargs["n_c"] = 1
    test_kwargs["grid"] = "JJJ3"
    test_kwargs["seed"] = 22
    unbiased_uncor_result = compare_CDL(sim_fit_model,sim_fit_model2,**test_kwargs)

    test_kwargs = copy.deepcopy(default_compare_test_kwargs)
    test_kwargs["correct_t1"] = True
    test_kwargs["n_c"] = 1
    test_kwargs["grid"] = "JJJ3"
    test_kwargs["seed"] = 22
    test_kwargs["use_importance_weights"] = True
    test_kwargs["prior"] = prior
    unbiased_cor_result1 = compare_CDL(sim_fit_model,sim_fit_model2,**test_kwargs)

    def test_comp1(self):
        np.testing.assert_allclose(self.uncor_result['aic_diff'],1.912,atol=min(max_atol,0.002),rtol=min(max_rtol,0.001))

    def test_comp2(self):
        np.testing.assert_allclose(self.cor_result1['aic_diff'],1.877553,atol=min(max_atol,0.04),rtol=min(max_rtol,0.001))

    def test_comp3(self):
        np.testing.assert_allclose(self.cor_result2['aic_diff'],2.212,atol=min(max_atol,0.06),rtol=min(max_rtol,0.001))

    def test_comp4(self):
        np.testing.assert_allclose(self.cor_result3['aic_diff'],2.195,atol=min(max_atol,0.05),rtol=min(max_rtol,0.001))

    def test_comp5(self):
        np.testing.assert_allclose(self.cor_result4['aic_diff'],1.945,atol=min(max_atol,0.02),rtol=min(max_rtol,0.001))

    def test_comp6(self):
        np.testing.assert_allclose(self.cor_result5['aic_diff'],2.832,atol=min(max_atol,0.25),rtol=min(max_rtol,0.001))

    def test_comp7(self):
        np.testing.assert_allclose(self.unbiased_uncor_result['aic_diff'],1.912,atol=min(max_atol,0.002),rtol=min(max_rtol,0.001))

    def test_comp8(self):
        np.testing.assert_allclose(self.unbiased_cor_result1['aic_diff'],2.832,atol=min(max_atol,0.25),rtol=min(max_rtol,0.001))
    
    def test_edf1(self):
        np.testing.assert_allclose(self.unbiased_cor_result1['Res. DOF'],1.434149,atol=min(max_atol,0.002),rtol=min(max_rtol,0.08))


class Test_Vb_corrections:
    # Simulate some data - effect of x0 is very very small
    sim_fit_dat = sim3(n=500,scale=2,c=1,family=Gaussian(),seed=21)

    # Now fit nested models
    sim_fit_formula = Formula(lhs("y"),
                                [i(),f(["x0"],nk=20,rp=0),f(["x1"],nk=20,rp=0),f(["x2"],nk=20,rp=0),f(["x3"],nk=20,rp=0)],
                                data=sim_fit_dat,
                                print_warn=False)

    model = GAMM(sim_fit_formula,Gaussian())

    test_kwargs_model = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs_model["exclude_lambda"] = False
    test_kwargs_model["max_outer"] = 100

    model.fit(**test_kwargs_model)

    # Now fit nested models
    sim_fit_formula2 = Formula(lhs("y"),
                                [i(),f(["x0"],nk=10,rp=0),f(["x1"],nk=10,rp=0),f(["x2"],nk=10,rp=0)],
                                data=sim_fit_dat,
                                print_warn=False)

    model2 = GAMM(sim_fit_formula2,Gaussian())
    model2.fit(**test_kwargs_model)

    Vp1,_,_,_,_ = estimateVp(model,grid_type="JJJ1",verbose=True,seed=20)

    _,_,Vp2,_,_,total_edf,_,_,_,_ = correct_VB(model,grid_type="JJJ1",verbose=True,seed=20,df=2)

    _,LI,_,_,_,total_edf2,_,_,_,_ = correct_VB(model2,grid_type="JJJ2",verbose=True,seed=20,df=2)

    def test_Vp1(self):
        assert np.allclose(np.round(self.Vp1,decimals=3),np.array([[ 1.134000e+00, -7.000000e-03,  7.000000e-03, -1.050000e-01],
                                                                    [-7.000000e-03,  8.970000e-01,  1.600000e-02, -7.000000e-03],
                                                                    [ 7.000000e-03,  1.600000e-02,  4.470000e-01,  9.600000e-02],
                                                                    [-1.050000e-01, -7.000000e-03,  9.600000e-02,  5.989569e+03]]))
    
    def test_Vp2(self):
        assert np.allclose(self.Vp1,self.Vp2)
    
    def test_edf1(self):
        assert np.round(self.total_edf,decimals=3) == 22.15

    def test_edf2(self):
        assert np.round(self.total_edf2,decimals=3) == 16.857