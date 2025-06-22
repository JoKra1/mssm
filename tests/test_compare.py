from mssm.models import *
from mssm.src.python.compare import compare_CDL
from mssm.src.python.gamm_solvers import compute_S_emb_pinv_det,cpp_dChol,cpp_chol
from mssm.src.python.utils import estimateVp,correct_VB,DummyRhoPrior
import numpy as np
import os
from mssmViz.sim import*

max_atol = 100
max_rtol = 100

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
    sim_fit_model.fit(exclude_lambda=False,progress_bar=False,max_outer=100)

    sim_fit_formula2 = Formula(lhs("y"),
                                [i(),f(["x1"],nk=20,rp=1),f(["x2"],nk=20,rp=1),f(["x3"],nk=20,rp=1)],
                                data=sim_fit_dat,
                                print_warn=False)

    sim_fit_model2 = GAMM(sim_fit_formula2,Gaussian())
    sim_fit_model2.fit(exclude_lambda=False,progress_bar=False,max_outer=100)

    prior = DummyRhoPrior(b=np.log(1e12))

    # And perform a couple of bias corrected / smoothness uncertainty corrected / or not / comparisons...
    uncor_result = compare_CDL(sim_fit_model,sim_fit_model2,correct_V=False,correct_t1=False,seed=22)
    cor_result1 = compare_CDL(sim_fit_model,sim_fit_model2,correct_t1=False,n_c=1,grid='JJJ1',seed=22,Vp_fidiff=True)
    cor_result2 = compare_CDL(sim_fit_model,sim_fit_model2,correct_t1=False,n_c=1,grid='JJJ2',seed=22,only_expected_edf=True,use_importance_weights=False,Vp_fidiff=False)
    cor_result3 = compare_CDL(sim_fit_model,sim_fit_model2,correct_t1=False,n_c=10,grid='JJJ2',seed=22,use_importance_weights=False,Vp_fidiff=False)
    cor_result4 = compare_CDL(sim_fit_model,sim_fit_model2,correct_t1=False,n_c=1,grid='JJJ3',seed=22,Vp_fidiff=False,use_importance_weights=True)
    cor_result5 = compare_CDL(sim_fit_model,sim_fit_model2,correct_t1=False,n_c=1,grid='JJJ3',seed=22,Vp_fidiff=False,use_importance_weights=True,prior=prior)

    unbiased_uncor_result = compare_CDL(sim_fit_model,sim_fit_model2,correct_V=False,correct_t1=True,n_c=1,grid='JJJ3',seed=22)
    unbiased_cor_result1 = compare_CDL(sim_fit_model,sim_fit_model2,correct_t1=True,n_c=1,grid='JJJ3',seed=22,use_importance_weights=True,prior=prior)

    def test_comp1(self):
        assert round(self.uncor_result['aic_diff'],ndigits=3) == 1.945

    def test_comp2(self):
        assert round(self.cor_result1['aic_diff'],ndigits=3) == 1.946

    def test_comp3(self):
        assert round(self.cor_result2['aic_diff'],ndigits=3) == 1.668

    def test_comp4(self):
        assert round(self.cor_result3['aic_diff'],ndigits=3) == 1.701

    def test_comp5(self):
        assert round(self.cor_result4['aic_diff'],ndigits=3) == 1.946

    def test_comp6(self):
        assert round(self.cor_result5['aic_diff'],ndigits=3) == 2.942

    def test_comp7(self):
        assert round(self.unbiased_uncor_result['aic_diff'],ndigits=3) == 1.945

    def test_comp8(self):
        assert round(self.unbiased_cor_result1['aic_diff'],ndigits=3) == 2.942
    
    def test_edf1(self):
        assert round(self.unbiased_cor_result1['Res. DOF'],ndigits=3) == 0.981


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
    sim_fit_model.fit(exclude_lambda=False,progress_bar=False,max_outer=100)

    sim_fit_formula2 = Formula(lhs("y"),
                                [i(),f(["x1"],nk=20,rp=1),f(["x2"],nk=20,rp=1),f(["x3"],nk=20,rp=1)],
                                data=sim_fit_dat,
                                print_warn=False)

    sim_fit_model2 = GAMM(sim_fit_formula2,Gamma())
    sim_fit_model2.fit(exclude_lambda=False,progress_bar=False,max_outer=100)

    prior = DummyRhoPrior(b=np.log(1e12))
    # And perform a couple of bias corrected / smoothness uncertainty corrected / or not / comparisons...
    uncor_result = compare_CDL(sim_fit_model,sim_fit_model2,correct_V=False,correct_t1=False,seed=22)
    cor_result1 = compare_CDL(sim_fit_model,sim_fit_model2,correct_t1=False,n_c=1,grid='JJJ1',seed=22,Vp_fidiff=True)
    cor_result2 = compare_CDL(sim_fit_model,sim_fit_model2,correct_t1=False,n_c=1,grid='JJJ2',seed=22,only_expected_edf=True,use_importance_weights=False,Vp_fidiff=False)
    cor_result3 = compare_CDL(sim_fit_model,sim_fit_model2,correct_t1=False,n_c=10,grid='JJJ2',seed=22,use_importance_weights=False,Vp_fidiff=False)
    cor_result4 = compare_CDL(sim_fit_model,sim_fit_model2,correct_t1=False,n_c=1,grid='JJJ3',seed=22,Vp_fidiff=False,use_importance_weights=True)
    cor_result5 = compare_CDL(sim_fit_model,sim_fit_model2,correct_t1=False,n_c=10,grid='JJJ3',seed=22,Vp_fidiff=False,use_importance_weights=True,prior=prior)

    unbiased_uncor_result = compare_CDL(sim_fit_model,sim_fit_model2,correct_V=False,correct_t1=True,n_c=10,grid='JJJ3',seed=22)
    unbiased_cor_result1 = compare_CDL(sim_fit_model,sim_fit_model2,correct_t1=True,n_c=10,grid='JJJ3',seed=22,use_importance_weights=True,prior=prior)

    def test_comp1(self):
        assert round(self.uncor_result['aic_diff'],ndigits=3) == 25.323

    def test_comp2(self):
        assert round(self.cor_result1['aic_diff'],ndigits=3) == 25.149

    def test_comp3(self):
        assert round(self.cor_result2['aic_diff'],ndigits=3) == 24.838

    def test_comp4(self):
        assert round(self.cor_result3['aic_diff'],ndigits=3) == 24.819

    def test_comp5(self):
        assert round(self.cor_result4['aic_diff'],ndigits=3) == 25.058

    def test_comp6(self):
        assert round(self.cor_result5['aic_diff'],ndigits=3) == 25.058

    def test_comp7(self):
        assert round(self.unbiased_uncor_result['aic_diff'],ndigits=3) == 25.323

    def test_comp8(self):
        assert round(self.unbiased_cor_result1['aic_diff'],ndigits=3) == 25.058
    
    def test_edf1(self):
        assert round(self.unbiased_cor_result1['Res. DOF'],ndigits=3) == 22.091

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
    sim_fit_model.fit(progress_bar=False,max_outer=200,extend_lambda=False,control_lambda=2,max_inner=500)

    sim_fit_formula2 = Formula(lhs("y"),
                                [i(),f(["x1"],nk=20,rp=1),f(["x2"],nk=20,rp=1),f(["x3"],nk=20,rp=1)],
                                data=sim_fit_dat,
                                print_warn=False)

    sim_fit_model2 = GAMMLSS([sim_fit_formula2,copy.deepcopy(sim_fit_formula_sd)],family = GAMMALS([LOG(),LOGb(-0.01)]))
    sim_fit_model2.fit(progress_bar=False,max_outer=200,extend_lambda=False,control_lambda=2,max_inner=500)

    prior = DummyRhoPrior(b=np.log(1e12))
    # And perform a couple of bias corrected / smoothness uncertainty corrected / or not / comparisons...
    uncor_result = compare_CDL(sim_fit_model,sim_fit_model2,correct_V=False,correct_t1=False,seed=22)
    cor_result1 = compare_CDL(sim_fit_model,sim_fit_model2,correct_t1=False,n_c=1,grid='JJJ1',seed=22,Vp_fidiff=True)
    cor_result2 = compare_CDL(sim_fit_model,sim_fit_model2,correct_t1=False,n_c=1,grid='JJJ2',seed=22,only_expected_edf=True,use_importance_weights=False,Vp_fidiff=False)
    cor_result3 = compare_CDL(sim_fit_model,sim_fit_model2,correct_t1=False,n_c=1,grid='JJJ2',seed=22,use_importance_weights=False,Vp_fidiff=False)
    cor_result4 = compare_CDL(sim_fit_model,sim_fit_model2,correct_t1=False,n_c=1,grid='JJJ3',seed=22,Vp_fidiff=False,use_importance_weights=True)
    cor_result5 = compare_CDL(sim_fit_model,sim_fit_model2,correct_t1=False,n_c=1,grid='JJJ3',seed=22,Vp_fidiff=False,use_importance_weights=True,prior=prior)

    unbiased_uncor_result = compare_CDL(sim_fit_model,sim_fit_model2,correct_V=False,correct_t1=True,n_c=1,grid='JJJ3',seed=22)
    unbiased_cor_result1 = compare_CDL(sim_fit_model,sim_fit_model2,correct_t1=True,n_c=1,grid='JJJ3',seed=22,use_importance_weights=True,prior=prior)

    def test_comp1(self):
        np.testing.assert_allclose(self.uncor_result['aic_diff'],1.912,atol=min(max_atol,0.002),rtol=min(max_rtol,0.001))

    def test_comp2(self):
        np.testing.assert_allclose(self.cor_result1['aic_diff'],2.799,atol=min(max_atol,0.04),rtol=min(max_rtol,0.001))

    def test_comp3(self):
        np.testing.assert_allclose(self.cor_result2['aic_diff'],2.212,atol=min(max_atol,0.06),rtol=min(max_rtol,0.001))

    def test_comp4(self):
        np.testing.assert_allclose(self.cor_result3['aic_diff'],2.195,atol=min(max_atol,0.05),rtol=min(max_rtol,0.001))

    def test_comp5(self):
        np.testing.assert_allclose(self.cor_result4['aic_diff'],1.945,atol=min(max_atol,0.01),rtol=min(max_rtol,0.001))

    def test_comp6(self):
        np.testing.assert_allclose(self.cor_result5['aic_diff'],2.832,atol=min(max_atol,0.2),rtol=min(max_rtol,0.001))

    def test_comp7(self):
        np.testing.assert_allclose(self.unbiased_uncor_result['aic_diff'],1.912,atol=min(max_atol,0.002),rtol=min(max_rtol,0.001))

    def test_comp8(self):
        np.testing.assert_allclose(self.unbiased_cor_result1['aic_diff'],2.832,atol=min(max_atol,0.2),rtol=min(max_rtol,0.001))
    
    def test_edf1(self):
        np.testing.assert_allclose(self.unbiased_cor_result1['Res. DOF'],0.974,atol=min(max_atol,0.002),rtol=min(max_rtol,0.001))


class Test_Vb_corrections:
    # Simulate some data - effect of x0 is very very small
    sim_fit_dat = sim3(n=500,scale=2,c=1,family=Gaussian(),seed=21)

    # Now fit nested models
    sim_fit_formula = Formula(lhs("y"),
                                [i(),f(["x0"],nk=20,rp=0),f(["x1"],nk=20,rp=0),f(["x2"],nk=20,rp=0),f(["x3"],nk=20,rp=0)],
                                data=sim_fit_dat,
                                print_warn=False)

    model = GAMM(sim_fit_formula,Gaussian())
    model.fit(exclude_lambda=False,progress_bar=False,max_outer=100)

    # Now fit nested models
    sim_fit_formula2 = Formula(lhs("y"),
                                [i(),f(["x0"],nk=10,rp=0),f(["x1"],nk=10,rp=0),f(["x2"],nk=10,rp=0)],
                                data=sim_fit_dat,
                                print_warn=False)

    model2 = GAMM(sim_fit_formula2,Gaussian())
    model2.fit(exclude_lambda=False,progress_bar=False,max_outer=100)

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