from mssm.models import *
from mssm.src.python.compare import compare_CDL
import numpy as np
import os
from mssmViz.sim import*

class Test_model_comparisons1_hard:

    # Model comparison and smoothness uncertainty correction tests

    # Simulate some data - effect of x0 is very very small
    sim_fit_dat = sim3(n=500,scale=2,c=0.1,family=Gaussian(),seed=21)

    # Now fit nested models
    sim_fit_formula = Formula(lhs("y"),
                                [i(),f(["x0"],nk=20,rp=1),f(["x1"],nk=20,rp=1),f(["x2"],nk=20,rp=1),f(["x3"],nk=20,rp=1)],
                                data=sim_fit_dat,
                                print_warn=False)

    sim_fit_model = GAMM(sim_fit_formula,Gaussian())
    sim_fit_model.fit(exclude_lambda=False,progress_bar=False,maxiter=100)

    sim_fit_formula2 = Formula(lhs("y"),
                                [i(),f(["x1"],nk=20,rp=1),f(["x2"],nk=20,rp=1),f(["x3"],nk=20,rp=1)],
                                data=sim_fit_dat,
                                print_warn=False)

    sim_fit_model2 = GAMM(sim_fit_formula2,Gaussian())
    sim_fit_model2.fit(exclude_lambda=False,progress_bar=False,maxiter=100)

    # And perform a couple of bias corrected / smoothness uncertainty corrected / or not / comparisons...
    uncor_result = compare_CDL(sim_fit_model,sim_fit_model2,correct_V=False,correct_t1=False,grid='JJJ',seed=22)
    cor_result = compare_CDL(sim_fit_model,sim_fit_model2,nR=20,lR=100,correct_t1=False,n_c=1,grid='JJJ',seed=22)

    unbiased_uncor_result = compare_CDL(sim_fit_model,sim_fit_model2,correct_V=False,correct_t1=True,grid='JJJ',seed=22)
    unbiased_cor_result = compare_CDL(sim_fit_model,sim_fit_model2,nR=20,lR=100,correct_t1=True,n_c=2,grid='JJJ',seed=22)

    def test_comp1(self):
        assert round(self.unbiased_cor_result['Res. DOF'],ndigits=1) == 1.0
    
    def test_comp2(self):
        assert round(self.unbiased_cor_result['aic_diff'],ndigits=2) == 1.98
    
    def test_comp3(self):
        assert round(self.unbiased_uncor_result['Res. DOF'],ndigits=3) == 0.981
    
    def test_comp4(self):
        assert round(self.unbiased_uncor_result['aic_diff'],ndigits=3) == 1.945
    
    def test_comp5(self):
        assert round(self.cor_result['Res. DOF'],ndigits=1) == 1.0
    
    def test_comp6(self):
        assert round(self.cor_result['aic_diff'],ndigits=2) == 1.98
    
    def test_comp7(self):
        assert round(self.uncor_result['Res. DOF'],ndigits=3) == 0.983
    
    def test_comp8(self):
        assert round(self.uncor_result['aic_diff'],ndigits=3) == 1.945

class Test_model_comparison2_hard:

    # Model comparison and smoothness uncertainty correction tests for gamlss

    # Simulate some data - effect of x0 on scale parameter is very very small
    sim_fit_dat = sim8(n=500,c=0.1,seed=37)

    # Now fit nested models

    # Create Gaussian GAMMLSS family with identity link for mean
    # and log link for sigma
    family = GAUMLSS([Identity(),LOG()])

    # We need to model the mean: \mu_i = \alpha + f(x0)
    sim1_formula_m = Formula(lhs("y"),
                        [i(),f(["x0"],nk=15)],
                        data=sim_fit_dat)

    # and the standard deviation as well: log(\sigma_i) = \alpha + f(x0)
    sim1_formula_sd = Formula(lhs("y"),
                        [i(),f(["x0"],nk=15)],
                        data=sim_fit_dat)

    # Collect both formulas
    sim1_formulas = [sim1_formula_m,sim1_formula_sd]

    sim_fit_model = GAMMLSS(sim1_formulas,family)
    sim_fit_model.fit()

    # We need to model the mean: \mu_i = \alpha + f(x0)
    sim2_formula_m = Formula(lhs("y"),
                        [i(),f(["x0"],nk=15)],
                        data=sim_fit_dat)

    # and the standard deviation as well: log(\sigma_i) = \alpha + f(x0)
    sim2_formula_sd = Formula(lhs("y"),
                        [i()],
                        data=sim_fit_dat)

    # Collect both formulas
    sim2_formulas = [sim2_formula_m,sim2_formula_sd]

    sim_fit_model2 = GAMMLSS(sim2_formulas,family)
    sim_fit_model2.fit()

    # And perform a couple of bias corrected / smoothness uncertainty corrected / or not / comparisons...
    uncor_result = compare_CDL(sim_fit_model,sim_fit_model2,correct_V=False,correct_t1=False,grid='JJJ',seed=22,verbose=True)
    cor_result = compare_CDL(sim_fit_model,sim_fit_model2,nR=5,lR=100,correct_t1=False,n_c=1,grid='JJJ',seed=22,verbose=True)

    def test_comp1(self):
        assert round(self.cor_result['Res. DOF'],ndigits=1) == 1.0
    
    def test_comp2(self):
        assert round(self.cor_result['aic_diff'],ndigits=1) == 1.3
    
    def test_comp3(self):
        assert round(self.uncor_result['Res. DOF'],ndigits=1) == 1.1
    
    def test_comp4(self):
        assert round(self.uncor_result['aic_diff'],ndigits=1) == 1.4