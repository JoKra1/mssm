from mssm.models import *
from mssm.src.python.compare import compare_CDL
from mssm.src.python.gamm_solvers import compute_S_emb_pinv_det,cpp_dChol,cpp_chol
from mssm.src.python.utils import estimateVp,correct_VB
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
    uncor_result = compare_CDL(sim_fit_model,sim_fit_model2,correct_V=False,correct_t1=False,grid='JJJ3',seed=22)
    cor_result = compare_CDL(sim_fit_model,sim_fit_model2,nR=20,lR=100,correct_t1=False,n_c=1,grid='JJJ3',seed=22)

    unbiased_uncor_result = compare_CDL(sim_fit_model,sim_fit_model2,correct_V=False,correct_t1=True,grid='JJJ3',seed=22)
    unbiased_cor_result = compare_CDL(sim_fit_model,sim_fit_model2,nR=20,lR=100,correct_t1=True,n_c=2,grid='JJJ3',seed=22)

    def test_comp1(self):
        assert round(self.unbiased_cor_result['Res. DOF'],ndigits=1) == 1.0
    
    def test_comp2(self):
        assert round(self.unbiased_cor_result['aic_diff'],ndigits=2) == 2.04
    
    def test_comp3(self):
        assert round(self.unbiased_uncor_result['Res. DOF'],ndigits=3) == 0.981
    
    def test_comp4(self):
        assert round(self.unbiased_uncor_result['aic_diff'],ndigits=3) == 1.945
    
    def test_comp5(self):
        assert round(self.cor_result['Res. DOF'],ndigits=1) == 1.0
    
    def test_comp6(self):
        assert round(self.cor_result['aic_diff'],ndigits=2) == 2.04
    
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
    sim_fit_model.fit(extension_method_lam="nesterov")

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
    sim_fit_model2.fit(extension_method_lam="nesterov")

    # And perform a couple of bias corrected / smoothness uncertainty corrected / or not / comparisons...
    uncor_result = compare_CDL(sim_fit_model,sim_fit_model2,correct_V=False,correct_t1=False,grid='JJJ3',seed=22,verbose=True)
    cor_result = compare_CDL(sim_fit_model,sim_fit_model2,nR=5,lR=100,correct_t1=False,n_c=1,grid='JJJ3',seed=22,verbose=True)

    def test_comp1(self):
        assert round(self.cor_result['Res. DOF'],ndigits=1) == 1.1
    
    def test_comp2(self):
        assert round(self.cor_result['aic_diff'],ndigits=1) == 2.1
    
    def test_comp3(self):
        assert round(self.uncor_result['Res. DOF'],ndigits=1) == 1.0
    
    def test_comp4(self):
        assert round(self.uncor_result['aic_diff'],ndigits=1) == 1.9

class Test_Vb_corrections:
    # Simulate some data - effect of x0 is very very small
    sim_fit_dat = sim3(n=500,scale=2,c=1,family=Gaussian(),seed=21)

    # Now fit nested models
    sim_fit_formula = Formula(lhs("y"),
                                [i(),f(["x0"],nk=20,rp=0),f(["x1"],nk=20,rp=0),f(["x2"],nk=20,rp=0),f(["x3"],nk=20,rp=0)],
                                data=sim_fit_dat,
                                print_warn=False)

    model = GAMM(sim_fit_formula,Gaussian())
    model.fit(exclude_lambda=False,progress_bar=False,maxiter=100)

    # Now fit nested models
    sim_fit_formula2 = Formula(lhs("y"),
                                [i(),f(["x0"],nk=10,rp=0),f(["x1"],nk=10,rp=0),f(["x2"],nk=10,rp=0)],
                                data=sim_fit_dat,
                                print_warn=False)

    model2 = GAMM(sim_fit_formula2,Gaussian())
    model2.fit(exclude_lambda=False,progress_bar=False,maxiter=100)

    Vp1,_,_,_ = estimateVp(model,strategy="JJJ2",verbose=True,seed=20)

    _,_,Vp2,_,_,total_edf,_,_,_ = correct_VB(model,grid_type="JJJ1",nR=20,verbose=True,V_shrinkage_weight=0.75,b=1e7,seed=20,df=2)

    _,LI,_,_,_,total_edf2,_,_,upper_edf = correct_VB(model2,grid_type="JJJ2",nR=20,verbose=True,V_shrinkage_weight=1,b=1e7,seed=20,df=2)

    def test_Vp1(self):
        assert np.allclose(np.round(self.Vp1,decimals=3),np.array([[ 1.138000e+00, -5.000000e-03,  1.700000e-02, -1.080000e-01],
                                                                    [-5.000000e-03,  8.980000e-01,  2.300000e-02, -7.000000e-03],
                                                                    [ 1.700000e-02,  2.300000e-02,  4.670000e-01,  1.050000e-01],
                                                                    [-1.080000e-01, -7.000000e-03,  1.050000e-01,  5.955928e+03]]))
    
    def test_Vp2(self):
        assert np.allclose(self.Vp1,self.Vp2)
    
    def test_edf1(self):
        assert np.round(self.total_edf,decimals=3) == 22.195

    def test_edf2(self):
        assert np.round(self.total_edf2,decimals=3) == 17.284

    def test_edf3(self):
        assert np.round(self.upper_edf,decimals=3) == 17.269

class Test_chol_deriv:
    # Test of derivative of cholesky factor computation

    # Simulate some data - effect of x0 is very very small
    sim_fit_dat = sim3(n=500,scale=2,c=0.1,family=Gaussian(),seed=21)

    # Now fit nested models
    sim_fit_formula = Formula(lhs("y"),
                                [i(),f(["x0"],nk=20,rp=1),f(["x1"],nk=20,rp=1),f(["x2"],nk=20,rp=1),f(["x3"],nk=20,rp=1)],
                                data=sim_fit_dat,
                                print_warn=False)

    sim_fit_model = GAMM(sim_fit_formula,Gaussian())
    sim_fit_model.fit(exclude_lambda=False,progress_bar=False,maxiter=100)

    # Compute hessian of negative penalized llk
    S_emb,_,_,_ = compute_S_emb_pinv_det(sim_fit_model.hessian.shape[1],sim_fit_model.formula.penalties,"svd")
    H = -sim_fit_model.hessian + S_emb
    LP,code = cpp_chol(H)

    # Handle cpp side
    dDat, dRow, dcol = cpp_dChol(LP.T,sim_fit_model.formula.penalties[0].S_J_emb.tocsr())

    dDat = [d for r in dDat for d in r]
    dcol = [d for r in dcol for d in r]

    dChol = scp.sparse.csr_array((dDat,(dRow,dcol)),shape=H.shape)
    dChol.eliminate_zeros()

    # Below implements equation in Supp. materials section D from Wood, Pya, & Säfken (2016) in python for dense arrays and compares that against
    # the cpp implementation that works with sparse matrices.
    dChol2 = np.zeros(H.shape)
    A = sim_fit_model.formula.penalties[0].S_J_emb.toarray()
    R = LP.T.toarray()

    for i in range(H.shape[1]):
        Rii = 0
        dRii = 0
        for j in range(i,H.shape[1]):
            
            Bij = A[i,j]
            if i > 0:
                k = 0
                while k < i:
                    Bij -= ((dChol2[k,i]*R[k,j]) + (R[k,i]*dChol2[k,j]))
                    k += 1
                
            if i == j:
                Rii = R[i,i]
                dRii = 0.5*Bij/Rii
                dChol2[i,j] = dRii
            elif j > i:
                dChol2[i,j] = (Bij - (R[i,j] * dRii))/Rii
                
    def test_dChol(self):
        assert np.max(np.abs(self.dChol.toarray() - self.dChol2)) == 0