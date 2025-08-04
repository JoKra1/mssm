import mssm
from mssm.models import *
import numpy as np
import os
from mssmViz.sim import*
import io
from contextlib import redirect_stdout
from .defaults import default_gamm_test_kwargs,default_gammlss_test_kwargs,max_atol,max_rtol,init_coef_gaumlss_tests,init_coef_gammals_tests,init_penalties_tests_gammlss

mssm.src.python.exp_fam.GAUMLSS.init_coef = init_coef_gaumlss_tests
mssm.src.python.exp_fam.GAMMALS.init_coef = init_coef_gammals_tests
mssm.src.python.exp_fam.GAUMLSS.init_lambda = init_penalties_tests_gammlss
mssm.src.python.exp_fam.GAMMALS.init_lambda = init_penalties_tests_gammlss
mssm.src.python.exp_fam.MULNOMLSS.init_lambda = init_penalties_tests_gammlss

################################################################## Tests ##################################################################

class Test_GAUMLS:
    # Simulate 500 data points
    GAUMLSSDat = sim6(500,seed=20)

    # We need to model the mean: \mu_i = \alpha + f(x0)
    formula_m = Formula(lhs("y"),
                        [i(),f(["x0"],nk=10)],
                        data=GAUMLSSDat)

    # and the standard deviation as well: log(\sigma_i) = \alpha + f(x0)
    formula_sd = Formula(lhs("y"),
                        [i(),f(["x0"],nk=10)],
                        data=GAUMLSSDat)

    # Collect both formulas
    formulas = [formula_m,formula_sd]

    # Create Gaussian GAMMLSS family with identity link for mean
    # and log link for sigma
    family = GAUMLSS([Identity(),LOG()])

    # Now define the model and fit!
    model = GAMMLSS(formulas,family)

    test_kwargs = copy.deepcopy(default_gammlss_test_kwargs)
    test_kwargs["extension_method_lam"] = "nesterov"
    test_kwargs["max_inner"] = 50
    test_kwargs["min_inner"] = 50

    model.fit(**test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 18.358

    def test_GAMcoef(self):
        coef = self.model.coef.flatten()
        assert np.allclose(coef,np.array([ 3.58432714, -9.20689268, -0.3901218 ,  4.33485776, -2.83325345,
                                            -4.68428463, -1.93389004, -4.14504997, -6.70579839, -4.7240103 ,
                                            -5.19939664,  0.02026291, -1.35759472,  1.5379936 ,  2.39353569,
                                            2.33544107,  2.1749611 ,  2.3508268 ,  1.97975281,  1.4404345 ,
                                            -0.96502575, -4.28148962])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(lam,np.array([0.00909086, 1.07657659])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -772.602 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -719.601 
    
    def test_print_smooth(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.model.print_smooth_terms()
        capture = capture.getvalue()

        comp = "\nDistribution parameter: 1\n\nf(['x0']); edf: 9.799\n\nDistribution parameter: 2\n\nf(['x0']); edf: 6.559\n"
        assert comp == capture

    def test_print_parametric(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.model.print_parametric_terms()
        capture = capture.getvalue()

        comp = "\nDistribution parameter: 1\n\nIntercept: 3.584, z: 58.435, P(|Z| > |z|): 0.000e+00 ***\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: .\n\nDistribution parameter: 2\n\nIntercept: 0.02, z: 0.64, P(|Z| > |z|): 0.52195\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: .\n"
        assert comp == capture
   
    def test_print_smooth_p_hard(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.model.print_smooth_terms(p_values=True)
        capture = capture.getvalue()

        comp1 = "\nDistribution parameter: 1\n\nf(['x0']); edf: 9.799 chi^2: 5696.894 P(Chi^2 > chi^2) = 0.000e+00 ***\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n\nDistribution parameter: 2\n\nf(['x0']); edf: 6.559 chi^2: 569.167 P(Chi^2 > chi^2) = 0.000e+00 ***\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n"
        comp2 = "\nDistribution parameter: 1\n\nf(['x0']); edf: 9.799 chi^2: 5696.894 P(Chi^2 > chi^2) = 0.000e+00 ***\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n\nDistribution parameter: 2\n\nf(['x0']); edf: 6.559 chi^2: 571.891 P(Chi^2 > chi^2) = 0.000e+00 ***\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n"
        assert (comp1 == capture) or (comp2 == capture)

class Test_GAUMLS_MIXED:
    ## Simulate some data - effect of x0 on scale parameter is very very small
    sim_fit_dat = sim10(n=500,c=0.1,seed=20)

    # Now fit nested models

    # Create Gaussian GAMMLSS family with identity link for mean
    # and log link for sigma
    family = GAUMLSS([Identity(),LOGb(-0.3)])

    # We need to model the mean: \mu_i = \alpha + f(x0)
    sim1_formula_m = Formula(lhs("y"),
                        [i(),f(["x0"]),f(["x1"]),ri("x4")],
                        data=sim_fit_dat)

    # and the standard deviation as well: log(\sigma_i) = \alpha + f(x0)
    sim1_formula_sd = Formula(lhs("y"),
                        [i(),f(["x2"]),f(["x3"])],
                        data=sim_fit_dat)

    # Collect both formulas
    sim1_formulas = [sim1_formula_m,sim1_formula_sd]

    model = GAMMLSS(sim1_formulas,family)

    test_kwargs = copy.deepcopy(default_gammlss_test_kwargs)
    test_kwargs["max_inner"] = 500
    test_kwargs["max_outer"] = 250
    test_kwargs["seed"] = 30
    test_kwargs["method"] = "QR/Chol"

    model.fit(**test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 26.196

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(lam,np.array([1.00000000e+07, 6.04896788e+00, 5.14594390e-01, 9.12400516e-01,5.38283393e+02])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -1747.292

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -1715.506

class Test_GAMMALS:

    # Simulate 500 data points
    GAMMALSSDat = sim6(500,family=GAMMALS([LOG(),LOG()]),seed=10)

    # We need to model the mean: \mu_i = \alpha + f(x0)
    formula_m = Formula(lhs("y"),
                        [i(),f(["x0"],nk=10)],
                        data=GAMMALSSDat)

    # and the scale parameter as well: log(\scale_i) = \alpha + f(x0)
    formula_sd = Formula(lhs("y"),
                        [i(),f(["x0"],nk=10)],
                        data=GAMMALSSDat)

    # Collect both formulas
    formulas = [formula_m,formula_sd]

    # Create Gamma GAMMLSS family with log link for mean
    # and log link for sigma
    family = GAMMALS([LOG(),LOG()])

    # Now define the model and fit!
    model = GAMMLSS(formulas,family)

    test_kwargs = copy.deepcopy(default_gammlss_test_kwargs)
    test_kwargs["extension_method_lam"] = "nesterov"
    test_kwargs["max_inner"] = 50
    test_kwargs["min_inner"] = 50

    model.fit(**test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 14.677 

    def test_GAMcoef(self):
        coef = self.model.coef.flatten()
        assert np.allclose(coef,np.array([ 1.14440894, -0.79166857,  1.95574948,  2.46973458,  1.7725019 ,
                                            1.29047048,  1.89204546,  1.34016392,  0.24865949, -0.66762416,
                                            -1.696964  ,  0.7715949 , -0.72928057,  0.87660987,  1.12963098,
                                            1.22774166,  1.1121008 ,  1.27841939,  1.38510189,  0.72797382,
                                            0.02458096, -0.55655912])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(lam,np.array([0.51599945, 2.4946028])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -923.897 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -892.042

class Test_mulnom:
    # Test multinomial model

    # We need to specify K-1 formulas - see the `MULNOMLSS` docstring for details.
    formulas = [Formula(lhs("y"),
                        [i(),f(["x0"])],
                        data=sim5(1000,seed=91)) for k in range(4)]

    # Create family - again specifying K-1 pars - here 4!
    family = MULNOMLSS(4)

    # Fit the model
    model = GAMMLSS(formulas,family)

    test_kwargs = copy.deepcopy(default_gammlss_test_kwargs)
    test_kwargs["control_lambda"] = False
    test_kwargs["extend_lambda"] = False
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["should_keep_drop"] = False

    model.fit(**test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 15.114

    def test_GAMcoef(self):
        coef = self.model.coef.flatten()
        assert np.allclose(coef,np.array([ 1.32831232, -0.70411299,  0.26038397,  0.8585649 ,  1.31636532,
                                            1.46064575,  1.15467552,  0.34090563, -0.96911006, -2.2518714 ,
                                            0.66532703, -0.76857181, -0.20872371,  0.11240685,  0.42769554,
                                            0.70741374,  1.07469267,  1.32593173,  1.37436662,  1.48506959,
                                            2.06156634, -1.63789937,  7.86561269,  9.15961217,  4.10367107,
                                            1.21906228,  1.68896079,  1.01054086, -0.31833491, -0.59758569,
                                            0.4881652 , -0.46708693, -0.12681357,  0.06836815,  0.25998144,
                                            0.42995532,  0.65313117,  0.80576918,  0.83513642,  0.90234416])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(lam,np.array([3.37059261e+00, 7.24169569e+04, 5.02852103e-02, 1.00000000e+07]))

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -1002.689
    
    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -980.676


class Test_mulnom_repara:
    # Test multinomial model

    # We need to specify K-1 formulas - see the `MULNOMLSS` docstring for details.
    formulas = [Formula(lhs("y"),
                        [i(),f(["x0"])],
                        data=sim5(1000,seed=91)) for k in range(4)]

    # Create family - again specifying K-1 pars - here 4!
    family = MULNOMLSS(4)

    # Fit the model
    model = GAMMLSS(formulas,family)

    test_kwargs = copy.deepcopy(default_gammlss_test_kwargs)
    test_kwargs["control_lambda"] = 0
    test_kwargs["extend_lambda"] = False
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["should_keep_drop"] = False
    test_kwargs["repara"] = True

    model.fit(**test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 15.114

    def test_GAMcoef(self):
        coef = self.model.coef.flatten()
        assert np.allclose(coef,np.array([ 1.32831232, -0.70411299,  0.26038397,  0.85856489,  1.31636532,
                                            1.46064575,  1.15467552,  0.34090563, -0.96911006, -2.2518714 ,
                                            0.66532703, -0.76857181, -0.20872371,  0.11240685,  0.42769554,
                                            0.70741373,  1.07469267,  1.32593173,  1.37436662,  1.48506958,
                                            2.06156634, -1.63789936,  7.86561269,  9.15961218,  4.10367108,
                                            1.21906229,  1.6889608 ,  1.01054087, -0.31833491, -0.5975857 ,
                                            0.4881652 , -0.46708693, -0.12681357,  0.06836815,  0.25998144,
                                            0.42995532,  0.65313117,  0.80576918,  0.83513642,  0.90234416])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(lam,np.array([3.37059261e+00, 7.24170552e+04, 5.02852109e-02, 1.00000000e+07]))

    def test_GAMreml_hard(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(reml,-1002.688991,atol=min(max_atol,2),rtol=min(max_rtol,0.002))
    
    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -980.676


class Test_te_p_values:
    sim_dat = sim3(n=500,scale=2,c=0,seed=20)
    
    formula_m = Formula(lhs("y"),
                        [i(),f(["x0","x3"],te=True,nk=9),f(["x1"])],
                        data=sim_dat)

    formula_sd = Formula(lhs("y"),
                        [i(),f(["x0"]),f(["x2"])],
                        data=sim_dat)

    # Collect both formulas
    formulas = [formula_m,formula_sd]

    # Create Gaussian GAMMLSS family with identity link for mean
    # and log link for sigma
    family = GAUMLSS([Identity(),LOGb(-0.001)])

    # Now define the model and fit!
    model = GAMMLSS(formulas,family)

    test_kwargs = copy.deepcopy(default_gammlss_test_kwargs)
    test_kwargs["control_lambda"] = 2
    test_kwargs["max_inner"] = 500
    test_kwargs["min_inner"] = 50

    model.fit(**test_kwargs)

    ps0, Trs0 = approx_smooth_p_values(model,par=0,edf1=False,force_approx=True)
    ps1, Trs1 = approx_smooth_p_values(model,par=1,edf1=False,force_approx=True)

    ps = [ps0,ps1]
    Trs = [Trs0,Trs1]

    def test_p1(self):
        np.testing.assert_allclose(self.ps[0],np.array([np.float64(0.39714921685433136), np.float64(0.0)]),atol=min(max_atol,0),rtol=min(max_rtol,1e-6))

    def test_p2_hard(self):
        np.testing.assert_allclose(self.ps[1],np.array([np.float64(0.738724988815144), np.float64(0.0)]),atol=min(max_atol,0),rtol=min(max_rtol,1e-5))

    def test_trs1(self):
        np.testing.assert_allclose(self.Trs[0],np.array([np.float64(5.67384667149778), np.float64(226.9137363518526)]),atol=min(max_atol,0),rtol=min(max_rtol,1e-6))

    def test_trs2(self):
        np.testing.assert_allclose(self.Trs[1],np.array([np.float64(0.11127579603700907), np.float64(135.27602836358722)]),atol=min(max_atol,0),rtol=min(max_rtol,1e-6))


class Test_drop:
    sim_dat = sim13(5000,2,c=0,seed=0,family=Gaussian(),binom_offset = 0,n_ranef=20)

    formula = Formula(lhs("y"),
                    [i(),l(["x5"]),l(["x6"]),f(["x0"],by="x5"),f(["x0"],by="x6"),fs(["x0"],rf="x4")],
                    data=sim_dat)

    formula_sd = Formula(lhs("y"),
                        [i()],
                        data=sim_dat)

    model = GAMMLSS([formula,formula_sd],GAUMLSS([Identity(),LOGb(-0.001)]))

    test_kwargs = copy.deepcopy(default_gammlss_test_kwargs)
    test_kwargs["max_inner"] = 500
    test_kwargs["min_inner"] = 500
    test_kwargs["max_outer"] = 200
    test_kwargs["extend_lambda"] = False
    test_kwargs["control_lambda"] = 2
    test_kwargs["method"] = "LU/Chol"
    model.fit(**test_kwargs)

    def test_GAMedf_hard(self):
        np.testing.assert_allclose(self.model.edf,108.54811953082891,atol=min(max_atol,0),rtol=min(max_rtol,0.03))

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        np.testing.assert_allclose(lam,np.array([4.913869938536273, 5.087829328785329, 0.0029242223796059627, 10000000.0, 1.2211364471705635, 1.105677472005988]),atol=min(max_atol,0),rtol=min(max_rtol,1.5)) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(reml,-10648.977191385402,atol=min(max_atol,0),rtol=min(max_rtol,0.01)) 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(llk,-10435.44335659411,atol=min(max_atol,0),rtol=min(max_rtol,0.01))
    
    def test_drop(self):
        assert len(self.model.info.dropped) == 1
     

class Test_pred_whole_func_cor:
    # Simulate some data
    sim_fit_dat = sim6(n=500,seed=37)

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
    
    test_kwargs = copy.deepcopy(default_gammlss_test_kwargs)
    test_kwargs["extension_method_lam"] = "nesterov"

    sim_fit_model.fit(**test_kwargs)

    # Test prediction code + whole-function correction
    pred_dat = pd.DataFrame({"x0":np.linspace(0,1,50)})
    pred,pred_mat,ci = sim_fit_model.predict([0,1],pred_dat,ci=True,whole_interval=True,seed=20,par=0)

    def test_pred(self):
        assert np.allclose(self.pred,np.array([-0.01138392,  0.05310652,  0.44636549,  1.15072365,  2.14835029,
                                                3.39365478,  4.75023087,  6.06313038,  7.17850177,  8.00856086,
                                                8.56808424,  8.88068926,  8.96949259,  8.8500654 ,  8.53214278,
                                                8.02530808,  7.34686356,  6.5607974 ,  5.74856022,  4.99160243,
                                                4.35102523,  3.83236853,  3.43177651,  3.14527921,  2.9643237 ,
                                                2.87431683,  2.86025298,  2.90661517,  2.99179483,  3.09018223,
                                                3.17609768,  3.22449593,  3.21350857,  3.12225943,  2.92994466,
                                                2.63296809,  2.26712914,  1.87371669,  1.49344223,  1.15055278,
                                                0.85083955,  0.5991164 ,  0.39941563,  0.24828419,  0.13810493,
                                                0.06121592,  0.01075972, -0.01677633, -0.0240358 , -0.01348688]))
    
    def test_ci(self):
        assert np.allclose(self.ci,np.array([0.27251711, 0.11885948, 0.14575675, 0.20026483, 0.26016534,
                                                0.27019453, 0.31642028, 0.39883243, 0.43331877, 0.44002722,
                                                0.50908004, 0.59067928, 0.5932289 , 0.57954789, 0.64498692,
                                                0.71516238, 0.69827735, 0.69170816, 0.76685076, 0.82708777,
                                                0.82194123, 0.87061588, 0.97880911, 1.01529647, 0.96096182,
                                                0.93911891, 0.96154628, 0.91487953, 0.8149885 , 0.7760991 ,
                                                0.77936308, 0.71521874, 0.6455595 , 0.65966908, 0.67793687,
                                                0.61339351, 0.55758266, 0.57039461, 0.56076992, 0.49618751,
                                                0.47606143, 0.50317253, 0.46317742, 0.36127176, 0.30228488,
                                                0.28955845, 0.22256101, 0.15345764, 0.10679976, 0.2008102 ]))
        

class Test_diff_whole_func_cor:
    # Difference prediction for gammlss

    # Simulate some data - effect of x0 on scale parameter is very very small in one condition
    sim_fit_dat1 = sim8(n=500,c=0.1,seed=37)
    sim_fit_dat1["cond"] = "a"
    sim_fit_dat2 = sim8(n=500,c=1,seed=41)
    sim_fit_dat2["cond"] = "b"
    sim_fit_dat = pd.concat((sim_fit_dat1,sim_fit_dat2),axis=0,ignore_index=True)

    # Now fit model

    # Create Gaussian GAMMLSS family with identity link for mean
    # and log link for sigma
    family = GAUMLSS([Identity(),LOG()])

    # We need to model the mean: \mu_i = \alpha + f(x0)
    sim1_formula_m = Formula(lhs("y"),
                        [i(),f(["x0"],nk=10)],
                        data=sim_fit_dat)

    # and the standard deviation as well: log(\sigma_i) = \alpha + f(x0)
    sim1_formula_sd = Formula(lhs("y"),
                        [i(),f(["x0"],nk=10,by="cond")],
                        data=sim_fit_dat)

    # Collect both formulas
    sim1_formulas = [sim1_formula_m,sim1_formula_sd]

    sim_fit_model = GAMMLSS(sim1_formulas,family)

    test_kwargs = copy.deepcopy(default_gammlss_test_kwargs)
    test_kwargs["extension_method_lam"] = "nesterov2"

    sim_fit_model.fit(**test_kwargs)

    pred_dat1 = pd.DataFrame({"x0":np.linspace(0,1,50),
                          "cond":["a" for _ in range(50)]})

    pred_dat2 = pd.DataFrame({"x0":np.linspace(0,1,50),
                          "cond":["b" for _ in range(50)]})
    
    diff,ci = sim_fit_model.predict_diff(pred_dat2,pred_dat1,[1],whole_interval=True,seed=20,par=1)

    def test_print_smooth_p_hard(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.sim_fit_model.print_smooth_terms(p_values=True)
        capture = capture.getvalue()

        comp1 = "\nDistribution parameter: 1\n\nf(['x0']); edf: 7.079 chi^2: 457.542 P(Chi^2 > chi^2) = 0.000e+00 ***\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n\nDistribution parameter: 2\n\nf(['x0'],by=cond): a; edf: 1.0 chi^2: 0.056 P(Chi^2 > chi^2) = 0.81334\nf(['x0'],by=cond): b; edf: 2.789 chi^2: 13.49 P(Chi^2 > chi^2) = 0.00585 **\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n"
        comp2 = "\nDistribution parameter: 1\n\nf(['x0']); edf: 7.079 chi^2: 457.185 P(Chi^2 > chi^2) = 0.000e+00 ***\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n\nDistribution parameter: 2\n\nf(['x0'],by=cond): a; edf: 1.0 chi^2: 0.056 P(Chi^2 > chi^2) = 0.81334\nf(['x0'],by=cond): b; edf: 2.789 chi^2: 13.49 P(Chi^2 > chi^2) = 0.00585 **\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n"
        assert (comp1 == capture) or (comp2 == capture)

    def test_diff(self):
        assert np.allclose(self.diff,np.array([-0.30135358, -0.26600859, -0.23106805, -0.19676569, -0.16333525,
                                                -0.13101046, -0.10002506, -0.07060065, -0.0428877 , -0.01701087,
                                                0.00690523,  0.02873598,  0.04835675,  0.06565901,  0.08067376,
                                                0.09350389,  0.10425282,  0.11302403,  0.11992095,  0.12504712,
                                                0.12850733,  0.13040725,  0.13085254,  0.12994891,  0.12780203,
                                                0.12451593,  0.12015716,  0.11475508,  0.10833746,  0.10093206,
                                                0.09256666,  0.0832692 ,  0.0730749 ,  0.06202901,  0.05017749,
                                                0.03756631,  0.02424142,  0.01024862, -0.00438762, -0.01968347,
                                                -0.03565969, -0.05233703, -0.06973624, -0.08787804, -0.10675354,
                                                -0.12627384, -0.14633665, -0.16683967, -0.1876806 , -0.20875715]))
    
    def test_ci(self):
        assert np.allclose(self.ci,np.array([0.34629246, 0.30914794, 0.27596426, 0.24705564, 0.22265442,
                                            0.20269553, 0.18668468, 0.1738519 , 0.16380075, 0.15639354,
                                            0.15136071, 0.14811353, 0.14577211, 0.14347928, 0.14117464,
                                            0.13925221, 0.13794476, 0.1371093 , 0.13623457, 0.13471181,
                                            0.13267291, 0.13079453, 0.12955546, 0.12898548, 0.12865308,
                                            0.12792413, 0.12682113, 0.12604402, 0.1261187 , 0.12710645,
                                            0.12858775, 0.12991388, 0.13096505, 0.13233796, 0.13449826,
                                            0.13749826, 0.14098219, 0.14442194, 0.14773426, 0.1515958 ,
                                            0.15668965, 0.16343419, 0.17198806, 0.1824472 , 0.19525605,
                                            0.21136223, 0.23150348, 0.25598021, 0.28474244, 0.31760382]))