from mssm.models import *
import numpy as np
import os
from mssmViz.sim import*
import io
from contextlib import redirect_stdout


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
    model.fit(extension_method_lam="nesterov",max_inner=50,min_inner=50)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 18.357 

    def test_GAMcoef(self):
        coef = self.model.overall_coef
        assert np.allclose(coef,np.array([[3.58432716], [-9.20690455], [-0.39012586], [4.3348436], [-2.83326231],
                                          [-4.68429511], [-1.9338996], [-4.14505816], [-6.70581232], [-4.72400822],
                                          [-5.19942876], [0.0202634], [-1.35759577], [1.53796412], [2.3935147],
                                          [2.33541735], [2.17494259], [2.35080211], [1.97973794], [1.44040268],
                                          [-0.96502725], [-4.28139732]])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(lam,np.array([0.009090817585653532, 1.0766512124458893])) 

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

        comp = "\nDistribution parameter: 1\n\nIntercept: 3.584, z: 58.435, P(|Z| > |z|): 0.000e+00 ***\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: .\n\nDistribution parameter: 2\n\nIntercept: 0.02, z: 0.64, P(|Z| > |z|): 0.52194\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: .\n"
        assert comp == capture
   
    def test_print_smooth_p(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.model.print_smooth_terms(p_values=True)
        capture = capture.getvalue()

        comp = "\nDistribution parameter: 1\n\nf(['x0']); edf: 9.799 chi^2: 5768.015 P(Chi^2 > chi^2) = 0.000e+00 ***\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n\nDistribution parameter: 2\n\nf(['x0']); edf: 6.559 chi^2: 563.181 P(Chi^2 > chi^2) = 0.000e+00 ***\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n"
        assert comp == capture

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
    model.fit(seed=30,max_outer=250,max_inner=500,extend_lambda=True,method="QR/Chol")

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 26.196

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(lam,np.array([1.00000000e+07, 6.04907747e+00, 5.14585161e-01, 9.12402158e-01, 5.38284225e+02])) 

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
    model.fit(extension_method_lam="nesterov",max_inner=50,min_inner=50)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 14.677 

    def test_GAMcoef(self):
        coef = self.model.overall_coef
        assert np.allclose(coef,np.array([[1.14440925], [-0.79166747], [1.9557505], [2.46973486], [1.77250272],
                                          [1.29047325], [1.89204542], [1.34016457], [0.24866229], [-0.66762524],
                                          [-1.69696779], [0.77159535], [-0.72927693], [0.87658351], [1.1296147],
                                          [1.22771936], [1.11208552], [1.27840078], [1.38507509], [0.72796546],
                                          [0.02456734], [-0.55658569]])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(lam,np.array([0.5160027621613626, 2.4947669751349153])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -923.897 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -892.042 

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
    sim_fit_model.fit(extension_method_lam="nesterov")

    # Test prediction code + whole-function correction
    pred_dat = pd.DataFrame({"x0":np.linspace(0,1,50)})
    pred,pred_mat,ci = sim_fit_model.predict(0,[0,1],pred_dat,ci=True,whole_interval=True,seed=20)

    def test_pred(self):
        assert np.allclose(self.pred,np.array([-0.01129542,  0.0531048 ,  0.4463374 ,  1.15070572,  2.14835157,
                                                3.39366362,  4.75023696,  6.06312929,  7.17849487,  8.00855263,
                                                8.56807817,  8.88068744,  8.96949571,  8.85007311,  8.53215388,
                                                8.02532054,  7.3468747 ,  6.5608052 ,  5.74856372,  4.99160176,
                                                4.35102118,  3.8323613 ,  3.43176547,  3.14526299,  2.96430126,
                                                2.874289  ,  2.86022253,  2.90658667,  2.99177275,  3.09016969,
                                                3.17609643,  3.22450626,  3.21352876,  3.12228557,  2.92997066,
                                                2.6329868 ,  2.26713483,  1.87370543,  1.49341185,  1.1505029 ,
                                                0.85077147,  0.59903317,  0.39932222,  0.24818955,  0.13802317,
                                                0.06116629,  0.01075998, -0.01673754, -0.02400615, -0.01355005]))
    
    def test_ci(self):
        assert np.allclose(self.ci,np.array([0.27244221, 0.11883514, 0.14574115, 0.20026641, 0.26017835,
                                                0.270213  , 0.31644592, 0.39886276, 0.43334754, 0.44005059,
                                                0.50910306, 0.59070528, 0.59325352, 0.57956982, 0.64501183,
                                                0.71519071, 0.69830337, 0.69173394, 0.76688309, 0.82712509,
                                                0.82197981, 0.87066179, 0.97886606, 1.01535658, 0.96101622,
                                                0.93916932, 0.96159673, 0.91492501, 0.81502391, 0.77612948,
                                                0.77939425, 0.71524746, 0.6455857 , 0.65969849, 0.67796858,
                                                0.61341883, 0.557598  , 0.5704057 , 0.56077995, 0.49619339,
                                                0.47606434, 0.50317868, 0.46318685, 0.36127314, 0.30226284,
                                                0.28951974, 0.22250383, 0.15336903, 0.10672492, 0.20063921]))
        

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
    sim_fit_model.fit(extension_method_lam="nesterov")

    pred_dat1 = pd.DataFrame({"x0":np.linspace(0,1,50),
                          "cond":["a" for _ in range(50)]})

    pred_dat2 = pd.DataFrame({"x0":np.linspace(0,1,50),
                          "cond":["b" for _ in range(50)]})
    
    diff,ci = sim_fit_model.predict_diff(pred_dat2,pred_dat1,1,[1],whole_interval=True,seed=20)

    def test_print_smooth_p(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.sim_fit_model.print_smooth_terms(p_values=True)
        capture = capture.getvalue()

        comp = "\nDistribution parameter: 1\n\nf(['x0']); edf: 7.08 chi^2: 457.33 P(Chi^2 > chi^2) = 0.000e+00 ***\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n\nDistribution parameter: 2\n\nf(['x0'],by=cond): a; edf: 1.0 chi^2: 0.056 P(Chi^2 > chi^2) = 0.81254\nf(['x0'],by=cond): b; edf: 2.787 chi^2: 14.369 P(Chi^2 > chi^2) = 0.00561 **\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n"
        assert comp == capture

    def test_diff(self):
        assert np.allclose(self.diff,np.array([-0.30117801, -0.26587442, -0.23097423, -0.1967106 , -0.16331669,
                                                -0.13102568, -0.10007071, -0.07067289, -0.04298253, -0.01712426,
                                                0.00677733,  0.02859766,  0.04821214,  0.06551219,  0.08052851,
                                                0.09336344,  0.10411991,  0.11290086,  0.11980921,  0.12494801,
                                                0.12842162,  0.1303354 ,  0.13079472,  0.12990495,  0.12777145,
                                                0.12449794,  0.12015084,  0.11475953,  0.10835182,  0.10095552,
                                                0.09259843,  0.08330852,  0.07312107,  0.06208137,  0.05023544,
                                                0.0376293 ,  0.02430896,  0.01032025, -0.00431221, -0.01960433,
                                                -0.03557657, -0.05224941, -0.06964331, -0.0877787 , -0.10664652,
                                                -0.12615798, -0.14621095, -0.16670331, -0.18753294, -0.20859769]))
    
    def test_ci(self):
        assert np.allclose(self.ci,np.array([0.34623094, 0.30910642, 0.27593885, 0.24704208, 0.22264827,
                                                0.20269259, 0.18668176, 0.17384721, 0.16379334, 0.15638265,
                                                0.15134584, 0.14809469, 0.14575006, 0.14345542, 0.14115   ,
                                                0.13922699, 0.13791861, 0.13708189, 0.13620609, 0.1346832 ,
                                                0.1326449 , 0.13076704, 0.12952789, 0.12895726, 0.12862426,
                                                0.12789564, 0.1267939 , 0.12601816, 0.12609362, 0.12708147,
                                                0.12856267, 0.1298893 , 0.13094167, 0.13231571, 0.13447648,
                                                0.13747628, 0.14095995, 0.1444003 , 0.14771447, 0.15157863,
                                                0.15667533, 0.16342272, 0.1719795 , 0.18244157, 0.19525272,
                                                0.21135893, 0.23149658, 0.25596544, 0.28471545, 0.31756041]))