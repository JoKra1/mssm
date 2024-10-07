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
    model.fit()

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
    model.fit()

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
    sim_fit_model.fit()

    # Test prediction code + whole-function correction
    pred_dat = pd.DataFrame({"x0":np.linspace(0,1,50)})
    pred,pred_mat,ci = sim_fit_model.predict(0,[0,1],pred_dat,ci=True,whole_interval=True,seed=20)

    def test_pred(self):
        assert np.allclose(self.pred,np.array([-0.00766164,  0.05264753,  0.44488919,  1.15005394,  2.1489656 ,
                                            3.39499364,  4.75169151,  6.06427486,  7.17905639,  8.00844308,
                                            8.56744442,  8.87991424,  8.96917929,  8.85062296,  8.53348482,
                                            8.02684467,  7.34762305,  6.56016317,  5.7465462 ,  4.98885321,
                                            4.34855489,  3.83084749,  3.43141092,  3.145817  ,  2.96529597,
                                            2.87535612,  2.86111495,  2.90717605,  2.99202069,  3.09010881,
                                            3.17583   ,  3.22419774,  3.21334918,  3.12239713,  2.93052651,
                                            2.63407559,  2.26865388,  1.87534304,  1.4946562 ,  1.15089556,
                                            0.85019165,  0.59771271,  0.39780757,  0.24697709,  0.13735639,
                                            0.06103356,  0.01094003, -0.01648681, -0.02389859, -0.01377095]))
    
    def test_ci(self):
        assert np.allclose(self.ci,np.array([0.27384879, 0.11903284, 0.14551358, 0.1994852 , 0.25914081,
                                            0.26911891, 0.31556815, 0.39822631, 0.43291323, 0.44007542,
                                            0.50971105, 0.59158676, 0.59413523, 0.58059439, 0.6464563 ,
                                            0.71691553, 0.69990712, 0.6930809 , 0.76811388, 0.82810323,
                                            0.82220835, 0.8701253 , 0.97803617, 1.01439956, 0.95999927,
                                            0.93853541, 0.96163954, 0.91544704, 0.81592707, 0.77746659,
                                            0.7808667 , 0.71622337, 0.64568265, 0.65924267, 0.67749825,
                                            0.6135337 , 0.55950477, 0.57450552, 0.56600939, 0.50099435,
                                            0.47875031, 0.50366898, 0.46206757, 0.3579219 , 0.29646708,
                                            0.28369231, 0.21873725, 0.15284067, 0.10779222, 0.20600248]))