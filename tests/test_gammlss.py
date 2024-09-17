from mssm.models import *
import numpy as np
import os
from mssmViz.sim import*


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
    model = GAMLSS(formulas,family)
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
    model = GAMLSS(formulas,family)
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