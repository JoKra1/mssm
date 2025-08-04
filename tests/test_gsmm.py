import mssm
from mssm.models import *
import numpy as np
import os
from mssmViz.sim import*
from .defaults import default_gsmm_test_kwargs,max_atol,max_rtol,init_penalties_tests_gammlss,init_penalties_tests_gsmm

mssm.src.python.exp_fam.GAUMLSS.init_lambda = init_penalties_tests_gammlss
mssm.src.python.exp_fam.GAMMALS.init_lambda = init_penalties_tests_gammlss
mssm.src.python.exp_fam.MULNOMLSS.init_lambda = init_penalties_tests_gammlss
mssm.src.python.exp_fam.PropHaz.init_lambda = init_penalties_tests_gsmm
mssm.src.python.utils.GAMLSSGSMMFamily.init_lambda = init_penalties_tests_gsmm

################################################################## Tests ##################################################################

class Test_GAUMLSSGEN_hard:

    # Simulate 500 data points
    sim_dat = sim3(500,2,c=1,seed=0,family=Gaussian(),binom_offset = 0, correlate=False)

    # We need to model the mean: \mu_i
    formula_m = Formula(lhs("y"),
                        [i(),f(["x0"]),f(["x1"]),f(["x2"]),f(["x3"])],
                        data=sim_dat)

    # And for sd - here constant
    formula_sd = Formula(lhs("y"),
                        [i()],
                        data=sim_dat)

    # Collect both formulas
    formulas = [formula_m,formula_sd]
    links = [Identity(),LOG()]

    # Now define the general family + model and fit!
    gsmm_fam = GAMLSSGSMMFamily(2,GAUMLSS(links))
    model = GSMM(formulas=formulas,family=gsmm_fam)

    # First fit with SR1
    bfgs_opt={"gtol":1e-9,
            "ftol":1e-9,
            "maxcor":30,
            "maxls":200,
            "maxfun":1e7}
    
    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["method"] = 'qEFS'
    test_kwargs["extend_lambda"] = False
    test_kwargs["control_lambda"] = 1
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["min_inner"] = 500
    test_kwargs["seed"] = 0
    test_kwargs["max_restarts"] = 5
    test_kwargs["overwrite_coef"] = False
    test_kwargs["qEFS_init_converge"] = False
    test_kwargs["prefit_grad"] = True
    test_kwargs["bfgs_options"] = bfgs_opt

    model.fit(**test_kwargs)

    model2 = copy.deepcopy(model)
    test_kwargs["qEFSH"] = 'BFGS'
    test_kwargs["max_restarts"] = 0
    test_kwargs["overwrite_coef"] = True
    test_kwargs["qEFS_init_converge"] = True
    test_kwargs["prefit_grad"] = False

    # Now fit with BFGS
    model2.fit(**test_kwargs)

    def test_GAMedf(self):
        np.testing.assert_allclose(self.model.edf,19.06605212367808,atol=min(max_atol,0.5),rtol=min(max_rtol,0.025))

    def test_GAMedf2(self):
        np.testing.assert_allclose(self.model2.edf,20.085993154912806,atol=min(max_atol,1.3),rtol=min(max_rtol,0.07))

    def test_GAMcoef(self):
        coef = self.model.coef.flatten()
        np.testing.assert_allclose(coef,np.array([ 7.64896332, -0.87163571, -0.84714219,  0.05745359,  1.19376365,
                                                1.65146764,  0.45272538, -0.54724611, -0.76214353, -0.76137743,
                                                -1.8210173 , -1.04816578, -0.41222123,  0.34134332,  1.0342988 ,
                                                2.01876926,  3.18538175,  4.08327798,  5.17126124, -8.62853231,
                                                4.78466146,  5.76753761, -1.98400143,  0.10081098, -1.16777336,
                                                -3.57420725, -4.0600672 , -1.08798903, -0.04889937, -0.01052461,
                                                0.01277757,  0.02972515,  0.04714203,  0.06622719,  0.08729984,
                                                0.0969056 ,  0.10180631,  0.66763451]),atol=min(max_atol,0.7),rtol=min(max_rtol,1e-6))

    def test_GAMcoef2(self):
        coef = self.model2.coef.flatten()
        np.testing.assert_allclose(coef,np.array([ 7.64898791, -0.87442422, -0.7302071 ,  0.14596979,  1.27647368,
                                                1.70078317,  0.56790503, -0.41554354, -0.73983263, -0.90265256,
                                                -1.83369152, -1.02612303, -0.38474768,  0.35920819,  1.05915048,
                                                2.05300297,  3.20549805,  4.06563624,  5.12037278, -8.52043318,
                                                4.84751426,  5.85726955, -1.89210962,  0.16967413, -1.06785945,
                                                -3.53406784, -3.88453386, -2.01468058, -0.04870832, -0.01048352,
                                                0.0127276 ,  0.02960897,  0.04695781,  0.06596845,  0.08695881,
                                                0.09652709,  0.10140869,  0.66877099]),atol=min(max_atol,0.3),rtol=min(max_rtol,1e-6))  

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        np.testing.assert_allclose(lam,np.array([7.35504400e-01, 7.27801598e+00, 8.04428324e-03, 1.00000000e+07]),atol=min(max_atol,1.5),rtol=min(max_rtol,0.37))

    def test_GAMlam2(self):
        lam = np.array([p.lam for p in self.model2.overall_penalties])
        np.testing.assert_allclose(lam,np.array([9.58110974e-01, 8.39545471e+00, 8.87952624e-03, 1.00000000e+07]),atol=min(max_atol,15),rtol=min(max_rtol,0.35))  

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(reml,-1080.9668585544648,atol=min(max_atol,2),rtol=min(max_rtol,1e-6))

    def test_GAMreml2(self):
        reml = self.model2.get_reml()
        np.testing.assert_allclose(reml,-1083.887431273825,atol=min(max_atol,2),rtol=min(max_rtol,0.015))

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(llk,-1043.3141308788472,atol=min(max_atol,0.5),rtol=min(max_rtol,0.001))
    
    def test_GAMllk2(self):
        llk = self.model2.get_llk(False)
        np.testing.assert_allclose(llk,-1043.8544317953304,atol=min(max_atol,0.5),rtol=min(max_rtol,0.002))


class Test_PropHaz_hard:
    sim_dat = sim3(500,2,c=1,seed=0,family=PropHaz([0],[0]),binom_offset = 0.1,correlate=False)
        
    # Prep everything for prophaz model
    sim_dat = sim_dat.sort_values(['y'],ascending=[False])
    sim_dat = sim_dat.reset_index(drop=True)
    #print(sim_dat.head(),np.mean(sim_dat["delta"]))

    u,inv = np.unique(sim_dat["y"],return_inverse=True)
    ut = np.flip(u)
    r = np.abs(inv - max(inv))

    # We only need to model the mean: \mu_i
    sim_formula_m = Formula(lhs("delta"),
                        [f(["x0"]),f(["x1"]),f(["x2"]),f(["x3"])],
                        data=sim_dat)

    # Fit with Newton
    gsmm_newton_fam = PropHaz(ut,r)
    gsmm_newton = GSMM([copy.deepcopy(sim_formula_m)],gsmm_newton_fam)

    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["method"] = 'QR/Chol'
    test_kwargs["extend_lambda"] = False
    test_kwargs["control_lambda"] = False
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["min_inner"] = 500
    test_kwargs["seed"] = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gsmm_newton.fit(**test_kwargs)
        
    gsmm_qefs_fam = PropHaz(ut,r)
    gsmm_qefs = GSMM([copy.deepcopy(sim_formula_m)],gsmm_qefs_fam)

    # Fit with qEFS update without initialization
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bfgs_opt={"gtol":1e-7,
                    "ftol":1e-7,
                    "maxcor":30,
                    "maxls":20,
                    "maxfun":100}
        
        test_kwargs["method"] = 'qEFS'
        test_kwargs["extend_lambda"] = False
        test_kwargs["control_lambda"] = 1
        test_kwargs["max_outer"] = 200
        test_kwargs["max_inner"] = 500
        test_kwargs["min_inner"] = 500
        test_kwargs["seed"] = 0
        test_kwargs["max_restarts"] = 0
        test_kwargs["overwrite_coef"] = False
        test_kwargs["qEFS_init_converge"] = False
        test_kwargs["prefit_grad"] = True
        test_kwargs["bfgs_options"] = bfgs_opt
        
        gsmm_qefs.fit(**test_kwargs)

    def test_GAMcoef(self):

        np.testing.assert_allclose((self.gsmm_newton.coef - self.gsmm_qefs.coef).flatten(),
                           np.array([-5.44700213e-04, -5.57025326e-04,  1.64013870e-04,  8.67670135e-04,
                                    4.92493649e-04,  7.65198038e-04,  5.14628775e-04, -9.50230485e-04,
                                    -2.48488958e-03, -6.72884363e-04, -1.17596358e-03, -6.75266364e-04,
                                    6.72783782e-05,  1.07678488e-04, -2.63805422e-07,  1.65661496e-03,
                                    -1.61770568e-03, -6.64843399e-03, -4.78166505e-02, -3.54112498e-02,
                                    -4.53015190e-02, -3.99937127e-02, -4.29717381e-02, -4.15610992e-02,
                                    -4.73157234e-02, -1.93293685e-02, -2.61343991e-02,  5.38161813e-05,
                                    -3.18588497e-04, -3.70497505e-04,  7.79857596e-05,  5.19593795e-04,
                                    5.72184426e-04, -2.22179748e-04, -1.67729712e-03, -3.14518797e-03]),atol=0.025) 

    def test_GAMlam(self):
        np.testing.assert_allclose(np.round(np.array([p1.lam - p2.lam for p1,p2 in zip(self.gsmm_newton.overall_penalties,self.gsmm_qefs.overall_penalties)]),decimals=3),
                           np.array([-0.098,  0.149, -0.   ,  2.152]),atol=min(max_atol,2),rtol=min(max_rtol,0.35)) 

    def test_GAMreml(self):
        np.testing.assert_allclose(self.gsmm_newton.get_reml() - self.gsmm_qefs.get_reml(),0.0907684010567209,atol=min(max_atol,0.2),rtol=min(max_rtol,4e-4))

    def test_GAMllk(self):
        np.testing.assert_allclose(self.gsmm_newton.get_llk(True) - self.gsmm_qefs.get_llk(True),0.07796569660399655,atol=min(max_atol,0.1),rtol=min(max_rtol,9e-4))

class Test_PropHaz_repara_hard:
    sim_dat = sim3(500,2,c=1,seed=0,family=PropHaz([0],[0]),binom_offset = 0.1,correlate=False)
        
    # Prep everything for prophaz model
    sim_dat = sim_dat.sort_values(['y'],ascending=[False])
    sim_dat = sim_dat.reset_index(drop=True)
    #print(sim_dat.head(),np.mean(sim_dat["delta"]))

    u,inv = np.unique(sim_dat["y"],return_inverse=True)
    ut = np.flip(u)
    r = np.abs(inv - max(inv))

    # We only need to model the mean: \mu_i
    sim_formula_m = Formula(lhs("delta"),
                        [f(["x0"]),f(["x1"]),f(["x2"]),f(["x3"])],
                        data=sim_dat)

    # Fit with Newton
    gsmm_newton_fam = PropHaz(ut,r)
    gsmm_newton = GSMM([copy.deepcopy(sim_formula_m)],gsmm_newton_fam)

    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["method"] = 'QR/Chol'
    test_kwargs["extend_lambda"] = False
    test_kwargs["control_lambda"] = False
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["min_inner"] = 500
    test_kwargs["seed"] = 0
    test_kwargs["repara"] = True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gsmm_newton.fit(**test_kwargs)
        
    gsmm_qefs_fam = PropHaz(ut,r)
    gsmm_qefs = GSMM([copy.deepcopy(sim_formula_m)],gsmm_qefs_fam)

    # Fit with qEFS update without initialization
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bfgs_opt={"gtol":1e-7,
                    "ftol":1e-7,
                    "maxcor":30,
                    "maxls":20,
                    "maxfun":100}
        
        test_kwargs["method"] = 'qEFS'
        test_kwargs["extend_lambda"] = False
        test_kwargs["control_lambda"] = 1
        test_kwargs["max_outer"] = 200
        test_kwargs["max_inner"] = 500
        test_kwargs["min_inner"] = 500
        test_kwargs["seed"] = 0
        test_kwargs["max_restarts"] = 0
        test_kwargs["overwrite_coef"] = False
        test_kwargs["qEFS_init_converge"] = False
        test_kwargs["prefit_grad"] = True
        test_kwargs["repara"] = True
        test_kwargs["bfgs_options"] = bfgs_opt
        
        gsmm_qefs.fit(**test_kwargs)

    def test_GAMcoef(self):

        np.testing.assert_allclose((self.gsmm_newton.coef - self.gsmm_qefs.coef).flatten(),
                           np.array([-7.63719208e-05, -2.33388059e-03, -2.08491805e-03,  5.08375257e-04,
                                    1.83801694e-03, -2.16724620e-04, -2.62911257e-03, -3.04387685e-03,
                                    -3.04146788e-03, -4.36797594e-03, -5.87605352e-05,  3.44683488e-03,
                                    4.97359403e-03,  4.08998852e-03,  3.61352707e-03,  5.51083558e-03,
                                    -2.76692609e-06, -6.36423104e-03, -7.39506193e-02, -5.38276772e-02,
                                    -8.00034940e-02, -5.66388675e-02, -7.02723612e-02, -7.13573512e-02,
                                    -7.28973942e-02, -2.79028844e-02, -1.82273956e-02,  2.95831797e-03,
                                    -1.53464860e-02, -2.04331998e-02, -6.47717863e-03,  1.15027597e-02,
                                    1.73769619e-02, -3.68834284e-03, -4.49034467e-02, -8.78434801e-02]),atol=min(max_atol,0.02),rtol=min(max_rtol,5e-6)) 

    def test_GAMlam(self):
        np.testing.assert_allclose(np.array([p1.lam - p2.lam for p1,p2 in zip(self.gsmm_newton.overall_penalties,self.gsmm_qefs.overall_penalties)]),
                                   np.array([ 3.64040108e-02,  1.58535686e-01, -2.20926595e-04,  4.33667703e+01]),atol=min(max_atol,5),rtol=min(max_rtol,1.1)) 

    def test_GAMreml(self):
        np.testing.assert_allclose(self.gsmm_newton.get_reml() - self.gsmm_qefs.get_reml(),1.4408541227944625,atol=min(max_atol,0.5),rtol=min(max_rtol,6e-4))

    def test_GAMllk(self):
        np.testing.assert_allclose(self.gsmm_newton.get_llk(True) - self.gsmm_qefs.get_llk(True),-0.18673595259474496,atol=min(max_atol,0.1),rtol=min(max_rtol,2e-4))


class Test_drop:
    sim_dat = sim13(5000,2,c=0,seed=0,family=Gaussian(),binom_offset = 0,n_ranef=20)

    formula = Formula(lhs("y"),
                    [i(),l(["x5"]),l(["x6"]),f(["x0"],by="x5"),f(["x0"],by="x6"),fs(["x0"],rf="x4")],
                    data=sim_dat)

    formula_sd = Formula(lhs("y"),
                        [i()],
                        data=sim_dat)

    model = GSMM([formula,formula_sd],GAMLSSGSMMFamily(2,GAUMLSS([Identity(),LOGb(-0.001)])))

    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["max_inner"] = 500
    test_kwargs["min_inner"] = 500
    test_kwargs["max_outer"] = 200
    test_kwargs["extend_lambda"] = False
    test_kwargs["control_lambda"] = 2
    test_kwargs["repara"] = False
    test_kwargs["method"] = "LU/Chol"
    model.fit(**test_kwargs)

    def test_GAMedf_hard(self):
        np.testing.assert_allclose(self.model.edf,108.54783796715027,atol=min(max_atol,0),rtol=min(max_rtol,0.03))

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        np.testing.assert_allclose(lam,np.array([4.913881081300338, 5.087832012717992, 0.002924222705631079, 10000000.0, 1.2211531005561156, 1.1056775054652457]),atol=min(max_atol,0),rtol=min(max_rtol,1.5)) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(reml,-10648.977146470883,atol=min(max_atol,0),rtol=min(max_rtol,0.01)) 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(llk,-10435.443686415718,atol=min(max_atol,0),rtol=min(max_rtol,0.01))

    def test_drop(self):
        assert len(self.model.info.dropped) == 1