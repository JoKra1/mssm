import mssm
from mssm.models import *
import numpy as np
import os
import io
from contextlib import redirect_stdout
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
                           np.array([-6.00505655e-04, -9.64003472e-04, -3.83549187e-04,  1.17537678e-03,
                                    1.59927985e-03,  7.38051198e-04, -4.44997108e-04, -1.80074704e-03,
                                    -3.11562028e-03, -1.82326872e-03,  3.14446254e-03,  3.46507392e-03,
                                    2.31289573e-03,  2.99595003e-03,  3.18175605e-03,  1.01399062e-03,
                                    1.37963294e-02,  3.14823158e-02, -1.02728366e-01, -7.73625299e-02,
                                    -9.77585419e-02, -8.96668581e-02, -9.31593570e-02, -9.24209095e-02,
                                    -1.01698088e-01, -4.87830255e-02, -1.39815381e-02, -9.37696374e-05,
                                    -2.59179471e-03, -3.05352437e-03, -2.36146501e-04,  3.03758717e-03,
                                    3.90425367e-03, -7.14453093e-05, -7.63733239e-03, -1.53736159e-02]),atol=min(max_atol,0.05),rtol=min(max_rtol,5e-6)) 

    def test_GAMlam(self):
        np.testing.assert_allclose(np.array([p1.lam - p2.lam for p1,p2 in zip(self.gsmm_newton.overall_penalties,self.gsmm_qefs.overall_penalties)]),
                                   np.array([-9.87297096e-03, -5.29255787e-01, -2.50321169e-04,  1.10708062e+01]),atol=min(max_atol,5),rtol=min(max_rtol,1.1)) 

    def test_GAMreml(self):
        np.testing.assert_allclose(self.gsmm_newton.get_reml() - self.gsmm_qefs.get_reml(),2.9739911355409276,atol=min(max_atol,1),rtol=min(max_rtol,6e-4))

    def test_GAMllk(self):
        np.testing.assert_allclose(self.gsmm_newton.get_llk(True) - self.gsmm_qefs.get_llk(True),0.21546192785831408,atol=min(max_atol,0.2),rtol=min(max_rtol,2e-4))


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

class Test_shared:
    sim_dat = sim12(5000,c=0,seed=0,family=GAMMALS([LOG(),LOG()]),n_ranef=20)

    # We again need to model the mean: \mu_i = \alpha + f(x0) + f(x1) + f_{x4}(x0)
    sim_formula_m = Formula(lhs("y"),
                        [i(),f(["x0"],id=1),f(["x1"]),fs(["x0"],rf="x4")],
                        data=sim_dat)

    # and the standard deviation as well: log(\sigma_i) = \alpha + f(x2) + f(x3)
    sim_formula_sd = Formula(lhs("y"),
                        [i(),f(["x2"],id=1),f(["x3"])],
                        data=sim_dat)

    family = GAMMALS([LOG(),LOGb(-0.0001)])

    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["control_lambda"] = 1
    test_kwargs["extend_lambda"] = False
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["method"] = "QR/Chol"
    test_kwargs["repara"] = True
    test_kwargs["prefit_grad"] = True

    # Now define the model and fit!
    gsmm_fam = GAMLSSGSMMFamily(2,family)
    model = GSMM([sim_formula_m,sim_formula_sd],gsmm_fam)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(**test_kwargs)

    def test_GAMedf(self):
        np.testing.assert_allclose(self.model.edf,102.77775912832247,atol=min(max_atol,0),rtol=min(max_rtol,0.1)) 

    def test_GAMcoef(self):
        coef = self.model.coef
        np.testing.assert_allclose(coef,np.array([[2.16073927], [-0.30751272], [0.0080441], [0.41931181], [0.62922035],
                        [0.50293013], [0.4704501], [-0.0738919], [-0.41076225], [-0.67140247],
                        [-0.90969181], [-0.42080659], [-0.13732833], [0.13881868], [0.50286425],
                        [1.03349], [1.6793643], [2.30239738], [2.96343163], [0.01206083],
                        [0.0033454], [-0.04759717], [0.13968842], [0.04229945], [0.02889473],
                        [0.15241361], [0.05386424], [-0.05021071], [-0.40074808], [0.00208112],
                        [-0.00333146], [-0.08993652], [-0.10732406], [0.100861], [-0.14108559],
                        [0.29031209], [-0.11065826], [0.00393931], [0.15100358], [0.00589482],
                        [-0.00687222], [0.00495274], [-0.00020906], [0.03737696], [0.05552908],
                        [-0.03449133], [-0.0097648], [-0.04604962], [0.10512787], [0.00991475],
                        [0.0037229], [-0.05019526], [-0.12234672], [0.05175468], [0.09775051],
                        [0.01790728], [0.07605139], [0.04346028], [0.47216751], [-0.00666564],
                        [0.00018417], [0.02748734], [-0.04046904], [0.05844831], [-0.15203489],
                        [-0.00012805], [-0.05010673], [-0.16939733], [-1.52963783], [0.01177969],
                        [-0.01249833], [0.00922875], [0.06374963], [0.0578668], [0.06533716],
                        [-0.00313284], [0.18963941], [0.24170955], [0.35441665], [-0.0133694],
                        [-0.0074242], [-0.02388597], [-0.01773191], [-0.00752292], [0.05325798],
                        [-0.28002768], [0.02829416], [-0.11873481], [-0.71771167], [0.00389208],
                        [5.14316408e-05], [0.12618784], [0.00593139], [0.04264024], [-0.28599979],
                        [-0.1009471], [-0.20846597], [-0.06373693], [0.1362854], [-0.01359949],
                        [-0.00042742], [-0.04150794], [-0.08027705], [-0.02462074], [0.14259441],
                        [0.05245355], [0.06897394], [-0.14875724], [0.02826324], [-0.00600536],
                        [-0.00174409], [0.00132461], [0.0354341], [0.03920904], [0.18296244],
                        [-0.09240632], [0.10042862], [0.01040977], [0.17968613], [-0.00404793],
                        [0.00236923], [0.03087424], [0.00997214], [0.13750685], [0.10289528],
                        [0.19144892], [0.05877528], [-0.0618351], [0.25571306], [-0.01153247],
                        [0.02113273], [-0.05155993], [0.05829066], [-0.09072633], [-0.13871151],
                        [0.07059168], [-0.13652054], [-0.13836008], [0.38681681], [-0.00108428],
                        [0.00278972], [-0.03488456], [-0.11449833], [-0.08334598], [-0.08163141],
                        [-0.16017332], [-0.00662094], [0.00447672], [0.18545616], [0.00486269],
                        [-0.00414665], [0.04082678], [0.10303685], [-0.03331393], [-0.18193412],
                        [-0.04733239], [0.09104276], [0.01575009], [0.52304807], [0.0115638],
                        [-0.01083438], [0.0926606], [-0.01743124], [-0.06832854], [0.17006482],
                        [-0.07988479], [0.02286571], [0.0683522], [0.13080958], [-0.00484207],
                        [0.01334248], [-0.03279988], [-0.04569858], [-0.11798113], [0.14126632],
                        [0.11497013], [-0.20124139], [0.25121357], [0.31016981], [-0.00423849],
                        [-0.00652226], [-0.02483065], [-0.03680105], [-0.09674709], [-0.15096428],
                        [-0.0269077], [-0.24157866], [0.14560173], [-0.29431378], [0.00992868],
                        [-0.00390955], [-0.09044794], [0.04534677], [-0.15290562], [0.05058026],
                        [0.1979146], [0.3577543], [0.07162], [-0.51920407], [-0.01392384],
                        [0.00109518], [0.0334453], [-0.04458686], [0.05610666], [-0.00352582],
                        [-0.2548984], [-0.02150692], [-0.05828733], [0.24265154], [0.32934216],
                        [-0.40279218], [0.79291881], [1.10560481], [0.25250486], [0.53906551],
                        [0.44550204], [0.25445271], [-0.27622077], [-0.72065519], [0.04653782],
                        [0.01143135], [-0.01000722], [-0.02871148], [-0.04345467], [-0.05705033],
                        [-0.06786912], [-0.06543484], [-0.0645649]]),atol=min(max_atol,0.5),rtol=min(max_rtol,0.5)) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        np.testing.assert_allclose(lam,np.array([31.222097276578545, 4.07116662141881, 3.9187656008516716, 11302.32885672775, 2.3587359640672965]),atol=min(max_atol,0),rtol=min(max_rtol,2.5)) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(reml,-15526.867514090196,atol=min(max_atol,0),rtol=min(max_rtol,0.1)) 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(llk,-15331.745185948903,atol=min(max_atol,0),rtol=min(max_rtol,0.1)) 

    def test_edf1(self):
        compute_bias_corrected_edf(self.model,overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model.term_edf1])
        np.testing.assert_allclose(edf1,np.array([6.713831642615902, 4.921127633272276, 102.24427068803632, 6.9147867430206835, 1.358634183808314]),atol=min(max_atol,0),rtol=min(max_rtol,1.5)) 

    def test_ps(self):
        ps = []
        for par in range(len(self.model.formulas)):
            pps, _ = approx_smooth_p_values(self.model,par=par)
            ps.extend(pps)
        np.testing.assert_allclose(ps,np.array([-4.0011657986838145e-06, 0.0, 0.0, 0.07909902713107153]),atol=min(max_atol,0),rtol=min(max_rtol,0.5)) 

    def test_TRs(self):
        Trs = []
        for par in range(len(self.model.formulas)):
            _, pTrs = approx_smooth_p_values(self.model,par=par)
            Trs.extend(pTrs)
        np.testing.assert_allclose(Trs,np.array([38.39622887783163, 2835.761707764005, 243.20455192686484, 3.7792868990207036]),atol=min(max_atol,0),rtol=min(max_rtol,1.5))


class Test_shared_qefs:
    sim_dat = sim12(5000,c=0,seed=0,family=GAMMALS([LOG(),LOG()]),n_ranef=20)

    # We again need to model the mean: \mu_i = \alpha + f(x0) + f(x1) + f_{x4}(x0)
    sim_formula_m = Formula(lhs("y"),
                        [i(),f(["x0"],id=1),f(["x1"]),fs(["x0"],rf="x4")],
                        data=sim_dat)

    # and the standard deviation as well: log(\sigma_i) = \alpha + f(x2) + f(x3)
    sim_formula_sd = Formula(lhs("y"),
                        [i(),f(["x2"],id=1),f(["x3"])],
                        data=sim_dat)

    family = GAMMALS([LOG(),LOGb(-0.0001)])

    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["control_lambda"] = 1
    test_kwargs["extend_lambda"] = False
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["method"] = "QR/Chol"
    test_kwargs["repara"] = True
    test_kwargs["prefit_grad"] = True
    test_kwargs["method"] = 'qEFS'

    bfgs_opt={"gtol":1e-7,
            "ftol":1e-7,
            "maxcor":30,
            "maxls":20,
            "maxfun":100}

    test_kwargs["bfgs_options"] = bfgs_opt

    # Now define the model and fit!
    gsmm_fam = GAMLSSGSMMFamily(2,family)
    model = GSMM([sim_formula_m,sim_formula_sd],gsmm_fam)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(**test_kwargs)

    def test_GAMedf(self):
        np.testing.assert_allclose(self.model.edf,133.43747528447437,atol=min(max_atol,0),rtol=min(max_rtol,0.1)) 

    def test_GAMcoef(self):
        coef = self.model.coef
        np.testing.assert_allclose(coef,np.array([[2.16799851], [-0.3112196], [0.00676269], [0.4201406], [0.61797273],
                        [0.52202044], [0.44895006], [-0.06507334], [-0.42397616], [-0.68892731],
                        [-0.91668525], [-0.41196087], [-0.12509783], [0.15242029], [0.51521184],
                        [1.04286559], [1.69314981], [2.30579264], [2.95634309], [0.00375363],
                        [0.00095803], [-0.01749394], [0.04866508], [0.01610988], [0.01204409],
                        [0.09193799], [0.0424401], [-0.04805987], [-0.39819243], [0.00070066],
                        [-0.0011459], [-0.03429677], [-0.03873859], [0.04144056], [-0.07596874],
                        [0.18978641], [-0.08905066], [0.00387773], [0.14948812], [0.00179146],
                        [-0.0022638], [0.00123469], [-0.00121154], [0.01507921], [0.02633472],
                        [-0.02080303], [-0.00877894], [-0.04384797], [0.1124456], [0.00347343],
                        [0.00135215], [-0.0184494], [-0.04482129], [0.0241951], [0.04930618],
                        [0.00760179], [0.05943136], [0.04059714], [0.47713254], [-0.00196325],
                        [0.00022414], [0.00763534], [-0.01632584], [0.02269145], [-0.07613027],
                        [0.00019084], [-0.04063616], [-0.15955999], [-1.52317443], [0.00406978],
                        [-0.00372336], [0.00407742], [0.02397474], [0.02339283], [0.03242929],
                        [-0.00134213], [0.14833224], [0.224186], [0.35234378], [-0.00437312],
                        [-0.00253807], [-0.00999288], [-0.00803629], [-0.00258506], [0.02973766],
                        [-0.16955838], [0.01834036], [-0.11283042], [-0.72498566], [0.00142942],
                        [-7.92898283e-05], [0.04564999], [-0.00052317], [0.01740922], [-0.14523785],
                        [-0.06416053], [-0.16198612], [-0.05935355], [0.1304413], [-0.00411283],
                        [-6.86032369e-05], [-0.01502334], [-0.03025955], [-0.01128898], [0.06950945],
                        [0.03199717], [0.05202752], [-0.14165634], [0.03328117], [-0.00236461],
                        [-0.00083325], [-0.00040909], [0.01282327], [0.01477974], [0.08896886],
                        [-0.05683811], [0.0809826], [0.00947155], [0.18483563], [-0.00097363],
                        [0.00075122], [0.01051604], [0.00194582], [0.0562636], [0.04733838],
                        [0.11444094], [0.05288104], [-0.05736229], [0.2502938], [-0.00330095],
                        [0.00674206], [-0.01823235], [0.0199327], [-0.03656372], [-0.07094718],
                        [0.04274664], [-0.10715115], [-0.13001759], [0.38687831], [-0.00024124],
                        [0.00089502], [-0.01266976], [-0.04051792], [-0.03266625], [-0.04015137],
                        [-0.10518101], [-0.00267133], [0.00498951], [0.18554247], [0.00154738],
                        [-0.00080993], [0.01317374], [0.03361124], [-0.0131771], [-0.08460728],
                        [-0.03069324], [0.0651581], [0.01400267], [0.52748045], [0.00373322],
                        [-0.00349249], [0.03126245], [-0.00707543], [-0.02842607], [0.08250551],
                        [-0.04683272], [0.01836591], [0.06574071], [0.13342433], [-0.00208824],
                        [0.00513102], [-0.00744748], [-0.02077896], [-0.05379019], [0.07672374],
                        [0.0811137], [-0.16859225], [0.24116337], [0.30675057], [-0.0010866],
                        [-0.00180986], [-0.00553375], [-0.01335426], [-0.03612416], [-0.06825855],
                        [-0.01157465], [-0.18486844], [0.1374919], [-0.30247568], [0.00311401],
                        [-0.00117209], [-0.03376177], [0.01547755], [-0.06388321], [0.02557591],
                        [0.11584127], [0.28295314], [0.06720598], [-0.52709177], [-0.00421802],
                        [-6.01075582e-05], [0.01226497], [-0.01672551], [0.02306223], [-0.00239969],
                        [-0.16435384], [-0.02162077], [-0.05330939], [0.24324088], [0.33957445],
                        [-0.33838503], [0.81141732], [1.08682132], [0.33685749], [0.53350209],
                        [0.4929555], [0.27999358], [-0.25076651], [-0.74602954], [0.0478747],
                        [0.01060145], [-0.0120196], [-0.03127727], [-0.0459241], [-0.05889397],
                        [-0.06867861], [-0.06451001], [-0.0619268]]),atol=min(max_atol,0.5),rtol=min(max_rtol,0.5)) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        np.testing.assert_allclose(lam,np.array([41.17260229420759, 12.533666166891857, 4.092764472110347, 11118.20358119248, 4.15187531639246]),atol=min(max_atol,0),rtol=min(max_rtol,2.5)) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(reml,-15576.253064785291,atol=min(max_atol,0),rtol=min(max_rtol,0.1)) 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(llk,-15364.74212255429,atol=min(max_atol,0),rtol=min(max_rtol,0.1)) 

    def test_edf1(self):
        compute_bias_corrected_edf(self.model,overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model.term_edf1])
        np.testing.assert_allclose(edf1,np.array([8.517720894639751, 6.248437814610253, 148.4517833120425, 8.56828260261986, 1.5076592131479796]),atol=min(max_atol,0),rtol=min(max_rtol,1.5)) 

    def test_ps(self):
        ps = []
        for par in range(len(self.model.formulas)):
            pps, _ = approx_smooth_p_values(self.model,par=par)
            ps.extend(pps)
        np.testing.assert_allclose(ps,np.array([0.0, 0.0, 0.0, 0.1245955153970093]),atol=min(max_atol,0),rtol=min(max_rtol,0.5)) 

    def test_TRs(self):
        Trs = []
        for par in range(len(self.model.formulas)):
            _, pTrs = approx_smooth_p_values(self.model,par=par)
            Trs.extend(pTrs)
        np.testing.assert_allclose(Trs,np.array([211.9695958713893, 1711.8754238885585, 399.1431710461262, 3.21233862560394]),atol=min(max_atol,0),rtol=min(max_rtol,1.5))