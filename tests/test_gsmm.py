from mssm.models import *
import numpy as np
import os
from mssmViz.sim import*
from numdifftools import Gradient,Hessian
from mssm.src.python.gamm_solvers import deriv_transform_mu_eta,deriv_transform_eta_beta

class GAMLSSGENSMOOTHFamily(GENSMOOTHFamily):
    """Implementation of the ``GENSMOOTHFamily`` class that uses only information about the likelihood to estimate
    a GAMLSS model.

    References:

        - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
        - Nocedal & Wright (2006). Numerical Optimization. Springer New York.
    """

    def __init__(self, pars: int, links:[Link], llkfun:Callable, *llkargs) -> None:
        super().__init__(pars, links, *llkargs)
        self.llkfun = llkfun
    
    def llk(self, coef, coef_split_idx, y, Xs):
        return self.llkfun(coef, coef_split_idx, self.links, y, Xs,*self.llkargs)
    
    def gradient(self, coef, coef_split_idx, y, Xs):
        """
        Function to evaluate gradient for gsmm model.
        """
        coef = coef.reshape(-1,1)
        split_coef = np.split(coef,coef_split_idx)
        eta_mu = Xs[0]@split_coef[0]
        if len(Xs) > 1:
            eta_sd = Xs[1]@split_coef[1]
        
        # Get the Gamlss family
        gammlss_family = self.llkargs[0]
        
        if len(Xs) > 1:
            d1eta,d2eta,d2meta = deriv_transform_mu_eta(y,[self.links[0].fi(eta_mu),self.links[1].fi(eta_sd)],gammlss_family)
        else:
            d1eta,d2eta,d2meta = deriv_transform_mu_eta(y,[self.links[0].fi(eta_mu)],gammlss_family)
            
        grad,_ = deriv_transform_eta_beta(d1eta,d2eta,d2meta,Xs,only_grad=True)
        #print(pgrad.flatten())
        return grad.reshape(-1,1)
    
    def hessian(self, coef, coef_split_idx, y, Xs):
        return None
            

def llk_gamm_fun(coef,coef_split_idx,links,y,Xs,gammlss_family):
    """Likelihood for a GAM(LSS) - implemented so
    that the model can be estimated using the general smooth code.

    Note, gammlss_family is passed via llkargs so that this code works with
    Gaussian and Gamma models.
    """
    coef = coef.reshape(-1,1)
    split_coef = np.split(coef,coef_split_idx)
    eta_mu = Xs[0]@split_coef[0]
    if len(Xs) > 1:
        eta_sd = Xs[1]@split_coef[1]
    
    mu_mu = links[0].fi(eta_mu)
    if len(Xs) > 1:
        mu_sd = links[1].fi(eta_sd)
    
    if len(Xs) > 1:
        llk = gammlss_family.llk(y,mu_mu,mu_sd)
    else:
        llk = gammlss_family.llk(y,mu_mu)

    if np.isnan(llk):
        return -np.inf
    
    return llk

################################################################## Tests ##################################################################

class Test_GAUMLSSGEN:

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
    gsmm_fam = GAMLSSGENSMOOTHFamily(2,links,llk_gamm_fun,GAUMLSS(links))
    model = GSMM(formulas=formulas,family=gsmm_fam)

    # First fit with SR1 and, to speed things up, only then with Newton
    bfgs_opt={"gtol":1e-9,
            "ftol":1e-9,
            "maxcor":30,
            "maxls":200,
            "maxfun":1e7}
                    
    model.fit(init_coef=None,method='qEFS',extend_lambda=False,
            control_lambda=False,max_outer=200,max_inner=500,min_inner=500,
            seed=0,qEFSH='SR1',max_restarts=5,overwrite_coef=False,qEFS_init_converge=False,prefit_grad=True,
            progress_bar=True,**bfgs_opt)

    model2 = copy.deepcopy(model)

    # Now fit with BFGS
    model2.fit(init_coef=None,method='qEFS',extend_lambda=False,
            control_lambda=False,max_outer=200,max_inner=500,min_inner=500,
            seed=0,qEFSH='BFGS',max_restarts=0,overwrite_coef=True,qEFS_init_converge=True,prefit_grad=False,
            progress_bar=True,**bfgs_opt)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 19.049

    def test_GAMedf2(self):
        assert round(self.model2.edf,ndigits=3) == 16.305

    def test_GAMcoef(self):
        coef = self.model.overall_coef.flatten()
        assert np.allclose(coef,np.array([ 7.64905998, -0.86968557, -0.83494213,  0.07122906,  1.20919857,
                                            1.67258359,  0.46597024, -0.53378393, -0.76055135, -0.7941819 ,
                                            -1.81015845, -1.07691283, -0.44119683,  0.33501224,  1.01000229,
                                            1.9723699 ,  3.16101178,  4.12055134,  5.27092295, -9.19693724,
                                            4.33968853,  5.20719195, -2.48498028, -0.41483633, -1.68072058,
                                            -4.12405797, -4.32178097, -1.28126424, -0.05013006, -0.01078947,
                                            0.01309918,  0.0304733 ,  0.04832851,  0.06789398,  0.08949696,
                                            0.09934445,  0.10436846,  0.66643841]))

    def test_GAMcoef2(self):
        coef = self.model2.overall_coef.flatten()
        assert np.allclose(coef,np.array([  7.6489859 ,  -0.90359436,  -1.1995978 ,  -0.1219847 ,
                                            0.8902022 ,   1.53926404,   0.09832465,  -0.914336  ,
                                            -0.87410618,  -0.50078609,  -1.70675555,  -1.40414449,
                                            -0.78427345,   0.28350426,   0.78126954,   1.56534334,
                                            2.89252623,   4.27551005,   5.81263687, -10.35107199,
                                            3.37333916,   4.0633653 ,  -3.53414945,  -1.49195269,
                                            -2.73210049,  -5.26428994,  -4.80803947,  -2.02328384,
                                            -0.01671727,  -0.02500594,  -0.0339186 ,  -0.0572995 ,
                                            -0.03603238,   0.02741854,   0.08237743,   0.14099506,
                                            0.20398593,   0.66054218]))  

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(lam,np.array([7.21928261e-01, 5.68402996e+00, 7.31895390e-03, 1.00000000e+07]))

    def test_GAMlam2(self):
        lam = np.array([p.lam for p in self.model2.overall_penalties])
        assert np.allclose(lam,np.array([2.61309258e-01, 9.35511512e-01, 5.32567098e-03, 3.27034646e+01]))  

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -1078.492

    def test_GAMreml2(self):
        reml = self.model2.get_reml()
        assert round(reml,ndigits=3) == -1062.438

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -1042.691
    
    def test_GAMllk2(self):
        llk = self.model2.get_llk(False)
        assert round(llk,ndigits=3) == -1039.741


class Test_PropHaz:
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gsmm_newton.fit(init_coef=None,method="QR/Chol",extend_lambda=False,
                        control_lambda=False,max_outer=200,seed=0,max_inner=500,
                        min_inner=500,progress_bar=True)
        

    gsmm_qefs_fam = PropHaz(ut,r)
    gsmm_qefs = GSMM([copy.deepcopy(sim_formula_m)],gsmm_qefs_fam)

    # Fit with qEFS update without initialization
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bfgs_opt={"gtol":1e-9,
                    "ftol":1e-9,
                    "maxcor":30,
                    "maxls":200,
                    "maxfun":1e7}
        
        gsmm_qefs.fit(init_coef=None,method='qEFS',extend_lambda=False,
                        control_lambda=False,max_outer=200,max_inner=500,min_inner=500,
                        seed=0,qEFSH='SR1',max_restarts=0,overwrite_coef=False,qEFS_init_converge=False,prefit_grad=True,
                        progress_bar=True,**bfgs_opt)

    def test_GAMcoef(self):

        assert np.allclose((self.gsmm_newton.overall_coef - self.gsmm_qefs.overall_coef).flatten(),
                           np.array([ 0.00085094, -0.00112174, -0.00254822, -0.00133955,  0.00105183,
                                        -0.00174351, -0.00296606,  0.00099741,  0.00571504, -0.00296058,
                                        -0.00230932,  0.00126035,  0.00379059,  0.00195697,  0.00087278,
                                        0.00414338, -0.01210031, -0.03304087,  0.03062558,  0.02426205,
                                        0.01574717,  0.03415708,  0.02328558,  0.02059044,  0.02940937,
                                        0.01876927,  0.02136963,  0.00305857, -0.01617596, -0.02171221,
                                        -0.00669945,  0.01213701,  0.0186336 , -0.00328931, -0.04652364,
                                        -0.0910511 ])) 

    def test_GAMlam(self):

        assert np.allclose(np.round(np.array([p1.lam - p2.lam for p1,p2 in zip(self.gsmm_newton.overall_penalties,self.gsmm_qefs.overall_penalties)]),decimals=3),
                           np.array([ 0.282,  0.544,  0.   , 44.97 ])) 

    def test_GAMreml(self):

        assert round(self.gsmm_newton.get_reml() - self.gsmm_qefs.get_reml(),ndigits=3) == np.float64(-0.816)

    def test_GAMllk(self):

        assert round(self.gsmm_newton.get_llk(True) - self.gsmm_qefs.get_llk(True),ndigits=3) == np.float64(-0.498)


class Test_PropHaz_repara:
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gsmm_newton.fit(init_coef=None,method="QR/Chol",extend_lambda=False,
                        control_lambda=False,max_outer=200,seed=0,max_inner=500,
                        min_inner=500,progress_bar=True,repara=True)
        

    gsmm_qefs_fam = PropHaz(ut,r)
    gsmm_qefs = GSMM([copy.deepcopy(sim_formula_m)],gsmm_qefs_fam)

    # Fit with qEFS update without initialization
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bfgs_opt={"gtol":1e-9,
                    "ftol":1e-9,
                    "maxcor":30,
                    "maxls":200,
                    "maxfun":1e7}
        
        gsmm_qefs.fit(init_coef=None,method='qEFS',extend_lambda=False,
                        control_lambda=False,max_outer=200,max_inner=500,min_inner=500,
                        seed=0,qEFSH='SR1',max_restarts=0,overwrite_coef=False,qEFS_init_converge=False,prefit_grad=True,
                        progress_bar=True,repara=True,**bfgs_opt)

    def test_GAMcoef(self):

        assert np.allclose((self.gsmm_newton.overall_coef - self.gsmm_qefs.overall_coef).flatten(),
                           np.array([ 0.00130659, -0.00146633, -0.00482344, -0.00219018,  0.00408372,
                                    -0.00254493, -0.00538949,  0.00269739,  0.01220294, -0.00574834,
                                    -0.00341937,  0.00293582,  0.00723694,  0.0041163 ,  0.00185322,
                                    0.00810604, -0.0177597 , -0.05120267, -0.0156164 , -0.01146629,
                                    -0.03849618, -0.00090847, -0.02320778, -0.0266449 , -0.01706707,
                                    0.00320521, -0.00176434,  0.00787907, -0.03024446, -0.04263593,
                                    -0.01658554,  0.01727422,  0.0315337 , -0.0050121 , -0.08136764,
                                    -0.15992784])) 

    def test_GAMlam(self):

        assert np.allclose(np.round(np.array([p1.lam - p2.lam for p1,p2 in zip(self.gsmm_newton.overall_penalties,self.gsmm_qefs.overall_penalties)]),decimals=3),
                           np.array([ 0.671,  0.853, -0.   , 63.778])) 

    def test_GAMreml(self):

        assert round(self.gsmm_newton.get_reml() - self.gsmm_qefs.get_reml(),ndigits=3) == -0.865

    def test_GAMllk(self):

        assert round(self.gsmm_newton.get_llk(True) - self.gsmm_qefs.get_llk(True),ndigits=3) == -0.758