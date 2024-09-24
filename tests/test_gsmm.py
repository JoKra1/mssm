from mssm.models import *
import numpy as np
import os
from mssmViz.sim import*
from numdifftools import Gradient,Hessian


class Test_GAUMLSGEN:

    class BFGSGENSMOOTHFamily(GENSMOOTHFamily):
        """Implementation of the ``GENSMOOTHFamily`` class that allows model estimation based
        only on llk calls via BFGS optimization.

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
            return None
        
        def hessian(self, coef, coef_split_idx, y, Xs):
            return None
        

    def llk_gamm_fun(coef,coef_split_idx,links,y,Xs):
        """Likelihood for a Gaussian GAM(LSS) - implemented so
        that the model can be estimated using the general smooth code.
        """
        coef = coef.reshape(-1,1)
        split_coef = np.split(coef,coef_split_idx)
        eta_mu = Xs[0]@split_coef[0]
        eta_sd = Xs[1]@split_coef[1]
        
        mu_mu = links[0].fi(eta_mu)
        mu_sd = links[1].fi(eta_sd)
        
        family = GAUMLSS([Identity(),LOG()])
        llk = family.llk(y,mu_mu,mu_sd)
        return llk

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
    links = [Identity(),LOG()]

    # Now define the general family + model and fit!
    gsmm_fam = BFGSGENSMOOTHFamily(2,links,llk_gamm_fun)
    model = GSMM(formulas=formulas,family=gsmm_fam)
    model.fit(init_coef=None,method="BFGS",extend_lambda=False,max_outer=100,seed=100,conv_tol=1e-3)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 19.117 

    def test_GAMcoef(self):
        coef = self.model.overall_coef
        assert np.allclose(coef,np.array([[3.58466448], [-9.01937963], [-0.2732063], [4.57450569], [-2.6493674],
                        [-4.50599287], [-1.77012449], [-3.964982], [-6.55595615], [-4.59412855],
                        [-5.35322886], [0.02876581], [-1.28850628], [0.99546258], [1.97563086],
                        [1.96868409], [1.8516797], [1.9307447], [1.67949846], [0.90941805],
                        [-1.00639717], [-3.20354307]])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(lam,np.array([0.00950743385750192, 3.084017402392093])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -775.369 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -723.852

class Test_GAUMLSSGEN2:

    class NUMDIFFGENSMOOTHFamily(GENSMOOTHFamily):
        """Implementation of the ``GENSMOOTHFamily`` class that uses ``numdifftools`` to obtain the
        gradient and hessian of the likelihood to estimate a Gaussian GAMLSS via the general smooth code.

        References:

            - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
            - P. Brodtkorb (2014). numdifftools. see https://github.com/pbrod/numdifftools
        """

        def __init__(self, pars: int, links:[Link], llkfun:Callable, *llkargs) -> None:
            super().__init__(pars, links, *llkargs)
            self.llkfun = llkfun
        
        def llk(self, coef, coef_split_idx, y, Xs):
            return self.llkfun(coef, coef_split_idx, self.links, y, Xs,*self.llkargs)
        
        def gradient(self, coef, coef_split_idx, y, Xs):
            return Gradient(self.llkfun)(np.ndarray.flatten(coef),coef_split_idx,self.links,y,Xs,*self.llkargs)
        
        def hessian(self, coef, coef_split_idx, y, Xs):
            return scp.sparse.csc_array(Hessian(self.llkfun)(np.ndarray.flatten(coef),coef_split_idx,self.links,y,Xs))
            

    def llk_gamm_fun(coef,coef_split_idx,links,y,Xs):
            """Likelihood for a Gaussian GAM(LSS) - implemented so
            that the model can be estimated using the general smooth code.
            """
            coef = coef.reshape(-1,1)
            split_coef = np.split(coef,coef_split_idx)
            eta_mu = Xs[0]@split_coef[0]
            eta_sd = Xs[1]@split_coef[1]
            
            mu_mu = links[0].fi(eta_mu)
            mu_sd = links[1].fi(eta_sd)
            
            family = GAUMLSS([Identity(),LOG()])
            llk = family.llk(y,mu_mu,mu_sd)
            return llk

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
    links = [Identity(),LOG()]

    # Now define the general family + model and fit!
    gsmm_fam = NUMDIFFGENSMOOTHFamily(2,links,llk_gamm_fun)
    model = GSMM(formulas=formulas,family=gsmm_fam)

    # First fit with bfgs and, to speed things up, only then with Newton
    model.fit(init_coef=None,method="BFGS",extend_lambda=False,max_outer=100,seed=10,conv_tol=1e-3)

    # Use BFGS estimate as initial estimate for Newton model
    coef = model.overall_coef

    # Now re-fit with full Newton
    model.fit(init_coef=coef,method="Newton",extend_lambda=False,max_outer=100,seed=10,conv_tol=1e-3)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 18.33 

    def test_GAMcoef(self):
        coef = self.model.overall_coef
        assert np.allclose(coef,np.array([[3.58424912], [-9.19135148], [-0.38429367], [4.35288088], [-2.82065827],
                                          [-4.67000212], [-1.92324709], [-4.13190456], [-6.69019909], [-4.72319127],
                                          [-5.17038223], [0.01967291], [-1.35207541], [1.5948025], [2.43248755],
                                          [2.38238741], [2.20832098], [2.40089775], [2.00614383], [1.49760201],
                                          [-0.95130174], [-4.43919377]])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(lam,np.array([0.009142714577218717, 0.8980072136548634])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -772.284 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -719.311 

