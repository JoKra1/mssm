from mssm.models import *
import numpy as np
import os
from mssmViz.sim import*
from numdifftools import Gradient,Hessian

class Test_GAUMLSSGEN:

    class NUMDIFFGENSMOOTHFamily(GENSMOOTHFamily):
        """Implementation of the ``GENSMOOTHFamily`` class that uses ``numdifftools`` to obtain the
        gradient and hessian of the likelihood to estimate a Gaussian GAMLSS via the general smooth code.

        For BFGS :func:``gradient`` and :func:``hessian`` can also just return None.

        References:

            - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
            - P. Brodtkorb (2014). numdifftools. see https://github.com/pbrod/numdifftools
            - Nocedal & Wright (2006). Numerical Optimization. Springer New York.
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
    model.fit(init_coef=None,optimizer="BFGS",extend_lambda=True,max_outer=100,seed=10,conv_tol=1e-3,extension_method_lam="nesterov",max_inner=50,min_inner=50)

    # Use BFGS estimate as initial estimate for Newton model
    coef = model.overall_coef

    # Now re-fit with full Newton
    model.fit(init_coef=coef,optimizer="Newton",extend_lambda=False,max_outer=100,seed=10,conv_tol=1e-7,restart=True,extension_method_lam="nesterov",max_inner=50,min_inner=50)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 18.221 

    def test_GAMcoef(self):
        coef = self.model.overall_coef
        assert np.allclose(coef,np.array([[3.58430184], [-9.20188437], [-0.38827981], [4.34065911], [-2.82926309],
                                          [-4.67964528], [-1.93048668], [-4.14082775], [-6.7007262], [-4.72385874],
                                          [-5.1895823], [0.02006132], [-1.35601524], [1.55744438], [2.40681319],
                                          [2.35125284], [2.18636379], [2.36778765], [1.98870176], [1.46001798],
                                          [-0.96065735], [-4.33426414]])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(lam,np.array([0.00909552371282305, 1.0115210823791556])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -772.187 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -719.5


class Test_GAUMLSSGEN2:
    class NUMDIFFGENSMOOTHFamily(GENSMOOTHFamily):
        """Implementation of the ``GENSMOOTHFamily`` class that uses ``numdifftools`` to obtain the
        gradient and hessian of the likelihood to estimate a Gaussian GAMLSS via the general smooth code.

        For BFGS :func:``gradient`` and :func:``hessian`` can also just return None.

        References:

            - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
            - P. Brodtkorb (2014). numdifftools. see https://github.com/pbrod/numdifftools
            - Nocedal & Wright (2006). Numerical Optimization. Springer New York.
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
    model.fit(init_coef=None,optimizer="L-BFGS-B",extend_lambda=True,max_outer=100,seed=10,conv_tol=1e-3,extension_method_lam="nesterov",max_inner=50,min_inner=50)

    # Use BFGS estimate as initial estimate for Newton model
    coef = model.overall_coef

    # Now re-fit with full Newton
    model.fit(init_coef=coef,optimizer="Newton",extend_lambda=False,max_outer=100,seed=10,conv_tol=1e-7,restart=True,extension_method_lam="nesterov",method="QR/Chol",max_inner=50,min_inner=50)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 18.221 

    def test_GAMcoef(self):
        coef = self.model.overall_coef
        assert np.allclose(coef,np.array([[ 3.58430184],
                                          [-9.20188437],
                                          [-0.38827981],
                                          [ 4.34065911],
                                          [-2.82926309],
                                          [-4.67964528],
                                          [-1.93048668],
                                          [-4.14082775],
                                          [-6.7007262 ],
                                          [-4.72385874],
                                          [-5.1895823 ],
                                          [ 0.02006132],
                                          [-1.35601524],
                                          [ 1.55744438],
                                          [ 2.40681319],
                                          [ 2.35125284],
                                          [ 2.18636379],
                                          [ 2.36778765],
                                          [ 1.98870176],
                                          [ 1.46001798],
                                          [-0.96065735],
                                          [-4.33426414]])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(lam,np.array([0.00909552, 1.01152108])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -772.187 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -719.5