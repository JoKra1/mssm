import numpy as np
import scipy as scp
import math
from ...models import GAMM,GAMLSS


def generate_mssm_test1(model):
    """Generates copy-pase code for multiple test-cases for GAMM and GAMLSS models.

    :param model: GAMM or GAMLSS model
    :type model: GAMM or GAMLSS
    """

    # EDF test
    edf_test_str = f"def test_GAMedf(self):\n\tassert round(self.model.edf,ndigits=3) == {round(model.edf,ndigits=3)}"

    print(edf_test_str,"\n")

    # sigma test
    if isinstance(model,GAMLSS) == False:
        coef, sigma = model.get_pars()
        sigma_test = f"def test_GAMsigma(self):\n\t_, sigma = self.model.get_pars()\n\tassert round(sigma,ndigits=3) == {round(sigma,ndigits=3)}"

        print(sigma_test,"\n")
    else:
        coef = model.overall_coef

    # coef test
    if isinstance(model,GAMLSS) == False:
        coef_test = "def test_GAMcoef(self):\n\tcoef, _ = self.model.get_pars()\n\tassert np.allclose(coef,np.array(["
    else:
        coef_test = "def test_GAMcoef(self):\n\tcoef = self.model.overall_coef\n\tassert np.allclose(coef,np.array(["

    for cfi,cf in enumerate(coef):
        if cfi > 0:
            coef_test += ","
            if cfi % 5 == 0:
                coef_test += "\n\t\t\t\t\t  "
            else:
                coef_test += " "
        coef_test += f"{cf}"
    coef_test += "]))"

    print(coef_test,"\n")

    # Lambda test
    if isinstance(model,GAMLSS) == False:
        pens = model.formula.penalties
        lambda_test = "def test_GAMlam(self):\n\tlam = np.array([p.lam for p in self.model.formula.penalties])\n\tassert np.allclose(lam,np.array(["
    else:
        pens = model.overall_penalties
        lambda_test = "def test_GAMlam(self):\n\tlam = np.array([p.lam for p in self.model.overall_penalties])\n\tassert np.allclose(lam,np.array(["

    for pfi,p in enumerate(pens):
        if pfi > 0:
            lambda_test += ", "
        lambda_test += f"{p.lam}"
    lambda_test += "]))"

    print(lambda_test,"\n")

    # reml test
    reml = model.get_reml()
    reml_test = f"def test_GAMreml(self):\n\treml = self.model.get_reml()\n\tassert round(reml,ndigits=3) == {round(reml,ndigits=3)}"

    print(reml_test,"\n")

    # llk test
    llk = model.get_llk(False)
    llk_test = f"def test_GAMllk(self):\n\tllk = self.model.get_llk(False)\n\tassert round(llk,ndigits=3) == {round(llk,ndigits=3)}"

    print(llk_test,"\n")