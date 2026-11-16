# flake8: noqa
import mssm
from mssm.models import *
from mssm.src.python.utils import correct_VB, estimateVp
import numpy as np
import os
import copy
import io
from mssmViz.sim import *
from .defaults import (
    default_gamm_test_kwargs,
    default_gsmm_test_kwargs,
    max_atol,
    max_rtol,
    init_coef_gaumlss_tests,
    init_coef_gammals_tests,
    init_penalties_tests_gammlss,
    init_penalties_tests_gsmm,
    init_coef_gsmmgammlss,
)
import pytest

mssm.src.python.exp_fam.GAUMLSS.init_coef = init_coef_gaumlss_tests
mssm.src.python.exp_fam.GAMMALS.init_coef = init_coef_gammals_tests
mssm.src.python.exp_fam.GAUMLSS.init_lambda = init_penalties_tests_gammlss
mssm.src.python.exp_fam.GAMMALS.init_lambda = init_penalties_tests_gammlss
mssm.src.python.exp_fam.MULNOMLSS.init_lambda = init_penalties_tests_gammlss
mssm.src.python.exp_fam.PropHaz.init_lambda = init_penalties_tests_gsmm
mssm.src.python.utils.GAMLSSGSMMFamily.init_lambda = init_penalties_tests_gsmm
mssm.src.python.utils.GAMLSSGSMMFamily.init_coef = init_coef_gsmmgammlss

# Skip tests if jax is not installed
jax = pytest.importorskip("jax")
from mssm.src.python.custom_types import ModeAD
import jax.numpy as jnp
import jax.scipy as jscp


class GSMMGAUMLS(GSMMFamily):
    # Implementation of the ``GSMMFamily`` class to estimate a Gaussian GAMLSS via the general
    # smooth code. Relies on automatic differentiation to estimate gradients and hessian

    # We set up the constructor and then just call the constructor of the `GSMMFamily` class we
    # inherit from
    def __init__(self, links: list[Link]) -> None:
        super().__init__(2, links)

    def llk_pars(self, pars, extra, ys):

        y = ys[0]
        mu = pars[0]
        scale = pars[1]
        llk = jnp.sum(jscp.stats.norm.logpdf(y, loc=mu, scale=scale))
        return llk

    # Theoretically we also need to implement the ``get_resid`` method, but
    # it is not needed for fitting and thus for this example we simply overwrite it
    # with super, which will just ``pass``.
    def get_resid(self, coef, coef_split_idx, ys, Xs):
        return super().get_resid(coef, coef_split_idx, ys, Xs)


class GSMMGAUMLS2(GSMMFamily):
    # Implementation of the ``GSMMFamily`` class to estimate a Gaussian GAMLSS via the general
    # smooth code. Relies on automatic differentiation to estimate gradients and hessian.
    # Modified to cause eager execution

    # We set up the constructor and then just call the constructor of the `GSMMFamily` class we
    # inherit from
    def __init__(self, links: list[Link]) -> None:
        super().__init__(2, links)

    def llk_pars(self, pars, extra, ys):

        y = ys[0]
        mu = pars[0]
        scale = pars[1]
        if jnp.any(scale <= 0):
            # Control flow incompatible with jit
            return -jnp.inf
        llk = jnp.sum(jscp.stats.norm.logpdf(y, loc=mu, scale=scale))
        return llk

    # Theoretically we also need to implement the ``get_resid`` method, but
    # it is not needed for fitting and thus for this example we simply overwrite it
    # with super, which will just ``pass``.
    def get_resid(self, coef, coef_split_idx, ys, Xs):
        return super().get_resid(coef, coef_split_idx, ys, Xs)


class GSMMGAUMLS3(GSMMFamily):
    # Implementation of the ``GSMMFamily`` class to estimate a Gaussian GAMLSS via the general
    # smooth code. Triggers fallback to fd.

    # We set up the constructor and then just call the constructor of the `GSMMFamily` class we
    # inherit from
    def __init__(self, links: list[Link]) -> None:
        super().__init__(2, links)

    def llk_pars(self, pars, extra, ys):

        y = ys[0]
        mu = pars[0]
        scale = pars[1]
        llk = np.sum(scp.stats.norm.logpdf(y, loc=mu, scale=scale))
        return llk

    # Theoretically we also need to implement the ``get_resid`` method, but
    # it is not needed for fitting and thus for this example we simply overwrite it
    # with super, which will just ``pass``.
    def get_resid(self, coef, coef_split_idx, ys, Xs):
        return super().get_resid(coef, coef_split_idx, ys, Xs)


class Test_jit:
    # Simple gaussian gamlss test
    sim_dat = sim11(
        5000, 2, c=0, seed=20, family=Gaussian(), n_ranef=20, binom_offset=0
    )

    sim_dat = sim_dat.sort_values(["x4"], ascending=[True])

    sim_formula = Formula(
        lhs("y"),
        [i(), f(["x0"]), f(["x1"]), f(["x2"]), f(["x3"])],
        data=sim_dat,
        series_id="x4",
        discretize=True,
    )

    sim_formula_sd = Formula(
        lhs("y"),
        [i()],
        data=sim_dat,
        series_id="x4",
        discretize=True,
    )

    fam = GSMMGAUMLS([Identity(), LOGb(-0.0001)])
    model = GSMM([sim_formula, sim_formula_sd], fam)
    model.fit(**default_gsmm_test_kwargs)

    # Test fallback to eager
    fam2 = GSMMGAUMLS2([Identity(), LOGb(-0.0001)])
    model2 = GSMM([sim_formula, sim_formula_sd], fam2)
    model2.fit(max_outer=1, max_inner=1)

    # Test fallback to fd
    fam3 = GSMMGAUMLS3([Identity(), LOGb(-0.0001)])
    model3 = GSMM([sim_formula, sim_formula_sd], fam3)
    model3.fit(max_outer=1, max_inner=1)

    def test_assert_jit(self):
        assert self.model.family.ad_cache.mode == ModeAD.jit

    def test_assert_eager_fallback(self):
        assert self.model2.family.ad_cache.mode == ModeAD.eager

    def test_assert_fd_fallback(self):
        assert self.model3.family.ad_cache.mode == ModeAD.fd

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            21.17848704911578,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        np.testing.assert_allclose(
            lam,
            np.array(
                [
                    6.680041603909929,
                    2.9641979221651376,
                    0.004291918276357216,
                    7854928.251722907,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 2.5),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -11277.98565017528, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -11222.92013132029, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_edf1(self):
        compute_bias_corrected_edf(self.model, overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model.term_edf1])
        np.testing.assert_allclose(
            edf1,
            np.array(
                [
                    5.126896839769605,
                    5.834507342454058,
                    8.988131779403808,
                    1.0001846308346352,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.5),
        )

    def test_ps(self):
        ps = []
        for par in range(len(self.model.formulas)):
            pps, _ = approx_smooth_p_values(self.model, par=par)
            ps.extend(pps)
        np.testing.assert_allclose(
            ps,
            np.array([0.0, 0.0, 0.0, 0.4172323603217065]),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.5),
        )

    def test_TRs(self):
        Trs = []
        for par in range(len(self.model.formulas)):
            _, pTrs = approx_smooth_p_values(self.model, par=par)
            Trs.extend(pTrs)
        np.testing.assert_allclose(
            Trs,
            np.array(
                [
                    268.96281627023194,
                    3121.7796957847677,
                    7390.807810858431,
                    0.6583130020706411,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.5),
        )
