# flake8: noqa
import mssm
from mssm.models import *
from mssm.src.python.utils import correct_VB, estimateVp
import numpy as np
import os
import copy
import io
from contextlib import redirect_stdout
from mssmViz.sim import *
from .defaults import (
    default_gamm_test_kwargs,
    default_gsmm_test_kwargs,
    max_atol,
    max_rtol,
    init_penalties_tests_gammlss,
    init_penalties_tests_gsmm,
    init_coef_gsmmgammlss,
)
from mssm.src.python.mcmc import sample_mssm

mssm.src.python.exp_fam.GAUMLSS.init_lambda = init_penalties_tests_gammlss
mssm.src.python.exp_fam.GAMMALS.init_lambda = init_penalties_tests_gammlss
mssm.src.python.exp_fam.MULNOMLSS.init_lambda = init_penalties_tests_gammlss
mssm.src.python.exp_fam.PropHaz.init_lambda = init_penalties_tests_gsmm
mssm.src.python.utils.GAMLSSGSMMFamily.init_lambda = init_penalties_tests_gsmm
mssm.src.python.utils.GAMLSSGSMMFamily.init_coef = init_coef_gsmmgammlss

################################################################## Tests ##################################################################


class Test_Chol_updating:
    sim_dat = sim12(5000, c=0, seed=0, family=GAMMALS([LOG(), LOG()]), n_ranef=20)

    # We again need to model the mean: \mu_i = \alpha + f(x0) + f(x1) + f_{x4}(x0)
    sim_formula_m = Formula(
        lhs("y"), [i(), f(["x0"], id=1), f(["x1"]), fs(["x0"], rf="x4")], data=sim_dat
    )

    # and the standard deviation as well: log(\sigma_i) = \alpha + f(x2) + f(x3)
    sim_formula_sd = Formula(lhs("y"), [i(), f(["x2"], id=1), f(["x3"])], data=sim_dat)

    family = GAMMALS([LOG(), LOGb(-0.0001)])

    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["control_lambda"] = 1
    test_kwargs["extend_lambda"] = False
    test_kwargs["max_outer"] = 20
    test_kwargs["max_inner"] = 500
    test_kwargs["method"] = "qEFS"
    test_kwargs["repara"] = True
    test_kwargs["prefit_grad"] = True
    test_kwargs["sample_hessian_options"] = {"T": 0.1, "delta": 1}
    test_kwargs["max_restarts"] = -1
    test_kwargs["sample_hessian"] = True
    test_kwargs["structured_qefs"] = True
    test_kwargs["n_cores"] = 4
    test_kwargs["sqEFS_options"] = {
        "dampen_HBB": 0.1,
        "dampen_HBb": 1,
        "pre_cond": False,
        "PD_HBB": False,
    }
    test_kwargs["qEFS_memory_usage"] = (
        0.25  # Low enough memory to trigger Chol updating
    )
    test_kwargs["bfgs_options"] = None

    def callback(outer, pen_llk, coef, lam):
        print(outer, pen_llk, lam)

    # Now define the model and fit!
    gsmm_fam = GAMLSSGSMMFamily(2, family)
    model = GSMM([sim_formula_m, sim_formula_sd], gsmm_fam)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(**test_kwargs, callback=callback)

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            107.65988689459013,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_edf1(self):
        compute_bias_corrected_edf(self.model, overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model.term_edf1])
        np.testing.assert_allclose(
            edf1,
            np.array(
                [
                    6.822003954437688,
                    4.85270654666088,
                    119.85131677237061,
                    6.926201285757667,
                    2.211716646272749,
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
            np.array([0.0, 0.0, 0.0, 0.10946942004018534]),
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
                    56.579810558691,
                    2709.9201524805353,
                    237.11272829051055,
                    4.758159728976485,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.5),
        )


class Test_qefs_final_budget:
    sim_dat = sim4(500, 2, family=Gamma(), seed=0)

    # We again need to model the mean: \mu_i = \alpha + f(x0) + f(x1) + f_{x4}(x0)
    sim_formula_m = Formula(
        lhs("y"), [i(), f(["x0"]), f(["x1"]), f(["x2"]), f(["x3"])], data=sim_dat
    )

    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["control_lambda"] = 1
    test_kwargs["extend_lambda"] = False
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["method"] = "qEFS"
    test_kwargs["repara"] = True
    test_kwargs["prefit_grad"] = True
    test_kwargs["structured_qefs"] = False
    test_kwargs["sample_hessian_options"] = None
    test_kwargs["qEFS_final_memory_usage"] = 1

    # Now define the model and fit!
    gsmm_fam = GAMLSSGSMMFamily(2, Gamma())
    model = GSMM([sim_formula_m], gsmm_fam)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(**test_kwargs)

    def test_sksize(self):
        assert self.model.lvi_linop.sk.shape[0] == self.model.lvi_linop.sk.shape[1]


class Test_repara_skip:
    # Multi Gauss case
    sim_dat = sim16(500, c=1, seed=0, correlate=False)

    # Leave first two linear predictors un-penalized
    sim_formulas = [
        Formula(lhs("y0"), [i()], data=sim_dat),
        Formula(lhs("y1"), [i()], data=sim_dat),
        Formula(lhs("y2"), [i(), f(["x3"])], data=sim_dat),
    ]

    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["method"] = "QR/Chol"
    test_kwargs["extend_lambda"] = False
    test_kwargs["control_lambda"] = 2
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["min_inner"] = 500
    test_kwargs["seed"] = 0
    test_kwargs["repara"] = True
    test_kwargs["prefit_grad"] = True

    model = GSMM(sim_formulas, MultiGauss(3, [Identity() for _ in range(3)]))
    model.fit(**test_kwargs)

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            10.143423613146284,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_GAMcoef(self):
        coef = np.round(self.model.coef, decimals=6)
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    [1.245002e00],
                    [6.299080e00],
                    [2.620000e-02],
                    [5.053800e-02],
                    [1.305000e-03],
                    [-2.862000e-02],
                    [-4.703400e-02],
                    [-6.035600e-02],
                    [-6.955200e-02],
                    [-7.668700e-02],
                    [-6.803400e-02],
                    [-5.359900e-02],
                    [-3.651300e-01],
                    [-2.009400e-02],
                    [1.504210e-01],
                    [-1.218968e00],
                    [-5.074000e-03],
                    [-3.756540e-01],
                ]
            ),
            atol=min(max_atol, 10),
            rtol=min(max_rtol, 10),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        np.testing.assert_allclose(
            lam,
            np.array([1231.1931630229676]),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 2.5),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -1756.634350155022, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -1729.8759979948577, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_edf1(self):
        compute_bias_corrected_edf(self.model, overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model.term_edf1])
        np.testing.assert_allclose(
            edf1,
            np.array([1.2716437793312583]),
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
            np.array([0.5852398802652816]),
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
            np.array([0.5167373537113078]),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.5),
        )


class Test_GAUMLSSGEN_hard:

    # Simulate 500 data points
    sim_dat = sim3(
        500, 2, c=1, seed=0, family=Gaussian(), binom_offset=0, correlate=False
    )

    # We need to model the mean: \mu_i
    formula_m = Formula(
        lhs("y"), [i(), f(["x0"]), f(["x1"]), f(["x2"]), f(["x3"])], data=sim_dat
    )

    # And for sd - here constant
    formula_sd = Formula(lhs("y"), [i()], data=sim_dat)

    # Collect both formulas
    formulas = [formula_m, formula_sd]
    links = [Identity(), LOG()]

    # Now define the general family + model and fit!
    gsmm_fam = GAMLSSGSMMFamily(2, GAUMLSS(links))
    model = GSMM(formulas=formulas, family=gsmm_fam)

    # fit with BFGS
    bfgs_opt = {"gtol": 1e-7, "ftol": 1e-7, "maxcor": 30, "maxls": 200, "maxfun": 1e7}

    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["method"] = "qEFS"
    test_kwargs["extend_lambda"] = False
    test_kwargs["control_lambda"] = 1
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["min_inner"] = 500
    test_kwargs["seed"] = 0
    test_kwargs["max_restarts"] = 5
    test_kwargs["prefit_grad"] = True
    test_kwargs["sample_hessian"] = False
    test_kwargs["structured_qefs"] = False
    test_kwargs["qEFSH"] = "BFGS"
    test_kwargs["bfgs_options"] = bfgs_opt
    test_kwargs["sample_hessian_options"] = {"n_samples:": 0}  # Prevent sampling

    model.fit(**test_kwargs)

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            18.377187361889852,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_GAMcoef(self):
        coef = self.model.coef
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    [7.64905069],
                    [-0.86673491],
                    [-0.90094528],
                    [0.03276751],
                    [1.15914006],
                    [1.65315849],
                    [0.38976786],
                    [-0.62349091],
                    [-0.7531447],
                    [-0.62543635],
                    [-1.77827109],
                    [-1.16369955],
                    [-0.53206949],
                    [0.30195526],
                    [0.94751062],
                    [1.86610313],
                    [3.08262143],
                    [4.13731307],
                    [5.35515618],
                    [-7.9746717],
                    [5.31754213],
                    [6.43841665],
                    [-1.39549816],
                    [0.72747751],
                    [-0.56335123],
                    [-2.93264959],
                    [-3.69271431],
                    [-1.03991244],
                    [-0.04562291],
                    [-0.01165826],
                    [0.00888674],
                    [0.02409037],
                    [0.04134725],
                    [0.06172081],
                    [0.08447056],
                    [0.09714901],
                    [0.10562989],
                    [0.66701223],
                ]
            ),
            atol=min(max_atol, 0.3),
            rtol=min(max_rtol, 0.5),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        np.testing.assert_allclose(
            lam,
            np.array(
                [
                    0.5660081066190392,
                    3.6344843285366273,
                    0.009036189964515981,
                    2111.2780420530066,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 3),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -1073.8484425587437, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -1043.0223032680815, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_edf1(self):
        compute_bias_corrected_edf(self.model, overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model.term_edf1])
        np.testing.assert_allclose(
            edf1,
            np.array(
                [
                    5.31951572924431,
                    3.6329012604530178,
                    8.634074326598565,
                    1.0297773848735796,
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
            np.array([0.0047744792490702626, 0.0, 0.0, 0.8676565723545753]),
            atol=min(max_atol, 0.01),
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
                    17.544251844815605,
                    97.98996254745872,
                    346.09962445913857,
                    0.04502732327983149,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.5),
        )


class Test_PropHaz_hard:
    sim_dat = sim3(
        500, 2, c=1, seed=0, family=PropHaz([0], [0]), binom_offset=0.1, correlate=False
    )

    # Prep everything for prophaz model
    sim_dat = sim_dat.sort_values(["y"], ascending=[False])
    sim_dat = sim_dat.reset_index(drop=True)
    # print(sim_dat.head(),np.mean(sim_dat["delta"]))

    u, inv = np.unique(sim_dat["y"], return_inverse=True)
    ut = np.flip(u)
    r = np.abs(inv - max(inv))

    # We only need to model the mean: \mu_i
    sim_formula_m = Formula(
        lhs("delta"), [f(["x0"]), f(["x1"]), f(["x2"]), f(["x3"])], data=sim_dat
    )

    # Fit with Newton
    gsmm_newton_fam = PropHaz(ut, r)
    model = GSMM([copy.deepcopy(sim_formula_m)], gsmm_newton_fam)

    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["method"] = "QR/Chol"
    test_kwargs["extend_lambda"] = False
    test_kwargs["control_lambda"] = False
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["min_inner"] = 500
    test_kwargs["seed"] = 0
    test_kwargs["repara"] = False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(**test_kwargs)

    resid = model.get_resid(resid_type="Martingale")

    def resid_test(self):
        assert self.resid.shape == (500, 1)

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            17.70139744543187,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_GAMcoef(self):
        coef = self.model.coef
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    [-0.71387545],
                    [0.06955704],
                    [0.62401006],
                    [1.06531137],
                    [1.19839925],
                    [1.03063443],
                    [0.50277917],
                    [-0.5497074],
                    [-1.66865464],
                    [-1.45494926],
                    [-0.69155732],
                    [-0.2809881],
                    [0.11233951],
                    [0.70202522],
                    [1.68674218],
                    [2.79443246],
                    [3.88466532],
                    [5.20452557],
                    [-10.86368411],
                    [-0.12353043],
                    [0.69872199],
                    [-6.03417634],
                    [-4.54182276],
                    [-4.57937448],
                    [-8.17466594],
                    [-5.02892314],
                    [-4.86821306],
                    [0.11913661],
                    [0.02177487],
                    [-0.05112351],
                    [-0.12708209],
                    [-0.18859235],
                    [-0.20433111],
                    [-0.15688625],
                    [-0.03459626],
                    [0.10412985],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.5),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        np.testing.assert_allclose(
            lam,
            np.array(
                [
                    6.955990167024357,
                    6.718132388893934,
                    0.00536660757744937,
                    101.52004507997195,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 2.5),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -1809.885338980533, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -1772.5517665619386, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_edf1(self):
        compute_bias_corrected_edf(self.model, overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model.term_edf1])
        np.testing.assert_allclose(
            edf1,
            np.array(
                [
                    4.314129203718049,
                    4.330180049072045,
                    8.872759352525616,
                    2.4734068397134137,
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
            np.array([0.0, 0.0, 0.0, 0.21062684440882912]),
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
                    72.63212701087522,
                    328.5041277852474,
                    463.90284670844085,
                    4.721875845689623,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.5),
        )


class Test_PropHaz_repara_hard:
    sim_dat = sim3(
        500, 2, c=1, seed=0, family=PropHaz([0], [0]), binom_offset=0.1, correlate=False
    )

    # Prep everything for prophaz model
    sim_dat = sim_dat.sort_values(["y"], ascending=[False])
    sim_dat = sim_dat.reset_index(drop=True)
    # print(sim_dat.head(),np.mean(sim_dat["delta"]))

    u, inv = np.unique(sim_dat["y"], return_inverse=True)
    ut = np.flip(u)
    r = np.abs(inv - max(inv))

    # We only need to model the mean: \mu_i
    sim_formula_m = Formula(
        lhs("delta"), [f(["x0"]), f(["x1"]), f(["x2"]), f(["x3"])], data=sim_dat
    )

    # Fit with Newton
    gsmm_newton_fam = PropHaz(ut, r)
    model = GSMM([copy.deepcopy(sim_formula_m)], gsmm_newton_fam)

    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["method"] = "QR/Chol"
    test_kwargs["extend_lambda"] = False
    test_kwargs["control_lambda"] = False
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["min_inner"] = 500
    test_kwargs["seed"] = 0
    test_kwargs["repara"] = True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(**test_kwargs)

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            17.701397445431944,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_GAMcoef(self):
        coef = self.model.coef
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    [-0.71387545],
                    [0.06955704],
                    [0.62401006],
                    [1.06531137],
                    [1.19839925],
                    [1.03063443],
                    [0.50277917],
                    [-0.5497074],
                    [-1.66865464],
                    [-1.45494926],
                    [-0.69155732],
                    [-0.2809881],
                    [0.11233951],
                    [0.70202522],
                    [1.68674218],
                    [2.79443246],
                    [3.88466532],
                    [5.20452557],
                    [-10.86368411],
                    [-0.12353043],
                    [0.69872199],
                    [-6.03417634],
                    [-4.54182276],
                    [-4.57937448],
                    [-8.17466594],
                    [-5.02892314],
                    [-4.86821306],
                    [0.11913661],
                    [0.02177487],
                    [-0.05112351],
                    [-0.12708209],
                    [-0.18859235],
                    [-0.20433111],
                    [-0.15688625],
                    [-0.03459626],
                    [0.10412985],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.5),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        np.testing.assert_allclose(
            lam,
            np.array(
                [
                    6.955990167024334,
                    6.7181323888940545,
                    0.005366607577449276,
                    101.52004507996567,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 2.5),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -1809.8853389805336, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -1772.5517665619386, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_edf1(self):
        compute_bias_corrected_edf(self.model, overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model.term_edf1])
        np.testing.assert_allclose(
            edf1,
            np.array(
                [
                    4.314129203718056,
                    4.330180049072032,
                    8.87275935252572,
                    2.473406839713454,
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
            np.array([0.0, 0.0, 0.0, 0.21062684440883006]),
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
                    72.63212701087531,
                    328.5041277852453,
                    463.902846708443,
                    4.721875845689723,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.5),
        )


class Test_drop:
    sim_dat = sim13(5000, 2, c=0, seed=0, family=Gaussian(), binom_offset=0, n_ranef=20)

    formula = Formula(
        lhs("y"),
        [
            i(),
            l(["x5"]),
            l(["x6"]),
            f(["x0"], by="x5"),
            f(["x0"], by="x6"),
            fs(["x0"], rf="x4"),
        ],
        data=sim_dat,
    )

    formula_sd = Formula(lhs("y"), [i()], data=sim_dat)

    model = GSMM(
        [formula, formula_sd], GAMLSSGSMMFamily(2, GAUMLSS([Identity(), LOGb(-0.001)]))
    )

    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["max_inner"] = 500
    test_kwargs["min_inner"] = 500
    test_kwargs["max_outer"] = 200
    test_kwargs["extend_lambda"] = False
    test_kwargs["control_lambda"] = 2
    test_kwargs["repara"] = False
    test_kwargs["method"] = "LU/Chol"
    model.fit(**test_kwargs)

    # More extensive selection + posterior sim checks
    res = correct_VB(
        model,
        grid_type="JJJ1",
        method="LU/Chol",
        compute_Vcc=False,
        form_t1=False,
        n_c=1,
        recompute_H=False,
        only_expected_edf=False,
        prior=None,
        Vp_fidiff=False,
    )

    compute_bias_corrected_edf(model)

    res2 = correct_VB(
        model,
        grid_type="JJJ3",
        method="LU/Chol",
        compute_Vcc=False,
        recompute_H=True,
        n_c=1,
        seed=20,
        VP_grid_type="JJJ2",
        only_expected_edf=False,
        prior=None,
        Vp_fidiff=False,
    )

    Vp2, _, _, _, _, _ = estimateVp(
        model,
        grid_type="JJJ2",
        n_c=1,
        seed=20,
        method="LU/Chol",
        prior=None,
        Vp_fidiff=False,
    )

    # Set up some new data for prediction
    pred_dat = pd.DataFrame(
        {
            "x0": np.linspace(0, 1, 30),
            "x4": ["f_12" for _ in range(30)],
            "x5": ["l5.1" for _ in range(30)],
            "x6": ["l6.1" for _ in range(30)],
        }
    )

    _, pred_mat, _ = model.predict([3], pred_dat, par=0)

    # `use_post` identifies only coefficients related to f(x0):x5 in the model
    use_post = pred_mat.sum(axis=0) != 0
    use_post = np.arange(0, pred_mat.shape[1])[use_post]
    use_post

    post = model.sample_post(10, use_post, seed=2000, par=0)

    post2 = sample_MVN(
        10,
        model.coef.flatten(),
        model.scale,
        P=None,
        L=None,
        LI=res2[1].T,
        use=use_post,
        seed=2000,
    )

    def test_GAMedf_hard(self):
        np.testing.assert_allclose(
            self.model.edf,
            107.54783796715027,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.03),
        )

    def test_GAMedf1_hard(self):
        np.testing.assert_allclose(
            self.res[-3],
            self.model.edf1,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.03),
        )

    def test_GAMedf2_hard(self):
        np.testing.assert_allclose(
            self.res2[5],
            107.96499018523079,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.03),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        np.testing.assert_allclose(
            lam,
            np.array(
                [
                    4.913881081300338,
                    5.087832012717992,
                    0.002924222705631079,
                    10000000.0,
                    1.2211531005561156,
                    1.1056775054652457,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.5),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -10648.977146470883, atol=min(max_atol, 0), rtol=min(max_rtol, 0.01)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -10435.443686415718, atol=min(max_atol, 0), rtol=min(max_rtol, 0.01)
        )

    def test_drop(self):
        assert len(self.model.info.dropped) == 1

    def test_VP(self):
        np.testing.assert_allclose(
            self.res2[2], self.Vp2, atol=min(max_atol, 0), rtol=min(max_rtol, 1e-7)
        )

    def test_post(self):
        np.testing.assert_allclose(
            self.post,
            np.array(
                [
                    [
                        -0.4560724,
                        -0.82602553,
                        -0.26366509,
                        -0.79786615,
                        -0.44232674,
                        -0.67485869,
                        -0.57739543,
                        -0.27754304,
                        -0.62234539,
                        -0.55592201,
                    ],
                    [
                        0.89476297,
                        -0.18902322,
                        1.10211202,
                        -0.17050216,
                        0.18796238,
                        0.2116602,
                        0.26453479,
                        1.02570765,
                        -0.00924147,
                        0.38500421,
                    ],
                    [
                        1.51541037,
                        0.32782496,
                        1.42617662,
                        0.78165293,
                        0.51264206,
                        0.813669,
                        1.11409757,
                        1.27888749,
                        0.85963984,
                        1.10545554,
                    ],
                    [
                        1.77082129,
                        0.93602721,
                        1.75345879,
                        1.21871887,
                        1.2051166,
                        1.16784506,
                        1.37225383,
                        1.77805868,
                        1.27077078,
                        1.18965051,
                    ],
                    [
                        1.61898958,
                        1.11364274,
                        1.44364738,
                        0.9361292,
                        1.34435804,
                        1.05360936,
                        1.24873014,
                        1.75792905,
                        1.19790056,
                        1.14207836,
                    ],
                    [
                        1.28617459,
                        1.00387152,
                        0.91909753,
                        1.00041709,
                        0.83567829,
                        0.61076833,
                        0.87099997,
                        0.90827492,
                        1.04933694,
                        0.75481523,
                    ],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [
                        -1.39611824,
                        -1.39641603,
                        -1.67593239,
                        -0.45435125,
                        -1.30565055,
                        -1.14315217,
                        -1.16531328,
                        -1.75208037,
                        -0.91144856,
                        -1.20208605,
                    ],
                    [
                        -1.99219371,
                        -2.59138925,
                        -4.06696788,
                        -1.01156424,
                        -2.87855823,
                        -2.04325337,
                        -1.86322556,
                        -3.32331569,
                        -2.06247119,
                        -2.41667238,
                    ],
                ]
            ),
            atol=min(max_atol, 0.02),
            rtol=min(max_rtol, 0.5),
        )

    def test_post2(self):
        np.testing.assert_allclose(
            self.post2,
            np.array(
                [
                    [
                        -0.52690577,
                        -0.4715939,
                        -0.36183996,
                        -0.32414943,
                        -0.91275036,
                        -1.20464412,
                        -0.53649153,
                        -0.53424599,
                        -0.17486563,
                        -0.64009102,
                    ],
                    [
                        0.13960272,
                        0.07430689,
                        1.20721667,
                        0.51614097,
                        0.15821339,
                        0.16708544,
                        0.15951641,
                        0.79152901,
                        1.32990199,
                        -0.60087287,
                    ],
                    [
                        1.20015992,
                        0.70737728,
                        1.64223188,
                        1.21192978,
                        0.92281252,
                        1.04721168,
                        0.79654251,
                        1.28795665,
                        1.68626851,
                        0.25138685,
                    ],
                    [
                        1.45291242,
                        1.09950355,
                        1.49144334,
                        1.24994881,
                        1.52977727,
                        1.30250982,
                        1.09980782,
                        1.83160369,
                        1.59508934,
                        0.88542369,
                    ],
                    [
                        1.39519879,
                        1.29707267,
                        1.89537825,
                        1.34566072,
                        1.57058053,
                        1.34524571,
                        0.8432895,
                        1.83913713,
                        1.96249772,
                        0.92134167,
                    ],
                    [
                        1.05652733,
                        1.01682158,
                        1.16999037,
                        1.32749258,
                        0.98511004,
                        1.25423487,
                        0.66119846,
                        1.2613252,
                        1.25564428,
                        0.48800746,
                    ],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [
                        -1.15632245,
                        -0.88634856,
                        -0.77583557,
                        -1.1710059,
                        -1.33363731,
                        -0.35665739,
                        -1.50772735,
                        -1.23046941,
                        -1.03444225,
                        -0.73559183,
                    ],
                    [
                        -1.89692013,
                        -1.36920399,
                        -0.6712459,
                        -2.38006835,
                        -2.33193894,
                        -0.1106398,
                        -2.86704798,
                        -1.96512643,
                        -1.62278425,
                        -1.73532027,
                    ],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.5),
        )


class Test_shared:
    sim_dat = sim12(5000, c=0, seed=0, family=GAMMALS([LOG(), LOG()]), n_ranef=20)

    # We again need to model the mean: \mu_i = \alpha + f(x0) + f(x1) + f_{x4}(x0)
    sim_formula_m = Formula(
        lhs("y"), [i(), f(["x0"], id=1), f(["x1"]), fs(["x0"], rf="x4")], data=sim_dat
    )

    # and the standard deviation as well: log(\sigma_i) = \alpha + f(x2) + f(x3)
    sim_formula_sd = Formula(lhs("y"), [i(), f(["x2"], id=1), f(["x3"])], data=sim_dat)

    family = GAMMALS([LOG(), LOGb(-0.0001)])

    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["control_lambda"] = 1
    test_kwargs["extend_lambda"] = False
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["method"] = "QR/Chol"
    test_kwargs["repara"] = True
    test_kwargs["prefit_grad"] = True

    # Now define the model and fit!
    gsmm_fam = GAMLSSGSMMFamily(2, family)
    model = GSMM([sim_formula_m, sim_formula_sd], gsmm_fam)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(**test_kwargs)

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            102.77775912832247,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_GAMcoef(self):
        coef = self.model.coef
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    [2.16073927e00],
                    [-3.07512718e-01],
                    [8.04409623e-03],
                    [4.19311809e-01],
                    [6.29220355e-01],
                    [5.02930132e-01],
                    [4.70450097e-01],
                    [-7.38919041e-02],
                    [-4.10762252e-01],
                    [-6.71402469e-01],
                    [-9.09691808e-01],
                    [-4.20806585e-01],
                    [-1.37328325e-01],
                    [1.38818681e-01],
                    [5.02864247e-01],
                    [1.03349000e00],
                    [1.67936430e00],
                    [2.30239738e00],
                    [2.96343163e00],
                    [1.20608308e-02],
                    [3.34539643e-03],
                    [-4.75971735e-02],
                    [1.39688415e-01],
                    [4.22994513e-02],
                    [2.88947316e-02],
                    [1.52413607e-01],
                    [5.38642364e-02],
                    [-5.02107109e-02],
                    [-4.00748077e-01],
                    [2.08112341e-03],
                    [-3.33145826e-03],
                    [-8.99365243e-02],
                    [-1.07324062e-01],
                    [1.00861001e-01],
                    [-1.41085592e-01],
                    [2.90312086e-01],
                    [-1.10658261e-01],
                    [3.93930569e-03],
                    [1.51003582e-01],
                    [5.89482490e-03],
                    [-6.87222215e-03],
                    [4.95274478e-03],
                    [-2.09061926e-04],
                    [3.73769601e-02],
                    [5.55290804e-02],
                    [-3.44913327e-02],
                    [-9.76479972e-03],
                    [-4.60496181e-02],
                    [1.05127874e-01],
                    [9.91474573e-03],
                    [3.72290025e-03],
                    [-5.01952590e-02],
                    [-1.22346719e-01],
                    [5.17546822e-02],
                    [9.77505058e-02],
                    [1.79072768e-02],
                    [7.60513854e-02],
                    [4.34602818e-02],
                    [4.72167506e-01],
                    [-6.66563906e-03],
                    [1.84173174e-04],
                    [2.74873400e-02],
                    [-4.04690372e-02],
                    [5.84483055e-02],
                    [-1.52034889e-01],
                    [-1.28054640e-04],
                    [-5.01067289e-02],
                    [-1.69397331e-01],
                    [-1.52963783e00],
                    [1.17796923e-02],
                    [-1.24983253e-02],
                    [9.22875337e-03],
                    [6.37496283e-02],
                    [5.78667995e-02],
                    [6.53371647e-02],
                    [-3.13283589e-03],
                    [1.89639411e-01],
                    [2.41709547e-01],
                    [3.54416654e-01],
                    [-1.33693965e-02],
                    [-7.42420081e-03],
                    [-2.38859679e-02],
                    [-1.77319105e-02],
                    [-7.52292454e-03],
                    [5.32579814e-02],
                    [-2.80027685e-01],
                    [2.82941609e-02],
                    [-1.18734812e-01],
                    [-7.17711669e-01],
                    [3.89207632e-03],
                    [5.14316408e-05],
                    [1.26187844e-01],
                    [5.93138683e-03],
                    [4.26402420e-02],
                    [-2.85999786e-01],
                    [-1.00947102e-01],
                    [-2.08465970e-01],
                    [-6.37369339e-02],
                    [1.36285403e-01],
                    [-1.35994921e-02],
                    [-4.27416973e-04],
                    [-4.15079371e-02],
                    [-8.02770470e-02],
                    [-2.46207393e-02],
                    [1.42594412e-01],
                    [5.24535545e-02],
                    [6.89739426e-02],
                    [-1.48757241e-01],
                    [2.82632357e-02],
                    [-6.00536181e-03],
                    [-1.74408979e-03],
                    [1.32461039e-03],
                    [3.54341043e-02],
                    [3.92090408e-02],
                    [1.82962441e-01],
                    [-9.24063214e-02],
                    [1.00428618e-01],
                    [1.04097702e-02],
                    [1.79686135e-01],
                    [-4.04792988e-03],
                    [2.36922904e-03],
                    [3.08742383e-02],
                    [9.97214072e-03],
                    [1.37506845e-01],
                    [1.02895277e-01],
                    [1.91448922e-01],
                    [5.87752841e-02],
                    [-6.18351018e-02],
                    [2.55713064e-01],
                    [-1.15324737e-02],
                    [2.11327296e-02],
                    [-5.15599337e-02],
                    [5.82906550e-02],
                    [-9.07263320e-02],
                    [-1.38711514e-01],
                    [7.05916802e-02],
                    [-1.36520545e-01],
                    [-1.38360078e-01],
                    [3.86816807e-01],
                    [-1.08428116e-03],
                    [2.78971712e-03],
                    [-3.48845629e-02],
                    [-1.14498325e-01],
                    [-8.33459766e-02],
                    [-8.16314062e-02],
                    [-1.60173317e-01],
                    [-6.62094310e-03],
                    [4.47671872e-03],
                    [1.85456163e-01],
                    [4.86268726e-03],
                    [-4.14665154e-03],
                    [4.08267791e-02],
                    [1.03036853e-01],
                    [-3.33139278e-02],
                    [-1.81934119e-01],
                    [-4.73323907e-02],
                    [9.10427553e-02],
                    [1.57500932e-02],
                    [5.23048074e-01],
                    [1.15638026e-02],
                    [-1.08343774e-02],
                    [9.26605956e-02],
                    [-1.74312422e-02],
                    [-6.83285438e-02],
                    [1.70064825e-01],
                    [-7.98847921e-02],
                    [2.28657078e-02],
                    [6.83522022e-02],
                    [1.30809585e-01],
                    [-4.84207399e-03],
                    [1.33424789e-02],
                    [-3.27998760e-02],
                    [-4.56985828e-02],
                    [-1.17981135e-01],
                    [1.41266318e-01],
                    [1.14970128e-01],
                    [-2.01241394e-01],
                    [2.51213574e-01],
                    [3.10169808e-01],
                    [-4.23848580e-03],
                    [-6.52226441e-03],
                    [-2.48306490e-02],
                    [-3.68010532e-02],
                    [-9.67470869e-02],
                    [-1.50964278e-01],
                    [-2.69077025e-02],
                    [-2.41578657e-01],
                    [1.45601728e-01],
                    [-2.94313781e-01],
                    [9.92867912e-03],
                    [-3.90955421e-03],
                    [-9.04479412e-02],
                    [4.53467662e-02],
                    [-1.52905620e-01],
                    [5.05802577e-02],
                    [1.97914598e-01],
                    [3.57754296e-01],
                    [7.16199982e-02],
                    [-5.19204073e-01],
                    [-1.39238446e-02],
                    [1.09518468e-03],
                    [3.34452955e-02],
                    [-4.45868598e-02],
                    [5.61066596e-02],
                    [-3.52582431e-03],
                    [-2.54898399e-01],
                    [-2.15069173e-02],
                    [-5.82873305e-02],
                    [2.42651536e-01],
                    [3.29342163e-01],
                    [-4.02792180e-01],
                    [7.92918806e-01],
                    [1.10560481e00],
                    [2.52504862e-01],
                    [5.39065509e-01],
                    [4.45502043e-01],
                    [2.54452710e-01],
                    [-2.76220770e-01],
                    [-7.20655190e-01],
                    [4.65378246e-02],
                    [1.14313450e-02],
                    [-1.00072229e-02],
                    [-2.87114782e-02],
                    [-4.34546739e-02],
                    [-5.70503294e-02],
                    [-6.78691230e-02],
                    [-6.54348446e-02],
                    [-6.45649006e-02],
                ]
            ),
            atol=min(max_atol, 0.5),
            rtol=min(max_rtol, 0.5),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        np.testing.assert_allclose(
            lam,
            np.array(
                [
                    31.222097276578545,
                    4.07116662141881,
                    3.9187656008516716,
                    11302.32885672775,
                    2.3587359640672965,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 2.5),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -15526.867514090196, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -15331.745185948903, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_edf1(self):
        compute_bias_corrected_edf(self.model, overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model.term_edf1])
        np.testing.assert_allclose(
            edf1,
            np.array(
                [
                    6.713831642615902,
                    4.921127633272276,
                    102.24427068803632,
                    6.9147867430206835,
                    1.358634183808314,
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
            np.array([0.0, 0.0, 0.0, 0.07909902713107153]),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.75),
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
                    38.39622887783163,
                    2835.761707764005,
                    243.20455192686484,
                    3.7792868990207036,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.5),
        )


class Test_shared_qefs:
    sim_dat = sim12(5000, c=0, seed=0, family=GAMMALS([LOG(), LOG()]), n_ranef=20)

    # We again need to model the mean: \mu_i = \alpha + f(x0) + f(x1) + f_{x4}(x0)
    sim_formula_m = Formula(
        lhs("y"), [i(), f(["x0"], id=1), f(["x1"]), fs(["x0"], rf="x4")], data=sim_dat
    )

    # and the standard deviation as well: log(\sigma_i) = \alpha + f(x2) + f(x3)
    sim_formula_sd = Formula(lhs("y"), [i(), f(["x2"], id=1), f(["x3"])], data=sim_dat)

    family = GAMMALS([LOG(), LOGb(-0.0001)])

    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["control_lambda"] = 1
    test_kwargs["extend_lambda"] = False
    test_kwargs["max_outer"] = 20
    test_kwargs["max_inner"] = 500
    test_kwargs["method"] = "qEFS"
    test_kwargs["repara"] = True
    test_kwargs["prefit_grad"] = True
    test_kwargs["max_restarts"] = -1
    test_kwargs["sample_hessian"] = True
    test_kwargs["structured_qefs"] = True
    test_kwargs["n_cores"] = 4
    test_kwargs["sqEFS_options"] = {
        "dampen_HBB": 0.1,
        "dampen_HBb": 1,
        "pre_cond": False,
    }

    bfgs_options = {
        "gtol": 1e-7,
        "ftol": 1e-7,
        "maxcor": 30,
        "maxls": 100,
        "maxfun": 5000,
    }

    test_kwargs["bfgs_options"] = bfgs_options

    def callback(outer, pen_llk, coef, lam):
        print(outer, pen_llk, lam)

    # Now define the model and fit!
    gsmm_fam = GAMLSSGSMMFamily(2, family)
    model = GSMM([sim_formula_m, sim_formula_sd], gsmm_fam)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(**test_kwargs, callback=callback)

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            104.55529268735076,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.2),
        )

    def test_GAMcoef(self):
        coef = np.round(self.model.coef, decimals=6)
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    [2.160859e00],
                    [-3.019260e-01],
                    [1.493000e-02],
                    [4.273060e-01],
                    [6.373100e-01],
                    [5.064040e-01],
                    [4.789290e-01],
                    [-7.297900e-02],
                    [-4.102110e-01],
                    [-6.700340e-01],
                    [-9.107060e-01],
                    [-4.200180e-01],
                    [-1.359820e-01],
                    [1.399430e-01],
                    [5.040770e-01],
                    [1.035524e00],
                    [1.680524e00],
                    [2.302925e00],
                    [2.962823e00],
                    [1.132600e-02],
                    [3.345000e-03],
                    [-4.359900e-02],
                    [1.325220e-01],
                    [3.859000e-02],
                    [2.723800e-02],
                    [1.485620e-01],
                    [5.280300e-02],
                    [-5.111000e-02],
                    [-3.998180e-01],
                    [2.085000e-03],
                    [-3.333000e-03],
                    [-8.573000e-02],
                    [-1.028280e-01],
                    [9.646600e-02],
                    [-1.354060e-01],
                    [2.857480e-01],
                    [-1.111570e-01],
                    [3.560000e-03],
                    [1.507860e-01],
                    [5.628000e-03],
                    [-6.577000e-03],
                    [5.487000e-03],
                    [7.910000e-04],
                    [3.620700e-02],
                    [5.319800e-02],
                    [-3.357100e-02],
                    [-1.033900e-02],
                    [-4.652500e-02],
                    [1.049880e-01],
                    [9.428000e-03],
                    [3.644000e-03],
                    [-4.795200e-02],
                    [-1.158770e-01],
                    [5.046600e-02],
                    [9.349200e-02],
                    [1.688100e-02],
                    [7.421900e-02],
                    [4.286600e-02],
                    [4.715620e-01],
                    [-6.137000e-03],
                    [2.660000e-04],
                    [2.703400e-02],
                    [-3.789300e-02],
                    [5.592300e-02],
                    [-1.470580e-01],
                    [1.239000e-03],
                    [-4.895400e-02],
                    [-1.692410e-01],
                    [-1.530721e00],
                    [1.117200e-02],
                    [-1.183100e-02],
                    [9.310000e-03],
                    [6.175800e-02],
                    [5.494800e-02],
                    [6.415700e-02],
                    [-1.826000e-03],
                    [1.876970e-01],
                    [2.403420e-01],
                    [3.545210e-01],
                    [-1.267200e-02],
                    [-7.036000e-03],
                    [-2.263900e-02],
                    [-1.585900e-02],
                    [-8.165000e-03],
                    [5.389500e-02],
                    [-2.738780e-01],
                    [2.679400e-02],
                    [-1.186030e-01],
                    [-7.192430e-01],
                    [3.770000e-03],
                    [3.300000e-05],
                    [1.204120e-01],
                    [4.968000e-03],
                    [4.138200e-02],
                    [-2.765200e-01],
                    [-9.885200e-02],
                    [-2.067370e-01],
                    [-6.382000e-02],
                    [1.353960e-01],
                    [-1.303300e-02],
                    [-3.210000e-04],
                    [-3.886500e-02],
                    [-7.658500e-02],
                    [-2.268400e-02],
                    [1.380480e-01],
                    [5.127200e-02],
                    [6.724600e-02],
                    [-1.487760e-01],
                    [2.883700e-02],
                    [-5.767000e-03],
                    [-1.655000e-03],
                    [9.800000e-04],
                    [3.373400e-02],
                    [3.886400e-02],
                    [1.773770e-01],
                    [-9.028800e-02],
                    [9.953100e-02],
                    [9.604000e-03],
                    [1.800130e-01],
                    [-3.973000e-03],
                    [2.177000e-03],
                    [3.041000e-02],
                    [1.045100e-02],
                    [1.307760e-01],
                    [9.989700e-02],
                    [1.888810e-01],
                    [5.912000e-02],
                    [-6.200100e-02],
                    [2.547730e-01],
                    [-1.105000e-02],
                    [2.012100e-02],
                    [-4.849000e-02],
                    [5.514900e-02],
                    [-8.624000e-02],
                    [-1.329910e-01],
                    [6.799300e-02],
                    [-1.361920e-01],
                    [-1.388500e-01],
                    [3.876100e-01],
                    [-1.203000e-03],
                    [2.817000e-03],
                    [-3.197700e-02],
                    [-1.101640e-01],
                    [-8.070700e-02],
                    [-7.922700e-02],
                    [-1.574210e-01],
                    [-7.414000e-03],
                    [3.368000e-03],
                    [1.871460e-01],
                    [4.448000e-03],
                    [-3.833000e-03],
                    [3.858600e-02],
                    [9.820000e-02],
                    [-3.109600e-02],
                    [-1.743010e-01],
                    [-4.545300e-02],
                    [8.935300e-02],
                    [1.535000e-02],
                    [5.229660e-01],
                    [1.082900e-02],
                    [-1.027600e-02],
                    [8.840500e-02],
                    [-1.628400e-02],
                    [-6.541600e-02],
                    [1.648950e-01],
                    [-7.904500e-02],
                    [2.279300e-02],
                    [6.819100e-02],
                    [1.320600e-01],
                    [-4.667000e-03],
                    [1.263200e-02],
                    [-3.142300e-02],
                    [-4.189500e-02],
                    [-1.135150e-01],
                    [1.364650e-01],
                    [1.139900e-01],
                    [-2.003110e-01],
                    [2.506040e-01],
                    [3.098840e-01],
                    [-4.294000e-03],
                    [-6.098000e-03],
                    [-2.368900e-02],
                    [-3.308600e-02],
                    [-9.197100e-02],
                    [-1.466430e-01],
                    [-2.724400e-02],
                    [-2.389600e-01],
                    [1.448050e-01],
                    [-2.941750e-01],
                    [9.437000e-03],
                    [-3.635000e-03],
                    [-8.642200e-02],
                    [4.384100e-02],
                    [-1.463230e-01],
                    [4.937800e-02],
                    [1.927330e-01],
                    [3.537970e-01],
                    [7.132500e-02],
                    [-5.196380e-01],
                    [-1.335800e-02],
                    [9.000000e-04],
                    [3.297600e-02],
                    [-4.146100e-02],
                    [5.412100e-02],
                    [-3.121000e-03],
                    [-2.495630e-01],
                    [-2.129100e-02],
                    [-5.841400e-02],
                    [2.418940e-01],
                    [3.294900e-01],
                    [-4.384970e-01],
                    [7.378490e-01],
                    [1.054339e00],
                    [1.878210e-01],
                    [4.856550e-01],
                    [3.860240e-01],
                    [2.030690e-01],
                    [-3.083200e-01],
                    [-7.088660e-01],
                    [6.614500e-02],
                    [-5.042000e-03],
                    [-4.355900e-02],
                    [-6.567500e-02],
                    [-7.451800e-02],
                    [-8.043500e-02],
                    [-8.396300e-02],
                    [-7.001500e-02],
                    [-5.973100e-02],
                ]
            ),
            atol=min(max_atol, 0.5),
            rtol=min(max_rtol, 0.5),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        np.testing.assert_allclose(
            lam,
            np.array(
                [
                    30.911746855402686,
                    4.3516658497816705,
                    3.7369488920606613,
                    1393.9569727138053,
                    2.22643814913205,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 5),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -15507.1725031031, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -15332.199526832077, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_edf1(self):
        compute_bias_corrected_edf(self.model, overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model.term_edf1])
        np.testing.assert_allclose(
            edf1,
            np.array(
                [
                    6.788467635821958,
                    4.845035260681829,
                    119.25933972358285,
                    6.9067432166190805,
                    2.209226171258083,
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
            np.array([0.0, 0.0, 0.0, 0.1443052734859107]),
            atol=min(max_atol, 0.15),
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
                    53.796589281793814,
                    2630.3104920856767,
                    242.58795873779306,
                    4.620478620568921,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 2.5),
        )


class Test_mvn:
    sim_dat = sim16(500, seed=1134, correlate=True)

    # We need formulas for each mean
    formulas = [
        Formula(lhs("y0"), [i(), f(["x0"])], data=sim_dat),
        Formula(lhs("y1"), [i(), f(["x1"]), f(["x2"])], data=sim_dat),
        Formula(lhs("y2"), [i(), f(["x3"])], data=sim_dat),
    ]

    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["control_lambda"] = 2
    test_kwargs["extend_lambda"] = False
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500

    model = GSMM(formulas, MultiGauss(3, [Identity() for _ in range(3)]))
    model.fit(**test_kwargs)

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            25.793745260368667,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_GAMcoef(self):
        coef = self.model.coef
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    [1.31082693],
                    [-0.83904211],
                    [0.79829314],
                    [1.40874654],
                    [1.63370038],
                    [1.55070996],
                    [1.18823694],
                    [0.64193597],
                    [-0.1754198],
                    [-0.99554403],
                    [6.61798949],
                    [-1.7169128],
                    [-0.80645375],
                    [-0.15750756],
                    [0.22460516],
                    [0.85492211],
                    [2.02662297],
                    [3.40240258],
                    [4.38124648],
                    [5.37682581],
                    [-9.02417999],
                    [2.5243377],
                    [4.34755476],
                    [-3.49392843],
                    [-1.86807161],
                    [-2.68325549],
                    [-5.44785351],
                    [-4.43098558],
                    [-0.28990515],
                    [-0.00627918],
                    [0.04225959],
                    [0.02787751],
                    [0.02610044],
                    [0.020369],
                    [-0.00374639],
                    [-0.05163652],
                    [-0.11645258],
                    [-0.18593697],
                    [-0.25542956],
                    [-0.26799386],
                    [-0.09528978],
                    [0.16788946],
                    [-0.31081509],
                    [-0.023681],
                    [-0.35592018],
                ]
            ),
            atol=min(max_atol, 10),
            rtol=min(max_rtol, 10),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -1278.3141943943695, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -1217.3645666891293, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_edf1(self):
        compute_bias_corrected_edf(self.model, overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model.term_edf1])
        np.testing.assert_allclose(
            edf1,
            np.array(
                [
                    4.441788680673209,
                    4.199295349145187,
                    8.733107670679118,
                    1.7491247296088577,
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
            np.array([0.0, 0.0, 0.0, 0.5845488651154345]),
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
                    87.26216463615783,
                    165.50250394044605,
                    1157.3462255099012,
                    0.8883147952436188,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.5),
        )


class Test_mvn_qefs:
    # Simulate data
    sim_dat = sim16(500, seed=1134, correlate=True)

    # We need formulas for each mean
    formulas = [
        Formula(lhs("y0"), [i(), f(["x0"])], data=sim_dat),
        Formula(lhs("y1"), [i(), f(["x1"]), f(["x2"])], data=sim_dat),
        Formula(lhs("y2"), [i(), f(["x3"])], data=sim_dat),
    ]

    # Now define the model and fit!
    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["control_lambda"] = 2
    test_kwargs["extend_lambda"] = False
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["method"] = "qEFS"
    test_kwargs["repara"] = False
    test_kwargs["prefit_grad"] = True
    test_kwargs["progress_bar"] = True
    test_kwargs["max_restarts"] = 0
    test_kwargs["sample_hessian"] = False
    test_kwargs["structured_qefs"] = True
    test_kwargs["structured_qefs_budget"] = 10
    test_kwargs["n_cores"] = 4
    test_kwargs["sqEFS_options"] = {
        "dampen_HBB": 0.1,
        "dampen_HBb": 1,
        "pre_cond": False,
    }

    bfgs_options = {
        "gtol": 1e-7,
        "ftol": 1e-7,
        "maxcor": 30,
        "maxls": 100,
        "maxfun": 5000,
    }

    test_kwargs["bfgs_options"] = bfgs_options

    model = GSMM(formulas, MultiGauss(3, [Identity() for _ in range(3)]))
    pre_pars = model.get_pars()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(**test_kwargs)

    # get deviance residuals
    res = model.get_resid()
    res0 = model.get_resid(mean=0)

    # Get means and theta
    Xs = model.get_mmat()
    mus, theta = model.family.predict(model.coef, model.coef_split_idx, Xs)
    R, _ = model.family.getR(theta)

    # Get cov and rvs
    size = 1000
    seed = 1
    rvs = model.family.rvs(mus, theta, size=size, seed=seed)
    cov = np.linalg.inv(R.T @ R)

    # Check qEFS correct_VP/Vp code
    Vs = correct_VB(model, grid_type="JJJ3", method="qEFS", n_c=1)

    def test_coef_extract1(self):
        assert self.pre_pars is None

    def test_coef_extract2(self):
        # Check all coef
        np.testing.assert_allclose(
            self.model.get_pars(), self.model.coef, atol=0, rtol=0
        )

    def test_coef_extract3(self):
        # Check extra coef
        np.testing.assert_allclose(
            self.model.get_pars(par=3),
            np.split(self.model.coef, self.model.coef_split_idx)[-1],
            atol=0,
            rtol=0,
        )

    def test_coef_extract4(self):
        # Check coef of smooth term 1 in second formula
        np.testing.assert_allclose(
            self.model.get_pars(par=1, term=1), self.model.coef[11:20], atol=0, rtol=0
        )

    def test_GAMedf2(self):
        np.testing.assert_allclose(
            self.Vs[5],
            30.131739815076845,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_GAMedf3(self):
        np.testing.assert_allclose(
            self.Vs[7],
            30.131739815076845,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            28.20975476875229,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_GAMcoef(self):
        coef = self.model.coef
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    [1.31088836],
                    [-0.82992739],
                    [0.77090672],
                    [1.38012868],
                    [1.60956509],
                    [1.53250234],
                    [1.17423677],
                    [0.63181722],
                    [-0.19002177],
                    [-1.02264714],
                    [6.61800758],
                    [-1.71344221],
                    [-0.74808577],
                    [-0.15409663],
                    [0.27305012],
                    [0.93811064],
                    [2.06263213],
                    [3.43254006],
                    [4.37482423],
                    [5.32231485],
                    [-7.81498926],
                    [3.65854246],
                    [5.83704275],
                    [-2.29480441],
                    [-0.73467666],
                    [-1.32432241],
                    [-4.1362203],
                    [-3.75242783],
                    [-0.64853146],
                    [-0.00634907],
                    [0.04554204],
                    [0.02746899],
                    [0.02286081],
                    [0.01309488],
                    [-0.01265117],
                    [-0.05798398],
                    [-0.11635888],
                    [-0.17673058],
                    [-0.23673186],
                    [-0.26808252],
                    [-0.09508469],
                    [0.1681341],
                    [-0.31397236],
                    [-0.02463047],
                    [-0.35608044],
                ]
            ),
            atol=min(max_atol, 8),
            rtol=min(max_rtol, 8),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        np.testing.assert_allclose(
            lam,
            np.array(
                [
                    7.332279866837958,
                    7.6673172610934115,
                    0.0105283713667257,
                    420.4579418495015,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 3),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -1299.0945514750829, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -1219.0721578384341, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_edf1(self):
        compute_bias_corrected_edf(self.model, overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model.term_edf1])
        np.testing.assert_allclose(
            edf1,
            np.array(
                [
                    6.012496635579059,
                    4.413735735599454,
                    8.998486141423832,
                    1.7070231617859537,
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
            np.array([0.0, 0.0, 0.0, 0.5712410678886176]),
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
                    218.96559216075173,
                    2019.5546930105306,
                    10213.89163830023,
                    0.8945811694633254,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.5),
        )

    def test_res(self):

        np.testing.assert_allclose(
            self.res[:, 0].reshape(-1, 1),
            self.res0,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1e-6),
        )

    def test_fitted(self):
        np.testing.assert_allclose(
            np.concatenate(self.model.mus, axis=1),
            self.mus,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1e-6),
        )

    def test_rvs(self):

        seed = self.seed
        size = self.size

        rvs2 = np.zeros_like(self.rvs)

        for mui in range(self.mus.shape[0]):
            rvs2[:, :, mui] = scp.stats.multivariate_normal.rvs(
                size=size,
                mean=self.mus[mui, :],
                cov=self.cov,
                random_state=seed,
            )

            seed += 1

        np.testing.assert_allclose(
            self.rvs, rvs2, atol=min(max_atol, 0), rtol=min(max_rtol, 1e-6)
        )


class Test_Nuts:
    sim_dat = sim4(500, 2, family=Gamma(), seed=0)

    # We again need to model the mean: \mu_i = \alpha + f(x0) + f(x1) + f_{x4}(x0)
    sim_formula_m = Formula(
        lhs("y"), [i(), f(["x0"]), f(["x1"]), f(["x2"]), f(["x3"])], data=sim_dat
    )

    # and the standard deviation
    sim_formula_sd = Formula(lhs("y"), [i()], data=sim_dat)

    family = GAMMALS([LOG(), LOGb(-0.0001)])

    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["control_lambda"] = 2
    test_kwargs["extend_lambda"] = False
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["method"] = "qEFS"
    test_kwargs["repara"] = False
    test_kwargs["prefit_grad"] = True
    test_kwargs["sample_hessian"] = True
    test_kwargs["structured_qefs"] = False

    # Now define the model and fit!
    gsmm_fam = GAMLSSGSMMFamily(2, family)
    model = GSMM([sim_formula_m, sim_formula_sd], gsmm_fam)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(**test_kwargs)

    def test_NUTS(self):
        res = sample_mssm(
            self.model,
            auto_converge=True,
            M_adapt=100,
            parallelize_chains=False,
            n_chains=1,
            sample_rho=True,
            delta=0.6,
            n_iter=100,
        )

        llks, coef_samples, rho_samples = res.lps, res.coefs, res.rhos

        assert (
            rho_samples.shape == (1, 100, len(self.model.overall_penalties))
            and coef_samples.shape == (1, 100, len(self.model.coef))
            and llks.shape == (1, 100, 1)
        )


class Test_no_pen:
    # No penalties
    sim_dat, _ = sim1(100, random_seed=100)

    # Specify formula
    formula = Formula(lhs("y"), [i(), *li(["x", "fact"])], data=sim_dat)

    # ... and model
    model = GAMM(formula, Gaussian())

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["max_outer"] = 100
    test_kwargs["max_inner"] = 1

    model.fit(**test_kwargs)

    test_kwargs2 = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs2["method"] = "qEFS"

    # Now fit with GSMM
    # Now define the model and fit!
    gsmm_fam = GAMLSSGSMMFamily(1, Gaussian())
    gsmm = GSMM([formula], gsmm_fam)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gsmm.fit(**test_kwargs2)

    def test_coef(self):
        np.testing.assert_allclose(
            ((self.model.coef - self.gsmm.coef[:-1]) / self.gsmm.coef[:-1]),
            np.array(
                [
                    [-1.18189822e-05],
                    [-1.24714928e-05],
                    [-6.19873768e-06],
                    [-2.34373385e-05],
                    [-4.64458871e-05],
                    [-3.90664436e-05],
                ]
            ),
            atol=min(max_atol, 0.01),
            rtol=min(max_rtol, 1e-3),
        )


class Test_PSD_sqEFS:
    sim_dat = sim4(500, 2, family=Gamma(), seed=0)

    # We again need to model the mean: \mu_i = \alpha + f(x0) + f(x1) + f_{x4}(x0)
    sim_formula_m = Formula(
        lhs("y"), [i(), f(["x0"]), f(["x1"]), f(["x2"]), f(["x3"])], data=sim_dat
    )

    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["control_lambda"] = 1
    test_kwargs["extend_lambda"] = False
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["method"] = "qEFS"
    test_kwargs["repara"] = True
    test_kwargs["prefit_grad"] = True
    test_kwargs["structured_qefs"] = True

    test_kwargs["sqEFS_options"] = {
        "dampen_HBB": 0.1,
        "dampen_HBb": 1,
        "pre_cond": True,
        "PD_HBB": False,
    }

    # Now define the model and fit!
    gsmm_fam = GAMLSSGSMMFamily(2, Gamma())
    model = GSMM([sim_formula_m], gsmm_fam)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(**test_kwargs)

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            18.732245790906493,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_GAMcoef(self):
        coef = self.model.coef
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    [8.45354739],
                    [-0.8735335],
                    [0.30013066],
                    [1.20086454],
                    [1.69771368],
                    [1.58446224],
                    [0.94304784],
                    [0.2101506],
                    [-0.31309056],
                    [-0.79870264],
                    [-1.29263285],
                    [-1.33706054],
                    [-1.1854484],
                    [-0.60177176],
                    [0.14851162],
                    [1.46134001],
                    [2.887152],
                    [4.37663541],
                    [5.9642136],
                    [-5.43016981],
                    [6.2284757],
                    [9.59311486],
                    [0.43199816],
                    [2.21536543],
                    [2.38300593],
                    [-1.07742878],
                    [-2.14559854],
                    [-1.08845497],
                    [-0.05317477],
                    [0.18789601],
                    [0.30798841],
                    [0.30102977],
                    [0.21957997],
                    [0.07506113],
                    [-0.10952617],
                    [-0.33172341],
                    [-0.56277454],
                    [0.97763473],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.5),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        np.testing.assert_allclose(
            lam,
            np.array(
                [
                    2.3871503689043743,
                    3.2610224772895804,
                    0.013459685480859459,
                    37.76010512658826,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 2.5),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -4513.09626171529, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -4477.562504190566, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_edf1(self):
        compute_bias_corrected_edf(self.model, overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model.term_edf1])
        np.testing.assert_allclose(
            edf1,
            np.array(
                [
                    4.495552217924287,
                    4.198053626450506,
                    7.831578363579387,
                    2.5417272688279735,
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
            np.array([0.0, 0.0, 0.0, 0.12082735068125461]),
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
                    57.680315094570034,
                    465.2549450546292,
                    1327.879824946911,
                    5.454546919656915,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.5),
        )
