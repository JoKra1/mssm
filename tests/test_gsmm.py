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
    default_gsmm_test_kwargs,
    max_atol,
    max_rtol,
    init_penalties_tests_gammlss,
    init_penalties_tests_gsmm,
)
from mssm.src.python.mcmc import sample_mssm

mssm.src.python.exp_fam.GAUMLSS.init_lambda = init_penalties_tests_gammlss
mssm.src.python.exp_fam.GAMMALS.init_lambda = init_penalties_tests_gammlss
mssm.src.python.exp_fam.MULNOMLSS.init_lambda = init_penalties_tests_gammlss
mssm.src.python.exp_fam.PropHaz.init_lambda = init_penalties_tests_gsmm
mssm.src.python.utils.GAMLSSGSMMFamily.init_lambda = init_penalties_tests_gsmm

################################################################## Tests ##################################################################


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
            rtol=min(max_rtol, 2.5),
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
            104.11043984332206,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.2),
        )

    def test_GAMcoef(self):
        coef = np.round(self.model.coef, decimals=6)
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    [2.160864e00],
                    [-3.020230e-01],
                    [1.475600e-02],
                    [4.271040e-01],
                    [6.370190e-01],
                    [5.062180e-01],
                    [4.786030e-01],
                    [-7.319300e-02],
                    [-4.103900e-01],
                    [-6.698690e-01],
                    [-9.106850e-01],
                    [-4.200400e-01],
                    [-1.360100e-01],
                    [1.399220e-01],
                    [5.040610e-01],
                    [1.035494e00],
                    [1.680491e00],
                    [2.302906e00],
                    [2.962853e00],
                    [1.131700e-02],
                    [3.337000e-03],
                    [-4.359900e-02],
                    [1.324210e-01],
                    [3.860000e-02],
                    [2.723300e-02],
                    [1.485870e-01],
                    [5.281100e-02],
                    [-5.110800e-02],
                    [-3.998320e-01],
                    [2.081000e-03],
                    [-3.328000e-03],
                    [-8.567200e-02],
                    [-1.027450e-01],
                    [9.641600e-02],
                    [-1.354240e-01],
                    [2.858060e-01],
                    [-1.111280e-01],
                    [3.539000e-03],
                    [1.508050e-01],
                    [5.622000e-03],
                    [-6.572000e-03],
                    [5.469000e-03],
                    [7.700000e-04],
                    [3.619100e-02],
                    [5.319900e-02],
                    [-3.356200e-02],
                    [-1.032700e-02],
                    [-4.653500e-02],
                    [1.049990e-01],
                    [9.419000e-03],
                    [3.639000e-03],
                    [-4.792300e-02],
                    [-1.157870e-01],
                    [5.044500e-02],
                    [9.349900e-02],
                    [1.688100e-02],
                    [7.425200e-02],
                    [4.286500e-02],
                    [4.715890e-01],
                    [-6.137000e-03],
                    [2.630000e-04],
                    [2.698500e-02],
                    [-3.787900e-02],
                    [5.588700e-02],
                    [-1.470120e-01],
                    [1.162000e-03],
                    [-4.895500e-02],
                    [-1.692650e-01],
                    [-1.530730e00],
                    [1.116300e-02],
                    [-1.182200e-02],
                    [9.298000e-03],
                    [6.169800e-02],
                    [5.492900e-02],
                    [6.412700e-02],
                    [-1.829000e-03],
                    [1.877160e-01],
                    [2.403430e-01],
                    [3.545390e-01],
                    [-1.266100e-02],
                    [-7.032000e-03],
                    [-2.263800e-02],
                    [-1.586800e-02],
                    [-8.139000e-03],
                    [5.381900e-02],
                    [-2.739030e-01],
                    [2.681900e-02],
                    [-1.186510e-01],
                    [-7.192470e-01],
                    [3.767000e-03],
                    [3.100000e-05],
                    [1.203080e-01],
                    [4.963000e-03],
                    [4.134500e-02],
                    [-2.764830e-01],
                    [-9.887000e-02],
                    [-2.067350e-01],
                    [-6.383800e-02],
                    [1.353810e-01],
                    [-1.301800e-02],
                    [-3.210000e-04],
                    [-3.885100e-02],
                    [-7.652900e-02],
                    [-2.270700e-02],
                    [1.380100e-01],
                    [5.128900e-02],
                    [6.727500e-02],
                    [-1.487810e-01],
                    [2.881400e-02],
                    [-5.763000e-03],
                    [-1.654000e-03],
                    [9.770000e-04],
                    [3.371500e-02],
                    [3.880500e-02],
                    [1.773270e-01],
                    [-9.026600e-02],
                    [9.954800e-02],
                    [9.599000e-03],
                    [1.800480e-01],
                    [-3.968000e-03],
                    [2.174000e-03],
                    [3.037200e-02],
                    [1.042100e-02],
                    [1.307160e-01],
                    [9.985200e-02],
                    [1.888690e-01],
                    [5.911400e-02],
                    [-6.201100e-02],
                    [2.547910e-01],
                    [-1.103600e-02],
                    [2.010000e-02],
                    [-4.847000e-02],
                    [5.510700e-02],
                    [-8.618700e-02],
                    [-1.330080e-01],
                    [6.802800e-02],
                    [-1.361800e-01],
                    [-1.388490e-01],
                    [3.876210e-01],
                    [-1.199000e-03],
                    [2.810000e-03],
                    [-3.197000e-02],
                    [-1.100520e-01],
                    [-8.063100e-02],
                    [-7.920600e-02],
                    [-1.574580e-01],
                    [-7.393000e-03],
                    [3.366000e-03],
                    [1.870890e-01],
                    [4.445000e-03],
                    [-3.828000e-03],
                    [3.854800e-02],
                    [9.810900e-02],
                    [-3.109800e-02],
                    [-1.742910e-01],
                    [-4.549000e-02],
                    [8.935200e-02],
                    [1.533500e-02],
                    [5.229710e-01],
                    [1.082000e-02],
                    [-1.026600e-02],
                    [8.831800e-02],
                    [-1.628200e-02],
                    [-6.538200e-02],
                    [1.648420e-01],
                    [-7.902200e-02],
                    [2.281800e-02],
                    [6.818900e-02],
                    [1.320460e-01],
                    [-4.664000e-03],
                    [1.262500e-02],
                    [-3.137700e-02],
                    [-4.189900e-02],
                    [-1.134760e-01],
                    [1.364500e-01],
                    [1.139600e-01],
                    [-2.003000e-01],
                    [2.505810e-01],
                    [3.099090e-01],
                    [-4.283000e-03],
                    [-6.093000e-03],
                    [-2.365800e-02],
                    [-3.309900e-02],
                    [-9.191000e-02],
                    [-1.465900e-01],
                    [-2.720500e-02],
                    [-2.389790e-01],
                    [1.447770e-01],
                    [-2.942130e-01],
                    [9.428000e-03],
                    [-3.632000e-03],
                    [-8.634900e-02],
                    [4.379200e-02],
                    [-1.462510e-01],
                    [4.933000e-02],
                    [1.927510e-01],
                    [3.538430e-01],
                    [7.132400e-02],
                    [-5.196620e-01],
                    [-1.334200e-02],
                    [8.980000e-04],
                    [3.292800e-02],
                    [-4.144700e-02],
                    [5.409400e-02],
                    [-3.142000e-03],
                    [-2.496430e-01],
                    [-2.127800e-02],
                    [-5.843700e-02],
                    [2.419310e-01],
                    [3.294980e-01],
                    [-4.380640e-01],
                    [7.383870e-01],
                    [1.054881e00],
                    [1.885450e-01],
                    [4.861200e-01],
                    [3.867230e-01],
                    [2.035280e-01],
                    [-3.079280e-01],
                    [-7.092030e-01],
                    [6.573400e-02],
                    [-4.560000e-03],
                    [-4.274800e-02],
                    [-6.485200e-02],
                    [-7.393400e-02],
                    [-8.000000e-02],
                    [-8.352000e-02],
                    [-6.961400e-02],
                    [-5.919800e-02],
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
                    30.911273098167747,
                    4.352183603224263,
                    3.736828437514571,
                    1439.016790152987,
                    2.228446582494426,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 5),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -15507.167669639082, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -15332.22497719553, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_edf1(self):
        compute_bias_corrected_edf(self.model, overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model.term_edf1])
        np.testing.assert_allclose(
            edf1,
            np.array(
                [
                    6.7878354819141205,
                    4.84504964938634,
                    119.25437367377154,
                    6.906178817062174,
                    2.1930889513762137,
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
            np.array([0.0, 0.0, 0.0, 0.11794760560276107]),
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
                    53.793417631513805,
                    2636.023014251915,
                    242.62344340496455,
                    4.5733272875356015,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 2.5),
        )


class Test_mvn:
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

    bfgs_options = {
        "gtol": 1e-7,
        "ftol": 1e-7,
        "maxcor": 30,
        "maxls": 100,
        "maxfun": 5000,
    }

    test_kwargs["bfgs_options"] = bfgs_options

    model = GSMM(formulas, MultiGauss(3, [Identity() for _ in range(3)]))
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

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            27.796893748448205,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_GAMcoef(self):
        coef = self.model.coef
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    [1.31085255],
                    [-0.82700069],
                    [0.75118389],
                    [1.36193952],
                    [1.59491321],
                    [1.51905222],
                    [1.16122622],
                    [0.62005065],
                    [-0.19978613],
                    [-1.03364961],
                    [6.61798229],
                    [-1.70427927],
                    [-0.74244628],
                    [-0.16966458],
                    [0.27248373],
                    [0.94795],
                    [2.0577718],
                    [3.42681311],
                    [4.3719117],
                    [5.32095398],
                    [-8.14344531],
                    [3.32884735],
                    [5.422029],
                    [-2.63199994],
                    [-1.05668876],
                    [-1.70706796],
                    [-4.50295594],
                    [-3.93600855],
                    [-0.52875122],
                    [-0.00624929],
                    [0.04723265],
                    [0.02715231],
                    [0.02120844],
                    [0.01035369],
                    [-0.01613267],
                    [-0.06088083],
                    [-0.11776788],
                    [-0.17539974],
                    [-0.23243716],
                    [-0.2681111],
                    [-0.09533345],
                    [0.16817583],
                    [-0.31366122],
                    [-0.02427125],
                    [-0.35612782],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 5),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        np.testing.assert_allclose(
            lam,
            np.array(
                [
                    6.739596590851739,
                    8.800300445107826,
                    0.009777669982560133,
                    428.9017964308726,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 3),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -1293.1940638337865, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -1218.9501219830845, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_edf1(self):
        compute_bias_corrected_edf(self.model, overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model.term_edf1])
        np.testing.assert_allclose(
            edf1,
            np.array(
                [
                    5.626312975210457,
                    4.599043687932572,
                    8.999153674233385,
                    1.6795180442015647,
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
            np.array([0.0, 0.0, 0.0, 0.4337909849702717]),
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
                    146.05994394593205,
                    859.4280473472886,
                    3757.85867360191,
                    1.3228567960493294,
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
        llks, coef_samples, rho_samples = sample_mssm(
            self.model,
            auto_converge=True,
            M_adapt=100,
            parallelize_chains=False,
            n_chains=1,
            sample_rho=True,
            delta=0.6,
            n_iter=100,
        )

        assert (
            rho_samples.shape == (1, 100, len(self.model.overall_penalties))
            and coef_samples.shape == (1, 100, len(self.model.coef))
            and llks.shape == (1, 100, 1)
        )
