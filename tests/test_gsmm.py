# flake8: noqa
import mssm
from mssm.models import *
from mssm.src.python.utils import correct_VB, estimateVp
import numpy as np
import os
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

    # First fit with SR1
    bfgs_opt = {"gtol": 1e-9, "ftol": 1e-9, "maxcor": 30, "maxls": 200, "maxfun": 1e7}

    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["method"] = "qEFS"
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
    test_kwargs["qEFSH"] = "BFGS"
    test_kwargs["max_restarts"] = 0
    test_kwargs["overwrite_coef"] = True
    test_kwargs["qEFS_init_converge"] = True
    test_kwargs["prefit_grad"] = False

    # Now fit with BFGS
    model2.fit(**test_kwargs)

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            19.06605212367808,
            atol=min(max_atol, 0.5),
            rtol=min(max_rtol, 0.025),
        )

    def test_GAMedf2(self):
        np.testing.assert_allclose(
            self.model2.edf,
            20.085993154912806,
            atol=min(max_atol, 1.3),
            rtol=min(max_rtol, 0.07),
        )

    def test_GAMcoef(self):
        coef = self.model.coef.flatten()
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    7.64896332,
                    -0.87163571,
                    -0.84714219,
                    0.05745359,
                    1.19376365,
                    1.65146764,
                    0.45272538,
                    -0.54724611,
                    -0.76214353,
                    -0.76137743,
                    -1.8210173,
                    -1.04816578,
                    -0.41222123,
                    0.34134332,
                    1.0342988,
                    2.01876926,
                    3.18538175,
                    4.08327798,
                    5.17126124,
                    -8.62853231,
                    4.78466146,
                    5.76753761,
                    -1.98400143,
                    0.10081098,
                    -1.16777336,
                    -3.57420725,
                    -4.0600672,
                    -1.08798903,
                    -0.04889937,
                    -0.01052461,
                    0.01277757,
                    0.02972515,
                    0.04714203,
                    0.06622719,
                    0.08729984,
                    0.0969056,
                    0.10180631,
                    0.66763451,
                ]
            ),
            atol=min(max_atol, 0.7),
            rtol=min(max_rtol, 1e-6),
        )

    def test_GAMcoef2(self):
        coef = self.model2.coef.flatten()
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    7.64898791,
                    -0.87442422,
                    -0.7302071,
                    0.14596979,
                    1.27647368,
                    1.70078317,
                    0.56790503,
                    -0.41554354,
                    -0.73983263,
                    -0.90265256,
                    -1.83369152,
                    -1.02612303,
                    -0.38474768,
                    0.35920819,
                    1.05915048,
                    2.05300297,
                    3.20549805,
                    4.06563624,
                    5.12037278,
                    -8.52043318,
                    4.84751426,
                    5.85726955,
                    -1.89210962,
                    0.16967413,
                    -1.06785945,
                    -3.53406784,
                    -3.88453386,
                    -2.01468058,
                    -0.04870832,
                    -0.01048352,
                    0.0127276,
                    0.02960897,
                    0.04695781,
                    0.06596845,
                    0.08695881,
                    0.09652709,
                    0.10140869,
                    0.66877099,
                ]
            ),
            atol=min(max_atol, 0.3),
            rtol=min(max_rtol, 1e-6),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        np.testing.assert_allclose(
            lam,
            np.array([7.35504400e-01, 7.27801598e00, 8.04428324e-03, 1.00000000e07]),
            atol=min(max_atol, 1.5),
            rtol=min(max_rtol, 0.37),
        )

    def test_GAMlam2(self):
        lam = np.array([p.lam for p in self.model2.overall_penalties])
        np.testing.assert_allclose(
            lam,
            np.array([9.58110974e-01, 8.39545471e00, 8.87952624e-03, 1.00000000e07]),
            atol=min(max_atol, 15),
            rtol=min(max_rtol, 0.35),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -1080.9668585544648, atol=min(max_atol, 2), rtol=min(max_rtol, 1e-6)
        )

    def test_GAMreml2(self):
        reml = self.model2.get_reml()
        np.testing.assert_allclose(
            reml, -1083.887431273825, atol=min(max_atol, 2), rtol=min(max_rtol, 0.015)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -1043.3141308788472, atol=min(max_atol, 0.5), rtol=min(max_rtol, 0.001)
        )

    def test_GAMllk2(self):
        llk = self.model2.get_llk(False)
        np.testing.assert_allclose(
            llk, -1043.8544317953304, atol=min(max_atol, 0.5), rtol=min(max_rtol, 0.002)
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
    gsmm_newton = GSMM([copy.deepcopy(sim_formula_m)], gsmm_newton_fam)

    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["method"] = "QR/Chol"
    test_kwargs["extend_lambda"] = False
    test_kwargs["control_lambda"] = False
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["min_inner"] = 500
    test_kwargs["seed"] = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gsmm_newton.fit(**test_kwargs)

    resid = gsmm_newton.get_resid(resid_type="Martingale")

    gsmm_qefs_fam = PropHaz(ut, r)
    gsmm_qefs = GSMM([copy.deepcopy(sim_formula_m)], gsmm_qefs_fam)

    # Fit with qEFS update without initialization
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bfgs_opt = {
            "gtol": 1e-7,
            "ftol": 1e-7,
            "maxcor": 30,
            "maxls": 20,
            "maxfun": 100,
        }

        test_kwargs["method"] = "qEFS"
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

    def resid_test(self):
        assert self.resid.shape == (500, 1)

    def test_GAMcoef(self):

        np.testing.assert_allclose(
            (self.gsmm_newton.coef - self.gsmm_qefs.coef).flatten(),
            np.array(
                [
                    -5.44700213e-04,
                    -5.57025326e-04,
                    1.64013870e-04,
                    8.67670135e-04,
                    4.92493649e-04,
                    7.65198038e-04,
                    5.14628775e-04,
                    -9.50230485e-04,
                    -2.48488958e-03,
                    -6.72884363e-04,
                    -1.17596358e-03,
                    -6.75266364e-04,
                    6.72783782e-05,
                    1.07678488e-04,
                    -2.63805422e-07,
                    1.65661496e-03,
                    -1.61770568e-03,
                    -6.64843399e-03,
                    -4.78166505e-02,
                    -3.54112498e-02,
                    -4.53015190e-02,
                    -3.99937127e-02,
                    -4.29717381e-02,
                    -4.15610992e-02,
                    -4.73157234e-02,
                    -1.93293685e-02,
                    -2.61343991e-02,
                    5.38161813e-05,
                    -3.18588497e-04,
                    -3.70497505e-04,
                    7.79857596e-05,
                    5.19593795e-04,
                    5.72184426e-04,
                    -2.22179748e-04,
                    -1.67729712e-03,
                    -3.14518797e-03,
                ]
            ),
            atol=0.025,
        )

    def test_GAMlam(self):
        np.testing.assert_allclose(
            np.round(
                np.array(
                    [
                        p1.lam - p2.lam
                        for p1, p2 in zip(
                            self.gsmm_newton.overall_penalties,
                            self.gsmm_qefs.overall_penalties,
                        )
                    ]
                ),
                decimals=3,
            ),
            np.array([-0.098, 0.149, -0.0, 2.152]),
            atol=min(max_atol, 2),
            rtol=min(max_rtol, 0.35),
        )

    def test_GAMreml(self):
        np.testing.assert_allclose(
            self.gsmm_newton.get_reml() - self.gsmm_qefs.get_reml(),
            0.0907684010567209,
            atol=min(max_atol, 0.2),
            rtol=min(max_rtol, 4e-4),
        )

    def test_GAMllk(self):
        np.testing.assert_allclose(
            self.gsmm_newton.get_llk(True) - self.gsmm_qefs.get_llk(True),
            0.07796569660399655,
            atol=min(max_atol, 0.1),
            rtol=min(max_rtol, 9e-4),
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
    gsmm_newton = GSMM([copy.deepcopy(sim_formula_m)], gsmm_newton_fam)

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
        gsmm_newton.fit(**test_kwargs)

    gsmm_qefs_fam = PropHaz(ut, r)
    gsmm_qefs = GSMM([copy.deepcopy(sim_formula_m)], gsmm_qefs_fam)

    # Fit with qEFS update without initialization
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bfgs_opt = {
            "gtol": 1e-7,
            "ftol": 1e-7,
            "maxcor": 30,
            "maxls": 20,
            "maxfun": 100,
        }

        test_kwargs["method"] = "qEFS"
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

        np.testing.assert_allclose(
            (self.gsmm_newton.coef - self.gsmm_qefs.coef).flatten(),
            np.array(
                [
                    -6.00505655e-04,
                    -9.64003472e-04,
                    -3.83549187e-04,
                    1.17537678e-03,
                    1.59927985e-03,
                    7.38051198e-04,
                    -4.44997108e-04,
                    -1.80074704e-03,
                    -3.11562028e-03,
                    -1.82326872e-03,
                    3.14446254e-03,
                    3.46507392e-03,
                    2.31289573e-03,
                    2.99595003e-03,
                    3.18175605e-03,
                    1.01399062e-03,
                    1.37963294e-02,
                    3.14823158e-02,
                    -1.02728366e-01,
                    -7.73625299e-02,
                    -9.77585419e-02,
                    -8.96668581e-02,
                    -9.31593570e-02,
                    -9.24209095e-02,
                    -1.01698088e-01,
                    -4.87830255e-02,
                    -1.39815381e-02,
                    -9.37696374e-05,
                    -2.59179471e-03,
                    -3.05352437e-03,
                    -2.36146501e-04,
                    3.03758717e-03,
                    3.90425367e-03,
                    -7.14453093e-05,
                    -7.63733239e-03,
                    -1.53736159e-02,
                ]
            ),
            atol=min(max_atol, 0.05),
            rtol=min(max_rtol, 5e-6),
        )

    def test_GAMlam(self):
        np.testing.assert_allclose(
            np.array(
                [
                    p1.lam - p2.lam
                    for p1, p2 in zip(
                        self.gsmm_newton.overall_penalties,
                        self.gsmm_qefs.overall_penalties,
                    )
                ]
            ),
            np.array(
                [-9.87297096e-03, -5.29255787e-01, -2.50321169e-04, 1.10708062e01]
            ),
            atol=min(max_atol, 5),
            rtol=min(max_rtol, 1.1),
        )

    def test_GAMreml(self):
        np.testing.assert_allclose(
            self.gsmm_newton.get_reml() - self.gsmm_qefs.get_reml(),
            2.9739911355409276,
            atol=min(max_atol, 0.5),
            rtol=min(max_rtol, 1),
        )

    def test_GAMllk(self):
        np.testing.assert_allclose(
            self.gsmm_newton.get_llk(True) - self.gsmm_qefs.get_llk(True),
            0.21546192785831408,
            atol=min(max_atol, 0.2),
            rtol=min(max_rtol, 2e-4),
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
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["method"] = "QR/Chol"
    test_kwargs["repara"] = True
    test_kwargs["prefit_grad"] = True
    test_kwargs["method"] = "qEFS"

    bfgs_opt = {"gtol": 1e-7, "ftol": 1e-7, "maxcor": 30, "maxls": 20, "maxfun": 100}

    test_kwargs["bfgs_options"] = bfgs_opt

    # Now define the model and fit!
    gsmm_fam = GAMLSSGSMMFamily(2, family)
    model = GSMM([sim_formula_m, sim_formula_sd], gsmm_fam)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(**test_kwargs)

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            133.43747528447437,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_GAMcoef(self):
        coef = self.model.coef
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    [2.16799851e00],
                    [-3.11219604e-01],
                    [6.76269169e-03],
                    [4.20140604e-01],
                    [6.17972732e-01],
                    [5.22020441e-01],
                    [4.48950056e-01],
                    [-6.50733371e-02],
                    [-4.23976160e-01],
                    [-6.88927308e-01],
                    [-9.16685246e-01],
                    [-4.11960874e-01],
                    [-1.25097831e-01],
                    [1.52420289e-01],
                    [5.15211838e-01],
                    [1.04286559e00],
                    [1.69314981e00],
                    [2.30579264e00],
                    [2.95634309e00],
                    [3.75362661e-03],
                    [9.58028500e-04],
                    [-1.74939422e-02],
                    [4.86650755e-02],
                    [1.61098841e-02],
                    [1.20440949e-02],
                    [9.19379906e-02],
                    [4.24401022e-02],
                    [-4.80598711e-02],
                    [-3.98192431e-01],
                    [7.00659664e-04],
                    [-1.14589925e-03],
                    [-3.42967689e-02],
                    [-3.87385937e-02],
                    [4.14405578e-02],
                    [-7.59687391e-02],
                    [1.89786406e-01],
                    [-8.90506600e-02],
                    [3.87773026e-03],
                    [1.49488123e-01],
                    [1.79145734e-03],
                    [-2.26379646e-03],
                    [1.23469061e-03],
                    [-1.21154322e-03],
                    [1.50792147e-02],
                    [2.63347195e-02],
                    [-2.08030259e-02],
                    [-8.77894284e-03],
                    [-4.38479669e-02],
                    [1.12445596e-01],
                    [3.47342619e-03],
                    [1.35214943e-03],
                    [-1.84493981e-02],
                    [-4.48212936e-02],
                    [2.41951033e-02],
                    [4.93061792e-02],
                    [7.60179066e-03],
                    [5.94313560e-02],
                    [4.05971441e-02],
                    [4.77132541e-01],
                    [-1.96325187e-03],
                    [2.24136773e-04],
                    [7.63534394e-03],
                    [-1.63258393e-02],
                    [2.26914484e-02],
                    [-7.61302683e-02],
                    [1.90841820e-04],
                    [-4.06361616e-02],
                    [-1.59559986e-01],
                    [-1.52317443e00],
                    [4.06978098e-03],
                    [-3.72335832e-03],
                    [4.07741839e-03],
                    [2.39747417e-02],
                    [2.33928304e-02],
                    [3.24292915e-02],
                    [-1.34212631e-03],
                    [1.48332236e-01],
                    [2.24186003e-01],
                    [3.52343780e-01],
                    [-4.37312146e-03],
                    [-2.53807399e-03],
                    [-9.99288422e-03],
                    [-8.03628555e-03],
                    [-2.58506068e-03],
                    [2.97376643e-02],
                    [-1.69558376e-01],
                    [1.83403627e-02],
                    [-1.12830423e-01],
                    [-7.24985659e-01],
                    [1.42941593e-03],
                    [-7.92898283e-05],
                    [4.56499873e-02],
                    [-5.23165229e-04],
                    [1.74092228e-02],
                    [-1.45237854e-01],
                    [-6.41605328e-02],
                    [-1.61986118e-01],
                    [-5.93535520e-02],
                    [1.30441298e-01],
                    [-4.11282910e-03],
                    [-6.86032369e-05],
                    [-1.50233367e-02],
                    [-3.02595492e-02],
                    [-1.12889828e-02],
                    [6.95094525e-02],
                    [3.19971719e-02],
                    [5.20275202e-02],
                    [-1.41656341e-01],
                    [3.32811700e-02],
                    [-2.36461043e-03],
                    [-8.33253959e-04],
                    [-4.09088931e-04],
                    [1.28232717e-02],
                    [1.47797375e-02],
                    [8.89688614e-02],
                    [-5.68381113e-02],
                    [8.09825969e-02],
                    [9.47155239e-03],
                    [1.84835627e-01],
                    [-9.73627363e-04],
                    [7.51218395e-04],
                    [1.05160414e-02],
                    [1.94581800e-03],
                    [5.62636046e-02],
                    [4.73383804e-02],
                    [1.14440944e-01],
                    [5.28810359e-02],
                    [-5.73622860e-02],
                    [2.50293801e-01],
                    [-3.30094939e-03],
                    [6.74206424e-03],
                    [-1.82323505e-02],
                    [1.99326986e-02],
                    [-3.65637229e-02],
                    [-7.09471845e-02],
                    [4.27466413e-02],
                    [-1.07151149e-01],
                    [-1.30017590e-01],
                    [3.86878306e-01],
                    [-2.41244282e-04],
                    [8.95023332e-04],
                    [-1.26697612e-02],
                    [-4.05179231e-02],
                    [-3.26662516e-02],
                    [-4.01513732e-02],
                    [-1.05181012e-01],
                    [-2.67133069e-03],
                    [4.98950630e-03],
                    [1.85542473e-01],
                    [1.54738394e-03],
                    [-8.09925955e-04],
                    [1.31737357e-02],
                    [3.36112363e-02],
                    [-1.31771000e-02],
                    [-8.46072802e-02],
                    [-3.06932427e-02],
                    [6.51580992e-02],
                    [1.40026696e-02],
                    [5.27480447e-01],
                    [3.73322443e-03],
                    [-3.49249330e-03],
                    [3.12624515e-02],
                    [-7.07543436e-03],
                    [-2.84260688e-02],
                    [8.25055101e-02],
                    [-4.68327199e-02],
                    [1.83659115e-02],
                    [6.57407090e-02],
                    [1.33424326e-01],
                    [-2.08824134e-03],
                    [5.13101580e-03],
                    [-7.44748380e-03],
                    [-2.07789570e-02],
                    [-5.37901894e-02],
                    [7.67237393e-02],
                    [8.11137029e-02],
                    [-1.68592247e-01],
                    [2.41163368e-01],
                    [3.06750568e-01],
                    [-1.08659988e-03],
                    [-1.80985927e-03],
                    [-5.53374832e-03],
                    [-1.33542583e-02],
                    [-3.61241553e-02],
                    [-6.82585542e-02],
                    [-1.15746479e-02],
                    [-1.84868436e-01],
                    [1.37491898e-01],
                    [-3.02475682e-01],
                    [3.11400774e-03],
                    [-1.17209216e-03],
                    [-3.37617670e-02],
                    [1.54775546e-02],
                    [-6.38832106e-02],
                    [2.55759096e-02],
                    [1.15841268e-01],
                    [2.82953144e-01],
                    [6.72059770e-02],
                    [-5.27091767e-01],
                    [-4.21802428e-03],
                    [-6.01075582e-05],
                    [1.22649687e-02],
                    [-1.67255052e-02],
                    [2.30622321e-02],
                    [-2.39969177e-03],
                    [-1.64353835e-01],
                    [-2.16207721e-02],
                    [-5.33093904e-02],
                    [2.43240882e-01],
                    [3.39574446e-01],
                    [-3.38385034e-01],
                    [8.11417321e-01],
                    [1.08682132e00],
                    [3.36857490e-01],
                    [5.33502086e-01],
                    [4.92955502e-01],
                    [2.79993577e-01],
                    [-2.50766508e-01],
                    [-7.46029538e-01],
                    [4.78746975e-02],
                    [1.06014463e-02],
                    [-1.20196029e-02],
                    [-3.12772737e-02],
                    [-4.59241022e-02],
                    [-5.88939726e-02],
                    [-6.86786150e-02],
                    [-6.45100087e-02],
                    [-6.19267987e-02],
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
                    41.17260229420759,
                    12.533666166891857,
                    4.092764472110347,
                    11118.20358119248,
                    4.15187531639246,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 2.5),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -15576.253064785291, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -15364.74212255429, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_edf1(self):
        compute_bias_corrected_edf(self.model, overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model.term_edf1])
        np.testing.assert_allclose(
            edf1,
            np.array(
                [
                    8.517720894639751,
                    6.248437814610253,
                    148.4517833120425,
                    8.56828260261986,
                    1.5076592131479796,
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
            np.array([0.0, 0.0, 0.0, 0.1245955153970093]),
            atol=min(max_atol, 0.13),
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
                    211.9695958713893,
                    1711.8754238885585,
                    399.1431710461262,
                    3.21233862560394,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.5),
        )
