# flake8: noqa
import mssm
from mssm.models import *
from mssm.src.python.utils import estimateVp, correct_VB
import numpy as np
import os
import copy
from mssmViz.sim import *
from .defaults import (
    default_gamm_test_kwargs,
    default_gammlss_test_kwargs,
    default_gsmm_test_kwargs,
    max_atol,
    max_rtol,
    init_penalties_tests_gammlss,
    init_penalties_tests_gsmm,
    init_coef_gammals_tests,
    init_coef_gaumlss_tests,
    init_coef_gsmmgammlss,
)

from mssm.src.python.mcmc import sample_mssm
from mssm.src.python.formula import build_model_matrix, build_penalties

mssm.src.python.exp_fam.GAUMLSS.init_coef = init_coef_gaumlss_tests
mssm.src.python.exp_fam.GAMMALS.init_coef = init_coef_gammals_tests
mssm.src.python.exp_fam.GAUMLSS.init_lambda = init_penalties_tests_gammlss
mssm.src.python.exp_fam.GAMMALS.init_lambda = init_penalties_tests_gammlss
mssm.src.python.exp_fam.MULNOMLSS.init_lambda = init_penalties_tests_gammlss
mssm.src.python.exp_fam.PropHaz.init_lambda = init_penalties_tests_gsmm
mssm.src.python.utils.GAMLSSGSMMFamily.init_lambda = init_penalties_tests_gsmm
mssm.src.python.utils.GAMLSSGSMMFamily.init_coef = init_coef_gsmmgammlss

################################################################## Tests ##################################################################


class Test_hazard:
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
    test_kwargs["force_sparse"] = False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(**test_kwargs)

    res = model.get_resid()

    def test_resid(self):
        assert self.res.shape == (500, 1)

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            17.701397445431855,
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
                    6.955990167024333,
                    6.718132388894035,
                    0.005366607577449247,
                    101.5200450799711,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 2.5),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -1809.8853389805333, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
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
                    4.314129203718054,
                    4.33018004907203,
                    8.872759352525629,
                    2.4734068397134235,
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
            np.array([0.0, 0.0, 0.0, 0.2106268444088315]),
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
                    72.6321270108752,
                    328.5041277852464,
                    463.9028467084391,
                    4.721875845689631,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.5),
        )


class Test_mvn:
    sim_dat = sim16(500, seed=1134, correlate=True)

    # We need formulas for each mean
    formulas = [
        Formula(lhs("y0"), [i(), f(["x0"])], data=sim_dat, discretize=True),
        Formula(lhs("y1"), [i(), f(["x1"]), f(["x2"])], data=sim_dat, discretize=True),
        Formula(lhs("y2"), [i(), f(["x3"])], data=sim_dat, discretize=True),
    ]

    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["control_lambda"] = 2
    test_kwargs["extend_lambda"] = False
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["force_sparse"] = False

    model = GSMM(formulas, MultiGauss(3, [Identity() for _ in range(3)]))
    model.fit(**test_kwargs)

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            25.10218162447966,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_GAMcoef(self):
        coef = np.round(self.model.coef, decimals=6)
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    [1.310827e00],
                    [-5.933170e-01],
                    [7.875610e-01],
                    [1.358089e00],
                    [1.519697e00],
                    [1.373635e00],
                    [9.820540e-01],
                    [4.422080e-01],
                    [-4.009220e-01],
                    [-1.152197e00],
                    [6.617989e00],
                    [-1.572254e00],
                    [-5.353150e-01],
                    [4.227500e-02],
                    [4.312100e-01],
                    [1.157905e00],
                    [2.337078e00],
                    [3.669366e00],
                    [4.414431e00],
                    [5.341294e00],
                    [-6.238113e00],
                    [6.433356e00],
                    [5.565352e00],
                    [-1.624526e00],
                    [4.807960e-01],
                    [-7.382450e-01],
                    [-3.242323e00],
                    [-2.963026e00],
                    [1.426330e00],
                    [-6.279000e-03],
                    [4.324400e-02],
                    [2.437700e-02],
                    [1.461600e-02],
                    [-1.390000e-03],
                    [-2.945800e-02],
                    [-7.006800e-02],
                    [-1.170770e-01],
                    [-1.573210e-01],
                    [-2.011180e-01],
                    [-2.711340e-01],
                    [-9.060700e-02],
                    [1.694060e-01],
                    [-3.234080e-01],
                    [-2.800600e-02],
                    [-3.565580e-01],
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
                    4.871349282423669,
                    6.146368580007333,
                    0.01058421153672705,
                    599.8159257138534,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 2.5),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -1322.5385150572272, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -1263.9536891697567, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_edf1(self):
        compute_bias_corrected_edf(self.model, overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model.term_edf1])
        np.testing.assert_allclose(
            edf1,
            np.array(
                [
                    4.375799468199504,
                    3.9422771357324335,
                    8.510191584484993,
                    1.4807393921428138,
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
            np.array([0.0, 0.0, 0.0, 0.5879459444127703]),
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
                    85.48292084972854,
                    157.9214062755832,
                    1108.4251678221026,
                    0.6766049150064073,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.5),
        )


class Test_GAMMLSS:
    sim_dat = sim4(500, 2, family=Gamma(), seed=0)

    # We again need to model the mean: \mu_i = \alpha + f(x0) + f(x1) + f_{x4}(x0)
    sim_formula_m = Formula(
        lhs("y"),
        [i(), f(["x0"]), f(["x1"]), f(["x2"]), f(["x3"])],
        data=sim_dat,
        discretize=True,
    )

    # and the standard deviation
    sim_formula_sd = Formula(lhs("y"), [i()], data=sim_dat, discretize=True)

    family = GAMMALS([LOG(), LOGb(-0.0001)])
    model = GAMMLSS([sim_formula_m, sim_formula_sd], family)

    test_kwargs = copy.deepcopy(default_gammlss_test_kwargs)
    test_kwargs["method"] = "Chol"
    test_kwargs["extend_lambda"] = False
    test_kwargs["control_lambda"] = False
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["min_inner"] = 500
    test_kwargs["seed"] = 0
    test_kwargs["repara"] = False
    test_kwargs["force_sparse"] = False
    model.fit(**test_kwargs)

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            18.587721769587947,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_GAMcoef(self):
        coef = self.model.coef
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    [8.43010289],
                    [-0.75219001],
                    [0.31558864],
                    [1.18420216],
                    [1.6550525],
                    [1.55110259],
                    [0.946649],
                    [0.23183613],
                    [-0.6743029],
                    [-1.46082742],
                    [-1.39931907],
                    [-1.0712142],
                    [-0.79092569],
                    [-0.22434319],
                    [0.60487442],
                    [1.95340302],
                    [3.38142988],
                    [4.83921224],
                    [6.54578361],
                    [-4.36396547],
                    [9.01581333],
                    [7.60201788],
                    [0.49778057],
                    [2.67076712],
                    [2.10395297],
                    [-1.3349491],
                    [-1.56030732],
                    [-3.19767848],
                    [-0.07076496],
                    [0.2884208],
                    [0.43536556],
                    [0.38831095],
                    [0.28221137],
                    [0.1054602],
                    [-0.09102615],
                    [-0.34708564],
                    [-0.5868364],
                    [0.96625508],
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
                    3.2001781716897626,
                    3.669221525719383,
                    0.01286519320517135,
                    21.648458909740256,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 2.5),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -4507.940328358384, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -4473.131663148686, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_edf1(self):
        compute_bias_corrected_edf(self.model, overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model.term_edf1])
        np.testing.assert_allclose(
            edf1,
            np.array(
                [
                    4.197559618916131,
                    3.7260254342458374,
                    8.168678322641252,
                    2.6694418767443078,
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
            np.array([0.0, 0.0, 0.0, 0.0949528775319819]),
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
                    65.22927527222122,
                    354.10754974620284,
                    929.6950264305922,
                    5.79295504929375,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.5),
        )


class Test_dropGSMM:
    sim_dat = sim13(5000, 2, c=0, seed=0, family=Gaussian(), binom_offset=0, n_ranef=20)

    formula = Formula(
        lhs("y"),
        [
            i(),
            l(["x5"]),
            l(["x6"]),
            f(["x0"], by="x5"),
            f(["x0"], by="x6"),
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
    test_kwargs["force_dense"] = True
    test_kwargs["force_sparse"] = False
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
        n_c=4,
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

    def test_edf1(self):
        np.testing.assert_allclose(
            self.res[-3],
            self.model.edf1,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.03),
        )

    def test_edf2(self):
        np.testing.assert_allclose(
            self.res2[5],
            20.50840353814276,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.03),
        )

    def test_VP(self):
        np.testing.assert_allclose(
            self.res2[2], self.Vp2, atol=min(max_atol, 0), rtol=min(max_rtol, 1e-7)
        )


class Test_dropGAMMLSS:
    sim_dat = sim13(5000, 2, c=0, seed=0, family=Gaussian(), binom_offset=0, n_ranef=20)

    formula = Formula(
        lhs("y"),
        [
            i(),
            l(["x5"]),
            l(["x6"]),
            f(["x0"], by="x5"),
            f(["x0"], by="x6"),
        ],
        data=sim_dat,
    )

    formula_sd = Formula(lhs("y"), [i()], data=sim_dat)

    model = GAMMLSS([formula, formula_sd], GAUMLSS([Identity(), LOGb(-0.001)]))

    test_kwargs = copy.deepcopy(default_gammlss_test_kwargs)
    test_kwargs["max_inner"] = 500
    test_kwargs["min_inner"] = 500
    test_kwargs["max_outer"] = 200
    test_kwargs["extend_lambda"] = False
    test_kwargs["control_lambda"] = 2
    test_kwargs["repara"] = False
    test_kwargs["method"] = "LU/Chol"
    test_kwargs["force_dense"] = True
    test_kwargs["force_sparse"] = False
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

    def test_edf1(self):
        np.testing.assert_allclose(
            self.res[-3],
            self.model.edf1,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.03),
        )

    def test_edf2(self):
        np.testing.assert_allclose(
            self.res2[5],
            20.530963672100402,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.03),
        )

    def test_VP(self):
        np.testing.assert_allclose(
            self.res2[2], self.Vp2, atol=min(max_atol, 0), rtol=min(max_rtol, 1e-7)
        )


class Test_dropGAMM:
    sim_dat = sim13(5000, 2, c=0, seed=0, family=Gaussian(), binom_offset=0, n_ranef=20)

    formula = Formula(
        lhs("y"),
        [
            i(),
            l(["x5"]),
            l(["x6"]),
            f(["x0"], by="x5"),
            f(["x0"], by="x6"),
        ],
        data=sim_dat,
    )

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["max_inner"] = 1
    test_kwargs["control_lambda"] = 2
    test_kwargs["extend_lambda"] = False
    test_kwargs["progress_bar"] = True
    test_kwargs["method"] = "QR"
    test_kwargs["force_dense"] = True
    test_kwargs["force_sparse"] = False
    model = GAMM(formula, Gaussian())
    model.fit(**test_kwargs)

    # More extensive selection + posterior sim checks
    res = correct_VB(
        model,
        grid_type="JJJ1",
        method="QR",
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
        method="QR",
        compute_Vcc=False,
        recompute_H=True,
        n_c=1,
        seed=20,
        VP_grid_type="JJJ2",
        only_expected_edf=False,
        prior=None,
        Vp_fidiff=False,
        verbose=True,
    )

    Vp2, _, _, _, _, _ = estimateVp(
        model,
        grid_type="JJJ2",
        n_c=1,
        seed=20,
        method="QR",
        prior=None,
        Vp_fidiff=False,
    )

    def test_edf1(self):
        np.testing.assert_allclose(
            self.res[-3],
            self.model.edf1,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.03),
        )

    def test_edf2(self):
        np.testing.assert_allclose(
            self.res2[5],
            19.506176085904492,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.03),
        )

    def test_VP(self):
        np.testing.assert_allclose(
            self.res2[2], self.Vp2, atol=min(max_atol, 0), rtol=min(max_rtol, 1e-7)
        )
