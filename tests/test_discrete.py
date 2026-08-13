# flake8: noqa
import mssm
from mssm.models import *
from mssm.src.python.compare import compare_CDL
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


class Test_algorithms:
    sim_dat = sim4(500, 2, family=Gamma(), seed=0)

    sim_formula_m1 = Formula(
        lhs("y"),
        [i(), f(["x0"]), f(["x1"]), f(["x1", "x2"], te=True), f(["x3"])],
        data=sim_dat,
        discretize=True,
        find_nested=True,
    )

    _, cov_flat, _, _, _, _, _, _ = sim_formula_m1.encode_data(
        sim_formula_m1.data, prediction=True, discretize=True
    )
    sim_formula_m1.cov_flat = cov_flat
    sim_formula_m1.discretize_cov = False
    _ = build_penalties(sim_formula_m1)
    mmat1 = build_model_matrix(sim_formula_m1)

    sim_formula_m2 = Formula(
        lhs("y"),
        [i(), f(["x0"]), f(["x1"]), f(["x1", "x2"], te=True), f(["x3"])],
        data=sim_dat,
        discretize=True,
        find_nested=True,
    )
    pen = build_penalties(sim_formula_m2)
    mmat2 = build_model_matrix(sim_formula_m2)

    C = np.random.rand(mmat2.shape[1] * 30).reshape(mmat2.shape[1], 30)
    D = np.random.rand(500 * 30).reshape(30, 500)
    test_coef = np.random.rand(mmat2.shape[1]).reshape(-1, 1)
    test_coef2 = np.random.rand(2 * mmat2.shape[1]).reshape(-1, 2)

    mmat3 = mmat2 @ C
    mmat4 = D @ mmat2
    mmat5 = 10 * mmat2
    mmat6 = 10 * mmat2[:, :20]

    def testIndexC(self):
        assert np.abs(self.mmat2[:50, 0:3] - self.mmat1[:50, 0:3]).max() < 1e-7

    def testIndexC2(self):
        assert np.abs(self.mmat2[:50, 3] - self.mmat1[:50, 3]).max() < 1e-7

    def testIndexPre(self):
        assert (
            np.abs(
                self.mmat4[19:30, 15:30].toarray()
                - self.D[19:30, :] @ self.mmat1[:, 15:30]
            ).max()
            < 1e-7
        )

    def testIndexPre2(self):
        assert (
            np.abs(
                self.mmat4[20:30, 15:30] - self.D[20:30, :] @ self.mmat1[:, 15:30]
            ).max()
            < 1e-7
        )

    def testIndexPreT(self):
        assert (
            np.abs(
                self.mmat4.T[19:30, 15:30].toarray()
                - (self.D @ self.mmat1).T[19:30, 15:30]
            ).max()
            < 1e-7
        )

    def testIndexPreT2(self):
        assert (
            np.abs(
                self.mmat4.T[20:30, 15:30] - (self.D @ self.mmat1).T[20:30, 15:30]
            ).max()
            < 1e-7
        )

    def testIndexPost(self):
        assert (
            np.abs(
                self.mmat3[19:30, 15:30].toarray()
                - self.mmat1[19:30, :] @ self.C[:, 15:30]
            ).max()
            < 1e-7
        )

    def testIndexPost2(self):
        assert (
            np.abs(
                self.mmat3[20:30, :].toarray() - (self.mmat1 @ self.C)[20:30, :]
            ).max()
            < 1e-7
        )

    def testIndexPostT(self):
        assert (
            np.abs(
                self.mmat3.T[19:30, 15:30].toarray()
                - self.C.T[19:30, :] @ self.mmat1.T[:, 15:30]
            ).max()
            < 1e-7
        )

    def testXTy(self):
        assert (
            np.abs(
                (self.mmat1.T @ self.sim_formula_m2.y_flat)
                - (self.mmat2.T @ self.sim_formula_m2.y_flat)
            ).max()
            < 1e-6
        )

    def testXTY(self):
        assert (
            np.abs(
                (
                    self.mmat1.T
                    @ np.array(
                        [
                            self.sim_formula_m2.y_flat.flatten(),
                            self.sim_formula_m2.y_flat.flatten() * -0.5,
                        ]
                    ).T
                )
                - (
                    self.mmat2.T
                    @ np.array(
                        [
                            self.sim_formula_m2.y_flat.flatten(),
                            self.sim_formula_m2.y_flat.flatten() * -0.5,
                        ]
                    ).T
                )
            ).max()
            < 1e-6
        )

    def testXb(self):
        assert (
            np.abs((self.mmat1 @ self.test_coef) - (self.mmat2 @ self.test_coef)).max()
            < 1e-7
        )

    def testXB(self):
        assert (
            np.abs(
                (self.mmat1 @ self.test_coef2) - (self.mmat2 @ self.test_coef2)
            ).max()
            < 1e-7
        )

    def testXTX(self):
        assert (
            np.abs((self.mmat1.T @ self.mmat1) - (self.mmat2.T @ self.mmat2)).max()
            < 1e-7
        )

    def testXTXIndex(self):
        assert (
            np.abs(
                (self.mmat1[:25, :25].T @ self.mmat1[:25, :25])
                - (self.mmat2[:25, :25].T @ self.mmat2[:25, :25])
            ).max()
            < 1e-7
        )

    def testRMatMulT(self):
        assert (
            np.abs(
                self.mmat6[:, 0].T @ self.mmat5.toarray()
                - self.mmat6[:, 0].T @ self.mmat5
            ).max()
            < 1e-7
        )

    def testRMatMulT2(self):
        assert (
            np.abs(
                self.mmat6[0, :].toarray() @ self.mmat6.toarray().T
                - self.mmat6[0, :].toarray() @ self.mmat6.T
            ).max()
            < 1e-7
        )

    def testXTZ(self):
        assert (
            np.abs(
                self.mmat5.T @ self.mmat6
                - (self.mmat5.T.toarray() @ self.mmat6.toarray())
            ).max()
            < 1e-6
        )

    def testNpIndxe1(self):
        assert self.mmat2[0, :5].shape == self.mmat2.toarray()[0, :5].shape

    def testNpIndxe2(self):
        assert self.mmat2[0, 1].shape == self.mmat2.toarray()[0, 1].shape

    def testNpIndxe3(self):
        assert self.mmat2[0, [1]].shape == self.mmat2.toarray()[0, [1]].shape

    def testNpIndxe4(self):
        assert self.mmat2[0, [1, 2]].shape == self.mmat2.toarray()[0, [1, 2]].shape

    def testNpIndxe5(self):
        assert (
            self.mmat2[0, np.array([1, 2])].shape
            == self.mmat2.toarray()[0, np.array([1, 2])].shape
        )

    def testNpIndxe6(self):
        assert self.mmat2[[0], :5].shape == self.mmat2.toarray()[[0], :5].shape

    def testNpIndxe7(self):
        assert self.mmat2[[0], 1].shape == self.mmat2.toarray()[[0], 1].shape

    def testNpIndxe8(self):
        assert self.mmat2[[0], [1]].shape == self.mmat2.toarray()[[0], [1]].shape

    def testNpIndxe9(self):
        assert self.mmat2[[0], [1, 2]].shape == self.mmat2.toarray()[[0], [1, 2]].shape

    def testNpIndxe10(self):
        assert (
            self.mmat2[[0], np.array([1, 2])].shape
            == self.mmat2.toarray()[[0], np.array([1, 2])].shape
        )

    def testNpIndxe11(self):
        assert self.mmat2[[0, 1], :5].shape == self.mmat2.toarray()[[0, 1], :5].shape

    def testNpIndxe12(self):
        assert self.mmat2[[0, 1], 1].shape == self.mmat2.toarray()[[0, 1], 1].shape

    def testNpIndxe13(self):
        assert self.mmat2[[0, 1], [1]].shape == self.mmat2.toarray()[[0, 1], [1]].shape

    def testNpIndxe14(self):
        assert (
            self.mmat2[[0, 1], [1, 2]].shape
            == self.mmat2.toarray()[[0, 1], [1, 2]].shape
        )

    def testNpIndxe15(self):
        assert (
            self.mmat2[[0, 1], np.array([1, 2])].shape
            == self.mmat2.toarray()[[0, 1], np.array([1, 2])].shape
        )

        # Transpose

    def testNpIndxe16(self):
        assert self.mmat2.T[0, :5].shape == self.mmat2.toarray().T[0, :5].shape

    def testNpIndxe17(self):
        assert self.mmat2.T[0, 1].shape == self.mmat2.toarray().T[0, 1].shape

    def testNpIndxe18(self):
        assert self.mmat2.T[0, [1]].shape == self.mmat2.toarray().T[0, [1]].shape

    def testNpIndxe19(self):
        assert self.mmat2.T[0, [1, 2]].shape == self.mmat2.toarray().T[0, [1, 2]].shape

    def testNpIndxe20(self):
        assert (
            self.mmat2.T[0, np.array([1, 2])].shape
            == self.mmat2.toarray().T[0, np.array([1, 2])].shape
        )

    def testNpIndxe21(self):
        assert self.mmat2.T[[0], :5].shape == self.mmat2.toarray().T[[0], :5].shape

    def testNpIndxe22(self):
        assert self.mmat2.T[[0], 1].shape == self.mmat2.toarray().T[[0], 1].shape

    def testNpIndxe23(self):
        assert self.mmat2.T[[0], [1]].shape == self.mmat2.toarray().T[[0], [1]].shape

    def testNpIndxe24(self):
        assert (
            self.mmat2.T[[0], [1, 2]].shape == self.mmat2.toarray().T[[0], [1, 2]].shape
        )

    def testNpIndxe25(self):
        assert (
            self.mmat2.T[[0], np.array([1, 2])].shape
            == self.mmat2.toarray().T[[0], np.array([1, 2])].shape
        )

    def testNpIndxe26(self):
        assert (
            self.mmat2.T[[0, 1], :5].shape == self.mmat2.toarray().T[[0, 1], :5].shape
        )

    def testNpIndxe27(self):
        assert self.mmat2.T[[0, 1], 1].shape == self.mmat2.toarray().T[[0, 1], 1].shape

    def testNpIndxe28(self):
        assert (
            self.mmat2.T[[0, 1], [1]].shape == self.mmat2.toarray().T[[0, 1], [1]].shape
        )

    def testNpIndxe29(self):
        assert (
            self.mmat2.T[[0, 1], [1, 2]].shape
            == self.mmat2.toarray().T[[0, 1], [1, 2]].shape
        )

    def testNpIndxe30(self):
        assert (
            self.mmat2.T[[0, 1], np.array([1, 2])].shape
            == self.mmat2.toarray().T[[0, 1], np.array([1, 2])].shape
        )


class Test_GSMM:
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

    model = GSMM(formulas, MultiGauss(3, [Identity() for _ in range(3)]))
    model.fit(**test_kwargs)

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            25.10218162447965,
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
                    [-0.59331707],
                    [0.78756107],
                    [1.35808945],
                    [1.5196968],
                    [1.37363527],
                    [0.98205442],
                    [0.44220833],
                    [-0.40092235],
                    [-1.15219686],
                    [6.61798949],
                    [-1.57225428],
                    [-0.53531507],
                    [0.04227462],
                    [0.43120953],
                    [1.15790488],
                    [2.33707751],
                    [3.66936567],
                    [4.41443138],
                    [5.34129387],
                    [-6.23811349],
                    [6.43335623],
                    [5.56535205],
                    [-1.62452649],
                    [0.48079646],
                    [-0.7382446],
                    [-3.24232253],
                    [-2.96302565],
                    [1.42633009],
                    [-0.00627918],
                    [0.04324352],
                    [0.02437675],
                    [0.01461572],
                    [-0.00139048],
                    [-0.0294582],
                    [-0.07006832],
                    [-0.11707651],
                    [-0.15732075],
                    [-0.20111756],
                    [-0.27113372],
                    [-0.09060747],
                    [0.16940554],
                    [-0.32340814],
                    [-0.02800592],
                    [-0.35655767],
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
                    4.8713492824237,
                    6.1463685800070715,
                    0.010584211536727029,
                    599.8159257139712,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 2.5),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -1322.5385150572274, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
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
                    4.375799468199502,
                    3.9422771357324673,
                    8.510191584484986,
                    1.4807393921427447,
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
            np.array([0.0, 0.0, 0.0, 0.5879459444127463]),
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
                    85.48292084972877,
                    157.9214062755835,
                    1108.4251678221162,
                    0.6766049150064095,
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
    model.fit(**test_kwargs)

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            18.582756168040998,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_GAMcoef(self):
        coef = self.model.coef
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    [8.43011197],
                    [-0.75215712],
                    [0.31562305],
                    [1.18405507],
                    [1.65481747],
                    [1.55101384],
                    [0.94664059],
                    [0.23194423],
                    [-0.67402261],
                    [-1.46042456],
                    [-1.39906931],
                    [-1.07147589],
                    [-0.79148863],
                    [-0.22498688],
                    [0.60443417],
                    [1.95292989],
                    [3.38125868],
                    [4.84009737],
                    [6.54771136],
                    [-4.3625093],
                    [9.01634572],
                    [7.60509322],
                    [0.497342],
                    [2.67224996],
                    [2.10527736],
                    [-1.33431961],
                    [-1.55994964],
                    [-3.19522002],
                    [-0.07100366],
                    [0.28896089],
                    [0.43621859],
                    [0.38897004],
                    [0.28259207],
                    [0.10566728],
                    [-0.09074481],
                    [-0.34674109],
                    [-0.58637654],
                    [0.96625356],
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
                    3.199691606870352,
                    3.672885653523253,
                    0.012871544107903593,
                    21.648353025426527,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 2.5),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -4507.936791446896, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -4473.12874047772, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_edf1(self):
        compute_bias_corrected_edf(self.model, overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model.term_edf1])
        np.testing.assert_allclose(
            edf1,
            np.array(
                [
                    4.197496597907076,
                    3.719533288161946,
                    8.168029803077655,
                    2.667368932187713,
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
            np.array([0.0, 0.0, 0.0, 0.09469171220246564]),
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
                    65.17408389530277,
                    378.8797242103188,
                    928.0258256708686,
                    5.795651022778046,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.5),
        )


class Test_GAM:
    sim_dat = sim4(500, 2, family=Gamma(), seed=0)

    sim_formula_m1 = Formula(
        lhs("y"),
        [i(), f(["x0"]), f(["x1"]), f(["x2"], te=False), f(["x3"])],
        data=sim_dat,
        discretize=True,
        find_nested=True,
    )

    _, cov_flat, _, _, _, _, _, _ = sim_formula_m1.encode_data(
        sim_formula_m1.data, prediction=True, discretize=True
    )
    sim_formula_m1.cov_flat = cov_flat
    sim_formula_m1.discretize_cov = False

    sim_formula_m2 = Formula(
        lhs("y"),
        [i(), f(["x0"]), f(["x1"]), f(["x2"], te=False), f(["x3"])],
        data=sim_dat,
        discretize=True,
        find_nested=True,
    )

    model1 = GAMM(sim_formula_m1, Gamma())
    model1.fit(method="Chol")

    model = GAMM(sim_formula_m2, Gamma())
    model.fit(method="Chol")

    def test_coef_diff(self):
        assert (
            np.abs(np.array(self.model.coef) - np.array(self.model1.coef)).max() < 1e-7
        )

    def test_edf_diff(self):
        assert (
            np.abs(np.array(self.model.term_edf) - np.array(self.model1.term_edf)).max()
            < 1e-7
        )

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            16.217102086109605,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        np.testing.assert_allclose(
            sigma, 4.8690196365611, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    [8.44532501],
                    [-0.73300117],
                    [0.33067165],
                    [1.13126508],
                    [1.58108558],
                    [1.50232737],
                    [0.96116203],
                    [0.307195],
                    [-0.66194212],
                    [-1.52090779],
                    [-1.43501539],
                    [-1.06506671],
                    [-0.75927057],
                    [-0.20006361],
                    [0.65473182],
                    [2.00063642],
                    [3.44430471],
                    [4.79941558],
                    [6.40189429],
                    [-2.5616714],
                    [9.83664122],
                    [8.95210936],
                    [1.69736776],
                    [3.83990891],
                    [3.2440741],
                    [-0.03639162],
                    [-1.24093205],
                    [-2.56365836],
                    [-0.0237219],
                    [0.16427374],
                    [0.25416091],
                    [0.23909758],
                    [0.16270811],
                    [0.03213445],
                    [-0.12244013],
                    [-0.3153924],
                    [-0.5023955],
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
                    17.011127889091096,
                    19.042454809845644,
                    0.08344490631172775,
                    201.3330048705406,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 2.5),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -4562.50111774846, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -4537.152264070222, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_edf1(self):
        compute_bias_corrected_edf(self.model, overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model.term_edf1])
        np.testing.assert_allclose(
            edf1,
            np.array(
                [
                    3.8068091829338613,
                    3.7177926652731523,
                    7.773343135810451,
                    2.269803064911821,
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
            np.array([1.0176132997363752e-06, 0.0, 0.0, 0.29480269422544003]),
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
                    9.677726601950233,
                    83.94745537083544,
                    96.40503803785347,
                    1.2164726810908715,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.5),
        )
