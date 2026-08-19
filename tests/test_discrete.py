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

################################################################## Tests ##################################################################


class Test_ar:
    # We simulate some data including a random smooth - but then dont include it in the model:
    sim_dat = sim11(5000, 2, c=0, seed=20, family=Gamma(), n_ranef=20, binom_offset=0)

    sim_dat = sim_dat.sort_values(["x4"], ascending=[True])

    sim_formula = Formula(
        lhs("y"),
        [i(), f(["x0"]), f(["x1"]), f(["x2"]), f(["x3"])],
        data=sim_dat,
        series_id="x4",
        discretize=True,
    )  # Can already specify this.

    model = GAMM(sim_formula, Gamma())
    model.fit(rho=0.25, progress_bar=False, method="Chol")

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            19.313135175313306,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        np.testing.assert_allclose(
            sigma, 8.35100302829088, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        np.testing.assert_allclose(
            np.round(coef, decimals=6),
            np.array(
                [
                    [8.7168100e00],
                    [-9.7355500e-01],
                    [-1.1773000e-01],
                    [5.2074200e-01],
                    [1.0909320e00],
                    [1.5104660e00],
                    [1.2234120e00],
                    [9.1791300e-01],
                    [2.6675500e-01],
                    [-3.6675000e-01],
                    [-1.6541380e00],
                    [-8.7552800e-01],
                    [-4.7543900e-01],
                    [4.7086000e-02],
                    [7.9532700e-01],
                    [1.9027800e00],
                    [3.3956800e00],
                    [4.7064160e00],
                    [6.0658560e00],
                    [-1.0898159e01],
                    [1.5461650e00],
                    [1.4995680e00],
                    [-6.2810150e00],
                    [-3.8108120e00],
                    [-4.7128670e00],
                    [-7.8822830e00],
                    [-5.3751500e00],
                    [-3.7001760e00],
                    [9.6890000e-03],
                    [7.4800000e-04],
                    [-9.5810000e-03],
                    [-2.0779000e-02],
                    [-2.5539000e-02],
                    [-1.8471000e-02],
                    [3.1860000e-03],
                    [3.7437000e-02],
                    [7.3028000e-02],
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
                    49.40876648625623,
                    59.65114891702245,
                    0.041734816670011284,
                    7113.070883276762,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 2.5),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -47070.78983904525, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -47025.2958060189, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_edf1(self):
        compute_bias_corrected_edf(self.model, overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model.term_edf1])
        np.testing.assert_allclose(
            edf1,
            np.array(
                [
                    4.929279678060681,
                    4.786449840150404,
                    8.961976246822278,
                    1.7193061988301837,
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
            np.array([0.0, 0.0, 0.0, 0.8507391005107705]),
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
                    50.98512414086201,
                    477.4277024095954,
                    563.1979577943996,
                    0.1674604521271632,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.5),
        )


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
    mmat2S = mmat2.copy()
    mmat2S.return_sparse = True

    C = np.random.rand(mmat2.shape[1] * 30).reshape(mmat2.shape[1], 30)
    D = np.random.rand(500 * 30).reshape(30, 500)
    test_coef = np.random.rand(mmat2.shape[1]).reshape(-1, 1)
    test_coef2 = np.random.rand(2 * mmat2.shape[1]).reshape(-1, 2)

    mmat3 = mmat2 @ C
    mmat3S = mmat3.copy()
    mmat3S.return_sparse = True
    mmat4 = D @ mmat2
    mmat5 = mmat2
    mmat6 = mmat2[:, :20]
    mmat7 = mmat4 @ C
    mmat8 = mmat2 * 3.5

    def testIndexDXC(self):
        assert (
            np.abs(
                self.mmat7[:20, 10:30].toarray()
                - (self.D @ self.mmat1 @ self.C)[:20, 10:30]
            ).max()
            < 1e-7
        )

    def testIndexDXCT(self):
        assert (
            np.abs(
                self.mmat7.T[:20, 10:30].toarray()
                - (self.D @ self.mmat1 @ self.C).T[:20, 10:30]
            ).max()
            < 1e-7
        )

    def testIndexXT(self):
        assert (
            np.abs(self.mmat2.T[:20, 10:30].toarray() - self.mmat1.T[:20, 10:30]).max()
            < 1e-7
        )

    def testIndexXC(self):
        assert np.abs(self.mmat3[:, :10] - (self.mmat1 @ self.C)[:, :10]).max() < 1e-7

    def testIndexCTXT(self):
        assert (
            np.abs(self.mmat3.T[:, :10] - (self.mmat1 @ self.C).T[:, :10]).max() < 1e-7
        )

    def testIndexXCS(self):
        assert np.abs(self.mmat3S[:, :10] - (self.mmat1 @ self.C)[:, :10]).max() < 1e-7

    def testIndexCTXTS(self):
        assert (
            np.abs(self.mmat3S.T[:, :10] - (self.mmat1 @ self.C).T[:, :10]).max() < 1e-7
        )

    def testCOOXS(self):
        assert isinstance(self.mmat2S[:, 0], scp.sparse.coo_array)

    def testCOOXb(self):
        assert isinstance(
            self.mmat2S @ scp.sparse.coo_array(self.test_coef.flatten()),
            scp.sparse.coo_array,
        )

    def testCOOXTy(self):
        assert isinstance(
            self.mmat2S.T @ scp.sparse.coo_array(self.sim_formula_m2.y_flat.flatten()),
            scp.sparse.coo_array,
        )

    def testCSCXb(self):
        assert isinstance(
            self.mmat2S @ scp.sparse.csc_array(self.test_coef),
            scp.sparse.csc_array,
        )

    def testCSCXTy(self):
        assert isinstance(
            self.mmat2S.T @ scp.sparse.csc_array(self.sim_formula_m2.y_flat),
            scp.sparse.csc_array,
        )

    def testMulFloat(self):
        assert np.abs(self.mmat8.toarray() - (self.mmat1 * 3.5)).max() < 1e-7

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
            < 1e-4
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
            < 1e-4
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


class Test_algorithms2:
    sim_dat = sim13(5000, 2, c=0, seed=0, family=Gaussian(), binom_offset=0, n_ranef=20)

    sim_formula_m1 = Formula(
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
        discretize=True,
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
        [
            i(),
            l(["x5"]),
            l(["x6"]),
            f(["x0"], by="x5"),
            f(["x0"], by="x6"),
            fs(["x0"], rf="x4"),
        ],
        data=sim_dat,
        discretize=True,
    )
    pen = build_penalties(sim_formula_m2)
    mmat2 = build_model_matrix(sim_formula_m2)

    def test_mmat(self):
        assert np.abs(self.mmat1 - self.mmat2.toarray()).max() < 1e-10


class Test_algorithms3:
    sim_dat = sim13(5000, 2, c=0, seed=0, family=Gaussian(), binom_offset=0, n_ranef=20)

    sim_formula_m1 = Formula(
        lhs("y"),
        [
            i(),
            l(["x5"]),
            l(["x6"]),
            f(["x0"], by="x5"),
            f(["x0"], by="x6"),
            rs(["x5", "x4"], by="x6"),
        ],
        data=sim_dat,
        discretize=True,
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
        [
            i(),
            l(["x5"]),
            l(["x6"]),
            f(["x0"], by="x5"),
            f(["x0"], by="x6"),
            rs(["x5", "x4"], by="x6"),
        ],
        data=sim_dat,
        discretize=True,
    )
    pen = build_penalties(sim_formula_m2)
    mmat2 = build_model_matrix(sim_formula_m2)

    def test_mmat(self):
        assert np.abs(self.mmat1 - self.mmat2.toarray()).max() < 1e-10


class Test_algorithms4:
    sim_dat = sim13(5000, 2, c=0, seed=0, family=Gamma(), binom_offset=0, n_ranef=20)

    sim_formula_m1 = Formula(
        lhs("y"),
        [i(), l(["x5"]), l(["x6"]), f(["x0"], by="x5"), f(["x0"], by="x6"), ri("x4")],
        data=sim_dat,
        discretize=True,
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
        [i(), l(["x5"]), l(["x6"]), f(["x0"], by="x5"), f(["x0"], by="x6"), ri("x4")],
        data=sim_dat,
        discretize=True,
    )
    pen = build_penalties(sim_formula_m2)
    mmat2 = build_model_matrix(sim_formula_m2)

    model1 = GAMM(sim_formula_m1, Gamma())
    model1.fit()
    model2 = GAMM(sim_formula_m2, Gamma())
    model2.fit()

    def test_mmat(self):
        assert np.abs(self.mmat1 - self.mmat2.toarray()).max() < 1e-10

    def test_hess(self):
        assert np.abs(self.model1.hessian - self.model2.hessian).max() < 0.01

    def test_hess2(self):
        assert np.abs(self.model1.hessian_obs - self.model2.hessian_obs).max() < 0.01


class Test_algorithm5:
    dat = pd.read_csv(
        "https://raw.githubusercontent.com/JoKra1/mssmViz/main/data/GAMM/sim_dat.csv"
    )

    # mssm requires that the data-type for variables used as factors is 'O'=object
    dat = dat.astype({"series": "O", "cond": "O", "sub": "O", "series": "O"})

    sim_formula_m1 = Formula(
        lhs=lhs("y"),  # The dependent variable - here y!
        terms=[
            i(),  # The intercept, a
            l(["cond"]),  # For cond='b'
            f(
                ["time"], by="cond", nk=20
            ),  # to-way interaction between time and cond; one smooth over time per cond level
            f(
                ["x"], by="cond"
            ),  # to-way interaction between x and cond; one smooth over x per cond level
            f(
                ["time", "x"], by="cond", nk=9, rp=0, scale_te=False
            ),  # three-way interaction
            fs(["time"], rf="sub", nk=20),
        ],  # Random non-linear effect of time - one smooth per level of factor series
        data=dat,
        series_id="series",
        discretize=True,
    )

    _, cov_flat, _, _, _, _, _, _ = sim_formula_m1.encode_data(
        sim_formula_m1.data, prediction=True, discretize=True
    )
    sim_formula_m1.cov_flat = cov_flat
    sim_formula_m1.discretize_cov = False
    _ = build_penalties(sim_formula_m1)
    mmat1 = build_model_matrix(sim_formula_m1)

    sim_formula_m2 = Formula(
        lhs=lhs("y"),  # The dependent variable - here y!
        terms=[
            i(),  # The intercept, a
            l(["cond"]),  # For cond='b'
            f(
                ["time"], by="cond", nk=20
            ),  # to-way interaction between time and cond; one smooth over time per cond level
            f(
                ["x"], by="cond"
            ),  # to-way interaction between x and cond; one smooth over x per cond level
            f(
                ["time", "x"], by="cond", nk=9, rp=0, scale_te=False
            ),  # three-way interaction
            fs(["time"], rf="sub", nk=20),
        ],  # Random non-linear effect of time - one smooth per level of factor series
        data=dat,
        series_id="series",
        discretize=True,
    )
    pen = build_penalties(sim_formula_m2)
    mmat2 = build_model_matrix(sim_formula_m2)

    def test_cross(self):
        assert (
            np.abs(
                (self.mmat1.T @ self.mmat1).toarray() - self.mmat2.T @ self.mmat2
            ).max()
            < 1e-7
        )

    def colTnoQIdx(self):
        assert (
            np.abs(
                self.mmat2[:, self.sim_formula_m2.coef_idx_per_term[4][10]]
                - self.mmat1[:, self.sim_formula_m2.coef_idx_per_term[4][10]]
            ).max()
            < 1e-9
        )


class Test_algorithm6:
    # by cov test
    sim_dat = sim4(500, 2, family=Gamma(), seed=0)

    # We again need to model the mean: \mu_i = \alpha + f(x0) + f(x1) + f_{x4}(x0)
    sim_formula_m1 = Formula(
        lhs("y"),
        [i(), f(["x0", "x1"], te=True, by_cont="x2")],
        data=sim_dat,
        discretize=True,
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
        [i(), f(["x0", "x1"], te=True, by_cont="x2")],
        data=sim_dat,
        discretize=True,
    )
    _ = build_penalties(sim_formula_m2)
    mmat2 = build_model_matrix(sim_formula_m2)

    # and the standard deviation
    sim_formula_sd = Formula(lhs("y"), [i()], data=sim_dat)

    family = GAMMALS([LOG(), LOGb(-0.0001)])

    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["control_lambda"] = 2
    test_kwargs["extend_lambda"] = False
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["method"] = "Chol"
    test_kwargs["repara"] = True
    test_kwargs["prefit_grad"] = True
    test_kwargs["force_sparse"] = False

    # Now define the model and fit!
    gsmm_fam = GAMLSSGSMMFamily(2, family)
    model1 = GSMM([sim_formula_m1, sim_formula_sd], gsmm_fam)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model1.fit(**test_kwargs)
    model2 = GSMM([sim_formula_m1, sim_formula_sd], gsmm_fam)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model2.fit(**test_kwargs)

    def test_mmat1(self):
        assert np.abs(self.mmat1 - self.mmat2.toarray()).max() < 1e-4

    def test_mmat1(self):
        assert (
            np.abs(self.mmat1.T @ self.mmat1 - self.mmat2.T @ self.mmat2).max() < 1e-4
        )

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model1.edf,
            self.model2.edf,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.0001),
        )

    def test_GAMcoef(self):
        coef1 = self.model1.coef
        coef2 = self.model2.coef
        np.testing.assert_allclose(
            coef1, coef2, atol=min(max_atol, 0), rtol=min(max_rtol, 0.0001)
        )

    def test_GAMreml(self):
        reml1 = self.model1.get_reml()
        reml2 = self.model2.get_reml()
        np.testing.assert_allclose(
            reml1, reml2, atol=min(max_atol, 0), rtol=min(max_rtol, 0.0001)
        )

    def test_edf1(self):
        compute_bias_corrected_edf(self.model1, overwrite=False)
        compute_bias_corrected_edf(self.model2, overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model1.term_edf1])
        edf2 = np.array([edf2 for edf2 in self.model2.term_edf1])
        np.testing.assert_allclose(
            edf1, edf2, atol=min(max_atol, 0), rtol=min(max_rtol, 0.0001)
        )

    def test_ps(self):
        ps1 = []
        ps2 = []
        for par in range(len(self.model1.formulas)):
            pps1, _ = approx_smooth_p_values(self.model1, par=par)
            pps2, _ = approx_smooth_p_values(self.model2, par=par)
            ps1.extend(pps1)
            ps2.extend(pps2)
        np.testing.assert_allclose(
            ps1, ps2, atol=min(max_atol, 0), rtol=min(max_rtol, 0.0001)
        )

    def test_TRs(self):
        Trs1 = []
        Trs2 = []
        for par in range(len(self.model1.formulas)):
            _, pTrs1 = approx_smooth_p_values(self.model1, par=par)
            _, pTrs2 = approx_smooth_p_values(self.model2, par=par)
            Trs1.extend(pTrs1)
            Trs2.extend(pTrs2)
        np.testing.assert_allclose(
            Trs1, Trs2, atol=min(max_atol, 0), rtol=min(max_rtol, 0.0001)
        )


class Test_algorithm7:
    sim_dat = sim13(500, 2, c=0, seed=0, family=Gaussian(), binom_offset=0, n_ranef=25)

    sim_formula_m1 = Formula(
        lhs("y"),
        [
            i(),
            l(["x6"]),
            f(["x0"]),
            f(["x0"], binary=["x5", "l5.2"]),
        ],
        data=sim_dat,
        discretize=True,
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
        [
            i(),
            l(["x6"]),
            f(["x0"]),
            f(["x0"], binary=["x5", "l5.2"]),
        ],
        data=sim_dat,
        discretize=True,
    )

    pen = build_penalties(sim_formula_m2)
    mmat2 = build_model_matrix(sim_formula_m2)

    def test_cross(self):
        assert (
            np.abs(
                (self.mmat1.T @ self.mmat1).toarray() - self.mmat2.T @ self.mmat2
            ).max()
            < 1e-7
        )


class Test_GSMM:
    sim_dat = sim16(500, seed=1134, correlate=True)

    # We need formulas for each mean
    formulas1 = [
        Formula(lhs("y0"), [i(), f(["x0"])], data=sim_dat, discretize=True),
        Formula(lhs("y1"), [i(), f(["x1"]), f(["x2"])], data=sim_dat, discretize=True),
        Formula(lhs("y2"), [i(), f(["x3"])], data=sim_dat, discretize=True),
    ]

    for form in formulas1:
        _, cov_flat, _, _, _, _, _, _ = form.encode_data(
            form.data, prediction=True, discretize=True
        )
        form.cov_flat = cov_flat
        form.discretize_cov = False

    formulas2 = [
        Formula(lhs("y0"), [i(), f(["x0"])], data=sim_dat, discretize=True),
        Formula(lhs("y1"), [i(), f(["x1"]), f(["x2"])], data=sim_dat, discretize=True),
        Formula(lhs("y2"), [i(), f(["x3"])], data=sim_dat, discretize=True),
    ]

    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["control_lambda"] = 2
    test_kwargs["extend_lambda"] = False
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500

    model1 = GSMM(formulas1, MultiGauss(3, [Identity() for _ in range(3)]))
    model1.fit(**test_kwargs)

    model2 = GSMM(formulas1, MultiGauss(3, [Identity() for _ in range(3)]))
    model2.fit(**test_kwargs)

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model1.edf,
            self.model2.edf,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.0001),
        )

    def test_GAMcoef(self):
        coef1 = self.model1.coef
        coef2 = self.model2.coef
        np.testing.assert_allclose(
            coef1, coef2, atol=min(max_atol, 0), rtol=min(max_rtol, 0.0001)
        )

    def test_GAMreml(self):
        reml1 = self.model1.get_reml()
        reml2 = self.model2.get_reml()
        np.testing.assert_allclose(
            reml1, reml2, atol=min(max_atol, 0), rtol=min(max_rtol, 0.0001)
        )

    def test_edf1(self):
        compute_bias_corrected_edf(self.model1, overwrite=False)
        compute_bias_corrected_edf(self.model2, overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model1.term_edf1])
        edf2 = np.array([edf2 for edf2 in self.model2.term_edf1])
        np.testing.assert_allclose(
            edf1, edf2, atol=min(max_atol, 0), rtol=min(max_rtol, 0.0001)
        )

    def test_ps(self):
        ps1 = []
        ps2 = []
        for par in range(len(self.model1.formulas)):
            pps1, _ = approx_smooth_p_values(self.model1, par=par)
            pps2, _ = approx_smooth_p_values(self.model2, par=par)
            ps1.extend(pps1)
            ps2.extend(pps2)
        np.testing.assert_allclose(
            ps1, ps2, atol=min(max_atol, 0), rtol=min(max_rtol, 0.0001)
        )

    def test_TRs(self):
        Trs1 = []
        Trs2 = []
        for par in range(len(self.model1.formulas)):
            _, pTrs1 = approx_smooth_p_values(self.model1, par=par)
            _, pTrs2 = approx_smooth_p_values(self.model2, par=par)
            Trs1.extend(pTrs1)
            Trs2.extend(pTrs2)
        np.testing.assert_allclose(
            Trs1, Trs2, atol=min(max_atol, 0), rtol=min(max_rtol, 0.0001)
        )


class Test_GSMMd:
    sim_dat = sim16(500, seed=1134, correlate=True)

    # We need formulas for each mean
    formulas1 = [
        Formula(lhs("y0"), [i(), f(["x0"])], data=sim_dat, discretize=True),
        Formula(lhs("y1"), [i(), f(["x1"]), f(["x2"])], data=sim_dat, discretize=True),
        Formula(lhs("y2"), [i(), f(["x3"])], data=sim_dat, discretize=True),
    ]

    for form in formulas1:
        _, cov_flat, _, _, _, _, _, _ = form.encode_data(
            form.data, prediction=True, discretize=True
        )
        form.cov_flat = cov_flat
        form.discretize_cov = False

    formulas2 = [
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

    model1 = GSMM(formulas1, MultiGauss(3, [Identity() for _ in range(3)]))
    model1.fit(**test_kwargs)

    model2 = GSMM(formulas1, MultiGauss(3, [Identity() for _ in range(3)]))
    model2.fit(**test_kwargs)

    def test_post(self):
        np.testing.assert_allclose(
            self.model1.sample_post(100, seed=0),
            self.model2.sample_post(100, seed=0),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.0001),
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
            16.217102086109538,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        np.testing.assert_allclose(
            sigma, 4.86901963656099, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
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
                    17.011127889091426,
                    19.04245480984634,
                    0.0834449063117302,
                    201.3330048705442,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 2.5),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -4562.501117748458, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -4537.152264070219, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_edf1(self):
        compute_bias_corrected_edf(self.model, overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model.term_edf1])
        np.testing.assert_allclose(
            edf1,
            np.array(
                [
                    3.8068091829338453,
                    3.717792665273125,
                    7.773343135810432,
                    2.2698030649118124,
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
            np.array([1.0176132998473975e-06, 0.0, 0.0, 0.29480269422543426]),
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
                    9.677726601950566,
                    84.31997605238361,
                    96.40503803785562,
                    1.2164726810908904,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.5),
        )
