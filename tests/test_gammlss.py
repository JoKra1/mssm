# flake8: noqa
import mssm
from mssm.models import *
import numpy as np
import os
from mssmViz.sim import *
import io
from contextlib import redirect_stdout
from .defaults import (
    default_gamm_test_kwargs,
    default_gammlss_test_kwargs,
    max_atol,
    max_rtol,
    init_coef_gaumlss_tests,
    init_coef_gammals_tests,
    init_penalties_tests_gammlss,
)

mssm.src.python.exp_fam.GAUMLSS.init_coef = init_coef_gaumlss_tests
mssm.src.python.exp_fam.GAMMALS.init_coef = init_coef_gammals_tests
mssm.src.python.exp_fam.GAUMLSS.init_lambda = init_penalties_tests_gammlss
mssm.src.python.exp_fam.GAMMALS.init_lambda = init_penalties_tests_gammlss
mssm.src.python.exp_fam.MULNOMLSS.init_lambda = init_penalties_tests_gammlss

################################################################## Tests ##################################################################


class Test_GAUMLS:
    # Simulate 500 data points
    GAUMLSSDat = sim6(500, seed=20)

    # We need to model the mean: \mu_i = \alpha + f(x0)
    formula_m = Formula(lhs("y"), [i(), f(["x0"], nk=10)], data=GAUMLSSDat)

    # and the standard deviation as well: log(\sigma_i) = \alpha + f(x0)
    formula_sd = Formula(lhs("y"), [i(), f(["x0"], nk=10)], data=GAUMLSSDat)

    # Collect both formulas
    formulas = [formula_m, formula_sd]

    # Create Gaussian GAMMLSS family with identity link for mean
    # and log link for sigma
    family = GAUMLSS([Identity(), LOG()])

    # Now define the model and fit!
    model = GAMMLSS(formulas, family)

    test_kwargs = copy.deepcopy(default_gammlss_test_kwargs)
    test_kwargs["extension_method_lam"] = "nesterov"
    test_kwargs["max_inner"] = 50
    test_kwargs["min_inner"] = 50

    model.fit(**test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf, ndigits=3) == 18.358

    def test_GAMcoef(self):
        coef = self.model.coef.flatten()
        assert np.allclose(
            coef,
            np.array(
                [
                    3.58432714,
                    -9.20689268,
                    -0.3901218,
                    4.33485776,
                    -2.83325345,
                    -4.68428463,
                    -1.93389004,
                    -4.14504997,
                    -6.70579839,
                    -4.7240103,
                    -5.19939664,
                    0.02026291,
                    -1.35759472,
                    1.5379936,
                    2.39353569,
                    2.33544107,
                    2.1749611,
                    2.3508268,
                    1.97975281,
                    1.4404345,
                    -0.96502575,
                    -4.28148962,
                ]
            ),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(lam, np.array([0.00909086, 1.07657659]))

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml, ndigits=3) == -772.602

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk, ndigits=3) == -719.601

    def test_print_smooth(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.model.print_smooth_terms()
        capture = capture.getvalue()

        comp = "\nDistribution parameter: 1\n\nf(['x0']); edf: 9.799\n\nDistribution parameter: 2\n\nf(['x0']); edf: 6.559\n"
        assert comp == capture

    def test_print_parametric(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.model.print_parametric_terms()
        capture = capture.getvalue()

        comp = "\nDistribution parameter: 1\n\nIntercept: 3.584, z: 58.435, P(|Z| > |z|): 0.000e+00 ***\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: .\n\nDistribution parameter: 2\n\nIntercept: 0.02, z: 0.64, P(|Z| > |z|): 0.52195\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: .\n"
        assert comp == capture

    def test_print_smooth_p_hard(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.model.print_smooth_terms(p_values=True)
        capture = capture.getvalue()

        comp1 = "\nDistribution parameter: 1\n\nf(['x0']); edf: 9.799 chi^2: 5630.989 P(Chi^2 > chi^2) = 0.000e+00 ***\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n\nDistribution parameter: 2\n\nf(['x0']); edf: 6.559 chi^2: 569.167 P(Chi^2 > chi^2) = 0.000e+00 ***\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n"
        comp2 = "\nDistribution parameter: 1\n\nf(['x0']); edf: 9.799 chi^2: 5696.894 P(Chi^2 > chi^2) = 0.000e+00 ***\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n\nDistribution parameter: 2\n\nf(['x0']); edf: 6.559 chi^2: 571.891 P(Chi^2 > chi^2) = 0.000e+00 ***\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n"
        assert (comp1 == capture) or (comp2 == capture)


class Test_GAUMLS_MIXED:
    ## Simulate some data - effect of x0 on scale parameter is very very small
    sim_fit_dat = sim10(n=500, c=0.1, seed=20)

    # Now fit nested models

    # Create Gaussian GAMMLSS family with identity link for mean
    # and log link for sigma
    family = GAUMLSS([Identity(), LOGb(-0.3)])

    # We need to model the mean: \mu_i = \alpha + f(x0)
    sim1_formula_m = Formula(
        lhs("y"), [i(), f(["x0"]), f(["x1"]), ri("x4")], data=sim_fit_dat
    )

    # and the standard deviation as well: log(\sigma_i) = \alpha + f(x0)
    sim1_formula_sd = Formula(lhs("y"), [i(), f(["x2"]), f(["x3"])], data=sim_fit_dat)

    # Collect both formulas
    sim1_formulas = [sim1_formula_m, sim1_formula_sd]

    model = GAMMLSS(sim1_formulas, family)

    test_kwargs = copy.deepcopy(default_gammlss_test_kwargs)
    test_kwargs["max_inner"] = 500
    test_kwargs["max_outer"] = 250
    test_kwargs["seed"] = 30
    test_kwargs["method"] = "QR/Chol"

    model.fit(**test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf, ndigits=3) == 26.196

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(
            lam,
            np.array(
                [
                    1.00000000e07,
                    6.04896788e00,
                    5.14594390e-01,
                    9.12400516e-01,
                    5.38283393e02,
                ]
            ),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml, ndigits=3) == -1747.292

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk, ndigits=3) == -1715.506


class Test_GAMMALS:

    # Simulate 500 data points
    GAMMALSSDat = sim6(500, family=GAMMALS([LOG(), LOG()]), seed=10)

    # We need to model the mean: \mu_i = \alpha + f(x0)
    formula_m = Formula(lhs("y"), [i(), f(["x0"], nk=10)], data=GAMMALSSDat)

    # and the scale parameter as well: log(\scale_i) = \alpha + f(x0)
    formula_sd = Formula(lhs("y"), [i(), f(["x0"], nk=10)], data=GAMMALSSDat)

    # Collect both formulas
    formulas = [formula_m, formula_sd]

    # Create Gamma GAMMLSS family with log link for mean
    # and log link for sigma
    family = GAMMALS([LOG(), LOG()])

    # Now define the model and fit!
    model = GAMMLSS(formulas, family)

    test_kwargs = copy.deepcopy(default_gammlss_test_kwargs)
    test_kwargs["extension_method_lam"] = "nesterov"
    test_kwargs["max_inner"] = 50
    test_kwargs["min_inner"] = 50

    model.fit(**test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf, ndigits=3) == 12.268

    def test_GAMcoef(self):
        coef = self.model.coef.flatten()
        assert np.allclose(
            coef,
            np.array(
                [
                    1.14793023,
                    -0.78753333,
                    1.97054867,
                    2.47570424,
                    1.77919304,
                    1.31179936,
                    1.90257167,
                    1.34991222,
                    0.26584116,
                    -0.67303397,
                    -1.72023523,
                    0.77361312,
                    -0.69700968,
                    0.55005875,
                    0.92263986,
                    0.98818801,
                    0.97583,
                    1.05108543,
                    1.03544474,
                    0.64241168,
                    -0.10886021,
                    -0.84455636,
                ]
            ),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(lam, np.array([0.52037363, 4.41778594]))

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml, ndigits=3) == -916.176

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk, ndigits=3) == -893.662


class Test_mulnom:
    # Test multinomial model

    # We need to specify K-1 formulas - see the `MULNOMLSS` docstring for details.
    formulas = [
        Formula(lhs("y"), [i(), f(["x0"])], data=sim5(1000, seed=91)) for k in range(4)
    ]

    # Create family - again specifying K-1 pars - here 4!
    family = MULNOMLSS(4)

    # Fit the model
    model = GAMMLSS(formulas, family)

    test_kwargs = copy.deepcopy(default_gammlss_test_kwargs)
    test_kwargs["control_lambda"] = False
    test_kwargs["extend_lambda"] = False
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["should_keep_drop"] = False

    model.fit(**test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf, ndigits=3) == 15.114

    def test_GAMcoef(self):
        coef = self.model.coef.flatten()
        assert np.allclose(
            coef,
            np.array(
                [
                    1.32831232,
                    -0.70411299,
                    0.26038397,
                    0.8585649,
                    1.31636532,
                    1.46064575,
                    1.15467552,
                    0.34090563,
                    -0.96911006,
                    -2.2518714,
                    0.66532703,
                    -0.76857181,
                    -0.20872371,
                    0.11240685,
                    0.42769554,
                    0.70741374,
                    1.07469267,
                    1.32593173,
                    1.37436662,
                    1.48506959,
                    2.06156634,
                    -1.63789937,
                    7.86561269,
                    9.15961217,
                    4.10367107,
                    1.21906228,
                    1.68896079,
                    1.01054086,
                    -0.31833491,
                    -0.59758569,
                    0.4881652,
                    -0.46708693,
                    -0.12681357,
                    0.06836815,
                    0.25998144,
                    0.42995532,
                    0.65313117,
                    0.80576918,
                    0.83513642,
                    0.90234416,
                ]
            ),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(
            lam, np.array([3.37059261e00, 7.24169569e04, 5.02852103e-02, 1.00000000e07])
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml, ndigits=3) == -1002.689

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk, ndigits=3) == -980.676


class Test_mulnom_repara:
    # Test multinomial model

    # We need to specify K-1 formulas - see the `MULNOMLSS` docstring for details.
    formulas = [
        Formula(lhs("y"), [i(), f(["x0"])], data=sim5(1000, seed=91)) for k in range(4)
    ]

    # Create family - again specifying K-1 pars - here 4!
    family = MULNOMLSS(4)

    # Fit the model
    model = GAMMLSS(formulas, family)

    test_kwargs = copy.deepcopy(default_gammlss_test_kwargs)
    test_kwargs["control_lambda"] = 0
    test_kwargs["extend_lambda"] = False
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["should_keep_drop"] = False
    test_kwargs["repara"] = True

    model.fit(**test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf, ndigits=3) == 15.114

    def test_GAMcoef(self):
        coef = self.model.coef.flatten()
        assert np.allclose(
            coef,
            np.array(
                [
                    1.32831232,
                    -0.70411299,
                    0.26038397,
                    0.85856489,
                    1.31636532,
                    1.46064575,
                    1.15467552,
                    0.34090563,
                    -0.96911006,
                    -2.2518714,
                    0.66532703,
                    -0.76857181,
                    -0.20872371,
                    0.11240685,
                    0.42769554,
                    0.70741373,
                    1.07469267,
                    1.32593173,
                    1.37436662,
                    1.48506958,
                    2.06156634,
                    -1.63789936,
                    7.86561269,
                    9.15961218,
                    4.10367108,
                    1.21906229,
                    1.6889608,
                    1.01054087,
                    -0.31833491,
                    -0.5975857,
                    0.4881652,
                    -0.46708693,
                    -0.12681357,
                    0.06836815,
                    0.25998144,
                    0.42995532,
                    0.65313117,
                    0.80576918,
                    0.83513642,
                    0.90234416,
                ]
            ),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(
            lam, np.array([3.37059261e00, 7.24170552e04, 5.02852109e-02, 1.00000000e07])
        )

    def test_GAMreml_hard(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -1002.688991, atol=min(max_atol, 2), rtol=min(max_rtol, 0.002)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk, ndigits=3) == -980.676


class Test_te_p_values:
    sim_dat = sim3(n=500, scale=2, c=0, seed=20)

    formula_m = Formula(
        lhs("y"), [i(), f(["x0", "x3"], te=True, nk=9), f(["x1"])], data=sim_dat
    )

    formula_sd = Formula(lhs("y"), [i(), f(["x0"]), f(["x2"])], data=sim_dat)

    # Collect both formulas
    formulas = [formula_m, formula_sd]

    # Create Gaussian GAMMLSS family with identity link for mean
    # and log link for sigma
    family = GAUMLSS([Identity(), LOGb(-0.001)])

    # Now define the model and fit!
    model = GAMMLSS(formulas, family)

    test_kwargs = copy.deepcopy(default_gammlss_test_kwargs)
    test_kwargs["control_lambda"] = 2
    test_kwargs["max_inner"] = 500
    test_kwargs["min_inner"] = 50

    model.fit(**test_kwargs)

    ps0, Trs0 = approx_smooth_p_values(model, par=0, edf1=False, force_approx=True)
    ps1, Trs1 = approx_smooth_p_values(model, par=1, edf1=False, force_approx=True)

    ps = [ps0, ps1]
    Trs = [Trs0, Trs1]

    def test_p1(self):
        np.testing.assert_allclose(
            self.ps[0],
            np.array([np.float64(0.39714921685433136), np.float64(0.0)]),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1e-6),
        )

    def test_p2_hard(self):
        np.testing.assert_allclose(
            self.ps[1],
            np.array([np.float64(0.738724988815144), np.float64(0.0)]),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1e-5),
        )

    def test_trs1(self):
        np.testing.assert_allclose(
            self.Trs[0],
            np.array([np.float64(5.67384667149778), np.float64(226.9137363518526)]),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1e-6),
        )

    def test_trs2(self):
        np.testing.assert_allclose(
            self.Trs[1],
            np.array([np.float64(0.11127579603700907), np.float64(135.27602836358722)]),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1e-6),
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

    model = GAMMLSS([formula, formula_sd], GAUMLSS([Identity(), LOGb(-0.001)]))

    test_kwargs = copy.deepcopy(default_gammlss_test_kwargs)
    test_kwargs["max_inner"] = 500
    test_kwargs["min_inner"] = 500
    test_kwargs["max_outer"] = 200
    test_kwargs["extend_lambda"] = False
    test_kwargs["control_lambda"] = 2
    test_kwargs["method"] = "LU/Chol"
    model.fit(**test_kwargs)

    def test_GAMedf_hard(self):
        np.testing.assert_allclose(
            self.model.edf,
            107.54811953082891,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.03),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        np.testing.assert_allclose(
            lam,
            np.array(
                [
                    4.913869938536273,
                    5.087829328785329,
                    0.0029242223796059627,
                    10000000.0,
                    1.2211364471705635,
                    1.105677472005988,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.5),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -10648.977191385402, atol=min(max_atol, 0), rtol=min(max_rtol, 0.01)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -10435.44335659411, atol=min(max_atol, 0), rtol=min(max_rtol, 0.01)
        )

    def test_drop(self):
        assert len(self.model.info.dropped) == 1


class Test_pred_whole_func_cor:
    # Simulate some data
    sim_fit_dat = sim6(n=500, seed=37)

    # Now fit nested models

    # Create Gaussian GAMMLSS family with identity link for mean
    # and log link for sigma
    family = GAUMLSS([Identity(), LOG()])

    # We need to model the mean: \mu_i = \alpha + f(x0)
    sim1_formula_m = Formula(lhs("y"), [i(), f(["x0"], nk=15)], data=sim_fit_dat)

    # and the standard deviation as well: log(\sigma_i) = \alpha + f(x0)
    sim1_formula_sd = Formula(lhs("y"), [i(), f(["x0"], nk=15)], data=sim_fit_dat)

    # Collect both formulas
    sim1_formulas = [sim1_formula_m, sim1_formula_sd]

    sim_fit_model = GAMMLSS(sim1_formulas, family)

    test_kwargs = copy.deepcopy(default_gammlss_test_kwargs)
    test_kwargs["extension_method_lam"] = "nesterov"

    sim_fit_model.fit(**test_kwargs)

    # Test prediction code + whole-function correction
    pred_dat = pd.DataFrame({"x0": np.linspace(0, 1, 50)})
    pred, pred_mat, ci = sim_fit_model.predict(
        [0, 1], pred_dat, ci=True, whole_interval=True, seed=20, par=0
    )

    def test_pred(self):
        assert np.allclose(
            self.pred,
            np.array(
                [
                    -0.01138392,
                    0.05310652,
                    0.44636549,
                    1.15072365,
                    2.14835029,
                    3.39365478,
                    4.75023087,
                    6.06313038,
                    7.17850177,
                    8.00856086,
                    8.56808424,
                    8.88068926,
                    8.96949259,
                    8.8500654,
                    8.53214278,
                    8.02530808,
                    7.34686356,
                    6.5607974,
                    5.74856022,
                    4.99160243,
                    4.35102523,
                    3.83236853,
                    3.43177651,
                    3.14527921,
                    2.9643237,
                    2.87431683,
                    2.86025298,
                    2.90661517,
                    2.99179483,
                    3.09018223,
                    3.17609768,
                    3.22449593,
                    3.21350857,
                    3.12225943,
                    2.92994466,
                    2.63296809,
                    2.26712914,
                    1.87371669,
                    1.49344223,
                    1.15055278,
                    0.85083955,
                    0.5991164,
                    0.39941563,
                    0.24828419,
                    0.13810493,
                    0.06121592,
                    0.01075972,
                    -0.01677633,
                    -0.0240358,
                    -0.01348688,
                ]
            ),
        )

    def test_ci(self):
        assert np.allclose(
            self.ci,
            np.array(
                [
                    0.27251711,
                    0.11885948,
                    0.14575675,
                    0.20026483,
                    0.26016534,
                    0.27019453,
                    0.31642028,
                    0.39883243,
                    0.43331877,
                    0.44002722,
                    0.50908004,
                    0.59067928,
                    0.5932289,
                    0.57954789,
                    0.64498692,
                    0.71516238,
                    0.69827735,
                    0.69170816,
                    0.76685076,
                    0.82708777,
                    0.82194123,
                    0.87061588,
                    0.97880911,
                    1.01529647,
                    0.96096182,
                    0.93911891,
                    0.96154628,
                    0.91487953,
                    0.8149885,
                    0.7760991,
                    0.77936308,
                    0.71521874,
                    0.6455595,
                    0.65966908,
                    0.67793687,
                    0.61339351,
                    0.55758266,
                    0.57039461,
                    0.56076992,
                    0.49618751,
                    0.47606143,
                    0.50317253,
                    0.46317742,
                    0.36127176,
                    0.30228488,
                    0.28955845,
                    0.22256101,
                    0.15345764,
                    0.10679976,
                    0.2008102,
                ]
            ),
        )


class Test_diff_whole_func_cor:
    # Difference prediction for gammlss

    # Simulate some data - effect of x0 on scale parameter is very very small in one condition
    sim_fit_dat1 = sim8(n=500, c=0.1, seed=37)
    sim_fit_dat1["cond"] = "a"
    sim_fit_dat2 = sim8(n=500, c=1, seed=41)
    sim_fit_dat2["cond"] = "b"
    sim_fit_dat = pd.concat((sim_fit_dat1, sim_fit_dat2), axis=0, ignore_index=True)

    # Now fit model

    # Create Gaussian GAMMLSS family with identity link for mean
    # and log link for sigma
    family = GAUMLSS([Identity(), LOG()])

    # We need to model the mean: \mu_i = \alpha + f(x0)
    sim1_formula_m = Formula(lhs("y"), [i(), f(["x0"], nk=10)], data=sim_fit_dat)

    # and the standard deviation as well: log(\sigma_i) = \alpha + f(x0)
    sim1_formula_sd = Formula(
        lhs("y"), [i(), f(["x0"], nk=10, by="cond")], data=sim_fit_dat
    )

    # Collect both formulas
    sim1_formulas = [sim1_formula_m, sim1_formula_sd]

    sim_fit_model = GAMMLSS(sim1_formulas, family)

    test_kwargs = copy.deepcopy(default_gammlss_test_kwargs)
    test_kwargs["extension_method_lam"] = "nesterov2"

    sim_fit_model.fit(**test_kwargs)

    pred_dat1 = pd.DataFrame(
        {"x0": np.linspace(0, 1, 50), "cond": ["a" for _ in range(50)]}
    )

    pred_dat2 = pd.DataFrame(
        {"x0": np.linspace(0, 1, 50), "cond": ["b" for _ in range(50)]}
    )

    diff, ci = sim_fit_model.predict_diff(
        pred_dat2, pred_dat1, [1], whole_interval=True, seed=20, par=1
    )

    def test_print_smooth_p_hard(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.sim_fit_model.print_smooth_terms(p_values=True)
        capture = capture.getvalue()

        comp1 = "\nDistribution parameter: 1\n\nf(['x0']); edf: 7.079 chi^2: 457.542 P(Chi^2 > chi^2) = 0.000e+00 ***\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n\nDistribution parameter: 2\n\nf(['x0'],by=cond): a; edf: 1.0 chi^2: 0.056 P(Chi^2 > chi^2) = 0.81334\nf(['x0'],by=cond): b; edf: 2.789 chi^2: 13.49 P(Chi^2 > chi^2) = 0.00585 **\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n"
        comp2 = "\nDistribution parameter: 1\n\nf(['x0']); edf: 7.079 chi^2: 457.185 P(Chi^2 > chi^2) = 0.000e+00 ***\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n\nDistribution parameter: 2\n\nf(['x0'],by=cond): a; edf: 1.0 chi^2: 0.056 P(Chi^2 > chi^2) = 0.81334\nf(['x0'],by=cond): b; edf: 2.789 chi^2: 13.49 P(Chi^2 > chi^2) = 0.00585 **\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n"
        assert (comp1 == capture) or (comp2 == capture)

    def test_diff(self):
        assert np.allclose(
            self.diff,
            np.array(
                [
                    -0.30135358,
                    -0.26600859,
                    -0.23106805,
                    -0.19676569,
                    -0.16333525,
                    -0.13101046,
                    -0.10002506,
                    -0.07060065,
                    -0.0428877,
                    -0.01701087,
                    0.00690523,
                    0.02873598,
                    0.04835675,
                    0.06565901,
                    0.08067376,
                    0.09350389,
                    0.10425282,
                    0.11302403,
                    0.11992095,
                    0.12504712,
                    0.12850733,
                    0.13040725,
                    0.13085254,
                    0.12994891,
                    0.12780203,
                    0.12451593,
                    0.12015716,
                    0.11475508,
                    0.10833746,
                    0.10093206,
                    0.09256666,
                    0.0832692,
                    0.0730749,
                    0.06202901,
                    0.05017749,
                    0.03756631,
                    0.02424142,
                    0.01024862,
                    -0.00438762,
                    -0.01968347,
                    -0.03565969,
                    -0.05233703,
                    -0.06973624,
                    -0.08787804,
                    -0.10675354,
                    -0.12627384,
                    -0.14633665,
                    -0.16683967,
                    -0.1876806,
                    -0.20875715,
                ]
            ),
        )

    def test_ci(self):
        assert np.allclose(
            self.ci,
            np.array(
                [
                    0.34629246,
                    0.30914794,
                    0.27596426,
                    0.24705564,
                    0.22265442,
                    0.20269553,
                    0.18668468,
                    0.1738519,
                    0.16380075,
                    0.15639354,
                    0.15136071,
                    0.14811353,
                    0.14577211,
                    0.14347928,
                    0.14117464,
                    0.13925221,
                    0.13794476,
                    0.1371093,
                    0.13623457,
                    0.13471181,
                    0.13267291,
                    0.13079453,
                    0.12955546,
                    0.12898548,
                    0.12865308,
                    0.12792413,
                    0.12682113,
                    0.12604402,
                    0.1261187,
                    0.12710645,
                    0.12858775,
                    0.12991388,
                    0.13096505,
                    0.13233796,
                    0.13449826,
                    0.13749826,
                    0.14098219,
                    0.14442194,
                    0.14773426,
                    0.1515958,
                    0.15668965,
                    0.16343419,
                    0.17198806,
                    0.1824472,
                    0.19525605,
                    0.21136223,
                    0.23150348,
                    0.25598021,
                    0.28474244,
                    0.31760382,
                ]
            ),
        )


class Test_shared_pen:
    sim_dat = sim12(5000, c=0, seed=0, family=GAMMALS([LOG(), LOG()]), n_ranef=20)

    # We again need to model the mean: \mu_i = \alpha + f(x0) + f(x1) + f_{x4}(x0)
    sim_formula_m = Formula(
        lhs("y"), [i(), f(["x0"], id=1), f(["x1"]), fs(["x0"], rf="x4")], data=sim_dat
    )

    # and the standard deviation as well: log(\sigma_i) = \alpha + f(x2) + f(x3)
    sim_formula_sd = Formula(lhs("y"), [i(), f(["x2"], id=1), f(["x3"])], data=sim_dat)

    family = GAMMALS([LOG(), LOGb(-0.0001)])

    test_kwargs = copy.deepcopy(default_gammlss_test_kwargs)
    test_kwargs["control_lambda"] = 2
    test_kwargs["extend_lambda"] = False
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["method"] = "QR/Chol"
    test_kwargs["repara"] = True
    test_kwargs["prefit_grad"] = True

    # Now define the model and fit!
    model = GAMMLSS([sim_formula_m, sim_formula_sd], family)
    model.fit(**test_kwargs)

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            102.77768526577066,
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
                    [-3.07512708e-01],
                    [8.04396785e-03],
                    [4.19311781e-01],
                    [6.29220322e-01],
                    [5.02930132e-01],
                    [4.70449980e-01],
                    [-7.38918727e-02],
                    [-4.10762223e-01],
                    [-6.71402385e-01],
                    [-9.09691712e-01],
                    [-4.20806637e-01],
                    [-1.37328435e-01],
                    [1.38818620e-01],
                    [5.02864166e-01],
                    [1.03348977e00],
                    [1.67936417e00],
                    [2.30239745e00],
                    [2.96343188e00],
                    [1.20608285e-02],
                    [3.34539157e-03],
                    [-4.75972397e-02],
                    [1.39688486e-01],
                    [4.22995999e-02],
                    [2.88947907e-02],
                    [1.52413668e-01],
                    [5.38642733e-02],
                    [-5.02106563e-02],
                    [-4.00748201e-01],
                    [2.08111451e-03],
                    [-3.33144399e-03],
                    [-8.99365078e-02],
                    [-1.07323902e-01],
                    [1.00861041e-01],
                    [-1.41085870e-01],
                    [2.90312091e-01],
                    [-1.10658223e-01],
                    [3.93929316e-03],
                    [1.51003575e-01],
                    [5.89481573e-03],
                    [-6.87221948e-03],
                    [4.95267541e-03],
                    [-2.09143118e-04],
                    [3.73769141e-02],
                    [5.55292008e-02],
                    [-3.44913809e-02],
                    [-9.76471955e-03],
                    [-4.60496010e-02],
                    [1.05127968e-01],
                    [9.91475330e-03],
                    [3.72289203e-03],
                    [-5.01951951e-02],
                    [-1.22346813e-01],
                    [5.17546650e-02],
                    [9.77506260e-02],
                    [1.79073001e-02],
                    [7.60515035e-02],
                    [4.34602777e-02],
                    [4.72167633e-01],
                    [-6.66564717e-03],
                    [1.84169318e-04],
                    [2.74872961e-02],
                    [-4.04690537e-02],
                    [5.84483312e-02],
                    [-1.52034970e-01],
                    [-1.28140211e-04],
                    [-5.01068673e-02],
                    [-1.69397369e-01],
                    [-1.52963785e00],
                    [1.17796941e-02],
                    [-1.24983209e-02],
                    [9.22871812e-03],
                    [6.37495952e-02],
                    [5.78668361e-02],
                    [6.53371831e-02],
                    [-3.13292906e-03],
                    [1.89639431e-01],
                    [2.41709584e-01],
                    [3.54416656e-01],
                    [-1.33694060e-02],
                    [-7.42420178e-03],
                    [-2.38859292e-02],
                    [-1.77319531e-02],
                    [-7.52282491e-03],
                    [5.32578759e-02],
                    [-2.80027710e-01],
                    [2.82942321e-02],
                    [-1.18734851e-01],
                    [-7.17711640e-01],
                    [3.89207025e-03],
                    [5.14388861e-05],
                    [1.26187904e-01],
                    [5.93148111e-03],
                    [4.26402728e-02],
                    [-2.85999861e-01],
                    [-1.00947149e-01],
                    [-2.08466048e-01],
                    [-6.37369456e-02],
                    [1.36285515e-01],
                    [-1.35994852e-02],
                    [-4.27430531e-04],
                    [-4.15079404e-02],
                    [-8.02770093e-02],
                    [-2.46208343e-02],
                    [1.42594514e-01],
                    [5.24535985e-02],
                    [6.89740140e-02],
                    [-1.48757268e-01],
                    [2.82631716e-02],
                    [-6.00536369e-03],
                    [-1.74409632e-03],
                    [1.32468234e-03],
                    [3.54341493e-02],
                    [3.92089635e-02],
                    [1.82962490e-01],
                    [-9.24064278e-02],
                    [1.00428557e-01],
                    [1.04098144e-02],
                    [1.79686116e-01],
                    [-4.04790903e-03],
                    [2.36924546e-03],
                    [3.08741645e-02],
                    [9.97208597e-03],
                    [1.37506958e-01],
                    [1.02895320e-01],
                    [1.91448728e-01],
                    [5.87751897e-02],
                    [-6.18351266e-02],
                    [2.55713158e-01],
                    [-1.15324771e-02],
                    [2.11327455e-02],
                    [-5.15599029e-02],
                    [5.82907079e-02],
                    [-9.07263982e-02],
                    [-1.38711542e-01],
                    [7.05918233e-02],
                    [-1.36520508e-01],
                    [-1.38360052e-01],
                    [3.86816734e-01],
                    [-1.08426984e-03],
                    [2.78971190e-03],
                    [-3.48845980e-02],
                    [-1.14498217e-01],
                    [-8.33459165e-02],
                    [-8.16313748e-02],
                    [-1.60173351e-01],
                    [-6.62090978e-03],
                    [4.47679594e-03],
                    [1.85456063e-01],
                    [4.86270776e-03],
                    [-4.14665338e-03],
                    [4.08268374e-02],
                    [1.03036878e-01],
                    [-3.33139837e-02],
                    [-1.81934272e-01],
                    [-4.73324926e-02],
                    [9.10428037e-02],
                    [1.57500755e-02],
                    [5.23048134e-01],
                    [1.15638117e-02],
                    [-1.08343822e-02],
                    [9.26606280e-02],
                    [-1.74312004e-02],
                    [-6.83285709e-02],
                    [1.70064882e-01],
                    [-7.98845953e-02],
                    [2.28656180e-02],
                    [6.83521752e-02],
                    [1.30809531e-01],
                    [-4.84208418e-03],
                    [1.33424885e-02],
                    [-3.27998226e-02],
                    [-4.56987461e-02],
                    [-1.17981096e-01],
                    [1.41266472e-01],
                    [1.14970137e-01],
                    [-2.01241407e-01],
                    [2.51213577e-01],
                    [3.10169775e-01],
                    [-4.23847066e-03],
                    [-6.52227074e-03],
                    [-2.48306145e-02],
                    [-3.68011947e-02],
                    [-9.67471664e-02],
                    [-1.50964161e-01],
                    [-2.69075927e-02],
                    [-2.41578729e-01],
                    [1.45601727e-01],
                    [-2.94313864e-01],
                    [9.92868380e-03],
                    [-3.90957029e-03],
                    [-9.04478839e-02],
                    [4.53467584e-02],
                    [-1.52905672e-01],
                    [5.05802574e-02],
                    [1.97914743e-01],
                    [3.57754328e-01],
                    [7.16199602e-02],
                    [-5.19204100e-01],
                    [-1.39238423e-02],
                    [1.09518643e-03],
                    [3.34452609e-02],
                    [-4.45868068e-02],
                    [5.61065650e-02],
                    [-3.52593576e-03],
                    [-2.54898426e-01],
                    [-2.15069589e-02],
                    [-5.82873551e-02],
                    [2.42651628e-01],
                    [3.29342185e-01],
                    [-4.02792168e-01],
                    [7.92919443e-01],
                    [1.10560544e00],
                    [2.52505190e-01],
                    [5.39066060e-01],
                    [4.45502427e-01],
                    [2.54453008e-01],
                    [-2.76220531e-01],
                    [-7.20654835e-01],
                    [4.65359409e-02],
                    [1.14327237e-02],
                    [-1.00040927e-02],
                    [-2.87077747e-02],
                    [-4.34514441e-02],
                    [-5.70480571e-02],
                    [-6.78680325e-02],
                    [-6.54355442e-02],
                    [-6.45672414e-02],
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
                    31.222081138940332,
                    4.071165297330471,
                    3.9187652112113875,
                    11307.85547736765,
                    2.358734759947485,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 2.5),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -15526.867512177605, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -15331.745250625594, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_edf1(self):
        compute_bias_corrected_edf(self.model, overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model.term_edf1])
        np.testing.assert_allclose(
            edf1,
            np.array(
                [
                    6.713831972418086,
                    4.921128087309552,
                    102.24427710377597,
                    6.914787187470902,
                    1.3584941787931564,
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
            np.array([0.0, 0.0, 0.0, 0.07909498523766123]),
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
                    38.39621398859056,
                    2835.761588753815,
                    243.2047117086251,
                    3.7791188495195702,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.5),
        )

    def test_lcp(self):
        y = self.model.formulas[0].y_flat[self.model.formulas[0].NOT_NA_flat]
        alpha = 1 / self.model.mus[1]
        beta = alpha / self.model.mus[0]

        assert np.allclose(
            self.model.family.lcp(y, *self.model.mus),
            scp.stats.gamma.logcdf(y, a=alpha, scale=(1 / beta)),
        )


class Test_shared_pen2:
    sim_dat = sim12(5000, c=0, seed=0, family=GAUMLSS([Identity(), LOG()]), n_ranef=20)

    # We again need to model the mean: \mu_i = \alpha + f(x0) + f(x1) + f_{x4}(x0)
    sim_formula_m = Formula(
        lhs("y"), [i(), f(["x0"], id=1), f(["x1"]), fs(["x0"], rf="x4")], data=sim_dat
    )

    # and the standard deviation as well: log(\sigma_i) = \alpha + f(x2) + f(x3)
    sim_formula_sd = Formula(lhs("y"), [i(), f(["x2"], id=1), f(["x3"])], data=sim_dat)

    family = GAUMLSS([Identity(), LOGb(-0.0001)])

    test_kwargs = copy.deepcopy(default_gammlss_test_kwargs)
    test_kwargs["control_lambda"] = 2
    test_kwargs["extend_lambda"] = False
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["method"] = "QR/Chol"
    test_kwargs["repara"] = True
    test_kwargs["prefit_grad"] = True

    # Now define the model and fit!
    model = GAMMLSS([sim_formula_m, sim_formula_sd], family)
    model.fit(**test_kwargs)

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            130.72218261728594,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_GAMcoef(self):
        coef = self.model.coef
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    [4.33379340e00],
                    [-6.71839913e-01],
                    [3.61078026e-01],
                    [9.93472356e-01],
                    [1.26409781e00],
                    [1.19713900e00],
                    [9.37781252e-01],
                    [1.76237816e-01],
                    [-8.84380381e-01],
                    [-1.71278833e00],
                    [-1.70022093e00],
                    [-9.74279116e-01],
                    [-4.29111342e-01],
                    [4.15131256e-03],
                    [8.93702162e-01],
                    [1.89637360e00],
                    [3.02287818e00],
                    [4.71619347e00],
                    [6.46773665e00],
                    [4.82664880e-02],
                    [2.80365264e-02],
                    [1.13199034e-01],
                    [3.63474866e-02],
                    [2.69287685e-01],
                    [1.27666050e-02],
                    [2.63043599e-01],
                    [4.37104310e-02],
                    [-6.06453663e-02],
                    [-5.06842672e-01],
                    [2.87928853e-02],
                    [7.13705211e-03],
                    [1.58062860e-01],
                    [-2.63743894e-02],
                    [2.10274992e-01],
                    [-5.87136363e-01],
                    [4.18597079e-01],
                    [-1.56932273e-01],
                    [1.10279754e-01],
                    [1.10108546e-02],
                    [-2.25844403e-02],
                    [3.73788919e-02],
                    [-6.48407067e-02],
                    [2.63885755e-01],
                    [2.72761508e-01],
                    [-1.33989680e-01],
                    [1.99110186e-01],
                    [1.15461765e-01],
                    [-4.40949111e-02],
                    [5.56037557e-02],
                    [4.99822844e-02],
                    [2.38964963e-02],
                    [-1.43294373e-01],
                    [-6.47631919e-01],
                    [3.35760532e-01],
                    [-2.28919703e-02],
                    [-3.22050600e-02],
                    [2.56142184e-01],
                    [1.08720633e-01],
                    [1.08348498e00],
                    [-3.62425534e-03],
                    [-2.60578847e-02],
                    [2.88005425e-02],
                    [-1.16312198e-01],
                    [3.04693799e-01],
                    [-4.11608925e-01],
                    [-1.03913082e-01],
                    [-4.75006917e-02],
                    [-2.93832031e-01],
                    [-3.03072003e00],
                    [2.12870344e-02],
                    [1.24856380e-01],
                    [-1.22306232e-01],
                    [8.65752311e-02],
                    [4.80684929e-01],
                    [1.68977197e-01],
                    [3.49211391e-01],
                    [4.48327327e-01],
                    [4.49331664e-01],
                    [1.00082005e00],
                    [-7.29807542e-02],
                    [-1.90720979e-02],
                    [2.35278455e-02],
                    [1.03325690e-03],
                    [-5.10864552e-01],
                    [7.20335032e-02],
                    [-4.25620328e-01],
                    [1.46638722e-01],
                    [-1.91291435e-01],
                    [-1.06707937e00],
                    [-8.85967019e-02],
                    [3.27983961e-02],
                    [2.97461405e-01],
                    [-3.73233603e-02],
                    [-6.74580450e-03],
                    [-4.24556367e-01],
                    [-4.34365798e-01],
                    [-2.65635841e-01],
                    [-1.64717547e-01],
                    [2.67419684e-01],
                    [3.61678726e-02],
                    [-9.05427887e-03],
                    [-1.13802516e-01],
                    [2.94720509e-02],
                    [-1.79013825e-01],
                    [-1.89182411e-01],
                    [-3.13898986e-01],
                    [2.97679628e-01],
                    [-2.63460146e-01],
                    [-2.08550179e-01],
                    [-4.09662390e-02],
                    [4.51824713e-02],
                    [-2.89475430e-01],
                    [3.18689009e-01],
                    [5.82314957e-01],
                    [4.97669363e-01],
                    [-2.07755782e-01],
                    [-9.52999640e-02],
                    [-1.53883096e-02],
                    [2.09725233e-01],
                    [-1.66027926e-02],
                    [1.04132751e-02],
                    [7.47879103e-02],
                    [1.00654072e-01],
                    [-2.04240741e-01],
                    [7.71958249e-02],
                    [3.03577352e-01],
                    [2.00148101e-01],
                    [-1.68637151e-01],
                    [5.79575674e-01],
                    [2.69438344e-02],
                    [-6.78613267e-02],
                    [-1.46717029e-01],
                    [6.65003853e-02],
                    [-1.68857613e-01],
                    [-3.57154335e-01],
                    [1.65342325e-01],
                    [-4.58979263e-01],
                    [-3.51563587e-01],
                    [8.25435480e-01],
                    [3.71391337e-03],
                    [8.76052092e-02],
                    [2.24503489e-01],
                    [-2.89860075e-01],
                    [-2.52468283e-01],
                    [-2.37384206e-02],
                    [8.92435151e-02],
                    [-1.60532006e-01],
                    [-1.33376101e-01],
                    [4.83476647e-01],
                    [7.88642360e-02],
                    [-5.72480636e-03],
                    [-1.95310932e-01],
                    [1.19979791e-01],
                    [-1.64117817e-01],
                    [-5.61160315e-01],
                    [-1.68416166e-01],
                    [3.93189955e-01],
                    [1.15345252e-01],
                    [1.13474509e00],
                    [-1.21582925e-02],
                    [2.67041202e-02],
                    [2.63633582e-01],
                    [-3.47354987e-01],
                    [-5.13418729e-02],
                    [5.43130272e-01],
                    [-4.01132985e-01],
                    [-4.00475723e-02],
                    [-5.79446308e-04],
                    [1.78245611e-01],
                    [-3.50884638e-02],
                    [-1.72931200e-02],
                    [1.21890150e-01],
                    [-2.88756977e-01],
                    [-2.41485080e-01],
                    [9.18703868e-02],
                    [2.37622010e-01],
                    [-4.10201040e-01],
                    [5.10916021e-01],
                    [2.70026845e-01],
                    [-1.29984325e-02],
                    [4.09057110e-03],
                    [-9.36802855e-02],
                    [2.27931190e-01],
                    [-8.46690422e-02],
                    [1.21929400e-01],
                    [-2.43871745e-02],
                    [-4.67753330e-01],
                    [3.05165313e-01],
                    [-6.17834375e-01],
                    [1.97381096e-02],
                    [1.36668710e-02],
                    [-4.21264700e-01],
                    [7.55824948e-02],
                    [-3.71789175e-01],
                    [8.96640084e-01],
                    [5.17823886e-01],
                    [7.67995812e-01],
                    [2.26559258e-01],
                    [-1.23745417e00],
                    [3.80387949e-02],
                    [1.81907134e-02],
                    [3.76931246e-03],
                    [1.82675846e-01],
                    [-2.58719748e-01],
                    [-3.24454224e-01],
                    [-3.56528889e-01],
                    [-1.75358722e-01],
                    [-1.19504721e-01],
                    [5.68910879e-01],
                    [3.02190302e-01],
                    [-4.42297428e-01],
                    [7.08692840e-01],
                    [8.89823904e-01],
                    [3.09113525e-01],
                    [3.47597635e-01],
                    [3.28490386e-01],
                    [-5.91877007e-02],
                    [-2.51161409e-01],
                    [-2.62088040e-01],
                    [-2.28250491e-02],
                    [-7.88458749e-03],
                    [1.49379935e-03],
                    [1.08301596e-02],
                    [1.93731958e-02],
                    [2.79897122e-02],
                    [3.54714263e-02],
                    [3.66079396e-02],
                    [3.84156173e-02],
                ]
            ),
            atol=min(max_atol, 1),
            rtol=min(max_rtol, 0.5),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        np.testing.assert_allclose(
            lam,
            np.array(
                [
                    3.9614527961398274,
                    0.8717540995086903,
                    1.0050277835190533,
                    10000000.0,
                    3.0248416458589253,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 2.5),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -8886.226185427402, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -8606.027341258703, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_edf1(self):
        compute_bias_corrected_edf(self.model, overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model.term_edf1])
        np.testing.assert_allclose(
            edf1,
            np.array(
                [
                    5.621712839293441,
                    6.497582544082489,
                    128.47016448878225,
                    7.430870523057176,
                    1.0015037147259365,
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
            np.array([3.5717079275299213e-06, 0.0, 0.0, 0.09119187436952969]),
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
                    38.36237880945953,
                    9463.441910146503,
                    712.2431495413452,
                    2.856138998529643,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.5),
        )

    def test_lcp(self):
        y = self.model.formulas[0].y_flat[self.model.formulas[0].NOT_NA_flat]

        assert np.allclose(
            self.model.family.lcp(y, *self.model.mus),
            scp.stats.norm.logcdf(y, loc=self.model.mus[0], scale=self.model.mus[1]),
        )
