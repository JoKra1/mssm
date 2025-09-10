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

        comp1 = "\nDistribution parameter: 1\n\nf(['x0']); edf: 9.799 chi^2: 5696.894 P(Chi^2 > chi^2) = 0.000e+00 ***\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n\nDistribution parameter: 2\n\nf(['x0']); edf: 6.559 chi^2: 569.167 P(Chi^2 > chi^2) = 0.000e+00 ***\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n"
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
            108.54811953082891,
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
                    [2.16073927],
                    [-0.30751271],
                    [0.00804397],
                    [0.41931178],
                    [0.62922032],
                    [0.50293013],
                    [0.47044998],
                    [-0.07389187],
                    [-0.41076222],
                    [-0.67140238],
                    [-0.90969171],
                    [-0.42080664],
                    [-0.13732844],
                    [0.13881862],
                    [0.50286417],
                    [1.03348977],
                    [1.67936417],
                    [2.30239745],
                    [2.96343188],
                    [0.01206083],
                    [0.00334539],
                    [-0.04759724],
                    [0.13968849],
                    [0.0422996],
                    [0.02889479],
                    [0.15241367],
                    [0.05386427],
                    [-0.05021066],
                    [-0.4007482],
                    [0.00208111],
                    [-0.00333144],
                    [-0.08993651],
                    [-0.1073239],
                    [0.10086104],
                    [-0.14108587],
                    [0.29031209],
                    [-0.11065822],
                    [0.00393929],
                    [0.15100357],
                    [0.00589482],
                    [-0.00687222],
                    [0.00495268],
                    [-0.00020914],
                    [0.03737691],
                    [0.0555292],
                    [-0.03449138],
                    [-0.00976472],
                    [-0.0460496],
                    [0.10512797],
                    [0.00991475],
                    [0.00372289],
                    [-0.0501952],
                    [-0.12234681],
                    [0.05175467],
                    [0.09775063],
                    [0.0179073],
                    [0.0760515],
                    [0.04346028],
                    [0.47216763],
                    [-0.00666565],
                    [0.00018417],
                    [0.0274873],
                    [-0.04046905],
                    [0.05844833],
                    [-0.15203497],
                    [-0.00012814],
                    [-0.05010687],
                    [-0.16939737],
                    [-1.52963785],
                    [0.01177969],
                    [-0.01249832],
                    [0.00922872],
                    [0.0637496],
                    [0.05786684],
                    [0.06533718],
                    [-0.00313293],
                    [0.18963943],
                    [0.24170958],
                    [0.35441666],
                    [-0.01336941],
                    [-0.0074242],
                    [-0.02388593],
                    [-0.01773195],
                    [-0.00752282],
                    [0.05325788],
                    [-0.28002771],
                    [0.02829423],
                    [-0.11873485],
                    [-0.71771164],
                    [0.00389207],
                    [5.14388861e-05],
                    [0.1261879],
                    [0.00593148],
                    [0.04264027],
                    [-0.28599986],
                    [-0.10094715],
                    [-0.20846605],
                    [-0.06373695],
                    [0.13628551],
                    [-0.01359949],
                    [-0.00042743],
                    [-0.04150794],
                    [-0.08027701],
                    [-0.02462083],
                    [0.14259451],
                    [0.0524536],
                    [0.06897401],
                    [-0.14875727],
                    [0.02826317],
                    [-0.00600536],
                    [-0.0017441],
                    [0.00132468],
                    [0.03543415],
                    [0.03920896],
                    [0.18296249],
                    [-0.09240643],
                    [0.10042856],
                    [0.01040981],
                    [0.17968612],
                    [-0.00404791],
                    [0.00236925],
                    [0.03087416],
                    [0.00997209],
                    [0.13750696],
                    [0.10289532],
                    [0.19144873],
                    [0.05877519],
                    [-0.06183513],
                    [0.25571316],
                    [-0.01153248],
                    [0.02113275],
                    [-0.0515599],
                    [0.05829071],
                    [-0.0907264],
                    [-0.13871154],
                    [0.07059182],
                    [-0.13652051],
                    [-0.13836005],
                    [0.38681673],
                    [-0.00108427],
                    [0.00278971],
                    [-0.0348846],
                    [-0.11449822],
                    [-0.08334592],
                    [-0.08163137],
                    [-0.16017335],
                    [-0.00662091],
                    [0.0044768],
                    [0.18545606],
                    [0.00486271],
                    [-0.00414665],
                    [0.04082684],
                    [0.10303688],
                    [-0.03331398],
                    [-0.18193427],
                    [-0.04733249],
                    [0.0910428],
                    [0.01575008],
                    [0.52304813],
                    [0.01156381],
                    [-0.01083438],
                    [0.09266063],
                    [-0.0174312],
                    [-0.06832857],
                    [0.17006488],
                    [-0.0798846],
                    [0.02286562],
                    [0.06835218],
                    [0.13080953],
                    [-0.00484208],
                    [0.01334249],
                    [-0.03279982],
                    [-0.04569875],
                    [-0.1179811],
                    [0.14126647],
                    [0.11497014],
                    [-0.20124141],
                    [0.25121358],
                    [0.31016978],
                    [-0.00423847],
                    [-0.00652227],
                    [-0.02483061],
                    [-0.03680119],
                    [-0.09674717],
                    [-0.15096416],
                    [-0.02690759],
                    [-0.24157873],
                    [0.14560173],
                    [-0.29431386],
                    [0.00992868],
                    [-0.00390957],
                    [-0.09044788],
                    [0.04534676],
                    [-0.15290567],
                    [0.05058026],
                    [0.19791474],
                    [0.35775433],
                    [0.07161996],
                    [-0.5192041],
                    [-0.01392384],
                    [0.00109519],
                    [0.03344526],
                    [-0.04458681],
                    [0.05610657],
                    [-0.00352594],
                    [-0.25489843],
                    [-0.02150696],
                    [-0.05828736],
                    [0.24265163],
                    [0.32934218],
                    [-0.40279217],
                    [0.79291944],
                    [1.10560544],
                    [0.25250519],
                    [0.53906606],
                    [0.44550243],
                    [0.25445301],
                    [-0.27622053],
                    [-0.72065483],
                    [0.04653594],
                    [0.01143272],
                    [-0.01000409],
                    [-0.02870777],
                    [-0.04345144],
                    [-0.05704806],
                    [-0.06786803],
                    [-0.06543554],
                    [-0.06456724],
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
                    [4.3337934],
                    [-0.67183991],
                    [0.36107803],
                    [0.99347236],
                    [1.26409781],
                    [1.197139],
                    [0.93778125],
                    [0.17623782],
                    [-0.88438038],
                    [-1.71278833],
                    [-1.70022093],
                    [-0.97427912],
                    [-0.42911134],
                    [0.00415131],
                    [0.89370216],
                    [1.8963736],
                    [3.02287818],
                    [4.71619347],
                    [6.46773665],
                    [0.04826649],
                    [0.02803653],
                    [0.11319903],
                    [0.03634749],
                    [0.26928769],
                    [0.0127666],
                    [0.2630436],
                    [0.04371043],
                    [-0.06064537],
                    [-0.50684267],
                    [0.02879289],
                    [0.00713705],
                    [0.15806286],
                    [-0.02637439],
                    [0.21027499],
                    [-0.58713636],
                    [0.41859708],
                    [-0.15693227],
                    [0.11027975],
                    [0.01101085],
                    [-0.02258444],
                    [0.03737889],
                    [-0.06484071],
                    [0.26388575],
                    [0.27276151],
                    [-0.13398968],
                    [0.19911019],
                    [0.11546176],
                    [-0.04409491],
                    [0.05560376],
                    [0.04998228],
                    [0.0238965],
                    [-0.14329437],
                    [-0.64763192],
                    [0.33576053],
                    [-0.02289197],
                    [-0.03220506],
                    [0.25614218],
                    [0.10872063],
                    [1.08348498],
                    [-0.00362426],
                    [-0.02605788],
                    [0.02880054],
                    [-0.1163122],
                    [0.3046938],
                    [-0.41160893],
                    [-0.10391308],
                    [-0.04750069],
                    [-0.29383203],
                    [-3.03072003],
                    [0.02128703],
                    [0.12485638],
                    [-0.12230623],
                    [0.08657523],
                    [0.48068493],
                    [0.1689772],
                    [0.34921139],
                    [0.44832733],
                    [0.44933166],
                    [1.00082005],
                    [-0.07298075],
                    [-0.0190721],
                    [0.02352785],
                    [0.00103326],
                    [-0.51086455],
                    [0.0720335],
                    [-0.42562033],
                    [0.14663872],
                    [-0.19129143],
                    [-1.06707937],
                    [-0.0885967],
                    [0.0327984],
                    [0.2974614],
                    [-0.03732336],
                    [-0.0067458],
                    [-0.42455637],
                    [-0.4343658],
                    [-0.26563584],
                    [-0.16471755],
                    [0.26741968],
                    [0.03616787],
                    [-0.00905428],
                    [-0.11380252],
                    [0.02947205],
                    [-0.17901383],
                    [-0.18918241],
                    [-0.31389899],
                    [0.29767963],
                    [-0.26346015],
                    [-0.20855018],
                    [-0.04096624],
                    [0.04518247],
                    [-0.28947543],
                    [0.31868901],
                    [0.58231496],
                    [0.49766936],
                    [-0.20775578],
                    [-0.09529996],
                    [-0.01538831],
                    [0.20972523],
                    [-0.01660279],
                    [0.01041328],
                    [0.07478791],
                    [0.10065407],
                    [-0.20424074],
                    [0.07719582],
                    [0.30357735],
                    [0.2001481],
                    [-0.16863715],
                    [0.57957567],
                    [0.02694383],
                    [-0.06786133],
                    [-0.14671703],
                    [0.06650039],
                    [-0.16885761],
                    [-0.35715434],
                    [0.16534232],
                    [-0.45897926],
                    [-0.35156359],
                    [0.82543548],
                    [0.00371391],
                    [0.08760521],
                    [0.22450349],
                    [-0.28986007],
                    [-0.25246828],
                    [-0.02373842],
                    [0.08924352],
                    [-0.16053201],
                    [-0.1333761],
                    [0.48347665],
                    [0.07886424],
                    [-0.00572481],
                    [-0.19531093],
                    [0.11997979],
                    [-0.16411782],
                    [-0.56116032],
                    [-0.16841617],
                    [0.39318995],
                    [0.11534525],
                    [1.13474509],
                    [-0.01215829],
                    [0.02670412],
                    [0.26363358],
                    [-0.34735499],
                    [-0.05134187],
                    [0.54313027],
                    [-0.40113299],
                    [-0.04004757],
                    [-0.00057945],
                    [0.17824561],
                    [-0.03508846],
                    [-0.01729312],
                    [0.12189015],
                    [-0.28875698],
                    [-0.24148508],
                    [0.09187039],
                    [0.23762201],
                    [-0.41020104],
                    [0.51091602],
                    [0.27002684],
                    [-0.01299843],
                    [0.00409057],
                    [-0.09368029],
                    [0.22793119],
                    [-0.08466904],
                    [0.1219294],
                    [-0.02438717],
                    [-0.46775333],
                    [0.30516531],
                    [-0.61783438],
                    [0.01973811],
                    [0.01366687],
                    [-0.4212647],
                    [0.07558249],
                    [-0.37178917],
                    [0.89664008],
                    [0.51782389],
                    [0.76799581],
                    [0.22655926],
                    [-1.23745417],
                    [0.03803879],
                    [0.01819071],
                    [0.00376931],
                    [0.18267585],
                    [-0.25871975],
                    [-0.32445422],
                    [-0.35652889],
                    [-0.17535872],
                    [-0.11950472],
                    [0.56891088],
                    [0.3021903],
                    [-0.44229743],
                    [0.70869284],
                    [0.8898239],
                    [0.30911353],
                    [0.34759764],
                    [0.32849039],
                    [-0.0591877],
                    [-0.25116141],
                    [-0.26208804],
                    [-0.02282505],
                    [-0.00788459],
                    [0.0014938],
                    [0.01083016],
                    [0.0193732],
                    [0.02798971],
                    [0.03547143],
                    [0.03660794],
                    [0.03841562],
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
            np.array([3.574312402943036e-06, 0.0, 0.0, 0.09119187436952991]),
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
