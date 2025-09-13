# flake8: noqa
from mssm.models import *
from mssm.src.python.compare import compare_CDL
import numpy as np
import os
from mssmViz.sim import *
from mssm.src.python.repara import reparam
from mssm.src.python.gamm_solvers import (
    compute_S_emb_pinv_det,
    cpp_chol,
    cpp_cholP,
    compute_eigen_perm,
    compute_Linv,
)
from mssm.src.python.utils import correct_VB, estimateVp
import io
from contextlib import redirect_stdout
from .defaults import default_gamm_test_kwargs, max_atol, max_rtol


class Test_BIG_GAMM_Discretize:
    dat = pd.read_csv(
        "https://raw.githubusercontent.com/JoKra1/mssmViz/main/data/GAMM/sim_dat.csv"
    )

    # mssm requires that the data-type for variables used as factors is 'O'=object
    dat = dat.astype({"series": "O", "cond": "O", "sub": "O", "series": "O"})

    discretize = {
        "no_disc": [],
        "excl": [],
        "split_by": ["cond"],
        "restarts": 40,
        "seed": 20,
    }

    formula = Formula(
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
            f(["time", "x"], by="cond", nk=9),  # three-way interaction
            fs(["time"], rf="series", nk=20, approx_deriv=discretize),
        ],  # Random non-linear effect of time - one smooth per level of factor series
        data=dat,
        series_id="series",
    )  # When approximating the computations for a random smooth, the series identifier column needs to be specified!

    model = GAMM(formula, Gaussian())

    model.fit(**default_gamm_test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf, ndigits=0) == 2429.0

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma, ndigits=3) == 10.97

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml, ndigits=2) == -84062.14

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk, ndigits=0) == -75232.0


class Test_NUll_penalty_reparam:
    dat = pd.read_csv(
        "https://raw.githubusercontent.com/JoKra1/mssmViz/main/data/GAMM/sim_dat.csv"
    )

    # mssm requires that the data-type for variables used as factors is 'O'=object
    dat = dat.astype({"series": "O", "cond": "O", "sub": "O", "series": "O"})

    # Add Null-penalties to univariate by-term and tensor by-term
    formula = Formula(
        lhs=lhs("y"),  # The dependent variable - here y!
        terms=[
            i(),  # The intercept, a
            l(["cond"]),  # For cond='b'
            f(
                ["time"], by="cond", constraint=ConstType.QR, penalize_null=True
            ),  # to-way interaction between time and cond; one smooth over time per cond level
            f(
                ["x"], by="cond", constraint=ConstType.QR, penalize_null=False
            ),  # to-way interaction between x and cond; one smooth over x per cond level
            f(
                ["time", "x"],
                by="cond",
                constraint=ConstType.QR,
                penalize_null=True,
                nk=9,
            ),  # three-way interaction
            fs(["time"], rf="sub"),
        ],  # Random non-linear effect of time - one smooth per level of factor sub
        data=dat,
        print_warn=False,
    )

    model = GAMM(formula, Gaussian())

    model.fit(**default_gamm_test_kwargs)

    # Compute re-parameterization strategy from Wood (2011)
    S_emb, S_pinv, _, _ = compute_S_emb_pinv_det(
        len(model.coef), model.overall_penalties, "svd"
    )
    (
        Sj_reps,
        S_emb_rp,
        S_inv_rp,
        S_root_rp,
        S_reps,
        SJ_term_idx,
        S_idx,
        S_coefs,
        Q_reps,
        Mp,
    ) = reparam(None, model.overall_penalties, None, option=4, form_inverse=True)

    def test_reparam_1(self):
        # Transformation strategy from Wood (2011) &  Wood, Li, Shaddick, & Augustin (2017)
        assert np.allclose(
            (self.S_inv_rp @ self.Sj_reps[0].S_J_emb).trace(),
            self.Sj_reps[0].rank / self.Sj_reps[0].lam,
        )

    def test_reparam2(self):
        # General strategy, e.g. from Wood & Fasiolo, 2017
        assert np.allclose(
            (self.S_inv_rp @ self.Sj_reps[0].S_J_emb).trace(),
            (self.S_pinv @ self.model.overall_penalties[0].S_J_emb).trace(),
        )

    def test_reparam3(self):
        # General strategy (here for tensor), e.g. from Wood & Fasiolo, 2017
        assert np.allclose(
            (self.S_inv_rp @ self.Sj_reps[6].S_J_emb).trace(),
            (self.S_pinv @ self.model.overall_penalties[6].S_J_emb).trace(),
        )

    def test_GAMedf_hard(self):
        np.testing.assert_allclose(
            self.model.edf,
            151.4620626246641,
            atol=min(max_atol, 0.0),
            rtol=min(max_rtol, 0.002),
        )

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        np.testing.assert_allclose(
            sigma,
            577.1990564685502,
            atol=min(max_atol, 0.0),
            rtol=min(max_rtol, 0.001),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml,
            -134748.71831654513,
            atol=min(max_atol, 0.0),
            rtol=min(max_rtol, 0.01),
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk,
            -134264.97369100052,
            atol=min(max_atol, 0.0),
            rtol=min(max_rtol, 0.01),
        )


class Test_NUll_1:
    dat = pd.read_csv(
        "https://raw.githubusercontent.com/JoKra1/mssmViz/main/data/GAMM/sim_dat.csv"
    )

    # mssm requires that the data-type for variables used as factors is 'O'=object
    dat = dat.astype({"series": "O", "cond": "O", "sub": "O", "series": "O"})

    # Add Null-penalties to univariate by-term and tensor by-term
    formula = Formula(
        lhs=lhs("y"),  # The dependent variable - here y!
        terms=[
            i(),  # The intercept, a
            l(["cond"]),  # For cond='b'
            f(
                ["time"], by="cond", constraint=ConstType.QR, penalize_null=True
            ),  # to-way interaction between time and cond; one smooth over time per cond level
            f(
                ["x"], by="cond", constraint=ConstType.QR, penalize_null=False
            ),  # to-way interaction between x and cond; one smooth over x per cond level
            f(
                ["time", "x"],
                by="cond",
                constraint=ConstType.QR,
                penalize_null=True,
                nk=9,
            ),  # three-way interaction
            fs(["time"], rf="sub"),
        ],  # Random non-linear effect of time - one smooth per level of factor sub
        data=dat,
        print_warn=False,
        find_nested=False,
    )

    model = GAMM(formula, Gaussian())

    model.fit(**default_gamm_test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf, ndigits=2) == 151.46

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma, ndigits=3) == 577.199

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml, ndigits=3) == -134748.718

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk, ndigits=1) == -134265.0


class Test_NUll_2:
    dat = pd.read_csv(
        "https://raw.githubusercontent.com/JoKra1/mssmViz/main/data/GAMM/sim_dat.csv"
    )

    # mssm requires that the data-type for variables used as factors is 'O'=object
    dat = dat.astype({"series": "O", "cond": "O", "sub": "O", "series": "O"})

    # Add Null-penalties to univariate by-term and tensor by-term
    formula = Formula(
        lhs=lhs("y"),  # The dependent variable - here y!
        terms=[
            i(),  # The intercept, a
            l(["cond"]),  # For cond='b'
            f(
                ["time"], by="cond", constraint=ConstType.QR, penalize_null=True
            ),  # to-way interaction between time and cond; one smooth over time per cond level
            f(
                ["x"], by="cond", constraint=ConstType.QR, penalize_null=False
            ),  # to-way interaction between x and cond; one smooth over x per cond level
            f(
                ["time", "x"],
                by="cond",
                constraint=ConstType.QR,
                penalize_null=True,
                id=1,
                nk=9,
            ),  # three-way interaction
            fs(["time"], rf="sub"),
        ],  # Random non-linear effect of time - one smooth per level of factor sub
        data=dat,
        print_warn=False,
        find_nested=False,
    )

    model = GAMM(formula, Gaussian())

    model.fit(**default_gamm_test_kwargs)

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            151.45736656774903,
            atol=min(max_atol, 0.0),
            rtol=min(max_rtol, 0.01),
        )

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        np.testing.assert_allclose(
            sigma,
            577.1990984875418,
            atol=min(max_atol, 0.0),
            rtol=min(max_rtol, 0.01),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml,
            -134748.71762114676,
            atol=min(max_atol, 0.0),
            rtol=min(max_rtol, 0.01),
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk,
            -134264.9771024987,
            atol=min(max_atol, 0.0),
            rtol=min(max_rtol, 0.01),
        )


class Test_NUll_3:
    dat = pd.read_csv(
        "https://raw.githubusercontent.com/JoKra1/mssmViz/main/data/GAMM/sim_dat.csv"
    )

    # mssm requires that the data-type for variables used as factors is 'O'=object
    dat = dat.astype({"series": "O", "cond": "O", "sub": "O", "series": "O"})

    # Add Null-penalties to univariate by-term and tensor by-term
    formula = Formula(
        lhs=lhs("y"),  # The dependent variable - here y!
        terms=[
            i(),  # The intercept, a
            l(["cond"]),  # For cond='b'
            f(
                ["time"], by="cond", constraint=ConstType.QR, penalize_null=True
            ),  # to-way interaction between time and cond; one smooth over time per cond level
            f(
                ["x"], by="cond", constraint=ConstType.QR, penalize_null=False
            ),  # to-way interaction between x and cond; one smooth over x per cond level
            f(
                ["time", "x"],
                by="cond",
                constraint=ConstType.QR,
                penalize_null=True,
                nk=9,
            ),  # three-way interaction
            fs(["time"], rf="sub"),
        ],  # Random non-linear effect of time - one smooth per level of factor sub
        data=dat,
        print_warn=False,
        find_nested=True,
    )

    model = GAMM(formula, Gaussian())

    model.fit(**default_gamm_test_kwargs)

    def test_GAMedf_hard(self):
        np.testing.assert_allclose(
            self.model.edf,
            151.4620626246641,
            atol=min(max_atol, 0.0),
            rtol=min(max_rtol, 0.002),
        )

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        np.testing.assert_allclose(
            sigma,
            577.1990564685502,
            atol=min(max_atol, 0.0),
            rtol=min(max_rtol, 0.001),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml,
            -134748.71831654513,
            atol=min(max_atol, 0.0),
            rtol=min(max_rtol, 0.01),
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk,
            -134264.97369100052,
            atol=min(max_atol, 0.0),
            rtol=min(max_rtol, 0.001),
        )


class Test_NUll_4:
    dat = pd.read_csv(
        "https://raw.githubusercontent.com/JoKra1/mssmViz/main/data/GAMM/sim_dat.csv"
    )

    # mssm requires that the data-type for variables used as factors is 'O'=object
    dat = dat.astype({"series": "O", "cond": "O", "sub": "O", "series": "O"})

    # Add Null-penalties to univariate by-term and tensor by-term
    formula = Formula(
        lhs=lhs("y"),  # The dependent variable - here y!
        terms=[
            i(),  # The intercept, a
            l(["cond"]),  # For cond='b'
            f(
                ["time"], by="cond", constraint=ConstType.QR, penalize_null=True
            ),  # to-way interaction between time and cond; one smooth over time per cond level
            f(
                ["x"], by="cond", constraint=ConstType.QR, penalize_null=False
            ),  # to-way interaction between x and cond; one smooth over x per cond level
            f(
                ["time", "x"],
                by="cond",
                constraint=ConstType.QR,
                penalize_null=True,
                id=1,
                nk=9,
            ),  # three-way interaction
            fs(["time"], rf="sub"),
        ],  # Random non-linear effect of time - one smooth per level of factor sub
        data=dat,
        print_warn=False,
        find_nested=True,
    )

    model = GAMM(formula, Gaussian())

    model.fit(**default_gamm_test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf, ndigits=2) == 151.46

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma, ndigits=3) == 577.199

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml, ndigits=3) == -134748.718

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk, ndigits=1) == -134265.0


class Test_ar1_Gaussian:
    dat = pd.read_csv(
        "https://raw.githubusercontent.com/JoKra1/mssmViz/main/data/GAMM/sim_dat.csv"
    )

    # mssm requires that the data-type for variables used as factors is 'O'=object
    dat = dat.astype({"series": "O", "cond": "O", "sub": "O", "series": "O"})

    formula = Formula(
        lhs=lhs("y"),
        terms=[
            i(),
            l(["cond"]),
            f(["time"], by="cond"),
            f(["x"], by="cond"),
            f(["time", "x"], by="cond"),
        ],
        data=dat,
        print_warn=False,
        series_id="series",
    )  # 'series' variable identifies individual time-series

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["rho"] = 0.97
    test_kwargs["max_inner"] = 1

    model = GAMM(formula, Gaussian())
    model.fit(**test_kwargs)

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            20.848721033068795,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.0001),
        )

    def test_GAMcoef(self):
        coef = self.model.coef
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    [36.56746619],
                    [18.38610041],
                    [-194.86662089],
                    [-69.64843072],
                    [-151.3548762],
                    [-165.01347493],
                    [-96.31054075],
                    [-105.92452482],
                    [475.66390794],
                    [1047.67911573],
                    [1617.59990081],
                    [-73.51078279],
                    [33.0187979],
                    [14.37041146],
                    [-11.65065876],
                    [-13.83346069],
                    [-36.02625752],
                    [-36.8593549],
                    [42.77691521],
                    [115.10561624],
                    [-6.45145897],
                    [-4.46471154],
                    [-1.17417159],
                    [3.90436888],
                    [6.32542163],
                    [4.97724076],
                    [4.6508228],
                    [5.98385755],
                    [7.7258949],
                    [-6.54343469],
                    [-4.52837177],
                    [-1.190887],
                    [3.96008844],
                    [6.41555198],
                    [5.04813343],
                    [4.71715262],
                    [6.06929885],
                    [7.83627795],
                    [-0.13894973],
                    [0.48227914],
                    [0.42007256],
                    [0.5196528],
                    [-0.50033832],
                    [1.73661863],
                    [1.51262315],
                    [1.8711972],
                    [-0.52197612],
                    [1.81172224],
                    [1.57803973],
                    [1.95212164],
                    [-0.8028081],
                    [2.78645856],
                    [2.4270506],
                    [3.00239477],
                    [-0.08915421],
                    [0.30944679],
                    [0.26953233],
                    [0.33342608],
                    [-0.32103445],
                    [1.1142749],
                    [0.97054976],
                    [1.20062169],
                    [-0.33491735],
                    [1.16246303],
                    [1.01252275],
                    [1.25254547],
                    [-0.51510814],
                    [1.78788576],
                    [1.55727613],
                    [1.92643499],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.01),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        np.testing.assert_allclose(
            lam,
            np.array(
                [
                    0.002621769139735816,
                    0.043577859437411604,
                    10000000.0,
                    10000000.0,
                    10000000.0,
                    10000000.0,
                    10000000.0,
                    10000000.0,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.01),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -86642.72573737243, atol=min(max_atol, 0), rtol=min(max_rtol, 0.0001)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -86608.91815873925, atol=min(max_atol, 0), rtol=min(max_rtol, 0.0001)
        )


class Test_ar1_Gamma:

    # We simulate some data including a random smooth - but then dont include it in the model:
    sim_dat = sim11(5000, 2, c=0, seed=20, family=Gamma(), n_ranef=20, binom_offset=0)

    sim_dat = sim_dat.sort_values(["x4"], ascending=[True])

    sim_formula = Formula(
        lhs("y"),
        [i(), f(["x0"]), f(["x1"]), f(["x2"]), f(["x3"])],
        data=sim_dat,
        series_id="x4",
    )  # Can already specify this.

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["rho"] = 0.2
    test_kwargs["max_inner"] = 1

    model = GAMM(sim_formula, Gamma())
    model.fit(**test_kwargs)

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            18.88816369593533,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.006),
        )

    def test_GAMcoef(self):
        coef = self.model.coef
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    [8.71146084e00],
                    [-1.00898003e00],
                    [-1.35194807e-01],
                    [4.95614074e-01],
                    [1.06551493e00],
                    [1.51182884e00],
                    [1.28590751e00],
                    [9.62143497e-01],
                    [3.19487176e-01],
                    [-3.42961059e-01],
                    [-1.71386039e00],
                    [-8.81312293e-01],
                    [-4.48079882e-01],
                    [6.20992247e-02],
                    [7.73659940e-01],
                    [1.84859538e00],
                    [3.34299994e00],
                    [4.67604185e00],
                    [6.01029269e00],
                    [-1.17671872e01],
                    [-9.59794036e-03],
                    [7.83946095e-01],
                    [-7.24148541e00],
                    [-5.10575246e00],
                    [-5.50342036e00],
                    [-8.98985234e00],
                    [-6.08908095e00],
                    [-3.96555809e00],
                    [-8.23194020e-03],
                    [-2.82269559e-03],
                    [2.76233626e-04],
                    [3.21516898e-03],
                    [6.39627620e-03],
                    [9.94498032e-03],
                    [1.33678483e-02],
                    [1.52476383e-02],
                    [1.70276243e-02],
                ]
            ),
            atol=min(max_atol, 0.85),
            rtol=min(max_rtol, 0.01),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        np.testing.assert_allclose(
            lam,
            np.array(
                [
                    52.28279162655495,
                    55.95969508678077,
                    0.035347301657971626,
                    244928.41719951248,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.75),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -47019.61861396541, atol=min(max_atol, 0), rtol=min(max_rtol, 0.002)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -46974.16558162963, atol=min(max_atol, 0), rtol=min(max_rtol, 0.002)
        )


class Test_inval_checks_hard:

    sim_dat = sim3(
        500, 2, c=1, seed=66, family=Binomial(), binom_offset=-5, correlate=True
    )

    formula = Formula(
        lhs("y"), [i(), f(["x0"]), f(["x1"]), f(["x2"]), f(["x3"])], data=sim_dat
    )

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["max_inner"] = 1
    test_kwargs["max_outer"] = 29
    test_kwargs["control_lambda"] = 2
    test_kwargs["extend_lambda"] = False
    test_kwargs["progress_bar"] = False
    model = GAMM(formula, Binomial())
    model.fit(**test_kwargs)

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            9.230211003055484,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.17),
        )

    def test_GAMcoef(self):
        coef = self.model.coef
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    [1.35973108e01],
                    [4.38060893e00],
                    [1.18645243e00],
                    [-5.07909867e-01],
                    [-2.08160966e00],
                    [-3.81652788e00],
                    [-5.15206931e00],
                    [-6.98157830e00],
                    [-7.50767600e00],
                    [-7.76514484e00],
                    [-9.95033969e00],
                    [-3.35349174e00],
                    [1.22180197e00],
                    [5.97322489e00],
                    [8.98941311e00],
                    [1.20655134e01],
                    [1.55650518e01],
                    [1.73222400e01],
                    [1.79559096e01],
                    [-1.09770916e01],
                    [2.06106391e01],
                    [3.29897995e01],
                    [3.20131117e01],
                    [2.44324300e01],
                    [9.00198062e00],
                    [-3.81431806e00],
                    [-7.77424968e00],
                    [-1.55217198e01],
                    [-2.57942232e00],
                    [-2.01684861e-02],
                    [1.47911015e00],
                    [2.27565876e00],
                    [3.13416885e00],
                    [3.80017438e00],
                    [3.07103813e00],
                    [1.22007344e00],
                    [-8.16094394e-01],
                ]
            ),
            atol=min(max_atol, 20),
            rtol=min(max_rtol, 0.01),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -29.733544316317875, atol=min(max_atol, 0), rtol=min(max_rtol, 0.6)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -23.205125728326653, atol=min(max_atol, 0), rtol=min(max_rtol, 0.9)
        )


class Test_inval_checks_ar_hard:
    sim_dat = sim3(
        500, 2, c=1, seed=66, family=Binomial(), binom_offset=-5, correlate=True
    )

    formula = Formula(
        lhs("y"), [i(), f(["x0"]), f(["x1"]), f(["x2"]), f(["x3"])], data=sim_dat
    )

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["max_inner"] = 1
    test_kwargs["max_outer"] = 23
    test_kwargs["control_lambda"] = 2
    test_kwargs["extend_lambda"] = False
    test_kwargs["progress_bar"] = False
    test_kwargs["rho"] = 0.3
    model = GAMM(formula, Binomial())
    model.fit(**test_kwargs)

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            9.916179496952736,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.15),
        )

    def test_GAMcoef(self):
        coef = self.model.coef
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    [15.92982628],
                    [5.20305592],
                    [1.39435932],
                    [-0.62636022],
                    [-2.49782216],
                    [-4.55238561],
                    [-6.12388233],
                    [-8.27440849],
                    [-8.86604066],
                    [-9.13663688],
                    [-10.78167667],
                    [-3.63166588],
                    [1.32673993],
                    [6.47535892],
                    [9.74332076],
                    [13.07529235],
                    [16.86340316],
                    [18.76129413],
                    [19.44134501],
                    [-13.27893097],
                    [24.97448667],
                    [39.93154592],
                    [38.48734941],
                    [29.02149839],
                    [10.28687776],
                    [-4.20723914],
                    [-8.21816396],
                    [-17.58423002],
                    [-4.02800656],
                    [0.19886441],
                    [2.47788889],
                    [3.59030413],
                    [4.9727597],
                    [6.11882263],
                    [4.96189494],
                    [1.71710551],
                    [-1.91771102],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 5),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -7.043260375631441, atol=min(max_atol, 0), rtol=min(max_rtol, 2.7)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, 1.5261519408294362, atol=min(max_atol, 0), rtol=min(max_rtol, 13.5)
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

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["max_inner"] = 1
    test_kwargs["control_lambda"] = 2
    test_kwargs["extend_lambda"] = False
    test_kwargs["progress_bar"] = True
    test_kwargs["method"] = "QR"
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
        n_c=4,
        seed=20,
        method="QR",
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
            105.8437533607087,
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
            106.26517126946953,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.03),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        np.testing.assert_allclose(
            lam,
            np.array(
                [
                    19.137384994846876,
                    19.74907290324467,
                    0.011401944734128895,
                    10000000.0,
                    4.810489817526103,
                    4.3009070593302905,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.5),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -10643.879807223055, atol=min(max_atol, 0), rtol=min(max_rtol, 0.01)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -10436.834138445849, atol=min(max_atol, 0), rtol=min(max_rtol, 0.01)
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
                        -0.20450685,
                        -0.55404532,
                        -0.12835239,
                        -0.88806399,
                        -0.37087577,
                        -0.65576607,
                        -0.27318399,
                        -0.32180159,
                        -0.83148069,
                        -0.323851,
                    ],
                    [
                        0.487708,
                        0.06778094,
                        0.72737646,
                        0.8215387,
                        0.33416501,
                        -0.07580399,
                        -0.11998745,
                        0.48371935,
                        0.38062297,
                        0.2567328,
                    ],
                    [
                        1.0781216,
                        0.98005613,
                        1.33541841,
                        1.33421394,
                        0.79740396,
                        0.71500351,
                        0.25855318,
                        1.04483406,
                        0.83310168,
                        0.90248003,
                    ],
                    [
                        1.52136679,
                        1.30718454,
                        1.29826372,
                        1.79162056,
                        0.72285125,
                        1.15073818,
                        0.95442794,
                        1.00033799,
                        1.42767567,
                        1.28924757,
                    ],
                    [
                        1.41109431,
                        1.40768748,
                        1.1946468,
                        1.7006692,
                        1.01659908,
                        1.30628188,
                        1.04565058,
                        1.24843891,
                        1.57539931,
                        1.00681293,
                    ],
                    [
                        0.73897403,
                        0.95297849,
                        0.47191431,
                        1.17272045,
                        0.84831068,
                        0.94722309,
                        0.46459534,
                        0.74304449,
                        1.16898336,
                        0.65986357,
                    ],
                    [
                        -0.25426606,
                        0.0373777,
                        -0.71577945,
                        0.72021026,
                        -0.22637632,
                        0.08852591,
                        -0.54930259,
                        -0.49868811,
                        0.59128045,
                        -0.17501789,
                    ],
                    [
                        -1.95712733,
                        -1.35465835,
                        -1.5800743,
                        -0.5325116,
                        -1.1166962,
                        -0.71055709,
                        -1.5604723,
                        -1.62090308,
                        -0.54938167,
                        -0.91286259,
                    ],
                    [
                        -3.26823004,
                        -1.65025385,
                        -2.40660115,
                        -1.83902915,
                        -1.89431058,
                        -1.39347175,
                        -2.11934415,
                        -2.99049516,
                        -1.91326988,
                        -1.20348139,
                    ],
                ]
            ),
            atol=min(max_atol, 0.5),
            rtol=min(max_rtol, 0.5),
        )

    def test_post2(self):
        np.testing.assert_allclose(
            self.post2,
            np.array(
                [
                    [
                        -0.58969978,
                        -0.45762495,
                        -0.20635549,
                        -0.12487259,
                        -1.48061495,
                        -2.1540922,
                        -0.60922374,
                        -0.60559179,
                        0.22067118,
                        -0.85346274,
                    ],
                    [
                        -0.09031124,
                        -0.2860078,
                        1.9473088,
                        0.47497821,
                        0.36716251,
                        0.69964354,
                        -0.04093981,
                        1.27210715,
                        2.00421941,
                        -1.5045988,
                    ],
                    [
                        1.46537568,
                        0.42651864,
                        2.24785898,
                        1.30389918,
                        1.34214687,
                        1.90446929,
                        0.67923771,
                        1.70322115,
                        2.14707242,
                        -0.34351915,
                    ],
                    [
                        1.66806326,
                        0.87812887,
                        1.70652538,
                        1.07010206,
                        2.32196418,
                        2.22874478,
                        0.97286712,
                        2.55161443,
                        1.70783284,
                        0.55985965,
                    ],
                    [
                        1.55042858,
                        1.24158042,
                        2.61123563,
                        1.23274504,
                        2.53186977,
                        2.4561253,
                        0.40023125,
                        2.70691273,
                        2.52755364,
                        0.5428688,
                    ],
                    [
                        1.2937071,
                        1.09526394,
                        1.79506514,
                        1.71754123,
                        1.94020383,
                        3.00385906,
                        0.24376597,
                        2.22950797,
                        1.75249573,
                        -0.19550299,
                    ],
                    [
                        0.37034317,
                        0.32443744,
                        0.53423691,
                        0.50741044,
                        0.85755008,
                        1.72581553,
                        -0.16159212,
                        0.94273488,
                        -0.13667816,
                        -0.4537871,
                    ],
                    [
                        -0.90706986,
                        -0.41853814,
                        0.08967159,
                        -0.6928048,
                        -1.10320734,
                        1.27638547,
                        -1.98287686,
                        -0.58505599,
                        -0.84739371,
                        -0.88413497,
                    ],
                    [
                        -1.47852923,
                        -0.49585416,
                        1.28438915,
                        -2.10817684,
                        -2.45878584,
                        1.90603935,
                        -3.58932576,
                        -1.23550867,
                        -0.71811315,
                        -1.91406517,
                    ],
                ]
            ),
            atol=min(max_atol, 0.6),
            rtol=min(max_rtol, 0.5),
        )


class Test_drop_Gamma:
    sim_dat = sim13(5000, 2, c=0, seed=0, family=Gamma(), binom_offset=0, n_ranef=20)

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

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["max_inner"] = 1
    test_kwargs["control_lambda"] = 2
    test_kwargs["extend_lambda"] = False
    test_kwargs["progress_bar"] = True
    test_kwargs["method"] = "QR"
    model = GAMM(formula, Gamma())
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
        n_c=4,
        seed=20,
        method="QR",
        prior=None,
        Vp_fidiff=False,
    )

    def test_GAMedf_hard(self):
        np.testing.assert_allclose(
            self.model.edf,
            129.31792469246903,
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
            129.63885675109287,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.03),
        )

    def test_VP(self):
        np.testing.assert_allclose(
            self.res2[2], self.Vp2, atol=min(max_atol, 0), rtol=min(max_rtol, 1e-7)
        )

    def test_drop(self):
        assert len(self.model.info.dropped) == 1


class Test_BIG_GAMM:

    file_paths = [
        f"https://raw.githubusercontent.com/JoKra1/mssmViz/main/data/GAMM/sim_dat_cond_{cond}.csv"
        for cond in ["a", "b"]
    ]

    codebook = {"cond": {"a": 0, "b": 1}}

    formula = Formula(
        lhs=lhs("y"),  # The dependent variable - here y!
        terms=[
            i(),  # The intercept, a
            l(["cond"]),  # For cond='b'
            f(
                ["time"], by="cond", constraint=ConstType.QR
            ),  # to-way interaction between time and cond; one smooth over time per cond level
            f(
                ["x"], by="cond", constraint=ConstType.QR
            ),  # to-way interaction between x and cond; one smooth over x per cond level
            f(
                ["time", "x"], by="cond", constraint=ConstType.QR, nk=9
            ),  # three-way interaction
            fs(["time"], rf="sub"),
        ],  # Random non-linear effect of time - one smooth per level of factor sub
        data=None,  # No data frame!
        file_paths=file_paths,  # Just a list with paths to files.
        print_warn=False,
        codebook=codebook,
    )

    model = GAMM(formula, Gaussian())

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["exclude_lambda"] = True

    model.fit(**test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf, ndigits=3) == 153.707

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma, ndigits=3) == 577.194

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(
            lam,
            np.array(
                [
                    0.003576343523516708,
                    0.006011901683452655,
                    5028.094352875556,
                    230482.43912034066,
                    110804.13545750394,
                    38451.597466911124,
                    381047.3435206221,
                    330.2597296955685,
                    0.11887201661781975,
                    2.166381231196006,
                ]
            ),
        )


class Test_BIG_GAMM_keep_cov:
    file_paths = [
        f"https://raw.githubusercontent.com/JoKra1/mssmViz/main/data/GAMM/sim_dat_cond_{cond}.csv"
        for cond in ["a", "b"]
    ]

    codebook = {"cond": {"a": 0, "b": 1}}

    formula = Formula(
        lhs=lhs("y"),  # The dependent variable - here y!
        terms=[
            i(),  # The intercept, a
            l(["cond"]),  # For cond='b'
            f(
                ["time"], by="cond", constraint=ConstType.QR
            ),  # to-way interaction between time and cond; one smooth over time per cond level
            f(
                ["x"], by="cond", constraint=ConstType.QR
            ),  # to-way interaction between x and cond; one smooth over x per cond level
            f(
                ["time", "x"], by="cond", constraint=ConstType.QR, nk=9
            ),  # three-way interaction
            fs(["time"], rf="sub"),
        ],  # Random non-linear effect of time - one smooth per level of factor sub
        data=None,  # No data frame!
        file_paths=file_paths,  # Just a list with paths to files.
        print_warn=False,
        keep_cov=True,  # Keep encoded data structure in memory
        codebook=codebook,
    )

    model = GAMM(formula, Gaussian())

    model.fit(**default_gamm_test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf, ndigits=1) == 151.8

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma, ndigits=3) == 577.166
