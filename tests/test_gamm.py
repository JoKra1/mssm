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

################################################################## Tests ##################################################################


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


class Test_drop_no_pen_order:
    sim_dat = sim13(500, 2, c=0, seed=0, family=Gamma(), binom_offset=0, n_ranef=20)

    formula1 = Formula(
        lhs("y"),
        [i(), *li(["x0", "x5", "x6", "x4"])],
        data=sim_dat,
    )

    formula2 = Formula(
        lhs("y"),
        [i(), *li(["x5", "x6", "x0", "x4"])],
        data=sim_dat,
    )

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["max_inner"] = 1
    test_kwargs["max_outer"] = 200
    test_kwargs["control_lambda"] = 2
    test_kwargs["extend_lambda"] = False
    test_kwargs["progress_bar"] = True
    test_kwargs["method"] = "QR"
    model1 = GAMM(formula1, Gamma())
    model1.fit(**test_kwargs)
    model2 = GAMM(formula2, Gamma())
    model2.fit(**test_kwargs)

    def test_drop(self):
        assert len(self.model1.info.dropped) == 1

    def test_drop(self):
        assert len(self.model2.info.dropped) == 1

    def test_order(self):
        np.testing.assert_allclose(
            np.sort(self.model1.coef.flatten()),
            np.sort(self.model2.coef.flatten()),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1e-7),
        )

    def test_GAMllk1(self):
        llk = self.model1.get_llk(False)
        np.testing.assert_allclose(
            llk, -2209.29436493949, atol=min(max_atol, 0), rtol=min(max_rtol, 0.01)
        )

    def test_GAMllk2(self):
        llk = self.model2.get_llk(False)
        np.testing.assert_allclose(
            llk, -2209.29436493949, atol=min(max_atol, 0), rtol=min(max_rtol, 0.01)
        )

    def test_scale1(self):
        scale = self.model1.scale
        np.testing.assert_allclose(
            scale, 1.9825008568078437, atol=min(max_atol, 0), rtol=min(max_rtol, 0.01)
        )

    def test_scale2(self):
        scale = self.model2.scale
        np.testing.assert_allclose(
            scale, 1.9825008568078437, atol=min(max_atol, 0), rtol=min(max_rtol, 0.01)
        )


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


class Test_rs_ri:

    # Random slope + intercept model
    sim_dat, _ = sim1(100, random_seed=100)

    # Specify formula
    formula = Formula(
        lhs("y"),
        [i(), *li(["x", "fact"]), ri("series"), rs(["x", "series"])],
        data=sim_dat,
    )

    # ... and model
    model = GAMM(formula, Gaussian())

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["max_outer"] = 100

    model.fit(**test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf, ndigits=3) == 97.59

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma, ndigits=3) == 46.805

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(
            coef.flatten(),
            np.array(
                [
                    9.34823357e00,
                    -4.12480712e-01,
                    -9.79726654e00,
                    -8.84147613e00,
                    2.53418688e-02,
                    3.14635694e-01,
                    -4.06353720e00,
                    1.93833167e00,
                    2.76734845e-01,
                    1.23222086e00,
                    -2.55837222e00,
                    4.74284163e00,
                    3.98128948e00,
                    -9.93660084e-01,
                    -3.54564194e00,
                    4.29251585e00,
                    -1.32605854e00,
                    -4.14415477e00,
                    5.70593558e00,
                    8.25028306e00,
                    1.15002976e01,
                    1.31992300e00,
                    -1.57502454e00,
                    -6.61320296e-01,
                    3.58538448e00,
                    2.58927345e00,
                    7.10253184e00,
                    1.06922717e00,
                    2.37380561e00,
                    -2.57901169e00,
                    1.59727296e00,
                    -4.19457598e00,
                    -2.42430595e-01,
                    -2.08536268e-01,
                    7.56330983e00,
                    -1.16692771e00,
                    5.71760774e00,
                    -5.39240799e00,
                    -8.15056570e00,
                    -3.93691672e00,
                    5.86576614e00,
                    -1.12113744e00,
                    -5.23756836e-01,
                    -5.93227243e-01,
                    -2.52520476e00,
                    6.10156119e00,
                    -5.96965678e00,
                    -5.95474407e00,
                    3.08696984e00,
                    -4.79732675e00,
                    5.25291394e00,
                    -2.14507799e00,
                    -7.25884552e00,
                    -2.94826284e00,
                    -5.16986996e00,
                    -4.31583159e00,
                    8.14360337e-01,
                    1.67010523e00,
                    -4.55225606e00,
                    2.31206533e00,
                    2.84686087e00,
                    -5.89265083e00,
                    1.18688704e00,
                    -4.25610602e00,
                    4.35120004e-01,
                    2.97108387e00,
                    -1.72185347e-01,
                    2.46427700e-01,
                    -3.52651093e00,
                    2.41571936e-01,
                    7.17557045e-01,
                    4.81392854e00,
                    -7.86218803e-01,
                    -6.14760379e00,
                    -5.27462938e00,
                    3.03928369e00,
                    -2.37407768e00,
                    -4.01300413e-01,
                    8.68982320e00,
                    -1.86560253e00,
                    4.96356356e00,
                    -8.73902694e00,
                    2.67441566e00,
                    -6.34198980e00,
                    4.71752258e-02,
                    1.22108898e00,
                    2.05473847e00,
                    -5.08976695e00,
                    4.30813260e00,
                    5.53216376e00,
                    -6.50838857e-01,
                    4.74846838e00,
                    -4.35251328e-01,
                    -1.78397142e00,
                    -5.90334091e00,
                    6.77475154e-01,
                    -3.29062209e00,
                    1.22037194e00,
                    8.45255403e00,
                    -3.13018031e00,
                    9.47424367e00,
                    1.28155836e00,
                    -2.14077758e00,
                    5.15760303e-01,
                    2.82194429e-01,
                    -1.17679851e01,
                    -6.84532713e-03,
                    1.17214540e-03,
                    2.74926778e-04,
                    7.09663802e-04,
                    -2.87317738e-03,
                    7.37507574e-03,
                    4.58582893e-03,
                    -2.86135634e-05,
                    -1.07205799e-03,
                    8.21992121e-03,
                    -2.00473098e-03,
                    -7.27947239e-03,
                    4.60064791e-03,
                    7.95880306e-03,
                    2.25191552e-02,
                    2.73662440e-03,
                    -2.60789002e-03,
                    -4.95130068e-04,
                    5.05901441e-03,
                    3.83989411e-03,
                    1.16579489e-02,
                    1.90895538e-03,
                    4.47734494e-03,
                    -1.70810769e-03,
                    1.24187249e-03,
                    -7.73040312e-03,
                    -4.53769065e-04,
                    -1.20100743e-04,
                    1.28498484e-02,
                    -2.38581301e-03,
                    7.57366847e-03,
                    -1.14907550e-02,
                    -2.22969500e-03,
                    -5.89513389e-03,
                    3.54713846e-03,
                    -1.71107414e-03,
                    -6.86238695e-04,
                    -8.19967098e-04,
                    -3.34494155e-03,
                    1.84486407e-03,
                    -8.76705316e-03,
                    -3.60094508e-03,
                    6.00026265e-03,
                    -3.66082758e-03,
                    3.47906244e-03,
                    -2.47079765e-04,
                    -2.29929313e-03,
                    -2.54695666e-03,
                    -6.77368689e-03,
                    -2.92056216e-03,
                    5.27634559e-04,
                    2.59699849e-03,
                    -4.71914460e-03,
                    1.66446325e-04,
                    4.87772499e-03,
                    -5.09056590e-03,
                    1.52090995e-03,
                    -8.21147817e-03,
                    3.75893145e-04,
                    5.81778847e-03,
                    -1.46269005e-04,
                    3.54808185e-04,
                    -4.31586912e-03,
                    5.91288602e-05,
                    3.71931567e-04,
                    4.85178766e-03,
                    -1.50556383e-03,
                    0.00000000e00,
                    -8.20200088e-03,
                    2.45054890e-03,
                    -4.06767263e-03,
                    -8.26246724e-04,
                    3.00279918e-03,
                    -3.73368805e-03,
                    3.35888823e-03,
                    -6.54290370e-03,
                    1.23220507e-03,
                    -5.02218070e-03,
                    6.52062653e-05,
                    3.69207668e-04,
                    2.72175159e-03,
                    -1.08458531e-02,
                    3.65969742e-03,
                    7.40767777e-03,
                    -1.49933114e-04,
                    8.13588157e-03,
                    -2.63204616e-04,
                    -2.87680232e-03,
                    -2.29491040e-03,
                    4.97470869e-04,
                    -2.32155082e-03,
                    1.10697259e-03,
                    7.54542577e-03,
                    -5.99411544e-03,
                    3.41026922e-03,
                    1.29163717e-03,
                    -2.43502119e-03,
                    4.45556998e-05,
                    1.95026568e-04,
                    -1.94851637e-02,
                ]
            ),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(lam, np.array([2.21567798e00, 2.58200022e04]))

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml, ndigits=3) == -32120.91

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk, ndigits=3) == -31886.766


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

    def test_GAMedf(self):
        assert round(self.model.edf, ndigits=3) == 6

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma, ndigits=3) == 68.036

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(
            coef.flatten(),
            np.array(
                [
                    9.09462398,
                    -0.40030025,
                    -10.08258653,
                    -8.66437813,
                    0.07117496,
                    0.29090723,
                ]
            ),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(lam, np.array([]))

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk, ndigits=3) == -33719.718


class Test_te_rs_fact:
    # te + random slope with factor
    sim_dat, _ = sim1(100, random_seed=100)

    # Specify formula
    formula = Formula(
        lhs("y"),
        [
            i(),
            f(["time", "x"], te=True, nk=5),
            rs(["fact", "sub"]),
            rs(["x", "sub"], by="fact"),
        ],
        data=sim_dat,
    )

    # ... and model
    model = GAMM(formula, Gaussian())

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["max_outer"] = 100

    model.fit(**test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf, ndigits=3) == 92.601

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma, ndigits=3) == 47.63

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(
            coef.flatten(),
            np.array(
                [
                    -8.30300915e-01,
                    -1.62575662e01,
                    3.75545293e00,
                    3.28335584e01,
                    -1.03918612e02,
                    3.44105191e01,
                    5.27629918e00,
                    1.56969059e01,
                    1.26673363e01,
                    -1.00123881e02,
                    6.70143258e01,
                    3.86562330e00,
                    1.95649156e01,
                    4.86327309e00,
                    -7.25159919e01,
                    1.30742924e01,
                    1.95090107e00,
                    9.28040152e00,
                    7.11371053e00,
                    -4.42679942e01,
                    -1.76804150e01,
                    4.78613869e01,
                    -1.98052935e01,
                    -3.54456054e01,
                    -1.83416110e01,
                    0.00000000e00,
                    8.09245518e-01,
                    -1.88785044e00,
                    7.86729219e00,
                    2.24203537e00,
                    0.00000000e00,
                    1.63941579e01,
                    1.90460014e00,
                    4.75391046e00,
                    7.87148618e00,
                    5.89124427e00,
                    -2.05573658e00,
                    1.17824896e01,
                    5.67949983e00,
                    5.60320399e-01,
                    7.16489667e00,
                    4.25921467e00,
                    6.32718677e00,
                    2.33817316e00,
                    4.63554642e00,
                    0.00000000e00,
                    4.33279542e00,
                    -1.54508599e01,
                    -4.34825704e-01,
                    -1.31251334e01,
                    -8.82479024e00,
                    -2.04672844e00,
                    -6.63234939e00,
                    6.01391323e-01,
                    -3.45057541e00,
                    -2.36964763e01,
                    0.00000000e00,
                    -1.27173425e00,
                    -1.07503433e-01,
                    2.33531411e00,
                    -7.66907848e00,
                    -6.74989624e-01,
                    -1.78519227e00,
                    -1.07199785e01,
                    7.97691278e-01,
                    2.06075903e00,
                    6.57708624e00,
                    -6.56854566e00,
                    0.00000000e00,
                    5.72411711e00,
                    0.00000000e00,
                    8.51037151e00,
                    -3.26500076e-01,
                    -4.26368998e00,
                    0.00000000e00,
                    -1.73618955e00,
                    -2.70298267e00,
                    0.00000000e00,
                    -2.43800268e00,
                    4.33837458e00,
                    9.23391435e-01,
                    -7.11570185e-01,
                    -1.07767560e01,
                    1.16849067e01,
                    -9.00945920e00,
                    0.00000000e00,
                    1.10312730e-01,
                    -9.26525263e-02,
                    -3.58767133e-01,
                    1.32449931e-01,
                    0.00000000e00,
                    -2.67409758e-01,
                    9.34745733e-02,
                    1.98748907e-01,
                    -1.26535756e00,
                    2.35853706e-02,
                    4.38376559e-02,
                    -8.00092020e-02,
                    -2.24001329e-01,
                    2.54625771e-02,
                    -6.31102720e-01,
                    1.85562538e-01,
                    1.65676987e-01,
                    1.31754143e-01,
                    1.59439453e-01,
                    0.00000000e00,
                    -8.60517307e-01,
                    1.40166940e00,
                    -8.88715952e-02,
                    1.65764015e-01,
                    2.65134765e-01,
                    -6.40839325e-02,
                    1.83074204e-01,
                    1.03160813e-01,
                    1.58498835e-01,
                    4.10177221e00,
                    0.00000000e00,
                    -4.45581695e-01,
                    -2.31490987e-02,
                    1.78988174e-01,
                    1.20348890e00,
                    -2.53742995e-01,
                    -2.73649103e-01,
                    2.07396369e-01,
                    1.33922151e-01,
                    -1.48365256e-01,
                    4.77971694e-01,
                    4.39481522e-01,
                    0.00000000e00,
                    1.89639994e-01,
                    0.00000000e00,
                    1.13688951e-01,
                    2.49369716e-01,
                    -1.22585243e-01,
                    0.00000000e00,
                    -9.74128321e-02,
                    -4.98143303e-01,
                    0.00000000e00,
                    -1.19853781e-01,
                    -5.48244925e-02,
                    4.83550293e-02,
                    -3.38405554e-02,
                    8.04424864e-01,
                    -7.33133226e-01,
                    5.32136453e-01,
                ]
            ),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(
            lam,
            np.array(
                [
                    2.21073262e-02,
                    3.79563141e-03,
                    6.78414676e-01,
                    2.50485765e02,
                    3.11881227e01,
                    2.13019441e02,
                ]
            ),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml, ndigits=3) == -32236.023

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk, ndigits=3) == -31972.739

    def test_print_smooth(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.model.print_smooth_terms()
        capture = capture.getvalue()

        comp = "f(['time', 'x']); edf: 16.131\nrs(['fact', 'sub']); edf: 38.089\nrs(['x', 'sub'],by=fact):fact_1; edf: 12.887\nrs(['x', 'sub'],by=fact):fact_2; edf: 14.136\nrs(['x', 'sub'],by=fact):fact_3; edf: 10.358\n"
        assert comp == capture


class Test_te_rs_fact_QR:
    # te + random slope with factor
    sim_dat, _ = sim1(100, random_seed=100)

    # Specify formula
    formula = Formula(
        lhs("y"),
        [
            i(),
            f(["time", "x"], te=True, nk=5),
            rs(["fact", "sub"]),
            rs(["x", "sub"], by="fact"),
        ],
        data=sim_dat,
    )

    # ... and model
    model = GAMM(formula, Gaussian())

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["max_outer"] = 100
    test_kwargs["method"] = "QR"

    model.fit(**test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf, ndigits=3) == 92.601

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma, ndigits=3) == 47.63

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(
            coef.flatten(),
            np.array(
                [
                    -8.30300915e-01,
                    -1.62575662e01,
                    3.75545293e00,
                    3.28335584e01,
                    -1.03918612e02,
                    3.44105191e01,
                    5.27629918e00,
                    1.56969059e01,
                    1.26673363e01,
                    -1.00123881e02,
                    6.70143258e01,
                    3.86562330e00,
                    1.95649156e01,
                    4.86327309e00,
                    -7.25159919e01,
                    1.30742924e01,
                    1.95090107e00,
                    9.28040152e00,
                    7.11371053e00,
                    -4.42679942e01,
                    -1.76804150e01,
                    4.78613869e01,
                    -1.98052935e01,
                    -3.54456054e01,
                    -1.83416110e01,
                    0.00000000e00,
                    8.09245518e-01,
                    -1.88785044e00,
                    7.86729219e00,
                    2.24203537e00,
                    0.00000000e00,
                    1.63941579e01,
                    1.90460014e00,
                    4.75391046e00,
                    7.87148618e00,
                    5.89124427e00,
                    -2.05573658e00,
                    1.17824896e01,
                    5.67949983e00,
                    5.60320399e-01,
                    7.16489667e00,
                    4.25921467e00,
                    6.32718677e00,
                    2.33817316e00,
                    4.63554642e00,
                    0.00000000e00,
                    4.33279542e00,
                    -1.54508599e01,
                    -4.34825704e-01,
                    -1.31251334e01,
                    -8.82479024e00,
                    -2.04672844e00,
                    -6.63234939e00,
                    6.01391323e-01,
                    -3.45057541e00,
                    -2.36964763e01,
                    0.00000000e00,
                    -1.27173425e00,
                    -1.07503433e-01,
                    2.33531411e00,
                    -7.66907848e00,
                    -6.74989624e-01,
                    -1.78519227e00,
                    -1.07199785e01,
                    7.97691278e-01,
                    2.06075903e00,
                    6.57708624e00,
                    -6.56854566e00,
                    0.00000000e00,
                    5.72411711e00,
                    0.00000000e00,
                    8.51037151e00,
                    -3.26500076e-01,
                    -4.26368998e00,
                    0.00000000e00,
                    -1.73618955e00,
                    -2.70298267e00,
                    0.00000000e00,
                    -2.43800268e00,
                    4.33837458e00,
                    9.23391435e-01,
                    -7.11570185e-01,
                    -1.07767560e01,
                    1.16849067e01,
                    -9.00945920e00,
                    0.00000000e00,
                    1.10312730e-01,
                    -9.26525263e-02,
                    -3.58767133e-01,
                    1.32449931e-01,
                    0.00000000e00,
                    -2.67409758e-01,
                    9.34745733e-02,
                    1.98748907e-01,
                    -1.26535756e00,
                    2.35853706e-02,
                    4.38376559e-02,
                    -8.00092020e-02,
                    -2.24001329e-01,
                    2.54625771e-02,
                    -6.31102720e-01,
                    1.85562538e-01,
                    1.65676987e-01,
                    1.31754143e-01,
                    1.59439453e-01,
                    0.00000000e00,
                    -8.60517307e-01,
                    1.40166940e00,
                    -8.88715952e-02,
                    1.65764015e-01,
                    2.65134765e-01,
                    -6.40839325e-02,
                    1.83074204e-01,
                    1.03160813e-01,
                    1.58498835e-01,
                    4.10177221e00,
                    0.00000000e00,
                    -4.45581695e-01,
                    -2.31490987e-02,
                    1.78988174e-01,
                    1.20348890e00,
                    -2.53742995e-01,
                    -2.73649103e-01,
                    2.07396369e-01,
                    1.33922151e-01,
                    -1.48365256e-01,
                    4.77971694e-01,
                    4.39481522e-01,
                    0.00000000e00,
                    1.89639994e-01,
                    0.00000000e00,
                    1.13688951e-01,
                    2.49369716e-01,
                    -1.22585243e-01,
                    0.00000000e00,
                    -9.74128321e-02,
                    -4.98143303e-01,
                    0.00000000e00,
                    -1.19853781e-01,
                    -5.48244925e-02,
                    4.83550293e-02,
                    -3.38405554e-02,
                    8.04424864e-01,
                    -7.33133226e-01,
                    5.32136453e-01,
                ]
            ),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(
            lam,
            np.array(
                [
                    2.21073262e-02,
                    3.79563141e-03,
                    6.78414676e-01,
                    2.50485765e02,
                    3.11881227e01,
                    2.13019441e02,
                ]
            ),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml, ndigits=3) == -32236.023

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk, ndigits=3) == -31972.739


class Test_print_parametric:
    # te + random slope with factor
    sim_dat, _ = sim1(100, random_seed=100)

    # Specify formula
    formula = Formula(
        lhs("y"),
        [
            i(),
            l(["fact"]),
            f(["time", "x"], te=True, nk=5),
            rs(["fact", "sub"]),
            rs(["x", "sub"], by="fact"),
        ],
        data=sim_dat,
    )

    # ... and model
    model = GAMM(formula, Gaussian())

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["max_outer"] = 100

    model.fit(**test_kwargs)

    def test_print_parametric(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.model.print_parametric_terms()
        capture = capture.getvalue()

        comp = "Intercept: 4.992, t: 2.609, DoF.: 9409, P(|T| > |t|): 0.00911 **\nfact_fact_2: -13.855, t: -5.263, DoF.: 9409, P(|T| > |t|): 1.445e-07 ***\nfact_fact_3: -6.252, t: -2.395, DoF.: 9409, P(|T| > |t|): 0.01664 *\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: .\n"
        assert comp == capture


class Test_ti_rs_fact:
    # ti + random slope with factor
    sim_dat, _ = sim1(100, random_seed=100)

    # Specify formula
    formula = Formula(
        lhs("y"),
        [
            i(),
            f(["x"]),
            f(["time"]),
            f(["time", "x"], te=False, nk=5),
            rs(["fact", "sub"]),
            rs(["x", "sub"], by="fact"),
        ],
        data=sim_dat,
    )

    # ... and model
    model = GAMM(formula, Gaussian())

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["max_outer"] = 100

    model.fit(**test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf, ndigits=3) == 103.555

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma, ndigits=3) == 46.601

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(
            coef.flatten(),
            np.array(
                [
                    -9.71704143e-01,
                    1.54721686e00,
                    6.37094590e00,
                    -1.81024530e00,
                    1.49129313e00,
                    7.76170446e00,
                    -1.96237580e00,
                    -1.36937262e00,
                    -9.83761681e00,
                    -4.32302196e01,
                    9.49189307e00,
                    8.32820414e00,
                    1.03243783e01,
                    5.21269242e00,
                    6.91971180e00,
                    5.49551079e00,
                    -1.27418118e00,
                    5.40282355e00,
                    2.11740677e01,
                    -7.19128186e00,
                    2.80723247e01,
                    4.53116540e01,
                    2.93637653e01,
                    -4.81976007e00,
                    -1.04991125e01,
                    5.72442097e-01,
                    8.45846333e00,
                    2.17018141e00,
                    8.67020948e00,
                    -1.50099944e01,
                    5.94560673e00,
                    1.76938358e01,
                    1.10900870e01,
                    5.22591490e00,
                    1.50258617e01,
                    1.97543071e01,
                    3.41527520e00,
                    6.79301296e00,
                    1.21044533e01,
                    -2.43559622e01,
                    1.55643199e01,
                    4.39239465e01,
                    6.66970475e01,
                    8.69091468e01,
                    0.00000000e00,
                    -3.14951990e-01,
                    -1.10905894e00,
                    8.77585189e00,
                    2.26658681e00,
                    0.00000000e00,
                    1.40839030e01,
                    3.06020246e00,
                    4.07727301e00,
                    9.96448961e00,
                    7.67315584e00,
                    -2.26262888e00,
                    1.26004973e01,
                    6.61513178e00,
                    7.59755858e-01,
                    -3.60914249e00,
                    3.07808480e00,
                    6.00402073e00,
                    2.77619325e00,
                    6.88612689e00,
                    0.00000000e00,
                    7.36665376e00,
                    -1.93778847e01,
                    -1.11035288e-01,
                    -2.30469258e01,
                    -6.93658165e00,
                    4.38712596e00,
                    -3.04300412e00,
                    1.12959958e00,
                    -1.99702268e00,
                    -2.73114905e01,
                    0.00000000e00,
                    -1.21036113e00,
                    3.41254393e-02,
                    9.53244574e-01,
                    -5.57686089e00,
                    -4.63828161e-01,
                    -1.10994608e00,
                    -7.49500155e00,
                    1.34600035e00,
                    2.12919578e-01,
                    8.59037174e00,
                    -2.10410712e00,
                    0.00000000e00,
                    6.36640794e00,
                    0.00000000e00,
                    7.06794114e00,
                    -1.11517758e-01,
                    -6.10316341e00,
                    0.00000000e00,
                    -1.35803897e00,
                    -2.26213977e00,
                    0.00000000e00,
                    -4.08129879e00,
                    7.74754710e00,
                    1.17088060e00,
                    -2.52598120e00,
                    -6.72366968e00,
                    5.88345533e00,
                    -1.06319047e01,
                    0.00000000e00,
                    7.44618614e-02,
                    -4.60409879e-02,
                    -4.20709238e-01,
                    1.13261409e-01,
                    0.00000000e00,
                    -1.06900997e-01,
                    1.27039907e-01,
                    1.44186258e-01,
                    -1.35495627e00,
                    -1.93898758e-02,
                    1.05246077e-01,
                    -8.52466119e-02,
                    -2.12131368e-01,
                    2.92038635e-02,
                    -2.02115234e-02,
                    2.97284552e-01,
                    2.60330919e-01,
                    1.32323699e-01,
                    4.22244844e-02,
                    0.00000000e00,
                    -1.11968011e00,
                    1.97630129e00,
                    -2.57003864e-02,
                    7.03090968e-01,
                    1.97811256e-01,
                    -7.66132166e-01,
                    -2.64846408e-01,
                    2.19438580e-01,
                    1.07095112e-01,
                    4.67992499e00,
                    0.00000000e00,
                    -4.80260428e-01,
                    8.32186882e-03,
                    8.27397567e-02,
                    7.34822493e-02,
                    -1.97462641e-01,
                    -1.92682279e-01,
                    -1.66172481e-02,
                    2.55913742e-01,
                    -4.27052636e-02,
                    3.73577224e-01,
                    -4.21838602e-02,
                    0.00000000e00,
                    1.26216368e-01,
                    0.00000000e00,
                    5.65018226e-02,
                    2.50526369e-01,
                    -1.51895786e-01,
                    0.00000000e00,
                    -4.55964528e-02,
                    -4.42880493e-01,
                    0.00000000e00,
                    -1.20064811e-01,
                    -3.61032864e-01,
                    3.66917318e-02,
                    -7.18868817e-02,
                    5.92954524e-01,
                    -3.56891388e-01,
                    5.75216981e-01,
                ]
            ),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(
            lam,
            np.array(
                [
                    2.06462523e-01,
                    2.61797067e-01,
                    4.54850831e-03,
                    7.59911476e-02,
                    5.72234072e-01,
                    2.49782036e02,
                    2.32293389e01,
                    3.00260310e02,
                ]
            ),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml, ndigits=3) == -32166.474

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk, ndigits=3) == -31862.875


class Test_3way_li:
    # *li() with three variables: three-way interaction
    sim_dat, _ = sim1(100, random_seed=100)

    # Specify formula
    formula = Formula(lhs("y"), [i(), *li(["fact", "x", "time"])], data=sim_dat)

    # ... and model
    model = GAMM(formula, Gaussian())

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["max_inner"] = 1

    # then fit
    model.fit(**test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf, ndigits=3) == 12

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma, ndigits=3) == 67.004

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(
            coef.flatten(),
            np.array(
                [
                    9.49234564e00,
                    -1.11906237e01,
                    -8.23135271e00,
                    -3.19215668e-01,
                    -3.19669863e-04,
                    -1.05685883e-01,
                    2.18785853e-01,
                    8.56147251e-04,
                    -7.20358812e-04,
                    -7.44800816e-05,
                    1.88303305e-04,
                    8.13699273e-05,
                ]
            ),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(lam, np.array([]))


class Test_print_smooth_by_factor_p:
    # by factor
    sim_dat, _ = sim1(100, random_seed=100)

    # Specify formula
    formula = Formula(
        lhs("y"), [i(), l(["fact"]), f(["time"], nk=10, by="fact")], data=sim_dat
    )

    # ... and model
    model = GAMM(formula, Gaussian())

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["max_outer"] = 100

    model.fit(**test_kwargs)

    def test_print_smooth_p(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.model.print_smooth_terms(p_values=True)
        capture = capture.getvalue()

        comp = "f(['time'],by=fact): fact_1; edf: 9.43 f: 60.057 P(F > f) = 0.000e+00 ***\nf(['time'],by=fact): fact_2; edf: 7.607 f: 16.831 P(F > f) = 0.000e+00 ***\nf(['time'],by=fact): fact_3; edf: 4.839 f: 10.948 P(F > f) = 0.000e+00 ***\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n"
        assert comp == capture


class Test_print_smooth_by_factor_fs_p:
    # by factor
    sim_dat, _ = sim1(100, random_seed=100)

    # Specify formula
    formula = Formula(
        lhs("y"),
        [i(), l(["fact"]), f(["time"], nk=10, by="fact"), fs(["time"], rf="sub")],
        data=sim_dat,
    )

    # ... and model
    model = GAMM(formula, Gaussian())

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["max_outer"] = 100

    model.fit(**test_kwargs)

    def test_print_smooth_p(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.model.print_smooth_terms(p_values=True)
        capture = capture.getvalue()

        comp = "f(['time'],by=fact): fact_1; edf: 9.493 f: 34.404 P(F > f) = 0.000e+00 ***\nf(['time'],by=fact): fact_2; edf: 7.866 f: 8.889 P(F > f) = 0.000e+00 ***\nf(['time'],by=fact): fact_3; edf: 5.295 f: 3.238 P(F > f) = 0.00449 **\nf(['time'],by=sub); edf: 94.041\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n"
        assert comp == capture


class Test_print_smooth_binomial:
    Binomdat = sim3(10000, 0.1, family=Binomial(), seed=20)

    formula = Formula(
        lhs("y"), [i(), f(["x0"]), f(["x1"]), f(["x2"]), f(["x3"])], data=Binomdat
    )

    # By default, the Binomial family assumes binary data and uses the logit link.
    model = GAMM(formula, Binomial())

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["max_inner"] = 1
    test_kwargs["control_lambda"] = 2

    model.fit(**test_kwargs)

    def test_print_smooth_p(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.model.print_smooth_terms(p_values=True)
        capture = capture.getvalue()

        comp = "f(['x0']); edf: 2.856 chi^2: 18.417 P(Chi^2 > chi^2) = 7.220e-04 ***\nf(['x1']); edf: 1.962 chi^2: 59.723 P(Chi^2 > chi^2) = 0.000e+00 ***\nf(['x2']); edf: 6.243 chi^2: 168.267 P(Chi^2 > chi^2) = 0.000e+00 ***\nf(['x3']); edf: 1.407 chi^2: 2.731 P(Chi^2 > chi^2) = 0.29779\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n"
        assert comp == capture


class Test_Vp_estimation_hard:
    Binomdat = sim3(10000, 0.1, family=Binomial(), seed=20)

    formula = Formula(
        lhs("y"), [i(), f(["x0"]), f(["x1"]), f(["x2"]), f(["x3"])], data=Binomdat
    )

    # By default, the Binomial family assumes binary data and uses the logit link.
    model = GAMM(formula, Binomial())

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["max_inner"] = 1
    test_kwargs["control_lambda"] = 2

    model.fit(**test_kwargs)

    Vp, _, _, _, _, _ = estimateVp(model, strategy="JJJ1", Vp_fidiff=True)

    def test_Vp(self):
        np.testing.assert_allclose(
            np.round(self.Vp, decimals=3),
            np.array(
                [
                    [2.2810e00, 9.0000e-03, 3.0000e-03, 2.1000e-02],
                    [9.0000e-03, 2.7780e00, 1.0000e-03, -2.5000e-02],
                    [3.0000e-03, 1.0000e-03, 4.9400e-01, 2.2000e-02],
                    [2.1000e-02, -2.5000e-02, 2.2000e-02, 1.5413e01],
                ]
            ),
            atol=min(max_atol, 0.001),
        )


class Test_te_p_values:
    sim_dat = sim3(n=500, scale=2, c=0, seed=20)

    formula = Formula(
        lhs("y"),
        [i(), f(["x0", "x3"], te=True, nk=9), f(["x1"]), f(["x2"])],
        data=sim_dat,
    )
    model = GAMM(formula, Gaussian())

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["max_outer"] = 100

    model.fit(**test_kwargs)

    ps, Trs = approx_smooth_p_values(model, par=0, edf1=False, force_approx=True)

    def test_p(self):
        np.testing.assert_allclose(
            self.ps,
            np.array(
                [np.float64(0.19051092087804067), np.float64(0.0), np.float64(0.0)]
            ),
            atol=min(max_atol, 0.06),
            rtol=min(max_rtol, 1e-6),
        )

    def test_trs(self):
        np.testing.assert_allclose(
            self.Trs,
            np.array(
                [
                    np.float64(1.4741483696834052),
                    np.float64(128.17369821317232),
                    np.float64(126.04637199015741),
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.02),
        )


class Test_diff:
    # pred_diff test
    sim_dat, _ = sim1(100, random_seed=100)

    # Specify formula
    formula = Formula(
        lhs("y"),
        [
            i(),
            l(["fact"]),
            f(["time"], by="fact"),
            rs(["fact", "sub"]),
            rs(["x", "sub"], by="fact"),
        ],
        data=sim_dat,
    )

    # ... and model
    model = GAMM(formula, Gaussian())

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["max_outer"] = 100

    model.fit(**test_kwargs)

    pred_dat1 = pd.DataFrame(
        {
            "time": np.linspace(min(sim_dat["time"]), max(sim_dat["time"]), 50),
            "x": [0 for _ in range(50)],
            "fact": ["fact_1" for _ in range(50)],
            "sub": ["sub_0" for _ in range(50)],
        }
    )

    pred_dat2 = pd.DataFrame(
        {
            "time": np.linspace(min(sim_dat["time"]), max(sim_dat["time"]), 50),
            "x": [0 for _ in range(50)],
            "fact": ["fact_2" for _ in range(50)],
            "sub": ["sub_0" for _ in range(50)],
        }
    )

    diff, ci = model.predict_diff(pred_dat1, pred_dat2, [0, 1, 2])

    def test_diff(self):
        assert np.allclose(
            self.diff,
            np.array(
                [
                    17.20756874,
                    18.56006769,
                    19.3407868,
                    19.65067738,
                    19.59069076,
                    19.26177827,
                    18.76489122,
                    18.20098095,
                    17.64796193,
                    17.0916012,
                    16.49462896,
                    15.81977541,
                    15.02977076,
                    14.08734519,
                    12.95522891,
                    11.62544874,
                    10.20721807,
                    8.83904688,
                    7.6594452,
                    6.80692301,
                    6.41999034,
                    6.63715718,
                    7.53833267,
                    8.96902249,
                    10.71613147,
                    12.56656442,
                    14.30722615,
                    15.72502149,
                    16.60685525,
                    16.80334424,
                    16.41995315,
                    15.62585868,
                    14.59023753,
                    13.48226638,
                    12.47112193,
                    11.72598086,
                    11.3696111,
                    11.33914548,
                    11.52530807,
                    11.81882293,
                    12.11041412,
                    12.2908057,
                    12.25072176,
                    11.94170808,
                    11.55859751,
                    11.35704462,
                    11.59270397,
                    12.52123016,
                    14.39827774,
                    17.4795013,
                ]
            ),
        )

    def test_ci(self):
        assert np.allclose(
            self.ci,
            np.array(
                [
                    5.51585458,
                    5.30307541,
                    5.27423164,
                    5.27636045,
                    5.26677133,
                    5.25310522,
                    5.24996695,
                    5.25675651,
                    5.2592083,
                    5.25287576,
                    5.24373586,
                    5.23907861,
                    5.24219173,
                    5.25034428,
                    5.25617479,
                    5.25356667,
                    5.24600153,
                    5.24043816,
                    5.24155349,
                    5.24930369,
                    5.25886655,
                    5.26296068,
                    5.25755966,
                    5.24855614,
                    5.24400619,
                    5.24829547,
                    5.26007102,
                    5.27286368,
                    5.27834216,
                    5.27329519,
                    5.26651469,
                    5.26805321,
                    5.2818876,
                    5.30386531,
                    5.32358128,
                    5.32994662,
                    5.32191458,
                    5.31645209,
                    5.32872033,
                    5.36073382,
                    5.4007834,
                    5.43114323,
                    5.44312329,
                    5.45849684,
                    5.51697233,
                    5.62549037,
                    5.78645176,
                    6.09575666,
                    6.8585062,
                    8.57200825,
                ]
            ),
        )


class Test_shared:
    sim_fit_dat = sim3(n=500, scale=2, c=0.0, family=Gamma(), seed=1)

    # Now fit nested models
    sim_fit_formula = Formula(
        lhs("y"),
        [i(), f(["x0"], id=1), f(["x1"]), f(["x2"]), f(["x3"], id=1)],
        data=sim_fit_dat,
        print_warn=False,
    )

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["method"] = "QR"
    test_kwargs["control_lambda"] = 2
    test_kwargs["max_inner"] = 500
    test_kwargs["extend_lambda"] = False

    model = GAMM(sim_fit_formula, Gamma())
    model.fit(**test_kwargs)

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.model.edf,
            14.474560341691095,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        np.testing.assert_allclose(
            sigma, 1.7122644814590584, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    [6.63115934],
                    [-0.0728171],
                    [-0.0202448],
                    [0.01181259],
                    [0.0380147],
                    [0.06274025],
                    [0.09110549],
                    [0.11721487],
                    [0.12369589],
                    [0.13045535],
                    [-1.89545643],
                    [-1.02998279],
                    [-0.50963573],
                    [0.01338543],
                    [0.8723655],
                    [2.10927161],
                    [3.52039551],
                    [4.40419881],
                    [5.22845418],
                    [-7.05710007],
                    [3.770508],
                    [6.38517406],
                    [-1.7808684],
                    [-0.47531908],
                    [0.41080258],
                    [-5.39519788],
                    [-1.57547179],
                    [-8.82661351],
                    [-0.09628179],
                    [-0.02911359],
                    [0.0087516],
                    [0.05074923],
                    [0.09477548],
                    [0.12442737],
                    [0.152579],
                    [0.16546373],
                    [0.17582209],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.5),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        np.testing.assert_allclose(
            lam,
            np.array([14.629935583511205, 0.013963952614761084, 274960.77985164756]),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 2.5),
        )

    def test_GAMreml(self):
        reml = self.model.get_reml()
        np.testing.assert_allclose(
            reml, -3735.1471784723763, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        np.testing.assert_allclose(
            llk, -3704.806458939361, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_edf1(self):
        compute_bias_corrected_edf(self.model, overwrite=False)
        edf1 = np.array([edf1 for edf1 in self.model.term_edf1])
        np.testing.assert_allclose(
            edf1,
            np.array(
                [
                    1.0026619803714023,
                    3.9345191984495687,
                    8.730848583106779,
                    1.0027681370571946,
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
            np.array([0.3389291306383489, 0.0, 0.0, 0.19899991251845162]),
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
                    0.9194104426252923,
                    263.96555423099835,
                    238.8429544293655,
                    1.6554767488819178,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 1.5),
        )
