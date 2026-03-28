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
from mssm.src.python.formula import build_model_matrix

mssm.src.python.exp_fam.GAUMLSS.init_lambda = init_penalties_tests_gammlss
mssm.src.python.exp_fam.GAMMALS.init_lambda = init_penalties_tests_gammlss
mssm.src.python.exp_fam.MULNOMLSS.init_lambda = init_penalties_tests_gammlss
mssm.src.python.exp_fam.PropHaz.init_lambda = init_penalties_tests_gsmm
mssm.src.python.utils.GAMLSSGSMMFamily.init_lambda = init_penalties_tests_gsmm


class Test_MultivariateHSMM_hard:

    # Start by simulating some data
    np_gen = np.random.default_rng(0)
    n_series = 3
    n_obs = 250
    starts_with_first = False

    # Define True model
    pi = np.array([0.6, 0.2, 0.2])

    T = np.array([[0.0, 0.6, 0.4], [0.3, 0.0, 0.7], [0.6, 0.4, 0.0]])

    mus = [np.array([0.0, 2.5, 5.0]), np.array([-2, 0, -5]), np.array([0, 0, 0])]

    sigmas = [
        np.array([[1, -0.3, 0.5], [-0.3, 1, 0.0], [0.5, 0.0, 1]]),
        np.array([[1, 0.3, -0.5], [0.3, 1, 0.3], [-0.5, 0.3, 1]]),
        np.array([[1, 0.0, 0.2], [0.0, 1, -0.6], [0.2, -0.6, 1]]),
    ]

    # Duration distribution parameters
    mus2 = [17, 12, 7]
    scales = [0.05, 0.05, 0.05]
    mus2_init = [2, 4, 6]
    scales_init = scales

    y_all = [[], [], []]
    states_all = []
    time_all = []
    series_all = []
    durs_all = []
    seed = 0
    for s in range(n_series):

        state = np_gen.choice(np.arange(3), p=pi)

        if starts_with_first:
            alpha = 1 / scales[state]
            beta = alpha / mus2[state]
        else:
            alpha = 1 / scales_init[state]
            beta = alpha / mus2_init[state]

        # duration samples
        dur = int(
            scp.stats.gamma.rvs(a=alpha, scale=(1 / beta), size=1, random_state=seed)[0]
        )
        seed += 1

        y = [[], [], []]
        states = []
        durs = []
        t = 0

        while len(y[0]) < n_obs:
            Y = scp.stats.multivariate_normal.rvs(
                size=dur, mean=mus[state], cov=sigmas[state], random_state=seed
            )
            Y = Y.reshape(dur, 3)
            for m in range(3):
                y[m].extend(Y[:, m])

            for _ in range(dur):
                states.append(state)
            durs.append([state, dur])

            prev_state = state
            state = np_gen.choice(np.arange(3), p=T[state, :])

            alpha = 1 / scales[state]
            beta = alpha / mus2[state]

            # duration samples
            dur = int(
                scp.stats.gamma.rvs(
                    a=alpha, scale=(1 / beta), size=1, random_state=seed
                )[0]
            )
            t += 1
            seed += 1

        for m in range(3):
            y[m] = y[m][:n_obs]
        states = states[:n_obs]
        time = np.arange(n_obs)
        for m in range(3):
            y_all[m].extend(y[m])
        time_all.extend(time)

        for _ in range(n_obs):
            series_all.append(s)

        states_all.append(states)
        durs_all.append(durs)

    dat = pd.DataFrame(
        {
            "y0": y_all[0],
            "y1": y_all[1],
            "y2": y_all[2],
            "time": time_all,
            "series": series_all,
        }
    )

    # Create pseudo-y column for time-varying HsMM
    dat["pseudo"] = [1 for _ in range(dat.shape[0])]

    # Prep dat for hsmm model
    uval, id = np.unique(np.array(dat["series"]), return_index=True)
    sid = np.sort(id)
    tid = np.arange(len(id))

    # Now define models
    o_family = MultiGauss(3, [Identity() for _ in range(3)])
    d_family = GAMMALS(links=[LOG(), LOGb(-0.0001)])
    n_S = 3
    M = 3
    D = 100
    starts_with_first = True

    # Initial parameters
    Y = np.concatenate(
        [
            dat["y0"].values.reshape(-1, 1),
            dat["y1"].values.reshape(-1, 1),
            dat["y2"].values.reshape(-1, 1),
        ],
        axis=1,
    )
    init_size = 50
    init_choice = np_gen.choice(len(dat), size=init_size * n_S, replace=False)
    int_samples = []
    init_mus = []
    init_Rs = []

    for j in range(n_S):
        int_samples.append(Y[init_choice[j * init_size : (j + 1) * init_size]])
        init_mus.append(int_samples[j].mean(axis=0))

        cov = np.cov(int_samples[j].T)
        prec = np.linalg.inv(cov)
        R = np.linalg.cholesky(prec).T
        for m in range(M):
            R[m, m] = np.log(R[m, m])

        init_Rs.extend(R[np.triu_indices(M)])

    # Set up formulas
    ys = []
    Xs = []
    form_n_coef = []
    init_coef = []
    build_mat_idx = []
    build_matrix = []

    obs_families = []
    obs_formulas = []
    links = []
    pars = 0
    extra_coef = 0
    for j in range(n_S):
        obs_families.append([])

        obs_families[-1].append(o_family)
        extra_coef += o_family.extra_coef

        for m in range(3):
            # Model of mean of obs model in state j and of signal m
            obs_formulas.append(Formula(lhs(f"y{m}"), [i()], data=dat))

            init_coef.append(init_mus[j][m])
            form_n_coef.append(obs_formulas[-1].n_coef)
            _ = build_penalties(obs_formulas[-1])

            if j == 0 and m == 0:
                build_matrix.append(True)
                Xs.append(build_model_matrix(obs_formulas[-1]))
            else:
                build_matrix.append(False)
                Xs.append(None)

            build_mat_idx.append(0)
            ys.append(dat[f"y{m}"].values.reshape(-1, 1))

        links.extend(o_family.links)
        pars += 3

    d_dat = pd.DataFrame(
        {"x": [1 for _ in range(len(tid))], "pseudo": [1 for _ in range(len(tid))]}
    )

    d_families = []
    d_formulas = []

    # For time-varying model we need to split here
    d_formulas_tv = []
    tv_Xs = copy.deepcopy(Xs)
    tv_ys = copy.deepcopy(ys)

    for j in range(n_S * (1 if starts_with_first else 2)):
        d_families.append(d_family)

        # Model of mean of dur model in state j
        d_formulas.append(Formula(lhs("pseudo"), [i()], data=d_dat))
        d_formulas_tv.append(Formula(lhs("pseudo"), [i()], data=dat))

        init_coef.append(2.5)
        form_n_coef.append(d_formulas[-1].n_coef)
        _ = build_penalties(d_formulas[-1])
        _ = build_penalties(d_formulas_tv[-1])
        Xs.append(build_model_matrix(d_formulas[-1]))
        ys.append(d_dat["pseudo"].values.reshape(-1, 1))
        tv_Xs.append(build_model_matrix(d_formulas_tv[-1]))
        tv_ys.append(dat["pseudo"].values.reshape(-1, 1))
        build_matrix.append(True)
        build_mat_idx.append(len(build_mat_idx))

        # Model of scale parameter of dur model in state j
        d_formulas.append(Formula(lhs("pseudo"), [i()], data=d_dat))
        d_formulas_tv.append(Formula(lhs("pseudo"), [i()], data=dat))

        init_coef.append(-2)
        form_n_coef.append(d_formulas[-1].n_coef)
        _ = build_penalties(d_formulas[-1])
        _ = build_penalties(d_formulas_tv[-1])
        Xs.append(build_model_matrix(d_formulas[-1]))
        ys.append(d_dat["pseudo"].values.reshape(-1, 1))
        tv_Xs.append(build_model_matrix(d_formulas_tv[-1]))
        tv_ys.append(dat["pseudo"].values.reshape(-1, 1))
        build_matrix.append(True)
        build_mat_idx.append(len(build_mat_idx))

        links.extend(d_family.links)
        pars += 2

    # For hsmm with fixed state transitions and pi we have everything now,
    # so collect copies:
    od_init_coef = copy.deepcopy(init_coef)
    od_form_n_coef = copy.deepcopy(form_n_coef)
    od_build_matrix = copy.deepcopy(build_matrix)
    od_build_mat_idx = copy.deepcopy(build_mat_idx)
    od_pars = pars
    od_links = copy.deepcopy(links)
    od_ys = copy.deepcopy(ys)
    od_Xs = copy.deepcopy(Xs)

    t_formulas = []
    t_formulas_tv = []
    for j in range(n_S):
        for par in range(n_S - 2):

            # Model of state transition away from state j
            t_formulas.append(Formula(lhs("pseudo"), [i()], data=d_dat))
            t_formulas_tv.append(Formula(lhs("pseudo"), [i()], data=dat))

            init_coef.append(np.log(1))
            form_n_coef.append(t_formulas[-1].n_coef)
            _ = build_penalties(t_formulas[-1])
            _ = build_penalties(t_formulas_tv[-1])
            Xs.append(build_model_matrix(t_formulas[-1]))
            ys.append(d_dat["pseudo"].values.reshape(-1, 1))
            tv_Xs.append(build_model_matrix(t_formulas_tv[-1]))
            tv_ys.append(dat["pseudo"].values.reshape(-1, 1))
            build_matrix.append(True)
            build_mat_idx.append(len(build_mat_idx))
            pars += 1

    pi_formulas = []
    pi_formulas_tv = []
    for par in range(n_S - 1):
        # Model of initial state probability
        pi_formulas.append(Formula(lhs("pseudo"), [i()], data=d_dat))
        pi_formulas_tv.append(Formula(lhs("pseudo"), [i()], data=dat))

        init_coef.append(np.log(1))
        form_n_coef.append(pi_formulas[-1].n_coef)
        _ = build_penalties(pi_formulas[-1])
        _ = build_penalties(pi_formulas_tv[-1])
        Xs.append(build_model_matrix(pi_formulas[-1]))
        ys.append(d_dat["pseudo"].values.reshape(-1, 1))
        tv_Xs.append(build_model_matrix(pi_formulas_tv[-1]))
        tv_ys.append(dat["pseudo"].values.reshape(-1, 1))
        build_matrix.append(True)
        build_mat_idx.append(len(build_mat_idx))
        pars += 1

    # Initialize Families
    fam1 = HSMMFamily(
        pars,
        links,
        n_S,
        obs_fams=obs_families,
        d_fams=d_families,
        sid=sid,
        tid=tid,
        D=D,
        M=M,
        starts_with_first=starts_with_first,
        ends_with_last=False,
        ends_in_last=False,
        n_cores=1,
        build_mat_idx=build_mat_idx,
    )

    # Time-varying version
    fam2 = HSMMFamily(
        pars,
        links,
        n_S,
        obs_fams=obs_families,
        d_fams=d_families,
        sid=sid,
        tid=None,  # Can set to None since it's same as sid
        D=D,
        M=M,
        starts_with_first=starts_with_first,
        ends_with_last=False,
        ends_in_last=False,
        n_cores=1,
        build_mat_idx=build_mat_idx,
    )

    # Fixed pi and T
    fam3 = HSMMFamily(
        od_pars,
        od_links,
        n_S,
        obs_fams=obs_families,
        d_fams=d_families,
        sid=sid,
        tid=tid,
        D=D,
        M=M,
        starts_with_first=starts_with_first,
        ends_with_last=False,
        ends_in_last=False,
        T=T,
        pi=pi,
        n_cores=1,
        build_mat_idx=od_build_mat_idx,
    )

    # Initialize penalties and coef

    # Extra coef for covariances
    od_form_n_coef.append(extra_coef)
    form_n_coef.append(extra_coef)
    od_init_coef.extend(init_Rs)
    init_coef.extend(init_Rs)

    coef_split_idx = form_n_coef[:-1]

    for coef_i in range(1, len(coef_split_idx)):
        coef_split_idx[coef_i] += coef_split_idx[coef_i - 1]

    od_coef_split_idx = od_form_n_coef[:-1]

    for coef_i in range(1, len(od_coef_split_idx)):
        od_coef_split_idx[coef_i] += od_coef_split_idx[coef_i - 1]

    total_coef = np.sum(form_n_coef)
    od_total_coef = np.sum(od_form_n_coef)
    init_coef = np.array(init_coef).reshape(-1, 1)
    od_init_coef = np.array(od_init_coef).reshape(-1, 1)

    def test_llk(self):
        # Test llk
        llk1 = self.fam1.llk(self.init_coef, self.coef_split_idx, self.ys, self.Xs)
        llk2 = self.fam2.llk(
            self.init_coef, self.coef_split_idx, self.tv_ys, self.tv_Xs
        )
        llk3 = self.fam3.llk(
            self.od_init_coef, self.od_coef_split_idx, self.od_ys, self.od_Xs
        )

        np.testing.assert_allclose(
            llk1,
            llk2,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            llk1,
            -2422.1207972824186,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            llk3,
            -2422.408060704047,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_grad(self):
        # Test grad function of hsmm and time-varying hsmm
        grad1 = self.fam1.gradient(
            self.init_coef, self.coef_split_idx, self.ys, self.Xs
        )
        grad2 = self.fam2.gradient(
            self.init_coef, self.coef_split_idx, self.tv_ys, self.tv_Xs
        )

        pos_llk_warp = lambda x: self.fam1.llk(
            x.reshape(-1, 1), self.coef_split_idx, self.ys, self.Xs
        )

        grad3 = scp.optimize.approx_fprime(self.init_coef.flatten(), pos_llk_warp)

        grad_diff1 = np.abs(np.round(grad1 - grad3.reshape(-1, 1), decimals=3)).max()
        grad_diff2 = np.abs(np.round(grad2 - grad3.reshape(-1, 1), decimals=3)).max()

        np.testing.assert_allclose(
            grad_diff1,
            0.0,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            grad_diff2,
            0.0,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_grad_fixed(self):
        # Test grad function of fixed hsmm
        grad4 = self.fam3.gradient(
            self.od_init_coef, self.od_coef_split_idx, self.od_ys, self.od_Xs
        )
        pos_llk_warp = lambda x: self.fam3.llk(
            x.reshape(-1, 1), self.od_coef_split_idx, self.od_ys, self.od_Xs
        )
        grad5 = scp.optimize.approx_fprime(self.od_init_coef.flatten(), pos_llk_warp)

        grad_diff3 = np.abs(np.round(grad4 - grad5.reshape(-1, 1), decimals=3)).max()

        np.testing.assert_allclose(
            grad_diff3,
            0.0,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_ods(self):
        # Check observation and duration probs
        ods1 = self.fam1.compute_od_probs(
            self.init_coef, self.coef_split_idx, self.ys, self.Xs
        )

        ods2 = self.fam2.compute_od_probs(
            self.init_coef, self.coef_split_idx, self.tv_ys, self.tv_Xs
        )

        ods3 = self.fam3.compute_od_probs(
            self.od_init_coef, self.od_coef_split_idx, self.od_ys, self.od_Xs
        )

        np.testing.assert_allclose(
            ods1[0][0],
            ods2[0][0],
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            ods1[0][1],
            ods2[0][1][:, :, 0],
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            ods3[0][0][:5, :],
            np.array(
                [
                    [-4.40202288, -3.8485654, -4.34099828],
                    [-3.8074954, -3.87171952, -3.6856842],
                    [-5.10074097, -4.5286156, -4.96309288],
                    [-3.10149621, -3.24096177, -3.15586644],
                    [-3.06516362, -2.61534322, -2.90384498],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            ods3[0][1][:5, :],
            np.array(
                [
                    [-11.61207055, -11.61207055, -11.61207055],
                    [-7.79337881, -7.79337881, -7.79337881],
                    [-5.81113444, -5.81113444, -5.81113444],
                    [-4.58076989, -4.58076989, -4.58076989],
                    [-3.76239347, -3.76239347, -3.76239347],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_tps(self):
        # Check Ts and ps
        Tps1 = self.fam1.compute_Tpi(
            self.init_coef, self.coef_split_idx, self.ys, self.Xs
        )

        Tps2 = self.fam2.compute_Tpi(
            self.init_coef, self.coef_split_idx, self.tv_ys, self.tv_Xs
        )

        Tps3 = self.fam3.compute_Tpi(
            self.od_init_coef, self.od_coef_split_idx, self.od_ys, self.od_Xs
        )

        np.testing.assert_allclose(
            Tps1[0][0],
            Tps2[0][0][:, :, 0],
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            Tps1[0][1],
            Tps2[0][1],
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            Tps3[0][0],
            self.T,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            Tps3[0][1],
            self.pi,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_viterbi(self):
        # Test viterbi
        viterbi1 = self.fam1.decode_viterbi(
            self.init_coef, self.coef_split_idx, self.ys, self.Xs
        )

        viterbi2 = self.fam2.decode_viterbi(
            self.init_coef, self.coef_split_idx, self.tv_ys, self.tv_Xs
        )

        viterbi3 = self.fam3.decode_viterbi(
            self.od_init_coef, self.od_coef_split_idx, self.od_ys, self.od_Xs
        )

        np.testing.assert_allclose(
            viterbi1[0][1],
            viterbi2[0][1],
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            viterbi1[0][1][:20],
            viterbi3[0][1][:20],
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            viterbi1[0][1][:20],
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_state_samples(self):
        # test state samples
        state_samples1 = self.fam1.sample_posterior_states(
            self.init_coef, self.coef_split_idx, self.ys, self.Xs, 2
        )

        state_samples2 = self.fam2.sample_posterior_states(
            self.init_coef, self.coef_split_idx, self.tv_ys, self.tv_Xs, 2
        )

        state_samples3 = self.fam3.sample_posterior_states(
            self.od_init_coef, self.od_coef_split_idx, self.od_ys, self.od_Xs, 2
        )

        np.testing.assert_allclose(
            state_samples1[0][1],
            state_samples2[0][1],
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            state_samples1[0][1][:20, 1],
            np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0]),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            state_samples3[0][1][:20, 1],
            np.array([0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0]),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_predict(self):
        # test prediction
        prediction1 = self.fam1.predict(
            self.init_coef, self.coef_split_idx, self.ys, self.Xs, n_samples=2
        )

        prediction2 = self.fam2.predict(
            self.init_coef, self.coef_split_idx, self.tv_ys, self.tv_Xs, n_samples=2
        )

        prediction3 = self.fam3.predict(
            self.od_init_coef,
            self.od_coef_split_idx,
            self.od_ys,
            self.od_Xs,
            n_samples=2,
        )

        np.testing.assert_allclose(
            prediction1[0][0],
            prediction2[0][0],
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            prediction1[0][1],
            prediction2[0][1],
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            prediction1[0][0][:5, :, 0],
            np.array(
                [
                    [-2.96476414, -0.65006362, -7.60078721],
                    [0.25008563, 3.18114284, 1.81762191],
                    [-2.22843751, -1.68034947, -8.39775076],
                    [0.04533482, 1.47884745, -0.24022577],
                    [-1.90705985, -1.56167648, -1.1013139],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            prediction3[0][0][:5, :, 0],
            np.array(
                [
                    [-2.96476414, -0.65006362, -7.60078721],
                    [0.25008563, 3.18114284, 1.81762191],
                    [-2.22843751, -1.68034947, -8.39775076],
                    [0.04533482, 1.47884745, -0.24022577],
                    [-1.90705985, -1.56167648, -1.1013139],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_predictive_resid(self):
        # Test predictive resid
        pred_resid1 = self.fam1.get_resid(
            self.init_coef,
            self.coef_split_idx,
            self.ys,
            self.Xs,
            resid_type="predictive",
        )

        pred_resid2 = self.fam2.get_resid(
            self.init_coef,
            self.coef_split_idx,
            self.tv_ys,
            self.tv_Xs,
            resid_type="predictive",
        )

        pred_resid3 = self.fam3.get_resid(
            self.od_init_coef,
            self.od_coef_split_idx,
            self.od_ys,
            self.od_Xs,
            resid_type="predictive",
        )

        np.testing.assert_allclose(
            pred_resid1,
            pred_resid2,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            pred_resid1[:5, :],
            np.array(
                [
                    [-2.85046777, -1.7181105, -4.19853279],
                    [-0.7696748, -1.30788421, -1.40362903],
                    [-1.25885595, -1.64247998, -1.72167249],
                    [-1.24941138, -1.66806177, -1.99060636],
                    [-1.5177543, -1.92540368, -2.2091171],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            pred_resid3[:5, :],
            np.array(
                [
                    [-2.81573851, -1.71229736, -4.08310188],
                    [-0.75344712, -1.3088929, -1.4026881],
                    [-1.24805185, -1.64312734, -1.72083857],
                    [-1.23201096, -1.67057077, -1.9899709],
                    [-1.50021403, -1.92840169, -2.20746248],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_forward_resid(self):
        # Test forward resid
        forward_resid1 = self.fam1.get_resid(
            self.init_coef,
            self.coef_split_idx,
            self.ys,
            self.Xs,
            resid_type="forward",
        )

        forward_resid2 = self.fam2.get_resid(
            self.init_coef,
            self.coef_split_idx,
            self.tv_ys,
            self.tv_Xs,
            resid_type="forward",
        )

        forward_resid3 = self.fam3.get_resid(
            self.od_init_coef,
            self.od_coef_split_idx,
            self.od_ys,
            self.od_Xs,
            resid_type="forward",
        )

        np.testing.assert_allclose(
            forward_resid1,
            forward_resid2,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            forward_resid1[:5, :],
            np.array(
                [
                    [-1.94598191, -0.65367552, -0.96855481],
                    [1.56183752, 0.64444896, -1.20854879],
                    [-2.26434264, -0.52807118, -0.9964072],
                    [0.97091134, 1.09212051, -0.99332044],
                    [-0.26832785, 0.5483176, -1.38903507],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            forward_resid3[:5, :],
            np.array(
                [
                    [-1.97745873, -0.62443529, -0.9983917],
                    [1.53899815, 0.59128775, -1.25965858],
                    [-2.29025925, -0.51067471, -1.03167802],
                    [0.97242687, 1.0164081, -1.02992055],
                    [-0.28974535, 0.50511576, -1.44570645],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_viterbi_resid(self):
        # Test viterbi resid
        viterbi_resid1 = self.fam1.get_resid(
            self.init_coef,
            self.coef_split_idx,
            self.ys,
            self.Xs,
            resid_type="viterbi_dur",
        )

        viterbi_resid2 = self.fam2.get_resid(
            self.init_coef,
            self.coef_split_idx,
            self.tv_ys,
            self.tv_Xs,
            resid_type="viterbi_dur",
        )

        viterbi_resid3 = self.fam3.get_resid(
            self.od_init_coef,
            self.od_coef_split_idx,
            self.od_ys,
            self.od_Xs,
            resid_type="viterbi_dur",
        )

        np.testing.assert_allclose(
            viterbi_resid1,
            viterbi_resid2,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            viterbi_resid1,
            np.array(
                [
                    [0.26190762, 0.01786241, 0.523248],
                    [1.01132804, 0.85198392, 0.58350554],
                    [0.50689602, 0.76070477, 0.74922525],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            viterbi_resid3,
            np.array(
                [
                    [0.79597972, 1.0079424, 0.70741119],
                    [0.63609553, 0.73972272, 0.4817017],
                    [0.50689602, 0.76070477, 0.74922525],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_posterior_resid(self):
        # Test posterior resid
        posterior_resid1 = self.fam1.get_resid(
            self.init_coef,
            self.coef_split_idx,
            self.ys,
            self.Xs,
            resid_type="posterior_dur",
        )

        posterior_resid2 = self.fam2.get_resid(
            self.init_coef,
            self.coef_split_idx,
            self.tv_ys,
            self.tv_Xs,
            resid_type="posterior_dur",
        )

        posterior_resid3 = self.fam3.get_resid(
            self.od_init_coef,
            self.od_coef_split_idx,
            self.od_ys,
            self.od_Xs,
            resid_type="posterior_dur",
        )

        np.testing.assert_allclose(
            posterior_resid1,
            posterior_resid2,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            posterior_resid1[:, :, 0],
            np.array(
                [
                    [0.16469857, -0.4488697, 0.13306259],
                    [0.19098355, 0.11323988, 0.01077971],
                    [-0.39889136, -0.39197353, -0.53289776],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            posterior_resid3[:, :, 0],
            np.array(
                [
                    [0.39645963, -0.11576908, 0.22357649],
                    [0.2105846, -0.23401751, -0.20967159],
                    [-0.08712459, -0.5551245, -0.41983202],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )


class Test_UnivariateHSMM_hard:

    # Start by simulating some data - same as above
    np_gen = np.random.default_rng(0)
    n_series = 3
    n_obs = 250
    starts_with_first = False

    # Define True model
    pi = np.array([0.6, 0.2, 0.2])

    T = np.array([[0.0, 0.6, 0.4], [0.3, 0.0, 0.7], [0.6, 0.4, 0.0]])

    mus = [np.array([0.0, 2.5, 5.0]), np.array([-2, 0, -5]), np.array([0, 0, 0])]

    sigmas = [
        np.array([[1, -0.3, 0.5], [-0.3, 1, 0.0], [0.5, 0.0, 1]]),
        np.array([[1, 0.3, -0.5], [0.3, 1, 0.3], [-0.5, 0.3, 1]]),
        np.array([[1, 0.0, 0.2], [0.0, 1, -0.6], [0.2, -0.6, 1]]),
    ]

    # Duration distribution parameters
    mus2 = [17, 12, 7]
    scales = [0.05, 0.05, 0.05]
    mus2_init = [2, 4, 6]
    scales_init = scales

    y_all = [[], [], []]
    states_all = []
    time_all = []
    series_all = []
    durs_all = []
    seed = 0
    for s in range(n_series):

        state = np_gen.choice(np.arange(3), p=pi)

        if starts_with_first:
            alpha = 1 / scales[state]
            beta = alpha / mus2[state]
        else:
            alpha = 1 / scales_init[state]
            beta = alpha / mus2_init[state]

        # duration samples
        dur = int(
            scp.stats.gamma.rvs(a=alpha, scale=(1 / beta), size=1, random_state=seed)[0]
        )
        seed += 1

        y = [[], [], []]
        states = []
        durs = []
        t = 0

        while len(y[0]) < n_obs:
            Y = scp.stats.multivariate_normal.rvs(
                size=dur, mean=mus[state], cov=sigmas[state], random_state=seed
            )
            Y = Y.reshape(dur, 3)
            for m in range(3):
                y[m].extend(Y[:, m])

            for _ in range(dur):
                states.append(state)
            durs.append([state, dur])

            prev_state = state
            state = np_gen.choice(np.arange(3), p=T[state, :])

            alpha = 1 / scales[state]
            beta = alpha / mus2[state]

            # duration samples
            dur = int(
                scp.stats.gamma.rvs(
                    a=alpha, scale=(1 / beta), size=1, random_state=seed
                )[0]
            )
            t += 1
            seed += 1

        for m in range(3):
            y[m] = y[m][:n_obs]
        states = states[:n_obs]
        time = np.arange(n_obs)
        for m in range(3):
            y_all[m].extend(y[m])
        time_all.extend(time)

        for _ in range(n_obs):
            series_all.append(s)

        states_all.append(states)
        durs_all.append(durs)

    dat = pd.DataFrame(
        {
            "y0": y_all[0],
            "y1": y_all[1],
            "y2": y_all[2],
            "time": time_all,
            "series": series_all,
        }
    )

    # Prep dat for hsmm model
    uval, id = np.unique(np.array(dat["series"]), return_index=True)
    sid = np.sort(id)
    tid = np.arange(len(id))

    # Now set up model
    o_family = GAUMLSS(links=[Identity(), LOGb(-0.0001)])

    d_family = GAMMALS(links=[LOG(), LOGb(-0.0001)])
    n_S = 3
    M = 3
    D = 100
    starts_with_first = False

    # Initial parameters
    Y = np.concatenate(
        [
            dat["y0"].values.reshape(-1, 1),
            dat["y1"].values.reshape(-1, 1),
            dat["y2"].values.reshape(-1, 1),
        ],
        axis=1,
    )
    init_size = 50
    init_choice = np_gen.choice(len(dat), size=init_size * n_S, replace=False)
    int_samples = []
    init_mus = []
    init_sds = []

    for j in range(n_S):
        int_samples.append(Y[init_choice[j * init_size : (j + 1) * init_size]])
        init_mus.append(int_samples[j].mean(axis=0))

        cov = np.cov(int_samples[j].T)
        init_sds.append(np.diag(cov))

    # Set up formulas
    ys = []
    Xs = []
    form_n_coef = []
    init_coef = []
    build_mat_idx = []
    build_matrix = []

    obs_families = []
    obs_formulas = []
    links = []
    pars = 0
    extra_coef = 0
    for j in range(n_S):
        obs_families.append([])

        for m in range(M):
            obs_families[-1].append(o_family)
            # Model of mean of obs model in state j and of signal m
            obs_formulas.append(Formula(lhs(f"y{m}"), [i()], data=dat))

            init_coef.append(init_mus[j][m])
            form_n_coef.append(obs_formulas[-1].n_coef)
            _ = build_penalties(obs_formulas[-1])
            Xs.append(build_model_matrix(obs_formulas[-1]))
            ys.append(dat[f"y{m}"].values.reshape(-1, 1))

            build_matrix.append(True)
            build_mat_idx.append(len(build_mat_idx))

            # Model of scale parameter of obs model in state j and of signal m
            obs_formulas.append(Formula(lhs(f"y{m}"), [i()], data=dat))

            init_coef.append(np.log(np.sqrt(init_sds[j][m])))
            form_n_coef.append(obs_formulas[-1].n_coef)
            _ = build_penalties(obs_formulas[-1])
            Xs.append(build_model_matrix(obs_formulas[-1]))
            ys.append(dat[f"y{m}"].values.reshape(-1, 1))

            build_matrix.append(True)
            build_mat_idx.append(len(build_mat_idx))

            links.extend(o_family.links)
            pars += 2

    d_dat = pd.DataFrame(
        {"x": [1 for _ in range(len(tid))], "y": [1 for _ in range(len(tid))]}
    )

    d_families = []
    d_formulas = []

    for j in range(n_S * (1 if starts_with_first else 2)):
        d_families.append(d_family)

        # Model of mean of dur model in state j
        d_formulas.append(Formula(lhs("y"), [i()], data=d_dat))

        init_coef.append(2.5)
        form_n_coef.append(d_formulas[-1].n_coef)
        _ = build_penalties(d_formulas[-1])
        Xs.append(build_model_matrix(d_formulas[-1]))
        ys.append(d_dat["y"].values.reshape(-1, 1))
        build_matrix.append(True)
        build_mat_idx.append(len(build_mat_idx))

        # Model of scale parameter of dur model in state j
        d_formulas.append(Formula(lhs("y"), [i()], data=d_dat))

        init_coef.append(-2)
        form_n_coef.append(d_formulas[-1].n_coef)
        _ = build_penalties(d_formulas[-1])
        Xs.append(build_model_matrix(d_formulas[-1]))
        ys.append(d_dat["y"].values.reshape(-1, 1))
        build_matrix.append(True)
        build_mat_idx.append(len(build_mat_idx))

        links.extend(d_family.links)
        pars += 2

    t_formulas = []
    for j in range(n_S):
        for par in range(n_S - 2):

            # Model of state transition away from state j
            t_formulas.append(Formula(lhs("y"), [i()], data=d_dat))

            init_coef.append(np.log(1))
            form_n_coef.append(t_formulas[-1].n_coef)
            _ = build_penalties(t_formulas[-1])
            Xs.append(build_model_matrix(t_formulas[-1]))
            ys.append(d_dat["y"].values.reshape(-1, 1))
            build_matrix.append(True)
            build_mat_idx.append(len(build_mat_idx))
            pars += 1

    pi_formulas = []

    for par in range(n_S - 1):
        # Model of initial state probability
        pi_formulas.append(Formula(lhs("y"), [i()], data=d_dat))

        init_coef.append(np.log(1))
        form_n_coef.append(pi_formulas[-1].n_coef)
        _ = build_penalties(pi_formulas[-1])
        Xs.append(build_model_matrix(pi_formulas[-1]))
        ys.append(d_dat["y"].values.reshape(-1, 1))
        build_matrix.append(True)
        build_mat_idx.append(len(build_mat_idx))
        pars += 1

    # Initialize Family
    fam = HSMMFamily(
        pars,
        links,
        n_S,
        obs_fams=obs_families,
        d_fams=d_families,
        sid=sid,
        tid=tid,
        D=D,
        M=M,
        starts_with_first=starts_with_first,
        ends_with_last=False,
        ends_in_last=False,
        n_cores=1,
        build_mat_idx=build_mat_idx,
    )

    # Initialize coef and penalties
    coef_split_idx = form_n_coef[:-1]

    for coef_i in range(1, len(coef_split_idx)):
        coef_split_idx[coef_i] += coef_split_idx[coef_i - 1]

    total_coef = np.sum(form_n_coef)
    init_coef = np.array(init_coef).reshape(-1, 1)

    def test_llk(self):
        # Test log-likelihood of univariate hsmm
        llk = self.fam.llk(self.init_coef, self.coef_split_idx, self.ys, self.Xs)

        np.testing.assert_allclose(
            llk,
            -4870.823813514383,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_grad(self):
        # Test grad function of univariate hsmm
        grad = self.fam.gradient(self.init_coef, self.coef_split_idx, self.ys, self.Xs)
        pos_llk_warp = lambda x: self.fam.llk(
            x.reshape(-1, 1), self.coef_split_idx, self.ys, self.Xs
        )
        grad2 = scp.optimize.approx_fprime(self.init_coef.flatten(), pos_llk_warp)

        grad_diff = np.abs(np.round(grad - grad2.reshape(-1, 1), decimals=3)).max()

        np.testing.assert_allclose(
            grad_diff,
            0.0,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )


class Test_UnivariateHMP:
    # Start by simulating some data
    np_gen = np.random.default_rng(0)
    n_series = 100

    # Define True model
    pi = None

    T = None

    mus = [np.array([0.0, 2.5, 5.0]), np.array([-2, 0, -5])]

    # Duration distribution parameters
    mus2 = [17, 12, 7]
    scales = [0.05, 0.05, 0.05]

    y_all = [[], [], []]
    states_all = []
    time_all = []
    series_all = []
    seed = 0
    durs_all = []
    for s in range(n_series):
        y = [[], [], []]
        states = []
        durs = []
        t = 0
        state = 0

        while state < 5:

            if state % 2 == 1:
                dur = 5
                Y = scp.stats.multivariate_normal.rvs(
                    size=dur,
                    mean=mus[state // 2],
                    cov=np.identity(3),
                    random_state=seed,
                )

            else:
                alpha = 1 / scales[state // 2]
                beta = alpha / mus2[state // 2]

                # duration samples
                dur = int(
                    scp.stats.gamma.rvs(
                        a=alpha, scale=(1 / beta), size=1, random_state=seed
                    )[0]
                )

                Y = scp.stats.multivariate_normal.rvs(
                    size=dur, mean=np.zeros(3), cov=np.identity(3), random_state=seed
                )

            seed += 1

            for m in range(3):
                y[m].extend(Y[:, m])

            for _ in range(dur):
                states.append(state)

            state += 1

        n_obs = len(states)
        time = np.arange(n_obs)
        time_all.extend(time)
        durs_all.append(len(time))

        for _ in range(n_obs):
            series_all.append(s)

        for m in range(3):
            y_all[m].extend(y[m])

        states_all.append(states)

    dat = pd.DataFrame(
        {
            "y0": y_all[0],
            "y1": y_all[1],
            "y2": y_all[2],
            "time": time_all,
            "series": series_all,
        }
    )

    dat["x"] = np_gen.random(size=len(time_all))

    # Prep dat for hsmm model
    uval, id = np.unique(np.array(dat["series"]), return_index=True)
    sid = np.sort(id)
    tid = np.arange(len(id))

    # Event shape assumed by Anderson et al. (2016)
    event_shape = np.array([0.30901699, 0.80901699, 1.0, 0.80901699, 0.30901699])

    # Test gradients
    ars = [False, True]
    shared_ms = [False, True]
    init_coefs = []
    coef_split_idxs = []
    yss = []
    Xss = []
    families = []

    for ar in ars:
        for shared_m in shared_ms:

            o_family = None
            d_family = GAMMALS(links=[LOG(), LOGb(-0.0001)])
            n_S = 5
            M = 3
            D = max(durs_all)

            ys = []
            Xs = []
            form_n_coef = []
            init_coef = []

            obs_families = []
            shared_formulas = []
            obs_formulas = []
            build_matrix = []
            build_mat_idx = []
            links = []
            pars = 0
            shared_pars = [0]

            rho = 0.5
            Lrhoi = None

            if ar:
                ar_form = Formula(lhs(f"y{1}"), [i()], data=dat, series_id="series")
                Lrhoi, _ = computeAr1Chol(ar_form, rho)

            if len(shared_pars) > 0:
                midx = 1
                if shared_m is False:
                    midx = M

                for m in range(midx):

                    shared_formulas.append(Formula(lhs(f"y{m}"), [f(["x"])], data=dat))

                    init_coef.extend([0.01 for _ in range(shared_formulas[-1].n_coef)])
                    form_n_coef.append(shared_formulas[-1].n_coef)
                    _ = build_penalties(shared_formulas[-1])

                    ys.append(dat[f"y{m}"].values.reshape(-1, 1))

                    if m == 0:
                        build_matrix.append(True)
                        Xs.append(build_model_matrix(shared_formulas[-1]))
                    else:
                        build_matrix.append(False)
                        Xs.append(None)
                    build_mat_idx.append(0)

            event = 0
            start_sep = len(build_matrix)
            for j in range(n_S):
                obs_families.append([])

                for m in range(M):

                    obs_families[-1].append(o_family)

                    if j % 2 == 1:

                        # Model of mean of obs model in state j and of signal m
                        obs_formulas.append(Formula(lhs(f"y{m}"), [i()], data=dat))

                        init_coef.extend(
                            scp.stats.norm.rvs(
                                size=obs_formulas[-1].n_coef,
                                random_state=event * (m + 1) + m,
                            )
                        )
                        form_n_coef.append(obs_formulas[-1].n_coef)
                        _ = build_penalties(obs_formulas[-1])
                        ys.append(dat[f"y{m}"].values.reshape(-1, 1))

                        if len(build_matrix) == start_sep:
                            build_matrix.append(True)
                            Xs.append(build_model_matrix(obs_formulas[-1]))
                        else:
                            build_matrix.append(False)
                            Xs.append(None)
                        build_mat_idx.append(start_sep)

                        links.extend([Identity()])
                        if m >= (M - 1):
                            event += 1
                        pars += 1

            d_dat = pd.DataFrame({"d": np.arange(len(sid)), "sub": np.ones(len(sid))})

            d_dat = d_dat.astype({"sub": "O"})

            d_families = []
            d_formulas = []

            flat = 0
            start_durs = len(build_mat_idx)
            for j in range(n_S):

                if j % 2 == 1:
                    d_families.append(None)
                else:

                    d_families.append(d_family)

                    # Model of mean of dur model in state j
                    d_formulas.append(Formula(lhs("d"), [ri("sub", id=1)], data=d_dat))

                    build_matrix.append(True)
                    _ = build_penalties(d_formulas[-1])
                    Xs.append(build_model_matrix(d_formulas[-1]))
                    form_n_coef.append(d_formulas[-1].n_coef)
                    init_coef.extend([5])
                    ys.append(d_dat["d"].values.reshape(-1, 1))
                    build_mat_idx.append(len(build_mat_idx))

                    # Model of scale of dur model in state j
                    d_formulas.append(Formula(lhs("d"), [ri("sub", id=2)], data=d_dat))

                    build_matrix.append(True)
                    _ = build_penalties(d_formulas[-1])
                    Xs.append(build_model_matrix(d_formulas[-1]))
                    form_n_coef.append(d_formulas[-1].n_coef)
                    init_coef.extend([1])
                    ys.append(d_dat["d"].values.reshape(-1, 1))
                    build_mat_idx.append(len(build_mat_idx))

                    links.extend(d_family.links)
                    flat += 1
                    pars += 2

            # Initialize Family
            T = np.diag(
                np.ones(n_S - 1), 1
            )  # Super diagonal shift matrix for transitions

            pi = np.zeros(n_S)  # Start in first state
            pi[0] = 1
            fam = HSMMFamily(
                pars,
                links,
                n_S,
                obs_fams=obs_families,
                d_fams=d_families,
                sid=sid,
                tid=tid,
                D=D,
                M=M,
                starts_with_first=True,
                ends_with_last=True,
                ends_in_last=True,
                T=T,
                pi=pi,
                scale=1,
                event_template=event_shape,
                hmp_fam="Gaussian",
                n_cores=4,
                build_mat_idx=build_mat_idx,
                shared_pars=shared_pars,
                shared_m=shared_m,
                Lrhoi=Lrhoi,
            )

            # Setup coef
            coef_split_idx = form_n_coef[:-1]

            for coef_i in range(1, len(coef_split_idx)):
                coef_split_idx[coef_i] += coef_split_idx[coef_i - 1]

            total_coef = np.sum(form_n_coef)
            init_coef = np.array(init_coef).reshape(-1, 1)

            # Collect
            init_coefs.append(copy.deepcopy(init_coef))
            coef_split_idxs.append(copy.deepcopy(coef_split_idx))
            yss.append(copy.deepcopy(ys))
            Xss.append(copy.deepcopy(Xs))
            families.append(copy.deepcopy(fam))

    def test_grads(self):
        # Test grad function
        grad_diffs = []
        for fam, init_coef, coef_split_idx, ys, Xs in zip(
            self.families, self.init_coefs, self.coef_split_idxs, self.yss, self.Xss
        ):
            grad = fam.gradient(init_coef, coef_split_idx, ys, Xs)

            pos_llk_warp = lambda x: fam.llk(
                x.reshape(-1, 1),
                coef_split_idx,
                ys,
                Xs,
            )
            grad2 = scp.optimize.approx_fprime(init_coef.flatten(), pos_llk_warp)

            grad_diff = np.abs(np.round(grad - grad2.reshape(-1, 1), decimals=3)).max()
            grad_diffs.append(grad_diff)

        grad_diffs = np.array(grad_diffs)
        np.testing.assert_allclose(
            grad_diffs,
            np.array([0.001, 0.001, 0.001, 0.001]),
            atol=min(max_atol, 0.1),
            rtol=min(max_rtol, 0.1),
        )

    def test_total_coef(self):
        total_coefs = []
        for fam, init_coef, coef_split_idx, ys, Xs in zip(
            self.families, self.init_coefs, self.coef_split_idxs, self.yss, self.Xss
        ):
            total_coefs.append(len(init_coef))

        total_coefs = np.array(total_coefs)
        np.testing.assert_allclose(
            total_coefs,
            np.array([39, 21, 39, 21]),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_ods(self):
        # Test observation and duration probabilities for hmp
        ods = self.families[0].compute_od_probs(
            self.init_coefs[0],
            self.coef_split_idxs[0],
            self.yss[0],
            self.Xss[0],
            log=False,
        )

        np.testing.assert_allclose(
            np.round(ods[0][0][:5, :, 1], decimals=6),
            np.array(
                [
                    [0.002269, 0.005953, 0.005823, 0.005953, 0.002269],
                    [0.0002, 0.001067, 0.001368, 0.001067, 0.0002],
                    [0.029579, 0.023166, 0.014269, 0.023166, 0.029579],
                    [0.054175, 0.019046, 0.008643, 0.019046, 0.054175],
                    [0.04037, 0.023882, 0.013221, 0.023882, 0.04037],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            ods[0][1][:5, :],
            np.array(
                [
                    [0.10865586, 0.0, 0.10865586, 0.0, 0.10865586],
                    [0.06993352, 0.0, 0.06993352, 0.0, 0.06993352],
                    [0.0539878, 0.0, 0.0539878, 0.0, 0.0539878],
                    [0.04489946, 0.0, 0.04489946, 0.0, 0.04489946],
                    [0.03889596, 1.0, 0.03889596, 1.0, 0.03889596],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_predict(self):
        # Test signal + state prediction for hmp
        prediction = self.families[2].predict(
            self.init_coefs[2],
            self.coef_split_idxs[2],
            self.yss[2],
            self.Xss[2],
            n_samples=2,
        )

        np.testing.assert_allclose(
            prediction[0][0][:5, :, 0],
            np.array(
                [
                    [1.76405235, 1.62434536, -0.41675785],
                    [1.72963813, 0.35476253, -2.05837901],
                    [2.48217173, 0.92684626, -2.58235028],
                    [2.06388657, 1.97447444, -0.85566706],
                    [0.94255314, 1.26353318, -1.34404703],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

        np.testing.assert_allclose(
            prediction[0][1][:20, 0],
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )
