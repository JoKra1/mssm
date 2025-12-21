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

    # Initialize Families
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

    fam2 = HSMMFamily(
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

    S_emb = scp.sparse.csc_array(scp.sparse.eye(total_coef))
    od_S_emb = scp.sparse.csc_array(scp.sparse.eye(od_total_coef))

    penalties = [
        LambdaTerm(
            S_J=S_emb,
            S_J_emb=S_emb,
            D_J_emb=S_emb,
            start_index=0,
            lam=10,
            rank=total_coef,
        )
    ]
    od_penalties = [
        LambdaTerm(
            S_J=od_S_emb,
            S_J_emb=od_S_emb,
            D_J_emb=od_S_emb,
            start_index=0,
            lam=10,
            rank=od_total_coef,
        )
    ]

    # Now can estimate
    def callback(outer, pen_llk, coef, lam):
        print(outer, pen_llk, lam, coef[:9])

    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["control_lambda"] = 1
    test_kwargs["extend_lambda"] = False
    test_kwargs["max_outer"] = 500
    test_kwargs["max_inner"] = 500
    test_kwargs["method"] = "qEFS"
    test_kwargs["repara"] = False
    test_kwargs["prefit_grad"] = True
    test_kwargs["progress_bar"] = True
    test_kwargs["max_restarts"] = -1
    test_kwargs["seed"] = 5
    test_kwargs["global_opt_qefs"] = True
    test_kwargs["init_lambda"] = [10]

    bfgs_options = {
        "gtol": 1.1 * 1e-7,
        "ftol": 1.1 * 1e-7,
        "maxcor": 30,
        "maxls": 20,
        "maxfun": 500,
    }

    test_kwargs["bfgs_options"] = bfgs_options
    test_kwargs["init_coef"] = init_coef
    test_kwargs["extra_penalties"] = penalties
    test_kwargs["callback"] = callback

    hsmm = GSMM([*obs_formulas, *d_formulas, *t_formulas, *pi_formulas], family=fam)
    hsmm.fit(**test_kwargs)

    test_kwargs["init_coef"] = od_init_coef
    test_kwargs["extra_penalties"] = od_penalties

    hsmm2 = GSMM([*obs_formulas, *d_formulas], family=fam2)
    hsmm2.fit(**test_kwargs)

    # Standard tests
    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.hsmm.edf,
            35.695945757029314,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_GAMcoef(self):
        coef = self.hsmm.coef
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    [-0.01936465],
                    [0.00817976],
                    [0.02563425],
                    [-1.95208348],
                    [0.00517209],
                    [-5.00445096],
                    [0.13447266],
                    [2.40962538],
                    [5.05054534],
                    [1.90028138],
                    [-3.1597454],
                    [2.31954038],
                    [-2.09700616],
                    [2.81426287],
                    [-1.41867593],
                    [0.29349792],
                    [-1.1756628],
                    [0.09782385],
                    [1.08065604],
                    [0.4341386],
                    [-0.01586282],
                    [-0.02783987],
                    [-0.06346497],
                    [0.21290136],
                    [0.68564144],
                    [0.03753095],
                    [0.28118655],
                    [-0.71335842],
                    [0.85483309],
                    [0.05220327],
                    [-0.2932202],
                    [-0.03627107],
                    [0.23512547],
                    [0.36591222],
                    [-0.71050034],
                    [-0.01383196],
                    [0.03921854],
                    [0.0570436],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.5),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.hsmm.overall_penalties])
        np.testing.assert_allclose(
            lam,
            np.array([0.36071882606318695]),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 2.5),
        )

    def test_GAMreml(self):
        reml = self.hsmm.get_reml()
        np.testing.assert_allclose(
            reml, -1260.4943823469362, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMllk(self):
        llk = self.hsmm.get_llk(False)
        np.testing.assert_allclose(
            llk, -1134.711336277765, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMedf2(self):
        np.testing.assert_allclose(
            self.hsmm2.edf,
            32.858595174548306,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_GAMcoef2(self):
        coef = self.hsmm2.coef
        np.testing.assert_allclose(
            coef,
            np.array(
                [
                    [-0.01936699],
                    [0.00817927],
                    [0.0256361],
                    [-1.95204075],
                    [0.00499744],
                    [-5.00480245],
                    [0.13458375],
                    [2.40975425],
                    [5.05080119],
                    [1.90032838],
                    [-3.16462809],
                    [2.31979025],
                    [-2.10046128],
                    [2.81491996],
                    [-1.42122901],
                    [-0.0158635],
                    [-0.0278502],
                    [-0.06347819],
                    [0.21293755],
                    [0.68574448],
                    [0.03753274],
                    [0.28124869],
                    [-0.71348837],
                    [0.85497674],
                    [0.05220913],
                    [-0.29324338],
                    [-0.03627037],
                    [0.23515093],
                    [0.36594015],
                    [-0.71056214],
                    [-0.01383214],
                    [0.03922228],
                    [0.05704667],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.5),
        )

    def test_GAMlam2(self):
        lam = np.array([p.lam for p in self.hsmm2.overall_penalties])
        np.testing.assert_allclose(
            lam,
            np.array([0.3416055393086223]),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 2.5),
        )

    def test_GAMreml2(self):
        reml = self.hsmm2.get_reml()
        np.testing.assert_allclose(
            reml, -1271.0859442419023, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMllk2(self):
        llk = self.hsmm2.get_llk(False)
        np.testing.assert_allclose(
            llk, -1150.6714419260961, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    # Test family specific functions
    def test_t(self):
        ps = self.fam.compute_Tpi(self.hsmm.coef, self.coef_split_idx, self.ys, self.Xs)
        T = ps[0][0]

        np.testing.assert_allclose(
            T,
            np.array(
                [
                    [0.0, 0.42714773, 0.57285227],
                    [0.76416707, 0.0, 0.23583293],
                    [0.47556352, 0.52443648, 0.0],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_p(self):
        ps = self.fam.compute_Tpi(self.hsmm.coef, self.coef_split_idx, self.ys, self.Xs)
        p = ps[0][1]

        np.testing.assert_allclose(
            p,
            np.array([0.18214124, 0.53669957, 0.28115919]),
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

    S_emb = scp.sparse.csc_array(scp.sparse.eye(init_coef.shape[0]))

    penalties = [
        LambdaTerm(
            S_J=S_emb,
            S_J_emb=S_emb,
            D_J_emb=S_emb,
            start_index=0,
            lam=10,
            rank=init_coef.shape[0],
        )
    ]

    # And fit
    test_kwargs = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["control_lambda"] = 1
    test_kwargs["extend_lambda"] = False
    test_kwargs["max_outer"] = 500
    test_kwargs["max_inner"] = 500
    test_kwargs["method"] = "qEFS"
    test_kwargs["repara"] = False
    test_kwargs["prefit_grad"] = True
    test_kwargs["progress_bar"] = True
    test_kwargs["max_restarts"] = -1
    test_kwargs["seed"] = 5
    test_kwargs["global_opt_qefs"] = True
    test_kwargs["init_lambda"] = [10]

    bfgs_options = {
        "gtol": 1.1 * 1e-7,
        "ftol": 1.1 * 1e-7,
        "maxcor": 30,
        "maxls": 20,
        "maxfun": 500,
    }

    test_kwargs["bfgs_options"] = bfgs_options
    test_kwargs["init_coef"] = init_coef
    test_kwargs["extra_penalties"] = penalties
    test_kwargs["build_mat"] = build_matrix

    hsmm = GSMM([*obs_formulas, *d_formulas, *t_formulas, *pi_formulas], family=fam)
    hsmm.fit(**test_kwargs)

    def test_grad1(self):
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

    # Test coef

    def test_GAMedf(self):
        np.testing.assert_allclose(
            self.hsmm.edf,
            33.11384860717911,
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_GAMcoef(self):
        coef = self.hsmm.coef
        np.testing.assert_allclose(
            np.round(coef, decimals=4),
            np.array(
                [
                    [-1.9200e-02],
                    [1.7000e-02],
                    [7.8000e-03],
                    [-3.2000e-02],
                    [2.6100e-02],
                    [-3.8000e-02],
                    [-1.9485e00],
                    [4.5600e-02],
                    [2.1000e-03],
                    [-8.0000e-03],
                    [-5.0039e00],
                    [3.6800e-02],
                    [1.3650e-01],
                    [2.0000e-03],
                    [2.4099e00],
                    [1.4400e-02],
                    [5.0515e00],
                    [-5.7100e-02],
                    [-2.8500e-02],
                    [6.2000e-03],
                    [1.3584e00],
                    [-1.9502e00],
                    [-6.4820e-01],
                    [-5.4340e-01],
                    [1.9002e00],
                    [-3.1787e00],
                    [2.3785e00],
                    [-2.8071e00],
                    [2.8661e00],
                    [-2.9061e00],
                    [2.9660e-01],
                    [-1.1941e00],
                    [9.9200e-02],
                    [1.1991e00],
                    [5.2940e-01],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.5),
        )

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.hsmm.overall_penalties])
        np.testing.assert_allclose(
            lam,
            np.array([0.2911866149837378]),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 2.5),
        )

    def test_GAMreml(self):
        reml = self.hsmm.get_reml()
        np.testing.assert_allclose(
            reml, -3489.508246303468, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    def test_GAMllk(self):
        llk = self.hsmm.get_llk(False)
        np.testing.assert_allclose(
            llk, -3378.8345006947566, atol=min(max_atol, 0), rtol=min(max_rtol, 0.1)
        )

    # Test family specific functions
    def test_t(self):
        ps = self.fam.compute_Tpi(self.hsmm.coef, self.coef_split_idx, self.ys, self.Xs)
        T = ps[0][0]

        np.testing.assert_allclose(
            T,
            np.array(
                [
                    [0.0, 0.42638427, 0.57361573],
                    [0.76747515, 0.0, 0.23252485],
                    [0.47521251, 0.52478749, 0.0],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_p(self):
        ps = self.fam.compute_Tpi(self.hsmm.coef, self.coef_split_idx, self.ys, self.Xs)
        p = ps[0][1]

        np.testing.assert_allclose(
            p,
            np.array([0.16625215, 0.55147263, 0.28227523]),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_viterbi(self):
        viterbi = self.fam.decode_viterbi(
            self.hsmm.coef, self.coef_split_idx, self.ys, self.Xs
        )

        np.testing.assert_allclose(
            viterbi[0][1][:20],
            np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_state_samples(self):
        state_samples = self.fam.sample_posterior_states(
            self.hsmm.coef, self.coef_split_idx, self.ys, self.Xs, 2
        )

        np.testing.assert_allclose(
            state_samples[0][1][:20, 1],
            np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_pred_resid(self):
        pred_resid = self.fam.get_resid(
            self.hsmm.coef,
            self.coef_split_idx,
            self.ys,
            self.Xs,
            resid_type="predictive",
        )

        np.testing.assert_allclose(
            pred_resid[:5, :],
            np.array(
                [
                    [-2.44220001, -1.40776388, -2.3386541],
                    [-0.17296623, -0.80491917, -0.97356931],
                    [-0.7277487, -1.14872244, -0.80666366],
                    [-0.52777857, -0.50255834, -1.04803087],
                    [-0.79525784, -0.54793967, -1.43441559],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_forward_resid(self):
        forward_resid = self.fam.get_resid(
            self.hsmm.coef,
            self.coef_split_idx,
            self.ys,
            self.Xs,
            resid_type="forward",
        )

        np.testing.assert_allclose(
            forward_resid[:5, :],
            np.array(
                [
                    [-1.75596183, -0.97223235, -0.00774127],
                    [1.76195888, 0.25303344, 0.1062796],
                    [-1.94304417, -0.69401835, 0.61348917],
                    [0.51631522, 0.7035971, 0.13665445],
                    [-0.78088569, -0.21806228, -1.17172933],
                ]
            ),
            atol=min(max_atol, 0),
            rtol=min(max_rtol, 0.1),
        )

    def test_viterbi_post_resid(self):
        viterbi_resid = self.fam.get_resid(
            self.hsmm.coef,
            self.coef_split_idx,
            self.ys,
            self.Xs,
            resid_type="viterbi_dur",
        )
        posterior_resid = self.fam.get_resid(
            self.hsmm.coef,
            self.coef_split_idx,
            self.ys,
            self.Xs,
            resid_type="posterior_dur",
        )

        np.testing.assert_allclose(
            posterior_resid.mean(axis=2) - viterbi_resid,
            np.array(
                [
                    [-1.35929254e-03, 1.01871257e-03, 2.77555756e-17],
                    [-2.85539928e-05, -4.21063701e-05, 1.11022302e-16],
                    [1.11022302e-16, 0.00000000e00, 0.00000000e00],
                ]
            ),
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
            atol=min(max_atol, 0),
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
