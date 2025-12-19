import numpy as np
import scipy as scp
import warnings
import copy
from ..python.gamm_solvers import (
    compute_S_emb_pinv_det,
    cpp_chol,
    compute_Linv,
    apply_eigen_perm,
    cpp_cholP,
    map_csc_to_eigen,
)

from .utils import sample_MVN, estimateVp, GAMLSSGSMMFamily

try:
    import multiprocess as mp
except ImportError:
    warnings.warn(
        "Multi-processing mcmc computations might require the `multiprocess` package."
    )
    from .file_loading import mp

from ...models import (
    GAMM,
    GAMMLSS,
    GSMM,
    Family,
    ExtendedFamily,
    GAMLSSFamily,
    fs,
)

from ..python.repara import reparam
from collections.abc import Callable

import mcmc
from tqdm import tqdm

HAS_ARVIZ = True
try:
    import arviz
except ImportError:
    warnings.warn("Convergence checks are only available when `arviz` is installed.")
    HAS_ARVIZ = False


def check_convergence(
    coef_samples: np.ndarray,
    llk_samples: np.ndarray | None,
    rho_samples: np.ndarray | None,
    model: GAMMLSS | GSMM,
    type: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Determines MCMC convergence statistics for the set of coefficients,
    the log joint probability, and optionally the rho parameters.

    Statistics computed are the effective samples size (ESS), Rhat, and the MC standard error
    (MCSE). See Gelman et al. (2013).

    References:
     - Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013).\
        Bayesian Data Analysis (3rd ed.). Chapman and Hall/CRC. https://doi.org/10.1201/b16018

    :param coef_samples: Sample of coefficients as np.array of dimension
        ``(n_chains,n_samples,n_coef)``.
    :type coef_samples: np.ndarray
    :param llk_samples: Optional sample of log joint probability scores as np.array of dimension
        ``(n_chains,n_samples,1)`` or None.
    :type llk_samples: np.ndarray | None
    :param rho_samples: Optional sample of log regularization parameters as np.array of dimension
        ``(n_chains,n_samples,n_rho)`` or None.
    :type rho_samples: np.ndarray | None
    :param model: The model from which we are sampling.
    :type model: GAMMLSS | GSMM
    :param type: For which parameters to collect convergence statistics. Type 1 collects statistics
        only for fixed coefficients and those involved in smooth functions with an improper prior
        as well as all rho parameters and the log joint. Type 0 collects statistics for all
        parameters, defaults to 1
    :type type: int, optional
    :return: Three arrays, holding the ESS, Rhat, and MCSE scores of collected parameters. Order
        is: first (selected) coefficients, then log rho scores, last element will be the log joint
        (if ``llk_samples is not None``).
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """

    ess_all = []
    rhat_all = []
    mcse_all = []
    for samples in [coef_samples, rho_samples, llk_samples]:

        if samples is None:
            ess_all.append([])
            rhat_all.append([])
            mcse_all.append([])
            continue

        n_chains = samples.shape[0]
        N = samples.shape[1]

        if N % 2 != 0:
            N -= 1

        par_samples = samples[:, :N, :]

        par_samples_split = np.zeros((2 * n_chains, N // 2, par_samples.shape[2]))

        split_idx = 0
        for chain in range(n_chains):
            par_samples_split[split_idx, :, :] = par_samples[chain, : N // 2, :]
            split_idx += 1
            # fmt: off
            par_samples_split[split_idx, :, :] = par_samples[chain, N // 2 :, :]  # noqa: E203
            # fmt: on
            split_idx += 1

        arviz_samples = arviz.convert_to_dataset(par_samples_split)
        ess = arviz.ess(arviz_samples, method="tail")
        rhat = arviz.rhat(arviz_samples)
        mcse = arviz.mcse(arviz_samples, method="mean")

        ess_all.append(ess.x.data)
        rhat_all.append(rhat.x.data)
        mcse_all.append(mcse.x.data)

    ess_sel = []
    rhat_sel = []
    mcse_sel = []

    # All coefs
    if type == 0:
        ess_sel.extend(ess_all[0])
        rhat_sel.extend(rhat_all[0])
        mcse_sel.extend(mcse_all[0])

    else:
        # Only group-level coef + lams
        split_ess = np.split(ess_all[0], model.coef_split_idx)
        split_rhat = np.split(rhat_all[0], model.coef_split_idx)
        split_mcse = np.split(mcse_all[0], model.coef_split_idx)

        for fidx, form in enumerate(model.formulas):

            sidx = form.get_smooth_term_idx()
            sidx = [idx for idx in sidx if isinstance(form.terms[idx], fs) is False]

            _, fixedX, _ = model.predict(
                use_terms=[*form.get_linear_term_idx(), *sidx],
                n_dat=form.data,
                par=fidx,
            )

            fcidx = np.arange(fixedX.shape[1])[fixedX.sum(axis=0) > 0]

            ess_sel.extend(split_ess[fidx][fcidx])
            rhat_sel.extend(split_rhat[fidx][fcidx])
            mcse_sel.extend(split_mcse[fidx][fcidx])

    # Lambda parameters
    if rho_samples is not None:
        ess_sel.extend(ess_all[1])
        rhat_sel.extend(rhat_all[1])
        mcse_sel.extend(mcse_all[1])

    # log joint
    if llk_samples is not None:
        ess_sel.extend(ess_all[2])
        rhat_sel.extend(rhat_all[2])
        mcse_sel.extend(mcse_all[2])

    return np.array(ess_sel), np.array(rhat_sel), np.array(mcse_sel)


def advance_chain_mssm(
    chain_id: int,
    return_dict: dict,
    chain,
) -> None:
    """Wrapper function to advancethe state of a NUTS ``chain`` (implemented in c++ class).

    :param chain_id: Id of chain, just an integer >= 0.
    :type chain_id: int
    :param return_dict: The shared dictionary in which all states dump the outcome of the next step.
    :type return_dict: dict
    :param chain: Python wrapper around the c++ NUTS class.
    :type chain: Callable
    """

    cllk, gamma = chain.advance_chain()

    return_dict[chain_id] = [cllk, gamma]


def sample_mssm(
    model: GAMM | GAMMLSS | GSMM,
    n_iter: int = 500,
    min_iter: int = 100,
    max_rhat: float = 1.02,
    min_ess: int = 10,
    M_adapt: int = 500,
    delta: float = 0.6,
    callback: Callable | None = None,
    n_chains: int = 2,
    parallelize_chains: bool = True,
    auto_converge: bool = True,
    drop_NA: bool = True,
    sample_rho: bool = False,
    convergence_type: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Samples the posterior of any model using a No-U-Turn (NUTS) sampler (Hoffman & Gelman, 2014).

    **Currently only supports ``GAMMLSS`` and ``GSMM`` models. I am still thinking about how to best
    adapt the ``GAMM`` api to support this function.**

    Examples::

        sim_dat = sim4(500, 2, family=Gamma(), seed=0)

        # We again define a model of the mean: \\mu_i = \\alpha + f(x0) + f(x1) + f_{x4}(x0)
        sim_formula_m = Formula(
            lhs("y"), [i(), f(["x0"]), f(["x1"]), f(["x2"]), f(["x3"])], data=sim_dat
        )

        # and the standard deviation:
        sim_formula_sd = Formula(lhs("y"), [i()], data=sim_dat)

        family = GAMMALS([LOG(), LOGb(-0.0001)])

        # Now define the model and fit!
        gsmm_fam = GAMLSSGSMMFamily(2, family)
        model = GSMM([sim_formula_m, sim_formula_sd], gsmm_fam)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(method='qEFS')

        # Now sample posterior (of coef and \\rho = log(\\lambda)):
        llks, coef_samples, rho_samples = sample_mssm(model2,auto_converge=False,M_adapt=100,
            parallelize_chains=False,n_chains=1,sample_rho=True,delta=0.5)

    References:
     - Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively Setting Path Lengths\
        in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(47), 1593–1623.
     - Betancourt, M. J. (2013). Generalizing the No-U-Turn Sampler to Riemannian Manifolds\
        (No. arXiv:1304.1920). arXiv. https://doi.org/10.48550/arXiv.1304.1920
     - Betancourt, M. (2018). A Conceptual Introduction to Hamiltonian Monte Carlo\
        (No. arXiv:1701.02434). arXiv. https://doi.org/10.48550/arXiv.1701.02434

    :param model: The model from which to sample, currently only ``GAMMLSS`` and ``GSMM`` are
        supported.
    :type model: GAMM | GAMMLSS | GSMM
    :param n_iter: Number of iterations to sample (after adaptation), defaults to 500
    :type n_iter: int, optional
    :param min_iter: Minimum number of iterations (after adaptation) before auto-convergence can
        stop sampling, defaults to 100
    :type min_iter: int, optional
    :param max_rhat: Maximum value of Rhat for any parameter (see ``convergence_type``) below which
        auto-convergence may stop sampling, defaults to 1.02
    :type max_rhat: float, optional
    :param min_ess:  Maximum value of ESS for any parameter (see ``convergence_type``) below which
        auto-convergence may stop sampling, defaults to 10
    :type min_ess: int, optional
    :param M_adapt: Number of iterations to discard and use to tune the NUTS sampler, defaults to
        500
    :type M_adapt: int, optional
    :param delta: Expected rate of accepted states. Lower values might mean that the sampler
        get's stuck with particular values for more iterations, defaults to 0.6
    :type delta: float, optional
    :param callback: An optional callback of the form ``callback(iter:int, coef_samples:np.ndarray,\
        llk_samples:np.ndarray,rho_samples:np.ndarray|None)`` called every time the chain was
        advanced, defaults to None
    :type callback: Callable | None, optional
    :param n_chains: Number of chains to sample, defaults to 2
    :type n_chains: int, optional
    :param parallelize_chains: Whether to sample chains in parallel, defaults to True
    :type parallelize_chains: bool, optional
    :param auto_converge: Whther to monitor the maximum value of the effective sample size and
        rhat of some or all parameters to automatically determine the number of discarded iterations
        (``M_adapt`` iterations are always discarded). If this is set to True, then the sampler
        will automatically split the current number of samples (starting after iteration
        ``M_adapt``) in half and check the aforementioned convergence statistics on the second half.
        If both criteria are met, the sampler returns the current state of this second half for all
        parameters and the log joint probability, defaults to True
    :type auto_converge: bool, optional
    :param drop_NA: Whether to drop NANs from the model matrices and observation vectors of ``GSMM``
        models, defaults to True
    :type drop_NA: bool, optional
    :param sample_rho: Whether to sample the log regularization parameters as well, defaults to
        False
    :type sample_rho: bool, optional
    :param convergence_type: For which parameters to monitor convergence (**also determines which
        parameteters are considered for the ``auto-convergence`` feature). See the
        :func:`check_convergence` function for details, defaults to 1
    :type convergence_type: int, optional
    :raises ValueError: If the function is called for a model that has not previously been
        estimated for at least a single iteration.
    :return: Three np.arrays, holding the samples of the coefficients
        (dimension: ``(n_chains,n_samples,n_coef)``),
        log joint probability (dimension: ``(n_chains,n_samples,1)``), and
        (optionally) the log regularization parameters (dimension: ``(n_chains,n_samples,n_rho)``)
        or None.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray | None]
    """

    # Make sure model has been estimated
    if model.coef is None:
        raise ValueError(
            "Model must have been estimated before calling ``sample_mssm``!"
        )

    # Extract info
    formulas = model.formulas  # noqa
    family = model.family
    n_coef = len(model.coef)
    n_scale = 0
    n_theta = 0
    orig_scale = 1
    orig_theta = None
    n_lam = len(model.overall_penalties)

    # Now get ys and Xs
    if isinstance(family, Family):
        y = model.formulas[0].y_flat[model.formulas[0].NOT_NA_flat]

        if not model.formulas[0].get_lhs().f is None:
            # Optionally apply function to dep. var. before fitting.
            y = model.formulas[0].get_lhs().f(y)

        ys = [y]
        Xs = [model.get_mmat()]

        orig_scale = family.scale
        if family.twopar:
            _, orig_scale = model.get_pars()
            n_scale = 1

        if isinstance(family, ExtendedFamily):
            orig_theta = family.theta  # noqa
            n_theta = len(family.theta)

    else:
        if isinstance(family, GAMLSSFamily):
            y = model.formulas[0].y_flat[model.formulas[0].NOT_NA_flat]

            if not model.formulas[0].get_lhs().f is None:
                # Optionally apply function to dep. var. before fitting. Not sure why that would be
                # desirable for this model class...
                y = model.formulas[0].get_lhs().f(y)

            ys = [y]
            for _ in range(1, family.n_par):
                ys.append(None)
            Xs = model.get_mmat()

        else:  # Need all y vectors in y, i.e., y is actually ys
            ys = []
            for fi, form in enumerate(model.formulas):

                # Repeated y-variable - don't have to pass all of them
                if (
                    fi > 0
                    and form.get_lhs().variable == model.formulas[0].get_lhs().variable
                ):
                    ys.append(None)
                    continue

                # New y-variable
                if drop_NA:
                    y = form.y_flat[form.NOT_NA_flat]
                else:
                    y = form.y_flat

                # Optionally apply function to dep. var. before fitting. Not sure why that would be
                # desirable for this model class...
                if not form.get_lhs().f is None:
                    y = form.get_lhs().f(y)

                # And collect
                ys.append(y)

            Xs = model.get_mmat(drop_NA=drop_NA)

        # ToDo: Do we want to do anything with dropped terms?
        keep_drop = None
        if model.info.dropped is not None:
            keep = np.array(
                [
                    cidx
                    for cidx in range(model.hessian.shape[1])
                    if cidx not in model.info.dropped
                ]
            )
            keep_drop = (keep, model.info.dropped)  # noqa

    # Get initial penalty matrix
    S_emb, S_pinv, _, FS_use_rank = compute_S_emb_pinv_det(
        len(model.coef), model.overall_penalties, "svd"
    )

    # Can now start building Minv and MLT so that MLT.T@MLT = inv(Minv)
    Minv = (model.lvi.T @ model.lvi).tocsc()
    Lp, Pr, _ = cpp_cholP(Minv)
    Lpinv = compute_Linv(Lp)

    # M = MLT.T@MLT
    MLT = apply_eigen_perm(Pr, Lpinv)

    # Approximate covariance matrix for log lambda parameters
    if sample_rho:
        Vp, Vpreg, Vpr, Vpregr, ep, _ = estimateVp(model)

        eig, U = scp.linalg.eigh(Vpreg)

        # fmt: off
        # Make sure metric is pd..
        eig[eig < 0] = np.power(np.finfo(float).eps, 0.5) * np.max(eig)
        ire = 1 / np.sqrt(eig)
        re = np.sqrt(eig)
        # fmt: on

        Ri = np.diag(ire) @ U.T  # Root of Hp
        Re = np.diag(re) @ U.T  # Root of Vp
        Vpreg = Re.T @ Re

        # Now stack onto Minv and MLT

        # MLT.T@MLT = inv(Minv) defined below
        MLT = scp.sparse.vstack(
            (
                scp.sparse.hstack(
                    (MLT, scp.sparse.csc_matrix((MLT.shape[0], Vp.shape[1])))
                ),
                scp.sparse.hstack(
                    (scp.sparse.csc_matrix((Vp.shape[1], MLT.shape[1])), Ri)
                ),
            ),
            format="csc",
        )

        Minv = scp.sparse.vstack(
            (
                scp.sparse.hstack(
                    (Minv, scp.sparse.csc_matrix((Minv.shape[0], Vp.shape[1])))
                ),
                scp.sparse.hstack(
                    (scp.sparse.csc_matrix((Vp.shape[1], Minv.shape[1])), Vpreg)
                ),
            ),
            format="csc",
        )

    # Can combine dimension of total parameter vector
    n_gamma = n_coef + n_scale + n_theta

    r_pen = None
    if sample_rho:
        n_gamma += n_lam
        r_pen = copy.deepcopy(model.overall_penalties)

    # print(n_coef, n_scale, n_theta, n_gamma)

    proxy_fam = None
    if isinstance(family, GAMLSSFamily):
        proxy_fam = GAMLSSGSMMFamily(family.n_par, family)

    # Can now define wrappers for the joint log-likelihood and gradient + a function to sample
    # momentum variables.
    def llk_wrapper(c: np.ndarray):

        # Split up theta correctly
        coef = c[:n_coef]
        scale = None
        theta = None
        rho = None
        if n_scale > 0:
            scale = c[n_coef : n_coef + n_scale]  # noqa: E203

        elif n_theta > 0:
            # Cannot have scale and theta - scale will be part of theta
            theta = c[n_coef : n_coef + n_theta]  # noqa: E203,F841

        if proxy_fam is not None:
            c_llk = proxy_fam.llk(coef, model.coef_split_idx, ys, Xs)
        else:
            c_llk = family.llk(coef, model.coef_split_idx, ys, Xs)
        if sample_rho:
            rho = c[n_coef + n_scale + n_theta :]  # noqa: E203
        else:

            # log joint is simply proportional to penalized log-likelihood
            penalty = coef.T @ S_emb @ coef
            return c_llk - 0.5 * penalty[0, 0]

        # At this point we know we're sampling rho as well
        # Need: pseudo-determinant of penalty on coef and prior on lam/rho
        for lami, lrho in enumerate(rho):
            r_pen[lami].lam = np.exp(lrho[0])

        S_embr, _, _, _ = compute_S_emb_pinv_det(len(model.coef), r_pen, "svd")

        # Re-parameterize as shown in Wood (2011) to enable stable computation of log(|S_\\lambda|+)
        Sj_reps, _, _, _, S_reps, SJ_term_idx, S_idx, S_coefs, Q_reps, Mp = reparam(
            None, r_pen, None, option=4
        )

        # Now we need to compute log(|S_\\lambda|+), Wood shows that after the re-parameterization
        # log(|S_\\lambda|) can be computed separately from the diagonal or R if Q@R=S_reps[i] for
        # all terms i. Below we compute from the diagonal of the cholesky of the term specific
        # S_reps[i], applying conditioning as shown in Appendix B of Wood (2011).
        lgdetS = 0
        scale = model.scale
        for Si, S_rep in enumerate(S_reps):
            # We need to evaluate log(|S_\\lambda/\\phi|+) after re-parameterization of S_\\lambda
            # (so this will be a regular determinant).
            # We have that (https://en.wikipedia.org/wiki/Determinant):
            #   det(S_\\lambda * 1/\\phi) = (1/\\phi)^p * det(S_\\lambda)
            # taking logs:
            #    log(det(S_\\lambda * 1/\\phi)) = log((1/\\phi)^p) + log(det(S_\\lambda))
            # We know that log(det(S_\\lambda)) is insensitive to whether or not we re-parameterize,
            # so we can simply take S_rep/scale and compute log(det()) for that.
            Sdiag = np.power(np.abs((S_rep / scale).diagonal()), 0.5)
            PI = scp.sparse.diags(1 / Sdiag, format="csc")
            P = scp.sparse.diags(Sdiag, format="csc")

            L, code = cpp_chol(PI @ (S_rep / scale) @ PI)

            if code == 0:
                # fmt: off
                ldetSI = (2 * np.log((L @ P).diagonal()).sum()) * Sj_reps[SJ_term_idx[Si][0]].rep_sj
                # fmt: on
            else:
                warnings.warn(
                    "Cholesky for log-determinant to compute REML failed. Falling back on QR."
                )
                R = np.linalg.qr(S_rep.toarray(), mode="r")
                ldetSI = (
                    np.log(np.abs(R.diagonal())).sum()
                    * Sj_reps[SJ_term_idx[Si][0]].rep_sj
                )

            lgdetS += ldetSI

        # Now adjust c_llk for prior on rho - here uniform
        for rhov in rho:
            lprior = scp.stats.uniform.logpdf(rhov, loc=-10, scale=20)
            c_llk += lprior

        return (c_llk - 0.5 * coef.T @ S_embr @ coef + 0.5 * lgdetS)[0, 0]

    def grad_wrapper(c: np.ndarray):
        # Split up theta correctly
        coef = c[:n_coef]
        scale = None
        theta = None
        rho = None

        if n_scale > 0:
            scale = c[n_coef : n_coef + n_scale]  # noqa: E203,F841

        elif n_theta > 0:
            # Cannot have scale and theta - scale will be part of theta
            theta = c[n_coef : n_coef + n_theta]  # noqa: E203,F841

        if proxy_fam is not None:
            grad = proxy_fam.gradient(coef, model.coef_split_idx, ys, Xs)
        else:
            grad = family.gradient(coef, model.coef_split_idx, ys, Xs)

        if sample_rho:
            rho = c[n_coef + n_scale + n_theta :]  # noqa: E203
        else:
            pgrad = np.array(
                [grad[i] - (S_emb[[i], :] @ coef)[0] for i in range(len(grad))]
            )
            return pgrad.reshape(-1, 1)

        # At this point we know we're sampling rho as well
        # Need: pseudo-determinant of penalty on coef and prior on lam/rho
        for lami, lrho in enumerate(rho):
            r_pen[lami].lam = np.exp(lrho[0])

        S_embr, SJ_pinv, _, FS_use_rank = compute_S_emb_pinv_det(
            len(model.coef), r_pen, "svd"
        )

        pgrad = np.array(
            [grad[i] - (S_embr[[i], :] @ coef)[0] for i in range(len(grad))]
        )

        # Now grad with respect to rhos
        pen_grads = []
        for lami in range(len(rho)):
            lv = r_pen[lami].lam
            if FS_use_rank[lami]:
                tr = r_pen[lami].rank / lv
            else:
                tr = (r_pen[lami].S_J_emb @ SJ_pinv).trace()

            pen_grad = -0.5 * lv * coef.T @ r_pen[lami].S_J_emb @ coef
            det_grad = 0.5 * lv * tr
            pen_grads.extend(pen_grad + det_grad)

        pen_grads = np.array(pen_grads)
        pgrad = np.append(pgrad.reshape(-1, 1), pen_grads, axis=0)
        return pgrad

    def r_sampler():
        # Function to sample momentum variables
        return sample_MVN(1, 0, scale=1, P=None, L=None, LI=MLT)

    # Sample initial coefs for chains (n_coef,n_chains)
    init_coef = sample_MVN(
        n_chains, model.coef.flatten(), 1, L=None, P=None, LI=model.lvi
    )

    init_rho = None
    if sample_rho:
        # Sample initial rhos for chains (n_chains,n_lam)
        init_rho = scp.stats.multivariate_normal.rvs(
            mean=ep.flatten(),
            cov=Vpreg,
            size=n_chains,
        )

        if n_chains == 1:
            init_rho = init_rho.reshape(1, -1)

    # Initialize samplers
    samplers = []
    for chain in range(n_chains):

        # Build combined parameter vector
        gamma = init_coef[:, chain]

        if sample_rho:
            gamma = np.concatenate((gamma, init_rho[chain, :]))

        sampler = mcmc.NUTS(
            n_gamma,
            M_adapt,
            delta,
            gamma.reshape(-1, 1),
            llk_wrapper,
            grad_wrapper,
            r_sampler,
            *map_csc_to_eigen(Minv),
        )

        samplers.append(sampler)

    coef_samples = np.zeros((n_chains, n_iter, n_coef))
    llk_samples = np.zeros((n_chains, n_iter, 1))
    lam_samples = None
    if sample_rho:
        lam_samples = np.zeros((n_chains, n_iter, n_lam))

    manager = None
    if parallelize_chains and n_chains > 1:
        manager = mp.Manager()

    iterator = tqdm(range(n_iter + M_adapt), desc="Warming up...", leave=True)
    for iter in iterator:

        return_dict = manager.dict() if parallelize_chains and n_chains > 1 else dict()

        chains = []

        for chain in range(n_chains):

            args = (chain, return_dict, samplers[chain])

            if parallelize_chains and n_chains > 1:
                chains.append(mp.Process(target=advance_chain_mssm, args=args))
                chains[chain].start()
            else:
                advance_chain_mssm(*args)

        # Collect from chains
        for chain in range(n_chains):

            if parallelize_chains and n_chains > 1:
                chains[chain].join()

            llkprime = return_dict[chain][0]
            gammaprime = return_dict[chain][1]

            if iter >= M_adapt:
                coefprime = gammaprime[:n_coef]
                scaleprime = None
                thetaprime = None
                rhoprime = None
                if n_scale > 0:
                    # fmt:off
                    scaleprime = gammaprime[n_coef : n_coef + n_scale]  # noqa: E203,F841
                    # fmt:on

                elif n_theta > 0:
                    # Cannot have scale and theta - scale will be part of theta
                    # fmt:off
                    thetaprime = gammaprime[n_coef : n_coef + n_theta]  # noqa: E203,F841
                    # fmt:on

                if sample_rho:
                    rhoprime = gammaprime[n_coef + n_scale + n_theta :]  # noqa: E203
                    lam_samples[chain, iter - M_adapt, :] = rhoprime[:, 0]

                coef_samples[chain, iter - M_adapt, :] = coefprime[:, 0]
                llk_samples[chain, iter - M_adapt, 0] = llkprime

        # Convergence:
        if iter >= M_adapt:

            if (iter - M_adapt) == 0:
                iterator.set_description_str(desc="Sampling...", refresh=True)

            if (
                (iter - M_adapt)
                >= (
                    max(16, 2 * 4 * n_chains) if auto_converge else max(8, 4 * n_chains)
                )
                and (iter - M_adapt) % 2 == 0
                and HAS_ARVIZ
            ):

                conv_samples_coef = coef_samples[:, : (iter - M_adapt) + 1, :]
                conv_samples_lam = None
                if sample_rho:
                    conv_samples_lam = lam_samples[:, : (iter - M_adapt) + 1, :]

                if auto_converge:
                    # Base auto convergence check only on samples after discarding half of obs.
                    conv_samples_coef = conv_samples_coef[
                        :, (iter - M_adapt) // 2 :, :  # noqa: E203
                    ]

                    if sample_rho:
                        conv_samples_lam = conv_samples_lam[
                            :, (iter - M_adapt) // 2 :, :  # noqa: E203
                        ]

                ess, rhat, mcse = check_convergence(
                    conv_samples_coef,
                    llk_samples,
                    conv_samples_lam,
                    model,
                    type=convergence_type,
                )

                desc = (
                    f"Sampling... Iter: {iter}, Min. ESS: {np.round(np.min(ess), decimals=2)}, "
                    f"Max. Rhat: {np.round(np.max(rhat), decimals=2)}, "
                    f"Max. MCSE: {np.round(np.max(mcse), decimals=2)}"
                )

                iterator.set_description_str(desc=desc, refresh=True)

                # Auto convergence
                if (
                    (iter - M_adapt) >= 2 * min_iter
                    and np.min(ess) > (min_ess * n_chains)
                    and np.max(rhat) < max_rhat
                    and auto_converge
                ):
                    coef_samples = coef_samples[
                        :,
                        ((iter - M_adapt) // 2) : (iter - M_adapt) + 1,  # noqa: E203
                        :,
                    ]

                    llk_samples = llk_samples[
                        :,
                        ((iter - M_adapt) // 2) : (iter - M_adapt) + 1,  # noqa: E203
                        :,
                    ]

                    if sample_rho:
                        lam_samples = lam_samples[
                            :,
                            ((iter - M_adapt) // 2) : (iter - M_adapt)  # noqa: E203
                            + 1,
                            :,
                        ]

                    desc = (
                        f"Converged! Iter: {iter}, Min. ESS: {np.round(np.min(ess), decimals=2)}, "
                        f"Max. Rhat: {np.round(np.max(rhat), decimals=2)}, "
                        f"Max. MCSE: {np.round(np.max(mcse), decimals=2)}"
                    )

                    iterator.set_description_str(desc=desc, refresh=True)
                    iterator.close()
                    break

            # Callback (optional)
            if callback is not None:
                callback(
                    iter - M_adapt,
                    coef_samples[:, : (iter - M_adapt) + 1, :],
                    llk_samples[:, : (iter - M_adapt) + 1, :],
                    lam_samples[:, : (iter - M_adapt) + 1, :] if sample_rho else None,
                )

    return llk_samples, coef_samples, lam_samples
