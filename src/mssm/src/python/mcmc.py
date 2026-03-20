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
    scale_samples: np.ndarray | None,
    theta_samples: np.ndarray | None,
    llk_samples: np.ndarray | None,
    rho_samples: np.ndarray | None,
    model: GAMMLSS | GSMM | GAMM,
    type: int = 0,
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
    :param scale_samples: Optional sample of log scale parameters for GAMM families as np.array of
        dimension ``(n_chains,n_samples,1)`` or None.
    :type scale_samples: np.ndarray | None
    :param theta_samples: Optional sample of theta parameters for extended families as np.array of
        dimension ``(n_chains,n_samples,1)`` or None.
    :type theta_samples: np.ndarray | None
    :param llk_samples: Optional sample of log joint probability scores as np.array of dimension
        ``(n_chains,n_samples,1)`` or None.
    :type llk_samples: np.ndarray | None
    :param rho_samples: Optional sample of log regularization parameters as np.array of dimension
        ``(n_chains,n_samples,n_rho)`` or None.
    :type rho_samples: np.ndarray | None
    :param model: The model from which we are sampling.
    :type model: GAMMLSS | GSMM | GAMM
    :param type: For which parameters to collect convergence statistics. Type 1 collects statistics
        only for fixed coefficients and those involved in smooth functions with an improper prior
        as well as all scale, theta, and rho parameters and the log joint.
        Type 0 collects statistics for all parameters, defaults to 0
    :type type: int, optional
    :return: Three arrays, holding the ESS, Rhat, and MCSE scores of collected parameters. Order
        is: first (selected) coefficients, then log rho scores, last element will be the log joint
        (if ``llk_samples is not None``).
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """

    ess_all = []
    rhat_all = []
    mcse_all = []
    for samples in [
        coef_samples,
        scale_samples,
        theta_samples,
        rho_samples,
        llk_samples,
    ]:

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
        split_idx = model.coef_split_idx
        if split_idx is None:
            split_idx = []
        split_ess = np.split(ess_all[0], split_idx)
        split_rhat = np.split(rhat_all[0], split_idx)
        split_mcse = np.split(mcse_all[0], split_idx)

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

    # Scale parameters
    if scale_samples is not None:
        ess_sel.extend(ess_all[1])
        rhat_sel.extend(rhat_all[1])
        mcse_sel.extend(mcse_all[1])

    # Theta parameters
    if theta_samples is not None:
        ess_sel.extend(ess_all[2])
        rhat_sel.extend(rhat_all[2])
        mcse_sel.extend(mcse_all[2])

    # Lambda parameters
    if rho_samples is not None:
        ess_sel.extend(ess_all[3])
        rhat_sel.extend(rhat_all[3])
        mcse_sel.extend(mcse_all[3])

    # log joint
    if llk_samples is not None:
        ess_sel.extend(ess_all[4])
        rhat_sel.extend(rhat_all[4])
        mcse_sel.extend(mcse_all[4])

    return np.array(ess_sel), np.array(rhat_sel), np.array(mcse_sel)


def advance_chain_mssm(
    chain_id: int,
    return_dict: dict,
    iter: int,
    M_adapt: int,
    steps: int,
    cllk: float,
    omega: np.ndarray,
    epsilon: float,
    epsilonbar: float,
    Hbar: float,
    mu: float,
    Minv: scp.sparse.csc_array | None,
    Mrows: int | None,
    address_Mdat: str | None,
    address_Mptr: str | None,
    address_Midx: str | None,
    shape_Mdat: tuple[int, int] | None,
    shape_Mptr: tuple[int, int] | None,
    delta: float,
    kappa: float,
    gamma: float,
    t0: float,
    max_j: int,
    llk_fun: Callable,
    grad_fun: Callable,
    r_sampler: Callable,
) -> None:
    """Wrapper function to advancethe state of a NUTS ``chain`` (implemented in c++ class).

    :param chain_id: Id of chain, just an integer >= 0.
    :type chain_id: int
    :param steps: For how many steps to run the chain, just another integer >= 0.
    :type steps: int
    :param return_dict: The shared dictionary in which all states dump the outcome of the next step.
    :type return_dict: dict
    :param chain: Python wrapper around the c++ NUTS class.
    :type chain: Callable
    """

    if Minv is None:
        Mdat_shared = mp.shared_memory.SharedMemory(name=address_Mdat, create=False)
        Mptr_shared = mp.shared_memory.SharedMemory(name=address_Mptr, create=False)
        Midx_shared = mp.shared_memory.SharedMemory(name=address_Midx, create=False)

        Mdata = np.ndarray(shape_Mdat, dtype=np.double, buffer=Mdat_shared.buf)
        Mindptr = np.ndarray(shape_Mptr, dtype=np.int64, buffer=Mptr_shared.buf)
        Mindices = np.ndarray(shape_Mdat, dtype=np.int64, buffer=Midx_shared.buf)

        Minv = scp.sparse.csc_array(
            (Mdata, Mindices, Mindptr), shape=(Mrows, Mrows), copy=False
        )

    llks, omegas, epsilon, epsilonbar, Hbar = mcmc.advance_chain(
        iter,
        M_adapt,
        steps,
        cllk,
        omega,
        *map_csc_to_eigen(Minv),
        epsilon,
        epsilonbar,
        Hbar,
        mu,
        delta,
        kappa,
        gamma,
        t0,
        max_j,
        llk_fun,
        grad_fun,
        r_sampler,
    )

    # print(chain_id, max_j, epsilon, omegas[-6:-4, -1])

    return_dict[chain_id] = [llks, omegas, epsilon, epsilonbar, Hbar]


def sample_mssm(
    model: GAMM | GAMMLSS | GSMM,
    n_iter: int = 10000,
    n_steps: int = 100,
    min_iter: int = 100,
    max_rhat: float = 1.02,
    min_ess: int = 10,
    M_adapt: int = 500,
    delta: float = 0.6,
    kappa: float = 1.0,
    gamma: float = 0.05,
    t0: int = 10,
    max_j: int = 10,
    max_j_adapt: int = 6,
    callback: Callable | None = None,
    n_chains: int = 2,
    parallelize_chains: bool = True,
    auto_converge: bool = True,
    drop_NA: bool = True,
    sample_rho: bool = False,
    convergence_type: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Samples the posterior of any model using a No-U-Turn (NUTS) sampler (Hoffman & Gelman, 2014).

    Supports ``GAMM`` (including those with extended families), ``GAMMLSS``, and ``GSMM`` models.

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

    :param model: The model from which to sample, any ``mssm`` model is supported.
    :type model: GAMM | GAMMLSS | GSMM
    :param n_iter: Number of iterations to sample (after adaptation), defaults to 10000
    :type n_iter: int, optional
    :param n_steps: For how many steps the chains should be advanced before evaluating convergence
        statistics. Since there is some overhead associated with starting up the multi-processing
        pool and copying back and forth from cpp to python, it is generally desirable to set this to
        a reasonably large integer. However, as the cost to compute a single gradient/likelihood
        starts to dominate, it will often be more desirabel to check convergence more frequently.
        Note, that a minimum value of 2 is enforced for this parameter.
        Defaults to 100
    :type n_steps: int, optional
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
    :param kappa: , defaults to 1.0
    :type kappa: float, optional
    :param gamma: , defaults to 0.05
    :type gamma: float, optional
    :param t0: ,defaults to 10
    :type t0: int, optional
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
        :func:`check_convergence` function for details, defaults to 0
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

    # Enforce minimum of 2 for n_steps
    n_steps = max(n_steps, 2)

    # Makes no sense to parallelize
    if parallelize_chains and n_chains == 1:
        parallelize_chains = False

    # Create managers for parallelization
    manager = None
    mem_manager = None
    if parallelize_chains:
        manager = mp.Manager()
        mem_manager = mp.managers.SharedMemoryManager()

    # Extract info
    formulas = model.formulas  # noqa
    deriv_fam = None
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

        if family.twopar:
            # Sample scale parameter as well
            orig_scale = model.scale
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

            deriv_fam = GAMLSSGSMMFamily(family.n_par, family)

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
            deriv_fam = family

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
    S_emb, _, _, _ = compute_S_emb_pinv_det(
        len(model.coef), model.overall_penalties, "svd"
    )

    # Can now start building Minv and MLT so that MLT.T@MLT = inv(Minv)
    Minv = (model.lvi.T @ model.lvi).tocsc() * orig_scale
    Lp, Pr, _ = cpp_cholP(Minv)
    Lpinv = compute_Linv(Lp)

    # M = MLT.T@MLT
    MLT = apply_eigen_perm(Pr, Lpinv)

    if n_scale > 0:

        # Get negative 2nd partial of scale
        nH_scale = -1 * family.d2llkd2lscale(model.coef, ys[0], Xs[0], scale=orig_scale)

        # Enforce PD
        if nH_scale <= 0:
            nH_scale = 1e-4

        # Regularize
        nH_scale += 0.1

        V_scale = 1 / nH_scale

        MLT = scp.sparse.block_array(
            [[MLT, None], [None, np.identity(n_scale) * np.sqrt(nH_scale)]],
            format="csc",
        )

        Minv = scp.sparse.block_array(
            [[Minv, None], [None, np.identity(n_scale) * V_scale]], format="csc"
        )

    if n_theta > 0:

        # Get negative hessian block with respect to theta
        mu = family.link.fi(Xs[0] @ model.coef)
        nH_theta = -1 * family.hessianLTheta(ys[0], mu, theta=orig_theta)

        # Get inverse of nH_theta
        eig, U = scp.linalg.eigh(nH_theta)

        # If hessian is not PD find nearest pd
        thresh = np.power(np.finfo(float).eps, 0.5) * np.max(np.abs(eig))
        eig[eig < thresh] = thresh

        # Regularize
        eig += 0.1

        # ... invert ...
        inH_theta = U @ np.diag([1 / e for e in eig]) @ U.T

        # ... and build root Re_theta so that Re_theta.T@Re_theta = inv(inH_theta) = nH_theta
        Re_theta = np.diag([np.sqrt(e) for e in eig]) @ U.T

        MLT = scp.sparse.block_array([[MLT, None], [None, Re_theta]], format="csc")
        Minv = scp.sparse.block_array([[Minv, None], [None, inH_theta]], format="csc")

    # Approximate covariance matrix for log lambda parameters
    if sample_rho:
        Vp, Vpreg, Vpr, Vpregr, ep, _ = estimateVp(model)

        # Correct for scaling for GAMMs
        if isinstance(family, Family) and family.twopar is True:
            ep = np.log(np.exp(ep) / orig_scale)

            uspen = copy.deepcopy(model.overall_penalties)
            for pidx in range(len(model.overall_penalties)):
                uspen[pidx].lam = np.exp(ep[pidx])

            S_emb, _, _, _ = compute_S_emb_pinv_det(len(model.coef), uspen, "svd")

        eig, U = scp.linalg.eigh(Vpreg)

        # fmt: off
        # Make sure metric is pd..
        thresh = np.power(np.finfo(float).eps, 0.5) * np.max(eig)
        eig[eig < thresh] = thresh
        ire = 1 / np.sqrt(eig)
        re = np.sqrt(eig)
        # fmt: on

        Ri = np.diag(ire) @ U.T  # Root of Hp
        Re = np.diag(re) @ U.T  # Root of Vp
        Vpreg = Re.T @ Re

        # Now stack onto Minv and MLT

        # MLT.T@MLT = inv(Minv) defined below
        MLT = scp.sparse.block_array([[MLT, None], [None, Ri]], format="csc")

        Minv = scp.sparse.block_array([[Minv, None], [None, Vpreg]], format="csc")

    # Optionally create shared memory object for parallelization of Minv
    if parallelize_chains:
        Mrows, _, _, Mdata, Mindptr, Mindices = map_csc_to_eigen(Minv)
        shape_Mdat = Mdata.shape
        shape_Mptr = Mindptr.shape

        # Start memory manager
        mem_manager.start()

        dat_mem = mem_manager.SharedMemory(Mdata.nbytes)
        dat_shared = np.ndarray(Mdata.shape, dtype=np.double, buffer=dat_mem.buf)
        dat_shared[:] = Mdata[:]

        ptr_mem = mem_manager.SharedMemory(Mindptr.nbytes)
        ptr_shared = np.ndarray(Mindptr.shape, dtype=np.int64, buffer=ptr_mem.buf)
        ptr_shared[:] = Mindptr[:]

        idx_mem = mem_manager.SharedMemory(Mindices.nbytes)
        idx_shared = np.ndarray(Mindices.shape, dtype=np.int64, buffer=idx_mem.buf)
        idx_shared[:] = Mindices[:]

    # Can combine dimension of total parameter vector
    n_omega = n_coef + n_scale + n_theta

    r_pen = None
    if sample_rho:
        n_omega += n_lam
        if isinstance(family, Family) and family.twopar is True:
            r_pen = uspen
        else:
            r_pen = copy.deepcopy(model.overall_penalties)

    # print(n_coef, n_scale, n_theta, n_omega)
    # print(n_omega, MLT.shape, Minv.shape)

    # print(
    #    "M",
    #    np.abs((MLT.T @ MLT @ Minv) - np.identity(n_omega)).max(),
    # )

    # Can now define wrappers for the joint log-likelihood and gradient + a function to sample
    # momentum variables.
    def llk_wrapper(c: np.ndarray):

        # Split up theta correctly
        coef = c[:n_coef]
        scale = 1
        theta = None
        rho = None
        if n_scale > 0:
            scale = np.exp(c[n_coef : n_coef + n_scale])[0, 0]  # noqa: E203

        elif n_theta > 0:
            # Cannot have scale and theta - scale will be part of theta
            theta = c[n_coef : n_coef + n_theta]  # noqa: E203,F841

        if isinstance(family, Family):
            mu = family.link.fi(Xs[0] @ coef)

            if isinstance(family, ExtendedFamily):
                c_llk = family.llk(ys[0], mu, theta=theta)
            else:
                if family.twopar is True:
                    c_llk = family.llk(ys[0], mu, scale=scale)
                else:
                    c_llk = family.llk(ys[0], mu)
        else:
            c_llk = deriv_fam.llk(coef, model.coef_split_idx, ys, Xs)

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
        for Si, S_rep in enumerate(S_reps):
            Sdiag = np.power(np.abs((S_rep).diagonal()), 0.5)
            PI = scp.sparse.diags(1 / Sdiag, format="csc")
            P = scp.sparse.diags(Sdiag, format="csc")

            L, code = cpp_chol(PI @ (S_rep) @ PI)

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
            lprior = scp.stats.uniform.logpdf(rhov, loc=-20, scale=40)
            c_llk += lprior

        return (c_llk - 0.5 * coef.T @ S_embr @ coef + 0.5 * lgdetS)[0, 0]

    def grad_wrapper(c: np.ndarray):
        # Split up theta correctly
        coef = c[:n_coef]
        scale = 1
        theta = None
        rho = None

        if n_scale > 0:
            scale = np.exp(c[n_coef : n_coef + n_scale])[0, 0]  # noqa: E203,F841

        elif n_theta > 0:
            # Cannot have scale and theta - scale will be part of theta
            theta = c[n_coef : n_coef + n_theta]  # noqa: E203,F841

        if isinstance(family, Family):
            mu = family.link.fi(Xs[0] @ coef)

            if isinstance(family, ExtendedFamily):
                d1 = family.dDdmu(ys[0], mu, theta=theta)
                ld1 = family.link.dy1(d1)
                grad = -0.5 * ((d1 / ld1).T @ Xs[0]).T
                grad_theta = family.gradientLTheta(ys[0], mu, theta=theta)
            else:
                if family.twopar is True:
                    grad = family.dllkdcoef(coef, ys[0], Xs[0], scale=scale)
                    grad_scale = np.array(
                        [family.dllkdlscale(coef, ys[0], Xs[0], scale=scale)]
                    ).reshape(-1, 1)
                else:
                    grad = family.dllkdcoef(coef, ys[0], Xs[0])
        else:
            grad = deriv_fam.gradient(coef, model.coef_split_idx, ys, Xs)

        if sample_rho:
            rho = c[n_coef + n_scale + n_theta :]  # noqa: E203

            # At this point we know we're sampling rho as well
            # Need: pseudo-determinant of penalty on coef and prior on lam/rho
            for lami, lrho in enumerate(rho):
                r_pen[lami].lam = np.exp(lrho[0])

            S_embr, SJ_pinv, _, FS_use_rank = compute_S_emb_pinv_det(
                len(model.coef), r_pen, "svd"
            )

            pgrad = np.array(
                [grad[i] - (S_embr[[i], :] @ coef)[0] for i in range(len(grad))]
            ).reshape(-1, 1)

        else:
            # Can compute pgrad directly from S_emb
            pgrad = np.array(
                [grad[i] - (S_emb[[i], :] @ coef)[0] for i in range(len(grad))]
            ).reshape(-1, 1)

        # Handle extra un-penalized parameters
        if isinstance(family, Family):
            if isinstance(family, ExtendedFamily):
                pgrad = np.concatenate((pgrad, grad_theta), axis=0)
            elif family.twopar is True:
                pgrad = np.concatenate((pgrad, grad_scale), axis=0)

        # Can return here if not sampling rho
        if sample_rho is False:
            return pgrad

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
        pgrad = np.append(pgrad, pen_grads, axis=0)
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

        # Make sure initial value for rho is valid under prior...
        init_rho[init_rho <= -20] = -19
        init_rho[init_rho >= 20] = 19

        if n_chains == 1:
            init_rho = init_rho.reshape(1, -1)

    # Initialize samplers
    omegas = []
    cLs = []
    epsilons = []
    epsilonbars = []
    Hbars = []
    mus = []

    for chain in range(n_chains):

        # Build combined parameter vector
        omega = np.asfortranarray(init_coef[:, chain])

        if n_scale > 0:
            omega = np.concatenate((omega, np.array([np.log(orig_scale)])))
        elif n_theta > 0:
            omega = np.concatenate((omega, orig_theta.flatten()))

        if sample_rho:
            omega = np.concatenate((omega, init_rho[chain, :]))

        # Keep track of current vector of parameters
        omega = omega.reshape(-1, 1)
        omegas.append(omega)

        # And log-likelihood
        cLs.append(llk_wrapper(omega))

        # Initialize epsilon and mu per chain
        epsilons.append(
            mcmc.find_reasonable_epsilon(
                omega,
                grad_wrapper(omega),
                *map_csc_to_eigen(Minv),
                cLs[chain],
                llk_wrapper,
                grad_wrapper,
                r_sampler,
            )
        )
        epsilonbars.append(1)
        Hbars.append(0)

        mus.append(np.log(10 * epsilons[chain]))

    coef_samples = np.zeros((n_chains, n_iter, n_coef))
    llk_samples = np.zeros((n_chains, n_iter, 1))
    lam_samples = None
    scale_samples = None
    theta_samples = None

    if n_scale > 0:
        scale_samples = np.zeros((n_chains, n_iter, n_scale))

    if n_theta > 0:
        theta_samples = np.zeros((n_chains, n_iter, n_theta))

    if sample_rho:
        lam_samples = np.zeros((n_chains, n_iter, n_lam))

    pbar = tqdm(total=n_iter + M_adapt, desc="Warming up...", leave=True)
    iter = 0
    while iter < (n_iter + M_adapt):

        # Calculate steps to take
        if iter < M_adapt:
            if iter + n_steps <= M_adapt:
                steps = n_steps
            else:
                steps = M_adapt - iter
        else:
            if iter + n_steps <= (n_iter + M_adapt):
                steps = n_steps
            else:
                steps = (n_iter + M_adapt) - iter

        # print(iter, steps, M_adapt, n_iter)

        return_dict = manager.dict() if parallelize_chains else dict()

        chains = []

        for chain in range(n_chains):

            args = (
                chain,
                return_dict,
                iter,
                M_adapt,
                steps,
                cLs[chain],
                omegas[chain],
                epsilons[chain],
                epsilonbars[chain],
                Hbars[chain],
                mus[chain],
                None if parallelize_chains else Minv,
                Mrows if parallelize_chains else None,
                dat_mem.name if parallelize_chains else None,
                ptr_mem.name if parallelize_chains else None,
                idx_mem.name if parallelize_chains else None,
                shape_Mdat if parallelize_chains else None,
                shape_Mptr if parallelize_chains else None,
                delta,
                kappa,
                gamma,
                t0,
                max_j_adapt if iter < M_adapt else max_j,
                llk_wrapper,
                grad_wrapper,
                r_sampler,
            )

            if parallelize_chains:
                chains.append(mp.Process(target=advance_chain_mssm, args=args))
                chains[chain].start()
            else:
                advance_chain_mssm(*args)

        # Collect from chains
        for chain in range(n_chains):

            if parallelize_chains:
                chains[chain].join()

            llkprime = return_dict[chain][0]
            omegaprime = return_dict[chain][1]

            # Update latest parameter vector, llks, etc.
            omegas[chain] = omegaprime[:, [-1]]
            cLs[chain] = llkprime[-1]
            epsilons[chain] = return_dict[chain][2]
            epsilonbars[chain] = return_dict[chain][3]
            Hbars[chain] = return_dict[chain][4]

            # Store only after adaptation phase
            if iter >= M_adapt:
                sidx = iter - M_adapt
                eidx = sidx + steps
                coefprime = omegaprime[:n_coef, :]
                scaleprime = None
                thetaprime = None
                rhoprime = None
                if n_scale > 0:
                    # fmt:off
                    scaleprime = omegaprime[n_coef : n_coef + n_scale, :]  # noqa: E203,F841
                    scale_samples[chain, sidx:eidx, :] = scaleprime.T
                    # fmt:on

                elif n_theta > 0:
                    # Cannot have scale and theta - scale will be part of theta
                    # fmt:off
                    thetaprime = omegaprime[n_coef : n_coef + n_theta, :]  # noqa: E203,F841
                    theta_samples[chain, sidx:eidx, :] = thetaprime.T
                    # fmt:on

                if sample_rho:
                    rhoprime = omegaprime[n_coef + n_scale + n_theta :, :]  # noqa: E203
                    lam_samples[chain, sidx:eidx, :] = rhoprime.T

                coef_samples[chain, sidx:eidx, :] = coefprime.T
                llk_samples[chain, sidx:eidx, 0] = llkprime

        iter += steps
        pbar.update(steps)

        if (iter - M_adapt) == 0:
            # Average epsilons across chains, since we are done adapting.
            m_epsilon = np.mean(epsilonbars)
            epsilons = [m_epsilon for _ in range(n_chains)]
            epsilonbars = epsilons

            pbar.set_description_str(desc="Sampling...", refresh=True)

        # Convergence:
        if iter > M_adapt:

            if ((iter - M_adapt) // 4) > (2 * n_chains) and HAS_ARVIZ:

                conv_coef_samples = coef_samples[:, : (iter - M_adapt), :]
                conv_llk_samples = llk_samples[:, : (iter - M_adapt), :]
                conv_scale_samples = None
                conv_theta_samples = None
                conv_lam_samples = None

                if n_scale > 0:
                    conv_scale_samples = scale_samples[:, : (iter - M_adapt), :]

                if n_theta > 0:
                    conv_theta_samples = theta_samples[:, : (iter - M_adapt), :]

                if sample_rho:
                    conv_lam_samples = lam_samples[:, : (iter - M_adapt), :]

                if auto_converge:
                    # Base auto convergence check only on samples after discarding half of obs.
                    conv_coef_samples = conv_coef_samples[
                        :, (iter - M_adapt) // 2 :, :  # noqa: E203
                    ]

                    if n_scale > 0:
                        conv_scale_samples = conv_scale_samples[
                            :, (iter - M_adapt) // 2 :, :  # noqa: E203
                        ]

                    if n_theta > 0:
                        conv_theta_samples = conv_theta_samples[
                            :, (iter - M_adapt) // 2 :, :  # noqa: E203
                        ]

                    if sample_rho:
                        conv_lam_samples = conv_lam_samples[
                            :, (iter - M_adapt) // 2 :, :  # noqa: E203
                        ]

                ess, rhat, mcse = check_convergence(
                    conv_coef_samples,
                    conv_scale_samples,
                    conv_theta_samples,
                    conv_llk_samples,
                    conv_lam_samples,
                    model,
                    type=convergence_type,
                )

                desc = (
                    f"Sampling... Iter: {iter-M_adapt}, "
                    f"Min. ESS: {np.round(np.min(ess), decimals=2)}, "
                    f"Max. Rhat: {np.round(np.max(rhat), decimals=2)}, "
                    f"Max. MCSE: {np.round(np.max(mcse), decimals=2)}"
                )

                pbar.set_description_str(desc=desc, refresh=True)

                # Auto convergence
                if (
                    (iter - M_adapt) >= 2 * min_iter
                    and np.min(ess) > (min_ess * n_chains)
                    and np.max(rhat) < max_rhat
                    and auto_converge
                ):
                    coef_samples = coef_samples[
                        :,
                        ((iter - M_adapt) // 2) : (iter - M_adapt),  # noqa: E203
                        :,
                    ]

                    llk_samples = llk_samples[
                        :,
                        ((iter - M_adapt) // 2) : (iter - M_adapt),  # noqa: E203
                        :,
                    ]

                    if n_scale > 0:
                        scale_samples = scale_samples[
                            :,
                            ((iter - M_adapt) // 2) : (iter - M_adapt),  # noqa: E203
                            :,
                        ]

                    if n_theta > 0:
                        theta_samples = theta_samples[
                            :,
                            ((iter - M_adapt) // 2) : (iter - M_adapt),  # noqa: E203
                            :,
                        ]

                    if sample_rho:
                        lam_samples = lam_samples[
                            :,
                            ((iter - M_adapt) // 2) : (iter - M_adapt),  # noqa: E203
                            :,
                        ]

                    desc = (
                        f"Converged! Iter: {iter-M_adapt}, "
                        f"Min. ESS: {np.round(np.min(ess), decimals=2)}, "
                        f"Max. Rhat: {np.round(np.max(rhat), decimals=2)}, "
                        f"Max. MCSE: {np.round(np.max(mcse), decimals=2)}"
                    )

                    pbar.set_description_str(desc=desc, refresh=True)
                    pbar.close()
                    if parallelize_chains:
                        mem_manager.shutdown()
                    break

            # Callback (optional)
            if callback is not None:
                callback(
                    iter - M_adapt,
                    coef_samples[:, : (iter - M_adapt), :],
                    scale_samples[:, : (iter - M_adapt), :] if n_scale > 0 else None,
                    theta_samples[:, : (iter - M_adapt), :] if n_theta > 0 else None,
                    llk_samples[:, : (iter - M_adapt), :],
                    lam_samples[:, : (iter - M_adapt), :] if sample_rho else None,
                )

    if parallelize_chains:
        mem_manager.shutdown()

    return llk_samples, coef_samples, scale_samples, theta_samples, lam_samples
