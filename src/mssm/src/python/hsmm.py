import numpy as np
import scipy as scp
from mssm.src.python.gamm_solvers import deriv_transform_mu_eta, cpp_backsolve_tr
from itertools import repeat
import warnings
import hsmm
from .exp_fam import (
    GAMLSSFamily,
    GSMMFamily,
    MultiGauss,
    MULNOMLSS,
    Link,
)

HAS_MP = True
try:
    import multiprocess as mp
except ImportError:
    warnings.warn(
        "Multi-processing hsmm computations will require the `multiprocess` package."
    )
    HAS_MP = False


def _split_matrices(
    ys: list[np.ndarray | None],
    Xs: list[scp.sparse.csc_array],
    shared_pars: list[int],
    shared_m: bool,
    obs_fams: list[list[GAMLSSFamily | GSMMFamily]] | None,
    d_fams: list[GAMLSSFamily] | None,
    sid: np.ndarray,
    tid: np.ndarray | None,
    n_S: int,
    M: int,
    model_T: bool,
    model_pi: bool,
    starts_with_first: bool,
    Lrhoi: scp.sparse.csc_array | None = None,
) -> tuple[list[np.ndarray | None], list[scp.sparse.csc_array | None]]:
    """Splits the model matrices and vectors of observations into series-specific versions.

    Internal function.

    :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as
        ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the
        actual observed data is passed along via the first formula (so it is stored in ``ys[0]``).
        If multiple formulas have the same ``lhs.variable`` as this first formula, then ``ys``
        contains ``None`` at their indices to save memory.
    :type ys: [np.ndarray or None]
    :param Xs: A list of sparse model matrices per likelihood parameter.
    :type Xs: [scp.sparse.csc_array]
    :param shared_pars: A list containing indices of emission distribution parameters for which
        at least one (in case ``shared_m is False`` one for each of the ``M`` signals) formula
        has been provided containing terms to be shared between the ``n_S`` states.
    :type shared_pars: list[int]
    :param shared_m: Bool indicating whether any shared terms are not just shared between states but
        also between the ``M`` signals.
    :type shared_m: bool
    :param obs_fams: Distribution of emissions - one per state
    :type obs_fams: list[list[GAMLSSFamily]] | None
    :param d_fams: Distribution of state durations - one per state
    :type d_fams: list[GAMLSSFamily] | None
    :param sid: Array holding the first sample of each series in the data. Can be used to split the
        observation vectors and model matrices into time-series specific versions
    :type sid: np.ndarray
    :param tid: Often, the model matrices for the models of the duration distribution parameters
        will have fewer rows than those of the observation models (because covariates only
        vary with trials). The optional array ``tid`` can then hold indices corresponding to the
        onset of a row-block of a new trial in the model matrices of the duration models. Can then
        be used to split the duration model matrices into trial specific versions. If this is set
        to ``None``, it will be set to ``sid``.
    :type tid: np.ndarray | None
    :param n_S: Number of latent states.
    :type n_S: int
    :param M: Number of signals recorded if the HSMM is multivariate. All signals are assumed to be
        independent from one another given covariates and coefficients.
    :type M: int
    :param model_T: Whether to model the transition matrix or not
    :type model_T: bool
    :param model_pi: Whether to model the initial distribution of states vector or not
    :type model_pi: bool
    :param starts_with_first: Whether the first state starts with the first observation or might
        have been going on for a max duration of ``D``.
    :type starts_with_first: bool
    :param Lrhoi: Inverse of transpose of Cholesky of banded covariance matrix of an ar model of the
        residuals for a HMP-like HSMM. Defaults to None - indicating no ar model.
    :type Lrhoi: scp.sparse.csc_array | None, optional
    :return: A list of series-specific splits of ``ys`` and ``Xs``.
    :rtype: tuple[list[np.ndarray|None],list[scp.sparse.csc_array|None]]
    """

    # Split up ys, Xs
    split_Xs = []
    split_Ys = []

    idx = 0  # Keep track of indices for ys and Xs

    # First check for part of models shared between states (and m)
    for par in shared_pars:

        midx = 1
        if shared_m is False:
            midx = M

        # Only need to collect Xs for all par and m. For ys we only need first par and m
        # since functions expect first observation vector to be not None to identify number
        # of time-points.
        for m in range(midx):
            X_split = None
            if Xs[idx] is not None:
                Xidx = np.arange(Xs[idx].shape[0])
                Xmj = Xs[idx]
                X_split = [Xmj[split, :] for split in np.split(Xidx, sid[1:])]

            y_split = None
            if par == 0 and m == 0:
                ymj = ys[0] if Lrhoi is None else Lrhoi.T @ ys[0]
                y_split = np.split(ymj, sid[1:])

            split_Xs.append(X_split)
            split_Ys.append(y_split)
            idx += 1

    # Now state specific models
    for j in range(n_S):

        iterM = M

        # If emission model is multivariate Gaussian, then it has M n_par!
        if isinstance(obs_fams[j][0], MultiGauss):
            iterM = 1

        for m in range(iterM):

            jmbFam = obs_fams[j][m]  # Get family associated with state j and signal m

            if jmbFam is not None:

                for _ in range(jmbFam.n_par):

                    X_split = None
                    if Xs[idx] is not None:
                        Xidx = np.arange(Xs[idx].shape[0])
                        Xmj = Xs[idx]
                        X_split = [Xmj[split, :] for split in np.split(Xidx, sid[1:])]

                    y_split = None
                    if ys[idx] is not None:
                        ymj = ys[idx] if Lrhoi is None else Lrhoi.T @ ys[idx]
                        y_split = np.split(ymj, sid[1:])

                    split_Xs.append(X_split)
                    split_Ys.append(y_split)
                    idx += 1

            else:
                # Only have observation models for events = n_S-1
                if j % 2 == 1:
                    X_split = None
                    if Xs[idx] is not None:
                        Xidx = np.arange(Xs[idx].shape[0])
                        Xmj = Xs[idx]
                        X_split = [Xmj[split, :] for split in np.split(Xidx, sid[1:])]

                    y_split = None
                    if ys[idx] is not None:
                        ymj = ys[idx] if Lrhoi is None else Lrhoi.T @ ys[idx]
                        y_split = np.split(ymj, sid[1:])

                    split_Xs.append(X_split)
                    split_Ys.append(y_split)
                    idx += 1

    # Now state duration probabilities #
    for j in range(n_S * (1 if starts_with_first else 2)):
        jdFam = d_fams[j]

        if jdFam is not None:
            for _ in range(jdFam.n_par):
                X_split = None
                if Xs[idx] is not None:
                    Xidx = np.arange(Xs[idx].shape[0])
                    X_split = [Xs[idx][split, :] for split in np.split(Xidx, tid[1:])]

                y_split = np.split(ys[idx], tid[1:]) if ys[idx] is not None else None
                split_Xs.append(X_split)
                split_Ys.append(y_split)
                idx += 1

    # Now state transition probabilities #
    if model_T and n_S > 2:
        for j in range(n_S):

            for _ in range(n_S - 2):
                X_split = None
                if Xs[idx] is not None:
                    Xidx = np.arange(Xs[idx].shape[0])
                    X_split = [Xs[idx][split, :] for split in np.split(Xidx, tid[1:])]

                y_split = np.split(ys[idx], tid[1:]) if ys[idx] is not None else None
                split_Xs.append(X_split)
                split_Ys.append(y_split)
                idx += 1

    # Now initial state probabilities #
    if model_pi:
        for _ in range(n_S - 1):
            X_split = None
            if Xs[idx] is not None:
                Xidx = np.arange(Xs[idx].shape[0])
                X_split = [Xs[idx][split, :] for split in np.split(Xidx, tid[1:])]

            y_split = np.split(ys[idx], tid[1:]) if ys[idx] is not None else None
            split_Xs.append(X_split)
            split_Ys.append(y_split)
            idx += 1

    return split_Ys, split_Xs


def _compute_series_probs(
    coef: np.ndarray,
    coef_split_idx: list[int],
    shared_pars: list[int],
    shared_m: bool,
    ys: list[np.ndarray | None],
    Xs: list[scp.sparse.csc_array],
    n_S: int,
    obs_fams: list[list[GAMLSSFamily]] | None,
    d_fams: list[GAMLSSFamily] | None,
    D: int,
    M: int,
    event_width: int | None,
    event_template: np.ndarray | None,
    scale: float | None,
    log: bool,
    model_t: bool,
    model_pi: bool,
    starts_with_first: bool,
    build_mat_idx: list[int] | None,
    is_hmp: bool = False,
    hmp_fam: str = "Exponential",
    rho: float | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Computes observation, duration, state transition, and initial state (log) probabilities for
    individual time-series.

    Internal function.

    :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not be
        flattened!).
    :type coef: np.ndarray
    :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
        sub-sets associated with each paramter of the llk.
    :type coef_split_idx: list[int]
    :param shared_pars: A list containing indices of emission distribution parameters for which
        at least one (in case ``shared_m is False`` one for each of the ``M`` signals) formula
        has been provided containing terms to be shared between the ``n_S`` states.
    :type shared_pars: list[int]
    :param shared_m: Bool indicating whether any shared terms are not just shared between states but
        also between the ``M`` signals.
    :type shared_m: bool
    :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as
        ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the actual
        observed data is passed along via the first formula (so it is stored in ``ys[0]``). If
        multiple formulas have the same ``lhs.variable`` as this first formula, then ``ys``
        contains ``None`` at their indices to save memory.
    :type ys: [np.ndarray or None]
    :param Xs: A list of sparse model matrices per likelihood parameter.
    :type Xs: [scp.sparse.csc_array]
    :param n_S: Number of latent states.
    :type n_S: int
    :param obs_fams: Distribution of emissions - one per state
    :type obs_fams: list[list[GAMLSSFamily]] | None
    :param d_fams: Distribution of state durations - one per state
    :type d_fams: list[GAMLSSFamily]
    :param D: Max duration a state can take on (**in samples**)
    :type D: int
    :param M: Number of signals recorded if the HSMM is multivariate. All signals are assumed to be
        independent from one another given covariates and coefficients.
    :type M: int
    :param event_width: Width of the HMP event pattern in samples (or ``None`` if a standard hsmm
        is to be estimated).
    :type event_width: int | None
    :param scale: Scale parameter to use for the the observation probabilities under a hmp model
        (can be set to None for regular hsmms).
    :type scale: float | None
    :param log: Bool indicating whether to compute probabilities or log-probabilities.
    :type log: bool
    :param model_T: Whether to model the transition matrix or not
    :type model_T: bool
    :param model_pi: Whether to model the initial distribution of states vector or not
    :type model_pi: bool
    :param starts_with_first: Whether the first state starts with the first observation or might
        have been going on for a max duration of ``D``.
    :type starts_with_first: bool
    :param build_mat_idx: If not all matrices are built explicitly, this has to be a list of indices
        per parameter modeled, pointing at the index of ``Xs`` holding the matrix that should be
        used by that parameter.
    :type build_mat_idx: list[int] | None
    :param is_hmp: If the model is a hmp or not.
    :type is_hmp: bool, optional
    :param rho: Weight of an ar model of the residuals for a HMP-like HSMM. Defaults to
        None - indicating no ar model.
    :type rho: float | None, optional
    :raises ValueError: _description_
    :return: _description_
    :rtype: tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]
    """
    # Compute (log) probs for an individual series

    # First fix ys and Xs for Nones
    yfix = [y if y is not None else ys[0] for y in ys]
    ys = yfix

    Xfix = [X if X is not None else Xs[build_mat_idx[xi]] for xi, X in enumerate(Xs)]
    Xs = Xfix

    # Extract some extra info and define storage
    n_T = len(ys[0])  # Number of time-points in this series

    # We can pre-compute the duration probabilities
    ds = np.zeros((D - 1, n_S * (1 if starts_with_first else 2)), order="F")
    d_val = np.arange(1, D).reshape(-1, 1)

    # We can also pre-compute the observation probabilities
    y_mat = None
    mus = None
    jbs = None
    bs = None
    cbs = None

    y_mat = np.zeros((n_T, M), order="F")
    mus = np.zeros((n_T, M, n_S), order="F")
    mus_d = []

    if is_hmp is False:
        jbs = np.zeros((n_T, n_S, M), order="F") - np.inf
        cbs = np.zeros((n_T, n_S, M), order="F") - np.inf
        bs = np.zeros((n_T, n_S), order="F")
    else:
        jbs = np.zeros((n_T, M, event_width, n_S), order="F") - np.inf
        cbs = np.zeros((n_T, M, event_width, n_S), order="F") - np.inf
        bs = np.zeros((n_T, event_width, n_S), order="F")

        # Compute weights from rho
        if rho is not None:

            d0 = 1 / np.sqrt(1 - np.power(rho, 2))  # weight current mean
            d1 = -rho / np.sqrt(1 - np.power(rho, 2))  # weight previous mean

    split_coef = np.split(coef, coef_split_idx)

    # First observation probabilities #

    if is_hmp:
        # Initialize some density/probability functions for hmp-like models
        hmp_code = 2 if hmp_fam == "Gaussian" else 1

        # Define cumprob callback function
        def lcp_wrapper(y: float, pred: float, scale: float):

            # Exponential case
            if hmp_code == 1:
                return scp.stats.expon.logcdf(np.power(y - pred, 2), scale=scale)

            # Gaussian case
            elif hmp_code == 2:
                return scp.stats.norm.logcdf(y, loc=pred, scale=np.power(scale, 0.5))

        # And density callback
        def ld_wrapper(y: float, pred: float, scale: float):
            # Exponential case
            if hmp_code == 1:
                return scp.stats.expon.logpdf(np.power(y - pred, 2), scale=scale)

            # Gaussian case
            elif hmp_code == 2:
                return scp.stats.norm.logpdf(y, loc=pred, scale=np.power(scale, 0.5))

    idx = 0  # Keep track of indices for ys and Xs

    # First check for part of emission models shared between states (and m)
    shared_eta = []  # Holds list per par, each list contains at least one eta vector.
    for par in shared_pars:

        midx = 1
        if shared_m is False:
            midx = M

        shared_eta.append([])

        # Compute shared part of eta per parameter and m
        for m in range(midx):
            shared_eta[par].append(Xs[idx] @ split_coef[idx])
            idx += 1

    # Extract (potential) thetas for multivariate observation models
    thetas = split_coef[-1].flatten()
    theta_idx = 0

    for j in range(n_S):

        # Multivariate gaussian emission
        if isinstance(obs_fams[j][0], MultiGauss):

            jmbFam = obs_fams[j][0]

            # Get thetas for this family
            thetas_j = thetas[theta_idx : theta_idx + jmbFam.extra_coef]  # noqa: E203

            Y = np.concatenate(ys[idx : idx + jmbFam.n_par], axis=1)  # noqa: E203

            etasbjm = []
            musbjm = []

            for par in range(jmbFam.n_par):
                etasbjm.append(Xs[idx] @ split_coef[idx])

                # Add shared eta to etasbjm
                if 0 in shared_pars:
                    if shared_m:
                        etasbjm[-1] += shared_eta[0][0]
                    else:
                        etasbjm[-1] += shared_eta[0][par]

                musbjm.append(jmbFam.links[par].fi(etasbjm[-1]))
                idx += 1

            musbjm = np.concatenate(musbjm, axis=1)

            # Get transpose of Cholesky of precision
            R, logdet = jmbFam.getR(thetas_j)

            # yR is of shape n * m
            yR = (Y - musbjm) @ R.T

            # Log-probability of emission vector
            bs[:, j] = -0.5 * (yR * yR).sum(axis=1) + logdet
            mus[:, :, j] = musbjm

            # For m-specific (cumulative) log-probs we need to trick a bit

            # First get deviance residuals
            res = yR

            # Now each column should be approximately N(0,1)
            for m in range(jmbFam.n_par):

                lpb = scp.stats.norm.logpdf(res[:, m])
                lcpb = scp.stats.norm.logcdf(res[:, m])

                jbs[:, j, m] = lpb.flatten()
                cbs[:, j, m] = lcpb.flatten()

                if log is False:
                    # Transform log probs to probs
                    jbs[:, j, m] = np.exp(jbs[:, j, m])
                    cbs[:, j, m] = np.exp(cbs[:, j, m])

            if j == 0:
                y_mat[:] = Y

            # Update theta index
            theta_idx += jmbFam.extra_coef

        # Multiple independent emission signals
        else:

            for m in range(M):
                # Get family associated with state j and signal m
                jmbFam = obs_fams[j][m]

                if jmbFam is not None:

                    yJM = ys[idx]

                    etasbjm = []
                    musbjm = []

                    for par in range(jmbFam.n_par):
                        etasbjm.append(Xs[idx] @ split_coef[idx])

                        # Add shared eta to etasbjm
                        if par in shared_pars:
                            if shared_m is False:
                                etasbjm[-1] += shared_eta[par][m]
                            else:
                                etasbjm[-1] += shared_eta[par][0]

                        musbjm.append(jmbFam.links[par].fi(etasbjm[-1]))
                        idx += 1

                    lpb = jmbFam.lp(yJM, *musbjm)
                    lcpb = jmbFam.lcp(yJM, *musbjm)
                    mus[:, m, j] = musbjm[0][:, 0]

                    if j == 0:
                        y_mat[:, m] = yJM[:, 0]

                    jbs[:, j, m] = lpb.flatten()
                    cbs[:, j, m] = lcpb.flatten()

                    # All signals are assumed independent given state, covariates, and coefficients
                    bs[:, j] += lpb.flatten()

                    if log is False:
                        # Transform log probs to probs
                        jbs[:, j, m] = np.exp(jbs[:, j, m])
                        cbs[:, j, m] = np.exp(cbs[:, j, m])

                # Assume HMP/HSMMMVPA pattern for observation model
                else:

                    # Bumps
                    if j % 2 == 1:  #

                        yJM = ys[idx]

                        etasbjm = [Xs[idx] @ split_coef[idx]]

                        # Add shared eta to etasbjm
                        if 0 in shared_pars:
                            if shared_m is False:
                                etasbjm[-1] += shared_eta[0][m]
                            else:
                                etasbjm[-1] += shared_eta[0][0]

                        musbjm = etasbjm  # Assume identity link
                        mus[:, m, j] = musbjm[0][:, 0]

                        for d in range(event_width):

                            # Expected bump for every time-point given dur d in bump
                            Ebump = event_template[d] * mus[:, m, j]

                            if rho is not None:

                                if d > 0:
                                    # Previous mean was mu at previous time weighted by sine at
                                    # previous dur
                                    Ebump = d0 * Ebump + d1 * (
                                        event_template[d - 1]
                                        * np.concatenate(([0], mus[:-1, m, j]))
                                    )
                                else:
                                    # Previous mean was 0 (last time-point of preceding flat)
                                    Ebump = d0 * Ebump

                            jbs[:, m, d, j] = ld_wrapper(
                                y_mat[:, m],
                                Ebump,
                                scale,
                            )
                            cbs[:, m, d, j] = lcp_wrapper(
                                y_mat[:, m],
                                Ebump,
                                scale,
                            )

                            bs[:, d, j] += jbs[:, m, d, j]

                            if log is False:
                                # Transform log probs to probs
                                jbs[:, m, d, j] = np.exp(jbs[:, m, d, j])
                                cbs[:, m, d, j] = np.exp(cbs[:, m, d, j])

                        idx += 1

                    # Flats
                    else:
                        yJM = ys[m]
                        mus[:, m, j] = 0

                        if j == 0:
                            y_mat[:, m] = yJM[:, 0]

                        jbs[:, m, 0, j] = ld_wrapper(y_mat[:, m], 0, scale)
                        cbs[:, m, 0, j] = lcp_wrapper(y_mat[:, m], 0, scale)

                        if rho is not None and j > 0:
                            # Handle time-lag of last sample of previous bump into current flat
                            Ebump = event_template[-1] * np.concatenate(
                                ([0], mus[:-1, m, j - 1])
                            )

                            jbs[:, m, 1, j] = ld_wrapper(y_mat[:, m], d1 * Ebump, scale)
                            cbs[:, m, 1, j] = lcp_wrapper(
                                y_mat[:, m], d1 * Ebump, scale
                            )
                        else:
                            jbs[:, m, 1, j] = jbs[:, m, 0, j]
                            cbs[:, m, 1, j] = cbs[:, m, 0, j]

                        if m == 0:
                            bs[:, :, j] -= np.inf
                            bs[:, 0, j] = 0
                            bs[:, 1, j] = 0

                        bs[:, 0, j] += jbs[:, m, 0, j]
                        bs[:, 1, j] += jbs[:, m, 1, j]

                        if log is False:
                            # Transform log probs to probs
                            jbs[:, m, :, j] = np.exp(jbs[:, m, :, j])
                            cbs[:, m, :, j] = np.exp(cbs[:, m, :, j])

        if log is False:
            if is_hmp is False:
                bs[:, j] = np.exp(bs[:, j])
            else:
                bs[:, :, j] = np.exp(bs[:, :, j])

    # Now state duration probabilities #

    for j in range(n_S * (1 if starts_with_first else 2)):

        jdFam = d_fams[j]
        if jdFam is not None:

            etasdj = []
            musdj = []

            for par in range(jdFam.n_par):
                etasdj.append(Xs[idx] @ split_coef[idx])
                musdj.append(jdFam.links[par].fi(etasdj[-1]))
                idx += 1

            lpd = jdFam.lp(d_val, *musdj)

            mus_d.append(musdj)

        else:
            if j % 2 == 1:
                lpd = np.zeros(D - 1) - np.inf
                lpd[int(event_width - 1)] = 0
            else:
                raise ValueError(f"Need distribution duration family for state {j}.")

            mus_d.append(None)

        ds[:, j] = np.exp(lpd.flatten())
        ds[:, j] /= np.sum(ds[:, j])

        if log:
            ds[:, j] = np.log(ds[:, j])

    # Now state transition probabilities #
    T = None
    if model_t:
        T = np.zeros((n_S, n_S))
        if n_S > 2:
            for j in range(n_S):
                # No self-transitions!
                jtFam = MULNOMLSS(n_S - 2)

                etastj = []
                mustj = []

                for par in range(jtFam.n_par):
                    etastj.append(Xs[idx] @ split_coef[idx])
                    mustj.append(jtFam.links[par].fi(etastj[-1])[0, 0])
                    idx += 1

                # Transform now into the n_j state transition probabilities from j -> i(see the
                # MULNOMLSS methods doc-strings for details):
                mu_firstT = np.sum(mustj) + 1
                mustj = [mu / mu_firstT for mu in mustj]
                mustj.insert(0, 1 / mu_firstT)
                # print(mustj)

                Tj_idx = np.array([i for i in range(n_S) if i != j])
                T[j, Tj_idx] = mustj
        else:
            T[0, 1] = 1
            T[1, 0] = 1

        if log:
            T = np.log(T)

    # print(T)

    # Now initial state probabilities #
    pi = None
    if model_pi:
        pi_fam = MULNOMLSS(n_S - 1)

        etaspi = []
        muspi = []

        for par in range(pi_fam.n_par):
            etaspi.append(Xs[idx] @ split_coef[idx])
            muspi.append(pi_fam.links[par].fi(etaspi[-1])[0, 0])
            idx += 1

        mu_firstPI = np.sum(muspi) + 1
        muspi = [mu / mu_firstPI for mu in muspi]
        muspi.insert(0, 1 / mu_firstPI)  # zero is reference
        pi = np.array(muspi)

        if log:
            pi = np.log(pi)

    return y_mat, mus, mus_d, jbs, cbs, bs, ds, T, pi


def _sample_series_emissions(
    coef: np.ndarray,
    coef_split_idx: list[int],
    shared_pars: list[int],
    shared_m: bool,
    Xs: list[scp.sparse.csc_array],
    n_S: int,
    obs_fams: list[list[GAMLSSFamily]] | None,
    M: int,
    scale: float | None,
    build_mat_idx: list[int] | None,
    is_hmp: bool = False,
    n_samples: int = 1,
    seed: int | None = 0,
    hmp_fam: str = "Exponential",
):
    # First fix Xs for Nones
    Xfix = [X if X is not None else Xs[build_mat_idx[xi]] for xi, X in enumerate(Xs)]
    Xs = Xfix

    # Extract some extra info and define storage
    n_T = Xfix[0].shape[0]  # Number of time-points in this series
    split_coef = np.split(coef, coef_split_idx)

    emissions = np.zeros((n_T, n_S, M, n_samples))

    # Now sample emissions

    if is_hmp:
        # Initialize some density/probability functions for hmp-like models
        hmp_code = 2 if hmp_fam == "Gaussian" else 1

        # Define rvs callback function
        def rvs_wrapper(size: int, scale: float, seed: int):

            np_gen = np.random.default_rng(seed)

            # Exponential case
            if hmp_code == 1:
                flip = (np_gen.random(size) >= 0.5) * 1
                flip[flip < 1] = -1
                return flip * np.power(
                    scp.stats.expon.rvs(size=size, scale=scale, random_state=seed), 0.5
                )

            # Gaussian case
            elif hmp_code == 2:
                return scp.stats.norm.rvs(
                    size=size, scale=np.power(scale, 0.5), random_state=seed
                )

    idx = 0  # Keep track of indices for ys and Xs

    # First check for part of emission models shared between states (and m)

    # Holds list per par, each list contains at least one eta vector.
    shared_eta = []
    for par in shared_pars:

        midx = 1
        if shared_m is False:
            midx = M

        shared_eta.append([])

        # Compute shared part of eta per parameter and m
        for m in range(midx):
            shared_eta[par].append(Xs[idx] @ split_coef[idx])
            idx += 1

    # Extract (potential) thetas for multivariate observation models
    thetas = split_coef[-1].flatten()
    theta_idx = 0
    for j in range(n_S):

        # Multivariate gaussian emission
        if isinstance(obs_fams[j][0], MultiGauss):

            jmbFam = obs_fams[j][0]

            # Get thetas for this family
            thetas_j = thetas[theta_idx : theta_idx + jmbFam.extra_coef]  # noqa: E203

            etasbjm = []
            musbjm = []

            for par in range(jmbFam.n_par):
                etasbjm.append(Xs[idx] @ split_coef[idx])

                # Add shared eta to etasbjm
                if 0 in shared_pars:
                    if shared_m:
                        etasbjm[-1] += shared_eta[0][0]
                    else:
                        etasbjm[-1] += shared_eta[0][par]

                musbjm.append(jmbFam.links[par].fi(etasbjm[-1]))
                idx += 1

            musbjm = np.concatenate(musbjm, axis=1)

            # Get transpose of Cholesky of precision
            R, _ = jmbFam.getR(thetas_j)

            # Sample from multivariate
            emissions[:, j, :, :] = jmbFam.rvs(
                musbjm,
                thetas_j,
                size=n_samples,
                seed=seed,
            ).T

            if seed is not None:
                seed += 1

            # Update theta index
            theta_idx += jmbFam.extra_coef

        # Multiple independent emission signals
        else:

            for m in range(M):
                # Get family associated with state j and signal m
                jmbFam = obs_fams[j][m]

                if is_hmp is False:

                    etasbjm = []
                    musbjm = []

                    for par in range(jmbFam.n_par):
                        etasbjm.append(Xs[idx] @ split_coef[idx])

                        # Add shared eta to etasbjm
                        if par in shared_pars:
                            if shared_m is False:
                                etasbjm[-1] += shared_eta[par][m]
                            else:
                                etasbjm[-1] += shared_eta[par][0]

                        musbjm.append(jmbFam.links[par].fi(etasbjm[-1]))
                        idx += 1

                    # Can sample now
                    emissions[:, j, m, :] = jmbFam.rvs(
                        *musbjm, size=n_samples, seed=seed
                    ).T

                else:
                    # HMP case
                    emissions[:, j, m, :] = rvs_wrapper(
                        n_samples * n_T, scale, seed
                    ).reshape(n_T, n_samples)

                if seed is not None:
                    seed += 1

    return emissions


def compute_dur_res_series(
    states: np.ndarray,
    n_S: int,
    mus_d: list[list[np.ndarray] | None],
    d_fams: list[GAMLSSFamily | None],
    starts_with_first: bool,
) -> list[list[float] | float]:
    """Computes residual of a state duration agains the model for the stage duration.

    :param states: Array holding an estimate of the state sequence
    :type states: np.ndarray
    :param n_S: Number of states
    :type n_S: int
    :param mus_d: List holding predcitions for all parameters of each distribution included in
        ``d_fams``.
    :type mus_d: list[list[np.ndarray]  |  None]
    :param d_fams: List of distributions assumed for the durations of each state
    :type d_fams: list[GAMLSSFamily  |  None]
    :param starts_with_first: Whether first state is assumed to begin at first time-point.
    :type starts_with_first: bool
    :return: A list with the residuals of the durations per state. If a state has no duration model,
        then a ``np.nan`` is included at that index.
    :rtype: list[list[float]|np.nan]
    """
    res = [[] for _ in range(n_S)]
    d = 0
    j = states[0]
    transitions = 0

    for s in states:
        if s == j:
            d += 1
        else:
            # Identify correct duration distribution
            dj_idx = j if (transitions == 0 or starts_with_first is True) else j + n_S

            # Now compute residual of duration under model
            if mus_d[dj_idx] is None:
                res[j] = None
            else:
                d_res = d_fams[(dj_idx)].get_resid(
                    np.array([d]).reshape(-1, 1), *mus_d[dj_idx]
                )[0, 0]

                # And collect
                res[j].append(d_res)
            j = s
            d = 1
            transitions += 1

    # Catch dur from last state
    dj_idx = j if (starts_with_first is True) else j + n_S

    if mus_d[dj_idx] is None:
        res[j] = None
    else:
        d_res = d_fams[(dj_idx)].get_resid(
            np.array([d]).reshape(-1, 1), *mus_d[dj_idx]
        )[0, 0]

        res[j].append(d_res)

    return res


class FTPHSMMFamily(GSMMFamily):
    """_summary_

    :param pars: Number of parameters of the log-likelihood
    :type pars: int
    :param links: List of link functions to use in the models of different parameters
    :type links: list[Link]
    :param n_S: Number of latent states.
    :type n_S: int
    :param obs_fams: Distribution of emissions - one per state and signal
    :type obs_fams: list[list[GAMLSSFamily]] | None
    :param d_fams: Distribution of state durations - one per state
    :type d_fams: list[GAMLSSFamily]
    :param T: State transition matrix as np.ndarray (or a list of state transition matrices per
        series). If ``None``, this is set to a shifted matrix with unit elements on the super
        diagonal and zeroes everywhere else (corresponding to strict left-right hsmm)
    :type T: list[np.ndarray]|np.ndarray | None
    :param pi: Initial probability distribution of latent states. If ``None`` then a probability of
        1 is assigned to being in the first state
    :type pi: np.ndarray | None
    :param sid: Array holding the first sample of each series in the data. Can be used to split the
        observation vectors and model matrices into time-series specific versions
    :type sid: np.ndarray
    :param tid: Often, the model matrices for the models of the duration distribution parameters
        will have fewer rows than those of the observation models (e.g., because covariates only
        vary with trials). The optional array ``tid`` can then hold indices corresponding to the
        onset of a row-block of a new trial in the model matrices of the duration models. Can then
        be used to split the duration model matrices into trial specific versions. If this is set
        to ``None``, it will be set to ``sid``.
    :type tid: np.ndarray | None
    :param D: Max duration a state can take on (**in samples**)
    :type D: int
    :param M: Number of signals recorded if the HSMM is multivariate. All signals are assumed to be
        independent from one another given covariates and coefficients.
    :type M: int
    :param starts_with_first: Whether the first state starts with the first observation or might
        have been going on for a max duration of ``D``.
    :type starts_with_first: bool
    :param ends_with_last: Whether the last state ends with the last observation or can continue.
    :type ends_with_last: bool
    :param ends_in_last: Whether the Markov chain ends in the last state (useful if we want a
        left-to-right hsmm).
    :type ends_in_last: bool
    :param scale: Scale parameter to use for the the observation probabilities under a hmp model
        (can be set to None for regular hsmms).
    :type scale: float | None
    :param event_template: Template of the HMP event pattern (or ``None`` if a standard hsmm is
        to be estimated).
    :type event_template: np.ndarray | None
    :param hmp_fam: Emission distribution family for HMP models. Must be "Gaussian" or
        "Exponential" (the default).
    :type hmp_fam: str
    :param n_cores: Number of cores to use in the computations, defaults to 1.
    :type n_cores: int, optional
    :param build_mat_idx: If not all matrices are built explicitly, this has to be a list of indices
        per parameter modeled, pointing at the index of ``Xs`` holding the matrix that should be
        used by that parameter.
    :type build_mat_idx: list[int] | None, optional
    :param shared_pars: A list containing indices of emission distribution parameters for which
        at least one (in case ``shared_m is False`` one for each of the ``M`` signals) formula
        has been provided containing terms to be shared between the ``n_S`` states. Defaults to
        ``[]`` meaning no terms are shared.
    :type shared_pars: list[int], optional
    :param shared_m: Bool indicating whether any shared terms are not just shared between states but
        also between the ``M`` signals. Defaults to ``False``
    :type shared_m: bool, optional
    :param Lrhoi: Inverse of transpose of Cholesky of banded covariance matrix of an ar model of
        the residuals. Defaults to None - indicating no ar model.
    :type Lrhoi: scp.sparse.csc_array | None, optional
    """

    def __init__(
        self,
        pars: int,
        links: list[Link],
        n_S: int,
        obs_fams: list[list[GAMLSSFamily]] | None,
        d_fams: list[GAMLSSFamily],
        T: list[np.ndarray] | np.ndarray | None,
        pi: np.ndarray | None,
        sid: np.ndarray,
        tid: np.ndarray | None,
        D: int,
        M: int,
        starts_with_first: bool,
        ends_with_last: bool,
        ends_in_last: bool,
        scale: float | None,
        event_template: np.ndarray | None,
        hmp_fam: str = "Exponential",
        n_cores=1,
        build_mat_idx: list[int] | None = None,
        shared_pars: list[int] = [],
        shared_m: bool = False,
        Lrhoi: scp.sparse.csc_array | None = None,
    ) -> None:

        super().__init__(
            pars,
            links,
            n_S,
            obs_fams,
            d_fams,
            T,
            pi,
            sid,
            tid,
            D,
            M,
            starts_with_first,
            ends_with_last,
            ends_in_last,
            scale,
            event_template,
            hmp_fam,
            n_cores,
            build_mat_idx,
            shared_pars,
            shared_m,
            Lrhoi,
        )

        # Some checks
        self.__ishmp = False
        for state_obs in obs_fams:
            for signal_obs in state_obs:
                if signal_obs is None:
                    self.__ishmp = True

                if signal_obs is None and (starts_with_first is False):
                    raise ValueError(
                        (
                            "hmp like models are (currently) not "
                            "supported with ``starts_with_first==False``"
                        )
                    )

        for state_dur in d_fams:
            if state_dur is None:
                self.__ishmp = True
            if state_dur is None and (starts_with_first is False):
                raise ValueError(
                    (
                        "hmp like models are (currently) not "
                        "supported with ``starts_with_first==False``"
                    )
                )

        self.extra_coef = 0
        if self.__ishmp is False:
            for state_obs in obs_fams:
                if isinstance(state_obs[0], MultiGauss):
                    self.extra_coef += state_obs[0].extra_coef

    def compute_od_probs(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray],
        Xs: list[scp.sparse.csc_array],
        log: bool = True,
        sid: None | np.ndarray = None,
        tid: None | np.ndarray = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Computes the (log)-probabilities of observations and stage durations under given model.

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each paramter of the llk.
        :type coef_split_idx: [int]
        :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as
            ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the
            actual observed data is passed along via the first formula (so it is stored in
            ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula,
            then ``ys`` contains ``None`` at their indices to save memory.
        :type ys: [np.ndarray or None]
        :param Xs: A list of sparse model matrices per likelihood parameter.
        :type Xs: [scp.sparse.csc_array]
        :param log: Bool indicating whether to compute probabilities or log-probabilities. Defaults
            to ``True`` - meaning log-probabilities are computed.
        :type log: bool, optional
        :param sid: Optional Array holding the first sample of each series in the data. Is used
            to split the observation vectors and model matrices into time-series specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type sid: np.ndarray | None, optional
        :param tid: Often, the model matrices for the models of the duration distribution parameters
            will have fewer rows than those of the observation models (because covariates only
            vary with trials). The optional array ``tid`` can then hold indices corresponding to the
            onset of a row-block of a new trial in the model matrices of the duration models.
            Is then used to split the duration model matrices into trial specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type tid: np.ndarray | None, optional
        :return: List of tuples - one tuple per time-series. Each tuple holds two arrays, first
            dimension of both is of size ``n_S``, corresponding to the number of latent states.
            First array has three dimensions, holds (log)-probabilities of all observations per
            state (second dimension) and signal recorded (third dimension), second array is
            two-dimensional, holds (log)-probabilities of all possible stage durations per state.
        :rtype: list[tuple[np.ndarray,np.ndarray]]
        """

        def series_log_prob(
            coef,
            coef_split_idx,
            shared_pars,
            shared_m,
            ys,
            Xs,
            n_S,
            obs_fams,
            d_fams,
            D,
            M,
            event_template,
            scale,
            log,
            starts_with_first,
            build_mat_idx,
            is_hmp,
            hmp_fam,
            rho,
        ):
            # Compute log probs for an individual series

            event_width = None if event_template is None else len(event_template)

            y_mat, mus, _, _, _, bs, ds, _, _ = _compute_series_probs(
                coef,
                coef_split_idx,
                shared_pars,
                shared_m,
                ys,
                Xs,
                n_S,
                obs_fams,
                d_fams,
                D,
                M,
                event_width,
                event_template,
                scale,
                log,
                False,
                False,
                starts_with_first,
                build_mat_idx,
                is_hmp,
                hmp_fam,
                rho,
            )

            bs[(np.isnan(bs) | np.isinf(bs))] = -np.inf if log else 0
            ds[(np.isnan(ds) | np.isinf(ds))] = -np.inf if log else 0

            return bs, ds

        # Extract families, transition matrices, etc.
        n_S = self.llkargs[0]
        obs_fams = self.llkargs[1]
        d_fams = self.llkargs[2]
        if sid is None:
            sid = self.llkargs[5]
        if tid is None:
            tid = self.llkargs[6]
        D = self.llkargs[7]
        M = self.llkargs[8]
        starts_with_first = self.llkargs[9]
        scale = self.llkargs[12]
        event_template = self.llkargs[13]
        hmp_fam = self.llkargs[14]
        n_cores = self.llkargs[15]
        build_mat_idx = self.llkargs[16]
        shared_pars = self.llkargs[17]
        shared_m = self.llkargs[18]
        Lrhoi = self.llkargs[19]
        n_series = len(sid)
        is_hmp = self.__ishmp

        # Duration model matrices have rows for individual time-points not just trials
        if tid is None:
            tid = sid

        rho = None
        if Lrhoi is not None:
            rho = np.sqrt(-1 * (np.power(1 / Lrhoi[1, 1], 2) - 1))

        # Split up ys, Xs
        split_Ys, split_Xs = _split_matrices(
            ys,
            Xs,
            shared_pars,
            shared_m,
            obs_fams,
            d_fams,
            sid,
            tid,
            n_S,
            M,
            False,
            False,
            starts_with_first,
            Lrhoi,
        )

        # Now compute observation/duration probabilities for every individual series and then sum up
        if n_cores == 1:
            ps = []

            for series in range(n_series):
                bss, dss = series_log_prob(
                    coef,
                    coef_split_idx,
                    shared_pars,
                    shared_m,
                    [
                        y_split[series] if y_split is not None else None
                        for y_split in split_Ys
                    ],
                    [
                        X_split[series] if X_split is not None else None
                        for X_split in split_Xs
                    ],
                    n_S,
                    obs_fams,
                    d_fams,
                    D,
                    M,
                    event_template,
                    scale,
                    log,
                    starts_with_first,
                    build_mat_idx,
                    is_hmp,
                    hmp_fam,
                    rho,
                )

                ps.append([bss, dss])

        else:  # Compute in parallel

            args = zip(
                repeat(coef),
                repeat(coef_split_idx),
                repeat(shared_pars),
                repeat(shared_m),
                [
                    [
                        y_split[series] if y_split is not None else None
                        for y_split in split_Ys
                    ]
                    for series in range(n_series)
                ],
                [
                    [
                        X_split[series] if X_split is not None else None
                        for X_split in split_Xs
                    ]
                    for series in range(n_series)
                ],
                repeat(n_S),
                repeat(obs_fams),
                repeat(d_fams),
                repeat(D),
                repeat(M),
                repeat(event_template),
                repeat(scale),
                repeat(log),
                repeat(starts_with_first),
                repeat(build_mat_idx),
                repeat(is_hmp),
                repeat(hmp_fam),
                repeat(rho),
            )

            with mp.Pool(processes=n_cores) as pool:
                bs, ds = zip(*pool.starmap(series_log_prob, args))

            ps = [[bss, dss] for bss, dss in zip(bs, ds)]

        return ps

    def predict(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray],
        Xs: list[scp.sparse.csc_array],
        sid: None | np.ndarray = None,
        tid: None | np.ndarray = None,
        seed: int | None = 0,
        size: int = 1,
        scale: float | None = None,
    ) -> list[np.ndarray, np.ndarray]:
        """Predicts ``size`` state and observation sequences given model matrices ``Xs`` and
        coefficients ``coef``.

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each paramter of the llk.
        :type coef_split_idx: [int]
        :param Xs: A list of sparse model matrices per likelihood parameter.
        :type Xs: [scp.sparse.csc_array]
        :param sid: Optional Array holding the first sample of each series in the data. Is used
            to split the observation vectors and model matrices into time-series specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type sid: np.ndarray | None, optional
        :param tid: Often, the model matrices for the models of the duration distribution parameters
            will have fewer rows than those of the observation models (because covariates only
            vary with trials). The optional array ``tid`` can then hold indices corresponding to the
            onset of a row-block of a new trial in the model matrices of the duration models.
            Is then used to split the duration model matrices into trial specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type tid: np.ndarray | None, optional
        :param seed: An optional seed to initialize the sampler. If this is set to None, then random
            initialization is performed.
        :type seed: int, optional
        :param size: Number of state and observation sequences to generate (per series). Defaults to
            1.
        :type size: int, optional
        :param scale: Optional scale parameter to use for predictions. Defaults to None, in which
            case the one passed to the constructor of ``self`` is used.
        :type scale: float | None, optional
        :return: A list holding two numpy arrays per series implied by ``sid``. First numpy array is
            of dimension ``(nT, M, size)``, where ``nT`` is the number of time-points implied by
            the number of rows of the ``Xs`` corresponding to that series. ``M`` is the number of
            emitting signals. the second array is of dimension ``(nT, size)``. The first array holds
            predicted observation sequences, the second array holds predicted state sequences.
        :rtype: list[np.ndarray, np.ndarray]
        """

        def predict_series(
            coef,
            coef_split_idx,
            shared_pars,
            shared_m,
            ys,
            Xs,
            n_S,
            obs_fams,
            d_fams,
            T,
            pi,
            D,
            M,
            starts_with_first,
            ends_with_last,
            ends_in_last,
            event_template,
            scale,
            build_mat_idx,
            is_hmp,
            hmp_fam,
            rho,
            n_samples,
            seed,
        ):

            # Extract some extra info and define storage
            n_T = len(ys[0])  # Number of time-points in this series
            event_width = 0 if event_template is None else len(event_template)

            # Sample emissions
            emissions = _sample_series_emissions(
                coef,
                coef_split_idx,
                shared_pars,
                shared_m,
                Xs,
                n_S,
                obs_fams,
                M,
                scale,
                build_mat_idx,
                is_hmp,
                n_samples,
                seed,
                hmp_fam,
            )

            # Now sample state sequences and combine that with the emission samples
            emission_samples = np.zeros((n_T, M, n_samples))
            state_samples = np.zeros((n_T, n_samples))

            # Compute state transition, duration, and initial state disribution matrices for a
            # given series.
            _, mus, _, _, _, _, ds, _, _ = _compute_series_probs(
                coef,
                coef_split_idx,
                shared_pars,
                shared_m,
                ys,
                Xs,
                n_S,
                obs_fams,
                d_fams,
                D,
                M,
                event_width,
                event_template,
                scale,
                False,
                False,
                False,
                starts_with_first,
                build_mat_idx,
                is_hmp,
                hmp_fam,
                rho,
            )

            np_gen = np.random.default_rng(seed)
            d_vals = np.arange(1, D).reshape(-1, 1)
            state_vals = np.arange(n_S)

            if rho is not None:
                d0 = 1 / np.sqrt(1 - np.power(rho, 2))
                d1 = -rho / np.sqrt(1 - np.power(rho, 2))

            for sample in range(n_samples):

                # Run the state generation process forward and select correct emissions for
                # chosen states

                # Select initial state
                state = np_gen.choice(state_vals, p=pi)

                # and duration of initial state
                dur = np_gen.choice(d_vals, p=ds[:, state])

                if seed is not None:
                    seed += 1

                t = 0
                while t < n_T:

                    if dur < 0:

                        if is_hmp and state == (n_S - 1):
                            dur += 1  # Simply keep extending last state's duration
                        else:

                            # Sample next state and dur
                            state = np_gen.choice(state_vals, p=T[state, :])

                            dur = np_gen.choice(
                                d_vals,
                                p=ds[:, (state if starts_with_first else state + n_S)],
                            )

                    # Collect emission and state predictions
                    emission_samples[t, :, sample] = emissions[t, state, :, sample]
                    state_samples[t, sample] = state

                    # Add hmp bumps
                    if is_hmp and state % 2 == 1:
                        for m in range(M):

                            # Expected bump for every time-point given dur d in bump
                            d = (event_width - 1) - dur

                            Ebump = event_template[d] * mus[t, m, state]

                            if rho is not None:

                                if d > 0 and t > 0:
                                    # Previous mean was mu at previous time weighted by sine at
                                    # previous dur
                                    Ebump = d0 * Ebump + d1 * (
                                        event_template[d - 1] * mus[t - 1, m, state]
                                    )
                                else:
                                    # Previous mean was 0 (last time-point of preceding flat)
                                    Ebump = d0 * Ebump

                            emission_samples[t, m, sample] += Ebump

                    # Update time and dur
                    t += 1
                    dur -= 1

                    # Current bump will contaminate first sample of next state
                    if (
                        t < n_T
                        and is_hmp
                        and rho is not None
                        and state % 2 == 1
                        and dur == 0
                    ):
                        for m in range(M):
                            Ebump = d1 * event_template[-1] * mus[t - 1, m, state]
                            emission_samples[t, m, sample] += Ebump

                    # And seed
                    if seed is not None:
                        seed += 1

            if rho is not None:
                # Build series-specific Lrho
                d0 = np.tile(d0, n_T)
                d1 = np.tile(d1, n_T - 1)
                d0[0] = 1
                Lhrois = scp.sparse.diags_array([d0, d1], format="csr", offsets=[0, 1])
                Lrhos = cpp_backsolve_tr(
                    Lhrois.tocsc(), scp.sparse.eye_array(n_T, format="csc")
                )

                # emissions are Lhrois.T@y, so to get y -> Lrhos.T@Lhrois.T@y
                for m in range(M):
                    emission_samples[:, m, :] = Lrhos.T @ emission_samples[:, m, :]

            return emission_samples, state_samples

        # Extract families, transition matrices, etc.
        n_S = self.llkargs[0]
        obs_fams = self.llkargs[1]
        d_fams = self.llkargs[2]
        T = self.llkargs[3]
        pi = self.llkargs[4]
        if sid is None:
            sid = self.llkargs[5]
        if tid is None:
            tid = self.llkargs[6]
        D = self.llkargs[7]
        M = self.llkargs[8]
        starts_with_first = self.llkargs[9]
        ends_with_last = self.llkargs[10]
        ends_in_last = self.llkargs[11]
        if scale is None:
            scale = self.llkargs[12]
        event_template = self.llkargs[13]
        hmp_fam = self.llkargs[14]
        n_cores = self.llkargs[15]
        build_mat_idx = self.llkargs[16]
        shared_pars = self.llkargs[17]
        shared_m = self.llkargs[18]
        Lrhoi = self.llkargs[19]
        n_series = len(sid)
        is_hmp = self.__ishmp

        # Initialize potential None arguments

        # Initialize potential None arguments
        if T is None:
            T = np.diag(
                np.ones(n_S - 1), 1
            )  # Super diagonal shift matrix for transitions

        if pi is None:
            pi = np.zeros(n_S)  # Start in first state
            pi[0] = 1

        # Duration model matrices
        if tid is None:
            tid = sid

        # Extract rho
        rho = None
        if Lrhoi is not None:
            rho = np.sqrt(-1 * (np.power(1 / Lrhoi[1, 1], 2) - 1))

        # Split up ys, Xs
        split_Ys, split_Xs = _split_matrices(
            ys,
            Xs,
            shared_pars,
            shared_m,
            obs_fams,
            d_fams,
            sid,
            tid,
            n_S,
            M,
            False,
            False,
            starts_with_first,
            Lrhoi,
        )

        # Now compute predictions for each series
        if n_cores == 1:
            predictions = []

            for series in range(n_series):
                with warnings.catch_warnings():  # Supress warnings
                    warnings.simplefilter("ignore")
                    e_predss, s_predss = predict_series(
                        coef,
                        coef_split_idx,
                        shared_pars,
                        shared_m,
                        [
                            y_split[series] if y_split is not None else None
                            for y_split in split_Ys
                        ],
                        [
                            X_split[series] if X_split is not None else None
                            for X_split in split_Xs
                        ],
                        n_S,
                        obs_fams,
                        d_fams,
                        T[series] if isinstance(T, list) else T,
                        pi,
                        D,
                        M,
                        starts_with_first,
                        ends_with_last,
                        ends_in_last,
                        event_template,
                        scale,
                        build_mat_idx,
                        is_hmp,
                        hmp_fam,
                        rho,
                        size,
                        seed,
                    )

                    predictions.append([e_predss, s_predss])

        else:  # Compute in parallel

            args = zip(
                repeat(coef),
                repeat(coef_split_idx),
                repeat(shared_pars),
                repeat(shared_m),
                [
                    [
                        y_split[series] if y_split is not None else None
                        for y_split in split_Ys
                    ]
                    for series in range(n_series)
                ],
                [
                    [
                        X_split[series] if X_split is not None else None
                        for X_split in split_Xs
                    ]
                    for series in range(n_series)
                ],
                repeat(n_S),
                repeat(obs_fams),
                repeat(d_fams),
                T if isinstance(T, list) else repeat(T),
                repeat(pi),
                repeat(D),
                repeat(M),
                repeat(starts_with_first),
                repeat(ends_with_last),
                repeat(ends_in_last),
                repeat(event_template),
                repeat(scale),
                repeat(build_mat_idx),
                repeat(is_hmp),
                repeat(hmp_fam),
                repeat(rho),
                repeat(size),
                repeat(seed),
            )

            with warnings.catch_warnings():  # Supress warnings
                warnings.simplefilter("ignore")
                with mp.Pool(processes=n_cores) as pool:
                    e_preds, s_preds = zip(*pool.starmap(predict_series, args))

                predictions = [
                    [e_predss, s_predss] for e_predss, s_predss in zip(e_preds, s_preds)
                ]

        return predictions

    def decode_viterbi(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray],
        Xs: list[scp.sparse.csc_array],
        sid: None | np.ndarray = None,
        tid: None | np.ndarray = None,
        n_cores: None | int = None,
    ) -> list[tuple[int, np.ndarray]]:
        """Perform viterbi decoding of state sequence for a :class:`FTPHSMMFamily` model.

        :param coef: _description_
        :type coef: np.ndarray
        :param coef_split_idx: _description_
        :type coef_split_idx: list[int]
        :param ys: _description_
        :type ys: list[np.ndarray]
        :param Xs: _description_
        :type Xs: list[scp.sparse.csc_array]
        :param sid: Optional Array holding the first sample of each series in the data. Is used
            to split the observation vectors and model matrices into time-series specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type sid: np.ndarray | None, optional
        :param tid: Often, the model matrices for the models of the duration distribution parameters
            will have fewer rows than those of the observation models (because covariates only
            vary with trials). The optional array ``tid`` can then hold indices corresponding to the
            onset of a row-block of a new trial in the model matrices of the duration models.
            Is then used to split the duration model matrices into trial specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type tid: np.ndarray | None, optional
        :param n_cores: Optionally, the number of cores to use during decoding. Allows to over-write
            the value passed to the constructor of ``self``. If set to None, the value passed to
            the constructor is used. Defaults to None
        :type n_cores: int| None, optional
        :return: A list with a tuple per time-series. First element of tuple is excess duration of
            final state, second is a np.array with the decoded state sequence matching the
            length of the time-series.
        :rtype: list[tuple[int,np.ndarray]]
        """

        def series_viterbi(
            coef,
            coef_split_idx,
            shared_pars,
            shared_m,
            ys,
            Xs,
            n_S,
            obs_fams,
            d_fams,
            T,
            pi,
            D,
            M,
            starts_with_first,
            ends_with_last,
            ends_in_last,
            event_template,
            scale,
            build_mat_idx,
            is_hmp,
            hmp_fam,
            rho,
        ):
            # Compute state sequence for individual series

            # Extract some extra info and define storage
            n_T = len(ys[0])  # Number of time-points in this series
            event_width = 0 if event_template is None else len(event_template)

            # y_mat, mus, jbs, cbs, bs, ds, T, pi
            y_mat, mus, _, ljbs, _, lbs, lds, _, _ = _compute_series_probs(
                coef,
                coef_split_idx,
                shared_pars,
                shared_m,
                ys,
                Xs,
                n_S,
                obs_fams,
                d_fams,
                D,
                M,
                event_width,
                event_template,
                scale,
                True,
                False,
                False,
                starts_with_first,
                build_mat_idx,
                is_hmp,
                hmp_fam,
                rho,
            )

            lbs[np.isnan(lbs)] = -np.inf
            ljbs[np.isnan(ljbs)] = -np.inf
            lds[np.isnan(lds)] = -np.inf

            lT = np.log(T)
            lpi = np.log(pi)

            if is_hmp:
                hmp_code = 1
            else:
                hmp_code = 0
                event_template = np.zeros(1)
                scale = 1

            max_ed, states, deltas, psis = hsmm.viterbi(
                lbs,
                lds,
                lT,
                lpi,
                event_template,
                scale,
                n_T,
                D,
                n_S,
                event_width,
                starts_with_first,
                ends_with_last,
                ends_in_last,
                hmp_code,
            )

            return max_ed, states

        # Extract families, transition matrices, etc.
        n_S = self.llkargs[0]
        obs_fams = self.llkargs[1]
        d_fams = self.llkargs[2]
        T = self.llkargs[3]
        pi = self.llkargs[4]
        if sid is None:
            sid = self.llkargs[5]
        if tid is None:
            tid = self.llkargs[6]
        D = self.llkargs[7]
        M = self.llkargs[8]
        starts_with_first = self.llkargs[9]
        ends_with_last = self.llkargs[10]
        ends_in_last = self.llkargs[11]
        scale = self.llkargs[12]
        event_template = self.llkargs[13]
        hmp_fam = self.llkargs[14]

        if n_cores is None:
            n_cores = self.llkargs[15]
        build_mat_idx = self.llkargs[16]
        shared_pars = self.llkargs[17]
        shared_m = self.llkargs[18]
        Lrhoi = None
        if self.llkargs[19] is not None and ys[0].shape[0] == self.llkargs[19].shape[0]:
            Lrhoi = self.llkargs[19]
        n_series = len(sid)
        is_hmp = self.__ishmp

        # Initialize potential None arguments
        if T is None:
            T = np.diag(
                np.ones(n_S - 1), 1
            )  # Super diagonal shift matrix for transitions

        if pi is None:
            pi = np.zeros(n_S)  # Start in first state
            pi[0] = 1

        # Must extract rho, even if ``Lrhoi`` is set to None above (happens when this method is
        # called by get_resid)
        rho = None
        if self.llkargs[19] is not None:
            rho = np.sqrt(-1 * (np.power(1 / self.llkargs[19][1, 1], 2) - 1))

        # Duration model matrices
        if tid is None:
            tid = sid

        # Split up ys, Xs
        split_Ys, split_Xs = _split_matrices(
            ys,
            Xs,
            shared_pars,
            shared_m,
            obs_fams,
            d_fams,
            sid,
            tid,
            n_S,
            M,
            False,
            False,
            starts_with_first,
            Lrhoi,
        )

        # Now compute llk for every individual series and then sum up
        if n_cores == 1:
            states = []

            for series in range(n_series):
                with warnings.catch_warnings():  # Supress warnings
                    warnings.simplefilter("ignore")
                    edss, statess = series_viterbi(
                        coef,
                        coef_split_idx,
                        shared_pars,
                        shared_m,
                        [
                            y_split[series] if y_split is not None else None
                            for y_split in split_Ys
                        ],
                        [
                            X_split[series] if X_split is not None else None
                            for X_split in split_Xs
                        ],
                        n_S,
                        obs_fams,
                        d_fams,
                        T[series] if isinstance(T, list) else T,
                        pi,
                        D,
                        M,
                        starts_with_first,
                        ends_with_last,
                        ends_in_last,
                        event_template,
                        scale,
                        build_mat_idx,
                        is_hmp,
                        hmp_fam,
                        rho,
                    )

                    states.append([edss, statess])

        else:  # Compute in parallel

            args = zip(
                repeat(coef),
                repeat(coef_split_idx),
                repeat(shared_pars),
                repeat(shared_m),
                [
                    [
                        y_split[series] if y_split is not None else None
                        for y_split in split_Ys
                    ]
                    for series in range(n_series)
                ],
                [
                    [
                        X_split[series] if X_split is not None else None
                        for X_split in split_Xs
                    ]
                    for series in range(n_series)
                ],
                repeat(n_S),
                repeat(obs_fams),
                repeat(d_fams),
                T if isinstance(T, list) else repeat(T),
                repeat(pi),
                repeat(D),
                repeat(M),
                repeat(starts_with_first),
                repeat(ends_with_last),
                repeat(ends_in_last),
                repeat(event_template),
                repeat(scale),
                repeat(build_mat_idx),
                repeat(is_hmp),
                repeat(hmp_fam),
                repeat(rho),
            )

            with warnings.catch_warnings():  # Supress warnings
                warnings.simplefilter("ignore")
                with mp.Pool(processes=n_cores) as pool:
                    eds, states = zip(*pool.starmap(series_viterbi, args))

                states = [[edss, statess] for edss, statess in zip(eds, states)]

        return states

    def sample_posterior_states(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray],
        Xs: list[scp.sparse.csc_array],
        n_samples: int,
        seed: int = 0,
        sid: None | np.ndarray = None,
        tid: None | np.ndarray = None,
        n_cores: None | int = None,
    ) -> float:
        """Samples the posterior of state sequences given parameter estimates and data.

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each paramter of the llk.
        :type coef_split_idx: [int]
        :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as
            ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the
            actual observed data is passed along via the first formula (so it is stored in
            ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula,
            then ``ys`` contains ``None`` at their indices to save memory.
        :type ys: [np.ndarray or None]
        :param Xs: A list of sparse model matrices per likelihood parameter.
        :type Xs: [scp.sparse.csc_array]
        :param n_samples: Number of state sequences to sample.
        :type n_samples: int
        :param seed: An optional seed to initialize the sampler. **Note**: A value < 0 means random
            initialization is used. Thus set to >= 0 for reproducable results. Defaults to 0
        :type seed: int, optional
        :param sid: Optional Array holding the first sample of each series in the data. Is used
            to split the observation vectors and model matrices into time-series specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type sid: np.ndarray | None, optional
        :param tid: Often, the model matrices for the models of the duration distribution parameters
            will have fewer rows than those of the observation models (because covariates only
            vary with trials). The optional array ``tid`` can then hold indices corresponding to the
            onset of a row-block of a new trial in the model matrices of the duration models.
            Is then used to split the duration model matrices into trial specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type tid: np.ndarray | None, optional
        :param n_cores: Optionally, the number of cores to use during sampling. Allows to over-write
            the value passed to the constructor of ``self``. If set to None, the value passed to
            the constructor is used. Defaults to None
        :type n_cores: int| None, optional
        :return: The log-likelihood evaluated at ``coef``.
        :rtype: float
        """

        def sample_posterior_series(
            coef,
            coef_split_idx,
            shared_pars,
            shared_m,
            ys,
            Xs,
            n_S,
            obs_fams,
            d_fams,
            T,
            pi,
            D,
            M,
            starts_with_first,
            ends_with_last,
            ends_in_last,
            event_template,
            scale,
            build_mat_idx,
            is_hmp,
            hmp_fam,
            n_samples,
            seed,
            rho,
        ):
            # Compute llk for an individual series - total llk is just sum

            # Extract some extra info and define storage
            n_T = len(ys[0])  # Number of time-points in this series
            event_width = 0 if event_template is None else len(event_template)

            # We can pre-compute the bs and ds - i.e., observation probabilities and
            # duration probabilities
            y_mat, mus, _, ljbs, _, bs, ds, _, _ = _compute_series_probs(
                coef,
                coef_split_idx,
                shared_pars,
                shared_m,
                ys,
                Xs,
                n_S,
                obs_fams,
                d_fams,
                D,
                M,
                event_width,
                event_template,
                scale,
                False,
                False,
                False,
                starts_with_first,
                build_mat_idx,
                is_hmp,
                hmp_fam,
                rho,
            )

            bs[np.isnan(bs) | np.isinf(bs)] = 0
            ds[np.isnan(ds) | np.isinf(ds)] = 0

            # Now sample state sequences
            if is_hmp:
                hmp_code = 1
            else:
                event_template = np.zeros(1)
                scale = 1
                hmp_code = 0

            eds, states = hsmm.sample_backwards(
                bs,
                ds,
                T,
                pi,
                event_template,
                scale,
                n_T,
                D,
                n_S,
                event_width,
                n_samples,
                starts_with_first,
                ends_with_last,
                ends_in_last,
                hmp_code,
                seed,
            )

            return eds, states

        # Extract families, transition matrices, etc.
        n_S = self.llkargs[0]
        obs_fams = self.llkargs[1]
        d_fams = self.llkargs[2]
        T = self.llkargs[3]
        pi = self.llkargs[4]
        if sid is None:
            sid = self.llkargs[5]
        if tid is None:
            tid = self.llkargs[6]
        D = self.llkargs[7]
        M = self.llkargs[8]
        starts_with_first = self.llkargs[9]
        ends_with_last = self.llkargs[10]
        ends_in_last = self.llkargs[11]
        scale = self.llkargs[12]
        event_template = self.llkargs[13]
        hmp_fam = self.llkargs[14]
        if n_cores is None:
            n_cores = self.llkargs[15]
        build_mat_idx = self.llkargs[16]
        shared_pars = self.llkargs[17]
        shared_m = self.llkargs[18]
        Lrhoi = None
        if self.llkargs[19] is not None and ys[0].shape[0] == self.llkargs[19].shape[0]:
            Lrhoi = self.llkargs[19]
        n_series = len(sid)
        is_hmp = self.__ishmp

        # Initialize potential None arguments
        if T is None:
            T = np.diag(
                np.ones(n_S - 1), 1
            )  # Super diagonal shift matrix for transitions

        if pi is None:
            pi = np.zeros(n_S)  # Start in first state
            pi[0] = 1

        if (
            tid is None
        ):  # Duration model matrices have rows for individual time-points not just trials
            tid = sid

        # Must extract rho, even if ``Lrhoi`` is set to None above (happens when this method is
        # called by get_resid)
        rho = None
        if self.llkargs[19] is not None:
            rho = np.sqrt(-1 * (np.power(1 / self.llkargs[19][1, 1], 2) - 1))

        # Split up ys, Xs
        split_Ys, split_Xs = _split_matrices(
            ys,
            Xs,
            shared_pars,
            shared_m,
            obs_fams,
            d_fams,
            sid,
            tid,
            n_S,
            M,
            False,
            False,
            starts_with_first,
            Lrhoi,
        )

        # Now sample state sequences for every series
        if n_cores == 1:
            states = []

            for series in range(n_series):
                with warnings.catch_warnings():  # Supress warnings
                    warnings.simplefilter("ignore")
                edss, statess = sample_posterior_series(
                    coef,
                    coef_split_idx,
                    shared_pars,
                    shared_m,
                    [
                        y_split[series] if y_split is not None else None
                        for y_split in split_Ys
                    ],
                    [
                        X_split[series] if X_split is not None else None
                        for X_split in split_Xs
                    ],
                    n_S,
                    obs_fams,
                    d_fams,
                    T[series] if isinstance(T, list) else T,
                    pi,
                    D,
                    M,
                    starts_with_first,
                    ends_with_last,
                    ends_in_last,
                    event_template,
                    scale,
                    build_mat_idx,
                    is_hmp,
                    hmp_fam,
                    n_samples,
                    seed,
                    rho,
                )

                states.append([edss, statess])

        else:  # Compute in parallel

            args = zip(
                repeat(coef),
                repeat(coef_split_idx),
                repeat(shared_pars),
                repeat(shared_m),
                [
                    [
                        y_split[series] if y_split is not None else None
                        for y_split in split_Ys
                    ]
                    for series in range(n_series)
                ],
                [
                    [
                        X_split[series] if X_split is not None else None
                        for X_split in split_Xs
                    ]
                    for series in range(n_series)
                ],
                repeat(n_S),
                repeat(obs_fams),
                repeat(d_fams),
                T if isinstance(T, list) else repeat(T),
                repeat(pi),
                repeat(D),
                repeat(M),
                repeat(starts_with_first),
                repeat(ends_with_last),
                repeat(ends_in_last),
                repeat(event_template),
                repeat(scale),
                repeat(build_mat_idx),
                repeat(is_hmp),
                repeat(hmp_fam),
                repeat(n_samples),
                repeat(seed),
                repeat(rho),
            )

            with warnings.catch_warnings():  # Supress warnings
                warnings.simplefilter("ignore")
                with mp.Pool(processes=n_cores) as pool:
                    eds, states = zip(*pool.starmap(sample_posterior_series, args))

                states = [[edss, statess] for edss, statess in zip(eds, states)]

        return states

    def get_resid(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray],
        Xs: list[scp.sparse.csc_array],
        resid_type: str = "forward",
        transform_to_normal: bool = True,
        n_samples: int = 1000,
        seed: int = 0,
        sid: None | np.ndarray = None,
        tid: None | np.ndarray = None,
    ) -> np.ndarray:
        """Get forward or ordinary pseudo/quantile residuals for a :class:`FTPHSMMFamily` model.

        (Forward) "pseudo-residuals" (also known as "independent quantile residuals") have
        previously been defined by Zucchini et al. (2017) and Dunn & Smyth (1996).

        References:
         - Zucchini, W., MacDonald, I. L., & Langrock, R. (2017). Hidden Markov Models for \
            Time Series: An Introduction Using R, Second Edition (2nd ed.). Chapman and \
            Hall/CRC. https://doi.org/10.1201/b20790
         - Dunn, P. K., & Smyth, G. K. (1996). Randomized Quantile Residuals. Journal of \
            Computational and Graphical Statistics, 5(3), 236–244. \
            https://doi.org/10.2307/1390802

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each paramter of the llk.
        :type coef_split_idx: [int]
        :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as
            ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the
            actual observed data is passed along via the first formula (so it is stored in
            ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula,
            then ``ys`` contains ``None`` at their indices to save memory.
        :type ys: [np.ndarray or None]
        :param Xs: A list of sparse model matrices per likelihood parameter.
        :type Xs: [scp.sparse.csc_array]
        :param resid_type: The type of residual to compute, supported are "forward" and
            "predictive".
        :type resid_type: str, optional
        :param transform_to_normal: The pseudo-residuals are uniformly distributed and can thus be
            transformed to standard normal easily. This parameter controls whether that should
            happen.
        :type transform_to_normal: bool, optional
        :param n_samples: Number of state sequences to sample when ``resid_type=="posterior_dur".
        :type n_samples: int
        :param seed: An optional seed to initialize the sampler when ``resid_type=="posterior_dur".
            **Note**: A value < 0 means random initialization is used. Thus set to >= 0 for
            reproducable results. Defaults to 0
        :type seed: int, optional
        :param sid: Optional Array holding the first sample of each series in the data. Is used
            to split the observation vectors and model matrices into time-series specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type sid: np.ndarray | None, optional
        :param tid: Often, the model matrices for the models of the duration distribution parameters
            will have fewer rows than those of the observation models (because covariates only
            vary with trials). The optional array ``tid`` can then hold indices corresponding to the
            onset of a row-block of a new trial in the model matrices of the duration models.
            Is then used to split the duration model matrices into trial specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type tid: np.ndarray | None, optional
        :return: The residual vector of shape (-1,1)
        :rtype: np.ndarray
        """

        def series_resid(
            coef,
            coef_split_idx,
            shared_pars,
            shared_m,
            ys,
            Xs,
            n_S,
            obs_fams,
            d_fams,
            T,
            pi,
            D,
            M,
            starts_with_first,
            ends_with_last,
            ends_in_last,
            event_template,
            scale,
            resid_type,
            build_mat_idx,
            is_hmp,
            hmp_fam,
            n_samples,
            seed,
            rho,
        ):
            # Compute residual series for an individual series

            # Extract some extra info and define storage
            n_T = len(ys[0])  # Number of time-points in this series
            event_width = 0 if event_template is None else len(event_template)

            # We can pre-compute the bs, cbs, and ds - i.e., observation probabilities
            # and duration probabilities
            y_mat, mus, mus_d, ljbs, cbs, bs, ds, _, _ = _compute_series_probs(
                coef,
                coef_split_idx,
                shared_pars,
                shared_m,
                ys,
                Xs,
                n_S,
                obs_fams,
                d_fams,
                D,
                M,
                event_width,
                event_template,
                scale,
                False,
                False,
                False,
                starts_with_first,
                build_mat_idx,
                is_hmp,
                hmp_fam,
                rho,
            )

            cbs[(np.isnan(cbs) | np.isinf(cbs))] = 0
            bs[(np.isnan(bs) | np.isinf(bs))] = 0
            ds[(np.isnan(ds) | np.isinf(ds))] = 0

            if is_hmp:
                # print(bs.shape, cbs.shape)
                hmp_code = 2 if hmp_fam == "Gaussian" else 1
            else:
                hmp_code = 0
                event_template = np.zeros(1)
                scale = 1

            # Now compute residual vector #
            if resid_type == "forward":

                res = hsmm.forward_resid(
                    cbs,
                    bs,
                    ds,
                    T,
                    pi,
                    event_template,
                    scale,
                    n_T,
                    D,
                    n_S,
                    M,
                    event_width,
                    starts_with_first,
                    ends_with_last,
                    ends_in_last,
                    hmp_code,
                )

            elif resid_type == "predictive":

                res = hsmm.predictive_resid(
                    y_mat,
                    bs,
                    mus,
                    ds,
                    T,
                    pi,
                    event_template,
                    scale,
                    n_T,
                    D,
                    n_S,
                    M,
                    event_width,
                    starts_with_first,
                    ends_with_last,
                    ends_in_last,
                    hmp_code,
                    999 if rho is None else rho,
                )

            elif resid_type == "viterbi_dur":

                # Decode viterbi path
                viterbi = self.decode_viterbi(
                    coef,
                    coef_split_idx,
                    ys,
                    Xs,
                    sid=np.array([0]),
                    tid=np.array([0]),
                    n_cores=1,
                )

                res_sep = compute_dur_res_series(
                    viterbi[0][1],
                    n_S,
                    mus_d,
                    d_fams,
                    starts_with_first,
                )

                # Compute residual of duration MAP under model for stage durations
                res = np.zeros(n_S)

                for j in range(n_S):
                    if res_sep[j] is None:
                        res[j] = np.nan
                    else:
                        res[j] = np.sum(res_sep[j])
                        res[j] /= len(res_sep[j])

                res = res.reshape(1, -1)

            elif resid_type == "posterior_dur":
                state_samples = self.sample_posterior_states(
                    coef,
                    coef_split_idx,
                    ys,
                    Xs,
                    n_samples=n_samples,
                    seed=seed,
                    sid=np.array([0]),
                    tid=np.array([0]),
                    n_cores=1,
                )

                res = np.zeros((1, n_S, n_samples))

                for s in range(n_samples):

                    # Extract durations given sampled series
                    res_sep = compute_dur_res_series(
                        state_samples[0][1][:, s],
                        n_S,
                        mus_d,
                        d_fams,
                        starts_with_first,
                    )

                    # Compute residual of durations under model for stage durations
                    for j in range(n_S):
                        if res_sep[j] is None:
                            res[0, j, s] = np.nan
                        else:
                            res[0, j, s] = np.sum(res_sep[j])
                            res[0, j, s] /= len(res_sep[j])

            return res

        if resid_type not in ["forward", "predictive", "viterbi_dur", "posterior_dur"]:
            raise ValueError(
                f"Requested residual type {resid_type} is not (currently) supported."
            )

        # Extract families, transition matrices, etc.
        n_S = self.llkargs[0]
        obs_fams = self.llkargs[1]
        d_fams = self.llkargs[2]
        T = self.llkargs[3]
        pi = self.llkargs[4]
        if sid is None:
            sid = self.llkargs[5]
        if tid is None:
            tid = self.llkargs[6]
        D = self.llkargs[7]
        M = self.llkargs[8]
        starts_with_first = self.llkargs[9]
        ends_with_last = self.llkargs[10]
        ends_in_last = self.llkargs[11]
        scale = self.llkargs[12]
        event_template = self.llkargs[13]
        hmp_fam = self.llkargs[14]
        n_cores = self.llkargs[15]
        build_mat_idx = self.llkargs[16]
        shared_pars = self.llkargs[17]
        shared_m = self.llkargs[18]
        Lrhoi = self.llkargs[19]
        n_series = len(sid)
        is_hmp = self.__ishmp

        # Initialize potential None arguments
        if T is None:
            T = np.diag(
                np.ones(n_S - 1), 1
            )  # Super diagonal shift matrix for transitions

        if pi is None:
            pi = np.zeros(n_S)  # Start in first state
            pi[0] = 1

        if (
            tid is None
        ):  # Duration model matrices have rows for individual time-points not just trials
            tid = sid

        rho = None
        if Lrhoi is not None:
            rho = np.sqrt(-1 * (np.power(1 / Lrhoi[1, 1], 2) - 1))

        # Split up ys, Xs
        split_Ys, split_Xs = _split_matrices(
            ys,
            Xs,
            shared_pars,
            shared_m,
            obs_fams,
            d_fams,
            sid,
            tid,
            n_S,
            M,
            False,
            False,
            starts_with_first,
            Lrhoi,
        )

        res = []
        if n_cores == 1:

            for series in range(n_series):
                with warnings.catch_warnings():  # Supress warnings
                    warnings.simplefilter("ignore")
                    resids = series_resid(
                        coef,
                        coef_split_idx,
                        shared_pars,
                        shared_m,
                        [
                            y_split[series] if y_split is not None else None
                            for y_split in split_Ys
                        ],
                        [
                            X_split[series] if X_split is not None else None
                            for X_split in split_Xs
                        ],
                        n_S,
                        obs_fams,
                        d_fams,
                        T[series] if isinstance(T, list) else T,
                        pi,
                        D,
                        M,
                        starts_with_first,
                        ends_with_last,
                        ends_in_last,
                        event_template,
                        scale,
                        resid_type,
                        build_mat_idx,
                        is_hmp,
                        hmp_fam,
                        n_samples,
                        seed,
                        rho,
                    )

                    res.append(resids)

            res = np.concatenate(res, axis=0)

        else:  # Compute in parallel

            args = zip(
                repeat(coef),
                repeat(coef_split_idx),
                repeat(shared_pars),
                repeat(shared_m),
                [
                    [
                        y_split[series] if y_split is not None else None
                        for y_split in split_Ys
                    ]
                    for series in range(n_series)
                ],
                [
                    [
                        X_split[series] if X_split is not None else None
                        for X_split in split_Xs
                    ]
                    for series in range(n_series)
                ],
                repeat(n_S),
                repeat(obs_fams),
                repeat(d_fams),
                T if isinstance(T, list) else repeat(T),
                repeat(pi),
                repeat(D),
                repeat(M),
                repeat(starts_with_first),
                repeat(ends_with_last),
                repeat(ends_in_last),
                repeat(event_template),
                repeat(scale),
                repeat(resid_type),
                repeat(build_mat_idx),
                repeat(is_hmp),
                repeat(hmp_fam),
                repeat(n_samples),
                repeat(seed),
                repeat(rho),
            )

            with warnings.catch_warnings():  # Supress warnings
                warnings.simplefilter("ignore")
                with mp.Pool(processes=n_cores) as pool:

                    residss = pool.starmap(series_resid, args)

                res = np.concatenate(residss, axis=0)

        # Transform to normality?
        if transform_to_normal and resid_type == "forward":

            for m in range(res.shape[1]):
                res[:, m] = scp.stats.norm.ppf(res[:, m])

        return res

    def llk(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray],
        Xs: list[scp.sparse.csc_array],
        sid: None | np.ndarray = None,
        tid: None | np.ndarray = None,
    ) -> float:
        """Log-likelihood of model.

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each paramter of the llk.
        :type coef_split_idx: [int]
        :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as
            ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the
            actual observed data is passed along via the first formula (so it is stored in
            ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula,
            then ``ys`` contains ``None`` at their indices to save memory.
        :type ys: [np.ndarray or None]
        :param Xs: A list of sparse model matrices per likelihood parameter.
        :type Xs: [scp.sparse.csc_array]
        :param sid: Optional Array holding the first sample of each series in the data. Is used
            to split the observation vectors and model matrices into time-series specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type sid: np.ndarray | None, optional
        :param tid: Often, the model matrices for the models of the duration distribution parameters
            will have fewer rows than those of the observation models (because covariates only
            vary with trials). The optional array ``tid`` can then hold indices corresponding to the
            onset of a row-block of a new trial in the model matrices of the duration models.
            Is then used to split the duration model matrices into trial specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type tid: np.ndarray | None, optional
        :return: The log-likelihood evaluated at ``coef``.
        :rtype: float
        """

        def series_llk(
            coef,
            coef_split_idx,
            shared_pars,
            shared_m,
            ys,
            Xs,
            n_S,
            obs_fams,
            d_fams,
            T,
            pi,
            D,
            M,
            starts_with_first,
            ends_with_last,
            ends_in_last,
            event_template,
            scale,
            build_mat_idx,
            is_hmp,
            hmp_fam,
            rho,
        ):
            # Compute llk for an individual series - total llk is just sum

            # Extract some extra info and define storage
            n_T = len(ys[0])  # Number of time-points in this series
            event_width = 0 if event_template is None else len(event_template)

            # We can pre-compute the bs and ds - i.e., observation probabilities and
            # duration probabilities
            y_mat, mus, _, ljbs, _, bs, ds, _, _ = _compute_series_probs(
                coef,
                coef_split_idx,
                shared_pars,
                shared_m,
                ys,
                Xs,
                n_S,
                obs_fams,
                d_fams,
                D,
                M,
                event_width,
                event_template,
                scale,
                False,
                False,
                False,
                starts_with_first,
                build_mat_idx,
                is_hmp,
                hmp_fam,
                rho,
            )

            if np.any(np.isnan(ds) | np.isinf(ds)):
                return -np.inf

            if np.any(np.isnan(bs) | np.isinf(bs)):
                return -np.inf

            # Now compute llk #
            if is_hmp:
                hmp_code = 1
            else:
                event_template = np.zeros(1)
                scale = 1
                hmp_code = 0

            llk = hsmm.llk(
                bs,
                ds,
                T,
                pi,
                event_template,
                scale,
                n_T,
                D,
                n_S,
                event_width,
                starts_with_first,
                ends_with_last,
                ends_in_last,
                hmp_code,
            )

            return llk

        # Extract families, transition matrices, etc.
        n_S = self.llkargs[0]
        obs_fams = self.llkargs[1]
        d_fams = self.llkargs[2]
        T = self.llkargs[3]
        pi = self.llkargs[4]
        if sid is None:
            sid = self.llkargs[5]
        if tid is None:
            tid = self.llkargs[6]
        D = self.llkargs[7]
        M = self.llkargs[8]
        starts_with_first = self.llkargs[9]
        ends_with_last = self.llkargs[10]
        ends_in_last = self.llkargs[11]
        scale = self.llkargs[12]
        event_template = self.llkargs[13]
        hmp_fam = self.llkargs[14]
        n_cores = self.llkargs[15]
        build_mat_idx = self.llkargs[16]
        shared_pars = self.llkargs[17]
        shared_m = self.llkargs[18]
        Lrhoi = self.llkargs[19]
        n_series = len(sid)
        is_hmp = self.__ishmp

        # Initialize potential None arguments
        if T is None:
            T = np.diag(
                np.ones(n_S - 1), 1
            )  # Super diagonal shift matrix for transitions

        if pi is None:
            pi = np.zeros(n_S)  # Start in first state
            pi[0] = 1

        if (
            tid is None
        ):  # Duration model matrices have rows for individual time-points not just trials
            tid = sid

        llc = 0
        rho = None
        if Lrhoi is not None:
            d0 = Lrhoi.diagonal()
            llc = np.log(d0[d0 != 1]).sum()
            rho = np.sqrt(-1 * (np.power(1 / Lrhoi[1, 1], 2) - 1))

        # Split up ys, Xs
        split_Ys, split_Xs = _split_matrices(
            ys,
            Xs,
            shared_pars,
            shared_m,
            obs_fams,
            d_fams,
            sid,
            tid,
            n_S,
            M,
            False,
            False,
            starts_with_first,
            Lrhoi,
        )

        # Now compute llk for every individual series and then sum up
        if n_cores == 1:
            llk = 0

            for series in range(n_series):
                with warnings.catch_warnings():  # Supress warnings
                    warnings.simplefilter("ignore")
                    llks = series_llk(
                        coef,
                        coef_split_idx,
                        shared_pars,
                        shared_m,
                        [
                            y_split[series] if y_split is not None else None
                            for y_split in split_Ys
                        ],
                        [
                            X_split[series] if X_split is not None else None
                            for X_split in split_Xs
                        ],
                        n_S,
                        obs_fams,
                        d_fams,
                        T[series] if isinstance(T, list) else T,
                        pi,
                        D,
                        M,
                        starts_with_first,
                        ends_with_last,
                        ends_in_last,
                        event_template,
                        scale,
                        build_mat_idx,
                        is_hmp,
                        hmp_fam,
                        rho,
                    )

                    llk += llks

        else:  # Compute in parallel

            args = zip(
                repeat(coef),
                repeat(coef_split_idx),
                repeat(shared_pars),
                repeat(shared_m),
                [
                    [
                        y_split[series] if y_split is not None else None
                        for y_split in split_Ys
                    ]
                    for series in range(n_series)
                ],
                [
                    [
                        X_split[series] if X_split is not None else None
                        for X_split in split_Xs
                    ]
                    for series in range(n_series)
                ],
                repeat(n_S),
                repeat(obs_fams),
                repeat(d_fams),
                T if isinstance(T, list) else repeat(T),
                repeat(pi),
                repeat(D),
                repeat(M),
                repeat(starts_with_first),
                repeat(ends_with_last),
                repeat(ends_in_last),
                repeat(event_template),
                repeat(scale),
                repeat(build_mat_idx),
                repeat(is_hmp),
                repeat(hmp_fam),
                repeat(rho),
            )

            with warnings.catch_warnings():  # Supress warnings
                warnings.simplefilter("ignore")
                with mp.Pool(processes=n_cores) as pool:

                    llks = pool.starmap(series_llk, args)

                llk = np.sum(llks)

        if np.isnan(llk):
            llk = -np.inf

        # Correct for potential ar1 model
        llk -= llc

        return llk

    def gradient(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray],
        Xs: list[scp.sparse.csc_array],
        sid: None | np.ndarray = None,
        tid: None | np.ndarray = None,
    ) -> np.ndarray:
        """Gradient of log-likelihood evaluated at ``coef``.

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each paramter of the llk.
        :type coef_split_idx: [int]
        :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as
            ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the
            actual observed data is passed along via the first formula (so it is stored in
            ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula,
            then ``ys`` contains ``None`` at their indices to save memory.
        :type ys: [np.ndarray or None]
        :param Xs: A list of sparse model matrices per likelihood parameter.
        :type Xs: [scp.sparse.csc_array]
        :param sid: Optional Array holding the first sample of each series in the data. Is used
            to split the observation vectors and model matrices into time-series specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type sid: np.ndarray | None, optional
        :param tid: Often, the model matrices for the models of the duration distribution parameters
            will have fewer rows than those of the observation models (because covariates only
            vary with trials). The optional array ``tid`` can then hold indices corresponding to the
            onset of a row-block of a new trial in the model matrices of the duration models.
            Is then used to split the duration model matrices into trial specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type tid: np.ndarray | None, optional
        :return: Gradient as array of shape (-1,1)
        :rtype: np.ndarray
        """

        def series_grad(
            coef,
            coef_split_idx,
            shared_pars,
            shared_m,
            ys,
            Xs,
            n_S,
            obs_fams,
            d_fams,
            T,
            pi,
            D,
            M,
            starts_with_first,
            ends_with_last,
            ends_in_last,
            event_template,
            scale,
            build_mat_idx,
            is_hmp,
            hmp_fam,
            rho,
        ):
            # Compute grad for an individual series - total grad is just sum

            # First fix ys and Xs for Nones
            yfix = [y if y is not None else ys[0] for y in ys]
            ys = yfix

            Xfix = [
                X if X is not None else Xs[build_mat_idx[xi]] for xi, X in enumerate(Xs)
            ]
            Xs = Xfix

            # Extract some extra info and define storage
            n_T = len(ys[0])  # Number of time-points in this series
            event_width = 0 if event_template is None else len(event_template)

            if event_template is None:
                event_template = np.zeros(1)

            # We can pre-compute the duration probabilities and observation probabilities for
            # some models. Also their gradients.
            ds = np.zeros((D - 1, n_S * (1 if starts_with_first else 2)), order="F")
            d_val = np.arange(1, D).reshape(-1, 1)

            y_mat = np.zeros((n_T, M), order="F")
            mus = np.zeros((n_T, M, n_S), order="F")
            if is_hmp:
                bs = np.zeros((n_T, event_width, n_S), order="F")

                # Compute weights from rho
                if rho is not None:

                    d0 = 1 / np.sqrt(1 - np.power(rho, 2))  # weight current mean
                    d1 = -rho / np.sqrt(1 - np.power(rho, 2))  # weight previous mean

            else:
                bs = np.zeros((n_T, n_S), order="F")

            b_grad = None
            m_idx_grad = None
            j_idx_grad = None
            d_grad = None

            split_coef = np.split(coef, coef_split_idx)
            total_coef = 0

            # First observation probabilities #

            idx = 0  # Keep track of indices for ys and Xs

            # First check for shared part between emission models.

            # Lists below hold list per par. Each list contains at least one eta/X index/list to
            # be filled with duplicate coefficient indices
            shared_eta = []
            shared_X = []
            shared_coef_idx = []

            # List below holds indices of all duplicated coef so that we can later exclude them
            shared_coef_idx_flat = []
            for par in shared_pars:

                midx = 1
                if shared_m is False:
                    midx = M

                shared_eta.append([])
                shared_X.append([])
                shared_coef_idx.append([])

                # Compute shared part of eta per parameter and m and collect
                # X index + add list to be populated with duplicated/shared coef
                for m in range(midx):
                    shared_eta[par].append(Xs[idx] @ split_coef[idx])
                    shared_X[par].append(idx)
                    shared_coef_idx[par].append([])
                    idx += 1

            # print("OBS:")
            thetas = split_coef[-1].flatten()
            theta_idx = 0
            # Will need to keep track of position of theta gradients
            # since they will need to be moved to the end
            theta_grad_idx = []
            for j in range(n_S):

                iterM = M
                # If emission model is multivariate Gaussian, then it has M n_par!
                if isinstance(obs_fams[j][0], MultiGauss):
                    iterM = 1

                for m in range(iterM):
                    # Get family associated with state j and signal m
                    jmbFam = obs_fams[j][m]

                    if jmbFam is not None:

                        # Multivariate gaussian emission
                        if isinstance(jmbFam, MultiGauss):

                            # Get thetas for this family
                            thetas_j = thetas[
                                theta_idx : theta_idx + jmbFam.extra_coef  # noqa: E203
                            ]

                            Y = np.concatenate(
                                ys[idx : idx + jmbFam.n_par], axis=1  # noqa: E203
                            )

                        # Multiple independent emission signals
                        else:
                            # Only first y associated with state j and signal m is not None
                            yJM = ys[idx]

                        etasbjm = []
                        musbjm = []
                        Xsj = []

                        for par in range(jmbFam.n_par):
                            Xsj.append(Xs[idx])
                            # print(j,split_coef[idx])
                            etasbjm.append(Xs[idx] @ split_coef[idx])

                            # Now handle any shared model component:
                            # 1) add shared eta to etasbjm
                            # 2) expand Xsj[-1] column-wise by shared model matrix
                            # 3) collect indices of duplicated coefficients

                            shared_par_idx = par
                            if isinstance(jmbFam, MultiGauss):
                                shared_par_idx = 0

                            if shared_par_idx in shared_pars:
                                midx = 0
                                if shared_m is False:
                                    midx = m
                                    if isinstance(jmbFam, MultiGauss):
                                        midx = par

                                # 1)
                                etasbjm[-1] += shared_eta[shared_par_idx][midx]

                                # 2)
                                Xsj[-1] = scp.sparse.hstack(
                                    (Xsj[-1], Xs[shared_X[shared_par_idx][midx]])
                                )

                                # 3)
                                start_idx = total_coef + Xs[idx].shape[1]
                                stop_idx = (
                                    total_coef
                                    + Xs[idx].shape[1]
                                    + Xs[shared_X[shared_par_idx][midx]].shape[1]
                                )

                                # Make sure total coef reflects duplicated coef
                                total_coef += Xs[shared_X[shared_par_idx][midx]].shape[
                                    1
                                ]

                                shared_idx = np.arange(
                                    start_idx,
                                    stop_idx,
                                )

                                shared_coef_idx[shared_par_idx][midx].append(shared_idx)

                                shared_coef_idx_flat.extend(shared_idx)

                            musbjm.append(jmbFam.links[par].fi(etasbjm[-1]))
                            total_coef += Xs[idx].shape[1]
                            idx += 1

                        if isinstance(jmbFam, MultiGauss):
                            musbjm = np.concatenate(musbjm, axis=1)

                            # Get transpose of Cholesky of precision
                            R, logdet = jmbFam.getR(thetas_j)

                            # yR is of shape n * m
                            yR = (Y - musbjm) @ R.T

                            # Log-probability of emission vector
                            bs[:, j] = -0.5 * (yR * yR).sum(axis=1) + logdet
                            mus[:, :, j] = musbjm

                        else:
                            # Compute log-probs of signal m under state j
                            lpb = jmbFam.lp(yJM, *musbjm)

                            # Can also compute gradient of log-likelihood of observation
                            # distribution for state j and signal m with respect to coef:

                            # 1) Get partial first derivatives with respect to eta
                            if jmbFam.d_eta is False:
                                d1eta, _, _ = deriv_transform_mu_eta(
                                    yJM, musbjm, jmbFam
                                )
                            else:
                                d1eta = [fd1(yJM, *musbjm) for fd1 in jmbFam.d1]

                        # 2) Get derivatives of log-likelihood of each observation with respect to
                        # coef

                        for par in range(jmbFam.n_par):
                            # n_T * Xsj[par].shape[1] matrix. Each row holds partial derivative of
                            # log-likelihood of corresponding observation with respect to
                            # coefficients in model par.
                            if isinstance(jmbFam, MultiGauss):
                                # RRy is of shape m * n
                                RRy = R.T @ yR.T

                                yRmui = RRy[par, :]
                                grad_par = Xsj[par] * yRmui[:, None]
                            else:
                                grad_par = d1eta[par] * Xsj[par]

                            # Cast to array
                            grad_par = (
                                grad_par.toarray(order="F")
                                if isinstance(grad_par, np.ndarray) is False
                                else grad_par
                            )

                            # Check for nans or infs
                            grad_par[np.isnan(grad_par) | np.isinf(grad_par)] = 0

                            # Concatenate over pars so that in the end gradJM is a
                            # n_T * np.sum([x.shape[1] for x in Xsj]) matrix with each row holding
                            # partial derivatives of log-likelihood of corresponding observation
                            # with respect to **all** coefficients involved in observation model of
                            # state j and signal m
                            if b_grad is None:
                                b_grad = grad_par
                                j_idx_grad = np.zeros(grad_par.shape[1]) + j
                            else:
                                b_grad = np.concatenate([b_grad, grad_par], axis=1)
                                j_idx_grad = np.concatenate(
                                    [j_idx_grad, np.zeros(grad_par.shape[1]) + j]
                                )

                        # Have gradients with respect to coef but need gradients with respect
                        # to log(variance) and covariance parameters for multivariate gaussian
                        if isinstance(jmbFam, MultiGauss):
                            # partial of transpose of cholesky
                            Rdiag = R.diagonal()
                            Rp = np.zeros((jmbFam.n_par, jmbFam.n_par))
                            for mr in range(jmbFam.n_par):

                                for mc in range(mr, jmbFam.n_par):

                                    # Diagonal elements are exp(theta) so partial differs
                                    Rp[mr, mc] = Rdiag[mr] if mr == mc else 1

                                    yRp = (Y - musbjm) @ Rp.T
                                    dldtheta = -np.sum(yR * yRp, axis=1)

                                    # Index function part from WPS (2016)
                                    if mr == mc:
                                        dldtheta += 1

                                    theta_grad_idx.append(len(j_idx_grad))

                                    b_grad = np.concatenate(
                                        [b_grad, dldtheta.reshape(-1, 1)], axis=1
                                    )
                                    j_idx_grad = np.concatenate(
                                        [j_idx_grad, np.zeros(1) + j]
                                    )

                                    # Reset partial of R
                                    Rp[mr, mc] = 0

                            # Update total coef index to reflect theta parameters
                            total_coef += jmbFam.extra_coef

                            # Update theta index
                            theta_idx += jmbFam.extra_coef

                    # Assume HMP/HSMMMVPA pattern for observation model - response variable has to
                    # be cross-correlation between pca components and pattern
                    else:

                        if j % 2 == 1:
                            # Only first y associated with state j and signal m is not None
                            yJM = ys[idx]

                            etasbjm = [Xs[idx] @ split_coef[idx]]

                            # Now handle any shared model component:
                            # 1) add shared eta to etasbjm
                            # 2) expand Xsj[-1] column-wise by shared model matrix - this happens
                            #    later!
                            # 3) collect indices of duplicated coefficients
                            if len(shared_pars) > 0:
                                midx = 0
                                if shared_m is False:
                                    midx = m

                                # 1)
                                etasbjm[-1] += shared_eta[0][midx]

                                # 3)
                                start_idx = total_coef + Xs[idx].shape[1]
                                stop_idx = (
                                    total_coef
                                    + Xs[idx].shape[1]
                                    + Xs[shared_X[0][midx]].shape[1]
                                )

                                # Again make sure that total coef reflects duplicates
                                total_coef += Xs[shared_X[0][midx]].shape[1]

                                shared_idx = np.arange(
                                    start_idx,
                                    stop_idx,
                                )

                                shared_coef_idx[0][midx].append(shared_idx)

                                shared_coef_idx_flat.extend(shared_idx)

                            musbjm = etasbjm  # Assume identity link
                            total_coef += Xs[idx].shape[1]
                            mus[:, m, j] = musbjm[0][:, 0]

                            # Get partial derivative of lpb with respect to coef
                            if 0 in shared_pars:
                                # Can now complete step 2) for shared model component!
                                if shared_m is False:
                                    grad_par = scp.sparse.hstack(
                                        (Xs[idx], Xs[shared_X[0][m]])
                                    )
                                else:
                                    grad_par = scp.sparse.hstack(
                                        (Xs[idx], Xs[shared_X[0][0]])
                                    )
                            else:
                                grad_par = Xs[idx]

                            # Cast to array
                            grad_par = (
                                grad_par.toarray(order="F")
                                if isinstance(grad_par, np.ndarray) is False
                                else grad_par
                            )

                            # Check for nans or infs
                            grad_par[np.isnan(grad_par) | np.isinf(grad_par)] = 0

                            if b_grad is None:
                                b_grad = grad_par
                                m_idx_grad = np.zeros(grad_par.shape[1]) + m
                                j_idx_grad = np.zeros(grad_par.shape[1]) + j
                            else:
                                b_grad = np.concatenate([b_grad, grad_par], axis=1)
                                m_idx_grad = np.concatenate(
                                    [m_idx_grad, np.zeros(grad_par.shape[1]) + m]
                                )
                                j_idx_grad = np.concatenate(
                                    [j_idx_grad, np.zeros(grad_par.shape[1]) + j]
                                )

                            idx += 1

                        # Flat state has no bump and simply gets p(y == zero|\beta) for all obs
                        else:
                            yJM = ys[m]
                            mus[:, m, j] = 0
                            if j == 0:
                                y_mat[:, m] = yJM[:, 0]

                    if is_hmp is False and isinstance(jmbFam, MultiGauss) is False:
                        # All signals are assumed independent given state, covariates,
                        # and coefficients
                        bs[:, j] += lpb.flatten()

                if is_hmp is False:
                    # Turn log-probs into probs again
                    bs[:, j] = np.exp(bs[:, j])
                    if np.any(np.isnan(bs[:, j]) | np.isinf(bs[:, j])):
                        # print("problem with bs",j,m)
                        bs[(np.isnan(bs[:, j]) | np.isinf(bs[:, j])), j] = 0

            # Now state duration probabilities #
            # print("DUR:")
            for j in range(n_S * (1 if starts_with_first else 2)):

                jdFam = d_fams[j]
                if jdFam is not None:

                    etasdj = []
                    musdj = []
                    Xsj = []
                    coefj = []

                    for par in range(jdFam.n_par):
                        # print("d",j,split_coef[idx],par,jdFam.links[par])
                        Xsj.append(Xs[idx])
                        coefj.append(split_coef[idx])
                        etasdj.append(Xs[idx] @ split_coef[idx])
                        musdj.append(jdFam.links[par].fi(etasdj[-1]))
                        total_coef += Xs[idx].shape[1]
                        idx += 1

                    # Compute log-probs of durations under state j
                    lpd = jdFam.lp(d_val, *musdj)
                    # print(lpd[1])
                    # print(jdFam.lp(d_val[1],musdj[0][0],musdj[1][0]))

                    # Can also compute gradient of log-likelihood of duration distribution for
                    # state j with respect to coef:

                    # 1) Get partial first derivatives with respect to eta
                    if jdFam.d_eta is False:
                        d1eta, _, _ = deriv_transform_mu_eta(d_val, musdj, jdFam)
                    else:
                        d1eta = [fd1(d_val, *musdj) for fd1 in jdFam.d1]

                    # 2) Get derivatives of log-likelihood of each observation with respect to coef

                    # DEBUG
                    DEBUG = False
                    if DEBUG:
                        for par in range(jdFam.n_par):
                            print(par, coefj[par], Xsj[par].shape, jdFam, d_val[0])

                            def test_grad(c):
                                c = c.reshape(-1, 1)

                                eta = Xsj[par] @ c
                                mu = jdFam.links[par].fi(eta)

                                if par == 0:
                                    mus = [mu[0], musdj[1][0]]
                                else:
                                    mus = [musdj[0][0], mu[0]]
                                # print(mus,[mu2[0] for mu2 in musdj])
                                return np.exp(jdFam.lp(d_val[0], *mus))

                            gradTest = scp.optimize.approx_fprime(
                                coefj[par].flatten(), test_grad
                            )
                            print(gradTest)
                            gradTest2 = d1eta[par] * Xsj[par]
                            print(gradTest2[0] * np.exp(lpd.flatten())[0])

                            print("Test2")

                            def test_grad2(mu):
                                if par == 0:
                                    mus = [mu[0], musdj[1][0]]
                                else:
                                    mus = [musdj[0][0], mu[0]]
                                print(mus, [mu2[0] for mu2 in musdj])
                                return jdFam.lp(d_val[0], *mus)

                            print(musdj[par][0])
                            gradTest = scp.optimize.approx_fprime(
                                musdj[par][0], test_grad2
                            )
                            print(gradTest)
                            gradTest2 = [fd1(d_val, *musdj) for fd1 in jdFam.d1]
                            print(gradTest2[par][0])
                    # DEBUG END

                    for par in range(jdFam.n_par):
                        # n_D * Xsj[par].shape[1] matrix. Each row holds partial derivative of
                        # log-likelihood of corresponding duration with respect to coefficients in
                        # model par.
                        grad_par = d1eta[par] * Xsj[par]

                        # Cast to array
                        grad_par = (
                            grad_par.toarray(order="F")
                            if isinstance(grad_par, np.ndarray) is False
                            else grad_par
                        )

                        # So grad_par[0,:] holds partial derivatives of log(rds[0,j]) with respect
                        # to coefj[par]. Hence:  rds[0,j] * grad_par[0,:] gives partial derivatives
                        # of rds[0,j] with respect to coefj[par]:
                        rds = np.exp(lpd)  # p(rds[:,j])
                        grad_par_d = rds * grad_par

                        # To get partial derivatives of normalized probabilities
                        # ds[0,j]=rds[0,j]/np.sum(rds[:,j]) with respect to
                        # first coef for example, we need to compute:
                        # (grad_par_d[0,0] * np.sum(rds[:,j]) - rds[0,j] *
                        # (np.sum(grad_par_d[:,0]))) / np.power(np.sum(rds[:,j],2)
                        norm = np.sum(rds)
                        denom = np.power(norm, 2)
                        normed_sum = np.sum(grad_par_d, axis=0)

                        grad_par = grad_par_d * norm
                        grad_par -= rds * normed_sum
                        grad_par /= denom

                        grad_par[np.isnan(grad_par) | np.isinf(grad_par)] = 0

                        # Concatenate over pars so that in the end gradJ is a
                        # n_D * np.sum([x.shape[1] for x in Xsj]) matrix with each row holding
                        # partial derivatives of log-likelihood of corresponding duration
                        # with respect to **all** coefficients involved in duration model of state j
                        if d_grad is None:
                            d_grad = grad_par
                        else:
                            d_grad = np.concatenate([d_grad, grad_par], axis=1)
                        j_idx_grad = np.concatenate(
                            [j_idx_grad, np.zeros(grad_par.shape[1]) + j]
                        )

                else:
                    if j % 2 == 1:
                        lpd = np.zeros(D - 1) - np.inf
                        lpd[int(event_width - 1)] = 0
                    else:
                        raise ValueError(
                            f"Need distribution duration family for state {j}."
                        )

                # Collect duration probabilities associated with state j
                ds[:, j] = np.exp(lpd.flatten())
                ds[:, j] /= np.sum(ds[:, j])
                if np.any(np.isnan(ds[:, j]) | np.isinf(ds[:, j])):
                    # print("problem with ds",j)
                    ds[(np.isnan(ds[:, j]) | np.isinf(ds[:, j])), j] = 0

            # print(np.abs(d_grad).max())
            # print(np.abs(b_grad).max())

            # Now compute gradient #
            # print("total coef",total_coef)
            if total_coef != (b_grad.shape[1] + d_grad.shape[1]):
                raise ValueError(
                    (
                        f"Expected number of coefficients {total_coef} "
                        f"does not match gradient shape {b_grad.shape[1] + d_grad.shape[1]}."
                    )
                )

            if is_hmp:
                hmp_code = 2 if hmp_fam == "Gaussian" else 1

                # Define density callback function
                def ld_wrapper(y: float, pred: float, scale: float):
                    # Exponential case
                    if hmp_code == 1:
                        return scp.stats.expon.logpdf(
                            np.power(y - pred, 2), scale=scale
                        )

                    # Gaussian case
                    elif hmp_code == 2:
                        return scp.stats.norm.logpdf(
                            y, loc=pred, scale=np.power(scale, 0.5)
                        )

                # Compute log densities for bump and flat states
                lbs = np.zeros((n_T, M, event_width, n_S))

                for j in range(n_S):

                    for m in range(M):

                        if j % 2 == 1:
                            # bump

                            for d in range(event_width):

                                # Expected bump for every time-point given dur d in bump
                                Ebump = event_template[d] * mus[:, m, j]

                                if rho is not None:

                                    if d > 0:
                                        # Previous mean was mu at previous time weighted by sine at
                                        # previous dur
                                        Ebump = d0 * Ebump + d1 * (
                                            event_template[d - 1]
                                            * np.concatenate(([0], mus[:-1, m, j]))
                                        )
                                    else:
                                        # Previous mean was 0 (last time-point of preceding flat)
                                        Ebump = d0 * Ebump

                                lbs[:, m, d, j] = ld_wrapper(y_mat[:, m], Ebump, scale)

                                bs[:, d, j] += lbs[:, m, d, j]
                        else:
                            # flat
                            lbs[:, m, 0, j] = ld_wrapper(y_mat[:, m], 0, scale)

                            if rho is not None and j > 0:
                                # Handle time-lag of last sample of previous bump into current flat
                                Ebump = event_template[-1] * np.concatenate(
                                    ([0], mus[:-1, m, j - 1])
                                )

                                lbs[:, m, 1, j] = ld_wrapper(
                                    y_mat[:, m], d1 * Ebump, scale
                                )
                            else:
                                lbs[:, m, 1, j] = lbs[:, m, 0, j]

                            if m == 0:
                                bs[:, :, j] -= np.inf
                                bs[:, 0, j] = 0
                                bs[:, 1, j] = 0

                            bs[:, 0, j] += lbs[:, m, 0, j]
                            bs[:, 1, j] += lbs[:, m, 1, j]

                    bs[:, :, j] = np.exp(bs[:, :, j])

            else:
                hmp_code = 0
                scale = 1
                m_idx_grad = np.zeros_like(j_idx_grad)

            grad = hsmm.llkFTPgrad(
                bs,
                y_mat,
                mus,
                ds,
                T,
                pi,
                b_grad,
                d_grad,
                m_idx_grad,
                j_idx_grad,
                event_template,
                scale,
                n_T,
                D,
                n_S,
                total_coef,
                M,
                event_width,
                starts_with_first,
                ends_with_last,
                ends_in_last,
                hmp_code,
                999 if rho is None else rho,
            )

            # Still need to compute unique gradient vector in case of shared coef.
            if len(shared_pars) > 0:

                # Extend with theta indices if there are any
                shared_coef_idx_flat.extend(theta_grad_idx)
                idx = 0  # Reset index
                grad2 = []  # Unique grad
                for par in shared_pars:

                    midx = 1
                    if shared_m is False:
                        midx = M

                    # Compute shared part of grad by summing over shared coef indices
                    for m in range(midx):
                        grad_shared = np.zeros(Xs[idx].shape[1])

                        for shared_idx in shared_coef_idx[par][m]:
                            grad_shared += grad[shared_idx]

                        grad2.extend(grad_shared)

                        idx += 1

                # Can now identify indices corresponding to truly unique coefficients
                ind_coef_idx_flat = np.array(
                    [
                        cidx
                        for cidx in range(total_coef)
                        if cidx not in shared_coef_idx_flat
                    ]
                )

                # These can simply be appended to gradient
                grad2.extend(grad[ind_coef_idx_flat])

                if len(theta_grad_idx) > 0:
                    # Add theta gradients to end of grad
                    grad2.extend(grad[theta_grad_idx])

                # Cast to np array
                grad = np.array(grad2)

            elif len(theta_grad_idx) > 0:

                grad2 = []  # Unique grad

                # Find indices for coef
                coef_idx_flat = np.array(
                    [cidx for cidx in range(total_coef) if cidx not in theta_grad_idx]
                )

                # These need to be put first in gradient
                grad2.extend(grad[coef_idx_flat])

                # Add theta gradients to end of grad
                grad2.extend(grad[theta_grad_idx])

                # Cast to np array
                grad = np.array(grad2)

            # print(llk)
            return grad.reshape(-1, 1)

        # Extract families, transition matrices, etc.
        n_S = self.llkargs[0]
        obs_fams = self.llkargs[1]
        d_fams = self.llkargs[2]
        T = self.llkargs[3]
        pi = self.llkargs[4]
        if sid is None:
            sid = self.llkargs[5]
        if tid is None:
            tid = self.llkargs[6]
        D = self.llkargs[7]
        M = self.llkargs[8]
        starts_with_first = self.llkargs[9]
        ends_with_last = self.llkargs[10]
        ends_in_last = self.llkargs[11]
        scale = self.llkargs[12]
        event_template = self.llkargs[13]
        hmp_fam = self.llkargs[14]
        n_cores = self.llkargs[15]
        build_mat_idx = self.llkargs[16]
        shared_pars = self.llkargs[17]
        shared_m = self.llkargs[18]
        Lrhoi = self.llkargs[19]
        n_series = len(sid)
        is_hmp = self.__ishmp

        # Initialize potential None arguments
        if T is None:
            T = np.diag(
                np.ones(n_S - 1), 1
            )  # Super diagonal shift matrix for transitions

        if pi is None:
            pi = np.zeros(n_S)  # Start in first state
            pi[0] = 1

        if (
            tid is None
        ):  # Duration model matrices have rows for individual time-points not just trials
            tid = sid

        rho = None
        if Lrhoi is not None:
            rho = np.sqrt(-1 * (np.power(1 / Lrhoi[1, 1], 2) - 1))

        # Split up ys, Xs
        split_Ys, split_Xs = _split_matrices(
            ys,
            Xs,
            shared_pars,
            shared_m,
            obs_fams,
            d_fams,
            sid,
            tid,
            n_S,
            M,
            False,
            False,
            starts_with_first,
            Lrhoi,
        )

        # Now compute grad for every individual series and then sum up
        if n_cores == 1:
            grad = None
            for series in range(n_series):
                with warnings.catch_warnings():  # Supress warnings
                    warnings.simplefilter("ignore")
                    grads = series_grad(
                        coef,
                        coef_split_idx,
                        shared_pars,
                        shared_m,
                        [
                            y_split[series] if y_split is not None else None
                            for y_split in split_Ys
                        ],
                        [
                            X_split[series] if X_split is not None else None
                            for X_split in split_Xs
                        ],
                        n_S,
                        obs_fams,
                        d_fams,
                        T[series] if isinstance(T, list) else T,
                        pi,
                        D,
                        M,
                        starts_with_first,
                        ends_with_last,
                        ends_in_last,
                        event_template,
                        scale,
                        build_mat_idx,
                        is_hmp,
                        hmp_fam,
                        rho,
                    )
                # print(grads)
                if grad is None:
                    grad = grads
                else:
                    grad += grads

        else:  # Compute in parallel

            args = zip(
                repeat(coef),
                repeat(coef_split_idx),
                repeat(shared_pars),
                repeat(shared_m),
                [
                    [
                        y_split[series] if y_split is not None else None
                        for y_split in split_Ys
                    ]
                    for series in range(n_series)
                ],
                [
                    [
                        X_split[series] if X_split is not None else None
                        for X_split in split_Xs
                    ]
                    for series in range(n_series)
                ],
                repeat(n_S),
                repeat(obs_fams),
                repeat(d_fams),
                T if isinstance(T, list) else repeat(T),
                repeat(pi),
                repeat(D),
                repeat(M),
                repeat(starts_with_first),
                repeat(ends_with_last),
                repeat(ends_in_last),
                repeat(event_template),
                repeat(scale),
                repeat(build_mat_idx),
                repeat(is_hmp),
                repeat(hmp_fam),
                repeat(rho),
            )

            with warnings.catch_warnings():  # Supress warnings
                warnings.simplefilter("ignore")
                with mp.Pool(processes=n_cores) as pool:
                    grads = pool.starmap(series_grad, args)

                grad = np.sum(grads, axis=0)

        return grad


class HSMMFamily(GSMMFamily):
    """_summary_

    :param pars: Number of parameters of the log-likelihood
    :type pars: int
    :param links: List of link functions to use in the models of different parameters
    :type links: list[Link]
    :param n_S: Number of latent states.
    :type n_S: int
    :param obs_fams: Distribution of emissions - one per state
    :type obs_fams: list[list[GAMLSSFamily]]
    :param d_fams: Distribution of state durations - one per state
    :type d_fams: list[GAMLSSFamily]
    :param sid: Array holding the first sample of each series in the data. Can be used to split the
        observation vectors and model matrices into time-series specific versions
    :type sid: np.ndarray
    :param tid: Often, the model matrices for the models of the duration distribution parameters,
        initial state distribution parameters, and state transition parameters will have fewer rows
        than those of the observation models (e.g., because covariates only vary with trials). The
        optional array ``tid`` can then hold indices corresponding to the onset of a row-block of a
        new trial in the model matrices of these models. Can then be used to split their model
        matrices into trial specific versions. If this is set to ``None``, it will be set to
        ``sid``.
    :type tid: np.ndarray | None
    :param D: Max duration a state can take on (**in samples**)
    :type D: int
    :param M: Number of signals recorded if the HSMM is multivariate. All signals are assumed to be
        independent from one another given covariates and coefficients.
    :type M: int
    :param starts_with_first: Whether the first state starts with the first observation or might
        have been going on for a max duration of ``D``.
    :type starts_with_first: bool
    :param ends_with_last: Whether the last state ends with the last observation or can continue.
    :type ends_with_last: bool
    :param ends_in_last: Whether the Markov chain ends in the last state. Not really useful for
        this class..
    :type ends_in_last: bool
    :param n_cores: Number of cores to use in the computations, defaults to 1.
    :type n_cores: int
    :param build_mat_idx: If not all matrices are built explicitly, this has to be a list of indices
        per parameter modeled, pointing at the index of ``Xs`` holding the matrix that should be
        used by that parameter.
    :type build_mat_idx: list[int] | None
    :param shared_pars: A list containing indices of emission distribution parameters for which
        at least one (in case ``shared_m is False`` one for each of the ``M`` signals) formula
        has been provided containing terms to be shared between the ``n_S`` states. Defaults to
        ``[]`` meaning no terms are shared.
    :type shared_pars: list[int], optional
    :param shared_m: Bool indicating whether any shared terms are not just shared between states but
        also between the ``M`` signals. Defaults to ``False``
    :type shared_m: bool, optional
    """

    def __init__(
        self,
        pars: int,
        links: list[Link],
        n_S: int,
        obs_fams: list[list[GAMLSSFamily]],
        d_fams: list[GAMLSSFamily],
        sid: np.ndarray,
        tid: np.ndarray | None,
        D: int,
        M: int,
        starts_with_first: bool,
        ends_with_last: bool,
        ends_in_last: bool,
        n_cores=1,
        build_mat_idx: list[int] | None = None,
        shared_pars: list[int] = [],
        shared_m: bool = False,
    ) -> None:

        super().__init__(
            pars,
            links,
            n_S,
            obs_fams,
            d_fams,
            sid,
            tid,
            D,
            M,
            starts_with_first,
            ends_with_last,
            ends_in_last,
            n_cores,
            build_mat_idx,
            shared_pars,
            shared_m,
        )

        self.extra_coef = 0
        for state_obs in obs_fams:
            if isinstance(state_obs[0], MultiGauss):
                self.extra_coef += state_obs[0].extra_coef

    def compute_od_probs(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray],
        Xs: list[scp.sparse.csc_array],
        log: bool = True,
        sid: None | np.ndarray = None,
        tid: None | np.ndarray = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Computes the (log)-probabilities of observations and stage durations under given model.

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each paramter of the llk.
        :type coef_split_idx: [int]
        :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as
            ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the
            actual observed data is passed along via the first formula (so it is stored in
            ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula,
            then ``ys`` contains ``None`` at their indices to save memory.
        :type ys: [np.ndarray or None]
        :param Xs: A list of sparse model matrices per likelihood parameter.
        :type Xs: [scp.sparse.csc_array]
        :param log: Boolena indicating whether to compute probabilities or log-probabilities.
            Defaults to ``True`` - meaning log-probabilities are computed.
        :type log: bool, optional
        :param sid: Optional Array holding the first sample of each series in the data. Is used
            to split the observation vectors and model matrices into time-series specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type sid: np.ndarray | None, optional
        :param tid: Often, the model matrices for the models of the duration distribution parameters
            will have fewer rows than those of the observation models (because covariates only
            vary with trials). The optional array ``tid`` can then hold indices corresponding to the
            onset of a row-block of a new trial in the model matrices of the duration models.
            Is then used to split the duration model matrices into trial specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type tid: np.ndarray | None, optional
        :return: List of tuples - one tuple per time-series. Each tuple holds two arrays, first
            dimension of both is of size ``n_S``, corresponding to the number of latent states.
            First array has three dimensions, holds (log)-probabilities of all observations per
            state (second dimension) and signal recorded (third dimension), second array is
            two-dimensional, holds (log)-probabilities of all possible stage durations per state.
        :rtype: list[tuple[np.ndarray,np.ndarray]]
        """

        def series_log_prob(
            coef,
            coef_split_idx,
            shared_pars,
            shared_m,
            ys,
            Xs,
            n_S,
            obs_fams,
            d_fams,
            D,
            M,
            log,
            starts_with_first,
            build_mat_idx,
        ):
            # Compute log probs for an individual series

            _, _, _, _, _, bs, ds, _, _ = _compute_series_probs(
                coef,
                coef_split_idx,
                shared_pars,
                shared_m,
                ys,
                Xs,
                n_S,
                obs_fams,
                d_fams,
                D,
                M,
                None,
                None,
                None,
                log,
                False,
                False,
                starts_with_first,
                build_mat_idx,
            )
            bs[(np.isnan(bs) | np.isinf(bs))] = -np.inf if log else 0
            ds[(np.isnan(ds) | np.isinf(ds))] = -np.inf if log else 0

            return bs, ds

        # Extract families, transition matrices, etc.
        n_S = self.llkargs[0]
        obs_fams = self.llkargs[1]
        d_fams = self.llkargs[2]
        if sid is None:
            sid = self.llkargs[3]
        if tid is None:
            tid = self.llkargs[4]
        D = self.llkargs[5]
        M = self.llkargs[6]
        starts_with_first = self.llkargs[7]
        n_cores = self.llkargs[10]
        build_mat_idx = self.llkargs[11]
        shared_pars = self.llkargs[12]
        shared_m = self.llkargs[13]
        n_series = len(sid)

        if (
            tid is None
        ):  # Duration model matrices have rows for individual time-points not just trials
            tid = sid

        # Split up ys, Xs
        split_Ys, split_Xs = _split_matrices(
            ys,
            Xs,
            shared_pars,
            shared_m,
            obs_fams,
            d_fams,
            sid,
            tid,
            n_S,
            M,
            False,
            False,
            starts_with_first,
        )

        # Now compute observation/duration probabilities for every individual series and then sum up
        if n_cores == 1:
            ps = []

            for series in range(n_series):
                bss, dss = series_log_prob(
                    coef,
                    coef_split_idx,
                    shared_pars,
                    shared_m,
                    [
                        y_split[series] if y_split is not None else None
                        for y_split in split_Ys
                    ],
                    [
                        X_split[series] if X_split is not None else None
                        for X_split in split_Xs
                    ],
                    n_S,
                    obs_fams,
                    d_fams,
                    D,
                    M,
                    log,
                    starts_with_first,
                    build_mat_idx,
                )

                ps.append([bss, dss])

        else:  # Compute in parallel

            args = zip(
                repeat(coef),
                repeat(coef_split_idx),
                repeat(shared_pars),
                repeat(shared_m),
                [
                    [
                        y_split[series] if y_split is not None else None
                        for y_split in split_Ys
                    ]
                    for series in range(n_series)
                ],
                [
                    [
                        X_split[series] if X_split is not None else None
                        for X_split in split_Xs
                    ]
                    for series in range(n_series)
                ],
                repeat(n_S),
                repeat(obs_fams),
                repeat(d_fams),
                repeat(D),
                repeat(M),
                repeat(log),
                repeat(starts_with_first),
                repeat(build_mat_idx),
            )

            with mp.Pool(processes=n_cores) as pool:
                bs, ds = zip(*pool.starmap(series_log_prob, args))

            ps = [[bss, dss] for bss, dss in zip(bs, ds)]

        return ps

    def compute_Tpi(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray],
        Xs: list[scp.sparse.csc_array],
        sid: None | np.ndarray = None,
        tid: None | np.ndarray = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Computes the initial state distribution and state. transition matrix for ``coef`` and
        ``Xs``.

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each paramter of the llk.
        :type coef_split_idx: [int]
        :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as
            ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the
            actual observed data is passed along via the first formula (so it is stored in
            ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula,
            then ``ys`` contains ``None`` at their indices to save memory.
        :type ys: [np.ndarray or None]
        :param Xs: A list of sparse model matrices per likelihood parameter.
        :type Xs: [scp.sparse.csc_array]
        :param sid: Optional Array holding the first sample of each series in the data. Is used
            to split the observation vectors and model matrices into time-series specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type sid: np.ndarray | None, optional
        :param tid: Often, the model matrices for the models of the duration distribution parameters
            will have fewer rows than those of the observation models (because covariates only
            vary with trials). The optional array ``tid`` can then hold indices corresponding to the
            onset of a row-block of a new trial in the model matrices of the duration models.
            Is then used to split the duration model matrices into trial specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type tid: np.ndarray | None, optional
        :return: List of tuples - one tuple per time-series. Each tuple holds two arrays. First
            array is of dimension ``n_S * n_S`` and corresponds to the state transition matrix.
            Second array is of dimension ``n_S`` and corresponds to initial state distribution.
        :rtype: list[tuple[np.ndarray,np.ndarray]]
        """

        def series_Tpi(
            coef,
            coef_split_idx,
            shared_pars,
            shared_m,
            ys,
            Xs,
            n_S,
            obs_fams,
            d_fams,
            M,
            starts_with_first,
            build_mat_idx,
        ):
            # Compute state transition and initial state disribution matrix for a given series.

            _, _, _, _, _, _, _, T, pi = _compute_series_probs(
                coef,
                coef_split_idx,
                shared_pars,
                shared_m,
                ys,
                Xs,
                n_S,
                obs_fams,
                d_fams,
                D,
                M,
                None,
                None,
                None,
                False,
                True,
                True,
                starts_with_first,
                build_mat_idx,
            )

            return T, pi

        # Extract families, transition matrices, etc.
        n_S = self.llkargs[0]
        obs_fams = self.llkargs[1]
        d_fams = self.llkargs[2]
        if sid is None:
            sid = self.llkargs[3]
        if tid is None:
            tid = self.llkargs[4]
        D = self.llkargs[5]
        M = self.llkargs[6]
        starts_with_first = self.llkargs[7]
        n_cores = self.llkargs[10]
        build_mat_idx = self.llkargs[11]
        shared_pars = self.llkargs[12]
        shared_m = self.llkargs[13]
        n_series = len(sid)

        # Initialize potential None arguments

        if (
            tid is None
        ):  # Duration model matrices have rows for individual time-points not just trials
            tid = sid

        # Split up ys, Xs
        split_Ys, split_Xs = _split_matrices(
            ys,
            Xs,
            shared_pars,
            shared_m,
            obs_fams,
            d_fams,
            sid,
            tid,
            n_S,
            M,
            True,
            True,
            starts_with_first,
        )

        # Now compute observation/duration probabilities for every individual series and then sum up
        if n_cores == 1:
            ps = []

            for series in range(n_series):
                T, pi = series_Tpi(
                    coef,
                    coef_split_idx,
                    shared_pars,
                    shared_m,
                    [
                        y_split[series] if y_split is not None else None
                        for y_split in split_Ys
                    ],
                    [
                        X_split[series] if X_split is not None else None
                        for X_split in split_Xs
                    ],
                    n_S,
                    obs_fams,
                    d_fams,
                    M,
                    starts_with_first,
                    build_mat_idx,
                )

                ps.append([T, pi])

        else:  # Compute in parallel

            args = zip(
                repeat(coef),
                repeat(coef_split_idx),
                repeat(shared_pars),
                repeat(shared_m),
                [
                    [
                        y_split[series] if y_split is not None else None
                        for y_split in split_Ys
                    ]
                    for series in range(n_series)
                ],
                [
                    [
                        X_split[series] if X_split is not None else None
                        for X_split in split_Xs
                    ]
                    for series in range(n_series)
                ],
                repeat(n_S),
                repeat(obs_fams),
                repeat(d_fams),
                repeat(M),
                repeat(starts_with_first),
                repeat(build_mat_idx),
            )

            with mp.Pool(processes=n_cores) as pool:
                Ts, pis = zip(*pool.starmap(series_Tpi, args))

            ps = [[T, pi] for T, pi in zip(Ts, pis)]

        return ps

    def predict(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray],
        Xs: list[scp.sparse.csc_array],
        sid: None | np.ndarray = None,
        tid: None | np.ndarray = None,
        seed: int | None = 0,
        size: int = 1,
    ) -> list[np.ndarray, np.ndarray]:
        """Predicts ``size`` state and observation sequences given model matrices ``Xs`` and
        coefficients ``coef``.

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each paramter of the llk.
        :type coef_split_idx: [int]
        :param Xs: A list of sparse model matrices per likelihood parameter.
        :type Xs: [scp.sparse.csc_array]
        :param sid: Optional Array holding the first sample of each series in the data. Is used
            to split the observation vectors and model matrices into time-series specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type sid: np.ndarray | None, optional
        :param tid: Often, the model matrices for the models of the duration distribution parameters
            will have fewer rows than those of the observation models (because covariates only
            vary with trials). The optional array ``tid`` can then hold indices corresponding to the
            onset of a row-block of a new trial in the model matrices of the duration models.
            Is then used to split the duration model matrices into trial specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type tid: np.ndarray | None, optional
        :param seed: An optional seed to initialize the sampler. If this is set to None, then random
            initialization is performed.
        :type seed: int, optional
        :param size: Number of state and observation sequences to generate (per series). Defaults to
            1.
        :type size: int, optional
        :return: A list holding two numpy arrays per series implied by ``sid``. First numpy array is
            of dimension ``(nT, M, size)``, where ``nT`` is the number of time-points implied by
            the number of rows of the ``Xs`` corresponding to that series. ``M`` is the number of
            emitting signals. the second array is of dimension ``(nT, size)``. The first array holds
            predicted observation sequences, the second array holds predicted state sequences.
        :rtype: list[np.ndarray, np.ndarray]
        """

        def predict_series(
            coef,
            coef_split_idx,
            shared_pars,
            shared_m,
            ys,
            Xs,
            n_S,
            obs_fams,
            d_fams,
            D,
            M,
            starts_with_first,
            build_mat_idx,
            n_samples,
            seed,
        ):

            # Extract some extra info and define storage
            n_T = len(ys[0])  # Number of time-points in this series

            # Sample emissions
            emissions = _sample_series_emissions(
                coef,
                coef_split_idx,
                shared_pars,
                shared_m,
                Xs,
                n_S,
                obs_fams,
                M,
                None,
                build_mat_idx,
                False,
                n_samples,
                seed,
            )

            # Now sample state sequences and combine that with the emission samples
            emission_samples = np.zeros((n_T, M, n_samples))
            state_samples = np.zeros((n_T, n_samples))

            # Compute state transition, duration, and initial state disribution matrices for a
            # given series.
            _, _, _, _, _, _, ds, T, pi = _compute_series_probs(
                coef,
                coef_split_idx,
                shared_pars,
                shared_m,
                ys,
                Xs,
                n_S,
                obs_fams,
                d_fams,
                D,
                M,
                None,
                None,
                None,
                False,
                True,
                True,
                starts_with_first,
                build_mat_idx,
            )

            np_gen = np.random.default_rng(seed)
            d_vals = np.arange(1, D).reshape(-1, 1)
            state_vals = np.arange(n_S)

            for sample in range(n_samples):

                # Run the state generation process forward and select correct emissions for
                # chosen states

                # Select initial state
                state = np_gen.choice(state_vals, p=pi)

                # and duration of initial state
                dur = np_gen.choice(d_vals, p=ds[:, state])

                if seed is not None:
                    seed += 1

                t = 0
                while t < n_T:

                    if dur < 0:
                        # Sample next state and dur
                        state = np_gen.choice(state_vals, p=T[state, :])

                        dur = np_gen.choice(
                            d_vals,
                            p=ds[:, (state if starts_with_first else state + n_S)],
                        )

                    # Collect emission and state predictions
                    emission_samples[t, :, sample] = emissions[t, state, :, sample]
                    state_samples[t, sample] = state

                    # Update time and dur
                    t += 1
                    dur -= 1

                    # And seed
                    if seed is not None:
                        seed += 1

            return emission_samples, state_samples

        # Extract families, transition matrices, etc.
        n_S = self.llkargs[0]
        obs_fams = self.llkargs[1]
        d_fams = self.llkargs[2]
        if sid is None:
            sid = self.llkargs[3]
        if tid is None:
            tid = self.llkargs[4]
        D = self.llkargs[5]
        M = self.llkargs[6]
        starts_with_first = self.llkargs[7]
        n_cores = self.llkargs[10]
        build_mat_idx = self.llkargs[11]
        shared_pars = self.llkargs[12]
        shared_m = self.llkargs[13]
        n_series = len(sid)

        # Initialize potential None arguments

        if (
            tid is None
        ):  # Duration model matrices have rows for individual time-points not just trials
            tid = sid

        # Split up ys, Xs
        split_Ys, split_Xs = _split_matrices(
            ys,
            Xs,
            shared_pars,
            shared_m,
            obs_fams,
            d_fams,
            sid,
            tid,
            n_S,
            M,
            True,
            True,
            starts_with_first,
        )

        # Now compute predictions for each series
        if n_cores == 1:
            predictions = []

            for series in range(n_series):
                with warnings.catch_warnings():  # Supress warnings
                    warnings.simplefilter("ignore")
                    e_predss, s_predss = predict_series(
                        coef,
                        coef_split_idx,
                        shared_pars,
                        shared_m,
                        [
                            y_split[series] if y_split is not None else None
                            for y_split in split_Ys
                        ],
                        [
                            X_split[series] if X_split is not None else None
                            for X_split in split_Xs
                        ],
                        n_S,
                        obs_fams,
                        d_fams,
                        D,
                        M,
                        starts_with_first,
                        build_mat_idx,
                        size,
                        seed,
                    )

                    predictions.append([e_predss, s_predss])

        else:  # Compute in parallel

            args = zip(
                repeat(coef),
                repeat(coef_split_idx),
                repeat(shared_pars),
                repeat(shared_m),
                [
                    [
                        y_split[series] if y_split is not None else None
                        for y_split in split_Ys
                    ]
                    for series in range(n_series)
                ],
                [
                    [
                        X_split[series] if X_split is not None else None
                        for X_split in split_Xs
                    ]
                    for series in range(n_series)
                ],
                repeat(n_S),
                repeat(obs_fams),
                repeat(d_fams),
                repeat(D),
                repeat(M),
                repeat(starts_with_first),
                repeat(build_mat_idx),
                repeat(size),
                repeat(seed),
            )

            with warnings.catch_warnings():  # Supress warnings
                warnings.simplefilter("ignore")
                with mp.Pool(processes=n_cores) as pool:
                    e_preds, s_preds = zip(*pool.starmap(predict_series, args))

                predictions = [
                    [e_predss, s_predss] for e_predss, s_predss in zip(e_preds, s_preds)
                ]

        return predictions

    def decode_viterbi(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray],
        Xs: list[scp.sparse.csc_array],
        sid: None | np.ndarray = None,
        tid: None | np.ndarray = None,
        n_cores: None | int = None,
    ) -> list[tuple[int, np.ndarray]]:
        """Perform viterbi decoding of state sequence for a :class:`HSMMFamily` model.

        :param coef: _description_
        :type coef: np.ndarray
        :param coef_split_idx: _description_
        :type coef_split_idx: list[int]
        :param ys: _description_
        :type ys: list[np.ndarray]
        :param Xs: _description_
        :type Xs: list[scp.sparse.csc_array]
        :param sid: Optional Array holding the first sample of each series in the data. Is used
            to split the observation vectors and model matrices into time-series specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type sid: np.ndarray | None, optional
        :param tid: Often, the model matrices for the models of the duration distribution parameters
            will have fewer rows than those of the observation models (because covariates only
            vary with trials). The optional array ``tid`` can then hold indices corresponding to the
            onset of a row-block of a new trial in the model matrices of the duration models.
            Is then used to split the duration model matrices into trial specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type tid: np.ndarray | None, optional
        :param n_cores: Optionally, the number of cores to use during decoding. Allows to over-write
            the value passed to the constructor of ``self``. If set to None, the value passed to
            the constructor is used. Defaults to None
        :type n_cores: int| None, optional
        :return: A list with a tuple per time-series. First element of tuple is excess duration of
            final state, second is a np.array with the decoded state sequence matching the length
            of the time-series.
        :rtype: list[tuple[int,np.ndarray]]
        """

        def series_viterbi(
            coef,
            coef_split_idx,
            shared_pars,
            shared_m,
            ys,
            Xs,
            n_S,
            obs_fams,
            d_fams,
            D,
            M,
            starts_with_first,
            ends_with_last,
            ends_in_last,
            build_mat_idx,
        ):
            # Perform viterbi decoding for an individual series

            # Extract some extra info and define storage
            n_T = len(ys[0])  # Number of time-points in this series

            # We can pre-compute the bs and ds - i.e., observation probabilities and duration
            # probabilities
            _, _, _, _, _, lbs, lds, lT, lpi = _compute_series_probs(
                coef,
                coef_split_idx,
                shared_pars,
                shared_m,
                ys,
                Xs,
                n_S,
                obs_fams,
                d_fams,
                D,
                M,
                None,
                None,
                None,
                True,
                True,
                True,
                starts_with_first,
                build_mat_idx,
            )
            lbs[np.isnan(lbs)] = -np.inf
            lds[np.isnan(lds)] = -np.inf

            hmp_code = 0
            event_template = np.zeros(1)
            scale = 1
            event_width = 0

            max_ed, states, deltas, psis = hsmm.viterbi(
                lbs,
                lds,
                lT,
                lpi,
                event_template,
                scale,
                n_T,
                D,
                n_S,
                event_width,
                starts_with_first,
                ends_with_last,
                ends_in_last,
                hmp_code,
            )

            return max_ed, states

        # Extract families, transition matrices, etc.
        n_S = self.llkargs[0]
        obs_fams = self.llkargs[1]
        d_fams = self.llkargs[2]
        if sid is None:
            sid = self.llkargs[3]
        if tid is None:
            tid = self.llkargs[4]
        D = self.llkargs[5]
        M = self.llkargs[6]
        starts_with_first = self.llkargs[7]
        ends_with_last = self.llkargs[8]
        ends_in_last = self.llkargs[9]

        if n_cores is None:
            n_cores = self.llkargs[10]
        build_mat_idx = self.llkargs[11]
        shared_pars = self.llkargs[12]
        shared_m = self.llkargs[13]
        n_series = len(sid)

        # Initialize potential None arguments

        # Duration model matrices
        if tid is None:
            tid = sid

        # Split up ys, Xs
        split_Ys, split_Xs = _split_matrices(
            ys,
            Xs,
            shared_pars,
            shared_m,
            obs_fams,
            d_fams,
            sid,
            tid,
            n_S,
            M,
            True,
            True,
            starts_with_first,
        )

        # Now compute optimal state sequence for every series
        if n_cores == 1:
            states = []

            for series in range(n_series):
                with warnings.catch_warnings():  # Supress warnings
                    warnings.simplefilter("ignore")
                    edss, statess = series_viterbi(
                        coef,
                        coef_split_idx,
                        shared_pars,
                        shared_m,
                        [
                            y_split[series] if y_split is not None else None
                            for y_split in split_Ys
                        ],
                        [
                            X_split[series] if X_split is not None else None
                            for X_split in split_Xs
                        ],
                        n_S,
                        obs_fams,
                        d_fams,
                        D,
                        M,
                        starts_with_first,
                        ends_with_last,
                        ends_in_last,
                        build_mat_idx,
                    )

                    states.append([edss, statess])

        else:  # Compute in parallel

            args = zip(
                repeat(coef),
                repeat(coef_split_idx),
                repeat(shared_pars),
                repeat(shared_m),
                [
                    [
                        y_split[series] if y_split is not None else None
                        for y_split in split_Ys
                    ]
                    for series in range(n_series)
                ],
                [
                    [
                        X_split[series] if X_split is not None else None
                        for X_split in split_Xs
                    ]
                    for series in range(n_series)
                ],
                repeat(n_S),
                repeat(obs_fams),
                repeat(d_fams),
                repeat(D),
                repeat(M),
                repeat(starts_with_first),
                repeat(ends_with_last),
                repeat(ends_in_last),
                repeat(build_mat_idx),
            )

            with warnings.catch_warnings():  # Supress warnings
                warnings.simplefilter("ignore")
                with mp.Pool(processes=n_cores) as pool:
                    eds, states = zip(*pool.starmap(series_viterbi, args))

                states = [[edss, statess] for edss, statess in zip(eds, states)]

        return states

    def sample_posterior_states(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray],
        Xs: list[scp.sparse.csc_array],
        n_samples: int,
        seed: int = 0,
        sid: None | np.ndarray = None,
        tid: None | np.ndarray = None,
        n_cores: None | int = None,
    ) -> float:
        """Samples the posterior of state sequences given parameter estimates and data.

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each paramter of the llk.
        :type coef_split_idx: [int]
        :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as
            ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the
            actual observed data is passed along via the first formula (so it is stored in
            ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula,
            then ``ys`` contains ``None`` at their indices to save memory.
        :type ys: [np.ndarray or None]
        :param Xs: A list of sparse model matrices per likelihood parameter.
        :type Xs: [scp.sparse.csc_array]
        :param n_samples: Number of state sequences to sample.
        :type n_samples: int
        :param seed: An optional seed to initialize the sampler. **Note**: A value < 0 means random
            initialization is used. Thus set to >= 0 for reproducable results. Defaults to 0
        :type seed: int, optional
        :param sid: Optional Array holding the first sample of each series in the data. Is used
            to split the observation vectors and model matrices into time-series specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type sid: np.ndarray | None, optional
        :param tid: Often, the model matrices for the models of the duration distribution parameters
            will have fewer rows than those of the observation models (because covariates only
            vary with trials). The optional array ``tid`` can then hold indices corresponding to the
            onset of a row-block of a new trial in the model matrices of the duration models.
            Is then used to split the duration model matrices into trial specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type tid: np.ndarray | None, optional
        :param n_cores: Optionally, the number of cores to use during sampling. Allows to over-write
            the value passed to the constructor of ``self``. If set to None, the value passed to
            the constructor is used. Defaults to None
        :type n_cores: int| None, optional
        :return: The log-likelihood evaluated at ``coef``.
        :rtype: float
        """

        def sample_posterior_series(
            coef,
            coef_split_idx,
            shared_pars,
            shared_m,
            ys,
            Xs,
            n_S,
            obs_fams,
            d_fams,
            D,
            M,
            starts_with_first,
            ends_with_last,
            ends_in_last,
            build_mat_idx,
            n_samples,
            seed,
        ):
            # Compute llk for an individual series - total llk is just sum

            # Extract some extra info and define storage
            n_T = len(ys[0])  # Number of time-points in this series

            # We can pre-compute the bs and ds - i.e., observation probabilities and
            # duration probabilities
            _, _, _, _, _, bs, ds, T, pi = _compute_series_probs(
                coef,
                coef_split_idx,
                shared_pars,
                shared_m,
                ys,
                Xs,
                n_S,
                obs_fams,
                d_fams,
                D,
                M,
                None,
                None,
                None,
                False,
                True,
                True,
                starts_with_first,
                build_mat_idx,
            )

            bs[np.isnan(bs) | np.isinf(bs)] = 0
            ds[np.isnan(ds) | np.isinf(ds)] = 0

            # Now sample state sequences
            event_template = np.zeros(1)
            scale = 1
            hmp_code = 0
            event_width = 0

            eds, states = hsmm.sample_backwards(
                bs,
                ds,
                T,
                pi,
                event_template,
                scale,
                n_T,
                D,
                n_S,
                event_width,
                n_samples,
                starts_with_first,
                ends_with_last,
                ends_in_last,
                hmp_code,
                seed,
            )

            return eds, states

        # Extract families, transition matrices, etc.
        n_S = self.llkargs[0]
        obs_fams = self.llkargs[1]
        d_fams = self.llkargs[2]
        if sid is None:
            sid = self.llkargs[3]
        if tid is None:
            tid = self.llkargs[4]
        D = self.llkargs[5]
        M = self.llkargs[6]
        starts_with_first = self.llkargs[7]
        ends_with_last = self.llkargs[8]
        ends_in_last = self.llkargs[9]
        if n_cores is None:
            n_cores = self.llkargs[10]
        build_mat_idx = self.llkargs[11]
        shared_pars = self.llkargs[12]
        shared_m = self.llkargs[13]
        n_series = len(sid)

        # Initialize potential None arguments

        if (
            tid is None
        ):  # Duration model matrices have rows for individual time-points not just trials
            tid = sid

        # Split up ys, Xs
        split_Ys, split_Xs = _split_matrices(
            ys,
            Xs,
            shared_pars,
            shared_m,
            obs_fams,
            d_fams,
            sid,
            tid,
            n_S,
            M,
            True,
            True,
            starts_with_first,
        )

        # Now sample state sequences for every series
        if n_cores == 1:
            states = []

            for series in range(n_series):
                with warnings.catch_warnings():  # Supress warnings
                    warnings.simplefilter("ignore")
                edss, statess = sample_posterior_series(
                    coef,
                    coef_split_idx,
                    shared_pars,
                    shared_m,
                    [
                        y_split[series] if y_split is not None else None
                        for y_split in split_Ys
                    ],
                    [
                        X_split[series] if X_split is not None else None
                        for X_split in split_Xs
                    ],
                    n_S,
                    obs_fams,
                    d_fams,
                    D,
                    M,
                    starts_with_first,
                    ends_with_last,
                    ends_in_last,
                    build_mat_idx,
                    n_samples,
                    seed,
                )

                states.append([edss, statess])

        else:  # Compute in parallel

            args = zip(
                repeat(coef),
                repeat(coef_split_idx),
                repeat(shared_pars),
                repeat(shared_m),
                [
                    [
                        y_split[series] if y_split is not None else None
                        for y_split in split_Ys
                    ]
                    for series in range(n_series)
                ],
                [
                    [
                        X_split[series] if X_split is not None else None
                        for X_split in split_Xs
                    ]
                    for series in range(n_series)
                ],
                repeat(n_S),
                repeat(obs_fams),
                repeat(d_fams),
                repeat(D),
                repeat(M),
                repeat(starts_with_first),
                repeat(ends_with_last),
                repeat(ends_in_last),
                repeat(build_mat_idx),
                repeat(n_samples),
                repeat(seed),
            )

            with warnings.catch_warnings():  # Supress warnings
                warnings.simplefilter("ignore")
                with mp.Pool(processes=n_cores) as pool:
                    eds, states = zip(*pool.starmap(sample_posterior_series, args))

                states = [[edss, statess] for edss, statess in zip(eds, states)]

        return states

    def get_resid(
        self,
        coef,
        coef_split_idx,
        ys,
        Xs,
        resid_type: str = "forward",
        transform_to_normal: bool = True,
        n_samples: int = 1000,
        seed: int = 1000,
        sid: None | np.ndarray = None,
        tid: None | np.ndarray = None,
    ) -> np.ndarray:
        """Get forward or ordinary pseudo/quantile residuals for a :class:`HSMMFamily` model.

        (Forward) "pseudo-residuals" (also known as "independent quantile residuals") have
        previously been defined by Zucchini et al. (2017) and Dunn & Smyth (1996).

        References:
         - Zucchini, W., MacDonald, I. L., & Langrock, R. (2017). Hidden Markov Models for \
            Time Series: An Introduction Using R, Second Edition (2nd ed.). Chapman and \
            Hall/CRC. https://doi.org/10.1201/b20790
         - Dunn, P. K., & Smyth, G. K. (1996). Randomized Quantile Residuals. Journal of \
            Computational and Graphical Statistics, 5(3), 236–244. \
            https://doi.org/10.2307/1390802

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each paramter of the llk.
        :type coef_split_idx: [int]
        :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as
            ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the
            actual observed data is passed along via the first formula (so it is stored in
            ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula,
            then ``ys`` contains ``None`` at their indices to save memory.
        :type ys: [np.ndarray or None]
        :param Xs: A list of sparse model matrices per likelihood parameter.
        :type Xs: [scp.sparse.csc_array]
        :param resid_type: The type of residual to compute, supported are "forward".
        :type resid_type: str, optional
        :param transform_to_normal: The pseudo-residuals are uniformly distributed and can thus be
            transformed to standard normal easily. This parameter controls whether that should
            happen.
        :type transform_to_normal: bool, optional
        :param n_samples: Number of state sequences to sample when ``resid_type=="posterior_dur".
        :type n_samples: int
        :param seed: An optional seed to initialize the sampler when ``resid_type=="posterior_dur".
            **Note**: A value < 0 means random initialization is used. Thus set to >= 0 for
            reproducable results. Defaults to 0
        :type seed: int, optional
        :param sid: Optional Array holding the first sample of each series in the data. Is used
            to split the observation vectors and model matrices into time-series specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type sid: np.ndarray | None, optional
        :param tid: Often, the model matrices for the models of the duration distribution parameters
            will have fewer rows than those of the observation models (because covariates only
            vary with trials). The optional array ``tid`` can then hold indices corresponding to the
            onset of a row-block of a new trial in the model matrices of the duration models.
            Is then used to split the duration model matrices into trial specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type tid: np.ndarray | None, optional
        :return: The residual vector of shape (-1,1)
        :rtype: np.ndarray
        """

        def series_resid(
            coef,
            coef_split_idx,
            shared_pars,
            shared_m,
            ys,
            Xs,
            n_S,
            obs_fams,
            d_fams,
            D,
            M,
            starts_with_first,
            ends_with_last,
            ends_in_last,
            resid_type,
            build_mat_idx,
            n_samples,
            seed,
        ):
            # Compute residual series for an individual series

            # Extract some extra info and define storage
            n_T = len(ys[0])  # Number of time-points in this series

            y_mat, mus, mus_d, _, cbs, bs, ds, T, pi = _compute_series_probs(
                coef,
                coef_split_idx,
                shared_pars,
                shared_m,
                ys,
                Xs,
                n_S,
                obs_fams,
                d_fams,
                D,
                M,
                None,
                None,
                None,
                False,
                True,
                True,
                starts_with_first,
                build_mat_idx,
            )
            cbs[(np.isnan(cbs) | np.isinf(cbs))] = 0
            bs[(np.isnan(bs) | np.isinf(bs))] = 0
            ds[(np.isnan(ds) | np.isinf(ds))] = 0

            if np.any(np.isnan(pi) | np.isinf(pi)):
                return np.array([np.nan for _ in range(n_T)])

            if np.any(np.isnan(T) | np.isinf(T)):
                return np.array([np.nan for _ in range(n_T)])

            hmp_code = 0
            event_template = np.zeros(1)
            scale = 1
            event_width = 0

            # Now compute residual vector #
            if resid_type == "forward":
                res = hsmm.forward_resid(
                    cbs,
                    bs,
                    ds,
                    T,
                    pi,
                    event_template,
                    scale,
                    n_T,
                    D,
                    n_S,
                    M,
                    event_width,
                    starts_with_first,
                    ends_with_last,
                    ends_in_last,
                    hmp_code,
                )

            elif resid_type == "predictive":

                res = hsmm.predictive_resid(
                    y_mat,
                    bs,
                    mus,
                    ds,
                    T,
                    pi,
                    event_template,
                    scale,
                    n_T,
                    D,
                    n_S,
                    M,
                    event_width,
                    starts_with_first,
                    ends_with_last,
                    ends_in_last,
                    hmp_code,
                    999,
                )

            elif resid_type == "viterbi_dur":

                # Decode viterbi path
                viterbi = self.decode_viterbi(
                    coef,
                    coef_split_idx,
                    ys,
                    Xs,
                    sid=np.array([0]),
                    tid=np.array([0]),
                    n_cores=1,
                )

                res_sep = compute_dur_res_series(
                    viterbi[0][1],
                    n_S,
                    mus_d,
                    d_fams,
                    starts_with_first,
                )

                # Compute residual of duration MAP under model for stage durations
                res = np.zeros(n_S)

                for j in range(n_S):
                    if res_sep[j] is None:
                        res[j] = np.nan
                    else:
                        res[j] = np.sum(res_sep[j])
                        res[j] /= len(res_sep[j])

                res = res.reshape(1, -1)

            elif resid_type == "posterior_dur":
                state_samples = self.sample_posterior_states(
                    coef,
                    coef_split_idx,
                    ys,
                    Xs,
                    n_samples=n_samples,
                    seed=seed,
                    sid=np.array([0]),
                    tid=np.array([0]),
                    n_cores=1,
                )

                res = np.zeros((1, n_S, n_samples))

                for s in range(n_samples):

                    # Extract durations given sampled series
                    res_sep = compute_dur_res_series(
                        state_samples[0][1][:, s],
                        n_S,
                        mus_d,
                        d_fams,
                        starts_with_first,
                    )

                    # Compute residual of durations under model for stage durations
                    for j in range(n_S):
                        if res_sep[j] is None:
                            res[0, j, s] = np.nan
                        else:
                            res[0, j, s] = np.sum(res_sep[j])
                            res[0, j, s] /= len(res_sep[j])

            return res

        if resid_type not in ["forward", "predictive", "viterbi_dur", "posterior_dur"]:
            raise ValueError(
                f"Requested residual type {resid_type} is not (currently) supported."
            )

        # Extract families, transition matrices, etc.
        n_S = self.llkargs[0]
        obs_fams = self.llkargs[1]
        d_fams = self.llkargs[2]
        if sid is None:
            sid = self.llkargs[3]
        if tid is None:
            tid = self.llkargs[4]
        D = self.llkargs[5]
        M = self.llkargs[6]
        starts_with_first = self.llkargs[7]
        ends_with_last = self.llkargs[8]
        ends_in_last = self.llkargs[9]
        n_cores = self.llkargs[10]
        build_mat_idx = self.llkargs[11]
        shared_pars = self.llkargs[12]
        shared_m = self.llkargs[13]
        n_series = len(sid)

        # Initialize potential None arguments

        if (
            tid is None
        ):  # Duration model matrices have rows for individual time-points not just trials
            tid = sid

        # Split up ys, Xs
        split_Ys, split_Xs = _split_matrices(
            ys,
            Xs,
            shared_pars,
            shared_m,
            obs_fams,
            d_fams,
            sid,
            tid,
            n_S,
            M,
            True,
            True,
            starts_with_first,
        )

        # Now compute llk for every individual series and then sum up
        res = []
        if n_cores == 1:

            for series in range(n_series):
                with warnings.catch_warnings():  # Supress warnings
                    warnings.simplefilter("ignore")
                    resids = series_resid(
                        coef,
                        coef_split_idx,
                        shared_pars,
                        shared_m,
                        [
                            y_split[series] if y_split is not None else None
                            for y_split in split_Ys
                        ],
                        [
                            X_split[series] if X_split is not None else None
                            for X_split in split_Xs
                        ],
                        n_S,
                        obs_fams,
                        d_fams,
                        D,
                        M,
                        starts_with_first,
                        ends_with_last,
                        ends_in_last,
                        resid_type,
                        build_mat_idx,
                        n_samples,
                        seed,
                    )

                    res.append(resids)

            res = np.concatenate(res, axis=0)

        else:  # Compute in parallel

            args = zip(
                repeat(coef),
                repeat(coef_split_idx),
                repeat(shared_pars),
                repeat(shared_m),
                [
                    [
                        y_split[series] if y_split is not None else None
                        for y_split in split_Ys
                    ]
                    for series in range(n_series)
                ],
                [
                    [
                        X_split[series] if X_split is not None else None
                        for X_split in split_Xs
                    ]
                    for series in range(n_series)
                ],
                repeat(n_S),
                repeat(obs_fams),
                repeat(d_fams),
                repeat(D),
                repeat(M),
                repeat(starts_with_first),
                repeat(ends_with_last),
                repeat(ends_in_last),
                repeat(resid_type),
                repeat(build_mat_idx),
                repeat(n_samples),
                repeat(seed),
            )

            with warnings.catch_warnings():  # Supress warnings
                warnings.simplefilter("ignore")
                with mp.Pool(processes=n_cores) as pool:

                    residss = pool.starmap(series_resid, args)

                res = np.concatenate(residss, axis=0)

        # Transform to normality?
        if transform_to_normal and resid_type == "forward":

            for m in range(res.shape[1]):
                res[:, m] = scp.stats.norm.ppf(res[:, m])

        return res

    def llk(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray],
        Xs: list[scp.sparse.csc_array],
        sid: None | np.ndarray = None,
        tid: None | np.ndarray = None,
    ) -> float:
        """Log-likelihood of model.

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each paramter of the llk.
        :type coef_split_idx: [int]
        :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as
            ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the
            actual observed data is passed along via the first formula (so it is stored in
            ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula,
            then ``ys`` contains ``None`` at their indices to save memory.
        :type ys: [np.ndarray or None]
        :param Xs: A list of sparse model matrices per likelihood parameter.
        :type Xs: [scp.sparse.csc_array]
        :param sid: Optional Array holding the first sample of each series in the data. Is used
            to split the observation vectors and model matrices into time-series specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type sid: np.ndarray | None, optional
        :param tid: Often, the model matrices for the models of the duration distribution parameters
            will have fewer rows than those of the observation models (because covariates only
            vary with trials). The optional array ``tid`` can then hold indices corresponding to the
            onset of a row-block of a new trial in the model matrices of the duration models.
            Is then used to split the duration model matrices into trial specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type tid: np.ndarray | None, optional
        :return: The log-likelihood evaluated at ``coef``.
        :rtype: float
        """

        def series_llk(
            coef,
            coef_split_idx,
            shared_pars,
            shared_m,
            ys,
            Xs,
            n_S,
            obs_fams,
            d_fams,
            D,
            M,
            starts_with_first,
            ends_with_last,
            ends_in_last,
            build_mat_idx,
        ):
            # Compute llk for an individual series - total llk is just sum

            # Extract some extra info and define storage
            n_T = len(ys[0])  # Number of time-points in this series

            # We can pre-compute the bs and ds - i.e., observation probabilities and
            # duration probabilities
            _, _, _, _, _, bs, ds, T, pi = _compute_series_probs(
                coef,
                coef_split_idx,
                shared_pars,
                shared_m,
                ys,
                Xs,
                n_S,
                obs_fams,
                d_fams,
                D,
                M,
                None,
                None,
                None,
                False,
                True,
                True,
                starts_with_first,
                build_mat_idx,
            )

            if np.any(np.isnan(ds) | np.isinf(ds)):
                return -np.inf

            if np.any(np.isnan(bs) | np.isinf(bs)):
                return -np.inf

            if np.any(np.isnan(pi) | np.isinf(pi)):
                return -np.inf

            if np.any(np.isnan(T) | np.isinf(T)):
                return -np.inf

            # Now compute llk #
            event_template = np.zeros(1)
            scale = 1
            hmp_code = 0
            event_width = 0

            llk = hsmm.llk(
                bs,
                ds,
                T,
                pi,
                event_template,
                scale,
                n_T,
                D,
                n_S,
                event_width,
                starts_with_first,
                ends_with_last,
                ends_in_last,
                hmp_code,
            )
            return llk

        # Extract families, transition matrices, etc.
        n_S = self.llkargs[0]
        obs_fams = self.llkargs[1]
        d_fams = self.llkargs[2]
        if sid is None:
            sid = self.llkargs[3]
        if tid is None:
            tid = self.llkargs[4]
        D = self.llkargs[5]
        M = self.llkargs[6]
        starts_with_first = self.llkargs[7]
        ends_with_last = self.llkargs[8]
        ends_in_last = self.llkargs[9]
        n_cores = self.llkargs[10]
        build_mat_idx = self.llkargs[11]
        shared_pars = self.llkargs[12]
        shared_m = self.llkargs[13]
        n_series = len(sid)

        # Initialize potential None arguments

        if (
            tid is None
        ):  # Duration model matrices have rows for individual time-points not just trials
            tid = sid

        # Split up ys, Xs
        split_Ys, split_Xs = _split_matrices(
            ys,
            Xs,
            shared_pars,
            shared_m,
            obs_fams,
            d_fams,
            sid,
            tid,
            n_S,
            M,
            True,
            True,
            starts_with_first,
        )

        # Now compute llk for every individual series and then sum up
        if n_cores == 1:
            llk = 0

            for series in range(n_series):
                with warnings.catch_warnings():  # Supress warnings
                    warnings.simplefilter("ignore")
                    llks = series_llk(
                        coef,
                        coef_split_idx,
                        shared_pars,
                        shared_m,
                        [
                            y_split[series] if y_split is not None else None
                            for y_split in split_Ys
                        ],
                        [
                            X_split[series] if X_split is not None else None
                            for X_split in split_Xs
                        ],
                        n_S,
                        obs_fams,
                        d_fams,
                        D,
                        M,
                        starts_with_first,
                        ends_with_last,
                        ends_in_last,
                        build_mat_idx,
                    )

                    llk += llks

        else:  # Compute in parallel

            args = zip(
                repeat(coef),
                repeat(coef_split_idx),
                repeat(shared_pars),
                repeat(shared_m),
                [
                    [
                        y_split[series] if y_split is not None else None
                        for y_split in split_Ys
                    ]
                    for series in range(n_series)
                ],
                [
                    [
                        X_split[series] if X_split is not None else None
                        for X_split in split_Xs
                    ]
                    for series in range(n_series)
                ],
                repeat(n_S),
                repeat(obs_fams),
                repeat(d_fams),
                repeat(D),
                repeat(M),
                repeat(starts_with_first),
                repeat(ends_with_last),
                repeat(ends_in_last),
                repeat(build_mat_idx),
            )

            with warnings.catch_warnings():  # Supress warnings
                warnings.simplefilter("ignore")
                with mp.Pool(processes=n_cores) as pool:

                    llks = pool.starmap(series_llk, args)

                llk = np.sum(llks)

        if np.isnan(llk):
            llk = -np.inf

        return llk

    def gradient(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray],
        Xs: list[scp.sparse.csc_array],
        sid: None | np.ndarray = None,
        tid: None | np.ndarray = None,
    ) -> np.ndarray:
        """Gradient of log-likelihood evaluated at ``coef``.

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each paramter of the llk.
        :type coef_split_idx: [int]
        :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as
            ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the
            actual observed data is passed along via the first formula (so it is stored in
            ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula,
            then ``ys`` contains ``None`` at their indices to save memory.
        :type ys: [np.ndarray or None]
        :param Xs: A list of sparse model matrices per likelihood parameter.
        :type Xs: [scp.sparse.csc_array]
        :param sid: Optional Array holding the first sample of each series in the data. Is used
            to split the observation vectors and model matrices into time-series specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type sid: np.ndarray | None, optional
        :param tid: Often, the model matrices for the models of the duration distribution parameters
            will have fewer rows than those of the observation models (because covariates only
            vary with trials). The optional array ``tid`` can then hold indices corresponding to the
            onset of a row-block of a new trial in the model matrices of the duration models.
            Is then used to split the duration model matrices into trial specific versions.
            Defaults to None, in which case the array passed to the constructor of ``self`` is used.
        :type tid: np.ndarray | None, optional
        :return: Gradient as array of shape (-1,1)
        :rtype: np.ndarray
        """

        def series_grad(
            coef,
            coef_split_idx,
            shared_pars,
            shared_m,
            ys,
            Xs,
            n_S,
            obs_fams,
            d_fams,
            D,
            M,
            starts_with_first,
            ends_with_last,
            ends_in_last,
            build_mat_idx,
        ):
            # Compute grad for an individual series - total grad is just sum

            # First fix ys and Xs for Nones
            yfix = [y if y is not None else ys[0] for y in ys]
            ys = yfix

            Xfix = [
                X if X is not None else Xs[build_mat_idx[xi]] for xi, X in enumerate(Xs)
            ]
            Xs = Xfix

            # Extract some extra info and define storage
            n_T = len(ys[0])  # Number of time-points in this series

            # We can pre-compute the bs and ds - i.e., observation probabilities and duratio
            # probabilities - and their gradients
            # But we need to keep bs separate per m and then it makes sense to pre-compute the
            # log-probs and later turn them into probs whenever needed.
            # Can still pre-compute product of observation probabilities and store that in bs
            bs = np.zeros(
                (n_T, n_S), order="F"
            )  # This will again be product of observation probabilities not log ones
            ds = np.zeros((D - 1, n_S * (1 if starts_with_first else 2)), order="F")
            d_val = np.arange(1, D).reshape(-1, 1)
            state_vals = np.arange(0, n_S - 1).reshape(-1, 1)
            pi_vals = np.arange(0, n_S).reshape(-1, 1)

            b_grad = None
            d_grad = None
            T_grad = None
            pi_grad = None

            j_idx_grad = None

            split_coef = np.split(coef, coef_split_idx)
            total_coef = 0

            # First observation probabilities #

            idx = 0  # Keep track of indices for ys and Xs

            # First check for shared part between emission models.

            # Lists below hold list per par. Each list contains at least one eta/X index/list to
            # be filled with duplicate coefficient indices
            shared_eta = []
            shared_X = []
            shared_coef_idx = []

            # List below holds indices of all duplicated coef so that we can later exclude them
            shared_coef_idx_flat = []
            for par in shared_pars:

                midx = 1
                if shared_m is False:
                    midx = M

                shared_eta.append([])
                shared_X.append([])
                shared_coef_idx.append([])

                # Compute shared part of eta per parameter and m and collect
                # X index + add list to be populated with duplicated/shared coef
                for m in range(midx):
                    shared_eta[par].append(Xs[idx] @ split_coef[idx])
                    shared_X[par].append(idx)
                    shared_coef_idx[par].append([])
                    idx += 1

            # print("OBS:")
            thetas = split_coef[-1].flatten()
            theta_idx = 0
            # Will need to keep track of position of theta gradients
            # since they will need to be moved to the end
            theta_grad_idx = []
            for j in range(n_S):

                iterM = M
                # If emission model is multivariate Gaussian, then it has M n_par!
                if isinstance(obs_fams[j][0], MultiGauss):
                    iterM = 1

                for m in range(iterM):

                    # Get family associated with state j and signal m
                    jmbFam = obs_fams[j][m]

                    # Multivariate gaussian emission
                    if isinstance(jmbFam, MultiGauss):

                        # Get thetas for this family
                        thetas_j = thetas[
                            theta_idx : theta_idx + jmbFam.extra_coef  # noqa: E203
                        ]

                        Y = np.concatenate(
                            ys[idx : idx + jmbFam.n_par], axis=1  # noqa: E203
                        )

                    # Multiple independent emission signals
                    else:
                        # Only first y associated with state j and signal m is not None
                        yJM = ys[idx]

                    etasbjm = []
                    musbjm = []
                    Xsj = []

                    for par in range(jmbFam.n_par):
                        Xsj.append(Xs[idx])
                        # print(j,split_coef[idx])
                        etasbjm.append(Xs[idx] @ split_coef[idx])

                        # Now handle any shared model component:
                        # 1) add shared eta to etasbjm
                        # 2) expand Xsj[-1] column-wise by shared model matrix
                        # 3) collect indices of duplicated coefficients

                        shared_par_idx = par
                        if isinstance(jmbFam, MultiGauss):
                            shared_par_idx = 0

                        if shared_par_idx in shared_pars:
                            midx = 0
                            if shared_m is False:
                                midx = m
                                if isinstance(jmbFam, MultiGauss):
                                    midx = par

                            # 1)
                            etasbjm[-1] += shared_eta[shared_par_idx][midx]

                            # 2)
                            Xsj[-1] = scp.sparse.hstack(
                                (Xsj[-1], Xs[shared_X[shared_par_idx][midx]])
                            )

                            # 3)
                            start_idx = total_coef + Xs[idx].shape[1]
                            stop_idx = (
                                total_coef
                                + Xs[idx].shape[1]
                                + Xs[shared_X[shared_par_idx][midx]].shape[1]
                            )

                            # Make sure total coef reflects duplicated coef
                            total_coef += Xs[shared_X[shared_par_idx][midx]].shape[1]

                            shared_idx = np.arange(
                                start_idx,
                                stop_idx,
                            )

                            shared_coef_idx[shared_par_idx][midx].append(shared_idx)

                            shared_coef_idx_flat.extend(shared_idx)

                        musbjm.append(jmbFam.links[par].fi(etasbjm[-1]))
                        total_coef += Xs[idx].shape[1]
                        idx += 1

                    if isinstance(jmbFam, MultiGauss):
                        musbjm = np.concatenate(musbjm, axis=1)

                        # Get transpose of Cholesky of precision
                        R, logdet = jmbFam.getR(thetas_j)

                        # yR is of shape n * m
                        yR = (Y - musbjm) @ R.T

                        # Log-probability of emission vector
                        bs[:, j] = -0.5 * (yR * yR).sum(axis=1) + logdet

                    else:
                        # Compute log-probs of signal m under state j
                        lpb = jmbFam.lp(yJM, *musbjm)

                        # Can also compute gradient of log-likelihood of observation distribution
                        # for state j and signal m with respect to coef:

                        # 1) Get partial first derivatives with respect to eta
                        if jmbFam.d_eta is False:
                            d1eta, _, _ = deriv_transform_mu_eta(yJM, musbjm, jmbFam)
                        else:
                            d1eta = [fd1(yJM, *musbjm) for fd1 in jmbFam.d1]

                    # 2) Get derivatives of log-likelihood of each observation with respect to
                    # coef

                    for par in range(jmbFam.n_par):
                        # n_T * Xsj[par].shape[1] matrix. Each row holds partial derivative of
                        # log-likelihood of corresponding observation with respect to
                        # coefficients in model par.
                        if isinstance(jmbFam, MultiGauss):
                            # RRy is of shape m * n
                            RRy = R.T @ yR.T

                            yRmui = RRy[par, :]
                            grad_par = Xsj[par] * yRmui[:, None]
                        else:
                            grad_par = d1eta[par] * Xsj[par]

                        # Cast to array
                        grad_par = (
                            grad_par.toarray(order="F")
                            if isinstance(grad_par, np.ndarray) is False
                            else grad_par
                        )

                        # Check for nans or infs
                        grad_par[np.isnan(grad_par) | np.isinf(grad_par)] = 0

                        # Concatenate over pars so that in the end gradJM is a
                        # n_T * np.sum([x.shape[1] for x in Xsj]) matrix with each row holding
                        # partial derivatives of log-likelihood of corresponding observation
                        # with respect to **all** coefficients involved in observation model of
                        # state j and signal m
                        if b_grad is None:
                            b_grad = grad_par
                            j_idx_grad = np.zeros(grad_par.shape[1]) + j
                        else:
                            b_grad = np.concatenate([b_grad, grad_par], axis=1)
                            j_idx_grad = np.concatenate(
                                [j_idx_grad, np.zeros(grad_par.shape[1]) + j]
                            )

                    # Have gradients with respect to coef but need gradients with respect
                    # to log(variance) and covariance parameters for multivariate gaussian
                    if isinstance(jmbFam, MultiGauss):
                        # partial of transpose of cholesky
                        Rdiag = R.diagonal()
                        Rp = np.zeros((jmbFam.n_par, jmbFam.n_par))
                        for mr in range(jmbFam.n_par):

                            for mc in range(mr, jmbFam.n_par):

                                # Diagonal elements are exp(theta) so partial differs
                                Rp[mr, mc] = Rdiag[mr] if mr == mc else 1

                                yRp = (Y - musbjm) @ Rp.T
                                dldtheta = -np.sum(yR * yRp, axis=1)

                                # Index function part from WPS (2016)
                                if mr == mc:
                                    dldtheta += 1

                                theta_grad_idx.append(len(j_idx_grad))

                                b_grad = np.concatenate(
                                    [b_grad, dldtheta.reshape(-1, 1)], axis=1
                                )
                                j_idx_grad = np.concatenate(
                                    [j_idx_grad, np.zeros(1) + j]
                                )

                                # Reset partial of R
                                Rp[mr, mc] = 0

                        # Update total coef index to reflect theta parameters
                        total_coef += jmbFam.extra_coef

                        # Update theta index
                        theta_idx += jmbFam.extra_coef

                    else:
                        # All signals are assumed independent given state, covariates, and
                        # coefficients
                        bs[:, j] += lpb.flatten()

                # Turn log-probs into probs again
                bs[:, j] = np.exp(bs[:, j])
                if np.any(np.isnan(bs[:, j]) | np.isinf(bs[:, j])):
                    bs[(np.isnan(bs[:, j]) | np.isinf(bs[:, j])), j] = 0

            # Now state duration probabilities ##
            # print("DUR:")
            for j in range(n_S * (1 if starts_with_first else 2)):

                jdFam = d_fams[j]

                etasdj = []
                musdj = []
                Xsj = []
                coefj = []

                for par in range(jdFam.n_par):
                    # print("d",j,split_coef[idx],par,jdFam.links[par])
                    Xsj.append(Xs[idx])
                    coefj.append(split_coef[idx])
                    etasdj.append(Xs[idx] @ split_coef[idx])
                    musdj.append(jdFam.links[par].fi(etasdj[-1]))
                    total_coef += Xs[idx].shape[1]
                    idx += 1

                # Compute log-probs of durations under state j
                lpd = jdFam.lp(d_val, *musdj)
                # lpd[np.isnan(lpd) | np.isinf(lpd)] = -np.inf
                # print(lpd[1])
                # print(jdFam.lp(d_val[1],musdj[0][0],musdj[1][0]))

                # Can also compute gradient of log-likelihood of duration distribution for
                # state j with respect to coef:

                # 1) Get partial first derivatives with respect to eta
                if jdFam.d_eta is False:
                    d1eta, _, _ = deriv_transform_mu_eta(d_val, musdj, jdFam)
                else:
                    d1eta = [fd1(d_val, *musdj) for fd1 in jdFam.d1]

                # 2) Get derivatives of log-likelihood of each observation with respect to coef

                for par in range(jdFam.n_par):
                    # n_D * Xsj[par].shape[1] matrix. Each row holds partial derivative of
                    # log-likelihood of corresponding duration with respect to coefficients in
                    # model par.
                    grad_par = d1eta[par] * Xsj[par]

                    # Cast to array
                    grad_par = (
                        grad_par.toarray(order="F")
                        if isinstance(grad_par, np.ndarray) is False
                        else grad_par
                    )

                    # So grad_par[0,:] holds partial derivatives of log(rds[0,j]) with respect
                    # to coefj[par]. Hence:  rds[0,j] * grad_par[0,:] gives partial derivatives
                    # of rds[0,j] with respect to coefj[par]:
                    rds = np.exp(lpd)  # p(rds[:,j])
                    grad_par_d = rds * grad_par

                    # To get partial derivatives of normalized probabilities
                    # ds[0,j]=rds[0,j]/np.sum(rds[:,j]) with respect to
                    # first coef for example, we need to compute:
                    # (grad_par_d[0,0] * np.sum(rds[:,j]) - rds[0,j] *
                    # (np.sum(grad_par_d[:,0]))) / np.power(np.sum(rds[:,j],2)
                    norm = np.sum(rds)
                    denom = np.power(norm, 2)
                    normed_sum = np.sum(grad_par_d, axis=0)

                    grad_par = grad_par_d * norm
                    grad_par -= rds * normed_sum
                    grad_par /= denom

                    grad_par[np.isnan(grad_par) | np.isinf(grad_par)] = 0

                    # Concatenate over pars so that in the end gradJ is a
                    # n_D * np.sum([x.shape[1] for x in Xsj]) matrix with each row holding
                    # partial derivatives of log-likelihood of corresponding duration
                    # with respect to **all** coefficients involved in duration model of state j
                    if d_grad is None:
                        d_grad = grad_par
                    else:
                        d_grad = np.concatenate([d_grad, grad_par], axis=1)
                    j_idx_grad = np.concatenate(
                        [j_idx_grad, np.zeros(grad_par.shape[1]) + j]
                    )

                # Collect duration probabilities associated with state j
                ds[:, j] = np.exp(lpd.flatten())
                ds[:, j] /= np.sum(ds[:, j])

                if np.any(np.isnan(ds[:, j]) | np.isinf(ds[:, j])):
                    # print("problem with ds",j)
                    ds[(np.isnan(ds[:, j]) | np.isinf(ds[:, j])), j] = 0

            # Now state transition probabilities #
            T = np.zeros((n_S, n_S), order="F")
            if n_S > 2:
                for j in range(n_S):
                    # No self-transitions!
                    jtFam = MULNOMLSS(n_S - 2)

                    etastj = []
                    mustj = []
                    Xsj = []
                    coefj = []

                    for par in range(jtFam.n_par):
                        Xsj.append(Xs[idx])
                        coefj.append(split_coef[idx])
                        etastj.append(Xs[idx] @ split_coef[idx])
                        mustj_par = jtFam.links[par].fi(etastj[-1])[0, 0]

                        # Expand for all possible transitions
                        mustj_par_exp = np.array(
                            [mustj_par for _ in range(n_S - 1)]
                        ).reshape(-1, 1)
                        mustj.append(mustj_par_exp)

                        total_coef += Xs[idx].shape[1]
                        idx += 1
                    # print(state_vals,mustj)
                    # 1) Get partial first derivatives with respect to eta
                    if jtFam.d_eta is False:
                        d1eta, _, _ = deriv_transform_mu_eta(state_vals, mustj, jtFam)
                    else:
                        d1eta = [fd1(state_vals, *mustj) for fd1 in jtFam.d1]

                    for par in range(jtFam.n_par):
                        # (n_S-1) * Xsj[par].shape[1] matrix. Each row holds partial derivative of
                        # log-likelihood of corresponding state transition with respect to
                        # coefficients in model par.
                        grad_par = d1eta[par] * Xsj[par]

                        # Cast to array
                        grad_par = (
                            grad_par.toarray(order="F")
                            if isinstance(grad_par, np.ndarray) is False
                            else grad_par
                        )

                        # Check for nans or infs
                        grad_par[np.isnan(grad_par) | np.isinf(grad_par)] = 0

                        # Concatenate over pars so that in the end gradJ is a
                        # (n_S-1) * np.sum([x.shape[1] for x in Xsj])
                        # matrix with each row holding partial derivatives of log-likelihood
                        # of corresponding state transition with respect to **all** coefficients
                        if T_grad is None:
                            T_grad = grad_par
                        else:
                            T_grad = np.concatenate([T_grad, grad_par], axis=1)
                        j_idx_grad = np.concatenate(
                            [j_idx_grad, np.zeros(grad_par.shape[1]) + j]
                        )

                    # Transform now into the n_j state transition probabilities from j -> i(see
                    # the MULNOMLSS methods doc-strings for details):
                    mu_lastT = np.sum(mustj, axis=0) + 1
                    mustj = [mu / mu_lastT for mu in mustj]
                    mustj.insert(0, 1 / mu_lastT)
                    # print(state_vals,mustj)

                    # Collect final state transition probabilities
                    Tj_idx = np.array([i for i in range(n_S) if i != j])
                    T[j, Tj_idx] = np.array([mu[0, 0] for mu in mustj])
                # print(T)
                T[np.isnan(T) | np.isinf(T)] = 0
            else:
                T[0, 1] = 1
                T[1, 0] = 1

                T_grad = np.ones((n_T, 1))

            # Now initial state probabilities #
            pi_fam = MULNOMLSS(n_S - 1)

            etaspi = []
            muspi = []
            Xsj = []
            coefj = []

            for par in range(pi_fam.n_par):
                Xsj.append(Xs[idx])
                coefj.append(split_coef[idx])

                etaspi.append(Xs[idx] @ split_coef[idx])
                muspi_par = pi_fam.links[par].fi(etaspi[-1])[0, 0]

                muspi_par_exp = np.array([muspi_par for _ in range(n_S)]).reshape(-1, 1)
                muspi.append(muspi_par_exp)

                total_coef += Xs[idx].shape[1]
                idx += 1

            # print(pi_vals,muspi)

            # 1) Get partial first derivatives with respect to eta
            if pi_fam.d_eta is False:
                d1eta, _, _ = deriv_transform_mu_eta(pi_vals, muspi, pi_fam)
            else:
                d1eta = [fd1(pi_vals, *muspi) for fd1 in pi_fam.d1]

            for par in range(pi_fam.n_par):
                # n_S * Xsj[par].shape[1] matrix. Each row holds partial derivative of
                # log-likelihood of corresponding initial state probabilities with respect to
                # coefficients in model par.
                grad_par = d1eta[par] * Xsj[par]

                # Cast to array
                grad_par = (
                    grad_par.toarray(order="F")
                    if isinstance(grad_par, np.ndarray) is False
                    else grad_par
                )

                # Check for nans or infs
                grad_par[np.isnan(grad_par) | np.isinf(grad_par)] = 0

                # Concatenate over pars so that in the end gradJ is a
                # n_S * np.sum([x.shape[1] for x in Xsj]) matrix with each row holding partial
                # derivatives of log-likelihood of corresponding initial state probabilities with
                # respect to **all** coefficients
                if pi_grad is None:
                    pi_grad = grad_par
                else:
                    pi_grad = np.concatenate([pi_grad, grad_par], axis=1)

            mu_lastPI = np.sum(muspi, axis=0) + 1
            muspi = [mu / mu_lastPI for mu in muspi]
            muspi.insert(0, 1 / mu_lastPI)
            # print(pi_vals,muspi)
            pi = np.array([mu[0, 0] for mu in muspi], order="F")
            pi[np.isnan(pi) | np.isinf(pi)] = 0
            # print(pi)

            # Now compute gradient #
            # print("total coef",total_coef)
            if total_coef != (
                b_grad.shape[1] + d_grad.shape[1] + T_grad.shape[1] + pi_grad.shape[1]
            ) - (1 if n_S <= 2 else 0):
                raise ValueError(
                    (
                        f"Expected number of coefficients {total_coef} does not match "
                        "gradient shape "
                        f"{b_grad.shape[1] + d_grad.shape[1] + T_grad.shape[1] + pi_grad.shape[1]}."
                    )
                )

            grad = hsmm.llkgrad(
                bs,
                ds,
                T,
                pi,
                b_grad,
                d_grad,
                T_grad,
                pi_grad,
                j_idx_grad,
                M,
                n_T,
                D,
                n_S,
                total_coef,
                starts_with_first,
                ends_with_last,
                ends_in_last,
                n_S > 2,
            )

            # Still need to compute unique gradient vector in case of shared coef.
            if len(shared_pars) > 0:

                # Extend with theta indices if there are any
                shared_coef_idx_flat.extend(theta_grad_idx)
                idx = 0  # Reset index
                grad2 = []  # Unique grad
                for par in shared_pars:

                    midx = 1
                    if shared_m is False:
                        midx = M

                    # Compute shared part of grad by summing over shared coef indices
                    for m in range(midx):
                        grad_shared = np.zeros(Xs[idx].shape[1])

                        for shared_idx in shared_coef_idx[par][m]:
                            grad_shared += grad[shared_idx]

                        grad2.extend(grad_shared)

                        idx += 1

                # Can now identify indices corresponding to truly unique coefficients
                ind_coef_idx_flat = np.array(
                    [
                        cidx
                        for cidx in range(total_coef)
                        if cidx not in shared_coef_idx_flat
                    ]
                )

                # These can simply be appended to gradient
                grad2.extend(grad[ind_coef_idx_flat])

                if len(theta_grad_idx) > 0:
                    # Add theta gradients to end of grad
                    grad2.extend(grad[theta_grad_idx])

                # Cast to np array
                grad = np.array(grad2)

            elif len(theta_grad_idx) > 0:

                grad2 = []  # Unique grad

                # Find indices for coef
                coef_idx_flat = np.array(
                    [cidx for cidx in range(total_coef) if cidx not in theta_grad_idx]
                )

                # These need to be put first in gradient
                grad2.extend(grad[coef_idx_flat])

                # Add theta gradients to end of grad
                grad2.extend(grad[theta_grad_idx])

                # Cast to np array
                grad = np.array(grad2)

            # print(llk)
            return grad.reshape(-1, 1)

        # Extract families, transition matrices, etc.
        n_S = self.llkargs[0]
        obs_fams = self.llkargs[1]
        d_fams = self.llkargs[2]
        if sid is None:
            sid = self.llkargs[3]
        if tid is None:
            tid = self.llkargs[4]
        D = self.llkargs[5]
        M = self.llkargs[6]
        starts_with_first = self.llkargs[7]
        ends_with_last = self.llkargs[8]
        ends_in_last = self.llkargs[9]
        n_cores = self.llkargs[10]
        build_mat_idx = self.llkargs[11]
        shared_pars = self.llkargs[12]
        shared_m = self.llkargs[13]
        n_series = len(sid)

        # Initialize potential None arguments

        if (
            tid is None
        ):  # Duration model matrices have rows for individual time-points not just trials
            tid = sid

        # Split up ys, Xs
        split_Ys, split_Xs = _split_matrices(
            ys,
            Xs,
            shared_pars,
            shared_m,
            obs_fams,
            d_fams,
            sid,
            tid,
            n_S,
            M,
            True,
            True,
            starts_with_first,
        )

        # Now compute grad for every individual series and then sum up
        if n_cores == 1:
            grad = None
            for series in range(n_series):
                with warnings.catch_warnings():  # Supress warnings
                    warnings.simplefilter("ignore")
                    grads = series_grad(
                        coef,
                        coef_split_idx,
                        shared_pars,
                        shared_m,
                        [
                            y_split[series] if y_split is not None else None
                            for y_split in split_Ys
                        ],
                        [
                            X_split[series] if X_split is not None else None
                            for X_split in split_Xs
                        ],
                        n_S,
                        obs_fams,
                        d_fams,
                        D,
                        M,
                        starts_with_first,
                        ends_with_last,
                        ends_in_last,
                        build_mat_idx,
                    )
                # print(grads)
                if grad is None:
                    grad = grads
                else:
                    grad += grads

        else:  # Compute in parallel

            args = zip(
                repeat(coef),
                repeat(coef_split_idx),
                repeat(shared_pars),
                repeat(shared_m),
                [
                    [
                        y_split[series] if y_split is not None else None
                        for y_split in split_Ys
                    ]
                    for series in range(n_series)
                ],
                [
                    [
                        X_split[series] if X_split is not None else None
                        for X_split in split_Xs
                    ]
                    for series in range(n_series)
                ],
                repeat(n_S),
                repeat(obs_fams),
                repeat(d_fams),
                repeat(D),
                repeat(M),
                repeat(starts_with_first),
                repeat(ends_with_last),
                repeat(ends_in_last),
                repeat(build_mat_idx),
            )

            with warnings.catch_warnings():  # Supress warnings
                warnings.simplefilter("ignore")
                with mp.Pool(processes=n_cores) as pool:

                    grads = pool.starmap(series_grad, args)

                grad = np.sum(grads, axis=0)

        return grad
