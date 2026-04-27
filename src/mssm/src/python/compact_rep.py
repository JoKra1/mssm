import copy
import numpy as np
import scipy as scp
from .matrix_solvers import cpp_cholP, compute_Linv, apply_eigen_perm

########## Compact Representations of quasi-Newton updates for L-qEFS update ########## # noqa: E266


def compute_omega(yks: np.ndarray, sks: np.ndarray, method: str = "NoWr") -> float:
    """Computes scaling factor for quasi-Newton Hessian approximation.


    Setting ``method = "NoWr"`` computes the scaling factor as described by
    Nocedal & Wright (2004) for the final set of update vectors. Setting ``method = "mean"`` or
    ``method = "min"`` computes the scaling factor as described by Brust (2020) for every pair
    of update vectors and then either returns the mean or minimum.

    References:
     - Nocedal, J., & Wright, S. J. (2006). Numerical Optimization.\
        https://doi.org/10.1007/978-0-387-40065-5
     - Brust, J. (2020) Limited Memory Structured Quasi-Newton Methods. A LANS Seminar Presentation

    :param yks: Array of update vectors yk
    :type yks: np.ndarray
    :param sks: Array of update vectors sk
    :type sks: np.ndarray
    :param method: Which method to use to compute scaling factor. Default is version by Nocedal &
        Wright (2004).
    :type method: str, optional
    :return: Scaling factor for initial approximation to hessian.
    :rtype: float
    """
    if method == "NoWr":
        omega = np.dot(yks[-1], yks[-1]) / np.dot(yks[-1], sks[-1])

    elif method in ["mean", "min"]:
        omega = 0
        omega_min = 0

        for ui in range(yks.shape[0]):
            omegai = np.dot(sks[ui], yks[ui]) / np.dot(sks[ui], sks[ui])
            omega += omegai

            if omegai < omega_min or omega_min <= 0:
                omega_min = omegai

        omega /= yks.shape[0]
        if method == "min":
            omega = omega_min

    if omega < 0:
        omega = 1

    return omega


def computeH(
    s: np.ndarray,
    y: np.ndarray,
    rho: np.ndarray,
    H0: scp.sparse.csc_array,
    explicit: bool = True,
) -> (
    np.ndarray
    | tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
):
    """Computes (explicitly or implicitly) the quasi-Newton approximation to the negative Hessian
    of the (penalized) likelihood :math:`\\mathbf{H}` (:math:`\\mathcal{H}`) from the L-BFGS-B
    optimizer info.

    Relies on equations 2.16 in Byrd, Nocdeal & Schnabel (1992).

    References:
     - Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994). Representations of quasi-Newton \
        matrices and their use in limited memory methods. Mathematical Programming, 63(1), \
        129–156. https://doi.org/10.1007/BF01582063

    :param s: np.ndarray of shape (m,p), where p is the number of coefficients, holding the first
        set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
    :type s: np.ndarray
    :param y: np.ndarray of shape (m,p), where p is the number of coefficients, holding the second
        set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
    :type y: np.ndarray
    :param rho: flattened numpy.array of shape (m,), holding element-wise ``1/y.T@s`` from Byrd,
        Nocdeal & Schnabel (1992).
    :type rho: np.ndarray
    :param H0: Initial estimate for the hessian of the negative (penalized) likelihood. Here some
        multiple of the identity (multiplied by ``omega``).
    :type H0: scipy.sparse.csc_array
    :param explicit: Whether or not to return the approximate matrix explicitly or implicitly in
        form of four update matrices.
    :type explicit: bool
    :return: H, either as np.ndarray (``explicit=='True'``) or represented implicitly via four
        update vectors (also np.ndarrays)
    :rtype: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    # Number of updates?
    m = len(y)

    # First form S,Y, and D
    S = s.T
    Y = y.T

    STS = S.T @ H0 @ S
    DK = np.identity(m)
    DK[0, 0] *= np.dot(s[0], y[0])

    # Now use eq. 2.5 to compute R - only have to do this once
    R0 = np.dot(s[0], y[0]).reshape(1, 1)
    R = R0
    for k in range(1, m):

        DK[k, k] *= np.dot(s[k], y[k])

        R = np.concatenate(
            (
                np.concatenate((R0, S[:, :k].T @ Y[:, [k]]), axis=1),
                np.concatenate(
                    (np.zeros((1, R0.shape[1])), np.array([1 / rho[k]]).reshape(1, 1)),
                    axis=1,
                ),
            ),
            axis=0,
        )

        R0 = R

    # Eq 2.22
    L = S.T @ Y - R

    # Now compute term 2 in 3.13 of Byrd, Nocdeal & Schnabel (1992)
    t2 = np.zeros((2 * m, 2 * m))
    t2[:m, :m] = STS
    t2[:m, m:] = L
    t2[m:, :m] = L.T
    t2[m:, m:] = -1 * DK

    # We actually need the inverse to compute H

    # Eq 2.26 of Byrd, Nocdeal & Schnabel (1992)
    Dinv = copy.deepcopy(DK)
    Dpow = copy.deepcopy(DK)
    Dnpow = copy.deepcopy(DK)
    for k in range(m):
        Dinv[k, k] = 1 / Dinv[k, k]
        Dpow[k, k] = np.power(Dpow[k, k], 0.5)
        Dnpow[k, k] = np.power(Dnpow[k, k], -0.5)

    JJT = STS + L @ Dinv @ L.T
    J = scp.linalg.cholesky(JJT, lower=True)

    t2L = np.zeros((2 * m, 2 * m))
    t2L[:m, :m] = Dpow
    t2L[m:, :m] = (-1 * L) @ Dnpow
    t2L[m:, m:] = J

    t2U = np.zeros((2 * m, 2 * m))
    t2U[:m, :m] = -1 * Dpow
    t2U[:m:, m:] = Dnpow @ L.T
    t2U[m:, m:] = J.T

    t2_flip = t2L @ t2U

    invt2L = scp.linalg.inv(t2L)
    invT2U = scp.linalg.inv(t2U)
    invt2 = invt2L.T @ invT2U.T

    t2_sort = np.zeros((2 * m, 2 * m))
    # top left <- bottom right
    t2_sort[:m, :m] = t2_flip[m:, m:]
    # top right <- bottom left
    t2_sort[:m, m:] = t2_flip[m:, :m]
    # bottom left <- top right
    t2_sort[m:, :m] = t2_flip[:m, m:]
    # bottom right <- top left
    t2_sort[m:, m:] = t2_flip[:m, :m]

    invt2_sort = np.zeros((2 * m, 2 * m))
    # top left <- bottom right
    invt2_sort[:m, :m] = invt2[m:, m:]
    # top right <- bottom left
    invt2_sort[:m, m:] = invt2[m:, :m]
    # bottom left <- top right
    invt2_sort[m:, :m] = invt2[:m, m:]
    # bottom right <- top left
    invt2_sort[m:, m:] = invt2[:m, :m]

    # And terms 1 and 2
    t1 = np.concatenate((H0 @ S, Y), axis=1)
    t3 = np.concatenate((S.T @ H0, Y.T), axis=0)

    # Return matrix in compact representation
    if explicit is False:
        return t1, (-1 * t2_sort), (-1 * invt2_sort), t3

    H = H0 + t1 @ (-1 * invt2_sort) @ t3

    return H


def computeV(
    s: np.ndarray,
    y: np.ndarray,
    rho: np.ndarray,
    V0: scp.sparse.csc_array,
    explicit: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes (explicitly or implicitly) the quasi-Newton approximation to the inverse of the
    negative Hessian of the (penalized) likelihood :math:`\\mathcal{I}` (:math:`\\mathbf{V}`)
    from the L-BFGS-B optimizer info.

    Relies on equations 2.16 and 3.13 in Byrd, Nocdeal & Schnabel (1992).

    References:
     - Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994). Representations of quasi-Newton \
        matrices and their use in limited memory methods. Mathematical Programming, 63(1), \
        129–156. https://doi.org/10.1007/BF01582063

    :param s: np.ndarray of shape (m,p), where p is the number of coefficients, holding the
        first set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
    :type s: np.ndarray
    :param y: np.ndarray of shape (m,p), where p is the number of coefficients, holding the second
        set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
    :type y: np.ndarray
    :param rho: flattened numpy.array of shape (m,), holding element-wise ```1/y.T@s`` from Byrd,
        Nocdeal & Schnabel (1992).
    :type rho: np.ndarray
    :param V0: Initial estimate for the inverse of the hessian of the negative (penalized)
        likelihood. Here some multiple of the identity (multiplied by ``omega``).
    :type V0: scipy.sparse.csc_array
    :param explicit: Whether or not to return the approximate matrix explicitly or implicitly in
        form of three update matrices.
    :type explicit: bool
    :return: V, either as np.ndarray (``explicit=='True'``) or represented implicitly via three
        update vectors (also np.ndarrays)
    :rtype: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    m = len(y)
    # First form S,Y, and D
    S = s.T
    Y = y.T

    DYTY = Y.T @ V0 @ Y

    DYTY[0, 0] += np.dot(s[0], y[0])

    # Now use eq. 2.5 to compute R^{-1} - only have to do this once
    Rinv0 = 1 / np.dot(s[0], y[0]).reshape(1, 1)
    Rinv = Rinv0
    for k in range(1, m):

        DYTY[k, k] += np.dot(s[k], y[k])

        Rinv = np.concatenate(
            (
                np.concatenate(
                    (Rinv0, (-rho[k]) * Rinv0 @ S[:, :k].T @ Y[:, [k]]), axis=1
                ),
                np.concatenate(
                    (np.zeros((1, Rinv0.shape[1])), np.array([rho[k]]).reshape(1, 1)),
                    axis=1,
                ),
            ),
            axis=0,
        )

        Rinv0 = Rinv

    # Now compute term 2 in 3.13 used for all S_j
    t2 = np.zeros((2 * m, 2 * m))
    t2[:m, :m] = Rinv.T @ DYTY @ Rinv
    t2[:m, m:] = -Rinv.T
    t2[m:, :m] = -Rinv

    # And terms 1 and 2
    t1 = np.concatenate((S, V0 @ Y), axis=1)
    t3 = np.concatenate((S.T, Y.T @ V0), axis=0)

    if explicit:
        V = V0 + t1 @ t2 @ t3
        return V
    else:
        return t1, t2, t3


def computeVSR1(
    s: np.ndarray,
    y: np.ndarray,
    rho: np.ndarray,
    V0: scp.sparse.csc_array,
    omega: float = 1,
    make_psd: bool = False,
    explicit: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes (explicitly or implicitly) the symmetric rank one (SR1) approximation of the
    inverse of the negative Hessian of the (penalized) likelihood :math:`\\mathcal{I}`
    (:math:`\\mathbf{V}`).

    Relies on equations 2.16 and 3.13 in Byrd, Nocdeal & Schnabel (1992). Can ensure positive
    (semi) definiteness of the approximation via an eigen decomposition as shown by
    Burdakov et al. (2017). This is enforced via the ``make_psd`` argument.

    References:
     - Burdakov, O., Gong, L., Zikrin, S., & Yuan, Y. (2017). On efficiently combining \
        limited-memory and trust-region techniques. Mathematical Programming Computation, \
        9(1), 101–134. https://doi.org/10.1007/s12532-016-0109-7
     - Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994). Representations of quasi-Newton \
        matrices and their use in limited memory methods. Mathematical Programming, 63(1), \
        129–156. https://doi.org/10.1007/BF01582063

    :param s: np.ndarray of shape (m,p), where p is the number of coefficients, holding the first
        set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
    :type s: np.ndarray
    :param y: np.ndarray of shape (m,p), where p is the number of coefficients, holding the second
        set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
    :type y: np.ndarray
    :param rho: flattened numpy.array of shape (m,), holding element-wise ```1/y.T@s`` from Byrd,
        Nocdeal & Schnabel (1992).
    :type rho: np.ndarray
    :param V0: Initial estimate for the inverse of the hessian of the negative (penalized)
        likelihood. Here some multiple of the identity (multiplied by ``omega``).
    :type V0: scipy.sparse.csc_array
    :param omega: Multiple of the identity matrix used as initial estimate.
    :type omega: float, optional
    :param make_psd: Whether to enforce PSD as mentioned in the description. By default set to
        False.
    :type make_psd: bool, optional
    :param explicit: Whether or not to return the approximate matrix explicitly or implicitly in
        form of three update matrices.
    :type explicit: bool
    :return: V, either as np.ndarray (``explicit=='True'``) or represented implicitly via three
        update vectors (also np.ndarrays)
    :rtype: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    m = len(y)
    # First form S,Y, and D
    S = s.T
    Y = y.T

    YTY = Y.T @ V0 @ Y
    DK = np.identity(m)
    DK[0, 0] *= np.dot(s[0], y[0])

    # Now use eq. 2.5 to compute R - only have to do this once
    R0 = np.dot(s[0], y[0]).reshape(1, 1)
    R = R0
    for k in range(1, m):

        DK[k, k] *= np.dot(s[k], y[k])

        R = np.concatenate(
            (
                np.concatenate((R0, S[:, :k].T @ Y[:, [k]]), axis=1),
                np.concatenate(
                    (np.zeros((1, R0.shape[1])), np.array([1 / rho[k]]).reshape(1, 1)),
                    axis=1,
                ),
            ),
            axis=0,
        )

        R0 = R

    # Eq 2.22
    # L = S.T @ Y - R

    # Now compute term 2 in eq. 5.2
    t2 = scp.linalg.inv(R + R.T - DK - YTY)

    # And terms 1 and 2
    t1 = S - V0 @ Y
    t3 = t1.T

    if make_psd:
        # Compute implicit eigen decomposition as shown by Burdakov et al. (2017)
        Q, R = scp.linalg.qr(t1, mode="economic")
        Rit2R = R @ (t2) @ R.T

        # ev holds non-zero eigenvalues of U@D@U.T (e.g., Burdakov et al. (2017))
        ev, P = scp.linalg.eigh(Rit2R, driver="ev")

        # Now find closest PSD.
        fix_idx = (ev + omega) <= 0

        if np.sum(fix_idx) > 0:
            # print("fix VSR1",np.sum(fix_idx),omega,1/omega)
            ev[fix_idx] = -1 * omega  # + np.power(np.finfo(float).eps,0.9)

            # while np.any(np.abs(ev) < 1e-7):
            #   ev[np.abs(ev) < 1e-7] += np.power(np.finfo(float).eps,0.7)

            # print(f"implicit ev post shift. abs min: {np.min(np.abs(ev))}, min: {np.min(ev)},
            #   min + 1: {np.min(ev+omega)}, max + 1: {np.max(ev+omega)}",ev[:10]+omega)

            # Now Q @ P @ np.diag(ev) @ P.T @ Q.T = shifted PSD version of U@D@U.T
            # so we can set:
            # shifted_invt2=np.diag(ev)
            # shifted_t2 = np.diag(1/ev)
            # t1 = Q @ P
            # t3 = t1.T = P.T @ Q.T
            t1 = Q @ P
            t3 = P.T @ Q.T
            t2 = np.diag(ev)

    if explicit:
        V = V0 + t1 @ t2 @ t3
        return V
    else:
        return t1, t2, t3


def computeHSR1(
    s: np.ndarray,
    y: np.ndarray,
    rho: np.ndarray,
    H0: scp.sparse.csc_array,
    omega: float = 1,
    make_psd: bool = False,
    make_pd: bool = False,
    explicit: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes, (explicitly or implicitly) the symmetric rank one (SR1) approximation of the
    negative Hessian of the (penalized) likelihood :math:`\\mathbf{H}` (:math:`\\mathcal{H}`).

    Relies on equations 2.16 and 3.13 in Byrd, Nocdeal & Schnabel (1992). Can ensure positive (semi)
    definiteness of the approximation via an eigen decomposition as shown by Burdakov et al. (2017).
    This is enforced via the ``make_psd`` and ``make_pd`` arguments.

    References:
     - Burdakov, O., Gong, L., Zikrin, S., & Yuan, Y. (2017). On efficiently combining \
        limited-memory and trust-region techniques. Mathematical Programming Computation, \
        9(1), 101–134. https://doi.org/10.1007/s12532-016-0109-7
     - Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994). Representations of quasi-Newton \
        matrices and their use in limited memory methods. Mathematical Programming, 63(1), \
        129–156. https://doi.org/10.1007/BF01582063

    :param s: np.ndarray of shape (m,p), where p is the number of coefficients, holding the
        first set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
    :type s: np.ndarray
    :param y: np.ndarray of shape (m,p), where p is the number of coefficients, holding the
        second set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
    :type y: np.ndarray
    :param rho: flattened numpy.array of shape (m,), holding element-wise ``1/y.T@s`` from
        Byrd, Nocdeal & Schnabel (1992).
    :type rho: np.ndarray
    :param H0: Initial estimate for the hessian of the negative (penalized) likelihood. Here
        some multiple of the identity (multiplied by ``omega``).
    :type H0: scipy.sparse.csc_array
    :param omega: Multiple of the identity matrix used as initial estimate.
    :type omega: float, optional
    :param make_psd: Whether to enforce PSD as mentioned in the description. By default set to
        False.
    :type make_psd: bool, optional
    :param make_pd: Whether to enforce numeric positive definiteness, not just PSD. Ignored if
        ``make_psd=False``. By default set to False.
    :type make_pd: bool, optional
    :param explicit: Whether or not to return the approximate matrix explicitly or implicitly in
        form of three update matrices.
    :type explicit: bool
    :return: H, either as np.ndarray (``explicit=='True'``) or represented implicitly via three
        update vectors (also np.ndarrays)
    :rtype: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    m = len(y)
    # First form S,Y, and D
    S = s.T
    Y = y.T

    STS = S.T @ H0 @ S
    DK = np.identity(m)
    DK[0, 0] *= np.dot(s[0], y[0])

    # Now use eq. 2.5 to compute R - only have to do this once
    R0 = np.dot(s[0], y[0]).reshape(1, 1)
    R = R0
    for k in range(1, m):

        DK[k, k] *= np.dot(s[k], y[k])

        R = np.concatenate(
            (
                np.concatenate((R0, S[:, :k].T @ Y[:, [k]]), axis=1),
                np.concatenate(
                    (np.zeros((1, R0.shape[1])), np.array([1 / rho[k]]).reshape(1, 1)),
                    axis=1,
                ),
            ),
            axis=0,
        )

        R0 = R

    # Eq 2.22
    L = S.T @ Y - R

    # Now compute term 2 in eq. 5.2
    t2 = scp.linalg.inv(DK + L + L.T - STS)

    # And terms 1 and 2
    t1 = Y - H0 @ S
    t3 = t1.T

    if make_psd:
        # Compute implicit eigen decomposition as shown by Burdakov et al. (2017)
        Q, R = scp.linalg.qr(t1, mode="economic")
        Rit2R = R @ (t2) @ R.T

        # ev holds non-zero eigenvalues of U@D@U.T (e.g., Burdakov et al. (2017))
        ev, P = scp.linalg.eigh(Rit2R, driver="ev")

        # Now find closest PSD.
        fix_idx = (ev + omega) <= 0

        if np.sum(fix_idx) > 0:
            # print("fix VSR1",np.sum(fix_idx),omega,1/omega)
            ev[fix_idx] = -1 * omega  # + np.power(np.finfo(float).eps,0.9)

            # Useful to guarantee that penalized hessian is pd at convergence
            if make_pd:
                eps = np.power(np.finfo(float).eps, 0.5)
                ev[fix_idx] += eps * np.max(ev + omega)

            # while np.any(np.abs(ev) < 1e-7):
            #   ev[np.abs(ev) < 1e-7] += np.power(np.finfo(float).eps,0.7)

            # print(f"implicit ev post shift. abs min: {np.min(np.abs(ev))}, min: {np.min(ev)},
            #   min + 1: {np.min(ev+omega)}, max + 1: {np.max(ev+omega)}",ev[:10]+omega)

            # Now Q @ P @ np.diag(ev) @ P.T @ Q.T = shifted PSD version of U@D@U.T
            # so we can set:
            # shifted_invt2=np.diag(ev)
            # shifted_t2 = np.diag(1/ev)
            # t1 = Q @ P
            # t3 = t1.T = P.T @ Q.T
            t1 = Q @ P
            t3 = P.T @ Q.T
            t2 = np.diag(ev)

    if explicit:
        H = H0 + t1 @ t2 @ t3
        return H
    else:
        return t1, t2, t3


def computeSH(
    yks: np.ndarray,
    sks: np.ndarray,
    nHfd: scp.sparse.csc_array,
    IonHBB: np.ndarray,
    fcols: np.ndarray,
    acols: np.ndarray,
    sample_hessian: bool = True,
    fully_dampened_HBb: bool = False,
    dampen_HBB: float = 1,
    make_pd: bool = False,
    explicit: bool = True,
    form: str = "SR1",
) -> (
    np.ndarray
    | tuple[
        scp.sparse.csc_array,
        scp.sparse.csc_array,
        np.ndarray,
        scp.sparse.csr_array,
        scp.sparse.csc_array | None,
        scp.sparse.csr_array | None,
        scp.sparse.csc_array,
        np.ndarray,
        scp.sparse.csr_array,
    ]
):
    """Computes structured approximation to the negative Hessian of the log-likelihood. Combines
    available/partial information about the Hessian with a quasi-Newton approximation to the
    remaining/unavailable (mixed) second partial derivatives.

    First, note that we can always re-order the negative Hessian :math:`\\mathbf{H}` so that

    .. math::
        :nowrap:

        $$
        \\mathbf{H} =
        \\begin{bmatrix}
            \\mathbf{H}_{\\boldsymbol{\\beta}\\boldsymbol{\\beta}} &
            \\mathbf{H}_{\\boldsymbol{\\beta}\\mathbf{b}} \\\\
            \\mathbf{H}_{\\mathbf{b}\\boldsymbol{\\beta}} &
            \\mathbf{H}_{\\mathbf{b}\\mathbf{b}} \\\\
        \\end{bmatrix}
        $$

    Second, consider the Schur complement (:math:`\\mathbf{D}`) of :math:`\\mathbf{H}` with respect
    to :math:`\\mathbf{H}_{\\mathbf{b}\\mathbf{b}}`:

    .. math::
        \\mathbf{D} = \\mathbf{H}_{\\mathbf{b}\\mathbf{b}} -
        \\mathbf{H}_{\\boldsymbol{b}\\boldsymbol{\\beta}}
        (\\mathbf{H}_{\\boldsymbol{\\beta}\\boldsymbol{\\beta}})^{-1}
        \\mathbf{H}_{\\boldsymbol{\\beta}\\mathbf{b}}

    Third, note that (for quadratic negative log-likelihood or Taylor approximation to -llk) we can
    define a structured secant equation for :math:`\\mathbf{H}_{\\mathbf{b}\\mathbf{b}}` using the
    Schur complement:

    .. math::
        \\mathbf{D}\\mathbf{s}_{\\mathbf{b}} =
        \\mathbf{y}_{\\mathbf{b}} -
        \\left(\\left[\\mathbf{0}~ \\mathbf{H}_{\\boldsymbol{b}\\boldsymbol{\\beta}}
        (\\mathbf{H}_{\\boldsymbol{\\beta}\\boldsymbol{\\beta}})^{-1}
        \\mathbf{H}_{\\boldsymbol{\\beta}\\boldsymbol{b}}\\right] +
        \\left[\\mathbf{H}_{\\boldsymbol{b}\\boldsymbol{\\beta}}~\\mathbf{0}\\right]\\right)
        \\mathbf{s}

    Here :math:`\\mathbf{s}` is a step applied to the coefficient vector (i.e.,
    :math:`\\left[\\boldsymbol{\\beta}, \\boldsymbol{b} \\right])`, :math:`\\mathbf{y}` is the
    difference between the gradients of the negative llk after and before taking this step, and
    versions with :math:`\\mathbf{b}` subscripts index elements in :math:`\\boldsymbol{b}`.

    Since the Schur complement meets the structured secant equation, we can estimate it via
    any quasi-Newton routine so that:

    .. math::
        \\hat{\\mathbf{D}} = \\mathbf{I}\\omega +
        \\mathbf{Q}\\boldsymbol{\\Delta}\\mathbf{Q}^\\top

    In which case, considering the structured secant equation above, we have an approximation to
    the lower block of the negative Hessian defined as:

    .. math::
        \\hat{\\mathbf{H}}_{\\mathbf{b}\\mathbf{b}} =
        \\hat{\\mathbf{D}} + \\mathbf{H}_{\\mathbf{b}\\boldsymbol{\\beta}}
        (\\mathbf{H}_{\\boldsymbol{\\beta}\\boldsymbol{\\beta}})^{-1}
        \\mathbf{H}_{\\boldsymbol{\\beta}\\mathbf{b}}

    and the entire negative Hessian:

    .. math::
        :nowrap:

        $$
        \\hat{\\mathbf{H}} =
        \\begin{bmatrix}
            \\mathbf{H}_{\\boldsymbol{\\beta}\\boldsymbol{\\beta}} &
            \\mathbf{H}_{\\boldsymbol{\\beta}\\mathbf{b}} \\\\
            \\mathbf{H}_{\\mathbf{b}\\boldsymbol{\\beta}} &
            \\hat{\\mathbf{H}}_{\\mathbf{b}\\mathbf{b}} \\\\
        \\end{bmatrix}
        $$

    Note, that if we use a quasi Newton update that ensures positive semi-definiteness of the Schur
    complement :math:`\\hat{\\mathbf{D}}`, then :math:`\\hat{\\mathbf{H}}_{\\mathbf{b}\\mathbf{b}}`
    and :math:`\\hat{\\mathbf{H}}` are also guaranteed to be PSD. Finally, we can represent
    :math:`\\hat{\\mathbf{H}}` implicitly via an initial matrix and three updates (consisting of
    8 update matrices):

    .. math::
        :nowrap:

        $$
        \\hat{\\mathbf{H}} =
        \\begin{bmatrix}
            \\mathbf{H}_{\\boldsymbol{\\beta}\\boldsymbol{\\beta}} & \\mathbf{0} \\\\
            \\mathbf{0} & \\mathbf{I}\\omega \\\\
        \\end{bmatrix} +
        \\begin{bmatrix}
            \\mathbf{0} & \\mathbf{0} \\\\
            \\mathbf{0} & \\mathbf{H}_{\\mathbf{b}\\boldsymbol{\\beta}}
            (\\mathbf{H}_{\\boldsymbol{\\beta}\\boldsymbol{\\beta}})^{-1}
            \\mathbf{H}_{\\boldsymbol{\\beta}\\mathbf{b}} \\\\
        \\end{bmatrix} +
        \\begin{bmatrix}
            \\mathbf{0} & \\mathbf{H}_{\\boldsymbol{\\beta}\\mathbf{b}} \\\\
            \\mathbf{H}_{\\mathbf{b}\\boldsymbol{\\beta}} & \\mathbf{0} \\\\
        \\end{bmatrix} +
        \\begin{bmatrix}
            \\mathbf{0} & \\mathbf{0} \\\\
            \\mathbf{0} & \\mathbf{Q}\\boldsymbol{\\Delta}\\mathbf{Q}^\\top \\\\
        \\end{bmatrix}
        $$

    This is returned by this function, so that::

      nH = nH1 + (nH2t1 @ nH2t2 @ nH2t3) + (nH3t1 @ nH3t3) + (nH4t1 @ nH4t2 @ nH4t3)

    Finally, note that :math:`\\hat{\\mathbf{H}} + \\mathbf{S}_{\\boldsymbol{\\lambda}}` can be
    inverted (:math:`\\mathbf{S}_{\\boldsymbol{\\lambda}}` is the total penalty matrix)
    via repeated application of the modified Woodbury identity of Henderson & Searle (1981).

    :param yks: np.ndarray of shape (m,p), where p is the number of coefficients, holding the
        second set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992), modified to account
        for structured update.
    :type yks: np.ndarray
    :param sks: np.ndarray of shape (m,p), where p is the number of coefficients, holding the
        first set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992), modified to account
        for structured update.
    :type sks: np.ndarray
    :param nHfd: Finite difference approximation matrix which is symmetric sparse matrix with
        ``fcols`` rows and columns set to finite difference approximation of columns of negative
        Hessian of llk
    :type nHfd: scp.sparse.csc_array
    :param IonHBB: The inverse of the top left block of ``nHfd`` after ordering so that column
        (and row) order is ``[fcols, acols]``
    :type IonHBB: np.ndarray
    :param fcols: Array holding indices of columns of negative Hessian approximated via finite
        differencing.
    :type fcols: np.ndarray
    :param acols: Array holding indices of columns of negative Hessian not approximated via FD (
        to be approximated via quasi Newton)
    :type acols: np.ndarray
    :param sample_hessian: Whether the update vectors ``yks`` and ``sks`` have been sampled or
        obtained from the sequential quasi Newton routine used to update the coefficients, defaults
        to True
    :type sample_hessian: bool, optional
    :param fully_dampened_HBb: Whether the top right and lower left rectangular blocks (mixed
        partial derivatives of ``coef[fcols]`` with respect to ``coef[acols]``) have been zeroed,
        defaults to False
    :type fully_dampened_HBb: bool, optional
    :param dampen_HBB: Scaling factor by which to scale the initialidentity matrix used to
        approximate the quasi Newton block, defaults to 1
    :type dampen_HBB: float, optional
    :param make_pd: Whether to enforce numeric positive definiteness, not just PSD, defaults to
        False.
    :type make_pd: bool, optional
    :param explicit: Whether or not to return the approximate matrix explicitly or implicitly in
        form of 8 update matrices, defaults to True
    :type explicit: bool
    :param form: Should the quasi Newton Hessian approximation use a symmetric rank 1 update
        (``qEFSH='SR1'``) that is forced to result in positive semi-definiteness of the
        approximation or the standard bfgs update (``qEFSH='BFGS'``). Defaults to 'SR1'.
    :type form: str
    :return: H, either as np.ndarray (``explicit=='True'``) or represented implicitly via an initial
        matrix and 8 update matrices as defined in the description (i.e.,
        ``nH1, nH2t1, nH2t2, nH2t3, nH3t1, nH3t3, nH4t1, nH4t2, nH4t3``).
    :rtype: np.ndarray | tuple[scp.sparse.csc_array, scp.sparse.csc_array, np.ndarray,
        scp.sparse.csr_array, scp.sparse.csc_array | None, scp.sparse.csr_array | None,
        scp.sparse.csc_array, np.ndarray, scp.sparse.csr_array]
    """

    # Combined indices in order so that quasi Newton block is bottom right
    tcols = [*fcols, *acols]

    nF, nA, nT = len(fcols), len(acols), len(tcols)

    # To compute ordered version of hessian we need first permutation matrix P1:
    # len(tcols) * len(tcols) permutation matrix transforming entire nH into order (onH) so that
    # fd approximated block is top left and quasi Newton block is bottom right via
    # onH = P1.T @ nH @ P1
    P1 = scp.sparse.csc_array(
        (
            np.tile(1, nT),
            (tcols, np.arange(nT)),
        ),
        shape=(nT, nT),
    )

    # Now can compute ordered version of finite difference approximated part of nH and extract the
    # relevant blocks
    onHfd = P1.T @ nHfd @ P1

    # Can now extract the block on the diagonal and the off-diagonals holding the mixed derivs
    # onHBB should be PD
    # onHBb might have been dampened.
    onHBB = onHfd[np.ix_(np.arange(nF), np.arange(nF))]
    onHBb = onHfd[np.ix_(np.arange(nF), np.arange(nF, nT))]

    # Now quasi Newton part. First, retain only rows related to b and drop those related to B
    yks = yks[:, acols]
    sks = sks[:, acols]

    # Finally, recompute rho.
    rhos = 1 / np.einsum("ij,ij->i", sks, yks)

    # Can now compute the quasi newton approximation to the Schur complement onH/onHbb (SonHbbqa)
    # and from that the approximation to onHbb.

    # Form initial matrix H0. See Nocedal & Wright, 2004 for sacling factor
    # computed below.
    if len(yks) > 0:
        omega = dampen_HBB * compute_omega(
            yks,
            sks,
            method="mean" if sample_hessian else "NoWr",
        )
    else:
        omega = 1

    H0 = scp.sparse.identity(nA, format="csc") * omega

    # Implicit representation of Schur complement SonHbbqa
    if form == "SR1":
        qat1, qat2, qat3 = computeHSR1(
            sks, yks, rhos, H0, omega, make_psd=True, explicit=False, make_pd=make_pd
        )
    else:
        qat1, qat2, qat3 = computeH(sks, yks, rhos, H0, explicit=False)

    if explicit:
        # At this point we can explicitly form the PSD approximation to nH (conH). First we define
        # P3:
        # len(tcols) * len(acols) permutation matrix to embed quasi Newton approximated block into
        # bottom right corner of ordered negative hessian (onH) via P3 @ onHbbqa @P3.T
        P3 = scp.sparse.csc_array(
            (
                np.tile(1, nA),
                (np.arange(nF, nT), np.arange(nA)),
            ),
            shape=(nT, nA),
        )

        # Now have
        SonHbbqa = H0 + qat1 @ qat2 @ qat3
        onHbbqa = SonHbbqa + onHBb.T @ IonHBB @ onHBb

        # And thus
        onH = onHfd + P3 @ onHbbqa @ P3.T

        # Then we can use P1 to undo the permutation to finally get nH -> this time reversed
        # multiplication order
        nH = P1 @ onH @ P1.T
        return nH

    # Can represent nH implicitly via initial block diagonal matrix and three additive
    # corrections of the form t1 @ t2 @ t3, where t3 is sometimes t1.T

    # Need another permutation matrix for this, P2:
    # len(tcols) * len(acols) permutation matrix transforming quasi Newton approximated block
    # (nHqa) to fit into nH via P2 @ nHqa @ P2.T
    P2 = scp.sparse.csc_array(
        (
            np.tile(1, nA),
            (acols, np.arange(nA)),
        ),
        shape=(nT, nA),
    )

    # Start with block diagonal matrix (before permutation it's block diagonal)
    nH1 = (
        P1
        @ scp.sparse.block_array([[scp.sparse.csc_array(onHBB), None], [None, H0]])
        @ P1.T
    )

    # Now represent first additive update, P2 @ onHBb.T @ IonHBB @ onHBb @ P2.T, implicitly
    nH2t1 = P2 @ onHBb.T
    nH2t2 = IonHBB
    nH2t3 = onHBb @ P2.T

    # For second additive update t2, the central matrix, is the identity so we only need t1 and
    # t3
    if fully_dampened_HBb:
        # Fully dampened case where onHBb is just zeroes
        nH3t1 = None
        nH3t3 = None
    else:
        nH3t1 = P1 @ scp.sparse.block_array(
            [[scp.sparse.eye_array(nF, format="csc"), None], [None, onHBb.T]]
        )
        nH3t3 = (
            scp.sparse.block_array(
                [[None, onHBb], [scp.sparse.eye_array(nF, format="csc"), None]]
            )
            @ P1.T
        )

    # Now final additive update, which is again of the same form as the first, but this time
    # carries the quasi Newton information about the Schur complement
    nH4t1 = scp.sparse.csc_array(P2 @ qat1)
    nH4t2 = qat2
    nH4t3 = scp.sparse.csr_array(qat3 @ P2.T)

    # Now, nH = nH1 + (nH2t1@nH2t2@nH2t3) + (nH3t1@nH3t3) + (nH4t1@nH4t2@nH4t3)
    return nH1, nH2t1, nH2t2, nH2t3, nH3t1, nH3t3, nH4t1, nH4t2, nH4t3


def computeSVPS(
    nH1: scp.sparse.csc_array,
    nH2t1: scp.sparse.csc_array,
    nH2t2: np.ndarray,
    nH2t3: scp.sparse.csr_array,
    nH3t1: scp.sparse.csc_array | None,
    nH3t3: scp.sparse.csr_array | None,
    nH4t1: scp.sparse.csc_array,
    nH4t2: np.ndarray,
    nH4t3: scp.sparse.csr_array,
    S_emb: scp.sparse.csc_array,
    fully_dampened_HBb: bool = False,
    explicit: bool = True,
    pre_cond: bool = True,
    n_c: int = 10,
) -> (
    np.ndarray
    | tuple[
        scp.sparse.csc_array,
        scp.sparse.csc_array,
        np.ndarray,
        scp.sparse.csr_array,
        scp.sparse.csc_array | None,
        np.ndarray | None,
        scp.sparse.csr_array | None,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
):
    """Computes inverse of structured approximation to the negative Hessian of the **penalized
    log-likelihood**.

    Expects matrix of the form returned by :func:`computeSH`, i.e.,:

    .. math::
        :nowrap:

        $$
        \\hat{\\mathbf{H}} =
        \\begin{bmatrix}
            \\mathbf{H}_{\\boldsymbol{\\beta}\\boldsymbol{\\beta}} & \\mathbf{0} \\\\
            \\mathbf{0} & \\mathbf{I}\\omega \\\\
        \\end{bmatrix} +
        \\begin{bmatrix}
            \\mathbf{0} & \\mathbf{0} \\\\
            \\mathbf{0} & \\mathbf{H}_{\\mathbf{b}\\boldsymbol{\\beta}}
            (\\mathbf{H}_{\\boldsymbol{\\beta}\\boldsymbol{\\beta}})^{-1}
            \\mathbf{H}_{\\boldsymbol{\\beta}\\mathbf{b}} \\\\
        \\end{bmatrix} +
        \\begin{bmatrix}
            \\mathbf{0} & \\mathbf{H}_{\\boldsymbol{\\beta}\\mathbf{b}} \\\\
            \\mathbf{H}_{\\mathbf{b}\\boldsymbol{\\beta}} & \\mathbf{0} \\\\
        \\end{bmatrix} +
        \\begin{bmatrix}
            \\mathbf{0} & \\mathbf{0} \\\\
            \\mathbf{0} & \\mathbf{Q}\\boldsymbol{\\Delta}\\mathbf{Q}^\\top \\\\
        \\end{bmatrix}
        $$

    And then computes the inverse :math:`\\mathbf{V}` of
    :math:`\\hat{\\mathbf{H}} + \\mathbf{S}_{\\boldsymbol{\\lambda}}` by applying the modified
    Woodbury identity of Henderson & Searle (1981) three times. Optionally, the second update
    can be skipped in case the off-diagonal derivative blocks
    :math:`\\mathbf{H}_{\\mathbf{b}\\boldsymbol{\\beta}}` contain only zeroes
    (``fully_dampened_HBb is True``).

    Function either returns :math:`\\mathbf{V}` directly, or returns an initial matrix ``V0`` as
    well as nine update matrices, so that::

      V = V0 - (inv1t1 @ inv1t2 @ inv1t3) - (inv2t1 @ inv2t2 @ inv2t3) - (inv3t1 @ inv3t2 @ inv3t3)

    Support approximate diagonal pre-conditioning of the negative Hessian, based on the procedure
    outlined by Wood, Pya, & Säfken (2016).

    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
        Smooth Models.

    :param nH1: Initial approximation to the negative Hessian matrix
    :type nH1: scp.sparse.csc_array
    :param nH2t1: Hessian update matrix 1
    :type nH2t1: scp.sparse.csc_array
    :param nH2t2: Hessian update matrix 2
    :type nH2t2: np.ndarray
    :param nH2t3: Hessian update matrix 3
    :type nH2t3: scp.sparse.csr_array
    :param nH3t1: Hessian update matrix 4
    :type nH3t1: scp.sparse.csc_array | None
    :param nH3t3: Hessian update matrix 5
    :type nH3t3: scp.sparse.csr_array | None
    :param nH4t1: Hessian update matrix 6
    :type nH4t1: scp.sparse.csc_array
    :param nH4t2: Hessian update matrix 7
    :type nH4t2: np.ndarray
    :param nH4t3: Hessian update matrix 8
    :type nH4t3: scp.sparse.csr_array
    :param S_emb: Total penalty matrix to be added to the negative Hessian
    :type S_emb: scp.sparse.csc_array
    :param fully_dampened_HBb: Whether the top right and lower left rectangular blocks (mixed
        partial derivatives of ``coef[fcols]`` with respect to ``coef[acols]``) have been zeroed,
        defaults to False
    :type fully_dampened_HBb: bool, optional
    :param explicit: Whether or not to return the approximate matrix explicitly or implicitly in
        form of an initial matrix and 9 update matrices, defaults to True
    :type explicit: bool
    :param pre_cond: Whether a diagonal pre-conditioner (based on the diagonal of ``nH1 + S_emb``)
        should be applied to the negative Hessian before computing the inverse, defaults to True
    :type pre_cond: bool
    :param n_c: Number of cores to use for multi-processing parts. Defaults to 10
    :type n_c: int, optional
    :return: V, either as np.ndarray (``explicit=='True'``) or represented implicitly via an initial
        matrix and 9 update matrices as defined in the description (i.e.,
        ``V0, inv1t1, inv1t2, inv1t3, inv2t1, inv2t2, inv2t3, inv3t1, inv3t2, inv3t3``).
    :rtype: np.ndarray | tuple[scp.sparse.csc_array, scp.sparse.csc_array, np.ndarray,
        scp.sparse.csr_array, scp.sparse.csc_array | None, np.ndarray | None,
        scp.sparse.csr_array | None, np.ndarray, np.ndarray, np.ndarray,]
    """
    # Now seek implicit representation for inverse of nH + S_emb
    # Can be obtained by applying the modified Woodbury Identity of Henderson and Searle 3 times.
    eps = np.power(np.finfo(float).eps, 0.5)

    # Start with obtaining inverse of nH1 + S_emb
    nH1S = nH1 + S_emb

    # Diagonal pre-conditioning inspired by WPS (2016)
    if pre_cond:
        nHdgr = nH1S.diagonal()
        nHdgr = np.power(np.abs(nHdgr), -0.5)
    else:
        nHdgr = np.ones(nH1.shape[1])

    D = scp.sparse.diags_array(nHdgr, format="csc")

    nH1S = (D @ nH1S @ D).tocsc()
    nH2t1 = D @ nH2t1
    nH2t3 = nH2t3 @ D
    if nH3t1 is not None:
        nH3t1 = D @ nH3t1
        nH3t3 = nH3t3 @ D
    nH4t1 = D @ nH4t1
    nH4t3 = nH4t3 @ D

    Lp, Pr, _ = cpp_cholP(nH1S)  # noqa: F405
    LVp0 = compute_Linv(Lp, n_c)  # noqa: F405
    LV0 = apply_eigen_perm(Pr, LVp0)
    V0 = LV0.T @ LV0

    # Now apply Woodbury identity once to get implicit representation of inverse of
    # nH1 + (nH2t1@nH2t2@nH2t3) + S_emb

    invt2 = np.identity(nH2t2.shape[1]) + nH2t2 @ nH2t3 @ V0 @ nH2t1

    U, sv_invt2, VT = scp.linalg.svd(invt2, lapack_driver="gesvd")
    sv_invt2[sv_invt2 < eps] = eps

    # Now we we can again compute all parts for the modified Woodbury identy to obtain
    # (nH1 + (nH2t1@nH2t2@nH2t3) + S_emb)^{-1}
    inv1t2 = VT.T @ np.diag(1 / sv_invt2) @ U.T
    inv1t1 = (V0 @ nH2t1).tocsc()
    inv1t3 = scp.sparse.csr_array(nH2t2 @ (nH2t3 @ V0))

    if fully_dampened_HBb:
        inv2t2 = None
        inv2t1 = None
        inv2t3 = None
    else:
        # Can now form inverse of
        # nH1 + (nH2t1@nH2t2@nH2t3) + (nH3t1@nH3t3) + S_emb
        # From here on things get messy though, so need to be careful never to explicitly evaluate
        # inverse:
        # (nH1 + (nH2t1@nH2t2@nH2t3) + S_emb)^{-1}
        # so need to watch multiplication order.

        V0nH3t1 = scp.sparse.csc_array(
            (V0 @ nH3t1) - (inv1t1 @ inv1t2 @ (inv1t3 @ nH3t1))
        )

        invt2 = np.identity(nH3t1.shape[1]) + nH3t3 @ V0nH3t1

        U, sv_invt2, VT = scp.linalg.svd(invt2, lapack_driver="gesvd")
        sv_invt2[sv_invt2 < eps] = eps

        # Now we we can again compute all parts for the modified Woodbury identy to obtain
        # (nH1 + (nH2t1@nH2t2@nH2t3) + (nH3t1@nH3t3) + S_emb)^{-1}
        inv2t2 = VT.T @ np.diag(1 / sv_invt2) @ U.T
        inv2t1 = V0nH3t1
        inv2t3 = scp.sparse.csr_array(
            (nH3t3 @ V0) - ((nH3t3 @ inv1t1) @ inv1t2 @ inv1t3)
        )

    # Can now form final inverse of
    # nH1 + (nH2t1@nH2t2@nH2t3) + (nH3t1@nH3t3) + (nH4t1@nH4t2@nH4t3) + S_emb
    # Again need to watch multiplication order to not explicitly evaulate inverse of
    # (nH1 + (nH2t1@nH2t2@nH2t3) + (nH3t1@nH3t3) + S_emb)^{-1}

    if fully_dampened_HBb:
        V0nH4t1 = (V0 @ nH4t1) - (inv1t1 @ inv1t2 @ (inv1t3 @ nH4t1))
    else:
        V0nH4t1 = (
            (V0 @ nH4t1)
            - (inv1t1 @ inv1t2 @ (inv1t3 @ nH4t1))
            - (inv2t1 @ inv2t2 @ (inv2t3 @ nH4t1))
        )

    invt2 = np.identity(nH4t2.shape[1]) + nH4t2 @ nH4t3 @ V0nH4t1

    U, sv_invt2, VT = scp.linalg.svd(invt2, lapack_driver="gesvd")
    sv_invt2[sv_invt2 < eps] = eps

    # Now we we can again compute all parts for the modified Woodbury identy to obtain
    # (nH1 + (nH2t1@nH2t2@nH2t3) + (nH3t1@nH3t3) + (nH4t1@nH4t2@nH4t3) + S_emb)^{-1}
    inv3t2 = VT.T @ np.diag(1 / sv_invt2) @ U.T
    inv3t1 = V0nH4t1
    if fully_dampened_HBb:
        inv3t3 = nH4t2 @ ((nH4t3 @ V0) - ((nH4t3 @ inv1t1) @ inv1t2 @ inv1t3))
    else:
        inv3t3 = nH4t2 @ (
            (nH4t3 @ V0)
            - ((nH4t3 @ inv1t1) @ inv1t2 @ inv1t3)
            - ((nH4t3 @ inv2t1) @ inv2t2 @ inv2t3)
        )

    # Can form final inverse!
    if explicit:
        V = V0 - (inv1t1 @ inv1t2 @ inv1t3)

        if fully_dampened_HBb is False:
            V -= inv2t1 @ inv2t2 @ inv2t3

        V -= inv3t1 @ inv3t2 @ inv3t3

        return D @ V @ D

    return (
        (D @ V0 @ D).tocsc(),
        D @ inv1t1,
        inv1t2,
        inv1t3 @ D,
        D @ inv2t1 if inv2t1 is not None else None,
        inv2t2,
        inv2t3 @ D if inv2t3 is not None else None,
        D @ inv3t1,
        inv3t2,
        inv3t3 @ D,
    )
