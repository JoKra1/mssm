import numpy as np
import scipy as scp
import warnings
from .custom_types import DerivOrder
from .discrete import DiscreteModelMatrix


def deriv_transform_mu_eta(
    d1: list[np.ndarray] | None,
    d2: list[np.ndarray] | None,
    d2m: list[np.ndarray] | None,
    ld1: list[np.ndarray],
    ld2: list[np.ndarray] | None,
    order: DerivOrder | None = None,
) -> tuple[list[np.ndarray] | None, list[np.ndarray] | None, list[np.ndarray] | None]:
    """Transform derivatives (first and second order) of llk with respect to each parameter (i.e.,
    mean) into derivatives with respect to each linear predictor following steps outlined by Wood,
    Pya, & Säfken (2016)

    References:
      - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth \
        Models.

    :param d1: List holding the first order partial derivatives of llk with respect to each
        parameter. Can be None unless ``order`` is None, ``DerivOrder.d1``, or ``DerivOrder.d2``.
    :type d1: list[np.ndarray] | None
    :param d2: List holding the pure second order partial derivatives of llk with respect to each
        parameter. Can be None unless ``order`` is None or ``DerivOrder.d2``.
    :type d2: list[np.ndarray] | None
    :param d2m: List holding the mixed second order partial derivatives of llk with respect to
        each pair of parameters. Can be None unless ``order`` is None or ``DerivOrder.d2m``.
    :type d2m: list[np.ndarray] | None
    :param ld1: List holding the first derivative of each link function evaluated at the
        corresponding parameter.
    :type ld1: list[np.ndarray]
    :param ld2: List holding the second derivative of each link function evaluated at the
        corresponding parameter. Can be None unless ``order`` is None or ``DerivOrder.d2``.
    :type ld2: list[np.ndarray] | None
    :param order: Which derivatives to transform. If None, all of them are transformed, defaults
        to None
    :type order: DerivOrder | None, optional
    :return: A tuplewith three lists. First list contains the first order partial derivatives with
        respect to each linear predictor, second list is the same for pure second derivatives, and
        third list contains mixed second derivatives. Lists can be None if ``order is not None`` in
        which case only the list requested via ``order`` is evaluated (i.e., not None).
    :rtype: tuple[list[np.ndarray] | None, list[np.ndarray] | None, list[np.ndarray] | None]
    """
    n_par = len(ld1)
    d1eta = None
    d2eta = None
    d2meta = None

    # Transform first order derivatives via A.1 in Wood, Pya, & Säfken (2016)

    # WPS (2016) provide that $l_{\\eta}$ is obtained as $l^i_{\\mu}/h'(\\mu^i)$ - where
    # $h'$ is the derivative of the link function $h$.
    # This follows from applying the chain rule and the inversion rule of derivatives
    # $\frac{\\partial llk(h^{-1}(\\eta))}{\\partial \\eta} =
    # \frac{\\partial llk(\\mu)}{\\partial \\mu} \frac{\\partial h^{-1}(\\eta)}{\\partial \\eta} =
    # \frac{\\partial llk(\\mu)}{\\partial \\mu}\frac{1}{\frac{\\partial h(\\mu)}{\\mu}}$.

    if order is None or order == DerivOrder.d1:
        # d1eta = [d1[mui]/ld1[mui] for mui in range(n_par)]
        d1eta = []
        for mui in range(n_par):

            with warnings.catch_warnings():  # Divide by 0
                warnings.simplefilter("ignore")
                de = d1[mui] / ld1[mui]
                # d1eta.append(d1[mui]/ld1[mui])

            de[np.isnan(de) | np.isinf(de)] = 0
            d1eta.append(de)

    # Pure second order derivatives are transformed via A.2 in WPS (2016)

    # For second derivatives we need pure and mixed. Computation of
    # $l^\\mathbf{i}_{\\eta^l,\\eta^m}$ in general is again obtained by applying the steps outlined
    # for first order and provided by WPS (2016) for the pure case. For the mixed case it is even
    # simpler: We need
    # $\frac{\\partial^2 llk(h_1^{-1}(\\eta^1),h_2^{-1}(\\eta^2))}\
    # {\\partial \\eta^1 \\partial \\eta^2}$,
    # which is $\frac{\\partial llk / \\partial \\eta^1}{\\partial \\eta^2}$.
    # With the first partial being equal to
    # $\frac{\\partial llk}{\\partial \\mu^1}\frac{1}{\frac{\\partial h_1(\\mu^1)}{\\mu^1}}$
    # (see comment above for first order) we now have
    # $\frac{\\partial \frac{\\partial llk}{\\partial \\mu^1}\frac{1}\
    # {\frac{\\partial h_1(\\mu^1)}{\\mu^1}}}{\\partial \\eta^2}$.

    # We now apply the product rule (the second term in the sum disappears, because
    # $\\partial \frac{1}{\frac{\\partial h_1(\\mu^1)}{\\mu^1}} / \\partial \\eta^2 = 0$,
    # this is not the case for pure second derivatives as shown in WPS, 2016)
    # to get $\frac{\\partial \frac{\\partial llk}{\\partial \\mu^1}}{\\partial \\eta^2}\
    #  \frac{1}{\frac{\\partial h_1(\\mu^1)}{\\mu^1}}$.
    # We can now again rely on the same steps taken to get the first derivatives
    # (chain rule + inversion rule) to get
    # $\frac{\\partial \frac{\\partial llk}{\\partial \\mu^1}}{\\partial \\eta^2} =
    # \frac{\\partial^2 llk}{\\partial \\mu^1 \\partial \
    # \\mu^2}\frac{1}{\frac{\\partial h_2(\\mu^2)}{\\mu^2}}$.

    # Thus, $\frac{\\partial llk / \\partial \\eta^1}{\\partial \\eta^2} =
    # \frac{\\partial^2 llk}{\\partial \\mu^1 \\partial \\mu^2}\frac{1}{\frac{\\partial \
    # h_2(\\mu^2)}{\\mu^2}}\frac{1}{\frac{\\partial h_1(\\mu^1)}{\\mu^1}}$.

    if order is None or order == DerivOrder.d2:
        d2eta = []
        for mui in range(n_par):

            with warnings.catch_warnings():  # Divide by 0
                warnings.simplefilter("ignore")
                d2e = d2[mui] / np.power(ld1[mui], 2) - d1[mui] * ld2[mui] / np.power(
                    ld1[mui], 3
                )
                # d2eta.append(d2[mui]/np.power(ld1[mui],2) - d1[mui]*ld2[mui]/np.power(ld1[mui],3))

            d2e[np.isnan(d2e) | np.isinf(d2e)] = 0
            d2eta.append(d2e)

    # Mixed second derivatives thus also are transformed as proposed by WPS (2016)
    if order is None or order == DerivOrder.d2m:
        d2meta = []
        mixed_idx = 0
        for mui in range(n_par):
            for muj in range(mui + 1, n_par):

                with warnings.catch_warnings():  # Divide by 0
                    warnings.simplefilter("ignore")
                    d2em = d2m[mixed_idx] * (1 / ld1[mui]) * (1 / ld1[muj])
                    # d2meta.append(d2m[mixed_idx] * (1/ld1[mui]) * (1/ld1[muj]))

                d2em[np.isnan(d2em) | np.isinf(d2em)] = 0
                d2meta.append(d2em)

                mixed_idx += 1

    return d1eta, d2eta, d2meta


def deriv_transform_eta_beta(
    d1eta: list[np.ndarray],
    d2eta: list[np.ndarray] | None,
    d2meta: list[np.ndarray] | None,
    Xs: list[scp.sparse.csc_array | np.ndarray | DiscreteModelMatrix],
    only_grad: bool = False,
    sparse: bool = True,
) -> tuple[np.ndarray, np.ndarray | scp.sparse.csc_array | None]:
    """
    Further transforms derivatives of llk with respect to eta to get derivatives of llk with
    respect to coefficients.
    Based on section 3.2 and Appendix A in Wood, Pya, & Säfken (2016)

    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth
     Models.

    :param d1eta: List containing the first order partial derivatives of llk with respect to each
        linear predictor.
    :type d1eta: list[np.ndarray]
    :param d2eta: List containing the pure second order partial derivatives of llk with respect to
        each linear predictor. Can be None if ``only_grad=True``.
    :type d2eta: list[np.ndarray] | None
    :param d2meta: List containing the mixed second order partial derivatives of llk with respect
        to each pair of linear predictors. Can be None if ``only_grad=True``.
    :type d2meta: list[np.ndarray] | None
    :param Xs: List of model matrices - one per parameter of llk
    :type Xs: list[scp.sparse.csc_array | np.ndarray | DiscreteModelMatrix]
    :param only_grad: Whether to only compute the gradient, defaults to False
    :type only_grad: bool, optional
    :param sparse: Whether to evaluate the Hessian as a sparse array rather than a dense one,
        defaults to True
    :type sparse: bool, optional
    :return: A tuple containing the gradient of llk with respect to the coefficients and the
        Hessian of llk with respect to the coefficients (None if ``only_grad=True``).
    :rtype: tuple[np.ndarray, np.ndarray | scp.sparse.csc_array | None]
    """

    # Gradient: First order partial derivatives of llk with respect to coefficients

    # WPS (2016) provide $l^j_{\beta^l} = l^\\mathbf{i}_{\\eta^l}\\mathbf{X}^l_{\\mathbf{i},j}$.
    # See ```deriv_transform_mu_eta```.

    grad = []

    for etai in range(len(d1eta)):
        # Naive:
        # for j in range(Xs[etai].shape[1]):
        # grad.append(np.sum([d1eta[etai][i]*Xs[etai][i,j] for i in range(Xs[etai].shape[0])]))
        # but this is just product of d1eta[etai][:] and Xs[etai], so we can skip inner loop
        # val_idx = ((np.isnan(d1eta[etai]) | np.isinf(d1eta[etai])) == False).flatten()
        # grad.extend(((d1eta[etai][val_idx]).T @ Xs[etai][val_idx,:]).T)
        grad.extend((d1eta[etai].T @ Xs[etai]).T)

    if only_grad:
        return np.array(grad).reshape(-1, 1), None
    # Hessian: Second order partial derivatives of llk with respect to coefficients

    # WPS (2016) provide in general:
    # $l^{j,k}_{\beta^l,\beta^m} =
    # l^\\mathbf{i}_{\\eta^l,\\eta^m}\\mathbf{X}^l_{\\mathbf{i},j}\\mathbf{X}^m_{\\mathbf{i},k}$.
    # See ```deriv_transform_mu_eta```.

    mixed_idx = 0

    hr_idx = 0
    h_rows = []
    h_cols = []
    h_vals = []
    hessian = None if sparse else np.zeros((len(grad), len(grad)))
    cols = None if sparse else np.arange(len(grad))
    for etai in range(len(d1eta)):
        hc_idx = 0
        for etaj in range(len(d1eta)):

            if etaj < etai:
                hc_idx += Xs[etaj].shape[1]
                continue

            if etai == etaj:
                # Pure 2nd
                d2 = d2eta[etai]
            else:
                # Mixed partial
                d2 = d2meta[mixed_idx]
                mixed_idx += 1

            # val_idx = ((np.isnan(d2) | np.isinf(d2)) == False).flatten()
            if sparse:
                for coefi in range(Xs[etai].shape[1]):

                    d2beta = (d2 * Xs[etai][:, [coefi]]).T @ Xs[etaj][
                        :,
                        (coefi if etai == etaj else 0) : Xs[etaj].shape[  # noqa: E203
                            1
                        ],
                    ]
                    # print(d2beta)

                    if d2beta.nnz > 0:

                        # Sort, to make symmetric deriv extraction easier
                        if not d2beta.has_sorted_indices:
                            d2beta.sort_indices()

                        # Get non-zero column entries for current row
                        cols = d2beta.indices[
                            d2beta.indptr[0] : d2beta.indptr[1]  # noqa: E203
                        ] + (coefi if etai == etaj else 0)

                        # Get non-zero values for current row in sorted order
                        vals = d2beta.data[
                            d2beta.indptr[0] : d2beta.indptr[1]  # noqa: E203
                        ]

                        h_rows.extend(np.tile(coefi, d2beta.nnz) + hr_idx)
                        h_cols.extend(cols + hc_idx)
                        h_vals.extend(vals)

                        # Symmetric 2nd deriv..
                        if etai == etaj and ((cols[0] + hc_idx) == (coefi + hr_idx)):
                            # For diagonal block need to skip diagonal
                            # element if present (i.e., non-zero)
                            h_rows.extend(cols[1:] + hc_idx)
                            h_cols.extend(np.tile(coefi, d2beta.nnz - 1) + hr_idx)
                            h_vals.extend(vals[1:])
                        else:  # For off-diagonal block can assign everything
                            h_rows.extend(cols + hc_idx)
                            h_cols.extend(np.tile(coefi, d2beta.nnz) + hr_idx)
                            h_vals.extend(vals)
            else:
                # Unpack directly
                d2beta = Xs[etai].T @ scp.sparse.diags_array(d2.flatten()) @ Xs[etaj]

                hessian[
                    np.ix_(
                        cols[hr_idx : (hr_idx + Xs[etai].shape[1])],  # noqa: E203
                        cols[hc_idx : (hc_idx + Xs[etaj].shape[1])],  # noqa: E203
                    )
                ] = d2beta

                if etai != etaj:
                    hessian[
                        np.ix_(
                            cols[hc_idx : (hc_idx + Xs[etaj].shape[1])],  # noqa: E203
                            cols[hr_idx : (hr_idx + Xs[etai].shape[1])],  # noqa: E203
                        )
                    ] = d2beta.T

            hc_idx += Xs[etaj].shape[1]
        hr_idx += Xs[etai].shape[1]

    if sparse:
        hessian = scp.sparse.csc_array((h_vals, (h_rows, h_cols)))
    return np.array(grad).reshape(-1, 1), hessian
