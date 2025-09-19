import numpy as np
import scipy as scp
from .custom_types import Constraint, ConstType, LambdaTerm, PenType
from .matrix_solvers import translate_sparse
from collections.abc import Callable
import sys

################################## Penalty functions ################################## # noqa: E266


def sort_penalties(penalties: list[LambdaTerm]) -> list[LambdaTerm]:
    """Sorts penalties by ``start_index`` in ascending order.

    :param penalties: A list of term-specific penalties.
    :type penalties: list[LambdaTerm]
    :return: A list of term-specific penalties, sorted by start index.
    :rtype: list[LambdaTerm]
    """

    idx = [pen.start_index for pen in penalties]
    sort_idx = np.argsort(idx)
    return [penalties[sidx] for sidx in sort_idx]


def create_id_dict(penalties: list[LambdaTerm]) -> dict | None:
    """Identifies penalties that should share a lambda parameter and fills a dictionary holding
    penalty indices for each shared penalty (id).

    :param penalties: A list of term-specific penalties, some of which might have a ``id`` flag.
    :type penalties: list[LambdaTerm]
    :return: A dictionary. Keys are ids shared by two or more penalties. Values are lists holding
        corresponding penalty indices. If no id exists that is shared by two or more penalties,
        None is returned instead.
    :rtype: dict|None
    """
    id_dict = {}

    for peni, pen in enumerate(penalties):
        if pen.id is not None:
            if pen.id in id_dict:
                id_dict[pen.id].append(peni)
            else:
                id_dict[pen.id] = [peni]

    if len(id_dict) > 0 and max([len(v) for v in id_dict.values()]) > 1:
        return id_dict

    return None


def combine_shared_penalties(penalties: list[LambdaTerm]) -> list[LambdaTerm]:
    """Identifies penalties that should share a lambda parameter and merges them into a single
    :class:`LambdaTerm`.

    :param penalties: A list of term-specific penalties, some of which might have a ``id`` flag.
    :type penalties: list[LambdaTerm]
    :return: A list of penalty matrices in which individual penalties with a shared ``id`` flag
        have been combined into a single :class:`LambdaTerm`.
    :rtype: list[LambdaTerm]
    """

    id_dict = create_id_dict(penalties)

    if id_dict is None:
        return penalties
    else:
        merged_penalties = [
            pen
            for pen in penalties
            if (pen.id not in id_dict) or (len(id_dict[pen.id]) <= 1)
        ]

        for id in id_dict.keys():
            m_pens = [pen for pen in penalties if pen.id == id]
            S_J_emb = m_pens[0].S_J_emb
            D_J_emb = m_pens[0].D_J_emb

            for peni in range(1, len(m_pens)):
                S_J_emb += m_pens[peni].S_J_emb
                D_J_emb += m_pens[peni].D_J_emb

            m_pen = LambdaTerm(
                S_J=S_J_emb[
                    m_pens[0].start_index : (  # noqa: E203
                        m_pens[-1].start_index
                        + (m_pens[-1].S_J.shape[1] * m_pens[-1].rep_sj)
                    ),
                    m_pens[0].start_index : (  # noqa: E203
                        m_pens[-1].start_index
                        + (m_pens[-1].S_J.shape[1] * m_pens[-1].rep_sj)
                    ),
                ],
                S_J_emb=S_J_emb,
                D_J_emb=D_J_emb,
                lam=m_pens[0].lam,
                rep_sj=1,
                start_index=m_pens[0].start_index,
                type=PenType.SHARED,
                rank=np.sum([pen.rank for pen in m_pens]),
                term=[pen.term for pen in m_pens],
                dist_param=[pen.dist_param for pen in m_pens],
                S_Js=[pen.S_J for pen in m_pens],
                S_J_embs=[pen.S_J_emb for pen in m_pens],
                D_J_embs=[pen.D_J_emb for pen in m_pens],
                rep_sjs=[pen.rep_sj for pen in m_pens],
                start_indices=[pen.start_index for pen in m_pens],
                types=[pen.type for pen in m_pens],
                ranks=[pen.rank for pen in m_pens],
                id=id,
            )

            merged_penalties.append(m_pen)

        return merged_penalties


def split_shared_penalties(merged_penalties: list[LambdaTerm]) -> list[LambdaTerm]:
    """Identifies penalties that share a lambda parameter and splits them into individual
    :class:`LambdaTerm`s - all having the same :math:`\\lambda` value.

    Basically inverts what is achieved by the :func:`combine_shared_penalties` function.

    :param merged_penalties:  list of penalty matrices in which individual penalties with a shared
        ``id`` flag have been combined into a single :class:`LambdaTerm`.
    :type merged_penalties: list[LambdaTerm]
    :return: A list of penalty matrices in which merged penalties have been split into individual
        penalties again.
    :rtype: list[LambdaTerm]
    """

    merged_idx = []
    for peni, pen in enumerate(merged_penalties):
        if isinstance(pen.S_Js, list):
            merged_idx.append(peni)

    penalties = [
        merged_penalties[peni]
        for peni in range(len(merged_penalties))
        if peni not in merged_idx
    ]

    if len(merged_idx) == 0:
        return penalties

    else:
        for idx in merged_idx:

            merged_pen = merged_penalties[idx]

            for m in range(len(merged_pen.S_Js)):

                penalties.append(
                    LambdaTerm(
                        S_J=merged_pen.S_Js[m],
                        S_J_emb=merged_pen.S_J_embs[m],
                        D_J_emb=merged_pen.D_J_embs[m],
                        lam=merged_pen.lam,
                        rep_sj=merged_pen.rep_sjs[m],
                        start_index=merged_pen.start_indices[m],
                        type=merged_pen.types[m],
                        rank=merged_pen.ranks[m],
                        term=merged_pen.term[m],
                        dist_param=merged_pen.dist_param[m],
                        id=merged_pen.id,
                    )
                )

        # Sort back by start indices
        penalties = sort_penalties(penalties)
        return penalties


def adjust_pen_drop(
    dat: list[float], rows: list[int], cols: list[int], drop: list[int], offset: int = 0
) -> tuple[list[float], list[int], list[int], int]:
    """Adjusts penalty matrix (represented via ``dat``, ``rows``, and ``cols``) by dropping rows
    and columns indicated by ``drop``.

    Optionally, ``offset`` is added to the elements in ``rows`` and ``cols``, which is useful when
    indices in ``drop`` do not start at zero.

    :param dat: List of elements in penalty matrix.
    :type dat: [float]
    :param rows: List of row indices of penalty matrix.
    :type rows: [int]
    :param cols: List of column indices of penalty matrix.
    :type cols: [int]
    :param drop: Rows and columns to drop from penalty matrix. Might actually contain indices
        corresponding to ``rows + offset`` and ``cols + offset``, which can be corrected for via
        the ``offset`` argument.
    :type drop: [int]
    :param offset: An optional offset to add to ``rows`` and ``cols`` to adjust for the indexing in
        ``drop``, defaults to 0
    :type offset: int, optional
    :return: A tuple with 4 elements: the data, rows, and cols of the adjusted penalty matrix
        excluding dropped elements and the number of excluded elements.
    :rtype: tuple[list[float],list[int],list[int],int]
    """
    rows = np.array(rows)
    cols = np.array(cols)
    dat = np.array(dat)

    drop_idx = np.isin(drop, cols + offset)
    dropped = np.sum(drop_idx)
    drop = np.array(drop)[drop_idx]

    keep_col = ~np.isin(cols + offset, drop)
    keep_row = ~np.isin(rows + offset, drop)

    keep = keep_col & keep_row

    # Now adjust cols & rows
    rows_realign = np.zeros_like(rows)
    cols_realign = np.zeros_like(cols)

    rows_realign[:] = rows
    cols_realign[:] = cols

    for d in drop:
        rows_realign[rows + offset > d] -= 1
        cols_realign[cols + offset > d] -= 1

    # Now return
    return list(dat[keep]), list(rows_realign[keep]), list(cols_realign[keep]), dropped


def embed_in_S_sparse(
    pen_data: list[float],
    pen_rows: list[int],
    pen_cols: list[int],
    S_emb: scp.sparse.csc_array | None,
    S_col: int,
    SJ_col: int,
    cIndex: int,
) -> tuple[scp.sparse.csc_array, int]:
    """Embed a term-specific penalty matrix ``SJ`` (provided as three lists: ``pen_data``,
    ``pen_rows`` and ``pen_cols``) into the total penalty matrix ``S_emb`` (see Wood, 2017)

    :param pen_data: Data of ``SJ``
    :type pen_data: list[float]
    :param pen_rows: Row indices of ``SJ``
    :type pen_rows: list[int]
    :param pen_cols: Column indices of ``SJ``
    :type pen_cols: list[int]
    :param S_emb: Total penalty matrix or ``None`` in case ``S_emb`` will be initialized by the
        function.
    :type S_emb: scp.sparse.csc_array | None
    :param S_col: Columns of total penalty matrix
    :type S_col: int
    :param SJ_col: Columns of ``SJ``
    :type SJ_col: int
    :param cIndex: Current row and column index indicating the top left cell of the
        (``SJ_col`` * ``SJ_col``) block ``SJ`` should take up in ``S_emb``
    :type cIndex: int
    :return: ``S_emb`` with ``SJ`` embedded, the updated ``cIndex`` (i.e., ``cIndex + SJ_col``)
    :rtype: tuple[scp.sparse.csc_array,int]
    """

    embedding = np.array(pen_data)
    r_embedding = np.array(pen_rows) + cIndex
    c_embedding = np.array(pen_cols) + cIndex

    if S_emb is None:
        S_emb = scp.sparse.csc_array(
            (embedding, (r_embedding, c_embedding)), shape=(S_col, S_col)
        )
    else:
        S_emb += scp.sparse.csc_array(
            (embedding, (r_embedding, c_embedding)), shape=(S_col, S_col)
        )

    return S_emb, cIndex + SJ_col


def embed_in_Sj_sparse(
    pen_data: list[float],
    pen_rows: list[int],
    pen_cols: list[int],
    Sj: scp.sparse.csc_array | None,
    SJ_col: int,
) -> scp.sparse.csc_array:
    """Parameterize a term-specific penalty matrix ``SJ`` (provided as three lists: ``pen_data``,
    ``pen_rows`` and ``pen_cols``).

    :param pen_data: Data of ``SJ``
    :type pen_data: list[float]
    :param pen_rows: Row indices of ``SJ``
    :type pen_rows: list[int]
    :param pen_cols: Column indices of ``SJ``
    :type pen_cols: list[int]
    :param Sj: A sparse matrix or ``None``. In the latter case, ``SJ`` is simply initialized by the
        function. If not, then the function returns ``SJ + Sj``. The latter is useful if a term
        penalty is a sum of individual penalty matrices.
    :type Sj: scp.sparse.csc_array | None
    :param SJ_col: Columns of ``SJ``
    :type SJ_col: int
    :return: ``SJ`` which might actually be ``SJ + Sj``.
    :rtype: scp.sparse.csc_array
    """
    embedding = np.array(pen_data)

    if Sj is None:
        Sj = scp.sparse.csc_array(
            (embedding, (pen_rows, pen_cols)), shape=(SJ_col, SJ_col)
        )
    else:
        Sj += scp.sparse.csc_array(
            (embedding, (pen_rows, pen_cols)), shape=(SJ_col, SJ_col)
        )

    return Sj


def embed_shared_penalties(
    shared_penalties: list[list[LambdaTerm]], formulas: list, extra_coef: int
) -> list[LambdaTerm]:
    """Embed penalties from individual formulas into overall penalties for GAMMLSS/GSMM models.

    :param shared_penalties: Nested list, with the inner one containing the penalties associated
        with an individual formula in ``formulas``.
    :type shared_penalties: list[list[LambdaTerm]]
    :param formulas: List of :class:`mssm.src.python.formula.Formula` objects
    :type formulas: list
    :param extra_coef: Number of extra coefficients required by the model's family. Will result in
        the shared penalties being padded by an extra block of ``extra_coef`` zeroes.
    :type extra_coef: int
    :return: A list of the embedded penalties required by a GAMMLSS or GSMM model.
    :rtype: list[LambdaTerm]
    """
    # Assign original formula index to each penalty
    for fi in range(len(shared_penalties)):
        for lterm in shared_penalties[fi]:
            lterm.dist_param = fi

    for fi, form in enumerate(formulas):
        for ofi, other_form in enumerate(formulas):
            if fi == ofi:
                continue

            if ofi < fi:
                for lterm in shared_penalties[fi]:
                    lterm.S_J_emb = scp.sparse.vstack(
                        [
                            scp.sparse.csc_array(
                                (other_form.n_coef, lterm.S_J_emb.shape[1])
                            ),
                            lterm.S_J_emb,
                        ]
                    ).tocsc()
                    lterm.D_J_emb = scp.sparse.vstack(
                        [
                            scp.sparse.csc_array(
                                (other_form.n_coef, lterm.S_J_emb.shape[1])
                            ),
                            lterm.D_J_emb,
                        ]
                    ).tocsc()

                    lterm.S_J_emb = scp.sparse.hstack(
                        [
                            scp.sparse.csc_array(
                                (lterm.S_J_emb.shape[0], other_form.n_coef)
                            ),
                            lterm.S_J_emb,
                        ]
                    ).tocsc()

                    lterm.D_J_emb = scp.sparse.hstack(
                        [
                            scp.sparse.csc_array(
                                (lterm.S_J_emb.shape[0], other_form.n_coef)
                            ),
                            lterm.D_J_emb,
                        ]
                    ).tocsc()

                    lterm.start_index += other_form.n_coef

            elif ofi > fi:
                for lterm in shared_penalties[fi]:
                    lterm.S_J_emb = scp.sparse.vstack(
                        [
                            lterm.S_J_emb,
                            scp.sparse.csc_array(
                                (other_form.n_coef, lterm.S_J_emb.shape[1])
                            ),
                        ]
                    ).tocsc()

                    lterm.D_J_emb = scp.sparse.vstack(
                        [
                            lterm.D_J_emb,
                            scp.sparse.csc_array(
                                (other_form.n_coef, lterm.S_J_emb.shape[1])
                            ),
                        ]
                    ).tocsc()

                    lterm.S_J_emb = scp.sparse.hstack(
                        [
                            lterm.S_J_emb,
                            scp.sparse.csc_array(
                                (lterm.S_J_emb.shape[0], other_form.n_coef)
                            ),
                        ]
                    ).tocsc()

                    lterm.D_J_emb = scp.sparse.hstack(
                        [
                            lterm.D_J_emb,
                            scp.sparse.csc_array(
                                (lterm.S_J_emb.shape[0], other_form.n_coef)
                            ),
                        ]
                    ).tocsc()

    if extra_coef is not None:
        for fi, form in enumerate(formulas):
            for lterm in shared_penalties[fi]:
                lterm.S_J_emb = scp.sparse.vstack(
                    [
                        lterm.S_J_emb,
                        scp.sparse.csc_array((extra_coef, lterm.S_J_emb.shape[1])),
                    ]
                ).tocsc()

                lterm.D_J_emb = scp.sparse.vstack(
                    [
                        lterm.D_J_emb,
                        scp.sparse.csc_array((extra_coef, lterm.S_J_emb.shape[1])),
                    ]
                ).tocsc()

                lterm.S_J_emb = scp.sparse.hstack(
                    [
                        lterm.S_J_emb,
                        scp.sparse.csc_array((lterm.S_J_emb.shape[0], extra_coef)),
                    ]
                ).tocsc()

                lterm.D_J_emb = scp.sparse.hstack(
                    [
                        lterm.D_J_emb,
                        scp.sparse.csc_array((lterm.S_J_emb.shape[0], extra_coef)),
                    ]
                ).tocsc()

    return shared_penalties


class Penalty:
    """Penalty base-class. Generates penalty matrices for smooth terms.

    :param pen_type: Type of the penalty matrix
    :type pen_type: PenType
    :ivar PenType pen_type: Type of the penalty matrix passed to the init method.
    """

    def __init__(self, pen_type: PenType) -> None:
        self.type = pen_type

    def constructor(
        self, n: int, constraint: Constraint | None, *args, **kwargs
    ) -> tuple[
        list[float], list[int], list[int], list[float], list[int], list[int], int
    ]:
        """Creates penalty matrix + root of the penalty and returns both in list form (data, row
        indices, col indices).

        :param n: Dimension of square penalty matrix
        :type n: int
        :param constraint: Any contraint to absorb by the penalty or None if no constraint is
            required
        :type constraint: Constraint | None
        :return: penalty data, penalty row indices, penalty column indices, root of penalty data,
            root of penalty row indices, root of penalty column indices, rank of penalty
        :rtype: tuple[list[float], list[int], list[int], list[float], list[int], list[int], int]
        """
        pass


class DifferencePenalty(Penalty):
    """Difference Penalty class. Generates penalty matrices for smooth terms.

    :ivar PenType.DIFFERENCE pen_type: Type of the penalty matrix.
    """

    def __init__(self):
        super().__init__(PenType.DIFFERENCE)

    def constructor(
        self, n: int, constraint: Constraint | None, m: int = 2
    ) -> tuple[
        list[float], list[int], list[int], list[float], list[int], list[int], int
    ]:
        """Creates difference (order=m) n*n penalty matrix + root of the penalty. Based on code in
        Eilers & Marx (1996) and Wood (2017).

        References:
           - Eilers, P. H. C., & Marx, B. D. (1996). Flexible smoothing with B-splines and \
            penalties. Statistical Science, 11(2), 89–121. https://doi.org/10.1214/ss/1038425655
           - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param n: Dimension of square penalty matrix
        :type n: int
        :param constraint: Any contraint to absorb by the penalty or None if no constraint is
            required
        :type constraint: Constraint|None
        :param m: Differencing order to apply to the identity matrix to get the penalty (this will
            also be the dimension of the penalty's Kernel), defaults to 2
        :type m: int, optional
        :return: penalty data,penalty row indices,penalty column indices,root of penalty data,root
            of penalty row indices,root of penalty column indices,rank of penalty
        :rtype: tuple[list[float],list[int],list[int],list[float],list[int],list[int],int]
        """
        D = np.diff(np.identity(n), m)
        S = D @ D.T
        # Eilers & Marx (1996): P-spline penalties consider m-degree polynomial as
        # smooth, i.e., un-penalized!
        rank = n - m

        # Absorb any identifiability constraints
        if constraint is not None:
            Z = constraint.Z
            if constraint.type == ConstType.QR:
                S = Z.T @ S @ Z
                D = Z.T @ D
            elif constraint.type == ConstType.DROP:
                S = np.delete(np.delete(S, Z, axis=1), Z, axis=0)
                D = np.delete(D, Z, axis=0)
            elif constraint.type == ConstType.DIFF:
                if m == 0:
                    raise ValueError(
                        (
                            "When using ConstType.DIFF, for a term with a difference penalty, `m` "
                            "must be greater than zero!"
                        )
                    )
                # Correct for column differencing applied to X! See smoothCon help for
                # mgcv (Wood, 2017)
                D = np.diff(
                    np.concatenate(
                        (D[Z : D.shape[0], :], D[:Z, :]), axis=0  # noqa: E203
                    ),
                    axis=0,
                )
                D = np.concatenate(
                    (D[D.shape[0] - Z :, :], D[: D.shape[0] - Z, :]),  # noqa: E203
                    axis=0,
                )
                S = D @ D.T

            if m == 0:
                # Re-compute root
                eig, U = scp.linalg.eigh(S)
                D = U @ np.diag(
                    [
                        (
                            np.power(e, 0.5)
                            if e > np.power(sys.float_info.epsilon, 0.7)
                            else 0
                        )
                        for e in eig
                    ]
                )

        S = scp.sparse.csc_array(S)
        D = scp.sparse.csc_array(D)

        # Data in S and D is in canonical format, for competability this is translated to
        # data, rows, columns
        pen_data, pen_rows, pen_cols = translate_sparse(S)
        chol_data, chol_rows, chol_cols = translate_sparse(D)

        return pen_data, pen_rows, pen_cols, chol_data, chol_rows, chol_cols, rank


class IdentityPenalty(Penalty):
    """Difference Penalty class. Generates penalty matrices for smooth terms and random terms.

    :param pen_type: Type of the penalty matrix
    :type pen_type: PenType
    :ivar PenType pen_type: Type of the penalty matrix passed to init method.
    """

    def __init__(self, pen_type: PenType):
        if pen_type not in [PenType.IDENTITY, PenType.DISTANCE]:
            raise ValueError(
                f"pen_type must be PenType.IDENTITY or PenType.DISTANCE, but is {pen_type}"
            )
        super().__init__(pen_type)

    def constructor(
        self, n: int, constraint: Constraint | None, f: Callable | None = None
    ) -> tuple[
        list[float], list[int], list[int], list[float], list[int], list[int], int
    ]:
        """Creates identity matrix penalty + root in case ``f is None``.

        **Note**: This penalty never absorbs marginal constraints. It always returns an identity
        matrix but just decreases ``n`` by 1 if ``constraint is not None``
        to ensure that the returned penalty matrix is of suitable dimensions.

        :param n: Dimension of square penalty matrix
        :type n: int
        :param constraint: Any contraint to absorb by the penalty or None if no constraint is
            required
        :type constraint: Constraint|None
        :param f: Any kind of function to apply to the diagonal elements of the penalty, defaults
            to None
        :type f: Callable|None, optional
        :return: penalty data, penalty row indices, penalty column indices, root of penalty data,
            root of penalty row indices, root of penalty column indices, rank of penalty
        :rtype: tuple[list[float],list[int],list[int],list[float],list[int],list[int],int]
        """
        # Can be used to create event-distance weighted penalty matrices for deconvolving sms GAMMs
        if constraint is not None:
            n -= 1

        elements = [0.0 for _ in range(n)]
        idx = [0.0 for _ in range(n)]

        for i in range(n):
            if f is None:
                elements[i] = 1.0
            else:
                elements[i] = f(i + 1)
            idx[i] = i

        return (
            elements,
            idx,
            idx,
            elements,
            idx,
            idx,
            n,
        )  # I' @ I = I; also identity is full rank


def TP_pen(
    S_j: scp.sparse.csc_array,
    D_j: scp.sparse.csc_array,
    j: int,
    ks: list[int],
    constraint: Constraint | None,
    scale_pen: bool,
) -> tuple[list[float], list[int], list[int], list[float], list[int], list[int], int]:
    """Computes a tensor smooth penalty + root as defined in section 5.6 of Wood (2017) based on
    marginal penalty matrix ``S_j``.

    References:
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition \
        (2nd ed.).

    :param S_j: Marginal penalty matrix
    :type S_j: scp.sparse.csc_array
    :param D_j: Root of marginal penalty matrix
    :type D_j: scp.sparse.csc_array
    :param j: Index for current marginal
    :type j: int
    :param ks: List of number of basis functions of all marginals
    :type ks: list[int]
    :param constraint: Any constraint to absorb by the final penalty or None if no constraint is
        required
    :type constraint: Constraint | None
    :return: penalty data, penalty row indices, penalty column indices, root of penalty data,
        root of penalty row indices, root of penalty column indices, rank of penalty
    :rtype: tuple[list[float],list[int],list[int],list[float],list[int],list[int],int]
    """
    # Tensor smooth penalty - not including the reparameterization of Wood (2017) 5.6.2
    # but reflecting Eilers & Marx (2003) instead

    # Apply scaling as done by mgcv to improve numerical stability of tensor smooth terms.
    # see: https://github.com/cran/mgcv/blob/master/R/smooth.r#L824
    if scale_pen:
        eig, _ = scp.linalg.eigh(S_j.toarray())
        S_j /= eig[-1]
        D_j /= np.sqrt(eig[-1])

    if j == 0:
        S_TP = S_j
        D_TP = D_j
    else:
        S_TP = scp.sparse.identity(ks[0])
        D_TP = scp.sparse.identity(ks[0])
        # m_rank *= ks[0] # Modify rank of marginal - identities are full-rank.

    for i in range(1, len(ks)):
        if j == i:
            S_TP = scp.sparse.kron(S_TP, S_j, format="csc")
            D_TP = scp.sparse.kron(D_TP, D_j, format="csc")
        else:
            S_TP = scp.sparse.kron(S_TP, scp.sparse.identity(ks[i]), format="csc")
            D_TP = scp.sparse.kron(D_TP, scp.sparse.identity(ks[i]), format="csc")
            # m_rank *= ks[i]

    if constraint is not None:
        Z = constraint.Z
        if constraint.type == ConstType.QR:
            S_TP = Z.T @ S_TP @ Z
            D_TP = Z.T @ D_TP
            S_TP = scp.sparse.csc_array(S_TP)
            D_TP = scp.sparse.csc_array(D_TP)
        elif constraint.type == ConstType.DROP:
            S_TP = scp.sparse.csc_array(
                np.delete(np.delete(S_TP.toarray(), Z, axis=1), Z, axis=0)
            )
            D_TP = scp.sparse.csc_array(np.delete(D_TP.toarray(), Z, axis=0))
        elif constraint.type == ConstType.DIFF:
            D_TP = D_TP.toarray()
            # Correct for column differencing applied to X! See smoothCon help for mgcv (Wood, 2017)
            D_TP = np.diff(
                np.concatenate(
                    (D_TP[Z : D_TP.shape[0], :], D_TP[:Z, :]), axis=0  # noqa: E203
                ),
                axis=0,
            )
            D_TP = np.concatenate(
                (
                    D_TP[D_TP.shape[0] - Z :, :],  # noqa: E203
                    D_TP[: D_TP.shape[0] - Z, :],
                ),
                axis=0,
            )
            S_TP = D_TP @ D_TP.T
            S_TP = scp.sparse.csc_array(S_TP)
            D_TP = scp.sparse.csc_array(D_TP)

        # Check for full-rank marginal -> need new D_TP
        _, _, chol_cols = translate_sparse(D_TP)

        if (max(chol_cols) + 1) > D_TP.shape[0]:
            if constraint.type == ConstType.DIFF:
                raise ValueError(
                    (
                        "Cannot compute tensor smooth penalty for constraint.type=ConstType.DIFF. "
                        "Set a different constraint for the marginals or ensure that marginal "
                        "penalties have a non-trivial kernel!"
                    )
                )

            # Re-compute root
            eig, U = scp.linalg.eigh(S_TP.toarray())
            D_TP = scp.sparse.csc_array(
                U
                @ np.diag(
                    [
                        (
                            np.power(e, 0.5)
                            if e > np.power(sys.float_info.epsilon, 0.7)
                            else 0
                        )
                        for e in eig
                    ]
                )
            )

    pen_data, pen_rows, pen_cols = translate_sparse(S_TP)
    chol_data, chol_rows, chol_cols = translate_sparse(D_TP)
    return pen_data, pen_rows, pen_cols, chol_data, chol_rows, chol_cols
