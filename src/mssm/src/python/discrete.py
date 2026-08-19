import numpy as np
import scipy as scp
from dataclasses import dataclass
import copy
from typing import TypeVar
from .smooths import TP_basis_calc
from .matrix_solvers import map_csc_to_eigen
import discrete

TDiscreteMatrix = TypeVar("TDiscreteMatrix", bound="DiscreteModelMatrix")


@dataclass
class DiscreteTerm:
    """Discrete term storage for univariate and tensor (smooth) terms.

    A class:`DiscreteMatrix` is typically initialized from multiple of these.

    :ivar list[np.ndarray] | None unique_matrices: List of 2d matrices containing unique rows of
        term matrix per marginal. Initialized with ``None``.
    :ivar list[np.ndarray] | None indices: List of index arrays, indicating per observation and for
        each marginal ``i`` which row in ``unique_matrices[i]`` corresponds to the observation.
        Initialized with ``None``.
    :ivar int | None start_idx: Column index in the overal model matrix containing the first column
        of this term. Initialized with ``None``.
    :ivar int | None end_idx: Column index in the overal model matrix containing the last column
        of this term. Initialized with ``None``.
    :ivar list[int] | None zero_columns: List of indices indicating which columns of this term
        should be set to zero (e.g. ignored for the purpose of predictions). Initialized with
        ``None``.
    :ivar np.ndarray | None Q: Constraint matrix as 2D array for tensor terms (to be
        post-multiplied after computing the tensor product). Initialized with ``None``.
    """

    unique_matrices: list[np.ndarray] | None = None
    indices: list[np.ndarray] | None = None
    start_idx: int | None = None
    end_idx: int | None = None
    exclude_columns: list[int] | None = None
    zero_columns: list[int] | None = None
    Q: np.ndarray | None = None
    total_columns: int | None = None
    n_marginals: int = 1


def _XjTWXk(
    dtj: DiscreteTerm,
    dtk: DiscreteTerm,
    hasW: bool,
    W: list,
    alg,
) -> np.ndarray:
    # Extract all quantities needed
    psj = np.array([mat.shape[1] for mat in dtj.unique_matrices], dtype=np.int64)
    psk = np.array([mat.shape[1] for mat in dtk.unique_matrices], dtype=np.int64)
    qk = np.prod(psk)
    cidxj = np.arange(dtj.total_columns, dtype=np.int64)
    cidxj = cidxj[~np.isin(cidxj, dtj.exclude_columns)]
    cidxk = np.arange(dtk.total_columns, dtype=np.int64)
    cidxk = cidxk[~np.isin(cidxk, dtk.exclude_columns)]

    # Compute XjTWXk
    return alg(
        dtj.unique_matrices,
        dtk.unique_matrices,
        dtj.indices,
        dtk.indices,
        psj,
        psk,
        dtj.indices[0].shape[0],
        dtk.indices[0].shape[0],
        dtj.n_marginals,
        dtk.n_marginals,
        dtj.total_columns,
        dtk.total_columns,
        qk,
        cidxj,
        cidxk,
        dtj.Q is not None,
        dtk.Q is not None,
        dtj.Q if dtj.Q is not None else np.array([]),
        dtk.Q if dtk.Q is not None else np.array([]),
        hasW,
        *W,
    )


class DiscreteModelMatrix:

    def __init__(self, dTerms: list[DiscreteTerm]):
        self.terms: list[DiscreteTerm] = dTerms
        self.preM: scp.sparse.csc_array | np.ndarray | None = None
        self.postM: scp.sparse.csc_array | np.ndarray | None = None
        self.max_slize_size: int = 10
        self.id: int = id(self)
        self.return_sparse: bool = False

        # Compute shape
        self.shape: tuple[int, int] = (
            self.terms[0].indices[0].shape[0],
            int(self.terms[-1].end_idx),
        )

        self._T: bool = False

    def __prepareW(
        self, otherpreM: scp.sparse.csc_array | np.ndarray | None
    ) -> tuple[list | np.ndarray, bool, bool]:
        """_summary_

        :param otherpreM: _description_
        :type otherpreM: scp.sparse.csc_array | np.ndarray | None
        :return: _description_
        :rtype: tuple[list | np.ndarray, bool, bool]
        """
        # Check for W
        W: scp.sparse.csc_array | np.ndarray | None = None
        if self.postM is not None:
            W = self.postM
        if otherpreM is not None:
            if W is not None:
                W = W @ otherpreM
            else:
                W = otherpreM

        # Check if W is sparse
        W_is_sparse = False
        if isinstance(W, scp.sparse.sparray) or isinstance(W, scp.sparse.spmatrix):
            W = W.tocsc()
            W = map_csc_to_eigen(W)
            W_is_sparse = True

        hasW: bool = W is not None
        if W is None:
            W = np.array([])

        if not W_is_sparse:
            W = [W]

        return W, hasW, W_is_sparse

    def __XTWZ(self, other: TDiscreteMatrix) -> np.ndarray:
        """_summary_

        :param other: _description_
        :type other: DiscreteModelMatrix
        :return: _description_
        :rtype: np.ndarray
        """
        # Check for W
        W, hasW, W_is_sparse = self.__prepareW(other.preM)

        alg = discrete.XTWXS if W_is_sparse else discrete.XTWXD

        # Check dimensions:
        n_row = 0
        for dt in self.terms:
            if dt.end_idx <= dt.start_idx:
                # Skip terms removed completely
                continue

            n_row += dt.end_idx - dt.start_idx

        n_col = 0
        for dt in other.terms:
            if dt.end_idx <= dt.start_idx:
                # Skip terms removed completely
                continue

            n_col += dt.end_idx - dt.start_idx

        rows = np.arange(n_row)
        cols = np.arange(n_col)

        XTWZ = np.zeros((n_row, n_col))

        for dtji, dtj in enumerate(self.terms):

            for dtki, dtk in enumerate(other.terms):

                if (dtj.end_idx <= dtj.start_idx) or (dtk.end_idx <= dtk.start_idx):
                    continue

                XTWZ[
                    np.ix_(
                        rows[dtj.start_idx : dtj.end_idx],  # noqa: E203
                        cols[dtk.start_idx : dtk.end_idx],  # noqa: E203
                    )
                ] = _XjTWXk(dtj, dtk, hasW, W, alg)

        return XTWZ

    def __XTWX(self, otherpreM: scp.sparse.csc_array | np.ndarray | None) -> np.ndarray:
        """_summary_

        :param otherpreM: _description_
        :type otherpreM: scp.sparse.csc_array | np.ndarray | None
        :return: _description_
        :rtype: np.ndarray
        """

        # Check for W
        W, hasW, W_is_sparse = self.__prepareW(otherpreM)

        alg = discrete.XTWXS if W_is_sparse else discrete.XTWXD

        # Check dimensions:
        n_col = 0
        for dt in self.terms:
            if dt.end_idx <= dt.start_idx:
                # Skip terms removed completely
                continue

            n_col += dt.end_idx - dt.start_idx
        cols = np.arange(n_col)

        XTWX = np.zeros((n_col, n_col))
        for dtji, dtj in enumerate(self.terms):

            for dtki in range(dtji, len(self.terms)):
                dtk = self.terms[dtki]

                if (dtj.end_idx <= dtj.start_idx) or (dtk.end_idx <= dtk.start_idx):
                    continue

                XTWX[
                    np.ix_(
                        cols[dtj.start_idx : dtj.end_idx],  # noqa: E203
                        cols[dtk.start_idx : dtk.end_idx],  # noqa: E203
                    )
                ] = _XjTWXk(dtj, dtk, hasW, W, alg)

                if dtji != dtki:
                    XTWX[
                        np.ix_(
                            cols[dtk.start_idx : dtk.end_idx],  # noqa: E203
                            cols[dtj.start_idx : dtj.end_idx],  # noqa: E203
                        )
                    ] = XTWX[
                        np.ix_(
                            cols[dtj.start_idx : dtj.end_idx],  # noqa: E203
                            cols[dtk.start_idx : dtk.end_idx],  # noqa: E203
                        )
                    ].T

        return XTWX

    def __XTy(self, y: np.ndarray) -> np.ndarray:
        """_summary_

        :param y: _description_
        :type y: np.ndarray | scp.sparse.sparray
        :return: _description_
        :rtype: np.ndarray
        """

        res = []
        for dti, dt in enumerate(self.terms):
            if len(dt.unique_matrices) == 1:
                # Algorithm A3 from Wood et al., 2017

                cidx = np.arange(dt.unique_matrices[0].shape[1])
                cidx = cidx[~np.isin(cidx, dt.exclude_columns)]

                res.append(
                    discrete.A3(
                        dt.unique_matrices[0], y.flatten(), dt.indices[0], cidx
                    ).reshape(-1, 1)
                )
            else:
                # Algorithm A4 from Wood et al., 2017
                n_k = dt.total_columns
                cidx = np.arange(n_k)
                cidx = cidx[~np.isin(cidx, dt.exclude_columns)]

                # print("A4 n_k", n_k, dt.Q is not None)
                v = discrete.A4(
                    dt.unique_matrices,
                    dt.indices,
                    np.array(
                        [mat.shape[1] for mat in dt.unique_matrices], dtype=np.int64
                    ),
                    dt.indices[0].shape[0],
                    dt.n_marginals,
                    n_k,
                    cidx,
                    y,
                    dt.Q is not None,
                    dt.Q if dt.Q is not None else np.array([], dtype=np.float64),
                ).reshape(-1, 1)

                res.append(v)

        return np.concatenate(res, axis=0)

    def __Xb(self, b: np.ndarray) -> np.ndarray:
        """_summary_

        :param b: _description_
        :type b: np.ndarray
        :return: _description_
        :rtype: np.ndarray
        """

        res = np.zeros((self.terms[0].indices[0].shape[0], 1))
        for dti, dt in enumerate(self.terms):
            if dt.end_idx <= dt.start_idx:
                # Skip terms removed completely
                continue

            # Get rows in b associated with dt
            bt = b[dt.start_idx : dt.end_idx, 0]  # noqa: E203
            if len(dt.unique_matrices) == 1:
                # Algorithm A5 from Wood et al., 2017
                cidx = np.arange(dt.unique_matrices[0].shape[1])
                cidx = cidx[~np.isin(cidx, dt.exclude_columns)]

                res += discrete.A5(
                    dt.unique_matrices[0], bt, dt.indices[0], cidx
                ).reshape(-1, 1)

            else:
                # Algorithm A6 from Wood et al., 2017

                # Embed bt in full dimensions with zeros for dropped cols
                cidx = np.arange(dt.total_columns)
                cidx = cidx[~np.isin(cidx, dt.exclude_columns)]
                bte = np.zeros(dt.total_columns)
                bte[cidx] = bt

                if dt.Q is not None:
                    bte = (dt.Q @ bte.reshape(-1, 1)).flatten()

                # Prepare matrices and indices to implicitly represent A
                umatsA = [mat for mat in dt.unique_matrices[:-1]]
                indicesA = [idx for idx in dt.indices[:-1]]
                psA = [mat.shape[1] for mat in umatsA]
                qA = np.prod(psA)

                # Create C + index
                indexC = dt.indices[-1]
                B = np.reshape(bte, (dt.unique_matrices[-1].shape[1], qA), order="F")
                C = np.asfortranarray(dt.unique_matrices[-1] @ B)

                res += discrete.A6(
                    umatsA,
                    indicesA,
                    indexC,
                    np.array(psA, dtype=np.int64),
                    qA,
                    dt.indices[0].shape[0],
                    dt.n_marginals - 1,
                    C,
                ).reshape(-1, 1)

        return res

    def __get_row(self, row: int) -> np.ndarray:
        """Returns row of model matrix ``X`` (i.e., not transposed) **after** excluding columns.

        Columns that are zeroed have zero values in place.

        :param row: _description_
        :type row: int
        :return: _description_
        :rtype: np.ndarray
        """
        c_shape = self.shape
        if self._T:
            # Flip keys
            c_shape = np.flip(c_shape)

        if row < 0:
            row = c_shape[0] + row

        n_c = 0
        for dt in self.terms:
            n_c += dt.total_columns

        Xr = []

        for dt in self.terms:

            if len(dt.unique_matrices) == 1:
                Xrj = dt.unique_matrices[0][dt.indices[0][row], :].flatten()
            else:
                n_k = dt.total_columns
                if dt.Q is not None:
                    n_k += 1  # + 1 because of Q

                j = np.arange(n_k)
                ps = [mat.shape[1] for mat in dt.unique_matrices]
                q = np.prod(ps)

                Xrj = discrete.Xrtensor(
                    dt.unique_matrices,
                    np.array([idx[row] for idx in dt.indices], dtype=np.int64),
                    np.array(ps, dtype=np.int64),
                    q,
                    dt.n_marginals,
                    j,
                    n_k,
                )

                if dt.Q is not None:
                    Xrj = (Xrj.reshape(1, -1) @ dt.Q).flatten()

            cidx = np.arange(dt.total_columns)
            if dt.exclude_columns is not None:
                # Exclude columns
                cidx = cidx[~np.isin(cidx, dt.exclude_columns)]
                Xrj = Xrj[cidx]

            if dt.zero_columns is not None:
                # zero columns
                Xrj[dt.zero_columns] = 0

            Xr.extend(Xrj)

        return np.array(Xr)

    def __get_col(self, col: int) -> np.ndarray:
        """Returns column of model matrix ``X`` (i.e., not transposed).

        A column of zeros is returned if the column has been zeroed.

        :param col: _description_
        :type col: int
        :return: _description_
        :rtype: np.ndarray
        """

        c_shape = self.shape
        if self._T:
            # Flip keys
            c_shape = np.flip(c_shape)

        if col < 0:
            col = c_shape[1] + col

        for dt in self.terms:
            if dt.start_idx <= col and col < dt.end_idx:

                xcol = col - dt.start_idx
                # print(xcol)

                if dt.zero_columns is not None and xcol in dt.zero_columns:
                    return np.zeros(dt.indices[0].shape[0])

                # Compute column index
                cidx = np.arange(dt.total_columns)
                cidx = cidx[~np.isin(cidx, dt.exclude_columns)]
                cidx = cidx[xcol]  # Target column in original marginals

                if len(dt.unique_matrices) == 1:
                    # Algorithm A1 from Wood et al., 2017
                    return discrete.A1(dt.unique_matrices[0], dt.indices[0], cidx)
                else:
                    # print("Extract tensor")
                    if dt.Q is None:
                        # Algorithm A2 from Wood et al., 2017
                        ps = [mat.shape[1] for mat in dt.unique_matrices]
                        q = np.prod(ps)
                        j = cidx

                        Xj = discrete.A2(
                            dt.unique_matrices,
                            dt.indices,
                            np.array(ps, dtype=np.int64),
                            q,
                            dt.indices[0].shape[0],
                            dt.n_marginals,
                            j,
                        )

                    else:
                        # Modified Algorithm A2 from Wood et al., 2017
                        n_k = dt.total_columns + 1  # + 1 because of Q
                        j = np.arange(n_k)
                        ps = [mat.shape[1] for mat in dt.unique_matrices]
                        q = np.prod(ps)

                        Xj = discrete.A2Q(
                            dt.unique_matrices,
                            dt.indices,
                            np.array(ps, dtype=np.int64),
                            q,
                            dt.indices[0].shape[0],
                            dt.n_marginals,
                            j,
                            n_k,
                            dt.Q[:, cidx],
                        )

                    return Xj

    def __getitem__(self, key) -> np.ndarray | scp.sparse.sparray | TDiscreteMatrix:

        # print(key)
        advanced = False
        if isinstance(key, int) or isinstance(key, np.integer) or len(key) == 1:
            # Row indexing
            rows = key
            cols = slice(None, None, None)
        elif len(key) == 2:
            # Handle 2D slices (including column extraction)
            rows = key[0]
            cols = key[1]
            if (isinstance(rows, list) or isinstance(rows, np.ndarray)) and (
                isinstance(cols, list) or isinstance(cols, np.ndarray)
            ):
                advanced = True
        else:
            raise ValueError(
                "Slices > 2D are not supported with class:`DiscreteModelMatrix`."
            )

        c_shape = self.shape
        if self._T:
            # Flip keys
            trows = copy.deepcopy(rows)
            rows = cols
            cols = trows
            c_shape = np.flip(c_shape)

        # Start by extracting columns.
        flatten_col = False
        if isinstance(cols, int) or isinstance(cols, np.integer):
            cols = [cols]
            flatten_col = True
        elif isinstance(cols, slice):
            start = cols.start if cols.start is not None else 0
            stop = cols.stop if cols.stop is not None else c_shape[1]
            step = cols.step if cols.step is not None else 1
            cols = list(range(start, stop, step))

        # Compute new row indices after extraction
        ridx = np.arange(self.terms[0].indices[0].shape[0])
        ridx_new = ridx[rows]

        if (self.preM is not None) and (self.postM is not None):
            # Check postM size here

            check_shape = self.postM[:, rows].shape[1] if self._T else len(cols)

            if check_shape <= self.max_slize_size:
                # Return self.preM @ (X.T @ self.postM) if self._T and
                # self.preM @ (X @ self.postM) is self._T is False
                print("A3-A6")
                pass
        elif self.preM is not None:
            # Check X/X.T size here
            check_shape = len(ridx_new) if self.T else len(cols)
            # print("self.preM", check_shape)

            if check_shape <= self.max_slize_size:
                # Return self.preM[cols, :] @ self.X.T[:,rows] if self._T
                # else self.preM[rows, :] @ self.X[:,cols]
                if self._T:
                    rrows = np.array([self.__get_row(r) for r in ridx_new]).T
                    # print("self.preM pre check Xr shape", rrows.shape)
                    res = self.preM[cols, :] @ rrows
                    return scp.sparse.csc_array(res) if self.return_sparse else res
                else:
                    rcols = np.array([self.__get_col(col) for col in cols]).T
                    if rcols.shape == (0,):
                        rcols = rcols.reshape(c_shape[0], 0)
                    res = self.preM[rows, :] @ rcols
                    return scp.sparse.csc_array(res) if self.return_sparse else res

        elif self.postM is not None:

            check_shape = len(cols) if self.T else len(ridx_new)
            # print("self.postM", check_shape)

            if check_shape <= self.max_slize_size:
                # Return X.T[cols,:] @ self.postM[:, rows] if self._T
                # else Return X[rows,:] @ self.postM[:, cols]
                if self._T:
                    rcols = np.array([self.__get_col(col) for col in cols]).T
                    if rcols.shape == (0,):
                        rcols = rcols.reshape(c_shape[0], 0)
                    res = rcols.T @ self.postM[:, rows]
                    return scp.sparse.csc_array(res) if self.return_sparse else res

                else:
                    rrows = np.array([self.__get_row(r) for r in ridx_new])
                    # print("self.postM pre check Xr shape", rrows.shape)

                    res = rrows @ self.postM[:, cols]
                    return scp.sparse.csc_array(res) if self.return_sparse else res

        elif len(cols) <= self.max_slize_size:
            # Explicitly evaluate slize
            # ToDo: Handle postM and PreM
            # postM is easy if self._T and preM is easy if not self._T
            # for the hard cases we need a get_row method ideally.
            rcols = np.array([self.__get_col(col) for col in cols]).T

            if rcols.shape == (0,):
                rcols = rcols.reshape(c_shape[0], 0)

            if flatten_col:
                rcols = rcols.flatten()

            # Get entire columns?
            # full_row_check = rows == slice(None, None, None)
            # if isinstance(full_row_check, bool) and full_row_check:
            #    return rcols

            # Need to account for new rows
            # print(rcols)
            if len(rcols.shape) == 2:
                if advanced:
                    rcols = rcols[ridx_new, np.arange(len(cols))]
                else:
                    rcols = rcols[ridx_new, :]

                res = rcols.T if self._T else rcols
                if self.return_sparse:
                    if len(res.shape) == 1:
                        return scp.sparse.coo_array(res)
                    return scp.sparse.csc_array(res)
                return res

            res = rcols[ridx_new]
            return scp.sparse.coo_array(res) if self.return_sparse else res

        # At this point need to return implicit slice
        newS = copy.deepcopy(self)
        newS.id = id(newS)  # Update id

        # First need to check for preM and postM - remember cols and rows are flipped if self._T
        if (self.preM is not None) and (self.postM is not None):
            # No need to change anything to data in newS, can drop from
            # preM and postM only.
            if self._T:
                newS.preM = self.preM[cols, :]
                newS.postM = self.postM[:, rows]
            else:
                newS.preM = self.preM[rows, :]
                newS.postM = self.postM[:, cols]

            n_rows = newS.preM.shape[0]
            n_cols = newS.postM.shape[1]

        elif self.preM is not None:
            # Drop rows from preM and cols from self
            if self._T:
                newS.preM = self.preM[cols, :]

                # Need to drop rows instead -> columns after transpose
                newS.drop_rows(ridx[~np.isin(ridx, ridx_new)])
                n_cols = newS.terms[0].indices[0].shape[0]
            else:
                newS.preM = self.preM[rows, :]

                # And drop columns
                cidx = np.arange(c_shape[1])
                newS.drop_columns(cidx[~np.isin(cidx, cols)])
                n_cols = len(cols)

            n_rows = newS.preM.shape[0]

        elif self.postM is not None:
            # Drop rows from self and cols from postM
            if self._T:
                newS.postM = self.postM[:, rows]

                # Need to drop columns instead -> rows after transpose
                cidx = np.arange(c_shape[1])
                newS.drop_columns(cidx[~np.isin(cidx, cols)])
                n_rows = len(cols)

            else:
                newS.postM = self.postM[:, cols]

                # And drop rows
                newS.drop_rows(ridx[~np.isin(ridx, ridx_new)])
                n_rows = newS.terms[0].indices[0].shape[0]

            n_cols = newS.postM.shape[1]

        else:
            # Drop both, rows and columns, from self

            # First cols
            cidx = np.arange(c_shape[1])
            newS.drop_columns(cidx[~np.isin(cidx, cols)])
            n_cols = len(cols)

            # And then rows
            newS.drop_rows(ridx[~np.isin(ridx, ridx_new)])
            n_rows = newS.terms[0].indices[0].shape[0]

            if self._T:
                # Flip dims
                t_cols = n_cols
                n_cols = n_rows
                n_rows = t_cols

        # Compute correct dimensions
        newS.shape = (n_rows, n_cols)

        return newS

    def __matmul__(
        self,
        other: np.ndarray | scp.sparse.sparray | scp.sparse.spmatrix | TDiscreteMatrix,
    ) -> np.ndarray | scp.sparse.sparray | TDiscreteMatrix:
        # print("__matmul__", other.shape, type(other), self.id)
        flatten = False
        if len(other.shape) == 1:
            other = other.reshape(-1, 1)
            flatten = True
        if (
            isinstance(other, np.ndarray)
            or isinstance(other, scp.sparse.sparray)
            or isinstance(other, scp.sparse.spmatrix)
            or isinstance(other, DiscreteModelMatrix)
        ):

            if self.shape[1] != other.shape[0]:
                raise ArithmeticError(
                    (
                        f"Dimension of self is ({self.shape[0]},{self.shape[1]}), "
                        f"but other is of dimension ({other.shape[0]},{other.shape[1]})"
                    )
                )

            # Explicit evaluation of return
            if (
                other.shape[1] <= self.max_slize_size
                and isinstance(other, DiscreteModelMatrix) is False
            ):

                rsparse = self.return_sparse and isinstance(other, scp.sparse.sparray)
                if isinstance(other, scp.sparse.sparray):
                    other = other.toarray()

                if self._T:
                    # Handle transpose case
                    if other.shape[1] == 1:
                        XTy = self.__XTy(
                            self.postM @ other if self.postM is not None else other
                        )

                        if flatten:
                            XTy = XTy.flatten()

                        res = self.preM @ XTy if self.preM is not None else XTy
                        if rsparse:
                            if flatten:
                                return scp.sparse.coo_array(res)
                            return scp.sparse.csc_array(res)
                        return res

                    XTY = []
                    Y = self.postM @ other if self.postM is not None else other
                    for ci in range(other.shape[1]):
                        XTY.append(self.__XTy(Y[:, [ci]]).flatten())

                    XTY = np.array(XTY).T
                    # print("XTY shape", XTY.shape)
                    res = self.preM @ XTY if self.preM is not None else XTY
                    return scp.sparse.csc_array(res) if rsparse else res

                # Handle un-transposed case
                if other.shape[1] == 1:
                    Xb = self.__Xb(
                        self.postM @ other if self.postM is not None else other
                    )

                    if flatten:
                        Xb = Xb.flatten()

                    res = self.preM @ Xb if self.preM is not None else Xb
                    if rsparse:
                        if flatten:
                            return scp.sparse.coo_array(res)
                        return scp.sparse.csc_array(res)
                    return res

                XB = []
                B = self.postM @ other if self.postM is not None else other
                for ci in range(other.shape[1]):
                    XB.append(self.__Xb(B[:, [ci]]).flatten())

                XB = np.array(XB).T
                # print("XB shape", XB.shape)
                res = self.preM @ XB if self.preM is not None else XB
                return scp.sparse.csc_array(res) if rsparse else res

            elif (
                isinstance(other, DiscreteModelMatrix)
                and self.id == other.id
                and self._T
                and other._T is False
            ):
                XTWX = self.__XTWX(other.preM)
                # print(self.return_sparse)
                if self.preM is not None:
                    XTWX = self.preM @ XTWX
                if other.postM is not None:
                    XTWX = XTWX @ other.postM

                if self.return_sparse:
                    XTWX = scp.sparse.csc_array(XTWX)

                return XTWX

            elif (
                isinstance(other, DiscreteModelMatrix) and self._T and other._T is False
            ):
                XTWZ = self.__XTWZ(other)
                # print(self.return_sparse, other.return_sparse)

                if self.preM is not None:
                    XTWZ = self.preM @ XTWZ
                if other.postM is not None:
                    XTWZ = XTWZ @ other.postM

                if self.return_sparse and other.return_sparse:
                    XTWZ = scp.sparse.csc_array(XTWZ)

                return XTWZ

            elif isinstance(other, DiscreteModelMatrix) is False:
                # Store matrix in postM and update shape and return type
                newS = copy.deepcopy(self)
                newS.return_sparse = (
                    isinstance(other, scp.sparse.sparray)
                    or isinstance(other, scp.sparse.spmatrix)
                ) and self.return_sparse
                if self.postM is None:
                    newS.postM = other
                else:
                    newS.postM = self.postM @ other
                newS.shape = (self.shape[0], other.shape[1])
                return newS
            else:
                raise NotImplementedError("Requested Product is not implemented.")

        else:
            raise NotImplementedError(
                (
                    "Matrix multiplication is only implemented for scipy sparse,"
                    " numpy arrays and `DiscreteModelMatrix` objects."
                )
            )

    def __rmatmul__(
        self, other: np.ndarray | scp.sparse.sparray | scp.sparse.spmatrix
    ) -> np.ndarray | scp.sparse.sparray | TDiscreteMatrix:
        # print("__rmatmul__", other.shape, type(other), self.id)

        flatten = False
        if len(other.shape) == 1:
            other = other.reshape(1, -1)
            flatten = True

        if (
            isinstance(other, np.ndarray)
            or isinstance(other, scp.sparse.sparray)
            or isinstance(other, scp.sparse.spmatrix)
        ):
            if other.shape[1] != self.shape[0]:
                raise ArithmeticError(
                    (
                        f"Dimension of self is ({self.shape[0]},{self.shape[1]}), "
                        f"but other is of dimension ({other.shape[0]},{other.shape[1]})"
                    )
                )

            if other.shape[0] <= self.max_slize_size:
                resT = self.transpose() @ (other.flatten() if flatten else other.T)

                if flatten:
                    return resT
                return resT.T

            # Store matrix in preM and update shape and return type
            newS = copy.deepcopy(self)
            newS.return_sparse = (
                isinstance(other, scp.sparse.sparray)
                or isinstance(other, scp.sparse.spmatrix)
            ) and self.return_sparse
            if self.preM is None:
                newS.preM = other
            else:
                newS.preM = other @ self.preM
            newS.shape = (other.shape[0], self.shape[1])
            return newS

        else:
            raise NotImplementedError(
                (
                    "Reflected Matrix multiplication is only implemented for scipy sparse and"
                    " numpy arrays."
                )
            )

    __array_priority__ = 10000

    def __mul__(self, other: float | int | np.ndarray) -> TDiscreteMatrix:
        # print("__mul__", other, type(other), self.id)
        if (
            isinstance(other, float)
            or isinstance(other, int)
            or isinstance(other, np.integer)
        ):
            newS = copy.deepcopy(self)
            for dti, dt in enumerate(newS.terms):
                for mi in range(len(dt.unique_matrices)):
                    dt.unique_matrices[mi] *= other
            return newS
        elif isinstance(other, np.ndarray):
            # Have to evaluate.. Not very efficient.
            return self.toarray() * other

        else:
            raise NotImplementedError(
                (
                    "Element-wise multiplication is only supported "
                    "for float, int, and np.array arguments."
                )
            )

    def __rmul__(self, other):
        # print("__rmul__", other, type(other), self.id)
        return self.__mul__(other)

    def is_transposed(self) -> bool:
        return self.__T

    @property
    def T(self) -> TDiscreteMatrix:
        """Returns transpose. Data is not copied.

        :return: _description_
        :rtype: TDiscreteMatrix
        """
        return self.transpose()

    def transpose(self) -> TDiscreteMatrix:
        """Returns transpose. Data is not copied.

        :return: _description_
        :rtype: TDiscreteMatrix
        """

        # Trivial case when self is vector
        if self.shape[0] == 1 or self.shape[1] == 0:
            return self

        newS = DiscreteModelMatrix(self.terms)
        newS.return_sparse = self.return_sparse
        newS.max_slize_size = self.max_slize_size
        newS.id = self.id
        newS.postM = None if self.preM is None else self.preM.T
        newS.preM = None if self.postM is None else self.postM.T

        newS._T = not self._T  # Update flag
        newS.shape = (self.shape[1], self.shape[0])

        return newS

    def copy(self) -> TDiscreteMatrix:
        """Return copy of self.

        :return: _description_
        :rtype: TDiscreteMatrix
        """
        return copy.deepcopy(self)

    def toarray(self) -> np.ndarray:
        """Represents discrete matrix explicitly as a 2D numpy array."""

        mat = []
        for dt in self.terms:
            tmat = dt.unique_matrices[0][dt.indices[0]]

            # tensor
            if len(dt.unique_matrices) > 1:
                for midx in range(1, len(dt.unique_matrices)):
                    tmat = TP_basis_calc(
                        tmat, dt.unique_matrices[midx][dt.indices[midx]]
                    )

                # Constraint
                if dt.Q is not None:
                    tmat = tmat @ dt.Q

            # Drop excluded ones
            if dt.exclude_columns is not None:
                cidx = np.arange(tmat.shape[1])
                tmat = tmat[:, ~np.isin(cidx, dt.exclude_columns)]

            # Handle zeroed columns
            if dt.zero_columns is not None:
                # print(tmat.shape, dt)
                tmat[:, dt.zero_columns] = 0

            mat.append(tmat)

        mat = np.concatenate(mat, axis=1)

        if self._T:
            mat = mat.T

        if self.postM is not None:
            mat = mat @ self.postM

        if self.preM is not None:
            mat = self.preM @ mat

        return mat

    def eval(self) -> np.ndarray | scp.sparse.csc_array:
        """Explicitly returns matrix either as np.array or as scp.sparse.csc_array
        depending on ``self.return_sparse``

        :return: _description_
        :rtype: np.ndarray | scp.sparse.csc_array
        """

        return (
            scp.sparse.csc_array(self.toarray())
            if self.return_sparse
            else self.toarray()
        )

    def drop_rows(self, rows: list[int]) -> None:

        ridx = np.arange(self.terms[0].indices[0].shape[0])
        new_ridx = ridx[~np.isin(ridx, rows)]

        for dt in self.terms:
            for idx in range(len(dt.indices)):
                dt.indices[idx] = dt.indices[idx][new_ridx]

    def drop_columns(self, cols: list[int]) -> None:

        # Sort cols
        cols = np.unique(cols)
        ocols = copy.deepcopy(cols)  # For error

        for coli in range(len(cols)):
            col = cols[coli]

            dropped = False
            for dti in range(len(self.terms)):
                dt = self.terms[dti]
                if (
                    dt.start_idx <= col
                    and col < dt.end_idx
                    and dt.end_idx > dt.start_idx
                ):
                    # col is associated with current dt - handle drop
                    xcol = col - dt.start_idx
                    cidx = np.arange(dt.total_columns)
                    cidx = cidx[~np.isin(cidx, dt.exclude_columns)]
                    drop = cidx[xcol]
                    if dt.exclude_columns is not None:
                        dt.exclude_columns = np.sort([*dt.exclude_columns, drop])
                    else:
                        dt.exclude_columns = np.array([drop])

                    dt.end_idx -= 1

                    # Adjust zeroed columns
                    if dt.zero_columns is not None:
                        # print("before", xcol, dt.zero_columns)
                        if xcol in dt.zero_columns:
                            # print("True drop is in zero")
                            dt.zero_columns = [
                                zc for zc in dt.zero_columns if zc != xcol
                            ]

                        dt.zero_columns = [
                            zc - 1 if zc > xcol else zc for zc in dt.zero_columns
                        ]

                        if len(dt.zero_columns) == 0:
                            dt.zero_columns = None
                        # print("After", dt.zero_columns)

                    # And start-Stop indices for later terms
                    if dti < (len(self.terms) - 1):
                        for dtii in range(dti + 1, len(self.terms)):
                            dt2 = self.terms[dtii]
                            dt2.start_idx -= 1
                            dt2.end_idx -= 1

                    # Correct remaining columns
                    cols[(coli + 1) :] -= 1  # noqa: E203
                    dropped = True
                    break

            if dropped is False:
                raise ValueError(f"Could not drop column {ocols[coli]}.")
