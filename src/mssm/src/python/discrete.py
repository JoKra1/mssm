import numpy as np
import scipy as scp
from dataclasses import dataclass
import copy
from typing import Self
from .smooths import TP_basis_calc
import discrete


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
    by_cov: np.ndarray | None = None
    total_columns: int | None = None
    n_marginals: int = 1


class DiscreteModelMatrix:

    def __init__(self, dTerms: list[DiscreteTerm]):
        self.terms: list[DiscreteTerm] = copy.deepcopy(dTerms)
        self.preM: list[scp.sparse.csc_array] = []
        self.postM: list[scp.sparse.csc_array] = []
        self.exclude_rows: np.ndarray = np.array([], dtype=np.int64)
        self.max_slize_size: int = 10
        self.id: int = id(self)

        # Compute shape
        self.shape: tuple[int, int] = (
            self.terms[0].indices[0].shape[0],
            int(self.terms[-1].end_idx),
        )

    def __get_tensor_row(self, dti: int, ri: int) -> np.ndarray:
        dt = self.terms[dti]
        n_k = dt.total_columns
        if dt.Q is not None:
            n_k += 1

        cidx = np.arange(n_k)

        # Modified Algorithm A2 from Wood et al., 2017
        ps = [mat.shape[1] for mat in dt.unique_matrices]
        q = np.prod(ps)
        j = cidx
        j2 = j

        Xr = np.ones(n_k)

        for i in range(len(dt.unique_matrices)):
            q //= ps[i]
            ji = j2 // q
            j2 = j2 % q
            # print("2", j2)
            Xr *= dt.unique_matrices[i][dt.indices[i][ri], ji]

        return Xr

    def __get_cols(self, col: int) -> np.ndarray:

        if col < 0:
            col = self.shape[1] + col

        for dti, dt in enumerate(self.terms):
            if dt.start_idx <= col and col < dt.end_idx:

                xcol = col - dt.start_idx
                print(xcol)

                if dt.zero_columns is not None and xcol in dt.zero_columns:
                    return np.zeros(self.shape[0])

                # Compute column index
                cidx = np.arange(dt.total_columns)
                cidx = cidx[~np.isin(cidx, dt.exclude_columns)]
                cidx = cidx[xcol]  # Target column in original marginals

                if len(dt.unique_matrices) == 1:
                    # marginal case
                    return dt.unique_matrices[0][:, [cidx]][dt.indices[0]].flatten()
                else:
                    print("Extract tensor")
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

    def __getitem__(self, key) -> np.ndarray | Self:

        print(key)

        if len(key) == 1:
            # Row indexing
            rows = key
            cols = slice(None, None, None)
        elif len(key) == 2:
            # Handle 2D slices (including column extraction)
            rows = key[0]
            cols = key[1]
        else:
            raise ValueError(
                "Slices > 2D are not supported with class:`DiscreteModelMatrix`."
            )

        # Start by extracting columns.
        flatten_col = False
        if isinstance(cols, int):
            cols = [cols]
            flatten_col = True
        elif isinstance(cols, slice):
            start = cols.start if cols.start is not None else 0
            stop = cols.stop if cols.stop is not None else self.shape[1]
            step = cols.step if cols.step is not None else 1
            cols = list(range(start, stop, step))

        if len(cols) <= self.max_slize_size:
            # Explicitly evaluate slize

            rcols = np.array([self.__get_cols(col) for col in cols]).T

            if rcols.shape == (0,):
                rcols = rcols.reshape(self.shape[0], 0)

            if flatten_col:
                rcols = rcols.flatten()

            # Get entire columns?
            # full_row_check = rows == slice(None, None, None)
            # if isinstance(full_row_check, bool) and full_row_check:
            #    return rcols

            # Need rows
            ridx = np.arange(self.terms[0].indices[0].shape[0])
            ridx = ridx[~np.isin(ridx, self.exclude_rows)]
            ridx = ridx[rows]

            if len(rcols.shape) == 2:
                rcols = rcols[ridx, :]

                if rcols.shape[0] == 1:
                    return rcols.flatten()

                return rcols

            return rcols[ridx]

        # At this point need to return implicit slice
        newS = copy.deepcopy(self)

        # First drop columns
        cidx = np.arange(self.shape[1])
        print("tobedropped", cidx[~np.isin(cidx, cols)])
        newS.drop_columns(cidx[~np.isin(cidx, cols)])

        # Drop rows
        ridx = np.arange(self.terms[0].indices[0].shape[0])
        ridx_new = ridx[~np.isin(ridx, self.exclude_rows)][rows]
        print(ridx_new)
        new_exclude = ridx[~np.isin(ridx, ridx_new)]

        newS.exclude_rows = new_exclude
        newS.shape = (self.terms[0].indices[0].shape[0] - len(new_exclude), len(cols))
        return newS

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

            # by cov
            if dt.by_cov is not None:
                tmat *= dt.by_cov

            # Drop excluded ones
            if dt.exclude_columns is not None:
                cidx = np.arange(tmat.shape[1])
                tmat = tmat[:, ~np.isin(cidx, dt.exclude_columns)]

            # Handle zeroed columns
            if dt.zero_columns is not None:
                print(tmat.shape, dt)
                tmat[:, dt.zero_columns] = 0

            mat.append(tmat)

        mat = np.concatenate(mat, axis=1)

        # Drop excluded rows
        ridx = np.arange(mat.shape[0])
        mat = mat[~np.isin(ridx, self.exclude_rows), :]

        return mat

    def drop_columns(self, cols: list[int]):

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
                        print("before", xcol, dt.zero_columns)
                        if xcol in dt.zero_columns:
                            print("True drop is in zero")
                            dt.zero_columns = [
                                zc for zc in dt.zero_columns if zc != xcol
                            ]

                        dt.zero_columns = [
                            zc - 1 if zc > xcol else zc for zc in dt.zero_columns
                        ]

                        if len(dt.zero_columns) == 0:
                            dt.zero_columns = None
                        print("After", dt.zero_columns)

                    # And start-Stop indices for later terms
                    if dti < (len(self.terms) - 1):
                        for dtii in range(dti + 1, len(self.terms)):
                            dt2 = self.terms[dtii]
                            dt2.start_idx -= 1
                            dt2.end_idx -= 1

                    # Correct remaining columns
                    cols[(coli + 1) :] -= 1
                    dropped = True
                    break

            if dropped is False:
                raise ValueError(f"Could not drop column {ocols[coli]}.")
