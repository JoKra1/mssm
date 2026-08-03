import numpy as np
import scipy as scp
from dataclasses import dataclass
from .smooths import TP_basis_calc


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


class DiscreteModelMatrix:

    def __init__(self, dTerms: list[DiscreteTerm]):
        self.terms: list[DiscreteTerm] = dTerms
        self.preM: list[scp.sparse.csc_array] = []
        self.postM: list[scp.sparse.csc_array] = []
        self.exclude_cols: np.ndarray = []
        self.exclude_rows: np.ndarray = np.array([], dtype=np.int64)
        self.zero_cols: np.ndarray = []

        # Collect coefficients dropped/set to zero for predictions
        for dt in dTerms:
            if dt.zero_columns is not None:
                for c in dt.zero_columns:
                    self.zero_cols.append(c + dt.start_idx)
            if dt.exclude_columns is not None:
                for c in dt.exclude_columns:
                    self.exclude_cols.append(c + dt.start_idx)
        self.zero_cols = np.array(self.zero_cols, dtype=np.int64)
        self.exclude_cols = np.array(self.exclude_cols, dtype=np.int64)

        # Compute shape
        self.shape: tuple[int, int] = (
            self.terms[0].indices[0].shape[0],
            self.terms[-1].end_idx - len(self.exclude_cols),
        )

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

            if dt.exclude_columns is not None:
                cidx = np.arange(tmat.shape[1])
                tmat = tmat[:, ~np.isin(cidx, dt.exclude_columns)]
            if dt.zero_columns is not None:
                tmat = tmat[:, dt.zero_columns] = 0
            mat.append(tmat)

        mat = np.concatenate(mat, axis=1)

        # Set to zero
        # mat[:, self.zero_cols] = 0

        print(mat.shape)

        # Drop excluded ones
        # ridx = np.arange(mat.shape[0])
        # cidx = np.arange(mat.shape[1])
        # print(cidx[~np.isin(cidx, self.exclude_cols)])
        # mat = mat[:, ~np.isin(cidx, self.exclude_cols)]
        # mat = mat[~np.isin(ridx, self.exclude_rows), :]

        return mat
