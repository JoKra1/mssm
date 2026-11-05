import numpy as np
import scipy as scp
from dataclasses import dataclass
import copy
from typing import TypeVar
from collections.abc import Callable
from .smooths import TP_basis_calc
from .matrix_solvers import map_csc_to_eigen
import discrete
import uuid

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
    :ivar list[int] | None exclude_columns: List of indices indicating which columns of this term
            should be considered dropped (e.g., after indexing). Initialized with ``None``.
    :ivar list[int] | None zero_columns: List of indices indicating which columns of this term
        should be set to zero (e.g. ignored for the purpose of predictions). Initialized with
        ``None``.
    :ivar np.ndarray | None Q: Constraint matrix as 2D array for tensor terms (to be
        post-multiplied after computing the tensor product). Initialized with ``None``.
    :ivar int total_columns: Total columns of this term **after absorbing any constraints**.
        Initialized with 1.
    :ivar int n_marginals: Number of marginals this term has. Initialized with 1.
    :ivar np.ndarray | None nonzero_rows: Flattened array with indices of nonzero rows of this term.
        Initialized with ``None``.
    """

    unique_matrices: list[np.ndarray] | None = None
    indices: list[np.ndarray] | None = None
    start_idx: int | None = None
    end_idx: int | None = None
    exclude_columns: list[int] | None = None
    zero_columns: list[int] | None = None
    Q: np.ndarray | None = None
    total_columns: int = 1
    n_marginals: int = 1
    nonzero_rows: np.ndarray | None = None


def _convert_index(
    idx: int | list | np.ndarray | slice | np.integer, size: int
) -> int | np.ndarray | None:
    """Takes index ``idx`` targeting an axis of dimension ``size`` and converts it into an
    positive int, a flattened array positive ints, or ``None`` (to select everything from the axis).

    **Note**: negative indices, slices, and boolean masks handled as numpy would handle them.

    :param idx: Index to be converted
    :type idx: int | list | np.ndarray | slice | np.integer
    :param size: DImension of axis indexed by ``idx``
    :type size: int
    :raises IndexError: If ``idx`` is out of bounds or has multiple dimensions.
    :return: The converted index
    :rtype: int | np.ndarray | None
    """
    if isinstance(idx, int) or isinstance(idx, np.integer):
        # Catch out of bounds first, note:
        # -size maps back to zero so can keep
        # +size maps back to largest index + 1 so already needs to go
        if idx < -size or idx >= size:
            raise IndexError(f"Index {idx} is not within size {size}.")
        # Now safe to wrap negatives
        if idx < 0:
            return size + idx
        return idx

    if isinstance(idx, slice):
        # Handle : slices here, any other slice
        # can be handled with final return
        if idx == slice(None, None, None):
            return None
    else:
        # list or array can both be handled as array. Need to check
        # for > 1 dimension here!
        idx = np.asarray(idx)
        if idx.size == 0:
            # Cast empty lists to int
            idx = idx.astype(np.int64)

        if idx.ndim != 1:
            raise IndexError(
                "Class:`DiscreteModelMatrix` only supports single dimension indices."
            )

    # Evaluate list, array, or non-full slice
    return np.arange(size)[idx]


def _is_symmetric(W: scp.sparse.sparray | scp.sparse.spmatrix | np.ndarray) -> bool:
    """Function to check whether square matrix ``W`` is symmetric or not.

    :param W: Square matrix, can be sparse or dense
    :type W: scp.sparse.sparray | scp.sparse.spmatrix | np.ndarray
    :return: Bool, indicating whether ``W`` is symmetric or not.
    :rtype: bool
    """
    # Root of machine tolerance for both dense and sparse checks
    rtol = np.sqrt(np.finfo(float).eps)

    if isinstance(W, np.ndarray):
        # Take scipy's check
        return scp.linalg.issymmetric(W, rtol=rtol)

    # For sparse we need to do it manually
    diff = W - W.T

    if diff.nnz == 0 or np.abs(diff.data).max() <= rtol * np.abs(W.data).max():
        return True

    return False


def _eval_discrete_slice(
    res: (
        TDiscreteMatrix | np.ndarray | scp.sparse.sparray | scp.sparse.spmatrix | float
    ),
    return_sparse: bool,
) -> np.ndarray | scp.sparse.sparray | float:
    """Casts a DiscreteModelMatrix slice to a scipy sparse csc_array (coo_array if res is a vector)
    if ``return_sparse is True`` and the slice was evaluated explicitly. Otherwise returns directly.

    Note, that floats are always returned directly.

    :param res: The explicitly evaluated result
    :type res: np.ndarray | scp.sparse.sparray | scp.sparse.spmatrix | float
    :return: ``res``, converted to a sparse array if ``self.return_sparse``
    :rtype: TDiscreteMatrix | np.ndarray | scp.sparse.sparray | float
    """

    if isinstance(res, DiscreteModelMatrix):
        res.return_sparse = return_sparse
        return res

    if np.ndim(res) == 0 or not return_sparse:
        return res

    if res.ndim == 1:
        if isinstance(res, scp.sparse.coo_array):
            return res
        return scp.sparse.coo_array(res)
    return scp.sparse.csc_array(res)


def _nonzeroXjTWXk(
    dtjs: list[DiscreteTerm],
    dtks: list[DiscreteTerm],
    W: scp.sparse.sparray | scp.sparse.spmatrix | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Function that determines for which combinations of ``dtj`` and ``dtk`` the cross product
    needs to be computed (i.e., is non-zero).

    References:
     - Davis, T. A. (2006). Direct Methods for Sparse Linear Systems.

    :param dtjs: List of term matrices included in ``X_j``
    :type dtks: DiscreteTerm
    :param dtk: List of term matrices included in ``X_k``
    :type dtk: DiscreteTerm
    :param W: Inner matrix ``W``, either as sparse array/matrix or numpy array
    :type W: scp.sparse.sparray | scp.sparse.spmatrix | np.ndarray | None
    :return: Pairs of indices for which cross-product needs to be computed in separate arrays for
        ``X_j`` (first array) and ``X_k`` (second array)
    :rtype: tuple[np.ndarray, np.ndarray]
    """

    # Ok, assuming that no term is zero everywhere we have two cases:
    # 1) "full" terms with no nonzero rows (term.nonzero is None)
    # 2) "partial" terms with some nonzero rows (term.nonzero has indices)
    #
    # Only cross-products between partial terms j and k can potentially be zero and also only if W
    # is not fully dense.

    # Number of rows for both sets of terms has to be the same
    N = dtjs[0].indices[0].shape[0]

    # Get full and partial j term sorting
    full_j = []
    partial_j = []
    n_partial_j = 0

    for j, dtj in enumerate(dtjs):
        if dtj.nonzero_rows is None:
            full_j.append(j)
        else:
            partial_j.append(j)
            n_partial_j += 1

    # Same for k
    full_k = []
    partial_k = []
    n_partial_k = 0

    for k, dtk in enumerate(dtks):
        if dtk.nonzero_rows is None:
            full_k.append(k)
        else:
            partial_k.append(k)
            n_partial_k += 1

    # We definitely need the cross-products between full j terms and all k terms
    # as well as the cross-products between partial j terms and full k terms.

    # Stores index pairs for nonzero cross-product blocks between term j and k
    nonzero_j = []
    nonzero_k = []

    # First set of cross-products
    fulljk = np.meshgrid(full_j, np.arange(len(dtks)), indexing="ij")
    nonzero_j.extend(fulljk[0].flatten())
    nonzero_k.extend(fulljk[1].flatten())

    # Second set
    fullkj = np.meshgrid(partial_j, full_k, indexing="ij")
    nonzero_j.extend(fullkj[0].flatten())
    nonzero_k.extend(fullkj[1].flatten())

    # Now if W is dense, then we also need all remaining cross-products between
    # partial terms in j and k
    if isinstance(W, np.ndarray):
        # Assume W is fully dense and not just incorrectly stored in dense storage..
        partialjk = np.meshgrid(partial_j, partial_k, indexing="ij")
        nonzero_j.extend(partialjk[0].flatten())
        nonzero_k.extend(partialjk[1].flatten())

    elif n_partial_j > 0 and n_partial_k > 0:
        # If we still have partial blocks left, then we now need to check which
        # of the corresponding cross-products we need. To compute this efficiently
        # we can exploit the dt.non_zero arrays and use them to set up binary indicator
        # structure matrices to determine which of the remaining cross-products are zero everywhere.
        # The following relies on Theorem 2.1 originally by Gilbert, stated in Davis "Direct
        # Methods for Sparse Linear Systems".

        # Prepare sparse structure matrices. Each have N rows and number of columns matches
        # dimension of the partial terms. Columns are non_zero rows for terms.
        Sj_idx = [dtjs[j].nonzero_rows for j in partial_j]
        Sk_idx = [dtks[k].nonzero_rows for k in partial_k]

        Sjks = []
        for idx in [Sj_idx, Sk_idx]:

            # From scipy docs for sparse csc_array: "row indices for column i are stored in
            # indices[indptr[i]:indptr[i+1]]...". So indptr is cumsum over len of arrays in
            # idx arrays starting from zero
            ptr = np.concatenate(([0], np.cumsum([len(r) for r in idx])))
            print(idx, ptr.shape, np.concatenate(idx).shape)

            # Now can create structure indicator as sparse matrix. Again from scipy docs:
            # "...and their corresponding values are stored in data[indptr[i]:indptr[i+1]]", so
            # np.ones(ptr[-1]) gives sufficient ones to fill all nonzeros
            S = scp.sparse.csc_array(
                (np.ones(ptr[-1]), np.concatenate(idx), ptr), shape=(N, len(idx))
            )
            Sjks.append(S)

        # Given structure indicators we can compute the overlap based on straightforward extension
        # of Gilbert's theorem as:
        overlap = Sjks[0].T @ Sjks[1] if W is None else Sjks[0].T @ (abs(W) @ Sjks[1])

        # Extract term indices hat have overlap
        overlap_j, overlap_k = overlap.nonzero()
        nonzero_j.extend(partial_j[overlap_j])
        nonzero_k.extend(partial_k[overlap_k])

    # Now need to get things in order for each j for each k...
    # Can use lexsort but need to remember LAST index is primary sort dim!
    nonzero_j = np.array(nonzero_j)
    nonzero_k = np.array(nonzero_k)
    order = np.lexsort((nonzero_k, nonzero_j))
    return nonzero_j[order], nonzero_k[order]


def _XjTWXk(
    dtj: DiscreteTerm,
    dtk: DiscreteTerm,
    hasW: bool,
    W: list,
    alg: Callable,
) -> np.ndarray:
    """Computes cross product between term matrix ``dtj`` and ``dtk``. Optionally,
    the cross-product can involve an inner matrix ``W``.

    Relies on a slightly modified version of the cross-product algorithm by Wood, Shaddick &
    Augustin (2017) which can take into account for terms ``dtj`` and ``dtk`` having different
    dimensions.

    References:
     - Wood, S. N., Li, Z., Shaddick, G., & Augustin, N. H. (2017). Generalized Additive Models\
        for Gigadata: Modeling the U.K. Black Smoke Network Daily Data. Journal of the\
        American Statistical Association, 112(519), 1199–1210.\
        https://doi.org/10.1080/01621459.2016.1195744


    :param dtj: Term matrix for some term j
    :type dtj: DiscreteTerm
    :param dtk: Term matrix for some term k
    :type dtk: DiscreteTerm
    :param hasW: Whether the cross-product should involve an inner matrix ``W``.
    :type hasW: bool
    :param W: Inner matrix ``W`` either as a list including a single dense matrix or a
        list obtained from calling :func:`map_csc_to_eigen` on a sparse matrix ``W``.
    :type W: list
    :param alg: The c++ algorithm to use depending on dense/sparse matrix ``W``
    :type alg: Callable
    :return: A 2d numpy array containing the evaluated cross-product
    :rtype: np.ndarray
    """
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


class DiscreteCrossProduct:
    """A wrapper class to accumulate the (sparse) cross-product of two :class:`DiscreteModelMatrix`.

    :param shape: Dimensions of final cross-product.
    :type shape: tuple[int, int]
    :param sparse: Whether the cross-product should be accumulated in sparse matrix storage or in
        a dense array.
    :type sparse: bool
    """

    def __init__(self, shape: tuple[int, int], sparse: bool):
        self.shape = shape
        self.sparse = sparse

        # Can now prepare data storage
        if self.sparse:
            self.rows: list[int] = []
            self.cols: list[int] = []
            self.data: list[int] = []
        else:
            self.data = np.zeros(shape)

    def insert_block(self, srow: int, scol: int, block: np.ndarray) -> None:
        """Inserts the cross-product block ``block`` with upper left index ``srow`` and ``scol``.

        :param srow: Index of upper row
        :type srow: int
        :param scol: Index of left-most column
        :type scol: int
        :param block: Array holding dense cross-product between two :class:`DiscreteTerm`.
        :type block: np.ndarray
        """
        if self.sparse:
            # There is a chance that the cross-product itself is somewhat sparse, so it makes
            # sense to get only the rows and cols actually filled by this block.
            # see: https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html#numpy-nonzero
            nonzero = np.nonzero(block)  # (rows, cols) that are non-zero
            if len(nonzero[0]) == 0:
                # Block is completely empty
                return

            self.rows.extend(nonzero[0] + srow)
            self.cols.extend(nonzero[1] + scol)
            self.data.extend(block[nonzero])

        else:
            # Simply write to data for dense case
            erow = srow + block.shape[0]
            ecol = scol + block.shape[1]
            self.data[srow:erow, scol:ecol] = block

    def eval(self) -> np.ndarray | scp.sparse.csc_array:
        """Returns the accumulated cross-product as sparse csc_array or numpy array.

        :return: The accumulated cross-product
        :rtype: np.ndarray | scp.sparse.csc_array
        """

        if self.sparse:
            # Build sparse matrix from data, rows, and cols
            return scp.sparse.csc_array(
                (
                    self.data,
                    (self.rows, self.cols),
                ),
                shape=self.shape,
            )

        # Dense case can return directly
        return self.data


class DiscreteModelMatrix:
    """A compact representation for model matrices with discretized covariates.

    Relies on modified versions of the algorithms developed by Wood, Shaddick & Augustin (2017) to
    implement many matrix (vector) products and slicing/indexing operations. Specifically,
    the :class:`DiscreteModelMatrix` class implements ``X.T@Y``, ``Y@X.T``, ``X@B``, ``B@X``,
    and ``X.T@W@Z`` where ``Y``, ``B``, and ``W`` can be sparse or dense matrices or vectors and
    ``X`` and ``Z`` are instances of :class:`DiscreteModelMatrix`. The final kind of cross-product
    as well as operations involving sufficiently small matrices ``Y`` and ``B`` are evaluated
    explicitly.

    Specifically, if the resulting matrix has fewer than ``self.max_slice_size`` columns
    or fewer (or equal) rows than ``X`` has columns the matrix is returned as a numpy array or
    scipy sparse csc_array (depending on ``self.return_sparse``). Otherwise, the product is
    represented implcitily (i.e., these operations return a modified version of ``X`` as an instance
    of :class:`DiscreteModelMatrix`) and only evaluated if a future operation meets these conditions
    or when calling ``X.eval()`` (or the ``toarray()`` method directly).

    The same rules apply when indexing/slicing ``X``: Slices that are too big are again returned as
    a modified version of ``X`` as an instance of :class:`DiscreteModelMatrix`.

    References:
     - Wood, S. N., Li, Z., Shaddick, G., & Augustin, N. H. (2017). Generalized Additive Models\
        for Gigadata: Modeling the U.K. Black Smoke Network Daily Data. Journal of the\
        American Statistical Association, 112(519), 1199–1210.\
        https://doi.org/10.1080/01621459.2016.1195744

    :param dTerms: List of term matrices that make up this model matrix.
    :type dTerms: list[DiscreteTerm]
    :ivar scp.sparse.csc_array | np.ndarray | None preM: Any matrix by which to pre-multiply self
        when evaluating the matrix. Read-only. Initialized with ``None``.
    :ivar scp.sparse.csc_array | np.ndarray | None postM: Any matrix by which to post-multiply self
        when evaluating the matrix. Read-only. Initialized with ``None``.
    :ivar max_slice_size int: Maximum number of columns at which a slice/product is evaluated
        explicitly. Initialized with 10.
    :ivar id int: An identifier used internally to decide whether a cross-product involves two
        of the same matrices. Do not change this unless you know what you're doing.
        Initialized with ``id(self)``.
    :ivar return_sparse bool: Whether to return a sparse array or a dense one. Initialized with
        False.
    :ivar tuple[int, int] shape: The shape of self. Initialized based on ``dTerms``.
    """

    def __init__(self, dTerms: list[DiscreteTerm]):
        self.terms: list[DiscreteTerm] = dTerms
        self._preM: scp.sparse.csc_array | np.ndarray | None = None
        self._postM: scp.sparse.csc_array | np.ndarray | None = None
        self._prepost_id = uuid.uuid1()
        self.max_slice_size: int = 10
        self.id: int = id(self)
        self.return_sparse: bool = False

        # Compute shape
        self.shape: tuple[int, int] = (
            self.terms[0].indices[0].shape[0],
            int(self.terms[-1].end_idx),
        )

        self._T: bool = False

        # Initial computation of nonzero rows
        self.__update_term_nonzeros()

    # Getters for preM and postM
    @property
    def preM(self) -> scp.sparse.csc_array | np.ndarray | None:
        return self._preM

    @property
    def postM(self) -> scp.sparse.csc_array | np.ndarray | None:
        return self._postM

    def __update_prepostM(
        self,
        preM: scp.sparse.csc_array | np.ndarray | None,
        postM: scp.sparse.csc_array | np.ndarray | None,
    ) -> None:
        """Updates the matrices stored in self._preM and self.postM.

        Note, that this method always sets both matrices, so leaving one (or both)
        at the default None will set the corresponding matrices (or matric) to None.

        :param preM: Matrix by which to pre-multiply ``self`` or None
        :type preM: scp.sparse.csc_array | np.ndarray | None
        :param postM: Matrix by which to post-multiply ``self`` or None
        :type postM: scp.sparse.csc_array | np.ndarray | None
        """
        self._preM = preM
        self._postM = postM
        self._prepost_id = uuid.uuid1()

    def __update_term_nonzeros(self) -> None:
        """Computes for each term which rows of the term matrix can be nonzero.

        **Note**: For terms with multiple marginals (e.g., tensor terms) what counts as a zero rows
        are rows that are zero in all marginals (e.g., anything with a by-keyword).
        """

        for term in self.terms:

            # Get zero rows of unique matrices
            zeros = [np.all(umat == 0, axis=1) for umat in term.unique_matrices]

            # Early cheap check if term has no zero rows
            # To avoid storing an array of all rows, just use None
            if np.all([~np.any(uz) for uz in zeros]):
                term.nonzero_rows = None
                continue

            # Have zero rows in unique matrices, now expand to full matrices
            exp_zeros = np.ones(term.indices[0].shape[0], dtype=bool)
            for umati in range(term.n_marginals):
                exp_zeros &= zeros[umati][term.indices[umati]]

            # Only thing left is to get indices of nonzero rows
            term.nonzero_rows = np.nonzero(~exp_zeros)

    def __combineW(
        self, otherpreM: scp.sparse.csc_array | np.ndarray | None
    ) -> scp.sparse.csc_array | scp.sparse.spmatrix | np.ndarray | None:
        """Combines any matrix by which self is to be post-multiplied with matrix ``otherpreM`` by
        which another DicreteModelMatrix is post-multiplied.

        :param otherpreM: Optional matrix by which another DiscreteModelMatrix rather than self is
            pre-multiplied by.
        :type otherpreM: scp.sparse.csc_array | np.ndarray | None
        :return: The combined product or None
        :rtype: scp.sparse.csc_array | scp.sparse.spmatrix | np.ndarray | None
        """
        # Collect everything into W
        W: scp.sparse.csc_array | scp.sparse.spmatrix | np.ndarray | None = None
        if self.postM is not None:
            W = self.postM
        if otherpreM is not None:
            if W is not None:
                W = W @ otherpreM
            else:
                W = otherpreM

        return W

    def __prepareW(
        self, W: scp.sparse.csc_array | scp.sparse.spmatrix | np.ndarray | None
    ) -> tuple[list, bool, bool]:
        """Internal function to compute any diagonal matrix W involved in a cross-product

        :param W: Matrix involved in a cross-product between two matrices or None.
        :type W: scp.sparse.csc_array | np.ndarray | None
        :return: Inner matrix ``W`` (if it exists else None) either as a list including a single
            dense matrix or as a list obtained from calling :func:`map_csc_to_eigen` on a sparse
            matrix ``W``. Additionally a bool saying whether ``W`` exists at all and another one
            indicating whether it is sparse.
        :rtype: tuple[list, bool, bool]
        """

        # Check if W exists
        hasW: bool = W is not None

        if W is None:
            W = np.array([])

        # Check if W is sparse
        W_is_sparse = False
        if hasW and (
            isinstance(W, scp.sparse.sparray) or isinstance(W, scp.sparse.spmatrix)
        ):
            W = W.tocsc()
            W = map_csc_to_eigen(W)
            W_is_sparse = True

        if not W_is_sparse:
            if W is not None and not W.flags.f_contiguous:
                # C++ side expects Fortran style dense matrix
                W = np.asfortranarray(W)

            W = [W]

        return W, hasW, W_is_sparse

    def __XTWZ(
        self,
        other: TDiscreteMatrix,
        W: scp.sparse.csc_array | scp.sparse.spmatrix | np.ndarray | None,
    ) -> np.ndarray:
        """Compute cross-product between two different instances of :class:`DiscreteModelMatrix`.

        Relies on a slightly modified version of the cross-product algorithm by Wood, Shaddick &
        Augustin (2017).

        References:
         - Wood, S. N., Li, Z., Shaddick, G., & Augustin, N. H. (2017). Generalized Additive Models\
            for Gigadata: Modeling the U.K. Black Smoke Network Daily Data. Journal of the\
            American Statistical Association, 112(519), 1199–1210.\
            https://doi.org/10.1080/01621459.2016.1195744

        :param other: Another :class:`DiscreteModelMatrix`
        :type other: DiscreteModelMatrix
        :param W: An optional matrix ``W`` involved in the cross-product
        :type W: scp.sparse.csc_array | scp.sparse.spmatrix | np.ndarray | None
        :return: The cross-product as a 2D numpy array
        :rtype: np.ndarray
        """
        # Determine blocks to compute
        blocks_j, blocks_k = _nonzeroXjTWXk(self.terms, other.terms, W)
        # Now get W into correct list format
        W, hasW, W_is_sparse = self.__prepareW(W)

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

        # Fill cross-product - evaluate only non-zero blocks
        XTWZ = DiscreteCrossProduct(
            (n_row, n_col), self.return_sparse and other.return_sparse
        )

        for dtji, dtki in zip(blocks_j, blocks_k):
            dtj = self.terms[dtji]
            dtk = other.terms[dtki]

            if (dtj.end_idx <= dtj.start_idx) or (dtk.end_idx <= dtk.start_idx):
                continue

            block = _XjTWXk(dtj, dtk, hasW, W, alg)
            XTWZ.insert_block(dtj.start_idx, dtk.start_idx, block)

        # Now evaluate cross-product and return
        XTWZ = XTWZ.eval()

        return XTWZ

    def __XTWX(
        self,
        W: scp.sparse.csc_array | scp.sparse.spmatrix | np.ndarray | None,
        is_symmetric: bool,
    ) -> np.ndarray:
        """Compute cross-product between the same (up to different ``preM`` and ``postM``) instance
        of :class:`DiscreteModelMatrix`.

        Relies on the cross-product algorithm by Wood, Shaddick & Augustin (2017).

        References:
         - Wood, S. N., Li, Z., Shaddick, G., & Augustin, N. H. (2017). Generalized Additive Models\
            for Gigadata: Modeling the U.K. Black Smoke Network Daily Data. Journal of the\
            American Statistical Association, 112(519), 1199–1210.\
            https://doi.org/10.1080/01621459.2016.1195744

        :param W: An optional matrix ``W`` involved in the cross-product
        :type W: scp.sparse.csc_array | scp.sparse.spmatrix | np.ndarray | None
        :param is_symmetric: Whether ``W`` is symmetric or not
        :type is_symmetric: bool
        :return: The cross-product as a 2D numpy array
        :rtype: np.ndarray
        """
        # Determine blocks to compute
        blocks_j, blocks_k = _nonzeroXjTWXk(self.terms, self.terms, W)
        # Now get W into correct list format
        W, hasW, W_is_sparse = self.__prepareW(W)

        alg = discrete.XTWXS if W_is_sparse else discrete.XTWXD

        if is_symmetric:
            # Blocks below the diagonal can be obtained by transposing those above.
            upper = blocks_j <= blocks_k
            blocks_j = blocks_j[upper]
            blocks_k = blocks_k[upper]

        # Check dimensions:
        n_col = 0
        for dt in self.terms:
            if dt.end_idx <= dt.start_idx:
                # Skip terms removed completely
                continue

            n_col += dt.end_idx - dt.start_idx

        XTWX = DiscreteCrossProduct((n_col, n_col), self.return_sparse)
        for dtji, dtki in zip(blocks_j, blocks_k):
            dtj = self.terms[dtji]
            dtk = self.terms[dtki]

            if (dtj.end_idx <= dtj.start_idx) or (dtk.end_idx <= dtk.start_idx):
                continue

            block = _XjTWXk(dtj, dtk, hasW, W, alg)
            XTWX.insert_block(dtj.start_idx, dtk.start_idx, block)

            if is_symmetric and dtji != dtki:
                XTWX.insert_block(dtk.start_idx, dtj.start_idx, block.T)

        # Now evaluate cross-product and return
        XTWX = XTWX.eval()

        return XTWX

    def __XTy(self, y: np.ndarray) -> np.ndarray:
        """Compute ``X.T@y``, where ``X`` is a :class:`DiscreteModelMatrix`.

        Relies on the algorithms by Wood, Shaddick & Augustin (2017).

        References:
         - Wood, S. N., Li, Z., Shaddick, G., & Augustin, N. H. (2017). Generalized Additive Models\
            for Gigadata: Modeling the U.K. Black Smoke Network Daily Data. Journal of the\
            American Statistical Association, 112(519), 1199–1210.\
            https://doi.org/10.1080/01621459.2016.1195744

        :param y: Vector for which to evaluate the product
        :type y: np.ndarray
        :return: The matrix vector product evaluated
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
        """Compute ``X@b``, where ``X`` is a :class:`DiscreteModelMatrix`.

        Relies on the algorithms by Wood, Shaddick & Augustin (2017).

        References:
         - Wood, S. N., Li, Z., Shaddick, G., & Augustin, N. H. (2017). Generalized Additive Models\
            for Gigadata: Modeling the U.K. Black Smoke Network Daily Data. Journal of the\
            American Statistical Association, 112(519), 1199–1210.\
            https://doi.org/10.1080/01621459.2016.1195744

        :param b: Vector for which to evaluate the product
        :type b: np.ndarray
        :return: The matrix vector product evaluated
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

        :param row: Row index to extract
        :type row: int
        :return: Row of ``X`` after excluding any dropped columns
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

        Relies on the algorithms by Wood, Shaddick & Augustin (2017).

        References:
         - Wood, S. N., Li, Z., Shaddick, G., & Augustin, N. H. (2017). Generalized Additive Models\
            for Gigadata: Modeling the U.K. Black Smoke Network Daily Data. Journal of the\
            American Statistical Association, 112(519), 1199–1210.\
            https://doi.org/10.1080/01621459.2016.1195744

        :param col: Column index to extract
        :type col: int
        :return: The column extraced
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

    def __slice(
        self, rows: int | np.ndarray | None, cols: int | np.ndarray | None, pairs: bool
    ) -> np.ndarray | scp.sparse.sparray | float | TDiscreteMatrix:
        """Extracts a slice from ``self`` without any ``preM`` or ``postM``.

        Returned explicitly if:
         - at most ``self.max_slice_size`` columns
         - or at most ``max(self.max_slice_size, n_cols)`` rows from ``X``
        need to be extracted. Except that paired indices (lists/arrays) produce a vector and are
        thus always returned explicitly.

        :param rows: Rows of ``self`` to extract
        :type rows: int | np.ndarray | None
        :param cols: Columns of ``self`` to extract
        :type cols: int | np.ndarray | None
        :param pairs: Whether ``rows`` and ``cols`` are paired
        :type pairs: bool
        :return: The requested slice.
        :rtype: np.ndarray | scp.sparse.sparray | float | TDiscreteMatrix
        """
        # Work with X rather than X.T - so rows and columns are flipped if self._T
        c_shape = self.shape
        xrows, xcols = rows, cols
        if self._T:
            c_shape = np.flip(c_shape)
            xrows, xcols = cols, rows

        # Check whether we need to flatten in the end to match numpy and make sure
        # xrows and xcol are arrays to make unified handling easier
        flatten_rows = isinstance(rows, int)
        flatten_cols = isinstance(cols, int)
        xr = np.arange(c_shape[0]) if xrows is None else np.atleast_1d(xrows)
        xc = np.arange(c_shape[1]) if xcols is None else np.atleast_1d(xcols)

        if pairs:
            # Result is a vector, so always evaluate explicitly
            ucols, cinv = np.unique(xc, return_inverse=True)

            if len(ucols) <= self.max_slice_size:
                # Few columns involved - extract those
                rcols = np.array([self.__get_col(col) for col in ucols]).reshape(
                    len(ucols), c_shape[0]
                )
                res = rcols[cinv, xr]

            else:
                # Extract rows in chunks, so that every chunk is at most as big as X.T@X
                chunk_size = max(c_shape[1], self.max_slice_size)
                res = np.zeros(len(xr))
                for start in range(0, len(xr), chunk_size):
                    stop = start + chunk_size
                    rrows = np.array([self.__get_row(row) for row in xr[start:stop]])

                    # Extract diagonal subset and asign
                    res[start:stop] = rrows[np.arange(len(rrows)), xc[start:stop]]

            return _eval_discrete_slice(res, self.return_sparse)

        if len(xc) <= self.max_slice_size:
            # Explicitly evaluate column slice
            rcols = np.array([self.__get_col(col) for col in xc]).reshape(
                len(xc), c_shape[0]
            )
            res = rcols[:, xr].T

        elif len(xr) <= max(c_shape[1], self.max_slice_size):
            # Explicitly evaluate row slice
            rrows = np.array([self.__get_row(row) for row in xr]).reshape(
                len(xr), c_shape[1]
            )
            res = rrows[:, xc]

        else:
            # Return implicit slice
            newS = copy.deepcopy(self)
            newS.id = id(newS)  # Update id

            # Columns can only be dropped so remaining ones are sorted and unique
            ucols, cinv = np.unique(xc, return_inverse=True)
            cidx = np.arange(c_shape[1])
            newS.__drop_columns(cidx[~np.isin(cidx, ucols)])

            if len(ucols) != len(xc) or np.any(ucols != xc):
                # Re-order/repeat columns via selection matrix: X[:, xc] = X[:, ucols] @ S
                Sdat = np.ones(len(xc))
                # rows match unique col values, cinv puts a 1 into each row for every rep/re-order
                # into the corresponding column of S
                Srows = cinv
                Scols = np.arange(len(xc))
                S = scp.sparse.csc_array(
                    (Sdat, (Srows, Scols)),
                    shape=(len(ucols), len(xc)),
                )

                # Determine correct multiplication order depending on transpose
                if self._T:
                    newS.__update_prepostM(preM=S.T.tocsc(), postM=None)
                else:
                    newS.__update_prepostM(preM=None, postM=S)

            # (Re-ordered/repeated) rows can be selected directly from the index vectors
            if xrows is not None:
                for dt in newS.terms:
                    dt.indices = [idx[xr] for idx in dt.indices]

                # Need to re-compute non-zero indices after row selection
                newS.__update_term_nonzeros()

            newS.shape = (len(xc), len(xr)) if self._T else (len(xr), len(xc))
            return newS

        # Now fianlize explicit return. First transpose if necessary
        if self._T:
            res = res.T

        # Now flatten if necessary
        if flatten_rows and flatten_cols:
            res = res[0, 0]
        elif flatten_rows:
            res = res[0, :]
        elif flatten_cols:
            res = res[:, 0]

        return _eval_discrete_slice(res, self.return_sparse)

    def __slice_product(
        self, rows: int | np.ndarray | None, cols: int | np.ndarray | None, pairs: bool
    ) -> np.ndarray | scp.sparse.sparray | float | TDiscreteMatrix:
        """Extracts a slice from ``self`` with ``preM`` and/or ``postM``.

        Let ``C`` be ``preM`` and ``D`` be ``postM``, then this function extracts a slice from
        ``C@X@D`` where either ``C`` or ``D`` might be missing.

        Slices are evaluated as:
         - ``(C@X)[r, c] = C[r, :] @ X[:, c]``
         - ``(X@D)[r, c] = X[r, :] @ D[:, c]``
         - ``(C@X@D)[r, c] = (C[r, :] @ X) @ D[:, c]`` (``C[r, :] @ (X.T @ D[:, c])`` if ``X._T``)

        Whether the result is returned explicitly depends on which slice of ``X`` is extracted and
        which product is evaluated. Paired indices are again returned explicitly in any case.

        :param rows: Rows of ``self`` to extract
        :type rows: int | np.ndarray | None
        :param cols: Columns of ``self`` to extract
        :type cols: int | np.ndarray | None
        :param pairs: Whether ``rows`` and ``cols`` are paired
        :type pairs: bool
        :return: The requested slice.
        :rtype: np.ndarray | scp.sparse.sparray | float | TDiscreteMatrix
        """
        if pairs:
            # Result is a vector, so always evaluate explicitly
            res = [np.zeros(0)]
            for start in range(0, len(rows), self.max_slice_size):
                stop = start + self.max_slice_size
                blk = self.__slice_product(rows[start:stop], cols[start:stop], False)
                if not isinstance(blk, np.ndarray):
                    blk = blk.toarray()
                res.append(np.diagonal(blk))

            return _eval_discrete_slice(np.concatenate(res), self.return_sparse)

        # X (or X.T) without preM and postM so that we can rely on __slice
        X = copy.copy(self)
        X.__update_prepostM(None, None)
        X.shape = (
            self.shape[0] if self.preM is None else self.preM.shape[1],
            self.shape[1] if self.postM is None else self.postM.shape[0],
        )

        # Slice preM and postM
        if self.preM is not None:
            preM = self.preM if rows is None else self.preM[rows, :]

        if self.postM is not None:
            postM = self.postM if cols is None else self.postM[:, cols]

        # Now handle products and remaining slices
        if self.postM is None:
            res = preM @ (X if cols is None else X.__slice(None, cols, False))

        elif self.preM is None:
            res = (X if rows is None else X.__slice(rows, None, False)) @ postM

        elif self._T:
            # Evaluate X.T @ postM first (sums over rows of X)
            res = preM @ (X @ postM)

        else:
            # Evaluate preM @ X first (sums over rows of X)
            res = (preM @ X) @ postM

        return _eval_discrete_slice(res, self.return_sparse)

    def __getitem__(
        self, key: int | list | np.ndarray | slice | np.integer
    ) -> np.ndarray | scp.sparse.sparray | TDiscreteMatrix:
        """Extracts a slice from ``X``.

        Relies on the algorithms by Wood, Shaddick & Augustin (2017).

        References:
         - Wood, S. N., Li, Z., Shaddick, G., & Augustin, N. H. (2017). Generalized Additive Models\
            for Gigadata: Modeling the U.K. Black Smoke Network Daily Data. Journal of the\
            American Statistical Association, 112(519), 1199–1210.\
            https://doi.org/10.1080/01621459.2016.1195744

        :param key: The key determining how to slice ``X``
        :type key: int | list | np.ndarray | slice | np.integer
        :return: The required slice either evaluated explicitly or implicitly
        :rtype: np.ndarray | scp.sparse.sparray | TDiscreteMatrix
        """

        if isinstance(key, tuple):
            if len(key) > 2:
                raise ValueError(
                    "Slices > 2D are not supported with class:`DiscreteModelMatrix`."
                )
            rows = key[0]
            cols = key[1] if len(key) == 2 else slice(None, None, None)
        else:
            # Row indexing
            rows = key
            cols = slice(None, None, None)

        # Two index arrays are paired, as in numpy
        pairs = not isinstance(rows, (int, np.integer, slice)) and not isinstance(
            cols, (int, np.integer, slice)
        )

        # Unify indices
        rows = _convert_index(rows, self.shape[0])
        cols = _convert_index(cols, self.shape[1])

        if pairs:
            # Need to expand to have same indexing behavior
            rows, cols = np.broadcast_arrays(rows, cols)

        if self.preM is None and self.postM is None:
            return self.__slice(rows, cols, pairs)

        return self.__slice_product(rows, cols, pairs)

    def __matmul__(
        self,
        other: np.ndarray | scp.sparse.sparray | scp.sparse.spmatrix | TDiscreteMatrix,
    ) -> np.ndarray | scp.sparse.sparray | TDiscreteMatrix:
        """Computes ``X@other``.

        Relies on the algorithms by Wood, Shaddick & Augustin (2017).

        References:
         - Wood, S. N., Li, Z., Shaddick, G., & Augustin, N. H. (2017). Generalized Additive Models\
            for Gigadata: Modeling the U.K. Black Smoke Network Daily Data. Journal of the\
            American Statistical Association, 112(519), 1199–1210.\
            https://doi.org/10.1080/01621459.2016.1195744

        :param other: The other matrix/vector involved in the product
        :type other: np.ndarray | scp.sparse.sparray | scp.sparse.spmatrix | TDiscreteMatrix
        :return: The product evaluated either explicitly or implicitly
        :rtype: np.ndarray | scp.sparse.sparray | TDiscreteMatrix
        """
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
                (other.shape[1] <= self.max_slice_size)
                or (self._T and other.shape[1] <= self.shape[0])
            ) and isinstance(other, DiscreteModelMatrix) is False:

                rsparse = self.return_sparse and isinstance(other, scp.sparse.sparray)

                if other.shape[1] == 1:
                    if not isinstance(other, np.ndarray):
                        # Can simply cast here, not expensive
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

                        # Extract column before casting
                        y = Y[:, [ci]]
                        if not isinstance(y, np.ndarray):
                            y = y.toarray()

                        XTY.append(self.__XTy(y).flatten())

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
                    # Extract column before casting
                    b = B[:, [ci]]
                    if not isinstance(b, np.ndarray):
                        b = b.toarray()
                    XB.append(self.__Xb(b).flatten())

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

                # Extract W
                W = self.__combineW(other.preM)
                # If self is the transpose of other W = other.preM.T @ other.preM is symmetric
                is_symmetric = self._prepost_id == other._prepost_id
                if not is_symmetric:
                    # Otherwise need to check numerically
                    is_symmetric = _is_symmetric(W)
                XTWX = self.__XTWX(W, is_symmetric)

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
                # Extract W
                W = self.__combineW(other.preM)
                # And evaluate cross-product again
                XTWZ = self.__XTWZ(other, W)
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

                # Update postM, keep preM
                newS.__update_prepostM(
                    preM=newS.preM,
                    postM=other if newS.postM is None else newS.postM @ other,
                )
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
        """Computes ``other@X``.

        Relies on the algorithms by Wood, Shaddick & Augustin (2017).

        References:
         - Wood, S. N., Li, Z., Shaddick, G., & Augustin, N. H. (2017). Generalized Additive Models\
            for Gigadata: Modeling the U.K. Black Smoke Network Daily Data. Journal of the\
            American Statistical Association, 112(519), 1199–1210.\
            https://doi.org/10.1080/01621459.2016.1195744

        :param other: The other matrix/vector involved in the product
        :type other: np.ndarray | scp.sparse.sparray | scp.sparse.spmatrix
        :return: The desired product evaluated either explicitly or implicitly.
        :rtype: np.ndarray | scp.sparse.sparray | TDiscreteMatrix
        """

        flatten = False
        if len(other.shape) == 1:
            vec = other
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

            # Handles explicit and implicit case, so no need to separate those two.
            if flatten:
                # Pass on the original vector (sparse vectors cannot be flattened), so that
                # the result is a vector as well.
                return self.transpose() @ vec

            return (self.transpose() @ other.T).T

        else:
            raise NotImplementedError(
                (
                    "Reflected Matrix multiplication is only implemented for scipy sparse and"
                    " numpy arrays."
                )
            )

    __array_priority__ = 10000

    def __mul__(self, other: float | int | np.ndarray) -> np.ndarray | TDiscreteMatrix:
        """Computes ``X * other``.

        Relies on the algorithms by Wood, Shaddick & Augustin (2017).

        References:
         - Wood, S. N., Li, Z., Shaddick, G., & Augustin, N. H. (2017). Generalized Additive Models\
            for Gigadata: Modeling the U.K. Black Smoke Network Daily Data. Journal of the\
            American Statistical Association, 112(519), 1199–1210.\
            https://doi.org/10.1080/01621459.2016.1195744

        :param other: The other float/int/vector involved in the product
        :type other: float | int | np.ndarray
        :return: Returns the desired product either evaluated explicitly or implicitly.
        :rtype: np.ndarray | TDiscreteMatrix
        """
        if (
            isinstance(other, float)
            or isinstance(other, int)
            or isinstance(other, np.integer)
        ):
            newS = copy.deepcopy(self)
            newS.id = id(newS)  # Update id, since values differ from self now
            for dti, dt in enumerate(newS.terms):
                dt.unique_matrices[0] *= other
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

    def __rmul__(self, other: float | int | np.ndarray) -> np.ndarray | TDiscreteMatrix:
        """Computes ``other * X``.

        Relies on the algorithms by Wood, Shaddick & Augustin (2017).

        References:
         - Wood, S. N., Li, Z., Shaddick, G., & Augustin, N. H. (2017). Generalized Additive Models\
            for Gigadata: Modeling the U.K. Black Smoke Network Daily Data. Journal of the\
            American Statistical Association, 112(519), 1199–1210.\
            https://doi.org/10.1080/01621459.2016.1195744

        :param other: The other float/int/vector involved in the product
        :type other: float | int | np.ndarray
        :return: Returns the desired product either evaluated explicitly or implicitly.
        :rtype: np.ndarray | TDiscreteMatrix
        """
        return self.__mul__(other)

    def is_transposed(self) -> bool:
        return self._T

    @property
    def T(self) -> TDiscreteMatrix:
        """Returns transpose. Data is not copied.

        :return: View of transpose of self
        :rtype: TDiscreteMatrix
        """
        return self.transpose()

    def transpose(self) -> TDiscreteMatrix:
        """Returns transpose. Data is not copied.

        :return: View of transpose of self
        :rtype: TDiscreteMatrix
        """
        newS = DiscreteModelMatrix(self.terms)
        newS.return_sparse = self.return_sparse
        newS.max_slice_size = self.max_slice_size
        newS.id = self.id
        newS.__update_prepostM(
            preM=None if self.postM is None else self.postM.T,
            postM=None if self.preM is None else self.preM.T,
        )

        # Same matrices, just transposed can keep prepost id
        newS._prepost_id = self._prepost_id

        newS._T = not self._T  # Update flag
        newS.shape = (self.shape[1], self.shape[0])

        return newS

    def copy(self) -> TDiscreteMatrix:
        """Return copy of self.

        :return: Copy of self
        :rtype: TDiscreteMatrix
        """
        return copy.deepcopy(self)

    def toarray(self) -> np.ndarray:
        """Return self as a numpy array.

        :return: Self evaluated as numpy array
        :rtype: np.ndarray
        """

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

        :return: Self evaluated as dense or sparse matrix
        :rtype: np.ndarray | scp.sparse.csc_array
        """

        res = self.toarray()
        return _eval_discrete_slice(res, self.return_sparse)

    def __drop_columns(self, cols: list[int]) -> None:
        """Exclude columns from the model matrix.

        :param cols: Columns to exclude
        :type cols: list[int]
        """

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
