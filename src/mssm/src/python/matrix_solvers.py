import numpy as np
import scipy as scp
import eigen_solvers
import warnings
from multiprocessing import managers, shared_memory
import multiprocessing as mp
from itertools import repeat
from .custom_types import PenType, LambdaTerm


def map_csc_to_eigen(
    X: scp.sparse.csc_array,
) -> tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Pybind11 comes with copy overhead for sparse matrices, so instead of passing the
    sparse matrix to c++, I pass the data, indices, and indptr arrays as buffers to c++.
    see: https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html.

    An Eigen mapping can then be used to refer to these, without requiring an extra copy.
    see: https://eigen.tuxfamily.org/dox/classEigen_1_1Map_3_01SparseMatrixType_01_4.html

    The mapping needs to assume compressed storage, since then we can use the indices, indptr, and
    data arrays directly for the valuepointer, innerPointer, and outerPointer fields of the
    sparse array map constructor.
    see: https://eigen.tuxfamily.org/dox/group__TutorialSparse.html (section sparse matrix format).

    I got this idea from the NumpyEigen project, which also uses such a map!
    see: https://github.com/fwilliams/numpyeigen/blob/master/src/npe_sparse_array.h#L74

    :param X: Some sparse matrix
    :type X: scp.sparse.csc_array
    :return: Number of rows in X, Number of cols in X, Number of non-zero elements in X, X.data,
        X.indptr.astype(np.int64), X.indices.astype(np.int64)
    :rtype: tuple[int,int,int,np.ndarray,np.ndarray,np.ndarray]
    """

    if X.format != "csc":
        raise TypeError(
            f"Format of sparse matrix passed to c++ MUST be 'csc' but is {X.getformat()}"
        )

    if X.has_sorted_indices is False:
        raise TypeError(
            "Indices of sparse matrix passed to c++ MUST be sorted but are not."
        )

    rows, cols = X.shape

    # Cast to int64 here, since that's what the c++ side expects to be stored in the buffers
    return (
        rows,
        cols,
        X.nnz,
        X.data,
        X.indptr.astype(np.int64),
        X.indices.astype(np.int64),
    )


def map_csr_to_eigen(
    X: scp.sparse.csr_array,
) -> tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]:
    """see: :func:`map_csc_to_eigen`

    :param X: Some sparse matrix
    :type X: scp.sparse.csr_array
    :return: Number of rows in X, Number of cols in X, Number of non-zero elements in X, X.data,
        X.indptr.astype(np.int64), X.indices.astype(np.int64)
    :rtype: tuple[int,int,int,np.ndarray,np.ndarray,np.ndarray]
    """

    if X.format != "csr":
        raise TypeError(
            f"Format of sparse matrix passed to c++ MUST be 'csr' but is {X.getformat()}"
        )

    if X.has_sorted_indices is False:
        raise TypeError(
            "Indices of sparse matrix passed to c++ MUST be sorted but are not."
        )

    rows, cols = X.shape

    # Cast to int64 here, since that's what the c++ side expects to be stored in the buffers
    return (
        rows,
        cols,
        X.nnz,
        X.data,
        X.indptr.astype(np.int64),
        X.indices.astype(np.int64),
    )


def translate_sparse(
    mat: scp.sparse.csc_array,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Translate canonical sparse csc matrix representation into data, row, col representation

    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.html#scipy.sparse.csc_array # noqa: E501

    :param mat: sparse matrix
    :type mat: scp.sparse.csc_array
    :return: data, rows, cols of sparse matrix
    :rtype: tuple[np.ndarray,np.ndarray,np.ndarray]
    """

    if mat.format != "csc":
        raise TypeError(
            (
                "Format of sparse matrix passed to be translated must be 'csc' but "
                f"is {mat.getformat()}"
            )
        )

    elements = mat.data
    idx = mat.indices
    iptr = mat.indptr

    data = []
    rows = []
    cols = []

    for ci in range(mat.shape[1]):

        c_data = elements[iptr[ci] : iptr[ci + 1]]  # noqa: E203
        c_rows = idx[iptr[ci] : iptr[ci + 1]]  # noqa: E203

        data.extend(c_data)
        rows.extend(c_rows)
        cols.extend([ci for _ in range(len(c_rows))])

    return data, rows, cols


def compute_eigen_perm(Pr: list[int] | np.ndarray) -> scp.sparse.csc_array:
    """Internal function. Computes column permutation matrix obtained from Eigen.

    :param Pr: List of column indices
    :type Pr: list[int] | np.ndarray
    :return: Permutation matrix as sparse array
    :rtype: scp.sparse.csc_array
    """

    nP = len(Pr)
    P = [1 for _ in range(nP)]
    Pc = [c for c in range(nP)]
    Perm = scp.sparse.csc_array((P, (Pr, Pc)), shape=(nP, nP))
    return Perm


def apply_eigen_perm(
    Pr: list[int], InvCholXXSP: scp.sparse.csc_array | np.ndarray
) -> scp.sparse.csc_array | np.ndarray:
    """Internal function. Unpivots columns of ``InvCholXXSP`` (usually the inverse of a
    Cholesky factor) and returns the unpivoted version.

    :param Pr: List of column indices
    :type Pr: list[int]
    :param InvCholXXSP: Pivoted matrix
    :type InvCholXXSP: scp.sparse.csc_array | np.ndarray
    :return: Unpivoted matrix
    :rtype: scp.sparse.csc_array | np.ndarray
    """
    Perm = compute_eigen_perm(Pr)
    InvCholXXS = InvCholXXSP @ Perm
    return InvCholXXS


def cpp_chol(
    A: scp.sparse.csc_array | np.ndarray,
) -> tuple[scp.sparse.csc_array | np.ndarray, int]:
    """Computes Cholesky of ``A``.

    :param A: Some square symmetric matrix
    :type A: scp.sparse.csc_array | np.ndarray
    :return: Returns Cholesky and code indicating success
    :rtype: tuple[scp.sparse.csc_array | np.ndarray, int]
    """
    if isinstance(A, np.ndarray):
        return eigen_solvers.dchol(A)

    return eigen_solvers.chol(*map_csc_to_eigen(A))


def cpp_cholP(
    A: scp.sparse.csc_array | np.ndarray,
) -> tuple[scp.sparse.csc_array | np.ndarray, list[int], int]:
    """Computes pivoted Cholesky of ``A``.

    :param A: Some square symmetric matrix
    :type A: scp.sparse.csc_array | np.ndarray
    :return: Returns pivoted Cholesky, pivoted column order, and code indicating success
    :rtype: tuple[scp.sparse.csc_array | np.ndarray,list[int],int]
    """
    if isinstance(A, np.ndarray):
        # Perform stability oriented pivoted LDL decomp
        L, d, p, code = eigen_solvers.dcholP(A)

        if code == 0:
            # Check PD
            eps = np.power(np.finfo(float).eps, 0.5)
            thresh = eps * np.max(np.abs(d))
            code = int(np.any(d < thresh))
            dr = np.zeros_like(d)
            dr[d >= thresh] = np.sqrt(d[d >= thresh])
            L @= scp.sparse.diags_array(dr)  # Cholesky if PD

        return L, p, code

    return eigen_solvers.cholP(*map_csc_to_eigen(A))


def cpp_qr(
    A: scp.sparse.csc_array,
) -> tuple[scp.sparse.csc_array, scp.sparse.csc_array, list[int], int]:
    """Computes pivoted QR decomposition of ``A``.

    :param A: Some matrix
    :type A: scp.sparse.csc_array
    :return: Matrices Q, R, pivoted column order, and code indicating success
    :rtype: tuple[scp.sparse.csc_array,scp.sparse.csc_array,list[int],int]
    """
    return eigen_solvers.pqr(*map_csc_to_eigen(A))


def cpp_qrr(
    A: scp.sparse.csc_array | np.ndarray,
) -> tuple[scp.sparse.csc_array | np.ndarray, list[int], int, int]:
    """Computes pivoted QR decomposition of ``A`` only returning matrix R and rank estimate

    :param A: Some matrix
    :type A: scp.sparse.csc_array | np.ndarray
    :return: Matrix R, pivoted column order, estimated rank, and code indicating success
    :rtype: tuple[scp.sparse.csc_array | np.ndarray,list[int],int,int]
    """
    if isinstance(A, np.ndarray):
        R, p, r = cpp_dqrr(A)
        return R, p, r, int(r == 0)

    return eigen_solvers.pqrr(*map_csc_to_eigen(A))


def cpp_dqrr(A: np.ndarray) -> tuple[np.ndarray, list[int], int]:
    """Computes pivoted QR decomposition of dense matrix ``A``.

    :param A: Some matrix
    :type A: np.ndarray
    :return: Matrix R column pivot order for rank estimation, estimated rank
    :rtype: tuple[np.ndarray, list[int],int]
    """
    return eigen_solvers.dpqrr(A)


def cpp_symqr(
    A: scp.sparse.csc_array, tol: float
) -> tuple[scp.sparse.csc_array, list[int], list[int], int, int]:
    """Computes pivoted QR decomposition of symmetric matrix ``A``.

    :param A: Some symmetric matrix
    :type A: scp.sparse.csc_array
    :param tol: tolerance for rank estimation
    :type tol: float
    :return: Matrix R, column pivot order for sparsity, column pivot order for rank estimation,
        rank estimate, code indicating success
    :rtype: tuple[scp.sparse.csc_array,list[int],list[int],int,int]
    """
    return eigen_solvers.spqr(*map_csc_to_eigen(A), tol)


def cpp_solve_qr(A: scp.sparse.csc_array) -> tuple[scp.sparse.csc_array, int, int]:
    """Solves ``A@B=I`` for ``B``, where ``A`` is sparse, square, and full rank and ``I`` is an
    identity matrix of suitable dimension via QR decomposition.

    :param A: Some sparse square matrix
    :type A: scp.sparse.csc_array
    :return: ``B`` (inverse of ``A``), estimated rank, and code indicating success
    :rtype: tuple[scp.sparse.csc_array,int,int]
    """
    return eigen_solvers.solve_pqr(*map_csc_to_eigen(A))


def cpp_solve_am(
    y: np.ndarray, X: scp.sparse.csc_array, S: scp.sparse.csc_array
) -> tuple[scp.sparse.csc_array, list[int], np.ndarray, int]:
    """Solves ``(X.T@X + S)@b = X.T@y`` for ``b`` via sparse Cholesky decomposition and computes
    inverse of pivoted Cholesky of ``X.T@X + S``.

    :param y: vector of observations
    :type y: np.ndarray
    :param X: Some rectangular sparse matrix
    :type X: scp.sparse.csc_array
    :param S: Sparse square matrix
    :type S: scp.sparse.csc_array
    :return: Inverse of pivoted Cholesky of ``X.T@X + S``, column pivot indices in a list, ``b``,
        and code indicating success
    :rtype: tuple[scp.sparse.csc_array,list[int],np.ndarray,int]
    """
    return eigen_solvers.solve_am(y, *map_csc_to_eigen(X), *map_csc_to_eigen(S))


def cpp_solve_coef(
    y: np.ndarray, X: scp.sparse.csc_array | np.ndarray, S: scp.sparse.csc_array
) -> tuple[scp.sparse.csc_array | np.ndarray, list[int], np.ndarray, int]:
    """Solves ``(X.T@X + S)@b = X.T@y`` for ``b`` via sparse Cholesky decomposition.

    :param y: vector of observations
    :type y: np.ndarray
    :param X: Some rectangular (sparse) matrix
    :type X: scp.sparse.csc_array | np.ndarray
    :param S: Sparse square matrix
    :type S: scp.sparse.csc_array
    :return: Pivoted Cholesky of ``X.T@X + S``, column pivot indices in a list, ``b``, and code
        indicating success
    :rtype: tuple[scp.sparse.csc_array | np.ndarray, list[int],np.ndarray,int]
    """
    if isinstance(X, np.ndarray):

        L, d, p, b, code = eigen_solvers.dsolve_coef(y, X, *map_csc_to_eigen(S))

        if code == 0:
            # Check PD
            eps = np.power(np.finfo(float).eps, 0.5)
            thresh = eps * np.max(np.abs(d))
            code = int(np.any(d < thresh))
            dr = np.zeros_like(d)
            dr[d >= thresh] = np.sqrt(d[d >= thresh])
            L @= scp.sparse.diags_array(dr)  # Cholesky if PD

        return L, p, b, code

    return eigen_solvers.solve_coef(y, *map_csc_to_eigen(X), *map_csc_to_eigen(S))


def cpp_solve_coef_pqr(
    y: np.ndarray, X: scp.sparse.csc_array | np.ndarray, E: scp.sparse.csc_array
) -> tuple[
    scp.sparse.csc_array | np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int
]:
    """Solves ``(X.T@X + S)@b = X.T@y`` for ``b`` via sparse QR decomposition, where ``E.T@E=S``.

    **Does not form ``X.T@X + S`` for solve**. Potentially pivots twice - once for sparsity (always)
    and then once more whenever algorithm detects a diagonal element that is too small.

    Examples::

       # Solve
       RP,Pr1,Pr2,coef,rank,code = cpp_solve_coef_pqr(yb,Xb,S_root.T.tocsc())

       # Need to get overall pivot...
       P1 = compute_eigen_perm(Pr1)
       P2 = compute_eigen_perm(Pr2)
       P = P2.T@P1.T

       # Need to insert zeroes in case of rank deficiency - first insert nans to that we
       # can then easily find dropped coefs.
       if rank < S_emb.shape[1]:
          coef = np.concatenate((coef,[np.nan for _ in range(S_emb.shape[1]-rank)]))

       # Can now unpivot coef
       coef = coef @ P

       # And identify which coef was dropped
       idx = np.arange(len(coef))
       drop = idx[np.isnan(coef)]
       keep = idx[np.isnan(coef)==False]

       # Now actually set dropped ones to zero
       coef[drop] = 0

       # Convert R so that rest of code can just continue as with Chol (i.e., L)
       LP = RP.T if isinstance(R,np.ndarray) else RP.T.tocsc()

       # Keep only columns of Pr/P that belong to identifiable params. So P.T@LP is Cholesky of
       # negative penalized Hessian of model without unidentifiable coef. Important: LP and Pr/P no
       # longer match dimensions of embedded penalties after this! So we need to keep track of that
       # in the appropriate functions (i.e., `calculate_edf` which calls `compute_B` when called
       # with only LP and not Linv).
       P = P[:,keep]
       _,Pr,_ = translate_sparse(P.tocsc())
       P = compute_eigen_perm(Pr)

    :param y: vector of observations
    :type y: np.ndarray
    :param X: Some rectangular (sparse) matrix
    :type X: scp.sparse.csc_array | np.ndarray
    :param E: Sparse square matrix
    :type E: scp.sparse.csc_array
    :return: Pivoted Cholesky of ``X.T@X + S``, first column pivot indices in an array, second
        column pivot indices in an array, ``b``, estimated rank, and code indicating success.
    :rtype: tuple[scp.sparse.csc_array | np.ndarray,np.ndarray,np.ndarray,np.ndarray,int,int]
    """
    if isinstance(X, np.ndarray):
        R, p, b, r, code = eigen_solvers.dsolve_coef_pqr(
            y, np.concatenate((X, E.toarray()), axis=0)
        )
        # pseudo 2nd pivot vector to maintain compatability
        p2 = np.arange(p.shape[0])
        return R, p, p2, b, r, code

    return eigen_solvers.solve_coef_pqr(y, *map_csc_to_eigen(X), *map_csc_to_eigen(E))


def cpp_solve_coefXX(
    Xy: np.ndarray, XXS: scp.sparse.csc_array | np.ndarray
) -> tuple[scp.sparse.csc_array | np.ndarray, list[int], np.ndarray, int]:
    """Solves ``(X.T@X + S)@b = X.T@y`` for ``b`` via sparse Cholesky decomposition with
    ``(X.T@X + S)`` and ``X.T@y`` pre-computed.

    :param Xy: Holds ``X.T@y``
    :type Xy: np.ndarray
    :param XXS: Holds ``(X.T@X + S)``
    :type XXS: scp.sparse.csc_array | np.ndarray
    :return: Pivoted Cholesky of ``X.T@X + S``, column pivot indices in a list, ``b``, and
        code indicating success
    :rtype: tuple[scp.sparse.csc_array | np.ndarray,list[int],np.ndarray,int]
    """
    if isinstance(XXS, np.ndarray):

        L, d, p, b, code = eigen_solvers.dsolve_coefXX(Xy, XXS)

        if code == 0:
            # Check PD
            eps = np.power(np.finfo(float).eps, 0.5)
            thresh = eps * np.max(np.abs(d))
            code = int(np.any(d < thresh))
            dr = np.zeros_like(d)
            dr[d >= thresh] = np.sqrt(d[d >= thresh])
            L @= scp.sparse.diags_array(dr)  # Cholesky if PD

        return L, p, b, code

    return eigen_solvers.solve_coefXX(Xy, *map_csc_to_eigen(XXS))


def cpp_solve_L(
    X: scp.sparse.csc_array, S: scp.sparse.csc_array
) -> tuple[scp.sparse.csc_array, list[int], int]:
    """Solves ``(X.T@X + S)@B=I`` for ``B``, where ``(X.T@X + S)`` is sparse, symmetric, and full
    rank and ``I`` is an identity matrix of suitable dimension via Cholesky decomposition.

    :param X: Some rectangular sparse matrix
    :type X: scp.sparse.csc_array
    :param S: Sparse square matrix
    :type S: scp.sparse.csc_array
    :return: ``B`` (inverse of **pivoted** ``X.T@X + S``), list of pivot indices, and code
        indicating success
    :rtype: tuple[scp.sparse.csc_array,list[int],int]
    """
    return eigen_solvers.solve_L(*map_csc_to_eigen(X), *map_csc_to_eigen(S))


def cpp_solve_LXX(
    A: scp.sparse.csc_array,
) -> tuple[scp.sparse.csc_array, list[int], int]:
    """Solves ``A@B=I`` for ``B``, where ``A`` is sparse, symmetric, and full rank and ``I`` is an
    identity matrix of suitable dimension via Cholesky decomposition.

    :param A: Some sparse symmetric matrix
    :type A: scp.sparse.csc_array
    :return: ``B`` (inverse of **pivoted** ``A``), list of pivot indices, and code indicating
        success
    :rtype: tuple[scp.sparse.csc_array,list[int],int]
    """
    return eigen_solvers.cpp_solve_LXX(*map_csc_to_eigen(A))


def cpp_solve_tr(
    A: scp.sparse.csc_array | np.ndarray, C: scp.sparse.csc_array | np.ndarray
) -> scp.sparse.csc_array | np.ndarray:
    """Solves ``A@B=C``, where ``A`` is (sparse and) lower triangular. This can be utilized to
    obtain ``B = inv(A)``, when ``C`` is the identity.

    :param A: Lower triangluar sparse matrix
    :type A: scp.sparse.csc_array | np.ndarray
    :param C: Sparse potentially rectangular matrix
    :type C: scp.sparse.csc_array | np.ndarray
    :return: ``B``
    :rtype: scp.sparse.csc_array | np.ndarray
    """

    if isinstance(A, np.ndarray):
        return eigen_solvers.dsolve_tr(
            A, C if isinstance(C, np.ndarray) else C.toarray()
        )

    return eigen_solvers.solve_tr(
        *map_csc_to_eigen(A),
        scp.sparse.csc_array(C) if isinstance(C, np.ndarray) else C,
    )


def cpp_backsolve_tr(
    A: scp.sparse.csc_array | np.ndarray, C: scp.sparse.csc_array | np.ndarray
) -> scp.sparse.csc_array | np.ndarray:
    """Solves ``A@B=C``, where ``A`` (is sparse and) upper triangular. This can be utilized to
    obtain ``B = inv(A)``, when ``C`` is the identity.

    :param A: Lower triangluar sparse matrix
    :type A: scp.sparse.csc_array | np.ndarray
    :param C: Sparse potentially rectangular matrix
    :type C: scp.sparse.csc_array | np.ndarray
    :return: ``B``
    :rtype: scp.sparse.csc_array | np.ndarray
    """
    if isinstance(A, np.ndarray):
        return eigen_solvers.dbacksolve_tr(
            A, C if isinstance(C, np.ndarray) else C.toarray()
        )

    return eigen_solvers.backsolve_tr(
        *map_csc_to_eigen(A),
        scp.sparse.csc_array(C) if isinstance(C, np.ndarray) else C,
    )


def est_condition(
    L: scp.sparse.csc_array | np.ndarray,
    Linv: scp.sparse.csc_array | np.ndarray,
    seed: int | None = 0,
    verbose: bool = True,
) -> tuple[float, float, float, int]:
    """Estimate the condition number ``K`` - the ratio of the largest to smallest singular values -
    of matrix ``A``, where ``A.T@A = L@L.T``.

    ``L`` and ``Linv`` can either be obtained by Cholesky decomposition, i.e., ``A.T@A = L@L.T`` or
    by QR decomposition ``A=Q@R`` where ``R=L.T``.

    If ``verbose=True`` (default), separate warnings will be issued in case
    ``K>(1/(0.5*sqrt(epsilon)))`` and ``K>(1/(0.5*epsilon))``. If the former warning is raised,
    this indicates that computing ``L`` via a Cholesky decomposition is likely unstable
    and should be avoided. If the second warning is raised as well, obtaining ``L`` via QR
    decomposition (of ``A``) is also likely to be unstable (see Golub & Van Loan, 2013).

    References:
      - Cline et al. (1979). An Estimate for the Condition Number of a Matrix.
      - Golub & Van Loan (2013). Matrix computations, 4th edition.

    :param L: Cholesky or any other root of ``A.T@A`` as (a sparse) matrix.
    :type L: scp.sparse.csc_array | np.ndarray
    :param Linv: Inverse of Choleksy (or any other root) of ``A.T@A``.
    :type Linv: scp.sparse.csc_array | np.ndarray
    :param seed: The seed to use for the random parts of the singular value decomposition.
        Defaults to 0.
    :type seed: int or None or numpy.random.Generator
    :param verbose: Whether or not warnings should be printed. Defaults to True.
    :type verbose: bool
    :return: A tuple, containing the estimate of condition number ``K``, an estimate of the largest
        singular value of ``A``, an estimate of the smallest singular value of ``A``, and a
        ``code``. The latter will be zero in case no warning was raised, 1 in case the first
        warning described above was raised, and 2 if the second warning was raised as well.
    :rtype: tuple[float,float,float,int]
    """

    # Get unit round-off (Golub & Van Loan, 2013)
    u = 0.5 * np.finfo(float).eps

    # Now get estimates of largest and smallest singular values of A
    # from norms of L and Linv (Cline et al. 1979)

    try:
        min_sing = (
            np.min(scp.linalg.svd(Linv, compute_uv=False))
            if isinstance(Linv, np.ndarray)
            else scp.sparse.linalg.svds(
                Linv, k=1, return_singular_vectors=False, random_state=seed
            )[0]
        )
        max_sing = (
            np.max(scp.linalg.svd(L, compute_uv=False))
            if isinstance(L, np.ndarray)
            else scp.sparse.linalg.svds(
                L, k=1, return_singular_vectors=False, random_state=seed
            )[0]
        )
    except:  # noqa: E722
        try:
            min_sing = (
                np.min(scp.linalg.svd(Linv, compute_uv=False, lapack_driver="gesvd"))
                if isinstance(Linv, np.ndarray)
                else scp.sparse.linalg.svds(
                    Linv,
                    k=1,
                    return_singular_vectors=False,
                    random_state=seed,
                    solver="lobpcg",
                )[0]
            )
            max_sing = (
                np.max(scp.linalg.svd(L, compute_uv=False, lapack_driver="gesvd"))
                if isinstance(L, np.ndarray)
                else scp.sparse.linalg.svds(
                    L,
                    k=1,
                    return_singular_vectors=False,
                    random_state=seed,
                    solver="lobpcg",
                )[0]
            )
        except:  # noqa: E722
            # Solver failed.. get out
            warnings.warn(
                (
                    "Estimating the condition number of matrix A, where A.T@A=L.T@L failed. This "
                    "can happen but might indicate that something is wrong. Consider estimates "
                    "carefully!"
                )
            )
            return np.inf, np.inf, -np.inf, 1

    K = max_sing * min_sing
    code = 0

    if K > 1 / np.sqrt(u):
        if verbose:
            warnings.warn(
                (
                    "Condition number of matrix A, where A.T@A=L.T@L, is larger than 1/sqrt(u), "
                    "where u is half the machine precision."
                )
            )
        code = 1

    if K > 1 / u:
        if verbose:
            warnings.warn(
                (
                    "Condition number of matrix A, where A.T@A=L.T@L, is larger than 1/u, where u "
                    "is half the machine precision."
                )
            )
        code = 2

    return K, max_sing, 1 / min_sing, code


def compute_block_B_shared(
    address_dat: str,
    address_ptr: str | None,
    address_idx: str | None,
    shape_dat: tuple,
    shape_ptr: tuple | None,
    rows: int,
    cols: int,
    nnz: int | None,
    T: scp.sparse.csc_array,
) -> float:
    """Solves ``L @ B = T`` for ``B`` via forward solving and based on shared memory for ``L``,
    then computes and returns ``B.power(2).sum()``.

    :param address_dat: Address to data array of ``L``
    :type address_dat: str
    :param address_ptr: Address to pointer array of ``L``
    :type address_ptr: str | None
    :param address_idx: Address to indices array of ``L``
    :type address_idx: str | None
    :param shape_dat: Shape of data array of ``L``
    :type shape_dat: tuple
    :param shape_ptr: Shape of pointer array of ``L``
    :type shape_ptr: tuple | None
    :param rows: Number of rows of ``L``
    :type rows: int
    :param cols: Number of cols of ``L``
    :type cols: int
    :param nnz: Number of non-zero elements in ``L``
    :type nnz: int | None
    :param T: Target matrix
    :type T: scp.sparse.csc_array
    :return: ``B.power(2).sum()``
    :rtype: float
    """
    BB = compute_block_linv_shared(
        address_dat, address_ptr, address_idx, shape_dat, shape_ptr, rows, cols, nnz, T
    )
    return np.power(BB, 2).sum() if isinstance(BB, np.ndarray) else BB.power(2).sum()


def compute_block_B_shared_cluster(
    address_dat: str,
    address_ptr: str | None,
    address_idx: str | None,
    shape_dat: tuple,
    shape_ptr: tuple | None,
    rows: int,
    cols: int,
    nnz: int | None,
    T: scp.sparse.csc_array,
    cluster_weights: list[float],
) -> tuple[float, float]:
    """Solves ``L @ B = T`` for ``B`` via forward solving and based on shared memory for ``L``,
    then computes and returns ``sum(B.power(2).sum()*cluster_weights)`` and
    ``B.power(2).sum()*len(cluster_weights)``.

    :param address_dat: Address to data array of ``L``
    :type address_dat: str
    :param address_ptr: Address to pointer array of ``L``
    :type address_ptr: str | None
    :param address_idx: Address to indices array of ``L``
    :type address_idx: str | None
    :param shape_dat: Shape of data array of ``L``
    :type shape_dat: tuple
    :param shape_ptr: Shape of pointer array of ``L``
    :type shape_ptr: tuple | None
    :param rows: Number of rows of ``L``
    :type rows: int
    :param cols: Number of cols of ``L``
    :type cols: int
    :param nnz: Number of non-zero elements in ``L``
    :type nnz: int | None
    :param T: Target matrix
    :type T: scp.sparse.csc_array
    :param cluster_weights: Cluster weights obtained from
        :func:`mssm.src.python.formula.__cluster_discretize`.
    :type cluster_weights: list[float]
    :return: ``sum(B.power(2).sum()*cluster_weights)`` and
        ``B.power(2).sum()*len(cluster_weights)``
    :rtype: tuple[float,float]
    """
    BB = compute_block_linv_shared(
        address_dat, address_ptr, address_idx, shape_dat, shape_ptr, rows, cols, nnz, T
    )
    BBps = np.power(BB, 2).sum() if isinstance(BB, np.ndarray) else BB.power(2).sum()
    return np.sum(cluster_weights * BBps), len(cluster_weights) * BBps


def compute_B(
    L: scp.sparse.csc_array | np.ndarray,
    P: scp.sparse.csc_array,
    lTerm: LambdaTerm,
    n_c: int = 10,
    drop: np.typing.NDArray[np.int_] | None = None,
) -> float | tuple[float, float]:
    """Solves ``L @ B = P @ lTerm.D_J_emb`` for ``B``, then returns ``B.power(2).sum()`` or two
    approximations of this (for very big factor smooth models).

    :param L: Lower triangular (sparse) matrix
    :type L: scp.sparse.csc_array | np.ndarray
    :param P: Permuation matrix
    :type P: scp.sparse.csc_array
    :param lTerm: Current penalty term
    :type lTerm: LambdaTerm
    :param n_c: Number of cores, defaults to 10
    :type n_c: int, optional
    :param drop: Array of parameters (columns/rows of ``lTerm.D_J_emb``) to drop, defaults to None
    :type drop: np.typing.NDArray[np.int_] | None, optional
    :return: ``sum(B.power(2).sum()`` or  ``sum(B.power(2).sum()*cluster_weights)`` and
        ``B.power(2).sum()*len(cluster_weights)`` with cluster weights obtained from
        :func:`mssm.src.python.formula.__cluster_discretize`.
    :rtype: float | tuple[float, float]
    """
    # Solves L @ B = P @ D for B, parallelizing over column
    # blocks of D if int(D.shape[1]/2000) > 1

    # Also allows for approximate B computation for very big factor smooths.
    D_start = lTerm.start_index
    idx = np.arange(lTerm.S_J_emb.shape[1])
    if drop is None:
        drop = np.array([])
    keep = idx[np.isin(idx, drop) == False]  # noqa: E712

    if lTerm.clust_series is None:

        col_sums = lTerm.S_J.sum(axis=0)
        if lTerm.type == PenType.NULL and sum(col_sums[col_sums > 0]) == 1:
            # Null penalty for factor smooth has usually only non-zero element in a single colum,
            # so we only need to solve one linear system per level of the factor smooth.
            NULL_idx = np.argmax(col_sums)

            D_idx = np.arange(
                lTerm.start_index + NULL_idx,
                lTerm.S_J.shape[1] * (lTerm.rep_sj + 1),
                lTerm.S_J.shape[1],
            )

            D_len = len(D_idx)
            PD = P @ lTerm.D_J_emb[:, D_idx][keep, :]
        else:
            # First get columns associated to penalty
            D_len = lTerm.rep_sj * lTerm.S_J.shape[1]
            D_end = lTerm.start_index + D_len
            D_idx = idx[D_start:D_end]

            # Now check if dropped column is included, if so remove and update length.
            D_idx = D_idx[np.isin(D_idx, drop) == False]  # noqa: E712
            D_len = len(D_idx)
            PD = P @ lTerm.D_J_emb[:, D_idx][keep, :]

        D_r = int(D_len / 2000)

        if D_r > 1 and n_c > 1:
            # Parallelize over column blocks of P @ D
            # Can speed up computations considerably and is feasible memory-wise
            # since L itself is super sparse.
            n_c = min(D_r, n_c)
            split = np.array_split(range(D_len), n_c)
            PD = P @ lTerm.D_J_emb[:, D_idx][keep, :]
            PDs = [PD[:, split[i]] for i in range(n_c)]

            with (
                managers.SharedMemoryManager() as manager,
                mp.Pool(processes=n_c) as pool,
            ):
                # Create shared memory copies of data, indptr, and indices
                if isinstance(L, np.ndarray):
                    # dense case
                    rows, cols = L.shape
                    shape_dat = L.shape

                    dat_mem = manager.SharedMemory(L.nbytes)
                    dat_shared = np.ndarray(
                        shape_dat, dtype=np.double, buffer=dat_mem.buf
                    )
                    dat_shared[:] = L[:]

                    shape_ptr = None
                    nnz = None
                    ptr_mem = None
                    idx_mem = None

                else:
                    rows, cols, nnz, data, indptr, indices = map_csc_to_eigen(L)
                    shape_dat = data.shape
                    shape_ptr = indptr.shape

                    dat_mem = manager.SharedMemory(data.nbytes)
                    dat_shared = np.ndarray(
                        shape_dat, dtype=np.double, buffer=dat_mem.buf
                    )
                    dat_shared[:] = data[:]

                    ptr_mem = manager.SharedMemory(indptr.nbytes)
                    ptr_shared = np.ndarray(
                        shape_ptr, dtype=np.int64, buffer=ptr_mem.buf
                    )
                    ptr_shared[:] = indptr[:]

                    idx_mem = manager.SharedMemory(indices.nbytes)
                    idx_shared = np.ndarray(
                        shape_dat, dtype=np.int64, buffer=idx_mem.buf
                    )
                    idx_shared[:] = indices[:]

                args = zip(
                    repeat(dat_mem.name),
                    repeat(None if ptr_mem is None else ptr_mem.name),
                    repeat(None if idx_mem is None else idx_mem.name),
                    repeat(shape_dat),
                    repeat(shape_ptr),
                    repeat(rows),
                    repeat(cols),
                    repeat(nnz),
                    PDs,
                )

                pow_sums = pool.starmap(compute_block_B_shared, args)

            return sum(pow_sums)

        # Not worth parallelizing, solve directly
        B = cpp_solve_tr(L, PD)
        return np.power(B, 2).sum() if isinstance(B, np.ndarray) else B.power(2).sum()

    # Approximate the derivative based just on the columns in D_J that belong to the
    # maximum series identified for each cluster. Use the size of the cluster and the weights to
    # correct for the fact that all series in the cluster are slightly different after all.
    if len(drop) > 0:
        raise ValueError(
            "Approximate derivative computation cannot currently handle unidentifiable terms."
        )

    n_coef = lTerm.S_J.shape[0]
    rank = int(lTerm.rank / lTerm.rep_sj)

    sum_bs_lw = 0
    sum_bs_up = 0

    targets = [
        P
        @ lTerm.D_J_emb[
            :,
            (D_start + (s * n_coef)) : (  # noqa: E203
                D_start + ((s + 1) * n_coef) - (n_coef - rank)
            ),
        ]
        for s in lTerm.clust_series
    ]

    if len(targets) < 20 * n_c:

        for weights, target in zip(lTerm.clust_weights, targets):

            BB = cpp_solve_tr(L, target)
            BBps = BB.power(2).sum()
            sum_bs_lw += np.sum(weights * BBps)
            sum_bs_up += len(weights) * BBps

    else:
        # Parallelize
        with managers.SharedMemoryManager() as manager, mp.Pool(processes=n_c) as pool:
            # Create shared memory copies of data, indptr, and indices

            if isinstance(L, np.ndarray):
                # dense case
                rows, cols = L.shape
                shape_dat = L.shape

                dat_mem = manager.SharedMemory(L.nbytes)
                dat_shared = np.ndarray(shape_dat, dtype=np.double, buffer=dat_mem.buf)
                dat_shared[:] = L[:]

                shape_ptr = None
                nnz = None
                ptr_mem = None
                idx_mem = None

            else:
                rows, cols, nnz, data, indptr, indices = map_csc_to_eigen(L)
                shape_dat = data.shape
                shape_ptr = indptr.shape

                dat_mem = manager.SharedMemory(data.nbytes)
                dat_shared = np.ndarray(shape_dat, dtype=np.double, buffer=dat_mem.buf)
                dat_shared[:] = data[:]

                ptr_mem = manager.SharedMemory(indptr.nbytes)
                ptr_shared = np.ndarray(shape_ptr, dtype=np.int64, buffer=ptr_mem.buf)
                ptr_shared[:] = indptr[:]

                idx_mem = manager.SharedMemory(indices.nbytes)
                idx_shared = np.ndarray(shape_dat, dtype=np.int64, buffer=idx_mem.buf)
                idx_shared[:] = indices[:]

            args = zip(
                repeat(dat_mem.name),
                repeat(None if ptr_mem is None else ptr_mem.name),
                repeat(None if idx_mem is None else idx_mem.name),
                repeat(shape_dat),
                repeat(shape_ptr),
                repeat(rows),
                repeat(cols),
                repeat(nnz),
                targets,
                lTerm.clust_weights,
            )

            sum_bs_lw_all, sum_bs_up_all = zip(
                *pool.starmap(compute_block_B_shared_cluster, args)
            )

            sum_bs_lw = np.sum(sum_bs_lw_all)
            sum_bs_up = np.sum(sum_bs_up_all)

    return sum_bs_lw, sum_bs_up


def compute_block_linv_shared(
    address_dat: str,
    address_ptr: str | None,
    address_idx: str | None,
    shape_dat: tuple,
    shape_ptr: tuple | None,
    rows: int,
    cols: int,
    nnz: int | None,
    T: scp.sparse.csc_array,
) -> scp.sparse.csc_array | np.ndarray:
    """Solves ``L@B = T`` where ``L`` is available in shared memory and ``T`` is a column subset of
    the identity matrix.

    :param address_dat: Address to data array of ``L``
    :type address_dat: str
    :param address_ptr: Address to pointer array of ``L``
    :type address_ptr: str | None
    :param address_idx: Address to indices array of ``L``
    :type address_idx: str | None
    :param shape_dat: Shape of data array of ``L``
    :type shape_dat: tuple
    :param shape_ptr: Shape of pointer array of ``L``
    :type shape_ptr: tuple | None
    :param rows: Number of rows of ``L``
    :type rows: int
    :param cols: Number of cols of ``L``
    :type cols: int
    :param nnz: Number of non-zero elements in ``L``
    :type nnz: int | None
    :param T: Target matrix
    :type T: scp.sparse.csc_array
    :return: ``B``
    :rtype: scp.sparse.csc_array | np.ndarray
    """
    if nnz is None:
        # dense case
        dat_shared = shared_memory.SharedMemory(name=address_dat, create=False)
        L = np.ndarray(shape_dat, dtype=np.double, buffer=dat_shared.buf)
        B = cpp_solve_tr(L, T)
    else:
        dat_shared = shared_memory.SharedMemory(name=address_dat, create=False)
        ptr_shared = shared_memory.SharedMemory(name=address_ptr, create=False)
        idx_shared = shared_memory.SharedMemory(name=address_idx, create=False)

        data = np.ndarray(shape_dat, dtype=np.double, buffer=dat_shared.buf)
        indptr = np.ndarray(shape_ptr, dtype=np.int64, buffer=ptr_shared.buf)
        indices = np.ndarray(shape_dat, dtype=np.int64, buffer=idx_shared.buf)

        B = eigen_solvers.solve_tr(rows, cols, nnz, data, indptr, indices, T)

    return B


def compute_Linv(
    L: scp.sparse.csc_array | np.ndarray, n_c: int = 10
) -> scp.sparse.csc_array | np.ndarray:
    """Solves ``L @ inv(L) = I`` for ``inv(L)`` optionally parallelizing over column blocks of
    ``I``.

    :param L: Lower triangular (sparse) matrix
    :type L: scp.sparse.csc_array | np.ndarray
    :param n_c: Number of cores to use, defaults to 10
    :type n_c: int, optional
    :return: ``inv(L)``
    :rtype: scp.sparse.csc_array | np.ndarray
    """
    # Solves L @ inv(L) = I for inv(L) parallelizing over column
    # blocks of I if int(I.shape[1]/2000) > 1

    n_col = L.shape[1]
    r = int(n_col / 2000)
    T = scp.sparse.eye(n_col, format="csc")
    if r > 1 and n_c > 1:
        # Parallelize over column blocks of I
        # Can speed up computations considerably and is feasible memory-wise
        # since L itself is super sparse.

        n_c = min(r, n_c)
        split = np.array_split(range(n_col), n_c)
        Ts = [T[:, split[i]] for i in range(n_c)]

        with managers.SharedMemoryManager() as manager, mp.Pool(processes=n_c) as pool:
            # Create shared memory copies of data, indptr, and indices
            if isinstance(L, np.ndarray):
                # dense case
                rows, cols = L.shape
                shape_dat = L.shape

                dat_mem = manager.SharedMemory(L.nbytes)
                dat_shared = np.ndarray(shape_dat, dtype=np.double, buffer=dat_mem.buf)
                dat_shared[:] = L[:]

                shape_ptr = None
                nnz = None
                ptr_mem = None
                idx_mem = None

            else:
                rows, cols, nnz, data, indptr, indices = map_csc_to_eigen(L)
                shape_dat = data.shape
                shape_ptr = indptr.shape

                dat_mem = manager.SharedMemory(data.nbytes)
                dat_shared = np.ndarray(shape_dat, dtype=np.double, buffer=dat_mem.buf)
                dat_shared[:] = data[:]

                ptr_mem = manager.SharedMemory(indptr.nbytes)
                ptr_shared = np.ndarray(shape_ptr, dtype=np.int64, buffer=ptr_mem.buf)
                ptr_shared[:] = indptr[:]

                idx_mem = manager.SharedMemory(indices.nbytes)
                idx_shared = np.ndarray(shape_dat, dtype=np.int64, buffer=idx_mem.buf)
                idx_shared[:] = indices[:]

            args = zip(
                repeat(dat_mem.name),
                repeat(None if ptr_mem is None else ptr_mem.name),
                repeat(None if idx_mem is None else idx_mem.name),
                repeat(shape_dat),
                repeat(shape_ptr),
                repeat(rows),
                repeat(cols),
                repeat(nnz),
                Ts,
            )

            LBinvs = pool.starmap(compute_block_linv_shared, args)

        return (
            np.concatenate(LBinvs, axis=1) if nnz is None else scp.sparse.hstack(LBinvs)
        )

    return cpp_solve_tr(L, T)
