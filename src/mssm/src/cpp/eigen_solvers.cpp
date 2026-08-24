#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Householder>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <limits>


namespace py = pybind11;

typedef Eigen::Vector<long long int, Eigen::Dynamic> VectorXi64;

std::tuple<
    Eigen::MatrixXd,
    int
> dchol(
    const Eigen::Ref<Eigen::MatrixXd,0,Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &A
)
{
    // Compute A= LL.T for dense case

    // Compute LDLT decomposition
    Eigen::LLT<Eigen::MatrixXd> solver(A);

    // Check Failure
    if (solver.info()!=Eigen::Success)
    {
        Eigen::MatrixXd L(A.cols(),A.cols());
        L.setIdentity();
        Eigen::VectorXd d;
        d.setZero(A.cols());
        return std::make_tuple(std::move(L),1);
    }

    // Get the L matrix as a dense MatrixXd
    Eigen::MatrixXd L = solver.matrixL();

    return std::make_tuple(std::move(L),0);

}

std::tuple<Eigen::SparseMatrix<double,0,long long int>,int> chol(long long int Arows, long long int Acols, long long int Annz,
                                                 py::array_t<double, py::array::f_style | py::array::forcecast> Adata,
                                                 py::array_t<long long int, py::array::f_style | py::array::forcecast> Aidptr,
                                                 py::array_t<long long int, py::array::f_style | py::array::forcecast> Aindices){
    

    // Map idea based on: https://github.com/fwilliams/numpyeigen/blob/master/src/npe_sparse_array.h#L74
    // 22.06.24: changed type of Arows, Acols, Annz to long long int since int was just np.int32, which would not work for huge models.
    // Important, template parameters for SparseMatrix also had to be set to long long int, since the default was int.
    // see: https://eigen.tuxfamily.org/dox/classEigen_1_1SparseMatrix.html
    Eigen::Map<Eigen::SparseMatrix<double,0,long long int>> A(Arows,Acols,Annz,
                                                    (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Aidptr.data(),
                                                    (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Aindices.data(),
                                                    (Eigen::SparseMatrix<double,0,long long int>::Scalar*) Adata.data());

    // We prevent any sparsity preserving ordering, since we need the un-pivoted factor L so that L * L' = A
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double,0,long long int>,Eigen::Lower,Eigen::NaturalOrdering<long long int>> solver;
    solver.compute(A);

    if (solver.info()!=Eigen::Success)
    {
        Eigen::SparseMatrix<double,0,long long int> id(Arows,Acols);
        id.setIdentity();
        return std::make_tuple(std::move(id),1);
    }

    Eigen::SparseMatrix<double,0,long long int> L = solver.matrixL();
    
    return std::make_tuple(std::move(L),0);
}

std::tuple<
    Eigen::MatrixXd,
    Eigen::VectorXd,
    Eigen::VectorXi,
    int
> dcholP(
    const Eigen::Ref<Eigen::MatrixXd,0,Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &A
)
{
    // Compute A= PLDLP.T decomposition of dense matrix A and return L, D as a vector and the
    // column indices of P.

    // Compute LDLT decomposition
    Eigen::LDLT<Eigen::MatrixXd> solver(A);

    // Also get the permutation
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> P(solver.transpositionsP());

    // Check Failure
    if (solver.info()!=Eigen::Success)
    {
        Eigen::MatrixXd L(A.cols(),A.cols());
        L.setIdentity();
        Eigen::VectorXd d;
        d.setZero(A.cols());
        return std::make_tuple(std::move(L),std::move(d),P.indices(),1);
    }

    // Get the diagonal of D as a VectorXd
    Eigen::VectorXd d = solver.vectorD();

    // Get the L matrix as a dense MatrixXd
    Eigen::MatrixXd L = solver.matrixL();

    return std::make_tuple(std::move(L),std::move(d),P.indices(),0);

}

std::tuple<Eigen::SparseMatrix<double,0,long long int>,VectorXi64,int> cholP(long long int Arows, long long int Acols, long long int Annz,
                                                                  py::array_t<double, py::array::f_style | py::array::forcecast> Adata,
                                                                  py::array_t<long long int, py::array::f_style | py::array::forcecast> Aidptr,
                                                                  py::array_t<long long int, py::array::f_style | py::array::forcecast> Aindices){

    Eigen::Map<Eigen::SparseMatrix<double,0,long long int>> A(Arows,Acols,Annz,
                                                    (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Aidptr.data(),
                                                    (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Aindices.data(),
                                                    (Eigen::SparseMatrix<double,0,long long int>::Scalar*) Adata.data());
    // Like chol() but with sparsity preserving pivoting
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double,0,long long int>> solver;
    solver.analyzePattern(A.selfadjointView<Eigen::Lower>());
    solver.factorize(A);

    // Also get the permutation
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic,long long int> P(solver.permutationP());

    if (solver.info()!=Eigen::Success)
    {
        Eigen::SparseMatrix<double> id(Arows,Acols);
        id.setIdentity();
        return std::make_tuple(std::move(id),P.indices(),1);
    }

    Eigen::SparseMatrix<double,0,long long int> L = solver.matrixL();
    
    return std::make_tuple(std::move(L),P.indices(),0);
}

std::tuple<Eigen::SparseMatrix<double,0,long long int>,Eigen::SparseMatrix<double,0,long long int>,VectorXi64, int> pqr(long long int Arows, long long int Acols, long long int Annz,
                                                                                             py::array_t<double, py::array::f_style | py::array::forcecast> Adata,
                                                                                             py::array_t<long long int, py::array::f_style | py::array::forcecast> Aidptr,
                                                                                             py::array_t<long long int, py::array::f_style | py::array::forcecast> Aindices){

    Eigen::Map<Eigen::SparseMatrix<double,0,long long int>> A(Arows,Acols,Annz,
                                                    (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Aidptr.data(),
                                                    (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Aindices.data(),
                                                    (Eigen::SparseMatrix<double,0,long long int>::Scalar*) Adata.data());
    // Computed column-pivoted QR factorization of A.
    Eigen::SparseQR<Eigen::SparseMatrix<double,0,long long int>,Eigen::COLAMDOrdering<long long int>> solver;
    solver.setPivotThreshold(sqrt(std::numeric_limits<double>::epsilon())*A.norm());
    solver.compute(A);

    // Column permutation matrix
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic, long long int> P(solver.colsPermutation());

    if(solver.info()!=Eigen::Success)
    {
        Eigen::SparseMatrix<double,0,long long int> Q(Arows,Acols);
        Q.setIdentity();

        Eigen::SparseMatrix<double,0,long long int> R(Arows,Acols);
        R.setIdentity();
        return std::make_tuple(std::move(Q),std::move(R),P.indices(),1);
    }

    // see: https://eigen.tuxfamily.org/dox/classEigen_1_1SparseQR.html
    Eigen::SparseMatrix<double,0,long long int> Q;
    Q = solver.matrixQ();

    // Upper triagonal factor
    Eigen::SparseMatrix<double,0,long long int> R = solver.matrixR();

    // Upper triagonal factor after applying the permuation.
    //Eigen::SparseMatrix<double> R = solver.matrixR().eval() * P.transpose();

    return std::make_tuple(std::move(Q),R,P.indices(),0);
    
}

std::tuple<Eigen::SparseMatrix<double,0,long long int>,VectorXi64,int,int> pqrr(long long int Arows, long long int Acols, long long int Annz,
                                                                                             py::array_t<double, py::array::f_style | py::array::forcecast> Adata,
                                                                                             py::array_t<long long int, py::array::f_style | py::array::forcecast> Aidptr,
                                                                                             py::array_t<long long int, py::array::f_style | py::array::forcecast> Aindices){

    Eigen::Map<Eigen::SparseMatrix<double,0,long long int>> A(Arows,Acols,Annz,
                                                    (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Aidptr.data(),
                                                    (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Aindices.data(),
                                                    (Eigen::SparseMatrix<double,0,long long int>::Scalar*) Adata.data());
    // Computed column-pivoted QR factorization of A.
    Eigen::SparseQR<Eigen::SparseMatrix<double,0,long long int>,Eigen::COLAMDOrdering<long long int>> solver;
    solver.setPivotThreshold(sqrt(std::numeric_limits<double>::epsilon())*A.norm());
    solver.compute(A);

    // Column permutation matrix
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic, long long int> P(solver.colsPermutation());

    if(solver.info()!=Eigen::Success)
    {
        Eigen::SparseMatrix<double,0,long long int> R(Arows,Acols);
        R.setIdentity();
        return std::make_tuple(std::move(R),P.indices(),0,1);
    }

    // Upper triagonal factor before applying the permuation.
    Eigen::SparseMatrix<double,0,long long int> R = solver.matrixR().topLeftCorner(solver.rank(), solver.rank());//.eval() * P.transpose();

    return std::make_tuple(std::move(R),P.indices(),solver.rank(),0);
    
}

std::tuple<
    Eigen::MatrixXd,
    Eigen::VectorXi,
    int
> dpqrr(
    const Eigen::Ref<Eigen::MatrixXd,0,Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &A
)
{
    // Rank revealing QR decomposition of **dense** matrix A. Only pivot and rank is returned
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver;
    solver.compute(A);

    if(solver.info()!=Eigen::Success)
    {
        Eigen::MatrixXd R(A.cols(),A.cols());
        R.setIdentity();
        return std::make_tuple(std::move(R),solver.colsPermutation().indices(),0);
    }

    Eigen::MatrixXd R = solver.matrixR().topLeftCorner(solver.rank(), solver.rank()).template triangularView<Eigen::Upper>();

    return std::make_tuple(std::move(R),solver.colsPermutation().indices(),solver.rank());
}

std::tuple<Eigen::SparseMatrix<double,0,long long int>,VectorXi64, VectorXi64, int,int> spqr(long long int Arows, long long int Acols, long long int Annz,
                                                                                             py::array_t<double, py::array::f_style | py::array::forcecast> Adata,
                                                                                             py::array_t<long long int, py::array::f_style | py::array::forcecast> Aidptr,
                                                                                             py::array_t<long long int, py::array::f_style | py::array::forcecast> Aindices,
                                                                                             double piv_tol){

    Eigen::Map<Eigen::SparseMatrix<double,0,long long int>> A(Arows,Acols,Annz,
                                                    (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Aidptr.data(),
                                                    (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Aindices.data(),
                                                    (Eigen::SparseMatrix<double,0,long long int>::Scalar*) Adata.data());

    // Computed column-pivoted QR factorization of symmetric matrix A, with ordering computed so that L - where A=L@L.T - is sparse.
    // see Golub & Van Loan "Matrix Computations: 4ED" (2013)
    Eigen::AMDOrdering<long long int> ordering;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic,long long int> P1;
    ordering(A.selfadjointView<Eigen::Lower>(), P1);
    
    // Now permute A columns with P1 - then compute QR decomposition A@P1@P2 = QR
    // where P2 will be formed with concern for numerical stability if piv_tol << 0.5
    Eigen::SparseQR<Eigen::SparseMatrix<double,0,long long int>,Eigen::NaturalOrdering<long long int>> solver;
    solver.setPivotThreshold(piv_tol*sqrt(A.bottomLeftCorner(Acols,Acols).eval().diagonal().array().abs().maxCoeff())); // Find root of absolute maximum on diagonal of XXS and use that for thresholding.
    solver.compute(A*P1); // Now use ordering computed previously to pivot columns

    // Get second column permutation matrix
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic,long long int> P2(solver.colsPermutation());

    if(solver.info()!=Eigen::Success)
    {

        Eigen::SparseMatrix<double> R(Acols,Acols);
        R.setIdentity();
        return std::make_tuple(std::move(R),P1.indices(),P2.indices(),0,1);
    }

    // Upper triagonal factor
    Eigen::SparseMatrix<double> R = solver.matrixR().topLeftCorner(solver.rank(), solver.rank());


    return std::make_tuple(std::move(R),P1.indices(),P2.indices(),solver.rank(),0);
    
}

std::tuple<Eigen::SparseMatrix<double,0,long long int>, int, int> solve_pqr(long long int Arows, long long int Acols, long long int Annz,
                                                       py::array_t<double, py::array::f_style | py::array::forcecast> Adata,
                                                       py::array_t<long long int, py::array::f_style | py::array::forcecast> Aidptr,
                                                       py::array_t<long long int, py::array::f_style | py::array::forcecast> Aindices){

    Eigen::Map<Eigen::SparseMatrix<double,0,long long int>> A(Arows,Acols,Annz,
                                                    (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Aidptr.data(),
                                                    (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Aindices.data(),
                                                    (Eigen::SparseMatrix<double,0,long long int>::Scalar*) Adata.data());

    // Computed column-pivoted QR factorization of A and solve A @ B = I for B (inverse of A)
    Eigen::SparseQR<Eigen::SparseMatrix<double,0,long long int>,Eigen::AMDOrdering<long long int>> solver;
    solver.analyzePattern(A.selfadjointView<Eigen::Lower>());
    solver.factorize(A);

    // Also setup identity target for inverse of A
    Eigen::SparseMatrix<double,0,long long int> id(Acols,Acols);
    id.setIdentity();

    if(solver.info()!=Eigen::Success)
    {
        
        return std::make_tuple(std::move(id),0,1);
    }

    // see: https://eigen.tuxfamily.org/dox/classEigen_1_1SparseQR.html
    Eigen::SparseMatrix<double,0,long long int> invA(Acols,Acols);
    invA = solver.solve(id);

    if(solver.info()!=Eigen::Success)
    {
        
        return std::make_tuple(std::move(id),0,1);
    }

    return std::make_tuple(std::move(invA),solver.rank(),0);
    
}

std::tuple<Eigen::SparseMatrix<double,0,long long int>,VectorXi64,Eigen::VectorXd,int> solve_am(Eigen::VectorXd y, long long int Xrows, long long int Xcols, long long int Xnnz,
                                                                                     py::array_t<double, py::array::f_style | py::array::forcecast> Xdata,
                                                                                     py::array_t<long long int, py::array::f_style | py::array::forcecast> Xidptr,
                                                                                     py::array_t<long long int, py::array::f_style | py::array::forcecast> Xindices,
                                                                                     long long int Srows, long long int Scols, long long int Snnz,
                                                                                     py::array_t<double, py::array::f_style | py::array::forcecast> Sdata,
                                                                                     py::array_t<long long int, py::array::f_style | py::array::forcecast> Sidptr,
                                                                                     py::array_t<long long int, py::array::f_style | py::array::forcecast> Sindices){
    // Permuted Cholesky:
    // P * A * P' = L * L'
    // A = P' * L * L' * P
    // U = P' * L
    // U' = L' * P
    // A = U * U'
    // Inverse:
    // inv(A) = P' * Inv(L)' * inv(L) * Perm

    Eigen::Map<Eigen::SparseMatrix<double,0,long long int>> X(Xrows,Xcols,Xnnz,
                                              (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Xidptr.data(),
                                              (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Xindices.data(),
                                              (Eigen::SparseMatrix<double,0,long long int>::Scalar*) Xdata.data());

    Eigen::Map<Eigen::SparseMatrix<double,0,long long int>> S(Srows,Scols,Snnz,
                                              (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Sidptr.data(),
                                              (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Sindices.data(),
                                              (Eigen::SparseMatrix<double,0,long long int>::Scalar*) Sdata.data());

    // Prepare and compute Cholesky factor of X' * X + S
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double,0,long long int>> solver;
    solver.compute(X.transpose() * X + S);

    // Initialize coef vector
    Eigen::VectorXd coef;
    coef.setZero(Xcols);

    // Also setup identity target for inverse of L' (see below)
    Eigen::SparseMatrix<double,0,long long int> id(Xcols,Xcols);
    id.setIdentity();

    // We also need inv(L) * P from P * X' * X + S * P' = L * L'
    // so the inverse of the lower matrix from the solver times the
    // permutation matrix created for us by eigen.

    // First get the permutation
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic, long long int> P(solver.permutationP());

    if (solver.info()!=Eigen::Success)
    {

        return std::make_tuple(std::move(id),P.indices(),std::move(coef),1);
    }

    // Solve for coef
    coef = solver.solve(X.transpose() * y);

    if (solver.info()!=Eigen::Success)
    {

        return std::make_tuple(std::move(id),P.indices(),std::move(coef),2);
    }

    // Now get inv(L)
    solver.matrixL().solveInPlace(id);

    return std::make_tuple(std::move(id),P.indices(),std::move(coef),0);
}

std::tuple<Eigen::SparseMatrix<double,0,long long int>,VectorXi64,int> solve_L(long long int Xrows, long long int Xcols, long long int Xnnz,
                                                                    py::array_t<double, py::array::f_style | py::array::forcecast> Xdata,
                                                                    py::array_t<long long int, py::array::f_style | py::array::forcecast> Xidptr,
                                                                    py::array_t<long long int, py::array::f_style | py::array::forcecast> Xindices,
                                                                    long long int Srows, long long int Scols, long long int Snnz,
                                                                    py::array_t<double, py::array::f_style | py::array::forcecast> Sdata,
                                                                    py::array_t<long long int, py::array::f_style | py::array::forcecast> Sidptr,
                                                                    py::array_t<long long int, py::array::f_style | py::array::forcecast> Sindices){
    // Permuted Cholesky:
    // P * A * P' = L * L'
    // A = P' * L * L' * P
    // U = P' * L
    // U' = L' * P
    // A = U * U'
    // Inverse:
    // inv(A) = P' * Inv(L)' * inv(L) * Perm

    Eigen::Map<Eigen::SparseMatrix<double,0,long long int>> X(Xrows,Xcols,Xnnz,
                                              (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Xidptr.data(),
                                              (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Xindices.data(),
                                              (Eigen::SparseMatrix<double,0,long long int>::Scalar*) Xdata.data());

    Eigen::Map<Eigen::SparseMatrix<double,0,long long int>> S(Srows,Scols,Snnz,
                                              (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Sidptr.data(),
                                              (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Sindices.data(),
                                              (Eigen::SparseMatrix<double,0,long long int>::Scalar*) Sdata.data());

    // Prepare and compute Cholesky factor of X' * X + S
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double,0,long long int>> solver;
    solver.compute(X.transpose() * X + S);

    // Setup identity target for inverse of L' (see below)
    Eigen::SparseMatrix<double,0,long long int> id(Xcols,Xcols);
    id.setIdentity();

    // Also get the permutation
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic, long long int> P(solver.permutationP());

    if (solver.info()!=Eigen::Success)
    {

        return std::make_tuple(std::move(id),P.indices(),1);
    }

    // We need inv(L) * P from P * X' * X + S * P' = L * L'
    // so the inverse of the lower matrix from the solver times the
    // permutation matrix created for us by eigen (last part is done in Python).
    solver.matrixL().solveInPlace(id);

    return std::make_tuple(std::move(id),P.indices(),0);
}

std::tuple<Eigen::SparseMatrix<double,0,long long int>,VectorXi64,int> solve_LXX(long long int Xrows, long long int Xcols, long long int Xnnz,
                                                                      py::array_t<double, py::array::f_style | py::array::forcecast> Xdata,
                                                                      py::array_t<long long int, py::array::f_style | py::array::forcecast> Xidptr,
                                                                      py::array_t<long long int, py::array::f_style | py::array::forcecast> Xindices){
    // Permuted Cholesky:
    // P * A * P' = L * L'
    // A = P' * L * L' * P
    // U = P' * L
    // U' = L' * P
    // A = U * U'
    // Inverse:
    // inv(A) = P' * Inv(L)' * inv(L) * Perm

    Eigen::Map<Eigen::SparseMatrix<double,0,long long int>> XX(Xrows,Xcols,Xnnz,
                                              (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Xidptr.data(),
                                              (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Xindices.data(),
                                              (Eigen::SparseMatrix<double,0,long long int>::Scalar*) Xdata.data());

    // Prepare and compute Cholesky factor of X' * X + S or X' * X 
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double,0,long long int>> solver;
    solver.analyzePattern(XX.selfadjointView<Eigen::Lower>());
    solver.factorize(XX);

    // Setup identity target for inverse of L' (see below)
    Eigen::SparseMatrix<double,0,long long int> id(Xcols,Xcols);
    id.setIdentity();

    // Also get the permutation
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic, long long int> P(solver.permutationP());

    if (solver.info()!=Eigen::Success)
    {

        return std::make_tuple(std::move(id),P.indices(),1);
    }

    // We need inv(L) * P from P * X' * X + S * P' = L * L'
    // so the inverse of the lower matrix from the solver times the
    // permutation matrix created for us by eigen (last part is done in Python).
    solver.matrixL().solveInPlace(id);

    return std::make_tuple(std::move(id),P.indices(),0);
}

std::tuple<
    Eigen::MatrixXd,
    Eigen::VectorXd,
    Eigen::VectorXi,
    Eigen::VectorXd,
    int
> dsolve_coef(
    const Eigen::Ref<Eigen::VectorXd> &y,
    const Eigen::Ref<Eigen::MatrixXd,0,Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &X,
    long long int Srows, long long int Scols, long long int Snnz,
    py::array_t<double, py::array::f_style | py::array::forcecast> Sdata,
    py::array_t<long long int, py::array::f_style | py::array::forcecast> Sidptr,
    py::array_t<long long int, py::array::f_style | py::array::forcecast> Sindices
)
{
    // Stability pivoted LDL for solve

    // get S
    Eigen::Map<
        Eigen::SparseMatrix<
            double,0,long long int
        >
    > S(Srows,Scols,Snnz,
        (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Sidptr.data(),
        (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Sindices.data(),
        (Eigen::SparseMatrix<double,0,long long int>::Scalar*) Sdata.data());
    

    // Compute LDLT decomposition
    Eigen::LDLT<Eigen::MatrixXd> solver;
    solver.compute(X.transpose() * X + S);

    // Also get the permutation
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> P(solver.transpositionsP());

    Eigen::VectorXd coef;
    coef.setZero(Scols);

    // Check Failure
    if (solver.info()!=Eigen::Success)
    {
        Eigen::MatrixXd L(Scols,Scols);
        L.setIdentity();
        Eigen::VectorXd d;
        d.setZero(Scols);
        return std::make_tuple(std::move(L),std::move(d),P.indices(),std::move(coef),1);
    }

    // Get the diagonal of D as a VectorXd
    Eigen::VectorXd d = solver.vectorD();

    // Get the L matrix as a dense MatrixXd
    Eigen::MatrixXd L = solver.matrixL();

    // Solve for coef (see Wood & Fasiolo, 2017)
    coef = solver.solve(X.transpose() * y);

    int code = 0;
    if (solver.info()!=Eigen::Success)
    {
        code = 2;
    }

    return std::make_tuple(std::move(L),std::move(d),P.indices(),std::move(coef),code);

}

std::tuple<Eigen::SparseMatrix<double,0,long long int>,VectorXi64,Eigen::VectorXd,int> solve_coef(const Eigen::Ref<Eigen::VectorXd> &y,
                                                                                       long long int Xrows, long long int Xcols, long long int Xnnz,
                                                                                       py::array_t<double, py::array::f_style | py::array::forcecast> Xdata,
                                                                                       py::array_t<long long int, py::array::f_style | py::array::forcecast> Xidptr,
                                                                                       py::array_t<long long int, py::array::f_style | py::array::forcecast> Xindices,
                                                                                       long long int Srows, long long int Scols, long long int Snnz,
                                                                                       py::array_t<double, py::array::f_style | py::array::forcecast> Sdata,
                                                                                       py::array_t<long long int, py::array::f_style | py::array::forcecast> Sidptr,
                                                                                       py::array_t<long long int, py::array::f_style | py::array::forcecast> Sindices){
    // Permuted Cholesky:
    // P * A * P' = L * L'
    // A = P' * L * L' * P
    // U = P' * L
    // U' = L' * P
    // A = U * U'
    // Inverse:
    // inv(A) = P' * Inv(L)' * inv(L) * Perm

    Eigen::Map<Eigen::SparseMatrix<double,0,long long int>> X(Xrows,Xcols,Xnnz,
                                                    (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Xidptr.data(),
                                                    (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Xindices.data(),
                                                    (Eigen::SparseMatrix<double,0,long long int>::Scalar*) Xdata.data());

    Eigen::Map<Eigen::SparseMatrix<double,0,long long int>> S(Srows,Scols,Snnz,
                                                    (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Sidptr.data(),
                                                    (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Sindices.data(),
                                                    (Eigen::SparseMatrix<double,0,long long int>::Scalar*) Sdata.data());

    // Prepare and compute Cholesky factor of X' * X + S
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double,0,long long int>> solver;
    solver.compute(X.transpose() * X + S);

    // Initialize coef vector
    Eigen::VectorXd coef;
    coef.setZero(Xcols);

    // First get the permutation
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic, long long int> P(solver.permutationP());

    if (solver.info()!=Eigen::Success)
    {
        Eigen::SparseMatrix<double,0,long long int> id(Xcols,Xcols);
        id.setIdentity();
        return std::make_tuple(std::move(id),P.indices(),std::move(coef),1);
    }

    // Solve for coef (see Wood & Fasiolo, 2017)
    coef = solver.solve(X.transpose() * y);

    if (solver.info()!=Eigen::Success)
    {
        Eigen::SparseMatrix<double,0,long long int> id(Xcols,Xcols);
        id.setIdentity();
        return std::make_tuple(std::move(id),P.indices(),std::move(coef),2);
    }

    return std::make_tuple(solver.matrixL(),P.indices(),std::move(coef),0);
}

std::tuple<
    Eigen::MatrixXd,
    Eigen::VectorXd,
    Eigen::VectorXi,
    Eigen::VectorXd,
    int
> dsolve_coefXX(
    const Eigen::Ref<Eigen::VectorXd> &Xy,
    const Eigen::Ref<Eigen::MatrixXd,0,Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &XXS
)
{
    // Compute LDLT decomposition
    Eigen::LDLT<Eigen::MatrixXd> solver;
    solver.compute(XXS);

    // Also get the permutation
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> P(solver.transpositionsP());

    Eigen::VectorXd coef;
    coef.setZero(XXS.cols());

    // Check Failure
    if (solver.info()!=Eigen::Success)
    {
        Eigen::MatrixXd L(XXS.cols(),XXS.cols());
        L.setIdentity();
        Eigen::VectorXd d;
        d.setZero(XXS.cols());
        return std::make_tuple(std::move(L),std::move(d),P.indices(),std::move(coef),1);
    }

    // Get the diagonal of D as a VectorXd
    Eigen::VectorXd d = solver.vectorD();

    // Get the L matrix as a dense MatrixXd
    Eigen::MatrixXd L = solver.matrixL();

    // Solve for coef (see Wood & Fasiolo, 2017)
    coef = solver.solve(Xy);

    int code = 0;
    if (solver.info()!=Eigen::Success)
    {
        code = 2;
    }

    return std::make_tuple(std::move(L),std::move(d),P.indices(),std::move(coef),code);
}

std::tuple<Eigen::SparseMatrix<double,0,long long int>,VectorXi64,Eigen::VectorXd,int> solve_coefXX(const Eigen::Ref<Eigen::VectorXd> &Xy,
                                                                                         long long int Xrows, long long int Xcols, long long int Xnnz,
                                                                                         py::array_t<double, py::array::f_style | py::array::forcecast> Xdata,
                                                                                         py::array_t<long long int, py::array::f_style | py::array::forcecast> Xidptr,
                                                                                         py::array_t<long long int, py::array::f_style | py::array::forcecast> Xindices){
    // Permuted Cholesky:
    // P * A * P' = L * L'
    // A = P' * L * L' * P
    // U = P' * L
    // U' = L' * P
    // A = U * U'
    // Inverse:
    // inv(A) = P' * Inv(L)' * inv(L) * Perm

    Eigen::Map<Eigen::SparseMatrix<double,0,long long int>> XXS(Xrows,Xcols,Xnnz,
                                                (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Xidptr.data(),
                                                (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Xindices.data(),
                                                (Eigen::SparseMatrix<double,0,long long int>::Scalar*) Xdata.data());

    // Prepare and compute Cholesky factor of X' * X + S
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double,0,long long int>> solver;
    solver.compute(XXS);

    // Initialize coef vector
    Eigen::VectorXd coef;
    coef.setZero(Xcols);

    // First get the permutation
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic, long long int> P(solver.permutationP());

    if (solver.info()!=Eigen::Success)
    {
        Eigen::SparseMatrix<double,0,long long int> id(Xcols,Xcols);
        id.setIdentity();
        return std::make_tuple(std::move(id),P.indices(),std::move(coef),1);
    }

    // Solve for coef (see Wood & Fasiolo, 2017)
    coef = solver.solve(Xy);

    if (solver.info()!=Eigen::Success)
    {
        Eigen::SparseMatrix<double,0,long long int> id(Xcols,Xcols);
        id.setIdentity();
        return std::make_tuple(std::move(id),P.indices(),std::move(coef),2);
    }

    return std::make_tuple(solver.matrixL(),P.indices(),std::move(coef),0);
}

std::tuple<Eigen::SparseMatrix<double,0,long long int>,VectorXi64,Eigen::VectorXd,long long int,int> solve_coef_pqr(const Eigen::Ref<Eigen::VectorXd> &y,
                                                                                           long long int Xrows, long long int Xcols, long long int Xnnz,
                                                                                           py::array_t<double, py::array::f_style | py::array::forcecast> Xdata,
                                                                                           py::array_t<long long int, py::array::f_style | py::array::forcecast> Xidptr,
                                                                                           py::array_t<long long int, py::array::f_style | py::array::forcecast> Xindices,
                                                                                           long long int Erows, long long int Ecols, long long int Ennz,
                                                                                           py::array_t<double, py::array::f_style | py::array::forcecast> Edata,
                                                                                           py::array_t<long long int, py::array::f_style | py::array::forcecast> Eidptr,
                                                                                           py::array_t<long long int, py::array::f_style | py::array::forcecast> Eindices){
    // Stable QR approach from Wood (2011) with initial check for rank deficiency that
    // is not a result of the choices for lambda. Matrix E is square root of S_\lambda.

    Eigen::Map<Eigen::SparseMatrix<double,0,long long int>> X(Xrows,Xcols,Xnnz,
                                                    (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Xidptr.data(),
                                                    (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Xindices.data(),
                                                    (Eigen::SparseMatrix<double,0,long long int>::Scalar*) Xdata.data());

    Eigen::Map<Eigen::SparseMatrix<double,0,long long int>> E(Erows,Ecols,Ennz,
                                                    (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Eidptr.data(),
                                                    (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Eindices.data(),
                                                    (Eigen::SparseMatrix<double,0,long long int>::Scalar*) Edata.data());

    // Computed column-pivoted QR factorization of X.
    Eigen::SparseQR<Eigen::SparseMatrix<double,0,long long int>,Eigen::AMDOrdering<long long int>> solver;
    solver.setPivotThreshold(sqrt(std::numeric_limits<double>::epsilon())*X.norm());
    solver.compute(X);

    // Initialize coef vector
    Eigen::VectorXd coef;
    coef.setZero(Xcols);

    // Column permutation matrix
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic, long long int> P(solver.colsPermutation());

    if(solver.info()!=Eigen::Success)
    {
        Eigen::SparseMatrix<double,0,long long int> id(Xcols,Xcols);
        id.setIdentity();
        return std::make_tuple(std::move(id),P.indices(),std::move(coef),0,1);
    }

    // Get upper triagonal factor after applying the permuation.
    Eigen::SparseMatrix<double,0,long long int> RP;
    RP = solver.matrixR();
    Eigen::SparseMatrix<double,0,long long int> R = RP.topRows(Xcols).eval() * P.transpose();

    // Check for rank deficiency
    //ToDo.

    // Concatenate R & E
    // Based on: https://stackoverflow.com/questions/42555456
    Eigen::SparseMatrix<double,0,long long int> RE(2*Xcols,Xcols);
    
    // Pre-allocate storage...
    RE.reserve(R.nonZeros() + E.nonZeros());
    for(Eigen::Index c=0; c<RE.cols(); ++c) // Loop also suggested in the Eigen tutorial: https://eigen.tuxfamily.org/dox/group__TutorialSparse.html
    {
        RE.startVec(c); // .insertBack() doc-string says that this has to be called in advance.

        // Fill first Xcols rows in column c with values in same column from R
        for(Eigen::SparseMatrix<double,0,long long int>::InnerIterator itR(R, c); itR; ++itR){
            RE.insertBack(itR.row(), c) = itR.value();
        }
        
        // And now fill subsequent Xcols rows in same columns with values in same column in E
        for(Eigen::Map<Eigen::SparseMatrix<double,0,long long int>>::InnerIterator itE(E, c); itE; ++itE){
            RE.insertBack(itE.row()+Xcols, c) = itE.value();
        }
            
    }

    RE.finalize();

    // Now form root of X.T@X + S
    Eigen::SparseQR<Eigen::SparseMatrix<double,0,long long int>,Eigen::AMDOrdering<long long int>> solver2;
    solver2.setPivotThreshold(sqrt(std::numeric_limits<double>::epsilon())*RE.norm());
    solver2.compute(RE);

    // Column permutation matrix for second decomposition
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic, long long int> P2(solver2.colsPermutation());

    if(solver2.info()!=Eigen::Success)
    {
        Eigen::SparseMatrix<double,0,long long int> id(Xcols,Xcols);
        id.setIdentity();
        return std::make_tuple(std::move(id),P2.indices(),std::move(coef),0,2);
    }

    // Adjust y - for that we need Q1 (Q below) from Wood (2011)
    // Let Q = Q of first solver
    // and QQ = Q of second solver
    // then Q1 = Q * QQ[:Xcols,:]
    // we need:
    // Q1.T * y
    // = (Q * QQ[:Xcols,:]).T * y
    // = QQ[:Xcols,:].T * Q.T * y
    // = (y.T * Q * QQ[:Xcols,:]).T
    // Q is [Xrows,Xcols], so the second product is [Xcols,1]
    // QQ is [Xcols*2,Xcols] so we have to extract it unfortunately
    Eigen::SparseMatrix<double,0,long long int> QQ;
    Eigen::VectorXd Qy,Qy2;

    QQ = solver2.matrixQ();
    Qy = solver.matrixQ().adjoint() * y;

    Eigen::MatrixXd QQ2 = solver2.matrixQ();
    Eigen::MatrixXd Q1 = solver.matrixQ() * QQ2.topLeftCorner(Xcols,Xcols);

    Qy2 =  Q1.transpose() * y;

    // To solve for coefficients fill coef with rhs of solution by Wood (2011)
    coef = QQ.topLeftCorner(Xcols,Xcols).transpose() * Qy;

    // Extract root of X.T@X + S
    Eigen::SparseMatrix<double,0,long long int> R2 = solver2.matrixR().topRows(Xcols);

    // Now do the actual solve - but R2 will not be sparse
    R2.triangularView<Eigen::Upper>().solveInPlace(coef);

    return std::make_tuple(R2,P2.indices(),std::move(coef),solver2.rank(),0);
}

std::tuple<
    Eigen::MatrixXd,
    Eigen::VectorXi,
    Eigen::VectorXd,
    long int,
    int
> dsolve_coef_pqr(
    const Eigen::Ref<Eigen::VectorXd> &y,
    const Eigen::Ref<Eigen::MatrixXd,0,Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &XE
)
{
    // Variant of the stable QR approach from Wood (2011) without initial check for rank deficiency that
    // is not a result of the choices for lambda. Matrix XE has square root of S_\lambda
    // concatenated
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver;
    solver.compute(XE);

    // Initialize coef vector
    Eigen::VectorXd coef;
    coef.setZero(XE.cols());

    if(solver.info()!=Eigen::Success)
    {
        Eigen::MatrixXd R(XE.cols(),XE.cols());
        R.setIdentity();
        return std::make_tuple(std::move(R),solver.colsPermutation().indices(),std::move(coef),0,1);
    }

    // Extract matrix R (ideally root of X.T@X + S)
    Eigen::MatrixXd R = solver.matrixR().topLeftCorner(solver.rank(), solver.rank()).template triangularView<Eigen::Upper>();

    // And solve for coef. Here the lhs is like what is discussed in first chapter of Wood (2017)
    // Essentially coef holds f later.
    Eigen::VectorXd yE,Qy;
    yE.setZero(XE.rows());
    yE.head(y.rows()) = y;
    Qy = solver.householderQ().adjoint() * yE;
    coef = Qy.head(solver.rank());

    // Now do the actual solve
    R.triangularView<Eigen::Upper>().solveInPlace(coef);

    return std::make_tuple(std::move(R),solver.colsPermutation().indices(),std::move(coef),solver.rank(),0);
}

std::tuple<Eigen::SparseMatrix<double,0,long long int>,VectorXi64,VectorXi64,Eigen::VectorXd,long long int,int> solve_coef_pqr2(const Eigen::Ref<Eigen::VectorXd> &y,
                                                                                           long long int Xrows, long long int Xcols, long long int Xnnz,
                                                                                           py::array_t<double, py::array::f_style | py::array::forcecast> Xdata,
                                                                                           py::array_t<long long int, py::array::f_style | py::array::forcecast> Xidptr,
                                                                                           py::array_t<long long int, py::array::f_style | py::array::forcecast> Xindices,
                                                                                           long long int Erows, long long int Ecols, long long int Ennz,
                                                                                           py::array_t<double, py::array::f_style | py::array::forcecast> Edata,
                                                                                           py::array_t<long long int, py::array::f_style | py::array::forcecast> Eidptr,
                                                                                           py::array_t<long long int, py::array::f_style | py::array::forcecast> Eindices){
    // Variant of the stable QR approach from Wood (2011) without initial check for rank deficiency that
    // is not a result of the choices for lambda. Matrix E is square root of S_\lambda.
    // This preserves sparsity much better in R (root of X.T@X + S_\lambda) than what is achieved with solve_coef_pqr()

    Eigen::Map<Eigen::SparseMatrix<double,0,long long int>> X(Xrows,Xcols,Xnnz,
                                                    (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Xidptr.data(),
                                                    (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Xindices.data(),
                                                    (Eigen::SparseMatrix<double,0,long long int>::Scalar*) Xdata.data());

    Eigen::Map<Eigen::SparseMatrix<double,0,long long int>> E(Erows,Ecols,Ennz,
                                                    (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Eidptr.data(),
                                                    (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Eindices.data(),
                                                    (Eigen::SparseMatrix<double,0,long long int>::Scalar*) Edata.data());

    // Initialize coef vector
    Eigen::VectorXd coef;
    coef.setZero(Xcols);

    // Concatenate X & E
    // Based on: https://stackoverflow.com/questions/42555456
    Eigen::SparseMatrix<double,0,long long int> RE(Xrows+Xcols,Xcols);
    
    // Pre-allocate storage...
    RE.reserve(Xnnz + Ennz);
    for(Eigen::Index c=0; c<RE.cols(); ++c) // Loop also suggested in the Eigen tutorial: https://eigen.tuxfamily.org/dox/group__TutorialSparse.html
    {
        RE.startVec(c); // .insertBack() doc-string says that this has to be called in advance.

        // Fill first Xrows rows in column c with values in same column from X
        for(Eigen::Map<Eigen::SparseMatrix<double,0,long long int>>::InnerIterator  itR(X, c); itR; ++itR){
            RE.insertBack(itR.row(), c) = itR.value();
        }
        
        // And now fill subsequent Xcols rows in same columns with values in same column in E
        for(Eigen::Map<Eigen::SparseMatrix<double,0,long long int>>::InnerIterator itE(E, c); itE; ++itE){
            RE.insertBack(itE.row()+Xrows, c) = itE.value();
        }
            
    }

    RE.finalize();

    // Compute ordering so that Cholesky factor of X' * X + S is sparse
    // see Golub & Van Loan "Matrix Computations: 4ED" (2013)
    Eigen::AMDOrdering<long long int> ordering;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, long long int> P1;
    Eigen::SparseMatrix<double,0,long long int> XXS = RE.transpose() * RE;
    ordering(XXS.selfadjointView<Eigen::Lower>(), P1);

    // Now form root of X.T@X + S
    Eigen::SparseQR<Eigen::SparseMatrix<double,0,long long int>,Eigen::NaturalOrdering<long long int>> solver2;
    //solver2.setPivotThreshold(sqrt(std::numeric_limits<double>::epsilon())*sqrt(XXS.diagonal().array().abs().maxCoeff()));  // Just use default from Davis here.
    solver2.compute(RE*P1); // Now use ordering computed previously to pivot columns

    // Column permutation matrix for second decomposition
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic, long long int> P2(solver2.colsPermutation());

    if(solver2.info()!=Eigen::Success)
    {
        Eigen::SparseMatrix<double,0,long long int> id(Xcols,Xcols);
        id.setIdentity();
        return std::make_tuple(std::move(id),P1.indices(),P2.indices(),std::move(coef),0,2);
    }

    // Here the lhs is like what is discussed in first chapter of Wood (2017)
    // Essentially coef holds f later.
    Eigen::VectorXd yE,Qy;
    yE.setZero(Xrows+Xcols);
    yE.head(Xrows) = y;
    Qy = solver2.matrixQ().adjoint() * yE;
    coef = Qy.head(solver2.rank());

    // Extract root of X.T@X + S
    Eigen::SparseMatrix<double,0,long long int> R2 = solver2.matrixR().topLeftCorner(solver2.rank(),solver2.rank());

    // Now do the actual solve
    R2.triangularView<Eigen::Upper>().solveInPlace(coef);

    return std::make_tuple(R2,P1.indices(),P2.indices(),std::move(coef),solver2.rank(),0);
}

Eigen::MatrixXd dsolve_tr(
    const Eigen::Ref<Eigen::MatrixXd,0,Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &A,
    Eigen::MatrixXd C
)
{
    // Solves A*B=C, where A is lower triangular. This can be utilized to obtain B = inv(A), when C is
    // the identity. For dense case
    A.triangularView<Eigen::Lower>().solveInPlace(C);
    return C;
}

Eigen::SparseMatrix<double,0,long long int> solve_tr(long long int Arows, long long int Acols, long long int Annz,
                                     py::array_t<double, py::array::f_style | py::array::forcecast> Adata,
                                     py::array_t<long long int, py::array::f_style | py::array::forcecast> Aidptr,
                                     py::array_t<long long int, py::array::f_style | py::array::forcecast> Aindices,
                                     Eigen::SparseMatrix<double,0,long long int> C){
    // Solves A*B=C, where A is lower triangular. This can be utilized to obtain B = inv(A), when C is
    // the identity. Importantly, when A is a n*n matrix then C can also be specified as a n*m block of
    // the identity. In that case, inv(A) can be obtained in parallel.
    // Note: we copy C over, so we can solve in place and then just return.

    Eigen::Map<Eigen::SparseMatrix<double,0,long long int>> A(Arows,Acols,Annz,
                                              (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Aidptr.data(),
                                              (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Aindices.data(),
                                              (Eigen::SparseMatrix<double,0,long long int>::Scalar*) Adata.data()); 

    A.triangularView<Eigen::Lower>().solveInPlace(C);
    return C;
}

Eigen::MatrixXd dbacksolve_tr(
    const Eigen::Ref<Eigen::MatrixXd,0,Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &A,
    Eigen::MatrixXd C
)
{
    // Solves A*B=C, where A is UPPER triangular. . For dense case
    A.triangularView<Eigen::Upper>().solveInPlace(C);
    return C;
}

Eigen::SparseMatrix<double,0,long long int> backsolve_tr(long long int Arows, long long int Acols, long long int Annz,
                                         py::array_t<double, py::array::f_style | py::array::forcecast> Adata,
                                         py::array_t<long long int, py::array::f_style | py::array::forcecast> Aidptr,
                                         py::array_t<long long int, py::array::f_style | py::array::forcecast> Aindices,
                                         Eigen::SparseMatrix<double,0,long long int> C){
    // Solves A*B=C, where A is UPPER triangular. This can be utilized to obtain B = inv(A), when C is
    // the identity. Importantly, when A is a n*n matrix then C can also be specified as a n*m block of
    // the identity. In that case, inv(A) can be obtained in parallel.
    // Note: we copy C over, so we can solve in place and then just return.

    Eigen::Map<Eigen::SparseMatrix<double,0,long long int>> A(Arows,Acols,Annz,
                                              (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Aidptr.data(),
                                              (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Aindices.data(),
                                              (Eigen::SparseMatrix<double,0,long long int>::Scalar*) Adata.data());  

    A.triangularView<Eigen::Upper>().solveInPlace(C);
    return C;
}

std::tuple<VectorXi64,long long int> id_dependencies(const Eigen::Ref<Eigen::MatrixXd,0,Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &X1,
                                                     const Eigen::Ref<Eigen::MatrixXd,0,Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &X2,
                                                     double tol)
            {
                /*
                Identify linear dependencies between matrices X1 and X2, based on section 5.6.3 in Wood (2017) and
                the fixDependence function in mgcv, see: https://github.com/cran/mgcv/blob/fb7e8e718377513e78ba6c6bf7e60757fc6a32a9/R/mgcv.r#L501
                */

                size_t X1cols = X1.cols();

                // Need QR decomposition of X1
                Eigen::HouseholderQR<Eigen::MatrixXd> qr1(X1);
                Eigen::MatrixXd R1 = qr1.matrixQR().triangularView<Eigen::Upper>();

                // Need to check for zero blocks in R2 later on - mgcv uses first diagonal element on R1 as an indication of a "healthy" element to threshold, so we do the same
                double thresh = abs(R1(0,0))*tol;
                
                // B = Q1.T@X2
                Eigen::MatrixXd B(X2);
                B.applyOnTheLeft(qr1.householderQ().adjoint());

                // Extract lower block
                Eigen::MatrixXd Blb = B(Eigen::seq(X1cols,Eigen::last),Eigen::all);
                
                // Apply second QR decomposition, now with pivoting
                Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr2(Blb);
                
                // Extract pivot and rank estimate
                VectorXi64 piv = qr2.colsPermutation().indices().cast<long long int>();
                
                // Now check for zero block in lower right corner of R2.
                // Make sure that any previous rank determination by Eigen is aknolwedged.
                long long int r = qr2.rank();
                Eigen::MatrixXd R2 = qr2.matrixR().topLeftCorner(r,r).triangularView<Eigen::Upper>();
                
                double mb;
                while (r > 0)
                {
                    // The check of the mean absolute value of the sub-block is performed by mgcv, so we use the same here.
                    // see: https://github.com/cran/mgcv/blob/fb7e8e718377513e78ba6c6bf7e60757fc6a32a9/R/mgcv.r#L526
                    mb = R2(Eigen::seq(r-1,Eigen::last),
                            Eigen::seq(r-1,Eigen::last)).array().abs().mean();

                    // mean absolute block value too large to hint at zero block -> get out
                    if (mb >= thresh)
                    {
                        break;
                    }
                    
                    // Lower rank by 1
                    r--;
                    
                }

                return std::make_tuple(piv,r);

            }

PYBIND11_MODULE(eigen_solvers, m) {
    m.doc() = "cpp solvers for GAMM, GAMMLSS, and GSMM estimation";
    m.def("dchol", &dchol, "Compute cholesky factor L of A (dense case)");
    m.def("chol", &chol, "Compute cholesky factor L of A");
    m.def("cholP", &cholP, "Compute cholesky factor L of A after applying a sparsity enhancing permutation to A");
    m.def("dcholP", &dcholP, py::arg("A").noconvert(), "Compute LDL factorization of A with stability enhancing pivoting");
    m.def("pqr", &pqr, "Perform column pivoted QR decomposition of A");
    m.def("pqrr", &pqrr, "Perform column pivoted QR decomposition of A, but only return R.");
    m.def("dpqrr", &dpqrr, py::arg("A").noconvert(), "Perform column pivoted QR decomposition of dense matrix A.");
    m.def("spqr", &spqr, "Perform column pivoted QR decomposition of symmetric matrix A, so that L - where A=L@L.T - is sparse.");
    m.def("solve_pqr", &solve_pqr, "Perform column pivoted QR decomposition of A, then solve for inverse of A");
    m.def("solve_am", &solve_am, "Solve additive model, return coefficient vector and inverse");
    m.def("solve_L", &solve_L, "Solve cholesky of XX+S");
    m.def("solve_LXX", &solve_LXX, "Solve cholesky of XX+S, but with XX + S pre-computed.");
    m.def("dsolve_coef", &dsolve_coef, "Solve additive model coefficients (dense case)");
    m.def("solve_coef", &solve_coef, "Solve additive model coefficients");
    m.def("dsolve_coef_pqr", &dsolve_coef_pqr, "Solve additive model coefficients, using stable QR decomposition (dense case)");
    m.def("solve_coef_pqr", &solve_coef_pqr2, "Solve additive model coefficients, using stable QR decomposition");
    m.def("dsolve_coefXX", &dsolve_coefXX, "Solve additive model coefficients, but with XX + S and Xy pre-computed (dense case).");
    m.def("solve_coefXX", &solve_coefXX, "Solve additive model coefficients, but with XX + S and Xy pre-computed.");
    m.def("dsolve_tr",&dsolve_tr,"Solve A*B = C, where A is lower triangular (dense case).");
    m.def("solve_tr",&solve_tr,"Solve A*B = C, where A is lower triangular.");
    m.def("dbacksolve_tr",&dbacksolve_tr,"Solve A*B = C, where A is upper triangular (dense case).");
    m.def("backsolve_tr",&backsolve_tr,"Solve A*B = C, where A is upper triangular.");
    m.def("id_dependencies",&id_dependencies,"Identify linear dependencies between matrices X1 and X2.");
}