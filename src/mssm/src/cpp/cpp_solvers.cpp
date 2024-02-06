#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include<Eigen/Sparse>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <limits>


namespace py = pybind11;

std::tuple<Eigen::SparseMatrix<double>,int> chol(Eigen::SparseMatrix<double> A){
    // We prevent any sparsity preserving ordering, since we need the un-pivoted factor L so that L * L' = A
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>,Eigen::Lower,Eigen::NaturalOrdering<int>> solver;
    solver.compute(A);

    if (solver.info()!=Eigen::Success)
    {
        Eigen::SparseMatrix<double> id(A.rows(),A.cols());
        id.setIdentity();
        return std::make_tuple(std::move(id),1);
    }

    Eigen::SparseMatrix<double> L = solver.matrixL();
    
    return std::make_tuple(std::move(L),0);
}

std::tuple<Eigen::SparseMatrix<double>,Eigen::SparseMatrix<double>,Eigen::VectorXi, int> pqr(Eigen::SparseMatrix<double> A) {
    // Computed column-pivoted QR factorization of A.
    Eigen::SparseQR<Eigen::SparseMatrix<double>,Eigen::COLAMDOrdering<int>> solver;
    solver.compute(A);

    // Column permutation matrix
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> P(solver.colsPermutation());

    if(solver.info()!=Eigen::Success)
    {
        Eigen::SparseMatrix<double> Q(A.rows(),A.cols());
        Q.setIdentity();

        Eigen::SparseMatrix<double> R(A.cols(),A.cols());
        R.setIdentity();
        return std::make_tuple(std::move(Q),std::move(R),P.indices(),1);
    }

    // see: https://eigen.tuxfamily.org/dox/classEigen_1_1SparseQR.html
    Eigen::SparseMatrix<double> Q;
    Q = solver.matrixQ();

    // Upper triagonal factor
    Eigen::SparseMatrix<double> R = solver.matrixR();

    // Upper triagonal factor after applying the permuation.
    //Eigen::SparseMatrix<double> R = solver.matrixR().eval() * P.transpose();

    return std::make_tuple(std::move(Q),R,P.indices(),0);
    
}

std::tuple<Eigen::SparseMatrix<double>,Eigen::VectorXi,Eigen::VectorXd,int> solve_am(Eigen::VectorXd y, Eigen::SparseMatrix<double> X, Eigen::SparseMatrix<double> S){
    // Permuted Cholesky:
    // P * A * P' = L * L'
    // A = P' * L * L' * P
    // U = P' * L
    // U' = L' * P
    // A = U * U'
    // Inverse:
    // inv(A) = P' * Inv(L)' * inv(L) * Perm

    int Xcols = X.cols();

    // Prepare and compute Cholesky factor of X' * X + S
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(X.transpose() * X + S);

    // Initialize coef vector
    Eigen::VectorXd coef;
    coef.setZero(Xcols);

    // Also setup identity target for inverse of L' (see below)
    Eigen::SparseMatrix<double> id(Xcols,Xcols);
    id.setIdentity();

    // We also need inv(L) * P from P * X' * X + S * P' = L * L'
    // so the inverse of the lower matrix from the solver times the
    // permutation matrix created for us by eigen.

    // First get the permutation
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> P(solver.permutationP());

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

std::tuple<Eigen::SparseMatrix<double>,Eigen::VectorXi,int> solve_L(Eigen::SparseMatrix<double> X, Eigen::SparseMatrix<double> S){
    // Permuted Cholesky:
    // P * A * P' = L * L'
    // A = P' * L * L' * P
    // U = P' * L
    // U' = L' * P
    // A = U * U'
    // Inverse:
    // inv(A) = P' * Inv(L)' * inv(L) * Perm

    int Xcols = X.cols();

    // Prepare and compute Cholesky factor of X' * X + S
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(X.transpose() * X + S);

    // Setup identity target for inverse of L' (see below)
    Eigen::SparseMatrix<double> id(Xcols,Xcols);
    id.setIdentity();

    // Also get the permutation
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> P(solver.permutationP());

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

std::tuple<Eigen::SparseMatrix<double>,Eigen::VectorXi,Eigen::VectorXd,int> solve_coef(Eigen::VectorXd y, Eigen::SparseMatrix<double> X, Eigen::SparseMatrix<double> S){
    // Permuted Cholesky:
    // P * A * P' = L * L'
    // A = P' * L * L' * P
    // U = P' * L
    // U' = L' * P
    // A = U * U'
    // Inverse:
    // inv(A) = P' * Inv(L)' * inv(L) * Perm

    int Xcols = X.cols();

    // Prepare and compute Cholesky factor of X' * X + S
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(X.transpose() * X + S);

    // Initialize coef vector
    Eigen::VectorXd coef;
    coef.setZero(Xcols);

    // First get the permutation
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> P(solver.permutationP());

    if (solver.info()!=Eigen::Success)
    {
        Eigen::SparseMatrix<double> id(Xcols,Xcols);
        id.setIdentity();
        return std::make_tuple(std::move(id),P.indices(),std::move(coef),1);
    }

    // Solve for coef (see Wood & Fasiolo, 2017)
    coef = solver.solve(X.transpose() * y);

    if (solver.info()!=Eigen::Success)
    {
        Eigen::SparseMatrix<double> id(Xcols,Xcols);
        id.setIdentity();
        return std::make_tuple(std::move(id),P.indices(),std::move(coef),2);
    }

    return std::make_tuple(solver.matrixL(),P.indices(),std::move(coef),0);
}

Eigen::SparseMatrix<double> solve_tr(Eigen::SparseMatrix<double> A,Eigen::SparseMatrix<double> B){
    // Solves A*B=C, where B is lower triangular. This can be utilized to obtain B = inv(A), when C is
    // the identity. Importantly, when A is a n*n matrix then C can also be specified as a n*m block of
    // the identity. In that case, inv(A) can be obtained in parallel. 

    A.triangularView<Eigen::Lower>().solveInPlace(B);
    return B;
}

Eigen::SparseMatrix<double> backsolve_tr(Eigen::SparseMatrix<double> A,Eigen::SparseMatrix<double> B){
    // Solves A*B=C, where B is UPPER triangular. This can be utilized to obtain B = inv(A), when C is
    // the identity. Importantly, when A is a n*n matrix then C can also be specified as a n*m block of
    // the identity. In that case, inv(A) can be obtained in parallel. 

    A.triangularView<Eigen::Upper>().solveInPlace(B);
    return B;
}

PYBIND11_MODULE(cpp_solvers, m) {
    m.doc() = "cpp solvers for sms (DC) GAMM estimation";

    m.def("chol", &chol, "Compute cholesky factor L of A");
    m.def("pqr", &pqr, "Perform column pivoted QR decomposition of A");
    m.def("solve_am", &solve_am, "Solve additive model, return coefficient vector and inverse");
    m.def("solve_L", &solve_L, "Solve cholesky of XX+S");
    m.def("solve_coef", &solve_coef, "Solve additive model coefficients");
    m.def("solve_tr",&solve_tr,"Solve A*B = C, where A is lower triangular.");
    m.def("backsolve_tr",&backsolve_tr,"Solve A*B = C, where A is upper triangular.");
}