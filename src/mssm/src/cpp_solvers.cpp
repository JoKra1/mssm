
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

std::tuple<Eigen::SparseMatrix<double>,Eigen::VectorXd,int> solve_am(Eigen::VectorXd y, Eigen::SparseMatrix<double> X, Eigen::SparseMatrix<double> S){
    
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

    if (solver.info()!=Eigen::Success)
    {

        return std::make_tuple(std::move(id),std::move(coef),1);
    }

    // Solve for coef
    coef = solver.solve(X.transpose() * y);
    std::cout << "Solved for coef";

    if (solver.info()!=Eigen::Success)
    {

        return std::make_tuple(std::move(id),std::move(coef),2);
    }

    // We also need inv(L') * P from P * X' * X + S * P' = L * L'
    // so the inverse of the upper matrix from the solver times the
    // permutation matrix created for us by eigen.

    // First get the permutation
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> P(solver.permutationP());

    // P * A * P' = L * L'
    // A = P' * L * L' * P
    // U = P' * L
    // U' = L' * P
    // A = U * U'
    Eigen::SparseMatrix<double> L = solver.matrixU();
    std::cout << LT;

    // Now compute inverse
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>,Eigen::Upper,Eigen::NaturalOrdering<int>> inv_solver;
    inv_solver.compute(LT);

    if (inv_solver.info()!=Eigen::Success)
    {

        return std::make_tuple(std::move(id),std::move(coef),3);
    }

    Eigen::SparseMatrix<double> invXXP = inv_solver.solve(id);

    if (inv_solver.info()!=Eigen::Success)
    {

        return std::make_tuple(std::move(invXXP),std::move(coef),4);
    }
    
    // Cholesky of inv(X' * X + S), not necessarily triangular!
    Eigen::SparseMatrix<double> invXX = invXXP * P;

    return std::make_tuple(std::move(invXX),std::move(coef),0);;
}

PYBIND11_MODULE(cpp_solvers, m) {
    m.doc() = "cpp solvers for sms (DC) GAMM estimation";

    m.def("chol", &chol, "Compute cholesky factor L of A");
    m.def("solve_am", &solve_am, "Solve additive model, return coefficient vector and inverse");
}