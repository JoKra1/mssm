
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include<Eigen/Sparse>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <limits>


namespace py = pybind11;

Eigen::SparseMatrix<double> chol(Eigen::SparseMatrix<double> A){
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>,Eigen::Lower,Eigen::NaturalOrdering<int>> solver;
    solver.compute(A);

    Eigen::SparseMatrix<double> id(A.rows(),A.cols());
    id.setIdentity();

    if (solver.info()!=Eigen::Success)
    {
        return id;
    }

    Eigen::SparseMatrix<double> L = solver.matrixL();
    
    return L;
}

Eigen::VectorXd solve_am(Eigen::VectorXd y, Eigen::SparseMatrix<double> X, Eigen::SparseMatrix<double> S){
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(X.transpose() * X + S);

    if (solver.info()!=Eigen::Success)
    {
        std::cout << "Failure 1\n"
        return y;
    }

    Eigen::VectorXd coef = solver.solve(X.transpose() * y);

    if (solver.info()!=Eigen::Success)
    {
        std::cout << "Failure 2\n"
        return y;
    }

    return coef;
}

PYBIND11_MODULE(cpp_solvers, m) {
    m.doc() = "cpp solvers for sms (DC) GAMM estimation";

    m.def("chol", &chol, "Compute cholesky factor L of A");
    m.def("solve_am", &solve_am, "Solve additive model");
}