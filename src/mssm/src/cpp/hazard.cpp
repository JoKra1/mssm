#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include<Eigen/Sparse>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <limits>

namespace py = pybind11;
typedef Eigen::Vector<long long int, Eigen::Dynamic> VectorXi64;

double llk2(
        size_t nt,
        const Eigen::Ref<Eigen::VectorXd>& gamma,
        const Eigen::Ref<VectorXi64>& delta,
        py::list ris
    )
    {
        /*
        - nt is number of unique event times.
        - gamma is np.exp(eta)
        - delta are censoring indices.
        - ris holds index vectors as described by WPS (2016), holding for each unique end time
        the indices to the corresponding rows in model matrix ``X``

        Computations from: Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and
        Model Selection for General Smooth Models
        */

        double gamma_p = 0.0;
        double allk = 0.0;
        for (size_t j = 0; j < nt; j++)
        {
            // Get ri[j] as matrix, then array
            py::handle ri_jh = ris[j];
            py::array_t<long long int> ri_j = py::cast<py::array>(ri_jh);
            py::buffer_info info_j = ri_j.request();

            long long int *ptr_j = static_cast<long long int *>(info_j.ptr);

            Eigen::Map<VectorXi64> rij(ptr_j,ri_j.shape(0),1);
            
            // Now compute llk adjustment for ri[j]
            double dj = delta(rij.array()).array().sum();

            // Adjust gamma_p
            gamma_p += gamma(rij.array()).array().sum();

            allk -= dj * log(gamma_p);
        }
        
        return allk;
    }

Eigen::VectorXd grad2(
        size_t nt,
        const Eigen::Ref<Eigen::VectorXd>& gamma,
        const Eigen::Ref<VectorXi64>& delta,
        py::list ris,
        const Eigen::Ref<Eigen::MatrixXd,0,Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &X
    )
    {
        /*
        - nt is number of unique event times.
        - gamma is np.exp(eta)
        - delta are censoring indices.
        - ris holds index vectors as described by WPS (2016), holding for each unique end time
        the indices to the corresponding rows in ``X``
        - X is model matrix

        Computations from: Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and
        Model Selection for General Smooth Models
        */

        size_t n_coef = X.cols();
        Eigen::VectorXd grad2;
        grad2.setZero(n_coef);

        double gamma_p = 0.0;
        Eigen::VectorXd bp;
        bp.setZero(n_coef);
        for (size_t j = 0; j < nt; j++)
        {
            // Get ri[j] as matrix, then array
            py::handle ri_jh = ris[j];
            py::array_t<long long int> ri_j = py::cast<py::array>(ri_jh);
            py::buffer_info info_j = ri_j.request();

            long long int *ptr_j = static_cast<long long int *>(info_j.ptr);

            Eigen::Map<VectorXi64> rij(ptr_j,ri_j.shape(0),1);
            
            // Now compute llk adjustment for ri[j]
            double dj = delta(rij.array()).array().sum();

            // Adjust gamma_p
            gamma_p += gamma(rij.array()).array().sum();

            // Now gradient computations
            bp += (gamma(rij.array()).transpose() * X(rij.array(),Eigen::all)).transpose();

            grad2 -= (dj * (bp / gamma_p));
        }
        
        return grad2;
    }

Eigen::MatrixXd hessian(
        size_t nt,
        const Eigen::Ref<Eigen::VectorXd>& gamma,
        const Eigen::Ref<VectorXi64>& delta,
        py::list ris,
        const Eigen::Ref<Eigen::MatrixXd,0,Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &X
    )
    {
        /*
        - nt is number of unique event times.
        - gamma is np.exp(eta)
        - delta are censoring indices.
        - ris holds index vectors as described by WPS (2016), holding for each unique end time
        the indices to the corresponding rows in ``X``
        - X is model matrix

        Computations from: Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and
        Model Selection for General Smooth Models
        */

        size_t n_coef = X.cols();
        Eigen::MatrixXd hess;
        hess.setZero(n_coef,n_coef);
        Eigen::MatrixXd Ap;
        Ap.setZero(n_coef,n_coef);

        double gamma_p = 0.0;
        Eigen::VectorXd bp;
        bp.setZero(n_coef);
        for (size_t j = 0; j < nt; j++)
        {
            // Get ri[j] as matrix, then array
            py::handle ri_jh = ris[j];
            py::array_t<long long int> ri_j = py::cast<py::array>(ri_jh);
            py::buffer_info info_j = ri_j.request();

            long long int *ptr_j = static_cast<long long int *>(info_j.ptr);

            Eigen::Map<VectorXi64> rij(ptr_j,ri_j.shape(0),1);
            
            // Now compute llk adjustment for ri[j]
            double dj = delta(rij.array()).array().sum();

            // Adjust gamma_p
            Eigen::VectorXd gi = gamma(rij.array());
            gamma_p += gi.array().sum();

            // Now gradient related computations
            Eigen::MatrixXd Xi = X(rij.array(),Eigen::all);
            bp += (gi.transpose() * Xi).transpose();

            // And now hessian related computations
            Eigen::MatrixXd Ai = (gi.asDiagonal() * Xi).transpose() * Xi;
            Ap += Ai;

            hess += (dj * bp * bp.transpose() / pow(gamma_p,2) - dj * Ap / gamma_p);

        }
        
        return hess;
    }

PYBIND11_MODULE(hazard, m) {
    m.def("llk2", &llk2, "Second added of partial llk.");
    m.def("grad2", &grad2, "Second added of gradient of partial llk.");
    m.def("hessian", &hessian, "Hessian of partial llk.");
}