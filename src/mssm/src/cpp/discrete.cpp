#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
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
typedef Eigen::Vector<long int, Eigen::Dynamic> VectorXi32;

Eigen::VectorXd A2(
                   py::list uMats,
                   py::list indices,
                   py::array_t<size_t, py::array::f_style | py::array::forcecast> ps_a,
                   size_t q,
                   size_t n,
                   size_t n_t,
                   size_t j
                )
{
    /*
    Algorithm A2 from Wood et al., 2017
    */

    Eigen::VectorXd Xj;
    Xj.setOnes(n);

    size_t j2 = j;
    auto ps = ps_a.unchecked();

    for (size_t i = 0; i < n_t; i++)
    {
        q /= ps(i);
        size_t ji = j2 / q;
        j2 = j2 % q;

        /*
        Get matrix and index buffer
        */
        py::handle m_ih = uMats[i];
        py::array_t<double> m_i = py::cast<py::array>(m_ih);
        py::buffer_info info_i = m_i.request();

        double *ptr_i = static_cast<double *>(info_i.ptr);

        Eigen::Map<Eigen::MatrixXd> m(ptr_i,m_i.shape(0),m_i.shape(1));

        py::handle i_ih = indices[i];
        py::array_t<long int> i_i = py::cast<py::array>(i_ih);
        py::buffer_info info_ii = i_i.request();

        long int *ptr_ii = static_cast<long int *>(info_ii.ptr);

        Eigen::Map<VectorXi32> idx(ptr_ii, n);

        Xj = (Xj.array() * m(idx, ji).array()).matrix();
    }
    
    return Xj;
}

Eigen::VectorXd A2Q(
                   py::list uMats,
                   py::list indices,
                   py::array_t<size_t, py::array::f_style | py::array::forcecast> ps_a,
                   size_t q0,
                   size_t n,
                   size_t n_t,
                   VectorXi64 j,
                   size_t n_c,
                   Eigen::VectorXd Qi
                )
{
    /*
    Modified version of Algorithm A2 from Wood et al., 2017 to account for constraint matrix Q
    */

    Eigen::VectorXd Xj;
    Xj.setOnes(n);
    auto ps = ps_a.unchecked();

    for (size_t ri = 0; ri < n; ri++)
    {
        VectorXi64 j2 = j;

        Eigen::VectorXd Xr;
        Xr.setOnes(n_c);
        
        // Reset q
        size_t q = q0;
        
        for (size_t i = 0; i < n_t; i++)
        {
            q /= ps(i);
            VectorXi64 ji = j2 / q;
            j2 = (j2.array() - (q * (j2.array() / q))).matrix();
            //py::print(j2);
            //j2 = (j2.unaryExpr([q](const long long int x) { return x%q; })).eval();

            /*
            Get matrix and index buffer
            */
            py::handle m_ih = uMats[i];
            py::array_t<double> m_i = py::cast<py::array>(m_ih);
            py::buffer_info info_i = m_i.request();

            double *ptr_i = static_cast<double *>(info_i.ptr);

            Eigen::Map<Eigen::MatrixXd> m(ptr_i,m_i.shape(0),m_i.shape(1));

            py::handle i_ih = indices[i];
            py::array_t<long int> i_i = py::cast<py::array>(i_ih);
            py::buffer_info info_ii = i_i.request();

            long int *ptr_ii = static_cast<long int *>(info_ii.ptr);

            Eigen::Map<VectorXi32> idx(ptr_ii, n);

            Eigen::VectorXd mr = m(idx(ri), ji);
            //py::print(mr);

            Xr = (Xr.array() * mr.array()).matrix();
        }

        Xj(ri) = Xr.dot(Qi);
    }
    
    return Xj;
}

PYBIND11_MODULE(discrete, m) {
    m.def("A2", &A2, "Algorithm A2 from Wood et al., 2017.");
    m.def("A2Q", &A2Q, "Modified version of Algorithm A2 from Wood et al., 2017 to account for constraint matrix Q.");
}