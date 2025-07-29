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

Eigen::MatrixXd dChol(const Eigen::Ref<Eigen::MatrixXd,0,Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &R,
                      const Eigen::Ref<Eigen::MatrixXd,0,Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &A)
                {

                    /*
                    R is transpose of Cholesky factor of some matrix H + a*A. Function returns derivative of
                    this sum with respect to a.

                    Computations from: Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models
                    */
                    
                    size_t Rcols = R.cols();
                    Eigen::MatrixXd dC = Eigen::MatrixXd::Zero(Rcols,Rcols);
                    double Rii, dRii, Bij;

                    for (size_t i = 0; i < Rcols; i++)
                    {   
                        Rii = 0.0;
                        dRii = 0.0;
                        for (size_t j = i; j < Rcols; j++)
                        {
                            Bij = A(i,j);

                            if (i > 0)
                            {
                                auto k = Eigen::seq(0,i-1);
                                Bij -= ((dC(k,i).array()*R(k,j).array()) + (R(k,i).array()*dC(k,j).array())).sum();
                            }
                            
                            if (i == j)
                            {
                                Rii = R(i,i);
                                dRii = 0.5*Bij/Rii;
                                dC(i,j) = dRii;
                            }
                            else if (j > i)
                            {
                                dC(i,j) = (Bij - (R(i,j) * dRii))/Rii;
                            }
                        }
                    }
                
                    return dC;
                }

Eigen::MatrixXd invdChol(const Eigen::Ref<Eigen::MatrixXd,0,Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &R,
                         const Eigen::Ref<Eigen::MatrixXd,0,Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &Rinv,
                         const Eigen::Ref<Eigen::MatrixXd,0,Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &A)
                {
                    Eigen::MatrixXd dC = dChol(R,A);

                    return (Rinv*dC)*Rinv;
                }

Eigen::MatrixXd computeV2(const Eigen::Ref<Eigen::MatrixXd,0,Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> &Vpr,
                          py::list dRdRhos,
                          size_t Hcols)
                {
                    /*
                    Computes second term of the smoothing penalty uncertainty correction proposed by Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models

                    **Note**: The arrays in dRdRhos **must** be in column-major (Fortran) order!
                    */

                    Eigen::MatrixXd Vcc = Eigen::MatrixXd::Zero(Hcols,Hcols);
                    size_t npen = dRdRhos.size();
                    double Vjm;

                    for (size_t j = 0; j < Hcols; j++)
                    {
                        for (size_t m = j; m < Hcols; m++)
                        {
                            auto i = Eigen::seq(m,Hcols-1);
                            Vjm = 0.0;

                            for (size_t l = 0; l < npen; l++)
                            {
                                py::handle dRdRho_lh = dRdRhos[l];
                                py::array_t<double> dRdRho_l = py::cast<py::array>(dRdRho_lh);
                                py::buffer_info info_l = dRdRho_l.request();

                                double *ptr_l = static_cast<double *>(info_l.ptr);

                                Eigen::Map<Eigen::MatrixXd> ml(ptr_l,Hcols,Hcols);

                                for (size_t k = 0; k < npen; k++)
                                {
                                    py::handle dRdRho_kh = dRdRhos[k];
                                    py::array_t<double> dRdRho_k = py::cast<py::array>(dRdRho_kh);
                                    py::buffer_info info_k = dRdRho_k.request();

                                    double *ptr_k = static_cast<double *>(info_k.ptr);

                                    Eigen::Map<Eigen::MatrixXd> mk(ptr_k,Hcols,Hcols);

                                    Vjm += (mk(j,i).array() * Vpr(k,l) * ml(m,i).array()).sum();

                                }
                            }
                            Vcc(j,m) = Vcc(m,j) = Vjm;
                        }
                    }
                   return Vcc;
                }


PYBIND11_MODULE(dChol, m) {
    m.def("dChol", &dChol, py::arg("R").noconvert(), py::arg("A").noconvert(), "Let R be the transpose of Cholesky factor of some matrix H + a*A. Function returns derivative of this sum with respect to a.");
    m.def("invdChol", &invdChol, py::arg("R").noconvert(), py::arg("Rinv").noconvert(), py::arg("A").noconvert(), "Let R be the transpose of Cholesky factor of some matrix H + a*A and Rinv be the inverse of R. Function returns derivative of inverse of this sum with respect to a.");
    m.def("computeV2", &computeV2, "Computes second term of the smoothing penalty uncertainty correction proposed by Wood, S. N., Pya, N., Saefken, B., (2016).");
}