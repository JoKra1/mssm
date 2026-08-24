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

Eigen::VectorXd A1(
                   Eigen::Ref<Eigen::MatrixXd> umat,
                   Eigen::Ref<VectorXi64> idx, // Optionally after dropping rows
                   size_t col
)
{
    return umat(Eigen::all,col).eval()(idx,0);
}

Eigen::VectorXd A2(
                   py::list uMats,
                   py::list indices, // Optionally after dropping rows
                   Eigen::Ref<VectorXi64> ps,
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
        py::array_t<long long int> i_i = py::cast<py::array>(i_ih);
        py::buffer_info info_ii = i_i.request();

        long long int *ptr_ii = static_cast<long long int *>(info_ii.ptr);

        Eigen::Map<VectorXi64> idx(ptr_ii, n);

        Xj = (Xj.array() * m(idx, ji).array()).matrix();
    }
    
    return Xj;
}

Eigen::VectorXd A2Q(
                   py::list uMats,
                   py::list indices,
                   VectorXi64 ps,
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
            py::array_t<long long int> i_i = py::cast<py::array>(i_ih);
            py::buffer_info info_ii = i_i.request();

            long long int *ptr_ii = static_cast<long long int *>(info_ii.ptr);

            Eigen::Map<VectorXi64> idx(ptr_ii, n);

            Eigen::VectorXd mr = m(idx(ri), ji);
            //py::print(mr);

            Xr = (Xr.array() * mr.array()).matrix();
        }

        Xj(ri) = Xr.dot(Qi);
    }
    
    return Xj;
}

Eigen::VectorXd A3(Eigen::Ref<Eigen::MatrixXd> umat,
                   Eigen::Ref<Eigen::VectorXd> y,
                   Eigen::Ref<VectorXi64> idx, // Needs to have ridx dropped
                   VectorXi64 cidx
)
{
    Eigen::VectorXd ybar;
    ybar.setZero(umat.rows());

    for (size_t ri = 0; ri < y.rows(); ri++)
    {
        // np.sum(y[dt.indices[mi][ridx] == ri, :], axis=0)
        ybar(idx(ri)) += y(ri);
    }
    
    return umat(Eigen::all,cidx).transpose() * ybar;

}

Eigen::VectorXd A4(py::list uMats,
                   py::list indices,
                   Eigen::Ref<VectorXi64> ps,
                   size_t n,
                   size_t n_t,
                   size_t n_c,
                   VectorXi64 cidx,
                   Eigen::Ref<Eigen::MatrixXd> y,
                   bool hasQ,
                   Eigen::Ref<Eigen::MatrixXd> Q
)
{
    py::list uMats2;
    py::list indices2;
    VectorXi64 ps2;
    ps2.setZero(n);
    size_t pd = 1;

    for (size_t i = 0; i < n_t - 1; i++)
    {
        /*
        Loop over all but last marginals and
        get matrix and index buffer.
        */
        py::handle m_ih = uMats[i];
        py::array_t<double> m_i = py::cast<py::array>(m_ih);
        py::buffer_info info_i = m_i.request();

        double *ptr_i = static_cast<double *>(info_i.ptr);

        Eigen::Map<Eigen::MatrixXd> m(ptr_i,m_i.shape(0),m_i.shape(1));

        py::handle i_ih = indices[i];
        py::array_t<long long int> i_i = py::cast<py::array>(i_ih);
        py::buffer_info info_ii = i_i.request();

        long long int *ptr_ii = static_cast<long long int *>(info_ii.ptr);

        Eigen::Map<VectorXi64> idx(ptr_ii, n);
        
        // Collect
        uMats2.append(m);
        indices2.append(idx);
        ps2(i) = ps(i);
        pd *= ps(i);
    }

    // Get everything from last marginal
    size_t i = n_t - 1;
    size_t pdj = ps(i);
    size_t n_k = n_c;

    // Also need matrix and index vector
    py::handle m_ih = uMats[i];
    py::array_t<double> m_i = py::cast<py::array>(m_ih);
    py::buffer_info info_i = m_i.request();

    double *ptr_i = static_cast<double *>(info_i.ptr);

    Eigen::Map<Eigen::MatrixXd> m(ptr_i,m_i.shape(0),m_i.shape(1));
    VectorXi64 mcidx = VectorXi64::LinSpaced(Eigen::Sequential,m.cols(),0,m.cols() - 1);

    py::handle i_ih = indices[i];
    py::array_t<long long int> i_i = py::cast<py::array>(i_ih);
    py::buffer_info info_ii = i_i.request();

    long long int *ptr_ii = static_cast<long long int *>(info_ii.ptr);

    Eigen::Map<VectorXi64> idx(ptr_ii, n);

    // Correct for y
    uMats2.append(y);
    ps2(i) = 1;
    VectorXi64 yidx = VectorXi64::LinSpaced(Eigen::Sequential,y.rows(),0,y.rows() - 1);
    indices2.append(yidx);

    if (hasQ)
    {
        n_k += 1;
    }

    Eigen::VectorXd v;
    v.setZero(n_k);

    for (size_t l = 0; l < pd; l++)
    {

        Eigen::VectorXd Al = A2(uMats2,
                                indices2,
                                ps2,
                                pd,
                                n,
                                n_t,
                                l
                            );
        
        Eigen::VectorXd vl = A3(m, Al, idx, mcidx);
        v(Eigen::seq(l * pdj, (l * pdj + pdj - 1))) = vl;
    }
    
    if (hasQ)
    {
        return (Q.transpose() * v).eval()(cidx);
    }
    
    return v(cidx);
}

Eigen::VectorXd A5(
                   Eigen::Ref<Eigen::MatrixXd> umat,
                   Eigen::Ref<Eigen::VectorXd> beta,
                   VectorXi64 idx,
                   VectorXi64 cidx
)
{
    return (umat(Eigen::all,cidx) * beta).eval()(idx);
}

Eigen::VectorXd A6(py::list uMatsA,
                   py::list indicesA, // optionally after dropping rows
                   VectorXi64 indexC,
                   VectorXi64 psA,
                   size_t qA,
                   size_t n, // optionally after dropping rows
                   size_t n_t,
                   Eigen::Ref<Eigen::MatrixXd> C
)
{
    Eigen::VectorXd f;
    f.setZero(n);

    for (size_t l = 0; l < qA; l++)
    {
        Eigen::VectorXd Al = A2(uMatsA,
                                indicesA,
                                psA,
                                qA,
                                n,
                                n_t,
                                l
                            );

        f = (f.array() + (C(indexC,l).array() * Al.array())).matrix();
    }
    return f;
}

Eigen::VectorXd Xrtensor(
                   py::list uMats,
                   VectorXi64 rows, // desired row in marginal for each marginal
                   VectorXi64 ps,
                   size_t q,
                   size_t n_t,
                   VectorXi64 j,
                   size_t n_c
)
{
    VectorXi64 j2 = j;

    Eigen::VectorXd Xr;
    Xr.setOnes(n_c);

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
        
        // Get marginal row
        Eigen::VectorXd mr = m(rows(i), ji);
        //py::print(mr);

        Xr = (Xr.array() * mr.array()).matrix();
    }

    return Xr;

}

template<typename WMatrix>
Eigen::MatrixXd XTWXA(
                    py::list uMatsj,
                    py::list uMatsk,
                    py::list indicesj,
                    py::list indicesk,
                    Eigen::Ref<VectorXi64> psj,
                    Eigen::Ref<VectorXi64> psk,
                    size_t nj,
                    size_t nk,
                    size_t n_tj,
                    size_t n_tk,
                    size_t n_cj, // After absorbing constraints
                    size_t n_ck, // After absorbing constraints
                    size_t qk,
                    VectorXi64 cidxj,
                    VectorXi64 cidxk,
                    bool hasQj,
                    bool hasQk,
                    Eigen::Ref<Eigen::MatrixXd> Qj,
                    Eigen::Ref<Eigen::MatrixXd> Qk,
                    bool hasW,
                    const WMatrix &W                    
)
{   

    // Actual number of columns that need extraction for term on the right
    size_t n_cols = ((hasQk) ? n_ck + 1 : cidxk.rows());
    size_t n_rows = ((hasQj) ? n_cj + 1 : cidxj.rows());

    Eigen::MatrixXd XTWX;
    XTWX.setZero(n_rows,n_cols);

    for (size_t pki = 0; pki < n_cols; pki++)
    {
        size_t i = ((hasQk) ? pki : cidxk(pki)); // col i to extract form X_k

        Eigen::VectorXd Xi;
        Xi.setZero(nk);

        // Get column i from X_k
        if (n_tk == 1)
        {
            // Single marginal smooth -> A1

            /*
            Get matrix and index buffer
            */
            py::handle m_ih = uMatsk[0];
            py::array_t<double> m_i = py::cast<py::array>(m_ih);
            py::buffer_info info_i = m_i.request();

            double *ptr_i = static_cast<double *>(info_i.ptr);

            Eigen::Map<Eigen::MatrixXd> m(ptr_i,m_i.shape(0),m_i.shape(1));

            py::handle i_ih = indicesk[0];
            py::array_t<long long int> i_i = py::cast<py::array>(i_ih);
            py::buffer_info info_ii = i_i.request();

            long long int *ptr_ii = static_cast<long long int *>(info_ii.ptr);

            Eigen::Map<VectorXi64> idx(ptr_ii, nk);

            // Now get column i
            Xi = A1(m, idx, i);
        }
        else
        {
            // Tensor smooth -> A2
            Xi = A2(
                    uMatsk,
                    indicesk,
                    psk,
                    qk,
                    nk,
                    n_tk,
                    i
                );
        }

        //py::print("Column of j done", pki);

        // Optionally account for W
        if (hasW)
        {
            Xi = W * Xi;
        }

        // Form XTWXi
        if (n_tj == 1)
        {
            // Single marginal smooth -> A3

            /*
            Get matrix and index buffer
            */
            py::handle m_ih = uMatsj[0];
            py::array_t<double> m_i = py::cast<py::array>(m_ih);
            py::buffer_info info_i = m_i.request();

            double *ptr_i = static_cast<double *>(info_i.ptr);

            Eigen::Map<Eigen::MatrixXd> m(ptr_i,m_i.shape(0),m_i.shape(1));

            py::handle i_ih = indicesj[0];
            py::array_t<long long int> i_i = py::cast<py::array>(i_ih);
            py::buffer_info info_ii = i_i.request();

            long long int *ptr_ii = static_cast<long long int *>(info_ii.ptr);

            Eigen::Map<VectorXi64> idx(ptr_ii, nj);

            XTWX(Eigen::all,pki) = A3(m, Xi, idx, cidxj);

        }
        else
        {
            // Tensor smooth -> A4 (MUST BE WITH Q FALSE)
            XTWX(Eigen::all,pki) = A4(uMatsj,
                                      indicesj,
                                      psj,
                                      nj,
                                      n_tj,
                                      ((hasQj) ? n_cj + 1 : n_cj),
                                      ((hasQj) ?
                                        VectorXi64::LinSpaced(Eigen::Sequential,
                                                              n_cj + 1, 0, n_cj) :
                                        cidxj),
                                      Xi,
                                      false,
                                      Qj
                                    );
        }

        //py::print("Product with column j done",pki);
        
    }

    if (hasQj)
    {
        XTWX = Qj.transpose() * XTWX;
        XTWX = XTWX(cidxj, Eigen::all).eval(); // Can now drop columns from j
    }

    if(hasQk)
    {
        XTWX = XTWX * Qk;
        XTWX = XTWX(Eigen::all, cidxk).eval(); // Can now drop columns from k
    }

    return XTWX;
    
}

Eigen::MatrixXd XTWXD(
                    py::list uMatsj,
                    py::list uMatsk,
                    py::list indicesj,
                    py::list indicesk,
                    Eigen::Ref<VectorXi64> psj,
                    Eigen::Ref<VectorXi64> psk,
                    size_t nj,
                    size_t nk,
                    size_t n_tj,
                    size_t n_tk,
                    size_t n_cj,
                    size_t n_ck,
                    size_t qk,
                    VectorXi64 cidxj,
                    VectorXi64 cidxk,
                    bool hasQj,
                    bool hasQk,
                    Eigen::Ref<Eigen::MatrixXd> Qj,
                    Eigen::Ref<Eigen::MatrixXd> Qk,
                    bool hasW,
                    Eigen::Ref<Eigen::MatrixXd> W
)
{

    return XTWXA(uMatsj,
                uMatsk,
                indicesj,
                indicesk,
                psj,
                psk,
                nj,
                nk,
                n_tj,
                n_tk,
                n_cj,
                n_ck,
                qk,
                cidxj,
                cidxk,
                hasQj,
                hasQk,
                Qj,
                Qk,
                hasW,
                W
               );
    
    
}

Eigen::MatrixXd XTWXS(
                    py::list uMatsj,
                    py::list uMatsk,
                    py::list indicesj,
                    py::list indicesk,
                    Eigen::Ref<VectorXi64> psj,
                    Eigen::Ref<VectorXi64> psk,
                    size_t nj,
                    size_t nk,
                    size_t n_tj,
                    size_t n_tk,
                    size_t n_cj,
                    size_t n_ck,
                    size_t qk,
                    VectorXi64 cidxj,
                    VectorXi64 cidxk,
                    bool hasQj,
                    bool hasQk,
                    Eigen::Ref<Eigen::MatrixXd> Qj,
                    Eigen::Ref<Eigen::MatrixXd> Qk,
                    bool hasW,
                    long long int Wrows, long long int Wcols, long long int Wnnz,
                    py::array_t<double, py::array::f_style | py::array::forcecast> Wdata,
                    py::array_t<long long int, py::array::f_style | py::array::forcecast> Widptr,
                    py::array_t<long long int, py::array::f_style | py::array::forcecast> Windices
)
{

    // Get W
    Eigen::Map<Eigen::SparseMatrix<double,0,long long int>> W(Wrows,Wcols,Wnnz,
        (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Widptr.data(),
        (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Windices.data(),
        (Eigen::SparseMatrix<double,0,long long int>::Scalar*) Wdata.data());
    

    return XTWXA(uMatsj,
                uMatsk,
                indicesj,
                indicesk,
                psj,
                psk,
                nj,
                nk,
                n_tj,
                n_tk,
                n_cj,
                n_ck,
                qk,
                cidxj,
                cidxk,
                hasQj,
                hasQk,
                Qj,
                Qk,
                hasW,
                W
               );
    
    
}

PYBIND11_MODULE(discrete, m) {
    m.def("A1", &A1, "Algorithm A1 from Wood et al., 2017.");
    m.def("A2", &A2, "Algorithm A2 from Wood et al., 2017.");
    m.def("A2Q", &A2Q, "Modified version of Algorithm A2 from Wood et al., 2017 to account for constraint matrix Q.");
    m.def("A3", &A3, "Algorithm A3 from Wood et al., 2017.");
    m.def("A4", &A4, "Algorithm A4 from Wood et al., 2017.");
    m.def("A5", &A5, "Algorithm A5 from Wood et al., 2017.");
    m.def("A6", &A6, "Algorithm A6 from Wood et al., 2017.");
    m.def("XTWXD", &XTWXD, "Compute X.TWX based on Wood et al., 2017 with dense matrix W.");
    m.def("XTWXS", &XTWXS, "Compute X.TWX based on Wood et al., 2017 with sparse matrix W.");
    m.def("Xrtensor", &Xrtensor, "Extract row of tensor product before correcting for constraint matrix Q.");
}