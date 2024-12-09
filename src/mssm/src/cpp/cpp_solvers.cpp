#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include<Eigen/Sparse>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <limits>


namespace py = pybind11;

typedef Eigen::Vector<long long int, Eigen::Dynamic> VectorXi64;

std::tuple<std::vector<std::vector<double>>,std::vector<long long int>,std::vector<std::vector<long long int>>> dCholdRho(long long int Rrows, long long int Rcols, long long int Rnnz,
                                                                                                                          py::array_t<double, py::array::f_style | py::array::forcecast> Rdata,
                                                                                                                          py::array_t<long long int, py::array::f_style | py::array::forcecast> Ridptr,
                                                                                                                          py::array_t<long long int, py::array::f_style | py::array::forcecast> Rindices,
                                                                                                                          long long int Arows, long long int Acols, long long int Annz,
                                                                                                                          py::array_t<double, py::array::f_style | py::array::forcecast> Adata,
                                                                                                                          py::array_t<long long int, py::array::f_style | py::array::forcecast> Aidptr,
                                                                                                                          py::array_t<long long int, py::array::f_style | py::array::forcecast> Aindices){

    /*
    Derivative of transpose of Cholesky (R) of negative hessian of the penalized likelihood with respect to \rho.
    (A) holds derivative of negative hessian of the penalized likelihood with respect to said \rho.

    Based on equations in Sup. D in Wood, Pya, and Säfken (2016).
    */
    
    // Both A and R are row-storage!
    Eigen::Map<Eigen::SparseMatrix<double,1,long long int>> R(Rrows,Rcols,Rnnz,
                                                    (Eigen::SparseMatrix<double,1,long long int>::StorageIndex*) Ridptr.data(),
                                                    (Eigen::SparseMatrix<double,1,long long int>::StorageIndex*) Rindices.data(),
                                                    (Eigen::SparseMatrix<double,1,long long int>::Scalar*) Rdata.data());
    
    Eigen::Map<Eigen::SparseMatrix<double,1,long long int>> A(Arows,Acols,Annz,
                                                    (Eigen::SparseMatrix<double,1,long long int>::StorageIndex*) Aidptr.data(),
                                                    (Eigen::SparseMatrix<double,1,long long int>::StorageIndex*) Aindices.data(),
                                                    (Eigen::SparseMatrix<double,1,long long int>::Scalar*) Adata.data());

    // Also get a copy of R in column-storage for cheap indexing.
    Eigen::SparseMatrix<double,0,long long int> Rc = R;
    
    // Define vectors that will hold data, rows, and cols of desired derivative
    std::vector<std::vector<double>> dRdat(Rrows,std::vector<double>()); // One vector per row to preserve easy access to non-zero elements in loop below
    std::vector<long long int> dRrow;
    std::vector<std::vector<long long int>> dRcol(Rrows,std::vector<long long int>()); // Same as for dat

    /*
    Iterate over all columns in each row of R take care.

    Take care to keep track (via inner iterators) of non-zero elements in R and A in the current row.
    */

    for(Eigen::Index i=0; i<Rrows; ++i) // Loop also suggested in the Eigen tutorial: https://eigen.tuxfamily.org/dox/group__TutorialSparse.html
    {   

        /*
        Get iterators over columns of R & A for current row

        itR.col() needs to be checked against c essentially (but only if itR still has elements) and then itR needs to be increased.
        */
        Eigen::Map<Eigen::SparseMatrix<double,1,long long int>>::InnerIterator itR(R, i);
        Eigen::Map<Eigen::SparseMatrix<double,1,long long int>>::InnerIterator itA(A, i);

        // Initialize diagonal elements, which need to be accessed off-diagonal repeatedly.
        double Rii = 0;
        double dRii = 0;

        for(Eigen::Index j=i; j<Rcols; ++j)
        {
            
            /*
            Get itertor over rows of columns (i) and (j) in R
            */

            Eigen::SparseMatrix<double,0,long long int>::InnerIterator itRi(Rc, i);
            Eigen::SparseMatrix<double,0,long long int>::InnerIterator itRj(Rc, j);

            // Compute B_ij;
            double Bij = 0;
            if(itA && itA.col() == j)
            {
                Bij = itA.value();
                //py::print(Bij,itA.row(),itA.col());
                ++itA;
            }
            //py::print("Iteration: ", i,j, Bij);

            // Compute remainder of B_ij in WPS (2016)
            // Sum is zero for i==0.
            if(i > 0)
            {
                long long int k = 0;

                while(k < i)
                {   
                    double SBij = 0;
                    
                    if(itRi && itRi.row() == k)
                    {
                        for(size_t kj = 0; kj < dRcol[k].size(); ++kj)
                        {
                            if(dRcol[k][kj] == j)
                            {   
                                //py::print(k,i,j,itRi.value(),dRdat[k][kj],itRi.value()*dRdat[k][kj]);
                                SBij += itRi.value()*dRdat[k][kj];
                                break;
                            } else if (dRcol[k][kj] > j)
                            {
                                break;
                            }
                        }
                        ++itRi;
                    }

                    if(itRj && itRj.row() == k)
                    {
                        for(size_t ki = 0; ki < dRcol[k].size(); ++ki)
                        {
                            if(dRcol[k][ki] == i)
                            {
                                //py::print(k,i,j,itRj.value(),dRdat[k][ki],itRj.value()*dRdat[k][ki]);
                                SBij += itRj.value()*dRdat[k][ki];
                                break;
                            } else if (dRcol[k][ki] > i)
                            {
                                break;
                            }
                            
                        }
                        ++itRj;
                    }
                    
                    Bij -= SBij;
                    ++k;
                    
                }
            }

            //std::cout << Bij;
            //std::cout << "\n";

            // Now handle case if i == j
            double dR = 0;
            if(i==j)
            {   
                Rii = itR.value();
                dRii = 0.5 * Bij / Rii;
                dR = dRii;
                
                ++itR;

            } else if (j > i) // And case if j > i
            {
                dR = Bij;
                if(itR && itR.col() == j){
                    dR -= itR.value()*dRii;
                    //py::print(i,j,Bij,itR.value(),dRii,Rii,(Bij - itR.value()*dRii)/Rii);
                    ++itR;
                }
                dR /= Rii;
                
            }
            
            // Store dR, i, j in respective vectors..
            if(abs(dR) > 0){
                dRdat[i].push_back(dR);
                dRcol[i].push_back(j);
                dRrow.push_back(i);
            }
        }
    }
                                       
    return std::make_tuple(std::move(dRdat),std::move(dRrow),std::move(dRcol));
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

std::tuple<Eigen::SparseMatrix<double,0,long long int>,VectorXi64,Eigen::VectorXd,int> solve_coef(Eigen::VectorXd y, long long int Xrows, long long int Xcols, long long int Xnnz,
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

std::tuple<Eigen::SparseMatrix<double,0,long long int>,VectorXi64,Eigen::VectorXd,int> solve_coefXX(Eigen::VectorXd Xy, long long int Xrows, long long int Xcols, long long int Xnnz,
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

std::tuple<Eigen::SparseMatrix<double,0,long long int>,VectorXi64,Eigen::VectorXd,long long int,int> solve_coef_pqr(Eigen::VectorXd y, long long int Xrows, long long int Xcols, long long int Xnnz,
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

std::tuple<Eigen::SparseMatrix<double,0,long long int>,VectorXi64,VectorXi64,Eigen::VectorXd,long long int,int> solve_coef_pqr2(Eigen::VectorXd y, long long int Xrows, long long int Xcols, long long int Xnnz,
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
    coef = Qy.head(Xcols);

    // Extract root of X.T@X + S
    Eigen::SparseMatrix<double,0,long long int> R2 = solver2.matrixR().topRows(Xcols);

    // Now do the actual solve
    R2.triangularView<Eigen::Upper>().solveInPlace(coef);

    return std::make_tuple(R2,P1.indices(),P2.indices(),std::move(coef),solver2.rank(),0);
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

PYBIND11_MODULE(cpp_solvers, m) {
    m.doc() = "cpp solvers for sms (DC) GAMM estimation";
    m.def("dCholdRho", &dCholdRho, "Compute derivative of transpose of cholesky factor L of A.");
    m.def("chol", &chol, "Compute cholesky factor L of A");
    m.def("cholP", &cholP, "Compute cholesky factor L of A after applying a sparsity enhancing permutation to A");
    m.def("pqr", &pqr, "Perform column pivoted QR decomposition of A");
    m.def("pqrr", &pqrr, "Perform column pivoted QR decomposition of A, but only return R.");
    m.def("spqr", &spqr, "Perform column pivoted QR decomposition of symmetric matrix A, so that L - where A=L@L.T - is sparse.");
    m.def("solve_pqr", &solve_pqr, "Perform column pivoted QR decomposition of A, then solve for inverse of A");
    m.def("solve_am", &solve_am, "Solve additive model, return coefficient vector and inverse");
    m.def("solve_L", &solve_L, "Solve cholesky of XX+S");
    m.def("solve_LXX", &solve_LXX, "Solve cholesky of XX+S, but with XX + S pre-computed.");
    m.def("solve_coef", &solve_coef, "Solve additive model coefficients");
    m.def("solve_coef_pqr", &solve_coef_pqr2, "Solve additive model coefficients, using stable QR decomposition");
    m.def("solve_coefXX", &solve_coefXX, "Solve additive model coefficients, but with XX + S and Xy pre-computed.");
    m.def("solve_tr",&solve_tr,"Solve A*B = C, where A is lower triangular.");
    m.def("backsolve_tr",&backsolve_tr,"Solve A*B = C, where A is upper triangular.");
}