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
#include <random>

namespace py = pybind11;

double compute_energy(
    const Eigen::Ref<Eigen::MatrixXd> &r,
    const Eigen::SparseMatrix<double,0,long long int> &Minv
)
/*
Computes current kinetic energy of Hamiltonian as defined by Betancourt (2013,2018).
*/
{
    Eigen::MatrixXd energy = 0.5 * r.transpose() * Minv * r;
    return energy(0,0);
}


std::tuple<Eigen::MatrixXd, // stateprime
           Eigen::MatrixXd, // gradprime
           Eigen::MatrixXd, // rprime
           double // llkprime
>
leap_frog(
    const Eigen::Ref<Eigen::MatrixXd> &state,
    const Eigen::Ref<Eigen::MatrixXd> &grad,
    const Eigen::Ref<Eigen::MatrixXd> &r,
    const Eigen::SparseMatrix<double,0,long long int> &Minv,
    double epsilon,
    const std::function<double(const Eigen::Ref<Eigen::MatrixXd>&)> &llk_fun,
    const std::function<Eigen::MatrixXd(const Eigen::Ref<Eigen::MatrixXd>&)> &grad_fun
)
/*
Leapfrog algorithm as described in Algorithm 1 of Hoffman & Gelman (2014). Adapted to work
with inverse of metric ``Minv``, as described in their discussion.
*/
{
    
    // Define storeage for state', grad', and r'
    Eigen::MatrixXd stateprime;
    Eigen::MatrixXd gradprime;
    Eigen::MatrixXd rprime;
    
    // First update to rprime and update to stateprime
    rprime = r + 0.5 * epsilon * grad;
    stateprime = state + epsilon * Minv * rprime;

    // Compute new grad and llk
    double llkprime = llk_fun(stateprime);
    gradprime = grad_fun(stateprime);

    // Second update to rprime
    rprime += 0.5 * epsilon * gradprime;

    return std::make_tuple(std::move(stateprime),std::move(gradprime),std::move(rprime),llkprime);
}

int check_dynamic_divergence(
    const Eigen::Ref<Eigen::MatrixXd> &rb1,
    const Eigen::Ref<Eigen::MatrixXd> &rb2,
    const Eigen::Ref<Eigen::MatrixXd> &re1,
    const Eigen::Ref<Eigen::MatrixXd> &re2,
    const Eigen::Ref<Eigen::MatrixXd> &rsum1,
    const Eigen::Ref<Eigen::MatrixXd> &rsum2,
    const Eigen::SparseMatrix<double,0,long long int> &Minv
)
/*
Computes the dynamic termination criterion proposed by M. J. Betancourt (2013,2018)
*/
{
    int s = 1; // Divergence criterion 
    Eigen::MatrixXd rsum = rsum1 + rsum2;
    Eigen::MatrixXd rsum_sub = rsum1 + rb2; // first rsum1 + rb2 then rsum2 + re1

    Eigen::MatrixXd rbsharp = Minv * rb1;
    Eigen::MatrixXd r2bsharp = Minv * rb2;
    Eigen::MatrixXd r1esharp = Minv * re1;
    Eigen::MatrixXd resharp = Minv * re2;

    // Now integrate over r (Betancourt, 2013,2018)...

    // ... first across sub-trees ...
    Eigen::MatrixXd Db = rbsharp.transpose() * rsum;
    Eigen::MatrixXd De = resharp.transpose() * rsum;

    s *= (int(Db(0,0) > 0) * int(De(0,0) > 0));

    // ... then between sub-trees once ...
    Db = rbsharp.transpose() * rsum_sub;
    De = r2bsharp.transpose() * rsum_sub;

    s *= (int(Db(0,0) > 0) * int(De(0,0) > 0));

    // ... and twice.
    rsum_sub = rsum2 + re1;

    Db = r1esharp.transpose() * rsum_sub;
    De = resharp.transpose() * rsum_sub;

    s *= (int(Db(0,0) > 0) * int(De(0,0) > 0));

    // Now return
    return s;
}


std::tuple<Eigen::MatrixXd, // statem
           Eigen::MatrixXd, // statep
           Eigen::MatrixXd, // stateprime
           Eigen::MatrixXd, // rm
           Eigen::MatrixXd, // rp
           Eigen::MatrixXd, // rsum
           size_t, // nprime
           int, // sprime
           double, //aprime
           size_t, // naprime
           double // Lprime
>
build_tree(
    const Eigen::Ref<Eigen::MatrixXd> &state,
    const Eigen::Ref<Eigen::MatrixXd> &r,
    const Eigen::SparseMatrix<double,0,long long int> &Minv,
    double logu,
    int v,
    size_t j,
    double epsilon,
    double H0,
    double DeltaMax,
    const std::function<double(const Eigen::Ref<Eigen::MatrixXd>&)> &llk_fun,
    const std::function<Eigen::MatrixXd(const Eigen::Ref<Eigen::MatrixXd>&)> &grad_fun
)
/*
Tree building algorithm as described in Algorithm 6 of Hoffman & Gelman (2014), but adapted
to work with the dynamic termination criterion for Riemannian metrices proposed by
M. J. Betancourt (2013,2018) - requiring book-keeping for the additional rsum variable.
*/
{
    // Single prime variables
    Eigen::MatrixXd statem;
    Eigen::MatrixXd statep;
    Eigen::MatrixXd stateprime;
    Eigen::MatrixXd rm;
    Eigen::MatrixXd rp;
    Eigen::MatrixXd rsum;
    size_t nprime;
    int sprime;
    double aprime;
    size_t naprime;
    double Lprime;
    
    // Base case
    if (j == 0)
    {
        Eigen::MatrixXd grad = grad_fun(state);
        Eigen::MatrixXd gradprime;
        Eigen::MatrixXd rprime;
        
        std::tie(stateprime,gradprime,rprime,Lprime) = leap_frog(state,grad,r,Minv,
                                                                 v*epsilon,llk_fun,
                                                                 grad_fun);
    
        double Hprime = Lprime - compute_energy(rprime,Minv);

        // log of prob ratios
        double lp = Hprime - H0;

        // Start collection
        nprime = logu <= Hprime;
        sprime = logu < DeltaMax + Hprime;
        aprime = lp > 0.0 ? 1.0 : exp(lp);
        naprime = 1;

        rsum = rprime;
        statem = stateprime;
        statep = stateprime;
        rm = rprime;
        rp = rprime;

        //py::print(logu,Hprime,sprime);

        return std::make_tuple(std::move(statem),std::move(statep),std::move(stateprime),
                               std::move(rm),std::move(rp),std::move(rsum),
                               nprime,sprime,aprime,naprime,Lprime);
    }
    else
    {
        // Recursion - build left and right sub-trees implicitly
        std::tie(statem,statep,stateprime,rm,rp,
                 rsum,nprime,sprime,
                 aprime,naprime,Lprime) = build_tree(state,r,Minv,logu,v,j-1,epsilon,H0,
                                                     DeltaMax,llk_fun,grad_fun);
            
        if (sprime == 1)
        {
            
            // Double primes and extra variables to compute points in terms of end/beginning
            // for easier implementation of divergence check by Betancourt (2013,2018)
            Eigen::MatrixXd statep2; // + step for backward integration (at beginning of tree 2)
            Eigen::MatrixXd statem2; // - step for forward integration (at beginning of tree 2)
            Eigen::MatrixXd stateprime2; // proposal of tree 2
            Eigen::MatrixXd rb1; // Momentum at begin of tree 1
            Eigen::MatrixXd re1; // Momentum at end of tree 1
            Eigen::MatrixXd rb2; // Momentum at begin of tree 2
            Eigen::MatrixXd re2; // Momentum at end of tree 2
            Eigen::MatrixXd rsum2; // Integration over tree 2
            Eigen::MatrixXd rsum_sub; // first rsum + rb2 then rsum2 + re1

            // Rest of prime variables
            size_t nprime2;
            int sprime2;
            double aprime2;
            size_t naprime2;
            double Lprime2;        
            
            if (v == -1)
            {   

                // Backward integration - second sub-tree is to the left
                
                rb1 = rp; // momentum at beginning of tree 1
                re1 = rm; // momentum at end of tree 1
                
                // Now integrate
                std::tie(statem,statep2,stateprime2,rm,rb2,
                         rsum2,nprime2,sprime2,
                         aprime2,naprime2,Lprime2) = build_tree(statem,rm,Minv,logu,v,j-1,epsilon,H0,
                                                                DeltaMax,llk_fun,grad_fun);
                
                re2 = rm; // momentum at end of tree 2

            }
            else
            {
                // Forward integration - second sub-tree is to the right

                rb1 = rm; // momentum at beginning of tree 1
                re1 = rp; // momentum at end of tree 1
                
                // Now integrate
                std::tie(statem2,statep,stateprime2,rb2,rp,
                         rsum2,nprime2,sprime2,
                         aprime2,naprime2,Lprime2) = build_tree(statep,rp,Minv,logu,v,j-1,epsilon,H0,
                                                                DeltaMax,llk_fun,grad_fun);
                                                                
                re2 = rp; // momentum at end of tree 2
            }

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> U(0.0, 1.0);
            
            // Accept state'' with probability
            if (
                (nprime + nprime2 > 0) &
                (((1.0 * nprime2) / (nprime + nprime2)) > U(gen))
            )
            {
                stateprime = stateprime2;
                Lprime = Lprime2;
            }

            aprime += aprime2;
            naprime += naprime2;

            // integrate over r (Betancourt, 2013,2018) to check convergence
            sprime = sprime2 * check_dynamic_divergence(rb1,rb2,re1,re2,rsum,rsum2,Minv);
            
            // Update overall rsum after check
            rsum += rsum2;

            // Now update nprime and return
            nprime += nprime2;

        }

        // Return else case
        return std::make_tuple(std::move(statem),std::move(statep),std::move(stateprime),
                               std::move(rm),std::move(rp),std::move(rsum),
                               nprime,sprime,aprime,naprime,Lprime);
    }
}


double find_reasonable_epsilon(
    const Eigen::Ref<Eigen::MatrixXd> &state,
    const Eigen::Ref<Eigen::MatrixXd> &grad,
    long long int Mrows,
    long long int Mcols,
    long long int Mnnz,
    py::array_t<double, py::array::f_style | py::array::forcecast> Mdata,
    py::array_t<long long int, py::array::f_style | py::array::forcecast> Midptr,
    py::array_t<long long int, py::array::f_style | py::array::forcecast> Mindices,
    double L,
    std::function<double(const Eigen::Ref<Eigen::MatrixXd>&)> llk_fun,
    std::function<Eigen::MatrixXd(const Eigen::Ref<Eigen::MatrixXd>&)> grad_fun,
    std::function<Eigen::MatrixXd()> r_sampler_fun
)
/*
Heuristic algorithm to determine an initial value for epsilon as described in Algorithm 4 of
Hoffman & Gelman (2014).
*/
{
    // Build Minv
    Eigen::Map<Eigen::SparseMatrix<double,0,long long int>> Minv (  
        Mrows,Mcols,Mnnz,
        (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Midptr.data(),
        (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Mindices.data(),
        (Eigen::SparseMatrix<double,0,long long int>::Scalar*) Mdata.data()
    );
    
    // Init epsilon
    double epsilon = 0.1;

    // Get initial r
    Eigen::MatrixXd r = r_sampler_fun();

    // Define storeage for state', grad', and r'
    Eigen::MatrixXd stateprime;
    Eigen::MatrixXd gradprime;
    Eigen::MatrixXd rprime;
    double Lprime;

    double H = L - compute_energy(r,Minv);

    std::tie(stateprime,gradprime,rprime,Lprime) = leap_frog(state,grad,r,Minv,
                                                             epsilon,llk_fun,
                                                             grad_fun);
    
    double Hprime = Lprime - compute_energy(rprime,Minv);

    // log of prob ratios
    double lp = Hprime - H;

    // Compute alpha parameter
    int idx = lp > log(0.5);
    int alpha = 2 * idx - 1;

    // Decrease or increase epsilon
    while (lp * alpha > log(2) * -1 * alpha)
    {
        epsilon *= pow(2,alpha);

        // Update Hamiltonian and log of prob ratio
        std::tie(stateprime,gradprime,rprime,Lprime) = leap_frog(state,grad,r,Minv,
                                                                 epsilon,llk_fun,
                                                                 grad_fun);
        
        Hprime = Lprime - compute_energy(rprime,Minv);

        lp = Hprime - H;
    }

    return epsilon;
}


std::tuple<Eigen::VectorXd,
           Eigen::MatrixXd,
           double,
           double,
           double
>advance_chain(
            size_t m,
            size_t Madapt,
            size_t steps,
            double cL,
            Eigen::MatrixXd cstate,
            long long int Mrows,
            long long int Mcols,
            long long int Mnnz,
            py::array_t<double, py::array::f_style | py::array::forcecast> Mdata,
            py::array_t<long long int, py::array::f_style | py::array::forcecast> Midptr,
            py::array_t<long long int, py::array::f_style | py::array::forcecast> Mindices,
            double epsilon,
            double epsilonbar,
            double Hbar,
            double mu,
            double delta,
            double kappa,
            double gamma,
            int t0,
            size_t max_j,
            std::function<double(const Eigen::Ref<Eigen::MatrixXd>&)> llk,
            std::function<Eigen::MatrixXd(const Eigen::Ref<Eigen::MatrixXd>&)> grad,
            std::function<Eigen::MatrixXd()> r_sampler
)
/*
Complete steps of algorithm 6 as defined by Hoffman & Gelman (2014) to sample next states and llks
of a log-joint via No-U-Turn Sampler defined by Hoffman & Gelman (2014).
Adapted to work with any Riemannian metric, based on M. J. Betancourt (2013,2018), as implemented
in STAN (Carpenter et al., 2017) as well.

References:
 - Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively Setting Path Lengths in\
    Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(47), 1593–1623.
 - Betancourt, M. J. (2013). Generalizing the No-U-Turn Sampler to Riemannian Manifolds\
    (No. arXiv:1304.1920). arXiv. https://doi.org/10.48550/arXiv.1304.1920
 - Betancourt, M. (2018). A Conceptual Introduction to Hamiltonian Monte Carlo\
    (No. arXiv:1701.02434). arXiv. https://doi.org/10.48550/arXiv.1701.02434
 - Carpenter, B., Gelman, A., Hoffman, M. D., Lee, D., Goodrich, B., Betancourt, M., … Riddell, A.\
    (2017). Stan: A Probabilistic Programming Language. Journal of Statistical Software,\
    https://doi.org/10.18637/jss.v076.i01
*/
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> U(0.0, 1.0);
    std::exponential_distribution<> ed(1);
    double DeltaMax = 1000;
    long long int n_coef = Mrows;

    // Construct Minv from buffers
    Eigen::Map<Eigen::SparseMatrix<double,0,long long int>> Minv (  
        Mrows,Mcols,Mnnz,
        (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Midptr.data(),
        (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Mindices.data(),
        (Eigen::SparseMatrix<double,0,long long int>::Scalar*) Mdata.data()
    );

    // Create storage for llks and states
    Eigen::VectorXd llks;
    Eigen::MatrixXd states;
    llks.setZero(steps);
    states.setZero(n_coef,steps);
    // py::print(m,cstate(0,0));

    for (size_t step = 0; step < steps; step++)
    {

        // Sample r0
        Eigen::MatrixXd r0 = r_sampler();

        // Compute H0
        double H0 = cL - compute_energy(r0,Minv);

        // Sample log(u) where u is U[0,exp(H0)]
        // If u1 is U[0,1]
        // Then: u would be 0 + (exp(H0) - 0)*u1 (see Wikipedia...)
        // Taking log(exp(H0)*u1) gives
        // H0 + log(u1). Finally, -log(u1) ~ exp(1)
        double logu = H0 - ed(gen);
        int v;

        // No prime variables
        Eigen::MatrixXd statem = cstate;
        Eigen::MatrixXd statep = cstate;
        Eigen::MatrixXd rm = r0;
        Eigen::MatrixXd rp = r0;
        Eigen::MatrixXd rsum = r0; // Integration variable required by Betancourt (2013,2018)
        size_t n = 1;
        int s = 1;
        double a;
        size_t na;
        size_t j = 0;

        // Single prime variables and throwaways for p and m
        Eigen::MatrixXd statep2; // + step for backward integration
        Eigen::MatrixXd statem2; // - step for forward integration
        Eigen::MatrixXd stateprime;
        Eigen::MatrixXd rb1; // Momentum at begin of old trajectory/tree
        Eigen::MatrixXd re1; // Momentum at end of old trajectory/tree
        Eigen::MatrixXd rb2; // Momentum at begin of new trajectory/tree
        Eigen::MatrixXd re2; // Momentum at end of new trajectory/tree
        Eigen::MatrixXd rsumprime; // Integration over tree 2
        size_t nprime;
        int sprime;
        double Lprime;

        while (s == 1 && j < max_j)
        {
            v = int(2 * (U(gen) >= 0.5)) - 1;

            if (v == -1)
            {
                // Backward integration

                rb1 = rp; // momentum at beginning of old tree
                re1 = rm; // momentum at end of old tree

                std::tie(statem,statep2,stateprime,rm,rb2,
                        rsumprime,nprime,sprime,
                        a,na,Lprime) = build_tree(statem,rm,Minv,logu,v,j,epsilon,H0,
                                                DeltaMax,llk,grad);
                
                re2 = rm; // momentum at end of new tree
            }
            else
            {
                // Forward integration

                rb1 = rm; // momentum at beginning of old tree
                re1 = rp; // momentum at end of old tree

                std::tie(statem2,statep,stateprime,rb2,rp,
                        rsumprime,nprime,sprime,
                        a,na,Lprime) = build_tree(statep,rp,Minv,logu,v,j,epsilon,H0,
                                                DeltaMax,llk,grad);
                
                re2 = rp; // momentum at end of new tree
            }

            //py::print(j,s,sprime,logu,v,nprime,n);

            if (sprime == 1)
            {
                // Accept state' with probability
                if (((1.0 * nprime) / n) > U(gen))
                {
                    cstate = stateprime;
                    cL = Lprime;
                }
            }

            // Update variables
            n += nprime;

            // Integrate over r across and between trees for dynamic divergence check
            // as proposed by (Betancourt, 2013,2018)
            s = sprime * check_dynamic_divergence(rb1,rb2,re1,re2,rsum,rsumprime,Minv);
            
            // Update overall rsum after check
            rsum += rsumprime;

            // Next doubling of tree
            j += 1;
        }

        m += 1;

        // Complete dual averaging to tune epsilon, final part of algorithm
        // 6 by Hoffman & Gelman (2014)
        if (m < Madapt)
        {
            double oomt = 1.0 / (m + t0);
            Hbar = ((1.0 - oomt) * Hbar) + oomt * (delta - (a/na));
            epsilon = exp(mu - (sqrt(m) / gamma) * Hbar);
            oomt = pow(m,-kappa);
            epsilonbar = exp(oomt * log(epsilon) + (1.0 - oomt) * log(epsilonbar));
        }
        else
        {
            epsilon = epsilonbar;
        }

        llks(step) = cL;
        states(Eigen::all,step) = cstate(Eigen::all,0);
    }
    // py::print(m,cstate(0,0));
    
    return std::make_tuple(std::move(llks),std::move(states), epsilon, epsilonbar, Hbar);
}


PYBIND11_MODULE(mcmc, m) {
    m.def("find_reasonable_epsilon", &find_reasonable_epsilon);
    m.def("advance_chain", &advance_chain);
}