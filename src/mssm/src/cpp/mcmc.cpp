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
    const Eigen::SparseMatrix<double,0,long long int> &M
)
/*
Computes current kinetic energy of Hamiltonian after subtracting logp(state) as defined
by Betancourt (2013,2018).
*/
{
    Eigen::MatrixXd energy = 0.5 * r.transpose() * M * r;
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
    const Eigen::SparseMatrix<double,0,long long int> &M,
    double epsilon,
    const std::function<double(const Eigen::Ref<Eigen::MatrixXd>&)> &llk_fun,
    const std::function<Eigen::MatrixXd(const Eigen::Ref<Eigen::MatrixXd>&)> &grad_fun
)
/*
Leapfrog algorithm as described in Algorithm 1 of Hoffman & Gelman (2014). Adapted to work
with Riemannian metric ``M``, as described in their discussion.
*/
{
    
    // Define storeage for state', grad', and r'
    Eigen::MatrixXd stateprime;
    Eigen::MatrixXd gradprime;
    Eigen::MatrixXd rprime;
    
    // First update to rprime and update to stateprime
    rprime = r + 0.5 * epsilon * grad;
    stateprime = state + epsilon * M * rprime;

    // Compute new grad and llk
    double llkprime = llk_fun(stateprime);
    gradprime = grad_fun(stateprime);

    // Second update to rprime
    rprime += 0.5 * epsilon * gradprime;

    return std::make_tuple(std::move(stateprime),std::move(gradprime),std::move(rprime),llkprime);
}

double find_reasonable_epsilon(
    const Eigen::Ref<Eigen::MatrixXd> &state,
    const Eigen::Ref<Eigen::MatrixXd> &grad,
    const Eigen::SparseMatrix<double,0,long long int> &M,
    double L,
    const std::function<double(const Eigen::Ref<Eigen::MatrixXd>&)> &llk_fun,
    const std::function<Eigen::MatrixXd(const Eigen::Ref<Eigen::MatrixXd>&)> &grad_fun,
    const std::function<Eigen::MatrixXd()> &r_sampler_fun
)
/*
Heuristic algorithm to determine an initial value for epsilon as described in Algorithm 4 of
Hoffman & Gelman (2014).
*/
{
    // Init epsilon
    double epsilon = 0.1;

    // Get initial r
    Eigen::MatrixXd r = r_sampler_fun();

    // Define storeage for state', grad', and r'
    Eigen::MatrixXd stateprime;
    Eigen::MatrixXd gradprime;
    Eigen::MatrixXd rprime;
    double Lprime;

    double H = L - compute_energy(r,M);

    std::tie(stateprime,gradprime,rprime,Lprime) = leap_frog(state,grad,r,M,
                                                             epsilon,llk_fun,
                                                             grad_fun);
    
    double Hprime = Lprime - compute_energy(rprime,M);

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
        std::tie(stateprime,gradprime,rprime,Lprime) = leap_frog(state,grad,r,M,
                                                                 epsilon,llk_fun,
                                                                 grad_fun);
        
        Hprime = Lprime - compute_energy(rprime,M);

        lp = Hprime - H;
    }

    return epsilon;
}

std::tuple<Eigen::MatrixXd, // statem
           Eigen::MatrixXd, // statep
           Eigen::MatrixXd, // stateprime
           Eigen::MatrixXd, // rm
           Eigen::MatrixXd, // rp
           Eigen::MatrixXd, // rprime
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
    const Eigen::SparseMatrix<double,0,long long int> &M,
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
    Eigen::MatrixXd rprime;
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
        
        std::tie(stateprime,gradprime,rprime,Lprime) = leap_frog(state,grad,r,M,
                                                                 v*epsilon,llk_fun,
                                                                 grad_fun);
    
        double Hprime = Lprime - compute_energy(rprime,M);

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
                               std::move(rm),std::move(rp),std::move(rprime),std::move(rsum),
                               nprime,sprime,aprime,naprime,Lprime);
    }
    else
    {
        // Recursion - build left and right sub-trees implicitly

        std::tie(statem,statep,stateprime,rm,rp,
                 rprime,rsum,nprime,sprime,
                 aprime,naprime,Lprime) = build_tree(state,r,M,logu,v,j-1,epsilon,H0,
                                                     DeltaMax,llk_fun,grad_fun);
        
        if (sprime == 1)
        {
            
            // Double primes and throwaways for p and m
            Eigen::MatrixXd statemp;
            Eigen::MatrixXd statepm;
            Eigen::MatrixXd stateprime2;
            Eigen::MatrixXd rmp;
            Eigen::MatrixXd rpm;
            Eigen::MatrixXd rprime2;
            Eigen::MatrixXd rsum2;
            size_t nprime2;
            int sprime2;
            double aprime2;
            size_t naprime2;
            double Lprime2;
            
            if (v == -1)
            {
                // Backward integration
                std::tie(statem,statepm,stateprime2,rm,rpm,
                         rprime2,rsum2,nprime2,sprime2,
                         aprime2,naprime2,Lprime2) = build_tree(statem,rm,M,logu,v,j-1,epsilon,H0,
                                                                DeltaMax,llk_fun,grad_fun);
            }
            else
            {
                // Forward integration
                std::tie(statemp,statep,stateprime2,rmp,rp,
                         rprime2,rsum2,nprime2,sprime2,
                         aprime2,naprime2,Lprime2) = build_tree(statep,rp,M,logu,v,j-1,epsilon,H0,
                                                                DeltaMax,llk_fun,grad_fun);
            }

            // integrate over r (Betancourt, 2013,2018)

            rsum += rsum2;

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

            // Now new divergence check by Betancourt (2013,2018)
            Eigen::MatrixXd rpsharp = M * rp;
            Eigen::MatrixXd rmsharp = M * rm;
            Eigen::MatrixXd Dp = rpsharp.transpose() * rsum;
            Eigen::MatrixXd Dm = rmsharp.transpose() * rsum;

            sprime = sprime2 * (int(Dp(0,0) > 0) * int(Dm(0,0) > 0));

            nprime += nprime2;

        }
        // Return else case
        return std::make_tuple(std::move(statem),std::move(statep),std::move(stateprime),
                               std::move(rm),std::move(rp),std::move(rprime),std::move(rsum),
                               nprime,sprime,aprime,naprime,Lprime);
    }
}

class NUTS
/*
Class to hold current states of a No-U-Turn Sampler defined by Hoffman & Gelman (2014).
Adapted to work with any Riemannian metric, based on M. J. Betancourt (2013,2018).

References:
 - Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively Setting Path Lengths in\
    Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(47), 1593–1623.
 - Betancourt, M. J. (2013). Generalizing the No-U-Turn Sampler to Riemannian Manifolds\
    (No. arXiv:1304.1920). arXiv. https://doi.org/10.48550/arXiv.1304.1920
 - Betancourt, M. (2018). A Conceptual Introduction to Hamiltonian Monte Carlo\
    (No. arXiv:1701.02434). arXiv. https://doi.org/10.48550/arXiv.1701.02434
*/
{
public:
    // Constructor
    NUTS
    (
        long long int, // n_coef
        size_t, // Madapt
        double delta, // Expected Metropolis acceptance prob. Used to tune epsilon
        Eigen::MatrixXd, // init_coef
        std::function<double(const Eigen::Ref<Eigen::MatrixXd>&)>, // llk
        std::function<Eigen::MatrixXd(const Eigen::Ref<Eigen::MatrixXd>&)>, // grad
        std::function<Eigen::MatrixXd()>, // r sampler
        long long int, // Mrows
        long long int, // Mcols
        long long int, // Mnnz
        py::array_t<double, py::array::f_style | py::array::forcecast>, // Mdata
        py::array_t<long long int, py::array::f_style | py::array::forcecast>, // Midptr
        py::array_t<long long int, py::array::f_style | py::array::forcecast> // Mindices
    );
    // Advance chain
    void init_chain();
    std::tuple<double,Eigen::MatrixXd> advance_chain();
private:
    /*
    Need external functions for log-likelihood and gradient.
    Also an initial Riemannian metric. These will be passed to
    constructor.
    */
    std::function<double(const Eigen::Ref<Eigen::MatrixXd>&)> llk;
    std::function<Eigen::MatrixXd(const Eigen::Ref<Eigen::MatrixXd>&)> grad;
    std::function<Eigen::MatrixXd()> r_sampler;
    Eigen::SparseMatrix<double,0,long long int> M;
    // Need current state
    long long int n_coef;
    Eigen::MatrixXd cstate;
    double cL; // current log joint prob
    size_t max_j = 10;
    // And a bunch of hyper parameters, some of which we expose
    double DeltaMax = 10000;
    double epsilon;
    double mu;
    double epsilonbar = 1.0;
    size_t Madapt;
    size_t m = 0;
    int t0 = 10;
    double gamma = 0.05;
    double Hbar = 0.0;
    double kappa = 0.75;
    double delta = 0.6;

};

NUTS::NUTS(
    long long int n_coef_external,
    size_t Madapt,
    double delta,
    Eigen::MatrixXd init_coef,
    std::function<double(const Eigen::Ref<Eigen::MatrixXd>&)> llk_external,
    std::function<Eigen::MatrixXd(const Eigen::Ref<Eigen::MatrixXd>&)> grad_external,
    std::function<Eigen::MatrixXd()> r_sampler_external,
    long long int Mrows,
    long long int Mcols,
    long long int Mnnz,
    py::array_t<double, py::array::f_style | py::array::forcecast> Mdata,
    py::array_t<long long int, py::array::f_style | py::array::forcecast> Midptr,
    py::array_t<long long int, py::array::f_style | py::array::forcecast> Mindices
)
// Constructor for the NUTS sampler.
{
    n_coef = n_coef_external;
    llk = llk_external;
    grad = grad_external;
    r_sampler = r_sampler_external;

    M = Eigen::Map<Eigen::SparseMatrix<double,0,long long int>> (  
        Mrows,Mcols,Mnnz,
        (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Midptr.data(),
        (Eigen::SparseMatrix<double,0,long long int>::StorageIndex*) Mindices.data(),
        (Eigen::SparseMatrix<double,0,long long int>::Scalar*) Mdata.data()
    );

    // Init state (coef and llk)
    cstate = init_coef;
    cL = llk(cstate);

    // Set chosen hyper-parameters
    this->Madapt = Madapt;
    this->delta = delta;
}

void NUTS::init_chain()
{
    // Find suitable epsilon
    Eigen::MatrixXd cgrad = grad(cstate);
    
    epsilon = find_reasonable_epsilon(cstate,cgrad,M,cL,llk,grad,r_sampler);
    mu = log(10 * epsilon);
    m = 0;
    //py::print(epsilon);
}

std::tuple<double,
           Eigen::MatrixXd
>NUTS::advance_chain()
/*
Complete one step of algorithm 6 as defined by Hoffman & Gelman (2014) to sample next state.
*/
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> U(0.0, 1.0);
    std::exponential_distribution<> ed(1);

    // Sample r0
    Eigen::MatrixXd r0 = r_sampler();

    // Compute H0
    double H0 = cL - compute_energy(r0,M);

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
    Eigen::MatrixXd rsum;
    rsum.setZero(n_coef,1);
    size_t n = 1;
    int s = 1;
    double a;
    size_t na;
    size_t j = 0;

    // Single prime variables and throwaways for p and m
    Eigen::MatrixXd statemp;
    Eigen::MatrixXd statepm;
    Eigen::MatrixXd stateprime;
    Eigen::MatrixXd rmp;
    Eigen::MatrixXd rpm;
    Eigen::MatrixXd rprime;
    Eigen::MatrixXd rsumprime;
    size_t nprime;
    int sprime;
    double Lprime;

    while (s == 1 and j < max_j)
    {
        v = int(2 * (U(gen) >= 0.5)) - 1;

        if (v == -1)
        {
            // Backward integration
            std::tie(statem,statepm,stateprime,rm,rpm,
                     rprime,rsumprime,nprime,sprime,
                     a,na,Lprime) = build_tree(statem,rm,M,logu,v,j,epsilon,H0,
                                               DeltaMax,llk,grad);
        }
        else
        {
            // Forward integration
            std::tie(statemp,statep,stateprime,rmp,rp,
                     rprime,rsumprime,nprime,sprime,
                     a,na,Lprime) = build_tree(statep,rp,M,logu,v,j,epsilon,H0,
                                               DeltaMax,llk,grad);
        }

        // integrate over r (Betancourt, 2013,2018)
        rsum += rsumprime;
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

        // New divergence check by Betancourt (2013,2018)
        Eigen::MatrixXd rpsharp = M * rp;
        Eigen::MatrixXd rmsharp = M * rm;
        Eigen::MatrixXd Dp = rpsharp.transpose() * rsum;
        Eigen::MatrixXd Dm = rmsharp.transpose() * rsum;

        s = sprime * (int(Dp(0,0) > 0) * int(Dm(0,0) > 0));
        j += 1;
    }

    m += 1;

    // Complete dual averaging to tune epsilon, final part of algorithm 6 by Hoffman & Gelman (2014)
    if (m <= Madapt)
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
    //py::print(epsilon);
    
    return std::make_tuple(Lprime,cstate);
}


PYBIND11_MODULE(mcmc, m) {
    py::class_<NUTS>(m,"NUTS")
        .def(py::init<long long int,
             size_t,
             double,
             Eigen::MatrixXd,
             std::function<double(const Eigen::Ref<Eigen::MatrixXd>&)>,
             std::function<Eigen::MatrixXd(const Eigen::Ref<Eigen::MatrixXd>&)>,
             std::function<Eigen::MatrixXd()>,
             long long int,
             long long int,
             long long int,
             py::array_t<double, py::array::f_style | py::array::forcecast>,
             py::array_t<long long int, py::array::f_style | py::array::forcecast>,
             py::array_t<long long int, py::array::f_style | py::array::forcecast>>())
        .def("init_chain", &NUTS::init_chain)
        .def("advance_chain", &NUTS::advance_chain);
}