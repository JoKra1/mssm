#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <limits>

namespace py = pybind11;

py::array_t<
            double
> forward_resid(py::array_t<double, py::array::f_style | py::array::forcecast> cbs_a,
                py::array_t<double, py::array::f_style | py::array::forcecast> bs_a,
                py::array_t<double, py::array::f_style | py::array::forcecast> ds_a,
                py::array_t<double, py::array::f_style | py::array::forcecast> T_a,
                py::array_t<double, py::array::f_style | py::array::forcecast> pi_a,
                py::array_t<double, py::array::f_style | py::array::forcecast> weights_a,
                double scale, size_t n_T, size_t D, size_t n_S, size_t M, size_t event_width,
                bool starts_with_first, bool ends_with_last,bool ends_in_last, int hmp_code) {
    /*
    Compute forward "pseudo-residuals" (also known as "independent quantile residuals") as defined
    by Zucchini et al. (2017) and Dunn & Smyth (1996).

    References:
      - Zucchini, W., MacDonald, I. L., & Langrock, R. (2017). Hidden Markov Models for Time\
        Series: An Introduction Using R, Second Edition (2nd ed.). Chapman and Hall/CRC.\
        https://doi.org/10.1201/b20790
      - Dunn, P. K., & Smyth, G. K. (1996). Randomized Quantile Residuals. Journal of Computational\
        and Graphical Statistics, 5(3), 236–244. https://doi.org/10.2307/1390802
    */
   
    auto Lam_a = py::array_t<double>(n_T);
    auto resid_a = py::array_t<double>({n_T,M});
    auto lam_a = py::array_t<double>({n_T,D-1,n_S});
    auto clam_a = py::array_t<double>({n_T,M,D-1,n_S});
    Lam_a[py::make_tuple(py::ellipsis())] = 0.0;
    lam_a[py::make_tuple(py::ellipsis())] = 0.0;
    resid_a[py::make_tuple(py::ellipsis())] = 0.0;
    clam_a[py::make_tuple(py::ellipsis())] = 0.0;

    auto Lam = Lam_a.mutable_unchecked();
    auto lam = lam_a.mutable_unchecked();
    // p(Y <= y_t|y_1,...y_{t-1})
    auto resid = resid_a.mutable_unchecked();
    // p(Y <= y_t,S=j,D=d|y_1,...y_{t-1})
    auto clam = clam_a.mutable_unchecked(); 
    auto cbs = cbs_a.unchecked();
    auto bs = bs_a.unchecked();
    auto ds = ds_a.unchecked();
    auto T = T_a.unchecked();
    auto pi = pi_a.unchecked();
    auto weights = weights_a.unchecked();

    
    double llk = 0.0;
    double tp, b2, c2;
    for (size_t t = 0; t < n_T; t++)
    {
        for (size_t d = 0; d < (D-1); d++)
        {
            for (size_t j = 0; j < n_S; j++)
            {
                if (t == 0)
                {
                    if (hmp_code != 0)
                    {
                        lam(t,d,j) = pi(j)*bs(t,0,j)*ds(d,j);

                        for (size_t m = 0; m < M; m++)
                        {
                            clam(t,m,d,j) = pi(j)*cbs(t,m,0,j)*ds(d,j);
                        }
                    }
                    else
                    {
                        lam(t,d,j) = pi(j)*bs(t,j)*ds(d,j);

                        for (size_t m = 0; m < M; m++)
                        {
                            clam(t,m,d,j) = pi(j)*cbs(t,j,m)*ds(d,j);
                        }
                    }
                    
                    
                }
                else if ((t + d) < n_T || ends_with_last == false)
                {
                    // Probability of same state continuing
                    if ((d+1) <= (D-2))
                    {
                        
                        if (hmp_code != 0)
                        {
                            if (j % 2 == 1)
                            {
                                if (d <= event_width-1)
                                {
                                    /* bump */
                                    for (size_t m = 0; m < M; m++)
                                    {   
                                        // c prob for signal m in bump
                                        clam(t,m,d,j) = lam(t-1,d+1,j) *
                                                        cbs(t,m,(event_width-1) - d,j);
                                    }
                                    lam(t,d,j) = lam(t-1,d+1,j)*bs(t,(event_width-1) - d,j);
                                }
                                else
                                {
                                    lam(t,d,j) = 0;
                                }
                            }
                            else
                            {
                                for (size_t m = 0; m < M; m++)
                                {
                                    /* flat */

                                    // c prob for signal m in flat
                                    clam(t,m,d,j) = lam(t-1,d+1,j)*cbs(t,m,0,j);
                                }
                                lam(t,d,j) = lam(t-1,d+1,j)*bs(t,0,j);

                            }
                        }
                        else
                        {
                            lam(t,d,j) = lam(t-1,d+1,j)*bs(t,j);

                            for (size_t m = 0; m < M; m++)
                            {
                                clam(t,m,d,j) = lam(t-1,d+1,j)*cbs(t,j,m);
                            }
                        }
                        
                    }
                        
                    // Probability of other state ending on previous time point and transitioning
                    // to current state with dur d
                    tp = 0.0;
                    for (size_t i = 0; i < n_S; i++)
                    {
                        if (i == j)
                        {
                            continue;
                        }

                        tp += lam(t-1,0,i)*T(i,j);
                    }
                    
                    if (hmp_code != 0)
                    {
                        
                        b2 = bs(t,0,j);
                        if(j % 2 == 0)
                        {
                            // If we have an ar1 model first obs prob for flats depends on lag of
                            // previous bump (only for states > 0, but this is handled outside)
                            b2 = bs(t,1,j);
                        }

                        lam(t,d,j) += tp * b2 * ds(d,(starts_with_first) ? j : j + n_S);

                        for (size_t m = 0; m < M; m++)
                        {
                            
                            c2 = cbs(t,m,0,j);
                            if(j % 2 == 0)
                            {
                                // Same contamination for cbs
                                c2 = cbs(t,m,1,j);
                            }
                            
                            clam(t,m,d,j) += tp * c2 * ds(d,(starts_with_first) ? j : j + n_S);

                            // Scaling to get conditional probability
                            clam(t,m,d,j) /= Lam(t-1);
                        }
                        
                    }
                    else
                    {
                        lam(t,d,j) += tp * bs(t,j) * ds(d,(starts_with_first) ? j : j + n_S);

                        for (size_t m = 0; m < M; m++)
                        {
                            clam(t,m,d,j) += tp * cbs(t,j,m) *
                                             ds(d,(starts_with_first) ? j : j + n_S);

                            // Scaling to get conditional probability
                            clam(t,m,d,j) /= Lam(t-1);
                        }
                    }

                    // Scaling to get conditional probability
                    lam(t,d,j) /= Lam(t-1);
                    
                }
                    
                // Compute conditional probability of current obs given all previous obs
                if (t < (n_T - 1) || (ends_with_last == false))
                {   
                    // State does not have to end at last time point and also not in last state
                    // Before last time-point this is the right computation for all cases as well!
                    if (t < (n_T - 1) || ends_in_last == false)
                    {
                        Lam(t) += lam(t,d,j);
                        for (size_t m = 0; m < M; m++)
                        {
                            resid(t,m) += clam(t,m,d,j);
                        }
                        
                    }
                    // State does not have to end at last time point but in last state
                    else if (j == (n_S-1))
                    {
                        Lam(t) += lam(t,d,j);
                        for (size_t m = 0; m < M; m++)
                        {
                            resid(t,m) += clam(t,m,d,j);
                        }
                    }
                }
                else if (d == 0)
                {   
                    // State does have to end at last time point but not in last state
                    if (ends_in_last == false)
                    {
                        Lam(t) += lam(t,d,j);
                        for (size_t m = 0; m < M; m++)
                        {
                            resid(t,m) += clam(t,m,d,j);
                        }
                    }
                    // State does have to end at last time point and also in last state
                    else if (j == (n_S-1))
                    {
                        Lam(t) += lam(t,d,j);
                        for (size_t m = 0; m < M; m++)
                        {
                            resid(t,m) += clam(t,m,d,j);
                        }
                    }
                    
                }
            }
        }
        llk += log(Lam(t));
    }

    return resid_a;
}

py::array_t<
            double
> predictive_resid(py::array_t<double, py::array::f_style | py::array::forcecast> y_a,
                   py::array_t<double, py::array::f_style | py::array::forcecast> bs_a,
                   py::array_t<double, py::array::f_style | py::array::forcecast> mus_a,
                   py::array_t<double, py::array::f_style | py::array::forcecast> ds_a,
                   py::array_t<double, py::array::f_style | py::array::forcecast> T_a,
                   py::array_t<double, py::array::f_style | py::array::forcecast> pi_a,
                   py::array_t<double, py::array::f_style | py::array::forcecast> weights_a,
                   double scale, size_t n_T, size_t D, size_t n_S, size_t M, size_t event_width,
                   bool starts_with_first, bool ends_with_last,bool ends_in_last, int hmp_code,
                   double rho) {
    /*
    Compute "predictive-residuals" defined by Buckby et al. (2020).

    References:
      - Jodie Buckby, Ting Wang, Jiancang Zhuang & Kazushige Obara (2020) Model Checking for
        Hidden Markov Models, Journal of Computational and Graphical Statistics
    */
   
    auto Lam_a = py::array_t<double>(n_T);
    auto resid_a = py::array_t<double>({n_T,M});
    auto resid_sd_a = py::array_t<double>(M);
    auto old_res_a = py::array_t<double>(M);
    auto e_a = py::array_t<double>({n_T,M});
    auto lam_a = py::array_t<double>({n_T,D-1,n_S});
    auto elam_a = py::array_t<double>({n_T,M,D-1,n_S});
    Lam_a[py::make_tuple(py::ellipsis())] = 0.0;
    lam_a[py::make_tuple(py::ellipsis())] = 0.0;
    e_a[py::make_tuple(py::ellipsis())] = 0.0;
    resid_a[py::make_tuple(py::ellipsis())] = 0.0;
    old_res_a[py::make_tuple(py::ellipsis())] = 0.0;
    resid_sd_a[py::make_tuple(py::ellipsis())] = 0.0;
    elam_a[py::make_tuple(py::ellipsis())] = 0.0;

    auto Lam = Lam_a.mutable_unchecked();
    auto lam = lam_a.mutable_unchecked();
    // Predictive residual
    auto resid = resid_a.mutable_unchecked();
    auto resid1 = old_res_a.mutable_unchecked();
    // sd
    auto resid_sd = resid_sd_a.mutable_unchecked();
    // E(Y = y_t|y_1,...y_{t-1})
    auto e = e_a.mutable_unchecked();
    // E(Y = y_t, S=j,D=d|y_1,...y_{t-1}) 
    auto elam = elam_a.mutable_unchecked();
    auto y = y_a.unchecked(); 
    auto bs = bs_a.unchecked();
    auto mus = mus_a.unchecked();
    auto ds = ds_a.unchecked();
    auto T = T_a.unchecked();
    auto pi = pi_a.unchecked();
    auto weights = weights_a.unchecked();

    // Set up optional ar1 weights for residuals of HMP model
    double d0 = 0.0;
    double d1 = 0.0;
    double ebump = 0.0;
    if (rho < 999)
    {
        d0 = 1 / sqrt(1 - pow(rho,2));
        d1 = -rho / sqrt(1-pow(rho,2));
    }
    
    double llk = 0.0;
    double tp, b2;
    for (size_t t = 0; t < n_T; t++)
    {
        for (size_t d = 0; d < (D-1); d++)
        {
            for (size_t j = 0; j < n_S; j++)
            {
                if (t == 0)
                {
                    if (hmp_code != 0)
                    {
                        
                        lam(t,d,j) = pi(j)*bs(t,0,j)*ds(d,j);

                        for (size_t m = 0; m < M; m++)
                        {
                            // Expected value for signal m
                            if (j % 2 == 1)
                            {
                                
                                ebump = weights(0)*mus(t,m,j);
                                
                                if (rho < 999)
                                {
                                    ebump *= d0;
                                }
                                
                                elam(t,m,d,j) = pi(j)*ebump*ds(d,j);
                            }
                        }
                    }
                    else
                    {
                        lam(t,d,j) = pi(j)*bs(t,j)*ds(d,j);

                        for (size_t m = 0; m < M; m++)
                        {
                            // Expected value for signal m
                            elam(t,m,d,j) = pi(j)*mus(t,m,j)*ds(d,j);
                        }
                    }                    
                }
                else if ((t + d) < n_T || ends_with_last == false)
                {
                    // Probability of same state continuing
                    if ((d+1) <= (D-2))
                    {
                        
                        if (hmp_code != 0)
                        {
                            // Observation probabilities again - bump case depends on duration
                            // already spent in state.
                            if (j % 2 == 1)
                            {
                                if (d <= event_width-1)
                                {
                                    /* bump */
                                    lam(t,d,j) = lam(t-1,d+1,j)*bs(t,(event_width-1) - d,j);
                                    for (size_t m = 0; m < M; m++)
                                    {

                                        // Expected value for signal m
                                        ebump = weights((event_width-1) - d)*mus(t,m,j);
                                
                                        if (rho < 999)
                                        {
                                            ebump *= d0;

                                            if (((event_width-1) - d) > 0)
                                            {
                                                
                                                // Correct for expected value at previous time
                                                // point and for previous weight
                                                ebump += d1 * weights(((event_width-1) - d) - 1) *
                                                         mus(t-1,m,j);
                                            }
                                        }

                                        elam(t,m,d,j) = lam(t-1,d+1,j) * ebump;
                                    }
                                }
                                else
                                {
                                    lam(t,d,j) = 0;
                                }
                            }
                            else
                            {
                                // flat case
                                // Expected value is zero so only need lam
                                lam(t,d,j) = lam(t-1,d+1,j)*bs(t,0,j);
                            }   
                        }
                        else
                        {
                            lam(t,d,j) = lam(t-1,d+1,j)*bs(t,j);
                            for (size_t m = 0; m < M; m++)
                            {
                                // Expected value for signal m
                                elam(t,m,d,j) = lam(t-1,d+1,j) *
                                                mus(t,m,j);
                            }
                        }
                    }
                        
                    // Probability of other state ending on previous time point and
                    // transitioning to current state with dur d
                    tp = 0.0;
                    for (size_t i = 0; i < n_S; i++)
                    {
                        if (i == j)
                        {
                            continue;
                        }

                        tp += lam(t-1,0,i)*T(i,j);
                    }
                    
                    if (hmp_code != 0)
                    {

                        b2 = bs(t,0,j);
                        if(j % 2 == 0)
                        {
                            // If we have an ar1 model first obs prob for flats depends on lag of
                            // previous bump (only for states > 0, but this is handled outside)
                            b2 = bs(t,1,j);
                        }

                        lam(t,d,j) += tp * b2 * ds(d,(starts_with_first) ? j : j + n_S);

                        for (size_t m = 0; m < M; m++)
                        {   
                            // Expected value for signal m
                            if (j % 2 == 1)
                            {
                                
                                ebump = weights(0)*mus(t,m,j);
                                
                                if (rho < 999)
                                {
                                    ebump *= d0;
                                }
                                
                                elam(t,m,d,j) += tp * ebump *
                                                 ds(d,(starts_with_first) ? j : j + n_S);
                            }
                            else if (rho < 999)
                            {
                                // Lag of previous bump carries over
                                ebump = d1 * weights(event_width-1) * mus(t-1,m,j-1);

                                elam(t,m,d,j) += tp * ebump *
                                                 ds(d,(starts_with_first) ? j : j + n_S);

                            }

                            // Scaling to get conditional probability
                            elam(t,m,d,j) /= Lam(t-1);
                        }                        
                    }
                    else
                    {
                        lam(t,d,j) += tp * bs(t,j) * ds(d,(starts_with_first) ? j : j + n_S);

                        for (size_t m = 0; m < M; m++)
                        {   
                            // Expected value for signal m

                            elam(t,m,d,j) += tp * mus(t,j) *
                                            ds(d,(starts_with_first) ? j : j + n_S);


                            // Scaling to get conditional probability
                            elam(t,m,d,j) /= Lam(t-1);
                        }   
                    }
                    
                    // Scaling to get conditional probability
                    lam(t,d,j) /= Lam(t-1);
                    
                }
                    
                // Compute conditional probability of current obs given all previous obs
                if (t < (n_T - 1) || (ends_with_last == false))
                {   
                    // State does not have to end at last time point and also not in last state
                    // Before last time-point this is the right computation for all cases as well!
                    if (t < (n_T - 1) || ends_in_last == false)
                    {
                        Lam(t) += lam(t,d,j);
                        for (size_t m = 0; m < M; m++)
                        {
                            e(t,m) += elam(t,m,d,j);
                        }
                    }
                    // State does not have to end at last time point but in last state
                    else if (j == (n_S-1))
                    {
                        Lam(t) += lam(t,d,j);
                        for (size_t m = 0; m < M; m++)
                        {
                            e(t,m) += elam(t,m,d,j);
                        }
                    }
                }
                else if (d == 0)
                {   
                    // State does have to end at last time point but not in last state
                    if (ends_in_last == false)
                    {
                        Lam(t) += lam(t,d,j);
                        for (size_t m = 0; m < M; m++)
                        {
                            e(t,m) += elam(t,m,d,j);
                        }
                    }
                    // State does have to end at last time point and also in last state
                    else if (j == (n_S-1))
                    {
                        Lam(t) += lam(t,d,j);
                        for (size_t m = 0; m < M; m++)
                        {
                            e(t,m) += elam(t,m,d,j);
                        }
                    }
                    
                }
            }
        }
        llk += log(Lam(t));
    }

    /*
    Have expectations now, can compute predictive residuals
    */

    for (size_t m = 0; m < M; m++)
    {
        resid(0,m) = (y(0,m) - e(0,m));
    }

    for (size_t n = 1; n < n_T; n++)
    {
        for (size_t m = 0; m < M; m++)
        {
            resid(n,m) = resid(n-1,m) + (y(n,m) - e(n,m));
        }
    }

    /*
    And standardize
    */

    for (size_t m = 0; m < M; m++)
    {
        resid_sd(m) = pow(resid(0,m),2);
        resid1(m) = resid(0,m);
    }
    
    for (size_t n = 1; n < n_T; n++)
    {
        for (size_t m = 0; m < M; m++)
        {
            resid_sd(m) += pow(resid(n,m)-resid1(m),2);
            resid1(m) = resid(n,m);
            resid(n,m) /= sqrt(resid_sd(m));
        }
    }

    return resid_a;
}


void init_resid(py::module_ &m)
{
    m.def("forward_resid", &forward_resid, "Compute forward pseudo-residuals for hsmm.");
    m.def("predictive_resid", &predictive_resid, "Compute predictive residuals for a hsmm.");
}