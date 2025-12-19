#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <limits>



namespace py = pybind11;

double llk(py::array_t<double, py::array::f_style | py::array::forcecast> bs_a,
              py::array_t<double, py::array::f_style | py::array::forcecast> ds_a,
              py::array_t<double, py::array::f_style | py::array::forcecast> T_a,
              py::array_t<double, py::array::f_style | py::array::forcecast> pi_a,
              py::array_t<double, py::array::f_style | py::array::forcecast> weights_a,
              double scale, size_t n_T, size_t D, size_t n_S, size_t event_width,
              bool starts_with_first,bool ends_with_last,bool ends_in_last,int hmp_code) {
   
    
    auto Lam_a = py::array_t<double>(n_T);
    auto lam_a = py::array_t<double>({n_T,D-1,n_S});
    Lam_a[py::make_tuple(py::ellipsis())] = 0.0;
    lam_a[py::make_tuple(py::ellipsis())] = 0.0;

    auto Lam = Lam_a.mutable_unchecked();
    auto lam = lam_a.mutable_unchecked();
    auto bs = bs_a.unchecked();
    auto ds = ds_a.unchecked();
    auto T = T_a.unchecked();
    auto pi = pi_a.unchecked();
    auto weights = weights_a.unchecked();

    
    double llk = 0.0;
    double tp;
    double b2;
    for (size_t t = 0; t < n_T; t++)
    {
        for (size_t d = 0; d < (D-1); d++)
        {
            for (size_t j = 0; j < n_S; j++)
            {
                // Skip durations exceeding event_width for bump states
                if (hmp_code != 0 && j % 2 == 1 && d >= event_width)
                {
                    continue;
                }   
                    
                if (t == 0)
                {
                    if (hmp_code != 0)
                    {
                        lam(t,d,j) = pi(j)*bs(t,0,j)*ds(d,j);
                    }
                    else
                    {
                        lam(t,d,j) = pi(j)*bs(t,j)*ds(d,j);
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
                                    lam(t,d,j) = lam(t-1,d+1,j)*bs(t,(event_width-1) - d,j);
                                }
                                else
                                {
                                    lam(t,d,j) = 0;
                                }
                            }
                            else
                            {
                                lam(t,d,j) = lam(t-1,d+1,j)*bs(t,0,j);
                            }
                        }
                        else
                        {
                            lam(t,d,j) = lam(t-1,d+1,j)*bs(t,j);
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
                            // previous bump (only for states > 0, but this is taken care of outside)
                            b2 = bs(t,1,j);
                        }
                        tp *= b2*ds(d,(starts_with_first) ? j : j + n_S);
                        
                    }
                    else
                    {
                        tp *= bs(t,j)*ds(d,(starts_with_first) ? j : j + n_S);
                    }                    

                    lam(t,d,j) += tp;

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
                    }
                    // State does not have to end at last time point but in last state
                    else if (j == (n_S-1))
                    {
                        Lam(t) += lam(t,d,j);
                    }
                }
                else if (d == 0)
                {   
                    // State does have to end at last time point but not in last state
                    if (ends_in_last == false)
                    {
                        Lam(t) += lam(t,d,j);
                    }
                    // State does have to end at last time point and also in last state
                    else if (j == (n_S-1))
                    {
                        Lam(t) += lam(t,d,j);
                    }
                    
                }
            }
        }

        llk += log(Lam(t));
    }

    return llk;
}

py::array_t<double> llkFTPgrad(py::array_t<double, py::array::f_style | py::array::forcecast> bs_a,
                               py::array_t<double, py::array::f_style | py::array::forcecast> y_a,
                               py::array_t<double, py::array::f_style | py::array::forcecast> mu_a,
                               py::array_t<double, py::array::f_style | py::array::forcecast> ds_a,
                               py::array_t<double, py::array::f_style | py::array::forcecast> T_a,
                               py::array_t<double, py::array::f_style | py::array::forcecast> pi_a,
                               py::array_t<double, py::array::f_style | py::array::forcecast> b_grad_a,
                               py::array_t<double, py::array::f_style | py::array::forcecast> d_grad_a,
                               //py::array_t<double, py::array::f_style | py::array::forcecast> T_grad_a,
                               //py::array_t<double, py::array::f_style | py::array::forcecast> pi_grad_a,
                               py::array_t<size_t, py::array::c_style | py::array::forcecast> m_idx_grad_a,
                               py::array_t<size_t, py::array::c_style | py::array::forcecast> j_idx_grad_a,
                               py::array_t<double, py::array::c_style | py::array::forcecast> weights_a,
                               double scale, size_t n_T, size_t D, size_t n_S, size_t n_coef, size_t M,
                               size_t event_width, bool starts_with_first, bool ends_with_last, 
                               bool ends_in_last, int hmp_code, double rho) {
   
    
    auto Lam_a = py::array_t<double>(n_T);
    auto lam_a = py::array_t<double>({n_T,D-1,n_S});
    size_t n_switch = 2;
    auto psi_a = py::array_t<double>({n_switch,D-1,n_S,n_coef});
    Lam_a[py::make_tuple(py::ellipsis())] = 0.0;
    lam_a[py::make_tuple(py::ellipsis())] = 0.0;
    psi_a[py::make_tuple(py::ellipsis())] = 0.0;

    auto Lam = Lam_a.mutable_unchecked();
    auto lam = lam_a.mutable_unchecked();
    auto psi = psi_a.mutable_unchecked();
    auto mu = mu_a.unchecked();
    auto y = y_a.unchecked();
    auto bs = bs_a.unchecked();
    auto ds = ds_a.unchecked();
    auto T = T_a.unchecked();
    auto pi = pi_a.unchecked();
    auto weights = weights_a.unchecked();

    auto b_grad = b_grad_a.unchecked();
    auto d_grad = d_grad_a.unchecked();
    //auto T_grad = T_grad_a.unchecked();
    //auto pi_grad = pi_grad_a.unchecked();

    auto j_idx_grad = j_idx_grad_a.unchecked();
    auto m_idx_grad = m_idx_grad_a.unchecked();
    
    double llk = 0.0;
    double tp,tpg,b_grad_c,b1,b2;
    size_t tci,t1,t2;

    auto grad_a = py::array_t<double>(n_coef);
    grad_a[py::make_tuple(py::ellipsis())] = 0.0;
    auto grad = grad_a.mutable_unchecked();
    
    // Set up optional ar1 weights for residuals of HMP model
    double d0 = 0.0;
    double d1 = 0.0;
    double ebump = 0.0;
    double dbump = 0.0;
    if (rho < 999)
    {
        d0 = 1 / sqrt(1 - pow(rho,2));
        d1 = -rho / sqrt(1-pow(rho,2));
    }

    for (size_t t = 0; t < n_T; t++)
    {
        t1 = t % 2; // Index for current t
        t2 = 1 - t1; // Index for t - 1

        for (size_t d = 0; d < (D-1); d++)
        {
            for (size_t j = 0; j < n_S; j++)
            {
                
                // Skip durations exceeding event_width for bump states
                if (hmp_code != 0 && j % 2 == 1 && d >= event_width)
                {
                    continue;
                }
                
                if (t == 0)
                {

                    
                    if (hmp_code != 0)
                    {
                        lam(t,d,j) = pi(j)*bs(t,0,j)*ds(d,j);
                    }
                    else
                    {
                        lam(t,d,j) = pi(j)*bs(t,j)*ds(d,j);
                    }

                    // Now compute deriv of pi(j)*bs(t,j)*ds(d,j) with respect to individual coef.
                    // **NOTE**: Generally deriv will be single product, not sum over three products because the remaining
                    // sums will cancel, since each coef will only be associated with either pi, bs, or ds!
                    tci = 0;
                    for (size_t ci = 0; ci < b_grad_a.shape(1); ci++)
                    {

                        if (j_idx_grad(tci) == j)
                        {
                            // Compute deriv of bs(t,j) with respect to individual coef. Remember, bs(t,j) is a product over M signals:
                            // bs(t,j) = bs(t,j,1) * bs(t,j,2) * ... * bs(t,j,M)
                            // So deriv is sum over M products over M signals with one bs(t,j,...) replaced by deriv of bs(t,j,...) with
                            // respect to coef. Because we assume no coef shared between terms all but 1 of those products cancel due to zero derivative.
                            // Hence we only need to evaluate the product and then multiply with the correct gradient - automatically transforming it out of log-scale.
                            
                            if (hmp_code != 0)
                            {
                                ebump = weights(0)*mu(t,m_idx_grad(ci),j);
                                dbump = weights(0)*b_grad(t,ci);
                                
                                if (rho < 999)
                                {
                                    ebump *= d0;
                                    dbump *= d0;
                                }

                                b_grad_c = bs(t,0,j) * ((((hmp_code == 1) ? 2 : 1) *
                                                         (y(t,m_idx_grad(ci)) - ebump)
                                                        ) / scale) * dbump;
                            }
                            else
                            {
                                b_grad_c = bs(t,j) * b_grad(t,ci);
                            }
                            
                            // Remaining sums cancels, since pi in this implementation does not depend on beta and because ds(d,j) does not depend on coef(ci)
                            psi(t1,d,j,tci) = pi(j)*b_grad_c*ds(d,j);
                        } else
                        {
                            psi(t1,d,j,tci) = 0;
                        }
                        tci += 1;
                    }
                    for (size_t ci = 0; ci < d_grad_a.shape(1); ci++)
                    {
                        // Now onto deriv of ds(d,j) with respect to individual coef. Similar to above but simpler - gradient has already been transformed out
                        // of log-scale
                        if (j_idx_grad(tci) == j)
                        {
                            if (hmp_code != 0)
                            {
                                psi(t1,d,j,tci) = pi(j)*bs(t,0,j)*d_grad(d,ci);
                            }
                            else
                            {
                                psi(t1,d,j,tci) = pi(j)*bs(t,j)*d_grad(d,ci);
                            }
                            
                        } else
                        {
                            psi(t1,d,j,tci) = 0;
                        }
                        tci += 1;
                    }
                }
                else if ((t + d) < n_T || ends_with_last == false)
                {
                    // Probability of same state continuing
                    b1 = 0.0;
                    if ((d+1) <= (D-2))
                    {
                        if (hmp_code != 0)
                        {
                            if (j % 2 == 1)
                            {
                                if (d <= event_width-1)
                                {
                                    /* bump */
                                    b1 = bs(t,(event_width-1) - d,j);
                                }
                            }
                            else
                            {
                                /* flat */
                                b1 = bs(t,0,j);
                            }
                            lam(t,d,j) = lam(t-1,d+1,j)*b1;
                        }
                        else
                        {
                            lam(t,d,j) = lam(t-1,d+1,j)*bs(t,j);
                        }
                        
                    }

                    // Probability of other state ending on previous time point and transitioning to current state with dur d
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
                            // previous bump (only for states > 0, but this is taken care of outside)
                            b2 = bs(t,1,j);
                        }
                        tp *= b2*ds(d,(starts_with_first) ? j : j + n_S);
                    }
                    else
                    {
                        tp *= bs(t,j)*ds(d,(starts_with_first) ? j : j + n_S);
                    }
                    

                    lam(t,d,j) += tp;

                    // Now compute deriv of lam(t-1,d+1,j)*bs(t,j) + (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * bs(t,j)*ds(d,j) with respect to individual coef.
                    tci = 0;
                    for (size_t ci = 0; ci < b_grad_a.shape(1); ci++)
                    {
                        
                        // Coef ci is in model of state j
                        if (j_idx_grad(tci) == j)
                        {
                            // Start with computing deriv of lam(t-1,d+1,j)*bs(t,j) with respect to individual coef. Remember, bs(t,j) is again a product over M signals!
                            // All but one deriv of bs(t,j) again cancel so we again only need to compute the remaining product and then multiply that
                            // by lam(t-1,d+1,j). Then we have to add psi(t-1,d+1,j,tci) * bs(t,j) to that to account for the fact that lam(t-1,d+1,j) depends
                            // on coef(ci) as well

                            if ((d+1) <= (D-2))
                            {
                                // And now multiply with previous lam and the correct gradient - first has to be transformed out of log scale - then add
                                // to rest of derivative.
                                if (hmp_code != 0)
                                {
                                    if (d <= event_width-1)
                                    {
                                        
                                        ebump = weights((event_width-1) - d) *
                                                mu(t,m_idx_grad(ci),j);

                                        dbump = weights((event_width-1) - d) * b_grad(t,ci);
                                        
                                        if (rho < 999)
                                        {
                                            
                                            ebump *= d0;
                                            dbump *= d0;
                                            
                                            if (((event_width-1) - d) > 0)
                                            {
                                                
                                                // Correct for expected value at previous time
                                                // point and for previous weight
                                                ebump += d1 * weights(((event_width-1) - d) - 1) *
                                                         mu(t-1,m_idx_grad(ci),j);
                                                
                                                
                                                dbump += d1 * weights(((event_width-1) - d) - 1) *
                                                         b_grad(t - 1,ci);
                                            }
                                        }
                                        
                                        b_grad_c = b1 * ((((hmp_code == 1) ? 2 : 1) *
                                                          (y(t,m_idx_grad(ci)) - ebump)
                                                         ) / scale) * dbump;
                                    } else {
                                        b_grad_c = 0.0;
                                    }
                                    psi(t1,d,j,tci) = (psi(t2,d+1,j,tci) * b1) +
                                                    (b_grad_c * lam(t-1,d+1,j));
                                }
                                else
                                {
                                    psi(t1,d,j,tci) = (psi(t2,d+1,j,tci) * bs(t,j)) +
                                                      (b_grad(t,ci) * bs(t,j) * lam(t-1,d+1,j));
                                }
                                
                            } else
                            {
                                psi(t1,d,j,tci) = 0;
                            }

                            // Compute deriv of (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * bs(t,j)*ds(d,j) with respect to individual coef:

                            // Start with computing (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) and it's derivative. Technically the latter is
                            // a sum of two products, but T(i,j) does not depend on coef in this implementation - so this is only sum over
                            // previous psi variables multiplied by T(i,j).
                            tp = 0.0;
                            tpg = 0.0;

                            for (size_t i = 0; i < n_S; i++)
                            {
                                if (i == j)
                                {
                                    continue;
                                }
                                
                                tp += lam(t-1,0,i)*T(i,j);
                                tpg += psi(t2,0,i,tci)*T(i,j);
                            }

                            // Now derivative. Technically sum of three products - but final one cancels because ds(d,j) does not depend on coef.
                            // First is deriv of (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * bs(t,j)*ds(d,j):
                            if (hmp_code != 0)
                            {
                                psi(t1,d,j,tci) += tpg*bs(t,0,j)*ds(d,(starts_with_first) ? j : j + n_S);


                                ebump = weights(0)*mu(t,m_idx_grad(ci),j);
                                dbump = weights(0)*b_grad(t,ci);
                                
                                if (rho < 999)
                                {
                                    ebump *= d0;
                                    dbump *= d0;
                                }

                                b_grad_c = bs(t,0,j) * ((((hmp_code == 1) ? 2 : 1) *
                                                         (y(t,m_idx_grad(ci)) - ebump)
                                                        ) / scale) * dbump;

                                // Next is (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * deriv of bs(t,j) *
                                // ds(d,j):
                                psi(t1,d,j,tci) += tp*b_grad_c*ds(d,(starts_with_first) ? j : j + n_S);
                            }
                            else
                            {
                                psi(t1,d,j,tci) += tpg*bs(t,j)*ds(d,(starts_with_first) ? j : j + n_S);

                                // Next is (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * deriv of bs(t,j) * ds(d,j):
                                psi(t1,d,j,tci) += tp*b_grad(t,ci) * bs(t,j)*ds(d,(starts_with_first) ? j : j + n_S);
                            }
                        
                        }
                        // Coef ci is **NOT** in model of state j
                        else
                        {
                            // Again, start with computing deriv of lam(t-1,d+1,j)*bs(t,j).
                            // Only need deriv of lam(t-1,d+1,j) with respect to individual coef, since bs(t,j) does not depend on coef not in model of state j
                            if ((d+1) <= (D-2))
                            {
                                if (hmp_code != 0)
                                {
                                    psi(t1,d,j,tci) = psi(t2,d+1,j,tci) * b1;
                                }
                                else
                                {
                                    psi(t1,d,j,tci) = psi(t2,d+1,j,tci) * bs(t,j);
                                }
                                
                            } else
                            {
                                psi(t1,d,j,tci) = 0;
                            }
                            
                                
                            // Compute deriv of (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * bs(t,j)*ds(d,j) with respect to individual coef:
                            
                            tp = 0.0;
                            tpg = 0.0;
                            for (size_t i = 0; i < n_S; i++)
                            {
                                if (i == j)
                                {
                                    continue;
                                }
                                
                                tp += lam(t-1,0,i)*T(i,j);
                                tpg += psi(t2,0,i,tci)*T(i,j);
                            }

                            // Only need derivative of (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)),
                            // since neither bs(t,j) nor ds(d,j)) depend on coef not in model of state j
                            if (hmp_code != 0)
                            {
                                
                                b2 = bs(t,0,j);
                                if (j % 2 == 0)
                                {
                                    
                                    b2 = bs(t,1,j);
                                    
                                    // Unless we have rho, then bs(t,j) depends on coef in obs model of
                                    // previous state if for flat states > 0!
                                    if (rho < 999 & (j_idx_grad(tci) == (j-1)))
                                    {
                                        ebump = d1 * weights(event_width-1) * mu(t-1,m_idx_grad(ci),j-1);
                                        dbump = d1 * weights(event_width-1) * b_grad(t-1,ci);

                                        b_grad_c = b2 * ((((hmp_code == 1) ? 2 : 1) *
                                                            (y(t,m_idx_grad(ci)) - ebump)
                                                            ) / scale) * dbump;

                                        psi(t1,d,j,tci) += tp * b_grad_c * ds(d,(starts_with_first) ? j : j + n_S);
                                    }
                                }
                                psi(t1,d,j,tci) += tpg*b2*ds(d,(starts_with_first) ? j : j + n_S);

                            }
                            else
                            {
                                psi(t1,d,j,tci) += tpg*bs(t,j)*ds(d,(starts_with_first) ? j : j + n_S);
                            }
                        }

                        tci += 1;
                    }

                    for (size_t ci = 0; ci < d_grad_a.shape(1); ci++)
                    {
                        // Now onto deriv of ds(d,j) with respect to individual coef. Again similar to above but again simpler

                        // Coef ci is in model of state j
                        if (j_idx_grad(tci) == ((starts_with_first) ? j : j + n_S))
                        {
                            
                            // Compute deriv of lam(t-1,d+1,j)*bs(t,j) with respect to individual coef. Since bs(t,j) does not depend on this
                            // coef, all that remains is:
                            if ((d+1) <= (D-2))
                            {
                                if (hmp_code != 0)
                                {
                                    psi(t1,d,j,tci) = psi(t2,d+1,j,tci) * b1;
                                }
                                else
                                {
                                    psi(t1,d,j,tci) = psi(t2,d+1,j,tci) * bs(t,j);
                                }
                                
                            } else
                            {
                                psi(t1,d,j,tci) = 0;
                            }
                            
                            // Compute deriv of (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * bs(t,j)*ds(d,j) with respect to individual coef:

                            // Again start with computing (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) and it's derivative. Again simplifies to sum of two products
                            // now because bs(t,j) does not depend on coef.
                            tp = 0.0;
                            tpg = 0.0;

                            for (size_t i = 0; i < n_S; i++)
                            {
                                if (i == j)
                                {
                                    continue;
                                }
                                
                                tp += lam(t-1,0,i)*T(i,j);
                                tpg += psi(t2,0,i,tci)*T(i,j);
                            }
                            
                            if (hmp_code != 0)
                            {
                                b2 = bs(t,0,j);
                                if(j % 2 == 0)
                                {
                                    // If we have an ar1 model first obs prob for flats depends on lag of
                                    // previous bump (only for states > 0, but this is taken care of outside)
                                    b2 = bs(t,1,j);
                                }
                                
                                // First is deriv of (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * bs(t,j)*ds(d,j) - same as before:
                                psi(t1,d,j,tci) += tpg*b2*ds(d,(starts_with_first) ? j : j + n_S);

                                // Second is the third product missing before: (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * bs(t,j) * deriv of ds(d,j)
                                psi(t1,d,j,tci) += tp*b2*d_grad(d,ci);
                            }
                            else
                            {
                                // First is deriv of (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * bs(t,j)*ds(d,j) - same as before:
                                psi(t1,d,j,tci) += tpg*bs(t,j)*ds(d,(starts_with_first) ? j : j + n_S);

                                // Second is the third product missing before: (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * bs(t,j) * deriv of ds(d,j)
                                psi(t1,d,j,tci) += tp*bs(t,j)*d_grad(d,ci);
                            }
                        }
                        // Coef ci is **NOT** in model of state j
                        else
                        {
                            // Again, start with computing deriv of lam(t-1,d+1,j)*bs(t,j).
                            // Since bs(t,j) still does not depend on this coef, all that remains is:
                            if ((d+1) <= (D-2))
                            {
                                if (hmp_code != 0)
                                {
                                    psi(t1,d,j,tci) = psi(t2,d+1,j,tci) * b1;
                                }
                                else
                                {
                                    psi(t1,d,j,tci) = psi(t2,d+1,j,tci) * bs(t,j);
                                }
                                
                            } else
                            {
                                psi(t1,d,j,tci) = 0;
                            }
                            // Compute deriv of (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * bs(t,j)*ds(d,j) with respect to individual coef:

                            // Again start with computing derivative of (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j))
                            tpg = 0.0;
                            for (size_t i = 0; i < n_S; i++)
                            {
                                if (i == j)
                                {
                                    continue;
                                }

                                tpg += psi(t2,0,i,tci)*T(i,j);
                            }
                            
                            // Only need deriv of (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * bs(t,j)*ds(d,j) - same as before:
                            if (hmp_code != 0)
                            {
                                
                                b2 = bs(t,0,j);
                                if(j % 2 == 0)
                                {
                                    // If we have an ar1 model first obs prob for flats depends on lag of
                                    // previous bump (only for states > 0, but this is taken care of outside)
                                    b2 = bs(t,1,j);
                                }
                                psi(t1,d,j,tci) += tpg*b2*ds(d,(starts_with_first) ? j : j + n_S);
                                
                            }
                            else
                            {
                                psi(t1,d,j,tci) += tpg*bs(t,j)*ds(d,(starts_with_first) ? j : j + n_S);
                            }
                            
                        }

                        tci += 1;
                    }

                    // Scaling to get conditional probability
                    lam(t,d,j) /= Lam(t-1);
                    for (size_t tci = 0; tci < n_coef; tci++)
                    {
                        psi(t1,d,j,tci) /= Lam(t-1);
                    }
                    
                }

                // Compute conditional probability of current obs given all previous obs
                if (t < (n_T - 1) || (ends_with_last == false))
                {   
                    // State does not have to end at last time point and also not in last state
                    // Before last time-point this is the right computation for all cases as well!
                    if (t < (n_T - 1) || ends_in_last == false)
                    {
                        Lam(t) += lam(t,d,j);
                    }
                    // State does not have to end at last time point but in last state
                    else if (j == (n_S-1))
                    {
                        Lam(t) += lam(t,d,j);
                    }
                }
                else if (d == 0)
                {   
                    // State does have to end at last time point but not in last state
                    if (ends_in_last == false)
                    {
                        Lam(t) += lam(t,d,j);
                    }
                    // State does have to end at last time point and also in last state
                    else if (j == (n_S-1))
                    {
                        Lam(t) += lam(t,d,j);
                    }
                    
                }
            }
        }
        llk += log(Lam(t));
    }

    if (ends_with_last)
    {   
        for (size_t tci = 0; tci < n_coef; tci++)
        {   
            if (ends_in_last == false)
            {
                for (size_t j = 0; j < n_S; j++)
                {
                    grad(tci) += psi(t1,0,j,tci);
                }
            }
            else
            {
                grad(tci) += psi(t1,0,n_S-1,tci);
            }
            
            grad(tci) /= Lam(n_T-1);
        }
    }
    else
    {   
        for (size_t tci = 0; tci < n_coef; tci++)
        {
            for (size_t d = 0; d < (D-1); d++)
            {   
                if (ends_in_last == false)
                {
                    for (size_t j = 0; j < n_S; j++)
                    {
                        grad(tci) += psi(t1,d,j,tci);
                    }
                }
                else
                {
                    grad(tci) += psi(t1,d,n_S-1,tci);
                }
            }
            grad(tci) /= Lam(n_T-1);
        }
    }

    return grad_a;
}

py::array_t<double> llkgrad(py::array_t<double, py::array::f_style | py::array::forcecast> bs_a,
                            py::array_t<double, py::array::f_style | py::array::forcecast> ds_a,
                            py::array_t<double, py::array::f_style | py::array::forcecast> T_a,
                            py::array_t<double, py::array::f_style | py::array::forcecast> pi_a,
                            py::array_t<double, py::array::f_style | py::array::forcecast> b_grad_a,
                            py::array_t<double, py::array::f_style | py::array::forcecast> d_grad_a,
                            py::array_t<double, py::array::f_style | py::array::forcecast> T_grad_a,
                            py::array_t<double, py::array::f_style | py::array::forcecast> pi_grad_a,
                            py::array_t<size_t, py::array::c_style | py::array::forcecast> j_idx_grad_a,
                            size_t M, size_t n_T, size_t D, size_t n_S, size_t n_coef,
                            bool starts_with_first, bool ends_with_last, bool ends_in_last,bool model_T) {
   
    
    auto Lam_a = py::array_t<double>(n_T);
    auto lam_a = py::array_t<double>({n_T,D-1,n_S});
    size_t n_switch = 2;
    auto psi_a = py::array_t<double>({n_switch,D-1,n_S,n_coef});
    Lam_a[py::make_tuple(py::ellipsis())] = 0.0;
    lam_a[py::make_tuple(py::ellipsis())] = 0.0;
    psi_a[py::make_tuple(py::ellipsis())] = 0.0;

    auto Lam = Lam_a.mutable_unchecked();
    auto lam = lam_a.mutable_unchecked();
    auto psi = psi_a.mutable_unchecked();
    auto bs = bs_a.unchecked();
    auto ds = ds_a.unchecked();
    auto T = T_a.unchecked();
    auto pi = pi_a.unchecked();

    auto b_grad = b_grad_a.unchecked();
    auto d_grad = d_grad_a.unchecked();
    auto T_grad = T_grad_a.unchecked();
    auto pi_grad = pi_grad_a.unchecked();

    auto j_idx_grad = j_idx_grad_a.unchecked();
    
    double llk = 0.0;
    double tp,tpg,b_grad_c;
    size_t tci,tidx,t1,t2;

    auto grad_a = py::array_t<double>(n_coef);
    grad_a[py::make_tuple(py::ellipsis())] = 0.0;
    auto grad = grad_a.mutable_unchecked();

    for (size_t t = 0; t < n_T; t++)
    {   
        t1 = t % 2; // Index for current t
        t2 = 1 - t1; // Index for t - 1

        for (size_t d = 0; d < (D-1); d++)
        {
            for (size_t j = 0; j < n_S; j++)
            {
                if (t == 0)
                {

                    lam(t,d,j) = pi(j)*bs(t,j)*ds(d,j);

                    // Now compute deriv of pi(j)*bs(t,j)*ds(d,j) with respect to individual coef.
                    // **NOTE**: Generally deriv will be single product, not sum over three products because the remaining
                    // sums will cancel, since each coef will only be associated with either pi, bs, or ds!
                    tci = 0;
                    for (size_t ci = 0; ci < b_grad_a.shape(1); ci++)
                    {

                        if (j_idx_grad(tci) == j)
                        {
                            // Compute deriv of bs(t,j) with respect to individual coef. Remember, bs(t,j) is a product over M signals:
                            // bs(t,j) = bs(t,j,1) * bs(t,j,2) * ... * bs(t,j,M)
                            // So deriv is sum over M products over M signals with one bs(t,j,...) replaced by deriv of bs(t,j,...) with
                            // respect to coef. Because we assume no coef shared between terms all but 1 of those products cancel due to zero derivative.
                            // Hence we only need to evaluate the product and then multiply with the correct gradient - automatically transforming it out of log-scale.
                            b_grad_c = bs(t,j) * b_grad(t,ci);
                            
                            // Remaining sums cancels, since pi in this implementation does not depend on beta and because ds(d,j) does not depend on coef(ci)
                            psi(t1,d,j,tci) = pi(j)*b_grad_c*ds(d,j);
                        } else
                        {
                            psi(t1,d,j,tci) = 0;
                        }
                        tci += 1;
                    }
                    for (size_t ci = 0; ci < d_grad_a.shape(1); ci++)
                    {
                        // Now onto deriv of ds(d,j) with respect to individual coef. Similar to above but simpler - gradient has already been transformed out
                        // of log-scale
                        if (j_idx_grad(tci) == j)
                        {
                            psi(t1,d,j,tci) = pi(j)*bs(t,j)*d_grad(d,ci);
                        } else
                        {
                            psi(t1,d,j,tci) = 0;
                        }
                        tci += 1;
                    }

                    if (model_T)
                    {
                        for (size_t ci = 0; ci < T_grad_a.shape(1); ci++)
                        {
                            // Derivative of a_ij with respect to individual coef. Nothing to do here but update tci
                            tci += 1;
                        }
                    }

                    for (size_t ci = 0; ci < pi_grad_a.shape(1); ci++)
                    {
                        // Derivative of pi[j] with respect to individual coef. Similar to duration case.
                        psi(t1,d,j,tci) = (pi(j) * pi_grad(j,ci))*bs(t,j)*ds(d,j);
                        
                        tci += 1;
                    }
                }
                else if ((t + d) < n_T || ends_with_last == false)
                {
                    // Probability of same state continuing
                    if ((d+1) <= (D-2))
                    {
                        lam(t,d,j) = lam(t-1,d+1,j)*bs(t,j);
                    }

                    // Probability of other state ending on previous time point and transitioning to current state with dur d
                    tp = 0.0;
                    for (size_t i = 0; i < n_S; i++)
                    {
                        if (i == j)
                        {
                            continue;
                        }
                        tp += lam(t-1,0,i)*T(i,j);
                        
                    }
                    
                    tp *= bs(t,j)*ds(d,(starts_with_first) ? j : j + n_S);

                    lam(t,d,j) += tp;

                    // Now compute deriv of lam(t-1,d+1,j)*bs(t,j) + (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * bs(t,j)*ds(d,j) with respect to individual coef.
                    tci = 0;
                    for (size_t ci = 0; ci < b_grad_a.shape(1); ci++)
                    {
                        
                        // Coef ci is in model of state j
                        if (j_idx_grad(tci) == j)
                        {
                            // Start with computing deriv of lam(t-1,d+1,j)*bs(t,j) with respect to individual coef. Remember, bs(t,j) is again a product over M signals!
                            // All but one deriv of bs(t,j) again cancel so we again only need to compute the remaining product and then multiply that
                            // by lam(t-1,d+1,j). Then we have to add psi(t-1,d+1,j,tci) * bs(t,j) to that to account for the fact that lam(t-1,d+1,j) depends
                            // on coef(ci) as well

                            if ((d+1) <= (D-2))
                            {
                                // And now multiply with previous lam and the correct gradient - first has to be transformed out of log scale - then add
                                // to rest of derivative.
                                psi(t1,d,j,tci) = (psi(t2,d+1,j,tci) * bs(t,j)) + (b_grad(t,ci) * bs(t,j) * lam(t-1,d+1,j));
                            } else
                            {
                                psi(t1,d,j,tci) = 0;
                            }

                            // Compute deriv of (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * bs(t,j)*ds(d,j) with respect to individual coef:

                            // Start with computing (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) and it's derivative. Technically the latter is
                            // a sum of two products, but T(i,j) does not depend on coef in this implementation - so this is only sum over
                            // previous psi variables multiplied by T(i,j).
                            tp = 0.0;
                            tpg = 0.0;

                            for (size_t i = 0; i < n_S; i++)
                            {
                                if (i == j)
                                {
                                    continue;
                                }
                                
                                tp += lam(t-1,0,i)*T(i,j);
                                tpg += psi(t2,0,i,tci)*T(i,j);
                            }

                            // Now derivative. Technically sum of three products - but final one cancels because ds(d,j) does not depend on coef.
                            // First is deriv of (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * bs(t,j)*ds(d,j):
                            psi(t1,d,j,tci) += tpg*bs(t,j)*ds(d,(starts_with_first) ? j : j + n_S);

                            // Next is (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * deriv of bs(t,j) * ds(d,j):
                            psi(t1,d,j,tci) += tp*b_grad(t,ci) * bs(t,j)*ds(d,(starts_with_first) ? j : j + n_S);
                        
                        }
                        // Coef ci is **NOT** in model of state j
                        else
                        {
                            // Again, start with computing deriv of lam(t-1,d+1,j)*bs(t,j).
                            // Only need deriv of lam(t-1,d+1,j) with respect to individual coef, since bs(t,j) does not depend on coef not in model of state j
                            if ((d+1) <= (D-2))
                            {
                                psi(t1,d,j,tci) = psi(t2,d+1,j,tci) * bs(t,j);
                            } else
                            {
                                psi(t1,d,j,tci) = 0;
                            }
                                
                            // Compute deriv of (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * bs(t,j)*ds(d,j) with respect to individual coef:

                            tpg = 0.0;
                            for (size_t i = 0; i < n_S; i++)
                            {
                                if (i == j)
                                {
                                    continue;
                                }

                                tpg += psi(t2,0,i,tci)*T(i,j);
                            }

                            // Only need derivative of (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)), since neither bs(t,j) nor ds(d,j)) depend on coef not in model of state j
                            psi(t1,d,j,tci) += tpg*bs(t,j)*ds(d,(starts_with_first) ? j : j + n_S);
                        }

                        tci += 1;
                    }

                    for (size_t ci = 0; ci < d_grad_a.shape(1); ci++)
                    {
                        // Now onto deriv of ds(d,j) with respect to individual coef. Again similar to above but again simpler

                        // Coef ci is in model of state j
                        if (j_idx_grad(tci) == ((starts_with_first) ? j : j + n_S))
                        {
                            
                            // Compute deriv of lam(t-1,d+1,j)*bs(t,j) with respect to individual coef. Since bs(t,j) does not depend on this
                            // coef, all that remains is:
                            if ((d+1) <= (D-2))
                            {
                                psi(t1,d,j,tci) = psi(t2,d+1,j,tci) * bs(t,j);
                            } else
                            {
                                psi(t1,d,j,tci) = 0;
                            }
                            // Compute deriv of (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * bs(t,j)*ds(d,j) with respect to individual coef:

                            // Again start with computing (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) and it's derivative. Again simplifies to sum of two products
                            // now because bs(t,j) does not depend on coef.
                            tp = 0.0;
                            tpg = 0.0;

                            for (size_t i = 0; i < n_S; i++)
                            {
                                if (i == j)
                                {
                                    continue;
                                }
                                
                                tp += lam(t-1,0,i)*T(i,j);
                                tpg += psi(t2,0,i,tci)*T(i,j);
                            }
                            
                            // First is deriv of (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * bs(t,j)*ds(d,j) - same as before:
                            psi(t1,d,j,tci) += tpg*bs(t,j)*ds(d,(starts_with_first) ? j : j + n_S);

                            // Second is the third product missing before: (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * bs(t,j) * deriv of ds(d,j)
                            psi(t1,d,j,tci) += tp*bs(t,j)*d_grad(d,ci);
                        
                        }
                        // Coef ci is **NOT** in model of state j
                        else
                        {
                            // Again, start with computing deriv of lam(t-1,d+1,j)*bs(t,j).
                            // Since bs(t,j) still does not depend on this coef, all that remains is:
                            if ((d+1) <= (D-2))
                            {
                                psi(t1,d,j,tci) = psi(t2,d+1,j,tci) * bs(t,j);
                            } else
                            {
                                psi(t1,d,j,tci) = 0;
                            }
                            // Compute deriv of (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * bs(t,j)*ds(d,j) with respect to individual coef:

                            // Again start with computing derivative of (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j))
                            tpg = 0.0;
                            for (size_t i = 0; i < n_S; i++)
                            {
                                if (i == j)
                                {
                                    continue;
                                }

                                tpg += psi(t2,0,i,tci)*T(i,j);
                            }
                            
                            // Only need deriv of (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * bs(t,j)*ds(d,j) - same as before:
                            psi(t1,d,j,tci) += tpg*bs(t,j)*ds(d,(starts_with_first) ? j : j + n_S);
                        }

                        tci += 1;
                    }
                    if (model_T)
                    {
                        for (size_t ci = 0; ci < T_grad_a.shape(1); ci++)
                        {
                            // Now onto deriv of a_ij with respect to individual coef.

                            if (j_idx_grad(tci) == j)
                            {
                                
                                // Compute deriv of lam(t-1,d+1,j)*bs(t,j) with respect to individual coef. Again only lam(t-1,d+1,j) depends on this coef.
                                if ((d+1) <= (D-2))
                                {
                                    psi(t1,d,j,tci) = psi(t2,d+1,j,tci) * bs(t,j);
                                } else
                                {
                                    psi(t1,d,j,tci) = 0;
                                }
                                // Compute deriv of (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * bs(t,j)*ds(d,j) with respect to individual coef.
                                // Have to be **careful** here: if j_idx_grad(tci) == j, then T(i,j) **does not** depend on this coef.

                                // So we only need (\sum_{i \in S/j} psi(t-1,1,i,tci)*T(i,j)) * bs(t,j)*ds(d,j)
                                tpg = 0.0;
                                for (size_t i = 0; i < n_S; i++)
                                {
                                    if (i == j)
                                    {
                                        continue;
                                    }

                                    tpg += psi(t2,0,i,tci)*T(i,j);
                                }

                                // Hence, deriv of (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * bs(t,j)*ds(d,j):
                                psi(t1,d,j,tci) += tpg*bs(t,j)*ds(d,(starts_with_first) ? j : j + n_S);
                            
                            }
                            // Coef ci is **NOT** in model of transitions away from j
                            else
                            {
                                // Again, start with computing deriv of lam(t-1,d+1,j)*bs(t,j).
                                // Since bs(t,j) still does not depend on this coef, all that remains is:
                                if ((d+1) <= (D-2))
                                {
                                    psi(t1,d,j,tci) = psi(t2,d+1,j,tci) * bs(t,j);
                                } else
                                {
                                    psi(t1,d,j,tci) = 0;
                                }
                                
                                // Compute deriv of (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * bs(t,j)*ds(d,j) with respect to individual coef.
                                // This time T(i,j) might depend on this coef: if j_idx_grad(tci) == i in the sum below. For that case we need
                                // to compute: psi(t-1,0,i,tci)*T(i,j) + lam(t-1,0,i)*(T(i,j) * T_grad(tidx,ci)), otherwise the second product cancels
                                // and we only need: psi(t-1,0,i,tci)*T(i,j)
                                tpg = 0.0;
                                for (size_t i = 0; i < n_S; i++)
                                {
                                    if (i == j)
                                    {
                                        continue;
                                    }

                                    tpg += psi(t2,0,i,tci)*T(i,j);

                                    if (j_idx_grad(tci) == i)
                                    {
                                        // Figure out location of derivative of transition we need: away from i -> j
                                        tidx = 0;

                                        for (size_t ti = 0; ti < n_S; ti++)
                                        {
                                            if (ti == i)
                                            {
                                                continue;
                                            }
                                            
                                            if (ti == j)
                                            {
                                                break;
                                            }

                                            tidx += 1;
                                        }
                                        
                                        tpg += lam(t-1,0,i) * T(i,j) * T_grad(tidx,ci);
                                    }
                                }

                                // Done:
                                psi(t1,d,j,tci) += tpg*bs(t,j)*ds(d,(starts_with_first) ? j : j + n_S);
                            }

                            tci += 1;
                        }
                    }
                    for (size_t ci = 0; ci < pi_grad_a.shape(1); ci++)
                    {

                        // Now onto deriv of pi(j) with respect to individual coef. Only lambda in the recursion depends on these coef, so it's simply
                        // the else case for all the above

                        // Again, start with computing deriv of lam(t-1,d+1,j)*bs(t,j).
                        // Since bs(t,j) still does not depend on this coef, all that remains is:
                        if ((d+1) <= (D-2))
                        {
                            psi(t1,d,j,tci) = psi(t2,d+1,j,tci) * bs(t,j);
                        } else
                        {
                            psi(t1,d,j,tci) = 0;
                        }
                        
                        // Compute deriv of (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * bs(t,j)*ds(d,j) with respect to individual coef:

                        // Again start with computing derivative of (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j))                        
                        tpg = 0.0;
                        for (size_t i = 0; i < n_S; i++)
                        {
                            if (i == j)
                            {
                                continue;
                            }

                            tpg += psi(t2,0,i,tci)*T(i,j);
                        }
                        
                        // Only need deriv of (\sum_{i \in S/j} lam(t-1,1,i)*T(i,j)) * bs(t,j)*ds(d,j) - same as before:
                        psi(t1,d,j,tci) += tpg*bs(t,j)*ds(d,(starts_with_first) ? j : j + n_S);

                        tci += 1;
                    }
                    // Scaling to get conditional probability
                    lam(t,d,j) /= Lam(t-1);
                    for (size_t tci = 0; tci < n_coef; tci++)
                    {
                        psi(t1,d,j,tci) /= Lam(t-1);
                    }
                    
                }

                // Compute conditional probability of current obs given all previous obs
                // Compute conditional probability of current obs given all previous obs
                if (t < (n_T - 1) || (ends_with_last == false))
                {   
                    // State does not have to end at last time point and also not in last state
                    // Before last time-point this is the right computation for all cases as well!
                    if (t < (n_T - 1) || ends_in_last == false)
                    {
                        Lam(t) += lam(t,d,j);
                    }
                    // State does not have to end at last time point but in last state
                    else if (j == (n_S-1))
                    {
                        Lam(t) += lam(t,d,j);
                    }
                }
                else if (d == 0)
                {   
                    // State does have to end at last time point but not in last state
                    if (ends_in_last == false)
                    {
                        Lam(t) += lam(t,d,j);
                    }
                    // State does have to end at last time point and also in last state
                    else if (j == (n_S-1))
                    {
                        Lam(t) += lam(t,d,j);
                    }
                    
                }
            }
        }
        llk += log(Lam(t));
    }

    if (ends_with_last)
    {   
        for (size_t tci = 0; tci < n_coef; tci++)
        {
            if (ends_in_last == false)
            {
                for (size_t j = 0; j < n_S; j++)
                {
                    grad(tci) += psi(t1,0,j,tci);
                }
            }
            else
            {
                grad(tci) += psi(t1,0,n_S-1,tci);
            }
            
            grad(tci) /= Lam(n_T-1);
        }
    }
    else
    {   
        for (size_t tci = 0; tci < n_coef; tci++)
        {
            for (size_t d = 0; d < (D-1); d++)
            {   
                if (ends_in_last == false)
                {
                    for (size_t j = 0; j < n_S; j++)
                    {
                        grad(tci) += psi(t1,d,j,tci);
                    }
                }
                else
                {
                    grad(tci) += psi(t1,d,n_S-1,tci);
                }
            }
            grad(tci) /= Lam(n_T-1);
        }
    }

    return grad_a;
}

void init_llk(py::module_ &m)
{
    m.def("llk", &llk, "Compute the log-likelihood for a hsmm.");
    m.def("llkFTPgrad", &llkFTPgrad, "Compute the gradient of the log-likelihood for a hsmm with fixed transition matrix and initial state distribution.");
    m.def("llkgrad", &llkgrad, "Compute the gradient of the log-likelihood for a hsmm.");
}