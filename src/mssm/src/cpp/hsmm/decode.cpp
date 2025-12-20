#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <random>

namespace py = pybind11;


std::tuple<
           int,
           py::array_t<int>,
           py::array_t<double>,
           py::array_t<int>
> viterbi(py::array_t<double, py::array::f_style | py::array::forcecast> lbs_a,
          py::array_t<double, py::array::f_style | py::array::forcecast> lds_a,
          py::array_t<double, py::array::f_style | py::array::forcecast> lT_a,
          py::array_t<double, py::array::f_style | py::array::forcecast> lpi_a,
          py::array_t<double, py::array::f_style | py::array::forcecast> weights_a,
          double scale, size_t n_T, size_t D, size_t n_S, size_t event_width,
          bool starts_with_first, bool ends_with_last,bool ends_in_last,
          int hmp_code, bool tvdtpi) {
    
    /* Viterbi algorithm (see Rabiner, 1990) adapted for the forward pass defined by Yu & Kobayashi
    (2006).

    References:
      - Rabiner, L. R. (1990). A tutorial on hidden Markov models and selected applications in
        speech recognition. In Readings in speech recognition (pp. 267–296). Morgan Kaufmann
        Publishers Inc.
      - Yu, S.-Z., & Kobayashi, H. (2006). Practical implementation of an efficient forward-backward
        algorithm for an explicit-duration hidden Markov model. IEEE Transactions on Signal
        Processing, 54(5), 1947–1951. https://doi.org/10.1109/TSP.2006.872540
    */
    auto delta_a = py::array_t<double>({n_T,D-1,n_S});
    auto psi_a = py::array_t<int>({n_T,D-1,n_S});
    delta_a[py::make_tuple(py::ellipsis())] = -INFINITY;
    psi_a[py::make_tuple(py::ellipsis())] = -1;

    auto delta = delta_a.mutable_unchecked();
    auto psi = psi_a.mutable_unchecked();
    auto lbs = lbs_a.unchecked();
    auto lds = lds_a.unchecked();
    auto lT = lT_a.unchecked();
    auto lpi = lpi_a.unchecked();
    auto weights = weights_a.unchecked();

    double tmp_d,tmp_d2, lb2;
    int max_ed = 0;
    int tmp_j,tmp_j2,tmp_ed;

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
                        delta(t,d,j) = lpi(j)+lbs(t,0,j)+((tvdtpi) ? lds(d,j,t) : lds(d,j));
                    }
                    else
                    {
                        delta(t,d,j) = lpi(j)+lbs(t,j)+((tvdtpi) ? lds(d,j,t) : lds(d,j));
                    }
                    
                }
                else if ((t + d) < n_T || ends_with_last == false)
                {
                    // Probability of same state continuing
                    tmp_d = -INFINITY;
                    tmp_d2 = -INFINITY;
                    tmp_j = -1;
                    if ((d+1) <= (D-2))
                    {
                        if (hmp_code != 0)
                        {
                            if (j % 2 == 1)
                            {
                                if (d <= event_width-1)
                                {
                                    /* bump */
                                    tmp_d2 = delta(t-1,d+1,j) + lbs(t,(event_width-1) - d,j);
                                }
                                else
                                {
                                    tmp_d2 = delta(t-1,d+1,j) -INFINITY;
                                }
                            }
                            else
                            {
                                /* flat */
                                tmp_d2 = delta(t-1,d+1,j) + lbs(t,0,j);
                            }
                        }
                        else
                        {
                            tmp_d2 = delta(t-1,d+1,j) + lbs(t,j);
                        }
                        
                        if (tmp_d2 > tmp_d)
                        {
                            tmp_d = tmp_d2;
                            tmp_j = j;
                        }
                    }
                    // Probability of other state ending on previous time point and transitioning to
                    // current state with dur d
                    for (size_t i = 0; i < n_S; i++)
                    {
                        if (i == j)
                        {
                            continue;
                        }
                        
                        if (hmp_code != 0)
                        {
                            lb2 = lbs(t,0,j);
                            if(j % 2 == 0)
                            {
                                // If we have an ar1 model first obs prob for flats depends on lag
                                // of previous bump (only for states > 0, but this is taken care of
                                // outside)
                                lb2 = lbs(t,1,j);
                            }
                            tmp_d2 = delta(t-1,0,i) + ((tvdtpi) ? lT(i,j,t-1): lT(i,j)) + lb2 +
                                     ((tvdtpi) ? lds(d,(starts_with_first) ? j : j + n_S,t):
                                                 lds(d,(starts_with_first) ? j : j + n_S));
                        }
                        else
                        {
                            tmp_d2 = delta(t-1,0,i) + ((tvdtpi) ? lT(i,j,t-1): lT(i,j)) + lbs(t,j) +
                                     ((tvdtpi) ? lds(d,(starts_with_first) ? j : j + n_S,t):
                                                 lds(d,(starts_with_first) ? j : j + n_S));
                        }
                        

                        if (tmp_d2 > tmp_d)
                        {
                            tmp_d = tmp_d2;
                            tmp_j = i;
                        }
                    }
                    // tmp_d holds max transition prob and tmp_j argmax
                    delta(t,d,j) = tmp_d;
                    psi(t,d,j) = tmp_j;
                }
            }
        }
    }

    // Select final state
    tmp_d = -INFINITY;
    tmp_j = -1;
    if (ends_with_last == false)
    {
        if (ends_in_last == false) // argmax final state over j and d
        {
            for (size_t j = 0; j < n_S; j++)
            {
                for (size_t d = 0; d < (D-1); d++)
                {
                    if (delta(n_T-1,d,j) > tmp_d)
                    {
                        tmp_d = delta(n_T-1,d,j);
                        tmp_j = j;
                        max_ed = d;
                    }
                }
            }
        } else // argmax final state over d while fixing j at n_S-1
        {
            for (size_t d = 0; d < (D-1); d++)
            {
                if (delta(n_T-1,d,n_S-1) > tmp_d)
                {
                    tmp_d = delta(n_T-1,d,n_S-1);
                    tmp_j = n_S-1;
                    max_ed = d;
                }
            }
        }
    } else
    {
        if (ends_in_last == false) // argmax final state over j while fixing d at 0
        {
            for (size_t j = 0; j < n_S; j++)
            {

                if (delta(n_T-1,0,j) > tmp_d)
                {
                    tmp_d = delta(n_T-1,0,j);
                    tmp_j = j;
                }
                
            }
        } else // argmax final state is d==0 and n_S-1
        {
            tmp_d = delta(n_T-1,0,n_S-1);
            tmp_j = n_S-1;

        }
    }

    // Backward pass
    auto states_a = py::array_t<int>(n_T);
    states_a[py::make_tuple(py::ellipsis())] = 0;
    auto states = states_a.mutable_unchecked();
    states[n_T-1] = tmp_j;
    tmp_ed = max_ed;
    
    size_t t = n_T-1;
    while (t > 0)
    {   
        tmp_j2 = psi(t,tmp_ed,tmp_j);

        if (tmp_j2 == tmp_j) // Were in same state at time-point before
        {
            tmp_ed += 1;
        }
        else // Other state ended at time-point before
        {
            tmp_ed = 0;
        }
        
        // Collect state
        states[t-1] = tmp_j2;

        // Prepare next iteration
        tmp_j = tmp_j2;
        t--;
    }
    
    return std::make_tuple(max_ed,std::move(states_a),std::move(delta_a),std::move(psi_a));;
}

std::tuple<
           py::array_t<int>,
           py::array_t<int>
> sample_backwards(
              py::array_t<double, py::array::f_style | py::array::forcecast> bs_a,
              py::array_t<double, py::array::f_style | py::array::forcecast> ds_a,
              py::array_t<double, py::array::f_style | py::array::forcecast> T_a,
              py::array_t<double, py::array::f_style | py::array::forcecast> pi_a,
              py::array_t<double, py::array::f_style | py::array::forcecast> weights_a,
              double scale, size_t n_T, size_t D, size_t n_S, size_t event_width, size_t n_samples,
              bool starts_with_first,bool ends_with_last,bool ends_in_last,int hmp_code, int seed,
              bool tvdtpi) {
    
    /* Backwards sampling algorithm to obtain samples from the posterior of state sequences given
    paratmeters and data, based on Dewar et al. (2012). Adapted for the forward pass defined by
    Yu & Kobayashi (2006).

    References:
      - Dewar, M., Wiggins, C., & Wood, F. (2012). Inference in Hidden Markov Models with Explicit
        State Duration Distributions. IEEE Signal Processing Letters, 19(4), 235–238.
        https://doi.org/10.1109/LSP.2012.2184795
      - Yu, S.-Z., & Kobayashi, H. (2006). Practical implementation of an efficient forward-backward
        algorithm for an explicit-duration hidden Markov model. IEEE Transactions on Signal
        Processing, 54(5), 1947–1951. https://doi.org/10.1109/TSP.2006.872540
    */
    auto Lam_a = py::array_t<double>(n_T);
    auto lam_a = py::array_t<double>({n_T,D-1,n_S});

    // P(o_t|o_{1:t-1})
    Lam_a[py::make_tuple(py::ellipsis())] = 0.0;

    // P(S_t = j,D_t = d|o_{1:t})
    lam_a[py::make_tuple(py::ellipsis())] = 0.0;

    auto Lam = Lam_a.mutable_unchecked();
    auto lam = lam_a.mutable_unchecked();
    auto bs = bs_a.unchecked();
    auto ds = ds_a.unchecked();
    auto T = T_a.unchecked();
    auto pi = pi_a.unchecked();
    auto weights = weights_a.unchecked();

    // Probabilities for sampling last state
    std::vector<double> state_probsT(n_S);
    std::vector<double> state_probs(n_S);
    std::vector<std::vector<double>> dur_probs(n_S,std::vector<double>(D-1));

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
                        lam(t,d,j) = pi(j)*bs(t,0,j)*((tvdtpi) ? ds(d,j,t): ds(d,j));
                    }
                    else
                    {
                        lam(t,d,j) = pi(j)*bs(t,j)*((tvdtpi) ? ds(d,j,t): ds(d,j));
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

                        tp += lam(t-1,0,i)*((tvdtpi) ? T(i,j,t-1): T(i,j));
                    }
                    
                    if (hmp_code != 0)
                    {
                        b2 = bs(t,0,j);
                        if(j % 2 == 0)
                        {
                            // If we have an ar1 model first obs prob for flats depends on lag of
                            // previous bump (only for states > 0, but this is taken care of
                            // outside)
                            b2 = bs(t,1,j);
                        }
                        tp *= b2*((tvdtpi) ? ds(d,(starts_with_first) ? j : j + n_S,t):
                                             ds(d,(starts_with_first) ? j : j + n_S));
                    }
                    else
                    {
                        tp *= bs(t,j)*((tvdtpi) ? ds(d,(starts_with_first) ? j : j + n_S,t):
                                                  ds(d,(starts_with_first) ? j : j + n_S));
                    }                    

                    lam(t,d,j) += tp;

                }
                
                // Compute conditional probability of current obs given all previous obs
                if (t < (n_T - 1) || (ends_with_last == false))
                {   
                    // State does not have to end at last time point and also not in last state
                    // Before last time-point this is the right computation for all cases as well!
                    if (t < (n_T - 1) || ends_in_last == false)
                    {
                        Lam(t) += lam(t,d,j);

                        if (t == (n_T - 1))
                        {
                            state_probsT[j] += lam(t,d,j);
                            dur_probs[j][d] += lam(t,d,j);
                        }
                    }
                    // State does not have to end at last time point but in last state
                    else if (j == (n_S-1))
                    {
                        Lam(t) += lam(t,d,j);
                        state_probsT[j] += lam(t,d,j);
                        dur_probs[j][d] += lam(t,d,j);
                    }
                }
                else if (d == 0)
                {   
                    // State does have to end at last time point but not in last state
                    if (ends_in_last == false)
                    {
                        Lam(t) += lam(t,d,j);
                        state_probsT[j] += lam(t,d,j);
                        dur_probs[j][d] += lam(t,d,j);
                    }
                    // State does have to end at last time point and also in last state
                    else if (j == (n_S-1))
                    {
                        Lam(t) += lam(t,d,j);
                        state_probsT[j] += lam(t,d,j);
                        dur_probs[j][d] += lam(t,d,j);
                    }
                    
                }
            }
        }

        // Now need an additional scaling pass over d and j
        for (size_t d = 0; d < (D-1); d++)
        {
            for (size_t j = 0; j < n_S; j++)
            {
                // Scaling to get conditional probability P(S_t = j,D_t = d|o_{1:t})
                // from P(S_t = j,D_t = d, o_t|o_{1:t-1})
                lam(t,d,j) /= Lam(t);
            }
        } 
    }

    // Backward pass

    // Sampled state sequences
    auto states_a = py::array_t<int>({n_T,n_samples});
    states_a[py::make_tuple(py::ellipsis())] = 0;
    auto states = states_a.mutable_unchecked();
    
    // Samples excess duration
    auto ed_a = py::array_t<int>(n_samples);
    ed_a[py::make_tuple(py::ellipsis())] = 0;
    auto ed = ed_a.mutable_unchecked();

    // Compute dur probs conditional on last state
    for (size_t j = 0; j < n_S; j++)
    {
        /*
        state_probsT[j] = P(o_T,S_T=j|o_{1:T-1}).
        Lam(n_T-1) = P(o_T|o_{1:T-1}).

        So: state_probsT[j] / Lam(n_T-1) = P(S_T=j|o_{1:T})
        */
        state_probsT[j] /= Lam(n_T-1);
        
        for (size_t d = 0; d < (D-1); d++)
        {
                /*
            dur_probs[j][d] = P(o_T,S_T=j,D_T=d|o_{1:T-1}).
            Lam(n_T-1) = P(o_T|o_{1:T-1}).

            So: dur_probs[j][d] /= Lam(n_T-1) = P(S_T=j,D_T=d|o_{1:T})
            and: dur_probs[j][d] / state_probsT[j] = P(D_T=d|o_{1:T},S_T=j)
            */
            dur_probs[j][d] /= Lam(n_T-1);
            dur_probs[j][d] /= state_probsT[j];
        }
    }

    // Now can start sampling
    int tmp_j,tmp_j2,tmp_ed;
    std::random_device rd;
    std::mt19937 gen(rd());

    // Optionally re-seed random generator with provided seed
    if (seed >= 0) {
        gen.seed(seed);
    }

    // Sampler for last state:
    std::discrete_distribution<int>sT(state_probsT.begin(),state_probsT.end());

    for (size_t s = 0; s < n_samples; s++)
    {

        // Sample state at last time-point T
        tmp_j = sT(gen);
        
        // Sampler for last state's distribution:
        std::discrete_distribution<int>sD(dur_probs[tmp_j].begin(),dur_probs[tmp_j].end());

        tmp_ed = sD(gen);

        // Collect last state and excess duration
        states(n_T-1,s) = tmp_j;
        ed(s) = tmp_ed;

        size_t t = n_T-1;

        // Now sample state and residual duration at t-1 given state t
        while (t > 0)
        {   
            /*
            Ok, following is possible given S_t=tmp_j and D_t=tmp_ed
            1) State S_{t-1} = tmp_j and D_{t-1} = tmp_ed + 1. This is zero if tmp_ed + 1 > D - 2
            2) State S_{t-1} != tmp_j and D_{t-1} = 1. I.e., another state ended at previous time

            lam(t,d,j) has P(S_t=j,D_T=d|o_{1:t})!
            */

            for (size_t i = 0; i < n_S; i++)
            {
                if (i == tmp_j)
                {
                    if ((tmp_ed + 1) <= (D - 2))
                    {
                        state_probs[i] = lam(t-1,tmp_ed+1,i);
                    }
                    else
                    {
                        state_probs[i] = 0.0;
                    }
                }
                else
                {
                    state_probs[i] = lam(t-1,0,i) *
                                        ((tvdtpi) ? T(i,tmp_j,t-1): T(i,tmp_j)) *
                                        ((tvdtpi) ?
                                            ds(tmp_ed,(starts_with_first) ? tmp_j : tmp_j + n_S,t-1)
                                            :
                                            ds(tmp_ed,(starts_with_first) ? tmp_j : tmp_j + n_S)
                                        );
                }
            }
            
            // Can now sample S_{t-1}
            std::discrete_distribution<int>st(state_probs.begin(),state_probs.end());

            tmp_j2 = st(gen);
            //py::print(t,tmp_j,"->",tmp_j2,state_probs);
            
            // State continues
            if(tmp_j == tmp_j2)
            {
                tmp_ed++;
            }
            else
            {
                tmp_j = tmp_j2;
                tmp_ed = 0;
            }

            // Collect
            states(t-1,s) = tmp_j;

            // Update time
            t--;

        }
    }

    return std::make_tuple(std::move(ed_a),std::move(states_a));;
}

void init_decode(py::module_ &m)
{
    m.def("viterbi", &viterbi, "Perfrom viterbi decoding for a hsmm.");
    m.def("sample_backwards", &sample_backwards, "Sample posterior of state sequence for a hsmm.");
}