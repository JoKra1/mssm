from mssm.models import *
from mssm.src.python.compare import compare_CDL
import numpy as np
import os
from mssmViz.sim import*
from .defaults import default_compare_test_kwargs,default_gamm_test_kwargs,default_gammlss_test_kwargs,default_gsmm_test_kwargs,max_atol,max_rtol

mssm.src.python.exp_fam.PropHaz.init_lambda = init_penalties_tests_gsmm
mssm.src.python.utils.GAMLSSGSMMFamily.init_lambda = init_penalties_tests_gsmm

################################################################## Tests ##################################################################

class Test_AIC:

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs_gsmm = copy.deepcopy(default_gsmm_test_kwargs)
    test_kwargs["progress_bar"] = False
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["extend_lambda"] = False
    test_kwargs["control_lambda"] = 2

    test_kwargs_gsmm["progress_bar"] = False
    test_kwargs_gsmm["max_outer"] = 200
    test_kwargs_gsmm["max_inner"] = 500
    test_kwargs_gsmm["extend_lambda"] = False
    test_kwargs_gsmm["repara"] = True
    test_kwargs_gsmm["control_lambda"] = 2

    def test_smooth(self):
        target_UC_s = [[0.26,0.32,0.55,0.81,0.9,0.97,1.0,1.0,1.0,1.0],
                [0.28,0.31,0.49,0.66,0.86,0.95,1.0,1.0,1.0,1.0],
                [0.18,0.29,0.65,0.83,0.96,0.98,1.0,1.0,1.0,1.0],
                [0.41,0.59,0.83,0.98,1.0,1.0,1.0,1.0,1.0,1.0]]

        target_C_s = [[0.22,0.23,0.42,0.68,0.88,0.97,0.99,1.0,1.0,1.0],
                    [0.26,0.22,0.35,0.6,0.8,0.91,0.98,1.0,0.99,0.99],
                    [0.15,0.21,0.5,0.79,0.94,0.98,1.0,1.0,1.0,1.0],
                    [0.36,0.46,0.75,0.96,0.99,1.0,1.0,1.0,1.0,1.0]]

        n_c = 10
        n_sim = 100
        n_dat = 500
        df = 40
        nR = 250
        n_cores = 4

        families = ["Gaussian","Binomial","Gamma","PropHaz"]
        binom_offsets = [0,-5,0,0.1]
        scales = [2,2,2,2]

        ######################################## Smooth Selection ########################################

        for fi,(family,binom_offset,scale) in enumerate(zip(families,binom_offsets,scales)):

            AIC_rej = np.zeros(n_c)

            # WPS-like
            WPS_AIC_PQL_rej = np.zeros(n_c)

            for c_i,c_val in enumerate(np.linspace(0,1,n_c)):

                iterator = range(n_sim)
                for sim_i in iterator:

                    ######################################## Family setup ########################################
                    mod_fam = Gaussian()
                    if family == "Binomial":
                        mod_fam = Binomial()
                    if family == "Gamma":
                        mod_fam = Gamma()
                    
                    ######################################## Create Data ########################################
                    if family == "PropHaz":
                        sim_fit_dat = sim3(n=n_dat,scale=scale,c=c_val,family=PropHaz([0],[0]),binom_offset=binom_offset,seed=sim_i)
                        sim_fit_dat = sim_fit_dat.sort_values(['y'],ascending=[False])
                        sim_fit_dat = sim_fit_dat.reset_index(drop=True)

                        u,inv = np.unique(sim_fit_dat["y"],return_inverse=True)
                        ut = np.flip(u)
                        r = np.abs(inv - max(inv))
                    else:
                        sim_fit_dat = sim3(n=n_dat,scale=scale,c=c_val,family=mod_fam,binom_offset=binom_offset,seed=sim_i)

                    ######################################## First Formula + Models ########################################
                    if family == "PropHaz":
                        sim_fit_formula = Formula(lhs("delta"),
                                            [f(["x0"],nk=9),f(["x1"],nk=9),f(["x2"],nk=9),f(["x3"],nk=9)],
                                            data=sim_fit_dat,
                                            print_warn=False)
                
                        sim_formulas = [sim_fit_formula]

                        gsmm_newton_fam = PropHaz(ut,r)
                        sim_fit_model = GSMM(copy.deepcopy(sim_formulas),gsmm_newton_fam)
                        sim_fit_model.fit(**self.test_kwargs_gsmm)
                    else:
                        sim_fit_formula = Formula(lhs("y"),
                                                    [i(),f(["x0"],nk=9),f(["x1"],nk=9),f(["x2"],nk=9),f(["x3"],nk=9)],
                                                    data=sim_fit_dat,
                                                    print_warn=False)
                        
                        sim_fit_model = GAMM(sim_fit_formula,mod_fam)
                        sim_fit_model.fit(**self.test_kwargs)
                    
                    ######################################## Second Formula + Models ########################################
                    if family == "PropHaz":
                        sim_fit_formula2 = Formula(lhs("delta"),
                                            [f(["x1"],nk=9),f(["x2"],nk=9),f(["x3"],nk=9)],
                                            data=sim_fit_dat,
                                            print_warn=False)
                
                        sim_formulas2 = [sim_fit_formula2]

                        gsmm_newton_fam2 = PropHaz(ut,r)
                        sim_fit_model2 = GSMM(copy.deepcopy(sim_formulas2),gsmm_newton_fam2)
                        sim_fit_model2.fit(**self.test_kwargs_gsmm)
                    
                    else:
                        sim_fit_formula2 = Formula(lhs("y"),
                                                    [i(),f(["x1"],nk=9),f(["x2"],nk=9),f(["x3"],nk=9)],
                                                    data=sim_fit_dat,
                                                    print_warn=False)
                        

                        sim_fit_model2 = GAMM(sim_fit_formula2,mod_fam)
                        sim_fit_model2.fit(**self.test_kwargs)
                    
                    ######################################## Comparisons ########################################

                    # Uncorrected
                    uncor_result = compare_CDL(sim_fit_model,sim_fit_model2,correct_V=False,correct_t1=False,grid='JJJ1',seed=sim_i,verbose=False,only_expected_edf=False)

                    # WPS:
                    wps_pql_result = compare_CDL(sim_fit_model,sim_fit_model2,nR=nR,df=df,correct_t1=False,n_c=n_cores,grid='JJJ1',Vp_fidiff=False,seed=sim_i,verbose=False,only_expected_edf=False,compute_Vcc=True)
                    
                    if uncor_result["aic_diff"] < 0:
                        AIC_rej[c_i] += 1
                    
                    if wps_pql_result["aic_diff"] < 0:
                        WPS_AIC_PQL_rej[c_i] += 1
            
            ######################################## Compare ########################################
            AIC_rej /= n_sim
            WPS_AIC_PQL_rej /= n_sim

            np.testing.assert_allclose(AIC_rej,np.array(target_UC_s[fi]),atol=min(max_atol,0.1),rtol=min(max_rtol,0.001))
            np.testing.assert_allclose(WPS_AIC_PQL_rej,np.array(target_C_s[fi]),atol=min(max_atol,0.1),rtol=min(max_rtol,0.001))
    
    def test_re(self):

        target_UC_re = [[0.5,0.51,0.74,0.87,0.97,1.0,1.0,1.0,1.0,1.0],
                    [0.5,0.52,0.54,0.73,0.91,0.93,0.93,0.96,0.93,0.91],
                    [0.36,0.47,0.64,0.87,0.97,0.99,1.0,1.0,1.0,1.0],
                    [0.67,0.8,0.95,1.0,1.0,1.0,1.0,1.0,1.0,1.0]]
        
        target_C_re = [[0.11,0.12,0.2,0.32,0.65,0.86,0.99,1.0,1.0,1.0],
                    [0.17,0.11,0.11,0.14,0.28,0.63,0.79,0.89,0.88,0.87],
                    [0.18,0.21,0.33,0.71,0.89,0.98,1.0,1.0,1.0,1.0],
                    [0.24,0.35,0.69,0.93,0.99,1.0,1.0,1.0,1.0,1.0]]
        
        n_c = 10
        n_sim = 100
        n_dat = 500
        df = 40
        nR = 250
        n_cores = 4

        families = ["Gaussian","Binomial","Gamma","PropHaz"]
        binom_offsets = [0,-5,0,0.1]
        scales = [2,2,2,2]

        ######################################## Random Effect Selection ########################################

        for fi,(family,binom_offset,scale) in enumerate(zip(families,binom_offsets,scales)):

            AIC_rej = np.zeros(n_c)

            # WPS-like
            WPS_AIC_PQL_rej = np.zeros(n_c)

            for c_i,c_val in enumerate(np.linspace(0,1,n_c)):

                iterator = range(n_sim)
                for sim_i in iterator:

                    ######################################## Family setup ########################################
                    mod_fam = Gaussian()
                    if family == "Binomial":
                        mod_fam = Binomial()
                    if family == "Gamma":
                        mod_fam = Gamma()
                    
                    ######################################## Create Data ########################################
                    if family == "PropHaz":
                        sim_fit_dat = sim4(n=n_dat,scale=scale,c=c_val,family=PropHaz([0],[0]),binom_offset=binom_offset,seed=sim_i)
                        sim_fit_dat = sim_fit_dat.sort_values(['y'],ascending=[False])
                        sim_fit_dat = sim_fit_dat.reset_index(drop=True)

                        u,inv = np.unique(sim_fit_dat["y"],return_inverse=True)
                        ut = np.flip(u)
                        r = np.abs(inv - max(inv))
                    else:
                        sim_fit_dat = sim4(n=n_dat,scale=scale,c=c_val,family=mod_fam,binom_offset=binom_offset,seed=sim_i)

                    ######################################## First Formula + Models ########################################
                    if family == "PropHaz":
                        sim_fit_formula = Formula(lhs("delta"),
                                            [f(["x0"],nk=9),f(["x1"],nk=9),f(["x2"],nk=9),f(["x3"],nk=9),ri("x4")],
                                            data=sim_fit_dat,
                                            print_warn=False)
                
                        sim_formulas = [sim_fit_formula]

                        gsmm_newton_fam = PropHaz(ut,r)
                        sim_fit_model = GSMM(copy.deepcopy(sim_formulas),gsmm_newton_fam)
                        sim_fit_model.fit(**self.test_kwargs_gsmm)
                    else:
                        sim_fit_formula = Formula(lhs("y"),
                                                    [i(),f(["x0"],nk=9),f(["x1"],nk=9),f(["x2"],nk=9),f(["x3"],nk=9),ri("x4")],
                                                    data=sim_fit_dat,
                                                    print_warn=False)
                        
                        sim_fit_model = GAMM(sim_fit_formula,mod_fam)
                        sim_fit_model.fit(**self.test_kwargs)
                    
                    ######################################## Second Formula + Models ########################################
                    if family == "PropHaz":
                        sim_fit_formula2 = Formula(lhs("delta"),
                                            [f(["x0"],nk=9),f(["x1"],nk=9),f(["x2"],nk=9),f(["x3"],nk=9)],
                                            data=sim_fit_dat,
                                            print_warn=False)
                
                        sim_formulas2 = [sim_fit_formula2]

                        gsmm_newton_fam2 = PropHaz(ut,r)
                        sim_fit_model2 = GSMM(copy.deepcopy(sim_formulas2),gsmm_newton_fam2)
                        sim_fit_model2.fit(**self.test_kwargs_gsmm)
                    
                    else:
                        sim_fit_formula2 = Formula(lhs("y"),
                                                    [i(),f(["x0"],nk=9),f(["x1"],nk=9),f(["x2"],nk=9),f(["x3"],nk=9)],
                                                    data=sim_fit_dat,
                                                    print_warn=False)
                        

                        sim_fit_model2 = GAMM(sim_fit_formula2,mod_fam)
                        sim_fit_model2.fit(**self.test_kwargs)
                    
                    ######################################## Comparisons ########################################

                    # Uncorrected
                    uncor_result = compare_CDL(sim_fit_model,sim_fit_model2,correct_V=False,correct_t1=False,grid='JJJ1',seed=sim_i,verbose=False,only_expected_edf=False)

                    # WPS:
                    wps_pql_result = compare_CDL(sim_fit_model,sim_fit_model2,nR=nR,df=df,correct_t1=False,n_c=n_cores,grid='JJJ1',Vp_fidiff=False,seed=sim_i,verbose=False,only_expected_edf=False,compute_Vcc=True)
                    
                    if uncor_result["aic_diff"] < 0:
                        AIC_rej[c_i] += 1
                    
                    if wps_pql_result["aic_diff"] < 0:
                        WPS_AIC_PQL_rej[c_i] += 1
            
            ######################################## Compare ########################################
            AIC_rej /= n_sim
            WPS_AIC_PQL_rej /= n_sim

            np.testing.assert_allclose(AIC_rej,np.array(target_UC_re[fi]),atol=min(max_atol,0.1),rtol=min(max_rtol,0.001))
            np.testing.assert_allclose(WPS_AIC_PQL_rej,np.array(target_C_re[fi]),atol=min(max_atol,0.1),rtol=min(max_rtol,0.001))


class Test_p():
    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["progress_bar"] = False
    test_kwargs["max_outer"] = 200
    test_kwargs["max_inner"] = 500
    test_kwargs["extend_lambda"] = False
    test_kwargs["control_lambda"] = 2

    n_sim = 1000
    n_dat = 500

    ############################################### Univariate Smooth ############################################### 

    ps_n = []
    ps_bi = []
    ps_te = []
    ps_bi_te = []

    ps_GLRT_n = []
    ps_GLRT_bi = []
    ps_GLRT_te = []
    ps_GLRT_bi_te = []


    for simi in tqdm(range(n_sim)):
        sim_dat = sim3(n=n_dat,scale=2,c=0,seed=simi)
        
        formula1 = Formula(lhs("y"),[i(),f(["x0"]),f(["x1"]),f(["x2"]),f(["x3"])],data=sim_dat)
        model1 = GAMM(formula1,Gaussian())
        model1.fit(**test_kwargs)
        formula2 = Formula(lhs("y"),[i(),f(["x1"]),f(["x2"]),f(["x3"])],data=sim_dat)
        model2 = GAMM(formula2,Gaussian())
        model2.fit(**test_kwargs)

        # Compute bunch of p-values
        res = compare_CDL(model1,model2,correct_t1=True,perform_GLRT=True,grid='JJJ1',compute_Vcc=True,only_expected_edf=False)
        pss, Trs = approx_smooth_p_values(model1)
        ps_GLRT_n.append(res['p'])
        ps_n.append(pss[0])

        # Binomial
        sim_dat = sim3(n=n_dat,scale=2,c=0,seed=simi,family=Binomial(),binom_offset=-5)
        
        formula1 = Formula(lhs("y"),[i(),f(["x0"]),f(["x1"]),f(["x2"]),f(["x3"])],data=sim_dat)
        model1 = GAMM(formula1,Binomial())
        model1.fit(**test_kwargs)
        formula2 = Formula(lhs("y"),[i(),f(["x1"]),f(["x2"]),f(["x3"])],data=sim_dat)
        model2 = GAMM(formula2,Binomial())
        model2.fit(**test_kwargs)

        # Compute bunch of p-values
        res = compare_CDL(model1,model2,correct_t1=True,perform_GLRT=True,grid='JJJ1',compute_Vcc=True,only_expected_edf=False)
        pss, Trs = approx_smooth_p_values(model1)
        ps_GLRT_bi.append(res['p'])
        ps_bi.append(pss[0])

        ############################################### Tensor Smooth ############################################### 

        sim_dat = sim3(n=n_dat,scale=2,c=0,seed=simi)
        
        formula1 = Formula(lhs("y"),[i(),f(["x0","x3"],te=True,nk=5),f(["x1"]),f(["x2"])],data=sim_dat)
        model1 = GAMM(formula1,Gaussian())
        model1.fit(**test_kwargs)
        formula2 = Formula(lhs("y"),[i(),f(["x1"]),f(["x2"])],data=sim_dat)
        model2 = GAMM(formula2,Gaussian())
        model2.fit(**test_kwargs)

        # Compute bunch of p-values
        res = compare_CDL(model1,model2,correct_t1=True,perform_GLRT=True,grid='JJJ1',compute_Vcc=True,only_expected_edf=False)
        pss, Trs = approx_smooth_p_values(model1)
        ps_GLRT_te.append(res['p'])
        ps_te.append(pss[0])

        # Binomial
        sim_dat = sim3(n=n_dat,scale=2,c=0,seed=simi,family=Binomial(),binom_offset=-5)
        
        formula1 = Formula(lhs("y"),[i(),f(["x0","x3"],te=True,nk=5),f(["x1"]),f(["x2"])],data=sim_dat)
        model1 = GAMM(formula1,Binomial())
        model1.fit(**test_kwargs)
        formula2 = Formula(lhs("y"),[i(),f(["x1"]),f(["x2"])],data=sim_dat)
        model2 = GAMM(formula2,Binomial())
        model2.fit(**test_kwargs)

        # Compute bunch of p-values
        res = compare_CDL(model1,model2,correct_t1=True,perform_GLRT=True,grid='JJJ1',compute_Vcc=True,only_expected_edf=False)
        pss, Trs = approx_smooth_p_values(model1)
        ps_GLRT_bi_te.append(res['p'])
        ps_bi_te.append(pss[0])

        """
        fig = plt.figure(figsize=(full_width,2*single_width),layout='constrained')
        axs = fig.subplots(2,4,gridspec_kw=dict(wspace=0.01,hspace=0.1))
        axs = axs.flatten()

        all_ps = [ps_n,ps_bi,ps_te,ps_bi_te,ps_GLRT_n,ps_GLRT_bi,ps_GLRT_te,ps_GLRT_bi_te]
        all_titles = ["Gaussian Univariate", "Binomial Univariate",
                    "Gaussian Tensor", "Binomial Tensor",
                    "Gaussian Univariate GLRT", "Binomial Univariate GLRT",
                    "Gaussian Tensor GLRT", "Binomial Tensor GLRT"]

        for axi in range(len(all_ps)):
            ps = all_ps[axi]
            ax = axs[axi]

            ps = np.array(ps)
            ps[np.isnan(ps)] = 1
            qs = np.linspace(5/(len(ps)*10),1 - (5/(len(ps)*10)),len(ps))
            qs = scp.stats.uniform.ppf(qs)

            ax.plot(qs,np.sort(ps),color="gray",linestyle='dashed')
            ax.plot(qs,qs,color="black")

            ax.set_ylim(0,1)
            ax.set_xlim(0,1)
            ax.set_xticks(np.linspace(0,1,5))
            ax.set_yticks(np.linspace(0,1,5))
            ax.set_ylabel("Sorted p-values")
            ax.set_xlabel("Uniform Quantiles")
            ax.set_title(all_titles[axi])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)

        fig.suptitle(f"N-sim: {n_sim} N-dat: {n_dat}")
        plt.show()
        """

    
    def test_smooth_ps(self):
        target_maxs = [np.float64(0.0679164308433986), np.float64(0.06424000021977894), np.float64(0.08807165227309743), np.float64(0.06344989032965709)]
        target_mins = [np.float64(2.3445733646099143e-05), np.float64(6.428050062289026e-05), np.float64(1.8074690926600523e-07), np.float64(9.071336898897053e-06)]
        target_means = [np.float64(0.03891386943084419), np.float64(0.03513198819364352), np.float64(0.05422584548421288), np.float64(0.02910613985177542)]
        target_sds = [np.float64(0.020795458429035945), np.float64(0.01973019288235943), np.float64(0.024944702357081044), np.float64(0.019840937577255587)]

        smooth_ps = [self.ps_n,self.ps_bi,self.ps_te,self.ps_bi_te]

        for pi,ps in enumerate(smooth_ps):
            ps = np.sort(np.array(ps))
            qs = np.linspace(5/(len(ps)*10),1 - (5/(len(ps)*10)),len(ps))
            qs = scp.stats.uniform.ppf(qs)

            abs_diff = np.abs(ps - qs)

            np.testing.assert_allclose(np.max(abs_diff),target_maxs[pi],atol=min(max_atol,0.01),rtol=min(max_rtol,0.001))
            np.testing.assert_allclose(np.min(abs_diff),target_mins[pi],atol=min(max_atol,0.01),rtol=min(max_rtol,0.001))
            np.testing.assert_allclose(np.mean(abs_diff),target_means[pi],atol=min(max_atol,0.01),rtol=min(max_rtol,0.001))
            np.testing.assert_allclose(np.std(abs_diff),target_sds[pi],atol=min(max_atol,0.01),rtol=min(max_rtol,0.001))


    def test_glrt_ps(self):
        target_maxs = [np.float64(0.12150000000000005), np.float64(0.10847261242526818), np.float64(0.09324928971431867), np.float64(0.11305069761436304)]
        target_mins = [np.float64(5.8675509883751326e-05), np.float64(0.00024181133170919633), np.float64(5.73024513722039e-05), np.float64(7.108436494673853e-05)]
        target_means = [np.float64(0.05499944319067722), np.float64(0.0654051156331472), np.float64(0.055488402485988055), np.float64(0.06981106912555282)]
        target_sds = [np.float64(0.026160141507717727), np.float64(0.031557828969396476), np.float64(0.027950147640471663), np.float64(0.03177174225215185)]

        glrt_ps = [self.ps_GLRT_n,self.ps_GLRT_bi,self.ps_GLRT_te,self.ps_GLRT_bi_te]

        for pi,ps in enumerate(glrt_ps):
            ps = np.array(ps)
            ps[np.isnan(ps)] = 1
            ps = np.sort(ps)
            qs = np.linspace(5/(len(ps)*10),1 - (5/(len(ps)*10)),len(ps))
            qs = scp.stats.uniform.ppf(qs)

            abs_diff = np.abs(ps - qs)

            np.testing.assert_allclose(np.max(abs_diff),target_maxs[pi],atol=min(max_atol,0.1),rtol=min(max_rtol,0.001))
            np.testing.assert_allclose(np.min(abs_diff),target_mins[pi],atol=min(max_atol,0.1),rtol=min(max_rtol,0.001))
            np.testing.assert_allclose(np.mean(abs_diff),target_means[pi],atol=min(max_atol,0.1),rtol=min(max_rtol,0.001))
            np.testing.assert_allclose(np.std(abs_diff),target_sds[pi],atol=min(max_atol,0.1),rtol=min(max_rtol,0.001))


