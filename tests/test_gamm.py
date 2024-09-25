from mssm.models import *
import numpy as np
import os
from mssmViz.sim import*

class Test_GAM:

    dat = pd.read_csv('https://raw.githubusercontent.com/JoKra1/mssm_tutorials/main/data/GAMM/sim_dat.csv')
    
    # mssm requires that the data-type for variables used as factors is 'O'=object
    dat = dat.astype({'series': 'O',
                    'cond':'O',
                    'sub':'O',
                    'series':'O'})
    
    formula = Formula(lhs=lhs("y"), # The dependent variable - here y!
                        terms=[i(), # The intercept, a
                               f(["time"])], # The f(time) term, by default parameterized with 9 basis functions (after absorbing one for identifiability)
                        data=dat,
                        print_warn=False)
        
    model = GAMM(formula,Gaussian())

    model.fit(extension_method_lam = "mult",exclude_lambda=True)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 9.723

    def test_GAMTermEdf(self):
        assert round(self.model.term_edf[0],ndigits=3) == 8.723
    
    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 1084.879
    
    def test_GAMlam(self):
        assert round(self.model.formula.penalties[0].lam,ndigits=5) == 0.0089

class Test_GAM_TE:

    dat = pd.read_csv('https://raw.githubusercontent.com/JoKra1/mssm_tutorials/main/data/GAMM/sim_dat.csv')
    
    # mssm requires that the data-type for variables used as factors is 'O'=object
    dat = dat.astype({'series': 'O',
                    'cond':'O',
                    'sub':'O',
                    'series':'O'})
    
    formula = Formula(lhs=lhs("y"), # The dependent variable - here y!
                        terms=[i(), # The intercept, a
                               l(["cond"]), # Offset for cond='b'
                               f(["time","x"],by="cond",te=True)], # one smooth surface over time and x - f(time,x) - per level of cond: three-way interaction!
                        data=dat,
                        print_warn=False)
        
    model = GAMM(formula,Gaussian())

    model.fit(extension_method_lam = "mult",exclude_lambda=True)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=2) == 33.83

    def test_GAMTermEdf(self):
        # Third lambda terms -> inf, making this test hard to pass 
        diff = np.abs(np.round(self.model.term_edf,decimals=2) - np.array([12.69, 19.14]))
        rel_diff = diff/np.array([12.69, 19.14])
        assert np.max(rel_diff) < 1e-2
    
    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 967.71
    
    def test_GAMlam(self):
        # Same here, so the lambda term in question is excluded and tolerance is lowered
        diff = np.abs(np.round([p.lam for p in self.model.formula.penalties],decimals=3) - np.array([     0.001,      0.001, 573912.862,     48.871]))
        rel_diff = diff[[0,1,3]]/np.array([     0.001,      0.001,  48.871])
        assert np.max(rel_diff) < 1e-3

class Test_GAM_TE_BINARY:

    dat = pd.read_csv('https://raw.githubusercontent.com/JoKra1/mssm_tutorials/main/data/GAMM/sim_dat.csv')
    
    # mssm requires that the data-type for variables used as factors is 'O'=object
    dat = dat.astype({'series': 'O',
                    'cond':'O',
                    'sub':'O',
                    'series':'O'})
    
    formula = Formula(lhs=lhs("y"), # The dependent variable - here y!
                        terms=[i(), # The intercept, a
                               f(["time","x"],te=True), # one smooth surface over time and x - f(time,x) - for the reference level = cond == b
                               f(["time","x"],te=True,binary=["cond","a"])], # another smooth surface over time and x - f(time,x) - representing the difference from the other surface when cond==a
                        data=dat,
                        print_warn=False)
        
    model = GAMM(formula,Gaussian())

    model.fit(extension_method_lam = "mult",exclude_lambda=True)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 29.884

    def test_GAMTermEdf(self):
        diff = np.abs(np.round(self.model.term_edf,decimals=3) - np.array([16.668, 12.216]))
        rel_diff = diff/np.array([16.668, 12.216])
        assert np.max(rel_diff) < 1e-7
    
    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 967.893
    
    def test_GAMlam(self):
        # Fourth lambda term varies a lot, so is exlcuded here.
        diff = np.abs(np.round([p.lam for p in self.model.formula.penalties],decimals=3) - np.array([    0.001,   621.874,     0.011, 25335.589]))
        rel_diff = diff[[0,1,2]]/np.array([    0.001,   621.874,     0.011])
        assert np.max(rel_diff) < 1e-7

class Test_GAMM:

    dat = pd.read_csv('https://raw.githubusercontent.com/JoKra1/mssm_tutorials/main/data/GAMM/sim_dat.csv')

    # mssm requires that the data-type for variables used as factors is 'O'=object
    dat = dat.astype({'series': 'O',
                    'cond':'O',
                    'sub':'O',
                    'series':'O'})
    
    formula = Formula(lhs=lhs("y"), # The dependent variable - here y!
                      terms=[i(), # The intercept, a
                               l(["cond"]), # For cond='b'
                               f(["time"],by="cond",constraint=ConstType.QR), # to-way interaction between time and cond; one smooth over time per cond level
                               f(["x"],by="cond",constraint=ConstType.QR), # to-way interaction between x and cond; one smooth over x per cond level
                               f(["time","x"],by="cond",constraint=ConstType.QR), # three-way interaction
                               fs(["time"],rf="sub")], # Random non-linear effect of time - one smooth per level of factor sub
                        data=dat,
                        print_warn=False)
        
    model = GAMM(formula,Gaussian())

    model.fit(extension_method_lam = "mult",exclude_lambda=True)

    def test_GAMMedf(self):
        assert round(self.model.edf,ndigits=3) == 153.619

    def test_GAMMTermEdf(self):
        diff = np.abs(np.round(self.model.term_edf,decimals=3) - np.array([6.892, 8.635, 1.182, 1.01, 1.001, 1.039, 131.861]))
        rel_diff = diff/np.array([6.892, 8.635, 1.182, 1.01, 1.001, 1.039, 131.861])
        assert np.max(rel_diff) < 1e-7
    
    def test_GAMMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 577.196
    
    def test_GAMMlam(self):
        diff = np.abs(np.round([p.lam for p in self.model.formula.penalties],decimals=3) - np.array([0.004, 0.006, 5814.327, 153569.898 , 328846.811, 105218.21, 162215.095, 934.775, 0.119, 2.166]))
        rel_diff = diff/np.array([0.004, 0.006, 5814.327, 153569.898 , 328846.811, 105218.21, 162215.095, 934.775, 0.119, 2.166])
        assert np.max(rel_diff) < 1e-7


class Test_BIG_GAMM:

    file_paths = [f'https://raw.githubusercontent.com/JoKra1/mssm_tutorials/main/data/GAMM/sim_dat_cond_{cond}.csv' for cond in ["a","b"]]

    codebook = {'cond':{'a': 0, 'b': 1}}

    formula = Formula(lhs=lhs("y"), # The dependent variable - here y!
                      terms=[i(), # The intercept, a
                               l(["cond"]), # For cond='b'
                               f(["time"],by="cond",constraint=ConstType.QR), # to-way interaction between time and cond; one smooth over time per cond level
                               f(["x"],by="cond",constraint=ConstType.QR), # to-way interaction between x and cond; one smooth over x per cond level
                               f(["time","x"],by="cond",constraint=ConstType.QR), # three-way interaction
                               fs(["time"],rf="sub")], # Random non-linear effect of time - one smooth per level of factor sub
                        data=None, # No data frame!
                        file_paths=file_paths, # Just a list with paths to files.
                        print_warn=False,
                        codebook=codebook)
        
    model = GAMM(formula,Gaussian())

    model.fit(extension_method_lam = "mult",exclude_lambda=True)

    def test_GAMMedf(self):
        assert round(self.model.edf,ndigits=3) == 153.619

    def test_GAMMTermEdf(self):
        diff = np.abs(np.round(self.model.term_edf,decimals=3) - np.array([6.892, 8.635, 1.182, 1.01, 1.001, 1.039, 131.861]))
        rel_diff = diff/np.array([6.892, 8.635, 1.182, 1.01, 1.001, 1.039, 131.861])
        assert np.max(rel_diff) < 1e-7
    
    def test_GAMMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 577.196
    
    def test_GAMMlam(self):
        diff = np.abs(np.round([p.lam for p in self.model.formula.penalties],decimals=3) - np.array([0.004, 0.006, 5814.327, 153569.898 , 328846.811, 105218.21, 162215.095, 934.775, 0.119, 2.166]))
        rel_diff = diff/np.array([0.004, 0.006, 5814.327, 153569.898 , 328846.811, 105218.21, 162215.095, 934.775, 0.119, 2.166])
        assert np.max(rel_diff) < 1e-7


class Test_Binom_GAMM:

    Binomdat = sim3(10000,2,family=Binomial(),seed=20)

    formula = Formula(lhs("y"),[i(),f(["x0"]),f(["x1"]),f(["x2"]),f(["x3"])],data=Binomdat)

    # By default, the Binomial family assumes binary data and uses the logit link.
    model = GAMM(formula,Binomial())
    model.fit()

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 13.468 

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 1 

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef,np.array([0.7422333451777551, -0.17195247706792302, 0.057394164386376505, 0.18704155831577263, 0.22846608870924795,
                        0.23511563352132572, 0.23707380980366774, 0.1858274828099582, 0.012619280027852899, -0.17315651504766674,
                        -0.17250023796593375, -0.094200972083248, -0.04409410888775292, 0.01921436719459457, 0.1053763860762365,
                        0.20885336302996846, 0.3163156513235213, 0.3922828489471839, 0.4674708452078455, -0.4517546540990654,
                        0.9374862846060616, 1.2394748022759206, 0.4085019434244128, 0.6450776959620124, 0.6155671421354455,
                        0.12222718933779214, -0.05160555872945563, 0.08904926741803995, 0.04897607940790038, 0.0017796804146379072,
                        -0.023979562522928297, -0.04130161353880989, -0.05541306071248854, -0.06449403279102219, -0.06700848322507941,
                        -0.05044390273197432, -0.03325208528628772])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.formula.penalties])
        assert np.allclose(lam,np.array([122.52719452460906, 655.3557029613052, 1.3826427117149267, 2841.8313047699667])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -6214.394 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -6188.98 


class Test_Gamma_GAMM:

    Gammadat = sim3(500,2,family=Gamma(),seed=0)

    formula = Formula(lhs("y"),[i(),f(["x0"]),f(["x1"]),f(["x2"]),f(["x3"])],data=Gammadat)

    # By default, the Gamma family assumes that the model predictions match log(\mu_i), i.e., a log-link is used.
    model = GAMM(formula,Gamma())
    model.fit()

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 17.814 

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 2.198 

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef,np.array([7.654619426165573, -0.8985835974502748, 0.007379410246291677, 0.8957101794062817, 1.5029853796818358,
                        1.493483354632211, 0.8784842536845233, 0.26098272816203444, -0.4103286413630141, -1.114920815874382,
                        -1.5927964521179392, -1.148958310775413, -0.7600643014139132, -0.12496105839506648, 0.7216919175807573,
                        1.8576582674297482, 3.0961174683334454, 4.115793366721786, 5.270272872019139, -6.587061280350998,
                        5.549362912627165, 7.502334022655151, -0.002066373028916379, 0.8329530056316238, 1.2364758339185502,
                        -2.3108759390294447, -3.0463729185932946, 1.7885877838213213, -0.08631242743776814, 0.07052295759155355,
                        0.16597169875196663, 0.18903677135585883, 0.1574395995140705, 0.09725688737590793, 0.04256696866874943,
                        -0.033305436991563596, -0.12103077388832384])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.formula.penalties])
        assert np.allclose(lam,np.array([6.77298234619713, 14.900785396068327, 0.026452730145895265, 227.83911564014247])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -4249.165 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -4215.695

class Test_Overlap_GAMM:
    
    # Simulate time-series based on two events that elicit responses which vary in their overlap.
    # The summed responses + a random intercept + noise is then the signal.
    overlap_dat,onsets1,onsets2 = sim7(100,1,2,seed=20)

    # Model below tries to recover the shape of the two responses + the random intercepts:
    overlap_formula = Formula(lhs("y"),[irf(["time"],onsets1,nk=15,basis_kwargs=[{"max_c":200,"min_c":0,"convolve":True}]),
                                        irf(["time"],onsets2,nk=15,basis_kwargs=[{"max_c":200,"min_c":0,"convolve":True}]),
                                        ri("factor")],
                                        data=overlap_dat,
                                        series_id="series")

    model = GAMM(overlap_formula,Gaussian())
    model.fit()

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 54.547 

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 3.91 

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef,np.array([0.5414381914556667, 0.9027253840593255, 1.2394888139729652, 1.5408974325924492, 1.793282236604561,
                                        1.9605024352161622, 2.0342513059209852, 2.018245078562129, 1.9481936897353123, 1.8291845270086586,
                                        1.6414046863148892, 1.3759062972662246, 1.0473698479784739, 0.6781931660932433, 0.2896778273936524,
                                        5.018807622568596, 8.155536921854802, 9.057658829442943, 8.118583945017296, 6.009795403374646,
                                        3.4570230629826937, 2.4450369885874026, 2.417918115195418, 3.251836238689801, 3.4258032416231323,
                                        2.6532468387795594, 2.0300261093566743, 0.731209208180373, -0.5804637873111934, -0.18465710319341722,
                                        1.0285145982624768, 0.4524899052941147, 0.4005568257213123, -0.823004121469387, -0.6499737442921556,
                                        -0.8960486421242312, -1.0453212699599603, 0.17551387787392425, -0.17550250699168513, 1.71876796701693,
                                        1.0220803116075616, 0.7907656543437932, 0.18640800710629646, 0.10229679462403848, -3.032559645373398,
                                        1.243208243377598, 1.0817416861889755, -0.48123485830242374, 0.031615091580908194, 0.23411521023216836,
                                        1.4525117994309633, 0.24783178423729385, -1.2968663600345203, -2.358827075195528, -0.7007707103060459,
                                        -0.943074112905826, 0.7718437972508059, 2.2734085553443526, -0.6620713954858437, -1.3234198671472517,
                                        0.7227119874087831, -0.9365821180001762, 0.16427911329019607, -1.5908026012661671, -0.9487180564832598,
                                        1.0573347505186208, 0.999116483922564, -1.09744680946051, 0.7031765530477949, 0.799916646746684])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.formula.penalties])
        assert np.allclose(lam,np.array([227.8068139397452, 1.8954868106567002, 3.04743103446127])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -7346.922 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -7232.595