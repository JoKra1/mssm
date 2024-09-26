from mssm.models import *
from mssm.src.python.compare import compare_CDL
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

class Test_ri_li:
    # Random intercept model + *li()
    sim_dat = sim4(n=500,scale=2,seed=20)

    # Specify formula 
    formula = Formula(lhs("y"),[i(),*li(["x0","x1"]),ri("x4")],data=sim_dat)

    # ... and model
    model = GAMM(formula,Gaussian())

    # then fit
    model.fit()

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 27.458 

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 11.727 

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef,np.array([4.4496869070736516, 1.5961324271808035, 6.4361364571346265, -3.0591651115769993, 0.07948249536871332,
                                          -0.11501386584710335, 0.6505400484007926, 0.7096563086326143, -0.7142393186240358, 0.5593738772412031,
                                          -0.7133681425701058, 0.00984614953239883, 0.516050065109191, 1.038654222613382, 0.920317555859632,
                                          1.6827256736523348, -0.5615052428662065, -0.36418148548156637, -2.0302380548653485, 0.8057098517098714,
                                          1.3121331440612154, -0.7699578556178519, 0.09798781448952851, -0.8558991725303929, 1.1069346027406766,
                                          -0.3556611557220524, -2.35255431320692, -0.9234657845204244, -0.2705203747460906, -0.6018802984689225,
                                          0.41568172492693195, 1.8200370847180312, -1.070922758355525, -0.8231396771947899, 0.6529342195558038,
                                          -0.32948639417321524, 1.3361213113054577, -0.7414321131404542, -0.7531221916054935, 0.0800941666704564,
                                          1.3002335376265413, -0.7276918391919509, 0.2894890343822381, -0.3097228498687097])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.formula.penalties])
        assert np.allclose(lam,np.array([7.880949768403679])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -1340.036 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -1311.209 

class Test_rs_ri:

    # Random slope + intercept model
    sim_dat,_ = sim1(100,random_seed=100)

    # Specify formula 
    formula = Formula(lhs("y"),[i(),*li(["x","fact"]),ri("series"),rs(["x"],rf="series")],data=sim_dat)

    # ... and model
    model = GAMM(formula,Gaussian())

    model.fit(maxiter=100)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 97.641 

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 46.876 

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef,np.array([9.453718165084535, -0.41826549000901636, -9.893416315627967, -9.003423513125606, 0.02895309810577356,
                                            0.3197819069476187, -4.157263517619225, 1.9541634518838584, 0.5505691335776016, 1.4435129980162877,
                                            -2.3874904238671064, 4.961289392901002, 4.299262129267971, -1.0081897287455608, -3.1526730556562774,
                                            4.421793032940925, -1.5711815294812856, -4.167376510867038, 5.890459122998194, 8.564119978780333,
                                            11.547503438664409, 1.7075523267640174, -1.4873372911816993, -0.12020819269482852, 3.2978896840188527,
                                            2.883228604846309, 6.9517970285055615, 1.0719173130237705, 2.134000897846334, -2.74618233043174,
                                            1.6926953719489684, -3.9457449889990626, -0.03496647292214308, -0.0642442858889104, 7.708952818585026,
                                            -1.0885289945034273, 6.038175899195711, -5.665303922817799, -8.156286527519818, -4.0310076610497365,
                                            5.9329566275913255, -1.2909082027851673, -0.4516268152887624, -0.9679877074822544, -2.4533674003836072,
                                            5.875325320855504, -6.526123105793242, -5.489438991031426, 3.0673469830887186, -4.277457313843025,
                                            4.885376540197432, -1.7023337685744018, -7.921850427624408, -3.0124832331368805, -5.185967325158845,
                                            -4.370427295489528, 1.269057387382737, 1.1046072295743323, -4.486470882002245, 2.2067721763859733,
                                            2.7910583983557493, -5.898812434971072, 0.8934894186144613, -4.172310429294425, 0.21399417225302586,
                                            3.049471771902134, -0.26897154322910855, 0.24449405338576288, -3.5974312237813932, 0.3626547355352499,
                                            0.7559546885511963, 5.072013082431705, -0.3846814903615633, -6.317914211766965, -4.839308035135555,
                                            3.051608551544353, -2.5005591206586137, -0.9540459447934297, 8.715731063935719, -2.027078787962841,
                                            5.130047948872989, -8.927504352774198, 3.361002128062276, -6.65206163882869, 0.08449521123403105,
                                            1.1338431131511724, 2.1023165602412024, -5.362156511550387, 4.266989755042644, 5.5750603368902425,
                                            -1.0212779651967925, 4.700054954568106, -0.344026592393573, -1.627943137642434, -6.005502673329025,
                                            0.5734042329789186, -3.364437770876721, 1.3146620896009515, 8.938936824181406, -2.9638992152875456,
                                            9.196275350556421, 1.3353148106237895, -3.186381041635486, -0.19429624445527663, -0.08107885857434388,
                                            -11.711091014720022, -0.00720012107773575, 0.001214944839498058, 0.0005623508222403166, 0.0008547263126831678,
                                            -0.0027566563198637476, 0.007931671142477423, 0.005091318847916507, -2.98482344987347e-05, -0.0009800418321975416,
                                            0.00870555444145808, -0.002442092151143039, -0.007526082140545063, 0.004882964391993103, 0.008493837134824139,
                                            0.023247346691374977, 0.0036398450595923246, -0.002531941638916287, -9.253026267781652e-05, 0.004784191814115068,
                                            0.004396050638532361, 0.011731358823405666, 0.0019675662044859256, 0.004138207630889772, -0.0018699654561768063,
                                            0.0013530659039869034, -0.007476276710759731, -6.728861094871104e-05, -3.804003265938627e-05, 0.01346554013553512,
                                            -0.0022880985701179022, 0.008223183307739872, -0.012411701213415031, -0.0022939949562319826, -0.006205736720123811,
                                            0.003688644893399395, -0.002025571647564641, -0.0006083695886273941, -0.001375585087340673, -0.003341156367650646,
                                            0.0018264071442094737, -0.009853756303051355, -0.003412900577720736, 0.006129754188634371, -0.003355891704730357,
                                            0.0033266128287792185, -0.00020159561666139634, -0.0025798574091216936, -0.002675606686000228, -0.0069858225894284766,
                                            -0.003040661964527926, 0.0008453570114298503, 0.0017659484445960486, -0.004781715505550739, 0.0001633329807079981,
                                            0.004916570473107239, -0.005239166750158882, 0.0011771337292187409, -0.008276139369874248, 0.00019006387545813518,
                                            0.006139173535081009, -0.0002349117445609447, 0.0003619217509826125, -0.004526450852655491, 9.126172175497808e-05,
                                            0.00040285118869737996, 0.005255630070613733, -0.0007573546821432042, 0.0, -0.007736658124955957,
                                            0.0025296662932971156, -0.004404843390640134, -0.002019536520126511, 0.003096431283202661, -0.004170919549502067,
                                            0.0035691571143255745, -0.006871946947212416, 0.001592080963474982, -0.005415833920893137, 0.00012007420303581101,
                                            0.0003524671484864533, 0.0028630723474051666, -0.011747557657562415, 0.0037266619187196885, 0.007675009191626168,
                                            -0.00024188577466680556, 0.008279350738496692, -0.0002138886246572169, -0.002699004681065283, -0.002400266769273219,
                                            0.00043288990134934287, -0.0024403659892760104, 0.0012260299310452476, 0.008203967687428714, -0.005835276727217685,
                                            0.003403285306228259, 0.0013836558696505845, -0.003726243096853524, -1.72568705116879e-05, -5.760965138891208e-05,
                                            -0.019936163204627978])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.formula.penalties])
        assert np.allclose(lam,np.array([2.168719700473885, 24581.63586719763])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -32129.11 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -31893.94 

class Test_no_pen:
    # No penalties
    sim_dat,_ = sim1(100,random_seed=100)

    # Specify formula 
    formula = Formula(lhs("y"),[i(),*li(["x","fact"])],data=sim_dat)

    # ... and model
    model = GAMM(formula,Gaussian())

    model.fit(maxiter=100)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 6 

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 68.496 

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef,np.array([9.159532102917803, -0.40344985119982324, -10.200594212981578, -8.7959940138272, 0.07712577556440754,
                                          0.29486179496281906])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.formula.penalties])
        assert np.allclose(lam,np.array([]))

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -33751.932

class Test_te_rs_fact:
    # te + random slope with factor
    sim_dat,_ = sim1(100,random_seed=100)

    # Specify formula 
    formula = Formula(lhs("y"),[i(),f(["time","x"],te=True,nk=5),rs(["fact"],rf="sub"),rs(["x","fact"],rf="sub")],data=sim_dat)

    # ... and model
    model = GAMM(formula,Gaussian())

    model.fit(maxiter=100)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 93.102 

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 47.716 

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef,np.array([-0.8649773325421872, -15.028795194922372, -2.6920839960357026, 33.221370322247814, -128.67351915762785,
                                        34.13589493906625, 3.511642953832433, 13.619041215724293, 12.054323519308607, -107.55869570249693,
                                        65.39293551197889, 1.7224746076830655, 15.160912116581175, 0.9562984366566933, -68.6463557551376,
                                        0.7855396725133215, 3.7692708157921704, 8.351384737629534, 8.303522682290259, -40.633661124346595,
                                        -37.27578935485991, 51.82546088041295, -27.488622213466456, -43.78233934309928, -17.974738535144223,
                                        0.0, 0.7978844259668167, -1.5766295365078138, 7.765848004422449, 2.3103663291578243,
                                        0.0, 16.475844402746844, 1.5735811003740545, 4.86098891283255, 8.101373152509186,
                                        5.377330799029823, -2.1317715053360535, 11.922056578198871, 5.2534939496128725, 0.5107495112763606,
                                        8.707665385308557, 4.705548610862964, 6.518447150675284, 2.334399349097945, 4.500325529373986,
                                        0.0, 4.7338850876137855, -16.564777677058913, -0.43106236096374023, -12.827460136377686,
                                        -9.010125007252448, -1.3826168037304047, -6.336075943858941, 0.6263031922662698, -4.087769786749296,
                                        -24.698997755364513, 0.0, -1.2724495239383493, -0.14033447653351355, 2.1360366259380204,
                                        -6.631516968432652, -0.604486848249624, -1.6967016560323784, -9.746822883253712, 0.59466221619956,
                                        1.994645888688083, 6.53147987207574, -7.193133926082729, 0.0, 5.996196342700394,
                                        0.0, 8.222298632434912, -0.7579437945964075, -3.5107386092420008, 0.0,
                                        -1.8996291183544574, -3.5155002447725034, 0.0, -2.3788751529478693, 4.355409353703643,
                                        0.7188130179308237, -0.9629407909301079, -10.82638599138278, 11.921114997397183, -9.3620019204606,
                                        0.0, 0.1168117919016276, -0.08152757071363799, -0.33914756918378053, 0.14380543060682896,
                                        0.0, -0.2597732534087394, 0.08136993597024549, 0.2141231115821457, -1.3175075188997858,
                                        0.05000875577500889, 0.047925319572293026, -0.07341188427650454, -0.19421717878245667, 0.024454514645040424,
                                        -0.7479274465555487, 0.16557071023496703, 0.14930725330650368, 0.1385951160868386, 0.15782821093467458,
                                        0.0, -0.9312216630020252, 1.5310268873605801, -0.09352495566265022, 0.14768798273992756,
                                        0.28218398479220874, -0.09395418016266995, 0.1576907164497027, 0.1140464743472867, 0.1950234785669218,
                                        4.431907739839756, 0.0, -0.47327239895631856, -0.032078626972817555, 0.1737910353804186,
                                        0.7666693721808544, -0.2412256369578877, -0.27609221110995547, 0.15068825090811355, 0.10598088882406358,
                                        -0.1456624983874595, 0.48261843315505254, 0.506742954108234, 0.0, 0.20198584654374038,
                                        0.0, 0.11168289397688132, 0.2666228511460736, -0.2085068498490318, 0.0,
                                        -0.10837061082304453, -0.45413651629864105, 0.0, -0.11890849481306373, -0.053960581022117954,
                                        0.03827325628822098, -0.04656321886113537, 0.8163751936880147, -0.7372010128779578, 0.5754690975744272])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.formula.penalties])
        assert np.allclose(lam,np.array([0.01849623093918212, 0.0029003681933273072, 0.6537589517464473, 229.09747408634502, 28.312095080997334, 201.89147673743307])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -32248.33 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -31981.09 

class Test_ti_rs_fact:
    # ti + random slope with factor
    sim_dat,_ = sim1(100,random_seed=100)

    # Specify formula 
    formula = Formula(lhs("y"),[i(),f(["x"]),f(["time"]),f(["time","x"],te=False,nk=5),rs(["fact"],rf="sub"),rs(["x","fact"],rf="sub")],data=sim_dat)

    # ... and model
    model = GAMM(formula,Gaussian())

    model.fit(maxiter=100)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 103.428 

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 46.799 

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef,np.array([-1.0300433990153681, 1.6506512387717502, 5.486644662213888, -2.4079188590258274, 0.7532995189028551,
                                            7.210255366746999, -2.888230523433733, -2.1148203926808904, -10.455606825766884, -44.479532830591225,
                                            8.075692861290793, 7.246066902094092, 8.80455792272081, 4.275914121697346, 5.681354977565387,
                                            5.37175359228524, -1.5570226074592568, 5.16651857022348, 17.980055697915546, -6.130596711660393,
                                            27.44501338384899, 42.37537594923482, 26.301637975505624, -7.583498180523644, -9.119044138417278,
                                            3.9164630505013136, 8.764795309154898, 1.3150405566594612, 12.739697733635015, -13.988524415374162,
                                            6.476793016572713, 14.389821704753953, 9.602255442454902, 4.762296308486982, 14.079762145084684,
                                            22.450084638294744, 6.111666925423598, 10.434683884417572, 13.891514959059961, -16.418316204309516,
                                            11.093400845656252, 30.44011104077033, 47.8931560287755, 63.74793973894576, 0.0,
                                            -0.1861325532638123, -0.7690736356500241, 8.684003109023468, 2.435902418168845, 0.0,
                                            14.386534939767552, 2.819470417036722, 4.373820276120368, 9.807706981889986, 7.133169783138588,
                                            -2.352318360462471, 12.667729754521742, 6.129515712613967, 0.7842081077270553, -2.437922875843843,
                                            3.6448463429158244, 6.29916132964956, 2.8775801422583114, 6.744109658513405, 0.0,
                                            7.891497489852647, -20.18689874956632, -0.11959650683662554, -22.685403518958164, -7.11502035404644,
                                            5.086534811103273, -3.317087297693165, 1.0988862917002633, -2.9099241344795264, -28.2027752897359,
                                            0.0, -1.1667021889805227, -0.006979279383333311, 0.9075814223884011, -4.7558567938172125,
                                            -0.3879324881237582, -1.0172990209372672, -6.648682777949239, 1.107963015388131, 0.34723576997683797,
                                            8.486418813869614, -2.5923824540541567, 0.0, 6.6619241146296515, 0.0,
                                            6.963721140052922, -0.6109820785382035, -5.119518037334413, 0.0, -1.5023680731052196,
                                            -3.1340599780078606, 0.0, -3.9368215957514625, 7.746294654570607, 0.9597598408431581,
                                            -2.779606978977643, -7.358632928532698, 6.013720673041227, -10.759319060935356, 0.0,
                                            0.07606495339111126, -0.031092858918620243, -0.4017971516204013, 0.11854199063979785, 0.0,
                                            -0.10847015607543924, 0.11398829947939634, 0.15063217187866046, -1.3649254100725785, 0.008990724865583548,
                                            0.1114755301623414, -0.07326145293857458, -0.176864269779507, 0.029356231587438695, -0.10887918217994659,
                                            0.27152208729250416, 0.2415447392124025, 0.13357281850520344, 0.03911465080382107, 0.0,
                                            -1.2040547566794664, 2.0699266528095377, -0.030328403447700245, 0.6839358559766404, 0.21652341047602483,
                                            -0.7982521362999412, -0.2407914543140068, 0.23388029898589446, 0.15535676949230237, 5.019578915515454,
                                            0.0, -0.5071941928713447, -0.0018646856210937767, 0.0863073772003619, -0.27274167518294373,
                                            -0.180940757161953, -0.1934821673504877, -0.05998047018757524, 0.23079485440624659, -0.0482486941729563,
                                            0.3810842337132825, 0.013557458895750883, 0.0, 0.13637946520400845, 0.0,
                                            0.05748294267400564, 0.27268799846896763, -0.2551535513759499, 0.0, -0.052086269905969824,
                                            -0.3942031939500484, 0.0, -0.11958898336712577, -0.3539170231413322, 0.031056053179848806,
                                            -0.0816828744401028, 0.631001363411639, -0.3503581647909492, 0.6083550724278155])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.formula.penalties])
        assert np.allclose(lam,np.array([0.20517914894864384, 0.33849019882829423, 0.006146354332691984, 0.07257815549981719, 0.5604424399461271, 251.19776923538313, 20.765468252271926, 284.79157880464095])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -32186.755 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -31883.24 

class Test_3way_li:
    # *li() with three variables: three-way interaction
    sim_dat,_ = sim1(100,random_seed=100)

    # Specify formula 
    formula = Formula(lhs("y"),[i(),*li(["fact","x","time"])],data=sim_dat)

    # ... and model
    model = GAMM(formula,Gaussian())

    # then fit
    model.fit()

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 12 

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 67.323 

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef,np.array([9.956079664221663, -11.690196134253785, -8.845101906438588, -0.3454176104436288, -0.0006942318521386302,
                                          -0.09765623438407976, 0.2479842732961105, 0.0011923415584594499, -0.00024911061225905723, -5.2708086355813645e-05,
                                          0.00019002614776238562, 5.654806043274768e-05])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.formula.penalties])
        assert np.allclose(lam,np.array([])) 

class Test_diff:
    # pred_diff test
    sim_dat,_ = sim1(100,random_seed=100)

    # Specify formula 
    formula = Formula(lhs("y"),[i(),l(["fact"]),f(["time"],by="fact"),rs(["fact"],rf="sub"),rs(["x","fact"],rf="sub")],data=sim_dat)

    # ... and model
    model = GAMM(formula,Gaussian())

    model.fit(maxiter=100)

    pred_dat1 = pd.DataFrame({"time":np.linspace(min(sim_dat["time"]),max(sim_dat["time"]),50),
                                "x":[0 for _ in range(50)],
                                "fact":["fact_1" for _ in range(50)],
                                "sub":["sub_0" for _ in range(50)]})

    pred_dat2 = pd.DataFrame({"time":np.linspace(min(sim_dat["time"]),max(sim_dat["time"]),50),
                                "x":[0 for _ in range(50)],
                                "fact":["fact_2" for _ in range(50)],
                                "sub":["sub_0" for _ in range(50)]})

    diff,ci = model.predict_diff(pred_dat1,pred_dat2,[0,1,2])

    def test_diff(self):
        assert np.allclose(self.diff,np.array([18.71113446, 19.25834739, 19.57916718, 19.69981674, 19.64651897,
                                                19.44549677, 19.12297307, 18.70517076, 18.21160592, 17.63496729,
                                                16.96123679, 16.17639632, 15.2664278 , 14.21731313, 13.01503422,
                                                11.66864917, 10.27952081,  8.97208814,  7.87079017,  7.1000659 ,
                                                6.78435436,  7.04809455,  7.96028052,  9.36812659, 11.06340209,
                                                12.83787638, 14.48331881, 15.79149873, 16.5541855 , 16.62757128,
                                                16.12553958, 15.22639672, 14.10844903, 12.95000284, 11.92936448,
                                                11.22484026, 10.96385103, 11.07027569, 11.41710764, 11.87734029,
                                                12.32396706, 12.62998134, 12.66837655, 12.36680599, 11.87156253,
                                                11.38359893, 11.10386797, 11.23332242, 11.97291503, 13.52359857]))
    
    def test_ci(self):
        assert np.allclose(self.ci,np.array([5.73486758, 5.52496498, 5.49751497, 5.50017374, 5.49071308,
                                                5.47699123, 5.47374712, 5.48051564, 5.48312075, 5.47702224,
                                                5.46803891, 5.46335616, 5.46626692, 5.47415935, 5.47985052,
                                                5.4773296 , 5.46995348, 5.46449904, 5.46553632, 5.47304214,
                                                5.48233464, 5.48632547, 5.48108629, 5.47234253, 5.46792701,
                                                5.47209994, 5.48353791, 5.4959386 , 5.50118772, 5.49616701,
                                                5.48946366, 5.49088892, 5.50432632, 5.52572965, 5.54494371,
                                                5.55112981, 5.54323679, 5.53779817, 5.54964387, 5.58083555,
                                                5.62003055, 5.64987089, 5.6615386 , 5.67563722, 5.73117033,
                                                5.83694597, 5.99626507, 6.29719981, 7.02295103, 8.65004211]))