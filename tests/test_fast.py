from mssm.models import *
from mssm.src.python.compare import compare_CDL
import numpy as np
import os
from mssmViz.sim import*
from .defaults import default_gamm_test_kwargs

################################################################## Tests ##################################################################

class Test_GAM:

    dat = pd.read_csv('https://raw.githubusercontent.com/JoKra1/mssmViz/main/data/GAMM/sim_dat.csv')
    
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

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["exclude_lambda"] = True

    model.fit(**test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 9.723

    def test_GAMTermEdf(self):
        assert round(self.model.term_edf[0],ndigits=3) == 8.723
    
    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 1084.879
    
    def test_GAMlam(self):
        assert round(self.model.overall_penalties[0].lam,ndigits=5) == 0.0089

class Test_GAM_TE:

    dat = pd.read_csv('https://raw.githubusercontent.com/JoKra1/mssmViz/main/data/GAMM/sim_dat.csv')
    
    # mssm requires that the data-type for variables used as factors is 'O'=object
    dat = dat.astype({'series': 'O',
                    'cond':'O',
                    'sub':'O',
                    'series':'O'})
    
    formula = Formula(lhs=lhs("y"), # The dependent variable - here y!
                        terms=[i(), # The intercept, a
                               l(["cond"]), # Offset for cond='b'
                               f(["time","x"],by="cond",te=True,nk=9)], # one smooth surface over time and x - f(time,x) - per level of cond: three-way interaction!
                        data=dat,
                        print_warn=False)
        
    model = GAMM(formula,Gaussian())

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["exclude_lambda"] = True

    model.fit(**test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 33.835 

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 967.709 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -141942.109 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -141872.652 

class Test_GAM_TE_BINARY:

    dat = pd.read_csv('https://raw.githubusercontent.com/JoKra1/mssmViz/main/data/GAMM/sim_dat.csv')
    
    # mssm requires that the data-type for variables used as factors is 'O'=object
    dat = dat.astype({'series': 'O',
                    'cond':'O',
                    'sub':'O',
                    'series':'O'})
    
    formula = Formula(lhs=lhs("y"), # The dependent variable - here y!
                        terms=[i(), # The intercept, a
                               f(["time","x"],te=True,nk=9), # one smooth surface over time and x - f(time,x) - for the reference level = cond == b
                               f(["time","x"],te=True,binary=["cond","a"],nk=9)], # another smooth surface over time and x - f(time,x) - representing the difference from the other surface when cond==a
                        data=dat,
                        print_warn=False)
        
    model = GAMM(formula,Gaussian())

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["exclude_lambda"] = True

    model.fit(**test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 29.845 

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 967.895 

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef.flatten(),np.array([52.82251019977649, 443.11341640457823, 444.8417119315177, 442.81025980578295, 441.8124532480443,
                        449.15233147591147, 456.0654049813386, 459.6018424108562, 462.4183191702384, -161.15323911424358,
                        -159.623877790315, -168.32352305206663, -219.50255970350344, -262.8874671436067, -209.74096633357095,
                        -157.34589159088728, -144.89197515466134, -141.64907622342892, 83.02517725702134, 83.61685004722325,
                        67.29419223813039, -25.06059967446428, -118.81837516724774, -31.677977965846583, 69.46809675899978,
                        92.6800417952727, 95.52231215469003, -34.46440487835692, -31.56166663527097, -41.65630493489493,
                        -121.97632332516825, -225.6291011810352, -154.72590102494797, -45.0485734036091, -14.677745638601879,
                        -9.840527238633761, 6.044688514334028, 5.104454227519869, -2.10428200739398, -50.18964225788333,
                        -123.96887428889397, -93.77297203121753, -23.898920403710566, -3.7663027508644435, -3.8432183042729093,
                        -93.29184393677372, -83.9169534222666, -75.82587234162825, -77.93818561644619, -90.79896242796625,
                        -76.42106235813338, -46.018204981476124, -29.699889228791616, -20.29203443093002, 25.480871160132605,
                        17.654931518571917, 9.661947466639308, 0.22350596663755762, -11.263452647618768, -18.84680891692399,
                        -22.706198798189927, -28.53557459722842, -35.804864747296804, 5.598623232539643, 16.914075305821147,
                        28.206633986243915, 39.35513288626826, 50.24573721343646, 61.59929302230163, 73.43778231255267,
                        85.05273662653907, 96.48241145293561, 365.7053005116908, 358.8991736633464, 352.09177909576186,
                        345.2766885459626, 338.433822164632, 331.60692609167864, 324.81094672983323, 318.00028519671633,
                        311.17714345679923, -34.38833096738923, -36.33273007631386, -38.27712953648592, -40.221533080169614,
                        -42.1659191744123, -44.11022727529732, -46.05450647906014, -47.99877694106652, -49.94304653932069,
                        -13.31962844374681, -13.900096823977188, -14.480560200724193, -15.061003287394056, -15.64092331317976,
                        -16.21949553729438, -16.79771184987033, -17.37592499463898, -17.954132345740973, 8.807826257948406,
                        8.630021087606462, 8.452202249562353, 8.27427488588766, 8.096907895156507, 7.9221186630858185,
                        7.7483570641671164, 7.574678869087, 7.401001416804583, -35.17196488842441, -34.93795813475585,
                        -34.70392260717207, -34.46992789149443, -34.236595815870864, -34.00306974349889, -33.767762565123626,
                        -33.531746609362074, -33.29568990896897, -37.50697835944442, -36.12499118124349, -34.74300148421378,
                        -33.36104240876872, -31.97930816076038, -30.597381780141696, -29.21455652105496, -27.831493374687216,
                        -26.448471664415372, 20.979119561997774, 16.330339274524295, 11.681593828356748, 7.032945462204448,
                        2.3846307938606297, -2.2626581206506287, -6.909130891082743, -11.5553914097942, -16.201600856273288,
                        312.33467407725584, 311.98661581479803, 311.63853753530424, 311.2903985817903, 310.9421588395778,
                        310.5938160460782, 310.24536220243675, 309.8968289561669, 309.54826863428946, 602.3304012658087,
                        606.2696148550797, 610.208828559826, 614.1480426121827, 618.0872573471594, 622.0264729693242,
                        625.9656893065949, 629.9049060137926, 633.8441228444658, 892.3257551261239, 900.5524841180436,
                        908.779213109995, 917.0059421020208, 925.2326710941554, 933.4594000864129, 941.6861290787813,
                        949.9128580712264, 958.1395870637045])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -141930.049 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -141877.445 

class Test_GAMM:

    dat = pd.read_csv('https://raw.githubusercontent.com/JoKra1/mssmViz/main/data/GAMM/sim_dat.csv')

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
                               f(["time","x"],by="cond",constraint=ConstType.QR,nk=9), # three-way interaction
                               fs(["time"],rf="sub")], # Random non-linear effect of time - one smooth per level of factor sub
                        data=dat,
                        print_warn=False,find_nested=False)
        
    model = GAMM(formula,Gaussian())

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["exclude_lambda"] = True

    model.fit(**test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 153.707 

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 577.194 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(lam,np.array([0.003576343523507944, 0.006011901683446546, 5028.094352541816, 230482.43896738067, 110804.13553081625, 38451.59746745403, 381047.3436998889, 330.25972969656937, 0.11887201661809699, 2.166381231169934])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -134726.053 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -134263.723 


class Test_Binom_GAMM:

    Binomdat = sim3(10000,0.1,family=Binomial(),seed=20)

    formula = Formula(lhs("y"),[i(),f(["x0"]),f(["x1"]),f(["x2"]),f(["x3"])],data=Binomdat)

    # By default, the Binomial family assumes binary data and uses the logit link.
    model = GAMM(formula,Binomial())
    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["max_inner"] = 1
    test_kwargs["control_lambda"] = 2

    model.fit(**test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 13.468 

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 1 

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef.flatten(),np.array([0.7422333451777551, -0.17195247706792302, 0.057394164386376505, 0.18704155831577263, 0.22846608870924795,
                        0.23511563352132572, 0.23707380980366774, 0.1858274828099582, 0.012619280027852899, -0.17315651504766674,
                        -0.17250023796593375, -0.094200972083248, -0.04409410888775292, 0.01921436719459457, 0.1053763860762365,
                        0.20885336302996846, 0.3163156513235213, 0.3922828489471839, 0.4674708452078455, -0.4517546540990654,
                        0.9374862846060616, 1.2394748022759206, 0.4085019434244128, 0.6450776959620124, 0.6155671421354455,
                        0.12222718933779214, -0.05160555872945563, 0.08904926741803995, 0.04897607940790038, 0.0017796804146379072,
                        -0.023979562522928297, -0.04130161353880989, -0.05541306071248854, -0.06449403279102219, -0.06700848322507941,
                        -0.05044390273197432, -0.03325208528628772])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
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

    test_kwargs = copy.deepcopy(default_gamm_test_kwargs)
    test_kwargs["max_inner"] = 1

    model.fit(**test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 17.814 

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 2.198 

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef.flatten(),np.array([7.654619964489497, -0.8985855581443037, 0.00737676745266728, 0.8957095904668417, 1.5029863477484788,
                        1.4934830568869881, 0.8784847301646945, 0.2609849004219141, -0.4103254152440804, -1.1149172513050318,
                        -1.5927978049483915, -1.1489580382151177, -0.7600627541048914, -0.12495962121796242, 0.7216923726284495,
                        1.8576586195422136, 3.0961194280261264, 4.115797342436188, 5.270279060056066, -6.587035189026303,
                        5.549379799722564, 7.502355381272832, -0.0020521313276835014, 0.8329746706816913, 1.236491327346999,
                        -2.310855232645864, -3.046358383920367, 1.7885515697103103, -0.08630704882506356, 0.070517783667561,
                        0.16596005609696318, 0.18902432159809482, 0.157430338870762, 0.09725214332690277, 0.0425648394552737,
                        -0.03330475732996358, -0.12102663295792625])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(lam,np.array([6.772970379249816, 14.900828781781744, 0.026452879018106484, 227.86741305016199])) 

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
    model.fit(**default_gamm_test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 54.547 

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 3.91 

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef.flatten(),np.array([0.5414381914556667, 0.9027253840593255, 1.2394888139729652, 1.5408974325924492, 1.793282236604561,
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
        lam = np.array([p.lam for p in self.model.overall_penalties])
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
    model.fit(**default_gamm_test_kwargs)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 27.458 

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 11.727 

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef.flatten(),np.array([4.4496869070736516, 1.5961324271808035, 6.4361364571346265, -3.0591651115769993, 0.07948249536871332,
                                          -0.11501386584710335, 0.6505400484007926, 0.7096563086326143, -0.7142393186240358, 0.5593738772412031,
                                          -0.7133681425701058, 0.00984614953239883, 0.516050065109191, 1.038654222613382, 0.920317555859632,
                                          1.6827256736523348, -0.5615052428662065, -0.36418148548156637, -2.0302380548653485, 0.8057098517098714,
                                          1.3121331440612154, -0.7699578556178519, 0.09798781448952851, -0.8558991725303929, 1.1069346027406766,
                                          -0.3556611557220524, -2.35255431320692, -0.9234657845204244, -0.2705203747460906, -0.6018802984689225,
                                          0.41568172492693195, 1.8200370847180312, -1.070922758355525, -0.8231396771947899, 0.6529342195558038,
                                          -0.32948639417321524, 1.3361213113054577, -0.7414321131404542, -0.7531221916054935, 0.0800941666704564,
                                          1.3002335376265413, -0.7276918391919509, 0.2894890343822381, -0.3097228498687097])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.overall_penalties])
        assert np.allclose(lam,np.array([7.880949768403679])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -1340.036 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -1311.209 