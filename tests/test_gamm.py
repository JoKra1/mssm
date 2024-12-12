from mssm.models import *
from mssm.src.python.compare import compare_CDL
import numpy as np
import os
from mssmViz.sim import*
from mssm.src.python.formula import reparam
from mssm.src.python.gamm_solvers import compute_S_emb_pinv_det,cpp_chol,cpp_cholP,compute_eigen_perm,compute_Linv
from mssm.src.python.utils import estimateVp
import io
from contextlib import redirect_stdout

class Test_BIG_GAMM_Discretize_hard:
    dat = pd.read_csv('https://raw.githubusercontent.com/JoKra1/mssm_tutorials/main/data/GAMM/sim_dat.csv')

    # mssm requires that the data-type for variables used as factors is 'O'=object
    dat = dat.astype({'series': 'O',
                    'cond':'O',
                    'sub':'O',
                    'series':'O'})

    discretize = {"no_disc":[],"excl":[],"split_by":["cond"],"restarts":40,"seed":20}

    formula = Formula(lhs=lhs("y"), # The dependent variable - here y!
                            terms=[i(), # The intercept, a
                                l(["cond"]), # For cond='b'
                                f(["time"],by="cond",nk=20), # to-way interaction between time and cond; one smooth over time per cond level
                                f(["x"],by="cond"), # to-way interaction between x and cond; one smooth over x per cond level
                                f(["time","x"],by="cond"), # three-way interaction
                                fs(["time"],rf="series",nk=20,approx_deriv=discretize)], # Random non-linear effect of time - one smooth per level of factor series
                            data=dat,
                            series_id="series") # When approximating the computations for a random smooth, the series identifier column needs to be specified!
        
    model = GAMM(formula,Gaussian())

    model.fit()

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=0) == 2434

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 10.969 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=2) == -84025.32

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=0) == -75228


class Test_NUll_penalty_reparam_hard:
    dat = pd.read_csv('https://raw.githubusercontent.com/JoKra1/mssm_tutorials/main/data/GAMM/sim_dat.csv')

    # mssm requires that the data-type for variables used as factors is 'O'=object
    dat = dat.astype({'series': 'O',
                    'cond':'O',
                    'sub':'O',
                    'series':'O'})

    # Add Null-penalties to univariate by-term and tensor by-term
    formula = Formula(lhs=lhs("y"), # The dependent variable - here y!
                        terms=[i(), # The intercept, a
                                l(["cond"]), # For cond='b'
                                f(["time"],by="cond",constraint=ConstType.QR,penalize_null=True), # to-way interaction between time and cond; one smooth over time per cond level
                                f(["x"],by="cond",constraint=ConstType.QR,penalize_null=False), # to-way interaction between x and cond; one smooth over x per cond level
                                f(["time","x"],by="cond",constraint=ConstType.QR,penalize_null=True), # three-way interaction
                                fs(["time"],rf="sub")], # Random non-linear effect of time - one smooth per level of factor sub
                        data=dat,
                        print_warn=False)
        
    model = GAMM(formula,Gaussian())

    model.fit()

    #Compute re-parameterization strategy from Wood (2011)
    S_emb,S_pinv,_,_ = compute_S_emb_pinv_det(len(model.coef),formula.penalties,"svd")
    Sj_reps,S_reps,SJ_term_idx,S_idx,S_coefs,Q_reps,_,Mp = reparam(None,formula.penalties,None,option=4)

    # For Computing derivative of log(|S_{\lambda}|) with respect to \lambda_j of univariate smooth term (Wood, 2011)
    S_rep = S_reps[0]

    L,code = cpp_chol(S_rep)
    Linv = compute_Linv(L)
    S_inv = Linv.T@Linv

    # And the same for tensor term
    S_rep2 = S_reps[4]

    L2,code2 = cpp_chol(S_rep2)
    Linv2 = compute_Linv(L2)
    S_inv2 = Linv2.T@Linv2

    def test_reparam_1(self):
        # Transformation strategy from Wood (2011) &  Wood, Li, Shaddick, & Augustin (2017)
        assert np.allclose((self.S_inv@self.Sj_reps[0].S_J).trace(),
                            self.Sj_reps[0].rank/self.Sj_reps[0].lam)
    
    def test_reparam2(self):
        # General strategy, e.g. from Wood & Fasiolo, 2017
        assert np.allclose((self.S_inv@self.Sj_reps[0].S_J).trace(),
                           (self.S_pinv@self.formula.penalties[0].S_J_emb).trace())
        
    def test_reparam3(self):
        # General strategy (here for tensor), e.g. from Wood & Fasiolo, 2017
        assert np.allclose((self.S_inv2@self.Sj_reps[6].S_J).trace(),
                           (self.S_pinv@self.formula.penalties[6].S_J_emb).trace())
    
    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=2) == 151.46 

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 577.199 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -134748.718 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=1) == -134265.0


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

    model.fit(exclude_lambda=True)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 153.707 

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 577.194

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.formula.penalties])
        assert np.allclose(lam,np.array([0.003576343523516708, 0.006011901683452655, 5028.094352875556, 230482.43912034066, 110804.13545750394, 38451.597466911124, 381047.3435206221, 330.2597296955685, 0.11887201661781975, 2.166381231196006])) 


class Test_BIG_GAMM_keep_cov_hard:
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
                        keep_cov=True, # Keep encoded data structure in memory
                        codebook=codebook)
        
    model = GAMM(formula,Gaussian())

    model.fit()

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=1) == 153.7

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 577.194

class Test_rs_ri_hard:

    # Random slope + intercept model
    sim_dat,_ = sim1(100,random_seed=100)

    # Specify formula 
    formula = Formula(lhs("y"),[i(),*li(["x","fact"]),ri("series"),rs(["x"],rf="series")],data=sim_dat)

    # ... and model
    model = GAMM(formula,Gaussian())

    model.fit(maxiter=100)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 97.613 

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 46.684 

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef,np.array([9.292102142920502, -0.40892384450483493, -9.786309327768105, -8.67262117407748, 0.024143996186701936,
                        0.3009527570662818, -3.9300344773787748, 1.968629948257126, 0.6142552685888969, 1.22740830612768,
                        -2.6443197809718577, 4.82314248611761, 3.8365446953118445, -0.9325024654333054, -3.506119267291391,
                        4.310431890080565, -1.07373680673549, -4.1688295595172296, 5.8366625569045025, 8.269576742221073,
                        11.66592070247694, 1.4049420384012947, -1.5145449538262596, -0.7282078462276366, 3.7648064878326304,
                        2.3846108393024124, 7.183293665336137, 1.2317856730307786, 2.291019071949271, -2.6536615102474044,
                        1.562431058032735, -4.22350800210904, -0.07610238963111311, -0.12839437551921315, 7.7451056764315895,
                        -0.9932207002957826, 5.8028693033164185, -5.222025745926126, -8.149656664522322, -3.962544463843295,
                        5.806396727329483, -1.0692822297255062, -0.5644371240188973, -0.5944977885122457, -2.5072675280998418,
                        6.089021045235532, -5.994910098683583, -5.8296136673612216, 2.8952277560088726, -4.666712648209863,
                        5.26709384079597, -2.314444608818328, -7.467895065131839, -2.9898032460556125, -5.233368438847724,
                        -4.356649745504148, 0.8367821960980883, 1.5533203477845428, -4.5770883567255565, 2.3859147424774654,
                        2.790453220382528, -5.863177155806287, 0.8486557355346606, -4.264080919319636, 0.3888779755608768,
                        2.970168564061654, -0.2060336869650933, 0.3235539865467182, -3.5119517835917167, 0.21457732071448982,
                        0.8653763338751287, 4.752732130060801, -0.8451319328967635, -6.176249057329475, -5.373309414630066,
                        3.148371786473841, -2.3129252181700704, -0.516133751800335, 8.733047742330319, -1.878281227560063,
                        5.195350044708304, -8.845900361760474, 3.0118925341421003, -6.574445657657411, -0.007378214169837501,
                        1.138872605702014, 1.9401156675637163, -5.268888728711436, 4.3544830155808665, 5.428918871423306,
                        -0.5815030771103155, 4.713637184888023, -0.5560231617241566, -1.8894499052302822, -5.973063602067792,
                        0.6791691847508271, -2.973945018027564, 1.1566650330246968, 8.582933405387745, -3.1925133748746797,
                        9.62338167756082, 1.3638751634265993, -2.416113573233821, 0.33540560615465864, -0.001869473843980292,
                        -12.015960004320537, -0.0049969941558250675, 0.0008985453425318233, 0.0004606007070955495, 0.0005335506688331253,
                        -0.002241481691438179, 0.005660834616764134, 0.003335468691179667, -2.026779970594704e-05, -0.0008001521923345031,
                        0.006230157810059881, -0.0012252191018067241, -0.005527141583863775, 0.0035520513211605466, 0.0060212228897932145,
                        0.01724187720725454, 0.0021986065581168987, -0.0018928082843652108, -0.00041151466541644577, 0.004009547412605312,
                        0.002669201485195651, 0.008899284527590113, 0.0016599058405530735, 0.0032615693677098133, -0.0013265694036517076,
                        0.0009168984583263741, -0.005875027589359425, -0.00010751480353036738, -5.581264571103137e-05, 0.009931983159125236,
                        -0.0015327125947828951, 0.005801726287575088, -0.00839899540138454, -0.0016827497440283909, -0.004478515763551045,
                        0.0026502242033112585, -0.0012317564186923066, -0.0005581919667923588, -0.0006202233263641139, -0.002506773660997476,
                        0.0013896114670050824, -0.0066452215662738436, -0.002660821153414857, 0.004247595201625361, -0.002687902734576445,
                        0.0026330281795085304, -0.00020121641068372557, -0.001785449237051812, -0.0019494876077295058, -0.005175464365321356,
                        -0.002225241359722905, 0.00040921501870046646, 0.0018231038417319278, -0.0035813635734124323, 0.00012964373797541684,
                        0.0036086796215910226, -0.003823057996288213, 0.000820820466604786, -0.006209503203290135, 0.0002535661151859781,
                        0.004389819112646085, -0.00013210427298622634, 0.00035161984211715407, -0.003244098931128399, 3.964234702926821e-05,
                        0.00033855882144820284, 0.0036154969293488983, -0.0012215261594521727, 0.0, -0.006306555534793473,
                        0.0019160227363872499, -0.0029911291972597007, -0.0008020933522303581, 0.002277737623547664, -0.002837278960682521,
                        0.0026536234200721016, -0.004998871883260329, 0.0010474084413618218, -0.003929600938794378, -7.697489585300385e-06,
                        0.000259908845869401, 0.0019397335147640605, -0.008474368828489384, 0.002791998830245711, 0.00548684787856841,
                        -0.00010111104973682057, 0.006095786278885334, -0.0002537866615051943, -0.002299746804516308, -0.0017526190291931047,
                        0.00037642180110832366, -0.0015836369798554206, 0.0007919080824808744, 0.005783011444524312, -0.004614354812559884,
                        0.002614534271554239, 0.0010375266963176264, -0.0020742975568242565, 2.1869969992960452e-05, -9.751855335176923e-07,
                        -0.015016991462244434])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.formula.penalties])
        assert np.allclose(lam,np.array([2.18441866069889, 33725.85735049408])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -32109.002 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -31874.389 

class Test_no_pen_hard:
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
        assert round(sigma,ndigits=3) == 68.104 

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef,np.array([9.039869547728102, -0.3969510814209464, -10.076553592962266, -8.52733518891349, 0.07048038861662916,
                        0.27975885115386095])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.formula.penalties])
        assert np.allclose(lam,np.array([])) 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -33724.491

class Test_te_rs_fact_hard:
    # te + random slope with factor
    sim_dat,_ = sim1(100,random_seed=100)

    # Specify formula 
    formula = Formula(lhs("y"),[i(),f(["time","x"],te=True,nk=5),rs(["fact"],rf="sub"),rs(["x","fact"],rf="sub")],data=sim_dat)

    # ... and model
    model = GAMM(formula,Gaussian())

    model.fit(maxiter=100)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 92.799 

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 47.504 

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef,np.array([-0.8794666789951235, -14.26651329169169, -0.7127806292527972, 34.05959998663827, -108.75280013033043,
                        35.00802630725822, 4.95989497589471, 15.473877149954204, 12.259330935333999, -101.77021193220882,
                        69.49913296359024, 2.646343171211558, 17.944976399817435, 3.6455015680469725, -70.12955249749142,
                        10.534101239683888, 2.0553559328404316, 10.040736649943291, 6.762861234405522, -43.386058365613785,
                        -27.2537484522925, 44.0916432035976, -24.702573519774607, -36.32395934438752, -18.11234184448792,
                        0.0, 0.819502434308355, -1.8986966014974132, 7.772131350296927, 2.3923108908561344,
                        0.0, 16.404159561955385, 1.8952695946974691, 4.866953627044129, 7.827111826864078,
                        5.408426871807656, -2.0255401048073, 11.997340226545631, 5.494559017121331, 0.6327461411454073,
                        7.219363532074583, 4.225100157670351, 6.536296782009104, 2.489235412459764, 4.6979031143398675,
                        0.0, 4.61136211491378, -15.80081621050213, -0.3714294823481911, -13.587798897188847,
                        -8.820864283210737, -1.8305944749477536, -6.530319589079725, 0.686532324745895, -3.834844778416352,
                        -24.290816084573546, 0.0, -1.1988681623495587, -0.0863507795356144, 2.301448465613074,
                        -8.155476514772168, -0.6614740031247903, -1.7698032891005502, -10.333385978453146, 0.786808056312848,
                        2.087556586579283, 6.64058356325592, -6.26749300401964, 0.0, 5.864264077224323,
                        0.0, 8.74772877249856, -0.5644315048275774, -4.077887158498237, 0.0,
                        -1.621366128404254, -2.832183519823368, 0.0, -2.433388741896052, 4.490366799780031,
                        1.022432248908266, -0.9316230391302142, -10.830936943300138, 11.766143887666116, -8.927248164917215,
                        0.0, 0.1077406922846634, -0.09105173929087894, -0.34985368782087556, 0.13809243490685938,
                        0.0, -0.2569719421139166, 0.09088739764231948, 0.19881720002503808, -1.2684551634686096,
                        0.048011879673526395, 0.04682451472567237, -0.08609597714980723, -0.2062759712459226, 0.028095607229575836,
                        -0.630483693505413, 0.1884418477979666, 0.15988239781180794, 0.13705553320565575, 0.1558347137138879,
                        0.0, -0.892050570714284, 1.4399528681038616, -0.08084578527380973, 0.1969490801544412,
                        0.26780204681696296, -0.09136216852131465, 0.1824040525960194, 0.12541567851557237, 0.18130905856395432,
                        4.279808781699123, 0.0, -0.4473378476453399, -0.019802100390211335, 0.18785100938939486,
                        1.3903978913660517, -0.26481524851513005, -0.2889131251433626, 0.18264814991870051, 0.14067587124234768,
                        -0.147606039758077, 0.4748110719945615, 0.4476998457780754, 0.0, 0.19115290023666998,
                        0.0, 0.11497700743606294, 0.2608370690995577, -0.14245624428979856, 0.0,
                        -0.08950474883840426, -0.500499378491731, 0.0, -0.11769960488178866, -0.0729515187284164,
                        0.05267885151725163, -0.04359191116973939, 0.7944783610715709, -0.7426219079537028, 0.5244580344101646])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.formula.penalties])
        assert np.allclose(lam,np.array([0.020598737955595477, 0.003494843482710244, 0.661282336487626, 249.88014253966026, 28.546158900703794, 211.04009784721802])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -32225.017 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -31959.948
    
    def test_print_smooth(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.model.print_smooth_terms()
        capture = capture.getvalue()

        comp = "f(['time', 'x']); edf: 16.262\nrs(['fact'],sub); edf: 38.08\nrs(['x', 'fact'],sub):0; edf: 12.841\nrs(['x', 'fact'],sub):1; edf: 14.283\nrs(['x', 'fact'],sub):2; edf: 10.333\n"
        assert comp == capture

class Test_te_rs_fact_QR_hard:
    # te + random slope with factor
    sim_dat,_ = sim1(100,random_seed=100)

    # Specify formula 
    formula = Formula(lhs("y"),[i(),f(["time","x"],te=True,nk=5),rs(["fact"],rf="sub"),rs(["x","fact"],rf="sub")],data=sim_dat)

    # ... and model
    model = GAMM(formula,Gaussian())

    model.fit(maxiter=100,method="QR")

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 92.799

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 47.504 

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef,np.array([-8.79466679e-01, -1.42665133e+01, -7.12780629e-01,  3.40596000e+01,
                                          -1.08752800e+02,  3.50080263e+01,  4.95989498e+00,  1.54738772e+01,
                                          1.22593309e+01, -1.01770212e+02,  6.94991330e+01,  2.64634317e+00,
                                          1.79449764e+01,  3.64550157e+00, -7.01295525e+01,  1.05341012e+01,
                                          2.05535593e+00,  1.00407366e+01,  6.76286123e+00, -4.33860584e+01,
                                          -2.72537485e+01,  4.40916432e+01, -2.47025735e+01, -3.63239593e+01,
                                          -1.81123418e+01,  0.00000000e+00,  8.19502434e-01, -1.89869660e+00,
                                          7.77213135e+00,  2.39231089e+00,  0.00000000e+00,  1.64041596e+01,
                                          1.89526959e+00,  4.86695363e+00,  7.82711183e+00,  5.40842687e+00,
                                          -2.02554010e+00,  1.19973402e+01,  5.49455902e+00,  6.32746141e-01,
                                          7.21936353e+00,  4.22510016e+00,  6.53629678e+00,  2.48923541e+00,
                                          4.69790311e+00,  0.00000000e+00,  4.61136211e+00, -1.58008162e+01,
                                          -3.71429482e-01, -1.35877989e+01, -8.82086428e+00, -1.83059447e+00,
                                          -6.53031959e+00,  6.86532325e-01, -3.83484478e+00, -2.42908161e+01,
                                          0.00000000e+00, -1.19886816e+00, -8.63507795e-02,  2.30144847e+00,
                                          -8.15547651e+00, -6.61474003e-01, -1.76980329e+00, -1.03333860e+01,
                                          7.86808056e-01,  2.08755659e+00,  6.64058356e+00, -6.26749300e+00,
                                          0.00000000e+00,  5.86426408e+00,  0.00000000e+00,  8.74772877e+00,
                                          -5.64431505e-01, -4.07788716e+00,  0.00000000e+00, -1.62136613e+00,
                                          -2.83218352e+00,  0.00000000e+00, -2.43338874e+00,  4.49036680e+00,
                                          1.02243225e+00, -9.31623039e-01, -1.08309369e+01,  1.17661439e+01,
                                          -8.92724816e+00,  0.00000000e+00,  1.07740692e-01, -9.10517393e-02,
                                          -3.49853688e-01,  1.38092435e-01,  0.00000000e+00, -2.56971942e-01,
                                          9.08873976e-02,  1.98817200e-01, -1.26845516e+00,  4.80118797e-02,
                                          4.68245147e-02, -8.60959771e-02, -2.06275971e-01,  2.80956072e-02,
                                          -6.30483694e-01,  1.88441848e-01,  1.59882398e-01,  1.37055533e-01,
                                          1.55834714e-01,  0.00000000e+00, -8.92050571e-01,  1.43995287e+00,
                                          -8.08457853e-02,  1.96949080e-01,  2.67802047e-01, -9.13621685e-02,
                                          1.82404053e-01,  1.25415679e-01,  1.81309059e-01,  4.27980878e+00,
                                          0.00000000e+00, -4.47337848e-01, -1.98021004e-02,  1.87851009e-01,
                                          1.39039789e+00, -2.64815249e-01, -2.88913125e-01,  1.82648150e-01,
                                          1.40675871e-01, -1.47606040e-01,  4.74811072e-01,  4.47699846e-01,
                                          0.00000000e+00,  1.91152900e-01,  0.00000000e+00,  1.14977007e-01,
                                          2.60837069e-01, -1.42456244e-01,  0.00000000e+00, -8.95047488e-02,
                                          -5.00499378e-01,  0.00000000e+00, -1.17699605e-01, -7.29515187e-02,
                                          5.26788515e-02, -4.35919112e-02,  7.94478361e-01, -7.42621908e-01,
                                          5.24458034e-01])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.formula.penalties])
        assert np.allclose(lam,np.array([2.05987380e-02, 3.49484348e-03, 6.61282336e-01, 2.49880143e+02, 2.85461589e+01, 2.11040098e+02])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -32225.017 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -31959.948

class Test_print_parametric_hard:
    # te + random slope with factor
    sim_dat,_ = sim1(100,random_seed=100)

    # Specify formula 
    formula = Formula(lhs("y"),[i(),l(["fact"]),f(["time","x"],te=True,nk=5),rs(["fact"],rf="sub"),rs(["x","fact"],rf="sub")],data=sim_dat)

    # ... and model
    model = GAMM(formula,Gaussian())

    model.fit(maxiter=100)
    
    def test_print_parametric(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.model.print_parametric_terms()
        capture = capture.getvalue()

        comp = "Intercept: 4.918, t: 2.549, DoF.: 9409, P(|T| > |t|): 0.01082 *\nfact_fact_2: -14.004, t: -5.246, DoF.: 9409, P(|T| > |t|): 1.584e-07 ***\nfact_fact_3: -6.16, t: -2.335, DoF.: 9409, P(|T| > |t|): 0.01957 *\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: .\n"
        assert comp == capture

class Test_ti_rs_fact_hard:
    # ti + random slope with factor
    sim_dat,_ = sim1(100,random_seed=100)

    # Specify formula 
    formula = Formula(lhs("y"),[i(),f(["x"]),f(["time"]),f(["time","x"],te=False,nk=5),rs(["fact"],rf="sub"),rs(["x","fact"],rf="sub")],data=sim_dat)

    # ... and model
    model = GAMM(formula,Gaussian())

    model.fit(maxiter=100)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 104.426 

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 46.486 

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef,np.array([-0.9966590287606822, 1.4236003849146899, 6.47425869365001, -1.617928499789969, 1.692625032614929,
                        7.675576098202798, -1.7363013499021043, -1.2767695408510342, -9.889616000465907, -42.31343299575543,
                        10.639833505108795, 9.510527154203045, 11.67618431079849, 6.1774688080977125, 7.7374794464585,
                        5.914091100552806, -0.4651058775975218, 4.5340140126488455, 23.692294253850967, -6.100832412048143,
                        38.57938910119707, 57.461347819169106, 32.8781444187536, -11.363005804199965, -13.498306592670552,
                        6.729309257955318, 15.239914650856281, 2.6082473077680994, 5.747590559343884, -18.3036144673893,
                        9.727608560230767, 21.81713631882147, 11.525601529930436, 1.0033848161936922, 18.9037978478436,
                        25.65094936472521, 9.15685653297018, 9.115861040192229, 9.799927629536255, -29.287854556977162,
                        22.931866012268202, 56.33740892429906, 81.19206460243826, 102.76652680531406, 0.0,
                        -0.30321269031504633, -1.1356112263238594, 8.595686232603963, 2.3708813041035506, 0.0,
                        14.09555933439548, 2.958264360871084, 4.143191879265263, 10.111338918346814, 7.129395592974233,
                        -2.274624330276146, 12.738634314087234, 6.337532040928034, 0.7910725943931505, -3.245633483084778,
                        3.0642921740170674, 6.164662732419756, 2.859025762028847, 6.820404955179471, 0.0,
                        7.470643906551309, -19.634621356076487, -0.08839010143650042, -23.202841186278153, -7.063349932491601,
                        4.379343094971098, -2.747255295234261, 1.1861875036861265, -2.1989578253955657, -27.57401820846403,
                        0.0, -1.1799697951081132, 0.029307620852654783, 0.9420326928698906, -5.84413843385486,
                        -0.48179511901005445, -1.168974086934075, -7.269100046712463, 1.3130229142514256, 0.23777189892697956,
                        8.62966117400785, -1.8757649516521553, 0.0, 6.475077259178598, 0.0,
                        7.27147911091483, -0.29414995226076973, -5.971496301800591, 0.0, -1.2519726960118873,
                        -2.4554945963644372, 0.0, -4.032258724142517, 7.701168887134785, 1.2744951130936586,
                        -2.7602147918930795, -6.526066320115132, 6.055428273529817, -10.565650194546546, 0.0,
                        0.07441782295935119, -0.04805132144454618, -0.40873631733272003, 0.12075498468771992, 0.0,
                        -0.09968968461727758, 0.12517357034450888, 0.14933952891334176, -1.389783509589982, 0.004278309370366842,
                        0.10731770399917685, -0.0898791605538539, -0.1932364596231581, 0.03099333012302995, -0.03797419990934741,
                        0.295282504337062, 0.2530691437205495, 0.1388966133791108, 0.04418680238385609, 0.0,
                        -1.1411569401073998, 1.9957862878898898, -0.020957040152281448, 0.7162375435416818, 0.20540530907440854,
                        -0.7725752892893041, -0.28166206467280364, 0.23604211022768073, 0.12092368753919094, 4.772831560949575,
                        0.0, -0.4796013640046953, 0.007321007678360615, 0.08375747106456681, 0.19456908778975548,
                        -0.21010573491095091, -0.20787030854115082, -0.03119731282568737, 0.2557221947343299, -0.04357602255968131,
                        0.3713612191335093, -0.0290072726255912, 0.0, 0.12702831587341257, 0.0,
                        0.05752103152194779, 0.25958582826353155, -0.16403142366388504, 0.0, -0.04159566316792108,
                        -0.44607907650871376, 0.0, -0.11738162397335605, -0.3643991406864173, 0.03952102014322359,
                        -0.07773145225105302, 0.5737764041561451, -0.37492669276140106, 0.5690742062853811])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.formula.penalties])
        assert np.allclose(lam,np.array([0.2170753136944066, 0.2153080889209196, 0.0032036778387331583, 0.05238495548571929, 0.5653696817919165, 242.12205588662113, 22.40515587170834, 299.7936016565231])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -32158.151 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -31850.663 

class Test_3way_li_hard:
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
        assert round(sigma,ndigits=3) == 67.042 

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef,np.array([9.583040721472326, -11.156573068861054, -8.182249469164455, -0.3276918687327172, -0.0004580393533053191,
                        -0.11085582888258752, 0.22349809249409192, 0.0008203545755894872, -0.0006623936292670472, -6.338051206131253e-05,
                        0.00019413901971086924, 6.814072809149788e-05])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.formula.penalties])
        assert np.allclose(lam,np.array([])) 

class Test_print_smooth_by_factor_p_hard:
    # by factor
    sim_dat,_ = sim1(100,random_seed=100)

    # Specify formula 
    formula = Formula(lhs("y"),[i(),l(["fact"]),f(["time"],nk=10,by="fact")],data=sim_dat)

    # ... and model
    model = GAMM(formula,Gaussian())

    model.fit(maxiter=100)
    
    def test_print_smooth_p(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.model.print_smooth_terms(p_values=True)
        capture = capture.getvalue()

        comp = "f(['time'],by=fact): fact_1; edf: 9.391 f: 60.049 P(F > f) = 0.000e+00 ***\nf(['time'],by=fact): fact_2; edf: 7.587 f: 18.299 P(F > f) = 0.000e+00 ***\nf(['time'],by=fact): fact_3; edf: 4.689 f: 13.384 P(F > f) = 2.313e-12 ***\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n"
        assert comp == capture

class Test_print_smooth_by_factor_fs_p_hard:
    # by factor
    sim_dat,_ = sim1(100,random_seed=100)

    # Specify formula 
    formula = Formula(lhs("y"),[i(),l(["fact"]),f(["time"],nk=10,by="fact"),fs(["time"],rf="sub")],data=sim_dat)

    # ... and model
    model = GAMM(formula,Gaussian())

    model.fit(maxiter=100)
    
    def test_print_smooth_p(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.model.print_smooth_terms(p_values=True)
        capture = capture.getvalue()

        comp = "f(['time'],by=fact): fact_1; edf: 9.45 f: 33.87 P(F > f) = 0.000e+00 ***\nf(['time'],by=fact): fact_2; edf: 7.851 f: 9.624 P(F > f) = 2.139e-10 ***\nf(['time'],by=fact): fact_3; edf: 5.084 f: 2.667 P(F > f) = 0.0238 *\nf(['time'],by=sub); edf: 95.075\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n"
        assert comp == capture

class Test_print_smooth_binomial:
    Binomdat = sim3(10000,0.1,family=Binomial(),seed=20)

    formula = Formula(lhs("y"),[i(),f(["x0"]),f(["x1"]),f(["x2"]),f(["x3"])],data=Binomdat)

    # By default, the Binomial family assumes binary data and uses the logit link.
    model = GAMM(formula,Binomial())
    model.fit()

    def test_print_smooth_p(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.model.print_smooth_terms(p_values=True)
        capture = capture.getvalue()

        comp = "f(['x0']); edf: 2.856 chi^2: 18.441 P(Chi^2 > chi^2) = 3.017e-04 ***\nf(['x1']); edf: 1.962 chi^2: 60.923 P(Chi^2 > chi^2) = 1.421e-13 ***\nf(['x2']); edf: 6.243 chi^2: 168.288 P(Chi^2 > chi^2) = 0.000e+00 ***\nf(['x3']); edf: 1.407 chi^2: 2.62 P(Chi^2 > chi^2) = 0.26934\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n"
        assert comp == capture

class Test_Vp_estimation:
    Binomdat = sim3(10000,0.1,family=Binomial(),seed=20)

    formula = Formula(lhs("y"),[i(),f(["x0"]),f(["x1"]),f(["x2"]),f(["x3"])],data=Binomdat)

    # By default, the Binomial family assumes binary data and uses the logit link.
    model = GAMM(formula,Binomial())
    model.fit()

    Vp,_,_,_ = estimateVp(model,strategy="JJJ1")

    def test_Vp(self):
        assert np.allclose(np.round(self.Vp,decimals=3),np.array([[ 2.2810e+00,  9.0000e-03,  3.0000e-03,  2.1000e-02],
                                                                  [ 9.0000e-03,  2.7790e+00,  1.0000e-03, -2.5000e-02],
                                                                  [ 3.0000e-03,  1.0000e-03,  4.9400e-01,  2.2000e-02],
                                                                  [ 2.1000e-02, -2.5000e-02,  2.2000e-02,  1.5408e+01]]))

class Test_diff_hard:
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
        assert np.allclose(self.diff,np.array([17.62072375, 18.82722105, 19.50432417, 19.74516923, 19.64289238,
                                                19.29062976, 18.7815175 , 18.20869176, 17.64400706, 17.07419154,
                                                16.46469173, 15.78095415, 14.98842535, 14.05255184, 12.93878015,
                                                11.64060684, 10.26372855,  8.94189194,  7.80884367,  6.99833041,
                                                6.64409882,  6.87989555,  7.7822363 ,  9.19871281, 10.91968587,
                                                12.73551625, 14.43656473, 15.81319209, 16.65575909, 16.8183427 ,
                                                16.40988459, 15.60304261, 14.57047462, 13.48483847, 12.51879201,
                                                11.8449931 , 11.58605263, 11.66439368, 11.95239235, 12.32242477,
                                                12.64686705, 12.7980953 , 12.64848564, 12.14961963, 11.5699006 ,
                                                11.2569373 , 11.55833851, 12.82171299, 15.39466952, 19.62481685]))
    
    def test_ci(self):
        assert np.allclose(self.ci,np.array([5.58416926, 5.36879518, 5.34075097, 5.34356764, 5.33383195,
                                            5.31966795, 5.31628247, 5.32322685, 5.32591974, 5.31966873,
                                            5.31043843, 5.30561321, 5.30858819, 5.31669344, 5.32255914,
                                            5.32000883, 5.31247196, 5.30689155, 5.30795407, 5.31563874,
                                            5.32514934, 5.32922455, 5.32384336, 5.31487837, 5.31035838,
                                            5.31464504, 5.32637238, 5.33907601, 5.34445365, 5.33935084,
                                            5.33260623, 5.33426276, 5.34822254, 5.37022067, 5.38976225,
                                            5.39578115, 5.38748736, 5.38221553, 5.3951352 , 5.42779883,
                                            5.46782474, 5.49716096, 5.50796728, 5.52481257, 5.58823505,
                                            5.69931063, 5.85390424, 6.15591458, 6.95084486, 8.7965458 ]))