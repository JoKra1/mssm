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

    model.fit(max_outer=100)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 97.59 

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 46.805

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef,np.array([ 9.34823357e+00, -4.12480712e-01, -9.79726654e+00, -8.84147613e+00,
                                            2.53418688e-02,  3.14635694e-01, -4.06353720e+00,  1.93833167e+00,
                                            2.76734845e-01,  1.23222086e+00, -2.55837222e+00,  4.74284163e+00,
                                            3.98128948e+00, -9.93660084e-01, -3.54564194e+00,  4.29251585e+00,
                                            -1.32605854e+00, -4.14415477e+00,  5.70593558e+00,  8.25028306e+00,
                                            1.15002976e+01,  1.31992300e+00, -1.57502454e+00, -6.61320296e-01,
                                            3.58538448e+00,  2.58927345e+00,  7.10253184e+00,  1.06922717e+00,
                                            2.37380561e+00, -2.57901169e+00,  1.59727296e+00, -4.19457598e+00,
                                            -2.42430595e-01, -2.08536268e-01,  7.56330983e+00, -1.16692771e+00,
                                            5.71760774e+00, -5.39240799e+00, -8.15056570e+00, -3.93691672e+00,
                                            5.86576614e+00, -1.12113744e+00, -5.23756836e-01, -5.93227243e-01,
                                            -2.52520476e+00,  6.10156119e+00, -5.96965678e+00, -5.95474407e+00,
                                            3.08696984e+00, -4.79732675e+00,  5.25291394e+00, -2.14507799e+00,
                                            -7.25884552e+00, -2.94826284e+00, -5.16986996e+00, -4.31583159e+00,
                                            8.14360337e-01,  1.67010523e+00, -4.55225606e+00,  2.31206533e+00,
                                            2.84686087e+00, -5.89265083e+00,  1.18688704e+00, -4.25610602e+00,
                                            4.35120004e-01,  2.97108387e+00, -1.72185347e-01,  2.46427700e-01,
                                            -3.52651093e+00,  2.41571936e-01,  7.17557045e-01,  4.81392854e+00,
                                            -7.86218803e-01, -6.14760379e+00, -5.27462938e+00,  3.03928369e+00,
                                            -2.37407768e+00, -4.01300413e-01,  8.68982320e+00, -1.86560253e+00,
                                            4.96356356e+00, -8.73902694e+00,  2.67441566e+00, -6.34198980e+00,
                                            4.71752258e-02,  1.22108898e+00,  2.05473847e+00, -5.08976695e+00,
                                            4.30813260e+00,  5.53216376e+00, -6.50838857e-01,  4.74846838e+00,
                                            -4.35251328e-01, -1.78397142e+00, -5.90334091e+00,  6.77475154e-01,
                                            -3.29062209e+00,  1.22037194e+00,  8.45255403e+00, -3.13018031e+00,
                                            9.47424367e+00,  1.28155836e+00, -2.14077758e+00,  5.15760303e-01,
                                            2.82194429e-01, -1.17679851e+01, -6.84532713e-03,  1.17214540e-03,
                                            2.74926778e-04,  7.09663802e-04, -2.87317738e-03,  7.37507574e-03,
                                            4.58582893e-03, -2.86135634e-05, -1.07205799e-03,  8.21992121e-03,
                                            -2.00473098e-03, -7.27947239e-03,  4.60064791e-03,  7.95880306e-03,
                                            2.25191552e-02,  2.73662440e-03, -2.60789002e-03, -4.95130068e-04,
                                            5.05901441e-03,  3.83989411e-03,  1.16579489e-02,  1.90895538e-03,
                                            4.47734494e-03, -1.70810769e-03,  1.24187249e-03, -7.73040312e-03,
                                            -4.53769065e-04, -1.20100743e-04,  1.28498484e-02, -2.38581301e-03,
                                            7.57366847e-03, -1.14907550e-02, -2.22969500e-03, -5.89513389e-03,
                                            3.54713846e-03, -1.71107414e-03, -6.86238695e-04, -8.19967098e-04,
                                            -3.34494155e-03,  1.84486407e-03, -8.76705316e-03, -3.60094508e-03,
                                            6.00026265e-03, -3.66082758e-03,  3.47906244e-03, -2.47079765e-04,
                                            -2.29929313e-03, -2.54695666e-03, -6.77368689e-03, -2.92056216e-03,
                                            5.27634559e-04,  2.59699849e-03, -4.71914460e-03,  1.66446325e-04,
                                            4.87772499e-03, -5.09056590e-03,  1.52090995e-03, -8.21147817e-03,
                                            3.75893145e-04,  5.81778847e-03, -1.46269005e-04,  3.54808185e-04,
                                            -4.31586912e-03,  5.91288602e-05,  3.71931567e-04,  4.85178766e-03,
                                            -1.50556383e-03,  0.00000000e+00, -8.20200088e-03,  2.45054890e-03,
                                            -4.06767263e-03, -8.26246724e-04,  3.00279918e-03, -3.73368805e-03,
                                            3.35888823e-03, -6.54290370e-03,  1.23220507e-03, -5.02218070e-03,
                                            6.52062653e-05,  3.69207668e-04,  2.72175159e-03, -1.08458531e-02,
                                            3.65969742e-03,  7.40767777e-03, -1.49933114e-04,  8.13588157e-03,
                                            -2.63204616e-04, -2.87680232e-03, -2.29491040e-03,  4.97470869e-04,
                                            -2.32155082e-03,  1.10697259e-03,  7.54542577e-03, -5.99411544e-03,
                                            3.41026922e-03,  1.29163717e-03, -2.43502119e-03,  4.45556998e-05,
                                            1.95026568e-04, -1.94851637e-02])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.formula.penalties])
        assert np.allclose(lam,np.array([2.21567798e+00, 2.58200022e+04])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -32120.91 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -31886.766

class Test_no_pen_hard:
    # No penalties
    sim_dat,_ = sim1(100,random_seed=100)

    # Specify formula 
    formula = Formula(lhs("y"),[i(),*li(["x","fact"])],data=sim_dat)

    # ... and model
    model = GAMM(formula,Gaussian())

    model.fit(max_outer=100)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 6 

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 68.036 

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef,np.array([  9.09462398,  -0.40030025, -10.08258653,  -8.66437813,
                                            0.07117496,   0.29090723])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.formula.penalties])
        assert np.allclose(lam,np.array([])) 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -33719.718

class Test_te_rs_fact_hard:
    # te + random slope with factor
    sim_dat,_ = sim1(100,random_seed=100)

    # Specify formula 
    formula = Formula(lhs("y"),[i(),f(["time","x"],te=True,nk=5),rs(["fact"],rf="sub"),rs(["x","fact"],rf="sub")],data=sim_dat)

    # ... and model
    model = GAMM(formula,Gaussian())

    model.fit(max_outer=100)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 92.601 

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 47.63 

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef,np.array([-8.30300915e-01, -1.62575662e+01,  3.75545293e+00,  3.28335584e+01,
                                            -1.03918612e+02,  3.44105191e+01,  5.27629918e+00,  1.56969059e+01,
                                            1.26673363e+01, -1.00123881e+02,  6.70143258e+01,  3.86562330e+00,
                                            1.95649156e+01,  4.86327309e+00, -7.25159919e+01,  1.30742924e+01,
                                            1.95090107e+00,  9.28040152e+00,  7.11371053e+00, -4.42679942e+01,
                                            -1.76804150e+01,  4.78613869e+01, -1.98052935e+01, -3.54456054e+01,
                                            -1.83416110e+01,  0.00000000e+00,  8.09245518e-01, -1.88785044e+00,
                                            7.86729219e+00,  2.24203537e+00,  0.00000000e+00,  1.63941579e+01,
                                            1.90460014e+00,  4.75391046e+00,  7.87148618e+00,  5.89124427e+00,
                                            -2.05573658e+00,  1.17824896e+01,  5.67949983e+00,  5.60320399e-01,
                                            7.16489667e+00,  4.25921467e+00,  6.32718677e+00,  2.33817316e+00,
                                            4.63554642e+00,  0.00000000e+00,  4.33279542e+00, -1.54508599e+01,
                                            -4.34825704e-01, -1.31251334e+01, -8.82479024e+00, -2.04672844e+00,
                                            -6.63234939e+00,  6.01391323e-01, -3.45057541e+00, -2.36964763e+01,
                                            0.00000000e+00, -1.27173425e+00, -1.07503433e-01,  2.33531411e+00,
                                            -7.66907848e+00, -6.74989624e-01, -1.78519227e+00, -1.07199785e+01,
                                            7.97691278e-01,  2.06075903e+00,  6.57708624e+00, -6.56854566e+00,
                                            0.00000000e+00,  5.72411711e+00,  0.00000000e+00,  8.51037151e+00,
                                            -3.26500076e-01, -4.26368998e+00,  0.00000000e+00, -1.73618955e+00,
                                            -2.70298267e+00,  0.00000000e+00, -2.43800268e+00,  4.33837458e+00,
                                            9.23391435e-01, -7.11570185e-01, -1.07767560e+01,  1.16849067e+01,
                                            -9.00945920e+00,  0.00000000e+00,  1.10312730e-01, -9.26525263e-02,
                                            -3.58767133e-01,  1.32449931e-01,  0.00000000e+00, -2.67409758e-01,
                                            9.34745733e-02,  1.98748907e-01, -1.26535756e+00,  2.35853706e-02,
                                            4.38376559e-02, -8.00092020e-02, -2.24001329e-01,  2.54625771e-02,
                                            -6.31102720e-01,  1.85562538e-01,  1.65676987e-01,  1.31754143e-01,
                                            1.59439453e-01,  0.00000000e+00, -8.60517307e-01,  1.40166940e+00,
                                            -8.88715952e-02,  1.65764015e-01,  2.65134765e-01, -6.40839325e-02,
                                            1.83074204e-01,  1.03160813e-01,  1.58498835e-01,  4.10177221e+00,
                                            0.00000000e+00, -4.45581695e-01, -2.31490987e-02,  1.78988174e-01,
                                            1.20348890e+00, -2.53742995e-01, -2.73649103e-01,  2.07396369e-01,
                                            1.33922151e-01, -1.48365256e-01,  4.77971694e-01,  4.39481522e-01,
                                            0.00000000e+00,  1.89639994e-01,  0.00000000e+00,  1.13688951e-01,
                                            2.49369716e-01, -1.22585243e-01,  0.00000000e+00, -9.74128321e-02,
                                            -4.98143303e-01,  0.00000000e+00, -1.19853781e-01, -5.48244925e-02,
                                            4.83550293e-02, -3.38405554e-02,  8.04424864e-01, -7.33133226e-01,
                                            5.32136453e-01])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.formula.penalties])
        assert np.allclose(lam,np.array([2.21073262e-02, 3.79563141e-03, 6.78414676e-01, 2.50485765e+02,
                                         3.11881227e+01, 2.13019441e+02])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -32236.023 

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -31972.739
    
    def test_print_smooth(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.model.print_smooth_terms()
        capture = capture.getvalue()

        comp = "f(['time', 'x']); edf: 16.131\nrs(['fact'],sub); edf: 38.089\nrs(['x', 'fact'],sub):0; edf: 12.887\nrs(['x', 'fact'],sub):1; edf: 14.136\nrs(['x', 'fact'],sub):2; edf: 10.358\n"
        assert comp == capture

class Test_te_rs_fact_QR_hard:
    # te + random slope with factor
    sim_dat,_ = sim1(100,random_seed=100)

    # Specify formula 
    formula = Formula(lhs("y"),[i(),f(["time","x"],te=True,nk=5),rs(["fact"],rf="sub"),rs(["x","fact"],rf="sub")],data=sim_dat)

    # ... and model
    model = GAMM(formula,Gaussian())

    model.fit(max_outer=100,method="QR")

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 92.601

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 47.63 

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef,np.array([-8.30300915e-01, -1.62575662e+01,  3.75545293e+00,  3.28335584e+01,
                                            -1.03918612e+02,  3.44105191e+01,  5.27629918e+00,  1.56969059e+01,
                                            1.26673363e+01, -1.00123881e+02,  6.70143258e+01,  3.86562330e+00,
                                            1.95649156e+01,  4.86327309e+00, -7.25159919e+01,  1.30742924e+01,
                                            1.95090107e+00,  9.28040152e+00,  7.11371053e+00, -4.42679942e+01,
                                            -1.76804150e+01,  4.78613869e+01, -1.98052935e+01, -3.54456054e+01,
                                            -1.83416110e+01,  0.00000000e+00,  8.09245518e-01, -1.88785044e+00,
                                            7.86729219e+00,  2.24203537e+00,  0.00000000e+00,  1.63941579e+01,
                                            1.90460014e+00,  4.75391046e+00,  7.87148618e+00,  5.89124427e+00,
                                            -2.05573658e+00,  1.17824896e+01,  5.67949983e+00,  5.60320399e-01,
                                            7.16489667e+00,  4.25921467e+00,  6.32718677e+00,  2.33817316e+00,
                                            4.63554642e+00,  0.00000000e+00,  4.33279542e+00, -1.54508599e+01,
                                            -4.34825704e-01, -1.31251334e+01, -8.82479024e+00, -2.04672844e+00,
                                            -6.63234939e+00,  6.01391323e-01, -3.45057541e+00, -2.36964763e+01,
                                            0.00000000e+00, -1.27173425e+00, -1.07503433e-01,  2.33531411e+00,
                                            -7.66907848e+00, -6.74989624e-01, -1.78519227e+00, -1.07199785e+01,
                                            7.97691278e-01,  2.06075903e+00,  6.57708624e+00, -6.56854566e+00,
                                            0.00000000e+00,  5.72411711e+00,  0.00000000e+00,  8.51037151e+00,
                                            -3.26500076e-01, -4.26368998e+00,  0.00000000e+00, -1.73618955e+00,
                                            -2.70298267e+00,  0.00000000e+00, -2.43800268e+00,  4.33837458e+00,
                                            9.23391435e-01, -7.11570185e-01, -1.07767560e+01,  1.16849067e+01,
                                            -9.00945920e+00,  0.00000000e+00,  1.10312730e-01, -9.26525263e-02,
                                            -3.58767133e-01,  1.32449931e-01,  0.00000000e+00, -2.67409758e-01,
                                            9.34745733e-02,  1.98748907e-01, -1.26535756e+00,  2.35853706e-02,
                                            4.38376559e-02, -8.00092020e-02, -2.24001329e-01,  2.54625771e-02,
                                            -6.31102720e-01,  1.85562538e-01,  1.65676987e-01,  1.31754143e-01,
                                            1.59439453e-01,  0.00000000e+00, -8.60517307e-01,  1.40166940e+00,
                                            -8.88715952e-02,  1.65764015e-01,  2.65134765e-01, -6.40839325e-02,
                                            1.83074204e-01,  1.03160813e-01,  1.58498835e-01,  4.10177221e+00,
                                            0.00000000e+00, -4.45581695e-01, -2.31490987e-02,  1.78988174e-01,
                                            1.20348890e+00, -2.53742995e-01, -2.73649103e-01,  2.07396369e-01,
                                            1.33922151e-01, -1.48365256e-01,  4.77971694e-01,  4.39481522e-01,
                                            0.00000000e+00,  1.89639994e-01,  0.00000000e+00,  1.13688951e-01,
                                            2.49369716e-01, -1.22585243e-01,  0.00000000e+00, -9.74128321e-02,
                                            -4.98143303e-01,  0.00000000e+00, -1.19853781e-01, -5.48244925e-02,
                                            4.83550293e-02, -3.38405554e-02,  8.04424864e-01, -7.33133226e-01,
                                            5.32136453e-01])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.formula.penalties])
        assert np.allclose(lam,np.array([2.21073262e-02, 3.79563141e-03, 6.78414676e-01, 2.50485765e+02,
                                         3.11881227e+01, 2.13019441e+02])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -32236.023

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -31972.739

class Test_print_parametric_hard:
    # te + random slope with factor
    sim_dat,_ = sim1(100,random_seed=100)

    # Specify formula 
    formula = Formula(lhs("y"),[i(),l(["fact"]),f(["time","x"],te=True,nk=5),rs(["fact"],rf="sub"),rs(["x","fact"],rf="sub")],data=sim_dat)

    # ... and model
    model = GAMM(formula,Gaussian())

    model.fit(max_outer=100)
    
    def test_print_parametric(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.model.print_parametric_terms()
        capture = capture.getvalue()

        comp = 'Intercept: 4.992, t: 2.609, DoF.: 9409, P(|T| > |t|): 0.00911 **\nfact_fact_2: -13.855, t: -5.263, DoF.: 9409, P(|T| > |t|): 1.445e-07 ***\nfact_fact_3: -6.252, t: -2.395, DoF.: 9409, P(|T| > |t|): 0.01664 *\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: .\n'
        assert comp == capture

class Test_ti_rs_fact_hard:
    # ti + random slope with factor
    sim_dat,_ = sim1(100,random_seed=100)

    # Specify formula 
    formula = Formula(lhs("y"),[i(),f(["x"]),f(["time"]),f(["time","x"],te=False,nk=5),rs(["fact"],rf="sub"),rs(["x","fact"],rf="sub")],data=sim_dat)

    # ... and model
    model = GAMM(formula,Gaussian())

    model.fit(max_outer=100)

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 103.555 

    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 46.601

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef,np.array([-9.71704143e-01,  1.54721686e+00,  6.37094590e+00, -1.81024530e+00,
                                            1.49129313e+00,  7.76170446e+00, -1.96237580e+00, -1.36937262e+00,
                                            -9.83761681e+00, -4.32302196e+01,  9.49189307e+00,  8.32820414e+00,
                                            1.03243783e+01,  5.21269242e+00,  6.91971180e+00,  5.49551079e+00,
                                            -1.27418118e+00,  5.40282355e+00,  2.11740677e+01, -7.19128186e+00,
                                            2.80723247e+01,  4.53116540e+01,  2.93637653e+01, -4.81976007e+00,
                                            -1.04991125e+01,  5.72442097e-01,  8.45846333e+00,  2.17018141e+00,
                                            8.67020948e+00, -1.50099944e+01,  5.94560673e+00,  1.76938358e+01,
                                            1.10900870e+01,  5.22591490e+00,  1.50258617e+01,  1.97543071e+01,
                                            3.41527520e+00,  6.79301296e+00,  1.21044533e+01, -2.43559622e+01,
                                            1.55643199e+01,  4.39239465e+01,  6.66970475e+01,  8.69091468e+01,
                                            0.00000000e+00, -3.14951990e-01, -1.10905894e+00,  8.77585189e+00,
                                            2.26658681e+00,  0.00000000e+00,  1.40839030e+01,  3.06020246e+00,
                                            4.07727301e+00,  9.96448961e+00,  7.67315584e+00, -2.26262888e+00,
                                            1.26004973e+01,  6.61513178e+00,  7.59755858e-01, -3.60914249e+00,
                                            3.07808480e+00,  6.00402073e+00,  2.77619325e+00,  6.88612689e+00,
                                            0.00000000e+00,  7.36665376e+00, -1.93778847e+01, -1.11035288e-01,
                                            -2.30469258e+01, -6.93658165e+00,  4.38712596e+00, -3.04300412e+00,
                                            1.12959958e+00, -1.99702268e+00, -2.73114905e+01,  0.00000000e+00,
                                            -1.21036113e+00,  3.41254393e-02,  9.53244574e-01, -5.57686089e+00,
                                            -4.63828161e-01, -1.10994608e+00, -7.49500155e+00,  1.34600035e+00,
                                            2.12919578e-01,  8.59037174e+00, -2.10410712e+00,  0.00000000e+00,
                                            6.36640794e+00,  0.00000000e+00,  7.06794114e+00, -1.11517758e-01,
                                            -6.10316341e+00,  0.00000000e+00, -1.35803897e+00, -2.26213977e+00,
                                            0.00000000e+00, -4.08129879e+00,  7.74754710e+00,  1.17088060e+00,
                                            -2.52598120e+00, -6.72366968e+00,  5.88345533e+00, -1.06319047e+01,
                                            0.00000000e+00,  7.44618614e-02, -4.60409879e-02, -4.20709238e-01,
                                            1.13261409e-01,  0.00000000e+00, -1.06900997e-01,  1.27039907e-01,
                                            1.44186258e-01, -1.35495627e+00, -1.93898758e-02,  1.05246077e-01,
                                            -8.52466119e-02, -2.12131368e-01,  2.92038635e-02, -2.02115234e-02,
                                            2.97284552e-01,  2.60330919e-01,  1.32323699e-01,  4.22244844e-02,
                                            0.00000000e+00, -1.11968011e+00,  1.97630129e+00, -2.57003864e-02,
                                            7.03090968e-01,  1.97811256e-01, -7.66132166e-01, -2.64846408e-01,
                                            2.19438580e-01,  1.07095112e-01,  4.67992499e+00,  0.00000000e+00,
                                            -4.80260428e-01,  8.32186882e-03,  8.27397567e-02,  7.34822493e-02,
                                            -1.97462641e-01, -1.92682279e-01, -1.66172481e-02,  2.55913742e-01,
                                            -4.27052636e-02,  3.73577224e-01, -4.21838602e-02,  0.00000000e+00,
                                            1.26216368e-01,  0.00000000e+00,  5.65018226e-02,  2.50526369e-01,
                                            -1.51895786e-01,  0.00000000e+00, -4.55964528e-02, -4.42880493e-01,
                                            0.00000000e+00, -1.20064811e-01, -3.61032864e-01,  3.66917318e-02,
                                            -7.18868817e-02,  5.92954524e-01, -3.56891388e-01,  5.75216981e-01])) 

    def test_GAMlam(self):
        lam = np.array([p.lam for p in self.model.formula.penalties])
        assert np.allclose(lam,np.array([2.06462523e-01, 2.61797067e-01, 4.54850831e-03, 7.59911476e-02,
                                        5.72234072e-01, 2.49782036e+02, 2.32293389e+01, 3.00260310e+02])) 

    def test_GAMreml(self):
        reml = self.model.get_reml()
        assert round(reml,ndigits=3) == -32166.474

    def test_GAMllk(self):
        llk = self.model.get_llk(False)
        assert round(llk,ndigits=3) == -31862.875 

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
        assert round(sigma,ndigits=3) == 67.004 

    def test_GAMcoef(self):
        coef, _ = self.model.get_pars()
        assert np.allclose(coef.flatten(),np.array([ 9.49234564e+00, -1.11906237e+01, -8.23135271e+00, -3.19215668e-01,
                                                    -3.19669863e-04, -1.05685883e-01,  2.18785853e-01,  8.56147251e-04,
                                                    -7.20358812e-04, -7.44800816e-05,  1.88303305e-04,  8.13699273e-05])) 

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

    model.fit(max_outer=100)
    
    def test_print_smooth_p(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.model.print_smooth_terms(p_values=True)
        capture = capture.getvalue()

        comp = "f(['time'],by=fact): fact_1; edf: 9.43 f: 62.313 P(F > f) = 0.000e+00 ***\nf(['time'],by=fact): fact_2; edf: 7.607 f: 18.735 P(F > f) = 0.000e+00 ***\nf(['time'],by=fact): fact_3; edf: 4.839 f: 12.721 P(F > f) = 5.083e-12 ***\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n"
        assert comp == capture

class Test_print_smooth_by_factor_fs_p_hard:
    # by factor
    sim_dat,_ = sim1(100,random_seed=100)

    # Specify formula 
    formula = Formula(lhs("y"),[i(),l(["fact"]),f(["time"],nk=10,by="fact"),fs(["time"],rf="sub")],data=sim_dat)

    # ... and model
    model = GAMM(formula,Gaussian())

    model.fit(max_outer=100)
    
    def test_print_smooth_p(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.model.print_smooth_terms(p_values=True)
        capture = capture.getvalue()

        comp = "f(['time'],by=fact): fact_1; edf: 9.493 f: 35.033 P(F > f) = 0.000e+00 ***\nf(['time'],by=fact): fact_2; edf: 7.866 f: 10.265 P(F > f) = 1.559e-11 ***\nf(['time'],by=fact): fact_3; edf: 5.295 f: 3.324 P(F > f) = 0.00656 **\nf(['time'],by=sub); edf: 94.041\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n"
        assert comp == capture

class Test_print_smooth_binomial:
    Binomdat = sim3(10000,0.1,family=Binomial(),seed=20)

    formula = Formula(lhs("y"),[i(),f(["x0"]),f(["x1"]),f(["x2"]),f(["x3"])],data=Binomdat)

    # By default, the Binomial family assumes binary data and uses the logit link.
    model = GAMM(formula,Binomial())
    model.fit(max_inner=1)

    def test_print_smooth_p(self):
        capture = io.StringIO()
        with redirect_stdout(capture):
            self.model.print_smooth_terms(p_values=True)
        capture = capture.getvalue()

        comp = "f(['x0']); edf: 2.856 chi^2: 18.441 P(Chi^2 > chi^2) = 3.017e-04 ***\nf(['x1']); edf: 1.962 chi^2: 60.923 P(Chi^2 > chi^2) = 1.421e-13 ***\nf(['x2']); edf: 6.243 chi^2: 168.288 P(Chi^2 > chi^2) = 0.000e+00 ***\nf(['x3']); edf: 1.407 chi^2: 2.62 P(Chi^2 > chi^2) = 0.16737\n\nNote: p < 0.001: ***, p < 0.01: **, p < 0.05: *, p < 0.1: . p-values are approximate!\n"
        assert comp == capture

class Test_Vp_estimation:
    Binomdat = sim3(10000,0.1,family=Binomial(),seed=20)

    formula = Formula(lhs("y"),[i(),f(["x0"]),f(["x1"]),f(["x2"]),f(["x3"])],data=Binomdat)

    # By default, the Binomial family assumes binary data and uses the logit link.
    model = GAMM(formula,Binomial())
    model.fit(max_inner=1)

    Vp,_,_,_,_ = estimateVp(model,strategy="JJJ1",Vp_fidiff=True)

    def test_Vp(self):
        assert np.allclose(np.round(self.Vp,decimals=3),np.array([[ 2.281e+00,  9.000e-03,  3.000e-03,  2.100e-02],
                                                                    [ 9.000e-03,  2.779e+00,  1.000e-03, -2.400e-02],
                                                                    [ 3.000e-03,  1.000e-03,  4.940e-01,  2.200e-02],
                                                                    [ 2.100e-02, -2.400e-02,  2.200e-02,  1.541e+01]]))

class Test_diff_hard:
    # pred_diff test
    sim_dat,_ = sim1(100,random_seed=100)

    # Specify formula 
    formula = Formula(lhs("y"),[i(),l(["fact"]),f(["time"],by="fact"),rs(["fact"],rf="sub"),rs(["x","fact"],rf="sub")],data=sim_dat)

    # ... and model
    model = GAMM(formula,Gaussian())

    model.fit(max_outer=100)

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
        assert np.allclose(self.diff,np.array([17.20756874, 18.56006769, 19.3407868 , 19.65067738, 19.59069076,
                                                19.26177827, 18.76489122, 18.20098095, 17.64796193, 17.0916012 ,
                                                16.49462896, 15.81977541, 15.02977076, 14.08734519, 12.95522891,
                                                11.62544874, 10.20721807,  8.83904688,  7.6594452 ,  6.80692301,
                                                6.41999034,  6.63715718,  7.53833267,  8.96902249, 10.71613147,
                                                12.56656442, 14.30722615, 15.72502149, 16.60685525, 16.80334424,
                                                16.41995315, 15.62585868, 14.59023753, 13.48226638, 12.47112193,
                                                11.72598086, 11.3696111 , 11.33914548, 11.52530807, 11.81882293,
                                                12.11041412, 12.2908057 , 12.25072176, 11.94170808, 11.55859751,
                                                11.35704462, 11.59270397, 12.52123016, 14.39827774, 17.4795013 ]))
    
    def test_ci(self):
        assert np.allclose(self.ci,np.array([5.51585458, 5.30307541, 5.27423164, 5.27636045, 5.26677133,
                                                5.25310522, 5.24996695, 5.25675651, 5.2592083 , 5.25287576,
                                                5.24373586, 5.23907861, 5.24219173, 5.25034428, 5.25617479,
                                                5.25356667, 5.24600153, 5.24043816, 5.24155349, 5.24930369,
                                                5.25886655, 5.26296068, 5.25755966, 5.24855614, 5.24400619,
                                                5.24829547, 5.26007102, 5.27286368, 5.27834216, 5.27329519,
                                                5.26651469, 5.26805321, 5.2818876 , 5.30386531, 5.32358128,
                                                5.32994662, 5.32191458, 5.31645209, 5.32872033, 5.36073382,
                                                5.4007834 , 5.43114323, 5.44312329, 5.45849684, 5.51697233,
                                                5.62549037, 5.78645176, 6.09575666, 6.8585062 , 8.57200825]))