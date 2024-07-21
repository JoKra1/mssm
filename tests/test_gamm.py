from mssm.models import *
import numpy as np
import os

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