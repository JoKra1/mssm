from mssm.models import *
import numpy as np
import os

class Test_GAM:

    dat = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tutorials/data/GAMM/sim_dat.csv'))
    
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

    model.fit()

    def test_GAMedf(self):
        assert round(self.model.edf,ndigits=3) == 9.723

    def test_GAMTermEdf(self):
        assert round(self.model.term_edf[0],ndigits=3) == 8.723
    
    def test_GAMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 1084.879
    
    def test_GAMlam(self):
        assert round(self.model.formula.penalties[0].lam,ndigits=5) == 0.0089


class Test_GAMM:

    dat = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tutorials/data/GAMM/sim_dat.csv'))

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

    model.fit()

    def test_GAMMedf(self):
        assert round(self.model.edf,ndigits=3) == 153.601

    def test_GAMMTermEdf(self):
        assert np.max(np.abs(np.round(self.model.term_edf,decimals=3) - np.array([6.892, 8.635, 1.181, 1.001, 1.001, 1.029, 131.861]))) < 1e-6
    
    def test_GAMMsigma(self):
        _, sigma = self.model.get_pars()
        assert round(sigma,ndigits=3) == 577.196
    
    def test_GAMMlam(self):
        assert np.max(np.abs(np.round([p.lam for p in self.model.formula.penalties],decimals=3) - np.array([0.004, 0.006, 5842.507, 1101786.56 , 328846.811, 174267.629, 162215.095, 1178.787, 0.119, 2.166]))) < 1e-6