import numpy as np
import scipy as scp
import math

class Link:
   
   def f(self,mu):
      """
      Link function f() mapping mean mu of an exponential family to the model prediction f(mu) = eta
      see Wood (2017, 3.1.2) and Faraway (2016)
      """
      pass

   def fi(self,eta):
      """
      Inverse of the link function mapping eta = f(mu) to the mean fi(eta) = fi(f(mu)) = mu
      see Faraway (2016)
      """
      pass

   def dy1(self,mu):
      """
      First derivative of f(mu) with respect to mu
      Needed for Fisher scoring/PIRLS (Wood, 2017)
      """
      pass

class Logit(Link):

   def f(self, mu):
      """
      Canonical link for binomial distribution
      with mu = p the probability of success
      """
      return np.log(mu / (1 - mu))

   def fi(self,eta):
      """
      eta = log-odds, so mu = exp(eta) / (1 + exp(eta))
      see Faraway (2016)
      """
      return np.exp(eta) / (1 + np.exp(eta))
   
   def dy1(self,mu):
      """
      f(mu) = log(mu / (1 - mu))
             = log(mu) - log(1 - mu)
      dln(x)/dx = 1/x and sum rule: 
      df(mu)/dmu = 1/mu - 1/(1 - mu)
      
      Faraway (2016) simplifies to: df(mu)/dmu = 1 / (mu - mu**2) = 1/ ((1-mu)mu)
      """
      return 1 / ((1 - mu) * mu)

class Identity(Link):
   def f(self, mu):
      # Canonical link for normal distribution
      # mu = eta
      return mu

   def fi(self,eta):
      return eta
   
   def dy1(self,mu):
      return np.ones_like(mu)
   
   def dy2(self,mu):
      return np.zeros_like(mu)
   
class LOG(Link):
   # Log link
   def f(self,mu):
      return np.log(mu)
   
   def fi(self,eta):
      return np.exp(eta)
   
   def dy1(self,mu):
      return 1/mu
   
   def dy2(self,mu):
      return -1*(1/np.power(mu,2))

def est_scale(res,rows_X,total_edf):
   """
   Scale estimate from Wood & Fasiolo (2016)
   """
   resDot = res.T.dot(res)[0,0]

   sigma = resDot / (rows_X - total_edf)
   
   return sigma

class Family:
   """
   Base class to be implemented by Exp. family member
   """
   def __init__(self,link:Link or None,twopar:bool,scale:float=None) -> None:
      self.link = link
      self.twopar = twopar
      self.scale = scale # Known scale parameter!

   def init_mu(self,y):
      """
      Convenience function to compute an initial mu estimate passed to the
      GAMM/PIRLS estimation routine
      """
      return y
   
   def V(self,mu,**kwargs):
      """
      Variance function (Wood, 2017, 3.1.2)
      """
      pass
   
   def llk(self,y,mu,**kwargs):
      """
      log-likelihood of y under this family with mean=mu
      """
      pass
   
   def lp(self,y,mu,**kwargs):
      """
      Log-probability of observing every value in y under this family with mean=mu
      """
      pass

   def deviance(self,y,mu):
      """
      Deviance: 2 * (llk_max - llk_c) * scale (Wood, 2017; Faraway, 2016)
      """
      pass

class Binomial(Family):
   """
   Dependent variables specified in formula needs to hold proportions.
   ``n`` is the number of independent samples from Binomial for each proportion.
   """
   
   def __init__(self, link: Link=Logit(), scale: float = 1, n: int or [int] = 1) -> None:
      super().__init__(link,False,scale)
      self.n = n # Number of independent samples from Binomial!
      self.__max_llk = None # Needed for Deviance calculation.
   
   def init_mu(self,y):
      """
      Estimation assumes proportions as dep. variable
      According to: https://stackoverflow.com/questions/60526586/
      the glm() function in R always initializes mu = 0.75 for observed proportions (y)
      of 1 and mu = 0.25 for proportions of zero.
      This can be achieved by adding 0.5 to the observed proportion of success
      (and adding one observation)
      """
      prop = (y+0.5)/(2)
      self.__max_llk = self.llk(y,y)
      return prop
   
   def V(self,mu):
      # Faraway (2016):
      return mu * (1 - mu)/self.n
   
   def lp(self,y,mu):
      # y is observed proportion of success
      return scp.stats.binom.logpmf(k=y*self.n,p=mu,n=self.n)
   
   def llk(self,y,mu):
      # y is observed proportion of success
      return sum(self.lp(y,mu))[0]
   
   def deviance(self,y,mu):
      D = 2 * (self.__max_llk - self.llk(y,mu))
      return D[0]

class Gaussian(Family):
   def __init__(self, link: Link=None, scale: float = None) -> None:
      super().__init__(link, True, scale)

   def V(self,mu):
      # Faraway (2016)
      return np.ones(len(mu))
   
   def lp(self,y,mu,sigma=1):
      return scp.stats.norm.logpdf(y,loc=mu,scale=math.sqrt(sigma))
   
   def llk(self,y,mu,sigma = 1):
      return sum(self.lp(y,mu,sigma))[0]
   
   def deviance(self,y,mu):
      # Based on Faraway (2016)
      res = y - mu
      rss = res.T @ res
      return rss[0,0]
   
class GAMLSSFamily:
   def __init__(self,pars:int,links:[Link]) -> None:
      self.n_par = pars
      self.links = links
      self.d1 = [] # list with functions to evaluate derivative of llk with respect to corresponding mean
      self.d2 = [] # list with function to evaluate pure second derivative of llk with respect to corresponding mean
      self.d2m = [] # list with functions to evaluate mixed second derivative of llk. Order is 12,13,1k,23,24,...
      self.mean_init_fam:Family or None = None # Family to fit for the mean model to initialize coef.

   def llk(self,y,*args):
      # log-likelihood of y under this family
      pass
   
   def lp(self,y,*args):
      # Log-probability of observing every value in y under this family
      pass

class GAUMLSS(GAMLSSFamily):
   def __init__(self, links: [Link]) -> None:
      super().__init__(2, links)

      # All derivatives taken from gamlss.dist: https://github.com/gamlss-dev/gamlss.dist
      # see also: Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, Scale and Shape.
      self.d1 = [lambda y, mu, sigma: (1/np.power(sigma,2))*(y-mu),lambda y, mu, sigma: (np.power(y-mu,2)-np.power(sigma,2))/(np.power(sigma,3))]
      self.d2 = [lambda y, mu, sigma: -(1/np.power(sigma,2)), lambda y, mu, sigma: -(2/np.power(sigma,2))]
      self.d2m = [lambda y, mu, sigma: np.zeros_like(y)]
      self.mean_init_fam = Gaussian()
   
   def lp(self,y,mu,sigma):
      return scp.stats.norm.logpdf(y,loc=mu,scale=sigma)
   
   def llk(self,y,mu,sigma):
      return sum(self.lp(y,mu,sigma))[0]