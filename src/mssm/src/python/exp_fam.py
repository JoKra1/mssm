import numpy as np
import scipy as scp
import math
import sys

class Link:
   """
   Link function base class. To be implemented by any link functiion used for GAMMs and GAMMLSS models.
   Only links used by GAMLSS models require implementing the dy2 function.

   Methods:

   :method f: Link function
   :method fi: Inverse of the link function
   :method dy1: First derivative of link with respect to mean \mu
   :method dy2: Second derivative of link with respect to mean \mu
   """
   
   def f(self,mu):
      """
      Link function f() mapping mean \mu of an exponential family to the model prediction \eta, so that f(\mu) = \eta.
      see Wood (2017, 3.1.2) and Faraway (2016)

      References:
      
       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      """
      pass

   def fi(self,eta):
      """
      Inverse of the link function mapping \eta = f(\mu) to the mean fi(\eta) = fi(f(\mu)) = \mu.
      see Faraway (2016) and the ``Link.f`` function.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param eta: The vector containing the model prediction corresponding to each observation.
      :type eta: [float]
      """
      pass

   def dy1(self,mu):
      """
      First derivative of f(mu) with respect to mu. Needed for Fisher scoring/PIRLS (Wood, 2017).

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      """
      pass

class Logit(Link):
   """
   Logit Link function, which is canonical for the binomial model. \eta=log-odds of success.

   Methods:

   :method f: Link function
   :method fi: Inverse of the link function
   :method dy1: First derivative of link with respect to mean \mu
   """

   def f(self, mu):
      """
      Canonical link for binomial distribution with \mu = p the probability of success, so that the model prediction \eta
      is equal to the log-odds.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      """
      return np.log(mu / (1 - mu))

   def fi(self,eta):
      """
      For the logit link and the binomial model, \eta = log-odds, so the inverse to go from \eta to \mu is \mu = exp(\eta) / (1 + exp(\eta)).
      see Faraway (2016)

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param eta: The vector containing the model prediction corresponding to each observation.
      :type eta: [float]
      """
      return np.exp(eta) / (1 + np.exp(eta))
   
   def dy1(self,mu):
      """
      First derivative of f(mu) with respect to mu. Needed for Fisher scoring/PIRLS (Wood, 2017).
      f(mu) = log(mu / (1 - mu))
             = log(mu) - log(1 - mu)
      dln(x)/dx = 1/x and sum rule: 
      df(mu)/dmu = 1/mu - 1/(1 - mu)
      
      Faraway (2016) simplifies to: df(mu)/dmu = 1 / (mu - mu**2) = 1/ ((1-mu)mu)

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      """
      return 1 / ((1 - mu) * mu)

class Identity(Link):
   """
   Identity Link function. \mu=\eta and so this link is trivial.

   Methods:

   :method f: Link function
   :method fi: Inverse of the link function
   :method dy1: First derivative of link with respect to mean \mu
   :method dy2: Second derivative of link with respect to mean \mu
   """

   def f(self, mu):
      """
      Canonical link for normal distribution with \mu = \eta.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      """
      return mu

   def fi(self,eta):
      """
      For the identity link, \eta = \mu, so the inverse is also just the identity.
      see Faraway (2016)

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param eta: The vector containing the model prediction corresponding to each observation.
      :type eta: [float]
      """
      return eta
   
   def dy1(self,mu):
      """
      First derivative of f(mu) with respect to mu. Needed for Fisher scoring/PIRLS (Wood, 2017).

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      """
      return np.ones_like(mu)
   
   def dy2(self,mu):
      """
      Second derivative of f(mu) with respect to mu. Needed for GAMMLSS models (Wood, 2017).

      References:

       - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      """
      return np.zeros_like(mu)
   
class LOG(Link):
   """
   Log Link function. log(\mu)=\eta.

   Methods:

   :method f: Link function
   :method fi: Inverse of the link function
   :method dy1: First derivative of link with respect to mean \mu
   :method dy2: Second derivative of link with respect to mean \mu
   """
   
   def f(self,mu):
      """
      Non-canonical link for Gamma distribution with log(\mu) = \eta.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      """
      return np.log(mu)
   
   def fi(self,eta):
      """
      For the log link, \eta = log(\mu), so exp(\eta)=\mu.
      see Faraway (2016)

      References:
       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param eta: The vector containing the model prediction corresponding to each observation.
      :type eta: [float]
      """
      return np.exp(eta)
   
   def dy1(self,mu):
      """
      First derivative of f(mu) with respect to mu. Needed for Fisher scoring/PIRLS (Wood, 2017).

      References:
       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      """
      return 1/mu
   
   def dy2(self,mu):
      """
      Second derivative of f(mu) with respect to mu. Needed for GAMMLSS models (Wood, 2017).

      References:

       - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).      

      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      """
      return -1*(1/np.power(mu,2))

def est_scale(res,rows_X,total_edf):
   """
   Scale estimate from Wood & Fasiolo (2017).

   Refereces:
    - Wood, S. N., & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models.

   :param res: The vector containing the difference between the model prediction and the (pseudo) data.
   :type res: [float]
   :param rows_X: The number of observations collected.
   :type rows_X: int
   :param total_edf: The expected degrees of freedom for the model.
   :type total_edf: float
   """
   resDot = res.T.dot(res)[0,0]

   sigma = resDot / (rows_X - total_edf)
   
   return sigma

class Family:
   """
   Base class to be implemented by Exp. family member.   

   :param link: The link function to be used by the model of the mean of this family.
   :type link: Link
   :param twopar: Whether the family has two parameters (mean,scale) to be estimated, or only a single one (mean).
   :type twopar: bool
   :param scale: Known/fixed scale parameter for this family.
   :type scale: float or None, optional
   """

   def __init__(self,link:Link or None,twopar:bool,scale:float=None) -> None:
      self.link = link
      self.twopar = twopar
      self.scale = scale # Known scale parameter!

   def init_mu(self,y):
      """
      Convenience function to compute an initial mu estimate passed to the GAMM/PIRLS estimation routine.

      :param y: The vector containing each observation.
      :type y: [float]
      """
      return y
   
   def V(self,mu,**kwargs):
      """
      The variance function (of the mean; see Wood, 2017, 3.1.2). Different exponential families allow for different relationships
      between the variance in our random response variable and the mean of it. For the normal model this is assumed to be constant.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      """
      pass
   
   def llk(self,y,mu,**kwargs):
      """
      log-probability of y under this family with mean=\mu. Essentially sum over all elements in the vector returned by the ``lp`` method.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: The vector containing each observation.
      :type y: [float]
      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      """
      pass
   
   def lp(self,y,mu,**kwargs):
      """
      Log-probability of observing every value in y under this family with mean=\mu.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: The vector containing each observation.
      :type y: [float]
      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      :return: a N-dimensional vector containing the log-probability of observing each data-point under the current model.
      :rtype: [float]
      """
      pass

   def deviance(self,y,mu):
      """
      Deviance of the model under this family: 2 * (llk_max - llk_c) * scale (Wood, 2017; Faraway, 2016).

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: The vector containing each observation.
      :type y: [float]
      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      """
      pass

   def D(self,y,mu):
      """
      Contribution of each observation to model Deviance (Wood, 2017; Faraway, 2016)

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: The vector containing each observation.
      :type y: [float]
      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      """
      pass

class Binomial(Family):
   """
   Binomial family. For this implementation we assume that we have collected proportions of success, i.e., the dependent variables specified in the model `Formula` needs to hold observed proportions and not counts!
   If we assume that each observation $y_i$ reflects a single independent draw from a binomial, (with $n=1$, and $p_i$ being the probability that the result is 1) then the dependent variable should either hold 1 or 0.
   If we have multiple independent draws from the binomial per observation (i.e., row in our data-frame), then $n$ will usually differ between observations/rows in our data-frame (i.e., we observe $k_i$ counts of success
   out of $n_i$ draws - so that $y_i=k_i/n_i$). In that case, the `Binomial()` family accepts a vector for argument `n` (which is simply set to 1 by default, assuming binary data), containing $n_i$ for every observation $y_i$.

   By default the scale parameter is kept fixed/known at 1, but setting ``scale=None`` allows to estimate it.

   References:

    - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

   :param link: The link function to be used by the model of the mean of this family. By default set to the canonical logit link.
   :type link: Link
   :param scale: Known scale parameter for this family - by default set to 1.
   :type scale: float or None, optional
   :param n: Number of independent draws from a Binomial per observation/row of data-frame. For binary data this can simply be set to 1, which is the default.
   :type n: int or [int], optional
   """
   
   def __init__(self, link: Link=Logit(), scale: float = 1, n: int or [int] = 1) -> None:
      super().__init__(link,False,scale)
      self.n = n # Number of independent samples from Binomial!
      self.__max_llk = None # Needed for Deviance calculation.
   
   def init_mu(self,y):
      """
      Function providing initial \mu vector for GAMM.

      Estimation assumes proportions as dep. variable. According to: https://stackoverflow.com/questions/60526586/
      the glm() function in R always initializes mu = 0.75 for observed proportions (y) of 1 and mu = 0.25 for proportions of zero.
      This can be achieved by adding 0.5 to the observed proportion of success (and adding one observation).

      :param y: The vector containing each observation.
      :type y: [float]
      """
      prop = (y+0.5)/(2)
      self.__max_llk = self.llk(y,y)
      return prop
   
   def V(self,mu):
      """
      The variance function (of the mean; see Wood, 2017, 3.1.2) for the Binomial model. Variance is minimal for \mu=1 and \mu=0, maximal for \mu=0.5.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: The vector containing the predicted probability for the response distribution corresponding to each observation.
      :type mu: [float]
      """
      # Faraway (2016):
      return mu * (1 - mu)/self.n
   
   def lp(self,y,mu):
      """
      Log-probability of observing every proportion in y under their respective binomial with mean=\mu.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: The vector containing each observed proportion.
      :type y: [float]
      :param mu: The vector containing the predicted probability for the response distribution corresponding to each observation.
      :type mu: [float]
      :return: a N-dimensional vector containing the log-probability of observing each data-point under the current model.
      :rtype: [float]
      """
      # y is observed proportion of success
      return scp.stats.binom.logpmf(k=y*self.n,p=mu,n=self.n)
   
   def llk(self,y,mu):
      """
      log-probability of data under given model. Essentially sum over all elements in the vector returned by the ``lp`` method.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: The vector containing each observation.
      :type y: [float]
      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      """
      # y is observed proportion of success
      return sum(self.lp(y,mu))[0]
   
   def deviance(self,y,mu):
      """
      Deviance of the model under this family: 2 * (llk_max - llk_c) * scale (Wood, 2017; Faraway, 2016).

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: The vector containing each observation.
      :type y: [float]
      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      """
      D = 2 * (self.__max_llk - self.llk(y,mu))
      return D
   
   def D(self,y,mu):
      """
      Contribution of each observation to model Deviance (Wood, 2017; Faraway, 2016)

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: The vector containing each observation.
      :type y: [float]
      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      """
      # Based on Table 3.1 in Wood (2017)
      # Adds float_min**0.9 to log terms that could potentially be zero..
      k = y*self.n
      kmu = mu*self.n
      return 2 * (k*(np.log(k + np.power(sys.float_info.min,0.9)) - np.log(kmu)) + (self.n-k) * (np.log(self.n-k + np.power(sys.float_info.min,0.9)) - np.log(self.n-kmu)))


class Gaussian(Family):
   """Normal/Gaussian Family. 

   We assume: Y_i \sim N(\mu_i,\sigma) - i.e., each of the N observations is generated from a normally distributed RV with observation-specific
   mean and shared scale parameter \sigma. Equivalent to the assumption that the observed residual vector - the difference between the model
   prediction and the observed data - should look like what could be expected from drawing N independent samples from a Normal with mean zero and
   standard deviation equal to sigma.

   References:

    - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

   :param link: The link function to be used by the model of the mean of this family. By default set to the canonical identity link.
   :type link: Link
   :param scale: Known scale parameter for this family - by default set to None so that the scale parameter is estimated.
   :type scale: float or None, optional
   """
   def __init__(self, link: Link=Identity(), scale: float = None) -> None:
      super().__init__(link, True, scale)

   def V(self,mu):
      """Variance function for the Normal family.

      Not really a function since the link between variance and mean of the RVs is assumed constant for this model.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: a N-dimensional vector of the model prediction/the predicted mean
      :type mu: [float]
      :return: a N-dimensional vector of 1s
      :rtype: [float]
      """
      # Faraway (2016)
      return np.ones(len(mu))
   
   def lp(self,y,mu,sigma=1):
      """Log-probability of observing every proportion in y under their respective Normal with mean=\mu.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: The vector containing each observation.
      :type y: [float]
      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      :param sigma: The (estimated) sigma parameter, defaults to 1
      :type sigma: float, optional
      :return: a N-dimensional vector containing the log-probability of observing each data-point under the current model.
      :rtype: [float]
      """
      return scp.stats.norm.logpdf(y,loc=mu,scale=math.sqrt(sigma))
   
   def llk(self,y,mu,sigma = 1):
      """log-probability of data under given model. Essentially sum over all elements in the vector returned by the ``lp`` method.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: The vector containing each observation.
      :type y: [float]
      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      :param sigma: The (estimated) sigma parameter, defaults to 1
      :type sigma: float, optional
      :return: The log-probability of observing all data under the current model.
      :rtype: float
      """
      return sum(self.lp(y,mu,sigma))[0]
   
   def deviance(self,y,mu):
      """Deviance of the model under this family: 2 * (llk_max - llk_c) * scale (Wood, 2017; Faraway, 2016).

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: The vector containing each observation.
      :type y: [float]
      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      :return: The model deviance.
      :rtype: float
      """
      # Based on Faraway (2016)
      res = y - mu
      rss = res.T @ res
      return rss[0,0]
   
   def D(self,y,mu):
      """Contribution of each observation to model Deviance (Wood, 2017; Faraway, 2016)

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: The vector containing each observation.
      :type y: [float]
      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      :return: A N-dimensional vector containing the contribution of each data-point to the overall model deviance.
      :rtype: [float]
      """
      res = y - mu
      return np.power(res,2)

class Gamma(Family):
   """Gamma Family. 

   We assume: Y_i \sim \Gamma(\mu_i,\phi). The Gamma distribution is usually not expressed in terms of the mean and scale (\phi) parameter
   but rather in terms of a shape and rate parameter - called \alpha and \beta respectively. Wood (2017) provides \alpha = 1/\phi.
   With this we can obtain \beta = 1/\phi/\mu (see the source-code for :func:`lp` method for details).

   References:

    - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

   :param link: The link function to be used by the model of the mean of this family. By default set to the log link.
   :type link: Link
   :param scale: Known scale parameter for this family - by default set to None so that the scale parameter is estimated.
   :type scale: float or None, optional
   """

   def __init__(self, link: Link= LOG(), scale: float = None) -> None:
      super().__init__(link, True, scale)
   
   def V(self,mu):
      """Variance function for the Gamma family.

      The variance of random variable Y is proportional to it's mean raised to the second power.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: a N-dimensional vector of the model prediction/the predicted mean
      :type mu: [float]
      :return: mu raised to the power of 2
      :rtype: [float]
      """
      # Faraway (2016)
      return np.power(mu,2)
   
   def lp(self,y,mu,scale=1):
      """Log-probability of observing every proportion in y under their respective Gamma with mean=\mu.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: The vector containing each observed value.
      :type y: [float]
      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      :param scale: The (estimated) scale parameter, defaults to 1
      :type scale: float, optional
      :return: a N-dimensional vector containing the log-probability of observing each data-point under the current model.
      :rtype: [float]
      """
      # Need to transform from mean and scale to \alpha & \beta
      # From Wood (2017), we have that
      # \phi = 1/\alpha
      # so \alpha = 1/\phi
      # From https://en.wikipedia.org/wiki/Gamma_distribution, we have that:
      # \mu = \alpha/\beta
      # \mu = 1/\phi/\beta
      # \beta = 1/\phi/\mu
      # scipy docs, say to set scale to 1/\beta.
      # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
      alpha = 1/scale
      beta = alpha/mu  
      return scp.stats.gamma.logpdf(y,a=alpha,scale=(1/beta))
   
   def llk(self,y,mu,scale = 1):
      """log-probability of data under given model. Essentially sum over all elements in the vector returned by the ``lp`` method.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: The vector containing each observation.
      :type y: [float]
      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      :param scale: The (estimated) scale parameter, defaults to 1
      :type scale: float, optional
      :return: The log-probability of observing all data under the current model.
      :rtype: float
      """
      return sum(self.lp(y,mu,scale))[0]
   
   def D(self,y,mu):
      """Contribution of each observation to model Deviance (Wood, 2017; Faraway, 2016)

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: The vector containing each observation.
      :type y: [float]
      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      :return: A N-dimensional vector containing the contribution of each data-point to the overall model deviance.
      :rtype: [float]
      """
      # Based on Table 3.1 in Wood (2017)
      diff = (y - mu)/mu
      ratio = -(np.log(y) - np.log(mu))
      return 2 * (diff + ratio)
   
   def deviance(self,y,mu):
      """Deviance of the model under this family: 2 * (llk_max - llk_c) * scale (Wood, 2017; Faraway, 2016).

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: The vector containing each observation.
      :type y: [float]
      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      :return: The model deviance.
      :rtype: float
      """
      # Based on Table 3.1 in Wood (2017)
      dev = np.sum(self.D(y,mu))
      return dev

class GAMLSSFamily:
   """Base-class for families of Generalized Additive Mixed Models of Location, Scale, and Shape (GAMMLSS; Rigby & Stasinopoulos, 2005).


   References:

    - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, Scale and Shape.
    - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.

   :param pars: Number of parameters of the distribution belonging to the random variables assumed to have generated the observations.
   e.g., 2 for the Normal: mean and standard deviation.
   :type pars: int
   :param links:Link functions for each of the parameters of the distribution.
   :type links: [Link]
   """
   def __init__(self,pars:int,links:[Link]) -> None:
      self.n_par = pars
      self.links = links
      self.d1 = [] # list with functions to evaluate derivative of llk with respect to corresponding mean
      self.d2 = [] # list with function to evaluate pure second derivative of llk with respect to corresponding mean
      self.d2m = [] # list with functions to evaluate mixed second derivative of llk. Order is 12,13,1k,23,24,...
      self.mean_init_fam:Family or None = None # Family to fit for the mean model to initialize coef.

   def llk(self,y,*mus):
      """log-probability of data under given model. Essentially sum over all elements in the vector returned by the ``lp`` method.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: The vector containing each observation.
      :type y: [float]
      :param mus: A list including `self.n_par` lists - one for each parameter of the distribution. Each of those lists contains
      the expected value for a particular parmeter for each of the N observations.
      :type mus: [[float]]
      :return: The log-probability of observing all data under the current model.
      :rtype: float
      """
      pass
   
   def lp(self,y,*mus):
      """Log-probability of observing every proportion in y under their respective Gamma with mean=\mu.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: The vector containing each observed value.
      :type y: [float]
      :param mus: A list including `self.n_par` lists - one for each parameter of the distribution. Each of those lists contains
      the expected value for a particular parmeter for each of the N observations.
      :type mus: [[float]]
      :return: a N-dimensional vector containing the log-probability of observing each data-point under the current model.
      :rtype: [float]
      """
      pass

   def get_resid(self,y,*mus):
      """Get standardized residuals for a GAMMLSS model (Rigby & Stasinopoulos, 2005).

      References:

       - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, Scale and Shape.
       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).


      :param y: The vector containing each observed value.
      :type y: [float]
      :param mus: A list including `self.n_par` lists - one for each parameter of the distribution. Each of those lists contains
      the expected value for a particular parmeter for each of the N observations.
      :type mus: [[float]]
      :return: a N-dimensional vector containing the log-probability of observing each data-point under the current model.
      :rtype: [float]
      """
      pass


class GAUMLSS(GAMLSSFamily):
   """Family for a Normal GAMMLSS model (Rigby & Stasinopoulos, 2005).

   This Family follows the :class:`Gaussian` family, in that we assume: Y_i \sim N(\mu_i,\sigma_i). i.e., each of the N observations
   is still believed to have been generated from an independent normally distributed RV with observation-specific mean.
   
   The important difference is that the scale parameter, \sigma, is now also observation-specific and modeled as an additive combination
   of smooth functions and other parametric terms, just like the mean is in a Normal GAM. Note, that this explicitly models heteroscedasticity -
   the residuals are no longer assumed to be i.i.d samples from N(0,\sigma), since \sigma can now differ between residual realizations.

   References:

    - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, Scale and Shape.
    - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.

   :param links: Link functions for the mean and standard deviation. Standard would be `links=[Identity(),LOG()]`.
   :type links: [Link]
   """
   def __init__(self, links: [Link]) -> None:
      super().__init__(2, links)

      # All derivatives taken from gamlss.dist: https://github.com/gamlss-dev/gamlss.dist
      # see also: Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, Scale and Shape.
      self.d1 = [lambda y, mu, sigma: (1/np.power(sigma,2))*(y-mu),lambda y, mu, sigma: (np.power(y-mu,2)-np.power(sigma,2))/(np.power(sigma,3))]
      self.d2 = [lambda y, mu, sigma: -(1/np.power(sigma,2)), lambda y, mu, sigma: -(2/np.power(sigma,2))]
      self.d2m = [lambda y, mu, sigma: np.zeros_like(y)]
      self.mean_init_fam = Gaussian()
   
   def lp(self,y,mu,sigma):
      """Log-probability of observing every proportion in y under their respective Normal with observation-specific mean and standard deviation.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: The vector containing each observed value.
      :type y: [float]
      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      :param sigma: The vector containing the predicted stdandard deviation for the response distribution corresponding to each observation.
      :type sigma: [float]
      :return: a N-dimensional vector containing the log-probability of observing each data-point under the current model.
      :rtype: [float]
      """
      return scp.stats.norm.logpdf(y,loc=mu,scale=sigma)
   
   def llk(self,y,mu,sigma):
      """log-probability of data under given model. Essentially sum over all elements in the vector returned by the ``lp`` method.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: The vector containing each observation.
      :type y: [float]
      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      :param sigma: The vector containing the predicted stdandard deviation for the response distribution corresponding to each observation.
      :type sigma: [float]
      :return: The log-probability of observing all data under the current model.
      :rtype: float
      """
      return sum(self.lp(y,mu,sigma))[0]
   
   def get_resid(self,y,mu,sigma):
      """Get standardized residuals for a Normal GAMMLSS model (Rigby & Stasinopoulos, 2005).
      
      Essentially, each residual should reflect a realization of a normal with mean zero and observation-specific standard deviation.
      After scaling each residual by their observation-specific standard deviation we should end up with standardized
      residuals that can be expected to be i.i.d N(0,1) - assuming that our model is correct.

      :param y: The vector containing each observation.
      :type y: [float]
      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      :param sigma: The vector containing the predicted stdandard deviation for the response distribution corresponding to each observation.
      :type sigma: [float]
      :return: A list of standardized residuals that should be ~ N(0,1) if the model is correct.
      :rtype: [float]
      """
      res = y - mu
      res /= sigma
      return res

class MULNOMLSS(GAMLSSFamily):
   """Family for a Multinomial GAMMLSS model (Rigby & Stasinopoulos, 2005).

   This Family assumes that each observation y_i corresponds to one of K classes (labeled as 0,...,K) and reflects a
   realization of an independent RV Y_i with observation-specific probability mass function defined over the K classes. These K
   probabilities - that Y_i takes on class 1,...K - are modeled as additive combinations of smooth functions of covariates and
   other parametric terms.

   As an example, consider a visual search experiment where K-1 distractors are presented on a computer screen together with
   a single target and subjects are instructed to find the target and fixate it. With a Multinomial model we can estimate how
   the probability of looking at each of the K stimuli on the screen changes (smoothly) over time and as a function of other
   predictor variables of interest (e.g., contrast of stimuli, dependening on whether parfticipants are instructed to be fast or accurate).

   References:

    - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, Scale and Shape.
    - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.

   :param pars: K-1, i.e., 1- Number of classes or the number of linear predictors.
   :type pars: int
   """
   def __init__(self, pars: int) -> None:
      super().__init__(pars, [LOG() for _ in range(pars)])

      # All derivatives taken from gamlss.dist: https://github.com/gamlss-dev/gamlss.dist but made general for all number of pars.
      # see also: Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, Scale and Shape.
      self.d1 = []
      for ii in range(self.n_par):
         def d1(y,*mus,i=ii): dy1 = -(1/(np.sum(mus,axis=0)+1)); dy1[y == i] += 1/(mus[i][y == i]); return dy1
         self.d1.append(d1)

      self.d2 = []
      for ii in range(self.n_par):
         def d2(y,*mus,i=ii): dy2 = (-1*(np.sum([mus[iii] for iii in range(self.n_par) if i != iii],axis=0)+1)) / (mus[i]*np.power(np.sum(mus,axis=0)+1,2)); return dy2
         self.d2.append(d2)

      self.d2m = []
      for ii in range(int(self.n_par*(self.n_par-1)/2)):
         def d2m(y,*mus,i=ii): dy2m = 1/np.power(np.sum(mus,axis=0)+1,2); return dy2m
         self.d2m.append(d2m)
   

   def lp(self, y, *mus):
      """Log-probability of observing class k under current model.

      Our DV consists of K classes but we essentially enforce a sum-to zero constraint on the DV so that we end up modeling only
      K-1 (non-normalized) probabilities of observing class k (for all k except k==K) as an additive combination of smooth functions of our
      covariates and other parametric terms. The probability of observing class K as well as the normalized probabilities of observing each
      other class can readily be computed from these K-1 non-normalized probabilities. This is explained quite well on Wikipedia (see refs).

      Specifically, the probability of the outcome being class k is simply:

      p(Y_i == k) = mu_k / (1 + \sum_j^{K-1} mu_j) where mu_k is the aforementioned non-normalized probability of observing class k - which is simply set to 1 for class K (this follows from the sum-to-zero constraint; see Wikipedia).

      So, the log-prob of the outcome being class k is:

      log(p(Y_i == k)) = log(mu_k) - log(1 + \sum_j^{K-1} mu_j)

      References:

       - Wikipedia. https://en.wikipedia.org/wiki/Multinomial_logistic_regression
       - gamlss.dist on Github (see Rigby & Stasinopoulos, 2005). https://github.com/gamlss-dev/gamlss.dist/blob/main/R/MN4.R
       - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, Scale and Shape.

      :param y: The vector containing each observed class, every element must be larger than or equal to 0 and smaller than `self.n_par + 1`.
      :type y: [float]
      :param mus: A list containing K-1 (`self.n_par`) lists, each containing the non-normalized probabilities of observing class k for
      every observation.
      :type mus: [[float]]
      :return: a N-dimensional vector containing the log-probability of observing each data-point under the current model.
      :rtype: [float]
      """
      # Note, log(1) = 0, so we can simply initialize to -log(1 + \sum_j^{K-1} mu_j)
      # and then add for the K-1 probs we actually modeled.
      lp = -np.log(np.sum(mus,axis=0)+1)

      for pi in range(self.n_par):
         lp[y == pi] += np.log(mus[pi])[y == pi]

      return lp

   def llk(self, y, *mus):
      """log-probability of data under given model. Essentially sum over all elements in the vector returned by the :func:``lp`` method.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: The vector containing each observed class, every element must be larger than or equal to 0 and smaller than `self.n_par + 1`.
      :type y: [float]
      :param mus: A list containing K-1 (`self.n_par`) lists, each containing the non-normalized probabilities of observing class k for
      every observation.
      :type mus: [[float]]
      :return: The log-probability of observing all data under the current model.
      :rtype: float
      """
      return sum(self.lp(y,*mus))[0]
   
   def get_resid(self,y,*mus):
      pass

class GAMMALS(GAMLSSFamily):
   """Family for a GAMMA GAMMLSS model (Rigby & Stasinopoulos, 2005).

   This Family follows the :class:`Gamma` family, in that we assume: Y_i \sim \Gamma(\mu_i,\phi_i). The difference to the :class:`Gamma` family
   is that we now also model \phi as an additive combination of smooth variables and other parametric terms. The Gamma distribution is usually
   not expressed in terms of the mean and scale (\phi) parameter but rather in terms of a shape and rate parameter - called \alpha and \beta
   respectively. Wood (2017) provides \alpha = 1/\phi. With this we can obtain \beta = 1/\phi/\mu (see the source-code for :func:`lp` method
   of the :class:`Gamma` family for details).

   References:

    - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, Scale and Shape.
    - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.

   :param links: Link functions for the mean and standard deviation. Standard would be `links=[LOG(),LOG()]`.
   :type links: [Link]
   """

   def __init__(self,links: [Link]) -> None:
      super().__init__(2, links)
      # All derivatives based on gamlss.dist: https://github.com/gamlss-dev/gamlss.dist, but adjusted so that \phi (the scale) is \sigma^2.
      # see also: Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, Scale and Shape.
      self.d1 = [lambda y, mu, scale: (y-mu)/(scale*np.power(mu,2)),lambda y, mu, scale: (2/(scale*np.sqrt(scale)))*((y/mu)-np.log(y)+np.log(mu)+np.log(scale)-1+scp.special.digamma(1/(scale)))]
      self.d2 = [lambda y, mu, scale:  -1/(scale*np.power(mu,2)), lambda y, mu, scale: (4/np.power(scale,2))-(4/np.power(scale,3))*scp.special.polygamma(1,1/scale)]
      self.d2m = [lambda y, mu, scale: np.zeros_like(y)]
      self.mean_init_fam = Gamma()
   
   def lp(self,y,mu,scale):
      """Log-probability of observing every proportion in y under their respective Gamma with mean=\mu and scale=\phi.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: The vector containing each observed value.
      :type y: [float]
      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      :param scale: The vector containing the predicted scale parameter for the response distribution corresponding to each observation.
      :type scale: [float]
      :return: a N-dimensional vector containing the log-probability of observing each data-point under the current model.
      :rtype: [float]
      """
      # Need to transform from mean and scale to \alpha & \beta
      # From Wood (2017), we have that
      # \phi = 1/\alpha
      # so \alpha = 1/\phi
      # From https://en.wikipedia.org/wiki/Gamma_distribution, we have that:
      # \mu = \alpha/\beta
      # \mu = 1/\phi/\beta
      # \beta = 1/\phi/\mu
      # scipy docs, say to set scale to 1/\beta.
      # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
      alpha = 1/scale
      beta = alpha/mu  
      return scp.stats.gamma.logpdf(y,a=alpha,scale=(1/beta))
   
   def llk(self,y,mu,scale):
      """log-probability of data under given model. Essentially sum over all elements in the vector returned by the ``lp`` method.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: The vector containing each observation.
      :type y: [float]
      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      :param scale: The vector containing the predicted scale parameter for the response distribution corresponding to each observation.
      :type scale: [float]
      :return: The log-probability of observing all data under the current model.
      :rtype: float
      """
      return sum(self.lp(y,mu,scale))[0]
   
   def get_resid(self,y,mu,scale):
      """Get standardized residuals for a Gamma GAMMLSS model (Rigby & Stasinopoulos, 2005).
      
      Essentially, to get a standaridzed residual vector we first have to account for the mean-variance relationship of our RVs
      (which we also have to do for the :class:`Gamma` family) - for this we can simply compute deviance residuals again (see Wood, 2017).
      These should be ~ N(0,\scale) - so if we divide each of those by the observation-specific scale we can expect the resulting
      standardized residuals to be ~ N(0,1) if the model is correct.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).


      :param y: The vector containing each observation.
      :type y: [float]
      :param mu: The vector containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: [float]
      :param scale: The vector containing the predicted scale parameter for the response distribution corresponding to each observation.
      :type scale: [float]
      :return: A list of standardized residuals that should be ~ N(0,1) if the model is correct.
      :rtype: [float]
      """
      res = np.sign(y - mu) * np.sqrt(Gamma().D(y,mu)/scale)
      return res