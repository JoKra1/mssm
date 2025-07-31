import numpy as np
import scipy as scp
import math
import sys
import warnings
import copy
from collections.abc import Callable

class Link:
   """
   Link function base class. To be implemented by any link functiion used for GAMMs and GAMMLSS models.
   Only links used by ``GAMLSS`` models require implementing the dy2 function. Note, that care must be taken
   that every method returns only valid values. Specifically, no returned element may be ``numpy.nan`` or ``numpy.inf``.
   """
   
   def f(self,mu:np.ndarray) -> np.ndarray:
      """
      Link function :math:`f()` mapping mean :math:`\\boldsymbol{\\mu}` of an exponential family to the model prediction :math:`\\boldsymbol{\\eta}`, so that :math:`f(\\boldsymbol{\\mu}) = \\boldsymbol{\\eta}`.
      see Wood (2017, 3.1.2) and Faraway (2016).

      References:
      
       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      """
      pass

   def fi(self,eta:np.ndarray) -> np.ndarray:
      """
      Inverse of the link function mapping :math:`\\boldsymbol{\\eta} = f(\\boldsymbol{\\mu})` to the mean :math:`fi(\\boldsymbol{\\eta}) = fi(f(\\boldsymbol{\\mu})) = \\boldsymbol{\\mu}`.
      see Faraway (2016) and the ``Link.f`` function.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param eta: A numpy array containing the model prediction corresponding to each observation.
      :type eta: np.ndarray
      """
      pass

   def dy1(self,mu:np.ndarray) -> np.ndarray:
      """
      First derivative of :math:`f(\\boldsymbol{\\mu})` with respect to :math:`\\boldsymbol{\\mu}` Needed for Fisher scoring/PIRLS (Wood, 2017).

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      """
      pass

   def dy2(self,mu:np.ndarray) -> np.ndarray:
      """
      Second derivative of :math:`f(\\boldsymbol{\\mu})` with respect to :math:`\\boldsymbol{\\mu}`. Needed for GAMMLSS models (Wood, 2017).

      References:

       - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).      

      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      """
      pass

class Logit(Link):
   """
   Logit Link function, which is canonical for the binomial model. :math:`\\boldsymbol{\\eta}` = log-odds of success.
   """

   def f(self, mu:np.ndarray) -> np.ndarray:
      """
      Canonical link for binomial distribution with :math:`\\boldsymbol{\\mu}` holding the probabilities of success, so that the model prediction :math:`\\boldsymbol{\\eta}`
      is equal to the log-odds.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      """
      with warnings.catch_warnings(): # Divide by 0
         warnings.simplefilter("ignore")
         eta = np.log(mu / (1 - mu))

      return eta

   def fi(self,eta:np.ndarray) -> np.ndarray:
      """
      For the logit link and the binomial model, :math:`\\boldsymbol{\\eta}` = log-odds, so the inverse to go from :math:`\\boldsymbol{\\eta}` to :math:`\\boldsymbol{\\mu}` is :math:`\\boldsymbol{\\mu} = exp(\\boldsymbol{\\eta}) / (1 + exp(\\boldsymbol{\\eta}))`.
      see Faraway (2016)

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param eta: A numpy array containing the model prediction corresponding to each observation.
      :type eta: np.ndarray
      """
      with warnings.catch_warnings(): # Overflow
         warnings.simplefilter("ignore")
         mu = np.exp(eta) / (1 + np.exp(eta))
         
      return mu
   
   def dy1(self,mu:np.ndarray) -> np.ndarray:
      """
      First derivative of :math:`f(\\boldsymbol{\\mu})` with respect to :math:`\\boldsymbol{\\mu}`. Needed for Fisher scoring/PIRLS (Wood, 2017):

      .. math::


         f(\\mu) = log(\\mu / (1 - \\mu))

         f(\\mu) = log(\\mu) - log(1 - \\mu)

         \\partial f(\\mu)/ \\partial \\mu = 1/\\mu - 1/(1 - \\mu)
      
      Faraway (2016) simplifies this to: :math:`\\partial f(\\mu)/ \\partial \\mu = 1 / (\\mu - \\mu^2) = 1/ ((1-\\mu)\\mu)`

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      """
      with warnings.catch_warnings(): # Divide by 0
         warnings.simplefilter("ignore")
         d = 1 / ((1 - mu) * mu)
         
      return d

   def dy2(self,mu:np.ndarray) -> np.ndarray:
      """
      Second derivative of :math:`f(\\boldsymbol{\\mu})` with respect to :math:`\\boldsymbol{\\mu}`. Needed for GAMMLSS models (Wood, 2017).

      References:

       - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      """
      with warnings.catch_warnings(): # Divide by 0
         warnings.simplefilter("ignore")
         d2 = (2 * mu - 1) / (np.power(mu,2) * np.power(1-mu,2))

      return d2

class Identity(Link):
   """
   Identity Link function. :math:`\\boldsymbol{\\mu}=\\boldsymbol{\\eta}` and so this link is trivial.
   """

   def f(self, mu:np.ndarray) -> np.ndarray:
      """
      Canonical link for normal distribution with :math:`\\boldsymbol{\\eta} = \\boldsymbol{\\mu}`.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      """
      return mu

   def fi(self,eta:np.ndarray) -> np.ndarray:
      """
      For the identity link, :math:`\\boldsymbol{\\eta} = \\boldsymbol{\\mu}`, so the inverse is also just the identity.
      see Faraway (2016)

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param eta: A numpy array containing the model prediction corresponding to each observation.
      :type eta: np.ndarray
      """
      return eta
   
   def dy1(self,mu:np.ndarray) -> np.ndarray:
      """
      First derivative of :math:`f(\\boldsymbol{\\mu})` with respect to :math:`\\boldsymbol{\\mu}`. Needed for Fisher scoring/PIRLS (Wood, 2017).

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      """
      return np.ones_like(mu)
   
   def dy2(self,mu:np.ndarray) -> np.ndarray:
      """
      Second derivative of :math:`f(\\boldsymbol{\\mu})` with respect to :math:`\\boldsymbol{\\mu}`. Needed for GAMMLSS models (Wood, 2017).

      References:

       - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      """
      return np.zeros_like(mu)
   
class LOG(Link):
   """
   Log Link function. :math:`log(\\boldsymbol{\\mu}) = \\boldsymbol{\\eta}`.
   """
   
   def f(self,mu:np.ndarray) -> np.ndarray:
      """
      Non-canonical link for Gamma distribution with :math:`log(\\boldsymbol{\\mu}) = \\boldsymbol{\\eta}`.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      """
      with warnings.catch_warnings(): # log of < 0
         warnings.simplefilter("ignore")
         eta = np.log(mu)
      
      return eta
   
   def fi(self,eta:np.ndarray) -> np.ndarray:
      """
      For the log link, :math:`\\boldsymbol{\\eta} = log(\\boldsymbol{\\mu})`, so :math:`exp(\\boldsymbol{\\eta})=\\boldsymbol{\\mu}`.
      see Faraway (2016)

      References:
       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param eta: A numpy array containing the model prediction corresponding to each observation.
      :type eta: np.ndarray
      """
      with warnings.catch_warnings(): # Overflow
         warnings.simplefilter("ignore")
         mu = np.exp(eta)
      
      return mu
   
   def dy1(self,mu:np.ndarray) -> np.ndarray:
      """
      First derivative of :math:`f(\\boldsymbol{\\mu})` with respect to :math:`\\boldsymbol{\\mu}`. Needed for Fisher scoring/PIRLS (Wood, 2017).

      References:
       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      """
      with warnings.catch_warnings(): # Divide by 0
         warnings.simplefilter("ignore")
         d = 1/mu

      return d
   
   def dy2(self,mu:np.ndarray) -> np.ndarray:
      """
      Second derivative of :math:`f(\\boldsymbol{\\mu})` with respect to :math:`\\boldsymbol{\\mu}`. Needed for GAMMLSS models (Wood, 2017).

      References:

       - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).      

      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      """
      with warnings.catch_warnings(): # Divide by 0
         warnings.simplefilter("ignore")
         d2 = -1*(1/np.power(mu,2))
      
      return d2

class LOGb(Link):
   """
   Log + b Link function. :math:`log(\\boldsymbol{\\mu} + b) = \\boldsymbol{\\eta}`.

   :param b: The constant to add to :math:`\\mu` before taking the log.
   :type b: float
   """
   
   def __init__(self,b:float):
      super().__init__()
      self.b = b

   def f(self,mu:np.ndarray) -> np.ndarray:
      """
      :math:`log(\\boldsymbol{\\mu} + b) = \\boldsymbol{\\eta}`.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      """
      with warnings.catch_warnings(): # Log of < 0
         warnings.simplefilter("ignore")
         eta = np.log(mu + self.b)

      return eta
   
   def fi(self,eta:np.ndarray) -> np.ndarray:
      """
      For the logb link, :math:`\\boldsymbol{\\eta} = log(\\boldsymbol{\\mu} + b)`, so :math:`exp(\\boldsymbol{\\eta})-b =\\boldsymbol{\\mu}`

      References:
       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param eta: A numpy array containing the model prediction corresponding to each observation.
      :type eta: np.ndarray
      """
      with warnings.catch_warnings(): # Overflow
         warnings.simplefilter("ignore")
         mu = np.exp(eta) - self.b

      return mu
   
   def dy1(self,mu:np.ndarray) -> np.ndarray:
      """
      First derivative of :math:`f(\\boldsymbol{\\mu})` with respect to :math:`\\boldsymbol{\\mu}`. Needed for Fisher scoring/PIRLS (Wood, 2017).

      References:
       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      """
      with warnings.catch_warnings(): # Divide by 0
         warnings.simplefilter("ignore")
         d =  1/(self.b+mu)

      return d
   
   def dy2(self,mu:np.ndarray) -> np.ndarray:
      """
      Second derivative of :math:`f(\\boldsymbol{\\mu})` with respect to :math:`\\boldsymbol{\\mu}`. Needed for GAMMLSS models (Wood, 2017).

      References:

       - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).      

      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      """
      with warnings.catch_warnings(): # Divide by 0
         warnings.simplefilter("ignore")
         d2 =  -1*(1/np.power(mu+self.b,2))

      return d2

def est_scale(res:np.ndarray,rows_X:int,total_edf:float) -> float:
   """
   Scale estimate from Wood & Fasiolo (2017).

   Refereces:
    - Wood, S. N., & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models.

   :param res: A numpy array containing the difference between the model prediction and the (pseudo) data.
   :type res: np.ndarray
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
   :param twopar: Whether the family has two parameters (mean,scale) to be estimated (i.e., whether the likelihood is a function of two parameters), or only a single one (usually the mean).
   :type twopar: bool
   :param scale: Known/fixed scale parameter for this family. Setting this to None means the parameter has to be estimated. **Must be set to 1 if the family has no scale parameter** (i.e., when ``twopar = False``)
   :type scale: float or None, optional
   """

   def __init__(self,link:Link,twopar:bool,scale:float=None) -> None:
      self.link = link
      self.twopar = twopar
      self.scale = scale # Known scale parameter!
      self.is_canonical = False # Canonical link for generalized model?

   def init_mu(self,y:np.ndarray) -> np.ndarray | None:
      """
      Convenience function to compute an initial :math:`\\boldsymbol{\\mu}` estimate passed to the GAMM/PIRLS estimation routine.

      :param y: A numpy array of shape (-1,1) containing each observation.
      :type y: np.ndarray
      :return: a N-dimensional vector of shape (-1,1) containing an initial estimate of the mean
      :rtype: np.ndarray
      """
      return y
   
   def V(self,mu:np.ndarray) -> np.ndarray:
      """
      The variance function (of the mean; see Wood, 2017, 3.1.2). Different exponential families allow for different relationships
      between the variance in our random response variable and the mean of it. For the normal model this is assumed to be constant.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :return: a N-dimensional vector of shape (-1,1) containing the variance function evaluated for each mean
      :rtype: np.ndarray
      """
      pass

   def dVy1(self,mu:np.ndarray) -> np.ndarray:
      """
      The first derivative of the variance function (of the mean; see Wood, 2017, 3.1.2) with respect ot the mean.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :return: a N-dimensional vector of shape (-1,1) containing the first derivative of the variance function with respect to each mean
      :rtype: np.ndarray
      """
      pass
   
   def llk(self,y:np.ndarray,mu:np.ndarray,**kwargs) -> float:
      """
      log-probability of :math:`\\mathbf{y}` under this family with mean = :math:`\\boldsymbol{\\mu}`. Essentially sum over all elements in the vector returned by the :func:`lp` method.

      Families with more than one parameter that needs to be estimated in order to evaluate the model's log-likelihood (i.e., ``two_par=True``) must pass as key-word argument a ``scale``
      parameter with a default value, e.g.,::

         def llk(self, mu, scale=1):
            ...
      
      You can check the implementation of the :class:`Gaussian` Family for an example.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array of shape (-1,1) containing each observation.
      :type y: np.ndarray
      :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :return: log-likelihood of the model under this family
      :rtype: float
      """
      pass
   
   def lp(self,y:np.ndarray,mu:np.ndarray,**kwargs) -> np.ndarray:
      """
      Log-probability of observing every value in :math:`\\mathbf{y}` under this family with mean = :math:`\\boldsymbol{\\mu}`.

      Families with more than one parameter that needs to be estimated in order to evaluate the model's log-likelihood (i.e., ``two_par=True``) must pass as key-word argument a ``scale``
      parameter with a default value, e.g.,::

         def lp(self, mu, scale=1):
            ...
      
      You can check the implementation of the :class:`Gaussian` Family for an example.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array of shape (-1,1) containing each observation.
      :type y: np.ndarray
      :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :return: a N-dimensional vector of shape (-1,1) containing the log-probability of observing each data-point under the current model.
      :rtype: np.ndarray
      """
      pass

   def deviance(self,y:np.ndarray,mu:np.ndarray) -> float:
      """
      Deviance of the model under this family: 2 * (llk_max - llk_c) * scale (Wood, 2017; Faraway, 2016).

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array of shape (-1,1) containing each observation.
      :type y: np.ndarray
      :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :return: Deviance of the model under this family
      :rtype: float
      """
      pass

   def D(self,y:np.ndarray,mu:np.ndarray) -> np.ndarray:
      """
      Contribution of each observation to model Deviance (Wood, 2017; Faraway, 2016)

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array of shape (-1,1) containing each observation.
      :type y: np.ndarray
      :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :return: a N-dimensional vector of shape (-1,1) containing the contribution of each observation to the overall deviance.
      :rtype: np.ndarray
      """
      pass

class Binomial(Family):
   """
   Binomial family. For this implementation we assume that we have collected proportions of success, i.e., the dependent variables specified in the model `Formula` needs to hold observed proportions and not counts!
   If we assume that each observation :math:`y_i` reflects a single independent draw from a binomial, (with :math:`n=1`, and :math:`p_i` being the probability that the result is 1) then the dependent variable should either hold 1 or 0.
   If we have multiple independent draws from the binomial per observation (i.e., row in our data-frame), then :math:`n` will usually differ between observations/rows in our data-frame (i.e., we observe :math:`k_i` counts of success
   out of :math:`n_i` draws - so that :math:`y_i=k_i/n_i`). In that case, the `Binomial()` family accepts a vector for argument :math:`\\mathbf{n}` (which is simply set to 1 by default, assuming binary data), containing :math:`n_i` for every observation :math:`y_i`.

   In this implementation, the scale parameter is kept fixed/known at 1.

   References:

    - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

   :param link: The link function to be used by the model of the mean of this family. By default set to the canonical logit link.
   :type link: Link
   :param n: Number of independent draws from a Binomial per observation/row of data-frame. For binary data this can simply be set to 1, which is the default.
   :type n: int or [int], optional
   """
   
   def __init__(self, link: Link=Logit(), n: int | list[int] = 1) -> None:
      super().__init__(link,False,1)
      self.n:int | list[int] = n # Number of independent samples from Binomial!
      self.__max_llk:float|None = None # Needed for Deviance calculation.
      self.is_canonical:bool = isinstance(link,Logit)
   
   def init_mu(self,y:np.ndarray) -> np.ndarray:
      """
      Function providing initial :math:`\\boldsymbol{\\mu}` vector for GAMM.

      Estimation assumes proportions as dep. variable. According to: https://stackoverflow.com/questions/60526586/
      the glm() function in R always initializes :math:`\\mu` = 0.75 for observed proportions (i.e., elements in :math:`\\mathbf{y}`) of 1 and :math:`\\mu` = 0.25 for proportions of zero.
      This can be achieved by adding 0.5 to the observed proportion of success (and adding one observation).

      :param y: A numpy array containing each observation.
      :type y: np.ndarray
      :return: a N-dimensional vector of shape (-1,1) containing an initial estimate of the probability of success per observation
      :rtype: np.ndarray
      """
      prop = (y+0.5)/(2)
      self.__max_llk = self.llk(y,y)
      return prop
   
   def V(self,mu:np.ndarray) -> np.ndarray:
      """
      The variance function (of the mean; see Wood, 2017, 3.1.2) for the Binomial model. Variance is minimal for :math:`\\mu=1` and :math:`\\mu=0`, maximal for :math:`\\mu=0.5`.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: A numpy array containing the predicted probability for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :return: a N-dimensional vector of shape (-1,1) containing the variance function evaluated for each mean
      :rtype: np.ndarray
      """
      # Faraway (2016):
      return mu * (1 - mu)/self.n
   
   def dVy1(self,mu:np.ndarray) -> np.ndarray:
      """
      The first derivative of the variance function (of the mean; see Wood, 2017, 3.1.2) with respect ot the mean.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :return: a N-dimensional vector of shape (-1,1) containing the first derivative of the variance function with respect to each mean
      :rtype: np.ndarray
      """
      return (1 - 2*mu)/self.n
   
   def lp(self,y:np.ndarray,mu:np.ndarray) -> np.ndarray:
      """
      Log-probability of observing every proportion in :math:`\\mathbf{y}` under their respective binomial with mean = :math:`\\boldsymbol{\\mu}`.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array containing each observed proportion.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted probability for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :return: a N-dimensional vector containing the log-probability of observing each data-point under the current model.
      :rtype: np.ndarray
      """
      # y is observed proportion of success
      return scp.stats.binom.logpmf(k=y*self.n,p=mu,n=self.n)
   
   def llk(self,y:np.ndarray,mu:np.ndarray) -> float:
      """
      log-probability of data under given model. Essentially sum over all elements in the vector returned by the :func:`lp` method.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array containing each observation.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :return: log-likelihood of the model
      :rtype: float
      """
      # y is observed proportion of success
      return sum(self.lp(y,mu))[0]
   
   def deviance(self,y:np.ndarray,mu:np.ndarray) -> float:
      """
      Deviance of the model under this family: 2 * (llk_max - llk_c) * scale (Wood, 2017; Faraway, 2016).

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array containing each observation.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :return: Deviance of the model
      :rtype: float
      """
      dev = np.sum(self.D(y,mu))
      return dev
   
   def D(self,y:np.ndarray,mu:np.ndarray) -> np.ndarray:
      """
      Contribution of each observation to model Deviance (Wood, 2017; Faraway, 2016)

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array containing each observation.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :return: a N-dimensional vector of shape (-1,1) containing the contribution of each observation to the model deviance
      :rtype: np.ndarray
      """
      # Based on Table 3.1 in Wood (2017)
      k = y*self.n
      kmu = mu*self.n

      with warnings.catch_warnings(): # Divide by zero
         warnings.simplefilter("ignore")
         ratio1 = np.log(k) - np.log(kmu)
         ratio2 = np.log(self.n-k) - np.log(self.n-kmu)
      
      # Limiting behavior of y.. (see Wood, 2017)
      ratio1[np.isinf(ratio1) | np.isnan(ratio1)] = 0
      ratio2[np.isinf(ratio2) | np.isnan(ratio2)] = 0

      return 2 * (k*(ratio1) + ((self.n-k) * ratio2))


class Gaussian(Family):
   """Normal/Gaussian Family. 

   We assume: :math:`Y_i \\sim N(\\mu_i,\\sigma)` - i.e., each of the :math:`N` observations is generated from a normally distributed RV with observation-specific
   mean and shared scale parameter :math:`\\sigma`. Equivalent to the assumption that the observed residual vector - the difference between the model
   prediction and the observed data - should look like what could be expected from drawing :math:`N` independent samples from a Normal with mean zero and
   standard deviation equal to :math:`\\sigma`.

   References:

    - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

   :param link: The link function to be used by the model of the mean of this family. By default set to the canonical identity link.
   :type link: Link
   :param scale: Known scale parameter for this family - by default set to None so that the scale parameter is estimated.
   :type scale: float or None, optional
   """
   def __init__(self, link: Link=Identity(), scale: float = None) -> None:
      super().__init__(link, True, scale)
      self.is_canonical:bool = isinstance(link,Identity)

   def V(self,mu:np.ndarray) -> np.ndarray:
      """Variance function for the Normal family.

      Not really a function since the link between variance and mean of the RVs is assumed constant for this model.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: a N-dimensional vector of the model prediction/the predicted mean
      :type mu: np.ndarray
      :return: a N-dimensional vector of 1s
      :rtype: np.ndarray
      :return: a N-dimensional vector of shape (-1,1) containing the variance function evaluated for each mean
      :rtype: np.ndarray
      """
      # Faraway (2016)
      return np.ones_like(mu)
   
   def dVy1(self,mu:np.ndarray) -> np.ndarray:
      """
      The first derivative of the variance function (of the mean; see Wood, 2017, 3.1.2) with respect ot the mean.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :return: a N-dimensional vector of shape (-1,1) containing the first derivative of the variance function with respect to each mean
      :rtype: np.ndarray
      """
      return np.zeros_like(mu)
   
   def lp(self,y:np.ndarray,mu:np.ndarray,sigma:float=1) -> np.ndarray:
      """Log-probability of observing every proportion in :math:`\\mathbf{y}` under their respective Normal with mean = :math:`\\boldsymbol{\\mu}`.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array containing each observation.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :param sigma: The (estimated) sigma parameter, defaults to 1
      :type sigma: float, optional
      :return: a N-dimensional vector containing the log-probability of observing each data-point under the current model.
      :rtype: np.ndarray
      """
      return scp.stats.norm.logpdf(y,loc=mu,scale=math.sqrt(sigma))
   
   def llk(self,y:np.ndarray,mu:np.ndarray,sigma:float = 1) -> float:
      """log-probability of data under given model. Essentially sum over all elements in the vector returned by the :func:`lp` method.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array containing each observation.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :param sigma: The (estimated) sigma parameter, defaults to 1
      :type sigma: float, optional
      :return: The log-probability of observing all data under the current model.
      :rtype: float
      """
      return sum(self.lp(y,mu,sigma))[0]
   
   def deviance(self,y:np.ndarray,mu:np.ndarray) -> float:
      """Deviance of the model under this family: 2 * (llk_max - llk_c) * scale (Wood, 2017; Faraway, 2016).

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array containing each observation.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :return: The model deviance.
      :rtype: float
      """
      # Based on Faraway (2016)
      res = y - mu
      rss = res.T @ res
      return rss[0,0]
   
   def D(self,y:np.ndarray,mu:np.ndarray) -> np.ndarray:
      """Contribution of each observation to model Deviance (Wood, 2017; Faraway, 2016)

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array containing each observation.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :return: A N-dimensional vector containing the contribution of each data-point to the overall model deviance.
      :rtype: np.ndarray
      """
      res = y - mu
      return np.power(res,2)

class Gamma(Family):
   """Gamma Family. 

   We assume: :math:`Y_i \\sim \\Gamma(\\mu_i,\\phi)`. The Gamma distribution is usually not expressed in terms of the mean and scale (:math:`\\phi`) parameter
   but rather in terms of a shape and rate parameter - called :math:`\\alpha` and :math:`\\beta` respectively. Wood (2017) provides :math:`\\alpha = 1/\\phi`.
   With this we can obtain :math:`\\beta = 1/\\phi/\\mu` (see the source-code for :func:`lp` method for details).

   References:

    - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

   :param link: The link function to be used by the model of the mean of this family. By default set to the log link.
   :type link: Link
   :param scale: Known scale parameter for this family - by default set to None so that the scale parameter is estimated.
   :type scale: float or None, optional
   """

   def __init__(self, link: Link= LOG(), scale: float = None) -> None:
      super().__init__(link, True, scale)
      self.is_canonical:bool = False # Inverse link not implemented..
   
   def V(self,mu:np.ndarray) -> np.ndarray:
      """Variance function for the Gamma family.

      The variance of random variable :math:`Y` is proportional to it's mean raised to the second power.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: a N-dimensional vector of the model prediction/the predicted mean
      :type mu: np.ndarray
      :return: mu raised to the power of 2
      :rtype: np.ndarray
      """
      # Faraway (2016)
      return np.power(mu,2)
   
   def dVy1(self,mu:np.ndarray) -> np.ndarray:
      """
      The first derivative of the variance function (of the mean; see Wood, 2017, 3.1.2) with respect ot the mean.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :return: a N-dimensional vector of shape (-1,1) containing the first derivative of the variance function with respect to each mean
      :rtype: np.ndarray
      """
      return 2*mu
   
   def lp(self,y:np.ndarray,mu:np.ndarray,scale:float=1) -> np.ndarray:
      """Log-probability of observing every proportion in :math:`\\mathbf{y}` under their respective Gamma with mean = :math:`\\boldsymbol{\\mu}`.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array containing each observed value.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :param scale: The (estimated) scale parameter, defaults to 1
      :type scale: float, optional
      :return: a N-dimensional vector containing the log-probability of observing each data-point under the current model.
      :rtype: np.ndarray
      """
      # Need to transform from mean and scale to \alpha & \beta
      # From Wood (2017), we have that
      # \\phi = 1/\alpha
      # so \alpha = 1/\\phi
      # From https://en.wikipedia.org/wiki/Gamma_distribution, we have that:
      # \\mu = \alpha/\beta
      # \\mu = 1/\\phi/\beta
      # \beta = 1/\\phi/\\mu
      # scipy docs, say to set scale to 1/\beta.
      # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
      alpha = 1/scale
      beta = alpha/mu  
      return scp.stats.gamma.logpdf(y,a=alpha,scale=(1/beta))
   
   def llk(self,y:np.ndarray,mu:np.ndarray,scale:float = 1) -> float:
      """log-probability of data under given model. Essentially sum over all elements in the vector returned by the :func:`lp` method.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array containing each observation.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :param scale: The (estimated) scale parameter, defaults to 1
      :type scale: float, optional
      :return: The log-probability of observing all data under the current model.
      :rtype: float
      """
      return sum(self.lp(y,mu,scale))[0]
   
   def D(self,y:np.ndarray,mu:np.ndarray) -> np.ndarray:
      """Contribution of each observation to model Deviance (Wood, 2017; Faraway, 2016)

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array containing each observation.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :return: A N-dimensional vector containing the contribution of each data-point to the overall model deviance.
      :rtype: np.ndarray
      """
      # Based on Table 3.1 in Wood (2017)
      diff = (y - mu)/mu
      ratio = -(np.log(y) - np.log(mu))
      return 2 * (diff + ratio)
   
   def deviance(self,y:np.ndarray,mu:np.ndarray) -> float:
      """Deviance of the model under this family: 2 * (llk_max - llk_c) * scale (Wood, 2017; Faraway, 2016).

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array containing each observation.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :return: The model deviance.
      :rtype: float
      """
      # Based on Table 3.1 in Wood (2017)
      dev = np.sum(self.D(y,mu))
      return dev
   
class InvGauss(Family):
   """Inverse Gaussian Family. 

   We assume: :math:`Y_i \\sim IG(\\mu_i,\\phi)`. The Inverse Gaussian distribution is usually not expressed in terms of the mean and scale (:math:`\\phi`) parameter
   but rather in terms of a shape and scale parameter - called :math:`\\nu` and :math:`\\lambda` respectively (see the scipy implementation).
   We can simply set :math:`\\nu=\\mu` (compare scipy density to the one in table 3.1 of Wood, 2017).
   Wood (2017) shows that :math:`\\phi=1/\\lambda`, so this provides :math:`\\lambda=1/\\phi`

   References:

    - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
    - scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgauss.html

   :param link: The link function to be used by the model of the mean of this family. By default set to the log link.
   :type link: Link
   :param scale: Known scale parameter for this family - by default set to None so that the scale parameter is estimated.
   :type scale: float or None, optional
   """

   def __init__(self, link: Link= LOG(), scale: float|None = None) -> None:
      super().__init__(link, True, scale)
      self.is_canonical:bool = False # Modified inverse link not implemented..
   
   def V(self,mu:np.ndarray) -> np.ndarray:
      """Variance function for the Inverse Gaussian family.

      The variance of random variable :math:`Y` is proportional to it's mean raised to the third power.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: a N-dimensional vector of the model prediction/the predicted mean
      :type mu: np.ndarray
      :return: mu raised to the power of 3
      :rtype: np.ndarray
      """
      # Faraway (2016)
      return np.power(mu,3)
   
   def dVy1(self,mu:np.ndarray) -> np.ndarray:
      """
      The first derivative of the variance function (of the mean; see Wood, 2017, 3.1.2) with respect ot the mean.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :return: a N-dimensional vector of shape (-1,1) containing the first derivative of the variance function with respect to each mean
      :rtype: np.ndarray
      """
      return 3*np.power(mu,2)
   
   def lp(self,y:np.ndarray,mu:np.ndarray,scale:float=1) -> np.ndarray:
      """Log-probability of observing every value in :math:`\\mathbf{y}` under their respective inverse Gaussian with mean = :math:`\\boldsymbol{\\mu}`.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array containing each observed value.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :param scale: The (estimated) scale parameter, defaults to 1
      :type scale: float, optional
      :return: a N-dimensional vector containing the log-probability of observing each data-point under the current model.
      :rtype: np.ndarray
      """
      # Need to transform from mean and scale to \nu & \\lambda
      # From Wood (2017), we have that
      # \\phi = 1/\\lambda
      # so \\lambda = 1/\\phi
      # From the density in https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgauss.html,
      # we have that \nu=\\mu
      lam = 1/scale
      nu = mu  
      return scp.stats.invgauss.logpdf(y,mu=nu/lam,scale=lam)
   
   def llk(self,y:np.ndarray,mu:np.ndarray,scale:float = 1) -> float:
      """log-probability of data under given model. Essentially sum over all elements in the vector returned by the :func:`lp` method.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array containing each observation.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :param scale: The (estimated) scale parameter, defaults to 1
      :type scale: float, optional
      :return: The log-probability of observing all data under the current model.
      :rtype: float
      """
      return sum(self.lp(y,mu,scale))[0]
   
   def D(self,y:np.ndarray,mu:np.ndarray) -> np.ndarray:
      """Contribution of each observation to model Deviance (Wood, 2017; Faraway, 2016)

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array containing each observation.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :return: A N-dimensional vector containing the contribution of each data-point to the overall model deviance.
      :rtype: np.ndarray
      """
      # Based on Table 3.1 in Wood (2017)
      diff = np.power(y - mu,2)
      prod = np.power(mu,2)*y
      return diff/prod
   
   def deviance(self,y:np.ndarray,mu:np.ndarray) -> float:
      """Deviance of the model under this family: 2 * (llk_max - llk_c) * scale (Wood, 2017; Faraway, 2016).

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array containing each observation.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :return: The model deviance.
      :rtype: float
      """
      # Based on Table 3.1 in Wood (2017)
      dev = np.sum(self.D(y,mu))
      return dev
   
class Poisson(Family):
   """Poisson Family. 

   We assume: :math:`Y_i \\sim P(\\lambda)`. We can simply set :math:`\\lambda=\\mu` (compare scipy density to the one in table 3.1 of Wood, 2017)
   and treat the scale parameter of a GAMM (:math:`\\phi`) as fixed/known at 1.

   References:

    - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
    - scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html

   :param link: The link function to be used by the model of the mean of this family. By default set to the log link.
   :type link: Link
   """

   def __init__(self, link: Link= LOG()) -> None:
      super().__init__(link, False, 1)
      self.is_canonical:bool = isinstance(link,LOG)
   
   def V(self,mu:np.ndarray) -> np.ndarray:
      """Variance function for the Poisson family.

      The variance of random variable :math:`Y` is proportional to it's mean.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: a N-dimensional vector of the model prediction/the predicted mean
      :type mu: np.ndarray
      :return: mu 
      :rtype: np.ndarray
      """
      # Wood (2017)
      return mu
   
   def dVy1(self,mu:np.ndarray) -> np.ndarray:
      """
      The first derivative of the variance function (of the mean; see Wood, 2017, 3.1.2) with respect ot the mean.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :return: a N-dimensional vector of shape (-1,1) containing the first derivative of the variance function with respect to each mean
      :rtype: np.ndarray
      """
      return np.ones_like(mu)
   
   def lp(self,y:np.ndarray,mu:np.ndarray) -> np.ndarray:
      """Log-probability of observing every value in :math:`\\mathbf{y}` under their respective Poisson with mean = :math:`\\boldsymbol{\\mu}`.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array containing each observed value.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :return: a N-dimensional vector containing the log-probability of observing each data-point under the current model.
      :rtype: np.ndarray
      """
      # Need to transform from mean to \\lambda
      # From Wood (2017) and the density in https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html,
      # we have that \\lam=\\mu
      lam = mu
      return scp.stats.poisson.logpmf(y,mu=lam)
   
   def llk(self,y:np.ndarray,mu:np.ndarray) -> float:
      """log-probability of data under given model. Essentially sum over all elements in the vector returned by the :func:`lp` method.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array containing each observation.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :param scale: The (estimated) scale parameter, defaults to 1
      :type scale: float, optional
      :return: The log-probability of observing all data under the current model.
      :rtype: float
      """
      return sum(self.lp(y,mu))[0]
   
   def D(self,y:np.ndarray,mu:np.ndarray) -> np.ndarray:
      """Contribution of each observation to model Deviance (Wood, 2017; Faraway, 2016)

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array containing each observation.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :return: A N-dimensional vector containing the contribution of each data-point to the overall model deviance.
      :rtype: np.ndarray
      """
      # Based on Table 3.1 in Wood (2017)
      diff = y - mu

      with warnings.catch_warnings(): # Divide by zero
         warnings.simplefilter("ignore")
         ratio = y*(np.log(y) - np.log(mu))

      ratio[np.isinf(ratio) | np.isnan(ratio)] = 0
      
      return 2*ratio - 2*diff
   
   def deviance(self,y:np.ndarray,mu:np.ndarray) -> float:
      """Deviance of the model under this family: 2 * (llk_max - llk_c) * scale (Wood, 2017; Faraway, 2016).

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array containing each observation.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :return: The model deviance.
      :rtype: float
      """
      # Based on Table 3.1 in Wood (2017)
      dev = np.sum(self.D(y,mu))
      return dev
   
   def init_mu(self,y:np.ndarray) -> np.ndarray:
      """
      Function providing initial :math:`\\boldsymbol{\\mu}` vector for Poisson GAMM.

      We shrink extreme observed counts towards mean.

      :param y: A numpy array containing each observation.
      :type y: np.ndarray
      :return: a N-dimensional vector of shape (-1,1) containing an intial estimate of the mean of the response variables
      :rtype: np.ndarray
      """

      gmu = np.mean(y)
      norm = y / gmu
      
      norm[norm > 1.9] = 1.9
      norm[norm < 0.1] = 0.1
      mu = gmu * norm

      return mu


class GAMLSSFamily:
   """Base-class to be implemented by families of Generalized Additive Mixed Models of Location, Scale, and Shape (GAMMLSS; Rigby & Stasinopoulos, 2005).

   Apart from the required methods, three mandatory attributes need to be defined by the :func:`__init__` constructor of implementations of this class. These are required
   to evaluate the first and second (pure & mixed) derivative of the log-likelihood with respect to any of the log-likelihood's parameters (alternatively the linear predictors
   of the parameters - see the description of the ``d_eta`` instance variable.). See the variables below.

   Optionally, a ``mean_init_fam`` attribute can be defined - specfiying a :class:`Family` member that is fitted to the data to get an initial estimate of the mean parameter of the assumed distribution.

   References:
    - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, Scale and Shape.
    - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.

   :param pars: Number of parameters of the distribution belonging to the random variables assumed to have generated the observations, e.g., 2 for the Normal: mean and standard deviation.
   :type pars: int
   :param links: Link functions for each of the parameters of the distribution.
   :type links: [Link]
   :ivar bool d_eta: A boolean indicating whether partial derivatives of llk are provided with respect to the linear predictor instead of parameters (i.e., the mean), defaults to False (derivatives are provided with respect to parameters)
   :ivar [Callable] d1: A list holding ``n_par`` functions to evaluate the first partial derivatives of llk with respect to each parameter of the llk. Needs to be initialized when calling :func:`__init__`.
   :ivar [Callable] d2: A list holding ``n_par`` functions to evaluate the second (pure) partial derivatives of llk with respect to each parameter of the llk. Needs to be initialized when calling :func:`__init__`.
   :ivar [Callable] d2m: A list holding ``n_par*(n_par-1)/2`` functions to evaluate the second mixed partial derivatives of llk with respect to each parameter of the llk in **order**: ``d2m[0]`` = :math:`\\partial l/\\partial \\mu_1 \\partial \\mu_2`, ``d2m[1]`` = :math:`\\partial l/\\partial \\mu_1 \\partial \\mu_3`, ..., ``d2m[n_par-1]`` = :math:`\\partial l/\\partial \\mu_1 \\partial \\mu_{n_{par}}`, ``d2m[n_par]`` = :math:`\\partial l/\\partial \\mu_2 \\partial \\mu_3`, ``d2m[n_par+1]`` = :math:`\\partial l/\\partial \\mu_2 \\partial \\mu_4`, ... . Needs to be initialized when calling :func:`__init__`.
   """
   def __init__(self,pars:int,links:list[Link]) -> None:
      self.n_par:int = pars
      self.links:list[Link] = links
      self.d_eta:bool = False # Whether partial derivatives of llk are provided with respect to the linear predictor instead of parameters (i.e., the mean), defaults to False (derivatives are provided with respect to parameters)
      self.d1:list[Callable] = [] # list with functions to evaluate derivative of llk with respect to corresponding mean
      self.d2:list[Callable] = [] # list with function to evaluate pure second derivative of llk with respect to corresponding mean
      self.d2m:list[Callable] = [] # list with functions to evaluate mixed second derivative of llk. Order is 12,13,1k,23,24,...
      self.mean_init_fam: Family|None = None
   

   def llk(self,y:np.ndarray,*mus:list[np.ndarray]) -> float:
      """log-probability of data under given model. Essentially sum over all elements in the vector returned by the :func:`lp` method.

      References:
       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array of shape (-1,1) containing each observation.
      :type y: np.ndarray
      :param mus: A list including `self.n_par` lists - one for each parameter of the distribution. Each of those lists contains a numpy array of shape (-1,1) holding the expected value for a particular parmeter for each of the N observations.
      :type mus: [np.ndarray]
      :return: The log-probability of observing all data under the current model.
      :rtype: float
      """
      pass
   
   def lp(self,y:np.ndarray,*mus:list[np.ndarray]) -> np.ndarray:
      """Log-probability of observing every element in :math:`\\mathbf{y}` under their respective distribution parameterized by ``mus``.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array of shape (-1,1) containing each observed value.
      :type y: np.ndarray
      :param mus: A list including `self.n_par` lists - one for each parameter of the distribution. Each of those lists contains a numpy array of shape (-1,1) holding the expected value for a particular parmeter for each of the N observations.
      :type mus: [np.ndarray]
      :return: a N-dimensional vector of shape (-1,1) containing the log-probability of observing each data-point under the current model.
      :rtype: np.ndarray
      """
      pass

   def get_resid(self,y:np.ndarray,*mus:list[np.ndarray],**kwargs) -> np.ndarray|None:
      """Get standardized residuals for a GAMMLSS model (Rigby & Stasinopoulos, 2005).

      Any implementation of this function should return a vector that looks like what could be expected from taking ``len(y)`` independent draws from :math:`N(0,1)`.
      Any additional arguments required by a specific implementation can be passed along via ``kwargs``.

      **Note**: Families for which no residuals are available can return None.

      References:
       - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, Scale and Shape.
       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array of shape (-1,1) containing each observed value.
      :type y: np.ndarray
      :param mus: A list including `self.n_par` lists - one for each parameter of the distribution. Each of those lists contains a numpy array of shape (-1,1) holding the expected value for a particular parmeter for each of the N observations.
      :type mus: [np.ndarray]
      :return: a vector of shape (-1,1) containing standardized residuals under the current model or None in case residuals are not readily available.
      :rtype: np.ndarray | None
      """
      pass

   def init_coef(self,models:list[Callable]) -> np.ndarray:
      """(Optional) Function to initialize the coefficients of the model.

      Can return ``None`` , in which case random initialization will be used.

      :param models: A list of :class:`mssm.models.GAMM`'s, - each based on one of the formulas provided to a model.
      :type models: [mssm.models.GAMM]
      :return: A numpy array of shape (-1,1), holding initial values for all model coefficients.
      :rtype: np.ndarray
      """
      return None
   
   def init_lambda(self,penalties:list[Callable]) -> list[float]:
      """(Optional) Function to initialize the smoothing parameters of the model.

      Can return ``None`` , in which case random initialization will be used.

      :param penalties: A list of all penalties to be estimated by the model.
      :type penalties: [mssm.src.python.penalties.LambdaTerm]
      :return: A list, holding - for each :math:`\\lambda` parameter to be estimated - an initial value.
      :rtype: [float]
      """
      return None


class GAUMLSS(GAMLSSFamily):
   """Family for a Normal GAMMLSS model (Rigby & Stasinopoulos, 2005).

   This Family follows the :class:`Gaussian` family, in that we assume: :math:`Y_i \\sim N(\\mu_i,\\sigma_i)`. i.e., each of the :math:`N` observations
   is still believed to have been generated from an independent normally distributed RV with observation-specific mean.
   
   The important difference is that the scale parameter, :math:`\\sigma`, is now also observation-specific and modeled as an additive combination
   of smooth functions and other parametric terms, just like the mean is in a Normal GAM. Note, that this explicitly models heteroscedasticity -
   the residuals are no longer assumed to be i.i.d samples from :math:`\\sim N(0,\\sigma)`, since :math:`\\sigma` can now differ between residual realizations.

   References:

    - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, Scale and Shape.
    - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.

   :param links: Link functions for the mean and standard deviation. Standard would be ``links=[Identity(),LOG()]``.
   :type links: [Link]
   """
   def __init__(self, links: list[Link]) -> None:
      super().__init__(2, links)

      # All derivatives taken from gamlss.dist: https://github.com/gamlss-dev/gamlss.dist
      # see also: Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, Scale and Shape.
      self.d1:list[Callable] = [lambda y, mu, sigma: (1/np.power(sigma,2))*(y-mu),lambda y, mu, sigma: (np.power(y-mu,2)-np.power(sigma,2))/(np.power(sigma,3))]
      self.d2:list[Callable] = [lambda y, mu, sigma: -(1/np.power(sigma,2)), lambda y, mu, sigma: -(2/np.power(sigma,2))]
      self.d2m:list[Callable] = [lambda y, mu, sigma: np.zeros_like(y)]
      self.mean_init_fam:Family = Gaussian(link=links[0])
   
   def lp(self,y:np.ndarray,mu:np.ndarray,sigma:np.ndarray) -> np.ndarray:
      """Log-probability of observing every proportion in y under their respective Normal with observation-specific mean and standard deviation.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array containing each observed value.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :param sigma: A numpy array containing the predicted stdandard deviation for the response distribution corresponding to each observation.
      :type sigma: np.ndarray
      :return: a N-dimensional vector containing the log-probability of observing each data-point under the current model.
      :rtype: np.ndarray
      """
      return scp.stats.norm.logpdf(y,loc=mu,scale=sigma)
   
   def llk(self,y:np.ndarray,mu:np.ndarray,sigma:np.ndarray) -> float:
      """log-probability of data under given model. Essentially sum over all elements in the vector returned by the :func:`lp` method.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array containing each observation.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :param sigma: A numpy array containing the predicted stdandard deviation for the response distribution corresponding to each observation.
      :type sigma: np.ndarray
      :return: The log-probability of observing all data under the current model.
      :rtype: float
      """
      return sum(self.lp(y,mu,sigma))[0]
   
   def get_resid(self,y:np.ndarray,mu:np.ndarray,sigma:np.ndarray) -> float:
      """Get standardized residuals for a Normal GAMMLSS model (Rigby & Stasinopoulos, 2005).
      
      Essentially, each residual should reflect a realization of a normal with mean zero and observation-specific standard deviation.
      After scaling each residual by their observation-specific standard deviation we should end up with standardized
      residuals that can be expected to be i.i.d :math:`\\sim N(0,1)` - assuming that our model is correct.

      :param y: A numpy array containing each observation.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :param sigma: A numpy array containing the predicted stdandard deviation for the response distribution corresponding to each observation.
      :type sigma: np.ndarray
      :return: A list of standardized residuals that should be ~ N(0,1) if the model is correct.
      :rtype: np.ndarray
      """
      res = y - mu
      res /= sigma
      return res
   
   def init_coef(self, models:list[Callable]) -> np.ndarray:
      """Function to initialize the coefficients of the model.

      Fits a GAMM for the mean and initializes all coef. for the standard deviation to 1.

      :param models: A list of :class:`mssm.models.GAMM`'s, - each based on one of the formulas provided to a model.
      :type models: [mssm.models.GAMM]
      :return: A numpy array of shape (-1,1), holding initial values for all model coefficients.
      :rtype: np.ndarray
      """
      
      mean_model = models[0]
      mean_model.family = Gaussian(self.links[0])
      mean_model.fit(progress_bar=False)

      m_coef,_ = mean_model.get_pars()
      coef = np.concatenate((m_coef.reshape(-1,1),np.ones((models[1].formulas[0].n_coef)).reshape(-1,1)))

      return coef

class MULNOMLSS(GAMLSSFamily):
   """Family for a Multinomial GAMMLSS model (Rigby & Stasinopoulos, 2005).

   This Family assumes that each observation :math:`y_i` corresponds to one of :math:`K` classes (labeled as 0, ..., :math:`K`) and reflects a
   realization of an independent RV :math:`Y_i` with observation-specific probability mass function defined over the :math:`K` classes. These :math:`K`
   probabilities - that :math:`Y_i` takes on class 1, ..., :math:`K` - are modeled as additive combinations of smooth functions of covariates and
   other parametric terms.

   As an example, consider a visual search experiment where :math:`K-1` distractors are presented on a computer screen together with
   a single target and subjects are instructed to find the target and fixate it. With a Multinomial model we can estimate how
   the probability of looking at each of the :math:`K` stimuli on the screen changes (smoothly) over time and as a function of other
   predictor variables of interest (e.g., contrast of stimuli, dependening on whether parfticipants are instructed to be fast or accurate).

   References:

    - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, Scale and Shape.
    - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.

   :param pars: K-1, i.e., 1- Number of classes or the number of linear predictors.
   :type pars: int
   """
   def __init__(self, pars: int) -> None:
      super().__init__(pars, [LOG() for _ in range(pars)])

      # All derivatives taken from gamlss.r in mgcv: https://github.com/cran/mgcv/blob/master/R/gamlss.r#L1224 and have been adapted to work in Python code
      # see also: Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, Scale and Shape.
      self.d_eta:bool = True # Derivatives below are implemented with respect to linear predictor.

      self.d1:list[Callable] = []
      for ii in range(self.n_par):
         def d1(y,*mus,i=ii):
            dy1 = -(mus[i]/(np.sum(mus,axis=0)+1))
            dy1[y == (i+1)] += 1
            return dy1
         self.d1.append(d1)

      self.d2:list[Callable] = []
      for ii in range(self.n_par):
         def d2(y,*mus,i=ii):
            norm = np.sum(mus,axis=0)+1
            dy1 = -(mus[i]/norm)
            dy2 = dy1 + np.power(mus[i],2)/np.power(norm,2)
            return dy2
         self.d2.append(d2)

      self.d2m:list[Callable] = []
      for ii in range(self.n_par):
         for jj in range(ii+1,self.n_par):
            def d2m(y,*mus,i=ii,j=jj):
               norm = np.sum(mus,axis=0)+1
               dy2m = (mus[i]*mus[j])/np.power(norm,2)
               return dy2m
            self.d2m.append(d2m)

   def lp(self, y:np.ndarray, *mus:list[np.ndarray]) -> np.ndarray:
      """Log-probability of observing class k under current model.

      Our DV consists of K classes but we essentially enforce a sum-to zero constraint on the DV so that we end up modeling only
      K-1 (non-normalized) probabilities of observing class k (for all k except k==K) as an additive combination of smooth functions of our
      covariates and other parametric terms. The probability of observing class K as well as the normalized probabilities of observing each
      other class can readily be computed from these K-1 non-normalized probabilities. This is explained quite well on Wikipedia (see refs).

      Specifically, the probability of the outcome being class k is simply:

      :math:`p(Y_i == k) = \\mu_k / (1 + \\sum_j^{K-1} \\mu_j)` where :math:`\\mu_k` is the aforementioned non-normalized probability of observing class :math:`k` - which is simply set to 1 for class :math:`K` (this follows from the sum-to-zero constraint; see Wikipedia).

      So, the log-prob of the outcome being class k is:

      :math:`log(p(Y_i == k)) = log(\\mu_k) - log(1 + \\sum_j^{K-1} \\mu_j)`

      References:

       - Wikipedia. https://en.wikipedia.org/wiki/Multinomial_logistic_regression
       - gamlss.dist on Github (see Rigby & Stasinopoulos, 2005). https://github.com/gamlss-dev/gamlss.dist/blob/main/R/MN4.R
       - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, Scale and Shape.

      :param y: A numpy array containing each observed class, every element must be larger than or equal to 0 and smaller than `self.n_par + 1`.
      :type y: np.ndarray
      :param mus: A list containing K-1 (`self.n_par`) lists, each containing the non-normalized probabilities of observing class k for every observation.
      :type mus: [np.ndarray]
      :return: a N-dimensional vector containing the log-probability of observing each data-point under the current model.
      :rtype: np.ndarray
      """
      # Note, log(1) = 0, so we can simply initialize to -log(1 + \\sum_j^{K-1} mu_j)
      # and then add for the K-1 probs we actually modeled.
      lp = -np.log(np.sum(mus,axis=0)+1)

      for pi in range(self.n_par):
         lp[y == (pi+1)] += np.log(mus[pi])[y == (pi+1)]

      return lp

   def llk(self, y:np.ndarray, *mus:list[np.ndarray]):
      """log-probability of data under given model. Essentially sum over all elements in the vector returned by the :func:`lp` method.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array containing each observed class, every element must be larger than or equal to 0 and smaller than `self.n_par + 1`.
      :type y: np.ndarray
      :param mus: A list containing K-1 (`self.n_par`) lists, each containing the non-normalized probabilities of observing class k for every observation.
      :type mus: [np.ndarray]
      :return: The log-probability of observing all data under the current model.
      :rtype: float
      """
      return sum(self.lp(y,*mus))[0]
   
   def get_resid(self,y:np.ndarray,*mus:list[np.ndarray]) -> None:
      """Placeholder function for residuals of a Multinomial model - yet to be implemented.

      :param y: A numpy array containing each observed class, every element must be larger than or equal to 0 and smaller than `self.n_par + 1`.
      :type y: np.ndarray
      :param mus: A list containing K-1 (`self.n_par`) lists, each containing the non-normalized probabilities of observing class k for every observation.
      :type mus: [np.ndarray]
      :return: Currently None - since no residuals are implemented
      """
      warnings.warn("Getting residuals for multinomial model are currently not supported.")
      return None

class GAMMALS(GAMLSSFamily):
   """Family for a GAMMA GAMMLSS model (Rigby & Stasinopoulos, 2005).

   This Family follows the :class:`Gamma` family, in that we assume: :math:`Y_i \\sim \\Gamma(\\mu_i,\\phi_i)`. The difference to the :class:`Gamma` family
   is that we now also model :math:`\\phi` as an additive combination of smooth variables and other parametric terms. The Gamma distribution is usually
   not expressed in terms of the mean and scale (:math:`\\phi`) parameter but rather in terms of a shape and rate parameter - called :math:`\\alpha` and :math:`\\beta`
   respectively. Wood (2017) provides :math:`\\alpha = 1/\\phi`. With this we can obtain :math:`\\beta = 1/\\phi/\\mu` (see the source-code for :func:`lp` method
   of the :class:`Gamma` family for details).

   References:

    - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, Scale and Shape.
    - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.

   :param links: Link functions for the mean and standard deviation. Standard would be ``links=[LOG(),LOG()]``.
   :type links: [Link]
   """

   def __init__(self,links: list[Link]) -> None:
      super().__init__(2, links)
      # All derivatives based on gamlss.dist: https://github.com/gamlss-dev/gamlss.dist, but adjusted so that \\phi (the scale) is \\sigma^2.
      # see also: Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, Scale and Shape.
      self.d1:list[Callable] = [lambda y, mu, scale: (y-mu)/(scale*np.power(mu,2)),lambda y, mu, scale: (2/(scale*np.sqrt(scale)))*((y/mu)-np.log(y)+np.log(mu)+np.log(scale)-1+scp.special.digamma(1/(scale)))]
      self.d2:list[Callable] = [lambda y, mu, scale:  -1/(scale*np.power(mu,2)), lambda y, mu, scale: (4/np.power(scale,2))-(4/np.power(scale,3))*scp.special.polygamma(1,1/scale)]
      self.d2m:list[Callable] = [lambda y, mu, scale: np.zeros_like(y)]
   
   def lp(self,y:np.ndarray,mu:np.ndarray,scale:np.ndarray) -> np.ndarray:
      """Log-probability of observing every proportion in :math:`\\mathbf{y}` under their respective Gamma with mean = :math:`\\boldsymbol{\\mu}` and scale = :math:`\\boldsymbol{\\phi}`.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array containing each observed value.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :param scale: A numpy array containing the predicted scale parameter for the response distribution corresponding to each observation.
      :type scale: np.ndarray
      :return: a N-dimensional vector containing the log-probability of observing each data-point under the current model.
      :rtype: np.ndarray
      """
      # Need to transform from mean and scale to \alpha & \beta
      # From Wood (2017), we have that
      # \\phi = 1/\alpha
      # so \alpha = 1/\\phi
      # From https://en.wikipedia.org/wiki/Gamma_distribution, we have that:
      # \\mu = \alpha/\beta
      # \\mu = 1/\\phi/\beta
      # \beta = 1/\\phi/\\mu
      # scipy docs, say to set scale to 1/\beta.
      # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
      alpha = 1/scale
      beta = alpha/mu  
      return scp.stats.gamma.logpdf(y,a=alpha,scale=(1/beta))
   
   def llk(self,y:np.ndarray,mu:np.ndarray,scale:np.ndarray) -> float:
      """log-probability of data under given model. Essentially sum over all elements in the vector returned by the :func:`lp` method.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param y: A numpy array containing each observation.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :param scale: A numpy array containing the predicted scale parameter for the response distribution corresponding to each observation.
      :type scale: np.ndarray
      :return: The log-probability of observing all data under the current model.
      :rtype: float
      """
      return sum(self.lp(y,mu,scale))[0]
   
   def get_resid(self,y:np.ndarray,mu:np.ndarray,scale:np.ndarray) -> np.ndarray:
      """Get standardized residuals for a Gamma GAMMLSS model (Rigby & Stasinopoulos, 2005).
      
      Essentially, to get a standaridzed residual vector we first have to account for the mean-variance relationship of our RVs
      (which we also have to do for the :class:`Gamma` family) - for this we can simply compute deviance residuals again (see Wood, 2017).
      These should be :math:`\\sim N(0,\\phi_i)` (where :math:`\\phi_i` is the element in ``scale`` for a specific observation) - so if we divide each of those by the observation-specific scale we can expect the resulting
      standardized residuals to be :math:` \\sim N(0,1)` if the model is correct.

      References:

       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).


      :param y: A numpy array containing each observation.
      :type y: np.ndarray
      :param mu: A numpy array containing the predicted mean for the response distribution corresponding to each observation.
      :type mu: np.ndarray
      :param scale: A numpy array containing the predicted scale parameter for the response distribution corresponding to each observation.
      :type scale: np.ndarray
      :return: A list of standardized residuals that should be ~ N(0,1) if the model is correct.
      :rtype: np.ndarray
      """
      res = np.sign(y - mu) * np.sqrt(Gamma().D(y,mu)/scale)
      return res
   
   def init_coef(self, models:list[Callable]) -> np.ndarray:
      """Function to initialize the coefficients of the model.

      Fits a GAMM for the mean and initializes all coef. for the scale parameter to 1.

      :param models: A list of :class:`mssm.models.GAMM`'s, - each based on one of the formulas provided to a model.
      :type models: [mssm.models.GAMM]
      :return: A numpy array of shape (-1,1), holding initial values for all model coefficients.
      :rtype: np.ndarray
      """
      
      mean_model = models[0]
      mean_model.family = Gamma(self.links[0])
      mean_model.fit(progress_bar=False,max_inner=1)

      m_coef,_ = mean_model.get_pars()
      coef = np.concatenate((m_coef.reshape(-1,1),np.ones((models[1].formulas[0].n_coef)).reshape(-1,1)))
      return coef
   

class GSMMFamily:
   """Base-class for General Smooth "families" as discussed by Wood, Pya, & Säfken (2016). For estimation of :class:`mssm.models.GSMM` models via
   ``L-qEFS`` (Krause et al., submitted) it is sufficient to implement :func:`llk`. :func:`gradient` and :func:`hessian` can then simply return ``None``. For exact estimation via
   Newton's method, the latter two functions need to be implemented and have to return the gradient and hessian at the current coefficient estimate
   respectively.

   Additional parameters needed for likelihood, gradient, or hessian evaluation can be passed along via the ``llkargs``. They are then made available in ``self.llkargs``.

   References:
    - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
    - Nocedal & Wright (2006). Numerical Optimization. Springer New York.
    - Krause et al. (submitted). The Mixed-Sparse-Smooth-Model Toolbox (MSSM): Efficient Estimation and Selection of Large Multi-Level Statistical Models. https://doi.org/10.48550/arXiv.2506.13132

   :param pars: Number of parameters of the likelihood.
   :type pars: int
   :param links: List of Link functions for each parameter of the likelihood, e.g., `links=[Identity(),LOG()]`.
   :type links: [Link]
   :ivar int, optional extra_coef: Number of extra coefficients required by specific family or ``None``. By default set to ``None`` and changed to ``int`` by specific families requiring this.
   
   """
   def __init__(self,pars:int,links:list[Link],*llkargs) -> None:
      self.n_par:int = pars
      self.links:list[Link] = links
      self.llkargs = llkargs # Any arguments that need to be passed to evaluate the likelihood/gradiant/hessian
      self.extra_coef:int|None = None

   def llk(self,coef:np.ndarray,coef_split_idx:list[int],ys:list[np.ndarray],Xs:list[scp.sparse.csc_array]) -> float:
      """log-probability of data under given model.

      References:
       - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
       - Krause et al. (submitted). The Mixed-Sparse-Smooth-Model Toolbox (MSSM): Efficient Estimation and Selection of Large Multi-Level Statistical Models. https://doi.org/10.48550/arXiv.2506.13132

      :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not be flattened!).
      :type coef: np.ndarray
      :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the sub-sets associated with each paramter of the llk.
      :type coef_split_idx: [int]
      :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the actual observed data is passed along via the first formula (so it is stored in ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula, then ``ys`` contains ``None`` at their indices to save memory.
      :type ys: [np.ndarray or None]
      :param Xs: A list of sparse model matrices per likelihood parameter.
      :type Xs: [scp.sparse.csc_array]
      :return: The log-likelihood evaluated at ``coef``.
      :rtype: float
      """
      pass
   
   def gradient(self,coef:np.ndarray,coef_split_idx:list[int],ys:list[np.ndarray],Xs:list[scp.sparse.csc_array]) -> np.ndarray:
      """Function to evaluate the gradient of the llk at current coefficient estimate ``coef``.

      By default relies on numerical differentiation as implemented in scipy to approximate the Gradient from the implemented log-likelihood function.
      See the link in the references for more details. 

      References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
         - ``scipy.optimize.approx_fprime``: at https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.approx_fprime.html
         - Krause et al. (submitted). The Mixed-Sparse-Smooth-Model Toolbox (MSSM): Efficient Estimation and Selection of Large Multi-Level Statistical Models. https://doi.org/10.48550/arXiv.2506.13132

      :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not be flattened!).
      :type coef: np.ndarray
      :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the sub-sets associated with each paramter of the llk.
      :type coef_split_idx: [int]
      :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the actual observed data is passed along via the first formula (so it is stored in ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula, then ``ys`` contains ``None`` at their indices to save memory.
      :type ys: [np.ndarray or None]
      :param Xs: A list of sparse model matrices per likelihood parameter.
      :type Xs: [scp.sparse.csc_array]
      :return: The Gradient of the log-likelihood evaluated at ``coef`` as numpy array of shape (-1,1).
      :rtype: np.ndarray
      """
      llk_warp = lambda x: self.llk(x.reshape(-1,1),coef_split_idx,ys,Xs)
      
      grad = scp.optimize.approx_fprime(coef.flatten(),llk_warp)
      return grad.reshape(-1,1)
   
   def hessian(self,coef:np.ndarray,coef_split_idx:list[int],ys:list[np.ndarray],Xs:list[scp.sparse.csc_array]) -> scp.sparse.csc_array | None:
      """Function to evaluate the hessian of the llk at current coefficient estimate ``coef``.

      Only has to be implemented if full Newton is to be used to estimate coefficients. If the L-qEFS update by Krause et al. (in preparation) is
      to be used insetad, this method does not have to be implemented.

      References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
         - ``scipy.optimize.approx_fprime``: at https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.approx_fprime.html
         - Krause et al. (submitted). The Mixed-Sparse-Smooth-Model Toolbox (MSSM): Efficient Estimation and Selection of Large Multi-Level Statistical Models. https://doi.org/10.48550/arXiv.2506.13132

      :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not be flattened!).
      :type coef: np.ndarray
      :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the sub-sets associated with each paramter of the llk.
      :type coef_split_idx: [int]
      :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the actual observed data is passed along via the first formula (so it is stored in ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula, then ``ys`` contains ``None`` at their indices to save memory.
      :type ys: [np.ndarray or None]
      :param Xs: A list of sparse model matrices per likelihood parameter.
      :type Xs: [scp.sparse.csc_array]
      :return: The Hessian of the log-likelihood evaluated at ``coef``.
      :rtype: scp.sparse.csc_array
      """
      return None
   
   def get_resid(self,coef:np.ndarray,coef_split_idx:list[int],ys:list[np.ndarray],Xs:list[scp.sparse.csc_array],**kwargs) -> np.ndarray | None:
      """Get standardized residuals for a GSMM model.

      Any implementation of this function should return a vector that looks like what could be expected from taking independent draws from :math:`N(0,1)`.
      Any additional arguments required by a specific implementation can be passed along via ``kwargs``.

      **Note**: Families for which no residuals are available can return None.

      References:
       - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
       - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not be flattened!).
      :type coef: np.ndarray
      :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the sub-sets associated with each paramter of the llk.
      :type coef_split_idx: [int]
      :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the actual observed data is passed along via the first formula (so it is stored in ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula, then ``ys`` contains ``None`` at their indices to save memory.
      :type ys: [np.ndarray or None]
      :param Xs: A list of sparse model matrices per likelihood parameter.
      :type Xs: [scp.sparse.csc_array]
      :return: a vector of shape (-1,1) containing standardized residuals under the current model (**Note**, the first axis will not necessarily match the dimension of any of the response vectors (this will depend on the specific Family's implementation)) or None in case residuals are not readily available.
      :rtype: np.ndarray | None
      """
      pass
   
   def init_coef(self,models:list[Callable]) -> np.ndarray:
      """(Optional) Function to initialize the coefficients of the model.

      Can return ``None`` , in which case random initialization will be used.

      :param models: A list of :class:`mssm.models.GAMM`'s, - each based on one of the formulas provided to a model.
      :type models: [mssm.models.GAMM]
      :return: A numpy array of shape (-1,1), holding initial values for all model coefficients.
      :rtype: np.ndarray
      """
      return None
   
   def init_lambda(self,penalties:list[Callable]) -> list[float]:
      """(Optional) Function to initialize the smoothing parameters of the model.

      Can return ``None`` , in which case random initialization will be used.

      :param penalties: A list of all penalties to be estimated by the model.
      :type penalties: [mssm.src.python.penalties.LambdaTerm]
      :return: A list, holding - for each :math:`\\lambda` parameter to be estimated - an initial value.
      :rtype: np.ndarray
      """
      return None
   
class PropHaz(GSMMFamily):
   """Family for proportional Hazard model - a type of General Smooth model as discussed by Wood, Pya, & Säfken (2016).
   
   Based on Supplementary materials G in Wood, Pya, & Säfken (2016). The dependent variable passed to the :class:`mssm.src.python.formula.Formula`
   needs to hold ``delta`` indicating whether the event was observed or not (i.e., only values in ``{0,1}``).

   Examples::

      from mssm.models import *
      from mssmViz.sim import *
      from mssmViz.plot import *
      import matplotlib.pyplot as plt

      # Simulate some data
      sim_dat = sim3(500,2,c=1,seed=0,family=PropHaz([0],[0]),binom_offset = 0.1,correlate=False)
            
      # Prep everything for prophaz model
      sim_dat = sim_dat.sort_values(['y'],ascending=[False])
      sim_dat = sim_dat.reset_index(drop=True)
      print(sim_dat.head(),np.mean(sim_dat["delta"]))

      u,inv = np.unique(sim_dat["y"],return_inverse=True)
      ut = np.flip(u)
      r = np.abs(inv - max(inv))

      # Now specify formula and model
      sim_formula_m = Formula(lhs("delta"),
                              [f(["x0"]),f(["x1"]),f(["x2"]),f(["x3"])],
                              data=sim_dat)

      PropHaz_fam = PropHaz(ut,r)
      model = GSMM([copy.deepcopy(sim_formula_m)],PropHaz_fam)

      # Fit with Newton
      model.fit()

      # Can plot the estimated effects on the scale of the linear predictor (i.e., log hazard) via mssmViz
      plot(model)

   References:
    - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
    - Nocedal & Wright (2006). Numerical Optimization. Springer New York.

   :param ut: Unique event time vector (each time represnted as ``int``) as described by WPS (2016), holding unique event times in decreasing order.
   :type ut: np.ndarray
   :param r: Index vector as described by WPS (2016), holding for each data-point (i.e., for each row in ``Xs[0``]) the index to it's corresponding event time in ``ut``.
   :type r: np.ndarray
   
   """

   def __init__(self, ut:np.ndarray, r:np.ndarray):
      super().__init__(1, [Identity()], ut, r)
      self.__hs = None
      self.__qs = None
      self.__avs = None
   
   def llk(self,coef:np.ndarray,coef_split_idx:list[int],ys:list[np.ndarray],Xs:list[scp.sparse.csc_array]) -> float:
      """Log-likelihood function as defined by Wood, Pya, & Säfken (2016).

      References:
       - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.

      :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not be flattened!).
      :type coef: np.ndarray
      :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the sub-sets associated with each paramter of the llk - not required by this family, which has a single parameter.
      :type coef_split_idx: [int]
      :param ys: List containing the ``delta`` vector at the first and only index - see description of the model family.
      :type ys: [np.ndarray]
      :param Xs: A list containing the sparse model matrix at the first and only index.
      :type Xs: [scp.sparse.csc_array]
      :return: The log-likelihood evaluated at ``coef``.
      :rtype: float
      """
      
      # Extract and define all variables defined by WPS (2016)
      delta = ys[0]
      ut = self.llkargs[0]
      r = self.llkargs[1]
      nt = len(ut)
      X = Xs[0]
      eta = X@coef

      with warnings.catch_warnings(): # Overflow
         warnings.simplefilter("ignore")
         gamma = np.exp(eta)

      # Now compute first sum
      llk = np.sum(delta*eta)

      # and second sum
      gamma_p = 0
      for j in range(nt):
         ri = r == j
         dj = np.sum(delta[ri])
         with warnings.catch_warnings(): # Overflow
            warnings.simplefilter("ignore")
            gamma_p += np.sum(gamma[ri])

         with warnings.catch_warnings(): # Divide by zero
            warnings.simplefilter("ignore")
            log_gamma_p = np.log(gamma_p)
         

         llk -= dj*log_gamma_p

      return llk

   def gradient(self, coef:np.ndarray,coef_split_idx:list[int],ys:list[np.ndarray],Xs:list[scp.sparse.csc_array]) -> np.ndarray:
      """Gradient as defined by Wood, Pya, & Säfken (2016).

      References:
       - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.

      :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not be flattened!).
      :type coef: np.ndarray
      :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the sub-sets associated with each paramter of the llk - not required by this family, which has a single parameter.
      :type coef_split_idx: [int]
      :param ys: List containing the ``delta`` vector at the first and only index - see description of the model family.
      :type ys: [np.ndarray]
      :param Xs: A list containing the sparse model matrix at the first and only index.
      :type Xs: [scp.sparse.csc_array]
      :return: The Gradient of the log-likelihood evaluated at ``coef`` as numpy array of shape (-1,1).
      :rtype: np.ndarray
      """
      
      # Extract and define all variables defined by WPS (2016)
      delta = ys[0]
      ut = self.llkargs[0]
      r = self.llkargs[1]
      nt = len(ut)
      X = Xs[0]
      eta = X@coef

      with warnings.catch_warnings(): # Overflow
         warnings.simplefilter("ignore")
         gamma = np.exp(eta)
      gamma = gamma.reshape(-1,1)

      # Now compute first sum
      g = delta.T@X

      # and second sum
      b_p = np.zeros_like(g)
      
      gamma_p = 0
      for j in range(nt):
         ri = r == j
         dj = np.sum(delta[ri])
         gamma_i = (gamma[ri,0]).reshape(-1,1)
         with warnings.catch_warnings(): # Overflow
            warnings.simplefilter("ignore")
            gamma_p += np.sum(gamma_i)

         X_i = X[ri,:]
         bi = gamma_i.T@X_i
         b_p += bi
         
         with warnings.catch_warnings(): # Divide by zero
            warnings.simplefilter("ignore")
            bpg = b_p/gamma_p

         g -= dj*bpg
      
      return g.reshape(-1,1)


   def hessian(self, coef:np.ndarray,coef_split_idx:list[int],ys:list[np.ndarray],Xs:list[scp.sparse.csc_array]) -> scp.sparse.csc_array:
      """Hessian as defined by Wood, Pya, & Säfken (2016).

      References:
       - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.

      :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not be flattened!).
      :type coef: np.ndarray
      :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the sub-sets associated with each paramter of the llk - not required by this family, which has a single parameter.
      :type coef_split_idx: [int]
      :param ys: List containing the ``delta`` vector at the first and only index - see description of the model family.
      :type ys: [np.ndarray]
      :param Xs: A list containing the sparse model matrix at the first and only index.
      :type Xs: [scp.sparse.csc_array]
      :return: The Hessian of the log-likelihood evaluated at ``coef``.
      :rtype: scp.sparse.csc_array
      """

      # Extract and define all variables defined by WPS (2016)
      delta = ys[0]
      ut = self.llkargs[0]
      r = self.llkargs[1]
      nt = len(ut)
      X = Xs[0]
      eta = X@coef
      
      with warnings.catch_warnings(): # Overflow
         warnings.simplefilter("ignore")
         gamma = np.exp(eta)
      gamma = gamma.reshape(-1,1)

      # Only sum over nt
      b_p = np.zeros((1,X.shape[1]))

      gamma_p = 0
      A_p = scp.sparse.csc_array((X.shape[1],X.shape[1]))
      H = scp.sparse.csc_array((X.shape[1],X.shape[1]))
      for j in range(nt):
         ri = r == j
         dj = np.sum(delta[ri])
         gamma_i = (gamma[ri,0]).reshape(-1,1)
         gamma_p += np.sum(gamma_i)

         X_i = X[ri,:]
         bi = gamma_i.T@X_i
         b_p += bi
         
         A_i = (gamma_i * X_i).T@X_i
         A_p += A_i
         

         with warnings.catch_warnings(): # Divide by zero or overflow
            warnings.simplefilter("ignore")
            Hj = dj*b_p.T@b_p/np.power(gamma_p,2) - dj*A_p/gamma_p
         
         H += Hj

      return scp.sparse.csc_array(H)
   
   def get_resid(self, coef, coef_split_idx, ys, Xs, resid_type:str="Martingale", reorder:np.ndarray|None=None) -> np.ndarray:
      """Get Martingale or Deviance residuals for a proportional Hazard model.

      See the :func:`PropHaz.get_survival` function for examples.

      References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

      :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not be flattened!).
      :type coef: np.ndarray
      :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the sub-sets associated with each paramter of the llk - not required by this family, which has a single parameter.
      :type coef_split_idx: [int]
      :param ys: List containing the ``delta`` vector at the first and only index - see description of the model family.
      :type ys: [np.ndarray]
      :param Xs: A list containing the sparse model matrix at the first and only index.
      :type Xs: [scp.sparse.csc_array]
      :param resid_type: The type of residual to compute, supported are "Martingale" and "Deviance".
      :type resid_type: str, optional
      :param reorder: A flattened np.ndarray containing for each data point the original index in the data-set before sorting. Used to re-order the residual vector into the original order. If this is set to None, the residual vector is not re-ordered and instead returned in the order of the sorted data-frame passed to the model formula.
      :type reorder: np.ndarray
      :return: The residual vector of shape (-1,1)
      :rtype: np.ndarray
      """

      if resid_type not in ["Martingale","Deviance"]:
         raise ValueError("`resid_type` must be one of 'Martingale' or 'Deviance'.")

      # Extract all quantities needed to evaluate residuals
      delta = ys[0]
      ut = self.llkargs[0]
      r = self.llkargs[1]
      X = Xs[0]

      # Following based on derivation by Wood, Pya, and Säfken (2016)
      res = np.zeros(X.shape[0])
      for idx, tidx in enumerate(r):
         Xi = X[idx,:].toarray()
         ti = ut[tidx]
         di = delta[idx]
         Si,_ = self.get_survival(coef,Xs,delta,ti,Xi,None,compute_var=False)
         mi = di + np.log(Si[0])

         if resid_type == "Martingale":
            res[idx] = mi
         else:
            # Deviance requires a bit more work
            Di = np.sign(mi) * np.power(-2*(mi + di*np.log(-min(np.log(Si[0]),-np.finfo(float).eps))),0.5)
            res[idx] = Di

      # Return to order of original dataframe
      if reorder is not None:
         res = res[reorder]
      
      return res.reshape(-1,1)
   
   def __prepare_predictions(self,coef:np.ndarray,delta:np.ndarray,Xs:list[scp.sparse.csc_array]) -> None:
    """Computes all the quantities defined by Wood, Pya, & Säfken (2016) that are necessary for predictions.

    This includes the cumulative base-line hazard, as well as the :math`\\mathbf{a}` vectors from WPS (2016). These are assigned to the instance of this family.
    
    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.

    :param coef: Coefficient vector as numpy array of shape (-1,1).
    :type coef: np.ndarray
    :param delta: Dependent variable passed to :func:`mssm.src.python.formula.Formula`, holds (for each row in ``Xs[0``]) a value in ``{0,1}``, indicating whether for that observation the event was observed or not.
    :type delta: np.ndarray
    :param Xs The list model matrices (here holding a single model matrix) obtained from :func:`mssm.models.GAMMLSS.get_mmat`.
    :type Xs: [scp.sparse.csc_array]
    """
    # Extract and define all variables defined by WPS (2016)
    ut = self.llkargs[0]
    r = self.llkargs[1]
    nt = len(ut)
    X = Xs[0]
    eta = X@coef

    with warnings.catch_warnings(): # Overflow
        warnings.simplefilter("ignore")
        gamma = np.exp(eta)

    gamma[np.isnan(gamma) | np.isinf(gamma)] = np.power(np.finfo(float).max,0.9)
    gamma = gamma.reshape(-1,1)

    # We need gamma_ps, b_ps, and djs
    gamma_ps = []
    djs = []
    b_ps = []

    gamma_p = 0
    b_p = np.zeros((1,X.shape[1]))
    for j in range(nt):
        ri = r == j

        # Get dj
        dj = np.sum(delta[ri])
        djs.append(dj)
        
        # gamma_p
        gamma_i = (gamma[ri,0]).reshape(-1,1)
        with warnings.catch_warnings(): # Overflow
            warnings.simplefilter("ignore")
            gamma_p += np.sum(gamma_i)

        if np.isnan(gamma_p) | np.isinf(gamma_p):
            gamma_p = np.power(np.finfo(float).max,0.9)
        gamma_ps.append(gamma_p)
        
        # b_p vector
        X_i = X[ri,:]
        bi = gamma_i.T@X_i
        b_p += bi
        b_ps.append(copy.deepcopy(b_p))
    
    # Now base-line hazard + variance and a vectors
    hs = np.zeros(nt)
    qs = np.zeros(nt)
    avs = [np.zeros_like(b_p) for _ in range(nt)]

    hs[-1] = djs[-1]/gamma_ps[-1]
    qs[-1] = djs[-1]/np.power(gamma_ps[-1],2)
    avs[-1] = b_ps[-1] * djs[-1]/np.power(gamma_ps[-1],2)
    #print(hs[-1],qs[-1])

    for j in range(nt-2,-1,-1):
        #print(j,hs[j+1])
        hs[j] = hs[j+1] + djs[j]/gamma_ps[j]
        qs[j] = qs[j+1] + djs[j]/np.power(gamma_ps[j],2)
        avs[j] = avs[j+1] + b_ps[j] * djs[j]/np.power(gamma_ps[j],2)
    
    self.__hs = hs
    self.__qs = qs
    self.__avs = avs

   
   def get_baseline_hazard(self,coef:np.ndarray,delta:np.ndarray,Xs:list[scp.sparse.csc_array]) -> np.ndarray:
    """Get the cumulative baseline hazard function as defined by Wood, Pya, & Säfken (2016).

    The function is evaluated for all ``k`` unique event times that were available in the data.

    Examples::

      from mssm.models import *
      from mssmViz.sim import *
      from mssmViz.plot import *
      import matplotlib.pyplot as plt

      # Simulate some data
      sim_dat = sim3(500,2,c=1,seed=0,family=PropHaz([0],[0]),binom_offset = 0.1,correlate=False)
            
      # Prep everything for prophaz model
      sim_dat = sim_dat.sort_values(['y'],ascending=[False])
      sim_dat = sim_dat.reset_index(drop=True)
      print(sim_dat.head(),np.mean(sim_dat["delta"]))

      u,inv = np.unique(sim_dat["y"],return_inverse=True)
      ut = np.flip(u)
      r = np.abs(inv - max(inv))

      # Now specify formula and model
      sim_formula_m = Formula(lhs("delta"),
                              [f(["x0"]),f(["x1"]),f(["x2"]),f(["x3"])],
                              data=sim_dat)

      PropHaz_fam = PropHaz(ut,r)
      model = GSMM([copy.deepcopy(sim_formula_m)],PropHaz_fam)

      # Fit with Newton
      model.fit()

      # Now get cumulative baseline hazard estimate
      H = PropHaz_fam.get_baseline_hazard(model.coef,sim_formula_m.y_flat[sim_formula_m.NOT_NA_flat],model.get_mmat())

      # And plot it
      plt.plot(ut,H)
      plt.xlabel("Time")
      plt.ylabel("Cumulative Baseline Hazard")

    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.

    :param coef: Coefficient vector as numpy array of shape (-1,1).
    :type coef: np.ndarray
    :param Xs: The list model matrices (here holding a single model matrix) obtained from :func:`mssm.models.GAMMLSS.get_mmat`.
    :type Xs: [scp.sparse.csc_array]
    :param delta: Dependent variable passed to :func:`mssm.src.python.formula.Formula`, holds (for each row in ``Xs[0``]) a value in ``{0,1}``, indicating whether for that observation the event was observed or not.
    :type delta: np.ndarray
    :return: numpy array, holding ``k`` baseline hazard function estimates
    :rtype: np.ndarray
    """

    if self.__hs is None:
       self.__prepare_predictions(coef,delta,Xs)
    
    return self.__hs

   def get_survival(self,coef:np.ndarray,Xs:list[scp.sparse.csc_array],delta:np.ndarray,t:int,x:np.ndarray|scp.sparse.csc_array,V:scp.sparse.csc_array,compute_var:bool=True) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute survival function + variance at time-point ``t``, given ``k`` optional covariate vector(s) x as defined by Wood, Pya, & Säfken (2016).

    Examples::

      from mssm.models import *
      from mssmViz.sim import *
      from mssmViz.plot import *
      import matplotlib.pyplot as plt

      # Simulate some data
      sim_dat = sim3(500,2,c=1,seed=0,family=PropHaz([0],[0]),binom_offset = 0.1,correlate=False)
            
      # Prep everything for prophaz model

      # Create index variable for residual ordering
      sim_dat["index"] = np.arange(sim_dat.shape[0])

      # Now sort
      sim_dat = sim_dat.sort_values(['y'],ascending=[False])
      sim_dat = sim_dat.reset_index(drop=True)
      print(sim_dat.head(),np.mean(sim_dat["delta"]))

      u,inv = np.unique(sim_dat["y"],return_inverse=True)
      ut = np.flip(u)
      r = np.abs(inv - max(inv))
      res_idx = np.argsort(sim_dat["index"].values)

      # Now specify formula and model
      sim_formula_m = Formula(lhs("delta"),
                              [f(["x0"]),f(["x1"]),f(["x2"]),f(["x3"])],
                              data=sim_dat)

      PropHaz_fam = PropHaz(ut,r)
      model = GSMM([copy.deepcopy(sim_formula_m)],PropHaz_fam)

      # Fit with Newton
      model.fit()

      # Now get estimate of survival function and see how it changes with x0
      new_dat = pd.DataFrame({"x0":np.linspace(0,1,5),
                              "x1":np.linspace(0,1,5),
                              "x2":np.linspace(0,1,5),
                              "x3":np.linspace(0,1,5)})

      # Get model matrix using only f0
      _,Xt,_ = model.predict(use_terms=[0],n_dat=new_dat)

      # Now iterate over all time-points and obtain the predicted survival function + standard error estimate
      # for all 5 values of x0:
      S = np.zeros((len(ut),Xt.shape[0]))
      VS = np.zeros((len(ut),Xt.shape[0]))
      for idx,ti in enumerate(ut):

         # Su and VSu are of shape (5,1) here but will generally be of shape (Xt.shape[0],1)
         Su,VSu = PropHaz_fam.get_survival(model.coef,model.get_mmat(),sim_formula_m.y_flat[sim_formula_m.NOT_NA_flat],
                                          ti,Xt,model.lvi.T@model.lvi)
         S[idx,:] = Su.flatten()
         VS[idx,:] = VSu.flatten()

      # Now we can plot the estimated survival functions + approximate cis:
      for xi in range(Xt.shape[0]):

         plt.fill([*ut,*np.flip(ut)],
                  [*(S[:,xi] + 1.96*VS[:,xi]),*np.flip(S[:,xi] - 1.96*VS[:,xi])],alpha=0.5)
         plt.plot(ut,S[:,xi],label=f"x0 = {new_dat["x0"][xi]}")
      plt.legend()
      plt.xlabel("Time")
      plt.ylabel("Survival")
      plt.show()

      # Note how the main effect of x0 is reflected in the plot above:
      plot(model,which=[0])

      # Residual plots can be created via `plot_val` from `mssmViz` - by default Martingale residuals are returned (see Wood, 2017)
      fig = plt.figure(figsize=(10,3),layout='constrained')
      axs = fig.subplots(1,3,gridspec_kw={"wspace":0.2})
      # Note the use of `gsmm_kwargs_pred={}` to ensure that the re-ordering is not applied to the plot against predicted values
      plot_val(model,gsmm_kwargs={"reorder":res_idx},gsmm_kwargs_pred={},ar_lag=25,axs=axs)

      # Can also get Deviance residuals:
      fig = plt.figure(figsize=(10,3),layout='constrained')
      axs = fig.subplots(1,3,gridspec_kw={"wspace":0.2})

      plot_val(model,gsmm_kwargs={"reorder":res_idx,"resid_type":"Deviance"},gsmm_kwargs_pred={"resid_type":"Deviance"},ar_lag=25,axs=axs)

    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.

    :param coef: Coefficient vector as numpy array of shape (-1,1).
    :type coef: np.ndarray
    :param Xs: The list model matrices (here holding a single model matrix) obtained from :func:`mssm.models.GAMMLSS.get_mmat`.
    :type Xs: [scp.sparse.csc_array]
    :param delta: Dependent variable passed to :func:`mssm.src.python.formula.Formula`, holds (for each row in ``Xs[0``]) a value in ``{0,1}``, indicating whether for that observation the event was observed or not.
    :type delta: np.ndarray
    :param t: Time-point at which to evaluate the survival function.
    :type t: int
    :param x: Optional vector (or matrix - can also be sparse) of covariate values. Needs to be of shape ``(k,len(coef))``.
    :type x: np.ndarray or scp.sparse.csc_array
    :param V: Estimated Co-variance matrix of posterior for ``coef``
    :type V: scp.sparse.csc_array
    :param compute_var: Whether to compue the variance estimate of the survival as well. Otherwise None will be returned as the second argument.
    :type compute_var: bool, optional
    :return: Two arrays, the first holds ``k`` survival function estimates, the latter holds ``k`` variance estimates for each of the survival function estimates. The second argument will be None instead if ``compute_var = False``.
    :rtype: tuple[np.ndarray, np.ndarray | None]
    """

    if self.__hs is None:
       self.__prepare_predictions(coef,delta,Xs)
    
    # Extract and define all variables defined by WPS (2016)
    ut = self.llkargs[0]
    eta = x@coef
    #print(eta)

    # Find nearest larger time-point
    if not t in ut:
        t = min(ut[ut > t])

    # Find index in h corresponding to t
    ti = ut == t
    tiv = np.arange(len(ut))[ti][0]
    #print(t,tiv)
    
    # Compute (log) survival
    lS = -self.__hs[ti] * np.exp(eta)
    S = np.exp(lS)
    
    varS = None
    if compute_var:
      # Compute variance
      v =  - self.__hs[ti]*x + self.__avs[tiv]
      
      varS = np.exp(eta) * S * np.power(self.__qs[ti] + np.sum(v@V * v,axis=1).reshape(-1,1),0.5)
    return S, varS    

   def init_coef(self,models:list[Callable]) -> np.ndarray:
      """Function to initialize the coefficients of the model.

      :param models: A list of GAMMs, - each based on one of the formulas provided to a model.
      :type models: [mssm.models.GAMM]
      :return: A numpy array of shape (-1,1), holding initial values for all model coefficients.
      :rtype: np.ndarray
      """

      # Just set to very small positive values
      coef = np.array([1e-4 for _ in range(models[0].formulas[0].n_coef)]).reshape(-1,1)
      return coef