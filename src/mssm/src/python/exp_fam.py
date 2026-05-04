import numpy as np
import scipy as scp
import math
import warnings
import copy
from itertools import repeat
from collections.abc import Callable
from .custom_types import DerivOrder
from abc import ABC, abstractmethod

HAS_MP = True
try:
    import multiprocess as mp
except ImportError:
    HAS_MP = False


class Link(ABC):
    """
    Link function base class. To be implemented by any link functiion used for GAMMs and GAMMLSS
    models. Only links used by ``GAMLSS`` models require implementing the dy2 function. Note, that
    care must be taken that every method returns only valid values. Specifically, no returned
    element may be ``numpy.nan`` or ``numpy.inf``.
    """

    @abstractmethod
    def f(self, mu: np.ndarray) -> np.ndarray:
        """
        Link function :math:`f()` mapping mean :math:`\\boldsymbol{\\mu}` of an exponential family
        to the model prediction :math:`\\boldsymbol{\\eta}`, so that
        :math:`f(\\boldsymbol{\\mu}) = \\boldsymbol{\\eta}`. See Wood (2017, 3.1.2) and
        Faraway (2016).

        References:

         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        """
        pass

    @abstractmethod
    def fi(self, eta: np.ndarray) -> np.ndarray:
        """
        Inverse of the link function mapping :math:`\\boldsymbol{\\eta} = f(\\boldsymbol{\\mu})` to
        the mean :math:`fi(\\boldsymbol{\\eta}) = fi(f(\\boldsymbol{\\mu})) = \\boldsymbol{\\mu}`.
        see Faraway (2016) and the ``Link.f`` function.

        References:

         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param eta: A numpy array containing the model prediction corresponding to each observation.
        :type eta: np.ndarray
        """
        pass

    @abstractmethod
    def dy1(self, mu: np.ndarray) -> np.ndarray:
        """
        First derivative of :math:`f(\\boldsymbol{\\mu})` with respect to
        :math:`\\boldsymbol{\\mu}` Needed for Fisher scoring/PIRLS (Wood, 2017).

        References:

         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        """
        pass

    @abstractmethod
    def dy2(self, mu: np.ndarray) -> np.ndarray:
        """
        Second derivative of :math:`f(\\boldsymbol{\\mu})` with respect to
        :math:`\\boldsymbol{\\mu}`. Needed for GAMMLSS models (Wood, 2017).

        References:

         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        """
        pass


class Logit(Link):
    """
    Logit Link function, which is canonical for the binomial model.
    :math:`\\boldsymbol{\\eta}` = log-odds of success.
    """

    def f(self, mu: np.ndarray) -> np.ndarray:
        """
        Canonical link for binomial distribution with :math:`\\boldsymbol{\\mu}` holding the
        probabilities of success, so that the model prediction :math:`\\boldsymbol{\\eta}`
        is equal to the log-odds.

        References:

         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        """
        with warnings.catch_warnings():  # Divide by 0
            warnings.simplefilter("ignore")
            eta = np.log(mu / (1 - mu))

        return eta

    def fi(self, eta: np.ndarray) -> np.ndarray:
        """
        For the logit link and the binomial model, :math:`\\boldsymbol{\\eta}` = log-odds, so the
        inverse to go from :math:`\\boldsymbol{\\eta}` to :math:`\\boldsymbol{\\mu}` is
        :math:`\\boldsymbol{\\mu} = exp(\\boldsymbol{\\eta}) / (1 + exp(\\boldsymbol{\\eta}))`.
        see Faraway (2016)

        References:

         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param eta: A numpy array containing the model prediction corresponding to each observation.
        :type eta: np.ndarray
        """
        with warnings.catch_warnings():  # Overflow
            warnings.simplefilter("ignore")
            mu = np.exp(eta) / (1 + np.exp(eta))

        return mu

    def dy1(self, mu: np.ndarray) -> np.ndarray:
        """
        First derivative of :math:`f(\\boldsymbol{\\mu})` with respect to
        :math:`\\boldsymbol{\\mu}`. Needed for Fisher scoring/PIRLS (Wood, 2017):

        .. math::


           f(\\mu) = log(\\mu / (1 - \\mu))

           f(\\mu) = log(\\mu) - log(1 - \\mu)

           \\partial f(\\mu)/ \\partial \\mu = 1/\\mu - 1/(1 - \\mu)

        Faraway (2016) simplifies this to:
        :math:`\\partial f(\\mu)/ \\partial \\mu = 1 / (\\mu - \\mu^2) = 1/ ((1-\\mu)\\mu)`

        References:

         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        """
        with warnings.catch_warnings():  # Divide by 0
            warnings.simplefilter("ignore")
            d = 1 / ((1 - mu) * mu)

        return d

    def dy2(self, mu: np.ndarray) -> np.ndarray:
        """
        Second derivative of :math:`f(\\boldsymbol{\\mu})` with respect to
        :math:`\\boldsymbol{\\mu}`. Needed for GAMMLSS models (Wood, 2017).

        References:

         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, \
            Second Edition (2nd ed.).

        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        """
        with warnings.catch_warnings():  # Divide by 0
            warnings.simplefilter("ignore")
            d2 = (2 * mu - 1) / (np.power(mu, 2) * np.power(1 - mu, 2))

        return d2


class Identity(Link):
    """
    Identity Link function. :math:`\\boldsymbol{\\mu}=\\boldsymbol{\\eta}` and so this link is
    trivial.
    """

    def f(self, mu: np.ndarray) -> np.ndarray:
        """
        Canonical link for normal distribution with
        :math:`\\boldsymbol{\\eta} = \\boldsymbol{\\mu}`.

        References:

         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array containing the predicted mean for the response distribution \
            corresponding to each observation.
        :type mu: np.ndarray
        """
        return mu

    def fi(self, eta: np.ndarray) -> np.ndarray:
        """
        For the identity link, :math:`\\boldsymbol{\\eta} = \\boldsymbol{\\mu}`, so the inverse is
        also just the identity. see Faraway (2016)

        References:

         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param eta: A numpy array containing the model prediction corresponding to each observation.
        :type eta: np.ndarray
        """
        return eta

    def dy1(self, mu: np.ndarray) -> np.ndarray:
        """
        First derivative of :math:`f(\\boldsymbol{\\mu})` with respect to
        :math:`\\boldsymbol{\\mu}`. Needed for Fisher scoring/PIRLS (Wood, 2017).

        References:

         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        """
        return np.ones_like(mu)

    def dy2(self, mu: np.ndarray) -> np.ndarray:
        """
        Second derivative of :math:`f(\\boldsymbol{\\mu})` with respect to
        :math:`\\boldsymbol{\\mu}`. Needed for GAMMLSS models (Wood, 2017).

        References:

         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        """
        return np.zeros_like(mu)


class LOG(Link):
    """
    Log Link function. :math:`log(\\boldsymbol{\\mu}) = \\boldsymbol{\\eta}`.
    """

    def f(self, mu: np.ndarray) -> np.ndarray:
        """
        Non-canonical link for Gamma distribution with
        :math:`log(\\boldsymbol{\\mu}) = \\boldsymbol{\\eta}`.

        References:

         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        """
        with warnings.catch_warnings():  # log of < 0
            warnings.simplefilter("ignore")
            eta = np.log(mu)

        return eta

    def fi(self, eta: np.ndarray) -> np.ndarray:
        """
        For the log link, :math:`\\boldsymbol{\\eta} = log(\\boldsymbol{\\mu})`, so
        :math:`exp(\\boldsymbol{\\eta})=\\boldsymbol{\\mu}`. see Faraway (2016)

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param eta: A numpy array containing the model prediction corresponding to each observation.
        :type eta: np.ndarray
        """
        with warnings.catch_warnings():  # Overflow
            warnings.simplefilter("ignore")
            mu = np.exp(eta)

        return mu

    def dy1(self, mu: np.ndarray) -> np.ndarray:
        """
        First derivative of :math:`f(\\boldsymbol{\\mu})` with respect to
        :math:`\\boldsymbol{\\mu}`. Needed for Fisher scoring/PIRLS (Wood, 2017).

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        """
        with warnings.catch_warnings():  # Divide by 0
            warnings.simplefilter("ignore")
            d = 1 / mu

        return d

    def dy2(self, mu: np.ndarray) -> np.ndarray:
        """
        Second derivative of :math:`f(\\boldsymbol{\\mu})` with respect to
        :math:`\\boldsymbol{\\mu}`. Needed for GAMMLSS models (Wood, 2017).

        References:

         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        """
        with warnings.catch_warnings():  # Divide by 0
            warnings.simplefilter("ignore")
            d2 = -1 * (1 / np.power(mu, 2))

        return d2


class LOGb(Link):
    """
    Log + b Link function. :math:`log(\\boldsymbol{\\mu} + b) = \\boldsymbol{\\eta}`.

    :param b: The constant to add to :math:`\\mu` before taking the log.
    :type b: float
    """

    def __init__(self, b: float):
        super().__init__()
        self.b = b

    def f(self, mu: np.ndarray) -> np.ndarray:
        """
        :math:`log(\\boldsymbol{\\mu} + b) = \\boldsymbol{\\eta}`.

        References:

         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        """
        with warnings.catch_warnings():  # Log of < 0
            warnings.simplefilter("ignore")
            eta = np.log(mu + self.b)

        return eta

    def fi(self, eta: np.ndarray) -> np.ndarray:
        """
        For the logb link, :math:`\\boldsymbol{\\eta} = log(\\boldsymbol{\\mu} + b)`, so
        :math:`exp(\\boldsymbol{\\eta})-b =\\boldsymbol{\\mu}`

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param eta: A numpy array containing the model prediction corresponding to each observation.
        :type eta: np.ndarray
        """
        with warnings.catch_warnings():  # Overflow
            warnings.simplefilter("ignore")
            mu = np.exp(eta) - self.b

        return mu

    def dy1(self, mu: np.ndarray) -> np.ndarray:
        """
        First derivative of :math:`f(\\boldsymbol{\\mu})` with respect to
        :math:`\\boldsymbol{\\mu}`. Needed for Fisher scoring/PIRLS (Wood, 2017).

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        """
        with warnings.catch_warnings():  # Divide by 0
            warnings.simplefilter("ignore")
            d = 1 / (self.b + mu)

        return d

    def dy2(self, mu: np.ndarray) -> np.ndarray:
        """
        Second derivative of :math:`f(\\boldsymbol{\\mu})` with respect to
        :math:`\\boldsymbol{\\mu}`. Needed for GAMMLSS models (Wood, 2017).

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        """
        with warnings.catch_warnings():  # Divide by 0
            warnings.simplefilter("ignore")
            d2 = -1 * (1 / np.power(mu + self.b, 2))

        return d2


def est_scale(res: np.ndarray, rows_X: int, total_edf: float) -> float:
    """
    Scale estimate from Wood & Fasiolo (2017).

    Refereces:
     - Wood, S. N., & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing \
        parameter optimization with application to Tweedie location, scale and shape models.

    :param res: A numpy array containing the difference between the model prediction and the
        (pseudo) data.
    :type res: np.ndarray
    :param rows_X: The number of observations collected.
    :type rows_X: int
    :param total_edf: The expected degrees of freedom for the model.
    :type total_edf: float
    """
    resDot = res.T.dot(res)[0, 0]

    sigma = resDot / (rows_X - total_edf)

    return sigma


class Family(ABC):
    """
    Base class to be implemented by Exp. family member with (optional) scale parameter
    :math:`\\phi`.

    :param link: The link function to be used by the model of the mean of this family.
    :type link: Link
    :param twopar: Whether the family has two parameters (mean,scale) to be estimated (i.e.,
        whether the likelihood is a function of two parameters), or only a single
        one (usually the mean).
    :type twopar: bool
    :param scale: Known/fixed scale parameter for this family. Setting this to None means
        the parameter has to be estimated. **Must be set to 1 if the family has no scale
        parameter** (i.e., when ``twopar = False``)
    :type scale: float | None, optional
    """

    def __init__(self, link: Link, twopar: bool, scale: float | None = None) -> None:
        self.link = link
        self.twopar = twopar
        self.scale = scale  # Known scale parameter!
        self.is_canonical = False  # Canonical link for generalized model?

    def init_mu(self, y: np.ndarray) -> np.ndarray:
        """
        Convenience function to compute an initial :math:`\\boldsymbol{\\mu}` estimate
        passed to the GAMM/PIRLS estimation routine.

        Returns ``y`` by default.

        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :return: a N-dimensional vector of shape (-1,1) containing an initial estimate of the mean
        :rtype: np.ndarray
        """
        return y

    @abstractmethod
    def V(self, mu: np.ndarray) -> np.ndarray:
        """
        The variance function (of the mean; see Wood, 2017, 3.1.2). Different exponential
        families allow for different relationships between the variance in our random response
        variable and the mean of it. For the normal model this is assumed to be constant.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response
            distribution corresponding to each observation.
        :type mu: np.ndarray
        :return: a N-dimensional vector of shape (-1,1) containing the variance function
            evaluated for each mean
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def dVy1(self, mu: np.ndarray) -> np.ndarray:
        """
        The first derivative of the variance function (of the mean; see Wood, 2017, 3.1.2) with
        respect ot the mean.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the
            response distribution corresponding to each observation.
        :type mu: np.ndarray
        :return: a N-dimensional vector of shape (-1,1) containing the first derivative of
            the variance function with respect to each mean
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def llk(self, y: np.ndarray, mu: np.ndarray) -> float:
        """
        log-probability of :math:`\\mathbf{y}` under this family with
        mean = :math:`\\boldsymbol{\\mu}`. Essentially sum over all elements in the vector returned
        by the :func:`lp` method.

        Families with more than one parameter that needs to be estimated in order to evaluate the
        model's log-likelihood (i.e., ``two_par=True``) must add as key-word argument a ``scale``
        parameter with a default value, e.g.,::

           def llk(self, y, mu, scale=1):
              ...

        You can check the implementation of the :class:`Gaussian` Family for an example.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response
            distribution corresponding to each observation.
        :type mu: np.ndarray
        :return: log-likelihood of the model under this family
        :rtype: float
        """
        pass

    @abstractmethod
    def lp(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """
        Log-probability of observing every value in :math:`\\mathbf{y}` under this family with
        mean = :math:`\\boldsymbol{\\mu}`.

        Families with more than one parameter that needs to be estimated in order to evaluate the
        model's log-likelihood (i.e., ``two_par=True``) must add as key-word argument a ``scale``
        parameter with a default value, e.g.,::

           def lp(self, y, mu, scale=1):
              ...

        You can check the implementation of the :class:`Gaussian` Family for an example.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response
            distribution corresponding to each observation.
        :type mu: np.ndarray
        :return: a N-dimensional vector of shape (-1,1) containing the log-probability of observing
            each data-point under the current model.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def deviance(self, y: np.ndarray, mu: np.ndarray) -> float:
        """
        Deviance of the model under this family: 2 * (llk_max - llk_c) * scale
        (Wood, 2017; Faraway, 2016).

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response
            distribution corresponding to each observation.
        :type mu: np.ndarray
        :return: Deviance of the model under this family
        :rtype: float
        """
        pass

    @abstractmethod
    def D(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """
        Contribution of each observation to model Deviance (Wood, 2017; Faraway, 2016)

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response
            distribution corresponding to each observation.
        :type mu: np.ndarray
        :return: a N-dimensional vector of shape (-1,1) containing the contribution of each
            observation to the overall deviance.
        :rtype: np.ndarray
        """
        pass

    def dllkdcoef(
        self, coef: np.ndarray, y: np.ndarray, X: scp.sparse.csc_array, scale: float = 1
    ) -> np.ndarray:
        """Returns vector of partial derivatives of the log-likelihood with respect to ``coef``
        evaluated at the (optional) ``scale`` parameter.

        Applies to all families, so does not have to be re-implemented by families implementing
        the base class. Derivation follows from IRLS routine and is given for example in Wood
        (2017).

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param X: Model matrix used by a model
        :type X: scp.sparse.csc_array
        :param scale: Optional scale parameter if ``self.twopar is True``, defaults to 1
        :type scale: float, optional
        :return: The Gradient of the log-likelihood evaluated at ``coef`` as numpy array of
            shape (-1,1).
        :rtype: np.ndarray
        """

        # Compute mean
        mu = self.link.fi(X @ coef)

        # Get derivative of link and variance function at ``mu``
        dy1 = self.link.dy1(mu)
        V = self.V(mu)

        # Compute gradient
        with warnings.catch_warnings():  # Divide by zero or invalid value in multiply
            warnings.simplefilter("ignore")
            G = (y - mu) / (dy1 * V)
        G[np.isnan(G) | np.isinf(G)] = 0
        grad = np.sum(G * X, axis=0).reshape(-1, 1) / scale

        return grad

    def dllkdlscale(
        self, coef: np.ndarray, y: np.ndarray, X: scp.sparse.csc_array, scale: float = 1
    ) -> float:
        """Returns partial derivative of the log-likelihood with respect to ``log(scale)`` if this
        family has a scale parameter (i.e., if ``self.twopar is True``). Otherwise this function
        returns zero.

        Base-class implementation relies on finite differencing to compute the required partial
        derivative. For efficiency reasons, families inheriting from the base class should
        implement an analytic solution. Note however, that this function is only used by the
        mcmc samplers and thus this is only necessary if you want to perform fully Bayesian
        inference.

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param X: Model matrix used by a model
        :type X: scp.sparse.csc_array
        :param scale: Optional scale parameter if ``self.twopar is True``, defaults to 1
        :type scale: float, optional
        :return: The partial derivative of the log-likelihood with respect to ``log(scale)`` if
            this family has a scale parameter, otherwise zero.
        :rtype: None | float
        """

        if self.twopar is True:
            # Compute mean and log-scale
            mu = self.link.fi(X @ coef)
            lscale = np.log(scale)

            def llk_wrap(x: float) -> float:

                return self.llk(y, mu, scale=np.exp(x[0]))

            deriv = scp.optimize.approx_fprime(np.array([lscale]), llk_wrap)
            return deriv[0]

        return 0.0

    def d2llkd2lscale(
        self, coef: np.ndarray, y: np.ndarray, X: scp.sparse.csc_array, scale: float = 1
    ) -> float:
        """Returns second partial derivative of the log-likelihood with respect to ``log(scale)``
        if this family has a scale parameter (i.e., if ``self.twopar is True``). Otherwise this
        function returns zero.

        Base-class implementation relies on finite differencing to compute the required partial
        derivative. For efficiency reasons, families inheriting from the base class should
        implement an analytic solution. Note however, that this function is only used by the
        mcmc samplers (and only once) and thus this is only necessary if you want to perform fully
        Bayesian inference.

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param X: Model matrix used by a model
        :type X: scp.sparse.csc_array
        :param scale: Optional scale parameter if ``self.twopar is True``, defaults to 1
        :type scale: float, optional
        :return: The second partial derivative of the log-likelihood with respect to ``log(scale)``
            if this family has a scale parameter, otherwise zero.
        :rtype: None | float
        """

        if self.twopar is True:
            mu = self.link.fi(X @ coef)
            lscale = np.log(scale)

            def llk_wrap(x: float) -> float:

                return self.llk(y, mu, scale=np.exp(x[0]))

            H = scp.differentiate.hessian(
                lambda r: np.apply_along_axis(llk_wrap, axis=0, arr=r),
                np.array([lscale]),
            )

            return H.ddf[0, 0]

        return 0.0


class Binomial(Family):
    """
    Binomial family. For this implementation we assume that we have collected proportions of
    success, i.e., the dependent variables specified in the model `Formula` needs to hold observed
    proportions and not counts! If we assume that each observation :math:`y_i` reflects a single
    independent draw from a binomial, (with :math:`n=1`, and :math:`p_i` being the probability that
    the result is 1) then the dependent variable should either hold 1 or 0.
    If we have multiple independent draws from the binomial per observation (i.e., row in our
    data-frame), then :math:`n` will usually differ between observations/rows in our data-frame
    (i.e., we observe :math:`k_i` counts of success out of :math:`n_i` draws - so that
    :math:`y_i=k_i/n_i`). In that case, the `Binomial()` family accepts a vector for argument
    :math:`\\mathbf{n}` (which is simply set to 1 by default, assuming binary data), containing
    :math:`n_i` for every observation :math:`y_i`.

    In this implementation, the scale parameter is kept fixed/known at 1.

    References:
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
        Edition (2nd ed.).

    :param link: The link function to be used by the model of the mean of this family. By default
        set to the canonical logit link.
    :type link: Link
    :param n: Number of independent draws from a Binomial per observation/row of data-frame. For
        binary data this can simply be set to 1, which is the default.
    :type n: int or [int], optional
    """

    def __init__(self, link: Link = Logit(), n: int | list[int] = 1) -> None:
        super().__init__(link, False, 1)
        self.n: int | list[int] = n  # Number of independent samples from Binomial!
        self.__max_llk: float | None = None  # Needed for Deviance calculation.
        self.is_canonical: bool = isinstance(link, Logit)

    def init_mu(self, y: np.ndarray) -> np.ndarray:
        """
        Function providing initial :math:`\\boldsymbol{\\mu}` vector for GAMM.

        Estimation assumes proportions as dep. variable. According to:
        https://stackoverflow.com/questions/60526586/ the glm() function in R always initializes
        :math:`\\mu` = 0.75 for observed proportions (i.e., elements in :math:`\\mathbf{y}`) of 1
        and :math:`\\mu` = 0.25 for proportions of zero.
        This can be achieved by adding 0.5 to the observed proportion of success
        (and adding one observation).

        :param y: A numpy array containing each observation.
        :type y: np.ndarray
        :return: a N-dimensional vector of shape (-1,1) containing an initial estimate of the
            probability of success per observation
        :rtype: np.ndarray
        """
        prop = (y + 0.5) / (2)
        self.__max_llk = self.llk(y, y)
        return prop

    def V(self, mu: np.ndarray) -> np.ndarray:
        """
        The variance function (of the mean; see Wood, 2017, 3.1.2) for the Binomial model. Variance
        is minimal for :math:`\\mu=1` and :math:`\\mu=0`, maximal for :math:`\\mu=0.5`.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array containing the predicted probability for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :return: a N-dimensional vector of shape (-1,1) containing the variance function evaluated
            for each mean
        :rtype: np.ndarray
        """
        # Faraway (2016):
        return mu * (1 - mu) / self.n

    def dVy1(self, mu: np.ndarray) -> np.ndarray:
        """
        The first derivative of the variance function (of the mean; see Wood, 2017, 3.1.2) with
        respect ot the mean.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :return: a N-dimensional vector of shape (-1,1) containing the first derivative of the
            variance function with respect to each mean
        :rtype: np.ndarray
        """
        return (1 - 2 * mu) / self.n

    def lp(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """
        Log-probability of observing every proportion in :math:`\\mathbf{y}` under their respective
        binomial with mean = :math:`\\boldsymbol{\\mu}`.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array containing each observed proportion.
        :type y: np.ndarray
        :param mu: A numpy array containing the predicted probability for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :return: a N-dimensional vector containing the log-probability of observing each data-point
            under the current model.
        :rtype: np.ndarray
        """
        # y is observed proportion of success
        return scp.stats.binom.logpmf(k=y * self.n, p=mu, n=self.n)

    def llk(self, y: np.ndarray, mu: np.ndarray) -> float:
        """
        log-probability of data under given model. Essentially sum over all elements in the vector
        returned by the :func:`lp` method.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :return: log-likelihood of the model
        :rtype: float
        """
        # y is observed proportion of success
        return np.sum(self.lp(y, mu))

    def deviance(self, y: np.ndarray, mu: np.ndarray) -> float:
        """
        Deviance of the model under this family: 2 * (llk_max - llk_c) * scale
        (Wood, 2017; Faraway, 2016).

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, \
            Second Edition (2nd ed.).

        :param y: A numpy array containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :return: Deviance of the model
        :rtype: float
        """
        dev = np.sum(self.D(y, mu))
        return dev

    def D(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """
        Contribution of each observation to model Deviance (Wood, 2017; Faraway, 2016)

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :return: a N-dimensional vector of shape (-1,1) containing the contribution of each
            observation to the model deviance
        :rtype: np.ndarray
        """
        # Based on Table 3.1 in Wood (2017)
        k = y * self.n
        kmu = mu * self.n

        with warnings.catch_warnings():  # Divide by zero
            warnings.simplefilter("ignore")
            ratio1 = np.log(k) - np.log(kmu)
            ratio2 = np.log(self.n - k) - np.log(self.n - kmu)

        # Limiting behavior of y.. (see Wood, 2017)
        ratio1[np.isinf(ratio1) | np.isnan(ratio1)] = 0
        ratio2[np.isinf(ratio2) | np.isnan(ratio2)] = 0

        return 2 * (k * (ratio1) + ((self.n - k) * ratio2))


class Gaussian(Family):
    """Normal/Gaussian Family.

    We assume: :math:`Y_i \\sim N(\\mu_i,\\sigma)` - i.e., each of the :math:`N` observations is
    generated from a normally distributed RV with observation-specific mean and shared scale
    parameter :math:`\\sigma`. Equivalent to the assumption that the observed residual vector -
    the difference between the model prediction and the observed data - should look like what could
    be expected from drawing :math:`N` independent samples from a Normal with mean zero and
    standard deviation equal to :math:`\\sigma`.

    References:
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
        Edition (2nd ed.).

    :param link: The link function to be used by the model of the mean of this family. By default
        set to the canonical identity link.
    :type link: Link
    :param scale: Known scale parameter for this family - by default set to None so that the scale
        parameter is estimated.
    :type scale: float or None, optional
    """

    def __init__(self, link: Link = Identity(), scale: float | None = None) -> None:
        super().__init__(link, True, scale)
        self.is_canonical: bool = isinstance(link, Identity)

    def V(self, mu: np.ndarray) -> np.ndarray:
        """Variance function for the Normal family.

        Not really a function since the link between variance and mean of the RVs is assumed
        constant for this model.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: a N-dimensional vector of the model prediction/the predicted mean
        :type mu: np.ndarray
        :return: a N-dimensional vector of 1s
        :rtype: np.ndarray
        :return: a N-dimensional vector of shape (-1,1) containing the variance function evaluated
            for each mean
        :rtype: np.ndarray
        """
        # Faraway (2016)
        return np.ones_like(mu)

    def dVy1(self, mu: np.ndarray) -> np.ndarray:
        """
        The first derivative of the variance function (of the mean; see Wood, 2017, 3.1.2) with
        respect ot the mean.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :return: a N-dimensional vector of shape (-1,1) containing the first derivative of the
            variance function with respect to each mean
        :rtype: np.ndarray
        """
        return np.zeros_like(mu)

    def lp(self, y: np.ndarray, mu: np.ndarray, scale: float = 1) -> np.ndarray:
        """Log-probability of observing every proportion in :math:`\\mathbf{y}` under their
        respective Normal with mean = :math:`\\boldsymbol{\\mu}`.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :param scale: The (estimated) sigma (variance) parameter, defaults to 1
        :type scale: float, optional
        :return: a N-dimensional vector containing the log-probability of observing each data-point
            under the current model.
        :rtype: np.ndarray
        """
        return scp.stats.norm.logpdf(y, loc=mu, scale=math.sqrt(scale))

    def llk(self, y: np.ndarray, mu: np.ndarray, scale: float = 1) -> float:
        """log-probability of data under given model. Essentially sum over all elements in the
        vector returned by the :func:`lp` method.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :param scale: The (estimated) sigma (variance) parameter, defaults to 1
        :type scale: float, optional
        :return: The log-probability of observing all data under the current model.
        :rtype: float
        """
        return np.sum(self.lp(y, mu, scale))

    def deviance(self, y: np.ndarray, mu: np.ndarray) -> float:
        """Deviance of the model under this family: 2 * (llk_max - llk_c) * scale
        (Wood, 2017; Faraway, 2016).

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :return: The model deviance.
        :rtype: float
        """
        # Based on Faraway (2016)
        res = y - mu
        rss = res.T @ res
        return rss[0, 0]

    def D(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Contribution of each observation to model Deviance (Wood, 2017; Faraway, 2016)

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :return: A N-dimensional vector containing the contribution of each data-point to the
            overall model deviance.
        :rtype: np.ndarray
        """
        res = y - mu
        return np.power(res, 2)


class Gamma(Family):
    """Gamma Family.

    We assume: :math:`Y_i \\sim \\Gamma(\\mu_i,\\phi)`. The Gamma distribution is usually not
    expressed in terms of the mean and scale (:math:`\\phi`) parameter
    but rather in terms of a shape and rate parameter - called :math:`\\alpha` and :math:`\\beta`
    respectively. Wood (2017) provides :math:`\\alpha = 1/\\phi`.
    With this we can obtain :math:`\\beta = 1/\\phi/\\mu` (see the source-code for :func:`lp`
    method for details).

    References:
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
        Edition (2nd ed.).

    :param link: The link function to be used by the model of the mean of this family. By default
        set to the log link.
    :type link: Link
    :param scale: Known scale parameter for this family - by default set to None so that the
        scale parameter is estimated.
    :type scale: float or None, optional
    """

    def __init__(self, link: Link = LOG(), scale: float | None = None) -> None:
        super().__init__(link, True, scale)
        self.is_canonical: bool = False  # Inverse link not implemented..

    def V(self, mu: np.ndarray) -> np.ndarray:
        """Variance function for the Gamma family.

        The variance of random variable :math:`Y` is proportional to it's mean raised to the
        second power.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: a N-dimensional vector of the model prediction/the predicted mean
        :type mu: np.ndarray
        :return: mu raised to the power of 2
        :rtype: np.ndarray
        """
        # Faraway (2016)
        return np.power(mu, 2)

    def dVy1(self, mu: np.ndarray) -> np.ndarray:
        """
        The first derivative of the variance function (of the mean; see Wood, 2017, 3.1.2) with
        respect ot the mean.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :return: a N-dimensional vector of shape (-1,1) containing the first derivative of the
            variance function with respect to each mean
        :rtype: np.ndarray
        """
        return 2 * mu

    def lp(self, y: np.ndarray, mu: np.ndarray, scale: float = 1) -> np.ndarray:
        """Log-probability of observing every proportion in :math:`\\mathbf{y}` under their
        respective Gamma with mean = :math:`\\boldsymbol{\\mu}`.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array containing each observed value.
        :type y: np.ndarray
        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :param scale: The (estimated) scale parameter, defaults to 1
        :type scale: float, optional
        :return: a N-dimensional vector containing the log-probability of observing each data-point
            under the current model.
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
        alpha = 1 / scale
        beta = alpha / mu
        return scp.stats.gamma.logpdf(y, a=alpha, scale=(1 / beta))

    def llk(self, y: np.ndarray, mu: np.ndarray, scale: float = 1) -> float:
        """log-probability of data under given model. Essentially sum over all elements in the
        vector returned by the :func:`lp` method.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :param scale: The (estimated) scale parameter, defaults to 1
        :type scale: float, optional
        :return: The log-probability of observing all data under the current model.
        :rtype: float
        """
        return np.sum(self.lp(y, mu, scale))

    def D(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Contribution of each observation to model Deviance (Wood, 2017; Faraway, 2016)

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :return: A N-dimensional vector containing the contribution of each data-point to the
            overall model deviance.
        :rtype: np.ndarray
        """
        # Based on Table 3.1 in Wood (2017)
        diff = (y - mu) / mu
        ratio = -(np.log(y) - np.log(mu))
        return 2 * (diff + ratio)

    def deviance(self, y: np.ndarray, mu: np.ndarray) -> float:
        """Deviance of the model under this family: 2 * (llk_max - llk_c) * scale
        (Wood, 2017; Faraway, 2016).

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :return: The model deviance.
        :rtype: float
        """
        # Based on Table 3.1 in Wood (2017)
        dev = np.sum(self.D(y, mu))
        return dev


class InvGauss(Family):
    """Inverse Gaussian Family.

    We assume: :math:`Y_i \\sim IG(\\mu_i,\\phi)`. The Inverse Gaussian distribution is usually not
    expressed in terms of the mean and scale (:math:`\\phi`) parameter but rather in terms of a
    shape and scale parameter - called :math:`\\nu` and :math:`\\lambda` respectively
    (see the scipy implementation).
    We can simply set :math:`\\nu=\\mu` (compare scipy density to the one in table 3.1 of
    Wood, 2017). Wood (2017) shows that :math:`\\phi=1/\\lambda`, so this provides
    :math:`\\lambda=1/\\phi`

    References:
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
        Edition (2nd ed.).
     - scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgauss.html

    :param link: The link function to be used by the model of the mean of this family. By default
        set to the log link.
    :type link: Link
    :param scale: Known scale parameter for this family - by default set to None so that the scale
        parameter is estimated.
    :type scale: float or None, optional
    """

    def __init__(self, link: Link = LOG(), scale: float | None = None) -> None:
        super().__init__(link, True, scale)
        self.is_canonical: bool = False  # Modified inverse link not implemented..

    def V(self, mu: np.ndarray) -> np.ndarray:
        """Variance function for the Inverse Gaussian family.

        The variance of random variable :math:`Y` is proportional to it's mean raised to the
        third power.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: a N-dimensional vector of the model prediction/the predicted mean
        :type mu: np.ndarray
        :return: mu raised to the power of 3
        :rtype: np.ndarray
        """
        # Faraway (2016)
        return np.power(mu, 3)

    def dVy1(self, mu: np.ndarray) -> np.ndarray:
        """
        The first derivative of the variance function (of the mean; see Wood, 2017, 3.1.2) with
        respect ot the mean.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :return: a N-dimensional vector of shape (-1,1) containing the first derivative of the
            variance function with respect to each mean
        :rtype: np.ndarray
        """
        return 3 * np.power(mu, 2)

    def lp(self, y: np.ndarray, mu: np.ndarray, scale: float = 1) -> np.ndarray:
        """Log-probability of observing every value in :math:`\\mathbf{y}` under their respective
        inverse Gaussian with mean = :math:`\\boldsymbol{\\mu}`.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array containing each observed value.
        :type y: np.ndarray
        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :param scale: The (estimated) scale parameter, defaults to 1
        :type scale: float, optional
        :return: a N-dimensional vector containing the log-probability of observing each
            data-point under the current model.
        :rtype: np.ndarray
        """
        # Need to transform from mean and scale to \nu & \\lambda
        # From Wood (2017), we have that
        # \\phi = 1/\\lambda
        # so \\lambda = 1/\\phi
        # From the density in
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgauss.html,
        # we have that \nu=\\mu
        lam = 1 / scale
        nu = mu
        return scp.stats.invgauss.logpdf(y, mu=nu / lam, scale=lam)

    def llk(self, y: np.ndarray, mu: np.ndarray, scale: float = 1) -> float:
        """log-probability of data under given model. Essentially sum over all elements in the
        vector returned by the :func:`lp` method.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :param scale: The (estimated) scale parameter, defaults to 1
        :type scale: float, optional
        :return: The log-probability of observing all data under the current model.
        :rtype: float
        """
        return np.sum(self.lp(y, mu, scale))

    def D(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Contribution of each observation to model Deviance (Wood, 2017; Faraway, 2016)

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :return: A N-dimensional vector containing the contribution of each data-point to the
            overall model deviance.
        :rtype: np.ndarray
        """
        # Based on Table 3.1 in Wood (2017)
        diff = np.power(y - mu, 2)
        prod = np.power(mu, 2) * y
        return diff / prod

    def deviance(self, y: np.ndarray, mu: np.ndarray) -> float:
        """Deviance of the model under this family: 2 * (llk_max - llk_c) * scale
        (Wood, 2017; Faraway, 2016).

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :return: The model deviance.
        :rtype: float
        """
        # Based on Table 3.1 in Wood (2017)
        dev = np.sum(self.D(y, mu))
        return dev


class Poisson(Family):
    """Poisson Family.

    We assume: :math:`Y_i \\sim P(\\lambda)`. We can simply set :math:`\\lambda=\\mu`
    (compare scipy density to the one in table 3.1 of Wood, 2017) and treat the scale parameter of
    a GAMM (:math:`\\phi`) as fixed/known at 1.

    References:
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
        Edition (2nd ed.).
     - scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html

    :param link: The link function to be used by the model of the mean of this family. By
        default set to the log link.
    :type link: Link
    """

    def __init__(self, link: Link = LOG()) -> None:
        super().__init__(link, False, 1)
        self.is_canonical: bool = isinstance(link, LOG)

    def V(self, mu: np.ndarray) -> np.ndarray:
        """Variance function for the Poisson family.

        The variance of random variable :math:`Y` is proportional to it's mean.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: a N-dimensional vector of the model prediction/the predicted mean
        :type mu: np.ndarray
        :return: mu
        :rtype: np.ndarray
        """
        # Wood (2017)
        return mu

    def dVy1(self, mu: np.ndarray) -> np.ndarray:
        """
        The first derivative of the variance function (of the mean; see Wood, 2017, 3.1.2) with
        respect ot the mean.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :return: a N-dimensional vector of shape (-1,1) containing the first derivative of the
            variance function with respect to each mean
        :rtype: np.ndarray
        """
        return np.ones_like(mu)

    def lp(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Log-probability of observing every value in :math:`\\mathbf{y}` under their respective
        Poisson with mean = :math:`\\boldsymbol{\\mu}`.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array containing each observed value.
        :type y: np.ndarray
        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :return: a N-dimensional vector containing the log-probability of observing each data-point
            under the current model.
        :rtype: np.ndarray
        """
        # Need to transform from mean to \\lambda
        # From Wood (2017) and the density in
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html,
        # we have that \\lam=\\mu
        lam = mu
        return scp.stats.poisson.logpmf(y, mu=lam)

    def llk(self, y: np.ndarray, mu: np.ndarray) -> float:
        """log-probability of data under given model. Essentially sum over all elements in the
        vector returned by the :func:`lp` method.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :param scale: The (estimated) scale parameter, defaults to 1
        :type scale: float, optional
        :return: The log-probability of observing all data under the current model.
        :rtype: float
        """
        return np.sum(self.lp(y, mu))

    def D(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Contribution of each observation to model Deviance (Wood, 2017; Faraway, 2016)

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :return: A N-dimensional vector containing the contribution of each data-point to the
            overall model deviance.
        :rtype: np.ndarray
        """
        # Based on Table 3.1 in Wood (2017)
        diff = y - mu

        with warnings.catch_warnings():  # Divide by zero
            warnings.simplefilter("ignore")
            ratio = y * (np.log(y) - np.log(mu))

        ratio[np.isinf(ratio) | np.isnan(ratio)] = 0

        return 2 * ratio - 2 * diff

    def deviance(self, y: np.ndarray, mu: np.ndarray) -> float:
        """Deviance of the model under this family: 2 * (llk_max - llk_c) * scale
        (Wood, 2017; Faraway, 2016).

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array containing the predicted mean for the response distribution
            corresponding to each observation.
        :type mu: np.ndarray
        :return: The model deviance.
        :rtype: float
        """
        # Based on Table 3.1 in Wood (2017)
        dev = np.sum(self.D(y, mu))
        return dev

    def init_mu(self, y: np.ndarray) -> np.ndarray:
        """
        Function providing initial :math:`\\boldsymbol{\\mu}` vector for Poisson GAMM.

        Matches initialization of ``poisson()`` family in R.

        :param y: A numpy array containing each observation.
        :type y: np.ndarray
        :return: a N-dimensional vector of shape (-1,1) containing an intial estimate of the mean
            of the response variables
        :rtype: np.ndarray
        """

        mu = np.zeros_like(y, dtype=np.float64)
        mu[y > 0] = y[y > 0]
        mu += 0.1

        return mu


class ExtendedFamily(Family):
    """
    Base class to be implemented by any "extended family" member. This family, defined by
    Wood et al. (2016) essentially includes any model which we can estimate via iterative
    reweighted least-squares. Likelihood can have additional parameters beyond scale and mean
    which can be estimated along model coefficients (see ``theta`` parameter).

    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
        Smooth Models.
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
        Edition (2nd ed.).

    :param link: The link function to be used by the model of the mean of this family.
    :type link: Link
    :param theta: Any additional parameters of the likelihood (**inculding any required scale
        parameter**). Array needs to be of shape (-1,1). Setting this to None means the parameters
        have to be estimated.
    :type theta | None: np.ndarray, optional
    :ivar np.ndarray theta: The (estimated) extra parameters of the log-likelihood.
        Each implementation of this class must initalize these if not provided (i.e., by
        implementing the ``init_theta`` method) and calls to :func:`GAMM.fit` will overwrite this
        attribute if the initial value for ``theta`` passed to the constructor was None.
    """

    def __init__(
        self,
        link: Link,
        theta: None | np.ndarray = None,
    ):
        super().__init__(link, False, 1)
        self.est_theta = theta is None
        self.theta = theta

        if self.theta is None:
            self.theta = self.init_theta()

        if self.theta is None:
            raise ValueError(
                "self.theta must be initialized to a np.array of shape (-1,1)."
            )

    @abstractmethod
    def init_theta(self) -> np.ndarray:
        """Function to initialize ``theta``, the extra parameters of the log-likelihood, if no value
        (i.e., ``None``) was passed to the constructor.

        Take a look at the :class:`ScaledT` implementation as an example.

        :return: Any additional parameters of the likelihood (``theta``). Array needs to be of shape
            (-1,1).
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def V(self, mu: np.ndarray, theta: None | np.ndarray = None) -> np.ndarray:
        """
        The variance function (of the mean; see Wood, 2017, 3.1.2) for an extended family evaluated
        at ``theta``.

        Take a look at the :class:`ScaledT` implementation as an example.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response
            distribution corresponding to each observation.
        :type mu: np.ndarray
        :param theta: Any additional parameters of the likelihood. Array needs to be of shape
            (-1,1). When this is set to None, ``self.theta`` should be used.
        :type theta: None | np.ndarray, optional
        :return: a N-dimensional vector of shape (-1,1) containing the variance function
            evaluated for each mean
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def dVy1(
        self, mu: np.ndarray, theta: None | np.ndarray = None
    ) -> np.ndarray | None:
        """
        The first derivative of the variance function (of the mean; see Wood, 2017, 3.1.2) with
        respect ot the mean evaluated at ``theta``.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the
            response distribution corresponding to each observation.
        :type mu: np.ndarray
        :param theta: Any additional parameters of the likelihood. Array needs to be of shape
            (-1,1). When this is set to None, ``self.theta`` should be used.
        :type theta: None | np.ndarray, optional
        :return: a N-dimensional vector of shape (-1,1) containing the first derivative of
            the variance function with respect to each mean
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def llk(
        self, y: np.ndarray, mu: np.ndarray, theta: None | np.ndarray = None
    ) -> float:
        """
        log-probability of :math:`\\mathbf{y}` under this family with
        mean = :math:`\\boldsymbol{\\mu}`. Essentially sum over all elements in the vector returned
        by the :func:`lp` method.

        Take a look at the :class:`ScaledT` implementation as an example.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response
            distribution corresponding to each observation.
        :type mu: np.ndarray
        :param theta: Any additional parameters of the likelihood. Array needs to be of shape
            (-1,1). When this is set to None, ``self.theta`` should be used.
        :type theta: None | np.ndarray, optional
        :return: log-likelihood of the model under this family
        :rtype: float
        """
        pass

    @abstractmethod
    def lp(
        self, y: np.ndarray, mu: np.ndarray, theta: None | np.ndarray = None
    ) -> np.ndarray:
        """
        Log-probability of observing every value in :math:`\\mathbf{y}` under this family with
        mean = :math:`\\boldsymbol{\\mu}`.

        Take a look at the :class:`ScaledT` implementation as an example.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response
            distribution corresponding to each observation.
        :type mu: np.ndarray
        :param theta: Any additional parameters of the likelihood. Array needs to be of shape
            (-1,1). When this is set to None, ``self.theta`` should be used.
        :type theta: None | np.ndarray, optional
        :return: a N-dimensional vector of shape (-1,1) containing the log-probability of observing
            each data-point under the current model.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def deviance(
        self, y: np.ndarray, mu: np.ndarray, theta: None | np.ndarray = None
    ) -> float:
        """
        Deviance of the model under this family: 2 * (llk_max - llk_c)
        (Wood, 2017; Faraway, 2016).

        Take a look at the :class:`ScaledT` implementation as an example.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response
            distribution corresponding to each observation.
        :type mu: np.ndarray
        :param theta: Any additional parameters of the likelihood. Array needs to be of shape
            (-1,1). When this is set to None, ``self.theta`` should be used.
        :type theta: None | np.ndarray, optional
        :return: Deviance of the model under this family
        :rtype: float
        """
        pass

    @abstractmethod
    def D(
        self, y: np.ndarray, mu: np.ndarray, theta: None | np.ndarray = None
    ) -> np.ndarray:
        """
        Contribution of each observation to model Deviance (Wood, 2017; Faraway, 2016).

        Take a look at the :class:`ScaledT` implementation as an example.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response
            distribution corresponding to each observation.
        :type mu: np.ndarray
        :param theta: Any additional parameters of the likelihood. Array needs to be of shape
            (-1,1). When this is set to None, ``self.theta`` should be used.
        :type theta: None | np.ndarray, optional
        :return: a N-dimensional vector of shape (-1,1) containing the contribution of each
            observation to the overall deviance.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def dDdmu(
        self, y: np.ndarray, mu: np.ndarray, theta: None | np.ndarray = None
    ) -> np.ndarray:
        """
        Computes derivative of the deviance **or twice the negative log-likelihood** with respect to
        ``mu``.

        Take a look at the :class:`ScaledT` implementation as an example.

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response
            distribution corresponding to each observation.
        :type mu: np.ndarray
        :param theta: Any additional parameters of the likelihood. Array needs to be of shape
            (-1,1). When this is set to None, ``self.theta`` should be used.
        :type theta: None | np.ndarray, optional
        :return: a N-dimensional vector of shape ``(len(mu),1)`` containing the derivatives of
            the deviance with respect to ``mu`` for each observation.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def d2Ddmu(
        self, y: np.ndarray, mu: np.ndarray, theta: None | np.ndarray = None
    ) -> np.ndarray:
        """
        Computes second derivative of the deviance **or twice the negative log-likelihood** with
        respect to ``mu``. This function is by default used as fallback during fitting, but this can
        be overwritten by implementing the ``Ed2Ddmu`` method. In that case, this function is only
        called after estimation has been completed to get the observed hessian at the final
        coefficient estimate.

        Take a look at the :class:`ScaledT` implementation as an example.

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response
            distribution corresponding to each observation.
        :type mu: np.ndarray
        :param theta: Any additional parameters of the likelihood. Array needs to be of shape
            (-1,1). When this is set to None, ``self.theta`` should be used.
        :type theta: None | np.ndarray, optional
        :return: a N-dimensional vector of shape ``(len(mu),1)`` containing
            the second derivative of the deviance with respect to ``mu`` for each observation.
        :rtype: np.ndarray
        """
        pass

    def Ed2Ddmu(
        self, y: np.ndarray, mu: np.ndarray, theta: None | np.ndarray = None
    ) -> np.ndarray:
        """
        Computes expected second derivative of the deviance **or twice the negative log-likelihood**
        with respect to ``mu``. This function is used during fitting, but by default simply falls
        back to calling the ``d2Ddmu`` function to get the observed second derivatives.

        Take a look at the :class:`ScaledT` implementation as an example.

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response
            distribution corresponding to each observation.
        :type mu: np.ndarray
        :param theta: Any additional parameters of the likelihood. Array needs to be of shape
            (-1,1). When this is set to None, ``self.theta`` should be used.
        :type theta: None | np.ndarray, optional
        :return: a N-dimensional vector of shape ``(len(mu),1)`` containing
            the expected second derivatives of the deviance with respect to ``mu`` per observation.
        :rtype: np.ndarray
        """
        return self.d2Ddmu(y, mu, theta)

    @abstractmethod
    def gradientLTheta(
        self, y: np.ndarray, mu: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """
        Computes gradient of the log-likelihood with respect to ``theta``, given ``mu``.

        Take a look at the :class:`ScaledT` implementation as an example.

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response
            distribution corresponding to each observation.
        :type mu: np.ndarray
        :param theta: Any additional parameters of the likelihood. Array needs to be of shape
            (-1,1).
        :type theta: np.ndarray
        :return: a N-dimensional vector of shape ``(len(self.theta),1)`` containing the gradient of
            the log-likelihood with respect to theta.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def hessianLTheta(
        self, y: np.ndarray, mu: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """
        Computes (expected) hessian of the log-likelihood with respect to ``theta``, given
        ``mu``.

        Take a look at the :class:`ScaledT` implementation as an example.

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response
            distribution corresponding to each observation.
        :type mu: np.ndarray
        :param theta: Any additional parameters of the likelihood. Array needs to be of shape
            (-1,1).
        :type theta: np.ndarray
        :return: a N-dimensional vector of shape ``(len(self.theta),len(self.theta))`` containing
            the hessian of the log-likelihood with respect to theta.
        :rtype: np.ndarray
        """
        pass


class ScaledT(ExtendedFamily):
    """
    This class implements the scaled T family, based on the implementation in ``mgcv`` by
    Natalya Pya.

    Specifically, we assume that :math:`(y_i-\\mu_i)/\\phi) \\sim t_{\\nu}`, so that
    :math:`\\phi` takes on the role of the scale parameter and :math:`\\nu` are the degrees of
    freedom of the T-distribution. Note, that as :math:`\\nu \\to \\infty`, this family will behave
    like a Normal distribution with standard deviation :math:`\\phi`.

    Examples::

       from mssm.models import *
       from mssmViz.sim import *
       from mssmViz.plot import *

       # Simulate some data
       sim_fit_dat = sim3(n=500, scale=2, c=0.0, family=Gaussian(), seed=1)

       # Specify formula
       sim_fit_formula = Formula(
            lhs("y"),
            [i(), f(["x0"]), f(["x1"]), f(["x2"]), f(["x3"])],
            data=sim_fit_dat,
        )

        # Now fit a model assuming a scaled T family
        model = GAMM(sim_fit_formula, ScaledT(link=Identity()))
        model.fit()

    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
        Smooth Models.
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
        Edition (2nd ed.).
     - ``scat`` Family implemented in ``mgcv`` by Natalya Pya, see: \
        https://github.com/cran/mgcv/blob/master/R/efam.r#L2195

    :param link: The link function to be used by the model of the mean of this family. Defaults to
        class:`Identity`.
    :type link: Link, optional
    :param theta: An optional array containing an estimate of the log of the scale parameter and
        an estimate of the log of :math:`\\nu`. Setting this to None means both parameters
        have to be estimated.
    :type theta: None | np.ndarray, optional
    :ivar None | np.ndarray theta: The latest estimate of ``theta``. Calls to :func:`GAMM.fit` will
        overwrite this attribute if the initial value for ``theta`` passed to
        the constructor was None. Defaults to ``np.array([0.5, 10]).reshape(-1, 1)`` or whatever
        is passed to the constructor for ``theta``.
    """

    def __init__(self, link=Identity(), theta=None, min_df=3):
        super().__init__(link, theta)
        self.min_df = min_df

    def init_theta(self) -> np.ndarray:
        """Function that automatically initializes ``theta`` to the default.

        :return: Default value for theta: ``np.array([0.5, 10]).reshape(-1, 1)``
        :rtype: np.ndarray
        """
        return np.array([0.5, 10]).reshape(-1, 1)

    def V(self, mu: np.ndarray, theta: None | np.ndarray = None) -> np.ndarray:
        """
        The variance function (of the mean; see Wood, 2017, 3.1.2) for the scaled T family.

        The variance function is computed as in the ``scat`` implementation available in
        ``mgcv`` by Natalya Pya. Specifically, function returns :math:`\\phi^2 * \\nu / (\\nu - 2)`
        for each element in ``mu``.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response
            distribution corresponding to each observation.
        :type mu: np.ndarray
        :param theta: Optionally, the latest estimate of ``theta``, containing an estimate of
            the log of the scale parameter and the log of the degrees of freedom parameter.
            Array needs to be of shape (-1,1). When this is set to None, ``self.theta`` is used.
        :type theta: None | np.ndarray, optional
        :return: a N-dimensional vector of shape (-1,1) containing the variance function
            evaluated for each mean
        :rtype: np.ndarray
        """

        # Get theta
        if theta is None:
            theta = self.theta

        # Transform to DoF parameter nu and scale parameter phi
        phi = np.exp(theta[0, 0])
        nu = np.exp(theta[1, 0]) + self.min_df

        return np.ones_like(mu) * (np.power(phi, 2) * nu / (nu - 2))

    def dVy1(self, mu: np.ndarray, theta: None | np.ndarray = None) -> np.ndarray:
        """
        The first derivative of the variance function (of the mean; see Wood, 2017, 3.1.2) with
        respect ot the mean evaluated at ``theta``.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the
            response distribution corresponding to each observation.
        :type mu: np.ndarray
        :param theta: Optionally, the latest estimate of ``theta``, containing an estimate of
            the log of the scale parameter and the log of the degrees of freedom parameter.
            Array needs to be of shape (-1,1). When this is set to None, ``self.theta`` is used.
        :type theta: None | np.ndarray, optional
        :return: a N-dimensional vector of shape (-1,1) containing the first derivative of
            the variance function with respect to each mean
        :rtype: np.ndarray
        """
        return np.zeros_like(mu)

    def lp(
        self, y: np.ndarray, mu: np.ndarray, theta: None | np.ndarray = None
    ) -> np.ndarray:
        """
        Log-probability of observing every value in :math:`\\mathbf{y}` under this family with
        mean = :math:`\\boldsymbol{\\mu}`.

        Log-likelihood contributions are computed as in the ``scat`` implementation available in
        ``mgcv`` by Natalya Pya.

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).
         - ``scat`` Family implemented in ``mgcv`` by Natalya Pya, see: \
            https://github.com/cran/mgcv/blob/master/R/efam.r#L2195

        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response
            distribution corresponding to each observation.
        :type mu: np.ndarray
        :param theta: Optionally, the latest estimate of ``theta``, containing an estimate of
            the log of the scale parameter and the log of the degrees of freedom parameter.
            Array needs to be of shape (-1,1). When this is set to None, ``self.theta`` is used.
        :type theta: None | np.ndarray, optional
        :return: a N-dimensional vector of shape (-1,1) containing the log-probability of observing
            each data-point under the current model.
        :rtype: np.ndarray
        """

        # Get theta
        if theta is None:
            theta = self.theta

        # Transform to DoF parameter nu and scale parameter phi
        phi = np.exp(theta[0, 0])
        nu = np.exp(theta[1, 0]) + self.min_df

        # Now can compute llk for each obs
        lp = (
            -scp.special.gammaln((nu + 1) / 2)
            + scp.special.gammaln(nu / 2)
            + np.log(phi * np.power(np.pi * nu, 0.5))
            + (nu + 1) * np.log1p(np.power((y - mu) / phi, 2) / nu) / 2
        )
        return -1 * lp

    def llk(
        self, y: np.ndarray, mu: np.ndarray, theta: None | np.ndarray = None
    ) -> float:
        """
        log-probability of :math:`\\mathbf{y}` under this family with
        mean = :math:`\\boldsymbol{\\mu}`. Essentially sum over all elements in the vector returned
        by the :func:`lp` method.

        Log-likelihood is computed as in the ``scat`` implementation available in
        ``mgcv`` by Natalya Pya.

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).
         - ``scat`` Family implemented in ``mgcv`` by Natalya Pya, see: \
            https://github.com/cran/mgcv/blob/master/R/efam.r#L2195

        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response
            distribution corresponding to each observation.
        :type mu: np.ndarray
        :param theta: Optionally, the latest estimate of ``theta``, containing an estimate of
            the log of the scale parameter and the log of the degrees of freedom parameter.
            Array needs to be of shape (-1,1). When this is set to None, ``self.theta`` is used.
        :type theta: None | np.ndarray, optional
        :return: log-likelihood of the model under this family
        :rtype: float
        """
        return np.sum(self.lp(y, mu, theta))

    def D(
        self, y: np.ndarray, mu: np.ndarray, theta: None | np.ndarray = None
    ) -> np.ndarray:
        """
        Contribution of each observation to model Deviance (Wood, 2017; Faraway, 2016).

        Deviance contributions are computed as in the ``scat`` implementation available in
        ``mgcv`` by Natalya Pya.

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).
         - ``scat`` Family implemented in ``mgcv`` by Natalya Pya, see: \
            https://github.com/cran/mgcv/blob/master/R/efam.r#L2195

        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response
            distribution corresponding to each observation.
        :type mu: np.ndarray
        :param theta: Optionally, the latest estimate of ``theta``, containing an estimate of
            the log of the scale parameter and the log of the degrees of freedom parameter.
            Array needs to be of shape (-1,1). When this is set to None, ``self.theta`` is used.
        :type theta: None | np.ndarray, optional
        :return: a N-dimensional vector of shape (-1,1) containing the contribution of each
            observation to the overall deviance.
        :rtype: np.ndarray
        """

        # Get theta
        if theta is None:
            theta = self.theta

        # Transform to DoF parameter nu and scale parameter phi
        phi = np.exp(theta[0, 0])
        nu = np.exp(theta[1, 0]) + self.min_df

        D = (nu + 1) * np.log1p(np.power((y - mu) / phi, 2) / nu)
        return D

    def deviance(
        self, y: np.ndarray, mu: np.ndarray, theta: None | np.ndarray = None
    ) -> float:
        """
        Deviance of the model under this family: 2 * (llk_max - llk_c)
        (Wood, 2017; Faraway, 2016).

        Deviance is computed as in the ``scat`` implementation available in ``mgcv`` by
        Natalya Pya.

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).
         - ``scat`` Family implemented in ``mgcv`` by Natalya Pya, see: \
            https://github.com/cran/mgcv/blob/master/R/efam.r#L2195

        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response
            distribution corresponding to each observation.
        :type mu: np.ndarray
        :param theta: Optionally, the latest estimate of ``theta``, containing an estimate of
            the log of the scale parameter and the log of the degrees of freedom parameter.
            Array needs to be of shape (-1,1). When this is set to None, ``self.theta`` is used.
        :type theta: None | np.ndarray, optional
        :return: Deviance of the model under this family
        :rtype: float
        """

        return np.sum(self.D(y, mu, theta))

    def dDdmu(
        self, y: np.ndarray, mu: np.ndarray, theta: None | np.ndarray = None
    ) -> np.ndarray:
        """
        Computes derivative of the deviance with respect to ``mu``.

        Derivatives are computed as in the ``scat`` implementation available in ``mgcv`` by
        Natalya Pya.

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).
         - ``scat`` Family implemented in ``mgcv`` by Natalya Pya, see: \
            https://github.com/cran/mgcv/blob/master/R/efam.r#L2195

        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response
            distribution corresponding to each observation.
        :type mu: np.ndarray
        :param theta: Optionally, the latest estimate of ``theta``, containing an estimate of
            the log of the scale parameter and the log of the degrees of freedom parameter.
            Array needs to be of shape (-1,1). When this is set to None, ``self.theta`` is used.
        :type theta: None | np.ndarray, optional
        :return: a N-dimensional vector of shape ``(len(mu),1)`` containing the derivatives of
            the deviance with respect to ``mu`` for each observation.
        :rtype: np.ndarray
        """

        # Get theta
        if theta is None:
            theta = self.theta

        # Transform to DoF parameter nu and scale parameter phi
        phi = np.exp(theta[0, 0])
        nu = np.exp(theta[1, 0]) + self.min_df

        # Get gradient of deviance
        ymu = y - mu
        dDdmu = ((nu + 1) * ymu) / (
            nu * (np.power(phi, 2) * (1 + np.power(ymu / phi, 2) / nu))
        )
        return -2 * dDdmu

    def d2Ddmu(
        self, y: np.ndarray, mu: np.ndarray, theta: None | np.ndarray = None
    ) -> np.ndarray:
        """
        Computes second derivative of the deviance with respect to ``mu``. This function is only
        called after estimation has been completed to get the observed hessian at the final
        coefficient estimate.

        Derivatives are computed as in the ``scat`` implementation available in ``mgcv`` by
        Natalya Pya.

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).
         - ``scat`` Family implemented in ``mgcv`` by Natalya Pya, see: \
            https://github.com/cran/mgcv/blob/master/R/efam.r#L2195

        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response
            distribution corresponding to each observation.
        :type mu: np.ndarray
        :param theta: Optionally, the latest estimate of ``theta``, containing an estimate of
            the log of the scale parameter and the log of the degrees of freedom parameter.
            Array needs to be of shape (-1,1). When this is set to None, ``self.theta`` is used.
        :type theta: None | np.ndarray, optional
        :return: a N-dimensional vector of shape ``(len(mu),1)`` containing
            the second derivative of the deviance with respect to ``mu`` for each observation.
        :rtype: np.ndarray
        """

        # Get theta
        if theta is None:
            theta = self.theta

        # Transform to DoF parameter nu and scale parameter phi
        phi = np.exp(theta[0, 0])
        nu = np.exp(theta[1, 0]) + self.min_df

        ymu = y - mu
        sig2a = np.power(phi, 2) * (1 + np.power(ymu / phi, 2) / nu)
        d2Ddmu = (nu + 1) * (1 / (nu * sig2a) - 2 * np.power(ymu / (nu * sig2a), 2))
        return 2 * d2Ddmu

    def Ed2Ddmu(
        self, y: np.ndarray, mu: np.ndarray, theta: None | np.ndarray = None
    ) -> np.ndarray:
        """
        Computes expected second derivative of the deviance with respect to ``mu``.
        This function is used during fitting, i.e., estimation is based on Fisher weights.

        Derivatives are computed as in the ``scat`` implementation available in ``mgcv`` by
        Natalya Pya.

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).
         - ``scat`` Family implemented in ``mgcv`` by Natalya Pya, see: \
            https://github.com/cran/mgcv/blob/master/R/efam.r#L2195

        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response
            distribution corresponding to each observation.
        :type mu: np.ndarray
        :param theta: Optionally, the latest estimate of ``theta``, containing an estimate of
            the log of the scale parameter and the log of the degrees of freedom parameter.
            Array needs to be of shape (-1,1). When this is set to None, ``self.theta`` is used.
        :type theta: None | np.ndarray, optional
        :return: a N-dimensional vector of shape ``(len(mu),1)`` containing
            the expected second derivatives of the deviance with respect to ``mu`` per observation.
        :rtype: np.ndarray
        """

        # Get theta
        if theta is None:
            theta = self.theta

        # Transform to DoF parameter nu and scale parameter phi
        phi = np.exp(theta[0, 0])
        nu = np.exp(theta[1, 0]) + self.min_df

        Ed2Ddmu = np.ones_like(y) * (nu + 1) / np.power(phi, 2) / (nu + 3)
        return 2 * Ed2Ddmu

    def gradientLTheta(
        self, y: np.ndarray, mu: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """
        Computes gradient of the log-likelihood with respect to ``theta``, given ``mu``.

        Gradient is based on the derivatives of the deviance and saturated log-likelihood with
        respect to theta. The latter are computed as in the ``scat`` implementation available in
        ``mgcv`` by Natalya Pya.

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).
         - ``scat`` Family implemented in ``mgcv`` by Natalya Pya, see: \
            https://github.com/cran/mgcv/blob/master/R/efam.r#L2195

        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response
            distribution corresponding to each observation.
        :type mu: np.ndarray
        :param theta: Estimate of ``theta``, containing an estimate of
            the log of the scale parameter and the log of the degrees of freedom parameter.
            Array needs to be of shape (-1,1).
        :type theta: np.ndarray
        :return: a N-dimensional vector of shape ``(len(self.theta),1)`` containing the gradient of
            the log-likelihood with respect to theta.
        :rtype: np.ndarray
        """

        # Transform to DoF parameter nu and scale parameter phi
        phi = np.exp(theta[0, 0])
        nu = np.exp(theta[1, 0]) + self.min_df

        # Compute gradient of deviance with respect to theta
        grad = np.zeros(2).reshape(-1, 1)

        a = 1 + np.power((y - mu) / phi, 2) / nu
        ymu = y - mu
        nu1ymu = (nu + 1) * ymu
        nusig2a = nu * np.power(phi, 2) * a
        f = nu1ymu / nusig2a
        fymu = f * ymu
        # print(fymu)
        grad[0, 0] = np.sum(-2 * fymu)
        grad[1, 0] = np.sum((nu - self.min_df) * (np.log(a) - fymu / nu))

        grad *= 0.5

        # grad is now grad(l_sat) - grad(llk)

        # Now need to adjust gradient by subtracting gradient of saturated llk with respect to theta
        grad[0, 0] += len(y)
        grad[1, 0] -= len(y) * (
            (nu - self.min_df) * scp.special.digamma((nu + 1) / 2) / 2
            - (nu - self.min_df) * scp.special.digamma(nu / 2) / 2
            - 0.5 * ((nu - self.min_df) / nu)
        )

        return -1 * grad

    def hessianLTheta(
        self, y: np.ndarray, mu: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """
        Computes hessian of the log-likelihood with respect to ``theta``, given
        ``mu``.

        Hessian is based on the derivatives of the deviance and saturated log-likelihood with
        respect to theta. The latter are computed as in the ``scat`` implementation available in
        ``mgcv`` by Natalya Pya.

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).
         - ``scat`` Family implemented in ``mgcv`` by Natalya Pya, see: \
            https://github.com/cran/mgcv/blob/master/R/efam.r#L2195

        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param mu: A numpy array of shape (-1,1) containing the predicted mean for the response
            distribution corresponding to each observation.
        :type mu: np.ndarray
        :param theta: Estimate of ``theta``, containing an estimate of
            the log of the scale parameter and the log of the degrees of freedom parameter.
            Array needs to be of shape (-1,1).
        :type theta: np.ndarray
        :return: a N-dimensional vector of shape ``(len(self.theta),len(self.theta))`` containing
            the hessian of the log-likelihood with respect to theta.
        :rtype: np.ndarray
        """

        # Transform to DoF parameter nu and scale parameter phi
        phi = np.exp(theta[0, 0])
        nu = np.exp(theta[1, 0]) + self.min_df

        a = 1 + np.power((y - mu) / phi, 2) / nu
        ymu = y - mu
        nu1ymu = (nu + 1) * ymu
        nusig2a = nu * np.power(phi, 2) * a
        f = nu1ymu / nusig2a
        f1 = ymu / nusig2a
        fymu = f * ymu
        f1ymu = f1 * ymu

        H = np.zeros((2, 2))

        # Compute hessian of deviance with respect to theta
        d2Ddphi = 4 * fymu * (1 - f1ymu)
        d2Ddnu = (nu - self.min_df) * np.log(a) + ((nu - self.min_df) / nu) * np.power(
            ymu, 2
        ) * (
            -2 * (nu - self.min_df)
            - (nu + 1)
            + 2 * (nu + 1) * ((nu - self.min_df) / nu)
            - (nu + 1) * ((nu - self.min_df) / nu) * f1ymu
        ) / nusig2a
        dDdphidnu = (
            2
            * (fymu - ymu * (ymu / (np.power(phi, 2) * a)) - (fymu * f1ymu))
            * ((nu - self.min_df) / nu)
        )
        H[0, 0] = np.sum(d2Ddphi)
        H[1, 1] = np.sum(d2Ddnu)
        H[0, 1] = H[1, 0] = np.sum(dDdphidnu)

        H *= 0.5

        # Now H is H of sat llk - H of llk

        # Now need to adjust hessian by subtracting hessian of saturated llk with respect to theta
        H[1, 1] -= len(y) * (
            np.power(nu - self.min_df, 2) * scp.special.polygamma(1, (nu + 1) / 2) / 4
            + (nu - self.min_df) * scp.special.digamma((nu + 1) / 2) / 2
            - np.power(nu - self.min_df, 2) * scp.special.polygamma(1, nu / 2) / 4
            - (nu - self.min_df) * scp.special.digamma(nu / 2) / 2
            + 0.5 * np.power((nu - self.min_df) / nu, 2)
            - 0.5 * ((nu - self.min_df) / nu)
        )

        return -1 * H


class GAMLSSFamily(ABC):
    """Base-class to be implemented by families of Generalized Additive Mixed Models of Location,
    Scale, and Shape (GAMMLSS; Rigby & Stasinopoulos, 2005).

    References:
     - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, \
        Scale and Shape.
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth \
        Models.

    :param pars: Number of parameters of the distribution belonging to the random variables assumed
        to have generated the observations, e.g., 2 for the Normal: mean and standard deviation.
    :type pars: int
    :param links: Link functions for each of the parameters of the distribution.
    :type links: [Link]
    :ivar int n_par: Value passed for ``pars``.
    :ivar list[Link] links: List passed for ``links``.
    :ivar bool d_eta: A boolean indicating whether partial derivatives of llk are provided with
        respect to the linear predictor instead of parameters (i.e., the mean), defaults to False
        (derivatives are provided with respect to parameters)
    :ivar int n_d2m: How many mixed partial second derivatives are defined for this family.
    """

    def __init__(self, pars: int, links: list[Link]) -> None:
        self.n_par: int = pars
        self.links: list[Link] = links
        # Whether partial derivatives of llk are provided with respect to the linear predictor
        # instead of parameters (i.e., the mean), defaults to False (derivatives are provided with
        # respect to parameters)
        self.d_eta: bool = False
        # How many mixed derivs?
        self.n_d2m = int(pars * (pars - 1) / 2)

    @abstractmethod
    def llk(self, y: np.ndarray, *mus: np.ndarray) -> float:
        """log-probability of data under given model. Essentially sum over all elements in the
        vector returned by the :func:`lp` method.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array of shape (-1,1) containing each observation.
        :type y: np.ndarray
        :param mus: `self.n_par` np arrays - one for each parameter of the
            distribution. Each numpy array is of shape (-1,1), holding the
            expected value for a particular parmeter for each of the N observations.
        :type mus: np.ndarray
        :return: The log-probability of observing all data under the current model.
        :rtype: float
        """
        pass

    @abstractmethod
    def lp(self, y: np.ndarray, *mus: np.ndarray) -> np.ndarray:
        """Log-probability of observing every element in :math:`\\mathbf{y}` under their respective
        distribution parameterized by ``mus``.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array of shape (-1,1) containing each observed value.
        :type y: np.ndarray
        :param mus: `self.n_par` np arrays - one for each parameter of the
            distribution. Each numpy array is of shape (-1,1), holding the
            expected value for a particular parmeter for each of the N observations.
        :type mus: np.ndarray
        :return: a N-dimensional vector of shape (-1,1) containing the log-probability of observing
            each data-point under the current model.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def lcp(self, y: np.ndarray, *mus: np.ndarray) -> np.ndarray | None:
        """Log of the cumulative probability of observing a value as extreme or less extreme for
        every element in :math:`\\mathbf{y}` under their respective distribution parameterized by
        ``mus``.

        **Important:** Families for which this function is not implemented can return None.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array of shape (-1,1) containing each observed value.
        :type y: np.ndarray
        :param mus: `self.n_par` np arrays - one for each parameter of the
            distribution. Each numpy array is of shape (-1,1), holding the
            expected value for a particular parmeter for each of the N observations.
        :type mus: np.ndarray
        :return: a N-dimensional vector of shape (-1,1) containing the log cumulative probability
            of observing a value as extreme or less extreme for every data-point under the current
            model or None if this function is not implemented by the specific family.
        :rtype: np.ndarray
        """
        return None

    @abstractmethod
    def dpars(
        self, y: np.ndarray, *mus: np.ndarray, index: int, order: DerivOrder
    ) -> np.ndarray:
        """Returns partial derivatives of the log-likelihood with respect to a specific ``mu``
        (combination) or the linear predictor of that ``mu`` (if self.d_eta is True) indexed by
        ``index`` of ``order`` (first order, pure second, mixed second).

        Explanation for index:
         - if ``order == DerivOrder.d1``, ``dpars(y,*mus,i,order)`` returns first partial\
            derivative of the log-likelihood with respect to parameter ``mus[i]`` for every\
            observation in ``y``.
         - if ``order == DerivOrder.d2``, ``dpars(y,*mus,i,order)`` returns pure partial second\
            derivative of the log-likelihood with respect to parameter ``mus[i]`` for every\
            observation in ``y``.

        If ``order == DerivOrder.d2m``, ``dpars(y,*mus,i,order)`` returns up to
        ``n_par*(n_par-1)/2`` mixed partial second derivatives as follows
         - ``i=0`` = :math:`\\partial l/\\partial \\mu_1 \\partial \\mu_2`,
         - ``i=1`` = :math:`\\partial l/\\partial \\mu_1 \\partial \\mu_3`,
         - ...
         - ``i=n_par-1`` = :math:`\\partial l/\\partial \\mu_1 \\partial \\mu_{n_{par}}`,
         - ``i=n_par`` = :math:`\\partial l/\\partial \\mu_2 \\partial \\mu_3`,
         - ``i=n_par+1`` = :math:`\\partial l/\\partial \\mu_2 \\partial \\mu_4`, ... .
         - ...

        :param y: A numpy array of shape (-1,1) containing each observed value.
        :type y: np.ndarray
        :param mus: `self.n_par` np arrays - one for each parameter of the
            distribution. Each numpy array is of shape (-1,1), holding the
            expected value for a particular parmeter for each of the N observations.
        :type mus: np.ndarray
        :param index: Index for specific derivative vector to return.
        :type index: int
        :param order: Order of partial derivative.
        :type order: DerivOrder
        :return: a N-dimensional vector of shape (-1,1) containing the desired derivative evaluated
            for every observation in ``y``.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def rvs(
        self, *mus: np.ndarray, size: int = 1, seed: int | None = 0
    ) -> np.ndarray | None:
        """Returns ``size`` random samples for each of the distributions parameterized by ``mus``.

        **Note**, the returned array - if this function is implemented - will be of size
        ``(size, mus[0].shape[0])``. I.e., ``size`` random samples will be obtained for each of
        the ``mus[0].shape[0])`` distributions, all parameterized by their individual ``mus``.

        **Important:** Families for which this function is not implemented can return None.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mus: `self.n_par` np arrays - one for each parameter of the
            distribution. Each numpy array is of shape (-1,1), holding the
            expected value for a particular parmeter for each of the N observations.
        :type mus: np.ndarray
        :param size: Number of random samples to return per distribution. Defaults to 1.
        :type size: int, optional
        :param seed: Seed to use for random number generation. Defaults to 0.
        :type seed: int, optional
        :return: a numpy array of shape ``(size, mus[0].shape[0])`` containing random
            samples from every distribution parameterized by their ``mus``. Can also return None if
            this function is not implemented by the specific family.
        :rtype: np.ndarray
        """
        return None

    @abstractmethod
    def get_resid(self, y: np.ndarray, *mus: np.ndarray) -> np.ndarray | None:
        """Get standardized residuals for a GAMMLSS model (Rigby & Stasinopoulos, 2005).

        Any implementation of this function should return a vector that looks like what could be
        expected from taking ``len(y)`` independent draws from :math:`N(0,1)`. Any additional
        arguments required by a specific implementation need to be passed along as keyword arguments
        with default values.

        **Note**: Families for which no residuals are available can return None.

        References:
         - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for \
            Location, Scale and Shape.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array of shape (-1,1) containing each observed value.
        :type y: np.ndarray
        :param mus: `self.n_par` np arrays - one for each parameter of the
            distribution. Each numpy array is of shape (-1,1), holding the
            expected value for a particular parmeter for each of the N observations.
        :type mus: np.ndarray
        :return: a vector of shape (-1,1) containing standardized residuals under the current model
            or None in case residuals are not readily available.
        :rtype: np.ndarray | None
        """
        pass

    def init_coef(self, models: list[Callable]) -> np.ndarray:
        """(Optional) Function to initialize the coefficients of the model.

        Can return ``None`` , in which case random initialization will be used.

        :param models: A list of :class:`mssm.models.GAMM`'s, - each based on one of the formulas
            provided to a model.
        :type models: [mssm.models.GAMM]
        :return: A numpy array of shape (-1,1), holding initial values for all model coefficients.
        :rtype: np.ndarray
        """
        return None

    def init_lambda(self, penalties: list[Callable]) -> list[float]:
        """(Optional) Function to initialize the smoothing parameters of the model.

        Can return ``None`` , in which case random initialization will be used.

        :param penalties: A list of all penalties to be estimated by the model.
        :type penalties: [mssm.src.python.penalties.LambdaTerm]
        :return: A list, holding - for each :math:`\\lambda` parameter to be estimated - an initial
            value.
        :rtype: [float]
        """
        return None


class GAUMLSS(GAMLSSFamily):
    """Family for a Normal GAMMLSS model (Rigby & Stasinopoulos, 2005).

    This Family follows the :class:`Gaussian` family, in that we assume:
    :math:`Y_i \\sim N(\\mu_i,\\sigma_i)`. i.e., each of the :math:`N` observations
    is still believed to have been generated from an independent normally distributed RV with
    observation-specific mean.

    The important difference is that the scale parameter, :math:`\\sigma`, is now also
    observation-specific and modeled as an additive combination of smooth functions and other
    parametric terms, just like the mean is in a Normal GAM. Note, that this explicitly models
    heteroscedasticity - the residuals are no longer assumed to be i.i.d samples from
    :math:`\\sim N(0,\\sigma)`, since :math:`\\sigma` can now differ between residual realizations.

    References:
     - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for \
        Location, Scale and Shape.
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
        Smooth Models.

    :param links: Link functions for the mean and standard deviation. Defaults to
        ``links=[Identity(),LOGb(-0.0001)]``.
    :type links: [Link]
    :ivar list[Link] links: List passed for ``links``.
    """

    def __init__(self, links: list[Link] = [Identity(), LOGb(-0.0001)]) -> None:
        super().__init__(2, links)

    def lp(self, y: np.ndarray, *mus: np.ndarray) -> np.ndarray:
        """Log-probability of observing every value in y under their respective Normal with
        observation-specific mean and standard deviation.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array containing each observed value.
        :type y: np.ndarray
        :param mus: 2 np arrays - one for the mean and one for the standard deviation for the
            response distribution corresponding to each of N observations. Each numpy array is of
            shape (N,1).
        :type mus: np.ndarray
        :return: a N-dimensional vector containing the log-probability of observing each data-point
            under the current model.
        :rtype: np.ndarray
        """
        mu = mus[0]
        sigma = mus[1]
        return scp.stats.norm.logpdf(y, loc=mu, scale=sigma)

    def lcp(self, y: np.ndarray, *mus: np.ndarray) -> np.ndarray:
        """Log of the cumulative probability of observing every value in y under their respective
        Normal with observation-specific mean and standard deviation.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array containing each observed value.
        :type y: np.ndarray
        :param mus: 2 np arrays - one for the mean and one for the standard deviation for the
            response distribution corresponding to each of N observations. Each numpy array is of
            shape (N,1).
        :type mus: np.ndarray
        :return: a N-dimensional vector containing the log of the cumulative probability of
            observing each data-point under the current model.
        :rtype: np.ndarray
        """
        mu = mus[0]
        sigma = mus[1]
        return scp.stats.norm.logcdf(y, loc=mu, scale=sigma)

    def llk(self, y: np.ndarray, *mus: np.ndarray) -> float:
        """log-probability of data under given model. Essentially sum over all elements in the
        vector returned by the :func:`lp` method.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array containing each observation.
        :type y: np.ndarray
        :param mus: 2 np arrays - one for the mean and one for the standard deviation for the
            response distribution corresponding to each of N observations. Each numpy array is of
            shape (N,1).
        :type mus: np.ndarray
        :return: The log-probability of observing all data under the current model.
        :rtype: float
        """
        mu = mus[0]
        sigma = mus[1]
        return np.sum(self.lp(y, mu, sigma))

    def dpars(
        self, y: np.ndarray, *mus: np.ndarray, index: int, order: DerivOrder
    ) -> np.ndarray:
        """Returns partial derivatives of the log-likelihood with respect to the mean and standard
        deviation or a combination indexed by ``index`` of ``order`` (first order, pure second,
        mixed second).

        All derivatives taken from gamlss.dist: https://github.com/gamlss-dev/gamlss.dist see
        also: Rigby, R. A., & Stasinopoulos, D. M. (2005).

        References:
         - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for \
            Location, Scale and Shape.

        :param y: A numpy array of shape (-1,1) containing each observed value.
        :type y: np.ndarray
        :param mus: 2 np arrays - one for the mean and one for the standard deviation for the
            response distribution corresponding to each of N observations. Each numpy array is of
            shape (N,1).
        :type mus: np.ndarray
        :param index: Index for specific derivative vector to return.
        :type index: int
        :param order: Order of partial derivative.
        :type order: DerivOrder
        :return: a N-dimensional vector of shape (-1,1) containing the desired derivative evaluated
            for every observation in ``y``.
        :rtype: np.ndarray
        """
        mu = mus[0]
        sigma = mus[1]
        if order == DerivOrder.d1:
            if index == 0:
                return (1 / np.power(sigma, 2)) * (y - mu)
            elif index == 1:
                return (np.power(y - mu, 2) - np.power(sigma, 2)) / (np.power(sigma, 3))
            else:
                raise ValueError("No Derivative of order d1 exists at index > 1.")

        elif order == DerivOrder.d2:
            if index == 0:
                return -(1 / np.power(sigma, 2))
            elif index == 1:
                return -(2 / np.power(sigma, 2))
            else:
                raise ValueError("No Derivative of order d2 exists at index > 1.")

        elif order == DerivOrder.d2m:
            if index == 0:
                return np.zeros_like(y)
            else:
                raise ValueError("No Derivative of order d2m exists at index > 0.")

        else:
            raise ValueError("No Derivative > order d2m exists.")

    def rvs(self, *mus: np.ndarray, size: int = 1, seed: int | None = 0) -> np.ndarray:
        """Returns ``size`` random samples for each of the distributions parameterized by ``mu``
        and ``sigma``.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

       :param mus: 2 np arrays - one for the mean and one for the standard deviation for the
            response distribution corresponding to each of N observations. Each numpy array is of
            shape (N,1).
        :type mus: np.ndarray
        :param size: Number of random samples to return per distribution. Defaults to 1.
        :type size: int, optional
        :param seed: Seed to use for random number generation. Defaults to 0.
        :type seed: int, optional
        :return: a numpy array of shape ``(size, mus[0].shape[0])`` containing random
            samples from every distribution parameterized by ``mu`` and ``sigma``.
        :rtype: np.ndarray
        """
        mu = mus[0]
        sigma = mus[1]
        return scp.stats.norm.rvs(
            size=(size, mu.shape[0]),
            loc=mu.flatten(),
            scale=sigma.flatten(),
            random_state=seed,
        )

    def get_resid(self, y: np.ndarray, *mus: np.ndarray) -> float:
        """Get standardized residuals for a Normal GAMMLSS model (Rigby & Stasinopoulos, 2005).

        Essentially, each residual should reflect a realization of a normal with mean zero and
        observation-specific standard deviation. After scaling each residual by their
        observation-specific standard deviation we should end up with standardized
        residuals that can be expected to be i.i.d :math:`\\sim N(0,1)` - assuming that our model
        is correct.

        :param y: A numpy array containing each observation.
        :type y: np.ndarray
        :param mus: 2 np arrays - one for the mean and one for the standard deviation for the
            response distribution corresponding to each of N observations. Each numpy array is of
            shape (N,1).
        :type mus: np.ndarray
        :return: A list of standardized residuals that should be ~ N(0,1) if the model is correct.
        :rtype: np.ndarray
        """
        mu = mus[0]
        sigma = mus[1]
        res = y - mu
        res /= sigma
        return res

    def init_coef(self, models: list[Callable]) -> np.ndarray:
        """Function to initialize the coefficients of the model.

        Fits a GAMM for the mean and initializes all coef. for the standard deviation to 1.

        :param models: A list of :class:`mssm.models.GAMM`'s, - each based on one of the formulas
            provided to a model.
        :type models: [mssm.models.GAMM]
        :return: A numpy array of shape (-1,1), holding initial values for all model coefficients.
        :rtype: np.ndarray
        """

        mean_model = models[0]
        mean_model.family = Gaussian(self.links[0])
        mean_model.fit(progress_bar=False)

        m_coef, _ = mean_model.get_pars()
        coef = np.concatenate(
            (
                m_coef.reshape(-1, 1),
                np.ones((models[1].formulas[0].n_coef)).reshape(-1, 1),
            )
        )

        return coef


class MULNOMLSS(GAMLSSFamily):
    """Family for a Multinomial GAMMLSS model (Rigby & Stasinopoulos, 2005).

    This Family assumes that each observation :math:`y_i` corresponds to one of :math:`K` classes
    (labeled as 0, ..., :math:`K`) and reflects a realization of an independent RV :math:`Y_i` with
    observation-specific probability mass function defined over the :math:`K` classes. These
    :math:`K` probabilities - that :math:`Y_i` takes on class 1, ..., :math:`K` - are modeled as
    additive combinations of smooth functions of covariates and other parametric terms.

    As an example, consider a visual search experiment where :math:`K` distractors are presented on
    a computer screen together with a single target and subjects are instructed to find the target
    and fixate it. With a Multinomial model we can estimate how the probability of looking at each
    of the :math:`K` stimuli on the screen changes (smoothly) over time and as a function of other
    predictor variables of interest (e.g., contrast of stimuli, dependening on whether parfticipants
    are instructed to be fast or accurate).

    References:
     - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, \
        Scale and Shape.
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth \
        Models.

    :param pars: K-1, i.e., 1- Number of classes or the number of linear predictors.
    :type pars: int
    :ivar int n_par: Value passed for ``pars``.
    """

    def __init__(self, pars: int) -> None:
        super().__init__(pars, [LOG() for _ in range(pars)])

        # Derivatives are with respect to linear predictor
        self.d_eta: bool = True
        self.__dpairs: list[tuple[int, int]] = []  # Deriv pairs at given index
        for i in range(self.n_par):
            for j in range(i + 1, self.n_par):
                self.__dpairs.append((i, j))

    def lp(self, y: np.ndarray, *mus: np.ndarray) -> np.ndarray:
        """Log-probability of observing classes in ``y`` under current model.

        Our DV consists of K classes but we essentially enforce a sum-to zero constraint on the DV
        so that we end up modeling only K-1 (non-normalized) probabilities of observing class k
        (for all k except k==0) as an additive combination of smooth functions of our covariates
        and other parametric terms. The probability of observing class 0 as well as the normalized
        probabilities of observing each other class can readily be computed from these K-1
        non-normalized probabilities. This is explained quite well on Wikipedia (see refs).

        Specifically, the probability of the outcome being class k is simply:

        :math:`p(Y_i == k) = \\mu_k / (1 + \\sum_j^{K-1} \\mu_j)` where :math:`\\mu_k` is the
        aforementioned non-normalized probability of observing class :math:`k` - which is simply set
        to 1 for class :math:`k==0` (this follows from the sum-to-zero constraint; see Wikipedia).

        So, the log-prob of the outcome being class k is:

        :math:`log(p(Y_i == k)) = log(\\mu_k) - log(1 + \\sum_j^{K-1} \\mu_j)`

        References:

         - Wikipedia. https://en.wikipedia.org/wiki/Multinomial_logistic_regression
         - gamlss.dist on Github (see Rigby & Stasinopoulos, 2005). \
            https://github.com/gamlss-dev/gamlss.dist/blob/main/R/MN4.R
         - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for \
            Location, Scale and Shape.

        :param y: A numpy array of shape (-1,1) containing each observed class, every element must
            be larger than or equal to 0 and smaller than `self.n_par + 1`.
        :type y: np.ndarray
        :param mus: K-1 (`self.n_par`) numpy arrays of shape (-1,1), each containing the
            non-normalized probabilities of observing class k for every observation.
        :type mus: np.ndarray
        :return: a N-dimensional vector containing the log-probability of observing each data-point
            under the current model.
        :rtype: np.ndarray
        """
        # Note, log(1) = 0, so we can simply initialize to -log(1 + \\sum_j^{K-1} mu_j)
        # and then add for the K-1 probs we actually modeled.
        lp = -np.log(np.sum(mus, axis=0) + 1)

        for pi in range(self.n_par):
            lp[y == (pi + 1)] += np.log(mus[pi])[y == (pi + 1)]

        return lp

    def lcp(self, y: np.ndarray, *mus: np.ndarray) -> np.ndarray:
        """Cumulative log-probability of observing classes in ``y`` under current model.

        References:
         - Wikipedia. https://en.wikipedia.org/wiki/Multinomial_logistic_regression
         - gamlss.dist on Github (see Rigby & Stasinopoulos, 2005). \
            https://github.com/gamlss-dev/gamlss.dist/blob/main/R/MN4.R
         - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for \
            Location, Scale and Shape.

        :param y: A numpy array of shape (-1,1) containing each observed class, every element must
            be larger than or equal to 0 and smaller than `self.n_par + 1`.
        :type y: np.ndarray
        :param mus: K-1 (`self.n_par`) numpy arrays of shape (-1,1), each containing the
            non-normalized probabilities of observing class k for every observation.
        :type mus: np.ndarray
        :return: a N-dimensional vector containing the cumulatiove log-probability of observing each
            data-point under the current model.
        :rtype: np.ndarray
        """
        # Note, log(1) = 0, so we can simply initialize to -log(1 + \\sum_j^{K-1} mu_j)
        # and then add for the K-1 probs we actually modeled.
        lcp = np.zeros_like(mus[0])

        for pi in range(self.n_par):
            lcp[y > pi] += mus[pi][y > pi]

        # Normalize for all but class 0
        lcp[y > 0] = np.log(lcp[y > 0])
        lcp -= np.log(np.sum(mus, axis=0) + 1)

        # Fix cum log prob for class zero which is log(1) = 0
        lcp[y == 0] = 0

        return lcp

    def llk(self, y: np.ndarray, *mus: np.ndarray) -> float:
        """log-probability of data under given model. Essentially sum over all elements in the
        vector returned by the :func:`lp` method.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array of shape (-1,1) containing each observed class, every element must
            be larger than or equal to 0 and smaller than `self.n_par + 1`.
        :type y: np.ndarray
        :param mus: K-1 (`self.n_par`) numpy arrays of shape (-1,1), each containing the
            non-normalized probabilities of observing class k for every observation.
        :type mus: np.ndarray
        :return: The log-probability of observing all data under the current model.
        :rtype: float
        """
        return np.sum(self.lp(y, *mus))

    def dpars(
        self, y: np.ndarray, *mus: np.ndarray, index: int, order: DerivOrder
    ) -> np.ndarray:
        """Returns partial derivatives of the log-likelihood with respect to the linear predictor
        of a specifc ``mu``.
        deviation or a combination indexed by ``index`` of ``order`` (first order, pure second,
        mixed second).

        All derivatives taken from gamlss.r in mgcv:
        https://github.com/cran/mgcv/blob/master/R/gamlss.r#L1224 and have been adapted to work
        in Python code see also: Rigby, R. A., & Stasinopoulos, D. M. (2005). Derivatives are
        implemented with respect to linear predictor.

        References:
         - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for \
            Location, Scale and Shape.

        :param y: A numpy array of shape (-1,1) containing each observed class, every element must
            be larger than or equal to 0 and smaller than `self.n_par + 1`.
        :type y: np.ndarray
        :param mus: K-1 (`self.n_par`) numpy arrays of shape (-1,1), each containing the
            non-normalized probabilities of observing class k for every observation.
        :type mus: np.ndarray
        :param index: Index for specific derivative vector to return.
        :type index: int
        :param order: Order of partial derivative.
        :type order: DerivOrder
        :return: a N-dimensional vector of shape (-1,1) containing the desried derivative evaluated
            for every observation in ``y``.
        :rtype: np.ndarray
        """

        if order == DerivOrder.d1:
            if index < self.n_par:
                dy1 = -(mus[index] / (np.sum(mus, axis=0) + 1))
                dy1[y == (index + 1)] += 1
                return dy1
            else:
                raise ValueError(f"No derivative of order d1 exists for index {index}")

        elif order == DerivOrder.d2:
            if index < self.n_par:
                norm = np.sum(mus, axis=0) + 1
                dy1 = -(mus[index] / norm)
                dy2 = dy1 + np.power(mus[index], 2) / np.power(norm, 2)
                return dy2
            else:
                raise ValueError(f"No derivative of order d2 exists for index {index}")

        elif order == DerivOrder.d2m:
            if index < len(self.__dpairs):
                # Get pair for specified index
                didx = self.__dpairs[index]
                i = didx[0]
                j = didx[1]
                norm = np.sum(mus, axis=0) + 1
                dy2m = (mus[i] * mus[j]) / np.power(norm, 2)
                return dy2m
            else:
                raise ValueError(f"No derivative of order d2m exists for index {index}")

        raise ValueError("No Derivative > order d2m exists.")

    def rvs(self, *mus: np.ndarray, size: int = 1, seed: int | None = 0) -> np.ndarray:
        """Returns ``size`` random samples from the Multinomial distribution assumed for every
        observation parameterized by a row of ``mus``.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mus: K-1 (`self.n_par`) numpy arrays of shape (-1,1), each containing the
            non-normalized probabilities of observing class k for every observation.
        :type mus: np.ndarray
        :param size: Number of random samples to return per distribution. Defaults to 1.
        :type size: int, optional
        :param seed: Seed to use for random number generation. Defaults to 0.
        :type seed: int, optional
        :return: a numpy array of shape ``(size, mus[0].shape[0])`` containing random
            samples from every Multinomial distribution.
        :rtype: np.ndarray
        """
        np_gen = np.random.default_rng(seed)
        y_dim = len(mus[0])
        support = np.zeros((y_dim, self.n_par + 1))
        K = np.arange(self.n_par + 1)

        for k in K:
            support[:, k] = self.lp(np.zeros_like(mus[0]) + k, *mus)[:, 0]

        samples = np.zeros((size, y_dim))
        for i in range(y_dim):
            samples[:, i] = np_gen.choice(a=K, size=size, p=np.exp(support[i, :]))

        return samples

    def get_resid(
        self, y: np.ndarray, *mus: np.ndarray, seed: int | None = 0
    ) -> np.ndarray:
        """Returns randomized quantile residuals of a Multinomial model as defined
        by Dunn & Smyth (1996). See also Rigby, R. A., & Stasinopoulos, D. M. (2005)

        References:
         - Dunn, P. K., & Smyth, G. K. (1996). Randomized Quantile Residuals. \
            https://doi.org/10.2307/1390802
         - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for \
            Location, Scale and Shape.

        :param y: A numpy array of shape (-1,1) containing each observed class, every element must
            be larger than or equal to 0 and smaller than `self.n_par + 1`.
        :type y: np.ndarray
        :param mus: K-1 (`self.n_par`) numpy arrays of shape (-1,1), each containing the
            non-normalized probabilities of observing class k for every observation.
        :type mus: np.ndarray
        :param seed: Seed to use for the random part of the residual calculation. Residual vector
            should be replicated for multiple values. Defaults to 0.
        :type seed: int | None, optional
        :return: Array of shape (-1,1) holding randomized quantil residuals.
        """
        np_gen = np.random.default_rng(seed)
        res = np.zeros_like(y, dtype=np.float64)

        # Determine boundaries for intervals for randomized quantile
        # residuals as defined by Dunn and Smyth (1996)
        intervals = np.zeros((len(y), 2))

        # Order of comparison: class 0 > class n_par > class n_par -1, ... > class 1
        comp = [0, *np.arange(self.n_par, 0, -1)]
        for yi in range(len(comp) - 1):

            yeval = np.zeros_like(y)[y == comp[yi]]

            for ii, yval in enumerate([comp[yi + 1], comp[yi]]):

                # Set y to correct value
                yeval[:] = yval

                intervals[y.flatten() == comp[yi], ii] = np.exp(
                    self.lcp(yeval, *[mu[y == comp[yi]] for mu in mus])
                )

        # First class
        intervals[y.flatten() == 1, 1] = np.exp(
            self.lcp(y[y == 1], *[mu[y == 1] for mu in mus])
        )

        for i in range(len(y)):
            u = scp.stats.uniform.rvs(
                size=1,
                loc=intervals[i, 0],
                scale=intervals[i, 1] - intervals[i, 0],
                random_state=np_gen,
            )

            # Uniform residual
            res[i, 0] = u[0]

        # Inverse cdf transform
        res = scp.stats.norm.ppf(res)

        return res

    def get_probs(self, *mus: np.ndarray) -> np.ndarray:
        """Get the probability of being in each of the K classes for every row of ``mus``.

        :param mus: K-1 (`self.n_par`) numpy arrays of shape (-1,1), each containing the
            non-normalized probabilities of observing class k for every observation.
        :type mus: np.ndarray
        :return: Array of size ``(len(mus[0]),self.n_par+1)`` holding for each observation the
            probability of being in each of the ``K=self.n_par+1`` classes.
        :rtype: np.ndarray
        """
        probs = np.zeros((len(mus[0]), self.n_par + 1))

        # Iterate over classes
        K = np.arange(self.n_par + 1)
        for k in K:
            probs[:, k] = np.exp(self.lp(np.zeros_like(mus[0]) + k, *mus))[:, 0]

        return probs


class GAMMALS(GAMLSSFamily):
    """Family for a GAMMA GAMMLSS model (Rigby & Stasinopoulos, 2005).

    This Family follows the :class:`Gamma` family, in that we assume:
    :math:`Y_i \\sim \\Gamma(\\mu_i,\\phi_i)`. The difference to the :class:`Gamma` family is that
    we now also model :math:`\\phi` as an additive combination of smooth variables and other
    parametric terms. The Gamma distribution is usually not expressed in terms of the mean and
    scale (:math:`\\phi`) parameter but rather in terms of a shape and rate parameter - called
    :math:`\\alpha` and :math:`\\beta` respectively. Wood (2017) provides :math:`\\alpha = 1/\\phi`.
    With this we can obtain :math:`\\beta = 1/\\phi/\\mu` (see the source-code for :func:`lp` method
    of the :class:`Gamma` family for details).

    References:
     - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for Location, \
        Scale and Shape.
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
        Smooth Models.

    :param links: Link functions for the mean and standard deviation. Default is
        ``links=[LOG(),LOGb(-0.0001)]``.
    :type links: [Link]
    :ivar list[Link] links: List passed for ``links``.
    """

    def __init__(self, links: list[Link] = [LOG(), LOGb(-0.0001)]) -> None:
        super().__init__(2, links)

    def lp(self, y: np.ndarray, *mus: np.ndarray) -> np.ndarray:
        """Log-probability of observing every proportion in :math:`\\mathbf{y}` under their
        respective Gamma with mean = :math:`\\boldsymbol{\\mu}` and
        scale = :math:`\\boldsymbol{\\phi}`.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array containing each observed value.
        :type y: np.ndarray
        :param mus: 2 np arrays - one for the mean and one for the scale for the
            response distribution corresponding to each of N observations. Each numpy array is of
            shape (N,1).
        :type mus: np.ndarray
        :return: a N-dimensional vector containing the log-probability of observing each data-point
            under the current model.
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
        mu = mus[0]
        scale = mus[1]
        alpha = 1 / scale
        beta = alpha / mu
        return scp.stats.gamma.logpdf(y, a=alpha, scale=(1 / beta))

    def lcp(self, y: np.ndarray, *mus: np.ndarray) -> np.ndarray:
        """Log of the cumulative probability of observing every value in y under their respective
        Gamma with mean = :math:`\\boldsymbol{\\mu}` and scale = :math:`\\boldsymbol{\\phi}`.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array containing each observed value.
        :type y: np.ndarray
        :param mus: 2 np arrays - one for the mean and one for the scale for the
            response distribution corresponding to each of N observations. Each numpy array is of
            shape (N,1).
        :type mus: np.ndarray
        :return: a N-dimensional vector containing the log of the cumulative probability of
            observing each data-point under the current model.
        :rtype: np.ndarray
        """
        mu = mus[0]
        scale = mus[1]
        alpha = 1 / scale
        beta = alpha / mu
        return scp.stats.gamma.logcdf(y, a=alpha, scale=(1 / beta))

    def llk(self, y: np.ndarray, *mus: np.ndarray) -> float:
        """log-probability of data under given model. Essentially sum over all elements in the
        vector returned by the :func:`lp` method.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param y: A numpy array containing each observation.
        :type y: np.ndarray
        :param mus: 2 np arrays - one for the mean and one for the scale for the
            response distribution corresponding to each of N observations. Each numpy array is of
            shape (N,1).
        :type mus: np.ndarray
        :return: The log-probability of observing all data under the current model.
        :rtype: float
        """
        mu = mus[0]
        scale = mus[1]
        return np.sum(self.lp(y, mu, scale))

    def dpars(
        self, y: np.ndarray, *mus: np.ndarray, index: int, order: DerivOrder
    ) -> np.ndarray:
        """Returns partial derivatives of the log-likelihood with respect to the mean and scale
        or a combination indexed by ``index`` of ``order`` (first order, pure second,
        mixed second).

        All derivatives taken from gamlss.dist: https://github.com/gamlss-dev/gamlss.dist see
        also: Rigby, R. A., & Stasinopoulos, D. M. (2005).

        References:
         - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized Additive Models for \
            Location, Scale and Shape.

        :param y: A numpy array of shape (-1,1) containing each observed value.
        :type y: np.ndarray
        :param mus: 2 np arrays - one for the mean and one for the scale for the
            response distribution corresponding to each of N observations. Each numpy array is of
            shape (N,1).
        :param index: Index for specific derivative vector to return.
        :type index: int
        :param order: Order of partial derivative.
        :type order: DerivOrder
        :return: a N-dimensional vector of shape (-1,1) containing the desired derivative evaluated
            for every observation in ``y``.
        :rtype: np.ndarray
        """
        mu = mus[0]
        scale = mus[1]

        if order == DerivOrder.d1:
            if index == 0:
                return (y - mu) / (scale * np.power(mu, 2))
            elif index == 1:
                return (1 / np.power(scale, 2)) * (
                    (y / mu)
                    - np.log(y)
                    + np.log(mu)
                    + np.log(scale)
                    - 1
                    + scp.special.digamma(1 / (scale))
                )
            else:
                raise ValueError("No Derivative of order d1 exists at index > 1.")

        elif order == DerivOrder.d2:
            if index == 0:
                return -1 / (scale * np.power(mu, 2))
            elif index == 1:
                return (1 / np.power(scale, 3)) - (
                    1 / np.power(scale, 4)
                ) * scp.special.polygamma(1, 1 / scale)
            else:
                raise ValueError("No Derivative of order d2 exists at index > 1.")

        elif order == DerivOrder.d2m:
            if index == 0:
                return np.zeros_like(y)
            else:
                raise ValueError("No Derivative of order d2m exists at index > 0.")

        else:
            raise ValueError("No Derivative > order d2m exists.")

    def rvs(self, *mus: np.ndarray, size: int = 1, seed: int | None = 0) -> np.ndarray:
        """Returns ``size`` random samples for each of the distributions parameterized by ``mu``
        and ``scale``.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param mus: 2 np arrays - one for the mean and one for the scale for the
            response distribution corresponding to each of N observations. Each numpy array is of
            shape (N,1).
        :type mus: np.ndarray
        :param size: Number of random samples to return per distribution. Defaults to 1.
        :type size: int, optional
        :param seed: Seed to use for random number generation. Defaults to 0.
        :type seed: int, optional
        :return: a numpy array of shape ``(size, mus[0].shape[0])`` containing random
            samples from every distribution parameterized by ``mu`` and ``scale``.
        :rtype: np.ndarray
        """
        mu = mus[0]
        scale = mus[1]
        alpha = 1 / scale
        beta = alpha / mu
        return scp.stats.gamma.rvs(
            size=(size, mu.shape[0]),
            a=alpha.flatten(),
            scale=(1 / beta.flatten()),
            random_state=seed,
        )

    def get_resid(self, y: np.ndarray, *mus: np.ndarray) -> np.ndarray:
        """Get standardized residuals for a Gamma GAMMLSS model (Rigby & Stasinopoulos, 2005).

        Essentially, to get a standaridzed residual vector we first have to account for the
        mean-variance relationship of our RVs (which we also have to do for the :class:`Gamma`
        family) - for this we can simply compute deviance residuals again (see Wood, 2017).
        These should be :math:`\\sim N(0,\\phi_i)` (where :math:`\\phi_i` is the element in
        ``scale`` for a specific observation) - so if we divide each of those by the
        observation-specific scale we can expect the resulting standardized residuals to be
        :math:` \\sim N(0,1)` if the model is correct.

        References:
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).


        :param y: A numpy array containing each observation.
        :type y: np.ndarray
        :param mus: 2 np arrays - one for the mean and one for the scale for the
            response distribution corresponding to each of N observations. Each numpy array is of
            shape (N,1).
        :type mus: np.ndarray
        :return: A list of standardized residuals that should be ~ N(0,1) if the model is correct.
        :rtype: np.ndarray
        """
        mu = mus[0]
        scale = mus[1]
        res = np.sign(y - mu) * np.sqrt(Gamma().D(y, mu) / scale)
        return res

    def init_coef(self, models: list[Callable]) -> np.ndarray:
        """Function to initialize the coefficients of the model.

        Fits a GAMM for the mean and initializes all coef. for the scale parameter to 1.

        :param models: A list of :class:`mssm.models.GAMM`'s, - each based on one of the formulas
            provided to a model.
        :type models: [mssm.models.GAMM]
        :return: A numpy array of shape (-1,1), holding initial values for all model coefficients.
        :rtype: np.ndarray
        """

        mean_model = models[0]
        mean_model.family = Gamma(self.links[0])
        mean_model.fit(progress_bar=False, max_inner=1)

        m_coef, _ = mean_model.get_pars()
        coef = np.concatenate(
            (
                m_coef.reshape(-1, 1),
                np.ones((models[1].formulas[0].n_coef)).reshape(-1, 1),
            )
        )
        return coef


class GSMMFamily(ABC):
    """Base-class for General Smooth "families" as discussed by Wood, Pya, & Säfken (2016).
    For estimation of :class:`mssm.models.GSMM` models via ``L-qEFS`` (Krause et al., submitted) it
    is sufficient to implement :func:`llk`. :func:`gradient` and :func:`hessian` can then simply
    return ``None``. For exact estimation via Newton's method, the latter two functions need to be
    implemented and have to return the gradient and hessian at the current coefficient estimate
    respectively.

    Additional parameters needed for likelihood, gradient, or hessian evaluation can be passed
    along via the ``init`` of a specific implementation.

    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
        Smooth Models.
     - Nocedal & Wright (2006). Numerical Optimization. Springer New York.
     - Krause et al. (submitted). The Mixed-Sparse-Smooth-Model Toolbox (MSSM): Efficient \
        Estimation and Selection of Large Multi-Level Statistical Models. \
        https://doi.org/10.48550/arXiv.2506.13132

    :param pars: Number of parameters of the likelihood for which an additive (mixed) model is
        specified. **Note**, that extra parameters that are constant (and thus do not need an
        extra :class:`mssm.src.python.formula.Formula` specified) but nevertheless need to be
        estimated can be handled via the ``extra_coef`` argument.
    :type pars: int
    :param links: List of Link functions for each parameter of the likelihood,
        e.g., `links=[Identity(),LOG()]`.
    :type links: [Link]
    :ivar int n_par: Value passed for ``pars``.
    :ivar list[Link] links: List passed for ``links``.
    :ivar int, optional extra_coef: Number of extra coefficients required by specific family for
        parameters of the log-likelihood that are constant (i.e., not a function of predictor
        variables) or ``None``. If this is not set to ``None``, ``mssm`` will automatically append
        ``extra_coef`` elements to the coefficient vector passed to the log-likelihood and gradient
        methods of this family. Additionally, ``coef_split_idx`` will be modified, so that the last
        list of the split holds the ``extra_coef``. By default set to ``None`` and changed to
        ``int`` by specific families requiring this.

    """

    def __init__(self, pars: int, links: list[Link]) -> None:
        self.n_par: int = pars
        self.links: list[Link] = links
        self.extra_coef: int | None = None

    @abstractmethod
    def llk(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray | None],
        Xs: list[scp.sparse.csc_array | None],
    ) -> float:
        """log-probability of data under given model.

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).
         - Krause et al. (submitted). The Mixed-Sparse-Smooth-Model Toolbox (MSSM): Efficient \
            Estimation and Selection of Large Multi-Level Statistical Models. \
            https://doi.org/10.48550/arXiv.2506.13132

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each paramter of the llk.
        :type coef_split_idx: [int]
        :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as
            ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the
            actual observed data is passed along via the first formula (so it is stored in
            ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula,
            then ``ys`` contains ``None`` at their indices to save memory.
        :type ys: list[np.ndarray | None]
        :param Xs: A list of sparse model matrices per likelihood parameter. Might contain ``None``
            at indices for matrices which were flagged as "do not build" via the ``build_mat``
            argument of the :func:`mssm.models.GSMM.fit` method.
        :type Xs: list[scp.sparse.csc_array | None]
        :return: The log-likelihood evaluated at ``coef``.
        :rtype: float
        """
        pass

    def gradient(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray | None],
        Xs: list[scp.sparse.csc_array | None],
    ) -> np.ndarray:
        """Function to evaluate the gradient of the llk at current coefficient estimate ``coef``.

        By default relies on numerical differentiation as implemented in scipy to approximate the
        Gradient from the implemented log-likelihood function. See the link in the references for
        more details.

        References:
           - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.
           - ``scipy.optimize.approx_fprime``: at \
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.approx_fprime.html
           - Krause et al. (submitted). The Mixed-Sparse-Smooth-Model Toolbox (MSSM): Efficient \
            Estimation and Selection of Large Multi-Level Statistical Models. \
            https://doi.org/10.48550/arXiv.2506.13132

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each paramter of the llk.
        :type coef_split_idx: [int]
        :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as
            ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the
            actual observed data is passed along via the first formula (so it is stored in
            ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula,
            then ``ys`` contains ``None`` at their indices to save memory.
        :type ys: list[np.ndarray | None]
        :param Xs: A list of sparse model matrices per likelihood parameter. Might contain ``None``
            at indices for matrices which were flagged as "do not build" via the ``build_mat``
            argument of the :func:`mssm.models.GSMM.fit` method.
        :type Xs: list[scp.sparse.csc_array | None]
        :return: The Gradient of the log-likelihood evaluated at ``coef`` as numpy array of
            shape (-1,1).
        :rtype: np.ndarray
        """

        def llk_wrap(x: np.ndarray) -> float:
            return self.llk(x.reshape(-1, 1), coef_split_idx, ys, Xs)

        grad = scp.optimize.approx_fprime(coef.flatten(), llk_wrap)
        return grad.reshape(-1, 1)

    def jcolhessian(
        self,
        j: int,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray | None],
        Xs: list[scp.sparse.csc_array | None],
    ) -> np.ndarray:
        """(Optional) method to compute only column ``j`` of the Hessian of the log-likelihood.

        By default the method relies on a finite difference approximation to evaluate the
        column of the Hessian of the llk. Result is returned as a np.array.

        :param j: Index of the column to approximate.
        :type jcols: int
        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each paramter of the llk.
        :type coef_split_idx: [int]
        :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as
            ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the
            actual observed data is passed along via the first formula (so it is stored in
            ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula,
            then ``ys`` contains ``None`` at their indices to save memory.
        :type ys: list[np.ndarray | None]
        :param Xs: A list of sparse model matrices per likelihood parameter. Might contain ``None``
            at indices for matrices which were flagged as "do not build" via the ``build_mat``
            argument of the :func:`mssm.models.GSMM.fit` method.
        :type Xs: list[scp.sparse.csc_array | None]
        :return: Finite difference approximation of column ``j`` of the Hessian of the llk as an
            array of shape (-1,1)
        :rtype: np.ndarray
        """

        def __d2llkj(r):
            # Function to evaluate Hessian column via finite difference approximation
            n_coef = copy.deepcopy(coef)
            n_coef[j] = r

            n_grad = self.gradient(n_coef, coef_split_idx, ys, Xs)

            return n_grad.flatten()

        def vectorized_d2(r):
            return np.apply_along_axis(__d2llkj, axis=0, arr=r)

        # Column j of the hessian
        Hsk = scp.differentiate.jacobian(vectorized_d2, coef[j], order=2)

        return Hsk.df

    def jhessian(
        self,
        jcols: np.ndarray,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray | None],
        Xs: list[scp.sparse.csc_array | None],
        n_c: int = 1,
    ) -> scp.sparse.csc_array:
        """(Optional) method to compute a sparse approximation to the Hessian of the llk, containing
        only the ``j`` columns and rows of the Hessian indexed by ``jcols``.

        By default the method relies on a finite difference approximation to evaluate the ``j``
        columns of the Hessian of the llk. Result is returned as symmetric sparse matrix.

        :param jcols: Array holding indices of columns to approximate.
        :type jcols: np.ndarray
        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each paramter of the llk.
        :type coef_split_idx: [int]
        :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as
            ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the
            actual observed data is passed along via the first formula (so it is stored in
            ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula,
            then ``ys`` contains ``None`` at their indices to save memory.
        :type ys: list[np.ndarray | None]
        :param Xs: A list of sparse model matrices per likelihood parameter. Might contain ``None``
            at indices for matrices which were flagged as "do not build" via the ``build_mat``
            argument of the :func:`mssm.models.GSMM.fit` method.
        :type Xs: list[scp.sparse.csc_array | None]
        :param n_c: Number of cores to use to parallelize computation over ``j`` cols, defaults to
            1.
        :type n_c: int, optional
        :return: Finite difference approximation matrix which is symmetric sparse matrix with
            ``jcols`` rows and columns set to finite difference approximation of columns of
            Hessian of llk
        :rtype: scp.sparse.csc_array
        """

        ccols = []

        Hdat = []
        Hrows = []
        Hcols = []

        Hdim = len(coef)

        if HAS_MP and n_c > 1:
            # Compute columns in parallel
            args = zip(
                jcols,
                repeat(coef),
                repeat(coef_split_idx),
                repeat(ys),
                repeat(Xs),
            )

            with mp.Pool(processes=n_c) as pool:
                Hjs = pool.starmap(self.jcolhessian, args)

        for ji, j in enumerate(jcols):

            if HAS_MP and n_c > 1:
                # Simply extract
                Hj = Hjs[ji].flatten()
            else:

                # Entire column j of negative hessian
                Hj = self.jcolhessian(j, coef, coef_split_idx, ys, Xs).flatten()

            # Take out elements previously computed
            Hjrows = np.arange(Hdim)
            Hjc = np.delete(Hj, ccols)
            Hjrows = np.delete(Hjrows, ccols)

            Hdat.extend(Hjc)
            Hrows.extend(Hjrows)
            Hcols.extend(np.tile(j, len(Hjrows)))

            # Keep symmetric - remove element on diagonal
            ccols.append(j)

            Hjcols = np.arange(Hdim)
            Hjr = np.delete(Hj, ccols)
            Hjcols = np.delete(Hjcols, ccols)

            Hdat.extend(Hjr)
            Hcols.extend(Hjcols)
            Hrows.extend(np.tile(j, len(Hjcols)))

        # Build sparse hessian approximation
        Ha = scp.sparse.csc_array((Hdat, (Hrows, Hcols)), shape=(Hdim, Hdim))

        return Ha

    def hessian(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray | None],
        Xs: list[scp.sparse.csc_array | None],
    ) -> scp.sparse.csc_array:
        """Function to evaluate the hessian of the llk at current coefficient estimate ``coef``.

        Only has to be implemented if full Newton is to be used to estimate coefficients. If the
        L-qEFS update by Krause et al. (in preparation) is to be used instead, this method does not
        have to be implemented.

        References:
           - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.
           - ``scipy.optimize.approx_fprime``: at \
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.approx_fprime.html
           - Krause et al. (submitted). The Mixed-Sparse-Smooth-Model Toolbox (MSSM): Efficient \
            Estimation and Selection of Large Multi-Level Statistical Models. \
            https://doi.org/10.48550/arXiv.2506.13132

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each paramter of the llk.
        :type coef_split_idx: [int]
        :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as
            ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the
            actual observed data is passed along via the first formula (so it is stored in
            ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula,
            then ``ys`` contains ``None`` at their indices to save memory.
        :type ys: list[np.ndarray | None]
        :param Xs: A list of sparse model matrices per likelihood parameter. Might contain ``None``
            at indices for matrices which were flagged as "do not build" via the ``build_mat``
            argument of the :func:`mssm.models.GSMM.fit` method.
        :type Xs: list[scp.sparse.csc_array | None]
        :return: The Hessian of the log-likelihood evaluated at ``coef``.
        :rtype: scp.sparse.csc_array
        """
        return self.jhessian(np.arange(len(coef)), coef, coef_split_idx, ys, Xs)

    @abstractmethod
    def get_resid(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray | None],
        Xs: list[scp.sparse.csc_array | None],
    ) -> np.ndarray | None:
        """Get standardized residuals for a GSMM model.

        Any implementation of this function should return a vector that looks like what could be
        expected from taking independent draws from :math:`N(0,1)`. Any additional arguments
        required by a specific implementation need to be passed along via additional
        keyword arguments with default values.

        **Note**: Families for which no residuals are available can return None.

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.
         - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must
            not be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each paramter of the llk.
        :type coef_split_idx: [int]
        :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as
            ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the
            actual observed data is passed along via the first formula (so it is stored in
            ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula,
            then ``ys`` contains ``None`` at their indices to save memory.
        :type ys: list[np.ndarray | None]
        :param Xs: A list of sparse model matrices per likelihood parameter. Might contain ``None``
            at indices for matrices which were flagged as "do not build" via the ``build_mat``
            argument of the :func:`mssm.models.GSMM.fit` method.
        :type Xs: list[scp.sparse.csc_array | None]
        :return: a vector of shape (-1,1) containing standardized residuals under the current model
            (**Note**, the first axis will not necessarily match the dimension of any of the
            response vectors (this will depend on the specific Family's implementation)) or None in
            case residuals are not readily available.
        :rtype: np.ndarray | None
        """
        pass

    def init_coef(self, models: list[Callable]) -> np.ndarray:
        """(Optional) Function to initialize the coefficients of the model.

        Can return ``None`` , in which case random initialization will be used.

        :param models: A list of :class:`mssm.models.GAMM`'s, - each based on one of the formulas
            provided to a model.
        :type models: [mssm.models.GAMM]
        :return: A numpy array of shape (-1,1), holding initial values for all model coefficients.
        :rtype: np.ndarray
        """
        return None

    def init_lambda(self, penalties: list[Callable]) -> list[float]:
        """(Optional) Function to initialize the smoothing parameters of the model.

        Can return ``None`` , in which case random initialization will be used.

        :param penalties: A list of all penalties to be estimated by the model.
        :type penalties: [mssm.src.python.penalties.LambdaTerm]
        :return: A list, holding - for each :math:`\\lambda` parameter to be estimated - an initial
            value.
        :rtype: np.ndarray
        """
        return None


class PropHaz(GSMMFamily):
    """Family for proportional Hazard model - a type of General Smooth model as discussed by
    Wood, Pya, & Säfken (2016).

    Based on Supplementary materials G in Wood, Pya, & Säfken (2016). The dependent variable
    passed to the :class:`mssm.src.python.formula.Formula` needs to hold ``delta`` indicating
    whether the event was observed or not (i.e., only values in ``{0,1}``).

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

       # Can plot the estimated effects on the scale of the
       # linear predictor (i.e., log hazard) via mssmViz
       plot(model)

    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for \
        General Smooth Models.
     - Nocedal & Wright (2006). Numerical Optimization. Springer New York.

    :param ut: Unique event time vector (each time represnted as ``int``) as described by
        WPS (2016), holding unique event times in decreasing order.
    :type ut: np.ndarray
    :param r: Index vector as described by WPS (2016), holding for each data-point
        (i.e., for each row in ``Xs[0``]) the index to it's corresponding event time in ``ut``.
    :type r: np.ndarray
    :ivar np.ndarray ut: Array passed for ``ut``.
    :ivar np.ndarray r: Array passed for ``r``.

    """

    def __init__(self, ut: np.ndarray, r: np.ndarray):
        super().__init__(1, [Identity()])
        self.ut = ut
        self.r = r
        self.__hs = None
        self.__qs = None
        self.__avs = None

    def llk(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray],
        Xs: list[scp.sparse.csc_array],
    ) -> float:
        """Log-likelihood function as defined by Wood, Pya, & Säfken (2016).

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each paramter of the llk - not required by this family,
            which has a single parameter.
        :type coef_split_idx: [int]
        :param ys: List containing the ``delta`` vector at the first and only index - see
            description of the model family.
        :type ys: [np.ndarray]
        :param Xs: A list containing the sparse model matrix at the first and only index.
        :type Xs: [scp.sparse.csc_array]
        :return: The log-likelihood evaluated at ``coef``.
        :rtype: float
        """

        # Extract and define all variables defined by WPS (2016)
        delta = ys[0]
        ut = self.ut
        r = self.r
        nt = len(ut)
        X = Xs[0]
        eta = X @ coef

        with warnings.catch_warnings():  # Overflow
            warnings.simplefilter("ignore")
            gamma = np.exp(eta)

        # Now compute first sum
        llk = np.sum(delta * eta)

        # and second sum
        gamma_p = 0
        for j in range(nt):
            ri = r == j
            dj = np.sum(delta[ri])
            with warnings.catch_warnings():  # Overflow
                warnings.simplefilter("ignore")
                gamma_p += np.sum(gamma[ri])

            with warnings.catch_warnings():  # Divide by zero
                warnings.simplefilter("ignore")
                log_gamma_p = np.log(gamma_p)

            llk -= dj * log_gamma_p

        return llk

    def gradient(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray],
        Xs: list[scp.sparse.csc_array],
    ) -> np.ndarray:
        """Gradient as defined by Wood, Pya, & Säfken (2016).

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each paramter of the llk - not required by this family,
            which has a single parameter.
        :type coef_split_idx: [int]
        :param ys: List containing the ``delta`` vector at the first and only index - see
            description of the model family.
        :type ys: [np.ndarray]
        :param Xs: A list containing the sparse model matrix at the first and only index.
        :type Xs: [scp.sparse.csc_array]
        :return: The Gradient of the log-likelihood evaluated at ``coef`` as numpy array of
            shape (-1,1).
        :rtype: np.ndarray
        """

        # Extract and define all variables defined by WPS (2016)
        delta = ys[0]
        ut = self.ut
        r = self.r
        nt = len(ut)
        X = Xs[0]
        eta = X @ coef

        with warnings.catch_warnings():  # Overflow
            warnings.simplefilter("ignore")
            gamma = np.exp(eta)
        gamma = gamma.reshape(-1, 1)

        # Now compute first sum
        g = delta.T @ X

        # and second sum
        b_p = np.zeros_like(g)

        gamma_p = 0
        for j in range(nt):
            ri = r == j
            dj = np.sum(delta[ri])
            gamma_i = (gamma[ri, 0]).reshape(-1, 1)
            with warnings.catch_warnings():  # Overflow
                warnings.simplefilter("ignore")
                gamma_p += np.sum(gamma_i)

            X_i = X[ri, :]
            bi = gamma_i.T @ X_i
            b_p += bi

            with warnings.catch_warnings():  # Divide by zero
                warnings.simplefilter("ignore")
                bpg = b_p / gamma_p

            g -= dj * bpg

        return g.reshape(-1, 1)

    def hessian(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray],
        Xs: list[scp.sparse.csc_array],
    ) -> scp.sparse.csc_array:
        """Hessian as defined by Wood, Pya, & Säfken (2016).

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each paramter of the llk - not required by this family,
            which has a single parameter.
        :type coef_split_idx: [int]
        :param ys: List containing the ``delta`` vector at the first and only index - see
            description of the model family.
        :type ys: [np.ndarray]
        :param Xs: A list containing the sparse model matrix at the first and only index.
        :type Xs: [scp.sparse.csc_array]
        :return: The Hessian of the log-likelihood evaluated at ``coef``.
        :rtype: scp.sparse.csc_array
        """

        # Extract and define all variables defined by WPS (2016)
        delta = ys[0]
        ut = self.ut
        r = self.r
        nt = len(ut)
        X = Xs[0]
        eta = X @ coef

        with warnings.catch_warnings():  # Overflow
            warnings.simplefilter("ignore")
            gamma = np.exp(eta)
        gamma = gamma.reshape(-1, 1)

        # Only sum over nt
        b_p = np.zeros((1, X.shape[1]))

        gamma_p = 0
        A_p = scp.sparse.csc_array((X.shape[1], X.shape[1]))
        H = scp.sparse.csc_array((X.shape[1], X.shape[1]))
        for j in range(nt):
            ri = r == j
            dj = np.sum(delta[ri])
            gamma_i = (gamma[ri, 0]).reshape(-1, 1)
            gamma_p += np.sum(gamma_i)

            X_i = X[ri, :]
            bi = gamma_i.T @ X_i
            b_p += bi

            A_i = (gamma_i * X_i).T @ X_i
            A_p += A_i

            with warnings.catch_warnings():  # Divide by zero or overflow
                warnings.simplefilter("ignore")
                Hj = dj * b_p.T @ b_p / np.power(gamma_p, 2) - dj * A_p / gamma_p

            H += Hj

        return scp.sparse.csc_array(H)

    def get_resid(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray],
        Xs: list[scp.sparse.csc_array],
        resid_type: str = "Martingale",
        reorder: np.ndarray | None = None,
    ) -> np.ndarray:
        """Get Martingale or Deviance residuals for a proportional Hazard model.

        See the :func:`PropHaz.get_survival` function for examples.

        References:
           - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.
           - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second \
            Edition (2nd ed.).

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each paramter of the llk - not required by this family,
            which has a single parameter.
        :type coef_split_idx: [int]
        :param ys: List containing the ``delta`` vector at the first and only index - see
            description of the model family.
        :type ys: [np.ndarray]
        :param Xs: A list containing the sparse model matrix at the first and only index.
        :type Xs: [scp.sparse.csc_array]
        :param resid_type: The type of residual to compute, supported are "Martingale" and
            "Deviance".
        :type resid_type: str, optional
        :param reorder: A flattened np.ndarray containing for each data point the original index in
            the data-set before sorting. Used to re-order the residual vector into the original
            order. If this is set to None, the residual vector is not re-ordered and instead
            returned in the order of the sorted data-frame passed to the model formula.
        :type reorder: np.ndarray
        :return: The residual vector of shape (-1,1)
        :rtype: np.ndarray
        """

        if resid_type not in ["Martingale", "Deviance"]:
            raise ValueError("`resid_type` must be one of 'Martingale' or 'Deviance'.")

        # Extract all quantities needed to evaluate residuals
        delta = ys[0]
        ut = self.ut
        r = self.r
        X = Xs[0]

        # Following based on derivation by Wood, Pya, and Säfken (2016)
        res = np.zeros(X.shape[0])
        for idx, tidx in enumerate(r):
            Xi = X[idx, :].toarray()
            ti = ut[tidx]
            di = delta[idx]
            Si, _ = self.get_survival(coef, Xs, delta, ti, Xi, None, compute_var=False)
            mi = di + np.log(Si[0])

            if resid_type == "Martingale":
                res[idx] = mi[0]
            else:
                # Deviance requires a bit more work
                Di = np.sign(mi) * np.power(
                    -2 * (mi + di * np.log(-min(np.log(Si[0]), -np.finfo(float).eps))),
                    0.5,
                )
                res[idx] = Di[0]

        # Return to order of original dataframe
        if reorder is not None:
            res = res[reorder]

        return res.reshape(-1, 1)

    def __prepare_predictions(
        self, coef: np.ndarray, delta: np.ndarray, Xs: list[scp.sparse.csc_array]
    ) -> None:
        """Computes all the quantities defined by Wood, Pya, & Säfken (2016) that are necessary for
        predictions.

        This includes the cumulative base-line hazard, as well as the :math`\\mathbf{a}` vectors
        from WPS (2016). These are assigned to the instance of this family.

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.

        :param coef: Coefficient vector as numpy array of shape (-1,1).
        :type coef: np.ndarray
        :param delta: Dependent variable passed to :func:`mssm.src.python.formula.Formula`, holds
            (for each row in ``Xs[0``]) a value in ``{0,1}``, indicating whether for that
            observation the event was observed or not.
        :type delta: np.ndarray
        :param Xs: The list of model matrices (here holding a single model matrix) obtained from
            :func:`mssm.models.GAMMLSS.get_mmat`.
        :type Xs: [scp.sparse.csc_array]
        """
        # Extract and define all variables defined by WPS (2016)
        ut = self.ut
        r = self.r
        nt = len(ut)
        X = Xs[0]
        eta = X @ coef

        with warnings.catch_warnings():  # Overflow
            warnings.simplefilter("ignore")
            gamma = np.exp(eta)

        gamma[np.isnan(gamma) | np.isinf(gamma)] = np.power(np.finfo(float).max, 0.9)
        gamma = gamma.reshape(-1, 1)

        # We need gamma_ps, b_ps, and djs
        gamma_ps = []
        djs = []
        b_ps = []

        gamma_p = 0
        b_p = np.zeros((1, X.shape[1]))
        for j in range(nt):
            ri = r == j

            # Get dj
            dj = np.sum(delta[ri])
            djs.append(dj)

            # gamma_p
            gamma_i = (gamma[ri, 0]).reshape(-1, 1)
            with warnings.catch_warnings():  # Overflow
                warnings.simplefilter("ignore")
                gamma_p += np.sum(gamma_i)

            if np.isnan(gamma_p) | np.isinf(gamma_p):
                gamma_p = np.power(np.finfo(float).max, 0.9)
            gamma_ps.append(gamma_p)

            # b_p vector
            X_i = X[ri, :]
            bi = gamma_i.T @ X_i
            b_p += bi
            b_ps.append(copy.deepcopy(b_p))

        # Now base-line hazard + variance and a vectors
        hs = np.zeros(nt)
        qs = np.zeros(nt)
        avs = [np.zeros_like(b_p) for _ in range(nt)]

        hs[-1] = djs[-1] / gamma_ps[-1]
        qs[-1] = djs[-1] / np.power(gamma_ps[-1], 2)
        avs[-1] = b_ps[-1] * djs[-1] / np.power(gamma_ps[-1], 2)
        # print(hs[-1],qs[-1])

        for j in range(nt - 2, -1, -1):
            # print(j,hs[j+1])
            hs[j] = hs[j + 1] + djs[j] / gamma_ps[j]
            qs[j] = qs[j + 1] + djs[j] / np.power(gamma_ps[j], 2)
            avs[j] = avs[j + 1] + b_ps[j] * djs[j] / np.power(gamma_ps[j], 2)

        self.__hs = hs
        self.__qs = qs
        self.__avs = avs

    def get_baseline_hazard(
        self, coef: np.ndarray, delta: np.ndarray, Xs: list[scp.sparse.csc_array]
    ) -> np.ndarray:
        """Get the cumulative baseline hazard function as defined by Wood, Pya, & Säfken (2016).

        The function is evaluated for all ``k`` unique event times that were available in the data.

        Examples::

          from mssm.models import *
          from mssmViz.sim import *
          from mssmViz.plot import *
          import matplotlib.pyplot as plt

          # Simulate some data
          sim_dat = sim3(500,2,c=1,seed=0,family=PropHaz([0],[0]),binom_offset = 0.1,
            correlate=False)

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
          H = PropHaz_fam.get_baseline_hazard(model.coef,
            sim_formula_m.y_flat[sim_formula_m.NOT_NA_flat],model.get_mmat())

          # And plot it
          plt.plot(ut,H)
          plt.xlabel("Time")
          plt.ylabel("Cumulative Baseline Hazard")

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.

        :param coef: Coefficient vector as numpy array of shape (-1,1).
        :type coef: np.ndarray
        :param Xs: The list of model matrices (here holding a single model matrix) obtained from
            :func:`mssm.models.GAMMLSS.get_mmat`.
        :type Xs: [scp.sparse.csc_array]
        :param delta: Dependent variable passed to :func:`mssm.src.python.formula.Formula`, holds
            (for each row in ``Xs[0``]) a value in ``{0,1}``, indicating whether for that
            observation the event was observed or not.
        :type delta: np.ndarray
        :return: numpy array, holding ``k`` baseline hazard function estimates
        :rtype: np.ndarray
        """

        if self.__hs is None:
            self.__prepare_predictions(coef, delta, Xs)

        return self.__hs

    def get_survival(
        self,
        coef: np.ndarray,
        Xs: list[scp.sparse.csc_array],
        delta: np.ndarray,
        t: int,
        x: np.ndarray | scp.sparse.csc_array,
        V: scp.sparse.csc_array,
        compute_var: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Compute survival function + variance at time-point ``t``, given ``k`` optional covariate
        vector(s) x as defined by Wood, Pya, & Säfken (2016).

        Examples::

          from mssm.models import *
          from mssmViz.sim import *
          from mssmViz.plot import *
          import matplotlib.pyplot as plt

          # Simulate some data
          sim_dat = sim3(500,2,c=1,seed=0,family=PropHaz([0],[0]),binom_offset = 0.1,
            correlate=False)

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

          # Now iterate over all time-points and obtain the predicted survival
          # function + standard error estimate
          # for all 5 values of x0:
          S = np.zeros((len(ut),Xt.shape[0]))
          VS = np.zeros((len(ut),Xt.shape[0]))
          for idx,ti in enumerate(ut):

             # Su and VSu are of shape (5,1) here but will generally be of shape (Xt.shape[0],1)
             Su,VSu = PropHaz_fam.get_survival(model.coef,model.get_mmat(),
                sim_formula_m.y_flat[sim_formula_m.NOT_NA_flat],
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

          # Residual plots can be created via `plot_val` from `mssmViz` - by default Martingale
          # residuals are returned (see Wood, 2017)
          fig = plt.figure(figsize=(10,3),layout='constrained')
          axs = fig.subplots(1,3,gridspec_kw={"wspace":0.2})
          # Note the use of `gsmm_kwargs_pred={}` to ensure that the re-ordering is not applied
          # to the plot against predicted values
          plot_val(model,gsmm_kwargs={"reorder":res_idx},gsmm_kwargs_pred={},ar_lag=25,axs=axs)

          # Can also get Deviance residuals:
          fig = plt.figure(figsize=(10,3),layout='constrained')
          axs = fig.subplots(1,3,gridspec_kw={"wspace":0.2})

          plot_val(model,
            gsmm_kwargs={"reorder":res_idx,"resid_type":"Deviance"},
            gsmm_kwargs_pred={"resid_type":"Deviance"},ar_lag=25,axs=axs)

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General \
            Smooth Models.

        :param coef: Coefficient vector as numpy array of shape (-1,1).
        :type coef: np.ndarray
        :param Xs: The list of model matrices (here holding a single model matrix) obtained from
            :func:`mssm.models.GAMMLSS.get_mmat`.
        :type Xs: [scp.sparse.csc_array]
        :param delta: Dependent variable passed to :func:`mssm.src.python.formula.Formula`, holds
            (for each row in ``Xs[0``]) a value in ``{0,1}``, indicating whether for that
            observation the event was observed or not.
        :type delta: np.ndarray
        :param t: Time-point at which to evaluate the survival function.
        :type t: int
        :param x: Optional vector (or matrix - can also be sparse) of covariate values. Needs to be
            of shape ``(k,len(coef))``.
        :type x: np.ndarray or scp.sparse.csc_array
        :param V: Estimated Co-variance matrix of posterior for ``coef``
        :type V: scp.sparse.csc_array
        :param compute_var: Whether to compue the variance estimate of the survival as well.
            Otherwise None will be returned as the second argument.
        :type compute_var: bool, optional
        :return: Two arrays, the first holds ``k`` survival function estimates, the latter holds
            ``k`` variance estimates for each of the survival function estimates. The second
            argument will be None instead if ``compute_var = False``.
        :rtype: tuple[np.ndarray, np.ndarray | None]
        """

        if self.__hs is None:
            self.__prepare_predictions(coef, delta, Xs)

        # Extract and define all variables defined by WPS (2016)
        ut = self.ut
        eta = x @ coef
        # print(eta)

        # Find nearest larger time-point
        if t not in ut:
            t = min(ut[ut > t])

        # Find index in h corresponding to t
        ti = ut == t
        tiv = np.arange(len(ut))[ti][0]
        # print(t,tiv)

        # Compute (log) survival
        lS = -self.__hs[ti] * np.exp(eta)
        S = np.exp(lS)

        varS = None
        if compute_var:
            # Compute variance
            v = -self.__hs[ti] * x + self.__avs[tiv]

            varS = (
                np.exp(eta)
                * S
                * np.power(
                    self.__qs[ti] + np.sum(v @ V * v, axis=1).reshape(-1, 1), 0.5
                )
            )
        return S, varS

    def init_coef(self, models: list[Callable]) -> np.ndarray:
        """Function to initialize the coefficients of the model.

        :param models: A list of GAMMs, - each based on one of the formulas provided to a model.
        :type models: [mssm.models.GAMM]
        :return: A numpy array of shape (-1,1), holding initial values for all model coefficients.
        :rtype: np.ndarray
        """

        # Just set to very small positive values
        coef = np.array([1e-4 for _ in range(models[0].formulas[0].n_coef)]).reshape(
            -1, 1
        )
        return coef


class MultiGauss(GSMMFamily):
    """Family for multivariate additive models - a type of General Smooth model as discussed by
    Wood, Pya, & Säfken (2016).

    Implementation based on Supplementary materials H in Wood, Pya, & Säfken (2016). Currently,
    these models can only be estimated via the ``L-qEFS`` update in ``mssm``.

    Examples::

        from mssm.models import *
        from mssmViz.sim import *
        from mssmViz.plot import *
        import matplotlib.pyplot as plt

        # Simulate data
        sim_dat = sim16(500,seed=1134,correlate=True)

        # We need formulas for each mean!
        formulas = [
            Formula(lhs("y0"), [i(), f(["x0"])], data=sim_dat),
            Formula(lhs("y1"), [i(), f(["x1"]), f(["x2"])], data=sim_dat),
            Formula(lhs("y2"), [i(), f(["x3"])], data=sim_dat)
        ]

        # Now define the model...
        model = GSMM(formulas, MultiGauss(3,[Identity() for _ in range(3)]))

        # ... and fit!
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(method='qEFS')

        # Get overview:
        model.print_parametric_terms()
        model.print_smooth_terms(p_values=True)

        # And plot smooth function estimates for mean at index 1
        plot(model,dist_par=1)

    References:
     - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for \
        General Smooth Models.
     - Nocedal & Wright (2006). Numerical Optimization. Springer New York.

    :param pars: Number of means (i.e., dimension of the multivariate Gaussian)
    :type pars: int
    :param links: List of link functions for the models of the means. For example
        ``[Identity() for _ in range(pars)]``.
    :type links: list[Link]
    :ivar int n_par: Value passed for ``pars``.
    :ivar list[Link] links: List passed for ``links``.
    :ivar int extra_coef: Number of extra coef. required by this family.
    """

    def __init__(self, pars: int, links: list[Link]):
        super().__init__(pars, links)

        # Elements of cholesky of precision matrix of multivariate Gaussian
        self.extra_coef = int(pars * (pars + 1) / 2)

        self.__thet_idx: list[tuple[int, int]] = []  # Cell indices in R
        for i in range(self.n_par):
            for j in range(i, self.n_par):
                self.__thet_idx.append([i, j])

    def getR(self, theta: np.ndarray) -> tuple[np.ndarray, float]:
        """Returns transpose of Cholesky of precision matrix of multivariate Gaussian.

        Examples::

            from mssm.models import *
            from mssmViz.sim import *
            from mssmViz.plot import *
            import matplotlib.pyplot as plt

            # Simulate data
            sim_dat = sim16(500,seed=1134,correlate=True)

            # We need formulas for each mean!
            formulas = [
                Formula(lhs("y0"), [i(), f(["x0"])], data=sim_dat),
                Formula(lhs("y1"), [i(), f(["x1"]), f(["x2"])], data=sim_dat),
                Formula(lhs("y2"), [i(), f(["x3"])], data=sim_dat)
            ]

            # Now define the model...
            model = GSMM(formulas, MultiGauss(3,[Identity() for _ in range(3)]))

            # ... and fit!
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(method='qEFS')

            # Extract R
            split_coef = np.split(model.coef,model.coef_split_idx)
            theta = split_coef[-1].flatten()
            R,log_det = model.family.getR(theta)

            # R is the transpose of the Cholesky of the precision matrix. So to get the
            # Covariance matrix of the multivariate Gaussian we need to compute:
            Sigma = np.linalg.inv(R.T@R)

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for \
            General Smooth Models.

        :param theta: Flattened array holding inverses of log(variance) and co-variance parameters
        :type theta: np.ndarray
        :return: Transpose of Cholesky as a numpy array and log-determinant of Cholesky
        :rtype: tuple[np.ndarray,float]
        """

        R = np.zeros((self.n_par, self.n_par))

        dat_idx = 0
        logdet = 0
        for m in range(self.n_par):

            R[m, m:] = theta[dat_idx : dat_idx + (self.n_par - m)]  # noqa: E203
            logdet += R[m, m]
            R[m, m] = np.exp(R[m, m])

            dat_idx += self.n_par - m

        return R, logdet

    def llk(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray],
        Xs: list[scp.sparse.csc_array | None],
    ) -> float:
        """Computes the log-likelihood under a multivariate normal given coefficients in the
        additive models of the mean and the log(variance) and co-variance parameters.

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for \
            General Smooth Models.

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!). **Note** the last ``int(pars * (pars + 1) / 2)`` elements contain the
            log(variance) and co-variance parameters.
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each mean and a final sub-set containing all log(variance) and
            co-variance parameters.
        :type coef_split_idx: [int]
        :param ys: List containing the vectors of observations.
        :type ys: [np.ndarray]
        :param Xs: A list containing the sparse model matrices associated with the models of the
            means. **Note**, this implementation allows to make use of the ``build_mat`` argument
            of the :func:`GSMM.fit` method. Specifically, for means that have the same predictor
            structure as the first mean we can set ``build_mat[idx] = False``. The code then
            automatically assigns ``X[idx] = X[0]``. See the :func:`GSMM.fit` documentation for
            more details.
        :type Xs: [scp.sparse.csc_array | None]
        """

        # Extract extra info and fix model matrices, then compute mu and theta
        y = np.concatenate(ys, axis=1)
        mus, theta = self.predict(coef, coef_split_idx, Xs)

        # Get transpose of Cholesky of precision
        R, logdet = self.getR(theta)

        # yR is of shape n * m
        yR = (y - mus) @ R.T

        # Compute np.sum([yR[oi, :] @ yR.T[:, oi] for oi in range(Xfix[0].shape[0])]) as shown
        # by WPS (2016)
        llk = -0.5 * (yR * yR).sum() + yR.shape[0] * logdet

        return llk

    def gradient(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray],
        Xs: list[scp.sparse.csc_array | None],
    ) -> np.ndarray:
        """Computes gradient of a multivariate normal containing partial derivatives with
        respect to coefficients in the additive models of the mean and the log(variance) and
        co-variance parameters.

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for \
            General Smooth Models.

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!). **Note** the last ``int(pars * (pars + 1) / 2)`` elements contain the
            log(variance) and co-variance parameters.
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each mean and a final sub-set containing all log(variance) and
            co-variance parameters.
        :type coef_split_idx: [int]
        :param ys: List containing the vectors of observations.
        :type ys: [np.ndarray]
        :param Xs: A list containing the sparse model matrices associated with the models of the
            means. **Note**, this implementation allows to make use of the ``build_mat`` argument
            of the :func:`GSMM.fit` method. Specifically, for means that have the same predictor
            structure as the first mean we can set ``build_mat[idx] = False``. The code then
            automatically assigns ``X[idx] = X[0]``. See the :func:`GSMM.fit` documentation for
            more details.
        :type Xs: [scp.sparse.csc_array | None]
        :return: The gradient at the current parameters estimates in a numpy array of shape (-1,1)
        :rtype: np.ndarray
        """

        # Extract extra info and fix model matrices, then compute mu and theta
        Xfix = [X if X is not None else Xs[0] for X in Xs]
        y = np.concatenate(ys, axis=1)
        split_coef = np.split(coef, coef_split_idx)
        mus = np.concatenate(
            [
                self.links[mui].fi(Xfix[mui] @ split_coef[mui].reshape(-1, 1))
                for mui in range(self.n_par)
            ],
            axis=1,
        )
        theta = split_coef[-1].flatten()

        # Get transpose of Cholesky of precision
        R, logdet = self.getR(theta)
        Rdiag = R.diagonal()

        # yR is of shape n * m
        yR = (y - mus) @ R.T
        # RRy is of shape m * n
        RRy = R.T @ yR.T

        total_idx = 0
        n_obs = Xfix[0].shape[0]
        grad = np.zeros(coef.shape[0])

        # Compute np.sum([xbarR[oi, :] @ yR.T[:, oi] for oi in range(n_obs)]) as
        # shown by WPS (2016)
        for mui in range(self.n_par):

            yRmui = RRy[mui, :]

            grad[total_idx : total_idx + Xfix[mui].shape[1]] = (  # noqa: E203
                Xfix[mui] * yRmui[:, None]
            ).sum(axis=0)
            total_idx += Xfix[mui].shape[1]

        # Now partials for variance parameters
        Rp = np.zeros((self.n_par, self.n_par))  # partial of transpose of cholesky
        for mr in range(self.n_par):

            for mc in range(mr, self.n_par):

                # Diagonal elements are exp(theta) so partial differs
                Rp[mr, mc] = Rdiag[mr] if mr == mc else 1

                # Compute np.sum([yR[oi, :] @ Rpy[:, oi] for oi in range(n_obs)]) as shown
                # by Wps (2016)
                yRp = (y - mus) @ Rp.T
                dldtheta = -np.sum(yR * yRp)

                # Index function part from WPS (2016)
                if mr == mc:
                    dldtheta += n_obs

                grad[total_idx] = dldtheta

                # Reset partial of R
                Rp[mr, mc] = 0
                total_idx += 1

        return grad.reshape(-1, 1)

    def jcolhessian(self, j, coef, coef_split_idx, ys, Xs) -> np.ndarray:
        """Method to compute column ``j`` of the Hessian of the log-likelihood.

        Result is returned as a np.array.

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for \
            General Smooth Models.

        :param j: Index of the column to approximate.
        :type jcols: int
        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!).
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each paramter of the llk.
        :type coef_split_idx: [int]
        :param ys: List containing the vectors of observations (each of shape (-1,1)) passed as
            ``lhs.variable`` to the formulas. **Note**: by convention ``mssm`` expectes that the
            actual observed data is passed along via the first formula (so it is stored in
            ``ys[0]``). If multiple formulas have the same ``lhs.variable`` as this first formula,
            then ``ys`` contains ``None`` at their indices to save memory.
        :type ys: list[np.ndarray | None]
        :param Xs: A list of sparse model matrices per likelihood parameter. Might contain ``None``
            at indices for matrices which were flagged as "do not build" via the ``build_mat``
            argument of the :func:`mssm.models.GSMM.fit` method.
        :type Xs: list[scp.sparse.csc_array | None]
        :return: Column ``j`` of the Hessian of the llk as an array of shape (-1,1)
        :rtype: np.ndarray
        """

        # Extract extra info and fix model matrices, then compute mu and theta
        Xfix = [X if X is not None else Xs[0] for X in Xs]
        y = np.concatenate(ys, axis=1)
        split_coef = np.split(coef, coef_split_idx)
        mus = np.concatenate(
            [
                self.links[mui].fi(Xfix[mui] @ split_coef[mui].reshape(-1, 1))
                for mui in range(self.n_par)
            ],
            axis=1,
        )
        theta = split_coef[-1].flatten()

        # Need some info about number of coefs and indices
        n_coef = len(coef) - len(theta)
        n_obs = mus.shape[0]
        par_n_coef = []

        checked_idx = 0
        for mui in range(self.n_par):
            checked_idx += len(split_coef[mui])
            par_n_coef.append(checked_idx)

        # Get transpose of Cholesky of precision
        R, logdet = self.getR(theta)
        Rdiag = R.diagonal()

        # ymu is of shape (n * m)
        ymu = y - mus
        # RR is of shape m * m
        RR = R.T @ R

        # Compute column j of the hessian
        Hj = []

        # First figure out to which parameter j belongs
        is_coef = j < n_coef

        if is_coef:
            parj_idx = 0
            for mui in range(self.n_par):
                if j < par_n_coef[mui]:
                    break
                parj_idx += 1

            coef_j_idx = j if parj_idx == 0 else j - par_n_coef[parj_idx - 1]
            xbarj = scp.sparse.csc_array(
                (
                    Xfix[parj_idx][:, coef_j_idx].toarray(),
                    (np.tile(parj_idx, n_obs), np.arange(n_obs)),
                ),
                shape=(self.n_par, n_obs),
            )
            # xbarj[parj_idx, :] = Xfix[parj_idx][:, coef_j_idx]
            RRxbarj = RR @ xbarj
        else:
            parj_idx = j - n_coef  # only imagine + self.n_par
            theta_idx = self.__thet_idx[parj_idx]
            is_diag = theta_idx[0] == theta_idx[1]
            # partial of transpose of cholesky
            Rpj = np.zeros((self.n_par, self.n_par))
            Rpj[theta_idx[0], theta_idx[1]] = Rdiag[theta_idx[0]] if is_diag else 1
            RJRpRRJ = Rpj.T @ R + R.T @ Rpj
            RJRpRRJy = RJRpRRJ @ ymu.T

        # Now iterate over means
        for mui in range(self.n_par):
            if is_coef:
                RRxbarjmui = RRxbarj[mui, :]
                d2 = -np.sum(Xfix[mui] * RRxbarjmui[:, None], axis=0)
            else:
                RJRpRRJymui = RJRpRRJy[mui, :]
                d2 = (Xfix[mui] * RJRpRRJymui[:, None]).sum(axis=0)
            Hj.extend(d2)

        # And theta:
        Rpi = np.zeros((self.n_par, self.n_par))
        for mr in range(self.n_par):
            for mc in range(mr, self.n_par):

                Rpi[mr, mc] = Rdiag[mr] if mr == mc else 1

                if is_coef:
                    RiRpRRi = Rpi.T @ R + R.T @ Rpi
                    RiRpRRiy = RiRpRRi @ ymu.T
                    RiRpRRiymui = RiRpRRiy[parj_idx, :]
                    xbarjmui = xbarj[parj_idx, :]
                    d2 = np.sum(xbarjmui * RiRpRRiymui)
                else:
                    d2 = (ymu @ Rpj.T) * (ymu @ Rpi.T)
                    if (mr == theta_idx[0]) and (mc == theta_idx[1]):
                        d2 += (ymu @ R.T) * (ymu @ Rpj.T)
                    d2 = -np.sum(d2)

                Rpi[mr, mc] = 0
                Hj.append(d2)

        return np.array(Hj).reshape(-1, 1)

    def get_resid(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        ys: list[np.ndarray],
        Xs: list[scp.sparse.csc_array | None],
        mean: int | None = None,
    ) -> np.ndarray:
        """Computes Deviance residuals of a multivariate normal model given coefficients in the
        additive models of the mean and the log(variance) and co-variance parameters.

        If the model is correct, each column in the returned matrix should look like an i.i.d sample
        of size ``N`` from :math:`N(0,1)`.

        Examples::

            from mssm.models import *
            from mssmViz.sim import *
            from mssmViz.plot import *
            import matplotlib.pyplot as plt

            # Simulate data
            sim_dat = sim16(500,seed=1134,correlate=True)

            # We need formulas for each mean!
            formulas = [
                Formula(lhs("y0"), [i(), f(["x0"])], data=sim_dat),
                Formula(lhs("y1"), [i(), f(["x1"]), f(["x2"])], data=sim_dat),
                Formula(lhs("y2"), [i(), f(["x3"])], data=sim_dat)
            ]

            # Now define the model...
            model = GSMM(formulas, MultiGauss(3,[Identity() for _ in range(3)]))

            # ... and fit!
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(method='qEFS')

            # Can now extract the residual matrix
            res = model.get_resid()

            # The ``get_resid`` method supports a ``mean`` key-word to extract univariate residuals
            # for an individual mean. We can use mssmViz's plot function to visualize these
            # for example for mean at index 2:
            plot_val(model,gsmm_kwargs={"mean":2})

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for \
            General Smooth Models.

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!). **Note** the last ``int(pars * (pars + 1) / 2)`` elements contain the
            log(variance) and co-variance parameters.
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each mean and a final sub-set containing all log(variance) and
            co-variance parameters.
        :type coef_split_idx: [int]
        :param ys: List containing the vectors of observations.
        :type ys: [np.ndarray]
        :param Xs: A list containing the sparse model matrices associated with the models of the
            means. **Note**, this implementation allows to make use of the ``build_mat`` argument
            of the :func:`GSMM.fit` method. Specifically, for means that have the same predictor
            structure as the first mean we can set ``build_mat[idx] = False``. The code then
            automatically assigns ``X[idx] = X[0]``. See the :func:`GSMM.fit` documentation for
            more details.
        :type Xs: [scp.sparse.csc_array | None]
        :param mean: Optionally, the index of a specific mean for which to extract the residuals.
            This allows to extract univariate residuals for a specific mean. Setting this to
            ``None`` means the ``(N * self.n_par)`` residual matrix is returned where ``N`` is the
            number of observations. Defaults to None
        :type mean: int | None, optional
        :return: Residual matrix. Will be a residual vector if ``mean`` is not set to None.
        :rtype: np.ndarray
        """
        # Extract extra info and fix model matrices, then compute mu and theta
        y = np.concatenate(ys, axis=1)
        mus, theta = self.predict(coef, coef_split_idx, Xs)

        # Get transpose of Cholesky of precision
        R, logdet = self.getR(theta)

        # Standard covariance transform for normal random variable:
        res = (y - mus) @ R.T

        if mean is not None:
            res = res[:, mean].reshape(-1, 1)

        return res

    def predict(
        self,
        coef: np.ndarray,
        coef_split_idx: list[int],
        Xs: list[scp.sparse.csc_array | None],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gets the predicted means, variance, and co-variance parameters for given coefficients
        and model matrices.

        Examples::

            from mssm.models import *
            from mssmViz.sim import *
            from mssmViz.plot import *
            import matplotlib.pyplot as plt

            # Simulate data
            sim_dat = sim16(500,seed=1134,correlate=True)

            # We need formulas for each mean!
            formulas = [
                Formula(lhs("y0"), [i(), f(["x0"])], data=sim_dat),
                Formula(lhs("y1"), [i(), f(["x1"]), f(["x2"])], data=sim_dat),
                Formula(lhs("y2"), [i(), f(["x3"])], data=sim_dat)
            ]

            # Now define the model...
            model = GSMM(formulas, MultiGauss(3,[Identity() for _ in range(3)]))

            # ... and fit!
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(method='qEFS')

            # Can now extract fitted means and theta vector
            Xs = model.get_mmat()
            mus, theta = model.family.predict(model.coef,model.coef_split_idx,Xs)

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for \
            General Smooth Models.

        :param coef: The current coefficient estimate (as np.array of shape (-1,1) - so it must not
            be flattened!). **Note** the last ``int(pars * (pars + 1) / 2)`` elements contain the
            log(variance) and co-variance parameters.
        :type coef: np.ndarray
        :param coef_split_idx: A list used to split (via :func:`np.split`) the ``coef`` into the
            sub-sets associated with each mean and a final sub-set containing all log(variance) and
            co-variance parameters.
        :type coef_split_idx: [int]
        :param ys: List containing the vectors of observations.
        :type ys: [np.ndarray]
        :param Xs: A list containing the sparse model matrices associated with the models of the
            means. **Note**, this implementation allows to make use of the ``build_mat`` argument
            of the :func:`GSMM.fit` method. Specifically, for means that have the same predictor
            structure as the first mean we can set ``build_mat[idx] = False``. The code then
            automatically assigns ``X[idx] = X[0]``. See the :func:`GSMM.fit` documentation for
            more details.
        :type Xs: [scp.sparse.csc_array | None]
        :return: The predicted means as a ``(N, self.n_par)`` numpy array and the theta vector (as
            a flattened numpy array), holding the variance and covariance parameters.
        :rtype: tuple[np.ndarray, np.ndarray]
        """

        # Extract extra info and fix model matrices, then compute mu and theta
        Xfix = [X if X is not None else Xs[0] for X in Xs]

        split_coef = np.split(coef, coef_split_idx)
        mus = np.concatenate(
            [
                self.links[mui].fi(Xfix[mui] @ split_coef[mui].reshape(-1, 1))
                for mui in range(self.n_par)
            ],
            axis=1,
        )
        theta = split_coef[-1].flatten()

        return mus, theta

    def rvs(
        self,
        mus: np.ndarray,
        theta: np.ndarray,
        size: int = 1,
        seed: int | None = 0,
    ) -> np.ndarray:
        """Computes Random samples from a multivariate normal model given coefficients in the
        additive models of the mean and the log(variance) and co-variance parameters.

        **Note**, the numpy array returned is of shape ``(size, self.n_pars, mus.shape[0])``. See
        the return description for details.

        Examples::

            from mssm.models import *
            from mssmViz.sim import *
            from mssmViz.plot import *
            import matplotlib.pyplot as plt

            # Simulate data
            sim_dat = sim16(500,seed=1134,correlate=True)

            # We need formulas for each mean!
            formulas = [
                Formula(lhs("y0"), [i(), f(["x0"])], data=sim_dat),
                Formula(lhs("y1"), [i(), f(["x1"]), f(["x2"])], data=sim_dat),
                Formula(lhs("y2"), [i(), f(["x3"])], data=sim_dat)
            ]

            # Now define the model...
            model = GSMM(formulas, MultiGauss(3,[Identity() for _ in range(3)]))

            # ... and fit!
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(method='qEFS')

            # Can now extract fitted means and theta vector
            Xs = model.get_mmat()
            mus, theta = model.family.predict(model.coef,model.coef_split_idx,Xs)

            # Can now generate random samples
            # rvs will be of shape (10000, 3, 500) for 10000 samples, a 3 dimensional
            # multivariate Gaussian, and 500 predicted mean vectors
            rvs = model.family.rvs(mus,theta,size=10000)

        References:
         - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for \
            General Smooth Models.

        :param mus: Numpy array of shape ``(N, self.n_par)``, where ``self.n_par`` is the dimension
            of the multivariate normal distributions.
        :type mus: np.ndarray
        :param theta: Flattened array holding inverses of log(variance) and co-variance parameters
        :type theta: np.ndarray
        :param size: Number of random samples to return per distribution. Defaults to 1.
        :type size: int, optional
        :param seed: Seed to use for random number generation. Defaults to 0.
        :type seed: int, optional
        :return: a numpy array of shape ``(size, self.n_pars, mus.shape[0])`` containing random
            samples from ``i`` mulitvariate normal distributions with means ``mus[i,:]``.
        :rtype: np.ndarray
        """

        rvs = np.zeros((size, self.n_par, mus.shape[0]))

        # Get transpose of Cholesky of precision
        R, _ = self.getR(theta)

        cov = np.linalg.inv(R.T @ R)

        for mui in range(mus.shape[0]):
            rvs[:, :, mui] = scp.stats.multivariate_normal.rvs(
                size=size,
                mean=mus[mui, :],
                cov=cov,
                random_state=seed,
            )

            # Update seed
            if seed is not None:
                seed += 1

        return rvs
