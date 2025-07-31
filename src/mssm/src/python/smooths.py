import numpy as np
import scipy as scp

##################################### Spline Convolution Code  #####################################

def convolve_event(f:np.ndarray,pulse_location:int) -> np.ndarray:
  """Convolution of function ``f`` with dirac delta spike centered around sample ``pulse_locations``.

  Based on code by Wierda et al. 2012

  References:
    - Wierda, S. M., van Rijn, H., Taatgen, N. A., & Martens, S. (2012). Pupil dilation deconvolution reveals the dynamics of attention at high temporal resolution. https://doi.org/10.1073/pnas.1201858109

  :param f: Function evaluated over some samples
  :type f: np.ndarray
  :param pulse_location: Location of spike (in sample)
  :type pulse_location: int
  :return: Convolved function as array
  :rtype: np.ndarray
  """
  # Convolution of function f with dirac delta spike centered around
  # sample pulse_locations[i].
  # Based on code by Wierda et al. 2012
  
  # Create spike
  spike = np.array([0 for _ in range(pulse_location+1)])
  spike[pulse_location] = 1
  
  # Convolve "spike" with function template f
  o = scp.signal.fftconvolve(f,spike,mode="full")
  return o

##################################### B-spline functions #####################################

def tpower(x:np.ndarray, t:np.ndarray, p:int) -> np.ndarray:
  """Computes truncated ``p-t`` power function of ``x``.

  Function taken from "Splines, Knots, and Penalties" by Eilers & Marx (2010)

  References:
    - Eilers, P., & Marx, B. (2010). Splines, knots, and penalties. https://doi.org/10.1002/WICS.125

  :param x: Covariate
  :type x: np.ndarray
  :param t: knot location vector
  :type t: np.ndarray
  :param p: degrees of spline basis
  :type p: int
  :return: ``np.power(x - t,p) * (x > t)``
  :rtype: np.ndarray
  """
  return np.power(x - t,p) * (x > t)

def bbase(x:np.ndarray, knots:np.ndarray, dx:float, deg:int) -> np.ndarray:
  """Computes B-spline basis of degree ``deg`` given ``knots`` and interval spacing ``dx``.

  Function taken from "Splines, Knots, and Penalties" by Eilers & Marx (2010)

  References:
    - Eilers, P., & Marx, B. (2010). Splines, knots, and penalties. https://doi.org/10.1002/WICS.125

  :param x: Covariate
  :type x: np.ndarray
  :param knots: knot location vector
  :type knots: np.ndarray
  :param dx: Interval spacing ``(xr-xl) / ndx`` where ``xr`` and ``xl`` are max and min of ``x`` and ``ndx=nk-deg`` where ``nk`` is the number of basis functions.
  :type dx: float
  :param deg: Degree of basis
  :type deg: int
  :return: numpy.array of shape (-1,``nk``)
  :rtype: np.ndarray
  """
  P = tpower(x[:,None],knots,deg)
  n = P.shape[1] # Actually n + 1 + 2*deg
  D = np.diff(np.identity(n),n=deg+1) / (scp.special.gamma(deg + 1) * np.power(dx,deg))
  B = np.power(-1, deg + 1) * P @ D
  return B

def B_spline_basis(cov:np.ndarray, event_onset:int|None, nk:int, min_c:float|None=None, max_c:float|None=None, drop_outer_k:bool=False, convolve:bool=False, deg:int=3) -> np.ndarray:
  """Computes B-spline basis of degree ``deg`` given ``knots``.

  Based on code and definitions in "Splines, Knots, and Penalties" by Eilers & Marx (2010) and adapted to allow for convolving B-spline bases.

  References:
    - Eilers, P., & Marx, B. (2010). Splines, knots, and penalties. https://doi.org/10.1002/WICS.125

  :param cov: Flattened covariate array (i.e., of shape (-1,))
  :type cov: np.ndarray
  :param event_onset: Sample on which to place a dirac delta with which the B-spline bases should be convolved - ignored if ``convolve==False``.
  :type event_onset: int | None
  :param nk: Number of basis functions to create
  :type nk: int
  :param min_c: Minimum covariate value, defaults to None
  :type min_c: float | None, optional
  :param max_c: Maximum covariate value, defaults to None
  :type max_c: float | None, optional
  :param drop_outer_k: Deprecated, defaults to False
  :type drop_outer_k: bool, optional
  :param convolve: Whether basis functions should be convolved (i.e., time-shifted) with an impulse response function triggered at ``event_onset``, defaults to False
  :type convolve: bool, optional
  :param deg: Degree of basis, defaults to 3
  :type deg: int, optional
  :return: An array of shape ``(-1,nk)`` holding the ``nk`` Basis functions evaluated over ``x`` and optionally convolved with an impulse response function triggered at ``event_onset``
  :rtype: np.ndarray
  """
  # Setup basis with even knot locations.
  # Code based on Eilers, P., & Marx, B. (2010). Splines, knots, and penalties. https://doi.org/10.1002/WICS.125
  # See also: Eilers, P. H. C., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. https://doi.org/10.1214/ss/1038425655

  xl = min(cov)
  xr = max(cov)

  if not max_c is None:
    xr = max_c

  if not min_c is None:
    xl = min_c
    
  # ndx is equal to n in Eilers & Marx (2011)
  # So there will be n-1 knots (without expansion)
  # n + 1 + 2*deg knots with expansion
  # and nk basis functions computed.
  ndx = nk - deg

  if convolve:
    # For the IR GAMM, the ***outer***-most knots should be close to min_c and max_c.
    # Hence, we simply provide n + 1 + 2*deg (see above) equally spaced knots from min_c to max_c.
    dx = (xr-xl) / (ndx + 2 * deg)
    knots = np.linspace(min_c, max_c, ndx + 1 + 2 * deg)
  else:
    dx = (xr-xl) / ndx
    knots = np.linspace(xl - dx * deg, xr + dx * deg,ndx + 1 + 2 * deg)

  B = bbase(cov,knots,dx,deg)

  if convolve:
    o_restr = np.zeros(B.shape)

    for nki in range(nk):
      o = convolve_event(B[:,nki],event_onset)
      o_restr[:,nki] = o[0:len(cov)]

    B = o_restr
  
  return B

def TP_basis_calc(cTP:np.ndarray,nB:np.ndarray) -> np.ndarray:
  """Computes row-wise Kroenecker product between ``cTP`` and ``nB``. Useful to create a Tensor smooth basis.

  See Wood(2017) 5.6.1 and B.4.

  References:
    - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).

  :param cTP: Marginal basis or partially accumulated tensor smooth basis
  :type cTP: np.ndarray
  :param nB: Marginal basis to include in the tensor smooth
  :type nB: np.ndarray
  :return: The row-wise Kroenecker product between ``cTP`` and ``nB``
  :rtype: np.ndarray
  """
  # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.khatri_rao.html
  # Function performs col-wise Kron - we need row-wise for the Tensor smooths
  # see Wood(2017) 5.6.1 and B.4
  # ToDo: Sparse calculation might be desirable..
  return scp.linalg.khatri_rao(cTP.T,nB.T).T