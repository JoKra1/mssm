import numpy as np
import scipy as scp

##################################### Spline Convolution Code  #####################################

def convolve_event(f,pulse_location):
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

def tpower(x, t, p):
  # Truncated p-th power function
  # Function taken from "Splines, Knots, and Penalties" by Eilers & Marx (2010)
  return np.power(x - t,p) * (x > t)

def bbase(x, knots, dx, deg):
   # Function taken from "Splines, Knots, and Penalties" by Eilers & Marx (2010)
   P = tpower(x[:,None],knots,deg)
   n = P.shape[1] # Actually n + 1 + 2*deg
   D = np.diff(np.identity(n),n=deg+1) / (scp.special.gamma(deg + 1) * np.power(dx,deg))
   B = np.power(-1, deg + 1) * P @ D
   return B

def B_spline_basis(cov, event_onset, nk, drop_outer_k=False, convolve=False, min_c=None, max_c=None, deg=3):
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

def TP_basis_calc(cTP,nB):
   # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.khatri_rao.html
   # Function performs col-wise Kron - we need row-wise for the Tensor smooths
   # see Wood(2017) 5.6.1 and B.4
   # ToDo: Sparse calculation might be desirable..
   return scp.linalg.khatri_rao(cTP.T,nB.T).T