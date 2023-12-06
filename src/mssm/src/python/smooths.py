import numpy as np
import scipy as scp

##################################### Conventional Pupil basis  #####################################

def convolve_event(f,pulse_locations,i):
  # Convolution of function f with dirac delta spike centered around
  # sample pulse_locations[i].
  # Based on code by Wierda et al. 2012
  
  # Create spike
  spike = np.array([0 for _ in range(max(pulse_locations)+1)])
  spike[pulse_locations[i]] = 1
  
  # Convolve "spike" with function template f
  o = scp.signal.fftconvolve(f,spike,mode="full")
  return o


def h_basis(i,time,pulse_locations,n=10.1,t_max=930,f=1e-24):
  # Response function from Hoeks and Levelt
  # + scale parameter introduced by Wierda et al. 2012
  # Based on code by Wierda et al. 2012
  # n+1 = number of laters
  # t_max = response maximum
  # f = scaling factor
  h = f*(time**n)*np.exp(-n*time/t_max)
  
  # Convolve "spike" defined by peaks with h
  o = convolve_event(h,pulse_locations,i)
  
  # Keep only the realization of the response function
  # within the un-expanded time window
  o_restr = o[0:len(time)]
  return o_restr

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

def B_spline_basis(i, cov, state_est, nk, drop_outer_k=False, convolve=False, min_c=None, max_c=None, deg=3):
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
  # drop_outer is experimental.
  if drop_outer_k:
    ndx = (nk - deg + 2*deg)
     
  else:
    ndx = nk - deg

  dx = (xr-xl) / ndx
  knots = np.linspace(xl - dx * deg, xr + dx * deg,ndx + 1 + 2 * deg)

  B = bbase(cov,knots,dx,deg)

  if drop_outer_k:
     B = B[:,deg:-deg]

  if convolve:
    o_restr = np.zeros(B.shape)

    for nki in range(nk):
      o = convolve_event(B[:,nki],state_est,i)
      o_restr[:,nki] = o[0:len(cov)]

    B = o_restr
  
  return B

def TP_basis_calc(cTP,nB):
   # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.khatri_rao.html
   # Function performs col-wise Kron - we need row-wise for the Tensor smooths
   # see Wood(2017) 5.6.1 and B.4
   # ToDo: Sparse calculation might be desirable..
   return scp.linalg.khatri_rao(cTP.T,nB.T).T