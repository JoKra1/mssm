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
  return (x - t) ** (p * (x > t))

def bbase(x, knots, dx, deg):
   # Function taken from "Splines, Knots, and Penalties" by Eilers & Marx (2010)
   P = P = tpower(x[:,None],knots,deg)
   n = P.shape[1]
   D = np.diff(np.identity(n),n=deg+1) / (scp.special.gamma(deg + 1) * dx ** deg)
   B = (-1) ** (deg + 1) * P.dot(D)
   return B

def B_spline_basis(i, cov, state_est, nk, drop_outer_k=False, convolve=False, min_c=None, max_c=None, deg=2):
  # Setup basis with even knot locations.
  # Code based on "Splines, Knots, and Penalties" by Eilers & Marx (2010)
  # However, knot location calculation is taken directly from mgcv (Wood, 2017)

  xl = min(cov)
  xr = max(cov)

  if not max_c is None:
     xr = max_c

  if not min_c is None:
     xl = min_c

  rg = xr - xl

  if drop_outer_k:
    ndx = (nk - deg + 2*deg)
     
  else:
    ndx = nk - deg

  xr += 0.001*rg
  xl -= 0.001*rg

  dx = (xr-xl) / (ndx-1)
  knots = np.linspace(xl - dx * (deg + 1), xr + dx * (deg + 1),ndx+2*deg+2)

  B = bbase(cov,knots,dx,deg+1)

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