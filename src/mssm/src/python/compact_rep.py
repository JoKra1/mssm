import copy
import numpy as np
import scipy as scp

################################################ Compact Representations of quasi-Newton updates for L-qEFS update ################################################

def computeH(s:np.ndarray,y:np.ndarray,rho:np.ndarray,H0:scp.sparse.csc_array,explicit:bool=True)  -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
   """Computes (explicitly or implicitly) the quasi-Newton approximation to the negative Hessian of the (penalized) likelihood :math:`\\mathbf{H}` (:math:`\\mathcal{H}`) from the L-BFGS-B optimizer info.

   Relies on equations 2.16 in Byrd, Nocdeal & Schnabel (1992).

   References:
    - Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994). Representations of quasi-Newton matrices and their use in limited memory methods. Mathematical Programming, 63(1), 129–156. https://doi.org/10.1007/BF01582063

   :param s: np.ndarray of shape (m,p), where p is the number of coefficients, holding the first set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type s: np.ndarray
   :param y: np.ndarray of shape (m,p), where p is the number of coefficients, holding the second set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type y: np.ndarray
   :param rho: flattened numpy.array of shape (m,), holding element-wise ```1/y.T@s`` from Byrd, Nocdeal & Schnabel (1992).
   :type rho: np.ndarray
   :param H0: Initial estimate for the hessian of the negative (penalized) likelihood. Here some multiple of the identity (multiplied by ``omega``).
   :type H0: scipy.sparse.csc_array
   :param explicit: Whether or not to return the approximate matrix explicitly or implicitly in form of four update matrices.
   :type explicit: bool
   :return: H, either as np.ndarray (``explicit=='True'``) or represented implicitly via four update vectors (also np.ndarrays)
   :rtype: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
   """
   # Number of updates?
   m = len(y)

   # First form S,Y, and D
   S = s.T
   Y = y.T
   
   STS = S.T@H0@S
   DK = np.identity(m)
   DK[0,0] *= np.dot(s[0],y[0])

   # Now use eq. 2.5 to compute R - only have to do this once
   R0 = np.dot(s[0], y[0]).reshape(1,1)
   R = R0
   for k in range(1,m):
   
      DK[k,k] *= np.dot(s[k],y[k])
      
      R = np.concatenate((np.concatenate((R0,S[:,:k].T@Y[:,[k]]),axis=1),
                          np.concatenate((np.zeros((1,R0.shape[1])),
                                             np.array([1/rho[k]]).reshape(1,1)),axis=1)),axis=0)
      
      R0 = R
   
   # Eq 2.22
   L = S.T@Y - R

   # Now compute term 2 in 3.13 of Byrd, Nocdeal & Schnabel (1992)
   t2 = np.zeros((2*m,2*m))
   t2[:m,:m] = STS
   t2[:m,m:] = L
   t2[m:,:m] = L.T
   t2[m:,m:] = -1*DK

   # We actually need the inverse to compute H

   # Eq 2.26 of Byrd, Nocdeal & Schnabel (1992)
   Dinv = copy.deepcopy(DK)
   Dpow = copy.deepcopy(DK)
   Dnpow = copy.deepcopy(DK)
   for k in range(m):
      Dinv[k,k] = 1/Dinv[k,k]
      Dpow[k,k] = np.power(Dpow[k,k],0.5)
      Dnpow[k,k] = np.power(Dnpow[k,k],-0.5)

   JJT = STS + L@Dinv@L.T
   J = scp.linalg.cholesky(JJT, lower=True)

   t2L = np.zeros((2*m,2*m))
   t2L[:m,:m] = Dpow
   t2L[m:,:m] = (-1*L)@Dnpow
   t2L[m:,m:] = J

   t2U = np.zeros((2*m,2*m))
   t2U[:m,:m] = -1*Dpow
   t2U[:m:,m:] = Dnpow@L.T
   t2U[m:,m:] = J.T

   t2_flip = t2L@t2U

   invt2L = scp.linalg.inv(t2L)
   invT2U = scp.linalg.inv(t2U)
   invt2 = invt2L.T@invT2U.T

   t2_sort = np.zeros((2*m,2*m))
   # top left <- bottom right
   t2_sort[:m,:m] = t2_flip[m:,m:]
   # top right <- bottom left
   t2_sort[:m,m:] = t2_flip[m:,:m]
   # bottom left <- top right
   t2_sort[m:,:m] = t2_flip[:m,m:]
   # bottom right <- top left
   t2_sort[m:,m:] = t2_flip[:m,:m]
   
   invt2_sort = np.zeros((2*m,2*m))
   # top left <- bottom right
   invt2_sort[:m,:m] = invt2[m:,m:]
   # top right <- bottom left
   invt2_sort[:m,m:] = invt2[m:,:m]
   # bottom left <- top right
   invt2_sort[m:,:m] = invt2[:m,m:]
   # bottom right <- top left
   invt2_sort[m:,m:] = invt2[:m,:m]

   # And terms 1 and 2
   t1 = np.concatenate((H0@S,Y),axis=1)
   t3 = np.concatenate((S.T@H0,Y.T),axis=0)

   # Return matrix in compact representation
   if explicit == False:
      return t1, (-1*t2_sort), (-1*invt2_sort), t3
      
   H = H0 + t1@(-1*invt2_sort)@t3
   
   return H

def computeV(s:np.ndarray,y:np.ndarray,rho:np.ndarray,V0:scp.sparse.csc_array,explicit:bool=True) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
   """Computes (explicitly or implicitly) the quasi-Newton approximation to the inverse of the negative Hessian of the (penalized) likelihood :math:`\\mathcal{I}` (:math:`\\mathbf{V}`) from the L-BFGS-B optimizer info.

   Relies on equations 2.16 and 3.13 in Byrd, Nocdeal & Schnabel (1992).

   References:
    - Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994). Representations of quasi-Newton matrices and their use in limited memory methods. Mathematical Programming, 63(1), 129–156. https://doi.org/10.1007/BF01582063

   :param s: np.ndarray of shape (m,p), where p is the number of coefficients, holding the first set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type s: np.ndarray
   :param y: np.ndarray of shape (m,p), where p is the number of coefficients, holding the second set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type y: np.ndarray
   :param rho: flattened numpy.array of shape (m,), holding element-wise ```1/y.T@s`` from Byrd, Nocdeal & Schnabel (1992).
   :type rho: np.ndarray
   :param V0: Initial estimate for the inverse of the hessian of the negative (penalized) likelihood. Here some multiple of the identity (multiplied by ``omega``).
   :type V0: scipy.sparse.csc_array
   :param explicit: Whether or not to return the approximate matrix explicitly or implicitly in form of three update matrices.
   :type explicit: bool
   :return: V, either as np.ndarray (``explicit=='True'``) or represented implicitly via three update vectors (also np.ndarrays)
   :rtype: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]
   """
   m = len(y)
   # First form S,Y, and D
   S = s.T
   Y = y.T
   
   DYTY = Y.T@V0@Y

   DYTY[0,0] += np.dot(s[0],y[0])

   # Now use eq. 2.5 to compute R^{-1} - only have to do this once
   Rinv0 = 1/np.dot(s[0], y[0]).reshape(1,1)
   Rinv = Rinv0
   for k in range(1,m):
   
      DYTY[k,k] += np.dot(s[k],y[k])
      
      Rinv = np.concatenate((np.concatenate((Rinv0,(-rho[k])*Rinv0@S[:,:k].T@Y[:,[k]]),axis=1),
                           np.concatenate((np.zeros((1,Rinv0.shape[1])),
                                             np.array([rho[k]]).reshape(1,1)),axis=1)),axis=0)
      
      Rinv0 = Rinv
   
   # Now compute term 2 in 3.13 used for all S_j
   t2 = np.zeros((2*m,2*m))
   t2[:m,:m] = Rinv.T@DYTY@Rinv
   t2[:m,m:] = -Rinv.T
   t2[m:,:m] = -Rinv

   # And terms 1 and 2
   t1 = np.concatenate((S,V0@Y),axis=1)
   t3 = np.concatenate((S.T,Y.T@V0),axis=0)

   if explicit:
      V = V0 + t1@t2@t3
      return V
   else:
      return t1, t2, t3
   
def computeVSR1(s:np.ndarray,y:np.ndarray,rho:np.ndarray,V0:scp.sparse.csc_array,omega:float=1,make_psd:bool=False,explicit:bool=True) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
   """Computes (explicitly or implicitly) the symmetric rank one (SR1) approximation of the inverse of the negative Hessian of the (penalized) likelihood :math:`\\mathcal{I}` (:math:`\\mathbf{V}`).

   Relies on equations 2.16 and 3.13 in Byrd, Nocdeal & Schnabel (1992). Can ensure positive (semi) definiteness of the approximation via an eigen decomposition as shown by Burdakov et al. (2017). This is enforced via the ``make_psd`` argument.

   References:
    - Burdakov, O., Gong, L., Zikrin, S., & Yuan, Y. (2017). On efficiently combining limited-memory and trust-region techniques. Mathematical Programming Computation, 9(1), 101–134. https://doi.org/10.1007/s12532-016-0109-7
    - Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994). Representations of quasi-Newton matrices and their use in limited memory methods. Mathematical Programming, 63(1), 129–156. https://doi.org/10.1007/BF01582063

   :param s: np.ndarray of shape (m,p), where p is the number of coefficients, holding the first set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type s: np.ndarray
   :param y: np.ndarray of shape (m,p), where p is the number of coefficients, holding the second set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type y: np.ndarray
   :param rho: flattened numpy.array of shape (m,), holding element-wise ```1/y.T@s`` from Byrd, Nocdeal & Schnabel (1992).
   :type rho: np.ndarray
   :param V0: Initial estimate for the inverse of the hessian of the negative (penalized) likelihood. Here some multiple of the identity (multiplied by ``omega``).
   :type V0: scipy.sparse.csc_array
   :param omega: Multiple of the identity matrix used as initial estimate.
   :type omega: float, optional
   :param make_psd: Whether to enforce PSD as mentioned in the description. By default set to False.
   :type make_psd: bool, optional
   :param explicit: Whether or not to return the approximate matrix explicitly or implicitly in form of three update matrices.
   :type explicit: bool
   :return: V, either as np.ndarray (``explicit=='True'``) or represented implicitly via three update vectors (also np.ndarrays)
   :rtype: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]
   """
   m = len(y)
   # First form S,Y, and D
   S = s.T
   Y = y.T
   
   YTY = Y.T@V0@Y
   DK = np.identity(m)
   DK[0,0] *= np.dot(s[0],y[0])

   # Now use eq. 2.5 to compute R - only have to do this once
   R0 = np.dot(s[0], y[0]).reshape(1,1)
   R = R0
   for k in range(1,m):
   
      DK[k,k] *= np.dot(s[k],y[k])
      
      R = np.concatenate((np.concatenate((R0,S[:,:k].T@Y[:,[k]]),axis=1),
                          np.concatenate((np.zeros((1,R0.shape[1])),
                                             np.array([1/rho[k]]).reshape(1,1)),axis=1)),axis=0)
      
      R0 = R
   
   # Eq 2.22
   L = S.T@Y - R
   
   # Now compute term 2 in eq. 5.2
   t2 = scp.linalg.inv(R + R.T - DK - YTY)

   # And terms 1 and 2
   t1 = S - V0@Y
   t3 = t1.T

   if make_psd:
      # Compute implicit eigen decomposition as shown by Burdakov et al. (2017)
      Q,R = scp.linalg.qr(t1,mode='economic')
      Rit2R = R@(t2)@R.T

      # ev holds non-zero eigenvalues of U@D@U.T (e.g., Burdakov et al. (2017))
      ev, P = scp.linalg.eigh(Rit2R,driver='ev')
      
      # Now find closest PSD.
      fix_idx = (ev + omega) <= 0
      
      if np.sum(fix_idx) > 0:
         #print("fix VSR1",np.sum(fix_idx),omega,1/omega)
         ev[fix_idx] =  (-1*omega) #+ np.power(np.finfo(float).eps,0.9)

         #while np.any(np.abs(ev) < 1e-7):
         #   ev[np.abs(ev) < 1e-7] += np.power(np.finfo(float).eps,0.7)

         #print(f"implicit ev post shift. abs min: {np.min(np.abs(ev))}, min: {np.min(ev)}, min + 1: {np.min(ev+omega)}, max + 1: {np.max(ev+omega)}",ev[:10]+omega)

         # Now Q @ P @ np.diag(ev) @ P.T @ Q.T = shifted PSD version of U@D@U.T
         # so we can set:
         # shifted_invt2=np.diag(ev)
         # shifted_t2 = np.diag(1/ev)
         # t1 = Q @ P
         # t3 = t1.T = P.T @ Q.T
         t1 = Q @ P
         t3 = P.T@Q.T
         t2 = np.diag(ev)

   if explicit:
      V = V0 + t1@t2@t3
      return V
   else:
      return t1, t2, t3
   
def computeHSR1(s:np.ndarray,y:np.ndarray,rho:np.ndarray,H0:scp.sparse.csc_array,omega:float=1,make_psd:bool=False,make_pd:bool=False,explicit:bool=True) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
   """Computes, (explicitly or implicitly) the symmetric rank one (SR1) approximation of the negative Hessian of the (penalized) likelihood :math:`\\mathbf{H}` (:math:`\\mathcal{H}`).

   Relies on equations 2.16 and 3.13 in Byrd, Nocdeal & Schnabel (1992). Can ensure positive (semi) definiteness of the approximation via an eigen decomposition as shown by Burdakov et al. (2017). This is enforced via the ``make_psd`` and ``make_pd`` arguments.

   References:
    - Burdakov, O., Gong, L., Zikrin, S., & Yuan, Y. (2017). On efficiently combining limited-memory and trust-region techniques. Mathematical Programming Computation, 9(1), 101–134. https://doi.org/10.1007/s12532-016-0109-7
    - Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994). Representations of quasi-Newton matrices and their use in limited memory methods. Mathematical Programming, 63(1), 129–156. https://doi.org/10.1007/BF01582063

   :param s: np.ndarray of shape (m,p), where p is the number of coefficients, holding the first set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type s: np.ndarray
   :param y: np.ndarray of shape (m,p), where p is the number of coefficients, holding the second set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type y: np.ndarray
   :param rho: flattened numpy.array of shape (m,), holding element-wise ```1/y.T@s`` from Byrd, Nocdeal & Schnabel (1992).
   :type rho: np.ndarray
   :param H0: Initial estimate for the hessian of the negative (penalized) likelihood. Here some multiple of the identity (multiplied by ``omega``).
   :type H0: scipy.sparse.csc_array
   :param omega: Multiple of the identity matrix used as initial estimate.
   :type omega: float, optional
   :param make_psd: Whether to enforce PSD as mentioned in the description. By default set to False.
   :type make_psd: bool, optional
   :param make_pd: Whether to enforce numeric positive definiteness, not just PSD. Ignored if ``make_psd=False``. By default set to False.
   :type make_pd: bool, optional
   :param explicit: Whether or not to return the approximate matrix explicitly or implicitly in form of three update matrices.
   :type explicit: bool
   :return: H, either as np.ndarray (``explicit=='True'``) or represented implicitly via three update vectors (also np.ndarrays)
   :rtype: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]
   """
   m = len(y)
   # First form S,Y, and D
   S = s.T
   Y = y.T
   
   STS = S.T@H0@S
   DK = np.identity(m)
   DK[0,0] *= np.dot(s[0],y[0])

   # Now use eq. 2.5 to compute R - only have to do this once
   R0 = np.dot(s[0], y[0]).reshape(1,1)
   R = R0
   for k in range(1,m):
   
      DK[k,k] *= np.dot(s[k],y[k])
      
      R = np.concatenate((np.concatenate((R0,S[:,:k].T@Y[:,[k]]),axis=1),
                          np.concatenate((np.zeros((1,R0.shape[1])),
                                             np.array([1/rho[k]]).reshape(1,1)),axis=1)),axis=0)
      
      R0 = R
   
   # Eq 2.22
   L = S.T@Y - R
   
   # Now compute term 2 in eq. 5.2
   t2 = scp.linalg.inv(DK + L + L.T - STS)

   # And terms 1 and 2
   t1 = Y - H0@S
   t3 = t1.T

   if make_psd:
      # Compute implicit eigen decomposition as shown by Burdakov et al. (2017)
      Q,R = scp.linalg.qr(t1,mode='economic')
      Rit2R = R@(t2)@R.T

      # ev holds non-zero eigenvalues of U@D@U.T (e.g., Burdakov et al. (2017))
      ev, P = scp.linalg.eigh(Rit2R,driver='ev')
      
      # Now find closest PSD.
      fix_idx = (ev + omega) <= 0
      
      if np.sum(fix_idx) > 0:
         #print("fix VSR1",np.sum(fix_idx),omega,1/omega)
         ev[fix_idx] =  (-1*omega) #+ np.power(np.finfo(float).eps,0.9)

         # Useful to guarantee that penalized hessian is pd at convergence
         if make_pd:
            ev[fix_idx] += np.power(np.finfo(float).eps,0.9)

         #while np.any(np.abs(ev) < 1e-7):
         #   ev[np.abs(ev) < 1e-7] += np.power(np.finfo(float).eps,0.7)

         #print(f"implicit ev post shift. abs min: {np.min(np.abs(ev))}, min: {np.min(ev)}, min + 1: {np.min(ev+omega)}, max + 1: {np.max(ev+omega)}",ev[:10]+omega)

         # Now Q @ P @ np.diag(ev) @ P.T @ Q.T = shifted PSD version of U@D@U.T
         # so we can set:
         # shifted_invt2=np.diag(ev)
         # shifted_t2 = np.diag(1/ev)
         # t1 = Q @ P
         # t3 = t1.T = P.T @ Q.T
         t1 = Q @ P
         t3 = P.T@Q.T
         t2 = np.diag(ev)

   if explicit:
      H = H0 + t1@t2@t3
      return H
   else:
      return t1, t2, t3
