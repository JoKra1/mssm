import copy
import numpy as np
import scipy as scp

def computeH_Brust(s,y,rho,H0):
   """Computes explicitly the negative Hessian of the penalized likelihood :math:`\mathbf{H}` from the L-BFGS-B optimizer info.

   Relies on equations 2.6 in Brust (2024).

   References:
    - Brust, J. J. (2024). Useful Compact Representations for Data-Fitting (arXiv:2403.12206). arXiv. https://doi.org/10.48550/arXiv.2403.12206
    - Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994). Representations of quasi-Newton matrices and their use in limited memory methods. Mathematical Programming, 63(1), 129–156. https://doi.org/10.1007/BF01582063

   :param s: List holding the first set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type s: [numpy.array]
   :param y: List holding the second set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type y: [numpy.array]
   :param rho: List holding element-wise ```1/y.T@s`` from Byrd, Nocdeal & Schnabel (1992).
   :type rho: [numpy.array]
   :param H0: Initial estimate for the hessian fo the negative penalized likelihood. Here some multiple of the identity (multiplied by ``omega``) plus the embedded penalty matrix.
   :type H0: scipy.sparse.csc_array
   :param make_psd: Whether to enforce PSD as mentioned in the description. By default set to False.
   :type make_psd: bool, optional
   :param omega: Multiple used to get initial estimate for the hessian fo the negative penalized likelihood. Defaults to 1.
   :type omega: float, optional
   """
   # Number of updates?
   m = len(y)
   S = np.array(s).T
   Y = np.array(y).T
   C = S

   # Compute D & R
   D = np.identity(m)
   D[0,0] *= np.dot(s[0],y[0])

   # Now use eq. 2.5 of Brust (2024) to compute R.
   # This is the same as in Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994) essentially.
   R0 = np.dot(s[0], y[0]).reshape(1,1)
   R = R0
   for k in range(1,m):
   
      D[k,k] *= np.dot(s[k],y[k])
      
      R = np.concatenate((np.concatenate((R0,S[:,:k].T@Y[:,[k]]),axis=1),
                          np.concatenate((np.zeros((1,R0.shape[1])),
                                             np.array([1/rho[k]]).reshape(1,1)),axis=1)),axis=0)
      
      R0 = R
   
   # Compute C.T@S and extract the upper triangular part from that matrix as shown by Brust (2024)
   CTS = C.T@S
   RCS = np.triu(CTS)

   # We now need inverse of RCS
   RCS_inv = scp.linalg.solve_triangular(RCS, np.identity(m),lower=False)

   # Can now form inverse of middle block from Brust (2024)
   t2inv = np.zeros((2*m,2*m))
   t2inv[:m,:m] = (-1*RCS_inv.T) @ (R + R.T - (D + S.T@H0@S)) @ RCS_inv # Upper left
   t2inv[:m,m:] = RCS_inv.T # Upper right
   t2inv[m:,:m] = RCS_inv # Lower left
   #t2inv[m:,m:] = 0 # Lower right remains empty

   t2 = np.zeros((2*m,2*m))
   t2[:m,m:] = RCS # Upper right
   t2[m:,:m] = RCS.T # Lower left
   t2[m:,m:] = (R + R.T - (D + S.T@H0@S)) # Lower right

   # Can now compute remaining terms to compute H as shown by Brust (2024)
   t1 = np.concatenate((C,Y - H0@S),axis=1)
   t3 = np.concatenate((C.T, (Y - H0@S).T),axis=0)

   H = H0 + t1@t2inv@t3
   
   return H

def computeH(s,y,rho,H0,make_psd=False,omega=1,explicit=True):
   """Computes explicitly the negative Hessian of the penalized likelihood :math:`\mathbf{H}` from the L-BFGS-B optimizer info.

   Relies on equations 2.16 in Byrd, Nocdeal & Schnabel (1992). Adapted here to work for the case where ``H0``:math:`=\mathbf{I}*\omega + \mathbf{S}_{\lambda}` and
   we need :math:`\mathbf{I}*\omega + \mathbf{U}\mathbf{D}\mathbf{U}^T` to be PSD. :math:`\mathbf{U}\mathbf{D}\mathbf{U}^T` is the update matrix for the
   negative Hessian of the penalized likelihood, **not** the inverse (:math:`\mathbf{V}`)! For this the implicit eigenvalue decomposition of Brust (2024) is used.

   References:
    - Brust, J. J. (2025). Useful Compact Representations for Data-Fitting (arXiv:2403.12206). arXiv. https://doi.org/10.48550/arXiv.2403.12206
    - Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994). Representations of quasi-Newton matrices and their use in limited memory methods. Mathematical Programming, 63(1), 129–156. https://doi.org/10.1007/BF01582063


   :param s: List holding the first set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type s: [numpy.array]
   :param y: List holding the second set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type y: [numpy.array]
   :param rho: List holding element-wise ```1/y.T@s`` from Byrd, Nocdeal & Schnabel (1992).
   :type rho: [numpy.array]
   :param H0: Initial estimate for the hessian fo the negative penalized likelihood. Here some multiple of the identity (multiplied by ``omega``) plus the embedded penalty matrix.
   :type H0: scipy.sparse.csc_array
   :param make_psd: Whether to enforce PSD as mentioned in the description. By default set to False.
   :type make_psd: bool, optional
   :param omega: Multiple used to get initial estimate for the hessian fo the negative penalized likelihood. Defaults to 1.
   :type omega: float, optional
   """
   # Number of updates?
   m = len(y)

   # First form S,Y, and D
   S = np.array(s).T
   Y = np.array(y).T
   
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


   # We have H0 + U@D@U.T with H0 = I*omega + S_emb and U@D@U.T=t1@(-t2)@t1.T
   # Now enforce that I*omega + t1@(-t2)@t1.T is psd
   if make_psd:
      correction = t1@(-1*invt2_sort)@t3

      # Compute implicit eigen decomposition as shown by Burst (2024)
      Q,R = scp.linalg.qr(t1,mode='economic')
      Rit2R = R@(-1*invt2_sort)@R.T

      # ev holds non-zero eigenvalues of U@D@U.T (e.g., Brust, 2024)
      ev, P = scp.linalg.eigh(Rit2R,driver='ev')

      # Now find closest PSD
      fix_idx = (ev + omega) <= 0
      if np.sum(fix_idx) > 0:
         ev[fix_idx] = (-1*omega)

         # Re-compute correction
         if explicit == False:
            return Q @ P,np.diag(ev),P.T @ Q.T
         
         correction = Q @ P @ np.diag(ev) @ P.T @ Q.T
         
      H = H0 + correction

   else:
      if explicit == False:
         return t1, (-1*invt2_sort), t3
      
      H = H0 + t1@(-1*invt2_sort)@t3
   
   return H

def computeV(s,y,rho,V0,explicit=True):
   """Computes, explicitly (or implicitly) the inverse of the negative Hessian of the penalized likelihood :math:`\mathbf{V}` from the L-BFGS-B optimizer info.

   Relies on equations 2.16 and 3.13 in Byrd, Nocdeal & Schnabel (1992).

   References:
    - Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994). Representations of quasi-Newton matrices and their use in limited memory methods. Mathematical Programming, 63(1), 129–156. https://doi.org/10.1007/BF01582063


   :param s: List holding the first set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type s: [numpy.array]
   :param y: List holding the second set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type y: [numpy.array]
   :param rho: List holding element-wise ```1/y.T@s`` from Byrd, Nocdeal & Schnabel (1992).
   :type rho: [numpy.array]
   :param V0: Initial estimate for the inverse of the hessian fo the negative penalized likelihood.
   :type V0: scipy.sparse.csc_array
   :param explicit: Whether or not to return the approximate matrix explicitly or implicitly in form of the three update vectors.
   :type explicit: bool
   """
   m = len(y)
   # First form S,Y, and D
   S = np.array(s).T
   Y = np.array(y).T
   
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
   
def computeVSR1(s,y,rho,V0,omega=1,make_psd=False,explicit=True):
   """Computes, explicitly (or implicitly) the symmetric rank one (SR1) approximation of the inverse of the negative Hessian of the penalized likelihood :math:`\mathbf{V}`.

   Can ensure positive (semi) definiteness of the approximation via an eigen decomposition as shown by Brust (2024). This is enforced via the ``make_psd`` argument.

   Relies on equations 2.16 and 3.13 in Byrd, Nocdeal & Schnabel (1992).

   References:
    - Brust, J. J. (2025). Useful Compact Representations for Data-Fitting (arXiv:2403.12206). arXiv. https://doi.org/10.48550/arXiv.2403.12206
    - Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994). Representations of quasi-Newton matrices and their use in limited memory methods. Mathematical Programming, 63(1), 129–156. https://doi.org/10.1007/BF01582063

   :param s: List holding the first set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type s: [numpy.array]
   :param y: List holding the second set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type y: [numpy.array]
   :param rho: List holding element-wise ```1/y.T@s`` from Byrd, Nocdeal & Schnabel (1992).
   :type rho: [numpy.array]
   :param V0: Initial estimate for the inverse of the hessian fo the negative penalized likelihood.
   :type V0: scipy.sparse.csc_array
   :param make_psd: Whether to enforce PSD as mentioned in the description. By default set to False.
   :type make_psd: bool, optional
   :param explicit: Whether or not to return the approximate matrix explicitly or implicitly in form of the three update vectors.
   :type explicit: bool
   """
   m = len(y)
   # First form S,Y, and D
   S = np.array(s).T
   Y = np.array(y).T
   
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
      # Compute implicit eigen decomposition as shown by Brust (2024)
      Q,R = scp.linalg.qr(t1,mode='economic')
      Rit2R = R@(t2)@R.T

      # ev holds non-zero eigenvalues of U@D@U.T (e.g., Brust, 2024)
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
   
def computeHSR1(s,y,rho,H0,omega=1,make_psd=False,make_pd=False,explicit=True):
   """Computes, explicitly (or implicitly) the symmetric rank one (SR1) approximation of the negative Hessian of the penalized likelihood :math:`\mathbf{H}`.

   Can ensure positive (semi) definiteness of the approximation via an eigen decomposition as shown by Brust (2024). This is enforced via the ``make_psd`` argument.

   Relies on equations 2.16 and 3.13 in Byrd, Nocdeal & Schnabel (1992).

   References:
    - Brust, J. J. (2025). Useful Compact Representations for Data-Fitting (arXiv:2403.12206). arXiv. https://doi.org/10.48550/arXiv.2403.12206
    - Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994). Representations of quasi-Newton matrices and their use in limited memory methods. Mathematical Programming, 63(1), 129–156. https://doi.org/10.1007/BF01582063

   :param s: List holding the first set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type s: [numpy.array]
   :param y: List holding the second set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type y: [numpy.array]
   :param rho: List holding element-wise ```1/y.T@s`` from Byrd, Nocdeal & Schnabel (1992).
   :type rho: [numpy.array]
   :param H0: Initial estimate for the hessian fo the negative penalized likelihood.
   :type H0: scipy.sparse.csc_array
   :param make_psd: Whether to enforce PSD as mentioned in the description. By default set to False.
   :type make_psd: bool, optional
   :param make_pd: Whether to enforce numeric positive definiteness, not just PSD. Ignored if ``make_psd=False``. By default set to False.
   :type make_pd: bool, optional
   :param explicit: Whether or not to return the approximate matrix explicitly or implicitly in form of the three update vectors.
   :type explicit: bool
   """
   m = len(y)
   # First form S,Y, and D
   S = np.array(s).T
   Y = np.array(y).T
   
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
      # Compute implicit eigen decomposition as shown by Brust (2024)
      Q,R = scp.linalg.qr(t1,mode='economic')
      Rit2R = R@(t2)@R.T

      # ev holds non-zero eigenvalues of U@D@U.T (e.g., Brust, 2024)
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

def compute_t1_shifted_t2_t3(s,y,rho,H0,omega=1,form='Byrd'):
   """Computes the compact update to get the inverse of the negative Hessian of the penalized likelihood :math:`\mathbf{V}` from the L-BFGS-B optimizer info.

   Relies on equations 2.16 and 3.13 in Byrd, Nocdeal & Schnabel (1992) or on 2.6 in Brust (2024). Adapted here to work for the case where ``H0``:math:`=\mathbf{I}*\omega + \mathbf{S}_{\lambda}` and
   we need :math:`\mathbf{I}*\omega + \mathbf{U}\mathbf{D}\mathbf{U}^T` to be PSD. :math:`\mathbf{U}\mathbf{D}\mathbf{U}^T` is the update matrix for the
   negative Hessian of the penalized likelihood, **not** the inverse (:math:`\mathbf{V}`)! For this the implicit eigenvalue decomposition of Brust (2024) is used.

   References:
    - Brust, J. J. (2025). Useful Compact Representations for Data-Fitting (arXiv:2403.12206). arXiv. https://doi.org/10.48550/arXiv.2403.12206
    - Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994). Representations of quasi-Newton matrices and their use in limited memory methods. Mathematical Programming, 63(1), 129–156. https://doi.org/10.1007/BF01582063

   :param s: List holding the first set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type s: [numpy.array]
   :param y: List holding the second set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type y: [numpy.array]
   :param rho: List holding element-wise ```1/y.T@s`` from Byrd, Nocdeal & Schnabel (1992).
   :type rho: [numpy.array]
   :param H0: Initial estimate for the hessian fo the negative penalized likelihood. Here some multiple of the identity (multiplied by ``omega``) plus the embedded penalty matrix.
   :type H0: scipy.sparse.csc_array
   :param omega: Multiple used to get initial estimate for the hessian fo the negative penalized likelihood. Defaults to 1.
   :type omega: float, optional
   :param form: Which compact form to compute - the one from Byrd et al. (1992) or the one from Brust (2024). Defaults to "Byrd".
   :type form: float, optional
   """

   # Number of updates?
   m = len(y)

   # Now form S,Y, and D
   S = np.array(s).T
   Y = np.array(y).T

   if form == "Byrd": # Compact representation from Byrd, Nocdeal & Schnabel (1992)
   
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

      # And t1 and t3
      t1 = np.concatenate((H0@S,Y),axis=1)
      t3 = np.concatenate((S.T@H0,Y.T),axis=0)

      shifted_invt2 = -1*invt2_sort
      shifted_t2 = -1*t2_sort
   elif form == "ByrdSR1":
      # SR1 approximation      
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
      shifted_t2 = DK + L + L.T - STS
      shifted_invt2 = scp.linalg.inv(shifted_t2)

      # And terms 1 and 2
      t1 = Y - H0@S
      t3 = t1.T
   else: #  Brust (2024)
      C = S

      # Compute D & R
      D = np.identity(m)
      D[0,0] *= np.dot(s[0],y[0])

      # Now use eq. 2.5 of Brust (2024) to compute R.
      # This is the same as in Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994) essentially.
      R0 = np.dot(s[0], y[0]).reshape(1,1)
      R = R0
      for k in range(1,m):
      
         D[k,k] *= np.dot(s[k],y[k])
         
         R = np.concatenate((np.concatenate((R0,S[:,:k].T@Y[:,[k]]),axis=1),
                           np.concatenate((np.zeros((1,R0.shape[1])),
                                                np.array([1/rho[k]]).reshape(1,1)),axis=1)),axis=0)
         
         R0 = R
      
      # Compute C.T@S and extract the upper triangular part from that matrix as shown by Brust (2024)
      CTS = C.T@S
      RCS = np.triu(CTS)

      # We now need inverse of RCS
      RCS_inv = scp.linalg.solve_triangular(RCS, np.identity(m),lower=False)

      # Can now form inverse of middle block from Brust (2024)
      t2inv = np.zeros((2*m,2*m))
      t2inv[:m,:m] = (-1*RCS_inv.T) @ (R + R.T - (D + S.T@H0@S)) @ RCS_inv # Upper left
      t2inv[:m,m:] = RCS_inv.T # Upper right
      t2inv[m:,:m] = RCS_inv # Lower left
      #t2inv[m:,m:] = 0 # Lower right remains empty

      t2 = np.zeros((2*m,2*m))
      t2[:m,m:] = RCS # Upper right
      t2[m:,:m] = RCS.T # Lower left
      t2[m:,m:] = (R + R.T - (D + S.T@H0@S)) # Lower right

      # Can now compute remaining terms to compute H as shown by Brust (2024)
      t1 = np.concatenate((C,Y - H0@S),axis=1)
      t3 = np.concatenate((C.T, (Y - H0@S).T),axis=0)

      # H = H0 + t1@t2inv@t3
      shifted_invt2 = t2inv
      shifted_t2 = t2


   # We have H0 + U@D@U.T with H0 = I*omega + S_emb and U@D@U.T=t1@(-t2)@t1.T
   # Now enforce that I*omega + t1@(-t2)@t1.T is psd

   # Compute implicit eigen decomposition as shown by Brust (2024)
   Q,R = scp.linalg.qr(t1,mode='economic')
   Rit2R = R@(shifted_invt2)@R.T

   # ev holds non-zero eigenvalues of U@D@U.T (e.g., Brust, 2024)
   ev, P = scp.linalg.eigh(Rit2R,driver='ev')
   
   # Now find closest PSD.
   fix_idx = (ev + omega) <= 0
   
   if np.sum(fix_idx) > 0:
      #print("fix",np.sum(fix_idx),omega)
      ev[fix_idx] =  (-1*omega)
      
      if form != "ByrdSR1":
         ev[fix_idx] += np.power(np.finfo(float).eps,0.9)

         while np.any(np.abs(ev) < 1e-7):
            ev[np.abs(ev) < 1e-7] += np.power(np.finfo(float).eps,0.7)

      #print(f"implicit ev post shift. abs min: {np.min(np.abs(ev))}, min: {np.min(ev)}, min + 1: {np.min(ev+omega)}, max + 1: {np.max(ev+omega)}",ev[:10]+omega)

      # Now Q @ P @ np.diag(ev) @ P.T @ Q.T = shifted PSD version of U@D@U.T
      # so we can set:
      # shifted_invt2=np.diag(ev)
      # shifted_t2 = np.diag(1/ev)
      # t1 = Q @ P
      # t3 = t1.T = P.T @ Q.T
      t1 = Q @ P
      t3 = P.T@Q.T
      shifted_invt2=np.diag(ev)
      if form != "ByrdSR1":
         shifted_t2 = np.diag(1/ev)
      else:
         shifted_t2 = np.diag([1/evi if np.abs(evi) != 0 else 0 for evi in ev])

   return t1, shifted_t2, shifted_invt2, t3, 0


def compute_H_adjust_ev(s,y,rho,H0,omega=1,form='Byrd'):
   """Computes the non-zero eigenvalues of the update to get the negative Hessian of the penalized likelihood :math:`\mathcal{H}` from the L-BFGS-B optimizer info.

   Relies on equations 2.16 and 3.13 in Byrd, Nocdeal & Schnabel (1992) or on 2.6 in Brust (2024). Adapted here to work for the case where ``H0``:math:`=\mathbf{I}*\omega + \mathbf{S}_{\lambda}` and
   we need the eigenvalues for :math:`\mathbf{I}*\omega + \mathbf{U}\mathbf{D}\mathbf{U}^T`. To get those, we simply add ``omega`` to the evs of :math:`\mathbf{U}\mathbf{D}\mathbf{U}^T`

   References:
    - Brust, J. J. (2025). Useful Compact Representations for Data-Fitting (arXiv:2403.12206). arXiv. https://doi.org/10.48550/arXiv.2403.12206
    - Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994). Representations of quasi-Newton matrices and their use in limited memory methods. Mathematical Programming, 63(1), 129–156. https://doi.org/10.1007/BF01582063

   :param s: List holding the first set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type s: [numpy.array]
   :param y: List holding the second set ``m`` of update vectors from Byrd, Nocdeal & Schnabel (1992).
   :type y: [numpy.array]
   :param rho: List holding element-wise ```1/y.T@s`` from Byrd, Nocdeal & Schnabel (1992).
   :type rho: [numpy.array]
   :param H0: Initial estimate for the hessian fo the negative penalized likelihood. Here some multiple of the identity (multiplied by ``omega``) plus the embedded penalty matrix.
   :type H0: scipy.sparse.csc_array
   :param omega: Multiple used to get initial estimate for the hessian fo the negative penalized likelihood. Defaults to 1.
   :type omega: float, optional
   :param form: Which compact form to compute - the one from Byrd et al. (1992) or the one from Brust (2024). Defaults to "Byrd".
   :type form: float, optional
   """

   # Number of updates?
   m = len(y)

   # Now form S,Y, and D
   S = np.array(s).T
   Y = np.array(y).T

   if form == "Byrd": # Compact representation from Byrd, Nocdeal & Schnabel (1992)
   
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

      # And t1 and t3
      t1 = np.concatenate((H0@S,Y),axis=1)
      t3 = np.concatenate((S.T@H0,Y.T),axis=0)

      shifted_invt2 = -1*invt2_sort
      shifted_t2 = -1*t2_sort
   else: #  Brust (2024)
      C = S

      # Compute D & R
      D = np.identity(m)
      D[0,0] *= np.dot(s[0],y[0])

      # Now use eq. 2.5 of Brust (2024) to compute R.
      # This is the same as in Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994) essentially.
      R0 = np.dot(s[0], y[0]).reshape(1,1)
      R = R0
      for k in range(1,m):
      
         D[k,k] *= np.dot(s[k],y[k])
         
         R = np.concatenate((np.concatenate((R0,S[:,:k].T@Y[:,[k]]),axis=1),
                           np.concatenate((np.zeros((1,R0.shape[1])),
                                                np.array([1/rho[k]]).reshape(1,1)),axis=1)),axis=0)
         
         R0 = R
      
      # Compute C.T@S and extract the upper triangular part from that matrix as shown by Brust (2024)
      CTS = C.T@S
      RCS = np.triu(CTS)

      # We now need inverse of RCS
      RCS_inv = scp.linalg.solve_triangular(RCS, np.identity(m),lower=False)

      # Can now form inverse of middle block from Brust (2024)
      t2inv = np.zeros((2*m,2*m))
      t2inv[:m,:m] = (-1*RCS_inv.T) @ (R + R.T - (D + S.T@H0@S)) @ RCS_inv # Upper left
      t2inv[:m,m:] = RCS_inv.T # Upper right
      t2inv[m:,:m] = RCS_inv # Lower left
      #t2inv[m:,m:] = 0 # Lower right remains empty

      t2 = np.zeros((2*m,2*m))
      t2[:m,m:] = RCS # Upper right
      t2[m:,:m] = RCS.T # Lower left
      t2[m:,m:] = (R + R.T - (D + S.T@H0@S)) # Lower right

      # Can now compute remaining terms to compute H as shown by Brust (2024)
      t1 = np.concatenate((C,Y - H0@S),axis=1)
      t3 = np.concatenate((C.T, (Y - H0@S).T),axis=0)

      # H = H0 + t1@t2inv@t3
      shifted_invt2 = t2inv
      shifted_t2 = t2


   # We have H0 + U@D@U.T with H0 = I*omega + S_emb and U@D@U.T=t1@(-t2)@t1.T
   # Now enforce that I*omega + t1@(-t2)@t1.T is psd

   # Compute implicit eigen decomposition as shown by Brust (2024)
   Q,R = scp.linalg.qr(t1,mode='economic')
   Rit2R = R@(shifted_invt2)@R.T

   # ev holds non-zero eigenvalues of U@D@U.T (e.g., Brust, 2024)
   ev, P = scp.linalg.eigh(Rit2R,driver='ev')
   
   return ev + omega