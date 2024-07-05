import numpy as np
import scipy as scp
import math
from ...models import GAMM
from .utils import correct_VB
import warnings

def GLRT_CDL(model1:GAMM,
            model2:GAMM,
            correct_V:bool=True,
            lR=20,
            nR=5,
            alpha=0.05):
    
    """
    Performs an approximate GLRT on twice the difference in unpenalized likelihood between the models. For the degrees of freedom the expected degrees of freedom (EDF) of each
    model are used (i.e., this is the conditional test discussed in Wood (2017: 6.12.4)). By default (``correct_V=True``), ``mssm`` will attempt to correct the edf for uncertainty
    in the estimated \lambda parameters. This requires computing a costly correction (see Greven & Scheipl, 2016 and the ``correct_VB`` function in the utils module) which will
    take quite some time for reasonably large models with more than 3-4 smoothing parameters. In that case relying on CIs and penalty-based comparisons might be preferable
    (see Marra & Wood, 2011 for details on the latter). The difference between the models in EDF serves as DoF for computing the Chi-Square statistic.

    The computation here is different to the one performed by the ``compareML`` function in the R-package ``itsadug`` - which rather performs a version of the marginal GLRT
    (also discussed in Wood, 2017: 6.12.4). The p-value is very **very** much approximate. Both tests should not be used to compare models differing in their random effect structures,
    (see Wood, 2017: 6.12.4).

    References:
     - Marra, G., & Wood, S. N. (2011) Practical variable selection for generalized additive models.
     - Greven, S., & Scheipl, F. (2016). Comment on: Smoothing Parameter and Model Selection for General Smooth Models
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
     - ``compareML`` function from ``itsadug`` R-package: https://rdrr.io/cran/itsadug/man/compareML.html
    """

    if type(model1.family) != type(model2.family):
        raise ValueError("Both models should be estimated using the same family.")
    
    # Collect total DOF for uncertainty in \lambda using correction proposed by Greven & Scheipl (2016)
    if correct_V:
        print("Correcting for uncertainty in lambda estimates...\n")
        _,_,DOF1 = correct_VB(model1,nR=nR,lR=lR)
        _,_,DOF2 = correct_VB(model2,nR=nR,lR=lR)
    else:
        DOF1 = model1.edf
        DOF2 = model2.edf

    # Compute un-penalized likelihood()
    llk1 = model1.get_llk(False)
    llk2 = model2.get_llk(False)
    
    if DOF1 < DOF2:
        # Re-order, making sure that more complex model is 1
        llk_tmp = llk1
        DOF_tmp = DOF1
        llk1 = llk2
        llk2 = llk_tmp
        DOF1 = DOF2
        DOF2 = DOF_tmp

    # Compute Chi-square statistic
    stat = 2 * (llk1 - llk2)

    if DOF1-DOF2 < 1:
        warnings.warn("Difference in EDF is extremely small. Enforcing a minimum of 1 for the DOF of the CHI^2 distribution...")
    
    # Compute p-value under reference distribution.
    # scipy seems to handle non-integer DOF quite well, so I won't bother rounding here.
    p = 1 - scp.stats.chi2.cdf(stat,max(DOF1-DOF2,1))

    # Reject NULL?
    H1 = p <= alpha

    return H1,p,stat,DOF1,DOF2
