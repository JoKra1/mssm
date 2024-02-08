import numpy as np
import scipy as scp
import math
from ...models import GAMM
import warnings

def GLRT_CDL(model1:GAMM,
            model2:GAMM,
            alpha=0.05):
    
    """
    Performs an approximate GLRT on twice the difference in unpenalized likelihood between the models. For the degrees of freedom the expected degrees of freedom (EDF) of each
    model are used (i.e., this is the conditional test discussed in Wood (2017: 6.12.4)). The difference between the models in EDF serves as DoF for computing the Chi-Square statistic.

    The computation here is different to the one performed by the ``compareML`` function in the R-package ``itsadug`` - which rather performs a version of the marginal GLRT
    (also discussed in Wood, 2017: 6.12.4). The p-value is very **very** much approximate. Even more so than when using for example ``anova()`` in R to perform this test. The reason
    is that the lambda uncertainty correction applied by mgcv can not be obtained by ``mssm``. Also, the test should not be used to compare models differing in their random effect structures,
    (see Wood, 2017: 6.12.4) for details on those two points.

    References:
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
     - ``compareML`` function from ``itsadug`` R-package: https://rdrr.io/cran/itsadug/man/compareML.html
    """

    if type(model1.family) != type(model2.family):
        raise ValueError("Both models should be estimated using the same family.")
    
    # Collect total DOF
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
