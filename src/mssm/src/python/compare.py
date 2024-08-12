import numpy as np
import scipy as scp
import math
from ...models import GAMM
from .utils import correct_VB
import warnings

def compare_CDL(model1:GAMM,
                model2:GAMM,
                correct_V:bool=True,
                correct_t1:bool=True,
                perform_GLRT:bool=True,
                lR=20,
                nR=5,
                n_c=10,
                alpha=0.05,
                grid='JJJ',
                verbose=False):
    
    """(Optionally) performs an approximate GLRT on twice the difference in unpenalized likelihood between ``model1`` and ``model2`` (see Wood, 2017).
    
    Also computes the AIC difference (see Wood et al., 2016). For the GLRT to be appropriate model1 should be set to the model containing more effects and model2 should be a nested, simpler, variant of model1.
    
    For the degrees of freedom for the test, the expected degrees of freedom (EDF) of each model are used (i.e., this is the conditional test discussed in Wood (2017: 6.12.4)).
    The difference between the models in EDF serves as DoF for computing the Chi-Square statistic. Similarly, for each model 2*edf is added to twice the negative (conditional) likelihood to
    compute the aic (see Wood et al., 2016).
    
    By default (``correct_V=True``), ``mssm`` will attempt to correct the edf for uncertainty in the estimated \lambda parameters. This requires computing a costly
    correction (see Greven & Scheipl, 2016 and the ``correct_VB`` function in the utils module) which will take quite some time for reasonably large models with more than 3-4 smoothing parameters.
    In that case relying on CIs and penalty-based comparisons might be preferable (see Marra & Wood, 2011 for details on the latter).

    In case ``correct_t1=True`` the EDF will be set to the (smoothness uncertainty corrected in case ``correct_V=True``) and smoothness bias corrected exprected degrees of freedom (t1 in section 6.1.2 of Wood, 2017),
    for the GLRT (based on reccomendation given in section 6.12.4 in Wood, 2017). The AIC (Wood, 2017) of both models will still be based on the regular (smoothness uncertainty corrected) edf.

    The computation here is different to the one performed by the ``compareML`` function in the R-package ``itsadug`` - which rather performs a version of the marginal GLRT
    (also discussed in Wood, 2017: 6.12.4). The p-value is approximate - very **very** much so if ``correct_V=False`` and the test should not be used to compare models differing in their random effect structures
    (see Wood, 2017: 6.12.4).

    References:
    
     - Marra, G., & Wood, S. N. (2011) Practical variable selection for generalized additive models.
     - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models
     - Greven, S., & Scheipl, F. (2016). Comment on: Smoothing Parameter and Model Selection for General Smooth Models
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
     - ``compareML`` function from ``itsadug`` R-package: https://rdrr.io/cran/itsadug/man/compareML.html
     - ``anova.gam`` function from ``mgcv``, see: https://www.rdocumentation.org/packages/mgcv/versions/1.9-1/topics/anova.gam

    :param model1: GAMM model 1.
    :type model1: GAMM
    :param model2: GAMM model 2.
    :type model2: GAMM
    :param alpha: alpha level of the GLRT.
    :type model1: GAMM
    :raises ValueError: If both models are from different families.
    :raises ValueError: If ``perform_GLRT=True`` and ``model1`` has fewer coef than ``model2`` - i.e., ``model1`` has to be the notationally more complex one.
    :return: A dictionary with outcomes of all tests. Key ``H1`` will be a bool indicating whether Null hypothesis was rejected or not, ``p`` will be the p-value, ``chi^2`` will be the test statistic used,
    ``Res. DOF`` will be the degrees of freedom used by the test, ``aic1`` and ``aic2`` will be the aic scores for both models.
    :rtype: dict
    """

    if type(model1.family) != type(model2.family):
        raise ValueError("Both models should be estimated using the same family.")
    
    if perform_GLRT and model1.formula.n_coef < model2.formula.n_coef:
        raise ValueError("For the GLRT, model1 needs to be set to the more complex model (i.e., needs to have more coefficients than model2).")
    
    # Collect total DOF for uncertainty in \lambda using correction proposed by Greven & Scheipl (2016)
    if correct_V:
        if verbose:
            print("Correcting for uncertainty in lambda estimates...\n")
        _,_,DOF1,DOF12,expected_aic1 = correct_VB(model1,nR=nR,lR=lR,n_c=n_c,form_t1=correct_t1,grid_type=grid,verbose=verbose)
        _,_,DOF2,DOF22,expected_aic2 = correct_VB(model2,nR=nR,lR=lR,n_c=n_c,form_t1=correct_t1,grid_type=grid,verbose=verbose)
        
        if correct_t1:
            # Section 6.12.4 suggests replacing t (edf) with t1 (2*t - (F@F).trace()) with F=(X.T@X+S_\llambda)^{-1}@X.T@X for GLRT - with the latter also being corrected for
            # uncertainty in lambda. However, Wood et al., (2016) suggest that the aic should be computed based on t - so some book-keeping is ncessary.
            aic_DOF1 = DOF1
            aic_DOF2 = DOF2
            DOF1 = DOF12
            DOF2 = DOF22

    else:
        DOF1 = model1.edf
        DOF2 = model2.edf
        
        if correct_t1:
            # Compute uncertainty uncorrected but smoothness bias corrected edf (t1 in section 6.1.2 of Wood, 2017)
            X1 = model1.get_mmat()
            X2 = model2.get_mmat()
            V1 = model1.lvi.T @ model1.lvi
            V2 = model2.lvi.T @ model2.lvi
            F1 = V1@(X1.T@X1)
            F2 = V2@(X2.T@X2)
            DOF12 = 2*DOF1 - (F1@F1).trace()
            DOF22 = 2*DOF2 - (F2@F2).trace()

            aic_DOF1 = DOF1
            aic_DOF2 = DOF2
            DOF1 = DOF12
            DOF2 = DOF22


    # Compute un-penalized likelihood based on scale estimate of more complex (in terms of edf - so actually more complex) model if a scale was estimated (see section 3.1.4, Wood, 2017).
    ext_scale = None
    if model1.family.twopar:
        if DOF1 > DOF2:
            _,ext_scale = model1.get_pars()
        else:
            _,ext_scale = model2.get_pars()

    llk1 = model1.get_llk(penalized=False,ext_scale=ext_scale)
    llk2 = model2.get_llk(penalized=False,ext_scale=ext_scale)

    # Compute Chi-square statistic...
    stat = 2 * (llk1 - llk2)
    test_stat = stat
    
    # ... and degrees of freedom under NULL (see Wood, 2017)
    DOF_diff = DOF1-DOF2
    test_DOF_diff = abs(DOF_diff)

    # Multiple scenarios that this test needs to cover...
    # 1) LLK1 < LLK2, DOF1 < DOF2; This is a valid test, essentially model2 turns out to be the more complicated one.
    # 2) LLK1 < LLK2, DOF1 > DOF2; This makes no sense. Model 1 - the more complex one - has worse llk but more DOF.
    # 3) LLK1 > LLK2, DOF1 < DOF2; Notationally correct: model1 should after all be more complex. But in terms of edf makes little sense (as pointed out by Wood, 2017).
    # 4) LLK1 > LLK2, DOF1 > DOF2; Valid, inverse of case 1.
    
    # Personally, I think cases 2 & 3 should both return NAs for p-values.. But anova.gam for mgcv returns a p-value for case 3 so we will do the same here
    # and just raise a warning. For case 1, we need to take -1*test_stat.
    if  llk1 < llk2 and DOF1 < DOF2:
        test_stat = -1*test_stat
    
    # Compute p-value under reference distribution.
    if perform_GLRT == False or test_stat < 0: # Correct for aforementioned possibility 2: model 1 has lower llk and higher edf.
        H1 = np.nan
        p = np.nan
    else:
        if llk1 > llk2 and DOF1 < DOF2:
            warnings.warn("Model with more coefficients has higher likelihood but lower expected degrees of freedom. Interpret results with caution.")

        p = 1 - scp.stats.chi2.cdf(test_stat,test_DOF_diff)

        # Reject NULL?
        H1 = p <= alpha

    # Also correct AIC for GAM (see Wood et al., 2017)
    if correct_t1:
        aic1 = -2*llk1 + 2*aic_DOF1
        aic2 = -2*llk2 + 2*aic_DOF2
    else:
        aic1 = -2*llk1 + 2*DOF1
        aic2 = -2*llk2 + 2*DOF2

    result = {"H1":H1,
              "p":p,
              "chi^2":stat,
              "DOF1":DOF1,
              "DOF2":DOF2,
              "Res. DOF":DOF_diff,
              "aic1":aic1,
              "aic2":aic2,
              "aic_diff":aic1-aic2}
    
    return result
