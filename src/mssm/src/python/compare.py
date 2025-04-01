import numpy as np
import scipy as scp
import math
from ...models import GAMM,GAMMLSS,GSMM,Family,GAMLSSFamily,GENSMOOTHFamily,Gaussian,Identity
from .utils import correct_VB
import warnings

def compare_CDL(model1:GAMM or GAMMLSS or GSMM,
                model2:GAMM or GAMMLSS or GSMM,
                correct_V:bool=True,
                correct_t1:bool=False,
                perform_GLRT:bool=False,
                nR=250,
                n_c=10,
                alpha=0.05,
                grid='JJJ2',
                a=1e-7,b=1e7,df=40,
                verbose=False,
                drop_NA=True,
                method="Chol",
                seed=None,
                use_upper=True,
                Vp_fidiff=True,
                use_reml_weights=True,
                **bfgs_options):
    
    """ Computes the AIC difference and (optionally) performs an approximate GLRT on twice the difference in unpenalized likelihood between models ``model1`` and ``model2`` (see Wood et al., 2016).
    
    For the GLRT to be appropriate ``model1`` should be set to the model containing more effects and ``model2`` should be a nested, simpler, variant of ``model1``.
    For the degrees of freedom for the test, the expected degrees of freedom (EDF) of each model are used (i.e., this is the conditional test discussed in Wood (2017: 6.12.4)).
    The difference between the models in EDF serves as DoF for computing the Chi-Square statistic. In addition, ``correct_t1`` should be set to True, when computing the GLRT.
    
    To get the AIC for each model, 2*edf is added to twice the negative (conditional) likelihood (see Wood et al., 2016).
    
    By default (``correct_V=True``), ``mssm`` will attempt to correct the edf for uncertainty in the estimated :math:`\lambda` parameters. Which correction is computed depends on the
    choice for the ``grid`` argument. **Approximately** the analytic solution for the correction proposed by Wood, Pya, & Säfken (2016) is computed  when ``grid='JJJ1'`` - which is exact for
    strictly Gaussian and canonical Generalized additive models. This is not efficient for sparse models and will thus become too expensive for large sparse multi-level models.
    When setting ``grid='JJJ2'`` (default), the analytic solution proposed by Wood, Pya, & Säfken (2016) is used as a basis for the correction and subsequently refined via numeric integration. This requires a costly sampling step
    (see Greven & Scheipl, 2016 and the ``correct_VB`` function in the utils module) which will take quite some time for reasonably large models with more than 3-4 smoothing parameters, but remains
    efficient for sparse multi-level models (assuming ``use_upper`` is set to True and ``correct_t1`` is set to False).

    In case either form of correction is too expensive, it might be better to rely on hypothesis tests for individual smooths, confidence intervals, and penalty-based selection approaches instead (see Marra & Wood, 2011 for details on the latter).

    In case ``correct_t1=True`` the EDF will be set to the (smoothness uncertainty corrected in case ``correct_V=True``) smoothness bias corrected exprected degrees of freedom (t1 in section 6.1.2 of Wood, 2017),
    for the GLRT (based on recomendation given in section 6.12.4 in Wood, 2017). The AIC (Wood, 2017) of both models will **still be based on the regular (smoothness uncertainty corrected) edf**.

    The computation here is different to the one performed by the ``compareML`` function in the R-package ``itsadug`` - which rather performs a version of the marginal GLRT
    (also discussed in Wood, 2017: 6.12.4) - and more similar to the ``anova.gam`` implementation provided by ``mgcv`` (if ``grid='JJJ1'). The returned p-value is approximate - very **very**
    much so if ``correct_V=False``. Also, the GLRT should **no**t be used to compare models differing in their random effect structures - the AIC is more appropriate for this (see Wood, 2017: 6.12.4).

    References:
     - Marra, G., & Wood, S. N. (2011) Practical variable selection for generalized additive models.
     - Wood, S. N., Pya, N., Saefken, B., (2016). Smoothing Parameter and Model Selection for General Smooth Models
     - Greven, S., & Scheipl, F. (2016). Comment on: Smoothing Parameter and Model Selection for General Smooth Models
     - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, Second Edition (2nd ed.).
     - ``compareML`` function from ``itsadug`` R-package: https://rdrr.io/cran/itsadug/man/compareML.html
     - ``anova.gam`` function from ``mgcv``, see: https://www.rdocumentation.org/packages/mgcv/versions/1.9-1/topics/anova.gam

    :param model1: GAMM, GAMMLSS, or GSMM 1.
    :type model1: GAMM or GAMMLSS or GSMM
    :param model2: GAMM, GAMMLSS, or GSMM 2.
    :type model2: GAMM or GAMMLSS or GSMM
    :param correct_V: Whether or not to correct for smoothness uncertainty. Defaults to True
    :type correct_V: bool, optional
    :param correct_t1: Whether or not to also correct the smoothness bias corrected edf for smoothness uncertainty. Defaults to True.
    :type correct_t1: bool, optional
    :param perform_GLRT: Whether to perform both a GLRT and to compute the AIC or to only compute the AIC. Defaults to True.
    :type perform_GLRT: bool, optional
    :param nR: In case ``grid!="JJJ1"``, ``nR`` samples/reml scores are generated/computed to numerically evaluate the expectations necessary for the uncertainty correction, defaults to 250
    :type nR: int, optional
    :param n_c: Number of cores to use to compute the smoothness uncertaincy correction, defaults to 10
    :type n_c: int, optional
    :param alpha: alpha level of the GLRT. Defaults to 0.05
    :type alpha: float, optional
    :param grid: How to compute the smoothness uncertainty correction, defaults to 'JJJ2'
    :type grid: str, optional
    :param a: Minimum :math:`\lambda` value that is included when correcting for uncertainty and ``grid!='JJJ1'``. In addition, any of the :math:`\lambda` estimates obtained from ``model`` (used to define the mean for the posterior of :math:`\\boldsymbol{p}|y \sim N(log(\hat{\\boldsymbol{p}}),\mathbf{V}_{\\boldsymbol{p}})`) which are smaller than this are set to this value as well, defaults to 1e-7 the minimum possible estimate
    :type a: float, optional
    :param b: Maximum :math:`\lambda` value that is included when correcting for uncertainty and ``grid!='JJJ1'``. In addition, any of the :math:`\lambda` estimates obtained from ``model`` (used to define the mean for the posterior of :math:`\\boldsymbol{p}|y \sim N(log(\hat{\\boldsymbol{p}}),\mathbf{V}_{\\boldsymbol{p}})`) which are larger than this are set to this value as well, defaults to 1e7 the maximum possible estimate
    :type b: float, optional
    :param df: Degrees of freedom used for the multivariate t distribution used to sample the next set of candidates. Setting this to ``np.inf`` means a multivariate normal is used for sampling, defaults to 40
    :type df: int, optional
    :param verbose: Whether to print progress information or not, defaults to False
    :type verbose: bool, optional
    :param drop_NA: Whether to drop rows in the **model matrices** corresponding to NAs in the dependent variable vector. Defaults to True.
    :type drop_NA: bool,optional
    :param method: Which method to use to solve for the coefficients (and smoothing parameters). The default ("Chol") relies on Cholesky decomposition. This is extremely efficient but in principle less stable, numerically speaking. For a maximum of numerical stability set this to "QR/Chol". In that case a QR decomposition is used - which is first pivoted to maximize sparsity in the resulting decomposition but also pivots for stability in order to get an estimate of rank defficiency. A Cholesky is than used using the combined pivoting strategy obtained from the QR. This takes substantially longer. If this is set to ``'qEFS'``, then the coefficients are estimated via quasi netwon and the smoothing penalties are estimated from the quasi newton approximation to the hessian. This only requieres first derviative information. Defaults to "Chol".
    :type method: str,optional
    :param seed: Seed to use for random parts of the correction. Defaults to None
    :type seed: int,optional
    :param use_upper: Whether to compute edf. from trace of covariance matrix (``use_upper=False``) or based on numeric integration weights. The latter is much more efficient for sparse models. Only makes sense when ``grid='JJJ2'``. Defaults to True
    :type use_upper: bool,optional
    :param Vp_fidiff: Whether to rely on a finite difference approximation to compute :math:`\mathbf{V}_{\\boldsymbol{p}}` for grid ``JJJ1`` and ``JJJ2`` or on a PQL approximation. The latter is exact for Gaussian and canonical GAMs and far cheaper if many penalties are to be estimated. Defaults to True (Finite difference approximation)
    :type Vp_fidiff: bool,optional
    :param use_reml_weights: Whether to rely on REMl scores to compute the numerical integration when ``grid_type != 'JJJ1'`` or on the log-densities of :math:`\mathbf{V}_{\\boldsymbol{p}}` - the latter assumes that the unconditional posterior is normal. Defaults to True (REML scores are used)
    :type use_reml_weights: bool,optional
    :param bfgs_options: Any additional keyword arguments that should be passed on to the call of :func:`scipy.optimize.minimize`. If none are provided, the ``gtol`` argument will be initialized to 1e-3. Note also, that in any case the ``maxiter`` argument is automatically set to 100. Defaults to None.
    :type bfgs_options: key=value,optional
    :raises ValueError: If both models are from different families.
    :raises ValueError: If ``perform_GLRT=True`` and ``model1`` has fewer coef than ``model2`` - i.e., ``model1`` has to be the notationally more complex one.
    :return: A dictionary with outcomes of all tests. Key ``H1`` will be a bool indicating whether Null hypothesis was rejected or not, ``p`` will be the p-value, ``chi^2`` will be the test statistic used, ``Res. DOF`` will be the degrees of freedom used by the test, ``aic1`` and ``aic2`` will be the aic scores for both models.
    :rtype: dict
    """

    if type(model1.family) != type(model2.family):
        raise ValueError("Both models should be estimated using the same family.")
    
    if perform_GLRT and isinstance(model1.family,Family) and model1.formula.n_coef < model2.formula.n_coef:
        raise ValueError("For the GLRT, model1 needs to be set to the more complex model (i.e., needs to have more coefficients than model2).")
    
    if perform_GLRT and (isinstance(model1.family,Family) == False) and len(model1.overall_coef) < len(model2.overall_coef):
        raise ValueError("For the GLRT, model1 needs to be set to the more complex model (i.e., needs to have more coefficients than model2).")
    
    # Collect total DOF for uncertainty in \lambda using correction proposed by Greven & Scheipl (2016)
    aic_DOF1 = None
    aic_DOF2 = None
    if correct_V:
        if verbose:
            print("Correcting for uncertainty in lambda estimates...\n")
        
        #V,LV,Vp,Vpr,edf,total_edf,edf2,total_edf2,upper_edf
        _,_,_,_,_,DOF1,_,DOF12,upper_edf1 = correct_VB(model1,nR=nR,n_c=n_c,form_t1=correct_t1,grid_type=grid,a=a,b=b,df=df,verbose=verbose,drop_NA=drop_NA,method=method,only_expected_edf=(use_upper and (correct_t1==False)),Vp_fidiff=Vp_fidiff,use_reml_weights=use_reml_weights,seed=seed,**bfgs_options)
        _,_,_,_,_,DOF2,_,DOF22,upper_edf2 = correct_VB(model2,nR=nR,n_c=n_c,form_t1=correct_t1,grid_type=grid,a=a,b=b,df=df,verbose=verbose,drop_NA=drop_NA,method=method,only_expected_edf=(use_upper and (correct_t1==False)),Vp_fidiff=Vp_fidiff,use_reml_weights=use_reml_weights,seed=seed,**bfgs_options)
        
        if use_upper:
            DOF1 = upper_edf1
            DOF2 = upper_edf2
        
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
            # Compute uncertainty un-corrected but smoothness bias corrected edf (t1 in section 6.1.2 of Wood, 2017)
            if isinstance(model1.family,Family):
                X1 = model1.get_mmat()
                X2 = model2.get_mmat()
                V1 = model1.lvi.T @ model1.lvi
                V2 = model2.lvi.T @ model2.lvi
                if isinstance(model1.family,Gaussian) and isinstance(model1.family.link,Identity): # Strictly additive case
                    F1 = V1@(X1.T@X1)
                    F2 = V2@(X2.T@X2)
                else: # Generalized case
                    W1 = model1.Wr@model1.Wr
                    W2 = model2.Wr@model2.Wr
                    F1 = V1@(X1.T@W1@X1)
                    F2 = V2@(X2.T@W2@X2)
            else: # GAMLSS or GSMM case
                V1 = model1.overall_lvi.T @ model1.overall_lvi
                V2 = model2.overall_lvi.T @ model2.overall_lvi
                F1 = V1@(-1*model1.hessian)
                F2 = V2@(-1*model2.hessian)

            DOF12 = 2*DOF1 - (F1@F1).trace()
            DOF22 = 2*DOF2 - (F2@F2).trace()

            aic_DOF1 = DOF1
            aic_DOF2 = DOF2
            DOF1 = DOF12
            DOF2 = DOF22


    # Compute un-penalized likelihood based on scale estimate of more complex (in terms of edf - so actually more complex) model if a scale was estimated (see section 3.1.4, Wood, 2017).
    if isinstance(model1.family,Family):
        ext_scale = None
        if model1.family.twopar:
            if DOF1 > DOF2:
                _,ext_scale = model1.get_pars()
            else:
                _,ext_scale = model2.get_pars()

        llk1 = model1.get_llk(penalized=False,ext_scale=ext_scale)
        llk2 = model2.get_llk(penalized=False,ext_scale=ext_scale)
    else:
        llk1 = model1.get_llk(penalized=False)
        llk2 = model2.get_llk(penalized=False)

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
              "DOF12":aic_DOF1,
              "DOF22":aic_DOF2,
              "Res. DOF":DOF_diff,
              "aic1":aic1,
              "aic2":aic2,
              "aic_diff":aic1-aic2}
    
    return result
