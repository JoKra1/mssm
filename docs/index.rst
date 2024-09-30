.. mssm documentation master file, created by
   sphinx-quickstart on Fri Sep 27 15:37:49 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

mssm documentation
==================

This is the documentation for the **exposed** API of ``mssm``. The source code for internal functions is available on `GitHub <https://github.com/JoKra1/mssm>`_ and can be
consulted to understand their implementation. ``mssm`` is a toolbox to estimate Generalized Additive Mixed Models (GAMMs), Generalized Additive Mixed Models of Location Scale
and Shape (GAMMLSS), and more general (mixed) smooth models in the sense defined by `Wood, Pya, & Säfken (2016) <https://doi.org/10.1080/01621459.2016.1180986>`_.
mssm is an excellent choice for the modeling of multi-level time-series data, often estimating additive models with separate smooths for thousands of levels in a couple of minutes.

Currently, ``mssm`` supports Gaussian (G)AMMs (:func:`mssm.src.python.exp_fam.Gaussian`), Gamma GAMMs (:func:`mssm.src.python.exp_fam.Gamma`), and Binomial GAMMs (:func:`mssm.src.python.exp_fam.Binomial`).
For GAMMLSS models, Gaussian models (:func:`mssm.src.python.exp_fam.GAUMLSS`), Gamma models (:func:`mssm.src.python.exp_fam.GAMMALS`), and Multionimal models (:func:`mssm.src.python.exp_fam.MULNOMLSS`) are supported.

If you are missing a family, don't worry - all you need to do is to implement the :func:`mssm.src.python.exp_fam.Family` template class for GAMM models and the :func:`mssm.src.python.exp_fam.GAMLSSFamily` template class for GAMMLSS models.
To estimate the most general kind of smooth model, you only have to implement the :func:`mssm.src.python.exp_fam.GENSMOOTHFamily` template class.

To get started with ``mssm`` the tutorial and visualization code available `here <https://github.com/JoKra1/mssm_tutorials>`_ might also be helpful.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules