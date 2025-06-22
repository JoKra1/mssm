.. mssm documentation master file, created by
   sphinx-quickstart on Fri Sep 27 15:37:49 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

mssm documentation
==================

This is the documentation of the **public** API of ``mssm``. The entire source code is available for inspection on `GitHub <https://github.com/JoKra1/mssm>`_.
For the purpose of **applying** the models available in this toolbox, the documentation here should however be sufficient.

``mssm`` is a Python toolbox to estimate Generalized Additive Mixed Models (**GAMMs**), Generalized Additive Mixed Models of Location Scale
and Shape (**GAMMLSS**), and more general smooth (mixed) models (**GSMMs**) in the sense defined by `Wood, Pya, & Säfken (2016) <https://doi.org/10.1080/01621459.2016.1180986>`_.
mssm is an excellent choice for the modeling of multi-level time-series data, often estimating additive models with separate smooths for thousands of levels in a couple of minutes.

Use the side-bar on the left to navigate through the document tree (just click on ``src`` and it will expand).

Installation
----------------

The latest stable release of mssm can be installed from `pypi <https://pypi.org/project/mssm/\#description>`_. So to complete both steps (after installing ``conda`` - see `here <https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_ for instructions), simply run::


   conda create -n mssm_env python=3.13
   conda activate mssm_env
   pip install mssm mssmViz # 'mssmViz' only needed for plotting


For more detailed instructions see the `README <https://github.com/JoKra1/mssm>`_.

Supported Models
----------------

**GAMMs**:

- Gaussian (G)AMMs (:func:`mssm.src.python.exp_fam.Gaussian`)
- Gamma GAMMs (:func:`mssm.src.python.exp_fam.Gamma`)
- Inverse Gaussian GAMMs (:func:`mssm.src.python.exp_fam.InvGauss`)
- Binomial GAMMs (:func:`mssm.src.python.exp_fam.Binomial`)
- Poisson GAMMs (:func:`mssm.src.python.exp_fam.Poisson`)

If you are missing a family, don't worry - all you need to do is to implement the :func:`mssm.src.python.exp_fam.Family` template class for **GAMM** models.

**GAMMLSS**:

- Gaussian models of location and scale (:func:`mssm.src.python.exp_fam.GAUMLSS`)
- Gamma models of location and scale (:func:`mssm.src.python.exp_fam.GAMMALS`)
- Multionimal models (:func:`mssm.src.python.exp_fam.MULNOMLSS`)

To implement a new familiy for a **GAMMLSS** model requires implementing the :func:`mssm.src.python.exp_fam.GAMLSSFamily` template class.

**GSMMs**

- Cox proportional Hazard models (:func:`mssm.src.python.exp_fam.PropHaz`)

To implement a new  member of the most general kind of smooth model, you will only need to implement the :func:`mssm.src.python.exp_fam.GENSMOOTHFamily` template class - ``mssm`` even supports completely derivative-free estimation. You can check the :class:`mssm.models.GSMM` documentation for an example!

Supported Terms
----------------

If you want to know which (smooth, parametric, & random) terms are supported by these models, you should take a look at the :class:`mssm.src.python.formula.Formula` class and the terms implemented in :py:mod:`mssm.src.python.terms`.

Tutorial & Model Visualization
------------------------------

To get started with ``mssm`` the tutorial and visualization code available as part of the `mssmViz <https://github.com/JoKra1/mssm_tutorials>`_ package might also be helpful. ``mssmViz`` for example offers functions to visualize predictions & residuals for
GAMM & GAMMLSS.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules