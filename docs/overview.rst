Getting Started
==================

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

To implement a new  member of the most general kind of smooth model, you will only need to implement the :func:`mssm.src.python.exp_fam.GSMMFamily` template class - ``mssm`` even supports completely derivative-free estimation. You can check the :class:`mssm.models.GSMM` documentation for an example!

Supported Terms
----------------

If you want to know which (smooth, parametric, & random) terms are supported by these models, you should take a look at the :class:`mssm.src.python.formula.Formula` class and the terms implemented in :py:mod:`mssm.src.python.terms`.

Tutorial & Model Visualization
------------------------------

To get started with ``mssm`` the tutorial and visualization code available as part of the `mssmViz <https://github.com/JoKra1/mssm_tutorials>`_ package might also be helpful. ``mssmViz`` for example offers functions to visualize predictions & residuals for
GAMM & GAMMLSS.