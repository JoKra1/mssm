mssm Documentation
==================

This is the documentation of the **public** API of ``mssm``. The entire source code is available for inspection on `GitHub <https://github.com/JoKra1/mssm>`_.

``mssm`` is a Python toolbox to estimate Generalized Additive Mixed Models (**GAMMs**), Generalized Additive Mixed Models of Location Scale
and Shape (**GAMMLSS**), and more general smooth (mixed) models (**GSMMs**) in the sense defined by `Wood, Pya, & Säfken (2016) <https://doi.org/10.1080/01621459.2016.1180986>`_.
``mssm`` is an excellent choice for the modeling of multi-level time-series data, often estimating additive models with separate smooths for thousands of levels in a couple of minutes.

You can either use the side-bar on the left to navigate through the document tree or click on the links below. To get started, we suggest that you first work through the `Getting Started`
section, which includes installation instructions. Then you should complete the `Tutorial`, in which you will familiarize yourself with the syntax of ``mssm`` and the `mssmViz <https://github.com/JoKra1/mssmViz>`_ package. ``mssmViz``
offers functions to visualize & validate models estimated via the ``mssm`` toolbox. If you are interested in adding your own custom smooth basis and penalty matrices, head to the `Programmer's Guide` section.
If you just want to take a look at the api, check out the `api` section.

Getting Started
==================
.. toctree::
   :maxdepth: 2

   overview

Tutorial
==================
.. toctree::
   :maxdepth: 2

   tutorial

Programmer's Guide
==================
.. toctree::
   :maxdepth: 2

   guide

api
==================
.. toctree::
   :maxdepth: 3

   api