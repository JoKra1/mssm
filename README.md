# mssm: Markov-switching Spline Models

## Description

``mssm`` is a toolbox to estimate Generalized Additive Mixed Models (GAMMs) semi Markov-switching GAMMs (sMs-GAMMs) and sMs Impulse Response GAMMs (sMs-IR-GAMMs). The ``main`` branch is updated frequently to reflect new developments. The ``stable`` branch should reflect the latest releases. If you don't need the newest functionality, you should install from the ``stable`` branch (see below for instructions).

## Installation

The easiest option is to install from pypi via ``pip``.

1) Setup a conda environment with python > 3.10
2) Install mssm via ``pip``

The latest release of mssm can be installed from [pypi](https://pypi.org/project/mssm/#description). So to install ``mssm`` simply run:

```
conda create -n mssm_env python=3.10
conda activate mssm_env
pip install mssm
pip install matplotlib # Only needed for tutorials
```

The fourth line, installing ``matplotlib`` is only necessary if you want to run the tutorials. Note: pypi will only reflect releases (Basically, the state of the stable branch). If you urgently need a feature currently only available on the main branch, consider building from source.

### Building from source

You can also build directly from source. This requires ``conda`` or an installation of [eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) (``setup.py`` then expects ``eigen`` in "usr/local/include/eigen3". This will probably not work on windows - the ``conda`` strategy should.). Once you have ``conda`` installed,
[install eigen from conda-forge](https://anaconda.org/conda-forge/eigen). After cloning and navigating into the downloaded repository you can then install via:

```
pip install .
```

## To get started

 - With GAMMs: Take a look at tutorial 1 in the tutorial folder.
 - With sms-IR-GAMMs: Take a look at tutorial 2.
 - With sms-GAMMs: Take a look at tutorial 3.

## Contributing

Contributions are welcome! Feel free to open issues or make pull-requests to main. Some problems that could use work are listed below.

 - Tensor smooth term setup is not very efficient and takes quite a lot of time.
 - Spline bases are always first evaluated as dense matrices followed by pruning zeros away afterwards. This is quite wasteful.
 - Fellner-Schall update computes generalized inverse for every term. This works but is not necessary. For univariate terms with a single penalty this could be based on the pre-computed rank (see Wood, 2020: *Inference and computation with generalized additive models and their extensions*) of the penalty matrix (for B-splines this is super easy to obtain).
 - Constraints. Sum-to-zero constraints (1.T @ f(x) = 0) are not so attractive from a sparse model perspective. However, that alternatives result in wider CIs is well documented (see Wood, 2017 for example). Finding a good trade-off is still high on my list, the alternatives implemented so far perform okayish.
