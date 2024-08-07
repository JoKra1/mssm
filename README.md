# mssm: Massive Sparse Smooth Models

![GitHub CI Stable](https://github.com/jokra1/mssm/actions/workflows/python-package.yml/badge.svg?branch=stable)
[![codecov](https://codecov.io/gh/JoKra1/mssm/graph/badge.svg?token=B2NZBO4XJ3)](https://codecov.io/gh/JoKra1/mssm)

## Description

``mssm`` is a toolbox to estimate Generalized Additive Mixed Models (GAMMs), Generalized Additive Mixed Models of Location Scale and Shape (GAMMLSS), and more general smooth models such as semi Markov-switching GAMMs (sMs-GAMMs; experimental) and sMs Impulse Response GAMMs (sMs-IR-GAMMs; experimental). The ``main`` branch is updated frequently to reflect new developments. The ``stable`` branch should reflect the latest releases. If you don't need the newest functionality, you should install from the ``stable`` branch (see below for instructions).

Plotting code to visualize and validate `mssm` models is provided in this [repository](https://github.com/JoKra1/mssm_tutorials) together with a tutorial for `mssm`!

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

The fourth line, installing ``matplotlib`` is only necessary if you want to run the tutorial. Note: pypi will only reflect releases (Basically, the state of the stable branch). If you urgently need a feature currently only available on the main branch, consider building from source.

### Building from source

You can also build directly from source. This requires ``conda`` or an installation of [eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) (``setup.py`` then expects ``eigen`` in "usr/local/include/eigen3". This will probably not work on windows - the ``conda`` strategy should.). Once you have ``conda`` installed,
[install eigen from conda-forge](https://anaconda.org/conda-forge/eigen). After cloning and navigating into the downloaded repository you can then install via:

```
pip install .
```

## Contributing

Contributions are welcome! Feel free to open issues or make pull-requests to main.
