# mssm: Mixed Sparse Smooth Models

![GitHub CI Stable](https://github.com/jokra1/mssm/actions/workflows/python-package.yml/badge.svg?branch=stable)
![Docs](https://github.com/jokra1/mssm/actions/workflows/documentation.yml/badge.svg?branch=main)
[![codecov](https://codecov.io/gh/JoKra1/mssm/graph/badge.svg?token=B2NZBO4XJ3)](https://codecov.io/gh/JoKra1/mssm)
![preprint](https://img.shields.io/badge/preprint-arXiv%3A2506.13132-orange?link=https%3A%2F%2Farxiv.org%2Fabs%2F2506.13132)

## Description

> [!NOTE]
> Our preprint detailing the algorithms implemented in the mssm toolbox is now available on [arXiv](https://arxiv.org/abs/2506.13132).

``mssm`` is a toolbox to estimate Generalized Additive Mixed Models (GAMMs), Generalized Additive Mixed Models of Location Scale and Shape (GAMMLSS), and more general (mixed) smooth models in the sense defined by [Wood, Pya, & Säfken (2016)](https://doi.org/10.1080/01621459.2016.1180986). Approximate estimation (and automatic regularization) of the latter only requires users to provide the (gradient of) the log-likelihood. Furthermore, ``mssm`` is an excellent choice for the modeling of multi-level time-series data, often estimating additive models with separate smooths for thousands of levels in a couple of minutes.

**Note**: The ``main`` branch is updated frequently to reflect new developments. The ``stable`` branch should reflect the latest releases. If you don't need the newest functionality, you should install from the ``stable`` branch (see below for instructions). **Documentation** is hosted [here](https://jokra1.github.io/mssm/index.html) - together with a tutorial for `mssm`! Plotting code to visualize and validate `mssm` models is provided in this [repository](https://github.com/JoKra1/mssmViz)!

## Installation

The easiest option is to install from pypi via ``pip``. This can be achieved in two steps:

1) Setup a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) with python > 3.10
2) Install mssm via ``pip``

The latest release of mssm can be installed from [pypi](https://pypi.org/project/mssm/#description). So to complete both steps (after installing ``conda`` - see [here](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) for instructions), simply run:

```
conda create -n mssm_env python=3.13
conda activate mssm_env
pip install mssm mssmViz==0.1.48 # 'mssmViz' only needed for plotting
```

**Note**: pypi will only reflect releases (Basically, the state of the stable branch). If you urgently need a feature currently only available on the main branch, consider building from source.

### Building from source

You can also build directly from source. This requires ``conda`` or an installation of [eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) (``setup.py`` then expects ``eigen`` in "usr/local/include/eigen3". This will probably not work on windows - the ``conda`` strategy should.). Once you have ``conda`` installed,
[install eigen from conda-forge](https://anaconda.org/conda-forge/eigen). After cloning and navigating into the downloaded repository you can then install via:

```
pip install .
```

## Contributing

Contributions are welcome! Feel free to open issues or make pull-requests to main.
