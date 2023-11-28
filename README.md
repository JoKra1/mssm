# mssm: Markov-switching Spline Models

## Description

``mssm`` is a toolbox to estimate Generalized Additive Mixed Models (GAMMs) semi Markov-switching GAMMs (sMs-GAMMs) and sMs Impulse Response GAMMs (sMs-IR-GAMMs). The ``main`` branch is updated frequently to reflect new developments. The ``stable`` branch should reflect the latest releases. if you don't need the newest functionality, you should install from the ``stable`` branch (see below for instructions).

## Installation

The easiest option is to install from pypi via ``pip``.

1) Setup a conda environment with python > 3.10
2) Install mssm via ``pip``

Currently, mssm can only be installed from [test.pypi](https://test.pypi.org/project/mssm/#description) because sign-ups are disabled for pypi. So to install ``mssm`` run:

```
conda create -n mssm_env python=3.10
conda activate mssm_env
pip install -i https://test.pypi.org/simple/ mssm
pip install matplotlib # Only needed for tutorials
```

Once ``mssm`` can be published to pypi, you can replace the third line with ``pip install mssm``. The fourth line, installing ``matplotlib`` is only necessary if you want to run the tutorials. Note: pypi will only reflect releases on the ``stable`` branch. Tagged pushes to main will however continue to be distributed to test.pypi, so if you need the latest changes you can get them from there.

### Building from source

You can also build directly from source. This requires ``conda`` or an installation of [eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)(``setup.py`` then expects ``eigen`` in "usr/local/include/eigen3". This will probably not work on windows.). Once you have ``conda`` installed,
[install eigen from conda-forge](https://anaconda.org/conda-forge/eigen). After cloning and navigating into the downloaded repository you can then install via ``pip install . ``.

## To get started

 - With GAMMs: Take a look at tutorial 1 in the tutorial folder.
 - With sms-IR-GAMMs: Take a look at tutorial 2.
 - With sms-GAMMs: Take a look at tutorial 3.

## Contributing

Contributions are welcome! Feel free to open issues or make pull-requests to main.