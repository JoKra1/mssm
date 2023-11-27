# mssm: Markov-switching Spline Models

## Description
``mssm`` is a toolbox to estimate Generalized Additive Mixed Models (GAMMs) semi Markov-switching GAMMs (sMs-GAMMs) and sMs Impulse respopnse GAMMs (sMs-IR-GAMMs).
The ``main`` branch is updated frequently to reflect new developments. The ``stable`` branch should reflect the latest releases. if you don't need the latest functionality, you
should install from the ``stable`` branch.

## Installation
There are two options!

Option 1 is to build directly from source. This requires ``conda`` or an installation of [eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)(``setup.py`` then expects ``eigen`` in "usr/include/eigen3"). If you have ``conda`` installed
[install eigen from conda-forge](https://anaconda.org/conda-forge/eigen). After cloning and navigating into the downloaded repository you can then install via ``pip install . ``.

Option 2 (currently being worked on) is to install one of the releases from the ``stable`` branch.

## To get started

 - With GAMMs: Take a look at tutorial 1 in the tutorial folder.
 - With sms-IR-GAMMs: Take a look at tutorial 2.
 - With sms-GAMMs: Take a look at tutorial 3.
