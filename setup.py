from setuptools import setup
import git
from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension
import os

# Eigen can either be provided by conda or needs to be in "usr/include/eigen3"
# ToDo: The latter might break under windows.

class get_eigen_include(object):
    # Based on the implementation from: https://github.com/MatPiq/RlassoModels/blob/master/setup.py
    # This finally, allowed me to fix my building issues since the author of the repo above had
    # the same problem: https://github.com/readthedocs/readthedocs.org/issues/9034.
    # I made some changes to stream-line the cloning into Eigen a bit.

    def __str__(self) -> str:

        # Local build based on conda
        if "CONDA_PREFIX" in os.environ:
            if os.path.isdir(os.environ["CONDA_PREFIX"] + "/include/eigen3"):
                return os.environ["CONDA_PREFIX"] + "/include/eigen3"

        # Local build with alternative accepted location.
        if os.path.isdir("usr/local/include/eigen3"):
            return "usr/local/include/eigen3"

        target_dir = Path(__file__).resolve().parent / "eigen"
        if target_dir.exists():
            return target_dir.name

        # If we cannot find an eigen installation in the current step we have to download it first.
        # See: https://gist.github.com/plembo/a786ce2851cec61ac3a051fcaf3ccdab
        git.Repo.clone_from("https://gitlab.com/libeigen/eigen.git",target_dir,branch="3.4.0")

        return target_dir.name

# Create Pybind setuptools extension
ext1 = Pybind11Extension(name='eigen_solvers',
                        sources=['src/mssm/src/cpp/eigen_solvers.cpp'],
                        include_dirs=[get_eigen_include()],
                        cxx_std=14)

ext2 = Pybind11Extension(name='davies',
                        sources=['src/mssm/src/cpp/davies.cpp'],
                        cxx_std=14)

ext3 = Pybind11Extension(name='dChol',
                        sources=['src/mssm/src/cpp/dchol.cpp'],
                        include_dirs=[get_eigen_include()],
                        cxx_std=14)

setup(ext_modules=[ext1,ext2,ext3])
