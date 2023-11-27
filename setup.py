from setuptools import setup
import pathlib
from pybind11.setup_helpers import Pybind11Extension
import os

# Eigen can either be provided by conda or needs to be in "usr/include/eigen3"
# ToDo: The latter might break under windows.

eigen_path = None
if "CONDA_PREFIX" in os.environ:
    if os.path.isdir(os.environ["CONDA_PREFIX"] + "/include/eigen3"):
        eigen_path = os.environ["CONDA_PREFIX"] + "/include/eigen3"

if os.getenv('CI') is not None:
    print("CI")
    eigen_path = pathlib.Path(__file__).resolve().parent + "eigen/Eigen"
    eigen_path = eigen_path
    print(eigen_path)
    
if eigen_path is None:
    eigen_path = "usr/local/" + "/include/eigen3"

# Create Pybind setuptools extension
ext = Pybind11Extension(name='cpp_solvers',
                        sources=['src/mssm/src/cpp/cpp_solvers.cpp'],
                        cxx_std=14)

# Add path to Eigen for compiler
ext.include_dirs.append(eigen_path)

setup(ext_modules=[ext])
