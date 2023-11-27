from setuptools import setup
import git
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
    c_path = os.getcwd()
    git.Repo.clone_from("https://gitlab.com/libeigen/eigen.git",c_path+"/eigen",branch="3.4.0")
    print(os.listdir(c_path))
    eigen_path = c_path + "/eigen/Eigen"
    print(eigen_path)
    
if eigen_path is None:
    eigen_path = "usr/local/" + "/include/eigen3"

# Create Pybind setuptools extension
ext = Pybind11Extension(name='cpp_solvers',
                        sources=['src/mssm/src/cpp/cpp_solvers.cpp'],
                        include_dirs=[eigen_path],
                        cxx_std=14)

setup(ext_modules=[ext])
