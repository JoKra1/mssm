from setuptools import setup
import git
from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension
import os

# Eigen can either be provided by conda or needs to be in "usr/include/eigen3"
# ToDo: The latter might break under windows.

SETUP_DIRECTORY = Path(__file__).resolve().parent

class get_eigen_include(object):
    # Copied from: https://github.com/MatPiq/RlassoModels/blob/master/setup.py
    # I hope this might help me with my building issues, because it was suggested in this
    # thread, that it might work: https://github.com/readthedocs/readthedocs.org/issues/9034
    EIGEN3_URL = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
    EIGEN3_DIRNAME = "eigen-3.4.0"

    def __str__(self) -> str:
        eigen_include_dir = os.environ.get("EIGEN3_INCLUDE_DIR", None)

        if eigen_include_dir is not None:
            return eigen_include_dir

        target_dir = SETUP_DIRECTORY / self.EIGEN3_DIRNAME
        if target_dir.exists():
            return target_dir.name

        download_target_dir = SETUP_DIRECTORY / "eigen3.zip"
        import zipfile

        import requests

        response = requests.get(self.EIGEN3_URL, stream=True)
        with download_target_dir.open("wb") as ofs:
            for chunk in response.iter_content(chunk_size=1024):
                ofs.write(chunk)

        with zipfile.ZipFile(download_target_dir) as ifs:
            ifs.extractall()

        return target_dir.name

eigen_path = None
if "CONDA_PREFIX" in os.environ:
    if os.path.isdir(os.environ["CONDA_PREFIX"] + "/include/eigen3"):
        eigen_path = os.environ["CONDA_PREFIX"] + "/include/eigen3"

if os.getenv('CI') is not None:
    print("CI")
    c_path = os.getcwd()
    if "eigen" not in os.listdir(c_path):
        print("Cloning into Eigen.")
        git.Repo.clone_from("https://gitlab.com/libeigen/eigen.git",c_path+"/eigen",branch="3.4.0")
    print(os.listdir(c_path))
    eigen_path = c_path + "/eigen/Eigen"
    print(eigen_path)
    
if eigen_path is None:
    eigen_path = "usr/local/" + "/include/eigen3"

# Create Pybind setuptools extension
ext = Pybind11Extension(name='cpp_solvers',
                        sources=['src/mssm/src/cpp/cpp_solvers.cpp'],
                        include_dirs=[get_eigen_include()],
                        cxx_std=14)

setup(ext_modules=[ext])
