import pandas as pd
import numpy as np
import scipy as scp
import os
import warnings
import multiprocessing as mp
from itertools import repeat

# Functions to load & read data used to accumulate cross product of model matrix iteratively

def read_unique_single(x:str,file:str,file_loading_kwargs:dict) -> np.ndarray:
    """Read unique values of covariate ``x`` from ``file``.

    :param x: covariate name
    :type x: str
    :param file: file name
    :type file: str
    :param file_loading_kwargs: Any optional file loading key-word arguments.
    :type file_loading_kwargs: dict
    :return: numpy array holding unique values
    :rtype: np.ndarray
    """

    dat = pd.read_csv(file,**file_loading_kwargs)
    unq_dat = np.unique(dat[x].values)

    return unq_dat

def read_unique(x:str,files:list[str],nc:int,file_loading_kwargs:dict) -> np.ndarray:
    """Read unique values of covariate ``x`` from ``files``.

    :param x: covariate name
    :type x: str
    :param files: list of file names
    :type files: list[str]
    :param nc: Number of cores to use to read in parallel
    :type nc: int
    :param file_loading_kwargs: Any optional file loading key-word arguments.
    :type file_loading_kwargs: dict
    :return: numpy array holding unique values
    :rtype: np.ndarray
    """
    unq = set()
    with mp.Pool(processes=nc) as pool:
        unq_cov = pool.starmap(read_unique_single,zip(repeat(x),files,repeat(file_loading_kwargs)))

    unq.update(*unq_cov)

    return np.array(list(unq))

def read_cor_cov_single(y:str,x:str,file:str,file_loading_kwargs:dict) -> np.ndarray:
    """Read values of covariate ``x`` from ``file`` correcting for NaNs in ``y``.

    :param y: name of covariate potentially having NaNs
    :type y: str
    :param x: covariate name
    :type x: str
    :param file: file name
    :type file: str
    :param file_loading_kwargs: Any optional file loading key-word arguments.
    :type file_loading_kwargs: dict
    :return: numpy array holding values in ``x`` for which ``y`` is not NaN
    :rtype: np.ndarray
    """
    dat = pd.read_csv(file,**file_loading_kwargs)
    x_f = dat[x].values
    return x_f[np.isnan(dat[y]) == False]

def read_cov(y:str,x:str,files:list[str],nc:int,file_loading_kwargs:dict) -> np.ndarray:
    """Read values of covariate ``x`` from ``files`` correcting for NaNs in ``y``.

    :param y: name of covariate potentially having NaNs
    :type y: str
    :param x: covariate name
    :type x: str
    :param files: list of file names
    :type files: list[str]
    :param nc: Number of cores to use to read in parallel
    :type nc: int
    :param file_loading_kwargs: Any optional file loading key-word arguments.
    :type file_loading_kwargs: dict
    :return: numpy array holding values in ``x`` for which ``y`` is not NaN
    :rtype: np.ndarray
    """

    with mp.Pool(processes=nc) as pool:
        cov = pool.starmap(read_cor_cov_single,zip(repeat(y),repeat(x),files,repeat(file_loading_kwargs)))
    
    # Flatten
    cov = np.array([cv for cs in cov for cv in cs])
    return cov

def read_no_cor_cov_single(x:str,file:str,file_loading_kwargs:dict) -> np.ndarray:
    """Read values of covariate ``x`` from ``file``.

    :param x: covariate name
    :type x: str
    :param file: file name
    :type file: str
    :param file_loading_kwargs: Any optional file loading key-word arguments.
    :type file_loading_kwargs: dict
    :return: numpy array holding values in ``x``
    :rtype: np.ndarray
    """
    dat = pd.read_csv(file,**file_loading_kwargs)
    x_f = dat[x].values
    return x_f

def read_cov_no_cor(x:str,files:list[str],nc:int,file_loading_kwargs:dict) -> np.ndarray:
    """Read values of covariate ``x`` from ``files``.

    :param x: covariate name
    :type x: str
    :param files: list of file names
    :type files: list[str]
    :param nc: Number of cores to use to read in parallel
    :type nc: int
    :param file_loading_kwargs: Any optional file loading key-word arguments.
    :type file_loading_kwargs: dict
    :return: numpy array holding values in ``x``
    :rtype: np.ndarray
    """

    with mp.Pool(processes=nc) as pool:
        cov = pool.starmap(read_no_cor_cov_single,zip(repeat(x),files,repeat(file_loading_kwargs)))
    
    # Flatten
    cov = np.array([cv for cs in cov for cv in cs])
    return cov

def read_dtype(column:str,file:str,file_loading_kwargs:dict) -> np.dtype:
    """Read datatype of variable ``column`` in ``file``.

    :param column: Name of covariate
    :type column: str
    :param file: file name
    :type file: str
    :param file_loading_kwargs: Any optional file loading key-word arguments.
    :type file_loading_kwargs: dict
    :return: Datatype (numpy) of ``colum``
    :rtype: np.dtype
    """
    dtype = None
    dat = pd.read_csv(file,**file_loading_kwargs)
    dtype = dat[column].dtype
    
    return dtype

def setup_cache(cache_dir:str,should_cache:bool) -> None:
    """Set up cache for row-subsets of model matrix.

    :param cache_dir: path to cache directory
    :type cache_dir: str
    :param should_cache: whether or not the directory should actually be created
    :type should_cache: bool
    :raises ValueError: if the directory already exists
    """
    if should_cache:
        # Check if cache directory exists
        if not os.path.isdir(cache_dir):
            warnings.warn(f"Creating cache directory {cache_dir}")
            os.makedirs(cache_dir)
        else:
            raise ValueError(f"Cache directory {cache_dir} already exists. That either means it was not properly removed (maybe fitting crashed?) or a directory with the name already exists. Please delete/remove the directory '{cache_dir}' manually.")

def clear_cache(cache_dir:str,should_cache:bool) -> None:
    """
    Clear up cache for row-subsets of model matrix.

    :param cache_dir: path to cache directory
    :type cache_dir: str
    :param should_cache: whether or not the directory should actually be created
    :type should_cache: bool
    """
    if should_cache:
        warnings.warn(f"Removing cache directory {cache_dir}")
        for file in os.listdir(cache_dir):
            os.remove(f"{cache_dir}/" + file)
        os.removedirs(cache_dir)
