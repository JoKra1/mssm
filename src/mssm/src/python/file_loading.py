import pandas as pd
import numpy as np
import scipy as scp
import os
import warnings
import multiprocessing as mp
from itertools import repeat

CACHE_DIR = './.db'

# Functions to load & read data used to accumulate cross product of model matrix iteratively

def read_unique_single(x,file,file_loading_kwargs):

    dat = pd.read_csv(file,**file_loading_kwargs)
    unq_dat = np.unique(dat[x].values)

    return unq_dat

def read_unique(x,files,nc,file_loading_kwargs):
    """
    Get unique values for a specific variable over split data-files.
    """
    unq = set()
    with mp.Pool(processes=nc) as pool:
        unq_cov = pool.starmap(read_unique_single,zip(repeat(x),files,repeat(file_loading_kwargs)))

    unq.update(*unq_cov)

    return np.array(list(unq))

def read_cor_cov_single(y,x,file,file_loading_kwargs):
    dat = pd.read_csv(file,**file_loading_kwargs)
    x_f = dat[x].values
    return x_f[np.isnan(dat[y]) == False]

def read_cov(y,x,files,nc,file_loading_kwargs):
    """
    Collect an entire column on variable x, corrected for any NA values in the y column.
    """

    with mp.Pool(processes=nc) as pool:
        cov = pool.starmap(read_cor_cov_single,zip(repeat(y),repeat(x),files,repeat(file_loading_kwargs)))
    
    # Flatten
    cov = np.array([cv for cs in cov for cv in cs])
    return cov

def read_no_cor_cov_single(x,file,file_loading_kwargs):
    dat = pd.read_csv(file,**file_loading_kwargs)
    x_f = dat[x].values
    return x_f

def read_cov_no_cor(x,files,nc,file_loading_kwargs):
    """
    Collect an entire column on variable x, without correcting for any NA values in the y column.
    """

    with mp.Pool(processes=nc) as pool:
        cov = pool.starmap(read_no_cor_cov_single,zip(repeat(x),files,repeat(file_loading_kwargs)))
    
    # Flatten
    cov = np.array([cv for cs in cov for cv in cs])
    return cov

def read_dtype(column,file,file_loading_kwargs):
    dtype = None
    dat = pd.read_csv(file,**file_loading_kwargs)
    dtype = dat[column].dtype
    
    return dtype

def setup_cache(cache_dir:str):
    """
    Set up cache for row-subsets of model matrix.
    """
    # Check if cache directory exists
    if not os.path.isdir(cache_dir):
        warnings.warn(f"Creating cache directory {cache_dir}")
        os.makedirs(cache_dir)
    else:
        raise ValueError(f"Cache directory {cache_dir} already exists. That either means it was not properly removed (maybe fitting crashed?) or a directory with the name already exists. Please delete/remove the directory '{cache_dir}' manually.")

def clear_cache(cache_dir:str):
    """
    Clear up cache for row-subsets of model matrix.
    """
    warnings.warn(f"Removing cache directory {cache_dir}")
    for file in os.listdir(cache_dir):
        os.remove(f"{cache_dir}/" + file)
    os.removedirs(cache_dir)

def cache_mmat(cache_dir:str):
    """
    Cache row-subsets of model matrix.
    """
    def decorator(u_mmat_func):
        
        def n_mmat_func(*args):
            # Check if matrix has been created
            target = args[0].split("/")[-1].split(".csv")[0] + ".npz"
            if target not in os.listdir(cache_dir):
                mmat = u_mmat_func(*args)
                scp.sparse.save_npz(f"{cache_dir}/" + target,mmat)
            else:
                mmat = scp.sparse.load_npz(f"{cache_dir}/" + target)
            
            return mmat
        
        return n_mmat_func
    
    return decorator