import pandas as pd
import numpy as np
import scipy as scp
import os
import warnings

CACHE_DIR = './.db'

# Functions to load & read data used to accumulate cross product of model matrix iteratively

def read_min_max(column,files,header=0,row_index=False):
    """
    Accumulates minimum and max for a specific variable over split data-files.
    """

    min_var = None
    max_var = None
    for fi,file in enumerate(files):
        dat = pd.read_csv(file,header=header,index_col=row_index)

        if fi == 0:
            min_var = min(dat[column])
            max_var = max(dat[column])
        else:
            if min_var > min(dat[column]):
                min_var = min(dat[column])
            if max_var < max(dat[column]):
                max_var = max(dat[column])
    
    return min_var,max_var

def read_unique(column,files,header=0,row_index=False):
    """
    Get unique values for a specific variable over split data-files.
    """
    unq = set()
    for fi,file in enumerate(files):
        dat = pd.read_csv(file,header=header,index_col=row_index)
        
        unq_dat = np.unique(dat[column].values)
        for u in unq_dat:
            unq.add(u)
    
    return np.array(list(unq))

def read_cov(y,x,files,header=0,row_index=False):
    """
    Collect an entire column on variable x, corrected for any NA values in the y column.
    """

    cov = []
    for fi,file in enumerate(files):
        dat = pd.read_csv(file,header=header,index_col=row_index)
        x_f = dat[x].values
        cov.extend(x_f[np.isnan(dat[y]) == False])

    return np.array(cov)

def read_dtype(column,files,header=0,row_index=False):
    dtype = None
    for fi,file in enumerate(files):
        
        dat = pd.read_csv(file,header=header,index_col=row_index)

        if fi == 0:
            dtype = dat[column].dtype
        else:
            if dtype != dat[column].dtype:
                raise TypeError("Column data type varies between different files.")
    
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