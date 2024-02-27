import pandas as pd
import numpy as np


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