from glob import glob
import numpy as np
from pathlib import Path
import pickle
import multiprocessing as mp
import pandas as pd

from .constants import *


def compute_probability_matrix(synapse_matrix, syncounts_discrete=False):
    if(syncounts_discrete):
        M = np.zeros_like(synapse_matrix)
        M[synapse_matrix > 0] = 1
        return M
    else:
        return 1-np.exp(-synapse_matrix)


def get_submatrix_from_index_ranges(A, index_range_tuple):
    row_low, row_high, col_low, col_high = index_range_tuple
    return A[row_low:row_high, col_low:col_high]



def get_possible_matrix(M_empirical, M_null, possible_value = POSSIBLE_VALUE):
    connected = M_empirical != 0
    overlapping = M_null != 0
    possible_connected = overlapping & ~connected
    M_empirical_possible = M_empirical + possible_connected.astype(int) * possible_value * np.ones_like(M_empirical)    
    return M_empirical_possible


