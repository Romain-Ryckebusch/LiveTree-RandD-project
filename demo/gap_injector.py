"""
Synthetic gap injection for testing imputation robustness.

Operates on a DataFrame column by replacing values with NaN to simulate
data acquisition outages.
"""
import numpy as np
import pandas as pd


def inject_block_gaps(df, column, block_length=36, n_blocks=2, seed=None):
    """
    Inject contiguous blocks of NaN into a DataFrame column.

    Parameters
    ----------
    df : DataFrame
        Must have the specified column. Modified in place.
    column : str
        Column name to inject gaps into.
    block_length : int
        Number of consecutive NaN values per block (36 = 6 hours at 10-min).
    n_blocks : int
        Number of gap blocks to inject.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    DataFrame
        The modified DataFrame (same object, modified in place).
    list of tuple
        List of (start_idx, end_idx) for each injected gap.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    gaps = []

    for _ in range(n_blocks):
        max_start = n - block_length
        if max_start <= 0:
            break
        start = rng.integers(0, max_start)
        end = start + block_length
        df.loc[df.index[start:end], column] = np.nan
        gaps.append((start, end))

    return df, gaps


def inject_random_gaps(df, column, n_points=50, seed=None):
    """
    Inject randomly scattered NaN values into a DataFrame column.

    Parameters
    ----------
    df : DataFrame
    column : str
    n_points : int
        Number of individual points to set to NaN.
    seed : int, optional

    Returns
    -------
    DataFrame, list of int (indices set to NaN)
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(df), size=min(n_points, len(df)), replace=False)
    df.loc[df.index[indices], column] = np.nan
    return df, sorted(indices.tolist())
