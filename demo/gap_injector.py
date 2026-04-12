import numpy as np
import pandas as pd


def inject_block_gaps(df, column, block_length=36, n_blocks=2, seed=None):
    """Replace n_blocks contiguous runs of length block_length with NaN."""
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
    """Set n_points random rows of column to NaN."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(df), size=min(n_points, len(df)), replace=False)
    df.loc[df.index[indices], column] = np.nan
    return df, sorted(indices.tolist())
