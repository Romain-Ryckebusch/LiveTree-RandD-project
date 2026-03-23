#!/usr/bin/env python3
"""
Introduce artificial missing data ("holes") into a parquet time-series file.

- Can remove individual points at a given rate
- Can remove blocks of consecutive points
- Can target one or several building consumption columns or all of them
- Non-consumption columns (e.g. is_closed, is_holiday, DateTime, etc.) are left untouched
"""

import argparse
from typing import List, Sequence, Optional

import numpy as np
import pandas as pd


# Default consumption columns (adapt if your schema differs)
DEFAULT_CONSUMPTION_COLS = ["HA", "HEI1", "HEI2", "RIZOMM", "Campus"]


def select_target_columns(df: pd.DataFrame, buildings: str) -> List[str]:
    """
    Decide which columns to apply missingness to.
    `buildings` can be "all" or a comma-separated list of column names.
    Only columns present in the DataFrame will be used.
    """
    if buildings.lower() == "all":
        candidate_cols = DEFAULT_CONSUMPTION_COLS
    else:
        candidate_cols = [c.strip() for c in buildings.split(",") if c.strip()]

    cols = [c for c in candidate_cols if c in df.columns]
    if not cols:
        raise ValueError(
            f"No valid target columns found in DataFrame for specification '{buildings}'. "
            f"Available columns: {list(df.columns)}"
        )
    return cols


def introduce_missing_points(
    df: pd.DataFrame,
    cols: Sequence[str],
    missing_rate: float,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Randomly set individual (row, col) entries to NaN at a given rate.

    missing_rate: fraction of timestamps to make missing for the chosen columns.
                  Example: 0.05 -> ~5% of rows will have NaNs in those columns.
    """
    if not (0.0 < missing_rate < 1.0):
        raise ValueError("missing_rate must be in (0, 1).")

    rng = np.random.default_rng(random_state)
    n = len(df)

    # Boolean mask of rows where we will remove values
    row_mask = rng.random(n) < missing_rate

    # Apply to all selected columns
    df = df.copy()
    df.loc[row_mask, cols] = np.nan
    return df


def introduce_missing_blocks(
    df: pd.DataFrame,
    cols: Sequence[str],
    block_length: int,
    n_blocks: int,
    random_state: Optional[int] = None,
    allow_overlap: bool = False,
) -> pd.DataFrame:
    """
    Introduce missing blocks (consecutive points) in the given columns.

    block_length: number of consecutive points per block
    n_blocks    : number of blocks to insert
    allow_overlap: if False, try to avoid overlapping blocks
    """
    if block_length <= 0:
        raise ValueError("block_length must be > 0.")
    if n_blocks <= 0:
        raise ValueError("n_blocks must be > 0.")

    n = len(df)
    if block_length > n:
        raise ValueError("block_length cannot be larger than the number of rows.")

    rng = np.random.default_rng(random_state)
    df = df.copy()

    mask = np.zeros(n, dtype=bool)

    if allow_overlap:
        # Simpler: just choose random starts, allow overlaps
        starts = rng.integers(0, n - block_length + 1, size=n_blocks)
        for s in starts:
            mask[s : s + block_length] = True
    else:
        # Try to avoid overlaps
        chosen_starts = []
        attempts = 0
        max_attempts = n * 10  # simple safeguard

        while len(chosen_starts) < n_blocks and attempts < max_attempts:
            s = int(rng.integers(0, n - block_length + 1))
            if not mask[s : s + block_length].any():
                mask[s : s + block_length] = True
                chosen_starts.append(s)
            attempts += 1

        if len(chosen_starts) < n_blocks:
            print(
                f"Warning: could only place {len(chosen_starts)} non-overlapping blocks "
                f"out of requested {n_blocks}."
            )

    # Apply mask to the selected columns
    df.loc[mask, cols] = np.nan
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Introduce artificial missing data into a parquet file."
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input parquet file."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to output parquet file with holes."
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["points", "blocks"],
        help="Type of missingness to introduce.",
    )

    parser.add_argument(
        "--buildings",
        default="all",
        help=(
            "Which building consumption columns to affect: "
            "'all' or comma-separated list (e.g. 'HA,HEI1'). "
            f"Default: 'all' (uses {DEFAULT_CONSUMPTION_COLS})."
        ),
    )

    parser.add_argument(
        "--missing-rate",
        type=float,
        default=None,
        help="For mode=points: fraction of timestamps to make missing, in (0, 1).",
    )

    parser.add_argument(
        "--block-length",
        type=int,
        default=None,
        help="For mode=blocks: length of each missing block (number of consecutive rows).",
    )
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=None,
        help="For mode=blocks: number of blocks to insert.",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="For mode=blocks: allow blocks to overlap in time.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load data
    df = pd.read_parquet(args.input)

    # Decide which consumption columns to affect
    target_cols = select_target_columns(df, args.buildings)
    print(f"Applying missingness to columns: {target_cols}")

    if args.mode == "points":
        if args.missing_rate is None:
            raise SystemExit(
                "Error: --missing-rate is required for mode=points."
            )
        df_out = introduce_missing_points(
            df,
            cols=target_cols,
            missing_rate=args.missing_rate,
            random_state=args.seed,
        )

    elif args.mode == "blocks":
        if args.block_length is None or args.n_blocks is None:
            raise SystemExit(
                "Error: --block-length and --n-blocks are required for mode=blocks."
            )
        df_out = introduce_missing_blocks(
            df,
            cols=target_cols,
            block_length=args.block_length,
            n_blocks=args.n_blocks,
            random_state=args.seed,
            allow_overlap=args.allow_overlap,
        )
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")

    # Save result
    df_out.to_parquet(args.output, index=False)
    print(f"Written output with artificial holes to: {args.output}")


if __name__ == "__main__":
    main()
