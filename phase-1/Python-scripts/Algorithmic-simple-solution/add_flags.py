import pandas as pd
from pathlib import Path

def add_weekend_night_flags(input_parquet: str, output_parquet: str | None = None) -> str:
    """
    Read a parquet file, add is_weekend and is_night columns, and write it back.

    Parameters
    ----------
    input_parquet : str
        Path to the input parquet file.
    output_parquet : str | None
        Path for the output parquet file. If None, will create
        '<original_name>_with_flags.parquet' in the same directory.

    Returns
    -------
    str
        Path of the written parquet file.
    """
    # Read original data
    df = pd.read_parquet(input_parquet)

    # Ensure DateTime is a proper datetime
    # (utc=True is usually safe with "...Z", you can drop it if you prefer naive)
    dt = pd.to_datetime(df["DateTime"], utc=True)

    # Weekend: Saturday (5) or Sunday (6)
    df["is_weekend"] = dt.dt.dayofweek >= 5

    # Night: 22:00–06:00 (22, 23, 0–5)
    df["is_night"] = (dt.dt.hour >= 22) | (dt.dt.hour < 6)

    # Define output file if not provided
    if output_parquet is None:
        input_path = Path(input_parquet)
        output_parquet = str(
            input_path.with_name(input_path.stem + "_with_flags" + input_path.suffix)
        )

    # Write updated data
    df.to_parquet(output_parquet, index=False)

    return output_parquet


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Add is_weekend and is_night flags to a parquet file."
    )
    parser.add_argument("input", help="Path to input parquet file")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output parquet file (default: <name>_with_flags.parquet)",
        default=None,
    )
    args = parser.parse_args()

    out_path = add_weekend_night_flags(args.input, args.output)
    print(f"Written: {out_path}")
