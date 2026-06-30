#!/usr/bin/env python3
"""Shared utilities for covMeth smoothing modules."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


class CovMethInputError(ValueError):
    """Raised when input data or parameters violate covMeth requirements."""


def parse_group_columns(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_table(path: Path, sep: str | None = None) -> pd.DataFrame:
    if sep is None:
        sep = "\t" if path.suffix.lower() in {".tsv", ".txt", ".bed"} else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def atomic_write(frame: pd.DataFrame, output: Path, sep: str = "\t") -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        frame.to_csv(tmp, sep=sep, index=False)
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink()


def validate_input(
    frame: pd.DataFrame,
    *,
    chrom_col: str,
    position_col: str,
    methylation_col: str,
    coverage_col: str,
    group_columns: Sequence[str],
    extra_columns: Sequence[str] = (),
) -> pd.DataFrame:
    required = {
        chrom_col,
        position_col,
        methylation_col,
        coverage_col,
        *group_columns,
        *extra_columns,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise CovMethInputError("Missing required column(s): " + ", ".join(missing))

    data = frame.copy()
    data["_covmeth_row_order"] = np.arange(len(data), dtype=np.int64)
    data[position_col] = pd.to_numeric(data[position_col], errors="coerce")
    data[coverage_col] = pd.to_numeric(data[coverage_col], errors="coerce")
    data[methylation_col] = pd.to_numeric(data[methylation_col], errors="coerce")

    if data[chrom_col].isna().any():
        raise CovMethInputError(f"{chrom_col} contains missing values.")
    if data[position_col].isna().any() or (data[position_col] < 0).any():
        raise CovMethInputError(
            f"{position_col} must contain non-negative genomic coordinates."
        )
    if data[coverage_col].isna().any() or (data[coverage_col] < 0).any():
        raise CovMethInputError(
            f"{coverage_col} must contain non-negative integer coverage values."
        )
    values = data[coverage_col].to_numpy(dtype=float)
    if not np.allclose(values, np.round(values)):
        raise CovMethInputError(f"{coverage_col} must contain integers.")

    valid_m = data[methylation_col].dropna()
    if ((valid_m < 0) | (valid_m > 1)).any():
        raise CovMethInputError(f"{methylation_col} values must lie in [0, 1].")

    data[position_col] = data[position_col].astype(np.int64)
    data[coverage_col] = data[coverage_col].astype(np.int64)
    return data


def initialize_output(
    data: pd.DataFrame,
    *,
    methylation_col: str,
    coverage_col: str,
    coverage_threshold: int,
) -> pd.DataFrame:
    data = data.copy()
    data["observed_methylation"] = data[methylation_col]
    data["inferred_methylation"] = data[methylation_col]
    high = data[coverage_col] >= coverage_threshold
    data["coverage_type"] = np.where(
        high, "high_coverage", "low_coverage_unclassified"
    )
    data["inference_status"] = np.where(high, "retained", "not_inferred")
    return data


def low_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive [start, end] index pairs for consecutive True values."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def restore_order(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.sort_values("_covmeth_row_order", kind="mergesort")
        .drop(columns=["_covmeth_row_order"])
        .reset_index(drop=True)
    )
