#!/usr/bin/env python3
"""covMeth Type II: local low-coverage block inference.

A Type-II block contains at least two consecutive low-coverage CpGs bounded by
high-coverage anchors. The algorithm performs:

1. left-to-right recursive estimation from the left anchor;
2. right-to-left recursive estimation from the right anchor;
3. distance-aware fusion of both estimates and the original observation.

The recursive update is Kalman-like:
    m_i^L = (1-k_i^L)m_{i-1}^L + k_i^L m_i
    m_i^R = (1-k_i^R)m_{i+1}^R + k_i^R m_i

The observation gain increases with coverage and with the genomic gap from the
preceding state. Final fusion weights decay with distance to each anchor and
are normalized to sum to one.
"""
from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from common import (
    CovMethInputError,
    atomic_write,
    initialize_output,
    low_runs,
    parse_group_columns,
    read_table,
    restore_order,
    validate_input,
)

LOGGER = logging.getLogger("covmeth.local_block")


@dataclass(frozen=True)
class Config:
    coverage_threshold: int = 5
    max_anchor_distance: int = 500
    min_block_sites: int = 2
    propagation_length: float = 200.0
    observation_scale: float = 5.0
    min_observation_gain: float = 0.02
    max_observation_gain: float = 0.95
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.coverage_threshold < 1:
            raise CovMethInputError("coverage_threshold must be >= 1.")
        if self.max_anchor_distance < 1:
            raise CovMethInputError("max_anchor_distance must be >= 1.")
        if self.min_block_sites < 2:
            raise CovMethInputError("min_block_sites must be >= 2.")
        if min(self.propagation_length, self.observation_scale, self.epsilon) <= 0:
            raise CovMethInputError(
                "propagation_length, observation_scale and epsilon must be > 0."
            )
        if not 0 <= self.min_observation_gain <= self.max_observation_gain <= 1:
            raise CovMethInputError(
                "observation gains must satisfy 0 <= min <= max <= 1."
            )


def _base_gain(coverage: int, missing: bool, config: Config) -> float:
    if missing or coverage <= 0:
        return 0.0
    gain = coverage / (coverage + config.observation_scale)
    return float(
        np.clip(gain, config.min_observation_gain, config.max_observation_gain)
    )


def _recursive_gain(coverage: int, gap: int, missing: bool, config: Config) -> float:
    """Combine observation confidence with spatial separation."""
    if missing:
        return 0.0
    base = _base_gain(coverage, False, config)
    persistence = math.exp(-gap / config.propagation_length)
    gain = 1.0 - persistence * (1.0 - base)
    return float(
        np.clip(gain, config.min_observation_gain, config.max_observation_gain)
    )


def infer_block(
    *,
    positions: np.ndarray,
    methylation: np.ndarray,
    coverage: np.ndarray,
    left_anchor_position: int,
    left_anchor_methylation: float,
    right_anchor_position: int,
    right_anchor_methylation: float,
    config: Config,
) -> dict[str, np.ndarray]:
    """Infer one local low-coverage block and return all diagnostics."""
    n = len(positions)
    left_estimate = np.empty(n, dtype=float)
    right_estimate = np.empty(n, dtype=float)
    left_gain = np.empty(n, dtype=float)
    right_gain = np.empty(n, dtype=float)

    previous_estimate = float(left_anchor_methylation)
    previous_position = int(left_anchor_position)
    for i in range(n):
        missing = bool(np.isnan(methylation[i]))
        gain = _recursive_gain(
            int(coverage[i]),
            int(positions[i] - previous_position),
            missing,
            config,
        )
        observation = previous_estimate if missing else float(methylation[i])
        estimate = (1.0 - gain) * previous_estimate + gain * observation
        left_estimate[i] = np.clip(estimate, 0.0, 1.0)
        left_gain[i] = gain
        previous_estimate = left_estimate[i]
        previous_position = int(positions[i])

    next_estimate = float(right_anchor_methylation)
    next_position = int(right_anchor_position)
    for i in range(n - 1, -1, -1):
        missing = bool(np.isnan(methylation[i]))
        gain = _recursive_gain(
            int(coverage[i]),
            int(next_position - positions[i]),
            missing,
            config,
        )
        observation = next_estimate if missing else float(methylation[i])
        estimate = (1.0 - gain) * next_estimate + gain * observation
        right_estimate[i] = np.clip(estimate, 0.0, 1.0)
        right_gain[i] = gain
        next_estimate = right_estimate[i]
        next_position = int(positions[i])

    d_left = positions - left_anchor_position
    d_right = right_anchor_position - positions

    raw_left = np.exp(-d_left / config.propagation_length)
    raw_right = np.exp(-d_right / config.propagation_length)
    observation_confidence = np.array(
        [
            _base_gain(int(cov), bool(np.isnan(meth)), config)
            for cov, meth in zip(coverage, methylation)
        ],
        dtype=float,
    )
    mean_anchor_distance = 0.5 * (d_left + d_right)
    raw_observed = observation_confidence * np.exp(
        -mean_anchor_distance / config.propagation_length
    )

    normalizer = np.maximum(raw_left + raw_right + raw_observed, config.epsilon)
    beta_left = raw_left / normalizer
    beta_right = raw_right / normalizer
    beta_observed = raw_observed / normalizer

    observed_for_fusion = np.where(
        np.isnan(methylation),
        0.5 * (left_estimate + right_estimate),
        methylation,
    )
    final = (
        beta_left * left_estimate
        + beta_right * right_estimate
        + beta_observed * observed_for_fusion
    )

    return {
        "left_estimate": np.clip(left_estimate, 0.0, 1.0),
        "right_estimate": np.clip(right_estimate, 0.0, 1.0),
        "final": np.clip(final, 0.0, 1.0),
        "left_gain": left_gain,
        "right_gain": right_gain,
        "beta_left": beta_left,
        "beta_right": beta_right,
        "beta_observed": beta_observed,
        "left_distance": d_left.astype(float),
        "right_distance": d_right.astype(float),
    }


def smooth_local_blocks(
    frame: pd.DataFrame,
    *,
    config: Config = Config(),
    chrom_col: str = "chrom",
    position_col: str = "position",
    methylation_col: str = "methylation",
    coverage_col: str = "coverage",
    group_columns: Sequence[str] = ("sample", "haplotype"),
) -> pd.DataFrame:
    data = validate_input(
        frame,
        chrom_col=chrom_col,
        position_col=position_col,
        methylation_col=methylation_col,
        coverage_col=coverage_col,
        group_columns=group_columns,
    )
    data = initialize_output(
        data,
        methylation_col=methylation_col,
        coverage_col=coverage_col,
        coverage_threshold=config.coverage_threshold,
    )
    data["block_id"] = pd.NA
    for col in [
        "left_recursive_estimate",
        "right_recursive_estimate",
        "left_recursive_gain",
        "right_recursive_gain",
        "beta_left",
        "beta_right",
        "beta_observed",
        "left_anchor_distance",
        "right_anchor_distance",
    ]:
        data[col] = np.nan

    grouping = [*group_columns, chrom_col]
    ordered = data.sort_values(
        grouping + [position_col, "_covmeth_row_order"], kind="mergesort"
    )
    block_counter = 0

    for _, group in ordered.groupby(grouping, sort=False, dropna=False):
        rows = group.index.to_numpy()
        pos = group[position_col].to_numpy(dtype=np.int64)
        cov = group[coverage_col].to_numpy(dtype=np.int64)
        meth = group[methylation_col].to_numpy(dtype=float)

        for start, end in low_runs(cov < config.coverage_threshold):
            if end - start + 1 < config.min_block_sites:
                continue
            if start == 0 or end == len(group) - 1:
                continue
            if cov[start - 1] < config.coverage_threshold:
                continue
            if cov[end + 1] < config.coverage_threshold:
                continue
            if np.isnan(meth[start - 1]) or np.isnan(meth[end + 1]):
                continue

            left_edge_distance = int(pos[start] - pos[start - 1])
            right_edge_distance = int(pos[end + 1] - pos[end])
            if min(left_edge_distance, right_edge_distance) <= 0:
                continue
            if max(left_edge_distance, right_edge_distance) > config.max_anchor_distance:
                continue

            block_counter += 1
            result = infer_block(
                positions=pos[start : end + 1],
                methylation=meth[start : end + 1],
                coverage=cov[start : end + 1],
                left_anchor_position=int(pos[start - 1]),
                left_anchor_methylation=float(meth[start - 1]),
                right_anchor_position=int(pos[end + 1]),
                right_anchor_methylation=float(meth[end + 1]),
                config=config,
            )
            target_rows = rows[start : end + 1]
            data.loc[target_rows, "coverage_type"] = "type_II_local_block"
            data.loc[target_rows, "inference_status"] = "corrected"
            data.loc[target_rows, "block_id"] = f"LB{block_counter:08d}"
            data.loc[target_rows, "inferred_methylation"] = result["final"]
            data.loc[target_rows, "left_recursive_estimate"] = result[
                "left_estimate"
            ]
            data.loc[target_rows, "right_recursive_estimate"] = result[
                "right_estimate"
            ]
            data.loc[target_rows, "left_recursive_gain"] = result["left_gain"]
            data.loc[target_rows, "right_recursive_gain"] = result["right_gain"]
            data.loc[target_rows, "beta_left"] = result["beta_left"]
            data.loc[target_rows, "beta_right"] = result["beta_right"]
            data.loc[target_rows, "beta_observed"] = result["beta_observed"]
            data.loc[target_rows, "left_anchor_distance"] = result[
                "left_distance"
            ]
            data.loc[target_rows, "right_anchor_distance"] = result[
                "right_distance"
            ]

    return restore_order(data)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", required=True, type=Path)
    p.add_argument("-o", "--output", required=True, type=Path)
    p.add_argument("--sep", default=None)
    p.add_argument("--output-sep", default="\t")
    p.add_argument("--chrom-col", default="chrom")
    p.add_argument("--position-col", default="position")
    p.add_argument("--methylation-col", default="methylation")
    p.add_argument("--coverage-col", default="coverage")
    p.add_argument("--group-columns", default="sample,haplotype")
    p.add_argument("--coverage-threshold", type=int, default=5)
    p.add_argument("--max-anchor-distance", type=int, default=500)
    p.add_argument("--min-block-sites", type=int, default=2)
    p.add_argument("--propagation-length", type=float, default=200.0)
    p.add_argument("--observation-scale", type=float, default=5.0)
    p.add_argument("--min-observation-gain", type=float, default=0.02)
    p.add_argument("--max-observation-gain", type=float, default=0.95)
    p.add_argument("--log-level", default="INFO")
    return p


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )
    frame = read_table(args.input, args.sep)
    result = smooth_local_blocks(
        frame,
        config=Config(
            coverage_threshold=args.coverage_threshold,
            max_anchor_distance=args.max_anchor_distance,
            min_block_sites=args.min_block_sites,
            propagation_length=args.propagation_length,
            observation_scale=args.observation_scale,
            min_observation_gain=args.min_observation_gain,
            max_observation_gain=args.max_observation_gain,
        ),
        chrom_col=args.chrom_col,
        position_col=args.position_col,
        methylation_col=args.methylation_col,
        coverage_col=args.coverage_col,
        group_columns=parse_group_columns(args.group_columns),
    )
    atomic_write(result, args.output, args.output_sep)
    LOGGER.info(
        "Wrote %d rows; corrected %d sites in %d local blocks.",
        len(result),
        int((result["coverage_type"] == "type_II_local_block").sum()),
        int(result["block_id"].nunique(dropna=True)),
    )


if __name__ == "__main__":
    main()
