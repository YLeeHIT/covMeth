#!/usr/bin/env python3
"""covMeth Type I: isolated low-coverage CpG inference.

A target CpG is processed when it is low coverage, its immediately adjacent
CpGs are high coverage, and both anchors are within the configured distance.

Model
-----
w_L = log(1+d_L)/(Delta_L+epsilon)
w_R = log(1+d_R)/(Delta_R+epsilon)
m_context = (w_L*m_L + w_R*m_R)/(w_L+w_R)
S = (w_L+w_R)*exp(-abs(m_L-m_R)/delta)
lambda = S/(S+tau)
m_final = (1-lambda)*m_observed + lambda*m_context
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
    parse_group_columns,
    read_table,
    restore_order,
    validate_input,
)

LOGGER = logging.getLogger("covmeth.isolated")


@dataclass(frozen=True)
class Config:
    coverage_threshold: int = 5
    max_anchor_distance: int = 500
    delta: float = 0.10
    tau: float = 0.10
    min_shrinkage: float = 0.0
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.coverage_threshold < 1:
            raise CovMethInputError("coverage_threshold must be >= 1.")
        if self.max_anchor_distance < 1:
            raise CovMethInputError("max_anchor_distance must be >= 1.")
        if min(self.delta, self.tau, self.epsilon) <= 0:
            raise CovMethInputError("delta, tau and epsilon must be > 0.")
        if not 0 <= self.min_shrinkage <= 1:
            raise CovMethInputError("min_shrinkage must be in [0, 1].")


def estimate_isolated(
    *,
    observed: float,
    left_methylation: float,
    right_methylation: float,
    left_coverage: int,
    right_coverage: int,
    left_distance: int,
    right_distance: int,
    config: Config,
) -> dict[str, float | bool]:
    """Estimate one isolated low-coverage CpG and return diagnostics."""
    w_left = math.log1p(left_coverage) / (left_distance + config.epsilon)
    w_right = math.log1p(right_coverage) / (right_distance + config.epsilon)
    total = w_left + w_right
    if not math.isfinite(total) or total <= 0:
        raise ArithmeticError("The sum of anchor weights is not positive.")

    context = (
        w_left * left_methylation + w_right * right_methylation
    ) / total
    evidence = total * math.exp(
        -abs(left_methylation - right_methylation) / config.delta
    )
    shrinkage = evidence / (evidence + config.tau)

    if pd.isna(observed):
        final = context
        applied = True
        applied_shrinkage = 1.0
    elif shrinkage >= config.min_shrinkage:
        final = (1.0 - shrinkage) * observed + shrinkage * context
        applied = True
        applied_shrinkage = shrinkage
    else:
        final = observed
        applied = False
        applied_shrinkage = 0.0

    return {
        "context": float(np.clip(context, 0.0, 1.0)),
        "final": float(np.clip(final, 0.0, 1.0)),
        "w_left": float(w_left),
        "w_right": float(w_right),
        "evidence": float(evidence),
        "shrinkage": float(applied_shrinkage),
        "applied": applied,
    }


def smooth_isolated_sites(
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
    for col in [
        "context_methylation",
        "left_anchor_distance",
        "right_anchor_distance",
        "left_anchor_weight",
        "right_anchor_weight",
        "neighborhood_evidence",
        "shrinkage_weight",
    ]:
        data[col] = np.nan

    grouping = [*group_columns, chrom_col]
    ordered = data.sort_values(
        grouping + [position_col, "_covmeth_row_order"], kind="mergesort"
    )

    for _, group in ordered.groupby(grouping, sort=False, dropna=False):
        rows = group.index.to_numpy()
        pos = group[position_col].to_numpy(dtype=np.int64)
        cov = group[coverage_col].to_numpy(dtype=np.int64)
        meth = group[methylation_col].to_numpy(dtype=float)

        for i in range(1, len(group) - 1):
            if cov[i] >= config.coverage_threshold:
                continue
            if cov[i - 1] < config.coverage_threshold or cov[i + 1] < config.coverage_threshold:
                continue
            if np.isnan(meth[i - 1]) or np.isnan(meth[i + 1]):
                continue

            d_left = int(pos[i] - pos[i - 1])
            d_right = int(pos[i + 1] - pos[i])
            if min(d_left, d_right) <= 0:
                continue
            if max(d_left, d_right) > config.max_anchor_distance:
                continue

            result = estimate_isolated(
                observed=float(meth[i]),
                left_methylation=float(meth[i - 1]),
                right_methylation=float(meth[i + 1]),
                left_coverage=int(cov[i - 1]),
                right_coverage=int(cov[i + 1]),
                left_distance=d_left,
                right_distance=d_right,
                config=config,
            )
            row = rows[i]
            data.at[row, "coverage_type"] = "type_I_isolated"
            data.at[row, "inferred_methylation"] = result["final"]
            data.at[row, "context_methylation"] = result["context"]
            data.at[row, "left_anchor_distance"] = d_left
            data.at[row, "right_anchor_distance"] = d_right
            data.at[row, "left_anchor_weight"] = result["w_left"]
            data.at[row, "right_anchor_weight"] = result["w_right"]
            data.at[row, "neighborhood_evidence"] = result["evidence"]
            data.at[row, "shrinkage_weight"] = result["shrinkage"]
            data.at[row, "inference_status"] = (
                "corrected" if result["applied"] else "retained_weak_evidence"
            )

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
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--tau", type=float, default=0.10)
    p.add_argument("--min-shrinkage", type=float, default=0.0)
    p.add_argument("--log-level", default="INFO")
    return p


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )
    frame = read_table(args.input, args.sep)
    result = smooth_isolated_sites(
        frame,
        config=Config(
            coverage_threshold=args.coverage_threshold,
            max_anchor_distance=args.max_anchor_distance,
            delta=args.delta,
            tau=args.tau,
            min_shrinkage=args.min_shrinkage,
        ),
        chrom_col=args.chrom_col,
        position_col=args.position_col,
        methylation_col=args.methylation_col,
        coverage_col=args.coverage_col,
        group_columns=parse_group_columns(args.group_columns),
    )
    atomic_write(result, args.output, args.output_sep)
    LOGGER.info(
        "Wrote %d rows; corrected %d isolated sites.",
        len(result),
        int((result["inference_status"] == "corrected").sum()),
    )


if __name__ == "__main__":
    main()
