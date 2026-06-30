#!/usr/bin/env python3
"""covMeth Type III: annotation-guided continuous low-coverage inference.

A Type-III region contains at least two consecutive low-coverage CpGs and lacks
two reliable nearby high-coverage anchors. Each target site must belong to a
specific functional annotation unit identified by ``annotation_id``.

For annotation unit B:
    mu_B = sum_j(d_j*m_j) / sum_j(d_j)

When enough reliable CpGs are available, a coverage-weighted linear spatial
trend is fitted. The context estimate is:
    m_context_i = mu_B + beta_B*(z_i-z_bar_B)

The final estimate uses coverage-dependent shrinkage:
    c_i = shrinkage_scale/(d_i+shrinkage_scale)
    m_final_i = (1-c_i)m_i + c_i*m_context_i
"""
from __future__ import annotations

import argparse
import logging
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

LOGGER = logging.getLogger("covmeth.continuous_region")


@dataclass(frozen=True)
class Config:
    coverage_threshold: int = 5
    max_anchor_distance: int = 500
    min_region_sites: int = 2
    min_trend_points: int = 3
    shrinkage_scale: float = 5.0
    max_abs_slope_per_kb: float = 1.0
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.coverage_threshold < 1:
            raise CovMethInputError("coverage_threshold must be >= 1.")
        if self.max_anchor_distance < 1:
            raise CovMethInputError("max_anchor_distance must be >= 1.")
        if self.min_region_sites < 2:
            raise CovMethInputError("min_region_sites must be >= 2.")
        if self.min_trend_points < 2:
            raise CovMethInputError("min_trend_points must be >= 2.")
        if min(self.shrinkage_scale, self.max_abs_slope_per_kb, self.epsilon) <= 0:
            raise CovMethInputError(
                "shrinkage_scale, max_abs_slope_per_kb and epsilon must be > 0."
            )


def identify_type_iii(
    positions: np.ndarray,
    coverage: np.ndarray,
    *,
    config: Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a Type-III mask and local region labels for one sorted group."""
    mask = np.zeros(len(coverage), dtype=bool)
    labels = np.full(len(coverage), -1, dtype=np.int64)
    label = 0

    for start, end in low_runs(coverage < config.coverage_threshold):
        if end - start + 1 < config.min_region_sites:
            continue

        left_reliable = False
        if start > 0 and coverage[start - 1] >= config.coverage_threshold:
            distance = int(positions[start] - positions[start - 1])
            left_reliable = 0 < distance <= config.max_anchor_distance

        right_reliable = False
        if end < len(coverage) - 1 and coverage[end + 1] >= config.coverage_threshold:
            distance = int(positions[end + 1] - positions[end])
            right_reliable = 0 < distance <= config.max_anchor_distance

        # Two nearby reliable anchors define a Type-II block.
        if left_reliable and right_reliable:
            continue

        mask[start : end + 1] = True
        labels[start : end + 1] = label
        label += 1

    return mask, labels


def weighted_baseline(
    methylation: np.ndarray,
    coverage: np.ndarray,
    *,
    epsilon: float,
) -> float:
    valid = (~np.isnan(methylation)) & (coverage > 0)
    if not valid.any():
        return float("nan")
    weights = coverage[valid].astype(float)
    return float(
        np.sum(weights * methylation[valid]) / max(np.sum(weights), epsilon)
    )


def weighted_trend(
    positions: np.ndarray,
    methylation: np.ndarray,
    coverage: np.ndarray,
    *,
    config: Config,
) -> tuple[float, float, int]:
    """Fit a coverage-weighted linear trend and return slope, center and n."""
    valid = (
        (coverage >= config.coverage_threshold)
        & (~np.isnan(methylation))
    )
    n_points = int(valid.sum())

    # Fallback to all observed CpGs in the annotation unit.
    if n_points < config.min_trend_points:
        valid = (coverage > 0) & (~np.isnan(methylation))
        n_points = int(valid.sum())
    if n_points < config.min_trend_points:
        return 0.0, float(np.mean(positions)), n_points

    x = positions[valid].astype(float)
    y = methylation[valid].astype(float)
    w = coverage[valid].astype(float)
    w_sum = float(np.sum(w))
    x_bar = float(np.sum(w * x) / w_sum)
    y_bar = float(np.sum(w * y) / w_sum)
    centered_x = x - x_bar
    denominator = float(np.sum(w * centered_x * centered_x))
    if denominator <= config.epsilon:
        return 0.0, x_bar, n_points

    slope = float(np.sum(w * centered_x * (y - y_bar)) / denominator)
    limit = config.max_abs_slope_per_kb / 1000.0
    slope = float(np.clip(slope, -limit, limit))
    return slope, x_bar, n_points


def smooth_continuous_regions(
    frame: pd.DataFrame,
    *,
    config: Config = Config(),
    chrom_col: str = "chrom",
    position_col: str = "position",
    methylation_col: str = "methylation",
    coverage_col: str = "coverage",
    annotation_col: str = "annotation_id",
    group_columns: Sequence[str] = ("sample", "haplotype"),
) -> pd.DataFrame:
    data = validate_input(
        frame,
        chrom_col=chrom_col,
        position_col=position_col,
        methylation_col=methylation_col,
        coverage_col=coverage_col,
        group_columns=group_columns,
        extra_columns=(annotation_col,),
    )
    data = initialize_output(
        data,
        methylation_col=methylation_col,
        coverage_col=coverage_col,
        coverage_threshold=config.coverage_threshold,
    )
    data["continuous_region_id"] = pd.NA
    for col in [
        "regional_baseline",
        "regional_slope_per_kb",
        "trend_point_count",
        "context_methylation",
        "context_weight",
    ]:
        data[col] = np.nan

    grouping = [*group_columns, chrom_col]
    ordered = data.sort_values(
        grouping + [position_col, "_covmeth_row_order"], kind="mergesort"
    )

    target_rows: set[int] = set()
    region_counter = 0
    for _, group in ordered.groupby(grouping, sort=False, dropna=False):
        rows = group.index.to_numpy()
        positions = group[position_col].to_numpy(dtype=np.int64)
        coverage = group[coverage_col].to_numpy(dtype=np.int64)
        _, labels = identify_type_iii(positions, coverage, config=config)

        for local_label in np.unique(labels[labels >= 0]):
            region_counter += 1
            selected = rows[labels == local_label]
            target_rows.update(int(row) for row in selected)
            data.loc[selected, "continuous_region_id"] = f"CR{region_counter:08d}"

    if not target_rows:
        return restore_order(data)

    # The annotation ID must identify one concrete genomic interval.
    annotation_groups = [*group_columns, chrom_col, annotation_col]
    for _, annotation_unit in data.groupby(
        annotation_groups, sort=False, dropna=False
    ):
        annotation_value = annotation_unit[annotation_col].iloc[0]
        if pd.isna(annotation_value) or str(annotation_value).strip() == "":
            continue

        unit_rows = annotation_unit.index.to_numpy()
        positions = annotation_unit[position_col].to_numpy(dtype=np.int64)
        methylation = annotation_unit[methylation_col].to_numpy(dtype=float)
        coverage = annotation_unit[coverage_col].to_numpy(dtype=np.int64)

        baseline = weighted_baseline(
            methylation, coverage, epsilon=config.epsilon
        )
        if np.isnan(baseline):
            continue
        slope, position_center, n_trend = weighted_trend(
            positions, methylation, coverage, config=config
        )

        for row in unit_rows:
            if int(row) not in target_rows:
                continue

            position = int(data.at[row, position_col])
            observed = data.at[row, methylation_col]
            depth = int(data.at[row, coverage_col])
            context = baseline + slope * (position - position_center)
            context = float(np.clip(context, 0.0, 1.0))

            if pd.isna(observed):
                context_weight = 1.0
                final = context
            else:
                context_weight = config.shrinkage_scale / (
                    depth + config.shrinkage_scale
                )
                final = (
                    (1.0 - context_weight) * float(observed)
                    + context_weight * context
                )

            data.at[row, "coverage_type"] = "type_III_continuous"
            data.at[row, "inference_status"] = "corrected"
            data.at[row, "inferred_methylation"] = float(
                np.clip(final, 0.0, 1.0)
            )
            data.at[row, "regional_baseline"] = baseline
            data.at[row, "regional_slope_per_kb"] = slope * 1000.0
            data.at[row, "trend_point_count"] = n_trend
            data.at[row, "context_methylation"] = context
            data.at[row, "context_weight"] = context_weight

    for row in target_rows:
        if data.at[row, "coverage_type"] != "type_III_continuous":
            data.at[row, "coverage_type"] = "type_III_no_annotation_model"
            data.at[row, "inference_status"] = "not_inferred"

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
    p.add_argument("--annotation-col", default="annotation_id")
    p.add_argument("--group-columns", default="sample,haplotype")
    p.add_argument("--coverage-threshold", type=int, default=5)
    p.add_argument("--max-anchor-distance", type=int, default=500)
    p.add_argument("--min-region-sites", type=int, default=2)
    p.add_argument("--min-trend-points", type=int, default=3)
    p.add_argument("--shrinkage-scale", type=float, default=5.0)
    p.add_argument("--max-abs-slope-per-kb", type=float, default=1.0)
    p.add_argument("--log-level", default="INFO")
    return p


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )
    frame = read_table(args.input, args.sep)
    result = smooth_continuous_regions(
        frame,
        config=Config(
            coverage_threshold=args.coverage_threshold,
            max_anchor_distance=args.max_anchor_distance,
            min_region_sites=args.min_region_sites,
            min_trend_points=args.min_trend_points,
            shrinkage_scale=args.shrinkage_scale,
            max_abs_slope_per_kb=args.max_abs_slope_per_kb,
        ),
        chrom_col=args.chrom_col,
        position_col=args.position_col,
        methylation_col=args.methylation_col,
        coverage_col=args.coverage_col,
        annotation_col=args.annotation_col,
        group_columns=parse_group_columns(args.group_columns),
    )
    atomic_write(result, args.output, args.output_sep)
    LOGGER.info(
        "Wrote %d rows; corrected %d sites in %d continuous regions.",
        len(result),
        int((result["coverage_type"] == "type_III_continuous").sum()),
        int(result["continuous_region_id"].nunique(dropna=True)),
    )


if __name__ == "__main__":
    main()
