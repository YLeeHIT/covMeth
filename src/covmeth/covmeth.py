#!/usr/bin/env python3
"""Unified covMeth workflow for low-coverage CpG smoothing.

The workflow automatically assigns each low-coverage CpG to one of the three
covMeth structures and invokes the corresponding inference module:

Type I  : one isolated low-coverage CpG bounded by two nearby high-coverage
          CpGs;
Type II : at least two consecutive low-coverage CpGs bounded by two nearby
          high-coverage CpGs;
Type III: at least two consecutive low-coverage CpGs without two reliable
          nearby anchors, recovered using an annotation-unit baseline and
          spatial trend.

Sites that do not satisfy one of these structures are retained unchanged and
reported as ``low_coverage_unclassified``. Type-III candidates without a valid
annotation model are retained unchanged and reported as
``type_III_no_annotation_model``.

Example
-------
python covmeth.py \
    --input example_input.tsv \
    --output covmeth_output.tsv \
    --coverage-threshold 5 \
    --max-anchor-distance 500 \
    --annotation-col annotation_id
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

# Allow both ``python covmeth.py`` and imports from the same source directory.
MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from common import (  # noqa: E402
    CovMethInputError,
    atomic_write,
    parse_group_columns,
    read_table,
)
from covmeth_continuous_region import (  # noqa: E402
    Config as ContinuousConfig,
    smooth_continuous_regions,
)
from covmeth_isolated import (  # noqa: E402
    Config as IsolatedConfig,
    smooth_isolated_sites,
)
from covmeth_local_block import (  # noqa: E402
    Config as LocalBlockConfig,
    smooth_local_blocks,
)

LOGGER = logging.getLogger("covmeth")


@dataclass(frozen=True)
class CovMethConfig:
    """Configuration for the complete covMeth workflow."""

    coverage_threshold: int = 5
    max_anchor_distance: int = 500

    # Type I: isolated low-coverage CpG.
    isolated_delta: float = 0.10
    isolated_tau: float = 0.10
    isolated_min_shrinkage: float = 0.0

    # Type II: local low-coverage block.
    local_min_block_sites: int = 2
    local_propagation_length: float = 200.0
    local_observation_scale: float = 5.0
    local_min_observation_gain: float = 0.02
    local_max_observation_gain: float = 0.95

    # Type III: continuous low-coverage region.
    continuous_min_region_sites: int = 2
    continuous_min_trend_points: int = 3
    continuous_shrinkage_scale: float = 5.0
    continuous_max_abs_slope_per_kb: float = 1.0

    def __post_init__(self) -> None:
        # Constructing sub-configurations centralizes validation and ensures
        # that the unified CLI obeys exactly the same constraints as each
        # standalone module.
        self.isolated_config()
        self.local_config()
        self.continuous_config()

    def isolated_config(self) -> IsolatedConfig:
        return IsolatedConfig(
            coverage_threshold=self.coverage_threshold,
            max_anchor_distance=self.max_anchor_distance,
            delta=self.isolated_delta,
            tau=self.isolated_tau,
            min_shrinkage=self.isolated_min_shrinkage,
        )

    def local_config(self) -> LocalBlockConfig:
        return LocalBlockConfig(
            coverage_threshold=self.coverage_threshold,
            max_anchor_distance=self.max_anchor_distance,
            min_block_sites=self.local_min_block_sites,
            propagation_length=self.local_propagation_length,
            observation_scale=self.local_observation_scale,
            min_observation_gain=self.local_min_observation_gain,
            max_observation_gain=self.local_max_observation_gain,
        )

    def continuous_config(self) -> ContinuousConfig:
        return ContinuousConfig(
            coverage_threshold=self.coverage_threshold,
            max_anchor_distance=self.max_anchor_distance,
            min_region_sites=self.continuous_min_region_sites,
            min_trend_points=self.continuous_min_trend_points,
            shrinkage_scale=self.continuous_shrinkage_scale,
            max_abs_slope_per_kb=self.continuous_max_abs_slope_per_kb,
        )


TYPE_I_COLUMNS = (
    "context_methylation",
    "left_anchor_distance",
    "right_anchor_distance",
    "left_anchor_weight",
    "right_anchor_weight",
    "neighborhood_evidence",
    "shrinkage_weight",
)

TYPE_II_COLUMNS = (
    "block_id",
    "left_recursive_estimate",
    "right_recursive_estimate",
    "left_recursive_gain",
    "right_recursive_gain",
    "beta_left",
    "beta_right",
    "beta_observed",
    "left_anchor_distance",
    "right_anchor_distance",
)

TYPE_III_COLUMNS = (
    "continuous_region_id",
    "regional_baseline",
    "regional_slope_per_kb",
    "trend_point_count",
    "context_methylation",
    "context_weight",
)

STANDARD_RESULT_COLUMNS = (
    "observed_methylation",
    "inferred_methylation",
    "coverage_type",
    "inference_status",
)


def _ensure_same_length(
    input_frame: pd.DataFrame,
    *module_results: pd.DataFrame,
) -> None:
    expected = len(input_frame)
    for result in module_results:
        if len(result) != expected:
            raise RuntimeError(
                "A covMeth submodule changed the number of rows: "
                f"expected {expected}, observed {len(result)}."
            )


def _assert_disjoint_masks(**masks: pd.Series) -> None:
    """Fail loudly if two structural classes select the same row."""
    names = list(masks)
    for i, left_name in enumerate(names):
        left = masks[left_name].to_numpy(dtype=bool)
        for right_name in names[i + 1 :]:
            right = masks[right_name].to_numpy(dtype=bool)
            overlap = int(np.count_nonzero(left & right))
            if overlap:
                raise RuntimeError(
                    "Internal covMeth classification conflict: "
                    f"{overlap} row(s) were assigned to both "
                    f"{left_name} and {right_name}."
                )


def _add_prefixed_diagnostics(
    result: pd.DataFrame,
    source: pd.DataFrame,
    mask: pd.Series,
    *,
    columns: Sequence[str],
    prefix: str,
) -> None:
    """Copy diagnostics only for rows processed by one submodule."""
    for column in columns:
        if column not in source.columns:
            continue
        target = f"{prefix}{column}"
        if pd.api.types.is_numeric_dtype(source[column].dtype):
            result[target] = np.nan
        else:
            result[target] = pd.NA
        result.loc[mask, target] = source.loc[mask, column].to_numpy()


def _make_summary(
    result: pd.DataFrame,
    *,
    config: CovMethConfig,
    annotation_available: bool,
) -> dict[str, Any]:
    site_types = result["covmeth_site_type"].value_counts(dropna=False)
    statuses = result["covmeth_status"].value_counts(dropna=False)
    total = int(len(result))
    corrected = int(result["covmeth_smoothed"].sum())

    return {
        "program": "covMeth",
        "total_sites": total,
        "high_coverage_sites": int(
            (result["covmeth_site_type"] == "high_coverage").sum()
        ),
        "low_coverage_sites": int(
            (result["covmeth_site_type"] != "high_coverage").sum()
        ),
        "corrected_sites": corrected,
        "corrected_fraction": corrected / total if total else 0.0,
        "annotation_available": bool(annotation_available),
        "site_type_counts": {
            str(key): int(value) for key, value in site_types.items()
        },
        "status_counts": {
            str(key): int(value) for key, value in statuses.items()
        },
        "type_II_block_count": int(
            result.loc[
                result["covmeth_site_type"] == "type_II_local_block",
                "covmeth_region_id",
            ].nunique(dropna=True)
        ),
        "type_III_region_count": int(
            result.loc[
                result["covmeth_site_type"].isin(
                    [
                        "type_III_continuous",
                        "type_III_no_annotation_model",
                    ]
                ),
                "covmeth_region_id",
            ].nunique(dropna=True)
        ),
        "parameters": asdict(config),
    }


def run_covmeth(
    frame: pd.DataFrame,
    *,
    config: CovMethConfig = CovMethConfig(),
    chrom_col: str = "chrom",
    position_col: str = "position",
    methylation_col: str = "methylation",
    coverage_col: str = "coverage",
    annotation_col: str = "annotation_id",
    group_columns: Sequence[str] = ("sample", "haplotype"),
    require_annotation: bool = False,
    include_diagnostics: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the complete covMeth classification and inference workflow.

    Parameters
    ----------
    frame
        CpG-level table. Required columns are chromosome, genomic position,
        methylation level, coverage, and any requested grouping columns.
    config
        Unified parameters for all three covMeth modules.
    annotation_col
        Column identifying a concrete functional annotation interval. This is
        required for Type-III recovery, but the workflow can still run Types I
        and II when the column is absent unless ``require_annotation=True``.
    group_columns
        Columns defining independent methylation series. The default processes
        every sample and haplotype independently.
    require_annotation
        Raise an input error if ``annotation_col`` is absent.
    include_diagnostics
        Retain module-specific intermediate estimates and weights.

    Returns
    -------
    result, summary
        The completed CpG table and a JSON-serializable run summary.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    working = frame.reset_index(drop=True).copy()
    annotation_column_present = annotation_col in working.columns
    if annotation_column_present:
        annotation_text = working[annotation_col].astype("string").str.strip()
        annotation_available = bool(
            (annotation_text.notna() & annotation_text.ne("")).any()
        )
    else:
        annotation_available = False

    if not annotation_column_present:
        working[annotation_col] = pd.NA
        LOGGER.warning(
            "Annotation column '%s' was not found. Types I and II will run, "
            "but Type-III candidates will be retained without correction.",
            annotation_col,
        )
    elif not annotation_available:
        LOGGER.warning(
            "Annotation column '%s' contains no usable interval IDs. "
            "Type-III candidates will be retained without correction.",
            annotation_col,
        )

    if require_annotation and not annotation_available:
        raise CovMethInputError(
            f"No usable Type-III annotations were found in column: {annotation_col}"
        )

    common_kwargs = {
        "chrom_col": chrom_col,
        "position_col": position_col,
        "methylation_col": methylation_col,
        "coverage_col": coverage_col,
        "group_columns": tuple(group_columns),
    }

    isolated = smooth_isolated_sites(
        working,
        config=config.isolated_config(),
        **common_kwargs,
    )
    local = smooth_local_blocks(
        working,
        config=config.local_config(),
        **common_kwargs,
    )
    continuous = smooth_continuous_regions(
        working,
        config=config.continuous_config(),
        annotation_col=annotation_col,
        **common_kwargs,
    )
    _ensure_same_length(working, isolated, local, continuous)

    type_i = isolated["coverage_type"].eq("type_I_isolated")
    type_ii = local["coverage_type"].eq("type_II_local_block")
    type_iii_corrected = continuous["coverage_type"].eq(
        "type_III_continuous"
    )
    type_iii_unmodeled = continuous["coverage_type"].eq(
        "type_III_no_annotation_model"
    )
    type_iii = type_iii_corrected | type_iii_unmodeled

    _assert_disjoint_masks(
        type_I=type_i,
        type_II=type_ii,
        type_III=type_iii,
    )

    original_columns = list(frame.columns)
    result = working[original_columns].copy()

    # Start with the common baseline produced by the isolated module. All
    # submodules use the same validation and initialization functions.
    for column in STANDARD_RESULT_COLUMNS:
        result[column] = isolated[column].to_numpy()

    # Replace only the rows selected by each structural class.
    for source, mask in (
        (isolated, type_i),
        (local, type_ii),
        (continuous, type_iii),
    ):
        for column in STANDARD_RESULT_COLUMNS:
            result.loc[mask, column] = source.loc[mask, column].to_numpy()

    result["covmeth_methylation"] = result["inferred_methylation"]
    result["covmeth_site_type"] = result["coverage_type"]
    result["covmeth_status"] = result["inference_status"]
    result["covmeth_smoothed"] = result["covmeth_status"].eq("corrected")
    result["covmeth_method"] = "none"
    result.loc[
        result["covmeth_site_type"].eq("low_coverage_unclassified"),
        "covmeth_method",
    ] = "unclassified"
    result.loc[type_i, "covmeth_method"] = "isolated"
    result.loc[type_ii, "covmeth_method"] = "local_block"
    result.loc[type_iii, "covmeth_method"] = "continuous_region"

    result["covmeth_region_id"] = pd.NA
    if "block_id" in local.columns:
        result.loc[type_ii, "covmeth_region_id"] = local.loc[
            type_ii, "block_id"
        ].to_numpy()
    if "continuous_region_id" in continuous.columns:
        result.loc[type_iii, "covmeth_region_id"] = continuous.loc[
            type_iii, "continuous_region_id"
        ].to_numpy()

    observed = pd.to_numeric(
        result["observed_methylation"], errors="coerce"
    )
    inferred = pd.to_numeric(
        result["covmeth_methylation"], errors="coerce"
    )
    result["covmeth_adjustment"] = inferred - observed

    if include_diagnostics:
        _add_prefixed_diagnostics(
            result,
            isolated,
            type_i,
            columns=TYPE_I_COLUMNS,
            prefix="type_I_",
        )
        _add_prefixed_diagnostics(
            result,
            local,
            type_ii,
            columns=TYPE_II_COLUMNS,
            prefix="type_II_",
        )
        _add_prefixed_diagnostics(
            result,
            continuous,
            type_iii,
            columns=TYPE_III_COLUMNS,
            prefix="type_III_",
        )

    # Output integrity checks.
    corrected_values = pd.to_numeric(
        result.loc[result["covmeth_smoothed"], "covmeth_methylation"],
        errors="coerce",
    )
    if corrected_values.isna().any():
        raise RuntimeError("covMeth produced missing corrected values.")
    if ((corrected_values < 0) | (corrected_values > 1)).any():
        raise RuntimeError("covMeth produced values outside [0, 1].")

    summary = _make_summary(
        result,
        config=config,
        annotation_available=annotation_available,
    )
    return result, summary


def _write_json_atomic(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete covMeth workflow and automatically select "
            "isolated-site, local-block, or continuous-region inference."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = parser.add_argument_group("input and output")
    io_group.add_argument("-i", "--input", required=True, type=Path)
    io_group.add_argument("-o", "--output", required=True, type=Path)
    io_group.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional JSON file containing run statistics and parameters.",
    )
    io_group.add_argument(
        "--sep",
        default=None,
        help="Input delimiter. Inferred from the filename when omitted.",
    )
    io_group.add_argument("--output-sep", default="\t")
    io_group.add_argument(
        "--compact",
        action="store_true",
        help="Omit module-specific diagnostic columns.",
    )

    column_group = parser.add_argument_group("column names")
    column_group.add_argument("--chrom-col", default="chrom")
    column_group.add_argument("--position-col", default="position")
    column_group.add_argument("--methylation-col", default="methylation")
    column_group.add_argument("--coverage-col", default="coverage")
    column_group.add_argument("--annotation-col", default="annotation_id")
    column_group.add_argument(
        "--group-columns",
        default="sample,haplotype",
        help=(
            "Comma-separated columns processed independently. Use an empty "
            "string for a single ungrouped series."
        ),
    )
    column_group.add_argument(
        "--require-annotation",
        action="store_true",
        help="Stop if the Type-III annotation column is absent.",
    )

    common_group = parser.add_argument_group("common covMeth parameters")
    common_group.add_argument("--coverage-threshold", type=int, default=5)
    common_group.add_argument("--max-anchor-distance", type=int, default=500)

    isolated_group = parser.add_argument_group("Type-I isolated-site parameters")
    isolated_group.add_argument("--isolated-delta", type=float, default=0.10)
    isolated_group.add_argument("--isolated-tau", type=float, default=0.10)
    isolated_group.add_argument(
        "--isolated-min-shrinkage", type=float, default=0.0
    )

    local_group = parser.add_argument_group("Type-II local-block parameters")
    local_group.add_argument("--local-min-block-sites", type=int, default=2)
    local_group.add_argument(
        "--local-propagation-length", type=float, default=200.0
    )
    local_group.add_argument(
        "--local-observation-scale", type=float, default=5.0
    )
    local_group.add_argument(
        "--local-min-observation-gain", type=float, default=0.02
    )
    local_group.add_argument(
        "--local-max-observation-gain", type=float, default=0.95
    )

    continuous_group = parser.add_argument_group(
        "Type-III continuous-region parameters"
    )
    continuous_group.add_argument(
        "--continuous-min-region-sites", type=int, default=2
    )
    continuous_group.add_argument(
        "--continuous-min-trend-points", type=int, default=3
    )
    continuous_group.add_argument(
        "--continuous-shrinkage-scale", type=float, default=5.0
    )
    continuous_group.add_argument(
        "--continuous-max-abs-slope-per-kb", type=float, default=1.0
    )

    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    config = CovMethConfig(
        coverage_threshold=args.coverage_threshold,
        max_anchor_distance=args.max_anchor_distance,
        isolated_delta=args.isolated_delta,
        isolated_tau=args.isolated_tau,
        isolated_min_shrinkage=args.isolated_min_shrinkage,
        local_min_block_sites=args.local_min_block_sites,
        local_propagation_length=args.local_propagation_length,
        local_observation_scale=args.local_observation_scale,
        local_min_observation_gain=args.local_min_observation_gain,
        local_max_observation_gain=args.local_max_observation_gain,
        continuous_min_region_sites=args.continuous_min_region_sites,
        continuous_min_trend_points=args.continuous_min_trend_points,
        continuous_shrinkage_scale=args.continuous_shrinkage_scale,
        continuous_max_abs_slope_per_kb=(
            args.continuous_max_abs_slope_per_kb
        ),
    )

    frame = read_table(args.input, args.sep)
    result, summary = run_covmeth(
        frame,
        config=config,
        chrom_col=args.chrom_col,
        position_col=args.position_col,
        methylation_col=args.methylation_col,
        coverage_col=args.coverage_col,
        annotation_col=args.annotation_col,
        group_columns=parse_group_columns(args.group_columns),
        require_annotation=args.require_annotation,
        include_diagnostics=not args.compact,
    )
    atomic_write(result, args.output, args.output_sep)
    if args.summary is not None:
        _write_json_atomic(summary, args.summary)

    type_counts = summary["site_type_counts"]
    LOGGER.info("covMeth completed: %d total CpG sites.", summary["total_sites"])
    LOGGER.info(
        "Corrected %d site(s): Type I=%d, Type II=%d, Type III=%d.",
        summary["corrected_sites"],
        type_counts.get("type_I_isolated", 0),
        type_counts.get("type_II_local_block", 0),
        type_counts.get("type_III_continuous", 0),
    )
    LOGGER.info(
        "Retained without correction: unclassified=%d, "
        "Type-III without annotation model=%d.",
        type_counts.get("low_coverage_unclassified", 0),
        type_counts.get("type_III_no_annotation_model", 0),
    )
    LOGGER.info("Output written to %s", args.output)
    if args.summary is not None:
        LOGGER.info("Summary written to %s", args.summary)


if __name__ == "__main__":
    main()
