#!/usr/bin/env python3
from pathlib import Path

import pandas as pd

from covmeth_isolated import Config as IsolatedConfig
from covmeth_isolated import smooth_isolated_sites
from covmeth_local_block import Config as LocalConfig
from covmeth_local_block import smooth_local_blocks
from covmeth_continuous_region import Config as ContinuousConfig
from covmeth_continuous_region import smooth_continuous_regions

HERE = Path(__file__).resolve().parent


def main() -> None:
    frame = pd.read_csv(HERE / "example_input.tsv", sep="\t")

    isolated = smooth_isolated_sites(frame, config=IsolatedConfig())
    row = isolated.loc[isolated["position"] == 120].iloc[0]
    assert row["coverage_type"] == "type_I_isolated"
    assert 0 <= row["inferred_methylation"] <= 1

    local = smooth_local_blocks(frame, config=LocalConfig())
    rows = local[local["position"].isin([1020, 1040])]
    assert (rows["coverage_type"] == "type_II_local_block").all()
    assert rows["inferred_methylation"].between(0, 1).all()

    continuous = smooth_continuous_regions(frame, config=ContinuousConfig())
    rows = continuous[continuous["position"].isin([3600, 3620, 3640])]
    assert (rows["coverage_type"] == "type_III_continuous").all()
    assert rows["inferred_methylation"].between(0, 1).all()

    print("All covMeth module tests passed.")


if __name__ == "__main__":
    main()
