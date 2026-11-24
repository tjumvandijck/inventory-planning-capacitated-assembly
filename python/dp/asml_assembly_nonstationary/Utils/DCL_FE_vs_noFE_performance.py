"""Compare DCL policy performance as the DCL training-set size increases.

Enter mean trajectory costs in ``MEAN_COSTS`` below. Costs are converted to a
percentage of BS-TW-S, so BS-TW-S is always the horizontal 100% benchmark.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np


PathLike = Union[str, Path]


def plot_dcl_training_sample_performance(
    sample_counts: Sequence[int],
    bs_tw_s_mean_cost: float,
    dcl_a_mean_costs: Sequence[float],
    dcl_a_fe_mean_costs: Sequence[float],
    *,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[PathLike] = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot DCL mean costs relative to BS-TW-S = 100%.

    Parameters
    ----------
    sample_counts:
        DCL training-set sizes, for example ``[50_000, 75_000, 150_000]``.
    bs_tw_s_mean_cost:
        Mean trajectory cost of BS-TW-S. It is used as the common baseline,
        yielding the dashed horizontal 100% line.
    dcl_a_mean_costs, dcl_a_fe_mean_costs:
        Mean trajectory costs in the same order as ``sample_counts``.
    """
    samples = np.asarray(sample_counts, dtype=float)
    dcl_a_costs = np.asarray(dcl_a_mean_costs, dtype=float)
    dcl_a_fe_costs = np.asarray(dcl_a_fe_mean_costs, dtype=float)

    if samples.ndim != 1 or len(samples) == 0:
        raise ValueError("Provide at least one DCL training-set size.")
    if len(dcl_a_costs) != len(samples) or len(dcl_a_fe_costs) != len(samples):
        raise ValueError("Supply one DCL-A and one DCL-A-FE cost for every sample size.")
    if np.any(samples <= 0):
        raise ValueError("Training-set sizes must be positive.")
    if bs_tw_s_mean_cost <= 0 or not np.isfinite(bs_tw_s_mean_cost):
        raise ValueError("The BS-TW-S mean cost must be finite and positive.")
    if not np.all(np.isfinite(dcl_a_costs)) or not np.all(np.isfinite(dcl_a_fe_costs)):
        raise ValueError("All DCL mean costs must be finite numbers.")

    sort_order = np.argsort(samples)
    samples = samples[sort_order]
    dcl_a_relative = 100.0 * dcl_a_costs[sort_order] / bs_tw_s_mean_cost
    dcl_a_fe_relative = 100.0 * dcl_a_fe_costs[sort_order] / bs_tw_s_mean_cost
    baseline_relative = np.full(len(samples), 100.0)

    if ax is None:
        figure, ax = plt.subplots(figsize=(9, 5.5))
    else:
        figure = ax.figure

    ax.plot(
        samples,
        baseline_relative,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="BS-TW-S",
        zorder=1,
    )
    ax.plot(
        samples,
        dcl_a_relative,
        color="#c43c39",
        marker="^",
        markersize=8,
        linewidth=2.2,
        label="DCL-A",
        zorder=2,
    )
    ax.plot(
        samples,
        dcl_a_fe_relative,
        color="#2878b5",
        marker="o",
        markersize=8,
        linewidth=2.2,
        label="DCL-A-FE",
        zorder=3,
    )

    all_relative_costs = np.concatenate(
        (baseline_relative, dcl_a_relative, dcl_a_fe_relative)
    )
    y_padding = max(1.0, 0.08 * np.ptp(all_relative_costs))
    ax.set_ylim(
        all_relative_costs.min() - y_padding,
        all_relative_costs.max() + y_padding,
    )
    ax.set_xlabel("Number of DCL training samples", fontsize=13)
    ax.set_ylabel("Mean cost relative to BS-TW-S (%)", fontsize=13)
    ax.set_xticks(samples)
    ax.set_xticklabels([f"{int(sample):,}" for sample in samples])
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=12, frameon=False)
    figure.tight_layout()

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight")
    if show:
        plt.show()

    return figure, ax


if __name__ == "__main__":
    SAMPLE_COUNTS = [37_500, 50_000, 75_000, 150_000]

    # Replace None with the mean trajectory costs from the three evaluations.
    # BS-TW-S is one common benchmark because it does not depend on DCL
    # training-set size.
    MEAN_COSTS = {
        "BS-TW-S": 2208.3170100000034,
        "DCL-A": [2153.583309999988, None, None],
        "DCL-A-FE": [2145.975164999997, None, None],
    }

    if MEAN_COSTS["BS-TW-S"] is None or any(
        value is None
        for values in (MEAN_COSTS["DCL-A"], MEAN_COSTS["DCL-A-FE"])
        for value in values
    ):
        raise ValueError("Fill in all values in MEAN_COSTS before running this script.")

    plot_dcl_training_sample_performance(
        SAMPLE_COUNTS,
        MEAN_COSTS["BS-TW-S"],
        MEAN_COSTS["DCL-A"],
        MEAN_COSTS["DCL-A-FE"],
        output_path=(
            Path(__file__).resolve().parents[1]
            / "Figures"
            / "dcl_fe_vs_nofe_training_sample_performance.pdf"
        ),
    )
