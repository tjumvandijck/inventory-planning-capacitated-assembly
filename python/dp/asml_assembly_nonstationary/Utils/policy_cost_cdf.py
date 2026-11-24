"""Construct empirical CDF plots from saved policy trajectory costs.

The C++ evaluator writes ``policy_costs_<run>.csv`` with one row per policy
and one column per trajectory.  This module plots a selected subset of those
rows as empirical cumulative distribution functions (ECDFs).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np


PathLike = Union[str, Path]
Labels = Optional[Union[Mapping[int, str], Sequence[str]]]
Colors = Optional[Union[Mapping[int, str], Sequence[str]]]


def load_policy_costs(csv_path: PathLike) -> list[np.ndarray]:
    """Load a ``policy_costs_*.csv`` file into one cost vector per policy.

    Empty cells are ignored so this also accepts legacy CSV files that end
    each row with a trailing comma.
    """
    path = Path(csv_path)
    rows: list[np.ndarray] = []

    with path.open(newline="") as csv_file:
        for row_index, row in enumerate(csv.reader(csv_file)):
            cells = [cell.strip() for cell in row if cell.strip()]
            if not cells:
                continue
            try:
                costs = np.asarray([float(cell) for cell in cells], dtype=float)
            except ValueError as error:
                raise ValueError(
                    f"Non-numeric value in row {row_index} of {path}."
                ) from error
            if not np.all(np.isfinite(costs)):
                raise ValueError(
                    f"Non-finite cost in row {row_index} of {path}."
                )
            rows.append(costs)

    if not rows:
        raise ValueError(f"No policy costs were found in {path}.")
    return rows


def _value_for_policy(
    values: Union[Mapping[int, str], Sequence[str]],
    policy_index: int,
    position: int,
    value_name: str,
) -> str:
    if isinstance(values, Mapping):
        try:
            return values[policy_index]
        except KeyError as error:
            raise ValueError(
                f"No {value_name} was supplied for policy index {policy_index}."
            ) from error

    if position >= len(values):
        raise ValueError(
            f"Supply one {value_name} for each selected policy, in the same order."
        )
    return values[position]


def _gaussian_kde(samples: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    """Evaluate a Gaussian KDE using Scott's rule for its bandwidth.

    Keeping this small implementation local avoids adding SciPy as a runtime
    dependency for a plotting utility that otherwise needs only NumPy and
    Matplotlib.
    """
    standard_deviation = np.std(samples, ddof=1)
    bandwidth = standard_deviation * len(samples) ** (-1.0 / 5.0)
    if bandwidth <= 0.0 or not np.isfinite(bandwidth):
        raise ValueError("KDE bandwidth must be finite and positive.")

    scaled_distance = (x_values[:, np.newaxis] - samples[np.newaxis, :]) / bandwidth
    normal_density = np.exp(-0.5 * scaled_distance**2) / np.sqrt(2.0 * np.pi)
    return normal_density.mean(axis=1) / bandwidth


def plot_policy_cost_cdf(
    csv_path: PathLike,
    policy_indices: Optional[Sequence[int]] = None,
    policy_labels: Labels = None,
    colors: Colors = None,
    *,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[PathLike] = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot transformed ECDFs of selected policies' trajectory costs.

    The horizontal axis is the natural logarithm of the trajectory cost. The
    vertical axis is the negative logarithm of the empirical survival
    probability, ``-log(1 - F)``. This makes differences in the right tail
    easier to inspect.

    Parameters
    ----------
    csv_path:
        Path to a C++-generated ``policy_costs_<run>.csv`` file.
    policy_indices:
        The zero-based policy rows to plot. ``None`` plots every policy.
        For example, ``[0, 3, 5]`` plots only those three policies.
    policy_labels:
        Either a mapping such as ``{0: "BS-N", 5: "BS-TW-S"}`` or a list
        in the same order as ``policy_indices``.
    colors:
        Optional mapping or ordered list, analogous to ``policy_labels``.
    ax:
        Existing axes to draw on. If omitted, a new figure is created.
    output_path:
        Optional image destination, e.g. ``"Figures/cost_cdf.pdf"``.
    show:
        Whether to call ``plt.show()``. Set to ``False`` in batch scripts.

    Returns
    -------
    (figure, axes)
        The created (or supplied) Matplotlib figure and axes.
    """
    policy_costs = load_policy_costs(csv_path)
    if policy_indices is None:
        selected_indices = list(range(len(policy_costs)))
    else:
        selected_indices = list(policy_indices)

    if not selected_indices:
        raise ValueError("Select at least one policy to plot.")
    if len(set(selected_indices)) != len(selected_indices):
        raise ValueError("Each selected policy index must be unique.")
    for policy_index in selected_indices:
        if policy_index < 0 or policy_index >= len(policy_costs):
            raise IndexError(
                f"Policy index {policy_index} is outside [0, {len(policy_costs) - 1}]."
            )

    if ax is None:
        figure, ax = plt.subplots(figsize=(8, 5))
    else:
        figure = ax.figure

    for position, policy_index in enumerate(selected_indices):
        costs = np.sort(policy_costs[policy_index])
        num_trajectories = len(costs)
        if num_trajectories == 0:
            raise ValueError(f"Policy {policy_index} has no trajectory costs.")
        if np.any(costs <= 0.0):
            raise ValueError(
                f"Policy {policy_index} has non-positive trajectory costs; "
                "log(cost) is undefined."
            )

        # Use i / (n + 1), rather than i / n, so the final observation does
        # not make -log(1 - F) infinite. This is the usual finite plotting
        # position for an empirical survival plot.
        cumulative_probability = np.arange(1, num_trajectories + 1) / (
            num_trajectories + 1
        )
        # transformed_survival = -np.log1p(-cumulative_probability)
        label = (
            _value_for_policy(policy_labels, policy_index, position, "label")
            if policy_labels is not None
            else f"Policy {policy_index}"
        )
        color = (
            _value_for_policy(colors, policy_index, position, "color")
            if colors is not None
            else None
        )
        ax.step(
            np.log(costs),
            cumulative_probability,
            where="post",
            label=label,
            color=color,
        )

    ax.set_xlabel("log(Cost per trajectory)", fontsize=14)
    #ax.set_ylabel(r"$-\log(1 - \mathrm{Empirical\ cumulative\ probability})$", fontsize=12)
    ax.set_ylabel(r"Empirical cumulative probability", fontsize=14)

    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14)
    figure.tight_layout()

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight")
    if show:
        plt.show()

    return figure, ax


def plot_policy_cost_kde(
    csv_path: PathLike,
    policy_indices: Optional[Sequence[int]] = None,
    policy_labels: Labels = None,
    colors: Colors = None,
    *,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[PathLike] = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot Gaussian-kernel density estimates of selected policy costs.

    The density is shown on the original cost scale. Each curve integrates to
    one, so its height represents relative concentration rather than an
    absolute trajectory count.
    """
    policy_costs = load_policy_costs(csv_path)
    if policy_indices is None:
        selected_indices = list(range(len(policy_costs)))
    else:
        selected_indices = list(policy_indices)

    if not selected_indices:
        raise ValueError("Select at least one policy to plot.")
    if len(set(selected_indices)) != len(selected_indices):
        raise ValueError("Each selected policy index must be unique.")
    for policy_index in selected_indices:
        if policy_index < 0 or policy_index >= len(policy_costs):
            raise IndexError(
                f"Policy index {policy_index} is outside [0, {len(policy_costs) - 1}."
            )

    if ax is None:
        figure, ax = plt.subplots(figsize=(8, 5))
    else:
        figure = ax.figure

    selected_costs = [policy_costs[policy_index] for policy_index in selected_indices]
    x_min = min(costs.min() for costs in selected_costs)
    x_max = max(costs.max() for costs in selected_costs)
    if x_min == x_max:
        raise ValueError("Cannot estimate a KDE when all selected costs are identical.")
    padding = 0.02 * (x_max - x_min)

    for position, policy_index in enumerate(selected_indices):
        costs = policy_costs[policy_index]
        if len(costs) < 2 or np.all(costs == costs[0]):
            raise ValueError(
                f"Policy {policy_index} needs at least two distinct trajectory costs for a KDE."
            )
        label = (
            _value_for_policy(policy_labels, policy_index, position, "label")
            if policy_labels is not None
            else f"Policy {policy_index}"
        )
        color = (
            _value_for_policy(colors, policy_index, position, "color")
            if colors is not None
            else None
        )
        # A Gaussian KDE is positive everywhere, even beyond the observed
        # costs. Draw it only over this policy's empirical range so the curve
        # does not suggest that a different policy shares its maximum cost.
        policy_x_values = np.linspace(costs.min(), costs.max(), 1_000)
        ax.plot(
            policy_x_values,
            _gaussian_kde(costs, policy_x_values),
            label=label,
            color=color,
        )

    ax.set_xlabel("Cost per trajectory", fontsize=14)
    ax.set_ylabel("Probability density", fontsize=14)
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(bottom=-0.00002)
    ax.set_yticks(np.arange(0.0, 0.00031, 0.00005))
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14)
    figure.tight_layout()

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight")
    if show:
        plt.show()

    return figure, ax


def plot_paired_absolute_cost_difference_boxplot(
    csv_path: PathLike,
    baseline_policy_index: int = 5,
    comparison_policy_index: int = 6,
    baseline_policy_label: str = "BS-TW-S",
    comparison_policy_label: str = "DCL-A-FE",
    *,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[PathLike] = None,
    show: bool = True,
    remove_outliers: bool = True,
    remove_outliers_fraction: float = 0.05,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot paired absolute cost differences between two policy rows.

    The C++ evaluation uses the same event stream for every policy within a
    trajectory. Therefore, element ``t`` of both policy rows forms one paired
    observation. The plotted values are
    ``abs(comparison_cost[t] - baseline_cost[t])``.
    """
    policy_costs = load_policy_costs(csv_path)
    for policy_index in (baseline_policy_index, comparison_policy_index):
        if policy_index < 0 or policy_index >= len(policy_costs):
            raise IndexError(
                f"Policy index {policy_index} is outside [0, {len(policy_costs) - 1}]."
            )

    baseline_costs = policy_costs[baseline_policy_index]
    comparison_costs = policy_costs[comparison_policy_index]
    if len(baseline_costs) != len(comparison_costs):
        raise ValueError(
            "Paired differences require both policies to have the same number "
            "of trajectory costs."
        )
    if len(baseline_costs) == 0:
        raise ValueError("Paired differences require at least one trajectory.")

    differences = comparison_costs - baseline_costs
    mean_difference = float(np.mean(differences))

    if (remove_outliers): 
        num_trim = int(np.floor(remove_outliers_fraction * len(differences)))

        sorted_indices = np.argsort(differences)
        if num_trim > 0:
            kept_indices = sorted_indices[num_trim:-num_trim]
        else:
            kept_indices = sorted_indices

        differences = differences[kept_indices]
        mean_difference = float(np.mean(differences))

    
    if ax is None:
        figure, ax = plt.subplots(figsize=(6.5, 5))
    else:
        figure = ax.figure

    ax.boxplot(
        differences,
        patch_artist=True,
        widths=0.45,
        boxprops={"facecolor": "#8ecae6", "edgecolor": "#2878b5"},
        medianprops={"color": "#1d3557", "linewidth": 2},
        whiskerprops={"color": "#2878b5"},
        capprops={"color": "#2878b5"},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#2878b5",
            "markeredgecolor": "none",
            "markersize": 3,
            "alpha": 0.35,
        },
    )
    ax.axhline(
        mean_difference,
        color="black",
        linestyle=":",
        linewidth=2,
        label=f"Mean difference = {mean_difference:,.2f}",
    )

    ax.set_xticks([1])
    ax.set_xticklabels([f"{comparison_policy_label} - {baseline_policy_label}"], fontsize=12)
    ax.set_ylabel("Paired cost difference", fontsize=13)
    ax.set_ylim(bottom=differences.min()*1.1)
    ax.tick_params(axis="y", which="major", labelsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=11, frameon=False)
    figure.tight_layout()

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight")
    if show:
        plt.show()

    return figure, ax


def plot_paired_cost_difference_vs_periods(
    cost_csv_path: PathLike,
    periods_csv_path: PathLike,
    baseline_policy_index: int = 5,
    comparison_policy_index: int = 6,
    periods_policy_index: Optional[int] = None,
    baseline_policy_label: str = "BS-TW-S",
    comparison_policy_label: str = "DCL-A-FE",
    *,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[PathLike] = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot signed paired cost differences against post-warm-up periods.

    The y-value for trajectory ``t`` is
    ``comparison_cost[t] - baseline_cost[t]``. Negative values therefore mean
    that the comparison policy has lower cost. The trend line is an ordinary
    least-squares linear fit across the paired trajectories.

    ``periods_policy_index`` selects the policy row from the periods CSV. It
    defaults to the comparison policy, so that each observation uses DCL-A-FE's
    recorded number of post-warm-up periods.
    """
    cost_rows = load_policy_costs(cost_csv_path)
    period_rows = load_policy_costs(periods_csv_path)
    if periods_policy_index is None:
        periods_policy_index = comparison_policy_index

    for policy_index in (baseline_policy_index, comparison_policy_index):
        if policy_index < 0 or policy_index >= len(cost_rows):
            raise IndexError(
                f"Cost policy index {policy_index} is outside "
                f"[0, {len(cost_rows) - 1}]."
            )
    if periods_policy_index < 0 or periods_policy_index >= len(period_rows):
        raise IndexError(
            f"Periods policy index {periods_policy_index} is outside "
            f"[0, {len(period_rows) - 1}]."
        )

    baseline_costs = cost_rows[baseline_policy_index]
    comparison_costs = cost_rows[comparison_policy_index]
    periods_after_warmup = period_rows[periods_policy_index]
    num_trajectories = len(baseline_costs)
    if len(comparison_costs) != num_trajectories or len(periods_after_warmup) != num_trajectories:
        raise ValueError(
            "The baseline costs, comparison costs, and period counts must have "
            "the same number of trajectories."
        )
    if num_trajectories < 2:
        raise ValueError("At least two paired trajectories are required for a trend line.")
    if np.any(periods_after_warmup < 0):
        raise ValueError("Post-warm-up period counts cannot be negative.")
    if np.all(periods_after_warmup == periods_after_warmup[0]):
        raise ValueError("A trend line requires at least two distinct period counts.")

    paired_cost_differences = comparison_costs - baseline_costs
    slope, intercept = np.polyfit(periods_after_warmup, paired_cost_differences, 1)
    trend_periods = np.linspace(
        periods_after_warmup.min(), periods_after_warmup.max(), 200
    )
    trend_differences = slope * trend_periods + intercept
    fitted_differences = slope * periods_after_warmup + intercept
    total_sum_of_squares = np.sum(
        (paired_cost_differences - paired_cost_differences.mean()) ** 2
    )
    r_squared = (
        1.0
        - np.sum((paired_cost_differences - fitted_differences) ** 2)
        / total_sum_of_squares
        if total_sum_of_squares > 0.0
        else 1.0
    )

    if ax is None:
        figure, ax = plt.subplots(figsize=(8, 5.5))
    else:
        figure = ax.figure

    ax.scatter(
        periods_after_warmup,
        paired_cost_differences,
        color="#2878b5",
        alpha=0.45,
        edgecolors="none",
        s=24,
        label="Paired trajectories",
    )
    ax.plot(
        trend_periods,
        trend_differences,
        color="#c43c39",
        linewidth=2.2,
        label=f"Linear trend ($R^2$ = {r_squared:.3f})",
    )
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)

    ax.set_xlabel("Number of post-warm-up periods", fontsize=13)
    ax.set_ylabel(
        f"Paired cost difference ({comparison_policy_label} - {baseline_policy_label})",
        fontsize=13,
    )
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11, frameon=False)
    figure.tight_layout()

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight")
    if show:
        plt.show()

    return figure, ax


def plot_paired_cost_difference_vs_time_to_highest_demand(
    cost_csv_path: PathLike,
    periods_per_vertpos_csv_path: PathLike,
    baseline_policy_index: int = 5,
    comparison_policy_index: int = 6,
    periods_policy_index: Optional[int] = None,
    num_pre_highest_demand_levels: int = 3,
    baseline_policy_label: str = "BS-TW-S",
    comparison_policy_label: str = "DCL-A-FE",
    *,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[PathLike] = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot paired cost differences against time taken to reach high demand.

    For each trajectory, the x-value is the sum of the period counts at the
    first ``num_pre_highest_demand_levels`` vertical positions. Trajectories
    that never reach the highest vertical position are excluded. The C++ CSV
    must retain its native row order: policy, then trajectory, with vertical
    positions as columns.
    """
    cost_rows = load_policy_costs(cost_csv_path)
    periods_rows = load_policy_costs(periods_per_vertpos_csv_path)
    period_rows_total = load_policy_costs(r"C:\\Users\\u618593\\Desktop\\DynaplexPrivate\\IO_DynaPlex\\20261105 run output snellius with period and backlog units\\num_periods_after_warmup_per_policy_per_traj_FE_asml_assembly_nonstationary_NN10_p100_A_NS_N150000_M600_G2_FE_lowN.csv")

    if periods_policy_index is None:
        periods_policy_index = comparison_policy_index

    for policy_index in (baseline_policy_index, comparison_policy_index):
        if policy_index < 0 or policy_index >= len(cost_rows):
            raise IndexError(
                f"Cost policy index {policy_index} is outside "
                f"[0, {len(cost_rows) - 1}]."
            )
    if periods_policy_index < 0 or periods_policy_index >= len(cost_rows):
        raise IndexError(
            f"Periods policy index {periods_policy_index} is outside "
            f"[0, {len(cost_rows) - 1}]."
        )

    baseline_costs = cost_rows[baseline_policy_index]
    comparison_costs = cost_rows[comparison_policy_index]
    num_trajectories = len(baseline_costs)
    if len(comparison_costs) != num_trajectories:
        raise ValueError("The baseline and comparison policies must have paired costs.")
    if num_trajectories < 2:
        raise ValueError("At least two paired trajectories are required for a trend line.")

    first_period_row = periods_policy_index * num_trajectories
    last_period_row = first_period_row + num_trajectories
    if last_period_row > len(periods_rows):
        raise ValueError(
            "The vertical-position periods CSV does not contain one row per "
            "trajectory for the requested policy."
        )
    periods_per_vertical_position = np.asarray(
        periods_rows[first_period_row:last_period_row], dtype=float
    )
    if periods_per_vertical_position.ndim != 2:
        raise ValueError("Could not read the vertical-position period counts.")
    num_vertical_positions = periods_per_vertical_position.shape[1]
    if num_pre_highest_demand_levels <= 0 or (
        num_pre_highest_demand_levels >= num_vertical_positions
    ):
        raise ValueError(
            "The number of pre-highest demand levels must be between 1 and "
            "the number of vertical positions minus one."
        )
    if np.any(periods_per_vertical_position < 0):
        raise ValueError("Vertical-position period counts cannot be negative.")

    # The final column represents the highest demand level. Rows with zero in
    # that column never reached it and would distort the time-to-reach plot.
    reached_highest_demand = periods_per_vertical_position[:, -1] > 0
    if reached_highest_demand.sum() < 2:
        raise ValueError(
            "At least two trajectories must reach the highest demand level "
            "to fit a trend line."
        )

    periods_to_highest_demand = periods_per_vertical_position[
        :, :num_pre_highest_demand_levels
    ].sum(axis=1)
    paired_cost_differences = comparison_costs - baseline_costs
    periods_to_highest_demand = periods_to_highest_demand[reached_highest_demand]
    paired_cost_differences = paired_cost_differences[reached_highest_demand]
    if np.all(periods_to_highest_demand == periods_to_highest_demand[0]):
        raise ValueError("A trend line requires at least two distinct times to reach high demand.")

    slope, intercept = np.polyfit(
        periods_to_highest_demand, paired_cost_differences, 1
    )
    trend_periods = np.linspace(
        periods_to_highest_demand.min(), periods_to_highest_demand.max(), 200
    )
    trend_differences = slope * trend_periods + intercept
    fitted_differences = slope * periods_to_highest_demand + intercept
    total_sum_of_squares = np.sum(
        (paired_cost_differences - paired_cost_differences.mean()) ** 2
    )
    r_squared = (
        1.0
        - np.sum((paired_cost_differences - fitted_differences) ** 2)
        / total_sum_of_squares
        if total_sum_of_squares > 0.0
        else 1.0
    )

    if ax is None:
        figure, ax = plt.subplots(figsize=(8, 5.5))
    else:
        figure = ax.figure

    ax.scatter(
        periods_to_highest_demand,
        paired_cost_differences,
        color="#2878b5",
        alpha=0.45,
        edgecolors="none",
        s=24,
        label="Trajectories reaching highest demand",
    )
    ax.plot(
        trend_periods,
        trend_differences,
        color="#c43c39",
        linewidth=2.2,
        label=f"Linear trend ($R^2$ = {r_squared:.3f})",
    )
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)

    ax.set_xlabel("Periods before reaching the highest demand level", fontsize=13)
    ax.set_ylabel(
        f"Paired cost difference ({comparison_policy_label} - {baseline_policy_label})",
        fontsize=13,
    )
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11, frameon=False)
    figure.tight_layout()

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight")
    if show:
        plt.show()

    return figure, ax


def plot_paired_cost_difference_3d_vs_highest_demand_timing(
    cost_csv_path: PathLike,
    periods_per_vertpos_csv_path: PathLike,
    baseline_policy_index: int = 5,
    comparison_policy_index: int = 6,
    periods_policy_index: Optional[int] = None,
    num_pre_highest_demand_levels: int = 3,
    baseline_policy_label: str = "BS-TW-S",
    comparison_policy_label: str = "DCL-A-FE",
    *,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[PathLike] = None,
    show_trend_plane: bool = True,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Visualize paired cost differences against high-demand timing in 3D.

    The x-axis is the sum of the first three vertical-position period counts;
    the y-axis is the number of periods at the highest demand level; and the
    z-axis is ``DCL-A-FE cost - BS-TW-S cost``. Trajectories that never reach
    the highest demand level are excluded.
    """
    cost_rows = load_policy_costs(cost_csv_path)
    periods_rows = load_policy_costs(periods_per_vertpos_csv_path)
    if periods_policy_index is None:
        periods_policy_index = comparison_policy_index

    for policy_index in (baseline_policy_index, comparison_policy_index):
        if policy_index < 0 or policy_index >= len(cost_rows):
            raise IndexError(
                f"Cost policy index {policy_index} is outside "
                f"[0, {len(cost_rows) - 1}]."
            )
    if periods_policy_index < 0 or periods_policy_index >= len(cost_rows):
        raise IndexError(
            f"Periods policy index {periods_policy_index} is outside "
            f"[0, {len(cost_rows) - 1}]."
        )

    baseline_costs = cost_rows[baseline_policy_index]
    comparison_costs = cost_rows[comparison_policy_index]
    num_trajectories = len(baseline_costs)
    if len(comparison_costs) != num_trajectories:
        raise ValueError("The baseline and comparison policies must have paired costs.")

    first_period_row = periods_policy_index * num_trajectories
    last_period_row = first_period_row + num_trajectories
    if last_period_row > len(periods_rows):
        raise ValueError(
            "The vertical-position periods CSV does not contain one row per "
            "trajectory for the requested policy."
        )
    periods_per_vertical_position = np.asarray(
        periods_rows[first_period_row:last_period_row], dtype=float
    )
    if periods_per_vertical_position.ndim != 2:
        raise ValueError("Could not read the vertical-position period counts.")
    num_vertical_positions = periods_per_vertical_position.shape[1]
    if num_pre_highest_demand_levels <= 0 or (
        num_pre_highest_demand_levels >= num_vertical_positions
    ):
        raise ValueError(
            "The number of pre-highest demand levels must be between 1 and "
            "the number of vertical positions minus one."
        )

    reached_highest_demand = periods_per_vertical_position[:, -1] > 0
    if reached_highest_demand.sum() < 3:
        raise ValueError(
            "At least three trajectories must reach the highest demand level "
            "for the 3D plot."
        )

    periods_to_highest_demand = periods_per_vertical_position[
        :, :num_pre_highest_demand_levels
    ].sum(axis=1)[reached_highest_demand]
    periods_at_highest_demand = periods_per_vertical_position[
        :, -1
    ][reached_highest_demand]
    paired_cost_differences = (comparison_costs - baseline_costs)[
        reached_highest_demand
    ]

    if ax is None:
        figure = plt.figure(figsize=(9, 7))
        ax = figure.add_subplot(projection="3d")
    else:
        figure = ax.figure

    points = ax.scatter(
        periods_to_highest_demand,
        periods_at_highest_demand,
        paired_cost_differences,
        c=paired_cost_differences,
        cmap="coolwarm",
        s=18,
        alpha=0.5,
        depthshade=True,
    )
    colorbar = figure.colorbar(points, ax=ax, shrink=0.65, pad=0.1)
    colorbar.set_label("Paired cost difference", fontsize=11)

    if show_trend_plane:
        design_matrix = np.column_stack(
            (
                np.ones(len(paired_cost_differences)),
                periods_to_highest_demand,
                periods_at_highest_demand,
            )
        )
        coefficients, _, _, _ = np.linalg.lstsq(
            design_matrix, paired_cost_differences, rcond=None
        )
        plane_x, plane_y = np.meshgrid(
            np.linspace(
                periods_to_highest_demand.min(),
                periods_to_highest_demand.max(),
                25,
            ),
            np.linspace(
                periods_at_highest_demand.min(),
                periods_at_highest_demand.max(),
                25,
            ),
        )
        plane_z = (
            coefficients[0] + coefficients[1] * plane_x + coefficients[2] * plane_y
        )
        ax.plot_surface(
            plane_x,
            plane_y,
            plane_z,
            color="#555555",
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xlabel("Periods before highest demand", labelpad=10)
    ax.set_ylabel("Periods at highest demand", labelpad=10)
    ax.set_zlabel(
        f"Cost difference\n({comparison_policy_label} - {baseline_policy_label})",
        labelpad=10,
    )
    ax.view_init(elev=24, azim=-58)
    figure.tight_layout()

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight")
    if show:
        plt.show()

    return figure, ax


def plot_paired_cost_difference_vs_periods_at_highest_demand(
    cost_csv_path: PathLike,
    periods_per_vertpos_csv_path: PathLike,
    baseline_policy_index: int = 5,
    comparison_policy_index: int = 6,
    periods_policy_index: Optional[int] = None,
    baseline_policy_label: str = "BS-TW-S",
    comparison_policy_label: str = "DCL-A-FE",
    *,
    ax: Optional[plt.Axes] = None,
    output_path: Optional[PathLike] = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot paired cost differences against periods at highest demand.

    The final vertical-position column is the highest demand level. Rows with
    zero periods in that column are excluded before plotting and fitting the
    ordinary least-squares linear trend.
    """
    cost_rows = load_policy_costs(cost_csv_path)
    periods_rows = load_policy_costs(periods_per_vertpos_csv_path)
    if periods_policy_index is None:
        periods_policy_index = comparison_policy_index

    for policy_index in (baseline_policy_index, comparison_policy_index):
        if policy_index < 0 or policy_index >= len(cost_rows):
            raise IndexError(
                f"Cost policy index {policy_index} is outside "
                f"[0, {len(cost_rows) - 1}]."
            )
    if periods_policy_index < 0 or periods_policy_index >= len(cost_rows):
        raise IndexError(
            f"Periods policy index {periods_policy_index} is outside "
            f"[0, {len(cost_rows) - 1}]."
        )

    baseline_costs = cost_rows[baseline_policy_index]
    comparison_costs = cost_rows[comparison_policy_index]
    num_trajectories = len(baseline_costs)
    if len(comparison_costs) != num_trajectories:
        raise ValueError("The baseline and comparison policies must have paired costs.")

    first_period_row = periods_policy_index * num_trajectories
    last_period_row = first_period_row + num_trajectories
    if last_period_row > len(periods_rows):
        raise ValueError(
            "The vertical-position periods CSV does not contain one row per "
            "trajectory for the requested policy."
        )
    periods_per_vertical_position = np.asarray(
        periods_rows[first_period_row:last_period_row], dtype=float
    )
    if periods_per_vertical_position.ndim != 2:
        raise ValueError("Could not read the vertical-position period counts.")
    if np.any(periods_per_vertical_position < 0):
        raise ValueError("Vertical-position period counts cannot be negative.")

    periods_at_highest_demand = periods_per_vertical_position[:, -1]
    reached_highest_demand = periods_at_highest_demand > 0
    if reached_highest_demand.sum() < 2:
        raise ValueError(
            "At least two trajectories must reach the highest demand level "
            "to fit a trend line."
        )

    periods_at_highest_demand = periods_at_highest_demand[reached_highest_demand]
    paired_cost_differences = (comparison_costs - baseline_costs)[
        reached_highest_demand
    ]
    if np.all(periods_at_highest_demand == periods_at_highest_demand[0]):
        raise ValueError(
            "A trend line requires at least two distinct highest-demand durations."
        )

    slope, intercept = np.polyfit(
        periods_at_highest_demand, paired_cost_differences, 1
    )
    trend_periods = np.linspace(
        periods_at_highest_demand.min(), periods_at_highest_demand.max(), 200
    )
    trend_differences = slope * trend_periods + intercept
    fitted_differences = slope * periods_at_highest_demand + intercept
    total_sum_of_squares = np.sum(
        (paired_cost_differences - paired_cost_differences.mean()) ** 2
    )
    r_squared = (
        1.0
        - np.sum((paired_cost_differences - fitted_differences) ** 2)
        / total_sum_of_squares
        if total_sum_of_squares > 0.0
        else 1.0
    )

    if ax is None:
        figure, ax = plt.subplots(figsize=(8, 5.5))
    else:
        figure = ax.figure

    ax.scatter(
        periods_at_highest_demand,
        paired_cost_differences,
        color="#2878b5",
        alpha=0.45,
        edgecolors="none",
        s=24,
        label="Trajectories reaching highest demand",
    )
    ax.plot(
        trend_periods,
        trend_differences,
        color="#c43c39",
        linewidth=2.2,
        label=f"Linear trend ($R^2$ = {r_squared:.3f})",
    )
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)

    ax.set_xlabel("Periods at the highest demand level", fontsize=13)
    ax.set_ylabel(
        f"Paired cost difference ({comparison_policy_label} - {baseline_policy_label})",
        fontsize=13,
    )
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11, frameon=False)
    figure.tight_layout()

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight")
    if show:
        plt.show()

    return figure, ax


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot empirical CDFs of selected policy trajectory costs."
    )
    parser.add_argument("csv_path", type=Path, help="Path to policy_costs_<run>.csv")
    parser.add_argument(
        "--policies",
        type=int,
        nargs="+",
        help="Zero-based policy rows to plot; omit to plot all rows.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Labels in the same order as --policies.",
    )
    parser.add_argument("--output", type=Path, help="Optional PDF or PNG output path.")
    parser.add_argument("--no-show", action="store_true", help="Do not open a plot window.")
    return parser.parse_args()


if __name__ == "__main__":


    plot_paired_cost_difference_vs_periods_at_highest_demand(
        cost_csv_path=r"C:\\Users\\u618593\\Desktop\\DynaplexPrivate\\IO_DynaPlex\\20261105 run output snellius with period and backlog units\\policy_costs_asml_assembly_nonstationary_NN10_p100_A_NS_N150000_M600_G2_FE_lowN.csv",
        periods_per_vertpos_csv_path=r"C:\\Users\\u618593\\Desktop\\DynaplexPrivate\\IO_DynaPlex\\20261105 run output snellius with period and backlog units\\num_periods_per_vertpos_per_policy_per_traj_FE_asml_assembly_nonstationary_NN10_p100_A_NS_N150000_M600_G2_FE_lowN.csv",
        baseline_policy_index=5,
        comparison_policy_index=6,
        output_path=r"C:\\Users\\u618593\\Desktop\\DynaplexPrivate\\IO_DynaPlex\\20261105 run output snellius with period and backlog units\\paired_cost_difference_vs_periods_at_highest_demand.pdf",
    )
    # plot_paired_absolute_cost_difference_boxplot(
    #     csv_path=r"C:\\Users\\u618593\\Desktop\\DynaplexPrivate\\IO_DynaPlex\\20261103 run output revision snellius 2\\policy_costs_asml_assembly_nonstationary_NN10_p100_A_NS_N150000_M600_G2_FE_lowN.csv",
    #     baseline_policy_index=5,
    #     comparison_policy_index=6,
    #     baseline_policy_label="BS-TW-S",
    #     comparison_policy_label="DCL-A-FE",
    #     output_path=r"C:\\Users\\u618593\\source\\repos\\DynaPlexPrivate\\python\\dp\\asml_assembly_nonstationary\\Figures\\paired_cost_difference_asml_assembly_nonstationary_NN10_p100_A_NS_N150000_M600_G2_FE_lowN.pdf",
    #     show=True,
    #     remove_outliers=True,
    #     remove_outliers_fraction=0.00
    # )
