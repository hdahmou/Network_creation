"""Plots for posterior parameter-recovery analysis."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .posterior import PosteriorArtifact, category_probabilities


def plot_parameter_marginals(
    artifact: PosteriorArtifact,
    *,
    prior_samples: Any,
    posterior_samples: Any,
    true_theta: Any,
):
    import matplotlib.pyplot as plt

    prior = np.asarray(prior_samples, dtype=np.float32)
    posterior = np.asarray(posterior_samples, dtype=np.float32)
    truth = np.asarray(true_theta, dtype=np.float32)
    n_cols = min(3, len(artifact.theta_order))
    n_rows = math.ceil(len(artifact.theta_order) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3 * n_cols, 2.6 * n_rows),
        squeeze=False,
    )

    for i, name in enumerate(artifact.theta_order):
        ax = axes.ravel()[i]
        if artifact.is_discrete_theta(name):
            _category_bars(
                ax,
                artifact,
                name,
                prior[:, i],
                posterior[:, i],
                truth[i],
            )
            ax.set_ylabel("probability")
        else:
            bins = np.histogram_bin_edges(
                np.r_[prior[:, i], posterior[:, i]],
                bins=40,
            )
            ax.hist(
                prior[:, i],
                bins=bins,
                density=True,
                alpha=0.35,
                label="prior",
            )
            ax.hist(
                posterior[:, i],
                bins=bins,
                density=True,
                alpha=0.75,
                label="posterior",
            )
            ax.axvline(truth[i], linewidth=2, label="true")
            ax.set_ylabel("density")
        ax.set_title(name, fontsize=9)
        ax.grid(alpha=0.15)

    for ax in axes.ravel()[len(artifact.theta_order) :]:
        ax.set_axis_off()
    axes.ravel()[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_posterior_ridgelines(
    artifact: PosteriorArtifact,
    *,
    prior_samples: Any,
    posterior_samples: Any,
    true_theta: Any,
    n_cols: int = 3,
    bins: int = 80,
    title: str = "Posterior marginals across held-out cases",
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, NullLocator

    prior = np.asarray(prior_samples, dtype=np.float32)
    posterior = np.asarray(posterior_samples, dtype=np.float32)
    truth = np.asarray(true_theta, dtype=np.float32)
    if posterior.ndim != 3:
        raise ValueError("posterior_samples must have shape (case, sample, theta)")
    if truth.shape != (posterior.shape[0], posterior.shape[2]):
        raise ValueError("true_theta must have shape (case, theta)")

    n_cases, _, n_theta = posterior.shape
    n_cols = min(max(1, n_cols), n_theta)
    n_rows = math.ceil(n_theta / n_cols)
    panel_height = max(3.0, 0.24 * n_cases + 1.8)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.6 * n_cols, panel_height * n_rows),
        squeeze=False,
    )
    posterior_color = "#4477AA"
    prior_color = "#B8B8B8"
    truth_color = "#CC6677"
    ridge_height = 0.72
    prior_y = n_cases + 0.65

    for i, name in enumerate(artifact.theta_order):
        ax = axes.ravel()[i]
        x_min, x_max = _parameter_limits(artifact, prior, posterior, truth, i, name)
        order = np.argsort(_ridgeline_axis_values(artifact, truth[:, i], name))

        _draw_ridge(
            ax,
            artifact,
            name,
            prior[:, i],
            baseline=prior_y,
            x_min=x_min,
            x_max=x_max,
            bins=bins,
            height=0.45,
            color=prior_color,
            alpha=0.8,
        )
        for row, case_index in enumerate(order):
            _draw_ridge(
                ax,
                artifact,
                name,
                posterior[case_index, :, i],
                baseline=float(row),
                x_min=x_min,
                x_max=x_max,
                bins=bins,
                height=ridge_height,
                color=posterior_color,
                alpha=0.75,
            )
            true_value = _ridgeline_axis_values(
                artifact,
                [truth[case_index, i]],
                name,
            )[0]
            ax.vlines(
                true_value,
                row,
                row + ridge_height,
                color=truth_color,
                linewidth=1.6,
            )

        ax.set_title(_parameter_display_name(name))
        ax.set_xlabel("Parameter value")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.35, n_cases + 1.3)
        ax.set_yticks([])
        if artifact.is_discrete_theta(name):
            categories = artifact.category_values(name)
            ax.xaxis.set_major_locator(
                FixedLocator(categories.astype(np.float32).tolist())
            )
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticklabels([str(value) for value in categories])
            ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        ax.grid(axis="x", which="major", alpha=0.18)

    for ax in axes.ravel()[n_theta:]:
        ax.set_axis_off()

    fig.suptitle(title, y=0.995)
    fig.legend(
        handles=[
            Patch(
                facecolor=posterior_color,
                alpha=0.75,
                label="Posterior density/mass",
            ),
            Patch(facecolor=prior_color, alpha=0.8, label="Prior density/mass"),
            Line2D([0], [0], color=truth_color, linewidth=1.6, label="True value"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        frameon=False,
        ncol=3,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return fig


def plot_parameter_pairs(
    artifact: PosteriorArtifact,
    *,
    prior_samples: Any,
    posterior_samples: Any,
    true_theta: Any,
    max_points: int = 2_000,
    seed: int = 0,
):
    import matplotlib.pyplot as plt

    prior = _thin(np.asarray(prior_samples, dtype=np.float32), max_points, seed)
    posterior = _thin(
        np.asarray(posterior_samples, dtype=np.float32),
        max_points,
        seed + 1,
    )
    truth = np.asarray(true_theta, dtype=np.float32)
    n_pairs = len(artifact.theta_order) - 1
    if n_pairs < 1:
        raise ValueError("pair plot requires at least two theta dimensions")

    fig, axes = plt.subplots(
        n_pairs,
        n_pairs,
        figsize=(2.6 * n_pairs, 2.6 * n_pairs),
        squeeze=False,
    )
    rng = np.random.default_rng(seed)
    handles = labels = None
    for row in range(n_pairs):
        y_index = row + 1
        y_name = artifact.theta_order[y_index]
        for col in range(n_pairs):
            ax = axes[row, col]
            if col > row:
                ax.set_axis_off()
                continue

            x_name = artifact.theta_order[col]
            prior_x, prior_y = _plot_values(artifact, prior, col, y_index, rng)
            post_x, post_y = _plot_values(artifact, posterior, col, y_index, rng)
            true_x = _axis_values(artifact, [truth[col]], x_name)[0]
            true_y = _axis_values(artifact, [truth[y_index]], y_name)[0]
            ax.scatter(
                prior_x,
                prior_y,
                s=4,
                alpha=0.08,
                linewidths=0,
                label="prior",
            )
            ax.scatter(
                post_x,
                post_y,
                s=6,
                alpha=0.25,
                linewidths=0,
                label="posterior",
            )
            ax.axvline(true_x, linewidth=1, label="true")
            ax.axhline(true_y, linewidth=1)
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()
            _format_pair_axis(ax, artifact, x_name, y_name, row, col, n_pairs)

    legend_axis = axes[0, -1]
    if handles and labels:
        if n_pairs > 1:
            legend_axis.set_axis_off()
            legend_axis.legend(handles, labels, loc="center", frameon=False, fontsize=8)
        else:
            legend_axis.legend(handles, labels, loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def _thin(
    values: NDArray[np.float32],
    max_points: int,
    seed: int,
) -> NDArray[np.float32]:
    if len(values) <= max_points:
        return values
    rng = np.random.default_rng(seed)
    return values[np.sort(rng.choice(len(values), size=max_points, replace=False))]


def _parameter_display_name(name: str) -> str:
    return name.rsplit(".", 1)[-1].replace("_", " ").capitalize()


def _category_bars(ax, artifact, name: str, prior, posterior, true_value) -> None:
    categories = artifact.category_values(name)
    positions = np.arange(len(categories), dtype=np.float32)
    ax.bar(
        positions - 0.19,
        category_probabilities(prior, categories),
        width=0.38,
        alpha=0.35,
        label="prior",
    )
    ax.bar(
        positions + 0.19,
        category_probabilities(posterior, categories),
        width=0.38,
        alpha=0.75,
        label="posterior",
    )
    ax.axvline(
        _category_positions(artifact, [true_value], name)[0],
        linewidth=1.5,
        label="true",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([str(value) for value in categories])
    ax.set_xlim(float(positions[0]) - 0.5, float(positions[-1]) + 0.5)
    ax.set_ylim(bottom=0)


def _draw_ridge(
    ax,
    artifact,
    name: str,
    values,
    *,
    baseline: float,
    x_min: float,
    x_max: float,
    bins: int,
    height: float,
    color: str,
    alpha: float,
) -> None:
    if artifact.is_discrete_theta(name):
        categories = artifact.category_values(name)
        values = _ridgeline_axis_values(artifact, values, name)
        probs = category_probabilities(values, categories)
        if probs.size and float(probs.max()) > 0:
            probs = probs / float(probs.max())
        width = _category_width(categories)
        ax.bar(
            categories.astype(np.float32),
            height * probs,
            bottom=baseline,
            width=width,
            color=color,
            alpha=alpha,
            linewidth=0,
        )
        ax.hlines(baseline, x_min, x_max, color=color, alpha=0.2, linewidth=0.7)
        return

    x, density = _density_curve(values, x_min=x_min, x_max=x_max, bins=bins)
    y = baseline + height * density
    ax.fill_between(x, baseline, y, color=color, alpha=alpha, linewidth=0)
    ax.plot(x, y, color=color, alpha=min(1.0, alpha + 0.15), linewidth=0.8)


def _density_curve(
    values,
    *,
    x_min: float,
    x_max: float,
    bins: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    hist, edges = np.histogram(
        np.asarray(values, dtype=np.float32),
        bins=max(8, int(bins)),
        range=(x_min, x_max),
        density=True,
    )
    density = np.nan_to_num(hist.astype(np.float64), copy=False)
    if density.size >= 5:
        kernel = np.asarray([1, 2, 3, 2, 1], dtype=np.float64)
        density = np.convolve(density, kernel / kernel.sum(), mode="same")
    peak = float(density.max()) if density.size else 0.0
    if peak > 0:
        density = density / peak
    return (edges[:-1] + edges[1:]) / 2.0, density


def _ridgeline_axis_values(artifact, values: Any, name: str) -> NDArray[np.float32]:
    if not artifact.is_discrete_theta(name):
        return np.asarray(values, dtype=np.float32)

    categories = artifact.category_values(name)
    rounded = np.rint(np.asarray(values)).astype(np.int64)
    if not np.isin(rounded, categories).all():
        raise ValueError(f"samples for {name!r} contain unknown categories")
    return rounded.astype(np.float32)


def _parameter_limits(
    artifact,
    prior: NDArray[np.float32],
    posterior: NDArray[np.float32],
    truth: NDArray[np.float32],
    index: int,
    name: str,
) -> tuple[float, float]:
    if artifact.is_discrete_theta(name):
        categories = artifact.category_values(name).astype(np.float32)
        pad = max(0.5, _category_width(categories))
        return float(categories.min() - pad), float(categories.max() + pad)

    low = artifact.metadata.dataset.prior_low[index]
    high = artifact.metadata.dataset.prior_high[index]
    if np.isfinite(low) and np.isfinite(high) and low < high:
        return low, high

    values = np.r_[
        prior[:, index],
        posterior[:, :, index].reshape(-1),
        truth[:, index],
    ]
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if not np.isfinite(low) or not np.isfinite(high):
        return 0.0, 1.0
    if low == high:
        return low - 0.5, high + 0.5
    margin = 0.04 * (high - low)
    return low - margin, high + margin


def _category_width(categories: NDArray[np.int64]) -> float:
    if len(categories) < 2:
        return 0.28
    return max(0.18, 0.22 * float(np.min(np.diff(np.sort(categories)))))


def _plot_values(
    artifact,
    samples,
    x_index: int,
    y_index: int,
    rng,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    x_name = artifact.theta_order[x_index]
    y_name = artifact.theta_order[y_index]
    x = _axis_values(artifact, samples[:, x_index], x_name)
    y = _axis_values(artifact, samples[:, y_index], y_name)
    if artifact.is_discrete_theta(x_name):
        x = x + rng.uniform(-0.08, 0.08, len(x))
    if artifact.is_discrete_theta(y_name):
        y = y + rng.uniform(-0.08, 0.08, len(y))
    return x, y


def _axis_values(artifact, values: Any, name: str) -> NDArray[np.float32]:
    if artifact.is_discrete_theta(name):
        return _category_positions(artifact, values, name)
    return np.asarray(values, dtype=np.float32)


def _category_positions(artifact, values: Any, name: str) -> NDArray[np.float32]:
    categories = artifact.category_values(name)
    rounded = np.rint(np.asarray(values)).astype(np.int64)
    return np.asarray(
        [int(np.flatnonzero(categories == value)[0]) for value in rounded],
        dtype=np.float32,
    )


def _format_pair_axis(
    ax,
    artifact,
    x_name: str,
    y_name: str,
    row: int,
    col: int,
    n_pairs: int,
) -> None:
    if artifact.is_discrete_theta(x_name):
        _category_ticks(ax, artifact, x_name, axis="x", show=row == n_pairs - 1)
    elif row < n_pairs - 1:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    if artifact.is_discrete_theta(y_name):
        _category_ticks(ax, artifact, y_name, axis="y", show=col == 0)
    elif col:
        ax.tick_params(axis="y", left=False, labelleft=False)
    if row == n_pairs - 1:
        ax.set_xlabel(x_name, fontsize=8)
    if col == 0:
        ax.set_ylabel(y_name, fontsize=8)
    ax.grid(alpha=0.15)
    ax.set_box_aspect(1)


def _category_ticks(ax, artifact, name: str, *, axis: str, show: bool) -> None:
    categories = artifact.category_values(name)
    positions = np.arange(len(categories), dtype=np.float32)
    labels = [str(value) for value in categories] if show else []
    if axis == "x":
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlim(float(positions[0]) - 0.5, float(positions[-1]) + 0.5)
    else:
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.set_ylim(float(positions[0]) - 0.5, float(positions[-1]) + 0.5)
