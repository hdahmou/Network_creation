"""Plot helpers for episode-series statistic observers."""

from __future__ import annotations

from matplotlib.axes import Axes
import numpy as np
from network_creation.core.observability import StatisticResult
from network_creation.observers.statistics.episode_series import (
    EpisodeSeriesStatisticObserver,
)

_PRIMARY_COLOR = "#1f4e79"
_BAND_COLOR = "#f6bd60"


def plot_episode_series_observer(
    observer: EpisodeSeriesStatisticObserver,
    ax: Axes,
) -> Axes:
    """Plot one episode-series observer on one explicit matplotlib axis."""

    if observer.n_episodes == 0:
        raise ValueError("observer has not recorded any episode yet")

    if observer.aggregations:
        result = observer.get()
        _plot_aggregated_outputs(observer=observer, result=result, ax=ax)
    else:
        ax.plot(
            observer.steps,
            observer.values,
            linewidth=2,
            color=_PRIMARY_COLOR,
        )
        ax.set_title(f"{observer.label} over one episode")

    ax.set_xlabel(observer.x_label)
    ax.set_ylabel(observer.y_label)
    if observer.y_min is not None and observer.y_max is not None:
        ax.set_ylim(observer.y_min, observer.y_max)
    ax.set_xlim(observer.steps[0], observer.steps[-1])
    ax.grid(alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend()
    return ax


def _plot_aggregated_outputs(
    observer: EpisodeSeriesStatisticObserver,
    result: StatisticResult,
    ax: Axes,
) -> None:
    if result.steps is None:
        raise ValueError("episode-series statistic result must include steps")

    primary_output_name = None
    for output_name in ("mean", "std"):
        if output_name in observer.aggregations:
            primary_output_name = output_name
            break
    if primary_output_name is None:
        if result.n_episodes == 1:
            ax.set_title(f"{observer.label} over one episode")
            return
        ax.set_title(f"{observer.label} across multiple episodes")
        return

    primary_values = result.outputs[primary_output_name]
    if isinstance(primary_values, tuple):
        raise ValueError(f"output '{primary_output_name}' must be one series")

    if primary_output_name == "mean" and "std" in observer.aggregations:
        std_values = result.outputs["std"]
        if isinstance(std_values, tuple):
            raise ValueError("output 'std' must be one series")

        lower_band = primary_values - std_values
        upper_band = primary_values + std_values
        if observer.y_min is not None:
            lower_band = np.maximum(lower_band, observer.y_min)
        if observer.y_max is not None:
            upper_band = np.minimum(upper_band, observer.y_max)

        ax.fill_between(
            result.steps,
            lower_band,
            upper_band,
            color=_BAND_COLOR,
            alpha=0.35,
            label="mean ± std",
        )

    ax.plot(
        result.steps,
        primary_values,
        color=_PRIMARY_COLOR,
        linewidth=2.5,
        label={
            "mean": f"mean {observer.y_label}",
            "std": f"std {observer.y_label}",
        }.get(primary_output_name, primary_output_name),
    )
    ax.set_xlim(result.steps[0], result.steps[-1])
    if result.n_episodes == 1:
        ax.set_title(f"{observer.label} over one episode")
        return
    ax.set_title(f"{observer.label} across multiple episodes")
