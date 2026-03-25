"""Plot helpers for episode-series statistic observers."""

from __future__ import annotations

from matplotlib.axes import Axes
import numpy as np
from network_creation.core.observability import StatisticResult
from network_creation.observers.statistics.episode_series import (
    EpisodeSeriesStatisticObserver,
)
from network_creation.observers.statistics.episode_tensor_series import (
    EpisodeSeriesTensorStatisticObserver,
)

_PRIMARY_COLOR = "#1f4e79"
_BAND_COLOR = "#f6bd60"


def plot_episode_series_observer(
    observer: EpisodeSeriesStatisticObserver | EpisodeSeriesTensorStatisticObserver,
    ax: Axes,
    *,
    dimension_name: str | None = None,
    show_runs: bool = False,
    color: str | None = None,
    band_color: str | None = None,
    label: str | None = None,
    set_title: bool = True,
) -> Axes:
    """Plot one episode-series observer on one explicit matplotlib axis."""

    if observer.n_episodes == 0:
        raise ValueError("observer has not recorded any episode yet")
    if (
        not isinstance(observer, EpisodeSeriesTensorStatisticObserver)
        and dimension_name is not None
    ):
        raise ValueError("dimension_name is only supported for tensor-valued observers")

    if observer.aggregations:
        result = observer.get()
        if isinstance(observer, EpisodeSeriesTensorStatisticObserver):
            result = _select_tensor_result_dimension(result, dimension_name)
        _plot_aggregated_outputs(
            observer=observer,
            result=result,
            ax=ax,
            show_runs=show_runs,
            color=color,
            band_color=band_color,
            label=label,
            set_title=set_title,
        )
    else:
        if isinstance(observer, EpisodeSeriesTensorStatisticObserver):
            if dimension_name is None:
                raise ValueError(
                    "dimension_name must be provided for tensor-valued observers"
                )
            if dimension_name not in observer.dimension_names:
                raise ValueError(f"unknown dimension name: {dimension_name}")
            dimension_idx = observer.dimension_names.index(dimension_name)
            selected_values = observer.values.reshape(observer.values.shape[0], -1)[
                :, dimension_idx
            ]
        else:
            selected_values = observer.values
        ax.plot(
            observer.steps,
            selected_values,
            linewidth=2,
            color=_PRIMARY_COLOR if color is None else color,
            label=label,
        )
        if set_title:
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
    observer: EpisodeSeriesStatisticObserver | EpisodeSeriesTensorStatisticObserver,
    result: StatisticResult,
    ax: Axes,
    *,
    show_runs: bool,
    color: str | None,
    band_color: str | None,
    label: str | None,
    set_title: bool,
) -> None:
    if result.steps is None:
        raise ValueError("episode-series statistic result must include steps")

    primary_output_name = None
    for output_name in ("mean", "std"):
        if output_name in observer.aggregations:
            primary_output_name = output_name
            break

    primary_color = _PRIMARY_COLOR if color is None else color
    fill_color = band_color
    if fill_color is None:
        fill_color = _BAND_COLOR if color is None else primary_color

    runs = result.outputs.get("runs")
    if runs is not None and show_runs:
        if not isinstance(runs, tuple):
            raise ValueError("output 'runs' must be a tuple of series")
        for run_values in runs:
            ax.plot(
                result.steps,
                run_values,
                color=primary_color,
                alpha=0.02,
                linewidth=1.5,
            )

    if primary_output_name is None:
        if not set_title:
            return
        if result.n_episodes == 1:
            ax.set_title(f"{observer.label} over one episode")
            return
        ax.set_title(f"{observer.label} across multiple episodes")
        return

    primary_values = result.outputs[primary_output_name]
    if isinstance(primary_values, tuple):
        raise ValueError(f"output '{primary_output_name}' must be one series")
    selected_primary_values = primary_values

    if primary_output_name == "mean" and "std" in observer.aggregations:
        std_values = result.outputs["std"]
        if isinstance(std_values, tuple):
            raise ValueError("output 'std' must be one series")
        selected_std_values = std_values

        lower_band = selected_primary_values - selected_std_values
        upper_band = selected_primary_values + selected_std_values
        if observer.y_min is not None:
            lower_band = np.maximum(lower_band, observer.y_min)
        if observer.y_max is not None:
            upper_band = np.minimum(upper_band, observer.y_max)

        ax.fill_between(
            result.steps,
            lower_band,
            upper_band,
            color=fill_color,
            alpha=0.35,
            label=None if label is not None else "mean ± std",
        )

    ax.plot(
        result.steps,
        selected_primary_values,
        color=primary_color,
        linewidth=2.5,
        label=(
            label
            if label is not None
            else {
                "mean": f"mean {observer.y_label}",
                "std": f"std {observer.y_label}",
            }.get(primary_output_name, primary_output_name)
        ),
    )
    ax.set_xlim(result.steps[0], result.steps[-1])
    if not set_title:
        return
    if result.n_episodes == 1:
        ax.set_title(f"{observer.label} over one episode")
        return
    ax.set_title(f"{observer.label} across multiple episodes")


def _select_tensor_result_dimension(
    result: StatisticResult,
    dimension_name: str | None,
) -> StatisticResult:
    dimension_idx = result.dimension_names.index(dimension_name)

    def select(values: np.ndarray) -> np.ndarray:
        return values.reshape(values.shape[0], -1)[:, dimension_idx]

    return StatisticResult(
        steps=result.steps,
        n_episodes=result.n_episodes,
        outputs={
            output_name: (
                tuple(select(run_values) for run_values in output_values)
                if isinstance(output_values, tuple)
                else select(output_values)
            )
            for output_name, output_values in result.outputs.items()
        },
        sample_shape=(),
        dimension_names=None,
    )
