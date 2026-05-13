"""Plot helpers for episode-series statistic observers."""

from __future__ import annotations

from matplotlib.axes import Axes
import numpy as np
from numpy.typing import ArrayLike

from network_creation.core.observability import StatisticResult
from network_creation.observers.agent_payoff_observer import AgentPayoffObserver
from network_creation.observers.statistics.episode_series import (
    EpisodeSeriesStatisticObserver,
)
from network_creation.observers.statistics.episode_tensor_series import (
    EpisodeSeriesTensorStatisticObserver,
)

_PRIMARY_COLOR = "#1f4e79"
_BAND_COLOR = "#f6bd60"
_SPREAD_LEGEND_TITLE = "solid line: mean\nshaded band: ± SD"


def plot_episode_series_observer(
    observer: EpisodeSeriesStatisticObserver | EpisodeSeriesTensorStatisticObserver,
    ax: Axes,
    dimension_name: str | None = None,
    show_runs: bool = False,
    color: str | None = None,
    band_color: str | None = None,
    label: str | None = None,
    set_title: bool = True,
    x_values: ArrayLike | None = None,
    x_label: str | None = None,
) -> Axes:
    """Plot one episode-series observer on one explicit matplotlib axis."""

    if observer.n_episodes == 0:
        raise ValueError("observer has not recorded any episode yet")
    if (
        not isinstance(observer, EpisodeSeriesTensorStatisticObserver)
        and dimension_name is not None
    ):
        raise ValueError("dimension_name is only supported for tensor-valued observers")

    plot_steps = _plot_steps(observer.steps, x_values=x_values)
    legend_title = None
    if observer.aggregations:
        if _is_agent_payoff_virtual_dimension(observer, dimension_name):
            assert isinstance(observer, AgentPayoffObserver)
            assert dimension_name is not None
            result = observer.degree_ranked_result(dimension_name)
        else:
            result = observer.get()
        if isinstance(observer, EpisodeSeriesTensorStatisticObserver) and (
            not _is_agent_payoff_virtual_dimension(observer, dimension_name)
        ):
            result = _select_tensor_result_dimension(result, dimension_name)
        legend_title = _plot_aggregated_outputs(
            observer=observer,
            result=result,
            ax=ax,
            show_runs=show_runs,
            color=color,
            band_color=band_color,
            label=label,
            set_title=set_title,
            x_values=plot_steps,
        )
    else:
        if isinstance(observer, EpisodeSeriesTensorStatisticObserver):
            selected_values = _select_tensor_observer_values(observer, dimension_name)
        else:
            selected_values = observer.values
        ax.plot(
            plot_steps,
            selected_values,
            linewidth=2,
            color=_PRIMARY_COLOR if color is None else color,
            label=label,
        )
        if set_title:
            ax.set_title(f"{observer.label} over one episode")

    ax.set_xlabel(observer.x_label if x_label is None else x_label)
    ax.set_ylabel(observer.y_label)
    if observer.y_min is not None and observer.y_max is not None:
        ax.set_ylim(observer.y_min, observer.y_max)
    ax.set_xlim(plot_steps[0], plot_steps[-1])
    ax.grid(alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(title=legend_title)
    return ax


def _is_agent_payoff_virtual_dimension(
    observer: object,
    dimension_name: str | None,
) -> bool:
    return isinstance(
        observer,
        AgentPayoffObserver,
    ) and observer.is_degree_ranked_dimension_name(dimension_name)


def _select_tensor_observer_values(
    observer: EpisodeSeriesTensorStatisticObserver,
    dimension_name: str | None,
) -> np.ndarray:
    if _is_agent_payoff_virtual_dimension(observer, dimension_name):
        assert isinstance(observer, AgentPayoffObserver)
        assert dimension_name is not None
        return observer.degree_ranked_values(dimension_name)

    if dimension_name is None:
        raise ValueError("dimension_name must be provided for tensor-valued observers")
    if dimension_name not in observer.dimension_names:
        raise ValueError(f"unknown dimension name: {dimension_name}")
    dimension_idx = observer.dimension_names.index(dimension_name)
    return observer.values.reshape(observer.values.shape[0], -1)[:, dimension_idx]


def _plot_aggregated_outputs(
    observer: EpisodeSeriesStatisticObserver | EpisodeSeriesTensorStatisticObserver,
    result: StatisticResult,
    ax: Axes,
    show_runs: bool,
    color: str | None,
    band_color: str | None,
    label: str | None,
    set_title: bool,
    x_values: np.ndarray,
) -> str | None:
    if result.steps is None:
        raise ValueError("episode-series statistic result must include steps")
    plot_steps = _plot_steps(result.steps, x_values=x_values)

    primary_output_name = None
    for output_name in ("mean", "std"):
        if output_name in observer.aggregations:
            primary_output_name = output_name
            break

    primary_color = _PRIMARY_COLOR if color is None else color
    fill_color = band_color
    if fill_color is None:
        fill_color = _BAND_COLOR if color is None else primary_color
    legend_title = None

    runs = result.outputs.get("runs")
    if runs is not None and show_runs:
        if not isinstance(runs, tuple):
            raise ValueError("output 'runs' must be a tuple of series")
        for run_values in runs:
            ax.plot(
                plot_steps,
                run_values,
                color=primary_color,
                alpha=0.02,
                linewidth=1.5,
            )

    if primary_output_name is None:
        if not set_title:
            return None
        if result.n_episodes == 1:
            ax.set_title(f"{observer.label} over one episode")
            return None
        ax.set_title(f"{observer.label} across multiple episodes")
        return None

    primary_values = result.outputs[primary_output_name]
    if isinstance(primary_values, tuple):
        raise ValueError(f"output '{primary_output_name}' must be one series")
    selected_primary_values = primary_values

    if primary_output_name == "mean" and "std" in result.outputs:
        std_values = result.outputs["std"]
        if isinstance(std_values, tuple):
            raise ValueError("output 'std' must be one series")
        lower_band = selected_primary_values - std_values
        upper_band = selected_primary_values + std_values
        if observer.y_min is not None:
            lower_band = np.maximum(lower_band, observer.y_min)
        if observer.y_max is not None:
            upper_band = np.minimum(upper_band, observer.y_max)

        ax.fill_between(
            plot_steps,
            lower_band,
            upper_band,
            color=fill_color,
            alpha=0.35,
            label=None if label is not None else "mean ± SD",
        )
        legend_title = _SPREAD_LEGEND_TITLE

    ax.plot(
        plot_steps,
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
    if not set_title:
        return legend_title
    if result.n_episodes == 1:
        ax.set_title(f"{observer.label} over one episode")
        return legend_title
    ax.set_title(f"{observer.label} across multiple episodes")
    return legend_title


def _select_tensor_result_dimension(
    result: StatisticResult,
    dimension_name: str | None,
) -> StatisticResult:
    if dimension_name is None:
        raise ValueError("dimension_name must be provided for tensor-valued observers")
    if result.dimension_names is None or dimension_name not in result.dimension_names:
        raise ValueError(f"unknown dimension name: {dimension_name}")
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


def _plot_steps(
    raw_steps: ArrayLike,
    *,
    x_values: ArrayLike | None,
) -> np.ndarray:
    steps = np.asarray(raw_steps, dtype=float)
    if steps.ndim != 1:
        raise ValueError("observer steps must be one-dimensional")
    if x_values is None:
        return steps

    values = np.asarray(x_values, dtype=float)
    if values.shape != steps.shape:
        raise ValueError(f"x_values must have shape {steps.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("x_values must contain only finite values")
    if values.shape[0] > 1 and not np.all(np.diff(values) > 0.0):
        raise ValueError("x_values must be strictly increasing")
    return values
